package ch08

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"sort"
	"strings"

	"gorm.io/gorm"

	"rag/infrastructure"
	"rag/internal/ragcore"
)

// louvain 是简化版 Louvain(单层,无 super-graph 折叠):对每个节点贪心搬社区,
// 直到一轮无任何 ΔQ > 0 的搬移即停。课程目的不是 Leiden 精度,只要给出"密集子图"概念。
// 时间 O(iter * |E|);我们 100 节点规模下毫秒级。
func louvain(entIDs []int64, adj map[int64][]relEdge) map[int64]int64 {
	idx := make(map[int64]int, len(entIDs))
	for i, id := range entIDs {
		idx[id] = i
	}
	// k[i] 加权度(出 + 入)
	k := make([]float64, len(entIDs))
	// m 总边权(单向累加,后面 ΔQ 用 2m)
	m := 0.0
	for _, edges := range adj {
		for _, e := range edges {
			m += float64(e.weight)
		}
	}
	for i, id := range entIDs {
		for _, e := range adj[id] {
			k[i] += float64(e.weight)
		}
	}
	if m == 0 {
		// 无边:每个节点自成一社区
		out := make(map[int64]int64, len(entIDs))
		for _, id := range entIDs {
			out[id] = id
		}
		return out
	}
	twoM := 2 * m

	// 初始社区:每节点自成一社区
	comm := make([]int, len(entIDs))
	for i := range entIDs {
		comm[i] = i
	}
	// Σ_in[c]: 社区 c 内部边权和(每条边算两次,符合 Q 公式)
	sigmaIn := make(map[int]float64, len(entIDs))
	// Σ_tot[c]: 社区 c 所有节点的 k 之和
	sigmaTot := make(map[int]float64, len(entIDs))
	for i := range entIDs {
		c := comm[i]
		sigmaIn[c] += k[i] // 单节点社区内部边权 0,但 Σ_in 起点是 k_i,后面减回去
		sigmaTot[c] = k[i]
	}
	// 修正:单节点社区 Σ_in 应为 0(没有自环),实际 Louvain 公式从 0 开始累加。这里沿用经典做法:
	// Σ_in 在节点加入时 +k_i,移除时 -k_i;最终值 = 该社区内部边权和 × 2。
	// 单节点社区 Σ_in = 0 更直观,但 ΔQ 公式等价。直接以 0 起步更清晰:
	for c := range sigmaIn {
		sigmaIn[c] = 0
	}

	for range 20 {
		improved := false
		for i := range entIDs {
			iID := entIDs[i]
			cur := comm[i]
			// 把 i 从当前社区临时移出,更新 Σ_in/Σ_tot
			sigmaIn[cur] -= k[i]
			sigmaTot[cur] -= k[i]
			// k_i,in[c]: i 到社区 c(不含当前社区)中节点的边权和
			kiIn := make(map[int]float64)
			for _, e := range adj[iID] {
				if idx[e.other] < 0 {
					continue
				}
				j := idx[e.other]
				if j == i {
					continue
				}
				c := comm[j]
				kiIn[c] += float64(e.weight)
			}
			// 评估每个候选社区 + "独立成新社区"
			bestC := cur
			bestGain := 0.0
			for c, w := range kiIn {
				gain := w/twoM - sigmaTot[c]*k[i]/(twoM*twoM)
				if gain > bestGain {
					bestGain = gain
					bestC = c
				}
			}
			// 单节点社区的 gain = -k[i]^2/(2m)^2(从无社区到自己)
			soloGain := -k[i] * k[i] / (twoM * twoM)
			if soloGain > bestGain {
				bestGain = soloGain
				bestC = i
			}
			if bestC != cur {
				improved = true
				comm[i] = bestC
				if bestC == i {
					// 新建单节点社区
					sigmaIn[bestC] = 0
					sigmaTot[bestC] = k[i]
				} else {
					sigmaIn[bestC] += k[i]
					sigmaTot[bestC] += k[i]
				}
			} else {
				// 留在原社区,恢复 Σ_in/Σ_tot
				sigmaIn[cur] += k[i]
				sigmaTot[cur] += k[i]
			}
		}
		if !improved {
			break
		}
	}

	// 把社区 id 压缩成连续整数(便于排序),并按社区大小降序
	commRenamed := make(map[int]int, len(entIDs))
	next := 0
	for _, c := range comm {
		if _, ok := commRenamed[c]; !ok {
			commRenamed[c] = next
			next++
		}
	}
	out := make(map[int64]int64, len(entIDs))
	for i, id := range entIDs {
		out[id] = int64(commRenamed[comm[i]])
	}
	return out
}

type relEdge struct {
	other  int64
	weight int
}

type loadedGraph struct {
	entities  []Entity
	relations []Relation
	adj       map[int64][]relEdge // 出/入 邻接(无向图视角)
}

func loadGraph(ctx context.Context, db *gorm.DB) (*loadedGraph, error) {
	var entities []Entity
	rows, err := db.WithContext(ctx).Raw(`SELECT id, name, type, description, degree FROM kg_entities`).Rows()
	if err != nil {
		return nil, fmt.Errorf("load entities: %w", err)
	}
	for rows.Next() {
		var e Entity
		if err := rows.Scan(&e.ID, &e.Name, &e.Type, &e.Description, &e.Degree); err != nil {
			rows.Close()
			return nil, err
		}
		entities = append(entities, e)
	}
	rows.Close()

	var relations []Relation
	rows, err = db.WithContext(ctx).Raw(
		`SELECT id, source_id, target_id, COALESCE(description,''), weight FROM kg_relations`).Rows()
	if err != nil {
		return nil, fmt.Errorf("load relations: %w", err)
	}
	for rows.Next() {
		var r Relation
		if err := rows.Scan(&r.ID, &r.SourceID, &r.TargetID, &r.Description, &r.Weight); err != nil {
			rows.Close()
			return nil, err
		}
		relations = append(relations, r)
	}
	rows.Close()

	adj := make(map[int64][]relEdge, len(entities))
	for _, r := range relations {
		adj[r.SourceID] = append(adj[r.SourceID], relEdge{other: r.TargetID, weight: r.Weight})
		adj[r.TargetID] = append(adj[r.TargetID], relEdge{other: r.SourceID, weight: r.Weight})
	}
	return &loadedGraph{entities: entities, relations: relations, adj: adj}, nil
}

const communitySummaryTpl = `以下是知识图谱中一个社区的实体和关系列表。请用 2-3 句话总结这个社区的主题和核心实体。

实体:
%s

关系:
%s

社区摘要:`

// detectAndSummarize 跑 Louvain → 按社区分组 → 每社区 LLM 摘要 → 一次性写 kg_communities。
func detectAndSummarize(ctx context.Context, db *gorm.DB, llm infrastructure.LLM) error {
	g, err := loadGraph(ctx, db)
	if err != nil {
		return err
	}
	if len(g.entities) == 0 {
		fmt.Println("[COMMUNITY] no entities, skip")
		return nil
	}
	entIDs := make([]int64, len(g.entities))
	for i, e := range g.entities {
		entIDs[i] = e.ID
	}
	label := louvain(entIDs, g.adj)
	// 按 label 分组
	groups := make(map[int64][]Entity)
	for _, e := range g.entities {
		c := label[e.ID]
		groups[c] = append(groups[c], e)
	}
	// 大小 < 2 的社区归到 "singleton",不写库(避免噪声)
	var keep []int64
	for c, members := range groups {
		if len(members) >= 2 {
			keep = append(keep, c)
		}
	}
	sort.Slice(keep, func(i, j int) bool { return len(groups[keep[i]]) > len(groups[keep[j]]) })
	slog.Info(fmt.Sprintf("[COMMUNITY] %d entities → %d non-trivial communities\n", len(g.entities), len(keep)))

	// 关系按 (src,tgt) label 落入同一社区的写进 prompt
	relByCommunity := make(map[int64][]Relation)
	for _, r := range g.relations {
		sc, sok := label[r.SourceID]
		tc, tok := label[r.TargetID]
		if !sok || !tok || sc != tc {
			continue
		}
		relByCommunity[sc] = append(relByCommunity[sc], r)
	}

	// 清空旧社区
	if err := db.WithContext(ctx).Exec(`DELETE FROM kg_communities`).Error; err != nil {
		return fmt.Errorf("clear communities: %w", err)
	}
	for ci, cid := range keep {
		members := groups[cid]
		rels := relByCommunity[cid]
		summary, err := summarizeCommunity(ctx, llm, members, rels)
		if err != nil {
			slog.Info(fmt.Sprintf("            community %d summary failed: %v\n", ci+1, err))
			continue
		}
		entityIDs := make([]int64, len(members))
		for i, m := range members {
			entityIDs[i] = m.ID
		}
		if err := db.WithContext(ctx).Exec(`
			INSERT INTO kg_communities(level, entity_ids, summary)
			VALUES (1, ?, ?)`,
			pqInt64(entityIDs), summary,
		).Error; err != nil {
			return fmt.Errorf("insert community: %w", err)
		}
		fmt.Printf("            community %d: %d entities → %s\n",
			ci+1, len(members), ragcore.Snippet(summary, 80))
	}
	return nil
}

func summarizeCommunity(ctx context.Context, llm infrastructure.LLM, members []Entity, rels []Relation) (string, error) {
	var entLines, relLines []string
	for _, m := range members {
		entLines = append(entLines, fmt.Sprintf("- %s (%s): %s", m.Name, m.Type, m.Description))
	}
	for _, r := range rels {
		relLines = append(relLines, fmt.Sprintf("- %d → %d: %s (w=%d)",
			r.SourceID, r.TargetID, r.Description, r.Weight))
	}
	prompt := fmt.Sprintf(communitySummaryTpl,
		joinLines(entLines), joinLines(relLines))
	out, err := llm.Complete(ctx, prompt)
	if err != nil {
		return "", err
	}
	return ragcore.StripThink(out), nil
}

func joinLines(s []string) string {
	var out strings.Builder
	for _, l := range s {
		out.WriteString(l + "\n")
	}
	return out.String()
}

// sanityModularity 留个诊断函数,runGraph 末尾打印当前 Q 方便看 Louvain 质量。
// 不重要,纯教学用。
func sanityModularity(g *loadedGraph, label map[int64]int64) float64 {
	m := 0.0
	for _, edges := range g.adj {
		for _, e := range edges {
			m += float64(e.weight)
		}
	}
	if m == 0 {
		return 0
	}
	twoM := 2 * m
	idx := make(map[int64]int, len(g.entities))
	for i, e := range g.entities {
		idx[e.ID] = i
	}
	sigmaIn := make(map[int64]float64)
	sigmaTot := make(map[int64]float64)
	for _, e := range g.entities {
		c := label[e.ID]
		var ki float64
		for _, ed := range g.adj[e.ID] {
			ki += float64(ed.weight)
		}
		sigmaTot[c] += ki
	}
	for _, r := range g.relations {
		if label[r.SourceID] == label[r.TargetID] {
			sigmaIn[label[r.SourceID]] += 2 * float64(r.Weight)
		}
	}
	q := 0.0
	for c := range sigmaTot {
		q += sigmaIn[c]/twoM - math.Pow(sigmaTot[c]/twoM, 2)
	}
	return q
}
