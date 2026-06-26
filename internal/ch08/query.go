package ch08

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/pgvector/pgvector-go"
	"gorm.io/gorm"

	"rag/infrastructure"
	"rag/internal/ch02/splitter"
	"rag/internal/ch03"
	"rag/internal/ragcore"
)

type QueryMode string

const (
	ModeBasic  QueryMode = "basic"
	ModeLocal  QueryMode = "local"
	ModeGlobal QueryMode = "global"
	ModeDRIFT  QueryMode = "drift"
)

const (
	localTopEntities = 3
	localFanoutHops  = 2
)

var globalKeywords = []string{"总结", "概括", "整体", "全文", "主要", "主题", "趋势", "全局", "概况", "都讲了什么"}

// routeQuery 关键词分类:命中"全局"语义词 → Global;否则默认 Local(占比最高、最稳)。
func routeQuery(q string) QueryMode {
	for _, kw := range globalKeywords {
		if strings.Contains(q, kw) {
			return ModeGlobal
		}
	}
	return ModeLocal
}

// BasicSearch 走 ch03.DenseSearch + 直答,等价 L1 朴素 RAG。留这条路径是为了和 Graph 模式做 baseline 对比。
const basicPromptTpl = `基于以下参考资料回答问题。若资料不足,请回答"我不知道"。

参考资料:
%s

问题:%s
答案:`

func BasicSearch(ctx context.Context, db *gorm.DB, emb infrastructure.Embedder, llm infrastructure.LLM, q string, topK int) (string, error) {
	hits, err := ch03.DenseSearch(ctx, db, emb, q, topK)
	if err != nil {
		return "", fmt.Errorf("basic dense: %w", err)
	}
	chunks, err := ch03.LoadChunks(ctx, db, hits)
	if err != nil {
		return "", fmt.Errorf("basic load: %w", err)
	}
	slog.Info(fmt.Sprintf("            dense top-%d → %d chunks\n", topK, len(chunks)))
	return llm.Complete(ctx, fmt.Sprintf(basicPromptTpl, ragcore.FormatNumbered(chunks), q))
}

// =============== helpers ===============

// pqInt64 把 []int64 转成 Postgres 数组字面量 "{1,2,3}",免引入 pq.Array。
func pqInt64(ids []int64) string {
	if len(ids) == 0 {
		return "{}"
	}
	parts := make([]string, len(ids))
	for i, id := range ids {
		parts[i] = fmt.Sprintf("%d", id)
	}
	return "{" + strings.Join(parts, ",") + "}"
}

func loadTopEntities(ctx context.Context, db *gorm.DB, emb infrastructure.Embedder, q string, k int) ([]Entity, error) {
	vecs, err := emb.Embed(ctx, []string{q})
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}
	rows, err := db.WithContext(ctx).Raw(`
		SELECT id, name, type, description, degree
		FROM kg_entities
		ORDER BY embedding <=> ?
		LIMIT ?`, pgvector.NewVector(vecs[0]), k).Rows()
	if err != nil {
		return nil, fmt.Errorf("top entities: %w", err)
	}
	defer rows.Close()
	var out []Entity
	for rows.Next() {
		var e Entity
		if err := rows.Scan(&e.ID, &e.Name, &e.Type, &e.Description, &e.Degree); err != nil {
			return nil, err
		}
		out = append(out, e)
	}
	return out, rows.Err()
}

// expandNeighbors 从 seeds 出发 fan-out N 跳,返回闭包内全部 entity id(含 seeds)。
func expandNeighbors(db *gorm.DB, seeds []int64, hops int) ([]int64, error) {
	if len(seeds) == 0 || hops <= 0 {
		return seeds, nil
	}
	visited := make(map[int64]struct{}, len(seeds))
	for _, s := range seeds {
		visited[s] = struct{}{}
	}
	frontier := append([]int64(nil), seeds...)
	for h := range hops {
		if len(frontier) == 0 {
			break
		}
		rows, err := db.Raw(`
			SELECT DISTINCT
			       CASE WHEN source_id = ANY(?) THEN target_id ELSE source_id END AS other
			FROM kg_relations
			WHERE source_id = ANY(?) OR target_id = ANY(?)`,
			pqInt64(frontier), pqInt64(frontier), pqInt64(frontier),
		).Rows()
		if err != nil {
			return nil, fmt.Errorf("fanout hop %d: %w", h, err)
		}
		var next []int64
		for rows.Next() {
			var id int64
			if err := rows.Scan(&id); err != nil {
				rows.Close()
				return nil, err
			}
			if _, ok := visited[id]; ok {
				continue
			}
			visited[id] = struct{}{}
			next = append(next, id)
		}
		rows.Close()
		frontier = next
	}
	out := make([]int64, 0, len(visited))
	for id := range visited {
		out = append(out, id)
	}
	return out, nil
}

func loadChunksForEntities(ctx context.Context, db *gorm.DB, entIDs []int64) ([]splitter.Chunk, error) {
	if len(entIDs) == 0 {
		return nil, nil
	}
	rows, err := db.WithContext(ctx).Raw(`
		SELECT DISTINCT c.id
		FROM document_chunks c
		JOIN kg_entity_mentions m ON m.text_unit_id = c.id
		WHERE m.entity_id = ANY(?)`, pqInt64(entIDs)).Rows()
	if err != nil {
		return nil, fmt.Errorf("chunks for entities: %w", err)
	}
	defer rows.Close()
	var ids []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	hits := make([]ch03.Hit, len(ids))
	for i, id := range ids {
		hits[i] = ch03.Hit{ChunkID: id, Rank: i}
	}
	return ch03.LoadChunks(ctx, db, hits)
}

func loadRelationsForEntities(ctx context.Context, db *gorm.DB, entIDs []int64) ([]Relation, error) {
	if len(entIDs) == 0 {
		return nil, nil
	}
	rows, err := db.WithContext(ctx).Raw(`
		SELECT id, source_id, target_id, COALESCE(description,''), weight
		FROM kg_relations
		WHERE source_id = ANY(?) OR target_id = ANY(?)`,
		pqInt64(entIDs), pqInt64(entIDs),
	).Rows()
	if err != nil {
		return nil, fmt.Errorf("relations: %w", err)
	}
	defer rows.Close()
	var out []Relation
	for rows.Next() {
		var r Relation
		if err := rows.Scan(&r.ID, &r.SourceID, &r.TargetID, &r.Description, &r.Weight); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

func loadEntitiesByIDs(ctx context.Context, db *gorm.DB, ids []int64) ([]Entity, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	rows, err := db.WithContext(ctx).Raw(`
		SELECT id, name, type, description, degree
		FROM kg_entities WHERE id = ANY(?)`, pqInt64(ids)).Rows()
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Entity
	for rows.Next() {
		var e Entity
		if err := rows.Scan(&e.ID, &e.Name, &e.Type, &e.Description, &e.Degree); err != nil {
			return nil, err
		}
		out = append(out, e)
	}
	return out, rows.Err()
}

func formatEntities(ents []Entity) string {
	if len(ents) == 0 {
		return "(no entities)"
	}
	var b strings.Builder
	for i, e := range ents {
		fmt.Fprintf(&b, "- [%d] %s (%s): %s (degree=%d)\n", i+1, e.Name, e.Type, e.Description, e.Degree)
	}
	return b.String()
}

func formatRelations(rels []Relation) string {
	if len(rels) == 0 {
		return "(no relations)"
	}
	var b strings.Builder
	for _, r := range rels {
		fmt.Fprintf(&b, "- %d → %d: %s (w=%d)\n", r.SourceID, r.TargetID, r.Description, r.Weight)
	}
	return b.String()
}

// =============== Local Search ===============

const localPromptTpl = `基于以下知识图谱信息回答用户问题(中文)。优先依据参考资料;若信息不足,请回答"我不知道"。

相关实体:
%s

实体间关系:
%s

相关 chunk 文本:
%s

问题:%s
答案:`

func LocalSearch(ctx context.Context, db *gorm.DB, emb infrastructure.Embedder, llm infrastructure.LLM, q string) (string, error) {
	seeds, err := loadTopEntities(ctx, db, emb, q, localTopEntities)
	if err != nil {
		return "", err
	}
	seedIDs := make([]int64, len(seeds))
	for i, s := range seeds {
		seedIDs[i] = s.ID
	}
	neighbors, err := expandNeighbors(db, seedIDs, localFanoutHops)
	if err != nil {
		return "", err
	}
	rels, err := loadRelationsForEntities(ctx, db, neighbors)
	if err != nil {
		return "", err
	}
	chunks, err := loadChunksForEntities(ctx, db, neighbors)
	if err != nil {
		return "", err
	}
	allEnts := append([]Entity{}, seeds...)
	seen := make(map[int64]bool, len(seeds))
	for _, s := range seeds {
		seen[s.ID] = true
	}
	if len(neighbors) > len(seeds) {
		extra, err := loadEntitiesByIDs(ctx, db, neighbors)
		if err != nil {
			return "", err
		}
		for _, e := range extra {
			if !seen[e.ID] {
				allEnts = append(allEnts, e)
				seen[e.ID] = true
			}
		}
	}

	fmt.Printf("            seeds=%d, fanout(%d hops)=%d, chunks=%d, relations=%d\n",
		len(seeds), localFanoutHops, len(neighbors), len(chunks), len(rels))

	prompt := fmt.Sprintf(localPromptTpl,
		formatEntities(allEnts),
		formatRelations(rels),
		ragcore.FormatNumbered(chunks),
		q,
	)
	return llm.Complete(ctx, prompt)
}

// =============== Global Search ===============

const globalMapTpl = `你是问答系统的一个 worker。基于以下社区摘要,回答与用户问题相关的部分。只写与问题相关的要点,1-2 句;若不相关,回答"无关"。

社区摘要:%s

问题:%s

相关要点:`

const globalReduceTpl = `综合多个 worker 给出的部分要点,给出对用户问题的最终回答(中文)。
- 优先要点充分、互相印证的内容。
- 若要点互相冲突,列出冲突并给出你的判断。
- 若所有要点都不相关,回答"我不知道"。

问题:%s

部分要点:
%s

最终答案:`

func loadCommunities(ctx context.Context, db *gorm.DB) ([]Community, error) {
	rows, err := db.WithContext(ctx).Raw(`
		SELECT id, level, COALESCE(entity_ids::text, '{}'), COALESCE(summary,'')
		FROM kg_communities ORDER BY id`).Rows()
	if err != nil {
		return nil, fmt.Errorf("load communities: %w", err)
	}
	defer rows.Close()
	var out []Community
	for rows.Next() {
		var c Community
		var arrStr string
		if err := rows.Scan(&c.ID, &c.Level, &arrStr, &c.Summary); err != nil {
			return nil, err
		}
		c.EntityIDs = parseInt64Array(arrStr)
		out = append(out, c)
	}
	return out, rows.Err()
}

// parseInt64Array 解析 Postgres 数组字面量 "{1,2,3}" → []int64。
func parseInt64Array(s string) []int64 {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "{")
	s = strings.TrimSuffix(s, "}")
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]int64, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" || p == "NULL" {
			continue
		}
		var n int64
		fmt.Sscanf(p, "%d", &n)
		if n != 0 || p == "0" {
			out = append(out, n)
		}
	}
	return out
}

func GlobalSearch(ctx context.Context, db *gorm.DB, llm infrastructure.LLM, q string) (string, error) {
	comms, err := loadCommunities(ctx, db)
	if err != nil {
		return "", err
	}
	if len(comms) == 0 {
		return "", fmt.Errorf("no communities; run indexing first")
	}
	slog.Info(fmt.Sprintf("            communities=%d\n", len(comms)))
	partials := make([]string, 0, len(comms))
	for i, c := range comms {
		out, err := llm.Complete(ctx, fmt.Sprintf(globalMapTpl, c.Summary, q))
		if err != nil {
			slog.Info(fmt.Sprintf("            community %d map failed: %v\n", i+1, err))
			continue
		}
		if strings.Contains(out, "无关") {
			continue
		}
		partials = append(partials, fmt.Sprintf("[社区 %d]\n%s", i+1, strings.TrimSpace(out)))
	}
	if len(partials) == 0 {
		return "I don't know based on current community summaries.", nil
	}
	return llm.Complete(ctx, fmt.Sprintf(globalReduceTpl, q, strings.Join(partials, "\n\n")))
}

// =============== DRIFT Search ===============

const driftPromptTpl = `基于以下信息回答用户问题(中文)。信息分两层:
1. 局部:命中的实体、关系、相关文本片段(用于细节回答)
2. 社区:所属社区的摘要(用于主题对齐)

实体:
%s

实体关系:
%s

相关文本片段:
%s

所属社区摘要:
%s

问题:%s
答案:`

func loadCommunitiesForEntities(ctx context.Context, db *gorm.DB, entIDs []int64) (string, error) {
	if len(entIDs) == 0 {
		return "(no community context)", nil
	}
	rows, err := db.WithContext(ctx).Raw(`
		SELECT DISTINCT c.id, c.summary
		FROM kg_communities c
		WHERE c.entity_ids && ?`, pqInt64(entIDs)).Rows()
	if err != nil {
		return "", fmt.Errorf("load community summaries: %w", err)
	}
	defer rows.Close()
	var parts []string
	for rows.Next() {
		var id int64
		var summary string
		if err := rows.Scan(&id, &summary); err != nil {
			return "", err
		}
		parts = append(parts, fmt.Sprintf("[社区 %d] %s", id, summary))
	}
	if len(parts) == 0 {
		return "(no community context)", nil
	}
	return strings.Join(parts, "\n"), nil
}

func DRIFTSearch(ctx context.Context, db *gorm.DB, emb infrastructure.Embedder, llm infrastructure.LLM, q string) (string, error) {
	seeds, err := loadTopEntities(ctx, db, emb, q, localTopEntities)
	if err != nil {
		return "", err
	}
	seedIDs := make([]int64, len(seeds))
	for i, s := range seeds {
		seedIDs[i] = s.ID
	}
	neighbors, err := expandNeighbors(db, seedIDs, localFanoutHops)
	if err != nil {
		return "", err
	}
	rels, err := loadRelationsForEntities(ctx, db, neighbors)
	if err != nil {
		return "", err
	}
	chunks, err := loadChunksForEntities(ctx, db, neighbors)
	if err != nil {
		return "", err
	}
	allEnts := append([]Entity{}, seeds...)
	seen := make(map[int64]bool, len(seeds))
	for _, s := range seeds {
		seen[s.ID] = true
	}
	extra, err := loadEntitiesByIDs(ctx, db, neighbors)
	if err != nil {
		return "", err
	}
	for _, e := range extra {
		if !seen[e.ID] {
			allEnts = append(allEnts, e)
			seen[e.ID] = true
		}
	}
	commSummaries, err := loadCommunitiesForEntities(ctx, db, seedIDs)
	if err != nil {
		return "", err
	}

	fmt.Printf("            seeds=%d, fanout(%d hops)=%d, chunks=%d, relations=%d\n",
		len(seeds), localFanoutHops, len(neighbors), len(chunks), len(rels))

	prompt := fmt.Sprintf(driftPromptTpl,
		formatEntities(allEnts),
		formatRelations(rels),
		ragcore.FormatNumbered(chunks),
		commSummaries,
		q,
	)
	return llm.Complete(ctx, prompt)
}
