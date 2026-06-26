package ch08

import (
	"context"
	"fmt"
	"log/slog"

	"rag/infrastructure"
	"rag/internal"
	"rag/internal/ch02/splitter"
	"rag/internal/ch03"
	"rag/internal/ragcore"
	"rag/sample"
)

func init() {
	internal.Register(internal.Lesson{
		Name:        "graph",
		Description: "L8: Graph RAG (知识图谱与社区检索)",
		Migrate:     migrateGraph,
		Run:         runGraph,
	})
}

var l8Sample = sample.BlueWhale

// runGraph 索引一次语料(ingest → extract → community → summary),然后跑 4 个 demo question
// 覆盖 Basic / Local / Global / DRIFT 四种 query 模式,直观看各自召回差异。
func runGraph(ctx context.Context, deps internal.Deps, _ []string) error {
	parentCfg := splitter.DefaultConfig()
	childCfg := splitter.DefaultConfig()
	childCfg.ChunkSize = 200
	childCfg.ChunkOverlap = 0

	pc := splitter.SplitParentChild(l8Sample, parentCfg, childCfg)
	fmt.Printf("[INDEXING] parent + child → %d chunks (across %d parents)\n",
		len(pc.Children), len(pc.Parents))

	if err := ch03.Ingest(ctx, deps.DB, deps.Embedder,
		infrastructure.Document{Title: "L8 sample", Lang: "zh"},
		pc.Parents, pc.Children,
	); err != nil {
		return fmt.Errorf("ingest: %w", err)
	}

	// 图谱构建:仅当 kg_entities 为空时跑(便于多次 demo 不重复抽)。
	var count int64
	if err := deps.DB.WithContext(ctx).Raw(`SELECT COUNT(*) FROM kg_entities`).Scan(&count).Error; err != nil {
		return fmt.Errorf("count entities: %w", err)
	}
	if count == 0 {
		if err := extractFromChunks(ctx, deps.DB, deps.LLM, deps.Embedder, pc.Children); err != nil {
			return fmt.Errorf("extract: %w", err)
		}
	} else {
		slog.Info(fmt.Sprintf("[INDEXING] kg_entities already has %d rows, skip extract\n", count))
	}

	// 社区检测 + 摘要:同上,有则跳过。
	if err := deps.DB.WithContext(ctx).Raw(`SELECT COUNT(*) FROM kg_communities`).Scan(&count).Error; err != nil {
		return fmt.Errorf("count communities: %w", err)
	}
	if count == 0 {
		if err := detectAndSummarize(ctx, deps.DB, deps.LLM); err != nil {
			return fmt.Errorf("community: %w", err)
		}
	} else {
		slog.Info(fmt.Sprintf("[INDEXING] kg_communities already has %d rows, skip\n", count))
	}

	type demo struct {
		question string
		mode     QueryMode
	}
	demos := []demo{
		{"蓝鲸 2024 赛季最终排名如何?", ModeBasic},    // 走 dense 召回原文
		{"陈昊本赛季表现和奖项情况?", ModeLocal},        // 走实体 → 邻居 → chunk
		{"总结蓝鲸本赛季的整体表现", ModeGlobal},        // 走社区摘要
		{"马丁内斯的战术调整和赛季关键节点有哪些?", ModeDRIFT}, // 走局部+社区
	}
	for i, d := range demos {
		slog.Info(fmt.Sprintf("\n[RUN %d] mode=%s  %q\n", i+1, d.mode, d.question))
		var ans string
		var err error
		switch d.mode {
		case ModeBasic:
			ans, err = BasicSearch(ctx, deps.DB, deps.Embedder, deps.LLM, d.question, 5)
		case ModeLocal:
			ans, err = LocalSearch(ctx, deps.DB, deps.Embedder, deps.LLM, d.question)
		case ModeGlobal:
			ans, err = GlobalSearch(ctx, deps.DB, deps.LLM, d.question)
		case ModeDRIFT:
			ans, err = DRIFTSearch(ctx, deps.DB, deps.Embedder, deps.LLM, d.question)
		default:
			err = fmt.Errorf("unknown mode %s", d.mode)
		}
		if err != nil {
			slog.Info(fmt.Sprintf("[ERROR] %v\n", err))
			continue
		}
		slog.Info(fmt.Sprintf("[ANSWER]\n%s\n", ragcore.Snippet(ragcore.StripThink(ans), 800)))
	}

	// 顺带演示 router:用一句新 query 看分类结果。
	routedQ := "性能优化方面有哪些建议?"
	slog.Info(fmt.Sprintf("\n[ROUTER] %q → mode=%s\n", routedQ, routeQuery(routedQ)))

	return nil
}
