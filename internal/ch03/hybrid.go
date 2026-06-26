package ch03

import (
	"context"
	"fmt"
	"log/slog"

	"rag/infrastructure"
	"rag/internal"
	"rag/internal/ch02/splitter"
	"rag/internal/ragcore"
	"rag/sample"
)

func init() {
	internal.Register(internal.Lesson{
		Name:        "hybrid",
		Description: "L3: Hybrid RAG (BM25 + 向量混合检索)",
		Migrate:     migrateHybrid,
		Run:         runHybrid,
	})
}

// migrateHybrid 落 L3 schema：在 L2 基础上加 BM25 索引(paradedb pg_search 扩展)。
// pg_search 基于 Tantivy + jieba 中文分词,@@@ 是 BM25 匹配算子,paradedb.score(id) 取 BM25 分。
func migrateHybrid(ctx context.Context, deps internal.Deps) error {
	db := deps.DB
	return infrastructure.EnsureHybridSchema(ctx, db)
}

var l3Sample = sample.InferX

const (
	demoQuestion3 = "P99 延迟突然飙高,按什么顺序排查?"
	topK3         = 5
)

func runHybrid(ctx context.Context, deps internal.Deps, _ []string) error {
	parentCfg := splitter.DefaultConfig()
	childCfg := splitter.DefaultConfig()
	childCfg.ChunkSize = 120
	childCfg.ChunkOverlap = 0

	pc := splitter.SplitParentChild(l3Sample, parentCfg, childCfg)
	fmt.Printf("[INDEXING] dense + bm25 → %d chunks (across %d parents)\n",
		len(pc.Children), len(pc.Parents))

	if err := Ingest(ctx, deps.DB, deps.Embedder,
		Document{Title: "L3 sample", Lang: "zh"},
		pc.Parents, pc.Children,
	); err != nil {
		return fmt.Errorf("ingest: %w", err)
	}

	slog.Info(fmt.Sprintf("[QUERY] %q\n", demoQuestion3))
	dense, err := DenseSearch(ctx, deps.DB, deps.Embedder, demoQuestion3, topK3)
	if err != nil {
		return fmt.Errorf("dense: %w", err)
	}
	bm25, err := BM25Search(ctx, deps.DB, demoQuestion3, topK3)
	if err != nil {
		return fmt.Errorf("bm25: %w", err)
	}
	fused := RRF([][]Hit{dense, bm25}, 60)
	if len(fused) > topK3 {
		fused = fused[:topK3]
	}

	slog.Info(fmt.Sprintf("[DENSE-ONLY]  top-%d hits = %s\n", topK3, ragcore.IDsOf(dense)))
	slog.Info(fmt.Sprintf("[BM25-ONLY]   top-%d hits = %s\n", topK3, ragcore.IDsOf(bm25)))
	slog.Info(fmt.Sprintf("[HYBRID-RRF]  top-%d hits = %s\n", len(fused), ragcore.IDsOf(fused)))

	chunks, err := LoadChunks(ctx, deps.DB, fused)
	if err != nil {
		return fmt.Errorf("load chunks: %w", err)
	}

	fmt.Println("[ANSWERING]")
	ans, err := Generate(ctx, deps.LLM, demoQuestion3, chunks)
	if err != nil {
		return fmt.Errorf("generate: %w", err)
	}
	fmt.Println(ans)
	return nil
}
