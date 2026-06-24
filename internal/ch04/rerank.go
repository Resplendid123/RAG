package ch04

import (
	"context"
	"fmt"
	"strings"

	"gorm.io/gorm"

	"rag/internal"
	"rag/internal/ch02"
	"rag/internal/ch02/splitter"
	"rag/internal/ch03"
	"rag/sample"
)

func init() {
	internal.Register(internal.Lesson{
		Name:        "rerank",
		Description: "L4: Rerank RAG (Cross-Encoder 精排)",
		Migrate:     migrateRerank,
		Run:         runRerank,
	})
}

// migrateRerank L4 schema 与 L3 hybrid 一致:child + parent + dense + bm25,rerank 在 query 时跑,不落库。
func migrateRerank(ctx context.Context, db *gorm.DB) error {
	if err := db.WithContext(ctx).Exec(`CREATE EXTENSION IF NOT EXISTS pg_search`).Error; err != nil {
		return fmt.Errorf("create extension pg_search: %w", err)
	}
	return db.WithContext(ctx).Exec(`
		DROP TABLE IF EXISTS document_chunks CASCADE;
		DROP TABLE IF EXISTS document_chunks_parent CASCADE;
		DROP TABLE IF EXISTS documents CASCADE;

		CREATE TABLE documents (
			id          BIGSERIAL PRIMARY KEY,
			title       TEXT NOT NULL,
			source_url  TEXT,
			lang        TEXT DEFAULT 'zh',
			content_hash TEXT,
			created_at  TIMESTAMPTZ DEFAULT now()
		);

		CREATE TABLE document_chunks_parent (
			id          BIGSERIAL PRIMARY KEY,
			document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
			chunk_index INTEGER NOT NULL,
			content     TEXT NOT NULL,
			token_count INTEGER,
			created_at  TIMESTAMPTZ DEFAULT NOW()
		);

		CREATE TABLE document_chunks (
			id          BIGSERIAL PRIMARY KEY,
			parent_id   BIGINT REFERENCES document_chunks_parent(id) ON DELETE CASCADE,
			document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
			chunk_index INT NOT NULL,
			content     TEXT NOT NULL,
			embedding   VECTOR(1024),
			created_at  TIMESTAMPTZ DEFAULT now()
		);
		CREATE INDEX document_chunks_embedding_hnsw_idx
			ON document_chunks USING hnsw (embedding vector_cosine_ops);
		CREATE INDEX document_chunks_bm25_idx
			ON document_chunks
			USING bm25 (id, content)
			WITH (key_field='id');
	`).Error
}

var l4Sample = sample.InferX

const (
	demoQuestion4 = "P99 延迟突然飙高,按什么顺序排查?"
	recallTopN4   = 10 // 宽召回;生产 50-100
	rerankTopK4   = 5  // rerank 后喂 LLM 的 top-K
)

func runRerank(ctx context.Context, deps internal.Deps, _ []string) error {
	parentCfg := splitter.DefaultConfig()
	childCfg := splitter.DefaultConfig()
	childCfg.ChunkSize = 120
	childCfg.ChunkOverlap = 0

	pc := ch02.SplitParentChild(l4Sample, parentCfg, childCfg)
	fmt.Printf("[INDEXING] dense + bm25 → %d chunks (across %d parents)\n",
		len(pc.Children), len(pc.Parents))

	if err := ch03.Ingest(ctx, deps.DB, deps.Embedder,
		ch03.Document{Title: "L4 sample", Lang: "zh"},
		pc.Parents, pc.Children,
	); err != nil {
		return fmt.Errorf("ingest: %w", err)
	}

	fmt.Printf("[QUERY] %q\n", demoQuestion4)
	dense, err := ch03.DenseTopN(ctx, deps.DB, deps.Embedder, demoQuestion4, recallTopN4)
	if err != nil {
		return fmt.Errorf("dense: %w", err)
	}
	bm25, err := ch03.BM25TopN(ctx, deps.DB, demoQuestion4, recallTopN4)
	if err != nil {
		return fmt.Errorf("bm25: %w", err)
	}
	fused := ch03.RRF([][]ch03.Hit{dense, bm25}, 60)
	fmt.Printf("[RECALL] hybrid → %d candidates\n", len(fused))

	// before:RRF 排序直接取 top-K,作为 rerank 对照基线。
	before := fused
	if len(before) > rerankTopK4 {
		before = before[:rerankTopK4]
	}

	// rerank:抽 content 喂 LLM,按 rerank 后的 index 重排 fused。
	docs := make([]string, len(fused))
	for i, h := range fused {
		docs[i] = h.Content
	}
	rr := NewLLMReranker(deps.LLM)
	ranked := rr.Rerank(ctx, demoQuestion4, docs)

	after := make([]ch03.Hit, 0, rerankTopK4)
	scoreSummary := make([]string, 0, rerankTopK4)
	for i, r := range ranked {
		if i >= rerankTopK4 {
			break
		}
		after = append(after, fused[r.Index])
		scoreSummary = append(scoreSummary, fmt.Sprintf("idx=%d score=%.1f", fused[r.Index].ChunkID, r.Score))
	}
	fmt.Printf("[BEFORE RERANK] top-%d hits = %s   (RRF 排序)\n", len(before), idsOf4(before))
	fmt.Printf("[AFTER RERANK]  top-%d hits = %s   (%s)\n", len(after), idsOf4(after), strings.Join(scoreSummary, ", "))

	chunks, err := ch03.LoadChunks(ctx, deps.DB, after)
	if err != nil {
		return fmt.Errorf("load chunks: %w", err)
	}

	fmt.Println("[ANSWERING]")
	ans, err := ch03.Generate(ctx, deps.LLM, demoQuestion4, chunks)
	if err != nil {
		return fmt.Errorf("generate: %w", err)
	}
	fmt.Println(ans)
	return nil
}

func idsOf4(hits []ch03.Hit) string {
	if len(hits) == 0 {
		return "[]"
	}
	parts := make([]string, len(hits))
	for i, h := range hits {
		parts[i] = fmt.Sprintf("%d", h.ChunkID)
	}
	return "[" + strings.Join(parts, " ") + "]"
}
