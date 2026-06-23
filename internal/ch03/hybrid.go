package ch03

import (
	"context"
	"fmt"
	"strings"

	"gorm.io/gorm"

	"rag/internal"
	"rag/internal/ch02"
	"rag/internal/ch02/splitter"
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
func migrateHybrid(ctx context.Context, db *gorm.DB) error {
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

var l3Sample = sample.RAG

const (
	demoQuestion3 = "RAG 的核心流程是什么?"
	topK3         = 3
)

func runHybrid(ctx context.Context, deps internal.Deps, _ []string) error {
	parentCfg := splitter.DefaultConfig()
	childCfg := splitter.DefaultConfig()
	childCfg.ChunkSize = 120
	childCfg.ChunkOverlap = 0

	pc := ch02.SplitParentChild(l3Sample, parentCfg, childCfg)
	fmt.Printf("[INDEXING] dense + bm25 → %d chunks (across %d parents)\n",
		len(pc.Children), len(pc.Parents))

	if err := Ingest(ctx, deps.DB, deps.Embedder,
		Document{Title: "L3 sample", Lang: "zh"},
		pc.Parents, pc.Children,
	); err != nil {
		return fmt.Errorf("ingest: %w", err)
	}

	fmt.Printf("[QUERY] %q\n", demoQuestion3)
	dense, err := DenseTopN(ctx, deps.DB, deps.Embedder, demoQuestion3, topK3)
	if err != nil {
		return fmt.Errorf("dense: %w", err)
	}
	bm25, err := BM25TopN(ctx, deps.DB, demoQuestion3, topK3)
	if err != nil {
		return fmt.Errorf("bm25: %w", err)
	}
	fused := RRF([][]Hit{dense, bm25}, 60)
	if len(fused) > topK3 {
		fused = fused[:topK3]
	}

	fmt.Printf("[DENSE-ONLY]  top-%d hits = %s\n", topK3, idsOf(dense))
	fmt.Printf("[BM25-ONLY]   top-%d hits = %s\n", topK3, idsOf(bm25))
	fmt.Printf("[HYBRID-RRF]  top-%d hits = %s\n", len(fused), idsOf(fused))

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

func idsOf(hits []Hit) string {
	if len(hits) == 0 {
		return "[]"
	}
	parts := make([]string, len(hits))
	for i, h := range hits {
		parts[i] = fmt.Sprintf("%d", h.ChunkID)
	}
	return "[" + strings.Join(parts, " ") + "]"
}
