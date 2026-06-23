package ch02

import (
	"context"
	"fmt"

	"gorm.io/gorm"

	"rag/internal"
	"rag/internal/ch02/splitter"
	"rag/sample"
)

func init() {
	internal.Register(internal.Lesson{
		Name:        "chunking",
		Description: "L2: 高级 Chunking 策略",
		Migrate:     migrateChunking,
		Run:         runChunking,
	})
}

func migrateChunking(ctx context.Context, db *gorm.DB) error {
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
	`).Error
}

var l2Sample = sample.RAG

const demoQuestion = "RAG 的核心流程是什么?"

func runChunking(ctx context.Context, deps internal.Deps, _ []string) error {
	parentCfg := splitter.DefaultConfig()
	childCfg := splitter.DefaultConfig()
	childCfg.ChunkSize = 120
	childCfg.ChunkOverlap = 0

	pc := SplitParentChild(l2Sample, parentCfg, childCfg)
	fmt.Printf("[PARENT CHUNKING] → %d parents (size=%d)\n", len(pc.Parents), parentCfg.ChunkSize)
	fmt.Printf("[CHILD CHUNKING]  → %d children (size=%d)\n",
		len(pc.Children), childCfg.ChunkSize)

	if err := Ingest(ctx, deps.DB, deps.Embedder,
		Document{Title: "L2 sample", Lang: "zh"},
		pc.Parents, pc.Children,
	); err != nil {
		return fmt.Errorf("ingest: %w", err)
	}

	fmt.Println("[RETRIEVING] → top-3 children → expand to parents")
	chunks, err := Retrieve(ctx, deps.DB, deps.Embedder, demoQuestion, 3)
	if err != nil {
		return fmt.Errorf("retrieve: %w", err)
	}
	fmt.Printf("[RETRIEVED] %d parents:\n", len(chunks))
	for i, c := range chunks {
		snippet := c.Content
		if r := []rune(snippet); len(r) > 80 {
			snippet = string(r[:80]) + "..."
		}
		fmt.Printf("  [%d] %s\n", i+1, snippet)
	}

	fmt.Println("[ANSWERING]")
	ans, err := Generate(ctx, deps.LLM, demoQuestion, chunks)
	if err != nil {
		return fmt.Errorf("generate: %w", err)
	}
	fmt.Println(ans)
	return nil
}
