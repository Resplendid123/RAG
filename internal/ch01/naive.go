package ch01

import (
	"context"
	"fmt"

	"gorm.io/gorm"

	"rag/internal"
	"rag/sample"
)

func init() {
	internal.Register(internal.Lesson{
		Name:        "naive",
		Description: "L1: Naive baseline",
		Migrate:     migrateNaive,
		Run:         runNaive,
	})
}

// Document 是 documents 表的最小化业务视图。
type Document struct {
	ID        int64
	Title     string
	SourceURL string
	Lang      string
}

var l1Sample = sample.RAG

const demoQuestion = "RAG 的核心流程是什么?"

func migrateNaive(ctx context.Context, db *gorm.DB) error {
	return db.WithContext(ctx).Exec(`
        CREATE TABLE IF NOT EXISTS documents (
            id          BIGSERIAL PRIMARY KEY,
            title       TEXT NOT NULL,
            source_url  TEXT,
            lang        TEXT DEFAULT 'zh',
            content_hash TEXT UNIQUE,
            created_at  TIMESTAMPTZ DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS document_chunks (
            id          BIGSERIAL PRIMARY KEY,
            document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INT NOT NULL,
            content     TEXT NOT NULL,
            embedding   VECTOR(1024),  -- 与 config.yaml 的 embedding.dimension 一致
            created_at  TIMESTAMPTZ DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx
            ON document_chunks USING hnsw (embedding vector_cosine_ops);
    `).Error
}

// runNaive 是 L1 的编排入口:chunk → ingest → retrieve → generate 四步串起来。
func runNaive(ctx context.Context, deps internal.Deps, _ []string) error {
	chunksIn := Chunk(l1Sample, 200, 30)
	fmt.Printf("[CHUNKING] → %d chunks (size=200, overlap=30)\n", len(chunksIn))

	if err := Ingest(ctx, deps.DB, deps.Embedder,
		Document{Title: "L1 sample", Lang: "zh"},
		chunksIn,
	); err != nil {
		return fmt.Errorf("ingest: %w", err)
	}

	fmt.Println("[RETRIEVING] → top-3 from l_naive.document_chunks")
	chunks, err := Retrieve(ctx, deps.DB, deps.Embedder, demoQuestion, 3)
	if err != nil {
		return fmt.Errorf("retrieve: %w", err)
	}
	fmt.Printf("[RETRIEVED] %d chunks:\n", len(chunks))
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
