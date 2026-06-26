package ch02

import (
	"context"
	"fmt"
	"log/slog"

	"rag/infrastructure"
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

func migrateChunking(ctx context.Context, deps internal.Deps) error {
	db := deps.DB
	return db.WithContext(ctx).Exec(infrastructure.BaseSchemaSQL).Error
}

var l2Sample = sample.RAG

const demoQuestion = "RAG 的核心流程是什么?"

func runChunking(ctx context.Context, deps internal.Deps, _ []string) error {
	parentCfg := splitter.DefaultConfig()
	childCfg := splitter.DefaultConfig()
	childCfg.ChunkSize = 120
	childCfg.ChunkOverlap = 0

	pc := splitter.SplitParentChild(l2Sample, parentCfg, childCfg)
	slog.Info(fmt.Sprintf("[PARENT CHUNKING] → %d parents (size=%d)\n", len(pc.Parents), parentCfg.ChunkSize))
	fmt.Printf("[CHILD CHUNKING]  → %d children (size=%d)\n",
		len(pc.Children), childCfg.ChunkSize)

	if err := Ingest(ctx, deps.DB, deps.Embedder,
		infrastructure.Document{Title: "L2 sample", Lang: "zh"},
		pc.Parents, pc.Children,
	); err != nil {
		return fmt.Errorf("ingest: %w", err)
	}

	fmt.Println("[RETRIEVING] → top-3 children → expand to parents")
	chunks, err := Retrieve(ctx, deps.DB, deps.Embedder, demoQuestion, 3)
	if err != nil {
		return fmt.Errorf("retrieve: %w", err)
	}
	slog.Info(fmt.Sprintf("[RETRIEVED] %d parents:\n", len(chunks)))
	for i, c := range chunks {
		snippet := c.Content
		if r := []rune(snippet); len(r) > 80 {
			snippet = string(r[:80]) + "..."
		}
		slog.Info(fmt.Sprintf("  [%d] %s\n", i+1, snippet))
	}

	fmt.Println("[ANSWERING]")
	ans, err := Generate(ctx, deps.LLM, demoQuestion, chunks)
	if err != nil {
		return fmt.Errorf("generate: %w", err)
	}
	fmt.Println(ans)
	return nil
}
