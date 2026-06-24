package ch07

import (
	"context"
	"fmt"
	"strings"
	"time"

	"gorm.io/gorm"

	"rag/internal"
	"rag/internal/agent"
	"rag/internal/ch02"
	"rag/internal/ch02/splitter"
	"rag/internal/ch03"
	"rag/sample"
)

func init() {
	internal.Register(internal.Lesson{
		Name:        "agentic",
		Description: "L7: Agentic RAG (工具调用与 self-critique)",
		Migrate:     migrateAgentic,
		Run:         runAgentic,
	})
}

// migrateAgentic 与 L4/L5/L6 同 schema:parent + child + dense + bm25,Agent 只在 query 时跑。
func migrateAgentic(ctx context.Context, db *gorm.DB) error {
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

var l7Sample = sample.Handbook

// demoQuestion7s 每个问题触发不同 tool,看 LLM 选对没。
//
//	LLM 选错的话应该看 [RUN] 输出的 tool 名 + 最终答案。
var demoQuestion7s = []string{
	"rag 怎么搞",                      // → search_documents
	"现在表里有多少个 chunk?",              // → sql_query
	"Python 3.13 有什么新特性?简要说 2 点即可", // → webfetch (LLM 已知 URL,直接抓)
	"现在最热门的 RAG 研究方向是什么?",          // → web_search (LLM 不知道 URL,先搜)
}

// runAgentic 索引一次语料,跑 3 个 demo question 看 LLM 选对 tool 没。
func runAgentic(ctx context.Context, deps internal.Deps, _ []string) error {
	parentCfg := splitter.DefaultConfig()
	childCfg := splitter.DefaultConfig()
	childCfg.ChunkSize = 120
	childCfg.ChunkOverlap = 0

	pc := ch02.SplitParentChild(l7Sample, parentCfg, childCfg)
	fmt.Printf("[INDEXING] dense + bm25 → %d chunks (across %d parents)\n",
		len(pc.Children), len(pc.Parents))

	if err := ch03.Ingest(ctx, deps.DB, deps.Embedder,
		ch03.Document{Title: "L7 sample", Lang: "zh"},
		pc.Parents, pc.Children,
	); err != nil {
		return fmt.Errorf("ingest: %w", err)
	}

	r := agent.NewRegistry()
	registerTools(r, deps.DB, deps.Embedder, deps.Cfg.WebSearch.TavilyAPIKey)

	fmt.Printf("\n[TOOLS] registered: %s\n", registeredNames(r))

	system := `你是企业知识助手。可以调 4 个工具:
- search_documents:检索企业知识库,返回 top-K chunk
- sql_query:对 document_chunks 跑只读 SQL(SELECT / WITH)
- webfetch:抓公网 URL 纯文本(已知具体 URL 时用)
- web_search:公网搜索引擎(不知道 URL 时先用这个找)

规则:能搜库就别抓公网;能 count/select 就别搜;知道 URL 用 webfetch,不知道先用 web_search 再 webfetch。
回答中文,先给结论再展开。`

	for i, q := range demoQuestion7s {
		fmt.Printf("\n[RUN %d] %q\n", i+1, q)
		budget := agent.NewBudget(8, 60*time.Second, map[string]int{
			"search_documents": 3,
			"sql_query":        3,
			"webfetch":         2,
			"web_search":       2,
		})
		a := &agent.Agent{
			LLM:     deps.LLM,
			Tools:   r,
			MaxIter: 5,
			Budget:  budget,
			System:  system,
		}
		ans, err := a.Run(ctx, q)
		if err != nil {
			fmt.Printf("[ERROR] %v\n", err)
			continue
		}
		fmt.Printf("[ANSWER]\n%s\n", snippet7(ans, 600))
	}
	return nil
}

func registeredNames(r *agent.Registry) string {
	specs := r.Specs()
	names := make([]string, 0, len(specs))
	for _, t := range specs {
		names = append(names, t.Name)
	}
	return strings.Join(names, ", ")
}

func snippet7(s string, n int) string {
	s = strings.TrimSpace(s)
	r := []rune(s)
	if len(r) > n {
		return string(r[:n]) + "..."
	}
	return s
}
