package ch06

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"gorm.io/gorm"

	"rag/infrastructure"
	"rag/internal"
	"rag/internal/ch02/splitter"
	"rag/internal/ch03"
	"rag/internal/ch04"
	"rag/internal/ch05"
	"rag/sample"
)

func init() {
	internal.Register(internal.Lesson{
		Name:        "pipeline",
		Description: "L6: Pipeline RAG (可插拔流水线编排)",
		Migrate:     migratePipeline,
		Run:         runPipeline,
	})
}

// migratePipeline 与 L4/L5 同 schema:parent + child + dense + bm25,流水线只在 query 时跑。
func migratePipeline(ctx context.Context, deps internal.Deps) error {
	db := deps.DB
	return infrastructure.EnsureHybridSchema(ctx, db)
}

var l6Sample = sample.Handbook

const (
	demoQuestion6 = "RAG 的核心流程是什么?"
	recallTopN6   = 10 // chunk_search 宽召回(给 rerank 留余量)
	filterTopK6   = 5  // filter_top_k 截到 LLM 的最终 top-K
)

// l6History 替身 session history:跨 preset run 累积 Q&A,让 "rag_stream" 看到的 history 非空。
// 生产应放 session store (DB/Redis),不在包级 var;教学 demo 够用。
var l6History []string

// ---------- Plugin 实现 ----------

// LoadHistoryPlugin:从 l6History 读历史写入 cc.History(教学 demo 替身,生产对接 session store)。
type LoadHistoryPlugin struct{}

func (LoadHistoryPlugin) ActivationEvents() []EventType { return []EventType{LOAD_HISTORY} }

func (LoadHistoryPlugin) OnEvent(_ context.Context, _ EventType, cc *ChatContext, next func() error) error {
	if len(l6History) > 0 {
		cc.History = append([]string(nil), l6History...)
	}
	slog.Info(fmt.Sprintf("            history rounds: %d\n", len(cc.History)))
	return next()
}

// QueryUnderstandPlugin:用 ch05.RewriteQuery 做 query 改写(读 history + 原 query,写 cc.RewriteQuery)。
// 改写失败时降级为原 query(见 ch05.RewriteQuery 行为),不会让链断。
type QueryUnderstandPlugin struct {
	LLM infrastructure.LLM
}

func (*QueryUnderstandPlugin) ActivationEvents() []EventType { return []EventType{QUERY_UNDERSTAND} }

func (p *QueryUnderstandPlugin) OnEvent(ctx context.Context, _ EventType, cc *ChatContext, next func() error) error {
	rewritten, _ := ch05.RewriteQuery(ctx, p.LLM, cc.Query, cc.History)
	cc.RewriteQuery = rewritten
	if rewritten != cc.Query {
		slog.Info(fmt.Sprintf("            rewrite: %q → %q\n", cc.Query, rewritten))
	}
	return next()
}

// ChunkSearchPlugin:读 cc.RewriteQuery(否则 cc.Query),写 cc.Chunks(dense + BM25 + RRF)。
type ChunkSearchPlugin struct {
	DB       *gorm.DB
	Embedder infrastructure.Embedder
	TopK     int
}

func (p *ChunkSearchPlugin) ActivationEvents() []EventType { return []EventType{CHUNK_SEARCH} }

func (p *ChunkSearchPlugin) OnEvent(ctx context.Context, _ EventType, cc *ChatContext, next func() error) error {
	q := cc.RewriteQuery
	if q == "" {
		q = cc.Query
	}
	dense, err := ch03.DenseSearch(ctx, p.DB, p.Embedder, q, p.TopK)
	if err != nil {
		return fmt.Errorf("chunk_search dense: %w", err)
	}
	bm25, err := ch03.BM25Search(ctx, p.DB, q, p.TopK)
	if err != nil {
		cc.Chunks = dense
		return next()
	}
	cc.Chunks = ch03.RRF([][]ch03.Hit{dense, bm25}, 60)
	return next()
}

// RerankPlugin:读 cc.Chunks(用 rewrite 后的 query),写 cc.Reranked(按相关性降序,不去截 K)。
// 截 K 是下游 filter_top_k 的事,这样 "rag" / "rag_stream" preset 在这一段行为一致。
type RerankPlugin struct {
	Reranker ch04.Reranker
}

func (p *RerankPlugin) ActivationEvents() []EventType { return []EventType{CHUNK_RERANK} }

func (p *RerankPlugin) OnEvent(ctx context.Context, _ EventType, cc *ChatContext, next func() error) error {
	if len(cc.Chunks) == 0 {
		return next()
	}
	q := cc.RewriteQuery
	if q == "" {
		q = cc.Query
	}
	docs := make([]string, len(cc.Chunks))
	for i, h := range cc.Chunks {
		docs[i] = h.Content
	}
	ranked := p.Reranker.Rerank(ctx, q, docs)
	out := make([]ch03.Hit, 0, len(ranked))
	for _, r := range ranked {
		if r.Index < 0 || r.Index >= len(cc.Chunks) {
			continue
		}
		out = append(out, cc.Chunks[r.Index])
	}
	cc.Reranked = out
	return next()
}

// MergePlugin:读 cc.Reranked(否则 cc.Chunks),写 cc.Merged(去重)。
// 教学版没有父子 chunk,这步主要是 plugin 拆分点;生产里是父子拼接去重。
type MergePlugin struct{}

func (MergePlugin) ActivationEvents() []EventType { return []EventType{CHUNK_MERGE} }

func (MergePlugin) OnEvent(_ context.Context, _ EventType, cc *ChatContext, next func() error) error {
	src := cc.Reranked
	if len(src) == 0 {
		src = cc.Chunks
	}
	seen := make(map[int64]struct{}, len(src))
	out := make([]ch03.Hit, 0, len(src))
	for _, h := range src {
		if _, ok := seen[h.ChunkID]; ok {
			continue
		}
		seen[h.ChunkID] = struct{}{}
		out = append(out, h)
	}
	cc.Merged = out
	return next()
}

// FilterTopKPlugin:读 cc.Merged,截到 top-K 后写回 cc.Merged。"rag_stream" 才有这一步。
type FilterTopKPlugin struct{ K int }

func (FilterTopKPlugin) ActivationEvents() []EventType { return []EventType{FILTER_TOP_K} }

func (p FilterTopKPlugin) OnEvent(_ context.Context, _ EventType, cc *ChatContext, next func() error) error {
	before := len(cc.Merged)
	if p.K > 0 && before > p.K {
		cc.Merged = cc.Merged[:p.K]
	}
	slog.Info(fmt.Sprintf("            filter: %d → %d\n", before, len(cc.Merged)))
	return next()
}

// IntoChatMessagePlugin:读 cc.History + cc.Merged + cc.Query,写 cc.Prompt(参考资料 + history + 问题)。
type IntoChatMessagePlugin struct {
	DB *gorm.DB
}

func (*IntoChatMessagePlugin) ActivationEvents() []EventType { return []EventType{INTO_CHAT_MESSAGE} }

func (p *IntoChatMessagePlugin) OnEvent(ctx context.Context, _ EventType, cc *ChatContext, next func() error) error {
	chunks, err := ch03.LoadChunks(ctx, p.DB, cc.Merged)
	if err != nil {
		return fmt.Errorf("load chunks: %w", err)
	}
	var b strings.Builder
	for i, c := range chunks {
		fmt.Fprintf(&b, "[%d] %s\n", i+1, c.Content)
	}
	var histStr string
	if len(cc.History) > 0 {
		histStr = "对话历史:\n" + strings.Join(cc.History, "\n") + "\n\n"
	}
	cc.Prompt = fmt.Sprintf(promptTpl, histStr, b.String(), cc.Query)
	return next()
}

// ChatCompletionPlugin:读 cc.Prompt,写 cc.Answer。
type ChatCompletionPlugin struct {
	LLM infrastructure.LLM
}

func (*ChatCompletionPlugin) ActivationEvents() []EventType { return []EventType{CHAT_COMPLETION} }

func (p *ChatCompletionPlugin) OnEvent(ctx context.Context, _ EventType, cc *ChatContext, next func() error) error {
	out, err := p.LLM.Complete(ctx, cc.Prompt)
	if err != nil {
		return fmt.Errorf("chat completion: %w", err)
	}
	cc.Answer = out
	return next()
}

const promptTpl = `基于以下参考资料回答问题。若参考资料不足以回答,请回答"我不知道"。

%s参考资料:
%s

问题:%s
答案:`

// ---------- 入口 ----------

// runPipeline 索引一次语料,跑两个 preset:history 在两次之间累积,让 "rag_stream" 真正用上 history + rewrite + filter_top_k。
func runPipeline(ctx context.Context, deps internal.Deps, _ []string) error {
	parentCfg := splitter.DefaultConfig()
	childCfg := splitter.DefaultConfig()
	childCfg.ChunkSize = 120
	childCfg.ChunkOverlap = 0

	pc := splitter.SplitParentChild(l6Sample, parentCfg, childCfg)
	fmt.Printf("[INDEXING] dense + bm25 → %d chunks (across %d parents)\n",
		len(pc.Children), len(pc.Parents))

	if err := ch03.Ingest(ctx, deps.DB, deps.Embedder,
		infrastructure.Document{Title: "L6 sample", Lang: "zh"},
		pc.Parents, pc.Children,
	); err != nil {
		return fmt.Errorf("ingest: %w", err)
	}

	// 共享 plugin 实例:两个 preset 都注册它们,只是 preset 顺序 + 是否触发不同。
	em := NewEventManager()
	em.Register(LoadHistoryPlugin{})
	em.Register(&QueryUnderstandPlugin{LLM: deps.LLM})
	em.Register(&ChunkSearchPlugin{DB: deps.DB, Embedder: deps.Embedder, TopK: recallTopN6})
	em.Register(&RerankPlugin{Reranker: ch04.NewLLMReranker(deps.LLM)})
	em.Register(MergePlugin{})
	em.Register(FilterTopKPlugin{K: filterTopK6})
	em.Register(&IntoChatMessagePlugin{DB: deps.DB})
	em.Register(&ChatCompletionPlugin{LLM: deps.LLM})

	printPresets()
	for _, preset := range []string{"rag", "rag_stream"} {
		cc := &ChatContext{Query: demoQuestion6}
		slog.Info(fmt.Sprintf("\n[RUN] preset=%q query=%q\n", preset, demoQuestion6))
		if err := em.TriggerPreset(ctx, preset, cc); err != nil {
			return fmt.Errorf("preset %s: %w", preset, err)
		}
		fmt.Println("\n[ANSWER]")
		fmt.Println(cc.Answer)
		// 写入 history,让下一个 preset 看到
		l6History = append(l6History, fmt.Sprintf("Q: %s\nA: %s", cc.Query, cc.Answer))
		if len(l6History) > 6 {
			l6History = l6History[len(l6History)-6:]
		}
	}
	return nil
}

func printPresets() {
	for _, name := range []string{"rag", "rag_stream"} {
		parts := make([]string, len(Pipeline[name]))
		for i, et := range Pipeline[name] {
			parts[i] = string(et)
		}
		slog.Info(fmt.Sprintf("[PRESET] %-12q: %s\n", name, strings.Join(parts, " → ")))
	}
}
