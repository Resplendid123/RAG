package ch05

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
	"rag/internal/ragcore"
	"rag/sample"
)

func init() {
	internal.Register(internal.Lesson{
		Name:        "query",
		Description: "L5: Query RAG (查询理解与改写)",
		Migrate:     migrateQuery,
		Run:         runQuery,
	})
}

// migrateQuery 与 L3/L4 同 schema:child + parent + dense + bm25,query 改造在 query 时跑。
func migrateQuery(ctx context.Context, deps internal.Deps) error {
	db := deps.DB
	return infrastructure.EnsureHybridSchema(ctx, db)
}

var l5Sample = sample.Handbook

const (
	demoQuestion5 = "rag 怎么搞" // 口语化,与语料表述错位,便于看 query 改造价值
	topN5         = 5         // 召回宽度
)

// runQuery 演示 4 种 query 改造手段:baseline + rewrite + multi-q + hyde + step-back,
// 各自走 hybrid(dense+bm25+RRF)召回,top-K 一并打出便于横向对比。
func runQuery(ctx context.Context, deps internal.Deps, _ []string) error {
	parentCfg := splitter.DefaultConfig()
	childCfg := splitter.DefaultConfig()
	childCfg.ChunkSize = 120
	childCfg.ChunkOverlap = 0

	pc := splitter.SplitParentChild(l5Sample, parentCfg, childCfg)
	fmt.Printf("[INDEXING] dense + bm25 → %d chunks (across %d parents)\n",
		len(pc.Children), len(pc.Parents))

	if err := ch03.Ingest(ctx, deps.DB, deps.Embedder,
		infrastructure.Document{Title: "L5 sample", Lang: "zh"},
		pc.Parents, pc.Children,
	); err != nil {
		return fmt.Errorf("ingest: %w", err)
	}

	slog.Info(fmt.Sprintf("\n[QUERY] 原始: %q\n\n", demoQuestion5))

	// baseline:原 query 直接 hybrid 检索。
	baseline, err := hybridTopN(ctx, deps.DB, deps.Embedder, demoQuestion5, topN5)
	if err != nil {
		return fmt.Errorf("baseline retrieve: %w", err)
	}
	slog.Info(fmt.Sprintf("[BASELINE   ] top-%d chunk ids: %s\n", len(baseline), ragcore.IDsOf(baseline)))

	// 1) Rewriting:LLM 单次改写 → hybrid。
	rewritten, err := RewriteQuery(ctx, deps.LLM, demoQuestion5, nil)
	if err != nil {
		slog.Info(fmt.Sprintf("[REWRITE   ] failed (%v), skip\n", err))
	}
	slog.Info(fmt.Sprintf("[REWRITE    ] %q\n", rewritten))
	rewHits, err := hybridTopN(ctx, deps.DB, deps.Embedder, rewritten, topN5)
	if err != nil {
		return fmt.Errorf("rewrite retrieve: %w", err)
	}
	slog.Info(fmt.Sprintf("             top-%d chunk ids: %s\n", len(rewHits), ragcore.IDsOf(rewHits)))

	// 2) Multi-Query:原 query + N 个变体,各走 hybrid,RRF 融合后取 top-K。
	variants := MultiQueryVariants(ctx, deps.LLM, demoQuestion5, 3)
	slog.Info(fmt.Sprintf("[MULTI-Q    ] variants: %v\n", variants))
	var lists [][]ch03.Hit
	lists = append(lists, baseline)
	for _, v := range variants {
		h, err := hybridTopN(ctx, deps.DB, deps.Embedder, v, topN5*2)
		if err != nil {
			return fmt.Errorf("multi-q retrieve %q: %w", v, err)
		}
		lists = append(lists, h)
	}
	multiFused := ch03.RRF(lists, 60)
	if len(multiFused) > topN5 {
		multiFused = multiFused[:topN5]
	}
	slog.Info(fmt.Sprintf("             top-%d chunk ids: %s\n\n", len(multiFused), ragcore.IDsOf(multiFused)))

	// 3) HyDE:用 LLM 生成的假设答案去 dense 检索,bm25 跳过(假设答案常常 BM25 跑偏)。
	hypo, err := HyDEAnswer(ctx, deps.LLM, demoQuestion5,
		"使用正式书面语,陈述句为主,200 字以内")
	if err != nil {
		slog.Info(fmt.Sprintf("[HYDE       ] failed (%v), skip\n", err))
	}
	slog.Info(fmt.Sprintf("[HYDE       ] hypo: %s\n", snippet(hypo, 80)))
	hydeHits, err := ch03.DenseSearch(ctx, deps.DB, deps.Embedder, hypo, topN5)
	if err != nil {
		return fmt.Errorf("hyde retrieve: %w", err)
	}
	slog.Info(fmt.Sprintf("             top-%d chunk ids: %s\n", len(hydeHits), ragcore.IDsOf(hydeHits)))

	// 4) Step-Back:抽象问题 + 原问题双路,dense + bm25 RRF 后各取 5,合在一起给 StepBackResult。
	sb, err := StepBack(ctx, deps.LLM, demoQuestion5)
	if err != nil {
		slog.Info(fmt.Sprintf("[STEP-BACK  ] failed (%v), skip\n", err))
	}
	slog.Info(fmt.Sprintf("[STEP-BACK  ] %q\n", sb))
	sbOrig, err := hybridTopN(ctx, deps.DB, deps.Embedder, demoQuestion5, topN5)
	if err != nil {
		return fmt.Errorf("stepback orig retrieve: %w", err)
	}
	sbAbs, err := hybridTopN(ctx, deps.DB, deps.Embedder, sb, topN5)
	if err != nil {
		return fmt.Errorf("stepback abs retrieve: %w", err)
	}
	slog.Info(fmt.Sprintf("             orig top-%d: %s\n", len(sbOrig), ragcore.IDsOf(sbOrig)))
	slog.Info(fmt.Sprintf("             abs  top-%d: %s\n", len(sbAbs), ragcore.IDsOf(sbAbs)))

	// 展示 StepBackResult 拼 context 的样子(给下游 Generate 用)。
	origCtx, _ := ch03.LoadChunks(ctx, deps.DB, sbOrig)
	absCtx, _ := ch03.LoadChunks(ctx, deps.DB, sbAbs)
	slog.Info(fmt.Sprintf("\n[STEP-BACK RESULT] originalCtx = %s...\n", snippet(joinContents(origCtx), 60)))
	slog.Info(fmt.Sprintf("                  stepBackCtx = %s...\n", snippet(joinContents(absCtx), 60)))

	// 最终生成:step-back 双路 context 一起喂 LLM,作为 L5 端到端示例。
	fmt.Println("\n[ANSWERING] step-back dual-context → LLM")
	ans, err := generateStepBack(ctx, deps.LLM, demoQuestion5, sb, origCtx, absCtx)
	if err != nil {
		return fmt.Errorf("generate: %w", err)
	}
	fmt.Println(ans)
	return nil
}

// hybridTopN dense + bm25 + RRF 取 topN;共用检索逻辑。
// BM25 走 paradedb/Lucene 语法,query 里若有 `: ( )` 等保留字符会解析失败,此时降级为 dense-only。
func hybridTopN(ctx context.Context, db *gorm.DB, emb infrastructure.Embedder, q string, topN int) ([]ch03.Hit, error) {
	dense, err := ch03.DenseSearch(ctx, db, emb, q, topN)
	if err != nil {
		return nil, err
	}
	bm25, err := ch03.BM25Search(ctx, db, q, topN)
	if err != nil {
		slog.Info(fmt.Sprintf("[WARN] "+"bm25 query parse failed for %q, dense-only fallback"+"\n", q))
		return dense, nil
	}
	fused := ch03.RRF([][]ch03.Hit{dense, bm25}, 60)
	if len(fused) > topN {
		fused = fused[:topN]
	}
	return fused, nil
}

func denseTopN(ctx context.Context, db *gorm.DB, emb infrastructure.Embedder, q string, topN int) ([]ch03.Hit, error) {
	return ch03.DenseSearch(ctx, db, emb, q, topN)
}

const stepBackPromptTpl = `基于以下两类参考资料回答用户问题:第一类是上层抽象资料,第二类是具体细节资料。
上层抽象问题是:%s,具体问题是:%s。
若参考资料不足以回答,请回答"我不知道"。

上层抽象资料:
%s

具体细节资料:
%s

答案:`

// generateStepBack step-back 专用 Generate:同时喂上层抽象 + 具体细节,引导 LLM 先对齐概念再补细节。
func generateStepBack(ctx context.Context, llm infrastructure.LLM, q, sb string, origCtx, absCtx []splitter.Chunk) (string, error) {
	absStr := joinContents(absCtx)
	origStr := joinContents(origCtx)
	return llm.Complete(ctx, fmt.Sprintf(stepBackPromptTpl, sb, q, absStr, origStr))
}

func snippet(s string, n int) string {
	if r := []rune(s); len(r) > n {
		return string(r[:n]) + "..."
	}
	return s
}

func joinContents(chunks []splitter.Chunk) string {
	parts := make([]string, len(chunks))
	for i, c := range chunks {
		parts[i] = c.Content
	}
	return strings.Join(parts, " | ")
}
