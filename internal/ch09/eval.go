package ch09

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"gorm.io/gorm"

	"rag/infrastructure"
	"rag/internal"
	"rag/internal/ch02/splitter"
	"rag/internal/ch03"
	"rag/internal/ch08neo4j"
	"rag/internal/ragcore"
	"rag/sample"
)

func init() {
	internal.Register(internal.Lesson{
		Name:        "eval",
		Description: "L9: Eval RAG (离线评测 + 在线追踪)",
		Migrate:     migrateEval,
		Run:         runEval,
	})
}

var (
	l9Sample = sample.BlueWhale
)

const (
	topK9 = 5

	// Pycharm 项目的 RAGAS 数据集目录;Go 端写入,Python 端 read。
	defaultRAGASDatasetDir = "/Users/resplendid/PycharmProjects/rag/ragas_results/datasets"
	l9RealJSONL            = "l9_real.jsonl"
)

// l9Questions 只留 1 题:同时需要"实体细节 + 多跳关系 + 时间线节点"的问题,
// 三个变体差异最明显:
//
//	l1_dense:     只召回含"马丁内斯"的 top-5,能答细节,缺全局节点
//	l3_hybrid:    dense+BM25 多召回 1-2 个,跟 l1 差异不大
//	l8neo4j_local: 实体直接命中 → 1 跳拿到教练-球员关系 → 同时拿到"4-3-3 / 8 连胜 / 客胜海港"全局节点
var l9Questions = []struct {
	ID         string
	Question   string
	GroundHint string
}{
	{"q_drift", "马丁内斯的战术调整和赛季关键节点有哪些?",
		"4-3-3 → 4-2-3-1;8 连胜 / 客胜海港 / 变阵后连胜稳第三。"},
}

// EvalSample 是导到 RAGAS 的单条样本,字段名跟 ragas Dataset 对齐。
type EvalSample struct {
	UserInput         string   `json:"user_input"`
	Response          string   `json:"response"`
	RetrievedContexts []string `json:"retrieved_contexts"`
	Reference         string   `json:"reference,omitempty"`
	Variant           string   `json:"variant"`
	QuestionID        string   `json:"question_id"`
}

func runEval(ctx context.Context, deps internal.Deps, _ []string) error {
	// ch09 跟 ch08neo4j 共用 l_neo4j schema:documents/chunks + Neo4j pg_id 都在同 schema 下,不会错位。
	// 切 search_path 后,SQL 走 l_neo4j 的 documents 表;落 eval 表时再切回 l_eval。
	if err := deps.DB.WithContext(ctx).Exec(`SET LOCAL search_path TO l_neo4j, public`).Error; err != nil {
		return fmt.Errorf("set search_path l_neo4j: %w", err)
	}
	// ch08neo4j 的 BaseSchemaSQL 不带 BM25,ch03.BM25Search 走 l3_hybrid 时需要;补建 IF NOT EXISTS。
	if err := deps.DB.WithContext(ctx).Exec(`
		CREATE INDEX IF NOT EXISTS document_chunks_bm25_idx
		ON document_chunks USING bm25 (id, content) WITH (key_field='id')
	`).Error; err != nil {
		return fmt.Errorf("create bm25 idx: %w", err)
	}

	// 1. 确保语料 + Neo4j 图已建(用 ch08neo4j 的 skip-if-exists,多次跑不重复)。
	if err := ensureIndexed(ctx, deps); err != nil {
		return fmt.Errorf("ensure indexed: %w", err)
	}

	// 2. 跑 3 变体 × 4 question。
	// 三个变体对应 L1 / L3 / L8(neo4j)三种能力通道,清晰对比每加一层增强的真实收益。
	variants := []string{"l1_dense", "l3_hybrid", "l8neo4j_local"}
	samples := make([]EvalSample, 0, len(variants)*len(l9Questions))
	for _, v := range variants {
		fmt.Printf("\n[VARIANT] %s\n", v)
		for _, q := range l9Questions {
			fmt.Printf("  [Q %s] %q\n", q.ID, q.Question)
			s, err := runVariant(ctx, deps, v, q)
			if err != nil {
				slog.Info(fmt.Sprintf("    failed: %v", err))
				continue
			}
			samples = append(samples, s)
		}
	}

	// 3. 导出 RAGAS JSONL(供 Pycharm 端 RAGAS 评测)。
	outPath, err := writeRAGASJSONL(samples)
	if err != nil {
		return fmt.Errorf("write jsonl: %w", err)
	}
	fmt.Printf("\n[RAGAS INPUT] %d samples → %s\n", len(samples), outPath)

	// 4. 切回 l_eval 落 eval_datasets/eval_runs。
	if err := deps.DB.WithContext(ctx).Exec(`SET LOCAL search_path TO l_eval, public`).Error; err != nil {
		return fmt.Errorf("set search_path l_eval: %w", err)
	}
	configHash := computeConfigHash(deps)
	meta := map[string]any{
		"embedding": deps.Cfg.Embedding.Model,
		"llm":       deps.Cfg.LLM.Model,
		"top_k":     topK9,
		"variants":  variants,
		"questions": len(l9Questions),
	}
	datasetID, err := ensureDataset(ctx, deps.DB, "bluewhale-l1-l3-l8neo4j", configHash, meta)
	if err != nil {
		return fmt.Errorf("ensure dataset: %w", err)
	}
	runID, err := createRun(ctx, deps.DB, datasetID)
	if err != nil {
		return fmt.Errorf("create run: %w", err)
	}
	fmt.Printf("\n[EVAL DB] dataset_id=%d run_id=%d (awaiting Python RAGAS results)\n", datasetID, runID)
	return nil
}

// ensureIndexed 调 ch08neo4j 既有 skip-if-exists 流程:documents=0 ingest,Neo4j entities=0 extract。
func ensureIndexed(ctx context.Context, deps internal.Deps) error {
	if deps.Neo4j == nil {
		return fmt.Errorf("neo4j driver is nil; set neo4j.uri in config.yaml")
	}
	var n int64
	if err := deps.DB.WithContext(ctx).Raw(`SELECT COUNT(*) FROM documents`).Scan(&n).Error; err != nil {
		return fmt.Errorf("count docs: %w", err)
	}
	if n == 0 {
		parentCfg := splitter.DefaultConfig()
		childCfg := splitter.DefaultConfig()
		childCfg.ChunkSize = 200
		childCfg.ChunkOverlap = 0
		pc := splitter.SplitParentChild(l9Sample, parentCfg, childCfg)
		fmt.Printf("[INDEXING] parent + child → %d chunks (across %d parents)\n",
			len(pc.Children), len(pc.Parents))
		if err := ch03.Ingest(ctx, deps.DB, deps.Embedder,
			infrastructure.Document{Title: "L9 sample", Lang: "zh"},
			pc.Parents, pc.Children,
		); err != nil {
			return fmt.Errorf("ingest: %w", err)
		}
	}
	// ch09 跟 ch08neo4j 共享 l_neo4j schema + Neo4j 图谱。
	// 强制每次 reset + rebuild,保证 pg_id 跟当前 document_chunks.id 严格一致(skip-if-exists
	// 在跨 lesson 共享图时容易因 chunk id 漂移导致 LocalSearch 永远 seeds=0)。
	if haveEnts, err := ch08neo4j.CountEntities(ctx, deps.Neo4j); err == nil && haveEnts > 0 {
		fmt.Printf("[EXTRACT] reset Neo4j graph (%d entities)\n", haveEnts)
		if err := ch08neo4j.ResetGraph(ctx, deps.Neo4j); err != nil {
			return fmt.Errorf("reset neo4j: %w", err)
		}
	}
	children, err := loadChildChunks(ctx, deps.DB)
	if err != nil {
		return fmt.Errorf("load children: %w", err)
	}
	fmt.Printf("[EXTRACT] %d chunks → LLM (Neo4j)\n", len(children))
	if err := ch08neo4j.ExtractFromChunks(ctx, deps.DB, deps.Neo4j, deps.LLM, children); err != nil {
		return fmt.Errorf("extract: %w", err)
	}
	return nil
}

// loadChildChunks 从 document_chunks 表读全量 child chunk content(kg 抽取时只需要 content,
// id 由 ch08neo4j.extractFromChunks 内部用 LoadChildChunkIDs 重新拉一次跟 length 对齐)。
func loadChildChunks(ctx context.Context, db *gorm.DB) ([]splitter.ChildChunk, error) {
	rows, err := db.WithContext(ctx).Raw(
		`SELECT content FROM document_chunks ORDER BY id`).Rows()
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []splitter.ChildChunk
	seq := 0
	for rows.Next() {
		var content string
		if err := rows.Scan(&content); err != nil {
			return nil, err
		}
		out = append(out, splitter.ChildChunk{
			Chunk: splitter.Chunk{Content: content, Seq: seq},
		})
		seq++
	}
	return out, rows.Err()
}

// runVariant 调一个变体,返回 RAGAS 样本。
func runVariant(ctx context.Context, deps internal.Deps, variant string, q struct {
	ID, Question, GroundHint string
}) (EvalSample, error) {
	var (
		resp     string
		contexts []string
		err      error
	)
	switch variant {
	case "l1_dense":
		resp, contexts, err = variantL1(ctx, deps, q.Question)
	case "l3_hybrid":
		resp, contexts, err = variantL3(ctx, deps, q.Question)
	case "l8neo4j_local":
		resp, contexts, err = variantL8Neo4j(ctx, deps, q.Question)
	default:
		return EvalSample{}, fmt.Errorf("unknown variant: %s", variant)
	}
	if err != nil {
		return EvalSample{}, err
	}
	return EvalSample{
		UserInput:         q.Question,
		Response:          resp,
		RetrievedContexts: contexts,
		Reference:         q.GroundHint,
		Variant:           variant,
		QuestionID:        q.ID,
	}, nil
}

// variantL1 = L1 朴素 dense + 直答(走 ch03.LoadChunks + ch03.Generate,语义近似 baseline)。
func variantL1(ctx context.Context, deps internal.Deps, q string) (string, []string, error) {
	hits, err := ch03.DenseSearch(ctx, deps.DB, deps.Embedder, q, topK9)
	if err != nil {
		return "", nil, err
	}
	chunks, err := ch03.LoadChunks(ctx, deps.DB, hits)
	if err != nil {
		return "", nil, err
	}
	ans, err := ch03.Generate(ctx, deps.LLM, q, chunks)
	return ans, chunkContents(chunks), err
}

// variantL3 = dense + bm25 + RRF 融合,再 LLM 答。
func variantL3(ctx context.Context, deps internal.Deps, q string) (string, []string, error) {
	dense, err := ch03.DenseSearch(ctx, deps.DB, deps.Embedder, q, topK9)
	if err != nil {
		return "", nil, err
	}
	bm25, err := ch03.BM25Search(ctx, deps.DB, q, topK9)
	if err != nil {
		return "", nil, err
	}
	fused := ch03.RRF([][]ch03.Hit{dense, bm25}, 60)
	if len(fused) > topK9 {
		fused = fused[:topK9]
	}
	chunks, err := ch03.LoadChunks(ctx, deps.DB, fused)
	if err != nil {
		return "", nil, err
	}
	ans, err := ch03.Generate(ctx, deps.LLM, q, chunks)
	return ans, chunkContents(chunks), err
}

// variantL8Neo4j = ch08neo4j.LocalSearch 抓 answer;contexts 用 ch08neo4j.LocalContexts 抓
// (等价 LocalSearch 的 retrieve 步骤但跳过 LLM,省时给 RAGAS 评测)。
func variantL8Neo4j(ctx context.Context, deps internal.Deps, q string) (string, []string, error) {
	ans, err := ch08neo4j.LocalSearch(ctx, deps.DB, deps.Neo4j, deps.Embedder, deps.LLM, q)
	if err != nil {
		return "", nil, err
	}
	contexts, err := ch08neo4j.LocalContexts(ctx, deps.DB, deps.Neo4j, deps.Embedder, q, topK9)
	return ans, contexts, err
}

func chunkContents(chunks []splitter.Chunk) []string {
	out := make([]string, len(chunks))
	for i, c := range chunks {
		out[i] = c.Content
	}
	return out
}

// writeRAGASJSONL 写到 Pycharm 项目的 RAGAS datasets 目录,确保目录存在。
func writeRAGASJSONL(samples []EvalSample) (string, error) {
	if err := os.MkdirAll(defaultRAGASDatasetDir, 0o755); err != nil {
		return "", err
	}
	outPath := filepath.Join(defaultRAGASDatasetDir, l9RealJSONL)
	f, err := os.Create(outPath)
	if err != nil {
		return "", err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetEscapeHTML(false)
	for _, s := range samples {
		// 落盘前剥 think 块:DeepSeek-R1 等会在 response 里夹 推理过程,RAGAS 看 response 时只要结论。
		s.Response = ragcore.StripThink(s.Response)
		if err := enc.Encode(s); err != nil {
			return "", err
		}
	}
	return outPath, nil
}

// computeConfigHash 用 embedding model + llm model + topk 算个指纹,eval_datasets 去重用。
func computeConfigHash(deps internal.Deps) string {
	parts := []string{
		"emb=" + deps.Cfg.Embedding.Model,
		"llm=" + deps.Cfg.LLM.Model,
		fmt.Sprintf("topk=%d", topK9),
	}
	return shortHash(strings.Join(parts, "|"))
}

func shortHash(s string) string {
	const offset, prime uint32 = 2166136261, 16777619
	h := offset
	for i := 0; i < len(s); i++ {
		h ^= uint32(s[i])
		h *= prime
	}
	return fmt.Sprintf("%08x", h)
}
