package rag

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"strings"

	"github.com/pgvector/pgvector-go"
	"gorm.io/gorm"

	"rag/internal"
)

func init() {
	Register(Lesson{
		Name:        "naive",
		Description: "L1: Naive rag baseline",
		Migrate:     migrateNaive,
		Run:         runNaive,
	})
}

func migrateNaive(ctx context.Context, db *gorm.DB) error {
	return db.WithContext(ctx).Exec(`
        CREATE TABLE IF NOT EXISTS documents (
            id          BIGSERIAL PRIMARY KEY,
            title       TEXT NOT NULL,
            source_url  TEXT,
            lang        TEXT DEFAULT 'zh',
            content_hash TEXT,
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

// l1Sample 是 L1 演示用样本。
const l1Sample = `Retrieval-Augmented Generation（RAG）是一种把"检索"和"生成"结合的大模型应用架构，旨在解决大语言模型（LLM）在知识更新、事实准确性和可解释性方面的核心痛点。该概念最早由 Meta AI 研究团队在 2020 年的论文《Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks》中正式提出，并迅速成为大模型落地的关键技术范式。

RAG 的提出源于一个朴素的观察：尽管 LLM 在预训练阶段从海量文本中学习了大量知识，但这些知识是"冻结"在模型权重中的，无法动态更新，也无法追溯来源。当用户询问特定领域、最新事件或小众知识时，模型要么无法回答，要么产生"幻觉"（Hallucination）——即生成看似合理但事实上错误的陈述。RAG 通过在生成过程中引入外部知识库的实时检索，有效缓解了这一问题。

RAG 的核心流程分为两大阶段：索引（Indexing）和检索生成（Retrieval-Generation）。

在索引阶段，系统首先对原始文档进行预处理，包括格式清洗、元数据提取等。然后将长文档切分成语义完整的文本块（Chunk），通常每个块包含 200-500 个字符，且块与块之间保留一定的重叠（Overlap），以避免语义断裂。接着，使用嵌入模型（Embedding Model）将每个文本块转换为固定维度的稠密向量（Dense Vector），并存入向量数据库（Vector Database），如 Chroma、Milvus、Qdrant 或 PGVector 等。向量数据库为这些向量建立高效索引（如 HNSW、IVF），以便后续快速检索。

在检索生成阶段，当用户输入一个问题时，系统使用同一个嵌入模型将问题也转化为向量，然后在向量数据库中执行相似度搜索（通常使用余弦距离、L2 距离或内积），找出与问题向量最相似的前 K 个文本块（Top-K）。这些检索到的文本块作为"证据"或"上下文"，连同用户的原始问题一起，按照特定的 Prompt 模板（如 "基于以下参考资料回答问题..."）拼接成一个完整的提示，最后交由 LLM 生成最终答案。LLM 在生成时会参考这些参考资料，从而给出更准确、更有依据的回答。

RAG 架构的优势非常明显。首先，它显著降低了 LLM 的幻觉现象，因为生成答案时有了事实依据。其次，知识库可以独立更新，新增文档只需重新索引，无需重新训练或微调大模型，极大降低了维护成本。第三，检索到的文本块可以作为答案的引用来源，提高了系统的可解释性和用户信任度。第四，RAG 适用于各种垂直领域，如法律文书检索、医疗知识问答、企业内部文档助手、在线客服等，具有很强的通用性和可定制性。

然而，RAG 也存在一些代价和挑战。最核心的问题是"检索质量决定了回答的上限"——如果检索阶段无法召回最相关的文档块，那么即便 LLM 再强大，也无法生成正确的答案，正所谓"垃圾进，垃圾出"（Garbage In, Garbage Out）。因此，检索阶段的召回率（Recall）和精度（Precision）直接决定了整个系统的表现上限。此外，RAG 还面临文本分块策略的选择、嵌入模型的效果、向量数据库的性能调优、上下文窗口长度限制、多轮对话中的记忆管理等工程挑战。同时，如果知识库中存在错误或过时的信息，LLM 也可能被误导，因此知识库的质量管理也是关键环节。

近年来，RAG 技术不断演进，衍生出了众多高级变体，如 Self-RAG（引入自我反思机制）、Corrective RAG（引入纠正机制）、Agentic RAG（结合 Agent 工具调用）、Graph RAG（基于知识图谱增强检索）、Adaptive RAG（根据问题动态选择检索策略）、RAPTOR（递归摘要树检索）等，进一步提升了系统的智能性和鲁棒性。

总的来说，RAG 已成为大模型时代连接"静态知识"与"动态需求"的桥梁，是当前企业级 AI 应用中最成熟、最广泛落地的技术方案之一。`

type Document struct {
	ID        int64
	Title     string
	SourceURL string
	Lang      string
}

type Chunk struct {
	ID      int64
	DocID   int64
	Index   int
	Content string
}

func runNaive(ctx context.Context, deps Deps, args []string) error {
	fs := flag.NewFlagSet("naive", flag.ContinueOnError)
	i := fs.String("i", "", "text to ingest")
	q := fs.String("q", "", "user question")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *i == "" && *q == "" {
		return fmt.Errorf("--i <text> or --q <question> is required")
	}

	text := *i
	if text == "" {
		text = l1Sample
	}
	chunksIn := ChunkText(text, 200, 30)
	fmt.Printf("[CHUNKING] → %d chunks (size=200, overlap=30)\n", len(chunksIn))
	if err := Ingest(ctx, deps.DB, deps.Embedder,
		Document{Title: "L1 sample", Lang: "zh"},
		chunksIn,
	); err != nil {
		return fmt.Errorf("ingest: %w", err)
	}

	if *q != "" {
		fmt.Println("[RETRIEVING] → top-3 from l_naive.document_chunks")
		chunks, err := Retrieve(ctx, deps.DB, deps.Embedder, *q, 3)
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
		ans, err := Answer(ctx, deps.LLM, *q, chunks)
		if err != nil {
			return fmt.Errorf("answer: %w", err)
		}
		fmt.Println(ans)
	}
	return nil
}

// ChunkText 按固定长度硬切 + 重叠。
func ChunkText(text string, size, overlap int) []string {
	if size <= 0 {
		return nil
	}
	if overlap >= size {
		overlap = size / 2
	}
	runes := []rune(text)
	var out []string
	for i := 0; i < len(runes); i += size - overlap {
		end := min(i+size, len(runes))
		out = append(out, string(runes[i:end]))
		if end == len(runes) {
			break
		}
	}
	return out
}

// Ingest 把 doc + chunks 写入 documents / document_chunks。批 embedding，单事务。
// 同一份 chunks 第二次写入会被 content_hash 唯一约束拦下，直接跳过。
func Ingest(ctx context.Context, db *gorm.DB, emb internal.Embedder, doc Document, chunks []string) error {
	if len(chunks) == 0 {
		return nil
	}
	sum := sha256.Sum256([]byte(strings.Join(chunks, "\n")))
	hash := hex.EncodeToString(sum[:])

	fmt.Printf("[EMBEDDING] → %d vectors\n", len(chunks))
	vecs, err := emb.Embed(ctx, chunks)
	if err != nil {
		return err
	}
	return db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var docID int64
		err := tx.Raw(
			`INSERT INTO documents(title, source_url, lang, content_hash) VALUES (?,?,?,?)
             ON CONFLICT (content_hash) DO NOTHING
             RETURNING id`,
			doc.Title, doc.SourceURL, doc.Lang, hash,
		).Scan(&docID).Error
		if err != nil {
			return err
		}
		if docID == 0 {
			fmt.Println("document already exists, skipping")
			return nil
		}
		for i, c := range chunks {
			if err := tx.Exec(
				`INSERT INTO document_chunks (document_id, chunk_index, content, embedding) VALUES (?,?,?,?)`,
				docID, i, c, pgvector.NewVector(vecs[i]),
			).Error; err != nil {
				return err
			}
		}
		return nil
	})
}

// Retrieve 走 pgvector 余弦距离取 top-K。
func Retrieve(ctx context.Context, db *gorm.DB, emb internal.Embedder, q string, topK int) ([]Chunk, error) {
	if topK <= 0 {
		topK = 3
	}
	qVecs, err := emb.Embed(ctx, []string{q})
	if err != nil {
		return nil, err
	}
	rows, err := db.WithContext(ctx).Raw(`
        SELECT id, document_id, chunk_index, content
        FROM document_chunks
        ORDER BY embedding <=> ?
        LIMIT ?`, pgvector.NewVector(qVecs[0]), topK).Rows()
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []Chunk
	for rows.Next() {
		var c Chunk
		if err := rows.Scan(&c.ID, &c.DocID, &c.Index, &c.Content); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

const promptTpl = `基于以下参考资料回答问题。若参考资料不足以回答，请回答"我不知道"。

参考资料：
%s

问题：%s
答案：`

// Answer 拼 prompt 调 LLM。
func Answer(ctx context.Context, llm internal.LLM, q string, chunks []Chunk) (string, error) {
	var b strings.Builder
	for i, c := range chunks {
		fmt.Fprintf(&b, "[%d] %s\n", i+1, c.Content)
	}
	prompt := fmt.Sprintf(promptTpl, b.String(), q)
	return llm.Complete(ctx, prompt)
}
