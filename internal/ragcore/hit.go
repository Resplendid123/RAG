// Package ragcore 跨章节共享的检索 / 工具 helpers。Lesson 包可独立引用,不构成 chapter 之间的耦合。
package ragcore

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"

	"gorm.io/gorm"

	"rag/internal/ch02/splitter"
)

// Hit 是 dense / bm25 单路召回的统一结构;RRF 用 ChunkID 对齐,Content 给回显/生成用。
type Hit struct {
	ChunkID int64
	Content string
	Source  string
	Rank    int
}

// IDsOf 把 hits 的 ChunkID 拼成 "[1 2 3]" 形式用于日志回显。
func IDsOf(hits []Hit) string {
	if len(hits) == 0 {
		return "[]"
	}
	parts := make([]string, len(hits))
	for i, h := range hits {
		parts[i] = strconv.FormatInt(h.ChunkID, 10)
	}
	return "[" + strings.Join(parts, " ") + "]"
}

// Snippet 把字符串裁到 n 个 rune,超出加 "...";先 TrimSpace。常用于 LLM 答案回显。
func Snippet(s string, n int) string {
	s = strings.TrimSpace(s)
	r := []rune(s)
	if len(r) > n {
		return string(r[:n]) + "..."
	}
	return s
}

// StripThink 剥掉 DeepSeek-R1 等模型的 <think>...</think> 块,多次出现也一并剥离;块不闭合则不处理。
// 优先用 string-scan 而非 regex,避免未闭合 think 块炸开。
func StripThink(s string) string {
	for {
		start := strings.Index(s, "<think>")
		end := strings.Index(s, "</think>")
		if start < 0 || end < 0 || end < start {
			return s
		}
		s = strings.TrimSpace(s[:start] + s[end+len("</think>"):])
	}
}

// JoinContents 把 chunks 拼成单字符串,sep 通常为 "\n" 或 " | "。
func JoinContents(chunks []splitter.Chunk, sep string) string {
	parts := make([]string, len(chunks))
	for i, c := range chunks {
		parts[i] = c.Content
	}
	return strings.Join(parts, sep)
}

// FormatNumbered 把 chunks 拼成 "[i] content\n" 的形式,用于 prompt 模板里的"参考资料"段。
func FormatNumbered(chunks []splitter.Chunk) string {
	var b strings.Builder
	for i, c := range chunks {
		fmt.Fprintf(&b, "[%d] %s\n", i+1, c.Content)
	}
	return b.String()
}

// ContentHash 用 SHA-256 给一组 chunk 计算指纹,用作 documents.content_hash 唯一约束。
func ContentHash(chunks []splitter.Chunk) string {
	src := JoinContents(chunks, "\n")
	sum := sha256.Sum256([]byte(src))
	return hex.EncodeToString(sum[:])
}

// LoadChildChunkIDs 按 chunk_index 升序读 document_chunks.id,常用于 LLM 抽取时把 child 对应回 DB 主键。
func LoadChildChunkIDs(ctx context.Context, db *gorm.DB) ([]int64, error) {
	rows, err := db.WithContext(ctx).Raw(
		`SELECT id FROM document_chunks ORDER BY chunk_index`).Rows()
	if err != nil {
		return nil, fmt.Errorf("load chunk ids: %w", err)
	}
	defer rows.Close()
	var out []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		out = append(out, id)
	}
	return out, rows.Err()
}
