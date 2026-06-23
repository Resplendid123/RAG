package ch03

import (
	"context"
	"fmt"
	"strings"

	"gorm.io/gorm"

	"rag/infrastructure"
	"rag/internal/ch02/splitter"
)

const promptTpl = `基于以下参考资料回答问题。若参考资料不足以回答，请回答"我不知道"。

参考资料：
%s

问题：%s
答案：`

// Generate 把 RRF 选出来的 child chunks 编号拼进 prompt,交 LLM 生成答案。
func Generate(ctx context.Context, llm infrastructure.LLM, q string, chunks []splitter.Chunk) (string, error) {
	var b strings.Builder
	for i, c := range chunks {
		fmt.Fprintf(&b, "[%d] %s\n", i+1, c.Content)
	}
	prompt := fmt.Sprintf(promptTpl, b.String(), q)
	return llm.Complete(ctx, prompt)
}

// LoadChunks 按 hit 顺序从 document_chunks 取出 content,给 Generate 用。
func LoadChunks(ctx context.Context, db *gorm.DB, hits []Hit) ([]splitter.Chunk, error) {
	if len(hits) == 0 {
		return nil, nil
	}
	byID := make(map[int64]splitter.Chunk, len(hits))
	for _, h := range hits {
		if _, ok := byID[h.ChunkID]; ok {
			continue
		}
		var c splitter.Chunk
		if err := db.WithContext(ctx).Raw(
			`SELECT content, chunk_index FROM document_chunks WHERE id = ?`,
			h.ChunkID,
		).Row().Scan(&c.Content, &c.Seq); err != nil {
			return nil, err
		}
		byID[h.ChunkID] = c
	}
	out := make([]splitter.Chunk, 0, len(hits))
	for _, h := range hits {
		if c, ok := byID[h.ChunkID]; ok {
			out = append(out, c)
		}
	}
	return out, nil
}
