package ch02

import (
	"context"
	"fmt"
	"strings"

	"rag/infrastructure"
	"rag/internal/ch02/splitter"
)

const promptTpl = `基于以下参考资料回答问题。若参考资料不足以回答，请回答"我不知道"。

参考资料：
%s

问题：%s
答案：`

// Generate 把 top-K parent chunks 编号拼进 prompt,交 LLM 生成答案。chunks 是 parent 全文(非 child 切片,与 ch01 区别在此)。
func Generate(ctx context.Context, llm infrastructure.LLM, q string, chunks []splitter.Chunk) (string, error) {
	var b strings.Builder
	for i, c := range chunks {
		fmt.Fprintf(&b, "[%d] %s\n", i+1, c.Content)
	}
	prompt := fmt.Sprintf(promptTpl, b.String(), q)
	return llm.Complete(ctx, prompt)
}
