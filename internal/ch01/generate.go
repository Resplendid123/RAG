package ch01

import (
	"context"
	"fmt"
	"strings"

	"rag/infrastructure"
)

// promptTpl 兜底句"若参考资料不足以回答,请回答'我不知道'"是必填,
// 防止 LLM 在检索为空/不相关时自由发挥导致幻觉。
const promptTpl = `基于以下参考资料回答问题。若参考资料不足以回答，请回答"我不知道"。

参考资料：
%s

问题：%s
答案：`

// Generate 把 top-K chunks 按 [i] content 编号拼进 promptTpl,交 LLM 生成答案。
func Generate(ctx context.Context, llm infrastructure.LLM, q string, chunks []NaiveChunk) (string, error) {
	var b strings.Builder
	for i, c := range chunks {
		fmt.Fprintf(&b, "[%d] %s\n", i+1, c.Content)
	}
	prompt := fmt.Sprintf(promptTpl, b.String(), q)
	return llm.Complete(ctx, prompt)
}
