package ch05

import (
	"context"
	"fmt"

	"rag/infrastructure"
	"rag/internal/ragcore"
)

const hydePrompt = `基于你的知识,用一段话简洁回答以下问题(不需要真实准确,只要像文档中的写法)。
回答风格:%s
问题:%s
回答:`

// HyDEAnswer 让 LLM 生成"假设答案",后续用其 embedding 去检索。
// style 控文风(如"使用正式书面语"),缓解 LLM 口语化文本与目标文档的分布错位。
// 失败时降级返原 query,主流程不至于断。
func HyDEAnswer(ctx context.Context, llm infrastructure.LLM, q, style string) (string, error) {
	if style == "" {
		style = "使用正式书面语,陈述句为主"
	}
	out, err := llm.Complete(ctx, fmt.Sprintf(hydePrompt, style, q))
	if err != nil {
		return q, err
	}
	out = ragcore.StripThink(out)
	if out == "" {
		return q, nil
	}
	return out, nil
}
