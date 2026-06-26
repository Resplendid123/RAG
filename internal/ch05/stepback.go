package ch05

import (
	"context"
	"fmt"

	"rag/infrastructure"
	"rag/internal/ragcore"
)

const stepBackPrompt = `把以下具体问题抽象成一个更通用、更利于检索的上层问题。
规则:
- 保留关键概念,移除具体版本号、库名等限定
- 用更宽泛的术语替换具体术语
- 只输出抽象后的问题,不要任何解释

具体问题:%s
抽象问题:`

// StepBackResult 携带双路检索结果,供下游 prompt 拼接原问题上下文 + 抽象上下文。
type StepBackResult struct {
	Original    string
	StepBack    string
	OriginalCtx string
	StepBackCtx string
}

// StepBack 把具体问题抽象成上层问题。LLM 失败时降级返原 query。
func StepBack(ctx context.Context, llm infrastructure.LLM, q string) (string, error) {
	out, err := llm.Complete(ctx, fmt.Sprintf(stepBackPrompt, q))
	if err != nil {
		return q, err
	}
	out = ragcore.StripThink(out)
	if out == "" {
		return q, nil
	}
	return out, nil
}
