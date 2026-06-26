package ch05

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"rag/infrastructure"
	"rag/internal/ragcore"
)

const rewritePrompt = `你是检索查询改写器。规则:
- 保持原意不变,只改述法
- 禁止添加新概念、限定词或猜测
- 输出适合关键词检索的形式(名词化、补全口语省略)
- 只输出改写后的问题,不要任何解释

对话历史:
%s
用户问题:%s
改写后:`

// RewriteQuery 把口语化 query 改写成检索友好形式。history 给对话上文,空表示单轮。
// LLM 失败时降级返回原 query,不阻塞主流程。
func RewriteQuery(ctx context.Context, llm infrastructure.LLM, q string, history []string) (string, error) {
	out, err := llm.Complete(ctx, fmt.Sprintf(rewritePrompt, strings.Join(history, "\n"), q))
	if err != nil {
		return q, err
	}
	out = ragcore.StripThink(out)
	if out == "" {
		return q, nil
	}
	return out, nil
}

const multiQueryPrompt = `为以下问题生成 %d 个不同视角的检索变体,每行一个。规则:
- 覆盖不同关键词、不同表述、不同切入角度
- 不要重复原问题
- 只输出变体本身,不要编号或解释

原问题:%s`

// MultiQueryVariants 让 LLM 生成多个检索变体。失败/空时降级只返原 query。
func MultiQueryVariants(ctx context.Context, llm infrastructure.LLM, q string, n int) []string {
	if n <= 0 {
		n = 3
	}
	out, err := llm.Complete(ctx, fmt.Sprintf(multiQueryPrompt, n, q))
	if err != nil {
		return []string{q}
	}
	variants := parseVariants(ragcore.StripThink(out))
	if len(variants) == 0 {
		return []string{q}
	}
	return variants
}

var (
	// 去掉 "- xxx" / "1. xxx" / "1) xxx" 这类常见前缀。
	leadingPrefixRE = regexp.MustCompile(`^[\s\-•·]*(\d+[.)、]?)?\s*`)
	// 撇掉代码围栏。
	fenceRE = regexp.MustCompile("```[a-zA-Z]*\\n?|```")
)

// parseVariants 把 LLM 输出拆成一行一条变体,容忍编号、破折号、markdown 围栏。
func parseVariants(s string) []string {
	s = fenceRE.ReplaceAllString(s, "")
	var out []string
	for line := range strings.SplitSeq(s, "\n") {
		line = leadingPrefixRE.ReplaceAllString(line, "")
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		out = append(out, line)
	}
	return out
}
