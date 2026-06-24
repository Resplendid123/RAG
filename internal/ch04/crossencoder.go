package ch04

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"rag/infrastructure"
)

// RerankResult 精排后单条结果:Index 指向原始候选下标,Score 是相关性分(越高越相关)。
type RerankResult struct {
	Index int
	Score float64
}

// Reranker 精排接口,docs 给候选,返回按相关性降序的 RerankResult 列表。
type Reranker interface {
	Rerank(ctx context.Context, query string, docs []string) []RerankResult
}

// LLMReranker 用 LLM 当 cross-encoder:一次 prompt 给所有候选打分,JSON 解析后排序。
// 真 cross-encoder 每对一次;LLM 一次 batch 是 listwise 折中,演示 rerank 效果足够。
type LLMReranker struct {
	llm infrastructure.LLM
}

func NewLLMReranker(llm infrastructure.LLM) *LLMReranker {
	return &LLMReranker{llm: llm}
}

const rerankPrompt = `你是相关性评分模型。对每条文档,根据它回答 query 的相关度给出 0-10 分(10=直接命中,0=完全无关),只输出 JSON 数组。

query: %s

文档:
%s

返回格式:[{"index":0,"score":8.5},{"index":1,"score":3.2},...]`

func (r *LLMReranker) Rerank(ctx context.Context, query string, docs []string) []RerankResult {
	if len(docs) == 0 {
		return nil
	}
	var b strings.Builder
	for i, d := range docs {
		fmt.Fprintf(&b, "[%d] %s\n", i, d)
	}
	out, err := r.llm.Complete(ctx, fmt.Sprintf(rerankPrompt, query, b.String()))
	if err != nil {
		return fallbackOrder(len(docs))
	}
	var scores []struct {
		Index int     `json:"index"`
		Score float64 `json:"score"`
	}
	cleaned := extractJSON(out)
	if err := json.Unmarshal([]byte(cleaned), &scores); err != nil {
		return fallbackOrder(len(docs))
	}
	res := make([]RerankResult, 0, len(scores))
	for _, s := range scores {
		if s.Index < 0 || s.Index >= len(docs) {
			continue
		}
		res = append(res, RerankResult{Index: s.Index, Score: s.Score})
	}
	sort.Slice(res, func(i, j int) bool { return res[i].Score > res[j].Score })
	if len(res) == 0 {
		return fallbackOrder(len(docs))
	}
	return res
}

// fallbackOrder LLM 解析失败时按原序返回,rerank 退化为 passthrough。
func fallbackOrder(n int) []RerankResult {
	res := make([]RerankResult, n)
	for i := range res {
		res[i] = RerankResult{Index: i, Score: 0}
	}
	return res
}

// extractJSON 从 LLM 输出里抠出 JSON 数组,容错 ```json``` 包裹、<think> 块和前后噪声。
// 思路:剥 markdown 围栏 → 剥 <think> 块(DeepSeek-R1 等会用)→ 整体 Valid → 退而求其次取首 [ 到末 ]。
func extractJSON(s string) string {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "```") {
		if i := strings.Index(s, "\n"); i >= 0 {
			s = s[i+1:]
		}
		s = strings.TrimSuffix(s, "```")
		s = strings.TrimSpace(s)
	}
	for {
		start := strings.Index(s, "<think>")
		end := strings.Index(s, "</think>")
		if start < 0 || end < 0 || end < start {
			break
		}
		s = strings.TrimSpace(s[:start] + s[end+len("</think>"):])
	}
	if json.Valid([]byte(s)) {
		return s
	}
	start, end := strings.Index(s, "["), strings.LastIndex(s, "]")
	if start < 0 || end <= start {
		return ""
	}
	return s[start : end+1]
}
