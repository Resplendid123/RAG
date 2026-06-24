package ch05

import "strings"

// stripThink 剥掉 DeepSeek-R1 等模型的 <think>...</think> 块,返回剩下的内容。
// 多次出现也一并剥离;块不闭合则不处理,避免误伤。
func stripThink(s string) string {
	for {
		start := strings.Index(s, "<think>")
		end := strings.Index(s, "</think>")
		if start < 0 || end < 0 || end < start {
			return s
		}
		s = strings.TrimSpace(s[:start] + s[end+len("</think>"):])
	}
}

// cleanAnswer 组合剥 <think> 和前后空白;供 RewriteQuery / HyDEAnswer / StepBack 复用。
func cleanAnswer(s string) string {
	return strings.TrimSpace(stripThink(s))
}
