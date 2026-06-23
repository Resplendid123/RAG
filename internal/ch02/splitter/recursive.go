package splitter

import (
	"strings"
	"unicode/utf8"
)

// RecursiveSplit 按 separators 递归切;代码块/表格段整体保留(抽 protected span 后切 gap)。
// Separators 为空返回 nil,链降级到 FixedSplit。cfg 由链入口 normalize。
func RecursiveSplit(text string, cfg SplitterConfig) []Chunk {
	if text == "" {
		return nil
	}
	spans := FindProtectedSpans(text)
	if len(spans) == 0 {
		return StringChunks(recursiveSplitCore(text, cfg.Separators, cfg.ChunkSize))
	}
	var parts []string
	cursor := 0
	for _, p := range spans {
		if p.Start > cursor {
			parts = append(parts, recursiveSplitCore(text[cursor:p.Start], cfg.Separators, cfg.ChunkSize)...)
		}
		parts = append(parts, text[p.Start:p.End])
		cursor = p.End
	}
	if cursor < len(text) {
		parts = append(parts, recursiveSplitCore(text[cursor:], cfg.Separators, cfg.ChunkSize)...)
	}
	return StringChunks(parts)
}

// recursiveSplitCore 按 separators 递归切,长 part 用更细分隔符降级;无 separators 返回 nil 让链降级。
func recursiveSplitCore(text string, separators []string, size int) []string {
	if len(separators) == 0 {
		return nil
	}
	if utf8.RuneCountInString(text) <= size {
		return []string{text}
	}
	sep := separators[0]
	rest := separators[1:]
	parts := strings.Split(text, sep)
	var out []string
	var buf strings.Builder
	for _, part := range parts {
		if utf8.RuneCountInString(buf.String()+sep+part) > size {
			if buf.Len() > 0 {
				out = append(out, strings.TrimSpace(buf.String()))
				buf.Reset()
			}
			if utf8.RuneCountInString(part) > size {
				out = append(out, recursiveSplitCore(part, rest, size)...)
				continue
			}
		}
		if buf.Len() > 0 {
			buf.WriteString(sep)
		}
		buf.WriteString(part)
	}
	if buf.Len() > 0 {
		out = append(out, strings.TrimSpace(buf.String()))
	}
	return out
}
