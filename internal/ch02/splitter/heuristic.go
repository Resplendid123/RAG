package splitter

import (
	"regexp"
	"sort"
	"strings"
)

var (
	// 编号小节: 1. / 1) / 一、 / § N / 第N章
	sectionRe = regexp.MustCompile(`(?m)^(?:\d+[.)]|第[一二三四五六七八九十百千]+[章节]|§\s*\d+|[一二三四五六七八九十]+、)\s*`)
	// 页码标记: 第 N 页 / Page N (of M) / ---- Page 12 ----
	pageRe = regexp.MustCompile(`(?m)^\s*-{0,3}\s*(?:第\s*\d+\s*页|Page\s+\d+(?:\s+of\s+\d+)?)\s*-{0,3}\s*$`)
	// 全大写标题(英文 PDF 章节名):ABSTRACT / INDEX / CHAPTER 3
	capsRe = regexp.MustCompile(`(?m)^[A-Z][A-Z0-9 \t_-]{3,}$`)
)

// HeuristicSplit 正则识别编号小节 / 页码 / 全大写标题,适用于无 Markdown 结构的纯文本 / PDF / OCR。落在 protected span 内的 cut 直接剔除;长 piece 交给 ValidateChunks 触发链降级。
func HeuristicSplit(text string, cfg SplitterConfig) []Chunk {
	var raw []int
	raw = append(raw, findHeuristicCuts(text, sectionRe)...)
	raw = append(raw, findHeuristicCuts(text, pageRe)...)
	raw = append(raw, findHeuristicCuts(text, capsRe)...)
	cuts := mergeCuts(raw, cfg.ChunkSize/6)
	cuts = filterSafeCuts(text, cuts)
	return cutAt(text, cuts)
}

// filterSafeCuts 剔除落在 protected span 内的 cut。
func filterSafeCuts(text string, cuts []int) []int {
	spans := FindProtectedSpans(text)
	if len(spans) == 0 {
		return cuts
	}
	out := make([]int, 0, len(cuts))
	for _, c := range cuts {
		inside := false
		for _, s := range spans {
			if c > s.Start && c < s.End {
				inside = true
				break
			}
		}
		if !inside {
			out = append(out, c)
		}
	}
	return out
}

func findHeuristicCuts(text string, re *regexp.Regexp) []int {
	var cuts []int
	for _, loc := range re.FindAllStringIndex(text, -1) {
		if loc[0] > 0 {
			cuts = append(cuts, loc[0])
		}
	}
	return cuts
}

func mergeCuts(cuts []int, minGap int) []int {
	if len(cuts) == 0 {
		return cuts
	}
	sort.Ints(cuts)
	out := []int{cuts[0]}
	for _, c := range cuts[1:] {
		if c-out[len(out)-1] >= minGap {
			out = append(out, c)
		}
	}
	return out
}

func cutAt(text string, cuts []int) []Chunk {
	boundaries := make([]int, 0, len(cuts)+2)
	boundaries = append(boundaries, 0)
	boundaries = append(boundaries, cuts...)
	boundaries = append(boundaries, len(text))

	var out []Chunk
	for i := 0; i < len(boundaries)-1; i++ {
		piece := strings.TrimSpace(text[boundaries[i]:boundaries[i+1]])
		if piece == "" {
			continue
		}
		out = append(out, Chunk{Content: piece})
	}
	for i := range out {
		out[i].Seq = i
	}
	return out
}
