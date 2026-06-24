package splitter

import (
	"regexp"
	"sort"
	"strings"
)

// Chunk 是切分产物;检索 / 入库 / 生成各层共用,不分 Parent / Child 子集。
type Chunk struct {
	Content       string
	ContextHeader string
	Seq           int
}

// SplitterConfig 是 splitter 链的配置。Strategy 留空走全 4 层链,显式指定从对应 tier 起跑。
type SplitterConfig struct {
	ChunkSize    int
	ChunkOverlap int
	Separators   []string
	Strategy     string
	Languages    []string
}

const (
	DefaultChunkSize    = 512
	DefaultChunkOverlap = 80
)

func DefaultConfig() SplitterConfig {
	return SplitterConfig{
		ChunkSize:    DefaultChunkSize,
		ChunkOverlap: DefaultChunkOverlap,
		Separators:   []string{"\n\n", "\n", "。", "!", "?", "，", " ", ""},
	}
}

// EnsureDefaults 兜底零值;ChunkOverlap > ChunkSize/2 时砍半(避免每 chunk 都克隆前一个)。
func EnsureDefaults(cfg SplitterConfig) SplitterConfig {
	if cfg.ChunkSize <= 0 {
		cfg.ChunkSize = DefaultChunkSize
	}
	if cfg.ChunkOverlap < 0 {
		cfg.ChunkOverlap = 0
	}
	if len(cfg.Separators) == 0 {
		cfg.Separators = []string{"\n\n", "\n", "。", "!", "?", "，", " ", ""}
	}
	if cfg.ChunkOverlap > cfg.ChunkSize/2 && cfg.ChunkSize > 0 {
		cfg.ChunkOverlap = cfg.ChunkSize / 2
	}
	return cfg
}

// StringChunks 把 string 列表包装成 Chunk,顺序赋 Seq。
func StringChunks(parts []string) []Chunk {
	out := make([]Chunk, len(parts))
	for i, p := range parts {
		out[i] = Chunk{Seq: i + 1, Content: strings.TrimSpace(p)}
	}
	return out
}

// protectedPatterns 是 splitter 不可切穿区间(代码块 / 表格);中间断开会让 embedding 召回彻底失效。
var protectedPatterns = []*regexp.Regexp{
	regexp.MustCompile("(?s)```\\w*[\\r\\n].*?```"), // fenced code block
	regexp.MustCompile("(?m)(?:^\\|.*[\\r\\n]+)+"),  // 连续 markdown 表格行
}

// Span 是 text 里的一个 [Start, End) 区间。
type Span struct{ Start, End int }

// FindProtectedSpans 返回 text 里所有受保护区间(start 升序、不重叠)。
func FindProtectedSpans(text string) []Span {
	var all []Span
	for _, pat := range protectedPatterns {
		for _, loc := range pat.FindAllStringIndex(text, -1) {
			if loc[1] > loc[0] {
				all = append(all, Span{loc[0], loc[1]})
			}
		}
	}
	sort.Slice(all, func(i, j int) bool { return all[i].Start < all[j].Start })
	var out []Span
	lastEnd := 0
	for _, s := range all {
		if s.Start >= lastEnd {
			out = append(out, s)
			lastEnd = s.End
		}
	}
	return out
}
