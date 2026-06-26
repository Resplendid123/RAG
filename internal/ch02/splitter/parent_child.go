package splitter

import (
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"
)

// tier 是 chain 一层,Name 给日志用,Split 是切分实现。
type tier struct {
	name  string
	split func(string, SplitterConfig) []Chunk
}

// ValidateChunks 校验切分结果:chunk 数 > 0、单 chunk ≤ size×3、覆盖原文 ≥ 60%。
// 60% 经验值——Markdown 把 heading 抽到 ContextHeader,总覆盖会下降。
func ValidateChunks(chunks []Chunk, orig string, cfg SplitterConfig) error {
	if len(chunks) == 0 {
		return errors.New("no chunks")
	}
	origLen := utf8.RuneCountInString(orig)
	if origLen == 0 {
		return errors.New("empty input")
	}
	var sum int
	for _, c := range chunks {
		n := utf8.RuneCountInString(c.Content)
		if n > cfg.ChunkSize*3 {
			return fmt.Errorf("chunk too large: %d runes", n)
		}
		sum += n
	}
	if float64(sum)/float64(origLen) < 0.6 {
		return fmt.Errorf("low coverage: %d/%d", sum, origLen)
	}
	return nil
}

// SplitWithTierChain 按链顺序跑,ValidateChunks 通过即返回,失败降级下一层。
func SplitWithTierChain(text string, cfg SplitterConfig) (chunks []Chunk, used string, err error) {
	cfg = EnsureDefaults(cfg)
	for _, t := range buildChain(cfg) {
		got := t.split(text, cfg)
		if vErr := ValidateChunks(got, text, cfg); vErr == nil {
			return got, t.name, nil
		} else {
			err = vErr
		}
	}
	return FixedSplit(text, cfg), "fixed", err
}

// buildChain 按 cfg.Strategy 选链顺序;空值走全 4 层。
func buildChain(cfg SplitterConfig) []tier {
	switch cfg.Strategy {
	case "heuristic":
		return []tier{{"heuristic", HeuristicSplit}, {"recursive", RecursiveSplit}, {"fixed", FixedSplit}}
	case "recursive":
		return []tier{{"recursive", RecursiveSplit}, {"fixed", FixedSplit}}
	case "fixed":
		return []tier{{"fixed", FixedSplit}}
	default: // "heading" / "" / 未知:全 4 层
		return []tier{{"markdown", MarkdownSplit}, {"heuristic", HeuristicSplit}, {"recursive", RecursiveSplit}, {"fixed", FixedSplit}}
	}
}

// ParentChildResult 是 SplitParentChild 的产物,parents 是入库的 parent,children 是带父索引的 child。
type ParentChildResult struct {
	Parents  []Chunk
	Children []ChildChunk
}

// SplitParentChild 跑 parent 链切,再对每个 parent 跑 child 链切,返回 parents/children。
// parent 真被切碎才入库;否则只留 child,ParentIndex = -1。
func SplitParentChild(text string, parentCfg, childCfg SplitterConfig) ParentChildResult {
	if text == "" {
		return ParentChildResult{}
	}
	parents, _, _ := SplitWithTierChain(text, parentCfg)
	if len(parents) == 0 {
		return ParentChildResult{}
	}
	var keptParents []Chunk
	var children []ChildChunk
	childSeq := 0
	for _, p := range parents {
		parts, _, _ := SplitWithTierChain(p.Content, childCfg)
		var subs []Chunk
		for _, sub := range parts {
			content := strings.TrimSpace(sub.Content)
			if content == "" {
				continue
			}
			subs = append(subs, Chunk{Content: content})
		}
		parentIndex := -1
		if len(subs) > 1 || (len(subs) == 1 && subs[0].Content != p.Content) {
			p.Seq = len(keptParents) + 1
			keptParents = append(keptParents, p)
			parentIndex = len(keptParents) - 1
		}
		for _, sub := range subs {
			children = append(children, ChildChunk{
				Chunk: Chunk{
					Seq:           childSeq + 1,
					Content:       sub.Content,
					ContextHeader: mergeBreadcrumb(p.ContextHeader, sub.ContextHeader),
				},
				ParentIndex: parentIndex,
			})
			childSeq++
		}
	}
	return ParentChildResult{Parents: keptParents, Children: children}
}

// mergeBreadcrumb 拼接父子面包屑,parent 末行 == child 首行则去重,避免 embedding 重复 heading。
func mergeBreadcrumb(parent, child string) string {
	if parent == "" {
		return child
	}
	if child == "" {
		return parent
	}
	parentLines := strings.Split(parent, "\n")
	childLines := strings.Split(child, "\n")
	if len(parentLines) > 0 && len(childLines) > 0 &&
		strings.TrimSpace(parentLines[len(parentLines)-1]) == strings.TrimSpace(childLines[0]) {
		childLines = childLines[1:]
	}
	if len(childLines) == 0 {
		return parent
	}
	return parent + "\n" + strings.Join(childLines, "\n")
}
