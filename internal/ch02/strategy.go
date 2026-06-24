package ch02

import (
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"

	"rag/internal/ch02/splitter"
)

// tier 是 chain 一层,Name 给日志用,Split 是切分实现。
type tier struct {
	name  string
	split func(string, splitter.SplitterConfig) []splitter.Chunk
}

// ValidateChunks 校验切分结果:chunk 数 > 0、单 chunk ≤ size×3、覆盖原文 ≥ 60%。
// 60% 经验值——Markdown 把 heading 抽到 ContextHeader,总覆盖会下降。
func ValidateChunks(chunks []splitter.Chunk, orig string, cfg splitter.SplitterConfig) error {
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
func SplitWithTierChain(text string, cfg splitter.SplitterConfig) (chunks []splitter.Chunk, used string, err error) {
	cfg = splitter.EnsureDefaults(cfg)
	for _, t := range buildChain(cfg) {
		got := t.split(text, cfg)
		if vErr := ValidateChunks(got, text, cfg); vErr == nil {
			return got, t.name, nil
		} else {
			err = vErr
		}
	}
	return splitter.FixedSplit(text, cfg), "fixed", err
}

// buildChain 按 cfg.Strategy 选链顺序;空值走全 4 层。
func buildChain(cfg splitter.SplitterConfig) []tier {
	switch cfg.Strategy {
	case "heuristic":
		return []tier{{"heuristic", splitter.HeuristicSplit}, {"recursive", splitter.RecursiveSplit}, {"fixed", splitter.FixedSplit}}
	case "recursive":
		return []tier{{"recursive", splitter.RecursiveSplit}, {"fixed", splitter.FixedSplit}}
	case "fixed":
		return []tier{{"fixed", splitter.FixedSplit}}
	default: // "heading" / "" / 未知:全 4 层
		return []tier{{"markdown", splitter.MarkdownSplit}, {"heuristic", splitter.HeuristicSplit}, {"recursive", splitter.RecursiveSplit}, {"fixed", splitter.FixedSplit}}
	}
}

// ChildChunk 扩展 splitter.Chunk 带父索引,检索/入库层都消费 Chunk 本身。
type ChildChunk struct {
	splitter.Chunk
	ParentIndex int
}

type ParentChildResult struct {
	Parents  []splitter.Chunk
	Children []ChildChunk
}

func SplitParentChild(text string, parentCfg, childCfg splitter.SplitterConfig) ParentChildResult {
	if text == "" {
		return ParentChildResult{}
	}
	parents, _, _ := SplitWithTierChain(text, parentCfg)
	if len(parents) == 0 {
		return ParentChildResult{}
	}
	var keptParents []splitter.Chunk
	var children []ChildChunk
	childSeq := 0
	for _, p := range parents {
		parts, _, _ := SplitWithTierChain(p.Content, childCfg)
		var subs []splitter.Chunk
		for _, sub := range parts {
			content := strings.TrimSpace(sub.Content)
			if content == "" {
				continue
			}
			subs = append(subs, splitter.Chunk{Content: content})
		}
		// parent 真被切碎才入库;否则只留 child,ParentIndex = -1。
		parentIndex := -1
		if len(subs) > 1 || (len(subs) == 1 && subs[0].Content != p.Content) {
			p.Seq = len(keptParents) + 1
			keptParents = append(keptParents, p)
			parentIndex = len(keptParents) - 1
		}
		for _, sub := range subs {
			children = append(children, ChildChunk{
				Chunk: splitter.Chunk{
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
