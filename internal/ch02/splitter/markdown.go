package splitter

import (
	"strings"
	"unicode/utf8"

	"github.com/yuin/goldmark"
	"github.com/yuin/goldmark/ast"
	"github.com/yuin/goldmark/text"
)

// MarkdownSplit 走 goldmark AST,heading 进 ContextHeader,代码块/列表整体输出,段落累积到 ChunkSize 阈值 flush。
// 阈值用 rune 数比(非 byte):中文 1 字 = 3 bytes,按 byte 算会导致每段单独成 chunk,parent chain 因此拿不到大段。
func MarkdownSplit(src string, cfg SplitterConfig) []Chunk {
	bs := []byte(src)
	root := goldmark.New().Parser().Parse(text.NewReader(bs))
	var chunks []Chunk
	var buf strings.Builder
	var heading string
	flush := func() {
		if buf.Len() == 0 {
			return
		}
		chunks = append(chunks, Chunk{
			Seq:           len(chunks) + 1,
			Content:       strings.TrimSpace(buf.String()),
			ContextHeader: heading,
		})
		buf.Reset()
	}
	for n := root.FirstChild(); n != nil; n = n.NextSibling() {
		switch v := n.(type) {
		case *ast.Heading:
			flush()
			heading = strings.Repeat("#", v.Level) + " " + headingText(v, bs)
		case *ast.FencedCodeBlock, *ast.CodeBlock, *ast.List:
			flush()
			chunks = append(chunks, Chunk{
				Seq:           len(chunks) + 1,
				Content:       strings.TrimRight(string(v.Lines().Value(bs)), "\n"),
				ContextHeader: heading,
			})
		case *ast.Paragraph, *ast.Blockquote, *ast.TextBlock:
			body := strings.TrimSpace(string(v.Lines().Value(bs)))
			bufRunes := utf8.RuneCountInString(buf.String())
			bodyRunes := utf8.RuneCountInString(body)
			if bufRunes > 0 && bufRunes+bodyRunes+1 > cfg.ChunkSize {
				flush()
				bufRunes = 0
			}
			if bufRunes > 0 {
				buf.WriteString("\n\n")
			}
			buf.WriteString(body)
		}
	}
	flush()
	return chunks
}

func headingText(h *ast.Heading, src []byte) string {
	var b strings.Builder
	for c := h.FirstChild(); c != nil; c = c.NextSibling() {
		if t, ok := c.(*ast.Text); ok {
			b.Write(t.Segment.Value(src))
		}
	}
	return b.String()
}
