package splitter

// FixedSplit 按 rune 数硬切,忽略任何结构(代码块/表格/heading);链的兜底层。
func FixedSplit(text string, cfg SplitterConfig) []Chunk {
	size, overlap := cfg.ChunkSize, cfg.ChunkOverlap
	runes := []rune(text)
	var parts []string
	for i := 0; i < len(runes); i += size - overlap {
		end := min(i+size, len(runes))
		parts = append(parts, string(runes[i:end]))
		if end == len(runes) {
			break
		}
	}
	return StringChunks(parts)
}
