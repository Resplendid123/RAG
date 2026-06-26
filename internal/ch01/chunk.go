package ch01

// Split 按 rune 数硬切 + 重叠;不感知语义边界。
func Split(text string, size, overlap int) []string {
	if size <= 0 {
		return nil
	}
	if overlap >= size {
		overlap = size / 2
	}
	runes := []rune(text)
	var out []string
	for i := 0; i < len(runes); i += size - overlap {
		end := min(i+size, len(runes))
		out = append(out, string(runes[i:end]))
		if end == len(runes) {
			break
		}
	}
	return out
}
