package ch02

import (
	"unicode"
	"unicode/utf8"
)

const (
	langChinese = "zh"
	langEnglish = "en"
	langMixed   = "mixed"
)

// detectLanguage 按主导字符判断语种:>70% 汉字归 zh,>70% 拉丁字母归 en,否则 mixed。
func detectLanguage(text string) string {
	if text == "" {
		return langMixed
	}
	var cjk, latin int
	for _, r := range text {
		switch {
		case unicode.Is(unicode.Han, r):
			cjk++
		case unicode.IsLetter(r):
			latin++
		}
	}
	total := cjk + latin
	if total == 0 {
		return langMixed
	}
	if cjk*10 >= total*7 {
		return langChinese
	}
	if latin*10 >= total*7 {
		return langEnglish
	}
	return langMixed
}

// EstimateTokens 按语种密度估算 LLM token 数(中文 1.7、英文 4、混合 3 char/token);误差 ±30%,够用于 parent.token_count 粗参考。
func EstimateTokens(text string) int {
	if text == "" {
		return 0
	}
	density := 3.0
	switch detectLanguage(text) {
	case langChinese:
		density = 1.7
	case langEnglish:
		density = 4.0
	}
	return int(float64(utf8.RuneCountInString(text)) / density)
}
