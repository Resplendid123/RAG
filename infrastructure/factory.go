package infrastructure

import "fmt"

// NewLLM 起 OpenAI 协议 LLM:ollama 时把 apiKey 占位为 "ollama",openai 走 yaml 配的 key。
func NewLLM(cfg LLMConfig) (LLM, error) {
	apiKey := cfg.APIKey
	if cfg.Provider == "ollama" {
		apiKey = "ollama"
	}
	if cfg.Provider != "" && cfg.Provider != "ollama" && cfg.Provider != "openai" {
		return nil, fmt.Errorf("unsupported llm provider: %q", cfg.Provider)
	}
	return newOpenAIBackend(cfg.BaseUrl, apiKey, cfg.Model), nil
}

// NewEmbedder 起 OpenAI 协议 Embedder,ollama/openai 走同一实现。
func NewEmbedder(cfg EmbeddingConfig) (Embedder, error) {
	apiKey := cfg.APIKey
	if cfg.Provider == "ollama" {
		apiKey = "ollama"
	}
	if cfg.Provider != "" && cfg.Provider != "ollama" && cfg.Provider != "openai" {
		return nil, fmt.Errorf("unsupported embedding provider: %q", cfg.Provider)
	}
	return NewOpenAIEmbedder(cfg.BaseUrl, apiKey, cfg.Model, cfg.Dimension), nil
}
