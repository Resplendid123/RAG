package internal

import "fmt"

func NewLLM(cfg LLMConfig) (LLM, error) {
	switch cfg.Provider {
	case "ollama":
		return NewOllamaLLM(cfg.BaseUrl, cfg.Model), nil
	case "openai", "":
		return NewOpenAILLM(cfg.BaseUrl, cfg.APIKey, cfg.Model), nil
	default:
		return nil, fmt.Errorf("unsupported llm provider: %q", cfg.Provider)
	}
}

func NewEmbedder(cfg EmbeddingConfig) (Embedder, error) {
	switch cfg.Provider {
	case "ollama":
		return NewOllamaEmbedder(cfg.BaseUrl, cfg.Model, cfg.Dimension), nil
	case "openai", "":
		return NewOpenAIEmbedder(cfg.BaseUrl, cfg.APIKey, cfg.Model, cfg.Dimension), nil
	default:
		return nil, fmt.Errorf("unsupported embedding provider: %q", cfg.Provider)
	}
}
