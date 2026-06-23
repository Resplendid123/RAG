package infrastructure

import (
	"context"
	"fmt"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	Dimension() int
}

type openaiEmbedder struct {
	client    openai.Client
	model     string
	dimension int
}

func (e *openaiEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	resp, err := e.client.Embeddings.New(ctx, openai.EmbeddingNewParams{
		Input: openai.EmbeddingNewParamsInputUnion{OfArrayOfStrings: texts},
		Model: e.model,
	})
	if err != nil {
		return nil, fmt.Errorf("openai embed: %w", err)
	}
	out := make([][]float32, 0, len(resp.Data))
	for _, d := range resp.Data {
		v := make([]float32, len(d.Embedding))
		for i, f := range d.Embedding {
			v[i] = float32(f)
		}
		out = append(out, v)
	}
	return out, nil
}

func (e *openaiEmbedder) Dimension() int { return e.dimension }

func NewOpenAIEmbedder(baseURL, apiKey, model string, dimension int) Embedder {
	return &openaiEmbedder{
		client:    openai.NewClient(option.WithAPIKey(apiKey), option.WithBaseURL(baseURL)),
		model:     model,
		dimension: dimension,
	}
}

func NewOllamaEmbedder(baseURL, model string, dimension int) Embedder {
	return &openaiEmbedder{
		client:    openai.NewClient(option.WithAPIKey("ollama"), option.WithBaseURL(baseURL)),
		model:     model,
		dimension: dimension,
	}
}
