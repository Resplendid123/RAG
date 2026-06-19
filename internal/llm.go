package internal

import (
	"context"
	"fmt"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

type Message struct {
	Role    string
	Content string
}

type LLM interface {
	Complete(ctx context.Context, prompt string, opts ...CompleteOption) (string, error)
	Chat(ctx context.Context, msgs []Message, opts ...CompleteOption) (string, error)
}

type CompleteConfig struct {
	Temperature float64
	MaxTokens   int
	Stop        []string
}

type CompleteOption func(*CompleteConfig)

func WithTemperature(t float64) CompleteOption {
	return func(c *CompleteConfig) { c.Temperature = t }
}

func WithMaxTokens(n int) CompleteOption {
	return func(c *CompleteConfig) { c.MaxTokens = n }
}

func WithStop(s ...string) CompleteOption {
	return func(c *CompleteConfig) { c.Stop = s }
}

func applyOpts(opts []CompleteOption) CompleteConfig {
	var c CompleteConfig
	for _, o := range opts {
		o(&c)
	}
	return c
}

// openaiBackend 是 OpenAI / Ollama（兼容协议）的共享后端。
type openaiBackend struct {
	client openai.Client
	model  string
}

func newOpenAIBackend(baseURL, apiKey, model string) *openaiBackend {
	return &openaiBackend{
		client: openai.NewClient(
			option.WithAPIKey(apiKey),
			option.WithBaseURL(baseURL),
		),
		model: model,
	}
}

func (b *openaiBackend) complete(ctx context.Context, prompt string, opts CompleteConfig) (string, error) {
	return b.chat(ctx, []openai.ChatCompletionMessageParamUnion{openai.UserMessage(prompt)}, opts)
}

func convertMessages(msgs []Message) []openai.ChatCompletionMessageParamUnion {
	out := make([]openai.ChatCompletionMessageParamUnion, 0, len(msgs))
	for _, m := range msgs {
		switch m.Role {
		case "system":
			out = append(out, openai.SystemMessage(m.Content))
		case "assistant":
			out = append(out, openai.AssistantMessage(m.Content))
		case "user":
			out = append(out, openai.UserMessage(m.Content))
		default:
			// 未知 role 当 user 处理（保持向后兼容；rag 场景里不会出）
			out = append(out, openai.UserMessage(m.Content))
		}
	}
	return out
}

func (b *openaiBackend) chat(ctx context.Context, msgs []openai.ChatCompletionMessageParamUnion, opts CompleteConfig) (string, error) {
	params := openai.ChatCompletionNewParams{
		Model:    b.model,
		Messages: msgs,
	}
	if opts.Temperature > 0 {
		params.Temperature = openai.Float(opts.Temperature)
	}
	if opts.MaxTokens > 0 {
		params.MaxTokens = openai.Int(int64(opts.MaxTokens))
	}
	if len(opts.Stop) > 0 {
		params.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: opts.Stop}
	}

	resp, err := b.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return "", fmt.Errorf("openai chat: %w", err)
	}
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("openai chat: no choices returned")
	}
	return resp.Choices[0].Message.Content, nil
}

type OpenAILLM struct {
	b *openaiBackend
}

func NewOpenAILLM(baseURL, apiKey, model string) *OpenAILLM {
	return &OpenAILLM{b: newOpenAIBackend(baseURL, apiKey, model)}
}

func (l *OpenAILLM) Complete(ctx context.Context, prompt string, opts ...CompleteOption) (string, error) {
	return l.b.complete(ctx, prompt, applyOpts(opts))
}

func (l *OpenAILLM) Chat(ctx context.Context, msgs []Message, opts ...CompleteOption) (string, error) {
	return l.b.chat(ctx, convertMessages(msgs), applyOpts(opts))
}

type OllamaLLM struct {
	b *openaiBackend
}

func NewOllamaLLM(baseURL, model string) *OllamaLLM {
	return &OllamaLLM{b: newOpenAIBackend(baseURL, "ollama", model)}
}

func (l *OllamaLLM) Complete(ctx context.Context, prompt string, opts ...CompleteOption) (string, error) {
	return l.b.complete(ctx, prompt, applyOpts(opts))
}

func (l *OllamaLLM) Chat(ctx context.Context, msgs []Message, opts ...CompleteOption) (string, error) {
	return l.b.chat(ctx, convertMessages(msgs), applyOpts(opts))
}
