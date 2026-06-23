package infrastructure

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
	Complete(ctx context.Context, prompt string) (string, error)
	Chat(ctx context.Context, msgs []Message) (string, error)
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

func (b *openaiBackend) complete(ctx context.Context, prompt string) (string, error) {
	return b.chat(ctx, []openai.ChatCompletionMessageParamUnion{openai.UserMessage(prompt)})
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
			// 未知 role 兜底为 user,避免上游拼错时整条消息被吞。
			out = append(out, openai.UserMessage(m.Content))
		}
	}
	return out
}

func (b *openaiBackend) chat(ctx context.Context, msgs []openai.ChatCompletionMessageParamUnion) (string, error) {
	params := openai.ChatCompletionNewParams{
		Model:    b.model,
		Messages: msgs,
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

func (l *OpenAILLM) Complete(ctx context.Context, prompt string) (string, error) {
	return l.b.complete(ctx, prompt)
}

func (l *OpenAILLM) Chat(ctx context.Context, msgs []Message) (string, error) {
	return l.b.chat(ctx, convertMessages(msgs))
}

type OllamaLLM struct {
	b *openaiBackend
}

func NewOllamaLLM(baseURL, model string) *OllamaLLM {
	return &OllamaLLM{b: newOpenAIBackend(baseURL, "ollama", model)}
}

func (l *OllamaLLM) Complete(ctx context.Context, prompt string) (string, error) {
	return l.b.complete(ctx, prompt)
}

func (l *OllamaLLM) Chat(ctx context.Context, msgs []Message) (string, error) {
	return l.b.chat(ctx, convertMessages(msgs))
}
