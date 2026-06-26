package infrastructure

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared"
)

// Message 是 agent 消息:支持 system / user / assistant(含 tool_calls) / tool 4 种 role。
// 透传上下文时,ToolCalls 和 ToolCallID 用来拼 function calling 协议。
type Message struct {
	Role       string
	Content    string
	ToolCallID string     // role=tool 时,对应 tool_call.id
	ToolCalls  []ToolCall // role=assistant 时,模型本轮要调的工具
}

type ToolSpec struct {
	Name        string
	Description string
	Parameters  json.RawMessage // JSON Schema
}

type ToolCall struct {
	ID        string
	Name      string
	Arguments json.RawMessage
}

type ChatResponse struct {
	Content   string
	ToolCalls []ToolCall
}

type LLM interface {
	Complete(ctx context.Context, prompt string) (string, error)
	Chat(ctx context.Context, msgs []Message) (string, error)
	ChatWithTools(ctx context.Context, msgs []Message, tools []ToolSpec) (ChatResponse, error)
}

// openaiBackend 是 OpenAI / Ollama（兼容协议）的共享实现,同时是 LLM 接口。
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

func (b *openaiBackend) Complete(ctx context.Context, prompt string) (string, error) {
	return b.chat(ctx, []openai.ChatCompletionMessageParamUnion{openai.UserMessage(prompt)})
}

func (b *openaiBackend) Chat(ctx context.Context, msgs []Message) (string, error) {
	return b.chat(ctx, convertMessages(msgs))
}

func (b *openaiBackend) ChatWithTools(ctx context.Context, msgs []Message, tools []ToolSpec) (ChatResponse, error) {
	return b.chatWithTools(ctx, msgs, tools)
}

func convertMessages(msgs []Message) []openai.ChatCompletionMessageParamUnion {
	out := make([]openai.ChatCompletionMessageParamUnion, 0, len(msgs))
	for _, m := range msgs {
		param, err := convertOneMessage(m)
		if err != nil {
			// 兜底: 转换失败按 user 走,避免整条消息被吞。
			out = append(out, openai.UserMessage(m.Content))
			continue
		}
		out = append(out, param)
	}
	return out
}

func convertOneMessage(m Message) (openai.ChatCompletionMessageParamUnion, error) {
	switch m.Role {
	case "system":
		return openai.SystemMessage(m.Content), nil
	case "user":
		return openai.UserMessage(m.Content), nil
	case "tool":
		return openai.ToolMessage(m.Content, m.ToolCallID), nil
	case "assistant":
		if len(m.ToolCalls) == 0 {
			return openai.AssistantMessage(m.Content), nil
		}
		tcs := make([]openai.ChatCompletionMessageToolCallUnionParam, 0, len(m.ToolCalls))
		for _, tc := range m.ToolCalls {
			tcs = append(tcs, openai.ChatCompletionMessageToolCallUnionParam{
				OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
					ID: tc.ID,
					Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
						Name:      tc.Name,
						Arguments: string(tc.Arguments),
					},
				},
			})
		}
		assistant := openai.ChatCompletionAssistantMessageParam{
			Content:   openai.ChatCompletionAssistantMessageParamContentUnion{OfString: param.NewOpt(m.Content)},
			ToolCalls: tcs,
		}
		return openai.ChatCompletionMessageParamUnion{OfAssistant: &assistant}, nil
	default:
		// 未知 role 兜底为 user。
		return openai.UserMessage(m.Content), nil
	}
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

func (b *openaiBackend) chatWithTools(ctx context.Context, msgs []Message, tools []ToolSpec) (ChatResponse, error) {
	convertedMsgs := convertMessages(msgs)
	params := openai.ChatCompletionNewParams{
		Model:    b.model,
		Messages: convertedMsgs,
	}
	for _, t := range tools {
		// Parameters 是 JSON Schema map[string]any,需 unmarshal RawMessage 进去。
		var p shared.FunctionParameters
		if len(t.Parameters) > 0 {
			_ = json.Unmarshal(t.Parameters, &p)
		}
		params.Tools = append(params.Tools, openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
			Name:        t.Name,
			Description: param.NewOpt(t.Description),
			Parameters:  p,
		}))
	}
	resp, err := b.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return ChatResponse{}, fmt.Errorf("openai chat with tools: %w", err)
	}
	if len(resp.Choices) == 0 {
		return ChatResponse{}, fmt.Errorf("openai chat with tools: no choices returned")
	}
	msg := resp.Choices[0].Message
	out := ChatResponse{Content: msg.Content}
	for _, tc := range msg.ToolCalls {
		out.ToolCalls = append(out.ToolCalls, ToolCall{
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Arguments: json.RawMessage(tc.Function.Arguments),
		})
	}
	return out, nil
}
