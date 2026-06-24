package agent

import (
	"context"
	"encoding/json"
	"fmt"
)

// Tool 喂给 LLM 的工具描述 + 执行函数。ACI 设计:Description 要写边界,Parameters 用 JSON Schema 强约束。
type Tool struct {
	Name        string
	Description string
	Parameters  json.RawMessage
	Fn          func(ctx context.Context, args json.RawMessage) (any, error)
}

// Registry 持有所有可用 tool。LLM 通过 Specs() 拿到所有工具描述去选。
type Registry struct {
	tools map[string]Tool
}

func NewRegistry() *Registry { return &Registry{tools: map[string]Tool{}} }

func (r *Registry) Register(t Tool) { r.tools[t.Name] = t }

func (r *Registry) Get(name string) (Tool, bool) { t, ok := r.tools[name]; return t, ok }

func (r *Registry) Execute(ctx context.Context, name string, args json.RawMessage) (any, error) {
	t, ok := r.tools[name]
	if !ok {
		return nil, fmt.Errorf("unknown tool: %s", name)
	}
	return t.Fn(ctx, args)
}

func (r *Registry) Specs() []Tool {
	out := make([]Tool, 0, len(r.tools))
	for _, t := range r.tools {
		out = append(out, t)
	}
	return out
}
