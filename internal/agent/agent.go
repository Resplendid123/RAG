package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync/atomic"
	"time"

	"rag/infrastructure"
)

// 终止条件:模型决定不调工具 / 超迭代 / 超 budget / 超时。
var (
	ErrMaxIter    = errors.New("agent: max iterations reached")
	ErrTimeBudget = errors.New("agent: time budget exhausted")
	ErrCallBudget = errors.New("agent: call budget exhausted")
	ErrToolBudget = errors.New("agent: per-tool budget exhausted")
)

// Budget 控制 agent 总成本。Consume 在每次 think 前调一次,ConsumeTool 在每次 execute 前调一次。
// 生产默认:MaxCalls=10,MaxWallTime=30s,PerToolLimit map[string]int{tool: 3}。
type Budget struct {
	MaxCalls     int
	MaxWallTime  time.Duration
	PerToolLimit map[string]int

	usedCalls atomic.Int32
	toolUsed  map[string]*atomic.Int32
	start     time.Time
}

func NewBudget(maxCalls int, maxWall time.Duration, perTool map[string]int) *Budget {
	toolUsed := make(map[string]*atomic.Int32, len(perTool))
	for k := range perTool {
		toolUsed[k] = &atomic.Int32{}
	}
	return &Budget{
		MaxCalls:     maxCalls,
		MaxWallTime:  maxWall,
		PerToolLimit: perTool,
		toolUsed:     toolUsed,
		start:        time.Now(),
	}
}

func (b *Budget) Consume() error {
	if b.MaxWallTime > 0 && time.Since(b.start) > b.MaxWallTime {
		return ErrTimeBudget
	}
	if b.MaxCalls > 0 && b.usedCalls.Add(1) > int32(b.MaxCalls) {
		return ErrCallBudget
	}
	return nil
}

func (b *Budget) ConsumeTool(name string) error {
	limit, ok := b.PerToolLimit[name]
	if !ok || limit <= 0 {
		return nil
	}
	counter, ok := b.toolUsed[name]
	if !ok {
		counter = &atomic.Int32{}
		b.toolUsed[name] = counter
	}
	if counter.Add(1) > int32(limit) {
		return ErrToolBudget
	}
	return nil
}

// Agent ReAct 主循环:每轮 think → 若有 tool_call 就 execute → 把结果回灌 message → 下一轮。
// 模型决定不调工具时返回最终答案。
type Agent struct {
	LLM     infrastructure.LLM
	Tools   *Registry
	MaxIter int
	Budget  *Budget
	System  string
}

func (a *Agent) Run(ctx context.Context, query string) (string, error) {
	if a.MaxIter <= 0 {
		a.MaxIter = 5
	}
	msgs := []infrastructure.Message{}
	if a.System != "" {
		msgs = append(msgs, infrastructure.Message{Role: "system", Content: a.System})
	}
	msgs = append(msgs, infrastructure.Message{Role: "user", Content: query})

	for i := 0; i < a.MaxIter; i++ {
		if a.Budget != nil {
			if err := a.Budget.Consume(); err != nil {
				return "", err
			}
		}
		specs := make([]infrastructure.ToolSpec, 0, len(a.Tools.Specs()))
		for _, t := range a.Tools.Specs() {
			specs = append(specs, infrastructure.ToolSpec{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
			})
		}
		resp, err := a.LLM.ChatWithTools(ctx, msgs, specs)
		if err != nil {
			return "", fmt.Errorf("think: %w", err)
		}
		// 记 assistant 消息(含 tool_calls)以维持多轮上下文
		msgs = append(msgs, infrastructure.Message{
			Role:      "assistant",
			Content:   resp.Content,
			ToolCalls: resp.ToolCalls,
		})
		if len(resp.ToolCalls) == 0 {
			return resp.Content, nil
		}
		// 执行所有 tool_call,把结果作为 tool 消息回灌
		for _, tc := range resp.ToolCalls {
			if a.Budget != nil {
				if err := a.Budget.ConsumeTool(tc.Name); err != nil {
					msgs = append(msgs, infrastructure.Message{
						Role: "tool", ToolCallID: tc.ID,
						Content: fmt.Sprintf("budget error: %v", err),
					})
					continue
				}
			}
			result, err := a.Tools.Execute(ctx, tc.Name, tc.Arguments)
			if err != nil {
				msgs = append(msgs, infrastructure.Message{
					Role: "tool", ToolCallID: tc.ID,
					Content: fmt.Sprintf("tool error: %v", err),
				})
				continue
			}
			out, _ := json.Marshal(result)
			msgs = append(msgs, infrastructure.Message{
				Role: "tool", ToolCallID: tc.ID, Content: string(out),
			})
		}
	}
	return "", ErrMaxIter
}
