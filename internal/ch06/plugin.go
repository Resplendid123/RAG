package ch06

import (
	"context"
	"fmt"
	"time"
)

// Plugin 是流水线的可插拔单元。ActivationEvents 声明挂到哪些 EventType;
// OnEvent 内调用 next() 让链上下一个 plugin 继续,不调用即短路。
type Plugin interface {
	ActivationEvents() []EventType
	OnEvent(ctx context.Context, et EventType, cm *ChatManage, next func() error) error
}

// EventManager 按 EventType 持有 plugin 列表,Trigger 时把列表编译成 next-call chain。
type EventManager struct {
	listeners map[EventType][]Plugin
}

func NewEventManager() *EventManager {
	return &EventManager{listeners: map[EventType][]Plugin{}}
}

// Register 把 plugin 按 ActivationEvents 挂到对应 EventType。
// 同 EventType 下的 plugin 按注册顺序串成 chain(后注册排在后)。
func (e *EventManager) Register(p Plugin) {
	for _, et := range p.ActivationEvents() {
		e.listeners[et] = append(e.listeners[et], p)
	}
}

// Trigger 执行某个 EventType 的 chain(从 i=0 到 i=n-1),用递归闭包做 next-call。
func (e *EventManager) Trigger(ctx context.Context, et EventType, cm *ChatManage) error {
	plugins := e.listeners[et]
	var run func(int) error
	run = func(i int) error {
		if i >= len(plugins) {
			return nil
		}
		return plugins[i].OnEvent(ctx, et, cm, func() error { return run(i + 1) })
	}
	return run(0)
}

// TriggersForPreset 按 preset 顺序逐个 Trigger,打每步耗时 + 总耗时。
// preset 不存在返回 error,中间任何一步出错立即返回。
func (e *EventManager) TriggersForPreset(ctx context.Context, name string, cm *ChatManage) error {
	stages, ok := Pipeline[name]
	if !ok {
		return fmt.Errorf("unknown preset: %s", name)
	}
	total := time.Now()
	for _, et := range stages {
		step := time.Now()
		if err := e.Trigger(ctx, et, cm); err != nil {
			return fmt.Errorf("event %s: %w", et, err)
		}
		fmt.Printf("[PLUGIN] %-22s %s\n", et, time.Since(step).Round(time.Millisecond))
	}
	fmt.Printf("[TOTAL]  %s\n", time.Since(total).Round(time.Millisecond))
	return nil
}
