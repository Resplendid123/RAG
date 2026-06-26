package infrastructure

import (
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
)

// InitLogger 把 slog 默认 logger 配成同时写 stderr 和 logs/{name}.log,每次覆盖。
// name 为空时用 "rag"。返回 file 句柄供调用方 defer Close;同时把 logger 设为 slog.Default()。
func InitLogger(name string) (*slog.Logger, *os.File, error) {
	if name == "" {
		name = "rag"
	}
	if err := os.MkdirAll("logs", 0o755); err != nil {
		return nil, nil, fmt.Errorf("mkdir logs: %w", err)
	}
	path := filepath.Join("logs", name+".log")
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return nil, nil, fmt.Errorf("open log file: %w", err)
	}
	w := io.MultiWriter(os.Stderr, f)
	h := slog.NewTextHandler(w, &slog.HandlerOptions{Level: slog.LevelInfo})
	logger := slog.New(h)
	slog.SetDefault(logger)
	return logger, f, nil
}
