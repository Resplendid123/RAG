package rag

import (
	"context"
	"log/slog"

	"gorm.io/gorm"

	"rag/internal"
)

type Deps struct {
	DB       *gorm.DB
	LLM      internal.LLM
	Embedder internal.Embedder
	Cfg      *internal.Config
	Logger   *slog.Logger
}

type Lesson struct {
	Name        string
	Description string
	Migrate     func(ctx context.Context, db *gorm.DB) error
	Run         func(ctx context.Context, deps Deps, args []string) error
}

var lessons []Lesson

func Register(l Lesson) { lessons = append(lessons, l) }
func All() []Lesson     { return lessons }
