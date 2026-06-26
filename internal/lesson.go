package internal

import (
	"context"
	"log/slog"

	"github.com/neo4j/neo4j-go-driver/v6/neo4j"
	"gorm.io/gorm"

	"rag/infrastructure"
)

type Deps struct {
	DB       *gorm.DB
	Neo4j    neo4j.Driver // 可选;nil 表示未配 Neo4j,只用 Postgres 的 chapter 不受影响。
	LLM      infrastructure.LLM
	Embedder infrastructure.Embedder
	Cfg      *infrastructure.Config
	Logger   *slog.Logger
}

type Lesson struct {
	Name        string
	Description string
	Migrate     func(ctx context.Context, deps Deps) error
	Run         func(ctx context.Context, deps Deps, args []string) error
}

var lessons []Lesson

func Register(l Lesson) { lessons = append(lessons, l) }
func All() []Lesson     { return lessons }
