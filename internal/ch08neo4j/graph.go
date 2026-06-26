package ch08neo4j

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/neo4j/neo4j-go-driver/v6/neo4j"

	"rag/infrastructure"
	"rag/internal"
	"rag/internal/ch02/splitter"
	"rag/internal/ch03"
	"rag/internal/ragcore"
	"rag/sample"
)

func init() {
	internal.Register(internal.Lesson{
		Name:        "neo4j",
		Description: "L8-neo4j: Graph RAG (Neo4j 存图 + Postgres 存向量)",
		Migrate:     migrateLesson,
		Run:         runNeo4j,
	})
}

var l8Neo4jSample = sample.BlueWhale

// migrateLesson 双侧 DDL:Postgres(chunks + vector) + Neo4j(Entity 唯一约束)。
func migrateLesson(ctx context.Context, deps internal.Deps) error {
	if deps.Neo4j == nil {
		return fmt.Errorf("neo4j driver not initialized; set neo4j.uri in config.yaml")
	}
	return migrateGraph(ctx, deps.DB, deps.Neo4j)
}

// runNeo4j 索引一次(ingest → extract),跑 basic + local 两个 demo + 一次 router。
func runNeo4j(ctx context.Context, deps internal.Deps, _ []string) error {
	if deps.Neo4j == nil {
		return fmt.Errorf("neo4j driver not initialized; set neo4j.uri in config.yaml")
	}

	parentCfg := splitter.DefaultConfig()
	childCfg := splitter.DefaultConfig()
	childCfg.ChunkSize = 200
	childCfg.ChunkOverlap = 0

	pc := splitter.SplitParentChild(l8Neo4jSample, parentCfg, childCfg)
	slog.Info(fmt.Sprintf("[INDEXING] parent + child → %d chunks (across %d parents)",
		len(pc.Children), len(pc.Parents)))

	if err := ch03.Ingest(ctx, deps.DB, deps.Embedder,
		infrastructure.Document{Title: "L8-neo4j sample", Lang: "zh"},
		pc.Parents, pc.Children,
	); err != nil {
		return fmt.Errorf("ingest: %w", err)
	}

	haveEnts, err := countEntities(ctx, deps.Neo4j)
	if err != nil {
		return err
	}
	if haveEnts == 0 {
		if err := extractFromChunks(ctx, deps.DB, deps.Neo4j, deps.LLM, pc.Children); err != nil {
			return fmt.Errorf("extract: %w", err)
		}
	} else {
		slog.Info(fmt.Sprintf("[INDEXING] entities already exist (%d), skip extract", haveEnts))
	}

	type demo struct {
		question string
		mode     QueryMode
	}
	demos := []demo{
		{"蓝鲸 2024 赛季最终排名如何?", ModeBasic},
		{"陈昊本赛季表现和奖项情况?", ModeLocal},
	}
	for i, d := range demos {
		slog.Info(fmt.Sprintf("\n[RUN %d] mode=%s  %q", i+1, d.mode, d.question))
		var ans string
		var err error
		switch d.mode {
		case ModeBasic:
			ans, err = BasicSearch(ctx, deps.DB, deps.Embedder, deps.LLM, d.question, 5)
		case ModeLocal:
			ans, err = LocalSearch(ctx, deps.DB, deps.Neo4j, deps.Embedder, deps.LLM, d.question)
		default:
			err = fmt.Errorf("unknown mode %s", d.mode)
		}
		if err != nil {
			slog.Error(fmt.Sprintf("[ERROR] %v", err))
			continue
		}
		slog.Info(fmt.Sprintf("[ANSWER]\n%s", ragcore.Snippet(ragcore.StripThink(ans), 800)))
	}

	routedQ := "性能优化方面有哪些建议?"
	slog.Info(fmt.Sprintf("\n[ROUTER] %q → mode=%s", routedQ, routeQuery(routedQ)))
	return nil
}

// countEntities 用 Cypher COUNT,判断是否要重跑 extract。
func countEntities(ctx context.Context, drv neo4j.Driver) (int64, error) {
	sess := drv.NewSession(ctx, neo4j.SessionConfig{DatabaseName: "neo4j", AccessMode: neo4j.AccessModeRead})
	defer sess.Close(ctx)
	rows, err := sess.Run(ctx, `MATCH (:Entity) RETURN count(*) AS c`, nil)
	if err != nil {
		return 0, err
	}
	if rows.Next(ctx) {
		v, _ := rows.Record().Get("c")
		switch x := v.(type) {
		case int64:
			return x, nil
		case int:
			return int64(x), nil
		}
	}
	return 0, rows.Err()
}
