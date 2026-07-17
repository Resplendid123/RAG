package ch10

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	"rag/infrastructure"
	"rag/internal"
	"rag/internal/ch08neo4j"
	"rag/internal/ragcore"
)

func init() {
	internal.Register(internal.Lesson{
		Name:        "otel",
		Description: "L10: Observability RAG (OpenTelemetry + Langfuse)",
		Migrate:     nil,
		Run:         runServing,
	})
}

// l10Query 跟 L9 用同一题,方便 trace 直接对照 RAGAS 评分。
const l10Query = "马丁内斯的战术调整和赛季关键节点有哪些?"

func runServing(ctx context.Context, deps internal.Deps, _ []string) error {
	// 1) 配 tracer:OTEL_EXPORTER_OTLP_ENDPOINT 设了 → OTLP HTTP(Langfuse 用);
	//    没设 → stdout,本地 demo 也能直接看 span。
	endpoint := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
	exporter := "stdout"
	if endpoint != "" {
		exporter = "otlp-http " + endpoint
	}
	shutdown, err := infrastructure.InitTracer(ctx, "rag-l10")
	if err != nil {
		return fmt.Errorf("init tracer: %w", err)
	}
	defer func() {
		if err := shutdown(ctx); err != nil {
			slog.Info(fmt.Sprintf("[OTEL] shutdown error: %v", err))
		}
	}()
	fmt.Printf("[OTEL] exporter=%s service=rag-l10\n", exporter)

	// 2) ch10 复用 ch08neo4j 的 schema + Neo4j 图,跟前置 lesson 共用。
	if err := deps.DB.WithContext(ctx).Exec(`SET LOCAL search_path TO l_neo4j, public`).Error; err != nil {
		return fmt.Errorf("set search_path: %w", err)
	}
	if deps.Neo4j == nil {
		return fmt.Errorf("neo4j driver is nil; set neo4j.uri in config.yaml")
	}

	// 3) 跑一个真实 query,LocalSearch 内部 3 个 span(retrieve / kg_expand / llm_answer)。
	fmt.Printf("[RUN 1] query=%q mode=l8neo4j_local\n", l10Query)
	ans, err := ch08neo4j.LocalSearch(ctx, deps.DB, deps.Neo4j, deps.Embedder, deps.LLM, l10Query)
	if err != nil {
		return fmt.Errorf("local search: %w", err)
	}
	fmt.Printf("[ANSWER]\n%s\n\n", ragcore.Snippet(ragcore.StripThink(ans), 800))

	// 4) 给一个常用查询(便于肉眼对照 L9 那条 trace 的 LLM 评分)。
	fmt.Println("[HINT] 配 Langfuse:")
	fmt.Println("       export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:3000/api/public/otel")
	fmt.Println("       export OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <base64(public_key:secret_key)>")
	fmt.Println("       docker compose -f deploy/langfuse/docker-compose.yml up -d")
	fmt.Println("       → http://localhost:3000 看 trace")
	return nil
}
