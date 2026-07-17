package ch08neo4j

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/neo4j/neo4j-go-driver/v6/neo4j"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"gorm.io/gorm"

	"rag/infrastructure"
	"rag/internal/ch02/splitter"
	"rag/internal/ch03"
	"rag/internal/ragcore"
)

const localTopK = 5

type QueryMode string

const (
	ModeBasic QueryMode = "basic"
	ModeLocal QueryMode = "local"
)

var globalKeywords = []string{"总结", "概括", "整体", "全文", "主要", "主题", "趋势", "全局", "概况", "都讲了什么"}

// routeQuery 关键词分类:命中"全局"语义词也仍走 Local(本 lesson 只实现 basic + local),保留 router 演示。
func routeQuery(q string) QueryMode {
	_ = globalKeywords
	return ModeLocal
}

// ========== Basic Search ==========

const basicPromptTpl = `基于以下参考资料回答问题。若资料不足,请回答"我不知道"。

参考资料:
%s

问题:%s
答案:`

func BasicSearch(ctx context.Context, db *gorm.DB, emb infrastructure.Embedder, llm infrastructure.LLM, q string, topK int) (string, error) {
	hits, err := ch03.DenseSearch(ctx, db, emb, q, topK)
	if err != nil {
		return "", fmt.Errorf("basic dense: %w", err)
	}
	chunks, err := ch03.LoadChunks(ctx, db, hits)
	if err != nil {
		return "", fmt.Errorf("basic load: %w", err)
	}
	slog.Info(fmt.Sprintf("            dense top-%d → %d chunks", topK, len(chunks)))
	return llm.Complete(ctx, fmt.Sprintf(basicPromptTpl, ragcore.FormatNumbered(chunks), q))
}

// ========== Local Search ==========

const localPromptTpl = `基于以下知识图谱信息回答用户问题(中文)。优先依据参考资料;若信息不足,请回答"我不知道"。

相关实体:
%s

实体间关系:
%s

相关 chunk 文本:
%s

问题:%s
答案:`

// entityRec / relRec 与 ch08.Entity / ch08.Relation 概念一致,
// 但 Neo4j 用 name 作端点,Postgres 用 ID 作端点;两套类型共存是 store 差异的合理表达。
type entityRec struct {
	Name        string
	Type        string
	Description string
}

type relRec struct {
	Source      string
	Target      string
	Description string
	Weight      int
}

// LocalSearch:Postgres dense 召回 top-K chunks → Neo4j 1 跳邻居(沿 MENTIONS→Entity→RELATES→Entity) → 回 Postgres 拿 chunk 文本。
// L10 起在 retrieve / kg_expand / llm_answer 三个 OTel span,带 hits / seeds / tokens 等 attribute,
// 不接 tracer 时 span 是 no-op,不影响 L8 调用。
func LocalSearch(ctx context.Context, db *gorm.DB, drv neo4j.Driver, emb infrastructure.Embedder, llm infrastructure.LLM, q string) (string, error) {
	tr := infrastructure.Tracer()
	ctx, rootSpan := tr.Start(ctx, "local_search", trace.WithAttributes(
		attribute.String("query", q),
		attribute.String("variant", "l8neo4j_local"),
	))
	defer rootSpan.End()

	// 1) retrieve: dense top-K
	var hits []ch03.Hit
	err := func() error {
		ctx, span := tr.Start(ctx, "retrieve", trace.WithAttributes(
			attribute.String("query", q),
			attribute.Int("top_k", localTopK),
		))
		defer span.End()
		var e error
		hits, e = ch03.DenseSearch(ctx, db, emb, q, localTopK)
		if e != nil {
			span.RecordError(e)
			span.SetStatus(codes.Error, "dense search failed")
			return e
		}
		span.SetAttributes(attribute.Int("hits", len(hits)))
		return nil
	}()
	if err != nil {
		rootSpan.RecordError(err)
		return "", fmt.Errorf("local dense: %w", err)
	}

	chunkIDs := make([]int64, len(hits))
	for i, h := range hits {
		chunkIDs[i] = h.ChunkID
	}

	session := drv.NewSession(ctx, neo4j.SessionConfig{DatabaseName: "neo4j", AccessMode: neo4j.AccessModeRead})
	defer session.Close(ctx)

	// 2) kg_expand: Neo4j 1 跳 + 回 Postgres 拉 chunk 文本
	var (
		seeds, allEnts []entityRec
		rels           []relRec
		chunkIDsAll    []int64
		chunks         []splitter.Chunk
	)
	err = func() error {
		ctx, span := tr.Start(ctx, "kg_expand")
		defer span.End()
		var e error
		seeds, allEnts, rels, chunkIDsAll, e = graphLocalOneHop(ctx, session, chunkIDs)
		if e != nil {
			span.RecordError(e)
			span.SetStatus(codes.Error, "graph expand failed")
			return e
		}
		finalHits := make([]ch03.Hit, len(chunkIDsAll))
		for i, id := range chunkIDsAll {
			finalHits[i] = ch03.Hit{ChunkID: id, Rank: i}
		}
		chunks, e = ch03.LoadChunks(ctx, db, finalHits)
		if e != nil {
			span.RecordError(e)
			span.SetStatus(codes.Error, "load chunks failed")
			return fmt.Errorf("local load chunks: %w", e)
		}
		span.SetAttributes(
			attribute.Int("seeds", len(seeds)),
			attribute.Int("neighbors", len(allEnts)-len(seeds)),
			attribute.Int("relations", len(rels)),
			attribute.Int("chunks", len(chunks)),
		)
		return nil
	}()
	if err != nil {
		rootSpan.RecordError(err)
		return "", err
	}
	slog.Info(fmt.Sprintf("            seeds=%d, neighbors=%d, chunks=%d, relations=%d",
		len(seeds), len(allEnts)-len(seeds), len(chunks), len(rels)))

	prompt := fmt.Sprintf(localPromptTpl,
		formatEntities(allEnts),
		formatRelations(rels),
		ragcore.FormatNumbered(chunks),
		q,
	)

	// 3) llm_answer
	ctx, llmSpan := tr.Start(ctx, "llm_answer", trace.WithAttributes(
		attribute.Int("prompt_chars", len(prompt)),
		attribute.Int("entities_count", len(allEnts)),
		attribute.Int("chunks_count", len(chunks)),
	))
	defer llmSpan.End()
	out, err := llm.Complete(ctx, prompt)
	if err != nil {
		llmSpan.RecordError(err)
		llmSpan.SetStatus(codes.Error, "llm complete failed")
		rootSpan.RecordError(err)
		return "", err
	}
	llmSpan.SetAttributes(attribute.Int("response_chars", len(out)))
	return out, nil
}

// graphLocalOneHop 在一个 ExecuteRead 事务里跑 3 段单跳 Cypher,返回 seed / all-ents / rels / chunk-ids。
func graphLocalOneHop(
	ctx context.Context,
	session neo4j.Session,
	seedChunkIDs []int64,
) (seeds, allEnts []entityRec, rels []relRec, chunkIDs []int64, err error) {
	_, rerr := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		seedEnts, e := collectEntities(ctx, tx, `
			UNWIND $ids AS pid
			MATCH (c:Chunk {pg_id: pid})-[:MENTIONS]->(e:Entity)
			RETURN DISTINCT e.name AS name, e.type AS type, coalesce(e.description, '') AS description`,
			map[string]any{"ids": seedChunkIDs})
		if e != nil {
			return nil, fmt.Errorf("load seed entities: %w", e)
		}
		seeds = seedEnts
		seedNames := make([]string, len(seedEnts))
		for i, s := range seedEnts {
			seedNames[i] = s.Name
		}
		if len(seedNames) == 0 {
			allEnts = seedEnts
			return nil, nil
		}
		extraEnts, e := collectEntities(ctx, tx, `
			UNWIND $names AS n
			MATCH (e:Entity {name: n})-[:RELATES]->(m:Entity)
			WHERE m.name <> n
			RETURN DISTINCT m.name AS name, m.type AS type, coalesce(m.description, '') AS description`,
			map[string]any{"names": seedNames})
		if e != nil {
			return nil, fmt.Errorf("load neighbor entities: %w", e)
		}
		rs, e := collectRels(ctx, tx, `
			UNWIND $names AS n
			MATCH (s:Entity {name: n})-[r:RELATES]->(t:Entity)
			RETURN DISTINCT s.name AS s, t.name AS t,
			        coalesce(r.description, '') AS d,
			        coalesce(r.weight, 1) AS w`,
			map[string]any{"names": seedNames})
		if e != nil {
			return nil, fmt.Errorf("load relations: %w", e)
		}
		rels = rs
		allEnts = append(allEnts, seedEnts...)
		seen := make(map[string]struct{}, len(seedEnts))
		for _, s := range seedEnts {
			seen[s.Name] = struct{}{}
		}
		for _, n := range extraEnts {
			if _, ok := seen[n.Name]; !ok {
				allEnts = append(allEnts, n)
				seen[n.Name] = struct{}{}
			}
		}
		allNames := make([]string, 0, len(allEnts))
		for _, e := range allEnts {
			allNames = append(allNames, e.Name)
		}
		ids, e := collectInt64(ctx, tx, `
			UNWIND $names AS n
			MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {name: n})
			RETURN DISTINCT c.pg_id AS id ORDER BY id`,
			map[string]any{"names": allNames})
		if e != nil {
			return nil, fmt.Errorf("load chunk ids: %w", e)
		}
		chunkIDs = ids
		return nil, nil
	})
	err = rerr
	return
}

func collectEntities(ctx context.Context, tx neo4j.ManagedTransaction, cypher string, params map[string]any) ([]entityRec, error) {
	rows, err := tx.Run(ctx, cypher, params)
	if err != nil {
		return nil, err
	}
	var out []entityRec
	for rows.Next(ctx) {
		rec := rows.Record()
		n, _ := rec.Get("name")
		t, _ := rec.Get("type")
		d, _ := rec.Get("description")
		name, _ := n.(string)
		typ, _ := t.(string)
		desc, _ := d.(string)
		if name == "" {
			continue
		}
		out = append(out, entityRec{Name: name, Type: typ, Description: desc})
	}
	return out, rows.Err()
}

func collectRels(ctx context.Context, tx neo4j.ManagedTransaction, cypher string, params map[string]any) ([]relRec, error) {
	rows, err := tx.Run(ctx, cypher, params)
	if err != nil {
		return nil, err
	}
	var out []relRec
	for rows.Next(ctx) {
		rec := rows.Record()
		s, _ := rec.Get("s")
		t, _ := rec.Get("t")
		d, _ := rec.Get("d")
		w, _ := rec.Get("w")
		sN, _ := s.(string)
		tN, _ := t.(string)
		desc, _ := d.(string)
		ww := 1
		switch x := w.(type) {
		case int64:
			ww = int(x)
		case int:
			ww = x
		}
		out = append(out, relRec{Source: sN, Target: tN, Description: desc, Weight: ww})
	}
	return out, rows.Err()
}

func collectInt64(ctx context.Context, tx neo4j.ManagedTransaction, cypher string, params map[string]any) ([]int64, error) {
	rows, err := tx.Run(ctx, cypher, params)
	if err != nil {
		return nil, err
	}
	var out []int64
	for rows.Next(ctx) {
		rec := rows.Record()
		v, _ := rec.Get("id")
		switch x := v.(type) {
		case int64:
			out = append(out, x)
		case int:
			out = append(out, int64(x))
		}
	}
	return out, rows.Err()
}

// ========== 格式化 ==========

func formatEntities(ents []entityRec) string {
	if len(ents) == 0 {
		return "(no entities)"
	}
	var b strings.Builder
	for _, e := range ents {
		typ := e.Type
		if typ == "" {
			typ = "?"
		}
		fmt.Fprintf(&b, "- %s (%s)\n", e.Name, typ)
	}
	return b.String()
}

func formatRelations(rels []relRec) string {
	if len(rels) == 0 {
		return "(no relations)"
	}
	var b strings.Builder
	for _, r := range rels {
		fmt.Fprintf(&b, "- %s → %s: %s (w=%d)\n", r.Source, r.Target, r.Description, r.Weight)
	}
	return b.String()
}
