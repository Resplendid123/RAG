package ch08neo4j

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/neo4j/neo4j-go-driver/v6/neo4j"
	"gorm.io/gorm"

	"rag/infrastructure"
	"rag/internal/ch02/splitter"
	"rag/internal/kg/extract"
	"rag/internal/ragcore"
)

// ========== 抽取 + 写图 ==========

type entAgg struct {
	name, typ, description string
	tus                    map[int64]struct{}
}

type relAgg struct {
	srcKey, tgtKey string
	description    string
	weight         int
}

// extractFromChunks:LLM 抽实体/关系 → 内存去重 → 一次事务写图(Chunk/Entity/RELATES/MENTIONS)。
// 实体不 embed:向量召回走 Postgres ch03.DenseSearch,Neo4j 只做图遍历。
func extractFromChunks(ctx context.Context, db *gorm.DB, drv neo4j.Driver, llm infrastructure.LLM, chunks []splitter.ChildChunk) error {
	chunkIDs, err := ragcore.LoadChildChunkIDs(ctx, db)
	if err != nil {
		return err
	}
	if len(chunkIDs) != len(chunks) {
		return fmt.Errorf("chunk id count mismatch: chunks=%d ids=%d", len(chunks), len(chunkIDs))
	}

	ents := make(map[string]*entAgg)
	rels := make(map[string]*relAgg)

	slog.Info(fmt.Sprintf("[EXTRACT] %d chunks → LLM", len(chunks)))
	for i, c := range chunks {
		raw, lerr := llm.Complete(ctx, fmt.Sprintf(extract.PromptTpl, c.Content))
		if lerr != nil {
			slog.Warn(fmt.Sprintf("            chunk %d: LLM call failed: %v", i+1, lerr))
			continue
		}
		res, parseErr := extract.Parse(ragcore.StripThink(raw))
		if parseErr != nil {
			slog.Warn(fmt.Sprintf("            chunk %d: %v", i+1, parseErr))
			continue
		}
		slog.Info(fmt.Sprintf("            chunk %d: %d ents, %d rels", i+1, len(res.Entities), len(res.Relations)))

		localKey := make(map[string]string, len(res.Entities))
		for _, e := range res.Entities {
			name := strings.TrimSpace(e.Name)
			if name == "" {
				continue
			}
			k := extract.EntityKey(name, e.Type)
			localKey[name] = k
			a, ok := ents[k]
			if !ok {
				a = &entAgg{name: name, typ: e.Type, description: e.Description, tus: map[int64]struct{}{}}
				ents[k] = a
			}
			if len(e.Description) > len(a.description) {
				a.description = e.Description
			}
			a.tus[chunkIDs[i]] = struct{}{}
		}
		for _, rel := range res.Relations {
			srcK, ok := localKey[strings.TrimSpace(rel.Source)]
			if !ok {
				continue
			}
			tgtK, ok := localKey[strings.TrimSpace(rel.Target)]
			if !ok {
				continue
			}
			if srcK == tgtK {
				continue
			}
			rk := srcK + "→" + tgtK
			if cur, ok := rels[rk]; !ok {
				rels[rk] = &relAgg{srcKey: srcK, tgtKey: tgtK, description: rel.Description, weight: 1}
			} else {
				cur.weight++
				if len(rel.Description) > len(cur.description) {
					cur.description = rel.Description
				}
			}
		}
	}

	slog.Info(fmt.Sprintf("[EXTRACT] %d unique entities, %d unique relations", len(ents), len(rels)))

	chunkRows := make([]map[string]any, len(chunkIDs))
	for i, id := range chunkIDs {
		chunkRows[i] = map[string]any{"pg_id": id, "chunk_index": i}
	}
	entRows := make([]map[string]any, 0, len(ents))
	for _, a := range ents {
		entRows = append(entRows, map[string]any{
			"name": a.name, "type": a.typ, "description": a.description,
		})
	}
	var mentionRows []map[string]any
	for _, a := range ents {
		for pgID := range a.tus {
			mentionRows = append(mentionRows, map[string]any{
				"pg_id": pgID, "name": a.name, "type": a.typ,
			})
		}
	}
	relRows := make([]map[string]any, 0, len(rels))
	for _, r := range rels {
		srcA, sok := ents[r.srcKey]
		tgtA, tok := ents[r.tgtKey]
		if !sok || !tok {
			continue
		}
		relRows = append(relRows, map[string]any{
			"src_name": srcA.name, "src_type": srcA.typ,
			"tgt_name": tgtA.name, "tgt_type": tgtA.typ,
			"description": r.description, "weight": r.weight,
		})
	}

	session := drv.NewSession(ctx, neo4j.SessionConfig{DatabaseName: "neo4j", AccessMode: neo4j.AccessModeWrite})
	defer session.Close(ctx)
	_, err = session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		if _, err := tx.Run(ctx, `
			UNWIND $rows AS row
			MERGE (c:Chunk {pg_id: row.pg_id})
			SET c.chunk_index = row.chunk_index`,
			map[string]any{"rows": chunkRows}); err != nil {
			return nil, fmt.Errorf("write chunks: %w", err)
		}
		if _, err := tx.Run(ctx, `
			UNWIND $rows AS row
			MERGE (n:Entity {name: row.name, type: row.type})
			SET n.description = row.description`,
			map[string]any{"rows": entRows}); err != nil {
			return nil, fmt.Errorf("write entities: %w", err)
		}
		if len(mentionRows) > 0 {
			if _, err := tx.Run(ctx, `
				UNWIND $rows AS row
				MATCH (c:Chunk {pg_id: row.pg_id})
				MATCH (e:Entity {name: row.name, type: row.type})
				MERGE (c)-[:MENTIONS]->(e)`,
				map[string]any{"rows": mentionRows}); err != nil {
				return nil, fmt.Errorf("write mentions: %w", err)
			}
		}
		if len(relRows) > 0 {
			if _, err := tx.Run(ctx, `
				UNWIND $rows AS row
				MATCH (s:Entity {name: row.src_name, type: row.src_type})
				MATCH (t:Entity {name: row.tgt_name, type: row.tgt_type})
				MERGE (s)-[r:RELATES]->(t)
				SET r.description = row.description, r.weight = row.weight`,
				map[string]any{"rows": relRows}); err != nil {
				return nil, fmt.Errorf("write relations: %w", err)
			}
		}
		return nil, nil
	})
	return err
}
