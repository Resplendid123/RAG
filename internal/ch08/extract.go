package ch08

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/pgvector/pgvector-go"
	"gorm.io/gorm"

	"rag/infrastructure"
	"rag/internal/ch02/splitter"
	"rag/internal/kg/extract"
	"rag/internal/ragcore"
)

// extractFromChunks 两遍走:第一遍逐 chunk 调 LLM,只存内存;第二遍一次性写 entities/relations/mentions。
// 内存聚合能避免"每 chunk 一次 upsert + embed",同一实体多次出现合并后只 embed 一次。
func extractFromChunks(ctx context.Context, db *gorm.DB, llm infrastructure.LLM, emb infrastructure.Embedder, chunks []splitter.ChildChunk) error {
	chunkIDs, err := ragcore.LoadChildChunkIDs(ctx, db)
	if err != nil {
		return err
	}
	if len(chunkIDs) != len(chunks) {
		return fmt.Errorf("chunk id count mismatch: chunks=%d ids=%d", len(chunks), len(chunkIDs))
	}

	type entAgg struct {
		entity Entity
		tus    map[int64]struct{}
	}
	agg := make(map[string]*entAgg)
	type relAgg struct {
		rel         Relation
		description string
	}
	relMap := make(map[string]*relAgg)

	slog.Info(fmt.Sprintf("[EXTRACT] %d chunks → LLM\n", len(chunks)))
	for i, c := range chunks {
		raw, err := llm.Complete(ctx, fmt.Sprintf(extract.PromptTpl, c.Content))
		if err != nil {
			slog.Info(fmt.Sprintf("            chunk %d: LLM call failed: %v\n", i+1, err))
			continue
		}
		res, parseErr := extract.Parse(ragcore.StripThink(raw))
		if parseErr != nil {
			slog.Info(fmt.Sprintf("            chunk %d: %v\n", i+1, parseErr))
			continue
		}
		slog.Info(fmt.Sprintf("            chunk %d: %d ents, %d rels\n", i+1, len(res.Entities), len(res.Relations)))
		for _, e := range res.Entities {
			slog.Info(fmt.Sprintf("              ent: %s (%s) — %s\n", e.Name, e.Type, e.Description))
		}
		for _, r := range res.Relations {
			slog.Info(fmt.Sprintf("              rel: %s → %s — %s\n", r.Source, r.Target, r.Description))
		}

		localKey := make(map[string]string, len(res.Entities))
		for _, e := range res.Entities {
			name := strings.TrimSpace(e.Name)
			if name == "" {
				continue
			}
			k := extract.EntityKey(name, e.Type)
			localKey[name] = k
			a, ok := agg[k]
			if !ok {
				a = &entAgg{entity: Entity{Name: name, Type: e.Type, Description: e.Description}, tus: map[int64]struct{}{}}
				agg[k] = a
			}
			if len(e.Description) > len(a.entity.Description) {
				a.entity.Description = e.Description
			}
			a.tus[chunkIDs[i]] = struct{}{}
		}
		for _, rel := range res.Relations {
			srcKey, ok := localKey[strings.TrimSpace(rel.Source)]
			if !ok {
				continue
			}
			tgtKey, ok := localKey[strings.TrimSpace(rel.Target)]
			if !ok {
				continue
			}
			if srcKey == tgtKey {
				continue
			}
			rk := srcKey + "→" + tgtKey
			if _, ok := relMap[rk]; !ok {
				relMap[rk] = &relAgg{rel: Relation{Description: rel.Description, Weight: 1}, description: rel.Description}
			} else {
				relMap[rk].rel.Weight++
				if len(rel.Description) > len(relMap[rk].description) {
					relMap[rk].description = rel.Description
				}
			}
		}
	}

	type entRow struct {
		key, name, typ, description string
	}
	rows := make([]entRow, 0, len(agg))
	for k, a := range agg {
		rows = append(rows, entRow{k, a.entity.Name, a.entity.Type, a.entity.Description})
	}
	texts := make([]string, len(rows))
	for i, r := range rows {
		texts[i] = strings.TrimSpace(r.name + " " + r.typ + " " + r.description)
	}
	slog.Info(fmt.Sprintf("[EXTRACT] embed %d unique entities\n", len(texts)))
	vecs, err := emb.Embed(ctx, texts)
	if err != nil {
		return fmt.Errorf("embed entities: %w", err)
	}
	idByKey := make(map[string]int64, len(rows))
	for i, r := range rows {
		var id int64
		err := db.WithContext(ctx).Raw(`
			INSERT INTO kg_entities(name, type, description, embedding)
			VALUES (?,?,?,?)
			ON CONFLICT (name, type) DO UPDATE
				SET description = EXCLUDED.description
			RETURNING id`,
			r.name, r.typ, r.description, pgvector.NewVector(vecs[i]),
		).Scan(&id).Error
		if err != nil {
			return fmt.Errorf("upsert entity %q: %w", r.name, err)
		}
		idByKey[r.key] = id
	}

	for k, a := range agg {
		id, ok := idByKey[k]
		if !ok {
			continue
		}
		for tuID := range a.tus {
			if err := db.WithContext(ctx).Exec(`
				INSERT INTO kg_entity_mentions(entity_id, text_unit_id)
				VALUES (?,?)
				ON CONFLICT DO NOTHING`,
				id, tuID,
			).Error; err != nil {
				return fmt.Errorf("upsert mention %d/%d: %w", id, tuID, err)
			}
		}
	}

	relFinal := make([]Relation, 0, len(relMap))
	for rk, ra := range relMap {
		parts := strings.SplitN(rk, "→", 2)
		if len(parts) != 2 {
			continue
		}
		srcID, sok := idByKey[parts[0]]
		tgtID, tok := idByKey[parts[1]]
		if !sok || !tok || srcID == tgtID {
			continue
		}
		ra.rel.SourceID = srcID
		ra.rel.TargetID = tgtID
		ra.rel.Description = ra.description
		relFinal = append(relFinal, ra.rel)
	}
	slog.Info(fmt.Sprintf("[EXTRACT] %d unique relations\n", len(relFinal)))
	for _, r := range relFinal {
		if err := db.WithContext(ctx).Exec(`
			INSERT INTO kg_relations(source_id, target_id, description, weight)
			VALUES (?,?,?,?)
			ON CONFLICT (source_id, target_id) DO UPDATE
				SET weight = kg_relations.weight + 1,
				    description = EXCLUDED.description`,
			r.SourceID, r.TargetID, r.Description, r.Weight,
		).Error; err != nil {
			return fmt.Errorf("upsert relation %d→%d: %w", r.SourceID, r.TargetID, err)
		}
	}
	return updateDegrees(ctx, db)
}

// updateDegrees 回填 degree(出度+入度),Local Search 排序时用得上。
func updateDegrees(ctx context.Context, db *gorm.DB) error {
	return db.WithContext(ctx).Exec(`
		UPDATE kg_entities e SET degree = COALESCE(sub.c, 0)
		FROM (
			SELECT entity_id, COUNT(*) AS c FROM (
				SELECT source_id AS entity_id FROM kg_relations
				UNION ALL
				SELECT target_id AS entity_id FROM kg_relations
			) t GROUP BY entity_id
		) sub
		WHERE e.id = sub.entity_id`).Error
}
