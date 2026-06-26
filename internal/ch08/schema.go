package ch08

import (
	"context"
	"fmt"

	"rag/infrastructure"
	"rag/internal"
)

// migrateGraph 复 BaseSchemaSQL (documents/parent/chunks + HNSW) 再叠 4 张图谱表。
// TextUnit = document_chunks(id);实体向量用 entity 名 + type + description 拼成一段文本 embed。
// 社区只跑 level=1 单层(简化 Louvain),entity_ids 直接存 BIGINT[] 方便下游一次取。
func migrateGraph(ctx context.Context, deps internal.Deps) error {
	db := deps.DB
	if err := db.WithContext(ctx).Exec(`CREATE EXTENSION IF NOT EXISTS vector`).Error; err != nil {
		return fmt.Errorf("create extension vector: %w", err)
	}
	return db.WithContext(ctx).Exec(`
		DROP TABLE IF EXISTS kg_communities     CASCADE;
		DROP TABLE IF EXISTS kg_entity_mentions CASCADE;
		DROP TABLE IF EXISTS kg_relations       CASCADE;
		DROP TABLE IF EXISTS kg_entities        CASCADE;
	` + infrastructure.BaseSchemaSQL + `
		CREATE TABLE kg_entities (
			id            BIGSERIAL PRIMARY KEY,
			name          TEXT NOT NULL,
			type          TEXT,
			description   TEXT,
			embedding     VECTOR(1024),
			degree        INT DEFAULT 0,
			created_at    TIMESTAMPTZ DEFAULT NOW(),
			UNIQUE(name, type)
		);
		CREATE INDEX kg_entities_embedding_hnsw_idx
			ON kg_entities USING hnsw (embedding vector_cosine_ops);

		CREATE TABLE kg_relations (
			id            BIGSERIAL PRIMARY KEY,
			source_id     BIGINT NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
			target_id     BIGINT NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
			description   TEXT,
			weight        INT DEFAULT 1,
			UNIQUE(source_id, target_id)
		);

		CREATE TABLE kg_entity_mentions (
			entity_id     BIGINT NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
			text_unit_id  BIGINT NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
			PRIMARY KEY (entity_id, text_unit_id)
		);

		CREATE TABLE kg_communities (
			id            BIGSERIAL PRIMARY KEY,
			level         INT NOT NULL,
			entity_ids    BIGINT[],
			summary       TEXT,
			created_at    TIMESTAMPTZ DEFAULT NOW()
		);
	`).Error
}

type Entity struct {
	ID          int64
	Name        string
	Type        string
	Description string
	Degree      int
}

type Relation struct {
	ID          int64
	SourceID    int64
	TargetID    int64
	Description string
	Weight      int
}

type Community struct {
	ID        int64
	Level     int
	EntityIDs []int64
	Summary   string
}
