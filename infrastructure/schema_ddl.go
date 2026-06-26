package infrastructure

import (
	"context"
	"fmt"

	"gorm.io/gorm"
)

// BaseSchemaSQL 是 RAG 系列的最小化 DDL:documents + document_chunks_parent + document_chunks + HNSW 索引。
// ch02~ch08 通用基础;各自在此之上加 BM25 索引 / kg_* 表 / etc。
const BaseSchemaSQL = `DROP TABLE IF EXISTS document_chunks CASCADE;
DROP TABLE IF EXISTS document_chunks_parent CASCADE;
DROP TABLE IF EXISTS documents CASCADE;

CREATE TABLE documents (
    id          BIGSERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    source_url  TEXT,
    lang        TEXT DEFAULT 'zh',
    content_hash TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE document_chunks_parent (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    token_count INTEGER,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE document_chunks (
    id          BIGSERIAL PRIMARY KEY,
    parent_id   BIGINT REFERENCES document_chunks_parent(id) ON DELETE CASCADE,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content     TEXT NOT NULL,
    embedding   VECTOR(1024),
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX document_chunks_embedding_hnsw_idx
    ON document_chunks USING hnsw (embedding vector_cosine_ops);
`

// HybridSchemaSQL 在 BaseSchemaSQL 之上加 BM25 索引(paradedb pg_search),给 L3~L7 用。
const HybridSchemaSQL = BaseSchemaSQL + `
    CREATE INDEX document_chunks_bm25_idx
        ON document_chunks
        USING bm25 (id, content)
        WITH (key_field='id');
`

// EnsureHybridSchema 跑 hybrid schema DDL + 启用 pg_search 扩展。L3~L7 的 migrate 收口到这。
func EnsureHybridSchema(ctx context.Context, db *gorm.DB) error {
	if err := db.WithContext(ctx).Exec(`CREATE EXTENSION IF NOT EXISTS pg_search`).Error; err != nil {
		return fmt.Errorf("create extension pg_search: %w", err)
	}
	return db.WithContext(ctx).Exec(HybridSchemaSQL).Error
}
