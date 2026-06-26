package ch09

import (
	"context"
	"encoding/json"
	"fmt"

	"gorm.io/gorm"

	"rag/internal"
	"rag/internal/ch08neo4j"
)

// migrateEval 调 ch08neo4j.MigrateGraph(documents + parent + chunks + HNSW + Neo4j Entity 约束)
// 再叠 3 张评测表。这样 ch09 不依赖 ch08neo4j 先跑过 migrate,自己一次建好完整 schema。
func migrateEval(ctx context.Context, deps internal.Deps) error {
	if deps.Neo4j == nil {
		return fmt.Errorf("neo4j driver is nil; set neo4j.uri in config.yaml")
	}
	if err := ch08neo4j.MigrateGraph(ctx, deps.DB, deps.Neo4j); err != nil {
		return err
	}
	db := deps.DB
	return db.WithContext(ctx).Exec(`
		CREATE TABLE IF NOT EXISTS eval_datasets (
			id          BIGSERIAL PRIMARY KEY,
			name        TEXT NOT NULL,
			config_hash TEXT NOT NULL,
			metadata    JSONB,
			created_at  TIMESTAMPTZ DEFAULT NOW(),
			UNIQUE(name, config_hash)
		);
		CREATE TABLE IF NOT EXISTS eval_runs (
			id         BIGSERIAL PRIMARY KEY,
			dataset_id BIGINT NOT NULL REFERENCES eval_datasets(id) ON DELETE CASCADE,
			run_at     TIMESTAMPTZ DEFAULT NOW(),
			summary    JSONB
		);
		CREATE TABLE IF NOT EXISTS eval_results (
			id          BIGSERIAL PRIMARY KEY,
			run_id      BIGINT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
			question_id TEXT NOT NULL,
			variant     TEXT NOT NULL,
			user_input  TEXT,
			response    TEXT,
			retrieved_contexts JSONB,
			metrics     JSONB,
			UNIQUE(run_id, question_id, variant)
		);
	`).Error
}

type EvalDataset struct {
	ID         int64
	Name       string
	ConfigHash string
	Metadata   map[string]any
	CreatedAt  string
}

type EvalRun struct {
	ID        int64
	DatasetID int64
	RunAt     string
	Summary   map[string]float64
}

type EvalResult struct {
	ID                int64
	RunID             int64
	QuestionID        string
	Variant           string
	UserInput         string
	Response          string
	RetrievedContexts []string
	Metrics           map[string]float64
}

// ensureDataset upsert 一条 eval_datasets,返回 dataset_id。name+config_hash 唯一。
func ensureDataset(ctx context.Context, db *gorm.DB, name, configHash string, meta map[string]any) (int64, error) {
	if name == "" || configHash == "" {
		return 0, fmt.Errorf("name and config_hash required")
	}
	metaJSON, _ := json.Marshal(meta)
	var id int64
	err := db.WithContext(ctx).Raw(`
		INSERT INTO eval_datasets (name, config_hash, metadata)
		VALUES (?, ?, ?::jsonb)
		ON CONFLICT (name, config_hash) DO UPDATE SET metadata = EXCLUDED.metadata
		RETURNING id`,
		name, configHash, string(metaJSON),
	).Scan(&id).Error
	return id, err
}

// createRun 落一条 eval_runs 行,返回 run_id。
func createRun(ctx context.Context, db *gorm.DB, datasetID int64) (int64, error) {
	var id int64
	err := db.WithContext(ctx).Raw(
		`INSERT INTO eval_runs (dataset_id) VALUES (?) RETURNING id`, datasetID,
	).Scan(&id).Error
	return id, err
}
