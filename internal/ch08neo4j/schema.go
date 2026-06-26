package ch08neo4j

import (
	"context"
	"fmt"

	"github.com/neo4j/neo4j-go-driver/v6/neo4j"
	"gorm.io/gorm"

	"rag/infrastructure"
)

// migrateGraph 双侧建表:Postgres 沿用 BaseSchemaSQL,Neo4j 建 Entity 唯一约束。
func migrateGraph(ctx context.Context, db *gorm.DB, drv neo4j.Driver) error {
	if err := db.WithContext(ctx).Exec(`CREATE EXTENSION IF NOT EXISTS vector`).Error; err != nil {
		return fmt.Errorf("pg vector extension: %w", err)
	}
	if err := db.WithContext(ctx).Exec(infrastructure.BaseSchemaSQL).Error; err != nil {
		return fmt.Errorf("pg schema: %w", err)
	}

	if drv == nil {
		return fmt.Errorf("neo4j driver is nil; set neo4j.uri in config.yaml")
	}
	session := drv.NewSession(ctx, neo4j.SessionConfig{DatabaseName: "neo4j", AccessMode: neo4j.AccessModeWrite})
	defer session.Close(ctx)
	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		// Entity 用 (name,type) 当业务主键;description 走 SET 覆盖(每次抽取可能扩写)。
		stmts := []string{
			`CREATE CONSTRAINT entity_unique IF NOT EXISTS FOR (n:Entity) REQUIRE (n.name, n.type) IS UNIQUE`,
		}
		for _, stmt := range stmts {
			if _, err := tx.Run(ctx, stmt, nil); err != nil {
				return nil, fmt.Errorf("neo4j migrate: %w (stmt=%s)", err, stmt)
			}
		}
		return nil, nil
	})
	return err
}
