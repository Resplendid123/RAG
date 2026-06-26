package ch08neo4j

// 对外 export 包装:ch09 复用 ch08neo4j 的 internal helper,但不破坏包内调用关系。

import (
	"context"
	"fmt"

	"github.com/neo4j/neo4j-go-driver/v6/neo4j"
	"gorm.io/gorm"

	"rag/infrastructure"
	"rag/internal/ch02/splitter"
	"rag/internal/ch03"
)

// MigrateGraph 是 migrateGraph 的 export 版本,ch09 复用 ch08neo4j 的 schema。
func MigrateGraph(ctx context.Context, db *gorm.DB, drv neo4j.Driver) error {
	return migrateGraph(ctx, db, drv)
}

// ExtractFromChunks 是 extractFromChunks 的 export 版本,ch09 复用 KG 抽取逻辑。
func ExtractFromChunks(ctx context.Context, db *gorm.DB, drv neo4j.Driver, llm infrastructure.LLM, chunks []splitter.ChildChunk) error {
	return extractFromChunks(ctx, db, drv, llm, chunks)
}

// CountEntities 是 countEntities 的 export 版本,ch09 评测前判断图是否已建。
func CountEntities(ctx context.Context, drv neo4j.Driver) (int64, error) {
	return countEntities(ctx, drv)
}

// CountMentions 统计 Neo4j 里 MENTIONS 关系数,用于检测"有 entity 但 chunk id 已失效"的脏图。
func CountMentions(ctx context.Context, drv neo4j.Driver) (int64, error) {
	sess, err := readSession(ctx, drv)
	if err != nil {
		return 0, err
	}
	defer sess.Close(ctx)
	rows, err := sess.Run(ctx, `MATCH ()-[r:MENTIONS]->() RETURN count(r) AS c`, nil)
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

// ResetGraph 清空 Neo4j 全图(Entity/Chunk/RELATES/MENTIONS),用于重建场景。
func ResetGraph(ctx context.Context, drv neo4j.Driver) error {
	sess, err := writeSession(ctx, drv)
	if err != nil {
		return err
	}
	defer sess.Close(ctx)
	_, err = sess.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		_, e := tx.Run(ctx, `MATCH (n) DETACH DELETE n`, nil)
		return nil, e
	})
	return err
}

// LocalContexts 复刻 LocalSearch 的 retrieve 步骤(只跑 dense → Neo4j 1 跳 → chunks),
// 不调 LLM,直接返回 []string contexts,给 ch09 评测时拿 RAGAS 评测用。
func LocalContexts(ctx context.Context, db *gorm.DB, drv neo4j.Driver, emb infrastructure.Embedder, q string, topK int) ([]string, error) {
	if drv == nil {
		return nil, fmt.Errorf("neo4j driver is nil")
	}
	hits, err := ch03.DenseSearch(ctx, db, emb, q, topK)
	if err != nil {
		return nil, fmt.Errorf("dense: %w", err)
	}
	chunkIDs := make([]int64, len(hits))
	for i, h := range hits {
		chunkIDs[i] = h.ChunkID
	}

	sess, err := readSession(ctx, drv)
	if err != nil {
		return nil, err
	}
	defer sess.Close(ctx)

	_, _, _, chunkIDsAll, err := graphLocalOneHop(ctx, sess, chunkIDs)
	if err != nil {
		return nil, err
	}
	finalHits := make([]ch03.Hit, len(chunkIDsAll))
	for i, id := range chunkIDsAll {
		finalHits[i] = ch03.Hit{ChunkID: id, Rank: i}
	}
	chunks, err := ch03.LoadChunks(ctx, db, finalHits)
	if err != nil {
		return nil, fmt.Errorf("load chunks: %w", err)
	}
	out := make([]string, len(chunks))
	for i, c := range chunks {
		out[i] = c.Content
	}
	return out, nil
}
