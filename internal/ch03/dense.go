package ch03

import (
	"context"
	"fmt"

	"github.com/pgvector/pgvector-go"
	"gorm.io/gorm"

	"rag/infrastructure"
	"rag/internal/ragcore"
)

// Hit 是 dense / bm25 单路召回的统一结构;RRF 用 ChunkID 对齐,Content 给回显/生成用。
type Hit = ragcore.Hit

// DenseSearch 走 embedding 余弦距离,返回 topN。Rank 从 0 起。
func DenseSearch(ctx context.Context, db *gorm.DB, emb infrastructure.Embedder, q string, topN int) ([]Hit, error) {
	if topN <= 0 {
		topN = 3
	}
	qVecs, err := emb.Embed(ctx, []string{q})
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}
	rows, err := db.WithContext(ctx).Raw(`
		SELECT id, content, embedding <=> ? AS dist
		FROM document_chunks
		ORDER BY dist
		LIMIT ?`, pgvector.NewVector(qVecs[0]), topN).Rows()
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Hit
	rank := 0
	for rows.Next() {
		var h Hit
		var dist float64
		if err := rows.Scan(&h.ChunkID, &h.Content, &dist); err != nil {
			return nil, err
		}
		h.Source = "dense"
		h.Rank = rank
		rank++
		out = append(out, h)
	}
	return out, rows.Err()
}
