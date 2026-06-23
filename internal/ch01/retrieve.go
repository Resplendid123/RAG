package ch01

import (
	"context"

	"github.com/pgvector/pgvector-go"
	"gorm.io/gorm"

	"rag/infrastructure"
)

// NaiveChunk 是从 document_chunks 表里读回来的一行,带最少的字段供检索 + 拼 prompt 使用。
type NaiveChunk struct {
	ID      int64
	DocID   int64
	Index   int
	Content string
}

// Retrieve 走 pgvector 余弦距离取 top-K。topK<=0 时回落到 3。
func Retrieve(ctx context.Context, db *gorm.DB, emb infrastructure.Embedder, q string, topK int) ([]NaiveChunk, error) {
	if topK <= 0 {
		topK = 3
	}
	qVecs, err := emb.Embed(ctx, []string{q})
	if err != nil {
		return nil, err
	}
	rows, err := db.WithContext(ctx).Raw(`
        SELECT id, document_id, chunk_index, content
        FROM document_chunks
        ORDER BY embedding <=> ?
        LIMIT ?`, pgvector.NewVector(qVecs[0]), topK).Rows()
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []NaiveChunk
	for rows.Next() {
		var c NaiveChunk
		if err := rows.Scan(&c.ID, &c.DocID, &c.Index, &c.Content); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}
