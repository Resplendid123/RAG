package ch02

import (
	"context"

	"github.com/pgvector/pgvector-go"
	"gorm.io/gorm"

	"rag/infrastructure"
	"rag/internal/ch02/splitter"
)

// Retrieve 走 child embedding 命中,DISTINCT ON (p.id) 折叠到 parent 后按距离保留 topK。
func Retrieve(ctx context.Context, db *gorm.DB, emb infrastructure.Embedder, q string, topK int) ([]splitter.Chunk, error) {
	if topK <= 0 {
		topK = 3
	}
	qVecs, err := emb.Embed(ctx, []string{q})
	if err != nil {
		return nil, err
	}

	rows, err := db.WithContext(ctx).Raw(`
        SELECT DISTINCT ON (grp)
               content,
               chunk_index
        FROM (
            SELECT COALESCE(p.id, c.id) AS grp,
                   COALESCE(p.content, c.content) AS content,
                   COALESCE(p.chunk_index, c.chunk_index) AS chunk_index,
                   c.embedding
            FROM document_chunks c
            LEFT JOIN document_chunks_parent p ON p.id = c.parent_id
        ) sub
        ORDER BY grp, embedding <=> ?
        LIMIT ?`, pgvector.NewVector(qVecs[0]), topK).Rows()
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []splitter.Chunk
	for rows.Next() {
		var p splitter.Chunk
		if err := rows.Scan(&p.Content, &p.Seq); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}
