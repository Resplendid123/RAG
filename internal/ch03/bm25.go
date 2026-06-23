package ch03

import (
	"context"

	"gorm.io/gorm"
)

// BM25TopN 走 paradedb pg_search BM25 索引,返回 topN。
// @@@ 是 BM25 匹配算子,paradedb.score(id) 拿 BM25 分
func BM25TopN(ctx context.Context, db *gorm.DB, q string, topN int) ([]Hit, error) {
	if topN <= 0 {
		topN = 3
	}
	rows, err := db.WithContext(ctx).Raw(`
		SELECT id, content, paradedb.score(id) AS rank
		FROM document_chunks
		WHERE content @@@ ?
		ORDER BY rank DESC
		LIMIT ?`, q, topN).Rows()
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Hit
	rank := 0
	for rows.Next() {
		var h Hit
		var score float64
		if err := rows.Scan(&h.ChunkID, &h.Content, &score); err != nil {
			return nil, err
		}
		h.Source = "bm25"
		h.Rank = rank
		rank++
		out = append(out, h)
	}
	return out, rows.Err()
}
