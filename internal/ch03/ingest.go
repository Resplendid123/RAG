package ch03

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/pgvector/pgvector-go"
	"gorm.io/gorm"

	"rag/infrastructure"
	"rag/internal/ch02/splitter"
	"rag/internal/ragcore"
)

type Document = infrastructure.Document

// Ingest 把 doc + parents + children 写进 documents / document_chunks_parent / document_chunks。
func Ingest(
	ctx context.Context,
	db *gorm.DB,
	emb infrastructure.Embedder,
	doc infrastructure.Document,
	parents []splitter.Chunk,
	children []splitter.ChildChunk,
) error {
	if len(parents) == 0 && len(children) == 0 {
		return nil
	}

	hashSrc := parents
	if len(hashSrc) == 0 {
		hashSrc = make([]splitter.Chunk, len(children))
		for i, k := range children {
			hashSrc[i] = splitter.Chunk{Content: k.Content}
		}
	}
	hash := ragcore.ContentHash(hashSrc)

	flat := make([]string, len(children))
	for i, k := range children {
		flat[i] = k.Content
	}
	slog.Info(fmt.Sprintf("[EMBEDDING] → %d child vectors\n", len(flat)))
	vecs, err := emb.Embed(ctx, flat)
	if err != nil {
		return err
	}

	return db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var docID int64
		if err := tx.Raw(
			`INSERT INTO documents(title, source_url, lang, content_hash) VALUES (?,?,?,?) RETURNING id`,
			doc.Title, doc.SourceURL, doc.Lang, hash,
		).Scan(&docID).Error; err != nil {
			return err
		}

		parentIDs := make([]int64, len(parents))
		for pi, p := range parents {
			if err := tx.Raw(
				`INSERT INTO document_chunks_parent (document_id, chunk_index, content, token_count)
				 VALUES (?,?,?,?) RETURNING id`,
				docID, pi, p.Content, splitter.EstimateTokens(p.Content),
			).Scan(&parentIDs[pi]).Error; err != nil {
				return err
			}
		}

		for i, k := range children {
			var parentID any
			if k.ParentIndex >= 0 && k.ParentIndex < len(parentIDs) {
				parentID = parentIDs[k.ParentIndex]
			}
			if err := tx.Exec(
				`INSERT INTO document_chunks (parent_id, document_id, chunk_index, content, embedding)
				 VALUES (?,?,?,?,?)`,
				parentID, docID, k.Seq, k.Content, pgvector.NewVector(vecs[i]),
			).Error; err != nil {
				return err
			}
		}
		return nil
	})
}
