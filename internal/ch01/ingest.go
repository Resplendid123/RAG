package ch01

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"log/slog"
	"strings"

	"github.com/pgvector/pgvector-go"
	"gorm.io/gorm"

	"rag/infrastructure"
)

// Ingest 把 doc + chunks 写入 documents / document_chunks。批 embedding,单事务。
// 同一份 chunks 第二次写入会被 content_hash 唯一约束拦下,跳过 chunks 插入。
func Ingest(ctx context.Context, db *gorm.DB, emb infrastructure.Embedder, doc infrastructure.Document, chunks []string) error {
	if len(chunks) == 0 {
		return nil
	}
	sum := sha256.Sum256([]byte(strings.Join(chunks, "\n")))
	hash := hex.EncodeToString(sum[:])

	slog.Info("embedding", "n", len(chunks))
	vecs, err := emb.Embed(ctx, chunks)
	if err != nil {
		return err
	}
	return db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var docID int64
		err := tx.Raw(
			`INSERT INTO documents(title, source_url, lang, content_hash) VALUES (?,?,?,?)
             ON CONFLICT (content_hash) DO NOTHING
             RETURNING id`,
			doc.Title, doc.SourceURL, doc.Lang, hash,
		).Scan(&docID).Error
		if err != nil {
			return err
		}
		if docID == 0 {
			slog.Info("document already exists, skipping")
			return nil
		}
		for i, c := range chunks {
			if err := tx.Exec(
				`INSERT INTO document_chunks (document_id, chunk_index, content, embedding) VALUES (?,?,?,?)`,
				docID, i, c, pgvector.NewVector(vecs[i]),
			).Error; err != nil {
				return err
			}
		}
		return nil
	})
}
