package ch03

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/pgvector/pgvector-go"
	"gorm.io/gorm"

	"rag/infrastructure"
	"rag/internal/ch02"
	"rag/internal/ch02/splitter"
)

type Document struct {
	ID        int64
	Title     string
	SourceURL string
	Lang      string
}

// Ingest 把 doc + parents + children 写进 documents / document_chunks_parent / document_chunks。
func Ingest(
	ctx context.Context,
	db *gorm.DB,
	emb infrastructure.Embedder,
	doc Document,
	parents []splitter.Chunk,
	children []ch02.ChildChunk,
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
	hash := sha256Hex(joinContents(hashSrc))

	flat := make([]string, len(children))
	for i, k := range children {
		flat[i] = k.Content
	}
	fmt.Printf("[EMBEDDING] → %d child vectors\n", len(flat))
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
				docID, pi, p.Content, ch02.EstimateTokens(p.Content),
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

func sha256Hex(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

func joinContents(chunks []splitter.Chunk) string {
	parts := make([]string, len(chunks))
	for i, c := range chunks {
		parts[i] = c.Content
	}
	return strings.Join(parts, "\n")
}
