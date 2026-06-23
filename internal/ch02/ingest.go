package ch02

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/pgvector/pgvector-go"
	"gorm.io/gorm"

	"rag/infrastructure"
	"rag/internal/ch02/splitter"
)

// Document 是 documents 表的最小化业务视图。
type Document struct {
	ID        int64
	Title     string
	SourceURL string
	Lang      string
}

// Ingest 把 doc + parents + children 写入 documents / document_chunks_parent / document_chunks。
// children 平铺一次性批 embedding 省 round trip;parent 只记 token_count。
// parents 为空时(整篇太小,retention 把 parent 全部丢了)依然会插 children(ParentIndex = -1 → parent_id NULL),
// 否则 retrieve 就拿不到任何候选,LLM 只能答"我不知道"。
func Ingest(
	ctx context.Context,
	db *gorm.DB,
	emb infrastructure.Embedder,
	doc Document,
	parents []splitter.Chunk,
	children []ChildChunk,
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
	fmt.Printf("[EMBEDDING] → %d child vectors (across %d parents)\n", len(flat), len(parents))
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
				docID, pi, p.Content, EstimateTokens(p.Content),
			).Scan(&parentIDs[pi]).Error; err != nil {
				return err
			}
		}

		for i, k := range children {
			var parentID any
			if k.ParentIndex >= 0 && k.ParentIndex < len(parentIDs) {
				parentID = parentIDs[k.ParentIndex]
			} // ParentIndex 越界或 -1 → 孤儿 child,parent_id = NULL。
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
