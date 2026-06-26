package infrastructure

// Document 是 documents 表的最小化业务视图。
type Document struct {
	ID        int64
	Title     string
	SourceURL string
	Lang      string
}
