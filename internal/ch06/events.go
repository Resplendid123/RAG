package ch06

import "rag/internal/ch03"

// EventType 流水线阶段:一个 preset = 有序 EventType 列表,同 EventType 下的 plugin 串成 chain。
type EventType string

const (
	LOAD_HISTORY      EventType = "load_history"
	QUERY_UNDERSTAND  EventType = "query_understand"
	CHUNK_SEARCH      EventType = "chunk_search"
	CHUNK_RERANK      EventType = "chunk_rerank"
	CHUNK_MERGE       EventType = "chunk_merge"
	FILTER_TOP_K      EventType = "filter_top_k"
	INTO_CHAT_MESSAGE EventType = "into_chat_message"
	CHAT_COMPLETION   EventType = "chat_completion"
)

// ChatContext 是 plugin 间的共享状态:plugin A 写字段,plugin B 读字段。
// 字段读写约定写在各 plugin 顶部注释里,避免隐式依赖。
type ChatContext struct {
	Query        string     // 用户原始问题
	History      []string   // load_history 写入,query_understand / into_chat_message 读
	RewriteQuery string     // query_understand 写入,chunk_search / chunk_rerank 读
	Chunks       []ch03.Hit // chunk_search 写入
	Reranked     []ch03.Hit // chunk_rerank 写入
	Merged       []ch03.Hit // chunk_merge / filter_top_k 写入
	Prompt       string     // into_chat_message 写入
	Answer       string     // chat_completion 写入
}

// Pipeline 预设组合:加新组合 = 改一行,不动 plugin 也不动 Run 主体。
// "rag"        离线批处理(5 步,最小可用)。
// "rag_stream" 在线对话(8 步,带 history 加载 + query 改写 + top-K 截断)。
var Pipeline = map[string][]EventType{
	"rag": {
		CHUNK_SEARCH,
		CHUNK_RERANK,
		CHUNK_MERGE,
		INTO_CHAT_MESSAGE,
		CHAT_COMPLETION,
	},
	"rag_stream": {
		LOAD_HISTORY,
		QUERY_UNDERSTAND,
		CHUNK_SEARCH,
		CHUNK_RERANK,
		CHUNK_MERGE,
		FILTER_TOP_K,
		INTO_CHAT_MESSAGE,
		CHAT_COMPLETION,
	},
}
