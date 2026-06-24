package ch07

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"

	"gorm.io/gorm"

	"rag/infrastructure"
	"rag/internal/agent"
	"rag/internal/ch03"
)

// registerTools 装 4 个 tool:search_documents / sql_query / webfetch / web_search。
// 每个 tool 的 Description 必须写清"适用/不适用",这是 L7 的 ACI 设计原则。
// tavilyKey 来自 config.yaml 的 web_search.tavily_api_key;空字符串时 web_search 工具注册但调用会报错。
func registerTools(r *agent.Registry, db *gorm.DB, emb infrastructure.Embedder, tavilyKey string) {
	r.Register(agent.Tool{
		Name: "search_documents",
		Description: `在企业知识库中检索相关文档段落。
适用:事实查询、政策查询、操作步骤、产品定义。
不适用:实时数据(用 sql_query)、公网信息(用 webfetch)、数学计算。
返回:命中 chunk 列表,带 chunk_id 和 content 摘要。`,
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"query": {"type": "string", "description": "检索关键词或问题原文"},
				"top_k": {"type": "integer", "default": 5, "maximum": 20}
			},
			"required": ["query"]
		}`),
		Fn: func(ctx context.Context, args json.RawMessage) (any, error) {
			var in struct {
				Query string `json:"query"`
				TopK  int    `json:"top_k"`
			}
			if err := json.Unmarshal(args, &in); err != nil {
				return nil, fmt.Errorf("parse args: %w", err)
			}
			if in.TopK <= 0 {
				in.TopK = 5
			}
			// 走 ch03 hybrid:dense + BM25 + RRF
			dense, err := ch03.DenseTopN(ctx, db, emb, in.Query, in.TopK)
			if err != nil {
				return nil, fmt.Errorf("dense: %w", err)
			}
			bm25, err := ch03.BM25TopN(ctx, db, in.Query, in.TopK)
			if err != nil {
				// BM25 语法失败时降级 dense-only
				return hitsToMap(dense), nil
			}
			fused := ch03.RRF([][]ch03.Hit{dense, bm25}, 60)
			if len(fused) > in.TopK {
				fused = fused[:in.TopK]
			}
			return hitsToMap(fused), nil
		},
	})

	r.Register(agent.Tool{
		Name: "sql_query",
		Description: `对知识库的 document_chunks 表执行只读 SQL。
适用:聚合统计(总数、按条件 count)、最新 chunk 查询。
禁止:DML/DDL(INSERT/UPDATE/DELETE/DROP 等),只允许 SELECT 或 WITH。
返回:行数组,每行是 map[string]any。

【schema】(注意列名是 id 不是 chunk_id):
- document_chunks(id BIGINT, parent_id BIGINT, document_id BIGINT, chunk_index INT, content TEXT, embedding VECTOR, created_at TIMESTAMPTZ)
- documents(id BIGINT, title TEXT, source_url TEXT, lang TEXT, content_hash TEXT, created_at TIMESTAMPTZ)
- document_chunks_parent(id BIGINT, document_id BIGINT, chunk_index INT, content TEXT, token_count INT, created_at TIMESTAMPTZ)

【常见错误】LLM 容易写 chunk_id(不存在),实际是 id。`,
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"sql": {"type": "string", "description": "只读 SELECT 或 WITH,使用 $1,$2 占位符"}
			},
			"required": ["sql"]
		}`),
		Fn: func(ctx context.Context, args json.RawMessage) (any, error) {
			var in struct {
				SQL string `json:"sql"`
			}
			if err := json.Unmarshal(args, &in); err != nil {
				return nil, fmt.Errorf("parse args: %w", err)
			}
			if !selectOnly(in.SQL) {
				return nil, fmt.Errorf("only SELECT/WITH allowed (no DML/DDL)")
			}
			// 走独立连接避开 gorm transaction 污染:一个 query 失败不会让后续 query 也炸
			return runReadOnlySQLStandalone(ctx, db, in.SQL)
		},
	})

	r.Register(agent.Tool{
		Name: "webfetch",
		Description: `抓取公网 URL 的纯文本内容(自动剥 HTML 标签 / script / style)。
适用:实时信息、Wikipedia 定义、官方文档片段。
不适用:需要 JS 渲染的页面(只取首屏 HTML)、登录后才能看的页面。
返回:页面纯文本前 2000 字。`,
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"url": {"type": "string", "description": "公网 http/https URL"}
			},
			"required": ["url"]
		}`),
		Fn: func(ctx context.Context, args json.RawMessage) (any, error) {
			var in struct {
				URL string `json:"url"`
			}
			if err := json.Unmarshal(args, &in); err != nil {
				return nil, fmt.Errorf("parse args: %w", err)
			}
			return fetchURL(ctx, in.URL)
		},
	})

	r.Register(agent.Tool{
		Name: "web_search",
		Description: `公网搜索(走 Tavily Search API,需要 API key,免费档每月 1000 次)。
适用:实时信息、最新研究动态、新闻、博客、产品发布。
不适用:需要登录的内容(用 webfetch + 具体 URL)、内网知识(用 search_documents)。
返回:最多 5 条结果(标题 + URL + snippet),带 AI 提取的相关性 score。拿到 URL 后可继续调 webfetch 抓全文。`,
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"query": {"type": "string", "description": "搜索关键词"}
			},
			"required": ["query"]
		}`),
		Fn: func(ctx context.Context, args json.RawMessage) (any, error) {
			var in struct {
				Query string `json:"query"`
			}
			if err := json.Unmarshal(args, &in); err != nil {
				return nil, fmt.Errorf("parse args: %w", err)
			}
			return tavilySearch(ctx, tavilyKey, in.Query)
		},
	})
}

func hitsToMap(hits []ch03.Hit) []map[string]any {
	out := make([]map[string]any, 0, len(hits))
	for _, h := range hits {
		preview := h.Content
		if len(preview) > 200 {
			preview = preview[:200] + "..."
		}
		out = append(out, map[string]any{
			"chunk_id": h.ChunkID,
			"source":   h.Source,
			"preview":  preview,
		})
	}
	return out
}

// selectOnly 简单 read-only 校验:去掉注释和字符串后必须以 SELECT/WITH 开头,且无 DDL/DML 关键字。
func selectOnly(sql string) bool {
	// 删行注释
	for _, line := range strings.Split(sql, "\n") {
		if i := strings.Index(line, "--"); i >= 0 {
			sql = strings.Replace(sql, line, line[:i], 1)
		}
	}
	upper := strings.ToUpper(strings.TrimSpace(sql))
	if !strings.HasPrefix(upper, "SELECT") && !strings.HasPrefix(upper, "WITH") {
		return false
	}
	for _, kw := range []string{"INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER", "CREATE", "GRANT", "REVOKE", "COPY"} {
		if strings.Contains(upper, kw) {
			// 简单检查:关键字前必须有 word boundary
			re := regexp.MustCompile(`\b` + kw + `\b`)
			if re.MatchString(upper) {
				return false
			}
		}
	}
	return true
}

func runReadOnlySQL(ctx context.Context, db *gorm.DB, sql string) (any, error) {
	rows, err := db.WithContext(ctx).Raw(sql).Rows()
	if err != nil {
		return nil, fmt.Errorf("query: %w", err)
	}
	defer rows.Close()
	cols, err := rows.Columns()
	if err != nil {
		return nil, err
	}
	var out []map[string]any
	for rows.Next() {
		vals := make([]any, len(cols))
		scan := make([]any, len(cols))
		for i := range vals {
			scan[i] = &vals[i]
		}
		if err := rows.Scan(scan...); err != nil {
			return nil, err
		}
		row := make(map[string]any, len(cols))
		for i, c := range cols {
			v := vals[i]
			if b, ok := v.([]byte); ok {
				row[c] = string(b)
			} else {
				row[c] = v
			}
		}
		out = append(out, row)
	}
	return out, rows.Err()
}

// runReadOnlySQLStandalone 在当前 tx 里走 SAVEPOINT,query 失败不会让 tx 进入 aborted 状态。
// 绕开"一个 query 失败 → 整个 tx abort → 后续 query 报 'current transaction is aborted'"。
// 必须保持在 tx 里,否则 search_path 不生效、表解析不到。
func runReadOnlySQLStandalone(ctx context.Context, gormDB *gorm.DB, query string) (any, error) {
	if err := gormDB.WithContext(ctx).Exec("SAVEPOINT l7_sql").Error; err != nil {
		return nil, fmt.Errorf("savepoint: %w", err)
	}
	rows, err := gormDB.WithContext(ctx).Raw(query).Rows()
	if err != nil {
		// 回滚到 savepoint,tx 整体仍然可用
		_ = gormDB.WithContext(ctx).Exec("ROLLBACK TO SAVEPOINT l7_sql").Error
		return nil, fmt.Errorf("query: %w", err)
	}
	defer rows.Close()
	cols, err := rows.Columns()
	if err != nil {
		_ = gormDB.WithContext(ctx).Exec("ROLLBACK TO SAVEPOINT l7_sql").Error
		return nil, err
	}
	var out []map[string]any
	for rows.Next() {
		vals := make([]any, len(cols))
		scan := make([]any, len(cols))
		for i := range vals {
			scan[i] = &vals[i]
		}
		if err := rows.Scan(scan...); err != nil {
			_ = gormDB.WithContext(ctx).Exec("ROLLBACK TO SAVEPOINT l7_sql").Error
			return nil, err
		}
		row := make(map[string]any, len(cols))
		for i, c := range cols {
			v := vals[i]
			if b, ok := v.([]byte); ok {
				row[c] = string(b)
			} else {
				row[c] = v
			}
		}
		out = append(out, row)
	}
	if err := rows.Err(); err != nil {
		_ = gormDB.WithContext(ctx).Exec("ROLLBACK TO SAVEPOINT l7_sql").Error
		return nil, err
	}
	_ = gormDB.WithContext(ctx).Exec("RELEASE SAVEPOINT l7_sql").Error
	return out, nil
}

// fetchURL HTTP GET 抓公网页面,剥 HTML / script / style 后取前 N 字。
func fetchURL(ctx context.Context, url string) (any, error) {
	if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
		return nil, fmt.Errorf("url must start with http(s)://")
	}
	cctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(cctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "rag-l7-demo/1.0")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("http get: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("http status %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 512*1024))
	if err != nil {
		return nil, err
	}
	text := stripHTML(string(body))
	if len(text) > 2000 {
		text = text[:2000] + "..."
	}
	return text, nil
}

var (
	reScript = regexp.MustCompile(`(?is)<script[^>]*>.*?</script>`)
	reStyle  = regexp.MustCompile(`(?is)<style[^>]*>.*?</style>`)
	reTag    = regexp.MustCompile(`(?s)<[^>]+>`)
	reSpan   = regexp.MustCompile(`(?s)<[^>]+>`)
	reSpaces = regexp.MustCompile(`[ \t]+`)
	reNewln  = regexp.MustCompile(`\n{3,}`)
)

func stripHTML(s string) string {
	s = reScript.ReplaceAllString(s, " ")
	s = reStyle.ReplaceAllString(s, " ")
	s = reTag.ReplaceAllString(s, " ")
	s = reSpaces.ReplaceAllString(s, " ")
	s = reNewln.ReplaceAllString(s, "\n\n")
	return strings.TrimSpace(s)
}

// tavilySearch 调 Tavily Search API (POST https://api.tavily.com/search)。
// api_key 在 config.yaml 的 web_search.tavily_api_key 配,空字符串时报错给 LLM 看到。
// 返回 results[{title, url, content, score}] + answer(AI 总结,可选)。
func tavilySearch(ctx context.Context, apiKey, query string) (any, error) {
	if strings.TrimSpace(query) == "" {
		return nil, fmt.Errorf("empty query")
	}
	if strings.TrimSpace(apiKey) == "" {
		return nil, fmt.Errorf("tavily api key not set (config.yaml web_search.tavily_api_key)")
	}
	cctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	body, _ := json.Marshal(map[string]any{
		"api_key":        apiKey,
		"query":          query,
		"max_results":    5,
		"search_depth":   "basic",
		"include_answer": false,
	})
	req, err := http.NewRequestWithContext(cctx, "POST", "https://api.tavily.com/search", strings.NewReader(string(body)))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("tavily http: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("tavily http %d: %s", resp.StatusCode, string(respBody))
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1024*1024))
	if err != nil {
		return nil, err
	}
	var out struct {
		Results []struct {
			Title   string  `json:"title"`
			URL     string  `json:"url"`
			Content string  `json:"content"`
			Score   float64 `json:"score"`
		} `json:"results"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("tavily parse: %w", err)
	}
	results := make([]map[string]any, 0, len(out.Results))
	for i, r := range out.Results {
		snippet := r.Content
		if len(snippet) > 400 {
			snippet = snippet[:400] + "..."
		}
		results = append(results, map[string]any{
			"rank":    i + 1,
			"title":   r.Title,
			"url":     r.URL,
			"snippet": snippet,
			"score":   r.Score,
		})
	}
	return map[string]any{"results": results, "count": len(results)}, nil
}
