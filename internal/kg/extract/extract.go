// Package extract 是 LLM 实体/关系抽取的公共逻辑,被 ch08 (Postgres) 和 ch08neo4j 共享。
// 各 lesson 自己实现"内存聚合 + 持久化"部分,这里只做 JSON 解析 + 去重键 + 模板。
package extract

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// Result 是 LLM 抽取的原始 JSON 结构;空 Relations 不报错。
type Result struct {
	Entities  []Entity  `json:"entities"`
	Relations []RelEdge `json:"relations"`
}

type Entity struct {
	Name        string `json:"name"`
	Type        string `json:"type"`
	Description string `json:"description"`
}

type RelEdge struct {
	Source      string `json:"source"`
	Target      string `json:"target"`
	Description string `json:"description"`
}

// PromptTpl 实体/关系抽取 prompt;中文场景,各 chapter 直接复用。
const PromptTpl = `从以下文本中抽取实体(人物 / 组织 / 概念 / 技术 / 产品 / 标准等)和实体之间的关系。
要求:
1. 实体名保持原文形式,中文不要翻译成英文。
2. 描述尽量具体,1-2 句话,写明实体的关键属性或它在文中的角色。
3. 关系描述写明 source 对 target 的作用或联系。
4. 只输出 JSON,不要任何额外说明、注释、Markdown 包裹或思考过程。

JSON 格式:
{"entities":[{"name":"...","type":"...","description":"..."}],"relations":[{"source":"...","target":"...","description":"..."}]}

文本:
%s`

var codeFenceRE = regexp.MustCompile(`(?s)` + "```" + `(?:json)?\s*(\{.*?\})\s*` + "```")

// Parse 从 LLM 原始输出里抽 JSON 并反序列化。失败时返回错误。
// 兼容多种包裹:```json fence / 直接 JSON / JSON 前后有自然语言。
func Parse(raw string) (Result, error) {
	cleaned := codeFenceRE.FindStringSubmatch(raw)
	if len(cleaned) >= 2 {
		raw = cleaned[1]
	}
	if i := strings.Index(raw, "{"); i >= 0 {
		if j := strings.LastIndex(raw, "}"); j > i {
			raw = raw[i : j+1]
		}
	}
	var out Result
	if err := json.Unmarshal([]byte(raw), &out); err == nil {
		return out, nil
	}
	return out, fmt.Errorf("parse extract json (raw=%s)", raw[:min(200, len(raw))])
}

// EntityKey 实体唯一键;name+type 拼接(类型冲突时区分同名不同类)。
func EntityKey(name, typ string) string {
	return strings.TrimSpace(name) + "|" + strings.TrimSpace(typ)
}
