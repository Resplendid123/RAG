# 从 Naive RAG 走到生产化

一门用 Go 写 RAG 工程实战的课。从 L1 的 4 步基线管线到 L10 的上线工程化，每节课都是可运行的 Go 代码。

## 快速开始

修改配置文件 `config.yaml` 以切换 LLM 和 embedding 模型，或连接自己的 pgvector 实例。

```bash
make up                                # 启 pgvector (docker compose)
ollama pull qwen3-embedding:0.6b       # 默认 embedding 模型
make run-l1                             # 运行 L1 基线课
```

## 课程地图

| 课  | 主题                 | 难度   | 状态   |
|----|--------------------|------|------|
| L1 | Naive RAG 基线       | ⭐    | 骨架就位 |
| L2 | 高级 Chunking 策略      | ⭐⭐   | 文档就绪 |
| L3 | Hybrid RAG（BM25 + 向量）  | ⭐⭐⭐  | 文档就绪 |
| L4 | Rerank（Cross-Encoder 精排）  | ⭐⭐⭐  | 文档就绪 |
| L5 | Query 理解与改写         | ⭐⭐⭐  | 文档就绪 |
| L6 | Pipeline（可插拔流水线编排）   | ⭐⭐⭐  | 文档就绪 |
| L7 | Agentic RAG（工具调用）  | ⭐⭐⭐⭐ | 文档就绪 |
| L8 | GraphRAG           | ⭐⭐⭐⭐ | 文档就绪 |
| L9 | 评估、可观测、可重现         | ⭐⭐⭐  | 文档就绪 |
| L10 | 上线工程化与安全治理        | ⭐⭐⭐⭐ | 文档就绪 |

每节课的代码入口在 `rag/`，文档在`docs/`。

## 目录

```
cmd/        cobra 入口，lesson runner
rag/        Lesson 注册表 + 每节课实现
internal/   跨课共享的 LLM / Embedder / Config / DB
docs/       L1–L10 课程文档
config.yaml 唯一需要改的配置文件
```
