// Package sample 暴露 //go:embed 进二进制的演示语料,供各 lesson 复用。
package sample

import _ "embed"

//go:embed rag.md
var RAG string

//go:embed inferx.md
var InferX string
