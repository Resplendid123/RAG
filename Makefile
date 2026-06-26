.PHONY: up down run migrate run-l1 run-l2 run-l3 run-l4 run-l5 run-l6 run-l7 run-l8 run-l8-neo4j run-l9 run-l10 fix

up:
	docker compose up -d

down:
	docker compose down -v

migrate:
	go run ./cmd migrate --all

run:
	go run ./cmd $(ARGS)

fix:
	go fix ./...

run-l1:
	go run ./cmd migrate naive
	go run ./cmd naive

run-l2:
	go run ./cmd migrate chunking
	go run ./cmd chunking

run-l3:
	go run ./cmd migrate hybrid
	go run ./cmd hybrid

run-l4:
	go run ./cmd migrate rerank
	go run ./cmd rerank

run-l5:
	go run ./cmd migrate query
	go run ./cmd query

run-l6:
	go run ./cmd migrate pipeline
	go run ./cmd pipeline

run-l7:
	go run ./cmd migrate agentic
	go run ./cmd agentic

run-l8:
	go run ./cmd migrate graph
	go run ./cmd graph

run-l8-neo4j:
	go run ./cmd migrate neo4j
	go run ./cmd neo4j

run-l9:
	go run ./cmd eval

run-l10:
	go run ./cmd serving
