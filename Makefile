.PHONY: up down run migrate run-l1 run-l2 run-l3 run-l4 run-l5 run-l6 run-l7 run-l8 run-l9 run-l10 fix

up:
	docker compose up -d

down:
	docker compose down -v

migrate:
	go run ./cmd migrate --all

run:
	go run ./cmd $(ARGS)

run-l1:
	go run ./cmd naive

fix:
	go fix ./...

run-l2:
	go run ./cmd chunking

run-l3:
	go run ./cmd hybrid

run-l4:
	go run ./cmd rerank

run-l5:
	go run ./cmd query

run-l6:
	go run ./cmd pipeline

run-l7:
	go run ./cmd agentic

run-l8:
	go run ./cmd graph

run-l9:
	go run ./cmd eval

run-l10:
	go run ./cmd serving
