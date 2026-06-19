.PHONY: up down run migrate run-L1

up:
	docker compose up -d

down:
	docker compose down -v

migrate:
	go run ./cmd migrate --all

run:
	go run ./cmd $(ARGS)

run-l1:
	go run ./cmd naive --q '什么是RAG'

fix:
	go fix ./...