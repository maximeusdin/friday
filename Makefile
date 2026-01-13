# Makefile for local backend workflow
DB_CONTAINER=neh_postgres
DB_USER=neh
DB_NAME=neh

.PHONY: up down logs ps psql schema tracking ingest verify reset

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker logs $(DB_CONTAINER) --tail=100

ps:
	docker ps

psql:
	docker exec -it $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME)

schema:
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < db/002_schema.sql

tracking:
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < db/003_ingest_tracking.sql

ingest:
	python scripts/ingest_one.py

verify:
	docker exec -it $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -c "SELECT 'collections' t, COUNT(*) FROM collections UNION ALL SELECT 'documents', COUNT(*) FROM documents UNION ALL SELECT 'pages', COUNT(*) FROM pages UNION ALL SELECT 'chunks', COUNT(*) FROM chunks UNION ALL SELECT 'chunk_pages', COUNT(*) FROM chunk_pages UNION ALL SELECT 'ingest_runs', COUNT(*) FROM ingest_runs;"

# DANGER: deletes container + volume (wipes DB). Use only early in dev.
reset:
	docker compose down -v
	docker compose up -d
