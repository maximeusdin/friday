# Makefile for local backend workflow
DB_CONTAINER=neh_postgres
DB_USER=neh
DB_NAME=neh

.PHONY: up down logs ps psql schema tracking ingest verify reset

up:
	docker rm -f neh_postgres
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

metadata:
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < db/004_page_metadata.sql
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < db/005_chunk_metadata.sql
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < db/006_constraints_and_indexes.sql
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < db/007_documents_lang_and_links.sql

ingest:
	python scripts/ingest_one.py

retrieval:
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < db/008_add_chunks_text_tsv.sql

retrieval-expansion:
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < db/009_add_retrieval_runs_expansion.sql

result-sets:
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < db/010_result_sets.sql
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < db/011_result_sets_view.sql

research-sessions:
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < db/012_research_sessions.sql
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < db/013_research_sessions_view.sql

# Newer-style migrations (forward-compatible)
run-chunk-evidence:
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0010_retrieval_run_chunk_evidence.sql
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0011_retrieval_runs_immutable_receipt.sql
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0012_retrieval_runs_tsquery_text.sql

enable-pg-trgm:
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0013_enable_pg_trgm.sql

index-concordance-trigram:
	@echo "Adding trigram indexes for fuzzy expansion (may take a few minutes)..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0014_concordance_trigram_indexes.sql
	@echo "Running ANALYZE to update statistics..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -c "ANALYZE entities; ANALYZE entity_aliases;"

index-chunks-text-trigram:
	@echo "Adding GIST index to chunks.text for soft lex matching (may take several minutes)..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0015_chunks_text_trigram_index.sql
	@echo "Running ANALYZE to update statistics..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -c "ANALYZE chunks;"

eval-table:
	@echo "Creating retrieval_evaluations table (Phase 7)..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0016_retrieval_evaluations.sql

corpus-dictionary:
	@echo "Creating corpus dictionary tables..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0017_corpus_dictionary.sql

research-plans:
	@echo "Creating research_plans table (Day 9)..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0018_research_plans.sql

entities:
	@echo "Creating entity system tables (Day 9)..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0019_entities.sql

verify:
	docker exec -it $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -c "SELECT 'collections' t, COUNT(*) FROM collections UNION ALL SELECT 'documents', COUNT(*) FROM documents UNION ALL SELECT 'pages', COUNT(*) FROM pages UNION ALL SELECT 'chunks', COUNT(*) FROM chunks UNION ALL SELECT 'chunk_pages', COUNT(*) FROM chunk_pages UNION ALL SELECT 'ingest_runs', COUNT(*) FROM ingest_runs;"

# DANGER: deletes container + volume (wipes DB). Use only early in dev.
reset:
	docker compose down -v
	docker compose up -d
