# Makefile for local backend workflow
DB_CONTAINER=neh_postgres
DB_USER=neh
DB_NAME=neh

# Default values for optional parameters
AUTO_THRESHOLD ?= 85
LIMIT ?= 100
COLLECTION ?= 
MAX_CHUNKS ?= 1000

.PHONY: up down logs ps psql schema tracking metadata ingest verify reset check-docker \
	retrieval retrieval-expansion result-sets research-sessions \
	run-chunk-evidence enable-pg-trgm index-concordance-trigram index-chunks-text-trigram \
	eval-table corpus-dictionary research-plans entities date-mentions mention-review-queue \
	entity-aliases-policy entity-mentions-surface-quality entity-aliases-class-policy \
	entity-relationships ner-methods-and-text-quality entity-aliases-performance-indexes \
	add-derived-alias-classes entity-alias-preferred transcript-turns speaker-turn-spans \
	proposal-corpus proposal-corpus-indexes refresh-proposal-corpus \
	all-migrations \
	ingest-mccarthy assess-mccarthy ingest-rosenberg assess-rosenberg \
	ocr-pipeline-setup build-alias-lexicon run-ocr-pipeline extract-ocr-candidates \
	resolve-ocr-candidates resolve-ocr-v3 cluster-ocr-variants export-ocr-review apply-ocr-review \
	export-simple-review export-and-auto-approve \
	ner-setup ner-extract ner-test \
	learn-junk-patterns junk-stats ocr-metrics ocr-status ocr-test

# Check if Docker is accessible before running commands
check-docker:
	@docker ps > /dev/null 2>&1 || (echo "ERROR: Docker is not accessible. If running from WSL, ensure Docker Desktop WSL integration is enabled:" && echo "  1. Open Docker Desktop" && echo "  2. Go to Settings > Resources > WSL Integration" && echo "  3. Enable integration for your WSL distro" && echo "  4. Click 'Apply & Restart'" && exit 1)

up: check-docker
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

date-mentions:
	@echo "Creating date_mentions table (Day 10)..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0020_date_mentions.sql

mention-review-queue:
	@echo "Creating mention_review_queue table (Day 10)..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0030_mention_review_queue.sql

entity-aliases-policy:
	@echo "Adding policy columns to entity_aliases (Day 10)..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0022_entity_aliases_policy.sql

entity-mentions-surface-quality:
	@echo "Adding surface quality tracking to entity_mentions (Day 10)..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0023_entity_mentions_surface_quality.sql

entity-aliases-class-policy:
	@echo "Adding alias_class and enhanced case matching to entity_aliases (Day 10)..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0024_entity_aliases_class_policy.sql

entity-relationships:
	@echo "Creating entity_relationships table (Day 11)..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0025_entity_relationships.sql

ner-methods-and-text-quality:
	@echo "Adding NER methods and text quality tracking (Day 11)..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0026_ner_methods_and_text_quality.sql

entity-aliases-performance-indexes:
	@echo "Adding performance indexes to entity_aliases (Performance optimization)..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0027_entity_aliases_performance_indexes.sql
	@echo "Running ANALYZE to update statistics..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -c "ANALYZE entity_aliases;"

add-derived-alias-classes:
	@echo "Adding derived alias classes (person_last) for surname derivation (2-Tier System)..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0028_add_derived_alias_classes.sql

entity-alias-preferred:
	@echo "Creating entity_alias_preferred table for human overrides..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0029_entity_alias_preferred.sql

transcript-turns:
	@echo "Creating transcript_turns table and speaker tracking for hearing transcripts..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0027_transcript_turns.sql

speaker-turn-spans:
	@echo "Adding speaker_turn_spans JSONB for precise attribution..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0031_speaker_turn_spans.sql

proposal-corpus:
	@echo "Creating proposal corpus system (entity_surface_stats, tiers, overrides)..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0031_proposal_corpus.sql

proposal-corpus-indexes:
	@echo "Adding additional indexes for proposal corpus system..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0032_proposal_corpus_indexes.sql

research-messages:
	@echo "Adding research_messages table..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0039_research_messages.sql

cooccurrence-performance-indexes:
	@echo "Adding cooccurring entities performance indexes..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0040_cooccurrence_performance_indexes.sql

refresh-proposal-corpus:
	@echo "Refreshing entity_surface_stats materialized view (CONCURRENTLY)..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -c "REFRESH MATERIALIZED VIEW CONCURRENTLY entity_surface_stats;"
	@echo "Refreshed. Checking tier counts..."
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -c "SELECT tier, COUNT(*) as surfaces FROM entity_surface_tiers GROUP BY tier ORDER BY tier;"

# Run all migrations in order (useful for fresh setup)
all-migrations: schema tracking metadata retrieval retrieval-expansion result-sets research-sessions \
	run-chunk-evidence enable-pg-trgm index-concordance-trigram index-chunks-text-trigram \
	eval-table corpus-dictionary research-plans entities date-mentions \
	entity-aliases-policy entity-mentions-surface-quality entity-aliases-class-policy \
	entity-relationships ner-methods-and-text-quality entity-aliases-performance-indexes \
	add-derived-alias-classes entity-alias-preferred mention-review-queue \
	transcript-turns speaker-turn-spans
	@echo "All migrations complete!"

verify:
	docker exec -it $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -c "SELECT 'collections' t, COUNT(*) FROM collections UNION ALL SELECT 'documents', COUNT(*) FROM documents UNION ALL SELECT 'pages', COUNT(*) FROM pages UNION ALL SELECT 'chunks', COUNT(*) FROM chunks UNION ALL SELECT 'chunk_pages', COUNT(*) FROM chunk_pages UNION ALL SELECT 'ingest_runs', COUNT(*) FROM ingest_runs;"

# =============================================================================
# Collection Ingest Targets
# =============================================================================

ingest-mccarthy:
	@echo "Ingesting McCarthy hearing transcripts (turn-aware)..."
	python scripts/ingest_mccarthy_v2.py

assess-mccarthy:
	@echo "Assessing McCarthy ingest..."
	python scripts/assess_mccarthy_ingest.py

ingest-rosenberg:
	@echo "Ingesting Rosenberg FBI case files..."
	python scripts/ingest_rosenberg.py

assess-rosenberg:
	@echo "Assessing Rosenberg ingest..."
	python scripts/assess_rosenberg_ingest.py

# =============================================================================
# OCR Extraction Pipeline
# =============================================================================

ocr-pipeline-setup:
	@echo "Setting up OCR extraction pipeline (all migrations)..."
	python scripts/run_migration.py migrations/0033_ocr_extraction_pipeline.sql
	python scripts/run_migration.py migrations/0034_ocr_idempotency_evidence.sql
	python scripts/run_migration.py migrations/0035_ocr_advanced_features.sql
	python scripts/build_alias_lexicon.py --rebuild
	@echo "OCR pipeline ready."

build-alias-lexicon:
	@echo "Building/refreshing alias lexicon index..."
	python scripts/build_alias_lexicon.py --rebuild

# Run full OCR pipeline on a collection
# Usage: make run-ocr-pipeline COLLECTION=silvermaster MAX_CHUNKS=1000
run-ocr-pipeline:
	python scripts/run_ocr_pipeline.py --collection $(COLLECTION) --max-chunks $(MAX_CHUNKS)

# Run only candidate extraction
extract-ocr-candidates:
	python scripts/extract_ocr_candidates.py --collection $(COLLECTION) --limit $(LIMIT)

# Run resolution V2 (batch, 22x faster)
resolve-ocr-candidates:
	python scripts/resolve_ocr_candidates_v2.py --batch-size 500 --limit $(LIMIT)

# Run resolution V3 (full features: context, anchoring, OCR-weighted edit)
resolve-ocr-v3:
	python scripts/resolve_ocr_candidates_v3.py --batch-size 500 --collection $(COLLECTION)

# =============================================================================
# OCR Clustering & Adjudication Workflow
# =============================================================================

# Cluster OCR variants from queued items
cluster-ocr-variants:
	python scripts/cluster_ocr_variants.py --source queued --min-mentions 2

# Export review file for offline adjudication (enhanced with context)
export-ocr-review:
	python scripts/export_ocr_review_file_v2.py --output-dir review_export/ --format both --limit $(LIMIT)

# Export simplified review batches (auto-approves high-confidence, batches by type)
# Usage: make export-simple-review
#        make export-simple-review AUTO_THRESHOLD=90
export-simple-review:
	python scripts/export_simplified_review.py --output-dir review_batches/ --auto-approve-threshold $(AUTO_THRESHOLD)

# Export and apply auto-approvals
export-and-auto-approve:
	python scripts/export_simplified_review.py --output-dir review_batches/ --apply-auto-approve --reviewer auto

# Apply reviewed adjudication decisions
# Usage: make apply-ocr-review FILE=review_batches/batch_1_confirm_match.csv
apply-ocr-review:
	python scripts/apply_ocr_adjudication.py $(FILE)

# =============================================================================
# OCR Maintenance & Metrics
# =============================================================================

# Learn junk patterns from rejected queue items
learn-junk-patterns:
	python scripts/learn_junk_pattern.py --from-queue --min-rejections 3

# Show junk pattern stats
junk-stats:
	python scripts/learn_junk_pattern.py --stats

# Show OCR pipeline metrics
ocr-metrics:
	@python -c "import psycopg2; c=psycopg2.connect(host='localhost',port=5432,dbname='neh',user='neh',password='neh'); r=c.cursor(); \
	r.execute('SELECT resolution_status, COUNT(*) FROM mention_candidates GROUP BY resolution_status ORDER BY COUNT(*) DESC'); \
	print('=== Mention Candidates ==='); [print(f'  {x[0]}: {x[1]}') for x in r.fetchall()]; \
	r.execute('SELECT method, COUNT(*) FROM entity_mentions WHERE method LIKE %s GROUP BY method', ('ocr%',)); \
	print('\\n=== OCR Entity Mentions ==='); [print(f'  {x[0]}: {x[1]}') for x in r.fetchall()]; \
	r.execute('SELECT status, COUNT(*) FROM mention_review_queue GROUP BY status'); \
	print('\\n=== Review Queue ==='); [print(f'  {x[0]}: {x[1]}') for x in r.fetchall()]; \
	r.execute('SELECT recommendation, COUNT(*) FROM ocr_variant_clusters WHERE status = %s GROUP BY recommendation', ('pending',)); \
	print('\\n=== Variant Clusters (pending) ==='); [print(f'  {x[0]}: {x[1]}') for x in r.fetchall()]; c.close()"

# Pipeline status
ocr-status:
	python scripts/run_ocr_pipeline.py --status --collection $(COLLECTION)

# Run acceptance tests
ocr-test:
	python tests/test_ocr_pipeline_acceptance.py

# =============================================================================
# NER Integration
# =============================================================================

# Install spaCy and download model
ner-setup:
	pip install spacy>=3.7.0
	python -m spacy download en_core_web_lg

# Run NER migrations (required before using NER extraction)
ner-migrate:
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) < migrations/0036_spacy_ner_signals.sql
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) < migrations/0037_ner_date_extraction.sql

# Run NER extraction on a collection
# Usage: make ner-extract COLLECTION=venona LIMIT=100
ner-extract:
	python scripts/run_ner_extraction.py --collection $(COLLECTION) --limit $(LIMIT) --threshold 0.5

# Run NER on specific documents for testing (dry-run)
# Usage: make ner-test DOC_IDS=123,456
ner-test:
	python scripts/run_ner_extraction.py --doc-ids $(DOC_IDS) --dry-run

# Full NER setup: install + migrate + test
ner-full-setup: ner-setup ner-migrate

# =============================================================================
# NER Corpus Sweep (Entity Discovery)
# =============================================================================

# Migrate corpus sweep tables
ner-corpus-migrate:
	docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) < migrations/0038_ner_corpus_discovery.sql

# Dry run corpus sweep on 100 chunks
ner-sweep-test:
	python scripts/run_ner_corpus_sweep.py --limit 100 --dry-run

# Run corpus sweep on a collection (set COLLECTION=venona LIMIT=500)
ner-sweep:
	python scripts/run_ner_corpus_sweep.py --collection $(COLLECTION) --limit $(LIMIT)

# Run full corpus sweep (WARNING: takes a long time)
ner-sweep-all:
	python scripts/run_ner_corpus_sweep.py --all

# Show sweep summary
ner-sweep-summary:
	python scripts/promote_ner_surfaces.py --summary

# Export tier 1 surfaces for review
ner-export-tier1:
	python scripts/promote_ner_surfaces.py --export ner_tier1_surfaces.csv --tier 1

# Export tier 2 surfaces for review
ner-export-tier2:
	python scripts/promote_ner_surfaces.py --export ner_tier2_surfaces.csv --tier 2

# Promote tier 1 surfaces (after review)
ner-promote-tier1:
	python scripts/promote_ner_surfaces.py --tier 1

# =============================================================================
# Danger Zone
# =============================================================================

# DANGER: deletes container + volume (wipes DB). Use only early in dev.
reset:
	docker compose down -v
	docker compose up -d
