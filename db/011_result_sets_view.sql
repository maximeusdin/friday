CREATE OR REPLACE VIEW result_sets_with_run AS
SELECT
  rs.*,
  rr.query_text,
  rr.expanded_query_text,
  rr.search_type,
  rr.chunk_pv,
  rr.embedding_model,
  rr.top_k,
  rr.expand_concordance,
  rr.concordance_source_slug,
  rr.created_at AS run_created_at
FROM result_sets rs
JOIN retrieval_runs rr ON rr.id = rs.retrieval_run_id;
