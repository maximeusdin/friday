-- All runs with session label (if present)
CREATE OR REPLACE VIEW retrieval_runs_with_session AS
SELECT
  rr.*,
  rs.label AS session_label
FROM retrieval_runs rr
LEFT JOIN research_sessions rs ON rs.id = rr.session_id;

-- All result sets with session label + run context
CREATE OR REPLACE VIEW result_sets_with_session_and_run AS
SELECT
  rset.*,
  rs.label AS session_label,
  rr.query_text,
  rr.expanded_query_text,
  rr.search_type,
  rr.chunk_pv,
  rr.embedding_model,
  rr.top_k,
  rr.expand_concordance,
  rr.concordance_source_slug,
  rr.created_at AS run_created_at
FROM result_sets rset
LEFT JOIN research_sessions rs ON rs.id = rset.session_id
JOIN retrieval_runs rr ON rr.id = rset.retrieval_run_id;
