#!/usr/bin/env python3
"""
export_result_set.py --id <result_set_id> [--out path.csv]

Exports a saved result_set to CSV with columns:
  chunk_id, document_id, collection_slug, first_page_id, last_page_id, snippet, matched_terms

Prefers persisted evidence (retrieval_run_chunk_evidence). Falls back to heuristics only for older runs.

Assumptions (based on your schema so far):
- result_sets(id, retrieval_run_id, chunk_ids, name, ...)
- retrieval_runs(id, query_text, expanded_query_text, ...)
- retrieval_chunks_v1 view exists and exposes at least:
    chunk_id, text, document_id, collection_slug, first_page_id, last_page_id
  (If your view uses different names, adjust the SELECT accordingly.)

Env:
- DATABASE_URL must be set (same pattern as your other CLIs)
"""

import os
import re
import csv
import sys
import json
import argparse
from typing import List, Set, Optional, Tuple, Dict, Any

import psycopg2


def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL")
    return psycopg2.connect(dsn)


_ws_re = re.compile(r"\s+")
_word_re = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]{1,}")  # basic tokenization


def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = _ws_re.sub(" ", s).strip()
    return s


def extract_query_terms(query: str, max_terms: int = 50) -> List[str]:
    """
    Take expanded_query_text (if present) or query_text and derive a small, stable set
    of terms to check against chunk text. This is intentionally simple + deterministic.
    """
    if not query:
        return []
    tokens = [t.lower() for t in _word_re.findall(query)]
    # remove very short tokens and common noise
    stop = {
        "the", "and", "or", "not", "to", "of", "in", "on", "for", "with", "by",
        "a", "an", "as", "is", "are", "was", "were", "be", "from", "at", "this",
        "that", "these", "those", "it", "its",
    }
    terms: List[str] = []
    seen: Set[str] = set()
    for t in tokens:
        if len(t) < 3 or t in stop:
            continue
        if t not in seen:
            seen.add(t)
            terms.append(t)
        if len(terms) >= max_terms:
            break
    return terms


def matched_terms_in_text(terms: List[str], text: str) -> str:
    """
    Return a '|' separated list of query terms that appear in the chunk text,
    using word-boundary-ish matching.
    """
    if not terms or not text:
        return ""
    t = text.lower()
    hits: List[str] = []
    for term in terms:
        # fast substring check first
        if term in t:
            # try to ensure it's a "token-ish" match
            if re.search(rf"\b{re.escape(term)}\b", t):
                hits.append(term)
    return "|".join(hits)


def extract_matched_terms_from_evidence(
    matched_lexemes: Optional[List[str]],
    explain_json: Optional[Dict[str, Any]],
) -> str:
    """
    Extract matched_terms from evidence using evidence-first logic:
    1. If matched_lexemes present: "|"-join them
    2. Else if explain_json.semantic_terms present: "|"-join them
    3. Else return empty string (caller should use heuristic fallback)
    
    Returns matched_terms string (empty if no evidence found).
    """
    # Priority 1: matched_lexemes (from evidence table)
    if matched_lexemes:
        return "|".join([t for t in matched_lexemes if t])
    
    # Priority 2: semantic_terms from explain_json (for vector-only runs)
    if explain_json and isinstance(explain_json, dict):
        if "semantic" in explain_json and isinstance(explain_json["semantic"], dict):
            sem = explain_json["semantic"]
            if "semantic_terms" in sem and isinstance(sem["semantic_terms"], list):
                semantic_terms = [t for t in sem["semantic_terms"] if t]
                if semantic_terms:
                    return "|".join(semantic_terms)
    
    # No evidence found - return empty (caller should use heuristic)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True, help="result_sets.id to export")
    ap.add_argument("--out", type=str, default=None, help="output CSV path (default: stdout)")
    ap.add_argument("--snippet_chars", type=int, default=350, help="snippet length")
    ap.add_argument("--include-match-type", action="store_true", default=True, help="Include match_type column (exact, approximate, semantic, mixed) [default: True]")
    ap.add_argument("--no-match-type", dest="include_match_type", action="store_false", help="Exclude match_type column")
    ap.add_argument("--include-highlight", action="store_true", default=True, help="Include highlight column (from evidence table) [default: True]")
    ap.add_argument("--no-highlight", dest="include_highlight", action="store_false", help="Exclude highlight column")
    args = ap.parse_args()

    with get_conn() as conn, conn.cursor() as cur:
        # 1) Load result set + associated run query text
        cur.execute(
            """
            SELECT
              rs.id,
              rs.name,
              rs.retrieval_run_id,
              rs.chunk_ids,
              COALESCE(rr.expanded_query_text, rr.query_text) AS effective_query,
              rr.chunk_pv
            FROM result_sets rs
            JOIN retrieval_runs rr ON rr.id = rs.retrieval_run_id
            WHERE rs.id = %s
            """,
            (args.id,),
        )
        row = cur.fetchone()
        if not row:
            raise SystemExit(f"No result_set found with id={args.id}")

        rs_id, rs_name, run_id, chunk_ids, effective_query, chunk_pv = row
        if not chunk_ids:
            raise SystemExit(f"result_set id={rs_id} has empty chunk_ids")

        terms = extract_query_terms(effective_query or "")

        # Determine whether we have persisted evidence for this run.
        cur.execute(
            """
            SELECT COUNT(*)
            FROM retrieval_run_chunk_evidence
            WHERE retrieval_run_id = %s;
            """,
            (run_id,),
        )
        evidence_count = int(cur.fetchone()[0])
        expected = len(chunk_ids)
        use_evidence = evidence_count == expected

        # 2) Pull ordered chunks + metadata (preserve order using unnest ... WITH ORDINALITY)
        # If evidence exists, use it for snippet + matched_terms + explain_json.
        if use_evidence:
            cur.execute(
                """
                WITH ordered AS (
                  SELECT u.chunk_id, u.ord
                  FROM unnest(%s::bigint[]) WITH ORDINALITY AS u(chunk_id, ord)
                )
                SELECT
                  o.ord,
                  c.id AS chunk_id,
                  cm.document_id,
                  cm.collection_slug,
                  cm.first_page_id,
                  cm.last_page_id,
                  COALESCE(e.highlight, COALESCE(c.clean_text, c.text)) AS text_for_snippet,
                  e.matched_lexemes,
                  e.explain_json,
                  e.highlight
                FROM ordered o
                JOIN chunks c ON c.id = o.chunk_id
                JOIN chunk_metadata cm ON cm.chunk_id = c.id AND cm.pipeline_version = %s
                JOIN retrieval_run_chunk_evidence e
                  ON e.retrieval_run_id = %s AND e.chunk_id = o.chunk_id
                ORDER BY o.ord ASC
                """,
                (chunk_ids, chunk_pv, run_id),
            )
        else:
            # Backward-compat fallback (older runs): use heuristic term matching.
            cur.execute(
                """
                WITH ordered AS (
                  SELECT u.chunk_id, u.ord
                  FROM unnest(%s::bigint[]) WITH ORDINALITY AS u(chunk_id, ord)
                )
                SELECT
                  o.ord,
                  c.id AS chunk_id,
                  cm.document_id,
                  cm.collection_slug,
                  cm.first_page_id,
                  cm.last_page_id,
                  COALESCE(c.clean_text, c.text) AS text_for_snippet,
                  NULL::text[] AS matched_lexemes,
                  NULL::jsonb AS explain_json,
                  NULL::text AS highlight
                FROM ordered o
                JOIN chunks c ON c.id = o.chunk_id
                JOIN chunk_metadata cm ON cm.chunk_id = c.id AND cm.pipeline_version = %s
                ORDER BY o.ord ASC
                """,
                (chunk_ids, chunk_pv),
            )

        out_fh = open(args.out, "w", newline="", encoding="utf-8") if args.out else sys.stdout
        close_out = bool(args.out)

        try:
            w = csv.writer(out_fh)
            headers = [
                "chunk_id",
                "document_id",
                "collection_slug",
                "first_page_id",
                "last_page_id",
                "snippet",
                "matched_terms",
            ]
            if args.include_match_type:
                headers.append("match_type")
            if args.include_highlight:
                headers.append("highlight")
            w.writerow(headers)

            for row in cur.fetchall():
                if use_evidence:
                    ord_, chunk_id, document_id, collection_slug, first_page_id, last_page_id, text_for_snippet, matched_lexemes, explain_json, highlight = row
                else:
                    ord_, chunk_id, document_id, collection_slug, first_page_id, last_page_id, text_for_snippet, matched_lexemes, explain_json, highlight = row
                
                snippet = clean_text(text_for_snippet or "")
                if args.snippet_chars and len(snippet) > args.snippet_chars:
                    snippet = snippet[: args.snippet_chars].rstrip() + "…"

                # Evidence-first matched_terms logic:
                # 1. matched_lexemes (if present)
                # 2. semantic_terms from explain_json (if present)
                # 3. Legacy heuristic (fallback for old runs)
                matched = ""
                if use_evidence:
                    try:
                        # Parse explain_json if it's a string
                        explain_dict = explain_json
                        if explain_json and isinstance(explain_json, str):
                            explain_dict = json.loads(explain_json)
                        
                        # Extract from evidence (matched_lexemes > semantic_terms)
                        matched = extract_matched_terms_from_evidence(matched_lexemes, explain_dict)
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        # If parsing fails, try matched_lexemes directly
                        if matched_lexemes:
                            matched = "|".join([t for t in matched_lexemes if t])
                
                # Fallback to heuristic if no evidence found
                if not matched:
                    matched = matched_terms_in_text(terms, snippet)

                row_data = [
                    chunk_id,
                    document_id,
                    collection_slug,
                    first_page_id,
                    last_page_id,
                    snippet,
                    matched,
                ]
                if args.include_match_type:
                    # Determine match_type for display
                    match_type = "unknown"
                    if use_evidence and explain_json:
                        try:
                            explain_dict = explain_json if isinstance(explain_json, dict) else json.loads(explain_json)
                            has_exact = False
                            has_approx = False
                            has_semantic = False
                            
                            if "lex" in explain_dict and isinstance(explain_dict["lex"], dict):
                                if explain_dict["lex"].get("matched_lexemes"):
                                    has_exact = True
                            if "approx_lex" in explain_dict and isinstance(explain_dict["approx_lex"], dict):
                                if explain_dict["approx_lex"].get("matched_terms"):
                                    has_approx = True
                            if "semantic" in explain_dict and isinstance(explain_dict["semantic"], dict):
                                if explain_dict["semantic"].get("semantic_terms"):
                                    has_semantic = True
                            
                            if has_exact and has_approx:
                                match_type = "mixed"
                            elif has_exact:
                                match_type = "exact"
                            elif has_approx:
                                match_type = "approximate"
                            elif has_semantic:
                                match_type = "semantic"
                        except:
                            # Fallback: check matched_lexemes directly
                            if matched_lexemes:
                                match_type = "exact"
                    elif matched_lexemes:
                        match_type = "exact"
                    elif not matched:
                        # If no matched terms and no evidence, it's unknown
                        match_type = "unknown"
                    
                    row_data.append(match_type)
                if args.include_highlight:
                    row_data.append(clean_text(highlight or ""))
                w.writerow(row_data)
        finally:
            if close_out:
                out_fh.close()

    if args.out:
        print(f"Wrote CSV: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
