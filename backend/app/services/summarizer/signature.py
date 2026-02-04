"""
Summary Signature Computation and Cache Lookup

Signatures enable:
- Cache lookup to avoid redundant LLM calls
- Deduplication of identical summaries
- Race condition handling via upsert
"""

import hashlib
import json
from typing import Optional, List, Dict, Any

import psycopg2.errors

from .models import SummaryArtifact


def compute_summary_signature(
    result_set_id: int,
    retrieval_run_id: int,
    profile: str,
    summary_type: str,
    question: Optional[str],
    filters: Optional[dict],
    chunk_ids: List[int],  # ORDERED - do not sort
    prompt_version: str,
    model_name: str,
) -> str:
    """
    Compute SHA256 signature for cache lookup and dedupe.
    
    IMPORTANT: chunk_ids preserves order from Stage A selection.
    - Order matters for chronological bias, delta summarization, progressive expansion
    - "Same chunks, different order" = different summary
    
    Args:
        result_set_id: ID of the result set
        retrieval_run_id: ID of the retrieval run (prevents semantic drift)
        profile: Profile name
        summary_type: "sample" or "page_window"
        question: User question (normalized)
        filters: Applied filters
        chunk_ids: Selected chunk IDs IN SELECTION ORDER
        prompt_version: Prompt template version
        model_name: LLM model name
    
    Returns:
        SHA256 hex digest string
    """
    normalized = json.dumps({
        "result_set_id": result_set_id,
        "retrieval_run_id": retrieval_run_id,
        "profile": profile,
        "summary_type": summary_type,
        "question": (question or "").strip().lower(),
        "filters": filters or {},
        "chunk_ids": chunk_ids,  # Keep original order - do NOT sort
        "prompt_version": prompt_version,
        "model_name": model_name,
    }, sort_keys=True)
    
    return hashlib.sha256(normalized.encode()).hexdigest()


def lookup_by_signature(conn, signature: str) -> Optional[SummaryArtifact]:
    """
    Look up existing summary by signature.
    
    Args:
        conn: Database connection
        signature: Summary signature hash
    
    Returns:
        SummaryArtifact if found, None otherwise
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT output_json, summary_id
            FROM session_summaries
            WHERE summary_signature = %s
            """,
            (signature,)
        )
        row = cur.fetchone()
        
        if not row:
            return None
        
        output_json = row[0]
        
        # Parse and return artifact
        return SummaryArtifact(**output_json)


def persist_summary(
    conn,
    artifact: SummaryArtifact,
    signature: str,
    result_set_id: int,
    retrieval_run_id: int,
    session_id: Optional[int],
    selection_spec: Dict[str, Any],
    selection_inputs: Dict[str, Any],
    question: Optional[str],
    profile: str,
    summary_type: str,
    model_name: str,
    prompt_version: str,
) -> str:
    """
    Persist summary to database.
    
    Args:
        conn: Database connection
        artifact: The summary artifact
        signature: Computed signature
        result_set_id: Result set ID
        retrieval_run_id: Retrieval run ID
        session_id: Optional session ID
        selection_spec: Selection spec dict
        selection_inputs: Selection inputs dict
        question: User question
        profile: Profile name
        summary_type: "sample" or "page_window"
        model_name: Model name
        prompt_version: Prompt version
    
    Returns:
        Summary ID (UUID)
    """
    from psycopg2.extras import Json
    
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO session_summaries (
                result_set_id, session_id, summary_signature,
                selection_spec, selection_inputs, selected_chunk_ids,
                user_question, profile, summary_type,
                output_json, model_name, prompt_version
            ) VALUES (
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s
            )
            RETURNING summary_id
            """,
            (
                result_set_id,
                session_id,
                signature,
                Json(selection_spec),
                Json(selection_inputs),
                selection_spec.get("chunk_ids", []),
                question,
                profile,
                summary_type,
                Json(artifact.model_dump()),
                model_name,
                prompt_version,
            )
        )
        summary_id = str(cur.fetchone()[0])
        conn.commit()
        
        return summary_id


def persist_or_fetch_summary(
    conn,
    artifact: SummaryArtifact,
    signature: str,
    result_set_id: int,
    retrieval_run_id: int,
    session_id: Optional[int],
    selection_spec: Dict[str, Any],
    selection_inputs: Dict[str, Any],
    question: Optional[str],
    profile: str,
    summary_type: str,
    model_name: str,
    prompt_version: str,
) -> SummaryArtifact:
    """
    Persist summary with upsert pattern for race handling.
    
    If signature already exists (race condition), return existing summary.
    
    Args:
        (same as persist_summary)
    
    Returns:
        The summary artifact (may be cached version if race occurred)
    """
    try:
        summary_id = persist_summary(
            conn=conn,
            artifact=artifact,
            signature=signature,
            result_set_id=result_set_id,
            retrieval_run_id=retrieval_run_id,
            session_id=session_id,
            selection_spec=selection_spec,
            selection_inputs=selection_inputs,
            question=question,
            profile=profile,
            summary_type=summary_type,
            model_name=model_name,
            prompt_version=prompt_version,
        )
        # Update artifact with actual summary_id
        artifact_dict = artifact.model_dump()
        artifact_dict["summary_id"] = summary_id
        return SummaryArtifact(**artifact_dict)
        
    except psycopg2.errors.UniqueViolation:
        # Race condition: another request inserted first
        conn.rollback()
        cached = lookup_by_signature(conn, signature)
        if cached:
            return cached.with_cached_flag(True)
        # Should not happen, but fall back to returning original
        return artifact
