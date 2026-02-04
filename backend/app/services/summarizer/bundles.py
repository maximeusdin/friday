"""
Evidence Bundle Construction

Builds compact evidence bundles for LLM synthesis with:
- Quote anchors for drill-down UX
- Bundle ID mapping (B1, B2...) for safe LLM citations
- Match trace information
"""

import re
from typing import List, Dict, Tuple, Optional

from .models import (
    EvidenceBundle,
    QuoteAnchor,
    MatchTraceInfo,
    LineContext,
    ChunkCandidate,
)


# =============================================================================
# Quote Anchor Extraction
# =============================================================================

def extract_first_sentence(text: str, max_length: int = 120) -> str:
    """Extract first sentence from text, capped at max_length."""
    if not text:
        return ""
    
    # Try to find sentence boundary
    match = re.search(r'^[^.!?]*[.!?]', text)
    if match:
        sentence = match.group(0).strip()
        if len(sentence) <= max_length:
            return sentence
    
    # Fall back to truncation at word boundary
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.5:
        return truncated[:last_space] + "..."
    return truncated + "..."


def extract_quote_anchor(
    chunk_text: str,
    snippet: str,
    match_trace: MatchTraceInfo,
    line_context: Optional[LineContext],
    max_length: int = 120
) -> QuoteAnchor:
    """
    Extract quote anchor with priority order.
    
    Priority (use first that works):
    1. Phrase/entity offsets from match trace highlights
       - Best: anchors around the actual matched signal
    2. Line context highlight_line (if available)
       - Good: the middle/highlighted line
    3. First sentence of snippet
       - Fallback: at least something meaningful
    
    Store anchor_method for debugging "why is this quote weird?"
    """
    # Priority 1: Match trace highlights with offsets
    if match_trace.highlight_spans:
        span = match_trace.highlight_spans[0]  # Take first highlight
        start = span.get("start", 0)
        end = span.get("end", start + max_length)
        excerpt = chunk_text[start:end][:max_length]
        return QuoteAnchor(
            start_char=start,
            end_char=start + len(excerpt),
            quote_excerpt=excerpt,
            anchor_method="highlight"
        )
    
    # Try to find matched phrases in text
    if match_trace.matched_phrases:
        for phrase in match_trace.matched_phrases:
            idx = chunk_text.lower().find(phrase.lower())
            if idx >= 0:
                # Expand context around the phrase
                context_start = max(0, idx - 20)
                context_end = min(len(chunk_text), idx + len(phrase) + 20)
                excerpt = chunk_text[context_start:context_end][:max_length]
                return QuoteAnchor(
                    start_char=context_start,
                    end_char=context_start + len(excerpt),
                    quote_excerpt=excerpt,
                    anchor_method="highlight"
                )
    
    # Priority 2: Line context (highlight line)
    if line_context and line_context.highlight_line:
        line = line_context.highlight_line[:max_length]
        # Try to find offset in full text
        start = chunk_text.find(line) if chunk_text else -1
        return QuoteAnchor(
            start_char=start if start >= 0 else 0,
            end_char=(start if start >= 0 else 0) + len(line),
            quote_excerpt=line,
            anchor_method="line_context"
        )
    
    # Priority 3: First sentence fallback
    first_sentence = extract_first_sentence(snippet, max_length)
    return QuoteAnchor(
        start_char=0,
        end_char=len(first_sentence),
        quote_excerpt=first_sentence,
        anchor_method="snippet_fallback"
    )


# =============================================================================
# Bundle Building
# =============================================================================

def build_bundles(
    conn,
    selected_candidates: List[ChunkCandidate],
    result_set_id: int,
    snippet_length: int = 600,
    include_line_context: bool = True,
) -> Tuple[List[EvidenceBundle], Dict[str, int]]:
    """
    Build evidence bundles from selected candidates.
    
    Args:
        conn: Database connection
        selected_candidates: Candidates selected by Stage A (in order)
        result_set_id: ID of the result set
        snippet_length: Max chars for snippet (300-800 recommended)
        include_line_context: Whether to extract line context
    
    Returns:
        (bundles, bundle_id_to_chunk_id_map)
    """
    if not selected_candidates:
        return [], {}
    
    bundles: List[EvidenceBundle] = []
    bundle_map: Dict[str, int] = {}
    
    chunk_ids = [c.chunk_id for c in selected_candidates]
    
    with conn.cursor() as cur:
        # Fetch chunk text and metadata
        cur.execute(
            """
            SELECT 
                c.id as chunk_id,
                COALESCE(c.clean_text, c.text) as text,
                cm.document_id,
                d.source_name as doc_title,
                p.page_number,
                cm.date_min,
                mt.matched_entity_ids,
                mt.matched_phrases,
                mt.in_lexical,
                mt.in_vector,
                mt.score_lexical,
                mt.score_vector,
                mt.score_hybrid
            FROM chunks c
            LEFT JOIN chunk_metadata cm ON cm.chunk_id = c.id
            LEFT JOIN documents d ON d.id = cm.document_id
            LEFT JOIN pages p ON p.id = cm.first_page_id
            LEFT JOIN result_set_match_traces mt 
                ON mt.result_set_id = %s AND mt.chunk_id = c.id
            WHERE c.id = ANY(%s)
            """,
            (result_set_id, chunk_ids)
        )
        
        # Build lookup by chunk_id
        chunk_data = {}
        for row in cur.fetchall():
            chunk_data[row[0]] = {
                "text": row[1] or "",
                "doc_id": row[2],
                "doc_title": row[3] or "Unknown",
                "page": row[4],
                "date_min": row[5],
                "matched_entity_ids": list(row[6] or []),
                "matched_phrases": list(row[7] or []),
                "in_lexical": row[8] or False,
                "in_vector": row[9] or False,
                "score_lexical": row[10],
                "score_vector": row[11],
                "score_hybrid": row[12],
            }
        
        # Get place IDs for chunks (if entity_mentions exist)
        place_ids_by_chunk: Dict[int, List[int]] = {cid: [] for cid in chunk_ids}
        try:
            cur.execute(
                """
                SELECT em.chunk_id, e.id
                FROM entity_mentions em
                JOIN entities e ON e.id = em.entity_id
                WHERE em.chunk_id = ANY(%s)
                  AND e.entity_type = 'place'
                """,
                (chunk_ids,)
            )
            for row in cur.fetchall():
                if row[0] in place_ids_by_chunk:
                    place_ids_by_chunk[row[0]].append(row[1])
        except Exception:
            # entity_mentions may not exist or have different schema
            pass
    
    # Build bundles in selection order
    for i, candidate in enumerate(selected_candidates):
        chunk_id = candidate.chunk_id
        bundle_id = f"B{i + 1}"
        bundle_map[bundle_id] = chunk_id
        
        data = chunk_data.get(chunk_id, {})
        text = data.get("text", "")
        
        # Build snippet (truncate if needed)
        snippet = text[:snippet_length]
        if len(text) > snippet_length:
            # Try to end at word boundary
            last_space = snippet.rfind(' ')
            if last_space > snippet_length * 0.7:
                snippet = snippet[:last_space] + "..."
            else:
                snippet += "..."
        
        # Build match trace info
        match_trace = MatchTraceInfo(
            matched_entity_ids=data.get("matched_entity_ids", []),
            matched_phrases=data.get("matched_phrases", []),
            in_lexical=data.get("in_lexical", False),
            in_vector=data.get("in_vector", False),
            score_lexical=data.get("score_lexical"),
            score_vector=data.get("score_vector"),
            score_hybrid=data.get("score_hybrid"),
        )
        
        # Build line context if requested
        line_context = None
        if include_line_context and text:
            lines = text.split('\n')
            if len(lines) >= 3:
                mid = len(lines) // 2
                line_context = LineContext(
                    line_before=lines[mid - 1] if mid > 0 else None,
                    highlight_line=lines[mid],
                    line_after=lines[mid + 1] if mid < len(lines) - 1 else None,
                )
            elif len(lines) == 2:
                line_context = LineContext(
                    line_before=None,
                    highlight_line=lines[0],
                    line_after=lines[1],
                )
            elif len(lines) == 1:
                line_context = LineContext(
                    line_before=None,
                    highlight_line=lines[0],
                    line_after=None,
                )
        
        # Extract quote anchor
        quote_anchor = extract_quote_anchor(
            chunk_text=text,
            snippet=snippet,
            match_trace=match_trace,
            line_context=line_context,
        )
        
        # Format date
        date_key = None
        if data.get("date_min"):
            date_key = data["date_min"].strftime("%Y-%m-%d") if hasattr(data["date_min"], "strftime") else str(data["date_min"])
        
        bundle = EvidenceBundle(
            bundle_id=bundle_id,
            chunk_id=chunk_id,
            doc_id=data.get("doc_id", candidate.doc_id),
            doc_title=data.get("doc_title", candidate.doc_title),
            page=data.get("page", candidate.page),
            snippet=snippet,
            quote_anchor=quote_anchor,
            line_context=line_context,
            match_trace=match_trace,
            date_key=date_key,
            place_ids=place_ids_by_chunk.get(chunk_id, []),
        )
        
        bundles.append(bundle)
    
    return bundles, bundle_map


def format_bundles_for_prompt(bundles: List[EvidenceBundle]) -> str:
    """Format all bundles for inclusion in LLM prompt."""
    return "\n\n".join(b.to_prompt_format() for b in bundles)
