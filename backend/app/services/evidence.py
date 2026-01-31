"""
Evidence assembly helpers.

Two main entry points:
- build_evidence_refs_from_chunk(chunk_id): chunk → chunk_pages → pages → documents
- build_evidence_refs_from_citations(citation_ids): entity_citations → documents

All EvidenceRef shapes follow docs/v1_contract.md:
- document_id (required)
- pdf_page (required, 1-based)
- chunk_id (optional)
- span (optional)
- quote (optional)
- why (optional)
"""
from typing import List, Dict, Any, Optional


def build_evidence_refs_from_chunk(conn, chunk_id: int) -> List[Dict[str, Any]]:
    """
    Build evidence refs for a chunk.
    
    Follows the chain: chunk → chunk_pages → pages → documents
    
    Args:
        conn: Database connection
        chunk_id: The chunk ID
    
    Returns:
        List of EvidenceRef dicts, one per page the chunk spans
    """
    evidence_refs = []
    
    with conn.cursor() as cur:
        # Get pages this chunk spans via chunk_pages bridge table
        cur.execute(
            """
            SELECT 
                cp.page_id,
                cp.span_order,
                cp.start_char,
                cp.end_char,
                p.document_id,
                p.pdf_page_number,
                p.raw_text
            FROM chunk_pages cp
            JOIN pages p ON p.id = cp.page_id
            WHERE cp.chunk_id = %s
            ORDER BY cp.span_order
            """,
            (chunk_id,),
        )
        rows = cur.fetchall()
        
        if not rows:
            # Fallback: try chunk_metadata for first/last page
            cur.execute(
                """
                SELECT 
                    cm.document_id,
                    fp.pdf_page_number as first_page,
                    lp.pdf_page_number as last_page
                FROM chunk_metadata cm
                LEFT JOIN pages fp ON fp.id = cm.first_page_id
                LEFT JOIN pages lp ON lp.id = cm.last_page_id
                WHERE cm.chunk_id = %s
                LIMIT 1
                """,
                (chunk_id,),
            )
            meta_row = cur.fetchone()
            
            if meta_row:
                doc_id, first_page, last_page = meta_row
                if doc_id and first_page:
                    evidence_refs.append({
                        "document_id": doc_id,
                        "pdf_page": first_page,  # 1-based from database
                        "chunk_id": chunk_id,
                    })
                    
                    # Add additional pages if chunk spans multiple
                    if last_page and last_page != first_page:
                        for page_num in range(first_page + 1, last_page + 1):
                            evidence_refs.append({
                                "document_id": doc_id,
                                "pdf_page": page_num,
                                "chunk_id": chunk_id,
                            })
            
            return evidence_refs
        
        # Build refs from chunk_pages
        for row in rows:
            page_id, span_order, start_char, end_char, doc_id, pdf_page, raw_text = row
            
            ref = {
                "document_id": doc_id,
                "pdf_page": pdf_page if pdf_page else 1,  # Default to 1 if NULL
                "chunk_id": chunk_id,
            }
            
            # Add span if available
            if start_char is not None and end_char is not None:
                ref["span"] = {"start": start_char, "end": end_char}
            
            # Extract quote from page text if we have offsets
            if raw_text and start_char is not None and end_char is not None:
                quote = raw_text[start_char:end_char]
                if len(quote) > 200:
                    quote = quote[:200] + "..."
                ref["quote"] = quote
            
            evidence_refs.append(ref)
    
    return evidence_refs


def build_evidence_refs_from_citations(
    conn,
    citation_ids: List[int],
    include_context: bool = False
) -> List[Dict[str, Any]]:
    """
    Build evidence refs from entity_citations records.
    
    This is the citation-first path for evidence that doesn't come directly
    from chunk retrieval (e.g., entity mention citations, concordance refs).
    
    Args:
        conn: Database connection
        citation_ids: List of entity_citation IDs
        include_context: Whether to include surrounding context as quote
    
    Returns:
        List of EvidenceRef dicts
    """
    if not citation_ids:
        return []
    
    evidence_refs = []
    
    with conn.cursor() as cur:
        # Get citation data
        cur.execute(
            """
            SELECT 
                ec.id,
                ec.citation_text,
                ec.collection_slug,
                ec.document_label,
                ec.page_list,
                ec.notes,
                d.id as document_id
            FROM entity_citations ec
            LEFT JOIN documents d ON d.source_name ILIKE '%%' || ec.document_label || '%%'
                AND (ec.collection_slug IS NULL OR EXISTS (
                    SELECT 1 FROM collections c 
                    WHERE c.id = d.collection_id 
                    AND c.slug = ec.collection_slug
                ))
            WHERE ec.id = ANY(%s)
            """,
            (citation_ids,),
        )
        rows = cur.fetchall()
        
        for row in rows:
            cit_id, cit_text, collection_slug, doc_label, page_list, notes, doc_id = row
            
            if not doc_id:
                # Can't resolve to a document, skip
                continue
            
            # Parse page_list (e.g., "1,2,3" or "1-5")
            pages = parse_page_list(page_list) if page_list else [1]
            
            for page in pages:
                ref = {
                    "document_id": doc_id,
                    "pdf_page": page,
                }
                
                if notes:
                    ref["why"] = notes
                
                evidence_refs.append(ref)
    
    return evidence_refs


def build_evidence_refs_from_mention(
    conn,
    mention_id: int
) -> List[Dict[str, Any]]:
    """
    Build evidence refs from an entity_mention record.
    
    Args:
        conn: Database connection
        mention_id: The entity_mention ID
    
    Returns:
        List of EvidenceRef dicts (usually one, for the chunk)
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 
                em.chunk_id,
                em.document_id,
                em.surface,
                em.start_char,
                em.end_char
            FROM entity_mentions em
            WHERE em.id = %s
            """,
            (mention_id,),
        )
        row = cur.fetchone()
        
        if not row:
            return []
        
        chunk_id, doc_id, surface, start_char, end_char = row
        
        # Get evidence from the chunk
        refs = build_evidence_refs_from_chunk(conn, chunk_id)
        
        # Enhance with mention-specific data
        for ref in refs:
            if surface:
                ref["quote"] = surface
            if start_char is not None and end_char is not None:
                # Note: these are char offsets in the chunk, not page
                # Would need additional mapping for page-level spans
                pass
        
        return refs


def parse_page_list(page_str: str) -> List[int]:
    """
    Parse a page list string into individual page numbers.
    
    Examples:
        "1" -> [1]
        "1,2,3" -> [1, 2, 3]
        "1-5" -> [1, 2, 3, 4, 5]
        "1,3-5,7" -> [1, 3, 4, 5, 7]
    """
    if not page_str:
        return []
    
    pages = []
    for part in page_str.split(","):
        part = part.strip()
        if "-" in part:
            try:
                start, end = part.split("-", 1)
                start = int(start.strip())
                end = int(end.strip())
                pages.extend(range(start, end + 1))
            except ValueError:
                continue
        else:
            try:
                pages.append(int(part))
            except ValueError:
                continue
    
    return sorted(set(pages))
