"""
Query Expansion & Suggestions Service.

Feature 2: Query Expansion & Suggestions
- Related terms based on corpus co-occurrence
- "Did you mean?" suggestions for potential typos
- Query metrics tracking
- Low-result query expansion suggestions
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from difflib import SequenceMatcher


@dataclass
class QuerySuggestion:
    """A query suggestion."""
    suggestion_type: str  # "related_term", "did_you_mean", "expansion", "entity"
    original: str
    suggested: str
    score: float  # Relevance/confidence score (0-1)
    reason: str  # Why this suggestion was made


@dataclass
class QueryMetrics:
    """Metrics for a query execution."""
    query_text: str
    result_count: int
    execution_time_ms: float
    search_type: str  # "lexical", "vector", "hybrid"
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    recorded_at: datetime = field(default_factory=datetime.utcnow)


def get_conn():
    """Get database connection from environment."""
    import psycopg2
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    return psycopg2.connect(database_url)


# =============================================================================
# Related Terms (Co-occurrence Based)
# =============================================================================

def get_related_terms(
    conn,
    term: str,
    limit: int = 10,
    min_cooccurrence: int = 5,
) -> List[QuerySuggestion]:
    """
    Find related terms based on corpus co-occurrence.
    
    Uses the lexemes from chunks' text_tsv to find terms that frequently
    appear in the same chunks as the query term.
    
    Args:
        conn: Database connection
        term: The query term to find related terms for
        limit: Maximum number of related terms to return
        min_cooccurrence: Minimum co-occurrence count to include
    
    Returns:
        List of QuerySuggestion objects with related terms
    """
    normalized_term = term.lower().strip()
    if not normalized_term:
        return []
    
    # Find chunks containing the term and extract co-occurring terms
    # ts_stat requires a literal query string, so we use a different approach:
    # Get matching chunk IDs first, then extract words using unnest
    with conn.cursor() as cur:
        # Use a simpler approach: extract lexemes from matching chunks
        # and count their frequency
        cur.execute("""
            WITH matching_chunks AS (
                SELECT id, text_tsv FROM chunks
                WHERE text_tsv @@ to_tsquery('simple', %s)
                LIMIT 5000
            ),
            lexemes AS (
                SELECT (unnest(tsvector_to_array(text_tsv)))::text as word
                FROM matching_chunks
            )
            SELECT word, COUNT(*) as count
            FROM lexemes
            WHERE word != %s
            AND length(word) > 2
            GROUP BY word
            HAVING COUNT(*) >= %s
            ORDER BY count DESC
            LIMIT %s
        """, (normalized_term, normalized_term, min_cooccurrence, limit))
        
        rows = cur.fetchall()
    
    # Convert to suggestions
    suggestions = []
    max_count = rows[0][1] if rows else 1
    
    for word, count in rows:
        score = count / max_count  # Normalize score
        suggestions.append(QuerySuggestion(
            suggestion_type="related_term",
            original=term,
            suggested=word,
            score=score,
            reason=f"Appears in {count} chunks with '{term}'",
        ))
    
    return suggestions


def get_entity_suggestions(
    conn,
    term: str,
    limit: int = 5,
) -> List[QuerySuggestion]:
    """
    Suggest entities that match the query term.
    
    Searches the entities table for canonical names matching the term,
    useful for suggesting specific entities when user types a general term.
    
    Args:
        conn: Database connection
        term: The query term
        limit: Maximum number of entity suggestions
    
    Returns:
        List of QuerySuggestion objects with entity suggestions
    """
    normalized_term = term.lower().strip()
    if not normalized_term:
        return []
    
    with conn.cursor() as cur:
        # Search entities by canonical name similarity
        cur.execute("""
            SELECT id, canonical_name, entity_type,
                   similarity(lower(canonical_name), %s) as sim
            FROM entities
            WHERE lower(canonical_name) %% %s
            ORDER BY sim DESC
            LIMIT %s
        """, (normalized_term, normalized_term, limit))
        
        rows = cur.fetchall()
    
    suggestions = []
    for entity_id, name, entity_type, sim in rows:
        suggestions.append(QuerySuggestion(
            suggestion_type="entity",
            original=term,
            suggested=f"{name} (ID: {entity_id})",
            score=sim,
            reason=f"{entity_type.capitalize()} entity matching '{term}'",
        ))
    
    return suggestions


# =============================================================================
# "Did You Mean?" (Typo Detection)
# =============================================================================

def get_typo_suggestions(
    conn,
    term: str,
    max_suggestions: int = 3,
    min_similarity: float = 0.6,
) -> List[QuerySuggestion]:
    """
    Suggest corrections for potential typos.
    
    Uses PostgreSQL's pg_trgm extension for fuzzy matching against
    a vocabulary extracted from the corpus.
    
    Args:
        conn: Database connection
        term: The potentially misspelled term
        max_suggestions: Maximum number of suggestions
        min_similarity: Minimum trigram similarity threshold
    
    Returns:
        List of QuerySuggestion objects with typo corrections
    """
    normalized_term = term.lower().strip()
    if not normalized_term or len(normalized_term) < 3:
        return []
    
    with conn.cursor() as cur:
        # Check if the term exists in the corpus vocabulary
        # If it does, no typo correction needed
        cur.execute("""
            SELECT COUNT(*) FROM chunks
            WHERE text_tsv @@ to_tsquery('simple', %s)
            LIMIT 1
        """, (normalized_term,))
        
        if cur.fetchone()[0] > 0:
            # Term exists, no correction needed
            return []
        
        # Find similar terms from entity canonical names
        # (good source of proper names that might be misspelled)
        cur.execute("""
            SELECT DISTINCT canonical_name,
                   similarity(lower(canonical_name), %s) as sim
            FROM entities
            WHERE similarity(lower(canonical_name), %s) >= %s
            AND lower(canonical_name) != %s
            ORDER BY sim DESC
            LIMIT %s
        """, (normalized_term, normalized_term, min_similarity, 
              normalized_term, max_suggestions))
        
        rows = cur.fetchall()
    
    suggestions = []
    for name, sim in rows:
        suggestions.append(QuerySuggestion(
            suggestion_type="did_you_mean",
            original=term,
            suggested=name,
            score=sim,
            reason=f"Similar spelling (similarity: {sim:.2f})",
        ))
    
    return suggestions


def get_typo_suggestions_local(
    term: str,
    vocabulary: List[str],
    max_suggestions: int = 3,
    min_similarity: float = 0.6,
) -> List[QuerySuggestion]:
    """
    Suggest corrections for potential typos using local vocabulary.
    
    Uses difflib's SequenceMatcher for similarity, doesn't require database.
    Useful when you have a known vocabulary list.
    
    Args:
        term: The potentially misspelled term
        vocabulary: List of valid terms to match against
        max_suggestions: Maximum number of suggestions
        min_similarity: Minimum similarity threshold
    
    Returns:
        List of QuerySuggestion objects with typo corrections
    """
    normalized_term = term.lower().strip()
    if not normalized_term:
        return []
    
    # Calculate similarity to each vocabulary word
    similarities = []
    for word in vocabulary:
        word_lower = word.lower()
        if word_lower == normalized_term:
            # Exact match, no correction needed
            return []
        
        sim = SequenceMatcher(None, normalized_term, word_lower).ratio()
        if sim >= min_similarity:
            similarities.append((word, sim))
    
    # Sort by similarity and take top suggestions
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    suggestions = []
    for word, sim in similarities[:max_suggestions]:
        suggestions.append(QuerySuggestion(
            suggestion_type="did_you_mean",
            original=term,
            suggested=word,
            score=sim,
            reason=f"Similar spelling (similarity: {sim:.2f})",
        ))
    
    return suggestions


# =============================================================================
# Low-Result Query Expansion
# =============================================================================

def get_expansion_suggestions(
    conn,
    query_text: str,
    result_count: int,
    threshold: int = 5,
) -> List[QuerySuggestion]:
    """
    Suggest query expansions when result count is low.
    
    Provides suggestions to broaden a query that returned few results.
    
    Args:
        conn: Database connection
        query_text: The original query text
        result_count: Number of results the query returned
        threshold: If result_count < threshold, suggest expansions
    
    Returns:
        List of QuerySuggestion objects with expansion suggestions
    """
    if result_count >= threshold:
        return []
    
    suggestions = []
    
    # Extract terms from query
    terms = re.findall(r'\b\w+\b', query_text.lower())
    
    # For each term, find related terms that might expand the query
    for term in terms:
        related = get_related_terms(conn, term, limit=3, min_cooccurrence=3)
        for r in related:
            suggestions.append(QuerySuggestion(
                suggestion_type="expansion",
                original=query_text,
                suggested=f"{query_text} OR {r.suggested}",
                score=r.score * 0.8,  # Slightly lower confidence for expansions
                reason=f"Add related term '{r.suggested}' to expand results",
            ))
    
    # Deduplicate and limit
    seen = set()
    unique_suggestions = []
    for s in sorted(suggestions, key=lambda x: x.score, reverse=True):
        if s.suggested not in seen:
            seen.add(s.suggested)
            unique_suggestions.append(s)
            if len(unique_suggestions) >= 5:
                break
    
    return unique_suggestions


# =============================================================================
# Query Metrics Tracking
# =============================================================================

def record_query_metrics(
    conn,
    metrics: QueryMetrics,
) -> int:
    """
    Record query execution metrics to the database.
    
    Args:
        conn: Database connection
        metrics: QueryMetrics object with execution data
    
    Returns:
        ID of the recorded metrics row
    """
    with conn.cursor() as cur:
        # Create table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS query_metrics (
                id SERIAL PRIMARY KEY,
                query_text TEXT NOT NULL,
                result_count INT NOT NULL,
                execution_time_ms FLOAT NOT NULL,
                search_type TEXT NOT NULL,
                user_id TEXT,
                session_id TEXT,
                recorded_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        cur.execute("""
            INSERT INTO query_metrics (
                query_text, result_count, execution_time_ms, 
                search_type, user_id, session_id, recorded_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            metrics.query_text,
            metrics.result_count,
            metrics.execution_time_ms,
            metrics.search_type,
            metrics.user_id,
            metrics.session_id,
            metrics.recorded_at,
        ))
        
        metrics_id = cur.fetchone()[0]
        conn.commit()
        
        return metrics_id


def get_query_performance_stats(
    conn,
    query_text: Optional[str] = None,
    days: int = 30,
) -> Dict[str, Any]:
    """
    Get query performance statistics.
    
    Args:
        conn: Database connection
        query_text: Optional specific query to get stats for
        days: Number of days to look back
    
    Returns:
        Dictionary with performance statistics
    """
    with conn.cursor() as cur:
        if query_text:
            cur.execute("""
                SELECT 
                    COUNT(*) as executions,
                    AVG(result_count) as avg_results,
                    AVG(execution_time_ms) as avg_time_ms,
                    MIN(result_count) as min_results,
                    MAX(result_count) as max_results
                FROM query_metrics
                WHERE query_text = %s
                AND recorded_at >= NOW() - INTERVAL '%s days'
            """, (query_text, days))
        else:
            cur.execute("""
                SELECT 
                    COUNT(*) as executions,
                    AVG(result_count) as avg_results,
                    AVG(execution_time_ms) as avg_time_ms,
                    COUNT(DISTINCT query_text) as unique_queries
                FROM query_metrics
                WHERE recorded_at >= NOW() - INTERVAL '%s days'
            """, (days,))
        
        row = cur.fetchone()
    
    if query_text:
        return {
            "query_text": query_text,
            "executions": row[0] or 0,
            "avg_results": float(row[1] or 0),
            "avg_time_ms": float(row[2] or 0),
            "min_results": row[3] or 0,
            "max_results": row[4] or 0,
        }
    else:
        return {
            "total_executions": row[0] or 0,
            "avg_results": float(row[1] or 0),
            "avg_time_ms": float(row[2] or 0),
            "unique_queries": row[3] or 0,
            "period_days": days,
        }


def get_low_result_queries(
    conn,
    threshold: int = 5,
    limit: int = 20,
    days: int = 7,
) -> List[Dict[str, Any]]:
    """
    Get queries that returned few results (candidates for improvement).
    
    Args:
        conn: Database connection
        threshold: Result count threshold for "low"
        limit: Maximum number of queries to return
        days: Number of days to look back
    
    Returns:
        List of dictionaries with low-result query info
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT query_text, COUNT(*) as executions, 
                   AVG(result_count) as avg_results,
                   MAX(recorded_at) as last_run
            FROM query_metrics
            WHERE result_count < %s
            AND recorded_at >= NOW() - INTERVAL '%s days'
            GROUP BY query_text
            ORDER BY executions DESC
            LIMIT %s
        """, (threshold, days, limit))
        
        rows = cur.fetchall()
    
    return [
        {
            "query_text": row[0],
            "executions": row[1],
            "avg_results": float(row[2] or 0),
            "last_run": row[3].isoformat() if row[3] else None,
        }
        for row in rows
    ]


# =============================================================================
# Combined Suggestion Engine
# =============================================================================

def get_all_suggestions(
    conn,
    query_text: str,
    result_count: Optional[int] = None,
) -> Dict[str, List[QuerySuggestion]]:
    """
    Get all types of suggestions for a query.
    
    Combines related terms, typo corrections, entity matches, and
    expansion suggestions into a single response.
    
    Args:
        conn: Database connection
        query_text: The query text
        result_count: Optional result count (triggers expansion suggestions if low)
    
    Returns:
        Dictionary with suggestion lists by type
    """
    terms = re.findall(r'\b\w{3,}\b', query_text.lower())
    
    all_suggestions: Dict[str, List[QuerySuggestion]] = {
        "related_terms": [],
        "did_you_mean": [],
        "entities": [],
        "expansions": [],
    }
    
    for term in terms[:3]:  # Limit to first 3 terms
        # Related terms
        related = get_related_terms(conn, term, limit=5, min_cooccurrence=3)
        all_suggestions["related_terms"].extend(related)
        
        # Typo corrections
        typos = get_typo_suggestions(conn, term, max_suggestions=2)
        all_suggestions["did_you_mean"].extend(typos)
        
        # Entity suggestions
        entities = get_entity_suggestions(conn, term, limit=3)
        all_suggestions["entities"].extend(entities)
    
    # Expansion suggestions (only if result count is low)
    if result_count is not None and result_count < 5:
        expansions = get_expansion_suggestions(conn, query_text, result_count)
        all_suggestions["expansions"].extend(expansions)
    
    # Deduplicate each category
    for category in all_suggestions:
        seen = set()
        unique = []
        for s in sorted(all_suggestions[category], key=lambda x: x.score, reverse=True):
            if s.suggested not in seen:
                seen.add(s.suggested)
                unique.append(s)
        all_suggestions[category] = unique[:5]  # Limit each category
    
    return all_suggestions
