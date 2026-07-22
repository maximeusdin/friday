# Implementation Strategy: Two Retrieval Modes + Evidence-First Conversation

## Executive Summary

This document outlines a detailed implementation strategy for transforming the retrieval system from a single top-k model to a dual-mode architecture with explicit "thorough" (exhaustive) and "conversational" (precision) modes, grounded by evidence bundles with full match traces.

**Current State Assessment:**
- Primitives system is mature (15+ primitives, well-structured compilation)
- Search infrastructure exists (vector, lexical, hybrid via RRF)
- Evidence tracking partially implemented (`retrieval_run_chunk_evidence`)
- No pagination on result sets
- Top-k hardcoded defaults (k=20, candidate pools of 200)
- Session/plan management functional

**Target State:**
- Two explicit retrieval modes with different behaviors
- Paginated delivery for thorough mode
- Threshold-based vector search
- First-class "why surfaced" match traces
- Three new entity-centric primitives
- Evidence bundles as conversation anchors

---

## Phase 1: Database Schema & Core Infrastructure (Days 1-3)

### 1.1 New Tables

#### `retrieval_run_match_traces`
Stores the first-class "why surfaced" trace for each chunk in a retrieval run.

```sql
-- migrations/0042_match_traces.sql
CREATE TABLE retrieval_run_match_traces (
    id BIGSERIAL PRIMARY KEY,
    retrieval_run_id BIGINT NOT NULL REFERENCES retrieval_runs(id) ON DELETE CASCADE,
    chunk_id BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    
    -- Primitive match details (JSONB array)
    primitive_matches JSONB NOT NULL DEFAULT '[]',
    -- Example:
    -- [
    --   {"primitive": "ENTITY", "entity_id": 72144, "name": "Silvermaster", "hit_type": "mention"},
    --   {"primitive": "PHRASE", "value": "atomic secrets", "hit_type": "exact_match", "positions": [[23, 37]]},
    --   {"primitive": "FILTER_DATE_RANGE", "start": "1945-01-01", "end": "1950-12-31", "hit_type": "pass"},
    --   {"primitive": "CO_OCCURS_WITH", "entity_a": 72144, "entity_b": 8821, "window": "document", "hit_type": "pass"}
    -- ]
    
    -- Search type details
    search_type VARCHAR(20) NOT NULL, -- 'lex', 'vector', 'hybrid', 'scope_only'
    score_lexical FLOAT,
    score_vector FLOAT,
    score_hybrid FLOAT,
    similarity_threshold_used FLOAT,
    
    -- Rank explanation (conversational mode only)
    rank_trace JSONB, -- {"reason": "highest hybrid score", "position": 3, "score_breakdown": {...}}
    
    -- For vector search
    distance FLOAT,
    threshold_passed BOOLEAN,
    
    -- Cap/truncation info
    was_capped BOOLEAN DEFAULT FALSE,
    cap_reason VARCHAR(100), -- 'conversational_max_hits', 'safety_cap', null
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(retrieval_run_id, chunk_id)
);

CREATE INDEX idx_match_traces_run ON retrieval_run_match_traces(retrieval_run_id);
CREATE INDEX idx_match_traces_chunk ON retrieval_run_match_traces(chunk_id);
CREATE INDEX idx_match_traces_primitives ON retrieval_run_match_traces USING GIN(primitive_matches);
```

#### Extend `retrieval_runs` for mode tracking

```sql
-- migrations/0043_retrieval_runs_modes.sql
ALTER TABLE retrieval_runs ADD COLUMN retrieval_mode VARCHAR(20) DEFAULT 'conversational';
-- Values: 'conversational', 'thorough'

ALTER TABLE retrieval_runs ADD COLUMN total_hits_before_cap INTEGER;
ALTER TABLE retrieval_runs ADD COLUMN similarity_threshold FLOAT;
ALTER TABLE retrieval_runs ADD COLUMN max_hits_cap INTEGER;
ALTER TABLE retrieval_runs ADD COLUMN cap_applied BOOLEAN DEFAULT FALSE;

-- Add index for mode queries
CREATE INDEX idx_retrieval_runs_mode ON retrieval_runs(retrieval_mode);

COMMENT ON COLUMN retrieval_runs.retrieval_mode IS 'conversational: fast + explainable with caps; thorough: exhaustive, no caps, paginated';
COMMENT ON COLUMN retrieval_runs.total_hits_before_cap IS 'Total chunks matching criteria before any cap applied';
COMMENT ON COLUMN retrieval_runs.similarity_threshold IS 'Vector similarity threshold used (0.0-1.0)';
```

#### Paginated result delivery table

```sql
-- migrations/0044_result_set_chunks.sql
-- For thorough mode: store chunk IDs in a normalized table for efficient pagination

CREATE TABLE result_set_chunks (
    result_set_id BIGINT NOT NULL REFERENCES result_sets(id) ON DELETE CASCADE,
    chunk_id BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    rank INTEGER NOT NULL,
    PRIMARY KEY (result_set_id, chunk_id)
);

CREATE INDEX idx_result_set_chunks_rank ON result_set_chunks(result_set_id, rank);

-- For large result sets, this is more efficient than BIGINT[] arrays
-- Enables: SELECT chunk_id FROM result_set_chunks WHERE result_set_id = ? ORDER BY rank LIMIT ? OFFSET ?
```

### 1.2 Configuration Constants

Add to `retrieval/config.py` (new file):

```python
# retrieval/config.py
"""
Retrieval mode configuration and defaults.
"""
from dataclasses import dataclass
from typing import Literal

RetrievalMode = Literal["conversational", "thorough"]

@dataclass(frozen=True)
class ConversationalModeConfig:
    """Settings for fast, explainable retrieval."""
    default_top_k: int = 20
    max_hits_soft_cap: int = 2000  # Safety cap to protect UX
    similarity_threshold: float = 0.35  # Tuned for precision
    enable_rank_trace: bool = True
    summarization_chunk_limit: int = 50  # Max chunks to send to LLM

@dataclass(frozen=True)
class ThoroughModeConfig:
    """Settings for exhaustive retrieval."""
    similarity_threshold: float = 0.25  # Lower threshold for recall
    max_hits_hard_cap: int | None = None  # No cap by default
    pagination_default_limit: int = 100
    pagination_max_limit: int = 500
    enable_rank_trace: bool = False  # Not meaningful for exhaustive

# Trigger phrases for automatic mode detection
THOROUGH_MODE_TRIGGERS = frozenset([
    "thorough", "exhaustive", "everything", "all", "complete",
    "don't miss", "comprehensive", "full search", "every mention",
    "all occurrences", "entire corpus"
])

def detect_retrieval_mode(utterance: str) -> RetrievalMode:
    """Detect intended retrieval mode from user phrasing."""
    utterance_lower = utterance.lower()
    for trigger in THOROUGH_MODE_TRIGGERS:
        if trigger in utterance_lower:
            return "thorough"
    return "conversational"
```

---

## Phase 2: Primitives Layer Changes (Days 3-5)

### 2.1 New Primitive: `SET_RETRIEVAL_MODE`

Add to `retrieval/primitives.py`:

```python
# New primitive for explicit mode control
class SetRetrievalMode(Primitive):
    """
    SET_RETRIEVAL_MODE(mode)
    
    Explicitly sets the retrieval mode for the query.
    
    Args:
        mode: "conversational" | "thorough"
        
    conversational (default):
        - Fast, explainable results
        - Top-k with similarity threshold
        - Rank traces explain why each result surfaced
        - Soft cap to protect summarization costs
        
    thorough:
        - Exhaustive retrieval, no result caps
        - Paginated delivery
        - All chunks above threshold returned
        - Deterministic, index-driven
    """
    name = "SET_RETRIEVAL_MODE"
    
    def __init__(self, mode: Literal["conversational", "thorough"]):
        if mode not in ("conversational", "thorough"):
            raise ValueError(f"Invalid retrieval mode: {mode}")
        self.mode = mode
    
    def to_dict(self) -> dict:
        return {"primitive": self.name, "mode": self.mode}
    
    @classmethod
    def from_dict(cls, d: dict) -> "SetRetrievalMode":
        return cls(mode=d["mode"])
```

### 2.2 New Primitive: `SET_SIMILARITY_THRESHOLD`

```python
class SetSimilarityThreshold(Primitive):
    """
    SET_SIMILARITY_THRESHOLD(threshold)
    
    Sets the minimum similarity score for vector search results.
    Replaces pure top-k with threshold-based filtering.
    
    Args:
        threshold: float between 0.0 and 1.0
                   Higher = more precise, fewer results
                   Lower = more recall, more noise
                   
    Recommended ranges:
        - 0.4+ : High precision (named entity lookups)
        - 0.3-0.4 : Balanced (typical queries)
        - 0.2-0.3 : High recall (thorough searches)
    """
    name = "SET_SIMILARITY_THRESHOLD"
    
    def __init__(self, threshold: float):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be 0.0-1.0, got {threshold}")
        self.threshold = threshold
    
    def to_dict(self) -> dict:
        return {"primitive": self.name, "threshold": self.threshold}
```

### 2.3 New Entity Primitives

```python
class RelatedEntities(Primitive):
    """
    RELATED_ENTITIES(entity_id, window, top_n)
    
    Find entities that co-occur with the target entity.
    Returns aggregated entity mentions constrained by the current result set or scope.
    
    Args:
        entity_id: The source entity to find relationships for
        window: "chunk" | "document" - co-occurrence scope
        top_n: Maximum number of related entities to return (default: 20)
        
    Example:
        RELATED_ENTITIES(72144, window="document", top_n=10)
        → Returns top 10 entities that appear in the same documents as entity 72144
    """
    name = "RELATED_ENTITIES"
    
    def __init__(self, entity_id: int, window: Literal["chunk", "document"] = "document", top_n: int = 20):
        self.entity_id = entity_id
        self.window = window
        self.top_n = top_n
    
    def to_dict(self) -> dict:
        return {
            "primitive": self.name,
            "entity_id": self.entity_id,
            "window": self.window,
            "top_n": self.top_n
        }


class EntityRole(Primitive):
    """
    ENTITY_ROLE(entity_id, role)
    
    Filter to entities of a specific type/role.
    Uses the entity_type field from the entities table.
    
    Args:
        entity_id: Optional - filter specific entity by role
        role: "person" | "org" | "place" | "event" | "document"
        
    If entity_id is provided: asserts that entity has the specified role
    If entity_id is None: filters result set to chunks mentioning entities of that role
    """
    name = "ENTITY_ROLE"
    
    def __init__(self, role: str, entity_id: int | None = None):
        self.role = role
        self.entity_id = entity_id
    
    def to_dict(self) -> dict:
        return {
            "primitive": self.name,
            "role": self.role,
            "entity_id": self.entity_id
        }


class ExceptEntities(Primitive):
    """
    EXCEPT_ENTITIES(entity_ids)
    
    Exclude chunks that mention any of the specified entities.
    Simple negative filter for refining searches.
    
    Args:
        entity_ids: List of entity IDs to exclude
        
    Example:
        EXCEPT_ENTITIES([72144, 8821])
        → Exclude all chunks mentioning Silvermaster or White
    """
    name = "EXCEPT_ENTITIES"
    
    def __init__(self, entity_ids: list[int]):
        self.entity_ids = entity_ids
    
    def to_dict(self) -> dict:
        return {"primitive": self.name, "entity_ids": self.entity_ids}
```

### 2.4 Primitive Compilation Updates

Update `compile_primitives_to_scope()` to handle new primitives:

```python
def compile_primitives_to_scope(primitives: list[Primitive]) -> tuple[str, dict]:
    """
    Compile primitives to SQL WHERE clause.
    
    Returns:
        (sql_fragment, params_dict)
    """
    clauses = []
    params = {}
    
    for p in primitives:
        if isinstance(p, ExceptEntities):
            # Negative entity filter
            clauses.append("""
                c.id NOT IN (
                    SELECT DISTINCT em.chunk_id 
                    FROM entity_mentions em 
                    WHERE em.entity_id = ANY(%(except_entity_ids)s)
                )
            """)
            params["except_entity_ids"] = p.entity_ids
            
        elif isinstance(p, EntityRole):
            if p.entity_id is None:
                # Filter to chunks with entities of this role
                clauses.append("""
                    c.id IN (
                        SELECT DISTINCT em.chunk_id
                        FROM entity_mentions em
                        JOIN entities e ON em.entity_id = e.id
                        WHERE e.entity_type = %(entity_role)s
                    )
                """)
                params["entity_role"] = p.role
            else:
                # Assert specific entity has this role (validation)
                # This is checked at plan validation time
                pass
        
        # ... existing primitive handling ...
    
    return " AND ".join(clauses) if clauses else "TRUE", params
```

---

## Phase 3: Search Layer Refactoring (Days 5-8)

### 3.1 Threshold-Based Vector Search

Refactor `retrieval/ops.py`:

```python
def vector_search_threshold(
    embedding: list[float],
    threshold: float,
    max_hits: int | None = None,
    scope_sql: str | None = None,
    scope_params: dict | None = None,
) -> tuple[list[dict], int]:
    """
    Vector search with similarity threshold instead of pure top-k.
    
    Args:
        embedding: Query embedding vector
        threshold: Minimum similarity (1 - cosine_distance)
        max_hits: Optional cap on results (for conversational mode)
        scope_sql: Optional WHERE clause for scope filtering
        scope_params: Parameters for scope SQL
        
    Returns:
        (results, total_above_threshold)
        
    Note: similarity = 1 - distance for cosine distance
    """
    conn = get_connection()
    
    # Convert threshold to max distance (cosine distance = 1 - similarity)
    max_distance = 1.0 - threshold
    
    base_query = """
        WITH scored AS (
            SELECT 
                c.id as chunk_id,
                c.embedding <=> %(embedding)s::vector as distance,
                1 - (c.embedding <=> %(embedding)s::vector) as similarity
            FROM chunks c
            WHERE c.embedding IS NOT NULL
            {scope_clause}
        ),
        above_threshold AS (
            SELECT * FROM scored
            WHERE distance <= %(max_distance)s
        )
        SELECT 
            chunk_id,
            distance,
            similarity,
            (SELECT COUNT(*) FROM above_threshold) as total_count
        FROM above_threshold
        ORDER BY distance ASC
        {limit_clause}
    """
    
    scope_clause = f"AND {scope_sql}" if scope_sql else ""
    limit_clause = f"LIMIT %(max_hits)s" if max_hits else ""
    
    query = base_query.format(scope_clause=scope_clause, limit_clause=limit_clause)
    
    params = {
        "embedding": embedding,
        "max_distance": max_distance,
        **(scope_params or {}),
    }
    if max_hits:
        params["max_hits"] = max_hits
    
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    
    if not rows:
        return [], 0
    
    total_count = rows[0]["total_count"] if rows else 0
    results = [
        {
            "chunk_id": r["chunk_id"],
            "distance": r["distance"],
            "similarity": r["similarity"],
        }
        for r in rows
    ]
    
    return results, total_count
```

### 3.2 Hybrid Search Refactoring

Update hybrid search to use intersection/union model:

```python
def hybrid_search_scope_intersection(
    query_text: str,
    embedding: list[float],
    mode: Literal["union", "intersection"] = "union",
    similarity_threshold: float = 0.3,
    scope_sql: str | None = None,
    scope_params: dict | None = None,
    max_hits: int | None = None,
) -> tuple[list[dict], dict]:
    """
    Hybrid search as scope ∩ semantic.
    
    Pipeline:
        1. Lexical (tsvector) finds candidate set A
        2. Vector (threshold) finds candidate set B
        3. Union or intersection based on mode
        4. Apply scope primitives as filters
        5. Return with full scoring breakdown
        
    Args:
        query_text: Text for lexical search
        embedding: Vector for semantic search
        mode: "union" (A ∪ B) or "intersection" (A ∩ B)
        similarity_threshold: Min similarity for vector candidates
        scope_sql: WHERE clause from scope primitives
        scope_params: Parameters for scope SQL
        max_hits: Optional result cap
        
    Returns:
        (results, metadata)
        metadata includes: lexical_count, vector_count, combined_count, mode_used
    """
    # Step 1: Get lexical candidates
    lexical_candidates = lexical_search_candidates(
        query_text=query_text,
        scope_sql=scope_sql,
        scope_params=scope_params,
        max_candidates=5000,  # Large pool for filtering
    )
    
    # Step 2: Get vector candidates (threshold-based)
    vector_candidates, vector_total = vector_search_threshold(
        embedding=embedding,
        threshold=similarity_threshold,
        scope_sql=scope_sql,
        scope_params=scope_params,
        max_hits=5000,  # Large pool
    )
    
    # Step 3: Combine based on mode
    lex_ids = {r["chunk_id"] for r in lexical_candidates}
    vec_ids = {r["chunk_id"] for r in vector_candidates}
    
    if mode == "intersection":
        combined_ids = lex_ids & vec_ids
    else:  # union
        combined_ids = lex_ids | vec_ids
    
    # Step 4: Build scored results with both scores
    lex_scores = {r["chunk_id"]: r["score"] for r in lexical_candidates}
    vec_scores = {r["chunk_id"]: r["similarity"] for r in vector_candidates}
    
    results = []
    for chunk_id in combined_ids:
        score_lex = lex_scores.get(chunk_id)
        score_vec = vec_scores.get(chunk_id)
        
        # RRF-style combination for ranking
        rrf_k = 50
        rank_lex = list(lex_ids).index(chunk_id) + 1 if chunk_id in lex_ids else 10000
        rank_vec = list(vec_ids).index(chunk_id) + 1 if chunk_id in vec_ids else 10000
        score_hybrid = 1/(rrf_k + rank_lex) + 1/(rrf_k + rank_vec)
        
        results.append({
            "chunk_id": chunk_id,
            "score_lex": score_lex,
            "score_vec": score_vec,
            "score_hybrid": score_hybrid,
            "in_lexical": chunk_id in lex_ids,
            "in_vector": chunk_id in vec_ids,
        })
    
    # Sort by hybrid score
    results.sort(key=lambda x: x["score_hybrid"], reverse=True)
    
    # Apply cap if specified
    total_before_cap = len(results)
    if max_hits and len(results) > max_hits:
        results = results[:max_hits]
    
    metadata = {
        "lexical_candidate_count": len(lex_ids),
        "vector_candidate_count": len(vec_ids),
        "combined_count": len(combined_ids),
        "total_before_cap": total_before_cap,
        "cap_applied": max_hits is not None and total_before_cap > max_hits,
        "mode_used": mode,
        "similarity_threshold": similarity_threshold,
    }
    
    return results, metadata
```

---

## Phase 4: Match Trace Construction (Days 8-10)

### 4.1 Match Trace Builder

Create `retrieval/match_trace.py`:

```python
"""
Match trace construction for "why surfaced" explanations.
"""
from dataclasses import dataclass, field
from typing import Any

@dataclass
class PrimitiveHit:
    """A single primitive match for a chunk."""
    primitive: str
    hit_type: str  # 'mention', 'exact_match', 'pass', 'fail', 'boost'
    details: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "primitive": self.primitive,
            "hit_type": self.hit_type,
            **self.details
        }


@dataclass
class MatchTrace:
    """Complete match trace for a chunk in a retrieval run."""
    chunk_id: int
    primitive_hits: list[PrimitiveHit] = field(default_factory=list)
    search_type: str = ""
    score_lexical: float | None = None
    score_vector: float | None = None
    score_hybrid: float | None = None
    similarity_threshold_used: float | None = None
    distance: float | None = None
    threshold_passed: bool | None = None
    rank_trace: dict | None = None
    was_capped: bool = False
    cap_reason: str | None = None
    
    def add_entity_hit(self, entity_id: int, entity_name: str, mention_count: int = 1):
        self.primitive_hits.append(PrimitiveHit(
            primitive="ENTITY",
            hit_type="mention",
            details={
                "entity_id": entity_id,
                "name": entity_name,
                "mention_count": mention_count
            }
        ))
    
    def add_phrase_hit(self, phrase: str, positions: list[tuple[int, int]]):
        self.primitive_hits.append(PrimitiveHit(
            primitive="PHRASE",
            hit_type="exact_match",
            details={"value": phrase, "positions": positions}
        ))
    
    def add_term_hit(self, term: str, matched_lexemes: list[str]):
        self.primitive_hits.append(PrimitiveHit(
            primitive="TERM",
            hit_type="lexical_match",
            details={"value": term, "matched_lexemes": matched_lexemes}
        ))
    
    def add_filter_pass(self, filter_type: str, **filter_params):
        self.primitive_hits.append(PrimitiveHit(
            primitive=filter_type,
            hit_type="pass",
            details=filter_params
        ))
    
    def add_co_occurrence_hit(self, entity_a: int, entity_b: int, window: str):
        self.primitive_hits.append(PrimitiveHit(
            primitive="CO_OCCURS_WITH",
            hit_type="pass",
            details={
                "entity_a": entity_a,
                "entity_b": entity_b,
                "window": window
            }
        ))
    
    def to_db_row(self, retrieval_run_id: int) -> dict:
        """Convert to database row for insertion."""
        return {
            "retrieval_run_id": retrieval_run_id,
            "chunk_id": self.chunk_id,
            "primitive_matches": [h.to_dict() for h in self.primitive_hits],
            "search_type": self.search_type,
            "score_lexical": self.score_lexical,
            "score_vector": self.score_vector,
            "score_hybrid": self.score_hybrid,
            "similarity_threshold_used": self.similarity_threshold_used,
            "distance": self.distance,
            "threshold_passed": self.threshold_passed,
            "rank_trace": self.rank_trace,
            "was_capped": self.was_capped,
            "cap_reason": self.cap_reason,
        }


def build_match_traces(
    chunk_ids: list[int],
    primitives: list,
    search_results: list[dict],
    search_metadata: dict,
    mode: str,
) -> list[MatchTrace]:
    """
    Build match traces for all chunks in a retrieval result.
    
    Args:
        chunk_ids: List of retrieved chunk IDs
        primitives: Original query primitives
        search_results: Raw search results with scores
        search_metadata: Metadata from search (thresholds, caps, etc.)
        mode: "conversational" or "thorough"
        
    Returns:
        List of MatchTrace objects ready for DB insertion
    """
    traces = []
    results_by_chunk = {r["chunk_id"]: r for r in search_results}
    
    for rank, chunk_id in enumerate(chunk_ids, 1):
        result = results_by_chunk.get(chunk_id, {})
        
        trace = MatchTrace(
            chunk_id=chunk_id,
            search_type=search_metadata.get("search_type", "unknown"),
            score_lexical=result.get("score_lex"),
            score_vector=result.get("score_vec"),
            score_hybrid=result.get("score_hybrid"),
            similarity_threshold_used=search_metadata.get("similarity_threshold"),
            distance=result.get("distance"),
            threshold_passed=result.get("similarity") is not None and 
                           result.get("similarity", 0) >= search_metadata.get("similarity_threshold", 0),
            was_capped=search_metadata.get("cap_applied", False),
            cap_reason=search_metadata.get("cap_reason"),
        )
        
        # Add rank trace for conversational mode
        if mode == "conversational":
            trace.rank_trace = {
                "position": rank,
                "reason": _explain_rank(result, search_metadata),
                "score_breakdown": {
                    "lexical": result.get("score_lex"),
                    "vector": result.get("score_vec"),
                    "hybrid": result.get("score_hybrid"),
                }
            }
        
        traces.append(trace)
    
    return traces


def _explain_rank(result: dict, metadata: dict) -> str:
    """Generate human-readable rank explanation."""
    reasons = []
    
    if result.get("in_lexical") and result.get("in_vector"):
        reasons.append("matched both lexical and semantic search")
    elif result.get("in_lexical"):
        reasons.append("strong lexical match")
    elif result.get("in_vector"):
        reasons.append("strong semantic similarity")
    
    if result.get("score_hybrid"):
        reasons.append(f"hybrid score: {result['score_hybrid']:.4f}")
    
    return "; ".join(reasons) if reasons else "included in result set"
```

### 4.2 Enrich Match Traces with Primitive Details

```python
def enrich_traces_with_primitive_details(
    traces: list[MatchTrace],
    primitives: list,
    conn,
) -> list[MatchTrace]:
    """
    Post-process traces to add detailed primitive hit information.
    
    Queries the database to find which primitives actually matched each chunk.
    """
    chunk_ids = [t.chunk_id for t in traces]
    trace_map = {t.chunk_id: t for t in traces}
    
    # Extract entity primitives
    entity_primitives = [p for p in primitives if p.name == "ENTITY"]
    entity_ids = [p.entity_id for p in entity_primitives]
    
    if entity_ids:
        # Query entity mentions for these chunks
        with conn.cursor() as cur:
            cur.execute("""
                SELECT em.chunk_id, em.entity_id, e.canonical_name, COUNT(*) as mention_count
                FROM entity_mentions em
                JOIN entities e ON em.entity_id = e.id
                WHERE em.chunk_id = ANY(%(chunk_ids)s)
                  AND em.entity_id = ANY(%(entity_ids)s)
                GROUP BY em.chunk_id, em.entity_id, e.canonical_name
            """, {"chunk_ids": chunk_ids, "entity_ids": entity_ids})
            
            for row in cur.fetchall():
                trace = trace_map.get(row["chunk_id"])
                if trace:
                    trace.add_entity_hit(
                        entity_id=row["entity_id"],
                        entity_name=row["canonical_name"],
                        mention_count=row["mention_count"]
                    )
    
    # Extract phrase primitives and check matches
    phrase_primitives = [p for p in primitives if p.name == "PHRASE"]
    if phrase_primitives:
        # Query chunk text and find phrase positions
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, text FROM chunks WHERE id = ANY(%(chunk_ids)s)
            """, {"chunk_ids": chunk_ids})
            
            for row in cur.fetchall():
                trace = trace_map.get(row["id"])
                if trace:
                    text_lower = row["text"].lower()
                    for pp in phrase_primitives:
                        phrase_lower = pp.value.lower()
                        # Find all positions
                        positions = []
                        start = 0
                        while True:
                            pos = text_lower.find(phrase_lower, start)
                            if pos == -1:
                                break
                            positions.append((pos, pos + len(phrase_lower)))
                            start = pos + 1
                        
                        if positions:
                            trace.add_phrase_hit(pp.value, positions)
    
    # Add filter pass information
    date_range_primitives = [p for p in primitives if p.name == "FILTER_DATE_RANGE"]
    for drp in date_range_primitives:
        for trace in traces:
            trace.add_filter_pass(
                "FILTER_DATE_RANGE",
                start=drp.start,
                end=drp.end
            )
    
    # Add co-occurrence information
    co_occurs_primitives = [p for p in primitives if p.name == "CO_OCCURS_WITH"]
    for cop in co_occurs_primitives:
        for trace in traces:
            trace.add_co_occurrence_hit(
                entity_a=cop.entity_a,
                entity_b=cop.entity_b,
                window=cop.window
            )
    
    return traces
```

---

## Phase 5: Execution Layer Updates (Days 10-12)

### 5.1 Update Plan Execution

Modify `scripts/execute_plan.py`:

```python
def execute_plan_retrieval(
    plan: dict,
    conn,
    mode: str | None = None,
) -> dict:
    """
    Execute retrieval with mode-aware behavior.
    
    Args:
        plan: The research plan with primitives
        conn: Database connection
        mode: Override retrieval mode (or detect from plan/primitives)
        
    Returns:
        Execution result with result_set_id, traces, metadata
    """
    primitives = parse_primitives(plan["plan_json"]["query"]["primitives"])
    
    # Determine retrieval mode
    if mode is None:
        mode = extract_retrieval_mode(primitives, plan.get("raw_utterance", ""))
    
    # Get mode-specific config
    if mode == "conversational":
        config = ConversationalModeConfig()
    else:
        config = ThoroughModeConfig()
    
    # Extract search parameters
    search_type = extract_search_type(primitives) or "hybrid"
    similarity_threshold = extract_threshold(primitives) or config.similarity_threshold
    top_k = extract_top_k(primitives) or config.default_top_k if mode == "conversational" else None
    max_hits = config.max_hits_soft_cap if mode == "conversational" else config.max_hits_hard_cap
    
    # Compile scope
    scope_sql, scope_params = compile_primitives_to_scope(primitives)
    
    # Execute search based on type and mode
    if search_type == "vector":
        results, total = vector_search_threshold(
            embedding=get_query_embedding(plan),
            threshold=similarity_threshold,
            max_hits=max_hits,
            scope_sql=scope_sql,
            scope_params=scope_params,
        )
        search_metadata = {
            "search_type": "vector",
            "similarity_threshold": similarity_threshold,
            "total_above_threshold": total,
            "cap_applied": max_hits and total > max_hits,
            "cap_reason": "conversational_max_hits" if max_hits and total > max_hits else None,
        }
        
    elif search_type == "lexical":
        results = lexical_search(
            query_text=plan["plan_json"]["compiled"]["tsquery"],
            scope_sql=scope_sql,
            scope_params=scope_params,
            top_n=top_k or 1000,
        )
        search_metadata = {"search_type": "lexical"}
        
    else:  # hybrid
        results, search_metadata = hybrid_search_scope_intersection(
            query_text=plan["plan_json"]["compiled"]["expanded"],
            embedding=get_query_embedding(plan),
            similarity_threshold=similarity_threshold,
            scope_sql=scope_sql,
            scope_params=scope_params,
            max_hits=max_hits,
        )
    
    # Apply top-k for conversational mode
    if mode == "conversational" and top_k and len(results) > top_k:
        results = results[:top_k]
    
    chunk_ids = [r["chunk_id"] for r in results]
    
    # Build match traces
    traces = build_match_traces(
        chunk_ids=chunk_ids,
        primitives=primitives,
        search_results=results,
        search_metadata=search_metadata,
        mode=mode,
    )
    
    # Enrich with primitive details
    traces = enrich_traces_with_primitive_details(traces, primitives, conn)
    
    # Log retrieval run
    retrieval_run_id = log_retrieval_run(
        conn=conn,
        plan_id=plan["id"],
        mode=mode,
        search_metadata=search_metadata,
        chunk_ids=chunk_ids,
    )
    
    # Persist match traces
    persist_match_traces(conn, retrieval_run_id, traces)
    
    # Create result set
    if mode == "thorough":
        # Use normalized table for pagination support
        result_set_id = create_paginated_result_set(
            conn=conn,
            retrieval_run_id=retrieval_run_id,
            chunk_ids=chunk_ids,
        )
    else:
        # Standard result set with array
        result_set_id = create_result_set(
            conn=conn,
            retrieval_run_id=retrieval_run_id,
            chunk_ids=chunk_ids,
        )
    
    return {
        "result_set_id": result_set_id,
        "retrieval_run_id": retrieval_run_id,
        "mode": mode,
        "total_results": len(chunk_ids),
        "total_before_cap": search_metadata.get("total_before_cap", len(chunk_ids)),
        "cap_applied": search_metadata.get("cap_applied", False),
        "traces_count": len(traces),
    }
```

---

## Phase 6: API Layer Updates (Days 12-14)

### 6.1 Paginated Result Set Endpoint

Add to `backend/app/routes/results.py`:

```python
from fastapi import Query

@router.get("/result-sets/{result_set_id}/chunks")
async def get_result_set_chunks_paginated(
    result_set_id: int,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db=Depends(get_db),
):
    """
    Paginated chunk retrieval for thorough mode result sets.
    
    Supports indefinite "show more" in the UI without loading
    all chunks at once.
    """
    # Get result set metadata
    result_set = await get_result_set_metadata(db, result_set_id)
    if not result_set:
        raise HTTPException(404, "Result set not found")
    
    # Check if using normalized chunks table (thorough mode)
    if result_set.get("is_paginated"):
        query = """
            SELECT 
                rsc.chunk_id,
                rsc.rank,
                c.text,
                cm.document_id,
                cm.collection_slug,
                cm.date_min,
                cm.date_max
            FROM result_set_chunks rsc
            JOIN chunks c ON rsc.chunk_id = c.id
            LEFT JOIN chunk_metadata cm ON c.id = cm.chunk_id
            WHERE rsc.result_set_id = %(result_set_id)s
            ORDER BY rsc.rank
            OFFSET %(offset)s
            LIMIT %(limit)s
        """
        chunks = await db.fetch_all(query, {
            "result_set_id": result_set_id,
            "offset": offset,
            "limit": limit,
        })
        
        # Get total count
        total = await db.fetch_val(
            "SELECT COUNT(*) FROM result_set_chunks WHERE result_set_id = %(id)s",
            {"id": result_set_id}
        )
    else:
        # Legacy array-based result set
        chunk_ids = result_set["chunk_ids"]
        total = len(chunk_ids)
        page_ids = chunk_ids[offset:offset + limit]
        
        chunks = await fetch_chunks_by_ids(db, page_ids)
    
    return {
        "result_set_id": result_set_id,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + limit < total,
        "chunks": [format_chunk_response(c) for c in chunks],
    }


@router.get("/result-sets/{result_set_id}/match-traces")
async def get_match_traces(
    result_set_id: int,
    chunk_ids: list[int] = Query(None),
    db=Depends(get_db),
):
    """
    Get match traces explaining why chunks surfaced.
    
    Args:
        result_set_id: The result set to get traces for
        chunk_ids: Optional filter to specific chunks
    """
    # Get retrieval run ID from result set
    result_set = await get_result_set_metadata(db, result_set_id)
    if not result_set:
        raise HTTPException(404, "Result set not found")
    
    retrieval_run_id = result_set["retrieval_run_id"]
    
    query = """
        SELECT 
            chunk_id,
            primitive_matches,
            search_type,
            score_lexical,
            score_vector,
            score_hybrid,
            similarity_threshold_used,
            rank_trace,
            was_capped,
            cap_reason
        FROM retrieval_run_match_traces
        WHERE retrieval_run_id = %(run_id)s
    """
    params = {"run_id": retrieval_run_id}
    
    if chunk_ids:
        query += " AND chunk_id = ANY(%(chunk_ids)s)"
        params["chunk_ids"] = chunk_ids
    
    traces = await db.fetch_all(query, params)
    
    return {
        "result_set_id": result_set_id,
        "retrieval_run_id": retrieval_run_id,
        "traces": [
            {
                "chunk_id": t["chunk_id"],
                "primitive_matches": t["primitive_matches"],
                "search_type": t["search_type"],
                "scores": {
                    "lexical": t["score_lexical"],
                    "vector": t["score_vector"],
                    "hybrid": t["score_hybrid"],
                },
                "threshold": t["similarity_threshold_used"],
                "rank_explanation": t["rank_trace"],
                "was_capped": t["was_capped"],
                "cap_reason": t["cap_reason"],
            }
            for t in traces
        ]
    }
```

### 6.2 Entity Aggregation Endpoints

```python
@router.get("/result-sets/{result_set_id}/entities")
async def get_result_set_entities(
    result_set_id: int,
    limit: int = Query(50, ge=1, le=200),
    db=Depends(get_db),
):
    """
    Get top entities mentioned in a result set.
    
    Powers the "Top entities in these results" sidebar.
    """
    result_set = await get_result_set_metadata(db, result_set_id)
    if not result_set:
        raise HTTPException(404, "Result set not found")
    
    chunk_ids = await get_result_set_chunk_ids(db, result_set_id)
    
    query = """
        SELECT 
            e.id as entity_id,
            e.canonical_name,
            e.entity_type,
            COUNT(DISTINCT em.chunk_id) as chunk_count,
            COUNT(*) as mention_count
        FROM entity_mentions em
        JOIN entities e ON em.entity_id = e.id
        WHERE em.chunk_id = ANY(%(chunk_ids)s)
        GROUP BY e.id, e.canonical_name, e.entity_type
        ORDER BY chunk_count DESC, mention_count DESC
        LIMIT %(limit)s
    """
    
    entities = await db.fetch_all(query, {
        "chunk_ids": chunk_ids,
        "limit": limit,
    })
    
    return {
        "result_set_id": result_set_id,
        "total_chunks": len(chunk_ids),
        "entities": [
            {
                "entity_id": e["entity_id"],
                "name": e["canonical_name"],
                "type": e["entity_type"],
                "chunk_count": e["chunk_count"],
                "mention_count": e["mention_count"],
            }
            for e in entities
        ]
    }


@router.get("/result-sets/{result_set_id}/co-mentions")
async def get_co_mentioned_entities(
    result_set_id: int,
    entity_id: int,
    window: str = Query("document", regex="^(chunk|document)$"),
    limit: int = Query(20, ge=1, le=100),
    db=Depends(get_db),
):
    """
    Get entities that co-occur with a target entity within the result set.
    
    Powers "Entities co-mentioned with X" feature.
    """
    result_set = await get_result_set_metadata(db, result_set_id)
    if not result_set:
        raise HTTPException(404, "Result set not found")
    
    chunk_ids = await get_result_set_chunk_ids(db, result_set_id)
    
    if window == "chunk":
        query = """
            WITH target_chunks AS (
                SELECT DISTINCT chunk_id
                FROM entity_mentions
                WHERE entity_id = %(entity_id)s
                  AND chunk_id = ANY(%(chunk_ids)s)
            )
            SELECT 
                e.id as entity_id,
                e.canonical_name,
                e.entity_type,
                COUNT(DISTINCT em.chunk_id) as co_occurrence_count
            FROM entity_mentions em
            JOIN entities e ON em.entity_id = e.id
            WHERE em.chunk_id IN (SELECT chunk_id FROM target_chunks)
              AND em.entity_id != %(entity_id)s
            GROUP BY e.id, e.canonical_name, e.entity_type
            ORDER BY co_occurrence_count DESC
            LIMIT %(limit)s
        """
    else:  # document
        query = """
            WITH target_documents AS (
                SELECT DISTINCT document_id
                FROM entity_mentions
                WHERE entity_id = %(entity_id)s
                  AND chunk_id = ANY(%(chunk_ids)s)
            )
            SELECT 
                e.id as entity_id,
                e.canonical_name,
                e.entity_type,
                COUNT(DISTINCT em.document_id) as co_occurrence_count
            FROM entity_mentions em
            JOIN entities e ON em.entity_id = e.id
            WHERE em.document_id IN (SELECT document_id FROM target_documents)
              AND em.entity_id != %(entity_id)s
              AND em.chunk_id = ANY(%(chunk_ids)s)
            GROUP BY e.id, e.canonical_name, e.entity_type
            ORDER BY co_occurrence_count DESC
            LIMIT %(limit)s
        """
    
    entities = await db.fetch_all(query, {
        "entity_id": entity_id,
        "chunk_ids": chunk_ids,
        "limit": limit,
    })
    
    return {
        "result_set_id": result_set_id,
        "source_entity_id": entity_id,
        "window": window,
        "co_mentioned_entities": [
            {
                "entity_id": e["entity_id"],
                "name": e["canonical_name"],
                "type": e["entity_type"],
                "co_occurrence_count": e["co_occurrence_count"],
            }
            for e in entities
        ]
    }
```

### 6.3 Evidence Bundle Response Format

Update result item response to include evidence bundles:

```python
@dataclass
class EvidenceBundle:
    """Evidence bundle for a single chunk in results."""
    chunk_id: int
    document_id: int
    page_range: str  # "pp. 45-47"
    snippet: str  # Highlighted text excerpt
    matched_primitives: list[dict]  # From match trace
    scores: dict  # lex, vec, hybrid
    
    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "page_range": self.page_range,
            "snippet": self.snippet,
            "matched_primitives": self.matched_primitives,
            "scores": self.scores,
        }


async def build_evidence_bundles(
    db,
    result_set_id: int,
    chunk_ids: list[int],
) -> list[EvidenceBundle]:
    """
    Build evidence bundles from result set chunks.
    
    Each bundle contains:
    - Chunk reference (id, document, pages)
    - Snippet with highlights
    - Match trace (why surfaced)
    - Scores breakdown
    """
    # Get chunk details with page info
    chunk_query = """
        SELECT 
            c.id as chunk_id,
            c.text,
            cm.document_id,
            cm.collection_slug,
            d.source_name,
            COALESCE(
                (SELECT string_agg(p.logical_page_label, '-' ORDER BY cp.span_order)
                 FROM chunk_pages cp
                 JOIN pages p ON cp.page_id = p.id
                 WHERE cp.chunk_id = c.id),
                'n/a'
            ) as page_range
        FROM chunks c
        LEFT JOIN chunk_metadata cm ON c.id = cm.chunk_id
        LEFT JOIN documents d ON cm.document_id = d.id
        WHERE c.id = ANY(%(chunk_ids)s)
    """
    chunks = await db.fetch_all(chunk_query, {"chunk_ids": chunk_ids})
    chunk_map = {c["chunk_id"]: c for c in chunks}
    
    # Get match traces
    result_set = await get_result_set_metadata(db, result_set_id)
    trace_query = """
        SELECT chunk_id, primitive_matches, score_lexical, score_vector, score_hybrid
        FROM retrieval_run_match_traces
        WHERE retrieval_run_id = %(run_id)s
          AND chunk_id = ANY(%(chunk_ids)s)
    """
    traces = await db.fetch_all(trace_query, {
        "run_id": result_set["retrieval_run_id"],
        "chunk_ids": chunk_ids,
    })
    trace_map = {t["chunk_id"]: t for t in traces}
    
    bundles = []
    for chunk_id in chunk_ids:
        chunk = chunk_map.get(chunk_id, {})
        trace = trace_map.get(chunk_id, {})
        
        bundles.append(EvidenceBundle(
            chunk_id=chunk_id,
            document_id=chunk.get("document_id"),
            page_range=f"pp. {chunk.get('page_range', 'n/a')}",
            snippet=truncate_with_highlight(chunk.get("text", ""), 300),
            matched_primitives=trace.get("primitive_matches", []),
            scores={
                "lexical": trace.get("score_lexical"),
                "vector": trace.get("score_vector"),
                "hybrid": trace.get("score_hybrid"),
            }
        ))
    
    return bundles
```

---

## Phase 7: Planner Integration (Day 14)

### 7.1 Update Planner Prompts

Add retrieval mode awareness to the planner:

```python
PLANNER_SYSTEM_PROMPT_ADDITION = """
## Retrieval Modes

The system supports two retrieval modes. Choose based on user intent:

### conversational (default)
- Fast, explainable results
- Returns top-k results with rank explanations
- Use for: typical questions, exploration, "tell me about X"
- Primitive: SET_RETRIEVAL_MODE("conversational")

### thorough
- Exhaustive retrieval, no result caps
- Paginated delivery for large result sets
- Use for: "find everything", "don't miss anything", "all mentions of"
- Primitive: SET_RETRIEVAL_MODE("thorough")

### Mode Detection
Automatically use "thorough" when user says:
- "thorough", "exhaustive", "everything", "complete"
- "all occurrences", "every mention", "don't miss"
- "comprehensive search", "full corpus"

### Example Plans

User: "Tell me about Silvermaster's role in the network"
→ SET_RETRIEVAL_MODE("conversational"), ENTITY(72144), SET_TOP_K(20)

User: "Find every document mentioning both Silvermaster and White"
→ SET_RETRIEVAL_MODE("thorough"), CO_OCCURS_WITH(72144, 8821, "document")

User: "Who else appears in documents about Silvermaster?"
→ RELATED_ENTITIES(72144, window="document", top_n=20)

User: "Show me network members but exclude Silvermaster"
→ FILTER_COLLECTION("silvermaster"), EXCEPT_ENTITIES([72144])
"""
```

### 7.2 Response Format with Evidence Bundles

```python
SYNTHESIS_PROMPT_WITH_EVIDENCE = """
You are synthesizing an answer based on retrieved evidence.

## Evidence Bundles
Each evidence bundle contains:
- chunk_id: Unique identifier for citation
- page_range: Source pages
- snippet: Relevant text excerpt
- matched_primitives: Why this evidence surfaced

## Response Format
1. Answer the question directly
2. Cite evidence inline using [chunk_id] format
3. For each major claim, reference supporting bundles
4. Include a "Sources" section listing cited chunks

## Example Response
Based on the evidence, Silvermaster operated as a key coordinator [chunk_12345].
The network included Treasury Department employees [chunk_12346, chunk_12347].

**Sources:**
- [chunk_12345]: Silvermaster file, pp. 23-25 - describes coordination role
- [chunk_12346]: Treasury memo, pp. 1-2 - names department members
"""
```

---

## Implementation Checklist

### Week 1 (Days 1-7)

- [ ] **Day 1-2**: Database migrations
  - [ ] Create `retrieval_run_match_traces` table
  - [ ] Extend `retrieval_runs` with mode columns
  - [ ] Create `result_set_chunks` table for pagination
  - [ ] Run migrations on dev

- [ ] **Day 3-4**: Primitives layer
  - [ ] Add `SET_RETRIEVAL_MODE` primitive
  - [ ] Add `SET_SIMILARITY_THRESHOLD` primitive
  - [ ] Add `RELATED_ENTITIES` primitive
  - [ ] Add `ENTITY_ROLE` primitive
  - [ ] Add `EXCEPT_ENTITIES` primitive
  - [ ] Update primitive compilation

- [ ] **Day 5-7**: Search layer refactoring
  - [ ] Implement `vector_search_threshold()`
  - [ ] Implement `hybrid_search_scope_intersection()`
  - [ ] Add mode-aware config loading
  - [ ] Unit tests for search functions

### Week 2 (Days 8-14)

- [ ] **Day 8-10**: Match trace construction
  - [ ] Create `retrieval/match_trace.py`
  - [ ] Implement `build_match_traces()`
  - [ ] Implement `enrich_traces_with_primitive_details()`
  - [ ] Integration tests

- [ ] **Day 10-12**: Execution layer
  - [ ] Update `execute_plan_retrieval()` for modes
  - [ ] Add trace persistence
  - [ ] Add paginated result set creation
  - [ ] End-to-end tests

- [ ] **Day 12-14**: API layer
  - [ ] Add `/result-sets/{id}/chunks` pagination endpoint
  - [ ] Add `/result-sets/{id}/match-traces` endpoint
  - [ ] Add `/result-sets/{id}/entities` endpoint
  - [ ] Add `/result-sets/{id}/co-mentions` endpoint
  - [ ] Update planner prompts
  - [ ] API tests

---

## Configuration Defaults

```python
# retrieval/config.py - Final configuration

CONVERSATIONAL_DEFAULTS = {
    "top_k": 20,
    "max_hits_soft_cap": 2000,
    "similarity_threshold": 0.35,
    "enable_rank_trace": True,
    "summarization_chunk_limit": 50,
}

THOROUGH_DEFAULTS = {
    "similarity_threshold": 0.25,
    "max_hits_hard_cap": None,  # No cap
    "pagination_default_limit": 100,
    "pagination_max_limit": 500,
    "enable_rank_trace": False,
}

HYBRID_SEARCH_DEFAULTS = {
    "candidate_pool_size": 5000,
    "rrf_k": 50,
    "default_mode": "union",  # union or intersection
}
```

---

## Testing Strategy

### Unit Tests
- Primitive parsing and compilation
- Threshold-based vector search
- Match trace construction
- Mode detection from utterance

### Integration Tests
- Full plan execution in both modes
- Pagination correctness
- Entity aggregation queries
- Co-mention queries

### End-to-End Tests
- Conversational flow: query → plan → execute → summarize
- Thorough flow: query → plan → execute → paginate
- Mode switching within session
- Evidence bundle formatting

---

## Migration Path

1. Deploy database migrations (backward compatible)
2. Deploy search layer changes (feature-flagged)
3. Deploy API endpoints (additive)
4. Deploy planner changes (prompt update)
5. Enable feature flags
6. Monitor and tune thresholds

---

## Success Metrics

- **Thorough mode**: 100% recall within scope constraints
- **Conversational mode**: <2s latency for typical queries
- **Match traces**: 100% of results have complete trace
- **Pagination**: Support result sets >10k chunks
- **Entity endpoints**: <500ms for aggregation queries
