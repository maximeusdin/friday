# Agentic Workflow V2: FocusBundle Architecture

## Core Invariants (Non-Negotiable)

1. **All citations must come from FocusBundle spans**
   - If it's not in FocusBundle, it cannot be cited, cannot support a claim, cannot appear as evidence

2. **All rendered statements must map to supported evidence**
   - Each bullet must reference 1+ FocusSpans
   - Rejected if it introduces unsupported facts

Everything else is flexible. These two invariants prevent nonsense.

---

## Phase 1: FocusBundle Gating (Highest Leverage)

**Goal:** Stop spewing citations, reduce false confidence, eliminate hub dominance.

### 1.1 New Files to Create

```
retrieval/spans.py          → SpanMiner
retrieval/focus_bundle.py   → FocusBundleBuilder + MMR selection  
migrations/0049_focus_spans.sql → focus_spans table
```

### 1.2 SpanMiner (`retrieval/spans.py`)

```python
@dataclass
class Span:
    chunk_id: int
    doc_id: int
    page_ref: str
    start_char: int
    end_char: int
    text: str
    
    @property
    def span_id(self) -> str:
        """Deterministic ID: chunk_id:start:end"""
        return f"{self.chunk_id}:{self.start_char}:{self.end_char}"

class SpanMiner:
    """
    Converts chunk text into cite-able windows.
    
    Window types:
    - sentence windows (1-2 sentences)
    - fixed char windows (500 chars, 50% overlap)
    - mention-centered windows (if mention offsets available)
    
    Default: sentence splitting + merge short sentences to 120-600 chars.
    """
    
    def __init__(
        self,
        min_chars: int = 120,
        max_chars: int = 600,
        max_spans_per_chunk: int = 12,
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.max_spans_per_chunk = max_spans_per_chunk
    
    def mine_spans(
        self, 
        chunk_id: int,
        doc_id: int,
        page_ref: str,
        text: str,
        anchor_terms: List[str] = None,  # prioritize spans containing these
    ) -> List[Span]:
        """
        1. Split into sentence windows
        2. Merge short windows to target length
        3. Cap spans per chunk using deterministic prefilter:
           - Spans containing anchors first
           - Otherwise evenly spaced
        """
        pass
    
    def mine_chunks(
        self,
        chunks: List[ChunkWithProvenance],
        anchor_terms: List[str] = None,
    ) -> List[Span]:
        """Mine spans from multiple chunks."""
        all_spans = []
        for chunk in chunks:
            spans = self.mine_spans(
                chunk.chunk_id,
                chunk.doc_id,
                "",  # page_ref from metadata
                chunk.text,
                anchor_terms,
            )
            all_spans.extend(spans)
        return all_spans
```

### 1.3 FocusBundleBuilder (`retrieval/focus_bundle.py`)

```python
@dataclass
class FocusSpan:
    """A span that has been selected for the FocusBundle."""
    chunk_id: int
    doc_id: int
    page_ref: str
    start_char: int
    end_char: int
    text: str
    score: float              # similarity to query
    rank: int                 # position in FocusBundle
    source_lanes: List[str]   # which retrieval lanes found this chunk
    
    @property
    def span_id(self) -> str:
        return f"{self.chunk_id}:{self.start_char}:{self.end_char}"
    
    def to_evidence_ref(self) -> dict:
        """Convert to citation format."""
        return {
            "span_id": self.span_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "page_ref": self.page_ref,
            "char_range": [self.start_char, self.end_char],
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "score": self.score,
        }

@dataclass
class FocusBundle:
    """
    The single source of truth for citations.
    
    HARD RULE: Only FocusSpans can be cited.
    """
    query_text: str
    spans: List[FocusSpan]
    params: Dict[str, Any]    # builder params for reproducibility
    retrieval_run_id: Optional[int] = None
    
    def get_span(self, span_id: str) -> Optional[FocusSpan]:
        """Look up span by ID."""
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None
    
    def contains_span(self, span_id: str) -> bool:
        """Check if span is in FocusBundle (citation validation)."""
        return any(s.span_id == span_id for s in self.spans)
    
    def get_spans_containing(self, text: str) -> List[FocusSpan]:
        """Get all spans containing a text substring."""
        text_lower = text.lower()
        return [s for s in self.spans if text_lower in s.text.lower()]


class FocusBundleBuilder:
    """
    Builds FocusBundle from retrieved chunks.
    
    FocusBundleBuilder(query, retrieved_chunks) -> FocusBundle
    
    Process:
    1. Mine spans from chunks
    2. Embed query once
    3. Embed spans (on-the-fly)
    4. Rank spans by similarity
    5. Apply soft diversity (don't let one doc flood)
    6. Output top N spans
    """
    
    def __init__(
        self,
        top_n_spans: int = 80,
        diversity_weight: float = 0.3,  # MMR lambda
        min_span_score: float = 0.3,    # minimum similarity threshold
    ):
        self.top_n_spans = top_n_spans
        self.diversity_weight = diversity_weight
        self.min_span_score = min_span_score
        self.span_miner = SpanMiner()
    
    def build(
        self,
        query_text: str,
        chunks: List[ChunkWithProvenance],
        anchor_terms: List[str] = None,
        conn = None,
    ) -> FocusBundle:
        """
        1. Mine spans from chunks
        2. Embed query
        3. Score spans by similarity
        4. Select top N with MMR diversity
        5. Return FocusBundle
        """
        # Mine spans
        all_spans = self.span_miner.mine_chunks(chunks, anchor_terms)
        
        # Embed query
        query_embedding = self._embed_query(query_text)
        
        # Score and rank spans
        scored_spans = self._score_spans(all_spans, query_embedding)
        
        # Select with diversity
        selected = self._select_with_diversity(scored_spans)
        
        return FocusBundle(
            query_text=query_text,
            spans=selected,
            params={
                "top_n_spans": self.top_n_spans,
                "diversity_weight": self.diversity_weight,
                "min_span_score": self.min_span_score,
                "total_spans_mined": len(all_spans),
                "total_chunks": len(chunks),
            },
        )
    
    def _embed_query(self, query_text: str) -> List[float]:
        """Embed query using OpenAI."""
        from retrieval.ops import embed_query
        return embed_query(query_text)
    
    def _score_spans(
        self,
        spans: List[Span],
        query_embedding: List[float],
    ) -> List[Tuple[Span, float]]:
        """Score spans by cosine similarity to query."""
        # Batch embed spans
        span_texts = [s.text for s in spans]
        span_embeddings = self._batch_embed(span_texts)
        
        # Compute similarities
        scored = []
        for span, emb in zip(spans, span_embeddings):
            sim = self._cosine_similarity(query_embedding, emb)
            if sim >= self.min_span_score:
                scored.append((span, sim))
        
        # Sort by score descending
        scored.sort(key=lambda x: -x[1])
        return scored
    
    def _select_with_diversity(
        self,
        scored_spans: List[Tuple[Span, float]],
    ) -> List[FocusSpan]:
        """
        MMR-style selection for diversity.
        
        Don't let one doc flood the results.
        """
        selected = []
        selected_embeddings = []
        doc_counts = {}
        
        for span, score in scored_spans:
            if len(selected) >= self.top_n_spans:
                break
            
            # Soft diversity: penalize docs that already have many spans
            doc_id = span.doc_id
            doc_count = doc_counts.get(doc_id, 0)
            diversity_penalty = 0.1 * doc_count  # small penalty per existing span
            
            adjusted_score = score - diversity_penalty
            
            # Still accept if above threshold
            if adjusted_score >= self.min_span_score * 0.8:
                focus_span = FocusSpan(
                    chunk_id=span.chunk_id,
                    doc_id=span.doc_id,
                    page_ref=span.page_ref,
                    start_char=span.start_char,
                    end_char=span.end_char,
                    text=span.text,
                    score=score,
                    rank=len(selected) + 1,
                    source_lanes=[],  # filled from chunk provenance
                )
                selected.append(focus_span)
                doc_counts[doc_id] = doc_count + 1
        
        return selected
    
    def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Batch embed texts using OpenAI."""
        # Implementation with batching for efficiency
        pass
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity."""
        import numpy as np
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

### 1.4 Database Migration (`migrations/0049_focus_spans.sql`)

```sql
-- Focus spans per retrieval run
-- Only these spans can be cited in answers

CREATE TABLE IF NOT EXISTS focus_spans (
    retrieval_run_id BIGINT NOT NULL REFERENCES retrieval_runs(id) ON DELETE CASCADE,
    chunk_id BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    start_char INT NOT NULL,
    end_char INT NOT NULL,
    score NUMERIC NOT NULL,
    rank INT NOT NULL,
    source_lanes TEXT[] NOT NULL DEFAULT '{}',
    span_text TEXT,  -- cached for display
    
    PRIMARY KEY (retrieval_run_id, chunk_id, start_char, end_char)
);

CREATE INDEX idx_focus_spans_run ON focus_spans(retrieval_run_id);
CREATE INDEX idx_focus_spans_rank ON focus_spans(retrieval_run_id, rank);

-- Add focus_bundle_json to result_sets for full audit
ALTER TABLE result_sets 
    ADD COLUMN IF NOT EXISTS focus_bundle_json JSONB;
```

### 1.5 Modify Existing Code

**`retrieval/verifier.py`** → `retrieval/verifier_v2.py`

```python
class FocusBundleVerifier:
    """
    Enforces the two invariants:
    1. All citations must be in FocusBundle
    2. All rendered statements must map to supported evidence
    """
    
    def verify_citation(
        self,
        span_id: str,
        focus_bundle: FocusBundle,
    ) -> bool:
        """Check if a citation is valid."""
        return focus_bundle.contains_span(span_id)
    
    def verify_bullet(
        self,
        bullet_text: str,
        cited_span_ids: List[str],
        focus_bundle: FocusBundle,
    ) -> Tuple[bool, List[str]]:
        """
        Verify a bullet point:
        1. All cited spans must be in FocusBundle
        2. Key facts must be grounded in cited spans
        
        Returns (passed, errors)
        """
        errors = []
        
        # Check all citations are valid
        for span_id in cited_span_ids:
            if not focus_bundle.contains_span(span_id):
                errors.append(f"Citation {span_id} not in FocusBundle")
        
        # Could add: fact grounding check using NLI or keyword overlap
        
        return len(errors) == 0, errors
    
    def verify_answer(
        self,
        rendered_answer: "RenderedAnswer",
        focus_bundle: FocusBundle,
    ) -> "VerificationResult":
        """
        Verify entire answer against FocusBundle.
        """
        all_errors = []
        
        for bullet in rendered_answer.bullets:
            passed, errors = self.verify_bullet(
                bullet.text,
                bullet.cited_span_ids,
                focus_bundle,
            )
            all_errors.extend(errors)
        
        return VerificationResult(
            passed=len(all_errors) == 0,
            errors=all_errors,
        )
```

**`backend/app/services/summarizer/synthesis.py`** → constrained rendering

```python
def render_from_focus_bundle(
    focus_bundle: FocusBundle,
    candidates: List[ScoredCandidate],
    max_items: int = 25,
    max_citations_per_item: int = 2,
) -> RenderedAnswer:
    """
    Render answer from FocusBundle ONLY.
    
    STRICT RULES:
    1. Each bullet cites 1-2 FocusSpans
    2. Every fact must be grounded in cited spans
    3. No facts from outside FocusBundle
    """
    bullets = []
    
    for candidate in candidates[:max_items]:
        # Get best supporting spans for this candidate
        supporting_spans = focus_bundle.get_spans_containing(candidate.name)
        
        if not supporting_spans:
            continue  # Skip candidates without FocusBundle support
        
        # Take top 1-2 spans by score, with doc diversity
        best_spans = select_diverse_spans(
            supporting_spans, 
            max_count=max_citations_per_item,
        )
        
        # Build bullet text grounded in spans
        bullet = build_grounded_bullet(candidate, best_spans)
        bullets.append(bullet)
    
    return RenderedAnswer(
        short_answer=build_short_answer(bullets),
        bullets=bullets,
        focus_bundle_id=focus_bundle.retrieval_run_id,
    )
```

### 1.6 Integration with CLI

Modify `scripts/friday_cli.py`:

```python
def execute_agentic_query_v2(conn, session_id: int, query_text: str):
    """
    V2 Agentic workflow with FocusBundle gating.
    """
    # Step 1: Retrieval (unchanged, but quota-scoped)
    chunks = run_retrieval_lanes(query_text, conn, max_chunks=300)
    
    # Step 2: Build FocusBundle
    builder = FocusBundleBuilder(top_n_spans=80)
    focus_bundle = builder.build(query_text, chunks, conn=conn)
    
    # Step 3: Extract candidates FROM FocusBundle ONLY
    candidates = extract_candidates_from_focus(focus_bundle, conn)
    
    # Step 4: Score candidates (support + hubness in Phase 2)
    scored = score_candidates(candidates, focus_bundle)
    
    # Step 5: Render with FocusBundle gating
    rendered = render_from_focus_bundle(focus_bundle, scored)
    
    # Step 6: Verify invariants
    verifier = FocusBundleVerifier()
    result = verifier.verify_answer(rendered, focus_bundle)
    
    if not result.passed:
        # Log errors, potentially retry
        pass
    
    return {
        "rendered_answer": rendered,
        "focus_bundle": focus_bundle,
        "verification": result,
    }
```

### 1.7 Acceptance Criteria (Phase 1)

- [ ] A query that previously produced 1000+ citations now produces:
  - `focus_spans`: <= 120
  - `citations in answer`: <= 50
- [ ] "Moscow/NYU" style hub outputs drop sharply
- [ ] Every citation in the answer is verifiable against `focus_spans` table
- [ ] Span text is shown instead of "chunk:X" references

---

## Phase 2: Hubness Penalty

**Goal:** Prevent hub entities (Moscow, NYU, etc.) from dominating results.

### 2.1 New Files

```
retrieval/hubness.py        → entity_df computation + scoring
migrations/0050_entity_df.sql → entity_df table
```

### 2.2 Entity Document Frequency Table

```sql
-- migrations/0050_entity_df.sql

CREATE TABLE IF NOT EXISTS entity_df (
    entity_id BIGINT PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
    doc_df INT NOT NULL,      -- number of unique documents mentioning entity
    chunk_df INT NOT NULL,    -- number of unique chunks mentioning entity
    updated_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX idx_entity_df_doc ON entity_df(doc_df DESC);

-- Refresh function
CREATE OR REPLACE FUNCTION refresh_entity_df() RETURNS void AS $$
BEGIN
    TRUNCATE entity_df;
    
    INSERT INTO entity_df (entity_id, doc_df, chunk_df, updated_at)
    SELECT 
        em.entity_id,
        COUNT(DISTINCT cm.document_id) as doc_df,
        COUNT(DISTINCT em.chunk_id) as chunk_df,
        now()
    FROM entity_mentions em
    JOIN chunk_metadata cm ON cm.chunk_id = em.chunk_id
    GROUP BY em.entity_id;
END;
$$ LANGUAGE plpgsql;
```

### 2.3 Hubness Scoring (`retrieval/hubness.py`)

```python
@dataclass
class CandidateScore:
    candidate_id: str
    name: str
    support: float       # best FocusSpan score containing this candidate
    df_focus: int        # count in FocusBundle
    df_global: int       # count in whole corpus
    specificity: float   # log((df_focus+α)/(df_global+α))
    final_score: float   # support + λ * clamp(specificity)

def score_candidates_with_hubness(
    candidates: List[EntityCandidate],
    focus_bundle: FocusBundle,
    entity_df: Dict[int, Tuple[int, int]],  # entity_id -> (doc_df, chunk_df)
    lambda_weight: float = 0.3,
    spec_floor: float = -2.0,
    alpha: float = 1.0,
) -> List[CandidateScore]:
    """
    Score candidates using support + hubness penalty.
    
    support(x) = best FocusSpan score among spans containing x
    spec(x) = log((df_focus(x)+α) / (df_global(x)+α))
    final_score(x) = support(x) + λ * clamp(spec(x), spec_floor)
    
    This prevents hubs from winning even if co-mentioned everywhere.
    """
    scores = []
    
    for candidate in candidates:
        # Get support score
        supporting_spans = focus_bundle.get_spans_containing(candidate.display_name)
        support = max((s.score for s in supporting_spans), default=0.0)
        
        # Get frequency stats
        df_focus = len(supporting_spans)
        df_global = 0
        if candidate.entity_id and candidate.entity_id in entity_df:
            df_global = entity_df[candidate.entity_id][1]  # chunk_df
        
        # Compute specificity (IDF-like)
        specificity = math.log((df_focus + alpha) / (df_global + alpha + 1))
        specificity = max(specificity, spec_floor)  # clamp
        
        # Final score
        final_score = support + lambda_weight * specificity
        
        scores.append(CandidateScore(
            candidate_id=candidate.key,
            name=candidate.display_name,
            support=support,
            df_focus=df_focus,
            df_global=df_global,
            specificity=specificity,
            final_score=final_score,
        ))
    
    # Sort by final score descending
    scores.sort(key=lambda x: -x.final_score)
    return scores
```

### 2.4 Acceptance Criteria (Phase 2)

- [ ] Known hubs (Moscow, NYU, KGB) stop appearing in top 5 for roster queries
- [ ] Specificity score is logged and visible in trace
- [ ] Candidates with high support but low specificity are downranked

---

## Phase 3: Expansion Loop from FocusBundle

**Goal:** Handle "other names" queries (proximity fuse → VT fuse).

### 3.1 Term Extraction from FocusBundle

```python
def extract_expansion_terms(focus_bundle: FocusBundle) -> List[str]:
    """
    Extract high-specificity terms from FocusBundle for expansion.
    
    Targets:
    - ALL CAPS terms (codenames)
    - Hyphenated technical terms
    - Quoted phrases
    - Simple noun phrases
    """
    terms = set()
    
    for span in focus_bundle.spans:
        # ALL CAPS (codenames)
        caps = re.findall(r'\b[A-Z]{3,}\b', span.text)
        terms.update(caps)
        
        # Hyphenated terms
        hyphenated = re.findall(r'\b\w+-\w+(?:-\w+)*\b', span.text)
        terms.update(hyphenated)
        
        # Quoted phrases
        quoted = re.findall(r'"([^"]{3,50})"', span.text)
        terms.update(quoted)
    
    return list(terms)
```

### 3.2 Multi-Query Expansion

```python
def expansion_loop(
    query_text: str,
    initial_focus_bundle: FocusBundle,
    conn,
    max_rounds: int = 2,
) -> FocusBundle:
    """
    1. Extract high-specificity terms from FocusBundle
    2. Run second retrieval pass with query + terms
    3. Rebuild FocusBundle
    4. Stop when FocusBundle spans stabilize
    """
    current_bundle = initial_focus_bundle
    
    for round_num in range(max_rounds):
        # Extract expansion terms
        terms = extract_expansion_terms(current_bundle)
        if not terms:
            break
        
        # Run multi-query retrieval
        expanded_chunks = run_expanded_retrieval(
            query_text, 
            terms[:5],  # top 5 terms
            conn,
        )
        
        # Rebuild FocusBundle
        builder = FocusBundleBuilder()
        new_bundle = builder.build(query_text, expanded_chunks, terms)
        
        # Check stability (span overlap)
        old_span_ids = {s.span_id for s in current_bundle.spans}
        new_span_ids = {s.span_id for s in new_bundle.spans}
        
        jaccard = len(old_span_ids & new_span_ids) / len(old_span_ids | new_span_ids)
        if jaccard >= 0.85:
            break  # Stable
        
        current_bundle = new_bundle
    
    return current_bundle
```

### 3.3 Acceptance Criteria (Phase 3)

- [ ] "proximity fuse" query finds "VT fuse" spans via expansion
- [ ] Expansion terms are logged in trace
- [ ] FocusBundle stability stopping works (Jaccard >= 0.85)

---

## Phase 4: Agent Chooses Primitives (Optional)

**Goal:** Let agent decide which primitives to run, but can't break invariants.

### 4.1 Primitive Registry

```python
# retrieval/primitive_registry.py

@dataclass
class PrimitiveMetadata:
    name: str
    requires: List[str]      # e.g., ["focus_bundle"]
    emits: List[str]         # e.g., ["candidates", "joins"]
    cost_class: str          # "cheap", "medium", "expensive"
    description: str

PRIMITIVE_REGISTRY = {
    "entity_extraction": PrimitiveMetadata(
        name="entity_extraction",
        requires=["focus_bundle"],
        emits=["entity_candidates"],
        cost_class="cheap",
        description="Extract entity mentions from FocusBundle spans",
    ),
    "date_extraction": PrimitiveMetadata(
        name="date_extraction",
        requires=["focus_bundle"],
        emits=["date_candidates"],
        cost_class="cheap",
        description="Extract date mentions from FocusBundle spans",
    ),
    "place_extraction": PrimitiveMetadata(
        name="place_extraction",
        requires=["focus_bundle"],
        emits=["place_candidates"],
        cost_class="cheap",
        description="Extract place mentions from FocusBundle spans",
    ),
    "term_expansion": PrimitiveMetadata(
        name="term_expansion",
        requires=["focus_bundle"],
        emits=["expansion_terms"],
        cost_class="medium",
        description="Extract high-specificity terms for query expansion",
    ),
    "hubness_scoring": PrimitiveMetadata(
        name="hubness_scoring",
        requires=["candidates", "entity_df"],
        emits=["scored_candidates"],
        cost_class="cheap",
        description="Apply hubness penalty to candidate scores",
    ),
}
```

### 4.2 Agent Plan-of-Record Schema

```python
# retrieval/agent_plan_schema.py

@dataclass
class AgentPlanV2:
    """
    Agent's plan-of-record (structured, deterministic).
    
    The agent outputs this plan, then it's executed deterministically.
    """
    query_text: str
    
    # Retrieval configuration
    lanes: List[str]          # ["hybrid", "lexical_must_hit", "entity_mentions"]
    lane_budgets: Dict[str, int]  # {"hybrid": 200, ...}
    
    # FocusBundle configuration
    focus_top_n: int = 80
    focus_diversity: float = 0.3
    
    # Primitives to execute (in order)
    primitives: List[str]     # ["entity_extraction", "hubness_scoring"]
    primitive_params: Dict[str, Any] = field(default_factory=dict)
    
    # Output configuration
    max_items: int = 25
    max_citations_per_item: int = 2
    
    # Metadata
    model_version: str = "gpt-4.1-mini-2025-04-14"
    temperature: float = 0.0  # deterministic
    created_at: str = ""
```

### 4.3 Verifier-Driven Retry

```python
def execute_with_retry(
    plan: AgentPlanV2,
    conn,
    max_retries: int = 1,
) -> Tuple[RenderedAnswer, FocusBundle]:
    """
    Execute plan with verifier-driven retry.
    
    1. Execute plan
    2. Verify invariants
    3. If fail: give agent errors, get revised plan, retry once
    """
    for attempt in range(max_retries + 1):
        # Execute
        result = execute_plan_v2(plan, conn)
        
        # Verify
        verifier = FocusBundleVerifier()
        verification = verifier.verify_answer(
            result.rendered_answer,
            result.focus_bundle,
        )
        
        if verification.passed:
            return result.rendered_answer, result.focus_bundle
        
        if attempt < max_retries:
            # Get revised plan from agent
            plan = get_revised_plan(plan, verification.errors)
    
    # Return with warnings
    return result.rendered_answer, result.focus_bundle
```

---

## Summary: File Changes

### New Files (8)

| File | Purpose |
|------|---------|
| `retrieval/spans.py` | SpanMiner - convert chunks to cite-able windows |
| `retrieval/focus_bundle.py` | FocusBundleBuilder + MMR selection |
| `retrieval/hubness.py` | entity_df refresh + specificity scoring |
| `retrieval/candidate_ranker.py` | support + hubness scoring |
| `retrieval/primitive_registry.py` | primitive metadata |
| `retrieval/agent_plan_schema.py` | plan-of-record contract |
| `retrieval/verifier_v2.py` | FocusBundle invariant enforcement |
| `migrations/0049_focus_spans.sql` | focus_spans table |
| `migrations/0050_entity_df.sql` | entity_df table |

### Modified Files (4)

| File | Changes |
|------|---------|
| `retrieval/lanes.py` | Retrieval returns chunks + provenance only |
| `backend/.../synthesis.py` | Render only from FocusSpan evidence, cap citations |
| `scripts/friday_cli.py` | New `execute_agentic_query_v2` |
| Tests | Add span mining, focus bundle, hubness tests |

---

## Default Knobs

```python
# Focus Bundle
FOCUS_TOP_N_SPANS = 80       # maybe 120 for hard queries
MIN_SPAN_CHARS = 120
MAX_SPAN_CHARS = 600
MAX_SPANS_PER_CHUNK = 12
MIN_SPAN_SCORE = 0.3

# Hubness
LAMBDA_HUBNESS = 0.3
SPEC_FLOOR = -2.0

# Output
MAX_ITEMS = 25
MAX_CITATIONS_PER_ITEM = 2

# Retrieval
RETRIEVAL_TOP_K_PER_LANE = 300
```

---

## Test Plan

### Unit Tests

- [ ] Span mining is deterministic and stable offsets
- [ ] FocusBundle selection deterministic with pinned embeddings
- [ ] Hubness penalty monotonic (hubs downrank)
- [ ] Evidence gating rejects citations not in FocusBundle

### Regression Tests (Golden Queries)

- [ ] Roster: "members of Silvermaster network"
- [ ] Existence: "proximity fuse in Vassiliev"
- [ ] Relationship: "who handled Rosenberg"
- [ ] Technical alias: "VT fuse"

For each:
- Max citations bounded
- No hub entities in top 5
- Negative answers include lane summary

### Trace/Audit

Persist:
- Plan-of-record
- focus_spans
- Candidate scores with breakdown (support, spec)
