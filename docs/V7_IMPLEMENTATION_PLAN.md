# V7 Agentic Retrieval Implementation Plan

## Overview

V7 fixes the "amnesia" problem where the agent repeats identical tool calls, adds evidence bundling for split evidence, introduces pagination/novelty controls, and ensures every claim is citation-backed.

**Key improvements over V6:**
- Round-aware state (RoundSummary) prevents repetition
- Pagination + exclude_seen enables novelty without changing queries
- Evidence Bundles handle split evidence (roster tables, continuation lines)
- Expanded Summary forces every claim to have ≥1 citation
- Hard stop gate: can't answer without citation-backed claims

---

## File Structure

```
retrieval/agent/
├── v7_types.py              # Core data structures (RoundSummary, Bundle, ClaimWithCitation)
├── v7_query_parser.py       # CONTROL vs CONTENT parsing (enhanced from V6)
├── v7_pagination.py         # Cursor/exclude logic for retrieval tools
├── v7_evidence_store.py     # Bounded evidence store with tournament selection
├── v7_bundle_builder.py     # LLM-generated bundles from related spans
├── v7_round_summary.py      # Per-round LLM summary for decision state
├── v7_claim_enumerator.py   # Extract claims + require citations
├── v7_expanded_summary.py   # Final renderer with claims & citations
├── v7_stop_gate.py          # Citation-backed stop validation
├── v7_controller.py         # Main orchestration loop
├── v7_runner.py             # Entry point + CLI
└── __init__.py              # Exports
```

---

## Phase 1: Core Types & Query Parsing

### 1.1 v7_types.py - Data Structures

```python
@dataclass
class RoundSummary:
    """LLM-generated, schema-enforced per-round state."""
    round_num: int
    outcome: Literal["progress", "stalled", "regressed"]
    new_facts: List[str]
    open_questions: List[str]
    best_evidence_refs: List[str]  # span_ids or bundle_ids
    next_actions: List[Dict[str, Any]]  # [{tool, params}]
    avoid_repeats: List[str]  # tool+param fingerprints

@dataclass
class EvidenceSpan:
    """Single quotable span from a chunk."""
    span_id: str
    chunk_id: int
    doc_id: Optional[int]
    page: str
    source_label: str
    text: str
    relevance_score: float
    claim_supported: str

@dataclass
class EvidenceBundle:
    """Group of related spans (for split evidence)."""
    bundle_id: str
    span_ids: List[str]  # max 6 spans
    bundle_claim: str  # 1-2 sentence description
    confidence: float
    spans: List[EvidenceSpan]  # Denormalized for convenience

@dataclass
class ClaimWithCitation:
    """Atomic claim with required citation(s)."""
    claim_text: str
    citations: List[str]  # span_ids or bundle_ids (≥1 required)
    support_level: Literal["strong", "weak", "inferred"]

@dataclass
class ExpandedSummary:
    """Final output with enumerated claims."""
    short_answer: str
    claims: List[ClaimWithCitation]  # Every claim has ≥1 citation
    unsupported_claims: List[str]  # Claims that couldn't be cited (for debugging)
    evidence_used: List[str]  # All span/bundle IDs referenced
```

### 1.2 v7_query_parser.py - Enhanced CONTROL vs CONTENT

Reuse V6's QueryParser but ensure stricter separation:

```python
@dataclass
class ParsedQueryV7:
    original_query: str
    task_type: TaskType
    
    # CONTENT - eligible for entity linking + retrieval
    content_terms: List[str]
    
    # CONTROL - never entity-linked, never searched as content
    control_terms: List[str]
    
    # Extracted constraints
    scope_constraints: Dict[str, Any]  # {collections: [...], date_range: ...}
    output_requirements: List[str]  # "must cite X", "list format"
    
    # For enforcement
    content_tokens: Set[str]
    control_tokens: Set[str]
```

**Acceptance Test:**
- "Provide citations from Vassiliev notebooks" → control_terms=["Provide", "citations", "Vassiliev notebooks"], content_terms=[]

---

## Phase 2: Pagination & Novelty Controls

### 2.1 v7_pagination.py - Cursor/Exclude Logic

```python
@dataclass
class PaginationState:
    """Tracks seen results for novelty."""
    seen_chunk_ids: Set[int] = field(default_factory=set)
    seen_entity_ids: Set[int] = field(default_factory=set)
    cursors: Dict[str, int] = field(default_factory=dict)  # tool_key -> offset

class PaginationManager:
    def __init__(self):
        self.state = PaginationState()
    
    def get_exclude_chunks(self) -> List[int]:
        """Get chunk IDs to exclude from next search."""
        return list(self.state.seen_chunk_ids)
    
    def get_cursor(self, tool_key: str) -> int:
        """Get cursor offset for a tool call."""
        return self.state.cursors.get(tool_key, 0)
    
    def advance_cursor(self, tool_key: str, page_size: int):
        """Advance cursor after successful retrieval."""
        current = self.state.cursors.get(tool_key, 0)
        self.state.cursors[tool_key] = current + page_size
    
    def mark_seen(self, chunk_ids: List[int], entity_ids: List[int] = None):
        """Mark results as seen."""
        self.state.seen_chunk_ids.update(chunk_ids)
        if entity_ids:
            self.state.seen_entity_ids.update(entity_ids)
    
    def is_duplicate_call(self, tool_name: str, params: Dict) -> bool:
        """Check if this exact call was already made without pagination."""
        # ... fingerprint check
```

### 2.2 Tool Modifications

Update `tools.py` to accept pagination params:

```python
def hybrid_search_tool(
    conn,
    query: str,
    top_k: int = 200,
    collections: List[str] = None,
    cursor: int = 0,  # NEW: offset for pagination
    exclude_chunk_ids: List[int] = None,  # NEW: exclude seen
) -> ToolResult:
    ...

def entity_mentions_tool(
    conn,
    entity_id: int = None,
    name: str = None,
    top_k: int = 100,
    cursor: int = 0,  # NEW
    exclude_chunk_ids: List[int] = None,  # NEW
) -> ToolResult:
    ...
```

---

## Phase 3: Evidence Store (Hard Gate)

### 3.1 v7_evidence_store.py - Bounded Store

```python
class EvidenceStore:
    """
    HARD GATE: Synthesis ONLY sees what's in this store.
    Raw chunks never flow directly to claim extraction.
    """
    
    def __init__(self, max_spans: int = 40, max_bundles: int = 20):
        self.max_spans = max_spans
        self.max_bundles = max_bundles
        self.spans: Dict[str, EvidenceSpan] = {}
        self.bundles: Dict[str, EvidenceBundle] = {}
    
    def add_span(self, span: EvidenceSpan) -> bool:
        """Add span, evict worst if at capacity (tournament)."""
        ...
    
    def add_bundle(self, bundle: EvidenceBundle) -> bool:
        """Add bundle, evict worst if at capacity."""
        ...
    
    def tournament_compare(self, new: EvidenceSpan, existing: EvidenceSpan) -> bool:
        """LLM pairwise comparison: is new better than existing?"""
        ...
    
    def get_synthesis_context(self) -> str:
        """Get formatted context for synthesis (ONLY this)."""
        ...
    
    def validate_citations(self, citation_ids: List[str]) -> bool:
        """Check all citations reference active evidence."""
        ...
```

---

## Phase 4: RoundSummary for Decision State

### 4.1 v7_round_summary.py - LLM-Generated State

```python
ROUND_SUMMARY_SCHEMA = {
    "type": "object",
    "required": ["round_outcome", "new_facts", "open_questions", "best_evidence_refs", "next_actions", "avoid_repeats"],
    "properties": {
        "round_outcome": {"enum": ["progress", "stalled", "regressed"]},
        "new_facts": {"type": "array", "items": {"type": "string"}},
        "open_questions": {"type": "array", "items": {"type": "string"}},
        "best_evidence_refs": {"type": "array", "items": {"type": "string"}},
        "next_actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string"},
                    "params": {"type": "object"},
                    "rationale": {"type": "string"}
                }
            },
            "maxItems": 3
        },
        "avoid_repeats": {"type": "array", "items": {"type": "string"}}
    }
}

class RoundSummarizer:
    """Generate schema-enforced round summary."""
    
    def summarize(
        self,
        round_num: int,
        question: str,
        actions_taken: List[Dict],
        evidence_store: EvidenceStore,
        previous_summary: Optional[RoundSummary],
    ) -> RoundSummary:
        """
        LLM generates summary with:
        - What was learned (new_facts)
        - What's still unknown (open_questions)
        - Best evidence found (best_evidence_refs)
        - Recommended next actions (next_actions)
        - What NOT to repeat (avoid_repeats)
        """
        ...
```

**Key Integration:**
- Round N+1 receives RoundSummary from Round N
- Searcher prompt includes `avoid_repeats` to prevent identical calls
- `next_actions` guide tool selection (but don't force it)

---

## Phase 5: Neighbor/Continuation Tool

### 5.1 Add chunk_neighbors Tool

```python
def chunk_neighbors_tool(
    conn,
    chunk_id: int,
    window: int = 2,  # chunks before and after
    exclude_chunk_ids: List[int] = None,
) -> ToolResult:
    """
    Get neighboring chunks from the same document.
    
    Use case: Found a roster table starting in chunk X, 
    need to get chunks X-1, X+1, X+2 to capture full table.
    """
    ...
```

**Alternative:** `page_window_tool` if page-based retrieval is better for the schema.

---

## Phase 6: Evidence Bundles

### 6.1 v7_bundle_builder.py - Group Related Spans

```python
class BundleBuilder:
    """
    LLM-powered grouping of related spans into citeable bundles.
    
    Use case: Roster table split across 3 chunks becomes 1 bundle.
    """
    
    def build_bundles(
        self,
        spans: List[EvidenceSpan],
        question: str,
        max_spans_per_bundle: int = 6,
    ) -> List[EvidenceBundle]:
        """
        Prompt LLM to identify spans that should be cited together:
        - Continuation lines ("...continued from previous page")
        - Table rows split across chunks
        - Pronoun referents that need context
        
        Returns bundles where each bundle.spans should be cited as a unit.
        """
        ...
```

**Bundle Structure:**
```python
EvidenceBundle(
    bundle_id="bnd_001",
    span_ids=["sp_012", "sp_013", "sp_014"],
    bundle_claim="Lists members X, Y, Z of Silvermaster network (table spans pp.45-46)",
    confidence=0.9,
    spans=[...]
)
```

---

## Phase 7: Stop Gate & Claim-Citation Enforcement

### 7.1 v7_stop_gate.py - Citation-Backed Stop

```python
class StopGate:
    """
    RULE: Cannot stop with answer unless every claim has citations.
    """
    
    def validate_stop(
        self,
        proposed_answer: str,
        claims: List[ClaimWithCitation],
        evidence_store: EvidenceStore,
    ) -> Tuple[bool, str]:
        """
        Returns (can_stop, reason).
        
        Fails if:
        - Any claim has empty citations
        - Any citation references non-existent evidence
        - For roster queries: members without citations
        """
        ...
    
    def suggest_repair(
        self,
        failed_claims: List[ClaimWithCitation],
        evidence_store: EvidenceStore,
    ) -> str:
        """Suggest how to fix unsupported claims."""
        ...
```

### 7.2 v7_claim_enumerator.py - Extract Claims + Require Citations

```python
class ClaimEnumerator:
    """
    Extract atomic claims from answer, require each to have ≥1 citation.
    """
    
    def enumerate_claims(
        self,
        answer: str,
        evidence_store: EvidenceStore,
    ) -> List[ClaimWithCitation]:
        """
        LLM extracts claims and assigns citations.
        
        If a claim cannot be supported:
        - Mark for removal
        - Or rewrite into supported version
        """
        ...
    
    def validate_all_cited(
        self,
        claims: List[ClaimWithCitation],
    ) -> Tuple[List[ClaimWithCitation], List[str]]:
        """
        Returns (valid_claims, unsupported_claims).
        
        Unsupported claims are removed from final output.
        """
        ...
```

---

## Phase 8: Expanded Summary Renderer

### 8.1 v7_expanded_summary.py - Final Output

```python
class ExpandedSummaryRenderer:
    """
    Produces researcher-grade output with every claim cited.
    """
    
    def render(
        self,
        short_answer: str,
        claims: List[ClaimWithCitation],
        evidence_store: EvidenceStore,
        unsupported_claims: List[str],
    ) -> ExpandedSummary:
        """
        Output structure:
        
        ## Answer
        <short_answer>
        
        ## Claims & Citations
        1. <claim_text> [1][2]
        2. <claim_text> [3]
        ...
        
        ## Evidence
        [1] <span_text> (Vassiliev, p.45)
        [2] <span_text> (Vassiliev, p.46)
        ...
        
        ## Notes
        - X claims could not be supported and were excluded
        """
        ...
```

---

## Phase 9: Main Controller

### 9.1 v7_controller.py - Orchestration

```python
class V7Controller:
    """
    Main loop with round-aware state and hard gates.
    """
    
    def __init__(self, config: V7Config):
        self.query_parser = QueryParserV7()
        self.pagination = PaginationManager()
        self.evidence_store = EvidenceStore()
        self.round_summarizer = RoundSummarizer()
        self.bundle_builder = BundleBuilder()
        self.claim_enumerator = ClaimEnumerator()
        self.stop_gate = StopGate()
        self.renderer = ExpandedSummaryRenderer()
    
    def run(self, question: str, conn) -> V7Result:
        # 1. Parse query (CONTROL vs CONTENT)
        parsed = self.query_parser.parse(question)
        
        # 2. Entity linking (CONTENT terms only)
        linking = self.entity_linker.link(parsed.content_terms, conn)
        
        # 3. Multi-round retrieval with RoundSummary
        round_summary = None
        for round_num in range(1, self.config.max_rounds + 1):
            # a. Searcher decides actions (informed by round_summary)
            actions = self.decide_actions(parsed, linking, round_summary)
            
            # b. Execute with pagination/exclude
            results = self.execute_with_novelty(actions, conn)
            
            # c. Grade and add to evidence store (tournament)
            self.update_evidence_store(results)
            
            # d. Build bundles from related spans
            bundles = self.bundle_builder.build_bundles(
                self.evidence_store.get_new_spans(),
                question
            )
            for b in bundles:
                self.evidence_store.add_bundle(b)
            
            # e. Generate RoundSummary for next round
            round_summary = self.round_summarizer.summarize(
                round_num, question, actions, 
                self.evidence_store, round_summary
            )
            
            # f. Check stop condition
            if round_summary.outcome == "stalled" and round_num >= 2:
                break
        
        # 4. Synthesize from evidence store ONLY (hard gate)
        answer = self.synthesize(parsed, self.evidence_store)
        
        # 5. Enumerate claims + require citations
        claims, unsupported = self.claim_enumerator.enumerate_claims(
            answer, self.evidence_store
        )
        
        # 6. Validate stop gate
        can_stop, reason = self.stop_gate.validate_stop(
            answer, claims, self.evidence_store
        )
        
        if not can_stop:
            # Repair loop or return "insufficient evidence"
            ...
        
        # 7. Render expanded summary
        return self.renderer.render(
            short_answer=answer,
            claims=claims,
            evidence_store=self.evidence_store,
            unsupported_claims=unsupported,
        )
```

---

## Phase 10: CLI & Backend Integration

### 10.1 Update friday_cli.py

```python
# Add /v7 command
elif cmd == "/v7":
    query_text = arg.strip()
    print(f"\n[V7 Mode] Processing: \"{query_text}\"...")
    from retrieval.agent.v7_runner import run_v7_query
    result = run_v7_query(conn, query_text, verbose=True)
    ...

# Make V7 the default (replace V6)
# DEFAULT: V7 Agentic Mode
query_text = user_input
print(f"\n[V7 Mode] Processing: \"{query_text}\"...")
from retrieval.agent.v7_runner import run_v7_query
result = run_v7_query(conn, query_text, verbose=True)
```

### 10.2 Update backend/app/routes/chat.py

```python
# Change imports
from retrieval.agent.v7_runner import run_v7_query, V7Result

# Update run_v7_chat_query to use V7
def run_v7_chat_query(conn, session_id: int, question: str) -> Dict[str, Any]:
    from retrieval.agent.v7_runner import run_v7_query
    result = run_v7_query(conn, question, verbose=True)
    
    # Map V7Result to chat response format
    return {
        "answer": result.short_answer,
        "claims": [c.to_dict() for c in result.claims],
        "expanded_summary": result.to_expanded_format(),
        ...
    }
```

---

## Implementation Order

| Phase | Component | Est. Effort | Dependencies |
|-------|-----------|-------------|--------------|
| 1 | v7_types.py | 1 day | None |
| 2 | v7_query_parser.py (enhance V6) | 0.5 day | Phase 1 |
| 3 | v7_pagination.py + tool mods | 1 day | Phase 1 |
| 4 | v7_evidence_store.py | 1 day | Phase 1 |
| 5 | v7_round_summary.py | 1 day | Phases 1, 4 |
| 6 | v7_bundle_builder.py | 1 day | Phases 1, 4 |
| 7 | chunk_neighbors tool | 0.5 day | Phase 3 |
| 8 | v7_claim_enumerator.py | 1 day | Phases 1, 4 |
| 9 | v7_stop_gate.py | 0.5 day | Phase 8 |
| 10 | v7_expanded_summary.py | 0.5 day | Phases 8, 9 |
| 11 | v7_controller.py | 2 days | All above |
| 12 | v7_runner.py + CLI/Backend | 0.5 day | Phase 11 |
| 13 | Testing & refinement | 2 days | All |

**Total: ~12-14 days**

---

## Acceptance Tests

### Test 1: No Dumb Entity Linking
```
Query: "Provide citations from Vassiliev notebooks"
Expected: content_terms=[], control_terms=["Provide", "citations", "Vassiliev notebooks"]
         No entity IDs generated for "Provide"
```

### Test 2: Pagination Novelty
```
Query: "Silvermaster network members" (2 rounds)
Round 1: hybrid_search(...) → chunks [1-20]
Round 2: hybrid_search(..., cursor=20, exclude=[1-20]) → chunks [21-40]
Expected: Different chunks in each round
```

### Test 3: No Repetition (RoundSummary)
```
Query: "Who was in the Silvermaster network?"
Round 1: entity_mentions(72144) → 15 chunks
Round 2: Should NOT repeat entity_mentions(72144)
         Should use chunk_neighbors or paginated search
```

### Test 4: Split Evidence Bundled
```
Scenario: Roster table split across chunks 100, 101, 102
Expected: Bundle created with span_ids=[sp_100, sp_101, sp_102]
          Answer cites bundle_id, UI expands to show all spans
```

### Test 5: Every Claim Cited
```
Query: "Silvermaster network members from Vassiliev"
Expected: 
  Claims:
  1. "Harry Dexter White was a member" [1][2]
  2. "Solomon Adler was a member" [3]
  ...
  NO claims without citations
```

### Test 6: Insufficient Evidence Handled
```
Query: "Who recruited Silvermaster?" (assume not in corpus)
Expected: "Insufficient evidence in Vassiliev-only scope"
          Best available evidence cited
          No speculative claims
```

---

## Notes

- V7 is **additive** - can reuse V6 components where applicable
- RoundSummary is the key innovation for preventing amnesia
- Bundles are optional but critical for roster/table queries
- Expanded Summary is the researcher-grade output format
- All LLM calls should have timeouts and fallbacks
