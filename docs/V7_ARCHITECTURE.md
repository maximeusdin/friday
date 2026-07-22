# V7 Architecture: Citation-Enforced Research Agent

## Executive Summary

V7 is a **citation-enforced research retrieval system** designed for archival document analysis. It extends the V6 pipeline with strict citation requirements: **every claim in the final answer must have at least one citation from the evidence**. V7 Phase 2 adds pagination, novelty controls, neighbor retrieval, structured round summaries, and evidence bundling.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [V6 Foundation](#2-v6-foundation)
3. [V7 Phase 1: Citation Enforcement](#3-v7-phase-1-citation-enforcement)
4. [V7 Phase 2: Advanced Features](#4-v7-phase-2-advanced-features)
5. [Data Structures](#5-data-structures)
6. [Tool Registry](#6-tool-registry)
7. [Configuration](#7-configuration)
8. [Pipeline Flow](#8-pipeline-flow)
9. [Design Decisions](#9-design-decisions)

---

## 1. System Overview

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          V7 Controller                                   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     V6 Pipeline (Foundation)                      │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐   │   │
│  │  │   Query     │→ │   Entity     │→ │    Agentic Retrieval   │   │   │
│  │  │   Parser    │  │   Linker     │  │  (Tools + LLM Searcher)│   │   │
│  │  └─────────────┘  └──────────────┘  └────────────────────────┘   │   │
│  │          ↓                                      ↓                 │   │
│  │  CONTROL/CONTENT              ┌────────────────────────────┐     │   │
│  │  Classification               │   Evidence Bottleneck      │     │   │
│  │                               │   (40 spans max)           │     │   │
│  │                               └────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                          ↓                               │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                   V7 Phase 1: Citation Enforcement                │   │
│  │  ┌─────────────────┐  ┌─────────────┐  ┌───────────────────┐     │   │
│  │  │ Claim Enumerator│→ │  Stop Gate  │→ │ Expanded Summary  │     │   │
│  │  │ (extract claims)│  │ (validate)  │  │   (render output) │     │   │
│  │  └─────────────────┘  └─────────────┘  └───────────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                          ↓                               │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                   V7 Phase 2: Advanced Features                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐    │   │
│  │  │ Pagination + │  │ Round        │  │ Evidence Bundles     │    │   │
│  │  │ Exclude-Seen │  │ Summaries    │  │ (group related spans)│    │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘    │   │
│  │  ┌──────────────────────────────────────────────────────────┐    │   │
│  │  │           Neighbor/Continuation Tools                     │    │   │
│  │  │  chunk_neighbors() | document_chunks()                    │    │   │
│  │  └──────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `retrieval/agent/v7_controller.py` | Main V7 controller, orchestrates the pipeline |
| `retrieval/agent/v7_types.py` | Core data structures (claims, bundles, round summaries) |
| `retrieval/agent/v7_claim_enumerator.py` | Extracts atomic claims and assigns citations |
| `retrieval/agent/v7_stop_gate.py` | Validates all claims have valid citations |
| `retrieval/agent/v7_expanded_summary.py` | Renders researcher-grade output |
| `retrieval/agent/v7_bundle_builder.py` | Post-bottleneck evidence grouping |
| `retrieval/agent/v7_bundle_types.py` | Pre-bundling data structures (ChunkAnnotation, BundleCandidate) |
| `retrieval/agent/v7_pre_bundler.py` | Pre-bottleneck concordance-aware bundling |
| `retrieval/agent/v6_controller.py` | V6 foundation: parsing, retrieval, bottleneck |
| `retrieval/agent/v6_query_parser.py` | CONTROL vs CONTENT token classification |
| `retrieval/agent/v6_evidence_bottleneck.py` | Forces convergence to 40 max spans (+ bundle filtering) |
| `retrieval/agent/tools.py` | Tool registry for agentic retrieval |
| `retrieval/ops.py` | Low-level search operations (SQL, vector) |

---

## 2. V6 Foundation

V7 builds on the V6 pipeline, which implements a **principled, no-heuristics architecture** for archival research.

### 2.1 Query Parser

**File:** `retrieval/agent/v6_query_parser.py`

**Purpose:** Separate CONTROL tokens from CONTENT tokens to prevent bad entity linking.

**Problem Solved:** The query "Provide citations from Vassiliev notebooks about Silvermaster" should NOT entity-link "Vassiliev" (it's a collection name, not a person).

**Token Classification:**

- **CONTROL tokens:** Instruction words, collection names, date constraints
  - Examples: "provide", "cite", "from", "Vassiliev", "between 1940 and 1950"
  - These are NOT entity-linked

- **CONTENT tokens:** Actual subjects to search for
  - Examples: "Silvermaster", "Harry White", "Treasury network"
  - These ARE entity-linked

**Task Types:**

```python
class TaskType(Enum):
    ROSTER_ENUMERATION = "roster_enumeration"  # "Who were members of X?"
    TIMELINE = "timeline"                       # "When did X happen?"
    EVIDENCE_SEARCH = "evidence_search"         # "Find evidence of X"
    RELATIONSHIP = "relationship"               # "How was X connected to Y?"
    FACTUAL = "factual"                         # "What was X's role?"
```

**Output:**

```python
@dataclass
class ParsedQuery:
    task_type: TaskType
    scope_constraints: Dict[str, Any]    # {"collections": ["vassiliev"]}
    topic_terms: List[str]               # ["Silvermaster network"]
    control_tokens: Set[str]             # {"provide", "cite", "vassiliev"}
    content_tokens: Set[str]             # {"silvermaster", "network"}
```

### 2.2 Entity Linker

**File:** `retrieval/agent/v6_entity_linker.py`

**Purpose:** Link CONTENT tokens to database entities, with `use_for_retrieval` flag.

**Key Behavior:**
- Only links tokens from `content_tokens`, never `control_tokens`
- Each linked entity has `use_for_retrieval` flag
- Concordance expansion (codenames → real names)

### 2.3 Agentic Retrieval

**Location:** `V6Controller._retrieve()` in `v6_controller.py`

**Purpose:** LLM-driven tool selection for evidence gathering.

**How It Works:**
1. Build searcher context (parsed query, linked entities, scope)
2. LLM decides which tool to call (hybrid_search, entity_mentions, etc.)
3. Execute tool, collect chunks
4. Repeat until budget exhausted or LLM says STOP
5. Deduplicate results

**Scope Enforcement:**
- `scope_constraints.collections` is INJECTED into every search tool call
- Entity-based searches are post-filtered by collection

### 2.4 Evidence Bottleneck

**File:** `retrieval/agent/v6_evidence_bottleneck.py`

**Purpose:** Force convergence before synthesis by limiting to 40 spans max.

**Why This Is Critical:**
- Without bottleneck: 164 chunks → thousands of claims → unmanageable
- With bottleneck: 40 curated spans → focused synthesis

**Process:**
1. Grade each span for relevance
2. Extract claim each span supports
3. Identify members (for roster queries)
4. Keep top 40 spans by relevance

**Grading Modes:**

| Mode | Description | Best For |
|------|-------------|----------|
| `tournament` (default) | Spans compete via pairwise ranking | When all spans are "somewhat relevant"; more robust to LLM score drift |
| `absolute` | Each span graded independently 0-10 | Clear-cut relevance distinctions; faster for small batches |

**Tournament Mode Process (Elo-Style Ratings):**
1. Initialize all spans with rating 1000
2. Run random 1v1 matchups (each span faces ~4 opponents)
3. Execute matchups via LLM (see **Tournament Batching** below)
4. Update Elo ratings after each matchup (winners gain, losers lose points)
5. Take top `max_spans` by final rating

**Tournament Batching (configurable):**

| Setting | Behavior | Trade-off |
|---------|----------|-----------|
| `tournament_batch=False` (default) | 1 API call per matchup | More accurate, slower (recommended for quality) |
| `tournament_batch=True` | 8 matchups per API call | ~8x faster, may be less focused per decision |

When batching is disabled, each matchup gets the LLM's full attention ("A vs B, which is better?").
When enabled, the LLM sees multiple matchups in one prompt and returns all decisions at once.

**Absolute Mode Process:**
1. Split chunks into batches of 15
2. LLM grades each chunk with pass/fail and 0-10 score
3. Keep only passing spans
4. Sort by score, take top `max_spans`

**Output:**

```python
@dataclass
class BottleneckSpan:
    span_id: str
    chunk_id: int
    span_text: str
    relevance_score: float      # 0-10 (or rank-derived in tournament mode)
    claim_supported: str        # What this span supports
    is_directly_responsive: bool
    identifies_member: bool     # For roster queries
    member_name: str
```

### 2.5 Progress Gate

**File:** `retrieval/agent/v6_progress_gate.py`

**Purpose:** Decide whether to continue, pivot, or stop retrieval.

**Decisions:**
- **CONTINUE:** Making progress, keep searching
- **PIVOT:** Not working, try different approach
- **STOP:** Enough evidence or exhausted options

---

## 3. V7 Phase 1: Citation Enforcement

V7's core innovation: **every claim must have citations**.

### 3.1 Claim Enumerator

**File:** `retrieval/agent/v7_claim_enumerator.py`

**Purpose:** Extract atomic claims from the answer and assign citations from evidence.

**Process:**
1. Take the synthesized answer and evidence spans
2. Use LLM to extract individual claims
3. For each claim, assign citations from available evidence
4. Flag claims that cannot be supported

**Output:**

```python
@dataclass
class ClaimWithCitation:
    claim_text: str                # Single atomic assertion
    citations: List[str]           # span_ids (MUST be non-empty)
    support_level: Literal["strong", "weak", "inferred"]
```

**Example:**

```json
{
  "claims": [
    {
      "claim": "Harry White was a member of the Silvermaster network",
      "citations": ["sp_1", "sp_3"],
      "support": "strong"
    }
  ],
  "unsupported": ["Some claim with no evidence"]
}
```

### 3.2 Stop Gate

**File:** `retrieval/agent/v7_stop_gate.py`

**Purpose:** HARD gate - answer cannot be returned unless all claims are cited.

**Validation Rules:**
1. Every claim MUST have non-empty `citations` list
2. Every citation MUST reference an existing span in evidence store
3. For roster queries: every member must be cited

**Result:**

```python
@dataclass
class StopGateResult:
    can_stop: bool              # Is it safe to return this answer?
    reason: str                 # Why (or why not)
    invalid_claims: List[ClaimWithCitation]
    invalid_citations: List[str]
```

**If Validation Fails:**
- `drop_uncited_claims=True` (default): Remove unsupported claims
- `drop_uncited_claims=False`: Return failure, don't hallucinate

### 3.3 Expanded Summary Renderer

**File:** `retrieval/agent/v7_expanded_summary.py`

**Purpose:** Produce researcher-grade output with enumerated claims and evidence.

**Output Format:**

```
============================================================
ANSWER
============================================================

<short answer text>

============================================================
CLAIMS & CITATIONS
============================================================

1. Harry White was a Treasury official [sp_1][sp_3]
2. He passed economic data to Silvermaster [sp_5]
3. The network operated from 1941-1945 [sp_7] (weak)

============================================================
EVIDENCE
============================================================

[sp_1] (Vassiliev, p.45)
  "White met regularly with Silvermaster at Treasury..."

[sp_3] (Vassiliev, p.46)
  "Confirmed membership in the economic intelligence group"

============================================================
UNSUPPORTED CLAIMS (excluded from answer)
============================================================

- The network had 50+ members (no citation found)

------------------------------------------------------------
Total claims: 4
Valid (cited): 3
Dropped (unsupported): 1
------------------------------------------------------------
```

---

## 4. V7 Phase 2: Advanced Features

Phase 2 addresses four remaining problems:

| Problem | Solution |
|---------|----------|
| Agent repeats identical tool calls | **Pagination + Exclude-Seen** |
| Cross-round amnesia | **RoundSummary** |
| Can't expand context around hits | **Neighbor Tools** |
| Split evidence across chunks | **Evidence Bundles** |

### 4.1 Pagination + Exclude-Seen (Novelty Controls)

**Files:** `retrieval/ops.py`, `retrieval/agent/tools.py`, `v6_controller.py`

**Problem:** Model repeats identical tool calls because it can't ask for "next page" or exclude already-seen results.

**Implementation:**

1. **SearchFilters extended:**

```python
@dataclass
class SearchFilters:
    # ... existing fields ...
    exclude_chunk_ids: Optional[List[int]] = None
    exclude_page_ids: Optional[List[int]] = None
    exclude_document_ids: Optional[List[int]] = None
```

2. **SQL exclusion clauses** in `_build_where()`:

```sql
AND c.id != ALL(%(exclude_chunk_ids)s)
AND cm.first_page_id != ALL(%(exclude_page_ids)s)
AND cm.document_id != ALL(%(exclude_document_ids)s)
```

3. **Tool parameters:**

```python
def hybrid_search_tool(
    conn,
    query: str,
    top_k: int = 200,
    exclude_chunk_ids: Optional[List[int]] = None,
    exclude_page_ids: Optional[List[int]] = None,
    exclude_document_ids: Optional[List[int]] = None,
) -> ToolResult:
```

4. **Controller tracking:**

```python
class V6Controller:
    def __init__(self):
        self.seen_chunk_ids: Set[int] = set()
        self.seen_page_ids: Set[int] = set()
        self.seen_document_ids: Set[int] = set()
```

5. **Auto-injection in `_call_tool()`:**

```python
if self.config.exclude_seen_mode != "off":
    if self.seen_chunk_ids:
        params["exclude_chunk_ids"] = list(self.seen_chunk_ids)
```

**Configuration:**

```python
@dataclass
class V6Config:
    exclude_seen_mode: str = "soft"  # off | soft | hard
    top_k_budget_multiplier: float = 1.5
```

- **off:** No auto-exclusion
- **soft:** Exclude only bottlenecked/stored chunks (preserves recall)
- **hard:** Exclude everything ever retrieved (aggressive novelty)

**Tool Result Metadata:**

```python
@dataclass
class ToolResult:
    # ... existing fields ...
    has_more: bool = False
    next_cursor: Optional[str] = None
    total_available: Optional[int] = None
```

### 4.2 RoundSummary (Structured Decision State)

**Files:** `retrieval/agent/v7_types.py`, `retrieval/agent/v7_controller.py`

**Problem:** Round 2 has "amnesia" about Round 1 strategies. Tool calls repeat across rounds.

**Data Structure:**

```python
@dataclass
class RoundSummary:
    round_number: int
    decision: RoundDecisionType  # CONTINUE, PIVOT, NARROW, STOP_*
    decision_rationale: str
    
    # What we learned
    key_findings: List[KeyFinding]
    
    # What to do next
    actionable_leads: List[ActionableLead]
    
    # Progress metrics
    evidence_count: int
    new_evidence_count: int
    coverage_estimate: float
    
    # What worked / didn't
    successful_strategies: List[str]
    failed_strategies: List[str]
    information_gaps: List[str]
```

**Supporting Types:**

```python
@dataclass
class ActionableLead:
    lead_type: str        # "entity", "document", "term", "codename"
    target: str           # Entity name, search term, etc.
    rationale: str        # Why this lead is promising
    priority: LeadPriority  # HIGH, MEDIUM, LOW, EXHAUSTED
    suggested_tool: Optional[str]

@dataclass
class KeyFinding:
    finding: str
    confidence: float
    evidence_ids: List[int]
```

**Generator:**

```python
class RoundSummaryGenerator:
    def generate(
        self,
        round_number: int,
        question: str,
        evidence_chunks: List[Dict],
        previous_summary: Optional[RoundSummary],
        tool_observations: List[Dict],
    ) -> RoundSummary:
        # LLM analyzes round results and produces structured summary
```

**Integration into searcher context:**

```
=== Round 1 Summary ===
Evidence: 45 total (45 new)
Decision: continue - More evidence available

Key Findings:
  - Silvermaster operated a Treasury network (confidence: 0.8)
  - Harry White was a key contact (confidence: 0.9)

Actionable Leads:
  [high] entity: Harry White
      → Multiple mentions suggest central role
  [medium] term: economic data
      → Mentioned in context of intelligence passing

Information Gaps:
  ? How many members total?
  ? What was the network's timespan?

Failed Strategies (avoid repeating):
  ✗ lexical_exact(term="PAL")
```

### 4.3 Neighbor/Continuation Tools

**Files:** `retrieval/ops.py`, `retrieval/agent/tools.py`

**Problem:** Agent finds a high-value span (roster, table) but can't pull adjacent chunks to capture the full context.

**Functions in `ops.py`:**

```python
def get_chunk_neighbors(
    conn,
    chunk_id: int,
    before: int = 2,
    after: int = 2,
    include_seed: bool = True,
) -> List[ChunkNeighbor]:
    """
    Get neighboring chunks from the same document.
    
    Ordering: document_id → page_id → chunk_id
    """

def get_document_chunks(
    conn,
    document_id: int,
    page_id: Optional[int] = None,
    limit: int = 50,
) -> List[ChunkNeighbor]:
    """Get all chunks from a document or specific page."""
```

**Data Structure:**

```python
@dataclass
class ChunkNeighbor:
    chunk_id: int
    text: str
    position: int      # Negative = before, positive = after, 0 = seed
    document_id: Optional[int]
    page_id: Optional[int]
    collection_slug: Optional[str]
```

**Tools:**

```python
def chunk_neighbors_tool(
    conn,
    chunk_id: int,
    before: int = 2,
    after: int = 2,
) -> ToolResult:
    """Get neighboring chunks from the same document."""

def document_chunks_tool(
    conn,
    document_id: int,
    page_id: Optional[int] = None,
    limit: int = 30,
) -> ToolResult:
    """Get all chunks from a document or specific page."""
```

**Caps (server-side enforced):**
- `before`, `after`: max 10 each
- `limit`: max 100

### 4.4 Evidence Bundles

**Files:** `retrieval/agent/v7_types.py`, `retrieval/agent/v7_bundle_builder.py`

**Problem:** Roster tables split across chunks can't be cited as one unit. Pronoun referents need context from previous chunks.

**Data Structures:**

```python
@dataclass
class BundleSpan:
    span_id: str
    chunk_id: int
    text: str
    relevance_score: float
    source_label: Optional[str]
    page: Optional[str]

@dataclass
class EvidenceBundle:
    bundle_id: str          # "b_1_0" - stable ID for citation
    topic: str              # What this bundle is about
    spans: List[BundleSpan] # Max ~6 spans
    
    status: BundleStatus    # FORMING, COMPLETE, CITED, SUPERSEDED
    confidence: float
    summary: str            # LLM-generated summary
    key_claims: List[str]   # Claims this bundle supports
    
    created_round: int
    last_updated_round: int

@dataclass
class BundleCollection:
    bundles: List[EvidenceBundle]
    
    def add_bundle(self, bundle: EvidenceBundle)
    def get_sufficient_bundles(self) -> List[EvidenceBundle]
```

**Bundle Lifecycle:**

1. **FORMING:** Created when related spans are identified
2. **COMPLETE:** Has sufficient evidence (2+ spans, confidence ≥ 0.5)
3. **CITED:** Used in synthesis
4. **SUPERSEDED:** Better evidence found

**Builder:**

```python
class BundleBuilder:
    def build_bundles(
        self,
        spans: List[Dict],
        question: str,
        round_number: int,
    ) -> BundleCollection:
        """
        LLM identifies spans that should be cited together:
        - Continuation lines ("...continued from previous page")
        - Table rows split across chunks
        - Pronoun referents needing context
        """
    
    def update_bundles(
        self,
        collection: BundleCollection,
        new_spans: List[Dict],
        round_number: int,
    ) -> BundleCollection:
        """Add new spans to existing bundles or create new ones."""
```

**Citation Rule:**
> Synthesis sees bundles as primary; spans are only used if not in a bundle.

### 4.5 Pre-Bundling (Concordance-Aware Grouping)

**Files:** `retrieval/agent/v7_pre_bundler.py`, `retrieval/agent/v7_bundle_types.py`

**Problem:** Raw chunks go directly to the bottleneck, where:
- Codenames like "Ruble, Raid, Mole" may be treated as member names
- Related evidence is scattered across individual chunks
- Entity resolution happens too late in the pipeline

**Solution:** Insert a **Pre-Bundling** phase between retrieval and the bottleneck that:
1. Extracts named entity surfaces from chunks (persons, codenames, orgs)
2. Resolves codenames to canonical entity IDs via concordance
3. Groups related chunks into `BundleCandidate` objects
4. Passes bundles (not raw chunks) to the bottleneck for scoring

**Architecture:**

```
Retrieval → Chunks (100-300)
    ↓
Pre-Bundler (NEW)
    ├── Candidate Selection (cap to 80)
    ├── Surface Extraction (LLM)
    ├── Concordance Resolution
    └── Semantic Bundling
    ↓
BundleCandidates (max 10)
    ↓
Evidence Bottleneck (modified)
    ├── Tournament/Absolute mode for bundles
    └── Output: selected bundle_ids + flattened chunk_ids
    ↓
Synthesis (bundles as primary evidence)
```

**Pre-Bundling Modes:**

| Mode | Description | Use Case |
|------|-------------|----------|
| `off` | No bundling, chunks go directly to bottleneck | Default, backward compatible |
| `passthrough` | Single bundle with all chunks, minimal processing | Fast mode, post-selection codename guard only |
| `micro` | Small bundles around top seed chunks | Medium speed/quality tradeoff |
| `semantic` | Full LLM-based bundling with concordance resolution | Comprehensive, best for roster queries |

**Configuration:**

```python
@dataclass
class V6Config:
    # Pre-Bundling settings
    pre_bundling_mode: str = "off"  # off | passthrough | micro | semantic
    pre_bundling_config: Optional[PreBundlingConfig] = None
    max_bottleneck_bundles: int = 5  # Max bundles after bundle-level bottleneck
```

**Data Structures:**

```python
@dataclass
class ChunkAnnotation:
    chunk_id: int
    person_surfaces: List[str]       # ["Harry White", "Silvermaster"]
    codename_surfaces: List[str]     # ["Pal", "Mole"]
    org_surfaces: List[str]          # ["Treasury", "Soviet intelligence"]
    resolved_people: List[int]       # entity_ids from concordance
    codename_links: List[CodenameLink]
    unresolved_codenames: List[str]
    is_roster_evidence: bool
    self_contained: bool

@dataclass
class BundleCandidate:
    bundle_id: str                   # "bc_0", "bc_1", etc.
    bundle_kind: BundleKind          # PERSON_EVIDENCE, CODENAME_EVIDENCE, MIXED
    topic: str                       # "Evidence about Harry White"
    chunk_ids: List[int]
    chunks: List[Dict]               # Raw chunk data
    primary_entities: List[int]      # Resolved entity_ids
    unresolved_codenames: List[str]
    member_yield_estimate: int       # For roster: expected members named
    confidence: float
    summary: str
    key_claims: List[str]
```

**Bundle Types:**

| Kind | Description | Roster Handling |
|------|-------------|-----------------|
| `PERSON_EVIDENCE` | Contains resolved person entity IDs | Preferred for roster answers |
| `CODENAME_EVIDENCE` | Contains only unresolved codenames | Lower priority, may not identify members |
| `MIXED` | Both resolved entities and unresolved codenames | Moderate priority |

**Integration in Controller:**

```python
# In V6Controller retrieval loop
if self.pre_bundler is not None:
    # Run pre-bundling before bottleneck
    bundling_result = self.pre_bundler.run(chunks, parsed, conn)
    
    if bundling_result.bundles:
        # Score bundles through bottleneck
        bundle_selection = self.bottleneck.filter_bundles(
            bundling_result.bundles, 
            parsed_query, 
            max_bundles=self.config.max_bottleneck_bundles,
        )
        
        # Extract chunks from selected bundles for synthesis
        selected_chunks = bundle_selection.get_chunks_for_synthesis()
        bottleneck_result = self.bottleneck.filter(selected_chunks, parsed, conn)
else:
    # Original behavior: filter chunks directly
    bottleneck_result = self.bottleneck.filter(chunks, parsed, conn)
```

**Concordance Resolution:**

The pre-bundler uses concordance tools to resolve surfaces:

1. **Person surfaces** → Look up in `entity_surfaces` table
2. **Codename surfaces** → Check `concordance_mappings` table
3. **Strong links** (confidence ≥ 0.75): Safe to merge codename → person
4. **Weak links** (confidence 0.55-0.75): Keep as "possible", don't merge
5. **Unresolved**: Mark as codename-only, lower priority for roster

**Benefits:**

1. **Codename Safety:** Prevents codenames from being listed as members
2. **Entity Cohesion:** Chunks about the same person are grouped together
3. **Bundle-Level Scoring:** Bottleneck can evaluate evidence coherence
4. **Better Synthesis:** Related evidence is cited together

---

## 5. Data Structures

### 5.1 Core V7 Types

**File:** `retrieval/agent/v7_types.py`

```python
# Claims
@dataclass
class ClaimWithCitation:
    claim_text: str
    citations: List[str]
    support_level: Literal["strong", "weak", "inferred"]

# Output
@dataclass
class ExpandedSummary:
    short_answer: str
    claims: List[ClaimWithCitation]
    unsupported_claims: List[str]
    evidence_used: List[str]

# Result
@dataclass
class V7Result:
    answer: str
    expanded_summary: Optional[ExpandedSummary]
    claims: List[Dict]
    members_identified: List[str]
    all_claims_cited: bool
    citation_validation_passed: bool

# Validation
@dataclass
class StopGateResult:
    can_stop: bool
    reason: str
    invalid_claims: List[ClaimWithCitation]
    invalid_citations: List[str]
```

### 5.2 Phase 2 Types

```python
# Round Summary
@dataclass
class RoundSummary:
    round_number: int
    key_findings: List[KeyFinding]
    actionable_leads: List[ActionableLead]
    decision: RoundDecisionType
    decision_rationale: str
    evidence_count: int
    new_evidence_count: int

# Evidence Bundles
@dataclass
class EvidenceBundle:
    bundle_id: str
    topic: str
    spans: List[BundleSpan]
    status: BundleStatus
    confidence: float
    summary: str
```

---

## 6. Tool Registry

**File:** `retrieval/agent/tools.py`

### Search Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `hybrid_search` | Vector + lexical via RRF | `query`, `top_k`, `exclude_*` |
| `vector_search` | Pure semantic similarity | `query`, `top_k`, `exclude_*` |
| `lexical_search` | All terms must appear | `terms`, `top_k`, `exclude_*` |
| `lexical_exact` | Exact substring match | `term`, `top_k`, `exclude_*` |

### Entity Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `entity_lookup` | Find entity by name | `name` |
| `entity_surfaces` | Get aliases for entity | `entity_id` or `name` |
| `entity_mentions` | Chunks mentioning entity | `name` or `entity_id`, `top_k` |
| `co_mention_entities` | Entities that co-occur | `name` or `entity_id` |
| `first_mention` | Earliest dated mention | `name` or `entity_id` |

### Concordance Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `expand_aliases` | Get variants from concordance | `term`, `max_aliases` |

### Neighbor Tools (Phase 2)

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `chunk_neighbors` | Adjacent chunks in document | `chunk_id`, `before`, `after` |
| `document_chunks` | All chunks from document/page | `document_id`, `page_id`, `limit` |

### Tool Result Structure

```python
@dataclass
class ToolResult:
    tool_name: str
    params: Dict[str, Any]
    chunk_ids: List[int]
    scores: Dict[int, float]
    metadata: Dict[str, Any]
    elapsed_ms: float
    success: bool
    error: Optional[str]
    # Phase 2 pagination
    has_more: bool
    next_cursor: Optional[str]
    total_available: Optional[int]
```

---

## 7. Configuration

### V6 Configuration

```python
@dataclass
class V6Config:
    # Bottleneck
    max_bottleneck_spans: int = 40
    bottleneck_grading_mode: str = "tournament"  # "tournament" or "absolute"
    tournament_batch: bool = False  # True = batch matchups for speed, False = 1 call per matchup (more accurate)
    
    # Rounds
    max_rounds: int = 5
    min_progress_per_round: int = 3
    max_no_progress_rounds: int = 2
    
    # Entity linking
    entity_confidence_threshold: float = 0.6
    
    # Retrieval
    chunks_per_search: int = 150
    
    # Synthesis
    synthesis_model: str = "gpt-4.1-mini-2025-04-14"
    
    # Phase 2: Novelty controls
    exclude_seen_mode: str = "soft"  # off | soft | hard
    top_k_budget_multiplier: float = 1.5
    
    # Phase 2: Round summary
    enable_round_summary: bool = True
    round_summary_model: str = "gpt-4.1-mini-2025-04-14"
```

**Bottleneck Grading Modes:**
- `"tournament"` (default): Pairwise comparison ranking. Spans compete in groups, top half advances. More robust when all spans are moderately relevant.
- `"absolute"`: Each span graded independently with 0-10 score and pass/fail. Original behavior. Faster for small batches with clear relevance distinctions.

### V7 Configuration

```python
@dataclass
class V7Config:
    # V6 settings
    v6_config: Optional[V6Config] = None
    
    # Citation enforcement
    max_repair_attempts: int = 2
    drop_uncited_claims: bool = True
    
    # Output
    include_evidence_section: bool = True
    
    # Phase 2: Bundle builder
    enable_bundles: bool = True
    bundle_config: Optional[BundleBuilderConfig] = None
```

### Bundle Builder Configuration

```python
@dataclass
class BundleBuilderConfig:
    model: str = "gpt-4.1-mini-2025-04-14"
    temperature: float = 0.3
    min_spans_per_bundle: int = 2
    max_bundles: int = 10
    min_confidence: float = 0.5
    max_spans_to_analyze: int = 50
```

---

## 8. Pipeline Flow

### Complete V7 Pipeline

```
1. PARSE QUERY (V6)
   └── Split CONTROL vs CONTENT tokens
   └── Identify task type, scope constraints, topic terms

2. ENTITY LINKING (V6)
   └── Link only CONTENT tokens
   └── Mark use_for_retrieval flag
   └── Concordance expansion

3. AGENTIC RETRIEVAL (V6, enhanced in Phase 2)
   └── For each round:
       ├── Reset round_observations
       ├── LLM decides tool calls
       ├── Execute tools with:
       │   ├── Scope injection (collections)
       │   └── Exclude injection (seen chunks/pages)
       ├── Load and deduplicate chunks
       ├── Record seen IDs
       └── Generate RoundSummary (Phase 2)

4. EVIDENCE BOTTLENECK (V6)
   └── Grade spans for relevance
   └── Keep top 40 spans max
   └── Extract claims each span supports

5. BUILD EVIDENCE BUNDLES (V7 Phase 2)
   └── LLM groups related spans
   └── Generate bundle summaries
   └── Track bundle lifecycle

6. CLAIM ENUMERATION (V7)
   └── Extract atomic claims from answer
   └── Assign citations from evidence
   └── Flag unsupported claims

7. STOP GATE VALIDATION (V7)
   └── Verify all claims have citations
   └── Verify all citations reference real evidence
   └── If invalid: drop uncited claims OR fail

8. RENDER OUTPUT (V7)
   └── Format expanded summary
   └── Include claims with citation references
   └── Include full evidence quotes
   └── Note dropped claims
```

### Sequence Diagram

```
User Question
      │
      ▼
┌─────────────┐
│ QueryParser │ ─── CONTROL: "provide", "cite", "vassiliev"
└─────────────┘     CONTENT: "silvermaster", "network"
      │
      ▼
┌──────────────┐
│ EntityLinker │ ─── Silvermaster → Entity #123
└──────────────┘     (with use_for_retrieval=True)
      │
      ▼
┌──────────────────────────────────────────────────────┐
│              Retrieval Loop (per round)               │
│  ┌─────────────────┐                                  │
│  │ Searcher LLM    │ ── "Call hybrid_search(query=..)"│
│  │ decides tools   │                                  │
│  └─────────────────┘                                  │
│         │                                             │
│         ▼                                             │
│  ┌─────────────────┐                                  │
│  │ Execute Tool    │ ── With exclude_chunk_ids       │
│  │ (novelty inject)│                                  │
│  └─────────────────┘                                  │
│         │                                             │
│         ▼                                             │
│  ┌─────────────────┐                                  │
│  │ Load Chunks     │ ── Track seen_chunk_ids         │
│  └─────────────────┘                                  │
│         │                                             │
│         ▼                                             │
│  ┌─────────────────┐                                  │
│  │ RoundSummary    │ ── "winning leads: Treasury"    │
│  │ Generator       │    "next: chunk_neighbors(14)"  │
│  └─────────────────┘                                  │
└──────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────┐
│   Bottleneck   │ ── 164 chunks → 40 spans
└────────────────┘
      │
      ▼
┌────────────────┐
│ BundleBuilder  │ ── Group related spans
└────────────────┘
      │
      ▼
┌────────────────┐
│   Synthesize   │ ── Generate answer from spans
└────────────────┘
      │
      ▼
┌─────────────────┐
│ClaimEnumerator  │ ── Extract claims, assign citations
└─────────────────┘
      │
      ▼
┌─────────────────┐
│   Stop Gate     │ ── Validate all claims cited
└─────────────────┘
      │
      ▼
┌─────────────────┐
│    Renderer     │ ── Format expanded output
└─────────────────┘
      │
      ▼
  V7Result
```

---

## 9. Design Decisions

### 9.1 Why CONTROL vs CONTENT Separation?

**Problem:** Naive entity linking links "Vassiliev" in "cite Vassiliev notebooks" as a person, causing bad retrieval.

**Solution:** The query parser classifies tokens:
- "Vassiliev" after "cite" or "from" → CONTROL (collection selector)
- "Silvermaster" as subject → CONTENT (entity to search)

**Impact:** Prevents entity linking of instruction words, collection names, and date constraints.

### 9.2 Why Evidence Bottleneck?

**Problem:** Without constraint, 164 chunks lead to thousands of claims, making synthesis unmanageable.

**Solution:** Hard limit of 40 spans forces the system to prioritize.

**Impact:** 
- Synthesis sees only curated evidence
- Claims are grounded in specific spans
- Output is focused and verifiable

### 9.2.1 Why Two Grading Modes (Tournament vs Absolute)?

**Problem with Absolute Grading:** When most spans are "somewhat relevant," LLM scores tend to cluster (e.g., all 6-7), making it hard to distinguish the best evidence. Score drift across batches can cause inconsistent filtering.

**Tournament Mode Solution:** Elo-style rating system with head-to-head comparisons:
- All spans start with rating 1000
- LLM sees two spans and picks the winner: "Which is more relevant: A or B?"
- Ratings update using Elo formula: winner gains points, loser loses points
- Each span participates in ~4 matchups
- Top `max_spans` by final rating are selected

**Why Elo over Elimination:**
- One bad matchup doesn't knock out a good span
- Ratings emerge from multiple comparisons (more stable)
- No seeding or bracket ordering needed
- Same number of LLM calls as elimination

**When to Use Each:**

| Mode | Use When |
|------|----------|
| `tournament` (default) | Large result sets, ambiguous relevance, roster/enumeration queries |
| `absolute` | Small batches, clear relevance distinctions, faster processing needed |

**Impact:** Tournament mode typically produces better evidence selection for complex queries where all spans have some relevance.

### 9.2.2 Tournament Batching Trade-off

The tournament mode supports batching multiple matchups per API call via `tournament_batch` config.

**Single Mode (default, `tournament_batch=False`):**
- Each 1v1 matchup gets its own API call
- LLM gives full attention to each decision
- Slower (N matchups = N API calls)
- Recommended for quality-critical applications

**Batched Mode (`tournament_batch=True`):**
- 8 matchups per API call (configurable via `TOURNAMENT_BATCH_SIZE`)
- ~8x fewer API calls, significant speedup
- LLM must process multiple comparisons at once
- May be slightly less accurate per decision
- Good for high-volume scenarios where speed matters

**Recommendation:** Start with `tournament_batch=False` for quality. Switch to `True` if latency is critical and quality remains acceptable.

### 9.3 Why Citation Enforcement?

**Problem:** LLM-generated answers can include hallucinated claims not supported by evidence.

**Solution:** Every claim must have at least one citation. Claims without citations are:
- Dropped from the answer (default)
- Or cause the pipeline to fail

**Impact:**
- Researcher-grade output
- Verifiable claims
- No hallucination in final answer

### 9.4 Why Exclude-Seen Mode?

**Problem:** Agent repeats identical tool calls, wasting retrieval budget on already-seen content.

**Solution:** Three modes:
- `off`: No exclusion (legacy behavior)
- `soft`: Exclude only bottlenecked chunks (balanced)
- `hard`: Exclude everything seen (maximum novelty)

**Impact:** Each round retrieves genuinely new content.

### 9.5 Why RoundSummary?

**Problem:** Round 2 has no memory of Round 1's strategies, leading to repetition.

**Solution:** LLM generates structured summary after each round with:
- Key findings
- Actionable leads (prioritized)
- Failed strategies to avoid
- Suggested next actions

**Impact:** Rounds build on each other, reducing repetition.

### 9.6 Why Neighbor Tools?

**Problem:** Agent finds a valuable roster span but can't see adjacent content (rest of the list).

**Solution:** Tools to fetch chunks before/after a given chunk in document order.

**Impact:** Agent can expand context around high-value hits without broad re-search.

### 9.7 Why Evidence Bundles?

**Problem:** A roster table split across 3 chunks can't be cited as one unit.

**Solution:** LLM groups related spans into bundles that can be cited together.

**Lifecycle Rule:** Synthesis sees bundles as primary. Spans only appear if not bundled.

**Impact:** Split evidence is properly attributed.

### 9.8 Why Pre-Bundling?

**Problem:** For roster queries, the bottleneck may pass codenames (e.g., "Ruble", "Pal", "Mole") which then appear in answers as if they were member names. Entity resolution happens too late.

**Solution:** Pre-bundling inserts a new phase BEFORE the bottleneck that:
1. Extracts named entity surfaces from chunks (persons, codenames, orgs)
2. Resolves codenames to canonical entity IDs via concordance lookup
3. Groups related chunks into `BundleCandidate` objects with `PERSON_EVIDENCE` vs `CODENAME_EVIDENCE` classification
4. The bottleneck then scores bundles, preferring `PERSON_EVIDENCE` over `CODENAME_EVIDENCE`

**Impact:**
- Codenames are properly identified and not treated as member names
- Related evidence (about same person) is grouped together
- Bottleneck can evaluate bundle coherence, not just individual chunk relevance
- Synthesis receives cleaner, entity-resolved evidence

**When to Enable:**
- `pre_bundling_mode="semantic"` for roster/enumeration queries where codename safety is critical
- `pre_bundling_mode="off"` (default) for simple queries where overhead isn't justified

---

## Appendix: File Reference

```
retrieval/
├── ops.py                          # Low-level search operations
├── agent/
│   ├── tools.py                    # Tool registry
│   ├── entity_surfaces.py          # Entity surface form index
│   │
│   ├── v6_controller.py            # V6 main controller
│   ├── v6_query_parser.py          # CONTROL/CONTENT parser
│   ├── v6_entity_linker.py         # Entity linking
│   ├── v6_evidence_bottleneck.py   # Evidence filtering
│   ├── v6_progress_gate.py         # Round decision gate
│   ├── v6_responsiveness.py        # Answer responsiveness check
│   │
│   ├── v7_controller.py            # V7 main controller
│   ├── v7_types.py                 # Core data structures
│   ├── v7_claim_enumerator.py      # Claim extraction
│   ├── v7_stop_gate.py             # Citation validation
│   ├── v7_expanded_summary.py      # Output rendering
│   ├── v7_bundle_builder.py        # Post-bottleneck evidence bundling
│   ├── v7_bundle_types.py          # Pre-bundling data structures
│   └── v7_pre_bundler.py           # Pre-bottleneck concordance-aware bundling
```

---

*Document generated: February 2026*
*Version: V7 Phase 3 (with Concordance-Aware Pre-Bundling)*
