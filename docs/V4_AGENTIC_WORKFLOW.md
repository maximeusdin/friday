# V4 Agentic Workflow: Grounded Answer Units

## Summary

V4 is a reasoning-first agentic architecture for answering complex research queries against archival documents. Every user-visible response is a list of **Grounded Answer Units** - structured statements with explicit citations and supporting phrases that can be mechanically verified.

**Core invariant**: If you cannot cite it from the evidence, do not state it.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           V4 Agentic Workflow                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    RETRIEVAL LOOP (Outer)                            │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │  1. Plan Generation                                            │  │  │
│  │  │     - Analyze query intent                                     │  │  │
│  │  │     - Select tools (hybrid_search, lexical_exact, etc.)        │  │  │
│  │  │     - Generate execution steps                                 │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │                              ↓                                       │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │  2. Tool Execution                                             │  │  │
│  │  │     - Execute search tools against database                    │  │  │
│  │  │     - Merge results via RRF                                    │  │  │
│  │  │     - Return ranked chunk_ids                                  │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │                              ↓                                       │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │  3. Evidence Building                                          │  │  │
│  │  │     - Load chunk texts from database                           │  │  │
│  │  │     - Mine spans (sentence boundaries, ~500 chars)             │  │  │
│  │  │     - Rerank by embedding similarity                           │  │  │
│  │  │     - Band into cite_spans (top 120) + harvest_spans           │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │                              ↓                                       │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │  4. Span Preparation                                           │  │  │
│  │  │     - Doc-balanced sampling (max 5 per doc, 60 total)          │  │  │
│  │  │     - Attach attest_text (+/- 200 chars context window)        │  │  │
│  │  │     - Load entity mentions + aliases                           │  │  │
│  │  │     - Detect span_type_hint (paragraph/list/header)            │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │                              ↓                                       │  │
│  │  ┌──────────────────────────────────────────────────────────────┐    │  │
│  │  │              INTERPRETATION LOOP (Inner)                     │    │  │
│  │  │  ┌────────────────────────────────────────────────────────┐  │    │  │
│  │  │  │  5. Interpretation (GPT-4o)                            │  │    │  │
│  │  │  │     - Cite-first generation (spans → phrases → text)   │  │    │  │
│  │  │  │     - Produces AnswerUnits with citations              │  │    │  │
│  │  │  │     - Emits uncertainty units for gaps                 │  │    │  │
│  │  │  └────────────────────────────────────────────────────────┘  │    │  │
│  │  │                          ↓                                   │    │  │
│  │  │  ┌────────────────────────────────────────────────────────┐  │    │  │
│  │  │  │  6. Verification (Code, not LLM)                       │  │    │  │
│  │  │  │     - Hard checks: citations, span_idx, phrases        │  │    │  │
│  │  │  │     - Soft warnings: low overlap, overuse              │  │    │  │
│  │  │  │     - Entity attestation via surfaces index            │  │    │  │
│  │  │  └────────────────────────────────────────────────────────┘  │    │  │
│  │  │                          ↓                                   │    │  │
│  │  │  ┌────────────────────────────────────────────────────────┐  │    │  │
│  │  │  │  7. Repair Decision                                    │  │    │  │
│  │  │  │     Stage A: Reinterpret (same evidence)               │  │    │  │
│  │  │  │     Stage B: Expand evidence (revise plan)             │  │    │  │
│  │  │  └────────────────────────────────────────────────────────┘  │    │  │
│  │  └──────────────────────────────────────────────────────────────┘    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                              ↓                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  8. Rendering                                                      │    │
│  │     - Drop failed units, downgrade as needed                       │    │
│  │     - Organize by response_shape (roster/narrative/timeline)       │    │
│  │     - Add "What's unclear" section for uncertainties               │    │
│  │     - Fallback to raw evidence if all units fail                   │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                              ↓                                             │
│                      V4RenderedResponse                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Design Principles

### 1. Grounded Answer Units

The universal primitive for all responses:

```python
@dataclass
class AnswerUnit:
    text: str                              # 1-2 sentences max
    citations: List[SpanCitation]          # 1-3 spans (required)
    supporting_phrases: List[str]          # Verbatim from cited quotes
    confidence: str = "supported"          # supported|suggestive
    about_entities: List[int] = []         # Optional entity IDs
```

- **Every** user-visible statement maps to an AnswerUnit
- Same format works for roster, narrative, timeline - just different rendering
- Uncertainty acknowledged explicitly with empty citations + suggestive confidence

### 2. Cite-First Generation

The prompt forces the model to identify evidence BEFORE writing:

```
For each fact you want to assert:
1. FIRST: Select 1-3 span indices that support it
2. SECOND: Extract the EXACT supporting phrase(s) from those spans
3. THIRD: Write a 1-2 sentence statement constrained to what those phrases say

CRITICAL: If you cannot find a supporting phrase, emit an uncertainty unit.
```

This mechanically prevents "write first, cite later" failures.

### 3. Universal Verifier

The verifier checks contracts, not "understanding." All checks apply to every unit regardless of response shape:

| Check | Type | Description |
|-------|------|-------------|
| `citations_exist` | Hard | Supported units must have citations |
| `span_idx_valid` | Hard | Cited span index exists |
| `supporting_phrase_present` | Hard | Phrases appear verbatim in quotes |
| `entity_attested` | Hard | Entity surfaces found in attest_text |
| `alias_only` | Soft | Entity attested by alias only |
| `low_overlap` | Soft | <30% word overlap with citations |
| `span_overuse` | Soft | Same span cited 3+ times |
| `unit_too_long` | Soft | >2 sentences |

### 4. Two-Stage Repair Loop

**Stage A (Interpretation Retry)** - cheap, same evidence:
- Triggered by: missing_citations, invalid_span_idx, supporting_phrase_missing, entity_not_attested
- Action: Re-run interpretation with error feedback

**Stage B (Evidence Expansion)** - expensive, revise plan:
- Triggered ONLY if:
  1. Model says missing info (via diagnostics.missing_info_questions)
  2. AND keywords absent from all attest_text
- Action: Revise plan, re-retrieve, re-interpret

### 5. Safety Valve: Top Evidence Fallback

If all units fail verification, return raw evidence instead of nothing:

```python
if stats.units_rendered == 0:
    return self._render_fallback(prepared_spans)  # Top 10 evidence spans
```

Never return 0 results to the user.

## Data Flow

### Input
```python
query = "Who were members of the Silvermaster network?"
```

### Phase 1-3: Retrieval (reuses V3)
```python
plan = generate_plan(query, conn)
# AgentPlanV3(steps=[
#   ToolStep(tool="hybrid_search", params={"query": "Silvermaster network members"}),
#   ToolStep(tool="lexical_exact", params={"phrase": "Silvermaster group"}),
# ])

execution_result = executor.execute(plan, conn)
# ExecutionResult(chunk_ids=[123, 456, ...], scores={123: 0.92, ...})

evidence_set = evidence_builder.build(execution_result.chunk_ids, query, conn)
# EvidenceSet(cite_spans=[EvidenceSpan(...), ...], harvest_spans=[...])
```

### Phase 4: Span Preparation
```python
prepared_spans = prepare_spans_for_interpretation(evidence_set, conn)
# [PreparedSpan(
#     span_idx=0,
#     span=EvidenceSpan(quote="Silvermaster group consisted of..."),
#     attest_text="...extended context window...",
#     entities_in_span=[{"entity_id": 123, "canonical_name": "Nathan Silvermaster", "aliases": ["Gregory"]}],
#     span_type_hint="paragraph",
# ), ...]
```

### Phase 5: Interpretation (GPT-4o)
```python
interpretation = interpret_evidence(evidence_set, query, conn)
# InterpretationV4(
#     response_shape="roster",
#     answer_units=[
#         AnswerUnit(
#             text="Nathan Silvermaster led a group of Soviet agents in the Treasury Department.",
#             citations=[SpanCitation(span_idx=0)],
#             supporting_phrases=["Silvermaster group consisted of"],
#             confidence="supported",
#             about_entities=[123],
#         ),
#         AnswerUnit(
#             text="The exact membership dates are not specified in the evidence.",
#             citations=[],  # Uncertainty unit
#             supporting_phrases=[],
#             confidence="suggestive",
#         ),
#     ],
#     diagnostics=DiagnosticsInfo(
#         missing_info_questions=["When did members join?"],
#         followup_queries=["Silvermaster timeline"],
#     ),
# )
```

### Phase 6: Verification
```python
report = verify_interpretation(interpretation, prepared_spans, conn)
# V4VerificationReport(
#     passed=True,
#     passed_units=["abc123"],
#     failed_units=[],
#     hard_errors=[],
#     soft_warnings=[VerificationWarning(type="alias_only", ...)],
# )
```

### Phase 7: Rendering
```python
response = render_interpretation(interpretation, report, prepared_spans)
# V4RenderedResponse(
#     response_shape="roster",
#     sections=[
#         RenderedSection(heading="Confirmed (supported by evidence)", units=[...]),
#         RenderedSection(heading="What's unclear", units=[...]),
#     ],
#     warnings=[],
#     stats=RenderStats(units_rendered=2, units_dropped=0),
# )
```

### Output
```
Confirmed (supported by evidence)
---------------------------------
  • Nathan Silvermaster led a group of Soviet agents in the Treasury Department. [p1]
      "Silvermaster group consisted of..."

What's unclear
--------------
  ? The exact membership dates are not specified in the evidence.

  Questions the evidence doesn't answer:
    - When did members join?

  Suggested followup searches:
    - Silvermaster timeline

  --- Rendering Stats ---
  Rendered: 2 units
  Citations: 1
```

## File Structure

```
retrieval/agent/
├── __init__.py              # Module exports
├── tools.py                 # Tool registry (hybrid_search, lexical_exact, etc.)
├── executor.py              # Tool executor
├── v3_plan.py               # Plan generation/revision
├── v3_evidence.py           # Evidence building (SpanMiner, EvidenceBuilder)
├── entity_surfaces.py       # Entity surface index for attestation
├── v4_interpret.py          # AnswerUnit, InterpretationV4, cite-first prompt
├── v4_verify.py             # Universal verifier (hard/soft checks)
├── v4_render.py             # Grounded response builder + fallback
└── v4_runner.py             # Main orchestrator with repair loop
```

## Configuration

```python
V4_BUDGETS = {
    "max_interpret_rounds": 2,      # Stage A retries
    "max_retrieval_rounds": 2,      # Stage B retries
    "max_spans_to_interpret": 60,   # Spans sent to 4o
    "max_spans_per_doc": 5,         # Doc-balanced sampling
    "context_window_chars": 200,    # attest_text padding
    "max_answer_units": 25,         # Max units per response
}
```

## Usage

### CLI
```bash
python scripts/friday_cli.py
> /v4 Who were members of the Silvermaster network?
```

### Programmatic
```python
from retrieval.agent import V4Runner

runner = V4Runner()
result = runner.run("Who were members of the Silvermaster network?", conn)

print(result.response.format_text())
```

## Determinism & Auditability

- **Span hashing**: Each span has a content-based hash for reproducibility
- **Run traces**: Every retrieval/interpretation round logged
- **Verification reports**: Full per-unit pass/fail status preserved
- **Model versioning**: V4_VERSION and model names recorded

## Response Shapes

The `response_shape` is purely a rendering preference. Same AnswerUnits, different organization:

| Shape | Organization |
|-------|-------------|
| `roster` | Group by confidence (Confirmed / Possible) |
| `narrative` | Key findings / Additional context |
| `timeline` | Sort by extracted dates |
| `index` | Documents and sources |
| `qa` | Direct answer first |
| `fallback` | Raw evidence (when interpretation fails) |

## Error Handling

1. **No evidence found**: Empty interpretation with uncertainty unit
2. **All units fail verification**: Fallback to raw evidence
3. **LLM unavailable**: Fallback interpretation from span previews
4. **Stage B exhausted**: Return best partial result

## Future Work

- [ ] Persist runs to `v4_run_summary` / `v4_run_trace` tables
- [ ] Add streaming support for long interpretations
- [ ] Implement confidence calibration from user feedback
- [ ] Add multi-query batching for efficiency
