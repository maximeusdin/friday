# [Documentation] V4 Agentic Workflow: Grounded Answer Units Architecture

## Overview

This issue documents the V4 agentic architecture, a reasoning-first approach to answering complex research queries against archival documents. V4 introduces **Grounded Answer Units** as the universal primitive for all responses - structured statements with explicit citations and supporting phrases that can be mechanically verified.

**Core invariant**: If you cannot cite it from the evidence, do not state it.

## Motivation

Previous approaches (V2/V3) had limitations:
- **V2**: Claim synthesis could generate ungrounded statements
- **V3**: Shape-specific verification rules (roster anchors, relationship tags) created brittleness
- **Both**: No mechanism to verify that statements were actually supported by cited text

V4 addresses these with:
1. **Cite-first generation**: Model must pick evidence BEFORE writing statements
2. **Universal verification**: Same mechanical checks for all response types
3. **Supporting phrases**: Verbatim substrings from quotes for strict grounding
4. **Safety fallback**: Never return 0 results - show raw evidence if interpretation fails

## Architecture

```
User Query
    ↓
┌─────────────────────────────────────────┐
│         RETRIEVAL LOOP (Outer)          │
│  1. Plan Generation (GPT-4o-mini)       │
│  2. Tool Execution (hybrid/lexical)     │
│  3. Evidence Building (SpanMiner)       │
│  4. Span Preparation (doc-balanced)     │
│     ↓                                   │
│  ┌───────────────────────────────────┐  │
│  │   INTERPRETATION LOOP (Inner)     │  │
│  │  5. Interpretation (GPT-4o)       │  │
│  │  6. Verification (code, not LLM)  │  │
│  │  7. Repair Decision               │  │
│  │     - Stage A: Reinterpret        │  │
│  │     - Stage B: Expand evidence    │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
    ↓
8. Rendering (shape-based organization)
    ↓
V4RenderedResponse
```

## Key Components

### 1. AnswerUnit (Universal Primitive)

```python
@dataclass
class AnswerUnit:
    text: str                              # 1-2 sentences max
    citations: List[SpanCitation]          # 1-3 spans (required)
    supporting_phrases: List[str]          # Verbatim from cited quotes
    confidence: str = "supported"          # supported|suggestive
    about_entities: List[int] = []         # Optional entity IDs
```

- Every user-visible statement is an AnswerUnit
- Same format for roster, narrative, timeline - just different rendering
- Uncertainty = empty citations + suggestive confidence

### 2. Cite-First Prompt

```
For each fact you want to assert:
1. FIRST: Select 1-3 span indices that support it
2. SECOND: Extract the EXACT supporting phrase(s) from those spans
3. THIRD: Write a 1-2 sentence statement constrained to what those phrases say

CRITICAL: If you cannot find a supporting phrase, emit an uncertainty unit.
```

### 3. Universal Verifier

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

No shape-specific logic - all checks apply universally.

### 4. Two-Stage Repair

**Stage A (cheap)**: Re-interpret with same evidence
- Triggers: missing_citations, invalid_span_idx, supporting_phrase_missing, entity_not_attested

**Stage B (expensive)**: Revise plan + re-retrieve
- Triggers ONLY if model says missing info AND keywords absent from evidence

### 5. Safety Fallback

```python
if stats.units_rendered == 0:
    return self._render_fallback(prepared_spans)  # Top 10 raw evidence spans
```

Never return empty results.

## File Structure

```
retrieval/agent/
├── v4_interpret.py    # AnswerUnit, cite-first prompt, InterpretationV4
├── v4_verify.py       # Universal verifier (hard/soft checks)
├── v4_render.py       # Shape-based renderer + fallback
├── v4_runner.py       # Main orchestrator with repair loop
└── entity_surfaces.py # Entity attestation index
```

## Configuration

```python
V4_BUDGETS = {
    "max_interpret_rounds": 2,
    "max_retrieval_rounds": 2,
    "max_spans_to_interpret": 60,
    "max_spans_per_doc": 5,
    "context_window_chars": 200,
    "max_answer_units": 25,
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

## Example Output

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

  --- Rendering Stats ---
  Rendered: 2 units
  Citations: 1
```

## Response Shapes

| Shape | Organization |
|-------|-------------|
| `roster` | Group by confidence (Confirmed / Possible) |
| `narrative` | Key findings / Additional context |
| `timeline` | Sort by extracted dates |
| `index` | Documents and sources |
| `fallback` | Raw evidence (when interpretation fails) |

## Tests

- `tests/test_v4_verification.py` - Universal verifier checks
- `tests/test_v4_rendering.py` - Shape rendering + fallback

## Future Work

- [ ] Persist runs to `v4_run_summary` / `v4_run_trace` tables
- [ ] Streaming support for long interpretations
- [ ] Confidence calibration from user feedback
- [ ] Multi-query batching

## Labels

`documentation`, `architecture`, `agentic-v4`
