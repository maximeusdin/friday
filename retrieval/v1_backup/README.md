# Agentic Workflow V1 Backup

This directory contains the backup of the v1 agentic workflow implementation.

## Files

- `plan.py` - AgenticPlan, LaneSpec, Budgets, StopConditions
- `evidence_bundle.py` - EvidenceBundle, Claim, EvidenceRef, EntityCandidate
- `lanes.py` - Multi-lane retrieval with iterative execution
- `intent.py` - Intent classification (EXISTENCE_EVIDENCE, ROSTER_ENUMERATION, RELATIONSHIP_CONSTRAINED)
- `claim_extraction.py` - Deterministic claim extraction
- `verifier.py` - Intent-specific verification
- `answer_trace.py` - Human-readable trace generator
- `codename_resolution.py` - Codename mapping extraction
- `synthesis_v1.py` - Rendering from evidence bundle

## To Restore

```bash
copy retrieval\v1_backup\*.py retrieval\
copy retrieval\v1_backup\synthesis_v1.py backend\app\services\summarizer\synthesis.py
```

## Architecture Summary (V1)

```
Query → Intent Classification → Entity Linking → Plan Building →
  → Multi-Lane Retrieval (iterative) → Claim Extraction →
  → Codename Resolution → Verification → Rendering → Answer
```

Key characteristics:
- Intent-driven control flow
- Chunk-level evidence
- Co-occurrence claims (ASSOCIATED_WITH) as findings
- Jaccard stability on chunk IDs
- Mention count ranking

## Why V2?

V1 had issues with:
- Hub entities (Moscow, NYU) dominating results due to mention count
- Co-occurrence treated as evidence rather than candidate generation
- Chunk-level citations allowing noise
- Intent families controlling correctness logic
- Volume-based confidence

V2 addresses these with FocusBundle (span-level evidence gating) and hubness penalties.
