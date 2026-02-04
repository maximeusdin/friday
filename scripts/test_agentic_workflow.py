#!/usr/bin/env python3
"""
test_agentic_workflow.py - Test the agentic workflow implementation

Tests the full Plan → Execute → Verify → Render architecture:
1. Intent classification
2. Multi-lane retrieval
3. Claim extraction
4. Codename resolution
5. Verification
6. Rendering

Usage:
    python scripts/test_agentic_workflow.py
    python scripts/test_agentic_workflow.py --query "Who were handlers of Julius Rosenberg?"
    python scripts/test_agentic_workflow.py --verbose
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from retrieval.plan import (
            AgenticPlan, LaneSpec, ExtractionSpec, VerificationSpec,
            Budgets, StopConditions, BundleConstraints,
        )
        print("  [OK] retrieval.plan")
    except ImportError as e:
        print(f"  [FAIL] retrieval.plan: {e}")
        return False
    
    try:
        from retrieval.evidence_bundle import (
            EvidenceBundle, Claim, EvidenceRef, EntityCandidate,
            RetrievalLaneRun, ChunkWithProvenance, Predicate, SupportType,
        )
        print("  [OK] retrieval.evidence_bundle")
    except ImportError as e:
        print(f"  [FAIL] retrieval.evidence_bundle: {e}")
        return False
    
    try:
        from retrieval.intent import (
            IntentFamily, classify_intent, compute_coverage,
        )
        print("  [OK] retrieval.intent")
    except ImportError as e:
        print(f"  [FAIL] retrieval.intent: {e}")
        return False
    
    try:
        from retrieval.lanes import (
            execute_plan_iterative, execute_lane, merge_lane_chunks,
            expand_query_ephemeral, lexical_hit_test,
        )
        print("  [OK] retrieval.lanes")
    except ImportError as e:
        print(f"  [FAIL] retrieval.lanes: {e}")
        return False
    
    try:
        from retrieval.claim_extraction import (
            extract_claims, classify_support_type,
        )
        print("  [OK] retrieval.claim_extraction")
    except ImportError as e:
        print(f"  [FAIL] retrieval.claim_extraction: {e}")
        return False
    
    try:
        from retrieval.codename_resolution import (
            resolve_codenames, extract_explicit_mappings,
        )
        print("  [OK] retrieval.codename_resolution")
    except ImportError as e:
        print(f"  [FAIL] retrieval.codename_resolution: {e}")
        return False
    
    try:
        from retrieval.verifier import (
            verify_evidence_bundle, verify_role_evidence,
        )
        print("  [OK] retrieval.verifier")
    except ImportError as e:
        print(f"  [FAIL] retrieval.verifier: {e}")
        return False
    
    try:
        from retrieval.answer_trace import (
            generate_answer_trace, AnswerTrace,
        )
        print("  [OK] retrieval.answer_trace")
    except ImportError as e:
        print(f"  [FAIL] retrieval.answer_trace: {e}")
        return False
    
    try:
        from backend.app.services.summarizer.synthesis import (
            render_from_bundle, BulletWithCitation, NegativeAnswerTemplate,
        )
        print("  [OK] backend.app.services.summarizer.synthesis (render_from_bundle)")
    except ImportError as e:
        print(f"  [FAIL] backend.app.services.summarizer.synthesis: {e}")
        return False
    
    print("All imports successful!\n")
    return True


def test_intent_classification():
    """Test intent classification."""
    print("Testing intent classification...")
    
    from retrieval.intent import classify_intent, IntentFamily
    
    test_cases = [
        ("Who were members of the Silvermaster network?", IntentFamily.ROSTER_ENUMERATION),
        ("Who handled Julius Rosenberg?", IntentFamily.RELATIONSHIP_CONSTRAINED),
        ("Is there evidence about the proximity fuse?", IntentFamily.EXISTENCE_EVIDENCE),
        ("Soviet intelligence officers associated with atomic espionage", IntentFamily.RELATIONSHIP_CONSTRAINED),
        ("List all people in the Venona decrypts", IntentFamily.ROSTER_ENUMERATION),
    ]
    
    all_passed = True
    for query, expected_intent in test_cases:
        result = classify_intent(query, [])
        status = "[OK]" if result.intent_family == expected_intent else "[FAIL]"
        if result.intent_family != expected_intent:
            all_passed = False
        print(f"  {status} \"{query[:50]}...\"")
        print(f"      Expected: {expected_intent.value}, Got: {result.intent_family.value} "
              f"(confidence: {result.confidence:.2f})")
    
    print()
    return all_passed


def test_plan_creation():
    """Test plan creation."""
    print("Testing plan creation...")
    
    from retrieval.plan import (
        AgenticPlan, LaneSpec, ExtractionSpec, VerificationSpec,
        Budgets, StopConditions, BundleConstraints,
        build_default_lanes_for_intent,
    )
    from retrieval.intent import IntentFamily
    
    # Test building lanes for different intents
    for intent in IntentFamily:
        lanes = build_default_lanes_for_intent(
            intent=intent,
            entity_ids=[1, 2],
            query_terms=["test", "query"],
            collection_scope=["venona"],
        )
        print(f"  [OK] {intent.value}: {len(lanes)} lanes created")
    
    # Test plan validation
    plan = AgenticPlan(
        intent=IntentFamily.EXISTENCE_EVIDENCE,
        constraints=BundleConstraints(),
        lanes=[LaneSpec(lane_id="hybrid", query_terms=["test"])],
        extraction=ExtractionSpec(),
        verification=VerificationSpec(),
        budgets=Budgets(),
        stop_conditions=StopConditions(),
        query_text="test query",
    )
    
    errors = plan.validate()
    if errors:
        print(f"  [FAIL] Plan validation: {errors}")
        return False
    else:
        print(f"  [OK] Plan validation passed")
    
    # Test serialization
    plan_dict = plan.to_dict()
    print(f"  [OK] Plan serialization: {len(plan_dict)} keys")
    
    print()
    return True


def test_evidence_bundle_creation():
    """Test evidence bundle creation."""
    print("Testing evidence bundle creation...")
    
    from retrieval.evidence_bundle import (
        EvidenceBundle, Claim, EvidenceRef, EntityCandidate,
        Predicate, SupportType,
    )
    
    # Create evidence ref
    ref = EvidenceRef(
        chunk_id=1,
        doc_id=1,
        page_ref="p.5",
        char_start=100,
        char_end=200,
        quote_span="Test quote span for evidence",
    )
    print(f"  [OK] EvidenceRef created, ref_id: {ref.ref_id}")
    
    # Create claim
    claim = Claim(
        subject="entity:123",
        predicate=Predicate.MENTIONS,
        object="entity:456",
        evidence=[ref],
        support_type=SupportType.EXPLICIT_STATEMENT,
        support_strength=3,
        source_lane="hybrid",
        confidence=0.9,
    )
    print(f"  [OK] Claim created, claim_id: {claim.claim_id}")
    
    # Create entity candidate
    entity = EntityCandidate.from_entity_mention(
        entity_id=123,
        canonical_name="Test Person",
        evidence_ref=ref,
        round_num=1,
    )
    print(f"  [OK] EntityCandidate created, key: {entity.key}")
    
    # Create bundle
    bundle = EvidenceBundle()
    bundle.add_claim(claim)
    bundle.add_entity(entity)
    
    print(f"  [OK] EvidenceBundle: {len(bundle.claims)} claims, {len(bundle.entities)} entities")
    
    # Test serialization
    bundle_dict = bundle.to_dict()
    print(f"  [OK] Bundle serialization: {len(bundle_dict)} keys")
    
    print()
    return True


def test_full_workflow(conn, query: str, verbose: bool = False):
    """Test the full agentic workflow with a real query."""
    print(f"Testing full workflow: \"{query}\"")
    print("=" * 60)
    
    from retrieval.plan import (
        AgenticPlan, LaneSpec, ExtractionSpec, VerificationSpec,
        Budgets, StopConditions, BundleConstraints,
        build_default_lanes_for_intent, build_verification_spec_for_intent,
    )
    from retrieval.intent import classify_intent, IntentFamily
    from retrieval.lanes import execute_plan_iterative
    from retrieval.claim_extraction import extract_claims
    from retrieval.codename_resolution import resolve_codenames
    from retrieval.verifier import verify_evidence_bundle
    from retrieval.answer_trace import generate_answer_trace
    from backend.app.services.summarizer.synthesis import render_from_bundle
    
    # Step 1: Classify intent
    print("\n1. Classifying intent...")
    intent_result = classify_intent(query, [])
    print(f"   Intent: {intent_result.intent_family.value}")
    print(f"   Confidence: {intent_result.confidence:.2f}")
    if intent_result.uncertainties:
        print(f"   Uncertainties: {intent_result.uncertainties}")
    
    # Step 2: Build plan
    print("\n2. Building plan...")
    anchors = intent_result.anchors
    constraints = BundleConstraints(
        collection_scope=anchors.constraints.get("collection_scope", []),
        required_role_evidence_patterns=anchors.constraints.get("role_evidence_patterns", []),
    )
    
    lanes = build_default_lanes_for_intent(
        intent=intent_result.intent_family,
        entity_ids=anchors.target_entities,
        query_terms=anchors.key_concepts + anchors.target_tokens,
        collection_scope=constraints.collection_scope,
    )
    print(f"   Lanes: {[l.lane_id for l in lanes]}")
    
    verification = build_verification_spec_for_intent(
        intent=intent_result.intent_family,
        role_patterns=constraints.required_role_evidence_patterns,
        collection_scope=constraints.collection_scope,
    )
    
    plan = AgenticPlan(
        intent=intent_result.intent_family,
        constraints=constraints,
        lanes=lanes,
        extraction=ExtractionSpec(),
        verification=verification,
        budgets=Budgets(),
        stop_conditions=StopConditions(),
        query_text=query,
    )
    
    errors = plan.validate()
    if errors:
        print(f"   [FAIL] Plan validation failed: {errors}")
        return False
    print(f"   [OK] Plan validated")
    
    # Step 3: Execute
    print("\n3. Executing retrieval...")
    bundle = execute_plan_iterative(plan, conn)
    print(f"   Rounds: {bundle.rounds_executed}")
    print(f"   Chunks: {len(bundle.all_chunks)}")
    print(f"   Entities: {len(bundle.entities)}")
    print(f"   Stable: {bundle.stable}")
    
    if verbose and bundle.entities:
        print("   Top entities:")
        for entity in list(bundle.entities)[:5]:
            print(f"     - {entity.display_name} ({entity.mention_count} mentions)")
    
    # Step 4: Extract claims
    print("\n4. Extracting claims...")
    chunks_list = list(bundle.all_chunks.values())
    candidates_dict = {e.key: e for e in bundle.entities}
    
    claims = extract_claims(
        chunks=chunks_list,
        candidates=candidates_dict,
        intent=intent_result.intent_family,
        extraction_spec=plan.extraction,
        conn=conn,
    )
    bundle.claims = claims
    print(f"   Claims: {len(claims)}")
    
    if verbose and claims:
        print("   Sample claims:")
        for claim in claims[:3]:
            print(f"     - {claim.subject} {claim.predicate.value} {claim.object}")
    
    # Step 5: Resolve codenames
    if bundle.unresolved_tokens:
        print(f"\n5. Resolving {len(bundle.unresolved_tokens)} codenames...")
        resolution = resolve_codenames(
            bundle.unresolved_tokens,
            constraints.collection_scope,
            conn,
        )
        bundle.claims.extend(resolution.claims)
        print(f"   Resolved: {len(resolution.resolved)}")
        print(f"   Unresolved: {len(resolution.unresolved)}")
    else:
        print("\n5. No codenames to resolve")
    
    # Step 6: Verify
    print("\n6. Verifying bundle...")
    verification_result = verify_evidence_bundle(bundle, conn)
    print(f"   Passed: {verification_result.passed}")
    print(f"   Errors: {len(verification_result.errors)}")
    print(f"   Warnings: {len(verification_result.warnings)}")
    print(f"   Dropped claims: {len(verification_result.dropped_claims)}")
    
    if verbose and verification_result.errors:
        print("   Errors:")
        for error in verification_result.errors[:3]:
            print(f"     - {error.message}")
    
    # Step 7: Generate trace
    print("\n7. Generating trace...")
    trace = generate_answer_trace(bundle, verification_result)
    print(f"   Trace ID: {trace.trace_id}")
    
    # Step 8: Render
    print("\n8. Rendering answer...")
    rendered = render_from_bundle(bundle)
    print(f"   Claims rendered: {rendered.claims_rendered}")
    print(f"   Citations: {rendered.citations_included}")
    
    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(f"\n{rendered.short_answer}")
    
    if rendered.bullets:
        print("\nFindings:")
        for bullet in rendered.bullets:
            marker = "•" if bullet.inference_level == "explicit" else "o"
            print(f"  {marker} {bullet.text}")
    
    if rendered.negative_answer:
        neg = rendered.negative_answer
        print(f"\n{neg.statement}")
        print(f"({neg.caveat})")
    
    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)
    
    return True


def get_conn():
    """Get database connection."""
    import psycopg2
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL not set")
        print("Run: source friday_env.sh")
        sys.exit(1)
    return psycopg2.connect(dsn)


def main():
    parser = argparse.ArgumentParser(description="Test agentic workflow")
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Query to test with full workflow"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output"
    )
    parser.add_argument(
        "--imports-only",
        action="store_true",
        help="Only test imports"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  AGENTIC WORKFLOW TEST")
    print("=" * 60 + "\n")
    
    # Test imports
    if not test_imports():
        print("\n[FAIL] Import tests failed. Fix imports before continuing.")
        sys.exit(1)
    
    if args.imports_only:
        print("\n[OK] All imports successful!")
        sys.exit(0)
    
    # Test intent classification
    if not test_intent_classification():
        print("\n[WARN] Some intent classification tests failed (non-critical)")
    
    # Test plan creation
    if not test_plan_creation():
        print("\n[FAIL] Plan creation tests failed")
        sys.exit(1)
    
    # Test evidence bundle creation
    if not test_evidence_bundle_creation():
        print("\n[FAIL] Evidence bundle tests failed")
        sys.exit(1)
    
    # Test full workflow if query provided
    if args.query:
        print("\n" + "-" * 60)
        conn = get_conn()
        try:
            if not test_full_workflow(conn, args.query, args.verbose):
                print("\n[FAIL] Full workflow test failed")
                sys.exit(1)
        finally:
            conn.close()
    else:
        print("\n" + "-" * 60)
        print("To test full workflow, run:")
        print("  python scripts/test_agentic_workflow.py --query \"Who were handlers of Rosenberg?\"")
    
    print("\n[OK] All tests passed!")


if __name__ == "__main__":
    main()
