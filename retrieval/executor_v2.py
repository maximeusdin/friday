"""
Executor V2 for Agentic V2 Architecture.

Orchestrates the full pipeline:
Plan → Retrieve → FocusBundle → Candidates → Constraints → Hubness → Render → Verify

All steps are deterministic and auditable.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json

from retrieval.query_intent import QueryContract, FocusBundleMode
from retrieval.focus_bundle import (
    FocusBundle,
    FocusBundleBuilder,
    persist_focus_bundle,
    load_focus_bundle,
)
from retrieval.spans import get_mention_offsets_for_chunks, get_page_ref
from retrieval.constraints import (
    build_constraint_scorers,
    assess_candidate,
    CandidateAssessment,
)
from retrieval.candidate_proposer import (
    propose_all_candidates,
    filter_candidates_by_target,
    ProposedCandidate,
)
from retrieval.hubness import (
    score_candidates_with_hubness,
    load_entity_df,
    CandidateScore,
)
from retrieval.expansion import (
    entity_expansion_loop,
    term_expansion_loop,
    should_expand,
)
from retrieval.rendering import (
    render_from_focus_bundle_with_constraints,
    RenderedAnswer,
)
from retrieval.verifier_v2 import (
    FocusBundleVerifier,
    VerificationResult,
)
from retrieval.agent_plan_v2 import AgentPlanV2


@dataclass
class ExecutionResult:
    """Result of executing a V2 plan."""
    answer: RenderedAnswer
    focus_bundle: FocusBundle
    candidates: List[CandidateScore]
    assessments: List[CandidateAssessment]
    verification: VerificationResult
    stats: Dict[str, Any]


class ExecutorV2:
    """
    Executes V2 plans against the database.
    
    Pipeline stages:
    1. Retrieve chunks using lanes
    2. Build FocusBundle (with optional expansion)
    3. Propose candidates from FocusBundle
    4. Score candidates against constraints
    5. Apply hubness penalty
    6. Render answer with constraint-aware citations
    7. Verify against FocusBundle invariants
    """
    
    def __init__(
        self,
        conn,
        retrieval_fn=None,  # Custom retrieval function: (query, terms, collections, conn) -> chunks
    ):
        self.conn = conn
        self.retrieval_fn = retrieval_fn or self._default_retrieval
        self.verifier = FocusBundleVerifier()
    
    def execute(
        self,
        plan: AgentPlanV2,
        retrieval_run_id: int = None,
    ) -> ExecutionResult:
        """
        Execute a V2 plan.
        
        Args:
            plan: The AgentPlanV2 to execute
            retrieval_run_id: Optional ID for persisting results
        
        Returns:
            ExecutionResult with answer, bundle, and verification
        """
        import sys
        
        stats = {"pipeline_stages": []}
        
        print(f"\n  [V2 Execute] Starting execution...", file=sys.stderr)
        print(f"    Query: \"{plan.query_text}\"", file=sys.stderr)
        print(f"    Mode: {plan.mode.value}", file=sys.stderr)
        if plan.constraints:
            print(f"    Constraints: {[c.constraint_key for c in plan.constraints]}", file=sys.stderr)
        
        # Stage 1: Retrieve chunks
        stats["pipeline_stages"].append("retrieval")
        print(f"\n  [Stage 1: Retrieval]", file=sys.stderr)
        chunks = self._retrieve_chunks(plan)
        stats["initial_chunks"] = len(chunks)
        print(f"    Retrieved {len(chunks)} chunks total", file=sys.stderr)
        
        if not chunks:
            print(f"    WARNING: No chunks retrieved! Returning empty result.", file=sys.stderr)
            return self._empty_result(plan, stats)
        
        # Stage 2: Build FocusBundle
        stats["pipeline_stages"].append("focus_bundle")
        print(f"\n  [Stage 2: FocusBundle]", file=sys.stderr)
        contract = plan.query_contract or self._build_default_contract(plan)
        print(f"    Contract mode: {contract.mode.value}", file=sys.stderr)
        print(f"    Targets: {len(contract.targets)}", file=sys.stderr)
        print(f"    Constraints: {len(contract.constraints)}", file=sys.stderr)
        
        builder = FocusBundleBuilder(
            top_n_spans=plan.focus_bundle.top_n_spans,
            lambda_mmr=plan.focus_bundle.lambda_mmr,
            min_span_score=plan.focus_bundle.min_span_score,
            max_spans_per_doc=plan.focus_bundle.max_spans_per_doc,
            max_spans_per_chunk=plan.focus_bundle.max_spans_per_chunk,
            context_fill_quota=plan.focus_bundle.context_fill_quota,
        )
        print(f"    Builder config: top_n={plan.focus_bundle.top_n_spans}, lambda_mmr={plan.focus_bundle.lambda_mmr}, min_score={plan.focus_bundle.min_span_score}", file=sys.stderr)
        
        focus_bundle = builder.build(contract, chunks, self.conn)
        stats["focus_spans"] = len(focus_bundle.spans)
        stats["focus_docs"] = len(focus_bundle.get_unique_doc_ids())
        print(f"    Built FocusBundle: {len(focus_bundle.spans)} spans from {len(focus_bundle.get_unique_doc_ids())} docs", file=sys.stderr)
        
        # Show top spans
        if focus_bundle.spans:
            print(f"    Top 3 spans:", file=sys.stderr)
            for i, span in enumerate(focus_bundle.spans[:3]):
                preview = span.text[:80].replace("\n", " ")
                print(f"      {i+1}. score={span.score:.3f}, chunk={span.chunk_id}, \"{preview}...\"", file=sys.stderr)
        
        # Stage 2b: Optional expansion
        if plan.expansion.enabled and should_expand(focus_bundle, contract):
            stats["pipeline_stages"].append("expansion")
            print(f"\n  [Stage 2b: Expansion]", file=sys.stderr)
            print(f"    Expansion enabled, mode={plan.expansion.mode}", file=sys.stderr)
            
            if plan.expansion.mode == "entity":
                focus_bundle = entity_expansion_loop(
                    contract,
                    focus_bundle,
                    self.conn,
                    self.retrieval_fn,
                    max_rounds=plan.expansion.rounds,
                    max_expanded_chunks=plan.expansion.max_chunks,
                    stability_threshold=plan.expansion.stability_threshold,
                )
            else:
                focus_bundle = term_expansion_loop(
                    contract,
                    focus_bundle,
                    self.conn,
                    self.retrieval_fn,
                    max_rounds=plan.expansion.rounds,
                    max_expanded_chunks=plan.expansion.max_chunks,
                    stability_threshold=plan.expansion.stability_threshold,
                )
            
            stats["expanded_spans"] = len(focus_bundle.spans)
            print(f"    After expansion: {len(focus_bundle.spans)} spans", file=sys.stderr)
        
        # Stage 3: Propose candidates
        stats["pipeline_stages"].append("candidate_proposal")
        print(f"\n  [Stage 3: Candidate Proposal]", file=sys.stderr)
        candidates = propose_all_candidates(
            focus_bundle,
            self.conn,
            include_persons=True,
            include_codenames=True,
        )
        print(f"    Proposed {len(candidates)} candidates from FocusBundle", file=sys.stderr)
        
        # Filter out targets (for relationship queries)
        target_entity_ids = contract.get_target_entity_ids()
        if target_entity_ids:
            before = len(candidates)
            candidates = filter_candidates_by_target(candidates, target_entity_ids)
            print(f"    Filtered out {before - len(candidates)} target entities", file=sys.stderr)
        
        # Show top candidates
        if candidates:
            print(f"    Top 5 candidates:", file=sys.stderr)
            for i, c in enumerate(candidates[:5]):
                print(f"      {i+1}. {c.display_name} (key={c.key}, mentions={c.mention_count}, resolved={c.is_resolved})", file=sys.stderr)
        
        stats["proposed_candidates"] = len(candidates)
        
        # Stage 4: Score constraints
        stats["pipeline_stages"].append("constraint_scoring")
        print(f"\n  [Stage 4: Constraint Scoring]", file=sys.stderr)
        target_entity_map = {
            t.surface_forms[0]: t.entity_id
            for t in contract.targets
            if t.entity_id and t.surface_forms
        }
        
        scorers = build_constraint_scorers(contract.constraints, target_entity_map)
        print(f"    Built {len(scorers)} constraint scorers: {[s.constraint_key for s in scorers]}", file=sys.stderr)
        
        assessments = []
        for candidate in candidates:
            assessment = assess_candidate(
                candidate.key,
                candidate.entity_id,
                candidate.display_name,
                scorers,
                focus_bundle,
                self.conn,
            )
            assessments.append(assessment)
        
        # Show assessment summary
        if assessments:
            print(f"    Assessed {len(assessments)} candidates", file=sys.stderr)
            for i, a in enumerate(assessments[:3]):
                supports_str = ", ".join([f"{s.constraint_name}={s.score:.2f}" for s in a.supports]) or "no constraints"
                print(f"      {i+1}. {a.display_name}: final={a.final_score:.3f}, [{supports_str}]", file=sys.stderr)
        
        stats["assessed_candidates"] = len(assessments)
        
        # Stage 5: Apply hubness penalty
        stats["pipeline_stages"].append("hubness_scoring")
        print(f"\n  [Stage 5: Hubness Scoring]", file=sys.stderr)
        entity_df = load_entity_df(self.conn)
        print(f"    Loaded entity_df with {len(entity_df)} entities", file=sys.stderr)
        
        hubness_scores = score_candidates_with_hubness(
            candidates,
            focus_bundle,
            entity_df,
            self.conn,
        )
        stats["scored_candidates"] = len(hubness_scores)
        print(f"    Scored {len(hubness_scores)} candidates with hubness penalty", file=sys.stderr)
        
        if hubness_scores:
            print(f"    Top 5 after hubness:", file=sys.stderr)
            for i, s in enumerate(hubness_scores[:5]):
                print(f"      {i+1}. {s.display_name}: support={s.support:.3f}, spec={s.specificity:.3f}, final={s.final_score:.3f}", file=sys.stderr)
        
        # Stage 6: Render answer
        stats["pipeline_stages"].append("rendering")
        print(f"\n  [Stage 6: Rendering]", file=sys.stderr)
        print(f"    Constraints for rendering: {[c.constraint_key for c in contract.constraints]}", file=sys.stderr)
        print(f"    Constraint thresholds: {[(c.constraint_key, c.min_score) for c in contract.constraints]}", file=sys.stderr)
        print(f"    Hubness scores available: {len(hubness_scores)} (for evidence linkage check)", file=sys.stderr)
        
        answer = render_from_focus_bundle_with_constraints(
            focus_bundle,
            assessments,
            contract.constraints,
            max_items=plan.output.max_items,
            max_citations_per_item=plan.output.max_citations_per_item,
            conservative_language=plan.output.conservative_language,
            hubness_scores=hubness_scores,  # Pass for evidence linkage check
        )
        stats["rendered_bullets"] = len(answer.bullets)
        print(f"    Rendered {len(answer.bullets)} bullets from {len(assessments)} assessments", file=sys.stderr)
        
        if not answer.bullets and assessments:
            print(f"    INFO: No candidate bullets rendered - falling back to evidence spans", file=sys.stderr)
        
        # Stage 7: Verify
        stats["pipeline_stages"].append("verification")
        print(f"\n  [Stage 7: Verification]", file=sys.stderr)
        verification = self.verifier.verify_answer(
            answer.bullets,
            focus_bundle,
            assessments,
            contract.constraints,
        )
        stats["verification_passed"] = verification.passed
        stats["verification_errors"] = len(verification.errors)
        print(f"    Verification: {'PASSED' if verification.passed else 'FAILED'}", file=sys.stderr)
        if verification.errors:
            print(f"    Errors: {verification.errors[:3]}", file=sys.stderr)
        if verification.warnings:
            print(f"    Warnings: {verification.warnings[:3]}", file=sys.stderr)
        
        # Persist if requested
        if retrieval_run_id:
            persist_focus_bundle(focus_bundle, retrieval_run_id, self.conn, contract)
        
        return ExecutionResult(
            answer=answer,
            focus_bundle=focus_bundle,
            candidates=hubness_scores,
            assessments=assessments,
            verification=verification,
            stats=stats,
        )
    
    def _retrieve_chunks(self, plan: AgentPlanV2) -> List[dict]:
        """Retrieve chunks using configured lanes."""
        all_chunks = []
        seen_ids = set()
        
        # Build terms from contract
        terms = []
        if plan.query_contract:
            terms.extend(plan.query_contract.get_target_surface_forms())
        
        collections = []
        if plan.query_contract:
            collections = plan.query_contract.scope_collections
        
        # Run retrieval
        chunks = self.retrieval_fn(
            plan.query_text,
            terms,
            collections,
            self.conn,
        )
        
        # Deduplicate
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id') or chunk.get('id')
            if chunk_id not in seen_ids:
                all_chunks.append(chunk)
                seen_ids.add(chunk_id)
        
        return all_chunks
    
    def _default_retrieval(
        self,
        query: str,
        terms: List[str],
        collections: List[str],
        conn,
    ) -> List[dict]:
        """
        Default retrieval implementation using existing retrieval ops.
        
        Can be overridden with custom retrieval_fn.
        """
        import sys
        
        try:
            from retrieval.ops import hybrid_rrf, SearchFilters
            
            # Build search filters
            filters = SearchFilters()
            if collections:
                filters.collection_slugs = collections
                print(f"    [Retrieval] Collection filter: {collections}", file=sys.stderr)
            
            print(f"    [Retrieval] Query: \"{query}\"", file=sys.stderr)
            print(f"    [Retrieval] Running hybrid_rrf with k=200, expand_concordance=True...", file=sys.stderr)
            
            # Run hybrid RRF search with concordance expansion
            results = hybrid_rrf(
                conn=conn,
                query=query,
                filters=filters,
                k=200,
                expand_concordance=True,  # Enable query expansion
                fuzzy_lex_enabled=True,   # Enable fuzzy lexical matching
            )
            
            print(f"    [Retrieval] Got {len(results)} chunks from hybrid_rrf", file=sys.stderr)
            
            if not results:
                print(f"    [Retrieval] WARNING: No results from hybrid search!", file=sys.stderr)
                return []
            
            # Show top results
            print(f"    [Retrieval] Top 5 chunks:", file=sys.stderr)
            for i, hit in enumerate(results[:5]):
                score_info = f"score={hit.score:.3f}" if hit.score else f"r_vec={hit.r_vec}, r_lex={hit.r_lex}"
                print(f"      {i+1}. chunk_id={hit.chunk_id}, doc={hit.document_id}, {score_info}", file=sys.stderr)
                preview = (hit.preview or "")[:100].replace("\n", " ")
                print(f"         \"{preview}...\"", file=sys.stderr)
            
            # Get chunk IDs to load full text
            chunk_ids = [hit.chunk_id for hit in results]
            
            # Load full text and page numbers from database
            chunk_data = self._load_chunk_texts(conn, chunk_ids)
            print(f"    [Retrieval] Loaded full text for {len(chunk_data)} chunks", file=sys.stderr)
            
            # Convert ChunkHit objects to chunk format with full text
            chunks = []
            for hit in results:
                data = chunk_data.get(hit.chunk_id, {})
                chunk = {
                    'chunk_id': hit.chunk_id,
                    'doc_id': hit.document_id,
                    'text': data.get('text', hit.preview or ''),
                    'page_ref': get_page_ref(data.get('page_number')),
                    'source_lanes': ['hybrid_rrf'],
                }
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            import traceback
            print(f"Retrieval error: {e}", file=sys.stderr)
            traceback.print_exc()
            return []
    
    def _load_chunk_texts(self, conn, chunk_ids: List[int]) -> Dict[int, dict]:
        """Load full chunk text and metadata from database."""
        if not chunk_ids:
            return {}
        
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                c.id as chunk_id,
                COALESCE(c.clean_text, c.text) as text,
                cm.first_page_id
            FROM chunks c
            LEFT JOIN chunk_metadata cm ON cm.chunk_id = c.id
            WHERE c.id = ANY(%s)
        """, (chunk_ids,))
        
        result = {}
        for chunk_id, text, first_page_id in cur.fetchall():
            result[chunk_id] = {
                'text': text or '',
                'page_number': first_page_id,  # Use first_page_id as page number
            }
        
        return result
    
    def _build_default_contract(self, plan: AgentPlanV2) -> QueryContract:
        """Build default QueryContract from plan."""
        from retrieval.query_intent import build_keyword_intent_contract
        return build_keyword_intent_contract(plan.query_text)
    
    def _empty_result(
        self,
        plan: AgentPlanV2,
        stats: Dict[str, Any],
    ) -> ExecutionResult:
        """Return empty result when no chunks found."""
        from retrieval.verifier_v2 import VerificationResult
        
        empty_bundle = FocusBundle(
            query_text=plan.query_text,
            spans=[],
            params={},
        )
        
        empty_answer = RenderedAnswer(
            short_answer="No relevant documents found.",
            bullets=[],
            focus_bundle_id=None,
            total_candidates=0,
            rendered_count=0,
            negative_findings="No documents matched the query criteria.",
        )
        
        return ExecutionResult(
            answer=empty_answer,
            focus_bundle=empty_bundle,
            candidates=[],
            assessments=[],
            verification=VerificationResult(
                passed=True,
                errors=[],
                warnings=["No documents found"],
                stats={},
            ),
            stats=stats,
        )


def execute_v2_plan(
    plan: AgentPlanV2,
    conn,
    retrieval_fn=None,
    retrieval_run_id: int = None,
) -> ExecutionResult:
    """
    Convenience function to execute a V2 plan.
    
    Args:
        plan: The AgentPlanV2 to execute
        conn: Database connection
        retrieval_fn: Optional custom retrieval function
        retrieval_run_id: Optional ID for persisting results
    
    Returns:
        ExecutionResult
    """
    executor = ExecutorV2(conn, retrieval_fn)
    return executor.execute(plan, retrieval_run_id)


def execute_relationship_query(
    query_text: str,
    target_entity_id: int,
    target_name: str,
    conn,
    target_aliases: List[str] = None,
    role_constraint: str = None,
    collections: List[str] = None,
) -> ExecutionResult:
    """
    Execute a relationship query.
    
    Example: "officers closely associated with Julius Rosenberg"
    """
    from retrieval.agent_plan_v2 import build_relationship_plan
    
    plan = build_relationship_plan(
        query_text=query_text,
        target_entity_id=target_entity_id,
        target_name=target_name,
        target_aliases=target_aliases,
        role_constraint=role_constraint,
        collections=collections,
    )
    
    return execute_v2_plan(plan, conn)


def execute_affiliation_query(
    query_text: str,
    org_name: str,
    conn,
    org_aliases: List[str] = None,
    role_constraint: str = None,
    collections: List[str] = None,
) -> ExecutionResult:
    """
    Execute an affiliation query.
    
    Example: "Soviet agents in the OSS"
    """
    from retrieval.agent_plan_v2 import build_affiliation_plan
    
    plan = build_affiliation_plan(
        query_text=query_text,
        org_name=org_name,
        org_aliases=org_aliases,
        role_constraint=role_constraint,
        collections=collections,
    )
    
    return execute_v2_plan(plan, conn)


def execute_keyword_query(
    query_text: str,
    conn,
    collections: List[str] = None,
) -> ExecutionResult:
    """
    Execute a simple keyword query.
    
    Example: "proximity fuse"
    """
    from retrieval.agent_plan_v2 import build_keyword_intent_plan
    
    plan = build_keyword_intent_plan(query_text, collections)
    
    return execute_v2_plan(plan, conn)
