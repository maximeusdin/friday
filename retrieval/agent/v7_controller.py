"""
V7 Controller - Citation Enforcement Layer

V7 extends V6 with citation enforcement:
- Every claim in the final answer MUST have at least one citation
- Claims without citations are either repaired or removed
- Output includes an expanded summary with claims & citations

V7 Phase 2 adds:
- RoundSummary generation for structured decision state between rounds

Pipeline:
1. Run V6 (query parsing, entity linking, retrieval, bottleneck, synthesis)
2. Enumerate claims from the V6 answer
3. Validate via stop gate (all claims must have citations)
4. If invalid, repair or remove uncited claims
5. Render expanded summary
"""
import os
import json
import sys
import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

from retrieval.agent.v6_controller import V6Controller, V6Config, V6Trace
from retrieval.agent.v7_types import (
    ClaimWithCitation,
    ExpandedSummary,
    V7Result,
    StopGateResult,
    # V7 Phase 2 types
    RoundSummary,
    RoundDecisionType,
    ActionableLead,
    KeyFinding,
    LeadPriority,
    # V7 Phase 2 bundles
    EvidenceBundle,
    BundleCollection,
    BundleStatus,
)
from retrieval.agent.v7_claim_enumerator import ClaimEnumerator
from retrieval.agent.v7_stop_gate import StopGate
from retrieval.agent.v7_expanded_summary import ExpandedSummaryRenderer
from retrieval.agent.v7_bundle_builder import BundleBuilder, BundleBuilderConfig


# =============================================================================
# V7 Phase 2: Round Summary Generator
# =============================================================================

class RoundSummaryGenerator:
    """
    V7 Phase 2: LLM-based round summary generator.
    
    Generates structured RoundSummary after each retrieval round by:
    - Analyzing evidence found so far
    - Identifying actionable leads (entities, documents, terms)
    - Summarizing key findings
    - Recommending next steps (continue, pivot, narrow, stop)
    """
    
    def __init__(self, verbose: bool = True, model: str = "gpt-4o-mini"):
        self.verbose = verbose
        self.model = model
    
    def generate(
        self,
        round_number: int,
        question: str,
        evidence_chunks: List[Dict[str, Any]],
        previous_summary: Optional[RoundSummary] = None,
        tool_observations: Optional[List[Dict[str, Any]]] = None,
    ) -> RoundSummary:
        """
        Generate a structured round summary.
        
        Args:
            round_number: Current round number
            question: Original user question
            evidence_chunks: Chunks found this round
            previous_summary: Summary from previous round (if any)
            tool_observations: Results from tool calls this round
        
        Returns:
            RoundSummary with findings, leads, and recommendations
        """
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Return a basic summary without LLM
            return self._fallback_summary(round_number, evidence_chunks, previous_summary)
        
        # Build context for LLM
        chunks_context = self._format_chunks_for_prompt(evidence_chunks[:30])
        previous_context = previous_summary.format_for_context() if previous_summary else ""
        tools_context = self._format_tools_for_prompt(tool_observations or [])
        
        prompt = f"""Analyze this retrieval round and generate a structured summary.

ORIGINAL QUESTION: {question}

{previous_context}

ROUND {round_number} EVIDENCE ({len(evidence_chunks)} chunks):
{chunks_context}

TOOL RESULTS THIS ROUND:
{tools_context}

Generate a structured analysis. Output JSON with:
{{
  "key_findings": [
    {{"finding": "...", "confidence": 0.0-1.0, "finding_type": "fact|relationship|date|context"}}
  ],
  "actionable_leads": [
    {{
      "lead_type": "entity|document|term|codename|date_range",
      "target": "the entity/term/etc",
      "rationale": "why this lead is promising",
      "priority": "high|medium|low",
      "suggested_tool": "optional tool suggestion"
    }}
  ],
  "information_gaps": ["questions still unanswered"],
  "successful_strategies": ["tools/queries that worked"],
  "failed_strategies": ["tools/queries that didn't help"],
  "decision": "continue|pivot|narrow|expand|stop_sufficient|stop_exhausted",
  "decision_rationale": "why this decision",
  "coverage_estimate": 0.0-1.0
}}

DECISION GUIDELINES:
- "continue": More evidence available, current strategy working
- "pivot": Current strategy not working, try different approach
- "narrow": Found promising lead, focus on it
- "expand": Too narrow, need broader search
- "stop_sufficient": Have enough quality evidence to answer
- "stop_exhausted": Tried everything, no more leads"""

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You analyze archival research evidence and produce structured summaries. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=1500,
            )
            
            content = response.choices[0].message.content
            if content:
                data = json.loads(content)
                return self._parse_llm_response(round_number, data, evidence_chunks, previous_summary)
                
        except Exception as e:
            if self.verbose:
                print(f"    [RoundSummary] LLM error: {e}", file=sys.stderr)
        
        return self._fallback_summary(round_number, evidence_chunks, previous_summary)
    
    def _format_chunks_for_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks for LLM prompt."""
        lines = []
        for i, chunk in enumerate(chunks[:20]):
            text = chunk.get("text", "")[:300]
            source = chunk.get("source_label", "")
            page = chunk.get("page", "")
            lines.append(f"[{i}] ({source}, {page}): {text}")
        return "\n".join(lines)
    
    def _format_tools_for_prompt(self, observations: List[Dict[str, Any]]) -> str:
        """Format tool observations for LLM prompt."""
        if not observations:
            return "(no tool observations)"
        
        lines = []
        for obs in observations:
            tool = obs.get("tool", "unknown")
            params = obs.get("params", {})
            chunks_found = obs.get("chunks_found", 0)
            error = obs.get("error", "")
            
            if error:
                lines.append(f"- {tool}({params}) → ERROR: {error}")
            elif chunks_found == 0:
                lines.append(f"- {tool}({params}) → 0 results")
            else:
                lines.append(f"- {tool}({params}) → {chunks_found} results")
        
        return "\n".join(lines)
    
    def _parse_llm_response(
        self,
        round_number: int,
        data: Dict[str, Any],
        evidence_chunks: List[Dict[str, Any]],
        previous_summary: Optional[RoundSummary],
    ) -> RoundSummary:
        """Parse LLM response into RoundSummary."""
        
        # Parse key findings
        key_findings = []
        for f in data.get("key_findings", []):
            key_findings.append(KeyFinding(
                finding=f.get("finding", ""),
                confidence=f.get("confidence", 0.5),
                finding_type=f.get("finding_type"),
            ))
        
        # Parse actionable leads
        actionable_leads = []
        for l in data.get("actionable_leads", []):
            try:
                priority = LeadPriority(l.get("priority", "medium"))
            except ValueError:
                priority = LeadPriority.MEDIUM
            
            actionable_leads.append(ActionableLead(
                lead_type=l.get("lead_type", "term"),
                target=l.get("target", ""),
                rationale=l.get("rationale", ""),
                priority=priority,
                suggested_tool=l.get("suggested_tool"),
            ))
        
        # Parse decision
        try:
            decision = RoundDecisionType(data.get("decision", "continue"))
        except ValueError:
            decision = RoundDecisionType.CONTINUE
        
        # Calculate metrics
        prev_evidence = previous_summary.evidence_count if previous_summary else 0
        
        return RoundSummary(
            round_number=round_number,
            key_findings=key_findings,
            actionable_leads=actionable_leads,
            decision=decision,
            decision_rationale=data.get("decision_rationale", ""),
            evidence_count=prev_evidence + len(evidence_chunks),
            new_evidence_count=len(evidence_chunks),
            coverage_estimate=data.get("coverage_estimate", 0.0),
            successful_strategies=data.get("successful_strategies", []),
            failed_strategies=data.get("failed_strategies", []),
            information_gaps=data.get("information_gaps", []),
        )
    
    def _fallback_summary(
        self,
        round_number: int,
        evidence_chunks: List[Dict[str, Any]],
        previous_summary: Optional[RoundSummary],
    ) -> RoundSummary:
        """Generate a basic summary without LLM."""
        
        prev_evidence = previous_summary.evidence_count if previous_summary else 0
        new_count = len(evidence_chunks)
        total = prev_evidence + new_count
        
        # Simple decision logic
        if new_count == 0:
            decision = RoundDecisionType.STOP_EXHAUSTED
            rationale = "No new evidence found"
        elif total >= 50:
            decision = RoundDecisionType.STOP_SUFFICIENT
            rationale = "Sufficient evidence collected"
        else:
            decision = RoundDecisionType.CONTINUE
            rationale = "More evidence may be available"
        
        return RoundSummary(
            round_number=round_number,
            decision=decision,
            decision_rationale=rationale,
            evidence_count=total,
            new_evidence_count=new_count,
        )


# =============================================================================
# V7 Configuration
# =============================================================================

@dataclass
class V7Config:
    """Configuration for V7 pipeline."""
    
    # V6 settings
    v6_config: Optional[V6Config] = None
    
    # Citation enforcement
    max_repair_attempts: int = 2  # How many times to try repairing uncited claims
    drop_uncited_claims: bool = True  # If true, remove uncited claims; if false, fail
    
    # Output
    include_evidence_section: bool = True  # Include full evidence quotes in output
    
    verbose: bool = True
    
    # V7 Phase 2: Bundle builder settings
    enable_bundles: bool = True           # Enable evidence bundling
    bundle_config: Optional[BundleBuilderConfig] = None  # Bundle builder config
    
    # Progress callback for streaming updates (inherited from V6 config if not set)
    # Called with (step: str, status: str, message: str, details: dict)
    progress_callback: Optional[Any] = None  # typing.Callable[[str, str, str, Dict[str, Any]], None]


# =============================================================================
# V7 Trace
# =============================================================================

@dataclass
class V7Trace:
    """Complete trace of V7 execution."""
    
    # V6 trace
    v6_trace: Optional[V6Trace] = None
    
    # Citation enforcement
    claims_extracted: int = 0
    claims_valid: int = 0
    claims_unsupported: int = 0
    
    # Stop gate
    stop_gate_passed: bool = False
    stop_gate_reason: str = ""
    repair_attempts: int = 0
    
    # Output
    expanded_summary: Optional[ExpandedSummary] = None
    
    # V7 Phase 2: Evidence bundles
    bundle_collection: Optional[BundleCollection] = None
    
    # Timing
    v7_elapsed_ms: float = 0.0
    total_elapsed_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "v6_trace": self.v6_trace.to_dict() if self.v6_trace else None,
            "claims_extracted": self.claims_extracted,
            "claims_valid": self.claims_valid,
            "claims_unsupported": self.claims_unsupported,
            "stop_gate_passed": self.stop_gate_passed,
            "stop_gate_reason": self.stop_gate_reason,
            "repair_attempts": self.repair_attempts,
            "expanded_summary": self.expanded_summary.to_dict() if self.expanded_summary else None,
            "bundle_collection": self.bundle_collection.to_dict() if self.bundle_collection else None,
            "v7_elapsed_ms": self.v7_elapsed_ms,
            "total_elapsed_ms": self.total_elapsed_ms,
        }


# =============================================================================
# V7 Controller
# =============================================================================

class V7Controller:
    """
    V7 Controller - extends V6 with citation enforcement.
    
    Key behavior:
    - Runs V6 to get an answer with evidence
    - Extracts claims from the answer
    - Validates that every claim has citations
    - Repairs or removes uncited claims
    - Outputs expanded summary with claims & citations
    """
    
    def __init__(
        self,
        config: Optional[V7Config] = None,
    ):
        self.config = config or V7Config()
        
        # Initialize V6 controller with progress callback from V7 config if set
        v6_cfg = self.config.v6_config or V6Config(verbose=self.config.verbose)
        # Ensure progress callback is propagated to V6
        if self.config.progress_callback and not v6_cfg.progress_callback:
            v6_cfg.progress_callback = self.config.progress_callback
        self.v6_controller = V6Controller(config=v6_cfg)
        
        # V7 components
        self.claim_enumerator = ClaimEnumerator(verbose=self.config.verbose)
        self.stop_gate = StopGate(verbose=self.config.verbose)
        self.renderer = ExpandedSummaryRenderer(verbose=self.config.verbose)
        
        # V7 Phase 2: Bundle builder
        self.bundle_builder = None
        if self.config.enable_bundles:
            bundle_cfg = self.config.bundle_config or BundleBuilderConfig(verbose=self.config.verbose)
            self.bundle_builder = BundleBuilder(config=bundle_cfg)
        
        # State
        self.trace = V7Trace()
    
    def _emit_progress(self, step: str, status: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Emit a progress event if callback is configured."""
        # Try V7 config first, then fall back to V6 config
        callback = self.config.progress_callback
        if callback is None and self.config.v6_config:
            callback = self.config.v6_config.progress_callback
        
        if callback:
            try:
                callback(step, status, message, details or {})
            except Exception:
                pass  # Don't let callback errors break the workflow
    
    def run(
        self,
        question: str,
        conn,
    ) -> V7Result:
        """
        Run the V7 pipeline.
        
        Returns:
            V7Result with answer, expanded summary, and validation status
        """
        total_start = time.time()
        
        if self.config.verbose:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"[V7 Controller] Starting with citation enforcement", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
        
        # Step 1: Run V6 pipeline
        if self.config.verbose:
            print(f"\n[V7 Step 1] Running V6 pipeline...", file=sys.stderr)
        
        v6_trace = self.v6_controller.run(question, conn)
        self.trace.v6_trace = v6_trace
        
        # Get V6 results
        answer = v6_trace.final_answer
        v6_claims = v6_trace.final_claims
        bottleneck_spans = v6_trace.bottleneck_spans
        
        if self.config.verbose:
            print(f"\n[V7 Step 2] V6 complete. Answer length: {len(answer)}, "
                  f"Claims: {len(v6_claims)}, Evidence spans: {len(bottleneck_spans)}", 
                  file=sys.stderr)
        
        # V7 Phase 2: Build evidence bundles
        v7_start = time.time()
        
        if self.bundle_builder and bottleneck_spans:
            if self.config.verbose:
                print(f"\n[V7 Phase 2] Building evidence bundles...", file=sys.stderr)
            
            self._emit_progress("bundle_building", "running", "Organizing evidence into bundles...", {
                "spans_count": len(bottleneck_spans),
            })
            
            try:
                # Convert spans to dict format
                span_dicts = self._spans_to_dicts(bottleneck_spans)
                bundle_collection = self.bundle_builder.build_bundles(
                    spans=span_dicts,
                    question=question,
                    conn=conn,
                    round_number=len(v6_trace.rounds),
                )
                self.trace.bundle_collection = bundle_collection
                
                self._emit_progress("bundle_building", "completed", 
                    f"Built {len(bundle_collection.bundles)} evidence bundles", {
                        "bundles_count": len(bundle_collection.bundles),
                        "total_spans": bundle_collection.total_spans(),
                    })
                
                if self.config.verbose:
                    print(f"    Built {len(bundle_collection.bundles)} bundles with "
                          f"{bundle_collection.total_spans()} total spans", file=sys.stderr)
                    for bundle in bundle_collection.bundles[:3]:
                        print(f"      - [{bundle.bundle_id}] {bundle.topic} ({bundle.span_count()} spans)", 
                              file=sys.stderr)
            except Exception as e:
                self._emit_progress("bundle_building", "error", f"Bundle building failed: {e}", {})
                if self.config.verbose:
                    print(f"    Bundle building failed: {e}", file=sys.stderr)
        
        # Step 2: Run citation enforcement
        
        if self.config.verbose:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"[V7 Step 2] CITATION ENFORCEMENT", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
        
        result = self._enforce_citations(
            answer=answer,
            evidence_spans=bottleneck_spans,
            v6_claims=v6_claims,
            task_type=v6_trace.parsed_query.task_type.value if v6_trace.parsed_query else "general",
        )
        
        self.trace.v7_elapsed_ms = (time.time() - v7_start) * 1000
        self.trace.total_elapsed_ms = (time.time() - total_start) * 1000
        
        # Set trace reference
        result.trace = self.trace
        
        # Emit completion event
        self._emit_progress("complete", "completed", "V7 workflow complete", {
            "citation_validation_passed": result.citation_validation_passed,
            "claims_valid": self.trace.claims_valid,
            "claims_dropped": self.trace.claims_unsupported,
            "bundles_count": len(self.trace.bundle_collection.bundles) if self.trace.bundle_collection else 0,
            "elapsed_ms": self.trace.total_elapsed_ms,
        })
        
        if self.config.verbose:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"[V7] COMPLETE", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            print(f"  Stop gate: {'PASSED' if result.citation_validation_passed else 'FAILED'}", 
                  file=sys.stderr)
            print(f"  Valid claims: {self.trace.claims_valid}", file=sys.stderr)
            print(f"  Dropped claims: {self.trace.claims_unsupported}", file=sys.stderr)
            if self.trace.bundle_collection:
                print(f"  Evidence bundles: {len(self.trace.bundle_collection.bundles)}", file=sys.stderr)
            print(f"  V7 overhead: {self.trace.v7_elapsed_ms:.0f}ms", file=sys.stderr)
            print(f"  Total: {self.trace.total_elapsed_ms:.0f}ms", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
        
        return result
    
    def _enforce_citations(
        self,
        answer: str,
        evidence_spans: List[Any],
        v6_claims: List[Dict[str, Any]],
        task_type: str = "general",
    ) -> V7Result:
        """
        Enforce citation requirements on the answer.
        
        Steps:
        1. Enumerate claims from the answer
        2. Validate via stop gate
        3. Repair or remove uncited claims
        4. Render expanded summary
        """
        
        # Convert evidence spans to dict format and build span_id set
        evidence_dicts = self._spans_to_dicts(evidence_spans)
        evidence_span_ids = {d.get("span_id", f"sp_{i}") for i, d in enumerate(evidence_dicts)}
        
        if self.config.verbose:
            print(f"\n  [Citation Enforcement] {len(evidence_span_ids)} evidence spans available", 
                  file=sys.stderr)
        
        # Step 2.1: Enumerate claims
        if self.config.verbose:
            print(f"\n  [Step 2.1] Enumerating claims from answer...", file=sys.stderr)
        
        self._emit_progress("claim_enumeration", "running", "Extracting atomic claims from answer...", {
            "answer_length": len(answer),
            "evidence_count": len(evidence_dicts),
        })
        
        valid_claims, unsupported_claims = self.claim_enumerator.enumerate_claims(
            answer=answer,
            evidence_spans=evidence_dicts,
        )
        
        self.trace.claims_extracted = len(valid_claims) + len(unsupported_claims)
        self.trace.claims_valid = len(valid_claims)
        self.trace.claims_unsupported = len(unsupported_claims)
        
        self._emit_progress("claim_enumeration", "completed", 
            f"Extracted {len(valid_claims)} claims with citations, {len(unsupported_claims)} unsupported", {
                "claims_extracted": self.trace.claims_extracted,
                "claims_valid": len(valid_claims),
                "claims_unsupported": len(unsupported_claims),
            })
        
        # Step 2.2: Validate via stop gate
        if self.config.verbose:
            print(f"\n  [Step 2.2] Running stop gate validation...", file=sys.stderr)
        
        self._emit_progress("stop_gate", "running", "Validating citation requirements...", {
            "claims_to_validate": len(valid_claims),
        })
        
        gate_result = self.stop_gate.validate(
            claims=valid_claims,
            evidence_span_ids=evidence_span_ids,
            task_type=task_type,
        )
        
        self.trace.stop_gate_passed = gate_result.can_stop
        self.trace.stop_gate_reason = gate_result.reason
        
        self._emit_progress("stop_gate", "completed", 
            f"{'✓ Passed' if gate_result.can_stop else '✗ Failed'}: {gate_result.reason}", {
                "passed": gate_result.can_stop,
                "reason": gate_result.reason,
                "invalid_claims": len(gate_result.invalid_claims) if gate_result.invalid_claims else 0,
            })
        
        # Step 2.3: Handle validation failure
        if not gate_result.can_stop:
            if self.config.verbose:
                print(f"\n  [Step 2.3] Stop gate FAILED: {gate_result.reason}", file=sys.stderr)
            
            if self.config.drop_uncited_claims:
                # Remove invalid claims
                valid_claims = [c for c in valid_claims if c.is_valid() 
                               and all(cid in evidence_span_ids for cid in c.citations)]
                
                # Move removed claims to unsupported
                for claim in gate_result.invalid_claims:
                    unsupported_claims.append(claim.claim_text)
                
                self.trace.claims_valid = len(valid_claims)
                self.trace.claims_unsupported = len(unsupported_claims)
                
                if self.config.verbose:
                    print(f"    Dropped {len(gate_result.invalid_claims)} uncited claims", 
                          file=sys.stderr)
            else:
                # Return failure without expanded summary
                return V7Result(
                    answer=answer,
                    expanded_summary=None,
                    claims=[c.to_dict() for c in valid_claims],
                    responsiveness_status=self.trace.v6_trace.stopped_reason if self.trace.v6_trace else "",
                    all_claims_cited=False,
                    citation_validation_passed=False,
                )
        
        # Step 2.4: Render expanded summary
        if self.config.verbose:
            print(f"\n  [Step 2.4] Rendering expanded summary...", file=sys.stderr)
        
        self._emit_progress("expanded_summary", "running", "Generating structured summary with citations...", {
            "claims_count": len(valid_claims),
        })
        
        expanded_summary = self.renderer.render(
            short_answer=answer,
            claims=valid_claims,
            evidence_spans=evidence_dicts,
            unsupported_claims=unsupported_claims,
        )
        
        self.trace.expanded_summary = expanded_summary
        
        self._emit_progress("expanded_summary", "completed", "Summary generated with citation links", {
            "claims_included": len(valid_claims),
            "unsupported_claims_dropped": len(unsupported_claims),
        })
        
        # All claims now have valid citations (after dropping invalid ones)
        all_cited = all(c.is_valid() for c in valid_claims)
        
        return V7Result(
            answer=answer,
            expanded_summary=expanded_summary,
            claims=[c.to_dict() for c in valid_claims],
            members_identified=list(self.v6_controller.all_members_found),
            responsiveness_status=self.trace.v6_trace.stopped_reason if self.trace.v6_trace else "",
            all_claims_cited=all_cited,
            citation_validation_passed=gate_result.can_stop or all_cited,
        )
    
    def _spans_to_dicts(self, spans: List[Any]) -> List[Dict[str, Any]]:
        """Convert evidence spans to dict format."""
        dicts = []
        for i, span in enumerate(spans):
            if hasattr(span, 'to_dict'):
                d = span.to_dict()
            elif hasattr(span, '__dict__'):
                d = {
                    "span_id": getattr(span, 'span_id', f"sp_{i}"),
                    "text": getattr(span, 'span_text', getattr(span, 'text', str(span))),
                    "source_label": getattr(span, 'source_label', ''),
                    "page": getattr(span, 'page', ''),
                    "claim_supported": getattr(span, 'claim_supported', ''),
                }
            elif isinstance(span, dict):
                d = span
                if "span_id" not in d:
                    d["span_id"] = f"sp_{i}"
            else:
                d = {
                    "span_id": f"sp_{i}",
                    "text": str(span),
                }
            
            dicts.append(d)
        
        return dicts


# =============================================================================
# Convenience function
# =============================================================================

def run_v7(
    question: str,
    conn,
    verbose: bool = True,
    drop_uncited_claims: bool = True,
) -> V7Result:
    """
    Convenience function to run V7 pipeline.
    
    Args:
        question: User question
        conn: Database connection
        verbose: Enable verbose logging
        drop_uncited_claims: If True, drop claims without citations; if False, fail
    
    Returns:
        V7Result with answer, expanded summary, and validation status
    """
    config = V7Config(
        v6_config=V6Config(verbose=verbose),
        drop_uncited_claims=drop_uncited_claims,
        verbose=verbose,
    )
    
    controller = V7Controller(config=config)
    return controller.run(question, conn)
