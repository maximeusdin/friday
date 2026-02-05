"""
V6 Evidence Bottleneck - Force convergence before synthesis

The SINGLE MOST IMPORTANT "make it smart" step:
- After retrieval, grade spans to keep 30-50 MAX
- Claim extraction / answer writing sees ONLY those
- This prevents "164 chunks → thousands of claims"

The bottleneck is a HARD gate - synthesis cannot proceed without it.

GRADING MODES:
- "tournament" (default): Pairwise comparisons to find best spans. More robust
  to LLM score drift and better at relative ranking when all spans are "somewhat relevant".
- "absolute": Each span graded independently on 0-10 scale. Original behavior.
"""
import os
import json
import sys
import time
import random
from typing import List, Dict, Any, Optional, Tuple, Literal, Set
from dataclasses import dataclass, field

from retrieval.agent.v6_query_parser import ParsedQuery, TaskType


# =============================================================================
# Configuration
# =============================================================================

BOTTLENECK_MODEL = "gpt-3.5-turbo"  # Faster than gpt-4o-mini, good enough for filtering
BOTTLENECK_BATCH_SIZE = 15
DEFAULT_BOTTLENECK_SIZE = 40  # Max spans after bottleneck

# Tournament config (Elo-style)
TOURNAMENT_MODEL = "gpt-5-nano"
TOURNAMENT_BATCH_SIZE = 8  # Matchups per API call when batching enabled
TOURNAMENT_BATCH_BY_DEFAULT = False  # If True, batch matchups; if False, single call per matchup

# Elo rating config
ELO_INITIAL_RATING = 1000.0
ELO_K_FACTOR = 32.0  # How much ratings change per match
ELO_MATCHES_PER_SPAN = 4  # Each span participates in ~this many matchups

# Grading modes
GradingMode = Literal["tournament", "absolute"]
DEFAULT_GRADING_MODE: GradingMode = "tournament"


# =============================================================================
# Bottleneck Span
# =============================================================================

@dataclass
class BottleneckSpan:
    """A span that passed the evidence bottleneck."""
    
    span_id: str
    chunk_id: int
    doc_id: Optional[int]
    page: Optional[str]
    source_label: str
    
    # The actual quotable text
    span_text: str
    
    # Grading from bottleneck
    relevance_score: float  # 0-10
    claim_supported: str  # What this span supports
    is_directly_responsive: bool  # Does it DIRECTLY answer the question?
    
    # For roster queries: does this identify a member?
    identifies_member: bool = False
    member_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "page": self.page,
            "source_label": self.source_label,
            "span_text": self.span_text,
            "relevance_score": self.relevance_score,
            "claim_supported": self.claim_supported,
            "is_directly_responsive": self.is_directly_responsive,
            "identifies_member": self.identifies_member,
            "member_name": self.member_name,
        }


# =============================================================================
# Bottleneck Result
# =============================================================================

@dataclass
class BottleneckResult:
    """Result of the evidence bottleneck."""
    
    spans: List[BottleneckSpan] = field(default_factory=list)
    
    # Stats
    chunks_input: int = 0
    spans_extracted: int = 0
    spans_passed: int = 0
    
    # For roster queries
    members_identified: List[str] = field(default_factory=list)
    
    # Timing
    elapsed_ms: float = 0.0
    
    def get_synthesis_context(self) -> str:
        """Get the context for synthesis (ONLY these spans)."""
        lines = []
        for i, span in enumerate(self.spans):
            lines.append(f"[{i}] (chunk:{span.chunk_id}, {span.source_label}, p.{span.page})")
            lines.append(f'    "{span.span_text}"')
            lines.append(f"    Supports: {span.claim_supported}")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "spans": [s.to_dict() for s in self.spans],
            "chunks_input": self.chunks_input,
            "spans_extracted": self.spans_extracted,
            "spans_passed": self.spans_passed,
            "members_identified": self.members_identified,
            "elapsed_ms": self.elapsed_ms,
        }


# =============================================================================
# Bottleneck Prompt
# =============================================================================

BOTTLENECK_SYSTEM_PROMPT = """You are an evidence quality gate. Your job is to identify the BEST spans for answering a specific question.

Be STRICT. Only spans with direct, quotable evidence should pass.

For roster queries ("who were members"):
- PASS: "X was a member of Y network" → identifies_member=true
- PASS: "X reported to Y about Z" (implies membership)
- FAIL: "The network operated in Washington" (no member named)
- FAIL: "HUAC investigated espionage" (too general)

For each span, output:
- pass: true if this is quality evidence
- score: 0-10 relevance
- claim: what specific claim does this support?
- responsive: does it DIRECTLY answer the question?
- identifies_member: (for roster) does it name a member?
- member_name: (for roster) who is named as member?"""


def build_bottleneck_prompt(
    question: str,
    task_type: TaskType,
    chunks: List[Dict[str, Any]],
) -> str:
    """Build prompt for bottleneck grading."""
    
    task_guidance = ""
    if task_type == TaskType.ROSTER_ENUMERATION:
        task_guidance = """
TASK: Roster enumeration - identify MEMBERS
Only pass spans that:
- Explicitly name someone as a member
- Describe someone's role in the network
- Show evidence of someone's involvement
Do NOT pass spans that just mention the network generally."""
    
    chunks_section = []
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")[:1000]
        source = chunk.get("source_label", "")
        page = chunk.get("page", "")
        chunks_section.append(f"""
CHUNK {i} ({source}, p.{page}):
"{text}"
""")
    
    chunks_text = "\n---\n".join(chunks_section)
    
    return f"""QUESTION: {question}
{task_guidance}

CHUNKS TO EVALUATE:
{chunks_text}

For EACH chunk, output whether it should PASS the bottleneck.
Be strict - only high-quality, directly responsive evidence should pass.

Output JSON:
{{
  "evaluations": [
    {{
      "chunk_index": 0,
      "pass": true,
      "score": 8,
      "claim": "X was a member of Y",
      "responsive": true,
      "identifies_member": true,
      "member_name": "Harry White"
    }},
    {{
      "chunk_index": 1,
      "pass": false,
      "score": 2,
      "claim": "",
      "responsive": false,
      "identifies_member": false,
      "member_name": ""
    }}
  ]
}}"""


# =============================================================================
# Tournament Mode Prompts (True Head-to-Head)
# =============================================================================

TOURNAMENT_SYSTEM_PROMPT = """You are a research evidence judge. Pick which span is MORE relevant for answering the question.

You must pick exactly ONE winner - A or B. No ties allowed.

Consider:
- Does the span directly answer the question?
- Does it name specific people, dates, or facts?
- Is it quotable evidence vs vague background?

Be decisive."""


def build_single_matchup_prompt(
    question: str,
    task_type: TaskType,
    span_a: Dict[str, Any],
    span_b: Dict[str, Any],
) -> str:
    """Build prompt for a single 1v1 matchup."""
    
    task_guidance = ""
    if task_type == TaskType.ROSTER_ENUMERATION:
        task_guidance = "TASK: Finding network MEMBERS. Prefer spans that name specific members."
    
    text_a = span_a.get("text", "")[:800]
    text_b = span_b.get("text", "")[:800]
    source_a = span_a.get("source_label", "")
    source_b = span_b.get("source_label", "")
    page_a = span_a.get("page", "")
    page_b = span_b.get("page", "")
    
    return f"""QUESTION: {question}
{task_guidance}

Which span is MORE relevant for answering this question?

SPAN A ({source_a}, p.{page_a}):
"{text_a}"

SPAN B ({source_b}, p.{page_b}):
"{text_b}"

Pick A or B. Output JSON:
{{"winner": "A", "reason": "brief reason", "identifies_member": true, "member_name": "Name if applicable"}}"""


def build_batched_matchups_prompt(
    question: str,
    task_type: TaskType,
    matchups: List[Tuple[Dict[str, Any], Dict[str, Any]]],
) -> str:
    """Build prompt for multiple matchups in one call (faster but less focused)."""
    
    task_guidance = ""
    if task_type == TaskType.ROSTER_ENUMERATION:
        task_guidance = "TASK: Finding network MEMBERS. Prefer spans that name specific members."
    
    matchup_sections = []
    for i, (span_a, span_b) in enumerate(matchups):
        text_a = span_a.get("text", "")[:500]
        text_b = span_b.get("text", "")[:500]
        source_a = span_a.get("source_label", "")
        source_b = span_b.get("source_label", "")
        
        matchup_sections.append(f"""MATCHUP {i}:
A ({source_a}): "{text_a}"
B ({source_b}): "{text_b}"
""")
    
    matchups_text = "\n---\n".join(matchup_sections)
    
    return f"""QUESTION: {question}
{task_guidance}

Pick the winner (A or B) for EACH matchup:

{matchups_text}

Output JSON with a decision for each matchup:
{{"decisions": [
  {{"matchup": 0, "winner": "A", "reason": "...", "identifies_member": true, "member_name": "..."}},
  {{"matchup": 1, "winner": "B", "reason": "...", "identifies_member": false, "member_name": ""}}
]}}"""


# =============================================================================
# Evidence Bottleneck
# =============================================================================

class EvidenceBottleneck:
    """
    The HARD gate that forces convergence.
    
    Synthesis CANNOT proceed without passing through this bottleneck.
    Only 30-50 spans maximum emerge from the other side.
    
    GRADING MODES:
    - "tournament" (default): Pairwise comparisons to find best spans.
      More robust to LLM score drift. Better when all spans are "somewhat relevant".
    - "absolute": Each span graded independently on 0-10 scale.
    
    TOURNAMENT BATCHING:
    - batch_tournament=False (default): One API call per matchup. More accurate, slower.
    - batch_tournament=True: Multiple matchups per API call. Faster, may be less focused.
    """
    
    def __init__(
        self,
        max_spans: int = DEFAULT_BOTTLENECK_SIZE,
        model: str = BOTTLENECK_MODEL,
        grading_mode: GradingMode = DEFAULT_GRADING_MODE,
        batch_tournament: bool = TOURNAMENT_BATCH_BY_DEFAULT,
        verbose: bool = True,
    ):
        self.max_spans = max_spans
        self.model = model
        self.grading_mode = grading_mode
        self.batch_tournament = batch_tournament
        self.verbose = verbose
    
    def filter(
        self,
        chunks: List[Dict[str, Any]],
        parsed_query: ParsedQuery,
        conn = None,
    ) -> BottleneckResult:
        """
        Filter chunks through the bottleneck.
        
        This is the HARD gate - synthesis sees ONLY what passes.
        
        Args:
            chunks: Raw chunks from retrieval (can be 100-300)
            parsed_query: The parsed query with task type
            conn: Database connection (for loading chunk text if needed)
        
        Returns:
            BottleneckResult with max 30-50 spans
        """
        start = time.time()
        result = BottleneckResult(chunks_input=len(chunks))
        
        if not chunks:
            return result
        
        # Dispatch to appropriate grading mode
        if self.grading_mode == "tournament":
            top_spans = self._tournament_filter(chunks, parsed_query, result)
        else:
            top_spans = self._absolute_filter(chunks, parsed_query, result)
        
        # Convert to BottleneckSpan objects
        members_seen = set()
        for g in top_spans:
            chunk = chunks[g["chunk_index"]] if g["chunk_index"] < len(chunks) else {}
            
            span = BottleneckSpan(
                span_id=f"bs_{chunk.get('id', g['chunk_index'])}",
                chunk_id=chunk.get("id", 0),
                doc_id=chunk.get("doc_id"),
                page=chunk.get("page", ""),
                source_label=chunk.get("source_label", ""),
                span_text=chunk.get("text", "")[:800],
                relevance_score=g["score"],
                claim_supported=g["claim"],
                is_directly_responsive=g["responsive"],
                identifies_member=g.get("identifies_member", False),
                member_name=g.get("member_name", ""),
            )
            result.spans.append(span)
            
            if span.identifies_member and span.member_name:
                members_seen.add(span.member_name)
        
        result.spans_passed = len(result.spans)
        result.members_identified = list(members_seen)
        result.elapsed_ms = (time.time() - start) * 1000
        
        if self.verbose:
            print(f"    ┌─────────────────────────────────────────────────────────────", file=sys.stderr)
            print(f"    │ BOTTLENECK RESULT (HARD GATE)", file=sys.stderr)
            print(f"    │ Input: {result.chunks_input} chunks", file=sys.stderr)
            print(f"    │ Graded: {result.spans_extracted} spans", file=sys.stderr)
            print(f"    │ Passed: {result.spans_passed} spans (max allowed: {self.max_spans})", file=sys.stderr)
            print(f"    │ Rejected: {result.spans_extracted - result.spans_passed} spans", file=sys.stderr)
            if result.members_identified:
                print(f"    │ Members identified: {result.members_identified[:10]}", file=sys.stderr)
            print(f"    │", file=sys.stderr)
            print(f"    │ TOP PASSING SPANS:", file=sys.stderr)
            for i, span in enumerate(result.spans[:5]):
                print(f"    │   [{i}] score={span.relevance_score:.1f}, member={span.member_name or 'N/A'}", file=sys.stderr)
                print(f"    │       \"{span.span_text[:100]}...\"", file=sys.stderr)
            if len(result.spans) > 5:
                print(f"    │   ... and {len(result.spans) - 5} more spans", file=sys.stderr)
            print(f"    └─────────────────────────────────────────────────────────────", file=sys.stderr)
        
        return result
    
    def _absolute_filter(
        self,
        chunks: List[Dict[str, Any]],
        parsed_query: ParsedQuery,
        result: BottleneckResult,
    ) -> List[Dict[str, Any]]:
        """
        Absolute grading mode: each span is graded independently on 0-10 scale.
        Original bottleneck behavior.
        """
        if self.verbose:
            print(f"  [Bottleneck] Filtering {len(chunks)} chunks -> max {self.max_spans} spans", 
                  file=sys.stderr)
            print(f"    Mode: ABSOLUTE (each span graded independently)", file=sys.stderr)
            print(f"    What this does: Sends chunks to {self.model} in batches of {BOTTLENECK_BATCH_SIZE}", file=sys.stderr)
            print(f"    LLM grades each chunk for relevance (0-10) and decides pass/fail", file=sys.stderr)
            print(f"    Only top {self.max_spans} passing spans proceed to synthesis", file=sys.stderr)
        
        # Grade in batches (preserve original chunk indices)
        all_graded = []
        total_batches = (len(chunks) + BOTTLENECK_BATCH_SIZE - 1) // BOTTLENECK_BATCH_SIZE
        
        for batch_num, batch_start in enumerate(range(0, len(chunks), BOTTLENECK_BATCH_SIZE)):
            batch = chunks[batch_start:batch_start + BOTTLENECK_BATCH_SIZE]
            
            if self.verbose:
                print(f"    [Batch {batch_num + 1}/{total_batches}] Grading {len(batch)} chunks (LLM call)...", 
                      file=sys.stderr, end="", flush=True)
            
            batch_start_time = time.time()
            graded = self._grade_batch(batch, parsed_query)
            batch_elapsed = (time.time() - batch_start_time) * 1000
            
            if self.verbose:
                passing_in_batch = len([g for g in graded if g.get("pass")])
                print(f" done ({batch_elapsed:.0f}ms, {passing_in_batch}/{len(batch)} passed)", file=sys.stderr)
            
            # Adjust chunk indices to global indices
            for g in graded:
                g["chunk_index"] = batch_start + g["chunk_index"]
            all_graded.extend(graded)
        
        result.spans_extracted = len(all_graded)
        
        # Filter to only passing spans
        passing = [g for g in all_graded if g["pass"]]
        
        # Sort by score descending
        passing.sort(key=lambda x: x["score"], reverse=True)
        
        # Take top N
        return passing[:self.max_spans]
    
    def _tournament_filter(
        self,
        chunks: List[Dict[str, Any]],
        parsed_query: ParsedQuery,
        result: BottleneckResult,
    ) -> List[Dict[str, Any]]:
        """
        Tournament grading mode: Elo-style rating system.
        
        Instead of elimination, all spans get a rating that updates based on
        head-to-head comparisons. Top K by rating are selected.
        
        More robust than elimination because:
        - One bad matchup doesn't knock out a good span
        - Rating emerges from multiple comparisons
        - No seeding/ordering needed
        
        Process:
        1. Initialize all spans with rating 1000
        2. Run random matchups (each span faces ~4 opponents)
        3. Update Elo ratings after each matchup
        4. Take top max_spans by final rating
        """
        if self.verbose:
            print(f"  [Bottleneck] Filtering {len(chunks)} chunks -> max {self.max_spans} spans", 
                  file=sys.stderr)
            print(f"    Mode: TOURNAMENT (Elo-style ratings)", file=sys.stderr)
            batching_status = "BATCHED" if self.batch_tournament else "SINGLE"
            print(f"    API mode: {batching_status} ({TOURNAMENT_BATCH_SIZE}/call)" if self.batch_tournament 
                  else f"    API mode: {batching_status} (1 matchup/call)", file=sys.stderr)
            print(f"    Top {self.max_spans} by final rating are selected", file=sys.stderr)
        
        # Initialize all spans with same Elo rating
        spans = [
            {
                "chunk_index": i, 
                "chunk": chunks[i],
                "rating": ELO_INITIAL_RATING,
                "matches": 0,
                "metadata": {}  # Will store member info from matchups
            } 
            for i in range(len(chunks))
        ]
        
        # Calculate total matchups needed
        # Each span should participate in ~ELO_MATCHES_PER_SPAN matches
        # Each matchup involves 2 spans, so total matchups = (N * matches_per_span) / 2
        total_matchups = max(len(spans), (len(spans) * ELO_MATCHES_PER_SPAN) // 2)
        
        if self.verbose:
            print(f"    Running ~{total_matchups} matchups ({ELO_MATCHES_PER_SPAN} per span avg)", 
                  file=sys.stderr)
        
        # Generate all matchups upfront (random pairing)
        all_matchups = []
        all_matchup_indices = []
        
        for _ in range(total_matchups):
            # Pick two different spans, preferring those with fewer matches
            # This ensures more even coverage
            indices = list(range(len(spans)))
            # Weight by inverse of match count (prefer less-matched spans)
            weights = [1.0 / (spans[i]["matches"] + 1) for i in indices]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Sample two distinct indices
            idx_a = random.choices(indices, weights=weights, k=1)[0]
            remaining_indices = [i for i in indices if i != idx_a]
            remaining_weights = [weights[i] for i in range(len(indices)) if indices[i] != idx_a]
            if remaining_weights:
                total_remaining = sum(remaining_weights)
                remaining_weights = [w / total_remaining for w in remaining_weights]
                idx_b = random.choices(remaining_indices, weights=remaining_weights, k=1)[0]
            else:
                idx_b = (idx_a + 1) % len(spans)
            
            all_matchups.append((spans[idx_a]["chunk"], spans[idx_b]["chunk"]))
            all_matchup_indices.append((idx_a, idx_b))
        
        # Execute matchups - either batched or single depending on config
        start_time = time.time()
        
        if self.batch_tournament:
            # BATCHED MODE: Group matchups and send multiple per API call
            matchups_processed = 0
            for batch_start in range(0, len(all_matchups), TOURNAMENT_BATCH_SIZE):
                batch_end = min(batch_start + TOURNAMENT_BATCH_SIZE, len(all_matchups))
                batch_matchups = all_matchups[batch_start:batch_end]
                batch_indices = all_matchup_indices[batch_start:batch_end]
                
                # Run batch of matchups in single API call
                batch_decisions = self._run_batched_matchups(batch_matchups, parsed_query)
                
                # Apply Elo updates for each matchup in batch
                for i, ((span_a, span_b), (idx_a, idx_b)) in enumerate(zip(batch_matchups, batch_indices)):
                    decision = batch_decisions[i] if i < len(batch_decisions) else {"winner": "A"}
                    winner_label = decision.get("winner", "A").upper()
                    
                    # Get current ratings
                    rating_a = spans[idx_a]["rating"]
                    rating_b = spans[idx_b]["rating"]
                    
                    # Calculate expected scores (Elo formula)
                    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
                    expected_b = 1.0 - expected_a
                    
                    # Actual scores (1 for win, 0 for loss)
                    if winner_label == "A":
                        score_a, score_b = 1.0, 0.0
                        winner = spans[idx_a]
                    else:
                        score_a, score_b = 0.0, 1.0
                        winner = spans[idx_b]
                    
                    # Update ratings
                    spans[idx_a]["rating"] += ELO_K_FACTOR * (score_a - expected_a)
                    spans[idx_b]["rating"] += ELO_K_FACTOR * (score_b - expected_b)
                    
                    # Track match counts
                    spans[idx_a]["matches"] += 1
                    spans[idx_b]["matches"] += 1
                    
                    # Store metadata from the winner
                    if decision.get("identifies_member"):
                        winner["metadata"]["identifies_member"] = True
                        winner["metadata"]["member_name"] = decision.get("member_name", "")
                    winner["metadata"]["last_reason"] = decision.get("reason", "")
                    
                    matchups_processed += 1
                
                # Progress logging per batch
                if self.verbose:
                    elapsed_total = time.time() - start_time
                    avg_ms = (elapsed_total * 1000) / matchups_processed if matchups_processed > 0 else 0
                    remaining = len(all_matchups) - matchups_processed
                    eta_seconds = (remaining * avg_ms) / 1000 if avg_ms > 0 else 0
                    
                    sorted_spans = sorted(spans, key=lambda x: x["rating"], reverse=True)
                    leader_rating = sorted_spans[0]["rating"]
                    
                    print(f"    [Batch {batch_start//TOURNAMENT_BATCH_SIZE + 1}] "
                          f"{matchups_processed}/{len(all_matchups)} matchups | "
                          f"Leader: {leader_rating:.0f} | "
                          f"ETA: {eta_seconds:.0f}s", 
                          file=sys.stderr)
        
        else:
            # SINGLE MODE: One API call per matchup (more accurate, slower)
            for matchup_num, ((span_a, span_b), (idx_a, idx_b)) in enumerate(zip(all_matchups, all_matchup_indices)):
                
                # Run single 1v1 matchup
                decision = self._run_single_matchup(span_a, span_b, parsed_query)
                
                winner_label = decision.get("winner", "A").upper()
                
                # Get current ratings
                rating_a = spans[idx_a]["rating"]
                rating_b = spans[idx_b]["rating"]
                
                # Calculate expected scores (Elo formula)
                expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
                expected_b = 1.0 - expected_a
                
                # Actual scores (1 for win, 0 for loss)
                if winner_label == "A":
                    score_a, score_b = 1.0, 0.0
                    winner = spans[idx_a]
                else:
                    score_a, score_b = 0.0, 1.0
                    winner = spans[idx_b]
                
                # Update ratings
                old_winner_rating = winner["rating"]
                spans[idx_a]["rating"] += ELO_K_FACTOR * (score_a - expected_a)
                spans[idx_b]["rating"] += ELO_K_FACTOR * (score_b - expected_b)
                
                # Track match counts
                spans[idx_a]["matches"] += 1
                spans[idx_b]["matches"] += 1
                
                # Store metadata from the winner
                if decision.get("identifies_member"):
                    winner["metadata"]["identifies_member"] = True
                    winner["metadata"]["member_name"] = decision.get("member_name", "")
                winner["metadata"]["last_reason"] = decision.get("reason", "")
                
                # Progress logging - every 10 matchups or at the end
                if self.verbose and ((matchup_num + 1) % 10 == 0 or matchup_num == len(all_matchups) - 1):
                    elapsed_total = time.time() - start_time
                    avg_ms = (elapsed_total * 1000) / (matchup_num + 1)
                    remaining = len(all_matchups) - matchup_num - 1
                    eta_seconds = (remaining * avg_ms) / 1000
                    
                    # Find current leader
                    sorted_spans = sorted(spans, key=lambda x: x["rating"], reverse=True)
                    leader = sorted_spans[0]
                    leader_rating = leader["rating"]
                    
                    print(f"    [{matchup_num + 1}/{len(all_matchups)}] "
                          f"Winner: {winner_label} ({old_winner_rating:.0f}→{winner['rating']:.0f}) | "
                          f"Leader: {leader_rating:.0f} | "
                          f"ETA: {eta_seconds:.0f}s", 
                          file=sys.stderr)
        
        result.spans_extracted = len(spans)
        
        # Sort by rating (highest first) and take top max_spans
        spans.sort(key=lambda x: x["rating"], reverse=True)
        
        if self.verbose:
            top_5 = [round(s["rating"]) for s in spans[:5]]
            bottom_5 = [round(s["rating"]) for s in spans[-5:]]
            print(f"    Top 5 ratings: {top_5}", file=sys.stderr)
            print(f"    Bottom 5 ratings: {bottom_5}", file=sys.stderr)
        
        top_spans = []
        for i, s in enumerate(spans[:self.max_spans]):
            # Convert Elo rating to 0-10 scale for compatibility
            # Map rating range to score (higher rating = higher score)
            normalized_score = min(10.0, max(0.0, (s["rating"] - 900) / 20.0))
            
            top_spans.append({
                "chunk_index": s["chunk_index"],
                "pass": True,
                "score": normalized_score,
                "claim": s["metadata"].get("last_reason", ""),
                "responsive": True,
                "identifies_member": s["metadata"].get("identifies_member", False),
                "member_name": s["metadata"].get("member_name", ""),
            })
        
        return top_spans
    
    def _run_single_matchup(
        self,
        span_a: Dict[str, Any],
        span_b: Dict[str, Any],
        parsed_query: ParsedQuery,
    ) -> Dict[str, Any]:
        """Run a single 1v1 matchup."""
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return self._fallback_single_matchup(span_a, span_b)
        
        prompt = build_single_matchup_prompt(
            question=parsed_query.original_query,
            task_type=parsed_query.task_type,
            span_a=span_a,
            span_b=span_b,
        )
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=TOURNAMENT_MODEL,  # gpt-5-nano
                messages=[
                    {"role": "system", "content": TOURNAMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=200,
            )
            
            content = response.choices[0].message.content
            if not content:
                return self._fallback_single_matchup(span_a, span_b)
            
            return json.loads(content)
            
        except Exception as e:
            if self.verbose:
                print(f"    [Matchup] Error: {e}", file=sys.stderr)
            return self._fallback_single_matchup(span_a, span_b)
    
    def _fallback_single_matchup(
        self, 
        span_a: Dict[str, Any],
        span_b: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fallback: use heuristics to decide a single matchup."""
        text_a = span_a.get("text", "").lower()
        text_b = span_b.get("text", "").lower()
        
        # Simple heuristics
        score_a = sum([
            3 if "member" in text_a else 0,
            2 if "agent" in text_a else 0,
            2 if "recruited" in text_a else 0,
            1 if "source" in text_a else 0,
        ])
        score_b = sum([
            3 if "member" in text_b else 0,
            2 if "agent" in text_b else 0,
            2 if "recruited" in text_b else 0,
            1 if "source" in text_b else 0,
        ])
        
        winner = "A" if score_a >= score_b else "B"
        winner_text = text_a if winner == "A" else text_b
        
        return {
            "winner": winner,
            "reason": "heuristic",
            "identifies_member": "member" in winner_text,
            "member_name": "",
        }
    
    def _run_batched_matchups(
        self,
        matchups: List[Tuple[Dict[str, Any], Dict[str, Any]]],
        parsed_query: ParsedQuery,
    ) -> List[Dict[str, Any]]:
        """
        Run multiple matchups in a single API call (faster, but may be less focused).
        
        Returns a list of decisions, one per matchup in the input order.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Fallback: use heuristics for each matchup
            return [
                self._fallback_single_matchup(span_a, span_b)
                for span_a, span_b in matchups
            ]
        
        prompt = build_batched_matchups_prompt(
            question=parsed_query.original_query,
            task_type=parsed_query.task_type,
            matchups=matchups,
        )
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=TOURNAMENT_MODEL,  # gpt-5-nano
                messages=[
                    {"role": "system", "content": TOURNAMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=1000,  # More tokens for batched response
            )
            
            content = response.choices[0].message.content
            if not content:
                return [
                    self._fallback_single_matchup(span_a, span_b)
                    for span_a, span_b in matchups
                ]
            
            data = json.loads(content)
            decisions = data.get("decisions", [])
            
            # Ensure we have a decision for each matchup
            result = []
            for i, (span_a, span_b) in enumerate(matchups):
                if i < len(decisions):
                    result.append(decisions[i])
                else:
                    # Fallback for missing decisions
                    result.append(self._fallback_single_matchup(span_a, span_b))
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"    [Batched Matchup] Error: {e}", file=sys.stderr)
            return [
                self._fallback_single_matchup(span_a, span_b)
                for span_a, span_b in matchups
            ]
    
    def _grade_batch(
        self,
        batch: List[Dict[str, Any]],
        parsed_query: ParsedQuery,
    ) -> List[Dict[str, Any]]:
        """Grade a batch of chunks (for absolute mode)."""
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return self._fallback_grade(batch)
        
        prompt = build_bottleneck_prompt(
            question=parsed_query.original_query,
            task_type=parsed_query.task_type,
            chunks=batch,
        )
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": BOTTLENECK_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2000,
            )
            
            content = response.choices[0].message.content
            if not content:
                return self._fallback_grade(batch)
            
            data = json.loads(content)
            return data.get("evaluations", [])
            
        except Exception as e:
            if self.verbose:
                print(f"    [Bottleneck] Error: {e}", file=sys.stderr)
            return self._fallback_grade(batch)
    
    def _fallback_grade(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple fallback grading."""
        results = []
        for i, chunk in enumerate(batch):
            text = chunk.get("text", "").lower()
            # Simple heuristic: pass if contains key indicators
            has_member_indicator = any(w in text for w in ["member", "agent", "source", "recruited"])
            results.append({
                "chunk_index": i,
                "pass": has_member_indicator,
                "score": 5.0 if has_member_indicator else 2.0,
                "claim": "",
                "responsive": has_member_indicator,
                "identifies_member": False,
                "member_name": "",
            })
        return results


# =============================================================================
# Convenience function
# =============================================================================

def apply_bottleneck(
    chunks: List[Dict[str, Any]],
    parsed_query: ParsedQuery,
    max_spans: int = DEFAULT_BOTTLENECK_SIZE,
    grading_mode: GradingMode = DEFAULT_GRADING_MODE,
    verbose: bool = True,
) -> BottleneckResult:
    """
    Apply the evidence bottleneck to chunks.
    
    This MUST be called before any synthesis/claim extraction.
    
    Args:
        chunks: Raw chunks from retrieval
        parsed_query: Parsed query with task type
        max_spans: Maximum spans to keep (default 40)
        grading_mode: "tournament" (default) or "absolute"
            - tournament: Pairwise comparisons, more robust to LLM score drift
            - absolute: Each span graded independently 0-10
        verbose: Print progress
    
    Returns:
        BottleneckResult with filtered spans
    """
    bottleneck = EvidenceBottleneck(
        max_spans=max_spans, 
        grading_mode=grading_mode,
        verbose=verbose,
    )
    return bottleneck.filter(chunks, parsed_query)
