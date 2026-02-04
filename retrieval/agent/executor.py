"""
V3 Tool Executor - Deterministic execution of tool steps with logging.

The executor:
- Takes a plan with ordered tool steps
- Executes each step via the tool registry
- Logs all calls to a trace
- Merges results across steps
"""

import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set

from retrieval.agent.tools import (
    ToolResult,
    ToolSpec,
    TOOL_REGISTRY,
    get_tool,
)


@dataclass
class ToolStep:
    """A single step in an execution plan."""
    tool_name: str
    params: Dict[str, Any]
    description: str = ""
    step_index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "params": self.params,
            "description": self.description,
            "step_index": self.step_index,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolStep":
        return cls(
            tool_name=data.get("tool_name", ""),
            params=data.get("params", {}),
            description=data.get("description", ""),
            step_index=data.get("step_index", 0),
        )


@dataclass
class ExecutionTrace:
    """Trace of all tool executions."""
    steps: List[ToolStep] = field(default_factory=list)
    results: List[ToolResult] = field(default_factory=list)
    total_elapsed_ms: float = 0.0
    merged_chunk_ids: List[int] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "results": [r.to_dict() for r in self.results],
            "total_elapsed_ms": self.total_elapsed_ms,
            "merged_chunk_ids": self.merged_chunk_ids,
            "error_count": len(self.errors),
            "errors": self.errors,
        }


@dataclass  
class ExecutionResult:
    """Result of executing a plan."""
    chunk_ids: List[int]
    scores: Dict[int, float]
    trace: ExecutionTrace
    success: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_ids": self.chunk_ids,
            "scores": self.scores,
            "trace": self.trace.to_dict(),
            "success": self.success,
        }


class ToolExecutor:
    """
    Executes tool steps deterministically with logging.
    
    Usage:
        executor = ToolExecutor()
        result = executor.execute_plan(plan, conn)
        
        # Access merged chunks
        chunks = result.chunk_ids
        
        # Access full trace
        trace = result.trace
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._trace: Optional[ExecutionTrace] = None
    
    def execute_step(
        self,
        step: ToolStep,
        conn,
    ) -> ToolResult:
        """
        Execute a single tool step.
        
        Args:
            step: The tool step to execute
            conn: Database connection
        
        Returns:
            ToolResult from the tool execution
        """
        tool_spec = get_tool(step.tool_name)
        
        if not tool_spec:
            return ToolResult(
                tool_name=step.tool_name,
                params=step.params,
                chunk_ids=[],
                scores={},
                metadata={},
                elapsed_ms=0,
                success=False,
                error=f"Unknown tool: {step.tool_name}",
            )
        
        if self.verbose:
            print(f"    [Step {step.step_index}] {step.tool_name}: {step.description}", 
                  file=sys.stderr)
            # Print detailed parameters
            self._log_params(step.tool_name, step.params)
        
        # Execute the tool - filter params to only valid ones for this tool
        valid_params = self._filter_params(tool_spec, step.params)
        
        try:
            result = tool_spec.fn(conn, **valid_params)
        except Exception as e:
            # Rollback to recover from database errors
            try:
                conn.rollback()
            except:
                pass
            
            result = ToolResult(
                tool_name=step.tool_name,
                params=valid_params,
                chunk_ids=[],
                scores={},
                metadata={"error_type": type(e).__name__},
                elapsed_ms=0,
                success=False,
                error=str(e),
            )
        
        if self.verbose:
            self._log_result(step.tool_name, result)
        
        # If tool failed, try to rollback to allow subsequent tools to work
        if not result.success:
            try:
                conn.rollback()
            except:
                pass
        
        return result
    
    def _filter_params(self, tool_spec: "ToolSpec", params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter params to only those accepted by the tool.
        
        The LLM may generate extra params that aren't valid - we filter them out
        and log warnings rather than failing.
        """
        if not tool_spec.params_schema:
            return params
        
        valid_keys = set(tool_spec.params_schema.keys())
        filtered = {}
        ignored = []
        
        for key, value in params.items():
            if key in valid_keys:
                filtered[key] = value
            else:
                ignored.append(key)
        
        if ignored and self.verbose:
            print(f"        (ignoring unknown params: {ignored})", file=sys.stderr)
        
        return filtered
    
    def _log_params(self, tool_name: str, params: Dict[str, Any]):
        """Log detailed parameters for a tool call."""
        if tool_name == "hybrid_search":
            print(f"        query=\"{params.get('query', '')[:80]}...\"", file=sys.stderr)
            print(f"        top_k={params.get('top_k', 200)}, "
                  f"collections={params.get('collections')}, "
                  f"expand_concordance={params.get('expand_concordance', True)}", file=sys.stderr)
        elif tool_name == "vector_search":
            print(f"        query=\"{params.get('query', '')[:80]}...\"", file=sys.stderr)
            print(f"        top_k={params.get('top_k', 200)}", file=sys.stderr)
        elif tool_name == "lexical_search":
            print(f"        terms={params.get('terms', [])}", file=sys.stderr)
            print(f"        top_k={params.get('top_k', 200)}", file=sys.stderr)
        elif tool_name == "lexical_exact":
            print(f"        term=\"{params.get('term', '')}\"", file=sys.stderr)
            print(f"        top_k={params.get('top_k', 200)}, case_sensitive={params.get('case_sensitive', False)}", file=sys.stderr)
        elif tool_name == "expand_aliases":
            print(f"        term=\"{params.get('term', '')}\"", file=sys.stderr)
            print(f"        max_aliases={params.get('max_aliases', 25)}", file=sys.stderr)
        else:
            # Generic param logging
            for key, value in params.items():
                val_str = str(value)[:50]
                print(f"        {key}={val_str}", file=sys.stderr)
    
    def _log_result(self, tool_name: str, result: ToolResult):
        """Log detailed results from a tool call."""
        if result.success:
            # Check for graceful "no results" scenarios
            message = result.metadata.get("message", "")
            entity_found = result.metadata.get("entity_found", True)
            found = result.metadata.get("found", True)
            
            if not entity_found or found is False:
                print(f"      -> (not found) {message}", file=sys.stderr)
                if result.metadata.get("suggestion"):
                    print(f"         Suggestion: {result.metadata['suggestion']}", file=sys.stderr)
            elif message and not result.chunk_ids:
                print(f"      -> (empty) {message}", file=sys.stderr)
            else:
                print(f"      -> {len(result.chunk_ids)} chunks in {result.elapsed_ms:.0f}ms", 
                      file=sys.stderr)
            
            # Tool-specific metadata logging
            if tool_name == "expand_aliases":
                aliases = result.metadata.get("aliases", [])
                if aliases:
                    print(f"         Aliases found ({len(aliases)}): {aliases[:5]}{'...' if len(aliases) > 5 else ''}", 
                          file=sys.stderr)
                else:
                    print(f"         No aliases found in concordance", file=sys.stderr)
            elif tool_name == "entity_lookup":
                if result.metadata.get("found"):
                    matched_via = result.metadata.get('matched_via', '')
                    concordance_note = f" [via {matched_via}]" if matched_via and 'concordance' in str(matched_via) else ""
                    print(f"         Found: {result.metadata.get('canonical_name')} "
                          f"(id={result.metadata.get('entity_id')}, type={result.metadata.get('entity_type')}){concordance_note}", 
                          file=sys.stderr)
            elif tool_name == "entity_mentions":
                if result.metadata.get("entity_name"):
                    matched_via = result.metadata.get('matched_via', '')
                    concordance_note = f" [via {matched_via}]" if matched_via and 'concordance' in str(matched_via) else ""
                    print(f"         Entity: {result.metadata.get('entity_name')} ({result.metadata.get('total_mentions', 0)} mentions){concordance_note}", 
                          file=sys.stderr)
            elif tool_name == "co_mention_entities":
                co_entities = result.metadata.get("co_entities", [])
                matched_via = result.metadata.get('matched_via', '')
                if matched_via and 'concordance' in str(matched_via):
                    print(f"         Resolved via concordance: {matched_via}", file=sys.stderr)
                if co_entities:
                    print(f"         Co-mentioned entities ({len(co_entities)}):", file=sys.stderr)
                    for e in co_entities[:5]:
                        print(f"           - {e.get('name')} ({e.get('co_occurrence_count')} co-occurrences)", file=sys.stderr)
            elif tool_name == "first_mention":
                if result.chunk_ids:
                    matched_via = result.metadata.get('matched_via', '')
                    concordance_note = f" [via {matched_via}]" if matched_via and 'concordance' in str(matched_via) else ""
                    date = result.metadata.get('date', 'unknown date')
                    print(f"         First mention: {date}, entity={result.metadata.get('entity_name')}{concordance_note}", 
                          file=sys.stderr)
            elif tool_name == "lexical_search":
                mode = result.metadata.get("search_mode", "unknown")
                print(f"         Search mode: {mode}", file=sys.stderr)
                if result.chunk_ids:
                    print(f"         First chunks: {result.chunk_ids[:5]}", file=sys.stderr)
            elif tool_name == "lexical_search":
                mode = result.metadata.get("search_mode", "unknown")
                print(f"         Search mode: {mode}", file=sys.stderr)
                if result.metadata.get("aliases_expanded"):
                    alias_summary = result.metadata.get("alias_summary", {})
                    for term, aliases in alias_summary.items():
                        if aliases:
                            print(f"         Aliases for '{term}': {aliases[:5]}{'...' if len(aliases) > 5 else ''}", 
                                  file=sys.stderr)
            elif tool_name in ("hybrid_search", "vector_search"):
                if result.metadata.get("aliases_expanded"):
                    aliases = result.metadata.get("expanded_aliases", [])
                    count = result.metadata.get("total_alias_count", 0)
                    if aliases:
                        print(f"         Concordance aliases ({count}): {aliases[:5]}{'...' if len(aliases) > 5 else ''}", 
                              file=sys.stderr)
                if result.chunk_ids:
                    # Show score distribution
                    scores = list(result.scores.values())
                    if scores:
                        print(f"         Score range: {min(scores):.3f} - {max(scores):.3f}", file=sys.stderr)
            elif tool_name == "lexical_exact":
                if result.metadata.get("aliases_expanded"):
                    search_terms = result.metadata.get("search_terms", [])
                    if len(search_terms) > 1:
                        print(f"         Search terms ({len(search_terms)}): {search_terms[:5]}{'...' if len(search_terms) > 5 else ''}", 
                              file=sys.stderr)
                        term_hits = result.metadata.get("term_hit_counts", {})
                        for t, count in list(term_hits.items())[:5]:
                            if count > 0:
                                print(f"           '{t}': {count} hits", file=sys.stderr)
                if result.chunk_ids:
                    scores = list(result.scores.values())
                    if scores:
                        print(f"         Score range: {min(scores):.3f} - {max(scores):.3f}", file=sys.stderr)
        else:
            print(f"      -> ERROR: {result.error}", file=sys.stderr)
            if result.metadata.get("error_details"):
                print(f"         Details: {result.metadata['error_details'][:200]}", file=sys.stderr)
            if result.metadata.get("error_type"):
                print(f"         Type: {result.metadata['error_type']}", file=sys.stderr)
    
    def execute_steps(
        self,
        steps: List[ToolStep],
        conn,
    ) -> ExecutionResult:
        """
        Execute multiple steps and merge results.
        
        Args:
            steps: List of tool steps to execute
            conn: Database connection
        
        Returns:
            ExecutionResult with merged chunks and trace
        """
        trace = ExecutionTrace()
        all_chunk_ids: Set[int] = set()
        all_scores: Dict[int, float] = {}
        
        start_time = time.time()
        
        for i, step in enumerate(steps):
            step.step_index = i
            trace.steps.append(step)
            
            result = self.execute_step(step, conn)
            trace.results.append(result)
            
            if result.success:
                # Merge chunk_ids (preserve order, no duplicates)
                for chunk_id in result.chunk_ids:
                    if chunk_id not in all_chunk_ids:
                        all_chunk_ids.add(chunk_id)
                
                # Merge scores (take max for duplicates)
                for chunk_id, score in result.scores.items():
                    if chunk_id not in all_scores or score > all_scores[chunk_id]:
                        all_scores[chunk_id] = score
            else:
                trace.errors.append(f"Step {i} ({step.tool_name}): {result.error}")
        
        trace.total_elapsed_ms = (time.time() - start_time) * 1000
        
        # Sort merged chunks by score (highest first)
        sorted_chunks = sorted(
            all_chunk_ids,
            key=lambda c: all_scores.get(c, 0.0),
            reverse=True,
        )
        trace.merged_chunk_ids = sorted_chunks
        
        self._trace = trace
        
        return ExecutionResult(
            chunk_ids=sorted_chunks,
            scores=all_scores,
            trace=trace,
            success=len(trace.errors) == 0,
        )
    
    def execute_plan(
        self,
        plan: "AgentPlanV3",
        conn,
    ) -> ExecutionResult:
        """
        Execute a full plan.
        
        Args:
            plan: The AgentPlanV3 to execute
            conn: Database connection
        
        Returns:
            ExecutionResult with merged chunks and trace
        """
        if self.verbose:
            print(f"\n  [Executor] Executing plan with {len(plan.steps)} steps", file=sys.stderr)
        
        return self.execute_steps(plan.steps, conn)
    
    def get_trace(self) -> Optional[ExecutionTrace]:
        """Get the trace from the last execution."""
        return self._trace


def merge_tool_results(results: List[ToolResult]) -> tuple:
    """
    Merge multiple tool results into unified chunk list.
    
    Args:
        results: List of ToolResult objects
    
    Returns:
        (chunk_ids, scores) tuple with merged, deduplicated results
    """
    all_chunk_ids: Set[int] = set()
    all_scores: Dict[int, float] = {}
    
    for result in results:
        if not result.success:
            continue
        
        for chunk_id in result.chunk_ids:
            all_chunk_ids.add(chunk_id)
        
        for chunk_id, score in result.scores.items():
            if chunk_id not in all_scores or score > all_scores[chunk_id]:
                all_scores[chunk_id] = score
    
    # Sort by score descending
    sorted_chunks = sorted(
        all_chunk_ids,
        key=lambda c: all_scores.get(c, 0.0),
        reverse=True,
    )
    
    return sorted_chunks, all_scores


# Type hint for circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from retrieval.agent.v3_plan import AgentPlanV3
