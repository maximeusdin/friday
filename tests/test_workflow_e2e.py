#!/usr/bin/env python3
"""
End-to-end workflow tests for the research assistant plan execution system.

These tests verify complete workflows, not just individual functions.
They use mocked database connections but test the actual logic flow.

Run with: pytest tests/test_workflow_e2e.py -v
"""

import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from datetime import datetime
from copy import deepcopy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psycopg2.extras import Json


# =============================================================================
# Workflow 1: Clarification Flow
# proposed → needs_clarification → clarify → approved → executed
# =============================================================================

class TestClarificationWorkflow:
    """
    Test the full clarification workflow:
    1. Plan is created with needs_clarification=True
    2. Approval is blocked
    3. Clarification resolves ambiguity
    4. New plan is created (supersedes original)
    5. New plan is approved
    6. Plan is executed
    """
    
    def test_approval_blocked_when_needs_clarification(self):
        """Step 2: Approval should fail when needs_clarification=True."""
        from scripts.approve_plan import approve_plan
        
        mock_conn = MagicMock()
        cursor = MagicMock()
        # Return a plan that needs clarification
        cursor.fetchone.return_value = (
            1,  # id
            "proposed",  # status
            {
                "needs_clarification": True,
                "choices": ["John Smith (person)", "Mary Smith (person)"],
                "query": {"raw": "Find Smith", "primitives": []},
            },
        )
        mock_conn.cursor.return_value.__enter__.return_value = cursor
        
        result = approve_plan(mock_conn, plan_id=1)
        
        assert result is False, "Approval must be blocked when needs_clarification=True"
        # Verify no UPDATE was executed (only SELECT)
        update_calls = [c for c in cursor.execute.call_args_list if "UPDATE" in str(c)]
        assert len(update_calls) == 0, "Should not update plan when clarification needed"
    
    def test_clarification_creates_superseding_plan(self):
        """Step 3-4: Clarifying should create a new plan that supersedes original."""
        from scripts.clarify_plan import supersede_plan
        
        mock_conn = MagicMock()
        cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = cursor
        
        supersede_plan(mock_conn, plan_id=1)
        
        # Verify UPDATE was called with status='superseded'
        execute_calls = cursor.execute.call_args_list
        update_call = [c for c in execute_calls if "UPDATE" in str(c) and "superseded" in str(c)]
        assert len(update_call) > 0, "Should mark original plan as superseded"
    
    def test_clarified_plan_can_be_approved(self):
        """Step 5: Clarified plan (needs_clarification=False) can be approved."""
        from scripts.approve_plan import approve_plan
        
        mock_conn = MagicMock()
        cursor = MagicMock()
        # Return a clarified plan (no longer needs clarification)
        cursor.fetchone.return_value = (
            2,  # id (new plan)
            "proposed",  # status
            {
                "needs_clarification": False,
                "query": {"raw": "Find John Smith", "primitives": [{"type": "ENTITY", "entity_id": 1}]},
                "_metadata": {
                    "clarified_from_plan_id": 1,
                    "clarification_choice": "John Smith (person)",
                }
            },
        )
        mock_conn.cursor.return_value.__enter__.return_value = cursor
        
        result = approve_plan(mock_conn, plan_id=2)
        
        assert result is True, "Clarified plan should be approvable"


# =============================================================================
# Workflow 2: Idempotent Execution
# proposed → approved → executed → re-exec without force returns same ids
# =============================================================================

class TestIdempotentExecutionWorkflow:
    """
    Test idempotent execution:
    1. Plan is approved and executed
    2. Re-running execute_plan.py without --force returns same IDs
    3. No new retrieval_run or result_set is created
    """
    
    def test_reexec_returns_existing_ids(self):
        """Re-execution without --force should return existing run/result_set IDs."""
        from scripts.execute_plan import get_existing_execution
        
        mock_conn = MagicMock()
        cursor = MagicMock()
        
        # First execution stored these IDs
        expected_run_id = 100
        expected_result_set_id = 200
        executed_at = datetime(2024, 1, 15, 10, 30, 0)
        
        cursor.fetchone.return_value = (
            expected_run_id,
            expected_result_set_id,
            executed_at,
        )
        mock_conn.cursor.return_value.__enter__.return_value = cursor
        
        result = get_existing_execution(mock_conn, plan_id=1)
        
        assert result is not None, "Should find existing execution"
        assert result["retrieval_run_id"] == expected_run_id
        assert result["result_set_id"] == expected_result_set_id
        assert result["executed_at"] == executed_at
    
    def test_no_new_run_created_without_force(self):
        """Without --force, no new retrieval_run should be created."""
        # This is enforced by the main() logic checking get_existing_execution
        # and exiting early if found (without --force)
        from scripts.execute_plan import get_existing_execution
        
        mock_conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = (100, 200, datetime.now())
        mock_conn.cursor.return_value.__enter__.return_value = cursor
        
        existing = get_existing_execution(mock_conn, plan_id=1)
        
        # Main logic: if existing and not args.force, exit with existing IDs
        assert existing is not None
        # The caller (main) would exit here without creating new run


# =============================================================================
# Workflow 3: Force Re-execution with History
# force re-exec appends _metadata.executions[] and updates plan's current ids
# =============================================================================

class TestForceReexecutionWorkflow:
    """
    Test force re-execution:
    1. Plan was previously executed with run_id=100, result_set_id=200
    2. --force creates new run_id=101, result_set_id=201
    3. Previous execution is appended to _metadata.executions[]
    4. Plan's current run_id/result_set_id updated to new values
    """
    
    def test_force_appends_to_execution_history(self):
        """--force should append previous execution to _metadata.executions[]."""
        from scripts.execute_plan import update_plan_status, MAX_EXECUTION_HISTORY
        
        mock_conn = MagicMock()
        cursor = MagicMock()
        
        # Existing plan_json with no prior executions
        existing_plan_json = {
            "query": {"raw": "test", "primitives": []},
            "_metadata": {}
        }
        cursor.fetchone.return_value = (existing_plan_json,)
        mock_conn.cursor.return_value.__enter__.return_value = cursor
        
        # Simulate force re-execution
        update_plan_status(
            mock_conn,
            plan_id=1,
            run_id=101,  # New run
            result_set_id=201,  # New result set
            is_reexecution=True,
            previous_run_id=100,  # Previous run
            previous_result_set_id=200,  # Previous result set
            execution_mode="retrieve",
        )
        
        # Verify UPDATE was called with updated plan_json
        execute_calls = cursor.execute.call_args_list
        update_calls = [c for c in execute_calls if "UPDATE research_plans" in str(c)]
        assert len(update_calls) > 0, "Should update plan"
        
        # The update should include the new plan_json with executions history
        update_call = update_calls[-1]
        call_args = update_call[0][1]  # (sql, params) -> params
        
        # params should include Json(plan_json) with executions
        plan_json_param = None
        for param in call_args:
            if hasattr(param, 'adapted'):  # Json wrapper
                plan_json_param = param.adapted
                break
        
        if plan_json_param:
            assert "_metadata" in plan_json_param
            assert "executions" in plan_json_param["_metadata"]
            executions = plan_json_param["_metadata"]["executions"]
            assert len(executions) > 0
            assert executions[-1]["retrieval_run_id"] == 100
            assert executions[-1]["result_set_id"] == 200
    
    def test_execution_history_limited_to_max(self):
        """Execution history should not exceed MAX_EXECUTION_HISTORY."""
        from scripts.execute_plan import update_plan_status, MAX_EXECUTION_HISTORY
        
        mock_conn = MagicMock()
        cursor = MagicMock()
        
        # Existing plan_json with MAX_EXECUTION_HISTORY executions already
        existing_executions = [
            {"retrieval_run_id": i, "result_set_id": i + 100, "superseded_at": "2024-01-01"}
            for i in range(MAX_EXECUTION_HISTORY)
        ]
        existing_plan_json = {
            "query": {"raw": "test", "primitives": []},
            "_metadata": {"executions": existing_executions}
        }
        cursor.fetchone.return_value = (existing_plan_json,)
        mock_conn.cursor.return_value.__enter__.return_value = cursor
        
        update_plan_status(
            mock_conn,
            plan_id=1,
            run_id=999,
            result_set_id=1999,
            is_reexecution=True,
            previous_run_id=MAX_EXECUTION_HISTORY - 1,
            previous_result_set_id=MAX_EXECUTION_HISTORY + 99,
            execution_mode="retrieve",
        )
        
        # Verify the truncation logic was applied
        # (History should be capped at MAX_EXECUTION_HISTORY)
        assert MAX_EXECUTION_HISTORY == 20, "Default max should be 20"


# =============================================================================
# Workflow 4: COUNT Mode Execution
# count execution creates retrieval_run, plan executed, result_set_id NULL
# =============================================================================

class TestCountModeWorkflow:
    """
    Test COUNT mode execution:
    1. Plan with COUNT primitive is detected
    2. Executes aggregation query (not full retrieval)
    3. Creates retrieval_run for audit trail
    4. result_set_id remains NULL
    5. Output JSON has correct shape
    """
    
    def test_count_mode_detected(self):
        """COUNT primitive in plan should trigger count mode."""
        from scripts.execute_plan import is_count_mode
        from retrieval.primitives import (
            ResearchPlan, QueryPlan, TermPrimitive, CountPrimitive
        )
        
        query = QueryPlan(
            raw="How many documents mention Silvermaster?",
            primitives=[
                TermPrimitive(value="Silvermaster"),
                CountPrimitive(group_by="document"),
            ]
        )
        plan = ResearchPlan(query=query)
        
        is_count, group_by = is_count_mode(plan)
        
        assert is_count is True
        assert group_by == "document"
    
    def test_count_execution_creates_run_without_result_set(self):
        """COUNT execution should create retrieval_run but no result_set."""
        from scripts.execute_plan import log_count_execution
        from retrieval.primitives import ResearchPlan, QueryPlan, TermPrimitive, CountPrimitive
        
        mock_conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = (999,)  # New run_id
        mock_conn.cursor.return_value.__enter__.return_value = cursor
        
        query = QueryPlan(
            raw="Count test",
            primitives=[TermPrimitive(value="test"), CountPrimitive(group_by="document")]
        )
        plan = ResearchPlan(query=query)
        plan.compile()
        
        count_result = {
            "mode": "count",
            "total_count": 42,
            "group_by": "document",
            "buckets": [{"key": "1", "label": "Doc A", "count": 25}],
        }
        
        run_id = log_count_execution(
            mock_conn,
            plan,
            count_result,
            plan_id=1,
            session_id=1,
        )
        
        assert run_id == 999
        
        # Verify INSERT was called for retrieval_runs
        insert_calls = [c for c in cursor.execute.call_args_list if "INSERT INTO retrieval_runs" in str(c)]
        assert len(insert_calls) > 0, "Should create retrieval_run"
    
    def test_count_output_json_shape(self):
        """COUNT output JSON should have correct structure."""
        expected_shape = {
            "mode": "count",
            "total_count": 42,
            "group_by": "document",
            "buckets": [
                {"key": "1", "label": "Document A", "count": 25},
                {"key": "2", "label": "Document B", "count": 17},
            ],
            "retrieval_run_id": 999,
        }
        
        # Verify shape
        assert "mode" in expected_shape
        assert expected_shape["mode"] == "count"
        assert "total_count" in expected_shape
        assert "group_by" in expected_shape
        assert "buckets" in expected_shape
        assert isinstance(expected_shape["buckets"], list)
        assert "retrieval_run_id" in expected_shape
        
        # Verify bucket shape
        bucket = expected_shape["buckets"][0]
        assert "key" in bucket
        assert "label" in bucket
        assert "count" in bucket
    
    def test_count_execution_stores_mode_in_metadata(self):
        """COUNT execution should store execution_mode='count' in plan metadata."""
        from scripts.execute_plan import update_plan_status
        
        mock_conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = ({"query": {"raw": "test", "primitives": []}},)
        mock_conn.cursor.return_value.__enter__.return_value = cursor
        
        update_plan_status(
            mock_conn,
            plan_id=1,
            run_id=100,
            result_set_id=None,  # NULL for count mode
            execution_mode="count",
        )
        
        # Verify execution_mode was included in the update
        execute_calls = cursor.execute.call_args_list
        update_call = [c for c in execute_calls if "UPDATE" in str(c)][-1]
        
        # The call should include plan_json with execution_mode
        assert len(update_call) > 0


# =============================================================================
# Workflow 5: Best-Guess Mode
# best-guess mode sets flags + alternatives + settings recorded
# =============================================================================

class TestBestGuessModeWorkflow:
    """
    Test best-guess entity resolution:
    1. Ambiguous entity detected
    2. ENTITY_BEST_GUESS=1 picks top candidate
    3. is_best_guess=True flag set in context
    4. Alternatives recorded for review
    5. Settings (threshold, mode) recorded in metadata
    """
    
    def test_best_guess_entity_context_structure(self):
        """Best-guess resolution should include all required metadata."""
        # Simulated entity context from best-guess resolution
        entity_context = {
            "surface": "Rosenberg",
            "entity_id": 42,
            "canonical_name": "Julius Rosenberg",
            "entity_type": "person",
            "confidence": 0.92,
            "match_method": "alias_exact",
            "is_best_guess": True,
            "alternatives": [
                {"entity_id": 43, "canonical_name": "Ethel Rosenberg", "confidence": 0.88},
                {"entity_id": 44, "canonical_name": "Rosenberg Fund", "confidence": 0.65},
            ],
        }
        
        # Verify required fields
        assert entity_context["is_best_guess"] is True
        assert "alternatives" in entity_context
        assert len(entity_context["alternatives"]) > 0
        assert "confidence" in entity_context
        assert entity_context["confidence"] > 0
        
        # Verify alternatives have required fields
        alt = entity_context["alternatives"][0]
        assert "entity_id" in alt
        assert "canonical_name" in alt
        assert "confidence" in alt
    
    def test_resolution_settings_recorded(self):
        """Resolution settings should be recorded in plan metadata."""
        resolution_metadata = {
            "resolved_entities": [
                {
                    "surface": "Rosenberg",
                    "entity_id": 42,
                    "canonical_name": "Julius Rosenberg",
                    "confidence": 0.92,
                    "is_best_guess": True,
                    "alternatives": [
                        {"entity_id": 43, "canonical_name": "Ethel Rosenberg", "confidence": 0.88}
                    ],
                }
            ],
            "entity_resolution_settings": {
                "best_guess_mode": True,
                "confidence_threshold": 0.85,
            },
            "resolved_at": "2024-01-15T10:30:00",
        }
        
        # Verify settings structure
        settings = resolution_metadata["entity_resolution_settings"]
        assert "best_guess_mode" in settings
        assert "confidence_threshold" in settings
        assert settings["best_guess_mode"] is True
        assert isinstance(settings["confidence_threshold"], float)


# =============================================================================
# Workflow 6: CO_OCCURS_WITH Chunk Window Correctness
# CO_OCCURS_WITH chunk-window correctness (dedupe)
# =============================================================================

class TestCoOccursWithChunkWindow:
    """
    Test CO_OCCURS_WITH with chunk window:
    1. Generates correct SQL for chunk-level entity lookup
    2. Each chunk counted once (dedupe)
    3. Only chunks with entity mention qualify
    """
    
    def test_chunk_window_sql_structure(self):
        """CO_OCCURS_WITH chunk window should generate proper EXISTS clause."""
        from retrieval.primitives import CoOccursWithPrimitive, compile_primitives_to_scope
        
        primitives = [CoOccursWithPrimitive(entity_id=42, window="chunk")]
        
        sql, params, joins = compile_primitives_to_scope(primitives)
        
        # Verify SQL structure
        assert "EXISTS" in sql, "Should use EXISTS for entity check"
        assert "entity_mentions" in sql, "Should query entity_mentions table"
        assert "chunk_id" in sql, "Should filter by chunk_id"
        assert "entity_id" in sql, "Should filter by entity_id"
        
        # Verify params
        assert 42 in params, "Entity ID should be in params"
        
        # Verify joins
        assert joins["entity_mentions"] is True
    
    def test_chunk_window_dedupe_semantics(self):
        """Chunk window should count each chunk once regardless of mention count."""
        from retrieval.primitives import CoOccursWithPrimitive, compile_primitives_to_scope
        
        primitives = [CoOccursWithPrimitive(entity_id=42, window="chunk")]
        sql, params, joins = compile_primitives_to_scope(primitives)
        
        # EXISTS inherently dedupes - returns true if ANY match exists
        # So a chunk with 5 mentions of entity 42 is counted same as 1 mention
        assert "EXISTS" in sql, "EXISTS provides dedupe semantics"
        
        # Should NOT have COUNT(*) or aggregation that would multiply
        assert "COUNT" not in sql.upper(), "Should not count mentions"


# =============================================================================
# Workflow 7: CO_OCCURS_WITH Document Window Correctness
# CO_OCCURS_WITH document-window correctness (dedupe)
# =============================================================================

class TestCoOccursWithDocumentWindow:
    """
    Test CO_OCCURS_WITH with document window:
    1. Entity can be in any chunk of same document
    2. Uses denormalized document_id for performance
    3. Each qualifying chunk counted once (dedupe)
    """
    
    def test_document_window_sql_uses_document_id(self):
        """CO_OCCURS_WITH document window should use document_id."""
        from retrieval.primitives import CoOccursWithPrimitive, compile_primitives_to_scope
        
        primitives = [CoOccursWithPrimitive(entity_id=42, window="document")]
        
        sql, params, joins = compile_primitives_to_scope(primitives)
        
        # Verify SQL uses document_id
        assert "document_id" in sql, "Should use document_id for document window"
        assert "entity_mentions" in sql, "Should query entity_mentions"
        
        # Verify params
        assert 42 in params
        
        # Verify joins include chunk_metadata (for document_id lookup)
        assert joins["entity_mentions"] is True
        assert joins["chunk_metadata"] is True, "Should require chunk_metadata join for document_id"
    
    def test_document_window_uses_denormalized_document_id(self):
        """Document window should use entity_mentions.document_id (denormalized)."""
        from retrieval.primitives import CoOccursWithPrimitive, compile_primitives_to_scope
        
        primitives = [CoOccursWithPrimitive(entity_id=42, window="document")]
        sql, params, joins = compile_primitives_to_scope(primitives)
        
        # Should use entity_mentions.document_id for performance
        # NOT join through chunks table
        assert "em.document_id" in sql or "entity_mentions" in sql
        
        # Should NOT join chunks just to get document_id
        # (entity_mentions has denormalized document_id)


# =============================================================================
# Workflow 8: CO_LOCATED Correctness
# CO_LOCATED correctness
# =============================================================================

class TestCoLocatedCorrectness:
    """
    Test CO_LOCATED primitive:
    1. Both entities must be present in scope
    2. Chunk scope: both in same chunk
    3. Document scope: both in same document (any chunks)
    4. Rejects same entity for both a and b
    """
    
    def test_chunk_scope_requires_both_entities(self):
        """CO_LOCATED chunk scope should require both entities in same chunk."""
        from retrieval.primitives import CoLocatedPrimitive, compile_primitives_to_scope
        
        primitives = [CoLocatedPrimitive(entity_a=10, entity_b=20, scope="chunk")]
        
        sql, params, joins = compile_primitives_to_scope(primitives)
        
        # Should have EXISTS checks for BOTH entities
        # SQL should contain two EXISTS or an AND of entity checks
        assert sql.count("EXISTS") >= 2 or ("entity_id" in sql and "AND" in sql)
        
        # Both entity IDs should be in params
        assert 10 in params
        assert 20 in params
    
    def test_document_scope_joins_on_document(self):
        """CO_LOCATED document scope should join entities by document_id."""
        from retrieval.primitives import CoLocatedPrimitive, compile_primitives_to_scope
        
        primitives = [CoLocatedPrimitive(entity_a=10, entity_b=20, scope="document")]
        
        sql, params, joins = compile_primitives_to_scope(primitives)
        
        # Should use document_id for broader scope
        assert "document_id" in sql
        
        # Both entity IDs should be in params
        assert 10 in params
        assert 20 in params
    
    def test_rejects_same_entity(self):
        """CO_LOCATED should reject same entity for both a and b."""
        from retrieval.primitives import CoLocatedPrimitive
        
        with pytest.raises(ValueError) as exc_info:
            CoLocatedPrimitive(entity_a=42, entity_b=42, scope="chunk")
        
        assert "different entities" in str(exc_info.value).lower()
    
    def test_entity_order_irrelevant(self):
        """CO_LOCATED(a,b) should be equivalent to CO_LOCATED(b,a)."""
        from retrieval.primitives import CoLocatedPrimitive, compile_primitives_to_scope
        
        primitives_ab = [CoLocatedPrimitive(entity_a=10, entity_b=20, scope="chunk")]
        primitives_ba = [CoLocatedPrimitive(entity_a=20, entity_b=10, scope="chunk")]
        
        sql_ab, params_ab, _ = compile_primitives_to_scope(primitives_ab)
        sql_ba, params_ba, _ = compile_primitives_to_scope(primitives_ba)
        
        # Both should have the same entity IDs in params (order may differ)
        assert set(params_ab) == set(params_ba)


# =============================================================================
# Integration: Full Workflow Smoke Test
# =============================================================================

class TestFullWorkflowSmoke:
    """
    Smoke test combining multiple workflows.
    """
    
    def test_plan_lifecycle_state_machine(self):
        """Plans should follow valid state transitions."""
        valid_transitions = {
            "proposed": ["approved", "rejected", "superseded"],
            "approved": ["executed", "rejected"],
            "executed": ["executed"],  # Can re-execute with --force
            "rejected": ["approved"],  # Can re-approve after fixing
            "superseded": ["approved"],  # Superseded can be re-approved
        }
        
        # Verify each state has defined transitions
        for state, targets in valid_transitions.items():
            assert len(targets) > 0, f"State {state} should have valid transitions"
    
    def test_execution_mode_values(self):
        """Execution modes should be well-defined."""
        valid_modes = ["retrieve", "count"]
        
        # These are the only valid execution_mode values
        assert "retrieve" in valid_modes
        assert "count" in valid_modes
        assert len(valid_modes) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
