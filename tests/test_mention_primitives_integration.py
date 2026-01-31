#!/usr/bin/env python3
"""
Integration test for ENTITY and DATE_RANGE primitives in plan compilation.
Verifies that mention-driven primitives compile correctly and produce expected SQL.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.primitives import (
    EntityPrimitive,
    FilterDateRangePrimitive,
    FilterCollectionPrimitive,
    TermPrimitive,
    compile_primitives,
    ResearchPlan,
    QueryPlan,
)


def test_entity_primitive_compilation():
    """Test that ENTITY primitive compiles to correct SQL."""
    print("Testing ENTITY primitive compilation...")
    
    primitives = [
        EntityPrimitive(entity_id=12345),
        FilterCollectionPrimitive(slug="venona"),
    ]
    
    compiled = compile_primitives(primitives)
    
    # Check that scope was compiled
    assert "scope" in compiled, "Scope should be compiled"
    scope = compiled["scope"]
    
    # Check SQL contains entity_mentions
    assert "entity_mentions" in scope["where_sql"].lower(), \
        f"SQL should contain 'entity_mentions', got: {scope['where_sql']}"
    
    # Check required_joins indicates entity_mentions
    assert scope["required_joins"]["entity_mentions"] == True, \
        "required_joins should indicate entity_mentions is needed"
    
    # Check params contain entity_id
    assert 12345 in scope["params"], "Params should contain entity_id"
    
    print(f"  [OK] ENTITY SQL: {scope['where_sql'][:100]}...")
    print(f"  [OK] Required joins: {scope['required_joins']}")
    print("[OK] ENTITY primitive compilation test passed\n")


def test_date_range_primitive_compilation():
    """Test that FILTER_DATE_RANGE primitive compiles to use date_mentions."""
    print("Testing FILTER_DATE_RANGE primitive compilation...")
    
    primitives = [
        FilterDateRangePrimitive(start="1940-01-01", end="1945-12-31"),
    ]
    
    compiled = compile_primitives(primitives)
    
    # Check that scope was compiled
    assert "scope" in compiled, "Scope should be compiled"
    scope = compiled["scope"]
    
    # Check SQL contains date_mentions
    assert "date_mentions" in scope["where_sql"].lower(), \
        f"SQL should contain 'date_mentions', got: {scope['where_sql']}"
    
    # Check required_joins indicates date_mentions
    assert scope["required_joins"]["date_mentions"] == True, \
        "required_joins should indicate date_mentions is needed"
    
    # Check params contain dates
    assert "1940-01-01" in scope["params"], "Params should contain start date"
    assert "1945-12-31" in scope["params"], "Params should contain end date"
    
    print(f"  [OK] DATE_RANGE SQL: {scope['where_sql'][:150]}...")
    print(f"  [OK] Required joins: {scope['required_joins']}")
    print("[OK] FILTER_DATE_RANGE primitive compilation test passed\n")


def test_combined_entity_and_date_range():
    """Test combining ENTITY and DATE_RANGE primitives."""
    print("Testing combined ENTITY + DATE_RANGE primitives...")
    
    primitives = [
        EntityPrimitive(entity_id=12345),
        FilterDateRangePrimitive(start="1940-01-01", end="1945-12-31"),
        FilterCollectionPrimitive(slug="venona"),
    ]
    
    compiled = compile_primitives(primitives)
    
    scope = compiled["scope"]
    sql = scope["where_sql"]
    
    # Check both are present
    assert "entity_mentions" in sql.lower(), "Should contain entity_mentions"
    assert "date_mentions" in sql.lower() or "cm.date" in sql.lower(), \
        "Should contain date_mentions or chunk_metadata date fields"
    
    # Check required joins
    assert scope["required_joins"]["entity_mentions"] == True
    assert scope["required_joins"]["date_mentions"] == True or \
           scope["required_joins"]["chunk_metadata"] == True
    
    print(f"  [OK] Combined SQL: {sql[:200]}...")
    print(f"  [OK] Required joins: {scope['required_joins']}")
    print("[OK] Combined ENTITY + DATE_RANGE test passed\n")


def test_plan_json_serialization():
    """Test that plans with ENTITY and DATE_RANGE serialize correctly."""
    print("Testing plan JSON serialization...")
    
    query = QueryPlan(
        raw="Find mentions of entity 12345 between 1940-1945",
        primitives=[
            EntityPrimitive(entity_id=12345),
            FilterDateRangePrimitive(start="1940-01-01", end="1945-12-31"),
        ]
    )
    
    plan = ResearchPlan(query=query)
    plan.compile()
    
    # Convert to dict
    plan_dict = plan.to_dict()
    
    # Verify structure
    assert "query" in plan_dict
    assert "compiled" in plan_dict
    assert len(plan_dict["query"]["primitives"]) == 2
    
    # Verify primitives are correct
    prim_types = [p["type"] for p in plan_dict["query"]["primitives"]]
    assert "ENTITY" in prim_types
    assert "FILTER_DATE_RANGE" in prim_types
    
    # Verify compiled scope exists
    assert "scope" in plan_dict["compiled"]
    assert "required_joins" in plan_dict["compiled"]["scope"]
    
    print(f"  [OK] Plan JSON structure valid")
    print(f"  [OK] Compiled scope present: {list(plan_dict['compiled']['scope'].keys())}")
    print("[OK] Plan JSON serialization test passed\n")


def test_multiple_entities():
    """Test multiple ENTITY primitives (OR logic)."""
    print("Testing multiple ENTITY primitives...")
    
    primitives = [
        EntityPrimitive(entity_id=12345),
        EntityPrimitive(entity_id=67890),
    ]
    
    compiled = compile_primitives(primitives)
    scope = compiled["scope"]
    sql = scope["where_sql"]
    
    # Should have OR logic for multiple entities
    assert sql.count("EXISTS") >= 2 or "OR" in sql.upper(), \
        "Multiple entities should use OR logic"
    
    # Both entity IDs should be in params
    assert 12345 in scope["params"]
    assert 67890 in scope["params"]
    
    print(f"  [OK] Multiple entities SQL: {sql[:200]}...")
    print("[OK] Multiple ENTITY primitives test passed\n")


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("Mention-Driven Primitives Integration Tests")
    print("=" * 60)
    print()
    
    try:
        test_entity_primitive_compilation()
        test_date_range_primitive_compilation()
        test_combined_entity_and_date_range()
        test_plan_json_serialization()
        test_multiple_entities()
        
        print("=" * 60)
        print("[OK] All integration tests passed!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
