#!/usr/bin/env python3
"""
Benchmark harness for index retrieval operations.

This script measures and reports performance metrics for index operations
to help identify performance regressions and optimization opportunities.

Usage:
    python scripts/benchmark_index_ops.py
    
    # With custom database URL
    DATABASE_URL=postgresql://... python scripts/benchmark_index_ops.py
    
    # With EXPLAIN ANALYZE output
    python scripts/benchmark_index_ops.py --explain
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    operation: str
    params: Dict[str, Any]
    execution_time_ms: float
    rows_returned: int
    total_hits: Optional[int] = None
    explain_plan: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    timestamp: str
    database_url_hash: str  # Don't expose full URL
    results: List[BenchmarkResult] = field(default_factory=list)
    
    def add_result(self, result: BenchmarkResult):
        self.results.append(result)
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        by_operation = {}
        for r in self.results:
            if r.operation not in by_operation:
                by_operation[r.operation] = []
            by_operation[r.operation].append(r.execution_time_ms)
        
        return {
            op: {
                "count": len(times),
                "min_ms": min(times),
                "max_ms": max(times),
                "avg_ms": sum(times) / len(times),
            }
            for op, times in by_operation.items()
        }


def benchmark_first_mention(conn, entity_id: int, explain: bool = False) -> BenchmarkResult:
    """Benchmark FIRST_MENTION operation."""
    from retrieval.index_ops import first_mention
    
    start = time.perf_counter()
    try:
        hit, metadata = first_mention(conn, entity_id)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return BenchmarkResult(
            operation="FIRST_MENTION",
            params={"entity_id": entity_id},
            execution_time_ms=elapsed_ms,
            rows_returned=1 if hit else 0,
            total_hits=metadata.total_hits,
        )
    except Exception as e:
        return BenchmarkResult(
            operation="FIRST_MENTION",
            params={"entity_id": entity_id},
            execution_time_ms=(time.perf_counter() - start) * 1000,
            rows_returned=0,
            error=str(e),
        )


def benchmark_mentions_paginated(
    conn, 
    entity_id: int, 
    limit: int = 100,
    after_rank: Optional[int] = None,
) -> BenchmarkResult:
    """Benchmark MENTIONS pagination."""
    from retrieval.index_ops import mentions_paginated
    
    start = time.perf_counter()
    try:
        hits, metadata = mentions_paginated(
            conn, entity_id, limit=limit, after_rank=after_rank
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return BenchmarkResult(
            operation="MENTIONS_PAGINATED",
            params={"entity_id": entity_id, "limit": limit, "after_rank": after_rank},
            execution_time_ms=elapsed_ms,
            rows_returned=len(hits),
            total_hits=metadata.total_hits,
        )
    except Exception as e:
        return BenchmarkResult(
            operation="MENTIONS_PAGINATED",
            params={"entity_id": entity_id, "limit": limit},
            execution_time_ms=(time.perf_counter() - start) * 1000,
            rows_returned=0,
            error=str(e),
        )


def benchmark_first_co_mention(conn, entity_ids: List[int]) -> BenchmarkResult:
    """Benchmark FIRST_CO_MENTION operation."""
    from retrieval.index_ops import first_co_mention
    
    start = time.perf_counter()
    try:
        hit, metadata = first_co_mention(conn, entity_ids)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return BenchmarkResult(
            operation="FIRST_CO_MENTION",
            params={"entity_ids": entity_ids},
            execution_time_ms=elapsed_ms,
            rows_returned=1 if hit else 0,
            total_hits=metadata.total_hits,
        )
    except Exception as e:
        return BenchmarkResult(
            operation="FIRST_CO_MENTION",
            params={"entity_ids": entity_ids},
            execution_time_ms=(time.perf_counter() - start) * 1000,
            rows_returned=0,
            error=str(e),
        )


def benchmark_date_range_filter(
    conn,
    date_start: str,
    date_end: str,
    time_basis: str = "mentioned_date",
) -> BenchmarkResult:
    """Benchmark DATE_RANGE_FILTER operation."""
    from retrieval.index_ops import date_range_filter
    
    start = time.perf_counter()
    try:
        hits, metadata = date_range_filter(
            conn, date_start, date_end, time_basis=time_basis, limit=100
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return BenchmarkResult(
            operation="DATE_RANGE_FILTER",
            params={"date_start": date_start, "date_end": date_end, "time_basis": time_basis},
            execution_time_ms=elapsed_ms,
            rows_returned=len(hits),
            total_hits=metadata.total_hits,
        )
    except Exception as e:
        return BenchmarkResult(
            operation="DATE_RANGE_FILTER",
            params={"date_start": date_start, "date_end": date_end},
            execution_time_ms=(time.perf_counter() - start) * 1000,
            rows_returned=0,
            error=str(e),
        )


def benchmark_place_mentions(conn, place_id: int) -> BenchmarkResult:
    """Benchmark PLACE_MENTIONS operation."""
    from retrieval.index_ops import place_mentions
    
    start = time.perf_counter()
    try:
        hits, metadata = place_mentions(conn, place_id, limit=100)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return BenchmarkResult(
            operation="PLACE_MENTIONS",
            params={"place_id": place_id},
            execution_time_ms=elapsed_ms,
            rows_returned=len(hits),
            total_hits=metadata.total_hits,
        )
    except Exception as e:
        return BenchmarkResult(
            operation="PLACE_MENTIONS",
            params={"place_id": place_id},
            execution_time_ms=(time.perf_counter() - start) * 1000,
            rows_returned=0,
            error=str(e),
        )


def find_test_entities(conn) -> Dict[str, Any]:
    """Find suitable entities for benchmarking."""
    entities = {}
    
    with conn.cursor() as cur:
        # Find entity with many mentions
        cur.execute("""
            SELECT entity_id, COUNT(*) as cnt
            FROM entity_mentions
            GROUP BY entity_id
            ORDER BY cnt DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        if row:
            entities["high_cardinality_entity"] = {"id": row[0], "mention_count": row[1]}
        
        # Find two entities that co-occur
        cur.execute("""
            SELECT e1.entity_id, e2.entity_id
            FROM entity_mentions e1
            JOIN entity_mentions e2 ON e1.chunk_id = e2.chunk_id AND e1.entity_id < e2.entity_id
            GROUP BY e1.entity_id, e2.entity_id
            HAVING COUNT(*) >= 5
            LIMIT 1
        """)
        row = cur.fetchone()
        if row:
            entities["co_occurring_entities"] = [row[0], row[1]]
        
        # Find a place entity
        cur.execute("""
            SELECT e.id
            FROM entities e
            WHERE e.entity_type = 'place'
            AND EXISTS (SELECT 1 FROM entity_mentions em WHERE em.entity_id = e.id)
            LIMIT 1
        """)
        row = cur.fetchone()
        if row:
            entities["place_entity"] = row[0]
    
    return entities


def run_benchmarks(conn, explain: bool = False) -> BenchmarkReport:
    """Run all benchmarks and return report."""
    import hashlib
    from datetime import datetime
    
    database_url = os.environ.get("DATABASE_URL", "")
    url_hash = hashlib.md5(database_url.encode()).hexdigest()[:8]
    
    report = BenchmarkReport(
        timestamp=datetime.utcnow().isoformat(),
        database_url_hash=url_hash,
    )
    
    # Find test entities
    print("Finding test entities...")
    entities = find_test_entities(conn)
    
    # Benchmark FIRST_MENTION
    if "high_cardinality_entity" in entities:
        entity = entities["high_cardinality_entity"]
        print(f"Benchmarking FIRST_MENTION (entity with {entity['mention_count']} mentions)...")
        result = benchmark_first_mention(conn, entity["id"], explain)
        report.add_result(result)
        print(f"  -> {result.execution_time_ms:.2f}ms")
    
    # Benchmark MENTIONS pagination (page 1, 10, deep)
    if "high_cardinality_entity" in entities:
        entity_id = entities["high_cardinality_entity"]["id"]
        
        print("Benchmarking MENTIONS_PAGINATED (page 1)...")
        result = benchmark_mentions_paginated(conn, entity_id, limit=100)
        report.add_result(result)
        print(f"  -> {result.execution_time_ms:.2f}ms, {result.rows_returned} rows")
        
        # Get last rank for next page
        if result.rows_returned > 0:
            print("Benchmarking MENTIONS_PAGINATED (page 10 simulation)...")
            result2 = benchmark_mentions_paginated(conn, entity_id, limit=100, after_rank=900)
            report.add_result(result2)
            print(f"  -> {result2.execution_time_ms:.2f}ms")
    
    # Benchmark FIRST_CO_MENTION
    if "co_occurring_entities" in entities:
        entity_ids = entities["co_occurring_entities"]
        print(f"Benchmarking FIRST_CO_MENTION ({entity_ids})...")
        result = benchmark_first_co_mention(conn, entity_ids)
        report.add_result(result)
        print(f"  -> {result.execution_time_ms:.2f}ms")
    
    # Benchmark DATE_RANGE_FILTER
    print("Benchmarking DATE_RANGE_FILTER (1944-1945)...")
    result = benchmark_date_range_filter(conn, "1944-01-01", "1945-12-31")
    report.add_result(result)
    print(f"  -> {result.execution_time_ms:.2f}ms, {result.rows_returned} rows")
    
    # Benchmark PLACE_MENTIONS
    if "place_entity" in entities:
        print(f"Benchmarking PLACE_MENTIONS ({entities['place_entity']})...")
        result = benchmark_place_mentions(conn, entities["place_entity"])
        report.add_result(result)
        print(f"  -> {result.execution_time_ms:.2f}ms")
    
    return report


def main():
    """Main entry point."""
    import psycopg2
    
    parser = argparse.ArgumentParser(description="Benchmark index operations")
    parser.add_argument("--explain", action="store_true", help="Include EXPLAIN ANALYZE output")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()
    
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)
    
    print("Connecting to database...")
    conn = psycopg2.connect(database_url)
    
    print("\n" + "=" * 60)
    print("Index Operations Benchmark")
    print("=" * 60 + "\n")
    
    report = run_benchmarks(conn, explain=args.explain)
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    summary = report.summary()
    for op, stats in summary.items():
        print(f"\n{op}:")
        print(f"  Runs: {stats['count']}")
        print(f"  Min: {stats['min_ms']:.2f}ms")
        print(f"  Max: {stats['max_ms']:.2f}ms")
        print(f"  Avg: {stats['avg_ms']:.2f}ms")
    
    if args.json:
        print("\n" + "=" * 60)
        print("JSON Output")
        print("=" * 60)
        print(json.dumps({
            "timestamp": report.timestamp,
            "results": [r.to_dict() for r in report.results],
            "summary": summary,
        }, indent=2))
    
    conn.close()
    
    # Check for errors
    errors = [r for r in report.results if r.error]
    if errors:
        print(f"\nWARNING: {len(errors)} benchmarks had errors")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
