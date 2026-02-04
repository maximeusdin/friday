#!/usr/bin/env python3
"""
Result Set Comparison Tool

Compares two result sets and generates detailed comparison reports.

Features:
- Set operations: overlap, unique to A, unique to B, symmetric difference
- Metrics: Jaccard similarity, overlap coefficient, inclusion scores
- Multiple output formats: console, JSON, CSV, markdown
- Optional database persistence of comparison results

Usage:
    python scripts/compare_result_sets.py --set-a 25 --set-b 30
    python scripts/compare_result_sets.py --set-a 25 --set-b 30 --format json
    python scripts/compare_result_sets.py --set-a 25 --set-b 30 --save
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Set, Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2


@dataclass
class ResultSetInfo:
    """Information about a result set."""
    id: int
    name: Optional[str]
    chunk_count: int
    chunk_ids: Set[int]
    retrieval_run_id: Optional[int]
    created_at: Optional[datetime]


@dataclass
class ComparisonResult:
    """Result of comparing two result sets."""
    set_a_id: int
    set_b_id: int
    set_a_name: Optional[str]
    set_b_name: Optional[str]
    set_a_count: int
    set_b_count: int
    
    # Set operations
    overlap_count: int
    unique_to_a_count: int
    unique_to_b_count: int
    union_count: int
    
    # Metrics
    jaccard_similarity: float
    overlap_coefficient: float
    inclusion_a_in_b: float  # What fraction of A is in B
    inclusion_b_in_a: float  # What fraction of B is in A
    
    # Sample chunk IDs
    overlap_sample: List[int]
    unique_to_a_sample: List[int]
    unique_to_b_sample: List[int]
    
    # Metadata
    compared_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['compared_at'] = self.compared_at.isoformat()
        return result


def get_conn():
    """Get database connection from environment."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    return psycopg2.connect(database_url)


def fetch_result_set(conn, result_set_id: int) -> ResultSetInfo:
    """Fetch result set information from database."""
    with conn.cursor() as cur:
        # First try result_sets table
        cur.execute("""
            SELECT rs.id, rs.name, rs.chunk_ids, rs.retrieval_run_id, rs.created_at
            FROM result_sets rs
            WHERE rs.id = %s
        """, (result_set_id,))
        
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Result set {result_set_id} not found")
        
        rs_id, name, chunk_ids, retrieval_run_id, created_at = row
        
        # chunk_ids may be stored as array or we need to fetch from result_set_chunks
        if chunk_ids is not None:
            chunk_set = set(chunk_ids)
        else:
            # Fetch from result_set_chunks table
            cur.execute("""
                SELECT chunk_id FROM result_set_chunks
                WHERE result_set_id = %s
            """, (result_set_id,))
            chunk_set = {r[0] for r in cur.fetchall()}
        
        return ResultSetInfo(
            id=rs_id,
            name=name,
            chunk_count=len(chunk_set),
            chunk_ids=chunk_set,
            retrieval_run_id=retrieval_run_id,
            created_at=created_at,
        )


def compare_result_sets(
    set_a: ResultSetInfo,
    set_b: ResultSetInfo,
    sample_size: int = 10
) -> ComparisonResult:
    """Compare two result sets and compute metrics."""
    
    # Set operations
    overlap = set_a.chunk_ids & set_b.chunk_ids
    unique_to_a = set_a.chunk_ids - set_b.chunk_ids
    unique_to_b = set_b.chunk_ids - set_a.chunk_ids
    union = set_a.chunk_ids | set_b.chunk_ids
    
    # Metrics
    union_count = len(union)
    jaccard = len(overlap) / union_count if union_count > 0 else 0.0
    
    # Overlap coefficient = |A ∩ B| / min(|A|, |B|)
    min_size = min(len(set_a.chunk_ids), len(set_b.chunk_ids))
    overlap_coef = len(overlap) / min_size if min_size > 0 else 0.0
    
    # Inclusion metrics
    inclusion_a_in_b = len(overlap) / len(set_a.chunk_ids) if set_a.chunk_ids else 0.0
    inclusion_b_in_a = len(overlap) / len(set_b.chunk_ids) if set_b.chunk_ids else 0.0
    
    # Sample chunk IDs
    overlap_sample = sorted(list(overlap))[:sample_size]
    unique_a_sample = sorted(list(unique_to_a))[:sample_size]
    unique_b_sample = sorted(list(unique_to_b))[:sample_size]
    
    return ComparisonResult(
        set_a_id=set_a.id,
        set_b_id=set_b.id,
        set_a_name=set_a.name,
        set_b_name=set_b.name,
        set_a_count=set_a.chunk_count,
        set_b_count=set_b.chunk_count,
        overlap_count=len(overlap),
        unique_to_a_count=len(unique_to_a),
        unique_to_b_count=len(unique_to_b),
        union_count=union_count,
        jaccard_similarity=jaccard,
        overlap_coefficient=overlap_coef,
        inclusion_a_in_b=inclusion_a_in_b,
        inclusion_b_in_a=inclusion_b_in_a,
        overlap_sample=overlap_sample,
        unique_to_a_sample=unique_a_sample,
        unique_to_b_sample=unique_b_sample,
        compared_at=datetime.utcnow(),
    )


def format_console(result: ComparisonResult) -> str:
    """Format comparison result for console output."""
    lines = [
        "",
        "=" * 60,
        "Result Set Comparison",
        "=" * 60,
        "",
        f"Set A (ID={result.set_a_id}): {result.set_a_count} chunks",
    ]
    
    if result.set_a_name:
        lines.append(f"  Name: {result.set_a_name}")
    
    lines.extend([
        "",
        f"Set B (ID={result.set_b_id}): {result.set_b_count} chunks",
    ])
    
    if result.set_b_name:
        lines.append(f"  Name: {result.set_b_name}")
    
    lines.extend([
        "",
        "-" * 60,
        "Set Operations",
        "-" * 60,
        f"Overlap (A ∩ B):     {result.overlap_count} chunks",
        f"Unique to A (A - B): {result.unique_to_a_count} chunks",
        f"Unique to B (B - A): {result.unique_to_b_count} chunks",
        f"Union (A ∪ B):       {result.union_count} chunks",
        "",
        "-" * 60,
        "Similarity Metrics",
        "-" * 60,
        f"Jaccard Similarity:     {result.jaccard_similarity:.4f}",
        f"Overlap Coefficient:    {result.overlap_coefficient:.4f}",
        f"% of A contained in B:  {result.inclusion_a_in_b * 100:.1f}%",
        f"% of B contained in A:  {result.inclusion_b_in_a * 100:.1f}%",
    ])
    
    if result.overlap_sample:
        lines.extend([
            "",
            "-" * 60,
            "Sample Chunks",
            "-" * 60,
            f"Overlapping chunks: {result.overlap_sample}",
        ])
    
    if result.unique_to_a_sample:
        lines.append(f"Unique to A:        {result.unique_to_a_sample}")
    
    if result.unique_to_b_sample:
        lines.append(f"Unique to B:        {result.unique_to_b_sample}")
    
    lines.extend([
        "",
        "-" * 60,
        f"Compared at: {result.compared_at.isoformat()}",
        "=" * 60,
        "",
    ])
    
    return "\n".join(lines)


def format_json(result: ComparisonResult) -> str:
    """Format comparison result as JSON."""
    return json.dumps(result.to_dict(), indent=2)


def format_csv(result: ComparisonResult) -> str:
    """Format comparison result as CSV."""
    headers = [
        "set_a_id", "set_b_id", "set_a_count", "set_b_count",
        "overlap_count", "unique_to_a_count", "unique_to_b_count", "union_count",
        "jaccard_similarity", "overlap_coefficient",
        "inclusion_a_in_b", "inclusion_b_in_a", "compared_at"
    ]
    values = [
        str(result.set_a_id), str(result.set_b_id),
        str(result.set_a_count), str(result.set_b_count),
        str(result.overlap_count), str(result.unique_to_a_count),
        str(result.unique_to_b_count), str(result.union_count),
        f"{result.jaccard_similarity:.6f}", f"{result.overlap_coefficient:.6f}",
        f"{result.inclusion_a_in_b:.6f}", f"{result.inclusion_b_in_a:.6f}",
        result.compared_at.isoformat()
    ]
    return ",".join(headers) + "\n" + ",".join(values)


def format_markdown(result: ComparisonResult) -> str:
    """Format comparison result as Markdown."""
    lines = [
        "# Result Set Comparison",
        "",
        "## Summary",
        "",
        f"| Set | ID | Chunks | Name |",
        f"|-----|----:|-------:|------|",
        f"| A | {result.set_a_id} | {result.set_a_count} | {result.set_a_name or '-'} |",
        f"| B | {result.set_b_id} | {result.set_b_count} | {result.set_b_name or '-'} |",
        "",
        "## Set Operations",
        "",
        f"| Operation | Count |",
        f"|-----------|------:|",
        f"| Overlap (A ∩ B) | {result.overlap_count} |",
        f"| Unique to A (A - B) | {result.unique_to_a_count} |",
        f"| Unique to B (B - A) | {result.unique_to_b_count} |",
        f"| Union (A ∪ B) | {result.union_count} |",
        "",
        "## Similarity Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| Jaccard Similarity | {result.jaccard_similarity:.4f} |",
        f"| Overlap Coefficient | {result.overlap_coefficient:.4f} |",
        f"| % of A in B | {result.inclusion_a_in_b * 100:.1f}% |",
        f"| % of B in A | {result.inclusion_b_in_a * 100:.1f}% |",
        "",
        f"*Compared at: {result.compared_at.isoformat()}*",
    ]
    return "\n".join(lines)


def save_comparison(conn, result: ComparisonResult) -> int:
    """Save comparison result to database. Returns comparison ID."""
    with conn.cursor() as cur:
        # Create comparisons table if it doesn't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS result_set_comparisons (
                id SERIAL PRIMARY KEY,
                set_a_id INT NOT NULL,
                set_b_id INT NOT NULL,
                set_a_count INT NOT NULL,
                set_b_count INT NOT NULL,
                overlap_count INT NOT NULL,
                unique_to_a_count INT NOT NULL,
                unique_to_b_count INT NOT NULL,
                union_count INT NOT NULL,
                jaccard_similarity FLOAT NOT NULL,
                overlap_coefficient FLOAT NOT NULL,
                inclusion_a_in_b FLOAT NOT NULL,
                inclusion_b_in_a FLOAT NOT NULL,
                overlap_sample INT[],
                unique_to_a_sample INT[],
                unique_to_b_sample INT[],
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        cur.execute("""
            INSERT INTO result_set_comparisons (
                set_a_id, set_b_id, set_a_count, set_b_count,
                overlap_count, unique_to_a_count, unique_to_b_count, union_count,
                jaccard_similarity, overlap_coefficient,
                inclusion_a_in_b, inclusion_b_in_a,
                overlap_sample, unique_to_a_sample, unique_to_b_sample
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING id
        """, (
            result.set_a_id, result.set_b_id,
            result.set_a_count, result.set_b_count,
            result.overlap_count, result.unique_to_a_count,
            result.unique_to_b_count, result.union_count,
            result.jaccard_similarity, result.overlap_coefficient,
            result.inclusion_a_in_b, result.inclusion_b_in_a,
            result.overlap_sample, result.unique_to_a_sample,
            result.unique_to_b_sample,
        ))
        
        comparison_id = cur.fetchone()[0]
        conn.commit()
        
        return comparison_id


def get_what_is_new(
    conn,
    old_set_id: int,
    new_set_id: int,
    limit: int = 100
) -> Dict[str, Any]:
    """
    Get chunks that are new in set B compared to set A.
    
    Returns:
        Dictionary with new chunk IDs and their metadata.
    """
    old_set = fetch_result_set(conn, old_set_id)
    new_set = fetch_result_set(conn, new_set_id)
    
    new_chunks = new_set.chunk_ids - old_set.chunk_ids
    new_chunk_list = sorted(list(new_chunks))[:limit]
    
    # Fetch chunk metadata for new chunks
    chunk_info = []
    if new_chunk_list:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.id, cm.document_id, cm.collection_slug,
                       substring(c.text, 1, 200) as preview
                FROM chunks c
                LEFT JOIN chunk_metadata cm ON cm.chunk_id = c.id
                WHERE c.id = ANY(%s)
                ORDER BY c.id
            """, (new_chunk_list,))
            
            for row in cur.fetchall():
                chunk_info.append({
                    "chunk_id": row[0],
                    "document_id": row[1],
                    "collection_slug": row[2],
                    "preview": row[3],
                })
    
    return {
        "old_set_id": old_set_id,
        "new_set_id": new_set_id,
        "total_new_chunks": len(new_chunks),
        "returned_chunks": len(new_chunk_list),
        "new_chunks": chunk_info,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare two result sets and generate comparison reports."
    )
    parser.add_argument(
        "--set-a", type=int, required=True,
        help="ID of first result set (A)"
    )
    parser.add_argument(
        "--set-b", type=int, required=True,
        help="ID of second result set (B)"
    )
    parser.add_argument(
        "--format", "-f", 
        choices=["console", "json", "csv", "markdown"],
        default="console",
        help="Output format (default: console)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save comparison to database"
    )
    parser.add_argument(
        "--sample-size", type=int, default=10,
        help="Number of sample chunk IDs to include (default: 10)"
    )
    parser.add_argument(
        "--what-is-new", action="store_true",
        help="Show what's new in set B compared to set A"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        help="Output file (default: stdout)"
    )
    
    args = parser.parse_args()
    
    try:
        conn = get_conn()
        
        if args.what_is_new:
            # Special mode: show what's new
            result = get_what_is_new(conn, args.set_a, args.set_b)
            output = json.dumps(result, indent=2)
        else:
            # Standard comparison
            set_a = fetch_result_set(conn, args.set_a)
            set_b = fetch_result_set(conn, args.set_b)
            
            result = compare_result_sets(set_a, set_b, args.sample_size)
            
            # Format output
            if args.format == "console":
                output = format_console(result)
            elif args.format == "json":
                output = format_json(result)
            elif args.format == "csv":
                output = format_csv(result)
            elif args.format == "markdown":
                output = format_markdown(result)
            
            # Optionally save to database
            if args.save:
                comparison_id = save_comparison(conn, result)
                if args.format == "console":
                    output += f"\nSaved as comparison ID: {comparison_id}\n"
        
        # Write output
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"Output written to {args.output}")
        else:
            print(output)
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
