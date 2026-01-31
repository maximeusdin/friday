#!/usr/bin/env python3
"""
Export OCR Review File

Generates CSV/Excel files for offline adjudication of OCR variant clusters.

Output files:
- clusters.csv: One row per cluster with recommendation
- variants.csv: All variants with their cluster membership
- review_file.xlsx: Combined Excel with dropdown validation (optional)

Usage:
    python scripts/export_ocr_review_file.py --output-dir review_export/
    python scripts/export_ocr_review_file.py --output-dir review_export/ --limit 100 --format xlsx
"""

import argparse
import csv
import hashlib
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, '.')


DECISIONS = ['APPROVE_MERGE', 'APPROVE_NEW_ENTITY', 'BLOCK', 'DEFER', '']


def get_conn():
    return psycopg2.connect(
        host='localhost', port=5432, dbname='neh', user='neh', password='neh'
    )


def export_clusters_csv(conn, output_path: str, limit: Optional[int] = None) -> int:
    """Export clusters to CSV."""
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    limit_clause = f"LIMIT {limit}" if limit else ""
    
    cur.execute(f"""
        SELECT 
            c.cluster_id,
            c.proposed_canonical,
            c.canonical_entity_id,
            c.canonical_source,
            c.variant_count,
            c.total_mentions,
            c.doc_count,
            c.maps_to_multiple_entities,
            c.has_type_conflict,
            c.recommendation,
            c.priority_score,
            c.status,
            e.canonical_name as entity_name,
            e.entity_type
        FROM ocr_variant_clusters c
        LEFT JOIN entities e ON e.id = c.canonical_entity_id
        WHERE c.status = 'pending'
        ORDER BY c.priority_score DESC
        {limit_clause}
    """)
    
    rows = cur.fetchall()
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if rows:
            fieldnames = list(rows[0].keys()) + ['review_decision', 'reviewer_notes']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in rows:
                row_dict = dict(row)
                row_dict['review_decision'] = ''
                row_dict['reviewer_notes'] = ''
                writer.writerow(row_dict)
    
    return len(rows)


def export_variants_csv(conn, output_path: str, cluster_ids: Optional[List[str]] = None) -> int:
    """Export variants to CSV."""
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    if cluster_ids:
        cur.execute("""
            SELECT 
                v.cluster_id,
                v.variant_key,
                v.raw_examples,
                v.mention_count,
                v.doc_count,
                v.avg_quality_score,
                v.current_entity_id,
                c.proposed_canonical,
                c.recommendation
            FROM ocr_cluster_variants v
            JOIN ocr_variant_clusters c ON c.cluster_id = v.cluster_id
            WHERE v.cluster_id = ANY(%s)
            ORDER BY v.cluster_id, v.mention_count DESC
        """, (cluster_ids,))
    else:
        cur.execute("""
            SELECT 
                v.cluster_id,
                v.variant_key,
                v.raw_examples,
                v.mention_count,
                v.doc_count,
                v.avg_quality_score,
                v.current_entity_id,
                c.proposed_canonical,
                c.recommendation
            FROM ocr_cluster_variants v
            JOIN ocr_variant_clusters c ON c.cluster_id = v.cluster_id
            WHERE c.status = 'pending'
            ORDER BY c.priority_score DESC, v.cluster_id, v.mention_count DESC
        """)
    
    rows = cur.fetchall()
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                row_dict = dict(row)
                # Convert array to string for CSV
                if row_dict.get('raw_examples'):
                    row_dict['raw_examples'] = ' | '.join(row_dict['raw_examples'][:3])
                writer.writerow(row_dict)
    
    return len(rows)


def export_xlsx(conn, output_path: str, limit: Optional[int] = None) -> int:
    """Export to Excel with data validation."""
    try:
        import openpyxl
        from openpyxl.worksheet.datavalidation import DataValidation
    except ImportError:
        print("ERROR: openpyxl not installed. Run: pip install openpyxl")
        return 0
    
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    limit_clause = f"LIMIT {limit}" if limit else ""
    
    # Get clusters
    cur.execute(f"""
        SELECT 
            c.cluster_id,
            c.proposed_canonical,
            c.canonical_entity_id,
            c.canonical_source,
            c.variant_count,
            c.total_mentions,
            c.doc_count,
            c.maps_to_multiple_entities,
            c.has_type_conflict,
            c.recommendation,
            c.priority_score,
            e.canonical_name as entity_name,
            e.entity_type,
            (
                SELECT string_agg(v.variant_key, ' | ' ORDER BY v.mention_count DESC)
                FROM ocr_cluster_variants v
                WHERE v.cluster_id = c.cluster_id
                LIMIT 5
            ) as top_variants
        FROM ocr_variant_clusters c
        LEFT JOIN entities e ON e.id = c.canonical_entity_id
        WHERE c.status = 'pending'
        ORDER BY c.priority_score DESC
        {limit_clause}
    """)
    
    rows = cur.fetchall()
    
    if not rows:
        print("No pending clusters to export.")
        return 0
    
    # Create workbook
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Clusters"
    
    # Headers
    headers = [
        'cluster_id', 'proposed_canonical', 'entity_name', 'entity_type',
        'canonical_entity_id', 'variant_count', 'total_mentions', 'doc_count',
        'recommendation', 'priority_score', 'maps_to_multiple', 'type_conflict',
        'top_variants', 'review_decision', 'reviewer_notes'
    ]
    ws.append(headers)
    
    # Data
    for row in rows:
        ws.append([
            row['cluster_id'],
            row['proposed_canonical'],
            row['entity_name'],
            row['entity_type'],
            row['canonical_entity_id'],
            row['variant_count'],
            row['total_mentions'],
            row['doc_count'],
            row['recommendation'],
            float(row['priority_score']) if row['priority_score'] else 0,
            'YES' if row['maps_to_multiple_entities'] else '',
            'YES' if row['has_type_conflict'] else '',
            row['top_variants'],
            '',  # review_decision
            ''   # reviewer_notes
        ])
    
    # Add data validation for review_decision column
    dv = DataValidation(
        type="list",
        formula1='"APPROVE_MERGE,APPROVE_NEW_ENTITY,BLOCK,DEFER"',
        allow_blank=True
    )
    dv.error = "Please select a valid decision"
    dv.errorTitle = "Invalid Decision"
    ws.add_data_validation(dv)
    
    # Apply to review_decision column (column N = 14)
    decision_col = 14
    dv.add(f'N2:N{len(rows)+1}')
    
    # Adjust column widths
    for col in ws.columns:
        max_length = 0
        column = col[0].column_letter
        for cell in col:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        ws.column_dimensions[column].width = min(max_length + 2, 50)
    
    # Freeze header row
    ws.freeze_panes = 'A2'
    
    wb.save(output_path)
    return len(rows)


def compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash of file."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Export OCR review file')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--limit', type=int, help='Limit number of clusters')
    parser.add_argument('--format', choices=['csv', 'xlsx', 'both'], default='both',
                       help='Output format')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    conn = get_conn()
    
    print("=== Export OCR Review File ===")
    print(f"Output directory: {args.output_dir}")
    print(f"Format: {args.format}")
    print(f"Limit: {args.limit or 'none'}")
    print()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    exported = {}
    
    if args.format in ('csv', 'both'):
        # Export clusters CSV
        clusters_path = os.path.join(args.output_dir, f'clusters_{timestamp}.csv')
        count = export_clusters_csv(conn, clusters_path, args.limit)
        print(f"Exported {count} clusters to {clusters_path}")
        exported['clusters_csv'] = clusters_path
        
        # Export variants CSV
        variants_path = os.path.join(args.output_dir, f'variants_{timestamp}.csv')
        count = export_variants_csv(conn, variants_path)
        print(f"Exported {count} variants to {variants_path}")
        exported['variants_csv'] = variants_path
    
    if args.format in ('xlsx', 'both'):
        xlsx_path = os.path.join(args.output_dir, f'review_file_{timestamp}.xlsx')
        count = export_xlsx(conn, xlsx_path, args.limit)
        if count > 0:
            print(f"Exported {count} clusters to {xlsx_path}")
            exported['xlsx'] = xlsx_path
    
    # Write manifest
    manifest_path = os.path.join(args.output_dir, f'manifest_{timestamp}.txt')
    with open(manifest_path, 'w') as f:
        f.write(f"Export timestamp: {timestamp}\n")
        f.write(f"Limit: {args.limit or 'none'}\n\n")
        for key, path in exported.items():
            file_hash = compute_file_hash(path)
            f.write(f"{key}: {os.path.basename(path)}\n")
            f.write(f"  MD5: {file_hash}\n")
    
    print(f"\nManifest written to {manifest_path}")
    print("\nNext steps:")
    print("  1. Review the exported file(s)")
    print("  2. Fill in 'review_decision' column")
    print("  3. Run: python scripts/apply_ocr_adjudication.py <reviewed_file>")
    
    conn.close()


if __name__ == '__main__':
    main()
