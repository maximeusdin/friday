#!/usr/bin/env python3
"""Check existing schema for OCR pipeline implementation."""

import psycopg2

def main():
    conn = psycopg2.connect(host='localhost', port=5432, dbname='neh', user='neh', password='neh')
    cur = conn.cursor()

    # Check existing entity_aliases structure
    print('=== entity_aliases columns ===')
    cur.execute('''
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'entity_aliases'
        ORDER BY ordinal_position
    ''')
    for row in cur.fetchall():
        print('  %s: %s' % (row[0], row[1]))

    # Check chunks structure for text_quality
    print()
    print('=== chunks columns (quality-related) ===')
    cur.execute('''
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'chunks' AND column_name IN ('text_quality', 'alpha_ratio', 'garbage_score', 'is_trusted_text', 'text')
    ''')
    for row in cur.fetchall():
        print('  %s: %s' % (row[0], row[1]))

    # Sample of text_quality distribution
    print()
    print('=== text_quality distribution ===')
    cur.execute('''
        SELECT text_quality, COUNT(*) as cnt
        FROM chunks
        GROUP BY text_quality
        ORDER BY cnt DESC
    ''')
    for row in cur.fetchall():
        print('  %s: %d' % (row[0] or 'NULL', row[1]))

    # Check if pg_trgm is enabled
    print()
    print('=== pg_trgm status ===')
    cur.execute("SELECT * FROM pg_extension WHERE extname = 'pg_trgm'")
    row = cur.fetchone()
    print('  pg_trgm installed: %s' % ('YES' if row else 'NO'))

    # Check existing alias stats
    print()
    print('=== alias stats ===')
    cur.execute('SELECT COUNT(*) FROM entity_aliases')
    print('  Total aliases: %d' % cur.fetchone()[0])
    cur.execute('SELECT COUNT(DISTINCT entity_id) FROM entity_aliases')
    print('  Distinct entities: %d' % cur.fetchone()[0])
    cur.execute('SELECT COUNT(DISTINCT alias_norm) FROM entity_aliases')
    print('  Distinct alias_norms: %d' % cur.fetchone()[0])

    # Check collection slugs and doc quality proxy
    print()
    print('=== collections ===')
    cur.execute('''
        SELECT c.slug, COUNT(d.id) as doc_count
        FROM collections c
        LEFT JOIN documents d ON d.collection_id = c.id
        GROUP BY c.slug
        ORDER BY doc_count DESC
    ''')
    for row in cur.fetchall():
        print('  %s: %d docs' % (row[0], row[1]))

    # Check entity_surface_stats exists
    print()
    print('=== proposal corpus status ===')
    try:
        cur.execute('SELECT COUNT(*) FROM entity_surface_stats')
        print('  entity_surface_stats rows: %d' % cur.fetchone()[0])
        cur.execute('SELECT tier, COUNT(*) FROM entity_surface_tiers GROUP BY tier ORDER BY tier')
        for row in cur.fetchall():
            print('    Tier %d: %d surfaces' % (row[0], row[1]))
    except Exception as e:
        print('  entity_surface_stats not found: %s' % e)
        conn.rollback()

    conn.close()

if __name__ == '__main__':
    main()
