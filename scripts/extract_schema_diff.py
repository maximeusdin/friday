#!/usr/bin/env python3
"""
Extract detailed schema info from neh (gold) for objects missing in friday_clean.
"""

import psycopg2


def main():
    conn = psycopg2.connect(host="localhost", port=5432, dbname="neh", user="neh", password="neh")
    cur = conn.cursor()

    # Missing tables
    missing_tables = [
        "concordance_sources",
        "concordance_entries", 
        "retrieval_config",
        "app_kv",
        "alias_stats",
        "entity_links",
        "entity_citations",
        "entity_merges",
    ]

    for tbl in missing_tables:
        print(f"\n{'='*60}")
        print(f"TABLE: {tbl}")
        print('='*60)
        cur.execute("""
            SELECT column_name, data_type, udt_name, is_nullable, column_default, character_maximum_length
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position
        """, (tbl,))
        for row in cur.fetchall():
            print(f"  {row[0]}: {row[1]} ({row[2]}), nullable={row[3]}, default={row[4]}, maxlen={row[5]}")

        # Constraints
        cur.execute("""
            SELECT conname, pg_get_constraintdef(con.oid, true)
            FROM pg_constraint con
            JOIN pg_class rel ON rel.oid = con.conrelid
            JOIN pg_namespace n ON n.oid = rel.relnamespace
            WHERE n.nspname = 'public' AND rel.relname = %s
        """, (tbl,))
        constraints = cur.fetchall()
        if constraints:
            print("  CONSTRAINTS:")
            for name, defn in constraints:
                print(f"    {name}: {defn}")

        # Indexes
        cur.execute("SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'public' AND tablename = %s", (tbl,))
        indexes = cur.fetchall()
        if indexes:
            print("  INDEXES:")
            for name, defn in indexes:
                print(f"    {name}: {defn}")

    # Missing columns in existing tables
    print(f"\n{'='*60}")
    print("MISSING COLUMNS IN EXISTING TABLES")
    print('='*60)

    # chunks columns
    chunks_extra_cols = ["clean_text", "clean_text_tsv", "embedding_model", "embedding_dim", 
                         "embedding_status", "embedded_at", "token_count"]
    print("\n-- chunks:")
    for col in chunks_extra_cols:
        cur.execute("""
            SELECT column_name, data_type, udt_name, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'chunks' AND column_name = %s
        """, (col,))
        row = cur.fetchone()
        if row:
            print(f"  {row[0]}: {row[1]} ({row[2]}), nullable={row[3]}, default={row[4]}")

    # chunk_metadata columns
    print("\n-- chunk_metadata:")
    cur.execute("""
        SELECT column_name, data_type, udt_name, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'chunk_metadata' AND column_name = 'content_type'
    """)
    row = cur.fetchone()
    if row:
        print(f"  {row[0]}: {row[1]} ({row[2]}), nullable={row[3]}, default={row[4]}")

    # entities columns in neh
    print("\n-- entities (in neh):")
    cur.execute("""
        SELECT column_name, data_type, udt_name, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'entities'
        ORDER BY ordinal_position
    """)
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]} ({row[2]}), nullable={row[3]}, default={row[4]}")
    
    # entities constraints
    cur.execute("""
        SELECT conname, pg_get_constraintdef(con.oid, true)
        FROM pg_constraint con
        JOIN pg_class rel ON rel.oid = con.conrelid
        WHERE rel.relname = 'entities'
    """)
    for name, defn in cur.fetchall():
        print(f"  CONSTRAINT {name}: {defn}")

    # entity_aliases columns in neh
    print("\n-- entity_aliases (in neh):")
    cur.execute("""
        SELECT column_name, data_type, udt_name, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'entity_aliases'
        ORDER BY ordinal_position
    """)
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]} ({row[2]}), nullable={row[3]}, default={row[4]}")

    # entity_mentions method CHECK constraint in neh
    print("\n-- entity_mentions CHECK constraint (in neh):")
    cur.execute("""
        SELECT conname, pg_get_constraintdef(con.oid, true)
        FROM pg_constraint con
        JOIN pg_class rel ON rel.oid = con.conrelid
        WHERE rel.relname = 'entity_mentions' AND contype = 'c'
    """)
    for name, defn in cur.fetchall():
        print(f"  {name}: {defn}")

    # Missing views
    print(f"\n{'='*60}")
    print("MISSING VIEWS")
    print('='*60)
    missing_views = ["concordance_current", "retrieval_chunks_v1", "retrieval_chunks_current", "retrieval_config_current"]
    for vw in missing_views:
        cur.execute("SELECT definition FROM pg_views WHERE schemaname = 'public' AND viewname = %s", (vw,))
        row = cur.fetchone()
        if row:
            print(f"\n-- {vw}:")
            print(row[0])

    # retrieval_runs index
    print(f"\n{'='*60}")
    print("MISSING INDEXES ON EXISTING TABLES")
    print('='*60)
    cur.execute("SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'public' AND tablename = 'retrieval_runs'")
    for name, defn in cur.fetchall():
        print(f"  {name}: {defn}")

    # mention_review_queue unique constraint
    print("\n-- mention_review_queue constraints in neh:")
    cur.execute("""
        SELECT conname, pg_get_constraintdef(con.oid, true)
        FROM pg_constraint con
        JOIN pg_class rel ON rel.oid = con.conrelid
        WHERE rel.relname = 'mention_review_queue'
    """)
    for name, defn in cur.fetchall():
        print(f"  {name}: {defn}")

    conn.close()


if __name__ == "__main__":
    main()
