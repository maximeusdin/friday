#!/usr/bin/env python3
import psycopg2
import os

conn = psycopg2.connect(os.environ['DATABASE_URL'])
cur = conn.cursor()

tables = ['documents', 'chunks', 'chunk_pages', 'chunk_metadata', 'transcript_turns', 'chunk_turns', 'speaker_map', 'pages']

for table in tables:
    print(f"\n=== {table} columns ===")
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = %s ORDER BY ordinal_position", (table,))
    cols = cur.fetchall()
    if cols:
        for r in cols:
            print(f"  {r[0]}")
    else:
        print("  (table not found)")

conn.close()
