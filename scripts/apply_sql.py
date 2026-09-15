"""Apply a .sql file via psycopg2 (for machines without psql).

Usage:
    DATABASE_URL=... python scripts/apply_sql.py migrations/0075_ocr_variant_skeleton.sql
"""
import os
import sys

import psycopg2

if len(sys.argv) != 2:
    sys.exit("usage: apply_sql.py <file.sql>")

path = sys.argv[1]
sql = open(path, encoding="utf-8").read()

conn = psycopg2.connect(os.environ["DATABASE_URL"])
try:
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    print(f"applied {path}")
finally:
    conn.close()
