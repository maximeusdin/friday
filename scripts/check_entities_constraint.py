#!/usr/bin/env python3
import os
import psycopg2

conn = psycopg2.connect(os.environ.get('DATABASE_URL', 'postgresql://localhost/neh'))
cur = conn.cursor()
cur.execute("""
  SELECT conname, pg_get_constraintdef(oid) 
  FROM pg_constraint 
  WHERE conrelid = 'entities'::regclass AND contype = 'u'
""")
for row in cur.fetchall():
    print(row)
conn.close()
