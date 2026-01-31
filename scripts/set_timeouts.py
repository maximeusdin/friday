#!/usr/bin/env python3
"""Set database timeout limits for the neh role."""

import os
import sys
import psycopg2

dsn = os.environ.get("DATABASE_URL")
if not dsn:
    print("ERROR: DATABASE_URL not set")
    sys.exit(1)

conn = psycopg2.connect(dsn)
cur = conn.cursor()

# Set timeouts
cur.execute("ALTER ROLE neh SET statement_timeout = '120s'")
cur.execute("ALTER ROLE neh SET lock_timeout = '10s'")
conn.commit()

print("Timeout settings updated.")

# Verify
cur.execute("""
    SELECT rolname, unnest(rolconfig) as setting 
    FROM pg_roles 
    WHERE rolname = 'neh'
""")
rows = cur.fetchall()
print()
print("Current role settings for neh:")
for role, setting in rows:
    print(f"  {setting}")

conn.close()
print()
print("Done. New connections will use these limits.")
