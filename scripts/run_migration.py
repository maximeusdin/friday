#!/usr/bin/env python3
"""Run a SQL migration file."""

import sys
import psycopg2

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_migration.py <migration_file.sql>")
        sys.exit(1)
    
    migration_file = sys.argv[1]
    
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        dbname='neh',
        user='neh',
        password='neh'
    )
    cur = conn.cursor()
    
    print(f"Running migration: {migration_file}")
    
    with open(migration_file, 'r', encoding='utf-8') as f:
        sql = f.read()
    
    try:
        cur.execute(sql)
        conn.commit()
        print("Migration completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        conn.close()

if __name__ == '__main__':
    main()
