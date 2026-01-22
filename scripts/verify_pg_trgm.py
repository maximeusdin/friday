#!/usr/bin/env python3
"""
Verify pg_trgm extension is enabled and working.

Usage:
    export DATABASE_URL="postgresql://neh:neh@localhost:5432/neh"
    python scripts/verify_pg_trgm.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2


def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL")
    return psycopg2.connect(dsn)


def main():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Check if extension exists
            cur.execute("""
                SELECT extname, extversion
                FROM pg_extension
                WHERE extname = 'pg_trgm'
            """)
            row = cur.fetchone()
            
            if not row:
                print("❌ pg_trgm extension is NOT installed")
                print("\nTo install, run:")
                print("  make enable-pg-trgm")
                sys.exit(1)
            
            extname, extversion = row
            print(f"✅ pg_trgm extension is installed (version {extversion})")
            
            # Test similarity function
            cur.execute("SELECT similarity('test', 'tast')")
            sim = cur.fetchone()[0]
            print(f"✅ similarity() function works: similarity('test', 'tast') = {sim:.4f}")
            
            # Test word_similarity function
            cur.execute("SELECT word_similarity('test', 'testing')")
            word_sim = cur.fetchone()[0]
            print(f"✅ word_similarity() function works: word_similarity('test', 'testing') = {word_sim:.4f}")
            
            # Test with OCR-like example
            cur.execute("SELECT similarity('silvermaster', 'silvermastre')")
            ocr_sim = cur.fetchone()[0]
            print(f"✅ OCR example: similarity('silvermaster', 'silvermastre') = {ocr_sim:.4f}")
            
            print("\n✅ All pg_trgm functions are working correctly!")
            print("   Ready for soft lexical retrieval (qv2_softlex)")
            
    finally:
        conn.close()


if __name__ == "__main__":
    main()
