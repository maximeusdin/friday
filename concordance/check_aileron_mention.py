#!/usr/bin/env python3
"""Check the AILERON mention to see what's in the chunk."""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import DictCursor

def get_db_connection():
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = int(os.getenv("DB_PORT", "5432"))
    db_name = os.getenv("DB_NAME", "neh")
    db_user = os.getenv("DB_USER", "neh")
    db_pass = os.getenv("DB_PASS", "neh")
    return psycopg2.connect(host=db_host, port=db_port, dbname=db_name, user=db_user, password=db_pass)

conn = get_db_connection()
cur = conn.cursor(cursor_factory=DictCursor)

# Get the mention
cur.execute("""
    SELECT em.id, em.chunk_id, em.surface, em.start_char, em.end_char, em.surface_norm
    FROM entity_mentions em
    WHERE em.id = 5146
""")
mention = cur.fetchone()

if mention:
    print(f"Mention ID: {mention['id']}")
    print(f"Chunk ID: {mention['chunk_id']}")
    print(f"Surface: '{mention['surface']}'")
    print(f"Surface norm: '{mention['surface_norm']}'")
    print(f"Start char: {mention['start_char']}")
    print(f"End char: {mention['end_char']}")
    print()
    
    # Get the chunk text
    cur.execute("""
        SELECT c.text
        FROM chunks c
        WHERE c.id = %s
    """, (mention['chunk_id'],))
    chunk_row = cur.fetchone()
    
    if chunk_row:
        chunk_text = chunk_row['text']
        print(f"Chunk text length: {len(chunk_text)}")
        print()
        
        # Show context around the mention
        start = mention['start_char'] or 0
        end = mention['end_char'] or len(chunk_text)
        
        context_start = max(0, start - 50)
        context_end = min(len(chunk_text), end + 50)
        
        print(f"Context around mention (chars {start}-{end}):")
        print(f"  ...{chunk_text[context_start:start]}>>>{chunk_text[start:end]}<<<{chunk_text[end:context_end]}...")
        print()
        
        # Check what's actually at those positions
        actual_text = chunk_text[start:end] if start < len(chunk_text) and end <= len(chunk_text) else ""
        print(f"Actual text at positions {start}-{end}: '{actual_text}'")
        print()
        
        # Check if "AILERON [ELERON]" appears in the chunk
        if "AILERON [ELERON]" in chunk_text:
            pos = chunk_text.find("AILERON [ELERON]")
            print(f"Found 'AILERON [ELERON]' at position {pos}")
            print(f"  Text: '{chunk_text[pos:pos+16]}'")
        elif "AILERON [ELERON" in chunk_text:
            pos = chunk_text.find("AILERON [ELERON")
            print(f"Found 'AILERON [ELERON' (without closing bracket) at position {pos}")
            print(f"  Text: '{chunk_text[pos:pos+50]}'")
        else:
            print("'AILERON [ELERON]' not found in chunk text")
            # Try case-insensitive
            import re
            match = re.search(r'aileron\s*\[?\s*eleron', chunk_text, re.IGNORECASE)
            if match:
                print(f"Found case-insensitive match at position {match.start()}")
                print(f"  Text: '{chunk_text[match.start():match.end()+10]}'")

cur.close()
conn.close()
