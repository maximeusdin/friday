"""Append the OCR-improvement note to the Silvermaster collection description (prod write)."""
import os
import psycopg2

NOTE = (
    "\n\nA note on the searchable text: the original OCR of these poorly scanned "
    "photostats was severely damaged, with roughly one word in ten wrong and proper "
    "names worst affected. In August 2026 the collection was re-transcribed from the "
    "page images by an ensemble of AI vision readers, substantially improving search "
    "and readability. The searchable text is a machine transcription; the scan shown "
    "alongside is authoritative, and quotations should be confirmed against it. The "
    "prior OCR remains searchable, and no search of material like this should be "
    "considered exhaustive."
)

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()
cur.execute("SELECT description FROM collections WHERE slug = 'silvermaster'")
desc = cur.fetchone()[0] or ""
if "machine transcription" in desc:
    print("note already present; no change")
else:
    cur.execute(
        "UPDATE collections SET description = %s WHERE slug = 'silvermaster'",
        (desc.rstrip() + NOTE,),
    )
    conn.commit()
    print("description updated")
