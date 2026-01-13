import os
import hashlib
import psycopg2
from psycopg2.extras import Json

# Bump this when schema/pipeline changes and you want already-processed sources to re-run.
PIPELINE_VERSION = "v0_pages"

# Stable identity for this ingest "source" (file, transcript, decrypt, etc.)
SOURCE_KEY = "venona:sample_day2"

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "neh")
DB_USER = os.getenv("DB_USER", "neh")
DB_PASS = os.getenv("DB_PASS", "neh")


def connect():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )


def upsert_collection(cur, slug, title, description=None):
    cur.execute(
        """
        INSERT INTO collections (slug, title, description)
        VALUES (%s, %s, %s)
        ON CONFLICT (slug) DO UPDATE
        SET title = EXCLUDED.title,
            description = COALESCE(EXCLUDED.description, collections.description)
        RETURNING id
        """,
        (slug, title, description),
    )
    return cur.fetchone()[0]


def upsert_document(cur, collection_id, source_name, volume=None, source_ref=None, metadata=None):
    """
    Uses the UNIQUE(collection_id, source_name, volume_key) constraint (volume_key is generated in schema).
    """
    cur.execute(
        """
        INSERT INTO documents (collection_id, source_name, volume, source_ref, metadata)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (collection_id, source_name, volume_key) DO UPDATE
        SET source_ref = COALESCE(EXCLUDED.source_ref, documents.source_ref),
            metadata = documents.metadata || EXCLUDED.metadata
        RETURNING id
        """,
        (collection_id, source_name, volume, source_ref, Json(metadata or {})),
    )
    return cur.fetchone()[0]


def upsert_pages(cur, document_id, pages):
    """
    pages: list of dicts with keys:
      logical_page_label, pdf_page_number (nullable), page_seq, language, content_role, raw_text
    """
    for p in pages:
        cur.execute(
            """
            INSERT INTO pages (
              document_id, logical_page_label, pdf_page_number, page_seq,
              language, content_role, raw_text
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (document_id, logical_page_label) DO UPDATE
            SET
              pdf_page_number = EXCLUDED.pdf_page_number,
              page_seq = EXCLUDED.page_seq,
              language = EXCLUDED.language,
              content_role = EXCLUDED.content_role,
              raw_text = EXCLUDED.raw_text
            """,
            (
                document_id,
                p["logical_page_label"],
                p.get("pdf_page_number"),
                p["page_seq"],
                p["language"],
                p.get("content_role", "primary"),
                p["raw_text"],
            ),
        )


def ensure_ingest_runs_table(cur):
    """
    Creates ingest_runs if it doesn't exist yet.
    Keeps Day 2 friction low: you can also create this via a separate SQL file if you prefer.
    """
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_runs (
          id BIGSERIAL PRIMARY KEY,
          source_key TEXT NOT NULL UNIQUE,
          source_sha256 TEXT NOT NULL,
          pipeline_version TEXT NOT NULL,
          status TEXT NOT NULL CHECK (status IN ('success','failed','running')),
          started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          finished_at TIMESTAMPTZ,
          error TEXT
        )
        """
    )


def mark_run(cur, status, source_sha256=None, error=None):
    if status == "running":
        cur.execute(
            """
            INSERT INTO ingest_runs (source_key, source_sha256, pipeline_version, status, started_at)
            VALUES (%s, %s, %s, 'running', now())
            ON CONFLICT (source_key) DO UPDATE
            SET source_sha256 = EXCLUDED.source_sha256,
                pipeline_version = EXCLUDED.pipeline_version,
                status = 'running',
                started_at = now(),
                finished_at = NULL,
                error = NULL
            """,
            (SOURCE_KEY, source_sha256, PIPELINE_VERSION),
        )
    elif status == "success":
        cur.execute(
            """
            UPDATE ingest_runs
            SET status='success', finished_at=now()
            WHERE source_key=%s
            """,
            (SOURCE_KEY,),
        )
    elif status == "failed":
        cur.execute(
            """
            INSERT INTO ingest_runs (source_key, source_sha256, pipeline_version, status, started_at, finished_at, error)
            VALUES (%s, %s, %s, 'failed', now(), now(), %s)
            ON CONFLICT (source_key) DO UPDATE
            SET status='failed',
                finished_at=now(),
                error=%s,
                source_sha256=COALESCE(%s, ingest_runs.source_sha256),
                pipeline_version=%s
            """,
            (SOURCE_KEY, source_sha256 or "", PIPELINE_VERSION, error or "", error or "", source_sha256, PIPELINE_VERSION),
        )
    else:
        raise ValueError(f"Unknown status: {status}")


def should_run(cur, source_sha256):
    cur.execute(
        "SELECT source_sha256, pipeline_version, status FROM ingest_runs WHERE source_key = %s",
        (SOURCE_KEY,),
    )
    row = cur.fetchone()
    return (
        row is None
        or row[0] != source_sha256
        or row[1] != PIPELINE_VERSION
        or row[2] != "success"
    )


def main():
    # Replace this with ONE real source ASAP. For day 2, hardcode is fine.
    # Critical: preserve raw_text exactly as you have it (avoid aggressive cleanup/normalization).
    pages = []
    for i in range(1, 11):
        pages.append(
            {
                "logical_page_label": f"p. {i}",
                "pdf_page_number": None,
                "page_seq": i,
                "language": "en",
                "content_role": "primary",
                "raw_text": f"EXAMPLE PAGE {i}\n\nPaste real archival text here.\n",
            }
        )

    # Fingerprint input so changes to the underlying source force a re-run.
    source_blob = "".join(p["raw_text"] for p in pages)
    source_sha256 = hashlib.sha256(source_blob.encode("utf-8")).hexdigest()

    try:
        with connect() as conn:
            with conn.cursor() as cur:
                ensure_ingest_runs_table(cur)

                if not should_run(cur, source_sha256):
                    print("No-op: already ingested for this pipeline_version and source hash.")
                    return

                mark_run(cur, "running", source_sha256=source_sha256)

                collection_id = upsert_collection(
                    cur,
                    slug="venona",
                    title="Venona Decrypts",
                    description="Initial ingest for MVP validation",
                )

                doc_id = upsert_document(
                    cur,
                    collection_id=collection_id,
                    source_name="Venona Sample Decrypt (Day 2)",
                    volume=None,
                    source_ref=None,
                    metadata={
                        "ingested_by": "scripts/ingest_one.py",
                        "pipeline_version": PIPELINE_VERSION,
                    },
                )

                upsert_pages(cur, doc_id, pages)

                mark_run(cur, "success")

            conn.commit()

        print("Done. Ingested collection/document/pages (rerunnable + tracked).")

    except Exception as e:
        # Best-effort failure record; don't hide the original exception.
        try:
            with connect() as conn:
                with conn.cursor() as cur:
                    ensure_ingest_runs_table(cur)
                    mark_run(cur, "failed", source_sha256=source_sha256, error=str(e))
                conn.commit()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
