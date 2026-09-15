"""
Document and Evidence endpoints
"""
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

import psycopg2

from app.services.db import get_conn
from app.services.evidence import build_evidence_refs_from_chunk

router = APIRouter()

# Configurable PDF root - defaults to data/ in repo root; always resolve to absolute
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PDF_ROOT = Path(os.getenv("PDF_ROOT", str(_REPO_ROOT / "data"))).resolve()
REPO_ROOT = _REPO_ROOT

# S3 configuration for production
# Set S3_PDF_BUCKET to enable S3 mode (e.g., "fridayarchive.org")
# S3 mirrors the local data/ folder structure, so source_ref like "data/raw/vassiliev/file.pdf"
# becomes "https://fridayarchive.org/data/raw/vassiliev/file.pdf"
S3_PDF_BUCKET = os.getenv("S3_PDF_BUCKET", "")
S3_PDF_REGION = os.getenv("S3_PDF_REGION", "us-west-1")


# =============================================================================
# Response Models
# =============================================================================

class Document(BaseModel):
    id: int
    collection_id: int
    collection_slug: Optional[str] = None
    collection_title: Optional[str] = None
    source_name: str
    source_ref: Optional[str] = None
    volume: Optional[str] = None
    page_count: Optional[int] = None
    pdf_url: Optional[str] = None
    metadata: Optional[dict] = None
    created_at: datetime


class EvidenceRef(BaseModel):
    document_id: int
    pdf_page: int
    chunk_id: Optional[int] = None
    span: Optional[dict] = None
    quote: Optional[str] = None
    why: Optional[str] = None


class EvidenceContext(BaseModel):
    chunk_text: Optional[str] = None
    page_text: Optional[str] = None


class EvidenceResponse(BaseModel):
    document: Document
    evidence_refs: list[EvidenceRef]
    context: EvidenceContext


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/documents/{document_id:int}", response_model=Document)
def get_document(document_id: int):
    """Get document metadata."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    d.id,
                    d.collection_id,
                    c.slug as collection_slug,
                    d.source_name,
                    d.source_ref,
                    d.volume,
                    d.metadata,
                    d.created_at,
                    c.title as collection_title
                FROM documents d
                JOIN collections c ON c.id = d.collection_id
                WHERE d.id = %s
                """,
                (document_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Document not found")
            
            # Get page count
            cur.execute(
                "SELECT COUNT(*) FROM pages WHERE document_id = %s",
                (document_id,),
            )
            page_count = cur.fetchone()[0]
            
            metadata = row[6] or {}
            source_ref = row[4]
            source_name = row[3]
            collection_slug = row[2]
            
            # Build direct PDF URL for frontend iframe
            pdf_url = _build_pdf_url_for_client(source_ref, source_name, collection_slug, document_id)
            
            return Document(
                id=row[0],
                collection_id=row[1],
                collection_slug=collection_slug,
                collection_title=row[8],
                source_name=source_name,
                source_ref=source_ref,
                volume=row[5],
                page_count=page_count,
                pdf_url=pdf_url,
                metadata=metadata,
                created_at=row[7],
            )
    finally:
        conn.close()


class Witness(BaseModel):
    appearance_seq: int
    witness_name: str
    start_page: int
    end_page: int
    page_count: Optional[int] = None
    testimony_date: Optional[str] = None
    examiner: Optional[str] = None


@router.get("/documents/{document_id:int}/witnesses", response_model=list[Witness])
def get_document_witnesses(document_id: int):
    """Witness index for a transcript document (empty if none / table absent)."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    SELECT appearance_seq, witness_name, start_page, end_page,
                           page_count, testimony_date, examiner
                    FROM document_witnesses
                    WHERE document_id = %s
                    ORDER BY appearance_seq
                    """,
                    (document_id,),
                )
                rows = cur.fetchall()
            except psycopg2.errors.UndefinedTable:
                conn.rollback()
                return []
        return [
            Witness(
                appearance_seq=r[0], witness_name=r[1], start_page=r[2], end_page=r[3],
                page_count=r[4], testimony_date=r[5], examiner=r[6],
            )
            for r in rows
        ]
    finally:
        conn.close()


@router.api_route("/documents/{document_id:int}/pdf", methods=["GET", "HEAD"])
def get_document_pdf(document_id: int):
    """
    Serve the PDF file for a document.
    
    In production (S3_PDF_BUCKET set), redirects to the S3 URL.
    S3 mirrors the local data/ structure, so source_ref is used directly.
    
    In development, serves from local filesystem.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.source_ref, d.source_name, c.slug
                FROM documents d
                JOIN collections c ON c.id = d.collection_id
                WHERE d.id = %s
                """,
                (document_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Document not found")
            
            source_ref, source_name, collection_slug = row
    finally:
        conn.close()
    
    # Production mode: redirect to S3
    # S3 mirrors local structure, so source_ref like "data/raw/vassiliev/file.pdf" 
    # becomes "https://fridayarchive.org/data/raw/vassiliev/file.pdf"
    if S3_PDF_BUCKET:
        s3_url = _build_s3_url(source_ref, source_name, collection_slug)
        return RedirectResponse(url=s3_url, status_code=302)
    
    # Development mode: serve from local filesystem
    pdf_path = _resolve_local_pdf_path(source_ref, source_name, collection_slug)
    
    if not pdf_path or not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF file not found: {source_ref or source_name}. Set PDF_ROOT or S3_PDF_BUCKET.",
        )
    
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=source_name,
        headers={
            "Content-Disposition": f'inline; filename="{source_name}"',
        },
    )


def _build_pdf_url_for_client(
    source_ref: Optional[str],
    source_name: str,
    collection_slug: str,
    document_id: int,
) -> str:
    """
    Return the direct PDF URL for the frontend to embed in an iframe.

    In production (S3_PDF_BUCKET set), this is the direct S3/CloudFront URL
    so the browser loads the PDF without going through an API redirect.
    In development, falls back to the local API route.
    """
    if S3_PDF_BUCKET:
        return _build_s3_url(source_ref, source_name, collection_slug)
    # Dev: use the API PDF route (serves file directly, no redirect)
    return f"/api/documents/{document_id}/pdf"


# Collections whose PDFs live in a pdf/ subfolder (data/raw/{slug}/pdf/*.pdf)
_COLLECTIONS_WITH_PDF_SUBFOLDER = frozenset({"vassiliev", "silvermaster"})


def _fallback_s3_path(collection_slug: str, source_name: str) -> str:
    """Derive S3 path when source_ref is absent. Mirrors local data/ layout."""
    if not collection_slug:
        return f"data/{source_name}"
    if collection_slug in _COLLECTIONS_WITH_PDF_SUBFOLDER:
        return f"data/raw/{collection_slug}/pdf/{source_name}"
    return f"data/raw/{collection_slug}/{source_name}"


def _build_s3_url(source_ref: Optional[str], source_name: str, collection_slug: str) -> str:
    """
    Build the S3 URL for a PDF using the path stored in the database.

    We use documents.source_ref as the canonical path when present (portable relative
    path like data/raw/venona/Venona London GRU.pdf or data/raw/vassiliev/pdf/...).
    S3 is assumed to mirror the local data/ folder structure.
    When source_ref is absent or lacks path info, we derive the path; collections
    in _COLLECTIONS_WITH_PDF_SUBFOLDER use data/raw/{slug}/pdf/{source_name}.
    """
    from urllib.parse import quote

    path = None

    if source_ref:
        sr_norm = str(source_ref).replace("\\", "/")
        data_idx = sr_norm.lower().find("/data/")
        if data_idx >= 0:
            path = sr_norm[data_idx + 1:]
        elif sr_norm.lower().startswith("data/"):
            path = sr_norm
        else:
            raw_idx = sr_norm.lower().find("/raw/")
            if raw_idx >= 0:
                path = "data" + sr_norm[raw_idx:]
            elif sr_norm.lower().startswith("raw/"):
                path = "data/" + sr_norm
            else:
                path = _fallback_s3_path(collection_slug, source_name)
    else:
        path = _fallback_s3_path(collection_slug, source_name)

    return _public_data_url(path)


def _public_data_url(path: str) -> str:
    """Public URL for an object under the PDF bucket, path like "data/raw/x.pdf"."""
    from urllib.parse import quote

    # URL-encode each segment (handles spaces in filenames, e.g. "Venona London GRU.pdf")
    path_encoded = "/".join(quote(part, safe="") for part in path.lstrip("/").split("/"))

    # S3_PDF_BUCKET is typically a domain like "fridayarchive.org" (CloudFront /
    # S3 website endpoint); a bare bucket name falls back to the REST URL.
    if "." in S3_PDF_BUCKET:
        return f"https://{S3_PDF_BUCKET}/{path_encoded}"
    else:
        return f"https://{S3_PDF_BUCKET}.s3.{S3_PDF_REGION}.amazonaws.com/{path_encoded}"


def _resolve_local_pdf_path(source_ref: Optional[str], source_name: str, collection_slug: str) -> Optional[Path]:
    """Resolve the local filesystem path for a PDF. Uses source_ref from DB first."""
    pdf_path: Optional[Path] = None

    def _try(candidate: Path) -> bool:
        nonlocal pdf_path
        if candidate.exists():
            pdf_path = candidate
            return True
        return False

    # Try source_ref first (absolute or relative) — use path from database
    if source_ref:
        sr_norm = str(source_ref).replace("\\", "/")
        sr_path = Path(sr_norm)

        if sr_path.is_absolute():
            if sr_path.exists():
                pdf_path = sr_path
            else:
                # Absolute path from different machine (e.g. ingest path); try extracting data/raw/... segment
                sr_lower = sr_norm.lower()
                if "/data/raw/" in sr_lower:
                    idx = sr_lower.find("/data/raw/")
                    segment = "raw/" + sr_norm[idx + len("/data/raw/"):].lstrip("/")
                    _try(PDF_ROOT / segment)
                elif "/raw/" in sr_lower:
                    idx = sr_lower.find("/raw/")
                    segment = sr_norm[idx:].lstrip("/")
                    _try(PDF_ROOT / segment)
        else:
            # Normalize source_ref: "../data/raw/..." (from ingest cwd) -> "raw/..."
            sr_rel = sr_norm
            for prefix in ("../data/", "data/"):
                if sr_rel.lower().startswith(prefix.lower()):
                    sr_rel = sr_rel[len(prefix):]
                    break

            # Ensure we have raw/ prefix for standard layout (data/raw/collection/...)
            if sr_rel and not sr_rel.lower().startswith("raw/"):
                sr_rel = "raw/" + sr_rel.lstrip("/")

            # Try relative to PDF_ROOT (data/)
            _try(PDF_ROOT / sr_rel)

            # Try relative to repo root (source_ref may be "data/raw/...")
            if not pdf_path:
                _try(REPO_ROOT / sr_norm)

            # When source_ref omits "pdf/" but file lives in pdf/ (e.g. vassiliev)
            if not pdf_path and collection_slug and source_name and "/pdf/" not in sr_rel.lower():
                sr_rel_pdf = f"raw/{collection_slug}/pdf/{source_name}"
                _try(PDF_ROOT / sr_rel_pdf)

            # When source_ref path segment differs from collection_slug
            if not pdf_path and collection_slug and source_name:
                _try(PDF_ROOT / "raw" / collection_slug / source_name)

    # Fallback: try to find by collection/source_name (including pdf/ subfolder)
    if not pdf_path and collection_slug and source_name:
        for candidate in [
            PDF_ROOT / "raw" / collection_slug / source_name,
            PDF_ROOT / "raw" / collection_slug / "pdf" / source_name,
            PDF_ROOT / "raw" / collection_slug / "PDF" / source_name,
            PDF_ROOT / "raw" / collection_slug / "pdfs" / source_name,
        ]:
            if _try(candidate):
                break

    # Final fallback: search by filename anywhere under PDF_ROOT
    if not pdf_path and source_name:
        found = _find_pdf_by_filename(PDF_ROOT, source_name)
        if found is not None:
            pdf_path = found

    return pdf_path


@lru_cache(maxsize=256)
def _find_pdf_by_filename(pdf_root: Path, filename: str) -> Optional[Path]:
    """
    Best-effort: locate a PDF by filename under PDF_ROOT.

    Cached to avoid repeated os.walk() per click.
    """
    filename_lower = filename.lower()
    root_str = str(pdf_root)
    for dirpath, _dirnames, filenames in os.walk(root_str):
        for f in filenames:
            if f.lower() == filename_lower:
                return Path(dirpath) / f
    return None


# =============================================================================
# Collection zips (bulk download)
#
# scripts/build_collection_zips.py publishes data/zips/<slug>.zip plus
# data/zips/manifest.json to the PDF bucket. These endpoints surface that
# manifest to the frontend; the zips themselves are served straight from
# S3/CloudFront in production.
# =============================================================================

_ZIPS_MANIFEST_PATH = "data/zips/manifest.json"
_ZIPS_CACHE_TTL_SECONDS = 300
_ZIPS_NEGATIVE_TTL_SECONDS = 60
_ZIPS_MANIFEST_MISSING = object()  # sentinel: "looked, not there" (cached briefly)
_zips_manifest_cache: dict = {"at": 0.0, "data": None}


class CollectionZipInfo(BaseModel):
    slug: str
    title: Optional[str] = None
    num_files: int
    total_bytes: Optional[int] = None
    zip_bytes: Optional[int] = None
    built_at: Optional[str] = None
    url: str


class CollectionZipsResponse(BaseModel):
    generated_at: Optional[str] = None
    collections: list[CollectionZipInfo] = []
    complete: Optional[CollectionZipInfo] = None


def _load_zips_manifest() -> Optional[dict]:
    """Manifest from S3 (prod) or local data/zips/ (dev), cached briefly.

    CloudFront serves the SPA index.html with status 200 for ANY missing path,
    so a JSON parse guard — not the status code — decides whether it exists.
    """
    import time

    now = time.monotonic()
    cached = _zips_manifest_cache["data"]
    if cached is not None:
        age = now - _zips_manifest_cache["at"]
        if cached is _ZIPS_MANIFEST_MISSING:
            # Negative result cached briefly: before the first zip publish this
            # endpoint is hit on every modal open, and each miss would otherwise
            # be a fresh blocking network fetch pinning a threadpool worker.
            if age < _ZIPS_NEGATIVE_TTL_SECONDS:
                return None
        elif age < _ZIPS_CACHE_TTL_SECONDS:
            return cached

    manifest = None
    if S3_PDF_BUCKET:
        import httpx

        try:
            resp = httpx.get(_public_data_url(_ZIPS_MANIFEST_PATH), timeout=5.0)
            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "")
                if "html" not in content_type.lower():
                    data = resp.json()
                    if isinstance(data, dict) and isinstance(data.get("collections"), list):
                        manifest = data
        except Exception:
            manifest = None
    else:
        local = PDF_ROOT / "zips" / "manifest.json"
        if local.exists():
            try:
                import json

                data = json.loads(local.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("collections"), list):
                    manifest = data
            except Exception:
                manifest = None

    _zips_manifest_cache["data"] = manifest if manifest is not None else _ZIPS_MANIFEST_MISSING
    _zips_manifest_cache["at"] = now
    return manifest


def _zip_url_for_client(slug: str, zip_key: str) -> str:
    if S3_PDF_BUCKET:
        return _public_data_url(zip_key)
    return f"/api/collection_zips/{slug}/download"


@router.get("/collection_zips", response_model=CollectionZipsResponse)
def get_collection_zips():
    """List downloadable per-collection zip archives (empty until first publish)."""
    manifest = _load_zips_manifest()
    if not manifest:
        return CollectionZipsResponse()

    collections = []
    for entry in manifest.get("collections", []):
        slug = entry.get("slug")
        zip_key = entry.get("zip_key")
        if not slug or not zip_key or not entry.get("num_files"):
            continue
        collections.append(CollectionZipInfo(
            slug=slug,
            title=entry.get("title"),
            num_files=entry["num_files"],
            total_bytes=entry.get("total_bytes"),
            zip_bytes=entry.get("zip_bytes"),
            built_at=entry.get("built_at"),
            url=_zip_url_for_client(slug, zip_key),
        ))

    complete = None
    comp = manifest.get("complete")
    if comp and comp.get("zip_key") and comp.get("num_files"):
        complete = CollectionZipInfo(
            slug="friday_complete",
            title="Complete archive (all collections)",
            num_files=comp["num_files"],
            total_bytes=comp.get("total_bytes"),
            zip_bytes=comp.get("zip_bytes"),
            built_at=comp.get("built_at"),
            url=_zip_url_for_client("friday_complete", comp["zip_key"]),
        )

    return CollectionZipsResponse(
        generated_at=manifest.get("generated_at"),
        collections=collections,
        complete=complete,
    )


@router.get("/collection_zips/{slug}/download")
def download_collection_zip(slug: str):
    """Dev-mode zip download (prod links point straight at S3/CloudFront)."""
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]+", slug):
        raise HTTPException(status_code=404, detail="Unknown collection zip")

    if S3_PDF_BUCKET:
        return RedirectResponse(url=_public_data_url(f"data/zips/{slug}.zip"), status_code=302)

    local = PDF_ROOT / "zips" / f"{slug}.zip"
    if not local.exists():
        raise HTTPException(status_code=404, detail="Zip not built; run scripts/build_collection_zips.py")
    return FileResponse(
        path=local,
        media_type="application/zip",
        filename=f"{slug}.zip",
    )


@router.get("/evidence", response_model=EvidenceResponse)
def get_evidence(
    document_id: int = Query(..., description="Document ID"),
    pdf_page: Optional[int] = Query(None, description="PDF page number (1-based)"),
    chunk_id: Optional[int] = Query(None, description="Chunk ID"),
):
    """
    Get evidence package for a document/page/chunk.
    
    Returns document metadata, evidence refs, and context text.
    """
    conn = get_conn()
    try:
        # Get document
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    d.id,
                    d.collection_id,
                    c.slug as collection_slug,
                    d.source_name,
                    d.source_ref,
                    d.volume,
                    d.metadata,
                    d.created_at,
                    c.title as collection_title
                FROM documents d
                JOIN collections c ON c.id = d.collection_id
                WHERE d.id = %s
                """,
                (document_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Document not found")
            
            # Get page count
            cur.execute(
                "SELECT COUNT(*) FROM pages WHERE document_id = %s",
                (document_id,),
            )
            page_count = cur.fetchone()[0]
            
            source_ref_ev = row[4]
            source_name_ev = row[3]
            collection_slug_ev = row[2]
            pdf_url_ev = _build_pdf_url_for_client(source_ref_ev, source_name_ev, collection_slug_ev, document_id)
            
            document = Document(
                id=row[0],
                collection_id=row[1],
                collection_slug=collection_slug_ev,
                collection_title=row[8],
                source_name=source_name_ev,
                source_ref=source_ref_ev,
                volume=row[5],
                page_count=page_count,
                pdf_url=pdf_url_ev,
                metadata=row[6] or {},
                created_at=row[7],
            )
        
        # Build evidence refs
        evidence_refs = []
        context = EvidenceContext()
        
        if chunk_id:
            # Get evidence from chunk
            refs = build_evidence_refs_from_chunk(conn, chunk_id)
            evidence_refs = [EvidenceRef(**ref) for ref in refs]
            
            # Get chunk text for context
            with conn.cursor() as cur:
                cur.execute("SELECT text FROM chunks WHERE id = %s", (chunk_id,))
                row = cur.fetchone()
                if row:
                    context.chunk_text = row[0]
        
        elif pdf_page:
            # Get page text
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT raw_text FROM pages 
                    WHERE document_id = %s AND pdf_page_number = %s
                    """,
                    (document_id, pdf_page),
                )
                row = cur.fetchone()
                if row:
                    context.page_text = row[0]
            
            evidence_refs = [
                EvidenceRef(document_id=document_id, pdf_page=pdf_page)
            ]
        
        return EvidenceResponse(
            document=document,
            evidence_refs=evidence_refs,
            context=context,
        )
    finally:
        conn.close()


# =============================================================================
# Collections Tree and Documents (for Scope panel)
# These use /collections_tree and /collections/... paths (no /documents/ prefix)
# so they cannot be shadowed by /documents/{document_id}.
# =============================================================================

class CollectionNodeResponse(BaseModel):
    id: int
    slug: str
    title: str
    description: Optional[str] = None
    document_count: int = 0
    chunk_count: Optional[int] = None


class DocumentNodeResponse(BaseModel):
    id: int
    source_name: str
    source_ref: Optional[str] = None
    volume: Optional[str] = None
    chunk_count: Optional[int] = None
    size_bytes: Optional[int] = None
    pdf_url: Optional[str] = None


@router.get("/collections_tree", response_model=list[CollectionNodeResponse])
def get_collections_tree(include_counts: int = Query(0, description="Set to 1 to include chunk counts")):
    """Return all collections with document counts."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if include_counts:
                cur.execute("""
                    SELECT c.id, c.slug, c.title, c.description,
                           COUNT(DISTINCT d.id) AS document_count,
                           COUNT(DISTINCT cm.chunk_id) AS chunk_count
                    FROM collections c
                    LEFT JOIN documents d ON d.collection_id = c.id
                    LEFT JOIN chunk_metadata cm ON cm.collection_slug = c.slug
                    GROUP BY c.id
                    ORDER BY c.title
                """)
            else:
                cur.execute("""
                    SELECT c.id, c.slug, c.title, c.description,
                           COUNT(d.id) AS document_count
                    FROM collections c
                    LEFT JOIN documents d ON d.collection_id = c.id
                    GROUP BY c.id
                    ORDER BY c.title
                """)
            rows = cur.fetchall()
            result = []
            for row in rows:
                node = CollectionNodeResponse(
                    id=row[0], slug=row[1], title=row[2],
                    description=row[3], document_count=row[4],
                )
                if include_counts and len(row) > 5:
                    node.chunk_count = row[5]
                result.append(node)
            return result
    finally:
        conn.close()


@router.get("/collections/{collection_id:int}/documents", response_model=list[DocumentNodeResponse])
def get_collection_documents(
    collection_id: int,
    include_counts: int = Query(0, description="Set to 1 to include chunk counts per document"),
):
    """Return documents for a single collection (lazy-loaded by UI on expand)."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT slug FROM collections WHERE id = %s", (collection_id,))
            col_row = cur.fetchone()
            if not col_row:
                raise HTTPException(status_code=404, detail="Collection not found")
            collection_slug = col_row[0]

            def _select(with_size: bool):
                size_col = "d.size_bytes" if with_size else "NULL"
                if include_counts:
                    cur.execute(f"""
                        SELECT d.id, d.source_name, d.source_ref, d.volume, {size_col},
                               COUNT(cm.chunk_id) AS chunk_count
                        FROM documents d
                        LEFT JOIN chunk_metadata cm ON cm.document_id = d.id
                        WHERE d.collection_id = %s
                        GROUP BY d.id
                        ORDER BY d.source_name
                    """, (collection_id,))
                else:
                    cur.execute(f"""
                        SELECT d.id, d.source_name, d.source_ref, d.volume, {size_col}
                        FROM documents d
                        WHERE d.collection_id = %s
                        ORDER BY d.source_name
                    """, (collection_id,))

            # size_bytes is backfilled by scripts/backfill_document_sizes.py;
            # stay compatible with DBs that predate the column.
            try:
                _select(with_size=True)
            except psycopg2.errors.UndefinedColumn:
                conn.rollback()
                _select(with_size=False)
            rows = cur.fetchall()

            result = []
            for row in rows:
                node = DocumentNodeResponse(
                    id=row[0], source_name=row[1],
                    source_ref=row[2], volume=row[3],
                    size_bytes=row[4],
                    pdf_url=_build_pdf_url_for_client(row[2], row[1], collection_slug, row[0]),
                )
                if include_counts and len(row) > 5:
                    node.chunk_count = row[5]
                result.append(node)
            return result
    finally:
        conn.close()
