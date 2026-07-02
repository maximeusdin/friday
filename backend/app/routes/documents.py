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

    path = path.lstrip("/")
    # URL-encode each segment (handles spaces in filenames, e.g. "Venona London GRU.pdf")
    path_encoded = "/".join(quote(part, safe="") for part in path.split("/"))
    
    # Build URL - S3_PDF_BUCKET is typically a domain like "fridayarchive.org"
    # which serves as an S3 website endpoint
    if "." in S3_PDF_BUCKET:
        # Domain-style bucket (S3 website hosting or CloudFront)
        # https://fridayarchive.org/data/raw/...
        return f"https://{S3_PDF_BUCKET}/{path_encoded}"
    else:
        # Standard S3 bucket URL
        # https://bucket-name.s3.us-west-1.amazonaws.com/data/raw/...
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
            cur.execute("SELECT 1 FROM collections WHERE id = %s", (collection_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Collection not found")

            if include_counts:
                cur.execute("""
                    SELECT d.id, d.source_name, d.source_ref, d.volume,
                           COUNT(cm.chunk_id) AS chunk_count
                    FROM documents d
                    LEFT JOIN chunk_metadata cm ON cm.document_id = d.id
                    WHERE d.collection_id = %s
                    GROUP BY d.id
                    ORDER BY d.source_name
                """, (collection_id,))
            else:
                cur.execute("""
                    SELECT d.id, d.source_name, d.source_ref, d.volume
                    FROM documents d
                    WHERE d.collection_id = %s
                    ORDER BY d.source_name
                """, (collection_id,))
            rows = cur.fetchall()
            result = []
            for row in rows:
                node = DocumentNodeResponse(
                    id=row[0], source_name=row[1],
                    source_ref=row[2], volume=row[3],
                )
                if include_counts and len(row) > 4:
                    node.chunk_count = row[4]
                result.append(node)
            return result
    finally:
        conn.close()
