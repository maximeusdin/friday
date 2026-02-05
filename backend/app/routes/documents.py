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

from app.services.db import get_conn
from app.services.evidence import build_evidence_refs_from_chunk

router = APIRouter()

# Configurable PDF root - defaults to data/ in repo root
PDF_ROOT = Path(os.getenv("PDF_ROOT", Path(__file__).parent.parent.parent.parent / "data"))
REPO_ROOT = Path(__file__).parent.parent.parent.parent

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
    source_name: str
    source_ref: Optional[str] = None
    volume: Optional[str] = None
    page_count: Optional[int] = None
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

@router.get("/documents/{document_id}", response_model=Document)
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
                    d.created_at
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
            
            return Document(
                id=row[0],
                collection_id=row[1],
                collection_slug=row[2],
                source_name=row[3],
                source_ref=row[4],
                volume=row[5],
                page_count=page_count,
                metadata=metadata,
                created_at=row[7],
            )
    finally:
        conn.close()


@router.get("/documents/{document_id}/pdf")
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


def _build_s3_url(source_ref: Optional[str], source_name: str, collection_slug: str) -> str:
    """
    Build the S3 URL for a PDF file.
    
    S3 mirrors the local data/ structure exactly:
    - Local: data/raw/vassiliev/notebook1.pdf
    - S3:    s3://fridayarchive.org/data/raw/vassiliev/notebook1.pdf
    - URL:   https://fridayarchive.org/data/raw/vassiliev/notebook1.pdf
    
    source_ref in DB can be:
    - Relative: "data/raw/vassiliev/notebook1.pdf"
    - Absolute: "/path/to/friday/data/raw/vassiliev/notebook1.pdf"
    
    We extract the path starting from "data/" to match S3 structure.
    """
    from urllib.parse import quote
    
    path = None
    
    if source_ref:
        # Normalize path separators
        sr_norm = str(source_ref).replace("\\", "/")
        
        # Extract path starting from "data/" to match S3 structure
        # Handles both: "data/raw/..." and "/full/path/friday/data/raw/..."
        data_idx = sr_norm.lower().find("/data/")
        if data_idx >= 0:
            # Absolute path: extract from data/ onwards (including "data/")
            path = sr_norm[data_idx + 1:]  # +1 to skip the leading /
        elif sr_norm.lower().startswith("data/"):
            # Already relative from data/
            path = sr_norm
        else:
            # Doesn't contain data/ - maybe just the filename or raw/ path
            # Try to find raw/ pattern
            raw_idx = sr_norm.lower().find("/raw/")
            if raw_idx >= 0:
                path = "data" + sr_norm[raw_idx:]
            elif sr_norm.lower().startswith("raw/"):
                path = "data/" + sr_norm
            else:
                # Last resort: construct from collection/source_name
                path = f"data/raw/{collection_slug}/{source_name}" if collection_slug else f"data/{source_name}"
    else:
        # No source_ref: construct from collection/source_name
        path = f"data/raw/{collection_slug}/{source_name}" if collection_slug else f"data/{source_name}"
    
    # Ensure path doesn't start with /
    path = path.lstrip("/")
    
    # URL-encode the path components (handles spaces in filenames like "Rosenberg, Julius 01_text.pdf")
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
    """Resolve the local filesystem path for a PDF."""
    pdf_path: Optional[Path] = None
    
    # Try source_ref first (absolute or relative)
    if source_ref:
        sr_norm = str(source_ref).replace("\\", "/")
        sr_path = Path(source_ref)

        if sr_path.is_absolute():
            pdf_path = sr_path
        else:
            # Strip leading "data/" when present
            sr_rel = sr_norm
            if sr_rel.lower().startswith("data/"):
                sr_rel = sr_rel[5:]

            # Try relative to PDF_ROOT (data/)
            candidate = PDF_ROOT / sr_rel
            if candidate.exists():
                pdf_path = candidate

            # Try relative to repo root
            if not pdf_path or not pdf_path.exists():
                candidate = REPO_ROOT / sr_norm
                if candidate.exists():
                    pdf_path = candidate

            # Special-case: some collections store PDFs under an extra "pdf/" folder
            if (not pdf_path or not pdf_path.exists()) and collection_slug and source_name:
                needle_prefix = f"raw/{collection_slug}/"
                if sr_rel.lower().startswith(needle_prefix.lower()) and "/pdf/" not in sr_rel.lower():
                    sr_rel_pdf = f"raw/{collection_slug}/pdf/{source_name}"
                    candidate = PDF_ROOT / sr_rel_pdf
                    if candidate.exists():
                        pdf_path = candidate
    
    # Fallback: try to find by collection/source_name
    if not pdf_path or not pdf_path.exists():
        if collection_slug and source_name:
            for candidate in [
                PDF_ROOT / "raw" / collection_slug / source_name,
                PDF_ROOT / "raw" / collection_slug / "pdf" / source_name,
                PDF_ROOT / "raw" / collection_slug / "PDF" / source_name,
                PDF_ROOT / "raw" / collection_slug / "pdfs" / source_name,
            ]:
                if candidate.exists():
                    pdf_path = candidate
                    break

    # Final fallback: search by filename anywhere under PDF_ROOT
    if (not pdf_path or not pdf_path.exists()) and source_name:
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
                    d.created_at
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
            
            document = Document(
                id=row[0],
                collection_id=row[1],
                collection_slug=row[2],
                source_name=row[3],
                source_ref=row[4],
                volume=row[5],
                page_count=page_count,
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
