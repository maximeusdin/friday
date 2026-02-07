'use client';

import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { EvidenceRef } from '@/types/api';

const ZOOM_LEVELS = [50, 75, 100, 125, 150, 200];

interface EvidenceViewerProps {
  evidence: EvidenceRef | null;
  onClose: () => void;
}

export function EvidenceViewer({ evidence, onClose }: EvidenceViewerProps) {
  const [currentPage, setCurrentPage] = useState(1);
  const [zoom, setZoom] = useState(125); // default slightly above 100 for readability
  const [pdfOk, setPdfOk] = useState<boolean | null>(null);
  const [pdfError, setPdfError] = useState<string | null>(null);

  // Fetch document metadata (includes pdf_url — the direct CDN/S3 link)
  const { data: document, isLoading: docLoading } = useQuery({
    queryKey: ['document', evidence?.document_id],
    queryFn: () => api.getDocument(evidence!.document_id),
    enabled: !!evidence,
  });

  // Update current page when evidence changes
  useEffect(() => {
    if (evidence?.pdf_page) {
      setCurrentPage(evidence.pdf_page);
    }
  }, [evidence?.pdf_page]);

  // Resolve the direct PDF URL from document metadata (CDN/S3).
  // This avoids the cross-origin API redirect that breaks fetch and iframes.
  // Falls back to the API endpoint if pdf_url is not available (older backend).
  const pdfBaseUrl = document?.pdf_url || api.getDocumentPdfUrl(evidence?.document_id ?? 0);
  const resolvedBaseUrl = pdfBaseUrl.startsWith('http')
    ? pdfBaseUrl
    : `${typeof window !== 'undefined' ? window.location.origin : ''}${pdfBaseUrl}`;

  // Best-effort: detect missing PDFs and show a friendly message.
  // Uses the resolved direct URL to avoid cross-origin redirect issues
  // that cause opaque "network error" failures in fetch.
  useEffect(() => {
    let cancelled = false;
    async function checkPdf() {
      // Wait for document metadata so we have the direct pdf_url
      if (!evidence || !document) return;
      setPdfOk(null);
      setPdfError(null);
      try {
        // Use the direct CDN/S3 URL (same-origin or CORS-enabled) instead of
        // the API redirect endpoint which causes cross-origin redirect failures.
        const checkUrl = document.pdf_url || api.getDocumentPdfUrl(evidence.document_id);
        const res = await fetch(checkUrl, { method: 'HEAD' });
        if (cancelled) return;
        if (!res.ok) {
          setPdfOk(false);
          setPdfError(`PDF unavailable (HTTP ${res.status}).`);
          return;
        }
        setPdfOk(true);
      } catch (e) {
        if (cancelled) return;
        setPdfOk(false);
        setPdfError('PDF unavailable (network error).');
      }
    }
    checkPdf();
    return () => {
      cancelled = true;
    };
  }, [evidence?.document_id, document]);

  // Zoom handlers
  const handleZoomIn = () => {
    const idx = ZOOM_LEVELS.indexOf(zoom);
    if (idx < ZOOM_LEVELS.length - 1) setZoom(ZOOM_LEVELS[idx + 1]);
  };
  const handleZoomOut = () => {
    const idx = ZOOM_LEVELS.indexOf(zoom);
    if (idx > 0) setZoom(ZOOM_LEVELS[idx - 1]);
  };
  const handleZoomReset = () => setZoom(100);

  if (!evidence) {
    return (
      <div className="empty-state">
        <p>No evidence selected</p>
        <p className="text-sm">Click a citation to view the source</p>
      </div>
    );
  }

  // #page=N is supported by Chrome/Firefox/Edge built-in PDF viewers.
  // We remount the iframe via `key` to force page changes.
  const pdfUrl = `${resolvedBaseUrl}#page=${currentPage}`;

  const handleOpenNewTab = () => {
    window.open(pdfUrl, '_blank', 'noopener,noreferrer');
    onClose(); // Return to chat
  };

  return (
    <div className="pdf-viewer">
      {/* Toolbar: navigation + actions */}
      <div className="pdf-toolbar">
        <button className="btn-back-to-chat" onClick={onClose}>
          ← Back to Chat
        </button>
        
        <div className="pdf-toolbar-separator" />
        
        <button
          className="btn-secondary"
          onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
          disabled={currentPage <= 1}
        >
          ‹ Prev
        </button>
        
        <span className="pdf-page-info">
          Page {currentPage}
          {document?.page_count && ` / ${document.page_count}`}
        </span>
        
        <button
          className="btn-secondary"
          onClick={() => setCurrentPage((p) => p + 1)}
          disabled={document?.page_count !== undefined && currentPage >= document.page_count}
        >
          Next ›
        </button>
        
        <div className="pdf-toolbar-separator" />

        {/* Zoom controls */}
        <div className="zoom-controls">
          <button
            className="zoom-btn"
            onClick={handleZoomOut}
            disabled={zoom <= ZOOM_LEVELS[0]}
            title="Zoom out"
          >
            −
          </button>
          <button
            className="zoom-level"
            onClick={handleZoomReset}
            title="Reset to 100%"
          >
            {zoom}%
          </button>
          <button
            className="zoom-btn"
            onClick={handleZoomIn}
            disabled={zoom >= ZOOM_LEVELS[ZOOM_LEVELS.length - 1]}
            title="Zoom in"
          >
            +
          </button>
        </div>

        <div className="flex-1" />
        
        <a
          href={resolvedBaseUrl}
          download
          className="btn-secondary"
          style={{ textDecoration: 'none' }}
          title="Download PDF"
        >
          ↓ Download
        </a>
        
        <button
          className="btn-secondary"
          onClick={handleOpenNewTab}
          title="Open PDF in new tab and return to chat"
        >
          ↗ Open in New Tab
        </button>
      </div>

      {/* Document info */}
      {document && (
        <div
          style={{
            padding: 'var(--spacing-sm) var(--spacing-md)',
            background: 'var(--color-bg-secondary)',
            borderBottom: '1px solid var(--color-border)',
            fontSize: '12px',
          }}
        >
          <strong>{document.source_name}</strong>
          {document.collection_slug && (
            <span className="text-muted"> · {document.collection_slug}</span>
          )}
        </div>
      )}

      {/* Quote preview (if available) */}
      {evidence.quote && (
        <div
          style={{
            padding: 'var(--spacing-md)',
            background: 'var(--color-highlight)',
            borderBottom: '1px solid var(--color-border)',
            fontSize: '13px',
            fontStyle: 'italic',
          }}
        >
          &ldquo;{evidence.quote}&rdquo;
          {evidence.why && (
            <div className="text-sm text-muted mt-sm">
              Relevance: {evidence.why}
            </div>
          )}
        </div>
      )}

      {/* PDF embed: direct URL from CDN/S3, no API redirect */}
      <div className="pdf-container">
        {docLoading || pdfOk === null ? (
          <div className="loading">Loading document...</div>
        ) : pdfOk === false ? (
          <div className="empty-state">
            <p>PDF file missing</p>
            {pdfError && <p className="text-sm text-muted">{pdfError}</p>}
            {document && (
              <div className="card" style={{ textAlign: 'left', maxWidth: 520 }}>
                <div className="text-sm"><strong>Document</strong>: {document.source_name}</div>
                {document.collection_slug && (
                  <div className="text-sm text-muted">Collection: {document.collection_slug}</div>
                )}
                {document.source_ref && (
                  <div className="text-sm text-muted">source_ref: {document.source_ref}</div>
                )}
              </div>
            )}
            <a
              href={resolvedBaseUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="btn-secondary"
              style={{ textDecoration: 'none' }}
            >
              Try opening PDF directly ↗
            </a>
          </div>
        ) : (
          <div
            className="pdf-zoom-wrapper"
            style={{ width: `${zoom}%`, height: '100%' }}
          >
            <iframe
              key={`${evidence.document_id}-${currentPage}`}
              src={pdfUrl}
              referrerPolicy="no-referrer"
              style={{
                width: '100%',
                height: '100%',
                border: 'none',
              }}
              title={document?.source_name || 'PDF Document'}
            />
          </div>
        )}
      </div>
    </div>
  );
}
