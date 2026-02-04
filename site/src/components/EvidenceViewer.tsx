'use client';

import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { EvidenceRef } from '@/types/api';

interface EvidenceViewerProps {
  evidence: EvidenceRef | null;
  onClose: () => void;
}

export function EvidenceViewer({ evidence, onClose }: EvidenceViewerProps) {
  const [currentPage, setCurrentPage] = useState(1);
  const [pdfOk, setPdfOk] = useState<boolean | null>(null);
  const [pdfError, setPdfError] = useState<string | null>(null);

  // Fetch document metadata
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

  // Best-effort: detect missing PDFs and show a friendly message
  useEffect(() => {
    let cancelled = false;
    async function checkPdf() {
      if (!evidence) return;
      setPdfOk(null);
      setPdfError(null);
      try {
        const res = await fetch(api.getDocumentPdfUrl(evidence.document_id), { method: 'HEAD' });
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
  }, [evidence?.document_id]);

  if (!evidence) {
    return (
      <div className="empty-state">
        <p>No evidence selected</p>
        <p className="text-sm">Click a citation to view the source</p>
      </div>
    );
  }

  // Many built-in PDF viewers only honor `#page=` on initial load and may ignore
  // hash-only updates. We force a reload by changing the URL with a harmless query param
  // and remounting the iframe via `key`.
  const pdfBaseUrl = api.getDocumentPdfUrl(evidence.document_id);
  const pdfUrl = `${pdfBaseUrl}?page=${currentPage}#page=${currentPage}`;

  return (
    <div className="pdf-viewer">
      {/* Toolbar */}
      <div className="pdf-toolbar">
        <button className="btn-secondary" onClick={onClose}>
          ← Back
        </button>
        
        <div className="flex-1" />
        
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
        
        <div className="flex-1" />
        
        <a
          href={pdfUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="btn-secondary"
          style={{ textDecoration: 'none' }}
        >
          Open ↗
        </a>
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
          "{evidence.quote}"
          {evidence.why && (
            <div className="text-sm text-muted mt-sm">
              Relevance: {evidence.why}
            </div>
          )}
        </div>
      )}

      {/* PDF embed */}
      <div className="pdf-container">
        {docLoading ? (
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
              href={api.getDocumentPdfUrl(evidence.document_id)}
              target="_blank"
              rel="noopener noreferrer"
              className="btn-secondary"
              style={{ textDecoration: 'none' }}
            >
              Open raw PDF endpoint ↗
            </a>
          </div>
        ) : (
          <iframe
            key={pdfUrl}
            src={pdfUrl}
            style={{
              width: '100%',
              height: '100%',
              border: 'none',
            }}
            title={document?.source_name || 'PDF Document'}
          />
        )}
      </div>
    </div>
  );
}
