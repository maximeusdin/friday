'use client';

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/TextLayer.css';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import { api } from '@/lib/api';
import type { EvidenceRef } from '@/types/api';

// Wire up the PDF.js worker. `pdfjs.version` is the EXACT pdfjs-dist version react-pdf uses
// (its own bundled copy), so pinning the worker to that version can never drift from the API —
// this is what previously broke ("API version 4.8.69 does not match Worker version 4.10.38"),
// because a build-time copy of a different top-level pdfjs-dist got served from /public and
// browser-cached. Loading the version-matched worker guarantees they agree on every deploy.
pdfjs.GlobalWorkerOptions.workerSrc =
  `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

const ZOOM_LEVELS = [50, 75, 100, 125, 150, 200];

// CSS Custom Highlight API — modern browsers (Chrome/Edge/Safari) let us paint
// search highlights over the text layer without mutating the DOM.
const HIGHLIGHT_SUPPORTED =
  typeof window !== 'undefined' &&
  typeof (window as unknown as { Highlight?: unknown }).Highlight !== 'undefined' &&
  typeof CSS !== 'undefined' &&
  'highlights' in CSS;

interface EvidenceViewerProps {
  evidence: EvidenceRef | null;
  onClose: () => void;
  /** Context-aware label for back button, e.g. "Back to results" when opened from Search */
  backLabel?: string;
}

interface TextNodeEntry {
  node: Text;
  start: number;
}

/** Find the text node + offset that contains a given index in the concatenated string. */
function locate(nodes: TextNodeEntry[], offset: number): { node: Text; off: number } {
  for (const e of nodes) {
    const len = e.node.nodeValue?.length ?? 0;
    if (offset >= e.start && offset <= e.start + len) {
      return { node: e.node, off: offset - e.start };
    }
  }
  const last = nodes[nodes.length - 1];
  return { node: last.node, off: last.node.nodeValue?.length ?? 0 };
}

export function EvidenceViewer({ evidence, onClose, backLabel = 'Back to Chat' }: EvidenceViewerProps) {
  const [currentPage, setCurrentPage] = useState(1);
  const [numPages, setNumPages] = useState<number | null>(null);
  const [zoom, setZoom] = useState(125); // default slightly above 100 for readability
  const [loadError, setLoadError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  // --- Find-in-document state ---
  const [findOpen, setFindOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [matches, setMatches] = useState<{ page: number }[]>([]);
  const [activeMatchIdx, setActiveMatchIdx] = useState(-1);
  const [searching, setSearching] = useState(false);

  const pdfRef = useRef<Awaited<ReturnType<typeof pdfjs.getDocument>['promise']> | null>(null);
  const textCache = useRef<Map<number, string>>(new Map());
  const pageWrapRef = useRef<HTMLDivElement>(null);
  const findInputRef = useRef<HTMLInputElement>(null);

  // react-pdf touches browser-only APIs; defer rendering until mounted so the
  // static export prerender (and any SSR) stays safe.
  useEffect(() => setMounted(true), []);

  // Fetch document metadata (includes pdf_url — the direct CDN/S3 link)
  const { data: document, isLoading: docLoading } = useQuery({
    queryKey: ['document', evidence?.document_id],
    queryFn: () => api.getDocument(evidence!.document_id),
    enabled: !!evidence,
  });

  // Witness index (present for grand jury / hearing transcript documents)
  const { data: witnesses } = useQuery({
    queryKey: ['document-witnesses', evidence?.document_id],
    queryFn: () => api.getDocumentWitnesses(evidence!.document_id),
    enabled: !!evidence,
  });
  const [showWitnesses, setShowWitnesses] = useState(false);

  // Update current page when evidence changes
  useEffect(() => {
    if (evidence?.pdf_page) {
      setCurrentPage(evidence.pdf_page);
    }
  }, [evidence?.pdf_page]);

  // Reset per-document state when the source changes
  useEffect(() => {
    pdfRef.current = null;
    textCache.current = new Map();
    setNumPages(null);
    setLoadError(null);
    setMatches([]);
    setActiveMatchIdx(-1);
    setQuery('');
  }, [evidence?.document_id]);

  // Resolve the direct PDF URL from document metadata (CDN/S3).
  // This avoids the cross-origin API redirect that breaks fetch.
  // Falls back to the API endpoint if pdf_url is not available (older backend).
  const pdfBaseUrl = document?.pdf_url || api.getDocumentPdfUrl(evidence?.document_id ?? 0);
  const resolvedBaseUrl = pdfBaseUrl.startsWith('http')
    ? pdfBaseUrl
    : `${typeof window !== 'undefined' ? window.location.origin : ''}${pdfBaseUrl}`;

  // Stable `file` object so react-pdf doesn't re-fetch on every render.
  const fileProp = useMemo(() => ({ url: resolvedBaseUrl }), [resolvedBaseUrl]);

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

  // --- Find: scan the whole document for matches (lazy + cached per page) ---
  const runScan = useCallback(async (q: string) => {
    const pdf = pdfRef.current;
    if (!pdf || !q.trim()) {
      setMatches([]);
      setActiveMatchIdx(-1);
      CSS_clearHighlights();
      return;
    }
    setSearching(true);
    const needle = q.toLowerCase();
    const out: { page: number }[] = [];
    try {
      for (let p = 1; p <= pdf.numPages; p++) {
        let t = textCache.current.get(p);
        if (t === undefined) {
          const page = await pdf.getPage(p);
          const tc = await page.getTextContent();
          t = tc.items.map((it) => ('str' in it ? it.str : '')).join('');
          textCache.current.set(p, t);
        }
        const lower = t.toLowerCase();
        let idx = lower.indexOf(needle);
        while (idx !== -1) {
          out.push({ page: p });
          idx = lower.indexOf(needle, idx + needle.length);
        }
      }
    } finally {
      setSearching(false);
    }
    setMatches(out);
    if (out.length) {
      setActiveMatchIdx(0);
      setCurrentPage(out[0].page);
    } else {
      setActiveMatchIdx(-1);
    }
  }, []);

  // Debounce the scan as the user types
  useEffect(() => {
    if (!mounted) return;
    const h = setTimeout(() => runScan(query), 200);
    return () => clearTimeout(h);
  }, [query, numPages, mounted, runScan]);

  // --- Paint highlights on the currently-rendered page's text layer ---
  const applyHighlights = useCallback(() => {
    if (!HIGHLIGHT_SUPPORTED) return;
    const css = CSS as unknown as { highlights: Map<string, unknown> };
    css.highlights.delete('pdf-find');
    css.highlights.delete('pdf-find-active');
    const layer = pageWrapRef.current?.querySelector('.react-pdf__Page__textContent');
    if (!layer || !query.trim()) return;

    const needle = query.toLowerCase();
    const walker = window.document.createTreeWalker(layer, NodeFilter.SHOW_TEXT);
    const nodes: TextNodeEntry[] = [];
    let full = '';
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const text = n as Text;
      nodes.push({ node: text, start: full.length });
      full += text.nodeValue ?? '';
    }
    if (!nodes.length) return;

    const lower = full.toLowerCase();
    const ranges: Range[] = [];
    let idx = lower.indexOf(needle);
    while (idx !== -1) {
      const a = locate(nodes, idx);
      const b = locate(nodes, idx + needle.length);
      const r = window.document.createRange();
      r.setStart(a.node, a.off);
      r.setEnd(b.node, b.off);
      ranges.push(r);
      idx = lower.indexOf(needle, idx + needle.length);
    }
    if (!ranges.length) return;

    // Which on-page occurrence is the active match?
    const activeOnPage = activeMatchIdx >= 0 && matches[activeMatchIdx]?.page === currentPage;
    const ordinal = activeOnPage
      ? matches.slice(0, activeMatchIdx).filter((m) => m.page === currentPage).length
      : -1;
    const activeRange = activeOnPage ? ranges[ordinal] : null;
    const rest = activeRange ? ranges.filter((r) => r !== activeRange) : ranges;

    const HL = (window as unknown as { Highlight: new (...r: Range[]) => unknown }).Highlight;
    if (rest.length) css.highlights.set('pdf-find', new HL(...rest));
    if (activeRange) {
      css.highlights.set('pdf-find-active', new HL(activeRange));
      const el = activeRange.startContainer.parentElement;
      el?.scrollIntoView({ block: 'center', inline: 'nearest' });
    }
  }, [query, activeMatchIdx, matches, currentPage]);

  // Re-paint when match selection or page changes (text layer may already be rendered)
  useEffect(() => {
    applyHighlights();
  }, [applyHighlights]);

  // Clear highlights on unmount / document change
  useEffect(() => () => CSS_clearHighlights(), [evidence?.document_id]);

  // --- Find navigation ---
  const gotoMatch = useCallback((delta: number) => {
    if (!matches.length) return;
    const n = (activeMatchIdx + delta + matches.length) % matches.length;
    setActiveMatchIdx(n);
    setCurrentPage(matches[n].page);
  }, [matches, activeMatchIdx]);

  const openFind = useCallback(() => {
    setFindOpen(true);
    setTimeout(() => findInputRef.current?.focus(), 0);
  }, []);

  const closeFind = useCallback(() => {
    setFindOpen(false);
    setQuery('');
    setMatches([]);
    setActiveMatchIdx(-1);
    CSS_clearHighlights();
  }, []);

  // Keyboard: Ctrl/Cmd+F opens our find bar (overrides browser find), Esc closes
  useEffect(() => {
    if (!mounted) return;
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'f') {
        e.preventDefault();
        openFind();
      } else if (e.key === 'Escape' && findOpen) {
        closeFind();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [mounted, findOpen, openFind, closeFind]);

  if (!evidence) {
    return (
      <div className="empty-state">
        <p>No evidence selected</p>
        <p className="text-sm">Click a citation to view the source</p>
      </div>
    );
  }

  const pdfUrl = `${resolvedBaseUrl}#page=${currentPage}`;
  const handleOpenNewTab = () => {
    window.open(pdfUrl, '_blank', 'noopener,noreferrer');
    onClose(); // Return to chat
  };

  const totalPages = numPages ?? document?.page_count;

  return (
    <div className="pdf-viewer">
      {/* Toolbar: navigation + actions */}
      <div className="pdf-toolbar">
        <button className="btn-back-to-chat" onClick={onClose}>
          ← {backLabel}
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
          {totalPages && ` / ${totalPages}`}
        </span>

        <button
          className="btn-secondary"
          onClick={() => setCurrentPage((p) => p + 1)}
          disabled={totalPages !== undefined && totalPages !== null && currentPage >= totalPages}
        >
          Next ›
        </button>

        <div className="pdf-toolbar-separator" />

        {/* Zoom controls */}
        <div className="zoom-controls">
          <button className="zoom-btn" onClick={handleZoomOut} disabled={zoom <= ZOOM_LEVELS[0]} title="Zoom out">
            −
          </button>
          <button className="zoom-level" onClick={handleZoomReset} title="Reset to 100%">
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

        <div className="pdf-toolbar-separator" />

        <button
          className={`btn-secondary${findOpen ? ' btn-active' : ''}`}
          onClick={() => (findOpen ? closeFind() : openFind())}
          title="Find in document (Ctrl+F)"
        >
          ⌕ Find
        </button>

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

      {/* Docked find bar — sits in the toolbar area, never overlaps the page */}
      {findOpen && (
        <div className="pdf-find-bar">
          <span className="pdf-find-icon">⌕</span>
          <input
            ref={findInputRef}
            type="text"
            className="pdf-find-input"
            placeholder="Find in document…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                gotoMatch(e.shiftKey ? -1 : 1);
              } else if (e.key === 'Escape') {
                closeFind();
              }
            }}
          />
          <span className="pdf-find-count">
            {searching
              ? 'Searching…'
              : query.trim()
                ? matches.length
                  ? `${activeMatchIdx + 1} / ${matches.length}`
                  : 'No matches'
                : ''}
          </span>
          <button
            className="pdf-find-nav"
            onClick={() => gotoMatch(-1)}
            disabled={!matches.length}
            title="Previous match (Shift+Enter)"
          >
            ◀
          </button>
          <button
            className="pdf-find-nav"
            onClick={() => gotoMatch(1)}
            disabled={!matches.length}
            title="Next match (Enter)"
          >
            ▶
          </button>
          <button className="pdf-find-close" onClick={closeFind} title="Close (Esc)">
            ✕
          </button>
        </div>
      )}

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
          {(document.collection_title || document.collection_slug) && (
            <span className="text-muted"> · {document.collection_title || document.collection_slug}</span>
          )}
          {witnesses && witnesses.length > 0 && (
            <button
              className="btn-secondary"
              onClick={() => setShowWitnesses((v) => !v)}
              style={{ marginLeft: 'var(--spacing-md)', fontSize: '12px', padding: '2px 8px' }}
              title="Jump to a witness's testimony"
            >
              {showWitnesses ? '▾' : '▸'} Witnesses ({witnesses.length})
            </button>
          )}
        </div>
      )}

      {/* Witness index: jump to where each witness's testimony begins */}
      {witnesses && witnesses.length > 0 && showWitnesses && (
        <div
          style={{
            maxHeight: 220,
            overflowY: 'auto',
            padding: 'var(--spacing-sm) var(--spacing-md)',
            background: 'var(--color-bg-secondary)',
            borderBottom: '1px solid var(--color-border)',
            fontSize: '12px',
          }}
        >
          {witnesses.map((w) => (
            <button
              key={w.appearance_seq}
              onClick={() => {
                setCurrentPage(w.start_page);
                setShowWitnesses(false);
              }}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                gap: 'var(--spacing-md)',
                width: '100%',
                textAlign: 'left',
                background:
                  w.start_page <= currentPage && currentPage <= w.end_page
                    ? 'var(--color-highlight)'
                    : 'transparent',
                border: 'none',
                borderRadius: 4,
                padding: '4px 6px',
                cursor: 'pointer',
                color: 'inherit',
              }}
              title={`Pages ${w.start_page}–${w.end_page}`}
            >
              <span>
                <strong>{w.witness_name}</strong>
                {w.testimony_date && <span className="text-muted"> · {w.testimony_date}</span>}
                {w.examiner && <span className="text-muted"> · examined by {w.examiner}</span>}
              </span>
              <span className="text-muted" style={{ whiteSpace: 'nowrap' }}>
                pp. {w.start_page}–{w.end_page}
              </span>
            </button>
          ))}
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
          {evidence.why && <div className="text-sm text-muted mt-sm">Relevance: {evidence.why}</div>}
        </div>
      )}

      {/* PDF render: react-pdf, single page at a time (bounded memory for large docs) */}
      <div className="pdf-container">
        {!mounted || docLoading ? (
          <div className="loading">Loading document...</div>
        ) : loadError ? (
          <div className="empty-state">
            <p>PDF file missing</p>
            <p className="text-sm text-muted">{loadError}</p>
            {document && (
              <div className="card" style={{ textAlign: 'left', maxWidth: 520 }}>
                <div className="text-sm"><strong>Document</strong>: {document.source_name}</div>
                {(document.collection_title || document.collection_slug) && (
                  <div className="text-sm text-muted">Collection: {document.collection_title || document.collection_slug}</div>
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
          <div className="pdf-page-scroll" ref={pageWrapRef}>
            <Document
              file={fileProp}
              loading={<div className="loading">Loading document...</div>}
              error={<div className="loading">Could not load PDF.</div>}
              onLoadSuccess={(pdf) => {
                pdfRef.current = pdf;
                setNumPages(pdf.numPages);
                setLoadError(null);
              }}
              onLoadError={(err) => setLoadError(err?.message || 'Failed to load PDF.')}
            >
              <Page
                key={currentPage}
                pageNumber={currentPage}
                scale={zoom / 100}
                renderAnnotationLayer
                renderTextLayer
                onRenderTextLayerSuccess={applyHighlights}
                loading={<div className="loading">Rendering page…</div>}
              />
            </Document>
          </div>
        )}
      </div>
    </div>
  );
}

/** Remove any active find highlights (safe no-op when unsupported). */
function CSS_clearHighlights() {
  if (!HIGHLIGHT_SUPPORTED) return;
  const css = CSS as unknown as { highlights: Map<string, unknown> };
  css.highlights.delete('pdf-find');
  css.highlights.delete('pdf-find-active');
}
