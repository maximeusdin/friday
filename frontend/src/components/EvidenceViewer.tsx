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

// ---------------------------------------------------------------------------
// Evidence-quote matching (semantic-search support quotes → text-layer ranges)
// ---------------------------------------------------------------------------

/** Normalize text for OCR-tolerant matching while keeping a map back to the
 * original string indices, so matched spans can become DOM Ranges. */
function normalizeWithMap(src: string): { norm: string; map: number[] } {
  let norm = '';
  const map: number[] = [];
  let lastWasSpace = true; // leading whitespace is dropped
  for (let i = 0; i < src.length; i++) {
    let ch = src[i];
    if (ch === '­') continue; // soft hyphen
    if (/[‘’‚′']/.test(ch)) ch = "'";
    else if (/[“”„″"]/.test(ch)) ch = '"';
    else if (/[–—−-]/.test(ch)) ch = '-';
    if (/\s/.test(ch)) {
      if (lastWasSpace) continue;
      norm += ' ';
      map.push(i);
      lastWasSpace = true;
      continue;
    }
    norm += ch.toLowerCase();
    map.push(i);
    lastWasSpace = false;
  }
  // trim trailing space
  if (norm.endsWith(' ')) {
    norm = norm.slice(0, -1);
    map.pop();
  }
  return { norm, map };
}

/** Plain normalization (no index map) for the quote side. */
function normalizePlain(src: string): string {
  return normalizeWithMap(src).norm;
}

interface QuoteMatch {
  start: number; // original-string start index (inclusive)
  end: number;   // original-string end index (exclusive)
  tier: 'exact' | 'fuzzy';
}

/**
 * Locate a (possibly OCR-divergent) verbatim quote inside a page's text.
 * Tier 1: normalized exact substring.
 * Tier 2: fuzzy token window — best window of ~quote length by token-overlap
 *         score, accepted only above a strict threshold so we never paint a
 *         wrong highlight.
 */
function matchQuoteInText(pageText: string, quote: string): QuoteMatch | null {
  const { norm, map } = normalizeWithMap(pageText);
  const q = normalizePlain(quote);
  if (!q || q.length < 8 || !norm) return null;

  // Tier 1 — exact (normalized)
  const idx = norm.indexOf(q);
  if (idx !== -1) {
    return { start: map[idx], end: map[idx + q.length - 1] + 1, tier: 'exact' };
  }

  // Tier 2 — token window. Tokenize with offsets into `norm`.
  const tokens: { t: string; s: number; e: number }[] = [];
  const re = /[^ ]+/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(norm)) !== null) tokens.push({ t: m[0], s: m.index, e: m.index + m[0].length });
  const qTokens = q.split(' ').filter(Boolean);
  if (qTokens.length < 3 || tokens.length < 3) return null;

  const qFreq = new Map<string, number>();
  for (const t of qTokens) qFreq.set(t, (qFreq.get(t) ?? 0) + 1);

  let best: { score: number; s: number; e: number } | null = null;
  const win = qTokens.length;
  const widths = [win, Math.max(3, Math.round(win * 0.8)), Math.round(win * 1.2)];
  for (const w of widths) {
    if (w > tokens.length) continue;
    // rolling multiset-overlap over windows of width w
    const freq = new Map<string, number>();
    let overlap = 0;
    for (let i = 0; i < tokens.length; i++) {
      const t = tokens[i].t;
      const c = (freq.get(t) ?? 0) + 1;
      freq.set(t, c);
      if (c <= (qFreq.get(t) ?? 0)) overlap++;
      if (i >= w) {
        const old = tokens[i - w].t;
        const oc = freq.get(old)!;
        if (oc <= (qFreq.get(old) ?? 0)) overlap--;
        freq.set(old, oc - 1);
      }
      if (i >= w - 1) {
        const score = overlap / Math.max(w, qTokens.length);
        if (!best || score > best.score) {
          best = { score, s: tokens[i - w + 1].s, e: tokens[i].e };
        }
      }
    }
  }
  if (best && best.score >= 0.6) {
    return { start: map[best.s], end: map[best.e - 1] + 1, tier: 'fuzzy' };
  }
  return null;
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

  // --- Evidence-quote highlight state ---
  // 'exact' | 'fuzzy': painted on the page; 'none': quote couldn't be located
  // (approximate-location fallback); null: no quote or not on the quote's page yet.
  const [quoteTier, setQuoteTier] = useState<'exact' | 'fuzzy' | 'none' | null>(null);

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

  // --- Paint the evidence-quote highlight (amber) on the quote's page ---
  const applyEvidenceHighlight = useCallback(() => {
    if (!HIGHLIGHT_SUPPORTED) return;
    const css = CSS as unknown as { highlights: Map<string, unknown> };
    css.highlights.delete('pdf-evidence');
    const quote = evidence?.quote;
    if (!quote || quote.trim().length < 8) {
      setQuoteTier(null);
      return;
    }
    const quotePage = evidence?.quote_page ?? evidence?.pdf_page;
    if (currentPage !== quotePage) return; // keep tier state; just don't paint here

    const layer = pageWrapRef.current?.querySelector('.react-pdf__Page__textContent');
    if (!layer) return;
    const walker = window.document.createTreeWalker(layer, NodeFilter.SHOW_TEXT);
    const nodes: TextNodeEntry[] = [];
    let full = '';
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const text = n as Text;
      nodes.push({ node: text, start: full.length });
      full += text.nodeValue ?? '';
    }
    if (!nodes.length) {
      setQuoteTier('none');
      return;
    }

    const match = matchQuoteInText(full, quote);
    if (!match) {
      setQuoteTier('none');
      return;
    }
    const a = locate(nodes, match.start);
    const b = locate(nodes, match.end);
    const r = window.document.createRange();
    r.setStart(a.node, a.off);
    r.setEnd(b.node, b.off);
    const HL = (window as unknown as { Highlight: new (...r: Range[]) => unknown }).Highlight;
    css.highlights.set('pdf-evidence', new HL(r));
    setQuoteTier(match.tier);
    // Bring the evidence into view (center) once painted.
    r.startContainer.parentElement?.scrollIntoView({ block: 'center', inline: 'nearest' });
  }, [evidence?.quote, evidence?.quote_page, evidence?.pdf_page, currentPage]);

  // Re-paint when match selection or page changes (text layer may already be rendered)
  useEffect(() => {
    applyHighlights();
    applyEvidenceHighlight();
  }, [applyHighlights, applyEvidenceHighlight]);

  // Reset quote tier when the evidence target changes
  useEffect(() => {
    setQuoteTier(null);
  }, [evidence?.document_id, evidence?.quote]);

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
          {quoteTier && (
            <span className={`quote-hl-badge quote-hl-${quoteTier}`} style={{ fontStyle: 'normal' }}>
              {quoteTier === 'exact' && '● highlighted on page'}
              {quoteTier === 'fuzzy' && '● highlighted (approximate match)'}
              {quoteTier === 'none' && '≈ approximate location — exact text could not be pinpointed on this page'}
            </span>
          )}
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
                onRenderTextLayerSuccess={() => {
                  applyHighlights();
                  applyEvidenceHighlight();
                }}
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
  css.highlights.delete('pdf-evidence');
}
