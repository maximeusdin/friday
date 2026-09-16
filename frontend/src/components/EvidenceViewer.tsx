'use client';

import { useState, useEffect, useLayoutEffect, useRef, useCallback, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/TextLayer.css';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import { api } from '@/lib/api';
import type { EvidenceRef } from '@/types/api';
import { HelpModal } from './HelpModal';
import { Icon } from './ui/Icon';
import { Menu } from './ui/Menu';
import { useLayout } from '@/lib/useLayout';

// Wire up the PDF.js worker. `pdfjs.version` is the EXACT pdfjs-dist version react-pdf uses
// (its own bundled copy), so pinning the worker to that version can never drift from the API —
// this is what previously broke ("API version 4.8.69 does not match Worker version 4.10.38"),
// because a build-time copy of a different top-level pdfjs-dist got served from /public and
// browser-cached. Loading the version-matched worker guarantees they agree on every deploy.
pdfjs.GlobalWorkerOptions.workerSrc =
  `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

const ZOOM_LEVELS = [50, 75, 100, 125, 150, 200];
/** Pages kept mounted either side of the one being read. */
const RENDER_WINDOW = 2;
/** How far either side of a cited page to hunt for a quote that isn't on it.
 *  Chunks span tens of pages, so the passage can be well away from the citation. */
const LOCATE_RADIUS = 40;
/** Misses tolerated before deciding a quote really is not on the page. */
const QUOTE_MATCH_ATTEMPTS = 5;
/** Pinch/double-tap zoom range on touch layouts, as a multiple of fit-to-width. */
const TOUCH_ZOOM_MIN = 0.6;
const TOUCH_ZOOM_MAX = 4;
const DOUBLE_TAP_ZOOM = 2.2;

/**
 * Wrap a JPEG in a minimal single-page PDF (PDF 1.4, one image XObject drawn
 * to fill the page). Hand-built so the "Download Page" button needs no PDF
 * writer dependency. imgW/imgH are the JPEG's pixel dimensions; ptW/ptH the
 * page size in PDF points (viewport at scale 1).
 */
function jpegToSinglePagePdf(jpeg: Uint8Array, imgW: number, imgH: number, ptW: number, ptH: number): Blob {
  const enc = new TextEncoder();
  const parts: Uint8Array[] = [];
  let offset = 0;
  const offsets: number[] = [];
  const push = (chunk: string | Uint8Array) => {
    const bytes = typeof chunk === 'string' ? enc.encode(chunk) : chunk;
    parts.push(bytes);
    offset += bytes.length;
  };
  const beginObj = (n: number) => {
    offsets[n] = offset;
    push(`${n} 0 obj\n`);
  };
  const w = ptW.toFixed(2);
  const h = ptH.toFixed(2);

  push('%PDF-1.4\n');
  beginObj(1);
  push('<< /Type /Catalog /Pages 2 0 R >>\nendobj\n');
  beginObj(2);
  push('<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n');
  beginObj(3);
  push(
    `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${w} ${h}] ` +
    '/Contents 4 0 R /Resources << /XObject << /Im0 5 0 R >> >> >>\nendobj\n'
  );
  const content = `q ${w} 0 0 ${h} 0 0 cm /Im0 Do Q`;
  beginObj(4);
  push(`<< /Length ${content.length} >>\nstream\n${content}\nendstream\nendobj\n`);
  beginObj(5);
  push(
    `<< /Type /XObject /Subtype /Image /Width ${imgW} /Height ${imgH} ` +
    `/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${jpeg.length} >>\nstream\n`
  );
  push(jpeg);
  push('\nendstream\nendobj\n');

  const xrefStart = offset;
  push(
    'xref\n0 6\n0000000000 65535 f \n' +
    [1, 2, 3, 4, 5].map((n) => `${String(offsets[n]).padStart(10, '0')} 00000 n \n`).join('')
  );
  push(`trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n${xrefStart}\n%%EOF`);
  return new Blob(parts as BlobPart[], { type: 'application/pdf' });
}

// CSS Custom Highlight API — modern browsers (Chrome/Edge/Safari) let us paint
// search highlights over the text layer without mutating the DOM.
const HIGHLIGHT_SUPPORTED =
  typeof window !== 'undefined' &&
  typeof (window as unknown as { Highlight?: unknown }).Highlight !== 'undefined' &&
  typeof CSS !== 'undefined' &&
  'highlights' in CSS;

// --- Transcript provenance chip (document.metadata.transcript.status) ---
// Documents whose searchable text was produced by AI vision transcription carry
// transcript.status in their metadata: 'machine_unverified' until a human
// review pass promotes it to 'reviewed'. Most documents have neither.
const TRANSCRIPT_CHIPS: Record<string, { label: string; tooltip: string; chipClass: string }> = {
  machine_unverified: {
    label: 'Machine transcript',
    tooltip:
      'Searchable text for this document was produced by AI vision transcription. The scan is authoritative; confirm quotations against the image.',
    chipClass: 'chip-amber',
  },
  reviewed: {
    label: 'Reviewed transcript',
    tooltip: 'Transcript reviewed against the scan. The scan remains authoritative.',
    chipClass: 'chip-green',
  },
};

function transcriptChipFor(metadata: Record<string, unknown> | undefined | null) {
  const t = metadata?.transcript as { status?: unknown } | undefined | null;
  const status = typeof t?.status === 'string' ? t.status : null;
  return status ? TRANSCRIPT_CHIPS[status] ?? null : null;
}

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

/** Normalize text for matching while keeping a map back to the original string
 * indices, so matched spans can become DOM Ranges.
 *
 * ALL whitespace is dropped (not collapsed): the PDF.js text layer glues words
 * together at line breaks (no space/newline between line-end and line-start
 * items), while quotes come from OCR text with real line breaks. Any quote
 * spanning >1 line can therefore never match space-normalized text — matching
 * must be space-insensitive. */
function normalizeSpaceFree(src: string): { norm: string; map: number[] } {
  let norm = '';
  const map: number[] = [];
  for (let i = 0; i < src.length; i++) {
    let ch = src[i];
    if (ch === '­') continue; // soft hyphen
    if (/\s/.test(ch)) continue;
    if (/[‘’‚′']/.test(ch)) ch = "'";
    else if (/[“”„″"]/.test(ch)) ch = '"';
    else if (/[–—−-]/.test(ch)) ch = '-';
    norm += ch.toLowerCase();
    map.push(i);
  }
  return { norm, map };
}

interface QuoteMatch {
  start: number; // original-string start index (inclusive)
  end: number;   // original-string end index (exclusive)
  tier: 'exact' | 'fuzzy';
}

// Calibrated against sampled corpus pages: correct fuzzy windows score >= 0.66,
// wrong windows <= 0.53. 0.65 accepts the former and rejects the latter.
const FUZZY_TRIGRAM_THRESHOLD = 0.65;
const MIN_FUZZY_QUOTE_CHARS = 20; // space-free chars; shorter quotes are exact-only

/**
 * Locate a (possibly OCR-divergent) verbatim quote inside a page's text layer.
 * Tier 1: space-free normalized exact substring (immune to PDF.js line-break
 *         gluing and whitespace differences between OCR sources).
 * Tier 2: character-trigram rolling window over the space-free text — robust to
 *         a different OCR engine's character errors, and hard to fool with
 *         common-word soup (trigrams encode local ordering). Accepted only
 *         above a strict threshold so we never paint a wrong highlight.
 */
function matchQuoteInText(pageText: string, quote: string): QuoteMatch | null {
  const { norm: pn, map } = normalizeSpaceFree(pageText);
  // Search snippets arrive wrapped in '...'/'…' truncation markers — strip them so
  // the core text can exact-match the page.
  const coreQuote = quote.trim().replace(/^[.…\s]+/, '').replace(/[.…\s]+$/, '');
  const { norm: qn } = normalizeSpaceFree(coreQuote);
  if (qn.length < 12 || !pn) return null;

  // Tier 1 — exact (space-free normalized)
  const idx = pn.indexOf(qn);
  if (idx !== -1) {
    return { start: map[idx], end: map[idx + qn.length - 1] + 1, tier: 'exact' };
  }

  // Tier 2 — trigram rolling window of the quote's length
  if (qn.length < MIN_FUZZY_QUOTE_CHARS || pn.length < qn.length) return null;
  const qt = new Map<string, number>();
  for (let i = 0; i + 3 <= qn.length; i++) {
    const t = qn.slice(i, i + 3);
    qt.set(t, (qt.get(t) ?? 0) + 1);
  }
  const total = qn.length - 2; // trigram count in the quote
  const w = qn.length;         // window width in chars
  const freq = new Map<string, number>();
  let overlap = 0;
  let best = { score: -1, s: 0 };
  for (let i = 0; i + 3 <= pn.length; i++) {
    const t = pn.slice(i, i + 3);
    const c = (freq.get(t) ?? 0) + 1;
    freq.set(t, c);
    if (c <= (qt.get(t) ?? 0)) overlap++;
    const j = i - (w - 2); // trigram start leaving the window
    if (j >= 0) {
      const old = pn.slice(j, j + 3);
      const oc = freq.get(old)!;
      if (oc <= (qt.get(old) ?? 0)) overlap--;
      freq.set(old, oc - 1);
    }
    if (i >= w - 3) {
      const score = overlap / total;
      if (score > best.score) best = { score, s: i - w + 3 };
    }
  }
  if (best.score >= FUZZY_TRIGRAM_THRESHOLD) {
    const s = best.s;
    const e = Math.min(s + w, pn.length);
    return { start: map[s], end: map[e - 1] + 1, tier: 'fuzzy' };
  }
  return null;
}

export function EvidenceViewer({ evidence, onClose, backLabel = 'Back to Chat' }: EvidenceViewerProps) {
  const [currentPage, setCurrentPage] = useState(1);
  const [numPages, setNumPages] = useState<number | null>(null);
  const [zoom, setZoom] = useState(125); // desktop: percent, stepped through ZOOM_LEVELS
  const layout = useLayout();
  // Touch layouts: the page fits the width by default and pinch multiplies it.
  // `fitScale` is the pdf.js scale at which page 1 exactly fills the scroller.
  const [fitScale, setFitScale] = useState(1);
  const [zoomFactor, setZoomFactor] = useState(1);
  // Live CSS preview while two fingers are down; committed on release.
  const [pinchPreview, setPinchPreview] = useState<{ k: number; ox: number; oy: number } | null>(null);
  // The quote strip takes a third of a phone screen: collapsed to two lines there.
  const [quoteOpen, setQuoteOpen] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);
  const [downloadingDoc, setDownloadingDoc] = useState(false);
  const [showCollections, setShowCollections] = useState(false);
  // Typed page number, held while the field is being edited (null = show currentPage).
  const [pageInput, setPageInput] = useState<string | null>(null);

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
  // Continuous scroll: the scroller, one wrapper per page, and the page-1
  // viewport used to size the wrappers that have not rendered yet.
  const scrollerRef = useRef<HTMLDivElement>(null);
  const pageEls = useRef<Map<number, HTMLDivElement>>(new Map());
  const [baseSize, setBaseSize] = useState<{ w: number; h: number } | null>(null);
  // Where the quote actually turned out to be, when the cited page was wrong.
  const [locatedPage, setLocatedPage] = useState<number | null>(null);
  const [locating, setLocating] = useState(false);
  // Retry handle + attempt count for a text layer that is still filling in.
  const quoteRetry = useRef<number | undefined>(undefined);
  const quoteAttempts = useRef(0);
  // True while we are scrolling the view ourselves, so the scroll handler does
  // not fight the navigation that caused it.
  const programmatic = useRef(false);
  // Which evidence target we have already centred on. The quote highlight is
  // repainted continuously as pages mount; centring is a once-per-target event.
  const autoScrolledFor = useRef<string | null>(null);
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

  // …and bring it into view once the document has loaded and the page wrappers
  // exist. Opening a citation must land on the cited page, not page 1.
  useEffect(() => {
    if (!numPages || !evidence?.pdf_page) return;
    goToPage(evidence.pdf_page);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [numPages, evidence?.pdf_page]);

  // Track the page under the viewport as the reader scrolls, so the toolbar,
  // downloads and the witness index follow the page actually being read.
  useEffect(() => {
    const scroller = scrollerRef.current;
    if (!scroller || !numPages) return;
    let frame = 0;
    const onScroll = () => {
      if (programmatic.current || frame) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        const mid = scroller.scrollTop + scroller.clientHeight * 0.35;
        let best = 1;
        for (const [n, el] of pageEls.current) {
          if (el.offsetTop <= mid && n > best) best = n;
        }
        setCurrentPage((cur) => (cur === best ? cur : best));
      });
    };
    scroller.addEventListener('scroll', onScroll, { passive: true });
    return () => {
      scroller.removeEventListener('scroll', onScroll);
      if (frame) cancelAnimationFrame(frame);
    };
  }, [numPages]);

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

  // The scale pages actually render at. Desktop steps through fixed percentages;
  // touch layouts fit the width and let pinch multiply that.
  const effectiveScale = layout.touch ? fitScale * zoomFactor : zoom / 100;

  // Where the view should end up after a zoom commits, so the point under the
  // fingers (or the double-tap) stays put once the pages re-lay out.
  const zoomAnchor = useRef<{ vx: number; vy: number; sx: number; sy: number; ratio: number } | null>(null);

  /** Zoom on touch layouts around a viewport point, keeping that point still. */
  const zoomTouchTo = useCallback((factor: number, vx: number, vy: number) => {
    const scroller = scrollerRef.current;
    const next = Math.min(TOUCH_ZOOM_MAX, Math.max(TOUCH_ZOOM_MIN, factor));
    if (!scroller || next === zoomFactor) return;
    zoomAnchor.current = {
      vx, vy,
      sx: scroller.scrollLeft + vx,
      sy: scroller.scrollTop + vy,
      ratio: next / zoomFactor,
    };
    setZoomFactor(next);
  }, [zoomFactor]);

  useLayoutEffect(() => {
    const a = zoomAnchor.current;
    const scroller = scrollerRef.current;
    if (!a || !scroller) return;
    zoomAnchor.current = null;
    scroller.scrollLeft = a.sx * a.ratio - a.vx;
    scroller.scrollTop = a.sy * a.ratio - a.vy;
  }, [zoomFactor]);

  const centre = () => {
    const el = scrollerRef.current;
    return el ? { vx: el.clientWidth / 2, vy: el.clientHeight / 2 } : { vx: 0, vy: 0 };
  };

  // Zoom handlers
  const handleZoomIn = () => {
    if (layout.touch) { const c = centre(); zoomTouchTo(zoomFactor * 1.25, c.vx, c.vy); return; }
    const idx = ZOOM_LEVELS.indexOf(zoom);
    if (idx < ZOOM_LEVELS.length - 1) setZoom(ZOOM_LEVELS[idx + 1]);
  };
  const handleZoomOut = () => {
    if (layout.touch) { const c = centre(); zoomTouchTo(zoomFactor / 1.25, c.vx, c.vy); return; }
    const idx = ZOOM_LEVELS.indexOf(zoom);
    if (idx > 0) setZoom(ZOOM_LEVELS[idx - 1]);
  };
  const handleZoomReset = () => {
    if (layout.touch) { const c = centre(); zoomTouchTo(1, c.vx, c.vy); return; }
    setZoom(100);
  };

  // Fit-to-width: measure the scroller and size page 1 to it. Re-measured on
  // rotation and whenever the scroller resizes.
  useEffect(() => {
    const el = scrollerRef.current;
    if (!el || !baseSize) return;
    const measure = () => {
      const cs = getComputedStyle(el);
      const w = el.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
      if (w > 0) setFitScale(Math.max(0.2, w / baseSize.w));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [baseSize]);

  // --- Pinch to zoom and double-tap, on touch layouts ---------------------------
  // Two fingers scale the page stack with a CSS transform while they move (cheap),
  // then the real scale is committed on release and the pages re-render sharp.
  const pointers = useRef<Map<number, { x: number; y: number }>>(new Map());
  const pinch = useRef<{ dist0: number; factor0: number; k: number; vx: number; vy: number } | null>(null);
  const lastTap = useRef<{ t: number; x: number; y: number } | null>(null);

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!layout.touch || e.pointerType !== 'touch') return;
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()];
      const scroller = scrollerRef.current!;
      const rect = scroller.getBoundingClientRect();
      pinch.current = {
        dist0: Math.hypot(a.x - b.x, a.y - b.y) || 1,
        factor0: zoomFactor,
        k: 1,
        vx: (a.x + b.x) / 2 - rect.left,
        vy: (a.y + b.y) / 2 - rect.top,
      };
    }
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!pointers.current.has(e.pointerId)) return;
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const pz = pinch.current;
    if (!pz || pointers.current.size < 2) return;
    const [a, b] = [...pointers.current.values()];
    const raw = Math.hypot(a.x - b.x, a.y - b.y) / pz.dist0;
    const target = Math.min(TOUCH_ZOOM_MAX, Math.max(TOUCH_ZOOM_MIN, pz.factor0 * raw));
    pz.k = target / pz.factor0;
    const scroller = scrollerRef.current!;
    const pagesEl = pageWrapRef.current!;
    const pr = pagesEl.getBoundingClientRect();
    const sr = scroller.getBoundingClientRect();
    setPinchPreview({ k: pz.k, ox: sr.left + pz.vx - pr.left, oy: sr.top + pz.vy - pr.top });
  };

  const onPointerEnd = (e: React.PointerEvent<HTMLDivElement>) => {
    const had = pointers.current.has(e.pointerId);
    pointers.current.delete(e.pointerId);
    const pz = pinch.current;
    if (pz && pointers.current.size < 2) {
      pinch.current = null;
      setPinchPreview(null);
      zoomTouchTo(pz.factor0 * pz.k, pz.vx, pz.vy);
      lastTap.current = null;
      return;
    }
    // Double-tap: toggle between fit-to-width and a readable zoom, at the tap.
    if (had && layout.touch && e.pointerType === 'touch' && pointers.current.size === 0) {
      const now = Date.now();
      const prev = lastTap.current;
      const rect = scrollerRef.current!.getBoundingClientRect();
      const vx = e.clientX - rect.left;
      const vy = e.clientY - rect.top;
      if (prev && now - prev.t < 320 && Math.hypot(prev.x - e.clientX, prev.y - e.clientY) < 30) {
        lastTap.current = null;
        zoomTouchTo(zoomFactor > 1.05 ? 1 : DOUBLE_TAP_ZOOM, vx, vy);
      } else {
        lastTap.current = { t: now, x: e.clientX, y: e.clientY };
      }
    }
  };

  // The browser must not pan with two fingers while we are pinching.
  useEffect(() => {
    const el = scrollerRef.current;
    if (!el) return;
    const onTouchMove = (ev: TouchEvent) => { if (ev.touches.length >= 2) ev.preventDefault(); };
    el.addEventListener('touchmove', onTouchMove, { passive: false });
    return () => el.removeEventListener('touchmove', onTouchMove);
  }, [mounted]);

  /** Scroll a page into view inside the document scroller. */
  const scrollToPage = useCallback((n: number, behavior: ScrollBehavior = 'auto') => {
    const scroller = scrollerRef.current;
    const el = pageEls.current.get(n);
    if (!scroller || !el) return;
    programmatic.current = true;
    const top = el.getBoundingClientRect().top
      - scroller.getBoundingClientRect().top
      + scroller.scrollTop;
    scroller.scrollTo({ top: Math.max(0, top - 12), behavior });
    window.setTimeout(() => { programmatic.current = false; }, 400);
  }, []);

  /** Explicit navigation (toolbar, typed page, find, witnesses, evidence link):
   *  move the page *and* the viewport. Scrolling by hand only sets the page. */
  const goToPage = useCallback((n: number) => {
    const max = numPages ?? document?.page_count ?? Number.MAX_SAFE_INTEGER;
    const target = Math.min(Math.max(1, Math.round(n)), max);
    setCurrentPage(target);
    // The wrapper may not be mounted yet on a fresh document; retry next frame.
    if (pageEls.current.has(target)) scrollToPage(target);
    else requestAnimationFrame(() => scrollToPage(target));
  }, [numPages, document?.page_count, scrollToPage]);

  /** Jump to a typed page number, clamped to the document. */
  const commitPageInput = () => {
    if (pageInput == null) return;
    const n = Number(pageInput);
    if (Number.isFinite(n) && n >= 1) goToPage(n);
    setPageInput(null);
  };

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
      goToPage(out[0].page);
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
    if (!query.trim()) return;

    const needle = query.toLowerCase();
    const all: Range[] = [];
    let activeRange: Range | null = null;
    const activePage = activeMatchIdx >= 0 ? matches[activeMatchIdx]?.page : undefined;

    // Every mounted page is painted, so matches stay highlighted as they scroll
    // past rather than only on the page the toolbar happens to name.
    for (const [pageNum, wrapper] of pageEls.current) {
      const layer = wrapper.querySelector('.react-pdf__Page__textContent');
      if (!layer) continue;
      const walker = window.document.createTreeWalker(layer, NodeFilter.SHOW_TEXT);
      const nodes: TextNodeEntry[] = [];
      let full = '';
      for (let n = walker.nextNode(); n; n = walker.nextNode()) {
        const text = n as Text;
        nodes.push({ node: text, start: full.length });
        full += text.nodeValue ?? '';
      }
      if (!nodes.length) continue;

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
      if (!ranges.length) continue;

      // Which occurrence on this page is the active match?
      if (pageNum === activePage) {
        const ordinal = matches.slice(0, activeMatchIdx).filter((m) => m.page === pageNum).length;
        activeRange = ranges[ordinal] ?? null;
      }
      all.push(...ranges);
    }
    if (!all.length) return;

    const rest = activeRange ? all.filter((r) => r !== activeRange) : all;
    const HL = (window as unknown as { Highlight: new (...r: Range[]) => unknown }).Highlight;
    if (rest.length) css.highlights.set('pdf-find', new HL(...rest));
    if (activeRange) {
      css.highlights.set('pdf-find-active', new HL(activeRange));
      activeRange.startContainer.parentElement?.scrollIntoView({ block: 'center', inline: 'nearest' });
    }
  }, [query, activeMatchIdx, matches]);

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
    const quotePage = locatedPage ?? evidence?.quote_page ?? evidence?.pdf_page;
    // Paint whenever the quote's page is mounted — with continuous scrolling it
    // need not be the page the toolbar currently names.
    const wrapper = quotePage != null ? pageEls.current.get(quotePage) : null;
    const layer = wrapper?.querySelector('.react-pdf__Page__textContent');
    if (!layer) return;
    const walker = window.document.createTreeWalker(layer, NodeFilter.SHOW_TEXT);
    const nodes: TextNodeEntry[] = [];
    let full = '';
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const text = n as Text;
      nodes.push({ node: text, start: full.length });
      full += text.nodeValue ?? '';
    }
    // react-pdf fills the text layer incrementally, and every mounted page's
    // render callback runs this — so a miss can simply mean "the layer is not
    // finished yet". Declaring failure on the first miss is what made a quote
    // that was found and highlighted still report itself as unfindable: retry a
    // few times and only then conclude the passage is not on the page.
    const match = nodes.length ? matchQuoteInText(full, quote) : null;
    if (!match) {
      window.clearTimeout(quoteRetry.current);
      if (quoteAttempts.current < QUOTE_MATCH_ATTEMPTS) {
        quoteAttempts.current += 1;
        quoteRetry.current = window.setTimeout(() => applyEvidenceHighlightRef.current?.(), 300);
      } else {
        setQuoteTier('none');
      }
      return;
    }
    quoteAttempts.current = 0;
    const a = locate(nodes, match.start);
    const b = locate(nodes, match.end);
    const r = window.document.createRange();
    r.setStart(a.node, a.off);
    r.setEnd(b.node, b.off);
    const HL = (window as unknown as { Highlight: new (...r: Range[]) => unknown }).Highlight;
    css.highlights.set('pdf-evidence', new HL(r));
    setQuoteTier(match.tier);
    // Bring the evidence into view (center) the first time we paint *this*
    // target -- and only then. This callback re-runs as the reader scrolls,
    // because currentPage is a dependency (the quote must stay painted as pages
    // mount and unmount around it). Centring on every re-run is what pinned the
    // viewer to the cited page: scrolling by hand moved currentPage, which
    // repainted, which scrolled straight back. Explicit navigation still moves
    // the viewport, via goToPage.
    const target = `${evidence?.document_id ?? ''}|${quotePage ?? ''}|${quote.slice(0, 80)}`;
    if (autoScrolledFor.current !== target) {
      autoScrolledFor.current = target;
      // Centring scrolls the scroller, so claim it the way scrollToPage does or
      // the scroll handler reads our own movement as the reader's.
      programmatic.current = true;
      r.startContainer.parentElement?.scrollIntoView({ block: 'center', inline: 'nearest' });
      window.setTimeout(() => { programmatic.current = false; }, 400);
    }
  }, [evidence?.quote, evidence?.quote_page, evidence?.pdf_page, currentPage, locatedPage]);


  const applyEvidenceHighlightRef = useRef<(() => void) | null>(null);
  applyEvidenceHighlightRef.current = applyEvidenceHighlight;

  // Re-paint when match selection or page changes (text layer may already be rendered)
  useEffect(() => {
    applyHighlights();
    applyEvidenceHighlight();
  }, [applyHighlights, applyEvidenceHighlight]);

  useEffect(() => () => window.clearTimeout(quoteRetry.current), []);

  // Reset quote tier when the evidence target changes
  useEffect(() => {
    setQuoteTier(null);
    setLocatedPage(null);
    quoteAttempts.current = 0;
  }, [evidence?.document_id, evidence?.quote]);

  /** Page text, extracted once and cached (shared with find-in-document). */
  const pageText = useCallback(async (p: number): Promise<string> => {
    const cached = textCache.current.get(p);
    if (cached !== undefined) return cached;
    const pdf = pdfRef.current;
    if (!pdf) return '';
    const page = await pdf.getPage(p);
    const tc = await page.getTextContent();
    const t = tc.items.map((it) => ('str' in it ? it.str : '')).join('');
    textCache.current.set(p, t);
    return t;
  }, []);

  // When the quote is not on the page the citation named, go and find it.
  //
  // Chunks span many pages, and a citation is pinned to the chunk's page rather
  // than the page the sentence is printed on — so quotes routinely sit a page or
  // two away (a cable's footnotes are overleaf from the cable). Rather than
  // shrugging with "approximate location", search outwards from the cited page
  // for the passage itself.
  useEffect(() => {
    const quote = evidence?.quote;
    const cited = evidence?.quote_page ?? evidence?.pdf_page;
    if (!quote || quote.trim().length < 12 || quoteTier !== 'none') return;
    if (locatedPage != null || !numPages || !pdfRef.current) return;

    let cancelled = false;
    setLocating(true);
    (async () => {
      // Nearest pages first: the answer is almost always a page or two out.
      const order: number[] = [];
      for (let d = 1; d <= LOCATE_RADIUS; d++) {
        if ((cited ?? 1) - d >= 1) order.push((cited ?? 1) - d);
        if ((cited ?? 1) + d <= numPages) order.push((cited ?? 1) + d);
      }
      for (const p of order) {
        if (cancelled) return;
        let text = '';
        try {
          text = await pageText(p);
        } catch {
          continue;
        }
        if (matchQuoteInText(text, quote)) {
          if (cancelled) return;
          quoteAttempts.current = 0;
          setLocatedPage(p);
          goToPage(p);
          break;
        }
      }
      if (!cancelled) setLocating(false);
    })();
    return () => { cancelled = true; setLocating(false); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [quoteTier, evidence?.quote, evidence?.quote_page, evidence?.pdf_page, numPages, locatedPage]);

  // Clear highlights on unmount / document change
  useEffect(() => () => CSS_clearHighlights(), [evidence?.document_id]);

  // --- Find navigation ---
  const gotoMatch = useCallback((delta: number) => {
    if (!matches.length) return;
    const n = (activeMatchIdx + delta + matches.length) % matches.length;
    setActiveMatchIdx(n);
    goToPage(matches[n].page);
  }, [matches, activeMatchIdx, goToPage]);

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
  // Download just the CURRENT page as a single-page PDF, rendered from the
  // already-loaded PDF (no extra deps; a scanned page is an image anyway, so we
  // wrap a high-res JPEG render in a minimal hand-built PDF container).
  const handleDownloadPage = async () => {
    const pdf = pdfRef.current;
    if (!pdf) return;
    try {
      const page = await pdf.getPage(currentPage);
      const viewport = page.getViewport({ scale: 2 });
      const canvas = window.document.createElement('canvas');
      canvas.width = Math.ceil(viewport.width);
      canvas.height = Math.ceil(viewport.height);
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      await page.render({ canvasContext: ctx, viewport }).promise;
      // Page size in PDF points = viewport at scale 1
      const pt = page.getViewport({ scale: 1 });
      canvas.toBlob(async (blob) => {
        if (!blob) return;
        const jpeg = new Uint8Array(await blob.arrayBuffer());
        const pdfBlob = jpegToSinglePagePdf(jpeg, canvas.width, canvas.height, pt.width, pt.height);
        const a = window.document.createElement('a');
        const base = (document?.source_name || 'document').replace(/\.pdf$/i, '');
        a.href = URL.createObjectURL(pdfBlob);
        a.download = `${base}_page${currentPage}.pdf`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 5000);
      }, 'image/jpeg', 0.92);
    } catch {
      /* page render failed — full-document download remains available */
    }
  };

  // Download the FULL document with its PDF /OpenAction set to the page being
  // viewed, so desktop readers (Acrobat, most Windows viewers; macOS Preview
  // ignores it) open the file at the hit instead of page 1 — a researcher's
  // "the whole file is as good as the hit page, if it opens at the hit".
  // Stamping an existing PDF's catalog needs a real writer, so pdf-lib is
  // loaded on demand here (never in the page bundle); any failure falls back
  // to downloading the untouched original.
  const handleDownloadDocument = async () => {
    const filename = document?.source_name || 'document.pdf';
    const fallback = () => {
      const a = window.document.createElement('a');
      a.href = resolvedBaseUrl;
      a.download = filename;
      a.click();
    };
    const pdf = pdfRef.current;
    if (!pdf) return fallback();
    setDownloadingDoc(true);
    try {
      const [bytes, { PDFDocument, PDFName }] = await Promise.all([
        pdf.getData(),
        import('pdf-lib'),
      ]);
      const doc = await PDFDocument.load(bytes, { updateMetadata: false });
      const pageIdx = Math.min(Math.max(currentPage - 1, 0), doc.getPageCount() - 1);
      const dest = doc.context.obj([doc.getPage(pageIdx).ref, PDFName.of('Fit')]);
      doc.catalog.set(PDFName.of('OpenAction'), dest);
      const out = await doc.save();
      const a = window.document.createElement('a');
      a.href = URL.createObjectURL(new Blob([out as BlobPart], { type: 'application/pdf' }));
      a.download = filename;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    } catch {
      fallback();
    } finally {
      setDownloadingDoc(false);
    }
  };

  const handleOpenNewTab = () => {
    window.open(pdfUrl, '_blank', 'noopener,noreferrer');
    onClose(); // Return to chat
  };

  const totalPages = numPages ?? document?.page_count;

  return (
    <div className="doc">
      {/* Toolbar: back, identity, page nav, zoom, find, overflow */}
      <div className="doc-bar">
        <button className="btn-ghost" onClick={onClose} title={backLabel} aria-label={backLabel}>
          <Icon name="arrow-left" size={16} />
          <span className="btn-label">{backLabel}</span>
        </button>

        <div className="doc-title">
          <span className="doc-title-main" title={document?.source_name}>
            {document?.source_name || 'Document'}
          </span>
          {(document?.collection_title || document?.collection_slug) && (
            <span className="doc-title-sub">
              {document.collection_title || document.collection_slug}
            </span>
          )}
        </div>

        <div className="spacer" />

        {/* Page navigation — the number is typable, so page 412 is one action away. */}
        <div className="doc-group">
          <button
            className="icon-btn icon-btn-sm"
            onClick={() => goToPage(currentPage - 1)}
            disabled={currentPage <= 1}
            title="Previous page"
            aria-label="Previous page"
          >
            <Icon name="chevron-left" size={16} />
          </button>
          <input
            className="doc-page-input"
            value={pageInput ?? currentPage}
            onChange={(e) => setPageInput(e.target.value.replace(/[^0-9]/g, ''))}
            onBlur={() => commitPageInput()}
            onKeyDown={(e) => {
              if (e.key === 'Enter') { e.preventDefault(); commitPageInput(); }
              if (e.key === 'Escape') setPageInput(null);
            }}
            aria-label={`Page number, ${totalPages ? `of ${totalPages}` : ''}`}
          />
          <span className="doc-page-total">{totalPages ? `/ ${totalPages}` : ''}</span>
          <button
            className="icon-btn icon-btn-sm"
            onClick={() => goToPage(currentPage + 1)}
            disabled={totalPages != null && currentPage >= totalPages}
            title="Next page"
            aria-label="Next page"
          >
            <Icon name="chevron-right" size={16} />
          </button>
        </div>

        <div className="doc-group doc-group-zoom">
          <button
            className="icon-btn icon-btn-sm"
            onClick={handleZoomOut}
            disabled={layout.touch ? zoomFactor <= TOUCH_ZOOM_MIN : zoom <= ZOOM_LEVELS[0]}
            title="Zoom out"
            aria-label="Zoom out"
          >
            <Icon name="zoom-out" size={16} />
          </button>
          <button
            className="doc-zoom"
            onClick={handleZoomReset}
            title={layout.touch ? 'Fit to width' : 'Reset to 100%'}
          >
            {layout.touch ? `${Math.round(zoomFactor * 100)}%` : `${zoom}%`}
          </button>
          <button
            className="icon-btn icon-btn-sm"
            onClick={handleZoomIn}
            disabled={layout.touch ? zoomFactor >= TOUCH_ZOOM_MAX : zoom >= ZOOM_LEVELS[ZOOM_LEVELS.length - 1]}
            title="Zoom in"
            aria-label="Zoom in"
          >
            <Icon name="zoom-in" size={16} />
          </button>
        </div>

        <button
          className={`icon-btn${findOpen ? ' is-on' : ''}`}
          onClick={() => (findOpen ? closeFind() : openFind())}
          title="Find in document (Ctrl+F)"
          aria-label="Find in document"
        >
          <Icon name="search" size={17} />
        </button>

        <Menu
          label="Document actions"
          align="end"
          items={[
            // The zoom buttons are hidden on phones (pinch does the job); keep a
            // non-gesture path to the same controls here.
            ...(layout.isPhone ? [
              { label: 'Zoom in', icon: <Icon name="zoom-in" size={16} />, onSelect: handleZoomIn },
              { label: 'Zoom out', icon: <Icon name="zoom-out" size={16} />, onSelect: handleZoomOut },
              { label: 'Fit to width', icon: <Icon name="restore" size={16} />, onSelect: handleZoomReset, separated: false },
            ] : []),
            {
              label: 'Download this page',
              separated: layout.isPhone,
              icon: <Icon name="download" size={16} />,
              onSelect: () => { void handleDownloadPage(); },
            },
            {
              label: downloadingDoc ? 'Preparing…' : 'Download the document',
              icon: <Icon name="download" size={16} />,
              onSelect: () => { void handleDownloadDocument(); },
            },
            {
              label: 'Open PDF in a new tab',
              icon: <Icon name="external" size={16} />,
              onSelect: handleOpenNewTab,
              separated: true,
            },
            {
              label: 'Browse this collection',
              icon: <Icon name="library" size={16} />,
              onSelect: () => setShowCollections(true),
            },
          ]}
          trigger={(props) => (
            <button className="icon-btn" {...props} aria-label="Document actions" title="More">
              <Icon name="more" size={18} />
            </button>
          )}
        />
      </div>

      {/* Phones: the toolbar has no room for a filename, so it gets a line of
          its own, collection first because that is the part a reader recognises. */}
      {layout.isPhone && document && (
        <div className="doc-subbar" title={document.source_name}>
          {(document.collection_title || document.collection_slug) && (
            <span className="doc-subbar-coll">{document.collection_title || document.collection_slug}</span>
          )}
          <span className="doc-subbar-file">{document.source_name}</span>
        </div>
      )}

      {/* Docked find bar — sits in the toolbar area, never overlaps the page */}
      {findOpen && (
        <div className="doc-find">
          <label className="field" style={{ flex: '1 1 auto', maxWidth: '22rem' }}>
            <Icon name="search" size={15} />
            <input
              ref={findInputRef}
              type="text"
              placeholder="Find in this document…"
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
              aria-label="Find in document"
            />
          </label>
          <span className="doc-find-count">
            {searching
              ? 'Searching…'
              : query.trim()
                ? matches.length
                  ? `${activeMatchIdx + 1} of ${matches.length}`
                  : 'No matches'
                : ''}
          </span>
          <button
            className="icon-btn icon-btn-sm"
            onClick={() => gotoMatch(-1)}
            disabled={!matches.length}
            title="Previous match (Shift+Enter)"
            aria-label="Previous match"
          >
            <Icon name="chevron-up" size={16} />
          </button>
          <button
            className="icon-btn icon-btn-sm"
            onClick={() => gotoMatch(1)}
            disabled={!matches.length}
            title="Next match (Enter)"
            aria-label="Next match"
          >
            <Icon name="chevron-down" size={16} />
          </button>
          <div className="spacer" />
          <button className="icon-btn icon-btn-sm" onClick={closeFind} title="Close (Esc)" aria-label="Close find">
            <Icon name="close" size={16} />
          </button>
        </div>
      )}

      {/* Provenance strip: transcript status and the witness index, when present */}
      {document && (transcriptChipFor(document.metadata) || (witnesses && witnesses.length > 0)) && (
        <div className="doc-info">
          {(() => {
            const chip = transcriptChipFor(document.metadata);
            return chip ? (
              <span className={`chip ${chip.chipClass}`} title={chip.tooltip}>
                {chip.label}
              </span>
            ) : null;
          })()}
          {witnesses && witnesses.length > 0 && (
            <button
              className="btn-link"
              onClick={() => setShowWitnesses((v) => !v)}
              title="Jump to a witness's testimony"
            >
              <Icon name={showWitnesses ? 'chevron-down' : 'chevron-right'} size={14} />
              Witnesses ({witnesses.length})
            </button>
          )}
        </div>
      )}

      {/* Witness index: jump to where each witness's testimony begins */}
      {witnesses && witnesses.length > 0 && showWitnesses && (
        <div className="doc-witnesses">
          {witnesses.map((w) => (
            <button
              key={w.appearance_seq}
              className={`doc-witness${w.start_page <= currentPage && currentPage <= w.end_page ? ' is-current' : ''}`}
              onClick={() => {
                goToPage(w.start_page);
                setShowWitnesses(false);
              }}
              title={`Pages ${w.start_page}–${w.end_page}`}
            >
              <span className="truncate">
                <strong>{w.witness_name}</strong>
                {w.testimony_date && <span className="text-muted"> · {w.testimony_date}</span>}
                {w.examiner && <span className="text-muted"> · examined by {w.examiner}</span>}
              </span>
              <span className="count">pp. {w.start_page}–{w.end_page}</span>
            </button>
          ))}
        </div>
      )}

      {/* The cited passage, verbatim, with how confidently it was located */}
      {evidence.quote && (
        <div
          className="doc-quote"
          data-collapsed={layout.isPhone && !quoteOpen ? 'true' : 'false'}
          onClick={() => { if (layout.isPhone) setQuoteOpen((v) => !v); }}
          role={layout.isPhone ? 'button' : undefined}
          aria-expanded={layout.isPhone ? quoteOpen : undefined}
          title={layout.isPhone ? (quoteOpen ? 'Collapse' : 'Show the whole passage') : undefined}
        >
          <span>&ldquo;{evidence.quote}&rdquo;</span>
          <span className="doc-quote-meta">
            {quoteTier === 'exact' && (
              <><Icon name="check" size={13} /> highlighted on page {locatedPage ?? currentPage}</>
            )}
            {quoteTier === 'fuzzy' && (
              <><Icon name="check" size={13} /> highlighted on page {locatedPage ?? currentPage} (close match)</>
            )}
            {quoteTier === 'none' && locating && (
              <><span className="spinner" style={{ width: 12, height: 12 }} /> locating the passage…</>
            )}
            {quoteTier === 'none' && !locating && (
              <><Icon name="info" size={13} /> this passage isn&apos;t on the cited page and couldn&apos;t be found nearby</>
            )}
            {locatedPage != null && locatedPage !== (evidence.quote_page ?? evidence.pdf_page) && (
              <span className="text-muted">
                · the citation pointed at page {evidence.quote_page ?? evidence.pdf_page}
              </span>
            )}
            {evidence.why && <span className="text-muted">· {evidence.why}</span>}
          </span>
        </div>
      )}

      {/* PDF render: react-pdf, single page at a time (bounded memory for large docs) */}
      <div
        className="doc-canvas"
        ref={scrollerRef}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerEnd}
        onPointerCancel={onPointerEnd}
      >
        {!mounted || docLoading ? (
          <div className="loading"><span className="spinner" /> Loading document…</div>
        ) : loadError ? (
          <div className="empty-state">
            <strong>PDF file missing</strong>
            <span>{loadError}</span>
            {document && (
              <div className="card" style={{ textAlign: 'left', maxWidth: 520 }}>
                <div className="text-sm"><strong>Document</strong>: {document.source_name}</div>
                {(document.collection_title || document.collection_slug) && (
                  <div className="text-sm text-muted">
                    Collection: {document.collection_title || document.collection_slug}
                  </div>
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
              Try opening the PDF directly
              <Icon name="external" size={14} />
            </a>
          </div>
        ) : (
          <div
            className="doc-pages"
            ref={pageWrapRef}
            style={pinchPreview ? {
              transform: `scale(${pinchPreview.k})`,
              transformOrigin: `${pinchPreview.ox}px ${pinchPreview.oy}px`,
              willChange: 'transform',
            } : undefined}
          >
            <Document
              file={fileProp}
              loading={<div className="loading"><span className="spinner" /> Loading document…</div>}
              error={<div className="loading">Could not load this PDF.</div>}
              onLoadSuccess={async (pdf) => {
                pdfRef.current = pdf;
                setNumPages(pdf.numPages);
                setLoadError(null);
                // Page 1's size at scale 1 sizes the wrappers of pages that
                // have not rendered, so the scrollbar is the right length and
                // scrolling does not jump as pages mount.
                try {
                  const vp = (await pdf.getPage(1)).getViewport({ scale: 1 });
                  setBaseSize({ w: vp.width, h: vp.height });
                } catch { /* fall back to the default placeholder size */ }
              }}
              onLoadError={(err) => setLoadError(err?.message || 'Failed to load PDF.')}
            >
              {Array.from({ length: numPages ?? 0 }, (_, i) => i + 1).map((n) => {
                // Only pages near the viewport are mounted: these documents run
                // to hundreds of scanned pages, and rendering them all would
                // exhaust memory long before the reader got there.
                const mounted = Math.abs(n - currentPage) <= RENDER_WINDOW;
                const scale = effectiveScale;
                return (
                  <div
                    key={n}
                    className="doc-page"
                    data-page={n}
                    ref={(el) => {
                      if (el) pageEls.current.set(n, el);
                      else pageEls.current.delete(n);
                    }}
                    style={{
                      width: baseSize ? baseSize.w * scale : undefined,
                      minHeight: baseSize ? baseSize.h * scale : 600,
                    }}
                  >
                    {mounted ? (
                      <Page
                        pageNumber={n}
                        scale={scale}
                        renderAnnotationLayer
                        renderTextLayer
                        onRenderTextLayerSuccess={() => {
                          applyHighlights();
                          applyEvidenceHighlight();
                        }}
                        loading={<span className="doc-page-placeholder">{n}</span>}
                      />
                    ) : (
                      <span className="doc-page-placeholder">{n}</span>
                    )}
                  </div>
                );
              })}
            </Document>
          </div>
        )}
      </div>

      {showCollections && (
        <HelpModal
          section="collections"
          initialCollectionId={document?.collection_id}
          onClose={() => setShowCollections(false)}
        />
      )}
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
