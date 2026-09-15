'use client';

import type { ReactNode } from 'react';
import type { CitationDetail, EvidenceRef } from '@/types/api';

/**
 * AnswerText — renders an assistant answer as typeset prose with live citations.
 *
 * Two passes:
 *   1. Block pass  — paragraphs, bullet and numbered lists, headings, quotes.
 *      Answers arrive as lightly-marked-up text; rendering it with `pre-wrap`
 *      (as this used to) turned every list into a wall of hyphens.
 *   2. Inline pass — `[Label p.4]` citation brackets become buttons that open
 *      the document viewer at the cited page, plus `**bold**` and `code`.
 *
 * The citation resolution rules are load-bearing (labels are matched loosely
 * because the model echoes them with confidence suffixes and page numbers), so
 * they are preserved exactly as they were.
 */

/** Strip a confidence suffix like "(high)" from a citation label. */
function stripConfidence(label: string): string {
  return label.replace(/\s*\((?:high|medium|low)\)$/i, '').trim();
}

/** Strip the trailing page number — the document name alone is the button text. */
function stripPage(label: string): string {
  return label.replace(/\s+p\.?\s*\d+\s*$/i, '').trim() || label;
}

/**
 * Split the inside of a `[...]` into citation labels.
 *
 * Labels contain commas of their own ("Silvermaster FBI file, Vol. 3 p. 41"),
 * and a bracket can hold several of them, so a plain comma split shreds every
 * label into unresolvable fragments. Instead the comma-separated pieces are
 * rejoined greedily: at each position take the longest run of pieces that
 * resolves to a real citation, and move on.
 */
function splitLabels(
  inner: string,
  resolve: (label: string) => CitationDetail | undefined,
): string[] {
  const whole = inner.trim();
  if (resolve(whole)) return [whole];

  const parts = inner.split(',').map((p) => p.trim()).filter(Boolean);
  const out: string[] = [];
  const MAX_JOIN = 4; // a label is never more than a few comma-separated pieces
  let i = 0;
  while (i < parts.length) {
    let taken = 1;
    for (let n = Math.min(MAX_JOIN, parts.length - i); n >= 1; n--) {
      if (resolve(parts.slice(i, i + n).join(', '))) {
        taken = n;
        break;
      }
    }
    out.push(parts.slice(i, i + taken).join(', '));
    i += taken;
  }
  return out;
}

interface CiteChip {
  display: string;
  document: string;
  page?: number;
  detail?: CitationDetail;
}

/**
 * Turn the labels inside one `[...]` bracket into chips.
 *
 * Two things the raw labels get wrong on screen: the page is stripped for
 * readability, which makes two citations to different pages of the same file
 * render as identical twins; and the same page cited twice renders twice. So
 * pages are kept whenever they disambiguate, and exact repeats collapse.
 */
function citeChips(
  labels: string[],
  resolve: (label: string) => CitationDetail | undefined,
): CiteChip[] {
  const chips: CiteChip[] = labels.map((label) => {
    const detail = resolve(label);
    const rawLabel = detail?.label && /^\d+$/.test(label)
      ? detail.label
      : (detail?.label || stripConfidence(label));
    const document = stripPage(stripConfidence(rawLabel)) || label;
    const page = detail?.quote_page ?? detail?.page ?? undefined;
    return { display: document, document, page, detail };
  });

  // Same document cited on different pages: put the page back so they differ.
  const pagesPerDoc = new Map<string, Set<number>>();
  for (const c of chips) {
    if (c.page == null) continue;
    const set = pagesPerDoc.get(c.document) ?? new Set<number>();
    set.add(c.page);
    pagesPerDoc.set(c.document, set);
  }
  for (const c of chips) {
    if (c.page != null && (pagesPerDoc.get(c.document)?.size ?? 0) > 1) {
      c.display = `${c.document} p. ${c.page}`;
    }
  }

  // Collapse exact repeats (the same page cited twice in one bracket).
  const seen = new Set<string>();
  return chips.filter((c) => {
    const key = `${c.detail?.document_id ?? c.document}|${c.page ?? ''}|${c.display}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

interface Props {
  text: string;
  citationMap: Record<string, CitationDetail>;
  onEvidenceClick?: (evidence: EvidenceRef) => void;
}

export function AnswerText({ text, citationMap, onEvidenceClick }: Props) {
  const resolve = (label: string): CitationDetail | undefined => {
    const direct = citationMap[label] ?? citationMap[stripConfidence(label)];
    if (direct) return direct;
    const lower = label.toLowerCase();
    const key = Object.keys(citationMap).find((k) => k.toLowerCase() === lower);
    return key ? citationMap[key] : undefined;
  };

  const openCitation = (detail: CitationDetail) => {
    if (!onEvidenceClick || !detail.document_id) return;
    onEvidenceClick({
      document_id: detail.document_id,
      // Open the quote's exact page when known (mined citations on multi-page chunks).
      pdf_page: detail.quote_page ?? detail.page ?? 1,
      chunk_id: detail.chunk_id,
      quote: detail.quote,
      quote_page: detail.quote_page ?? undefined,
    });
  };

  const hasCitations = Object.keys(citationMap).length > 0;

  /** Inline pass: citations, then bold/code inside the remaining plain text. */
  const inline = (raw: string, keyBase: string): ReactNode[] => {
    const out: ReactNode[] = [];
    let cursor = 0;
    let n = 0;

    if (hasCitations) {
      const bracket = /\[([^\]\n]+)\]/g;
      let m: RegExpExecArray | null;
      while ((m = bracket.exec(raw)) !== null) {
        const labels = splitLabels(m[1], resolve);
        if (!labels.some((l) => resolve(l))) continue; // not a citation — leave as text
        if (m.index > cursor) {
          out.push(...emphasis(raw.slice(cursor, m.index), `${keyBase}-t${n++}`));
        }
        out.push(
          <span className="cite-group" key={`${keyBase}-c${n++}`}>
            {citeChips(labels, resolve).map((chip, j) => (
              chip.detail?.document_id ? (
                <button
                  type="button"
                  key={j}
                  className="cite"
                  onClick={() => openCitation(chip.detail!)}
                  title={`Open ${chip.document}${chip.page ? `, page ${chip.page}` : ''}`}
                >
                  {chip.display}
                </button>
              ) : (
                <span className="cite-unresolved" key={j}>[{chip.display}]</span>
              )
            ))}
          </span>,
        );
        cursor = m.index + m[0].length;
      }
    }

    if (cursor < raw.length) out.push(...emphasis(raw.slice(cursor), `${keyBase}-t${n++}`));
    return out;
  };

  return <div className="answer-prose">{renderBlocks(text, inline)}</div>;
}

/** `**bold**` and `` `code` `` inside a plain-text run. Deliberately minimal:
 *  anything more aggressive starts mangling archival quotations. */
function emphasis(raw: string, keyBase: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /\*\*([^*\n]+)\*\*|`([^`\n]+)`/g;
  let cursor = 0;
  let m: RegExpExecArray | null;
  let n = 0;
  while ((m = re.exec(raw)) !== null) {
    if (m.index > cursor) out.push(raw.slice(cursor, m.index));
    out.push(
      m[1] != null
        ? <strong key={`${keyBase}-b${n++}`}>{m[1]}</strong>
        : <code key={`${keyBase}-k${n++}`}>{m[2]}</code>,
    );
    cursor = m.index + m[0].length;
  }
  if (cursor < raw.length) out.push(raw.slice(cursor));
  return out.length ? out : [raw];
}

type Inline = (raw: string, keyBase: string) => ReactNode[];

const BULLET = /^\s*[-*•]\s+(.*)$/;
const ORDERED = /^\s*\d+[.)]\s+(.*)$/;
const HEADING = /^\s*#{1,4}\s+(.*)$/;
const QUOTE = /^\s*>\s?(.*)$/;
/** The answer builder's own rules: "--- Summary ---", "--- Narrative (…) ---". */
const RULE_HEADING = /^\s*-{2,}\s*(.+?)\s*-{2,}\s*$/;
/** Its section labels: "Findings:", "Members identified:", "Evidence:", … */
const SECTION = /^\s*([A-Z][^.!?]{0,70}):\s*$/;
/** Sections whose claims are, by construction, not fully supported. */
const UNVERIFIED = /unverified|partial or overlap-only|no claims passed/i;

/** Block pass: group lines into paragraphs, lists, headings and quotes. */
function renderBlocks(text: string, inline: Inline): ReactNode[] {
  const lines = (text || '').split('\n');
  const blocks: ReactNode[] = [];
  let para: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;
  let quote: string[] = [];
  let key = 0;

  // Claims under an "Unverified"/"draft" heading are marked, so a bullet with no
  // citation reads as "the evidence is partial" rather than as a missing link.
  let unverified = false;
  const cls = () => (unverified ? 'is-unverified' : undefined);

  const flushPara = () => {
    if (!para.length) return;
    const body = para.join(' ');
    blocks.push(<p key={`p${key++}`} className={cls()}>{inline(body, `p${key}`)}</p>);
    para = [];
  };
  const flushList = () => {
    if (!list) return;
    const { ordered, items } = list;
    const children = items.map((item, i) => <li key={i}>{inline(item, `l${key}-${i}`)}</li>);
    blocks.push(ordered
      ? <ol key={`o${key++}`} className={cls()}>{children}</ol>
      : <ul key={`u${key++}`} className={cls()}>{children}</ul>);
    list = null;
  };
  const flushQuote = () => {
    if (!quote.length) return;
    blocks.push(<blockquote key={`q${key++}`}>{inline(quote.join(' '), `q${key}`)}</blockquote>);
    quote = [];
  };
  const flushAll = () => { flushPara(); flushList(); flushQuote(); };

  for (const line of lines) {
    if (!line.trim()) { flushAll(); continue; }

    // The answer builder's section structure, rendered as structure rather than
    // as literal "--- Summary ---" text in the middle of the prose.
    const heading = HEADING.exec(line) || RULE_HEADING.exec(line) || SECTION.exec(line);
    if (heading) {
      flushAll();
      const title = heading[1].trim();
      unverified = UNVERIFIED.test(title);
      blocks.push(
        <h3 key={`h${key++}`} className={unverified ? 'is-unverified-heading' : undefined}>
          {SECTION.test(line) ? title.replace(/:$/, '') : title}
          {unverified && <span className="chip chip-amber">unverified</span>}
        </h3>,
      );
      continue;
    }

    const bullet = BULLET.exec(line);
    if (bullet) {
      flushPara(); flushQuote();
      if (!list || list.ordered) { flushList(); list = { ordered: false, items: [] }; }
      list.items.push(bullet[1]);
      continue;
    }

    const ordered = ORDERED.exec(line);
    if (ordered) {
      flushPara(); flushQuote();
      if (!list || !list.ordered) { flushList(); list = { ordered: true, items: [] }; }
      list.items.push(ordered[1]);
      continue;
    }

    const quoted = QUOTE.exec(line);
    if (quoted) {
      flushPara(); flushList();
      quote.push(quoted[1]);
      continue;
    }

    // A plain line continuing a list item wraps into that item rather than
    // starting a stray paragraph mid-list.
    if (list && /^\s{2,}\S/.test(line)) {
      list.items[list.items.length - 1] += ` ${line.trim()}`;
      continue;
    }

    flushList(); flushQuote();
    para.push(line.trim());
  }

  flushAll();
  return blocks;
}
