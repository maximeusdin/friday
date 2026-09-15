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
        // A bracket holds either one label or a comma-separated list — and labels
        // themselves contain commas ("Silvermaster FBI file, Vol. 3 p. 41"), so
        // the whole content is tried as a single label before splitting.
        const labels = resolve(m[1].trim())
          ? [m[1].trim()]
          : m[1].split(',').map((s) => s.trim()).filter(Boolean);
        if (!labels.some((l) => resolve(l))) continue; // not a citation — leave as text
        if (m.index > cursor) {
          out.push(...emphasis(raw.slice(cursor, m.index), `${keyBase}-t${n++}`));
        }
        out.push(
          <span className="cite-group" key={`${keyBase}-c${n++}`}>
            {labels.map((label, j) => {
              const detail = resolve(label);
              if (detail?.document_id) {
                const rawLabel = detail.label && /^\d+$/.test(label)
                  ? detail.label
                  : (detail.label || stripConfidence(label));
                const display = stripPage(stripConfidence(rawLabel));
                return (
                  <button
                    type="button"
                    key={j}
                    className="cite"
                    onClick={() => openCitation(detail)}
                    title={`Open ${display}${detail.page ? `, page ${detail.quote_page ?? detail.page}` : ''}`}
                  >
                    {display}
                  </button>
                );
              }
              return (
                <span className="cite-unresolved" key={j}>
                  [{stripPage(stripConfidence(label)) || label}]
                </span>
              );
            })}
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

/** Block pass: group lines into paragraphs, lists, headings and quotes. */
function renderBlocks(text: string, inline: Inline): ReactNode[] {
  const lines = (text || '').split('\n');
  const blocks: ReactNode[] = [];
  let para: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;
  let quote: string[] = [];
  let key = 0;

  const flushPara = () => {
    if (!para.length) return;
    const body = para.join(' ');
    blocks.push(<p key={`p${key++}`}>{inline(body, `p${key}`)}</p>);
    para = [];
  };
  const flushList = () => {
    if (!list) return;
    const { ordered, items } = list;
    const children = items.map((item, i) => <li key={i}>{inline(item, `l${key}-${i}`)}</li>);
    blocks.push(ordered
      ? <ol key={`o${key++}`}>{children}</ol>
      : <ul key={`u${key++}`}>{children}</ul>);
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

    const heading = HEADING.exec(line);
    if (heading) {
      flushAll();
      blocks.push(<h3 key={`h${key++}`}>{inline(heading[1], `h${key}`)}</h3>);
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
