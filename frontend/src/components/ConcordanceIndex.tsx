'use client';

import { useCallback, useEffect, useState } from 'react';
import { api, type ConcordanceEntry, type ConcordanceSummary } from '@/lib/api';

/**
 * Splash-screen card describing the Concordance Index, with actions to
 * browse it in a modal or download it as CSV.
 */
export function ConcordanceCard() {
  const [showIndex, setShowIndex] = useState(false);

  return (
    <div className="splash-section">
      <h3 className="splash-section-title">Concordance Index</h3>
      <div className="concordance-card">
        <p className="concordance-card-text">
          This index and concordance indexes twenty-one volumes of KGB archival material:
          nine notebooks written by Alexander Vassiliev and twelve compilations of the Soviet
          international telegraphic cables deciphered by the U.S. National Security Agency&apos;s
          Venona project. Indexed are proper names, code names, and organizational titles along
          with some geographic entities, events, diplomatic conferences, and subjects. When
          known, code names are cross-indexed with the real name behind the code name.
        </p>
        <div className="concordance-card-actions">
          <button type="button" className="btn-primary" onClick={() => setShowIndex(true)}>
            View the index
          </button>
          <button
            type="button"
            className="btn-secondary"
            onClick={() => window.open(api.getConcordancePdfUrl(), '_blank')}
            title="Download the original PDF edition — the full Index and Concordance to the Vassiliev Notebooks and Venona cables, as most researchers use it"
          >
            Download PDF
          </button>
          <button
            type="button"
            className="btn-secondary"
            onClick={() => window.open(api.getConcordanceExportUrl(), '_blank')}
            title="Download the full Concordance Index as CSV"
          >
            Download CSV
          </button>
        </div>
      </div>
      {showIndex && <ConcordanceModal onClose={() => setShowIndex(false)} />}
    </div>
  );
}

const PAGE_SIZE = 50;

/** Searchable, paginated browser over the entity/alias concordance. */
function ConcordanceModal({ onClose }: { onClose: () => void }) {
  const [summary, setSummary] = useState<ConcordanceSummary | null>(null);
  const [query, setQuery] = useState('');
  const [entries, setEntries] = useState<ConcordanceEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (q: string, offset: number, append: boolean) => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await api.getConcordanceEntries({ query: q || undefined, limit: PAGE_SIZE, offset });
      setTotal(res.total);
      setEntries((prev) => (append ? [...prev, ...res.entries] : res.entries));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load the index');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Initial load + summary
  useEffect(() => {
    void load('', 0, false);
    api.getConcordanceSummary().then(setSummary).catch(() => { /* summary is optional */ });
  }, [load]);

  // Debounced search
  useEffect(() => {
    const t = setTimeout(() => { void load(query, 0, false); }, 300);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query]);

  return (
    <div className="about-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-label="Concordance Index">
      <div className="about-card concordance-modal" onClick={(e) => e.stopPropagation()}>
        <div className="about-card-header">
          <h2>Concordance Index</h2>
          <button type="button" className="about-close-btn" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>
        <div className="concordance-modal-toolbar">
          <input
            type="text"
            className="concordance-search-input"
            placeholder="Search names, aliases, codenames…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            autoFocus
          />
          <button
            type="button"
            className="btn-secondary"
            onClick={() => window.open(api.getConcordancePdfUrl(), '_blank')}
            title="Download the original PDF edition of the index"
          >
            Download PDF
          </button>
          <button
            type="button"
            className="btn-secondary"
            onClick={() => window.open(api.getConcordanceExportUrl(), '_blank')}
          >
            Download CSV
          </button>
        </div>
        <div className="concordance-modal-meta">
          {summary && (
            <span>
              {summary.entities.toLocaleString()} entities · {summary.aliases.toLocaleString()} aliases
            </span>
          )}
          <span>
            {query ? `${total.toLocaleString()} matches` : `Showing ${entries.length.toLocaleString()} of ${total.toLocaleString()}`}
          </span>
        </div>
        <div className="concordance-modal-body">
          <ConcordanceIntro />
          {error && <div className="search-error">{error}</div>}
          {entries.map((e) => (
            <div key={e.id} className="concordance-entry">
              <div className="concordance-entry-name">
                {e.canonical_name}
                {e.entity_type && <span className="concordance-entry-type">{e.entity_type}</span>}
              </div>
              {e.aliases.length > 0 && (
                <div className="concordance-entry-aliases">
                  {e.aliases.join(' · ')}
                </div>
              )}
              {e.description && (
                <div className="concordance-entry-desc">{e.description}</div>
              )}
            </div>
          ))}
          {isLoading && <div className="loading">Loading…</div>}
          {!isLoading && entries.length === 0 && !error && (
            <div className="search-results-empty"><p>No matching entries.</p></div>
          )}
          {!isLoading && entries.length < total && (
            <button
              type="button"
              className="btn-secondary search-load-more"
              onClick={() => load(query, entries.length, true)}
            >
              Load more ({(total - entries.length).toLocaleString()} remaining)
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * John Earl Haynes's introduction to the printed index, collapsed by default so the
 * search UI stays primary.
 */
function ConcordanceIntro() {
  return (
    <details className="about-section concordance-intro">
      <summary>About this index — by John Earl Haynes</summary>
      <div className="about-section-body">
        <p className="concordance-intro-title">
          Index and Concordance to Alexander Vassiliev&apos;s Notebooks and Soviet Cables
          Deciphered by the National Security Agency&apos;s Venona Project
        </p>
        <p className="concordance-intro-byline">by John Earl Haynes · revised 17 August 2026</p>
        <p>
          This index and concordance indexes twenty-one volumes of KGB archival material: nine
          notebooks written by Alexander Vassiliev and twelve compilations of the Soviet
          international telegraphic cables deciphered by the U.S. National Security Agency&apos;s
          Venona project. Indexed are proper names, code names, and organizational titles along
          with some geographic entities, events, diplomatic conferences, and subjects. When known,
          code names are cross-indexed with the real name behind the code name. Brief biographical
          or explanatory information is provided for significant figures, tradecraft terminology is
          defined, and obscure abbreviations expanded.
        </p>
        <p>
          Alexander Vassiliev&apos;s Notebooks and the Soviet Cables Decrypted by the National
          Security Agency&apos;s Venona Project are the two most reliable guides to code names used
          in Soviet intelligence messages and reports in the 1930s and 1940s. In the case of
          Vassiliev&apos;s notebooks, Vassiliev provided real names for code names provided in the
          original KGB archival material he was using in preparing his notebooks. His material
          covers KGB (and predecessor agencies) activities in the United States in the 1930s and
          1940s with some 1950s material. In the case of the Venona decryptions, real names were
          sometimes provided in the decrypted texts or were deducted by Venona analysts (with FBI
          assistance) based on information supplied in the decrypted texts regarding the
          code-named source&apos;s material, activities, and travel. The bulk of decrypted Venona
          messages deal with KGB activities in the United States in 1943&ndash;1945 with limited
          material from 1941&ndash;42 and post-1945.
        </p>
        <p>
          All of the decrypted messages from American stations are indexed in this concordance. The
          Venona project also decrypted Soviet messages between Moscow and Soviet stations in
          Istanbul, Kazvin, London, Meshed, Mexico City, Montevideo, Ottawa, Paris, Prague, San
          Francisco, Sofia, Stockholm, and Tokyo. The decrypted messages from these non-U.S.
          stations are part of Friday&apos;s database of searchable archival collections but are not
          indexed in this concordance with this exception. Americans and some other significant
          figures who appear in these Moscow-USA volumes are also indexed to where they appear in
          non-USA traffic. There are citations to four non-U.S. volumes: Venona Mexico City KGB,
          Venona Ottawa GRU, Venona London KGB, and Venona London GRU.
        </p>
      </div>
    </details>
  );
}
