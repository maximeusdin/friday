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
        {/* Placeholder copy — replace with real description */}
        <p className="concordance-card-text">
          Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do
          eiusmod tempor incididunt ut labore et dolore magna aliqua — the
          Concordance Index is the master index of people, organizations, and
          cover names that Friday uses to expand aliases and resolve codenames
          across the archives. Ut enim ad minim veniam, quis nostrud
          exercitation ullamco laboris.
        </p>
        <div className="concordance-card-actions">
          <button type="button" className="btn-primary" onClick={() => setShowIndex(true)}>
            View the index
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
