'use client';

import { useState, useCallback, useEffect } from 'react';
import { api, type SearchCreateRequest, type SearchPageHitItem, type SearchResultSetResponse } from '@/lib/api';
import { SearchResultsList } from './SearchResultsList';
import type { EvidenceRef } from '@/types/api';
import type { UserSelectedScope, CollectionNode } from '@/types/api';

const INITIAL_LIMIT = 100;
const FETCH_MORE_BATCH = 100;

export interface SearchResultBlock {
  resultSetId: string;
  query: string;
  resultSet: SearchResultSetResponse;
  items: SearchPageHitItem[];
  totalHits: number;
  nextCursor: string | null;
  isFetchingMore?: boolean;
  notice?: string | null;  // e.g. sentence relaxed to keywords
}

interface SearchTabProps {
  activeScope: UserSelectedScope | null;
  sessionId: number | null;
  onOpenPage: (evidence: EvidenceRef, resultSetId: string) => void;
  /** When set (e.g. from Chat "View in Search tab"), load this result set. */
  externalResultSetId?: string | null;
  /** Called when user runs a new search (clears external focus). */
  onSearchRun?: () => void;
  /** Collection nodes for scope display */
  collections?: CollectionNode[];
}

export function SearchTab({ activeScope, sessionId, onOpenPage, externalResultSetId, onSearchRun, collections = [] }: SearchTabProps) {
  const [query, setQuery] = useState('');
  const [aliasExpand, setAliasExpand] = useState(true);
  const [fuzzyMode, setFuzzyMode] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [isExpandingFuzzy, setIsExpandingFuzzy] = useState(false);
  const [searchHistory, setSearchHistory] = useState<SearchResultBlock[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [showInstructions, setShowInstructions] = useState(true);

  const scopeForRequest = useCallback((): SearchCreateRequest['scope'] => {
    if (!activeScope) return { mode: 'full_archive' };
    if (activeScope.mode === 'full_archive') return { mode: 'full_archive' };
    return {
      mode: 'custom',
      included_collection_ids: activeScope.included_collection_ids,
      included_document_ids: activeScope.included_document_ids,
    };
  }, [activeScope]);

  const scopeLabel = useCallback((): string => {
    if (!activeScope) return 'Full archive';
    if (activeScope.mode === 'full_archive') return 'Full archive';
    const collIds = activeScope.included_collection_ids ?? [];
    if (collIds.length === 0) return 'No collections selected';
    const titles = collIds
      .map((id) => {
        const c = collections.find((col) => col.id === id);
        return c?.title || c?.slug || `Collection ${id}`;
      })
      .slice(0, 3);
    return collIds.length > 3 ? `${titles.join(', ')} +${collIds.length - 3} more` : titles.join(', ');
  }, [activeScope, collections]);

  const runSearch = useCallback(async () => {
    if (!query.trim() || !sessionId) return;
    setError(null);
    setIsSearching(true);
    setIsExpandingFuzzy(false);
    const searchQuery = query.trim();
    try {
      const req: SearchCreateRequest = {
        session_id: sessionId,
        scope: scopeForRequest(),
        query: searchQuery,
        mode: fuzzyMode ? 'fuzzy' : 'exact',
        unit: 'page',
        sort: 'canonical',
        alias_expand: aliasExpand,
        fuzzy_progressive: fuzzyMode,  // Exact first, then expand-fuzzy in background
      };
      const res = await api.createSearchResultSet(req);
      onSearchRun?.();
      const meta = await api.getSearchResultSet(res.result_set_id);
      const data = await api.getSearchResultSetItems(res.result_set_id, { limit: INITIAL_LIMIT });
      let block: SearchResultBlock = {
        resultSetId: res.result_set_id,
        query: searchQuery,
        resultSet: meta,
        items: data.items,
        totalHits: meta.total_hits ?? 0,
        nextCursor: data.next_cursor ?? null,
        notice: res.notice ?? null,
      };
      setSearchHistory((prev) => [...prev, block]);
      setIsSearching(false);

      // Progressive fuzzy: expand in background, then refetch and update block
      if (res.fuzzy_pending && res.result_set_id) {
        setIsExpandingFuzzy(true);
        try {
          const expandRes = await api.expandSearchFuzzy(res.result_set_id);
          const meta2 = await api.getSearchResultSet(res.result_set_id);
          const data2 = await api.getSearchResultSetItems(res.result_set_id, {
            limit: Math.min(meta2.total_hits ?? 500, 500),
          });
          block = {
            resultSetId: res.result_set_id,
            query: searchQuery,
            resultSet: meta2,
            items: data2.items,
            totalHits: meta2.total_hits ?? expandRes.total_hits ?? 0,
            nextCursor: data2.next_cursor ?? null,
          };
          setSearchHistory((prev) => {
            const idx = prev.findIndex((b) => b.resultSetId === res.result_set_id);
            if (idx < 0) return prev;
            const next = [...prev];
            next[idx] = block;
            return next;
          });
        } catch (e2) {
          setError(e2 instanceof Error ? e2.message : 'Fuzzy expansion failed');
        } finally {
          setIsExpandingFuzzy(false);
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Search failed');
    } finally {
      setIsSearching(false);
    }
  }, [query, aliasExpand, fuzzyMode, sessionId, scopeForRequest, onSearchRun]);

  // Clear search state when session changes (search is session-scoped)
  useEffect(() => {
    setSearchHistory([]);
    setError(null);
  }, [sessionId]);

  // When externalResultSetId is set (e.g. from Chat "View in Search tab"), load that result set and append
  useEffect(() => {
    if (!externalResultSetId) return;
    let cancelled = false;
    (async () => {
      try {
        const meta = await api.getSearchResultSet(externalResultSetId);
        if (cancelled) return;
        const data = await api.getSearchResultSetItems(externalResultSetId, { limit: INITIAL_LIMIT });
        if (cancelled) return;
        const block: SearchResultBlock = {
          resultSetId: externalResultSetId,
          query: meta.query_display ?? 'Search results',
          resultSet: meta,
          items: data.items,
          totalHits: meta.total_hits ?? 0,
          nextCursor: data.next_cursor ?? null,
        };
        setSearchHistory((prev) => {
          if (prev.some((b) => b.resultSetId === externalResultSetId)) return prev;
          return [...prev, block];
        });
        setError(null);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Failed to load search results');
      }
    })();
    return () => { cancelled = true; };
  }, [externalResultSetId]);

  const loadMore = useCallback(async (resultSetId: string) => {
    setSearchHistory((prev) => {
      const idx = prev.findIndex((b) => b.resultSetId === resultSetId);
      if (idx < 0) return prev;
      const block = prev[idx];
      if (!block.nextCursor) return prev;
      const next = [...prev];
      next[idx] = { ...block, isFetchingMore: true };
      return next;
    });
    setError(null);
    const block = searchHistory.find((b) => b.resultSetId === resultSetId);
    if (!block?.nextCursor) return;
    try {
      // Prefetch snippets for next batch (best-effort; items API works without it)
      try {
        await api.fetchMoreSearchSnippets(resultSetId, FETCH_MORE_BATCH);
      } catch {
        // Snippet fetch can fail; still fetch items
      }
      const data = await api.getSearchResultSetItems(resultSetId, {
        cursor: block.nextCursor,
        limit: FETCH_MORE_BATCH,
      });
      setSearchHistory((prev) => {
        const idx = prev.findIndex((b) => b.resultSetId === resultSetId);
        if (idx < 0) return prev;
        const b = prev[idx];
        const next = [...prev];
        next[idx] = {
          ...b,
          items: [...b.items, ...data.items],
          nextCursor: data.next_cursor ?? null,
          isFetchingMore: false,
        };
        return next;
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Load more failed');
      setSearchHistory((prev) => {
        const idx = prev.findIndex((b) => b.resultSetId === resultSetId);
        if (idx < 0) return prev;
        const next = [...prev];
        next[idx] = { ...next[idx], isFetchingMore: false };
        return next;
      });
    }
  }, [searchHistory]);

  const handleExport = useCallback((resultSetId: string) => {
    window.open(api.getSearchResultSetExportUrl(resultSetId, 'csv'), '_blank');
  }, []);

  // Require session (like Chat)
  if (!sessionId) {
    return (
      <div className="pane-content splash-content">
        <div className="splash-hero">
          <div className="splash-badge">Concordance Search</div>
          <h2 className="splash-title">Search</h2>
          <p className="splash-tagline">Boolean search across Cold War archives.</p>
          <p className="splash-subtitle">
            Create or select a session in the sidebar to begin. Search and Chat share the same session — your searches and conversations persist together.
          </p>
        </div>
        <div className="splash-section splash-examples">
          <h3 className="splash-section-title">Example queries</h3>
          <div className="splash-example-grid">
            <div className="splash-example-card">
              <span className="splash-example-icon">&#x1F50E;</span>
              <span className="splash-example-text">&quot;Harry Dexter White&quot;</span>
            </div>
            <div className="splash-example-card">
              <span className="splash-example-icon">&#x1F50E;</span>
              <span className="splash-example-text">Rosenberg OR Hiss</span>
            </div>
            <div className="splash-example-card">
              <span className="splash-example-icon">&#x1F50E;</span>
              <span className="splash-example-text">Soviet AND agent</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="search-tab">
      {/* Instructions (collapsible) */}
      <div className="search-instructions">
        <button
          type="button"
          className="search-instructions-toggle"
          onClick={() => setShowInstructions((s) => !s)}
          aria-expanded={showInstructions}
        >
          {showInstructions ? '▼' : '▶'} How to search
        </button>
        {showInstructions && (
          <div className="search-instructions-content">
            <p><strong>Boolean operators</strong> (exact mode only): <code>AND</code>, <code>OR</code>, <code>NOT</code>, parentheses.</p>
            <ul>
              <li><code>Harry AND White</code> — both terms must appear</li>
              <li><code>Rosenberg OR Hiss</code> — either term</li>
              <li><code>Soviet NOT Rosenberg</code> — exclude term</li>
              <li><code>&quot;Harry Dexter White&quot;</code> — exact phrase (use quotes)</li>
              <li><code>(Rosenberg OR Hiss) AND Soviet</code> — combine with parentheses</li>
            </ul>
            <p><strong>Fuzzy matching</strong> (toggle below): Off by default. Turn <strong>On</strong> to handle OCR errors and typos. Ignores boolean operators.</p>
            <p><strong>Alias expansion</strong> (toggle below): Expands names to known aliases and codenames. Venona/Vassiliev only when enabled.</p>
            <p className="search-instructions-note">Scope from the right panel applies to Search — narrow collections to speed up queries.</p>
          </div>
        )}
      </div>

      {/* Query bar */}
      <div className="search-query-bar">
        <input
          type="text"
          className="search-query-input"
          placeholder='e.g. "Harry Dexter White" OR (Rosenberg AND Soviet)'
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && runSearch()}
        />
        <div className="search-toggles">
          <label className="search-toggle-slider">
            <span className="search-toggle-label">Fuzzy</span>
            <button
              type="button"
              role="switch"
              aria-checked={fuzzyMode}
              aria-label={`Fuzzy matching ${fuzzyMode ? 'on' : 'off'}`}
              className={`search-slider ${fuzzyMode ? 'search-slider-on' : ''}`}
              onClick={() => setFuzzyMode((v) => !v)}
              title="Fuzzy: handles OCR/typos"
            >
              <span className="search-slider-knob" />
            </button>
          </label>
          <label className="search-toggle-slider">
            <span className="search-toggle-label">Aliases</span>
            <button
              type="button"
              role="switch"
              aria-checked={aliasExpand}
              aria-label={`Alias expansion ${aliasExpand ? 'on' : 'off'}`}
              className={`search-slider ${aliasExpand ? 'search-slider-on' : ''}`}
              onClick={() => setAliasExpand((v) => !v)}
              title="Expand aliases (codename lookup)"
            >
              <span className="search-slider-knob" />
            </button>
          </label>
        </div>
        <button
          type="button"
          className="btn-primary search-btn"
          onClick={runSearch}
          disabled={isSearching || !query.trim()}
        >
          {isSearching ? 'Searching…' : 'Search'}
        </button>
      </div>

      {/* Scope indicator */}
      <div className="search-scope-bar">
        <span className="search-scope-label">Scope:</span>
        <span className="search-scope-value">{scopeLabel()}</span>
      </div>

      {/* Progress indicator */}
      {isSearching && (
        <div className="search-progress">
          <div className="search-progress-spinner" />
          <span>Searching collections… Full archive may take 10–60 seconds.</span>
        </div>
      )}
      {isExpandingFuzzy && !isSearching && (
        <div className="search-progress">
          <div className="search-progress-spinner" />
          <span>Loading fuzzy matches… (exact results shown above)</span>
        </div>
      )}

      {error && (
        <div className="search-error">{error}</div>
      )}

      <div className="search-results-container">
        {searchHistory.map((block) => (
          <div key={block.resultSetId} className="search-result-block">
            {block.notice && (
              <div className="search-notice" style={{
                margin: '0 0 8px', padding: '6px 10px', borderRadius: 6,
                background: 'var(--surface-hover, #f3f4f6)', color: 'var(--text-secondary, #555)',
                fontSize: 13, lineHeight: 1.4,
              }}>
                {block.notice}
              </div>
            )}
            <div className="search-coverage-panel">
              <div className="search-coverage-stats">
                {block.resultSet.is_exhaustive === false && (
                  <span className="search-approximate-label">Approximate matches</span>
                )}
                <span className="search-block-query">&quot;{block.query}&quot;</span>
                <span>Total page hits: {block.totalHits}</span>
                {(() => {
                  const cov = block.resultSet.coverage_json as { collections_searched?: number; collections_total?: number } | undefined;
                  return cov?.collections_searched != null ? (
                    <span>Collections: {cov.collections_searched}/{cov.collections_total ?? '?'}</span>
                  ) : null;
                })()}
              </div>
              {(() => {
                const terms = block.resultSet.expanded_terms_json as Record<string, string[]> | undefined;
                if (!terms || Object.keys(terms).length === 0) return null;
                const text = Object.entries(terms)
                  .map(([term, aliases]) => `${term} → ${(aliases ?? []).slice(0, 5).join(', ')}${(aliases ?? []).length > 5 ? '…' : ''}`)
                  .join('; ');
                return (
                  <div className="search-expanded-terms">
                    Expanded: {text}
                  </div>
                );
              })()}
              {(() => {
                const cov = block.resultSet.coverage_json as { collections?: { id: number; title: string; hits: number }[] } | undefined;
                return cov?.collections && cov.collections.length > 0 ? (
                  <div className="search-coverage-breakdown">
                    {cov.collections.map((c) => (
                      <span key={c.id} className="search-coverage-item">
                        {c.title}: {c.hits} pages
                      </span>
                    ))}
                  </div>
                ) : null;
              })()}
              <button type="button" className="btn-secondary" onClick={() => handleExport(block.resultSetId)}>
                Export CSV
              </button>
            </div>
            <SearchResultsList
              items={block.items}
              totalHits={block.totalHits}
              onOpenPage={onOpenPage}
              resultSetId={block.resultSetId}
              isLoading={false}
            />
            {block.totalHits > block.items.length && block.nextCursor != null && (
              <button
                type="button"
                className="btn-secondary search-load-more"
                onClick={() => loadMore(block.resultSetId)}
                disabled={block.isFetchingMore}
              >
                {block.isFetchingMore ? 'Loading…' : `Load more (${block.totalHits - block.items.length} remaining)`}
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
