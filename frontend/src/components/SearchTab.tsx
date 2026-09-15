'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { api, type SearchCreateRequest, type SearchPageHitItem, type SearchResultSetResponse } from '@/lib/api';
import { SearchResultsList } from './SearchResultsList';
import type { EvidenceRef, UserSelectedScope, CollectionNode } from '@/types/api';
import { ScopeControl } from './ScopeControl';
import { Icon } from './ui/Icon';
import { Popover } from './ui/Popover';
import { plural } from '@/lib/format';
import { toast } from './ui/Toast';

const INITIAL_LIMIT = 100;
const FETCH_MORE_BATCH = 100;
/** How long to keep waiting for a search someone else started (Chat, another tab). */
const RUNNING_POLL_MS = 2000;
const RUNNING_POLL_MAX = 45;

/** Fetch a result set's metadata, waiting while the search is still executing.
 *  Reading a 'running' set gives total_hits = null, which renders as a finished
 *  search with no results. */
async function awaitResultSet(id: string, cancelled: () => boolean) {
  let meta = await api.getSearchResultSet(id);
  for (let i = 0; meta.status === 'running' && i < RUNNING_POLL_MAX; i++) {
    await new Promise((r) => setTimeout(r, RUNNING_POLL_MS));
    if (cancelled()) return meta;
    meta = await api.getSearchResultSet(id);
  }
  return meta;
}

export interface SearchResultBlock {
  resultSetId: string;
  query: string;
  resultSet: SearchResultSetResponse;
  items: SearchPageHitItem[];
  totalHits: number;
  nextCursor: string | null;
  isFetchingMore?: boolean;
  notice?: string | null;  // e.g. sentence relaxed to keywords
  showHidden?: boolean;    // "Show removed" toggle state for this search
  origin?: 'user' | 'chat';       // who ran this search
  originQuery?: string | null;    // chat-origin: the question that spawned it
}

interface SearchTabProps {
  activeScope: UserSelectedScope | null;
  onScopeChange: (scope: UserSelectedScope) => void;
  collections?: CollectionNode[];
  sessionId: number | null;
  onOpenPage: (evidence: EvidenceRef, resultSetId: string) => void;
  /** When set (e.g. from Chat "Open it in Search"), load this result set. */
  externalResultSetId?: string | null;
  /** Called when the user runs a new search (clears external focus). */
  onSearchRun?: () => void;
  /** Searched with no session open — the parent creates one, then queues the query. */
  onStartSession: (query: string) => void;
  /** Query queued from the parent; auto-run once a session is active. */
  pendingSearchQuery?: string | null;
  onPendingSearchConsumed?: () => void;
}

export function SearchTab({
  activeScope, onScopeChange, collections = [], sessionId, onOpenPage,
  externalResultSetId, onSearchRun, onStartSession,
  pendingSearchQuery, onPendingSearchConsumed,
}: SearchTabProps) {
  const [query, setQuery] = useState('');
  const [aliasExpand, setAliasExpand] = useState(true);
  const [fuzzyMode, setFuzzyMode] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [isExpandingFuzzy, setIsExpandingFuzzy] = useState(false);
  const [searchHistory, setSearchHistory] = useState<SearchResultBlock[]>([]);
  const [activeResultSetId, setActiveResultSetId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showChatTabs, setShowChatTabs] = useState(false);
  const [syntaxOpen, setSyntaxOpen] = useState(false);
  const syntaxRef = useRef<HTMLButtonElement>(null);
  // Result sets started from this tab. The backend inserts a search row (status
  // 'running') and commits it *before* running the search, which can take a
  // minute on the full archive — so a concurrent "load this session's saved
  // searches" can see a half-born search, read total_hits as null, and render
  // it as a finished search with zero results. These ids are skipped there: the
  // request that started them is the one that reports them.
  const startedHere = useRef<Set<string>>(new Set());

  const scopeEmpty = activeScope?.mode === 'custom'
    && (activeScope.included_collection_ids?.length ?? 0) === 0
    && (activeScope.included_document_ids?.length ?? 0) === 0;

  const scopeForRequest = useCallback((): SearchCreateRequest['scope'] => {
    if (!activeScope) return { mode: 'full_archive' };
    if (activeScope.mode === 'full_archive') return { mode: 'full_archive' };
    return {
      mode: 'custom',
      included_collection_ids: activeScope.included_collection_ids,
      included_document_ids: activeScope.included_document_ids,
    };
  }, [activeScope]);

  const runSearchWith = useCallback(async (rawQuery: string) => {
    const searchQuery = rawQuery.trim();
    if (!searchQuery || !sessionId) return;
    setError(null);
    setIsSearching(true);
    setIsExpandingFuzzy(false);
    try {
      const req: SearchCreateRequest = {
        session_id: sessionId,
        scope: scopeForRequest(),
        query: searchQuery,
        mode: fuzzyMode ? 'fuzzy' : 'exact',
        unit: 'page',
        sort: 'canonical',
        alias_expand: aliasExpand,
        fuzzy_progressive: fuzzyMode,  // exact first, then expand-fuzzy in the background
      };
      const res = await api.createSearchResultSet(req);
      startedHere.current.add(res.result_set_id);
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
      setSearchHistory((prev) => {
        const idx = prev.findIndex((b) => b.resultSetId === block.resultSetId);
        if (idx < 0) return [...prev, block];
        const next = [...prev];
        next[idx] = block;
        return next;
      });
      setActiveResultSetId(res.result_set_id);
      setIsSearching(false);

      // Progressive fuzzy: expand in the background, then refetch and update the block.
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
  }, [aliasExpand, fuzzyMode, sessionId, scopeForRequest, onSearchRun]);

  const runSearch = useCallback(() => {
    const q = query.trim();
    if (!q || scopeEmpty) return;
    if (!sessionId) {
      onStartSession(q);
      return;
    }
    void runSearchWith(q);
  }, [runSearchWith, query, sessionId, onStartSession, scopeEmpty]);

  // Reload the session's saved searches when the session changes (search history is
  // persisted server-side, so it survives reloads and session switches — like chat).
  useEffect(() => {
    setError(null);
    setSearchHistory([]);
    setActiveResultSetId(null);
    // Per-session: on the way back into a session its own searches must load
    // normally. Protection for a search in flight *right now* comes from the
    // 'running' status check below.
    startedHere.current.clear();
    if (!sessionId) return;
    let cancelled = false;
    (async () => {
      try {
        const summaries = (await api.listSearchResultSets(sessionId))
          .filter((s) => !startedHere.current.has(s.id));
        if (cancelled || summaries.length === 0) return;
        const blocks = await Promise.all(
          summaries.map(async (s): Promise<SearchResultBlock | null> => {
            try {
              const [meta, data] = await Promise.all([
                api.getSearchResultSet(s.id),
                api.getSearchResultSetItems(s.id, { limit: INITIAL_LIMIT }),
              ]);
              // Still executing somewhere else: showing it now would report a
              // finished search with no hits. It will be there next time.
              if (meta.status === 'running') return null;
              return {
                resultSetId: s.id,
                query: s.query_display || s.query_raw || meta.query_display || 'Search',
                resultSet: meta,
                items: data.items,
                totalHits: meta.total_hits ?? s.total_hits ?? 0,
                nextCursor: data.next_cursor ?? null,
                origin: s.origin ?? 'user',
                originQuery: s.origin_query ?? null,
              };
            } catch {
              return null;
            }
          }),
        );
        if (cancelled) return;
        const loaded = blocks.filter((b): b is SearchResultBlock => b !== null);
        if (loaded.length === 0) return;
        // Merge, never replace: a search started while this was in flight must
        // survive, and it stays selected if the user is already looking at it.
        setSearchHistory((prev) => {
          const byId = new Map(loaded.map((b) => [b.resultSetId, b]));
          for (const b of prev) byId.set(b.resultSetId, b);
          return [...byId.values()];
        });
        setActiveResultSetId((cur) => {
          if (cur) return cur;
          // The most recent search is the active tab on reload — preferring the
          // researcher's own searches so Chat's don't steal focus.
          const lastUser = [...loaded].reverse().find((b) => b.origin !== 'chat');
          return (lastUser ?? loaded[loaded.length - 1]).resultSetId;
        });
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Failed to load saved searches');
      }
    })();
    return () => { cancelled = true; };
  }, [sessionId]);

  // Auto-run a query queued while the session was being created.
  useEffect(() => {
    if (!sessionId || !pendingSearchQuery) return;
    const q = pendingSearchQuery;
    onPendingSearchConsumed?.();
    setQuery(q);
    void runSearchWith(q);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, pendingSearchQuery]);

  // Load a result set opened from Chat and append it to this session's tabs.
  useEffect(() => {
    if (!externalResultSetId) return;
    let cancelled = false;
    (async () => {
      try {
        const meta = await awaitResultSet(externalResultSetId, () => cancelled);
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
          const idx = prev.findIndex((b) => b.resultSetId === externalResultSetId);
          if (idx < 0) return [...prev, block];
          const next = [...prev];
          next[idx] = block;
          return next;
        });
        setActiveResultSetId(externalResultSetId);
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
      // Prefetch snippets for the next batch (best-effort; the items API works without it).
      try {
        await api.fetchMoreSearchSnippets(resultSetId, FETCH_MORE_BATCH);
      } catch {
        /* snippet fetch can fail; still fetch items */
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

  // Close a search tab (deletes the saved search server-side).
  const handleCloseTab = useCallback(async (resultSetId: string) => {
    const block = searchHistory.find((b) => b.resultSetId === resultSetId);
    if (!window.confirm(`Delete the search “${block?.query ?? ''}” and its results?`)) return;
    try {
      await api.deleteSearchResultSet(resultSetId);
    } catch (e) {
      // A toast, not just the inline notice: the chip staying put with a quiet
      // red line elsewhere on the page reads as "nothing happened".
      const why = e instanceof Error ? e.message : 'unknown error';
      toast(`Could not delete that search. ${why}`);
      setError(`Could not delete the search “${block?.query ?? ''}”: ${why}`);
      return;
    }
    setSearchHistory((prev) => {
      const idx = prev.findIndex((b) => b.resultSetId === resultSetId);
      const next = prev.filter((b) => b.resultSetId !== resultSetId);
      setActiveResultSetId((cur) => {
        if (cur !== resultSetId) return cur;
        if (next.length === 0) return null;
        return next[Math.max(0, idx - 1)].resultSetId;
      });
      return next;
    });
  }, [searchHistory]);

  // Hide or restore a single hit (persists server-side, reversible; numbering skips hidden rows).
  const handleSetItemHidden = useCallback(async (resultSetId: string, item: SearchPageHitItem, hidden: boolean) => {
    try {
      await api.setSearchResultItemHidden(resultSetId, item.document.id, item.page.id, hidden);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to update result');
      return;
    }
    setSearchHistory((prev) => {
      const idx = prev.findIndex((b) => b.resultSetId === resultSetId);
      if (idx < 0) return prev;
      const b = prev[idx];
      const next = [...prev];
      next[idx] = {
        ...b,
        items: b.items.map((it) =>
          it.document.id === item.document.id && it.page.id === item.page.id
            ? { ...it, hidden }
            : it),
      };
      return next;
    });
  }, []);

  const toggleShowHidden = useCallback((resultSetId: string) => {
    setSearchHistory((prev) => {
      const idx = prev.findIndex((b) => b.resultSetId === resultSetId);
      if (idx < 0) return prev;
      const next = [...prev];
      next[idx] = { ...next[idx], showHidden: !next[idx].showHidden };
      return next;
    });
  }, []);

  const userBlocks = searchHistory.filter((b) => b.origin !== 'chat');
  const chatBlocks = searchHistory.filter((b) => b.origin === 'chat');
  const activeIsChat = chatBlocks.some((b) => b.resultSetId === activeResultSetId);

  const renderChip = (block: SearchResultBlock, isChat: boolean) => (
    <div
      key={block.resultSetId}
      className={
        `search-chip${isChat ? ' search-chip-chat' : ''}`
        + `${block.resultSetId === activeResultSetId ? ' is-active' : ''}`
      }
    >
      <button
        type="button"
        role="tab"
        aria-selected={block.resultSetId === activeResultSetId}
        className="search-chip-label"
        onClick={() => setActiveResultSetId(block.resultSetId)}
        title={
          isChat
            ? `Chat ran this search${block.originQuery ? ` while answering: “${block.originQuery}”` : ''} (${block.totalHits} hits)`
            : `${block.query} (${block.totalHits} hits)`
        }
      >
        {isChat && <Icon name="spark" size={12} />}
        {block.query}
      </button>
      <button
        type="button"
        className="search-chip-close"
        onClick={() => handleCloseTab(block.resultSetId)}
        aria-label={`Delete search: ${block.query}`}
        title="Delete this search"
      >
        <Icon name="close" size={13} />
      </button>
    </div>
  );

  return (
    <div className="search">
      <div className="search-head">
        <div className="search-head-inner">
          <div className="search-bar">
            <label className="field">
              <Icon name="search" size={16} />
              <input
                type="text"
                placeholder='"Harry Dexter White" OR (Rosenberg AND Soviet)'
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && runSearch()}
                aria-label="Search query"
              />
            </label>
            <button
              type="button"
              className="btn-primary btn-lg"
              onClick={runSearch}
              disabled={isSearching || !query.trim() || scopeEmpty}
            >
              {isSearching ? <span className="spinner" /> : <Icon name="search" size={16} />}
              {isSearching ? 'Searching…' : 'Search'}
            </button>
          </div>

          <div className="search-opts">
            <ScopeControl
              scope={activeScope}
              collections={collections}
              onChange={onScopeChange}
              placement="bottom"
            />
            <Toggle label="Fuzzy" hint="Tolerate OCR errors and typos" on={fuzzyMode} onToggle={() => setFuzzyMode((v) => !v)} />
            <Toggle label="Aliases" hint="Expand names to known cover names" on={aliasExpand} onToggle={() => setAliasExpand((v) => !v)} />
            <div className="spacer" />
            <button
              ref={syntaxRef}
              type="button"
              className="btn-link"
              onClick={() => setSyntaxOpen((v) => !v)}
              aria-expanded={syntaxOpen}
            >
              <Icon name="help" size={14} />
              Syntax
            </button>
            <Popover
              anchorRef={syntaxRef}
              open={syntaxOpen}
              onClose={() => setSyntaxOpen(false)}
              placement="bottom"
              align="end"
              label="Search syntax"
            >
              <div className="popover-head"><span className="popover-title">Search syntax</span></div>
              <div className="popover-body" style={{ padding: 'var(--s-4)', maxWidth: '26rem' }}>
                <div className="prose" style={{ fontSize: 'var(--text-base)' }}>
                  <ul>
                    <li><code>Harry AND White</code>: both terms must appear</li>
                    <li><code>Rosenberg OR Hiss</code>: either term</li>
                    <li><code>Soviet NOT Rosenberg</code>: exclude a term</li>
                    <li><code>&quot;Harry Dexter White&quot;</code>: exact phrase</li>
                    <li><code>(Rosenberg OR Hiss) AND Soviet</code>: group with parentheses</li>
                  </ul>
                  <p>
                    Boolean operators work in exact mode only. <strong>Fuzzy</strong> handles OCR
                    errors and typos but ignores operators. <strong>Aliases</strong> expands names
                    to known cover names and codenames.
                  </p>
                </div>
              </div>
            </Popover>
          </div>
        </div>
      </div>

      {(userBlocks.length > 0 || chatBlocks.length > 0) && (
        <div className="search-tabs" role="tablist" aria-label="Searches in this session">
          {userBlocks.map((b) => renderChip(b, false))}
          {chatBlocks.length > 0 && (
            <>
              <button
                type="button"
                className="search-chip-more"
                onClick={() => setShowChatTabs((v) => !v)}
                aria-expanded={showChatTabs || activeIsChat}
                title="Searches Chat ran while answering your questions. Open one to carry on where it left off."
              >
                <Icon name="spark" size={12} />
                From Chat
                <span className="count">{chatBlocks.length}</span>
                <Icon name={showChatTabs || activeIsChat ? 'chevron-up' : 'chevron-down'} size={12} />
              </button>
              {(showChatTabs || activeIsChat) && chatBlocks.map((b) => renderChip(b, true))}
            </>
          )}
        </div>
      )}

      <div className="search-body">
        <div className="search-body-inner">
          {error && <div className="notice notice-danger"><Icon name="info" size={16} />{error}</div>}

          {isSearching && (
            <div className="loading">
              <span className="spinner" />
              Searching collections. The whole archive can take up to a minute.
            </div>
          )}
          {isExpandingFuzzy && !isSearching && (
            <div className="loading">
              <span className="spinner" />
              Loading fuzzy matches. Exact results are shown below.
            </div>
          )}

          {!isSearching && searchHistory.length === 0 && (
            <div className="empty-state">
              <Icon name="search" size={22} />
              <p>
                Search returns every page that matches your terms. Hits are numbered, so you
                can work through them and pick up where you left off.
              </p>
              <div className="suggestions" style={{ justifyContent: 'center' }}>
                {['"Harry Dexter White"', 'Rosenberg OR Hiss', 'Silvermaster AND film'].map((q) => (
                  <button
                    key={q}
                    type="button"
                    className="suggestion"
                    onClick={() => { setQuery(q); if (sessionId) void runSearchWith(q); else onStartSession(q); }}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {searchHistory.filter((block) => block.resultSetId === activeResultSetId).map((block) => {
            const coverage = block.resultSet.coverage_json as {
              collections_searched?: number;
              collections_total?: number;
              collections?: { id: number; title: string; hits: number }[];
            } | undefined;
            const terms = block.resultSet.expanded_terms_json as Record<string, string[]> | undefined;
            const hiddenCount = block.items.filter((it) => it.hidden).length;

            return (
              <div key={block.resultSetId}>
                {block.notice && (
                  <div className="notice" style={{ marginBottom: 'var(--s-3)' }}>
                    <Icon name="info" size={16} />{block.notice}
                  </div>
                )}

                <div className="search-meta">
                  <span className="search-meta-figure">{block.totalHits.toLocaleString()}</span>
                  <span>page {block.totalHits === 1 ? 'hit' : 'hits'} for “{block.query}”</span>
                  {block.resultSet.is_exhaustive === false && (
                    <span className="chip chip-amber">approximate</span>
                  )}
                  {coverage?.collections_searched != null && (
                    <span className="count">
                      {coverage.collections_searched}/{coverage.collections_total ?? '?'} collections
                    </span>
                  )}
                  <div className="spacer" />
                  {hiddenCount > 0 && (
                    <button
                      type="button"
                      className="btn-link"
                      onClick={() => toggleShowHidden(block.resultSetId)}
                    >
                      <Icon name={block.showHidden ? 'eye-off' : 'restore'} size={14} />
                      {block.showHidden ? 'Hide removed' : `Show ${hiddenCount} removed`}
                    </button>
                  )}
                  <button
                    type="button"
                    className="btn-secondary btn-sm"
                    onClick={() => handleExport(block.resultSetId)}
                  >
                    <Icon name="download" size={14} />
                    CSV
                  </button>

                  {(terms && Object.keys(terms).length > 0) || coverage?.collections?.length ? (
                    <div className="search-meta-detail">
                      {terms && Object.entries(terms).map(([term, aliases]) => (
                        <span key={term}>
                          {term} → {(aliases ?? []).slice(0, 5).join(', ')}
                          {(aliases ?? []).length > 5 ? '…' : ''}
                        </span>
                      ))}
                      {coverage?.collections?.map((c) => (
                        <span key={c.id}>{c.title}: {plural(c.hits, 'page')}</span>
                      ))}
                    </div>
                  ) : null}
                </div>

                <SearchResultsList
                  items={block.items}
                  totalHits={block.totalHits}
                  onOpenPage={onOpenPage}
                  resultSetId={block.resultSetId}
                  isLoading={false}
                  showHidden={!!block.showHidden}
                  onSetItemHidden={(item, hidden) => handleSetItemHidden(block.resultSetId, item, hidden)}
                />

                {block.totalHits > block.items.length && block.nextCursor != null && (
                  <button
                    type="button"
                    className="btn-secondary"
                    style={{ marginTop: 'var(--s-3)' }}
                    onClick={() => loadMore(block.resultSetId)}
                    disabled={block.isFetchingMore}
                  >
                    {block.isFetchingMore ? <span className="spinner" /> : <Icon name="chevron-down" size={14} />}
                    {block.isFetchingMore
                      ? 'Loading…'
                      : `Load more (${(block.totalHits - block.items.length).toLocaleString()} remaining)`}
                  </button>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function Toggle({
  label, hint, on, onToggle,
}: {
  label: string;
  hint: string;
  on: boolean;
  onToggle: () => void;
}) {
  return (
    <span className="switch" title={hint}>
      <button
        type="button"
        role="switch"
        aria-checked={on}
        aria-label={`${label}: ${on ? 'on' : 'off'}`}
        className="switch-track"
        onClick={onToggle}
      >
        <span className="switch-knob" />
      </button>
      <span style={{ color: on ? 'var(--ink)' : undefined }}>{label}</span>
    </span>
  );
}
