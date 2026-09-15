'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '@/lib/api';
import { describeScope, scopeFingerprint, scopeDocumentCount } from '@/lib/scope';
import { plural } from '@/lib/format';
import type { CollectionNode, UserSelectedScope } from '@/types/api';
import { Icon } from './ui/Icon';
import { Popover } from './ui/Popover';
import { toast } from './ui/Toast';

/**
 * ScopeControl — one control for "what does the archive mean right now".
 *
 * This replaces a staged-commit panel (draft → "Apply to Next Query" → active,
 * plus Revert, a rebase banner, a dirty banner, a diff strip and a separate
 * always-on scope bar) that spent a quarter of the screen explaining its own
 * state machine. There was never anything to commit: a query reads the scope
 * when you press Ask, so selections can simply take effect as you make them.
 *
 * What's left is the pattern every filter UI converged on: a chip showing the
 * current filter next to the thing it filters, a picker behind it, and Undo
 * instead of Revert. The chip lives in the composer and in the search bar, so
 * scope is always visible exactly where it changes the outcome.
 */

const FULL: UserSelectedScope = { mode: 'full_archive' };
const RECENTS_KEY = 'friday.recentScopes';
const MAX_RECENTS = 4;

interface RecentScope {
  label: string;
  scope: UserSelectedScope;
}

function readRecents(): RecentScope[] {
  try {
    const raw = localStorage.getItem(RECENTS_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.slice(0, MAX_RECENTS) : [];
  } catch {
    return [];
  }
}

function writeRecent(entry: RecentScope): void {
  try {
    const fp = scopeFingerprint(entry.scope);
    const next = [entry, ...readRecents().filter((r) => scopeFingerprint(r.scope) !== fp)];
    localStorage.setItem(RECENTS_KEY, JSON.stringify(next.slice(0, MAX_RECENTS)));
  } catch {
    /* storage unavailable — recents are a convenience, never a requirement */
  }
}

interface ScopeControlProps {
  scope: UserSelectedScope | null;
  collections: CollectionNode[];
  onChange: (scope: UserSelectedScope) => void;
  /** Placement hint: the composer opens upward, the search bar downward. */
  placement?: 'top' | 'bottom';
  disabled?: boolean;
}

export function ScopeControl({
  scope, collections, onChange, placement = 'top', disabled,
}: ScopeControlProps) {
  const anchor = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const active = scope ?? FULL;

  const isCustom = active.mode === 'custom';
  const colCount = active.included_collection_ids?.length ?? 0;
  const docCount = active.included_document_ids?.length ?? 0;
  const empty = isCustom && colCount === 0 && docCount === 0;

  // Apply immediately, and hand back a one-click Undo instead of a Revert button.
  const apply = useCallback((next: UserSelectedScope, note?: string) => {
    const previous = active;
    onChange(next);
    if (note) {
      toast(note, {
        label: 'Undo',
        onSelect: () => onChange(previous),
      });
    }
  }, [active, onChange]);

  return (
    <>
      <button
        ref={anchor}
        type="button"
        className="scope-chip"
        data-state={empty ? 'empty' : isCustom ? 'custom' : 'all'}
        onClick={() => setOpen((v) => !v)}
        disabled={disabled}
        aria-expanded={open}
        aria-haspopup="dialog"
        title="Choose which collections this session searches"
      >
        <Icon name="layers" size={14} className="scope-chip-icon" />
        <span className="scope-chip-text">{describeScope(active, collections)}</span>
        <Icon name="chevron-down" size={14} className="scope-chip-icon" />
      </button>

      <Popover
        anchorRef={anchor}
        open={open}
        onClose={() => setOpen(false)}
        placement={placement}
        align="start"
        label="Search scope"
        className="scope-picker"
      >
        <ScopePicker
          scope={active}
          collections={collections}
          onApply={apply}
          onClose={() => setOpen(false)}
        />
      </Popover>
    </>
  );
}

// =============================================================================
// Picker body
// =============================================================================

function ScopePicker({
  scope, collections, onApply, onClose,
}: {
  scope: UserSelectedScope;
  collections: CollectionNode[];
  onApply: (next: UserSelectedScope, note?: string) => void;
  onClose: () => void;
}) {
  const [filter, setFilter] = useState('');
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  const [docs, setDocs] = useState<Record<number, CollectionNode['documents']>>({});
  const [recents, setRecents] = useState<RecentScope[]>([]);

  useEffect(() => { setRecents(readRecents()); }, []);

  const selectedCols = useMemo(
    () => new Set(scope.included_collection_ids || []),
    [scope.included_collection_ids],
  );
  const selectedDocs = useMemo(
    () => new Set(scope.included_document_ids || []),
    [scope.included_document_ids],
  );
  const isCustom = scope.mode === 'custom';

  const set = (cols: Iterable<number>, docIds: Iterable<number>, note?: string) => {
    const next: UserSelectedScope = {
      mode: 'custom',
      included_collection_ids: Array.from(cols),
      included_document_ids: Array.from(docIds),
    };
    onApply(next, note);
    if ((next.included_collection_ids?.length ?? 0) + (next.included_document_ids?.length ?? 0) > 0) {
      writeRecent({ label: describeScope(next, collections), scope: next });
    }
  };

  const loadDocs = useCallback(async (colId: number) => {
    if (docs[colId]) return;
    try {
      const list = await api.getCollectionDocuments(colId);
      setDocs((prev) => ({ ...prev, [colId]: list }));
    } catch {
      setDocs((prev) => ({ ...prev, [colId]: [] }));
    }
  }, [docs]);

  const toggleExpand = (colId: number) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(colId)) next.delete(colId);
      else { next.add(colId); void loadDocs(colId); }
      return next;
    });
  };

  /** Whole-collection toggle. Clearing also drops any individually-picked
   *  documents inside it, so the two selection levels can't contradict. */
  const toggleCollection = (col: CollectionNode) => {
    const known = docs[col.id] ?? [];
    const cols = new Set(selectedCols);
    const ds = new Set(selectedDocs);
    const wasOn = cols.has(col.id) || known.some((d) => ds.has(d.id));
    if (wasOn) {
      cols.delete(col.id);
      for (const d of known) ds.delete(d.id);
    } else {
      cols.add(col.id);
      for (const d of known) ds.delete(d.id);
    }
    set(cols, ds);
  };

  /** Single-document toggle. Unpicking one document out of a fully-selected
   *  collection expands that collection into its documents first. */
  const toggleDocument = (col: CollectionNode, docId: number) => {
    const known = docs[col.id] ?? [];
    const cols = new Set(selectedCols);
    const ds = new Set(selectedDocs);
    if (cols.has(col.id)) {
      cols.delete(col.id);
      for (const d of known) if (d.id !== docId) ds.add(d.id);
    } else if (ds.has(docId)) {
      ds.delete(docId);
    } else {
      ds.add(docId);
      if (known.length > 0 && known.every((d) => ds.has(d.id))) {
        for (const d of known) ds.delete(d.id);
        cols.add(col.id);
      }
    }
    set(cols, ds);
  };

  const visible = filter.trim()
    ? collections.filter((c) => {
        const q = filter.trim().toLowerCase();
        return (c.title || '').toLowerCase().includes(q) || c.slug.toLowerCase().includes(q);
      })
    : collections;

  const totalDocs = scopeDocumentCount(scope, collections);
  const orphanDocs = (scope.included_document_ids || []).length;

  const state = (col: CollectionNode): 'on' | 'partial' | 'off' => {
    if (selectedCols.has(col.id)) return 'on';
    const known = docs[col.id] ?? [];
    return known.some((d) => selectedDocs.has(d.id)) ? 'partial' : 'off';
  };

  return (
    <>
      <div className="popover-head">
        <span className="popover-title">Search scope</span>
        <div className="spacer" />
        <button type="button" className="icon-btn icon-btn-sm" onClick={onClose} aria-label="Close">
          <Icon name="close" size={15} />
        </button>
      </div>

      <div className="scope-modes">
        <button
          type="button"
          className="scope-mode"
          aria-pressed={!isCustom}
          onClick={() => onApply(FULL, isCustom ? 'Searching the entire archive' : undefined)}
        >
          <span className="scope-mode-title">
            <Icon name="library" size={15} />
            Entire archive
          </span>
          <span className="scope-mode-desc">
            {plural(collections.length, 'collection')}
          </span>
        </button>
        <button
          type="button"
          className="scope-mode"
          aria-pressed={isCustom}
          onClick={() => {
            if (!isCustom) {
              onApply({ mode: 'custom', included_collection_ids: [], included_document_ids: [] });
            }
          }}
        >
          <span className="scope-mode-title">
            <Icon name="filter" size={15} />
            Chosen sources
          </span>
          <span className="scope-mode-desc">
            {isCustom && totalDocs != null
              ? plural(totalDocs, 'document')
              : 'Pick collections or files'}
          </span>
        </button>
      </div>

      <div className="scope-search">
        <label className="field">
          <Icon name="search" size={15} />
          <input
            type="text"
            placeholder="Filter collections…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') e.preventDefault(); }}
            aria-label="Filter collections"
          />
        </label>
      </div>

      <div className="scope-summary">
        <span>
          {isCustom
            ? (selectedCols.size + orphanDocs === 0
                ? 'Nothing selected yet'
                : [
                    selectedCols.size > 0 ? plural(selectedCols.size, 'collection') : null,
                    orphanDocs > 0 ? plural(orphanDocs, 'single file') : null,
                  ].filter(Boolean).join(' · '))
            : 'Every collection is searched'}
        </span>
        <div className="spacer" />
        {isCustom && (
          <>
            <button
              type="button"
              className="btn-link"
              onClick={() => set(collections.map((c) => c.id), [])}
            >
              All
            </button>
            <button type="button" className="btn-link" onClick={() => set([], [])}>
              None
            </button>
          </>
        )}
      </div>

      <div className="popover-body">
        <div className="scope-tree">
          {collections.length === 0 && (
            <div className="loading"><span className="spinner" /> Loading collections…</div>
          )}
          {collections.length > 0 && visible.length === 0 && (
            <div className="empty-state">No collection matches “{filter}”.</div>
          )}
          {visible.map((col) => {
            const st = state(col);
            const isOpen = expanded.has(col.id);
            const colDocs = docs[col.id];
            return (
              <div key={col.id}>
                <div
                  className="scope-row"
                  // A finger lands anywhere on the row; only the bare row toggles here,
                  // the checkbox and labels toggle themselves and the expander expands.
                  onClick={(e) => { if (e.target === e.currentTarget) toggleCollection(col); }}
                >
                  <input
                    type="checkbox"
                    checked={st === 'on'}
                    ref={(el) => { if (el) el.indeterminate = st === 'partial'; }}
                    onChange={() => toggleCollection(col)}
                    id={`scope-col-${col.id}`}
                    aria-label={col.title || col.slug}
                  />
                  <button
                    type="button"
                    className="scope-expand"
                    aria-expanded={isOpen}
                    onClick={() => toggleExpand(col.id)}
                    aria-label={`${isOpen ? 'Hide' : 'Show'} files in ${col.title || col.slug}`}
                  >
                    <Icon name="chevron-right" size={13} />
                  </button>
                  <label className="scope-row-label" htmlFor={`scope-col-${col.id}`} title={col.title || col.slug}>
                    {col.title || col.slug || `Collection #${col.id}`}
                  </label>
                  <label className="scope-row-count" htmlFor={`scope-col-${col.id}`}>
                    {col.document_count}
                  </label>
                </div>

                {isOpen && (
                  <div className="scope-docs">
                    {colDocs == null && <div className="scope-doc-row">Loading…</div>}
                    {colDocs?.length === 0 && <div className="scope-doc-row">No files</div>}
                    {colDocs?.map((doc) => (
                      <div
                        className="scope-doc-row"
                        key={doc.id}
                        onClick={(e) => { if (e.target === e.currentTarget) toggleDocument(col, doc.id); }}
                      >
                        <input
                          type="checkbox"
                          id={`scope-doc-${doc.id}`}
                          checked={selectedCols.has(col.id) || selectedDocs.has(doc.id)}
                          onChange={() => toggleDocument(col, doc.id)}
                        />
                        <label className="scope-doc-label" htmlFor={`scope-doc-${doc.id}`}>
                          {doc.source_name || doc.source_ref || `Document #${doc.id}`}
                        </label>
                        <a
                          className="scope-doc-open"
                          href={api.getDocumentPdfUrl(doc.id)}
                          target="_blank"
                          rel="noreferrer"
                          title="Open the PDF in a new tab"
                          aria-label={`Open ${doc.source_name || doc.id} as PDF`}
                        >
                          <Icon name="external" size={13} />
                        </a>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {recents.length > 0 && (
        <div className="popover-foot">
          <span className="count">Recent</span>
          <div className="scope-recent">
            {recents.map((r, i) => (
              <button
                key={i}
                type="button"
                className="chip"
                onClick={() => onApply(r.scope, 'Scope restored')}
                title={r.label}
              >
                <span className="truncate" style={{ maxWidth: '11rem' }}>{r.label}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
