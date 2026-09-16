'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '@/lib/api';
import { describeScope, scopeFingerprint, scopeDocumentCount } from '@/lib/scope';
import { plural } from '@/lib/format';
import { useLayout } from '@/lib/useLayout';
import { useDocumentName } from '@/lib/documentNames';
import type { CollectionNode, DocumentNode, UserSelectedScope } from '@/types/api';
import { Icon } from './ui/Icon';
import { Popover } from './ui/Popover';
import { toast } from './ui/Toast';

/**
 * ScopeControl: one control for "what does the archive mean right now".
 *
 * A chip showing the current scope sits next to the thing it filters (the
 * composer, the search bar); the picker behind it applies changes as they are
 * made, with Undo instead of Revert. A query reads the scope when you press
 * Ask, so there is nothing to commit.
 *
 * The picker is one component with two shapes. On desktop it is an anchored
 * panel with a tree. On a phone it is a sheet that reads like a settings
 * screen: a single "entire archive" switch, collections in sections with
 * whole-row taps, and files behind a drill-in instead of a nested tree.
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
    /* storage unavailable: recents are a convenience, never a requirement */
  }
}

// --- Sections -----------------------------------------------------------------
// Forty-odd collections in one flat list is a lot to scan on any screen and
// hopeless on a phone. Titles follow the archive's own naming closely enough
// to sort them into a few kinds; anything that matches nothing lands in the
// last section, so no collection is ever hidden by a bad guess.
const SECTIONS: { label: string; test: RegExp }[] = [
  { label: 'FBI files', test: /\bFBI\b/i },
  { label: 'Hearings and testimony', test: /HUAC|hearing|testimony|grand jury|committee|deposition|trial/i },
  { label: 'Decrypts and notebooks', test: /venona|vassiliev|notebook|decrypt|cable/i },
  { label: 'Security service files', test: /\bMI5\b|\bKV\b|canadian|british|security service|gouzenko|corby/i },
];
const OTHER_SECTION = 'Other collections';

function groupCollections(cols: CollectionNode[]): { label: string; items: CollectionNode[] }[] {
  const buckets = new Map<string, CollectionNode[]>();
  for (const c of cols) {
    const name = c.title || c.slug || '';
    const hit = SECTIONS.find((s) => s.test.test(name));
    const label = hit ? hit.label : OTHER_SECTION;
    const arr = buckets.get(label) ?? [];
    arr.push(c);
    buckets.set(label, arr);
  }
  const order = [...SECTIONS.map((s) => s.label), OTHER_SECTION];
  return order.filter((l) => buckets.has(l)).map((l) => ({ label: l, items: buckets.get(l)! }));
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
  const layout = useLayout();
  const active = scope ?? FULL;

  const isCustom = active.mode === 'custom';
  const colCount = active.included_collection_ids?.length ?? 0;
  const docCount = active.included_document_ids?.length ?? 0;
  const empty = isCustom && colCount === 0 && docCount === 0;

  // Apply immediately, and hand back a one-tap Undo instead of a Revert button.
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

  // Phones get a shorter label: the chip shares a row with the Ask button.
  const label = layout.isPhone && !isCustom ? 'All sources' : describeScope(active, collections);

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
        <span className="scope-chip-text">{label}</span>
        <Icon name="chevron-down" size={14} className="scope-chip-icon" />
      </button>

      <Popover
        anchorRef={anchor}
        open={open}
        onClose={() => setOpen(false)}
        placement={placement}
        align="start"
        label="Where to search"
        className="scope-picker"
      >
        <ScopePicker
          scope={active}
          collections={collections}
          onApply={apply}
          onClose={() => setOpen(false)}
          phone={layout.isPhone}
        />
      </Popover>
    </>
  );
}

// =============================================================================
// Picker body
// =============================================================================

type RowState = 'on' | 'partial' | 'off';

function ScopePicker({
  scope, collections, onApply, onClose, phone,
}: {
  scope: UserSelectedScope;
  collections: CollectionNode[];
  onApply: (next: UserSelectedScope, note?: string) => void;
  onClose: () => void;
  phone: boolean;
}) {
  const [filter, setFilter] = useState('');
  // Desktop: collections expand inline. Phone: one collection at a time, drilled into.
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  const [drill, setDrill] = useState<CollectionNode | null>(null);
  const [docs, setDocs] = useState<Record<number, DocumentNode[] | undefined>>({});
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

  /** Whole-collection toggle. Clearing also drops any individually-picked
   *  documents inside it, so the two selection levels can't contradict.
   *  From "entire archive", tapping a row means "just this one". */
  const toggleCollection = (col: CollectionNode) => {
    const known = docs[col.id] ?? [];
    const cols = new Set(isCustom ? selectedCols : []);
    const ds = new Set(isCustom ? selectedDocs : []);
    const wasOn = isCustom && (cols.has(col.id) || known.some((d) => ds.has(d.id)));
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
    const cols = new Set(isCustom ? selectedCols : []);
    const ds = new Set(isCustom ? selectedDocs : []);
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

  const openFiles = (col: CollectionNode) => {
    void loadDocs(col.id);
    if (phone) {
      setDrill(col);
    } else {
      setExpanded((prev) => {
        const next = new Set(prev);
        if (next.has(col.id)) next.delete(col.id);
        else next.add(col.id);
        return next;
      });
    }
  };

  const state = (col: CollectionNode): RowState => {
    if (!isCustom) return 'off';
    if (selectedCols.has(col.id)) return 'on';
    const known = docs[col.id] ?? [];
    return known.some((d) => selectedDocs.has(d.id)) ? 'partial' : 'off';
  };

  const q = filter.trim().toLowerCase();
  const visible = q
    ? collections.filter((c) => (c.title || '').toLowerCase().includes(q) || c.slug.toLowerCase().includes(q))
    : collections;
  const sections = useMemo(() => groupCollections(visible), [visible]);

  const totalDocs = scopeDocumentCount(scope, collections);
  const orphanDocs = (scope.included_document_ids || []).length;

  const summary = !isCustom
    ? `Searching the entire archive · ${plural(collections.length, 'collection')}`
    : selectedCols.size + orphanDocs === 0
      ? 'Nothing selected yet. Tap a collection to search just that one.'
      : selectedCols.size === 0
        ? `Searching ${plural(orphanDocs, 'file')}`
        : `Searching ${plural(selectedCols.size, 'collection')}${orphanDocs > 0 ? ` + ${plural(orphanDocs, 'file')}` : ''} · ${plural(totalDocs ?? 0, 'document')}`;

  // Files picked one at a time belong to collections whose file lists may not
  // be loaded, so their parent rows can't show it. List them by name instead,
  // so what is selected is always visible and removable.
  const orphanIds = scope.included_document_ids || [];
  const removeDoc = (docId: number) => {
    set(selectedCols, [...selectedDocs].filter((d) => d !== docId));
  };

  // --- Phone drill-in: one collection's files -------------------------------------
  if (phone && drill) {
    const col = drill;
    const list = docs[col.id];
    const colOn = selectedCols.has(col.id);
    return (
      <>
        <div className="popover-head">
          <button
            type="button"
            className="icon-btn"
            onClick={() => setDrill(null)}
            aria-label="Back to collections"
          >
            <Icon name="arrow-left" size={18} />
          </button>
          <span className="popover-title scope-drill-title" title={col.title || col.slug}>
            {col.title || col.slug}
          </span>
        </div>

        <div className="popover-body">
          <div className="scope-list">
            <CheckRow
              state={colOn ? 'on' : (list ?? []).some((d) => selectedDocs.has(d.id)) ? 'partial' : 'off'}
              label={
                col.document_count === 1
                  ? 'The only file in this collection'
                  : `All ${plural(col.document_count, 'file')} in this collection`
              }
              onToggle={() => toggleCollection(col)}
              strong
            />
            <div className="scope-section-title">Files</div>
            {list == null && <div className="loading"><span className="spinner" /> Loading files…</div>}
            {list?.length === 0 && <div className="empty-state">No files in this collection.</div>}
            {list?.map((doc) => (
              <CheckRow
                key={doc.id}
                state={colOn || selectedDocs.has(doc.id) ? 'on' : 'off'}
                label={doc.source_name || doc.source_ref || `Document #${doc.id}`}
                onToggle={() => toggleDocument(col, doc.id)}
                trailing={
                  <a
                    className="icon-btn icon-btn-sm"
                    href={api.getDocumentPdfUrl(doc.id)}
                    target="_blank"
                    rel="noreferrer"
                    onClick={(e) => e.stopPropagation()}
                    aria-label="Open the PDF"
                    title="Open the PDF"
                  >
                    <Icon name="external" size={14} />
                  </a>
                }
              />
            ))}
          </div>
        </div>

        <div className="popover-foot scope-foot">
          <span className="scope-summary-text">{summary}</span>
          <div className="spacer" />
          <button type="button" className="btn-primary btn-sm" onClick={onClose}>Done</button>
        </div>
      </>
    );
  }

  // --- Collections list (both shapes) --------------------------------------------
  return (
    <>
      <div className="popover-head">
        <span className="popover-title">Where to search</span>
        <div className="spacer" />
        {phone ? (
          <button type="button" className="btn-primary btn-sm" onClick={onClose}>Done</button>
        ) : (
          <button type="button" className="icon-btn icon-btn-sm" onClick={onClose} aria-label="Close">
            <Icon name="close" size={15} />
          </button>
        )}
      </div>

      {phone ? (
        // One switch. Off means "I'll choose", and the list below wakes up.
        <label className="scope-switch-row">
          <span className="scope-switch-text">
            <span className="scope-switch-title">Entire archive</span>
            <span className="scope-switch-desc">
              {isCustom ? 'Off: only the collections you tick below' : `All ${plural(collections.length, 'collection')}`}
            </span>
          </span>
          <button
            type="button"
            role="switch"
            aria-checked={!isCustom}
            aria-label="Search the entire archive"
            className="switch-track switch-track-lg"
            onClick={() => (isCustom
              ? onApply(FULL, 'Searching the entire archive')
              : onApply({ mode: 'custom', included_collection_ids: [], included_document_ids: [] }))}
          >
            <span className="switch-knob" />
          </button>
        </label>
      ) : (
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
            <span className="scope-mode-desc">{plural(collections.length, 'collection')}</span>
          </button>
          <button
            type="button"
            className="scope-mode"
            aria-pressed={isCustom}
            onClick={() => {
              if (!isCustom) onApply({ mode: 'custom', included_collection_ids: [], included_document_ids: [] });
            }}
          >
            <span className="scope-mode-title">
              <Icon name="filter" size={15} />
              Chosen sources
            </span>
            <span className="scope-mode-desc">
              {isCustom && totalDocs != null ? plural(totalDocs, 'document') : 'Pick collections or files'}
            </span>
          </button>
        </div>
      )}

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

      {recents.length > 0 && (
        <div className="scope-recent-row">
          <span className="count">Recent</span>
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
      )}

      <div className="popover-body">
        <div className={`scope-list${!isCustom && phone ? ' is-quiet' : ''}`}>
          {collections.length === 0 && (
            <div className="loading"><span className="spinner" /> Loading collections…</div>
          )}
          {collections.length > 0 && visible.length === 0 && (
            <div className="empty-state">No collection matches “{filter}”.</div>
          )}
          {isCustom && orphanIds.length > 0 && !q && (
            <div className="scope-section">
              <div className="scope-section-title">
                Selected files
                <span className="count">{orphanIds.length}</span>
              </div>
              {orphanIds.map((id) => (
                <SelectedFileRow key={id} docId={id} onRemove={() => removeDoc(id)} />
              ))}
            </div>
          )}
          {sections.map((section) => (
            <div key={section.label} className="scope-section">
              <div className="scope-section-title">
                {section.label}
                <span className="count">{section.items.length}</span>
              </div>
              {section.items.map((col) => {
                const st = state(col);
                const isOpen = !phone && expanded.has(col.id);
                const colDocs = docs[col.id];
                return (
                  <div key={col.id}>
                    <CheckRow
                      state={st}
                      label={col.title || col.slug || `Collection #${col.id}`}
                      count={col.document_count}
                      onToggle={() => toggleCollection(col)}
                      trailing={
                        <button
                          type="button"
                          className={`icon-btn icon-btn-sm scope-drill${isOpen ? ' is-open' : ''}`}
                          onClick={(e) => { e.stopPropagation(); openFiles(col); }}
                          aria-expanded={phone ? undefined : isOpen}
                          aria-label={phone ? `Choose files in ${col.title || col.slug}` : `${isOpen ? 'Hide' : 'Show'} files in ${col.title || col.slug}`}
                          title={phone ? 'Choose individual files' : 'Show files'}
                        >
                          <Icon name="chevron-right" size={15} />
                        </button>
                      }
                    />

                    {isOpen && (
                      <div className="scope-docs">
                        {colDocs == null && <div className="scope-doc-row">Loading…</div>}
                        {colDocs?.length === 0 && <div className="scope-doc-row">No files</div>}
                        {colDocs?.map((doc) => (
                          <CheckRow
                            key={doc.id}
                            small
                            state={selectedCols.has(col.id) || selectedDocs.has(doc.id) ? 'on' : 'off'}
                            label={doc.source_name || doc.source_ref || `Document #${doc.id}`}
                            onToggle={() => toggleDocument(col, doc.id)}
                            trailing={
                              <a
                                className="scope-doc-open"
                                href={api.getDocumentPdfUrl(doc.id)}
                                target="_blank"
                                rel="noreferrer"
                                onClick={(e) => e.stopPropagation()}
                                title="Open the PDF in a new tab"
                                aria-label={`Open ${doc.source_name || doc.id} as PDF`}
                              >
                                <Icon name="external" size={13} />
                              </a>
                            }
                          />
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      <div className="popover-foot scope-foot">
        <span className="scope-summary-text">{summary}</span>
        <div className="spacer" />
        {isCustom && (
          <>
            <button type="button" className="btn-link" onClick={() => set(collections.map((c) => c.id), [])}>
              All
            </button>
            <button type="button" className="btn-link" onClick={() => set([], [])}>
              None
            </button>
          </>
        )}
      </div>
    </>
  );
}

/** One hand-picked file, named, with a way to drop it. */
function SelectedFileRow({ docId, onRemove }: { docId: number; onRemove: () => void }) {
  const name = useDocumentName(docId);
  return (
    <div className="scope-row scope-row-small">
      <span className="scope-check" data-state="on" aria-hidden="true">
        <Icon name="check" size={13} strokeWidth={2.2} />
      </span>
      <span className="scope-row-label" title={name}>{name ?? `Document #${docId}`}</span>
      <button
        type="button"
        className="icon-btn icon-btn-sm"
        onClick={onRemove}
        aria-label={`Remove ${name ?? 'this file'}`}
        title="Remove from scope"
      >
        <Icon name="close" size={14} />
      </button>
    </div>
  );
}

/** A row that toggles as a whole. The check is drawn, not a native input, so
 *  it can show the "some files" state and size itself for a finger. */
function CheckRow({
  state, label, count, onToggle, trailing, small, strong,
}: {
  state: RowState;
  label: string;
  count?: number;
  onToggle: () => void;
  trailing?: React.ReactNode;
  small?: boolean;
  strong?: boolean;
}) {
  return (
    <div
      className={`scope-row${small ? ' scope-row-small' : ''}${strong ? ' scope-row-strong' : ''}`}
      role="checkbox"
      aria-checked={state === 'partial' ? 'mixed' : state === 'on'}
      tabIndex={0}
      onClick={onToggle}
      onKeyDown={(e) => {
        if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); onToggle(); }
      }}
    >
      <span className="scope-check" data-state={state} aria-hidden="true">
        {state === 'on' && <Icon name="check" size={13} strokeWidth={2.2} />}
        {state === 'partial' && <Icon name="minus" size={13} strokeWidth={2.2} />}
      </span>
      <span className="scope-row-label" title={label}>{label}</span>
      {count != null && <span className="scope-row-count">{count}</span>}
      {trailing}
    </div>
  );
}
