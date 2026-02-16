'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import type {
  V9ChatResponse, CollectionNode, DocumentNode, UserSelectedScope, RunScopeInfo,
} from '@/types/api';
import { api } from '@/lib/api';
import { scopeFingerprint } from '@/lib/scope';

// =============================================================================
// RightPane — scope-only (investigation moved to main chat)
// =============================================================================

interface RightPaneProps {
  v9Response: V9ChatResponse | null;
  sessionId?: number | null;
  activeScope?: UserSelectedScope | null;
  onApplyScope: (scope: UserSelectedScope) => void;
  onDraftDirtyChange: (dirty: boolean) => void;
  activeScopeRevision: number;
  collections: CollectionNode[];
}

export function RightPane({
  v9Response,
  sessionId,
  activeScope,
  onApplyScope,
  onDraftDirtyChange,
  activeScopeRevision,
  collections,
}: RightPaneProps) {
  const effectiveActive: UserSelectedScope = activeScope ?? { mode: 'full_archive' };

  // --- Draft scope state (survives tab switching because it lives here, not in ScopePanel) ---
  const [draftScope, setDraftScope] = useState<UserSelectedScope>({ ...effectiveActive });
  const [draftBaseRevision, setDraftBaseRevision] = useState(activeScopeRevision);

  // Initialize draft from active on session change ONLY
  useEffect(() => {
    setDraftScope({ ...effectiveActive });
    setDraftBaseRevision(activeScopeRevision);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  // Stale-draft detection
  const activeChanged = activeScopeRevision !== draftBaseRevision;
  const dirty = scopeFingerprint(draftScope) !== scopeFingerprint(effectiveActive);

  // Silent rebase when draft is clean and active scope changed externally
  useEffect(() => {
    if (activeChanged && !dirty) {
      setDraftScope({ ...effectiveActive });
      setDraftBaseRevision(activeScopeRevision);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeScopeRevision]);

  // Report dirty status to parent (callback must be useCallback-stable)
  useEffect(() => {
    onDraftDirtyChange(dirty);
  }, [dirty, onDraftDirtyChange]);

  // --- Handlers ---
  const handleDraftChange = useCallback((scope: UserSelectedScope) => {
    setDraftScope(scope);
  }, []);

  const handleApply = useCallback(() => {
    onApplyScope(draftScope);
    // activeScopeRevision will increment, silent rebase will sync draftBaseRevision
  }, [draftScope, onApplyScope]);

  const handleRevert = useCallback(() => {
    setDraftScope({ ...effectiveActive });
    setDraftBaseRevision(activeScopeRevision);
  }, [effectiveActive, activeScopeRevision]);

  const handleRebase = useCallback(() => {
    setDraftScope({ ...effectiveActive });
    setDraftBaseRevision(activeScopeRevision);
  }, [effectiveActive, activeScopeRevision]);

  const handleKeepEditing = useCallback(() => {
    setDraftBaseRevision(activeScopeRevision);
  }, [activeScopeRevision]);

  return (
    <>
      <div className="pane-header">Scope</div>
      <div className="pane-content" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: 0, alignItems: 'stretch' }}>
        {/* Rebase banner (active scope changed while draft is dirty) */}
            {activeChanged && dirty && (
              <div className="scope-rebase-banner">
                <div style={{ fontSize: '12px', fontWeight: 500, marginBottom: 6 }}>
                  Active scope changed while you were editing
                </div>
                <div style={{ display: 'flex', gap: 8 }}>
                  <button onClick={handleRebase} className="btn-primary" style={{ fontSize: '11px', padding: '3px 10px' }}>
                    Rebase draft
                  </button>
                  <button onClick={handleKeepEditing} className="btn-secondary" style={{ fontSize: '11px', padding: '3px 10px' }}>
                    Keep editing draft
                  </button>
                </div>
              </div>
            )}
            <ScopePanel
              draftScope={draftScope}
              activeScope={effectiveActive}
              collections={collections}
              onDraftChange={handleDraftChange}
              onApply={handleApply}
              onRevert={handleRevert}
              hasDraftChanges={dirty}
              lastRunScope={v9Response?.scope_override?.run_scope as RunScopeInfo | undefined}
              expansionInfo={v9Response?.expansion_info}
            />
      </div>
    </>
  );
}


// =============================================================================
// Scope Panel — Pure editor view (zero local scope state)
// =============================================================================

function ScopePanel({
  draftScope,
  activeScope,
  collections,
  onDraftChange,
  onApply,
  onRevert,
  hasDraftChanges,
  lastRunScope,
  expansionInfo,
}: {
  draftScope: UserSelectedScope;
  activeScope: UserSelectedScope;
  collections: CollectionNode[];
  onDraftChange: (scope: UserSelectedScope) => void;
  onApply: () => void;
  onRevert: () => void;
  hasDraftChanges: boolean;
  lastRunScope?: RunScopeInfo;
  expansionInfo?: { policy: string; collections: string[]; triggered: boolean; reason?: string };
}) {
  // --- Local UI state (view concerns only, not scope state) ---
  const [expandedCollections, setExpandedCollections] = useState<Set<number>>(new Set());
  const [searchFilter, setSearchFilter] = useState('');
  const [enrichedCollections, setEnrichedCollections] = useState<CollectionNode[]>(collections);
  const [showDiffDetail, setShowDiffDetail] = useState(false);

  // Sync enriched collections from prop (preserve already-loaded documents)
  useEffect(() => {
    setEnrichedCollections(prev => {
      if (prev.length === 0 && collections.length > 0) return collections;
      if (collections.length === 0) return prev;
      return collections.map(c => {
        const existing = prev.find(ec => ec.id === c.id);
        return existing?._docsLoaded ? { ...c, documents: existing.documents, _docsLoaded: true } : c;
      });
    });
  }, [collections]);

  // --- Derived state from draftScope ---
  const isCustom = draftScope.mode === 'custom';
  const selectedCollectionIds = useMemo(
    () => new Set(draftScope.included_collection_ids || []),
    [draftScope.included_collection_ids]
  );
  const selectedDocumentIds = useMemo(
    () => new Set(draftScope.included_document_ids || []),
    [draftScope.included_document_ids]
  );
  const selectedColCount = selectedCollectionIds.size;
  const selectedDocCount = selectedDocumentIds.size;

  // --- Changes computation (draft vs active) ---
  const changes = useMemo(() => {
    if (!hasDraftChanges) return null;
    if (activeScope.mode !== draftScope.mode) {
      return {
        modeChanged: true,
        fromMode: activeScope.mode,
        toMode: draftScope.mode,
        addedCols: [] as number[], removedCols: [] as number[],
        addedDocs: [] as number[], removedDocs: [] as number[],
      };
    }
    const activeColIds = new Set(activeScope.included_collection_ids || []);
    const activeDocIds = new Set(activeScope.included_document_ids || []);
    return {
      modeChanged: false,
      addedCols: [...selectedCollectionIds].filter(id => !activeColIds.has(id)),
      removedCols: [...activeColIds].filter(id => !selectedCollectionIds.has(id)),
      addedDocs: [...selectedDocumentIds].filter(id => !activeDocIds.has(id)),
      removedDocs: [...activeDocIds].filter(id => !selectedDocumentIds.has(id)),
    };
  }, [hasDraftChanges, activeScope, draftScope.mode, selectedCollectionIds, selectedDocumentIds]);

  // --- Handlers ---
  const handleModeChange = (mode: 'full_archive' | 'custom') => {
    if (mode === 'full_archive') {
      onDraftChange({ mode: 'full_archive' });
    } else {
      onDraftChange({
        mode: 'custom',
        included_collection_ids: Array.from(selectedCollectionIds),
        included_document_ids: Array.from(selectedDocumentIds),
      });
    }
  };

  const toggleCollection = (colId: number) => {
    const next = new Set(selectedCollectionIds);
    if (next.has(colId)) next.delete(colId);
    else next.add(colId);
    onDraftChange({
      mode: 'custom',
      included_collection_ids: Array.from(next),
      included_document_ids: Array.from(selectedDocumentIds),
    });
  };

  const toggleDocument = (docId: number) => {
    const next = new Set(selectedDocumentIds);
    if (next.has(docId)) next.delete(docId);
    else next.add(docId);
    onDraftChange({
      mode: 'custom',
      included_collection_ids: Array.from(selectedCollectionIds),
      included_document_ids: Array.from(next),
    });
  };

  const toggleExpand = async (colId: number) => {
    const next = new Set(expandedCollections);
    if (next.has(colId)) {
      next.delete(colId);
    } else {
      next.add(colId);
      const col = enrichedCollections.find(c => c.id === colId);
      if (col && !col._docsLoaded) {
        try {
          const docs = await api.getCollectionDocuments(colId);
          setEnrichedCollections(prev => prev.map(c =>
            c.id === colId ? { ...c, documents: docs, _docsLoaded: true } : c
          ));
        } catch (e) {
          console.error('Failed to load documents for collection', colId, e);
        }
      }
    }
    setExpandedCollections(next);
  };

  const selectAll = () => {
    onDraftChange({
      mode: 'custom',
      included_collection_ids: enrichedCollections.map(c => c.id),
      included_document_ids: Array.from(selectedDocumentIds),
    });
  };

  const selectNone = () => {
    onDraftChange({ mode: 'custom', included_collection_ids: [], included_document_ids: [] });
  };

  const loadRunScopeIntoSelection = () => {
    if (!lastRunScope) return;
    onDraftChange({
      mode: lastRunScope.mode || 'full_archive',
      included_collection_ids: lastRunScope.included_collection_ids || [],
      included_document_ids: lastRunScope.included_document_ids || [],
    });
  };

  const filteredCollections = searchFilter
    ? enrichedCollections.filter(c =>
        (c.title || c.slug).toLowerCase().includes(searchFilter.toLowerCase()) ||
        c.slug.toLowerCase().includes(searchFilter.toLowerCase())
      )
    : enrichedCollections;

  const resolveCollectionName = (id: number) => {
    const col = enrichedCollections.find(c => c.id === id);
    return col ? (col.title || col.slug) : `Collection #${id}`;
  };

  // Build changes summary text
  const buildChangesText = () => {
    if (!changes) return null;
    if (changes.modeChanged) {
      const from = changes.fromMode === 'full_archive' ? 'Full archive' : 'Custom';
      const to = changes.toMode === 'full_archive' ? 'Full archive' : 'Custom';
      return <span>Mode: {from} &rarr; {to}</span>;
    }
    const parts: React.ReactNode[] = [];
    if (changes.addedCols.length > 0) parts.push(<span key="ac" style={{ color: 'var(--color-success)' }}>+{changes.addedCols.length} col{changes.addedCols.length !== 1 ? 's' : ''}</span>);
    if (changes.removedCols.length > 0) parts.push(<span key="rc" style={{ color: 'var(--color-danger)' }}>-{changes.removedCols.length} col{changes.removedCols.length !== 1 ? 's' : ''}</span>);
    if (changes.addedDocs.length > 0) parts.push(<span key="ad" style={{ color: 'var(--color-success)' }}>+{changes.addedDocs.length} doc{changes.addedDocs.length !== 1 ? 's' : ''}</span>);
    if (changes.removedDocs.length > 0) parts.push(<span key="rd" style={{ color: 'var(--color-danger)' }}>-{changes.removedDocs.length} doc{changes.removedDocs.length !== 1 ? 's' : ''}</span>);
    return parts.length > 0
      ? parts.reduce<React.ReactNode[]>((acc, p, i) => (i > 0 ? [...acc, <span key={`sep${i}`}>, </span>, p] : [p]), [])
      : null;
  };

  return (
    <div style={{ fontSize: '13px', width: '100%', textAlign: 'left', display: 'flex', flexDirection: 'column', alignItems: 'stretch', flex: 1, minHeight: 0 }}>
      {/* Status banner */}
      <div className={hasDraftChanges ? 'scope-banner-dirty' : 'scope-banner-clean'}>
        {hasDraftChanges ? (
          <><span className="scope-draft-dot" /> Unapplied changes</>
        ) : (
          'No pending changes'
        )}
      </div>

      {/* Changes summary strip */}
      {hasDraftChanges && changes && (
        <div className="scope-changes-strip">
          <span style={{ flex: 1 }}>{buildChangesText()}</span>
          <button
            onClick={() => setShowDiffDetail(!showDiffDetail)}
            style={{ background: 'none', border: 'none', color: 'var(--color-primary)', fontSize: '11px', cursor: 'pointer', padding: 0 }}
          >
            {showDiffDetail ? 'Hide diff' : 'View diff'}
          </button>
        </div>
      )}

      {/* Expanded diff detail */}
      {showDiffDetail && changes && !changes.modeChanged && (
        <div className="scope-changes-detail">
          {changes.addedCols.map(id => (
            <div key={`+c${id}`} style={{ color: 'var(--color-success)', fontSize: '11px' }}>+ {resolveCollectionName(id)}</div>
          ))}
          {changes.removedCols.map(id => (
            <div key={`-c${id}`} style={{ color: 'var(--color-danger)', fontSize: '11px' }}>- {resolveCollectionName(id)}</div>
          ))}
          {changes.addedDocs.map(id => (
            <div key={`+d${id}`} style={{ color: 'var(--color-success)', fontSize: '11px' }}>+ Document #{id}</div>
          ))}
          {changes.removedDocs.map(id => (
            <div key={`-d${id}`} style={{ color: 'var(--color-danger)', fontSize: '11px' }}>- Document #{id}</div>
          ))}
        </div>
      )}

      {/* Scrollable content area */}
      <div style={{ flex: 1, overflowY: 'auto', overflowX: 'hidden', padding: '0 8px', textAlign: 'left', width: '100%' }}>
        {/* Used in last run */}
        {lastRunScope && (
          <div style={{
            padding: '6px 8px',
            background: 'var(--color-surface, #f5f5f5)',
            borderRadius: '4px',
            marginBottom: '8px',
            fontSize: '12px',
          }}>
            <div style={{ fontWeight: 600, marginBottom: 2 }}>Used in last run</div>
            <div>
              {lastRunScope.mode === 'full_archive' ? 'Full archive' : 'Custom scope'}
              {lastRunScope.source === 'query_override' && ' (overridden by query)'}
            </div>
            {expansionInfo && (
              <div style={{ marginTop: 2, color: 'var(--color-text-secondary, #666)' }}>
                Expansion: {expansionInfo.triggered ? 'Triggered' : 'Not triggered'}
                {expansionInfo.reason && ` - ${expansionInfo.reason.substring(0, 80)}`}
              </div>
            )}
            <button
              onClick={loadRunScopeIntoSelection}
              style={{
                marginTop: 4, fontSize: '11px', padding: '2px 6px',
                border: '1px solid var(--color-border, #ddd)', borderRadius: '3px',
                background: 'white', cursor: 'pointer',
              }}
            >
              Load into draft
            </button>
          </div>
        )}

        {/* Mode selector — inline row */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: '8px' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer', fontSize: '13px' }}>
            <input
              type="radio"
              name="scope-mode"
              checked={draftScope.mode === 'full_archive'}
              onChange={() => handleModeChange('full_archive')}
              style={{ margin: 0 }}
            />
            Full archive
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer', fontSize: '13px' }}>
            <input
              type="radio"
              name="scope-mode"
              checked={draftScope.mode === 'custom'}
              onChange={() => handleModeChange('custom')}
              style={{ margin: 0 }}
            />
            Custom
          </label>
        </div>

        {/* Search filter */}
        <input
          type="text"
          placeholder="Filter collections..."
          value={searchFilter}
          onChange={(e) => setSearchFilter(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') e.preventDefault(); }}
          disabled={!isCustom}
          style={{
            padding: '4px 6px',
            border: '1px solid var(--color-border, #ddd)',
            borderRadius: '3px',
            fontSize: '12px',
            marginBottom: '6px',
            opacity: isCustom ? 1 : 0.5,
            color: '#212529',
          }}
        />

        {/* Quick actions */}
        {isCustom && (
          <div style={{ display: 'flex', gap: 8, marginBottom: 6, fontSize: '11px' }}>
            <button onClick={selectAll} style={{ cursor: 'pointer', color: 'var(--color-accent, #1a73e8)', background: 'none', border: 'none', padding: 0, fontSize: 'inherit' }}>
              Select all
            </button>
            <button onClick={selectNone} style={{ cursor: 'pointer', color: 'var(--color-accent, #1a73e8)', background: 'none', border: 'none', padding: 0, fontSize: 'inherit' }}>
              Select none
            </button>
          </div>
        )}

        {/* Collections tree */}
        {enrichedCollections.length === 0 ? (
          <div style={{ color: '#666', padding: '12px 0', fontSize: '12px' }}>
            Loading collections&hellip;
            <button
              onClick={() => {
                console.log('[ScopePanel] Manual retry — fetching collections');
                api.getCollectionsTree().then(data => {
                  console.log('[ScopePanel] Retry got', data.length, 'collections');
                  setEnrichedCollections(data);
                }).catch(err => console.error('[ScopePanel] Retry failed:', err));
              }}
              style={{
                marginLeft: 8, fontSize: '11px', padding: '2px 8px',
                border: '1px solid #ccc', borderRadius: 3, background: '#f8f8f8',
                cursor: 'pointer', color: '#333',
              }}
            >
              Retry
            </button>
          </div>
        ) : (
          <div>
            {filteredCollections.map(col => (
              <div key={col.id} style={{ marginBottom: 1 }}>
                {/* Collection row */}
                <div style={{
                  display: 'flex', alignItems: 'center', gap: 4, padding: '3px 0',
                  opacity: isCustom ? 1 : 0.5,
                }}>
                  <input
                    type="checkbox"
                    checked={selectedCollectionIds.has(col.id)}
                    onChange={() => toggleCollection(col.id)}
                    disabled={!isCustom}
                    style={{ cursor: isCustom ? 'pointer' : 'default', margin: 0, flexShrink: 0 }}
                  />
                  <span
                    onClick={() => toggleExpand(col.id)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') toggleExpand(col.id); }}
                    style={{
                      cursor: 'pointer', fontSize: '10px', width: 16, flexShrink: 0,
                      color: '#333', display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                      userSelect: 'none',
                    }}
                    title="Expand / collapse documents"
                  >
                    {expandedCollections.has(col.id) ? '\u25BC' : '\u25B6'}
                  </span>
                  <span style={{
                    fontWeight: 500, flex: 1, overflow: 'hidden',
                    textOverflow: 'ellipsis', whiteSpace: 'nowrap', minWidth: 0,
                    color: '#212529',
                  }}>
                    {col.title || col.slug || `Collection #${col.id}`}
                  </span>
                  <span style={{ color: '#555', fontSize: '11px', flexShrink: 0, marginLeft: 4 }}>
                    ({col.document_count})
                  </span>
                </div>

                {/* Expanded documents */}
                {expandedCollections.has(col.id) && (
                  <div style={{ paddingLeft: 24 }}>
                    {col.documents && col.documents.length > 0 ? (
                      col.documents.map(doc => (
                        <div key={doc.id} style={{
                          display: 'flex', alignItems: 'center', gap: 4, padding: '2px 0',
                          fontSize: '12px', opacity: isCustom ? 1 : 0.5,
                        }}>
                          <input
                            type="checkbox"
                            checked={selectedDocumentIds.has(doc.id)}
                            onChange={() => toggleDocument(doc.id)}
                            disabled={!isCustom}
                            style={{ cursor: isCustom ? 'pointer' : 'default', margin: 0, flexShrink: 0 }}
                          />
                          <span style={{
                            flex: 1, overflow: 'hidden', textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap', minWidth: 0, color: '#212529',
                          }} title={doc.source_name || doc.source_ref || `Document #${doc.id}`}>
                            {doc.source_name || doc.source_ref || `Document #${doc.id}`}
                          </span>
                        </div>
                      ))
                    ) : (
                      <div style={{ color: '#888', fontSize: '11px', padding: '2px 0' }}>
                        {col._docsLoaded ? 'No documents' : 'Loading\u2026'}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Sticky footer — Apply / Revert + selection count */}
      <div className="scope-footer">
        <div style={{ fontSize: '12px', color: '#333', marginBottom: 6 }}>
          {draftScope.mode === 'full_archive' ? (
            'Selected: Full archive'
          ) : (
            <>
              Selected: {selectedColCount > 0 ? `${selectedColCount} collection${selectedColCount !== 1 ? 's' : ''}` : ''}
              {selectedColCount > 0 && selectedDocCount > 0 ? ', ' : ''}
              {selectedDocCount > 0 ? `${selectedDocCount} document${selectedDocCount !== 1 ? 's' : ''}` : ''}
              {selectedColCount === 0 && selectedDocCount === 0 && (
                <span style={{ color: 'var(--color-error, #d32f2f)' }}>
                  No documents selected
                </span>
              )}
            </>
          )}
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button
            onClick={onApply}
            disabled={!hasDraftChanges}
            className="btn-primary"
            style={{ fontSize: '12px', padding: '6px 12px' }}
          >
            Apply to Next Query
          </button>
          <button
            onClick={onRevert}
            disabled={!hasDraftChanges}
            className="btn-secondary"
            style={{ fontSize: '12px', padding: '6px 12px' }}
          >
            Revert
          </button>
        </div>
      </div>
    </div>
  );
}


