'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import type {
  V9ChatResponse, EvidenceRef, V9ProgressEvent, V9EvidenceBullet,
  CollectionNode, DocumentNode, UserSelectedScope, RunScopeInfo,
} from '@/types/api';
import { api } from '@/lib/api';
import { scopeFingerprint } from '@/lib/scope';

// =============================================================================
// RightPane — controlled tabs, draft scope, rebase logic
// =============================================================================

interface RightPaneProps {
  v9Response: V9ChatResponse | null;
  isProcessing: boolean;
  processingSessionId?: number | null;
  onEvidenceClick: (evidence: EvidenceRef | null) => void;
  progressSteps?: V9ProgressEvent[];
  evidenceBullets?: V9EvidenceBullet[];
  sessionId?: number | null;
  activeScope?: UserSelectedScope | null;
  activeTab: 'scope' | 'investigation';
  onTabChange: (tab: 'scope' | 'investigation') => void;
  onApplyScope: (scope: UserSelectedScope) => void;
  onDraftDirtyChange: (dirty: boolean) => void;
  activeScopeRevision: number;
  collections: CollectionNode[];
}

export function RightPane({
  v9Response,
  isProcessing,
  processingSessionId = null,
  onEvidenceClick,
  progressSteps = [],
  evidenceBullets = [],
  sessionId,
  activeScope,
  activeTab,
  onTabChange,
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
      {/* Tab Header */}
      <div className="pane-header" style={{ display: 'flex', gap: '0', padding: 0 }}>
        <button
          onClick={() => onTabChange('scope')}
          style={{
            flex: 1,
            padding: '10px 12px',
            border: 'none',
            borderBottom: activeTab === 'scope' ? '2px solid var(--color-accent, #1a73e8)' : '2px solid transparent',
            background: activeTab === 'scope' ? 'var(--color-bg, #fff)' : 'transparent',
            fontWeight: activeTab === 'scope' ? 600 : 400,
            fontSize: '13px',
            cursor: 'pointer',
            color: activeTab === 'scope' ? 'var(--color-text, #333)' : 'var(--color-text-secondary, #666)',
          }}
        >
          Scope
          {dirty && <span style={{ marginLeft: 6, color: '#e67e22', fontSize: 10 }}>&#x25CF;</span>}
        </button>
        <button
          onClick={() => onTabChange('investigation')}
          style={{
            flex: 1,
            padding: '10px 12px',
            border: 'none',
            borderBottom: activeTab === 'investigation' ? '2px solid var(--color-accent, #1a73e8)' : '2px solid transparent',
            background: activeTab === 'investigation' ? 'var(--color-bg, #fff)' : 'transparent',
            fontWeight: activeTab === 'investigation' ? 600 : 400,
            fontSize: '13px',
            cursor: 'pointer',
            color: activeTab === 'investigation' ? 'var(--color-text, #333)' : 'var(--color-text-secondary, #666)',
          }}
        >
          Investigation
          {isProcessing && sessionId === processingSessionId && <span style={{ marginLeft: 6, fontSize: 10 }}>&#x25CF;</span>}
        </button>
      </div>

      {/* Content — flex column when scope tab so footer sticks */}
      <div
        className="pane-content"
        style={activeTab === 'scope' ? { display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: 0, alignItems: 'stretch' } : undefined}
      >
        {activeTab === 'scope' ? (
          <>
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
          </>
        ) : (
          <InvestigationPanel
            v9={v9Response}
            isProcessing={isProcessing}
            isActiveSessionProcessing={sessionId === processingSessionId}
            progressSteps={progressSteps}
            evidenceBullets={evidenceBullets}
            onEvidenceClick={onEvidenceClick}
          />
        )}
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


// =============================================================================
// Investigation Panel — minimal progress + evidence-focused
// =============================================================================

const DEFAULT_MAX_TOOL_CALLS = 12;

/** Strip chunk_id=..., chunk_ids: [], etc. from bullet text (LLM sometimes echoes prompt format). */
function sanitizeBulletText(text: string): string {
  return text
    .replace(/\s*[\[\(]chunk_id=\d+[\)\]]\s*/gi, ' ')
    .replace(/\s*[\[\(]chunk_ids:\s*\[\d*(?:,\s*\d*)*\]\s*[\)\]]\s*/gi, ' ')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

const PROGRESS_PHRASES = [
  'Searching archives...',
  'Reviewing documents...',
  'Cross-referencing sources...',
  'Extracting key findings...',
  'Verifying evidence...',
  'Building answer...',
  'Following leads...',
  'Checking citations...',
];

function toUserFriendlyProgress(step: V9ProgressEvent, stepIndex?: number): string {
  switch (step.step) {
    case 'tool_call': {
      // Prefer backend message when it contains context (e.g. "Searching for: X", "Loading N chunks...")
      const msg = (step.message || '').trim();
      if (msg && (msg.startsWith('Searching for:') || msg.startsWith('Loading ') || msg.includes('chunks'))) {
        return msg;
      }
      const tool = (step.details?.tool as string) || '';
      const idx = stepIndex ?? 0;
      if (tool.startsWith('search_chunks') || tool === 'search_chunks') {
        return PROGRESS_PHRASES[idx % PROGRESS_PHRASES.length];
      }
      if (tool.startsWith('fetch_chunks') || tool === 'fetch_chunks') return 'Reading documents...';
      if (tool.startsWith('expand_entities') || tool === 'expand_entities') return 'Resolving identities...';
      if (tool.startsWith('alias_index') || tool.includes('alias')) return 'Looking up references...';
      return PROGRESS_PHRASES[idx % PROGRESS_PHRASES.length];
    }
    case 'turn_start':
      return 'Investigating...';
    case 'investigation':
      return 'Searching and analyzing...';
    case 'entity_resolution':
      return 'Resolving entities...';
    case 'synthesis':
      return 'Synthesizing answer...';
    case 'evidence_update':
      return 'Found evidence...';
    case 'routing':
      return 'Understanding your question...';
    case 'investigation_start':
    case 'retrieval_prepare':
      return 'Starting investigation...';
    case 'context_build':
      return 'Building context...';
    case 'follow_up_start':
    case 'follow_up':
      return 'Searching evidence...';
    case 'think_deeper_start':
      return 'Resuming Think Deeper...';
    default:
      if (step.message?.toLowerCase().includes('search')) return 'Searching archives...';
      if (step.message?.toLowerCase().includes('round')) return 'Cross-referencing sources...';
      if (step.message?.toLowerCase().includes('fetch')) return 'Reading documents...';
      if (step.message?.toLowerCase().includes('synthes')) return 'Synthesizing answer...';
      return PROGRESS_PHRASES[(stepIndex ?? 0) % PROGRESS_PHRASES.length] || step.message || 'Investigating...';
  }
}

function InvestigationPanel({
  v9,
  isProcessing,
  isActiveSessionProcessing,
  progressSteps = [],
  evidenceBullets = [],
  onEvidenceClick,
}: {
  v9: V9ChatResponse | null;
  isProcessing: boolean;
  isActiveSessionProcessing: boolean;
  progressSteps?: V9ProgressEvent[];
  evidenceBullets?: V9EvidenceBullet[];
  onEvidenceClick?: (evidence: EvidenceRef | null) => void;
}) {
  const hasData = v9 || evidenceBullets.length > 0;
  const showContent = hasData || isActiveSessionProcessing;
  if (!showContent) {
    return (
      <div className="workflow-panel">
        <div className="workflow-header">
          <h3>Investigation</h3>
        </div>
        <div className="empty-state">
          <p>No investigation yet</p>
          <p className="text-sm">Ask a question to start</p>
        </div>
      </div>
    );
  }

  // Tool-count-based progress: count tool_call steps, estimate % complete
  const progressStepsFiltered = progressSteps.filter(s => s.step !== 'evidence_update');
  const toolCallCount = progressStepsFiltered.filter(s => s.step === 'tool_call').length;
  const toolCallNumber = progressStepsFiltered
    .filter(s => s.details?.tool_call_number != null)
    .map(s => s.details?.tool_call_number as number)
    .pop();
  const effectiveToolCount = toolCallNumber ?? toolCallCount;
  const progressPercent = !isActiveSessionProcessing && v9
    ? 100
    : Math.min(95, Math.round((effectiveToolCount / DEFAULT_MAX_TOOL_CALLS) * 100));
  const showIndeterminate = isActiveSessionProcessing && effectiveToolCount === 0;

  const renderMinimalProgress = () => (
    <div className="workflow-progress-minimal">
      <div className="progress-bar">
        <div
          className={`progress-bar-fill ${showIndeterminate ? 'progress-indeterminate' : ''}`}
          style={!showIndeterminate ? { width: `${progressPercent}%` } : undefined}
        />
      </div>
      {isActiveSessionProcessing && (
        <span className="progress-spinner" aria-hidden>⟳</span>
      )}
    </div>
  );

  const renderProgressBullets = () => {
    const steps = progressStepsFiltered.filter(s => s.step !== 'evidence_update');
    if (steps.length === 0) return null;
    const seen = new Set<string>();
    const unique: { step: V9ProgressEvent; idx: number }[] = [];
    for (let i = 0; i < steps.length; i++) {
      const s = steps[i];
      const label = toUserFriendlyProgress(s, i);
      if (!seen.has(label)) {
        seen.add(label);
        unique.push({ step: s, idx: i });
      }
    }
    return (
      <div className="workflow-progress-bullets">
        <div className="actions-header">Progress</div>
        <ul className="progress-bullets-list">
          {unique.map(({ step, idx }, i) => (
            <li key={i} className="progress-bullet">
              {toUserFriendlyProgress(step, idx)}
            </li>
          ))}
        </ul>
      </div>
    );
  };

  const renderEvidenceBullets = () => (
    <div className="workflow-bullets-section workflow-evidence-primary">
      <div className="actions-header">Evidence ({evidenceBullets.length})</div>
      {evidenceBullets.length > 0 ? (
        <div className="evidence-bullets-list">
          {evidenceBullets.map((bullet, i) => (
            <div key={i} className="evidence-bullet-live">
              <div className="bullet-text">{sanitizeBulletText(bullet.text)}</div>
              <div className="bullet-meta">
                {bullet.doc_ids?.length > 0 && bullet.chunk_ids?.length > 0 && onEvidenceClick && (
                  <button
                    className="bullet-source-link"
                    onClick={() => onEvidenceClick({
                      document_id: bullet.doc_ids[0],
                      pdf_page: bullet.pages?.[0] ?? 1,
                      chunk_id: bullet.chunk_ids[0],
                    })}
                  >
                    {bullet.source_names?.[0] || (bullet.pages?.[0] ? `p.${bullet.pages[0]}` : 'View document')}
                    {bullet.pages?.[0] && bullet.source_names?.[0] && ` (p.${bullet.pages[0]})`}
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="evidence-placeholder">
          {isActiveSessionProcessing ? 'Gathering evidence...' : 'No evidence yet'}
        </div>
      )}
    </div>
  );

  // Processing: Evidence first (key), then progress bar, then progress bullets
  if (isActiveSessionProcessing && !v9) {
    return (
      <div className="workflow-panel workflow-panel-evidence">
        <div className="workflow-header">
          <h3>Investigation</h3>
        </div>
        {renderEvidenceBullets()}
        {renderMinimalProgress()}
        {renderProgressBullets()}
      </div>
    );
  }

  // Completed: Evidence first (key), then progress bar, then progress bullets, then summary
  return (
    <div className="workflow-panel workflow-panel-evidence">
      <div className="workflow-header">
        <h3>Investigation</h3>
      </div>
      {renderEvidenceBullets()}
      {renderMinimalProgress()}
      {renderProgressBullets()}
      {v9 && (
        <div className="workflow-summary-compact">
          <span>{v9.intent.replace(/_/g, ' ')}</span>
          {v9.cited_chunk_ids.length > 0 && (
            <span> · {v9.cited_chunk_ids.length} citations</span>
          )}
          {v9.elapsed_ms > 0 && (
            <span> · {(v9.elapsed_ms / 1000).toFixed(1)}s</span>
          )}
        </div>
      )}
    </div>
  );
}
