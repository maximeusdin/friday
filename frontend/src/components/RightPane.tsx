'use client';

import { useState, useEffect, useCallback } from 'react';
import type {
  V9ChatResponse, V9RunSummary, EvidenceRef, V9ProgressEvent, V9EvidenceBullet,
  CollectionNode, DocumentNode, UserSelectedScope, RunScopeInfo,
} from '@/types/api';
import { api } from '@/lib/api';

interface RightPaneProps {
  v9Response: V9ChatResponse | null;
  isProcessing: boolean;
  onEvidenceClick: (evidence: EvidenceRef | null) => void;
  progressSteps?: V9ProgressEvent[];
  evidenceBullets?: V9EvidenceBullet[];
  sessionId?: number | null;
  sessionScope?: UserSelectedScope | null;
  onScopeChange?: (scope: UserSelectedScope) => void;
}

type TabId = 'scope' | 'investigation';

export function RightPane({
  v9Response,
  isProcessing,
  onEvidenceClick,
  progressSteps = [],
  evidenceBullets = [],
  sessionId,
  sessionScope,
  onScopeChange,
}: RightPaneProps) {
  // Default tab: Scope for new/empty sessions, Investigation when messages exist
  const hasMessages = !!v9Response;
  const [activeTab, setActiveTab] = useState<TabId>(hasMessages ? 'investigation' : 'scope');

  // Switch to investigation when first response arrives
  useEffect(() => {
    if (v9Response && activeTab === 'scope') {
      setActiveTab('investigation');
    }
  }, [v9Response]); // eslint-disable-line react-hooks/exhaustive-deps

  // Reset to scope tab when session changes
  useEffect(() => {
    if (!v9Response) {
      setActiveTab('scope');
    }
  }, [sessionId]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <>
      {/* Tab Header */}
      <div className="pane-header" style={{ display: 'flex', gap: '0', padding: 0 }}>
        <button
          onClick={() => setActiveTab('scope')}
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
        </button>
        <button
          onClick={() => setActiveTab('investigation')}
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
          {isProcessing && <span style={{ marginLeft: 6, fontSize: 10 }}>&#x25CF;</span>}
        </button>
      </div>

      {/* Content */}
      <div className="pane-content">
        {activeTab === 'scope' ? (
          <ScopePanel
            sessionId={sessionId ?? null}
            scope={sessionScope ?? { mode: 'full_archive' }}
            onScopeChange={onScopeChange}
            lastRunScope={v9Response?.scope_override?.run_scope as RunScopeInfo | undefined}
            expansionInfo={v9Response?.expansion_info}
          />
        ) : (
          <InvestigationPanel
            v9={v9Response}
            isProcessing={isProcessing}
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
// Scope Panel
// =============================================================================

function ScopePanel({
  sessionId,
  scope,
  onScopeChange,
  lastRunScope,
  expansionInfo,
}: {
  sessionId: number | null;
  scope: UserSelectedScope;
  onScopeChange?: (scope: UserSelectedScope) => void;
  lastRunScope?: RunScopeInfo;
  expansionInfo?: { policy: string; collections: string[]; triggered: boolean; reason?: string };
}) {
  const [collections, setCollections] = useState<CollectionNode[]>([]);
  const [loading, setLoading] = useState(false);
  const [expandedCollections, setExpandedCollections] = useState<Set<number>>(new Set());
  const [searchFilter, setSearchFilter] = useState('');

  // Selected IDs (local state for immediate UI feedback)
  const [selectedCollectionIds, setSelectedCollectionIds] = useState<Set<number>>(
    new Set(scope.included_collection_ids || [])
  );
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<Set<number>>(
    new Set(scope.included_document_ids || [])
  );

  // Load collections tree
  useEffect(() => {
    setLoading(true);
    api.getCollectionsTree()
      .then(setCollections)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  // Sync local state when scope prop changes
  useEffect(() => {
    setSelectedCollectionIds(new Set(scope.included_collection_ids || []));
    setSelectedDocumentIds(new Set(scope.included_document_ids || []));
  }, [scope]);

  const isCustom = scope.mode === 'custom';

  // Persist scope changes
  const persistScope = useCallback((newScope: UserSelectedScope) => {
    onScopeChange?.(newScope);
    if (sessionId) {
      api.updateSessionScope(sessionId, newScope).catch(console.error);
    }
  }, [sessionId, onScopeChange]);

  // Mode switch
  const handleModeChange = (mode: 'full_archive' | 'custom') => {
    if (mode === 'full_archive') {
      persistScope({ mode: 'full_archive' });
    } else {
      persistScope({
        mode: 'custom',
        included_collection_ids: Array.from(selectedCollectionIds),
        included_document_ids: Array.from(selectedDocumentIds),
      });
    }
  };

  // Toggle collection
  const toggleCollection = (colId: number) => {
    const next = new Set(selectedCollectionIds);
    if (next.has(colId)) {
      next.delete(colId);
    } else {
      next.add(colId);
    }
    setSelectedCollectionIds(next);
    persistScope({
      mode: 'custom',
      included_collection_ids: Array.from(next),
      included_document_ids: Array.from(selectedDocumentIds),
    });
  };

  // Toggle document
  const toggleDocument = (docId: number) => {
    const next = new Set(selectedDocumentIds);
    if (next.has(docId)) {
      next.delete(docId);
    } else {
      next.add(docId);
    }
    setSelectedDocumentIds(next);
    persistScope({
      mode: 'custom',
      included_collection_ids: Array.from(selectedCollectionIds),
      included_document_ids: Array.from(next),
    });
  };

  // Expand/collapse collection to lazy-load documents
  const toggleExpand = async (colId: number) => {
    const next = new Set(expandedCollections);
    if (next.has(colId)) {
      next.delete(colId);
    } else {
      next.add(colId);
      // Lazy-load documents if not loaded yet
      const col = collections.find(c => c.id === colId);
      if (col && !col._docsLoaded) {
        try {
          const docs = await api.getCollectionDocuments(colId);
          setCollections(prev => prev.map(c =>
            c.id === colId ? { ...c, documents: docs, _docsLoaded: true } : c
          ));
        } catch (e) {
          console.error('Failed to load documents for collection', colId, e);
        }
      }
    }
    setExpandedCollections(next);
  };

  // Select all / none
  const selectAll = () => {
    const allIds = collections.map(c => c.id);
    setSelectedCollectionIds(new Set(allIds));
    persistScope({
      mode: 'custom',
      included_collection_ids: allIds,
      included_document_ids: Array.from(selectedDocumentIds),
    });
  };

  const selectNone = () => {
    setSelectedCollectionIds(new Set());
    setSelectedDocumentIds(new Set());
    // Don't persist empty custom (backend would reject)
  };

  // Count selected
  const selectedColCount = selectedCollectionIds.size;
  const selectedDocCount = selectedDocumentIds.size;

  // Filter collections by search
  const filteredCollections = searchFilter
    ? collections.filter(c =>
        c.title.toLowerCase().includes(searchFilter.toLowerCase()) ||
        c.slug.toLowerCase().includes(searchFilter.toLowerCase())
      )
    : collections;

  // Load last run scope into selection
  const loadRunScopeIntoSelection = () => {
    if (!lastRunScope) return;
    const newScope: UserSelectedScope = {
      mode: lastRunScope.mode || 'full_archive',
      included_collection_ids: lastRunScope.included_collection_ids || [],
      included_document_ids: lastRunScope.included_document_ids || [],
    };
    persistScope(newScope);
  };

  return (
    <div style={{ padding: '12px', fontSize: '13px' }}>
      {/* Used in last run */}
      {lastRunScope && (
        <div style={{
          padding: '8px 10px',
          background: 'var(--color-surface, #f5f5f5)',
          borderRadius: '6px',
          marginBottom: '12px',
          fontSize: '12px',
        }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>Used in last run</div>
          <div>
            {lastRunScope.mode === 'full_archive' ? 'Full archive' : 'Custom scope'}
            {lastRunScope.source === 'query_override' && ' (overridden by query)'}
          </div>
          {expansionInfo && (
            <div style={{ marginTop: 4, color: 'var(--color-text-secondary, #666)' }}>
              Expansion: {expansionInfo.triggered ? 'Triggered' : 'Not triggered'}
              {expansionInfo.reason && ` - ${expansionInfo.reason.substring(0, 80)}`}
            </div>
          )}
          <button
            onClick={loadRunScopeIntoSelection}
            style={{
              marginTop: 6,
              fontSize: '11px',
              padding: '3px 8px',
              border: '1px solid var(--color-border, #ddd)',
              borderRadius: '4px',
              background: 'white',
              cursor: 'pointer',
            }}
          >
            Load into selection
          </button>
        </div>
      )}

      {/* Mode selector */}
      <div style={{ marginBottom: '12px' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', marginBottom: 6 }}>
          <input
            type="radio"
            name="scope-mode"
            checked={scope.mode === 'full_archive'}
            onChange={() => handleModeChange('full_archive')}
          />
          Full archive
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
          <input
            type="radio"
            name="scope-mode"
            checked={scope.mode === 'custom'}
            onChange={() => handleModeChange('custom')}
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
        disabled={!isCustom}
        style={{
          width: '100%',
          padding: '6px 8px',
          border: '1px solid var(--color-border, #ddd)',
          borderRadius: '4px',
          fontSize: '12px',
          marginBottom: '8px',
          opacity: isCustom ? 1 : 0.5,
        }}
      />

      {/* Quick actions */}
      {isCustom && (
        <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
          <button onClick={selectAll} style={{ fontSize: '11px', cursor: 'pointer', color: 'var(--color-accent, #1a73e8)', background: 'none', border: 'none', padding: 0 }}>
            Select all
          </button>
          <button onClick={selectNone} style={{ fontSize: '11px', cursor: 'pointer', color: 'var(--color-accent, #1a73e8)', background: 'none', border: 'none', padding: 0 }}>
            Select none
          </button>
          <button onClick={() => handleModeChange('full_archive')} style={{ fontSize: '11px', cursor: 'pointer', color: 'var(--color-text-secondary, #666)', background: 'none', border: 'none', padding: 0, marginLeft: 'auto' }}>
            Reset
          </button>
        </div>
      )}

      {/* Collections tree */}
      {loading ? (
        <div style={{ color: 'var(--color-text-secondary, #666)', padding: '20px 0', textAlign: 'center' }}>
          Loading collections...
        </div>
      ) : (
        <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
          {filteredCollections.map(col => (
            <div key={col.id} style={{ marginBottom: 2 }}>
              <div style={{
                display: 'flex', alignItems: 'center', gap: 4, padding: '4px 0',
                opacity: isCustom ? 1 : 0.5,
              }}>
                <input
                  type="checkbox"
                  checked={selectedCollectionIds.has(col.id)}
                  onChange={() => toggleCollection(col.id)}
                  disabled={!isCustom}
                  style={{ cursor: isCustom ? 'pointer' : 'default' }}
                />
                <button
                  onClick={() => toggleExpand(col.id)}
                  style={{
                    background: 'none', border: 'none', cursor: 'pointer',
                    fontSize: '10px', width: 16, padding: 0,
                    color: 'var(--color-text-secondary, #666)',
                  }}
                >
                  {expandedCollections.has(col.id) ? '▼' : '▶'}
                </button>
                <span style={{ fontWeight: 500 }}>{col.title}</span>
                <span style={{ color: 'var(--color-text-secondary, #666)', fontSize: '11px', marginLeft: 'auto' }}>
                  {col.document_count} docs
                </span>
              </div>

              {/* Expanded documents */}
              {expandedCollections.has(col.id) && col.documents && (
                <div style={{ paddingLeft: 28 }}>
                  {col.documents.map(doc => (
                    <div key={doc.id} style={{
                      display: 'flex', alignItems: 'center', gap: 4, padding: '2px 0',
                      fontSize: '12px', opacity: isCustom ? 1 : 0.5,
                    }}>
                      <input
                        type="checkbox"
                        checked={selectedDocumentIds.has(doc.id)}
                        onChange={() => toggleDocument(doc.id)}
                        disabled={!isCustom}
                        style={{ cursor: isCustom ? 'pointer' : 'default' }}
                      />
                      <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {doc.source_name}
                      </span>
                    </div>
                  ))}
                  {!col._docsLoaded && (
                    <div style={{ color: 'var(--color-text-secondary, #666)', fontSize: '11px', padding: '4px 0' }}>
                      Loading...
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Status footer */}
      <div style={{
        marginTop: 12,
        padding: '8px 0',
        borderTop: '1px solid var(--color-border, #ddd)',
        fontSize: '12px',
        color: 'var(--color-text-secondary, #666)',
      }}>
        {scope.mode === 'full_archive' ? (
          'Selected: Full archive'
        ) : (
          <>
            Selected: {selectedColCount > 0 ? `${selectedColCount} collections` : ''}
            {selectedColCount > 0 && selectedDocCount > 0 ? ', ' : ''}
            {selectedDocCount > 0 ? `${selectedDocCount} documents` : ''}
            {selectedColCount === 0 && selectedDocCount === 0 && (
              <span style={{ color: 'var(--color-error, #d32f2f)' }}>
                No documents selected
              </span>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// =============================================================================
// Investigation Panel — V9 metadata
// =============================================================================

function InvestigationPanel({
  v9,
  isProcessing,
  progressSteps = [],
  evidenceBullets = [],
  onEvidenceClick,
}: {
  v9: V9ChatResponse | null;
  isProcessing: boolean;
  progressSteps?: V9ProgressEvent[];
  evidenceBullets?: V9EvidenceBullet[];
  onEvidenceClick?: (evidence: EvidenceRef | null) => void;
}) {
  if (!isProcessing && !v9) {
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

  if (isProcessing && !v9) {
    return (
      <div className="workflow-panel">
        <div className="workflow-header">
          <h3>Investigation</h3>
          <span className="workflow-status running">Running...</span>
        </div>

        {/* Live progress steps */}
        {progressSteps.length > 0 ? (
          <div className="workflow-actions">
            <div className="actions-header">Progress</div>
            {progressSteps.filter(s => s.step !== 'evidence_update').map((step, i, arr) => (
              <div
                key={i}
                className={`action-item ${i === arr.length - 1 ? 'action-running' : 'action-completed'}`}
              >
                <div className="action-main">
                  <span className="action-icon">
                    {i === arr.length - 1 ? '⟳' : '✓'}
                  </span>
                  <div className="action-content">
                    <div className="action-message">{step.message}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <>
            <div className="workflow-progress">
              <div className="progress-bar">
                <div className="progress-bar-fill progress-indeterminate" />
              </div>
            </div>
            <div className="empty-state">
              <p className="text-sm">Searching archives and analyzing evidence...</p>
            </div>
          </>
        )}

        {/* Live evidence bullets */}
        {evidenceBullets.length > 0 && (
          <div className="workflow-actions" style={{ marginTop: 'var(--spacing-sm)' }}>
            <div className="actions-header">Evidence Found ({evidenceBullets.length})</div>
            {evidenceBullets.map((bullet, i) => (
              <div key={i} className="evidence-bullet-live">
                <div className="bullet-text">{bullet.text}</div>
                <div className="bullet-meta">
                  {bullet.tags.map(t => (
                    <span key={t} className="bullet-tag">{t}</span>
                  ))}
                  {bullet.doc_ids.length > 0 && bullet.chunk_ids.length > 0 && onEvidenceClick && (
                    <button
                      className="bullet-source-link"
                      onClick={() => onEvidenceClick({
                        document_id: bullet.doc_ids[0],
                        pdf_page: 1,
                        chunk_id: bullet.chunk_ids[0],
                      })}
                    >
                      View source
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  if (!v9) return null;

  return (
    <div className="workflow-panel">
      <div className="workflow-header">
        <h3>Investigation</h3>
        <span className={`workflow-status ${v9.confidence === 'high' ? 'success' : 'partial'}`}>
          {v9.confidence} confidence
        </span>
      </div>

      {/* Summary stats */}
      <div className="workflow-summary">
        <div className="summary-item">
          <span className="summary-label">Intent</span>
          <span className="summary-value" style={{ textTransform: 'capitalize' }}>
            {v9.intent.replace(/_/g, ' ')}
          </span>
        </div>
        <div className="summary-item">
          <span className="summary-label">Citations</span>
          <span className="summary-value">{v9.cited_chunk_ids.length}</span>
        </div>
        {v9.elapsed_ms > 0 && (
          <div className="summary-item">
            <span className="summary-label">Time</span>
            <span className="summary-value">{(v9.elapsed_ms / 1000).toFixed(1)}s</span>
          </div>
        )}
        {v9.active_run_status && (
          <div className="summary-item">
            <span className="summary-label">Status</span>
            <span className="summary-value">{v9.active_run_status}</span>
          </div>
        )}
      </div>

      {/* Routing reasoning */}
      {v9.routing_reasoning && (
        <div className="workflow-actions">
          <div className="actions-header">Routing</div>
          <div className="action-item action-completed">
            <div className="action-main">
              <span className="action-icon">&#x1F9ED;</span>
              <div className="action-content">
                <div className="action-message">{v9.routing_reasoning}</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Remaining gaps */}
      {v9.remaining_gaps.length > 0 && (
        <div className="workflow-actions">
          <div className="actions-header">Remaining Gaps</div>
          {v9.remaining_gaps.map((gap, idx) => (
            <div key={idx} className="action-item action-completed">
              <div className="action-main">
                <span className="action-icon">&#x26A0;</span>
                <div className="action-content">
                  <div className="action-message">{gap}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Suggestion */}
      {v9.suggestion && (
        <div className="workflow-actions">
          <div className="actions-header">Suggested Next</div>
          <div className="action-item action-completed">
            <div className="action-main">
              <span className="action-icon">&#x1F4A1;</span>
              <div className="action-content">
                <div className="action-message">{v9.suggestion}</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Think deeper prompt */}
      {v9.can_think_deeper && (
        <div style={{
          padding: 'var(--spacing-md)',
          background: 'var(--color-highlight, #fff8e1)',
          borderRadius: 'var(--radius)',
          marginTop: 'var(--spacing-sm)',
          fontSize: '13px',
        }}>
          This run can be resumed with extended budget.
          Use the &ldquo;Think Deeper&rdquo; button below the answer.
        </div>
      )}

      {/* Run history */}
      {v9.run_history.length > 0 && (
        <div className="workflow-actions" style={{ marginTop: 'var(--spacing-md)' }}>
          <div className="actions-header">Run History ({v9.run_history.length})</div>
          {v9.run_history.map((run) => (
            <RunHistoryItem key={run.run_id} run={run} />
          ))}
        </div>
      )}
    </div>
  );
}

function RunHistoryItem({ run }: { run: V9RunSummary }) {
  return (
    <div className="action-item action-completed">
      <div className="action-main">
        <span className="action-icon">
          {run.status === 'completed' ? '✓' : run.status === 'paused' ? '⏸' : '⟳'}
        </span>
        <div className="action-content">
          <div className="action-label" style={{ fontSize: '12px', fontWeight: 500 }}>
            #{run.query_index} {run.label || run.query_text.substring(0, 50)}
          </div>
          <div className="action-message" style={{ fontSize: '11px' }}>
            {run.status}
            {run.evidence_summary && ` — ${run.evidence_summary}`}
          </div>
        </div>
      </div>
    </div>
  );
}
