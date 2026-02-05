'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { ResultsList } from './ResultsList';
import { EvidenceViewer } from './EvidenceViewer';
import type { V6Stats, WorkflowAction, ResultSetResponse, EvidenceRef } from '@/types/api';

type Tab = 'workflow' | 'results' | 'evidence';

interface RightPaneProps {
  v6Stats: V6Stats | null;
  isProcessing: boolean;
  resultSetId: number | null;
  activeEvidence: EvidenceRef | null;
  onResultSetIdUpdate: (resultSetId: number | null) => void;
  onEvidenceClick: (evidence: EvidenceRef | null) => void;
}

export function RightPane({
  v6Stats,
  isProcessing,
  resultSetId,
  activeEvidence,
  onResultSetIdUpdate,
  onEvidenceClick,
}: RightPaneProps) {
  const [activeTab, setActiveTab] = useState<Tab>('workflow');

  const { data: resultSet } = useQuery({
    queryKey: ['resultSet', resultSetId],
    queryFn: () => api.getResultSet(resultSetId as number),
    enabled: !!resultSetId,
  });

  // Auto-switch tabs based on state
  const effectiveTab = activeEvidence
    ? 'evidence'
    : activeTab;

  const renderTabContent = () => {
    switch (effectiveTab) {
      case 'workflow':
        return (
          <WorkflowPanel
            v6Stats={v6Stats}
            isProcessing={isProcessing}
          />
        );
      case 'results':
        return (
          <ResultsList
            resultSet={resultSet}
            onEvidenceClick={onEvidenceClick}
          />
        );
      case 'evidence':
        return (
          <EvidenceViewer
            evidence={activeEvidence}
            onClose={() => onEvidenceClick(null)}
          />
        );
      default:
        return null;
    }
  };

  return (
    <>
      {/* Tab headers */}
      <div
        className="pane-header flex gap-md"
        style={{ padding: '0' }}
      >
        <TabButton
          active={effectiveTab === 'workflow'}
          onClick={() => setActiveTab('workflow')}
        >
          Workflow
          {isProcessing && (
            <span className="badge badge-proposed" style={{ marginLeft: '6px' }}>
              running
            </span>
          )}
          {!isProcessing && v6Stats && (
            <span className="badge badge-executed" style={{ marginLeft: '6px' }}>
              done
            </span>
          )}
        </TabButton>
        <TabButton
          active={effectiveTab === 'results'}
          onClick={() => setActiveTab('results')}
          disabled={!resultSet}
        >
          Results
          {resultSet && (
            <span className="text-muted" style={{ marginLeft: '6px' }}>
              ({resultSet.summary.item_count})
            </span>
          )}
        </TabButton>
        <TabButton
          active={effectiveTab === 'evidence'}
          onClick={() => {}}
          disabled={!activeEvidence}
        >
          Evidence
        </TabButton>
      </div>

      {/* Tab content */}
      <div className="pane-content">
        {renderTabContent()}
      </div>
    </>
  );
}

function TabButton({
  children,
  active,
  onClick,
  disabled,
}: {
  children: React.ReactNode;
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: 'var(--spacing-md)',
        background: active ? 'var(--color-bg)' : 'transparent',
        borderBottom: active ? '2px solid var(--color-primary)' : '2px solid transparent',
        borderRadius: 0,
        opacity: disabled ? 0.5 : 1,
        fontWeight: active ? 600 : 400,
      }}
    >
      {children}
    </button>
  );
}

// =============================================================================
// Workflow Panel - Shows V6 workflow actions
// =============================================================================

function WorkflowPanel({
  v6Stats,
  isProcessing,
}: {
  v6Stats: V6Stats | null;
  isProcessing: boolean;
}) {
  // Show empty state only when not processing and no stats
  if (!isProcessing && !v6Stats) {
    return (
      <div className="workflow-panel">
        <div className="workflow-header">
          <h3>V7 Workflow</h3>
        </div>
        <div className="empty-state">
          <p>No workflow executed yet</p>
          <p className="text-sm">Ask a question to see the workflow steps</p>
        </div>
      </div>
    );
  }

  const actions = v6Stats?.actions || [];
  const statusLabel = isProcessing 
    ? 'Running...' 
    : (v6Stats?.responsiveness === 'responsive' ? '✓ Complete' : '~ Partial');
  const statusClass = isProcessing 
    ? 'running' 
    : (v6Stats?.responsiveness === 'responsive' ? 'success' : 'partial');

  return (
    <div className="workflow-panel">
      <div className="workflow-header">
        <h3>V7 Workflow</h3>
        <span className={`workflow-status ${statusClass}`}>
          {statusLabel}
        </span>
      </div>

      {/* Summary stats */}
      {v6Stats && (
        <div className="workflow-summary">
          <div className="summary-item">
            <span className="summary-label">Task Type</span>
            <span className="summary-value">{v6Stats.task_type}</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Rounds</span>
            <span className="summary-value">{v6Stats.rounds_executed}</span>
          </div>
          {!isProcessing && v6Stats.elapsed_ms > 0 && (
            <div className="summary-item">
              <span className="summary-label">Time</span>
              <span className="summary-value">{v6Stats.elapsed_ms.toFixed(0)}ms</span>
            </div>
          )}
          {/* V7 stats */}
          {v6Stats.claims_valid !== undefined && (
            <div className="summary-item">
              <span className="summary-label">Claims</span>
              <span className="summary-value">
                {v6Stats.claims_valid} valid
                {v6Stats.claims_dropped ? ` (${v6Stats.claims_dropped} dropped)` : ''}
              </span>
            </div>
          )}
          {v6Stats.citation_validation_passed !== undefined && !isProcessing && (
            <div className="summary-item">
              <span className="summary-label">Citations</span>
              <span className={`summary-value ${v6Stats.citation_validation_passed ? 'text-success' : 'text-warning'}`}>
                {v6Stats.citation_validation_passed ? '✓ Validated' : '~ Partial'}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Action list - shows streaming progress */}
      <div className="workflow-actions">
        <div className="actions-header">Execution Steps</div>
        {actions.length > 0 ? (
          actions.map((action, idx) => (
            <WorkflowActionItem
              key={action.step}
              action={action}
              isActive={isProcessing && idx === actions.length - 1}
            />
          ))
        ) : isProcessing ? (
          <WorkflowActionItem
            action={{
              step: 'starting',
              status: 'running',
              message: 'Starting V7 workflow...',
            }}
            isActive={true}
          />
        ) : null}
      </div>

      {/* Progress bar during processing */}
      {isProcessing && (
        <div className="workflow-progress">
          <div className="progress-bar">
            <div className="progress-bar-fill progress-indeterminate" />
          </div>
        </div>
      )}
    </div>
  );
}

function WorkflowActionItem({
  action,
  isActive,
}: {
  action: WorkflowAction;
  isActive: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  
  const getStepIcon = (step: string, status: string) => {
    if (status === 'running') return '⟳';
    if (status === 'error') return '✗';
    if (status === 'skipped') return '○';
    
    // Completed icons by step type
    if (step === 'query_parsing') return '📝';
    if (step === 'entity_linking') return '🔗';
    if (step.startsWith('retrieval_round')) return '🔍';
    if (step === 'synthesis') return '✨';
    if (step === 'responsiveness') return '✓';
    if (step === 'complete') return '✅';
    // V7 steps
    if (step === 'claim_enumeration') return '📋';
    if (step === 'stop_gate') return '🔒';
    if (step === 'expanded_summary') return '📄';
    if (step === 'bundle_building') return '📦';
    return '•';
  };

  const getStepLabel = (step: string) => {
    if (step === 'query_parsing') return 'Query Parsing';
    if (step === 'entity_linking') return 'Entity Linking';
    if (step.startsWith('retrieval_round')) {
      const round = step.replace('retrieval_round_', '');
      return `Retrieval Round ${round}`;
    }
    if (step === 'synthesis') return 'Synthesis';
    if (step === 'responsiveness') return 'Responsiveness Check';
    if (step === 'complete') return 'Complete';
    // V7 steps
    if (step === 'claim_enumeration') return 'Claim Extraction';
    if (step === 'stop_gate') return 'Citation Validation';
    if (step === 'expanded_summary') return 'Summary Generation';
    if (step === 'bundle_building') return 'Evidence Bundling';
    return step;
  };

  const hasDetails = action.details && Object.keys(action.details).length > 0;

  return (
    <div className={`action-item ${isActive ? 'action-active' : ''} action-${action.status}`}>
      <div 
        className="action-main"
        onClick={() => hasDetails && setExpanded(!expanded)}
        style={{ cursor: hasDetails ? 'pointer' : 'default' }}
      >
        <span className="action-icon">{getStepIcon(action.step, action.status)}</span>
        <div className="action-content">
          <div className="action-label">{getStepLabel(action.step)}</div>
          <div className="action-message">{action.message}</div>
        </div>
        {hasDetails && (
          <span className="action-expand">{expanded ? '▼' : '▶'}</span>
        )}
      </div>
      
      {expanded && action.details && (
        <div className="action-details">
          {Object.entries(action.details).map(([key, value]) => (
            <div key={key} className="detail-row">
              <span className="detail-key">{key.replace(/_/g, ' ')}:</span>
              <span className="detail-value">
                {Array.isArray(value) 
                  ? value.length > 0 
                    ? value.slice(0, 5).map(v => 
                        typeof v === 'object' ? JSON.stringify(v) : String(v)
                      ).join(', ') + (value.length > 5 ? '...' : '')
                    : '(none)'
                  : typeof value === 'object'
                    ? JSON.stringify(value)
                    : String(value)
                }
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
