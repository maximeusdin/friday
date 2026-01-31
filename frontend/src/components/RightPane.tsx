'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { PlanCard } from './PlanCard';
import { ResultsList } from './ResultsList';
import { EvidenceViewer } from './EvidenceViewer';
import type { Plan, ResultSetResponse, EvidenceRef } from '@/types/api';

type Tab = 'plan' | 'results' | 'evidence';

interface RightPaneProps {
  planId: number | null;
  resultSetId: number | null;
  activeEvidence: EvidenceRef | null;
  onPlanIdUpdate: (planId: number | null) => void;
  onResultSetIdUpdate: (resultSetId: number | null) => void;
  onEvidenceClick: (evidence: EvidenceRef | null) => void;
}

export function RightPane({
  planId,
  resultSetId,
  activeEvidence,
  onPlanIdUpdate,
  onResultSetIdUpdate,
  onEvidenceClick,
}: RightPaneProps) {
  const [activeTab, setActiveTab] = useState<Tab>('plan');

  const { data: plan } = useQuery({
    queryKey: ['plan', planId],
    queryFn: () => api.getPlan(planId as number),
    enabled: !!planId,
  });

  const { data: resultSet } = useQuery({
    queryKey: ['resultSet', resultSetId],
    queryFn: () => api.getResultSet(resultSetId as number),
    enabled: !!resultSetId,
  });

  // Auto-switch tabs based on state
  const effectiveTab = activeEvidence
    ? 'evidence'
    : resultSet
    ? 'results'
    : activeTab;

  const renderTabContent = () => {
    switch (effectiveTab) {
      case 'plan':
        return (
          <PlanCard
            plan={plan}
            onPlanIdUpdate={onPlanIdUpdate}
            onResultSetIdUpdate={onResultSetIdUpdate}
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
          active={effectiveTab === 'plan'}
          onClick={() => setActiveTab('plan')}
          disabled={!plan}
        >
          Plan
          {plan && (
            <span className={`badge badge-${plan.status}`} style={{ marginLeft: '6px' }}>
              {plan.status}
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
