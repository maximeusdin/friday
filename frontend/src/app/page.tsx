'use client';

import { useState } from 'react';
import { SessionList } from '@/components/SessionList';
import { Conversation } from '@/components/Conversation';
import { RightPane } from '@/components/RightPane';
import type { Session, EvidenceRef } from '@/types/api';
import { api } from '@/lib/api';

export default function Home() {
  const [activeSession, setActiveSession] = useState<Session | null>(null);
  const [activePlanId, setActivePlanId] = useState<number | null>(null);
  const [activeResultSetId, setActiveResultSetId] = useState<number | null>(null);
  const [activeEvidence, setActiveEvidence] = useState<EvidenceRef | null>(null);

  const handleSessionSelect = (session: Session) => {
    setActiveSession(session);
    setActivePlanId(null);
    setActiveResultSetId(null);
    setActiveEvidence(null);

    // Rehydrate right pane with latest known plan/result for this session
    api
      .getSessionState(session.id)
      .then((state) => {
        setActivePlanId(state.latest_plan_id ?? null);
        setActiveResultSetId(state.latest_result_set_id ?? null);
      })
      .catch(() => {
        // Best-effort: keep empty state if state endpoint fails
      });
  };

  const handlePlanIdUpdate = (planId: number) => {
    setActivePlanId(planId);
  };

  const handleEvidenceClick = (evidence: EvidenceRef) => {
    setActiveEvidence(evidence);
  };

  return (
    <div className="app-container">
      {/* Left Pane: Sessions */}
      <div className="pane">
        <div className="pane-header">Sessions</div>
        <div className="pane-content">
          <SessionList
            activeSessionId={activeSession?.id}
            onSessionSelect={handleSessionSelect}
          />
        </div>
      </div>

      {/* Center Pane: Conversation */}
      <div className="pane">
        <div className="pane-header">
          {activeSession ? activeSession.label : 'Select a session'}
        </div>
        <Conversation
          session={activeSession}
          onPlanIdUpdate={handlePlanIdUpdate}
        />
      </div>

      {/* Right Pane: Plan + Results + Evidence */}
      <div className="pane" style={{ borderRight: 'none' }}>
        <RightPane
          planId={activePlanId}
          resultSetId={activeResultSetId}
          activeEvidence={activeEvidence}
          onPlanIdUpdate={setActivePlanId}
          onResultSetIdUpdate={setActiveResultSetId}
          onEvidenceClick={handleEvidenceClick}
        />
      </div>
    </div>
  );
}
