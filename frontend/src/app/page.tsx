'use client';

import { useState } from 'react';
import { SessionList } from '@/components/SessionList';
import { Conversation } from '@/components/Conversation';
import { RightPane } from '@/components/RightPane';
import type { Session, EvidenceRef, V6Stats } from '@/types/api';
import { api } from '@/lib/api';

export default function Home() {
  const [activeSession, setActiveSession] = useState<Session | null>(null);
  const [activeResultSetId, setActiveResultSetId] = useState<number | null>(null);
  const [activeEvidence, setActiveEvidence] = useState<EvidenceRef | null>(null);
  const [latestV6Stats, setLatestV6Stats] = useState<V6Stats | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleSessionSelect = (session: Session) => {
    setActiveSession(session);
    setActiveResultSetId(null);
    setActiveEvidence(null);
    setLatestV6Stats(null);

    // Rehydrate right pane with latest known result for this session
    api
      .getSessionState(session.id)
      .then((state) => {
        setActiveResultSetId(state.latest_result_set_id ?? null);
      })
      .catch(() => {
        // Best-effort: keep empty state if state endpoint fails
      });
  };

  const handleResultSetUpdate = (resultSetId: number | null) => {
    setActiveResultSetId(resultSetId);
  };

  const handleV6StatsUpdate = (stats: V6Stats | null) => {
    setLatestV6Stats(stats);
  };

  const handleProcessingChange = (processing: boolean) => {
    setIsProcessing(processing);
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

      {/* Center Pane: Chat Conversation */}
      <div className="pane pane-center">
        <div className="pane-header">
          {activeSession ? (
            <>
              <span>{activeSession.label}</span>
              <span className="header-badge">V7 Citation-Enforced</span>
            </>
          ) : (
            'Friday Research Console'
          )}
        </div>
        <Conversation
          session={activeSession}
          onResultSetUpdate={handleResultSetUpdate}
          onV6StatsUpdate={handleV6StatsUpdate}
          onProcessingChange={handleProcessingChange}
          onEvidenceClick={handleEvidenceClick}
        />
      </div>

      {/* Right Pane: Workflow + Results + Evidence */}
      <div className="pane" style={{ borderRight: 'none' }}>
        <RightPane
          v6Stats={latestV6Stats}
          isProcessing={isProcessing}
          resultSetId={activeResultSetId}
          activeEvidence={activeEvidence}
          onResultSetIdUpdate={setActiveResultSetId}
          onEvidenceClick={handleEvidenceClick}
        />
      </div>
    </div>
  );
}
