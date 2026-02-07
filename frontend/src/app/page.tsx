'use client';

import { useState, useCallback, useEffect } from 'react';
import { SessionList } from '@/components/SessionList';
import { Conversation } from '@/components/Conversation';
import { RightPane } from '@/components/RightPane';
import { EvidenceViewer } from '@/components/EvidenceViewer';
import { AuthHeader } from '@/components/AuthHeader';
import type { Session, EvidenceRef, V9ChatResponse, V9ProgressEvent, V9EvidenceBullet, UserSelectedScope } from '@/types/api';
import type { AuthUser } from '@/lib/api';
import { api } from '@/lib/api';

export default function Home() {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [activeSession, setActiveSession] = useState<Session | null>(null);
  const [activeEvidence, setActiveEvidence] = useState<EvidenceRef | null>(null);
  const [lastV9Response, setLastV9Response] = useState<V9ChatResponse | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progressSteps, setProgressSteps] = useState<V9ProgressEvent[]>([]);
  const [evidenceBullets, setEvidenceBullets] = useState<V9EvidenceBullet[]>([]);
  const [sessionScope, setSessionScope] = useState<UserSelectedScope | null>(null);

  const handleSessionSelect = (session: Session) => {
    setActiveSession(session);
    setActiveEvidence(null);
    setLastV9Response(null);
    setProgressSteps([]);
    setEvidenceBullets([]);
    // Load scope from session
    setSessionScope(session.scope_json || { mode: 'full_archive' });
  };

  const handleV9Response = (response: V9ChatResponse | null) => {
    setLastV9Response(response);
  };

  const handleProcessingChange = (processing: boolean) => {
    setIsProcessing(processing);
  };

  const handleEvidenceClick = (evidence: EvidenceRef | null) => {
    setActiveEvidence(evidence);
  };

  const handleProgressUpdate = useCallback((steps: V9ProgressEvent[], bullets: V9EvidenceBullet[]) => {
    setProgressSteps(steps);
    setEvidenceBullets(bullets);
  }, []);

  const handleSessionDelete = () => {
    setActiveSession(null);
    setActiveEvidence(null);
    setLastV9Response(null);
    setIsProcessing(false);
    setProgressSteps([]);
    setEvidenceBullets([]);
    setSessionScope(null);
  };

  useEffect(() => {
    api.getAuthMe().then(setUser);
  }, []);

  const showingEvidence = !!activeEvidence;

  return (
    <div className="app-container">
      <AuthHeader user={user} onLogout={() => setUser(null)} />
      {/* Left Pane: Sessions */}
      <div className="pane">
        <div className="pane-header">Sessions</div>
        <div className="pane-content">
          <SessionList
            activeSessionId={activeSession?.id}
            onSessionSelect={handleSessionSelect}
            onSessionDelete={handleSessionDelete}
          />
        </div>
      </div>

      {/* Center Pane: Chat Conversation OR Document Viewer */}
      <div className="pane pane-center">
        {showingEvidence ? (
          <>
            <div className="pane-header">
              <span>Document Viewer</span>
              <button
                className="btn-secondary"
                onClick={() => setActiveEvidence(null)}
                style={{ fontSize: '13px', padding: '4px 12px' }}
              >
                ← Back to Chat
              </button>
            </div>
            <EvidenceViewer
              evidence={activeEvidence}
              onClose={() => setActiveEvidence(null)}
            />
          </>
        ) : (
          <>
            <div className="pane-header">
              {activeSession ? (
                <>
                  <span>{activeSession.label}</span>
                  <span className="header-badge">V9</span>
                </>
              ) : (
                'Friday Research Console'
              )}
            </div>
            <Conversation
              session={activeSession}
              onV9Response={handleV9Response}
              onProcessingChange={handleProcessingChange}
              onEvidenceClick={handleEvidenceClick}
              onProgressUpdate={handleProgressUpdate}
            />
          </>
        )}
      </div>

      {/* Right Pane: Investigation */}
      <div className="pane" style={{ borderRight: 'none' }}>
        <RightPane
          v9Response={lastV9Response}
          isProcessing={isProcessing}
          onEvidenceClick={handleEvidenceClick}
          progressSteps={progressSteps}
          evidenceBullets={evidenceBullets}
          sessionId={activeSession?.id}
          sessionScope={sessionScope}
          onScopeChange={setSessionScope}
        />
      </div>
    </div>
  );
}
