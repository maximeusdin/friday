'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { SessionList } from '@/components/SessionList';
import { Conversation } from '@/components/Conversation';
import { RightPane } from '@/components/RightPane';
import { EvidenceViewer } from '@/components/EvidenceViewer';
import { AuthHeader } from '@/components/AuthHeader';
import type { Session, EvidenceRef, V9ChatResponse, V9ProgressEvent, V9EvidenceBullet, UserSelectedScope, CollectionNode } from '@/types/api';
import type { AuthUser } from '@/lib/api';
import { api, getLoginUrl } from '@/lib/api';
import { normalizeScope } from '@/lib/scope';

export default function Home() {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [activeSession, setActiveSession] = useState<Session | null>(null);
  const [activeEvidence, setActiveEvidence] = useState<EvidenceRef | null>(null);
  const [lastV9Response, setLastV9Response] = useState<V9ChatResponse | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progressSteps, setProgressSteps] = useState<V9ProgressEvent[]>([]);
  const [evidenceBullets, setEvidenceBullets] = useState<V9EvidenceBullet[]>([]);

  // --- Scope state (staged-commit model) ---
  const [activeScope, setActiveScope] = useState<UserSelectedScope | null>(null);
  const [activeScopeRevision, setActiveScopeRevision] = useState(0);
  const [lastUsedScope, setLastUsedScope] = useState<UserSelectedScope | null>(null);
  const pendingRunScopeRef = useRef<UserSelectedScope | null>(null);
  const [activeRightTab, setActiveRightTab] = useState<'scope' | 'investigation'>('scope');
  const [hasDraftChanges, setHasDraftChanges] = useState(false);
  const [collections, setCollections] = useState<CollectionNode[]>([]);

  // Load collections cache once on mount (retry once after 2s on failure)
  useEffect(() => {
    const load = () => api.getCollectionsTree().then(setCollections);
    load().catch(() => {
      setTimeout(() => load().catch(console.error), 2000);
    });
  }, []);

  // Auto-switch to investigation tab when a query starts processing
  useEffect(() => {
    if (isProcessing) setActiveRightTab('investigation');
  }, [isProcessing]);

  // --- Session handlers ---

  const handleSessionSelect = (session: Session) => {
    setActiveSession(session);
    setActiveEvidence(null);
    setLastV9Response(null);
    setProgressSteps([]);
    setEvidenceBullets([]);
    // Scope: deterministic reset
    setActiveScope(session.scope_json || { mode: 'full_archive' });
    setActiveScopeRevision(1); // deterministic reset, not increment
    setLastUsedScope(null);
    pendingRunScopeRef.current = null;
    setActiveRightTab('scope');
    setHasDraftChanges(false);
  };

  const handleSessionDelete = () => {
    setActiveSession(null);
    setActiveEvidence(null);
    setLastV9Response(null);
    setIsProcessing(false);
    setProgressSteps([]);
    setEvidenceBullets([]);
    setActiveScope(null);
    setActiveScopeRevision(0);
    setLastUsedScope(null);
    pendingRunScopeRef.current = null;
    setHasDraftChanges(false);
  };

  // --- Scope handlers (useCallback-stable) ---

  const handleApplyScope = useCallback((scope: UserSelectedScope) => {
    setActiveScope(scope);
    setActiveScopeRevision(r => r + 1);
    if (activeSession?.id) {
      api.updateSessionScope(activeSession.id, scope).catch(console.error);
    }
  }, [activeSession?.id]);

  const handleQuerySent = useCallback((scopeSent: UserSelectedScope) => {
    pendingRunScopeRef.current = normalizeScope(scopeSent);
  }, []);

  const handleDraftDirtyChange = useCallback((dirty: boolean) => {
    setHasDraftChanges(dirty);
  }, []);

  const handleEditScope = useCallback(() => {
    setActiveRightTab('scope');
  }, []);

  // --- V9 response handler ---

  const handleV9Response = useCallback((response: V9ChatResponse | null) => {
    setLastV9Response(response);
    if (response?.scope_override?.run_scope) {
      setLastUsedScope(normalizeScope({
        mode: response.scope_override.run_scope.mode,
        included_collection_ids: response.scope_override.run_scope.included_collection_ids,
        included_document_ids: response.scope_override.run_scope.included_document_ids,
      }));
    } else if (pendingRunScopeRef.current) {
      setLastUsedScope(pendingRunScopeRef.current);
    }
  }, []);

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

  // --- Auth ---

  useEffect(() => {
    let cancelled = false;
    const checkAuth = async () => {
      let u = await api.getAuthMe();
      if (!u) {
        await new Promise((r) => setTimeout(r, 500));
        if (!cancelled) u = await api.getAuthMe();
      }
      if (!cancelled) {
        setUser(u);
        setAuthChecked(true);
      }
    };
    checkAuth();
    return () => { cancelled = true; };
  }, []);

  const isAuthenticated = !!user;
  const showingEvidence = !!activeEvidence;

  return (
    <div className="app-wrapper">
      {/* Persistent top header */}
      <AuthHeader user={user} onLogout={() => setUser(null)} />

      {/* Main content area (3-pane grid) */}
      <div className="app-container">
        {/* Auth gate overlay – blocks interaction when unauthenticated */}
        {authChecked && !isAuthenticated && (
          <div className="auth-overlay">
            <div className="auth-overlay-card">
              <h2>Sign in required</h2>
              <p>
                Friday uses secure login. You&rsquo;ll be redirected to sign in
                and then returned here.
              </p>
              <a href={getLoginUrl()} className="btn-signin">
                Sign in
              </a>
              <div className="auth-note">
                You&rsquo;ll be redirected to our secure login.
              </div>
            </div>
          </div>
        )}

        {/* Left Pane: Sessions */}
        <div className={`pane${!isAuthenticated ? ' pane-locked' : ''}`}>
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
                  &larr; Back to Chat
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
                activeScope={activeScope}
                lastUsedScope={lastUsedScope}
                collections={collections}
                hasDraftChanges={hasDraftChanges}
                onEditScope={handleEditScope}
                onQuerySent={handleQuerySent}
                onMakeActiveScope={handleApplyScope}
              />
            </>
          )}
        </div>

        {/* Right Pane: Scope & Investigation */}
        <div className="pane" style={{ borderRight: 'none' }}>
          <RightPane
            v9Response={lastV9Response}
            isProcessing={isProcessing}
            onEvidenceClick={handleEvidenceClick}
            progressSteps={progressSteps}
            evidenceBullets={evidenceBullets}
            sessionId={activeSession?.id}
            activeScope={activeScope}
            activeTab={activeRightTab}
            onTabChange={setActiveRightTab}
            onApplyScope={handleApplyScope}
            onDraftDirtyChange={handleDraftDirtyChange}
            activeScopeRevision={activeScopeRevision}
            collections={collections}
          />
        </div>
      </div>
    </div>
  );
}
