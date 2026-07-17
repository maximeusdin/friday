'use client';

import { useState, useCallback, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { useQueryClient } from '@tanstack/react-query';
import { SessionList } from '@/components/SessionList';
import { Conversation } from '@/components/Conversation';
import { RightPane } from '@/components/RightPane';
import { SearchTab } from '@/components/SearchTab';

// EvidenceViewer pulls in react-pdf / pdfjs-dist, which is browser-only: pdfjs 4.x
// calls Promise.withResolvers at module-init, so importing it during the static
// export's server render crashes on Node < 21.7. Load it client-side only.
const EvidenceViewer = dynamic(
  () => import('@/components/EvidenceViewer').then((m) => m.EvidenceViewer),
  { ssr: false },
);
import { AuthHeader } from '@/components/AuthHeader';
import type { Session, EvidenceRef, UserSelectedScope, CollectionNode } from '@/types/api';
import type { AuthUser } from '@/lib/api';
import { api, getLoginUrl } from '@/lib/api';
import { normalizeScope } from '@/lib/scope';
import { useChatRun } from '@/lib/chatRunStore';

export default function Home() {
  const queryClient = useQueryClient();
  const [user, setUser] = useState<AuthUser | null>(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [activeSession, setActiveSession] = useState<Session | null>(null);
  const [activeEvidence, setActiveEvidence] = useState<EvidenceRef | null>(null);

  // The active session's chat run comes from the shared store (per-session, so runs
  // continue when you switch sessions and several can be in flight at once).
  const activeRun = useChatRun(activeSession?.id ?? null);
  const lastV9Response = activeRun.lastV9;

  // --- Center pane tab: Chat | Search ---
  const [activeTab, setActiveTab] = useState<'chat' | 'search'>('chat');
  const [activeSearchResultSetId, setActiveSearchResultSetId] = useState<string | null>(null);

  // Question queued from a splash "Try asking" card — auto-sent once the new session mounts
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  // Query queued from a Search splash "Example queries" card — auto-run once the new session mounts
  const [pendingSearch, setPendingSearch] = useState<string | null>(null);

  // --- Scope state (staged-commit model) ---
  const [activeScope, setActiveScope] = useState<UserSelectedScope | null>(null);
  const [activeScopeRevision, setActiveScopeRevision] = useState(0);
  const [lastUsedScope, setLastUsedScope] = useState<UserSelectedScope | null>(null);
  const [hasDraftChanges, setHasDraftChanges] = useState(false);
  const [collections, setCollections] = useState<CollectionNode[]>([]);

  // --- Scope pane sizing (collapsible + draggable divider, persisted) ---
  const SCOPE_MIN = 320;
  const SCOPE_DEFAULT = 480;
  const [scopeWidth, setScopeWidth] = useState(SCOPE_DEFAULT);
  const [scopeCollapsed, setScopeCollapsed] = useState(false);

  // Load collections cache once on mount (retry once after 2s on failure)
  useEffect(() => {
    const load = () => api.getCollectionsTree().then(setCollections);
    load().catch(() => {
      setTimeout(() => load().catch(console.error), 2000);
    });
  }, []);

  // Restore persisted scope-pane width / collapsed state (client-only)
  useEffect(() => {
    try {
      const w = Number(localStorage.getItem('friday.scopeWidth'));
      if (w >= SCOPE_MIN) setScopeWidth(w);
      if (localStorage.getItem('friday.scopeCollapsed') === '1') setScopeCollapsed(true);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    try { localStorage.setItem('friday.scopeCollapsed', scopeCollapsed ? '1' : '0'); } catch { /* ignore */ }
  }, [scopeCollapsed]);

  // Drag the divider to resize the scope pane
  const startScopeResize = useCallback((e: React.PointerEvent) => {
    e.preventDefault();
    const onMove = (ev: PointerEvent) => {
      const max = Math.min(720, window.innerWidth * 0.6);
      const next = Math.max(SCOPE_MIN, Math.min(max, window.innerWidth - ev.clientX));
      setScopeWidth(next);
    };
    const onUp = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      setScopeWidth((w) => {
        try { localStorage.setItem('friday.scopeWidth', String(Math.round(w))); } catch { /* ignore */ }
        return w;
      });
    };
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  }, []);

  // --- Session handlers ---

  const handleSessionSelect = async (session: Session) => {
    // Fetch full session (scope_json, output_mode) when selecting
    try {
      const full = await api.getSession(session.id);
      setActiveSession(full);
    } catch {
      setActiveSession(session);
    }
    setActiveEvidence(null);
    setActiveSearchResultSetId(null);
    // Scope: deterministic reset
    setActiveScope(session.scope_json || { mode: 'full_archive' });
    setActiveScopeRevision(1); // deterministic reset, not increment
    setLastUsedScope(null);
    setHasDraftChanges(false);
  };

  // Start a fresh session from a splash example and queue the question to auto-send.
  const handleExampleQuestion = async (question: string) => {
    const base = question.length > 48 ? `${question.slice(0, 48).trimEnd()}…` : question;
    // Dedupe against the cached sessions list, mirroring SessionList's resolveLabel.
    const existing = new Set(
      ((queryClient.getQueryData(['sessions']) as Session[] | undefined) ?? []).map((s) => s.label)
    );
    let label = base;
    for (let n = 1; existing.has(label); n++) label = `${base} (${n})`;
    try {
      const created = await api.createSession({ label });
      queryClient.invalidateQueries({ queryKey: ['sessions'] });
      setActiveTab('chat');
      await handleSessionSelect(created);
      setPendingQuestion(question);
    } catch (err) {
      console.error('Failed to start session from example question', err);
    }
  };

  // Start a fresh session from a Search splash example and queue the query to auto-run.
  const handleExampleSearch = async (searchQuery: string) => {
    const base = searchQuery.length > 48 ? `${searchQuery.slice(0, 48).trimEnd()}…` : searchQuery;
    const existing = new Set(
      ((queryClient.getQueryData(['sessions']) as Session[] | undefined) ?? []).map((s) => s.label)
    );
    let label = base;
    for (let n = 1; existing.has(label); n++) label = `${base} (${n})`;
    try {
      const created = await api.createSession({ label });
      queryClient.invalidateQueries({ queryKey: ['sessions'] });
      setActiveTab('search');
      await handleSessionSelect(created);
      setPendingSearch(searchQuery);
    } catch (err) {
      console.error('Failed to start session from example search', err);
    }
  };

  const handleSessionDelete = () => {
    setActiveSession(null);
    setActiveEvidence(null);
    setActiveSearchResultSetId(null);
    setActiveScope(null);
    setActiveScopeRevision(0);
    setLastUsedScope(null);
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

  const handleDraftDirtyChange = useCallback((dirty: boolean) => {
    setHasDraftChanges(dirty);
  }, []);

  const handleEditScope = useCallback(() => {
    // Right pane is scope-only; no tab to switch
  }, []);

  // --- V9 response handler ---

  const handleViewSearchResultSet = useCallback((resultSetId: string) => {
    setActiveTab('search');
    setActiveSearchResultSetId(resultSetId);
  }, []);

  const handleSearchRun = useCallback(() => {
    setActiveSearchResultSetId(null);
  }, []);

  // Remember the scope the active session's last answer actually ran against, so the
  // scope bar can flag "changed since last query". Derived from the store: prefer the
  // answer's scope_override, else the scope the run was launched with.
  useEffect(() => {
    const response = activeRun.lastV9;
    if (!response) return;
    if (response.scope_override?.run_scope) {
      setLastUsedScope(normalizeScope({
        mode: response.scope_override.run_scope.mode,
        included_collection_ids: response.scope_override.run_scope.included_collection_ids,
        included_document_ids: response.scope_override.run_scope.included_document_ids,
      }));
    } else if (activeRun.runScope) {
      setLastUsedScope(normalizeScope(activeRun.runScope));
    }
  }, [activeRun.lastV9, activeRun.runScope]);

  const handleEvidenceClick = (evidence: EvidenceRef | null) => {
    setActiveEvidence(evidence);
    setActiveSearchResultSetId(null);
  };

  const handleOpenPageFromSearch = (evidence: EvidenceRef, resultSetId: string) => {
    setActiveSearchResultSetId(resultSetId);
    setActiveEvidence(evidence);
  };

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
  const fromSearch = !!activeSearchResultSetId;

  const handleCloseEvidence = () => {
    setActiveEvidence(null);
    if (fromSearch) {
      setActiveTab('search');
    }
    // Do NOT clear activeSearchResultSetId — keep results visible when returning from doc viewer
  };

  return (
    <div className="app-wrapper">
      {/* Persistent top header */}
      <AuthHeader user={user} onLogout={() => setUser(null)} />

      {/* Main content area (3-pane grid) */}
      <div
        className="app-container"
        style={{ ['--scope-w' as string]: scopeCollapsed ? '40px' : `${scopeWidth}px` } as React.CSSProperties}
      >
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

        {/* Center Pane: Chat | Search tabs, or Document Viewer overlay */}
        <div className="pane pane-center" style={{ position: 'relative' }}>
          {showingEvidence && (
            <>
              <div className="pane-header" style={{ position: 'relative', zIndex: 11 }}>
                <span>Document Viewer</span>
                <button
                  className="btn-secondary"
                  onClick={handleCloseEvidence}
                  style={{ fontSize: '13px', padding: '4px 12px' }}
                >
                  &larr; {fromSearch ? 'Back to results' : 'Back to Chat'}
                </button>
              </div>
              <div style={{ position: 'absolute', inset: 0, top: 49, zIndex: 10, background: 'var(--color-bg)', overflow: 'auto' }}>
                <EvidenceViewer
                  evidence={activeEvidence}
                  onClose={handleCloseEvidence}
                  backLabel={fromSearch ? 'Back to results' : 'Back to Chat'}
                />
              </div>
            </>
          )}
          <div style={{ display: showingEvidence ? 'none' : 'flex', flexDirection: 'column', flex: 1, minHeight: 0, overflow: 'hidden' }}>
            <>
              <div className="pane-header pane-header-tabs">
                <div className="tab-bar">
                  <button
                    type="button"
                    className={`tab-btn ${activeTab === 'chat' ? 'tab-btn-active' : ''}`}
                    onClick={() => setActiveTab('chat')}
                  >
                    Chat
                  </button>
                  <button
                    type="button"
                    className={`tab-btn ${activeTab === 'search' ? 'tab-btn-active' : ''}`}
                    onClick={() => setActiveTab('search')}
                  >
                    Search
                  </button>
                </div>
                {activeTab === 'chat' && activeSession && (
                  <span className="pane-header-label">{activeSession.label}</span>
                )}
                {activeTab === 'search' && (
                  <span className="pane-header-label">Archive Search</span>
                )}
              </div>
              {activeTab === 'chat' ? (
                <Conversation
                session={activeSession}
                onViewSearchResultSet={handleViewSearchResultSet}
                onOpenSearchTab={() => setActiveTab('search')}
                onEvidenceClick={handleEvidenceClick}
                activeScope={activeScope}
                lastUsedScope={lastUsedScope}
                collections={collections}
                hasDraftChanges={hasDraftChanges}
                onEditScope={handleEditScope}
                onMakeActiveScope={handleApplyScope}
                onExampleQuestion={handleExampleQuestion}
                pendingQuestion={pendingQuestion}
                onPendingQuestionConsumed={() => setPendingQuestion(null)}
              />
              ) : (
                <SearchTab
                  activeScope={activeScope}
                  sessionId={activeSession?.id ?? null}
                  onOpenPage={handleOpenPageFromSearch}
                  externalResultSetId={activeTab === 'search' ? activeSearchResultSetId : null}
                  onSearchRun={handleSearchRun}
                  collections={collections}
                  onExampleSearch={handleExampleSearch}
                  pendingSearchQuery={pendingSearch}
                  onPendingSearchConsumed={() => setPendingSearch(null)}
                />
              )}
            </>
          </div>
        </div>

        {/* Right Pane: Scope (collapsible + resizable) */}
        <div className="pane scope-pane" style={{ borderRight: 'none', position: 'relative' }}>
          {!scopeCollapsed && (
            <div
              className="scope-resize-handle"
              onPointerDown={startScopeResize}
              role="separator"
              aria-orientation="vertical"
              title="Drag to resize"
            />
          )}
          {scopeCollapsed ? (
            <button
              className="scope-rail"
              onClick={() => setScopeCollapsed(false)}
              title="Show scope panel"
            >
              <span className="scope-rail-chevron">‹</span>
              <span className="scope-rail-label">Scope</span>
              {hasDraftChanges && <span className="scope-rail-dot" title="Unapplied scope changes" />}
            </button>
          ) : (
            <RightPane
              v9Response={lastV9Response}
              sessionId={activeSession?.id}
              activeScope={activeScope}
              onApplyScope={handleApplyScope}
              onDraftDirtyChange={handleDraftDirtyChange}
              activeScopeRevision={activeScopeRevision}
              collections={collections}
              onCollapse={() => setScopeCollapsed(true)}
            />
          )}
        </div>
      </div>
    </div>
  );
}
