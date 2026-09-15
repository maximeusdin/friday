'use client';

import { useState, useCallback, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { useQueryClient } from '@tanstack/react-query';
import { SessionSidebar } from '@/components/SessionSidebar';
import { Conversation } from '@/components/Conversation';
import { SearchTab } from '@/components/SearchTab';
import { AppHeader } from '@/components/AppHeader';
import { Icon } from '@/components/ui/Icon';
import { Toaster } from '@/components/ui/Toast';

// EvidenceViewer pulls in react-pdf / pdfjs-dist, which is browser-only: pdfjs 4.x
// calls Promise.withResolvers at module-init, so importing it during the static
// export's server render crashes on Node < 21.7. Load it client-side only.
const EvidenceViewer = dynamic(
  () => import('@/components/EvidenceViewer').then((m) => m.EvidenceViewer),
  { ssr: false },
);

import type { Session, EvidenceRef, UserSelectedScope, CollectionNode } from '@/types/api';
import type { AuthUser } from '@/lib/api';
import { api, getLoginUrl } from '@/lib/api';
import { titleFromQuestion } from '@/lib/format';

const FULL_ARCHIVE: UserSelectedScope = { mode: 'full_archive' };

export default function Home() {
  const queryClient = useQueryClient();
  const [user, setUser] = useState<AuthUser | null>(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [activeSession, setActiveSession] = useState<Session | null>(null);
  const [activeEvidence, setActiveEvidence] = useState<EvidenceRef | null>(null);
  // True when the page was opened as a document deep link (/?document_id=…&pdf_page=…)
  // — the URLs emitted by the MCP connector, search exports, and share links.
  const [deepLinkedDoc, setDeepLinkedDoc] = useState(false);

  const [activeTab, setActiveTab] = useState<'chat' | 'search'>('chat');
  const [activeSearchResultSetId, setActiveSearchResultSetId] = useState<string | null>(null);

  // Queued from the welcome screen: the session is created on the first question,
  // so nobody has to name a session before they can ask anything.
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [pendingSearch, setPendingSearch] = useState<string | null>(null);

  // Scope applies as soon as it is chosen — see ScopeControl for why there is no
  // apply step. It is persisted to the session so it survives a reload.
  const [activeScope, setActiveScope] = useState<UserSelectedScope>(FULL_ARCHIVE);
  const [collections, setCollections] = useState<CollectionNode[]>([]);

  const [sidebarOpen, setSidebarOpen] = useState(true);
  // Bumped on every "New session" click. Clicking it while already on the welcome
  // screen changes no state, so without this the button would feel dead.
  const [newSessionNonce, setNewSessionNonce] = useState(0);

  // Deep link: open the document viewer directly from /?document_id=…&pdf_page=….
  // Plain window.location (not useSearchParams) so the static export needs no
  // Suspense boundary. Runs once on mount.
  useEffect(() => {
    try {
      const params = new URLSearchParams(window.location.search);
      const docId = Number(params.get('document_id'));
      if (Number.isFinite(docId) && docId > 0) {
        const page = Number(params.get('pdf_page'));
        setActiveEvidence({
          document_id: docId,
          pdf_page: Number.isFinite(page) && page > 0 ? page : 1,
        });
        setDeepLinkedDoc(true);
      }
    } catch { /* ignore */ }
  }, []);

  // Load the collections cache once on mount (retry once after 2s on failure).
  useEffect(() => {
    const load = () => api.getCollectionsTree().then(setCollections);
    load().catch(() => {
      setTimeout(() => load().catch(console.error), 2000);
    });
  }, []);

  // Restore the sidebar preference; narrow viewports start with it closed so the
  // drawer doesn't cover the conversation on first paint.
  useEffect(() => {
    try {
      const stored = localStorage.getItem('friday.sidebar');
      if (stored) setSidebarOpen(stored === 'open');
      else if (window.innerWidth < 900) setSidebarOpen(false);
    } catch { /* ignore */ }
  }, []);

  const toggleSidebar = useCallback(() => {
    setSidebarOpen((open) => {
      try { localStorage.setItem('friday.sidebar', open ? 'closed' : 'open'); } catch { /* ignore */ }
      return !open;
    });
  }, []);

  // --- Sessions ---

  const handleSessionSelect = async (session: Session) => {
    // Fetch the full session (scope_json, output_mode) when selecting.
    let full = session;
    try {
      full = await api.getSession(session.id);
    } catch { /* fall back to the list row */ }
    setActiveSession(full);
    setActiveEvidence(null);
    setActiveSearchResultSetId(null);
    setActiveScope(full.scope_json || FULL_ARCHIVE);
    if (window.innerWidth < 900) setSidebarOpen(false);
  };

  const handleNewSession = () => {
    setNewSessionNonce((n) => n + 1);
    setActiveSession(null);
    setActiveEvidence(null);
    setActiveSearchResultSetId(null);
    setActiveScope(FULL_ARCHIVE);
    setActiveTab('chat');
    if (window.innerWidth < 900) setSidebarOpen(false);
  };

  /** Create the session a question implies, keeping whatever scope was chosen first. */
  const createSessionFor = useCallback(async (seed: string): Promise<Session | null> => {
    const base = titleFromQuestion(seed);
    const existing = new Set(
      ((queryClient.getQueryData(['sessions']) as Session[] | undefined) ?? []).map((s) => s.label),
    );
    let label = base;
    for (let n = 1; existing.has(label); n++) label = `${base} (${n})`;
    try {
      const created = await api.createSession({ label });
      queryClient.invalidateQueries({ queryKey: ['sessions'] });
      setActiveSession(created);
      setActiveEvidence(null);
      setActiveSearchResultSetId(null);
      if (activeScope.mode === 'custom') {
        api.updateSessionScope(created.id, activeScope).catch(console.error);
      }
      return created;
    } catch (err) {
      console.error('Failed to create session', err);
      return null;
    }
  }, [queryClient, activeScope]);

  const handleStartSession = useCallback(async (question: string) => {
    setActiveTab('chat');
    const created = await createSessionFor(question);
    if (created) setPendingQuestion(question);
  }, [createSessionFor]);

  const handleStartSearchSession = useCallback(async (query: string) => {
    setActiveTab('search');
    const created = await createSessionFor(query);
    if (created) setPendingSearch(query);
  }, [createSessionFor]);

  const handleSessionDelete = () => {
    setActiveSession(null);
    setActiveEvidence(null);
    setActiveSearchResultSetId(null);
    setActiveScope(FULL_ARCHIVE);
  };

  // --- Scope ---

  const handleScopeChange = useCallback((scope: UserSelectedScope) => {
    setActiveScope(scope);
    if (activeSession?.id) {
      api.updateSessionScope(activeSession.id, scope).catch(console.error);
    }
  }, [activeSession?.id]);

  // --- Evidence / search ---

  const handleViewSearchResultSet = useCallback((resultSetId: string) => {
    setActiveTab('search');
    setActiveSearchResultSetId(resultSetId);
  }, []);

  const handleSearchRun = useCallback(() => setActiveSearchResultSetId(null), []);

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
    if (fromSearch) setActiveTab('search');
    // Keep activeSearchResultSetId so results are still there on return.
  };

  return (
    <div className="app" data-sidebar={sidebarOpen ? 'open' : 'closed'}>
      <AppHeader
        user={user}
        onLogout={() => setUser(null)}
        sidebarOpen={sidebarOpen}
        onToggleSidebar={toggleSidebar}
      />

      <div className="app-body">
        {/* The archive documents are public: a deep-linked document stays readable
            without an account, because citation links from Claude and from exports
            must open for visitors. Closing the viewer restores the gate. */}
        {authChecked && !isAuthenticated && !(deepLinkedDoc && showingEvidence) && (
          <div className="auth-gate">
            <div className="auth-card">
              <h2>Sign in to use Friday</h2>
              <p>
                You&rsquo;ll be redirected to our secure login and returned here.
              </p>
              <a href={getLoginUrl()} className="btn-primary btn-lg">Sign in</a>
            </div>
          </div>
        )}

        {/* Tapping outside the drawer closes it. Desktop hides this entirely — see
            the 900px breakpoint in globals.css. */}
        <button
          type="button"
          className="sidebar-scrim"
          aria-label="Close sessions"
          tabIndex={sidebarOpen ? 0 : -1}
          onClick={toggleSidebar}
        />

        <SessionSidebar
          activeSessionId={activeSession?.id}
          onSessionSelect={handleSessionSelect}
          onSessionDelete={handleSessionDelete}
          onNewSession={handleNewSession}
        />

        <main className="workspace">
          {showingEvidence ? (
            <EvidenceViewer
              evidence={activeEvidence}
              onClose={handleCloseEvidence}
              backLabel={fromSearch ? 'Back to results' : 'Back to chat'}
            />
          ) : (
            <>
              <div className="ws-bar">
                <div className="segmented" role="tablist" aria-label="Chat or Search">
                  <button
                    type="button"
                    role="tab"
                    aria-selected={activeTab === 'chat'}
                    className="segmented-item"
                    onClick={() => setActiveTab('chat')}
                  >
                    <Icon name="message" size={15} />
                    Chat
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={activeTab === 'search'}
                    className="segmented-item"
                    onClick={() => setActiveTab('search')}
                  >
                    <Icon name="search" size={15} />
                    Search
                  </button>
                </div>
                <span className="ws-bar-title">
                  {activeSession ? activeSession.label : 'New session'}
                </span>
              </div>

              {activeTab === 'chat' ? (
                <Conversation
                  session={activeSession}
                  collections={collections}
                  activeScope={activeScope}
                  onScopeChange={handleScopeChange}
                  onViewSearchResultSet={handleViewSearchResultSet}
                  onOpenSearchTab={() => setActiveTab('search')}
                  onEvidenceClick={handleEvidenceClick}
                  onStartSession={handleStartSession}
                  newSessionNonce={newSessionNonce}
                  pendingQuestion={pendingQuestion}
                  onPendingQuestionConsumed={() => setPendingQuestion(null)}
                />
              ) : (
                <SearchTab
                  activeScope={activeScope}
                  onScopeChange={handleScopeChange}
                  collections={collections}
                  sessionId={activeSession?.id ?? null}
                  onOpenPage={handleOpenPageFromSearch}
                  externalResultSetId={activeSearchResultSetId}
                  onSearchRun={handleSearchRun}
                  onStartSession={handleStartSearchSession}
                  pendingSearchQuery={pendingSearch}
                  onPendingSearchConsumed={() => setPendingSearch(null)}
                />
              )}
            </>
          )}
        </main>
      </div>

      <Toaster />
    </div>
  );
}
