'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
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
import { useLayout, TABLET_MAX } from '@/lib/useLayout';

/** Overlays that become history entries on touch layouts, so the back gesture
 *  closes them instead of leaving the site. */
type Layer = 'viewer' | 'drawer';

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

  const layout = useLayout();

  // Stamp the layout on <html> as well as the shell: popovers and modals are
  // portalled to <body>, and touch sizing has to reach them too.
  useEffect(() => {
    const root = document.documentElement;
    root.dataset.layout = layout.mode;
    root.dataset.touch = layout.touch ? 'true' : 'false';
  }, [layout.mode, layout.touch]);

  // --- History-backed overlays ---------------------------------------------
  // On a phone, the document viewer and the sessions drawer cover the whole
  // screen, and people close full-screen things with the back gesture. Each
  // one pushes a history entry when it opens; popstate closes it. Closing from
  // the UI goes through history.back() so the two paths stay in step.
  const layers = useRef<Layer[]>([]);

  const pushLayer = useCallback((layer: Layer) => {
    if (layout.isDesktop) return;
    try {
      window.history.pushState({ friday: layer }, '');
      layers.current.push(layer);
    } catch { /* history unavailable: fall back to plain state changes */ }
  }, [layout.isDesktop]);

  /** Close a layer from the UI: pop history if we pushed it, else just close. */
  const closeLayer = useCallback((layer: Layer, close: () => void) => {
    const top = layers.current[layers.current.length - 1];
    if (top === layer) {
      window.history.back(); // popstate does the closing
    } else {
      close();
    }
  }, []);

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

  // Restore the sidebar preference on desktop, where it is a persistent rail.
  // On tablet and phone it is an overlay drawer, and an overlay always starts
  // closed, whatever was saved on a larger screen.
  useEffect(() => {
    if (window.innerWidth < TABLET_MAX) {
      setSidebarOpen(false);
      return;
    }
    try {
      const stored = localStorage.getItem('friday.sidebar');
      if (stored) setSidebarOpen(stored === 'open');
    } catch { /* ignore */ }
  }, []);

  const openSidebar = useCallback(() => {
    setSidebarOpen(true);
    pushLayer('drawer');
  }, [pushLayer]);

  const closeSidebar = useCallback(() => {
    closeLayer('drawer', () => setSidebarOpen(false));
  }, [closeLayer]);

  const toggleSidebar = useCallback(() => {
    if (layout.isDesktop) {
      // Desktop: a persistent preference, not an overlay.
      setSidebarOpen((open) => {
        try { localStorage.setItem('friday.sidebar', open ? 'closed' : 'open'); } catch { /* ignore */ }
        return !open;
      });
      return;
    }
    if (sidebarOpen) closeSidebar();
    else openSidebar();
  }, [layout.isDesktop, sidebarOpen, openSidebar, closeSidebar]);

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
    if (!layout.isDesktop && sidebarOpen) closeSidebar();
  };

  const handleNewSession = () => {
    setNewSessionNonce((n) => n + 1);
    setActiveSession(null);
    setActiveEvidence(null);
    setActiveSearchResultSetId(null);
    setActiveScope(FULL_ARCHIVE);
    setActiveTab('chat');
    if (!layout.isDesktop && sidebarOpen) closeSidebar();
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
    setActiveSearchResultSetId(null);
    if (evidence && !activeEvidence) pushLayer('viewer');
    setActiveEvidence(evidence);
  };

  const handleOpenPageFromSearch = (evidence: EvidenceRef, resultSetId: string) => {
    setActiveSearchResultSetId(resultSetId);
    if (!activeEvidence) pushLayer('viewer');
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

  const closeEvidenceNow = useCallback(() => {
    setActiveEvidence(null);
    if (fromSearch) setActiveTab('search');
    // Keep activeSearchResultSetId so results are still there on return.
  }, [fromSearch]);

  const handleCloseEvidence = () => closeLayer('viewer', closeEvidenceNow);

  // The back gesture (or button) pops whatever we pushed last.
  useEffect(() => {
    const onPop = () => {
      const top = layers.current.pop();
      if (top === 'viewer') closeEvidenceNow();
      else if (top === 'drawer') setSidebarOpen(false);
    };
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, [closeEvidenceNow]);

  const view: 'chat' | 'search' | 'viewer' = showingEvidence ? 'viewer' : activeTab;

  return (
    <div
      className="app"
      data-sidebar={sidebarOpen ? 'open' : 'closed'}
      data-layout={layout.mode}
      data-touch={layout.touch ? 'true' : 'false'}
      data-view={view}
    >
      <AppHeader
        user={user}
        onLogout={() => setUser(null)}
        sidebarOpen={sidebarOpen}
        onToggleSidebar={toggleSidebar}
        onNewSession={handleNewSession}
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
          onClick={closeSidebar}
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

      {/* Phone: the three places you can be, always one tap away. Hidden while a
          document is open so the page gets the whole screen. */}
      {layout.isPhone && view !== 'viewer' && (
        <nav className="bottom-nav" aria-label="Sections">
          <button
            type="button"
            className="bottom-nav-item"
            aria-current={view === 'chat'}
            onClick={() => setActiveTab('chat')}
          >
            <Icon name="message" size={20} />
            <span>Chat</span>
          </button>
          <button
            type="button"
            className="bottom-nav-item"
            aria-current={view === 'search'}
            onClick={() => setActiveTab('search')}
          >
            <Icon name="search" size={20} />
            <span>Search</span>
          </button>
          <button
            type="button"
            className="bottom-nav-item"
            aria-current={sidebarOpen}
            onClick={toggleSidebar}
          >
            <Icon name="menu" size={20} />
            <span>Sessions</span>
          </button>
        </nav>
      )}

      <Toaster />
    </div>
  );
}
