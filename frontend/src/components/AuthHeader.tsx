'use client';

import { useEffect, useState } from 'react';
import { api, getLoginUrl, logout, type AuthUser } from '@/lib/api';
import type { CollectionNode, DocumentNode } from '@/types/api';

interface AuthHeaderProps {
  user: AuthUser | null;
  onLogout?: () => void;
}

export function AuthHeader({ user, onLogout }: AuthHeaderProps) {
  const [showAbout, setShowAbout] = useState(false);

  const handleLogout = async () => {
    await logout();
    onLogout?.();
  };

  return (
    <header className="global-header">
      <a href="/" className="global-header-brand" title="Return to home page">
        Friday
      </a>

      <div className="global-header-actions">
        <button
          type="button"
          className="header-about-btn"
          onClick={() => setShowAbout(true)}
        >
          About
        </button>
        {user ? (
          <>
            <span className="auth-user">
              {user.email || user.sub}
            </span>
            <button
              type="button"
              onClick={handleLogout}
              className="auth-signout"
            >
              Sign out
            </button>
          </>
        ) : (
          <a href={getLoginUrl()} className="auth-signin">
            Sign in
          </a>
        )}
      </div>

      {showAbout && <AboutModal onClose={() => setShowAbout(false)} />}
    </header>
  );
}

function AboutModal({ onClose }: { onClose: () => void }) {
  const [collections, setCollections] = useState<CollectionNode[] | null>(null);
  const [collectionsError, setCollectionsError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [docsByCollection, setDocsByCollection] = useState<Record<number, DocumentNode[] | 'loading' | 'error'>>({});

  // Load the collection list when the modal opens
  useEffect(() => {
    let cancelled = false;
    api.getCollectionsTree(true)
      .then((cols) => { if (!cancelled) setCollections(cols); })
      .catch((e) => { if (!cancelled) setCollectionsError(e instanceof Error ? e.message : 'Failed to load collections'); });
    return () => { cancelled = true; };
  }, []);

  const toggleCollection = (colId: number) => {
    const next = expandedId === colId ? null : colId;
    setExpandedId(next);
    if (next != null && docsByCollection[next] === undefined) {
      setDocsByCollection((prev) => ({ ...prev, [next]: 'loading' }));
      api.getCollectionDocuments(next)
        .then((docs) => setDocsByCollection((prev) => ({ ...prev, [next]: docs })))
        .catch(() => setDocsByCollection((prev) => ({ ...prev, [next]: 'error' })));
    }
  };

  return (
    <div className="about-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-label="About Friday">
      <div className="about-card" onClick={(e) => e.stopPropagation()}>
        <div className="about-card-header">
          <h2>About Friday</h2>
          <button
            type="button"
            className="about-close-btn"
            onClick={onClose}
            aria-label="Close"
          >
            ✕
          </button>
        </div>
        <div className="about-card-body">
          <h3>What is Friday?</h3>
          {/* Placeholder copy — replace with real description */}
          <p>
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do
            eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut
            enim ad minim veniam, quis nostrud exercitation ullamco laboris
            nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor
            in reprehenderit in voluptate velit esse cillum dolore eu fugiat
            nulla pariatur.
          </p>
          <p>
            Excepteur sint occaecat cupidatat non proident, sunt in culpa
            qui officia deserunt mollit anim id est laborum. Sed ut
            perspiciatis unde omnis iste natus error sit voluptatem
            accusantium doloremque laudantium.
          </p>
          <h3>Funding</h3>
          {/* Placeholder copy — replace with real funding acknowledgement */}
          <p>
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nemo
            enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut
            fugit, sed quia consequuntur magni dolores eos qui ratione
            voluptatem sequi nesciunt.
          </p>
          <h3>Collections</h3>
          <p className="about-collections-hint">
            Click a collection to see what it consists of, its files, and its source.
          </p>
          {collectionsError && <div className="search-error">{collectionsError}</div>}
          {!collections && !collectionsError && <div className="loading">Loading collections…</div>}
          {collections?.map((col) => {
            const docs = docsByCollection[col.id];
            const isOpen = expandedId === col.id;
            return (
              <div key={col.id} className="about-collection">
                <button
                  type="button"
                  className="about-collection-row"
                  onClick={() => toggleCollection(col.id)}
                  aria-expanded={isOpen}
                >
                  <span className="about-collection-chevron">{isOpen ? '▼' : '▶'}</span>
                  <span className="about-collection-name">{col.title || col.slug}</span>
                  {col.document_count != null && (
                    <span className="about-collection-count">
                      {col.document_count} document{col.document_count === 1 ? '' : 's'}
                    </span>
                  )}
                </button>
                {isOpen && (
                  <div className="about-collection-detail">
                    {/* Placeholder copy — replace with the real collection description */}
                    <p>
                      Lorem ipsum dolor sit amet, consectetur adipiscing elit —
                      quis nostrud exercitation ullamco laboris nisi ut aliquip
                      ex ea commodo consequat.
                    </p>
                    <p className="about-collection-source">
                      {/* Placeholder — replace with the real source/provenance */}
                      <strong>Source:</strong> Lorem ipsum dolor sit amet, consectetur adipiscing elit.
                    </p>
                    <div className="about-collection-files-label">Files</div>
                    {docs === 'loading' && <div className="loading">Loading files…</div>}
                    {docs === 'error' && <div className="search-error">Failed to load files.</div>}
                    {Array.isArray(docs) && (
                      docs.length > 0 ? (
                        <ul className="about-collection-files">
                          {docs.map((d) => (
                            <li key={d.id} title={d.source_ref || d.source_name}>
                              {d.source_name || d.source_ref || `Document #${d.id}`}
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <div className="about-collection-files-empty">No documents.</div>
                      )
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
