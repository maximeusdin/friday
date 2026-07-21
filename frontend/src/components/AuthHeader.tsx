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
          <details className="about-section">
            <summary>How to use Friday — a researcher&apos;s tutorial</summary>
            <div className="about-section-body">
              <p>
                Friday searches digitized, OCR-processed declassified archives and answers with
                citations to the actual pages. It finds and reads documents — it does not add
                outside knowledge, and it will tell you when the documents don&apos;t answer the question.
              </p>
              <p><strong>Asking questions that maximize returns:</strong></p>
              <ul>
                <li><strong>Name names.</strong> Include the people, organizations, and places you care
                  about. Friday automatically expands cover names and aliases (asking about Golos also
                  finds &ldquo;Sound&rdquo;; asking &ldquo;Who is Jurist?&rdquo; resolves the codename), but it can only
                  do that when a name is in the question.</li>
                <li><strong>Include distinctive details.</strong> Rare numbers, dates, and quoted phrases
                  are gold: &ldquo;500-600 meetings&rdquo; or a distinctive phrase from a document pins the search
                  to the exact passage.</li>
                <li><strong>Direct questions are fine.</strong> Friday rewrites questions like &ldquo;When was
                  X recruited?&rdquo; into the archive&apos;s own record language (&ldquo;initial contact&rdquo;,
                  &ldquo;memorandum&rdquo;, &ldquo;informant&rdquo;) behind the scenes, because FBI files rarely use everyday
                  words. Phrasing a question around records (&ldquo;what records exist about&hellip;&rdquo;) works well too.</li>
                <li><strong>Lists and counts enumerate the whole archive.</strong> &ldquo;Which journalists
                  were recruited?&rdquo; or &ldquo;How many engineers&hellip;?&rdquo; triggers coverage-first retrieval across
                  every collection, so members spread over multiple files aren&apos;t missed.</li>
                <li><strong>Scope narrows the hunt.</strong> Use the Scope panel (right side) to restrict
                  to specific collections or documents — it applies to both Chat and Search — or just
                  say it in the question (&ldquo;&hellip;in the Vassiliev notebooks&rdquo;).</li>
                <li><strong>Click the citations.</strong> Evidence links open the document at the right
                  page with the supporting passage highlighted. The quoted passage shown is taken
                  verbatim from the document, never paraphrased.</li>
              </ul>
              <p><strong>Reading the answers:</strong> findings marked with a source are grounded in a
                cited page. A summary labeled <em>unverified</em> means its claims didn&apos;t pass citation
                checks — treat it as a lead, and click through to the sources. When Friday says it
                could not find evidence, that is a statement about the search, not proof the fact
                isn&apos;t somewhere in the archive: rephrase with more specific names or details, or use
                <em> Think Deeper</em> to extend the investigation.</p>
              <p><strong>Default settings:</strong> full-archive scope; deep search effort on every
                query (Think Deeper extends even further, reusing the evidence already gathered);
                alias/codename expansion on; record-language rewriting on. In the Search tab, exact
                matching is the default — Fuzzy (for OCR errors and typos) and Aliases are toggles
                next to the search box.</p>
              <p><strong>What Friday doesn&apos;t do:</strong> it can&apos;t read text the OCR mangled beyond
                recognition (try Fuzzy in Search for near-miss spellings); it won&apos;t speculate beyond
                the documents; and it doesn&apos;t (yet) search outside the indexed collections listed below.</p>
            </div>
          </details>

          <details className="about-section">
            <summary>Chat vs. Search — which to use when</summary>
            <div className="about-section-body">
              <p>
                <strong>Search</strong> is a deterministic concordance: it matches your terms —
                exact, boolean (<code>AND</code>/<code>OR</code>/<code>NOT</code>, quoted phrases), or
                fuzzy — against every page and returns numbered page hits you can open, prune, and
                export as CSV. It never interprets your query. Use Search when you know words that
                actually appear on the page, when you want <em>every</em> occurrence (not a summary),
                or when you&apos;re building a citation list. Each search becomes a tab in your session,
                and numbering lets you stop at hit #40 and resume next week.
              </p>
              <p>
                <strong>Chat</strong> is an investigative assistant: it plans the question, runs many
                searches (semantic and exact, in your words and the archive&apos;s), resolves codenames,
                reads the retrieved pages, and writes an answer with citations. Use Chat when you have
                a question rather than a term, when your wording may not match the documents&apos;
                vocabulary, or when the answer must be assembled from several documents.
              </p>
              <p>
                They share the same session and the same Scope panel — and they meet in the middle:
                Chat runs the same boolean search engine you do, and every search it runs is saved
                into your session under <strong>⚡ Chat&apos;s searches</strong> in the Search tab.
                Open one to see exactly what Chat looked at, prune its results, or continue the
                investigation yourself where it left off.
              </p>
              <p>
                A rule of thumb: <em>Search finds pages; Chat answers questions.</em> Researchers
                often begin in Chat, then switch to Search to exhaustively walk the pages behind
                an answer.
              </p>
            </div>
          </details>

          <h3>Funding</h3>
          {/* Placeholder copy — replace with real funding acknowledgement */}
          <p>
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nemo
            enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut
            fugit, sed quia consequuntur magni dolores eos qui ratione
            voluptatem sequi nesciunt.
          </p>
          <h3>Origins of Friday</h3>
          {/* Placeholder copy — replace with the real origin story: who conceived and
              built Friday (its architects), the institutions involved, and how the
              project is funded. */}
          <p>
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed ut
            perspiciatis unde omnis iste natus error sit voluptatem accusantium
            doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo
            inventore veritatis et quasi architecto beatae vitae dicta sunt
            explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur.
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
                        <div className="about-collection-files-scroll">
                          <table className="about-collection-files-table">
                            <thead>
                              <tr>
                                <th>#</th>
                                <th>File</th>
                                <th>Source reference</th>
                              </tr>
                            </thead>
                            <tbody>
                              {docs.map((d, i) => (
                                <tr key={d.id}>
                                  <td className="files-table-num">{i + 1}</td>
                                  <td>{d.source_name || `Document #${d.id}`}</td>
                                  <td className="files-table-ref">{d.source_ref || '—'}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
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
