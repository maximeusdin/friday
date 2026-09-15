'use client';

/**
 * Collections & Downloads — the archive browser inside the info modal.
 *
 * Every collection expands into its file list; each file has a checkbox, and
 * the sticky bar at the bottom downloads the selection (zipped client-side —
 * see lib/bulkDownload.ts). Whole collections and the complete archive are
 * one-click links to zips pre-built on S3 by scripts/build_collection_zips.py.
 */
import { useEffect, useRef, useState } from 'react';
import { api } from '@/lib/api';
import type { CollectionNode, CollectionZipsResponse, DocumentNode } from '@/types/api';
import {
  type BulkDownloadItem,
  type BulkProgress,
  INDIVIDUAL_DOWNLOAD_MAX,
  downloadFilesAsZip,
  downloadFilesIndividually,
  formatBytes,
  totalKnownBytes,
} from '@/lib/bulkDownload';

type DownloadState =
  | { status: 'idle' }
  | { status: 'running'; progress: BulkProgress; asZip: boolean }
  | { status: 'done'; files: number; asZip: boolean; skipped: { name: string; reason: string }[] }
  | { status: 'error'; message: string };

export function CollectionsDownloadsBody({ initialCollectionId }: { initialCollectionId?: number }) {
  const [collections, setCollections] = useState<CollectionNode[] | null>(null);
  const [collectionsError, setCollectionsError] = useState<string | null>(null);
  const [zips, setZips] = useState<CollectionZipsResponse | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(initialCollectionId ?? null);
  const [docsByCollection, setDocsByCollection] = useState<Record<number, DocumentNode[] | 'loading' | 'error'>>({});
  const [selected, setSelected] = useState<Map<number, BulkDownloadItem>>(new Map());
  const [dl, setDl] = useState<DownloadState>({ status: 'idle' });
  const abortRef = useRef<AbortController | null>(null);
  const lastProgressRender = useRef(0);

  useEffect(() => {
    let cancelled = false;
    api.getCollectionsTree(true)
      .then((cols) => { if (!cancelled) setCollections(cols); })
      .catch((e) => { if (!cancelled) setCollectionsError(e instanceof Error ? e.message : 'Failed to load collections'); });
    // Zip links are progressive enhancement — ignore failures quietly.
    api.getCollectionZips()
      .then((z) => { if (!cancelled) setZips(z); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);

  // Closing the modal unmounts this panel: abort any in-flight download rather
  // than orphaning a transfer nothing can cancel anymore.
  useEffect(() => () => { abortRef.current?.abort(); }, []);

  const loadDocs = (colId: number) => {
    if (docsByCollection[colId] !== undefined) return;
    setDocsByCollection((prev) => ({ ...prev, [colId]: 'loading' }));
    api.getCollectionDocuments(colId)
      .then((docs) => setDocsByCollection((prev) => ({ ...prev, [colId]: docs })))
      .catch(() => setDocsByCollection((prev) => ({ ...prev, [colId]: 'error' })));
  };

  useEffect(() => {
    if (initialCollectionId != null) loadDocs(initialCollectionId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialCollectionId]);

  const toggleCollection = (colId: number) => {
    const next = expandedId === colId ? null : colId;
    setExpandedId(next);
    if (next != null) loadDocs(next);
  };

  const zipBySlug = new Map((zips?.collections ?? []).map((z) => [z.slug, z]));

  const itemForDoc = (col: CollectionNode, d: DocumentNode): BulkDownloadItem => ({
    docId: d.id,
    name: d.source_name || `document-${d.id}.pdf`,
    url: api.resolveDocumentPdfUrl(d),
    sizeBytes: d.size_bytes,
    collectionSlug: col.slug,
  });

  const toggleDoc = (col: CollectionNode, d: DocumentNode) => {
    setSelected((prev) => {
      const next = new Map(prev);
      if (next.has(d.id)) next.delete(d.id);
      else next.set(d.id, itemForDoc(col, d));
      return next;
    });
  };

  const setAllInCollection = (col: CollectionNode, docs: DocumentNode[], checked: boolean) => {
    setSelected((prev) => {
      const next = new Map(prev);
      for (const d of docs) {
        if (checked) next.set(d.id, itemForDoc(col, d));
        else next.delete(d.id);
      }
      return next;
    });
  };

  const busy = dl.status === 'running';
  const items = Array.from(selected.values());
  const knownBytes = totalKnownBytes(items);
  const unknownCount = items.filter((it) => it.sizeBytes == null).length;

  const onProgress = (p: BulkProgress) => {
    const now = Date.now();
    // Chunk-level events arrive constantly; cap re-renders.
    if (p.filesDone < p.filesTotal && now - lastProgressRender.current < 150) return;
    lastProgressRender.current = now;
    setDl({ status: 'running', progress: p, asZip: p.filesTotal > INDIVIDUAL_DOWNLOAD_MAX });
  };

  const startDownload = async () => {
    if (!items.length || busy) return;
    const asZip = items.length > INDIVIDUAL_DOWNLOAD_MAX;
    const slugs = new Set(items.map((it) => it.collectionSlug).filter(Boolean));
    const zipName = slugs.size === 1 ? `friday_${[...slugs][0]}_selection.zip` : 'friday_selection.zip';
    const controller = new AbortController();
    abortRef.current = controller;
    setDl({
      status: 'running',
      asZip,
      progress: {
        filesDone: 0, filesTotal: items.length, bytesDone: 0, bytesTotal: knownBytes,
        currentFile: null, skipped: [],
      },
    });
    try {
      const result = asZip
        ? await downloadFilesAsZip(items, zipName, onProgress, controller.signal)
        : await downloadFilesIndividually(items, onProgress, controller.signal);
      setDl({ status: 'done', files: items.length - result.skipped.length, asZip, skipped: result.skipped });
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') {
        setDl({ status: 'idle' });
      } else {
        setDl({ status: 'error', message: e instanceof Error ? e.message : 'Download failed' });
      }
    } finally {
      abortRef.current = null;
    }
  };

  const cancelDownload = () => abortRef.current?.abort();

  return (
    <>
      <p className="about-collections-hint">
        Click a collection to browse its files. Download a single file, tick a set of files and
        use <strong>Download selected</strong> (they arrive as one zip), or grab an entire
        collection with its <strong>↓ zip</strong> link.
      </p>

      {zips?.complete && (
        <div className="dl-complete-row">
          <span>
            <strong>Entire archive</strong>
            <span className="about-collection-count"> · {zips.complete.num_files} files</span>
          </span>
          <a className="col-zip-link" href={api.resolveArchiveAssetUrl(zips.complete.url)}>
            ↓ Download all · {formatBytes(zips.complete.zip_bytes)}
          </a>
        </div>
      )}

      {collectionsError && <div className="search-error">{collectionsError}</div>}
      {!collections && !collectionsError && <div className="loading">Loading collections…</div>}

      {collections?.map((col) => {
        const docs = docsByCollection[col.id];
        const isOpen = expandedId === col.id;
        const zip = zipBySlug.get(col.slug);
        const docList = Array.isArray(docs) ? docs : null;
        const allChecked = !!docList && docList.length > 0 && docList.every((d) => selected.has(d.id));
        return (
          <div key={col.id} className="about-collection">
            <div className="about-collection-header">
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
                    {col.document_count} file{col.document_count === 1 ? '' : 's'}
                  </span>
                )}
              </button>
              {zip && (
                <a
                  className="col-zip-link"
                  href={api.resolveArchiveAssetUrl(zip.url)}
                  title={`Download all ${zip.num_files} files of this collection as one zip`}
                >
                  ↓ {formatBytes(zip.zip_bytes)}
                </a>
              )}
            </div>
            {isOpen && (
              <div className="about-collection-detail">
                <div className="dl-files-header">
                  <span className="about-collection-files-label">Files</span>
                  {docList && docList.length > 0 && (
                    <label className="dl-select-all">
                      <input
                        type="checkbox"
                        checked={allChecked}
                        onChange={(e) => setAllInCollection(col, docList, e.target.checked)}
                        disabled={busy}
                      />
                      Select all
                    </label>
                  )}
                </div>
                {docs === 'loading' && <div className="loading">Loading files…</div>}
                {docs === 'error' && <div className="search-error">Failed to load files.</div>}
                {docList && (
                  docList.length > 0 ? (
                    <div className="about-collection-files-scroll">
                      <table className="about-collection-files-table">
                        <thead>
                          <tr>
                            <th className="files-table-check" />
                            <th>#</th>
                            <th>File</th>
                            <th className="files-table-size">Size</th>
                          </tr>
                        </thead>
                        <tbody>
                          {docList.map((d, i) => (
                            <tr
                              key={d.id}
                              className={selected.has(d.id) ? 'dl-row-selected' : undefined}
                              onClick={() => { if (!busy) toggleDoc(col, d); }}
                              style={{ cursor: busy ? undefined : 'pointer' }}
                            >
                              <td className="files-table-check">
                                <input
                                  type="checkbox"
                                  checked={selected.has(d.id)}
                                  onChange={() => toggleDoc(col, d)}
                                  onClick={(e) => e.stopPropagation()}
                                  disabled={busy}
                                  aria-label={`Select ${d.source_name}`}
                                />
                              </td>
                              <td className="files-table-num">{i + 1}</td>
                              <td>{d.source_name || `Document #${d.id}`}</td>
                              <td className="files-table-size">{formatBytes(d.size_bytes)}</td>
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

      {(selected.size > 0 || dl.status !== 'idle') && (
        <div className="dl-bar">
          {dl.status === 'running' ? (
            <>
              <div className="dl-bar-status">
                <div className="dl-progress-track">
                  <div
                    className="dl-progress-fill"
                    style={{
                      width: dl.progress.bytesTotal > 0
                        ? `${Math.min(100, (dl.progress.bytesDone / dl.progress.bytesTotal) * 100)}%`
                        : `${(dl.progress.filesDone / Math.max(1, dl.progress.filesTotal)) * 100}%`,
                    }}
                  />
                </div>
                <span className="dl-bar-text">
                  {dl.progress.filesDone}/{dl.progress.filesTotal} files
                  {dl.progress.bytesTotal > 0 && (
                    <> · {formatBytes(dl.progress.bytesDone)} of {formatBytes(dl.progress.bytesTotal)}</>
                  )}
                  {dl.progress.currentFile && <> · {dl.progress.currentFile}</>}
                </span>
              </div>
              <button type="button" className="dl-bar-btn" onClick={cancelDownload}>Cancel</button>
            </>
          ) : (
            <>
              <span className="dl-bar-text">
                {dl.status === 'error' ? (
                  <span className="dl-bar-error">{dl.message}</span>
                ) : dl.status === 'done' ? (
                  dl.skipped.length > 0 ? (
                    <span className="dl-bar-error" title={dl.skipped.map((s) => `${s.name}: ${s.reason}`).join('\n')}>
                      Downloaded {dl.files} of {dl.files + dl.skipped.length} files — {dl.skipped.length} skipped
                      (listed in the zip&apos;s _SKIPPED FILES.txt)
                    </span>
                  ) : (
                    <>Downloaded {dl.files} file{dl.files === 1 ? '' : 's'}{dl.asZip ? ' as a zip' : ''}.</>
                  )
                ) : (
                  <>
                    {selected.size} file{selected.size === 1 ? '' : 's'} selected
                    {knownBytes > 0 && <> · {unknownCount > 0 ? '≥' : ''}{formatBytes(knownBytes)}</>}
                  </>
                )}
              </span>
              {selected.size > 0 && (
                <>
                  <button type="button" className="dl-bar-btn dl-bar-btn-primary" onClick={startDownload}>
                    Download selected
                  </button>
                  <button
                    type="button"
                    className="dl-bar-btn"
                    onClick={() => { setSelected(new Map()); setDl({ status: 'idle' }); }}
                  >
                    Clear
                  </button>
                </>
              )}
            </>
          )}
        </div>
      )}
    </>
  );
}
