/**
 * Client-side bulk download of archive PDFs.
 *
 * Selected PDFs are fetched from S3/CloudFront (same-origin in prod; via the
 * API's /documents/{id}/pdf route in dev) and streamed into a single zip with
 * client-zip. Where the File System Access API exists (Chrome/Edge) the zip is
 * streamed straight to disk, so selection size is unbounded; elsewhere it is
 * assembled as an in-memory Blob, capped to keep the tab alive.
 *
 * CloudFront serves the SPA's index.html with HTTP 200 for ANY missing object,
 * so every fetched file is content-type checked — otherwise a missing PDF
 * would silently become HTML bytes inside the zip.
 */
import { downloadZip } from 'client-zip';

export interface BulkDownloadItem {
  docId: number;
  /** Filename to use inside the zip (documents.source_name). */
  name: string;
  url: string;
  sizeBytes?: number | null;
  /** Slug of the collection the document belongs to (names the zip). */
  collectionSlug?: string;
}

export interface BulkProgress {
  filesDone: number;
  filesTotal: number;
  bytesDone: number;
  /** Sum of known sizes; files with unknown size contribute 0. */
  bytesTotal: number;
  currentFile: string | null;
  /** Files skipped because they could not be fetched (name + reason). */
  skipped: { name: string; reason: string }[];
}

export type ProgressCallback = (p: BulkProgress) => void;

/** In-memory zip assembly cap for browsers without the File System Access API. */
export const BLOB_LIMIT_BYTES = 800_000_000;

/**
 * Only a single file downloads without zipping: browsers gate multiple
 * programmatic downloads behind a permission prompt and later a.click()s lose
 * user activation, with no way to detect the drop — so 2+ files always zip.
 */
export const INDIVIDUAL_DOWNLOAD_MAX = 1;

interface SaveFilePickerHandle {
  createWritable(): Promise<WritableStream<Uint8Array>>;
}

declare global {
  interface Window {
    showSaveFilePicker?: (options?: {
      suggestedName?: string;
      types?: { description?: string; accept: Record<string, string[]> }[];
    }) => Promise<SaveFilePickerHandle>;
  }
}

export function supportsStreamingSave(): boolean {
  return typeof window !== 'undefined' && typeof window.showSaveFilePicker === 'function';
}

export function totalKnownBytes(items: { sizeBytes?: number | null }[]): number {
  return items.reduce((sum, it) => sum + (it.sizeBytes ?? 0), 0);
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null) return '—';
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(bytes >= 1e10 ? 0 : 1)} GB`;
  if (bytes >= 1e6) return `${Math.round(bytes / 1e6)} MB`;
  if (bytes >= 1e3) return `${Math.round(bytes / 1e3)} KB`;
  return `${bytes} B`;
}

/** Mirror of the zip-builder's entry naming: sanitize, then dedupe. */
function zipEntryNames(items: BulkDownloadItem[]): string[] {
  const used = new Set<string>();
  return items.map((it) => {
    let name = (it.name || `document-${it.docId}.pdf`).replace(/[\\/]/g, '_').replace(/^\.+/, '').trim()
      || `document-${it.docId}.pdf`;
    if (used.has(name)) {
      const dot = name.lastIndexOf('.');
      const stem = dot > 0 ? name.slice(0, dot) : name;
      const ext = dot > 0 ? name.slice(dot) : '';
      let i = 2;
      while (used.has(`${stem} (${i})${ext}`)) i++;
      name = `${stem} (${i})${ext}`;
    }
    used.add(name);
    return name;
  });
}

async function fetchPdf(item: BulkDownloadItem, signal: AbortSignal): Promise<Response> {
  const resp = await fetch(item.url, { signal, credentials: 'omit' });
  if (!resp.ok) {
    throw new Error(`"${item.name}": server returned HTTP ${resp.status}`);
  }
  const contentType = (resp.headers.get('content-type') || '').toLowerCase();
  if (contentType.includes('text/html')) {
    // CloudFront SPA fallback: the object doesn't exist on S3.
    throw new Error(`"${item.name}" is not available in the archive's file store`);
  }
  return resp;
}

function countingStream(
  body: ReadableStream<Uint8Array>,
  onBytes: (n: number) => void
): ReadableStream<Uint8Array> {
  return body.pipeThrough(
    new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, controller) {
        onBytes(chunk.byteLength);
        controller.enqueue(chunk);
      },
    })
  );
}

function makeZipSource(
  items: BulkDownloadItem[],
  signal: AbortSignal,
  onProgress: ProgressCallback,
  options: {
    /** Abort once observed bytes exceed this (in-memory Blob assembly cap). */
    limitBytes?: number;
    /** Collector: files skipped because they could not be fetched. */
    skipped: { name: string; reason: string }[];
  }
): AsyncGenerator<{ name: string; input: Response | Blob }> {
  const names = zipEntryNames(items);
  const bytesTotal = totalKnownBytes(items);
  const { limitBytes, skipped } = options;
  let bytesDone = 0;
  let filesDone = 0;

  const report = (currentFile: string | null) =>
    onProgress({ filesDone, filesTotal: items.length, bytesDone, bytesTotal, currentFile, skipped });

  const overLimit = () =>
    new Error(
      `The selection turned out larger than this browser can zip in memory (over ${formatBytes(limitBytes!)}). ` +
        'Use the whole-collection zip, select fewer files at a time, or use a Chromium browser ' +
        '(Chrome/Edge), which can stream the zip to disk.'
    );

  return (async function* () {
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      report(item.name);
      let resp: Response;
      try {
        resp = await fetchPdf(item, signal);
      } catch (e) {
        // Abort/cancel ends the whole download; a merely-missing or failing
        // file is skipped so one bad object can't sink the rest.
        if (e instanceof DOMException && e.name === 'AbortError') throw e;
        skipped.push({ name: item.name, reason: e instanceof Error ? e.message : 'fetch failed' });
        report(null);
        continue;
      }
      if (resp.body) {
        const counted = countingStream(resp.body, (n) => {
          bytesDone += n;
          if (limitBytes != null && bytesDone > limitBytes) throw overLimit();
          report(item.name);
        });
        yield { name: names[i], input: new Response(counted) };
      } else {
        const blob = await resp.blob();
        bytesDone += blob.size;
        if (limitBytes != null && bytesDone > limitBytes) throw overLimit();
        yield { name: names[i], input: blob };
      }
      filesDone++;
      report(i + 1 < items.length ? items[i + 1].name : null);
    }
    if (skipped.length === items.length) {
      throw new Error('None of the selected files could be downloaded.');
    }
    if (skipped.length > 0) {
      let manifestName = '_SKIPPED FILES.txt';
      while (names.includes(manifestName)) manifestName = `_${manifestName}`;
      const text = 'These selected files could not be downloaded:\n\n'
        + skipped.map((s) => `${s.name} — ${s.reason}`).join('\n') + '\n';
      yield { name: manifestName, input: new Blob([text], { type: 'text/plain' }) };
    }
  })();
}

function triggerBlobDownload(blob: Blob, filename: string) {
  const a = window.document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  // Generous revoke delay: revoking too early can abort an in-flight save of a
  // large blob in some browsers. The click has either started the save (the
  // browser holds its own reference) or been blocked well within a minute.
  setTimeout(() => URL.revokeObjectURL(a.href), 60_000);
}

/**
 * Download the given files as a single zip named `zipName`.
 * Resolves (with any skipped files) when the zip has been handed to the
 * browser; rejects with DOMException(name="AbortError") on cancel.
 */
export async function downloadFilesAsZip(
  items: BulkDownloadItem[],
  zipName: string,
  onProgress: ProgressCallback,
  signal: AbortSignal
): Promise<{ skipped: { name: string; reason: string }[] }> {
  if (!items.length) return { skipped: [] };
  const skipped: { name: string; reason: string }[] = [];

  if (supportsStreamingSave()) {
    // Ask for the destination first, inside the user-gesture window.
    let handle: SaveFilePickerHandle;
    try {
      handle = await window.showSaveFilePicker!({
        suggestedName: zipName,
        types: [{ description: 'Zip archive', accept: { 'application/zip': ['.zip'] } }],
      });
    } catch {
      // User dismissed the save dialog — treat as cancel.
      throw new DOMException('Save dialog dismissed', 'AbortError');
    }
    const writable = await handle.createWritable();
    const zipResponse = downloadZip(makeZipSource(items, signal, onProgress, { skipped }));
    await zipResponse.body!.pipeTo(writable, { signal });
    return { skipped };
  }

  const bytesTotal = totalKnownBytes(items);
  if (bytesTotal > BLOB_LIMIT_BYTES) {
    throw new Error(
      `This selection is about ${formatBytes(bytesTotal)}, more than this browser can zip in memory. ` +
        'Use the whole-collection zip, select fewer files at a time, or use a Chromium browser ' +
        '(Chrome/Edge), which can stream the zip to disk.'
    );
  }
  // Known sizes may under-count (size_bytes can be null pre-backfill), so the
  // cap is also enforced on observed bytes inside makeZipSource.
  const blob = await downloadZip(
    makeZipSource(items, signal, onProgress, { skipped, limitBytes: BLOB_LIMIT_BYTES })
  ).blob();
  if (signal.aborted) throw new DOMException('Cancelled', 'AbortError');
  triggerBlobDownload(blob, zipName);
  return { skipped };
}

/** Single-file selection: hand the file to the browser as its own download. */
export async function downloadFilesIndividually(
  items: BulkDownloadItem[],
  onProgress: ProgressCallback,
  signal: AbortSignal
): Promise<{ skipped: { name: string; reason: string }[] }> {
  const bytesTotal = totalKnownBytes(items);
  let bytesDone = 0;
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    onProgress({ filesDone: i, filesTotal: items.length, bytesDone, bytesTotal, currentFile: item.name, skipped: [] });
    const resp = await fetchPdf(item, signal);
    const blob = await resp.blob();
    bytesDone += blob.size;
    triggerBlobDownload(blob, item.name);
    onProgress({ filesDone: i + 1, filesTotal: items.length, bytesDone, bytesTotal, currentFile: null, skipped: [] });
  }
  return { skipped: [] };
}
