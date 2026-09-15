'use client';

/**
 * Document-name resolution.
 *
 * Evidence bullets and citations carry a document id, but not always a name:
 * `source_names` is populated for some findings and empty for others, which is
 * why sources rendered as "View document" in some rows and as the real filename
 * in the next. The name is one cheap lookup away, so resolve it and cache it
 * process-wide — a document's name never changes within a session.
 */

import { useEffect, useState } from 'react';
import { api } from './api';

const names = new Map<number, string>();
const inflight = new Map<number, Promise<string | null>>();
const listeners = new Set<() => void>();

function resolve(id: number): Promise<string | null> {
  const cached = names.get(id);
  if (cached) return Promise.resolve(cached);

  const existing = inflight.get(id);
  if (existing) return existing;

  const p = api.getDocument(id)
    .then((doc) => {
      const name = doc?.source_name || null;
      if (name) {
        names.set(id, name);
        for (const l of listeners) l();
      }
      return name;
    })
    .catch(() => null)
    .finally(() => { inflight.delete(id); });

  inflight.set(id, p);
  return p;
}

/** Synchronously known name, if it has already been fetched. */
export function knownDocumentName(id?: number | null): string | undefined {
  return id == null ? undefined : names.get(id);
}

/**
 * The document's name, fetching it once if needed.
 * `given` short-circuits the lookup when the payload already carried a name.
 */
export function useDocumentName(id?: number | null, given?: string | null): string | undefined {
  const [, force] = useState(0);
  const has = given || (id != null ? names.get(id) : undefined);

  useEffect(() => {
    if (given || id == null || names.has(id)) return;
    let alive = true;
    const rerender = () => { if (alive) force((n) => n + 1); };
    listeners.add(rerender);
    void resolve(id);
    return () => { alive = false; listeners.delete(rerender); };
  }, [id, given]);

  return has;
}
