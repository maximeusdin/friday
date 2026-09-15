/**
 * Scope comparison utilities.
 *
 * Provides stable fingerprinting for UserSelectedScope objects so that
 * dirty detection, active-vs-lastUsed checks, and diff computations
 * are not affected by array ordering or duplicate IDs.
 */
import type { UserSelectedScope } from '@/types/api';

/** Deduplicate and sort a number array for stable comparison. */
const uniqSort = (xs: number[] = []): number[] =>
  Array.from(new Set(xs)).sort((a, b) => a - b);

/** Normalize a scope object: ensure mode is present, dedupe+sort ID arrays. */
export function normalizeScope(scope: UserSelectedScope): UserSelectedScope {
  return {
    mode: scope.mode,
    included_collection_ids: uniqSort(scope.included_collection_ids),
    included_document_ids: uniqSort(scope.included_document_ids),
  };
}

/** Deterministic JSON string for a scope — used as equality check. */
export function scopeFingerprint(scope: UserSelectedScope): string {
  return JSON.stringify(normalizeScope(scope));
}

/** Compare two scopes for logical equality (null-safe). */
export function scopesEqual(
  a: UserSelectedScope | null | undefined,
  b: UserSelectedScope | null | undefined,
): boolean {
  if (!a && !b) return true;
  if (!a || !b) return false;
  return scopeFingerprint(a) === scopeFingerprint(b);
}

/** Total documents a scope covers, when the collection tree is known. */
export function scopeDocumentCount(
  scope: UserSelectedScope,
  collections: { id: number; document_count: number }[],
): number | null {
  if (scope.mode === 'full_archive') {
    return collections.reduce((n, c) => n + (c.document_count || 0), 0);
  }
  const ids = new Set(scope.included_collection_ids || []);
  const fromCollections = collections
    .filter((c) => ids.has(c.id))
    .reduce((n, c) => n + (c.document_count || 0), 0);
  return fromCollections + (scope.included_document_ids?.length || 0);
}

/** Short human label for a scope — "Entire archive", a collection name, or a count. */
export function describeScope(
  scope: UserSelectedScope | null | undefined,
  collections: { id: number; title?: string; slug?: string }[],
): string {
  if (!scope || scope.mode === 'full_archive') return 'Entire archive';
  const cols = scope.included_collection_ids || [];
  const docs = scope.included_document_ids || [];
  if (cols.length === 0 && docs.length === 0) return 'No sources selected';
  const parts: string[] = [];
  if (cols.length === 1) {
    const c = collections.find((x) => x.id === cols[0]);
    parts.push(c ? (c.title || c.slug || `Collection ${cols[0]}`) : `1 collection`);
  } else if (cols.length > 1) {
    parts.push(`${cols.length} collections`);
  }
  if (docs.length > 0) parts.push(`${docs.length} document${docs.length === 1 ? '' : 's'}`);
  return parts.join(' · ');
}
