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
