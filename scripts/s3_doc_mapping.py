"""Map documents rows to S3 object keys under data/raw/.

Shared by build_collection_zips.py and backfill_document_sizes.py.

The bucket mirrors the repo's data/ layout. documents.source_ref is the
canonical key when it carries path info ("data/raw/<slug>/[pdf/]<file>.pdf"),
but historic ingest chains left some refs URL-encoded, absolute, or absent —
so matching falls back to basename lookup. The fallback may NOT be restricted
to the collection's own directory: 160 of 1005 prod documents live in S3 dirs
that don't match their slug (hiss_chambers -> AlgerHissWhittakerChambers/,
jack_childs -> "Jack Childs/", huac_reports -> unamerican_reports/, ...).

The poison case — a doc whose own file is absent from S3 silently basename-
matching a same-named file in another collection — is caught afterwards by
demote_basename_conflicts(): when one S3 object is claimed by several docs,
claims made via source_ref (or in the collection's own directory) win, and
basename-fallback claims are demoted to unmatched with a warning.
"""
from __future__ import annotations

from urllib.parse import unquote


def build_key_indexes(s3_objects: dict[str, int]):
    """s3_objects: {key: size_bytes}. Returns (exact keys set, basename index)."""
    by_basename: dict[str, list[str]] = {}
    for key in s3_objects:
        base = key.rsplit("/", 1)[-1]
        by_basename.setdefault(base, []).append(key)
    return set(s3_objects), by_basename


def _candidate_refs(source_ref: str | None) -> list[str]:
    """Normalize source_ref into candidate data/raw/... keys (exact first, then unquoted)."""
    if not source_ref:
        return []
    ref = str(source_ref).replace("\\", "/")
    idx = ref.lower().find("data/raw/")
    if idx < 0:
        return []
    tail = ref[idx:]
    cands = [tail]
    if "%" in tail:
        cands.append("data/raw/" + unquote(tail[len("data/raw/"):]))
    return cands


def _in_collection_dir(key: str, collection_slug: str) -> bool:
    return f"/{collection_slug.lower()}/" in f"/{key.lower()}"


def match_document_to_key(
    source_name: str,
    source_ref: str | None,
    collection_slug: str,
    keys: set[str],
    by_basename: dict[str, list[str]],
) -> tuple[str, str] | None:
    """Return (key, kind) for a document, or None. kind: 'ref' | 'basename'."""
    for cand in _candidate_refs(source_ref):
        if cand in keys:
            return cand, "ref"

    for base in (source_name, unquote(source_name) if "%" in source_name else None):
        if not base:
            continue
        matches = by_basename.get(base, [])
        if len(matches) == 1:
            return matches[0], "basename"
        if len(matches) > 1:
            in_slug = [k for k in matches if _in_collection_dir(k, collection_slug)]
            if len(in_slug) == 1:
                return in_slug[0], "basename"
    return None


def map_collection_documents(
    docs: list[tuple[int, str, str | None]],
    collection_slug: str,
    s3_objects: dict[str, int],
    keys: set[str],
    by_basename: dict[str, list[str]],
):
    """docs: [(doc_id, source_name, source_ref)].

    Returns (matched: [{doc_id, source_name, key, size, kind, slug}],
             unmatched: [(doc_id, source_name)]).
    """
    matched, unmatched = [], []
    for doc_id, source_name, source_ref in docs:
        hit = match_document_to_key(source_name, source_ref, collection_slug, keys, by_basename)
        if hit is None:
            unmatched.append((doc_id, source_name))
        else:
            key, kind = hit
            matched.append({
                "doc_id": doc_id, "source_name": source_name, "slug": collection_slug,
                "key": key, "size": s3_objects[key], "kind": kind,
            })
    return matched, unmatched


def demote_basename_conflicts(all_matched: list[dict]) -> list[dict]:
    """Given matches across ALL collections, find S3 keys claimed by more than
    one document and demote the weaker claimants (basename-fallback matches
    outside their own collection directory) when a stronger claim exists.

    Mutates nothing; returns the demoted entries (callers should treat them as
    unmatched and warn)."""
    by_key: dict[str, list[dict]] = {}
    for m in all_matched:
        by_key.setdefault(m["key"], []).append(m)

    demoted = []
    for key, claimants in by_key.items():
        if len(claimants) < 2:
            continue
        strong = [
            c for c in claimants
            if c["kind"] == "ref" or _in_collection_dir(key, c["slug"])
        ]
        if strong and len(strong) < len(claimants):
            demoted.extend(c for c in claimants if c not in strong)
    return demoted
