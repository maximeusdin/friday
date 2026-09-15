#!/usr/bin/env python3
"""Build per-collection zip archives of Friday's PDFs and publish them to S3.

For each collection in the DB, streams its PDFs from s3://<bucket>/data/raw/,
packs them into data/zips/<slug>.zip (ZIP_STORED — scanned PDFs don't
compress), uploads the zip back to the bucket, and maintains
data/zips/manifest.json describing every zip (the backend /api/collection_zips
endpoint serves that manifest to the frontend).

Collections whose current file set (count + total bytes) matches the published
manifest are skipped, so re-runs after an ingest only rebuild what changed.

Run (Mac):
  AWS_SHARED_CREDENTIALS_FILE=/Users/maxime/friday/.aws/credentials \
  DATABASE_URL=<dsn> python scripts/build_collection_zips.py \
    [--collections slug1,slug2] [--complete] [--skip-upload] [--force] [--invalidate]

The zips land under the bucket's data/ prefix, which frontend/deploy.sh's
`aws s3 sync --delete --exclude "data/*"` never touches.
"""
import argparse
import datetime
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import boto3
import psycopg2
from botocore.exceptions import ClientError

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from s3_doc_mapping import (
    build_key_indexes,
    demote_basename_conflicts,
    map_collection_documents,
)


def map_all_collections(collections: dict, s3_objects: dict, keys, by_basename):
    """Map every collection's documents to S3 keys, then demote basename-fallback
    matches whose S3 object is claimed more strongly by another document.

    Returns (matched_by_slug: {slug: [match dict]}, unmatched: [(slug, doc_id, name)]).
    """
    matched_by_slug = {}
    unmatched = []
    all_matched = []
    for slug in sorted(collections):
        matched, missing = map_collection_documents(
            collections[slug], slug, s3_objects, keys, by_basename)
        matched_by_slug[slug] = matched
        all_matched.extend(matched)
        unmatched.extend((slug, d, n) for d, n in missing)

    demoted = demote_basename_conflicts(all_matched)
    for m in demoted:
        print(f"WARNING: {m['slug']} doc={m['doc_id']} {m['source_name']!r} basename-matched "
              f"{m['key']} but another document claims that object — treating as unmatched")
        matched_by_slug[m["slug"]].remove(m)
        unmatched.append((m["slug"], m["doc_id"], m["source_name"]))
    return matched_by_slug, unmatched

ZIPS_PREFIX = "data/zips"
MANIFEST_KEY = f"{ZIPS_PREFIX}/manifest.json"
COMPLETE_SLUG = "friday_complete"


def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL")
    return psycopg2.connect(dsn)


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def list_s3_objects(s3, bucket: str, prefix: str) -> dict:
    objects = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            objects[obj["Key"]] = obj["Size"]
    return objects


def fetch_manifest(s3, bucket: str) -> dict:
    try:
        body = s3.get_object(Bucket=bucket, Key=MANIFEST_KEY)["Body"].read()
        return json.loads(body)
    except (ClientError, json.JSONDecodeError):
        return {}


def zip_entry_name(source_name: str, prefix: str, used: set) -> str:
    """Zip entry from source_name: sanitize the filename, prepend the folder
    prefix (e.g. "silvermaster/" inside the complete zip), dedupe just in case."""
    base = source_name.replace("\\", "_").replace("/", "_").lstrip(".").strip() or "document.pdf"
    name = prefix + base
    if name in used:
        stem, dot, ext = name.rpartition(".")
        root = stem if dot else name
        ext = f".{ext}" if dot else ""
        i = 2
        while f"{root} ({i}){ext}" in used:
            i += 1
        name = f"{root} ({i}){ext}"
    used.add(name)
    return name


def download_to_tmp(s3, bucket: str, key: str, expected_size: int, tmp_path: Path):
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(tmp_path))
    actual = tmp_path.stat().st_size
    if actual != expected_size:
        raise RuntimeError(f"size mismatch for {key}: got {actual}, expected {expected_size}")


def build_zip(s3, bucket: str, zip_path: Path, entries, workdir: Path):
    """entries: [(prefix, source_name, key, size_bytes)] where prefix is "" or
    "slug/". Writes zip_path atomically via .part."""
    part = zip_path.with_suffix(zip_path.suffix + ".part")
    tmp = workdir / ".tmp_download"
    used_names = set()
    done_bytes = 0
    total_bytes = sum(e[3] for e in entries)
    with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for i, (prefix, source_name, key, size) in enumerate(entries, 1):
            arcname = zip_entry_name(source_name, prefix, used_names)
            download_to_tmp(s3, bucket, key, size, tmp)
            zf.write(tmp, arcname)
            tmp.unlink()
            done_bytes += size
            print(f"    [{i}/{len(entries)}] {done_bytes/1e6:,.0f}/{total_bytes/1e6:,.0f} MB  {arcname[:60]}",
                  flush=True)
    with zipfile.ZipFile(part) as zf:
        names = zf.namelist()
        if len(names) != len(entries):
            raise RuntimeError(f"zip has {len(names)} entries, expected {len(entries)}")
    part.replace(zip_path)


def _zip_entry_count(zip_path: Path) -> int:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            return len(zf.namelist())
    except (OSError, zipfile.BadZipFile):
        return -1


def _local_zip_matches(zip_path: Path, expected_count: int, expected_bytes: int) -> bool:
    """True if an already-built local zip holds exactly the expected file set
    (entry count + total uncompressed bytes), so a re-run can upload it without
    re-downloading everything from S3."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            infos = zf.infolist()
            return (len(infos) == expected_count
                    and sum(i.file_size for i in infos) == expected_bytes)
    except (OSError, zipfile.BadZipFile):
        return False


def build_complete_from_local_zips(workdir: Path, slugs: list, zip_path: Path):
    """Assemble the complete-archive zip by copying entries out of the local
    per-collection zips (all ZIP_STORED, so this is pure disk I/O)."""
    part = zip_path.with_suffix(zip_path.suffix + ".part")
    with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as out:
        for slug in slugs:
            with zipfile.ZipFile(workdir / f"{slug}.zip") as zin:
                for info in zin.infolist():
                    zi = zipfile.ZipInfo(f"{slug}/{info.filename}", date_time=info.date_time)
                    zi.compress_type = zipfile.ZIP_STORED
                    with zin.open(info) as rf, out.open(zi, "w", force_zip64=True) as wf:
                        shutil.copyfileobj(rf, wf, 1024 * 1024)
            print(f"    + {slug}", flush=True)
    part.replace(zip_path)


def upload_zip(s3, bucket: str, zip_path: Path, key: str):
    s3.upload_file(
        str(zip_path), bucket, key,
        ExtraArgs={"ContentType": "application/zip",
                   "CacheControl": "public, max-age=300"},
    )


def upload_manifest(s3, bucket: str, manifest: dict):
    s3.put_object(
        Bucket=bucket, Key=MANIFEST_KEY,
        Body=json.dumps(manifest, indent=1).encode("utf-8"),
        ContentType="application/json",
        CacheControl="no-cache",
    )


def invalidate_cloudfront(bucket: str, region: str):
    """Best-effort: find the distribution whose alias is the bucket domain, invalidate zips."""
    try:
        cf = boto3.client("cloudfront", region_name=region)
        dists = cf.list_distributions().get("DistributionList", {}).get("Items", []) or []
        dist_id = next(
            (d["Id"] for d in dists
             if bucket in (d.get("Aliases", {}).get("Items", []) or [])),
            None,
        )
        if not dist_id:
            print("  CloudFront: no distribution with that alias found; skipping invalidation")
            return
        cf.create_invalidation(
            DistributionId=dist_id,
            InvalidationBatch={
                "Paths": {"Quantity": 1, "Items": [f"/{ZIPS_PREFIX}/*"]},
                "CallerReference": f"zips-{now_iso()}",
            },
        )
        print(f"  CloudFront: invalidation created on {dist_id} for /{ZIPS_PREFIX}/*")
    except Exception as e:  # noqa: BLE001 — invalidation is best-effort
        print(f"  CloudFront: invalidation skipped ({e})")


def main():
    ap = argparse.ArgumentParser(description="Build and publish per-collection PDF zips")
    ap.add_argument("--bucket", default="fridayarchive.org")
    ap.add_argument("--region", default="us-west-1")
    ap.add_argument("--collections", default="",
                    help="Comma-separated slugs (default: every collection in the DB)")
    ap.add_argument("--complete", action="store_true",
                    help=f"Also build {COMPLETE_SLUG}.zip containing every collection")
    ap.add_argument("--workdir", default=str(Path(_HERE).parent / "data" / "zips"))
    ap.add_argument("--force", action="store_true", help="Rebuild even if unchanged")
    ap.add_argument("--skip-upload", action="store_true", help="Build local zips only")
    ap.add_argument("--invalidate", action="store_true",
                    help="Create a CloudFront invalidation for the zips prefix after upload")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", region_name=args.region)

    print(f"Listing s3://{args.bucket}/data/raw/ ...")
    s3_objects = list_s3_objects(s3, args.bucket, "data/raw/")
    print(f"  {len(s3_objects)} objects, {sum(s3_objects.values())/1e9:.2f} GB")
    keys, by_basename = build_key_indexes(s3_objects)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT c.slug, c.title, d.id, d.source_name, d.source_ref
        FROM documents d JOIN collections c ON c.id = d.collection_id
        ORDER BY c.slug, d.source_name
    """)
    rows = cur.fetchall()
    conn.close()

    collections = {}
    titles = {}
    for slug, title, doc_id, source_name, source_ref in rows:
        titles[slug] = title
        collections.setdefault(slug, []).append((doc_id, source_name, source_ref))

    only = {s.strip() for s in args.collections.split(",") if s.strip()}
    unknown = only - set(collections)
    if unknown:
        print(f"ERROR: unknown collection slugs: {sorted(unknown)}")
        return 1

    prior_manifest = fetch_manifest(s3, args.bucket)
    prior_entries = {e["slug"]: e for e in prior_manifest.get("collections", [])}

    matched_by_slug, all_unmatched = map_all_collections(collections, s3_objects, keys, by_basename)

    new_entries = {}
    built = skipped = 0

    for slug in sorted(collections):
        matched = matched_by_slug[slug]
        n_missing = len(collections[slug]) - len(matched)
        if n_missing:
            print(f"WARNING: {slug}: {n_missing} document(s) not found on S3 — "
                  f"zip will contain {len(matched)} of {len(collections[slug])} files")
        if not matched:
            print(f"  {slug}: nothing on S3, skipping")
            continue

        total_bytes = sum(m["size"] for m in matched)
        entry = {
            "slug": slug,
            "title": titles[slug],
            "num_files": len(matched),
            "total_bytes": total_bytes,
            "zip_key": f"{ZIPS_PREFIX}/{slug}.zip",
        }

        selected = not only or slug in only
        prior = prior_entries.get(slug)
        unchanged = (
            prior is not None
            and prior.get("num_files") == entry["num_files"]
            and prior.get("total_bytes") == entry["total_bytes"]
            and prior.get("zip_bytes")
        )
        if (not selected or unchanged) and not (args.force and selected):
            if prior:
                new_entries[slug] = prior
                skipped += 1
                print(f"  {slug}: unchanged, keeping published zip"
                      if unchanged else f"  {slug}: not selected, keeping prior entry")
            else:
                print(f"  {slug}: not selected and never built — no manifest entry yet")
            continue

        zip_path = workdir / f"{slug}.zip"
        if _local_zip_matches(zip_path, len(matched), total_bytes) and not args.force:
            print(f"  {slug}: local zip already current ({len(matched)} files) — reusing")
        else:
            print(f"  {slug}: building {len(matched)} files, {total_bytes/1e9:.2f} GB")
            build_zip(s3, args.bucket, zip_path,
                      [("", m["source_name"], m["key"], m["size"]) for m in matched], workdir)
        entry["zip_bytes"] = zip_path.stat().st_size
        entry["built_at"] = now_iso()
        if not args.skip_upload:
            upload_zip(s3, args.bucket, zip_path, entry["zip_key"])
            print(f"    uploaded s3://{args.bucket}/{entry['zip_key']}")
        new_entries[slug] = entry
        built += 1

    if args.complete:
        all_matched = [m for slug in sorted(matched_by_slug) for m in matched_by_slug[slug]]
        total_bytes = sum(m["size"] for m in all_matched)
        zip_path = workdir / f"{COMPLETE_SLUG}.zip"
        if _local_zip_matches(zip_path, len(all_matched), total_bytes) and not args.force:
            print(f"  {COMPLETE_SLUG}: local zip already current ({len(all_matched)} files) — reusing")
        else:
            print(f"  {COMPLETE_SLUG}: building {len(all_matched)} files, {total_bytes/1e9:.2f} GB")
            # Assemble from the local per-collection zips when they're all present
            # (the usual case right after a full build) instead of re-streaming
            # every object from S3.
            local_ok = all(
                (workdir / f"{slug}.zip").exists()
                and _zip_entry_count(workdir / f"{slug}.zip") == len(matched_by_slug[slug])
                for slug in matched_by_slug if matched_by_slug[slug]
            )
            if local_ok:
                print("    assembling from local per-collection zips")
                build_complete_from_local_zips(
                    workdir, [s for s in sorted(matched_by_slug) if matched_by_slug[s]], zip_path)
            else:
                build_zip(s3, args.bucket, zip_path,
                          [(f"{m['slug']}/", m["source_name"], m["key"], m["size"]) for m in all_matched],
                          workdir)
        complete_entry = {
            "zip_key": f"{ZIPS_PREFIX}/{COMPLETE_SLUG}.zip",
            "num_files": len(all_matched),
            "total_bytes": total_bytes,
            "zip_bytes": zip_path.stat().st_size,
            "built_at": now_iso(),
        }
        if not args.skip_upload:
            upload_zip(s3, args.bucket, zip_path, complete_entry["zip_key"])
            print(f"    uploaded s3://{args.bucket}/{complete_entry['zip_key']}")
    else:
        complete_entry = prior_manifest.get("complete")

    manifest = {
        "generated_at": now_iso(),
        "bucket": args.bucket,
        "collections": [new_entries[s] for s in sorted(new_entries)],
    }
    if complete_entry:
        manifest["complete"] = complete_entry

    manifest_path = workdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"\nManifest: {manifest_path} ({len(manifest['collections'])} collections)")
    if not args.skip_upload:
        upload_manifest(s3, args.bucket, manifest)
        print(f"  uploaded s3://{args.bucket}/{MANIFEST_KEY}")
        if args.invalidate:
            invalidate_cloudfront(args.bucket, args.region)

    print(f"\nDone: built {built}, skipped(unchanged/kept) {skipped}")
    if all_unmatched:
        print(f"{len(all_unmatched)} document(s) with no S3 object:")
        for slug, doc_id, name in all_unmatched:
            print(f"  UNMATCHED {slug} doc={doc_id} {name!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
