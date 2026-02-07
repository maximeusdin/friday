#!/usr/bin/env python3
"""Interactive CLI to resolve needs_review_resolution.json.

Supports:
- Listing and selecting clusters
- Setting action (merge, multiple_referents, unresolved), survivor_entity_id, notes
- Adding referent_mappings with document lookup: type a document name -> fuzzy match
  against DB documents -> confirm -> pick entity_id and optional alias

Usage:
  python scripts/resolve_needs_review_cli.py [--file PATH] [--db URL]
  With --db (or DATABASE_URL): document name is fuzzy-matched to documents.id.
  Without --db: you can still add referent_mappings with source string only (document_id null).
"""
import argparse
import difflib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Document lookup (optional, when DB available)
# ---------------------------------------------------------------------------

def load_documents(conn) -> List[Dict[str, Any]]:
    """Return list of {id, source_name, collection_slug, display} for fuzzy lookup."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT d.id, d.source_name, c.slug, c.title
            FROM documents d
            JOIN collections c ON c.id = d.collection_id
            ORDER BY c.slug, d.source_name
        """)
        rows = cur.fetchall()
    out = []
    for r in rows:
        doc_id, source_name, slug, coll_title = r
        display = f"{slug}: {source_name}" if slug else source_name
        out.append({
            "id": doc_id,
            "source_name": source_name or "",
            "collection_slug": slug or "",
            "collection_title": coll_title or "",
            "display": display,
        })
    return out


def fuzzy_match_document(
    query: str,
    documents: List[Dict[str, Any]],
    top_n: int = 5,
    cutoff: float = 0.4,
) -> List[Dict[str, Any]]:
    """Return best matching documents. Uses exact match first, then difflib."""
    q = query.strip().lower()
    if not q:
        return []

    # Build searchable strings: display and source_name
    choices = []
    for d in documents:
        choices.append((d["display"], d))
        if d["source_name"].lower() != d["display"].lower():
            choices.append((d["source_name"], d))

    # Exact match (case-insensitive)
    for label, doc in choices:
        if label.lower() == q:
            return [doc]

    # Fuzzy: score each document by best match of any of its strings
    scored: List[Tuple[float, Dict[str, Any]]] = []
    seen_id = set()
    for label, doc in choices:
        if doc["id"] in seen_id:
            continue
        matches = difflib.get_close_matches(q, [label], n=1, cutoff=cutoff)
        if matches:
            ratio = difflib.SequenceMatcher(None, q, label.lower()).ratio()
            scored.append((ratio, doc))
            seen_id.add(doc["id"])
    scored.sort(key=lambda x: -x[0])
    return [doc for _, doc in scored[:top_n]]


def prompt_document(
    documents: List[Dict[str, Any]],
    prompt_str: str = "Document name (or part of): ",
) -> Optional[Dict[str, Any]]:
    """Let user type a name; fuzzy match; confirm; return chosen doc or None."""
    name = input(prompt_str).strip()
    if not name:
        return None
    matches = fuzzy_match_document(name, documents, top_n=5)
    if not matches:
        print("  No matching documents.")
        return None
    if len(matches) == 1:
        d = matches[0]
        print(f"  Match: [{d['id']}] {d['display']}")
        confirm = input("  Use this? [Y/n]: ").strip().lower()
        if confirm and confirm != "y":
            return None
        return d
    print("  Matches:")
    for i, d in enumerate(matches, 1):
        print(f"    {i}. [{d['id']}] {d['display']}")
    choice = input("  Number (or Enter to cancel): ").strip()
    if not choice:
        return None
    try:
        idx = int(choice)
        if 1 <= idx <= len(matches):
            return matches[idx - 1]
    except ValueError:
        pass
    return None


# ---------------------------------------------------------------------------
# Resolution data and edit flow
# ---------------------------------------------------------------------------

def load_resolution(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_resolution(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {path}")


def print_cluster(cluster: Dict[str, Any], index: int) -> None:
    ids = cluster["entity_ids"]
    names = cluster["canonical_names"]
    print(f"\n--- Cluster {index} ---")
    print(f"  Reason: {cluster.get('reason', '')}")
    print(f"  Action: {cluster.get('action') or '(empty)'}")
    print(f"  Survivor: {cluster.get('survivor_entity_id')}")
    print(f"  Notes: {cluster.get('notes') or ''}")
    print("  Entities:")
    for i, (eid, cname) in enumerate(zip(ids, names), 1):
        mark = " <- survivor" if eid == cluster.get("survivor_entity_id") else ""
        print(f"    {i}. [{eid}] {cname}{mark}")
    refs = cluster.get("referent_mappings") or []
    if refs:
        print("  Referent mappings:")
        for i, r in enumerate(refs, 1):
            doc_id = r.get("document_id")
            src = r.get("source") or "(no source)"
            eid = r.get("entity_id")
            alias = r.get("alias")
            print(f"    {i}. doc_id={doc_id} source={src!r} -> entity_id={eid} alias={alias!r}")
    print()


def edit_cluster(
    cluster: Dict[str, Any],
    index: int,
    documents: Optional[List[Dict[str, Any]]],
    walk_mode: bool = False,
) -> str:
    """Interactive edit of one cluster. Returns 'next' to advance to next cluster, 'done' to return to menu."""
    ids = cluster["entity_ids"]
    names = cluster["canonical_names"]
    refs = cluster.setdefault("referent_mappings", [])

    while True:
        print_cluster(cluster, index)
        if walk_mode:
            print("  (next = next cluster, done = back to menu)")
        line = input("edit> ").strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        rest = (parts[1] if len(parts) > 1 else "").strip()

        if cmd == "next":
            if walk_mode:
                return "next"
            print("  Use 'next' in walk mode only.")
            continue
        if cmd == "done" or cmd == "q":
            return "done"
        if cmd == "action":
            if rest in ("merge", "multiple_referents", "unresolved"):
                cluster["action"] = rest
                if rest != "merge":
                    cluster["survivor_entity_id"] = None
                print(f"  Action set to {rest}")
            else:
                print("  Use: action merge | multiple_referents | unresolved")
        elif cmd == "survivor":
            try:
                n = int(rest)
                if 1 <= n <= len(ids):
                    cluster["survivor_entity_id"] = ids[n - 1]
                    cluster["action"] = "merge"
                    print(f"  Survivor set to entity_id {ids[n - 1]} ({names[n - 1]})")
                else:
                    print(f"  Enter a number 1..{len(ids)}")
            except ValueError:
                print("  Enter survivor number (1-based index in entity list)")
        elif cmd == "notes":
            cluster["notes"] = rest
            print("  Notes updated.")
        elif cmd == "add-doc":
            if not documents:
                print("  No DB connection; use add-source <source string> instead.")
                continue
            doc = prompt_document(documents)
            if not doc:
                continue
            # Pick entity_id from cluster
            print("  Entities in this cluster:")
            for i, (eid, cname) in enumerate(zip(ids, names), 1):
                print(f"    {i}. [{eid}] {cname}")
            ent_choice = input("  Entity number (1-based): ").strip()
            try:
                ent_n = int(ent_choice)
                if 1 <= ent_n <= len(ids):
                    entity_id = ids[ent_n - 1]
                    alias = input("  Alias as in document (optional, Enter to skip): ").strip() or None
                    refs.append({
                        "document_id": doc["id"],
                        "source": doc["display"],
                        "entity_id": entity_id,
                        "alias": alias,
                    })
                    print("  Added.")
                else:
                    print("  Invalid number.")
            except ValueError:
                print("  Enter a number.")
        elif cmd == "add-source":
            # No DB: source string only
            if not rest:
                rest = input("  Source string (e.g. Fbicomrap p.129): ").strip()
            if not rest:
                continue
            print("  Entities in this cluster:")
            for i, (eid, cname) in enumerate(zip(ids, names), 1):
                print(f"    {i}. [{eid}] {cname}")
            ent_choice = input("  Entity number (1-based): ").strip()
            try:
                ent_n = int(ent_choice)
                if 1 <= ent_n <= len(ids):
                    entity_id = ids[ent_n - 1]
                    alias = input("  Alias as in document (optional, Enter to skip): ").strip() or None
                    refs.append({
                        "document_id": None,
                        "source": rest,
                        "entity_id": entity_id,
                        "alias": alias,
                    })
                    print("  Added.")
                else:
                    print("  Invalid number.")
            except ValueError:
                print("  Enter a number.")
        elif cmd == "remove-doc":
            try:
                i = int(rest)
                if 1 <= i <= len(refs):
                    refs.pop(i - 1)
                    print("  Removed.")
                else:
                    print(f"  Enter 1..{len(refs)}")
            except ValueError:
                print("  Use: remove-doc <number> (from list-docs)")
        elif cmd == "help":
            print("  EDIT COMMANDS (type exactly as below):")
            print("  -----------------")
            print("  action merge              - merge all entities in this cluster into one; you must set survivor.")
            print("  action multiple_referents - alias means different people in different docs; do not merge; add referent_mappings.")
            print("  action unresolved          - leave for later.")
            print("  survivor <N>              - set survivor to the N-th entity (N = 1-based index from the Entities list above). Also sets action to merge.")
            print("  notes <your text>         - free-form note (e.g. source, reason).")
            print("  add-doc                   - you will be prompted: document name (fuzzy match) -> confirm -> entity number -> optional alias. Fills document_id + source.")
            print("  add-source <source>       - add mapping with source string only (no DB). You will be prompted for entity number and optional alias.")
            print("  remove-doc <N>            - remove the N-th row from Referent mappings (N = 1-based).")
            if walk_mode:
                print("  next                      - save file and go to next cluster.")
            print("  done   or   q             - exit edit and return to main menu.")
            print("  -----------------")
            print("  Example (merge into person):  action merge   then   survivor 3   (if the person is 3rd in the list)")
            print("  Example (multiple referents): action multiple_referents   then   add-doc   or   add-source Fbicomrap p.129")
        else:
            print("  Type  help  for full instructions and examples.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive CLI to resolve needs_review_resolution.json",
    )
    parser.add_argument(
        "--file",
        default="needs_review_resolution.json",
        help="Path to resolution JSON (default: needs_review_resolution.json)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Database URL for document lookup (default: DATABASE_URL env)",
    )
    parser.add_argument(
        "--walk",
        action="store_true",
        help="Start in walk mode: go through clusters one by one",
    )
    args = parser.parse_args()

    path = args.file
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    data = load_resolution(path)
    print(f"Loaded {len(data)} clusters from {path}")

    conn = None
    documents: Optional[List[Dict[str, Any]]] = None
    db_url = args.db or os.environ.get("DATABASE_URL")
    if db_url:
        try:
            import psycopg2
            conn = psycopg2.connect(db_url)
            documents = load_documents(conn)
            print(f"Document lookup enabled ({len(documents)} documents). Use add-doc to map by name.")
        except Exception as e:
            print(f"DB connection failed: {e}. Document lookup disabled; use add-source for free-form source.", file=sys.stderr)
            documents = None
    else:
        print("No DATABASE_URL or --db. Use add-source to add referent mappings with source string only.")

    def run_walk(start_at: int = 1) -> None:
        """Present clusters one by one; user makes decisions then 'next' to continue."""
        for i in range(start_at, len(data) + 1):
            print(f"\n{'='*60}")
            print(f"Cluster {i} of {len(data)}")
            print("="*60)
            result = edit_cluster(data[i - 1], i, documents, walk_mode=True)
            if result == "next":
                save_resolution(path, data)
                if i < len(data):
                    continue
                print("\nAll clusters done.")
                break
            else:
                print("\nBack to menu.")
                break

    if args.walk:
        run_walk(1)
    else:
        print("\nCommands: walk [N], list [filter], show <N>, edit <N>, save, quit")
        print("  walk [N]    - go through clusters one by one (optionally start at N)")
        print("  list        - list all clusters (optionally: list unresolved, list multiple_referents)")
        print("  show N      - print cluster N (1-based)")
        print("  edit N      - edit cluster N")
        print("  save        - write JSON to file")
        print("  quit        - exit (save first if needed)")

    while True:
        try:
            line = input("\n> ").strip()
        except EOFError:
            break
        if not line:
            continue
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        rest = (parts[1] if len(parts) > 1 else "").strip()

        if cmd == "quit" or cmd == "exit":
            break
        if cmd == "save":
            save_resolution(path, data)
            continue
        if cmd == "list":
            filter_action = rest if rest else None
            for i, c in enumerate(data, 1):
                act = c.get("action") or ""
                if filter_action and act != filter_action:
                    continue
                names_preview = ", ".join((c.get("canonical_names") or [])[:3])
                if len(c.get("canonical_names") or []) > 3:
                    names_preview += "..."
                print(f"  {i:3d}. [{act or 'empty':18s}] {names_preview}")
            continue
        if cmd == "show":
            try:
                n = int(rest)
                if 1 <= n <= len(data):
                    print_cluster(data[n - 1], n)
                else:
                    print(f"  Enter 1..{len(data)}")
            except ValueError:
                print("  Use: show <number>")
            continue
        if cmd == "edit":
            try:
                n = int(rest)
                if 1 <= n <= len(data):
                    edit_cluster(data[n - 1], n, documents, walk_mode=False)
                else:
                    print(f"  Enter 1..{len(data)}")
            except ValueError:
                print("  Use: edit <number>")
            continue
        if cmd == "walk":
            try:
                start = int(rest) if rest else 1
                if 1 <= start <= len(data):
                    run_walk(start)
                else:
                    print(f"  Enter 1..{len(data)} or just 'walk' to start at 1")
            except ValueError:
                if rest:
                    print("  Use: walk [N] (N = cluster number to start at)")
                else:
                    run_walk(1)
            continue
        print("  Unknown command. Use: walk, list, show N, edit N, save, quit")

    if conn:
        conn.close()
    print("Bye.")


if __name__ == "__main__":
    main()
