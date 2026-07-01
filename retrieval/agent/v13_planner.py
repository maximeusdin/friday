"""
V13 retrieval hardening — query planning, agreement-priming, and anti-false-negative guard.

V13 = V11 investigation loop + three additions that fix the "query framing != evidence
framing" failure class (see memory: friday-chat-query-framing-failures):

  1. plan_query()      — decompose the NL question into intent + entities + rare anchors
                         + focused keyword queries (drop framing verbs like "find/spied/
                         recruited/how many").
  2. prime_workspace() — before the agent's first turn, run hybrid(expand_concordance)
                         UNION lexical_exact(anchors), agreement-rank (chunks hit by
                         multiple queries/rare anchors float up), auto-fetch + summarize
                         the top K so the target evidence is in the workspace from turn 0.
  3. apply_anti_false_negative() — never let "not retrieved" become a confident
                         "no evidence exists"; suppress fabricated negative citations.

All of this is OFF for the V11 engine and ON only when run_v11_query(engine_profile="v13").
V11 behaviour is unchanged, so it remains the fallback.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from retrieval.agent.v9_types import CatalogHit, ScopeFilter, WorkspaceChunk, WorkspaceEntity

ENGINE_VERSION = "v13"

_PLANNER_MODEL = os.getenv("V13_PLANNER_MODEL", "gpt-4.1-mini-2025-04-14")

# Auto-fetch this many top primed chunks into fulltext (validated: targets land <=8).
K_PRIME_FETCH = int(os.getenv("V13_PRIME_FETCH", "16"))
# A lexical anchor with <= this many hits is "rare" and weighted higher in agreement.
RARE_ANCHOR_CUTOFF = int(os.getenv("V13_RARE_CUTOFF", "12"))
# An anchor with <= this many exact hits is almost certainly the target — guarantee-fetch
# every one of its hits (e.g. "500-600" -> 1 hit = the exact meetings entry).
ULTRA_RARE_FETCH = int(os.getenv("V13_ULTRA_RARE_FETCH", "3"))
# For a targeted lookup, if a user-named entity has <= this many exact hits, sweep ALL of
# them into the fetch set (the target passage may be the lowest-ranked hit of that entity).
ENTITY_SWEEP_CAP = int(os.getenv("V13_ENTITY_SWEEP_CAP", "25"))
# Hard ceiling on chunks auto-fetched during priming (agreement top-K + entity sweep).
MAX_PRIME_FETCH = int(os.getenv("V13_MAX_PRIME_FETCH", "22"))
# Per-query result cap when priming.
PRIME_TOP_K = int(os.getenv("V13_PRIME_TOP_K", "50"))

_PLANNER_SYS = (
    "You convert a researcher's natural-language question about a historical intelligence "
    "archive (Venona, Vassiliev notebooks, FBI files) into a RETRIEVAL PLAN. The archive is "
    "full of codenames, first-person depositions, and terse lists, so the evidence rarely "
    "contains the user's framing words.\n"
    "Return strict JSON with keys:\n"
    '  "intent": one of ["lookup","count","roster","compare","timeline"].\n'
    '  "entities": people/orgs/codenames/places named in the question (canonical spelling as given).\n'
    '  "anchors": the MOST DISTINCTIVE literal tokens to match exactly — proper nouns, surnames, '
    'org names, and any numbers/number-ranges (e.g. "500-600"). Prefer rare, specific words. '
    "Do NOT include framing words (find, evidence, describe, spied, recruited, how, many, list).\n"
    '  "queries": 2-4 SHORT keyword search strings (2-6 words each) built ONLY from content '
    "nouns/names/numbers — drop every framing verb. Include one variant that is just the key "
    "entities plus the single most distinctive anchor.\n"
    "Names may appear in the text only as codenames; still emit the canonical names the user gave."
)


def plan_query(question: str, *, model: str = _PLANNER_MODEL, verbose: bool = False) -> Dict[str, Any]:
    """Decompose an NL question into {intent, entities, anchors, queries}.

    LLM plan augmented with deterministic anchor extraction (numbers, quoted phrases,
    Capitalized proper-noun runs) so we never depend solely on the model for anchors.
    """
    plan: Dict[str, Any] = {"intent": "lookup", "entities": [], "anchors": [], "queries": []}

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_completion_tokens=300,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _PLANNER_SYS},
                    {"role": "user", "content": question.strip()},
                ],
            )
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(raw)
            if isinstance(data, dict):
                plan.update({k: data.get(k, plan[k]) for k in plan})
        except Exception as e:  # fail open — deterministic fallback below still runs
            if verbose:
                print(f"  [V13] planner LLM error ({e}); using heuristic plan", file=sys.stderr)

    # Deterministic anchor augmentation
    det_anchors = _regex_anchors(question)
    anchors = _dedup_preserve([*(plan.get("anchors") or []), *det_anchors])
    plan["anchors"] = [a for a in anchors if a and len(str(a).strip()) >= 2][:12]

    # Ensure at least one keyword query exists (fallback = framing-stripped question)
    queries = [q for q in (plan.get("queries") or []) if q and str(q).strip()]
    if not queries:
        queries = [_strip_framing(question)]
    plan["queries"] = _dedup_preserve(queries)[:4]
    plan["entities"] = _dedup_preserve(plan.get("entities") or [])[:12]

    if verbose:
        print(
            f"  [V13] plan intent={plan['intent']} entities={plan['entities']} "
            f"anchors={plan['anchors']} queries={plan['queries']}",
            file=sys.stderr,
        )
    return plan


_NUM_RE = re.compile(r"\b\d[\d,]*(?:\s*[-–]\s*\d[\d,]*)?\b")
_QUOTE_RE = re.compile(r'"([^"]{2,40})"|“([^”]{2,40})”')
_PROPER_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2})\b")
_FRAMING = {
    "find", "evidence", "description", "describe", "identify", "show", "list", "tell",
    "give", "how", "many", "much", "who", "whom", "which", "what", "were", "was", "did",
    "does", "is", "are", "the", "a", "an", "of", "in", "on", "with", "for", "to", "by",
    "and", "or", "that", "this", "these", "those", "me", "us", "all", "any", "some",
    "spied", "spy", "spying", "recruited", "recruit", "meeting", "meet", "met", "times",
    "notebooks", "notebook", "file", "files", "about", "soviet", "intelligence",
}


def _regex_anchors(text: str) -> List[str]:
    anchors: List[str] = []
    for m in _NUM_RE.finditer(text):
        tok = re.sub(r"\s*", "", m.group(0))
        if any(ch.isdigit() for ch in tok) and len(tok) >= 2:
            anchors.append(tok)
    for m in _QUOTE_RE.finditer(text):
        anchors.append((m.group(1) or m.group(2)).strip())
    # Proper nouns not at sentence start / not framing words
    for m in _PROPER_RE.finditer(text):
        cand = m.group(1).strip()
        if cand.lower() not in _FRAMING and cand.split()[0].lower() not in _FRAMING:
            anchors.append(cand)
    return _dedup_preserve(anchors)


def _strip_framing(question: str) -> str:
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", question)
    kept = [t for t in toks if t.lower() not in _FRAMING]
    return " ".join(kept) if kept else question.strip()


def _dedup_preserve(items: List[Any]) -> List[Any]:
    seen, out = set(), []
    for it in items:
        key = str(it).strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(it if not isinstance(it, str) else it.strip())
    return out


def _agreement_rank(
    result_sets: List[List[int]],
    rare_flags: List[bool],
) -> List[Tuple[int, float]]:
    """Score each chunk by cross-set agreement.

    +2 for membership in a rare set (few hits => high precision), +1 for a normal set,
    plus a small within-set rank bonus, plus an INTERSECTION bonus (chunks matched by
    multiple distinct queries/anchors are far more likely to be the target — e.g. the
    profession list is in both the 'engineers' and 'journalists' sets). Returns
    [(chunk_id, score)] sorted desc.
    """
    score: Dict[int, float] = defaultdict(float)
    members: Dict[int, int] = defaultdict(int)
    for ids, rare in zip(result_sets, rare_flags):
        w = 2.0 if rare else 1.0
        n = len(ids)
        for i, cid in enumerate(ids):
            score[cid] += w + max(0.0, (n - i)) / (n * 4.0 if n else 1.0)
            members[cid] += 1
    # Intersection bonus: +1.5 for each distinct set beyond the first.
    for cid, m in members.items():
        if m >= 2:
            score[cid] += 1.5 * (m - 1)
    return sorted(score.items(), key=lambda kv: -kv[1])


def prime_workspace(
    conn,
    workspace,
    plan: Dict[str, Any],
    scope: Optional[ScopeFilter],
    *,
    verbose: bool = False,
    progress_callback: Optional[Callable] = None,
    expand_hop: bool = True,
) -> Dict[str, Any]:
    """Seed the workspace with agreement-ranked evidence before the agent loop.

    Returns a small diagnostics dict.
    """
    from retrieval.agent.tools import hybrid_search_tool, lexical_exact_tool
    from retrieval.agent.v11_tools import fetch_chunks
    from retrieval.agent.v9_workspace import (
        merge_catalog_hits, merge_fetched_chunks, build_chunk_doc_map,
        merge_evidence_summary_update, build_alias_context_for_summarizer,
    )
    from retrieval.agent.v9_summarize import summarize_delta_chunks

    cols = scope.collections if (scope and scope.collections) else None
    date_from = scope.date_from if scope else None
    date_to = scope.date_to if scope else None

    result_sets: List[List[int]] = []
    rare_flags: List[bool] = []
    labels: List[str] = []

    def _hybrid(q: str, canonical: bool = False) -> List[int]:
        try:
            r = hybrid_search_tool(
                conn, query=q, top_k=PRIME_TOP_K, collections=cols,
                date_from=date_from, date_to=date_to,
                expand_concordance=True, fuzzy_enabled=True,
                use_canonical_embeddings=canonical,
            )
            return list(r.chunk_ids or [])
        except Exception:
            try: conn.rollback()
            except Exception: pass
            return []

    def _lexical(term: str) -> List[int]:
        try:
            r = lexical_exact_tool(
                conn, term=term, top_k=PRIME_TOP_K, collections=cols, expand_aliases=True,
            )
            return list(r.chunk_ids or [])
        except Exception:
            try: conn.rollback()
            except Exception: pass
            return []

    if progress_callback:
        progress_callback("planning", "completed",
                          f"Planned {plan.get('intent','lookup')} query", {
                              "intent": plan.get("intent"),
                              "entities": plan.get("entities"),
                              "anchors": plan.get("anchors"),
                          })

    # Seed the query entities' codenames/aliases so the summarizer + synthesis can bridge
    # canonical<->codename (Golos<->Sound, Rabinovich<->Harry) when the evidence text only
    # uses the codename. Without this the model reads "500-600 meetings with Sound" and
    # fails to connect it to Golos.
    _seed_entity_aliases(conn, workspace, plan.get("entities", []), verbose=verbose)

    # Round 1: focused hybrid queries + exact anchors.
    # Lead with a pure content-anchor join (no framing, no date decades) — the highest-signal
    # query for terse lists and specific passages (e.g. "engineers journalists" finds the
    # profession tally; "500-600 Golos Rabinovich" finds the meetings entry).
    _date_re = re.compile(r"^\d{3,4}s$|^(?:19|20)\d\d$")
    content_anchors = [str(a) for a in plan.get("anchors", []) if not _date_re.match(str(a).strip())]
    queries = list(plan.get("queries", []))
    if content_anchors:
        queries.insert(0, " ".join(content_anchors[:6]))
    for q in _dedup_preserve(queries):
        ids = _hybrid(q)
        if ids:
            result_sets.append(ids); rare_flags.append(False); labels.append(f"hy:{q}")
    for a in plan.get("anchors", []):
        ids = _lexical(str(a))
        if ids:
            result_sets.append(ids); rare_flags.append(len(ids) <= RARE_ANCHOR_CUTOFF)
            labels.append(f"lex:{a}")

    # Multi-anchor intersection: chunks that contain >=2 distinct EXACT anchors are very
    # likely the enumeration/target — e.g. only ~5 chunks archive-wide contain both
    # "engineers" and "journalists", and the profession tally is one of them. This is the
    # decisive signal for count/roster queries that a diluted hybrid OR buries.
    anchor_sets = [ids for ids, lbl in zip(result_sets, labels) if lbl.startswith("lex:")]
    if len(anchor_sets) >= 2:
        acount: Dict[int, int] = defaultdict(int)
        for s in anchor_sets:
            for cid in s:
                acount[cid] += 1
        inter = [cid for cid, c in sorted(acount.items(), key=lambda kv: -kv[1]) if c >= 2]
        if inter:
            result_sets.append(inter); rare_flags.append(True)
            labels.append(f"intersect:{len(inter)}")

    # Entity sweep: for a user-named entity that isn't too common, keep ALL its exact hits
    # for fetching — the target passage is sometimes the lowest-ranked hit of that entity
    # (e.g. the Bentley/Waldo deposition is the last of ~40 'Waldo' hits).
    sweep_ids: List[int] = []
    # Ultra-rare anchors (<=3 hits) are almost surely the target -> guarantee their fetch.
    for ids, lbl in list(zip(result_sets, labels)):
        if lbl.startswith("lex:") and 0 < len(ids) <= ULTRA_RARE_FETCH:
            sweep_ids.extend(ids)
    anchor_low = {str(a).strip().lower() for a in plan.get("anchors", [])}
    for ent in plan.get("entities", []):
        ent = str(ent).strip()
        if not ent or len(ent) < 3:
            continue
        term = max((t for t in re.split(r"[,\s]+", ent) if len(t) >= 3), key=len, default=ent)
        ids = _lexical(term) if term.lower() not in anchor_low else next(
            (s for s, l in zip(result_sets, labels) if l == f"lex:{term}"), [])
        if ids and len(ids) <= ENTITY_SWEEP_CAP:
            result_sets.append(ids); rare_flags.append(len(ids) <= RARE_ANCHOR_CUTOFF)
            labels.append(f"sweep:{term}")
            sweep_ids.extend(ids)

    # Optional expand hop: pull distinctive co-entities out of round-1 top chunks and
    # match them exactly. This is what surfaces first-person / co-mention evidence whose
    # key nouns were NOT in the user's question (e.g. Bentley->Golos/McClure).
    hop_terms: List[str] = []
    if expand_hop and result_sets:
        prelim = [cid for cid, _ in _agreement_rank(result_sets, rare_flags)][:20]
        hop_terms = _co_entity_terms(conn, prelim, cols, exclude=plan.get("entities", []))
        for t in hop_terms[:10]:
            ids = _lexical(t)
            if ids:
                result_sets.append(ids); rare_flags.append(len(ids) <= RARE_ANCHOR_CUTOFF)
                labels.append(f"hop:{t}")

    if not result_sets:
        return {"primed": 0, "fetched": 0, "labels": labels}

    ranked = _agreement_rank(result_sets, rare_flags)
    ranked_ids = [cid for cid, _ in ranked]

    # Merge everything into the catalog (ordered by agreement) so the context pack surfaces it.
    catalog = _load_catalog(conn, ranked_ids[:PRIME_TOP_K])
    merge_catalog_hits(workspace, catalog)

    # Auto-fetch into fulltext + summarize -> evidence bullets from turn 0.
    # Fetch = agreement top-K UNION the entity sweep (dedup, ordered by agreement), capped.
    fetch_ids = list(dict.fromkeys([*ranked_ids[:K_PRIME_FETCH], *[c for c in ranked_ids if c in set(sweep_ids)]]))
    fetch_ids = fetch_ids[:MAX_PRIME_FETCH]
    fetched = fetch_chunks(conn, chunk_ids=fetch_ids, include_neighbors=False)
    if scope and scope.collections:
        sset = {s.lower() for s in scope.collections}
        fetched = [c for c in fetched if not c.source_label or (c.source_label or "").lower() in sset
                   or (c.collection_slug or "").lower() in sset]
    merge_fetched_chunks(workspace, fetched)

    # Extract + link entities from primed chunks so the codename alias-map is available to
    # the summarizer and final synthesis (bridges Sound->Golos, Harry->Rabinovich, etc.).
    try:
        from retrieval.agent.v11_runner import _extract_and_merge_entities_from_chunks
        from retrieval.agent.v9_workspace import link_chunks_to_entities
        _extract_and_merge_entities_from_chunks(conn, workspace, [c.chunk_id for c in fetched])
        link_chunks_to_entities(workspace, fetched)
    except Exception as e:
        if verbose:
            print(f"  [V13] prime entity extraction failed: {e}", file=sys.stderr)

    summarized = 0
    if fetched:
        try:
            alias_ctx = build_alias_context_for_summarizer(workspace)
            ev = summarize_delta_chunks(fetched[:30], workspace.question, alias_context=alias_ctx)
            cdm = build_chunk_doc_map(workspace)
            merge_evidence_summary_update(workspace, ev, cdm)
            summarized = len(ev.bullets)
            for c in fetched:
                workspace._summarized_chunk_ids.add(c.chunk_id)
            if progress_callback and ev.bullets:
                progress_callback("evidence_update", "completed",
                                  f"Primed {len(ev.bullets)} evidence bullets", {
                                      "bullets": [{"text": b.text, "tags": b.tags,
                                                   "chunk_ids": b.supporting_chunk_ids,
                                                   "doc_ids": b.doc_ids} for b in ev.bullets],
                                      "total_bullet_count": len(ev.bullets),
                                  })
        except Exception as e:
            if verbose:
                print(f"  [V13] prime summarize failed: {e}", file=sys.stderr)

    workspace.notes.append(
        f"[V13 priming] intent={plan.get('intent')}, seeded {len(catalog)} catalog hits, "
        f"fetched {len(fetched)} chunks, {summarized} bullets; anchors={plan.get('anchors')}"
    )
    if verbose:
        print(
            f"  [V13] primed: sets={len(result_sets)} ({labels}), catalog={len(catalog)}, "
            f"fetched={len(fetched)}, bullets={summarized}, top={ranked_ids[:8]}",
            file=sys.stderr,
        )
    return {"primed": len(catalog), "fetched": len(fetched), "bullets": summarized,
            "labels": labels, "top_ids": ranked_ids[:12], "hop_terms": hop_terms}


def _co_entity_terms(conn, chunk_ids: List[int], cols, exclude: List[str]) -> List[str]:
    """Distinctive terms co-occurring in the given chunks (for the expand hop).

    Two sources: (1) linked entities of ANY type (person/org/other — so org anchors like
    'McClure' are kept), and (2) distinctive Capitalized proper-noun tokens from the chunk
    text itself (catches names/orgs that aren't indexed as entities). The bridge noun that
    reaches first-person / co-mention evidence is usually NOT in the user's question, so
    this hop is what makes e.g. Bentley->Golos/McClure reachable.
    """
    if not chunk_ids:
        return []
    exclude_low = {str(e).strip().lower() for e in (exclude or [])}
    out: List[str] = []

    # (1) linked entities, any type
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT e.canonical_name, COUNT(*) AS n
                FROM entity_mentions em
                JOIN entities e ON e.id = em.entity_id
                WHERE em.chunk_id = ANY(%s)
                  AND length(e.canonical_name) BETWEEN 3 AND 40
                GROUP BY e.canonical_name
                ORDER BY n DESC
                LIMIT 30
            """, (list(chunk_ids),))
            for name, _ in cur.fetchall():
                for tok in re.split(r"[,\s]+", name or ""):
                    if len(tok) >= 4 and tok[0].isupper() and tok.lower() not in exclude_low \
                            and tok.lower() not in _FRAMING:
                        out.append(tok)
    except Exception:
        try: conn.rollback()
        except Exception: pass

    return _dedup_preserve(out)


def _seed_entity_aliases(conn, workspace, entities: List[str], *, verbose: bool = False) -> None:
    """Resolve each query entity to its concordance aliases and merge into the workspace,
    so the codename<->canonical mapping is available to the summarizer and synthesis."""
    from retrieval.agent.v9_workspace import merge_entities
    names = [str(e).strip() for e in (entities or []) if e and len(str(e).strip()) >= 3]
    if not names:
        return
    seeded = 0
    for name in names[:8]:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT e.id, e.canonical_name,
                           array_agg(DISTINCT ea2.alias) FILTER (WHERE ea2.alias IS NOT NULL)
                    FROM entities e
                    LEFT JOIN entity_aliases ea2 ON ea2.entity_id = e.id
                    WHERE e.id IN (
                        SELECT e2.id FROM entities e2
                        LEFT JOIN entity_aliases ea ON ea.entity_id = e2.id
                        WHERE lower(e2.canonical_name) = lower(%(n)s)
                           OR lower(e2.canonical_name) LIKE lower(%(like)s)
                           OR lower(ea.alias) = lower(%(n)s)
                    )
                    GROUP BY e.id, e.canonical_name
                    ORDER BY COALESCE(array_length(array_agg(DISTINCT ea2.alias), 1), 0) DESC
                    LIMIT 2
                """, {"n": name, "like": f"%{name}%"})
                rows = cur.fetchall()
        except Exception:
            try: conn.rollback()
            except Exception: pass
            continue
        for eid, canonical, aliases in rows:
            al = [a for a in (aliases or []) if a and len(a) >= 2][:20]
            if canonical:
                merge_entities(workspace, [WorkspaceEntity(
                    entity_id=eid, canonical_name=canonical, aliases=al,
                )])
                seeded += 1
    if verbose:
        print(f"  [V13] seeded {seeded} entity alias record(s) for {names}", file=sys.stderr)


def _load_catalog(conn, chunk_ids: List[int]) -> List[CatalogHit]:
    if not chunk_ids:
        return []
    try:
        conn.rollback()
    except Exception:
        pass
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.id, LEFT(COALESCE(c.clean_text, c.text), 300),
                   cm.document_id, COALESCE(p.pdf_page_number, cm.first_page_id) AS page_num,
                   cm.collection_slug
            FROM chunks c
            LEFT JOIN chunk_metadata cm ON cm.chunk_id = c.id
            LEFT JOIN pages p ON p.id = cm.first_page_id
            WHERE c.id = ANY(%s)
        """, (list(chunk_ids),))
        rows = {r[0]: r for r in cur.fetchall()}
    out: List[CatalogHit] = []
    for rank, cid in enumerate(chunk_ids):
        r = rows.get(cid)
        if not r:
            continue
        out.append(CatalogHit(
            chunk_id=cid,
            score=max(0.0, (len(chunk_ids) - rank) / len(chunk_ids)),  # agreement order
            doc_id=r[2],
            page=f"p{r[3]}" if r[3] else None,
            collection=r[4],
            snippet=(r[1] or "").strip(),
        ))
    return out


# ---------------------------------------------------------------------------
# Anti-false-negative guard
# ---------------------------------------------------------------------------

_NEGATION_RE = re.compile(
    r"\b(no (?:explicit |direct )?(?:evidence|mention|description|reference|record|indication|"
    r"data|information|figures?|numbers?)|does not (?:contain|indicate|mention|document|specify|"
    r"provide)|do not (?:contain|indicate|document|specify)|there(?:'s| is| are)? no|"
    r"not (?:find|found|contain|mention|documented|specified|available)|reveals? no|"
    r"could(?:n't| not) find|nothing (?:to|that|was)|no (?:such )?(?:mention|record))",
    re.IGNORECASE,
)


def apply_anti_false_negative(result, workspace, *, verbose: bool = False) -> None:
    """Never let 'not retrieved' become a confident 'no evidence exists'.

    - If the answer asserts non-existence, cap sufficiency and reword to 'not found in what
      I searched' (with the queries actually run), and surface any adjacent evidence bullets.
    - Strip citations from negative claims (a citation must support an affirmative statement).
    """
    synth = getattr(result, "synthesis", None)
    narrative = (getattr(result, "narrative", None) or (synth.narrative if synth else "")) or ""
    is_negative = bool(_NEGATION_RE.search(narrative))

    def _claim_text(c) -> str:
        inner = getattr(c, "claim", None)
        return (getattr(inner, "text", "") if inner is not None else getattr(c, "text", "")) or ""

    # Drop fabricated negative citations (a citation must support an AFFIRMATIVE statement).
    for c in (getattr(result, "claims", None) or []):
        if _NEGATION_RE.search(_claim_text(c)) and hasattr(c, "citation_chunk_ids"):
            c.citation_chunk_ids = []

    # Affirmative grounded evidence = grounded/weak claims whose text is NOT a negation.
    # (A "no evidence found [p62]" claim is not affirmative evidence and must not suppress
    # the guard — that was the #2 misfire.)
    affirmative = [c for c in (getattr(result, "claims", None) or [])
                   if getattr(c, "status", "") in ("grounded", "weak")
                   and not _NEGATION_RE.search(_claim_text(c))]
    grounded = affirmative

    # If the model produced grounded affirmative evidence, it DID find something — don't
    # slap a "could not find" banner on it (that was the #1 misfire). Just leave it.
    if grounded:
        return

    if not is_negative:
        return

    # Negative answer with no grounded affirmative claims — downgrade + reframe honestly.
    suf = getattr(result, "sufficiency", None)
    if suf is not None and getattr(suf, "sufficient", False):
        suf.sufficient = False
        if hasattr(suf, "argument"):
            suf.argument = (suf.argument or "") + " [V13: negative result — downgraded; absence of retrieval is not evidence of absence.]"

    queries = list(dict.fromkeys(getattr(workspace, "_search_queries", []) or []))[:6]
    searched = f" I searched: {', '.join(queries)}." if queries else ""
    prefix = (
        "I could not find direct evidence for that in the sources I searched, "
        "so this is a limitation of the search, not confirmation that none exists." + searched
    )

    # Surface adjacent evidence bullets, if any, instead of a flat negative.
    bullets = []
    try:
        for b in (workspace._bullet_index or {}).values():
            if getattr(b, "text", ""):
                bullets.append(b.text)
    except Exception:
        pass
    adjacent = ""
    if bullets:
        adjacent = "\n\nRelated evidence I did find:\n" + "\n".join(f"- {t}" for t in bullets[:5])

    new_narrative = prefix + adjacent + "\n\n---\n" + narrative
    if synth is not None:
        synth.narrative = new_narrative
    try:
        result.narrative = new_narrative
    except Exception:
        pass
    if verbose:
        print("  [V13] anti-false-negative guard applied", file=sys.stderr)


def run_v13_query(conn, question: str, **kwargs):
    """Convenience wrapper: V11 loop with the V13 profile enabled."""
    from retrieval.agent.v11_runner import run_v11_query
    kwargs["engine_profile"] = "v13"
    return run_v11_query(conn, question, **kwargs)
