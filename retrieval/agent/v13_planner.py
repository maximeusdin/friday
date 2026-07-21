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
# them into the fetch set (the target passage may be the lowest-ranked hit of that entity,
# e.g. the Bentley/Waldo deposition is the ~40th of ~50 "Waldo" hits).
ENTITY_SWEEP_CAP = int(os.getenv("V13_ENTITY_SWEEP_CAP", "55"))
# Hard ceiling on chunks auto-fetched during priming (agreement top-K only, no sweep).
MAX_PRIME_FETCH = int(os.getenv("V13_MAX_PRIME_FETCH", "22"))
# Ceiling when an entity sweep fires (exhaustive fetch of a bounded named entity).
MAX_EXHAUSTIVE_FETCH = int(os.getenv("V13_MAX_EXHAUSTIVE_FETCH", "50"))
# Summarizer batch size (fetched chunks are summarized in batches of this many).
SUMMARY_BATCH = int(os.getenv("V13_SUMMARY_BATCH", "30"))
# Per-query result cap when priming.
PRIME_TOP_K = int(os.getenv("V13_PRIME_TOP_K", "50"))

# --- v14 coverage mode (roster/count/aggregation) ---
# Top-N chunks per collection in the coverage sweep (keep SMALL: V7's bottleneck flooded
# synthesis by grading every span from every collection — we cap hard and rank within each).
COVERAGE_PER_COLLECTION = int(os.getenv("V14_COVERAGE_PER_COLLECTION", "12"))
# Max fetch slots any single collection can take. Equal allocation among the top-N richest
# collections gives each roster source (incl. Vassiliev) real depth without any single-subject
# FBI file dominating; combined with the normalized target set this is wording-invariant.
COVERAGE_PER_COLLECTION_FETCH_MAX = int(os.getenv("V14_COVERAGE_PER_COL_FETCH_MAX", "7"))
# Total chunks auto-fetched in coverage mode (allocated across collections), batched-summarized.
COVERAGE_MAX_FETCH = int(os.getenv("V14_COVERAGE_MAX_FETCH", "60"))
# Concentrate the fetch budget on the N richest collections for the target (by match count)
# rather than spreading 1 chunk across every collection with a marginal mention — this keeps
# real depth in genuine sources (Vassiliev, SISS, HUAC, the FBI case files) while still
# spanning the corpus. Fewer collections * more depth each = better roster coverage.
COVERAGE_MAX_COLLECTIONS = int(os.getenv("V14_COVERAGE_MAX_COLLECTIONS", "12"))
# Intents that trigger coverage-first retrieval. ROSTER ONLY — enumerating people across the
# corpus is where breadth wins. COUNT is deliberately excluded: a count needs the one
# authoritative list (e.g. "49 engineers, 22 journalists"), and breadth dilutes it (the V7
# lesson) — count is served better by v13's multi-anchor intersection.
_COVERAGE_INTENTS = {"roster", "list", "enumerate"}

# Generic espionage/category nouns that are NOT a distinctive scope (used to tell an OPEN roster
# — "which journalists were spies" — from a SCOPED one — "the Silvermaster ring", "ENORMOZ
# traffic"). A scope anchor is a distinctive proper noun beyond these + the enumeration target.
_ROSTER_GENERIC_WORDS = {
    "soviet", "agent", "agents", "spy", "spies", "espionage", "traffic", "source", "sources",
    "asset", "assets", "contact", "contacts", "operative", "operatives", "member", "members",
    "network", "networks", "ring", "rings", "group", "groups", "cell", "apparatus", "people",
    "person", "intelligence", "communist", "american", "russian", "recruited", "named",
    "journalist", "journalists", "engineer", "engineers", "scientist", "scientists",
    "official", "officials", "diplomat", "diplomats", "employee", "employees",
}


def _roster_scope_anchors(plan) -> List[str]:
    """Distinctive proper-noun anchors that SCOPE a roster to a specific ring/operation/traffic
    (e.g. "Silvermaster", "ENORMOZ", "Perlo") — as opposed to an OPEN roster whose only anchor is
    a generic category. Returns [] for open rosters (keep corpus-wide breadth), non-empty for
    scoped rosters (focus retrieval on the scope). Generalizable: any specific named subject.
    """
    if not plan:
        return []
    tgt = (plan.get("enumeration_target") or "").strip().lower()
    syns = {str(s).strip().lower() for s in (plan.get("target_synonyms") or [])}
    out = []
    for a in (plan.get("anchors") or []):
        a = str(a).strip()
        if len(a) < 4 or a.lower() in _FRAMING:
            continue
        # An anchor scopes the roster only if it contains a DISTINCTIVE proper noun — a
        # capitalized word that is NOT a generic espionage/category word. Check each WORD, so a
        # generic phrase like "Soviet spies" ("soviet"+"spies", both generic) is NOT mistaken for
        # a scope (which would wrongly disable coverage-first breadth on an OPEN roster).
        distinctive = [
            w for w in re.split(r"[\s/]+", a) if w
            and any(ch.isupper() for ch in w)
            and w.lower() not in _ROSTER_GENERIC_WORDS
            and w.lower() not in _FRAMING
            and w.lower() != tgt
            and w.lower() not in syns
            and len(w) >= 3
        ]
        if distinctive:
            out.append(a)
    return out


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
    '  "enumeration_target": for count/roster/aggregation questions, the SINGLE common noun '
    'being counted or listed (e.g. "journalists", "engineers", "agents", "sources"); empty '
    "for lookup/compare/timeline. This is the category to enumerate across the whole corpus.\n"
    '  "target_synonyms": for roster/count, NORMALIZE the enumeration target to its standard '
    "occupational/category noun PLUS common synonyms, so different phrasings map to the SAME "
    'set. Reporters, newspapermen, correspondents, pressmen -> ["journalist","reporter",'
    '"correspondent","newspaperman","editor","press"]. Scientists/physicists -> ["scientist",'
    '"physicist"]. Engineers -> ["engineer"]. Return [] for lookup/compare/timeline.\n'
    '  "records_queries": 1-2 rewrites of the question into DOCUMENT-SEEKING form: what the '
    "archive's own records about the event would be titled or say. The archive describes events "
    "in bureaucratic record language, not the asker's words. Swap event verbs for the archival "
    "vocabulary the records themselves use (recruited -> initial contact, developed, cooperation, "
    'informant; defected -> statement, deposition, interview; spied -> espionage activities, '
    "contacts), and add likely record-type nouns (memorandum, report, letter, teletype, interview). "
    'Example: "When did the FBI recruit X as an informer?" -> ["X initial contact cooperation '
    'informant memorandum", "X developed informant report"]. Keep every entity name. Return [] '
    "if the question ALREADY asks for documents/records/files/reports.\n"
    "Names may appear in the text only as codenames; still emit the canonical names the user gave."
)


def plan_query(question: str, *, model: str = _PLANNER_MODEL, verbose: bool = False) -> Dict[str, Any]:
    """Decompose an NL question into {intent, entities, anchors, queries}.

    LLM plan augmented with deterministic anchor extraction (numbers, quoted phrases,
    Capitalized proper-noun runs) so we never depend solely on the model for anchors.
    """
    plan: Dict[str, Any] = {"intent": "lookup", "entities": [], "anchors": [], "queries": [],
                            "enumeration_target": "", "target_synonyms": [],
                            "records_queries": []}

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_completion_tokens=400,
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
    # Drop framing/generic words the model leaks into anchors despite instructions
    # ("Names", "spies", "infiltrated") — as lexical sets they are 10k+-hit noise that
    # poisons the multi-anchor intersection.
    plan["anchors"] = [
        a for a in anchors
        if a and len(str(a).strip()) >= 2 and str(a).strip().lower() not in _FRAMING
    ][:12]

    # Ensure at least one keyword query exists (fallback = framing-stripped question)
    queries = [q for q in (plan.get("queries") or []) if q and str(q).strip()]
    if not queries:
        queries = [_strip_framing(question)]
    plan["queries"] = _dedup_preserve(queries)[:4]
    plan["entities"] = _dedup_preserve(plan.get("entities") or [])[:12]

    # Records-oriented rewrite (evidence-seeking reframe). Off when the question is
    # already document-seeking (belt to the planner-prompt suspenders), for count/roster
    # intents (coverage-first enumeration is the right tool there, and without a person
    # anchor the reframe degenerates into vague vocabulary soup that dilutes evidence),
    # or via env.
    rq = [q for q in (plan.get("records_queries") or []) if q and str(q).strip()]
    if os.getenv("FRIDAY_RECORDS_REWRITE", "1") != "1":
        rq = []
    elif (plan.get("intent") or "lookup").lower() in ("count", "roster"):
        rq = []
    elif re.search(r"\b(documents?|records?|files?|reports?|memos?|memoranda)\b", question, re.I):
        rq = []
    else:
        # Entity-anchor requirement: a records rewrite must carry at least one entity
        # name from the plan — an entity-less reframe is generic vocabulary soup that
        # dilutes evidence (the failure the count-gate caught, generalized).
        ent_tokens = {
            t.lower()
            for e in (plan.get("entities") or [])
            for t in re.split(r"[\s,./]+", str(e))
            if len(t) >= 3 and t.lower() not in _FRAMING
        }
        if ent_tokens:
            rq = [q for q in rq
                  if any(t in str(q).lower() for t in ent_tokens)]
        else:
            rq = []
    plan["records_queries"] = _dedup_preserve(rq)[:2]

    if verbose:
        print(
            f"  [V13] plan intent={plan['intent']} entities={plan['entities']} "
            f"anchors={plan['anchors']} queries={plan['queries']} "
            f"records_queries={plan['records_queries']}",
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
    # Roster-question framing: category nouns and verbs that are never distinctive
    # anchors ("Names of NKVD spies who infiltrated the OSS" must anchor on
    # NKVD + OSS, not on names/spies/infiltrated — those are 10k+-hit generics).
    "names", "name", "spies", "agents", "agent", "sources", "source", "members",
    "member", "infiltrated", "infiltrate", "penetrated", "penetrate", "worked",
    "working", "involved", "identified",
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
    coverage: bool = False,
) -> Dict[str, Any]:
    """Seed the workspace with agreement-ranked evidence before the agent loop.

    coverage=True (v14) enables coverage-first retrieval for roster/count/aggregation
    intents: guarantee per-collection representation so evidence spread across the corpus
    (e.g. the Vassiliev Black Notebook journalist roster) isn't drowned out by the
    highest-volume collection.

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
    # Full-archive hybrid searches (vector over all collections) are the expensive path;
    # run a leaner plan there (fewer hybrid variants, no second expand hop).
    lean = not cols
    max_hybrid = 3 if lean else 8
    if lean:
        expand_hop = False

    # Coverage-first mode: only for roster/count/aggregation intents (breadth is the goal
    # there). The expand hop is redundant with coverage, so drop it to control cost.
    intent = (plan.get("intent") or "lookup").lower()
    # Coverage-first (corpus-wide spread) is right for an OPEN roster ("which journalists were
    # spies") but WRONG for a SCOPED one ("ENORMOZ traffic", "Silvermaster ring"): spreading
    # across every collection dilutes the scope with unrelated agents. For scoped rosters, fall
    # back to focused anchor retrieval so the evidence stays on-scope.
    _scope_anchors = _roster_scope_anchors(plan)
    coverage_mode = coverage and intent in _COVERAGE_INTENTS and not _scope_anchors
    if coverage_mode:
        expand_hop = False
    if _scope_anchors and verbose and intent in _COVERAGE_INTENTS:
        print(f"  [V14] scoped roster (anchors={_scope_anchors}) -> focused retrieval, "
              f"coverage disabled", file=sys.stderr)

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

    # Resolve codenames NAMED IN THE QUERY to their real person and inject the real name into the
    # plan, so retrieval reaches the right person's collection (not a same-codename decoy). Must
    # run BEFORE query construction + alias seeding so both pick up the resolved names.
    try:
        _resolve_query_codenames(conn, plan, workspace.question, verbose=verbose)
    except Exception as _ce:
        if verbose:
            print(f"  [V13] query codename resolution failed: {_ce}", file=sys.stderr)

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
    for q in _dedup_preserve(queries)[:max_hybrid]:
        ids = _hybrid(q)
        if ids:
            result_sets.append(ids); rare_flags.append(False); labels.append(f"hy:{q}")

    # Records-oriented rewrites: retrieval in the archive's own record language
    # ("initial contact / cooperation / memorandum" instead of "recruit"). Primed as their
    # own sets so they can surface documents the direct phrasing can't reach.
    for q in list(plan.get("records_queries") or [])[:2]:
        ids = _hybrid(str(q))
        if ids:
            result_sets.append(ids); rare_flags.append(False); labels.append(f"rq:{q}")
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

    # Boolean-engine intersection (deterministic, the researcher's move): for roster/count
    # questions with >=2 content anchors, run the Search tab's boolean engine on
    # "a1 AND a2" — word-boundary matching + alias expansion over EVERY page, so the
    # enumeration pool is exhaustive rather than top-k sampled (this is how Halperin's
    # 295 OSS chunks stop being missable). The result set persists as a session search
    # (origin='chat') the researcher can open and continue.
    # Enumeration intents ONLY: on lookups the exhaustive pool quadruples latency
    # (fuchs 361s -> 1446s in the handoff matrix) for no accuracy gain, and the
    # widened pool can surface misleading recollections over the precise record.
    if intent in _COVERAGE_INTENTS:
        ba = [str(a) for a in content_anchors if len(str(a)) >= 3][:4]
        if len(ba) >= 2:
            try:
                from retrieval.agent.v11_tools import boolean_search
                sess = getattr(workspace, "_search_session", None) or {}

                def _bt(t: str) -> str:
                    return f'"{t}"' if " " in t else t

                # The researcher's query shape: pin the SCOPE term, OR the alternatives.
                # "OSS AND (NKVD OR Soviet OR agent OR espionage)" finds Halperin's
                # "Soviet"-phrased chunks that a literal "NKVD AND OSS" misses.
                # Primary = the RAREST candidate anchor (deterministic; planner anchor
                # order varies run-to-run and pinning the common term — NKVD over OSS —
                # excludes scope pages phrased without it).
                _cands = (_scope_anchors or ba)[:3]
                try:
                    _counts = {a: _true_lexical_count(conn, str(a), cols) for a in _cands}
                    _pos = {c: n for c, n in _counts.items() if n > 0}
                    primary = min(_pos, key=_pos.get) if _pos else _cands[0]
                except Exception:
                    primary = _cands[0]
                others = [a for a in ba if a != primary][:2]
                syns = [s for s in (plan.get("target_synonyms") or []) if s and len(s) >= 4][:3]
                or_group = list(dict.fromkeys(
                    [*others, *syns, "Soviet", "agent", "espionage"]))[:6]
                bq = f"{_bt(primary)} AND ({' OR '.join(_bt(t) for t in or_group)})"
                bres = boolean_search(
                    conn, bq, scope=scope,
                    session_id=sess.get("session_id"),
                    user_sub=sess.get("user_sub") or "chat-engine",
                    origin_query=workspace.question or "",
                    max_hits_returned=250,
                )
                bids = [h["chunk_id"] for h in (bres.get("hits") or []) if h.get("chunk_id")]
                if bids:
                    result_sets.append(bids); rare_flags.append(True)
                    labels.append(f"bool:{bq[:40]}({bres.get('total_hits', 0)})")
                    try:
                        workspace._boolean_result_sets.append({
                            "result_set_id": bres.get("result_set_id"),
                            "query": bq, "total_hits": bres.get("total_hits", 0),
                        })
                    except AttributeError:
                        workspace._boolean_result_sets = [{
                            "result_set_id": bres.get("result_set_id"),
                            "query": bq, "total_hits": bres.get("total_hits", 0),
                        }]
            except Exception as _be:
                if verbose:
                    print(f"  [V14] boolean priming failed: {_be}", file=sys.stderr)

    # Entity sweep: for a user-named entity that isn't too common, keep ALL its exact hits
    # for fetching — the target passage is sometimes the lowest-ranked hit of that entity
    # (e.g. the Bentley/Waldo deposition is the last of ~40 'Waldo' hits).
    sweep_ids: List[int] = []
    # Ultra-rare anchors (<=3 hits) are almost surely the target -> guarantee their fetch.
    for ids, lbl in list(zip(result_sets, labels)):
        if lbl.startswith("lex:") and 0 < len(ids) <= ULTRA_RARE_FETCH:
            sweep_ids.extend(ids)
    # Entity sweep: fetch ALL hits of the single most DISCRIMINATING user entity (smallest
    # true corpus count that is still bounded). Sweeping a common entity (e.g. "Bentley",
    # hundreds of hits) is useless and crowds out the target; sweeping the bounded one
    # ("Waldo", ~40) guarantees the low-ranked deposition is fetched.
    anchor_low = {str(a).strip().lower() for a in plan.get("anchors", [])}
    entity_counts: List[tuple] = []
    for ent in plan.get("entities", []):
        ent = str(ent).strip()
        if not ent or len(ent) < 3:
            continue
        term = max((t for t in re.split(r"[,\s]+", ent) if len(t) >= 3), key=len, default=ent)
        cnt = _true_lexical_count(conn, term, cols)
        if 0 < cnt <= ENTITY_SWEEP_CAP:
            entity_counts.append((cnt, term))
    if entity_counts:
        entity_counts.sort()  # smallest (most discriminating) first
        for _cnt, term in entity_counts[:1]:  # sweep only the most discriminating entity
            ids = next((s for s, l in zip(result_sets, labels) if l == f"lex:{term}"), None)
            if ids is None:
                ids = _lexical(term)
            if ids:
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

    # --- v14 coverage-first fetch (roster/count/aggregation only) ---
    # Guarantee per-collection representation so a high-volume collection can't crowd out
    # evidence spread across the corpus (the Vassiliev roster problem). Guardrails from the
    # V7 post-mortem: NO per-span LLM grading (FTS rank), NO concordance expansion in the
    # sweep (avoids V/V bias — the common noun + real names live everywhere), small
    # per-collection cap, round-robin fetch, hard total cap. Gated to aggregation intent so
    # normal queries are never diluted.
    coverage_diag = {}
    coverage_rows = []
    coverage_counts: Dict[str, int] = {}
    if coverage_mode:
        targets = _coverage_targets(plan, content_anchors)
        coverage_rows = _coverage_sweep(conn, targets, scope,
                                        per_collection=COVERAGE_PER_COLLECTION, verbose=verbose)
        coverage_counts = _coverage_counts(conn, targets, scope)
        if coverage_rows:
            merge_catalog_hits(workspace, _load_catalog(conn, [r[0] for r in coverage_rows][:PRIME_TOP_K]))
            coverage_diag = {"targets": targets, "collections": coverage_counts}

    # Auto-fetch into fulltext + summarize -> evidence bullets from turn 0.
    sweep_set = set(sweep_ids)
    if coverage_mode and coverage_rows:
        # Richness-weighted allocation across collections (deepen rich sources like the
        # Vassiliev notebooks, floor 1 for the long tail, cap so none dominates), then top up
        # with the most-relevant agreement chunks.
        rr = _weighted_coverage_fetch(coverage_rows, coverage_counts,
                                      COVERAGE_MAX_FETCH, COVERAGE_PER_COLLECTION_FETCH_MAX)
        fetch_ids = list(dict.fromkeys([*rr, *[c for c in ranked_ids if c in sweep_set], *ranked_ids[:K_PRIME_FETCH]]))[:COVERAGE_MAX_FETCH]
    elif sweep_set:
        swept_ranked = [c for c in ranked_ids if c in sweep_set]
        fetch_ids = list(dict.fromkeys([*swept_ranked, *ranked_ids[:K_PRIME_FETCH]]))[:MAX_EXHAUSTIVE_FETCH]
    else:
        fetch_ids = ranked_ids[:MAX_PRIME_FETCH]
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

    # Bridge codenames that appear in the primed evidence to their real people (unambiguous
    # only) so the summarizer/synthesis names them, not just the post-hoc finalize.
    try:
        _seed_codenames_from_evidence(conn, workspace, fetched, plan, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"  [V13] prime codename seeding failed: {e}", file=sys.stderr)

    summarized = 0
    if fetched:
        try:
            alias_ctx = build_alias_context_for_summarizer(workspace)
            cdm = build_chunk_doc_map(workspace)
            all_bullets = []
            # Summarize in batches so an exhaustive sweep's lowest-ranked (but on-topic)
            # chunks are still read by the summarizer.
            for i in range(0, len(fetched), SUMMARY_BATCH):
                batch = fetched[i:i + SUMMARY_BATCH]
                ev = summarize_delta_chunks(batch, workspace.question, alias_context=alias_ctx)
                merge_evidence_summary_update(workspace, ev, cdm)
                all_bullets.extend(ev.bullets)
                for c in batch:
                    workspace._summarized_chunk_ids.add(c.chunk_id)
            summarized = len(all_bullets)
            if progress_callback and all_bullets:
                progress_callback("evidence_update", "completed",
                                  f"Primed {len(all_bullets)} evidence bullets", {
                                      "bullets": [{"text": b.text, "tags": b.tags,
                                                   "chunk_ids": b.supporting_chunk_ids,
                                                   "doc_ids": b.doc_ids} for b in all_bullets],
                                      "total_bullet_count": len(all_bullets),
                                  })
        except Exception as e:
            if verbose:
                print(f"  [V13] prime summarize failed: {e}", file=sys.stderr)

    fetched_colls = sorted({(c.collection_slug or c.source_label or "?") for c in fetched})
    workspace.notes.append(
        f"[V13 priming] intent={intent}, seeded {len(catalog)} catalog hits, "
        f"fetched {len(fetched)} chunks across {len(fetched_colls)} collections, "
        f"{summarized} bullets; coverage={coverage_mode}"
    )
    if coverage_mode and coverage_diag:
        # Coverage accounting (honesty): record which collections had target evidence.
        workspace.notes.append(
            f"[V14 coverage] target={coverage_diag.get('targets')} found in "
            f"{len(coverage_diag.get('collections', {}))} collections: {coverage_diag.get('collections')}"
        )
    if verbose:
        print(
            f"  [V13] primed: sets={len(result_sets)} ({labels}), catalog={len(catalog)}, "
            f"fetched={len(fetched)} across {fetched_colls}, bullets={summarized}, "
            f"coverage_mode={coverage_mode}",
            file=sys.stderr,
        )
    return {"primed": len(catalog), "fetched": len(fetched), "bullets": summarized,
            "labels": labels, "top_ids": ranked_ids[:12], "hop_terms": hop_terms,
            "coverage": coverage_diag, "fetched_collections": fetched_colls}


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


def _coverage_targets(plan: Dict[str, Any], content_anchors: List[str]) -> List[str]:
    """Normalized common nouns to enumerate across the corpus.

    Prefer the planner's target_synonyms (a canonical category set — 'reporters',
    'newspapermen', 'correspondents' all normalize to the SAME set), so the coverage sweep
    is stable across query wording. Falls back to the enumeration target + a content anchor.
    """
    # Ultra-generic nouns match everything ("press conference", any "editor"/"agent"/"source")
    # and inflate unrelated collections — drop them from the coverage sweep.
    _GENERIC = {"editor", "editors", "press", "agent", "agents", "source", "sources", "spy",
                "spies", "operative", "operatives", "informant", "informants", "official",
                "officials", "member", "members", "person", "people", "contact", "contacts",
                "asset", "assets", "figure", "figures", "worker", "workers"}
    syns = [str(s).strip() for s in (plan.get("target_synonyms") or [])
            if str(s).strip() and str(s).strip().lower() not in _GENERIC]
    if syns:
        return list(dict.fromkeys(syns))[:6]
    targets: List[str] = []
    et = (plan.get("enumeration_target") or "").strip()
    if et:
        targets.append(et)
    for a in content_anchors:
        a = str(a).strip()
        if a and len(a.split()) == 1 and a.lower() not in " ".join(targets).lower():
            targets.append(a)
    return targets[:3]


def _coverage_counts(conn, targets: List[str], scope) -> Dict[str, int]:
    """Per-collection match count for the target term(s) — the (sqrt-dampened) weighting
    signal for fetch depth toward richer collections. Wording is stabilised upstream by
    normalising the target to a canonical synonym set (target_synonyms), so this count is
    the same regardless of phrasing; the per-collection fetch cap bounds any one collection.
    (DISTINCT-doc counting was tried but over-rewarded large multi-doc FBI case files that
    mention the term in passing; entity_mentions person-density would be ideal but is
    unpopulated for most collections.)"""
    variants: List[str] = []
    for t in targets:
        variants.extend(_enum_variants(t))
    variants = list(dict.fromkeys(variants))
    if not variants:
        return {}
    tsq = " | ".join(variants)
    cols = scope.collections if (scope and scope.collections) else None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM information_schema.columns WHERE table_name='chunks' AND column_name='tsv_simple'")
            tsv = "tsv_simple" if cur.fetchone() else "tsv"
            params: Dict[str, Any] = {"q": tsq}
            scope_clause = ""
            if cols:
                scope_clause = "AND cm.collection_slug = ANY(%(cols)s)"
                params["cols"] = list(cols)
            cur.execute(f"""
                SELECT cm.collection_slug, count(*)
                FROM chunks c JOIN chunk_metadata cm ON cm.chunk_id=c.id AND cm.pipeline_version=c.pipeline_version
                WHERE c.{tsv} @@ to_tsquery('simple', %(q)s) {scope_clause}
                GROUP BY cm.collection_slug
            """, params)
            return {r[0]: int(r[1]) for r in cur.fetchall()}
    except Exception:
        try: conn.rollback()
        except Exception: pass
        return {}


def _weighted_coverage_fetch(rows: List[tuple], counts: Dict[str, int], cap: int, per_col_max: int) -> List[int]:
    """Allocate `cap` fetch slots across the TOP-N richest collections, EQUALLY (round-robin).

    Why equal, not term-frequency-weighted: raw term count inverts roster richness — an FBI
    file ABOUT one journalist ("reporter" on every page) out-counts the Vassiliev notebooks
    that list dozens of distinct journalist-agents. Without a person-density signal
    (entity_mentions is unpopulated), equal depth among the richest collections is the robust,
    wording-invariant choice: every strong source (Vassiliev, Venona, SISS, HUAC, the FBI case
    files) is read to the same depth, and no single-subject file can dominate the roster.
    The top-N filter keeps the budget on genuine sources, not a 1-chunk long tail."""
    from collections import OrderedDict
    by_coll: "OrderedDict[str, List[int]]" = OrderedDict()
    for cid, coll in rows:
        by_coll.setdefault(coll or "?", []).append(cid)
    present = list(by_coll.keys())
    if not present:
        return []
    # Keep the N richest collections by true match count (concentrate budget on real sources).
    present = sorted(present, key=lambda c: -counts.get(c, len(by_coll[c])))[:max(1, COVERAGE_MAX_COLLECTIONS)]
    kept = OrderedDict((c, by_coll[c][:per_col_max]) for c in present)
    # Equal round-robin across the kept collections until the cap is hit.
    out: List[int] = []
    while len(out) < cap and any(kept.values()):
        for c in list(kept.keys()):
            if kept[c]:
                out.append(kept[c].pop(0))
                if len(out) >= cap:
                    break
    return out[:cap]


def _enum_variants(term: str) -> List[str]:
    """Crude singular/plural variants of a target noun for the 'simple' (non-stemming) FTS."""
    w = re.sub(r"[^a-z]", "", (term or "").lower())
    if len(w) < 3:
        return []
    out = {w, (w[:-1] if w.endswith("s") else w + "s")}
    return [x for x in out if len(x) >= 3]


def _coverage_sweep(conn, targets: List[str], scope, *, per_collection: int, verbose: bool = False):
    """Coverage-first retrieval: top-N chunks PER collection matching the enumeration target,
    ranked within each collection by FTS relevance. One SQL, guarantees representation from
    every collection that actually contains the target term — so a high-volume collection
    can't crowd out evidence spread across the corpus (the Vassiliev roster problem).

    Returns [(chunk_id, collection_slug)] ordered by collection then in-collection rank.
    Deliberately NOT concordance-expanded: the target is a common noun ("journalists"), and
    real names/topics live across ALL collections, so a plain per-collection FTS is right here.
    """
    variants: List[str] = []
    for t in targets:
        variants.extend(_enum_variants(t))
    variants = list(dict.fromkeys(variants))
    if not variants:
        return []
    tsq = " | ".join(variants)
    cols = scope.collections if (scope and scope.collections) else None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.columns WHERE table_name='chunks' AND column_name='tsv_simple'"
            )
            tsv = "tsv_simple" if cur.fetchone() else "tsv"
    except Exception:
        try: conn.rollback()
        except Exception: pass
        tsv = "tsv"
    params: Dict[str, Any] = {"q": tsq, "per": per_collection}
    scope_clause = ""
    if cols:
        scope_clause = "AND cm.collection_slug = ANY(%(cols)s)"
        params["cols"] = list(cols)
    sql = f"""
        WITH matched AS (
            SELECT c.id AS chunk_id, cm.collection_slug,
                   ts_rank_cd(c.{tsv}, to_tsquery('simple', %(q)s)) AS rank
            FROM chunks c
            JOIN chunk_metadata cm ON cm.chunk_id = c.id AND cm.pipeline_version = c.pipeline_version
            WHERE c.{tsv} @@ to_tsquery('simple', %(q)s) {scope_clause}
        ),
        ranked AS (
            SELECT chunk_id, collection_slug, rank,
                   ROW_NUMBER() OVER (PARTITION BY collection_slug ORDER BY rank DESC, chunk_id) AS rn
            FROM matched
        )
        SELECT chunk_id, collection_slug FROM ranked WHERE rn <= %(per)s
        ORDER BY collection_slug, rn
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    except Exception as e:
        if verbose:
            print(f"  [V14] coverage_sweep error: {e}", file=sys.stderr)
        try: conn.rollback()
        except Exception: pass
        return []
    if verbose:
        from collections import Counter as _C
        by_coll = _C(r[1] for r in rows)
        print(f"  [V14] coverage_sweep '{tsq}': {len(rows)} chunks across {len(by_coll)} collections {dict(by_coll)}",
              file=sys.stderr)
    return rows


def _true_lexical_count(conn, term: str, cols) -> int:
    """Approximate corpus frequency of a term (scoped), to pick the most discriminating
    entity to sweep. Uses a scoped ILIKE count — cheap and good enough for ranking."""
    try:
        params: List[Any] = [f"%{term}%"]
        where = "COALESCE(c.clean_text, c.text) ILIKE %s"
        if cols:
            where += " AND cm.collection_slug = ANY(%s)"
            params.append(list(cols))
        # Bounded count: stop after ENTITY_SWEEP_CAP+1 matches so a common term over the full
        # archive doesn't trigger a slow full seq-scan (we only need "<= cap or not").
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT count(*) FROM (
                        SELECT 1 FROM chunks c
                        JOIN chunk_metadata cm ON cm.chunk_id = c.id AND cm.pipeline_version = c.pipeline_version
                        WHERE {where} LIMIT %s
                    ) t""",
                tuple(params) + (ENTITY_SWEEP_CAP + 1,),
            )
            return int(cur.fetchone()[0])
    except Exception:
        try: conn.rollback()
        except Exception: pass
        return 10**9  # treat as too-common on error (skip sweep)


_CODENAME_CUE_RE = re.compile(
    r"(?:codenamed?|cover[- ]?name|cover|alias(?:es)?|called|known as)\s+"
    r"['\"“]?([A-Z][A-Za-z'’\-]{2,20})", re.IGNORECASE)


def _resolve_query_codenames(conn, plan, question: str, *, verbose: bool = False) -> Dict[str, str]:
    """Resolve codenames NAMED IN THE QUERY to their real person via the concordance, and inject
    the real name into the retrieval plan (entities + a query) so retrieval reaches the RIGHT
    person's collection instead of tunneling onto a decoy.

    This fixes the "confident wrong entity" trap for codename lookups (e.g. "who was codenamed
    'Liberal'"): without the real name in the retrieval, the loop may reach only a same-codename
    decoy (the 1930s journalist 'Liberal' = Frank Palmer) and never the dominant referent (Julius
    Rosenberg). We pick the DOMINANT referent (the person for whom this codename is one of the
    richest-attested aliases) — the sensible default for "who was codename X". The evidence-time
    resolver + grounded_finalize still disambiguate by profile as a safety net.
    """
    # Candidate codename surfaces: cue-marked ('codenamed X', "cover name X"), quoted names, and
    # ALL-CAPS / single-Capitalized plan entities+anchors. Require a CUE or quotes or ALL-CAPS so
    # ordinary words aren't resolved.
    surfaces = set()
    q = question or ""
    for m in _CODENAME_CUE_RE.finditer(q):
        surfaces.add(m.group(1).strip())
    for m in re.findall(r"['\"“]([A-Z][A-Za-z'’\-]{2,20})['\"”]", q):
        surfaces.add(m.strip())
    for a in (plan.get("entities") or []) + (plan.get("anchors") or []):
        a = str(a).strip()
        if a.isupper() and 2 < len(a) < 20 and a.isalpha():
            surfaces.add(a)
    surfaces = {s for s in surfaces if 2 < len(s) < 22 and s.lower() not in _FRAMING}
    if not surfaces:
        return {}
    resolved: Dict[str, str] = {}
    for surface in list(surfaces)[:3]:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT e.canonical_name, count(DISTINCT ea2.alias) AS naliases
                    FROM entity_aliases ea
                    JOIN entities e ON e.id = ea.entity_id
                    LEFT JOIN entity_aliases ea2 ON ea2.entity_id = e.id
                    WHERE lower(ea.alias) = lower(%s)
                      AND COALESCE(ea.is_matchable, true)
                      AND e.entity_type = 'person'
                      AND length(e.canonical_name) BETWEEN 3 AND 45
                      AND e.canonical_name !~ '[0-9;]'
                    GROUP BY e.id, e.canonical_name
                    ORDER BY naliases DESC
                    LIMIT 2
                """, (surface,))
                rows = cur.fetchall()
        except Exception:
            try: conn.rollback()
            except Exception: pass
            continue
        # Need a real codename bearer (>=2 aliases) and skip self-referential (canonical==surface).
        cands = [(c, n) for c, n in rows if c and c.lower() != surface.lower() and n >= 2]
        if not cands:
            continue
        canon = cands[0][0]
        resolved[surface] = canon
    if not resolved:
        return {}
    # Inject: add each resolved real name to entities + a codename+realname query so retrieval
    # reaches that person's material.
    ents = list(plan.get("entities") or [])
    qs = list(plan.get("queries") or [])
    for surface, canon in resolved.items():
        if canon not in ents:
            ents.append(canon)
        surname = canon.split(",")[0].split()[-1] if canon else canon
        qs.insert(0, f"{surname} {surface}")
    plan["entities"] = ents
    plan["queries"] = qs
    if verbose:
        print(f"  [V13] query codename(s) resolved -> "
              + ", ".join(f"{s}={c}" for s, c in resolved.items()), file=sys.stderr)
    return resolved


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
                    LIMIT 1
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


def _seed_codenames_from_evidence(conn, workspace, fetched, plan, *, verbose: bool = False) -> None:
    """Bridge codenames PRESENT in the primed evidence to their real people and merge them into
    the workspace, so the main summarizer/synthesis (not just the post-hoc finalize) names the
    real person. Only UNAMBIGUOUS resolutions are seeded (after corroboration-narrowing), so an
    ambiguous codename (Liberal -> two people) is left for the finalize step to disambiguate.
    Generalizable: any codename the archive concordance resolves and the evidence corroborates.
    """
    from retrieval.agent.v9_workspace import merge_entities
    try:
        codenames = resolve_codenames_in_evidence(
            conn, fetched, query_terms=(plan.get("entities") or []) + (plan.get("anchors") or []))
    except Exception:
        return
    unambig = {s: c[0] for s, c in codenames.items() if len(c) == 1}
    if not unambig:
        return
    seeded = 0
    for surface, canonical in list(unambig.items())[:12]:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT e.id, e.canonical_name,
                           array_agg(DISTINCT ea.alias) FILTER (WHERE ea.alias IS NOT NULL)
                    FROM entities e
                    LEFT JOIN entity_aliases ea ON ea.entity_id = e.id
                    WHERE e.canonical_name = %s AND e.entity_type = 'person'
                    GROUP BY e.id, e.canonical_name
                    ORDER BY COALESCE(array_length(array_agg(DISTINCT ea.alias), 1), 0) DESC
                    LIMIT 1
                """, (canonical,))
                row = cur.fetchone()
        except Exception:
            try: conn.rollback()
            except Exception: pass
            continue
        if not row:
            continue
        eid, canon, aliases = row
        al = [a for a in (aliases or []) if a and len(a) >= 2]
        if surface not in al:
            al = [surface] + al
        merge_entities(workspace, [WorkspaceEntity(entity_id=eid, canonical_name=canon, aliases=al[:24])])
        seeded += 1
    if verbose and seeded:
        print(f"  [V13] seeded {seeded} evidence codename bridge(s): "
              + ", ".join(f"{s}->{c}" for s, c in list(unambig.items())[:6]), file=sys.stderr)


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
    r"\b(no (?:explicit |direct |specific )?(?:evidence|mention|description|reference|record|"
    r"indication|data|information|figures?|numbers?|detail)|"
    r"(?:does|do|is|are|were|was) not (?:\w+ly\s+)?(?:contain|indicate|mention|document|specify|"
    r"provide|detail|name|identif|reveal|appear|found|find)|there(?:'s| is| are)? no|"
    r"not (?:\w+ly\s+)?(?:find|found|contain|mention|mentioned|documented|detailed|named|"
    r"specified|available|identified|revealed|provided)|reveals? no|could(?:n't| not) find|"
    r"nothing (?:to|that|was)|no (?:such )?(?:mention|record))",
    re.IGNORECASE,
)


# Cosmetic annotations the model sometimes appends to an already-bridged name, e.g.
# "Rabinovich (unresolved codename)" / "Sound [AMBIGUOUS]" — contradictory noise.
_CODENAME_NOISE_RE = re.compile(
    r"\s*(?:\((?:un)?resolved[^)]*\bcodename\b[^)]*\)|\(codename[^)]*unresolved[^)]*\)|"
    r"\((?:un)?resolved(?:\s+codename)?\)|\[AMBIGUOUS\]|\(tentative\))",
    re.IGNORECASE,
)


def _scrub_codename_noise(text: str) -> str:
    if not text:
        return text
    return _CODENAME_NOISE_RE.sub("", text)


def apply_anti_false_negative(result, workspace, *, verbose: bool = False) -> None:
    """Never let 'not retrieved' become a confident 'no evidence exists'.

    - Scrub cosmetic "(unresolved codename)"/"[AMBIGUOUS]" tags the model appends to names
      it has actually bridged.
    - If the answer asserts non-existence, cap sufficiency and reword to 'not found in what
      I searched' (with the queries actually run), and surface any adjacent evidence bullets.
    - Strip citations from negative claims (a citation must support an affirmative statement).
    """
    synth = getattr(result, "synthesis", None)

    # Scrub cosmetic codename annotations from the narrative + claim texts.
    if synth is not None and getattr(synth, "narrative", None):
        synth.narrative = _scrub_codename_noise(synth.narrative)
    if getattr(result, "narrative", None):
        try:
            result.narrative = _scrub_codename_noise(result.narrative)
        except Exception:
            pass
    for c in (getattr(result, "claims", None) or []):
        inner = getattr(c, "claim", None)
        if inner is not None and getattr(inner, "text", None):
            inner.text = _scrub_codename_noise(inner.text)
        elif getattr(c, "text", None):
            try: c.text = _scrub_codename_noise(c.text)
            except Exception: pass

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


_ROSTER_MODEL = os.getenv("V14_ROSTER_MODEL", "gpt-4.1-mini-2025-04-14")


def _boolean_pool_chunk_ids(conn, workspace, cap: int = 500):
    """Chunk ids of the primed boolean pool(s) for this run (roster mining input)."""
    sets = getattr(workspace, "_boolean_result_sets", None) or []
    ids = []
    for s in sets:
        rsid = s.get("result_set_id")
        if not rsid:
            continue
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT chunk_id FROM search_result_page_hits WHERE result_set_id = %s "
                    "ORDER BY collection_id, document_id, page_seq LIMIT %s",
                    (rsid, cap),
                )
                ids.extend(r[0] for r in cur.fetchall())
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
    return list(dict.fromkeys(ids))[:cap]



def assemble_roster(conn, workspace, target: str, *, question: str = "", plan=None,
                    verbose: bool = False):
    """Enumerate distinct target-people (e.g. journalists) from the corpus-wide FETCHED
    CHUNKS into roster entries with citations.

    Reads the chunk text directly in small batches (where "Grin – John Spivak, journalist"
    literally appears) rather than the batch-summary bullets — so it captures named people
    across the whole fetched set. Citations are attributed deterministically (a chunk cites
    an entry only if its text actually contains the name/codename). Bounded to a few
    gpt-4.1-mini calls; only runs for roster intent, where breadth is the whole point.

    When the question names a SPECIFIC scope (a ring/operation/traffic, e.g. "Silvermaster
    ring", "ENORMOZ traffic"), the roster is constrained to people the passages connect to
    that scope — so a scoped roster ("who was in the Silvermaster ring") does not over-broaden
    into every Soviet agent in the corpus. Open rosters ("which journalists were spies") keep
    full breadth.
    """
    from retrieval.agent.v9_types import GroundedClaim, V9Claim
    from retrieval.agent.v9_workspace import build_alias_context_for_summarizer
    from collections import OrderedDict
    api_key = os.getenv("OPENAI_API_KEY")
    all_chunks = [c for c in workspace.fulltext_chunks if (c.text or "").strip()]
    if not all_chunks or not api_key:
        return []
    # Balance the roster read across collections so a collection the AGENT over-fetched
    # (e.g. a file that is ABOUT one journalist) can't dominate the roster — makes the
    # output robust to query wording. Cap per collection, preserve fetch order.
    _ROSTER_PER_COLL = int(os.getenv("V14_ROSTER_PER_COLLECTION", "5"))
    _by: "OrderedDict[str, List]" = OrderedDict()
    for c in all_chunks:
        _by.setdefault(c.collection_slug or c.source_label or "?", []).append(c)
    chunks = []
    for coll, cs in _by.items():
        chunks.extend(cs[:_ROSTER_PER_COLL])
    tgt = (target or "person").strip() or "person"
    # Scope anchors: distinctive tokens in the question (ENORMOZ, Silvermaster, a ring/operation)
    # that are NOT the generic target noun. When present, this is a SCOPED roster — prioritise the
    # chunks that actually mention the scope and tell the model to keep only people tied to it, so
    # a scoped roster ("Silvermaster ring", "ENORMOZ traffic") does not over-broaden corpus-wide.
    scope_anchors = []
    for a in ((plan or {}).get("anchors") or []):
        a = str(a).strip()
        al = a.lower()
        if (len(a) >= 4 and al not in _FRAMING and al != tgt.lower()
                and al not in {"soviet", "agent", "agents", "spy", "espionage", "traffic"}):
            scope_anchors.append(a)
    if scope_anchors:
        _sa = [s.lower() for s in scope_anchors]
        _hit = [c for c in chunks if any(s in (c.text or "").lower() for s in _sa)]
        # Only treat as scoped if the scope actually appears in the evidence (else keep breadth).
        if len(_hit) >= 3:
            _miss = [c for c in chunks if c not in _hit]
            chunks = _hit + _miss  # scope-matching chunks read first; rest retained for context
    alias_ctx = build_alias_context_for_summarizer(workspace)
    # Codename key from the roster evidence so CHARLES->Fuchs, MLAD->Hall, etc. resolve to real
    # people. Generalizable — the same concordance bridge used by grounded_finalize.
    try:
        codekey = _codename_key_text(resolve_codenames_in_evidence(conn, chunks))
    except Exception:
        codekey = ""
    # A GENERIC espionage target ("agent", "source", "spy", "member", "person") means "enumerate
    # every Soviet-linked person"; a SPECIFIC role ("journalist", "engineer", "scientist") also
    # requires the person to hold that role. This replaces the old journalist-hardcoded prompt so
    # the roster works for any category (atomic agents, engineers, diplomats, ...).
    _GENERIC_TGT = {"person", "people", "agent", "agents", "source", "sources", "spy", "spies",
                    "asset", "assets", "contact", "contacts", "operative", "operatives",
                    "member", "members", "figure", "figures", "individual", "individuals",
                    "soviet agent", "soviet agents", "soviet spy", "soviet source"}
    tgt_generic = tgt.lower() in _GENERIC_TGT
    if tgt_generic:
        role_rule = ("Extract every distinct person the passages describe as a Soviet spy, agent, "
                     "source, contact, operative, or intelligence asset.")
    else:
        role_rule = (f"Extract every distinct person the passages describe as a {tgt} AND as a "
                     f"Soviet spy, agent, source, contact, or asset. Only include people the text "
                     f"actually presents as a {tgt}.")
    # GROUNDING RULES (all rosters): keep the list to the recruited network, faithful to the
    # passages — this is what fixes the "handlers/targets miscast as agents" and "asserts people
    # from background knowledge" failures. Generalizable across any roster category.
    grounding_rules = (
        " GROUNDING RULES: (a) KEEP anyone the passages show ACTING AS AN AGENT — recruited as a "
        "source/member, or who PROVIDED, PASSED, PHOTOGRAPHED, STOLE, or supplied material/"
        "information. EXCLUDE people described SOLELY as their Soviet handlers/case officers/"
        "rezidents/intelligence-officer staff who only RAN, SUPERVISED, or RECEIVED from others "
        "(e.g. NKVD/GRU officers, station chiefs, the Moscow Center), UNLESS the question asks for "
        "handlers/officers. When in doubt (a person both provided material AND ran others), KEEP "
        "them. (b) EXCLUDE people only TARGETED, approached, proposed, or cultivated for "
        "recruitment whom the passages do NOT confirm actually became agents/sources. (c) Every "
        "person must be supported by a passage tying them to Soviet intelligence; do NOT add "
        "well-known figures from your own background knowledge.")
    role_rule += grounding_rules
    # Scoped roster: require each person to be tied to the specific subject of the question, so a
    # named-ring/operation roster stays focused (drops corpus-wide Soviet agents unrelated to it).
    if scope_anchors and _hit and len(_hit) >= 3:
        scope_desc = question.strip() or (", ".join(scope_anchors))
        role_rule += (f" IMPORTANT SCOPE: include ONLY people the passages specifically connect to "
                      f"the subject of this question — \"{scope_desc}\". A person merely described "
                      f"as a Soviet agent elsewhere, with no tie to this specific "
                      f"{'/'.join(scope_anchors[:3])}, must be EXCLUDED. If a passage places a "
                      f"person in a DIFFERENT named ring/group/network (e.g. the Ware group, the "
                      f"Silvermaster group, the Rosenberg ring) than the one asked about, EXCLUDE "
                      f"them — do NOT merge separate rings together.")
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    entries: Dict[str, Dict[str, Any]] = {}
    BATCH = 10
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        doc_text = "\n\n".join(
            f"[chunk {c.chunk_id} | {(c.source_label or '').replace('_',' ')} {c.page or ''}]\n{(c.text or '')[:900]}"
            for c in batch
        )
        try:
            r = client.chat.completions.create(
                model=_ROSTER_MODEL, temperature=0.0, max_completion_tokens=900,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": (
                        f"{role_rule} Convert codenames to real names using the CODENAME KEY and "
                        f"identity key; keep the codename in the 'codename' field. If a person "
                        f"appears only under a codename absent from the key, keep the codename as "
                        f"the name (do not invent a real name). Do NOT invent people. If none, "
                        f"return an empty list.\n"
                        + (codekey + "\n" if codekey else "")
                        + f"{alias_ctx}\n"
                        f'Return JSON {{"people":[{{"real_name":"","codename":"","role":""}}]}}.')},
                    {"role": "user", "content": doc_text},
                ],
            )
            people = json.loads(r.choices[0].message.content or "{}").get("people", [])
        except Exception as e:
            if verbose:
                print(f"  [V14] roster batch failed: {e}", file=sys.stderr)
            continue
        for p in people:
            name = (p.get("real_name") or "").strip()
            code = (p.get("codename") or "").strip()
            role = (p.get("role") or "").strip()
            if not name and not code:
                continue
            key = (name or code).lower()
            e = entries.setdefault(key, {"name": name, "code": code, "role": role, "cids": []})
            if name and not e["name"]:
                e["name"] = name
            if code and not e["code"]:
                e["code"] = code
            if role and not e["role"]:
                e["role"] = role
            # Deterministic citation: cite batch chunks whose text mentions the name/codename OR
            # the person's surname (so a true member the passage cites by surname/codename is not
            # demoted to "Unverified" while a full-name distractor is asserted — the NT4 inversion).
            needles = [n for n in (name, code) if n and len(n) >= 3]
            surname = ""
            if name:
                toks = [t for t in re.split(r"[,\s]+", name) if len(t) >= 4 and t.isalpha()]
                surname = toks[-1] if toks else ""
            if surname and surname.lower() not in {n.lower() for n in needles}:
                needles.append(surname)
            for c in batch:
                txt = (c.text or "").lower()
                if any(n.lower() in txt for n in needles):
                    if c.chunk_id not in e["cids"]:
                        e["cids"].append(c.chunk_id)

    # Pool mining: exhaustive roster read over the primed boolean pool — membership
    # becomes structural (every pool page read once) instead of fetch-sampled, which
    # is what made Halperin (295 matching pages) missable. FRIDAY_POOL_MINING=0 off.
    if os.getenv("FRIDAY_POOL_MINING", "1") == "1":
        try:
            _pool_ids = _boolean_pool_chunk_ids(conn, workspace)
            if _pool_ids:
                from retrieval.agent.v11_tools import mine_pool
                _spec = (role_rule[:1500]
                         + " Report each person's name EXACTLY as written in the passage.")
                _mres = mine_pool(conn, _pool_ids, _spec,
                                  question or workspace.question, verbose=verbose)
                for _me in _mres.get("entries", []):
                    _nm = (_me.get("canonical") or "").strip()
                    if len(_nm) < 3:
                        continue
                    _key = _nm.lower()
                    _e = entries.setdefault(_key, {"name": _nm, "code": "",
                                                   "role": "", "cids": []})
                    if not _e["role"] and _me.get("roles"):
                        _e["role"] = _me["roles"][0]
                    for _c in _me.get("citations", []):
                        if _c["chunk_id"] not in _e["cids"]:
                            _e["cids"].append(_c["chunk_id"])
                if verbose:
                    print(f"  [V14] pool-mine merged: {len(_mres.get('entries', []))} candidates "
                          f"from {_mres.get('mined')}/{_mres.get('total_pool')} pool chunks "
                          f"(cutoff={_mres.get('cutoff')})", file=sys.stderr)
        except Exception as _me_err:
            if verbose:
                print(f"  [V14] pool mining failed: {_me_err}", file=sys.stderr)

    # Fuzzy-dedup entry keys (mined canonical "Duncan Chaplin Lee" vs fetched
    # "Duncan Lee") — merge citations into the richer-cited survivor.
    try:
        from difflib import SequenceMatcher
        _keys = sorted(entries.keys(), key=lambda k: -len(entries[k]["cids"]))
        _alias_of = {}
        for _k in _keys:
            _ktoks = set(_k.split())
            for _kept in list(_alias_of.values()):
                if _k == _kept:
                    continue
                _ptoks = set(_kept.split())
                if (SequenceMatcher(None, _k, _kept).ratio() >= 0.82
                        or (len(_ktoks) >= 2 and (_ktoks <= _ptoks or _ptoks <= _ktoks))):
                    _alias_of[_k] = _kept
                    break
            else:
                _alias_of[_k] = _k
        _merged_entries = {}
        for _k, _tgt in _alias_of.items():
            _src = entries[_k]
            _dst = _merged_entries.setdefault(_tgt, {"name": "", "code": "", "role": "", "cids": []})
            for _f in ("name", "code", "role"):
                if _src.get(_f) and not _dst.get(_f):
                    _dst[_f] = _src[_f]
            for _cid in _src["cids"]:
                if _cid not in _dst["cids"]:
                    _dst["cids"].append(_cid)
        entries = _merged_entries
    except Exception:
        pass

    out = []
    for e in entries.values():
        cids = e["cids"][:3]
        label = e["name"] or e["code"]
        if e["name"] and e["code"] and e["code"].lower() not in e["name"].lower():
            label = f'{e["name"]} ("{e["code"]}")'
        text = f'{label} — {e["role"]}' if e["role"] else label
        out.append(GroundedClaim(
            claim=V9Claim(text=text[:200], confidence="medium", requires_citation=True),
            status="grounded" if cids else "weak", citation_chunk_ids=cids,
        ))
    # Grounded (cited) entries first.
    out.sort(key=lambda gc: 0 if gc.citation_chunk_ids else 1)
    if verbose:
        print(f"  [V14] roster assembled: {len(out)} {tgt} entries "
              f"({sum(1 for gc in out if gc.citation_chunk_ids)} cited)", file=sys.stderr)
    return out


def assemble_group_roster(conn, workspace, *, question: str = "", verbose: bool = False):
    """Enumerate NAMED espionage networks/groups/rings from the fetched chunks — the right tool
    for "what networks operated" (a group question), where a person-enumerator wrongly returns a
    list of individuals. Extracts each distinct named group + its leader/organizer, grounded in a
    passage that names the group. Deterministic citations; bounded gpt-4.1-mini calls.
    """
    from retrieval.agent.v9_types import GroundedClaim, V9Claim
    from collections import OrderedDict
    api_key = os.getenv("OPENAI_API_KEY")
    all_chunks = [c for c in workspace.fulltext_chunks if (c.text or "").strip()]
    if not all_chunks or not api_key:
        return []
    _PER_COLL = int(os.getenv("V14_ROSTER_PER_COLLECTION", "5"))
    _by: "OrderedDict[str, List]" = OrderedDict()
    for c in all_chunks:
        _by.setdefault(c.collection_slug or c.source_label or "?", []).append(c)
    chunks = []
    for coll, cs in _by.items():
        chunks.extend(cs[:_PER_COLL])
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    entries: Dict[str, Dict[str, Any]] = {}
    BATCH = 10
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        doc_text = "\n\n".join(
            f"[chunk {c.chunk_id} | {(c.source_label or '').replace('_',' ')} {c.page or ''}]\n{(c.text or '')[:900]}"
            for c in batch
        )
        try:
            r = client.chat.completions.create(
                model=_ROSTER_MODEL, temperature=0.0, max_completion_tokens=800,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": (
                        "Extract every distinct NAMED Soviet espionage network/ring/group/apparatus "
                        "the passages describe (e.g. 'the Silvermaster group', 'the Perlo group', "
                        "'the Ware group', 'the Rosenberg ring', 'the Golos-Bentley apparatus'). "
                        "For each, give the group's NAME and its LEADER/organizer and the key "
                        "government agency/domain it penetrated, if the passages state them. Include "
                        "a group ONLY if the passages actually name it as a distinct network; do NOT "
                        "invent groups, and do NOT list individual agents who are not themselves a "
                        "named group. If none, return an empty list. Return JSON "
                        '{"groups":[{"name":"","leader":"","note":""}]}.')},
                    {"role": "user", "content": doc_text},
                ],
            )
            groups = json.loads(r.choices[0].message.content or "{}").get("groups", [])
        except Exception as e:
            if verbose:
                print(f"  [V14] group-roster batch failed: {e}", file=sys.stderr)
            continue
        for g in groups:
            name = (g.get("name") or "").strip()
            leader = (g.get("leader") or "").strip()
            note = (g.get("note") or "").strip()
            if not name or len(name) < 4:
                continue
            key = re.sub(r"\b(the|group|ring|network|apparatus|of)\b", "", name.lower()).strip()
            key = re.sub(r"\s+", " ", key) or name.lower()
            e = entries.setdefault(key, {"name": name, "leader": leader, "note": note, "cids": []})
            if leader and not e["leader"]:
                e["leader"] = leader
            if note and not e["note"]:
                e["note"] = note
            # Cite chunks that mention the group's distinctive token (leader surname or group word).
            needles = []
            lead_surname = ""
            if leader:
                toks = [t for t in re.split(r"[,\s]+", leader) if len(t) >= 4 and t.isalpha()]
                lead_surname = toks[-1] if toks else ""
            for t in re.split(r"[,\s]+", name):
                if len(t) >= 5 and t.lower() not in ("group", "ring", "network", "apparatus"):
                    needles.append(t)
            if lead_surname:
                needles.append(lead_surname)
            for c in batch:
                txt = (c.text or "").lower()
                if any(n.lower() in txt for n in needles):
                    if c.chunk_id not in e["cids"]:
                        e["cids"].append(c.chunk_id)
    out = []
    for e in entries.values():
        cids = e["cids"][:3]
        label = e["name"]
        detail = "; ".join([p for p in (
            (f"led by {e['leader']}" if e["leader"] else ""), e["note"]) if p])
        text = f"{label} — {detail}" if detail else label
        out.append(GroundedClaim(
            claim=V9Claim(text=text[:200], confidence="medium", requires_citation=True),
            status="grounded" if cids else "weak", citation_chunk_ids=cids,
        ))
    out.sort(key=lambda gc: 0 if gc.citation_chunk_ids else 1)
    if verbose:
        print(f"  [V14] group-roster assembled: {len(out)} networks "
              f"({sum(1 for gc in out if gc.citation_chunk_ids)} cited)", file=sys.stderr)
    return out


_FINALIZE_MODEL = os.getenv("V14_FINALIZE_MODEL", "gpt-4.1-mini-2025-04-14")
_YESNO_RE = re.compile(r"^\s*(did|does|do|was|were|is|are|has|have|had|can|could|would|should|will)\b", re.IGNORECASE)


def _answer_bearing_chunks(workspace, plan) -> List:
    """Fetched chunks whose text actually contains a query entity/anchor — the passages most
    likely to hold the answer. Ranked by how many distinct anchors they match."""
    needles = []
    for e in (plan.get("entities") or []):
        for tok in re.split(r"[,\s]+", str(e)):
            if len(tok) >= 4 and tok.lower() not in _FRAMING:
                needles.append(tok.lower())
    for a in (plan.get("anchors") or []):
        a = str(a).strip()
        if a and len(a) >= 4 and a.lower() not in _FRAMING:
            needles.append(a.lower())
    needles = list(dict.fromkeys(needles))
    if not needles:
        return []
    scored = []
    for c in workspace.fulltext_chunks:
        txt = (c.text or "").lower()
        hits = sum(1 for n in needles if n in txt)
        if hits:
            scored.append((hits, c))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:8]]


def _excerpt(text: str, needles: List[str], width: int = 1100) -> str:
    """Window of `text` centred a little before the first needle (so the sentence AFTER the
    name — often the answer — is included), else the head of the chunk."""
    t = text or ""
    if len(t) <= width:
        return t.strip()
    low = t.lower()
    pos = min([low.find(n) for n in needles if n in low] or [0])
    start = max(0, pos - 250)
    return t[start:start + width].strip()


def resolve_codenames_in_evidence(conn, chunks, query_terms=None, *, cap: int = 30):
    """Resolve codename surfaces PRESENT in the retrieved evidence against entity_aliases.

    Returns {surface: [canonical_names]} — the codename->real-person key that synthesis needs.
    Deterministic (DB lookup), generalizable (any codename in the evidence). Only considers
    codename-like surfaces (ALL-CAPS tokens, quoted names) plus the query's own terms, so
    ordinary words that happen to be aliases don't get resolved. Ambiguous codenames (e.g.
    "Liberal" -> Rosenberg AND Frank Palmer) return all candidates so synthesis can disambiguate
    instead of committing to the first hit.
    """
    surfaces = set()
    for t in (query_terms or []):
        t = str(t).strip()
        if 2 < len(t) < 30:
            surfaces.add(t)
    for c in chunks:
        txt = c.text or ""
        # quoted proper names ("Liberal", "Sound") and ALL-CAPS codenames (CHARLES, ENORMOZ)
        for m in re.findall(r'[""“]([A-Z][A-Za-z\'\-\. ]{2,22})[""”]', txt):
            surfaces.add(m.strip())
        for m in re.findall(r'\b([A-Z][A-Z\'\-]{2,15})\b', txt):
            surfaces.add(m.strip())
    surfaces = {s for s in surfaces if 2 < len(s) < 30}
    if not surfaces:
        return {}
    norms = list({s.lower() for s in surfaces})
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT lower(ea.alias) AS a, e.canonical_name
                FROM entity_aliases ea
                JOIN entities e ON e.id = ea.entity_id
                WHERE lower(ea.alias) = ANY(%s)
                  AND COALESCE(ea.is_matchable, true)
                  AND e.entity_type = 'person'
                  AND length(e.canonical_name) BETWEEN 3 AND 45
                  AND e.canonical_name !~ '[0-9;]'
            """, (norms,))
            rows = cur.fetchall()
    except Exception:
        try: conn.rollback()
        except Exception: pass
        return {}
    # surface(lower) -> ordered distinct canonical names (skip self-referential canon==surface)
    by_surface: Dict[str, List[str]] = {}
    surf_display = {s.lower(): s for s in surfaces}
    for a, canon in rows:
        canon = (canon or "").strip()
        if not canon or canon.lower() == a:
            continue
        by_surface.setdefault(a, [])
        if canon not in by_surface[a]:
            by_surface[a].append(canon)
    # Corroboration-based disambiguation: when one codename maps to several people, prefer the
    # candidate(s) whose surname is actually attested in the retrieved evidence. This resolves
    # noisy multi-mappings deterministically (e.g. "Liberal" -> Rosenberg/Palmer/"Man Behind":
    # only Rosenberg's surname appears in the pulled chunks) without guessing. Generalizable —
    # a candidate that no passage corroborates is dropped in favour of one that is.
    blob = " ".join((c.text or "") for c in chunks).lower()
    def _surname(name: str) -> str:
        toks = [t for t in re.split(r"[,\s]+", name) if len(t) >= 4 and t.isalpha()]
        return (toks[-1] if toks else "").lower()
    out = {}
    for a, canons in by_surface.items():
        if len(canons) > 1:
            attested = [c for c in canons if _surname(c) and _surname(c) in blob]
            if attested:
                canons = attested  # narrow to candidates the evidence actually supports
        if canons:
            out[surf_display.get(a, a)] = canons[:4]
    # prefer codenames that actually appear in the evidence text (cap)
    return dict(list(out.items())[:cap])


def _codename_key_text(codenames: Dict[str, List[str]]) -> str:
    """Render the codename->canonical key for the synthesis prompt, flagging ambiguity."""
    if not codenames:
        return ""
    lines = []
    for surf, canons in codenames.items():
        # collapse duplicate-name entities (Jacob Golos x3) to distinct names
        seen, distinct = set(), []
        for c in canons:
            key = c.lower().replace(",", "").replace(" ", "")
            if key not in seen:
                seen.add(key); distinct.append(c)
        if len(distinct) == 1:
            lines.append(f"  {surf} = {distinct[0]}")
        else:
            lines.append(f"  {surf} = {' OR '.join(distinct)}  [AMBIGUOUS — disambiguate by date/context, do not guess]")
    return "CODENAME KEY (from the archive concordance — use these to name real people; a codename may refer to different people in different years):\n" + "\n".join(lines)


def grounded_finalize(conn, result, workspace, plan, question: str, *, verbose: bool = False) -> bool:
    """Answer-faithfulness pass (v14): read the ACTUAL answer-bearing passages and answer the
    question directly and faithfully — fixing (a) needles the main synthesis had but omitted,
    and (b) self-contradictions where the answer negates its own evidence ("a search was
    conducted" -> "no evidence of searches"). Returns True if it replaced the answer.
    """
    from retrieval.agent.v9_types import GroundedClaim, V9Claim
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    chunks = _answer_bearing_chunks(workspace, plan)
    if not chunks:
        return False
    needles = [str(x).lower() for x in ((plan.get("entities") or []) + (plan.get("anchors") or []))]
    label = {}
    passages = []
    for c in chunks:
        src = (c.source_label or c.collection_slug or "").replace("_", " ").title()
        lbl = f"{src} {c.page or ''}".strip()
        label[c.chunk_id] = lbl
        passages.append(f"[chunk {c.chunk_id} | {lbl}]\n{_excerpt(c.text, needles)}")
    # Resolve codenames present in the evidence (+ the query's own terms) so the model can
    # name real people instead of restating codenames or guessing.
    codenames = resolve_codenames_in_evidence(
        conn, chunks, query_terms=(plan.get("entities") or []) + (plan.get("anchors") or []))
    codekey = _codename_key_text(codenames)
    if verbose and codenames:
        print(f"  [V14] grounded_finalize codename key: "
              + "; ".join(f"{k}={'/'.join(v)}" for k, v in list(codenames.items())[:8]), file=sys.stderr)
    is_yesno = bool(_YESNO_RE.match(question or ""))
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        r = client.chat.completions.create(
            model=_FINALIZE_MODEL, temperature=0.0, max_completion_tokens=1000,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": (
                    "Answer the question STRICTLY from the passages. Rules: "
                    "(1) Lead with a direct verdict when the question is yes/no. "
                    "(2) Answer the question AS ASKED — never narrow it with qualifiers it did "
                    "not contain (e.g. 'immediately', 'active', 'direct evidence of'). "
                    "(3) A failed/unsuccessful action still counts: 'a search that found nothing' "
                    "still means a search WAS conducted; 'they looked but didn't find him' = yes, "
                    "they looked. (4) Assert ONLY what a passage supports and cite the [chunk N] "
                    "id(s). (5) USE THE CODENAME KEY (it is AUTHORITATIVE — it comes from the "
                    "archive's own concordance): when a passage refers to a person by a "
                    "codename, name the real person from the key. If the key gives exactly ONE "
                    "real name for the codename the question asks about, STATE that identification "
                    "as your answer even when that real name does not appear verbatim in the "
                    "passages — cite the passage(s) that use the codename. When a codename is marked "
                    "AMBIGUOUS, actively DISAMBIGUATE: take the concrete profile in the passages "
                    "(birth year, recruitment date, recruiter, tradecraft, the agents they "
                    "handled) and pick the ONE key candidate that profile matches — for this "
                    "matching step ONLY you may use well-known historical facts about the "
                    "candidates (e.g. which of them was the radar/atomic spy, which was a "
                    "journalist), then briefly state the basis. Do NOT add other outside facts to "
                    "the answer, and do NOT invent a name for a codename absent from the key. "
                    "Report the ambiguity unresolved only if the passages truly do not "
                    "distinguish the candidates; do NOT answer 'insufficient' merely because the "
                    "key lists more than one name. (6) VERDICT: for a yes/no question use yes/no/"
                    "partly. For a 'who/which/identify' question use \"yes\" when the passages or "
                    "the key let you name/answer it, \"partly\" if only partially; reserve "
                    '"insufficient" for when neither passages nor key support ANY answer. '
                    "Do not invent. Return JSON "
                    '{"verdict":"yes|no|partly|insufficient","answer":"1-3 sentences","findings":'
                    '[{"text":"one fact","chunk_ids":[N]}]}.')},
                {"role": "user", "content":
                    (codekey + "\n\n" if codekey else "") +
                    f"Question: {question}\n\nPassages:\n" + "\n\n".join(passages)},
            ],
        )
        data = json.loads(r.choices[0].message.content or "{}")
    except Exception as e:
        if verbose:
            print(f"  [V14] grounded_finalize failed: {e}", file=sys.stderr)
        return False

    verdict = (data.get("verdict") or "").strip().lower()
    answer = (data.get("answer") or "").strip()
    valid_ids = {c.chunk_id for c in chunks}
    claims = []
    for f in (data.get("findings") or [])[:12]:
        txt = (f.get("text") or "").strip()
        if not txt:
            continue
        cids = [n for n in (f.get("chunk_ids") or []) if isinstance(n, int) and n in valid_ids][:3]
        claims.append(GroundedClaim(
            claim=V9Claim(text=txt[:220], confidence="medium", requires_citation=True),
            status="grounded" if cids else "weak", citation_chunk_ids=cids,
        ))
    # Replace policy: an AFFIRMATIVE verdict must be backed by a cited finding (trustworthy
    # correction); a NEGATIVE verdict replaces with a clean "No" and its findings carry no
    # citations (a citation must support an affirmative statement, not a non-mention).
    if verdict == "insufficient":
        if verbose:
            print("  [V14] grounded_finalize: insufficient — kept main answer", file=sys.stderr)
        return False
    if verdict in ("yes", "partly") and not any(c.citation_chunk_ids for c in claims):
        if verbose:
            print("  [V14] grounded_finalize: affirmative but uncited — kept main answer", file=sys.stderr)
        return False
    for c in claims:
        ctext = getattr(c.claim, "text", "") or ""
        if _NEGATION_RE.search(ctext) or ctext.lower().lstrip().startswith(("no ", "there is no", "there are no")):
            c.citation_chunk_ids = []
            c.status = "weak"

    lead = ""
    if is_yesno and verdict in ("yes", "no", "partly"):
        lead = {"yes": "Yes. ", "no": "No. ", "partly": "Partly. "}[verdict]
    new_narr = (lead + answer).strip()
    result.claims = claims
    result.narrative = new_narr
    try:
        result._authoritative_narrative = True  # authoritative answer, not a draft synthesis
    except Exception:
        pass
    # Drop the stale synthesis artifact + roster so format_answer can't render a leftover that
    # contradicts the corrected answer (e.g. an old "[HUAC p294]: No direct mention..." evidence
    # line). The verdict + grounded findings are the complete, faithful answer.
    result.grounded_roster = []
    if getattr(result, "synthesis", None):
        result.synthesis.narrative = new_narr
        for _attr in ("artifact",):
            if hasattr(result.synthesis, _attr):
                try:
                    setattr(result.synthesis, _attr, {})
                except Exception:
                    pass
    if verbose:
        print(f"  [V14] grounded_finalize: verdict={verdict}, {len(claims)} findings -> replaced answer", file=sys.stderr)
    return True


def run_v13_query(conn, question: str, **kwargs):
    """Convenience wrapper: V11 loop with the V13 profile enabled."""
    from retrieval.agent.v11_runner import run_v11_query
    kwargs["engine_profile"] = "v13"
    return run_v11_query(conn, question, **kwargs)


def run_v14_query(conn, question: str, **kwargs):
    """Convenience wrapper: V11 loop with the V14 profile (v13 + coverage) enabled."""
    from retrieval.agent.v11_runner import run_v11_query
    kwargs["engine_profile"] = "v14"
    return run_v11_query(conn, question, **kwargs)
