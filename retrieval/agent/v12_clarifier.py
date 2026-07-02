"""
V12 clarification layer — optional follow-up questions before investigation.

At initial query time the agent may ask 0–3 follow-up questions to pin down
intent, then incorporate the answers. Two question kinds:
  * single_choice / multi_choice — e.g. an ambiguous cover name ("Jurist" was
    used for BOTH Harry Dexter White and Alger Hiss) -> let the user pick.
  * free_text — open intent ("what aspect / time period / output do you want?").

Questions come from two sources, merged and capped at 3 (often fewer):
  1. Deterministic, data-grounded: surfaces in the query that resolve to >1
     distinct real person/entity in the concordance (codename ambiguity).
  2. LLM judgment: asks ONLY when a clarification would materially change the
     investigation (scope, time, sub-focus, output shape). Returns [] for clear
     queries — it is not forced to fill slots.

Answers are incorporated two ways (see incorporate_answers):
  * structured: chosen entities become *accepted* EntityCandidates that seed the
    workspace (the agent treats identity as resolved, no re-ambiguation), plus
    scope/date directives.
  * natural language: woven into the (augmented) query + clarification notes that
    the context pack surfaces to the LLM.
"""
from __future__ import annotations

import os
import re
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from retrieval.agent.v9_types import EntityCandidate
from retrieval.agent.v10_normalize import normalize_surface_for_lookup
from retrieval.agent.v10_spans import resolve_surface_to_entity_ids

logger = logging.getLogger(__name__)

MAX_QUESTIONS = 3
CLARIFIER_MODEL = os.getenv("V12_CLARIFIER_MODEL", os.getenv("V9_MODEL", "gpt-4.1-mini-2025-04-14"))

# Stopwords so we don't try to resolve "Who"/"What" as entities.
_STOP = {"who", "what", "when", "where", "why", "how", "is", "was", "the", "a", "an",
         "did", "do", "does", "of", "in", "to", "and", "or", "tell", "me", "about"}

# A name immediately followed by one of these is a proceeding/collection/group
# reference ("rosenberg trial", "silvermaster group"), not a who-is-this query.
_COLLECTIVE = r"(?:trial|case|hearing|hearings|group|network|ring|affair|apparatus|" \
              r"file|files|investigation|committee|notebook|notebooks|cables?)"

# Tokens that never appear in a real person's name — mark a "person" row as a descriptive
# alias/placeholder ("Man Behind", "The Unknown", "Cover Source", "Unidentified Man") to be
# dropped from disambiguation options.
_NON_NAME_TOKENS = {
    "man", "behind", "the", "unknown", "unidentified", "unsub", "cover", "name", "agent",
    "source", "group", "person", "subject", "someone", "unnamed", "anonymous", "individual",
    "contact", "informant", "operative", "asset", "official", "another",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class ClarificationOption:
    id: str                      # stable id, e.g. "grp:white" / "both" / "other"
    label: str                   # shown to user
    value: str = ""              # canonical value used when incorporating
    entity_ids: List[int] = field(default_factory=list)
    hint: Optional[str] = None

    def to_dict(self):
        return {"id": self.id, "label": self.label, "value": self.value,
                "entity_ids": self.entity_ids, "hint": self.hint}

    @classmethod
    def from_dict(cls, d):
        return cls(id=d["id"], label=d.get("label", ""), value=d.get("value", ""),
                   entity_ids=d.get("entity_ids", []) or [], hint=d.get("hint"))


@dataclass
class ClarificationQuestion:
    id: str
    question: str
    kind: str                    # "single_choice" | "multi_choice" | "free_text"
    category: str = "intent"     # "codename" | "scope" | "time" | "intent" | "output"
    options: List[ClarificationOption] = field(default_factory=list)
    allow_free_text: bool = True # choice questions still allow "something else"
    why: str = ""                # rationale (telemetry / UI tooltip)
    surface: Optional[str] = None  # for codename questions: the ambiguous term

    def to_dict(self):
        return {"id": self.id, "question": self.question, "kind": self.kind,
                "category": self.category, "options": [o.to_dict() for o in self.options],
                "allow_free_text": self.allow_free_text, "why": self.why, "surface": self.surface}

    @classmethod
    def from_dict(cls, d):
        return cls(id=d["id"], question=d.get("question", ""), kind=d.get("kind", "free_text"),
                   category=d.get("category", "intent"),
                   options=[ClarificationOption.from_dict(o) for o in d.get("options", [])],
                   allow_free_text=d.get("allow_free_text", True), why=d.get("why", ""),
                   surface=d.get("surface"))


@dataclass
class ClarificationPlan:
    questions: List[ClarificationQuestion] = field(default_factory=list)
    rationale: str = ""

    @property
    def needed(self) -> bool:
        return len(self.questions) > 0

    def to_dict(self):
        return {"questions": [q.to_dict() for q in self.questions], "rationale": self.rationale}

    @classmethod
    def from_dict(cls, d):
        return cls(questions=[ClarificationQuestion.from_dict(q) for q in (d or {}).get("questions", [])],
                   rationale=(d or {}).get("rationale", ""))


@dataclass
class ClarificationAnswer:
    question_id: str
    option_ids: List[str] = field(default_factory=list)
    free_text: Optional[str] = None

    @classmethod
    def from_dict(cls, d):
        return cls(question_id=d["question_id"], option_ids=d.get("option_ids", []) or [],
                   free_text=d.get("free_text"))


@dataclass
class ClarificationOutcome:
    augmented_question: str
    seed_entities: List[EntityCandidate] = field(default_factory=list)
    clarification_notes: List[str] = field(default_factory=list)
    scope_directives: Dict[str, Any] = field(default_factory=dict)
    transcript: List[Tuple[str, str]] = field(default_factory=list)  # (question, answer)


# ---------------------------------------------------------------------------
# Deterministic codename / entity-ambiguity detection
# ---------------------------------------------------------------------------
def _candidate_surfaces(question: str) -> List[Tuple[str, frozenset, int]]:
    """Candidate surfaces with their constituent words, longest-first.

    Case-insensitive (users type "who is jurist?"): we take quoted spans,
    capitalized phrases, every non-stopword word, and adjacent bigrams. Each
    carries the set of lowercased words it spans so the caller can prefer the
    longest unambiguous match and suppress its sub-words.
    """
    def words_of(s: str) -> frozenset:
        return frozenset(w.lower() for w in re.findall(r"[A-Za-z'\-]+", s) if w.lower() not in _STOP)

    cands: List[Tuple[str, frozenset, int]] = []
    for s in re.findall(r'"([^"]{2,40})"', question):
        cands.append((s, words_of(s), len(words_of(s))))
    for s in re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b', question):
        cands.append((s, words_of(s), len(words_of(s))))
    content = [re.sub(r"'s?$", "", w) for w in re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", question)
               if w.lower() not in _STOP]
    for a, b, c in zip(content, content[1:], content[2:]):
        cands.append((f"{a} {b} {c}", frozenset({a.lower(), b.lower(), c.lower()}), 3))
    for a, b in zip(content, content[1:]):
        cands.append((f"{a} {b}", frozenset({a.lower(), b.lower()}), 2))
    for w in content:
        cands.append((w, frozenset({w.lower()}), 1))

    out, seen = [], set()
    for surface, ws, n in cands:
        surface = surface.strip()
        if not surface or not ws:
            continue
        k = surface.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append((surface, ws, n))
    # longest (most words, then chars) first so full names win over their parts
    out.sort(key=lambda x: (-x[2], -len(x[0])))
    return out[:16]


def _strip_date_prefix(name: str) -> Tuple[str, Optional[str]]:
    """'1941-August 1944 Harry Dexter White' -> ('Harry Dexter White', '1941-August 1944')."""
    m = re.match(r'^\s*((?:\d{4}(?:[-–][A-Za-z]+\s+\d{4})?)|\d{3,})\s+(.*)$', name)
    if m:
        return m.group(2).strip(), m.group(1).strip()
    return name.strip(), None


def _group_person_entities(conn, entity_ids: List[int], surface: str) -> List[Dict[str, Any]]:
    """Group resolved entity_ids into distinct real people, dropping self-referential
    junk (entity whose name == the cover surface) and non-person noise."""
    if not entity_ids:
        return []
    with conn.cursor() as cur:
        cur.execute("SELECT id, canonical_name, entity_type FROM entities WHERE id = ANY(%s)", (entity_ids,))
        rows = cur.fetchall()
    surf_norm = re.sub(r"[^a-z]", "", surface.lower())
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for eid, name, etype in rows:
        if not name:
            continue
        clean, datequal = _strip_date_prefix(name)
        cn = re.sub(r"[^a-z]", "", clean.lower())
        # drop obvious citation/reference noise
        if "referencesto" in cn or "reference" in clean.lower():
            continue
        toks = [t for t in re.split(r"\s+", clean) if t]
        # require >= 2 name-like tokens (capitalized, alphabetic). This already
        # removes single-token cover-name-as-entity junk (e.g. "Jurist"), so we do
        # NOT drop name==surface: a full name like "Harry Gold" legitimately equals
        # the surface "harry gold" and must be kept.
        name_toks = [t for t in toks if re.match(r"^[A-Z][A-Za-z.'\-]+$", t)]
        if len(name_toks) < 2:
            continue
        # Drop descriptive non-person aliases ("Man Behind", "The Unknown", "Cover Source"):
        # a real person name has no function/placeholder word as a token.
        if any(t.lower() in _NON_NAME_TOKENS for t in name_toks):
            continue
        key = (name_toks[-1].lower().strip(".,"), name_toks[0][0].lower())  # (surname, first-initial)
        g = groups.setdefault(key, {"display": clean, "entity_ids": [], "dates": set()})
        g["entity_ids"].append(eid)
        # prefer the longest clean display name (e.g. "Harry Dexter White" over "Harry D. White")
        if len(clean) > len(g["display"]):
            g["display"] = clean
        if datequal:
            g["dates"].add(datequal)
    return [
        {"display": g["display"], "entity_ids": g["entity_ids"], "dates": sorted(g["dates"])}
        for g in groups.values()
    ]


def detect_codename_questions(conn, question: str, max_q: int = MAX_QUESTIONS) -> List[ClarificationQuestion]:
    """A single_choice question per query surface that maps to >1 distinct person.

    Greedy longest-first: a full name that resolves unambiguously ("harry gold")
    claims its words so its ambiguous parts ("harry", "gold") are not also asked.
    Only a surface not subsumed by a longer unambiguous match can trigger.
    """
    qs: List[ClarificationQuestion] = []
    claimed: set = set()  # lowercased words already accounted for by a longer match
    for surface, ws, _n in _candidate_surfaces(question):
        if ws & claimed:
            continue
        # "<name> trial/group/case/..." -> a proceeding/collection, not a person query
        if re.search(rf"\b{re.escape(surface)}\s+{_COLLECTIVE}\b", question, re.IGNORECASE):
            claimed |= ws
            continue
        try:
            nk = normalize_surface_for_lookup(surface)
            ids = resolve_surface_to_entity_ids(conn, nk, max_entities=12)
        except Exception:
            continue
        groups = _group_person_entities(conn, ids, surface)
        if len(groups) == 1:
            claimed |= ws   # unambiguous full match -> suppress its sub-words
            continue
        if len(groups) < 2:
            continue
        claimed |= ws       # ambiguous -> ask, and suppress sub-words too
        opts: List[ClarificationOption] = []
        for g in groups:
            dates = f" ({'; '.join(g['dates'])})" if g["dates"] else ""
            opts.append(ClarificationOption(
                id=f"grp:{re.sub(r'[^a-z]', '', g['display'].lower())}",
                label=f"{g['display']}{dates}",
                value=g["display"], entity_ids=g["entity_ids"],
            ))
        all_label = "Both / compare them" if len(groups) == 2 else "All of them / compare"
        opts.append(ClarificationOption(id="both", label=all_label, value="__ALL__",
                                        entity_ids=[e for g in groups for e in g["entity_ids"]]))
        names = " and ".join(g["display"] for g in groups)
        qs.append(ClarificationQuestion(
            id=f"codename:{nk}",
            question=f"“{surface}” was used as a cover name for more than one person "
                     f"({names}). Which do you mean?",
            kind="single_choice", category="codename", options=opts,
            allow_free_text=True, surface=surface,
            why=f"'{surface}' resolves to {len(groups)} distinct people in the concordance.",
        ))
        if len(qs) >= max_q:
            break
    return qs


# ---------------------------------------------------------------------------
# LLM intent questions (judgment; may return [])
# ---------------------------------------------------------------------------
_LLM_SCHEMA_HINT = """Return JSON: {"questions": [{"question": str, "kind": "single_choice|multi_choice|free_text",
"category": "scope|time|intent|output", "options": [str,...], "why": str}]}.
Ask a question ONLY if the answer would materially change the investigation (scope, time window,
which sub-entity/relationship, or output shape). Prefer 0. Never exceed the budget. Do not ask
about identity disambiguation (handled separately). Options only for choice kinds."""


def llm_intent_questions(question: str, budget: int, already: List[ClarificationQuestion]) -> List[ClarificationQuestion]:
    if budget <= 0 or os.getenv("V12_DISABLE_LLM", "0") in ("1", "true", "yes"):
        return []
    try:
        from openai import OpenAI
        client = OpenAI(timeout=8, max_retries=1)  # bound added latency; fail open
        sys_prompt = (
            "You triage a query against a Cold War / Soviet-espionage document archive "
            "(Venona & Vassiliev decrypts, FBI files, grand-jury and trial transcripts) and decide whether "
            f"to ask up to {budget} clarifying follow-up question(s) BEFORE searching. " + _LLM_SCHEMA_HINT
        )
        ctx = ""
        if already:
            ctx = "\nAlready asking (do not duplicate): " + "; ".join(q.question for q in already)
        resp = client.chat.completions.create(
            model=CLARIFIER_MODEL, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": f"Query: {question}{ctx}"}],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        logger.debug("v12 llm_intent_questions skipped: %s", e)
        return []
    out: List[ClarificationQuestion] = []
    for i, q in enumerate(data.get("questions", [])[:budget]):
        kind = q.get("kind", "free_text")
        opts = [ClarificationOption(id=f"o{j}", label=str(o), value=str(o))
                for j, o in enumerate(q.get("options", []) or [])] if kind != "free_text" else []
        out.append(ClarificationQuestion(
            id=f"intent:{i}", question=str(q.get("question", "")).strip(),
            kind=kind, category=q.get("category", "intent"), options=opts,
            allow_free_text=True, why=str(q.get("why", "")),
        ))
    return [q for q in out if q.question]


# ---------------------------------------------------------------------------
# Top-level: generate + incorporate
# ---------------------------------------------------------------------------
def generate_clarifications(conn, question: str, *, max_questions: int = MAX_QUESTIONS,
                            use_llm: bool = True) -> ClarificationPlan:
    codename_qs = detect_codename_questions(conn, question, max_q=max_questions)
    budget = max_questions - len(codename_qs)
    intent_qs = llm_intent_questions(question, budget, codename_qs) if use_llm else []
    # Guard: the LLM sometimes re-asks an identity question the codename layer
    # already covers — drop any intent question that mentions a claimed surface.
    if codename_qs and intent_qs:
        surfaces = {(q.surface or "").lower() for q in codename_qs if q.surface}
        intent_qs = [q for q in intent_qs
                     if not any(s and s in q.question.lower() for s in surfaces)]
    questions = (codename_qs + intent_qs)[:max_questions]
    rationale = (f"{len(codename_qs)} disambiguation + {len(intent_qs)} intent question(s)"
                 if questions else "Query is clear; no clarification needed.")
    return ClarificationPlan(questions=questions, rationale=rationale)


def incorporate_answers(conn, question: str, plan: ClarificationPlan,
                        answers: List[ClarificationAnswer]) -> ClarificationOutcome:
    by_id = {q.id: q for q in plan.questions}
    ans_by_id = {a.question_id: a for a in answers}
    seed: List[EntityCandidate] = []
    notes: List[str] = []
    transcript: List[Tuple[str, str]] = []
    scope: Dict[str, Any] = {}
    rewrites: List[Tuple[str, str]] = []   # (surface -> resolved name) to apply to the query
    subject_prefixes: List[str] = []       # foregrounded subjects when no clean substitution

    for q in plan.questions:
        a = ans_by_id.get(q.id)
        if not a:
            continue
        if q.category == "codename":
            picked = [o for o in q.options if o.id in a.option_ids]
            chose_both = any(o.id == "both" for o in picked)
            # the concrete person options the user is asking about
            person_opts = ([o for o in q.options if o.id != "both"] if chose_both
                           else [o for o in picked if o.id != "both"])
            person_names = [o.value for o in person_opts if o.value and o.value != "__ALL__"]
            chosen_eids: List[int] = [e for o in person_opts for e in o.entity_ids]
            if chosen_eids:
                # accepted identity -> seed workspace (resolve clean names)
                with conn.cursor() as cur:
                    cur.execute("SELECT id, canonical_name, entity_type FROM entities WHERE id = ANY(%s)",
                                (chosen_eids,))
                    for eid, name, etype in cur.fetchall():
                        clean, _ = _strip_date_prefix(name or "")
                        seed.append(EntityCandidate(
                            query_term=q.surface or "", entity_id=eid, canonical_name=clean,
                            entity_type=etype, matched_via="user_clarification",
                            accepted=True, confidence="user", ambiguous=False))
                subject = " and ".join(person_names) if person_names else "the person"
                # Foreground the REAL NAME in the query so retrieval searches the
                # person across ALL collections — not the cover name, which only
                # appears in Venona/Vassiliev.
                if q.surface and len(person_names) == 1:
                    rewrites.append((q.surface, person_names[0]))
                elif person_names:
                    subject_prefixes.append(subject)
                notes.append(
                    f"'{q.surface}' is the cover name/alias for {subject} (user-confirmed). "
                    f"Investigate {subject} BY NAME across ALL collections (FBI files, grand-jury "
                    f"and hearing transcripts, reports), not only Venona/Vassiliev where the cover "
                    f"name '{q.surface}' appears.")
                transcript.append((q.question, ", ".join(o.label for o in picked)))
            if a.free_text:
                notes.append(f"On '{q.surface}': {a.free_text}")
                transcript.append((q.question, a.free_text))
        else:
            picked_labels = [o.label for o in q.options if o.id in a.option_ids]
            parts = picked_labels + ([a.free_text] if a.free_text else [])
            if parts:
                note = f"{q.question} -> {'; '.join(parts)}"
                notes.append(note)
                transcript.append((q.question, "; ".join(parts)))
                if q.category in ("scope", "time"):
                    scope.setdefault(q.category, []).extend(parts)

    aug = question
    for surf, name in rewrites:  # "who was jurist?" -> "who was Harry Dexter White?"
        aug = re.sub(rf"\b{re.escape(surf)}('s)?\b", name, aug, flags=re.IGNORECASE)
    if subject_prefixes:
        aug = f"(Regarding {', '.join(subject_prefixes)}) " + aug
    if notes:
        aug = aug.rstrip() + "\n\n[User clarifications: " + " ".join(notes) + "]"
    return ClarificationOutcome(augmented_question=aug, seed_entities=seed,
                                clarification_notes=notes, scope_directives=scope,
                                transcript=transcript)
