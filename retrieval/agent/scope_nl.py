"""
Natural-language scope detection for chat queries.

Detects when a user names a collection in plain language ("in the Silvermaster files",
"only the Vassiliev notebooks", "search Venona") so the query can be scoped to it — and,
because NL scope is easy to get wrong, dispatch confirms the detected scope with the user
before running (see v9_dispatch scope-confirmation flow).

Design goals:
- Cover ALL collections, not a hardcoded 3.
- Avoid false positives from person names that are also collection names (Golos, Rosenberg,
  Greenglass, Coplon, Hiss...): those require a "source noun" (files/collection/testimony)
  right after the name, whereas unambiguous archive names (Venona, Vassiliev, Silvermaster,
  SOLO...) may be triggered by a scope preposition alone.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Scope-intent prepositions/verbs that precede a collection mention.
_PREP = r"(?:in|within|from|only(?:\s+in|\s+from)?|across|throughout|inside|under|search(?:ing)?|restrict(?:ed)?\s+to|limit(?:ed)?\s+to|scoped?\s+to|confined?\s+to|just)"
# Source nouns that follow a collection mention ("... files", "... notebooks").
_SOURCE_NOUN = r"(?:files?|collections?|notebooks?|decrypts?|cables?|records?|papers?|hearings?|transcripts?|testimony|grand\s+jury|archive|documents?|dossiers?)"

# Curated NL phrases per collection slug. Value = (list of phrases, needs_source_noun).
# needs_source_noun=True for slugs that are also common person/topic names in queries.
_CURATED: Dict[str, Tuple[List[str], bool]] = {
    "venona": (["venona"], False),
    "vassiliev": (["vassiliev"], False),
    "silvermaster": (["silvermaster"], False),
    "solo": (["operation solo", "solo operation", "solo"], False),
    "mccarthy": (["mccarthy"], False),
    "huac_hearings": (["huac hearings", "huac"], False),
    "huac_reports": (["huac reports"], False),
    "fbicomrap": (["comrap", "comintern apparatus"], False),
    "fbi_cinrad": (["cinrad", "radiation laboratory"], False),
    "siss_scope_soviet": (["siss", "scope of soviet activity"], False),
    "soviet_atomic_espionage_1951": (["soviet atomic espionage"], False),
    "hiss_chambers": (["hiss-chambers", "hiss and chambers", "alger hiss", "whittaker chambers", "hiss", "chambers"], True),
    "rosenberg": (["rosenberg case", "julius rosenberg", "rosenberg"], True),
    "rosenberg_grand_jury": (["rosenberg grand jury"], False),
    "rosenberg_trial_transcripts": (["rosenberg trial"], False),
    "judith_coplon": (["judith coplon", "coplon"], True),
    "david_greenglass": (["david greenglass", "greenglass"], True),
    "david_ruth_greenglass": (["david and ruth greenglass"], False),
    "golos": (["golos"], True),
    "pravdin": (["pravdin"], True),
    "mink": (["george mink", "mink"], True),
    "albertson": (["albertson"], True),
    "winton_burdett": (["winton burdett", "burdett"], True),
    "oscar_seborer": (["oscar seborer", "seborer"], True),
    "fbi_hiskey": (["hiskey"], True),
    "eva_childs": (["eva childs"], True),
    "jack_childs": (["jack childs"], True),
    "morris_childs": (["morris childs"], True),
    "brothman_moskowitz_grand_jury": (["brothman", "moskowitz"], True),
    "soviet_intel_travel_techniques": (["travel techniques", "intelligence techniques"], False),
    "volodarsky": (["volodarsky", "feldman"], True),
}

# "full archive" reset phrases.
_FULL_ARCHIVE_RE = re.compile(
    r"\b(?:full\s+archive|all\s+collections|entire\s+archive|whole\s+archive|every\s+collection|everything|all\s+sources|all\s+files)\b",
    re.IGNORECASE,
)


@dataclass
class NLScopeResult:
    collections: List[str] = field(default_factory=list)   # detected collection slugs
    matched_phrases: List[str] = field(default_factory=list)
    full_archive: bool = False                              # explicit "full archive" reset
    confidence: float = 0.0

    @property
    def detected(self) -> bool:
        return bool(self.collections) or self.full_archive


def _collection_titles(conn) -> Dict[str, str]:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT slug, title FROM collections")
            return {r[0]: (r[1] or "") for r in cur.fetchall()}
    except Exception:
        try: conn.rollback()
        except Exception: pass
        return {}


def detect_nl_scope(conn, question: str, *, verbose: bool = False) -> NLScopeResult:
    """Detect a collection scope stated in natural language.

    Returns collections (slugs) the user appears to be restricting to. Requires a scope
    cue (preposition before, or source noun after) so ordinary mentions of a person who
    shares a collection name (e.g. "Golos") don't scope the query.
    """
    q = " " + (question or "").lower().strip() + " "
    result = NLScopeResult()

    if _FULL_ARCHIVE_RE.search(question or ""):
        result.full_archive = True
        result.confidence = 0.9
        return result

    # Only consider collections that actually exist in this DB.
    valid_slugs = set(_collection_titles(conn).keys()) or set(_CURATED.keys())

    prep = _PREP
    noun = _SOURCE_NOUN
    found: Dict[str, str] = {}
    for slug, (phrases, needs_noun) in _CURATED.items():
        if slug not in valid_slugs:
            continue
        for phrase in phrases:
            p = re.escape(phrase)
            # preposition (optionally + "the") right before the phrase
            prep_before = re.search(rf"\b{prep}\s+(?:the\s+)?{p}\b", q)
            # a source noun right after the phrase (within 2 words)
            noun_after = re.search(rf"\b{p}(?:'s)?\s+(?:\w+\s+)?{noun}\b", q)
            if needs_noun:
                hit = noun_after  # ambiguous names require the source noun
            else:
                hit = prep_before or noun_after or re.search(rf"\b{p}\s+{noun}\b", q)
            if hit:
                found[slug] = phrase
                break

    if found:
        result.collections = list(found.keys())
        result.matched_phrases = list(found.values())
        # Higher confidence when a source noun was involved / multiple matches.
        result.confidence = 0.85 if len(found) == 1 else 0.7
        if verbose:
            print(f"  [NL scope] detected {result.collections} via {result.matched_phrases}", file=sys.stderr)
    return result


def strip_nl_scope(question: str, matched_phrases: List[str]) -> str:
    """Remove the matched scope phrases (and their cue words) from the query text so the
    retrieval query isn't polluted by 'in the vassiliev notebooks'."""
    q = question or ""
    for phrase in matched_phrases:
        q = re.sub(
            rf"\b{_PREP}\s+(?:the\s+)?{re.escape(phrase)}(?:'s)?(?:\s+{_SOURCE_NOUN})?\b",
            " ", q, flags=re.IGNORECASE,
        )
        q = re.sub(rf"\b{re.escape(phrase)}(?:'s)?\s+{_SOURCE_NOUN}\b", " ", q, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", q).strip(" ,.")
