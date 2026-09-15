"""
OCR-variant channel (Phase 2): expand query tokens to corpus lexemes that are
plausible OCR corruptions of the token (e.g. typed FUHR read as FUER).

How it works:
  - Every corpus lexeme gets a "skeleton": lowercase, non-alphanumerics
    stripped, each character collapsed to the representative of its
    OCR-confusion class (e.g. e/h/o -> "e"). Skeletons are precomputed and
    stored in corpus_dictionary_lexemes.skeleton (migration 0075, populated by
    scripts/build_corpus_dictionary.py), so candidate corruptions of a query
    token are found with an exact (build_id, skeleton) index lookup.
  - Candidates are then priced with a confusion-weighted edit distance
    (single-char substitution costs plus 2ch<->1ch pair transitions like
    rn<->m) and filtered/ranked by (cost ASC, chunk_freq ASC). Rarity is the
    point: garbled forms are rare, so low chunk_freq wins.

Costs come from config/ocr_confusion.json (built by
scripts/build_confusion_weights.py). If the artifact is absent or unreadable
we fall back to BUILTIN_PRIORS — retrieval must never crash because of this
module. No file is read at import time; the artifact loads lazily on first
load_confusion() call and is cached module-level.

Consumed by retrieval/fuzzy_lex.py (lazy import) and
scripts/build_corpus_dictionary.py (skeleton population).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFUSION_PATH = _REPO_ROOT / "config" / "ocr_confusion.json"

INF = float("inf")

# Merges between confusion-class members are only made for cheap confusions;
# classes are kept small so skeletons stay discriminative.
CLASS_COST_THRESHOLD = 0.35
CLASS_MAX_MEMBERS = 4


def derive_confusion_classes(
    sub_costs: Dict[str, Any],
    *,
    cost_threshold: float = CLASS_COST_THRESHOLD,
    max_members: int = CLASS_MAX_MEMBERS,
) -> List[List[str]]:
    """
    Union-find over single-char substitution pairs with cost <= cost_threshold,
    cheapest merges first, classes capped at max_members members. Returns
    sorted classes of sorted members (each class has >= 2 members).
    """
    edges = []
    for key, cost in (sub_costs or {}).items():
        parts = str(key).split("|")
        if len(parts) != 2 or len(parts[0]) != 1 or len(parts[1]) != 1:
            continue
        try:
            cost_f = float(cost)
        except (TypeError, ValueError):
            continue
        if cost_f > cost_threshold:
            continue
        a, b = sorted(parts)
        edges.append((cost_f, a, b))
    edges.sort()

    parent: Dict[str, str] = {}
    size: Dict[str, int] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for cost_f, a, b in edges:
        for ch in (a, b):
            if ch not in parent:
                parent[ch] = ch
                size[ch] = 1
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if size[ra] + size[rb] > max_members:
            continue
        # Merge into the alphabetically-smaller root for determinism.
        if rb < ra:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    groups: Dict[str, List[str]] = {}
    for ch in parent:
        groups.setdefault(find(ch), []).append(ch)
    classes = [sorted(members) for members in groups.values() if len(members) >= 2]
    classes.sort(key=lambda m: m[0])
    return classes


# Hand-tuned priors used when config/ocr_confusion.json is absent. Mirrors the
# artifact schema (ocr_confusion_v1). Keys of sub_costs are the two chars
# sorted; pair_costs keys are "<2ch>|<1ch>". Note e|h at 0.35 is deliberately
# within CLASS_COST_THRESHOLD so skeleton("fuhr") == skeleton("fuer") holds
# even under builtin priors (the FUHR->FUER motivating case).
BUILTIN_PRIORS: Dict[str, Any] = {
    "version": "ocr_confusion_v1",
    "generated": "builtin",
    "sources": ["builtin-priors"],
    "classes": [],  # derived below
    "sub_costs": {
        "e|o": 0.3,
        "n|u": 0.35,
        "i|l": 0.3,
        "1|i": 0.3,
        "1|l": 0.3,
        "b|h": 0.4,
        "5|s": 0.4,
        "c|e": 0.45,
        "c|o": 0.45,
        "u|v": 0.45,
        "f|t": 0.45,
        "g|q": 0.45,
        "h|n": 0.45,
        "e|h": 0.35,
        "a|s": 0.5,
        "m|w": 0.5,
        "n|r": 0.5,
    },
    "pair_costs": {
        "rn|m": 0.35,
        "vv|w": 0.4,
        "cl|d": 0.5,
        "ni|m": 0.5,
        "li|h": 0.55,
    },
    "default_sub_cost": 1.0,
    "indel_cost": 0.8,
}
BUILTIN_PRIORS["classes"] = derive_confusion_classes(BUILTIN_PRIORS["sub_costs"])


class Confusion:
    """
    Parsed confusion-cost artifact (ocr_confusion_v1). Normalizes key
    orientation: sub_costs keys become "<a>|<b>" with a < b; pair_costs keys
    become "<2ch>|<1ch>" regardless of the orientation in the source dict.
    """

    def __init__(self, data: Dict[str, Any]):
        data = data or {}
        self.version: str = str(data.get("version") or "ocr_confusion_v1")
        self.generated: Optional[str] = data.get("generated")
        self.sources: List[Any] = list(data.get("sources") or [])
        self.default_sub_cost: float = float(data.get("default_sub_cost", 1.0))
        self.indel_cost: float = float(data.get("indel_cost", 0.8))

        self.sub_costs: Dict[str, float] = {}
        for key, cost in (data.get("sub_costs") or {}).items():
            parts = str(key).split("|")
            if len(parts) != 2 or len(parts[0]) != 1 or len(parts[1]) != 1:
                continue
            a, b = sorted(parts)
            self.sub_costs[f"{a}|{b}"] = float(cost)

        self.pair_costs: Dict[str, float] = {}
        for key, cost in (data.get("pair_costs") or {}).items():
            parts = str(key).split("|")
            if len(parts) != 2:
                continue
            left, right = parts
            if len(left) == 2 and len(right) == 1:
                self.pair_costs[f"{left}|{right}"] = float(cost)
            elif len(left) == 1 and len(right) == 2:
                self.pair_costs[f"{right}|{left}"] = float(cost)

        classes = data.get("classes") or derive_confusion_classes(self.sub_costs)
        self.classes: List[List[str]] = []
        for members in classes:
            singles = sorted(str(m) for m in members if len(str(m)) == 1)
            if len(singles) >= 2:
                self.classes.append(singles)
        self.classes.sort(key=lambda m: m[0])

        # char -> class representative (alphabetically first member)
        self._char_rep: Dict[str, str] = {}
        for members in self.classes:
            rep = members[0]
            for ch in members:
                self._char_rep[ch] = rep

    def rep(self, ch: str) -> str:
        return self._char_rep.get(ch, ch)

    def sub_cost(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        key = f"{a}|{b}" if a < b else f"{b}|{a}"
        return self.sub_costs.get(key, self.default_sub_cost)

    def pair_cost(self, two: str, one: str) -> Optional[float]:
        return self.pair_costs.get(f"{two}|{one}")


_confusion_cache: Dict[str, Confusion] = {}


def load_confusion(path: Optional[str] = None) -> Confusion:
    """
    Load the confusion-cost artifact (default config/ocr_confusion.json at the
    repo root), falling back to BUILTIN_PRIORS when the file is absent or
    unreadable. Cached module-level per path; never raises.
    """
    target = Path(path) if path is not None else DEFAULT_CONFUSION_PATH
    key = str(target)
    cached = _confusion_cache.get(key)
    if cached is not None:
        return cached
    try:
        with open(target, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        conf = Confusion(data)
    except Exception:
        conf = Confusion(BUILTIN_PRIORS)
    _confusion_cache[key] = conf
    return conf


def skeleton(token: str, conf: Optional[Confusion] = None) -> str:
    """
    Collapse a token to its OCR-confusion skeleton: lowercase, strip
    non-alphanumerics, map each char to its confusion-class representative.
    """
    if conf is None:
        conf = load_confusion()
    out = []
    for ch in (token or "").lower():
        if not ch.isalnum():
            continue
        out.append(conf.rep(ch))
    return "".join(out)


def weighted_distance(a: str, b: str, conf: Optional[Confusion] = None, cap: float = 2.0) -> float:
    """
    Confusion-weighted edit distance between two tokens. Substitutions use
    conf.sub_costs (default_sub_cost otherwise), insert/delete cost
    conf.indel_cost, and 2ch<->1ch pair transitions (e.g. rn<->m) use
    conf.pair_costs. Returns INF when the distance exceeds cap (with an
    early exit once no alignment can stay within cap).
    """
    if conf is None:
        conf = load_confusion()
    a = (a or "").lower()
    b = (b or "").lower()
    if a == b:
        return 0.0
    n, m = len(a), len(b)
    indel = conf.indel_cost

    prev: List[float] = [j * indel for j in range(m + 1)]
    prev_prev: Optional[List[float]] = None
    prev_min = 0.0
    for i in range(1, n + 1):
        ca = a[i - 1]
        cur: List[float] = [i * indel] + [INF] * m
        for j in range(1, m + 1):
            cb = b[j - 1]
            best = prev[j - 1] + (0.0 if ca == cb else conf.sub_cost(ca, cb))
            d = prev[j] + indel
            if d < best:
                best = d
            d = cur[j - 1] + indel
            if d < best:
                best = d
            if i >= 2 and prev_prev is not None:
                # two chars of `a` read as one char of `b` (e.g. "rn" -> "m")
                pc = conf.pair_cost(a[i - 2:i], cb)
                if pc is not None:
                    d = prev_prev[j - 1] + pc
                    if d < best:
                        best = d
            if j >= 2:
                # one char of `a` read as two chars of `b` (e.g. "m" -> "rn")
                pc = conf.pair_cost(b[j - 2:j], ca)
                if pc is not None:
                    d = prev[j - 2] + pc
                    if d < best:
                        best = d
            cur[j] = best
        cur_min = min(cur)
        # Paths may skip a single row via pair transitions, so only exit when
        # the last TWO rows are entirely above cap (every path must touch one).
        if cur_min > cap and prev_min > cap:
            return INF
        prev_prev, prev, prev_min = prev, cur, cur_min

    dist = prev[m]
    return dist if dist <= cap else INF


_SKELETON_COLUMN_SQL = """
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'corpus_dictionary_lexemes' AND column_name = 'skeleton';
"""

_VARIANT_SQL = """
    SELECT lexeme, chunk_freq
    FROM corpus_dictionary_lexemes
    WHERE build_id = %s
      AND skeleton = %s
      AND lexeme <> %s
      AND chunk_freq <= %s;
"""


def _skeleton_column_exists(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute(_SKELETON_COLUMN_SQL)
        return cur.fetchone() is not None


def fetch_ocr_variants(
    conn,
    tokens: Iterable[str],
    build_id: Optional[int],
    conf: Optional[Confusion] = None,
    *,
    max_cost: float = 1.5,
    top_k: int = 4,
    freq_ceiling: int = 500,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    For each query token (len >= 4), look up corpus lexemes sharing its
    skeleton (rare ones only: chunk_freq <= freq_ceiling), keep those within
    max_cost by weighted_distance, and return the top_k per token ranked by
    (cost ASC, chunk_freq ASC) — garble is rare, so rarity wins ties.

    Returns {token: [{"lexeme", "cost", "chunk_freq", "source": "ocr"}, ...]}.
    Degrades gracefully to {} when build_id is None or the skeleton column
    (migration 0075) is missing.
    """
    tokens = [t for t in (tokens or []) if t]
    if build_id is None or not tokens:
        return {}
    try:
        if not _skeleton_column_exists(conn):
            return {}
    except Exception:
        return {}
    if conf is None:
        conf = load_confusion()

    max_cost_f = float(max_cost)
    out: Dict[str, List[Dict[str, Any]]] = {}
    seen = set()
    with conn.cursor() as cur:
        for tok in tokens:
            if tok in seen:
                continue
            seen.add(tok)
            if len(tok) < 4:
                continue
            sk = skeleton(tok, conf)
            if not sk:
                continue
            cur.execute(_VARIANT_SQL, (build_id, sk, tok, int(freq_ceiling)))
            rows = cur.fetchall()
            cands: List[Dict[str, Any]] = []
            for lexeme, chunk_freq in rows:
                lex = str(lexeme)
                if lex == tok:
                    continue
                cost = weighted_distance(tok, lex, conf, cap=max_cost_f)
                if cost > max_cost_f:
                    continue
                cands.append(
                    {
                        "lexeme": lex,
                        "cost": cost,
                        "chunk_freq": int(chunk_freq or 0),
                        "source": "ocr",
                    }
                )
            cands.sort(key=lambda e: (e["cost"], e["chunk_freq"], e["lexeme"]))
            if cands:
                out[tok] = cands[: int(top_k)]
    return out
