"""
V9 Think Deeper — Deterministic Safety Rails.

Embedding-based constraints that prevent degenerate behavior.
NOT the scoreboard.  The intelligence lives in the Judge.

Rails:
  1. Near-dup rejection (cosine sim to any selected chunk)
  2. Drift floor with must_include override + gap targeting
  3. Soft per-doc cap with Judge overflow
  4. Working set cap
  5. No-admissible-candidates detection
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from retrieval.agent.v9_deep_types import CandidateChunk

logger = logging.getLogger(__name__)


# ── Config ───────────────────────────────────────────────────────────────────

def _infer_roster_intent(directive_primary_question: str, user_directive: Optional[str] = None) -> bool:
    """Cheap heuristic: roster-style questions get relaxed doc cap."""
    text = (directive_primary_question or "") + " " + (user_directive or "")
    text_lower = text.lower()
    return any(
        phrase in text_lower
        for phrase in ("who was in", "members of", "network", "roster", "who were in")
    )


@dataclass
class RailsConfig:
    per_doc_soft_cap: int = 5
    per_doc_hard_cap: int = 12
    roster_intent: bool = False  # when True, per_doc_soft_cap lifted to 8
    dup_cosine_threshold: float = 0.92
    max_tool_calls: int = 10
    drift_sim_floor: float = 0.25
    max_total_selected: int = 120
    # Per-collection quota: no single collection can exceed this fraction
    # of the total working set.  0.0 = disabled (no quota enforcement).
    per_collection_max_pct: float = 0.70  # default 70%


# ── Report ───────────────────────────────────────────────────────────────────

@dataclass
class RailsReport:
    """Persisted per step for debugging and threshold tuning."""
    filtered_dup: List[int] = field(default_factory=list)
    filtered_drift: List[int] = field(default_factory=list)
    filtered_doc_cap: List[int] = field(default_factory=list)
    filtered_collection_quota: List[int] = field(default_factory=list)
    admitted_count: int = 0
    drift_sim_histogram: List[float] = field(default_factory=list)
    dup_sim_histogram: List[float] = field(default_factory=list)
    adaptive_drift_floor_used: float = 0.0
    adaptive_dup_threshold_used: float = 0.0
    doc_overflow_requested: bool = False
    doc_overflow_doc_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "filtered_dup": self.filtered_dup,
            "filtered_drift": self.filtered_drift,
            "filtered_doc_cap": self.filtered_doc_cap,
            "filtered_collection_quota": self.filtered_collection_quota,
            "admitted_count": self.admitted_count,
            "drift_sim_histogram": self.drift_sim_histogram[:20],  # cap for storage
            "dup_sim_histogram": self.dup_sim_histogram[:20],
            "adaptive_drift_floor_used": self.adaptive_drift_floor_used,
            "adaptive_dup_threshold_used": self.adaptive_dup_threshold_used,
            "doc_overflow_requested": self.doc_overflow_requested,
            "doc_overflow_doc_ids": self.doc_overflow_doc_ids,
        }


# ── Embedding math ───────────────────────────────────────────────────────────

def _cosine_sim(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors.  Returns 0.0 if either is empty."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def _percentile(values: List[float], pct: float) -> float:
    """Simple linear-interpolation percentile (pct in 0..100)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (pct / 100.0) * (len(s) - 1)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    d = k - f
    return s[f] + d * (s[c] - s[f])


# ── Adaptive thresholds ─────────────────────────────────────────────────────

def compute_adaptive_thresholds(
    sims_to_seed: List[float],
    max_sims_to_selected: List[float],
    config: RailsConfig,
) -> Tuple[float, float]:
    """Compute adaptive thresholds from TWO separate distributions.

    drift_floor  = max(config.drift_sim_floor, p10(sims_to_seed))
    dup_threshold = min(config.dup_cosine_threshold, p95(max_sims_to_selected))
    """
    if sims_to_seed:
        adaptive_drift = max(config.drift_sim_floor,
                             _percentile(sims_to_seed, 10))
    else:
        adaptive_drift = config.drift_sim_floor

    if max_sims_to_selected:
        adaptive_dup = min(config.dup_cosine_threshold,
                           _percentile(max_sims_to_selected, 95))
    else:
        adaptive_dup = config.dup_cosine_threshold

    return adaptive_drift, adaptive_dup


# ── Lazy embedding loading ───────────────────────────────────────────────────

def load_embeddings_for_chunks(
    conn,
    chunk_ids: List[int],
) -> Dict[int, List[float]]:
    """Batch-load embeddings from DB for the given chunk_ids.

    Returns {chunk_id: embedding_vector}.  Missing chunks are silently skipped.
    """
    if not chunk_ids:
        return {}
    result: Dict[int, List[float]] = {}
    try:
        with conn.cursor() as cur:
            # pgvector stores embeddings; cast to float array for Python
            cur.execute(
                "SELECT id, embedding::text FROM chunks WHERE id = ANY(%s) AND embedding IS NOT NULL",
                (list(chunk_ids),),
            )
            for row in cur.fetchall():
                cid = row[0]
                raw = row[1]
                # pgvector text format: "[0.1,0.2,...]"
                if raw and isinstance(raw, str):
                    raw = raw.strip("[]")
                    try:
                        vec = [float(x) for x in raw.split(",")]
                        result[cid] = vec
                    except ValueError:
                        pass
    except Exception as e:
        logger.warning("Failed to load embeddings: %s", e)
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
    return result


# ── Main rails function ─────────────────────────────────────────────────────

def apply_rails(
    new_candidates: List[CandidateChunk],
    current_selected: List[CandidateChunk],
    seed_embedding: List[float],
    config: RailsConfig,
    *,
    conn=None,
    doc_overflow_ids: Optional[Set[int]] = None,
    must_include_entity_ids: Optional[Set[int]] = None,
    gap_embedding: Optional[List[float]] = None,
) -> Tuple[List[CandidateChunk], RailsReport]:
    """Apply deterministic safety rails to new candidate chunks.

    Two-pass for performance:
      1. Score-based pre-filter (top N by retrieval score).
      2. Embedding-based rails for survivors.
    """
    report = RailsReport()
    if doc_overflow_ids:
        report.doc_overflow_requested = True
        report.doc_overflow_doc_ids = list(doc_overflow_ids)

    if not new_candidates:
        return [], report

    mi_entity_ids = must_include_entity_ids or set()
    overflow_ids = doc_overflow_ids or set()

    # Current doc counts
    doc_counts: Dict[int, int] = {}
    for c in current_selected:
        doc_counts[c.doc_id] = doc_counts.get(c.doc_id, 0) + 1

    # Working set cap: how many more can we admit?
    remaining_capacity = max(0, config.max_total_selected - len(current_selected))
    if remaining_capacity == 0:
        report.admitted_count = 0
        return [], report

    # ── Pass 1: pre-filter by score (top 40) to limit embedding IO ────
    scored = sorted(new_candidates, key=lambda c: c.score, reverse=True)
    pre_filtered = scored[:40]

    # ── Load embeddings lazily ────────────────────────────────────────
    need_embeddings = [c.chunk_id for c in pre_filtered if c.embedding is None]
    if need_embeddings and conn is not None:
        emb_map = load_embeddings_for_chunks(conn, need_embeddings)
        for c in pre_filtered:
            if c.embedding is None and c.chunk_id in emb_map:
                c.embedding = emb_map[c.chunk_id]

    # Also ensure selected chunks have embeddings cached
    selected_need = [c.chunk_id for c in current_selected if c.embedding is None]
    if selected_need and conn is not None:
        sel_emb_map = load_embeddings_for_chunks(conn, selected_need[:50])
        for c in current_selected:
            if c.embedding is None and c.chunk_id in sel_emb_map:
                c.embedding = sel_emb_map[c.chunk_id]

    selected_with_emb = [c for c in current_selected if c.embedding]

    # ── Compute similarity distributions for adaptive thresholds ──────
    has_seed_embedding = bool(seed_embedding and len(seed_embedding) > 10)
    sims_to_seed: List[float] = []
    max_sims_to_selected: List[float] = []

    for c in pre_filtered:
        if c.embedding:
            drift_sim = _cosine_sim(c.embedding, seed_embedding) if has_seed_embedding else 1.0
            sims_to_seed.append(drift_sim)
            if selected_with_emb:
                max_sim = max(_cosine_sim(c.embedding, s.embedding)
                              for s in selected_with_emb)
            else:
                max_sim = 0.0
            max_sims_to_selected.append(max_sim)

    report.drift_sim_histogram = sorted(sims_to_seed)
    report.dup_sim_histogram = sorted(max_sims_to_selected)

    adaptive_drift, adaptive_dup = compute_adaptive_thresholds(
        sims_to_seed, max_sims_to_selected, config)
    report.adaptive_drift_floor_used = adaptive_drift
    report.adaptive_dup_threshold_used = adaptive_dup

    # ── Collection quota setup ─────────────────────────────────────────
    # Track per-collection chunk counts for quota enforcement.
    coll_counts: Dict[str, int] = {}
    for c in current_selected:
        slug = c.collection_slug or "_unknown"
        coll_counts[slug] = coll_counts.get(slug, 0) + 1

    # Max chunks per collection = pct * max_total_selected
    # Only enforced if per_collection_max_pct > 0
    coll_quota_enabled = config.per_collection_max_pct > 0
    coll_max = int(config.per_collection_max_pct * config.max_total_selected) if coll_quota_enabled else 0

    # ── Pass 2: apply rails ───────────────────────────────────────────
    admitted: List[CandidateChunk] = []

    for c in pre_filtered:
        cid = c.chunk_id

        # Skip if already selected
        if any(s.chunk_id == cid for s in current_selected):
            continue

        has_embedding = c.embedding is not None

        # --- Near-dup rejection ---
        if has_embedding and selected_with_emb:
            max_sim = max(_cosine_sim(c.embedding, s.embedding)
                          for s in selected_with_emb)
            if max_sim > adaptive_dup:
                report.filtered_dup.append(cid)
                continue

        # --- Drift floor with must_include override ---
        if has_embedding and has_seed_embedding:
            drift_sim = _cosine_sim(c.embedding, seed_embedding)
            below_drift = drift_sim < adaptive_drift

            # Override: must_include entity bypass
            entity_bypass = bool(mi_entity_ids and
                                 set(c.entity_ids) & mi_entity_ids)
            # Override: gap embedding bypass
            gap_bypass = False
            if below_drift and gap_embedding and c.embedding:
                gap_sim = _cosine_sim(c.embedding, gap_embedding)
                if gap_sim >= adaptive_drift:
                    gap_bypass = True

            if below_drift and not entity_bypass and not gap_bypass:
                report.filtered_drift.append(cid)
                continue
        else:
            # No embedding — use score as drift proxy
            if c.score < 0.1:  # very low score = likely drift
                report.filtered_drift.append(cid)
                continue

        # --- Per-doc cap (soft + hard) ---
        soft_cap = 8 if config.roster_intent else config.per_doc_soft_cap
        current_doc_count = doc_counts.get(c.doc_id, 0)
        if current_doc_count >= config.per_doc_hard_cap:
            report.filtered_doc_cap.append(cid)
            continue
        if current_doc_count >= soft_cap:
            if c.doc_id not in overflow_ids:
                report.filtered_doc_cap.append(cid)
                continue

        # --- Per-collection quota ---
        if coll_quota_enabled:
            slug = c.collection_slug or "_unknown"
            if coll_counts.get(slug, 0) >= coll_max:
                report.filtered_collection_quota.append(cid)
                continue

        # --- Working set cap ---
        if len(admitted) >= remaining_capacity:
            break

        # Admitted
        admitted.append(c)
        doc_counts[c.doc_id] = doc_counts.get(c.doc_id, 0) + 1
        coll_slug = c.collection_slug or "_unknown"
        coll_counts[coll_slug] = coll_counts.get(coll_slug, 0) + 1

    report.admitted_count = len(admitted)
    return admitted, report
