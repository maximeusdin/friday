"""
V9 Think Deeper — FindingStore.

Judge-fed, deterministically managed.  Source of truth is validated Judge
findings; the deterministic part is dedup + validation + labeling + storage.

Seeded at step 0 from existing v9 evidence summarizer bullets.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from retrieval.agent.v9_deep_types import (
    Finding,
    FindingEntry,
    compute_finding_id,
)


class FindingStore:
    """Deterministic, deduped finding accumulator.

    Updated each step via ``add_from_judge_findings()`` — the Judge produces
    findings semantically, the controller validates chunk_ids and stores them
    here deterministically.
    """

    def __init__(self) -> None:
        self.entries: List[FindingEntry] = []
        self._seen_hashes: Set[str] = set()

    # ── seeding ──────────────────────────────────────────────────────────

    @classmethod
    def seed_from_evidence_summary(
        cls,
        evidence_memory: list,  # List[EvidenceSummaryUpdate]
    ) -> "FindingStore":
        """Seed step 0 from existing v9 evidence summarizer bullets.

        Each ``EvidenceBullet`` becomes a FindingEntry with
        ``finding_type='baseline'``.  This gives the Judge a stable baseline
        to compare against without any new LLM calls.
        """
        store = cls()
        now = datetime.now(timezone.utc).isoformat()
        for update in evidence_memory:
            for bullet in getattr(update, "bullets", []):
                text = getattr(bullet, "text", "")
                cids = getattr(bullet, "supporting_chunk_ids", [])
                if not text or not cids:
                    continue
                fid = compute_finding_id(text, cids)
                if fid in store._seen_hashes:
                    continue
                store._seen_hashes.add(fid)
                # Derive doc_ids from bullet if available
                doc_ids = list(getattr(bullet, "doc_ids", []))
                store.entries.append(FindingEntry(
                    finding_id=fid,
                    text=text,
                    finding_type="baseline",
                    supporting_chunk_ids=list(cids),
                    source_doc_ids=doc_ids,
                    source_step=0,
                    created_at=now,
                ))
        return store

    # ── adding validated Judge findings ──────────────────────────────────

    def add_from_judge_findings(
        self,
        findings: List[Finding],
        step: int,
        valid_chunk_ids: Set[int],
    ) -> int:
        """Add validated Judge findings to the store.

        1. Filter: drop findings with any cited_chunk_id not in valid_chunk_ids.
        2. Dedup: skip findings whose hash matches existing entries.
        3. Label: assign finding_id, source_step, created_at.

        Returns count of genuinely new findings added (used for stall guard).
        """
        now = datetime.now(timezone.utc).isoformat()
        added = 0
        for f in findings:
            # Validate chunk_ids — ALL must be in valid set
            cited = [int(c) for c in f.cited_chunk_ids]
            if not cited:
                continue
            if not all(c in valid_chunk_ids for c in cited):
                # Drop findings with any invalid chunk_id
                continue

            fid = compute_finding_id(f.text, cited)
            if fid in self._seen_hashes:
                continue

            self._seen_hashes.add(fid)
            self.entries.append(FindingEntry(
                finding_id=fid,
                text=f.text,
                finding_type=f.finding_type or "context",
                supporting_chunk_ids=cited,
                source_doc_ids=[],  # populated by caller if needed
                source_step=step,
                created_at=now,
            ))
            added += 1
        return added

    # ── queries ──────────────────────────────────────────────────────────

    def summary_for_judge(self, max_entries: int = 15) -> str:
        """Generate a stable text summary for Judge consumption.

        Sorted by step (oldest first), grouped by finding_type.
        Output is deterministic given the same entries.
        """
        if not self.entries:
            return "(no findings yet)"

        # Sort: step asc, finding_type asc, finding_id asc (stable)
        ordered = sorted(
            self.entries,
            key=lambda e: (e.source_step, e.finding_type, e.finding_id),
        )
        # Cap
        shown = ordered[:max_entries]

        lines: List[str] = []
        current_type = ""
        for e in shown:
            if e.finding_type != current_type:
                current_type = e.finding_type
                lines.append(f"\n[{current_type}]")
            cids_str = ",".join(str(c) for c in e.supporting_chunk_ids[:3])
            lines.append(f"  - {e.text}  (chunks: {cids_str})")

        if len(ordered) > max_entries:
            lines.append(f"\n  ... and {len(ordered) - max_entries} more findings")

        return "\n".join(lines)

    def entries_since_step(self, step: int) -> List[FindingEntry]:
        """New findings added since (inclusive) a given step."""
        return [e for e in self.entries if e.source_step >= step]

    def new_findings_count_last_n_steps(self, n: int, current_step: int) -> int:
        """Count of new findings added in the last *n* steps."""
        cutoff = max(0, current_step - n + 1)
        return sum(1 for e in self.entries if e.source_step >= cutoff)

    def doc_id_coverage(self) -> Dict[int, int]:
        """Count of findings per source doc."""
        cov: Dict[int, int] = {}
        for e in self.entries:
            for did in e.source_doc_ids:
                cov[did] = cov.get(did, 0) + 1
        return cov

    def total_count(self) -> int:
        return len(self.entries)

    def to_brief(
        self,
        top_n: int = 10,
        chunk_id_to_label: Optional[Dict[int, str]] = None,
    ) -> str:
        """Build a Findings Brief for synthesis scaffold.

        Format: numbered list with chunk_id + label + finding text.
        Pass chunk_id_to_label from selected_chunks: {chunk_id: "Vassiliev p3072"}.
        """
        return build_findings_brief(self.entries, top_n, chunk_id_to_label)


def build_findings_brief(
    entries: List[FindingEntry],
    top_n: int = 10,
    chunk_id_to_label: Optional[Dict[int, str]] = None,
) -> str:
    """Build a Findings Brief from FindingEntry list for synthesis scaffold."""
    if not entries:
        return ""
    chunk_id_to_label = chunk_id_to_label or {}
    # Prefer non-baseline, sort by step (newest first for relevance)
    ordered = sorted(
        entries,
        key=lambda e: (e.finding_type == "baseline", -e.source_step),
    )
    shown = ordered[:top_n]
    lines = [
        "## Findings Brief (cite these chunk_ids)",
        "Use the Findings Brief above as the primary scaffold. Each claim should cite chunk_ids from the brief.",
        "",
    ]
    for i, e in enumerate(shown, 1):
        cid = e.supporting_chunk_ids[0] if e.supporting_chunk_ids else 0
        label = chunk_id_to_label.get(cid, "") if cid else ""
        label_str = f", {label}" if label else ""
        lines.append(f"{i}. [chunk_id={cid}{label_str}] {e.text[:200]}{'...' if len(e.text) > 200 else ''}")
    return "\n".join(lines)
