"""
V9 Summarizer — distills newly-fetched chunks into compact evidence bullets.

Called after every fetch_chunks round.  Uses gpt-4.1-mini-2025-04-14 with structured
outputs to produce <= 6 bullets, each with provenance (supporting_chunk_ids),
plus open questions, leads, and warnings.

The summarizer is intentionally "dumb": it does NOT output doc_ids, pinned
flags, bullet_ids, or created_at.  Those are all system-derived at merge time.
"""
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from retrieval.agent.v9_types import (
    EvidenceBullet,
    EvidenceSummaryUpdate,
    WorkspaceChunk,
    compute_bullet_id,
)


# =============================================================================
# Structured-output schema for the summarizer (strict caps)
# =============================================================================

_SUMMARIZER_SCHEMA = {
    "name": "evidence_summary",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "bullets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "supporting_chunk_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "support_quote": {"type": "string"},
                    },
                    "required": ["text", "supporting_chunk_ids", "tags", "support_quote"],
                    "additionalProperties": False,
                },
            },
            "open_questions": {
                "type": "array",
                "items": {"type": "string"},
            },
            "leads": {
                "type": "array",
                "items": {"type": "string"},
            },
            "warnings": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["bullets", "open_questions", "leads", "warnings"],
        "additionalProperties": False,
    },
}

_SUMMARIZER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": _SUMMARIZER_SCHEMA,
}


# =============================================================================
# Prompt
# =============================================================================

_SUMMARIZER_PROMPT = """\
You are a research assistant summarizing newly-retrieved archival evidence.

Given the research question and a set of document chunks, produce:
- bullets (max 10): concise factual findings, each referencing which chunk_ids support it.
  COVER EVERY CHUNK: any provided chunk containing question-relevant content must
  contribute at least one bullet — do not let a famous or question-echoing passage
  crowd out a dry passage that states a specific fact, name, or date.
  Each bullet MUST have at least one supporting_chunk_id from the provided chunks.
  Tags (max 3 per bullet): short category labels like "identity", "alias", "roster",
  "timeline", "contradiction", "warning", "codename", etc.
  support_quote: the EXACT passage from the chunk that supports the bullet, COPIED
  VERBATIM character-for-character from the chunk text above (including OCR errors —
  do NOT fix spelling or punctuation). 1-2 sentences, under 350 characters, from the
  bullet's FIRST supporting chunk. This anchors an on-page highlight, so it must be a
  literal substring of the chunk text. Empty string only if no single passage supports
  the bullet.
- open_questions (max 4): unanswered questions raised by these chunks.
- leads (max 6): promising follow-up searches or chunk_ids to investigate.
- warnings (max 3): contradictions, OCR errors, or reliability concerns.

Be concise. Bullet text should be under 220 characters.
Only reference chunk_ids that appear in the provided chunks below."""


# =============================================================================
# Post-parse normalizer
# =============================================================================

_MAX_BULLET_TEXT = 220
_MAX_TAGS_PER_BULLET = 3
_MAX_TAG_LEN = 20
_MAX_BULLETS = int(os.getenv("V9_SUMMARIZER_MAX_BULLETS", "10"))
_MAX_OPEN_QUESTIONS = 4
_MAX_LEADS = 6
_MAX_WARNINGS = 3


def _coerce_int(val) -> int:
    """Try to coerce a value to int; raise ValueError if impossible."""
    if isinstance(val, int):
        return val
    if isinstance(val, float) and val == int(val):
        return int(val)
    if isinstance(val, str) and val.strip().isdigit():
        return int(val.strip())
    raise ValueError(f"Cannot coerce {val!r} to int")


# --- support_quote validation (verbatim-or-repair; never store a fabricated quote) ---

_MAX_QUOTE_LEN = 400
_QUOTE_TRANS = str.maketrans({"‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"',
                              "–": "-", "—": "-", "−": "-"})


def _norm_with_map(src: str):
    """Lowercase / unify quotes+dashes / collapse whitespace, keeping an index map
    back into the original string so validated windows can be recovered verbatim."""
    norm_chars: List[str] = []
    idx_map: List[int] = []
    last_space = True
    for i, ch in enumerate(src):
        ch = ch.translate(_QUOTE_TRANS)
        if ch == "­":  # soft hyphen
            continue
        if ch.isspace():
            if last_space:
                continue
            norm_chars.append(" ")
            idx_map.append(i)
            last_space = True
            continue
        norm_chars.append(ch.lower())
        idx_map.append(i)
        last_space = False
    if norm_chars and norm_chars[-1] == " ":
        norm_chars.pop()
        idx_map.pop()
    return "".join(norm_chars), idx_map


def _validate_or_repair_quote(quote: str, chunk_texts: List[tuple]) -> tuple:
    """Return (verbatim_quote_from_chunk, chunk_id) or ("", None).

    Tier 1: normalized substring of a supporting chunk -> recover the chunk's own
    original text for that window (so the stored quote is verbatim source text even
    if the model normalized punctuation).
    Tier 2: fuzzy token-window alignment (SequenceMatcher-free rolling overlap);
    accept the best window at >= 0.75 and store the CHUNK's text for it.
    Otherwise: no quote (highlighting falls back to page-level).
    """
    q = (quote or "").strip()[:_MAX_QUOTE_LEN]
    if len(q) < 15:
        return "", None
    q_norm, _ = _norm_with_map(q)
    q_tokens = q_norm.split(" ")
    if len(q_tokens) < 3:
        return "", None
    q_freq: Dict[str, int] = {}
    for t in q_tokens:
        q_freq[t] = q_freq.get(t, 0) + 1

    for chunk_id, text in chunk_texts:
        if not text:
            continue
        c_norm, c_map = _norm_with_map(text)
        # Tier 1: exact after normalization
        pos = c_norm.find(q_norm)
        if pos != -1:
            start, end = c_map[pos], c_map[pos + len(q_norm) - 1] + 1
            return text[start:end][:_MAX_QUOTE_LEN], chunk_id

        # Tier 2: best token window by multiset overlap
        tokens = []
        off = 0
        for tok in c_norm.split(" "):
            tokens.append((tok, off))
            off += len(tok) + 1
        w = len(q_tokens)
        if w > len(tokens):
            continue
        freq: Dict[str, int] = {}
        overlap = 0
        best = (-1.0, 0, 0)  # score, tok_start, tok_end
        for i, (tok, _o) in enumerate(tokens):
            c = freq.get(tok, 0) + 1
            freq[tok] = c
            if c <= q_freq.get(tok, 0):
                overlap += 1
            if i >= w:
                old = tokens[i - w][0]
                oc = freq[old]
                if oc <= q_freq.get(old, 0):
                    overlap -= 1
                freq[old] = oc - 1
            if i >= w - 1:
                score = overlap / w
                if score > best[0]:
                    best = (score, i - w + 1, i)
        if best[0] >= 0.75:
            s_norm = tokens[best[1]][1]
            e_norm = tokens[best[2]][1] + len(tokens[best[2]][0])
            start, end = c_map[s_norm], c_map[e_norm - 1] + 1
            return text[start:end][:_MAX_QUOTE_LEN], chunk_id
    return "", None


def _normalize_summary(
    raw: dict,
    provided_chunk_ids: Set[int],
    chunk_text_map: Optional[Dict[int, str]] = None,
) -> dict:
    """Apply hard caps, type-coerce chunk_ids, validate against provided set."""
    raw_bullet_count = len(raw.get("bullets") or [])
    clean_bullets = []
    dropped_reasons: List[str] = []
    for b in (raw.get("bullets") or [])[:_MAX_BULLETS]:
        text = (b.get("text") or "")[:_MAX_BULLET_TEXT]
        if not text.strip():
            dropped_reasons.append("empty_text")
            continue

        # Type-coerce chunk_ids
        coerced_ids = []
        for cid in (b.get("supporting_chunk_ids") or []):
            try:
                coerced_ids.append(_coerce_int(cid))
            except (ValueError, TypeError):
                continue

        # Validate against provided set
        valid_ids = [cid for cid in coerced_ids if cid in provided_chunk_ids]
        if not valid_ids:
            dropped_reasons.append(
                f"no_valid_chunk_ids(coerced={coerced_ids},provided={len(provided_chunk_ids)})"
            )
            continue

        # Cap tags
        tags = [str(t)[:_MAX_TAG_LEN] for t in (b.get("tags") or [])[:_MAX_TAGS_PER_BULLET]]

        # Compute bullet_id
        bid = compute_bullet_id(text, valid_ids)
        if not bid:
            dropped_reasons.append("empty_bullet_id")
            continue

        # Validate the support quote against the supporting chunks' own text.
        # Stored quotes are ALWAYS verbatim chunk text (validated or realigned) —
        # a paraphrase the aligner can't place is dropped, never stored.
        support_quote, quote_chunk_id = "", None
        if chunk_text_map:
            support_quote, quote_chunk_id = _validate_or_repair_quote(
                b.get("support_quote") or "",
                [(cid, chunk_text_map.get(cid, "")) for cid in valid_ids],
            )

        clean_bullets.append({
            "bullet_id": bid,
            "text": text,
            "supporting_chunk_ids": valid_ids,
            "tags": tags,
            "support_quote": support_quote,
            "quote_chunk_id": quote_chunk_id,
        })

    if dropped_reasons:
        print(
            f"  [V9] Summarizer normalizer: raw={raw_bullet_count}, "
            f"kept={len(clean_bullets)}, dropped={dropped_reasons}",
            file=sys.stderr,
        )

    return {
        "bullets": clean_bullets,
        "open_questions": [str(q)[:200] for q in (raw.get("open_questions") or [])[:_MAX_OPEN_QUESTIONS]],
        "leads": [str(l)[:200] for l in (raw.get("leads") or [])[:_MAX_LEADS]],
        "warnings": [str(w)[:200] for w in (raw.get("warnings") or [])[:_MAX_WARNINGS]],
    }


# =============================================================================
# Main entry point
# =============================================================================

# Chars of chunk text fed to the summarizer per pass. Was 4000 (~1.5 pages) —
# which silently starved evidence: a 15-chunk fetch got its first 2-3 chunks
# read and the rest never reached the model's eyes (bullets can only cite
# chunks actually fed in). 16k chars ≈ 4k tokens — trivial for the mini model.
_CHUNK_INPUT_BUDGET = int(os.getenv("V9_SUMMARIZER_INPUT_CHARS", "16000"))


def summarize_delta_chunks(
    chunks: List[WorkspaceChunk],
    question: str,
    *,
    alias_context: str = "",
    model: str = "gpt-4.1-mini-2025-04-14",
    max_completion_tokens: int = 2600,  # 10 bullets × (220 text + 350 quote); truncation under strict schema returns null
) -> EvidenceSummaryUpdate:
    """Summarize newly-fetched chunks into an EvidenceSummaryUpdate.

    alias_context: optional identity context string
      (e.g. "Known identities: PAL / Robert = Silvermaster").
      When provided, the summarizer uses canonical names and treats
      alias variants as the same individual.

    Returns an update with bullets, open_questions, leads, warnings.
    bullet_id is computed; doc_ids, pinned, pin_reason, created_at are NOT set
    (those are system-derived at merge time).
    """
    from openai import OpenAI

    if not chunks:
        return EvidenceSummaryUpdate(
            update_id=str(uuid.uuid4()),
            generated_from_chunk_ids=[],
            summarizer_model=model,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    # All chunk_ids in this batch (for generated_from tracking)
    all_chunk_ids: Set[int] = {c.chunk_id for c in chunks}

    # Build chunk text for the prompt (clipped to budget)
    # provided_chunk_ids = only those whose text appears in the prompt
    provided_chunk_ids: Set[int] = set()
    chunk_texts = []
    total_chars = 0
    for c in chunks:
        remaining = _CHUNK_INPUT_BUDGET - total_chars
        if remaining <= 0:
            break
        provided_chunk_ids.add(c.chunk_id)
        text = c.text[:remaining]
        source = c.source_label or ""
        page = c.page or ""
        chunk_texts.append(
            f"[chunk_id={c.chunk_id}] ({source} {page}): {text}"
        )
        total_chars += len(text)

    # Build user message with optional alias context
    parts = [f"Research question: {question}\n"]
    if alias_context:
        parts.append(
            f"\n{alias_context}\n"
            "Use canonical names in bullets. "
            "Treat different name forms for the same person as one individual.\n"
        )
    parts.append(f"\nChunks ({len(chunks)} total):\n")
    parts.append("\n\n".join(chunk_texts))
    user_msg = "".join(parts)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for summarizer")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SUMMARIZER_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=float(os.getenv("V9_SUMMARIZER_TEMPERATURE", "0.1")),
        max_completion_tokens=max_completion_tokens,
        response_format=_SUMMARIZER_RESPONSE_FORMAT,
    )

    choice = response.choices[0]
    finish_reason = choice.finish_reason
    content = choice.message.content

    # Detect truncation: strict-schema returns null content when truncated
    if finish_reason == "length" or not content:
        print(
            f"  [V9] Summarizer: finish_reason={finish_reason}, "
            f"content={'null' if content is None else f'{len(content)} chars'}. "
            f"Output likely truncated (max_completion_tokens={max_completion_tokens}).",
            file=sys.stderr,
        )
        content = content or "{}"

    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        print(
            f"  [V9] Summarizer: JSON parse failed on {len(content)} chars",
            file=sys.stderr,
        )
        raw = {}

    normalized = _normalize_summary(
        raw, provided_chunk_ids,
        chunk_text_map={c.chunk_id: c.text for c in chunks},
    )

    now = datetime.now(timezone.utc).isoformat()
    update_id = str(uuid.uuid4())

    bullets = [
        EvidenceBullet(
            bullet_id=b["bullet_id"],
            text=b["text"],
            supporting_chunk_ids=b["supporting_chunk_ids"],
            tags=b["tags"],
            support_quote=b.get("support_quote", ""),
            quote_chunk_id=b.get("quote_chunk_id"),
            # created_at, doc_ids, pinned, pin_reason set at merge time
        )
        for b in normalized["bullets"]
    ]

    return EvidenceSummaryUpdate(
        update_id=update_id,
        generated_from_chunk_ids=sorted(all_chunk_ids),
        summarizer_model=model,
        created_at=now,
        bullets=bullets,
        open_questions=normalized["open_questions"],
        leads=normalized["leads"],
        warnings=normalized["warnings"],
    )
