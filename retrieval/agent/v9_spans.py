"""
V9 Spans - Sentence extraction for evidence span validation.

Used by provenance-first grounding: evidence spans reference sentence_index
within a chunk. Splitting is robust to abbreviations and OCR:
- Prefer newline + punctuation
- Cap sentence length (~400 chars); split long segments on ; : or newline
"""
import re
from typing import List, Tuple


def extract_sentence_spans(text: str) -> List[Tuple[int, int]]:
    """
    Extract (start, end) character spans for each sentence in text.
    Robust to abbreviations and OCR; caps sentence length (~400 chars).

    Prefer newline + punctuation; fallback to punctuation. Long segments
    split further on ; : or newline.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    max_sentence_len = 400
    sentences: List[str] = []

    # Split on newline first, then on sentence-ending punctuation per line
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        for part in re.split(r'(?<=[.!?])\s+', line):
            part = part.strip()
            if not part:
                continue
            if len(part) > max_sentence_len:
                for sp in re.split(r'[;:\n]', part):
                    sp = sp.strip()
                    if sp:
                        sentences.append(sp)
            else:
                sentences.append(part)

    # Build spans by scanning for each sentence
    pos = 0
    spans: List[Tuple[int, int]] = []
    for sent in sentences:
        idx = text.find(sent, pos)
        if idx >= 0:
            spans.append((idx, idx + len(sent)))
            pos = idx + len(sent)
    return spans


def get_sentence_count(text: str) -> int:
    """Return number of sentences in text (for sentence_index range validation)."""
    return len(extract_sentence_spans(text))
