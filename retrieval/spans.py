"""
SpanMiner for Agentic V2.

Converts chunk text into cite-able windows (spans).

CONTRACT C1: All offsets are byte offsets into raw chunks.text (exactly as stored in DB).
Normalization is only for matching, never for offset computation.

Window types:
- Sentence windows (1-2 sentences)
- Fixed char windows (500 chars, 50% overlap)
- Mention-centered windows (around entity_mentions offsets)

Default: sentence splitting + merge short sentences to 120-600 chars.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set


# Version pinning for determinism (Contract C1)
SPAN_MINER_VERSION = "v1.0.0"
SENTENCE_SPLITTER_VERSION = "regex_v1"  # Using regex, not spacy for determinism


@dataclass
class Span:
    """
    A cite-able window from a chunk.
    
    All offsets are relative to raw chunks.text (Contract C1).
    """
    chunk_id: int
    doc_id: int
    page_ref: str  # Contract C2: f"p{page_num}", default "p0"
    start_char: int
    end_char: int
    text: str
    source_lanes: List[str] = field(default_factory=list)
    
    @property
    def span_id(self) -> str:
        """Deterministic ID: chunk_id:start:end"""
        return f"{self.chunk_id}:{self.start_char}:{self.end_char}"
    
    @property
    def span_hash(self) -> str:
        """
        SHA256 hash for reproducibility (Contract C1).
        
        Uses normalized text + coordinates for stable hash.
        """
        content = f"{self._normalize_for_hash(self.text)}|{self.chunk_id}|{self.start_char}|{self.end_char}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    @staticmethod
    def _normalize_for_hash(text: str) -> str:
        """Normalize text for hashing (whitespace folding, lowercase)."""
        return ' '.join(text.lower().split())
    
    def __hash__(self):
        return hash(self.span_id)
    
    def __eq__(self, other):
        if not isinstance(other, Span):
            return False
        return self.span_id == other.span_id


class SpanMiner:
    """
    Converts chunk text into cite-able windows.
    
    Priority order for span selection (Contract C3 - mention-centered windows):
    1. Mention-centered spans (around entity_mentions offsets)
    2. Normalized anchor term matches (fold whitespace, ASCII, lowercase)
    3. Evenly spaced (deterministic by index)
    """
    
    def __init__(
        self,
        min_chars: int = 120,
        max_chars: int = 600,
        max_spans_per_chunk: int = 12,
        sentence_overlap: int = 1,  # sentences to overlap between windows
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.max_spans_per_chunk = max_spans_per_chunk
        self.sentence_overlap = sentence_overlap
        self.version = SPAN_MINER_VERSION
    
    def mine_spans(
        self,
        chunk_id: int,
        doc_id: int,
        page_ref: str,
        text: str,
        mention_offsets: List[Tuple[int, int, int]] = None,  # (start, end, entity_id)
        anchor_terms: List[str] = None,
        source_lanes: List[str] = None,
    ) -> List[Span]:
        """
        Mine spans from a single chunk.
        
        Args:
            chunk_id: ID of the chunk
            doc_id: ID of the document
            page_ref: Page reference (Contract C2)
            text: Raw chunk text (Contract C1 - use exactly as stored)
            mention_offsets: Entity mention offsets from entity_mentions table
            anchor_terms: Terms to prioritize (from query/targets)
            source_lanes: Retrieval lanes that found this chunk
        
        Returns:
            List of Span objects, capped at max_spans_per_chunk
        """
        if not text or not text.strip():
            return []
        
        source_lanes = source_lanes or []
        
        # Step 1: Split into sentence windows
        sentence_windows = self._split_into_sentences(text)
        
        # Step 2: Merge short sentences to target length
        merged_windows = self._merge_windows(sentence_windows, text)
        
        # Step 3: Create spans from windows
        all_spans = []
        for start, end in merged_windows:
            span_text = text[start:end]
            if len(span_text.strip()) >= 20:  # minimum viable span
                all_spans.append(Span(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    page_ref=page_ref,
                    start_char=start,
                    end_char=end,
                    text=span_text,
                    source_lanes=source_lanes.copy(),
                ))
        
        # Step 4: Prioritize and cap spans
        prioritized = self._prioritize_spans(
            all_spans, 
            mention_offsets or [], 
            anchor_terms or [],
        )
        
        return prioritized[:self.max_spans_per_chunk]
    
    def mine_chunks(
        self,
        chunks: List[dict],  # Each has chunk_id, doc_id, text, page_ref, source_lanes
        mention_offsets_by_chunk: dict = None,  # chunk_id -> [(start, end, entity_id)]
        anchor_terms: List[str] = None,
    ) -> List[Span]:
        """
        Mine spans from multiple chunks.
        
        Args:
            chunks: List of chunk dicts with keys: chunk_id, doc_id, text, page_ref, source_lanes
            mention_offsets_by_chunk: Pre-fetched mention offsets by chunk_id
            anchor_terms: Terms to prioritize
        
        Returns:
            All mined spans (not deduplicated)
        """
        all_spans = []
        mention_offsets_by_chunk = mention_offsets_by_chunk or {}
        
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id') or chunk.get('id')
            doc_id = chunk.get('doc_id') or chunk.get('document_id')
            text = chunk.get('text', '')
            page_ref = chunk.get('page_ref', 'p0')
            source_lanes = chunk.get('source_lanes', [])
            
            mention_offsets = mention_offsets_by_chunk.get(chunk_id, [])
            
            spans = self.mine_spans(
                chunk_id=chunk_id,
                doc_id=doc_id,
                page_ref=page_ref,
                text=text,
                mention_offsets=mention_offsets,
                anchor_terms=anchor_terms,
                source_lanes=source_lanes,
            )
            all_spans.extend(spans)
        
        return all_spans
    
    def _split_into_sentences(self, text: str) -> List[Tuple[int, int]]:
        """
        Split text into sentence boundaries.
        
        Returns list of (start_char, end_char) tuples.
        Uses regex for determinism (not spacy).
        """
        # Sentence-ending patterns
        # Match period/question/exclamation followed by space and capital letter
        # Also handle end of text
        pattern = r'[.!?]+(?:\s+|$)'
        
        sentences = []
        start = 0
        
        for match in re.finditer(pattern, text):
            end = match.end()
            if end > start:
                sentences.append((start, end))
            start = end
        
        # Handle remaining text
        if start < len(text):
            sentences.append((start, len(text)))
        
        return sentences
    
    def _merge_windows(
        self, 
        sentence_windows: List[Tuple[int, int]], 
        text: str,
    ) -> List[Tuple[int, int]]:
        """
        Merge sentence windows to reach target length.
        
        Merges consecutive short sentences until min_chars is reached,
        but doesn't exceed max_chars.
        """
        if not sentence_windows:
            return []
        
        merged = []
        current_start = sentence_windows[0][0]
        current_end = sentence_windows[0][1]
        
        for i in range(1, len(sentence_windows)):
            next_start, next_end = sentence_windows[i]
            current_len = current_end - current_start
            next_len = next_end - current_end
            
            # If current window is too short, merge with next
            if current_len < self.min_chars and current_len + next_len <= self.max_chars:
                current_end = next_end
            else:
                # Save current window if it's viable
                if current_end - current_start >= 20:
                    merged.append((current_start, current_end))
                # Start new window (with overlap)
                if self.sentence_overlap > 0 and i > 0:
                    # Try to start from a previous sentence for overlap
                    overlap_start = max(0, i - self.sentence_overlap)
                    current_start = sentence_windows[overlap_start][0]
                else:
                    current_start = next_start
                current_end = next_end
        
        # Don't forget the last window
        if current_end - current_start >= 20:
            merged.append((current_start, current_end))
        
        return merged
    
    def _prioritize_spans(
        self,
        spans: List[Span],
        mention_offsets: List[Tuple[int, int, int]],
        anchor_terms: List[str],
    ) -> List[Span]:
        """
        Prioritize spans for selection.
        
        Priority order:
        1. Mention-centered spans (contain entity_mentions)
        2. Anchor term spans (contain query terms)
        3. Evenly spaced (deterministic by index)
        """
        if not spans:
            return []
        
        # Score each span
        scored = []
        for i, span in enumerate(spans):
            score = 0
            
            # Priority 1: Contains mention offsets
            for m_start, m_end, _ in mention_offsets:
                if span.start_char <= m_end and span.end_char >= m_start:
                    score += 100
                    break
            
            # Priority 2: Contains anchor terms (normalized match)
            span_text_norm = self._normalize_for_match(span.text)
            for term in anchor_terms:
                term_norm = self._normalize_for_match(term)
                if term_norm in span_text_norm:
                    score += 50
                    break
            
            # Priority 3: Position-based (prefer variety)
            # Give slight preference to evenly distributed spans
            position_score = 10 - abs(i - len(spans) // 2)
            score += position_score
            
            scored.append((score, i, span))
        
        # Sort by score descending, then by index for determinism
        scored.sort(key=lambda x: (-x[0], x[1]))
        
        return [s[2] for s in scored]
    
    @staticmethod
    def _normalize_for_match(text: str) -> str:
        """Normalize text for matching (lowercase, fold whitespace)."""
        return ' '.join(text.lower().split())
    
    def get_version_info(self) -> dict:
        """Get version info for reproducibility."""
        return {
            "span_miner_version": self.version,
            "sentence_splitter": SENTENCE_SPLITTER_VERSION,
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
            "max_spans_per_chunk": self.max_spans_per_chunk,
        }


def get_mention_offsets_for_chunks(conn, chunk_ids: List[int]) -> dict:
    """
    Fetch entity_mentions offsets for multiple chunks.
    
    Returns dict: chunk_id -> [(start_char, end_char, entity_id)]
    """
    if not chunk_ids:
        return {}
    
    cur = conn.cursor()
    cur.execute("""
        SELECT chunk_id, start_char, end_char, entity_id
        FROM entity_mentions
        WHERE chunk_id = ANY(%s) AND start_char IS NOT NULL
        ORDER BY chunk_id, start_char
    """, (chunk_ids,))
    
    result = {}
    for chunk_id, start_char, end_char, entity_id in cur.fetchall():
        if chunk_id not in result:
            result[chunk_id] = []
        result[chunk_id].append((start_char, end_char, entity_id))
    
    return result


def get_page_ref(page_num: Optional[int]) -> str:
    """
    Get page_ref string from page_num (Contract C2).
    
    Returns "p{page_num}" or "p0" if None.
    """
    if page_num is not None:
        return f"p{page_num}"
    return "p0"
