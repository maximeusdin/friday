"""
OCR Utilities Module

Provides:
- Weighted edit distance with OCR confusion table
- Context feature extraction
- Anchoring logic
- Clustering utilities
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from functools import lru_cache


# =============================================================================
# OCR CONFUSION TABLE
# =============================================================================

# Default OCR confusions (loaded from DB at runtime, but have fallback)
DEFAULT_OCR_CONFUSIONS = {
    # Multi-char substitutions (pattern_from -> pattern_to -> cost)
    ('rn', 'm'): 0.2,
    ('m', 'rn'): 0.2,
    ('cl', 'd'): 0.3,
    ('d', 'cl'): 0.3,
    ('vv', 'w'): 0.2,
    ('w', 'vv'): 0.2,
    ('ri', 'n'): 0.3,
    ('n', 'ri'): 0.3,
    ('nn', 'm'): 0.3,
    ('ii', 'u'): 0.3,
    ('li', 'h'): 0.4,
    ('fi', 'h'): 0.4,
    
    # Single char substitutions
    ('l', '1'): 0.1,
    ('1', 'l'): 0.1,
    ('I', 'l'): 0.1,
    ('l', 'I'): 0.1,
    ('O', '0'): 0.1,
    ('0', 'O'): 0.1,
    ('S', '5'): 0.2,
    ('5', 'S'): 0.2,
    ('Z', '2'): 0.3,
    ('B', '8'): 0.3,
    ('e', 'c'): 0.4,
    ('a', 'o'): 0.4,
    ('u', 'n'): 0.3,
    ('h', 'b'): 0.4,
    
    # Punctuation/space drops (empty target = deletion cost)
    ('.', ''): 0.1,
    (',', ''): 0.1,
    ("'", ''): 0.1,
    ('-', ''): 0.1,
    (' ', ''): 0.2,
}


class OCRConfusionTable:
    """Manages OCR confusion patterns for weighted edit distance."""
    
    def __init__(self, confusions: Optional[Dict[Tuple[str, str], float]] = None):
        self.confusions = confusions or DEFAULT_OCR_CONFUSIONS
        self._build_indexes()
    
    def _build_indexes(self):
        """Build indexes for fast lookup."""
        # Group by source pattern length for efficient matching
        self.by_source_len: Dict[int, Dict[str, List[Tuple[str, float]]]] = {}
        
        for (src, tgt), cost in self.confusions.items():
            src_len = len(src)
            if src_len not in self.by_source_len:
                self.by_source_len[src_len] = {}
            if src not in self.by_source_len[src_len]:
                self.by_source_len[src_len][src] = []
            self.by_source_len[src_len][src].append((tgt, cost))
        
        # Max source pattern length
        self.max_pattern_len = max(len(s) for s, _ in self.confusions.keys()) if self.confusions else 1
    
    def get_substitution_cost(self, src: str, tgt: str) -> float:
        """Get cost of substituting src with tgt. Returns 1.0 if no special rule."""
        key = (src, tgt)
        if key in self.confusions:
            return self.confusions[key]
        
        # Check case-insensitive for single chars
        if len(src) == 1 and len(tgt) == 1:
            if src.lower() == tgt.lower():
                return 0.1  # Case change is cheap
        
        return 1.0  # Default substitution cost
    
    def get_possible_substitutions(self, src: str) -> List[Tuple[str, float]]:
        """Get all possible substitutions for a source pattern."""
        src_len = len(src)
        if src_len in self.by_source_len and src in self.by_source_len[src_len]:
            return self.by_source_len[src_len][src]
        return []
    
    @classmethod
    def from_database(cls, conn) -> 'OCRConfusionTable':
        """Load confusion table from database."""
        cur = conn.cursor()
        cur.execute("SELECT pattern_from, pattern_to, weight FROM ocr_confusions")
        confusions = {(row[0], row[1]): float(row[2]) for row in cur.fetchall()}
        return cls(confusions) if confusions else cls()


# Global instance (lazy loaded)
_confusion_table: Optional[OCRConfusionTable] = None


def get_confusion_table(conn=None) -> OCRConfusionTable:
    """Get the OCR confusion table, loading from DB if available."""
    global _confusion_table
    if _confusion_table is None:
        if conn:
            try:
                _confusion_table = OCRConfusionTable.from_database(conn)
            except:
                _confusion_table = OCRConfusionTable()
        else:
            _confusion_table = OCRConfusionTable()
    return _confusion_table


def weighted_edit_distance(
    s1: str,
    s2: str,
    confusion_table: Optional[OCRConfusionTable] = None
) -> float:
    """
    Compute weighted edit distance using OCR confusion costs.
    
    Uses dynamic programming with:
    - Standard insert/delete costs of 1.0
    - Weighted substitution costs from confusion table
    - Multi-character pattern matching for common OCR errors
    
    Returns the minimum cost to transform s1 into s2.
    """
    if confusion_table is None:
        confusion_table = get_confusion_table()
    
    if not s1:
        return float(len(s2))
    if not s2:
        return float(len(s1))
    
    m, n = len(s1), len(s2)
    
    # DP table: dp[i][j] = cost to transform s1[:i] to s2[:j]
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = float(i)
    for j in range(n + 1):
        dp[0][j] = float(j)
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Standard operations
            delete_cost = dp[i-1][j] + 1.0
            insert_cost = dp[i][j-1] + 1.0
            
            # Single-char substitution
            if s1[i-1] == s2[j-1]:
                sub_cost = dp[i-1][j-1]
            else:
                sub_cost = dp[i-1][j-1] + confusion_table.get_substitution_cost(s1[i-1], s2[j-1])
            
            dp[i][j] = min(delete_cost, insert_cost, sub_cost)
            
            # Multi-character pattern matching
            max_look = min(confusion_table.max_pattern_len, i, j)
            for look in range(2, max_look + 1):
                src_pattern = s1[i-look:i]
                tgt_pattern = s2[j-look:j]
                
                # Check if this is a known confusion
                if src_pattern != tgt_pattern:
                    pattern_cost = confusion_table.get_substitution_cost(src_pattern, tgt_pattern)
                    if pattern_cost < 1.0:  # It's a known confusion
                        multi_cost = dp[i-look][j-look] + pattern_cost
                        dp[i][j] = min(dp[i][j], multi_cost)
    
    return dp[m][n]


def normalized_weighted_edit_distance(
    s1: str,
    s2: str,
    confusion_table: Optional[OCRConfusionTable] = None
) -> float:
    """
    Normalized weighted edit distance (0 = identical, 1 = very different).
    """
    if not s1 and not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    return weighted_edit_distance(s1, s2, confusion_table) / max_len


# =============================================================================
# CONTEXT FEATURES (Phase 3A)
# =============================================================================

# Hint patterns
PERSON_HINTS = {
    'mr', 'mrs', 'ms', 'miss', 'dr', 'prof', 'professor',
    'gen', 'general', 'col', 'colonel', 'maj', 'major',
    'capt', 'captain', 'lt', 'lieutenant', 'sgt', 'sergeant',
    'sen', 'senator', 'rep', 'representative', 'hon', 'honorable',
    'rev', 'reverend', 'fr', 'father', 'sr', 'sister',
    'said', 'testified', 'stated', 'replied', 'answered',
    'witness', 'defendant', 'plaintiff', 'agent', 'officer'
}

ORG_HINTS = {
    'bureau', 'department', 'dept', 'office', 'committee', 'commission',
    'agency', 'administration', 'division', 'section', 'branch', 'unit',
    'company', 'corp', 'corporation', 'inc', 'incorporated', 'ltd', 'limited',
    'co', 'llc', 'assoc', 'association', 'society', 'institute',
    'university', 'college', 'school', 'foundation', 'organization',
    'party', 'union', 'federation', 'league', 'council',
    'federal', 'national', 'state', 'central', 'headquarters', 'hq'
}

LOC_HINTS = {
    'in', 'at', 'near', 'from', 'to',
    'city', 'town', 'county', 'state', 'country', 'nation',
    'street', 'avenue', 'ave', 'road', 'rd', 'blvd', 'boulevard',
    'building', 'floor', 'room', 'office',
    # US state abbreviations
    'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga',
    'hi', 'id', 'il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 'md',
    'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj',
    'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc',
    'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy', 'dc'
}

# Initials pattern
INITIALS_PATTERN = re.compile(r'^[A-Z]\.\s*[A-Z]\.?$')


@dataclass
class ContextFeatures:
    """Context features extracted around a mention span."""
    person_hints: int = 0
    org_hints: int = 0
    loc_hints: int = 0
    has_initials: bool = False
    in_boilerplate: bool = False
    context_tokens: List[str] = field(default_factory=list)
    
    @property
    def best_type_hint(self) -> Optional[str]:
        """Return the strongest type hint."""
        if self.person_hints > self.org_hints and self.person_hints > self.loc_hints:
            return 'person'
        elif self.org_hints > self.loc_hints:
            return 'organization'
        elif self.loc_hints > 0:
            return 'location'
        return None
    
    @property
    def context_score(self) -> float:
        """Compute aggregate context score (0-1)."""
        hints = self.person_hints + self.org_hints + self.loc_hints
        if hints == 0:
            return 0.5  # Neutral
        # More hints = higher confidence
        return min(1.0, 0.5 + 0.1 * hints)
    
    def to_dict(self) -> dict:
        return {
            'person': self.person_hints,
            'org': self.org_hints,
            'loc': self.loc_hints,
            'initials': self.has_initials,
            'boilerplate': self.in_boilerplate
        }


def extract_context_features(
    text: str,
    span_start: int,
    span_end: int,
    window_tokens: int = 8
) -> ContextFeatures:
    """
    Extract context features from text around a span.
    
    Args:
        text: Full text
        span_start: Character start of mention span
        span_end: Character end of mention span
        window_tokens: Number of tokens to look in each direction
    
    Returns:
        ContextFeatures object
    """
    features = ContextFeatures()
    
    # Get window around span
    # Find token boundaries
    tokens_before = []
    tokens_after = []
    
    # Tokenize before span
    before_text = text[:span_start].split()
    tokens_before = before_text[-window_tokens:] if before_text else []
    
    # Tokenize after span
    after_text = text[span_end:].split()
    tokens_after = after_text[:window_tokens] if after_text else []
    
    features.context_tokens = tokens_before + tokens_after
    
    # Count hints
    for token in features.context_tokens:
        token_lower = token.lower().rstrip('.,;:!?')
        if token_lower in PERSON_HINTS:
            features.person_hints += 1
        if token_lower in ORG_HINTS:
            features.org_hints += 1
        if token_lower in LOC_HINTS:
            features.loc_hints += 1
        if INITIALS_PATTERN.match(token):
            features.has_initials = True
    
    # Check for boilerplate indicators
    span_text = text[span_start:span_end].lower()
    boilerplate_terms = {'page', 'deleted', 'redacted', 'classified', 'copy', 'continued'}
    if any(term in span_text for term in boilerplate_terms):
        features.in_boilerplate = True
    
    return features


# =============================================================================
# CLUSTERING UTILITIES (Phase 4)
# =============================================================================

def compute_variant_key(surface: str) -> str:
    """
    Compute a normalized key for variant clustering.
    
    More aggressive normalization than surface_norm:
    - Lowercase
    - Remove all non-alphanumeric
    - Collapse spaces
    """
    key = surface.lower()
    key = re.sub(r'[^a-z0-9\s]', '', key)
    key = re.sub(r'\s+', '', key)  # Remove all spaces for key
    return key


def are_variants_similar(
    v1: str,
    v2: str,
    confusion_table: Optional[OCRConfusionTable] = None,
    threshold: float = 0.3
) -> bool:
    """
    Check if two variants are similar enough to cluster.
    
    Uses normalized weighted edit distance.
    """
    if not v1 or not v2:
        return False
    
    # Quick length check
    len_diff = abs(len(v1) - len(v2))
    max_len = max(len(v1), len(v2))
    if len_diff / max_len > 0.5:  # Too different in length
        return False
    
    # Token overlap check
    t1, t2 = set(v1.lower().split()), set(v2.lower().split())
    if t1 and t2:
        overlap = len(t1 & t2) / len(t1 | t2)
        if overlap < 0.3 and len(t1) > 1:  # Multi-word with low overlap
            return False
    
    # Weighted edit distance
    dist = normalized_weighted_edit_distance(v1, v2, confusion_table)
    return dist <= threshold


class VariantClusterer:
    """
    Clusters variant strings using OCR-weighted similarity.
    
    Uses Union-Find for efficient connected components.
    """
    
    def __init__(self, confusion_table: Optional[OCRConfusionTable] = None, threshold: float = 0.3):
        self.confusion_table = confusion_table or get_confusion_table()
        self.threshold = threshold
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}
        self.variants: Set[str] = set()
    
    def _find(self, x: str) -> str:
        """Find with path compression."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self._find(self.parent[x])
        return self.parent[x]
    
    def _union(self, x: str, y: str):
        """Union by rank."""
        px, py = self._find(x), self._find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
    
    def add_variant(self, variant: str):
        """Add a variant to consider for clustering."""
        self.variants.add(variant)
        self._find(variant)  # Initialize
    
    def cluster(self) -> Dict[str, List[str]]:
        """
        Perform clustering and return clusters.
        
        Returns dict mapping cluster_id -> list of variants.
        """
        variants_list = list(self.variants)
        n = len(variants_list)
        
        # Compare all pairs (O(n^2) but n should be manageable)
        for i in range(n):
            for j in range(i + 1, n):
                if are_variants_similar(
                    variants_list[i], variants_list[j],
                    self.confusion_table, self.threshold
                ):
                    self._union(variants_list[i], variants_list[j])
        
        # Group by root
        clusters: Dict[str, List[str]] = {}
        for v in variants_list:
            root = self._find(v)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(v)
        
        return clusters
    
    def get_cluster_for(self, variant: str) -> Optional[str]:
        """Get cluster ID (root) for a variant."""
        if variant in self.variants:
            return self._find(variant)
        return None


# =============================================================================
# PRIORITY SCORING (Phase 3C)
# =============================================================================

def compute_priority_score(
    doc_count: int,
    mention_count: int,
    avg_quality: float = 0.5,
    has_tier1_match: bool = False,
    has_danger_flags: bool = False
) -> float:
    """
    Compute priority score for a cluster/variant.
    
    Higher = more important to review.
    """
    # Base: log-scaled document frequency
    import math
    doc_score = math.log(doc_count + 1) * 10
    
    # Mention volume bonus
    mention_score = math.log(mention_count + 1) * 5
    
    # Quality bonus
    quality_bonus = avg_quality * 20
    
    # Tier 1 match = high value
    tier_bonus = 30 if has_tier1_match else 0
    
    # Danger penalty (needs careful review)
    danger_penalty = -20 if has_danger_flags else 0
    
    return doc_score + mention_score + quality_bonus + tier_bonus + danger_penalty
