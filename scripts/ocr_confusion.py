#!/usr/bin/env python3
"""
OCR confusion weights and weighted edit distance.

Implements OCR-aware edit distance using weighted substitutions
based on common OCR error patterns.
"""

from typing import Dict, Tuple, Optional
import yaml
from pathlib import Path


# Default OCR confusion weights
# Format: (char1, char2): weight (lower = more likely confusion)
DEFAULT_OCR_CONFUSION = {
    # Common character substitutions
    ('rn', 'm'): 0.3,  # "rn" often misread as "m"
    ('m', 'rn'): 0.3,
    ('cl', 'd'): 0.3,  # "cl" often misread as "d"
    ('d', 'cl'): 0.3,
    ('vv', 'w'): 0.3,  # "vv" often misread as "w"
    ('w', 'vv'): 0.3,
    ('ii', 'h'): 0.3,  # "ii" sometimes misread as "h"
    ('h', 'ii'): 0.3,
    
    # Single character confusions
    ('l', '1'): 0.2,  # lowercase L vs one
    ('1', 'l'): 0.2,
    ('I', 'l'): 0.2,  # uppercase I vs lowercase L
    ('l', 'I'): 0.2,
    ('I', '1'): 0.2,
    ('1', 'I'): 0.2,
    ('O', '0'): 0.2,  # letter O vs zero
    ('0', 'O'): 0.2,
    ('S', '5'): 0.2,  # letter S vs five
    ('5', 'S'): 0.2,
    ('Z', '2'): 0.2,  # letter Z vs two
    ('2', 'Z'): 0.2,
    ('H', 'N'): 0.4,  # H and N sometimes confused
    ('N', 'H'): 0.4,
    ('E', 'F'): 0.4,
    ('F', 'E'): 0.4,
    ('C', 'G'): 0.4,
    ('G', 'C'): 0.4,
    
    # Character drops (low cost for missing characters)
    ("'", ''): 0.1,  # apostrophe often dropped
    ('-', ''): 0.1,  # hyphen often dropped
    ('.', ''): 0.15,  # period sometimes dropped
    (',', ''): 0.15,  # comma sometimes dropped
    
    # Character additions (less common, higher cost)
    ('', "'"): 0.2,
    ('', '-'): 0.2,
    ('', '.'): 0.25,
    ('', ','): 0.25,
}


def load_confusion_weights(config_path: Optional[str] = None) -> Dict[Tuple[str, str], float]:
    """
    Load OCR confusion weights from config file or use defaults.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Dict mapping (char1, char2) tuples to confusion weights
    """
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file) as f:
                config = yaml.safe_load(f)
                if 'confusion_weights' in config:
                    # Convert list of dicts to tuple keys
                    weights = {}
                    for item in config['confusion_weights']:
                        key = tuple(item['chars'])
                        weights[key] = item['weight']
                    return weights
    
    return DEFAULT_OCR_CONFUSION.copy()


def get_confusion_cost(char1: str, char2: str, confusion_weights: Dict[Tuple[str, str], float]) -> float:
    """
    Get the cost of substituting char1 with char2 based on OCR confusion weights.
    
    Args:
        char1: First character
        char2: Second character
        confusion_weights: Dict of confusion weights
    
    Returns:
        Cost (0.0-1.0), where 0.0 = very likely confusion, 1.0 = unlikely
    """
    # Exact match
    if char1 == char2:
        return 0.0
    
    # Check direct substitution
    key = (char1, char2)
    if key in confusion_weights:
        return confusion_weights[key]
    
    # Check reverse (symmetric)
    key_rev = (char2, char1)
    if key_rev in confusion_weights:
        return confusion_weights[key_rev]
    
    # Default cost for unknown substitutions
    return 1.0


def weighted_levenshtein(
    s1: str,
    s2: str,
    confusion_weights: Optional[Dict[Tuple[str, str], float]] = None
) -> float:
    """
    Compute weighted Levenshtein distance using OCR confusion weights.
    
    Args:
        s1: First string
        s2: Second string
        confusion_weights: Optional confusion weights (uses defaults if None)
    
    Returns:
        Weighted edit distance
    """
    if confusion_weights is None:
        confusion_weights = DEFAULT_OCR_CONFUSION
    
    # Normalize strings
    s1 = s1.lower()
    s2 = s2.lower()
    
    # Base case
    if len(s1) == 0:
        return len(s2) * 1.0  # Insertion cost
    if len(s2) == 0:
        return len(s1) * 1.0  # Deletion cost
    
    # Initialize matrix
    rows = len(s1) + 1
    cols = len(s2) + 1
    dist = [[0.0] * cols for _ in range(rows)]
    
    # Initialize first row and column
    for i in range(1, rows):
        dist[i][0] = i * 1.0  # Deletion cost
    for j in range(1, cols):
        dist[0][j] = j * 1.0  # Insertion cost
    
    # Fill matrix
    for i in range(1, rows):
        for j in range(1, cols):
            # Cost of substitution
            if s1[i-1] == s2[j-1]:
                cost = 0.0
            else:
                cost = get_confusion_cost(s1[i-1], s2[j-1], confusion_weights)
            
            dist[i][j] = min(
                dist[i-1][j] + 1.0,      # Deletion
                dist[i][j-1] + 1.0,      # Insertion
                dist[i-1][j-1] + cost    # Substitution
            )
    
    return dist[rows-1][cols-1]


def normalized_weighted_levenshtein(
    s1: str,
    s2: str,
    confusion_weights: Optional[Dict[Tuple[str, str], float]] = None
) -> float:
    """
    Compute normalized weighted Levenshtein distance (0.0-1.0).
    
    Returns:
        Normalized distance (0.0 = identical, 1.0 = completely different)
    """
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    
    distance = weighted_levenshtein(s1, s2, confusion_weights)
    return distance / max_len


def damerau_levenshtein_weighted(
    s1: str,
    s2: str,
    confusion_weights: Optional[Dict[Tuple[str, str], float]] = None
) -> float:
    """
    Compute weighted Damerau-Levenshtein distance (includes transpositions).
    
    This is more expensive but handles OCR transposition errors better.
    """
    if confusion_weights is None:
        confusion_weights = DEFAULT_OCR_CONFUSION
    
    s1 = s1.lower()
    s2 = s2.lower()
    
    # Base cases
    if len(s1) == 0:
        return len(s2) * 1.0
    if len(s2) == 0:
        return len(s1) * 1.0
    
    # Initialize matrix (need extra row/col for transpositions)
    rows = len(s1) + 2
    cols = len(s2) + 2
    dist = [[float('inf')] * cols for _ in range(rows)]
    
    # Initialize
    max_dist = len(s1) + len(s2)
    dist[0][0] = max_dist
    
    for i in range(0, len(s1) + 1):
        dist[i + 1][0] = max_dist
        dist[i + 1][1] = i * 1.0
    
    for j in range(0, len(s2) + 1):
        dist[0][j + 1] = max_dist
        dist[1][j + 1] = j * 1.0
    
    # Character last positions (for transpositions)
    da = {}
    for char in set(s1 + s2):
        da[char] = 0
    
    # Fill matrix
    for i in range(1, len(s1) + 1):
        db = 0
        for j in range(1, len(s2) + 1):
            k = da.get(s2[j-1], 0)
            l = db
            
            if s1[i-1] == s2[j-1]:
                cost = 0.0
                db = j
            else:
                cost = get_confusion_cost(s1[i-1], s2[j-1], confusion_weights)
            
            dist[i + 1][j + 1] = min(
                dist[i + 1][j] + 1.0,           # Insertion
                dist[i][j + 1] + 1.0,           # Deletion
                dist[i][j] + cost,              # Substitution
                dist[k][l] + (i - k - 1) + 1.0 + (j - l - 1)  # Transposition
            )
        
        da[s1[i-1]] = i
    
    return dist[len(s1) + 1][len(s2) + 1]


if __name__ == "__main__":
    # Test
    test_cases = [
        ("Ihilip", "Philip"),
        ("Mltad", "United"),
        ("Stataa", "States"),
        ("Jaffewere", "Jaffe"),
        ("Wwker", "Walker"),
    ]
    
    confusion_weights = load_confusion_weights()
    
    print("Testing weighted Levenshtein distance:")
    for s1, s2 in test_cases:
        dist = weighted_levenshtein(s1, s2, confusion_weights)
        norm_dist = normalized_weighted_levenshtein(s1, s2, confusion_weights)
        print(f"  '{s1}' vs '{s2}': distance={dist:.2f}, normalized={norm_dist:.2f}")
