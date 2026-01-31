#!/usr/bin/env python3
"""Simple test of normalization function."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concordance.validate_entity_mentions_from_citations import normalize_document_name

# Test cases
tests = [
    ("Venona New York KGB 1943", "Venona_New_York_KGB_1943.pdf"),
    ("Venona New York KGB 1941–42", "Venona_New_York_KGB_1941-42.pdf"),
    ("Venona San Francisco KGB", "Venona_San_Francisco_KGB.pdf"),
]

print("Normalization Test:")
print("=" * 60)
for citation, doc_name in tests:
    citation_norm = normalize_document_name(citation)
    doc_norm = normalize_document_name(doc_name)
    match = citation_norm == doc_norm
    print(f"Citation: {citation}")
    print(f"  -> {citation_norm}")
    print(f"Document: {doc_name}")
    print(f"  -> {doc_norm}")
    print(f"Match: {match}")
    print()
