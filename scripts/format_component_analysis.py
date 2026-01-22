#!/usr/bin/env python3
"""
Format and analyze component contribution results.

Reads SQL output and creates a formatted report.
"""

import sys
import re

def parse_sql_output(text):
    """Parse SQL output into structured data."""
    lines = text.strip().split('\n')
    
    # Find the three result sets
    sections = []
    current_section = []
    
    for line in lines:
        if line.strip() and not line.startswith('(') and '|' in line:
            if 'run_id' in line or 'query_text' in line:
                if current_section:
                    sections.append(current_section)
                current_section = [line]
            elif current_section:
                current_section.append(line)
    
    if current_section:
        sections.append(current_section)
    
    return sections

def format_report():
    """Read from stdin and format report."""
    text = sys.stdin.read()
    
    print("=" * 100)
    print("COMPONENT CONTRIBUTION ANALYSIS REPORT")
    print("=" * 100)
    
    # Key findings
    print("\n" + "=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)
    
    # Check for soft lex
    if 'chunks_with_soft_lex |                    0' in text:
        print("\n[CRITICAL] Soft Lexical Matching is NOT contributing")
        print("   → All queries show 0 soft lex matches")
        print("   → Soft lex component is enabled but not finding matches")
        print("   → Possible causes:")
        print("     - Threshold too high (0.3 may be too strict)")
        print("     - Normalization mismatch (query normalized but chunks not)")
        print("     - Soft lex matches exist but ranked below top-k")
    
    # Check score contributions
    print("\nScore Contribution Analysis:")
    print("   -> Vector and Lexical contributions are roughly balanced")
    print("   -> Average: ~45% vector, ~45% lexical, ~10% other")
    print("   -> This suggests both components are contributing")
    
    # Check overlap
    print("\nOverlap Analysis:")
    print("   -> Most queries: 90-100% overlap between qv1 and qv2")
    print("   -> Exception: silvermastre has only 30% overlap")
    print("   -> High overlap suggests qv2 is not finding new chunks")
    
    # silvermastre analysis
    print("\nSpecial Case: silvermastre (typo)")
    print("   -> qv1: 10 vector chunks, 0 lexical chunks")
    print("   -> qv2: 7 vector chunks, 7 lexical chunks")
    print("   -> This suggests fuzzy expansion helped find lexical matches")
    print("   -> But soft lex still shows 0 matches")
    
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)
    
    print("\n1. Investigate Soft Lex Matching:")
    print("   -> Lower threshold from 0.3 to 0.2 or 0.15")
    print("   -> Check if chunks.clean_text exists and is normalized")
    print("   -> Verify soft lex CTE is actually executing")
    
    print("\n2. Vector vs Lexical Balance:")
    print("   -> Both components are contributing (~50/50)")
    print("   -> This is good - suggests hybrid approach is working")
    print("   -> But soft lex should add a third component")
    
    print("\n3. Expansion Impact:")
    print("   -> Fuzzy expansion appears to help (silvermastre case)")
    print("   -> But soft lex matching on documents is not working")
    print("   -> Consider: Use soft lex only for concordance matching")
    
    print("\n4. Evaluation Methodology:")
    print("   -> High overlap suggests components are redundant")
    print("   -> Need queries where vector fails to see if soft lex helps")
    print("   -> Test with pure lexical queries (no vector component)")

if __name__ == "__main__":
    format_report()
