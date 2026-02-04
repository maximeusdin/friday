"""
Variant Generator - Synonym/variant generation with guardrails.

Generates alternative forms of terms for retrieval expansion.
Uses concordance first (known aliases), then LLM for unknown terms.
All variants are validated by code-enforced guardrails.
"""

import os
import json
import sys
from typing import List, Set


# Stopwords that should not dominate variant terms
STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
    'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'under', 'again', 'further', 'then', 'once',
    'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
    'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'also',
    'there', 'their', 'they', 'them', 'this', 'that', 'these', 'those',
}


def is_stopword_heavy(term: str, threshold: float = 0.5) -> bool:
    """Check if a term is dominated by stopwords."""
    tokens = term.lower().split()
    if not tokens:
        return True
    
    stopword_count = sum(1 for t in tokens if t in STOPWORDS)
    return stopword_count / len(tokens) > threshold


def validate_variant(variant: str, max_tokens: int = 4) -> bool:
    """
    Validate a variant term against guardrails.
    
    Rules:
    - 1-4 tokens
    - Not stopword-heavy
    - Not empty
    - Reasonable length
    """
    if not variant or not variant.strip():
        return False
    
    variant = variant.strip()
    tokens = variant.split()
    
    # Token count check
    if len(tokens) > max_tokens or len(tokens) == 0:
        return False
    
    # Stopword check
    if is_stopword_heavy(variant):
        return False
    
    # Length check (not too short, not too long)
    if len(variant) < 2 or len(variant) > 100:
        return False
    
    return True


def get_concordance_variants(concept: str, conn) -> List[str]:
    """
    Get known variants from concordance.
    
    This is the preferred source - deterministic and vetted.
    """
    from retrieval.ops import concordance_expand_terms
    
    try:
        variants = concordance_expand_terms(
            conn=conn,
            text=concept,
            max_aliases_out=15,
        )
        print(f"      Concordance variants for '{concept}': {len(variants)}", file=sys.stderr)
        return variants
    except Exception as e:
        print(f"      Concordance lookup failed: {e}", file=sys.stderr)
        return []


def get_llm_variants(concept: str, existing_variants: List[str] = None) -> List[str]:
    """
    Generate variants using LLM (only if concordance is insufficient).
    
    LLM variants are for retrieval only - never for claims.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []
    
    existing_str = ", ".join(existing_variants) if existing_variants else "none"
    
    prompt = f"""Generate alternative forms of this term for historical document search.

TERM: {concept}

EXISTING VARIANTS (don't repeat): {existing_str}

GENERATE VARIANTS FOR:
1. OCR errors (fuze/fuse, mn/man, etc.)
2. Abbreviations (VT for variable time)
3. Historical terminology (period-specific names)
4. Spelling variations (British/American)
5. Common misspellings

RULES:
- Each variant must be 1-4 words
- No generic words (evidence, documents, information)
- Focus on the distinctive technical/proper noun parts
- Maximum 8 variants

OUTPUT: JSON array of strings, e.g. ["variant 1", "variant 2"]"""

    try:
        from openai import OpenAI
        
        model = os.getenv("OPENAI_MODEL_PLAN", "gpt-4o-mini")
        client = OpenAI(api_key=api_key)
        
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Generate term variants for document retrieval. Output JSON array only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        
        content = resp.choices[0].message.content
        if content:
            result = json.loads(content)
            # Handle both array and object with "variants" key
            if isinstance(result, list):
                return result
            elif isinstance(result, dict) and "variants" in result:
                return result["variants"]
            elif isinstance(result, dict):
                # Try to find array in any key
                for v in result.values():
                    if isinstance(v, list):
                        return v
        return []
        
    except Exception as e:
        print(f"      LLM variant generation failed: {e}", file=sys.stderr)
        return []


def generate_variants(
    concept: str,
    conn,
    max_variants: int = 10,
    min_concordance: int = 3,
) -> List[str]:
    """
    Generate synonym/variant terms with strict guardrails.
    
    Priority:
    1. Concordance (known aliases - preferred)
    2. LLM (if concordance insufficient)
    3. Guardrail validation (all variants)
    
    Args:
        concept: The term to expand
        conn: Database connection
        max_variants: Maximum variants to return
        min_concordance: Minimum concordance results before trying LLM
    
    Returns:
        List of validated variant terms
    """
    print(f"    [Variants] Generating for: {concept}", file=sys.stderr)
    
    # 1. Try concordance first (preferred source)
    concordance_variants = get_concordance_variants(concept, conn)
    
    # 2. If insufficient, try LLM
    llm_variants = []
    if len(concordance_variants) < min_concordance:
        llm_variants = get_llm_variants(concept, concordance_variants)
        print(f"      LLM variants: {len(llm_variants)}", file=sys.stderr)
    
    # 3. Combine and dedupe
    all_variants = concordance_variants + llm_variants
    
    # 4. Apply guardrails (code-enforced, non-negotiable)
    validated: List[str] = []
    seen: Set[str] = {concept.lower()}  # Don't include original
    
    for variant in all_variants:
        if not variant:
            continue
            
        normalized = variant.strip().lower()
        
        # Skip duplicates
        if normalized in seen:
            continue
        seen.add(normalized)
        
        # Validate against guardrails
        if validate_variant(variant):
            validated.append(variant.strip())
        else:
            print(f"      Rejected variant (guardrail): {variant}", file=sys.stderr)
    
    # 5. Cap at max_variants
    result = validated[:max_variants]
    print(f"      Final variants: {result}", file=sys.stderr)
    
    return result


def expand_anchor_terms(
    anchor_terms: List[str],
    conn,
    max_per_term: int = 5,
) -> List[str]:
    """
    Expand a list of anchor terms with variants.
    
    Returns all unique variants (not including originals).
    """
    all_variants: List[str] = []
    seen: Set[str] = set()
    
    for term in anchor_terms:
        seen.add(term.lower())
    
    for term in anchor_terms:
        variants = generate_variants(term, conn, max_variants=max_per_term)
        for v in variants:
            norm = v.lower()
            if norm not in seen:
                seen.add(norm)
                all_variants.append(v)
    
    return all_variants
