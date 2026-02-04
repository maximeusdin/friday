#!/usr/bin/env python3
"""
fix_entity_types.py - Entity Type Classification Correction

Identifies and fixes misclassified entities from concordance migration.
Uses heuristics first to auto-fix high-confidence cases, then exports
only ambiguous cases for manual review.

Usage:
    # Step 1: See what we're dealing with
    python scripts/fix_entity_types.py --audit

    # Step 2: Auto-fix high-confidence cases (preview first)
    python scripts/fix_entity_types.py --auto-fix --dry-run
    python scripts/fix_entity_types.py --auto-fix

    # Step 3: Export remaining ambiguous cases for manual review
    python scripts/fix_entity_types.py --export ambiguous.csv

    # Step 4: After manual review, apply corrections
    python scripts/fix_entity_types.py --apply ambiguous_reviewed.csv

    # Final verification
    python scripts/fix_entity_types.py --verify
"""

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2


# =============================================================================
# High-Confidence Patterns (Auto-Fix Without Review)
# =============================================================================

# Organizations: These patterns are unambiguous
HIGH_CONFIDENCE_ORG_PATTERNS = [
    # Known acronym agencies (exact match on canonical name)
    r'^(FBI|CIA|KGB|NKVD|GRU|OSS|NSA|MGB|MVD|GPU|OGPU|CPUSA|ONI|OWI|SIS|MI5|MI6|Cheka|GUGB|INO|OMS|Comintern|Politburo)$',
    
    # US Government departments/agencies
    r'\b(Department|Bureau|Agency|Administration|Commission|Committee|Office|Service)\b',
    r'\b(State Department|Treasury Department|War Department|Justice Department)\b',
    r'\b(Agricultural Adjustment|Army Air|Lend-Lease|Office of)\b',
    
    # Military organizations
    r'\b(Army|Navy|Air Force|Marines|Corps|Division|Regiment|Battalion|Squadron|Fleet|Command|Forces)\b',
    r'\b(Proving Grounds?|Arsenal|Laboratory|Test Site)\b',
    
    # Soviet organizations
    r'\b(Soviet|Politburo|Central Committee|Rezidentura|TASS|Amtorg|Intourist)\b',
    r"\b(People's Commissariat|NKGB|NKVD|MGB|MVD|GRU)\b",
    
    # Communist party / political orgs
    r'\b(Communist Party|Party USA|CPUSA|Comintern|Young Communist|Workers Party)\b',
    r'\b(League|Federation|Union|Association|Society|Congress)\b',
    
    # Corporate/institutional
    r'\b(Corporation|Company|Inc\.|Corp\.|Ltd\.|Industries|Works|Bank|Press|Publishing)\b',
    r'\b(Institute|Foundation|University|College|School|Academy)\b',
    
    # Espionage networks/groups (named)
    r'\b(Group|Network|Ring|Apparatus|Cell|Station|Rezidentura)\b',
    r'(Silvermaster|Perlo|Ware|Golos|Bentley|Rosenberg)\s+(Group|Network|Ring|apparatus)',
    
    # Diplomatic
    r'\b(Embassy|Consulate|Mission|Legation)\b',
    
    # Generic org indicators
    r'^The\s+[A-Z]',  # "The Something" - usually org
]

# Places: High-confidence patterns
HIGH_CONFIDENCE_PLACE_PATTERNS = [
    r'\b(Proving Grounds?|Test Site|Laboratory|Facility)\b.*\b(at|in|near)\b',
    r'^(Los Alamos|Oak Ridge|Hanford|Aberdeen|White Sands)',
    r'\b(Field Office|Station)\s+(at|in)\s+',
]

# Person exclusions: These override org patterns - definitely people
PERSON_OVERRIDE_PATTERNS = [
    r',\s*(Jr\.?|Sr\.?|II|III|IV|V)$',           # "Smith, Jr."
    r'^(Mr|Mrs|Ms|Miss|Dr|Prof|Gen|Col|Maj|Lt|Capt|Cdr|Adm|Sen|Rep|Hon|Rev)\.\s',
    r'\b(wife|husband|son|daughter|brother|sister|father|mother)\s+of\b',
    r"'s\s+(wife|husband|son|daughter)",
    # "A. B. Smith" initials pattern
    r'^[A-Z]\.\s*[A-Z]?\.\s*[A-Z][a-z]+$',
    # Russian patronymic names like "Viktor Semenovich Abakumov"
    r'^[A-Z][a-z]+\s+[A-Z][a-z]+ovich\s+[A-Z][a-z]+$',
    r'^[A-Z][a-z]+\s+[A-Z][a-z]+ovna\s+[A-Z][a-z]+$',
]

# Words that indicate something is NOT a person name (used to exclude false positives)
NOT_PERSON_KEYWORDS = [
    # Military/government
    'navy', 'army', 'air force', 'marines', 'corps', 'division', 'regiment',
    'battalion', 'squadron', 'command', 'forces', 'fleet', 'intelligence',
    'department', 'bureau', 'agency', 'commission', 'committee', 'office',
    # Military facilities
    'fort', 'camp', 'base', 'barracks', 'arsenal', 'proving', 'airfield',
    # Media/publications
    'press', 'magazine', 'times', 'post', 'herald', 'tribune', 'journal',
    'news', 'radio', 'television', 'broadcast', 'publishing', 'publications',
    'daily', 'weekly', 'monthly', 'review', 'gazette', 'standard', 'eagle',
    'sun', 'star', 'today', 'affairs', 'digest', 'chronicle', 'bulletin',
    # Corporate/business
    'corporation', 'company', 'industries', 'aircraft', 'motors', 'electric',
    'export', 'import', 'trading', 'bank', 'trust', 'financial', 'insurance',
    'steel', 'chemical', 'rubber', 'lumber', 'mining', 'manufacturing',
    # Academic/institutional
    'university', 'college', 'institute', 'foundation', 'academy', 'school',
    'library', 'museum', 'center', 'society', 'association', 'union',
    # Political
    'party', 'socialist', 'communist', 'conference', 'congress', 'council',
    # Generic org indicators
    'group', 'network', 'ring', 'organization', 'service', 'administration',
    # Legal/government
    'court', 'tribunal', 'board', 'authority', 'ministry', 'consulate', 'embassy',
    # Foreign language publication indicators
    'soir', 'matin', 'monde', 'znamya', 'pravda', 'izvestia', 'trud',
    # Company name indicators
    'general', 'national', 'international', 'american', 'united', 'federal',
    'research', 'development', 'systems', 'technologies', 'products',
    # Place indicators
    'zealand', 'republic', 'kingdom', 'empire', 'islands', 'territories',
    # Misc org indicators
    'world', 'league', 'front', 'movement', 'alliance', 'coalition',
    # Place/infrastructure indicators
    'canal', 'rico', 'puerto', 'strait', 'channel', 'river', 'mountain',
    # Job titles / espionage terms
    'chief', 'officer', 'agent', 'director', 'secretary', 'chairman',
    'house', 'station', 'worker', 'operative',
    # Foreign language publication/place words
    'mir', 'polska', 'nuovo', 'neue', 'novoe', 'novy', 'nowa',
    # Movement/publication suffixes
    'guard', 'commentator', 'observer', 'correspondent', 'reporter',
]


# =============================================================================
# Weak Signals (Need Manual Review)
# =============================================================================

WEAK_ORG_SIGNALS = [
    r'^The\s+',                      # "The ..." could be org or codename
    r'\b(Center|Service|Board|Council)\b',  # Sometimes in names
    r'^[A-Z]{2,5}$',                 # Short all-caps - might be acronym
    r'\b(Section|Unit|Branch|Desk)\b',
]


def get_conn():
    """Get database connection from environment."""
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL")
    return psycopg2.connect(dsn)


@dataclass
class EntityInfo:
    """Entity information for classification."""
    id: int
    canonical_name: str
    current_type: str
    aliases: List[str]
    mention_count: int = 0


def matches_any_pattern(text: str, patterns: List[str]) -> Tuple[bool, Optional[str]]:
    """Check if text matches any pattern. Returns (matched, pattern_that_matched)."""
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True, pattern
    return False, None


def classify_entity(entity: EntityInfo) -> Tuple[str, str, float]:
    """
    Classify entity type based on canonical name and aliases.
    
    Returns: (suggested_type, reason, confidence)
    - confidence: 1.0 = auto-fix, <1.0 = needs review
    """
    name = entity.canonical_name
    all_text = name + " " + " ".join(entity.aliases)
    
    # Don't reclassify cover_names as person - they should stay cover_name or become org
    # Cover names are typically ALL CAPS or have special formatting
    if entity.current_type == "cover_name":
        # Only allow cover_name -> org if CANONICAL NAME matches org patterns
        is_org, pattern = matches_any_pattern(name, HIGH_CONFIDENCE_ORG_PATTERNS)
        if is_org:
            return "org", f"Org pattern: {pattern}", 1.0
        # Keep as cover_name - don't use aliases (they often have descriptions)
        return "cover_name", "Preserving cover_name type", 0.0
    
    # Check person overrides first (highest priority)
    is_person, pattern = matches_any_pattern(name, PERSON_OVERRIDE_PATTERNS)
    if is_person:
        return "person", f"Person pattern: {pattern}", 1.0
    
    # Special case: "First Last" pattern (two Title Case words)
    # Only applies if name doesn't contain org-like keywords
    name_lower = name.lower()
    if re.match(r'^[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}$', name):
        # Check for org-like keywords that disqualify as person name
        has_org_keyword = any(kw in name_lower for kw in NOT_PERSON_KEYWORDS)
        if not has_org_keyword:
            return "person", "Person pattern: First Last (no org keywords)", 1.0
    
    # Check high-confidence org patterns
    is_org, pattern = matches_any_pattern(name, HIGH_CONFIDENCE_ORG_PATTERNS)
    if is_org:
        return "org", f"Org pattern: {pattern}", 1.0
    
    # Check aliases for org indicators - but skip descriptive aliases
    for alias in entity.aliases:
        # Skip aliases that look like descriptions (contain "unidentified", "soviet" alone, etc.)
        alias_lower = alias.lower()
        if any(skip in alias_lower for skip in ["unidentified", "unknown", "see ", "possibly", "probably"]):
            continue
        # Skip very long aliases (likely descriptions)
        if len(alias) > 60:
            continue
        is_org, pattern = matches_any_pattern(alias, HIGH_CONFIDENCE_ORG_PATTERNS)
        if is_org:
            return "org", f"Alias org pattern ({alias}): {pattern}", 0.9
    
    # Check high-confidence place patterns
    is_place, pattern = matches_any_pattern(name, HIGH_CONFIDENCE_PLACE_PATTERNS)
    if is_place:
        return "place", f"Place pattern: {pattern}", 1.0
    
    # Check weak signals (need review)
    has_weak_signal, pattern = matches_any_pattern(name, WEAK_ORG_SIGNALS)
    if has_weak_signal and entity.current_type == "person":
        return "org", f"Weak signal: {pattern}", 0.5
    
    # No patterns matched, keep current type
    return entity.current_type, "No patterns matched", 0.0


def fetch_entities_with_aliases(conn) -> List[EntityInfo]:
    """Fetch all entities with their aliases."""
    with conn.cursor() as cur:
        # Get entities
        cur.execute("""
            SELECT e.id, e.canonical_name, e.entity_type,
                   COALESCE((SELECT COUNT(*) FROM entity_mentions em WHERE em.entity_id = e.id), 0) as mention_count
            FROM entities e
            ORDER BY e.id
        """)
        entities_raw = cur.fetchall()
        
        # Get all aliases
        cur.execute("""
            SELECT entity_id, alias
            FROM entity_aliases
            ORDER BY entity_id
        """)
        aliases_raw = cur.fetchall()
    
    # Group aliases by entity
    aliases_by_entity: Dict[int, List[str]] = {}
    for entity_id, alias in aliases_raw:
        if entity_id not in aliases_by_entity:
            aliases_by_entity[entity_id] = []
        aliases_by_entity[entity_id].append(alias)
    
    # Build EntityInfo objects
    entities = []
    for entity_id, canonical_name, entity_type, mention_count in entities_raw:
        entities.append(EntityInfo(
            id=entity_id,
            canonical_name=canonical_name,
            current_type=entity_type or "person",
            aliases=aliases_by_entity.get(entity_id, []),
            mention_count=mention_count,
        ))
    
    return entities


def cmd_audit(conn):
    """Audit current entity type distribution and potential issues."""
    print("=" * 60)
    print("Entity Type Classification Audit")
    print("=" * 60)
    
    with conn.cursor() as cur:
        # Overall distribution
        cur.execute("""
            SELECT entity_type, COUNT(*) as count
            FROM entities
            GROUP BY entity_type
            ORDER BY count DESC
        """)
        print("\nCurrent Distribution:")
        for entity_type, count in cur.fetchall():
            print(f"  {entity_type or 'NULL'}: {count}")
        
        # Find likely misclassified (persons that look like orgs)
        cur.execute("""
            SELECT COUNT(*)
            FROM entities
            WHERE entity_type = 'person'
            AND (
                canonical_name ~* '\\m(Committee|Bureau|Department|Agency|Ministry|Office|Division|Group|Party|Corps|Council|Board|Institute|Foundation|Company|Inc\\.|Corp\\.|Ltd\\.)\\M'
                OR canonical_name ~* '\\m(FBI|CIA|KGB|NKVD|GRU|OSS|NSA|MGB|MVD|GPU|OGPU|CPUSA)\\M'
                OR canonical_name ~* '^The\\s+'
                OR (canonical_name = UPPER(canonical_name) AND length(canonical_name) BETWEEN 2 AND 6)
            )
        """)
        likely_misclassified = cur.fetchone()[0]
        print(f"\nLikely misclassified (person -> org): {likely_misclassified}")
        
        # Sample of likely misclassified
        cur.execute("""
            SELECT id, canonical_name
            FROM entities
            WHERE entity_type = 'person'
            AND (
                canonical_name ~* '\\m(Committee|Bureau|Department|Agency|Ministry|Office|Division|Group|Party|Corps|Council|Board|Institute|Foundation|Company)\\M'
                OR canonical_name ~* '\\m(FBI|CIA|KGB|NKVD|GRU|OSS|NSA|MGB|MVD|GPU|OGPU|CPUSA)\\M'
                OR canonical_name ~* '^The\\s+'
            )
            ORDER BY canonical_name
            LIMIT 20
        """)
        rows = cur.fetchall()
        if rows:
            print("\nSample of likely misclassified:")
            for entity_id, name in rows:
                print(f"  [{entity_id}] {name}")
    
    print("\n" + "=" * 60)


def cmd_auto_fix(conn, dry_run: bool = False):
    """Apply high-confidence fixes automatically."""
    print("=" * 60)
    print(f"Auto-Fix High-Confidence Cases {'(DRY RUN)' if dry_run else ''}")
    print("=" * 60)
    
    entities = fetch_entities_with_aliases(conn)
    
    fixes = []
    for entity in entities:
        suggested_type, reason, confidence = classify_entity(entity)
        
        # Only auto-fix if confidence is high and type would change
        if confidence >= 0.9 and suggested_type != entity.current_type:
            fixes.append({
                "id": entity.id,
                "canonical_name": entity.canonical_name,
                "current_type": entity.current_type,
                "new_type": suggested_type,
                "reason": reason,
                "confidence": confidence,
            })
    
    print(f"\nFound {len(fixes)} high-confidence fixes")
    
    # Group by type change
    by_change = {}
    for fix in fixes:
        key = f"{fix['current_type']} -> {fix['new_type']}"
        if key not in by_change:
            by_change[key] = []
        by_change[key].append(fix)
    
    for change_type, change_fixes in by_change.items():
        print(f"\n{change_type}: {len(change_fixes)} entities")
        for fix in change_fixes[:10]:
            print(f"  [{fix['id']}] {fix['canonical_name']}")
            print(f"       Reason: {fix['reason']}")
        if len(change_fixes) > 10:
            print(f"  ... and {len(change_fixes) - 10} more")
    
    if not dry_run and fixes:
        print(f"\nApplying {len(fixes)} fixes...")
        with conn.cursor() as cur:
            for fix in fixes:
                cur.execute(
                    "UPDATE entities SET entity_type = %s WHERE id = %s",
                    (fix["new_type"], fix["id"])
                )
        conn.commit()
        print("Done!")
        
        # Log fixes to file
        log_file = REPO_ROOT / "entity_type_fixes.log"
        with open(log_file, "a") as f:
            f.write(f"\n# Auto-fix run at {datetime.now().isoformat()}\n")
            for fix in fixes:
                f.write(f"{fix['id']},{fix['current_type']},{fix['new_type']},{fix['reason']}\n")
        print(f"Logged to {log_file}")
    
    print("\n" + "=" * 60)


def cmd_export(conn, output_file: str):
    """Export ambiguous cases for manual review."""
    print("=" * 60)
    print(f"Exporting Ambiguous Cases to {output_file}")
    print("=" * 60)
    
    entities = fetch_entities_with_aliases(conn)
    
    ambiguous = []
    for entity in entities:
        suggested_type, reason, confidence = classify_entity(entity)
        
        # Export if:
        # 1. Has a weak signal (confidence 0.5-0.9) and type would change
        # 2. Current type is person but has some org-like qualities
        if 0.3 <= confidence < 0.9 and suggested_type != entity.current_type:
            ambiguous.append({
                "entity_id": entity.id,
                "canonical_name": entity.canonical_name,
                "current_type": entity.current_type,
                "suggested_type": suggested_type,
                "confidence": confidence,
                "reason": reason,
                "aliases": "; ".join(entity.aliases[:5]),  # First 5 aliases
                "mention_count": entity.mention_count,
                "final_type": "",  # For reviewer to fill in
            })
    
    print(f"Found {len(ambiguous)} ambiguous entities for review")
    
    if ambiguous:
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "entity_id", "canonical_name", "current_type", "suggested_type",
                "confidence", "reason", "aliases", "mention_count", "final_type"
            ])
            writer.writeheader()
            writer.writerows(ambiguous)
        
        print(f"Wrote {len(ambiguous)} entities to {output_file}")
        print("\nInstructions:")
        print("1. Open the CSV in a spreadsheet")
        print("2. Review each entity and fill in 'final_type' column")
        print("3. Use: 'person', 'org', 'place', or leave blank to skip")
        print(f"4. Run: python scripts/fix_entity_types.py --apply {output_file}")
    
    print("\n" + "=" * 60)


def cmd_apply(conn, input_file: str):
    """Apply corrections from reviewed CSV."""
    print("=" * 60)
    print(f"Applying Corrections from {input_file}")
    print("=" * 60)
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")
    
    corrections = []
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            final_type = row.get("final_type", "").strip().lower()
            if final_type in ("person", "org", "place"):
                corrections.append({
                    "entity_id": int(row["entity_id"]),
                    "canonical_name": row["canonical_name"],
                    "current_type": row["current_type"],
                    "new_type": final_type,
                })
    
    print(f"Found {len(corrections)} corrections to apply")
    
    if corrections:
        with conn.cursor() as cur:
            for fix in corrections:
                cur.execute(
                    "UPDATE entities SET entity_type = %s WHERE id = %s",
                    (fix["new_type"], fix["entity_id"])
                )
                print(f"  [{fix['entity_id']}] {fix['canonical_name']}: {fix['current_type']} -> {fix['new_type']}")
        conn.commit()
        print("\nDone!")
    
    print("\n" + "=" * 60)


def cmd_verify(conn):
    """Verify entity type distribution after fixes."""
    print("=" * 60)
    print("Post-Fix Verification")
    print("=" * 60)
    
    with conn.cursor() as cur:
        # Overall distribution
        cur.execute("""
            SELECT entity_type, COUNT(*) as count
            FROM entities
            GROUP BY entity_type
            ORDER BY count DESC
        """)
        print("\nFinal Distribution:")
        total = 0
        for entity_type, count in cur.fetchall():
            print(f"  {entity_type or 'NULL'}: {count}")
            total += count
        print(f"  Total: {total}")
        
        # Check for remaining likely misclassified
        cur.execute("""
            SELECT COUNT(*)
            FROM entities
            WHERE entity_type = 'person'
            AND (
                canonical_name ~* '\\m(Committee|Bureau|Department|Agency|Ministry|Office|Division|Group|Party|Corps|Council|Board|Institute|Foundation|Company)\\M'
                OR canonical_name ~* '\\m(FBI|CIA|KGB|NKVD|GRU|OSS|NSA|MGB|MVD|GPU|OGPU|CPUSA)\\M'
            )
        """)
        remaining = cur.fetchone()[0]
        
        misclass_rate = (remaining / total * 100) if total > 0 else 0
        print(f"\nRemaining likely misclassified: {remaining} ({misclass_rate:.1f}%)")
        
        if misclass_rate < 2:
            print("[OK] Misclassification rate below 2% target")
        elif misclass_rate < 5:
            print("[WARN] Misclassification rate below 5% threshold but above 2% target")
        else:
            print("[FAIL] Misclassification rate above 5% threshold - review needed")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Entity Type Classification Correction Tool"
    )
    parser.add_argument("--audit", action="store_true",
                        help="Audit current entity type distribution")
    parser.add_argument("--auto-fix", action="store_true",
                        help="Apply high-confidence fixes automatically")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without applying (use with --auto-fix)")
    parser.add_argument("--export", metavar="FILE",
                        help="Export ambiguous cases to CSV for manual review")
    parser.add_argument("--apply", metavar="FILE",
                        help="Apply corrections from reviewed CSV")
    parser.add_argument("--verify", action="store_true",
                        help="Verify entity type distribution after fixes")
    
    args = parser.parse_args()
    
    if not any([args.audit, args.auto_fix, args.export, args.apply, args.verify]):
        parser.print_help()
        return
    
    conn = get_conn()
    
    try:
        if args.audit:
            cmd_audit(conn)
        
        if args.auto_fix:
            cmd_auto_fix(conn, dry_run=args.dry_run)
        
        if args.export:
            cmd_export(conn, args.export)
        
        if args.apply:
            cmd_apply(conn, args.apply)
        
        if args.verify:
            cmd_verify(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
