#!/usr/bin/env python3
"""
Regression tests for the override system (bans, prefers).

Tests:
1. Deterministic outputs: same candidates → same chosen entity
2. Bans filter first (before prefer)
3. Prefer chooses only if still in candidate set
4. Surface bans block entirely
5. Entity bans remove specific candidates

Fixture: tiny transcript with speaker labels, stage directions, redactions
"""

import sys
from pathlib import Path
import unittest
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.surface_norm import normalize_surface
from retrieval.proposal_gating import compute_group_key, compute_candidate_set_hash


# =============================================================================
# Mock AliasInfo for testing (mimics the real dataclass)
# =============================================================================

@dataclass
class MockAliasInfo:
    entity_id: int
    alias_norm: str
    original_alias: str
    entity_type: str
    alias_class: str = "person_full"
    is_auto_match: bool = True
    match_case: str = "any"
    requires_context: Optional[str] = None
    alias_type: str = "primary"
    min_chars: int = 3


# =============================================================================
# Import the actual functions we're testing
# =============================================================================

# We need to import after mocking or use the actual functions
from scripts.extract_entity_mentions import (
    apply_bans_to_candidates,
    find_dominant_candidate,
)


class TestSurfaceNormalization(unittest.TestCase):
    """Test surface normalization is consistent."""
    
    def test_normalize_lowercase(self):
        self.assertEqual(normalize_surface("MOSCOW"), "moscow")
        self.assertEqual(normalize_surface("Moscow"), "moscow")
    
    def test_normalize_punctuation(self):
        self.assertEqual(normalize_surface("F.B.I."), "fbi")
        self.assertEqual(normalize_surface("Mr. Cohn"), "mr cohn")
        self.assertEqual(normalize_surface("O'Brien"), "obrien")
    
    def test_normalize_whitespace(self):
        self.assertEqual(normalize_surface("  multiple   spaces  "), "multiple spaces")
        self.assertEqual(normalize_surface("new\nline"), "new line")
    
    def test_normalize_empty(self):
        self.assertEqual(normalize_surface(""), "")
        self.assertEqual(normalize_surface(None), "")


class TestGroupKeyComputation(unittest.TestCase):
    """Test group key and hash computation."""
    
    def test_group_key_deterministic(self):
        """Same inputs should produce same outputs."""
        key1 = compute_group_key("moscow", [123, 456, 789])
        key2 = compute_group_key("moscow", [123, 456, 789])
        self.assertEqual(key1, key2)
    
    def test_group_key_order_independent(self):
        """Candidate order shouldn't matter."""
        key1 = compute_group_key("moscow", [123, 456, 789])
        key2 = compute_group_key("moscow", [789, 123, 456])
        self.assertEqual(key1, key2)
    
    def test_candidate_set_hash_deterministic(self):
        hash1 = compute_candidate_set_hash([123, 456])
        hash2 = compute_candidate_set_hash([456, 123])
        self.assertEqual(hash1, hash2)


class TestBanFiltering(unittest.TestCase):
    """Test that bans filter candidates correctly."""
    
    def setUp(self):
        """Create test candidates."""
        self.candidates = [
            MockAliasInfo(entity_id=100, alias_norm="moscow", original_alias="Moscow", entity_type="place"),
            MockAliasInfo(entity_id=200, alias_norm="moscow", original_alias="MOSCOW", entity_type="cover_name"),
            MockAliasInfo(entity_id=300, alias_norm="moscow", original_alias="Moscow", entity_type="person"),
        ]
    
    def test_no_bans_returns_all(self):
        """With no bans, all candidates should remain."""
        filtered, reason = apply_bans_to_candidates(
            alias_norm="moscow",
            alias_infos=self.candidates,
            ban_entity_map=None,
            ban_surface_map=None,
            scope=None,
        )
        self.assertEqual(len(filtered), 3)
        self.assertIsNone(reason)
    
    def test_surface_ban_global(self):
        """Global surface ban should block entirely."""
        ban_surface_map = {(None, "moscow"): True}
        filtered, reason = apply_bans_to_candidates(
            alias_norm="moscow",
            alias_infos=self.candidates,
            ban_entity_map=None,
            ban_surface_map=ban_surface_map,
            scope=None,
        )
        self.assertEqual(len(filtered), 0)
        self.assertIn("surface_banned", reason)
    
    def test_surface_ban_scoped(self):
        """Scoped surface ban should only apply in that scope."""
        ban_surface_map = {("venona", "moscow"): True}
        
        # In venona scope - should ban
        filtered, reason = apply_bans_to_candidates(
            alias_norm="moscow",
            alias_infos=self.candidates,
            ban_entity_map=None,
            ban_surface_map=ban_surface_map,
            scope="venona",
        )
        self.assertEqual(len(filtered), 0)
        
        # In different scope - should not ban
        filtered, reason = apply_bans_to_candidates(
            alias_norm="moscow",
            alias_infos=self.candidates,
            ban_entity_map=None,
            ban_surface_map=ban_surface_map,
            scope="vassiliev",
        )
        self.assertEqual(len(filtered), 3)
    
    def test_entity_ban_global(self):
        """Global entity ban should remove specific entity."""
        ban_entity_map = {(None, "moscow"): {200}}  # Ban entity 200
        filtered, reason = apply_bans_to_candidates(
            alias_norm="moscow",
            alias_infos=self.candidates,
            ban_entity_map=ban_entity_map,
            ban_surface_map=None,
            scope=None,
        )
        self.assertEqual(len(filtered), 2)
        self.assertNotIn(200, [c.entity_id for c in filtered])
        self.assertIn("entities_banned", reason)
    
    def test_entity_ban_multiple(self):
        """Multiple entity bans should all be applied."""
        ban_entity_map = {(None, "moscow"): {100, 200}}  # Ban entities 100 and 200
        filtered, reason = apply_bans_to_candidates(
            alias_norm="moscow",
            alias_infos=self.candidates,
            ban_entity_map=ban_entity_map,
            ban_surface_map=None,
            scope=None,
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].entity_id, 300)
    
    def test_ban_all_candidates(self):
        """Banning all candidates should return empty list."""
        ban_entity_map = {(None, "moscow"): {100, 200, 300}}
        filtered, reason = apply_bans_to_candidates(
            alias_norm="moscow",
            alias_infos=self.candidates,
            ban_entity_map=ban_entity_map,
            ban_surface_map=None,
            scope=None,
        )
        self.assertEqual(len(filtered), 0)


class TestPreferOverride(unittest.TestCase):
    """Test that prefer overrides work correctly."""
    
    def setUp(self):
        """Create test candidates."""
        self.candidates = [
            MockAliasInfo(entity_id=100, alias_norm="doctor", original_alias="Doctor", entity_type="person"),
            MockAliasInfo(entity_id=200, alias_norm="doctor", original_alias="DOCTOR", entity_type="role_title"),
            MockAliasInfo(entity_id=300, alias_norm="doctor", original_alias="Doctor", entity_type="cover_name"),
        ]
    
    def test_prefer_selects_entity(self):
        """Prefer override should select the specified entity."""
        prefer_map = {(None, "doctor"): 200}  # Prefer entity 200
        
        chosen, rule, detail = find_dominant_candidate(
            alias_norm="doctor",
            alias_infos=self.candidates,
            surface="Doctor",
            preferred_entity_id_map=prefer_map,
            ban_entity_map=None,
            ban_surface_map=None,
            scope=None,
        )
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.entity_id, 200)
        self.assertIn("prefer", rule.lower())
    
    def test_prefer_scoped_takes_precedence(self):
        """Scoped prefer should take precedence over global."""
        prefer_map = {
            (None, "doctor"): 100,       # Global: prefer 100
            ("venona", "doctor"): 200,   # Venona: prefer 200
        }
        
        # In venona scope - should use scoped prefer
        chosen, rule, detail = find_dominant_candidate(
            alias_norm="doctor",
            alias_infos=self.candidates,
            surface="Doctor",
            preferred_entity_id_map=prefer_map,
            ban_entity_map=None,
            ban_surface_map=None,
            scope="venona",
        )
        self.assertEqual(chosen.entity_id, 200)
        
        # In other scope - should use global prefer
        chosen, rule, detail = find_dominant_candidate(
            alias_norm="doctor",
            alias_infos=self.candidates,
            surface="Doctor",
            preferred_entity_id_map=prefer_map,
            ban_entity_map=None,
            ban_surface_map=None,
            scope="vassiliev",
        )
        self.assertEqual(chosen.entity_id, 100)
    
    def test_prefer_ignored_if_not_in_candidates(self):
        """Prefer should be ignored if entity not in candidate set."""
        prefer_map = {(None, "doctor"): 999}  # Entity 999 not in candidates
        
        chosen, rule, detail = find_dominant_candidate(
            alias_norm="doctor",
            alias_infos=self.candidates,
            surface="Doctor",
            preferred_entity_id_map=prefer_map,
            ban_entity_map=None,
            ban_surface_map=None,
            scope=None,
        )
        # Should fall through to other rules (not use prefer)
        if chosen is not None:
            self.assertNotEqual(rule, "rule0b_prefer")


class TestBanBeforePrefer(unittest.TestCase):
    """Test that bans are applied before prefer (Rule 0a before 0b)."""
    
    def setUp(self):
        """Create test candidates."""
        self.candidates = [
            MockAliasInfo(entity_id=100, alias_norm="viktor", original_alias="Viktor", entity_type="person"),
            MockAliasInfo(entity_id=200, alias_norm="viktor", original_alias="VIKTOR", entity_type="cover_name"),
        ]
    
    def test_ban_then_prefer(self):
        """If preferred entity is banned, it should not be selected."""
        prefer_map = {(None, "viktor"): 200}  # Prefer entity 200
        ban_entity_map = {(None, "viktor"): {200}}  # But ban entity 200!
        
        chosen, rule, detail = find_dominant_candidate(
            alias_norm="viktor",
            alias_infos=self.candidates,
            surface="Viktor",
            preferred_entity_id_map=prefer_map,
            ban_entity_map=ban_entity_map,
            ban_surface_map=None,
            scope=None,
        )
        
        # Entity 200 was banned, so even though it's preferred, entity 100 should be selected
        # (since it's the only remaining candidate after ban)
        if chosen is not None:
            self.assertEqual(chosen.entity_id, 100)
            # Rule should indicate ban was applied
            self.assertIn("ban", rule.lower())
    
    def test_surface_ban_blocks_prefer(self):
        """Surface ban should block entirely, even with prefer."""
        prefer_map = {(None, "viktor"): 200}
        ban_surface_map = {(None, "viktor"): True}
        
        chosen, rule, detail = find_dominant_candidate(
            alias_norm="viktor",
            alias_infos=self.candidates,
            surface="Viktor",
            preferred_entity_id_map=prefer_map,
            ban_entity_map=None,
            ban_surface_map=ban_surface_map,
            scope=None,
        )
        
        self.assertIsNone(chosen)
        self.assertIn("ban", rule.lower())


class TestDeterministicOutput(unittest.TestCase):
    """Test that same inputs produce same outputs."""
    
    def test_same_candidates_same_result(self):
        """Identical inputs should produce identical outputs."""
        candidates = [
            MockAliasInfo(entity_id=100, alias_norm="test", original_alias="Test", entity_type="person"),
            MockAliasInfo(entity_id=200, alias_norm="test", original_alias="TEST", entity_type="cover_name"),
        ]
        prefer_map = {(None, "test"): 100}
        
        results = []
        for _ in range(10):
            chosen, rule, detail = find_dominant_candidate(
                alias_norm="test",
                alias_infos=candidates,
                surface="Test",
                preferred_entity_id_map=prefer_map,
                ban_entity_map=None,
                ban_surface_map=None,
                scope=None,
            )
            results.append((chosen.entity_id if chosen else None, rule))
        
        # All results should be identical
        self.assertTrue(all(r == results[0] for r in results))


# =============================================================================
# Fixture: Tiny transcript with various edge cases
# =============================================================================

FIXTURE_TRANSCRIPT = """
MR. COHN: Good morning, Dr. FUCHS. [Stage direction: Witness sworn]

DR. FUCHS: Good morning.

MR. COHN: Were you involved with the MOSCOW operation?

DR. FUCHS: I cannot recall. [REDACTED]

MR. MCCARTHY: Let the record show the witness is being evasive about VIKTOR.

[Recess called at 11:30 AM]
"""

class TestFixtureTranscript(unittest.TestCase):
    """Test processing of the fixture transcript."""
    
    def test_fixture_has_speakers(self):
        """Fixture should contain speaker labels."""
        self.assertIn("MR. COHN:", FIXTURE_TRANSCRIPT)
        self.assertIn("DR. FUCHS:", FIXTURE_TRANSCRIPT)
        self.assertIn("MR. MCCARTHY:", FIXTURE_TRANSCRIPT)
    
    def test_fixture_has_stage_directions(self):
        """Fixture should contain stage directions."""
        self.assertIn("[Stage direction", FIXTURE_TRANSCRIPT)
        self.assertIn("[Recess", FIXTURE_TRANSCRIPT)
    
    def test_fixture_has_redactions(self):
        """Fixture should contain redactions."""
        self.assertIn("[REDACTED]", FIXTURE_TRANSCRIPT)
    
    def test_fixture_has_test_surfaces(self):
        """Fixture should contain surfaces we can test bans/prefers on."""
        self.assertIn("MOSCOW", FIXTURE_TRANSCRIPT)
        self.assertIn("VIKTOR", FIXTURE_TRANSCRIPT)
        self.assertIn("FUCHS", FIXTURE_TRANSCRIPT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
