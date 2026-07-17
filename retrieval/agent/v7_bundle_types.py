"""
V7 Pre-Bundling Types - Concordance-Aware Evidence Grouping

Data structures for the pre-bundling phase that occurs BEFORE the evidence
bottleneck. This phase:
1. Extracts named entity surfaces from chunks (persons, codenames, orgs)
2. Resolves codenames to canonical entity IDs via concordance
3. Groups related chunks into BundleCandidates
4. Passes bundles (not raw chunks) to the bottleneck for scoring

This prevents codenames like "Ruble, Raid, Mole" from being treated as members
and enables coherent evidence citation.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Literal
from enum import Enum


# =============================================================================
# Pre-Bundling Modes
# =============================================================================

class PreBundlingMode(str, Enum):
    """Pre-bundling operation modes."""
    
    OFF = "off"              # No bundling (current V6 behavior)
    PASSTHROUGH = "passthrough"  # Post-selection codename guard only (fast)
    MICRO = "micro"          # Bundle seed chunks with neighbors (medium)
    SEMANTIC = "semantic"    # Full concordance-aware bundling (comprehensive)


# =============================================================================
# Bundle Kind
# =============================================================================

class BundleKind(str, Enum):
    """Classification of bundle evidence type."""
    
    PERSON_EVIDENCE = "PERSON_EVIDENCE"      # Contains resolved person entities
    CODENAME_EVIDENCE = "CODENAME_EVIDENCE"  # Contains only unresolved codenames
    MIXED = "MIXED"                          # Both resolved and unresolved


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PreBundlingConfig:
    """
    Configuration for the pre-bundling phase.
    
    Controls how chunks are grouped into bundles before bottleneck scoring.
    """
    
    # Operation mode
    mode: PreBundlingMode = PreBundlingMode.MICRO
    
    # Candidate selection
    max_candidate_chunks: int = 80          # Cap chunks before bundling
    seed_chunks_from_entities: int = 20     # Entity-based seed chunks
    
    # Bundle limits
    max_bundles: int = 10                   # Maximum bundles to create
    max_chunks_per_bundle: int = 6          # Max chunks in a single bundle
    min_chunks_per_bundle: int = 2          # Min chunks for valid bundle
    
    # LLM settings
    surface_model: str = "gpt-4.1-mini-2025-04-14"      # Model for surface extraction
    bundle_model: str = "gpt-4.1-mini-2025-04-14"       # Model for semantic bundling
    surface_batch_size: int = 10            # Chunks per surface extraction call
    
    # Concordance resolution thresholds
    entity_conf_threshold: float = 0.6      # Min confidence for entity linking
    codename_link_strong: float = 0.75      # Safe to merge codename → person
    codename_link_weak: float = 0.55        # Keep as "possible", don't merge
    
    # Verbose logging
    verbose: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "max_candidate_chunks": self.max_candidate_chunks,
            "seed_chunks_from_entities": self.seed_chunks_from_entities,
            "max_bundles": self.max_bundles,
            "max_chunks_per_bundle": self.max_chunks_per_bundle,
            "min_chunks_per_bundle": self.min_chunks_per_bundle,
            "surface_model": self.surface_model,
            "bundle_model": self.bundle_model,
            "entity_conf_threshold": self.entity_conf_threshold,
            "codename_link_strong": self.codename_link_strong,
            "codename_link_weak": self.codename_link_weak,
        }


# =============================================================================
# Chunk Annotation
# =============================================================================

@dataclass
class CodenameLink:
    """A resolved or attempted codename-to-entity mapping."""
    
    codename: str               # The codename surface (e.g., "Pal")
    entity_id: Optional[int]    # Resolved entity ID (if found)
    entity_name: Optional[str]  # Resolved entity name (if found)
    confidence: float           # Resolution confidence (0-1)
    resolution_method: str      # How it was resolved ("concordance", "alias", "context")
    
    def is_strong(self, threshold: float = 0.75) -> bool:
        """Check if this is a high-confidence link."""
        return self.entity_id is not None and self.confidence >= threshold
    
    def is_weak(self, strong_threshold: float = 0.75, weak_threshold: float = 0.55) -> bool:
        """Check if this is a low-confidence (but possible) link."""
        return (self.entity_id is not None and 
                weak_threshold <= self.confidence < strong_threshold)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "codename": self.codename,
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "confidence": self.confidence,
            "resolution_method": self.resolution_method,
        }


@dataclass
class ChunkAnnotation:
    """
    Annotation for a single chunk with extracted surfaces and resolved entities.
    
    Created during the surface extraction and concordance resolution phases.
    Used as input for semantic bundling.
    """
    
    # Chunk identity
    chunk_id: int
    chunk_text: str
    source_label: str = ""
    page: str = ""
    doc_id: Optional[int] = None
    
    # Extracted surfaces (from LLM)
    person_surfaces: List[str] = field(default_factory=list)
    codename_surfaces: List[str] = field(default_factory=list)
    org_surfaces: List[str] = field(default_factory=list)
    
    # Resolved entities
    resolved_people: List[int] = field(default_factory=list)  # entity_ids
    codename_links: List[CodenameLink] = field(default_factory=list)
    unresolved_codenames: List[str] = field(default_factory=list)
    
    # Classification hints
    about_labels: List[str] = field(default_factory=list)  # ["roster", "membership", "communications"]
    is_roster_evidence: bool = False        # Does this chunk list members?
    self_contained: bool = True             # Are pronouns resolved?
    has_aka_mapping: bool = False           # Does it define a codename mapping?
    
    # Retrieval metadata
    retrieval_score: float = 0.0
    
    def get_all_entity_ids(self) -> List[int]:
        """Get all resolved entity IDs (people + strong codename links)."""
        ids = list(self.resolved_people)
        for link in self.codename_links:
            if link.is_strong() and link.entity_id:
                ids.append(link.entity_id)
        return list(set(ids))
    
    def get_strong_codename_links(self) -> List[CodenameLink]:
        """Get codename links with high confidence."""
        return [l for l in self.codename_links if l.is_strong()]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_text": self.chunk_text[:200] + "..." if len(self.chunk_text) > 200 else self.chunk_text,
            "source_label": self.source_label,
            "page": self.page,
            "person_surfaces": self.person_surfaces,
            "codename_surfaces": self.codename_surfaces,
            "org_surfaces": self.org_surfaces,
            "resolved_people": self.resolved_people,
            "codename_links": [l.to_dict() for l in self.codename_links],
            "unresolved_codenames": self.unresolved_codenames,
            "is_roster_evidence": self.is_roster_evidence,
            "self_contained": self.self_contained,
            "has_aka_mapping": self.has_aka_mapping,
        }


# =============================================================================
# Bundle Candidate
# =============================================================================

@dataclass
class BundleCandidate:
    """
    A candidate evidence bundle for bottleneck scoring.
    
    BundleCandidates are created by grouping related ChunkAnnotations based on:
    - Canonical entity cohesion (chunks about same person)
    - Topic/claim cohesion (roster vs communications)
    - Document/page proximity
    
    The bottleneck scores these bundles (not individual chunks) to select
    the best evidence for synthesis.
    """
    
    # Identity
    bundle_id: str                          # Unique ID for citation (e.g., "bc_0")
    bundle_kind: BundleKind                 # PERSON_EVIDENCE, CODENAME_EVIDENCE, MIXED
    
    # Content
    topic: str                              # What this bundle is about
    chunk_ids: List[int] = field(default_factory=list)
    chunks: List[Dict[str, Any]] = field(default_factory=list)  # Raw chunk data
    annotations: List[ChunkAnnotation] = field(default_factory=list)
    
    # Entity information
    primary_entities: List[int] = field(default_factory=list)   # Canonical entity_ids
    primary_entity_names: List[str] = field(default_factory=list)
    unresolved_codenames: List[str] = field(default_factory=list)
    
    # Quality signals
    confidence: float = 0.0                 # Bundle cohesion confidence (0-1)
    member_yield_estimate: int = 0          # For roster: estimated members named
    self_contained: bool = True             # All references resolved within bundle
    
    # LLM-generated content
    summary: str = ""                       # One-paragraph summary
    key_claims: List[str] = field(default_factory=list)  # Claims this bundle supports
    
    # Source tracking
    source_collections: List[str] = field(default_factory=list)
    unique_documents: int = 0
    
    def chunk_count(self) -> int:
        """Number of chunks in this bundle."""
        return len(self.chunk_ids)
    
    def is_valid(self, min_chunks: int = 2) -> bool:
        """Check if bundle meets minimum requirements."""
        return self.chunk_count() >= min_chunks
    
    def get_representative_text(self) -> str:
        """Get representative text from the bundle."""
        if self.chunks:
            best_chunk = max(self.chunks, key=lambda c: len(c.get("text", "")))
            return best_chunk.get("text", "")[:500]
        return ""
    
    def is_codename_only(self) -> bool:
        """Check if bundle contains only unresolved codenames (risky for roster)."""
        return (self.bundle_kind == BundleKind.CODENAME_EVIDENCE and 
                not self.primary_entities)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "bundle_kind": self.bundle_kind.value,
            "topic": self.topic,
            "chunk_ids": self.chunk_ids,
            "chunk_count": self.chunk_count(),
            "primary_entities": self.primary_entities,
            "primary_entity_names": self.primary_entity_names,
            "unresolved_codenames": self.unresolved_codenames,
            "confidence": self.confidence,
            "member_yield_estimate": self.member_yield_estimate,
            "self_contained": self.self_contained,
            "summary": self.summary,
            "key_claims": self.key_claims,
            "source_collections": self.source_collections,
        }
    
    def format_for_bottleneck(self) -> str:
        """Format bundle for bottleneck comparison prompt."""
        entity_info = ""
        if self.primary_entity_names:
            entity_info = f"Entities: {', '.join(self.primary_entity_names[:5])}"
        if self.unresolved_codenames:
            entity_info += f"\nCodenames (unresolved): {', '.join(self.unresolved_codenames[:5])}"
        
        return f"""BUNDLE: {self.topic}
Kind: {self.bundle_kind.value}
{entity_info}
Chunks: {self.chunk_count()}
Summary: {self.summary}
Key claims: {'; '.join(self.key_claims[:3])}
Representative quote: "{self.get_representative_text()[:300]}..."
"""


# =============================================================================
# Pre-Bundling Result
# =============================================================================

@dataclass
class PreBundlingResult:
    """
    Result of the pre-bundling phase.
    
    Contains the bundles to be passed to the bottleneck, along with
    statistics and metadata about the bundling process.
    """
    
    bundles: List[BundleCandidate] = field(default_factory=list)
    
    # Statistics
    chunks_input: int = 0                   # Total chunks from retrieval
    chunks_selected: int = 0                # Chunks after candidate selection
    chunks_annotated: int = 0               # Chunks with surface extraction
    bundles_created: int = 0                # Bundles formed
    
    # Concordance stats
    entities_resolved: int = 0              # Unique entity IDs found
    codenames_resolved: int = 0             # Codenames linked to entities
    codenames_unresolved: int = 0           # Codenames without links
    
    # Timing
    elapsed_ms: float = 0.0
    surface_extraction_ms: float = 0.0
    concordance_resolution_ms: float = 0.0
    bundling_ms: float = 0.0
    
    # Mode used
    mode: PreBundlingMode = PreBundlingMode.MICRO
    
    def get_all_chunk_ids(self) -> List[int]:
        """Get all chunk IDs across all bundles."""
        ids = []
        for bundle in self.bundles:
            ids.extend(bundle.chunk_ids)
        return list(set(ids))
    
    def get_person_bundles(self) -> List[BundleCandidate]:
        """Get bundles with resolved person evidence."""
        return [b for b in self.bundles if b.bundle_kind == BundleKind.PERSON_EVIDENCE]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundles": [b.to_dict() for b in self.bundles],
            "bundles_count": len(self.bundles),
            "chunks_input": self.chunks_input,
            "chunks_selected": self.chunks_selected,
            "chunks_annotated": self.chunks_annotated,
            "entities_resolved": self.entities_resolved,
            "codenames_resolved": self.codenames_resolved,
            "codenames_unresolved": self.codenames_unresolved,
            "elapsed_ms": self.elapsed_ms,
            "mode": self.mode.value,
        }


# =============================================================================
# Bottleneck Selection (for bundles)
# =============================================================================

@dataclass
class BundleBottleneckSelection:
    """
    Result of bottleneck filtering on bundles.
    
    Contains selected bundles and flattened span IDs for synthesis.
    """
    
    # Selected bundles
    selected_bundles: List[BundleCandidate] = field(default_factory=list)
    selected_bundle_ids: List[str] = field(default_factory=list)
    
    # Flattened spans from selected bundles
    selected_chunk_ids: List[int] = field(default_factory=list)
    
    # Rejected
    rejected_bundle_ids: List[str] = field(default_factory=list)
    
    # Scoring metadata
    bundle_scores: Dict[str, float] = field(default_factory=dict)
    
    # Stats
    bundles_input: int = 0
    bundles_selected: int = 0
    total_chunks_selected: int = 0
    
    def get_chunks_for_synthesis(self) -> List[Dict[str, Any]]:
        """Get all chunks from selected bundles for synthesis."""
        chunks = []
        for bundle in self.selected_bundles:
            chunks.extend(bundle.chunks)
        return chunks
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_bundle_ids": self.selected_bundle_ids,
            "selected_chunk_ids": self.selected_chunk_ids,
            "rejected_bundle_ids": self.rejected_bundle_ids,
            "bundle_scores": self.bundle_scores,
            "bundles_input": self.bundles_input,
            "bundles_selected": self.bundles_selected,
            "total_chunks_selected": self.total_chunks_selected,
        }
