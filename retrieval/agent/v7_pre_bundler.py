"""
V7 Pre-Bundler - Concordance-Aware Evidence Grouping

The Pre-Bundler runs BEFORE the evidence bottleneck to:
1. Select candidate chunks (cap at ~80)
2. Extract named entity surfaces from chunks (persons, codenames, orgs)
3. Resolve codenames to canonical entity IDs via concordance
4. Group related chunks into BundleCandidates
5. Pass bundles to the bottleneck for scoring

This prevents codenames like "Ruble, Raid, Mole" from being treated as members
and enables coherent evidence citation.

Usage:
    pre_bundler = PreBundler(config)
    result = pre_bundler.run(chunks, parsed_query, conn)
    # result.bundles contains BundleCandidate objects for bottleneck
"""
import os
import json
import sys
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict

from retrieval.agent.v7_bundle_types import (
    PreBundlingConfig,
    PreBundlingMode,
    ChunkAnnotation,
    CodenameLink,
    BundleCandidate,
    BundleKind,
    PreBundlingResult,
)
from retrieval.agent.v6_query_parser import ParsedQuery, TaskType


# =============================================================================
# Surface Extraction Prompts
# =============================================================================

SURFACE_EXTRACTION_SYSTEM = """You extract named entities from archival documents.

Extract ONLY what is explicitly present in the text:
- person_surfaces: Names of people (full names, last names, first names)
- codename_surfaces: Apparent codenames (single words in quotes, capitalized terms that seem like aliases)
- org_surfaces: Organization names

Rules:
- Only include text that ACTUALLY appears in the document
- Codenames are often single words, sometimes in quotes or capitalized
- Do NOT infer membership - just extract what you see
- Note if there's an "a.k.a." or alias mapping present

Output valid JSON only."""


def build_surface_extraction_prompt(chunks: List[Dict[str, Any]]) -> str:
    """Build prompt for surface extraction from chunks."""
    
    chunk_texts = []
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")[:600]
        source = chunk.get("source_label", "")
        chunk_texts.append(f"[{i}] ({source}): {text}")
    
    chunks_section = "\n\n".join(chunk_texts)
    
    return f"""Extract named entities from each chunk.

CHUNKS:
{chunks_section}

For EACH chunk, output:
{{
  "extractions": [
    {{
      "chunk_index": 0,
      "person_surfaces": ["Harry White", "Silvermaster"],
      "codename_surfaces": ["Pal", "Mole"],
      "org_surfaces": ["Treasury", "Soviet intelligence"],
      "has_aka_mapping": false,
      "notes": "optional notes"
    }}
  ]
}}"""


# =============================================================================
# Semantic Bundling Prompts
# =============================================================================

BUNDLING_SYSTEM = """You group related evidence into thematic bundles.

Each bundle should contain chunks that:
- Discuss the same person(s) or topic
- Support the same claim(s)
- Come from related context (same document section, related pages)

Output valid JSON only."""


def build_bundling_prompt(
    annotations: List[ChunkAnnotation],
    question: str,
    task_type: TaskType,
    max_bundles: int = 10,
) -> str:
    """Build prompt for semantic bundling."""
    
    # Format annotations for prompt
    ann_texts = []
    for i, ann in enumerate(annotations):
        entities = ann.resolved_people or []
        codenames = ann.unresolved_codenames or []
        text_preview = ann.chunk_text[:300] if ann.chunk_text else ""
        
        ann_texts.append(f"""[{i}] (chunk_id={ann.chunk_id}, {ann.source_label})
  Persons: {ann.person_surfaces}
  Codenames: {ann.codename_surfaces}
  Resolved entities: {entities}
  Unresolved codenames: {codenames}
  Text: "{text_preview}..."
""")
    
    annotations_section = "\n".join(ann_texts)
    
    task_guidance = ""
    if task_type == TaskType.ROSTER_ENUMERATION:
        task_guidance = """
TASK: Roster enumeration - the user wants to identify MEMBERS.
- Prefer bundles that name specific people (not just codenames)
- Group chunks that discuss the same member
- SEPARATE person evidence from codename-only evidence
- Mark bundles with only unresolved codenames as "CODENAME_EVIDENCE"
"""
    
    return f"""Group these annotated chunks into thematic bundles.

QUESTION: {question}
{task_guidance}

ANNOTATED CHUNKS:
{annotations_section}

Create {max_bundles} or fewer bundles. Each bundle should group related chunks.

Output JSON:
{{
  "bundles": [
    {{
      "topic": "Brief description of what this bundle is about",
      "chunk_indices": [0, 2, 5],
      "bundle_kind": "PERSON_EVIDENCE" or "CODENAME_EVIDENCE" or "MIXED",
      "primary_entities": [123, 456],
      "unresolved_codenames": ["Pal"],
      "confidence": 0.8,
      "member_yield_estimate": 3,
      "key_claims": ["Claim 1", "Claim 2"],
      "summary": "One paragraph summary"
    }}
  ]
}}

RULES:
- Each chunk should appear in exactly ONE bundle
- Chunks with the same resolved entity IDs should be grouped together
- Separate PERSON_EVIDENCE (has resolved people) from CODENAME_EVIDENCE (only codenames)
- member_yield_estimate = how many members this bundle names (for roster queries)"""


# =============================================================================
# Pre-Bundler
# =============================================================================

class PreBundler:
    """
    Concordance-Aware Pre-Bundler.
    
    Groups retrieval chunks into BundleCandidates before bottleneck scoring.
    This enables:
    - Codename resolution via concordance tools
    - Coherent evidence grouping
    - Bundle-level bottleneck scoring
    - Prevention of codename-as-member errors
    """
    
    def __init__(self, config: Optional[PreBundlingConfig] = None):
        self.config = config or PreBundlingConfig()
    
    def run(
        self,
        chunks: List[Dict[str, Any]],
        parsed_query: ParsedQuery,
        conn=None,
    ) -> PreBundlingResult:
        """
        Main entry point for pre-bundling.
        
        Args:
            chunks: Raw chunks from retrieval
            parsed_query: Parsed query with task type
            conn: Database connection for concordance lookups
            
        Returns:
            PreBundlingResult with BundleCandidates for bottleneck
        """
        start_time = time.time()
        result = PreBundlingResult(
            chunks_input=len(chunks),
            mode=self.config.mode,
        )
        
        if self.config.verbose:
            print(f"\n  [PreBundler] Starting ({self.config.mode.value} mode)", file=sys.stderr)
            print(f"    Input: {len(chunks)} chunks", file=sys.stderr)
        
        # Handle different modes
        if self.config.mode == PreBundlingMode.OFF:
            # No bundling - return empty result (controller will use original chunks)
            result.elapsed_ms = (time.time() - start_time) * 1000
            return result
        
        if self.config.mode == PreBundlingMode.PASSTHROUGH:
            # Fast mode: minimal bundling, just codename guard
            result = self._passthrough_mode(chunks, parsed_query, conn)
        elif self.config.mode == PreBundlingMode.MICRO:
            # Medium mode: micro-bundles around top seeds
            result = self._micro_mode(chunks, parsed_query, conn)
        else:  # SEMANTIC
            # Full mode: surface extraction + concordance + semantic bundling
            result = self._semantic_mode(chunks, parsed_query, conn)
        
        result.elapsed_ms = (time.time() - start_time) * 1000
        
        if self.config.verbose:
            print(f"    Output: {len(result.bundles)} bundles", file=sys.stderr)
            print(f"    Elapsed: {result.elapsed_ms:.0f}ms", file=sys.stderr)
        
        return result
    
    # =========================================================================
    # Semantic Mode (Full)
    # =========================================================================
    
    def _semantic_mode(
        self,
        chunks: List[Dict[str, Any]],
        parsed_query: ParsedQuery,
        conn,
    ) -> PreBundlingResult:
        """Full semantic bundling with concordance resolution."""
        
        result = PreBundlingResult(
            chunks_input=len(chunks),
            mode=PreBundlingMode.SEMANTIC,
        )
        
        # Step 1: Select candidates
        candidates = self._select_candidates(chunks)
        result.chunks_selected = len(candidates)
        
        if self.config.verbose:
            print(f"    [1/4] Selected {len(candidates)} candidates", file=sys.stderr)
        
        # Step 2: Extract surfaces
        surface_start = time.time()
        annotations = self._extract_surfaces(candidates)
        result.surface_extraction_ms = (time.time() - surface_start) * 1000
        result.chunks_annotated = len(annotations)
        
        if self.config.verbose:
            print(f"    [2/4] Extracted surfaces from {len(annotations)} chunks "
                  f"({result.surface_extraction_ms:.0f}ms)", file=sys.stderr)
        
        # Step 3: Resolve concordance
        concordance_start = time.time()
        annotations = self._resolve_concordance(annotations, conn)
        result.concordance_resolution_ms = (time.time() - concordance_start) * 1000
        
        # Count resolution stats
        all_entities = set()
        resolved_codenames = 0
        unresolved_codenames = 0
        for ann in annotations:
            all_entities.update(ann.resolved_people)
            for link in ann.codename_links:
                if link.entity_id:
                    resolved_codenames += 1
                    all_entities.add(link.entity_id)
            unresolved_codenames += len(ann.unresolved_codenames)
        
        result.entities_resolved = len(all_entities)
        result.codenames_resolved = resolved_codenames
        result.codenames_unresolved = unresolved_codenames
        
        if self.config.verbose:
            print(f"    [3/4] Resolved concordance: {result.entities_resolved} entities, "
                  f"{resolved_codenames} codenames ({result.concordance_resolution_ms:.0f}ms)", 
                  file=sys.stderr)
        
        # Step 4: Build bundles
        bundling_start = time.time()
        bundles = self._build_bundles(annotations, parsed_query, candidates)
        result.bundling_ms = (time.time() - bundling_start) * 1000
        result.bundles = bundles
        result.bundles_created = len(bundles)
        
        if self.config.verbose:
            print(f"    [4/4] Created {len(bundles)} bundles ({result.bundling_ms:.0f}ms)", 
                  file=sys.stderr)
            for i, b in enumerate(bundles[:3]):
                print(f"      [{i}] {b.bundle_kind.value}: {b.topic[:50]}... "
                      f"({b.chunk_count()} chunks)", file=sys.stderr)
        
        return result
    
    # =========================================================================
    # Candidate Selection
    # =========================================================================
    
    def _select_candidates(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Select candidate chunks for bundling.
        
        Strategy:
        - Take top N by retrieval score
        - Cap total at max_candidate_chunks
        """
        if len(chunks) <= self.config.max_candidate_chunks:
            return chunks
        
        # Sort by score if available, otherwise by position
        def get_score(chunk: Dict) -> float:
            return chunk.get("score", chunk.get("retrieval_score", 0.0))
        
        sorted_chunks = sorted(chunks, key=get_score, reverse=True)
        return sorted_chunks[:self.config.max_candidate_chunks]
    
    # =========================================================================
    # Surface Extraction
    # =========================================================================
    
    def _extract_surfaces(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[ChunkAnnotation]:
        """
        Extract named entity surfaces from chunks using LLM.
        
        Returns ChunkAnnotation objects with person_surfaces, codename_surfaces, etc.
        """
        annotations = []
        
        # Process in batches
        batch_size = self.config.surface_batch_size
        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start:batch_start + batch_size]
            batch_annotations = self._extract_surfaces_batch(batch, batch_start)
            annotations.extend(batch_annotations)
        
        return annotations
    
    def _extract_surfaces_batch(
        self,
        batch: List[Dict[str, Any]],
        start_idx: int,
    ) -> List[ChunkAnnotation]:
        """Extract surfaces from a batch of chunks."""
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Fallback: create annotations with empty surfaces
            return self._fallback_surface_extraction(batch, start_idx)
        
        prompt = build_surface_extraction_prompt(batch)
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=self.config.surface_model,
                messages=[
                    {"role": "system", "content": SURFACE_EXTRACTION_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2000,
            )
            
            content = response.choices[0].message.content
            if content:
                data = json.loads(content)
                return self._parse_surface_extractions(data, batch, start_idx)
                
        except Exception as e:
            if self.config.verbose:
                print(f"    [SurfaceExtraction] Error: {e}", file=sys.stderr)
        
        return self._fallback_surface_extraction(batch, start_idx)
    
    def _parse_surface_extractions(
        self,
        data: Dict[str, Any],
        batch: List[Dict[str, Any]],
        start_idx: int,
    ) -> List[ChunkAnnotation]:
        """Parse LLM surface extraction response."""
        
        annotations = []
        extractions = data.get("extractions", [])
        
        for i, chunk in enumerate(batch):
            # Find extraction for this chunk
            extraction = None
            for ext in extractions:
                if ext.get("chunk_index") == i:
                    extraction = ext
                    break
            
            ann = ChunkAnnotation(
                chunk_id=chunk.get("id", start_idx + i),
                chunk_text=chunk.get("text", ""),
                source_label=chunk.get("source_label", ""),
                page=chunk.get("page", ""),
                doc_id=chunk.get("doc_id"),
                retrieval_score=chunk.get("score", 0.0),
            )
            
            if extraction:
                ann.person_surfaces = extraction.get("person_surfaces", [])
                ann.codename_surfaces = extraction.get("codename_surfaces", [])
                ann.org_surfaces = extraction.get("org_surfaces", [])
                ann.has_aka_mapping = extraction.get("has_aka_mapping", False)
            
            annotations.append(ann)
        
        return annotations
    
    def _fallback_surface_extraction(
        self,
        batch: List[Dict[str, Any]],
        start_idx: int,
    ) -> List[ChunkAnnotation]:
        """Fallback when LLM is unavailable."""
        
        annotations = []
        for i, chunk in enumerate(batch):
            ann = ChunkAnnotation(
                chunk_id=chunk.get("id", start_idx + i),
                chunk_text=chunk.get("text", ""),
                source_label=chunk.get("source_label", ""),
                page=chunk.get("page", ""),
                doc_id=chunk.get("doc_id"),
                retrieval_score=chunk.get("score", 0.0),
            )
            # No surface extraction in fallback
            annotations.append(ann)
        
        return annotations
    
    # =========================================================================
    # Concordance Resolution
    # =========================================================================
    
    def _resolve_concordance(
        self,
        annotations: List[ChunkAnnotation],
        conn,
    ) -> List[ChunkAnnotation]:
        """
        Resolve surfaces to entity IDs using concordance tools.
        
        For each annotation:
        1. Look up person_surfaces → entity_ids
        2. Look up codename_surfaces → potential entity mappings
        """
        if conn is None:
            return annotations
        
        # Collect all unique surfaces to resolve
        person_surfaces: Set[str] = set()
        codename_surfaces: Set[str] = set()
        
        for ann in annotations:
            person_surfaces.update(ann.person_surfaces)
            codename_surfaces.update(ann.codename_surfaces)
        
        # Resolve person surfaces
        person_to_entity = self._resolve_person_surfaces(list(person_surfaces), conn)
        
        # Resolve codename surfaces
        codename_links = self._resolve_codename_surfaces(list(codename_surfaces), conn)
        
        # Update annotations
        for ann in annotations:
            # Resolve people
            for surface in ann.person_surfaces:
                entity_id = person_to_entity.get(surface.lower())
                if entity_id:
                    ann.resolved_people.append(entity_id)
            ann.resolved_people = list(set(ann.resolved_people))
            
            # Resolve codenames
            for surface in ann.codename_surfaces:
                link = codename_links.get(surface.lower())
                if link:
                    ann.codename_links.append(link)
                    if not link.entity_id:
                        ann.unresolved_codenames.append(surface)
                else:
                    ann.unresolved_codenames.append(surface)
        
        return annotations
    
    def _resolve_person_surfaces(
        self,
        surfaces: List[str],
        conn,
    ) -> Dict[str, int]:
        """Look up person surfaces in entity database."""
        
        if not surfaces:
            return {}
        
        result = {}
        
        # Rollback any failed transaction at the start
        try:
            conn.rollback()
        except:
            pass
        
        # Look up each surface individually to handle errors gracefully
        for surface in surfaces:
            try:
                with conn.cursor() as cur:
                    # First check canonical name
                    cur.execute("""
                        SELECT id, canonical_name
                        FROM entities
                        WHERE LOWER(canonical_name) = LOWER(%s)
                        AND entity_type = 'PERSON'
                        LIMIT 1
                    """, (surface,))
                    
                    row = cur.fetchone()
                    if not row:
                        # Check aliases
                        cur.execute("""
                            SELECT e.id, e.canonical_name
                            FROM entities e
                            JOIN entity_aliases ea ON ea.entity_id = e.id
                            WHERE LOWER(ea.alias) = LOWER(%s)
                            AND e.entity_type = 'PERSON'
                            LIMIT 1
                        """, (surface,))
                        row = cur.fetchone()
                    
                    if row:
                        result[surface.lower()] = row[0]
            except Exception as e:
                if self.config.verbose:
                    print(f"    [ConcordanceResolution] Person lookup error for '{surface}': {e}", file=sys.stderr)
                try:
                    conn.rollback()
                except:
                    pass
        
        return result
    
    def _resolve_codename_surfaces(
        self,
        surfaces: List[str],
        conn,
    ) -> Dict[str, CodenameLink]:
        """
        Look up codename surfaces using concordance tools.
        
        Resolution ladder:
        1. Check entities canonical_name for direct matches
        2. Check entity_aliases for alias matches
        3. Store confidence based on match quality (canonical = 0.9, alias = 0.7)
        """
        if not surfaces:
            return {}
        
        result = {}
        
        # Rollback any failed transaction at the start
        try:
            conn.rollback()
        except:
            pass
        
        # Look up each surface individually to handle errors gracefully
        for surface in surfaces:
            try:
                with conn.cursor() as cur:
                    # Try direct lookup - first canonical name, then aliases
                    # Check canonical name first (highest confidence)
                    cur.execute("""
                        SELECT id, canonical_name
                        FROM entities
                        WHERE LOWER(canonical_name) = LOWER(%s)
                        LIMIT 1
                    """, (surface,))
                    
                    row = cur.fetchone()
                    confidence = 0.9  # High confidence for canonical match
                    
                    if not row:
                        # Check aliases
                        cur.execute("""
                            SELECT e.id, e.canonical_name
                            FROM entities e
                            JOIN entity_aliases ea ON ea.entity_id = e.id
                            WHERE LOWER(ea.alias) = LOWER(%s)
                            LIMIT 1
                        """, (surface,))
                        row = cur.fetchone()
                        confidence = 0.7  # Lower confidence for alias match
                    
                    if row:
                        result[surface.lower()] = CodenameLink(
                            codename=surface,
                            entity_id=row[0],
                            entity_name=row[1],
                            confidence=confidence,
                            resolution_method="direct",
                        )
                    else:
                        # Try concordance table if it exists
                        try:
                            cur.execute("""
                                SELECT target_entity_id, confidence
                                FROM concordance_mappings
                                WHERE LOWER(source_term) = LOWER(%s)
                                AND mapping_type = 'codename'
                                LIMIT 1
                            """, (surface,))
                            
                            row = cur.fetchone()
                            if row:
                                # Get entity name
                                cur.execute("SELECT canonical_name FROM entities WHERE id = %s", (row[0],))
                                name_row = cur.fetchone()
                                
                                result[surface.lower()] = CodenameLink(
                                    codename=surface,
                                    entity_id=row[0],
                                    entity_name=name_row[0] if name_row else None,
                                    confidence=row[1] or 0.5,
                                    resolution_method="concordance",
                                )
                        except:
                            pass  # Table may not exist
                        
                        # If still not found, mark as unresolved
                        if surface.lower() not in result:
                            result[surface.lower()] = CodenameLink(
                                codename=surface,
                                entity_id=None,
                                entity_name=None,
                                confidence=0.0,
                                resolution_method="none",
                            )
                            
            except Exception as e:
                if self.config.verbose:
                    print(f"    [ConcordanceResolution] Codename lookup error for '{surface}': {e}", file=sys.stderr)
                try:
                    conn.rollback()
                except:
                    pass
                # Mark as unresolved on error
                result[surface.lower()] = CodenameLink(
                    codename=surface,
                    entity_id=None,
                    entity_name=None,
                    confidence=0.0,
                    resolution_method="error",
                )
        
        return result
    
    # =========================================================================
    # Bundle Building
    # =========================================================================
    
    def _build_bundles(
        self,
        annotations: List[ChunkAnnotation],
        parsed_query: ParsedQuery,
        original_chunks: List[Dict[str, Any]],
    ) -> List[BundleCandidate]:
        """
        Group annotations into BundleCandidates.
        
        Uses LLM for semantic grouping, with fallback to entity-based grouping.
        """
        if not annotations:
            return []
        
        # Try LLM-based bundling
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                return self._llm_build_bundles(annotations, parsed_query, original_chunks)
            except Exception as e:
                if self.config.verbose:
                    print(f"    [Bundling] LLM error: {e}", file=sys.stderr)
        
        # Fallback to heuristic bundling
        return self._heuristic_build_bundles(annotations, original_chunks)
    
    def _llm_build_bundles(
        self,
        annotations: List[ChunkAnnotation],
        parsed_query: ParsedQuery,
        original_chunks: List[Dict[str, Any]],
    ) -> List[BundleCandidate]:
        """Use LLM to group annotations into bundles."""
        
        prompt = build_bundling_prompt(
            annotations=annotations,
            question=parsed_query.original_query,
            task_type=parsed_query.task_type,
            max_bundles=self.config.max_bundles,
        )
        
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model=self.config.bundle_model,
            messages=[
                {"role": "system", "content": BUNDLING_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=3000,
        )
        
        content = response.choices[0].message.content
        if not content:
            return self._heuristic_build_bundles(annotations, original_chunks)
        
        data = json.loads(content)
        return self._parse_llm_bundles(data, annotations, original_chunks)
    
    def _parse_llm_bundles(
        self,
        data: Dict[str, Any],
        annotations: List[ChunkAnnotation],
        original_chunks: List[Dict[str, Any]],
    ) -> List[BundleCandidate]:
        """Parse LLM bundling response."""
        
        bundles = []
        
        # Create chunk lookup by ID
        chunk_by_id = {c.get("id"): c for c in original_chunks}
        ann_by_idx = {i: ann for i, ann in enumerate(annotations)}
        
        for i, bundle_data in enumerate(data.get("bundles", [])):
            chunk_indices = bundle_data.get("chunk_indices", [])
            
            # Gather chunks and annotations
            bundle_chunks = []
            bundle_annotations = []
            bundle_chunk_ids = []
            
            for idx in chunk_indices:
                if idx < len(annotations):
                    ann = annotations[idx]
                    bundle_annotations.append(ann)
                    bundle_chunk_ids.append(ann.chunk_id)
                    
                    # Get original chunk
                    chunk = chunk_by_id.get(ann.chunk_id)
                    if chunk:
                        bundle_chunks.append(chunk)
            
            if not bundle_chunks:
                continue
            
            # Determine bundle kind
            kind_str = bundle_data.get("bundle_kind", "MIXED")
            try:
                kind = BundleKind(kind_str)
            except:
                kind = BundleKind.MIXED
            
            # Collect entities and codenames
            primary_entities = []
            primary_names = []
            unresolved = []
            
            for ann in bundle_annotations:
                primary_entities.extend(ann.resolved_people)
                for link in ann.codename_links:
                    if link.is_strong():
                        primary_entities.append(link.entity_id)
                        if link.entity_name:
                            primary_names.append(link.entity_name)
                unresolved.extend(ann.unresolved_codenames)
            
            primary_entities = list(set([e for e in primary_entities if e]))
            primary_names = list(set(primary_names))
            unresolved = list(set(unresolved))
            
            # Collect source collections
            source_collections = list(set(
                ann.source_label for ann in bundle_annotations if ann.source_label
            ))
            
            bundle = BundleCandidate(
                bundle_id=f"bc_{i}",
                bundle_kind=kind,
                topic=bundle_data.get("topic", f"Bundle {i}"),
                chunk_ids=bundle_chunk_ids,
                chunks=bundle_chunks,
                annotations=bundle_annotations,
                primary_entities=primary_entities,
                primary_entity_names=primary_names,
                unresolved_codenames=unresolved,
                confidence=bundle_data.get("confidence", 0.5),
                member_yield_estimate=bundle_data.get("member_yield_estimate", 0),
                summary=bundle_data.get("summary", ""),
                key_claims=bundle_data.get("key_claims", []),
                source_collections=source_collections,
                unique_documents=len(set(ann.doc_id for ann in bundle_annotations if ann.doc_id)),
            )
            
            bundles.append(bundle)
        
        return bundles[:self.config.max_bundles]
    
    def _heuristic_build_bundles(
        self,
        annotations: List[ChunkAnnotation],
        original_chunks: List[Dict[str, Any]],
    ) -> List[BundleCandidate]:
        """
        Fallback heuristic bundling when LLM is unavailable.
        
        Groups by:
        1. Shared resolved entity IDs
        2. Source collection
        """
        # Create chunk lookup
        chunk_by_id = {c.get("id"): c for c in original_chunks}
        
        # Group by primary entity
        by_entity: Dict[int, List[ChunkAnnotation]] = defaultdict(list)
        no_entity: List[ChunkAnnotation] = []
        
        for ann in annotations:
            if ann.resolved_people:
                # Use first resolved person as primary
                by_entity[ann.resolved_people[0]].append(ann)
            else:
                no_entity.append(ann)
        
        bundles = []
        bundle_idx = 0
        
        # Create bundles for each entity
        for entity_id, anns in by_entity.items():
            if len(anns) < self.config.min_chunks_per_bundle:
                continue
            
            # Cap chunks per bundle
            anns = anns[:self.config.max_chunks_per_bundle]
            
            bundle_chunks = [chunk_by_id.get(ann.chunk_id) for ann in anns]
            bundle_chunks = [c for c in bundle_chunks if c]
            
            bundle = BundleCandidate(
                bundle_id=f"bc_{bundle_idx}",
                bundle_kind=BundleKind.PERSON_EVIDENCE,
                topic=f"Evidence about entity {entity_id}",
                chunk_ids=[ann.chunk_id for ann in anns],
                chunks=bundle_chunks,
                annotations=anns,
                primary_entities=[entity_id],
                confidence=0.5,
            )
            
            bundles.append(bundle)
            bundle_idx += 1
            
            if bundle_idx >= self.config.max_bundles:
                break
        
        # If room, add a bundle for unassigned chunks
        if no_entity and bundle_idx < self.config.max_bundles:
            anns = no_entity[:self.config.max_chunks_per_bundle]
            bundle_chunks = [chunk_by_id.get(ann.chunk_id) for ann in anns]
            bundle_chunks = [c for c in bundle_chunks if c]
            
            if len(bundle_chunks) >= self.config.min_chunks_per_bundle:
                unresolved = []
                for ann in anns:
                    unresolved.extend(ann.unresolved_codenames)
                
                bundle = BundleCandidate(
                    bundle_id=f"bc_{bundle_idx}",
                    bundle_kind=BundleKind.CODENAME_EVIDENCE if unresolved else BundleKind.MIXED,
                    topic="Other evidence",
                    chunk_ids=[ann.chunk_id for ann in anns],
                    chunks=bundle_chunks,
                    annotations=anns,
                    unresolved_codenames=list(set(unresolved)),
                    confidence=0.3,
                )
                bundles.append(bundle)
        
        return bundles
    
    # =========================================================================
    # Passthrough Mode (Fast)
    # =========================================================================
    
    def _passthrough_mode(
        self,
        chunks: List[Dict[str, Any]],
        parsed_query: ParsedQuery,
        conn,
    ) -> PreBundlingResult:
        """
        Fast passthrough mode: minimal bundling.
        
        Creates a single bundle from all chunks with basic codename detection.
        Synthesis will do post-selection codename guard.
        """
        result = PreBundlingResult(
            chunks_input=len(chunks),
            mode=PreBundlingMode.PASSTHROUGH,
        )
        
        # Select top chunks
        candidates = self._select_candidates(chunks)
        result.chunks_selected = len(candidates)
        
        # Create single bundle from all candidates
        bundle = BundleCandidate(
            bundle_id="bc_passthrough",
            bundle_kind=BundleKind.MIXED,
            topic="All retrieved evidence (passthrough)",
            chunk_ids=[c.get("id", i) for i, c in enumerate(candidates)],
            chunks=candidates,
            confidence=0.5,
            summary="Passthrough mode - single bundle with all evidence",
        )
        
        result.bundles = [bundle]
        result.bundles_created = 1
        
        return result
    
    # =========================================================================
    # Micro Mode (Medium)
    # =========================================================================
    
    def _micro_mode(
        self,
        chunks: List[Dict[str, Any]],
        parsed_query: ParsedQuery,
        conn,
    ) -> PreBundlingResult:
        """
        Micro-bundle mode: bundle seeds with neighbors.
        
        Takes top N chunks and creates micro-bundles around each,
        fetching adjacent chunks if available.
        """
        result = PreBundlingResult(
            chunks_input=len(chunks),
            mode=PreBundlingMode.MICRO,
        )
        
        # Select top seeds
        candidates = self._select_candidates(chunks)[:20]  # Top 20 seeds
        result.chunks_selected = len(candidates)
        
        bundles = []
        chunk_by_id = {c.get("id"): c for c in chunks}
        used_chunk_ids = set()
        
        for i, seed in enumerate(candidates):
            if i >= self.config.max_bundles:
                break
            
            seed_id = seed.get("id")
            if seed_id in used_chunk_ids:
                continue
            
            # Create micro-bundle with just the seed (neighbors would require DB)
            bundle_chunks = [seed]
            bundle_ids = [seed_id]
            used_chunk_ids.add(seed_id)
            
            # Try to get neighbors from same document
            seed_doc_id = seed.get("doc_id")
            if seed_doc_id:
                for other in chunks:
                    if other.get("doc_id") == seed_doc_id and other.get("id") not in used_chunk_ids:
                        if len(bundle_chunks) < self.config.max_chunks_per_bundle:
                            bundle_chunks.append(other)
                            bundle_ids.append(other.get("id"))
                            used_chunk_ids.add(other.get("id"))
            
            bundle = BundleCandidate(
                bundle_id=f"bc_micro_{i}",
                bundle_kind=BundleKind.MIXED,
                topic=f"Micro-bundle around chunk {seed_id}",
                chunk_ids=bundle_ids,
                chunks=bundle_chunks,
                confidence=0.4,
            )
            bundles.append(bundle)
        
        result.bundles = bundles
        result.bundles_created = len(bundles)
        
        return result
