"""
V7 Phase 2: Evidence Bundle Builder

LLM-powered grouping of evidence spans into thematic bundles.

The bundle builder:
1. Analyzes evidence spans to identify topics/themes
2. Groups related spans into bundles
3. Generates summaries for each bundle
4. Manages bundle lifecycle (forming → complete → cited)

Usage:
    builder = BundleBuilder()
    collection = builder.build_bundles(
        spans=evidence_spans,
        question="Who were the members of the Silvermaster network?",
        conn=db_connection,
    )
"""
import os
import json
import sys
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from retrieval.agent.v7_types import (
    EvidenceBundle,
    BundleSpan,
    BundleCollection,
    BundleStatus,
)


@dataclass
class BundleBuilderConfig:
    """Configuration for the bundle builder."""
    
    # LLM settings
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    
    # Bundle settings
    min_spans_per_bundle: int = 2
    max_bundles: int = 10
    min_confidence: float = 0.5
    
    # Grouping settings
    max_spans_to_analyze: int = 50  # Cap for LLM context
    
    verbose: bool = True


class BundleBuilder:
    """
    V7 Phase 2: LLM-powered evidence bundle builder.
    
    Groups related evidence spans into thematic bundles for easier
    citation and presentation.
    """
    
    def __init__(self, config: Optional[BundleBuilderConfig] = None):
        self.config = config or BundleBuilderConfig()
    
    def build_bundles(
        self,
        spans: List[Dict[str, Any]],
        question: str,
        conn=None,  # Optional database connection for additional lookups
        round_number: int = 0,
    ) -> BundleCollection:
        """
        Build evidence bundles from a list of spans.
        
        Args:
            spans: List of span dicts with 'span_id', 'text', 'chunk_id', etc.
            question: The original user question (for context)
            conn: Optional database connection
            round_number: Current round number for lifecycle tracking
            
        Returns:
            BundleCollection with grouped bundles
        """
        
        if not spans:
            return BundleCollection()
        
        # Limit spans for LLM analysis
        analysis_spans = spans[:self.config.max_spans_to_analyze]
        
        # Try LLM-powered grouping first
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                return self._llm_group_bundles(analysis_spans, question, round_number)
            except Exception as e:
                if self.config.verbose:
                    print(f"[BundleBuilder] LLM grouping failed: {e}", file=sys.stderr)
        
        # Fallback to heuristic grouping
        return self._heuristic_group_bundles(analysis_spans, question, round_number)
    
    def _llm_group_bundles(
        self,
        spans: List[Dict[str, Any]],
        question: str,
        round_number: int,
    ) -> BundleCollection:
        """Use LLM to group spans into bundles."""
        
        # Format spans for prompt
        spans_text = self._format_spans_for_prompt(spans)
        
        prompt = f"""Analyze these evidence spans and group them into thematic bundles.

QUESTION: {question}

EVIDENCE SPANS:
{spans_text}

Group the spans by topic/theme. Each bundle should contain spans that support 
the same claim or discuss the same topic.

Output JSON:
{{
  "bundles": [
    {{
      "topic": "Brief description of what this bundle is about",
      "span_indices": [0, 2, 5],  // Which spans belong here (0-indexed)
      "confidence": 0.8,  // How confident the grouping is (0-1)
      "summary": "One paragraph summary of what these spans collectively say",
      "key_claims": ["Claim 1", "Claim 2"]  // Claims this bundle supports
    }}
  ]
}}

GUIDELINES:
- Create 2-{self.config.max_bundles} bundles maximum
- Each span should appear in exactly one bundle
- Bundle topics should be specific (not just "general information")
- Don't create single-span bundles unless truly isolated
- Confidence should reflect how well spans support each other"""

        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You analyze archival evidence and group related information. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=self.config.temperature,
                max_tokens=2000,
            )
            
            content = response.choices[0].message.content
            if content:
                data = json.loads(content)
                return self._parse_llm_bundles(data, spans, round_number)
                
        except Exception as e:
            if self.config.verbose:
                print(f"[BundleBuilder] LLM error: {e}", file=sys.stderr)
            raise
        
        return BundleCollection()
    
    def _format_spans_for_prompt(self, spans: List[Dict[str, Any]]) -> str:
        """Format spans for LLM prompt."""
        lines = []
        for i, span in enumerate(spans):
            text = span.get("text", span.get("span_text", ""))[:400]
            source = span.get("source_label", "")
            page = span.get("page", "")
            lines.append(f"[{i}] ({source}, {page}): {text}")
        return "\n\n".join(lines)
    
    def _parse_llm_bundles(
        self,
        data: Dict[str, Any],
        spans: List[Dict[str, Any]],
        round_number: int,
    ) -> BundleCollection:
        """Parse LLM response into BundleCollection."""
        
        collection = BundleCollection()
        
        for i, bundle_data in enumerate(data.get("bundles", [])):
            topic = bundle_data.get("topic", f"Topic {i+1}")
            confidence = bundle_data.get("confidence", 0.5)
            summary = bundle_data.get("summary", "")
            key_claims = bundle_data.get("key_claims", [])
            span_indices = bundle_data.get("span_indices", [])
            
            # Create bundle
            bundle_id = f"b_{round_number}_{i}"
            bundle = EvidenceBundle(
                bundle_id=bundle_id,
                topic=topic,
                confidence=confidence,
                summary=summary,
                key_claims=key_claims,
                created_round=round_number,
                last_updated_round=round_number,
            )
            
            # Add spans to bundle
            source_collections = set()
            for idx in span_indices:
                if 0 <= idx < len(spans):
                    span_data = spans[idx]
                    bundle_span = BundleSpan(
                        span_id=span_data.get("span_id", f"sp_{idx}"),
                        chunk_id=span_data.get("chunk_id", span_data.get("id", 0)),
                        text=span_data.get("text", span_data.get("span_text", "")),
                        relevance_score=confidence,
                        source_label=span_data.get("source_label", ""),
                        page=span_data.get("page", ""),
                    )
                    bundle.add_span(bundle_span)
                    
                    if span_data.get("source_label"):
                        source_collections.add(span_data["source_label"])
            
            bundle.source_collections = list(source_collections)
            
            # Set status based on span count and confidence
            if bundle.is_sufficient(self.config.min_spans_per_bundle, self.config.min_confidence):
                bundle.status = BundleStatus.COMPLETE
            
            collection.add_bundle(bundle)
        
        return collection
    
    def _heuristic_group_bundles(
        self,
        spans: List[Dict[str, Any]],
        question: str,
        round_number: int,
    ) -> BundleCollection:
        """Fallback heuristic grouping when LLM is unavailable."""
        
        collection = BundleCollection()
        
        # Group by source collection
        by_collection: Dict[str, List[Dict[str, Any]]] = {}
        for span in spans:
            source = span.get("source_label", "unknown")
            if source not in by_collection:
                by_collection[source] = []
            by_collection[source].append(span)
        
        # Create a bundle per collection
        for i, (collection_name, coll_spans) in enumerate(by_collection.items()):
            if len(coll_spans) < self.config.min_spans_per_bundle:
                continue
            
            bundle_id = f"b_{round_number}_{i}"
            bundle = EvidenceBundle(
                bundle_id=bundle_id,
                topic=f"Evidence from {collection_name}",
                confidence=0.5,  # Default confidence for heuristic grouping
                summary=f"Collection of {len(coll_spans)} evidence spans from {collection_name}",
                created_round=round_number,
                last_updated_round=round_number,
                source_collections=[collection_name],
            )
            
            # Add spans
            for j, span_data in enumerate(coll_spans):
                bundle_span = BundleSpan(
                    span_id=span_data.get("span_id", f"sp_{i}_{j}"),
                    chunk_id=span_data.get("chunk_id", span_data.get("id", 0)),
                    text=span_data.get("text", span_data.get("span_text", "")),
                    relevance_score=0.5,
                    source_label=span_data.get("source_label", ""),
                    page=span_data.get("page", ""),
                )
                bundle.add_span(bundle_span)
            
            if bundle.is_sufficient():
                bundle.status = BundleStatus.COMPLETE
            
            collection.add_bundle(bundle)
        
        return collection
    
    def update_bundles(
        self,
        collection: BundleCollection,
        new_spans: List[Dict[str, Any]],
        question: str,
        round_number: int,
    ) -> BundleCollection:
        """
        Update existing bundles with new spans.
        
        Attempts to merge new spans into existing bundles where appropriate,
        or creates new bundles for unrelated spans.
        """
        
        if not new_spans:
            return collection
        
        # Try LLM-based assignment
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and collection.bundles:
            try:
                return self._llm_assign_to_bundles(collection, new_spans, round_number)
            except Exception as e:
                if self.config.verbose:
                    print(f"[BundleBuilder] LLM assignment failed: {e}", file=sys.stderr)
        
        # Fallback: create new bundles from new spans
        new_collection = self.build_bundles(new_spans, question, round_number=round_number)
        
        # Merge new bundles into existing collection
        for bundle in new_collection.bundles:
            collection.add_bundle(bundle)
        
        return collection
    
    def _llm_assign_to_bundles(
        self,
        collection: BundleCollection,
        new_spans: List[Dict[str, Any]],
        round_number: int,
    ) -> BundleCollection:
        """Use LLM to assign new spans to existing bundles."""
        
        # Format existing bundles
        bundles_text = ""
        for i, bundle in enumerate(collection.bundles):
            bundles_text += f"[{i}] {bundle.bundle_id}: {bundle.topic}\n"
            bundles_text += f"    Spans: {bundle.span_count()}, Summary: {bundle.summary[:100]}...\n\n"
        
        # Format new spans
        spans_text = self._format_spans_for_prompt(new_spans)
        
        prompt = f"""Assign these new evidence spans to existing bundles, or mark them for a new bundle.

EXISTING BUNDLES:
{bundles_text}

NEW SPANS TO ASSIGN:
{spans_text}

For each new span, determine:
1. Which existing bundle it best fits (if any)
2. Or if it should start a new bundle

Output JSON:
{{
  "assignments": [
    {{
      "span_index": 0,
      "bundle_index": 1,  // Index of existing bundle, or -1 for new bundle
      "relevance": 0.8    // How well it fits
    }}
  ],
  "new_bundles": [
    {{
      "topic": "New topic for unassigned spans",
      "span_indices": [2, 4]
    }}
  ]
}}"""

        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": "You categorize evidence into existing or new thematic groups. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=self.config.temperature,
            max_tokens=1500,
        )
        
        content = response.choices[0].message.content
        if content:
            data = json.loads(content)
            
            # Process assignments
            for assignment in data.get("assignments", []):
                span_idx = assignment.get("span_index", -1)
                bundle_idx = assignment.get("bundle_index", -1)
                relevance = assignment.get("relevance", 0.5)
                
                if 0 <= span_idx < len(new_spans) and 0 <= bundle_idx < len(collection.bundles):
                    span_data = new_spans[span_idx]
                    bundle_span = BundleSpan(
                        span_id=span_data.get("span_id", f"sp_new_{span_idx}"),
                        chunk_id=span_data.get("chunk_id", span_data.get("id", 0)),
                        text=span_data.get("text", span_data.get("span_text", "")),
                        relevance_score=relevance,
                        source_label=span_data.get("source_label", ""),
                        page=span_data.get("page", ""),
                    )
                    collection.bundles[bundle_idx].add_span(bundle_span)
                    collection.bundles[bundle_idx].last_updated_round = round_number
            
            # Process new bundles
            for i, new_bundle_data in enumerate(data.get("new_bundles", [])):
                topic = new_bundle_data.get("topic", f"New Topic {i+1}")
                span_indices = new_bundle_data.get("span_indices", [])
                
                bundle_id = f"b_{round_number}_new_{i}"
                bundle = EvidenceBundle(
                    bundle_id=bundle_id,
                    topic=topic,
                    created_round=round_number,
                    last_updated_round=round_number,
                )
                
                for idx in span_indices:
                    if 0 <= idx < len(new_spans):
                        span_data = new_spans[idx]
                        bundle_span = BundleSpan(
                            span_id=span_data.get("span_id", f"sp_new_{idx}"),
                            chunk_id=span_data.get("chunk_id", span_data.get("id", 0)),
                            text=span_data.get("text", span_data.get("span_text", "")),
                            relevance_score=0.5,
                            source_label=span_data.get("source_label", ""),
                            page=span_data.get("page", ""),
                        )
                        bundle.add_span(bundle_span)
                
                if bundle.spans:
                    collection.add_bundle(bundle)
        
        return collection
    
    def generate_bundle_summary(
        self,
        bundle: EvidenceBundle,
    ) -> str:
        """Generate or regenerate a summary for a bundle."""
        
        if not bundle.spans:
            return ""
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Fallback to simple concatenation
            texts = [s.text[:200] for s in bundle.spans[:5]]
            return " | ".join(texts)
        
        # Format spans for prompt
        spans_text = "\n".join([
            f"[{i}] ({s.source_label}, {s.page}): {s.text[:300]}"
            for i, s in enumerate(bundle.spans[:10])
        ])
        
        prompt = f"""Summarize this evidence in one paragraph.

TOPIC: {bundle.topic}

EVIDENCE:
{spans_text}

Write a concise summary (2-3 sentences) that captures the key information."""

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You summarize archival evidence concisely."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=300,
            )
            
            return response.choices[0].message.content or ""
            
        except Exception as e:
            if self.config.verbose:
                print(f"[BundleBuilder] Summary generation failed: {e}", file=sys.stderr)
            return bundle.summary  # Return existing summary
