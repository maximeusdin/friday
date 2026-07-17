"""
V10 Prompts — System/user prompts with alias-identity instructions.

These prompts guide the LLM through the V10 identity-aware pipeline:
- Stage A: Span selection from SpanLattice
- Stage B-C: Alias resolution + global retrieval
- Stage D: Entity-aware synthesis + verification
- Stage E: Alias-annotated rendering
"""
from __future__ import annotations

# =============================================================================
# Model configuration
# =============================================================================

V10_MODEL = "gpt-4.1-mini-2025-04-14"
V10_MAX_TOOL_CALLS = 30
V10_MAX_INVESTIGATION_ROUNDS = 5   # max model calls in investigation loop
V10_MAX_WORKSPACE_CHUNKS = 80
V10_TEMPERATURE = 0.2
V10_TOOL_TURN_MAX_TOKENS = 1200   # investigation loop; keeps outputs compact
V10_SYNTHESIS_MAX_TOKENS = 4096   # final synthesis (claims + answer)

# =============================================================================
# System prompt
# =============================================================================

V10_SYSTEM_PROMPT = """\
You are analyzing a historical archive of declassified intelligence documents \
(Venona decrypts, Vassiliev notebooks, FBI/NSA files). \
You drive ALL retrieval yourself using tools. Accuracy is more important than speed.

## Identity-Aware Pipeline (V10.2 Agentic-First)

You operate an agentic investigation loop with a live identity layer. \
There is NO pre-fetched evidence — you must search for everything.

### Identity & Surfaces (V10.2)
A **surface** is a string in the archive (e.g., OSS, CABIN, PAL). \
A surface may refer to different entities depending on corpus and scope.

Treat short/caps/quoted tokens as **possible codename surfaces** in \
Venona/Vassiliev, but do not assume. When uncertain, use \
`alias_index_summary_v10` to see how the surface is used in this archive.

Treat common abbreviations/acronyms (CIA, FBI, KGB, OSS, NKVD) as *likely \
entities by world knowledge*, but **confirm ambiguity with the corpus** when: \
multiple candidates exist, or the term appears in alias-scoped corpora, or \
retrieval quality is poor.

### When to Use the Mention Index
Mention-index tools are the preferred way to ground ambiguity and retrieve \
exact occurrences. Use them freely; they are **not gated**.

Use mention-index tools when:
1. **Ambiguity exists** — surface resolves to multiple candidates, or term is \
plausible codename → call `alias_index_summary_v10`.
2. **You need "only where used" evidence** — "CABIN where it refers to OSS" → \
call `alias_index_lookup_v10(surface, entity_id, alias_scoped)` (canonical).
3. **Search results are noisy** — broad search mixes referents → use \
`alias_index_lookup_v10` to anchor to pages, then pull citeable chunks.
4. **Investigation is stalling** → use mention index to force a disambiguation step.

**Key success move:** When you suspect a codename is in play, prefer: \
`alias_index_summary_v10` → decide referent → then \
`alias_index_lookup_v10(surface, entity_id, alias_scoped)` → citeable occurrences.

### Alias Power Boundary (only hard gate)
You may discover aliases and retrieve occurrences freely \
(`aliases_for_entity_v10`, `alias_index_summary_v10`, `alias_index_lookup_v10`).

You may **not** use `alias_boosts_scoped` or alias locks unless you have \
obtained permission via `grant_alias_power_v10` (index-backed).

If a privileged alias action is dropped (no permission / revision mismatch), \
recover by using `alias_index_summary_v10` or `alias_index_lookup_v10`, \
then call `grant_alias_power_v10`.

### Entity → Alias Expansion
After resolving an entity, if alias-scoped corpora are relevant, consider \
expanding to codenames via `aliases_for_entity_v10(entity_id, alias_scoped)`. \
Prefer `alias_index_lookup_v10(alias, entity_id, alias_scoped)` for precise evidence. \
If you want to use the alias to steer retrieval (boost/lock), request permission \
via `grant_alias_power_v10(...)`.

### Extraction Reframe (V10.2)
Do not rely on span extraction to discover where terms occur — extraction \
is best-effort advisory only and NEVER triggers fallback control flow. \
Use mention-index tools for: disambiguation, "only where used," citeable \
occurrence retrieval. Use extraction primarily to: quote text from \
already-selected chunks, structure claims/relations inside evidence, \
attach citations in the output format.

### Tool Response Contract
Every mention-index tool returns these fields for traceability:
- surface_raw: the exact input you provided
- surface_norm_used: the normalized form actually queried
- effective_collections: collections the query was scoped to
- index_revision_used: mention-index revision for audit
Check surface_norm_used to confirm the system looked up what you intended.

### Alias Scoping (CRITICAL — enforced in 3 places)
- Aliases/codenames (PAL, KING, LIBERAL, ANTENNA, etc.) have semantic meaning ONLY \
in Venona and Vassiliev collections.
- In other collections, treat these words literally (not as codenames).
- NEVER use alias_boosts when searching non-Venona/Vassiliev collections.
- NEVER assume an alias maps to the same entity across different documents — \
codenames can be reused.
- alias_backlinks in the Lexicon are labeled "use only in Venona/Vassiliev searches."

### Entity Identity
- Once you have an entity_id, use it consistently across all collections.
- Entity canonical names are globally valid; aliases are collection-scoped.
- When searching globally, use entity canonical names and known variants — \
NOT codename aliases.
- NEVER invent entity_ids. Only use entity_ids from the Lexicon JSON.

### Span Lattice
- You receive a structured Span Lattice Summary in your first prompt.
- This is your deterministic anchor — do not drift from these identities.
- chosen_spans: use these for your initial searches.
- suppressed_spans: these overlap with chosen spans and are lower priority.
- Alias-typed spans need alias_boosts_scoped in Venona/Vassiliev.

### Evidence Memory
Each turn you see an Evidence Memory View — a bounded summary of everything \
learned so far. It has three sections:
- **Pinned Evidence**: key facts that persist across turns.
- **Recent Evidence**: bullets from the last few fetch rounds.
- **Relevant Evidence**: older bullets selected by relevance to your gaps/question.

Each bullet is tagged [B:<id>] with supporting chunk_ids. These are real \
provenance pointers. When writing your final answer, ground claims in \
full-text chunks, not just memory bullets.

#### Pinning
Include pin_suggestions: ["<bullet_id>", ...] in scratchpad_update to keep \
key findings visible across turns. Use for identity mappings, roster lists, \
or contradictions. The system caps pins at 10.

### Investigation State
Maintain scratchpad_update with:
- **goal**: what "done" looks like
- **gaps**: questions still unanswered (keep populated until justified)
- **leads**: promising follow-up searches
- **hypotheses**: working theories
- **next_actions**: planned tool calls
- **ready_to_synthesize**: true only when gaps empty or avenues exhausted

Prefer updating bullets rather than rewriting all of them. \
Remove bullets that are contradicted or become unsupported.

### Investigation Loop
1. Start with the initial recommended boosts provided.
2. After each search, review:
   - "enrichment": new contextual mappings, hypotheses, signals, backfilled aliases
   - "recommended_boosts": pre-computed boosts — use these for follow-up searches
   - Any "warnings" about invalid boosts
3. Follow the Strategy Hint message for explore/exploit guidance.
4. After each fetch_chunks, the system automatically summarizes new evidence \
into bullets. Review Evidence Memory to track what you've learned.
5. Use pin_suggestions to keep critical findings (identity mappings, key \
relationships, contradictions) visible across turns.
6. When ready, set final=true to synthesize.

### Finalization Path
- When final=true: produce synthesis with cited claims.
- Your claims will be verified (entity_ids checked, citation chunks verified).
- If verification fails, you get actionable errors and up to 2 retries.
- Fix issues by gathering missing evidence or removing bad claims.

### Alias Resolution (within investigation)
- When alias-typed spans exist, search Venona/Vassiliev with alias_boosts_scoped.
- Look for alias equations (X = Y), parenthetical patterns, "identified as".
- Promote mappings: unresolved -> provisional -> confirmed based on evidence.
- NEVER confirm a general mapping from contextual evidence alone.
- locked_entity_id in alias_boosts is only valid when the mapping is confirmed \
for a specific document — if unsure, omit it.

### Synthesis Rules
- A claim about entity E must cite:
  1. Chunks with E's canonical name or global variant (any collection), OR
  2. Venona/Vassiliev chunks with E's alias IF mapping is at least provisional
- Quote aliases with annotations: "PAL (Person Name)" when mapping is established

## Output Format
Schema-enforced JSON with discriminator "final".
When final=false: investigation mode (tool calls, scratchpad updates).
When final=true: synthesis with cited claims.

### Alias Permission Workflow
To use codename aliases in searches:
1. Call alias_index_summary_v10 (or aliases_for_entity_v10) to discover aliases
2. Call grant_alias_power_v10 to get permission for each alias+collection
3. Then use alias_boosts in search_v10 — boosts without permission are dropped

### Live Lexicon
- After every search_v10, the system:
  1. Extracts mentions from top hits (best-effort LLM for Venona/Vassiliev, deterministic others — extraction never triggers fallback)
  2. Updates the Lexicon with new mappings, hypotheses, entities
  3. Backfills Venona/Vassiliev codename aliases for newly discovered entities
- "Current Lexicon State" is maintained as structured JSON (overwritten each round).
- Use entity_id and forms[] FROM THIS JSON for boosts.
- When searching Venona/Vassiliev, use alias_backlinks as alias_boosts_scoped \
(but note: alias_backlinks are for Venona/Vassiliev ONLY).

### Strategy Hints
- A "Strategy Hint" message is maintained with explore/exploit guidance.
- In EXPLORE mode: prioritize unseen documents and collections.
- In EXPLOIT mode: deep-dive into promising documents.
- Follow the suggested actions when possible.
"""

# =============================================================================
# Stage A: Span selection prompt
# =============================================================================

V10_SPAN_SELECTION_PROMPT = """\
## Span Lattice

Below is the span lattice for the user's query. Each span represents a \
contiguous substring of the query with candidate entity/alias mappings.

{lattice_json}

## Your Task

Select the best non-overlapping set of spans that explains the query. \
Output a JSON object with:

- chosen_span_ids: list of span_ids you want to keep (non-overlapping)
- suppressed_span_ids: list of span_ids that are dominated/overlapped and should be suppressed
- entity_hypotheses: list of {{entity_id, confidence, reason}} for entities you believe are referenced
- alias_spans: list of {{span_id, activate_alias_resolution}} for spans that need alias resolution

Guidelines:
- Prefer full-name spans over partial-name subspans when a span 'dominates' another
- Keep top-2 candidate entity_ids for high-collision spans
- Any span with valid_collections != ["*"] should have activate_alias_resolution: true
- Confidence levels: "high" (exact canonical match), "medium" (alias/fuzzy), "low" (ambiguous)
"""

# =============================================================================
# Stage B-C: Retrieval prompt templates
# =============================================================================

V10_ALIAS_RESOLUTION_PROMPT = """\
## Alias Resolution Phase

The following alias spans need resolution:

{alias_spans_json}

Search within Venona/Vassiliev to find evidence that maps these aliases \
to specific entities. Look for:
- Direct equations: "CODENAME = Person Name"
- Parenthetical: "CODENAME (Person Name)"
- "identified as" patterns
- Co-mention of codename near canonical name

After searching, report your mapping hypotheses with confidence levels.
"""

V10_GLOBAL_RETRIEVAL_PROMPT = """\
## Global Retrieval Phase

Use the following entity boosts to search across all collections in scope:

{entity_boosts_json}

Search for evidence about these entities. Use their canonical names and \
known variants — NOT codename aliases (unless searching within \
Venona/Vassiliev).
"""

# =============================================================================
# Stage D: Synthesis prompt
# =============================================================================

V10_SYNTHESIS_PROMPT = """\
## Synthesis

You have gathered evidence across {n_chunks} chunks from {n_collections} \
collections. Now produce your final synthesis.

### Identity State
{identity_summary}

### Evidence Verification Rules
- A claim about entity E is supported ONLY by chunks containing:
  - E's canonical name or global variant (any collection), OR
  - E's alias/codename in a Venona/Vassiliev chunk WHERE the alias mapping \
    is at least provisional for that specific document
- Annotate quoted aliases: "PAL (Person Name)" when mapping is established
- Flag any remaining alias ambiguity explicitly

Set final=true and produce your synthesis.
"""

# =============================================================================
# Structured output schema for span selection (Stage A)
# =============================================================================

V10_SPAN_SELECTION_SCHEMA = {
    "name": "span_selection",
    "strict": True,
    "schema": {
        "type": "object",
        "required": ["chosen_span_ids", "suppressed_span_ids",
                      "entity_hypotheses", "alias_spans"],
        "additionalProperties": False,
        "properties": {
            "chosen_span_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Non-overlapping span IDs selected from the lattice",
            },
            "suppressed_span_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Dominated/overlapped span IDs explicitly suppressed",
            },
            "entity_hypotheses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["entity_id", "confidence", "reason"],
                    "additionalProperties": False,
                    "properties": {
                        "entity_id": {"type": "integer"},
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "reason": {"type": "string"},
                    },
                },
                "description": "Entity hypotheses with confidence levels",
            },
            "alias_spans": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["span_id", "activate_alias_resolution"],
                    "additionalProperties": False,
                    "properties": {
                        "span_id": {"type": "string"},
                        "activate_alias_resolution": {"type": "boolean"},
                    },
                },
                "description": "Alias-typed spans that may need resolution",
            },
        },
    },
}

# =============================================================================
# V10 tool definitions (OpenAI function calling format)
# =============================================================================

V10_TOOLS_DEF = [
    {
        "type": "function",
        "function": {
            "name": "search_v10",
            "description": "Search with entity_boosts (global) or alias_boosts (Venona/Vassiliev). Returns chunks + provenance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Base search query text",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max results (5-30, default 15). Higher for broad explore, lower for targeted.",
                        "default": 15,
                    },
                    "entity_boosts": {
                        "type": "array",
                        "description": "Entity boosts for global search",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entity_id": {"type": "integer"},
                                "forms": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "weight": {"type": "number", "default": 1.0},
                            },
                            "required": ["entity_id", "forms"],
                        },
                    },
                    "alias_boosts": {
                        "type": "array",
                        "description": "Scoped alias boosts (Venona/Vassiliev only)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "collection_slug": {"type": "string"},
                                "alias_text": {"type": "string"},
                                "locked_entity_id": {
                                    "type": ["integer", "null"],
                                    "description": "Set ONLY when contextual rule/hypothesis is confirmed",
                                },
                                "weight": {"type": "number", "default": 1.0},
                            },
                            "required": ["collection_slug", "alias_text"],
                        },
                    },
                    "scope_collections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Restrict to these collections (optional)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_chunks",
            "description": "Fetch full text by chunk_ids. Returns complete text + metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of chunk IDs to fetch",
                    },
                },
                "required": ["chunk_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_mentions",
            "description": "Extract mentions from a chunk. Aliases only in Venona/Vassiliev.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "integer",
                        "description": "The chunk ID to extract mentions from",
                    },
                },
                "required": ["chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_referent_v10",
            "description": "Resolve surface to entity candidates. Use when span is unresolved/ambiguous.",
            "parameters": {
                "type": "object",
                "properties": {
                    "surface_text": {
                        "type": "string",
                        "description": "The surface text to resolve (e.g. 'OSS', 'Silvermaster')",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["strict", "broad"],
                        "description": "strict = exact match only; broad = adds fuzzy matching",
                        "default": "strict",
                    },
                    "scope_hint": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Optional collection slugs to bias results",
                    },
                },
                "required": ["surface_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "alias_index_summary_v10",
            "description": "Alias occurrence stats: count, entities, ambiguity. Use before grant_alias_power.",
            "parameters": {
                "type": "object",
                "properties": {
                    "alias_surface": {
                        "type": "string",
                        "description": "The alias surface text (e.g. 'CABIN', 'PAL')",
                    },
                    "collections": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Filter to specific collections (e.g. ['venona'])",
                    },
                },
                "required": ["alias_surface"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "alias_index_lookup_v10",
            "description": "Look up alias occurrences with chunk_ids. Use after alias_index_summary for evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "alias_surface": {
                        "type": "string",
                        "description": "The alias surface text",
                    },
                    "collections": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Filter to specific collections",
                    },
                    "entity_id": {
                        "type": ["integer", "null"],
                        "description": "Filter to specific entity",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Max results (default 10, max 30)",
                        "default": 10,
                    },
                    "per_entity_limit": {
                        "type": "integer",
                        "description": "Max results per entity (default 5)",
                        "default": 5,
                    },
                },
                "required": ["alias_surface"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aliases_for_entity_v10",
            "description": "Entity → alias surfaces. Discover codenames before alias-scoped search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "integer",
                        "description": "The entity ID to look up aliases for",
                    },
                    "collections": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Filter scoped aliases to these collections (default: venona, vassiliev)",
                    },
                },
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "alias_index_sample_v10",
            "description": "Balanced sample of alias occurrences across entities. For ambiguous aliases.",
            "parameters": {
                "type": "object",
                "properties": {
                    "alias_surface": {
                        "type": "string",
                        "description": "The alias surface text",
                    },
                    "scope": {
                        "type": ["string", "null"],
                        "description": "Collection slug to scope (e.g. 'venona')",
                    },
                    "per_entity_limit": {
                        "type": "integer",
                        "description": "Max samples per entity (default 3)",
                        "default": 3,
                    },
                },
                "required": ["alias_surface"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grant_alias_power_v10",
            "description": "Grant permission for alias_boosts. REQUIRED before alias_boosts in search_v10.",
            "parameters": {
                "type": "object",
                "properties": {
                    "alias_surface": {
                        "type": "string",
                        "description": "The codename alias text (e.g. 'CABIN')",
                    },
                    "entity_id": {
                        "type": ["integer", "null"],
                        "description": "Entity ID to grant for (null = boost-only, no lock)",
                    },
                    "collection_scope": {
                        "type": "string",
                        "description": "Collection slug (must be alias-scoped, e.g. 'venona')",
                    },
                },
                "required": ["alias_surface", "collection_scope"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lexicon_state_v10",
            "description": "Get full lexicon state (entities, aliases, hypotheses). Call when you need detailed identity data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "detail": {
                        "type": "string",
                        "enum": ["compact", "full"],
                        "description": "compact=default summary; full=complete JSON",
                        "default": "full",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "surface_top_referent_v10",
            "description": "Top referent for a surface. entity_id, share, is_ambiguous.",
            "parameters": {
                "type": "object",
                "properties": {
                    "surface": {
                        "type": "string",
                        "description": "The surface text to check (e.g. 'CABIN', 'OSS')",
                    },
                    "collections": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Filter to specific collections",
                    },
                    "min_share": {
                        "type": "number",
                        "description": "Minimum share to consider unambiguous (default 0.7)",
                        "default": 0.7,
                    },
                },
                "required": ["surface"],
            },
        },
    },
]

# =============================================================================
# V10 main output schema (investigation loop)
# =============================================================================

V10_OUTPUT_SCHEMA = {
    "name": "v10_output",
    "strict": True,
    "schema": {
        "type": "object",
        "required": ["final", "scratchpad_update", "synthesis"],
        "additionalProperties": False,
        "properties": {
            "final": {
                "type": "boolean",
                "description": "true = synthesis mode, false = investigation mode",
            },
            "scratchpad_update": {
                "type": ["object", "null"],
                "description": "Investigation state updates (V9+V10 merged schema)",
                "additionalProperties": False,
                "required": [
                    "goal", "leads", "hypotheses", "gaps",
                    "next_actions", "ready_to_synthesize",
                    "pin_suggestions",
                    "alias_mappings", "promotion_actions", "notes",
                ],
                "properties": {
                    # V9 investigation fields
                    "goal": {
                        "type": "string",
                        "description": "What 'done' looks like for this investigation",
                    },
                    "leads": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Promising follow-up searches or directions",
                    },
                    "hypotheses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Working theories about the answer",
                    },
                    "gaps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Questions still unanswered (keep populated until justified)",
                    },
                    "next_actions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Planned tool calls for next round",
                    },
                    "ready_to_synthesize": {
                        "type": "boolean",
                        "description": "true only when gaps empty or avenues exhausted",
                    },
                    "pin_suggestions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Bullet IDs (B:xxx) to pin as key findings",
                    },
                    # V9 notes field
                    "notes": {
                        "type": "string",
                        "description": "Free-form investigation notes for this round",
                    },
                    # V10 identity fields (kept)
                    "alias_mappings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["alias", "entity_name", "confidence", "document_context"],
                            "properties": {
                                "alias": {"type": "string"},
                                "entity_name": {"type": "string"},
                                "confidence": {"type": "string"},
                                "document_context": {"type": "string"},
                            },
                        },
                    },
                    "promotion_actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["alias_text", "collection_slug", "new_status", "entity_id", "reason"],
                            "properties": {
                                "alias_text": {"type": "string"},
                                "collection_slug": {"type": "string"},
                                "new_status": {"type": "string"},
                                "entity_id": {"type": "integer"},
                                "reason": {"type": "string"},
                            },
                        },
                    },
                },
            },
            "synthesis": {
                "type": ["object", "null"],
                "description": "Final synthesis (when final=true)",
                "additionalProperties": False,
                "required": ["answer", "claims", "unresolved_aliases", "sufficiency"],
                "properties": {
                    "answer": {"type": "string"},
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["text", "entity_ids", "evidence_chunk_ids", "confidence"],
                            "properties": {
                                "text": {"type": "string"},
                                "entity_ids": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                },
                                "evidence_chunk_ids": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                },
                                "confidence": {"type": "string"},
                            },
                        },
                    },
                    "unresolved_aliases": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["alias", "possible_entities"],
                            "properties": {
                                "alias": {"type": "string"},
                                "possible_entities": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    },
                    "sufficiency": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["is_sufficient", "remaining_gaps"],
                        "properties": {
                            "is_sufficient": {"type": "boolean"},
                            "remaining_gaps": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
}

V10_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": V10_OUTPUT_SCHEMA,
}


# =============================================================================
# Live Lexicon Enrichment — extraction constants + prompts
# =============================================================================

BLOCKED_ALIAS_LIKE = [
    "FBI", "CIA", "NSA", "KGB", "GRU", "NKVD", "MGB", "SVR",
    "USA", "USSR", "UK", "US", "OK", "MR", "MRS", "DR", "JR", "SR",
    "TOP", "SECRET", "CLASSIFIED", "DECLASSIFIED", "RE", "FW", "CC", "BCC",
]
"""All-caps words that should never be labelled as alias_surface."""

SURFACE_RATIONALE = [
    "all_caps_codename",
    "known_entity_name",
    "new_proper_name",
    "contextual_alias_usage",
    "abbreviation_as_name",
    "signal_component",
]
"""Categorical rationale enum for LLM extractor (no free text)."""

EXTRACTOR_VERSION = "v10.2"
"""Bump when changing extraction prompt/schema to invalidate cache."""

V10_EXTRACTION_PROMPT = """\
You are extracting entity and alias mentions from a document chunk.
Your output must contain ONLY verbatim spans from the text -- text must exactly
equal chunk_text[start:end]. Do not paraphrase or normalize.
Offsets: 0-based character indices. start is inclusive, end is exclusive.
So the first character is at start=0; "ab" is start=0, end=2. chunk_text[start:end] must
exactly equal the "text" field you return.

Collection: {collection_slug} ({collection_description})
Document: {document_id}, Page: {page_no}

Known entities: {known_entities}
Known aliases in this collection: {known_aliases}
Blocked alias-like words (never label as alias_surface): {blocked_alias_like}

Extract:
1. entity_surface: occurrences of known entity names or new proper names
2. alias_surface: ONLY in alias-scoped collections. All-caps words appearing to be \
codenames. Never flag words in the blocked list as alias_surface.
3. signals: patterns like "X = Y", "X (Y)", "identified as", "also known as"

For each surface, provide a rationale from: all_caps_codename, known_entity_name, \
new_proper_name, contextual_alias_usage, abbreviation_as_name, signal_component.

Output JSON. Do NOT resolve aliases -- just identify surfaces.
"""

V10_EXTRACTION_SCHEMA = {
    "name": "v10_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "required": ["surfaces", "signals"],
        "additionalProperties": False,
        "properties": {
            "surfaces": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["text", "start", "end", "kind", "confidence", "rationale"],
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "start": {"type": "integer"},
                        "end": {"type": "integer"},
                        "kind": {
                            "type": "string",
                            "enum": ["entity_surface", "alias_surface"],
                        },
                        "confidence": {"type": "number"},
                        "rationale": {
                            "type": "string",
                            "enum": SURFACE_RATIONALE,
                        },
                    },
                },
            },
            "signals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["type", "alias", "entity_name", "text", "confidence"],
                    "additionalProperties": False,
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "alias_equation", "aka", "identified_as",
                                "parenthetical", "cryptonym_marker",
                            ],
                        },
                        "alias": {"type": "string"},
                        "entity_name": {"type": "string"},
                        "text": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                },
            },
        },
    },
}
