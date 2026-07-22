Problem

The v5 agentic workflow repeats the same tool calls across rounds (e.g., hybrid_search / entity_mentions with identical params), fails to use Round 1 results to inform Round 2, and still risks “blob → claim explosion” when retrieval returns many chunks. Even with enforced collection scope, we see:

Tool-loop repetition due to lack of pagination/novelty controls

Model treating control/instruction tokens as topical entities (e.g., “Provide” entity-linked)

No robust way to follow strong leads like roster/table spans split across chunks

Need better researcher-grade output: member roster backed by Vassiliev-only citations

Current “summary” can be ungrounded, and outputs can contain statements without explicit citation support.

Goals

Make Round N tool choice depend on Round N-1 outcomes (no “amnesia”).

Guarantee scope constraints are applied consistently (Vassiliev-only means Vassiliev-only).

Prevent repeated top-k results from dominating (add cursor/pagination + exclude).

Introduce bounded evidence that improves over rounds (bottleneck + tournament).

Support evidence split across chunks via Bundles (no evidence-type taxonomy).

Improve researcher trust: stop only when answer is citation-backed from scoped corpus.

Add an expanded summary answer that enumerates all claims made and forces each claim to have ≥1 citation.

Proposed Changes
A) CONTROL vs CONTENT parsing (stop dumb entity linking)

Add a preprocessing step that partitions the user input into:

content_terms (topic terms for retrieval/entity linking)

control_terms (scope/output instructions: “provide citations”, “Vassiliev notebooks”, etc.)

scope_constraints (collections include/exclude)

Enforcement

Only content_terms are eligible for entity linking + query term expansion.

control_terms must never generate entity IDs for retrieval.

Acceptance Criteria

Query “Provide citations from the Vassiliev notebooks” does not produce entity IDs like “Provide → random entity”.

“Vassiliev notebooks” is treated as scope constraint, not a topical person/entity seed.

B) Make novelty possible: pagination + exclude_seen on retrieval tools

The model repeats calls because it can’t ask for “next results” and keeps hitting the same top-10.

Add to these tools

hybrid_search(query, top_k, collections, cursor=None, exclude_chunk_ids=None)

lexical_search(...) (if applicable)

entity_mentions(entity_id, top_k, cursor=None, exclude_chunk_ids=None)

co_mention_entities(entity_id, top_k, cursor=None, exclude_entity_ids=None)

Executor behavior

Maintain seen_chunk_ids and inject exclude_chunk_ids by default (configurable).

Allow the LLM to request cursor explicitly; if it repeats same params without cursor/exclude, reject and suggest cursor/exclude.

Acceptance Criteria

Round 2 can retrieve new results for the same query without changing the query string.

Duplicate tool calls become rare because novelty can be requested explicitly.

C) Evidence Bottleneck becomes mandatory input to synthesis (no more blob)

Hard gate: Retrieval results never flow directly into claim extraction / answer generation.
Only the bounded EvidenceStore (spans or bundles) is allowed as synthesis context.

Acceptance Criteria

Claim extraction / renderer never sees raw chunks outside the store.

Large retrieval (100–300 chunks) produces bounded store (e.g., ≤ 40 spans or ≤ 20 bundles).

D) Introduce Bundles (to handle split evidence without evidence-type taxonomy)

Evidence is often split across chunks (roster tables, continuation lines, pronoun referents).

Add a bundle-builder step (LLM-generated) that groups spans into atomic citeable bundles:

bundle_id

span_ids[] (max N spans per bundle, e.g., 6)

bundle_claim (1–2 sentence description of what the bundle supports)

confidence

Tournament selection operates on bundles (preferred) or spans.

Acceptance Criteria

Bundle builder groups “list continues” style evidence into a single citeable unit.

Roster answers can cite bundle IDs and expand to underlying spans in UI/export.

E) Add “Neighbor/Continuation” retrieval affordance (follow strong leads)

When the store includes roster/table spans, Round 2 should expand nearby context instead of repeating global search.

Add one minimal tool

chunk_neighbors(chunk_id, window=2) or

page_window(document_id, page_num, window_pages=1) or

document_cursor(document_id, after_chunk_id, limit=K)

(Choose the implementation that matches current schema best.)

Acceptance Criteria

Given a high-value roster span, the agent can pull adjacent chunks/pages to capture the remainder of the list.

Member discovery increases substantially without broad re-search.

F) RoundSummary: LLM-generated, schema-defined, used as decision state

Create a short per-round artifact used to drive next-round tool choice. This should be LLM-generated but schema-enforced.

RoundSummary JSON schema (general)

round_outcome: "progress" | "stalled" | "regressed"

new_facts: string[]

open_questions: string[]

best_evidence_refs: string[] (span_ids or bundle_ids)

next_actions: [{tool, params}] (1–3 candidates)

avoid_repeats: string[] (tool+param fingerprints)

Acceptance Criteria

Round 2 uses RoundSummary to choose a different action than Round 1 (e.g., paginate, exclude seen, neighbor fetch).

Tool repetition rate drops drastically on multi-round runs.

G) Stop gate: “Answer must be supported by store-backed citations”

For researchers, the agent must not “answer” unless it can cite evidence from the store.

Rule

STOP_ANSWER accepted only if every substantive statement has citations to active spans/bundles in the store and (optionally) passes entailment verification.

Acceptance Criteria

Roster queries produce a list of people with citations (Vassiliev-only if scoped).

If insufficient evidence in scope, agent returns “insufficient” with best evidence, not a speculative roster.

H) Expanded Summary Output: enumerate all claims + require ≥1 citation per claim

Add an additional final product: an expanded summary answer intended for researchers and debugging, which:

Lists every claim the system makes in the final response (including derived/bridging statements if desired)

Attaches at least one citation (span_id or bundle_id) to each claim

Fails closed: if a claim cannot be supported by at least one citation, it must be:

removed, or

rewritten into a supported claim, or

explicitly labeled as “unsupported / not found” and excluded from the “claims made” list

Implementation Details

Claim enumeration pass (LLM)

Input: final answer draft + evidence store (bundles/spans)

Output: ClaimList[] where each item includes:

claim_text (atomic, single assertion)

citations[] (≥1 required)

support_level (e.g., strong/weak) (optional)

Claim-citation gate

Validate mechanically that each ClaimList.citations is non-empty and references active evidence.

If any claim is missing citations:

reject the draft and trigger a repair loop (limited budget), or

auto-prune unsupported claims and regenerate the expanded summary.

Expanded summary renderer

Output format:

A short “Answer”

Then “Claims & Citations” list (every claim bullet with citations)

Acceptance Criteria

Expanded summary contains 0 claims without citations.

“Claims & Citations” list is stable/parseable (JSON stored; human-readable view rendered).

For roster queries, each listed member is a claim, and each member has ≥1 citation.

For “insufficient evidence” outcomes, claims are limited to what the evidence supports (e.g., “Vassiliev mentions X in connection with…”), each with citations.

Implementation Plan (Suggested Order)

CONTROL vs CONTENT parse + scope enforcement wiring

Pagination + exclude_seen for hybrid_search / entity_mentions (+ co_mention if possible)

EvidenceStore hard gate (synthesis reads only store)

RoundSummary schema + integrate into agent loop

Neighbor/continuation tool

Bundles (builder + store + tournament on bundles)

Stop gate tightening (store-backed citations only)

Expanded Summary: claim enumeration + claim-citation gate + renderer

Test Plan / Acceptance Tests

Regression: “Provide citations from Vassiliev notebooks” no longer links “Provide” as an entity seed.

Novelty: Running same query over 2 rounds yields new chunks via cursor/exclude_seen.

Scoped roster: “Who were members of the Silvermaster network? Vassiliev only.” returns members with only Vassiliev citations.

No repetition: Round 2 does not spam identical tool calls; if it tries, it uses cursor/exclude or neighbor tool.

Split evidence: A roster table split across chunks is combined into a bundle and cited as one unit.

Stop safety: System refuses to stop with an answer if roster items/claims lack citations.

Expanded Summary: Every claim in the expanded summary has ≥1 citation; unsupported claims are not emitted.

Fail-safe: If Vassiliev lacks explicit member roster, system stops with “insufficient evidence in Vassiliev-only scope” + best citations, and expanded summary only includes supported claims.