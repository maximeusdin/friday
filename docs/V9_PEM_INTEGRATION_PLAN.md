# V9 + PEM Lane: End-to-End Implementation Plan (Deploy-Safe)

## Outcome you want

- **V9 keeps its "always produce bullets / useful output" behavior.**
- **PEM lane guarantees codename/alias recall** even when vector would miss it (e.g., CABIN pages that never say "OSS", or Golos referenced only by alias).
- **Model sees page-scoped mappings in evidence** so it can interpret aliases correctly in context.
- **Multi-entity prioritization:** When multiple entities are resolved from the query, prioritize chunks that mention multiple of these entities (co-relation chunks), while maintaining diversity across surfaces and documents.

---

## 0) Preconditions and definitions

### Alias-scoped corpora

```python
ALIAS_SCOPED = {"venona", "vassiliev"}  # extend later if needed
```

### PEM (page_entity_mentions) contract

You already have:

- `page_entity_mentions(page_id, surface_norm, surface_raw, entity_id, truth_level, source, collection_slug, document_id)`
- `chunk_pages(chunk_id, page_id, span_order)`
- `entities(id, canonical_name)`

---

## 1) Integration points in V9

Two hooks:

| Hook | Location | Purpose |
|------|----------|---------|
| **Hook A: Retrieval assembly (PEM lane)** | Where V9 assembles evidence chunks (vector/lexical + PEM seed) | Add `pem_seed_chunks = pem_lane_seed_chunks(...)` and merge into workspace catalog |
| **Hook B: Evidence formatting (mapping injection)** | Where V9 formats chunk text for model context (`v9_context.py`, fulltext chunk rendering) | Append `pem_mapping_block = build_pem_mapping_block_for_chunk(...)` to chunk text |

Everything else in V9 remains intact.

---

## 2) Pipeline overview (end-to-end)

### Step 2.1 Query → candidate surfaces (cheap, deterministic)

**Goal:** Collect a small set of "surface hypotheses" from the query without any LLM extraction.

**Candidate generation:**

1. Tokenize query
2. Include:
   - Any token in quotes
   - Any token that is ALL CAPS (length ≥ 2)
   - Any short token (≤ 6) (optional; helps PAL etc.)
   - Optionally generate ngrams (2–4) but only keep them if they exist in PEM (fast exists check)

**Normalize:**

- `surface_norm = normalize_surface_for_lookup(token)` (v10 normalizer or equivalent)

**Prune using PEM exists check:**

Only keep `surface_norm` where:

```sql
SELECT 1
FROM page_entity_mentions
WHERE surface_norm = :s
  AND collection_slug = ANY(:alias_scoped_in_scope)
LIMIT 1;
```

This keeps candidate list tight.

---

### Step 2.2 Surfaces → entity candidates (direct from PEM)

For each candidate `surface_norm = s`, query distribution:

```sql
SELECT entity_id, COUNT(DISTINCT page_id) AS count_pages
FROM page_entity_mentions
WHERE surface_norm = :s
  AND collection_slug = ANY(:alias_scoped_in_scope)
GROUP BY entity_id
ORDER BY count_pages DESC, entity_id ASC;
```

**Pick entity candidates per surface:**

- Keep top 1 always
- Keep top 2 if ambiguity (e.g., top_share < 0.8 or second is close)

This solves "query may contain an alias": if query contains CABIN, this yields entity_id=OSS directly.

---

### Step 2.3 Seed entities-in-play

Create a seed set of entity_ids from:

- Entity candidates from Step 2.2
- Plus anything V9 already resolved (merged from `workspace.entities` and `workspace.entity_candidates`)

For key use cases you'll usually end with:

- `entity_ids = {OSS_entity_id, Golos_entity_id}` (or just OSS if that's all you can infer)

---

### Step 2.4 Entity → alias surfaces (from PEM, alias-scoped)

For each `entity_id = e`, pull top surfaces in alias-scoped corpora:

```sql
SELECT surface_norm, COUNT(DISTINCT page_id) AS count_pages
FROM page_entity_mentions
WHERE entity_id = :e
  AND collection_slug = ANY(:alias_scoped_in_scope)
GROUP BY surface_norm
ORDER BY count_pages DESC, surface_norm ASC
LIMIT :surface_pool_limit;
```

**Filter out canonical-ish surfaces:**

- Exclude `surface_norm == normalize_surface_for_lookup(entity.canonical_name)`
- Optionally exclude the literal acronym if it's the query surface (e.g., exclude "oss" when query contained OSS) to force codename diversity

**Keep up to:** `MAX_ALIAS_SURFACES_PER_ENTITY = 8` (safe)

**Guarantee representation across aliases:** This list is what you'll allocate quotas across.

---

### Step 2.5 Forced retrieval: (entity, alias surface) → pages → chunks (coverage quotas)

This is the PEM lane "guarantee step."

#### 2.5a Multi-entity prioritization (when |entities_in_play| ≥ 2)

**Design goal:** Prioritize pages with ≥2 entities in play, but within a quota and still preserve surface diversity. Avoid exploding the page pool with generic co-mention pages.

**Caveat:** The simple query below can accidentally prioritize pages that contain any two entities in play even if they're extremely common (e.g., OSS + USSR), yielding lots of generic pages.

**Two simple improvements (still deterministic):**

1. **Query-derived filter:** Require that at least one of the entities came from **query-derived surfaces** (Step 2.2), not just workspace backfill. Only run multi-entity prioritization when `query_derived_entity_ids ∩ entities_in_play` is non-empty.
2. **Entity-kind filter (if available):** If you have entity kinds, require at least one "person" entity among the entities in play—avoids surfacing only org+org co-mentions.

```sql
-- Pages that have mentions of 2+ entities in play
SELECT page_id, COUNT(DISTINCT entity_id) AS entity_count
FROM page_entity_mentions
WHERE entity_id = ANY(:entity_ids)
  AND collection_slug = ANY(:alias_scoped_in_scope)
GROUP BY page_id
HAVING COUNT(DISTINCT entity_id) >= 2
ORDER BY entity_count DESC, page_id ASC
LIMIT :multi_entity_page_limit;
```

**Allocation:**

- Reserve `MULTI_ENTITY_CHUNK_QUOTA = 8` (or ~30% of MAX_PEM_SEED_CHUNKS) for multi-entity chunks first
- Add these pages to the page pool and convert to chunks (via `chunk_pages`), applying per-page cap
- These chunks are **prepended** to the PEM seed list so they appear first in the "Alias-Scoped Evidence" section

**Diversity within multi-entity:**

- When selecting multi-entity pages, prefer variety: cap pages per document (e.g., MAX 2 per doc)
- Within the multi-entity quota, round-robin across documents to avoid one doc dominating

#### 2.5b Coverage pass (guarantee each alias represented)

For each alias surface `a` in the selected list for entity `e`:

Pull `PAGES_PER_ALIAS = 1` (or 2) pages:

```sql
SELECT document_id, page_id
FROM page_entity_mentions
WHERE entity_id = :e
  AND surface_norm = :a
  AND collection_slug = ANY(:alias_scoped_in_scope)
ORDER BY document_id ASC, page_id ASC
LIMIT :pages_per_alias;
```

Collect page_ids (dedupe). **Exclude pages already selected in 2.5a** to avoid double-counting.

#### 2.5c Optional depth pass (spend remaining budget)

If you still want more, allocate extra pages proportional to count_pages for each alias surface.

#### 2.5d Convert pages → chunks (citeable, deterministic)

Use chunk_pages:

```sql
SELECT page_id, chunk_id, span_order
FROM chunk_pages
WHERE page_id = ANY(:page_ids)
ORDER BY page_id ASC, span_order ASC, chunk_id ASC;
```

Then:

- Per-page cap: `MAX_CHUNKS_PER_PAGE = 2–3`
- Global cap: `MAX_PEM_SEED_CHUNKS = 20–30`

**Final ordering:** Multi-entity chunks first (up to quota), then coverage/depth chunks (round-robin for diversity).

Return `pem_seed_chunk_ids`.

This catches the critical failure mode: pages that contain only Golos-by-alias and never the string "Golos", **and** pages that discuss both OSS and Golos (relationship evidence).

---

## 3) Merge PEM lane with V9's existing retrieval

Run V9's normal retrieval as usual (vector/lexical/hybrid), get `base_chunk_ids`.

Then merge deterministically:

**Preferred merge strategy: two-section evidence (least distortion)**

In the model context, present:

1. **Primary Evidence:** top K normal hits
2. **Alias-Scoped Evidence (Mention Index Seed):** top N PEM seed hits

**Prompt hint (optional but worth it):** Add one line in the context header for the Alias-Scoped section:

> "Alias-Scoped Evidence is seeded from the mention index; use it when interpreting codenames and aliases."

This isn't a guardrail—just a label that prevents the model from ignoring the second section.

This prevents PEM seeds from "overruling" relevance ordering, while still guaranteeing visibility.

**Suggested:**

- `K_primary = 12–20`
- `N_pem = 8–15` (not all 30)

**Alternative:** Interleave (2 primary, 1 PEM, repeat) — but two-section is simplest and safest.

---

## 4) Evidence mapping injection (updated with heuristic)

Include mappings if the surface is quoted or ALL CAPS (hardcoded).

**Invariant:** Never inject ambiguous mappings. Caps/quotes only make a mapping **eligible**; they do not override unambiguity.

### 4.1 Compute PEM facts for a chunk

**Performance optimization (cache PEM facts per page):**

`build_pem_mapping_block_for_chunk()` can become expensive if it hits PEM per chunk. Instead:

- During PEM lane selection you already fetch pages → **bulk fetch PEM facts for the whole page pool once**
- Store `{page_id: [pem_rows...]}` in an in-memory dict for this run
- Mapping builder just looks up pages in that dict
- Same for `entities.id → canonical_name`: bulk fetch once per run

**Per-chunk fallback (when cache miss):**

Given `chunk_id`:

1. Get chunk page_ids:

```sql
SELECT page_id
FROM chunk_pages
WHERE chunk_id = :chunk_id
ORDER BY span_order ASC;
```

2. Pull PEM facts on those pages (or look up in cache):

```sql
SELECT surface_norm, surface_raw, entity_id, truth_level, source, collection_slug
FROM page_entity_mentions
WHERE page_id = ANY(:page_ids)
  AND collection_slug = ANY(:alias_scoped_in_scope);
```

3. Group by surface_norm and collect distinct entity_ids across those pages.

### 4.2 Decide which mapping lines to include (safe + capped)

Include a surface mapping line if **any** of the following is true:

| Rule | Condition | Purpose |
|------|-----------|---------|
| **A (unambiguous)** | `distinct_entity_ids == 1` for that surface across chunk pages | Safe baseline |
| **B (hardcoded heuristic)** | Surface appears in chunk text in **ALL CAPS** or **inside quotes** | Include if quoted or all caps |
| **C (question-relevant)** | Mapping's entity_id is one of `entities_in_play` | OSS, Golos etc. always get mappings |

**Safe constraint:** For B and C, still require local unambiguity (single entity_id across chunk pages). Caps/quotes only make a mapping eligible; they do not override unambiguity.

**Rule B implementation details:**

- Use the **raw chunk text** (not normalized) for the presence check
- Check both `surface_norm` and `surface_raw` variants (PEM stores raw and norm)
- Use a **simple regex boundary match** so you don't match "CABIN" inside "CABINET" (e.g., `\bCABIN\b` or word-boundary equivalent)

### 4.3 Cap and order deterministically

- Cap `MAX_MAPPING_LINES = 8–12`
- Sort by: source/truth_level rank (authoritative > derived), then surface_norm ASC

### 4.4 Output block (model-only)

Append to the chunk's model text:

```
[MENTION_INDEX page_scoped collection=venona]
cabin => Office of Strategic Services
<golos_alias> => Jacob Golos
[/MENTION_INDEX]
```

This makes V9 "understand" CABIN=OSS and Golos-alias=Golos immediately.

---

## 5) How V9 uses this (bullets + agentic follow-ups)

With the seeded alias chunks visible and mapping blocks injected:

- V9 will write bullets grounded in those chunks
- It can choose to: search again using discovered surfaces, narrow to alias-scoped evidence, or just answer with what it has
- You're not forcing behavior—just ensuring it sees the right material
- **Multi-entity chunks** give the model direct evidence for relationship questions (e.g., "OSS and Golos collaboration")

---

## 6) Tests (must-have, deploy confidence)

### 6.1 Fixture DB (small)

Create a test fixture with:

- **entities:** OSS (id=100), Golos (id=200)
- **PEM rows:**
  - page 10: cabin → OSS
  - page 11: pal → OSS
  - page 12: lib → OSS
  - page 13: another_alias → OSS
  - page 14: fifth_alias → OSS
  - (5 alias surfaces for OSS, each on a different page—for representation test)
  - pages 12–13: oss → OSS
  - page 15: oss → another_entity (to test ambiguity)
  - pages 20–21: golos_alias → Golos
  - **page 25: both cabin → OSS and golos_alias → Golos** (multi-entity page)
- **chunk_pages:**
  - page 10 → chunk 1000 (chunk text contains "CABIN" but not "OSS")
  - page 20 → chunk 2000 (chunk text contains alias but not "Golos")
  - page 25 → chunk 2500 (multi-entity chunk)
- Ensure `collection_slug` set to venona

### 6.2 Tests: query→surface→entity bootstrap

- Query: `"who were Jacob Golos's sources in the OSS?"` → candidate surface includes `oss` → PEM distribution returns OSS entity_id among top
- Query: `"sources in CABIN"` → candidate surface includes `cabin` → PEM distribution returns OSS entity_id as top

### 6.3 Tests: entity→aliases coverage quotas + representation across 5 aliases

- For entity OSS: aliases list returns 5 alias surfaces
- **Fixture:** OSS has 5 alias surfaces, each mapped on a **different page** (cabin→p10, pal→p11, lib→p12, etc.)
- **Assert:** PEM seed contains at least one chunk from each surface's page set
- **Assert:** Ordering is deterministic across runs

### 6.4 Tests: forced retrieval catches alias-only pages

- For Golos: PEM lane returns chunk 2000 even though chunk text doesn't include "Golos"
- Proves you're not dependent on vector/lexical overlap

### 6.5 Tests: mapping injection includes ALL CAPS / quoted surfaces

- Given chunk 1000 contains "CABIN" (all caps): mapping block includes `cabin => Office of Strategic Services`
- Given a chunk containing `"PAL"`: mapping block includes pal mapping if locally unambiguous
- Test that ambiguous surfaces are **not** injected (local ambiguity)

### 6.6 Tests: multi-entity prioritization

- Query: `"relationship between OSS and Golos"` → entities_in_play = {OSS, Golos}
- PEM lane returns chunk 2500 (multi-entity page) in the **top** of PEM seed, before single-entity chunks
- Diversity: still returns chunks from both entities' alias surfaces (not only multi-entity)

---

## 7) Suggested defaults (safe to start)

| Constant | Value |
|----------|-------|
| `ALIAS_SCOPED` | `{"venona","vassiliev"}` |
| `MAX_ALIAS_SURFACES_PER_ENTITY` | 8 |
| `PAGES_PER_ALIAS` | 1 |
| `MAX_CHUNKS_PER_PAGE` | 2 |
| `MAX_PEM_SEED_CHUNKS` | 25 |
| `MULTI_ENTITY_CHUNK_QUOTA` | 8 |
| `MULTI_ENTITY_PAGE_LIMIT` | 10 |
| `N_pem_shown` | 10 (in "Alias-Scoped Evidence" section) |
| `MAX_MAPPING_LINES` | 10 |

---

## 8) Why this will work (even under toughest failure modes)

| Failure mode | How PEM lane handles it |
|--------------|--------------------------|
| Query contains canonical "OSS" | PEM gives entity_id → alias seeding yields CABIN chunks |
| Query contains alias "CABIN" | PEM gives entity_id directly → still yields relevant chunks |
| Relevant pages never contain canonical strings ("Golos", "OSS") | PEM seeding still pulls them (operates on page facts, not embeddings) |
| Model needs to resolve aliases | Mapping blocks injected; include ALL CAPS / quoted surfaces |
| **Query asks about multiple entities' relationship** | Multi-entity prioritization surfaces pages that mention both (co-relation chunks) |
| **Multi-entity page pool explosion (OSS+USSR generic pages)** | Query-derived filter: only run multi-entity when ≥1 entity came from Step 2.2; optional entity-kind filter (require person) |
| **Diversity of evidence** | Round-robin per surface, per-page caps, doc caps; multi-entity quota is a slice, not the whole budget |

---

## 9) Implementation checklist

### Phase 1: V9-compatible PEM lane module

- [ ] Create `retrieval/agent/v9_pem_lane.py` (or adapt `v10_pem_lane.py` for V9)
- [ ] Implement query→surface→entity bootstrap (Steps 2.1–2.3) using workspace entities as fallback
- [ ] Implement entity→alias surfaces (Step 2.4)
- [ ] Implement forced retrieval with **multi-entity prioritization** (Step 2.5), including query-derived filter to avoid page pool explosion
- [ ] Implement `build_pem_mapping_block_for_chunk` with Rules A/B/C and ALL CAPS/quoted heuristic (Rule B: raw chunk text, surface_raw/surface_norm, boundary match)
- [ ] Bulk-fetch PEM facts per page pool and cache `{page_id: [pem_rows]}` for mapping builder; bulk-fetch entity canonical names

### Phase 2: V9 integration hooks

- [ ] **Hook A:** After `_prime_workspace_from_question`, call `pem_lane_seed_chunks(conn, workspace, scope, question)` and merge PEM chunk IDs into `workspace.catalog_hits` (with appropriate score/label)
- [ ] **Hook B:** In `v9_context.build_context_pack`, when rendering fulltext chunks, append PEM mapping block via `build_pem_mapping_block_for_chunk`
- [ ] Ensure two-section evidence presentation (Primary vs Alias-Scoped) with prompt hint: "Alias-Scoped Evidence is seeded from the mention index; use it when interpreting codenames and aliases."

### Phase 3: Tests

- [ ] Fixture DB (6.1)
- [ ] Query→surface→entity tests (6.2)
- [ ] Coverage quota + representation-across-5-aliases tests (6.3)
- [ ] Alias-only recall test (6.4)
- [ ] Mapping injection tests (6.5)
- [ ] Multi-entity prioritization test (6.6)

### Phase 4: Config and rollout

- [ ] Add feature flag `V9_PEM_LANE_ENABLED` (default 1 for venona/vassiliev scope)
- [ ] Validate on staging with Venona/Vassiliev queries
- [ ] Deploy
