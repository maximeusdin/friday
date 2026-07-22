# Entity Resolution: PEM as Sole Source

Entity name resolution (name/codename → entity_id) uses **PEM (page_entity_mentions) only** as the sole source. There is no fallback to concordance, entity_aliases, or canonical match.

## Design

- **PEM** = `page_entity_mentions` table, which maps surface forms (e.g. "PAL", "Silvermaster") to entity IDs within alias-scoped collections (venona, vassiliev).
- When a user or agent supplies a name (e.g. "PAL"), we resolve it via `resolve_keyword_via_pem` → `page_entity_mentions`.
- If PEM has no matching surface, or PEM is disabled/empty, resolution returns nothing.

## Affected Components

| Component | Role |
|-----------|------|
| `_lookup_entity_by_name` (tools.py) | Central lookup; PEM-only, no concordance fallback |
| `entity_surfaces_tool` | Resolves name via `_lookup_entity_by_name` |
| `entity_mentions_tool` | Resolves name via `_lookup_entity_by_name` |
| `entity_lookup_tool` | Resolves name via `_lookup_entity_by_name` |
| `co_mention_entities_tool` | Resolves name via `_lookup_entity_by_name` |
| `first_mention_tool` | Resolves name via `_lookup_entity_by_name` |
| `expand_entities` (v9_tools) | Resolves names via `entity_surfaces_tool` / `entity_lookup_tool` |
| `_resolve_keywords` (v9_runner) | Query priming; PEM-only (V10 spans + `_lookup_entity_by_name`) |
| `resolve_question_entities` (v9_tools) | Legacy question priming; PEM-only via `_resolve_query_entities` |

## Scope

- When `scope.collections` is empty, PEM defaults to venona+vassiliev (alias-scoped).
- When scope includes other collections, only alias-scoped collections in scope are searched.

## Rationale

- Single source of truth for entity resolution.
- Avoids drift between concordance, entity_aliases, and PEM.
- PEM is populated from concordance + rules; using it directly ensures consistency with the mention index.

## Downstream Data

- **Mention chunks**: still come from `entity_mentions` table (chunk-level extraction index).
- **Surfaces/aliases**: after resolution, `EntitySurfaceIndex` loads canonical + aliases from `entities` and `entity_aliases` for display.
