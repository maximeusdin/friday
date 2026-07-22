# Alias Cleanup: Migrations vs Python Script

## What the new script does (reversible, JSON-logged)

1. **Case-only dedup**: `alias` vs `ALIAS` — same text, only case differs. Keep one (canonical form), delete the other.
2. **Word-order dedup**: `word1 word2` vs `word2 word1` — same words, only order differs (after stripping punctuation, lowercasing). Keep one, delete the other.

## Comparison with migrations 0065 and 0066

| Aspect | Migration 0065 | Migration 0066 | New Python script |
|--------|----------------|----------------|-------------------|
| **Purpose** | Remove garbage aliases | Remove phrase-equivalent aliases | Case-only + word-order dedup |
| **Case-only** | No | No (comment: "already deduped") | Yes |
| **Word-order** | No | Yes (via `to_tsvector` lexemes) | Yes (sorted words, no stopwords) |
| **Garbage** | Yes (american, bureau, venona, etc.) | No | No |
| **Reversible** | No | No | Yes (JSON log) |
| **Scope** | All entity_aliases | Per-entity, compares to canonical | Per-entity, within same entity |
| **Canonical** | N/A | Keeps alias matching canonical_name | Keeps one per group (prefer canonical) |

**0065** removed single-word aliases and those containing substrings (venona, vassiliev, etc.). That deleted useful aliases like "sound" when the alias text contained "Venona". The new script does not do this.

**0066** removed phrase-equivalent aliases (e.g. "office strategic services" when canonical is "Office of Strategic Services") using PostgreSQL `to_tsvector`. The new script matches this with a simpler `phrase_key` (sorted words, dropping stopwords). It is reversible and logged.

## Migration deletion

Migrations 0065 and 0066 are removed. All alias cleanup is now done by the Python script with JSON logging for reversibility.
