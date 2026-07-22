# Linking a Chunk to Document ID and Page

This document describes how to get the **document id** and **page** (and page label/number) for any **chunk** in the database.

---

## Schema overview

Relevant tables:

| Table | Role |
|-------|------|
| **chunks** | Retrieval unit: `id`, `text`, `embedding`, `pipeline_version`. No document/page columns. |
| **chunk_metadata** | One row per `(chunk_id, pipeline_version)`: **document_id**, **first_page_id**, **last_page_id**, collection_slug, dates, etc. |
| **chunk_pages** | Many-to-many: which pages a chunk spans; `(chunk_id, page_id, span_order)` (span_order = 1 is first page). |
| **pages** | `id`, **document_id**, `logical_page_label`, `pdf_page_number`, `page_seq`, `raw_text`, ... |
| **documents** | `id`, `collection_id`, `source_name`, ... |

So a chunk is linked to document and page in two ways: via **chunk_metadata** (denormalized) and via **chunk_pages** → **pages**.

---

## 1. Chunk → document and page (canonical: chunk_metadata)

**chunk_metadata** is the main place that stores “this chunk belongs to this document and this page range.”

```sql
-- Given chunk_id, get document_id and first page id
SELECT
  cm.chunk_id,
  cm.document_id,
  cm.first_page_id,
  cm.last_page_id
FROM chunk_metadata cm
WHERE cm.chunk_id = :chunk_id
  AND cm.pipeline_version = (SELECT pipeline_version FROM chunks WHERE id = :chunk_id);
```

To get **page number** and **logical page label**, join to **pages**:

```sql
-- Chunk → document_id + first page number and label
SELECT
  cm.chunk_id,
  cm.document_id,
  p.id AS page_id,
  p.logical_page_label,
  p.pdf_page_number,
  p.page_seq
FROM chunk_metadata cm
JOIN pages p ON p.id = cm.first_page_id
WHERE cm.chunk_id = :chunk_id
  AND cm.pipeline_version = (SELECT pipeline_version FROM chunks c WHERE c.id = cm.chunk_id);
```

If you use the **retrieval_chunks_current** view, document and page ids are already there (no extra join for page number/label):

```sql
SELECT
  chunk_id,
  document_id,
  first_page_id,
  last_page_id
FROM retrieval_chunks_current
WHERE chunk_id = :chunk_id;
```

Then join **pages** on `first_page_id` when you need `logical_page_label` or `pdf_page_number`.

---

## 2. Chunk → document and page (via chunk_pages)

Chunks can span multiple pages. **chunk_pages** lists all pages for a chunk in order; the first page is `span_order = 1`.

```sql
-- Chunk → all pages (document_id, page label, page number)
SELECT
  c.id AS chunk_id,
  p.document_id,
  p.id AS page_id,
  cp.span_order,
  p.logical_page_label,
  p.pdf_page_number
FROM chunks c
JOIN chunk_pages cp ON cp.chunk_id = c.id
JOIN pages p ON p.id = cp.page_id
WHERE c.id = :chunk_id
ORDER BY cp.span_order;
```

The first row (`span_order = 1`) is the “first page” of the chunk; **chunk_metadata.first_page_id** should match that page’s `id`.

---

## 3. From entity_mentions (chunk + document + page)

**entity_mentions** denormalizes chunk and document (and in production/export often page). So if you have a mention, you already have chunk → document; page may be in the same row or via chunk_metadata/pages.

```sql
-- All (chunk_id, document_id) from mentions; for page use chunk_metadata or chunk_pages
SELECT DISTINCT
  em.chunk_id,
  em.document_id
FROM entity_mentions em
WHERE em.chunk_id = :chunk_id;
```

To attach the first page for each chunk from metadata:

```sql
SELECT
  em.chunk_id,
  em.document_id,
  cm.first_page_id,
  p.logical_page_label,
  p.pdf_page_number
FROM entity_mentions em
JOIN chunk_metadata cm ON cm.chunk_id = em.chunk_id
JOIN pages p ON p.id = cm.first_page_id
WHERE em.chunk_id = :chunk_id
LIMIT 1;
```

---

## 4. Concordance export (CSV) vs live DB

In the **concordance_export** CSVs:

- **concordance_entries**: index entries (id, entry_key, raw_text, …). These are “concordance entries,” not chunks.
- **entity_mentions** (export): has **chunk_id**, **document_id**, **document_name**, **first_page_id**, **pdf_page_number**, **logical_page_label**, and **concordance_entry_key**.

So in the export, each mention row already gives you **chunk_id → document_id + page** (first_page_id, pdf_page_number, logical_page_label). In the **live database**, **entity_mentions** has `chunk_id` and `document_id`; for page you use **chunk_metadata** (first_page_id) + **pages** as above.

---

## Summary

| Goal | Where to look |
|------|----------------|
| Chunk → **document_id** | **chunk_metadata.document_id** (or entity_mentions.document_id) |
| Chunk → **page** (id) | **chunk_metadata.first_page_id** (and last_page_id for range) |
| Chunk → **page number / label** | **pages** joined on chunk_metadata.first_page_id (or chunk_pages.page_id with span_order = 1) |
| Chunk → all pages it spans | **chunk_pages** → **pages**, ordered by span_order |

**Recommended:** For “chunk → document id and page it came from,” use **chunk_metadata** plus **pages** on **first_page_id** (and **last_page_id** if you need the full page range).
