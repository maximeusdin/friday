# Research Console API Contract v1

> **Contract Version:** v1  
> **Locked:** 2026-01-31  
> **Breaking changes require version bump.**

This document defines the stable JSON shapes for the Research Console UI.
All API endpoints MUST conform to these shapes.

---

## Core Types

### EvidenceRef (canonical)

The atomic unit of citation. Links a result to a specific location in source material.

```typescript
interface EvidenceRef {
  document_id: number;           // required - FK to documents.id
  pdf_page: number;              // required - 1-based PDF page number
  chunk_id?: number;             // optional - FK to chunks.id
  span?: {                       // optional - character offsets within chunk/page
    start: number;
    end: number;
  };
  quote?: string;                // optional - extracted text snippet
  why?: string;                  // optional - explanation of relevance
}
```

**Page semantics:**
- `pdf_page` is **1-based** (matches PDF viewer conventions and human expectations)
- Maps to `pages.pdf_page_number` in the database
- The UI opens PDFs at `#page={pdf_page}`

---

### Message

A single message in a research session conversation.

```typescript
interface Message {
  id: number;
  session_id: number;
  role: "user" | "assistant" | "system";
  content: string;
  plan_id?: number;              // set when message triggered/contains a plan
  result_set_id?: number;        // set when message references results
  metadata?: Record<string, unknown>;  // extensible metadata
  created_at: string;            // ISO 8601 timestamp
}
```

---

### Plan

A structured research plan proposed by the assistant.

```typescript
interface Plan {
  id: number;
  session_id: number;
  status: "proposed" | "approved" | "executed" | "rejected" | "superseded";
  user_utterance: string;        // original user query
  plan_json: PlanJson;           // structured plan definition
  plan_summary: string;          // human-readable summary
  parent_plan_id?: number;       // if this is a revision
  retrieval_run_id?: number;     // set after execution
  result_set_id?: number;        // set after execution
  created_at: string;
  approved_at?: string;
  executed_at?: string;
}

interface PlanJson {
  primitives: Primitive[];       // array of plan primitives
  execution_envelope?: {
    top_k?: number;
    search_type?: string;
    scope_sql?: string;
  };
}

interface Primitive {
  type: string;                  // e.g., "term", "phrase", "entity", "filter_collection"
  [key: string]: unknown;        // type-specific fields
}
```

---

### Session

A research session container.

```typescript
interface Session {
  id: number;
  label: string;
  created_at: string;
  message_count?: number;        // computed, for list view
  last_activity?: string;        // computed, for list view
}
```

---

### ResultSetResponse

UI-ready representation of execution results.

```typescript
interface ResultSetResponse {
  id: number;
  name: string;
  retrieval_run_id: number;
  summary: {
    item_count: number;
    document_count: number;
    entity_count?: number;
    date_range?: {
      min?: string;
      max?: string;
    };
  };
  items: ResultItem[];
  created_at: string;
}

interface ResultItem {
  id: string;                    // stable row ID for React keys (e.g., "chunk-123")
  kind?: "chunk" | "entity" | "doc" | "note";  // result type
  rank: number;
  text: string;                  // display text / snippet
  
  // Optional identifiers (depends on kind)
  chunk_id?: number;
  document_id?: number;
  entity_id?: number;
  
  // Scoring (if available)
  scores?: {
    lex?: number;
    vec?: number;
    hybrid?: number;
  };
  
  // Highlighting
  highlight?: string;            // text with <mark> tags
  matched_terms?: string[];      // terms that matched
  
  // Evidence chain
  evidence_refs: EvidenceRef[];
}
```

---

### Document

Document metadata for evidence viewer.

```typescript
interface Document {
  id: number;
  collection_id: number;
  collection_slug?: string;
  source_name: string;           // filename
  source_ref?: string;           // file path (internal)
  volume?: string;
  page_count?: number;
  metadata?: Record<string, unknown>;
  created_at: string;
}
```

---

## API Endpoints

### Meta & Health

```
GET /api/health
Response: { "status": "ok" }

GET /api/meta
Response: {
  "contract_version": "v1",
  "api_version": "0.1.0",
  "build": "<git-sha-or-timestamp>"
}
```

---

### Sessions

```
POST /api/sessions
Request:  { "label": "string" }
Response: Session

GET /api/sessions
Response: Session[]

GET /api/sessions/:id
Response: Session

GET /api/sessions/:id/messages
Response: Message[]
```

---

### Messages (Send Message - Core Loop)

```
POST /api/sessions/:id/messages
Request: {
  "content": "string"            // user's query
}
Response: {
  "user_message": Message,
  "assistant_message": Message,
  "plan": Plan                   // the proposed plan
}
```

This is the heart of the UI loop. Sending a message:
1. Stores the user message
2. Calls the planner to generate a plan
3. Stores the assistant message (with plan_id)
4. Returns all three objects

---

### Plans

```
GET /api/plans/:id
Response: Plan

POST /api/plans/:id/approve
Response: Plan                   // with status="approved"

POST /api/plans/:id/execute
Response: {
  "plan": Plan,                  // with status="executed"
  "result_set": ResultSetResponse
}

POST /api/plans/:id/clarify
Request: {
  "choice_id"?: number,          // 1-based index into plan_json.choices
  "choice_text"?: string         // alternative to choice_id
}
Response: Plan                   // new plan created from clarification
```

---

### Session State (UI Reload)

```
GET /api/sessions/:id/state
Response: {
  "session_id": number,
  "latest_plan_id": number | null,
  "latest_result_set_id": number | null
}
```

---

### Results

```
GET /api/result-sets/:id
Response: ResultSetResponse
```

---

### Documents & Evidence

```
GET /api/documents/:id
Response: Document

GET /api/documents/:id/pdf
Response: Binary PDF file (application/pdf)
Headers: Content-Disposition: inline; filename="..."

GET /api/evidence
Query params:
  - document_id (required)
  - pdf_page (optional)
  - chunk_id (optional)
Response: {
  "document": Document,
  "evidence_refs": EvidenceRef[],
  "context": {
    "chunk_text"?: string,
    "page_text"?: string
  }
}
```

---

## Error Responses

All errors follow this shape:

```typescript
interface ErrorResponse {
  error: {
    code: string;                // e.g., "NOT_FOUND", "VALIDATION_ERROR"
    message: string;             // human-readable
    details?: unknown;           // optional structured details
  };
}
```

HTTP status codes:
- `400` - Validation error
- `404` - Resource not found
- `409` - Conflict (e.g., plan already executed)
- `500` - Internal server error

---

## Versioning Policy

- This contract is **v1**
- Breaking changes require incrementing to **v2**
- Non-breaking additions (new optional fields) are allowed in v1
- Clients should ignore unknown fields
- Server includes `contract_version` in `/api/meta` response
