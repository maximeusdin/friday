/**
 * API Types - Generated from docs/v1_contract.md
 * Contract Version: v1
 * DO NOT EDIT manually - regenerate from contract
 */

// =============================================================================
// Core Types
// =============================================================================

/**
 * The atomic unit of citation. Links a result to a specific location in source material.
 */
export interface EvidenceRef {
  document_id: number;
  pdf_page: number;              // 1-based PDF page number
  chunk_id?: number;
  span?: {
    start: number;
    end: number;
  };
  quote?: string;
  why?: string;
}

/**
 * A single message in a research session conversation.
 */
export interface Message {
  id: number;
  session_id: number;
  role: 'user' | 'assistant' | 'system';
  content: string;
  plan_id?: number;
  result_set_id?: number;
  metadata?: Record<string, unknown>;
  created_at: string;
}

/**
 * A structured research plan proposed by the assistant.
 */
export interface Plan {
  id: number;
  session_id: number;
  status: 'proposed' | 'approved' | 'executed' | 'rejected' | 'superseded';
  user_utterance: string;
  plan_json: PlanJson;
  plan_summary: string;
  parent_plan_id?: number;
  retrieval_run_id?: number;
  result_set_id?: number;
  created_at: string;
  approved_at?: string;
  executed_at?: string;
}

export interface PlanJson {
  primitives: Primitive[];
  execution_envelope?: {
    top_k?: number;
    search_type?: string;
    scope_sql?: string;
  };
}

export interface Primitive {
  type: string;
  [key: string]: unknown;
}

/**
 * A research session container.
 */
export interface Session {
  id: number;
  label: string;
  created_at: string;
  message_count?: number;
  last_activity?: string;
  scope_json?: UserSelectedScope;
}

/**
 * UI-ready representation of execution results.
 */
export interface ResultSetResponse {
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

export interface ResultItem {
  id: string;                    // stable row ID for React keys
  kind?: 'chunk' | 'entity' | 'doc' | 'note';
  rank: number;
  text: string;
  chunk_id?: number;
  document_id?: number;
  entity_id?: number;
  scores?: {
    lex?: number;
    vec?: number;
    hybrid?: number;
  };
  highlight?: string;
  matched_terms?: string[];
  evidence_refs: EvidenceRef[];
}

/**
 * Document metadata for evidence viewer.
 */
export interface Document {
  id: number;
  collection_id: number;
  collection_slug?: string;
  source_name: string;
  source_ref?: string;
  volume?: string;
  page_count?: number;
  pdf_url?: string;
  metadata?: Record<string, unknown>;
  created_at: string;
}

// =============================================================================
// API Response Types
// =============================================================================

export interface HealthResponse {
  status: 'ok';
}

export interface MetaResponse {
  contract_version: string;
  api_version: string;
  build: string;
}

export interface SessionStateResponse {
  session_id: number;
  latest_plan_id?: number | null;
  latest_result_set_id?: number | null;
}

export interface SendMessageRequest {
  content: string;
}

export interface SendMessageResponse {
  user_message: Message;
  assistant_message: Message;
  plan: Plan;
}

export interface ExecutePlanResponse {
  plan: Plan;
  result_set: ResultSetResponse;
}

export interface ClarifyPlanRequest {
  choice_id?: number;      // 1-based
  choice_text?: string;
}

export interface EvidenceResponse {
  document: Document;
  evidence_refs: EvidenceRef[];
  context: {
    chunk_text?: string;
    page_text?: string;
  };
}

export interface CreateSessionRequest {
  label: string;
}

export interface ErrorResponse {
  error: {
    code: string;
    message: string;
    details?: unknown;
  };
}

// =============================================================================
// V6 Chat Types
// =============================================================================

/**
 * A citation reference for a claim in V6 responses.
 */
export interface ChatCitation {
  span_id: string;
  chunk_id?: number;
  document_id?: number;
  page_number?: number;
  quote: string;
  source_name?: string;
  relevance_score?: number;
}

/**
 * A claim with evidence citations from V6 workflow.
 */
export interface ChatClaim {
  text: string;
  confidence: 'supported' | 'partial' | 'unsupported';
  citations: ChatCitation[];
}

/**
 * An identified member (for roster queries).
 */
export interface ChatMember {
  name: string;
  citations: ChatCitation[];
}

/**
 * A single action/step in the V6 workflow.
 */
export interface WorkflowAction {
  step: string;
  status: 'running' | 'completed' | 'skipped' | 'error';
  message: string;
  details?: Record<string, unknown>;
  elapsed_ms?: number;
}

/**
 * Statistics about the V6/V7 workflow execution.
 */
export interface V6Stats {
  task_type: string;
  rounds_executed: number;
  total_spans: number;
  unique_docs: number;
  elapsed_ms: number;
  entity_linking: {
    total_linked?: number;
    used_for_retrieval?: number;
  };
  responsiveness: string;
  actions: WorkflowAction[];
  // V7-specific fields
  citation_validation_passed?: boolean;
  claims_extracted?: number;
  claims_valid?: number;
  claims_dropped?: number;
}

/**
 * Detail for a single inline citation label, mapping to a document + page.
 */
export interface CitationDetail {
  chunk_id: number;
  document_id?: number;
  page?: number;
  /** Human-readable label (e.g. "Vassiliev p4") when citation is keyed by chunk id */
  label?: string;
}

/**
 * V9-specific metadata attached to assistant messages.
 */
export interface V9Meta {
  intent?: string;
  confidence?: string;
  can_think_deeper: boolean;
  remaining_gaps: string[];
  suggestion: string;
  elapsed_ms: number;
  run_id?: number;
  evidence_set_id?: number;
  cited_chunk_ids: number[];
  citation_map?: Record<string, CitationDetail>;
  scope_meta?: ScopeMeta;
  escalations?: EscalationOption[];
  scope_override?: ScopeOverrideInfo;
  expansion_info?: ExpansionInfo;
}

/**
 * A message in the chat history with V6/V9 metadata.
 */
export interface ChatMessage {
  id: number;
  session_id: number;
  role: 'user' | 'assistant';
  content: string;
  claims?: ChatClaim[];
  members?: ChatMember[];
  v6_stats?: V6Stats;
  v9_meta?: V9Meta;
  result_set_id?: number;
  created_at: string;
}

/**
 * Request to send a chat message.
 */
export interface ChatRequest {
  message: string;
}

/**
 * Response from the V6 chat endpoint.
 */
export interface ChatResponse {
  user_message: ChatMessage;
  assistant_message: ChatMessage;
  is_responsive: boolean;
  result_set_id?: number;
}

// =============================================================================
// V9 Scope & Escalation Types
// =============================================================================

/**
 * Evidence set scope context shown during follow-up answers.
 * Tells the user what "lens" the follow-up is operating within.
 */
export interface ScopeMeta {
  origin_query: string;
  origin_run_id?: number;
  evidence_set_id: number;
  chunk_count: number;
  document_count: number;
  top_entities: Array<{ canonical_name: string; aliases: string[] }>;
  time_range?: string;
}

/**
 * A structured next-action offered when follow-up evidence is insufficient.
 */
export interface EscalationOption {
  action: 'think_deeper' | 'new_retrieval' | 'show_evidence';
  label: string;
  description: string;
  prefilled_query?: string;
  carry_entities: Array<{ canonical_name: string; aliases: string[] }>;
  recommended: boolean;
}

// =============================================================================
// Scope Window Types
// =============================================================================

export interface CollectionNode {
  id: number;
  slug: string;
  title: string;
  description?: string;
  document_count: number;
  chunk_count?: number;          // only with ?include_counts=1
  documents?: DocumentNode[];    // lazy-loaded via GET /collections/{id}/documents
  _docsLoaded?: boolean;         // client-side flag: true once documents fetched
}

export interface DocumentNode {
  id: number;
  source_name: string;
  source_ref?: string;
  volume?: string;
  chunk_count?: number;
}

export type ScopeMode = 'full_archive' | 'custom';

export interface UserSelectedScope {
  mode: ScopeMode;
  included_collection_ids?: number[];
  included_document_ids?: number[];
}

export interface RunScopeInfo {
  mode: ScopeMode;
  included_collection_ids?: number[];
  included_document_ids?: number[];
  source: 'user_selected' | 'query_override';
  reason?: string;
  filters?: { date_from?: string; date_to?: string };
  expansion?: ExpansionInfo;
}

export interface ExpansionInfo {
  policy: string;
  collections: string[];
  triggered: boolean;
  reason?: string;
}

export interface ScopeOverrideInfo {
  overridden: boolean;
  selected_scope?: UserSelectedScope;
  run_scope?: RunScopeInfo;
}


// =============================================================================
// V9 Session-Aware Types
// =============================================================================

export interface V9ChatRequest {
  text: string;
  action?: 'default' | 'think_deeper';
  carry_context?: { entities?: Array<{ canonical_name: string; aliases: string[] }>; intent_hint?: string };
}

export interface V9RunSummary {
  run_id: number;
  query_index: number;
  query_text: string;
  label?: string;
  status: string;
  evidence_set_id?: number;
  evidence_summary?: string;
}

export interface V9ChatResponse {
  intent: 'new_retrieval' | 'follow_up' | 'think_deeper';
  answer: string;
  cited_chunk_ids: number[];
  confidence: string;

  active_run_id?: number;
  active_run_status: string;
  active_evidence_set_id?: number;
  referenced_run_id?: number;
  referenced_evidence_set_id?: number;
  can_think_deeper: boolean;

  remaining_gaps: string[];
  next_best_actions: string[];

  run_history: V9RunSummary[];

  routing_reasoning: string;
  routing_confidence: number;

  /** Maps inline citation labels (e.g. "Vassiliev P4") to document details for PDF viewer linking. */
  citation_map?: Record<string, CitationDetail>;

  suggestion: string;

  /** Evidence set scope context (only present for follow_up intent). */
  scope_meta?: ScopeMeta;

  /** Structured escalation options when follow-up confidence is low/insufficient. */
  escalations?: EscalationOption[];

  /** Scope override info (present for new_retrieval when scope differs from session). */
  scope_override?: ScopeOverrideInfo;

  /** Stage 1.5 concordance expansion status. */
  expansion_info?: ExpansionInfo;

  elapsed_ms: number;
}

// =============================================================================
// V9 SSE Streaming Types
// =============================================================================

/**
 * Progress event from V9 investigation workflow streaming.
 */
export interface V9ProgressEvent {
  type: 'progress';
  step: string;
  status: string;
  message: string;
  details: Record<string, unknown>;
  timestamp: number;
}

/**
 * An evidence bullet discovered during investigation.
 */
export interface V9EvidenceBullet {
  text: string;
  tags: string[];
  chunk_ids: number[];
  doc_ids: number[];
}

/**
 * Evidence update event carrying actual discovered evidence.
 */
export interface V9EvidenceUpdateEvent {
  type: 'evidence_update';
  step: string;
  message: string;
  details: {
    bullets: V9EvidenceBullet[];
    open_questions: string[];
    leads: string[];
    total_bullet_count: number;
  };
}
