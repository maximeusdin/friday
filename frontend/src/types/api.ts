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
