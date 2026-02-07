/**
 * API Client for Research Console
 * Wraps fetch with type safety and error handling
 */

import type {
  Session,
  Message,
  Plan,
  ResultSetResponse,
  Document,
  HealthResponse,
  MetaResponse,
  SessionStateResponse,
  SendMessageRequest,
  SendMessageResponse,
  ExecutePlanResponse,
  ClarifyPlanRequest,
  EvidenceResponse,
  CreateSessionRequest,
  ErrorResponse,
  ChatRequest,
  ChatResponse,
  ChatMessage,
  V9ChatRequest,
  V9ChatResponse,
} from '@/types/api';

// Production: health is https://api.fridayarchive.org/health, API routes are under /api (e.g. /api/sessions/...)
const PRODUCTION_API_HOST = 'https://api.fridayarchive.org';
const PRODUCTION_API_BASE = `${PRODUCTION_API_HOST}/api`;

// API_BASE: Use environment variable if set, otherwise use relative path for dev proxy
const API_BASE = process.env.NEXT_PUBLIC_API_URL || '/api';

/** Normalize API base: ensure it ends with /api when it's a full URL so paths like /sessions/... become /api/sessions/... */
function normalizeApiBase(base: string): string {
  if (base.startsWith('http') && !base.endsWith('/api') && !base.endsWith('/api/')) {
    return base.replace(/\/?$/, '') + '/api';
  }
  return base;
}

/** Resolve API base at request time so production works even if build omitted NEXT_PUBLIC_API_URL */
function getRequestBase(): string {
  if (typeof window === 'undefined') return normalizeApiBase(API_BASE);
  // Production (non-localhost): if base is relative, use production API so static deploy works
  if (!window.location.hostname.includes('localhost') && (API_BASE.startsWith('/') || API_BASE === '/api')) {
    return PRODUCTION_API_BASE;
  }
  return normalizeApiBase(API_BASE);
}

// Direct backend URL for SSE streaming (bypasses Next.js proxy which buffers responses)
// In browser, we need to connect directly to avoid proxy buffering
const getDirectBackendUrl = (): string => {
  // Check for explicit override via environment variable first
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  // Check for window override (useful for testing)
  if (typeof window !== 'undefined' && (window as unknown as { __BACKEND_URL__?: string }).__BACKEND_URL__) {
    return (window as unknown as { __BACKEND_URL__?: string }).__BACKEND_URL__ as string;
  }
  // Try to get from localStorage (can be set manually for debugging)
  if (typeof window !== 'undefined') {
    const stored = localStorage.getItem('DIRECT_BACKEND_URL');
    if (stored) return stored;
  }
  // In production (non-localhost), use HTTPS production URL (routes live at .../api/sessions/...)
  if (typeof window !== 'undefined' && !window.location.hostname.includes('localhost')) {
    return PRODUCTION_API_BASE;
  }
  // Local development: use HTTP on same host port 8000
  if (typeof window !== 'undefined') {
    return `http://${window.location.hostname}:8000/api`;
  }
  return 'http://localhost:8000/api';
};

class ApiError extends Error {
  constructor(
    public code: string,
    message: string,
    public details?: unknown
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const base = getRequestBase();
  const url = path.startsWith('http') ? path : `${base}${path}`;

  const response = await fetch(url, {
    ...options,
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (!response.ok) {
    let errorData: any = null;
    try {
      errorData = await response.json();
    } catch {
      // Response wasn't JSON
    }
    
    throw new ApiError(
      errorData?.error?.code || `HTTP_${response.status}`,
      // Support both our ErrorResponse shape and FastAPI's default {"detail": "..."}
      errorData?.error?.message || errorData?.detail || response.statusText,
      errorData?.error?.details || (errorData?.detail ? { detail: errorData.detail } : undefined)
    );
  }

  // Handle empty responses (204 No Content)
  if (response.status === 204) {
    return {} as T;
  }

  return response.json();
}

// =============================================================================
// Auth (cookie-based session; routes at /auth/* on API host)
// =============================================================================

/** Base URL for auth routes (host only, no /api suffix) */
function getAuthBase(): string {
  if (typeof window === 'undefined') {
    const base = normalizeApiBase(API_BASE);
    return base.replace(/\/api\/?$/, '') || base;
  }
  if (!window.location.hostname.includes('localhost')) {
    return PRODUCTION_API_HOST;
  }
  if (process.env.NEXT_PUBLIC_API_URL) {
    const u = process.env.NEXT_PUBLIC_API_URL.replace(/\/api\/?$/, '');
    return u || process.env.NEXT_PUBLIC_API_URL;
  }
  return `http://${window.location.hostname}:8000`;
}

export interface AuthUser {
  sub: string;
  email?: string;
}

/** Frontend route for sign-in page (do not redirect browser to API). */
export const SIGNIN_PATH = '/signin';

export function getLoginUrl(): string {
  return SIGNIN_PATH;
}

export async function getAuthMe(): Promise<AuthUser | null> {
  try {
    const res = await fetch(`${getAuthBase()}/auth/me`, { credentials: 'include' });
    if (res.status === 401) return null;
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

/** POST /auth/login (non-OAuth). Call from /signin page with credentials: 'include'. */
export async function login(email: string, password: string): Promise<{ ok: boolean }> {
  const res = await fetch(`${getAuthBase()}/auth/login`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new ApiError(
      data?.error?.code || `HTTP_${res.status}`,
      data?.error?.message || data?.detail || res.statusText
    );
  }
  return res.json();
}

export async function logout(): Promise<void> {
  await fetch(`${getAuthBase()}/auth/logout`, {
    method: 'POST',
    credentials: 'include',
  });
  if (typeof window !== 'undefined') {
    window.location.href = '/';
  }
}

// =============================================================================
// Meta & Health
// =============================================================================

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health');
}

export async function getMeta(): Promise<MetaResponse> {
  return request<MetaResponse>('/meta');
}

// =============================================================================
// Sessions
// =============================================================================

export async function createSession(data: CreateSessionRequest): Promise<Session> {
  return request<Session>('/sessions', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function getSessions(): Promise<Session[]> {
  return request<Session[]>('/sessions');
}

export async function getSession(id: number): Promise<Session> {
  return request<Session>(`/sessions/${id}`);
}

export async function deleteSession(id: number): Promise<void> {
  return request<void>(`/sessions/${id}`, { method: 'DELETE' });
}

export async function getSessionState(id: number): Promise<SessionStateResponse> {
  return request<SessionStateResponse>(`/sessions/${id}/state`);
}

export async function getSessionMessages(sessionId: number): Promise<Message[]> {
  return request<Message[]>(`/sessions/${sessionId}/messages`);
}

// =============================================================================
// Messages
// =============================================================================

export async function sendMessage(
  sessionId: number,
  data: SendMessageRequest
): Promise<SendMessageResponse> {
  return request<SendMessageResponse>(`/sessions/${sessionId}/messages`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

// =============================================================================
// Plans
// =============================================================================

export async function getPlan(id: number): Promise<Plan> {
  return request<Plan>(`/plans/${id}`);
}

export async function approvePlan(id: number): Promise<Plan> {
  return request<Plan>(`/plans/${id}/approve`, {
    method: 'POST',
  });
}

export async function executePlan(id: number): Promise<ExecutePlanResponse> {
  return request<ExecutePlanResponse>(`/plans/${id}/execute`, {
    method: 'POST',
  });
}

export async function clarifyPlan(id: number, data: ClarifyPlanRequest): Promise<Plan> {
  return request<Plan>(`/plans/${id}/clarify`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

// =============================================================================
// Results
// =============================================================================

export async function getResultSet(id: number): Promise<ResultSetResponse> {
  return request<ResultSetResponse>(`/result-sets/${id}`);
}

// =============================================================================
// Documents & Evidence
// =============================================================================

export async function getDocument(id: number): Promise<Document> {
  return request<Document>(`/documents/${id}`);
}

export function getDocumentPdfUrl(id: number): string {
  return `${getRequestBase()}/documents/${id}/pdf`;
}

export async function deletePendingMessage(sessionId: number): Promise<void> {
  // Uses the direct backend URL (same as streaming) to bypass Next.js proxy
  const directUrl = getDirectBackendUrl();
  await fetch(`${directUrl}/sessions/${sessionId}/chat/last-pending`, {
    method: 'DELETE',
  });
}

export async function getEvidence(params: {
  document_id: number;
  pdf_page?: number;
  chunk_id?: number;
}): Promise<EvidenceResponse> {
  const searchParams = new URLSearchParams();
  searchParams.set('document_id', String(params.document_id));
  if (params.pdf_page !== undefined) {
    searchParams.set('pdf_page', String(params.pdf_page));
  }
  if (params.chunk_id !== undefined) {
    searchParams.set('chunk_id', String(params.chunk_id));
  }
  
  return request<EvidenceResponse>(`/evidence?${searchParams}`);
}

// =============================================================================
// V6 Chat
// =============================================================================

/**
 * Send a chat message and get a V6-powered response.
 * This is the main interaction endpoint using the V6 agentic workflow.
 */
export async function sendChatMessage(
  sessionId: number,
  data: ChatRequest
): Promise<ChatResponse> {
  return request<ChatResponse>(`/sessions/${sessionId}/chat`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

/**
 * Get the chat history for a session with V6 metadata.
 */
export async function getChatHistory(sessionId: number): Promise<ChatMessage[]> {
  return request<ChatMessage[]>(`/sessions/${sessionId}/chat/history`);
}

/**
 * Progress event from V6 workflow streaming.
 */
export interface V6ProgressEvent {
  type: 'progress';
  step: string;
  status: string;
  message: string;
  details: Record<string, unknown>;
  timestamp: number;
}

/**
 * Result event from V6 workflow streaming (final result).
 */
export interface V6ResultEvent {
  answer: string;
  claims: Array<{
    text: string;
    confidence: string;
    citations: Array<{
      span_id: string;
      chunk_id?: number;
      document_id?: number;
      page_number?: number;
      quote?: string;
      source_name?: string;
    }>;
  }>;
  members: Array<{
    name: string;
    citations: Array<{
      span_id: string;
      chunk_id?: number;
      document_id?: number;
      page_number?: number;
      quote?: string;
      source_name?: string;
    }>;
  }>;
  stats: {
    task_type: string;
    rounds_executed: number;
    total_spans: number;
    unique_docs: number;
    elapsed_ms: number;
    entity_linking: Record<string, unknown>;
    responsiveness: string;
    actions: Array<{
      step: string;
      status: string;
      message: string;
      details?: Record<string, unknown>;
    }>;
  };
  is_responsive: boolean;
}

/**
 * Callbacks for streaming chat.
 */
export interface StreamingChatCallbacks {
  onProgress?: (event: V6ProgressEvent) => void;
  onResult?: (result: V6ResultEvent) => void;
  onError?: (error: string) => void;
}

/**
 * Send a chat message with streaming progress updates.
 * Uses Server-Sent Events to receive real-time workflow progress.
 * 
 * IMPORTANT: This bypasses the Next.js proxy and connects directly to the backend
 * because the proxy buffers responses which breaks SSE streaming.
 */
export async function sendChatMessageStreaming(
  sessionId: number,
  message: string,
  callbacks: StreamingChatCallbacks,
  signal?: AbortSignal
): Promise<void> {
  // Use direct backend URL to bypass Next.js proxy (which buffers SSE)
  const directUrl = getDirectBackendUrl();
  
  // Helper: check if an error is an abort (robust across browsers/bundlers)
  const isAbort = (e: unknown): boolean =>
    signal?.aborted === true ||
    (e instanceof Error && e.name === 'AbortError') ||
    (typeof DOMException !== 'undefined' && e instanceof DOMException && e.name === 'AbortError');

  let response: Response;
  try {
    response = await fetch(`${directUrl}/sessions/${sessionId}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message }),
      signal,
    });
  } catch (e: unknown) {
    // AbortError is expected when the user clicks Stop — not a real error
    if (isAbort(e)) return;
    throw e;
  }

  if (!response.ok) {
    let error: string;
    try {
      error = await response.text();
    } catch (e: unknown) {
      if (isAbort(e)) return;
      error = response.statusText;
    }
    callbacks.onError?.(error);
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    callbacks.onError?.('No response body');
    return;
  }

  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      // Check abort before each read so we exit immediately
      if (signal?.aborted) return;

      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE events from buffer
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      let currentEvent = '';
      let currentData = '';

      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
          currentData = line.slice(6);
        } else if (line === '' && currentData) {
          // End of event
          try {
            const data = JSON.parse(currentData);
            if (currentEvent === 'progress') {
              callbacks.onProgress?.(data as V6ProgressEvent);
            } else if (currentEvent === 'result') {
              callbacks.onResult?.(data as V6ResultEvent);
            } else if (currentEvent === 'error') {
              callbacks.onError?.(data.error || 'Unknown error');
            }
          } catch {
            // Ignore parse errors
          }
          currentEvent = '';
          currentData = '';
        }
      }
    }
  } catch (e: unknown) {
    // AbortError is expected when the user clicks Stop
    if (isAbort(e)) return;
    throw e;
  }
}

// =============================================================================
// V9 Session-Aware Chat
// =============================================================================

/**
 * Send a message via the V9 session-aware endpoint (synchronous fallback).
 * Routes automatically to new_retrieval, follow_up, or think_deeper.
 */
export async function sendV9Message(
  sessionId: number,
  text: string,
  action: 'default' | 'think_deeper' = 'default',
): Promise<V9ChatResponse> {
  return request<V9ChatResponse>(`/sessions/${sessionId}/v9/message`, {
    method: 'POST',
    body: JSON.stringify({ text, action } as V9ChatRequest),
  });
}

// =============================================================================
// V9 SSE Streaming
// =============================================================================

/**
 * Progress event from V9 investigation workflow.
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
 * Evidence bullet discovered during investigation.
 */
export interface V9EvidenceBullet {
  text: string;
  tags: string[];
  chunk_ids: number[];
  doc_ids: number[];
}

/**
 * Evidence update event — carries actual discovered evidence.
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

/**
 * Callbacks for V9 streaming chat.
 */
export interface V9StreamingCallbacks {
  onProgress?: (event: V9ProgressEvent) => void;
  onEvidenceUpdate?: (event: V9EvidenceUpdateEvent) => void;
  onResult?: (result: V9ChatResponse) => void;
  onError?: (error: string) => void;
}

/**
 * Send a V9 message with SSE streaming progress updates.
 * Uses POST + ReadableStream (not EventSource) to support request body.
 * Bypasses Next.js proxy to avoid response buffering.
 */
export async function sendV9MessageStreaming(
  sessionId: number,
  text: string,
  action: 'default' | 'think_deeper',
  callbacks: V9StreamingCallbacks,
  signal?: AbortSignal,
  carryContext?: Record<string, unknown>,
): Promise<void> {
  const directUrl = getDirectBackendUrl();

  const isAbort = (e: unknown): boolean =>
    signal?.aborted === true ||
    (e instanceof Error && e.name === 'AbortError') ||
    (typeof DOMException !== 'undefined' && e instanceof DOMException && e.name === 'AbortError');

  const body: Record<string, unknown> = { text, action };
  if (carryContext) {
    body.carry_context = carryContext;
  }

  let response: Response;
  try {
    response = await fetch(`${directUrl}/sessions/${sessionId}/v9/message/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    });
  } catch (e: unknown) {
    if (isAbort(e)) return;
    throw e;
  }

  if (!response.ok) {
    let error: string;
    try {
      error = await response.text();
    } catch (e: unknown) {
      if (isAbort(e)) return;
      error = response.statusText;
    }
    callbacks.onError?.(error);
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    callbacks.onError?.('No response body');
    return;
  }

  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      if (signal?.aborted) return;

      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE events from buffer
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      let currentEvent = '';
      let currentData = '';

      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
          currentData = line.slice(6);
        } else if (line === '' && currentData) {
          // End of SSE event
          try {
            const data = JSON.parse(currentData);
            if (currentEvent === 'progress') {
              callbacks.onProgress?.(data as V9ProgressEvent);
            } else if (currentEvent === 'evidence_update') {
              callbacks.onEvidenceUpdate?.(data as V9EvidenceUpdateEvent);
            } else if (currentEvent === 'result') {
              callbacks.onResult?.(data as V9ChatResponse);
            } else if (currentEvent === 'error') {
              callbacks.onError?.(data.error || 'Unknown error');
            }
          } catch {
            // Ignore parse errors
          }
          currentEvent = '';
          currentData = '';
        }
      }
    }
  } catch (e: unknown) {
    if (isAbort(e)) return;
    throw e;
  }
}

// =============================================================================
// Scope Window API
// =============================================================================

import type { CollectionNode, DocumentNode, UserSelectedScope } from '@/types/api';

async function getCollectionsTree(includeCounts = false): Promise<CollectionNode[]> {
  const params = includeCounts ? '?include_counts=1' : '';
  return request<CollectionNode[]>(`/documents/collections_tree${params}`);
}

async function getCollectionDocuments(collectionId: number, includeCounts = false): Promise<DocumentNode[]> {
  const params = includeCounts ? '?include_counts=1' : '';
  return request<DocumentNode[]>(`/documents/collections/${collectionId}/documents${params}`);
}

async function getSessionScope(sessionId: number): Promise<UserSelectedScope> {
  const session = await getSession(sessionId);
  return session.scope_json || { mode: 'full_archive' };
}

async function updateSessionScope(sessionId: number, scope: UserSelectedScope): Promise<void> {
  await request(`/sessions/${sessionId}/scope`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(scope),
  });
}


// =============================================================================
// Export
// =============================================================================

export const api = {
  getHealth,
  getMeta,
  getLoginUrl,
  getAuthMe,
  login,
  logout,
  createSession,
  getSessions,
  getSession,
  deleteSession,
  getSessionState,
  getSessionMessages,
  sendMessage,
  getPlan,
  approvePlan,
  executePlan,
  clarifyPlan,
  getResultSet,
  getDocument,
  getDocumentPdfUrl,
  getEvidence,
  deletePendingMessage,
  // V6 Chat
  sendChatMessage,
  sendChatMessageStreaming,
  getChatHistory,
  // V9 Session-Aware Chat
  sendV9Message,
  sendV9MessageStreaming,
  // Scope Window
  getCollectionsTree,
  getCollectionDocuments,
  getSessionScope,
  updateSessionScope,
};

export { ApiError };
