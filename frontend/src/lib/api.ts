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
} from '@/types/api';

// Production API URL - must use HTTPS
// Can be set via NEXT_PUBLIC_API_URL environment variable at build time
const PRODUCTION_API_URL = 'https://api.fridayarchive.org';

// API_BASE: Use environment variable if set, otherwise use relative path for dev proxy
// In production builds, NEXT_PUBLIC_API_URL should be set to the full API URL
const API_BASE = process.env.NEXT_PUBLIC_API_URL || '/api';

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
  // In production (non-localhost), use HTTPS production URL
  if (typeof window !== 'undefined' && !window.location.hostname.includes('localhost')) {
    return `${PRODUCTION_API_URL}/api`;
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
  const url = `${API_BASE}${path}`;
  
  const response = await fetch(url, {
    ...options,
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
  return `${API_BASE}/documents/${id}/pdf`;
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
  callbacks: StreamingChatCallbacks
): Promise<void> {
  // Use direct backend URL to bypass Next.js proxy (which buffers SSE)
  const directUrl = getDirectBackendUrl();
  
  const response = await fetch(`${directUrl}/sessions/${sessionId}/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message }),
  });

  if (!response.ok) {
    const error = await response.text();
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

  while (true) {
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
}

// =============================================================================
// Export
// =============================================================================

export const api = {
  getHealth,
  getMeta,
  createSession,
  getSessions,
  getSession,
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
  // V6 Chat
  sendChatMessage,
  sendChatMessageStreaming,
  getChatHistory,
};

export { ApiError };
