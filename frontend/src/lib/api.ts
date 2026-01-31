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
} from '@/types/api';

const API_BASE = '/api';

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
};

export { ApiError };
