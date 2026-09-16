'use client';

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { useChatRun, startRun, stopRun } from '@/lib/chatRunStore';
import type {
  Session, ChatMessage, ChatClaim, ChatMember, V6Stats, EvidenceRef, ChatCitation,
  V9ChatResponse, V9Meta, V9ProgressEvent, V9EvidenceBullet, CitationDetail,
  ScopeMeta, EscalationOption, UserSelectedScope, CollectionNode, ClarificationAnswer,
} from '@/types/api';
import { describeScope } from '@/lib/scope';
import { duration, plural } from '@/lib/format';
import { useDocumentName } from '@/lib/documentNames';
import { AnswerText } from './AnswerText';
import { ClarificationCard } from './ClarificationCard';
import { ScopeControl } from './ScopeControl';
import { Welcome } from './Welcome';
import { Icon } from './ui/Icon';
import { toast } from './ui/Toast';
import { useLayout, currentLayout } from '@/lib/useLayout';

/** Shown when the backend reports a step without saying what it is doing.
 *  One honest line beats a rotating set of invented ones: the elapsed timer
 *  beside it already shows that work is happening. */
const WORKING = 'Searching the archive…';

/** Strip chunk_id=..., chunk_ids: [], etc. from bullet text (the model sometimes echoes prompt format). */
function sanitizeBulletText(text: string): string {
  return text
    .replace(/\s*[\[\(]chunk_id=\d+[\)\]]\s*/gi, ' ')
    .replace(/\s*[\[\(]chunk_ids:\s*\[\d*(?:,\s*\d*)*\]\s*[\)\]]\s*/gi, ' ')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

/** Map a backend progress step to something a reader understands. Prefers the
 *  backend's own message when it carries real context. */
function toUserFriendlyProgress(step: V9ProgressEvent): string {
  switch (step.step) {
    case 'tool_call': {
      const msg = (step.message || '').trim();
      if (msg && (msg.startsWith('Searching for:') || msg.startsWith('Loading ') || msg.includes('passages'))) {
        return msg;
      }
      const tool = (step.details?.tool as string) || '';
      if (tool.startsWith('fetch_chunks') || tool === 'fetch_chunks') return 'Reading documents…';
      if (tool.startsWith('expand_entities') || tool === 'expand_entities') return 'Looking up cover names…';
      if (tool.startsWith('alias_index') || tool.includes('alias')) return 'Looking up cover names…';
      return WORKING;
    }
    case 'turn_start':
    case 'turn_prepare':
      return 'Investigating…';
    case 'model_call':
      return 'Reading the evidence…';
    case 'investigation':
      return 'Searching and reading…';
    case 'entity_resolution':
      return 'Looking up cover names…';
    case 'synthesis':
      return 'Writing the answer…';
    case 'evidence_update':
      return 'Found something…';
    case 'routing':
    case 'routing_start':
      return 'Reading your question…';
    case 'investigation_start':
    case 'retrieval_prepare':
      return 'Getting started…';
    case 'context_build':
      return 'Gathering context…';
    case 'follow_up_start':
    case 'follow_up':
      return 'Searching evidence…';
    case 'think_deeper_start':
      return 'Resuming Think deeper…';
    default: {
      const msg = (step.message || '').toLowerCase();
      if (msg.includes('search')) return WORKING;
      if (msg.includes('fetch')) return 'Reading documents…';
      if (msg.includes('synthes')) return 'Writing the answer…';
      return WORKING;
    }
  }
}

/** Seconds as m:ss, for the in-flight timer. */
function clock(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return m > 0 ? `${m}:${String(s).padStart(2, '0')}` : `${s}s`;
}

const INTENT_LABEL: Record<string, string> = {
  new_retrieval: 'New search',
  follow_up: 'Follow-up',
  think_deeper: 'Think deeper',
};

interface ConversationProps {
  session: Session | null;
  onViewSearchResultSet?: (resultSetId: string) => void;
  onOpenSearchTab?: () => void;
  onEvidenceClick?: (evidence: EvidenceRef) => void;
  activeScope: UserSelectedScope | null;
  collections: CollectionNode[];
  onScopeChange: (scope: UserSelectedScope) => void;
  /** Asked with no session open — the parent creates one, then queues the question. */
  onStartSession: (question: string) => void;
  /** Incremented when "New session" is clicked, so the composer takes focus even
   *  when the welcome screen was already showing. */
  newSessionNonce?: number;
  /** Question queued by the parent to auto-send once the new session is active. */
  pendingQuestion?: string | null;
  onPendingQuestionConsumed?: () => void;
}

export function Conversation({
  session, onViewSearchResultSet, onOpenSearchTab, onEvidenceClick,
  activeScope, collections, onScopeChange, onStartSession, newSessionNonce,
  pendingQuestion, onPendingQuestionConsumed,
}: ConversationProps) {
  const [input, setInput] = useState('');
  const queryClient = useQueryClient();
  const layout = useLayout();
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Per-session run state lives in the shared store, so several sessions can
  // stream at once and a run keeps going when you switch away.
  const run = useChatRun(session?.id ?? null);
  const { isSending, sendError, lastV9, progressSteps, evidenceBullets, pendingText } = run;

  // Elapsed time while a run is in flight. Investigations can take minutes, so
  // the thinking row says how long it has been working rather than implying
  // progress with a bar that knows nothing.
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!isSending || !run.startedAt) {
      setElapsed(0);
      return;
    }
    const start = run.startedAt;
    const tick = () => setElapsed(Math.floor((Date.now() - start) / 1000));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [isSending, run.startedAt]);

  const { data: messages, isLoading } = useQuery({
    queryKey: ['chatHistory', session?.id],
    queryFn: () => api.getChatHistory(session!.id),
    enabled: !!session,
  });

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isSending]);

  // Starting a new session puts the cursor where the work begins. Not on touch:
  // focusing a field raises the keyboard over half the screen, and iOS Safari
  // then scrolls the page to it, so the welcome screen opened half-hidden.
  // Read the layout live: during hydration the hook still holds the server's
  // desktop snapshot, and this effect fires before the real value lands.
  useEffect(() => {
    if (!session && !currentLayout().touch) inputRef.current?.focus();
  }, [session, newSessionNonce]);

  // Auto-grow the composer up to the max height the stylesheet allows.
  useLayoutEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${el.scrollHeight}px`;
  }, [input]);

  const sendV9 = useCallback((
    text: string,
    action: 'default' | 'think_deeper' = 'default',
    carryContext?: Record<string, unknown>,
  ) => {
    if (!session || run.isSending) return;
    setInput('');
    startRun(session.id, text, {
      action,
      carryContext,
      scope: activeScope || { mode: 'full_archive' },
      queryClient,
    });
  }, [session, run.isSending, activeScope, queryClient]);

  const scopeEmpty = activeScope?.mode === 'custom'
    && (activeScope.included_collection_ids?.length ?? 0) === 0
    && (activeScope.included_document_ids?.length ?? 0) === 0;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || isSending || scopeEmpty) return;
    if (!session) {
      setInput('');
      onStartSession(text);
      return;
    }
    sendV9(text);
  };

  // V12: the user answered the follow-up question(s) — re-run the original query
  // with the answers + plan carried as context so the agent resolves intent.
  const submitClarification = (answers: ClarificationAnswer[]) => {
    if (!lastV9?.clarification) return;
    const queryText = messages?.filter((m) => m.role === 'user').pop()?.content?.trim();
    if (!queryText) return;
    sendV9(queryText, 'default', {
      clarification_answers: answers,
      clarification_plan: lastV9.clarification,
    });
  };

  const handleThinkDeeper = () => {
    // The API returns active_run_id / active_evidence_set_id; persisted metadata may use run_id / evidence_set_id.
    const runId = lastV9?.active_run_id ?? lastV9?.run_id;
    const evidenceSetId = lastV9?.active_evidence_set_id ?? lastV9?.evidence_set_id;
    if (!runId || !evidenceSetId) return;
    let queryText = messages?.filter((m) => m.role === 'user').pop()?.content?.trim();
    if (!queryText && lastV9?.run_history?.length) {
      queryText = lastV9.run_history.find((r) => r.run_id === runId)?.query_text?.trim();
    }
    if (queryText) {
      sendV9(queryText, 'think_deeper', { run_id: runId, evidence_set_id: evidenceSetId });
    }
  };

  // Auto-send a question queued from the welcome screen once the session exists.
  useEffect(() => {
    if (!session || !pendingQuestion) return;
    const q = pendingQuestion;
    onPendingQuestionConsumed?.();
    sendV9(q);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session?.id, pendingQuestion]);

  // Hide empty assistant placeholders (e.g. the v12 "clarify" message whose
  // questions render in the ClarificationCard, not as a chat bubble).
  const renderMessages = (messages ?? []).filter(
    (m) => !(m.role === 'assistant' && !(m.content || '').trim()),
  );

  const canThinkDeeper = !!(
    lastV9?.can_think_deeper
    || ((lastV9?.active_run_id ?? lastV9?.run_id) && (lastV9?.active_evidence_set_id ?? lastV9?.evidence_set_id))
  );

  return (
    <div className="ws-body">
      <div className="thread">
        {!session ? (
          <Welcome onAsk={onStartSession} />
        ) : (
          <div className="thread-inner">
            {isLoading && <div className="loading"><span className="spinner" /> Loading conversation…</div>}

            {renderMessages.map((message, idx) => {
              const isLast = idx === renderMessages.length - 1 && message.role === 'assistant';
              return (
                <ChatTurn
                  key={message.id}
                  message={message}
                  collections={collections}
                  onEvidenceClick={onEvidenceClick}
                  onViewSearchResultSet={onViewSearchResultSet}
                  onOpenSearchTab={onOpenSearchTab}
                  v9={isLast ? lastV9 : null}
                  onUseScope={onScopeChange}
                  liveBullets={isLast ? evidenceBullets : undefined}
                  isStreaming={isLast && isSending}
                  onThinkDeeper={isLast && canThinkDeeper && !isSending ? handleThinkDeeper : undefined}
                  onEscalate={(action, text, carry) => {
                    if (action === 'think_deeper') sendV9(text, 'think_deeper', carry);
                    else if (action === 'new_retrieval') sendV9(text, 'default', carry);
                  }}
                  onPrefillInput={(text) => {
                    setInput(text);
                    inputRef.current?.focus();
                  }}
                />
              );
            })}

            {/* The question appears the instant it is sent; the refetched history
                replaces it once the server has persisted the turn. */}
            {pendingText
              && renderMessages.filter((m) => m.role === 'user').pop()?.content !== pendingText && (
              <div className="msg msg-user">
                <div className="msg-user-bubble">{pendingText}</div>
              </div>
            )}

            {isSending && (
              <div className="msg">
                <div className="answer">
                  {evidenceBullets.length > 0 && (
                    <Findings
                      bullets={evidenceBullets}
                      onEvidenceClick={onEvidenceClick}
                      streaming
                    />
                  )}
                  <div className="thinking">
                    <span className="spinner" />
                    <span className="thinking-text">
                      {elapsed >= 360
                        ? 'Still going. This one is taking a while…'
                        : progressSteps.length > 0
                          ? toUserFriendlyProgress(progressSteps[progressSteps.length - 1])
                          : 'Investigating…'}
                    </span>
                    {elapsed >= 5 && (
                      <span className="thinking-elapsed">{clock(elapsed)}</span>
                    )}
                  </div>
                </div>
              </div>
            )}

            {sendError && (
              <div className="notice notice-danger">
                <Icon name="info" size={16} />
                <span>Couldn&apos;t get an answer: {sendError}</span>
              </div>
            )}

            {/* V12: follow-up clarification questions before the investigation runs */}
            {lastV9?.needs_clarification && lastV9.clarification && !isSending && (
              <ClarificationCard clarification={lastV9.clarification} onSubmit={submitClarification} />
            )}

            <div ref={bottomRef} />
          </div>
        )}
      </div>

      <div className="composer">
        <div className="composer-inner">
          <form className="composer-box" onSubmit={handleSubmit}>
            <textarea
              ref={inputRef}
              className="composer-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                // Desktop: Enter sends, Shift+Enter breaks a line. On a touch keyboard
                // Enter is "return" and the Ask button sends, as in any messaging app.
                if (e.key === 'Enter' && !e.shiftKey && !layout.touch) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
              placeholder={session ? 'Ask a follow-up, or start something new…' : 'Ask the archive a question…'}
              disabled={isSending}
              rows={1}
              aria-label="Your question"
            />
            <div className="composer-row">
              <ScopeControl
                scope={activeScope}
                collections={collections}
                onChange={onScopeChange}
                placement="top"
              />
              <div className="spacer" />
              {isSending ? (
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => session && stopRun(session.id)}
                >
                  <Icon name="stop" size={15} />
                  Stop
                </button>
              ) : (
                <button
                  type="submit"
                  className="btn-primary"
                  disabled={!input.trim() || scopeEmpty}
                  title={scopeEmpty ? 'Select at least one collection first' : 'Ask (Enter)'}
                >
                  Ask
                  <Icon name="arrow-up" size={15} />
                </button>
              )}
            </div>
          </form>
          <div className="composer-hint">
            {scopeEmpty
              ? 'No sources selected. Pick collections in the scope menu, or switch back to the entire archive.'
              : layout.touch ? '' : 'Enter to send · Shift + Enter for a new line'}
          </div>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// One turn
// =============================================================================

function ChatTurn({
  message, collections, onEvidenceClick, onViewSearchResultSet, onOpenSearchTab,
  v9, onEscalate, onPrefillInput, onUseScope, liveBullets, isStreaming, onThinkDeeper,
}: {
  message: ChatMessage;
  collections: CollectionNode[];
  onEvidenceClick?: (evidence: EvidenceRef) => void;
  onViewSearchResultSet?: (resultSetId: string) => void;
  onOpenSearchTab?: () => void;
  v9?: V9ChatResponse | null;
  onEscalate?: (action: string, text: string, carryContext?: Record<string, unknown>) => void;
  onPrefillInput?: (text: string) => void;
  onUseScope?: (scope: UserSelectedScope) => void;
  liveBullets?: V9EvidenceBullet[];
  isStreaming?: boolean;
  onThinkDeeper?: () => void;
}) {
  if (message.role === 'user') {
    return (
      <div className="msg msg-user">
        <div className="msg-user-bubble">{message.content}</div>
      </div>
    );
  }

  // Prefer the live response, fall back to the v9_meta stored with history.
  const meta: V9Meta | null = v9
    ? {
        intent: v9.intent, confidence: v9.confidence, can_think_deeper: v9.can_think_deeper,
        remaining_gaps: v9.remaining_gaps, suggested_queries: v9.suggested_queries,
        suggestion: v9.suggestion, elapsed_ms: v9.elapsed_ms, cited_chunk_ids: v9.cited_chunk_ids,
        citation_map: v9.citation_map, scope_meta: v9.scope_meta, escalations: v9.escalations,
        scope_override: v9.scope_override, expansion_info: v9.expansion_info,
        search_result_set_id: v9.search_result_set_id,
        search_result_preview: v9.search_result_preview,
        search_result_throttled: v9.search_result_throttled,
      }
    : message.v9_meta || null;

  const citationMap = meta?.citation_map || {};
  const bullets = (meta?.evidence_bullets?.length ? meta.evidence_bullets : liveBullets) ?? [];
  const runScope = meta?.scope_override?.run_scope;
  const searchResultSetId = meta?.search_result_set_id;

  return (
    <div className="msg">
      <div className="answer">
        {/* Follow-up answers state which evidence set they are reasoning over. */}
        {meta?.intent === 'follow_up' && meta.scope_meta && <FollowUpScope scope={meta.scope_meta} />}

        {bullets.length > 0 && (
          <Findings bullets={bullets} onEvidenceClick={onEvidenceClick} streaming={isStreaming} />
        )}

        <AnswerText
          text={message.content}
          citationMap={citationMap}
          onEvidenceClick={onEvidenceClick}
        />

        {message.members && message.members.length > 0 && (
          <MembersList members={message.members} onEvidenceClick={onEvidenceClick} />
        )}

        {message.claims && message.claims.length > 0 && (
          <ClaimsList claims={message.claims} onEvidenceClick={onEvidenceClick} />
        )}

        {/* Chat ran a full page search — offer to open it in the Search tab. */}
        {searchResultSetId && onViewSearchResultSet && (
          <div className="notice">
            <Icon name="spark" size={16} />
            <div>
              <div>Chat ran a page-level search for this answer.</div>
              <button
                type="button"
                className="btn-link"
                onClick={() => onViewSearchResultSet(searchResultSetId)}
              >
                Open it in Search
                <Icon name="chevron-right" size={14} />
              </button>
            </div>
          </div>
        )}

        {meta?.search_result_throttled && (
          <div className="notice notice-amber">
            <Icon name="info" size={16} />
            <div>
              <div>{meta.search_result_throttled.message}</div>
              <div className="text-sm">Query: “{meta.search_result_throttled.query}”</div>
              {onOpenSearchTab && (
                <button type="button" className="btn-link" onClick={onOpenSearchTab}>
                  Run it in Search
                  <Icon name="chevron-right" size={14} />
                </button>
              )}
            </div>
          </div>
        )}

        {meta?.escalations && meta.escalations.length > 0 && (
          <EscalationBlock
            escalations={meta.escalations}
            onEscalate={onEscalate}
            onPrefillInput={onPrefillInput}
          />
        )}

        {meta?.intent === 'think_deeper' && (meta.suggested_queries?.length ?? 0) > 0 && (
          <div>
            <div className="eyebrow" style={{ marginBottom: 'var(--s-2)' }}>Try next</div>
            <div className="suggestions">
              {(meta.suggested_queries ?? []).map((q, i) => (
                <button
                  key={i}
                  type="button"
                  className="suggestion"
                  onClick={() =>
                    onEscalate
                      ? onEscalate('new_retrieval', q, { intent_hint: 'new_retrieval' })
                      : onPrefillInput?.(q)}
                >
                  <Icon name="search" size={14} />
                  <span className="truncate">{q}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {onThinkDeeper && (
          <div>
            {meta?.suggestion && (
              <p className="text-sm text-muted" style={{ marginBottom: 'var(--s-2)' }}>
                {meta.suggestion}
              </p>
            )}
            <button type="button" className="btn-secondary" onClick={onThinkDeeper}>
              <Icon name="deeper" size={15} />
              Think deeper
            </button>
            {(meta?.remaining_gaps?.length ?? 0) > 0 && (
              <p className="text-sm text-muted" style={{ marginTop: 'var(--s-2)' }}>
                Still open: {meta!.remaining_gaps.slice(0, 2).join('; ')}
              </p>
            )}
          </div>
        )}

        {/* Provenance + actions. One quiet row instead of four banners. */}
        {(meta || message.v6_stats) && !isStreaming && (
          <div className="answer-foot">
            {meta?.intent && <span>{INTENT_LABEL[meta.intent] ?? meta.intent}</span>}
            {meta && meta.elapsed_ms > 0 && (
              <>
                <span className="answer-foot-sep">·</span>
                <span>{duration(meta.elapsed_ms)}</span>
              </>
            )}
            {(meta?.cited_chunk_ids?.length ?? 0) > 0 && (
              <>
                <span className="answer-foot-sep">·</span>
                <span>{plural(meta!.cited_chunk_ids.length, 'citation')}</span>
              </>
            )}
            {runScope && (
              <>
                <span className="answer-foot-sep">·</span>
                <span title={meta?.expansion_info?.reason || undefined}>
                  searched {describeScope(
                    {
                      mode: runScope.mode as 'full_archive' | 'custom',
                      included_collection_ids: runScope.included_collection_ids,
                      included_document_ids: runScope.included_document_ids,
                    },
                    collections,
                  ).toLowerCase()}
                </span>
                {meta?.scope_override?.overridden && onUseScope && (
                  <button
                    type="button"
                    className="btn-link"
                    title="This query overrode the session scope. Keep it for the next questions too."
                    onClick={() => onUseScope({
                      mode: runScope.mode as 'full_archive' | 'custom',
                      included_collection_ids: runScope.included_collection_ids,
                      included_document_ids: runScope.included_document_ids,
                    })}
                  >
                    keep this scope
                  </button>
                )}
              </>
            )}
            {message.v6_stats && !meta && <V6StatsFooter stats={message.v6_stats} />}
            <div className="spacer" />
            <CopyAnswerButton content={message.content} citationMap={citationMap} />
          </div>
        )}
      </div>
    </div>
  );
}

// =============================================================================
// Findings (evidence bullets)
// =============================================================================

function Findings({
  bullets, onEvidenceClick, streaming,
}: {
  bullets: V9EvidenceBullet[];
  onEvidenceClick?: (evidence: EvidenceRef) => void;
  streaming?: boolean;
}) {
  const [open, setOpen] = useState(true);
  if (bullets.length === 0) return null;

  return (
    <div className="findings" data-open={open}>
      <button
        type="button"
        className="findings-head"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <Icon name={open ? 'chevron-down' : 'chevron-right'} size={14} />
        Evidence
        <span className="count">{bullets.length}</span>
        {streaming && <span className="count">· gathering</span>}
      </button>
      {open && (
        <div className="findings-list">
          {bullets.map((bullet, i) => (
            <div className="finding" key={i}>
              <span className="finding-marker" />
              <div className="finding-body">
                <span>{sanitizeBulletText(bullet.text)}</span>
                {bullet.doc_ids?.length > 0 && bullet.chunk_ids?.length > 0 && onEvidenceClick && (
                  <SourceLink
                    documentId={bullet.doc_ids[0]}
                    name={bullet.source_names?.[0]}
                    onClick={() => onEvidenceClick({
                      document_id: bullet.doc_ids[0],
                      // Open the quote's exact page when known (multi-page chunks).
                      pdf_page: bullet.quote_page ?? bullet.pages?.[0] ?? 1,
                      chunk_id: bullet.quote_chunk_id ?? bullet.chunk_ids[0],
                      quote: bullet.quote,
                      quote_page: bullet.quote_page ?? undefined,
                    })}
                  />
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * A link to the document behind a finding or citation.
 *
 * Always shows the document's name: payloads carry one only sometimes, so when
 * it is missing the name is looked up by id (cached process-wide) rather than
 * falling back to an anonymous "View document".
 */
function SourceLink({
  documentId, name, hint, onClick,
}: {
  documentId?: number;
  name?: string | null;
  hint?: string | null;
  onClick: () => void;
}) {
  const resolved = useDocumentName(documentId, name);
  const label = resolved ?? (documentId ? 'Opening document…' : 'Document');
  return (
    <button
      type="button"
      className="finding-source"
      onClick={onClick}
      disabled={!documentId}
      title={hint ? `“${hint}”` : label}
    >
      <Icon name="file" size={13} />
      <span className="finding-source-name">{label}</span>
    </button>
  );
}

// =============================================================================
// Supporting blocks
// =============================================================================

function FollowUpScope({ scope }: { scope: ScopeMeta }) {
  const entities = (scope.top_entities || []).slice(0, 5);
  const extra = (scope.top_entities || []).length - entities.length;
  return (
    <div className="notice">
      <Icon name="layers" size={16} />
      <div>
        <div>
          Answering from the evidence gathered for{' '}
          <em>“{scope.origin_query.length > 90 ? `${scope.origin_query.slice(0, 90)}…` : scope.origin_query}”</em>
        </div>
        <div className="text-sm text-muted">
          {plural(scope.chunk_count, 'passage')} · {plural(scope.document_count, 'document')}
          {scope.time_range ? ` · ${scope.time_range}` : ''}
        </div>
        {entities.length > 0 && (
          <div className="flex gap-sm" style={{ marginTop: 'var(--s-2)', flexWrap: 'wrap' }}>
            {entities.map((e, i) => <span className="chip" key={i}>{e.canonical_name}</span>)}
            {extra > 0 && <span className="count">+{extra} more</span>}
          </div>
        )}
      </div>
    </div>
  );
}

function EscalationBlock({
  escalations, onEscalate, onPrefillInput,
}: {
  escalations: EscalationOption[];
  onEscalate?: (action: string, text: string, carryContext?: Record<string, unknown>) => void;
  onPrefillInput?: (text: string) => void;
}) {
  const [taken, setTaken] = useState<string | null>(null);

  const handleClick = (opt: EscalationOption) => {
    if (opt.action === 'show_evidence') {
      document.querySelector('.findings')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      return;
    }
    setTaken(opt.label);
    if (opt.action === 'think_deeper') {
      onEscalate?.(opt.action, opt.prefilled_query || '', {
        entities: opt.carry_entities, intent_hint: 'think_deeper',
      });
    } else if (opt.action === 'new_retrieval') {
      if (opt.prefilled_query) {
        onEscalate?.(opt.action, opt.prefilled_query, {
          entities: opt.carry_entities, intent_hint: 'new_retrieval',
        });
      } else {
        setTaken(null);
        onPrefillInput?.('');
      }
    }
  };

  if (taken) {
    return <div className="notice"><Icon name="deeper" size={16} /> <span>Running: {taken}…</span></div>;
  }

  const icons: Record<string, 'deeper' | 'search' | 'files'> = {
    think_deeper: 'deeper',
    new_retrieval: 'search',
    show_evidence: 'files',
  };

  return (
    <div className="choices">
      <div className="choices-title">The evidence so far doesn&apos;t settle this. What next?</div>
      {escalations.map((opt, i) => (
        <button
          key={i}
          type="button"
          className={`choice${opt.recommended ? ' choice-recommended' : ''}`}
          onClick={() => handleClick(opt)}
        >
          <span className="choice-icon"><Icon name={icons[opt.action] ?? 'search'} size={17} /></span>
          <span style={{ minWidth: 0 }}>
            <span className="choice-label">
              {opt.label}
              {opt.recommended && <span className="chip chip-accent" style={{ marginLeft: 8 }}>suggested</span>}
            </span>
            <span className="choice-desc">{opt.description}</span>
          </span>
        </button>
      ))}
    </div>
  );
}

/** Clipboard write with a fallback for the contexts where the async API is
 *  refused (older browsers, some embedded views, an unfocused document). */
async function writeToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    /* fall through to the legacy path */
  }
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    document.body.removeChild(ta);
    return ok;
  } catch {
    return false;
  }
}

function CopyAnswerButton({
  content, citationMap,
}: {
  content: string;
  citationMap: Record<string, CitationDetail>;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    let text = (content || '').trim();
    const seen = new Set<string>();
    const lines: string[] = [];
    for (const [key, det] of Object.entries(citationMap)) {
      const label = det.label || key;
      if (!det.document_id || seen.has(label)) continue;
      seen.add(label);
      let url = api.getDocumentPdfUrl(det.document_id);
      if (url.startsWith('/')) url = `${window.location.origin}${url}`;
      if (det.page != null) url += `#page=${det.page}`;
      lines.push(`- ${label}: ${url}`);
    }
    if (lines.length > 0) text += `\n\nSources:\n${lines.join('\n')}`;
    if (await writeToClipboard(text)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } else {
      toast('Could not copy. Your browser blocked clipboard access.');
    }
  };

  if (!content?.trim()) return null;
  return (
    <button
      type="button"
      className="btn-link"
      onClick={handleCopy}
      title="Copy this answer, including source links"
    >
      <Icon name={copied ? 'check' : 'copy'} size={14} />
      {copied ? 'Copied' : 'Copy'}
    </button>
  );
}

function MembersList({
  members, onEvidenceClick,
}: {
  members: ChatMember[];
  onEvidenceClick?: (evidence: EvidenceRef) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const shown = expanded ? members : members.slice(0, 10);

  const open = (cit: ChatCitation) => {
    if (onEvidenceClick && cit.document_id) {
      onEvidenceClick({
        document_id: cit.document_id,
        pdf_page: cit.page_number || 1,
        chunk_id: cit.chunk_id,
        quote: cit.quote,
      });
    }
  };

  return (
    <div className="findings">
      <div className="findings-head">
        <Icon name="files" size={14} />
        Members identified
        <span className="count">{members.length}</span>
      </div>
      <div className="findings-list">
        {shown.map((member, i) => (
          <div className="finding" key={i}>
            <span className="finding-marker" />
            <div className="finding-body">
              <span>{member.name}</span>
              {member.citations && member.citations.length > 0 && (
                <span className="flex gap-sm">
                  {member.citations.slice(0, 2).map((cit, j) => (
                    <SourceLink
                      key={j}
                      documentId={cit.document_id}
                      name={cit.source_name}
                      hint={cit.quote}
                      onClick={() => open(cit)}
                    />
                  ))}
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
      {members.length > 10 && (
        <div className="hit-foot">
          <button type="button" className="btn-link" onClick={() => setExpanded(!expanded)}>
            {expanded ? 'Show fewer' : `Show all ${members.length}`}
          </button>
        </div>
      )}
    </div>
  );
}

const CLAIM_MARK: Record<string, { icon: 'check' | 'minus' | 'help'; cls: string }> = {
  supported: { icon: 'check', cls: 'chip-green' },
  partial: { icon: 'minus', cls: 'chip-amber' },
  unsupported: { icon: 'help', cls: '' },
};

function ClaimsList({
  claims, onEvidenceClick,
}: {
  claims: ChatClaim[];
  onEvidenceClick?: (evidence: EvidenceRef) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const shown = expanded ? claims : claims.slice(0, 5);

  const open = (cit: ChatCitation) => {
    if (onEvidenceClick && cit.document_id) {
      onEvidenceClick({
        document_id: cit.document_id,
        pdf_page: cit.page_number || 1,
        chunk_id: cit.chunk_id,
        quote: cit.quote,
      });
    }
  };

  const supported = claims.filter((c) => c.confidence === 'supported').length;
  const partial = claims.filter((c) => c.confidence === 'partial').length;

  return (
    <div className="findings">
      <div className="findings-head">
        <Icon name="check" size={14} />
        Evidence-backed claims
        <span className="count">{claims.length}</span>
        <div className="spacer" />
        {supported > 0 && <span className="chip chip-green">{supported} supported</span>}
        {partial > 0 && <span className="chip chip-amber">{partial} partial</span>}
      </div>
      <div className="findings-list">
        {shown.map((claim, i) => {
          const mark = CLAIM_MARK[claim.confidence] ?? CLAIM_MARK.unsupported;
          return (
            <div className="finding" key={i}>
              <span className={`chip ${mark.cls}`} title={claim.confidence}>
                <Icon name={mark.icon} size={12} />
              </span>
              <div className="finding-body">
                <span>{claim.text}</span>
                {claim.citations.length > 0 && (
                  <span className="flex gap-sm" style={{ flexWrap: 'wrap' }}>
                    {claim.citations.map((cit, j) => (
                      <SourceLink
                        key={j}
                        documentId={cit.document_id}
                        name={cit.source_name}
                        hint={cit.quote}
                        onClick={() => open(cit)}
                      />
                    ))}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
      {claims.length > 5 && (
        <div className="hit-foot">
          <button type="button" className="btn-link" onClick={() => setExpanded(!expanded)}>
            {expanded ? 'Show fewer' : `Show all ${claims.length}`}
          </button>
        </div>
      )}
    </div>
  );
}

/** Legacy V6/V7 runs still carry these stats. */
function V6StatsFooter({ stats }: { stats: V6Stats }) {
  return (
    <span title={`Task type: ${stats.task_type}`}>
      {stats.rounds_executed} rounds · {duration(stats.elapsed_ms)}
    </span>
  );
}
