'use client';

import { useState, useRef, useEffect } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type {
  Session, ChatMessage, ChatClaim, ChatMember, V6Stats, EvidenceRef, ChatCitation,
  V9ChatResponse, V9Meta, V9ProgressEvent, V9EvidenceBullet, CitationDetail,
  ScopeMeta, EscalationOption, UserSelectedScope, CollectionNode,
} from '@/types/api';
import { scopeFingerprint } from '@/lib/scope';

/** Map technical progress messages to user-friendly text. */
function toUserFriendlyProgress(step: V9ProgressEvent): string {
  switch (step.step) {
    case 'tool_call': {
      const tool = (step.details?.tool as string) || '';
      if (tool.startsWith('search_chunks') || tool === 'search_chunks') return 'Searching documents...';
      if (tool.startsWith('fetch_chunks') || tool === 'fetch_chunks') return 'Reading documents...';
      if (tool.startsWith('expand_entities') || tool === 'expand_entities') return 'Resolving identities...';
      if (tool.startsWith('alias_index') || tool.includes('alias')) return 'Looking up references...';
      return 'Gathering evidence...';
    }
    case 'turn_start':
      return 'Investigating...';
    case 'investigation':
      return 'Searching and analyzing...';
    case 'entity_resolution':
      return 'Resolving entities...';
    case 'synthesis':
      return 'Synthesizing answer...';
    case 'evidence_update':
      return 'Found evidence...';
    case 'routing':
      return 'Understanding your question...';
    default:
      if (step.message.toLowerCase().includes('search')) return 'Searching...';
      if (step.message.toLowerCase().includes('round')) return 'Analyzing evidence...';
      if (step.message.toLowerCase().includes('fetch')) return 'Reading documents...';
      if (step.message.toLowerCase().includes('synthes')) return 'Synthesizing answer...';
      return 'Investigating...';
  }
}

interface ConversationProps {
  session: Session | null;
  onV9Response?: (response: V9ChatResponse | null) => void;
  onProcessingChange?: (processing: boolean) => void;
  onEvidenceClick?: (evidence: EvidenceRef) => void;
  onProgressUpdate?: (steps: V9ProgressEvent[], bullets: V9EvidenceBullet[]) => void;
  /** Ref setter so parent can pre-fill the input (used by escalation "Start new search" action). */
  inputRef?: React.RefObject<HTMLInputElement | null>;
  // Scope bar props
  activeScope?: UserSelectedScope | null;
  lastUsedScope?: UserSelectedScope | null;
  collections?: CollectionNode[];
  hasDraftChanges?: boolean;
  onEditScope?: () => void;
  onQuerySent?: (scopeSent: UserSelectedScope) => void;
  onMakeActiveScope?: (scope: UserSelectedScope) => void;
  isProcessing?: boolean;
  processingSessionId?: number | null;
}

export function Conversation({
  session, onV9Response, onProcessingChange, onEvidenceClick, onProgressUpdate,
  activeScope, lastUsedScope, collections, hasDraftChanges, onEditScope, onQuerySent, onMakeActiveScope,
  isProcessing = false, processingSessionId = null,
}: ConversationProps) {
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);
  const [lastV9, setLastV9] = useState<V9ChatResponse | null>(null);
  const [progressSteps, setProgressSteps] = useState<V9ProgressEvent[]>([]);
  const [evidenceBullets, setEvidenceBullets] = useState<V9EvidenceBullet[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const queryClient = useQueryClient();
  const abortControllerRef = useRef<AbortController | null>(null);
  const abortRef = useRef(false);

  // Toast state for scope actions
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  useEffect(() => {
    if (toastMessage) {
      const timer = setTimeout(() => setToastMessage(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [toastMessage]);

  // Fetch chat history (reuses existing endpoint — V9 now persists to research_messages)
  const { data: messages, isLoading } = useQuery({
    queryKey: ['chatHistory', session?.id],
    queryFn: () => api.getChatHistory(session!.id),
    enabled: !!session,
  });

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Reset V9 state when session changes
  useEffect(() => {
    setLastV9(null);
    onV9Response?.(null);
  }, [session?.id]);

  // ── Submit: send a question via V9 ─────────────────────────────────
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!session || !input.trim() || isSending) return;
    await sendV9(input.trim());
  };

  const sendV9 = async (text: string, action: 'default' | 'think_deeper' = 'default', carryContext?: Record<string, unknown>) => {
    if (!session || isSending) return;

    abortRef.current = false;
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setInput('');
    setIsSending(true);
    setSendError(null);
    setProgressSteps([]);
    setEvidenceBullets([]);
    onProcessingChange?.(true, session.id);
    onV9Response?.(null);
    onProgressUpdate?.([], []);
    onQuerySent?.(activeScope || { mode: 'full_archive' });

    // Accumulate in local vars so callbacks have latest state
    const steps: V9ProgressEvent[] = [];
    const bullets: V9EvidenceBullet[] = [];

    try {
      await api.sendV9MessageStreaming(
        session.id,
        text,
        action,
        {
        onProgress: (event) => {
            if (abortRef.current) return;
            steps.push(event);
            const copy = [...steps];
            setProgressSteps(copy);
            onProgressUpdate?.(copy, bullets);
          },
          onEvidenceUpdate: (event) => {
            if (abortRef.current) return;
            const newBullets = event.details?.bullets || [];
            bullets.push(...newBullets);
            const copy = [...bullets];
            setEvidenceBullets(copy);
            onProgressUpdate?.(steps, copy);
          },
          onResult: (response) => {
            if (abortRef.current) return;
            setLastV9(response);
            onV9Response?.(response);
            queryClient.invalidateQueries({ queryKey: ['chatHistory', session.id] });
          },
        onError: (error) => {
            if (abortRef.current) return;
          setSendError(error);
          },
        },
        controller.signal,
        carryContext,
      );
    } catch (err: unknown) {
      if (abortRef.current) return;
      let msg = err instanceof Error ? err.message : String(err);
      if (msg === 'Failed to fetch' || msg.includes('fetch')) {
        msg =
          'Failed to fetch — check network and CORS. Backend health: https://api.fridayarchive.org/health (API routes use /api/...).';
      }
      setSendError(msg);
    } finally {
      if (!abortRef.current) {
        setIsSending(false);
        onProcessingChange?.(false);
      }
      abortControllerRef.current = null;
    }
  };

  const handleStop = () => {
    abortRef.current = true;
    abortControllerRef.current?.abort();
    setIsSending(false);
    setSendError(null);
    setProgressSteps([]);
    setEvidenceBullets([]);
    onProcessingChange?.(false);
    onProgressUpdate?.([], []);
  };

  const handleMakeActive = (scope: UserSelectedScope) => {
    onMakeActiveScope?.(scope);
    setToastMessage('Scope updated for next queries.');
  };

  const handleThinkDeeper = () => {
    if (!lastV9?.can_think_deeper) return;
    // Re-send the last user question with think_deeper action
    const lastUserMsg = messages?.filter(m => m.role === 'user').pop();
    if (lastUserMsg) {
      sendV9(lastUserMsg.content, 'think_deeper');
    }
  };

  if (!session) {
    return (
      <div className="pane-content splash-content">
        <div className="splash-hero">
          <h2 className="splash-title">Friday Research Console</h2>
          <p className="splash-subtitle">Select a session to start, or create a new one from the sidebar.</p>
        </div>
        <div className="splash-section">
          <h3 className="splash-section-title">How to use</h3>
          <ul className="splash-list">
            <li>Ask research questions in plain language; Friday searches archived documents and resolves entities (people, places, codenames).</li>
            <li>Use <strong>Scope</strong> to limit searches to specific collections or documents, or run over the full archive.</li>
            <li>Follow-up questions run against the evidence from your last search; use &ldquo;New search&rdquo; when you want a fresh retrieval.</li>
            <li>Click any citation in an answer to open the source document and page in the viewer.</li>
          </ul>
        </div>
        <div className="splash-section">
          <h3 className="splash-section-title">Features</h3>
          <ul className="splash-list">
            <li>Entity-aware search with concordance and alias resolution</li>
            <li>Cited answers with clickable evidence links</li>
            <li>Session scope (full archive or custom collections/documents)</li>
            <li>Investigation panel with progress and evidence bullets</li>
            <li>Think Deeper to extend an investigation with more steps</li>
          </ul>
        </div>
        {collections && collections.length > 0 && (
          <div className="splash-section">
            <h3 className="splash-section-title">Indexed collections</h3>
            <p className="splash-collections-intro">The following collections are available for search.</p>
            <ul className="splash-collections-list">
              {collections.map((c) => (
                <li key={c.id}>
                  <span className="splash-collection-name">{c.title || c.slug}</span>
                  {c.document_count != null && (
                    <span className="splash-collection-meta"> — {c.document_count} document{c.document_count !== 1 ? 's' : ''}</span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  }

  return (
    <>
      <div className="pane-content">
        {isLoading ? (
          <div className="loading">Loading conversation...</div>
        ) : messages && messages.length > 0 ? (
          <>
            {messages.map((message, idx) => (
              <ChatBubble
                key={message.id}
                message={message}
                onEvidenceClick={onEvidenceClick}
                v9={idx === messages.length - 1 && message.role === 'assistant' ? lastV9 : null}
                onMakeActiveScope={handleMakeActive}
                collections={collections}
                onEscalate={(action, text, carryContext) => {
                  if (action === 'think_deeper') {
                    sendV9(text, 'think_deeper', carryContext);
                  } else if (action === 'new_retrieval') {
                    sendV9(text, 'default', carryContext);
                  }
                }}
                onPrefillInput={(text) => {
                  setInput(text);
                }}
              />
            ))}
            <div ref={messagesEndRef} />
          </>
        ) : (
          <div className="empty-state">
            <p>Ask a research question</p>
            <p className="text-sm">V9 will search archives, resolve entities, and generate cited answers</p>
          </div>
        )}

        {isSending && session?.id === processingSessionId && (
          <div className="chat-thinking">
            <div className="thinking-indicator">
              <span className="thinking-dot"></span>
              <span className="thinking-dot"></span>
              <span className="thinking-dot"></span>
            </div>
            <span className="thinking-text">
              {progressSteps.length > 0
                ? toUserFriendlyProgress(progressSteps[progressSteps.length - 1])
                : 'Investigating...'}
            </span>
            {evidenceBullets.length > 0 && (
              <span className="thinking-bullet-count">
                {evidenceBullets.length} evidence bullet{evidenceBullets.length !== 1 ? 's' : ''} found
              </span>
            )}
          </div>
        )}

        {sendError && (
          <div className="card" style={{ borderColor: 'var(--color-danger)' }}>
            <p style={{ color: 'var(--color-danger)' }}>
              Failed to get answer: {sendError}
            </p>
          </div>
        )}

        {/* Think Deeper button */}
        {lastV9?.can_think_deeper && !isSending && (
          <div style={{ padding: 'var(--spacing-sm) var(--spacing-md)', textAlign: 'center' }}>
            <button
              className="btn-secondary"
              onClick={handleThinkDeeper}
              style={{ fontSize: '13px' }}
            >
              Think Deeper — extend investigation
            </button>
            {lastV9.remaining_gaps.length > 0 && (
              <div className="text-sm text-muted" style={{ marginTop: '4px' }}>
                Gaps: {lastV9.remaining_gaps.slice(0, 2).join('; ')}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Input area */}
      <div className="chat-input-container">
        {toastMessage && (
          <div className="scope-toast">{toastMessage}</div>
        )}
        <form onSubmit={handleSubmit}>
          <div className="flex gap-sm">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={lastV9?.suggestion || 'Ask a research question...'}
              disabled={isSending}
              className="chat-input"
            />
            {isSending && session?.id === processingSessionId ? (
              <button
                type="button"
                className="btn-stop"
                onClick={handleStop}
                style={{
                  background: 'var(--color-danger, #dc3545)',
                  color: '#fff',
                  border: 'none',
                  borderRadius: 'var(--radius)',
                  padding: '0 var(--spacing-md)',
                  cursor: 'pointer',
                  fontWeight: 600,
                  whiteSpace: 'nowrap',
                }}
              >
                Stop
              </button>
            ) : (
              <button
                type="submit"
                className="btn-primary"
                disabled={!input.trim()}
              >
                Ask
              </button>
            )}
          </div>
        </form>
      </div>
    </>
  );
}

function ChatBubble({
  message,
  onEvidenceClick,
  v9,
  onEscalate,
  onPrefillInput,
  onMakeActiveScope,
  collections,
}: {
  message: ChatMessage;
  onEvidenceClick?: (evidence: EvidenceRef) => void;
  v9?: V9ChatResponse | null;
  onEscalate?: (action: string, text: string, carryContext?: Record<string, unknown>) => void;
  onPrefillInput?: (text: string) => void;
  onMakeActiveScope?: (scope: UserSelectedScope) => void;
  collections?: CollectionNode[];
}) {
  const isUser = message.role === 'user';

  // Dismissed state for override warning banner (must be before early returns)
  const [overrideDismissed, setOverrideDismissed] = useState(false);
  
  if (isUser) {
    return (
      <div className="chat-message chat-message-user">
        <div className="chat-message-content">{message.content}</div>
      </div>
    );
  }
  
  // V9 metadata: prefer live response, fall back to stored v9_meta from history
  const v9meta: V9Meta | null = v9 ? {
    intent: v9.intent,
    confidence: v9.confidence,
    can_think_deeper: v9.can_think_deeper,
    remaining_gaps: v9.remaining_gaps,
    suggestion: v9.suggestion,
    elapsed_ms: v9.elapsed_ms,
    cited_chunk_ids: v9.cited_chunk_ids,
    citation_map: v9.citation_map,
    scope_meta: v9.scope_meta,
    escalations: v9.escalations,
    scope_override: v9.scope_override,
    expansion_info: v9.expansion_info,
  } : message.v9_meta || null;

  // Citation map for inline citation linking
  const citationMap = v9meta?.citation_map || {};

  const isV9 = !!v9meta;
  const intent = v9meta?.intent;
  const scopeMeta = v9meta?.scope_meta;
  const escalations = v9meta?.escalations;
  const scopeOverride = v9meta?.scope_override;
  const expansionInfo = v9meta?.expansion_info;
  
  return (
    <div className="chat-message chat-message-assistant">
      <div className="chat-answer">
        {/* V9 intent badge */}
        {isV9 && intent && (
          <div style={{ marginBottom: '6px', display: 'flex', gap: '6px', alignItems: 'center' }}>
            <IntentBadge intent={intent} />
          </div>
        )}

        {/* Scope-used header for new_retrieval */}
        {intent === 'new_retrieval' && scopeOverride && (
          <UsedScopeDisplay scopeOverride={scopeOverride} collections={collections} />
        )}

        {/* Inline scope override banner */}
        {intent === 'new_retrieval' && scopeOverride?.overridden && !overrideDismissed && (
          <div className="scope-inline-banner">
            <div>
              <strong>Scope for this query: </strong>
              {scopeOverride.run_scope?.mode === 'full_archive'
                ? 'Full archive'
                : (() => {
                    const colIds = scopeOverride.run_scope?.included_collection_ids || [];
                    const names = colIds.slice(0, 3).map(id => {
                      const col = collections?.find(c => c.id === id);
                      return col ? (col.title || col.slug) : `#${id}`;
                    });
                    return names.join(', ') + (colIds.length > 3 ? ` +${colIds.length - 3} more` : '');
                  })()
              }
              <span style={{ color: 'var(--color-text-muted)', marginLeft: 4 }}>(not saved)</span>
            </div>
            {scopeOverride.run_scope?.reason && (
              <div style={{ fontSize: '11px', color: 'var(--color-text-muted)', marginTop: 2 }}>
                {scopeOverride.run_scope.reason}
              </div>
            )}
            <div style={{ marginTop: 6, display: 'flex', gap: 8 }}>
              <button
                onClick={() => {
                  if (scopeOverride.run_scope) {
                    onMakeActiveScope?.({
                      mode: scopeOverride.run_scope.mode,
                      included_collection_ids: scopeOverride.run_scope.included_collection_ids,
                      included_document_ids: scopeOverride.run_scope.included_document_ids,
                    });
                  }
                  setOverrideDismissed(true);
                }}
                className="btn-primary"
                style={{ fontSize: '11px', padding: '3px 10px' }}
              >
                Make active
              </button>
              <button
                onClick={() => setOverrideDismissed(true)}
                className="btn-secondary"
                style={{ fontSize: '11px', padding: '3px 10px' }}
              >
                Dismiss
              </button>
            </div>
          </div>
        )}

        {/* Expansion status line */}
        {intent === 'new_retrieval' && expansionInfo && (
          <div style={{
            fontSize: '11px',
            color: 'var(--color-text-secondary, #888)',
            marginBottom: '6px',
          }}>
            Concordance expansion: {expansionInfo.triggered
              ? `Triggered${expansionInfo.reason ? ` (${expansionInfo.reason.substring(0, 60)})` : ''}`
              : 'Not triggered'
            }
          </div>
        )}

        {/* Scope banner for follow-up answers */}
        {intent === 'follow_up' && scopeMeta && (
          <ScopeBanner scope={scopeMeta} />
        )}

        <RichAnswerText
          text={message.content}
          citationMap={citationMap}
          onEvidenceClick={onEvidenceClick}
        />
        
        {/* Show members if this is a roster-style answer */}
        {message.members && message.members.length > 0 && (
          <MembersList members={message.members} onEvidenceClick={onEvidenceClick} />
        )}
        
        {/* Show claims with citations */}
        {message.claims && message.claims.length > 0 && (
          <ClaimsList claims={message.claims} onEvidenceClick={onEvidenceClick} />
        )}
        
        {/* Escalation block when follow-up evidence is insufficient */}
        {escalations && escalations.length > 0 && (
          <EscalationBlock
            escalations={escalations}
            onEscalate={onEscalate}
            onPrefillInput={onPrefillInput}
          />
        )}
        
        {/* V9 footer with elapsed time */}
        {isV9 && v9meta && (
          <div className="v6-stats">
            <div className="v6-stats-summary">
              <span className="stats-brief">
                {v9meta.elapsed_ms > 0 && `${(v9meta.elapsed_ms / 1000).toFixed(1)}s`}
                {v9meta.cited_chunk_ids.length > 0 && ` · ${v9meta.cited_chunk_ids.length} citations`}
              </span>
            </div>
          </div>
        )}

        {/* Legacy V6/V7 Stats footer */}
        {!isV9 && message.v6_stats && (
          <V6StatsFooter stats={message.v6_stats} />
        )}
      </div>
    </div>
  );
}

function IntentBadge({ intent }: { intent: string }) {
  const config: Record<string, { label: string; color: string }> = {
    new_retrieval: { label: 'New search', color: 'var(--color-primary, #4a90d9)' },
    follow_up: { label: 'Follow-up', color: '#6c757d' },
    think_deeper: { label: 'Think deeper', color: '#e67e22' },
  };
  const c = config[intent] || { label: intent, color: '#888' };
  return (
    <span style={{
      display: 'inline-block',
      fontSize: '11px',
      fontWeight: 600,
      padding: '2px 8px',
      borderRadius: '10px',
      background: c.color,
      color: '#fff',
    }}>
      {c.label}
    </span>
  );
}

function ConfidenceBadge({ confidence }: { confidence: string }) {
  const colors: Record<string, string> = {
    high: '#28a745',
    medium: '#ffc107',
    low: '#dc3545',
  };
  return (
    <span style={{
      fontSize: '11px',
      color: colors[confidence] || '#888',
      fontWeight: 500,
    }}>
      {confidence} confidence
    </span>
  );
}


/**
 * Scope banner: shows what evidence set the follow-up is operating within.
 */
function ScopeBanner({ scope }: { scope: ScopeMeta }) {
  const MAX_ENTITIES = 5;
  const entities = scope.top_entities || [];
  const visibleEntities = entities.slice(0, MAX_ENTITIES);
  const extraCount = entities.length - MAX_ENTITIES;

  return (
    <div style={{
      marginBottom: '10px',
      padding: '8px 12px',
      borderLeft: '3px solid var(--color-primary, #4a90d9)',
      background: 'rgba(74, 144, 217, 0.06)',
      borderRadius: '0 6px 6px 0',
      fontSize: '12px',
      lineHeight: '1.5',
    }}>
      <div style={{ color: 'var(--color-text-secondary, #999)', fontWeight: 500, marginBottom: '2px' }}>
        Scope: Evidence from{scope.origin_run_id ? ` Run #${scope.origin_run_id}` : ''} &mdash;{' '}
        <span style={{ color: 'var(--color-text-primary, #e0e0e0)', fontStyle: 'italic' }}>
          &ldquo;{scope.origin_query.length > 80
            ? scope.origin_query.slice(0, 80) + '...'
            : scope.origin_query}&rdquo;
        </span>
      </div>
      <div style={{ color: 'var(--color-text-secondary, #999)' }}>
        {scope.chunk_count} chunks &middot; {scope.document_count} document{scope.document_count !== 1 ? 's' : ''}
        {scope.time_range && <> &middot; {scope.time_range}</>}
      </div>
      {visibleEntities.length > 0 && (
        <div style={{ marginTop: '4px', display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
          {visibleEntities.map((ent, i) => (
            <span key={i} style={{
              display: 'inline-block',
              padding: '1px 7px',
              borderRadius: '10px',
              background: 'rgba(255,255,255,0.08)',
              border: '1px solid rgba(255,255,255,0.12)',
              fontSize: '11px',
              color: 'var(--color-text-primary, #e0e0e0)',
            }}>
              {ent.canonical_name}
            </span>
          ))}
          {extraCount > 0 && (
            <span style={{
              display: 'inline-block',
              padding: '1px 7px',
              fontSize: '11px',
              color: 'var(--color-text-secondary, #999)',
            }}>
              +{extraCount} more
            </span>
          )}
        </div>
      )}
    </div>
  );
}


/**
 * Escalation block: structured "what now" actions when follow-up evidence is insufficient.
 */
function EscalationBlock({
  escalations,
  onEscalate,
  onPrefillInput,
}: {
  escalations: EscalationOption[];
  onEscalate?: (action: string, text: string, carryContext?: Record<string, unknown>) => void;
  onPrefillInput?: (text: string) => void;
}) {
  const handleClick = (opt: EscalationOption) => {
    if (opt.action === 'think_deeper') {
      onEscalate?.(opt.action, opt.prefilled_query || '', {
        entities: opt.carry_entities,
        intent_hint: opt.description,
      });
    } else if (opt.action === 'new_retrieval') {
      // Send the prefilled query as a new search immediately
      if (opt.prefilled_query) {
        onEscalate?.(opt.action, opt.prefilled_query, {
          entities: opt.carry_entities,
          intent_hint: opt.description,
        });
      } else {
        // No prefilled query — fall back to pre-filling the input for manual entry
        onPrefillInput?.('');
      }
    } else if (opt.action === 'show_evidence') {
      // Scroll to the claims/citations section of the current message
      const claimsEl = document.querySelector('.chat-claims');
      if (claimsEl) {
        claimsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  };

  const actionIcons: Record<string, string> = {
    think_deeper: '\u{1F50D}',   // magnifying glass
    new_retrieval: '\u{1F504}',  // arrows counterclockwise
    show_evidence: '\u{1F4CB}',  // clipboard
  };

  return (
    <div style={{
      marginTop: '12px',
      padding: '10px 14px',
      border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: '8px',
      background: 'rgba(255,255,255,0.03)',
    }}>
      <div style={{
        fontSize: '12px',
        color: 'var(--color-text-secondary, #999)',
        marginBottom: '8px',
        fontWeight: 500,
      }}>
        I can't answer immediately, what would you like to do?
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
        {escalations.map((opt, i) => (
          <button
            key={i}
            onClick={() => handleClick(opt)}
            style={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: '8px',
              padding: '8px 12px',
              border: opt.recommended
                ? '1px solid var(--color-primary, #4a90d9)'
                : '1px solid rgba(255,255,255,0.1)',
              borderRadius: '6px',
              background: opt.recommended
                ? 'rgba(74, 144, 217, 0.1)'
                : 'rgba(255,255,255,0.02)',
              cursor: 'pointer',
              textAlign: 'left',
              width: '100%',
              transition: 'background 0.15s',
            }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLButtonElement).style.background = opt.recommended
                ? 'rgba(74, 144, 217, 0.18)'
                : 'rgba(255,255,255,0.06)';
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLButtonElement).style.background = opt.recommended
                ? 'rgba(74, 144, 217, 0.1)'
                : 'rgba(255,255,255,0.02)';
            }}
          >
            <span style={{ fontSize: '14px', lineHeight: '1.4', flexShrink: 0 }}>
              {actionIcons[opt.action] || ''}
            </span>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{
                  fontSize: '13px',
                  fontWeight: 600,
                  color: '#888',
                }}>
                  {opt.label}
                </span>
                {opt.recommended && (
                  <span style={{
                    fontSize: '10px',
                    fontWeight: 600,
                    padding: '1px 6px',
                    borderRadius: '8px',
                    background: 'var(--color-primary, #4a90d9)',
                    color: '#fff',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                  }}>
                    recommended
                  </span>
                )}
              </div>
              <div style={{
                fontSize: '11px',
                color: 'var(--color-text-secondary, #999)',
                marginTop: '2px',
                lineHeight: '1.4',
              }}>
                {opt.description}
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

/** Strip confidence suffix like (high), (medium), (low) from citation labels. */
function stripConfidenceFromLabel(label: string): string {
  return label.replace(/\s*\((?:high|medium|low)\)$/i, '').trim();
}

/**
 * Renders answer text with inline citations as clickable links.
 *
 * Parses patterns like [Vassiliev P4, Venona P42] from the answer text.
 * Each citation label is looked up in the citation_map to resolve to a
 * document_id + page, then rendered as a clickable button that opens
 * the EvidenceViewer.
 * Strips confidence suffixes like (high) from display.
 */
function RichAnswerText({
  text,
  citationMap,
  onEvidenceClick,
}: {
  text: string;
  citationMap: Record<string, CitationDetail>;
  onEvidenceClick?: (evidence: EvidenceRef) => void;
}) {
  const hasCitationMap = Object.keys(citationMap).length > 0;

  // If no citation map available, fall back to plain text rendering
  if (!hasCitationMap) {
    return <div className="chat-answer-text">{text}</div>;
  }

  const resolveCitation = (label: string): CitationDetail | undefined =>
    citationMap[label] ?? citationMap[stripConfidenceFromLabel(label)];

  // Parse text to find [...] citation brackets and split into segments
  const segments: Array<{ type: 'text'; value: string } | { type: 'citations'; labels: string[] }> = [];
  // Match [...] blocks that contain citation-like content
  const citationBlockRegex = /\[([^\]]+)\]/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = citationBlockRegex.exec(text)) !== null) {
    const bracketContent = match[1];
    // Split by comma to get individual citations
    const labels = bracketContent.split(',').map(s => s.trim()).filter(Boolean);
    // Check if at least one label matches the citation map (with or without confidence suffix)
    const hasMatch = labels.some(label => resolveCitation(label));

    if (hasMatch) {
      // Add preceding text
      if (match.index > lastIndex) {
        segments.push({ type: 'text', value: text.slice(lastIndex, match.index) });
      }
      segments.push({ type: 'citations', labels });
      lastIndex = match.index + match[0].length;
    }
    // If no match, leave it as plain text (will be included in next text segment)
  }

  // Add remaining text
  if (lastIndex < text.length) {
    segments.push({ type: 'text', value: text.slice(lastIndex) });
  }

  const handleCitationClick = (detail: CitationDetail) => {
    if (!onEvidenceClick || !detail.document_id) return;
    onEvidenceClick({
      document_id: detail.document_id,
      pdf_page: detail.page || 1,
      chunk_id: detail.chunk_id,
    });
  };

  return (
    <div className="chat-answer-text">
      {segments.map((seg, i) => {
        if (seg.type === 'text') {
          return <span key={i}>{seg.value}</span>;
        }
        // Render citation buttons
        return (
          <span key={i} className="inline-citation-group">
            {seg.labels.map((label, j) => {
              const detail = resolveCitation(label);
              if (detail && detail.document_id) {
                const displayLabel = (detail.label && /^\d+$/.test(label))
                  ? detail.label
                  : (detail.label || stripConfidenceFromLabel(label));
                return (
                  <button
                    key={j}
                    className="inline-citation-link"
                    onClick={() => handleCitationClick(detail)}
                    title={`Open ${displayLabel} in document viewer`}
                  >
                    {displayLabel}
                  </button>
                );
              }
              // Label not in map — render as plain text (strip confidence for display)
              return <span key={j} className="inline-citation-unresolved">[{stripConfidenceFromLabel(label) || label}]</span>;
            })}
          </span>
        );
      })}
    </div>
  );
}

function MembersList({ members, onEvidenceClick }: { members: ChatMember[]; onEvidenceClick?: (evidence: EvidenceRef) => void }) {
  const [expanded, setExpanded] = useState(false);
  const displayCount = expanded ? members.length : 10;
  
  const handleCitationClick = (citation: ChatCitation) => {
    if (onEvidenceClick && citation.document_id) {
      onEvidenceClick({
        document_id: citation.document_id,
        pdf_page: citation.page_number || 1,
        chunk_id: citation.chunk_id,
        quote: citation.quote,
      });
    }
  };
  
  return (
    <div className="chat-members">
      <div className="chat-section-header">
        <span className="chat-section-title">Members Identified ({members.length})</span>
      </div>
      <ul className="members-list">
        {members.slice(0, displayCount).map((member, idx) => (
          <li key={idx} className="member-item">
            <span className="member-bullet">·</span>
            <span className="member-name">{member.name}</span>
            {member.citations && member.citations.length > 0 && (
              <span className="member-citations">
                {member.citations.slice(0, 2).map((cit, citIdx) => (
                  <button
                    key={citIdx}
                    className="citation-btn"
                    onClick={() => handleCitationClick(cit)}
                    title={cit.quote || 'View source'}
                    disabled={!cit.document_id}
                  >
                    {cit.source_name ? cit.source_name.substring(0, 15) : `p.${cit.page_number || '?'}`}
                  </button>
                ))}
              </span>
            )}
          </li>
        ))}
      </ul>
      {members.length > 10 && (
        <button 
          className="btn-link"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? 'Show less' : `Show all ${members.length} members`}
        </button>
      )}
    </div>
  );
}

function ClaimsList({ claims, onEvidenceClick }: { claims: ChatClaim[]; onEvidenceClick?: (evidence: EvidenceRef) => void }) {
  const [expanded, setExpanded] = useState(false);
  const displayCount = expanded ? claims.length : 5;
  
  const handleCitationClick = (citation: ChatCitation) => {
    if (onEvidenceClick && citation.document_id) {
      onEvidenceClick({
        document_id: citation.document_id,
        pdf_page: citation.page_number || 1,
        chunk_id: citation.chunk_id,
        quote: citation.quote,
      });
    }
  };

  const supportedCount = claims.filter(c => c.confidence === 'supported').length;
  const partialCount = claims.filter(c => c.confidence === 'partial').length;
  
  return (
    <div className="chat-claims">
      <div className="chat-section-header">
        <span className="chat-section-title">Evidence-Backed Claims ({claims.length})</span>
        <span className="claims-summary-badges">
          {supportedCount > 0 && (
            <span className="claims-badge claims-badge-supported">{supportedCount} supported</span>
          )}
          {partialCount > 0 && (
            <span className="claims-badge claims-badge-partial">{partialCount} partial</span>
          )}
        </span>
      </div>
      <ul className="claims-list">
        {claims.slice(0, displayCount).map((claim, idx) => (
          <li key={idx} className={`claim-item claim-item-${claim.confidence}`}>
            <span className={`claim-confidence claim-${claim.confidence}`}>
              {claim.confidence === 'supported' ? '✓' : claim.confidence === 'partial' ? '~' : '?'}
            </span>
            <div className="claim-content">
              <span className="claim-text">{claim.text}</span>
              {claim.citations.length > 0 && (
                <div className="claim-sources">
                  <span className="claim-sources-label">Sources:</span>
                  {claim.citations.map((cit, citIdx) => (
                    <button
                      key={citIdx}
                      className="claim-source-link"
                      onClick={() => handleCitationClick(cit)}
                      title={cit.quote ? `"${cit.quote}"` : 'View source document'}
                      disabled={!cit.document_id}
                    >
                      <span className="source-name">
                        {cit.source_name 
                          ? cit.source_name.substring(0, 25)
                          : 'Document'}
                      </span>
                      {cit.page_number && (
                        <span className="source-page">p.{cit.page_number}</span>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </li>
        ))}
      </ul>
      {claims.length > 5 && (
        <button 
          className="btn-link"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? 'Show less' : `Show all ${claims.length} claims`}
        </button>
      )}
    </div>
  );
}

function V6StatsFooter({ stats }: { stats: V6Stats }) {
  const [showDetails, setShowDetails] = useState(false);
  
  return (
    <div className="v6-stats">
      <div className="v6-stats-summary" onClick={() => setShowDetails(!showDetails)}>
        <span className="stats-brief">
          {stats.rounds_executed} rounds · {stats.elapsed_ms.toFixed(0)}ms
        </span>
        <span className="stats-toggle">{showDetails ? '▼' : '▶'}</span>
      </div>
      
      {showDetails && (
        <div className="v6-stats-details">
          <div className="stats-row">
            <span className="stats-label">Task Type:</span>
            <span className="stats-value">{stats.task_type}</span>
          </div>
          <div className="stats-row">
            <span className="stats-label">Execution Time:</span>
            <span className="stats-value">{stats.elapsed_ms.toFixed(0)}ms</span>
          </div>
        </div>
      )}
    </div>
  );
}


// =============================================================================
// Active Scope Bar — always-visible bar above chat input
// =============================================================================

function ActiveScopeBar({
  activeScope,
  lastUsedScope,
  collections,
  hasDraftChanges,
  hasMessages,
  onEditScope,
}: {
  activeScope: UserSelectedScope;
  lastUsedScope: UserSelectedScope | null;
  collections: CollectionNode[];
  hasDraftChanges: boolean;
  hasMessages: boolean;
  onEditScope?: () => void;
}) {
  const isFullArchive = activeScope.mode === 'full_archive';
  const scopeChanged = lastUsedScope
    ? scopeFingerprint(activeScope) !== scopeFingerprint(lastUsedScope)
    : false;

  // Resolve collection IDs to display pills
  const pills: string[] = [];
  if (isFullArchive) {
    pills.push('Full archive');
  } else {
    const colIds = activeScope.included_collection_ids || [];
    for (const id of colIds.slice(0, 4)) {
      const col = collections.find(c => c.id === id);
      pills.push(col ? (col.title || col.slug) : `#${id}`);
    }
    if (colIds.length > 4) pills.push(`+${colIds.length - 4} more`);
    const docCount = activeScope.included_document_ids?.length || 0;
    if (docCount > 0) pills.push(`${docCount} document${docCount !== 1 ? 's' : ''}`);
  }

  return (
    <div className="scope-bar">
      <span className="scope-bar-label">Scope:</span>
      <div className="scope-bar-pills">
        {pills.map((name, i) => (
          <span key={i} className={isFullArchive && i === 0 ? 'scope-pill scope-pill-archive' : 'scope-pill'}>
            {name}
          </span>
        ))}
      </div>
      {hasMessages && lastUsedScope && (
        scopeChanged
          ? <span className="scope-tag-changed">changed since last query</span>
          : <span style={{ fontSize: '11px', color: 'var(--color-text-muted)' }}>(same as last query)</span>
      )}
    </div>
  );
}


// =============================================================================
// Used Scope Display — collapsed/expandable per-message provenance
// =============================================================================

function UsedScopeDisplay({
  scopeOverride,
  collections,
}: {
  scopeOverride: { overridden?: boolean; run_scope?: { mode: string; included_collection_ids?: number[]; included_document_ids?: number[] } };
  collections?: CollectionNode[];
}) {
  const [expanded, setExpanded] = useState(false);
  const runScope = scopeOverride.run_scope;
  if (!runScope) return null;

  const isFullArchive = runScope.mode === 'full_archive';
  const colIds = runScope.included_collection_ids || [];
  const docIds = runScope.included_document_ids || [];

  const summary = isFullArchive
    ? 'Full archive'
    : (() => {
        const names = colIds.slice(0, 3).map(id => {
          const col = collections?.find(c => c.id === id);
          return col ? (col.title || col.slug) : `#${id}`;
        });
        const label = names.join(', ') + (colIds.length > 3 ? ` +${colIds.length - 3}` : '');
        return docIds.length > 0 ? `${label} \u2014 ${docIds.length} documents` : label;
      })();

  return (
    <div style={{ fontSize: '12px', color: 'var(--color-text-secondary, #666)', marginBottom: '4px' }}>
      <span
        onClick={() => setExpanded(!expanded)}
        style={{ cursor: 'pointer', userSelect: 'none' }}
      >
        Used scope: {summary} {expanded ? '▼' : '▶'}
      </span>
      {expanded && (
        <div style={{ marginTop: 4, paddingLeft: 8, fontSize: '11px' }}>
          <div>Mode: {runScope.mode}</div>
          {colIds.length > 0 && (
            <div>
              Collections ({colIds.length}):
              {colIds.map(id => {
                const col = collections?.find(c => c.id === id);
                return <div key={id} style={{ paddingLeft: 8 }}>&#x2022; {col ? (col.title || col.slug) : `#${id}`}</div>;
              })}
            </div>
          )}
          {docIds.length > 0 && <div>Documents: {docIds.length}</div>}
        </div>
      )}
    </div>
  );
}
