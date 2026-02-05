'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api, V6ProgressEvent, V6ResultEvent } from '@/lib/api';
import type { Session, ChatMessage, ChatClaim, ChatMember, V6Stats, EvidenceRef, ChatCitation, WorkflowAction } from '@/types/api';

interface ConversationProps {
  session: Session | null;
  onResultSetUpdate?: (resultSetId: number | null) => void;
  onV6StatsUpdate?: (stats: V6Stats | null) => void;
  onProcessingChange?: (processing: boolean) => void;
  onEvidenceClick?: (evidence: EvidenceRef) => void;
}

export function Conversation({ session, onResultSetUpdate, onV6StatsUpdate, onProcessingChange, onEvidenceClick }: ConversationProps) {
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [streamingActions, setStreamingActions] = useState<WorkflowAction[]>([]);
  const [sendError, setSendError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const queryClient = useQueryClient();

  // Fetch chat history using the V6 chat endpoint
  const { data: messages, isLoading } = useQuery({
    queryKey: ['chatHistory', session?.id],
    queryFn: () => api.getChatHistory(session!.id),
    enabled: !!session,
  });

  // Handle streaming progress events
  const handleProgress = useCallback((event: V6ProgressEvent) => {
    const action: WorkflowAction = {
      step: event.step,
      status: event.status,
      message: event.message,
      details: event.details as Record<string, unknown>,
    };
    
    setStreamingActions(prev => {
      // Update existing action or add new one
      const existing = prev.findIndex(a => a.step === event.step);
      let newActions: WorkflowAction[];
      if (existing >= 0) {
        newActions = [...prev];
        newActions[existing] = action;
      } else {
        newActions = [...prev, action];
      }
      
      // Update V6Stats with the NEW actions (not stale state)
      onV6StatsUpdate?.({
        task_type: 'processing',
        rounds_executed: newActions.filter(a => a.step.startsWith('retrieval_round')).length,
        total_spans: 0,
        unique_docs: 0,
        elapsed_ms: 0,
        entity_linking: {},
        responsiveness: 'processing',
        actions: newActions,
      });
      
      return newActions;
    });
  }, [onV6StatsUpdate]);

  // Handle final result
  const handleResult = useCallback((result: V6ResultEvent) => {
    // Update V6 stats with final result
    onV6StatsUpdate?.({
      task_type: result.stats.task_type,
      rounds_executed: result.stats.rounds_executed,
      total_spans: result.stats.total_spans,
      unique_docs: result.stats.unique_docs,
      elapsed_ms: result.stats.elapsed_ms,
      entity_linking: result.stats.entity_linking,
      responsiveness: result.stats.responsiveness,
      actions: result.stats.actions,
    });
    
    // Refresh chat history to show new messages
    queryClient.invalidateQueries({ queryKey: ['chatHistory', session?.id] });
    
    setIsSending(false);
    setStreamingActions([]);
    onProcessingChange?.(false);
  }, [onV6StatsUpdate, queryClient, session?.id, onProcessingChange]);

  // Handle error
  const handleError = useCallback((error: string) => {
    setSendError(error);
    setIsSending(false);
    setStreamingActions([]);
    onProcessingChange?.(false);
  }, [onProcessingChange]);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!session || !input.trim() || isSending) return;
    
    const message = input.trim();
    setInput('');
    setIsSending(true);
    setSendError(null);
    setStreamingActions([]);
    onProcessingChange?.(true);
    onV6StatsUpdate?.(null);
    
    // Use streaming API
    await api.sendChatMessageStreaming(session.id, message, {
      onProgress: handleProgress,
      onResult: handleResult,
      onError: handleError,
    });
  };

  if (!session) {
    return (
      <div className="pane-content">
        <div className="empty-state">
          <p>Select a session to start</p>
          <p className="text-sm">Or create a new one from the sidebar</p>
        </div>
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
            {messages.map((message) => (
              <ChatBubble key={message.id} message={message} onEvidenceClick={onEvidenceClick} />
            ))}
            <div ref={messagesEndRef} />
          </>
        ) : (
          <div className="empty-state">
            <p>Ask a research question</p>
            <p className="text-sm">The V6 workflow will find evidence and generate cited answers</p>
          </div>
        )}

        {isSending && (
          <div className="chat-thinking">
            <div className="thinking-indicator">
              <span className="thinking-dot"></span>
              <span className="thinking-dot"></span>
              <span className="thinking-dot"></span>
            </div>
            <span className="thinking-text">Searching archives and analyzing evidence...</span>
          </div>
        )}

        {sendError && (
          <div className="card" style={{ borderColor: 'var(--color-danger)' }}>
            <p style={{ color: 'var(--color-danger)' }}>
              Failed to get answer: {sendError}
            </p>
          </div>
        )}
      </div>

      {/* Input area */}
      <div className="chat-input-container">
        <form onSubmit={handleSubmit}>
          <div className="flex gap-sm">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a research question..."
              disabled={isSending}
              className="chat-input"
            />
            <button
              type="submit"
              className="btn-primary"
              disabled={!input.trim() || isSending}
            >
              {isSending ? 'Thinking...' : 'Ask'}
            </button>
          </div>
        </form>
      </div>
    </>
  );
}

function ChatBubble({ message, onEvidenceClick }: { message: ChatMessage; onEvidenceClick?: (evidence: EvidenceRef) => void }) {
  const isUser = message.role === 'user';
  
  if (isUser) {
    return (
      <div className="chat-message chat-message-user">
        <div className="chat-message-content">{message.content}</div>
      </div>
    );
  }
  
  // Assistant message with V6 data
  return (
    <div className="chat-message chat-message-assistant">
      <div className="chat-answer">
        <div className="chat-answer-text">{message.content}</div>
        
        {/* Show members if this is a roster-style answer */}
        {message.members && message.members.length > 0 && (
          <MembersList members={message.members} onEvidenceClick={onEvidenceClick} />
        )}
        
        {/* Show claims with citations */}
        {message.claims && message.claims.length > 0 && (
          <ClaimsList claims={message.claims} onEvidenceClick={onEvidenceClick} />
        )}
        
        {/* V6 Stats footer */}
        {message.v6_stats && (
          <V6StatsFooter stats={message.v6_stats} />
        )}
      </div>
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
            <span className="member-bullet">•</span>
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
                    📄 {cit.source_name ? cit.source_name.substring(0, 15) : `p.${cit.page_number || '?'}`}
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
  
  return (
    <div className="chat-claims">
      <div className="chat-section-header">
        <span className="chat-section-title">Evidence-Backed Claims ({claims.length})</span>
      </div>
      <ul className="claims-list">
        {claims.slice(0, displayCount).map((claim, idx) => (
          <li key={idx} className="claim-item">
            <span className={`claim-confidence claim-${claim.confidence}`}>
              {claim.confidence === 'supported' ? '✓' : claim.confidence === 'partial' ? '~' : '?'}
            </span>
            <div className="claim-content">
              <span className="claim-text">{claim.text}</span>
              {claim.citations.length > 0 && (
                <div className="claim-citations-row">
                  {claim.citations.map((cit, citIdx) => (
                    <button
                      key={citIdx}
                      className="citation-btn"
                      onClick={() => handleCitationClick(cit)}
                      title={cit.quote || 'View source'}
                      disabled={!cit.document_id}
                    >
                      📄 {cit.source_name 
                        ? cit.source_name.substring(0, 20) + (cit.page_number ? ` p.${cit.page_number}` : '')
                        : `p.${cit.page_number || '?'}`}
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
  
  const responsiveClass = 
    stats.responsiveness === 'responsive' ? 'status-good' :
    stats.responsiveness === 'partial' ? 'status-partial' : 'status-poor';
  
  return (
    <div className="v6-stats">
      <div className="v6-stats-summary" onClick={() => setShowDetails(!showDetails)}>
        <span className={`responsiveness-badge ${responsiveClass}`}>
          {stats.responsiveness === 'responsive' ? '✓ Responsive' : 
           stats.responsiveness === 'partial' ? '~ Partial' : '! Limited'}
        </span>
        <span className="stats-brief">
          {stats.rounds_executed} rounds • {stats.elapsed_ms.toFixed(0)}ms
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
            <span className="stats-label">Retrieval Rounds:</span>
            <span className="stats-value">{stats.rounds_executed}</span>
          </div>
          <div className="stats-row">
            <span className="stats-label">Total Spans:</span>
            <span className="stats-value">{stats.total_spans}</span>
          </div>
          <div className="stats-row">
            <span className="stats-label">Entity Linking:</span>
            <span className="stats-value">
              {stats.entity_linking.total_linked || 0} linked, {stats.entity_linking.used_for_retrieval || 0} used
            </span>
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
