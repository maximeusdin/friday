'use client';

import { useState, useRef, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { Session, Message, Plan } from '@/types/api';

interface ConversationProps {
  session: Session | null;
  onPlanIdUpdate: (planId: number) => void;
}

export function Conversation({ session, onPlanIdUpdate }: ConversationProps) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const queryClient = useQueryClient();

  const { data: messages, isLoading } = useQuery({
    queryKey: ['messages', session?.id],
    queryFn: () => api.getSessionMessages(session!.id),
    enabled: !!session,
  });

  const sendMutation = useMutation({
    mutationFn: ({ sessionId, content }: { sessionId: number; content: string }) =>
      api.sendMessage(sessionId, { content }),
    onSuccess: (response) => {
      queryClient.invalidateQueries({ queryKey: ['messages', session?.id] });
      setInput('');
      onPlanIdUpdate(response.plan.id);
    },
  });

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (session && input.trim() && !sendMutation.isPending) {
      sendMutation.mutate({ sessionId: session.id, content: input.trim() });
    }
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
          <div className="loading">Loading messages...</div>
        ) : messages && messages.length > 0 ? (
          <>
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
            <div ref={messagesEndRef} />
          </>
        ) : (
          <div className="empty-state">
            <p>No messages yet</p>
            <p className="text-sm">Start by asking a research question</p>
          </div>
        )}

        {sendMutation.error && (
          <div className="card" style={{ borderColor: 'var(--color-danger)' }}>
            <p style={{ color: 'var(--color-danger)' }}>
              Failed to send message: {String(sendMutation.error)}
            </p>
          </div>
        )}
      </div>

      {/* Input area */}
      <div style={{ padding: 'var(--spacing-md)', borderTop: '1px solid var(--color-border)' }}>
        <form onSubmit={handleSubmit}>
          <div className="flex gap-sm">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a research question..."
              disabled={sendMutation.isPending}
              style={{ flex: 1 }}
            />
            <button
              type="submit"
              className="btn-primary"
              disabled={!input.trim() || sendMutation.isPending}
            >
              {sendMutation.isPending ? 'Sending...' : 'Send'}
            </button>
          </div>
        </form>
      </div>
    </>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user';
  
  return (
    <div className={`message ${isUser ? 'message-user' : 'message-assistant'}`}>
      <div className="message-role">{message.role}</div>
      <div>{message.content}</div>
      {message.plan_id && (
        <div className="text-sm text-muted mt-sm">
          Plan #{message.plan_id}
        </div>
      )}
    </div>
  );
}
