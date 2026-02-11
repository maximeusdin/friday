'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { Session } from '@/types/api';
import clsx from 'clsx';

interface SessionListProps {
  activeSessionId?: number;
  onSessionSelect: (session: Session) => void;
  onSessionDelete?: (sessionId: number) => void;
}

export function SessionList({ activeSessionId, onSessionSelect, onSessionDelete }: SessionListProps) {
  const [newLabel, setNewLabel] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const [confirmDeleteId, setConfirmDeleteId] = useState<number | null>(null);
  const queryClient = useQueryClient();

  const { data: sessions, isLoading, error } = useQuery({
    queryKey: ['sessions'],
    queryFn: api.getSessions,
  });

  const createMutation = useMutation({
    mutationFn: api.createSession,
    onSuccess: (newSession) => {
      queryClient.invalidateQueries({ queryKey: ['sessions'] });
      setNewLabel('');
      setIsCreating(false);
      onSessionSelect(newSession);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: api.deleteSession,
    onSuccess: (_data, deletedId) => {
      queryClient.invalidateQueries({ queryKey: ['sessions'] });
      setConfirmDeleteId(null);
      if (deletedId === activeSessionId) {
        onSessionDelete?.(deletedId);
      }
    },
    onError: () => {
      setConfirmDeleteId(null);
    },
  });

  const handleCreate = (e: React.FormEvent) => {
    e.preventDefault();
    if (newLabel.trim()) {
      createMutation.mutate({ label: newLabel.trim() });
    }
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (isLoading) {
    return <div className="loading">Loading sessions...</div>;
  }

  if (error) {
    return (
      <div className="empty-state">
        <p>Failed to load sessions</p>
        <p className="text-sm text-muted">{String(error)}</p>
      </div>
    );
  }

  return (
    <div>
      {/* Create new session */}
      {isCreating ? (
        <form onSubmit={handleCreate} className="mb-md">
          <input
            type="text"
            value={newLabel}
            onChange={(e) => setNewLabel(e.target.value)}
            placeholder="Session name..."
            autoFocus
            disabled={createMutation.isPending}
          />
          <div className="flex gap-sm mt-sm">
            <button
              type="submit"
              className="btn-primary flex-1"
              disabled={!newLabel.trim() || createMutation.isPending}
            >
              {createMutation.isPending ? 'Creating...' : 'Create'}
            </button>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => setIsCreating(false)}
              disabled={createMutation.isPending}
            >
              Cancel
            </button>
          </div>
        </form>
      ) : (
        <button
          className="btn-primary"
          style={{ width: '100%', marginBottom: 'var(--spacing-md)' }}
          onClick={() => setIsCreating(true)}
        >
          + New Session
        </button>
      )}

      {/* Sessions list */}
      {sessions && sessions.length > 0 ? (
        sessions.map((session) => (
          <div
            key={session.id}
            className={clsx('session-item', {
              active: session.id === activeSessionId,
            })}
            onClick={() => onSessionSelect(session)}
          >
            <div style={{ display: 'flex', alignItems: 'start', justifyContent: 'space-between' }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div className="session-label">{session.label}</div>
                <div className="session-meta">
                  {formatDate(session.last_activity || session.created_at)}
                  {session.message_count !== undefined && (
                    <> · {session.message_count} messages</>
                  )}
                </div>
              </div>
              {confirmDeleteId === session.id ? (
                <div
                  style={{ display: 'flex', gap: '4px', flexShrink: 0 }}
                  onClick={(e) => e.stopPropagation()}
                >
                  <button
                    className="btn-danger-sm"
                    style={{
                      padding: '2px 8px',
                      fontSize: '11px',
                      background: 'var(--color-danger, #dc3545)',
                      color: '#fff',
                      border: 'none',
                      borderRadius: '3px',
                      cursor: 'pointer',
                    }}
                    onClick={() => deleteMutation.mutate(session.id)}
                    disabled={deleteMutation.isPending}
                  >
                    {deleteMutation.isPending ? '...' : 'Yes'}
                  </button>
                  <button
                    style={{
                      padding: '2px 8px',
                      fontSize: '11px',
                      border: '1px solid var(--color-border)',
                      borderRadius: '3px',
                      background: 'transparent',
                      cursor: 'pointer',
                    }}
                    onClick={() => setConfirmDeleteId(null)}
                  >
                    No
                  </button>
                </div>
              ) : (
                <button
                  className="btn-delete-session"
                  style={{
                    padding: '2px 6px',
                    fontSize: '12px',
                    background: 'transparent',
                    border: 'none',
                    cursor: 'pointer',
                    opacity: 0.4,
                    flexShrink: 0,
                  }}
                  title="Delete session"
                  onClick={(e) => {
                    e.stopPropagation();
                    setConfirmDeleteId(session.id);
                  }}
                >
                  ✕
                </button>
              )}
            </div>
          </div>
        ))
      ) : (
        <div className="empty-state">
          <p>No sessions yet</p>
          <p className="text-sm">Create one to get started</p>
        </div>
      )}
    </div>
  );
}
