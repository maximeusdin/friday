'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { Session } from '@/types/api';
import clsx from 'clsx';

interface SessionListProps {
  activeSessionId?: number;
  onSessionSelect: (session: Session) => void;
}

export function SessionList({ activeSessionId, onSessionSelect }: SessionListProps) {
  const [newLabel, setNewLabel] = useState('');
  const [isCreating, setIsCreating] = useState(false);
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
            <div className="session-label">{session.label}</div>
            <div className="session-meta">
              {formatDate(session.last_activity || session.created_at)}
              {session.message_count !== undefined && (
                <> · {session.message_count} messages</>
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
