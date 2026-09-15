'use client';

import { useMemo, useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { useRunningSessionIds, stopRun } from '@/lib/chatRunStore';
import { shortTime, timeBucket, TIME_BUCKET_ORDER, plural, type TimeBucket } from '@/lib/format';
import type { Session } from '@/types/api';
import { Icon } from './ui/Icon';
import { Menu } from './ui/Menu';
import { toast } from './ui/Toast';

interface SessionSidebarProps {
  activeSessionId?: number;
  onSessionSelect: (session: Session) => void;
  onSessionDelete?: (sessionId: number) => void;
  onNewSession: () => void;
}

/** Placeholder row for the session that exists but hasn't been created yet.
 *  Sessions are created from the first question, so without this the list gives
 *  no sign that "New session" did anything. */
function DraftRow({ onSelect }: { onSelect: () => void }) {
  return (
    <div className="sb-group">
      <div className="sb-item is-active is-draft" role="button" tabIndex={0} onClick={onSelect}>
        <span className="sb-item-body">
          <span className="sb-item-label">New session</span>
          <span className="sb-item-meta">Ask a question to start it</span>
        </span>
      </div>
    </div>
  );
}

/** Sessions with a filter above them, grouped by recency like every other
 *  conversation list people already know how to read. */
export function SessionSidebar({
  activeSessionId, onSessionSelect, onSessionDelete, onNewSession,
}: SessionSidebarProps) {
  const [filter, setFilter] = useState('');
  const queryClient = useQueryClient();
  const runningIds = useRunningSessionIds();

  const { data: sessions, isLoading, error } = useQuery({
    queryKey: ['sessions'],
    queryFn: api.getSessions,
  });

  const deleteMutation = useMutation({
    mutationFn: api.deleteSession,
    onSuccess: (_data, deletedId) => {
      stopRun(deletedId); // abort any in-flight chat run for the deleted session
      queryClient.invalidateQueries({ queryKey: ['sessions'] });
      if (deletedId === activeSessionId) onSessionDelete?.(deletedId);
    },
    onError: () => toast('Could not delete that session'),
  });

  const groups = useMemo(() => {
    const q = filter.trim().toLowerCase();
    const list = (sessions ?? [])
      .filter((s) => !q || s.label.toLowerCase().includes(q))
      .slice()
      .sort((a, b) =>
        new Date(b.last_activity || b.created_at).getTime()
        - new Date(a.last_activity || a.created_at).getTime());

    const map = new Map<TimeBucket, Session[]>();
    for (const s of list) {
      const bucket = timeBucket(s.last_activity || s.created_at);
      const arr = map.get(bucket);
      if (arr) arr.push(s);
      else map.set(bucket, [s]);
    }
    return TIME_BUCKET_ORDER
      .filter((b) => map.has(b))
      .map((b) => ({ bucket: b, sessions: map.get(b)! }));
  }, [sessions, filter]);

  const total = sessions?.length ?? 0;

  return (
    <aside className="sidebar" aria-label="Sessions">
      <div className="sidebar-top">
        <button type="button" className="btn-primary btn-block" onClick={onNewSession}>
          <Icon name="plus" size={16} />
          New session
        </button>
        {total > 6 && (
          <label className="field">
            <Icon name="search" size={15} />
            <input
              type="text"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Find a session…"
              aria-label="Filter sessions"
            />
          </label>
        )}
      </div>

      <div className="sidebar-scroll">
        {activeSessionId == null && !isLoading && !error && (
          <DraftRow onSelect={onNewSession} />
        )}

        {isLoading && <div className="loading"><span className="spinner" /></div>}

        {error && (
          <div className="sidebar-empty">
            Couldn&apos;t load sessions.
            <br />
            <button
              type="button"
              className="btn-link"
              onClick={() => queryClient.invalidateQueries({ queryKey: ['sessions'] })}
            >
              Try again
            </button>
          </div>
        )}

        {!isLoading && !error && total === 0 && (
          <div className="sidebar-empty">
            No sessions yet — your first question starts one.
          </div>
        )}

        {!isLoading && total > 0 && groups.length === 0 && (
          <div className="sidebar-empty">No session matches “{filter}”.</div>
        )}

        {groups.map(({ bucket, sessions: list }) => (
          <div className="sb-group" key={bucket}>
            <div className="sb-group-title">{bucket}</div>
            {list.map((session) => {
              const running = runningIds.includes(session.id);
              const isActive = session.id === activeSessionId;
              return (
                <div
                  key={session.id}
                  className={`sb-item${isActive ? ' is-active' : ''}`}
                  onClick={() => onSessionSelect(session)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      onSessionSelect(session);
                    }
                  }}
                  role="button"
                  tabIndex={0}
                  aria-current={isActive}
                >
                  {running && <span className="run-dot" title="Working…" aria-label="Working" />}
                  <span className="sb-item-body">
                    <span className="sb-item-label" title={session.label}>{session.label}</span>
                    <span className="sb-item-meta">
                      {running
                        ? 'Working…'
                        : [
                            shortTime(session.last_activity || session.created_at),
                            session.message_count ? plural(session.message_count, 'message') : null,
                            session.search_count ? plural(session.search_count, 'search', 'es') : null,
                          ].filter(Boolean).join(' · ')}
                    </span>
                  </span>
                  <span className="sb-item-actions">
                    <Menu
                      label={`Actions for ${session.label}`}
                      items={[
                        {
                          label: running ? 'Stop and delete' : 'Delete session',
                          icon: <Icon name="trash" size={15} />,
                          danger: true,
                          onSelect: () => {
                            if (window.confirm(`Delete “${session.label}” and its history?`)) {
                              deleteMutation.mutate(session.id);
                            }
                          },
                        },
                      ]}
                      trigger={(props) => (
                        <button
                          type="button"
                          className="icon-btn icon-btn-sm"
                          {...props}
                          aria-label="Session actions"
                        >
                          <Icon name="more" size={16} />
                        </button>
                      )}
                    />
                  </span>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </aside>
  );
}
