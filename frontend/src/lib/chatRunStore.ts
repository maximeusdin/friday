'use client';

/**
 * chatRunStore — per-session chat run registry.
 *
 * The chat backend already runs every request in its own thread, so multiple
 * sessions can process concurrently. The only thing that used to serialize them
 * was the frontend: a single <Conversation> instance held the in-flight stream
 * (AbortController, progress, evidence, isSending) in component-local state, so
 * switching sessions clobbered or hid a running query.
 *
 * This module lifts that run state OUT of the component into a module-level
 * Map<sessionId, ChatRunState>, driven through useSyncExternalStore. A run keeps
 * streaming and accumulating regardless of which session is on screen, and any
 * number of sessions can be in flight at once. Components read their session's
 * run with useChatRun(sessionId); the sidebar reads useRunningSessionIds().
 */

import { useSyncExternalStore } from 'react';
import type { QueryClient } from '@tanstack/react-query';
import { api } from './api';
import type {
  V9ChatResponse, V9ProgressEvent, V9EvidenceBullet, UserSelectedScope,
} from '@/types/api';

export interface ChatRunState {
  isSending: boolean;
  progressSteps: V9ProgressEvent[];
  evidenceBullets: V9EvidenceBullet[];
  lastV9: V9ChatResponse | null;
  sendError: string | null;
  /** epoch ms when the run started (for the elapsed-time progress bar); null when idle. */
  startedAt: number | null;
  /** scope the in-flight run was launched with. */
  runScope: UserSelectedScope | null;
}

/** Shared idle snapshot — stable reference so useSyncExternalStore doesn't loop. */
export const EMPTY_RUN: ChatRunState = Object.freeze({
  isSending: false,
  progressSteps: [],
  evidenceBullets: [],
  lastV9: null,
  sendError: null,
  startedAt: null,
  runScope: null,
});

const runs = new Map<number, ChatRunState>();
const controllers = new Map<number, AbortController>();
const listeners = new Set<() => void>();

// Cached snapshot of running session ids — replaced only when membership changes,
// so useSyncExternalStore consumers get a stable reference between real changes.
let runningSnapshot: number[] = [];
const EMPTY_IDS: number[] = [];

function recomputeRunning(): void {
  const cur: number[] = [];
  for (const [id, st] of runs) if (st.isSending) cur.push(id);
  cur.sort((a, b) => a - b);
  const same =
    cur.length === runningSnapshot.length && cur.every((v, i) => v === runningSnapshot[i]);
  if (!same) runningSnapshot = cur;
}

function emit(): void {
  recomputeRunning();
  for (const l of listeners) l();
}

function subscribe(cb: () => void): () => void {
  listeners.add(cb);
  return () => { listeners.delete(cb); };
}

function snapshotFor(sessionId: number | null): ChatRunState {
  if (sessionId == null) return EMPTY_RUN;
  return runs.get(sessionId) ?? EMPTY_RUN;
}

function patch(sessionId: number, next: Partial<ChatRunState>): void {
  const prev = runs.get(sessionId) ?? EMPTY_RUN;
  runs.set(sessionId, { ...prev, ...next });
  emit();
}

export function isSessionRunning(sessionId: number): boolean {
  return runs.get(sessionId)?.isSending ?? false;
}

export interface StartRunOptions {
  action?: 'default' | 'think_deeper';
  carryContext?: Record<string, unknown>;
  scope: UserSelectedScope;
  queryClient: QueryClient;
  /** Called with the final response (so the page can update side panels / scope memory). */
  onResult?: (response: V9ChatResponse) => void;
}

/**
 * Start (or ignore, if already running) a chat run for a session. Fire-and-forget:
 * progress streams into the store and re-renders whichever components read this session.
 */
export function startRun(sessionId: number, text: string, opts: StartRunOptions): void {
  if (runs.get(sessionId)?.isSending) return;

  const controller = new AbortController();
  controllers.set(sessionId, controller);

  const steps: V9ProgressEvent[] = [];
  const bullets: V9EvidenceBullet[] = [];

  patch(sessionId, {
    isSending: true,
    progressSteps: [],
    evidenceBullets: [],
    lastV9: null,
    sendError: null,
    startedAt: Date.now(),
    runScope: opts.scope,
  });

  api.sendV9MessageStreaming(
    sessionId,
    text,
    opts.action ?? 'default',
    {
      onProgress: (event) => {
        if (controller.signal.aborted) return;
        steps.push(event);
        patch(sessionId, { progressSteps: [...steps] });
      },
      onEvidenceUpdate: (event) => {
        if (controller.signal.aborted) return;
        const newBullets = event.details?.bullets || [];
        if (newBullets.length === 0) return;
        bullets.push(...newBullets);
        patch(sessionId, { evidenceBullets: [...bullets] });
      },
      onResult: (response) => {
        if (controller.signal.aborted) return;
        patch(sessionId, { lastV9: response });
        opts.queryClient.invalidateQueries({ queryKey: ['chatHistory', sessionId] });
        opts.onResult?.(response);
      },
      onError: (error) => {
        if (controller.signal.aborted) return;
        patch(sessionId, { sendError: error, isSending: false });
      },
    },
    controller.signal,
    opts.carryContext,
    opts.scope,
  )
    .catch((err: unknown) => {
      if (controller.signal.aborted) return;
      let msg = err instanceof Error ? err.message : String(err);
      if (msg === 'Failed to fetch' || msg === 'network error' || msg.includes('fetch')) {
        msg = 'Connection was lost — the request may have timed out. Try again.';
      }
      patch(sessionId, { sendError: msg });
    })
    .finally(() => {
      controllers.delete(sessionId);
      if (!controller.signal.aborted) {
        patch(sessionId, { isSending: false, startedAt: null });
      }
    });
}

/** Abort a session's in-flight run and clear its transient progress. */
export function stopRun(sessionId: number): void {
  controllers.get(sessionId)?.abort();
  controllers.delete(sessionId);
  if (runs.has(sessionId)) {
    patch(sessionId, { isSending: false, progressSteps: [], evidenceBullets: [], startedAt: null });
  }
}

/** React hook: subscribe to a single session's run state. */
export function useChatRun(sessionId: number | null): ChatRunState {
  return useSyncExternalStore(
    subscribe,
    () => snapshotFor(sessionId),
    () => EMPTY_RUN,
  );
}

/** React hook: the set (as a stable-sorted array) of sessions currently running. */
export function useRunningSessionIds(): number[] {
  return useSyncExternalStore(
    subscribe,
    () => runningSnapshot,
    () => EMPTY_IDS,
  );
}
