'use client';

import { useSyncExternalStore } from 'react';

export interface ToastItem {
  id: number;
  message: string;
  /** Optional single action — used for "Scope updated · Undo". */
  action?: { label: string; onSelect: () => void };
  duration?: number;
}

let items: ToastItem[] = [];
let seq = 0;
const listeners = new Set<() => void>();

function emit() {
  for (const l of listeners) l();
}

/** Show a transient message. Returns its id so it can be dismissed early. */
export function toast(message: string, action?: ToastItem['action'], duration = 5000): number {
  const id = ++seq;
  items = [...items, { id, message, action, duration }];
  emit();
  if (duration > 0) setTimeout(() => dismissToast(id), duration);
  return id;
}

export function dismissToast(id: number): void {
  if (!items.some((t) => t.id === id)) return;
  items = items.filter((t) => t.id !== id);
  emit();
}

const EMPTY: ToastItem[] = [];

/** Toaster — mount once in the app shell. */
export function Toaster() {
  const list = useSyncExternalStore(
    (cb) => { listeners.add(cb); return () => { listeners.delete(cb); }; },
    () => items,
    () => EMPTY,
  );

  if (list.length === 0) return null;

  return (
    <div className="toast-wrap" role="status" aria-live="polite">
      {list.map((t) => (
        <div key={t.id} className="toast">
          <span>{t.message}</span>
          {t.action && (
            <button
              type="button"
              className="toast-action"
              onClick={() => { t.action!.onSelect(); dismissToast(t.id); }}
            >
              {t.action.label}
            </button>
          )}
        </div>
      ))}
    </div>
  );
}
