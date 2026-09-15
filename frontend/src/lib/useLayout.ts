'use client';

/**
 * Device and layout detection.
 *
 * One hook decides which layout the app renders. It combines two signals:
 *
 *   - Viewport width, which decides the *shape* of the layout (phone, tablet
 *     or desktop). A phone in landscape is wide enough for the tablet shape,
 *     and an iPad in landscape is wide enough for the desktop one, and both
 *     of those are the right call.
 *   - Whether the primary input is a finger, which decides *behaviour*: no
 *     hover-revealed controls, larger targets, Enter inserts a newline, pinch
 *     zooms the document. This comes from the `pointer: coarse` media query,
 *     backed by the user-agent as a second opinion for the Android browsers
 *     that misreport it.
 *
 * The same breakpoints are used by the stylesheet (globals.css, section 17),
 * so CSS and components never disagree about which layout is showing.
 */

import { useSyncExternalStore } from 'react';

export type LayoutMode = 'phone' | 'tablet' | 'desktop';

export interface Layout {
  mode: LayoutMode;
  /** True when the primary pointer is a finger. */
  touch: boolean;
  isPhone: boolean;
  isDesktop: boolean;
}

/** Widths below this are phones. Matches the 680px block in globals.css. */
export const PHONE_MAX = 680;
/** Widths below this (and at or above PHONE_MAX) are tablets. */
export const TABLET_MAX = 1024;

const DESKTOP: Layout = { mode: 'desktop', touch: false, isPhone: false, isDesktop: true };

function uaSaysMobile(): boolean {
  if (typeof navigator === 'undefined') return false;
  const uad = (navigator as unknown as { userAgentData?: { mobile?: boolean } }).userAgentData;
  if (uad && typeof uad.mobile === 'boolean') return uad.mobile;
  return /Android|iPhone|iPad|iPod|Mobile|Silk/i.test(navigator.userAgent)
    // iPadOS 13+ reports as a Mac; the touch points give it away.
    || (/Macintosh/.test(navigator.userAgent) && navigator.maxTouchPoints > 1);
}

function read(): Layout {
  if (typeof window === 'undefined') return DESKTOP;
  const w = window.innerWidth;
  const mode: LayoutMode = w < PHONE_MAX ? 'phone' : w < TABLET_MAX ? 'tablet' : 'desktop';
  const touch = window.matchMedia('(pointer: coarse)').matches || uaSaysMobile();
  return { mode, touch, isPhone: mode === 'phone', isDesktop: mode === 'desktop' };
}

// A single cached snapshot, replaced only when something actually changes, so
// useSyncExternalStore consumers get a stable reference between real changes.
let snapshot: Layout = DESKTOP;
let initialised = false;

function refresh(): Layout {
  const next = read();
  if (!initialised || next.mode !== snapshot.mode || next.touch !== snapshot.touch) {
    snapshot = next;
    initialised = true;
  }
  return snapshot;
}

function subscribe(cb: () => void): () => void {
  if (typeof window === 'undefined') return () => {};
  const onChange = () => { refresh(); cb(); };
  window.addEventListener('resize', onChange);
  window.addEventListener('orientationchange', onChange);
  const mq = window.matchMedia('(pointer: coarse)');
  mq.addEventListener('change', onChange);
  return () => {
    window.removeEventListener('resize', onChange);
    window.removeEventListener('orientationchange', onChange);
    mq.removeEventListener('change', onChange);
  };
}

/** The current layout. Re-renders the caller when the mode or pointer changes. */
export function useLayout(): Layout {
  return useSyncExternalStore(subscribe, refresh, () => DESKTOP);
}

/** Non-hook read for event handlers. */
export function currentLayout(): Layout {
  return refresh();
}
