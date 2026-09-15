'use client';

import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

type Placement = 'top' | 'bottom';
type Align = 'start' | 'end' | 'center';

interface PopoverProps {
  /** The element the popover is anchored to. */
  anchorRef: React.RefObject<HTMLElement | null>;
  open: boolean;
  onClose: () => void;
  children: ReactNode;
  /** Preferred side; flips automatically when there isn't room. */
  placement?: Placement;
  align?: Align;
  className?: string;
  /** Accessible label for the dialog. */
  label?: string;
}

const GAP = 8;
const MARGIN = 12;
/** Below this viewport width the popover docks as a bottom sheet (see globals.css). */
const SHEET_BREAKPOINT = 680;

/**
 * Popover — a floating panel anchored to a trigger, portalled to <body> so it
 * is never clipped by a scrolling pane.
 *
 * On narrow viewports it turns into a bottom sheet instead of trying to squeeze
 * an anchored panel onto a phone screen: same component, same content, the
 * layout switch is one data attribute the stylesheet reads. Closing on Escape,
 * outside click and scroll is handled here so no caller has to repeat it.
 */
export function Popover({
  anchorRef, open, onClose, children, placement = 'top', align = 'start',
  className, label,
}: PopoverProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const [sheet, setSheet] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  // Position against the anchor, flipping and clamping to stay on screen.
  useLayoutEffect(() => {
    if (!open) return;

    const place = () => {
      const anchor = anchorRef.current;
      const panel = panelRef.current;
      if (!anchor || !panel) return;

      const isSheet = window.innerWidth <= SHEET_BREAKPOINT;
      setSheet(isSheet);
      if (isSheet) {
        setPos({ top: 0, left: 0 });
        return;
      }

      const a = anchor.getBoundingClientRect();
      const p = panel.getBoundingClientRect();
      const vw = window.innerWidth;
      const vh = window.innerHeight;

      const roomAbove = a.top - MARGIN;
      const roomBelow = vh - a.bottom - MARGIN;
      const side: Placement =
        placement === 'top'
          ? (roomAbove >= p.height || roomAbove >= roomBelow ? 'top' : 'bottom')
          : (roomBelow >= p.height || roomBelow >= roomAbove ? 'bottom' : 'top');

      const top = side === 'top'
        ? Math.max(MARGIN, a.top - p.height - GAP)
        : Math.min(vh - p.height - MARGIN, a.bottom + GAP);

      let left =
        align === 'end' ? a.right - p.width
        : align === 'center' ? a.left + a.width / 2 - p.width / 2
        : a.left;
      left = Math.max(MARGIN, Math.min(left, vw - p.width - MARGIN));

      setPos({ top, left });
    };

    place();
    // Re-place on anything that can move the anchor.
    const ro = new ResizeObserver(place);
    if (panelRef.current) ro.observe(panelRef.current);
    window.addEventListener('resize', place);
    window.addEventListener('scroll', place, true);
    return () => {
      ro.disconnect();
      window.removeEventListener('resize', place);
      window.removeEventListener('scroll', place, true);
    };
  }, [open, anchorRef, placement, align]);

  // Escape closes; focus moves into the panel so keyboard users land inside it.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onClose();
        anchorRef.current?.focus();
      }
    };
    document.addEventListener('keydown', onKey);
    const t = setTimeout(() => {
      const first = panelRef.current?.querySelector<HTMLElement>(
        'input, button, [tabindex]:not([tabindex="-1"])',
      );
      first?.focus();
    }, 0);
    return () => {
      document.removeEventListener('keydown', onKey);
      clearTimeout(t);
    };
  }, [open, onClose, anchorRef]);

  if (!open || !mounted) return null;

  return createPortal(
    <>
      <div className="popover-backdrop" onMouseDown={onClose} />
      <div
        ref={panelRef}
        className={`popover${className ? ` ${className}` : ''}`}
        data-sheet={sheet ? 'true' : 'false'}
        role="dialog"
        aria-modal="false"
        aria-label={label}
        style={pos ? { top: pos.top, left: pos.left } : { opacity: 0, top: 0, left: 0 }}
      >
        {children}
      </div>
    </>,
    document.body,
  );
}
