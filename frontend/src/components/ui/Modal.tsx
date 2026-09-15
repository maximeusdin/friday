'use client';

import { useEffect, useRef, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { Icon } from './Icon';

interface ModalProps {
  title: ReactNode;
  onClose: () => void;
  children: ReactNode;
  /** Wider shell for content that needs it (file tables, collection browser). */
  wide?: boolean;
  /** Extra controls rendered in the header, before the close button. */
  actions?: ReactNode;
  /** Set when the body handles its own scrolling/padding (e.g. split layouts). */
  bare?: boolean;
}

/**
 * Modal — the one dialog shell. Portalled, escape-closable, backdrop-closable,
 * and it restores focus to whatever opened it. At phone width the stylesheet
 * turns it into a bottom sheet.
 */
export function Modal({ title, onClose, children, wide, actions, bare }: ModalProps) {
  const restoreTo = useRef<HTMLElement | null>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    restoreTo.current = document.activeElement as HTMLElement;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    // Freeze the page behind the dialog.
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    panelRef.current?.focus();
    return () => {
      document.removeEventListener('keydown', onKey);
      document.body.style.overflow = prevOverflow;
      restoreTo.current?.focus?.();
    };
  }, [onClose]);

  if (typeof document === 'undefined') return null;

  return createPortal(
    <div
      className="modal-backdrop"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        ref={panelRef}
        className={`modal${wide ? ' modal-wide' : ''}`}
        role="dialog"
        aria-modal="true"
        aria-label={typeof title === 'string' ? title : undefined}
        tabIndex={-1}
      >
        <div className="modal-head">
          <h2 className="modal-title">{title}</h2>
          <div className="spacer" />
          {actions}
          <button type="button" className="icon-btn" onClick={onClose} aria-label="Close">
            <Icon name="close" size={18} />
          </button>
        </div>
        {bare ? children : <div className="modal-body">{children}</div>}
      </div>
    </div>,
    document.body,
  );
}
