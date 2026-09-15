'use client';

import { useRef, useState, type ReactNode } from 'react';
import { Popover } from './Popover';

export interface MenuAction {
  label: string;
  onSelect: () => void;
  icon?: ReactNode;
  danger?: boolean;
  /** Renders a separator above this item. */
  separated?: boolean;
}

interface MenuProps {
  /** The trigger. Receives the props it must spread onto a focusable element. */
  trigger: (props: {
    ref: React.Ref<HTMLButtonElement>;
    onClick: (e: React.MouseEvent) => void;
    'aria-expanded': boolean;
    'aria-haspopup': 'menu';
  }) => ReactNode;
  items: MenuAction[];
  label?: string;
  align?: 'start' | 'end';
  placement?: 'top' | 'bottom';
  /** Non-interactive heading rendered above the items. */
  heading?: ReactNode;
}

/** Menu — a dropdown of actions, built on the same Popover as everything else. */
export function Menu({
  trigger, items, label = 'Menu', align = 'end', placement = 'bottom', heading,
}: MenuProps) {
  const ref = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);

  return (
    <>
      {trigger({
        ref,
        onClick: (e) => { e.stopPropagation(); e.preventDefault(); setOpen((v) => !v); },
        'aria-expanded': open,
        'aria-haspopup': 'menu',
      })}
      <Popover
        anchorRef={ref}
        open={open}
        onClose={() => setOpen(false)}
        align={align}
        placement={placement}
        label={label}
        className="menu"
      >
        <div role="menu">
          {heading}
          {items.map((item, i) => (
            <div key={i}>
              {item.separated && <div className="menu-sep" />}
              <button
                type="button"
                role="menuitem"
                className={`menu-item${item.danger ? ' menu-item-danger' : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  setOpen(false);
                  item.onSelect();
                }}
              >
                {item.icon}
                {item.label}
              </button>
            </div>
          ))}
        </div>
      </Popover>
    </>
  );
}
