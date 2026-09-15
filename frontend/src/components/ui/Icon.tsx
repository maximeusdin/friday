/**
 * Icon — one inline SVG set for the whole product.
 *
 * Replaces the unicode glyphs (✕ ▶ ⌕ ↗ ⎘) that used to stand in for icons and
 * rendered differently on every platform. Stroke-based, inherits `currentColor`
 * and font-relative sizing, so an icon always matches the text it sits beside.
 */

export type IconName =
  | 'menu' | 'close' | 'plus' | 'check' | 'minus'
  | 'chevron-down' | 'chevron-right' | 'chevron-left' | 'chevron-up'
  | 'arrow-left' | 'arrow-up' | 'arrow-right'
  | 'search' | 'filter' | 'layers' | 'library' | 'file' | 'files'
  | 'copy' | 'download' | 'external' | 'more' | 'trash' | 'undo'
  | 'help' | 'info' | 'message' | 'spark' | 'stop' | 'eye-off' | 'restore'
  | 'zoom-in' | 'zoom-out' | 'sign-out' | 'plug' | 'book' | 'deeper';

const PATHS: Record<IconName, string> = {
  'menu': 'M3 6h14M3 10h14M3 14h14',
  'close': 'M5 5l10 10M15 5L5 15',
  'plus': 'M10 4v12M4 10h12',
  'minus': 'M4 10h12',
  'check': 'M4 10.5l4 4 8-9',
  'chevron-down': 'M5 8l5 5 5-5',
  'chevron-right': 'M8 5l5 5-5 5',
  'chevron-left': 'M12 5l-5 5 5 5',
  'chevron-up': 'M5 12l5-5 5 5',
  'arrow-left': 'M16 10H4m0 0l5-5m-5 5l5 5',
  'arrow-right': 'M4 10h12m0 0l-5-5m5 5l-5 5',
  'arrow-up': 'M10 16V4m0 0L5 9m5-5l5 5',
  'search': 'M9 15A6 6 0 109 3a6 6 0 000 12zm4.5 -1.5L17 17',
  'filter': 'M3 5h14M6 10h8M9 15h2',
  'layers': 'M10 3l7 3.5-7 3.5-7-3.5L10 3zM3 10.5L10 14l7-3.5M3 14L10 17.5 17 14',
  'library': 'M4 4v13M8 4v13M12 4.5l4.5 12.5M2.5 17h15',
  'file': 'M5 2.5h6l4 4V17a.5.5 0 01-.5.5h-9A.5.5 0 015 17V3a.5.5 0 010-.5zM11 2.5V7h4',
  'files': 'M7 2.5h5l3.5 3.5V15h-8.5V2.5zM4.5 5.5V17.5H13',
  'copy': 'M7.5 7.5h8a1 1 0 011 1v8a1 1 0 01-1 1h-8a1 1 0 01-1-1v-8a1 1 0 011-1zM13 4.5a1 1 0 00-1-1H4.5a1 1 0 00-1 1V12a1 1 0 001 1',
  'download': 'M10 3v9m0 0l-3.5-3.5M10 12l3.5-3.5M3.5 15.5h13',
  'external': 'M12 3h5v5M17 3l-7.5 7.5M14.5 11.5V16a1 1 0 01-1 1H4a1 1 0 01-1-1V6.5a1 1 0 011-1h4.5',
  'more': 'M5 10h.01M10 10h.01M15 10h.01',
  'trash': 'M4 5.5h12M8 5.5V4a.5.5 0 01.5-.5h3A.5.5 0 0112 4v1.5M5.5 5.5l.6 10a1 1 0 001 1h5.8a1 1 0 001-1l.6-10',
  'undo': 'M4 8h8a4 4 0 110 8H8M4 8l3-3M4 8l3 3',
  'help': 'M10 17.5a7.5 7.5 0 100-15 7.5 7.5 0 000 15zM8 8a2 2 0 113 1.7c-.6.4-1 .8-1 1.6M10 14h.01',
  'info': 'M10 17.5a7.5 7.5 0 100-15 7.5 7.5 0 000 15zM10 9v5M10 6h.01',
  'message': 'M3.5 6a2 2 0 012-2h9a2 2 0 012 2v6a2 2 0 01-2 2H8l-4 3v-3H5.5a2 2 0 01-2-2V6z',
  'spark': 'M10 2.5l1.8 4.2 4.2 1.8-4.2 1.8L10 14.5l-1.8-4.2L4 8.5l4.2-1.8L10 2.5z',
  'stop': 'M6.5 6.5h7v7h-7z',
  'eye-off': 'M4 4l12 12M8.2 8.3A2.2 2.2 0 0010 12.2M6 6.2C4.3 7.2 3 8.7 2.5 10c1 2.4 4 4.5 7.5 4.5 1.2 0 2.3-.2 3.3-.7M13.6 12A9 9 0 0017.5 10C16.5 7.6 13.5 5.5 10 5.5c-.5 0-1 0-1.4.1',
  'restore': 'M4 10a6 6 0 106-6 6 6 0 00-4.5 2M5.5 3.5V6H8',
  'zoom-in': 'M9 15A6 6 0 109 3a6 6 0 000 12zm4.5-1.5L17 17M6.5 9h5M9 6.5v5',
  'zoom-out': 'M9 15A6 6 0 109 3a6 6 0 000 12zm4.5-1.5L17 17M6.5 9h5',
  'sign-out': 'M8 17H4.5a1 1 0 01-1-1V4a1 1 0 011-1H8M12 13.5L16 10l-4-3.5M16 10H7',
  'plug': 'M7 2.5v4M13 2.5v4M4.5 6.5h11v3a5.5 5.5 0 01-11 0v-3zM10 15v2.5',
  'book': 'M4 3.5h5a2 2 0 012 2v11a2 2 0 00-2-2H4v-11zM16 3.5h-5a2 2 0 00-2 2v11a2 2 0 012-2h5v-11z',
  'deeper': 'M9 14A5 5 0 109 4a5 5 0 000 10zm3.6-1.4L17 17M6.8 9h4.4M9 6.8v4.4',
};

interface IconProps {
  name: IconName;
  size?: number;
  className?: string;
  strokeWidth?: number;
}

export function Icon({ name, size = 16, className, strokeWidth = 1.6 }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
      focusable="false"
    >
      <path d={PATHS[name]} />
    </svg>
  );
}
