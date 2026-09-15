/** Small, shared formatting helpers. Kept pure so they're trivially testable. */

/** "2 collections" / "1 collection" */
export function plural(n: number, word: string, suffix = 's'): string {
  return `${n.toLocaleString()} ${word}${n === 1 ? '' : suffix}`;
}

/** Compact, human time for list rows: "14:02", "Yesterday", "12 Sep", "Sep 2025". */
export function shortTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const days = Math.floor((startOfToday.getTime() - new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()) / 86_400_000);

  if (days <= 0) return d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });
  if (days === 1) return 'Yesterday';
  if (days < 7) return d.toLocaleDateString(undefined, { weekday: 'long' });
  if (d.getFullYear() === now.getFullYear()) {
    return d.toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
  }
  return d.toLocaleDateString(undefined, { month: 'short', year: 'numeric' });
}

export type TimeBucket = 'Today' | 'Yesterday' | 'Previous 7 days' | 'Previous 30 days' | 'Older';

/** Which recency group a timestamp belongs to, for sidebar grouping. */
export function timeBucket(iso: string): TimeBucket {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return 'Older';
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const days = Math.floor((startOfToday.getTime() - new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()) / 86_400_000);
  if (days <= 0) return 'Today';
  if (days === 1) return 'Yesterday';
  if (days < 7) return 'Previous 7 days';
  if (days < 30) return 'Previous 30 days';
  return 'Older';
}

export const TIME_BUCKET_ORDER: TimeBucket[] = [
  'Today', 'Yesterday', 'Previous 7 days', 'Previous 30 days', 'Older',
];

/** Bytes → "1.4 MB". */
export function fileSize(bytes?: number | null): string {
  if (bytes == null) return '';
  const units = ['B', 'KB', 'MB', 'GB'];
  let n = bytes;
  let u = 0;
  while (n >= 1024 && u < units.length - 1) {
    n /= 1024;
    u += 1;
  }
  return `${n < 10 && u > 0 ? n.toFixed(1) : Math.round(n)} ${units[u]}`;
}

/** Seconds → "48s" / "2m 14s", for the elapsed-time footer. */
export function duration(ms: number): string {
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`;
}

/** Derive a session title from the first question asked in it. */
export function titleFromQuestion(question: string, max = 52): string {
  const clean = question.replace(/\s+/g, ' ').trim();
  if (clean.length <= max) return clean;
  const cut = clean.slice(0, max);
  const lastSpace = cut.lastIndexOf(' ');
  return `${(lastSpace > max * 0.6 ? cut.slice(0, lastSpace) : cut).trimEnd()}…`;
}
