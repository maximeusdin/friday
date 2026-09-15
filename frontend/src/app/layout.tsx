import type { Metadata, Viewport } from 'next';
import { Providers } from './providers';
import './globals.css';

export const metadata: Metadata = {
  title: 'Friday: Cold War archives',
  description:
    'Search declassified Cold War archives. Ask a question and get answers that link to the scanned page.',
};

/**
 * viewport-fit=cover is what makes env(safe-area-inset-*) resolve to real values,
 * which the composer already pads for — so the phone layout inherits correct
 * spacing under the home indicator the day it lands.
 */
export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  viewportFit: 'cover',
  themeColor: '#faf9f7',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
