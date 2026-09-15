'use client';

import { useState } from 'react';
import { getLoginUrl, logout, type AuthUser } from '@/lib/api';
import { HelpModal, type HelpSection } from './HelpModal';
import { Icon } from './ui/Icon';
import { Menu } from './ui/Menu';

interface AppHeaderProps {
  user: AuthUser | null;
  onLogout?: () => void;
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
  /** Phone only: the sidebar is a drawer there, so "new session" needs a home in the header. */
  onNewSession?: () => void;
}

/**
 * AppHeader — wordmark, two destinations, an account menu.
 *
 * The five outlined buttons that used to sit here (About / Collections &
 * Downloads / How to Use / Chat vs. Search / Use in Chatbots) gave equal weight
 * to one real destination and four documents. Collections stays a first-class
 * link because it is content; the rest collapse into Guide, which opens the
 * same modal with a section rail.
 */
export function AppHeader({ user, onLogout, sidebarOpen, onToggleSidebar, onNewSession }: AppHeaderProps) {
  const [help, setHelp] = useState<HelpSection | null>(null);

  const handleLogout = async () => {
    await logout();
    onLogout?.();
  };

  const initial = (user?.email || user?.sub || '?').trim().charAt(0);

  return (
    <header className="hdr">
      <button
        type="button"
        className="icon-btn"
        onClick={onToggleSidebar}
        aria-label={sidebarOpen ? 'Hide sessions' : 'Show sessions'}
        aria-pressed={sidebarOpen}
        title={sidebarOpen ? 'Hide sessions' : 'Show sessions'}
      >
        <Icon name="menu" size={18} />
      </button>

      <a href="/" className="hdr-brand" title="Friday home">
        <span className="hdr-wordmark">Friday</span>
        <span className="hdr-sub">Cold War archive</span>
      </a>

      <div className="spacer" />

      <nav className="hdr-nav" aria-label="Main">
        {onNewSession && (
          <button
            type="button"
            className="icon-btn phone-only"
            onClick={onNewSession}
            aria-label="New session"
            title="New session"
          >
            <Icon name="plus" size={20} />
          </button>
        )}
        <button type="button" className="hdr-link" onClick={() => setHelp('collections')}>
          <Icon name="library" size={16} />
          <span className="hdr-link-text">Collections</span>
        </button>
        <button type="button" className="hdr-link" onClick={() => setHelp('howto')}>
          <Icon name="help" size={16} />
          <span className="hdr-link-text">Guide</span>
        </button>
      </nav>

      {user ? (
        <Menu
          label="Account"
          align="end"
          heading={
            <div className="menu-label" title={user.email || user.sub}>
              {user.email || user.sub}
            </div>
          }
          items={[
            {
              label: 'Use Friday in your chatbot',
              icon: <Icon name="plug" size={16} />,
              onSelect: () => setHelp('connect'),
            },
            {
              label: 'About Friday',
              icon: <Icon name="info" size={16} />,
              onSelect: () => setHelp('about'),
            },
            {
              label: 'Sign out',
              icon: <Icon name="sign-out" size={16} />,
              onSelect: handleLogout,
              separated: true,
            },
          ]}
          trigger={(props) => (
            <button type="button" className="hdr-account" {...props} title="Account">
              <span className="avatar">{initial}</span>
              <Icon name="chevron-down" size={14} />
            </button>
          )}
        />
      ) : (
        <a href={getLoginUrl()} className="btn-primary">
          Sign in
        </a>
      )}

      {help && <HelpModal section={help} onClose={() => setHelp(null)} />}
    </header>
  );
}
