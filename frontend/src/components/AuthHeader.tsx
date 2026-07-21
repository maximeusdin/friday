'use client';

import { useState } from 'react';
import { getLoginUrl, logout, type AuthUser } from '@/lib/api';
import { InfoModal, INFO_SECTIONS, type InfoSection } from './InfoModal';

interface AuthHeaderProps {
  user: AuthUser | null;
  onLogout?: () => void;
}

export function AuthHeader({ user, onLogout }: AuthHeaderProps) {
  const [infoSection, setInfoSection] = useState<InfoSection | null>(null);

  const handleLogout = async () => {
    await logout();
    onLogout?.();
  };

  return (
    <header className="global-header">
      <a href="/" className="global-header-brand" title="Return to home page">
        Friday
      </a>

      <nav className="header-info-nav" aria-label="Information">
        {INFO_SECTIONS.map((s) => (
          <button
            key={s.key}
            type="button"
            className="header-info-btn"
            onClick={() => setInfoSection(s.key)}
          >
            {s.label}
          </button>
        ))}
      </nav>

      <div className="global-header-actions">
        {user ? (
          <>
            <span className="auth-user">
              {user.email || user.sub}
            </span>
            <button
              type="button"
              onClick={handleLogout}
              className="auth-signout"
            >
              Sign out
            </button>
          </>
        ) : (
          <a href={getLoginUrl()} className="auth-signin">
            Sign in
          </a>
        )}
      </div>

      {infoSection && <InfoModal section={infoSection} onClose={() => setInfoSection(null)} />}
    </header>
  );
}
