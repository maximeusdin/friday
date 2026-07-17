'use client';

import { useState } from 'react';
import { getLoginUrl, logout, type AuthUser } from '@/lib/api';

interface AuthHeaderProps {
  user: AuthUser | null;
  onLogout?: () => void;
}

export function AuthHeader({ user, onLogout }: AuthHeaderProps) {
  const [showAbout, setShowAbout] = useState(false);

  const handleLogout = async () => {
    await logout();
    onLogout?.();
  };

  return (
    <header className="global-header">
      <a href="/" className="global-header-brand" title="Return to home page">
        Friday
      </a>

      <div className="global-header-actions">
        <button
          type="button"
          className="header-about-btn"
          onClick={() => setShowAbout(true)}
        >
          About
        </button>
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

      {showAbout && (
        <div className="about-overlay" onClick={() => setShowAbout(false)} role="dialog" aria-modal="true" aria-label="About Friday">
          <div className="about-card" onClick={(e) => e.stopPropagation()}>
            <div className="about-card-header">
              <h2>About Friday</h2>
              <button
                type="button"
                className="about-close-btn"
                onClick={() => setShowAbout(false)}
                aria-label="Close"
              >
                ✕
              </button>
            </div>
            <div className="about-card-body">
              <h3>What is Friday?</h3>
              {/* Placeholder copy — replace with real description */}
              <p>
                Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do
                eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut
                enim ad minim veniam, quis nostrud exercitation ullamco laboris
                nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor
                in reprehenderit in voluptate velit esse cillum dolore eu fugiat
                nulla pariatur.
              </p>
              <p>
                Excepteur sint occaecat cupidatat non proident, sunt in culpa
                qui officia deserunt mollit anim id est laborum. Sed ut
                perspiciatis unde omnis iste natus error sit voluptatem
                accusantium doloremque laudantium.
              </p>
              <h3>Funding</h3>
              {/* Placeholder copy — replace with real funding acknowledgement */}
              <p>
                Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nemo
                enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut
                fugit, sed quia consequuntur magni dolores eos qui ratione
                voluptatem sequi nesciunt.
              </p>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
