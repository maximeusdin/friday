'use client';

import { getLoginUrl, logout, type AuthUser } from '@/lib/api';

interface AuthHeaderProps {
  user: AuthUser | null;
  onLogout?: () => void;
}

export function AuthHeader({ user, onLogout }: AuthHeaderProps) {
  const handleLogout = async () => {
    await logout();
    onLogout?.();
  };

  return (
    <header className="global-header">
      <span className="global-header-brand">Friday</span>

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
    </header>
  );
}
