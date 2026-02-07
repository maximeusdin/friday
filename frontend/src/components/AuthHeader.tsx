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
    <header className="auth-header" style={{
      display: 'flex',
      justifyContent: 'flex-end',
      alignItems: 'center',
      padding: '0.5rem 1rem',
      gap: '0.75rem',
      borderBottom: '1px solid var(--border, #e5e7eb)',
      backgroundColor: 'var(--header-bg, #f9fafb)',
    }}>
      {user ? (
        <>
          <span className="auth-user" style={{ fontSize: '0.875rem', color: 'var(--muted, #6b7280)' }}>
            {user.email || user.sub}
          </span>
          <button
            type="button"
            onClick={handleLogout}
            className="auth-logout"
            style={{
              padding: '0.25rem 0.5rem',
              fontSize: '0.875rem',
              cursor: 'pointer',
            }}
          >
            Log out
          </button>
        </>
      ) : (
        <a
          href={getLoginUrl()}
          className="auth-login"
          style={{
            padding: '0.25rem 0.5rem',
            fontSize: '0.875rem',
            textDecoration: 'none',
            color: 'var(--link, #2563eb)',
          }}
        >
          Sign in
        </a>
      )}
    </header>
  );
}
