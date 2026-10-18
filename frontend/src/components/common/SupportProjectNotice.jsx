import { useState } from 'react';
import { Link } from 'react-router';
import { Heart, X } from 'lucide-react';
import { ACCESS_NOTICE_DISMISSED_KEY as DISMISSED_KEY, PATREON_ACCESS_SUMMARY } from '../patreon/patreonModel.js';

const PATREON_PAGE = 'https://www.patreon.com/c/Loraperfectgf';

export default function SupportProjectNotice() {
  const [dismissed, setDismissed] = useState(() => {
    try { return localStorage.getItem(DISMISSED_KEY) === '1'; }
    catch { return false; }
  });

  if (dismissed) return null;

  const dismiss = () => {
    setDismissed(true);
    try { localStorage.setItem(DISMISSED_KEY, '1'); }
    catch { /* Dismiss for this visit when browser storage is unavailable. */ }
  };

  return (
    <aside aria-label="LDS updates on Patreon"
      className="flex items-start gap-3 rounded-lg border border-border bg-surface-raised px-4 py-3">
      <Heart aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-content-muted" />
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-content">New LDS versions reach Patreon members first</p>
        <p className="mt-0.5 text-xs leading-relaxed text-content-muted">
          {PATREON_ACCESS_SUMMARY}
        </p>
        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1">
          <a href={PATREON_PAGE} target="_blank" rel="noopener noreferrer"
            className="inline-flex min-h-8 items-center rounded-md border border-primary/25 bg-primary/10 px-3 py-1 text-xs font-medium text-content no-underline transition-colors hover:bg-primary/20 focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary">
            Join on Patreon
          </a>
          <Link to="/plugins?tab=purchases"
            className="inline-flex min-h-8 items-center text-xs text-content-muted underline underline-offset-4 hover:text-content">
            Already a member? Connect Patreon
          </Link>
        </div>
      </div>
      <button type="button" onClick={dismiss} aria-label="Dismiss Patreon announcement"
        title="Hide this message"
        className="-mr-1 -mt-1 inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md text-content-muted hover:bg-surface-raised hover:text-content">
        <X aria-hidden="true" className="h-4 w-4" />
      </button>
    </aside>
  );
}
