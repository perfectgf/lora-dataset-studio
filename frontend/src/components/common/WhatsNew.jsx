/**
 * 🎁 What's new — in-app changelog surface.
 *
 * Two exports, wired together by a tiny DOM event bus so several triggers
 * (desktop nav + mobile bar) can share ONE modal and keep their badges in sync:
 *
 *   <WhatsNewButton /> — header button + unseen badge; dispatches the open event.
 *   <WhatsNewModal />   — mounted once in the Shell; lists entries newest-first,
 *                         marks the feed seen on open, and navigates on "Try it →".
 *
 * Content + all the unseen/seen logic live in ../../whatsNew.js (the file
 * maintainers edit each wave). This file is presentation only.
 */
import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import { useI18n } from '../../i18n/I18nContext';
import {
  WHATS_NEW,
  sortedEntries,
  unseenCount,
  markAllSeen,
  readSeenId,
  WHATS_NEW_SEEN_KEY,
  WHATS_NEW_OPEN_EVENT,
  WHATS_NEW_SEEN_EVENT,
} from '../../whatsNew';

const NAV_ITEM_BASE =
  'px-3 py-1.5 rounded-md text-sm font-medium no-underline transition-colors';

function formatDate(iso, locale) {
  // iso is 'YYYY-MM-DD'. Parse as local midnight so the day never shifts.
  const d = new Date(`${iso}T00:00:00`);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(locale === 'zh-CN' ? 'zh-CN' : 'en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

/**
 * Header trigger. Renders an icon button with a count badge when there are
 * unseen entries. Cheap to mount more than once (desktop + mobile); every
 * instance listens for the "seen" signal so opening the modal from one clears
 * the badge on all of them.
 */
export function WhatsNewButton() {
  const { t } = useI18n();
  const [seenId, setSeenId] = useState(() => readSeenId());
  const count = unseenCount(seenId);
  const has = count > 0;

  useEffect(() => {
    const refresh = () => setSeenId(readSeenId());
    const onStorage = (e) => { if (e.key === WHATS_NEW_SEEN_KEY) refresh(); };
    window.addEventListener(WHATS_NEW_SEEN_EVENT, refresh);
    window.addEventListener('storage', onStorage);
    return () => {
      window.removeEventListener(WHATS_NEW_SEEN_EVENT, refresh);
      window.removeEventListener('storage', onStorage);
    };
  }, []);

  const open = () => window.dispatchEvent(new CustomEvent(WHATS_NEW_OPEN_EVENT));

  return (
    <button
      type="button"
      onClick={open}
      title={has ? t('whatsNew.newTitle', { count }) : t('whatsNew.title')}
      className={`${NAV_ITEM_BASE} relative ${
        has ? 'text-content hover:text-content' : 'text-content-muted hover:text-content'
      } hover:bg-surface-raised`}
    >
      <span aria-hidden>🎁</span>
      {has && (
        <span
          aria-hidden
          className="absolute -right-0.5 -top-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-primary px-1 text-[10px] font-semibold leading-none text-white"
        >
          {count > 9 ? '9+' : count}
        </span>
      )}
      <span className="sr-only">
        {has ? t('whatsNew.unread', { count }) : t('whatsNew.title')}
      </span>
    </button>
  );
}

/**
 * The panel itself — mount exactly once (in the Shell). Opens on the shared
 * event, marks the feed seen (so the badge clears everywhere), and closes on
 * Esc, backdrop click, ✕, or a "Try it →" that navigates away.
 */
export function WhatsNewModal() {
  const { locale, t } = useI18n();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const dialogRef = useRef(null);
  const closeRef = useRef(null);
  const entries = sortedEntries(WHATS_NEW);

  useFocusTrap(dialogRef, open);

  // Open on the shared event; reading the panel clears the badge everywhere.
  useEffect(() => {
    const onOpen = () => {
      setOpen(true);
      const id = markAllSeen();
      window.dispatchEvent(new CustomEvent(WHATS_NEW_SEEN_EVENT, { detail: id }));
    };
    window.addEventListener(WHATS_NEW_OPEN_EVENT, onOpen);
    return () => window.removeEventListener(WHATS_NEW_OPEN_EVENT, onOpen);
  }, []);

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);

  useEffect(() => { if (open) closeRef.current?.focus(); }, [open]);

  if (!open) return null;

  const go = (to) => {
    setOpen(false);
    if (to) navigate(to);
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="whats-new-title"
      onClick={() => setOpen(false)}
      className="fixed inset-0 z-[9997] flex items-start justify-center bg-black/60 p-4 pt-16 backdrop-blur-sm sm:pt-24"
    >
      <div
        ref={dialogRef}
        onClick={(e) => e.stopPropagation()}
        className="flex w-full max-w-lg flex-col overflow-hidden rounded-2xl border border-border bg-surface-overlay shadow-2xl"
      >
        <header className="flex items-center justify-between gap-3 border-b border-border px-5 py-4">
          <h2 id="whats-new-title" className="flex items-center gap-2 text-base font-semibold text-content">
            <span aria-hidden>🎁</span> {t('whatsNew.title')}
          </h2>
          <button
            type="button"
            ref={closeRef}
            onClick={() => setOpen(false)}
            aria-label={t('whatsNew.close')}
            className="rounded-md p-1.5 text-content-subtle hover:bg-surface-raised hover:text-content"
          >
            <span aria-hidden>✕</span>
          </button>
        </header>

        <div className="max-h-[70vh] overflow-y-auto px-5">
          {entries.length === 0 ? (
            <p className="py-8 text-center text-sm text-content-muted">{t('whatsNew.empty')}</p>
          ) : (
            <ol className="divide-y divide-border">
              {entries.map((e) => (
                <li key={e.id} className="py-4">
                  <div className="flex items-baseline justify-between gap-3">
                    <h3 className="text-sm font-semibold text-content">{e.title}</h3>
                    <time dateTime={e.date} className="shrink-0 text-[11px] text-content-subtle">
                      {formatDate(e.date, locale)}
                    </time>
                  </div>
                  <p className="mt-1 text-sm leading-relaxed text-content-muted">{e.blurb}</p>
                  {e.to && (
                    <button
                      type="button"
                      onClick={() => go(e.to)}
                      className="mt-2 inline-flex items-center gap-1 text-xs font-semibold text-primary hover:underline"
                    >
                      {t('whatsNew.tryIt')} <span aria-hidden>→</span>
                    </button>
                  )}
                </li>
              ))}
            </ol>
          )}
        </div>
      </div>
    </div>
  );
}
