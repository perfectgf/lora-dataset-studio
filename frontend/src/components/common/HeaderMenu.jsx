import { useEffect, useRef, useState } from 'react';

/** A small header dropdown (the ? Help menu and the ⚙ Settings menu share it).
 *  Follows the same interaction contract as the app's other popovers
 *  (CaptionOptionsPopover / PromptEditPopover): closes on Escape, on outside
 *  click, and — because every item navigates or toggles — on any item click.
 *
 *  Presentational only. `children` is a render-prop receiving `close`, so items
 *  (NavLinks, the Help-mode toggle) close the menu when they fire:
 *    <HeaderMenu …>{(close) => <NavLink onClick={close} … />}</HeaderMenu>
 *
 *  Props:
 *   - triggerLabel : node shown inside the trigger button (the ? / ⚙ glyph).
 *   - triggerTitle : tooltip + accessible name for the trigger.
 *   - active       : true when the current route lives in this menu — the
 *                    trigger then reflects the active-nav style, discreetly.
 *   - dot          : true to paint a small primary attention dot on the trigger
 *                    (Setup's "recommended steps unmet" indicator moved up here).
 *   - align        : menu horizontal alignment; 'right' (default) or 'left'. */
const TRIGGER_BASE =
  'relative px-3 py-1.5 rounded-md text-sm font-medium transition-colors';

export default function HeaderMenu({ triggerLabel, triggerTitle, active = false, dot = false, align = 'right', children }) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef(null);
  const triggerRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false);
    };
    const onKey = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        setOpen(false);
        triggerRef.current?.focus();   // return focus to the trigger on Escape
      }
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const close = () => setOpen(false);

  return (
    <div ref={wrapRef} className="relative">
      <button
        ref={triggerRef}
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        title={triggerTitle}
        onClick={() => setOpen((v) => !v)}
        className={`${TRIGGER_BASE} ${
          open || active
            ? 'bg-surface-raised text-content'
            : 'text-content-muted hover:text-content hover:bg-surface-raised'
        }`}
      >
        {triggerLabel}
        {dot && (
          <span aria-hidden="true"
            className="absolute right-1 top-1 h-1.5 w-1.5 rounded-full bg-primary" />
        )}
        <span className="sr-only">{triggerTitle}</span>
      </button>
      {open && (
        <div
          role="menu"
          aria-label={triggerTitle}
          className={`absolute ${align === 'right' ? 'right-0' : 'left-0'} top-full mt-1 z-50 min-w-[11rem]
            flex flex-col gap-0.5 rounded-lg border border-border bg-surface-overlay p-1 shadow-2xl`}
        >
          {typeof children === 'function' ? children(close) : children}
        </div>
      )}
    </div>
  );
}
