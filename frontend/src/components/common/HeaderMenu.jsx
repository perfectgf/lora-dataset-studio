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
 *   - align        : menu horizontal alignment; 'right' (default) or 'left'.
 *   - placement    : 'below' (default) drops the menu under the trigger; 'side'
 *                    opens it beside the trigger, bottoms aligned — for a
 *                    trigger at the foot of the side menu, where there is no
 *                    room below; 'side-down' opens it beside the trigger, tops
 *                    aligned, and lets a long list scroll — for a trigger in
 *                    the upper part of the folded menu.
 *   - triggerClassName : ({ isActive }) => classes, replacing the trigger's own
 *                    (a row of the side menu instead of a header button).
 *   - labelled     : the trigger already shows its name as text, so neither the
 *                    tooltip nor the screen-reader copy of triggerTitle is added. */
const TRIGGER_BASE =
  'relative px-3 py-1.5 rounded-md text-sm font-medium transition-colors';

export default function HeaderMenu({
  triggerLabel, triggerTitle, active = false, dot = false, align = 'right',
  placement = 'below', triggerClassName, labelled = false, children,
}) {
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
        title={labelled ? undefined : triggerTitle}
        onClick={() => setOpen((v) => !v)}
        className={triggerClassName
          ? triggerClassName({ isActive: open || active })
          : `${TRIGGER_BASE} ${
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
        {!labelled && <span className="sr-only">{triggerTitle}</span>}
      </button>
      {open && (
        <div
          role="menu"
          aria-label={triggerTitle}
          className={`absolute z-50 flex min-w-[11rem] flex-col gap-0.5 rounded-lg border border-border bg-surface-overlay p-1 shadow-2xl ${
            placement === 'side'
              ? 'bottom-0 left-full ml-3'
              : placement === 'side-down'
                ? 'left-full top-0 ml-3 max-h-[75vh] overflow-y-auto'
                : `${align === 'right' ? 'right-0' : 'left-0'} top-full mt-1`}`}
        >
          {typeof children === 'function' ? children(close) : children}
        </div>
      )}
    </div>
  );
}
