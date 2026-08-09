import { useCallback, useEffect, useId, useRef, useState } from 'react';

/* One dropdown of the ◉ LoRA Canvas' filter bar: a chip that says its state,
 * and a popover that holds the controls.
 *
 * ── Why a hand-rolled menu and not <details> ────────────────────────────────
 * `<details>` is what the board's other disclosures use, and it is the right
 * answer for them: they are opened deliberately and closed deliberately. A
 * FILTER is not. You open it, tick two boxes and look at the board — so it has
 * to close when you look away, which `<details>` never does. A summary left
 * open behind the board is a popover permanently covering the thing it filters.
 *
 * Closing is therefore wired to the three gestures that all mean "I am done
 * here": a click outside, Escape, and moving the focus out of the menu. Escape
 * and the focus rule are not decoration — this bar replaced a panel where every
 * checkbox was reachable by Tab, and a menu you can open with a keyboard but
 * not close with one is a worse control than the panel it replaced.
 *
 * ── The chip says the state, closed ────────────────────────────────────────
 * A filter behind a dropdown is a filter you can forget you set — "the board is
 * empty" with no visible cause is the failure mode this whole redesign could
 * have introduced. So the chip carries its own count ("Datasets 3/17") and
 * lights up whenever it is NARROWING anything, at rest, with the popover shut.
 */
export default function CanvasFilterMenu({ label, glyph = null, summary = '',
  active = false, disabled = false, children, testId = null, align = 'left' }) {
  const [open, setOpen] = useState(false);
  const root = useRef(null);
  const id = useId();

  const close = useCallback(() => setOpen(false), []);

  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => {
      if (!root.current?.contains(e.target)) close();
    };
    const onKey = (e) => {
      if (e.key === 'Escape') { e.stopPropagation(); close(); }
    };
    // `pointerdown`, not `click`: a click that starts inside the popover and
    // ends outside it (a drag across a checkbox row) must not read as "away".
    document.addEventListener('pointerdown', onDown, true);
    document.addEventListener('keydown', onKey, true);
    return () => {
      document.removeEventListener('pointerdown', onDown, true);
      document.removeEventListener('keydown', onKey, true);
    };
  }, [open, close]);

  return (
    <div ref={root} className="relative"
      onBlur={(e) => {
        if (open && !root.current?.contains(e.relatedTarget)) close();
      }}>
      <button type="button" disabled={disabled}
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-haspopup="true"
        aria-controls={id}
        data-testid={testId}
        data-active={active ? 'true' : 'false'}
        className={'flex h-10 max-w-full items-center gap-1.5 rounded-md border px-2.5 '
          + 'text-[0.75rem] font-semibold disabled:opacity-40 lg:h-9 '
          + (active
            ? 'border-indigo-400/60 bg-indigo-500/15 text-indigo-100'
            : 'border-border bg-app/60 text-content hover:border-indigo-400/50')}>
        {glyph && <span aria-hidden>{glyph}</span>}
        <span className="truncate">{label}</span>
        {summary && (
          <span className="shrink-0 font-normal text-content-muted tabular-nums">{summary}</span>
        )}
        <span aria-hidden className="shrink-0 text-content-subtle">{open ? '▴' : '▾'}</span>
      </button>
      {open && (
        // ⚠️ Width. At 400 px a fixed 18-rem popover hangs off the screen, and a
        // filter half off the screen is a filter with no Clear button. It takes
        // the smaller of its ideal width and what the viewport actually has.
        // ⚠️ `bg-surface-overlay`, NOT `bg-surface`. They read as synonyms and
        // are not: `surface` is an alpha-baked tint (measured at 4 % white) meant
        // to lift a card off the page, and a popover painted with it is a sheet
        // of glass — the toolbar and the board underneath were legible straight
        // through the open menu, with the checkbox rows sitting in the middle of
        // the gesture sentence. `surface-overlay` is the near-opaque panel token
        // the app's modals already use. A floating layer needs an opaque one.
        <div id={id} role="group" aria-label={label}
          className={'absolute top-full z-50 mt-1 w-[min(20rem,calc(100vw-2rem))] '
            + 'rounded-lg border border-border bg-surface-overlay p-2 shadow-2xl '
            + (align === 'right' ? 'right-0' : 'left-0')}>
          {children}
        </div>
      )}
    </div>
  );
}
