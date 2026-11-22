import { useEffect, useState } from 'react';
import { pagerSlots } from './pagerSlots.js';

const number = (value) => Number(value || 0).toLocaleString('en-US');

const BUTTON = 'min-h-11 min-w-11 rounded-lg border px-2.5 font-semibold tabular-nums whitespace-nowrap '
  + 'transition-colors lg:min-h-9 lg:min-w-9';
const REST = 'border-border bg-surface text-content hover:bg-surface-raised disabled:opacity-40 disabled:hover:bg-surface';
const CURRENT = 'cursor-default border-primary bg-primary/10 text-primary';

/**
 * The pages of a long list: the entries shown ("49–96 of 1,775"), Previous and
 * Next, the first and the last page, and the pages around the current one.
 *
 * A wall of images used to be walked with ← Prev / Next → alone: reaching the
 * end of a long list took one press per page, with no way to jump or to see
 * how far along you were. Numbered pages are the rule for any list that can
 * run past a few pages; a button that piles more entries onto the same list
 * ("Load more") is not an alternative.
 *
 * The room the pager has decides how many neighbours it shows, so it stays on
 * one line from a phone to a wide window: two on each side of the current page
 * from 660 px, one from 460 px, none below — where Previous and Next also drop
 * their word and keep the arrow. Targets are 44 px below `lg`.
 * A list that fits one page has no pager at all.
 *
 * The buttons start at the LEFT edge, the range after them. Pushed to the
 * right edge, the pager under a grid ends in the window's bottom-right corner,
 * which is where the help tips and the setup notice are pinned: Next would sit
 * under a card for as long as a tip is up.
 *
 * `page` counts from 1; `onPage` receives the page to open. The name of the
 * list is passed as a literal `aria-label`, like on any landmark: the Bank's
 * surface inventory reads that attribute off the source of the screen that
 * mounts the pager, which is where "this list has pages" has to stay visible.
 */
export default function Pager({ page, size, total, onPage, 'aria-label': label = 'Pages', className = '' }) {
  const [node, setNode] = useState(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    if (!node || typeof ResizeObserver !== 'function') return undefined;
    const observer = new ResizeObserver((entries) => setWidth(entries[0].contentRect.width));
    observer.observe(node);
    return () => observer.disconnect();
  }, [node]);
  const pages = Math.ceil(total / size);
  if (!(pages > 1)) return null;
  const compact = width > 0 && width < 460;
  const slots = pagerSlots(page, pages, compact ? 0 : width >= 660 ? 2 : 1);
  return (
    <nav ref={setNode} aria-label={label}
      className={`flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-content-muted ${
        compact ? 'justify-center' : ''} ${className}`}>
      <div className="flex flex-wrap items-center justify-center gap-1">
        <button type="button" disabled={page <= 1} aria-label="Previous page"
          onClick={() => onPage(page - 1)} className={`${BUTTON} ${REST}`}>
          <span aria-hidden="true">←</span>{!compact && ' Previous'}
        </button>
        {slots.map((slot, index) => (slot
          ? (
            <button key={slot} type="button"
              aria-label={slot === page ? `Page ${slot}, current page` : `Page ${slot}`}
              aria-current={slot === page ? 'page' : undefined}
              onClick={() => { if (slot !== page) onPage(slot); }}
              className={`${BUTTON} ${slot === page ? CURRENT : REST}`}>
              {slot}
            </button>
          )
          : <span key={index === 1 ? 'before' : 'after'} aria-hidden="true" className="min-w-5 text-center">…</span>))}
        <button type="button" disabled={page >= pages} aria-label="Next page"
          onClick={() => onPage(page + 1)} className={`${BUTTON} ${REST}`}>
          {!compact && 'Next '}<span aria-hidden="true">→</span>
        </button>
      </div>
      <span role="status" className="whitespace-nowrap tabular-nums">
        {number((page - 1) * size + 1)}–{number(Math.min(total, page * size))} of {number(total)}
      </span>
    </nav>
  );
}
