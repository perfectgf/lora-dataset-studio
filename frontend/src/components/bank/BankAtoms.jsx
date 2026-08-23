/**
 * The Bank's small shared primitives — the chip, the counter, the eyebrow, the
 * facet cluster and the pass button.
 *
 * They were defined inside BankWorkspace.jsx while there was one file to define
 * them in. The Encre redesign splits that screen into a top bar, a filter rail
 * and a passes panel, and all three draw chips and eyebrows; a primitive that
 * lives in one of them would have been copied into the other two.
 *
 * These render UI only. Anything with a decision in it belongs in bankLayout.js
 * or one of the other pure modules, where `node --test` can actually reach it.
 *
 * These carry NO palette of their own. A trialled "Encre" skin was dropped in
 * favour of the app's existing colours, so an atom that hard-coded a hex here
 * would be the one thing standing between the Bank and the rest of the app.
 */
/** A filter chip. `active` is the selected state, not a hover — the rail is
 *  read at a glance and "which facets am I on" has to survive that glance. */
export function Chip({ active, onClick, children, title }) {
  return (
    <button type="button" onClick={onClick} title={title}
      className={`min-h-10 lg:min-h-0 rounded-full border px-2.5 py-0.5 text-xs font-medium transition-colors ${active
        ? 'border-indigo-400/60 bg-indigo-500/20 text-indigo-200'
        : 'border-border bg-surface text-content-muted hover:text-content hover:bg-surface-raised'}`}>
      {children}
    </button>
  )
}

/** One header stat — a bold tabular figure with a subtle label. Kept/rejected/
 *  promoted carry their status colour so the strip reads at a glance. */
export function Stat({ label, value, tone }) {
  const toneCls = { emerald: 'text-emerald-300', rose: 'text-rose-300', indigo: 'text-indigo-300' }[tone] || 'text-content'
  const n = typeof value === 'number' ? value.toLocaleString() : value
  return (
    <span className="inline-flex items-baseline gap-1 whitespace-nowrap">
      <span className={`font-semibold tabular-nums ${toneCls}`}>{n}</span>
      <span className="text-xs text-content-subtle">{label}</span>
    </span>
  )
}

/** A small uppercase eyebrow that names a group of controls (a filter facet, the
 *  passes toolbar) so dense rows read as grouped rather than as a flat wall. */
export function GroupLabel({ children }) {
  return (
    <span className="text-[10px] font-semibold uppercase tracking-wide text-content-subtle">{children}</span>
  )
}

/** A labelled cluster of filter chips — the eyebrow names the facet (Status,
 *  Quality, Score, Groups); chips wrap together within it on narrow viewports. */
export function FilterGroup({ label, children }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <GroupLabel>{label}</GroupLabel>
      {children}
    </div>
  )
}

/** The individual analysis passes share one quiet, uniform button so they read
 *  as a secondary group next to the prominent Launch all / Promote actions. */
export function PassButton({ onClick, disabled, title, children }) {
  return (
    <button type="button" onClick={onClick} disabled={disabled} title={title}
      className="min-h-10 lg:min-h-0 rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content transition-colors hover:bg-surface disabled:opacity-50 disabled:hover:bg-surface-raised">
      {children}
    </button>
  )
}
