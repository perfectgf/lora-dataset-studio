import { Link } from 'react-router'
import { deriveCapabilitySummary, capabilityDestination } from '../../hooks/useSetupSteps'

/* The health map in full: the sidebar LEDs summarized as tiles, plus where to
   go to fix what's off. Status is glyph + text, never color alone.

   Each tile is a DOOR, not a verdict. A row that reads "✗ Person masks" used to
   be a dead end — the user learned something was missing and then had to guess
   which of eight Settings sections (or which wizard screen) turned it on. Every
   row is now a real <Link> to the exact control, focus id included, resolved
   from the help registry by capabilityDestination so it cannot drift.

   Deliberately NOT eleven buttons: this is read at a glance, so the glyph and
   the label stay the loudest thing on the row and the affordance is a quiet
   chevron. Same reason the old four-entry "Where to fix it" card is gone — the
   rows above are strictly more precise; only the whole-rig guided path (the
   Setup wizard), which belongs to no single capability, is worth one line. */
export default function OverviewSection({ caps }) {
  const summary = deriveCapabilitySummary(caps)
  const ready = summary.filter((s) => s.ok).length
  const waiting = summary.filter((s) => s.pending).length
  return (
    <div className="space-y-6">
      {!caps.configured && (
        <div role="status" className="rounded-xl border border-primary/40 bg-primary/10 p-4 text-sm text-content">
          <p className="font-medium">Let's get you set up.</p>
          <p className="mt-1 text-content-muted">
            Add at least one image API key to start, or let the{' '}
            <Link to="/setup" className="font-medium text-sky-300 underline hover:text-sky-200">Setup wizard</Link>
            {' '}scan your machine and walk you through it.
          </p>
        </div>
      )}

      <section className="rounded-xl border border-border bg-surface p-5">
        <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
          <h2 className="text-base font-semibold text-content">Capabilities</h2>
          <span className="font-mono text-xs text-content-subtle">
            {ready}/{summary.length} ready{waiting ? ` · ${waiting} waiting` : ''}
          </span>
        </div>
        <p className="mt-1 text-xs text-content-subtle">Each row says what it unlocks — pick one to jump to the setting that turns it on.</p>
        <div className="mt-3 grid gap-1.5 sm:grid-cols-2">
          {summary.map((s) => {
            const dest = capabilityDestination(s)
            // Pending = installed, just not running. Amber ◐ like the wizard's
            // "Almost there", never the discouraging ✗ of something missing.
            const glyph = s.pending ? '◐' : (s.ok ? '✓' : '✗')
            const glyphCls = s.pending ? 'text-amber-400' : (s.ok ? 'text-emerald-400' : 'text-content-subtle')
            return (
              <Link key={s.label} to={dest.href} aria-label={dest.announce}
                className="group flex min-w-0 items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2
                  text-sm no-underline transition-colors hover:border-border-strong hover:bg-surface-raised
                  focus:outline-none focus-visible:ring-2 focus-visible:ring-primary">
                <span aria-hidden className={glyphCls}>{glyph}</span>
                <span className="min-w-0 flex-1">
                  <span className={`block truncate ${s.ok || s.pending ? 'text-content' : 'text-content-muted'}`}>
                    {s.label}
                  </span>
                  {/* What the row unlocks, always shown: a name alone ("Test
                      Studio") did not say it was where test IMAGES are made —
                      read from a phone, 2026-09-03. Wraps rather than truncates:
                      the second half of the sentence is the point. */}
                  {s.what && <span className="block text-[11px] leading-snug text-content-subtle">{s.what}</span>}
                  {s.note && <span className="block truncate text-[11px] text-amber-300/80">{s.note}</span>}
                </span>
                <span aria-hidden
                  className="shrink-0 text-content-subtle opacity-0 transition-opacity group-hover:opacity-100
                    group-focus-visible:opacity-100">›</span>
              </Link>
            )
          })}
        </div>
        <p className="mt-4 border-t border-border pt-3 text-xs text-content-muted">
          Not sure where to start?{' '}
          <Link to="/setup" className="font-medium text-sky-300 underline hover:text-sky-200">Run the Setup wizard</Link>
          {' '}— it scans your machine and installs what it can.
        </p>
      </section>
    </div>
  )
}
