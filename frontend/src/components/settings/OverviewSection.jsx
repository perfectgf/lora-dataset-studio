import { Link } from 'react-router-dom'
import { deriveCapabilitySummary } from '../../hooks/useSetupSteps'

const FIX_LINKS = [
  { to: '/settings/engines', label: 'Image engines', hint: 'API keys, ChatGPT subscription, default engine' },
  { to: '/settings/local-tools', label: 'Local tools', hint: 'ComfyUI, Ollama, ai-toolkit' },
  { to: '/settings/training', label: 'Training', hint: 'Default family, cloud GPU' },
  { to: '/setup', label: 'Setup wizard', hint: 'Guided scan + install of everything above' },
]

/* The health map in full: the sidebar LEDs summarized as tiles, plus where to
   go to fix what's off. Status is glyph + text, never color alone. */
export default function OverviewSection({ caps }) {
  const summary = deriveCapabilitySummary(caps)
  const ready = summary.filter((s) => s.ok).length
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
        <div className="flex items-baseline justify-between gap-3">
          <h2 className="text-base font-semibold text-content">Capabilities</h2>
          <span className="font-mono text-xs text-content-subtle">{ready}/{summary.length} ready</span>
        </div>
        <div className="mt-4 grid gap-2 sm:grid-cols-2">
          {summary.map((s) => (
            <div key={s.label}
              className="flex items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2 text-sm">
              <span aria-hidden className={s.ok ? 'text-emerald-400' : 'text-content-subtle'}>{s.ok ? '✓' : '✗'}</span>
              <span className={s.ok ? 'text-content' : 'text-content-muted'}>{s.label}</span>
              <span className="sr-only">{s.ok ? '(ready)' : '(not available)'}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="rounded-xl border border-border bg-surface p-5">
        <h2 className="text-base font-semibold text-content">Where to fix it</h2>
        <ul className="mt-3 divide-y divide-border">
          {FIX_LINKS.map((l) => (
            <li key={l.to}>
              <Link to={l.to}
                className="group flex items-baseline justify-between gap-3 py-2.5 no-underline">
                <span className="text-sm font-medium text-content group-hover:underline">{l.label}</span>
                <span className="text-right text-xs text-content-subtle">{l.hint}</span>
              </Link>
            </li>
          ))}
        </ul>
      </section>
    </div>
  )
}
