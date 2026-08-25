import { useId, useState } from 'react'
import { BarChart3 } from 'lucide-react';
import { bankOverviewModel } from './bankOverview.js'

const STATUS_TONE = {
  keep: 'bg-emerald-400', pending: 'bg-amber-300', reject: 'bg-rose-400',
}
const BAR_TONES = ['bg-indigo-400', 'bg-sky-400', 'bg-teal-400', 'bg-fuchsia-400', 'bg-amber-300']

function Distribution({ item }) {
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="text-[11px] font-semibold uppercase tracking-wide text-content-muted">
          {item.label}
        </h3>
        {item.total > 0 && <span className="text-[10px] tabular-nums text-content-subtle">{item.total} measured</span>}
      </div>
      {item.total > 0 ? (
        <>
          <div className="flex h-2 overflow-hidden rounded-full bg-surface-raised" aria-hidden="true">
            {item.rows.map((row, index) => (
              <span key={row.id} className={BAR_TONES[index % BAR_TONES.length]}
                style={{ width: `${row.widthPercent}%`, minWidth: row.value > 0 ? '1px' : undefined }} />
            ))}
          </div>
          <ul className="flex flex-wrap gap-x-2 gap-y-0.5 text-[10px] text-content-subtle">
            {item.rows.map((row) => (
              <li key={row.id}><span className="text-content-muted">{row.label}</span>{' '}
                <span className="tabular-nums">{row.value} ({row.percent}%)</span></li>
            ))}
          </ul>
        </>
      ) : <p className="text-[11px] text-content-subtle">{item.emptyHint}</p>}
    </div>
  )
}

export default function BankOverview({ payload }) {
  const model = bankOverviewModel(payload)
  // Deliberately component-local: a live payload refresh updates every number
  // without reopening a panel the user folded away (or closing one they were
  // reading). It starts open once, when this overview itself is mounted.
  const [open, setOpen] = useState(true)
  const contentId = useId()
  const totalText = model.available && model.total != null
    ? `${model.total.toLocaleString()} images`
    : 'Total unavailable'

  return (
    <section aria-labelledby="bank-overview-title"
      className="rounded-xl border border-border bg-surface p-4 space-y-3">
      <div className="flex items-center justify-between gap-3">
        <h2 id="bank-overview-title" className="min-w-0 text-sm font-semibold text-content">
          <button type="button" onClick={() => setOpen((value) => !value)}
            aria-expanded={open} aria-controls={contentId}
            className="min-h-10 lg:min-h-0 flex min-w-0 items-center gap-1.5 rounded text-left hover:text-indigo-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/70">
            <span aria-hidden="true" className="w-3 shrink-0 text-content-subtle">
              {open ? '▾' : '▸'}
            </span>
            <span className="inline-flex items-center gap-1.5"><BarChart3 aria-hidden="true" className="h-4 w-4" /> Bank overview</span>
          </button>
        </h2>
        <span className="text-xs tabular-nums text-content-subtle">
          {totalText}
        </span>
      </div>

      <div id={contentId} hidden={!open} className="space-y-3">
        {!model.available ? (
          <p role="status" className="text-xs text-amber-300/90">
            Overview unavailable — waiting for bank data.
          </p>
        ) : (
          <>
            <div className="space-y-1">
              <h3 className="sr-only">Curation status</h3>
              {model.total > 0 ? (
                <div className="flex h-3 overflow-hidden rounded-full bg-surface-raised"
                  role="img" aria-label={model.status.map((row) => `${row.label}: ${row.value}, ${row.percent}%`).join('; ')}>
                  {model.status.filter((row) => row.value > 0).map((row) => (
                    <span key={row.id} className={STATUS_TONE[row.id]}
                      style={{ width: `${row.widthPercent}%`, minWidth: '1px' }} />
                  ))}
                </div>
              ) : model.total === 0
                ? <p className="text-xs text-content-subtle">No images to summarize.</p>
                : <p className="text-xs text-amber-300/90">Curation totals unavailable.</p>}
              <ul className="flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-content-muted">
                {model.status.map((row) => (
                  <li key={row.id} className="tabular-nums">
                    {row.label} <span className="text-content">{row.value ?? '—'}</span>
                    {row.percent != null && <span className="text-content-subtle"> · {row.percent}%</span>}
                  </li>
                ))}
              </ul>
            </div>

            <div>
              <h3 className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-content-muted">Pass coverage</h3>
              <ul className="grid grid-cols-2 gap-1.5 sm:grid-cols-3 xl:grid-cols-2 2xl:grid-cols-3">
                {model.passes.map((pass) => (
                  <li key={pass.key} className="rounded-md border border-border bg-surface-raised px-2 py-1.5">
                    <span className="block text-[11px] text-content-muted">{pass.label}</span>
                    <span className={`block text-[10px] tabular-nums ${pass.value == null ? 'text-amber-300/90' : 'text-content'}`}>
                      {pass.text}
                    </span>
                    {pass.percent != null && (
                      <span className="mt-1 block h-1 overflow-hidden rounded-full bg-surface" aria-hidden="true">
                        <span className="block h-full bg-indigo-400"
                          style={{ width: `${pass.widthPercent}%`, minWidth: pass.value > 0 ? '1px' : undefined }} />
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            </div>

            <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
              {model.distributions.map((item) => <Distribution key={item.id} item={item} />)}
            </div>

            <div>
              <h3 className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-content-muted">Structure</h3>
              <dl className="grid grid-cols-2 gap-1.5 sm:grid-cols-4 xl:grid-cols-2 2xl:grid-cols-4">
                {model.kpis.map((kpi) => (
                  <div key={kpi.label} className="rounded-md border border-border bg-surface-raised px-2 py-1.5">
                    <dt className="text-[10px] text-content-subtle">{kpi.label}</dt>
                    <dd className="text-sm font-semibold tabular-nums text-content">{kpi.value}</dd>
                    <dd className="text-[10px] leading-tight text-content-subtle">{kpi.detail}</dd>
                  </div>
                ))}
              </dl>
            </div>
          </>
        )}
      </div>
    </section>
  )
}
