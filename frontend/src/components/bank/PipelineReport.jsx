import { useState } from 'react'

/** The morning-after summary of the last "Launch all" run — one row per
 * requested pass (done / skipped-with-reason / error / cancelled) plus the
 * headline counts. Persisted on the bank, so it's still here when the user
 * reopens it. Collapsible; hidden while a job is live (the progress bar owns
 * that moment). */
const STEP_LABEL = {
  scan: '🔎 Scan quality', auto_reject: '🧹 Auto-reject',
  score: '✨ Score', watermark: '🚩 Watermarks',
  faces: '👥 Group by person', caption: '🏷️ Caption',
}
const STATUS_STYLE = {
  done: { icon: '✅', cls: 'text-emerald-300' },
  skipped: { icon: '⏭️', cls: 'text-amber-300' },
  cancelled: { icon: '🛑', cls: 'text-content-subtle' },
  error: { icon: '⚠️', cls: 'text-rose-300' },
}

function fmtWhen(ts) {
  if (!ts) return ''
  try { return new Date(ts * 1000).toLocaleString() } catch { return '' }
}

export default function PipelineReport({ report, onDismiss }) {
  const [open, setOpen] = useState(true)
  if (!report || !Array.isArray(report.steps)) return null
  const done = report.steps.filter((s) => s.status === 'done').length
  const total = report.steps.length
  const c = report.counts || {}

  return (
    <div className="rounded-lg border border-border bg-surface-raised">
      <div className="flex items-center gap-2 px-3 py-2">
        <button type="button" onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          className="flex items-center gap-2 text-sm font-semibold text-content">
          <span aria-hidden>{report.cancelled ? '🛑' : '🚀'}</span>
          Last Launch-all run — {done}/{total} passes ran
          {report.cancelled && <span className="text-content-subtle">(stopped)</span>}
          <span aria-hidden className="text-content-subtle">{open ? '▾' : '▸'}</span>
        </button>
        <span className="ml-auto text-xs text-content-subtle">{fmtWhen(report.finished_at)}</span>
        {onDismiss && (
          <button type="button" onClick={onDismiss} aria-label="Dismiss the report"
            className="rounded border border-border px-1.5 text-xs text-content-subtle hover:text-content">✕</button>
        )}
      </div>
      {open && (
        <div className="border-t border-border px-3 py-2 space-y-2">
          <ul className="space-y-1">
            {report.steps.map((s, i) => {
              const st = STATUS_STYLE[s.status] || STATUS_STYLE.error
              return (
                <li key={`${s.step}-${i}`} className="flex items-start gap-2 text-sm">
                  <span aria-hidden>{st.icon}</span>
                  <span className="min-w-0">
                    <span className="font-medium text-content">{STEP_LABEL[s.step] || s.step}</span>
                    {s.detail && s.status === 'done' && (
                      <span className="text-content-muted"> — {s.detail}</span>
                    )}
                    {s.reason && s.status !== 'done' && (
                      <span className={st.cls}> — {s.reason}</span>
                    )}
                  </span>
                </li>
              )
            })}
          </ul>
          <p className="border-t border-border pt-2 text-xs text-content-subtle">
            {c.total ?? 0} images · {c.scanned ?? 0} scanned · {c.reject ?? 0} rejected
            {c.scored ? ` · ${c.scored} scored` : ''}
            {c.watermark_detected ? ` · ${c.watermark_detected} watermarked` : ''}
            {c.person_groups ? ` · ${c.person_groups} person group(s)` : ''}
            {c.captioned ? ` · ${c.captioned} captioned` : ''}
          </p>
        </div>
      )}
    </div>
  )
}
