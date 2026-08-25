import { useState } from 'react'
import { reportHeadline, stepView } from './pipelineReportView.js'

/** The morning-after summary of the last "Launch all" run — one row per
 * requested pass (done / skipped-with-reason / error / cancelled) plus the
 * headline counts. Persisted on the bank, so it's still here when the user
 * reopens it. Collapsible; hidden while a job is live (the progress bar owns
 * that moment). */
/* One entry per PIPELINE_STEPS value on the server (image_bank_service.py), and
 * the words are the ones on the BUTTONS — a report is read by the person who
 * pressed them.
 *
 * TWO WERE MISSING FOR AS LONG AS THEY HAVE EXISTED, and the report showed the
 * raw identifiers instead: `semantic_dedup` and `framing`. The maintainer read
 * "semantic_dedup" in his own report and concluded he did not know what it was —
 * he did, it is the ✂ button he uses. The `|| s.step` fallback below is what made
 * the omission invisible, so pipelineStepLabels.contract.test.js now fails if a
 * step ships without its label rather than letting the fallback absorb it. */
export const STEP_LABEL = {
  scan: 'Scan quality', auto_reject: 'Auto-reject',
  score: 'Score', semantic_index: 'Build semantic index',
  semantic_dedup: '✂ Find crops & variants',
  watermark: 'Watermarks', faces: 'Group by person',
  framing: 'Classify framing', caption: 'Caption',
}
function fmtWhen(ts) {
  if (!ts) return ''
  try { return new Date(ts * 1000).toLocaleString() } catch { return '' }
}

export default function PipelineReport({ report, onDismiss }) {
  const [open, setOpen] = useState(true)
  if (!report || !Array.isArray(report.steps)) return null
  // A pass re-run since counts as covered, and drops the 🛑 with it: this banner
  // used to keep announcing "cancelled before it ran" over a standalone run that
  // had just done the work. See pipelineReportView.js.
  const { covered, total, stopped, icon, redone } = reportHeadline(report)
  const c = report.counts || {}

  return (
    <div className="rounded-lg border border-border bg-surface-raised">
      <div className="flex items-center gap-2 px-3 py-2">
        <button type="button" onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          className="flex items-center gap-2 text-sm font-semibold text-content">
          <span aria-hidden>{icon}</span>
          Last Launch-all run — {covered}/{total} passes ran
          {redone > 0 && (
            <span className="text-content-subtle">({redone} re-run since)</span>
          )}
          {stopped && <span className="text-content-subtle">(stopped)</span>}
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
              const view = stepView(s)
              return (
                <li key={`${s.step}-${i}`} className="flex items-start gap-2 text-sm">
                  <span aria-hidden>{view.icon}</span>
                  <span className="min-w-0">
                    <span className="font-medium text-content">{STEP_LABEL[s.step] || s.step}</span>
                    {view.note && <span className={view.cls}> — {view.note}</span>}
                  </span>
                </li>
              )
            })}
          </ul>
          <p className="border-t border-border pt-2 text-xs text-content-subtle">
            {c.total ?? 0} images · {c.scanned ?? 0} scanned · {c.reject ?? 0} rejected
            {c.scored ? ` · ${c.scored} scored` : ''}
            {c.semantic_indexed ? ` · ${c.semantic_indexed} semantic-ready` : ''}
            {c.watermark_detected ? ` · ${c.watermark_detected} watermarked` : ''}
            {c.person_groups ? ` · ${c.person_groups} person group(s)` : ''}
            {c.captioned ? ` · ${c.captioned} captioned` : ''}
          </p>
        </div>
      )}
    </div>
  )
}
