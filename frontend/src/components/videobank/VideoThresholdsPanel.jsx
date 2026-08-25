import { useState } from 'react'
import { postJson, putJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import {
  cutSummary, draftThresholds, editThreshold, payloadFromDraft, thresholdFields,
} from './videoMetricsFilter'

/** 🎚 The quality cuts, edited as a DRAFT with a mandatory preview.
 *
 * Two design points carried over from the pure module and worth restating at the
 * component boundary:
 *
 * The grid keeps flagging against the SAVED cuts while the user types here —
 * nothing reaches config until Apply. Editing live would re-flag the bank on
 * every keystroke, including through values the user is merely passing through.
 *
 * And there are NO default cuts. The dry run is not a convenience, it is the
 * design: published thresholds measurably do not transfer between corpora (the
 * public motion floor lands at the 7th percentile of one real bank and at the
 * 12th of another), so a cut is only ever chosen against the bank's own numbers.
 */
export default function VideoThresholdsPanel({ bankId, saved, totalClips, onApplied }) {
  const toast = useToast()
  const [draft, setDraft] = useState(() => draftThresholds(saved))
  const [preview, setPreview] = useState(null)
  const [working, setWorking] = useState(false)

  const dryRun = async () => {
    setWorking(true)
    try {
      const result = await postJson(`/api/video-bank/${bankId}/metrics-dry-run`,
                                    payloadFromDraft(draft))
      setPreview(cutSummary(result, totalClips))
    } catch (e) {
      toast.error(e?.body?.error || 'Dry run failed.')
    } finally {
      setWorking(false)
    }
  }

  const apply = async () => {
    setWorking(true)
    try {
      // The full field set is sent, nulls included: applying a draft that
      // CLEARED a cut must clear it in config too, not silently keep it.
      await putJson('/api/settings', { config: { video_bank: draft } })
      setPreview(null)
      onApplied?.()
      toast.success('Cuts applied — the grid re-flags instantly, nothing is deleted.')
    } catch (e) {
      toast.error(e?.body?.error || 'Could not save the cuts.')
    } finally {
      setWorking(false)
    }
  }

  return (
    <details className="rounded-lg border border-border bg-surface">
      <summary className="cursor-pointer px-3 py-2 text-sm font-semibold text-content">
        🎚 Quality cuts
      </summary>
      <div className="space-y-3 border-t border-border p-3">
        <p className="text-xs text-content-muted">
          Flags mark clips to look at — nothing is rejected for you. Leave a field
          empty to disable that cut. Preview first: these numbers only mean
          something against this bank&apos;s own distribution.
        </p>
        {/* One column at 400 px; the hint under the input, never beside it. */}
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {thresholdFields().map((f) => (
            <label key={f.key} className="block min-w-0">
              <span className="text-xs font-semibold text-content">{f.label}</span>
              <input
                type="number" step="any" inputMode="decimal"
                value={draft[f.key] ?? ''}
                onChange={(e) => setDraft(editThreshold(draft, f.key, e.target.value))}
                placeholder="off"
                className="mt-0.5 w-full rounded-md border border-border bg-app px-2 py-1 text-sm text-content"
              />
              <span className="mt-0.5 block text-[0.7rem] leading-tight text-content-subtle">
                {f.hint}
              </span>
            </label>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button type="button" onClick={dryRun} disabled={working}
            className="rounded-md border border-border bg-surface-raised px-3 py-1.5 text-xs font-semibold text-content hover:bg-surface disabled:opacity-40">
            👁 Preview what these cuts would flag
          </button>
          <button type="button" onClick={apply} disabled={working}
            className="rounded-md bg-gradient-primary px-3 py-1.5 text-xs font-semibold text-gray-950 disabled:opacity-40">
            Apply cuts
          </button>
        </div>
        {preview && (
          <p className={`text-sm ${preview.startsWith('⚠')
            ? 'text-amber-300' : 'text-content'}`}>
            {preview}
          </p>
        )}
      </div>
    </details>
  )
}
