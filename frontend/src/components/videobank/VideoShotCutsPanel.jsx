import { useState } from 'react'
import { postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { videoRecutUrl, videoShotDryRunUrl, videoShotThresholdUrl } from './videoBankApi'
import {
  dryRunSummary, parseThreshold, recutSummary, sweepRows, thresholdLabel,
} from './videoShotCuts'

/** 🎬 Where the cuts get argued with.
 *
 * The detector answers "how likely is a transition on this frame" for every
 * frame, and the shot list is a THRESHOLD applied to that answer. Until this
 * panel existed the threshold was 0.5 and disagreeing with it meant running the
 * whole detector again — minutes per hour of footage, on the GPU. The
 * probabilities are kept on disk now, so this panel changes a number and re-cuts
 * a whole bank without decoding a single frame.
 *
 * PREVIEW FIRST, exactly like the quality cuts, and for a sharper reason: the
 * number here does not FLAG shots, it CREATES them. Moving 0.5 to 0.3 on a
 * folder of rushes is the difference between two hundred clips and nine hundred,
 * and nobody can guess which one their footage gives. So the preview counts what
 * each threshold would actually leave, on this bank, before anything is cut.
 *
 * The field is empty when the bank inherits the app default, and empty is NOT
 * zero — zero is a threshold that fires on every frame. The placeholder says
 * which default is in force so the empty box is never mistaken for "none".
 */
export default function VideoShotCutsPanel({ bankId, shotDetect, onChanged }) {
  const toast = useToast()
  const fallback = shotDetect?.default ?? 0.5
  const [draft, setDraft] = useState(
    shotDetect?.threshold === null || shotDetect?.threshold === undefined
      ? '' : String(shotDetect.threshold))
  const [preview, setPreview] = useState(null)
  const [working, setWorking] = useState(false)
  const cached = shotDetect?.cached_sources || 0

  const withThreshold = async (run) => {
    let value
    try {
      value = parseThreshold(draft)
    } catch (e) {
      toast.error(e.message)
      return
    }
    setWorking(true)
    try {
      await run(value)
    } catch (e) {
      toast.error(e?.body?.error || 'That did not go through.')
    } finally {
      setWorking(false)
    }
  }

  const dryRun = () => withThreshold(async (value) => {
    // The ladder is left to the server: a fixed spread plus whatever is in
    // force, so two banks answer the same question and can be compared.
    const result = await postJson(videoShotDryRunUrl(bankId), {})
    setPreview({ ...result, typed: value })
  })

  const save = () => withThreshold(async (value) => {
    await postJson(videoShotThresholdUrl(bankId), { threshold: value })
    onChanged?.()
    toast.success('Saved. Nothing is re-cut until you ask for it.')
  })

  const recut = () => withThreshold(async (value) => {
    await postJson(videoShotThresholdUrl(bankId), { threshold: value })
    const result = await postJson(videoRecutUrl(bankId), {})
    setPreview(null)
    onChanged?.()
    toast.success(`Re-cut: ${recutSummary(result)}`)
  })

  return (
    <details className="rounded-lg border border-border bg-surface">
      <summary className="cursor-pointer px-3 py-2 text-sm font-semibold text-content">
        🎬 Find shots — cut sensitivity
      </summary>
      <div className="space-y-3 border-t border-border p-3">
        <p className="text-xs text-content-muted">
          A higher number cuts less often — fewer, longer shots, and fewer cuts
          invented inside a single take. A lower one catches the boundaries a
          slow dissolve hides, and finds more of them everywhere else too.
          Preview first: no threshold is right for every folder.
        </p>
        {/* One column at 400 px; the hint sits under the field, never beside it. */}
        <label className="block min-w-0 sm:max-w-xs">
          <span className="text-xs font-semibold text-content">
            Threshold for this bank
          </span>
          <input
            id="video-shot-threshold"
            type="number" step="0.05" min="0" max="1" inputMode="decimal"
            value={draft} onChange={(e) => setDraft(e.target.value)}
            placeholder={thresholdLabel(null, fallback)}
            className="mt-0.5 w-full rounded-md border border-border bg-app px-2 py-1 text-sm text-content"
          />
          <span className="mt-0.5 block text-[0.7rem] leading-tight text-content-subtle">
            Leave empty to use the app default ({Number(fallback).toFixed(2)}).
            Empty is not zero — zero would cut on every frame.
          </span>
        </label>

        <div className="flex flex-wrap items-center gap-2">
          <button type="button" onClick={dryRun} disabled={working || !cached}
            className="rounded-md border border-border bg-surface-raised px-3 py-1.5 text-xs font-semibold text-content hover:bg-surface disabled:opacity-40">
            👁 Preview how many shots each threshold gives
          </button>
          <button type="button" onClick={save} disabled={working}
            className="rounded-md border border-border bg-surface-raised px-3 py-1.5 text-xs font-semibold text-content hover:bg-surface disabled:opacity-40">
            Save
          </button>
          <button type="button" onClick={recut} disabled={working || !cached}
            title="Cuts every file again from what the detector already measured — no GPU, no waiting"
            className="rounded-md bg-gradient-primary px-3 py-1.5 text-xs font-semibold text-gray-950 disabled:opacity-40">
            Save &amp; re-cut this bank
          </button>
        </div>

        {/* The honest disabled state. Without this line the two buttons are
            simply grey and the user has no idea what would make them work. */}
        {!cached && (
          <p className="text-xs text-amber-200/90">
            No file in this bank has been through 🎬 Find shots yet. Run it once
            and every threshold change afterwards is instant.
          </p>
        )}

        {preview && (
          <div className="space-y-1.5">
            <ul className="grid gap-1 sm:grid-cols-2">
              {sweepRows(preview).map((row) => (
                <li key={row.threshold}
                  className={`flex items-baseline justify-between gap-2 rounded border px-2 py-1 text-xs ${
                    row.current ? 'border-primary/60 bg-primary/10 text-content'
                      : 'border-border bg-app text-content-muted'}`}>
                  <span className="font-mono">{row.threshold.toFixed(2)}</span>
                  <span className="text-content">{row.shots} shots</span>
                  <span className="text-[0.7rem] text-content-subtle">
                    {row.deltaLabel}
                  </span>
                </li>
              ))}
            </ul>
            <p className="text-xs text-content-subtle">{dryRunSummary(preview)}</p>
          </div>
        )}

        <p className="text-[0.7rem] leading-tight text-content-subtle">
          A re-cut never touches a shot you cut by hand, a shot already in a
          built dataset, or a file you marked as a single take. Everything else
          is replaced, and loses its thumbnail and its quality scores — they
          measured bounds that no longer exist.
        </p>
      </div>
    </details>
  )
}
