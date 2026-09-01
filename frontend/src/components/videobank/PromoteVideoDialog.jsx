import { useEffect, useMemo, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import {
  frameOptions, defaultFrames, needsManualFrames, sizeOptions,
  promoteProblem, promotePayload, promoteScopeLabel, datasetScaleNote,
  insetProblem, insetHint, insetOutcome,
  capProblem, capHint, capBalanceNote,
} from './videoTargetChoice'
import {
  uncaptionedWarning, overBudgetWarning, overTokenBudgetWarning, servedShortNote,
} from './videoClipSearch'
import { passBlockedBy } from './videoCapability'
import VideoTargetPicker from './VideoTargetPicker'

/** 🎬 Turn the shots you kept into a training set.
 *
 * THE TWO FIELDS THAT MAKE THIS SCREEN WORTH ITS SPACE are `training_verified`
 * and `licence_note`, and they are rendered NEXT TO THE CHOICE, not in a doc:
 *
 *  · exactly one target in the catalogue is known to have a working LoRA
 *    trainer. Picking one of the other three and finding out afterwards costs a
 *    week of cutting, captioning and GPU time on a dataset nothing can read;
 *  · MiniMax H3's licence grants NO rights in the EU, the UK, South Korea or the
 *    USA, and that restriction reaches the OUTPUTS — so keeping the training
 *    private is not a way around it. Someone must not learn that from a forum
 *    thread after building the set.
 *
 * The length selector offers ONLY the catalogue's frame counts. Never a free
 * field in seconds: the frame rule is a property of each model's VAE (29 frames
 * is legal for Wan and illegal for LTX; MiniMax wants f % 17 == 5), and no
 * trainer refuses an illegal count — they floor it in latent space, in silence.
 */
export default function PromoteVideoDialog({
  bankId, capability, keepCount, selectedIds, onClose, onDone,
}) {
  const toast = useToast()
  const [targets, setTargets] = useState(null)
  const [targetKey, setTargetKey] = useState('')
  const [name, setName] = useState('')
  const [triggerWord, setTriggerWord] = useState('')
  const [frames, setFrames] = useState(null)
  const [sizeKey, setSizeKey] = useState('source')
  // ✂ Per-end trim, in seconds. Zero by default and kept as TEXT while typing:
  // "0." and "" are states a number input passes through, and coercing them
  // early wipes the field under the user's cursor.
  const [edgeInset, setEdgeInset] = useState('')
  // 🎚 Per-source cap. Text for the same reason as the trim: "" and mid-typing
  // states are values a number input passes through.
  const [maxPerSource, setMaxPerSource] = useState('')
  const [busy, setBusy] = useState(false)
  // Escape closes, unless a build is mid-flight — the same guard PassDialog
  // applies (a dialog must never vanish while its request is still running).
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape' && !busy) onClose?.() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [busy, onClose])
  const [error, setError] = useState(null)

  useEffect(() => {
    let alive = true
    apiFetch('/api/video/targets')
      .then((d) => {
        if (!alive) return
        const list = d.targets || []
        setTargets(list)
        // Default to the one target known to be trainable rather than to the
        // first row — the default is a recommendation whether we mean it to be
        // one or not.
        const preferred = list.find((t) => t.training_verified) || list[0]
        if (preferred) { setTargetKey(preferred.key); setFrames(defaultFrames(preferred)) }
      })
      .catch((e) => { if (alive) setError(e?.message || 'Could not load the target list.') })
    return () => { alive = false }
  }, [])

  const target = useMemo(
    () => (targets || []).find((t) => t.key === targetKey) || null, [targets, targetKey])
  const options = frameOptions(target)
  const sizes = sizeOptions(target)
  const manualFrames = needsManualFrames(target)
  const blocked = passBlockedBy(capability, 'promote')
  const insetIssue = insetProblem(edgeInset)
  const capIssue = capProblem(maxPerSource)
  const problem = blocked ? blocked.why
    : (promoteProblem({ name, target, frames }) || insetIssue || capIssue)
  const size = sizes.find((s) => s.key === sizeKey) || sizes[0]

  const pick = (key) => {
    const next = (targets || []).find((t) => t.key === key)
    setTargetKey(key)
    setFrames(defaultFrames(next))
    setSizeKey('source')
  }

  const submit = async (e) => {
    e.preventDefault()
    if (busy || problem) return
    setBusy(true)
    setError(null)
    try {
      const d = await postJson(`/api/video-bank/${bankId}/promote`,
        promotePayload({ name, targetKey, frames, size, ids: selectedIds,
          edgeInsetS: edgeInset, maxPerSource, triggerWord }))
      toast.success(`Building “${d.name}” — ${d.clips} clip(s) being encoded.`)
      // Said out loud rather than left in the job line: these clips were removed
      // by the user's OWN setting, and it is the only limit here they can undo.
      const cost = insetOutcome(d.composition)
      if (cost) toast.warning(cost)
      if (d.composition?.high_fps_clips > 0) {
        toast.warning(`${d.composition.high_fps_clips} clip(s) come from 48+ fps `
          + 'sources — often slow-motion footage, which teaches floaty movement. '
          + 'Worth a second look if that is not the style you want.')
      }
      // What the set turned out to be MADE OF. 60% from one source is invisible
      // on disk — the folder looks exactly like a diverse one — so the only
      // place it can be seen is here, right after it happened.
      const balance = capBalanceNote(d.composition, maxPerSource)
      if (balance) toast.warning(balance)
      // An empty sidecar trains as an EMPTY PROMPT and the trainer says nothing.
      // The one limit here that silently degrades the dataset itself.
      const captions = uncaptionedWarning(d.composition)
      if (captions) toast.warning(captions)
      // Its mirror image: a caption the encoder will CUT without saying so.
      const budget = overBudgetWarning(d.composition)
      if (budget) toast.warning(budget)
      // Tokens decide where words only warn: what the encoder will actually cut.
      const tokens = overTokenBudgetWarning(d.composition)
      if (tokens) toast.warning(tokens)
      // And what the export did about it — said, so a sidecar shorter than its
      // caption is no mystery.
      const short = servedShortNote(d.composition)
      if (short) toast.info(short, 8000)
      onDone?.(d)
      onClose?.()
    } catch (err) {
      // The server's 400 NAMES a legal length or a valid size. Showing it beats
      // any message we could invent from here.
      setError(err?.message || 'Could not start the export.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div role="dialog" aria-modal="true" aria-label="Build a video training set"
      data-probe-layer
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-3 sm:p-4"
      onMouseDown={(e) => { if (e.target === e.currentTarget && !busy) onClose?.() }}>
      <form onSubmit={submit}
        className="w-full max-w-lg max-h-[90vh] space-y-4 overflow-y-auto rounded-xl border border-border bg-surface-overlay p-4 shadow-2xl sm:p-5">
        <h2 className="text-base font-bold text-content">🎬 Build a video training set</h2>
        <p className="text-sm text-content-muted">
          Encodes {promoteScopeLabel((selectedIds || []).length, keepCount)} into a flat
          folder of clips with caption sidecars. This is the only step that writes
          video files — your source folder is never touched.
        </p>
        {(() => {
          const note = datasetScaleNote((selectedIds || []).length || keepCount)
          if (!note) return null
          const tone = note.tone === 'warning' ? 'text-amber-300'
            : note.tone === 'good' ? 'text-emerald-300' : 'text-content-subtle'
          return <p className={`text-xs ${tone}`}>{note.text}</p>
        })()}

        <div>
          <label htmlFor="video-ds-name" className="block text-sm font-medium text-content">Name</label>
          <input id="video-ds-name" value={name} onChange={(e) => setName(e.target.value)}
            placeholder="wan 14b — city rushes" required
            className="mt-1 w-full rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content" />
        </div>

        <div>
          <label htmlFor="video-ds-trigger" className="block text-sm font-medium text-content">
            Trigger word <span className="font-normal text-content-subtle">(optional)</span>
          </label>
          <input id="video-ds-trigger" value={triggerWord}
            onChange={(e) => setTriggerWord(e.target.value)}
            placeholder="mychar"
            className="mt-1 w-full rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content" />
          <p className="mt-1 text-xs text-content-subtle">
            Prepended once to every clip&rsquo;s caption at export. Use it in ONE place —
            here or in the captions, never both.
          </p>
        </div>

        <fieldset>
          <legend className="text-sm font-medium text-content">Target model</legend>
          {/* Its own component so a test can mount it with the real catalogue
              and prove the licence note and the "no trainer" label are on
              screen — see VideoTargetPicker. */}
          <VideoTargetPicker targets={targets} targetKey={targetKey} onPick={pick} />
        </fieldset>

        <div className="grid gap-3 sm:grid-cols-2">
          <div>
            <label htmlFor="video-ds-frames" className="block text-sm font-medium text-content">
              Clip length
            </label>
            {manualFrames ? (
              <>
                <input id="video-ds-frames" type="number" min="1" step="1"
                  value={frames ?? ''} onChange={(e) => setFrames(Number(e.target.value) || null)}
                  className="mt-1 w-full rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content" />
                {/* "No presets", never "any length is fine": we have no verified
                    lengths for this target, which is not the same claim. */}
                <p className="mt-1 text-xs text-content-muted">
                  In frames. We have no verified lengths for this target, so nothing is
                  suggested — check what your trainer expects.
                </p>
              </>
            ) : (
              <>
                <select id="video-ds-frames" value={frames ?? ''}
                  onChange={(e) => setFrames(Number(e.target.value) || null)}
                  className="mt-1 w-full rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content">
                  {options.map((o) => (
                    <option key={o.frames} value={o.frames}>{o.label}</option>
                  ))}
                </select>
                <p className="mt-1 text-xs text-content-muted">
                  Only lengths this model’s VAE can ingest. Seconds are shown at its own
                  frame rate — they are not something you type.
                </p>
              </>
            )}
          </div>
          <div>
            <label htmlFor="video-ds-size" className="block text-sm font-medium text-content">
              Size
            </label>
            <select id="video-ds-size" value={sizeKey} onChange={(e) => setSizeKey(e.target.value)}
              className="mt-1 w-full rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content">
              {sizes.map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}
            </select>
            {/* The chosen option's own caveat wins over the generic reassurance:
                for a canvas-capped target, "keeping the source size is fine" is
                exactly the sentence that would be false. */}
            {size?.hint ? (
              <p className="mt-1 text-xs text-amber-300">⚠ {size.hint}</p>
            ) : (
              <p className="mt-1 text-xs text-content-muted">
                Suggestions mirror the model’s own inference sizes — they are not training
                limits. Keeping the source size is fine.
              </p>
            )}
          </div>
        </div>

        <div>
          <label htmlFor="video-ds-cap" className="block text-sm font-medium text-content">
            Max clips per source <span className="text-content-subtle">(optional)</span>
          </label>
          <input id="video-ds-cap" type="number" min="1" step="1"
            value={maxPerSource} onChange={(e) => setMaxPerSource(e.target.value)}
            placeholder="no cap"
            className="mt-1 w-full rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content sm:w-40" />
          {capIssue ? (
            <p className="mt-1 text-xs text-rose-300">⚠ {capIssue}</p>
          ) : (
            <p className="mt-1 text-xs text-content-muted">{capHint()}</p>
          )}
        </div>

        {/* ✂ The edge trim. Below the target grid because it is a refinement of
            WHAT gets cut, not of which model it is cut for. */}
        <div>
          <label htmlFor="video-ds-inset" className="block text-sm font-medium text-content">
            Trim each end (seconds)
          </label>
          <input id="video-ds-inset" type="number" min="0" step="0.05"
            value={edgeInset} onChange={(e) => setEdgeInset(e.target.value)}
            placeholder="0"
            className="mt-1 w-full rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content sm:w-40" />
          {insetIssue ? (
            <p className="mt-1 text-xs text-rose-300">⚠ {insetIssue}</p>
          ) : (
            <p className="mt-1 text-xs text-content-muted">
              {insetHint(edgeInset)
                || 'A shot boundary is where a cut just happened, so the first and '
                 + 'last frames are often dissolves. 0.25 is a common trim. Leave '
                 + 'empty to cut exactly on the detected bounds.'}
            </p>
          )}
        </div>

        {error && (
          <p role="alert" className="rounded-md border border-rose-500/60 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
            {error}
          </p>
        )}

        <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
          <button type="button" onClick={onClose}
            className="rounded-md border border-border px-3 py-1.5 text-sm text-content hover:bg-surface-raised">
            Cancel
          </button>
          <button type="submit" disabled={busy || !!problem} title={problem || undefined}
            className="rounded-md bg-gradient-primary px-4 py-1.5 text-sm font-semibold text-gray-950 disabled:opacity-40">
            {busy ? 'Starting…' : `🎬 Encode ${promoteScopeLabel((selectedIds || []).length, keepCount)}`}
          </button>
        </div>
        {problem && <p className="text-right text-xs text-content-muted">{problem}</p>}
      </form>
    </div>
  )
}
