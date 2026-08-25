import { useState } from 'react'
import { patchJson, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import {
  I2V_FIRST_FRAME_HINT, boundsAtPlayhead, boundsChanged, draftSummary,
  frameStep, newShotBounds, nudgedBounds, retouchToast, splitAvailability,
} from './videoClipEdit'
import { videoClipBoundsUrl, videoClipSplitUrl, videoSourceClipsUrl } from './videoBankApi'

/** ✂ The retouch tools, folded under the ONE <video> the lightbox mounts.
 *
 * This component adds NO player. That is not a style rule, it is the constraint
 * the whole lane is built on: Chrome stops loading new WebMediaPlayers past
 * roughly forty with no error of any kind, and there is no per-clip file to point
 * a second element at anyway. The playhead it reasons about is the lightbox's own
 * element, read from its `timeupdate`/`seeked` events and handed down as a plain
 * number of SOURCE seconds (`playheadS`).
 *
 * FOLDED BY DEFAULT. Triage is a two-key rhythm (K, R, →) over hundreds of shots,
 * and a permanent row of eight buttons under the player would compete with it for
 * the eye on every single shot, to serve the few that are mis-cut.
 *
 * NO DRAGGABLE TIMELINE, deliberately — see videoClipEdit.js. Nudges plus
 * set-to-playhead cover what triage actually needs, at 400 px and on a trackpad.
 */
export default function VideoClipTrimTools({
  bankId, clip, source, playheadS, onChanged, defaultOpen = false,
}) {
  const toast = useToast()
  const [edit, setEdit] = useState({ of: null, draft: null })
  const [saving, setSaving] = useState(false)

  // The draft follows the clip, and it is derived DURING RENDER rather than in an
  // effect. Two reasons, and the second is the one that bit: moving to the next
  // shot with → must not leave the previous shot's pending edit sitting under a
  // different caption for one frame — and an effect does not run at all in the
  // static render the suite mounts components with, so an effect-seeded draft
  // makes this whole panel return null in every test that has ever "rendered" it.
  // Idempotent, so StrictMode's double render seeds it once.
  const key = clip ? `${clip.id}:${clip.start_s}:${clip.end_s}` : null
  const draft = edit.of === key ? edit.draft
    : (clip ? { start_s: clip.start_s, end_s: clip.end_s } : null)
  const setDraft = (next) => setEdit({ of: key, draft: next })

  if (!clip || !draft) return null

  const duration = source?.duration_s ?? null
  const step = frameStep(source?.fps_native)
  const dirty = boundsChanged(clip, draft)
  const split = splitAvailability(clip, playheadS)
  const newShot = newShotBounds(playheadS, duration)

  const move = (next, why) => {
    // A refused nudge is silent on purpose when it is obvious (the button is at
    // the wall); it says why when the wall is the one the user cannot see.
    if (next) setDraft(next)
    else if (why) toast.info(why)
  }

  const run = async (fn, kind) => {
    setSaving(true)
    try {
      const d = await fn()
      toast.success(retouchToast(kind))
      onChanged?.(d, kind)
    } catch (e) {
      // 409 means a pass owns the bank — the thumbs pass would otherwise stamp a
      // thumbnail of the OLD span as current. Name it rather than say "failed".
      if (e?.status === 409) {
        toast.warning('A pass is running on this bank — stop it before re-cutting.')
      } else {
        toast.error(e?.message || 'Could not save that cut.')
      }
    } finally {
      setSaving(false)
    }
  }

  const saveBounds = () => run(
    () => patchJson(videoClipBoundsUrl(bankId, clip.id),
      { start_s: draft.start_s, end_s: draft.end_s }), 'bounds')

  const doSplit = () => run(
    () => postJson(videoClipSplitUrl(bankId, clip.id), { at_s: split.at }), 'split')

  const doCreate = () => run(
    () => postJson(videoSourceClipsUrl(bankId, clip.source_id), newShot), 'create')

  const nudge = (edge, delta) => move(nudgedBounds(draft, edge, delta, duration))
  const toPlayhead = (edge) => move(
    boundsAtPlayhead(draft, edge, playheadS, duration),
    'The playhead is not somewhere this bound can go.')

  return (
    <details open={defaultOpen || undefined} className="rounded-md border border-white/15 bg-white/5">
      <summary className="cursor-pointer px-2.5 py-1.5 text-xs font-semibold text-white/80">
        ✂ Trim &amp; split this shot
      </summary>
      <div className="space-y-2 border-t border-white/10 p-2.5">
        {['start', 'end'].map((edge) => (
          <div key={edge} className="flex flex-wrap items-center gap-1">
            <span className="w-14 shrink-0 text-[0.6875rem] font-semibold uppercase tracking-wide text-white/60">
              {edge}
            </span>
            <span className="w-16 shrink-0 font-mono text-xs text-white">
              {Number(draft[`${edge}_s`]).toFixed(2)}s
            </span>
            {[[-step, '−1f'], [-1, '−1s'], [1, '+1s'], [step, '+1f']].map(([d, label]) => (
              <button key={label} type="button" onClick={() => nudge(edge, d)}
                disabled={saving}
                title={`${label} on ${edge} (one frame = ${step.toFixed(3)}s at this file's rate)`}
                className="rounded border border-white/20 px-1.5 py-0.5 font-mono text-[0.6875rem] text-white hover:bg-white/10 disabled:opacity-30">
                {label}
              </button>
            ))}
            <button type="button" onClick={() => toPlayhead(edge)} disabled={saving}
              className="rounded border border-white/20 px-1.5 py-0.5 text-[0.6875rem] text-white hover:bg-white/10 disabled:opacity-30">
              ⇤ playhead
            </button>
          </div>
        ))}

        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-[0.6875rem] text-white/70">
            {draftSummary(draft)}
          </span>
          <div className="ml-auto flex flex-wrap gap-1.5">
            {dirty && (
              <button type="button" disabled={saving}
                onClick={() => setDraft({ start_s: clip.start_s, end_s: clip.end_s })}
                className="rounded border border-white/20 px-2 py-1 text-[0.6875rem] text-white hover:bg-white/10 disabled:opacity-30">
                Reset
              </button>
            )}
            <button type="button" onClick={saveBounds} disabled={!dirty || saving}
              className="rounded bg-indigo-600 px-2.5 py-1 text-[0.6875rem] font-semibold text-gray-950 hover:bg-indigo-500 disabled:opacity-30">
              Save bounds
            </button>
          </div>
        </div>

        <div className="flex flex-wrap gap-1.5">
          <button type="button" onClick={doSplit} disabled={!split.at || saving}
            title={split.why || `Split at ${split.at}s`}
            className="rounded border border-amber-400/50 bg-amber-500/10 px-2.5 py-1 text-[0.6875rem] font-semibold text-amber-200 hover:bg-amber-500/20 disabled:opacity-30">
            ✂ Split here
          </button>
          <button type="button" onClick={doCreate} disabled={!newShot || saving}
            title={newShot
              ? `Add a shot from ${newShot.start_s}s to ${newShot.end_s}s`
              : 'There is no room for a shot at this point of the file.'}
            className="rounded border border-white/20 px-2.5 py-1 text-[0.6875rem] font-semibold text-white hover:bg-white/10 disabled:opacity-30">
            ＋ New shot from here
          </button>
        </div>
        {split.why && (
          <p className="text-[0.6875rem] text-white/50">{split.why}</p>
        )}

        {/* The line that repays the whole panel. Nothing else in the app says it,
            and no one would guess it from a control called "trim". */}
        <p className="text-[0.6875rem] leading-snug text-sky-200/80">
          🎯 {I2V_FIRST_FRAME_HINT}
        </p>
      </div>
    </details>
  )
}
