/**
 * ✂ Edits — cropping and upscaling done IN THE BANK, and the one button that
 * takes either of them back.
 *
 * WHY IT EXISTS. Asked for by nofaceman on Discord, backed by mr.arrow: the Bank
 * is where the filtering and the curation live, but it had neither crop nor
 * upscale. Reframing or upscaling a shot meant Bank → Dataset → export into a
 * NEW bank → Dataset again — a round trip around the screen you are already on.
 * With both here, the loop is curate → edit → re-analyse → promote, in one place.
 *
 * WHAT THIS PANEL HOLDS, AND WHAT IT DOES NOT.
 *   ✨ Upscale & improve is a PASS: a scope, a count, a progress bar and ⏹ Stop,
 *     so it opens the same launch window as every other pass.
 *   ✂ Crop is not. It is one image, one box, drawn by hand — it lives in ▶
 *     Review, which is the only place the Bank shows an image big enough to draw
 *     on. This panel says so rather than staying silent: a feature nobody can
 *     find is indistinguishable from one that was never shipped, and that
 *     conclusion has already been reached twice on this app.
 *   ↩ Revert undoes BOTH, for the same reason ↩ Undo cleaning does: the only
 *     thing deleted is a copy the app made itself.
 *
 * The "which engine can run, and why not" logic lives in the JSX-free
 * `bankEdits.js` so `node --test` covers it; this file is the shell.
 */
import { useEffect, useRef, useState } from 'react'
import { postJson } from '../../api/fetchClient'
import PassDialog from './PassDialog.jsx'
import KleinModelSetting from '../shared/KleinModelSetting'
import { useCapabilities } from '../../context/CapabilitiesContext'
import { useToast } from '../common/Toast'
import { HelpBadge } from '../../help/HelpMode'
import {
  BANK_IMPROVE_PROMISE, bankImproveEngines, editCounts, editSummary,
  revertConfirmMessage, revertOutcomeMessage,
} from './bankEdits.js'
import { passScopeCount } from './bankPassScope.js'

export default function BankEditPanel({
  bankId, live, payload = null, selectedIds = [], onChanged,
}) {
  const { caps } = useCapabilities()
  const toast = useToast()
  const [engine, setEngine] = useState(null)
  /* The launch window's scope lives HERE, not inside the dialog: closing a
     window must not silently undo a choice — the rule every other pass follows. */
  const [improveOpen, setImproveOpen] = useState(false)
  const [scope, setScope] = useState('')
  const [busy, setBusy] = useState(false)

  const counts = editCounts(payload)
  // What the pass has left to do in the chosen scope — the figure the engine
  // buttons need to know whether they can do anything at all. `null` is "the
  // payload has not answered yet", and is deliberately NOT folded into 0.
  const todo = passScopeCount(payload, 'improve', scope, false)

  /* COLLAPSED BY DEFAULT, like the 🚩 panel next to it: editing is an occasional
     errand, not the reason this page is open, and an always-expanded panel
     pushes the grid down on every visit. A bank that ALREADY HAS edits is the
     one exception — that is the state where this panel has something to report
     and an ↩ Revert to offer, so it opens itself.

     Once, and never again: the payload is re-fetched every couple of seconds
     while a pass runs, and re-opening on each new count would pop the panel back
     up under the user's hands every time they cropped an image in ▶ Review. The
     initial state reads the counts as well, for the render where the payload is
     already in hand; the effect only catches the case where it lands later. */
  const [open, setOpen] = useState(counts.total > 0)
  const autoOpened = useRef(open)
  useEffect(() => {
    if (!autoOpened.current && counts.total > 0) {
      autoOpened.current = true
      setOpen(true)
    }
  }, [counts.total])
  const engines = bankImproveEngines(caps, { live, todo })
  const picked = engines.find((e) => e.id === engine) || engines.find((e) => !e.disabled) || null
  const kleinPicked = (picked?.id || engine) === 'klein'

  /* The launch answers {ok,error} instead of toasting: the window owns its own
     refusal surface and stays open with the scope intact, so "every image in
     this bank has already been improved" lands next to the scope that produced
     it — and so does the itemized 409 a missing engine returns. */
  const launchImprove = async ({ statuses, imageIds }) => {
    const body = {
      ...(picked?.id ? { engine: picked.id } : {}),
      ...(statuses ? { statuses } : {}),
      ...(imageIds === 'selection' && selectedIds.length
        ? { image_ids: [...selectedIds] } : {}),
    }
    try {
      await postJson(`/api/bank/${bankId}/improve`, body)
    } catch (e) {
      return { ok: false, error: e?.message || 'Could not start the run.' }
    }
    await onChanged?.()
    return { ok: true }
  }

  const revert = async () => {
    const ids = selectedIds.length ? [...selectedIds] : null
    if (!window.confirm(revertConfirmMessage(payload, ids))) return
    setBusy(true)
    try {
      const d = await postJson(`/api/bank/${bankId}/edits/revert`,
        ids ? { image_ids: ids } : {})
      const msg = revertOutcomeMessage(d)
      toast[msg.type](msg.text)
      await onChanged?.()
    } catch (e) {
      toast.error(e?.message || 'Could not revert — nothing was changed.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div id="bank-edits" data-workspace-focus
      className="rounded-lg border border-border bg-surface-raised">
      <button type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open}
        className="flex w-full flex-wrap items-center gap-x-2 gap-y-1 p-3 text-left">
        <span className="text-sm font-semibold text-content">✂ Edits</span>
        <span className="text-[0.6875rem] text-content-subtle">{editSummary(payload)}</span>
        <HelpBadge topic="action-bank-crop" />
        <span aria-hidden className="ml-auto text-xs text-content-subtle">{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div className="space-y-2 px-3 pb-3">
          <p className="text-[0.6875rem] text-content-subtle">
            crop and upscale here, re-analyse, then promote — your original files are
            never modified
          </p>

          {/* ✂ WHERE THE CROP IS. Not a button that opens nothing: it names the
              gesture and the key, because the crop needs a full-size image and
              the grid tile is 96 px tall. */}
          <p className="text-xs text-content-muted">
            ✂ <b>Crop</b> is per image: open <b>▶ Review</b> (or the ▶ on a tile) and
            press <b>C</b>. Nothing is resampled — a Bank sits upstream of the training
            resolution, so the cut keeps its pixels and the dataset decides the size
            when it imports.
          </p>

          <div className="rounded-lg border border-border bg-app/40 p-2.5 space-y-1.5">
            <div className="flex items-baseline gap-1.5">
              <span className="text-sm font-semibold text-content">✨ Upscale &amp; improve</span>
              <span className="text-[0.6875rem] text-content-subtle">
                {todo == null
                  ? 'counting…'
                  : (todo ? `${todo} image(s) in this scope` : 'nothing left in this scope')}
              </span>
            </div>
            <p className="text-[0.6875rem] leading-snug text-content-subtle">
              Re-renders each image at a higher resolution. {BANK_IMPROVE_PROMISE}
            </p>
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[0.6875rem] font-semibold uppercase tracking-wide text-content-subtle">
                Engine
              </span>
              <div className="flex items-center gap-1 rounded-lg border border-border bg-app/60 p-0.5 text-xs">
                {engines.map((e) => (
                  <button key={e.id} type="button" aria-pressed={picked?.id === e.id}
                    onClick={() => setEngine(e.id)} disabled={e.disabled} title={e.title}
                    className={`rounded-md px-2.5 py-1 font-semibold disabled:opacity-40 ${
                      picked?.id === e.id
                        ? 'bg-sky-500/25 text-sky-100' : 'text-content-subtle hover:text-content'}`}>
                    {e.label}
                  </button>
                ))}
              </div>
              {/* WHICH Klein model is about to re-render these images. A bank has
                  no dataset to inherit a choice from, so there is nothing to pick
                  — but "no choice" was never a reason to stay silent about it. */}
              {kleinPicked && <KleinModelSetting className="w-full" />}
            </div>
            <button type="button" onClick={() => setImproveOpen(true)}
              disabled={!picked || picked.disabled}
              title={picked?.disabled ? picked.reason : (picked?.title || 'Upscale & improve')}
              className="rounded-lg border border-sky-400/40 bg-sky-500/15 px-3 py-1.5 text-sm font-semibold text-sky-200 disabled:opacity-40">
              ✨ Upscale &amp; improve…
            </button>
            <p className="text-[0.6875rem] text-content-subtle">
              {picked?.disabled
                ? picked.reason
                : 'One ComfyUI round-trip per image — minutes apiece on a modest card. '
                  + '⏹ Stop ends the run between two images and keeps what is done.'}
            </p>
          </div>

          {/* ↩ Only when there is something to take back. A revert button on a
              bank that has never been edited is a button that can only say "0". */}
          {counts.total > 0 && (
            <div className="flex flex-wrap items-center gap-2">
              <button type="button" onClick={revert} disabled={live || busy}
                title={selectedIds.length
                  ? `Throw away the crops and upscales of the ${selectedIds.length} selected image(s) and go back to what this bank started from.`
                  : 'Throw away every ✂ crop and ✨ upscale made in this bank and go back to what it started from.'}
                className="rounded-md border border-border px-2 py-1 text-xs font-medium text-content-muted hover:bg-surface-raised hover:text-content disabled:opacity-40">
                ↩ Revert {selectedIds.length ? `selection (${selectedIds.length})` : `all (${counts.total})`}
              </button>
              <span className="text-[0.6875rem] text-content-subtle">
                Deletes only copies the app made. The measurements taken from the edited
                pixels go with them, so those images are analysed again.
              </span>
            </div>
          )}
        </div>
      )}

      {improveOpen && (
        <PassDialog passId="improve" payload={payload} live={live}
          selectionSize={selectedIds.length}
          scope={scope} onScope={setScope}
          onClose={() => setImproveOpen(false)}
          onLaunch={launchImprove} />
      )}
    </div>
  )
}
