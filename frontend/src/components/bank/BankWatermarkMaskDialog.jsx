/**
 * 🚩 Watermark mask editor for a BANK image (idea/report: Qeeyana, Reddit).
 *
 * The detector's box is a guess: it misses a second logo, swallows half the face,
 * or lands next to the mark. In a dataset that was fixable — in a bank it was not,
 * so a bad box meant rejecting the image or shipping the watermark.
 *
 * This dialog is deliberately THIN. The drawing surface is the dataset's
 * WatermarkRegionEditor, the geometry is utils/watermarkRegions, and the mask is
 * validated server-side by the dataset's own validator: the two lanes cannot
 * drift apart because there is only one of each. What is bank-specific is the
 * lane the mask feeds — hand zones are repainted by 🧽 Inpaint and skipped by
 * ✂ Auto-crop — and that is spelled out on screen instead of being folklore.
 *
 * Every edit saves immediately (one PUT per commit, serialized). The source file
 * is never touched by any of this: a bank cleans into its own working copy.
 */
import { useCallback, useRef, useState } from 'react'
import { Flag } from 'lucide-react';
import { putJson } from '../../api/fetchClient'
import { useFocusTrap } from '../../hooks/useFocusTrap'
import { HelpBadge } from '../../help/HelpMode'
import {
  MAX_WATERMARK_REGIONS, cloneWatermarkRegions, deleteSelectedWatermarkRegion,
} from '../../utils/watermarkRegions.js'
import WatermarkRegionEditor from '../dataset/WatermarkRegionEditor'
import { applyMaskResponse, initialMask, maskPayload, maskStatus } from './bankWatermarkMask.js'

const TONE_CLASS = { ok: 'text-emerald-300', info: 'text-white/70', warn: 'text-amber-300' }

export default function BankWatermarkMaskDialog({ bankId, image, onSaved, onClose }) {
  const start = initialMask(image)
  const [regions, setRegions] = useState(start.regions)
  const [manual, setManual] = useState(start.manual)
  const [selected, setSelected] = useState(start.regions.length ? 0 : null)
  const [addMode, setAddMode] = useState(false)
  const [save, setSave] = useState({ status: 'saved', error: null })
  const dialogRef = useRef(null)
  const chainRef = useRef(Promise.resolve())
  const lastRef = useRef(null)          // the payload of the newest save, for Retry

  useFocusTrap(dialogRef, true)

  const persist = useCallback((regionsOrNull, visible, nextManual) => {
    const body = maskPayload(regionsOrNull)
    lastRef.current = { body, visible, manual: nextManual }
    setRegions(cloneWatermarkRegions(visible))
    setManual(nextManual)
    setSave({ status: 'saving', error: null })
    chainRef.current = chainRef.current
      .catch(() => undefined)
      .then(() => putJson(`/api/bank/${bankId}/image/${image.id}/watermark-regions`, body))
      .then((response) => {
        const merged = applyMaskResponse(image, response)
        const next = initialMask(merged)
        setRegions(next.regions)
        setManual(next.manual)
        setSelected((current) => (next.regions.length
          ? Math.min(current ?? 0, next.regions.length - 1) : null))
        setSave({ status: 'saved', error: null })
        onSaved?.(merged)
      })
      .catch((e) => {
        // Say it, and keep the drawing on screen: a mask that silently failed to
        // save is exactly the failure this whole feature exists to prevent.
        setSave({ status: 'failed', error: e?.message || 'Could not save the mask' })
      })
  }, [bankId, image, onSaved])

  const commit = useCallback((next) => {
    setAddMode(false)
    persist(next, next, true)
  }, [persist])

  const deleteSelected = useCallback(() => {
    const next = deleteSelectedWatermarkRegion(regions, selected)
    if (next.selectedIndex === null && selected === null) return
    setSelected(next.selectedIndex)
    commit(next.regions)
  }, [commit, regions, selected])

  const resetDetection = useCallback(() => {
    const detected = Array.isArray(image?.watermark_bbox) && image.watermark_bbox.length === 4
      ? [[...image.watermark_bbox]] : []
    setSelected(detected.length ? 0 : null)
    setAddMode(false)
    persist(null, detected, false)
  }, [image, persist])

  const retry = useCallback(() => {
    const last = lastRef.current
    if (!last) return
    persist(last.body.regions, last.visible, last.manual)
  }, [persist])

  const status = maskStatus({ regions, manual })
  const saving = save.status === 'saving'
  const atLimit = regions.length >= MAX_WATERMARK_REGIONS
  const hasSelection = Number.isInteger(selected) && selected >= 0 && selected < regions.length
  // min-h-11 = 44 px: the whole dialog is usable with a thumb at 400 px wide,
  // which is where drawing a rectangle by hand is actually hard.
  const btn = 'min-h-11 rounded-lg border border-white/20 bg-white/10 px-3 text-xs '
    + 'font-semibold text-white hover:bg-white/20 disabled:opacity-40'

  return (
    // bg-black is OPAQUE on purpose: this editor opens ON TOP of the ▶ Review
    // lightbox, and two stacked translucent layers let that viewer's header bleed
    // through the controls the user is aiming at.
    <div ref={dialogRef} role="dialog" aria-modal="true" aria-label="Edit the watermark mask"
      className="fixed inset-0 z-[9998] flex flex-col bg-black">
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-white/10 px-3 py-2">
        <span className="inline-flex items-center gap-1.5 text-sm font-semibold text-white"><Flag aria-hidden="true" className="h-4 w-4" /> Watermark mask</span>
        <span className="max-w-[14rem] truncate text-xs text-white/60" title={image?.name}>
          {image?.name}
        </span>
        <HelpBadge topic="bank-edit-watermark-mask" className="self-center" />
        <button type="button" onClick={onClose} disabled={saving}
          title="Close (the mask saves as you edit)" aria-label="Close the mask editor"
          className="ml-auto h-11 w-11 rounded-full bg-white/10 text-lg leading-none text-white hover:bg-white/20 disabled:opacity-40">
          ✕
        </button>
      </div>

      {/* [container-type:size] lets the editor cap its height to THIS cell, so a
          portrait photo on a 400 px phone never paints its zones over the
          controls below (and steals their taps). */}
      <div className="flex min-h-0 flex-1 items-center justify-center p-2 [container-type:size]">
        {/* ?original=1 on purpose: the whole bank watermark lane — detection,
            crop and inpaint — reads the UNROTATED source file, so the mask must
            be drawn in that same space or the zones would land elsewhere. A
            turned image is therefore shown as it sits on disk, and the note
            below says so instead of letting it look like a bug. */}
        <WatermarkRegionEditor
          src={`/api/bank/${bankId}/file/${image.id}?original=1`}
          alt={image?.name || `Bank image ${image.id}`}
          regions={regions}
          disabled={saving}
          addMode={addMode}
          selectedIndex={selected}
          onAddModeChange={setAddMode}
          onSelectedIndexChange={setSelected}
          onCommit={commit}
        />
      </div>

      <div className="shrink-0 space-y-2 border-t border-white/10 bg-black/70 px-3 py-2.5">
        <p className={`text-center text-xs ${TONE_CLASS[status.tone]}`}>{status.text}</p>

        {save.status === 'failed' && (
          <div role="alert" className="flex flex-wrap items-center justify-center gap-2 text-xs text-rose-300">
            <span>⚠ {save.error} — the mask on screen is NOT saved.</span>
            <button type="button" onClick={retry} className={btn}>Retry save</button>
          </div>
        )}

        <div className="flex flex-wrap items-center justify-center gap-2">
          <button type="button" aria-pressed={addMode} onClick={() => setAddMode((v) => !v)}
            disabled={saving || atLimit}
            title={atLimit ? `Maximum of ${MAX_WATERMARK_REGIONS} zones reached`
              : 'Then drag on the image to draw a zone over the watermark'}
            className={addMode
              ? 'min-h-11 rounded-lg border border-sky-300 bg-sky-500/25 px-3 text-xs font-semibold text-sky-100'
              : btn}>
            + Add zone
          </button>
          <button type="button" onClick={deleteSelected} disabled={saving || !hasSelection}
            title={hasSelection ? `Delete zone ${selected + 1}` : 'Tap a zone to select it first'}
            className={btn}>
            Delete zone
          </button>
          <button type="button" onClick={resetDetection} disabled={saving || !manual}
            title="Throw away the hand-drawn zones and go back to the detected box"
            className={btn}>
            Reset to detected
          </button>
          <span aria-live="polite" className={`text-xs font-semibold ${saving ? 'text-amber-200'
            : save.status === 'failed' ? 'text-rose-300' : 'text-emerald-300'}`}>
            {saving ? 'Saving…' : save.status === 'failed' ? '⚠ Save failed' : '✓ Saved'}
          </span>
        </div>

        <p className="text-center text-[11px] text-white/45">
          Drag a zone to move it, its corners to resize. Your own file is never modified —
          cleaning writes a separate copy inside the bank.
          {image?.rotation
            ? ' This image is shown unrotated: watermark cleaning works on your'
              + ' original file, which the ↻ turn never changed.'
            : ''}
        </p>
      </div>
    </div>
  )
}
