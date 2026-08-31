/**
 * 🚩 Watermark cleaning — the Bank's two-level escalation, launched by hand.
 *
 * A bank points at a folder that belongs to the user and we NEVER write to it,
 * so a cleaned image is a separate copy kept by the app; the original stays on
 * disk untouched. That is what lets both levels be tried freely: ↩ Undo just
 * throws our copy away.
 *
 * Level 1 (🚩 Find) records WHERE each mark sits — the two below route on it.
 * Level 2 (✂ Auto-crop) cuts off marks sitting in a border — no model, no GPU,
 * and it invents no pixel. Level 3 (🧽 Inpaint) repaints whatever level 2 could
 * not handle, with LaMa (fast) or Klein (slower, also handles marks ON the
 * subject). Each level shows its own remaining/handled counts, because the whole
 * point is to see how far down the funnel the bank already is.
 *
 * LEVELS 2 AND 3 OPEN A LAUNCH WINDOW, like every other pass. They were the last
 * two actions on this page that fired straight from the click, and they are the
 * only two that produce new image files: a real bank offered "✂ Auto-crop
 * (16 052)" and "🧽 Inpaint (16 507)" with no way to say which images. The window
 * carries the kept / undecided / unkept / all scope, the selection, the measured
 * count per line — and, because these two write files, what ↩ Undo really takes
 * back and what it does not.
 *
 * All of the "which button is live, and why not" logic lives in the JSX-free
 * `bankWatermark.js` so `node --test` covers it; this file is the shell.
 */
import { useCallback, useEffect, useState } from 'react'
import { Flag } from 'lucide-react';
import { apiFetch, postJson } from '../../api/fetchClient'
import PassDialog from './PassDialog.jsx'
import KleinModelSetting from '../shared/KleinModelSetting'
import KleinCompareDialog from '../shared/KleinCompareDialog'
import { useCapabilities } from '../../context/CapabilitiesContext'
import { useToast } from '../common/Toast'
import {
  CLEAN_ENGINES_BLURB, KLEIN_CLEAN_SHORT, KLEIN_CLEAN_TITLE,
} from '../../utils/watermarkCleanEngine.js'
import {
  cropLevelState, findLevelState, hasCleanedImages, inpaintLevelState,
  levelCounts, maskNote, progressSummary, rescanNote, sourceNote,
  textLevelState,
} from './bankWatermark.js'
import { openerLabel } from './scoringPython.js'
import { localEngineUnavailableReason } from '../../utils/localEngineReason'
import { activeLocalLlm } from '../../utils/localLlm'

// How many cleaned images the before/after strip offers. A sample is enough to
// judge a pass; the grid holds the full set.
const COMPARE_SAMPLE = 8

function LevelCard({ index, title, blurb, state, onRun }) {
  return (
    <div className="flex-1 min-w-[15rem] rounded-lg border border-border bg-app/40 p-2.5 space-y-1.5">
      <div className="flex items-baseline gap-1.5">
        <span className="text-[0.625rem] font-bold uppercase tracking-wide text-content-subtle">
          Level {index}
        </span>
        <span className="text-sm font-semibold text-content">{title}</span>
      </div>
      <p className="text-[0.6875rem] leading-snug text-content-subtle">{blurb}</p>
      <button type="button" onClick={onRun} disabled={state.disabled} title={state.reason || title}
        className="rounded-lg border border-amber-400/40 bg-amber-500/15 px-3 py-1.5 text-sm font-semibold text-amber-200 disabled:opacity-40">
        {state.label}
      </button>
      <p className="text-[0.6875rem] text-content-subtle">
        {state.done > 0 ? `${state.done} already handled here. ` : ''}
        {state.reason || state.note || `${state.remaining} image(s) waiting.`}
      </p>
    </div>
  )
}

export default function BankWatermarkPanel({
  bankId, live, onFind, onFindText = null, onChanged, payload = null,
  selectedIds = [], gpuPresent = true, onPickPython = null,
}) {
  const { caps } = useCapabilities()
  const toast = useToast()
  const [levels, setLevels] = useState(null)
  const [method, setMethod] = useState('auto')
  /* What 🧽 Repaint aims at: every flagged page, only the 🔤 text-flagged ones,
     or only the 🚩 watermark-flagged ones. The split is BY PAGE (a page carrying
     both counts as text — its zones share one channel, so one page is never
     split between two runs). Lives on the panel like the engine toggle, so the
     launch window names it without owning a second copy. */
  const [target, setTarget] = useState('all')
  // ⚖ Compare-before-the-batch. `kleinRunModel` is a PER-RUN override, armed by
  // the dialog's "use for this run" and spent by the next Klein inpaint launch —
  // a bank deliberately STORES no Klein pick (the dataset is the one authority),
  // so this lives and dies with the panel.
  const [kleinCompareOpen, setKleinCompareOpen] = useState(false)
  const [kleinCompare, setKleinCompare] = useState({ choices: [], stored: null })
  const [kleinRunModel, setKleinRunModel] = useState(null)
  const [comparing, setComparing] = useState(false)
  const [showOriginal, setShowOriginal] = useState(false)
  /* Which cleaning level's launch window is open ('watermark_crop' /
     'watermark_inpaint'), and the scope each one is aimed at. The scope lives
     HERE rather than inside the dialog so closing a window does not silently
     undo a choice — the rule the other pass windows already follow. */
  const [cleanOpen, setCleanOpen] = useState(null)
  const [cleanScope, setCleanScope] = useState({})

  const load = useCallback(async () => {
    try {
      setLevels(await apiFetch(`/api/bank/${bankId}/watermark/levels`))
    } catch {
      setLevels(null)                              // the panel simply says "not scanned"
    }
  }, [bankId])

  // Reload when the bank changes and every time a job ends (`live` flips back),
  // so the per-level counts follow the passes without their own poller.
  useEffect(() => { load() }, [load, live])

  const run = async (url, body, okMsg) => {
    try {
      await postJson(url, body || {})
      if (okMsg) toast.success(okMsg)
      await load()
      await onChanged?.()
    } catch (e) {
      toast.error(e?.message || 'Action failed.')
    }
  }

  /* A launch coming from a window. It answers {ok,error} instead of toasting:
     the window owns its own refusal surface and stays open with the choices
     intact, so "nothing to repaint in this scope" lands next to the scope that
     produced it (utils/submitOutcome.js).

     Every key is spread-if-set, so a run that changes nothing posts the SAME
     body the button posted before this window existed — the contract the other
     pass dialogs already follow. */
  const runLevel = async (endpoint, { statuses, imageIds }, extra = {}) => {
    const body = {
      ...extra,
      ...(statuses ? { statuses } : {}),
      ...(imageIds === 'selection' && selectedIds.length
        ? { image_ids: [...selectedIds] } : {}),
    }
    try {
      await postJson(`/api/bank/${bankId}/${endpoint}`, body)
    } catch (e) {
      return { ok: false, error: e?.message || 'Could not start the run.' }
    }
    await load()
    await onChanged?.()
    return { ok: true }
  }

  // The ONE shared sentence for "why not Klein", the same one the generation
  // picker and the ✦ Edit modal show — instead of a per-screen catch-all that
  // blames ComfyUI and the weights whatever the real gap is.
  const kleinReason = localEngineUnavailableReason('klein', caps)
  const find = findLevelState(levels, {
    live,
    visionReady: !!activeLocalLlm(caps).vision_model_ready,
    detectorReady: !!caps.watermark_detect,
  })
  const findText = textLevelState(levels, { live, ocrReady: !!caps.video_text })
  const source = sourceNote(levels)
  /* The bin is invisible to /watermark/levels (its pool has always excluded
     rejected images), so the two levels read their bin figure from the bank
     payload's per-pass table — the one the server computes from the SAME clause
     the run filters on. Without it, a bank whose flagged images all sit in the
     bin would show two dead buttons and no way to reach the scope that works. */
  const cropBin = payload?.pass_scopes?.watermark_crop?.todo?.reject || 0
  const inpaintBin = payload?.pass_scopes?.watermark_inpaint?.todo?.reject || 0
  const crop = cropLevelState(levels, { live, binWaiting: cropBin })
  const inpaint = inpaintLevelState(levels, {
    live,
    method,
    lamaReady: !!caps.watermark_inpaint,
    kleinReady: !!caps.watermark_klein,
    kleinReason,
    binWaiting: inpaintBin,
  })
  const note = rescanNote(levels)
  // What the hand-edited masks change for the two levels below (and, loudest, an
  // emptied mask that neither level will act on).
  const masks = maskNote(levels)
  const cleaned = hasCleanedImages(levels)
  // A handful of already-cleaned ids, served by the levels payload — enough to
  // judge a pass, and no extra endpoint just to list them.
  const sample = (levels?.cleaned_sample || []).slice(0, COMPARE_SAMPLE)

  // COLLAPSED BY DEFAULT: watermarks are an occasional errand, not the reason
  // this page is open — an always-expanded three-card panel pushed the grid down
  // on every visit, and off a phone screen entirely. The closed header still
  // carries the state (how many flagged, or that nothing is scanned), so folding
  // it hides the controls, never the situation. Anything needing attention
  // (rows a rescan must adopt) opens it on its own.
  const [open, setOpen] = useState(false)
  useEffect(() => { if (note) setOpen(true) }, [note])
  const c = levelCounts(levels)
  const headline = c.scanned === 0
    ? 'not scanned'
    : c.flagged > 0 ? `${c.flagged} flagged` : 'nothing flagged'

  return (
    <div id="bank-watermark-cleaning" data-workspace-focus
      className="rounded-lg border border-border bg-surface-raised">
      <button type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open}
        className="flex w-full flex-wrap items-center gap-x-2 gap-y-1 p-3 text-left">
        <span className="inline-flex items-center gap-1.5 text-sm font-semibold text-content"><Flag aria-hidden="true" className="h-4 w-4" /> Watermarks</span>
        <span className="text-[0.6875rem] text-content-subtle">{headline}</span>
        {note && <span aria-hidden className="text-[0.6875rem] text-amber-300/90">⚠️</span>}
        <span aria-hidden className="ml-auto text-xs text-content-subtle">{open ? '▲' : '▼'}</span>
      </button>
      {open && (
      <div className="space-y-2 px-3 pb-3">
      <p className="text-[0.6875rem] text-content-subtle">
        find them, then clear them in two manual steps — your original files are never modified
      </p>
      <p className="text-xs text-content-subtle">{progressSummary(levels)}</p>
      {/* WHO ruled, and who will rule next. Wraps onto its own line at 400 px
          rather than being truncated — the source is the actionable half of a
          flag the user disagrees with. */}
      {source && <p className="text-[0.6875rem] text-content-subtle">🔎 {source}</p>}
      {note && <p className="text-xs text-amber-300/90">⚠️ {note}</p>}
      {masks && <p className="text-xs text-content-subtle">{masks}</p>}

      <div className="flex flex-wrap gap-2">
        <LevelCard index={1} title="Find them" state={find}
          blurb="Scans the images you choose for an overlaid logo/URL and records WHERE it sits — the two steps below route on that box."
          onRun={onFind} />
        {/* The OTHER level-1 detection: same rung of the same ladder, different
            instrument. Its zones land in the mask channel, so 🧽 below repaints
            them and ✂ skips them (cropping a bubble out of a page is not a
            thing). Only rendered where the parent wires it. */}
        {onFindText && (
          <LevelCard index={1} title="Find text" state={findText}
            blurb="Reads burned-in text — speech bubbles, subtitles, captions, sound effects — and marks each zone for 🧽 Repaint below. CPU only, never the GPU."
            onRun={onFindText} />
        )}
        <LevelCard index={2} title="Crop it off" state={crop}
          blurb="Cuts the border strip holding the mark. No model, no GPU, and no invented pixel — try this one first."
          onRun={() => setCleanOpen('watermark_crop')} />
        <LevelCard index={3} title="Repaint what's left" state={inpaint}
          blurb={`Repaints the marks a crop can't remove. ${CLEAN_ENGINES_BLURB}`}
          onRun={() => setCleanOpen('watermark_inpaint')} />
      </div>

      {/* The detector's device, SAID — and fixable in place. The Setup install
          pins the app-managed CPU-torch venv, so a machine with a card scans
          tens of thousands of images on the CPU unless the user points the
          detector at a CUDA Python it already has. That gap used to be
          completely silent; this line is the difference between "the scan is
          slow" and a one-click fix. Only shown when the fast detector is
          installed AND its interpreter cannot reach CUDA on a machine that has
          a card — a CPU-only machine gets no pitch it cannot act on. */}
      {caps.watermark_detect && !caps.watermark_detect_gpu && gpuPresent && (
        <div className="space-y-1">
          <p className="text-xs text-amber-400/90">
            Find runs on the CPU — the detector’s Python has no CUDA torch.
            This machine has a GPU: point the detector at a Python that already
            has one (ComfyUI’s, ai-toolkit’s) and the scan runs ~10× faster.
          </p>
          {onPickPython && (
            <button type="button" onClick={() => onPickPython('watermark_detect')}
              className="rounded-md border border-amber-400/50 px-2 py-1 text-xs font-medium text-amber-300 hover:bg-amber-500/10">
              {openerLabel(gpuPresent)}
            </button>
          )}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[0.6875rem] font-semibold uppercase tracking-wide text-content-subtle">
          Level 3 engine
        </span>
        <div className="flex items-center gap-1 rounded-lg border border-border bg-app/60 p-0.5 text-xs">
          <button type="button" aria-pressed={method !== 'klein'} onClick={() => setMethod('auto')}
            title="LaMa: fast, non-generative repaint of small off-centre marks. Marks on the subject stay flagged."
            className={`rounded-md px-2.5 py-1 font-semibold ${method !== 'klein'
              ? 'bg-amber-500/25 text-amber-100' : 'text-content-subtle hover:text-content'}`}>
            LaMa <span className="font-normal opacity-70">fast</span>
          </button>
          <button type="button" aria-pressed={method === 'klein'} onClick={() => setMethod('klein')}
            disabled={!caps.watermark_klein}
            title={caps.watermark_klein
              ? KLEIN_CLEAN_TITLE
              : (kleinReason || 'Klein inpainting needs ComfyUI running + the Klein models (Setup ▸ ComfyUI).')}
            className={`rounded-md px-2.5 py-1 font-semibold disabled:opacity-40 ${method === 'klein'
              ? 'bg-amber-500/25 text-amber-100' : 'text-content-subtle hover:text-content'}`}>
            Klein <span className="font-normal opacity-70">quality</span>
          </button>
        </div>
        {/* WHICH Klein model is about to repaint these images. A bank has no
            dataset to inherit a choice from, so there is nothing to pick here —
            but "no choice" was never a reason to stay silent about the model.
            w-full: this drops onto its own line rather than squeezing the
            toggle at 400 px. */}
        {method === 'klein' && (
          <div className="w-full flex flex-wrap items-center gap-x-3 gap-y-1">
            <KleinModelSetting className="min-w-0 flex-1" />
            <button type="button"
              onClick={async () => {
                const d = await apiFetch('/api/klein-model').catch(() => null)
                setKleinCompare({ choices: d?.choices || [], stored: d?.stored || null })
                setKleinCompareOpen(true)
              }}
              title="Run each ticked Klein model on one flagged image (same zones, same seed) and pick the winner for THIS run — a bank stores no Klein choice."
              className="min-h-10 lg:min-h-0 px-2.5 py-1 rounded-lg border border-border text-xs font-semibold text-content-subtle hover:text-content hover:bg-surface-raised">
              ⚖ Compare models…
            </button>
            {kleinRunModel && (
              <span className="text-[0.6875rem] text-amber-200">
                Next Klein clean runs on <span className="font-mono break-all">{kleinRunModel}</span>{' '}
                <button type="button" onClick={() => setKleinRunModel(null)}
                  className="underline text-content-muted hover:text-content">use auto</button>
              </span>
            )}
          </div>
        )}
        {/* 🔤/🚩 WHAT to clean — only offered once Find text flagged something,
            because with no text-flagged page the three choices collapse into
            one and the control would be a dead dial. */}
        {c.textFound > 0 && (
          <>
            <span className="text-[0.6875rem] font-semibold uppercase tracking-wide text-content-subtle">
              What to clean
            </span>
            <div role="group" aria-label="What to clean"
              className="flex items-center gap-1 rounded-lg border border-border bg-app/60 p-0.5 text-xs">
              <button type="button" aria-pressed={target === 'all'} onClick={() => setTarget('all')}
                title="Repaint every flagged page — text and watermarks alike."
                className={`rounded-md px-2.5 py-1 font-semibold ${target === 'all'
                  ? 'bg-amber-500/25 text-amber-100' : 'text-content-subtle hover:text-content'}`}>
                Both
              </button>
              <button type="button" aria-pressed={target === 'text'} onClick={() => setTarget('text')}
                title="Only pages 🔤 Find text flagged. A page carrying both a watermark and text counts here — one page is never split between two runs."
                className={`rounded-md px-2.5 py-1 font-semibold ${target === 'text'
                  ? 'bg-amber-500/25 text-amber-100' : 'text-content-subtle hover:text-content'}`}>
                🔤 Text
              </button>
              <button type="button" aria-pressed={target === 'watermark'} onClick={() => setTarget('watermark')}
                title="Only pages 🚩 flagged with no text flag on them."
                className={`rounded-md px-2.5 py-1 font-semibold ${target === 'watermark'
                  ? 'bg-amber-500/25 text-amber-100' : 'text-content-subtle hover:text-content'}`}>
                🚩 Marks
              </button>
            </div>
          </>
        )}
        {cleaned && (
          <>
            <button type="button" disabled={live}
              onClick={() => run(`/api/bank/${bankId}/watermark/undo`, {},
                'Cleaned versions removed — your originals are back.')}
              title="Throw away every cleaned version and flag those images again. Your original files were never modified, so nothing is lost."
              className="rounded-lg border border-border bg-surface px-3 py-1.5 text-xs text-content disabled:opacity-40">
              ↩ Undo cleaning
            </button>
            <button type="button" onClick={() => setComparing((v) => !v)}
              aria-expanded={comparing}
              title="Show a sample of the cleaned images so you can flip between the cleaned version and your original."
              className="rounded-lg border border-border bg-surface px-3 py-1.5 text-xs text-content">
              {comparing ? '✕ Hide before/after' : '👁 Before / after'}
            </button>
          </>
        )}
      </div>

      {comparing && sample.length > 0 && (
        <div className="space-y-1">
          <label className="flex items-center gap-1.5 text-xs text-content-subtle">
            <input type="checkbox" checked={showOriginal}
              onChange={(e) => setShowOriginal(e.target.checked)} />
            Show my original files instead of the cleaned versions
          </label>
          <ul className="flex gap-2 overflow-x-auto pb-1">
            {sample.map((id) => (
              <li key={id} className="shrink-0">
                <img alt={showOriginal ? `Original of image ${id}` : `Cleaned image ${id}`}
                  src={`/api/bank/${bankId}/file/${id}${showOriginal ? '?original=1' : ''}`}
                  className="h-28 w-auto rounded-lg border border-border object-contain" />
              </li>
            ))}
          </ul>
        </div>
      )}
      </div>
      )}

      {/* The two file-writing levels' launch windows. Same component, same three
          blocks, same measured scope lines as the passes above — the point being
          that these two are not a different kind of action just because they live
          inside the funnel. */}
      {kleinCompareOpen && (
        <KleinCompareDialog
          choices={kleinCompare.choices}
          stored={kleinCompare.stored}
          compareUrl={`/api/bank/${bankId}/watermark/klein-compare`}
          adoptLabel="Use for this run"
          onAdopt={(model) => { setKleinRunModel(model); setKleinCompareOpen(false) }}
          onClose={() => setKleinCompareOpen(false)}
        />
      )}
      {cleanOpen && (
        <PassDialog passId={cleanOpen} payload={payload} live={live}
          selectionSize={selectedIds.length}
          scope={cleanScope[cleanOpen] || ''}
          onScope={(v) => setCleanScope((s) => ({ ...s, [cleanOpen]: v }))}
          onClose={() => setCleanOpen(null)}
          onLaunch={(r) => runLevel(
            cleanOpen === 'watermark_crop' ? 'watermark/crop' : 'watermark/inpaint',
            r,
            /* target only when narrowed: 'all' posts the SAME body as before the
               selector existed — the spread-if-set contract runLevel documents. */
            cleanOpen === 'watermark_inpaint'
              ? { method,
                  ...(target !== 'all' ? { target } : {}),
                  /* spread-if-set: absent, the body is byte-identical to before
                     the ⚖ dialog existed, and the run keeps auto resolution. */
                  ...(method === 'klein' && kleinRunModel
                    ? { klein_model: kleinRunModel } : {}) } : {},
          )}>
          {cleanOpen === 'watermark_inpaint' && (
            /* WHICH engine this run will use. The toggle stays on the panel —
               it also decides whether the button is live at all, and two
               authorities for one value is how they drift — but a window that
               named no engine would hide the single biggest difference between
               two runs of this level. */
            <p className="m-0 rounded-md border border-border bg-surface-raised px-2 py-1.5 text-[11px] leading-snug text-content-muted">
              Engine: <span className="font-semibold text-content">
                {method === 'klein' ? 'Klein' : 'LaMa'}
              </span>{method === 'klein'
                ? KLEIN_CLEAN_SHORT
                : ' — fast; a mark on the subject stays flagged instead of being smeared.'}
              {target !== 'all' && (
                <>
                  {' '}What to clean: <span className="font-semibold text-content">
                    {target === 'text' ? '🔤 text-flagged pages only' : '🚩 watermark-flagged pages only'}
                  </span> — a page carrying both counts as text-flagged and is repainted whole.
                </>
              )}
              {' '}Change these on the Watermarks panel, next to this level.
            </p>
          )}
        </PassDialog>
      )}
    </div>
  )
}
