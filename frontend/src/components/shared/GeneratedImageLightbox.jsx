import { useCallback, useEffect, useRef, useState } from 'react';
import { Camera } from 'lucide-react';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import { useImageZoomPan } from '../../hooks/useImageZoomPan';
import RepairDialog from './RepairDialog';
import ImproveModal from './ImproveModal';
import CameraAnglePicker from './CameraAnglePicker';
import { useCameraAngles } from '../../hooks/useCameraAngles';
import { useImageDownload } from '../../hooks/useImageDownload';
import { useCapabilities } from '../../context/CapabilitiesContext';
import { postJson } from '../../api/fetchClient';
import { cameraRefusal } from '../../utils/cameraAngles';
import { canImproveCanvasImage } from '../../utils/canvasImprove';
import { lightboxImproveButtons } from '../../utils/improveEngines';
import { canRestoreImproveSettings } from '../../utils/improveSettingsRestore';
import {
  imageHeadlineFacts, imagePromptBlocks, imageSettingFacts, promptFold,
} from '../../utils/generatedImageFacts';
import {
  FACTS_PANEL_CLASS, IMAGE_CLASS, IMAGE_CLASS_BARE, IMAGE_PANE_CLASS,
  IMAGE_PANE_CLASS_BARE, SHELL_CLASS,
} from './generatedImageLightboxLayout';

/* 🔍 ONE generated image, large, with what it was made from.

   It replaces two things: the zoom that lived inline in CheckpointGalleryPanel,
   and PreviewLightbox (which now renders this with the little it knows). The
   app already carries DatasetLightbox and BankReviewLightbox and neither could
   be reused here — DatasetLightbox is about a DATASET image (crop, mirror,
   Klein improve, Pexels attribution, a `filename` on a dataset route) and
   BankReviewLightbox is a triage queue with keep/reject/skip and its own
   session. Both are the wrong nouns and the wrong verbs. So this is not a
   fourth viewer: it is the third, and it absorbs the inline one that made four.

   ⚠️ What was wrong with the old one, because it is the whole reason this file
   exists. Under the image sat a single paragraph:

     step 2500 · seed 208607443 · strength 0 · <forty lines of prompt>

   — the three facts you are actually looking for buried at the head of a wall of
   text, set across the full width of a 2 000-px screen. Three fixes, all
   structural:

     1. HIERARCHY. The facts are not the same kind of thing as the prompt. Step,
        seed and strength are chips; the settings that decided the picture are a
        table; the prose is last.
     2. BOUNDED READING WIDTH. Above `md` the metadata is a column beside the
        image, not a line under it, so it can never run wider than a paragraph —
        and the image takes the whole height it leaves. See
        generatedImageLightboxLayout.js for why that split starts at `md` and
        not, as it first did, at `lg`.
     3. THE PROMPT FOLDS. Long by nature, it opens collapsed past a threshold
        and scrolls rather than pushing everything else off the screen.

   And the seed and the prompt COPY in one click, because that is what those two
   values are for — you re-play a seed, you re-use a prompt.

   Everything the panel decides (which facts, in which order, what folds) is in
   utils/generatedImageFacts.js, where `node --test` can reach it. */

/** A copy button that says it worked. Clipboard access fails on a plain-http
 *  remote origin; the value stays selectable, so the failure is a button that
 *  does nothing visible rather than an error the user cannot act on. */
function CopyButton({ value, label, className = '' }) {
  const [done, setDone] = useState(false);
  const timer = useRef(null);
  useEffect(() => () => clearTimeout(timer.current), []);
  const copy = useCallback(async (e) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(String(value));
      setDone(true);
      clearTimeout(timer.current);
      timer.current = setTimeout(() => setDone(false), 1400);
    } catch { /* clipboard blocked — the text is selectable */ }
  }, [value]);
  return (
    <button type="button" onClick={copy} title={`Copy ${label}`}
      aria-label={done ? `${label} copied` : `Copy ${label}`}
      className={'shrink-0 rounded border border-white/25 px-1.5 py-0.5 text-[0.625rem] '
        + 'text-white/70 hover:border-white/50 hover:text-white ' + className}>
      {done ? '✓ Copied' : '⧉ Copy'}
    </button>
  );
}

/** One prose block — prompt or negative — bounded, foldable, copyable. */
function PromptBlock({ block }) {
  const [expanded, setExpanded] = useState(false);
  const fold = promptFold(block.text, expanded);
  return (
    <section className="mt-3 border-t border-white/10 pt-2">
      <div className="mb-1 flex items-center gap-2">
        <h4 className="m-0 text-[0.6875rem] font-semibold text-white/80">{block.label}</h4>
        <CopyButton value={block.text} label={block.label.toLowerCase()} className="ml-auto" />
      </div>
      <p className={'m-0 whitespace-pre-wrap break-words text-[0.75rem] leading-relaxed text-white/70 '
        + (fold.collapsed
          // A fixed clamp, not a scroll: collapsed means "you can see there is
          // more", and a scrollbar inside a collapsed block invites scrolling a
          // thing that is meant to be opened.
          ? 'line-clamp-4 '
          // Expanded, it scrolls INSIDE its own box. A forty-line prompt must
          // not push the settings — or the image — off the screen.
          : 'max-h-[9rem] overflow-y-auto ')}>
        {block.text}
      </p>
      {fold.foldable && (
        <button type="button" onClick={() => setExpanded((v) => !v)}
          aria-expanded={expanded}
          className="mt-1 rounded text-[0.6875rem] text-indigo-300 underline decoration-dotted hover:text-indigo-200">
          {fold.label}
        </button>
      )}
    </section>
  );
}

/** ✨ The improve group: one button per engine this install can run, plus Klein's
 *  note. Deliberately its OWN component, mounted only when the host passed
 *  `onImprove` — `useCapabilities()` throws outside its provider, and two of this
 *  lightbox's three hosts must keep working with nothing added.
 *
 *  Everything about the two engines is reused, not restated: the labels, the
 *  per-engine disabled reasons, the trade-off sentences and the rule that Klein's
 *  amber note follows KLEIN alone all come from utils/improveEngines.js, shared
 *  with the dataset lightbox and the bulk toolbar. A second copy of those strings
 *  here is exactly how two surfaces drift into telling different stories about
 *  the same pass.
 */
function ImproveActions({ img, onImprove, improvePending, improveReady, busy,
  onOpenImproveModal }) {
  const { caps } = useCapabilities();
  const [improving, setImproving] = useState(false);
  const active = improving || improvePending;
  const buttons = lightboxImproveButtons({
    caps, engines: caps?.engines, improving, improvePending, improveReady, busy,
  });
  const run = (engineId, disabled) => async (event) => {
    event.stopPropagation();
    if (disabled) return;
    // ✨ Klein's settings and result live in the MODAL now (user-asked):
    // its button opens the dialog instead of firing blind. The dial-less
    // engines keep firing straight — nothing to check first.
    if (engineId === 'klein' && onOpenImproveModal) {
      onOpenImproveModal();
      return;
    }
    setImproving(true);
    try {
      await onImprove(img.id, engineId);
    } finally {
      setImproving(false);
    }
  };
  return (
    <>
      {buttons.map((btn) => (
        <button key={btn.id} type="button" data-testid={`lightbox-improve-${btn.id}`}
          onClick={run(btn.id, btn.disabled)} disabled={btn.disabled}
          aria-busy={active} title={btn.title}
          /* Full width under `sm`: this column is 27rem at its widest, so two
             engine buttons beside a Download would each be a 5rem stub on a
             400 px phone. Same class the dataset lightbox uses. */
          className="min-h-9 w-full rounded-lg border border-indigo-400/50 bg-indigo-500/20 px-3 py-1.5 text-[0.75rem] font-semibold text-indigo-100 hover:bg-indigo-500/30 disabled:cursor-not-allowed disabled:opacity-45 sm:w-auto">
          {btn.label}
        </button>
      ))}
    </>
  );
}

/**
 * `img` is a gallery image row (see services.cloud_training._gallery_image).
 * `alt` names it. `actions` is an optional node rendered in the metadata
 * column's footer — the canvas puts its 📌 Pin button there.
 *
 * `facts={false}` drops the metadata column entirely, for a picture that is
 * NOT a generated render: the canvas shows a dataset's 🪪 reference face here,
 * and a reference has no seed, no sampler and no prompt. Rendered anyway, the
 * column announced "SEED —" and offered a Download whose file name is built
 * from a run and a step this picture does not have. An empty table is not a
 * neutral default; it is a wrong answer.
 *
 * `onImprove(imageId, engineId)` turns on the ✨ Upscale & improve group. It is an
 * EXPLICIT opt-in, never inferred from the row's shape, because this component
 * serves three surfaces and only one of them has somewhere for the result to go:
 * the ◉ Canvas passes it, while the checkpoint gallery and the pill preview do
 * not and therefore render exactly what they always did. Inferring it from
 * `img.id` would have quietly lit the button in a preview whose "id" is a step.
 */
export default function GeneratedImageLightbox({ img, alt, actions = null,
  facts = true, onClose, onImprove = null, improvePending = false,
  improveReady = false, busy = false, subjectType = '', datasetId = null,
  /* ✦ Repair and 📷 Camera angles are the VIEWER's own verbs now, not the
     hosts': every host shows the same library row, the routes address that
     row's id, and the one time the verbs were host-wired the Canvas had ✦
     but no 📷 while the Gallery had 📷 but no ✦ — the exact "forgot one
     surface" hole this component exists to close. `onRepair`/`onRepairUndo`
     remain as OVERRIDES for a host that must route differently; passing
     nothing gets the standard wiring on any picture with a library row.
     `onRowChanged` is the one thing a host still knows better than the
     viewer: how to refresh ITS list after a repair rewrote the file. */
  onRowChanged = null,
  onRepair = null, onRepairUndo = null,
  /* ‹ › Walk the host's list without closing the viewer — the 🖼 Gallery's
     whole browsing loop. Same contract as every optional action here: absent =
     not drawn, so the three hosts that show ONE picture render exactly what
     they always did. The host passes null AT the ends rather than a disabled
     flag, for the same reason — a chevron that cannot go anywhere is not
     drawn, never greyed. */
  onPrev = null, onNext = null,
  /* ↩ Make future improves run like THIS ✨ result did (hooks/
     useRestoreImproveSettings). Only meaningful on a Klein improve row —
     utils/improveSettingsRestore gates it — and only where the host has a
     toast to answer with, hence the usual explicit opt-in. */
  onUseImproveSettings = null }) {
  const dialogRef = useRef(null);
  const closeRef = useRef(null);
  /* ONE reading of "are the facts on screen", used by the panel, the pane and
     the picture. Two of them disagreeing draws a bare pane around a framed
     picture, or a frame around nothing. Declared with the hooks, above the
     early return: an effect depends on it, and a hook may not sit behind a
     conditional return. */
  const [repairOpen, setRepairOpen] = useState(false);
  // 📷 The picker is a layer of this viewer now, exactly like ✦ — see the
  // props note: verbs belong to the viewer, hosts supply context.
  const [cameraOpen, setCameraOpen] = useState(false);
  // ✨ The improve modal — settings on demand, result in place. Held HERE like
  // every other layer, so the keydown effect can stand down under it.
  const [improveOpen, setImproveOpen] = useState(false);
  const shootCameraViews = useCameraAngles();
  /* 🔍 Are the facts on screen? They are what this viewer is FOR, so they open
     with it — but they are not what you want while you are looking. Measured at
     412x780, the panel open, the picture is 35 % of the screen; put away, it is
     the screen. The state lives here rather than in localStorage on purpose:
     hiding the details is a decision about the render in front of you, not a
     preference, and it survives flipping to the next image (this component
     stays mounted across `img` changes) which is the span that matters. */
  const [factsOpen, setFactsOpen] = useState(true);
  const showFacts = facts && factsOpen;
  const paneRef = useRef(null);
  const imgRef = useRef(null);
  /* 🔍 Pinch, wheel and double-tap, over utils/imageZoomPan's geometry.

     Folding the details away answered "let me see the render" on every shape
     but one: measured, a phone held UPRIGHT went from 35 % of the screen to
     39 % and stopped, because a 4:3 render at 412 px wide already has the whole
     of the scarce axis. Folding cannot give it more; only magnifying can. This
     is that half, and the phone is the device that needs it.

     `onTap` is the fold — the two gestures share a beginning, so the single tap
     waits DOUBLE_TAP_MS to find out which it was. See the hook. */
  const zoom = useImageZoomPan({
    imgRef,
    frameRef: paneRef,
    active: !!img,
    resetKey: img?.url || null,
    onTap: useCallback(() => { if (facts) setFactsOpen((v) => !v); }, [facts]),
  });
  const dl = useImageDownload();
  useFocusTrap(dialogRef, !!img);
  useEffect(() => {
    if (!img) return undefined;
    // Escape peels ONE layer — see RepairDialog: while it is open the key is
    // its own, and this listener would close the lightbox underneath it.
    // …and a magnified picture is a layer of its own: Escape puts the zoom back
    // before it closes the viewer, so the key never throws away the render you
    // were in the middle of inspecting.
    const onKey = (e) => {
      if (repairOpen || improveOpen) return;
      // 📷 The picker is a layer like ✦: while it is open, keys pressed inside
      // its tree never get here (its root stops them), and a key pressed with
      // focus elsewhere must not walk or close the viewer UNDER the dial —
      // Escape peels the picker first, arrows do nothing. The same lesson the
      // dataset lightbox already carries.
      if (cameraOpen) {
        if (e.key === 'Escape') setCameraOpen(false);
        return;
      }
      // ← → walk the list even while magnified: the zoom resets with the new
      // picture anyway (resetKey follows img), so making the user un-zoom
      // first would add a step that buys nothing.
      if (e.key === 'ArrowLeft' && onPrev) { onPrev(); return; }
      if (e.key === 'ArrowRight' && onNext) { onNext(); return; }
      if (e.key !== 'Escape') return;
      if (zoom.zoomed) { zoom.reset(); return; }
      onClose?.();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [img, onClose, repairOpen, cameraOpen, improveOpen, zoom, onPrev, onNext]);
  useEffect(() => { if (img) closeRef.current?.focus(); }, [img]);
  /* Folding the details resizes the frame under a held zoom, and a view that
     was legally at its edge before is a strip of backdrop afterwards. Settle
     re-applies the travel limit without touching the magnification — the zoom
     is the user's, the gap is not. */
  const settle = zoom.settle;
  useEffect(() => { settle(); }, [showFacts, settle]);

  // Every hook above this line, every read of `img` below it: nothing renders
  // for a viewer with no picture, and a hook may not sit behind a return.
  if (!img) return null;
  const head = imageHeadlineFacts(img);
  const settings = imageSettingFacts(img);
  const blocks = imagePromptBlocks(img);
  const label = alt || 'Generated image';
  /* One question, asked once, for every viewer-owned verb: does this picture
     have a library row to address? The preview hosts (a step preview, the
     lane's reference face) pass a bare URL — no id, no verbs, exactly as
     before. `cameraRefusal` then narrows 📷 further on rows the lane refuses
     (a camera view of a camera view), shown disabled WITH its reason. */
  const hasRow = canImproveCanvasImage(img);
  /* ✦ The standard wiring, owned here: the repair routes address the row id,
     which is the same id space on every host. A host override still wins —
     and `done` stays the host's voice either way, because only the host knows
     how to refresh its own list. */
  const repairApi = onRepair || (hasRow ? {
    submit: (imageId, boxes, prompt, mask) =>
      postJson(`/api/studio/image/${imageId}/repair`, { boxes, prompt, mask }),
    done: (result) => onRowChanged?.(result),
  } : null);
  const repairUndo = onRepairUndo
    || (hasRow ? () => postJson(`/api/studio/image/${img.id}/repair/undo`, {}) : null);

  // The ‹ › chevrons pin to the OVERLAY, like ✕ — inside the zoom pane they
  // would feed the pan/tap gesture machinery a press that meant "next". The
  // right one steps clear of the facts column in the split shape, so it always
  // sits over the PICTURE's edge; the widths mirror FACTS_PANEL_CLASS plus a
  // 1rem gutter.
  const NAV_BTN_CLASS = 'absolute top-1/2 z-10 flex h-11 w-11 -translate-y-1/2 '
    + 'items-center justify-center rounded-full bg-white/10 text-2xl leading-none '
    + 'text-white hover:bg-white/20';
  const NAV_NEXT_POS = showFacts
    ? 'right-2 md:landscape:right-[21rem] lg:landscape:right-[25rem] xl:landscape:right-[28rem]'
    : 'right-2';

  return (
    <div ref={dialogRef} role="dialog" aria-modal="true" aria-label={label}
      data-testid="generated-image-lightbox" data-probe-layer
      onClick={onClose}
      // ⚠️ /95, not an arbitrary /92: Tailwind only emits the opacities in its
      // scale, so bg-black/92 compiled to NOTHING and the board behind stayed at
      // full brightness under what was supposed to be a backdrop.
      // The stacked/split shape itself lives in generatedImageLightboxLayout.js.
      className={SHELL_CLASS}>
      {/* Pinned to the OVERLAY, not to either half, so it is the same target at
          every width — over the picture when stacked, over the panel's top
          padding when split. */}
      <button type="button" ref={closeRef}
        onClick={(e) => { e.stopPropagation(); onClose?.(); }}
        title="Close (Esc)" aria-label="Close image"
        className="absolute right-3 top-3 z-10 flex h-9 w-9 items-center justify-center rounded-full bg-white/10 text-lg leading-none text-white hover:bg-white/20">✕</button>

      {/* 📱 Put the details away and give the picture the screen.

          Beside ✕ and pinned to the OVERLAY for the same reason ✕ is: it must
          be the same target at every width, and it must not be inside the
          column it folds. Only drawn when there ARE facts — a 🪪 reference face
          passes `facts={false}` and has nothing to fold. */}
      {facts && (
        <button type="button" data-testid="lightbox-facts-toggle"
          onClick={(e) => { e.stopPropagation(); setFactsOpen((v) => !v); }}
          aria-expanded={factsOpen}
          title={factsOpen
            ? 'Hide the details and give the picture the whole screen — or just tap the picture'
            : 'Show the seed, the settings and the prompt again'}
          aria-label={factsOpen ? 'Hide the image details' : 'Show the image details'}
          className="absolute right-14 top-3 z-10 flex h-9 w-9 items-center justify-center rounded-full bg-white/10 text-base leading-none text-white hover:bg-white/20">
          <span aria-hidden>{factsOpen ? '⤢' : 'ⓘ'}</span>
        </button>
      )}

      {/* The gestures live on the PANE, not on the picture: zoomed in past the
          frame on one axis there are still bars of backdrop on the other, and a
          pan that dies the moment your thumb crosses onto one is a pan that
          fights you. `touch-none` because the browser would otherwise take the
          pinch for itself and zoom the whole page over a viewer that was
          already zooming the picture. */}
      {/* ⤾ The way back out, for the wheel and the pinch — a double tap already
          goes home, but nothing says so, and a magnified picture with no
          visible way back is the state people close the viewer to escape.
          Only while it can do something. */}
      {zoom.zoomed && (
        <button type="button" data-testid="lightbox-zoom-reset"
          onClick={(e) => { e.stopPropagation(); zoom.reset(); }}
          title="Back to the whole picture (double-tap it, or press Esc)"
          aria-label="Reset the zoom"
          className="absolute right-24 top-3 z-10 flex h-9 items-center rounded-full bg-white/10 px-3 text-[0.75rem] font-semibold leading-none text-white hover:bg-white/20">
          <span aria-hidden className="mr-1">⤾</span>{Math.round(zoom.view.scale * 100)}%
        </button>
      )}

      {/* ‹ › — drawn only where there IS somewhere to go (see the prop note). */}
      {onPrev && (
        <button type="button" data-testid="lightbox-prev"
          onClick={(e) => { e.stopPropagation(); onPrev(); }}
          title="Previous image (←)" aria-label="Previous image"
          className={`${NAV_BTN_CLASS} left-2`}>
          <span aria-hidden>‹</span>
        </button>
      )}
      {onNext && (
        <button type="button" data-testid="lightbox-next"
          onClick={(e) => { e.stopPropagation(); onNext(); }}
          title="Next image (→)" aria-label="Next image"
          className={`${NAV_BTN_CLASS} ${NAV_NEXT_POS}`}>
          <span aria-hidden>›</span>
        </button>
      )}

      <div ref={paneRef} {...zoom.handlers}
        className={(showFacts ? IMAGE_PANE_CLASS : IMAGE_PANE_CLASS_BARE) + ' touch-none'}
        /* A press on the backdrop still closes the viewer, which is the way out
           people reach for first — but not while the picture is magnified, or
           letting go after a pan would shut the thing you were inspecting. */
        onClick={(e) => { if (zoom.zoomed) e.stopPropagation(); }}>
        {/* Tapping the picture folds the details away and brings them back —
            the gesture every photo viewer on a phone already has, and the one a
            thumb reaches without aiming. Double-tap magnifies instead, and the
            press never reaches the backdrop, so neither can close the viewer by
            accident. */}
        <img ref={imgRef} src={img.url} alt={label} draggable={false}
          onClick={(e) => e.stopPropagation()}
          style={zoom.style}
          className={showFacts ? IMAGE_CLASS : IMAGE_CLASS_BARE} />
      </div>

      {/* THE fix for the wall of text: a column, not a line. Bounded width above
          `md` and the panel's own width below it — either way the prose has a
          reading width, never the width of the screen. Its own scroll, so the
          image never shrinks to make room for a long prompt. */}
      {showFacts && (
      <aside onClick={(e) => e.stopPropagation()}
        data-testid="generated-image-facts"
        /* OPAQUE, not a tint. At 60 % the page behind it stayed legible through
           the panel — a settings table you read the board through is a table you
           misread. The backdrop can be translucent; the thing you are reading
           cannot. */
        className={FACTS_PANEL_CLASS}>
        <div className="mx-auto max-w-md">
          {/* The three answers to "what am I looking at", as chips. Big, tabular,
              and the seed carries its own copy — a seed is a thing you re-play. */}
          <div className="flex flex-wrap items-center gap-1.5">
            {head.map((f) => (
              <span key={f.key} data-testid={`fact-${f.key}`}
                className="flex items-center gap-1 rounded-md border border-white/15 bg-white/5 px-2 py-1">
                <span className="text-[0.5625rem] uppercase tracking-wide text-white/45">{f.label}</span>
                <span className="text-[0.8125rem] font-semibold tabular-nums text-white">{f.value}</span>
                {f.copy && <CopyButton value={f.copy} label="seed" />}
              </span>
            ))}
          </div>

          {settings.length > 0 && (
            <section className="mt-3 border-t border-white/10 pt-2">
              <h4 className="m-0 mb-1 text-[0.6875rem] font-semibold text-white/80">
                <span aria-hidden>⚙</span> Made with
              </h4>
              {/* A grid, not a sentence: these are looked UP, one at a time,
                  while comparing two renders. Absent settings produce no row —
                  see imageSettingFacts. */}
              <dl className="m-0 grid grid-cols-[minmax(0,auto)_minmax(0,1fr)] gap-x-3 gap-y-0.5">
                {settings.map((r) => (
                  <div key={r.key} className="contents">
                    <dt className="m-0 text-[0.6875rem] text-white/45">{r.label}</dt>
                    <dd className="m-0 break-words text-[0.6875rem] tabular-nums text-white/80">{r.value}</dd>
                  </div>
                ))}
              </dl>
            </section>
          )}

          {blocks.map((b) => <PromptBlock key={b.key} block={b} />)}

          {/* ⬇ Keep it. In the SAME footer as 📌 Pin, because they are the two
              things you do once you have decided a render is good — and the
              file lands under a name that still says which dataset, run, step
              and seed made it (services/gallery_download.py). Without that name
              a saved render is anonymous within the week, which on a screen
              whose whole job is telling checkpoints apart is the whole loss.

              A refusal is shown here rather than swallowed: the gallery does
              list rows whose file a resume or a trash sweep has already taken
              off the disk. */}
          <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-white/10 pt-2.5">
            <button type="button" data-testid="lightbox-download"
              disabled={dl.busy || img.id == null}
              onClick={(e) => { e.stopPropagation(); dl.download(img.id); }}
              title="Download this image — the file name keeps its dataset, run, step and seed"
              className="rounded-md border border-white/25 px-3 py-1.5 text-[0.75rem] font-semibold text-white/85 hover:border-white/50 hover:text-white disabled:opacity-40">
              <span aria-hidden>⬇</span> {dl.busy ? 'Downloading…' : 'Download'}
            </button>
            {/* ✨ Beside ⬇, because they are the two things you do once a render
                is worth keeping: save it, or make it better. Only the host that
                has a route for it passes `onImprove` — see the prop's note. */}
            {onImprove && (
              <ImproveActions img={img} onImprove={onImprove}
                improvePending={improvePending} improveReady={improveReady}
                busy={busy} onOpenImproveModal={() => setImproveOpen(true)} />
            )}
            {/* ↩ On a ✨ result you LIKE: make every next improve run the way
                this one did — the recorded instruction back into the global
                setting, the chained LoRAs mapped back to their preset. The
                gate lives in improveSettingsRestore: a pure-restoration
                result recorded no instruction and offers nothing. */}
            {onUseImproveSettings && canRestoreImproveSettings(img) && (
              <button type="button" data-testid="lightbox-use-improve-settings"
                onClick={(e) => { e.stopPropagation(); onUseImproveSettings(img); }}
                disabled={busy}
                title="Make the next ✨ improves use what THIS image was made with — its instruction and LoRA preset become the app-wide improve settings"
                className="rounded-md border border-emerald-400/50 bg-emerald-500/15 px-3 py-1.5 text-[0.75rem] font-semibold text-emerald-100 hover:bg-emerald-500/25 disabled:opacity-40">
                <span aria-hidden>↩</span> Use these improve settings
              </button>
            )}
            {/* ✦ Repair sits with them because it answers the third thing you do
                with a render: keep it, improve it — or fix the ONE part that is
                wrong. Regenerating for a stray finger throws away the image you
                liked; this repaints only what you draw. (.samexit, Discord.) */}
            {repairApi && img.id != null && (
              <button type="button" data-testid="lightbox-repair"
                onClick={(e) => { e.stopPropagation(); setRepairOpen(true); }}
                disabled={busy}
                title="Repaint one area of this image from your own description — draw the zone, say what should be there, and everything outside it stays byte-identical"
                className="rounded-md border border-sky-400/50 bg-sky-500/20 px-3 py-1.5 text-[0.75rem] font-semibold text-sky-50 hover:bg-sky-500/30 disabled:opacity-40">
                <span aria-hidden>✦</span> Repair
              </button>
            )}
            {/* 📷 In the same footer, on every host that shows a library row.
                Shown DISABLED with its reason rather than hidden when the row
                cannot take it: a button that vanishes teaches nothing, and
                "why can't I?" is the question this panel exists to answer. */}
            {hasRow && (
              <button type="button" data-testid="lightbox-camera-angles"
                onClick={(e) => { e.stopPropagation(); setCameraOpen(true); }}
                disabled={busy || !!cameraRefusal(img)}
                title={cameraRefusal(img) || 'Re-shoot this scene from another camera position'}
                className="min-h-10 lg:min-h-0 inline-flex items-center gap-2 rounded-lg border border-indigo-400/50 bg-indigo-500/20 px-3 py-1.5 text-[0.75rem] font-semibold text-indigo-100 hover:bg-indigo-500/30 disabled:cursor-not-allowed disabled:opacity-45">
                <Camera className="size-3.5" aria-hidden />
                Camera angles
              </button>
            )}
            {actions}
          </div>
          {dl.error && (
            <p role="alert" data-testid="lightbox-download-error"
              className="m-0 mt-1.5 rounded-md border border-amber-400/40 bg-amber-500/10 px-2 py-1 text-[0.6875rem] text-amber-100">
              {dl.error}
            </p>
          )}
        </div>
      </aside>
      )}
      {/* The zone editor, mounted INSIDE the overlay so it inherits its stacking
          context — a sibling would need a fragment and would sit under it. */}
      <RepairDialog open={repairOpen} src={img?.url} alt={alt}
        onClose={(result) => { setRepairOpen(false); if (result && repairApi?.done) repairApi.done(result); }}
        onSubmit={({ boxes, mask, prompt }) => repairApi.submit(img.id, boxes, prompt, mask)}
        onUndo={repairUndo} />
      {/* 📷 The picker, a layer of this viewer like ✦ above it. The reference
          picture stays visible behind the dial — picking an angle of something
          you cannot see is guesswork — and the keydown effect freezes ‹ › and
          Escape-to-close while it is open, so the row under the dial cannot
          change. On success the views land in the Gallery feed; the hook's
          toast says so, which is true from every host. */}
      {/* ✨ The improve modal — a layer like ✦ and 📷. The viewer's rows are
          library rows on every host, so the modal polls the library routes. */}
      {improveOpen && hasRow && (
        <ImproveModal img={img} host="library" datasetId={datasetId}
          subjectType={subjectType} onClose={() => setImproveOpen(false)} />
      )}
      {cameraOpen && hasRow && (
        <CameraAnglePicker
          onClose={() => setCameraOpen(false)}
          onShoot={async (poses) => {
            const ok = await shootCameraViews(img.id, poses);
            if (ok) setCameraOpen(false);
          }} />
      )}
    </div>
  );
}
