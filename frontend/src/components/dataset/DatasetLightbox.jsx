/**
 * Full-screen inspection lightbox (F3): toggle fit ↔ 100 % (native pixels) to
 * hunt skin/eyes artefacts before keeping an image. Esc, ✕ or a click on the
 * backdrop close it; a click on the image toggles the zoom mode.
 *
 * The action bar is NOT always at the bottom: beside a portrait image on a wide
 * window it becomes a side rail in the space that image cannot use, which hands
 * the bar's height back to the photo; below `sm` it becomes ONE button opening a
 * panel, because on a phone neither axis has room and the picture was being left
 * 96 px tall. `lightboxActionPlacement.js` owns that decision and its
 * stability guarantees.
 */
import { Fragment, useCallback, useEffect, useId, useRef, useState } from 'react';
import RepairDialog from '../shared/RepairDialog';
import KleinImproveNote from './KleinImproveNote';
import { lightboxImproveButtons } from '../../utils/improveEngines';
import { useCapabilities } from '../../context/CapabilitiesContext';
import {
  decideActionPlacement, rememberImageRatio, readImageRatio,
} from './lightboxActionPlacement';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import { displayLabel } from '../../utils/labels';
import { describeReferenceComparison } from '../../utils/referenceCompare';
import SourceAttribution from './SourceAttribution';
import {
  freshLightboxImageState, lightboxImageState, lightboxNeighbours, stampedPatch,
} from './lightboxNavigation';
import {
  REVIEW_SHORTCUT_HINT, reviewKeyAction,
} from '../shared/reviewShortcuts';
import ShortcutKey from '../shared/ShortcutKey';
import { watermarkMaskButtonLabel } from '../../utils/watermarkRegions';

const COMPARE_HELP = 'Show the original this image was made from, next to it, at the same scale.';
/* The verdict this image currently carries, in the SAME words and the same
   colours as the Bank's review chips: green means kept in both screens, and a
   dataset image that has never been judged says so rather than looking kept. */
const VERDICT_CHIP = {
  keep: { text: '✓ kept', cls: 'bg-emerald-500/25 text-emerald-200' },
  reject: { text: '✕ rejected', cls: 'bg-rose-500/25 text-rose-200' },
  pending: { text: '· undecided', cls: 'bg-white/10 text-white/60' },
  failed: { text: '⚠ failed', cls: 'bg-amber-500/25 text-amber-200' },
};
/* The SECOND comparison, and deliberately a second BUTTON rather than a picker:
   an improved image can answer both questions and a selector would hide one of
   them behind the other. The two modes are mutually exclusive though — two
   pairs side by side stop showing anything at all — so one state holds which
   reading is on screen. */
const REFERENCE_COMPARE_HELP = 'Show the dataset\'s reference photo next to this image — '
  + 'the framings differ, so each pane fits its own image.';

/**
 * One half of the comparison. The two panes are cells of the SAME grid, so they
 * get identical boxes; `object-contain` then renders both images at the same
 * scale and the same framing whatever their pixel size — the improve pass
 * rescales to a megapixel budget and keeps the aspect ratio, so this is the only
 * reading where "it looks better" means something. Each side is named in text,
 * never by colour alone.
 *
 * The reference comparison reuses this component unchanged, and gets the right
 * behaviour for free rather than by accident: a square head reference and a
 * full-body plan have different aspect ratios, so each fills its own identical
 * box and the two are shown at whatever scale makes them whole. What must NOT
 * be reused is the "same scale" sentence — see the status line below.
 */
function ComparePane({ label, url, alt, accent }) {
  return (
    <figure className="m-0 flex min-h-0 min-w-0 flex-col overflow-hidden rounded-lg border border-white/15">
      <figcaption className={`shrink-0 border-b border-white/10 bg-black/70 px-2 py-1 text-[11px] font-semibold ${
        accent ? 'text-indigo-200' : 'text-white/80'}`}>
        {label}
      </figcaption>
      <div className="min-h-0 flex-1 p-1">
        {/* h-full w-full, NOT max-h/max-w: an <img> left at its intrinsic size
            is capped by max-* but never scaled UP, so a 0.4 MP original
            rendered small next to a 2 MP result that filled its pane — the two
            were shown at different scales, which is precisely the comparison
            this mode must not produce. Filling the box and letting
            object-contain letterbox makes both fit the SAME box. */}
        <img src={url} alt={`${label} — ${alt}`}
          className="h-full w-full select-none object-contain" />
      </div>
    </figure>
  );
}

/**
 * Where the action block is MOUNTED — the third placement's whole job.
 *
 * In the rail and the bottom bar the actions are a child of the dialog and this
 * is a pass-through: same element, same DOM order, no wrapper. In the sheet they
 * become the body of a labelled dialog that slides over the bottom of the
 * picture, and everything else on screen is unchanged — the image keeps its
 * zoom, its comparison and its place in the list, because opening this panel
 * writes ONE boolean and renders nothing else differently.
 *
 * It deliberately does NOT trap focus of its own. The lightbox already traps Tab
 * inside itself; a second trap here would be the one thing this panel must never
 * do — make the picture, its ⟨ / ⟩ and its ✕ unreachable from the keyboard while
 * the panel is up.
 */
function ActionsHost({ sheet, open, panelId, label, closeRef, onDone, children }) {
  if (!sheet) return children;
  if (!open) return null;
  return (
    <div id={panelId} role="dialog" aria-label={label}
      onClick={(e) => e.stopPropagation()}
      /* max-h-[70vh]: the panel is a drawer over the image, never a new screen.
         Seeing the picture you are about to rotate is the point of acting from
         here rather than from a menu. */
      className="absolute inset-x-0 bottom-0 z-30 flex max-h-[70vh] flex-col
        rounded-t-2xl border-t border-white/15 bg-neutral-950 shadow-2xl">
      <div className="flex shrink-0 items-center justify-between gap-3 border-b border-white/10 px-4 py-2">
        <h2 className="min-w-0 truncate text-sm font-semibold text-white">Image actions</h2>
        <button type="button" ref={closeRef} onClick={onDone}
          title="Close the actions panel (Esc)" aria-label="Close the actions panel"
          className="min-h-10 lg:min-h-9 shrink-0 rounded-lg bg-white/10 px-3 py-1.5 text-xs font-semibold text-white hover:bg-white/20">
          Done
        </button>
      </div>
      {/* The panel scrolls, the page does not: overscroll-contain stops a flick
          at the end of the list from dragging the lightbox behind it. */}
      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">{children}</div>
    </div>
  );
}

/**
 * One of the two edge arrows. At an end it goes disabled and its title AND
 * aria-label become the sentence saying which end — the brief being that a dead
 * end must be readable, not a mute no-op. Both channels carry it: a title alone
 * is invisible to a screen reader and unreachable on a touch screen.
 */
function NavArrow({ side, glyph, label, target, reason, onNavigate }) {
  const disabled = !target;
  const text = disabled ? reason : `${label} (${side === 'left' ? '←' : '→'})`;
  return (
    <button type="button" disabled={disabled}
      onClick={(e) => { e.stopPropagation(); if (target) onNavigate(target); }}
      title={text} aria-label={text}
      /* The backdrop is bg-black/95, so a black pill on it is an invisible pill:
         the ring is what makes the target readable over the gutter, and the
         dark fill is what keeps it readable over a bright photo. A DISABLED end
         must still be seen — an arrow faded to nothing is the mute no-op this
         feature exists not to be — so it dims to 50 %, not to a ghost. */
      className={`absolute top-1/2 z-10 flex h-11 w-11 -translate-y-1/2 items-center
        justify-center rounded-full border border-white/25 bg-black/60 text-xl
        leading-none text-white hover:bg-black/80 disabled:cursor-not-allowed
        disabled:opacity-50 disabled:hover:bg-black/60
        ${side === 'left' ? 'left-2' : 'right-2'}`}>
      <span aria-hidden="true">{glyph}</span>
    </button>
  );
}

export default function DatasetLightbox({
  img,
  datasetId,
  nonce = 0,
  compare = null,
  parentNonce = 0,
  refFilename = '',
  refNonce = 0,
  onClose,
  onCrop,
  onMirror,
  onRotate,
  onImprove,
  // Opens the watermark mask editor on THIS image, flagged or not. Optional like
  // the rest: a caller that does not pass it simply shows no button.
  onMarkWatermark,
  /* ✦ Repair, in ONE click. It used to hand the image to onMarkWatermark, which
     closed this lightbox and reopened the WATERMARK REVIEW filtered to it — so
     repainting a necklace meant pressing "✦ Repair", landing on a screen about
     watermarks, and pressing a second button of the same name. Two hops and a
     misleading destination for the gesture people actually asked for.
     Now it opens the dialog here. That also earns 🚩 its separate existence:
     until now both buttons went to the same place, which is the definition of a
     button too many. (imageId, prompt, boxes, mask) -> the repair call. */
  onRepair,
  onUndoRepair,
  busy = false,
  // ✨ Improve is queue work, not a write on this image: `improve_existing_image`
  // only refuses a candidate already in flight on THIS source, plus MAX_FANOUT.
  // So it reads the improve-lane gate the grid's ✨ buttons read, not the
  // conservative `busy` that guards crop/rotate/mirror/keep/reject (GitHub #44).
  // Same action, same answer on both surfaces of one screen. Defaults to `busy`
  // so a caller that passes neither keeps the old blanket.
  improveBusy = undefined,
  // Curating this image — keep/reject, crop, mirror, rotate, watermark, repair.
  // Same gate the grid behind the lightbox uses, because it is the same action:
  // one screen must not answer twice differently. Defaults to `busy`.
  curationBusy = undefined,
  // The sentence a refused write shows (which pass holds this dataset, where it
  // is, what to do). Opening, zooming and comparing never consult it: they read
  // the same bytes the grid is already showing.
  busyReason = null,
  mirrorBusy = false,
  improvePending = false,
  improveReady = false,
  subjectType = '',
  // The list the grid SHOWS, in its order, for ⟨ / ⟩. Omitted (rescue-review
  // preview) = no navigation, rather than arrows onto a list that isn't there.
  images = null,
  onNavigate = null,
  /* ✓ Keep / ✕ Reject from here — the SAME (imageId, status) write the grid
     tile does, never a second notion of "kept". Omitted = the verdict buttons
     and their keys are simply absent (rescue-review preview, where the pair is
     resolved in Curation instead). */
  onStatus = null,
}) {
  const { caps } = useCapabilities();
  /* Zoom, comparison and "improving" belong to ONE image and are stamped with
     its id: a render that finds another image's stamp uses a fresh state
     instead. See lightboxNavigation.js — this is what stops ⟩ from carrying a
     100 % zoom, or a pane labelled "original" showing the PREVIOUS image's
     parent, onto the next picture. */
  const [storedState, setStoredState] = useState(
    () => freshLightboxImageState(img?.id ?? null));
  const dialogRef = useRef(null);
  const closeRef = useRef(null);

  /* WHERE THE ACTIONS LIVE. A portrait photo on a wide monitor leaves two
     thirds of the width black while the buttons queue on one line under it, on
     the one axis the image is short of. The rule (and the geometry behind it)
     lives in lightboxActionPlacement.js because `node --test` cannot parse JSX
     and this is the part that must be tested case by case. */
  const imageId = img?.id ?? null;
  /* `compareMode` ('none' | 'derived' | 'reference') sits INSIDE the stamped
     state, where the boolean `comparing` used to. Two reasons, and the second
     is the one that matters: a mode held in its own useState would have been
     the one piece of per-image state that travels — ⟩ would have carried
     "reference comparison open" onto the next picture, and, worse, an open
     "original" pane onto an image whose parent is somebody else's. Inside the
     slot the guarantee is structural: a foreign stamp yields a fresh state, so
     moving image closes the comparison with no reset effect to get right. */
  const {
    full, compareMode, improving, actionsOpen, repairOpen, deciding,
  } = lightboxImageState(storedState, imageId);
  /* Which image is on screen when a setter actually RUNS — a ref, because the
     `finally` of an improve resolves long after the render that created its
     callback and must be compared against the present, not against the past it
     closed over. */
  const currentIdRef = useRef(imageId);
  useEffect(() => { currentIdRef.current = imageId; }, [imageId]);
  // Stamped with the id of the render that created it, and DROPPED if that is
  // no longer the image being shown: one slot, so a late writer stamping it
  // would reset the picture you moved to. See stampedPatch.
  const patchImageState = useCallback((patch) => {
    setStoredState((prev) => stampedPatch(prev, patch, imageId, currentIdRef.current));
  }, [imageId]);
  const nav = lightboxNeighbours(images, imageId);
  const canNavigate = !!onNavigate && nav.available;
  /* ── Reviewing, not just looking ──────────────────────────────────────────
     ✓ Keep / ✕ Reject / ⏭ Skip, on the same K/R/S the Bank's ▶ Review uses
     (components/shared/reviewShortcuts.js owns the grammar for both). The
     verdict is the dataset's OWN status — pending|keep|reject, the one the grid
     tile writes — so nothing here invents a second notion of "kept" that would
     have to be reconciled with captioning, export and training.

     The move happens only once the write LANDS. Advancing first and posting
     afterwards is how a decision gets silently dropped on a slow disk, and the
     Bank already paid for that lesson: a lost verdict is worse than a slow one.
     `nav.next` is captured BEFORE the call because refreshing can retire this
     very image from the shown list (a "◧ Undecided only" filter does exactly
     that), which would leave the neighbours unanswerable a moment later. */
  const nextImage = nav.next;
  const decide = useCallback(async (status) => {
    if (!onStatus || deciding || busy || imageId == null) return;
    const after = nextImage;
    patchImageState({ deciding: true });
    try {
      await onStatus(imageId, status);
      // At the end of the list there is nowhere to go: the picture stays under
      // the cursor wearing its new chip, rather than closing the overlay on the
      // user — the arrows already say which end this is.
      if (after && onNavigate) onNavigate(after);
    } finally {
      patchImageState({ deciding: false });
    }
  }, [onStatus, deciding, busy, imageId, nextImage, onNavigate, patchImageState]);
  // ⏭ Skip is deliberately nothing but "next": the image keeps whatever status
  // it already had. That is the promise that makes walking fast safe.
  const skipImage = useCallback(() => {
    if (nextImage && onNavigate) onNavigate(nextImage);
  }, [nextImage, onNavigate]);
  // A rail decided mid-rotation would move under the pointer that started it.
  const curationRefused = (curationBusy ?? busy);
  const curationRefusedReason = curationRefused ? busyReason : null;
  // Same one-tile rule as the grid: an upscale of THIS image copied its source
  // at enqueue, so editing the pixels now would send back an upscale of the
  // version from before the edit.
  const upscaleRendering = improvePending && !improveReady;
  const pixelEditRefused = curationRefused || upscaleRendering;
  const pixelEditReason = curationRefusedReason
    || (upscaleRendering
      ? 'An upscale of this image is still rendering — it would come back as an '
        + 'upscale of the version from before your edit. It will be available '
        + 'again once that result arrives.'
      : null);
  const refused = curationRefusedReason;
  const actionsLocked = curationRefused || mirrorBusy || improving || improvePending;
  const [ratio, setRatio] = useState(() => readImageRatio(imageId));
  // Decided for the FIRST painted frame, not corrected by an effect afterwards:
  // with the ratio already known (the grid measured it) there is no frame in
  // which the actions sit somewhere they are about to leave.
  const [placement, setPlacement] = useState(() => decideActionPlacement({
    viewportWidth: typeof window === 'undefined' ? 0 : window.innerWidth,
    viewportHeight: typeof window === 'undefined' ? 0 : window.innerHeight,
    imageWidth: ratio?.imageWidth,
    imageHeight: ratio?.imageHeight,
  }));

  // A new image starts from whatever was measured for it before — reopening one
  // therefore never replays the bottom→rail commit in front of the user.
  useEffect(() => { setRatio(readImageRatio(imageId)); }, [imageId]);

  const onImageLoad = useCallback((event) => {
    const { naturalWidth: w, naturalHeight: h } = event.currentTarget;
    rememberImageRatio(imageId, w, h);
    setRatio((prev) => (prev && prev.imageWidth === w && prev.imageHeight === h
      ? prev
      : { imageWidth: w, imageHeight: h }));
  }, [imageId]);

  useEffect(() => {
    let frame = 0;
    const apply = () => setPlacement((current) => decideActionPlacement({
      viewportWidth: window.innerWidth,
      viewportHeight: window.innerHeight,
      imageWidth: ratio?.imageWidth,
      imageHeight: ratio?.imageHeight,
      current,
      // The placement rule only cares that TWO panes want the width, not which
      // pair is on screen.
      comparing: compareMode !== 'none',
      locked: actionsLocked,
    }));
    apply();
    // One decision per frame: a window drag fires `resize` dozens of times a
    // second, and re-deciding per event is how a bar ends up shimmering.
    const onResize = () => {
      if (frame) return;
      frame = requestAnimationFrame(() => { frame = 0; apply(); });
    };
    window.addEventListener('resize', onResize);
    return () => {
      window.removeEventListener('resize', onResize);
      if (frame) cancelAnimationFrame(frame);
    };
  }, [ratio, compareMode, actionsLocked]);

  const rail = placement === 'rail';
  const sheet = placement === 'sheet';
  /* The panel only exists in the sheet: a stale `actionsOpen` left behind by a
     rotated phone must not paint a bottom sheet over a desktop rail. Derived
     rather than reset in an effect — there is no frame in which it is wrong. */
  const panelOpen = sheet && actionsOpen;
  const panelId = `lightbox-actions${useId()}`;
  const actionsBtnRef = useRef(null);
  const panelCloseRef = useRef(null);
  const closePanel = useCallback(() => {
    patchImageState({ actionsOpen: false });
    // Back to the button that opened it, not to the top of the dialog: the
    // panel is a detour, and losing your place in the tab order on the way out
    // is how a keyboard user ends up re-walking the whole overlay.
    actionsBtnRef.current?.focus();
  }, [patchImageState]);
  // Opening moves focus INTO the panel. The outer trap keeps Tab in the dialog,
  // so the image, its arrows and ✕ all stay reachable from there — the panel
  // deliberately does not trap on its own.
  useEffect(() => { if (panelOpen) panelCloseRef.current?.focus(); }, [panelOpen]);

  // Focus trap keeps Tab inside the dialog (P2-7).
  useFocusTrap(dialogRef, !!(img && img.filename));

  /* Keyboard. The grammar — K keep, R reject, S/→ skip, ← back, Esc close, and
     which keystrokes a focused field owns — is READ from the shared module, not
     re-decided here: it is the Bank's, and a reflex learned there has to be
     right here too. Modifier keys and text entry are handled inside it (⌘R must
     reload the page, not reject the picture). */
  const prev = nav.prev;
  useEffect(() => {
    const onKey = (e) => {
      /* ✦ Repair is a layer above this overlay. Both listen on `window`, so
         without standing down, every letter here is a verdict on the picture
         underneath the dialog — R rejected it while the user was drawing a
         zone. Escape-only was not enough: watermark review already returns on
         every key for the same reason, and the Bank does it for crop/mask. */
      if (repairOpen) return;
      const action = reviewKeyAction(e);
      /* Escape peels ONE layer: an open actions panel first, the lightbox only
         once it is closed. Closing everything at once would throw the user out
         of the image they were about to act on — the panel is a detour, not a
         second window. */
      if (action === 'close') {
        if (panelOpen) { closePanel(); return; }
        onClose();
        return;
      }
      if (!action) return;
      if (action === 'keep' || action === 'reject') {
        if (!onStatus) return;
        e.preventDefault();
        decide(action);
        return;
      }
      // 'back' and 'skip' move without touching anything.
      const target = action === 'back' ? prev : nextImage;
      if (!onNavigate || !target) return;
      e.preventDefault();
      onNavigate(target);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose, onNavigate, onStatus, decide, prev, nextImage, panelOpen, closePanel, repairOpen]);
  useEffect(() => { closeRef.current?.focus(); }, []);
  /* No "close the comparison when the image changes" effect on purpose: the id
     stamp above already guarantees it, for BOTH comparison modes, without a
     frame in which the previous image's panes are painted next to the new one. */

  if (!img || !img.filename) return null;
  const fileUrl = (filename, v) =>
    `/api/dataset/${datasetId}/img/${encodeURIComponent(filename)}${v ? `?v=${v}` : ''}`;
  const url = fileUrl(img.filename, nonce);
  const alt = displayLabel(img.variation_label) || 'dataset image';
  // A comparison is only ever entered when the other side is actually
  // renderable; an unavailable one degrades to a stated reason, never a dead
  // button. Both modes go through this same guard.
  const refCompare = describeReferenceComparison(img, refFilename);
  const usable = (c) => !!(c && c.available && c.parent?.filename);
  const canCompare = usable(compare);
  const canCompareRef = usable(refCompare);
  // The mode currently on screen, re-checked against availability: a payload
  // refresh can retire the parent under an open comparison, and a stale mode
  // must fall back to the plain view rather than render a broken pane.
  const activeCompare = (compareMode === 'derived' && canCompare) ? compare
    : (compareMode === 'reference' && canCompareRef) ? refCompare
      : null;
  const inCompare = !!activeCompare;
  // The reference lives beside the images, not among them: its own cache nonce.
  const activeParentNonce = compareMode === 'reference' ? refNonce : parentNonce;
  // Pressing the mode you are in leaves it; pressing the other one SWITCHES,
  // which is what makes the two exclusive without a single line of teardown.
  // Written through the stamped patch like every other per-image property, so a
  // press that resolves after ⟩ cannot reopen a pane on the next image.
  const toggleCompare = (mode) => (event) => {
    event.stopPropagation();
    // …and the narrow-screen panel closes with it. Entering a comparison is a
    // request to LOOK at something; leaving the drawer over the two panes you
    // just asked for would be the bug this placement was written to remove.
    // The edits (rotate, mirror, improve) deliberately do NOT close it: those
    // are chained — two quarter turns, then a mirror — and the picture stays
    // visible above a panel that only claims 70 % of the height.
    patchImageState({
      full: false,
      compareMode: compareMode === mode ? 'none' : mode,
      actionsOpen: false,
    });
  };
  const improvementActive = improving || improvePending;
  /* ONE button per engine that can run here, exactly like the selection
     toolbar. The lightbox is where you are when you are looking at the one
     image you want to fix, and until now it only offered Klein — so on a DRAWN
     dataset the amber note warned that Klein pulls the skin towards realism
     while the pass that does not, SeedVR2, was two screens away (reported by
     a user with a screenshot of exactly that). The wording, the gating and the
     per-engine disabled reasons all come from the shared pure module, so this
     surface can never drift from the toolbar's. */
  const improveButtons = onImprove
    ? lightboxImproveButtons({
      caps, engines: caps?.engines, improving, improvePending, improveReady,
      busy: (improveBusy ?? busy),
      busyReason,
    })
    : [];

  const improve = (engineId, disabled) => async (event) => {
    event.stopPropagation();
    if (!onImprove || disabled) return;
    patchImageState({ improving: true });
    try {
      await onImprove(img.id, engineId);
    } finally {
      patchImageState({ improving: false });
    }
  };

  const mirror = async (event) => {
    event.stopPropagation();
    if (!onMirror || pixelEditRefused || mirrorBusy) return;
    await onMirror(img.id);
  };

  // 🔄 Quarter turns (idea by 1Tomber, GitHub #17). `mirrorBusy` is the shared
  // "a pixel edit is running on this image" flag — both actions rewrite the same
  // file, so neither may start while the other is in flight.
  const rotate = (degrees) => async (event) => {
    event.stopPropagation();
    if (!onRotate || pixelEditRefused || mirrorBusy) return;
    await onRotate(img.id, degrees);
  };

  return (
    <div ref={dialogRef} role="dialog" aria-modal="true" aria-label={`Inspect — ${alt}`}
      data-probe-chrome="lightbox" data-probe-layer
      className={`fixed inset-0 z-[9996] bg-black/95 flex ${rail ? 'flex-row' : 'flex-col'}`}
      onClick={onClose}>
      <button type="button" ref={closeRef}
        onClick={(e) => { e.stopPropagation(); onClose(); }}
        title="Close (Esc)" aria-label="Close inspection"
        className="absolute top-3 right-3 z-10 w-10 h-10 lg:w-9 lg:h-9 rounded-full bg-white/10 hover:bg-white/20 text-white text-lg leading-none">✕</button>

      {/* The image area is the positioning context for ⟨ / ⟩ — NOT the dialog:
          in rail mode the dialog's right edge is the action rail, and an arrow
          anchored there would sit on top of the buttons. Anchoring here also
          keeps the arrows still while the 100 % view scrolls under them. */}
      <div className="relative flex min-h-0 min-w-0 flex-1">
      {inCompare ? (
        /* Side by side once there is room for it (≥640 px), stacked below —
           two 190 px-wide thumbnails on a phone would prove nothing, and a
           stacked pair keeps each image at full width where width is the scarce
           axis. Equal grid cells on both layouts = equal display scale. */
        <div onClick={(e) => e.stopPropagation()}
          /* The floating ☰ Actions button is the ONE thing allowed to overlap
             the picture, and in the stacked comparison the bottom pane is drawn
             right where it sits — so that pane pays for it in padding rather
             than in a covered chin. It costs each pane ~28 px; the placement it
             belongs to bought them ~210 each. */
          className={`flex-1 min-h-0 grid grid-rows-2 grid-cols-1 sm:grid-rows-1 sm:grid-cols-2 gap-2 p-2 sm:p-4 ${
            sheet ? 'pb-16' : ''}`}>
          <ComparePane label={activeCompare.beforeLabel} alt={alt}
            url={fileUrl(activeCompare.parent.filename, activeParentNonce)} />
          <ComparePane label={activeCompare.afterLabel} alt={alt} url={url} accent />
        </div>
      ) : full ? (
        <div className="flex-1 min-h-0 min-w-0 overflow-auto">
          <img src={url} alt={alt} onLoad={onImageLoad}
            onClick={(e) => { e.stopPropagation(); patchImageState({ full: false }); }}
            className="max-w-none cursor-zoom-out select-none" />
        </div>
      ) : (
        /* min-w-0 matters only in rail mode: without it this flex child refuses
           to shrink below its content and the rail gets pushed off-screen. */
        <div className="flex-1 min-h-0 min-w-0 flex items-center justify-center p-4">
          <img src={url} alt={alt} onLoad={onImageLoad}
            onClick={(e) => { e.stopPropagation(); patchImageState({ full: true }); }}
            className="max-h-full max-w-full object-contain cursor-zoom-in select-none" />
        </div>
      )}
      {canNavigate && (
        /* Both ends stay VISIBLE and say which end you reached. A button that
           disappears at the edge of a list moves the other one under the cursor
           mid-click, and "nothing happened" is indistinguishable from a bug.
           44 px so a thumb can reach them at 400 px. */
        <>
          <NavArrow side="left" glyph="⟨" label="Previous image"
            target={nav.prev} reason={nav.prevReason} onNavigate={onNavigate} />
          <NavArrow side="right" glyph="⟩" label="Next image"
            target={nav.next} reason={nav.nextReason} onNavigate={onNavigate} />
        </>
      )}
      </div>

      {/* DOM order is the SAME in both placements — meta, then compare, crop,
          mirror, rotate, improve, then the Klein note. A rail that reads left
          to right on screen but jumps around under Tab would trade one problem
          for a worse one, so nothing here reorders itself; only the axis
          changes. `overflow-y-auto` is the promise that a short window can
          still reach the last action. */}
      {/* THE one button, and only on a narrow screen. It floats over the
          picture instead of sitting in a strip of its own, because a strip is
          exactly the height this placement exists to give back: the panel it
          opens costs the image nothing while it is closed. It says what it
          opens — "Actions" alone would be a mystery-meat pill — and its state is
          carried by aria-expanded, not by its colour. */}
      {sheet && (
        <button type="button" ref={actionsBtnRef}
          onClick={(e) => { e.stopPropagation(); patchImageState({ actionsOpen: !actionsOpen }); }}
          aria-expanded={actionsOpen} aria-controls={panelId}
          aria-label={`Image actions for ${alt} — compare, crop, mirror, rotate, improve`}
          title="Image actions — compare, crop, mirror, rotate, improve"
          className="absolute bottom-3 left-1/2 z-20 flex min-h-11 -translate-x-1/2 items-center
            rounded-full border border-white/25 bg-black/75 px-5 py-2 text-sm font-semibold
            text-white shadow-lg hover:bg-black/90">
          <span aria-hidden="true">☰&nbsp;</span>Actions
        </button>
      )}
      <ActionsHost sheet={sheet} open={panelOpen} panelId={panelId}
        label={`Image actions — ${alt}`} closeRef={panelCloseRef} onDone={closePanel}>
      <div onClick={(e) => e.stopPropagation()}
        className={rail
          ? 'shrink-0 flex w-[17rem] flex-col items-stretch gap-2 overflow-y-auto border-l border-white/10 bg-black/60 px-4 pb-4 pt-14'
          : sheet
            ? 'flex flex-col items-stretch gap-2 px-4 pb-4 pt-3'
            : 'shrink-0 flex flex-wrap items-center justify-center gap-2 px-4 py-2.5 bg-black/60'}>
        {/* One group, so the rail stacks four lines of context instead of four
            full-width blocks (a stretched 10 px pill reads as a button). */}
        <div className={`flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 ${
          rail || sheet ? 'justify-start' : 'justify-center'}`}>
          <span className="text-white text-sm">{alt}</span>
          {/* Where you are in the list you are walking — the answer to "have I
              seen everything?", which arrows alone cannot give. It counts the
              images the grid SHOWS, so it moves when a filter does. */}
          {canNavigate && (
            <span className="rounded bg-white/10 px-1.5 py-0.5 text-[10px] tabular-nums text-white/80"
              title={`Image ${nav.position} of the images the current filters show — ← → to move`}>
              {nav.position}
            </span>
          )}
          {/* The verdict this image carries RIGHT NOW. Without it the three
              buttons below would be the only reading of a state they also
              change, and "did my K land?" would have no answer on a picture
              that happens to be the last of the list. Same words and colours as
              the Bank's review chips. */}
          {VERDICT_CHIP[img.status] && (
            <span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${VERDICT_CHIP[img.status].cls}`}
              title="The status this image carries in the dataset — only kept images are captioned, exported and trained on.">
              {VERDICT_CHIP[img.status].text}
            </span>
          )}
          <span className="px-1.5 py-0.5 rounded text-[10px] bg-white/10 text-white/80">
            {img.source === 'import' ? 'real' : 'generated'}{img.framing ? ` · ${img.framing}` : ''}
          </span>
          <SourceAttribution metadata={img.source_metadata}
            className="text-[11px] text-white/70" />
          <span className="text-white/50 text-[11px]">
            {/* Zoom is OFF in both comparisons, and says so. What the two panes
                GUARANTEE is not the same in both, and this line must not claim
                otherwise: against the original, equal boxes plus a preserved
                aspect ratio really do mean one shared scale, and that is the
                whole point. Against the reference the two images are unrelated
                crops — a square head next to a full-body plan — so each pane
                fits its own image at its own scale. Reusing "same scale" there
                would have been a sentence the pixels contradict. */}
            {compareMode === 'derived' && inCompare
              ? 'same scale — exit comparison to zoom to 100 %'
              : compareMode === 'reference' && inCompare
                ? 'different framings — each pane fits its own image; exit comparison to zoom to 100 %'
                : full ? '100 % — click image to fit' : 'fitted — click image for 100 %'}
          </span>
        </div>
        {onStatus && (
          /* ✓ Keep / ✕ Reject / ⏭ Skip — the Bank's review bar, in the screen
             where a dataset is actually curated. Same order, same glyphs, same
             colours and same keys, because the two screens ask the user the
             very same question and a learned reflex must not be right in only
             one of them. Each verdict then MOVES ON, which is what turns the
             lightbox from a viewer into a way through 300 pictures.
             The keys are printed on the buttons AND spelled out below: a
             shortcut nobody can discover is folklore. */
          <div className={`flex items-stretch gap-2 ${rail ? 'w-full flex-col' : 'w-full sm:w-auto'}`}>
            <button type="button" onClick={() => decide('keep')} disabled={deciding || curationRefused}
              aria-label={refused || `Keep ${alt} and move to the next image`}
              title={refused || 'Keep this image and move on (K) — kept images are the ones captioned, exported and trained on'}
              className="min-h-10 lg:min-h-9 flex-1 rounded-lg border border-emerald-400/60 bg-emerald-500/20 px-4 py-1.5 text-xs font-semibold text-emerald-100 hover:bg-emerald-500/30 disabled:cursor-not-allowed disabled:opacity-45">
              ✓ Keep<ShortcutKey>K</ShortcutKey>
            </button>
            <button type="button" onClick={() => decide('reject')} disabled={deciding || curationRefused}
              aria-label={refused || `Reject ${alt} and move to the next image`}
              title={refused || 'Reject this image and move on (R) — reversible, and nothing is deleted from disk'}
              className="min-h-10 lg:min-h-9 flex-1 rounded-lg border border-rose-400/60 bg-rose-500/20 px-4 py-1.5 text-xs font-semibold text-rose-100 hover:bg-rose-500/30 disabled:cursor-not-allowed disabled:opacity-45">
              ✕ Reject<ShortcutKey>R</ShortcutKey>
            </button>
            <button type="button" onClick={skipImage} disabled={!nextImage || !onNavigate}
              aria-label={nextImage
                ? `Skip ${alt} and move to the next image`
                : nav.nextReason || 'There is no next image to skip to'}
              title={nextImage
                ? 'Decide later (S) — moves on and leaves this image exactly as it is'
                : nav.nextReason || 'There is no next image to skip to'}
              className="min-h-10 lg:min-h-9 flex-1 rounded-lg border border-white/25 px-4 py-1.5 text-xs font-semibold text-white hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-45">
              ⏭ Skip<ShortcutKey>S</ShortcutKey>
            </button>
          </div>
        )}
        {onStatus && (
          <p className={`text-[11px] text-white/45 ${rail || sheet ? 'w-full' : ''}`}>
            {REVIEW_SHORTCUT_HINT} · ← → move without deciding · Esc close
          </p>
        )}
        {canCompare && (
          <button type="button" aria-pressed={compareMode === 'derived'}
            onClick={toggleCompare('derived')}
            aria-label={compareMode === 'derived'
              ? `Hide the original next to ${alt}`
              : `Show the original next to ${alt}`}
            title={COMPARE_HELP}
            className="min-h-10 lg:min-h-9 w-full sm:w-auto px-3 py-1.5 rounded-lg border border-indigo-400/50 bg-indigo-500/20 hover:bg-indigo-500/30 text-indigo-100 text-xs font-semibold">
            {compareMode === 'derived' ? '⊟ Exit comparison' : '⧉ Compare with original'}
          </button>
        )}
        {/* TWO buttons, not a picker. On an improved image both questions are
            live — "is it sharper than what it came from" and "is it still the
            same person" — and a selector would have hidden one behind the
            other. On a plainly generated variation only this one exists, which
            is the whole reason it was added: until now that image, the bulk of
            a character dataset, had no comparison at all. */}
        {canCompareRef && (
          <button type="button" aria-pressed={compareMode === 'reference'}
            onClick={toggleCompare('reference')}
            aria-label={compareMode === 'reference'
              ? `Hide the reference photo next to ${alt}`
              : `Show the reference photo next to ${alt}`}
            title={REFERENCE_COMPARE_HELP}
            className="min-h-10 lg:min-h-9 w-full sm:w-auto px-3 py-1.5 rounded-lg border border-sky-400/50 bg-sky-500/20 hover:bg-sky-500/30 text-sky-100 text-xs font-semibold">
            {compareMode === 'reference' ? '⊟ Exit comparison' : '◐ Compare with reference'}
          </button>
        )}
        {refCompare && !refCompare.available && (
          <span role="note"
            className="max-w-full break-words rounded-lg border border-amber-400/40 bg-amber-500/10 px-2 py-1 text-[11px] text-amber-100">
            <span aria-hidden>⚠ </span>{refCompare.reason}
          </span>
        )}
        {compare && !compare.available && (
          /* No affordance rather than a dead one — and it says why, because
             "there is no compare button here" is otherwise indistinguishable
             from a bug. */
          <span role="note"
            className="max-w-full break-words rounded-lg border border-amber-400/40 bg-amber-500/10 px-2 py-1 text-[11px] text-amber-100">
            <span aria-hidden>⚠ </span>{compare.reason}
          </span>
        )}
        {onCrop && (
          <button type="button" onClick={() => onCrop(img)} disabled={pixelEditRefused}
            title={pixelEditReason || 'Open the crop editor for this image (stretchable box, any ratio)'}
            aria-label={pixelEditReason || 'Open the crop editor for this image'}
            className="min-h-10 lg:min-h-9 px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-white text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-45">
            ✂ Crop
          </button>
        )}
        {/* The answer to a detector MISS. 🚩 Find watermarks is a classifier, and
            a mark it scores under the threshold used to be unanswerable: the mask
            editor only opened on images it had already flagged. Here the gesture
            starts from what the user can SEE, so drawing the zone is what flags
            the image. Hidden on an already-cleaned row — those pixels are gone,
            and ↩ Undo is the way back. */}
        {onMarkWatermark && img?.watermark_state !== 'cleaned' && (
          <button type="button" onClick={() => onMarkWatermark(img)} disabled={curationRefused}
            title={refused || 'Draw the watermark zones on this image — works even when the scan found nothing. What you draw becomes the flag, and 🧽 Clean then repaints exactly that.'}
            aria-label={refused || watermarkMaskButtonLabel(img)}
            className="min-h-10 lg:min-h-9 px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-white text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-45">
            🚩 {watermarkMaskButtonLabel(img)}
          </button>
        )}
        {/* ✦ TWO INTENTIONS, TWO BUTTONS — AND NOW TWO DESTINATIONS. 🚩 above
            flags a watermark the scan missed and sends you to the review screen
            that acts on watermarks. This one repaints anything you can describe,
            and opens the repair dialog right here: box or brush, a sentence, and
            only the area you pointed at changes. They shared one destination
            until now, which made the pair look like a duplicate. */}
        {onRepair && (
          <button type="button" onClick={() => patchImageState({ repairOpen: true })} disabled={curationRefused}
            title={refused || 'Repaint part of this image from your own description — draw a box or paint over the thing with the brush, say what should be there ("remove the necklace"), and everything outside it stays byte-identical.'}
            aria-label={refused || 'Repair an area of this image'}
            className="min-h-10 lg:min-h-9 px-3 py-1.5 rounded-lg bg-sky-500/25 hover:bg-sky-500/35 text-sky-50 text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-45">
            ✦ Repair
          </button>
        )}
        {onMirror && (
          <button type="button" onClick={mirror} disabled={pixelEditRefused || mirrorBusy}
            aria-busy={mirrorBusy}
            aria-label={pixelEditReason
              || (mirrorBusy ? `Mirroring ${alt} horizontally` : `Mirror ${alt} horizontally`)}
            title={refused
              || (mirrorBusy ? 'Mirroring horizontally…' : 'Mirror horizontally (flip left and right)')}
            className="min-h-10 lg:min-h-9 w-full sm:w-auto px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-white text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-45">
            {mirrorBusy ? '⇆ Mirroring…' : '⇆ Mirror horizontally'}
          </button>
        )}
        {onRotate && (
          /* The pair shares ONE row even on a 400 px screen: two full-width
             rows for two halves of the same gesture would push everything else
             below the fold. In the rail the same row spans its full width —
             `sm:flex-none` would otherwise leave them huddled on the left,
             reading as a different kind of control than their neighbours.
             Emoji stay aria-hidden — the label is the text. */
          <div className={`flex items-stretch gap-2 ${rail ? 'w-full' : 'w-full sm:w-auto'}`}>
            <button type="button" onClick={rotate(270)} disabled={pixelEditRefused || mirrorBusy}
              aria-busy={mirrorBusy} aria-label={pixelEditReason || `Rotate ${alt} 90 degrees left`}
              title={pixelEditReason
                || "Rotate 90° left (counter-clockwise) — keeps the file's format; four turns come back round"}
              className={`min-h-10 lg:min-h-9 flex-1 rounded-lg bg-white/10 px-3 py-1.5 text-xs font-semibold text-white hover:bg-white/20 disabled:cursor-not-allowed disabled:opacity-45 ${
                rail ? '' : 'sm:flex-none'}`}>
              <span aria-hidden="true">↺</span> Rotate left
            </button>
            <button type="button" onClick={rotate(90)} disabled={pixelEditRefused || mirrorBusy}
              aria-busy={mirrorBusy} aria-label={pixelEditReason || `Rotate ${alt} 90 degrees right`}
              title={pixelEditReason
                || "Rotate 90° right (clockwise) — keeps the file's format; four turns come back round"}
              className={`min-h-10 lg:min-h-9 flex-1 rounded-lg bg-white/10 px-3 py-1.5 text-xs font-semibold text-white hover:bg-white/20 disabled:cursor-not-allowed disabled:opacity-45 ${
                rail ? '' : 'sm:flex-none'}`}>
              <span aria-hidden="true">↻</span> Rotate right
            </button>
          </div>
        )}
        {improveButtons.map((btn) => (
          <Fragment key={btn.id}>
            <button type="button"
              onClick={improve(btn.id, btn.disabled)} disabled={btn.disabled}
              aria-busy={improvementActive} title={btn.title}
              className="min-h-10 lg:min-h-9 w-full sm:w-auto px-3 py-1.5 rounded-lg border border-indigo-400/50 bg-indigo-500/20 hover:bg-indigo-500/30 text-indigo-100 text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-45">
              {btn.label}
            </button>
            {/* Klein's note goes BETWEEN the two buttons in the rail, and only
                there. The rail is a column, so sitting under Klein is what makes
                it read as Klein's — which matters, because it warns that Klein's
                INSTRUCTION ("detailed texture, sharp details") pulls drawn skin
                towards realism, while SeedVR2 sends no instruction at all.
                In the BOTTOM bar the buttons are a horizontal ROW, and a
                full-width paragraph dropped mid-row pushes everything after it
                onto its own line: a user reported the second improve button
                stranded alone, centred, at the very bottom of the screen. There
                the note therefore follows the whole group (see below) — its own
                first words, "Improve asks Klein to:", carry the attribution that
                position gave it in the rail. */}
            {rail && btn.showKleinNote && !improvementActive && (
              <KleinImproveNote subjectType={subjectType} datasetId={datasetId}
                className="w-full border-t border-white/10 pt-2" />
            )}
          </Fragment>
        ))}
        {/* Bottom bar only: the note takes its OWN line under the buttons.
            `sm:w-auto` used to let it sit INLINE beside them, which was fine
            with a single improve button and is not with two — the paragraph
            took the width the second button needed and pushed it off alone.
            Full-width in a wrap container is a line break, so the buttons wrap
            among themselves and the note reads under the whole group. */}
        {!rail && improveButtons.some((b) => b.showKleinNote) && !improvementActive && (
          <KleinImproveNote subjectType={subjectType} datasetId={datasetId}
            className="w-full" />
        )}
        {/* Its strength, step count and instruction are all editable, and nothing
            here said so — the reported case for making settings discoverable from
            where the action happens. A link alone was not enough: it pointed at
            the strength knobs while the complaint ("anime comes back realistic",
            Qeeyana on Reddit) is caused by the INSTRUCTION. The note quotes that
            instruction live and links to both. */}
        {/* It stays glued to ✨, in both placements. It is what the button is
            about to ask the model for, so it is only worth reading in the
            second before clicking — parked anywhere else it becomes a stray
            paragraph. The rail is where it fits BEST: it is prose, and a 15rem
            column is a better shape for prose than a strip squeezed to the
            right of six buttons. A rule above it ties it to the ✨ it explains. */}

      </div>
      </ActionsHost>
      {/* Mounted INSIDE the root so it inherits the stacking context; the dialog
          stops its own clicks, which matters here because this root closes on a
          backdrop click. */}
      <RepairDialog open={repairOpen && !!img?.filename} src={url} alt={alt}
        onClose={() => patchImageState({ repairOpen: false })}
        onSubmit={({ boxes, mask, prompt }) => onRepair(img.id, prompt, boxes, mask)}
        onUndo={onUndoRepair ? () => onUndoRepair(img.id) : null} />
    </div>
  );
}
