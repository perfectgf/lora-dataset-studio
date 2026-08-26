/** One curation tile: image + keep/reject + source/framing badges + caption + crop. */
import { improvementBadge } from './improveCandidates.js';
import SelectionMark from '../shared/SelectionMark';
import { Drama, Eye, Flag, Sparkles, Trash2 } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { displayLabel } from '../../utils/labels';
// WHO wrote this tile's caption — the per-image half of the provenance the pass
// already reports in aggregate (utils/captionEngines.js).
import { captionIsAsserted, captionOriginInfo } from '../../utils/captionOrigin.js';
import { isSmallImageRescueRow } from '../../utils/smallImageRescue';
import CaptionEditorDialog from './CaptionEditorDialog';
import PromptEditPopover from './PromptEditPopover';
import SourceAttribution from './SourceAttribution';
import { ENGINE_ACCENTS, ENGINE_LABELS } from './engineSelection.js';
import { canRegenerateGeneric, improveRerunAffordance, isImageImproveRow } from './improveRerun.js';
import { rememberImageRatio } from './lightboxActionPlacement.js';
import { datasetThumbUrl } from '../../utils/datasetThumbUrl.js';
import { poseLabel } from '../../utils/cameraAngles.js';
import { FACE_BADGE_CLASS, PROVENANCE_BADGE_CLASS, TILE_BADGE_STACK_CLASS,
  WATERMARK_BADGE_CLASS } from './tileBadgeLayout.js';

/* Provenance wording per STORED derivation kind. `klein_image_improve` is a
   legacy key: it predates the second engine and is written in user databases,
   so a SeedVR2 result carries it too and the badge must NOT claim Klein made
   it. The engine is named by the candidate's own label ('Klein upscale &
   improve' / 'SeedVR2 upscale'), which is where a per-row truth belongs; this
   badge says only what KIND of row it is, which is true for both. */
const DERIVATION_LABEL = {
  klein_small_image: 'Klein rescue',
  small_image_source: 'rescue original',
  klein_image_improve: 'upscale candidate',
  camera_angle: 'camera view',
};

const STATUS_CLS = {
  keep: 'border-green-500',
  reject: 'border-red-500/50 opacity-50',
  pending: 'border-amber-400/40',
  failed: 'border-red-600',
};

// Seuils calibres antelopev2 (test3) — face_score brut persiste -> ajustables dans
// Settings (face_scoring.green/orange) ; ces valeurs ne servent que de repli.
const DEFAULT_FACE_VALID = 0.50, DEFAULT_FACE_ORANGE = 0.45;
const GREY_LABEL = { no_face: 'no face detected', low_det: 'low detection',
  too_small: 'face too small', extreme_pose: 'profile — not scored',
  unreadable: 'unreadable', error: 'error' };

// Retourne {border, icon, cls, label} d'apres face_state/face_score, ou null si pas analysé.
// La bordure encode la largeur ET le style (plein=jugé / pointillé=non-jugeable) pour
// ne PAS dépendre de la couleur seule (WCAG 1.4.1).
function faceBadge(img, thresholds) {
  if (img.face_state == null) return null;
  if (img.face_state !== 'scorable' || img.face_score == null) {
    return { border: 'border-2 border-dashed border-gray-500', icon: Eye, cls: 'text-gray-300',
      label: GREY_LABEL[img.face_state] || 'not scored' };
  }
  const green = thresholds?.green ?? DEFAULT_FACE_VALID;
  const orange = thresholds?.orange ?? DEFAULT_FACE_ORANGE;
  const s = img.face_score;
  if (s >= green) return { border: 'border-2 border-green-500', icon: '✓', cls: 'text-green-300', label: s.toFixed(2) };
  if (s >= orange) return { border: 'border-2 border-amber-500', icon: '~', cls: 'text-amber-300', label: `${s.toFixed(2)} to review` };
  return { border: 'border-4 border-red-500', icon: '⚠', cls: 'text-red-300', label: `${s.toFixed(2)} low` };
}

// Watermark V1 badge from watermark_state (🚩 detected / ⊘ dismissed / ✨ cleaned /
// ⚠ failed), or null when never scanned ('none' is also silent — nothing to show).
// `dismissed` shows a DISCREET grey ⊘ (not nothing): it confirms the user's "not a
// watermark" ruling took effect and explains why a re-scan won't re-flag it — silence
// would read as "did my dismiss work?". The tooltip names what Clean will do; when the
// payload carries the exact route (watermark_route, computed backend-side from the
// dims) the detected tooltip names the precise action, else it lists the possibilities.
const WATERMARK_ROUTE_HINT = {
  crop: 'Overlaid watermark on the border — Clean will crop it off',
  lama: 'Small off-center watermark — Clean will inpaint it (LaMa)',
  review: 'Watermark on the subject — Clean flags it for manual review (auto crop/inpaint would damage the photo); reject or crop manually',
};
const WATERMARK_BADGE = {
  detected: { icon: Flag, cls: 'text-amber-300', text: 'watermark',
    label: 'Overlaid watermark detected — Clean will crop the border, inpaint a small mark, or flag it for manual review (V2 handles on-subject watermarks)' },
  dismissed: { icon: '⊘', cls: 'text-content-subtle', text: 'not a watermark',
    label: 'You marked this “not a watermark” — future 🧽 Find passes skip it' },
  cleaned: { icon: Sparkles, cls: 'text-emerald-300', text: 'watermark', label: 'Watermark removed (original kept as a .orig backup)' },
  failed: { icon: '⚠', cls: 'text-red-300', text: 'watermark', label: 'Watermark removal failed' },
};

/* READS ARE NOT WRITES.
 * `busy` means "a pass is running on this dataset, so nothing here may CHANGE
 * it". It used to switch off the whole tile — including the button that opens
 * the image and the tick that selects it, neither of which writes anything —
 * which is how "during a generation everything around the dataset is blocked"
 * became the app's most-reported frustration. Inspecting and ticking now ignore
 * `busy` entirely; every button below that touches pixels, status, captions or
 * files still refuses, and `busyReason` is the sentence it shows instead of
 * going quietly grey.
 */
export default function DatasetGridItem({ img, datasetId, onStatus, onCaption, onCrop, onDelete,
                                          onMirror, mirrorBusy = false, busy = false,
                                          /* The buttons below that START a queued job read these instead of
                                             `busy`: they add a row to a queue that is already serialized, so a
                                             batch running elsewhere is no reason to refuse them (GitHub #44).
                                             TWO gates, not one, because they gate different work: 🔄✨ re-improve
                                             is improve-lane (the backend refuses a second one), while 🔄 and ✏️
                                             enqueue a plain 'generate' it accepts freely. One shared flag meant
                                             the retries inherited the improve refusal and stayed grey for the
                                             whole improve batch. Every other button here writes to THIS image
                                             and keeps `busy`. */
                                          improveBusy = undefined, generateBusy = undefined,
                                          /* And the writes that CURATE this image — keep/reject, caption,
                                             crop, mirror, rotate, delete, score, watermark. Queued work is
                                             not a reason to refuse them: `delete_image` cancels the job in
                                             flight and refuses outright when it cannot prove it,
                                             `gpu_exclusive_vision_window` is fail-closed and says so in
                                             words, and `crop_image` cannot touch a row with no file yet.
                                             A pass that owns the ROWS still blocks them. */
                                          curationBusy = undefined,
                                          busyReason = null,
                                          onScoreFace, scoreFaceBusy = false, faceScoringBusy = false, faceScoringBlocked = null,
                                          onRegenerate, onReimprove, onView, nonce = 0, faceThresholds,
                                          selected = false, onToggleSelect, tileSize = 'M',
                                          datasetKind = 'character', dualCaptions = false,
                                          improvementState = undefined }) {
  const [cap, setCap] = useState(img.caption || '');
  const [captionEditorOpen, setCaptionEditorOpen] = useState(false);
  // ✏️ edit-prompt bubble open state (regenerate this tile with an edited prompt).
  const [editingPrompt, setEditingPrompt] = useState(false);
  // While the textarea has focus, a poll-driven refresh must never overwrite
  // the draft (C1) — the server value only syncs in when nobody is typing.
  const editingRef = useRef(false);
  // Sync the textarea when the server fills/updates the caption (e.g. after the
  // Qwen3-VL captioning pass) — useState's initial value alone would stay stale.
  useEffect(() => { if (!editingRef.current) setCap(img.caption || ''); }, [img.caption]);
  // `nonce` busts the browser cache after an in-place crop (same filename).
  const url = img.filename
    ? `/api/dataset/${datasetId}/img/${encodeURIComponent(img.filename)}${nonce ? `?v=${nonce}` : ''}`
    : null;
  // Regenerate applies to generated tiles that are not mid-generation —
  // finished AND failed ones (failure recovery path) (F2).
  const isRescueDerived = isSmallImageRescueRow(img);
  // A manual Klein improvement is derived from THIS image, not the dataset's
  // main reference. Sending it through the generic regenerate route would lose
  // that source and silently make an unrelated variation instead — so it gets
  // its OWN re-run below (same parent, current improve settings) instead.
  const isImageImproveCandidate = isImageImproveRow(img);
  const canRegenerate = canRegenerateGeneric(img, { isRescueDerived });
  const rerunImprove = onReimprove ? improveRerunAffordance(img) : null;
  // Every refused write says WHICH pass holds it; idle, each keeps its own words.
  // There is no longer a single `refused`: a write is held by ONE of three gates
  // and must name that one, or it explains itself with a pass that is not the
  // one refusing it.
  const improveRefused = (improveBusy ?? busy);
  const curationRefused = (curationBusy ?? busy);
  const generateRefused = (generateBusy ?? busy);
  // The sentence each of them shows when IT is the one refusing. Reusing
  // `refused` would have named a pass that no longer blocks them.
  const improveRefusedReason = improveRefused ? busyReason : null;
  const curationRefusedReason = curationRefused ? busyReason : null;
  // Rewriting the pixels while an upscale of THIS image is rendering is the one
  // case queued work really does make awkward: the pass copied its source into
  // ComfyUI's input at enqueue time, so it would come back as an upscale of the
  // version from before your edit. Nothing is corrupted and nothing else is
  // held up — so it is refused on this tile only, and says exactly why, rather
  // than through a dialog the other tiles would also have to answer.
  const upscaleRendering = improvementState === 'generating';
  const pixelEditRefused = curationRefused || upscaleRendering;
  const pixelEditReason = curationRefusedReason
    || (upscaleRendering
      ? 'An upscale of this image is still rendering — it would come back as an '
        + 'upscale of the version from before your edit. It will be available '
        + 'again once that result arrives.'
      : null);

  const scoreFaceTitle = faceScoringBlocked
    || (scoreFaceBusy
      ? 'Scoring facial resemblance to the reference…'
      : faceScoringBusy
        ? 'Face scoring is already running for another image…'
        : curationRefused
          ? (busyReason || 'Wait for the current dataset action to finish before scoring.')
          : 'Score facial resemblance to the reference');
  const generateRefusedReason = generateRefused ? busyReason : null;

  const fb = faceBadge(img, faceThresholds);
  const wb = WATERMARK_BADGE[img.watermark_state];
  const borderCls = fb ? fb.border : `border-2 ${STATUS_CLS[img.status] || 'border-border'}`;
  // The tile stays a square (crop decisions need a stable grid), but at the L
  // size — fewer, bigger tiles, the whole point being to judge a composition
  // before deciding — a hard object-cover square crop hides exactly what you'd
  // need to see (is this shot portrait or landscape?). So L switches to
  // object-contain (letterboxed on the existing black tile background); S/M
  // stay object-cover so the dense overview grid reads as a clean tiled wall.
  const imgFitCls = tileSize === 'L' ? 'object-contain' : 'object-cover';
  // Provenance badge text. Kept as data (not inline JSX) so the SAME wording
  // can go into title/aria-label — the badge is clamped to two lines at the
  // bottom of a narrow tile, and a truncated engine name must stay readable
  // on hover and to a screen reader.
  // 'ready' | 'generating' | undefined, decided once for the whole grid
  // (DatasetGrid) so each tile is a lookup, not a scan of every sibling.
  const improvementBadgeInfo = improvementBadge(improvementState);
  // 📷 A camera view names its ANGLE, not just its kind: eight of them side by
  // side in a grid are unreadable as eight "camera view" chips, and the angle
  // is the one fact that tells them apart. poseLabel degrades to null on an
  // unreadable pose, and then the bare kind is still the truth.
  const originText = (img.derivation_kind === 'camera_angle'
    && poseLabel(img.camera_pose))
    || DERIVATION_LABEL[img.derivation_kind]
    || (img.source === 'import' ? 'real' : 'generated');
  const engineLabel = ENGINE_ACCENTS[img.engine] ? ENGINE_LABELS[img.engine] : null;
  const provenanceTitle = [originText, img.framing, engineLabel && `made with ${engineLabel}`]
    .filter(Boolean).join(' · ');

  return (
    <div tabIndex={0} aria-label={`${displayLabel(img.variation_label) || 'Dataset image'} card`}
      className={`dataset-grid-item rounded-lg ${borderCls} ${selected ? 'ring-2 ring-primary' : ''} bg-app/40 overflow-hidden flex flex-col`}>
      <div className="relative aspect-square bg-black">
        {selected && <SelectionMark />}
        {onToggleSelect && img.filename && (
          <label
            className="dataset-grid-item__actions absolute bottom-1 left-1 z-10 flex items-center justify-center w-6 h-6 rounded bg-black/60 cursor-pointer"
            title="Select for bulk actions"
            onClick={(e) => e.stopPropagation()}>
            {/* NOT gated on `busy`: ticking changes nothing on the server, and
                a running pass was handed its id list at launch time — a
                selection made afterwards cannot shift what it is processing.
                What DOES gate this tick is a bulk action currently CONSUMING
                the selection, and that one is decided by the grid, which
                withholds `onToggleSelect` entirely. */}
            <input type="checkbox" checked={selected}
              onChange={() => onToggleSelect(img.id)}
              aria-label={`Select ${displayLabel(img.variation_label) || 'this image'} for bulk actions`}
              className="w-4 h-4 accent-indigo-500 cursor-pointer" />
          </label>
        )}
        {url ? (
          // A pure read: it opens the same bytes the tile is already showing.
          // No `disabled={busy}` — that is the whole point of this change.
          <button type="button" onClick={() => onView?.(img)}
            title="Inspect (zoom)"
            aria-label={`Inspect ${displayLabel(img.variation_label) || 'the image'} full screen`}
            className="block w-full h-full cursor-zoom-in disabled:cursor-not-allowed">
            {/* The TILE fetches a thumbnail, the lightbox fetches the file: a
                grid of 512 px WebPs instead of a wall of 1-4 megapixel PNGs is
                the difference between a dataset that opens and one that churns.
                The intrinsic size recorded here is still the right one — a
                thumbnail keeps the source's ASPECT RATIO, and the ratio is all
                lightboxActionPlacement.js reads to open with its actions already
                on the correct side instead of committing them once the image
                paints. */}
            <img src={datasetThumbUrl(url, 512)} alt={displayLabel(img.variation_label)}
              loading="lazy" decoding="async"
              onLoad={(e) => rememberImageRatio(
                img.id, e.currentTarget.naturalWidth, e.currentTarget.naturalHeight)}
              className={`w-full h-full ${imgFitCls}`} />
          </button>
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center gap-1 px-2 text-center"
            title={img.status === 'failed' ? (img.fail_reason || 'generation failed') : undefined}>
            {img.status === 'failed' ? (
              <>
                <span className="text-red-300 text-xs font-semibold">⚠ failed</span>
                {img.fail_reason && (
                  <span className="text-content-subtle text-[0.5625rem] leading-tight line-clamp-4 break-words">
                    {img.fail_reason}
                  </span>
                )}
                <span className="text-content-subtle text-[0.5625rem]">🔄 to retry</span>
              </>
            ) : (
              <span className="text-content-subtle text-xs">…</span>
            )}
          </div>
        )}
        {/* Bottom-anchored badge stack. A container query (index.css) lifts the
            provenance badge back to the top-left as soon as the TILE — not the
            window — is wide enough to hold it next to the action buttons. */}
        <div className={TILE_BADGE_STACK_CLASS}>
          {/* An upscale of THIS image is waiting (or still rendering). The
              candidate is a separate row that lands on its own tile elsewhere
              in the grid, so from here nothing looked like it had happened —
              and the pass got re-run on images that already had a result,
              paying GPU time for a duplicate. */}
          {improvementBadgeInfo && (
            <span className={`${WATERMARK_BADGE_CLASS} bg-black/70 ${
              improvementBadgeInfo.tone === 'ready'
                ? 'text-indigo-200' : 'text-white/70'}`}
              title={improvementBadgeInfo.title} aria-label={improvementBadgeInfo.title}>
              {improvementBadgeInfo.text}
            </span>
          )}
          {wb && (
            <span className={`${WATERMARK_BADGE_CLASS} bg-black/70 ${wb.cls}`}
              title={(img.watermark_state === 'detected' && WATERMARK_ROUTE_HINT[img.watermark_route]) || wb.label}>
              {typeof wb.icon === 'string' ? wb.icon : <wb.icon aria-hidden="true" className="mr-0.5 inline h-3 w-3 align-[-1px]" />}{wb.text}
            </span>
          )}
          {/* Last child = closest to the bottom edge, and the engine pill names
              which engine made it — the only way a multi-engine batch is
              comparable ("this one came out of ChatGPT, that one out of
              Klein"). The server sends `engine` ONLY when it can tell for sure,
              so older rows show no pill rather than a made-up one. */}
          <span className={`${PROVENANCE_BADGE_CLASS} bg-black/60 text-white`}
            title={provenanceTitle} aria-label={provenanceTitle}>
            {originText}{img.framing ? ` · ${img.framing}` : ''}
            {engineLabel && (
              <span className={`dataset-tile-badge__engine ml-1 px-1 rounded ${ENGINE_ACCENTS[img.engine].pill}`}>
                {engineLabel}
              </span>
            )}
          </span>
        </div>
        {fb && (
          <span className={`${FACE_BADGE_CLASS} px-1.5 py-0.5 rounded bg-black/70 ${fb.cls}`}
            title={`Resemblance to the reference face — ${fb.label}`}>
            {typeof fb.icon === 'string' ? fb.icon : <fb.icon aria-hidden="true" className="mr-0.5 inline h-3 w-3 align-[-1px]" />} {fb.label}
          </span>
        )}
        <div className="dataset-grid-item__actions absolute top-1 right-1 flex max-w-[calc(100%_-_0.5rem)] flex-wrap justify-end gap-1">
          {url && onScoreFace && ['keep', 'pending'].includes(img.status) && (
            <button type="button"
              onClick={(e) => { e.stopPropagation(); onScoreFace(img.id); }}
              disabled={curationRefused || faceScoringBusy || !!faceScoringBlocked || scoreFaceBusy}
              aria-busy={scoreFaceBusy}
              title={scoreFaceTitle} aria-label={scoreFaceTitle}
              className="grid min-h-7 min-w-7 place-items-center rounded bg-black/60 text-[10px] text-white disabled:cursor-not-allowed disabled:opacity-45">
              <span aria-hidden="true" className={scoreFaceBusy ? 'animate-pulse' : ''}>{scoreFaceBusy
                ? '…' : <Drama aria-hidden="true" className="h-3.5 w-3.5" />}</span>
            </button>
          )}
          {canRegenerate && (
            <button type="button"
              onClick={(e) => { e.stopPropagation(); onRegenerate?.(img.id); }}
              disabled={generateRefused}
              title={generateRefusedReason || 'Regenerate this variation (new seed)'}
              aria-label={generateRefusedReason || 'Regenerate this variation (new seed)'}
              className="px-1.5 py-0.5 rounded bg-black/60 text-white text-[10px] disabled:cursor-not-allowed disabled:opacity-45">🔄</button>
          )}
          {canRegenerate && (
            <button type="button"
              onClick={(e) => { e.stopPropagation(); setEditingPrompt(true); }}
              disabled={generateRefused}
              title={generateRefusedReason || 'Edit the prompt, then regenerate this variation'}
              aria-label={generateRefusedReason || 'Edit the prompt, then regenerate this variation'}
              className="px-1.5 py-0.5 rounded bg-black/60 text-white text-[10px] disabled:cursor-not-allowed disabled:opacity-45">✏️</button>
          )}
          {rerunImprove && (
            <button type="button"
              onClick={(e) => { e.stopPropagation(); onReimprove?.(img.id); }}
              disabled={improveRefused || !rerunImprove.enabled}
              title={improveRefusedReason || rerunImprove.title}
              aria-label={improveRefusedReason || rerunImprove.title}
              className="grid min-h-7 min-w-7 place-items-center rounded bg-black/60 text-[10px] text-white disabled:cursor-not-allowed disabled:opacity-45">
              <span aria-hidden="true">🔄✨</span>
            </button>
          )}
          {url && onMirror && (
            <button type="button"
              onClick={(e) => { e.stopPropagation(); onMirror(img.id); }}
              disabled={pixelEditRefused || mirrorBusy}
              aria-busy={mirrorBusy}
              aria-label={pixelEditReason || (mirrorBusy
                ? `Mirroring ${displayLabel(img.variation_label) || 'this image'} horizontally`
                : `Mirror ${displayLabel(img.variation_label) || 'this image'} horizontally`)}
              title={pixelEditReason
                || (mirrorBusy ? 'Mirroring horizontally…' : 'Mirror horizontally (flip left and right)')}
              className="grid min-h-7 min-w-7 place-items-center rounded bg-black/60 text-[10px] text-white disabled:cursor-not-allowed disabled:opacity-45">
              <span aria-hidden="true">{mirrorBusy ? '…' : '⇆'}</span>
            </button>
          )}
          {url && (
            <button type="button" onClick={(e) => { e.stopPropagation(); onCrop(img); }}
              disabled={pixelEditRefused}
              title={pixelEditReason || 'Crop'} aria-label={pixelEditReason || 'Crop'}
              className="px-1.5 py-0.5 rounded bg-black/60 text-white text-[10px] disabled:cursor-not-allowed disabled:opacity-45">✂</button>
          )}
          {!isRescueDerived && (
            <button type="button"
              onClick={(e) => { e.stopPropagation(); if (window.confirm('Permanently delete this image?')) onDelete(img.id); }}
              disabled={curationRefused}
              title={curationRefusedReason || 'Delete permanently'}
              aria-label={curationRefusedReason || 'Delete permanently'}
              className="px-1.5 py-0.5 rounded bg-red-700/80 text-white text-[10px] disabled:cursor-not-allowed disabled:opacity-45"><Trash2 aria-hidden="true" className="h-3 w-3" /></button>
          )}
        </div>
        {editingPrompt && (
          <PromptEditPopover
            initialPrompt={img.variation_prompt || ''}
            onSubmit={(prompt) => onRegenerate?.(img.id, undefined, prompt, { silent: true })}
            onClose={() => setEditingPrompt(false)} />
        )}
      </div>
      <SourceAttribution metadata={img.source_metadata}
        className="mx-1.5 mt-1 block text-[0.625rem] leading-tight text-content-subtle" />
      {isRescueDerived ? (
        <p className="m-1.5 rounded border border-indigo-400/30 bg-indigo-500/10 px-2 py-1 text-center text-[0.625rem] text-indigo-200"
          title="This winner was chosen atomically with its provenance pair. Caption and crop remain available.">
          ✓ Chosen in Klein rescue review
        </p>
      ) : (
        <div className="dataset-grid-item__actions flex gap-1 p-1.5">
          <button type="button" onClick={() => onStatus(img.id, img.status === 'keep' ? 'pending' : 'keep')}
            disabled={curationRefused}
            title={curationRefusedReason || 'Keep'} aria-label={curationRefusedReason || 'Keep'}
            aria-pressed={img.status === 'keep'}
            className={`flex-1 py-1 rounded text-[11px] disabled:cursor-not-allowed disabled:opacity-45 ${img.status === 'keep' ? 'bg-green-600 text-white' : 'bg-surface text-content-muted'}`}>✓</button>
          <button type="button"
            onClick={() => {
              // Rejecting a GENERATED image offers an immediate retry of the same
              // variation (in place, new seed) so the composition stays on target.
              if (!isImageImproveCandidate && img.status !== 'reject'
                  && img.source === 'generated' && img.filename && onRegenerate
                  && window.confirm('Photo rejected — regenerate a new attempt of this variation?\n(OK = replace with a new attempt · Cancel = reject only)')) {
                onRegenerate(img.id);
                return;
              }
              onStatus(img.id, img.status === 'reject' ? 'pending' : 'reject');
            }}
            disabled={curationRefused}
            title={curationRefusedReason || 'Reject (offers a regeneration)'} aria-label={curationRefusedReason || 'Reject'}
            aria-pressed={img.status === 'reject'}
            className={`flex-1 py-1 rounded text-[11px] disabled:cursor-not-allowed disabled:opacity-45 ${img.status === 'reject' ? 'bg-red-600 text-white' : 'bg-surface text-content-muted'}`}>✕</button>
        </div>
      )}
      {img.status === 'keep' && (
        <div className="m-1.5 mt-0 flex flex-col gap-1">
          <div className="dataset-grid-item__actions flex items-center justify-end gap-1">
            <button type="button" onClick={() => setCaptionEditorOpen(true)}
              disabled={curationRefused}
              title={curationRefusedReason || 'Open a larger caption editor'}
              aria-label={curationRefusedReason || 'Expand caption editor'}
              className="rounded border border-border bg-surface px-1.5 py-0.5 text-[10px] text-content-muted hover:text-content disabled:cursor-not-allowed disabled:opacity-45">
              ⛶ Expand
            </button>
            {cap && (
              <button type="button"
                onClick={() => { editingRef.current = false; setCap(''); onCaption(img.id, ''); }}
                disabled={curationRefused}
                title={curationRefusedReason || 'Delete this image’s caption (then “Caption” regenerates it via JoyCaption)'}
                aria-label={curationRefusedReason || 'Delete this image’s caption'}
                className="rounded border border-red-500/40 bg-red-500/15 px-1.5 py-0.5 text-[10px] text-red-300 hover:bg-red-500/25 disabled:cursor-not-allowed disabled:opacity-45">
                <Trash2 aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Caption
              </button>
            )}
          </div>
          {/* WHO WROTE THE TEXT IN THE BOX BELOW.
              ITS OWN LINE, not a badge slipped into the action row: at the M tile
              density that row already holds ⛶ Expand and 🗑 Caption, and a third
              item there is clipped at the tile's left edge — measured headless at
              that density, where the chip read "yCaption".
              ALWAYS VISIBLE, unlike that row: this is a readout, not an action, and
              a provenance you have to hover to discover is one nobody discovers.
              Only when there IS a caption and its author was recorded — stamping
              "author not recorded" on every legacy tile would be a grid of identical
              chips, which is noise; the expanded editor has room to say it and does. */}
          {(cap || '').trim() && captionOriginInfo(img.caption_origin).known && (
            <span title={captionOriginInfo(img.caption_origin).title}
              aria-label={captionOriginInfo(img.caption_origin).short}
              className={`block truncate text-[10px] leading-none ${
                captionIsAsserted(img.caption_origin)
                  ? 'text-emerald-300' : 'text-content-subtle'}`}>
              {captionOriginInfo(img.caption_origin).chip}
            </span>
          )}
          <textarea value={cap} onChange={(e) => setCap(e.target.value)}
            disabled={curationRefused} title={curationRefusedReason || undefined}
            onFocus={() => { editingRef.current = true; }}
            onBlur={() => {
              editingRef.current = false;
              if (!busy && cap !== (img.caption || '')) onCaption(img.id, cap);
            }}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); e.currentTarget.blur(); } }}
            rows={2} placeholder={datasetKind === 'style'
              ? 'required: content only, no aesthetic or trigger…'
              : datasetKind === 'concept'
                ? 'caption without naming the concept…'
                : 'caption (without the face)…'} aria-label="Image caption"
            className="text-[11px] bg-app/60 border border-border rounded p-1 text-content resize-none" />
        </div>
      )}
      {captionEditorOpen && (
        <CaptionEditorDialog initialCaption={cap} imageUrl={url}
          datasetId={datasetId} imageId={img.id}
          initialShortCaption={img.caption_short || ''} showShort={dualCaptions}
          captionOrigin={img.caption_origin} shortCaptionOrigin={img.caption_short_origin}
          imageLabel={displayLabel(img.variation_label)}
          onClose={() => setCaptionEditorOpen(false)}
          /* Awaited, and its answer is handed BACK: an unawaited handler cannot
             know it was refused, which is how a refused save used to close the
             editor and destroy the caption. `silent` because the dialog draws
             the refusal itself, next to the text it is about. The tile's own
             copy is only advanced on a success — otherwise a failed save would
             show the new text on a tile the server never accepted. */
          onSave={async (nextCaption, nextShort) => {
            if (curationRefused) {
              return { ok: false,
                error: curationRefusedReason
                  || 'Wait for the running dataset pass to finish before saving.' };
            }
            // Persist when either field changed; `nextShort` is undefined unless dual is on.
            const changed = nextCaption !== (img.caption || '')
              || (nextShort !== undefined && nextShort !== (img.caption_short || ''));
            if (!changed) return { ok: true };
            const outcome = await onCaption(img.id, nextCaption, nextShort, { silent: true });
            if (outcome?.ok) { editingRef.current = false; setCap(nextCaption); }
            return outcome;
          }} />
      )}
    </div>
  );
}
