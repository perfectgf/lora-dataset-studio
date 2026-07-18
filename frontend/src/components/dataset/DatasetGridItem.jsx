/** One curation tile: image + keep/reject + source/framing badges + caption + crop. */
import { useEffect, useRef, useState } from 'react';
import { displayLabel } from '../../utils/labels';
import { isSmallImageRescueRow } from '../../utils/smallImageRescue';
import CaptionEditorDialog from './CaptionEditorDialog';
import PromptEditPopover from './PromptEditPopover';
import PexelsAttribution from './PexelsAttribution';

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
    return { border: 'border-2 border-dashed border-gray-500', icon: '👁', cls: 'text-gray-300',
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
  detected: { icon: '🚩', cls: 'text-amber-300', text: 'watermark',
    label: 'Overlaid watermark detected — Clean will crop the border, inpaint a small mark, or flag it for manual review (V2 handles on-subject watermarks)' },
  dismissed: { icon: '⊘', cls: 'text-content-subtle', text: 'not a watermark',
    label: 'You marked this “not a watermark” — future 🧽 Find passes skip it' },
  cleaned: { icon: '✨', cls: 'text-emerald-300', text: 'watermark', label: 'Watermark removed (original kept as a .orig backup)' },
  failed: { icon: '⚠', cls: 'text-red-300', text: 'watermark', label: 'Watermark removal failed' },
};

export default function DatasetGridItem({ img, datasetId, onStatus, onCaption, onCrop, onDelete,
                                          onMirror, mirrorBusy = false, busy = false,
                                          onRegenerate, onView, nonce = 0, faceThresholds,
                                          selected = false, onToggleSelect, tileSize = 'M',
                                          datasetKind = 'character', dualCaptions = false }) {
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
  // that source and silently make an unrelated variation instead.
  const isImageImproveCandidate = img.derivation_kind === 'klein_image_improve';
  const canRegenerate = !isRescueDerived && !isImageImproveCandidate && img.source === 'generated'
    && !(img.status === 'pending' && !img.filename);

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

  return (
    <div tabIndex={0} aria-label={`${displayLabel(img.variation_label) || 'Dataset image'} card`}
      className={`dataset-grid-item rounded-lg ${borderCls} ${selected ? 'ring-2 ring-indigo-400' : ''} bg-app/40 overflow-hidden flex flex-col`}>
      <div className="relative aspect-square bg-black">
        {onToggleSelect && img.filename && (
          <label
            className="dataset-grid-item__actions absolute bottom-1 left-1 z-10 flex items-center justify-center w-6 h-6 rounded bg-black/60 cursor-pointer"
            title="Select for bulk actions"
            onClick={(e) => e.stopPropagation()}>
            <input type="checkbox" checked={selected} disabled={busy}
              onChange={() => onToggleSelect(img.id)}
              aria-label={`Select ${displayLabel(img.variation_label) || 'this image'} for bulk actions`}
              className="w-4 h-4 accent-indigo-500 cursor-pointer" />
          </label>
        )}
        {url ? (
          <button type="button" onClick={() => onView?.(img)}
            title="Inspect (zoom)"
            aria-label={`Inspect ${displayLabel(img.variation_label) || 'the image'} full screen`}
            className="block w-full h-full cursor-zoom-in">
            <img src={url} alt={displayLabel(img.variation_label)} loading="lazy"
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
        <span className="absolute top-1 left-1 px-1.5 py-0.5 rounded text-[10px] bg-black/60 text-white pointer-events-none">
          {img.derivation_kind === 'klein_small_image'
            ? 'Klein rescue'
            : img.derivation_kind === 'small_image_source'
              ? 'rescue original'
              : isImageImproveCandidate
                ? 'Klein improve'
              : img.source === 'import' ? 'real' : 'generated'}{img.framing ? ` · ${img.framing}` : ''}
        </span>
        {fb && (
          <span className={`absolute top-6 left-1 px-1.5 py-0.5 rounded text-[10px] bg-black/70 ${fb.cls} pointer-events-none flex items-center gap-0.5`}
            title={`Resemblance to the reference face — ${fb.label}`}>
            {fb.icon} 🎭 {fb.label}
          </span>
        )}
        {wb && (
          <span className={`absolute bottom-1 right-1 px-1.5 py-0.5 rounded text-[10px] bg-black/70 ${wb.cls} flex items-center gap-0.5`}
            title={(img.watermark_state === 'detected' && WATERMARK_ROUTE_HINT[img.watermark_route]) || wb.label}>
            {wb.icon} {wb.text}
          </span>
        )}
        <div className="dataset-grid-item__actions absolute top-1 right-1 flex max-w-[calc(100%_-_0.5rem)] flex-wrap justify-end gap-1">
          {canRegenerate && (
            <button type="button"
              onClick={(e) => { e.stopPropagation(); onRegenerate?.(img.id); }}
              title="Regenerate this variation (new seed)"
              aria-label="Regenerate this variation (new seed)"
              className="px-1.5 py-0.5 rounded bg-black/60 text-white text-[10px]">🔄</button>
          )}
          {canRegenerate && (
            <button type="button"
              onClick={(e) => { e.stopPropagation(); setEditingPrompt(true); }}
              title="Edit the prompt, then regenerate this variation"
              aria-label="Edit the prompt, then regenerate this variation"
              className="px-1.5 py-0.5 rounded bg-black/60 text-white text-[10px]">✏️</button>
          )}
          {url && onMirror && (
            <button type="button"
              onClick={(e) => { e.stopPropagation(); onMirror(img.id); }}
              disabled={busy || mirrorBusy}
              aria-busy={mirrorBusy}
              aria-label={mirrorBusy
                ? `Mirroring ${displayLabel(img.variation_label) || 'this image'} horizontally`
                : `Mirror ${displayLabel(img.variation_label) || 'this image'} horizontally`}
              title={mirrorBusy ? 'Mirroring horizontally…' : 'Mirror horizontally (flip left and right)'}
              className="grid min-h-7 min-w-7 place-items-center rounded bg-black/60 text-[10px] text-white disabled:cursor-not-allowed disabled:opacity-45">
              <span aria-hidden="true">{mirrorBusy ? '…' : '⇆'}</span>
            </button>
          )}
          {url && (
            <button type="button" onClick={(e) => { e.stopPropagation(); onCrop(img); }}
              title="Crop" aria-label="Crop"
              className="px-1.5 py-0.5 rounded bg-black/60 text-white text-[10px]">✂</button>
          )}
          {!isRescueDerived && (
            <button type="button"
              onClick={(e) => { e.stopPropagation(); if (window.confirm('Permanently delete this image?')) onDelete(img.id); }}
              title="Delete permanently" aria-label="Delete permanently"
              className="px-1.5 py-0.5 rounded bg-red-700/80 text-white text-[10px]">🗑</button>
          )}
        </div>
        {editingPrompt && (
          <PromptEditPopover
            initialPrompt={img.variation_prompt || ''}
            onSubmit={(prompt) => onRegenerate?.(img.id, undefined, prompt)}
            onClose={() => setEditingPrompt(false)} />
        )}
      </div>
      <PexelsAttribution metadata={img.source_metadata}
        className="mx-1.5 mt-1 block text-[0.625rem] leading-tight text-content-subtle" />
      {isRescueDerived ? (
        <p className="m-1.5 rounded border border-indigo-400/30 bg-indigo-500/10 px-2 py-1 text-center text-[0.625rem] text-indigo-200"
          title="This winner was chosen atomically with its provenance pair. Caption and crop remain available.">
          ✓ Chosen in Klein rescue review
        </p>
      ) : (
        <div className="dataset-grid-item__actions flex gap-1 p-1.5">
          <button type="button" onClick={() => onStatus(img.id, img.status === 'keep' ? 'pending' : 'keep')}
            title="Keep" aria-label="Keep" aria-pressed={img.status === 'keep'}
            className={`flex-1 py-1 rounded text-[11px] ${img.status === 'keep' ? 'bg-green-600 text-white' : 'bg-surface text-content-muted'}`}>✓</button>
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
            title="Reject (offers a regeneration)" aria-label="Reject" aria-pressed={img.status === 'reject'}
            className={`flex-1 py-1 rounded text-[11px] ${img.status === 'reject' ? 'bg-red-600 text-white' : 'bg-surface text-content-muted'}`}>✕</button>
        </div>
      )}
      {img.status === 'keep' && (
        <div className="m-1.5 mt-0 flex flex-col gap-1">
          <div className="dataset-grid-item__actions flex items-center justify-end gap-1">
            <button type="button" onClick={() => setCaptionEditorOpen(true)}
              title="Open a larger caption editor"
              aria-label="Expand caption editor"
              className="rounded border border-border bg-surface px-1.5 py-0.5 text-[10px] text-content-muted hover:text-content">
              ⛶ Expand
            </button>
            {cap && (
              <button type="button"
                onClick={() => { editingRef.current = false; setCap(''); onCaption(img.id, ''); }}
                title="Delete this image's caption (then “Caption” regenerates it via JoyCaption)"
                aria-label="Delete this image's caption"
                className="rounded border border-red-500/40 bg-red-500/15 px-1.5 py-0.5 text-[10px] text-red-300 hover:bg-red-500/25">
                🗑 Caption
              </button>
            )}
          </div>
          <textarea value={cap} onChange={(e) => setCap(e.target.value)}
            onFocus={() => { editingRef.current = true; }}
            onBlur={() => {
              editingRef.current = false;
              if (cap !== (img.caption || '')) onCaption(img.id, cap);
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
          initialShortCaption={img.caption_short || ''} showShort={dualCaptions}
          imageLabel={displayLabel(img.variation_label)}
          onClose={() => setCaptionEditorOpen(false)}
          onSave={(nextCaption, nextShort) => {
            editingRef.current = false;
            setCap(nextCaption);
            // Persist when either field changed; `nextShort` is undefined unless dual is on.
            if (nextCaption !== (img.caption || '')
                || (nextShort !== undefined && nextShort !== (img.caption_short || ''))) {
              onCaption(img.id, nextCaption, nextShort);
            }
            setCaptionEditorOpen(false);
          }} />
      )}
    </div>
  );
}
