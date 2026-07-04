/** One curation tile: image + keep/reject + source/framing badges + caption + crop. */
import { useEffect, useRef, useState } from 'react';

const STATUS_CLS = {
  keep: 'border-green-500',
  reject: 'border-red-500/50 opacity-50',
  pending: 'border-amber-400/40',
  failed: 'border-red-600',
};

// Seuils calibres antelopev2 (test3) — face_score brut persiste -> ajustables ici.
const FACE_VALID = 0.50, FACE_ORANGE = 0.45;
const GREY_LABEL = { no_face: 'no face detected', low_det: 'low detection',
  too_small: 'face too small', extreme_pose: 'profile — not scored',
  unreadable: 'unreadable', error: 'error' };

// Retourne {border, icon, cls, label} d'apres face_state/face_score, ou null si pas analysé.
// La bordure encode la largeur ET le style (plein=jugé / pointillé=non-jugeable) pour
// ne PAS dépendre de la couleur seule (WCAG 1.4.1).
function faceBadge(img) {
  if (img.face_state == null) return null;
  if (img.face_state !== 'scorable' || img.face_score == null) {
    return { border: 'border-2 border-dashed border-gray-500', icon: '👁', cls: 'text-gray-300',
      label: GREY_LABEL[img.face_state] || 'not scored' };
  }
  const s = img.face_score;
  if (s >= FACE_VALID) return { border: 'border-2 border-green-500', icon: '✓', cls: 'text-green-300', label: s.toFixed(2) };
  if (s >= FACE_ORANGE) return { border: 'border-2 border-amber-500', icon: '~', cls: 'text-amber-300', label: `${s.toFixed(2)} to review` };
  return { border: 'border-4 border-red-500', icon: '⚠', cls: 'text-red-300', label: `${s.toFixed(2)} low` };
}

export default function DatasetGridItem({ img, datasetId, onStatus, onCaption, onCrop, onDelete,
                                          onRegenerate, onView, nonce = 0 }) {
  const [cap, setCap] = useState(img.caption || '');
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
  const canRegenerate = img.source === 'generated' && !(img.status === 'pending' && !img.filename);

  const fb = faceBadge(img);
  const borderCls = fb ? fb.border : `border-2 ${STATUS_CLS[img.status] || 'border-border'}`;

  return (
    <div className={`rounded-lg ${borderCls} bg-app/40 overflow-hidden flex flex-col`}>
      <div className="relative aspect-square bg-black">
        {url ? (
          <button type="button" onClick={() => onView?.(img)}
            title="Inspect (zoom)"
            aria-label={`Inspect ${img.variation_label || 'the image'} full screen`}
            className="block w-full h-full cursor-zoom-in">
            <img src={url} alt={img.variation_label || ''} loading="lazy"
              className="w-full h-full object-cover" />
          </button>
        ) : (
          <div className="w-full h-full flex items-center justify-center text-content-subtle text-xs">
            {img.status === 'failed' ? 'failed' : '…'}
          </div>
        )}
        <span className="absolute top-1 left-1 px-1.5 py-0.5 rounded text-[10px] bg-black/60 text-white pointer-events-none">
          {img.source === 'import' ? 'real' : 'generated'}{img.framing ? ` · ${img.framing}` : ''}
        </span>
        {fb && (
          <span className={`absolute top-6 left-1 px-1.5 py-0.5 rounded text-[10px] bg-black/70 ${fb.cls} pointer-events-none flex items-center gap-0.5`}
            title={`Resemblance to the reference face — ${fb.label}`}>
            {fb.icon} 🎭 {fb.label}
          </span>
        )}
        <div className="absolute top-1 right-1 flex gap-1">
          {canRegenerate && (
            <button type="button"
              onClick={(e) => { e.stopPropagation(); onRegenerate?.(img.id); }}
              title="Regenerate this variation (new seed)"
              aria-label="Regenerate this variation (new seed)"
              className="px-1.5 py-0.5 rounded bg-black/60 text-white text-[10px]">🔄</button>
          )}
          {url && (
            <button type="button" onClick={(e) => { e.stopPropagation(); onCrop(img); }}
              title="Crop" aria-label="Crop"
              className="px-1.5 py-0.5 rounded bg-black/60 text-white text-[10px]">✂</button>
          )}
          <button type="button"
            onClick={(e) => { e.stopPropagation(); if (window.confirm('Permanently delete this image?')) onDelete(img.id); }}
            title="Delete permanently" aria-label="Delete permanently"
            className="px-1.5 py-0.5 rounded bg-red-700/80 text-white text-[10px]">🗑</button>
        </div>
      </div>
      <div className="flex gap-1 p-1.5">
        <button type="button" onClick={() => onStatus(img.id, img.status === 'keep' ? 'pending' : 'keep')}
          title="Keep" aria-label="Keep" aria-pressed={img.status === 'keep'}
          className={`flex-1 py-1 rounded text-[11px] ${img.status === 'keep' ? 'bg-green-600 text-white' : 'bg-surface text-content-muted'}`}>✓</button>
        <button type="button"
          onClick={() => {
            // Rejecting a GENERATED image offers an immediate retry of the same
            // variation (in place, new seed) so the composition stays on target.
            if (img.status !== 'reject' && img.source === 'generated' && img.filename && onRegenerate
                && window.confirm('Photo rejected — regenerate a new attempt of this variation?\n(OK = replace with a new attempt · Cancel = reject only)')) {
              onRegenerate(img.id);
              return;
            }
            onStatus(img.id, img.status === 'reject' ? 'pending' : 'reject');
          }}
          title="Reject (offers a regeneration)" aria-label="Reject" aria-pressed={img.status === 'reject'}
          className={`flex-1 py-1 rounded text-[11px] ${img.status === 'reject' ? 'bg-red-600 text-white' : 'bg-surface text-content-muted'}`}>✕</button>
      </div>
      {img.status === 'keep' && (
        <div className="m-1.5 mt-0 flex flex-col gap-1">
          {cap && (
            <button type="button"
              onClick={() => { editingRef.current = false; setCap(''); onCaption(img.id, ''); }}
              title="Delete this image's caption (then “Caption” regenerates it via JoyCaption)"
              aria-label="Delete this image's caption"
              className="self-end px-1.5 py-0.5 rounded bg-red-500/15 border border-red-500/40 text-red-300 text-[10px] hover:bg-red-500/25">
              🗑 Caption
            </button>
          )}
          <textarea value={cap} onChange={(e) => setCap(e.target.value)}
            onFocus={() => { editingRef.current = true; }}
            onBlur={() => {
              editingRef.current = false;
              if (cap !== (img.caption || '')) onCaption(img.id, cap);
            }}
            rows={2} placeholder="caption (without the face)…" aria-label="Image caption"
            className="text-[11px] bg-app/60 border border-border rounded p-1 text-content resize-none" />
        </div>
      )}
    </div>
  );
}
