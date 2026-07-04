import { useRef } from 'react';

// Cap identique à MAX_EXTRA_REFS côté backend (face_dataset_service).
const MAX_EXTRA_REFS = 3;

export default function ReferencePanel({ refFilename, datasetId, onSetRef, onCropRef, busy, nonce = 0,
                                         extraRefs = [], onAddExtraRef, onRemoveExtraRef }) {
  const inp = useRef(null);
  const inpExtra = useRef(null);
  const imgUrl = (fn) => `/api/dataset/${datasetId}/img/${encodeURIComponent(fn)}${nonce ? `?v=${nonce}` : ''}`;
  return (
    <div className="flex flex-col gap-2 rounded-lg border border-border bg-surface p-3">
      <div className="flex items-center gap-3">
        <div className="w-20 h-20 rounded-lg bg-black overflow-hidden shrink-0 flex items-center justify-center">
          {refFilename
            ? <img src={imgUrl(refFilename)} alt="ref" className="w-full h-full object-cover" />
            : <span className="text-content-subtle text-xs">none</span>}
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-content text-sm font-medium">Reference photo</span>
          <span className="text-content-subtle text-[0.6875rem]">source of Klein variations (auto head-crop)</span>
          <div className="flex gap-1.5">
            <button type="button" onClick={() => inp.current?.click()} disabled={busy}
              className="px-2.5 py-1 rounded-lg bg-surface-raised text-content text-xs disabled:opacity-40">
              {refFilename ? 'Change' : 'Set'} reference
            </button>
            {refFilename && (
              <button type="button" onClick={onCropRef} disabled={busy}
                className="px-2.5 py-1 rounded-lg bg-surface-raised text-content text-xs disabled:opacity-40">✂ Crop</button>
            )}
          </div>
          <input ref={inp} type="file" accept="image/*" className="hidden"
            onChange={(e) => { if (e.target.files[0]) onSetRef(e.target.files[0]); e.target.value = ''; }} />
        </div>
      </div>

      {/* Références additionnelles — envoyées EN PLUS à Nano Banana (identité
          multi-angles) ; Klein/crop/scoring restent sur la principale. */}
      {refFilename && (
        <div className="flex items-center gap-2 flex-wrap border-t border-border pt-2">
          <span className="text-content-subtle text-[0.6875rem]">
            Extra refs <span className="opacity-70">(Nano Banana only)</span>
          </span>
          {extraRefs.map((fn) => (
            <div key={fn} className="relative w-12 h-12 rounded-lg overflow-hidden bg-black shrink-0">
              <img src={imgUrl(fn)} alt="extra reference" className="w-full h-full object-cover" />
              <button type="button" onClick={() => onRemoveExtraRef?.(fn)} disabled={busy}
                aria-label="Remove this extra reference"
                title="Remove this extra reference"
                className="absolute top-0 right-0 w-4 h-4 flex items-center justify-center rounded-bl bg-black/70 text-white text-[0.625rem] leading-none disabled:opacity-40">
                ✕
              </button>
            </div>
          ))}
          {extraRefs.length < MAX_EXTRA_REFS && (
            <button type="button" onClick={() => inpExtra.current?.click()} disabled={busy}
              aria-label="Add an extra reference photo (used by Nano Banana only)"
              title="Add an extra reference photo — Nano Banana uses all of them for identity consistency"
              className="w-12 h-12 rounded-lg border border-dashed border-border-strong text-content-muted text-lg leading-none disabled:opacity-40">
              +
            </button>
          )}
          <input ref={inpExtra} type="file" accept="image/*" className="hidden"
            onChange={(e) => { if (e.target.files[0]) onAddExtraRef?.(e.target.files[0]); e.target.value = ''; }} />
        </div>
      )}
    </div>
  );
}
