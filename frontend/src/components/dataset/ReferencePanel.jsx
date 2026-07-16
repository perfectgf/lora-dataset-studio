import { useRef, useState } from 'react';

// Cap identique à MAX_EXTRA_REFS côté backend (face_dataset_service).
const MAX_EXTRA_REFS = 3;

export default function ReferencePanel({ refFilename, datasetId, onSetRef, onCropRef, busy,
                                         importBusy = busy, visionBusy = false, nonce = 0,
                                         extraRefs = [], onAddExtraRef, onRemoveExtraRef }) {
  const inp = useRef(null);
  const inpExtra = useRef(null);
  // Auto head-crop = OPT-IN (vision pass, pauses ComfyUI). Default OFF: upload is
  // instant (centered square) and ✂ Crop adjusts manually — faster in practice.
  const [autoCrop, setAutoCrop] = useState(false);
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
          <span className="text-content-subtle text-[0.6875rem]">source of Klein variations — crop with ✂ after upload</span>
          <div className="flex gap-1.5 items-center flex-wrap">
            <button type="button" onClick={() => inp.current?.click()} disabled={importBusy}
              className="px-2.5 py-1 rounded-lg bg-surface-raised text-content text-xs disabled:opacity-40">
              {refFilename ? 'Change' : 'Set'} reference
            </button>
            {refFilename && (
              <button type="button" onClick={onCropRef} disabled={busy}
                className="px-2.5 py-1 rounded-lg bg-surface-raised text-content text-xs disabled:opacity-40">✂ Crop</button>
            )}
            <label className="flex items-center gap-1 text-[0.625rem] text-content-muted cursor-pointer"
              title={visionBusy ? 'Auto head-crop is unavailable during local generation; the reference imports with a centered crop.' : 'ON: a vision pass finds the head and crops around it (slower, pauses ComfyUI). OFF (default): instant centered square — adjust with ✂ Crop, usually faster.'}>
              <input type="checkbox" checked={autoCrop} disabled={visionBusy} onChange={(e) => setAutoCrop(e.target.checked)}
                className="accent-indigo-500 w-3 h-3" />
              ✂ Auto head-crop{visionBusy ? ' — unavailable during local generation' : ''}
            </label>
          </div>
          <input ref={inp} type="file" accept="image/*" className="hidden" disabled={importBusy}
            onChange={(e) => { if (e.target.files[0]) onSetRef(e.target.files[0], { autoCrop: autoCrop && !visionBusy }); e.target.value = ''; }} />
        </div>
      </div>

      {/* Références additionnelles — identité multi-angles, consommées par TOUS
          les moteurs : Nano Banana & ChatGPT (jointes à l'appel API) et Klein
          (chaînées en ReferenceLatent natifs). Crop/scoring restent sur la
          principale. */}
      {refFilename && (
        <div className="flex items-center gap-2 flex-wrap border-t border-border pt-2">
          <span className="text-content-subtle text-[0.6875rem]">
            Extra refs <span className="opacity-70">(all engines — stronger identity lock)</span>
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
            <button type="button" onClick={() => inpExtra.current?.click()} disabled={importBusy}
              aria-label="Add an extra reference photo (other angles of the same face)"
              title="Add an extra reference photo — every engine (Nano Banana, ChatGPT, Klein) uses them together to lock the identity"
              className="w-12 h-12 rounded-lg border border-dashed border-border-strong text-content-muted text-lg leading-none disabled:opacity-40">
              +
            </button>
          )}
          <input ref={inpExtra} type="file" accept="image/*" className="hidden" disabled={importBusy}
            onChange={(e) => { if (e.target.files[0]) onAddExtraRef?.(e.target.files[0]); e.target.value = ''; }} />
        </div>
      )}
    </div>
  );
}
