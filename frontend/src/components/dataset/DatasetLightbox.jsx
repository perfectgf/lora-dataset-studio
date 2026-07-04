/**
 * Full-screen inspection lightbox (F3): toggle fit ↔ 100 % (native pixels) to
 * hunt skin/eyes artefacts before keeping an image. Esc, ✕ or a click on the
 * backdrop close it; a click on the image toggles the zoom mode.
 */
import { useEffect, useRef, useState } from 'react';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import { displayLabel } from '../../utils/labels';

export default function DatasetLightbox({ img, datasetId, nonce = 0, onClose }) {
  const [full, setFull] = useState(false); // false = fit screen, true = 100 %
  const dialogRef = useRef(null);
  const closeRef = useRef(null);

  // Focus trap keeps Tab inside the dialog (P2-7).
  useFocusTrap(dialogRef, !!(img && img.filename));

  // Keyboard support: Escape closes, initial focus on the close button.
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);
  useEffect(() => { closeRef.current?.focus(); }, []);

  if (!img || !img.filename) return null;
  const url = `/api/dataset/${datasetId}/img/${encodeURIComponent(img.filename)}${nonce ? `?v=${nonce}` : ''}`;
  const alt = displayLabel(img.variation_label) || 'dataset image';

  return (
    <div ref={dialogRef} role="dialog" aria-modal="true" aria-label={`Inspect — ${alt}`}
      className="fixed inset-0 z-[9996] bg-black/95 flex flex-col" onClick={onClose}>
      <button type="button" ref={closeRef}
        onClick={(e) => { e.stopPropagation(); onClose(); }}
        title="Close (Esc)" aria-label="Close inspection"
        className="absolute top-3 right-3 z-10 w-9 h-9 rounded-full bg-white/10 hover:bg-white/20 text-white text-lg leading-none">✕</button>

      {full ? (
        <div className="flex-1 min-h-0 overflow-auto">
          <img src={url} alt={alt}
            onClick={(e) => { e.stopPropagation(); setFull(false); }}
            className="max-w-none cursor-zoom-out select-none" />
        </div>
      ) : (
        <div className="flex-1 min-h-0 flex items-center justify-center p-4">
          <img src={url} alt={alt}
            onClick={(e) => { e.stopPropagation(); setFull(true); }}
            className="max-h-full max-w-full object-contain cursor-zoom-in select-none" />
        </div>
      )}

      <div onClick={(e) => e.stopPropagation()}
        className="shrink-0 flex flex-wrap items-center justify-center gap-2 px-4 py-2.5 bg-black/60">
        <span className="text-white text-sm">{alt}</span>
        <span className="px-1.5 py-0.5 rounded text-[10px] bg-white/10 text-white/80">
          {img.source === 'import' ? 'real' : 'generated'}{img.framing ? ` · ${img.framing}` : ''}
        </span>
        <span className="text-white/50 text-[11px]">
          {full ? '100 % — click image to fit' : 'fitted — click image for 100 %'}
        </span>
      </div>
    </div>
  );
}
