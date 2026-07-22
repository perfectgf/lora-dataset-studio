/**
 * Full-screen inspection lightbox (F3): toggle fit ↔ 100 % (native pixels) to
 * hunt skin/eyes artefacts before keeping an image. Esc, ✕ or a click on the
 * backdrop close it; a click on the image toggles the zoom mode.
 */
import { useEffect, useRef, useState } from 'react';
import SettingsLink from '../common/SettingsLink';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import { displayLabel } from '../../utils/labels';
import PexelsAttribution from './PexelsAttribution';

const IMPROVE_HELP = 'Klein creates a new 2 MP version to validate and leaves the original intact.';

export default function DatasetLightbox({
  img,
  datasetId,
  nonce = 0,
  onClose,
  onCrop,
  onMirror,
  onImprove,
  busy = false,
  mirrorBusy = false,
  improvePending = false,
  improveReady = false,
  kleinAvailable = false,
}) {
  const [full, setFull] = useState(false); // false = fit screen, true = 100 %
  const [improving, setImproving] = useState(false);
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
  const improvementActive = improving || improvePending;
  const improveDisabled = busy || improvementActive || improveReady || !kleinAvailable;
  const improveTitle = !kleinAvailable
    ? `Klein is not available in this setup. ${IMPROVE_HELP}`
    : improveReady
      ? `A new version is waiting for validation. ${IMPROVE_HELP}`
    : improvePending
      ? `An improvement is already running for this image. ${IMPROVE_HELP}`
      : IMPROVE_HELP;

  const improve = async (event) => {
    event.stopPropagation();
    if (!onImprove || improveDisabled) return;
    setImproving(true);
    try {
      await onImprove(img.id);
    } finally {
      setImproving(false);
    }
  };

  const mirror = async (event) => {
    event.stopPropagation();
    if (!onMirror || busy || mirrorBusy) return;
    await onMirror(img.id);
  };

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
        <PexelsAttribution metadata={img.source_metadata}
          className="text-[11px] text-white/70" />
        <span className="text-white/50 text-[11px]">
          {full ? '100 % — click image to fit' : 'fitted — click image for 100 %'}
        </span>
        {onCrop && (
          <button type="button" onClick={() => onCrop(img)}
            title="Open the crop editor for this image (stretchable box, any ratio)"
            className="px-3 py-1 rounded-lg bg-white/10 hover:bg-white/20 text-white text-xs font-semibold">
            ✂ Crop
          </button>
        )}
        {onMirror && (
          <button type="button" onClick={mirror} disabled={busy || mirrorBusy}
            aria-busy={mirrorBusy}
            aria-label={mirrorBusy ? `Mirroring ${alt} horizontally` : `Mirror ${alt} horizontally`}
            title={mirrorBusy ? 'Mirroring horizontally…' : 'Mirror horizontally (flip left and right)'}
            className="min-h-9 w-full sm:w-auto px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-white text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-45">
            {mirrorBusy ? '⇆ Mirroring…' : '⇆ Mirror horizontally'}
          </button>
        )}
        {onImprove && (
          <button type="button" onClick={improve} disabled={improveDisabled}
            aria-busy={improvementActive} title={improveTitle}
            className="min-h-9 w-full sm:w-auto px-3 py-1.5 rounded-lg border border-indigo-400/50 bg-indigo-500/20 hover:bg-indigo-500/30 text-indigo-100 text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-45">
            {improveReady
              ? '✓ Review improvement first'
              : improvementActive ? '✨ Improving…' : '✨ Upscale & improve'}
          </button>
        )}
        {/* Its strength, step count and instruction are all editable, and nothing
            here said so — the reported case for making settings discoverable from
            where the action happens. */}
        {onImprove && !improvementActive && (
          <SettingsLink section="engines" className="self-center">
            Adjust improve strength
          </SettingsLink>
        )}
      </div>
    </div>
  );
}
