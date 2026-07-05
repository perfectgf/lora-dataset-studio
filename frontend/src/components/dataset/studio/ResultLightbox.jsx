// react-frontend/src/components/dataset/studio/ResultLightbox.jsx
/**
 * Aperçu plein écran d'UN résultat + notation 👍/👎 (toggle). Extrait 1:1 du bloc
 * `{lbImg && (...)}` de l'ancien LoraTestStudio (behavior-preserving), avec les
 * correctifs a11y déjà actés : focus-trap (useFocusTrap), fermeture sur Échap, vrais
 * boutons avec `aria-pressed` et libellés « Aimé ✓ / Pas fan ✓ » (état non-couleur).
 *
 * Le rating ne tient AUCUN état local : on appelle `onRate(img.id, nouvelleNote)` et
 * c'est le parent (StudioShell) qui met à jour `img` ensuite.
 */
import { useEffect, useRef } from 'react';
import { useFocusTrap } from '../../../hooks/useFocusTrap';

export default function ResultLightbox({ img, datasetId, onRate, onClose, fmt }) {
  const ref = useRef(null);
  useFocusTrap(ref, !!img);

  // Fermeture clavier : Échap (comme DatasetLightbox).
  useEffect(() => {
    if (!img) return undefined;
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [img, onClose]);

  if (!img) return null;

  return (
    <div ref={ref}
      className="fixed inset-0 z-[9998] bg-black/90 backdrop-blur-sm flex items-center justify-center p-4"
      onClick={onClose} role="dialog" aria-modal="true" aria-label="Result preview">
      <button type="button" onClick={onClose} aria-label="Close"
        className="absolute top-4 right-4 w-10 h-10 rounded-full bg-white/10 border border-white/20 text-white text-lg z-10 hover:bg-white/20">×</button>
      <div className="flex flex-col items-center gap-2" onClick={(e) => e.stopPropagation()}>
        <img src={`/api/dataset/${datasetId}/img/${encodeURIComponent(img.filename)}`}
          alt={img.label} className="max-w-[92vw] max-h-[80vh] object-contain rounded-lg border border-white/15" />
        <div className="text-content-muted text-xs tabular-nums text-center">
          {img.label} · strength {fmt(img.strength)}{img.aspect ? ` · ${img.aspect}` : ''}
          {img.seed != null ? ` · seed ${img.seed}` : ''}
        </div>
        <div className="flex items-center gap-2">
          <button type="button" aria-pressed={img.rating === 1}
            onClick={() => onRate(img.id, img.rating === 1 ? 0 : 1)}
            className={`px-3 py-1 rounded-lg text-sm border ${img.rating === 1 ? 'border-green-400/60 bg-green-500/20 text-green-200' : 'border-border bg-surface text-content'}`}>
            👍 {img.rating === 1 ? 'Liked ✓' : 'Like'}
          </button>
          <button type="button" aria-pressed={img.rating === -1}
            onClick={() => onRate(img.id, img.rating === -1 ? 0 : -1)}
            className={`px-3 py-1 rounded-lg text-sm border ${img.rating === -1 ? 'border-red-400/60 bg-red-500/20 text-red-200' : 'border-border bg-surface text-content'}`}>
            👎 {img.rating === -1 ? 'Not a fan ✓' : 'Not a fan'}
          </button>
        </div>
      </div>
    </div>
  );
}
