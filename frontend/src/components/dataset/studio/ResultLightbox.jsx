// react-frontend/src/components/dataset/studio/ResultLightbox.jsx
/**
 * Aperçu plein écran d'UN résultat + notation 👍/👎 (toggle) + NAVIGATION dans le
 * set courant (feuilleter d'une image à l'autre sans fermer). Extrait 1:1 du bloc
 * `{lbImg && (...)}` de l'ancien LoraTestStudio (behavior-preserving) pour le rating,
 * avec les correctifs a11y déjà actés : focus-trap (useFocusTrap), fermeture sur Échap,
 * vrais boutons avec `aria-pressed` et libellés « Aimé ✓ / Pas fan ✓ » (état non-couleur).
 *
 * Navigation (P- feuilletage) : le parent fournit `items` (le set ORDONNÉ pour la
 * comparaison — voir flipOrder : les variantes de strength d'un même rendu y sont
 * adjacentes) et `onNavigate(nextImg)`. On passe d'une image à l'autre via :
 *   - swipe tactile gauche/droite (même seuil que le vote rapide : 50 px, dominante
 *     horizontale). Swipe droite → précédent, gauche → suivant (convention galerie).
 *   - boutons ‹ › (translucides, pleins au survol/focus sur desktop).
 *   - flèches clavier ← / →.
 * Wrap-around (boucle) : le but étant de feuilleter en boucle pour comparer, on ne
 * bute pas aux extrémités ; un compteur « i / n » situe en permanence.
 *
 * Le rating ne tient AUCUN état local : on appelle `onRate(img.id, nouvelleNote)` et
 * c'est le parent (StudioShell) qui met à jour `img` ensuite.
 */
import { useEffect, useRef } from 'react';
import { useFocusTrap } from '../../../hooks/useFocusTrap';

export default function ResultLightbox({ img, items = [], datasetId, onRate, onNavigate, onClose, fmt }) {
  const ref = useRef(null);
  const touchRef = useRef(null);
  useFocusTrap(ref, !!img);

  const idx = img ? items.findIndex((it) => it.id === img.id) : -1;
  const hasNav = !!onNavigate && idx >= 0 && items.length > 1;

  // Décalage circulaire dans le set courant (wrap-around).
  const go = (delta) => {
    if (!hasNav) return;
    const n = items.length;
    onNavigate(items[(((idx + delta) % n) + n) % n]);
  };

  // Fermeture clavier : Échap (comme DatasetLightbox) + flèches ← / → pour naviguer.
  // Handler auto-contenu (recalcule l'index depuis `items`/`img`) → pas de closure
  // périmée quand on enchaîne les navigations.
  useEffect(() => {
    if (!img) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') { onClose(); return; }
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      if (!onNavigate || items.length < 2) return;
      const i = items.findIndex((it) => it.id === img.id);
      if (i < 0) return;
      e.preventDefault();
      const n = items.length;
      const delta = e.key === 'ArrowLeft' ? -1 : 1;
      onNavigate(items[(((i + delta) % n) + n) % n]);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [img, items, onNavigate, onClose]);

  const onTouchStart = (e) => {
    const t = e.touches && e.touches[0];
    if (t) touchRef.current = { x: t.clientX, y: t.clientY };
  };
  const onTouchEnd = (e) => {
    const st = touchRef.current; touchRef.current = null;
    const t = e.changedTouches && e.changedTouches[0];
    if (!st || !t) return;
    const dx = t.clientX - st.x; const dy = t.clientY - st.y;
    if (Math.abs(dx) < 50 || Math.abs(dx) < Math.abs(dy) * 1.5) return;
    go(dx > 0 ? -1 : 1); // swipe droite = précédent, gauche = suivant
  };

  if (!img) return null;

  const navBtn = 'absolute top-1/2 -translate-y-1/2 w-11 h-11 rounded-full bg-white/10 border border-white/20 text-white text-2xl leading-none z-10 opacity-60 hover:opacity-100 focus-visible:opacity-100 hover:bg-white/20 transition-opacity';

  return (
    <div ref={ref}
      className="fixed inset-0 z-[9998] bg-black/90 backdrop-blur-sm flex items-center justify-center p-4"
      onClick={onClose} role="dialog" aria-modal="true" aria-label="Result preview">
      <button type="button" onClick={onClose} aria-label="Close"
        className="absolute top-4 right-4 w-10 h-10 rounded-full bg-white/10 border border-white/20 text-white text-lg z-10 hover:bg-white/20">×</button>
      {hasNav && (
        <button type="button" aria-label="Previous image"
          onClick={(e) => { e.stopPropagation(); go(-1); }}
          className={`${navBtn} left-2 sm:left-4`}>‹</button>
      )}
      {hasNav && (
        <button type="button" aria-label="Next image"
          onClick={(e) => { e.stopPropagation(); go(1); }}
          className={`${navBtn} right-2 sm:right-4`}>›</button>
      )}
      <div className="flex flex-col items-center gap-2"
        onClick={(e) => e.stopPropagation()}
        onTouchStart={onTouchStart} onTouchEnd={onTouchEnd}>
        <img src={`/api/dataset/${datasetId}/img/${encodeURIComponent(img.filename)}`}
          alt={img.label} className="max-w-[92vw] max-h-[80vh] object-contain rounded-lg border border-white/15" />
        <div className="text-content-muted text-xs tabular-nums text-center">
          {hasNav && <span className="text-content-subtle">{idx + 1} / {items.length} · </span>}
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
