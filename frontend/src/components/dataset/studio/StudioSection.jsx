// react-frontend/src/components/dataset/studio/StudioSection.jsx
/**
 * StudioSection — section repliable réutilisable du rail de réglages du Studio.
 * Garde l'aside de 320px lisible : chaque groupe de knobs (Format, Sampling,
 * Détail, Engine, Négatif…) est plié/déplié indépendamment et son état est
 * persisté par `storageKey` (localStorage).
 *
 * A11y : l'en-tête est un <button aria-expanded aria-controls> avec un chevron
 * ▼/▶ (indicateur NON-couleur de l'état ouvert/fermé).
 *
 * Props : { title, defaultOpen=true, storageKey, anchorId, children }.
 * `anchorId` (optionnel) : id DOM posé sur la section + écoute de l'événement
 * global `studio:reveal` (émis par la barre de raccourcis du bas) — la section
 * s'OUVRE avant que la vue n'y scrolle, sinon on atterrit sur un en-tête plié.
 */
import { useEffect, useState } from 'react';

export default function StudioSection({ title, defaultOpen = true, storageKey, anchorId, children }) {
  const [open, setOpen] = useState(() => {
    if (!storageKey) return defaultOpen;
    try {
      const v = localStorage.getItem(storageKey);
      return v === null ? defaultOpen : v === 'true';
    } catch {
      return defaultOpen;
    }
  });

  useEffect(() => {
    if (!anchorId) return undefined;
    const onReveal = (e) => { if (e.detail === anchorId) setOpen(true); };
    window.addEventListener('studio:reveal', onReveal);
    return () => window.removeEventListener('studio:reveal', onReveal);
  }, [anchorId]);

  const toggle = () => setOpen((prev) => {
    const next = !prev;
    if (storageKey) { try { localStorage.setItem(storageKey, String(next)); } catch { /* private mode */ } }
    return next;
  });

  // id stable pour aria-controls (chevron/panneau).
  const bodyId = `studio-section-${String(storageKey || title).replace(/\W+/g, '-')}`;

  return (
    <div id={anchorId} className="rounded-lg border border-border bg-surface px-3 py-2 scroll-mt-16">
      <button
        type="button"
        onClick={toggle}
        aria-expanded={open}
        aria-controls={bodyId}
        className="flex w-full items-center justify-between gap-2 text-left"
      >
        <span className="text-content-muted text-[0.625rem] uppercase tracking-wide font-semibold">
          {title}
        </span>
        <span aria-hidden className="text-content-muted text-[0.75rem] leading-none">
          {open ? '▼' : '▶'}
        </span>
      </button>
      {open && (
        <div id={bodyId} className="border-t border-white/10 pt-2 mt-2 flex flex-col gap-2.5">
          {children}
        </div>
      )}
    </div>
  );
}
