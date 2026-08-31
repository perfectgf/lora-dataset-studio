// react-frontend/src/components/dataset/studio/StudioActionBar.jsx
/**
 * StudioActionBar — barre de commande FIXE en bas du Test Studio.
 *
 * Deux rôles (demande user 2026-07-03) :
 *   1. le bouton « Run the test » reste TOUJOURS visible (doublon assumé du
 *      bouton du rail de setup) ;
 *   2. des raccourcis qui amènent la vue directement sur chaque groupe
 *      d'options (LoRAs, Prompt & seed, Format, Sampling, Engine, Results…).
 *
 * Un raccourci émet d'abord `studio:reveal` (une StudioSection pliée s'OUVRE,
 * cf StudioSection.anchorId) puis scrolle sur l'ancre — scrollIntoView remonte
 * aussi l'aside interne (overflow-auto en desktop). Le FAB GlobalJobsDock est
 * relevé au-dessus via PAGES_WITH_BOTTOM_BAR ('/studio').
 *
 * `note` (optionnel, 2026-08-31) : la raison pour laquelle le bouton est gris,
 * affichée à sa gauche — la lane vidéo l'utilise (« Pick a start frame… ») ;
 * sans la prop, la barre est byte-identique à ce qu'elle a toujours été.
 */

export default function StudioActionBar({ shortcuts = [], canRun, running, onRun, runLabel = '🚀 Run the test', note = null }) {
  const jump = (id) => {
    try { window.dispatchEvent(new CustomEvent('studio:reveal', { detail: id })); } catch { /* ignore */ }
    // Laisse la section s'ouvrir (setState) avant de scroller vers elle.
    requestAnimationFrame(() => {
      document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  };
  return (
    <nav aria-label="Studio quick navigation" data-probe-chrome="action-bar"
      className="fixed bottom-0 left-0 right-0 z-[9960] border-t border-border bg-app/90 backdrop-blur-md">
      <div className="flex items-center gap-1.5 px-3 sm:px-5 py-2 overflow-x-auto">
        {shortcuts.map((s) => (
          <button key={s.id} type="button" onClick={() => jump(s.id)}
            // min-h-10 below lg: a 27-px chip is under the ~40 px a fingertip lands on,
            // and a miss goes to whatever sits behind it (the results grid). Measured by
            // the responsive probe; compact again from lg, where a pointer is precise.
            className="min-h-10 lg:min-h-0 shrink-0 px-2.5 py-1 rounded-full border border-border bg-surface text-content-muted hover:text-content hover:bg-surface-raised text-[0.6875rem] font-medium transition-colors">
            <span aria-hidden="true">{s.emoji}</span> {s.label}
          </button>
        ))}
        {note && (
          <span className="ml-auto min-w-0 shrink truncate text-[0.6875rem] text-content-subtle" title={note}>
            {note}
          </span>
        )}
        <button type="button" onClick={onRun} disabled={!canRun}
          className={`min-h-10 lg:min-h-0 ${note ? '' : 'ml-auto'} shrink-0 px-4 py-1.5 rounded-lg bg-gradient-primary text-gray-950 text-sm font-semibold disabled:opacity-40`}>
          {running ? '…' : runLabel}
        </button>
      </div>
    </nav>
  );
}
