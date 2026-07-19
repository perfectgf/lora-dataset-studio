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
 */

export default function StudioActionBar({ shortcuts = [], canRun, running, onRun, runLabel }) {
  const { t } = useI18n();
  const jump = (id) => {
    try { window.dispatchEvent(new CustomEvent('studio:reveal', { detail: id })); } catch { /* ignore */ }
    // Laisse la section s'ouvrir (setState) avant de scroller vers elle.
    requestAnimationFrame(() => {
      document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  };
  return (
    <nav aria-label={t('studio.actions.quickNavigation')}
      className="fixed bottom-0 left-0 right-0 z-[9960] border-t border-border bg-app/90 backdrop-blur-md">
      <div className="flex items-center gap-1.5 px-3 sm:px-5 py-2 overflow-x-auto">
        {shortcuts.map((s) => (
          <button key={s.id} type="button" onClick={() => jump(s.id)}
            className="shrink-0 px-2.5 py-1 rounded-full border border-border bg-surface text-content-muted hover:text-content hover:bg-surface-raised text-[0.6875rem] font-medium transition-colors">
            <span aria-hidden="true">{s.emoji}</span> {s.label}
          </button>
        ))}
        <button type="button" onClick={onRun} disabled={!canRun}
          className="ml-auto shrink-0 px-4 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
          {running ? '…' : (runLabel || `🚀 ${t('studio.actions.runTest')}`)}
        </button>
      </div>
    </nav>
  );
}
import { useI18n } from '../../../i18n/I18nContext';
