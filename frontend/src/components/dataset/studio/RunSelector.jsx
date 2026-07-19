// react-frontend/src/components/dataset/studio/RunSelector.jsx
/**
 * En-tête de la zone résultats : toggle « 📊 Résultats » (repli), bouton
 * « 🗳 Voter (N) » (file de vote rapide) et sélecteur du run à afficher.
 * Extraction behavior-preserving du bloc d'en-tête de l'ancien LoraTestStudio.jsx.
 */
import { useI18n } from '../../../i18n/I18nContext';

export default function RunSelector({
  runs,
  activeRunKey,
  onSelect,
  unvotedCount,
  onStartVote,
  greenCount,
  onStartReVote,
  displayedCount,
  showResults,
  onToggleResults,
  canExport,
  onExport,
}) {
  const { t } = useI18n();
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <button type="button" onClick={onToggleResults} aria-expanded={showResults}
        className="flex items-center gap-1.5 text-left text-content-muted text-[0.625rem] uppercase">
        <span aria-hidden>📊</span> {t('studio.results.title')}
        <span className="text-content-subtle normal-case">{t('studio.results.imageCount', { count: displayedCount })}</span>
        <span aria-hidden>{showResults ? '▾' : '▸'}</span>
      </button>
      {unvotedCount > 0 && (
        <button type="button" onClick={onStartVote}
          title={t('studio.results.voteTitle')}
          className="px-2.5 py-1 rounded-lg bg-gradient-primary text-white text-[0.6875rem] font-semibold">
          🗳 {t('studio.results.vote', { count: unvotedCount })}
        </button>
      )}
      {greenCount > 0 && (
        <button type="button" onClick={onStartReVote}
          title={t('studio.results.revoteTitle')}
          className="px-2.5 py-1 rounded-lg border border-green-400/60 bg-green-500/15 text-green-200 text-[0.6875rem] font-semibold">
          ♻️ {t('studio.results.revote', { count: greenCount })}
        </button>
      )}
      <button type="button" onClick={onExport} disabled={!canExport}
        title={canExport ? t('studio.results.exportTitle') : t('studio.results.exportUnavailable')}
        className="px-2.5 py-1 rounded-lg border border-border bg-surface text-content-muted text-[0.6875rem] font-semibold hover:text-content disabled:opacity-40 disabled:cursor-not-allowed">
        🖼 {t('studio.results.export')}
      </button>
      {runs.length > 1 && (
        <select value={activeRunKey || ''} onChange={(e) => onSelect(e.target.value)}
          aria-label={t('studio.results.chooseRun')}
          className="ml-auto rounded-lg border border-border bg-surface px-2 py-1 text-[0.6875rem] text-content max-w-[280px]">
          {runs.map((r, i) => {
            // Taux de 👍 parmi les votes du run (likes / votés), comme le « % 👍 »
            // affiché par cellule. Caché si aucun vote (division par zéro).
            const voted = r.likes + r.dislikes;
            const pct = voted ? Math.round((r.likes / voted) * 100) : null;
            return (
              <option key={r.key} value={r.key}>
                {i === 0 ? `● ${t('studio.results.currentRun')}` : t('studio.results.runNumber', { number: runs.length - i })} — {r.modelLabel || '?'} · 👍{r.likes} 👎{r.dislikes}{pct !== null ? ` · ${pct}% 👍` : ''} · «{(r.prompt || '').slice(0, 22)}»
              </option>
            );
          })}
        </select>
      )}
    </div>
  );
}
