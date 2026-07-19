// react-frontend/src/components/dataset/studio/FamilySelector.jsx
/**
 * Sélecteur de FAMILLE (pipeline) du studio de test. Un même dataset peut avoir été
 * entraîné sous plusieurs pipelines (ZIT / SDXL / Krea), chacune avec ses propres
 * checkpoints (dossiers loras/<famille>). On n'affiche QUE les familles réellement
 * présentes (`payload.available_families`). La famille choisie re-scope tout le studio
 * (pool de checkpoints, base, dimensions, workflow, meilleur réglage mémorisé).
 */
import { FAMILY_LABELS } from './constants';
import { useI18n } from '../../../i18n/I18nContext';

export default function FamilySelector({ families = [], active, onSelect }) {
  const { t } = useI18n();
  if (!families || families.length < 2) return null;  // 0/1 famille → aucun choix à offrir
  return (
    <div className="flex items-center gap-2 flex-wrap" role="group" aria-label={t('studio.family.pipeline')}>
      <span className="text-content-muted text-[0.6875rem] uppercase tracking-wide">{t('studio.family.trainedIn')}</span>
      {families.map((f) => {
        const on = f.family === active;
        return (
          <button
            key={f.family}
            type="button"
            onClick={() => onSelect?.(f.family)}
            aria-pressed={on}
            title={t('studio.family.testTitle', {
              family: FAMILY_LABELS[f.family] || f.family,
              count: f.count,
            })}
            className={`px-2.5 py-1 rounded-lg border text-[0.75rem] leading-none transition-colors ${
              on ? 'border-amber-400/60 bg-amber-400/15 text-amber-200 font-semibold'
                 : 'border-border bg-surface text-content-muted hover:text-content'}`}
          >
            {FAMILY_LABELS[f.family] || f.label || f.family}
            <span className="ml-1 text-content-subtle tabular-nums">{f.count}</span>
          </button>
        );
      })}
    </div>
  );
}
