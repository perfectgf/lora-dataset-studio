// react-frontend/src/components/dataset/studio/StudioShell.jsx
/**
 * Coquille du Studio de test : LoraPicker partagé en haut, puis bascule selon le
 * nombre de LoRA cochés (spec validée « 1 LoRA → réglage comme aujourd'hui ;
 * ≥2 LoRA → comparaison ») :
 *
 *   - ≥2 LoRA  → <ComparisonStudio> : run_id + grille colonnes=LoRA × lignes=strength
 *                + « 🏆 Classement LoRA ».
 *   - 1 LoRA   → <LegacyDatasetStudio datasetId={…}> : le studio RICHE d'origine
 *                (RunSetupPanel, ResultsArea, BestPerModelList, ModelComparison,
 *                best_settings/★ Appliquer→generate, presets, stats par checkpoint).
 *   - 0 LoRA   → même studio legacy si un dataset est pré-sélectionné (URL), sinon
 *                une invite à cocher un LoRA.
 *
 * Le picker reste visible dans tous les modes → ajouter un 2e LoRA bascule en
 * comparaison, en retirer un revient au studio riche. Chaque branche est un
 * composant distinct : ses hooks (useLoraTestStudio vs useStudioRun) sont appelés
 * inconditionnellement dans son propre sous-arbre (règle des hooks respectée), et
 * remonter/démonter au changement de branche réinitialise proprement son état.
 *
 * Rétrocompat : la route legacy /dataset/studio/:id fournit `preselectDataset` →
 * 0 LoRA coché initialement mais dataset pré-sélectionné → branche legacy → studio
 * riche identique à avant (et le LoRA est pré-coché dans le picker).
 */
import { useCallback, useEffect, useState } from 'react';
import { HelpBadge } from '../../../help/HelpMode';
import LoraPicker from './LoraPicker';
import LegacyDatasetStudio from './LegacyDatasetStudio';
import ComparisonStudio from './ComparisonStudio';
import { useI18n } from '../../../i18n/I18nContext';

export default function StudioShell({ preselectDataset = null, datasetId = null }) {
  const { t } = useI18n();
  // `datasetId` legacy est un alias de preselectDataset.
  const preselect = preselectDataset ?? datasetId;

  const [selection, setSelection] = useState([]);
  const onSelectionChange = useCallback((sel) => setSelection(sel), []);

  // train_type du run = celui du 1er LoRA coché (null si rien coché).
  const runType = selection.length > 0 ? (selection[0].train_type || 'zimage') : null;

  // Liste des bases correspondant au train_type courant.
  // Fetch à chaque changement de runType via /api/studio/base-models?type=…
  const [baseModels, setBaseModels] = useState([]);
  useEffect(() => {
    if (!runType) { setBaseModels([]); return; }
    let cancelled = false;
    fetch(`/api/studio/base-models?type=${encodeURIComponent(runType)}`, { credentials: 'include' })
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then((d) => { if (!cancelled) setBaseModels(d.models || []); })
      .catch(() => { if (!cancelled) setBaseModels([]); });
    return () => { cancelled = true; };
  }, [runType]);

  const comparison = selection.length >= 2;
  // Branche 1-LoRA : le dataset = le LoRA coché ; à 0 coché on retombe sur le
  // dataset pré-sélectionné (URL) s'il existe, sinon rien (invite).
  const soloDatasetId = selection.length === 1 ? selection[0].dataset_id : preselect;
  // Famille de la LIGNE cochée → le studio solo s'ouvre sur la bonne pipeline
  // (ex. cocher « Lola [KREA] » ouvre Krea, pas le train_type par défaut du dataset).
  const soloFamily = selection.length === 1 ? selection[0].family : null;

  return (
    <div className="flex flex-col gap-3">
      <header className="flex items-center gap-2 flex-wrap sticky top-0 z-10 bg-app/80 backdrop-blur py-2">
        <h1 className="text-content font-bold flex items-center gap-2">
          🧪 {t('studio.title')}<HelpBadge topic="page-studio" />
        </h1>
        {comparison && (
          <span className="px-2 py-0.5 rounded-lg border border-amber-400/40 bg-amber-400/10 text-amber-200 text-[0.6875rem] font-semibold">
            ⚖ {t('studio.comparing', { count: selection.length })}
          </span>
        )}
      </header>

      {/* Ancre de la barre de raccourcis du bas (StudioActionBar → 🧬 LoRAs). */}
      <div id="st-loras" className="scroll-mt-16">
        <LoraPicker preselectDataset={preselect} onSelectionChange={onSelectionChange} />
      </div>

      {comparison ? (
        <ComparisonStudio selection={selection} baseModels={baseModels} runType={runType} />
      ) : soloDatasetId ? (
        // `key` force un remontage propre quand on change de LoRA solo OU de famille
        // (reset des hooks/état du studio riche — sinon on garderait la grille du précédent).
        <LegacyDatasetStudio key={`${soloDatasetId}:${soloFamily ?? 'default'}`}
          datasetId={String(soloDatasetId)} initialFamily={soloFamily} />
      ) : (
        <p className="text-content-subtle text-sm rounded-lg border border-border bg-surface px-3 py-6 text-center">
          {t('studio.emptySelection')}
        </p>
      )}
    </div>
  );
}
