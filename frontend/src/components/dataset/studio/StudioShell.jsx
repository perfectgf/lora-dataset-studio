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

export default function StudioShell({ preselectDataset = null, preselectFamily = null,
  preselectBase = null, datasetId = null }) {
  // `datasetId` legacy est un alias de preselectDataset.
  const preselect = preselectDataset ?? datasetId;

  const [selection, setSelection] = useState([]);
  const onSelectionChange = useCallback((sel) => setSelection(sel), []);

  // train_type du run = celui du 1er LoRA coché (null si rien coché).
  const runType = selection.length > 0 ? (selection[0].train_type || 'zimage') : null;

  // Liste des bases correspondant au train_type courant.
  // Fetch à chaque changement de runType via /api/studio/base-models?type=…
  const [baseModels, setBaseModels] = useState([]);
  // Échelles CFG/steps de la famille, servies par le MÊME appel (clé `axes`).
  // Sans elles, la branche comparaison/blend n'avait aucun axe de rendu à
  // proposer — c'est ce qui la privait du réglage des steps (bug signalé).
  const [axes, setAxes] = useState(null);
  // CFG/steps PAR BASE, servis par le même appel. Sans eux la branche
  // comparaison lançait une base non distillée (Z-Image Base, un modèle complet
  // Krea 2 Raw) avec les chiffres de la Turbo — cfg 1 / 8 steps — qui rendent une
  // esquisse floue lue comme « l'entraînement a raté ». Le studio mono-LoRA les
  // recevait déjà par son propre payload ; c'est la même source.
  const [modelDefaults, setModelDefaults] = useState(null);
  // Ce que le défaut de base a d'anormal, quand il en a. Servi par le même appel,
  // et présent MÊME quand `models` est vide — c'est l'install sans alternative qui
  // en a le plus besoin.
  const [baseNote, setBaseNote] = useState(null);
  useEffect(() => {
    if (!runType) { setBaseModels([]); setAxes(null); setModelDefaults(null); setBaseNote(null); return; }
    let cancelled = false;
    fetch(`/api/studio/base-models?type=${encodeURIComponent(runType)}`, { credentials: 'include' })
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then((d) => {
        if (cancelled) return;
        setBaseModels(d.models || []);
        setAxes(d.axes || null);
        setModelDefaults(d.model_defaults || null);
        setBaseNote(d.base_note || null);
      })
      .catch(() => {
        if (!cancelled) { setBaseModels([]); setAxes(null); setModelDefaults(null); setBaseNote(null); }
      });
    return () => { cancelled = true; };
  }, [runType]);

  const comparison = selection.length >= 2;
  // Branche 1-LoRA : le dataset = le LoRA coché ; à 0 coché on retombe sur le
  // dataset pré-sélectionné (URL) s'il existe, sinon rien (invite).
  const soloDatasetId = selection.length === 1 ? selection[0].dataset_id : preselect;
  // Famille de la LIGNE cochée → le studio solo s'ouvre sur la bonne pipeline
  // (ex. cocher « Lola [KREA] » ouvre Krea, pas le train_type par défaut du dataset).
  const soloFamily = selection.length === 1 ? selection[0].family : preselectFamily;

  return (
    <div className="flex flex-col gap-3">
      <header data-probe-chrome="header"
        className="flex items-center gap-2 flex-wrap sticky top-0 z-10 bg-app/80 backdrop-blur py-2">
        <h1 className="text-content font-bold flex items-center gap-2">🧪 Test Studio<HelpBadge topic="page-studio" /></h1>
        {comparison && (
          <span className="px-2 py-0.5 rounded-lg border border-amber-400/40 bg-amber-400/10 text-amber-200 text-[0.6875rem] font-semibold">
            {/* Neutre : le mode réel (⚖ Compare / 🧬 Blend) est choisi et affiché
                juste en dessous par LoraStackPanel — annoncer « Comparing » ici
                mentirait dès que la pile est active. */}
            {selection.length} LoRAs checked
          </span>
        )}
      </header>

      {/* Ancre de la barre de raccourcis du bas (StudioActionBar → 🧬 LoRAs). */}
      <div id="st-loras" className="scroll-mt-16">
        <LoraPicker preselectDataset={preselect} preselectFamily={preselectFamily}
          onSelectionChange={onSelectionChange} />
      </div>

      {comparison ? (
        <ComparisonStudio selection={selection} baseModels={baseModels} axes={axes}
          modelDefaults={modelDefaults} runType={runType} baseNote={baseNote} />
      ) : soloDatasetId ? (
        // `key` force un remontage propre quand on change de LoRA solo OU de famille
        // (reset des hooks/état du studio riche — sinon on garderait la grille du précédent).
        <LegacyDatasetStudio key={`${soloDatasetId}:${soloFamily ?? 'default'}`}
          datasetId={String(soloDatasetId)} initialFamily={soloFamily}
          initialBase={preselectBase} />
      ) : (
        <p className="text-content-subtle text-sm rounded-lg border border-border bg-surface px-3 py-6 text-center">
          Check a LoRA above to tune and test it. Check ≥2 to compare them side by side.
        </p>
      )}
    </div>
  );
}
