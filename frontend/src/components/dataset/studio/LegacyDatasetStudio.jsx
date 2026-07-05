// react-frontend/src/components/dataset/studio/LegacyDatasetStudio.jsx
/**
 * Studio de test RICHE per-dataset (mono-LoRA) — le studio d'origine, intact :
 * rail de réglages (RunSetupPanel), grilles de résultats (ResultsArea), meilleur
 * réglage temps réel (BestPresetCard) + persisté (BestSettingsBanner), meilleur
 * réglage par modèle (BestPerModelList), comparaison des bases (ModelComparison),
 * vote rapide (QuickVoteModal) et lightbox (ResultLightbox).
 *
 * Extrait 1:1 du corps de l'ancien StudioShell (avant la réécriture multi-LoRA) :
 * c'est la branche « 1 LoRA coché » de StudioShell. Reçoit `datasetId` (le
 * dataset_id du LoRA coché) — tout le reste (axes, presets, ★ Appliquer→generate)
 * est piloté par useLoraTestStudio/useStudioForm comme avant.
 */
import { useEffect, useState } from 'react';
import { useLoraTestStudio } from '../../../hooks/useLoraTestStudio';
import { useStudioForm } from '../../../hooks/useStudioForm';
import { useQuickVote } from '../../../hooks/useQuickVote';
import { fmt } from '../../../utils/studioFormat';
import FamilySelector from './FamilySelector';
import RunSetupPanel from './RunSetupPanel';
import FaceRankingPanel from './FaceRankingPanel';
import BestSettingsBanner from './BestSettingsBanner';
import BestPresetCard from './BestPresetCard';
import BestPerModelList from './BestPerModelList';
import ModelComparison from './ModelComparison';
import ResultsArea from './ResultsArea';
import QuickVoteModal from './QuickVoteModal';
import ResultLightbox from './ResultLightbox';

export default function LegacyDatasetStudio({ datasetId, initialFamily = null }) {
  // Famille (pipeline) sélectionnée : null = défaut résolu côté serveur. `initialFamily`
  // = la famille de la LIGNE cochée dans le picker (ex. « Lola [KREA] » → ouvre sur krea).
  // Changer de famille REMONTE le corps du studio (key) → hook + formulaire repartent
  // propres pour la nouvelle pipeline (pas de checkpoints/réglages de l'autre qui traînent).
  const [family, setFamily] = useState(initialFamily);
  useEffect(() => { setFamily(initialFamily); }, [datasetId, initialFamily]);  // reset au changement de ligne
  return (
    <StudioBody key={`${datasetId}:${family ?? 'default'}`}
      datasetId={datasetId} family={family} onFamilyChange={setFamily} />
  );
}

function StudioBody({ datasetId, family, onFamilyChange }) {
  const studio = useLoraTestStudio(datasetId, family);
  const d = studio.data;
  const form = useStudioForm(d, datasetId, d?.family || family);
  const vote = useQuickVote(studio.rate);
  const [lbImg, setLbImg] = useState(null);

  // La lightbox délègue le vote ici ; on met à jour l'image affichée pour que le
  // bouton 👍/👎 reflète l'état immédiatement (comme l'ancien setLbImg local).
  const rateLightbox = (id, nv) => {
    studio.rate(id, nv);
    setLbImg((p) => (p && p.id === id ? { ...p, rating: nv } : p));
  };

  if (!d || !d.checkpoints?.length) {
    return (
      <div className="flex flex-col gap-3">
        {d && <FamilySelector families={d.available_families} active={d.family} onSelect={onFamilyChange} />}
        <p className="text-content-subtle text-sm rounded-lg border border-border bg-surface px-3 py-6 text-center">
          {d ? 'No testable checkpoint for this pipeline (train it first).' : 'Loading…'}
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <FamilySelector families={d.available_families} active={d.family} onSelect={onFamilyChange} />
      {d?.trigger_word && (
        <div className="flex items-center gap-2 flex-wrap">
          <code className="px-2 py-0.5 rounded-lg border border-indigo-400/40 bg-indigo-500/10 text-indigo-300 text-[0.6875rem] font-semibold">
            {d.trigger_word}
          </code>
          {d?.best_settings && (
            <span className="text-amber-300 text-[0.6875rem]" title="Saved winning settings">
              ★ {fmt(d.best_settings.strength)}
            </span>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-4 items-start">
        <aside className="flex flex-col gap-2 lg:sticky lg:top-16 lg:max-h-[calc(100vh-7rem)] lg:overflow-auto">
          <RunSetupPanel d={d} studio={studio} form={form} datasetId={datasetId} />
        </aside>
        <main id="st-results" className="flex flex-col gap-3 min-w-0 scroll-mt-16">
          {/* « Best epoch » OBJECTIF : classement InsightFace des checkpoints
              (complète le best_preset issu des votes 👍/👎 juste en dessous). */}
          <FaceRankingPanel ranking={d.face_ranking} onScore={studio.scoreFaces}
            scoring={studio.scoring}
            hasCells={(d.cells || []).some((c) => c.status === 'done')} />
          <BestPresetCard preset={d.best_preset} onMemorize={studio.setBest} fmt={fmt} />
          <BestSettingsBanner best={d.best_settings} onClear={() => studio.clearBest(d.family)} fmt={fmt} />
          <BestPerModelList items={d.best_per_model} breakdown={d.checkpoint_breakdown} datasetId={datasetId}
            onMemorize={studio.setBest} fmt={fmt} />
          <ModelComparison items={d.model_comparison} />
          <ResultsArea datasetId={datasetId} d={d} studio={studio} vote={vote} onOpen={setLbImg} />
        </main>
      </div>

      <QuickVoteModal vote={vote} datasetId={datasetId} fmt={fmt} />
      {lbImg && (
        <ResultLightbox img={lbImg} datasetId={datasetId}
          onRate={rateLightbox} onClose={() => setLbImg(null)} fmt={fmt} />
      )}
    </div>
  );
}
