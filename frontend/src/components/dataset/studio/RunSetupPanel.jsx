import { useState } from 'react';
import { useNavigate } from 'react-router';
import { STRENGTH_CHOICES } from './constants';
import { fmt } from '../../../utils/studioFormat';
import CheckpointPicker from './CheckpointPicker';
import StrengthPicker from './StrengthPicker';
import PromptField from './PromptField';
import AxisPickers from './AxisPickers';
import SeedControls from './SeedControls';
import LaunchBar from './LaunchBar';
import StudioGenerationSettings from './StudioGenerationSettings';
import StudioActionBar from './StudioActionBar';
import StudioPreflightBanner from './StudioPreflightBanner';
import { launchSettings, launchText as batchLaunchText, visibleBatch } from './promptBatch';
import ScenePromptsPanel from './ScenePromptsPanel';
import { combinedPromptBatch } from './scenePrompts';
import { heavyRunConfirm, heavyRunNotice, runCost } from './runCost';

// Rail gauche « Setup du run » : pickers + seed/launch + bandeaux d'état.
// Extraction behavior-preserving de LoraTestStudio.jsx :
//   - bandeaux gpu_busy / pending (→ studio.cancel) / resumable (→ studio.resume)
//   - le bloc {!d.pending && (...)} : checkpoints, strengths, prompt+récents,
//     modèle/formats/cfg/steps, seed/🎲/🔒/×N + compteur, bouton 🚀 Lancer.
// `d` = payload useLoraTestStudio ; `studio` = hook ; `form` = useStudioForm.
// `datasetId` (optionnel) : requis seulement par RecentPrompts pour les vignettes
//   (le payload `d` ne porte pas l'id du dataset → StudioShell le transmet). Voir
//   note de déviation §contrat dans le rapport de livraison de la Task 1.A.
//
// ◉ LoRA Canvas — ce panneau est MONTÉ TEL QUEL par le canvas, qui ne diffère que
// sur UN point : les checkpoints s'y choisissent en cliquant les pastilles des
// nœuds (sur plusieurs datasets), pas dans le CheckpointPicker. D'où trois props
// optionnelles, sans effet quand elles sont absentes :
//   `checkpointSlot`        remplace le CheckpointPicker par le récapitulatif de
//                           la sélection du board ;
//   `launchBlocked`/`launchLabel` : le bouton dit CE QU'IL VA FAIRE (« Deploy 2
//                           checkpoints, then generate ») ou POURQUOI il ne peut
//                           pas (familles mélangées). Jamais un bouton mort muet.
// Tout le reste — modèle, format, cfg, steps, steps2, seed, ×N, LoRA always-on,
// rebalance, négatif… — est le MÊME code, donc les deux écrans ne divergent pas.
//   `showStrengths`/`cellTotal` : le mode 🧬 Blend du board charge tous les
//                           checkpoints dans UNE image, chacun à son poids —
//                           l'axe strengths n'a plus rien à balayer, et le
//                           compteur ne doit plus le multiplier.
export default function RunSetupPanel({ d, studio, form, datasetId,
  checkpointSlot = null, launchBlocked = false, launchLabel = null, launchHint = null, actionBar = true,
  showStrengths = true, cellTotal = null, genStoragePrefix = null }) {
  const navigate = useNavigate();
  // Réglages de génération GLOBAUX (parité Generate, hors prompt builder) remontés par
  // StudioGenerationSettings : objet snake_case déjà prêt à fusionner dans le POST /run
  // (source unique de vérité pour rebalance/enhancer/precision/format/detail/negative +
  // pile LoRA « always-on »). Le composant est gaté PAR FAMILLE et se persiste seul.
  const [genSettings, setGenSettings] = useState({});
  // Manques de modèles/nodes remontés par un 409 `studio_missing` au lancement
  // (P0-a) → bandeau actionnable listant les fichiers/nodes absents.
  const [preflight, setPreflight] = useState(null);
  // 409 `studio_arch_mismatch` : un checkpoint sélectionné dont l'arch RÉELLE
  // contredit la famille du Studio (déploiement mal classé) → bandeau distinct.
  const [archMismatch, setArchMismatch] = useState(null);

  // 📝 LOT DE PROMPTS — les prompts cochés dans l'historique. Le lancement les
  // rejoue TOUS en un seul run (le backend en fait un axe : le GPU est sérialisé
  // et un second POST serait refusé par le garde « a test run is already in
  // progress »). Rien de coché = zéro changement : le prompt du champ, seul.
  //
  // Délibérément NON persisté (contrairement au mode 🧬 du board) : une sélection
  // de lot est l'intention d'UN lancement. Retrouver trois cases cochées après un
  // rechargement multiplierait par trois un run qu'on croyait simple.
  const [batchPrompts, setBatchPrompts] = useState([]);
  // La règle du lot vit dans promptBatch.js — pure, donc réellement testée, et
  // partagée par les deux surfaces plutôt que réécrite dans chacune.
  const pickedPrompts = visibleBatch(batchPrompts, d.recent_prompts);
  const toggleBatchPrompt = (p) => setBatchPrompts((cur) => (
    cur.includes(p) ? cur.filter((v) => v !== p) : [...cur, p]));

  // 🎬 Scenes : les captions d'une banque OU d'un dataset DANS L'ORDRE, chaque
  // scène cochée devenant une passe du même axe 📝. Non persisté, même raison que
  // le lot d'historique ci-dessus. La règle vit dans scenePrompts.js (pur, testé).
  const [sceneBatch, setSceneBatch] = useState({ source: null, scenes: [], picked: [] });
  const allPickedPrompts = combinedPromptBatch(
    pickedPrompts, sceneBatch.scenes, sceneBatch.picked);

  // Le nombre de cellules RÉELLEMENT lancées. `cellTotal` n'est fourni que par un
  // mode qui change la formule (🧬 Blend : une pile = une configuration) — sinon
  // c'est le total du formulaire, inchangé.
  // 📝 Chaque prompt coché est une passe de plus sur la MÊME grille : le compteur
  // et le bouton doivent le dire avant le clic, pas la file d'attente après.
  const promptMult = Math.max(1, allPickedPrompts.length);
  const cells = cellTotal != null ? cellTotal : form.total;
  const total = cells * promptMult;
  const canLaunch = total > 0 && !d.pending && !d.gpu_busy && !studio.launching
    && !launchBlocked;
  const launchText = batchLaunchText(launchLabel, allPickedPrompts);
  // Axe ⚖ batch (Always-on LoRA cochés batch) : chaque config tourne SANS puis
  // AVEC chaque LoRA coché → le compteur d'images/temps doit en tenir compte
  // (le backend multiplie déjà les cellules par 1 + nb cochés).
  const batchMult = 1 + ((genSettings.batch_loras || []).length);
  // The canvas swaps `studio.launch` itself (see useCanvasStudio), so EVERY
  // setting — genSettings included — travels through this one call site on both
  // screens. Overriding the handler here instead would have quietly dropped the
  // global generation settings from a canvas run.
  // ⏱ Ce que ce lancement va vraiment coûter : toutes les passes, au rythme
  // MESURÉ de la machine. Rien n'est refusé — au-delà du seuil on chiffre, et on
  // pose UNE question.
  const cost = runCost(total * batchMult * form.genCount, d.seconds_per_image);

  const onLaunch = async () => {
    if (cost.heavy && !window.confirm(heavyRunConfirm(cost))) return;
    // 📝 `prompts` voyage dans le MÊME canal que les réglages globaux (les deux
    // hooks étalent cet objet dans le corps du POST) — donc aucune signature à
    // changer, et le lot arrive identiquement sur les deux routes. Absent quand
    // rien n'est coché : le corps envoyé est alors octet pour octet celui d'avant.
    const settings = launchSettings(genSettings, allPickedPrompts);
    const res = await studio.launch(
      form.chosenCps, form.selSts, form.nextSeed(), form.effectivePrompt,
      form.effectiveModels, form.effectiveAspects, form.effectiveCfgs, form.effectiveSteps,
      form.effectiveSteps2, form.genCount, settings,
    );
    // Persist the itemized manques (toast is transient) — cleared on the next
    // launch that isn't blocked on missing assets.
    setPreflight(res && res.studio_missing ? res.studio_missing : null);
    setArchMismatch(res && res.studio_arch_mismatch ? res.studio_arch_mismatch : null);
  };

  return (
    <>
      {/* --- Preflight : modèles/nodes manquants (P0-a) + arch mismatch -- */}
      <StudioPreflightBanner missing={preflight} archMismatch={archMismatch}
        onDismiss={() => { setPreflight(null); setArchMismatch(null); }} />

      {/* --- Garde-fous ------------------------------------------------- */}
      {d.gpu_busy && !d.comfyui_recovery?.requires_comfyui_restart_confirmation && (
        <div className="m-0 flex flex-wrap items-center gap-2 rounded-lg border border-red-400/40 bg-red-500/10 px-3 py-2 text-red-300 text-sm" role="status">
          <span>{d.gpu_busy}</span>
          {d.comfyui_recovery_target?.dataset_id != null && (
            <button type="button"
              onClick={() => {
                const params = new URLSearchParams({
                  dataset: String(d.comfyui_recovery_target.dataset_id),
                  family: d.comfyui_recovery_target.family || 'zimage',
                });
                navigate(`/studio?${params.toString()}`);
              }}
              className="ml-auto rounded-lg border border-red-300/40 bg-red-400/15 px-2.5 py-1 text-xs font-semibold text-red-100">
              Open paused test →
            </button>
          )}
        </div>
      )}

      {/* --- Soumission ComfyUI inconnue : confirmation humaine requise ------ */}
      {d.comfyui_recovery?.requires_comfyui_restart_confirmation && (
        <div className="flex items-center gap-2 flex-wrap rounded-lg border border-amber-400/40 bg-amber-400/10 px-3 py-2" role="status">
          <span aria-hidden>⚠</span>
          <span className="text-content text-sm">ComfyUI could not confirm whether this image started. Restart ComfyUI, confirm it here, then click Resume test.</span>
          <button type="button" disabled={studio.confirmingComfyuiRestart || !studio.confirmComfyuiRestart}
            onClick={studio.confirmComfyuiRestart}
            className="ml-auto px-2.5 py-1 rounded-lg bg-gradient-primary text-white text-xs font-semibold disabled:opacity-40">
            {studio.confirmingComfyuiRestart ? 'Confirming…' : '✓ J’ai redémarré ComfyUI'}
          </button>
        </div>
      )}

      {/* --- Run en cours ------------------------------------------------ */}
      {d.pending > 0 && (
        <div className="flex items-center gap-2 rounded-lg border border-indigo-400/40 bg-indigo-500/10 px-3 py-2" role="status">
          <span className="inline-block w-4 h-4 border-2 border-indigo-400/40 border-t-indigo-400 rounded-full animate-spin" aria-hidden />
          <span className="text-content text-sm">
            {d.generating ?? d.running ?? 0} generating · {d.queued ?? d.pending} queued
          </span>
          <button type="button" onClick={studio.cancel}
            className="ml-auto px-2.5 py-1 rounded-lg bg-red-600/80 text-white text-xs font-semibold">
            Stop (resumable)
          </button>
        </div>
      )}

      {/* --- Run stoppé → reprenable ------------------------------------- */}
      {!d.pending && d.resumable > 0 && (
        <div className="flex items-center gap-2 rounded-lg border border-amber-400/40 bg-amber-400/10 px-3 py-2" role="status">
          <span aria-hidden>⏸</span>
          <span className="text-content text-sm">{d.resumable} stopped cell(s) — resumable with their settings</span>
          <button type="button" disabled={!!d.gpu_busy || studio.launching}
            onClick={() => studio.resume()}
            className="ml-auto px-2.5 py-1 rounded-lg bg-gradient-primary text-white text-xs font-semibold disabled:opacity-40">
            ▶ Resume test
          </button>
        </div>
      )}

      {/* --- Setup du run ------------------------------------------------ */}
      {!d.pending && (
        <div id="st-setup" data-probe-panel="setup" className="flex flex-col gap-2 scroll-mt-16">
          {/* The ONE thing the canvas does differently: its checkpoints are the
              pills ticked on the board, so it hands in its own recap here and
              the picker stays out of the way. */}
          {checkpointSlot ?? (
            <CheckpointPicker checkpoints={d.checkpoints} chosen={form.chosenCps}
              onToggle={form.toggleCp}
              guests={form.guestCps} onToggleGuest={form.toggleGuest}
              onAddGuest={form.addGuest} onRemoveGuest={form.removeGuest}
              family={d.family || 'zimage'} />
          )}

          {showStrengths && (
            <StrengthPicker choices={STRENGTH_CHOICES} selected={form.selSts} onToggle={form.toggleSt} fmt={fmt} />
          )}

          <PromptField
            value={form.effectivePrompt}
            placeholder={d.prompt}
            onChange={form.setPromptText}
            onReset={() => form.setPromptText(null)}
            isCustom={form.promptText !== null && form.promptText !== d.prompt}
            recentPrompts={d.recent_prompts}
            datasetId={datasetId}
            onDeletePrompt={studio.deletePrompt}
            batchPrompts={pickedPrompts}
            onToggleBatchPrompt={toggleBatchPrompt}
            onClearBatchPrompts={() => setBatchPrompts([])}
          />

          {/* 🎬 Les captions d'une banque ou d'un dataset, dans l'ordre, comme
              passes de prompt supplémentaires — juste sous le prompt qu'elles
              prolongent. */}
          <ScenePromptsPanel value={sceneBatch} onChange={setSceneBatch} />

          <AxisPickers
            zModels={d.z_models}
            effectiveModels={form.effectiveModels}
            onToggleModel={form.toggleModel}
            aspects={d.aspects}
            effectiveAspects={form.effectiveAspects}
            onToggleAspect={form.toggleAspect}
            cfgChoices={d.cfg_choices}
            effectiveCfgs={form.effectiveCfgs}
            onToggleCfg={form.toggleCfg}
            defaultCfg={form.modelDefaultCfg}
            stepsChoices={d.steps_choices}
            effectiveSteps={form.effectiveSteps}
            onToggleStep={form.toggleStep}
            defaultSteps={form.modelDefaultSteps}
            steps2Choices={d.steps2_choices}
            effectiveSteps2={form.effectiveSteps2}
            onToggleStep2={form.toggleStep2}
            defaultSteps2={d.default_steps2}
            mixedDefaults={form.mixedModelDefaults}
            baseNote={d.base_note}
            fmt={fmt}
          />

          {/* Réglages de génération globaux (parité Generate) : format/resolution, +
              selon la famille sampling/detail/engine (rebalance+enhancer+precision+LoRA
              always-on)/negative. Source unique de vérité, partagée avec la comparaison. */}
          <StudioGenerationSettings
            family={d.family}
            // The canvas overrides the namespace: its runs are cross-dataset, so
            // "the engine settings of dataset 7" would be restored (and saved)
            // under whichever pick happened to be first — the same reason
            // useStudioForm is namespaced by family there, not by dataset.
            storagePrefix={genStoragePrefix
              || `studioGen_${datasetId || 'x'}_${d.family || 'default'}`}
            permanentLoras={d.permanent_loras}
            onChange={setGenSettings}
          />

          <div className="flex items-center gap-2 flex-wrap">
            <SeedControls
              seed={form.seed}
              seedLocked={form.seedLocked}
              onReroll={() => form.setSeed(form.rollSeed())}
              onToggleLock={() => form.setSeedLocked((v) => !v)}
              genCount={form.genCount}
              onGenCount={form.setGenCount}
              total={total * batchMult}
              batchMult={batchMult}
              promptMult={promptMult}
              secondsPerImage={d.seconds_per_image}
              fmt={fmt}
            />
            <LaunchBar canLaunch={canLaunch} launching={studio.launching} onLaunch={onLaunch}
              label={launchText} title={launchHint} />
          </div>
          {/* A dead button that does not say why is what this replaces: the
              canvas passes the real reason (mixed families, nothing picked) and
              it is shown right under the button, not only in a tooltip. */}
          {launchHint && (
            <p className={'m-0 text-[0.6875rem] ' + (launchBlocked ? 'text-amber-200' : 'text-content-muted')}
              role={launchBlocked ? 'status' : undefined}>
              {launchHint}
            </p>
          )}
          {/* ⏱ Un long lancement est annoncé, pas interdit : le compte, la durée
              au rythme mesuré, et le rappel qu'un Stop garde ce qui est fait. */}
          {cost.heavy && (
            <p data-testid="heavy-run-notice"
              className="m-0 rounded-lg border border-amber-400/40 bg-amber-500/10 px-2.5 py-1.5 text-[0.6875rem] text-amber-200"
              role="status">
              <span aria-hidden>⏱</span> {heavyRunNotice(cost)}
            </p>
          )}
        </div>
      )}

      {/* Barre de commande fixe : Run toujours visible + raccourcis de sections
          (mêmes ancres que la comparaison ; le ratio reste l'axe Formats ici).
          Retirée sur le canvas, où le panneau est déjà un tiroir à pied collant :
          deux barres empilées au ras de l'écran, à 400 px, mangeaient la moitié
          de la hauteur utile. */}
      {actionBar && (
      <StudioActionBar
        shortcuts={[
          { id: 'st-loras', emoji: '🧬', label: 'LoRAs' },
          { id: 'st-setup', emoji: '📝', label: 'Prompt & seed' },
          { id: 'st-format', emoji: '📐', label: 'Format' },
          ...(d.family === 'krea' ? [
            { id: 'st-sampling', emoji: '🎛️', label: 'Sampling' },
            { id: 'st-engine', emoji: '⚙️', label: 'Engine' },
          ] : []),
          ...(d.family === 'sdxl' ? [{ id: 'st-detail', emoji: '✨', label: 'Detail' }] : []),
          ...(d.family === 'zimage' ? [{ id: 'st-negative', emoji: '🚫', label: 'Negative' }] : []),
          { id: 'st-results', emoji: '🖼️', label: 'Results' },
        ]}
        canRun={canLaunch}
        running={studio.launching}
        onRun={onLaunch}
        runLabel={launchText ? `🚀 ${launchText}` : undefined}
      />
      )}
    </>
  );
}
