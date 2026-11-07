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
import StudioModelDownloads from './StudioModelDownloads';
import { launchSettings, launchText as batchLaunchText, mergeBatches, visibleBatch } from './promptBatch';
import { readInjectTrigger, writeInjectTrigger } from './triggerPref';
import ScenePromptsPanel from './ScenePromptsPanel';
import { combinedPromptBatch } from './scenePrompts';
import { heavyRunConfirm, heavyRunNotice, runCost } from './runCost';
import { showsNegativePrompt, studioModelSettingsLink } from '../../../utils/studioFamilySettings';

// Left run-setup rail: pickers, seed/launch controls and status banners, extracted unchanged from
// LoraTestStudio.jsx. Preserve gpu_busy, pending/cancel and resumable/resume banners and the
// non-pending controls: checkpoints, strengths, prompt/history, model/aspect/CFG/steps,
// seed/reroll/lock/batch/count and launch. d is useLoraTestStudio payload, studio its hook and
// form useStudioForm. Optional datasetId is needed for RecentPrompts thumbnails because d lacks
// it; StudioShell passes it. Canvas mounts this SAME panel, selecting checkpoints through board
// nodes across datasets instead of CheckpointPicker. Optional checkpointSlot replaces that picker;
// launchBlocked/launchLabel explain deployment or mixed-family blocks. Without props, behavior is
// unchanged. All other generation settings stay shared. Canvas Blend supplies
// showStrengths/cellTotal because all checkpoints form ONE weighted image, so strengths no longer
// sweep or multiply the count.
export default function RunSetupPanel({ d, studio, form, datasetId,
  checkpointSlot = null, launchBlocked = false, launchLabel = null, launchHint = null, actionBar = true,
  showStrengths = true, cellTotal = null, genStoragePrefix = null,
  // Optional PARENT control of Trigger word lets Canvas share it with Blend text that must
  // describe injection accurately. Without it, the panel owns state as before.
  injectTrigger: injectTriggerProp = null, onInjectTrigger: onInjectTriggerProp = null }) {
  const navigate = useNavigate();
  // GLOBAL generation settings match Generate except for the prompt builder.
  // StudioGenerationSettings supplies a snake_case object ready for POST /run: the single source
  // for precision, format, detail, negative and always-on LoRAs. It gates by FAMILY and persists
  // its own settings.
  const [genSettings, setGenSettings] = useState({});
  // Launch 409 studio_missing (P0-a) produces an actionable banner naming missing model files and
  // nodes.
  const [preflight, setPreflight] = useState(null);
  // 409 studio_arch_mismatch means a selected checkpoint's ACTUAL architecture conflicts with the
  // Studio family, such as a misclassified deployment. Show a separate banner.
  const [archMismatch, setArchMismatch] = useState(null);

  // PROMPT BATCH: replay ALL checked history prompts in one run as a backend axis. GPU execution
  // is serial and another POST would be rejected as an already-running test. No checks preserves
  // the single field prompt. Intentionally NOT persisted: a batch expresses one launch's intent,
  // and restoring three checks after reload could silently triple a seemingly single run.
  const [batchPrompts, setBatchPrompts] = useState([]);
  // Prompt-batch rules live in pure, tested promptBatch.js and are shared by both surfaces rather
  // than reimplemented.
  const pickedPrompts = visibleBatch(batchPrompts, d.recent_prompts);
  const toggleBatchPrompt = (p) => setBatchPrompts((cur) => (
    cur.includes(p) ? cur.filter((v) => v !== p) : [...cur, p]));
  // Checked Civitai prompts are passes in the same batch without first entering history; launch
  // adds them there. Do not persist them, for the same reason as history batch selections.
  const [civitaiPicks, setCivitaiPicks] = useState([]);
  const toggleCivitaiPick = (p) => setCivitaiPicks((cur) => (
    cur.includes(p) ? cur.filter((v) => v !== p) : [...cur, p]));

  // Scenes use captions from a bank OR dataset IN ORDER. Each checked scene adds a pass to the
  // prompt axis. Do not persist, matching history batches. Rules live in pure, tested
  // scenePrompts.js.
  const [sceneBatch, setSceneBatch] = useState({ source: null, scenes: [], picked: [], extras: {} });
  const allPickedPrompts = combinedPromptBatch(
    mergeBatches(pickedPrompts, civitaiPicks),
    sceneBatch.scenes, sceneBatch.picked, sceneBatch.extras);

  // Trigger word optionally prefixes the dataset trigger; checked preserves the default. Share the
  // browser preference across launch surfaces through pure triggerPref, never direct storage
  // access in this panel, as required by the batch contract. Unchecked sends inject_trigger:false;
  // checked omits the field, preserving the previous body byte-for-byte.
  const [ownInjectTrigger, setOwnInjectTrigger] = useState(readInjectTrigger);
  const injectTrigger = injectTriggerProp ?? ownInjectTrigger;
  const toggleInjectTrigger = onInjectTriggerProp ?? ((v) => {
    setOwnInjectTrigger(v);
    writeInjectTrigger(v);
  });

  // Count cells ACTUALLY launched. cellTotal is supplied only when a mode changes the formula,
  // such as Blend where one stack is one configuration; otherwise use the form total. Each checked
  // prompt adds another pass over the SAME grid. Count and button must explain this before
  // clicking, not only in the resulting queue.
  const promptMult = Math.max(1, allPickedPrompts.length);
  const cells = cellTotal != null ? cellTotal : form.total;
  const total = cells * promptMult;
  const modelError = d.generation_readiness?.config_error || form.modelError;
  const canLaunch = total > 0 && !d.pending && !d.gpu_busy && !studio.launching
    && !launchBlocked && !modelError && d.generation_readiness?.models_ready !== false
    && !d.generation_readiness?.missing_nodes?.length;
  const launchText = batchLaunchText(launchLabel, allPickedPrompts);
  // Always-on LoRA batch comparison generates each configuration WITHOUT then WITH each checked
  // LoRA. Image/time estimates must include the backend's 1 + checked-count multiplier.
  const batchMult = 1 + ((genSettings.batch_loras || []).length);
  // Canvas replaces studio.launch itself in useCanvasStudio, so EVERY setting including
  // genSettings passes through this shared call site. Replacing the handler here would silently
  // drop global settings from Canvas runs. Estimate ALL passes using this machine's MEASURED pace.
  // Above the threshold, show cost and ask ONE question instead of rejecting the launch.
  const cost = runCost(total * batchMult * form.genCount, d.seconds_per_image);

  const onLaunch = async () => {
    if (!canLaunch) return;
    if (cost.heavy && !window.confirm(heavyRunConfirm(cost))) return;
    // prompts travels through the SAME channel as global settings, which both hooks spread into
    // POST bodies. No signature change is needed and both routes receive the same batch. Omit with
    // no selection to keep the old body byte-for-byte.
    const base = launchSettings(genSettings, allPickedPrompts);
    // With an empty batch, base IS the genSettings state object, an identity pinned by
    // promptBatch.test.js. NEVER mutate it: writing inject_trigger:false there made the flag stick
    // after rechecking. Checked returns the identical object and old body; unchecked returns a
    // copy carrying the field.
    const settings = injectTrigger ? base : { ...base, inject_trigger: false };
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
      {/* Preflight: missing models/nodes (P0-a) and architecture mismatch. */}
      <StudioPreflightBanner missing={preflight} archMismatch={archMismatch}
        onRefresh={async () => { await studio.refresh(); setPreflight(null); }}
        onDismiss={() => { setPreflight(null); setArchMismatch(null); }} />

      {/* Safeguards. */}
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
            className="ml-auto px-2.5 py-1 rounded-lg bg-gradient-primary text-gray-950 text-xs font-semibold disabled:opacity-40">
            {studio.confirmingComfyuiRestart ? 'Confirming…' : '✓ I restarted ComfyUI'}
          </button>
        </div>
      )}

      {/* Active run. */}
      {d.pending > 0 && (
        <div className="flex items-center gap-2 rounded-lg border border-border bg-surface-raised px-3 py-2" role="status">
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

      {/* Stopped run can be resumed. */}
      {!d.pending && d.resumable > 0 && (
        <div className="flex items-center gap-2 rounded-lg border border-amber-400/40 bg-amber-400/10 px-3 py-2" role="status">
          <span aria-hidden>⏸</span>
          <span className="text-content text-sm">{d.resumable} stopped cell(s) — resumable with their settings</span>
          <button type="button" disabled={!!d.gpu_busy || studio.launching}
            onClick={() => studio.resume()}
            className="ml-auto px-2.5 py-1 rounded-lg bg-gradient-primary text-gray-950 text-xs font-semibold disabled:opacity-40">
            ▶ Resume test
          </button>
        </div>
      )}

      {/* Run setup. */}
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
            civitaiPicks={civitaiPicks}
            onToggleCivitaiPick={toggleCivitaiPick}
            onClearCivitaiPicks={() => setCivitaiPicks([])}
            injectTrigger={injectTrigger}
            onInjectTrigger={toggleInjectTrigger}
          />

          {/*
           * Ordered bank or dataset captions provide extra prompt passes, directly below the
           * prompt they extend.
           */}
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
          {modelError && (
            <p className="m-0 text-[0.6875rem] text-amber-200" role="alert">
              {modelError}{' '}
              {!d.generation_readiness?.config_error && d.z_models?.length > 0 && (
                <button type="button" onClick={form.resetModels} className="underline">
                  Use {d.z_models[0].label}
                </button>
              )}
              {studioModelSettingsLink(d.family) && (
                <> <a href={studioModelSettingsLink(d.family)} className="underline">Open model settings →</a></>
              )}
            </p>
          )}

          <StudioModelDownloads readiness={d.generation_readiness}
            onRefresh={async () => { await studio.refresh(); setPreflight(null); }} />

          {/*
           * Global generation settings match Generate: aspect/resolution and family-specific
           * sampling, detail, engine precision/always-on LoRAs, and negative prompt. One source
           * shared with comparison.
           */}
          <StudioGenerationSettings
            family={d.family}
            capabilities={d.generation_capabilities}
            cfgs={form.effectiveCfgs}
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
          {/*
           * Announce long launches instead of forbidding them: show count, duration at measured
           * pace, and remind users that Stop preserves completed work.
           */}
          {cost.heavy && (
            <p data-testid="heavy-run-notice"
              className="m-0 rounded-lg border border-amber-400/40 bg-amber-500/10 px-2.5 py-1.5 text-[0.6875rem] text-amber-200"
              role="status">
              <span aria-hidden>⏱</span> {heavyRunNotice(cost)}
            </p>
          )}
        </div>
      )}

      {/*
       * Fixed command bar keeps Run visible with comparison's section anchors; aspect remains the
       * Formats axis here. Omit on Canvas because its drawer already has a sticky footer: two
       * stacked bars consumed half the usable height at 400 px.
       */}
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
          ...(showsNegativePrompt(d.generation_capabilities, d.family, form.effectiveCfgs) ? [{ id: 'st-negative', emoji: '🚫', label: 'Negative' }] : []),
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
