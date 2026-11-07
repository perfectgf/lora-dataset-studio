// react-frontend/src/components/dataset/studio/ComparisonStudio.jsx
/**
 * Multi-LoRA COMPARISON Studio for at least two checked LoRAs, selected by StudioShell.
 * StudioRunSetup configures the received selection, POST /api/studio/run launches, and
 * useStudioRun(run_id) manages polling, votes, cancel and resume. LoraComparisonGrid uses LoRA
 * columns and strength rows; data.lora_ranking feeds rankings. Quick voting and lightbox reuse
 * useQuickVote, QuickVoteModal and StudioResultViewer. LoraPicker stays in StudioShell, shared
 * with the single-LoRA branch; this component receives the fixed selection and manages only the
 * run.
 */
import { useEffect, useMemo, useState } from 'react';
import { postJson } from '../../../api/fetchClient';
import { useToast } from '../../common/Toast';
import { useStudioRun } from '../../../hooks/useStudioRun';
import { useQuickVote } from '../../../hooks/useQuickVote';
import { fmt } from '../../../utils/studioFormat';
import { flipOrder } from './flipOrder';
import { DEFAULT_STRENGTHS, FAMILY_LABELS } from './constants';
import { blendConfigCount, buildSelectionsPayload, combineBlocker } from './loraStack';
import { axisPayload, axisTotal, effectiveAxis, toggleAxisValue } from './studioAxes';
import { mergeBatches } from './promptBatch';
import AxisPickers from './AxisPickers';
import { isStackRun, stackMembers } from './stackResults';
import StudioRunSetup from './StudioRunSetup';
import { readInjectTrigger, writeInjectTrigger } from './triggerPref';
import LoraStackPanel from './LoraStackPanel';
import StackCompositionPanel from './StackCompositionPanel';
import StackVariantsGrid from './StackVariantsGrid';
import StudioGenerationSettings from './StudioGenerationSettings';
import StudioActionBar from './StudioActionBar';
import StudioPreflightBanner from './StudioPreflightBanner';
import StudioModelDownloads from './StudioModelDownloads';
import LoraComparisonGrid from './LoraComparisonGrid';
import LoraRankingPanel from './LoraRankingPanel';
import RunSelector from './RunSelector';
import QuickVoteModal from './QuickVoteModal';
import StudioResultViewer from './StudioResultViewer';
import {
  modelSelectionError, readComparisonAxis, showsNegativePrompt, studioModelSettingsLink, writeComparisonAxis,
} from '../../../utils/studioFamilySettings';

const rollSeed = () => Math.floor(Math.random() * 2 ** 31);

export default function ComparisonStudio({ selection, baseModels = [], axes = null,
  modelDefaults = null, runType = 'zimage', baseNote = null,
  generationCapabilities = null, generationReadiness = null, defaultModel = null, settingsError = null,
  onRefreshModels = null }) {
  const toast = useToast();

  // Run settings are persisted so page reloads preserve them.
  const [strengths, setStrengths] = useState(() => {
    try {
      const v = JSON.parse(localStorage.getItem('studioComp_strengths') || 'null');
      return Array.isArray(v) && v.length ? v : DEFAULT_STRENGTHS;
    } catch { return DEFAULT_STRENGTHS; }
  });
  const [prompt, setPrompt] = useState(() => {
    try { return localStorage.getItem('studioComp_prompt') || ''; } catch { return ''; }
  });
  // Trigger word checkbox shares the triggerPref preference with Test Studio and Canvas:
  // unchecking here applies everywhere.
  const [injectTrigger, setInjectTrigger] = useState(readInjectTrigger);
  const toggleInjectTrigger = (v) => {
    setInjectTrigger(v);
    writeInjectTrigger(v);
  };
  const [seed, setSeed] = useState(() => rollSeed());

  // This surface's PROMPT BATCH lives HERE because this component builds the POST body, matching
  // RunSetupPanel on the other two launch surfaces. Intentionally NOT persisted: a batch expresses
  // one launch's intent; restoring it after reload could multiply an apparently single run.
  const [historyBatch, setHistoryBatch] = useState([]);
  const [civitaiPicks, setCivitaiPicks] = useState([]);
  const toggleIn = (set) => (v) => set((cur) => (
    cur.includes(v) ? cur.filter((x) => x !== v) : [...cur, v]));
  // Deduplicate: a previously launched Civitai prompt is also in history and selectable from both
  // lists. Identical cells waste GPU time.
  const pickedPrompts = mergeBatches(historyBatch, civitaiPicks);
  // compare means one LoRA per cell; combine stacks all selected LoRAs in the SAME image at their
  // respective weights. Persist like other settings.
  const [mode, setMode] = useState(() => {
    try { return localStorage.getItem('studioComp_mode') === 'combine' ? 'combine' : 'compare'; }
    catch { return 'compare'; }
  });
  // Stack weights keyed by dataset_id:checkpoint survive unchecking a DIFFERENT LoRA.
  const [stackWeights, setStackWeights] = useState(() => {
    try { return JSON.parse(localStorage.getItem('studioComp_weights') || '{}') || {}; }
    catch { return {}; }
  });
  // Checked weights per LoRA define the blend sweep. Use a NEW storage key so existing data
  // retains its meaning. Missing data reads as {}, meaning no checks and slider-controlled
  // weights, exactly as before.
  const [stackSets, setStackSets] = useState(() => {
    try { return JSON.parse(localStorage.getItem('studioComp_weightSets') || '{}') || {}; }
    catch { return {}; }
  });
  const toggleStackChip = (k, w) => setStackSets((cur) => {
    const list = Array.isArray(cur[k]) ? cur[k] : [];
    const next = list.includes(w) ? list.filter((v) => v !== w) : [...list, w];
    return { ...cur, [k]: next };
  });
  const [count, setCount] = useState(() => {
    try { return Math.max(1, parseInt(localStorage.getItem('studioComp_count'), 10) || 1); } catch { return 1; }
  });
  // Render axes are scoped by family. Legacy shared values migrate once to an existing family;
  // newly supported pipelines start from their own server defaults.
  const readAxis = (key) => readComparisonAxis(globalThis.localStorage, key, runType);
  // Untouched selection follows the first server model; an explicit choice stays visible if it
  // disappears. The shell remounts this component when the family changes.
  const [baseChoice, setSelectedBase] = useState(null);
  const selectedBase = baseChoice ?? defaultModel ?? baseModels[0]?.filename ?? '';
  const modelError = modelSelectionError(baseModels.map((m) => ({ value: m.filename })),
    baseChoice == null && defaultModel == null ? [] : [selectedBase]);
  const [selCfgs, setSelCfgs] = useState(() => readAxis('studioComp_cfgs'));
  const [selSteps, setSelSteps] = useState(() => readAxis('studioComp_steps'));
  const [selSteps2, setSelSteps2] = useState(() => readAxis('studioComp_steps2'));
  // Axis defaults depend on BASE as well as family: undistilled Z-Image Base or full Krea 2 Raw
  // yields blurry sketches with Turbo's CFG 1 / 8 steps. Single-LoRA Studio already used
  // model_defaults; comparison lacked them and used wrong defaults on expensive models. Never
  // overwrite user-edited axes (non-null selCfgs). selectedBase is empty when no alternative
  // exists, and the payload's empty key IS the selected default; ignoring it caused the blurry
  // undistilled outputs in #18.
  const baseDefaults = (modelDefaults ? modelDefaults[selectedBase || ''] : null) || null;
  const defaultCfg = baseDefaults?.cfg ?? axes?.default_cfg;
  const defaultSteps = baseDefaults?.steps ?? axes?.default_steps;
  const effectiveCfgs = effectiveAxis(selCfgs, defaultCfg);
  const effectiveSteps = effectiveAxis(selSteps, defaultSteps);
  // The second pass belongs to SDXL. Without this guard, persisted SDXL choices would follow users
  // into Z-Image when selection changed family and send an unsupported axis. Families without a
  // second pass use an empty axis.
  const effectiveSteps2 = axes?.steps2_choices
    ? effectiveAxis(selSteps2, axes.default_steps2) : [];
  useEffect(() => {
    writeComparisonAxis(globalThis.localStorage, 'studioComp_cfgs', runType, selCfgs);
    writeComparisonAxis(globalThis.localStorage, 'studioComp_steps', runType, selSteps);
    writeComparisonAxis(globalThis.localStorage, 'studioComp_steps2', runType, selSteps2);
  }, [runType, selCfgs, selSteps, selSteps2]);
  useEffect(() => {
    try {
      localStorage.setItem('studioComp_strengths', JSON.stringify(strengths));
      localStorage.setItem('studioComp_prompt', prompt);
      localStorage.setItem('studioComp_count', String(count));
      localStorage.setItem('studioComp_mode', mode);
      localStorage.setItem('studioComp_weights', JSON.stringify(stackWeights));
      localStorage.setItem('studioComp_weightSets', JSON.stringify(stackSets));
    } catch { /* private mode */ }
  }, [strengths, prompt, count, mode, stackWeights, stackSets]);
  const [launching, setLaunching] = useState(false);
  // Launch 409 studio_missing (P0-a): show the missing models/nodes banner.
  const [preflight, setPreflight] = useState(null);
  // 409 studio_arch_mismatch means the checkpoint's ACTUAL architecture conflicts with the family.
  const [archMismatch, setArchMismatch] = useState(null);
  // GLOBAL generation settings, matching Generate, come from StudioGenerationSettings as a
  // snake_case object ready to merge into POST /run (see launch()).
  const [genSettings, setGenSettings] = useState({});
  const toggleStrength = (s) =>
    setStrengths((cur) => (cur.includes(s) ? cur.filter((v) => v !== s) : [...cur, s].sort((a, b) => a - b)));


  // Managed run.
  const [runId, setRunId] = useState(null);
  const run = useStudioRun(runId);
  const data = run.data;
  const loras = data?.loras || [];
  const cells = useMemo(() => data?.cells || [], [data]);

  const vote = useQuickVote(run.rate);
  const [lbImg, setLbImg] = useState(null);
  const [showResults, setShowResults] = useState(true);
  const rateLightbox = (id, nv) => {
    run.rate(id, nv);
    setLbImg((p) => (p && p.id === id ? { ...p, rating: nv } : p));
  };

  // Cells ACTUALLY visible: for stacks, all displayed weight variants, not only the open run.
  // Quick voting and lightbox must match what users see, or a three-vote count could appear above
  // six unvoted tiles.
  const displayedCells = useMemo(() => {
    const variantCells = (data?.stack_variants || []).flatMap((v) => v.cells || []);
    if (!variantCells.length) return cells;
    const seen = new Set(variantCells.map((c) => c.id));
    return [...variantCells, ...cells.filter((c) => !seen.has(c.id))];
  }, [cells, data]);

  const unvoted = useMemo(
    () => displayedCells.filter((c) => c.status === 'done' && c.filename && !c.rating),
    [displayedCells],
  );
  const greens = useMemo(
    () => displayedCells.filter((c) => c.status === 'done' && c.filename && c.rating === 1),
    [displayedCells],
  );

  // Lightbox navigation keeps strengths of the same LoRA/seed adjacent: sort by dataset_id,
  // aspect, seed, then STRENGTH last. Pass live cells directly. Use displayedCells rather than
  // cells; otherwise another stack variant's image has index -1 and no navigation arrows.
  const navImages = useMemo(
    () => flipOrder(displayedCells,
      (c) => [c.dataset_id ?? 0, c.aspect || '', c.seed ?? 0, c.strength ?? 0]),
    [displayedCells],
  );

  const combine = mode === 'combine';
  const combineBlocked = combine ? combineBlocker(selection) : null;
  const launchBlocked = settingsError || generationReadiness?.config_error || modelError
    || (generationReadiness?.models_ready === false ? 'Download the required model files to continue.' : null)
    || (generationReadiness?.missing_nodes?.length ? 'Repair the missing ComfyUI nodes to continue.' : null)
    || combineBlocked;

  // STACK view follows the DISPLAYED RUN, not the Compare/Blend toggle: yesterday's stack can be
  // opened while the toggle is on Compare, and vice versa.
  const shownStack = useMemo(() => stackMembers(data), [data]);
  const showStackView = isStackRun(data);
  const [savingBest, setSavingBest] = useState(false);
  const [bestSavedAt, setBestSavedAt] = useState(null);
  // Clear confirmation on run changes: Saved under a DIFFERENT stack would falsely describe the
  // one just pinned.
  useEffect(() => { setBestSavedAt(null); }, [runId]);

  const saveStackBest = async ({ dataset_id: dsId, ...body }) => {
    setSavingBest(true);
    try {
      await postJson(`/api/dataset/${dsId}/lora-test/best`, body);
      setBestSavedAt(Date.now());
      toast.success('★ Stack weights saved as the best setting');
    } catch (e) {
      toast.error(e.message || 'Could not save the best setting');
    } finally {
      setSavingBest(false);
    }
  };

  // Use these weights restores a variant's weights to sliders. Keys come from loraStack.stackKey
  // and can be read directly by those sliders.
  const useVariantWeights = (map) => {
    if (!map || Object.keys(map).length === 0) {
      toast.error('This run did not record enough to reload its weights');
      return;
    }
    setStackWeights((cur) => ({ ...cur, ...map }));
    setMode('combine');
    toast.success('Weights loaded — adjust them and run again to add a variant');
  };

  const launch = async () => {
    if (!selection.length || launchBlocked) return;
    if (!combine && !strengths.length) return;
    setLaunching(true);
    try {
      const body = {
        selections: buildSelectionsPayload(selection, { combine, weights: stackWeights, sets: stackSets }),
        // In combine mode every LoRA has its own weight, so omit the strengths axis; the backend
        // substitutes the leading LoRA's weight.
        ...(combine ? { combine: true } : { strengths }),
        seed,
        count,
        // Omit run base when it is empty (Krea's Official entry) or nothing is selected. The
        // backend retains the family default: configured UNET or first model.
        z_model: selectedBase || undefined,
        // Global resolution_tier, negative, sampler, detail and other settings are already gated
        // BY FAMILY on the backend. Omitted empty fields preserve defaults.
        ...genSettings,
        // CFG, steps and second pass go LAST: current panel choices override same-named global
        // settings. The route and engine already supported these axes; the missing pieces were the
        // request body and panel controls.
        ...axisPayload({ cfgs: effectiveCfgs, steps: effectiveSteps, steps2: effectiveSteps2 }),
      };
      if (prompt.trim()) body.prompt = prompt.trim();
      // Prompt-batch axis: with nothing checked, omit the key so the request body stays
      // byte-for-byte identical to the earlier behavior.
      if (pickedPrompts.length) body.prompts = [...pickedPrompts];
      // Unchecking Trigger word sends the prompt unchanged. When checked, omit the field to
      // preserve the previous request body byte-for-byte.
      if (!injectTrigger) body.inject_trigger = false;
      const dResp = await postJson('/api/studio/run', body);
      // Keep this defensive path even though apiFetch currently throws on
      // non-2xx: alternate clients/tests may return the structured 409 body.
      // Never announce success or retain a bogus run id in that case.
      if (!dResp?.ok) {
        let errorBody = dResp;
        if (typeof dResp?.json === 'function') {
          try { errorBody = await dResp.json(); } catch { errorBody = {}; }
        }
        setPreflight(errorBody?.studio_missing || null);
        setArchMismatch(errorBody?.studio_arch_mismatch || null);
        toast.error(errorBody?.error || 'Error on launch');
        return;
      }
      toast.success(`${dResp.created} generation(s) queued (seed ${dResp.seed}${dResp.count > 1 ? ` ×${dResp.count}` : ''})`);
      setRunId(dResp.run_id);
      setSeed(rollSeed());
      setPreflight(null);
      setArchMismatch(null);
    } catch (e) {
      // apiFetch throws on non-2xx; a 409 carries the itemized manques on e.body (P0-a)
      // or a wrong-arch checkpoint on e.body.studio_arch_mismatch.
      setPreflight(e?.body?.studio_missing || null);
      setArchMismatch(e?.body?.studio_arch_mismatch || null);
      toast.error(e.message || 'Error on launch');
    } finally {
      setLaunching(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-4 items-start">
      <aside className="flex flex-col gap-3 lg:sticky lg:top-16 lg:max-h-[calc(100vh-7rem)] lg:overflow-auto">
        {/*
         * Explain unusual default-base conditions even without a picker: single-base installations
         * still need this information.
         */}
        {baseNote && (
          <p className="m-0 rounded-lg border border-amber-400/30 bg-amber-400/5 px-3 py-2
                        text-[0.6875rem] leading-snug text-amber-300/80 break-words">
            {baseNote}
          </p>
        )}
        {/*
         * Base picker for all families. Krea returns a list of the selected default and
         * alternatives only when local Krea UNETs exist. Otherwise hide the picker and apply the
         * selected default to node 20.
         */}
        {baseModels.length > 0 && (
          <div className="flex flex-col gap-1 rounded-lg border border-border bg-surface p-3">
            <span className="text-content-muted text-[0.625rem] uppercase">
              Base model ({FAMILY_LABELS[runType] || runType})
            </span>
            <select
              value={selectedBase}
              onChange={(e) => setSelectedBase(e.target.value)}
              aria-label="Base model for this run"
              className="rounded border border-border bg-app/60 px-1.5 py-1 text-content text-sm"
            >
              {baseModels.map((m) => (
                <option key={m.filename} value={m.filename}>{m.label}</option>
              ))}
              {modelError && <option value={selectedBase}>{selectedBase} (unavailable)</option>}
            </select>
          </div>
        )}
        {(modelError || generationReadiness?.config_error) && studioModelSettingsLink(runType) && (
          <a href={studioModelSettingsLink(runType)} className="text-sm underline">Open model settings →</a>
        )}
        <StudioModelDownloads readiness={generationReadiness}
          onRefresh={async () => { await onRefreshModels?.(); setPreflight(null); }} />
        <LoraStackPanel selection={selection} mode={mode} onMode={setMode}
          weights={stackWeights}
          sets={stackSets}
          onToggleChip={toggleStackChip}
          count={count}
          secondsPerImage={axes?.seconds_per_image ?? null}
          injectTrigger={injectTrigger}
          onWeight={(k, v) => setStackWeights((cur) => ({ ...cur, [k]: v }))} />
        <div id="st-setup" className="scroll-mt-16">
          <StudioRunSetup
            selectionCount={selection.length}
            strengths={strengths}
            onToggleStrength={toggleStrength}
            prompt={prompt}
            onPrompt={setPrompt}
            seed={seed}
            onReroll={() => setSeed(rollSeed())}
            count={count}
            onCount={setCount}
            onLaunch={launch}
            launching={launching}
            gpuBusy={data?.gpu_busy}
            batchMult={1 + ((genSettings.batch_loras || []).length)}
            combine={combine}
            combineBlocked={launchBlocked}
            configCount={blendConfigCount(selection, { weights: stackWeights, sets: stackSets })}
            axisTotal={axisTotal({ cfgs: effectiveCfgs, steps: effectiveSteps, steps2: effectiveSteps2 })}
            secondsPerImage={axes?.seconds_per_image ?? null}
            injectTrigger={injectTrigger}
            onInjectTrigger={toggleInjectTrigger}
            batchPrompts={historyBatch}
            onToggleBatchPrompt={toggleIn(setHistoryBatch)}
            onClearBatchPrompts={() => setHistoryBatch([])}
            civitaiPicks={civitaiPicks}
            onToggleCivitaiPick={toggleIn(setCivitaiPicks)}
            onClearCivitaiPicks={() => setCivitaiPicks([])}
            pickedPrompts={pickedPrompts}
            /*
             * Render axes belong in run setup beside strength selection, using the SAME component
             * as single-LoRA Studio and Canvas. Base and format are already selected above, so
             * hide those two blocks here.
             */
            axisSlot={axes ? (
              /*
               * defaultCfg/defaultSteps belong to the selected BASE, falling back to family
               * defaults. The default chip must mark the value actually launched, not the one
               * producing blurry results.
               */
              <AxisPickers
                zModels={null} effectiveModels={[]} onToggleModel={() => {}}
                aspects={null} effectiveAspects={[]} onToggleAspect={() => {}}
                cfgChoices={axes.cfg_choices} effectiveCfgs={effectiveCfgs}
                onToggleCfg={(v) => setSelCfgs((cur) => toggleAxisValue(effectiveAxis(cur, defaultCfg), v))}
                defaultCfg={defaultCfg}
                stepsChoices={axes.steps_choices} effectiveSteps={effectiveSteps}
                onToggleStep={(v) => setSelSteps((cur) => toggleAxisValue(effectiveAxis(cur, defaultSteps), v))}
                defaultSteps={defaultSteps}
                steps2Choices={axes.steps2_choices} effectiveSteps2={effectiveSteps2}
                onToggleStep2={(v) => setSelSteps2((cur) => toggleAxisValue(effectiveAxis(cur, axes.default_steps2), v))}
                defaultSteps2={axes.default_steps2}
                fmt={fmt}
              />
            ) : null}
          />
        </div>
        {/*
         * Global generation settings match Generate except for the prompt builder. key=runType
         * remounts cleanly on family changes; state and localStorage are family-specific.
         * Comparison uses one GLOBAL aspect choice sent as a one-value axis through aspectPicker.
         */}
        <StudioGenerationSettings
          key={runType}
          family={runType}
          capabilities={generationCapabilities}
          cfgs={effectiveCfgs}
          storagePrefix={`studioGenComp_${runType}`}
          aspectPicker
          onChange={setGenSettings}
        />
        {/*
         * A stack has only one tested LoRA, making a one-row LoRA ranking uninformative. Show its
         * composition instead.
         */}
        {showStackView ? (
          <StackCompositionPanel members={shownStack} onSaveBest={saveStackBest}
            saving={savingBest} savedAt={bestSavedAt}
            // Use the displayed RUN's actual state, not the checkbox: one False cell is
            // sufficient; all cells are False when launched unchecked.
            injectTrigger={!cells.some((c) => c.inject_trigger === false)} />
        ) : (
          <LoraRankingPanel ranking={data?.lora_ranking} />
        )}
      </aside>

      <main id="st-results" className="flex flex-col gap-3 min-w-0 scroll-mt-16">
        <StudioPreflightBanner missing={preflight} archMismatch={archMismatch}
          onRefresh={async () => { await onRefreshModels?.(); setPreflight(null); }}
          onDismiss={() => { setPreflight(null); setArchMismatch(null); }} />
        {data?.comfyui_recovery?.requires_comfyui_restart_confirmation && (
          <div className="flex items-center gap-2 flex-wrap rounded-lg border border-amber-400/40 bg-amber-400/10 px-3 py-2" role="status">
            <span aria-hidden>⚠</span>
            <span className="text-content text-sm">A ComfyUI submission has an unknown outcome. Restart ComfyUI first, then confirm it here; the paused cell will become resumable.</span>
            <button type="button" disabled={run.confirmingComfyuiRestart}
              onClick={run.confirmComfyuiRestart}
              className="ml-auto px-2.5 py-1 rounded-lg bg-gradient-primary text-gray-950 text-xs font-semibold disabled:opacity-40">
              {run.confirmingComfyuiRestart ? 'Confirming…' : '✓ I restarted ComfyUI'}
            </button>
          </div>
        )}

        {data?.pending > 0 && (
          <div className="flex items-center gap-2 rounded-lg border border-border bg-surface-raised px-3 py-2" role="status">
            <span className="inline-block w-4 h-4 border-2 border-indigo-400/40 border-t-indigo-400 rounded-full animate-spin" aria-hidden />
            <span className="text-content text-sm">
              {data.generating ?? data.running ?? 0} generating · {data.queued ?? data.pending} queued
            </span>
            <button type="button" onClick={run.cancel}
              className="ml-auto px-2.5 py-1 rounded-lg bg-red-600/80 text-white text-xs font-semibold">
              Stop (resumable)
            </button>
          </div>
        )}
        {!data?.pending && data?.resumable > 0 && (
          <div className="flex items-center gap-2 rounded-lg border border-amber-400/40 bg-amber-400/10 px-3 py-2" role="status">
            <span aria-hidden>⏸</span>
            <span className="text-content text-sm">{data.resumable} stopped cell(s) — resumable with their settings</span>
            <button type="button" disabled={!!data?.gpu_busy} onClick={run.resume}
              className="ml-auto px-2.5 py-1 rounded-lg bg-gradient-primary text-gray-950 text-xs font-semibold disabled:opacity-40">
              ▶ Resume the test
            </button>
          </div>
        )}

        {!runId ? (
          <p className="text-content-subtle text-sm rounded-lg border border-border bg-surface px-3 py-6 text-center">
            Set up the run on the left then “🚀 Run the test”{combine
              ? ` to render the ${selection.length} LoRAs together in one image.`
              : ` to compare the ${selection.length} LoRAs side by side.`}
          </p>
        ) : (
          <div className="flex flex-col gap-2">
            <RunSelector
              runs={[]}
              activeRunKey={null}
              onSelect={() => {}}
              unvotedCount={unvoted.length}
              onStartVote={() => vote.startVoting(unvoted)}
              greenCount={greens.length}
              onStartReVote={() => vote.startVoting(greens, '♻️ Reconfirm the 👍')}
              displayedCount={cells.length}
              showResults={showResults}
              onToggleResults={() => setShowResults((v) => !v)}
            />
            {showResults && (showStackView ? (
              <StackVariantsGrid members={shownStack} variants={data?.stack_variants}
                onRate={run.rate} onOpen={setLbImg} onSelectRun={setRunId}
                onUseWeights={useVariantWeights} />
            ) : (
              <LoraComparisonGrid loras={loras} cells={cells} onRate={run.rate} onOpen={setLbImg} />
            ))}
          </div>
        )}
      </main>

      <QuickVoteModal vote={vote} datasetId={vote.current?.dataset_id} fmt={fmt} />
      {lbImg && (
        <StudioResultViewer img={lbImg} items={navImages}
          onRate={rateLightbox} onNavigate={setLbImg} onClose={() => setLbImg(null)} />
      )}

      {/* Fixed command bar keeps Run visible with section shortcuts. */}
      <StudioActionBar
        shortcuts={[
          { id: 'st-loras', emoji: '🧬', label: 'LoRAs' },
          { id: 'st-setup', emoji: '📝', label: 'Prompt & seed' },
          { id: 'st-format', emoji: '📐', label: 'Format' },
          ...(runType === 'krea' ? [
            { id: 'st-sampling', emoji: '🎛️', label: 'Sampling' },
            { id: 'st-engine', emoji: '⚙️', label: 'Engine' },
          ] : []),
          ...(runType === 'sdxl' ? [{ id: 'st-detail', emoji: '✨', label: 'Detail' }] : []),
          ...(showsNegativePrompt(generationCapabilities, runType, effectiveCfgs) ? [{ id: 'st-negative', emoji: '🚫', label: 'Negative' }] : []),
          { id: 'st-results', emoji: '🖼️', label: 'Results' },
        ]}
        canRun={!!selection.length && (combine || !!strengths.length) && !launching && !data?.gpu_busy && !launchBlocked}
        running={launching}
        onRun={launch}
      />
    </div>
  );
}
