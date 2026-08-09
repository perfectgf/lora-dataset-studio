import { useEffect, useMemo, useState } from 'react';
import RunSetupPanel from '../dataset/studio/RunSetupPanel';
import CanvasBlendPanel from './CanvasBlendPanel';
import { useCanvasStudio } from '../../hooks/useCanvasStudio';
import { useStudioForm } from '../../hooks/useStudioForm';
import {
  canvasBlendBlocker, canvasBlendConfigCount, canvasFamily, canvasSelectionSummary,
  describeCanvasLaunch,
} from '../../utils/canvasGeneration';
import { famLabel } from '../../utils/familyLabels';
import { DEPLOY_BAR_CLASS } from '../../utils/checkpointDeployState';
import { runNumber } from '../../utils/runIdentity';
import { HelpBadge } from '../../help/HelpMode';

/* 🎨 Generating from the board.

   This panel is deliberately thin. It mounts the Test Studio's own hooks and
   renders the Test Studio's own settings panel — same prompt field, same seed
   controls, same model/format/cfg/steps axes, same global generation settings.
   The ONE difference is where the checkpoints come from: ticked pills on the
   canvas instead of a picker, possibly from several datasets at once.

   Because it is the same code, a setting added to the Test Studio appears here
   without anyone touching this file, and the two screens cannot drift.

   Layout: a bottom sheet under `lg`, a side drawer above it. A full settings
   panel on a 400-px canvas is the hard case of this screen — a side drawer
   there would leave a sliver of board, and the board is what you are picking
   from. The switch used to be at `sm` (640 px) and that reasoning stopped one
   breakpoint too early: this drawer is a FIXED 26 rem, so at 768 px it took 54 %
   of the window and left a 352-px sliver — the very thing the paragraph above
   rules out, on the width a phone in landscape actually reports. A side drawer
   only earns its keep once the board keeps ~600 px, which is `lg`. */

/** The picks, as the panel shows them back: one removable chip per checkpoint,
 *  grouped so a cross-dataset run reads as one. Replaces the CheckpointPicker. */
function CanvasCheckpointRecap({ selection, onToggle, onClear }) {
  const lanes = useMemo(() => {
    const by = new Map();
    for (const e of selection) {
      if (!by.has(e.datasetId)) by.set(e.datasetId, { name: e.datasetName, picks: [] });
      by.get(e.datasetId).picks.push(e);
    }
    return [...by.entries()];
  }, [selection]);

  return (
    <div id="st-loras" className="scroll-mt-16 rounded-lg border border-border bg-app/40 p-2">
      <div className="mb-1.5 flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className="text-[0.6875rem] font-semibold text-content">
          <span aria-hidden>🧬</span> Checkpoints
        </span>
        <span className="text-[0.625rem] text-content-muted">{canvasSelectionSummary(selection)}</span>
        <HelpBadge topic="canvas-generate" />
        {selection.length > 0 && (
          <button type="button" onClick={onClear}
            className="ml-auto text-content-subtle text-[0.625rem] underline decoration-dotted hover:text-content">
            Clear
          </button>
        )}
      </div>
      {selection.length === 0 ? (
        <p className="m-0 text-content-subtle text-[0.6875rem]">
          Tick the ✓ box on a checkpoint pill to add it here. Picks from several datasets
          run together — that is what the board is for.
        </p>
      ) : (
        <div className="flex flex-col gap-1.5">
          {lanes.map(([datasetId, lane]) => (
            <div key={datasetId} className="min-w-0">
              <div className="truncate text-content-subtle text-[0.5625rem] font-semibold uppercase tracking-wide">
                {lane.name || `Dataset ${datasetId}`}
              </div>
              <div className="mt-0.5 flex flex-wrap gap-1">
                {/* Same code as the board's pills: a solid sky edge means
                    deployed, a dashed slate one means on disk only. The chip
                    used to say "· to deploy" in a register of its own, which
                    made the panel and the board two vocabularies for one fact.
                    The words stay too — the bar is never the only signal. */}
                {lane.picks.map((e) => (
                  <button key={`${e.recordId}:${e.step}`} type="button"
                    onClick={() => onToggle(e)}
                    title={`Remove step ${e.step} of run ${runNumber({ record_id: e.recordId })} from this run`
                      + (e.deployed ? ' — deployed to ComfyUI'
                        : ' — on disk but not deployed yet; the launch deploys it first')}
                    className={'flex max-w-full items-center gap-1 rounded-md border px-1.5 py-0.5 text-[0.625rem] tabular-nums '
                      + (e.deployed
                        ? 'border-indigo-400/60 bg-indigo-500/15 text-indigo-100 '
                        : 'border-amber-400/50 bg-amber-500/10 text-amber-100 ')
                      + DEPLOY_BAR_CLASS[e.deployed ? 'deployed' : 'on-disk'] + ' '}>
                    <span className="truncate">
                      #{e.recordId} · {e.step}
                      {!e.deployed && <span className="opacity-80"> · not deployed yet</span>}
                    </span>
                    <span aria-hidden className="opacity-70">✕</span>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ⚖/🧬 The run mode and its weights survive the panel being closed and the page
   being reloaded, like every other setting on this screen. New keys — nothing
   stored under an existing name changes meaning. */
const MODE_KEY = 'canvasStack_mode';
const WEIGHTS_KEY = 'canvasStack_weights';
// Les poids COCHÉS (balayage 🧬). Clé neuve : une install qui n'en a pas lit {},
// c'est-à-dire aucune case cochée, c'est-à-dire les curseurs — comme avant.
const SETS_KEY = 'canvasStack_weightSets';
const readMode = () => {
  try { return localStorage.getItem(MODE_KEY) === 'blend' ? 'blend' : 'compare'; }
  catch { return 'compare'; }
};
const readWeights = () => {
  try { return JSON.parse(localStorage.getItem(WEIGHTS_KEY) || '{}') || {}; }
  catch { return {}; }
};
const readSets = () => {
  try { return JSON.parse(localStorage.getItem(SETS_KEY) || '{}') || {}; }
  catch { return {}; }
};

export default function CanvasGenerationPanel({ selection, onToggle, onClear, onDeploy, onClose,
  tracker, externalLoras = [] }) {
  const family = canvasFamily(selection);
  const verdict = describeCanvasLaunch(selection);
  const [mode, setMode] = useState(readMode);
  // Keyed on the PILL (see canvasGeneration), so a weight survives un-ticking
  // another pick, deploying, and reloading the page.
  const [weights, setWeights] = useState(readWeights);
  const [sets, setSets] = useState(readSets);
  useEffect(() => {
    try {
      localStorage.setItem(MODE_KEY, mode);
      localStorage.setItem(WEIGHTS_KEY, JSON.stringify(weights));
      localStorage.setItem(SETS_KEY, JSON.stringify(sets));
    } catch { /* private mode — persistence is best effort */ }
  }, [mode, weights, sets]);
  const toggleChip = (k, w) => setSets((cur) => {
    const list = Array.isArray(cur[k]) ? cur[k] : [];
    return { ...cur, [k]: list.includes(w) ? list.filter((v) => v !== w) : [...list, w] };
  });

  // Mixed families kill the whole launch, blend or not, and `verdict` already
  // says so in the better words. Only when the launch is otherwise possible does
  // the blend rule get to speak (it is the one that catches "a stack of one").
  const familyReason = verdict.families.length > 1 ? verdict.reason : null;
  // Below two picks there is nothing to blend, and the toggle is not on screen.
  // The mode is REMEMBERED rather than reset — un-ticking down to one pick and
  // back must not silently drop the user out of Blend — but it does not get to
  // block a launch with a message whose panel is hidden.
  const blendAvailable = selection.length > 1;
  const blendBlocker = (mode === 'blend' && blendAvailable)
    ? canvasBlendBlocker(selection) : null;
  const blend = mode === 'blend' && blendAvailable && !blendBlocker;
  const launchVerdict = (!verdict.blocked && blendBlocker)
    ? { ...verdict, blocked: true, reason: blendBlocker }
    : verdict;

  const configCount = blend ? canvasBlendConfigCount(selection, { weights, sets }) : 1;
  const rawStudio = useCanvasStudio(selection, family, {
    onDeploy, tracker, blend, weights, sets });
  // 🔌 The board's own nodes never touch `genSettings` inside RunSetupPanel —
  // that state is owned by StudioGenerationSettings, which knows nothing about
  // the board. `launch` is the one call site both screens share (see the
  // comment on it in RunSetupPanel), so external LoRAs ride into the same
  // genSettings object here instead, right where `studio.launch` is invoked.
  const studio = {
    ...rawStudio,
    launch: (checkpoints, strengths, seed, prompt, zModels, aspects, cfgs,
      stepsList, steps2List, count, genSettings = {}) => rawStudio.launch(
      checkpoints, strengths, seed, prompt, zModels, aspects, cfgs, stepsList, steps2List, count,
      externalLoras.length ? { ...genSettings, external_loras: externalLoras } : genSettings),
  };
  const pinned = useMemo(
    () => (studio.data?.checkpoints || []).map((c) => c.filename), [studio.data]);
  // Namespaced per FAMILY, never per dataset: a canvas run is cross-dataset by
  // design, so "the settings of dataset 7" would be the wrong thing to restore.
  const form = useStudioForm(studio.data, 'canvas', family, { pinnedCheckpoints: pinned });

  const recap = (
    <>
      <CanvasCheckpointRecap selection={selection} onToggle={onToggle} onClear={onClear} />
      {/* 🔌 Checked board nodes that are not a checkpoint at all — pinned LoRA
          files stacked on top of whichever checkpoints are picked above. */}
      {externalLoras.length > 0 && (
        <p className="m-0 text-cyan-200 text-[0.6875rem]">
          <span aria-hidden>🔌</span> {externalLoras.length} external LoRA{externalLoras.length > 1 ? 's' : ''} will be stacked on this run.
        </p>
      )}
      {selection.length > 1 && (
        <CanvasBlendPanel selection={selection} mode={mode} onMode={setMode}
          weights={weights}
          onWeight={(k, v) => setWeights((cur) => ({ ...cur, [k]: v }))}
          sets={sets} onToggleChip={toggleChip} count={form?.genCount ?? 1}
          secondsPerImage={studio.data?.seconds_per_image ?? null}
          blocker={blendBlocker} familyReason={familyReason} />
      )}
    </>
  );

  return (
    <aside
      data-testid="canvas-generation-panel"
      aria-label="Generate from the canvas"
      className="fixed inset-x-0 bottom-0 z-50 flex max-h-[78vh] flex-col overflow-hidden border-t border-border bg-surface-overlay shadow-xl
                 lg:inset-x-auto lg:right-0 lg:top-0 lg:h-full lg:max-h-none lg:w-[26rem] lg:border-l lg:border-t-0">
      <header className="flex shrink-0 items-center gap-2 border-b border-border px-3 py-2">
        <h3 className="min-w-0 flex-1 truncate text-sm font-semibold text-content">
          <span aria-hidden>🎨</span> Generate from the board
          {family && <span className="font-normal text-content-muted"> · {famLabel(family)}</span>}
        </h3>
        {/* The way back to the board, and the control most often reached for on a
            phone: 44 px of thumb below `lg`. Closing keeps every pick — the board's
            🎨 button carries the count — so this is a cheap, reversible gesture and
            it must not be a 14-px glyph. */}
        <button type="button" onClick={onClose} aria-label="Close"
          className="-my-1 -mr-1 flex h-11 w-11 shrink-0 items-center justify-center rounded-md text-base text-content-subtle hover:text-content lg:h-8 lg:w-8">✕</button>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {!studio.data ? (
          // The recap stays even with nothing loaded: it is where the picks land,
          // and an empty panel would read as "the board did not register my click".
          <div className="flex flex-col gap-2">
            {recap}
            <p className="m-0 text-content-subtle text-[0.75rem]">
              {selection.length
                ? 'Loading this family’s settings…'
                : 'Tick a checkpoint on the board to set up a run.'}
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            <RunSetupPanel
              d={studio.data}
              studio={studio}
              form={form}
              datasetId={selection[0]?.datasetId ?? null}
              checkpointSlot={recap}
              launchBlocked={launchVerdict.blocked}
              launchLabel={launchVerdict.label}
              launchHint={launchVerdict.reason}
              // 🧬 A blend is ONE configuration: each LoRA carries its own weight,
              // so the strength sweep has nothing left to sweep and the image
              // count must stop multiplying by it — or the panel would announce
              // six images and queue one.
              showStrengths={!blend}
              // 🧬 Un balayage rend `configCount` configurations, pas une :
              // le compteur du bouton doit les multiplier, ou il annonce une
              // image là où la file en recevra neuf.
              cellTotal={blend ? form.axisTotal * configCount : null}
              genStoragePrefix={`studioGen_canvas_${family || 'default'}`}
              // The Studio's fixed bottom bar would sit ON this sheet at 400 px.
              actionBar={false}
            />
          </div>
        )}
      </div>
    </aside>
  );
}
