/**
 * ⚖ Compare / 🧬 Blend for « Generate from the board », plus one weight slider
 * per ticked checkpoint.
 *
 * Compare is the default and the historical behaviour: one pass per checkpoint,
 * swept across the strengths. Blend is the SAME mode the Test Studio's 🧬 toggle
 * drives — every ticked checkpoint loaded in the SAME generation, each at its own
 * weight, with every dataset's trigger word injected into the prompt.
 *
 * One word on both screens, deliberately: the Studio said « 🧬 Combine » until
 * 2026-08-03 and now says Blend too. The API argument is still `combine` and so
 * is the value the Studio persists — a label does not rename stored data.
 *
 * The logic lives in utils/canvasGeneration.js on top of the Test Studio's own
 * studio/loraStack.js: same clamp, same "two LoRAs of one family" rule, one
 * implementation. This file only draws it.
 *
 * The board's panel is a 26-rem drawer, and a bottom sheet under `sm` — so the
 * name and its slider are on TWO lines, the lesson LoraStackPanel already
 * learned: side by side, a 400-px screen truncated the checkpoint to "#1…".
 */
import { HelpBadge } from '../../help/HelpMode';
import BlendWeightRow from '../dataset/studio/BlendWeightRow';
import BlendSweepSummary from '../dataset/studio/BlendSweepSummary';
import {
  canvasBlendConfigCount, canvasStackKey, canvasStackTriggers, canvasStackWeight,
  canvasStackWeightSet, canvasStackWithoutTrigger,
} from '../../utils/canvasGeneration';
import { runNumber } from '../../utils/runIdentity';

export default function CanvasBlendPanel({
  selection, mode, onMode, weights, onWeight, blocker = null, familyReason = null,
  sets = {}, onToggleChip = null, count = 1, secondsPerImage = null,
  // 🔤 État de la case « Trigger word » du panneau de lancement voisin : la
  // promesse « Added to the front of your prompt » doit se taire quand elle
  // est décochée (rien ne sera préfixé).
  injectTrigger = true,
}) {
  const blend = mode === 'blend';
  const triggers = canvasStackTriggers(selection);
  const untriggered = canvasStackWithoutTrigger(selection);
  const configCount = canvasBlendConfigCount(selection, { weights, sets });

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-border bg-app/40 p-2">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className="text-[0.6875rem] font-semibold text-content">
          <span aria-hidden>⚗️</span> How to use these checkpoints
        </span>
        <HelpBadge topic="canvas-blend" />
        <div role="group" aria-label="Canvas run mode"
          className="ml-auto flex rounded-lg border border-border bg-app/60 p-0.5">
          {[['compare', '⚖ Compare'], ['blend', '🧬 Blend']].map(([value, label]) => {
            // Mixed families is not "a blend that fails", it is a run that cannot
            // exist: the toggle goes dead and SAYS which families are in the way,
            // instead of letting the user set weights for a launch that is refused.
            const dead = value === 'blend' && !!familyReason;
            return (
              <button key={value} type="button"
                onClick={() => !dead && onMode(value)}
                disabled={dead}
                aria-pressed={mode === value}
                title={dead ? familyReason : undefined}
                className={`rounded px-2.5 py-1 text-[0.6875rem] font-semibold ${
                  mode === value ? 'bg-primary/30 text-content' : 'text-content-subtle hover:text-content'
                } ${dead ? 'cursor-not-allowed opacity-40 hover:text-content-subtle' : ''}`}>
                {label}
              </button>
            );
          })}
        </div>
      </div>

      <p className="m-0 text-content-subtle text-[0.6875rem]">
        {blend
          ? 'One generation loads every ticked checkpoint together, each at its own '
            + 'weight. The strength sweep is replaced by these weights, so a blend '
            + 'costs one image per seed instead of one per checkpoint.'
          : 'Each checkpoint is rendered on its own, one pass per pick, swept across '
            + 'the strengths below.'}
      </p>

      {/* Mixed families: why the toggle is dead, in the panel and not only in a
          tooltip a touch screen never shows. */}
      {familyReason && (
        <p className="m-0 rounded-lg border border-amber-400/40 bg-amber-500/10 px-2.5 py-1.5 text-amber-200 text-[0.6875rem]"
          role="status">
          {familyReason}
        </p>
      )}

      {blend && blocker && !familyReason && (
        <p className="m-0 rounded-lg border border-amber-400/40 bg-amber-500/10 px-2.5 py-1.5 text-amber-200 text-[0.6875rem]"
          role="status">
          {blocker}
        </p>
      )}

      {blend && !blocker && (
        <>
          {/* The honest sentence. Two identity LoRAs DO blend — into someone who
              is neither, which is a result people ask for on purpose and also the
              one that surprises everybody who expected "both people". Saying it
              here costs a line; not saying it costs a GPU hour and a bug report. */}
          <p className="m-0 rounded-lg border border-border bg-surface px-2.5 py-1.5 text-content-muted text-[0.6875rem]">
            <span aria-hidden>💡</span> Two identity LoRAs blend into a hybrid person —
            neither of the two, sometimes exactly what you want. The usual sweet spot is
            identity + style, or identity + concept.
          </p>

          <ul className="m-0 flex list-none flex-col gap-1.5 p-0">
            {selection.map((e, i) => {
              const k = canvasStackKey(e);
              const label = `#${e.recordId} · step ${e.step}`;
              return (
                <BlendWeightRow
                  key={k}
                  index={i + 1}
                  label={`${e.datasetName || `Dataset ${e.datasetId}`} · ${label}`}
                  title={`${e.datasetName || `Dataset ${e.datasetId}`} — run ${runNumber({ record_id: e.recordId })}, step ${e.step}`}
                  weight={canvasStackWeight(weights, e)}
                  onWeight={(v) => onWeight(k, v)}
                  set={canvasStackWeightSet(sets, e)}
                  onToggleChip={(w) => onToggleChip?.(k, w)}
                  trigger={e.triggerWord ? (
                    <code className="shrink-0 rounded border border-indigo-400/40 bg-indigo-500/10 px-1.5 py-0.5 text-[0.625rem] font-semibold text-indigo-300">
                      {e.triggerWord}
                    </code>
                  ) : (
                    <span className="shrink-0 text-amber-300/80 text-[0.625rem]"
                      title="This dataset has no trigger word — nothing of it is added to the prompt">
                      no trigger
                    </span>
                  )}
                />
              );
            })}
          </ul>

          <BlendSweepSummary configCount={configCount} count={count}
            secondsPerImage={secondsPerImage} />

          {/* No silent magic: the exact tokens that will be prefixed to whatever
              prompt is typed below, in the order they will be prefixed in. */}
          <p className="m-0 text-content-subtle text-[0.6875rem]" data-testid="canvas-blend-triggers">
            {!injectTrigger
              ? 'The Trigger word box below is unticked — no trigger word is added to your prompt.'
              : triggers.length
                ? (
                  <>
                    Added to the front of your prompt:{' '}
                    {triggers.map((t, i) => (
                      <span key={t}>
                        {i > 0 && ', '}
                        <code className="rounded border border-indigo-400/40 bg-indigo-500/10 px-1 text-indigo-300">{t}</code>
                      </span>
                    ))}
                  </>
                )
                : 'None of these datasets has a trigger word, so nothing is added to your prompt.'}
            {injectTrigger && triggers.length > 0 && untriggered.length > 0
              && ` ${untriggered.length} pick${untriggered.length > 1 ? 's have' : ' has'} no trigger word.`}
          </p>
        </>
      )}
    </div>
  );
}
