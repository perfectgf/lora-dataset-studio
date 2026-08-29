// react-frontend/src/components/dataset/studio/LoraStackPanel.jsx
/**
 * Bascule Compare / Blend + poids par LoRA de la pile (≥2 LoRA cochés).
 *
 * « Compare » = comportement historique : chaque LoRA est testé SEUL, une colonne
 * par LoRA. « Blend » = les LoRA cochés sont chargés ENSEMBLE dans la même image,
 * chacun à son poids, et tous leurs triggers sont injectés dans le prompt.
 *
 * ⚠️ Ce mode s'est appelé « 🧬 Combine » jusqu'au 03/08/2026. SEUL LE LIBELLÉ a
 * changé, pour que le Test Studio et le ◉ LoRA Canvas cessent d'avoir deux mots
 * pour un seul mode : la valeur persistée (`studioComp_mode === 'combine'`), la
 * clé de l'API (`combine: true`) et l'id du sujet d'aide (`studio-combine-loras`)
 * restent CE QU'ILS ÉTAIENT — les renommer casserait le localStorage de tout le
 * monde pour un mot.
 *
 * La logique (clé de poids, clamp, blocage inter-familles, coût) vit dans
 * ./loraStack.js — testée sous `node --test`, que le JSX rend inaccessible ici.
 */
import { HelpBadge } from '../../../help/HelpMode';
import BlendWeightRow from './BlendWeightRow';
import BlendSweepSummary from './BlendSweepSummary';
import {
  blendConfigCount, combineBlocker, stackKey, stackWeight, stackWeightSet,
} from './loraStack';

export default function LoraStackPanel({ selection, mode, onMode, weights, onWeight,
  sets = {}, onToggleChip = null, count = 1, batchMult = 1, secondsPerImage = null,
  // 🔤 État de la case « Trigger word » du panneau de lancement : la phrase du
  // mode Blend ne doit pas promettre une injection que la case annule.
  injectTrigger = true }) {
  const combine = mode === 'combine';
  const blocker = combine ? combineBlocker(selection) : null;
  const configCount = blendConfigCount(selection, { weights, sets });

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-content-muted text-[0.6875rem] uppercase">
          How to use the {selection.length} LoRAs
        </span>
        <HelpBadge topic="studio-combine-loras" />
        <div role="group" aria-label="LoRA run mode"
          className="ml-auto flex rounded-lg border border-border bg-app/60 p-0.5">
          {/* La VALEUR reste 'combine' (elle est dans le localStorage de tous les
              utilisateurs et dans le corps du POST) ; seul le libellé dit Blend. */}
          {[['compare', '⚖ Compare'], ['combine', '🧬 Blend']].map(([value, label]) => (
            <button key={value} type="button" onClick={() => onMode(value)}
              aria-pressed={mode === value}
              className={`px-2.5 py-1 rounded text-[0.6875rem] font-semibold ${
                mode === value ? 'bg-primary/30 text-content' : 'text-content-subtle hover:text-content'
              }`}>
              {label}
            </button>
          ))}
        </div>
      </div>

      <p className="m-0 text-content-subtle text-[0.6875rem]">
        {combine
          ? 'All checked LoRAs load together in the same image, each at its own weight. '
            + (injectTrigger
              ? 'Every trigger word is injected into the prompt. '
              : 'The Trigger word box is unticked — no trigger words are injected. ')
            + 'Tick several weights on a LoRA and the launch renders every combination of them.'
          : 'Each LoRA is tested on its own, one column per LoRA, swept across the '
            + 'strengths below.'}
      </p>

      {combine && blocker && (
        <p className="m-0 rounded-lg border border-amber-400/40 bg-amber-500/10 px-2.5 py-1.5 text-amber-200 text-[0.6875rem]"
          role="status">
          {blocker}
        </p>
      )}

      {combine && !blocker && (
        <>
          <ul className="m-0 flex list-none flex-col gap-1.5 p-0">
            {selection.map((s, i) => (
              <BlendWeightRow
                key={stackKey(s)}
                index={i + 1}
                label={s.lora_label}
                weight={stackWeight(weights, s)}
                onWeight={(v) => onWeight(stackKey(s), v)}
                set={stackWeightSet(sets, s)}
                onToggleChip={(w) => onToggleChip?.(stackKey(s), w)}
                trigger={s.trigger_word ? (
                  <code className="shrink-0 rounded border border-indigo-400/40 bg-indigo-500/10 px-1.5 py-0.5 text-[0.625rem] font-semibold text-indigo-300">
                    {s.trigger_word}
                  </code>
                ) : null}
              />
            ))}
          </ul>
          <BlendSweepSummary configCount={configCount} count={count} batchMult={batchMult}
            secondsPerImage={secondsPerImage} />
        </>
      )}
    </div>
  );
}
