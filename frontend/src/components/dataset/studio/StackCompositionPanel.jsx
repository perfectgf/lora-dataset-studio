// react-frontend/src/components/dataset/studio/StackCompositionPanel.jsx
/**
 * « 🧬 Stack composition » — ce qu'il y a DANS la pile du run affiché : chaque LoRA,
 * son poids, son trigger. Remplace le « 🏆 LoRA Ranking » quand le run est une pile :
 * un classement à une entrée (une pile n'a qu'un LoRA « testé ») n'apprend rien, alors
 * que la composition est la seule chose qui distingue ce run de la même image générée
 * avec un LoRA seul.
 *
 * Porte aussi le « ★ Save as best setting » de la pile : le réglage gagnant d'une pile,
 * ce sont SES POIDS. Le corps du POST est fabriqué par ./stackResults (testé).
 *
 * Responsive : nom / trigger / poids sur deux lignes — à 400 px la colonne du studio
 * écrase un nom de LoRA mis côte à côte avec le reste (même leçon que LoraStackPanel).
 */
import { useState } from 'react';
import { HelpBadge } from '../../../help/HelpMode';
import { bestStackPayload, fmtWeight } from './stackResults';

// 🔤 `injectTrigger` = la vérité du RUN AFFICHÉ (relue de ses cellules par le
// parent), pas l'état courant de la case : un run lancé décoché doit le dire
// même si la case a été recochée depuis.
export default function StackCompositionPanel({ members, onSaveBest, saving = false, savedAt = null,
  injectTrigger = true }) {
  const [open, setOpen] = useState(true);
  if (!members?.length) return null;
  const payload = bestStackPayload(members);

  return (
    <div className="flex flex-col gap-1.5 rounded-lg border border-sky-400/40 bg-surface-raised px-3 py-2">
      <div className="flex items-center gap-2">
        <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open}
          className="flex items-center gap-2 text-left text-content-muted text-[0.625rem] uppercase">
          <span aria-hidden>{open ? '▾' : '▸'}</span>
          🧬 Stack composition ({members.length})
        </button>
        <HelpBadge topic="studio-stack-results" />
      </div>

      {open && (
        <>
          <ol className="m-0 flex list-none flex-col gap-1 p-0">
            {members.map((m, i) => (
              <li key={`${m.dataset_id}:${m.filename}`}
                className="flex flex-col gap-0.5 rounded bg-app/30 px-1.5 py-1 text-[0.6875rem]">
                <div className="flex min-w-0 items-center gap-1.5">
                  <span className="w-4 shrink-0 text-right text-content-subtle tabular-nums">{i + 1}.</span>
                  <span className="min-w-0 flex-1 truncate text-content font-medium" title={m.label}>
                    {m.label}
                  </span>
                  <span className="shrink-0 rounded border border-sky-400/50 bg-sky-400/15 px-1.5 py-px font-semibold tabular-nums text-sky-200"
                    title={`Weight of ${m.label} in this stack`}>
                    {fmtWeight(m.weight)}
                  </span>
                </div>
                <div className="flex items-center gap-1.5 pl-5">
                  {m.head && (
                    <span className="text-content-subtle text-[0.625rem]"
                      title="The first LoRA of the stack: it carries the run's dataset and default prompt.">
                      head
                    </span>
                  )}
                  {m.trigger ? (
                    <code className="min-w-0 truncate rounded border border-indigo-400/40 bg-indigo-500/10 px-1.5 py-px text-[0.625rem] font-semibold text-indigo-300"
                      title={injectTrigger
                        ? `Trigger word injected into the prompt: ${m.trigger}`
                        : `Trigger word (NOT injected in this run — the Trigger word box was unticked): ${m.trigger}`}>
                      {m.trigger}
                    </code>
                  ) : (
                    // Les runs lancés avant cette vue n'ont pas figé le trigger de leurs
                    // LoRA empilés : on le dit plutôt que de laisser croire qu'il n'y en a pas.
                    <span className="text-content-subtle text-[0.625rem]" title="This run predates the stack view, which is when trigger words started being recorded.">
                      trigger not recorded
                    </span>
                  )}
                </div>
              </li>
            ))}
          </ol>

          <p className="m-0 text-content-subtle text-[0.625rem] leading-relaxed">
            {injectTrigger
              ? 'All of these load in the same image and every trigger above is injected into the prompt. '
              : 'All of these load in the same image. This run was launched with the Trigger word box unticked — the triggers above were NOT injected into the prompt. '}
            Change the weights on the left and run again to add a variant.
          </p>

          {payload ? (
            <button type="button" onClick={() => onSaveBest?.(payload)} disabled={saving}
              className="rounded-lg border border-amber-400/40 bg-amber-400/10 px-2 py-1 text-[0.6875rem] font-semibold text-amber-200 disabled:opacity-40">
              {saving ? 'Saving…' : '★ Save these weights as the best setting'}
            </button>
          ) : (
            <p className="m-0 text-content-subtle text-[0.625rem]">
              This run predates the stack view, so it did not record which dataset each
              stacked LoRA came from — relaunch the stack to be able to pin its weights.
            </p>
          )}
          {savedAt && (
            <p className="m-0 text-emerald-300 text-[0.625rem]" role="status">
              ★ Saved — this stack and its weights are now the best setting.
            </p>
          )}
        </>
      )}
    </div>
  );
}
