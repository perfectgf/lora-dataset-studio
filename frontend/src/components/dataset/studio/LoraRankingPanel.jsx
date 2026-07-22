// react-frontend/src/components/dataset/studio/LoraRankingPanel.jsx
/**
 * Panneau « 🏆 Classement LoRA » : alimenté par `data.lora_ranking` (déjà trié
 * côté backend). Affiche, par LoRA, likes/dislikes/net/score Wilson. Repliable,
 * style calqué sur BestPerModelList (bg-surface-raised, header uppercase).
 *
 * a11y : la position est donnée par un numéro de rang (pas par la couleur seule) ;
 * le delta net porte un signe explicite (+/−).
 */
import { useState } from 'react';

export default function LoraRankingPanel({ ranking }) {
  const [open, setOpen] = useState(true);
  if (!Array.isArray(ranking) || ranking.length === 0) return null;

  return (
    <div className="flex flex-col gap-1.5 rounded-lg border border-border bg-surface-raised px-3 py-2">
      <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open}
        className="flex items-center gap-2 text-left text-content-muted text-[0.625rem] uppercase">
        <span aria-hidden>{open ? '▾' : '▸'}</span>
        🏆 LoRA Ranking ({ranking.length})
      </button>
      {open && (
        <ol className="flex flex-col gap-1 m-0 p-0 list-none">
          {ranking.map((r, i) => {
            const net = r.net ?? ((r.likes || 0) - (r.dislikes || 0));
            return (
              <li key={r.dataset_id}
                className="flex items-center gap-2 flex-wrap text-[0.6875rem] rounded px-1.5 py-1 bg-app/30">
                <span className="text-content-subtle tabular-nums w-5 text-right" aria-label={`Rank ${i + 1}`}>#{i + 1}</span>
                <span className="text-content font-medium truncate max-w-[160px]" title={`${r.lora_label} — ${r.dataset_name || ''}`}>
                  {r.lora_label}
                </span>
                <span className="text-green-300 tabular-nums" aria-label={`${r.likes || 0} likes`}>👍 {r.likes || 0}</span>
                <span className="text-red-300 tabular-nums" aria-label={`${r.dislikes || 0} dislikes`}>👎 {r.dislikes || 0}</span>
                <span className="text-content-subtle tabular-nums" title="Likes minus dislikes">
                  net {net > 0 ? `+${net}` : net}
                </span>
                {r.wilson != null && (
                  <span className="ml-auto text-content-muted tabular-nums" title="Confidence score (Wilson lower bound)">
                    {Math.round(r.wilson * 100)}%
                  </span>
                )}
              </li>
            );
          })}
        </ol>
      )}
      {open && <HowVotingWorks />}
    </div>
  );
}

/** The ranking is not "most likes wins", and that surprises people: a config with
 *  6👍4👎 sits BELOW one with 2👍0👎. Explaining it where the numbers are shown is
 *  the only place it lands — nobody goes looking for it in the docs. */
export function HowVotingWorks() {
  const [open, setOpen] = useState(false);
  return (
    <details className="mt-1 rounded border border-border bg-app"
      open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary className="cursor-pointer select-none px-2 py-1 text-[0.625rem] text-content-subtle hover:text-content">
        How does the ranking work?
      </summary>
      <div className="flex flex-col gap-1.5 px-2 pb-2 text-[0.6875rem] leading-relaxed text-content-muted">
        <p className="m-0">
          You rate each generated image 👍 or 👎. Ratings are grouped per
          <strong> configuration</strong> — the LoRA and its strength, plus the base model,
          aspect, CFG and steps — so what is being judged is a whole recipe, not one picture.
          Images that failed to generate are left out entirely: a broken config cannot be
          judged, and counting it would distort the result.
        </p>
        <p className="m-0">
          The percentage is <strong>not</strong> the share of 👍. It is the low end of the
          plausible range for that share, given how many votes you cast — so it rewards a
          high approval rate <em>and</em> enough votes to believe it. That is why a config
          with <strong>2👍/2</strong> (34%) outranks one with <strong>6👍4👎</strong> (31%),
          and <strong>5👍/5</strong> (57%) outranks <strong>2👍/2</strong>.
        </p>
        <p className="m-0">
          Ranking on raw 👍−👎 would just favour whatever you tested the most, and ranking on
          the plain rate would put a single lucky vote on top. Below 3 votes a configuration is
          flagged low-confidence: it is a hint, not a verdict — keep voting to settle it.
        </p>
      </div>
    </details>
  );
}
