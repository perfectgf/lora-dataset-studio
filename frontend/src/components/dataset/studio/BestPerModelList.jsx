// Meilleur réglage PAR MODÈLE (les votes varient selon le checkpoint).
// Extrait de LoraTestStudio.jsx (bloc d.best_per_model), rendu REPLIABLE + couleur
// corrigée : `bg-surface/60` rendait un panneau BLANC (surface = blanc, /60 force
// l'alpha à 0.6) → on utilise `bg-surface-raised` (surface sombre surélevée).
import { useState } from 'react';

export default function BestPerModelList({ items, breakdown, datasetId, onMemorize, fmt }) {
  const [open, setOpen] = useState(false); // replié par défaut (dépliable au besoin)
  if (!Array.isArray(items) || items.length === 0) return null;

  // Détail « générées/votées par base » regroupé par checkpoint (idée user :
  // voir où l'échantillon est mince, ex. testé 12× sur une base vs 3× sur une autre).
  const byCheckpoint = {};
  (breakdown || []).forEach((b) => { (byCheckpoint[b.checkpoint] ||= []).push(b); });

  return (
    <div className="flex flex-col gap-1.5 rounded-lg border border-border bg-surface-raised px-3 py-2">
      <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open}
        className="flex items-center gap-2 text-left text-content-muted text-[0.625rem] uppercase">
        <span aria-hidden>{open ? '▾' : '▸'}</span>
        Best setting per model ({items.length})
      </button>
      {open && items.map((m) => (
        <div key={m.checkpoint} className="flex flex-col gap-0.5">
          <div className="flex items-center gap-2 flex-wrap text-[0.6875rem]">
            {m.filename
              ? <img src={`/api/dataset/${datasetId}/img/${encodeURIComponent(m.filename)}`}
                  alt="" loading="lazy" className="w-8 h-10 object-cover rounded shrink-0" />
              : <span className="w-8 h-10 rounded bg-app/60 shrink-0" />}
            <span className="text-content font-medium truncate max-w-[150px]" title={m.label}>{m.label}</span>
            <span className="text-content-subtle">
              str {fmt(m.strength)}{m.cfg != null ? ` · cfg ${m.cfg}` : ''}{m.steps != null ? ` · ${m.steps}${m.steps2 != null ? '/' + m.steps2 : ''}st` : ''}{m.aspect ? ` · ${m.aspect}` : ''}
            </span>
            <span className="text-content-subtle tabular-nums">👍{m.likes}/{m.voted}</span>
            <button type="button" onClick={() => onMemorize(m)}
              title="Save this setting as the dataset's best"
              className="ml-auto px-2 py-0.5 rounded bg-amber-400/15 border border-amber-400/40 text-amber-200">★</button>
          </div>
          {byCheckpoint[m.checkpoint] && byCheckpoint[m.checkpoint].length > 0 && (
            <div className="flex flex-wrap gap-x-3 gap-y-0.5 pl-10 text-content-subtle text-[0.625rem]">
              {byCheckpoint[m.checkpoint].map((b) => (
                <span key={`${m.checkpoint}|${b.z_model || 'off'}`}
                  title={`${b.voted} voted out of ${b.images} generated`}>
                  {b.z_model_label || 'Official'} {b.voted}/{b.images}
                  {b.like_rate != null ? ` · ${Math.round(b.like_rate * 100)}%👍` : ''}
                </span>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
