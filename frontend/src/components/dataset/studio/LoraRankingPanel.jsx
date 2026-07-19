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
import { useI18n } from '../../../i18n/I18nContext';

export default function LoraRankingPanel({ ranking }) {
  const [open, setOpen] = useState(true);
  const { t } = useI18n();
  if (!Array.isArray(ranking) || ranking.length === 0) return null;

  return (
    <div className="flex flex-col gap-1.5 rounded-lg border border-border bg-surface-raised px-3 py-2">
      <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open}
        className="flex items-center gap-2 text-left text-content-muted text-[0.625rem] uppercase">
        <span aria-hidden>{open ? '▾' : '▸'}</span>
        🏆 {t('studio.ranking.title', { count: ranking.length })}
      </button>
      {open && (
        <ol className="flex flex-col gap-1 m-0 p-0 list-none">
          {ranking.map((r, i) => {
            const net = r.net ?? ((r.likes || 0) - (r.dislikes || 0));
            return (
              <li key={r.dataset_id}
                className="flex items-center gap-2 flex-wrap text-[0.6875rem] rounded px-1.5 py-1 bg-app/30">
                <span className="text-content-subtle tabular-nums w-5 text-right" aria-label={t('studio.ranking.rank', { rank: i + 1 })}>#{i + 1}</span>
                <span className="text-content font-medium truncate max-w-[160px]" title={`${r.lora_label} — ${r.dataset_name || ''}`}>
                  {r.lora_label}
                </span>
                <span className="text-green-300 tabular-nums" aria-label={t('studio.ranking.likes', { count: r.likes || 0 })}>👍 {r.likes || 0}</span>
                <span className="text-red-300 tabular-nums" aria-label={t('studio.ranking.dislikes', { count: r.dislikes || 0 })}>👎 {r.dislikes || 0}</span>
                <span className="text-content-subtle tabular-nums" title={t('studio.ranking.netTitle')}>
                  {t('studio.ranking.net')} {net > 0 ? `+${net}` : net}
                </span>
                {r.wilson != null && (
                  <span className="ml-auto text-content-muted tabular-nums" title={t('studio.ranking.confidence')}>
                    {Math.round(r.wilson * 100)}%
                  </span>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
