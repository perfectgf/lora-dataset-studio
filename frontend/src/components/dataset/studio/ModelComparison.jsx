// Comparaison ÉQUITABLE des modèles de base (z_model) selon les votes.
// Classé par Wilson lower bound (taux × confiance) côté backend → ne favorise PAS
// le modèle le plus testé (biais de volume). Affiche taux 👍 + n (générées/votées).
// Repliable, masqué s'il y a moins de 2 bases (rien à comparer).
import { useState } from 'react';
import { useI18n } from '../../../i18n/I18nContext';

export default function ModelComparison({ items }) {
  const [open, setOpen] = useState(false);
  const { t } = useI18n();
  if (!Array.isArray(items) || items.length < 2) return null;

  return (
    <div className="flex flex-col gap-1.5 rounded-lg border border-border bg-surface-raised px-3 py-2">
      <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open}
        className="flex items-center gap-2 text-left text-content-muted text-[0.625rem] uppercase">
        <span aria-hidden>{open ? '▾' : '▸'}</span>
        {t('studio.modelComparison.title', { count: items.length })}
      </button>
      {open && (
        <table className="w-full text-[0.6875rem]">
          <thead>
            <tr className="text-content-subtle text-left">
              <th className="font-normal py-0.5">{t('studio.best.model')}</th>
              <th className="font-normal text-right">👍 {t('studio.modelComparison.rate')}</th>
              <th className="font-normal text-right">{t('studio.modelComparison.voted')}</th>
              <th className="font-normal text-right">{t('studio.modelComparison.generated')}</th>
              <th className="font-normal text-right">{t('studio.ranking.net')}</th>
            </tr>
          </thead>
          <tbody>
            {items.map((m) => (
              <tr key={m.z_model || 'officiel'} className="border-t border-border">
                <td className="py-0.5 text-content truncate max-w-[140px]" title={m.z_model_label || t('studio.modelComparison.official')}>
                  {m.z_model_label || t('studio.modelComparison.official')}
                </td>
                <td className="text-right tabular-nums text-content">
                  {m.like_rate != null ? `${Math.round(m.like_rate * 100)}%` : '—'}
                  {m.low_confidence && m.voted > 0 && (
                    <span className="text-amber-400" title={t('studio.best.lowSampleTitle')}> ⚠</span>
                  )}
                </td>
                <td className="text-right tabular-nums text-content-subtle">{m.voted}</td>
                <td className="text-right tabular-nums text-content-subtle">{m.images}</td>
                <td className="text-right tabular-nums text-content-subtle">
                  {m.net > 0 ? `+${m.net}` : m.net}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
