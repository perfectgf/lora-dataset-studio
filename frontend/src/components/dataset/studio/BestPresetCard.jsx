// Carte « Meilleur réglage (selon tes votes) » — preset temps réel d'après les votes.
// Extrait behavior-preserving de LoraTestStudio.jsx (bloc {d.best_preset && (...)}).
// Contrat (spec §6) : BestPresetCard({ preset, onMemorize, fmt }).
import { useI18n } from '../../../i18n/I18nContext';

export default function BestPresetCard({ preset, onMemorize, fmt }) {
  const { t } = useI18n();
  if (!preset) return null;

  return (
    <div className="flex flex-col gap-1.5 rounded-lg border border-emerald-400/50 bg-emerald-400/10 px-3 py-2">
      <div className="flex items-center gap-2 flex-wrap">
        <span aria-hidden>🏆</span>
        <span className="text-content text-sm font-semibold">{t('studio.best.basedOnVotes')}</span>
        <span className="text-emerald-300 text-[0.6875rem] tabular-nums"
          title={t('studio.best.scoreTitle', { likes: preset.likes, dislikes: preset.dislikes, images: preset.images })}>
          score +{preset.score} (👍{preset.likes} 👎{preset.dislikes})
          {preset.like_rate != null ? ` · ${Math.round(preset.like_rate * 100)}% 👍 · ${t('studio.best.votes', { count: preset.voted })}` : ''}
        </span>
        {preset.low_confidence && (
          <span className="text-amber-300 text-[0.625rem] inline-flex items-center gap-1"
            title={t('studio.best.lowSampleTitle')}>
            ⚠ {t('studio.best.lowSample')}
          </span>
        )}
        <button type="button" onClick={() => onMemorize(preset)}
          title={t('studio.best.saveTitle')}
          className="ml-auto px-3 py-1.5 rounded-lg border border-amber-400/50 bg-amber-400/10 text-amber-200 text-xs font-semibold">
          ★ {t('common.save')}
        </button>
      </div>
      <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[0.6875rem]">
        <span className="text-content-subtle">LoRA</span>
        <span className="text-content">{preset.label}</span>
        <span className="text-content-subtle">{t('studio.strength.title')}</span>
        <span className="text-content tabular-nums">{fmt(preset.strength)}</span>
        {preset.z_model_label && (<><span className="text-content-subtle">{t('studio.best.model')}</span>
          <span className="text-content">{preset.z_model_label}</span></>)}
        {preset.aspect && (<><span className="text-content-subtle">{t('studio.generation.format')}</span>
          <span className="text-content tabular-nums">{preset.aspect}</span></>)}
        {preset.cfg != null && (<><span className="text-content-subtle">CFG</span>
          <span className="text-content tabular-nums">{fmt(preset.cfg)}</span></>)}
        {preset.steps != null && (<><span className="text-content-subtle">{t('studio.best.steps')}</span>
          <span className="text-content tabular-nums">{preset.steps}</span></>)}
        {preset.seed != null && (<><span className="text-content-subtle">{t('studio.seed.seed')}</span>
          <span className="text-content tabular-nums">{preset.seed}</span></>)}
        {preset.prompt && (<><span className="text-content-subtle">{t('studio.prompt.title')}</span>
          <span className="text-content break-words">{preset.prompt}</span></>)}
      </div>
    </div>
  );
}
