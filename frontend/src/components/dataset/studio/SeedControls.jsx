import { useI18n } from '../../../i18n/I18nContext';

// Contrôles de seed : affichage seed + 🎲 re-roll + 🔒/🔓 verrou + ×N gén/config + compteur.
// Extrait behavior-preserving de LoraTestStudio.jsx (barre seed/lock/×N/compteur).
// IMPORTANT a11y : le compteur d'images N'A PAS d'aria-live (correctif déjà acté) —
// il se recalcule à chaque clic de config, une région live le ré-annoncerait sans cesse.
export default function SeedControls({ seed, seedLocked, onReroll, onToggleLock, genCount, onGenCount, total, batchMult = 1, fmt }) {
  const { t } = useI18n();
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="text-content-subtle text-[0.6875rem] tabular-nums">
        {t('studio.seed.seed')} <code className="text-content-muted">{seed}</code>
      </span>
      <button type="button" onClick={onReroll}
        className="px-2 py-0.5 rounded bg-surface text-content-muted text-[0.6875rem]"
        title={t('studio.seed.rerollTitle')}>
        🎲 {t('studio.seed.reroll')}
      </button>
      <button type="button" onClick={onToggleLock}
        aria-pressed={seedLocked}
        className={`px-2 py-0.5 rounded text-[0.6875rem] ${seedLocked ? 'bg-indigo-500/20 border border-indigo-400/40 text-indigo-200' : 'bg-surface text-content-muted'}`}
        title={seedLocked ? t('studio.seed.lockedTitle') : t('studio.seed.autoTitle')}>
        {seedLocked ? `🔒 ${t('studio.seed.seed')}` : `🔓 ${t('studio.seed.auto')}`}
      </button>
      <label className="flex items-center gap-1 text-[0.6875rem] text-content-muted"
        title={t('studio.seed.countTitle')}>
        ×
        <select value={genCount} onChange={(e) => onGenCount(Number(e.target.value))}
          className="px-1 py-0.5 rounded bg-surface border border-border text-content text-[0.6875rem]">
          {[1, 2, 3, 4].map((n) => <option key={n} value={n}>{n}</option>)}
        </select>
        {t('studio.seed.perConfig')}
      </label>
      {/* Pas d'aria-live : ce compteur se recalcule à chaque clic de config
          → une région live le ré-annoncerait sans cesse (verbosité parasite). */}
      <span className="text-[0.6875rem] tabular-nums text-content-subtle"
        title={batchMult > 1 ? t('studio.seed.batchTitle', { count: batchMult }) : undefined}>
        {t('studio.seed.estimate', {
          images: total * genCount,
          minutes: Math.ceil(total * genCount * 12 / 60),
        })}
        {batchMult > 1 && <span className="text-amber-300"> · ⚖ ×{batchMult}</span>}
      </span>
    </div>
  );
}
