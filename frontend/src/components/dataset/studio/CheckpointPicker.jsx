import { useI18n } from '../../../i18n/I18nContext';

// Sélecteur des checkpoints à tester (cases à cocher multi-sélection).
// Extrait behavior-preserving de LoraTestStudio.jsx (bloc « Checkpoints à tester »).
export default function CheckpointPicker({ checkpoints, chosen, onToggle }) {
  const { t } = useI18n();
  return (
    <div className="flex flex-col gap-1">
      <span className="text-content-muted text-[0.625rem] uppercase">{t('studio.checkpoints.title')}</span>
      <div className="flex gap-2 flex-wrap">
        {checkpoints.map((c) => (
          <label key={c.filename} className="flex items-center gap-1.5 px-2 py-1 rounded-lg border border-border bg-surface cursor-pointer text-[0.75rem] text-content">
            <input type="checkbox" checked={chosen.includes(c.filename)}
              onChange={() => onToggle(c.filename)}
              aria-label={t('studio.checkpoints.test', { name: c.label })} />
            {c.label}
          </label>
        ))}
      </div>
    </div>
  );
}
