import { useState } from 'react';
import RecentPrompts from './RecentPrompts';
import DescribeImageModal from './DescribeImageModal';
import { useI18n } from '../../../i18n/I18nContext';

// Champ prompt de test : textarea + bouton « ↺ défaut » + « 🔎 Describe » + prompts récents.
// Extrait behavior-preserving de LoraTestStudio.jsx (bloc « Prompt de test »).
// `value` = effectivePrompt, `placeholder` = d.prompt, `isCustom` = prompt édité ≠ défaut.
// Le rendu de <RecentPrompts> reste conditionné à la présence de d.recent_prompts :
// on ne passe `recentPrompts` que si la liste est non vide.
export default function PromptField({ value, placeholder, onChange, onReset, isCustom, recentPrompts, datasetId, onDeletePrompt }) {
  const { t } = useI18n();
  const [describeOpen, setDescribeOpen] = useState(false);
  // A described prompt replaces the field; if the user already typed one, confirm
  // before clobbering it (never silently discard their text).
  const applyDescription = (text) => {
    if (value && value.trim()
      && !window.confirm(t('studio.prompt.replaceConfirm'))) return;
    onChange(text);
  };
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between gap-2">
        <span className="text-content-muted text-[0.625rem] uppercase">{t('studio.prompt.title')}</span>
        <button type="button" onClick={() => setDescribeOpen(true)}
          title={t('studio.prompt.describeTitle')}
          className="px-2 py-0.5 rounded border border-border bg-surface text-content-subtle text-[0.625rem] hover:text-content">
          🔎 {t('studio.prompt.describe')}
        </button>
      </div>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={5}
        placeholder={placeholder}
        aria-label={t('studio.prompt.label')}
        className="w-full rounded-lg border border-border bg-surface px-2 py-1.5 text-[0.75rem] text-content resize-y focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-400"
      />
      {isCustom && (
        <button type="button" onClick={onReset}
          className="self-start px-2 py-0.5 rounded bg-surface text-content-subtle text-[0.625rem] hover:text-content"
          title={t('studio.prompt.resetTitle')}>
          ↺ {t('studio.prompt.default')}
        </button>
      )}
      {Array.isArray(recentPrompts) && recentPrompts.length > 0 && (
        <RecentPrompts items={recentPrompts} datasetId={datasetId} selectedPrompt={value}
          onPick={onChange} onDelete={onDeletePrompt} />
      )}
      <DescribeImageModal open={describeOpen} onClose={() => setDescribeOpen(false)}
        onResult={applyDescription} />
    </div>
  );
}
