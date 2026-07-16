import { useState } from 'react';
import RecentPrompts from './RecentPrompts';
import DescribeImageModal from './DescribeImageModal';

// Champ prompt de test : textarea + bouton « ↺ défaut » + « 🔎 Describe » + prompts récents.
// Extrait behavior-preserving de LoraTestStudio.jsx (bloc « Prompt de test »).
// `value` = effectivePrompt, `placeholder` = d.prompt, `isCustom` = prompt édité ≠ défaut.
// Le rendu de <RecentPrompts> reste conditionné à la présence de d.recent_prompts :
// on ne passe `recentPrompts` que si la liste est non vide.
export default function PromptField({ value, placeholder, onChange, onReset, isCustom, recentPrompts, datasetId, onDeletePrompt }) {
  const [describeOpen, setDescribeOpen] = useState(false);
  // A described prompt replaces the field; if the user already typed one, confirm
  // before clobbering it (never silently discard their text).
  const applyDescription = (text) => {
    if (value && value.trim()
      && !window.confirm('Replace the current test prompt with the described one?')) return;
    onChange(text);
  };
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between gap-2">
        <span className="text-content-muted text-[0.625rem] uppercase">Test prompt</span>
        <button type="button" onClick={() => setDescribeOpen(true)}
          title="Describe an image into a test prompt (vision model)"
          className="px-2 py-0.5 rounded border border-border bg-surface text-content-subtle text-[0.625rem] hover:text-content">
          🔎 Describe
        </button>
      </div>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={5}
        placeholder={placeholder}
        aria-label="LoRA test prompt"
        className="w-full rounded-lg border border-border bg-surface px-2 py-1.5 text-[0.75rem] text-content resize-y focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-400"
      />
      {isCustom && (
        <button type="button" onClick={onReset}
          className="self-start px-2 py-0.5 rounded bg-surface text-content-subtle text-[0.625rem] hover:text-content"
          title="Revert to the default identity prompt">
          ↺ default
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
