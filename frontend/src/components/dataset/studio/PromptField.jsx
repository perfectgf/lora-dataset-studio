import RecentPrompts from './RecentPrompts';

// Champ prompt de test : textarea + bouton « ↺ défaut » + prompts récents.
// Extrait behavior-preserving de LoraTestStudio.jsx (bloc « Prompt de test »).
// `value` = effectivePrompt, `placeholder` = d.prompt, `isCustom` = prompt édité ≠ défaut.
// Le rendu de <RecentPrompts> reste conditionné à la présence de d.recent_prompts :
// on ne passe `recentPrompts` que si la liste est non vide.
export default function PromptField({ value, placeholder, onChange, onReset, isCustom, recentPrompts, datasetId, onDeletePrompt }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-content-muted text-[0.625rem] uppercase">Test prompt</span>
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
    </div>
  );
}
