// Sélecteur multi-toggle des strengths à balayer.
// Extrait behavior-preserving de LoraTestStudio.jsx (bloc « Strengths »).
export default function StrengthPicker({ choices, selected, onToggle, fmt }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-content-muted text-[0.625rem] uppercase">Strengths</span>
      <div className="flex gap-2 flex-wrap">
        {choices.map((s) => (
          <button key={s} type="button" onClick={() => onToggle(s)}
            aria-pressed={selected.includes(s)}
            className={`px-2.5 py-1 rounded-lg border text-[0.75rem] tabular-nums transition-colors ${
              selected.includes(s)
                ? 'border-purple-400/60 bg-purple-500/20 text-purple-200 font-semibold'
                : 'border-border bg-surface text-content-muted'}`}>
            {fmt(s)}
          </button>
        ))}
      </div>
    </div>
  );
}
