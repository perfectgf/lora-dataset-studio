// Bandeau « Réglages gagnants » : réglage persisté (best_settings).
// Extrait behavior-preserving de LoraTestStudio.jsx (bloc `{bs && (...)}`).
// `best` = d.best_settings ; `onClear` fourni par le parent (StudioShell).
export default function BestSettingsBanner({ best, onClear, fmt }) {
  if (!best) return null;
  return (
    <div className="flex items-center gap-2 flex-wrap rounded-lg border border-amber-400/40 bg-amber-400/10 px-3 py-2">
      <span aria-hidden>★</span>
      <span className="text-content text-sm">
        Best setting: <code className="text-amber-200">{best.lora_filename.split('\\').pop()}</code>
        {' '}@ <strong>{fmt(best.strength)}</strong>
      </span>
      <button type="button"
        onClick={() => { if (window.confirm('Delete the saved setting?')) onClear(); }}
        title="Delete this saved setting"
        aria-label="Delete the saved setting"
        className="ml-auto px-2 py-1.5 rounded-lg bg-red-500/15 border border-red-500/40 text-red-300 text-xs hover:bg-red-500/25">
        🗑
      </button>
    </div>
  );
}
