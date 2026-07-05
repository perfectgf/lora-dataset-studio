// Contrôles de seed : affichage seed + 🎲 re-roll + 🔒/🔓 verrou + ×N gén/config + compteur.
// Extrait behavior-preserving de LoraTestStudio.jsx (barre seed/lock/×N/compteur).
// IMPORTANT a11y : le compteur d'images N'A PAS d'aria-live (correctif déjà acté) —
// il se recalcule à chaque clic de config, une région live le ré-annoncerait sans cesse.
export default function SeedControls({ seed, seedLocked, onReroll, onToggleLock, genCount, onGenCount, total, batchMult = 1, fmt }) {
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="text-content-subtle text-[0.6875rem] tabular-nums">
        seed <code className="text-content-muted">{seed}</code>
      </span>
      <button type="button" onClick={onReroll}
        className="px-2 py-0.5 rounded bg-surface text-content-muted text-[0.6875rem]"
        title="Re-roll the seed manually">
        🎲 re-roll
      </button>
      <button type="button" onClick={onToggleLock}
        aria-pressed={seedLocked}
        className={`px-2 py-0.5 rounded text-[0.6875rem] ${seedLocked ? 'bg-indigo-500/20 border border-indigo-400/40 text-indigo-200' : 'bg-surface text-content-muted'}`}
        title={seedLocked ? 'Seed locked: same seed on every test (repro)' : 'Auto seed: new seed on every "Run test"'}>
        {seedLocked ? '🔒 seed' : '🔓 auto'}
      </button>
      <label className="flex items-center gap-1 text-[0.6875rem] text-content-muted"
        title="Number of images generated per config (different seeds) — batch">
        ×
        <select value={genCount} onChange={(e) => onGenCount(Number(e.target.value))}
          className="px-1 py-0.5 rounded bg-surface border border-border text-content text-[0.6875rem]">
          {[1, 2, 3, 4].map((n) => <option key={n} value={n}>{n}</option>)}
        </select>
        gen/config
      </label>
      {/* Pas d'aria-live : ce compteur se recalcule à chaque clic de config
          → une région live le ré-annoncerait sans cesse (verbosité parasite). */}
      <span className="text-[0.6875rem] tabular-nums text-content-subtle"
        title={batchMult > 1 ? `Includes the ⚖ batch axis: each config runs once without and once with each batch-checked LoRA (×${batchMult})` : undefined}>
        {total * genCount} image(s) (~{Math.ceil(total * genCount * 12 / 60)} min)
        {batchMult > 1 && <span className="text-amber-300"> · ⚖ ×{batchMult}</span>}
      </span>
    </div>
  );
}
