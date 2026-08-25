import { runCost } from './runCost';
import { Dices } from 'lucide-react';

// Contrôles de seed : affichage seed + 🎲 re-roll + 🔒/🔓 verrou + ×N gén/config + compteur.
// Extrait behavior-preserving de LoraTestStudio.jsx (barre seed/lock/×N/compteur).
// IMPORTANT a11y : le compteur d'images N'A PAS d'aria-live (correctif déjà acté) —
// il se recalcule à chaque clic de config, une région live le ré-annoncerait sans cesse.
// `promptMult` : nombre de prompts cochés dans l'historique (axe 📝 lot). Déjà
// compté dans `total` — il n'est ici que pour NOMMER d'où vient le multiplicateur,
// sinon le compteur triple sans que rien à l'écran dise pourquoi.
export default function SeedControls({ seed, seedLocked, onReroll, onToggleLock, genCount, onGenCount, total, batchMult = 1, promptMult = 1, secondsPerImage = null }) {
  // ⏱ L'estimation était « ×12 s » en dur : vraie sur une 4090 en Z-Image Turbo,
  // fausse partout ailleurs. `secondsPerImage` est la médiane RÉELLE mesurée sur
  // cette machine (null tant que l'historique est trop court → le repli).
  const cost = runCost(total * genCount, secondsPerImage);
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="text-content-subtle text-[0.6875rem] tabular-nums">
        seed <code className="text-content-muted">{seed}</code>
      </span>
      <button type="button" onClick={onReroll}
        className="px-2 py-0.5 rounded bg-surface text-content-muted text-[0.6875rem]"
        title="Re-roll the seed manually">
        <Dices aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />re-roll
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
        title={[
          batchMult > 1 ? `Includes the ⚖ batch axis: each config runs once without and once with each batch-checked LoRA (×${batchMult})` : null,
          cost.measured
            ? `Estimated from the ${cost.secondsPerImage}s median of your recent test generations`
            : 'Rough estimate — not enough finished generations yet to measure this machine',
        ].filter(Boolean).join(' · ')}>
        {cost.cells} image(s) ({cost.measured ? '' : '~'}{cost.label})
        {batchMult > 1 && <span className="text-amber-300"> · ⚖ ×{batchMult}</span>}
        {promptMult > 1 && <span className="text-purple-300"> · 📝 ×{promptMult}</span>}
      </span>
    </div>
  );
}
