/**
 * LockableSlider - a range slider that is LOCKED by default to prevent
 * accidental changes (esp. mobile scroll-drag / mistaps). A padlock toggles
 * editability; the choice is persisted per slider via `storageKey`.
 *
 * Same pattern as the prompt-builder strength sliders. Layout: label + value
 * (+ lock) on top, the range below, min/max captions underneath.
 *
 * Props:
 *   label       - string (also used for the aria-label)
 *   value       - number (controlled)
 *   onChange    - (event) => void  (caller parses e.target.value; only fired when unlocked)
 *   min,max,step- range bounds (strings or numbers)
 *   storageKey  - localStorage key for the lock state (e.g. 'loraStrengthLock')
 *   format      - (value) => displayed value (default: identity)
 *   accent      - tailwind accent class for the range (default 'accent-primary')
 */
import { useState } from 'react';

export default function LockableSlider({
  label,
  value,
  onChange,
  min,
  max,
  step,
  storageKey,
  format = (v) => v,
  accent = 'accent-primary',
}) {
  const [locked, setLocked] = useState(() => {
    try {
      const v = localStorage.getItem(storageKey);
      return v === null ? true : v === 'true';
    } catch {
      return true;
    }
  });

  const toggleLock = () => setLocked((prev) => {
    const next = !prev;
    try { localStorage.setItem(storageKey, String(next)); } catch {}
    return next;
  });

  // Garde-fou : une valeur non numérique (ex. la string "None" issue d'un param
  // stocké/restauré) sur un <input type="range"> déclenche le warning console
  // « The specified value None cannot be parsed, or is out of range » à CHAQUE
  // rendu. On retombe alors sur min (ou 0) → jamais "None" dans le DOM.
  const parsed = typeof value === 'number' ? value : parseFloat(value);
  const safeValue = Number.isFinite(parsed) ? parsed : (Number(min) || 0);

  return (
    <div>
      <div className="flex justify-between items-center mb-1.5">
        <span className="text-content-muted text-xs font-semibold uppercase tracking-wide">
          {label}
        </span>
        <div className="flex items-center gap-2">
          <span className="text-content-muted text-[0.8125rem] font-semibold">
            {format(safeValue)}
          </span>
          <button
            type="button"
            onClick={toggleLock}
            aria-label={locked ? `Unlock ${label}` : `Lock ${label}`}
            title={locked ? 'Unlock slider' : 'Lock slider'}
            className={`w-[22px] h-[22px] rounded-[5px] text-xs cursor-pointer flex items-center justify-center shrink-0 transition-all duration-150 ${
              locked
                ? 'bg-indigo-500/20 border border-indigo-500/40 text-indigo-300'
                : 'bg-white/5 border border-white/10 text-content-muted'
            }`}
          >
            {locked ? '🔒' : '🔓'}
          </button>
        </div>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={safeValue}
        disabled={locked}
        onChange={(e) => { if (!locked) onChange(e); }}
        className={`w-full ${accent} ${locked ? 'opacity-45 cursor-not-allowed' : ''}`}
      />
      <div className="flex justify-between text-content-muted text-[0.6875rem] mt-0.5">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}
