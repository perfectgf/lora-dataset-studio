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
import SliderLock, { useSliderLock } from './SliderLock';

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
  // The lock itself lives in SliderLock: one implementation of "locked by
  // default, one padlock, remembered per slider", worn here with this
  // component's own label row and worn elsewhere with somebody else's.
  const { locked, toggle, rangeProps } = useSliderLock(storageKey);

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
          <SliderLock locked={locked} onToggle={toggle} label={label} />
        </div>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={safeValue}
        onChange={(e) => { if (!locked) onChange(e); }}
        {...rangeProps}
        className={`w-full ${accent} ${rangeProps.className}`}
      />
      <div className="flex justify-between text-content-muted text-[0.6875rem] mt-0.5">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}
