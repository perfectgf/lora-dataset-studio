/**
 * 🔒 The padlock that guards one slider, and the hook behind it.
 *
 * Split out of LockableSlider so a dial with its OWN presentation can wear the
 * same guard: the video studio's Length ladder shows seconds and frames, its
 * Steps row carries an Auto chip, and neither fits a component that draws its
 * own label and captions. What must be identical across the app is the
 * BEHAVIOUR — locked by default, one padlock, the choice remembered — not the
 * markup around it.
 *
 * The lock guards the RANGE and nothing else. A small explicit control next to
 * it (Auto, a number field) is not what a scrolling thumb hits, and disabling
 * those too would make the padlock a mode instead of a guard.
 */
import { useCallback, useState } from 'react';
import { readLock, writeLock } from './sliderLockStorage';

/** `{locked, toggle, rangeProps}` — spread `rangeProps` on the <input>. */
export function useSliderLock(storageKey) {
  const [locked, setLocked] = useState(() => readLock(storageKey));
  const toggle = useCallback(() => {
    setLocked((prev) => writeLock(storageKey, !prev));
  }, [storageKey]);
  return {
    locked,
    toggle,
    rangeProps: {
      disabled: locked,
      // Even unlocked, a vertical swipe must scroll the page rather than drag
      // the dial: `pan-y` hands the browser every up/down gesture and leaves
      // only sideways ones to the slider. The global rule in index.css covers
      // every range; this repeats it where the guard is explicit.
      style: { touchAction: 'pan-y' },
      className: locked ? 'opacity-45 cursor-not-allowed' : '',
    },
  };
}

/** The padlock itself. Finger-sized below `lg`, untouched on a desktop. */
export default function SliderLock({ locked, onToggle, label }) {
  return (
    <button type="button" onClick={onToggle}
      aria-pressed={locked}
      aria-label={locked ? `Unlock ${label}` : `Lock ${label}`}
      title={locked ? `Unlock ${label} — it cannot be changed by accident`
        : `Lock ${label} against accidental changes`}
      className={`flex min-h-10 min-w-10 shrink-0 items-center justify-center rounded-md border text-xs transition-colors lg:min-h-0 lg:min-w-0 lg:h-[22px] lg:w-[22px] ${
        locked
          ? 'border-indigo-500/40 bg-indigo-500/20 text-indigo-300'
          : 'border-white/10 bg-white/5 text-content-muted'}`}>
      {locked ? '🔒' : '🔓'}
    </button>
  );
}
