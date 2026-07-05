/**
 * ResolutionSelector — output-resolution tier for Z-Image / SD3-family gens.
 * The format selector fixes the RATIO; this picks the SIZE. Mirrors the backend
 * app/utils/resolution.py (same ratios / tiers / ÷16 / 1536 cap) so the px shown
 * here match what's generated.
 *
 * Props:
 *   value        - 'fast'|'standard'|'hq'|'max'
 *   aspectRatio  - current format key (to display the resulting WxH)
 *   onChange     - (value) => void
 */

const RATIOS = {
  square: [1, 1], landscape: [4, 3], portrait: [3, 4], widescreen: [16, 9],
  tall: [9, 16], photo: [3, 2], phototall: [2, 3], ultrawide: [21, 9],
};
const TIERS = [
  { value: 'fast', label: 'Fast', mp: 0.7 },
  { value: 'standard', label: 'Standard', mp: 1.0 },
  { value: 'hq', label: 'HQ', mp: 1.3 },
  { value: 'max', label: 'Max', mp: 1.6 },
];
const CAP = 1536, FLOOR = 512, MULT = 16;
const snap = (v) => Math.max(FLOOR, Math.round(v / MULT) * MULT);

// `maxLongSide` (optionnel) : plafond par MODÈLE — SDXL casse au-delà de ~1 Mpx,
// donc le mode SDXL passe 1024 et l'affichage colle aux dimensions réellement
// générées (le backend applique la même re-borne).
export function tierDims(aspectRatio, mp, maxLongSide) {
  const [rw, rh] = RATIOS[aspectRatio] || RATIOS.square;
  const r = rw / rh;
  let h = Math.sqrt((mp * 1e6) / r);
  let w = r * h;
  const cap = Math.min(CAP, maxLongSide || CAP);
  const longest = Math.max(w, h);
  if (longest > cap) { const s = cap / longest; w *= s; h *= s; }
  return [snap(w), snap(h)];
}

export default function ResolutionSelector({ value = 'standard', aspectRatio = 'square', onChange, maxLongSide }) {
  return (
    <div className="grid grid-cols-4 gap-2">
      {TIERS.map((t) => {
        const selected = value === t.value;
        const [w, h] = tierDims(aspectRatio, t.mp, maxLongSide);
        return (
          <button
            key={t.value}
            type="button"
            onClick={() => onChange?.(t.value)}
            aria-pressed={selected}
            className={`flex flex-col items-center gap-0.5 py-2 px-1.5 rounded-[10px] border cursor-pointer transition-all duration-150
              ${selected
                ? 'border-primary/70 bg-primary/15 text-white'
                : 'border-white/10 bg-white/[0.04] text-content-muted'
              }`}
          >
            <span className="text-[0.6875rem] font-semibold">{t.label}</span>
            <span className="text-[0.625rem] opacity-60 tabular-nums">{w}×{h}</span>
          </button>
        );
      })}
    </div>
  );
}
