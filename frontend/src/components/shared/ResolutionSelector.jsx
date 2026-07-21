/**
 * ResolutionSelector — output-resolution tier for Z-Image / SD3-family gens.
 * The format selector fixes the RATIO; this picks the SIZE. Mirrors the backend
 * app/utils/resolution.py (same ratios / tiers / caps / multiplier clamp / ÷16
 * round-half-up snap) so the px shown here match what's generated.
 *
 * The 4 tiers are the base PRESET; the multiplier slider (1.0–1.9) below them
 * enlarges the chosen preset linearly (W and H both × multiplier). Default 1.0 =
 * preset unchanged (backward-compatible). Past ~1.5× the base model can degrade
 * or OOM, so that portion is marked with a discreet warning tint (no hard block).
 *
 * Props:
 *   value              - 'fast'|'standard'|'hq'|'max'
 *   aspectRatio        - current format key (to display the resulting WxH)
 *   onChange           - (value) => void
 *   maxLongSide        - optional per-model preset cap (SDXL passes 1024)
 *   multiplier         - resolution multiplier, 1.0–1.9 (default 1.0)
 *   onMultiplierChange - (number) => void
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
const CAP = 1536;        // tier cap: max px per side of the PRESET (Z-Image safe band)
const ABS_CAP = 3072;    // absolute safety cap applied AFTER the multiplier (long side)
const FLOOR = 512, MULT = 16;
export const MULT_MIN = 1.0, MULT_MAX = 1.9, MULT_STEP = 0.1;
// Past this the base model (Krea/Z-Image) can soften/duplicate or OOM on the GPU.
const MULT_WARN = 1.5;
const snap = (v) => Math.max(FLOOR, Math.round(v / MULT) * MULT);
const clampMult = (m) => {
  const v = Number(m);
  return Number.isFinite(v) ? Math.max(MULT_MIN, Math.min(MULT_MAX, v)) : MULT_MIN;
};

// `maxLongSide` (optionnel) : plafond par MODÈLE — SDXL casse au-delà de ~1 Mpx,
// donc le mode SDXL passe 1024 et l'affichage colle aux dimensions réellement
// générées (le backend applique la même re-borne). `multiplier` (1.0–1.9) agrandit
// le PRESET après le cap de palier, borné ensuite au cap absolu 3072.
export function tierDims(aspectRatio, mp, maxLongSide, multiplier = 1) {
  const [rw, rh] = RATIOS[aspectRatio] || RATIOS.square;
  const r = rw / rh;
  let h = Math.sqrt((mp * 1e6) / r);
  let w = r * h;
  const cap = Math.min(CAP, maxLongSide || CAP);
  let longest = Math.max(w, h);
  if (longest > cap) { const s = cap / longest; w *= s; h *= s; }
  const m = clampMult(multiplier);
  w *= m; h *= m;
  longest = Math.max(w, h);
  if (longest > ABS_CAP) { const s = ABS_CAP / longest; w *= s; h *= s; }
  return [snap(w), snap(h)];
}

export default function ResolutionSelector({
  value = 'standard', aspectRatio = 'square', onChange, maxLongSide,
  multiplier = 1, onMultiplierChange,
}) {
  const m = clampMult(multiplier);
  const enlarged = m > 1.0001;
  const overTrain = m > MULT_WARN + 1e-9;
  // Base (preset) vs final size of the CURRENTLY selected tier, for the live readout.
  const cur = TIERS.find((t) => t.value === value) || TIERS[1];
  const [bw, bh] = tierDims(aspectRatio, cur.mp, maxLongSide, 1);
  const [fw, fh] = tierDims(aspectRatio, cur.mp, maxLongSide, m);
  return (
    <div className="flex flex-col gap-2">
      <div className="grid grid-cols-4 gap-2">
        {TIERS.map((t) => {
          const selected = value === t.value;
          // Each chip shows its FINAL size (preset × current multiplier).
          const [w, h] = tierDims(aspectRatio, t.mp, maxLongSide, m);
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

      {/* Multiplier — enlarges the chosen preset linearly. Default 1.0 = unchanged. */}
      <div className="flex flex-col gap-1">
        <div className="flex items-center justify-between text-[0.625rem] uppercase tracking-wide text-content-muted">
          <span>Resolution multiplier</span>
          <span className={`tabular-nums font-semibold ${overTrain ? 'text-amber-400' : 'text-content'}`}>
            ×{m.toFixed(1)}
          </span>
        </div>
        <input
          type="range"
          min={MULT_MIN} max={MULT_MAX} step={MULT_STEP} value={m}
          onChange={(e) => onMultiplierChange?.(clampMult(parseFloat(e.target.value)))}
          aria-label="Resolution multiplier"
          className={`w-full h-1.5 rounded-full appearance-none cursor-pointer ${overTrain ? 'accent-amber-500' : 'accent-primary'}`}
        />
        <span className="text-[0.625rem] text-content-muted/70 tabular-nums normal-case tracking-normal">
          {enlarged
            ? `${bw}×${bh} → ${fw}×${fh}`
            : `${bw}×${bh} · slide to enlarge past the training resolution`}
        </span>
        {overTrain && (
          <span className="text-[0.625rem] text-amber-400/90 normal-case tracking-normal leading-snug">
            Beyond ~1.5× the training resolution, Krea/Z-Image may soften, duplicate or OOM on the GPU.
          </span>
        )}
      </div>
    </div>
  );
}
