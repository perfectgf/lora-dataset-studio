/**
 * ShotIllustration — parametric inline-SVG pictogram for a dataset shot.
 *
 * Draws a stylized mannequin cropped to the shot framing (face / bust / body /
 * back), with the head orientation derived from the FRENCH catalog label
 * (profil / 3-4 gauche-droite) and the photo aspect ratio (paysage / vertical /
 * cinéma) rendered as the outer frame. Everything uses `currentColor`, so the
 * pictogram inherits the state color of its parent chip (amber = quota deficit,
 * indigo = selected) and doubles as a state indicator.
 */

// First matching entry wins — keep the most specific words first.
const CONTEXT_EMOJI = [
  [/fenetre|fenêtre/i, '🪟'],
  [/studio/i, '💡'],
  [/golden/i, '🌇'],
  [/plage/i, '🏖️'],
  [/cafe|café|terrasse/i, '☕'],
  [/urbain/i, '🏙️'],
  [/champ/i, '🌾'],
  [/exterieur|extérieur/i, '🌳'],
  [/marche/i, '🚶'],
  [/assis/i, '🪑'],
  [/soiree|soirée/i, '👗'],
  [/veste/i, '🧥'],
  [/rire/i, '😄'],
  [/sourire/i, '🙂'],
  [/serieux|sérieux/i, '😐'],
  [/doux/i, '😌'],
  [/surprise/i, '😮'],
  [/regard haut/i, '👆'],
  [/regard bas/i, '👇'],
];

export function contextEmoji(label = '') {
  const hit = CONTEXT_EMOJI.find(([re]) => re.test(label));
  return hit ? hit[1] : null;
}

function orientationOf(label = '') {
  if (/profil gauche/i.test(label)) return 'profileL';
  if (/profil droite/i.test(label)) return 'profileR';
  if (/3\/4 gauche/i.test(label)) return 'threeQL';
  if (/3\/4 droite/i.test(label)) return 'threeQR';
  return 'front';
}

function aspectOf(label = '') {
  if (/cinema|cinéma/i.test(label)) return 'cine';
  if (/paysage|large/i.test(label)) return 'wide';
  if (/vertical/i.test(label)) return 'tall';
  return 'square';
}

// Outer frame rect per aspect (viewBox 0 0 32 32).
const FRAME = {
  square: { x: 5, y: 3, w: 22, h: 26 },
  wide: { x: 1.5, y: 7, w: 29, h: 18 },
  cine: { x: 1.5, y: 9.5, w: 29, h: 13 },
  tall: { x: 8.5, y: 1.5, w: 15, h: 29 },
};

/** Head circle + orientation cue (nose notch for profiles, offset eye dots for 3/4). */
function Head({ cx, cy, r, orient, back }) {
  const nose = r * 0.62;
  return (
    <g>
      <circle cx={cx} cy={cy} r={r} fill="currentColor" />
      {back && (
        // Hair bun seen from behind — the "back view" cue.
        <circle cx={cx} cy={cy - r * 0.9} r={r * 0.38} fill="currentColor" />
      )}
      {!back && orient === 'profileL' && (
        <path d={`M ${cx - r} ${cy - nose * 0.4} l ${-nose} ${nose * 0.7} l ${nose} ${nose * 0.7} Z`} fill="currentColor" />
      )}
      {!back && orient === 'profileR' && (
        <path d={`M ${cx + r} ${cy - nose * 0.4} l ${nose} ${nose * 0.7} l ${-nose} ${nose * 0.7} Z`} fill="currentColor" />
      )}
      {!back && (orient === 'threeQL' || orient === 'threeQR') && (
        // Two "eye" cutouts shifted to the turned side.
        <g fill="var(--shot-bg, #0b0d12)" opacity="0.9">
          <circle cx={cx + (orient === 'threeQL' ? -r * 0.55 : r * 0.15)} cy={cy - r * 0.15} r={r * 0.14} />
          <circle cx={cx + (orient === 'threeQL' ? -r * 0.15 : r * 0.55)} cy={cy - r * 0.15} r={r * 0.14} />
        </g>
      )}
      {!back && orient === 'front' && (
        <g fill="var(--shot-bg, #0b0d12)" opacity="0.9">
          <circle cx={cx - r * 0.32} cy={cy - r * 0.15} r={r * 0.14} />
          <circle cx={cx + r * 0.32} cy={cy - r * 0.15} r={r * 0.14} />
        </g>
      )}
    </g>
  );
}

export default function ShotIllustration({ framing = 'face', label = '', className = '' }) {
  const orient = framing === 'back' ? 'front' : orientationOf(label);
  const aspect = aspectOf(label);
  const f = FRAME[aspect];
  const back = framing === 'back';
  // Vertical center of the frame; silhouettes are laid out relative to it.
  const cy = f.y + f.h / 2;
  const cx = 16;

  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden="true" focusable="false">
      {/* Photo frame — shows the target aspect ratio. */}
      <rect x={f.x} y={f.y} width={f.w} height={f.h} rx="2"
        fill="none" stroke="currentColor" strokeWidth="1.1" opacity="0.45" />
      {framing === 'face' && (
        <g>
          <Head cx={cx} cy={cy - f.h * 0.08} r={Math.min(f.h, f.w) * 0.26} orient={orient} back={false} />
          {/* Shoulders entering the bottom of the frame. */}
          <path d={`M ${cx - f.w * 0.32} ${f.y + f.h} q ${f.w * 0.32} ${-f.h * 0.38} ${f.w * 0.64} 0 Z`}
            fill="currentColor" opacity="0.85" />
        </g>
      )}
      {(framing === 'bust') && (
        <g>
          <Head cx={cx} cy={f.y + f.h * 0.28} r={Math.min(f.h, f.w) * 0.17} orient={orient} back={false} />
          <path d={`M ${cx - f.w * 0.26} ${f.y + f.h} q 0 ${-f.h * 0.5} ${f.w * 0.2} ${-f.h * 0.52}
                    h ${f.w * 0.12} q ${f.w * 0.2} ${f.h * 0.02} ${f.w * 0.2} ${f.h * 0.52} Z`}
            fill="currentColor" opacity="0.85" />
        </g>
      )}
      {(framing === 'body' || framing === 'back') && (
        <g>
          <Head cx={cx} cy={f.y + f.h * 0.16} r={Math.min(f.h, f.w) * 0.105} orient={orient} back={back} />
          {/* Torso */}
          <path d={`M ${cx - f.w * 0.13} ${f.y + f.h * 0.30} h ${f.w * 0.26}
                    l ${f.w * 0.02} ${f.h * 0.26} h ${-f.w * 0.3} Z`}
            fill="currentColor" opacity="0.85" />
          {/* Legs */}
          <path d={`M ${cx - f.w * 0.10} ${f.y + f.h * 0.56} h ${f.w * 0.08} l ${f.w * 0.01} ${f.h * 0.36} h ${-f.w * 0.07} Z`}
            fill="currentColor" opacity="0.85" />
          <path d={`M ${cx + f.w * 0.02} ${f.y + f.h * 0.56} h ${f.w * 0.08} l ${-f.w * 0.01} ${f.h * 0.36} h ${-f.w * 0.07} Z`}
            fill="currentColor" opacity="0.85" />
        </g>
      )}
    </svg>
  );
}
