// Training-family badges — ONE source for every surface that names the family a
// LoRA was trained for (the Datasets library tiles and rows, the Studio's LoRA
// picker). The two used to keep hand-written copies of this table, with a
// comment on each asking the other to stay in sync; they drifted, which is how
// Krea kept the amber that became the app's accent colour.
//
// Colour rules, in order:
//   1. NEVER the accent. Amber (and `indigo`, which the Safelight theme remaps
//      onto the same ramp) belongs to controls — a label wearing it reads as
//      something you can press. That is what this table was doing wrong.
//   2. Never a hue already carrying meaning on the same tile: fuchsia and cyan
//      are the Concept / Style kind badges sitting right above these.
//   3. Distinct from each other at a glance, since they appear side by side on
//      a dataset trained for several families.
//
// The label is part of the contract too: a family shows its PRODUCT name
// ("FLUX.1", not "flux"), everywhere, or the same LoRA reads as two things.

const FAMILY = {
  zimage: ['Z-Image', 'border-sky-400/40 bg-sky-500/10 text-sky-300'],
  sdxl: ['SDXL', 'border-violet-400/40 bg-violet-500/10 text-violet-300'],
  // Was amber — moved to lime when amber became the accent. Lime reads apart
  // from FLUX.1's emerald: yellow-green against green-cyan.
  krea: ['Krea', 'border-lime-400/40 bg-lime-500/10 text-lime-300'],
  flux: ['FLUX.1', 'border-emerald-400/40 bg-emerald-500/10 text-emerald-300'],
  flux2klein: ['FLUX.2 Klein', 'border-rose-400/40 bg-rose-500/10 text-rose-300'],
  anima: ['Anima', 'border-teal-400/40 bg-teal-500/10 text-teal-300'],
};

const FALLBACK_CLASS = 'border-border bg-surface-raised text-content-muted';

/** [label, className] for a family id. An unknown family keeps its raw id and a
 *  neutral chip rather than borrowing another family's colour. */
export function familyBadge(family) {
  return FAMILY[family] || [family, FALLBACK_CLASS];
}

/** Just the label — for places that style themselves. */
export function familyLabel(family) {
  return (FAMILY[family] || [family])[0];
}

/** Just the chip classes. */
export function familyBadgeClass(family) {
  return (FAMILY[family] || [null, FALLBACK_CLASS])[1];
}

/** Every family this table knows, for tests and menus. */
export const FAMILY_IDS = Object.keys(FAMILY);
