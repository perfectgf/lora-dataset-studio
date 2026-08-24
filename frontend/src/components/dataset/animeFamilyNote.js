// The Anima pointer for anime datasets (PURE JS, JSX-free so node --test can
// import it).
//
// WHY A NOTE AND NOTHING MORE
// ---------------------------
// Marking a dataset "anime" describes WHAT it is. Preselecting the Anima family,
// or binding the two values, would turn that description into a launch failure:
// Anima is local-only and needs an up-to-date ai-toolkit, so on most machines the
// forced choice would refuse at launch time. And training an anime character on
// SDXL is a perfectly legitimate thing to want. So: no preselection, no binding,
// no blocked launch — one informational line the user is free to ignore.
//
// It stays quiet unless Anima is ACTUALLY runnable here (base-info.anima_supported,
// computed from the installed ai-toolkit). Recommending an option the reader
// cannot take is worse than saying nothing.

const ANIME_FAMILY_NOTE =
  'This dataset is an anime character; Anima trains on an anime base. '
  + 'Switching is optional — every family can train it.';

/** The note to show under the LoRA-type selector, or null. */
export function animeFamilyNote({ subjectType, trainType, animaSupported } = {}) {
  if (subjectType !== 'anime') return null;
  if (trainType === 'anima') return null;      // already there — nothing to point at
  if (animaSupported !== true) return null;    // unknown/older server/older ai-toolkit
  return ANIME_FAMILY_NOTE;
}
