// Progressive-disclosure logic for the Strengths picker, extracted pure so it can
// be unit-tested without a DOM. The base chip row tops out at BASE_STRENGTH_MAX;
// anything above lives in the « + » extended row.
export const BASE_STRENGTH_MAX = 2.0;

// Should the extended (>2.0) strength row be shown even before the user clicks « + »?
// Yes as soon as a selected value sits above the base range — a selection must NEVER
// be hidden (e.g. reloading a recent prompt that carried extended strengths, or a
// persisted selection), so the extended zone force-opens to keep the chip visible.
export function hasExtendedSelection(selected) {
  return (selected || []).some((s) => Number(s) > BASE_STRENGTH_MAX);
}
