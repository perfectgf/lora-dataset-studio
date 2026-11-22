/**
 * The page numbers a pager shows, pages counted from 1.
 *
 * Always the first and the last page, the current page with `siblings`
 * neighbours on each side, and 0 wherever pages are skipped (the pager draws an
 * ellipsis there). The list has the same length on every page of a long list
 * (2 × siblings + 5), so the buttons do not move under the pointer while pages
 * are turned; a list short enough shows every page.
 *
 *   pagerSlots(1, 130, 1)  → [1, 2, 3, 4, 5, 0, 130]
 *   pagerSlots(60, 130, 1) → [1, 0, 59, 60, 61, 0, 130]
 *   pagerSlots(3, 4, 1)    → [1, 2, 3, 4]
 */
export function pagerSlots(current, pages, siblings = 1) {
  const count = Math.max(1, Math.floor(Number(pages)) || 1);
  const reach = Math.max(0, Math.floor(Number(siblings)) || 0);
  const page = Math.min(Math.max(1, Math.floor(Number(current)) || 1), count);
  const run = (from, to) => Array.from({ length: Math.max(0, to - from + 1) }, (_, i) => from + i);
  if (count <= reach * 2 + 5) return run(1, count);
  // The window of numbers around the current page, pushed inward at both ends
  // so that it always holds 2 × reach + 1 numbers.
  const start = Math.max(Math.min(page - reach, count - reach * 2 - 2), 3);
  const end = Math.min(Math.max(page + reach, reach * 2 + 3), count - 2);
  return [1, start > 3 ? 0 : 2, ...run(start, end), end < count - 2 ? 0 : count - 1, count];
}
