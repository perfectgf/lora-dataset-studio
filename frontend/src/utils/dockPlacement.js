/** Where a bottom-left dock may sit, per route.
 *
 * Some screens already own the bottom of the window with a fixed bar. The Test
 * Studio's is the loud one: `StudioActionBar` is `fixed bottom-0 inset-x-0
 * z-[9960]` with an opaque background, so anything at `bottom-4` with an
 * ordinary z-index is not merely overlapped — it is invisible AND unclickable,
 * because the clicks land on the bar.
 *
 * This is not a new discovery. `StudioActionBar`'s own header still says the
 * old jobs FAB "est relevé au-dessus via PAGES_WITH_BOTTOM_BAR ('/studio')" —
 * that constant is gone from the codebase, the comment outlived it, and the
 * next thing to occupy that corner (the generation-queue dock, GitHub #44)
 * walked straight into the same wall. So the rule gets written down again,
 * here, with a test, instead of living in a comment about a deleted symbol.
 *
 * Raising the dock ABOVE the bar is the wrong answer: it would cover the Run
 * button, which is the whole point of that bar. It moves up instead.
 */

// Routes whose screen carries a fixed bottom bar. Both StudioPage routes do —
// they render the same shell (App.jsx: '/studio' and '/dataset/studio/:id').
export const ROUTES_WITH_BOTTOM_BAR = ['/studio', '/dataset/studio'];

/** Tailwind class for the dock's bottom offset on `pathname`. */
export function dockBottomClass(pathname) {
  const path = String(pathname || '');
  const covered = ROUTES_WITH_BOTTOM_BAR.some(
    (route) => path === route || path.startsWith(`${route}/`),
  );
  // bottom-20 = 5rem, clear of the bar's ~3rem plus its border and the dock's
  // own breathing room.
  return covered ? 'bottom-20' : 'bottom-4';
}
