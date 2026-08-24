/* When a deep link names a Settings SECTION, should the view move to it?

   Settings lays out as a two-column grid only from `lg`. Below that the search
   box and the section list stack ABOVE the panel, so `#/settings/engines` used
   to select the right section and leave the reader at the top of the page,
   looking at the rail — the section they asked for sitting off-screen. Every
   "Settings › Image engines →" link in the app landed like that on a phone.

   Kept as a pure decision so `node --test` can exercise it (it cannot parse
   JSX), and because the interesting part is WHEN not to scroll:
     - a `?focus=` deep link already scrolls to one exact field; two scrolls
       would fight, and the field is the better target;
     - a panel already on screen must not jump — that is the desktop case,
       where the grid puts the panel next to the rail;
     - nothing may move before the settings have loaded, since the panel is not
       rendered yet.  */


/* A ?focus= arrival is not over when the first scroll ends.

   MEASURED on the Engines section (400x800 headless, 2026-07-28): the deep link
   to `identity-prompt-klein-improve` rang the field and started scrolling, but
   panels ABOVE it finish rendering asynchronously (the composed-prompt preview
   fetches its text), so the document kept growing under the scroll. At t≈2 s the
   field was still 116 px BELOW the fold — and the ring had already expired. Only
   at t≈4 s did it come to rest in view, un-highlighted. The reader is then looking
   at the right screen with nothing pointing at anything.

   So the arrival re-checks itself while the highlight is lit, and re-scrolls ONLY
   when the field is not fully visible — a correction that fires unconditionally
   would fight a user who has already scrolled somewhere on purpose.

   Pure predicate, because node --test cannot parse the JSX that uses it. */
export function focusNeedsRescroll({ top, height, viewportHeight, margin = 24 } = {}) {
  if ([top, height, viewportHeight].some((v) => typeof v !== 'number')) return false
  if (viewportHeight <= 0) return false
  // Fully visible, with a little breathing room above (the sticky header) and below.
  return !(top >= margin && top + height <= viewportHeight - margin)
}

export function shouldScrollToSection({
  hasSection, hasFocus, loading, panelTop, viewportHeight,
} = {}) {
  if (!hasSection) return false      // bare /settings — no section was asked for
  if (hasFocus) return false         // the focus effect owns the scroll
  if (loading) return false          // the panel does not exist yet
  if (typeof panelTop !== 'number' || typeof viewportHeight !== 'number') return false
  // Already comfortably in view (desktop grid): leave the page alone.
  if (panelTop >= 0 && panelTop < viewportHeight / 2) return false
  return true
}
