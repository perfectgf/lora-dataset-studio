/**
 * The Bank workspace's LAYOUT state — the "structure B" half of the redesign:
 * a top bar, a filter rail beside the grid it filters, and the analysis passes
 * in a panel opened on demand.
 *
 * Why this is a .js and not part of the component: the rail's folding rules and
 * the facet-group model are the only genuinely NEW logic the redesign adds, and
 * `node --test` cannot parse JSX. Logic that lives in the .jsx is logic that
 * cannot be tested, so it lives here and the component stays a renderer.
 *
 * The problem the structure solves: the workspace used to be four zones stacked
 * vertically, so adjusting a filter meant scrolling up, clicking, and scrolling
 * back down to see what changed. On a bank of 20 000+ images that round trip is
 * the cost — not the click count.
 */

/* ── The rail ───────────────────────────────────────────────────────────────
   It FOLDS rather than overflows. At 400 px a 15-rem rail beside a grid leaves
   neither of them usable, so below the breakpoint the rail is a drawer the top
   bar opens; from `lg` up it is a column and the toggle only decides whether
   the grid gets that width back. */

/** Below this the rail cannot share the row with the grid — matches Tailwind's
 *  `sm`, which every other responsive decision on this screen already uses. */
export const RAIL_SIDE_BY_SIDE_PX = 1024;

/** Does the rail sit BESIDE the grid at this width, or over it as a drawer?
 *  A missing/absurd width answers "beside": the desktop layout is the one that
 *  degrades gracefully if it turns out to be wrong, whereas guessing "drawer"
 *  on a real desktop hides the filters behind a click nobody was told about. */
export function railIsColumn(width) {
  return !(typeof width === 'number' && Number.isFinite(width))
    || width >= RAIL_SIDE_BY_SIDE_PX;
}

/** Where the rail starts on a fresh visit. Open on a desktop — the filters are
 *  the point of the screen and hiding them by default would only teach people
 *  the old scroll habit somewhere new. Closed on a phone, where showing it
 *  means showing nothing else. */
export function railDefaultOpen(width) {
  return railIsColumn(width);
}

/* The rail's open/closed state is a SCREEN PREFERENCE, per the design note: it
   belongs with view state, never with the settings that describe what a dataset
   produces. Unlike the sort order it is NOT stored per bank — the sort that
   suits a 9 000-image dump genuinely differs from the one that suits a
   shortlist, but "do I want the filters showing" is about this screen and this
   monitor, and re-answering it per bank would be the friction, not the fix. */
export const RAIL_STORAGE_KEY = 'lds.bank.rail';

/** localStorage, or nothing at all: this module is imported by node tests, and
 *  a private-mode browser throws on ACCESS, not only on read. Injectable so the
 *  behaviour is testable without a DOM. */
function defaultStore() {
  try {
    return globalThis.localStorage || null;
  } catch { return null; }
}

/** The remembered rail state, or the width-appropriate default. Anything
 *  unreadable or unrecognised degrades to the default rather than to a guess:
 *  a corrupt preference must never be able to leave the screen with no way to
 *  reach the filters. */
export function loadRailOpen(width, store = defaultStore()) {
  try {
    const raw = store?.getItem(RAIL_STORAGE_KEY);
    if (raw === 'open') return true;
    if (raw === 'closed') return false;
  } catch { /* fall through to the default */ }
  return railDefaultOpen(width);
}

/** Remember the rail state. Storage failures are swallowed — a full quota must
 *  not break a toggle that otherwise worked. */
export function saveRailOpen(open, store = defaultStore()) {
  try {
    store?.setItem(RAIL_STORAGE_KEY, open ? 'open' : 'closed');
  } catch { /* private mode / quota — the session still has the right state */ }
  return !!open;
}

/* ── The facet groups ───────────────────────────────────────────────────────
   All of triage lives in the rail. The split is by HOW OFTEN a facet is
   reached, not by which pass produced it: Status and Quality are the everyday
   gesture, the six measured axes below them are the occasional one. Folding the
   occasional ones is what keeps the whole rail on one screen without a scroll.

   `id`s are stored in localStorage (which sections a user left open), so they
   are covered by the repo's rule on stored identifiers: never rename one
   without an alias path. */

/** Always visible, in rail order. */
export const RAIL_PRIMARY_GROUPS = ['status', 'quality', 'groups'];

/** Folded below, in rail order. */
export const RAIL_FOLDED_GROUPS = [
  'score', 'framing', 'medium', 'angle', 'resolution', 'origin',
];

export const RAIL_GROUPS = [...RAIL_PRIMARY_GROUPS, ...RAIL_FOLDED_GROUPS];

/**
 * Which rail groups render, given what the passes have actually produced.
 *
 * `availability` is a plain {id: boolean} — the component keeps computing those
 * booleans exactly as it did before (`availableScoreFlags.length > 0`,
 * `shownFramings.length > 0`, …), because those conditions are the existing,
 * already-correct answer to "has this pass produced anything". This function
 * only decides ORDER and FOLDING, so moving a facet into the rail cannot
 * change when it appears.
 *
 * A group whose data does not exist is omitted, never rendered empty: an empty
 * "🎨 Medium" heading reads as a broken pass rather than as one never run.
 */
export function railGroups(availability = {}) {
  const shown = (id) => availability[id] !== false;
  return {
    primary: RAIL_PRIMARY_GROUPS.filter(shown),
    folded: RAIL_FOLDED_GROUPS.filter(shown),
  };
}

/** How many measured axes are folded away right now — the number the "More
 *  filters" disclosure prints. A disclosure that does not say how much it
 *  hides is a disclosure nobody opens. */
export function foldedCount(availability = {}) {
  return railGroups(availability).folded.length;
}

/* ── The passes panel ───────────────────────────────────────────────────────
   ① Analyze became a panel opened on demand. All eight passes stay, with their
   existing dialogs untouched — the panel is a container, not a rewrite of what
   the buttons do. */

/** The passes panel starts CLOSED. It is the step you do once per bank, and it
 *  was occupying the top third of a screen whose subject is the grid. */
export const PASSES_PANEL_DEFAULT_OPEN = false;

/**
 * The label of the ⚙ Passes button. It carries the state of the work rather
 * than just naming the panel: while a pass is running the button is the only
 * thing still visible once the panel is closed, and "⚙ Passes" alone would hide
 * the fact that the bank is busy behind a click.
 */
export function passesButtonLabel(live) {
  return live ? '⚙ Passes (running…)' : '⚙ Passes';
}

/**
 * Should the passes panel be forced open? Yes exactly when a bank has nothing
 * measured yet: on a freshly imported dump the grid is a wall of unscanned
 * images and every useful action is inside that panel, so starting it closed
 * would hide the only thing there is to do.
 *
 * `counts` is the payload's own counter block. An absent payload answers
 * "don't force" — while it is still loading we know nothing, and popping a
 * panel open under the user a second after the page settles is worse than
 * leaving it shut.
 */
/** Below this width the panel does NOT open itself. Measured by the responsive
 *  probe on a fresh bank: at 360 px the panel is ~1 550 px tall — the eight pass
 *  buttons, the semantic-engine card, the watermark and edit panels and the
 *  overview, all stacked — so "opens itself once" put the first image two
 *  screens down. The ⚙ Passes button is in the header on every width; on a
 *  phone it stays a button. Same breakpoint as the rail (`lg`). */
export const PASSES_AUTO_OPEN_MIN_PX = 1024;

export function passesPanelStartsOpen(counts, viewportWidth = Infinity) {
  if (typeof viewportWidth === 'number' && viewportWidth < PASSES_AUTO_OPEN_MIN_PX) return false;
  if (!counts || typeof counts.total !== 'number') return PASSES_PANEL_DEFAULT_OPEN;
  if (counts.total <= 0) return PASSES_PANEL_DEFAULT_OPEN;
  return (counts.scanned || 0) <= 0;
}
