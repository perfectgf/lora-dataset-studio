/* 📱 The ◉ LoRA Canvas on a small screen.

   The board is used from a phone daily — over Tailscale, at 400 to 1060 px —
   and it had never had a responsive pass: the drawers switched to fixed-width
   side panels at 640 px, the toolbar's targets were 36 px, the gesture list was
   `hidden` below a laptop, and the page blurb pushed the board's bottom edge
   past the fold on every load.

   Pinned as TEXT because `node --test` cannot parse JSX and because none of it
   is behaviour: a class is not a function, nothing throws when it goes, and the
   symptom only appears on a device the test suite never renders on. Every
   assertion below is a measurement that was taken headless at 400/768/1060 px,
   not a preference. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const canvas = fs.readFileSync(new URL('./LineageCanvas.jsx', import.meta.url), 'utf8');
const page = fs.readFileSync(new URL('../../pages/CanvasPage.jsx', import.meta.url), 'utf8');
const filter = fs.readFileSync(new URL('./CanvasDatasetFilter.jsx', import.meta.url), 'utf8');

/* The zoom/Fit/Tidy/Generate row is the ONLY way to zoom without a wheel. At
   36 px its buttons sat under the ~40 px a finger lands on, and a miss lands on
   the board and pans it — which reads as "the zoom buttons are unreliable". */
test('the board toolbar carries 40-px targets on a phone and 36 on a desktop', () => {
  const bar = canvas.slice(canvas.indexOf('aria-label="Zoom out"') - 400,
    canvas.indexOf('data-testid="canvas-deploy-legend"'));
  // − and + are square; Fit, Tidy up and 🎨 Generate are tall only.
  assert.equal((bar.match(/h-10 w-10 [^"']*lg:h-9 lg:w-9/g) || []).length, 2);
  assert.ok((bar.match(/h-10 [^"'+]*lg:h-9/g) || []).length >= 3);
  // No 36-px target left in the row at phone width.
  assert.doesNotMatch(bar, /className="flex h-9 /);
  // …and it still WRAPS rather than overflowing: 400 px cannot hold this row.
  // The wrap moved UP one level when the row became a floating bar ON the board
  // instead of chrome stacked above it: the pill is the flex container now, and
  // the old inner div is `contents` so its buttons stay direct flex items. What
  // is pinned is the PROPERTY (it wraps, it does not overflow), not the element
  // that happens to own it — pinning the old class string would have made the
  // move look like a regression while 400 px still worked perfectly.
  assert.match(canvas, /pointer-events-auto inline-flex max-w-full flex-wrap items-center gap-1\.5/);
  assert.match(canvas, /className="contents"/);
});

/* The gesture list is the board's entire documentation. It was `lg:inline` with
   no small-screen counterpart, so on the one device with no wheel, no hover
   title and no shift key it did not exist at all. */
test('the board gestures are reachable below lg, from a single source', () => {
  assert.match(canvas, /const BOARD_GESTURES = \(/);
  // Inline from lg up…
  assert.match(canvas, /ml-auto hidden [^"]*lg:inline[^]{0,80}\{BOARD_GESTURES\}/);
  // …and behind a one-tap chip below it.
  // …one more chip in a row that already wraps, never a row of its own: every
  // pixel above the frame is a pixel of board pushed under the fold.
  assert.match(canvas, /data-testid="canvas-gestures-toggle"/);
  assert.match(canvas, /aria-expanded=\{gesturesOpen\}/);
  // Written ONCE: two copies would drift the first time a gesture is added.
  assert.equal((canvas.match(/\{BOARD_GESTURES\}/g) || []).length, 2);
  // Touch is named: a phone has no wheel and cannot shift-click.
  assert.match(canvas, /wheel or pinch to zoom/);
  assert.match(canvas, /on touch, hold it first/);
});

/* 📏 Measured at 400×800 on a headless Chrome, before and after: the bottom bar
   was 213 px of the 800 this screen has, and tapping ☝ Gestures took it to 380
   — the board vanished behind its own manual, with no way back but finding the
   same chip again in a row that had moved. It was a `<details>`, and an open
   `<details>` GROWS the box it is in; the box here is the floating toolbar. The
   text is a sheet beside the pill now: 100 px closed, 100 px open. */
test('asking for the gesture help never grows the board’s toolbar', () => {
  // The sheet is a SIBLING of the pill — inside the bottom overlay, before it —
  // not a child of the row. That is the whole fix: nothing it contains can add
  // a row to the toolbar, however long the sentence gets.
  const overlay = canvas.slice(canvas.indexOf('pointer-events-none absolute inset-x-0 bottom-0'));
  const sheetAt = overlay.indexOf('data-testid="canvas-gestures-sheet"');
  const pillAt = overlay.indexOf('pointer-events-auto inline-flex max-w-full flex-wrap');
  assert.ok(sheetAt > 0 && sheetAt < pillAt, 'the sheet renders before the toolbar pill');
  // Conditional, so it costs nothing at all while it is not asked for.
  assert.match(canvas, /\{gesturesOpen && \(/);
  // …and it can be PUT AWAY: a phone has no Escape key within reach, so the ×
  // is the one that matters, but both are wired.
  assert.match(canvas, /aria-label="Close the gesture help"/);
  assert.match(canvas, /e\.key === 'Escape'\) setGesturesOpen\(false\)/);
});

/* The same 400 px that could not hold the row still cannot: the labels go, the
   icons and the 40-px targets stay, and the row falls from five wraps to two.
   Everything hidden keeps a `title`, so the word is one hover/long-press away
   and the accessible name never becomes an emoji. */
test('the toolbar drops its words, never its targets, on a phone', () => {
  const presets = fs.readFileSync(new URL('./CanvasLayoutPresets.jsx', import.meta.url), 'utf8');
  assert.ok(canvas.includes('<span className="hidden sm:inline">Tidy up</span>'));
  assert.ok(canvas.includes('<span className="hidden sm:inline"> LoRA</span>'));
  assert.ok(presets.includes('<span className="hidden sm:inline">Layouts</span>'));
  // 📷 keeps its word only while it is SAYING something ("Exporting…"): a chip
  // that goes blank mid-export would read as a chip that did nothing.
  assert.match(canvas, /exporting \? '' : 'hidden sm:inline'/);
  // …and a hidden word is never a lost one: the button that loses its label
  // gains/keeps a title, so the accessible name is a sentence, not an emoji.
  assert.match(presets, /<summary title="Layouts — /);
  // The deploy key shortens instead of wrapping a whole row of its own.
  assert.match(canvas, /className="sm:hidden">\{l\.short\}</);
  assert.match(canvas, /className="hidden sm:inline">\{l\.label\}</);
  // Padding shrinks, height does NOT: 40 px is what the finger needs, 12 px of
  // side padding is not.
  assert.ok((canvas.match(/px-2 sm:px-3/g) || []).length >= 5);
  const bar = canvas.slice(canvas.indexOf('aria-label="Zoom out"'),
    canvas.indexOf('data-testid="canvas-gestures-toggle"'));
  assert.doesNotMatch(bar, /\sh-9\s/, 'no 36-px target left at phone width');
});

/* 💾 is one chip in a row that WRAPS, so its position is whatever the wrap left
   it. Measured at 400 px: the chip landed at x=243 and its 18-rem menu opened
   to x=531 — the Save button was off the right of the screen. A menu anchored
   to a chip that moves is a menu that can open anywhere; below `sm` it stops
   being anchored at all. */
test('the Layouts menu opens on the screen, not off the side of it', () => {
  const presets = fs.readFileSync(new URL('./CanvasLayoutPresets.jsx', import.meta.url), 'utf8');
  assert.match(presets, /fixed inset-x-2 bottom-28/);
  // …and from sm up it is the SAME anchored menu it has always been: under its
  // own button, 18 rem wide. Desktop must not notice this pass at all.
  assert.match(presets, /sm:absolute[^"]*sm:left-0 sm:top-full sm:mt-1 sm:w-\[min\(18rem,calc\(100vw-2rem\)\)\]/);
});

/* 400 px × 800: the page chrome above the board measured 304 px, the frame is
   65vh = 520, and 304 + 520 > 800 — the board's bottom edge never fit. The
   blurb is 72 of those pixels and it explains the page exactly once. */
test('the canvas page drops its blurb on a phone, never its help', () => {
  assert.match(page, /className="mt-1 hidden text-content-muted text-\[0\.75rem\] sm:block"/);
  // The ? badge stays at every width, so the explanation is still one tap away.
  assert.match(page, /<HelpBadge topic="page-canvas" \/>/);
});

/* …and the frame itself gives back the last 5vh, so the WHOLE board — bottom
   edge included — is on screen at 400×800 rather than hanging under the fold.
   A board whose bottom you have to scroll the page to reach is a board whose
   pan gesture competes with the page's scroll. */
test('the board frame fits the fold on a phone and is unchanged from sm up', () => {
  // 60vh was the height left AFTER ~290 px of chrome — the zoom row, the colour
  // key, the gestures sheet, the run tracker and the dataset filter, all stacked
  // above the frame at 400 px. Every one of them now floats ON the board, so the
  // frame takes that space back: the number went up because the reason it was
  // small went away. Still short of the fold, which is the actual contract —
  // a board whose bottom edge you have to scroll to is a board whose pan gesture
  // fights the page's, and it is what makes ✦ Fit mean anything.
  assert.match(canvas, /h-\[72vh\] min-h-\[380px\][^"]*sm:h-\[76vh\]/);
  // The overlays must stay SIBLINGS of the frame, never children: the frame owns
  // the pointer handlers and `touch-none`, so a control nested inside it would
  // hand every tap to the board underneath.
  assert.match(canvas, /pointer-events-none absolute inset-x-0 top-0 z-20/);
  assert.match(canvas, /pointer-events-none absolute inset-x-0 bottom-0 z-20/);
});

/* The 🎨 Generate chip carries the pick count, which is what makes closing the
   sheet a cheap gesture: the picks are still visible from the board. */
test('the Generate chip keeps showing the pick count with the panel closed', () => {
  assert.match(canvas, /aria-pressed=\{panelOpen\}/);
  assert.match(canvas, /\{picks\.length > 0 && \([^]{0,200}\{picks\.length\}/);
});

/* The fix that was won once by folding the panel is now won by DELETING it:
   the filter is a wrapping row of chips, ~40 px tall, and the controls live in
   popovers. Nothing about it can cost the board height any more, folded or not
   — which is why the old "opens folded" guard has no subject left.

   What is pinned instead is the property that replaced it: the bar wraps (a
   400-px screen cannot hold this row on one line), every target is 40 px on a
   phone and 36 from `lg`, and a popover is never wider than the viewport. */
test('the filter bar wraps instead of occupying the board’s height', () => {
  assert.match(filter, /className="lds-canvas-filter [^"]*flex flex-wrap items-center gap-1\.5"/);
  // No fold-out body left to grow: no max-height panel, no unfold state.
  assert.doesNotMatch(filter, /max-h-\[\d+vh\]/);
  assert.doesNotMatch(filter, /readCanvasFilterOpen/);
  assert.doesNotMatch(filter, /innerWidth/);
});

test('every filter target is finger-sized on a phone and 36 px from lg', () => {
  const menu = fs.readFileSync(new URL('./CanvasFilterMenu.jsx', import.meta.url), 'utf8');
  // Both tokens, asserted SEPARATELY: these class strings are concatenated
  // across source lines, and a regex spanning them would be testing where the
  // author happened to wrap rather than what the browser receives.
  assert.match(menu, /\bh-10\b/);
  assert.match(menu, /\blg:h-9\b/);
  // The controls the bar draws itself (Pinned, the search box, Reset).
  assert.ok((filter.match(/\bh-10\b/g) || []).length >= 3, 'three 40-px targets in the bar');
  assert.ok((filter.match(/\blg:h-9\b/g) || []).length >= 3, '…each falling back to 36 px');
});

/* 400 px: a fixed-width popover hangs off the screen, and a filter half off the
   screen is a filter with no Clear button in reach. */
test('a filter popover never grows past the viewport', () => {
  const menu = fs.readFileSync(new URL('./CanvasFilterMenu.jsx', import.meta.url), 'utf8');
  assert.match(menu, /w-\[min\(20rem,calc\(100vw-2rem\)\)\]/);
  // …and its list scrolls rather than making the menu taller than the window.
  assert.match(filter, /max-h-64[^"]*overflow-y-auto/);
});
