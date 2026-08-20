/* 📱 The ◉ LoRA Canvas on a small screen.

   The board is used from a phone daily — over Tailscale, at 360 to 1060 px —
   and it had never had a responsive pass: the drawers switched to fixed-width
   side panels at 640 px, the toolbar's targets were 36 px, the gesture list was
   `hidden` below a laptop, and the page blurb pushed the board's bottom edge
   past the fold on every load. That first pass shrank everything; this second
   one RANKS it, because shrinking had run out (see the ⋯ block below).

   Pinned as TEXT because `node --test` cannot parse JSX and because none of it
   is behaviour: a class is not a function, nothing throws when it goes, and the
   symptom only appears on a device the test suite never renders on. Every
   assertion below is a measurement that was taken headless at 360/412/904/1024/
   1920 px on a real board, not a preference. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const canvas = fs.readFileSync(new URL('./LineageCanvas.jsx', import.meta.url), 'utf8');
const page = fs.readFileSync(new URL('../../pages/CanvasPage.jsx', import.meta.url), 'utf8');
const filter = fs.readFileSync(new URL('./CanvasDatasetFilter.jsx', import.meta.url), 'utf8');
const app = fs.readFileSync(new URL('../../App.jsx', import.meta.url), 'utf8');

/* ── THE MEASUREMENT THIS WHOLE FILE EXISTS FOR ──────────────────────────────
   412×780, a real board, before this pass:

     app bar                      58 px
     page header (◉ LoRA Canvas)  67 px
     filter bar, floating         116 px   (two rows)
     toolbar, floating            116 px   (two rows)
     ---------------------------------------------
     board actually free          427 px   — 55 % of the screen

   After: the two bars are one row each, the page header is not drawn, and the
   board is free over 497 px — 64 %. The bars did not get smaller; the things on
   them got RANKED, and everything that is not a control moved behind ⋯. */

/* The zoom/Fit/Generate row is the ONLY way to zoom without a wheel. At 36 px
   its buttons sat under the ~40 px a finger lands on, and a miss lands on the
   board and pans it — which reads as "the zoom buttons are unreliable". */
test('the board toolbar carries 40-px targets on a phone and 36 on a desktop', () => {
  const bar = canvas.slice(canvas.indexOf('aria-label="Zoom out"') - 400,
    canvas.indexOf('data-testid="canvas-more-toggle"'));
  // − and + are square; Fit and 🎨 Generate are tall only.
  assert.equal((bar.match(/h-10 w-10 [^"']*lg:h-9 lg:w-9/g) || []).length, 2);
  assert.ok((bar.match(/h-10 [^"'+]*lg:h-9/g) || []).length >= 2);
  // No 36-px target left in the row at phone width.
  assert.doesNotMatch(bar, /className="flex h-9 /);
  // …and it still WRAPS rather than overflowing: 360 px cannot be trusted to
  // hold any row, whatever the ranking says. The pill is the flex container and
  // the old inner div is `contents`, so its buttons stay direct flex items.
  assert.match(canvas, /pointer-events-auto inline-flex max-w-full flex-wrap items-center gap-1\.5/);
  assert.match(canvas, /className="contents"/);
});

/* ── ⋯ — the ranking, and the three tiers it produced ───────────────────────
   Shrinking had run out. Measured with every control inline, the bar is two
   rows at 412, two at 904 (a Fold opened), two at 1440 and STILL two at 1920 —
   because the gesture line alone is ~500 characters. A bar that wraps at every
   width on earth is not a bar that needs smaller buttons.

   So each thing on it was asked what it IS, and the answer decided where it
   lives. The thresholds below are measurements: 768 was tried for the actions
   and gives two rows, 1024 gives one; 1536 was tried for the readouts with the
   gesture line still inline and gives two rows at 1920. */
test('the board toolbar is ranked into three tiers, not one row of equals', () => {
  // Tier 1 — inline at every width: zoom, Fit, Generate, and ⋯ itself.
  assert.match(canvas, /data-testid="canvas-more-toggle"/);
  // Tier 2 — actions: inline from `lg`, in ⋯ below it.
  assert.match(canvas, /const inlineActions = useMediaQuery\('\(min-width: 1024px\)'\)/);
  // Tier 3 — readouts: inline from `2xl`, in ⋯ below it.
  assert.match(canvas, /const inlineReadouts = useMediaQuery\('\(min-width: 1536px\)'\)/);
  // The two shelves are placed by those flags, never duplicated by CSS.
  assert.match(canvas, /\{inlineActions && boardActions\}/);
  assert.match(canvas, /\{inlineReadouts && boardReadouts\}/);
  // Declared once, used twice — the inline slot and the sheet slot, nowhere else.
  assert.equal((canvas.match(/\bboardActions\b/g) || []).length, 3, 'declared once, placed twice');
  assert.equal((canvas.match(/\bboardReadouts\b/g) || []).length, 3);
  // ⋯ exists at EVERY width, because the gesture line never comes back inline.
  assert.doesNotMatch(canvas, /hasOverflow/);
});

/* Tailwind can hide a chip at a width; it cannot MOVE one. The alternative was
   writing every secondary control twice — once for the toolbar, once for the
   sheet — and two copies of a control drift the first time one gains a prop.
   `matchMedia` and not a resize listener: a resize fires on every pixel of a
   drag and on every scroll that moves a mobile URL bar, and would re-render a
   board of hundreds of cards each time. */
test('a control is written once and PLACED, never rendered twice', () => {
  const hook = fs.readFileSync(new URL('../../hooks/useMediaQuery.js', import.meta.url), 'utf8');
  assert.match(hook, /window\.matchMedia\(query\)/);
  assert.doesNotMatch(hook, /window\.innerWidth/);
  // Safari below 14 has no addEventListener on MediaQueryList, and this app is
  // opened from phones.
  assert.match(hook, /mq\.addListener\(onChange\)/);
  // One ✦ Tidy up, one 💾 Layouts, one 📷 PNG, one 🔌 +LoRA in the whole file.
  assert.equal((canvas.match(/onClick=\{handleTidyUp\}/g) || []).length, 1);
  assert.equal((canvas.match(/<CanvasLayoutPresets/g) || []).length, 1);
  assert.equal((canvas.match(/data-testid="canvas-export-png"/g) || []).length, 1);
  assert.equal((canvas.match(/data-canvas-ext-lora-toggle/g) || []).length, 1);
});

/* 📏 Measured at 400×800 on a headless Chrome, before and after: the bottom bar
   was 213 px of the 800 this screen has, and tapping ☝ Gestures took it to 380
   — the board vanished behind its own manual, with no way back but finding the
   same chip again in a row that had moved. It was a `<details>`, and an open
   `<details>` GROWS the box it is in; the box here is the floating toolbar.

   The ⋯ sheet inherits that lesson whole: it is a SIBLING of the pill, so
   nothing it contains can add a row to the toolbar, however much it holds. */
test('opening ⋯ never grows the board’s toolbar', () => {
  const overlay = canvas.slice(canvas.indexOf('pointer-events-none absolute inset-x-0 bottom-0'));
  const sheetAt = overlay.indexOf('data-testid="canvas-more-sheet"');
  const pillAt = overlay.indexOf('pointer-events-auto inline-flex max-w-full flex-wrap');
  assert.ok(sheetAt > 0 && sheetAt < pillAt, 'the sheet renders before the toolbar pill');
  // Conditional, so it costs nothing at all while it is not asked for.
  assert.match(canvas, /\{moreOpen && \(/);
  // …and it can be PUT AWAY: a phone has no Escape key within reach, so the
  // Close button is the one that matters, but both are wired.
  assert.match(canvas, /aria-label="Close the board tools"/);
  assert.match(canvas, /e\.key === 'Escape'\) setMoreOpen\(false\)/);
});

/* The gesture list is the board's entire documentation — and it is ~500
   characters, which is why it had never fitted anywhere. It used to be
   `lg:inline` with no small-screen counterpart, then a chip of its own with its
   own sheet. It is now one paragraph in the ⋯ sheet, at every width: the same
   words, one door instead of two, and a toolbar row it no longer costs. */
test('the board gestures are reachable at every width, from a single source', () => {
  assert.match(canvas, /const BOARD_GESTURES = \(/);
  // In the sheet, unconditionally — not behind `!inlineReadouts`.
  const sheet = canvas.slice(canvas.indexOf('data-testid="canvas-more-sheet"'),
    canvas.indexOf('aria-label="Close the board tools"'));
  assert.match(sheet, /\{BOARD_GESTURES\}/);
  // Written ONCE: two copies would drift the first time a gesture is added, and
  // the chip that used to open its own sheet is gone with it.
  assert.equal((canvas.match(/\{BOARD_GESTURES\}/g) || []).length, 1);
  assert.doesNotMatch(canvas, /canvas-gestures-toggle/);
  assert.doesNotMatch(canvas, /canvas-gestures-sheet/);
  // Touch is named: a phone has no wheel and cannot shift-click.
  assert.match(canvas, /wheel or pinch to zoom/);
  assert.match(canvas, /on touch, hold it first/);
});

/* A shelf that hides state without saying so is a shelf that makes the board
   look broken: 🔌 external LoRAs are ON the board and stack onto the next run,
   and with the chip folded away there was nothing left saying so. */
test('⋯ says what is folded behind it', () => {
  assert.match(canvas, /\{!inlineActions && extNodes\.length > 0 && \(/);
  assert.match(canvas, /aria-expanded=\{moreOpen\}/);
  // An emoji is not an accessible name.
  assert.match(canvas, /aria-label="More board tools"/);
});

/* ✦ 💾 📷 🔌 ⏏ lost their WORDS in the first pass, and it was the right call at
   the time: they were in a toolbar that could not hold them. They are not any
   more — below `lg` they are in a sheet as wide as the screen, above `lg` in a
   toolbar measured to hold them — so the words come back. A shelf of five
   unlabelled glyphs is a shelf nobody opens twice.

   What did NOT come back is the height: the targets are still 40 px up to `lg`
   and 36 above, and the padding still shrinks below `sm`. */
test('the shelf’s chips carry their words, and the toolbar keeps its targets', () => {
  const presets = fs.readFileSync(new URL('./CanvasLayoutPresets.jsx', import.meta.url), 'utf8');
  assert.ok(canvas.includes('<span aria-hidden>✦</span> Tidy up'));
  assert.ok(canvas.includes('<span aria-hidden>🔌</span> + LoRA'));
  assert.ok(canvas.includes('<span aria-hidden>⏏</span> Undeploy…'));
  assert.doesNotMatch(canvas, /hidden sm:inline">Tidy up/);
  assert.doesNotMatch(presets, /hidden sm:inline">Layouts/);
  // 📷 says what it is doing while it does it, at every width now.
  assert.match(canvas, /\{exporting \? 'Exporting…' : 'PNG'\}/);
  // …and a title is still a sentence, not a repeat of the label.
  assert.match(presets, /<summary title="Layouts — /);
  // The deploy key still shortens below `sm` — it is a KEY, not a control, and
  // its long form is two full sentences.
  assert.match(canvas, /className="sm:hidden">\{l\.short\}</);
  assert.match(canvas, /className="hidden sm:inline">\{l\.label\}</);
  // Padding shrinks, height does NOT: 40 px is what the finger needs, 12 px of
  // side padding is not.
  assert.ok((canvas.match(/px-2 sm:px-3/g) || []).length >= 5);
  const bar = canvas.slice(canvas.indexOf('aria-label="Zoom out"'),
    canvas.indexOf('data-testid="canvas-more-toggle"'));
  assert.doesNotMatch(bar, /\sh-9\s/, 'no 36-px target left at phone width');
});

/* 📊 The machine-load readout was `hidden` below `sm`, and the reason was the
   toolbar: on a 400-px screen that row already wrapped twice. It is in the ⋯
   shelf now, where it costs the board nothing until opened — and the phone is
   the device that wants it most, being the screen you check the machine from
   when you are not sitting at it. */
test('the load readout is reachable from a phone', () => {
  const stats = fs.readFileSync(new URL('./CanvasSystemStats.jsx', import.meta.url), 'utf8');
  assert.doesNotMatch(stats, /className="hidden items-center gap-1\.5 sm:flex"/);
  assert.match(stats, /data-testid="canvas-system-stats"[^]{0,120}className="flex flex-wrap items-center/);
  // The ▾ toggle still STOPS THE POLL rather than just hiding the line — this
  // is the only thing on the page that polls forever.
  assert.match(stats, /data-testid="canvas-system-stats-toggle"/);
  assert.match(stats, /shouldPoll\(\{ enabled: enabledRef\.current, visibility \}\)/);
});

/* 📏 360 px: the toolbar needed 326 of the 316 it had, and 12 of the missing 10
   were a width reserve between − and + for a zoom percentage the board cannot
   reach. MAX_SCALE is 4, so "400%" is the widest string this ever shows. */
test('the zoom readout reserves what the board can show, not what a number could be', () => {
  assert.match(canvas, /min-w-\[2\.5rem\] sm:min-w-\[3\.25rem\][^"]*tabular-nums">\{pct\}%/);
  // tabular-nums stays: without it 100% and 111% are different widths and the
  // − and + shift under the thumb as you zoom.
  assert.match(canvas, /tabular-nums">\{pct\}%/);
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

/* 400 px × 800: the page chrome above the board measured 304 px and the board's
   bottom edge never fit. The blurb was 72 of those pixels; the TITLE was 67
   more, and it repeats a word the nav bar is already highlighting. Below `lg`
   neither is drawn — but the page still HAS an <h1>, because a screen with no
   heading is a screen a reader lands in the middle of. */
test('the canvas page folds its header on a phone, never its help', () => {
  // The visible header stops at `lg`…
  assert.match(page, /<header className="mb-2 hidden sm:mb-3 lg:block">/);
  // …and an sr-only title takes its place exactly where it was dropped.
  assert.match(page, /<h1 className="sr-only lg:hidden">LoRA Canvas<\/h1>/);
  // The blurb stays hidden right up to `lg`, as it already was.
  assert.match(page, /className="mt-1 hidden text-content-muted text-\[0\.75rem\] lg:block"/);
  assert.doesNotMatch(page, /text-\[0\.75rem\] sm:block/);
  // The ? badge is not lost with the header it sat in: it moves onto the ⋯
  // shelf, exactly like ⏏ Undeploy… did. "The ? next to the title explains this
  // page at every width" is a promise this page makes in its own comments.
  assert.match(page, /<HelpBadge topic="page-canvas" \/>/);
  assert.match(canvas, /\{onOpenUndeploy && <HelpBadge topic="page-canvas" \/>\}/);
});

/* ⏏ Undeploy… is an INSTALL-wide action and lives on the page header. That
   header is not drawn below `lg` — so the button it carried has to land
   somewhere, or folding the header would have deleted a feature. */
test('folding the page header moves ⏏ Undeploy, it never drops it', () => {
  assert.match(page, /onOpenUndeploy=\{\(\) => setUndeployOpen\(true\)\}/);
  assert.match(canvas, /onOpenUndeploy = null \}\) \{/);
  // In the ⋯ shelf, and only below `lg` — above it the page header has it and
  // two of the same button on one screen is one too many.
  assert.match(canvas, /data-testid="canvas-undeploy-more"/);
  const btn = canvas.slice(canvas.indexOf('data-testid="canvas-undeploy-more"'));
  assert.match(btn.slice(0, 700), /\blg:hidden\b/);
});

/* 📏 The board is the whole screen, and 8 px a side is a considered margin on a
   desktop and 16 px of a 360-px phone's toolbar row. Measured: with the gutter,
   the toolbar is two rows at 360; without it, one. */
test('the board goes edge to edge on a phone', () => {
  assert.match(app, /\? 'flex min-h-0 w-full flex-1 flex-col p-0 sm:px-3 sm:py-3'/);
  // `/canvas` only — the Bank is a scrolling GRID and keeps its reading measure.
  assert.match(app, /const boardRoute = pathname === '\/canvas';/);
});

/* 📏 Measured at 400×800 with a real board on screen: the filter bar wrapped to
   THREE rows (132 px) of the frame it floats on, and 224 of the 346 available
   px belonged to a search field that is empty on all but a handful of visits.
   Folded behind 🔍 and with the chips down to glyph + count, the same bar is two
   rows — 86 px. It is ONE row now, 40 px, at 412 and up. */
test('the board search folds behind 🔍 on a phone and is untouched from lg', () => {
  assert.match(filter, /data-testid="canvas-filter-search-toggle"/);
  // The toggle exists ONLY below lg — above it the field is in the bar, so a
  // magnifier next to a visible search box would be a control with no job.
  const toggle = filter.slice(filter.indexOf('data-testid="canvas-filter-search-toggle"'));
  assert.match(toggle.slice(0, 600), /\blg:hidden\b/);
  // The field itself: hidden while folded, its own full-width row while open,
  // and from `lg` the exact 12-rem flex item it has always been.
  assert.match(filter, /searchOpen \? 'basis-full ' : 'hidden '/);
  assert.match(filter, /lg:h-9 lg:block lg:basis-48/);
  // A filter you cannot see must still announce itself — the bar's own rule.
  assert.match(filter, /queryActive\n?\s*\? 'border-indigo-400\/60 bg-indigo-500\/15/);
  assert.match(filter, /max-w-\[6rem\] truncate font-normal">\{query\}</);
});

/* The chips carried "Datasets All 3 datasets" + "Models 1/1" + "Status 1/1" +
   "Pinned" — 400 px cannot hold that and a search box on two rows. The words go
   below `sm`, exactly like the board toolbar under them; the glyph, the count
   and the 40-px target never do. */
test('the filter chips drop their words on a phone, never their counts', () => {
  const menu = fs.readFileSync(new URL('./CanvasFilterMenu.jsx', import.meta.url), 'utf8');
  assert.match(menu, /className="hidden truncate md:inline">\{label\}</);
  // A hidden word is never a lost one: the accessible name stays a sentence.
  const named = (menu.match(/\{`\$\{label\}\$\{summary \? ` — \$\{summary\}` : ''\}`\}/g) || []);
  // Both the title AND the accessible name — an aria-label replaces the button's
  // contents, so labelling it "Datasets" alone would take the count away from
  // the one user who cannot see the chip light up.
  assert.equal(named.length, 2, 'title and aria-label both carry label + count');
  // The count survives at every width, in its short form where there is one —
  // that count is the whole reason a folded filter can be trusted.
  assert.match(menu, /\{short \?\? summary\}/);
  assert.match(filter, /short=\{`\$\{sel\.size\}\/\$\{total\}`\}/);
  // 🖼 Pinned loses its word too — but never its "off", which is the state that
  // explains an empty-looking board.
  assert.match(filter, /className="hidden md:inline">Pinned</);
  assert.match(filter, /\{!showPinned && <span className="font-normal">off<\/span>\}/);
  assert.match(filter, /aria-label="Pinned images on the board"/);
  // …and the readout keeps its number, dropping only the word "shown".
  assert.match(filter, /<span className="hidden lg:inline"> shown<\/span>/);
});

/* 📏 412 px, a real board with a filter active: the row came to 332 of the 366
   it had and ↺ Reset needed 37 more, so the bar took a second 46-px row off the
   board for want of five pixels. Four pixels of side padding and two of gap per
   chip is what paid for it. Height is untouched — 40 px is what a finger needs;
   10 px of side padding is not. */
test('the filter bar is one row on a phone, bought from padding not from height', () => {
  assert.match(filter, /className="lds-canvas-filter [^"]*flex flex-wrap items-center gap-1 md:gap-1\.5"/);
  const menu = fs.readFileSync(new URL('./CanvasFilterMenu.jsx', import.meta.url), 'utf8');
  assert.match(menu, /h-10 max-w-full items-center gap-1 md:gap-1\.5 rounded-md border px-2 md:px-2\.5/);
  assert.ok((filter.match(/px-2 md:px-2\.5/g) || []).length >= 3, 'the bar’s own chips too');
  // No fold-out body left to grow: no max-height panel, no unfold state.
  assert.doesNotMatch(filter, /max-h-\[\d+vh\]/);
  assert.doesNotMatch(filter, /readCanvasFilterOpen/);
  assert.doesNotMatch(filter, /innerWidth/);
});

/* ↺ Reset is disabled on most visits, and a disabled button costs exactly the
   width of an enabled one. Below `sm` it is not drawn until there is something
   to reset; from `sm` up it keeps the familiar always-there-but-greyed
   behaviour, because there the width is not the scarce thing. */
test('↺ Reset costs the phone nothing while there is nothing to reset', () => {
  assert.match(filter, /\+ \(anyNarrowing \? 'flex' : 'hidden'\)\}/);
  assert.match(filter, /lg:h-9 md:flex /);
  // The word comes back from `sm`; the glyph carries it below.
  assert.match(filter, /<span aria-hidden className="md:hidden">↺<\/span>/);
  assert.match(filter, /<span className="hidden md:inline">Reset<\/span>/);
  // …and a button that loses its word keeps its sentence.
  assert.match(filter, /aria-label="Reset the filters"/);
});

/* 🎨 The board's tracker and the settings panel's in-flight bar read the SAME
   numbers (useCanvasStudio hands one `run.data` to both), so an open panel on a
   phone said "N generating · M queued · Stop (resumable)" twice. */
test('a run in flight is announced once on a phone, and Stop stays reachable', () => {
  // Only while the panel is open AND the run is working — the state the panel
  // duplicates. It cannot be the other way round: RunSetupPanel hides its whole
  // form while `pending > 0`, so dropping the PANEL's bar would leave an open
  // sheet with nothing in it.
  assert.match(canvas, /panelOpen && runPhase === 'working' \? ' hidden lg:block' : ''/);
  // Stopped and finished runs keep the board's bar at every width: ▶ Resume,
  // 📌 Pin all and the result links exist nowhere else.
  assert.match(canvas, /runPhase !== 'idle' && \(/);
  // One reading of the phase, from the helper the bar itself renders from.
  assert.match(canvas, /const runPhase = describeCanvasRun\(tracker\.run\.data\)\.phase;/);
});

/* …and the frame takes the WHOLE fold, once, and then stops moving.

   Three heights have now been tried on this frame and only the third answers
   both halves of the complaint. `60vh`, then `72vh/76vh`, left dead page under
   the board on a desktop. `clamp(floor, --canvas-content-h, ceiling)` — the
   frame sized to its CONTENT, which shipped for a single day — fixed the dead
   strip and bought a worse problem: an elastic canvas. ✦ Tidy up compacted the
   board and the frame snapped shut around it, cutting cards off at the zoom the
   user was on; dragging a node downwards inflated the frame live under the
   hand.

   So: no `vh` and no content term. `flex-1 min-h-0` in a one-viewport-tall
   column (App.jsx pins the `/canvas` shell, CanvasPage the page root) — the
   fold contract holds by CONSTRUCTION rather than by a fraction that had to be
   re-measured every time the chrome above changed, and the frame is now the
   most room there is on the screen. */
test('the board frame is fixed and fills the fold, with no content term left', () => {
  // Fills what is left of the viewport, floors at 320px, and no longer carries
  // a height of its own at any breakpoint.
  assert.match(canvas, /className="lds-canvas-frame relative isolate z-0 min-h-\[320px\] w-full flex-1 /);
  assert.doesNotMatch(canvas, /--canvas-content-h/);
  assert.doesNotMatch(canvas, /previewFrameHeight/);
  const frameClass = canvas.slice(canvas.indexOf('className="lds-canvas-frame'));
  assert.doesNotMatch(frameClass.slice(0, frameClass.indexOf('"', 11)), /vh\]/);
  // The column the frame stretches inside: `min-h-0` is the load-bearing half —
  // without it a flex child refuses to shrink below its content and the PAGE
  // scrolls instead of the board.
  assert.match(canvas, /className="lds-canvas-stage relative flex min-h-0 flex-1 flex-col"/);
  // The overlays must stay SIBLINGS of the frame, never children: the frame owns
  // the pointer handlers and `touch-none`, so a control nested inside it would
  // hand every tap to the board underneath.
  assert.match(canvas, /pointer-events-none absolute inset-x-0 top-0 z-20/);
  assert.match(canvas, /pointer-events-none absolute inset-x-0 bottom-0 z-20/);
});

/* ✦ Tidy up compacts the board on purpose, and it is only reachable from a
   board the user has ARRANGED — so the auto-fit is switched off at exactly the
   moment the framing stops matching the content. It has to re-frame itself. */
test('✦ Tidy up re-fits the board it just compacted', () => {
  assert.match(canvas, /onClick=\{handleTidyUp\}/);
  assert.match(canvas, /refitAfterTidy\.current = true;\s*\n\s*onTidyUp\?\.\(\);/);
  // Deferred: the parent still holds the OLD positions at click time, so the
  // fit has to land on the next world, not this render's.
  assert.match(canvas, /if \(!refitAfterTidy\.current\) return;[^]{0,400}setView\(fitView\(world, viewport\)\);/);
});

/* The 🎨 Generate chip carries the pick count, which is what makes closing the
   sheet a cheap gesture: the picks are still visible from the board. */
test('the Generate chip keeps showing the pick count with the panel closed', () => {
  assert.match(canvas, /aria-pressed=\{panelOpen\}/);
  assert.match(canvas, /\{picks\.length > 0 && \([^]{0,200}\{picks\.length\}/);
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
