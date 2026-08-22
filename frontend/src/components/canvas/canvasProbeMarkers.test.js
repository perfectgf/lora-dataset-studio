/* 📐 The markers the responsive probe measures by.
 *
 * `scripts/responsiveProbe.mjs` renders the board at real viewport widths and
 * measures what came out — the one check in this repo that can see a panel full
 * of air, a box wider than the screen or a bar eating the fold. It finds the
 * surfaces to measure by attribute, not by class name, precisely so that
 * restyling a pill cannot quietly take it out of scope.
 *
 * Which leaves one hole, and it is the dangerous kind: delete the attributes and
 * the probe measures NOTHING and says everything is fine. The probe itself
 * refuses to pass with zero chrome markers, but that only fires when somebody
 * runs it. These assertions fire in `node --test`, which runs on every commit.
 *
 * ⚠️ This file cannot tell you the layout is good — nothing that reads source
 * text can, which is the whole reason the probe exists (see the header of
 * canvasResponsive.test.js). All it guarantees is that the thing which CAN tell
 * you is still pointed at the right elements. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const canvas = fs.readFileSync(new URL('./LineageCanvas.jsx', import.meta.url), 'utf8');
const probe = fs.readFileSync(
  new URL('../../../scripts/responsiveProbe.mjs', import.meta.url), 'utf8');

/* Every painted surface that sits permanently in front of the board. The list
   is explicit rather than a count: a NEW overlay that forgets its marker is
   invisible to the probe, and "we have five of something" would still pass. */
const CHROME = ['filter', 'tracker', 'bubble', 'shelf', 'toolbar'];

test('every fixed surface on the board is marked for the responsive probe', () => {
  for (const surface of CHROME) {
    assert.ok(canvas.includes(`data-probe-chrome="${surface}"`),
      `the ${surface} lost its data-probe-chrome marker — the probe stops measuring it`);
  }
});

test('the ⋯ shelf is marked as a PANEL, which is what the fill check reads', () => {
  // Chrome and panel are two different questions: chrome asks "how much of the
  // fold does this cost", panel asks "do its rows earn their lines". The shelf
  // is the surface that failed the second one while passing the first.
  assert.match(canvas, /data-testid="canvas-more-sheet" data-probe-chrome="shelf" data-probe-panel="shelf"/);
});

test('the board declares itself a pannable world, so its contents are not overflow', () => {
  // Without this the probe reports every pinned picture parked off-screen as a
  // layout defect — the board holds a world larger than its frame BY DESIGN,
  // and 20 false positives per width is how a check gets switched off.
  assert.match(canvas, /data-probe-world="board"/);
  assert.match(probe, /p\.hasAttribute\('data-probe-world'\)/);
});

test('the probe refuses to pass when it has measured nothing', () => {
  // The failure mode this whole file exists for: markers gone, zero elements
  // found, a clean report. Distinct exit codes back it up — 0 clean, 1 found
  // violations, 2 could not run.
  assert.match(probe, /kind: 'unmarked'/);
  assert.match(probe, /which is NOT a pass/);
  assert.match(probe, /process\.exit\(2\)/);
});

/* The thresholds are the contract, and each was a symptom before it was a
   number. Pinned so that a red probe cannot be silenced by editing the limit
   instead of the layout — moving one of these has to be a deliberate diff that
   shows up in review, next to the comment saying what was measured. */
test('the probe keeps the budgets it was given, and says what they mean', () => {
  assert.match(probe, /const MAX_CHROME_SHARE = \{ resting: 0\.28, open: 0\.50 \}/);
  assert.match(probe, /const MIN_ROW_FILL = 0\.35/);
  assert.match(probe, /const MIN_TOUCH_PX = 40/);
  // …and the note that says which way to fix a breach, since the tempting
  // direction is the wrong one.
  assert.match(probe, /take something OUT of[\s\S]*?the panel, not to move the line/);
});

/* 📏 The fold has TWO dimensions, and for a while this script only asked about
   one. A phone held sideways put the ⋯ shelf at 67 % while the very same shelf
   measured 27 % on the same width at 800 tall and passed — a viewport list of
   bare widths cannot see a landscape phone. */
test('the probe measures real devices, heights included', () => {
  assert.match(probe, /const DEFAULT_VIEWPORTS = \[/);
  assert.match(probe, /\[844, 390\]/, 'a phone held sideways is in the list');
  assert.match(probe, /--viewports/);
});

/* A surface at rest is not the surface people complain about: the reported
   layouts were all inside something that had been OPENED. */
test('the probe opens the board’s surfaces instead of only measuring it at rest', () => {
  assert.match(probe, /const PAGES = \{/);
  // includes(), not a RegExp: 'shelf+help' carries a + and building a pattern
  // from it turns the name into a quantifier — the assertion then passes or
  // fails for a reason that has nothing to do with the state list.
  for (const state of ['resting', 'shelf', 'shelf+help', 'layouts', 'datasets-menu', 'search']) {
    assert.ok(probe.includes(`name: '${state}'`), `the ${state} state is measured`);
  }
  // ⚠️ A hash-router goto to the SAME route does not reload — measured: the
  // shelf left open by one state was toggled SHUT by the next state's first
  // click, and every other state silently found nothing to open.
  assert.match(probe, /page\.goto\('about:blank'\)/);
});

/* The budget asks "how much of the fold is IN YOUR WAY". A dialog you opened in
   order to read is not in your way — charging it to the chrome budget forces a
   paragraph to be cut for a reason that was never true. */
test('a surface you opened to READ is not charged to the chrome budget', () => {
  assert.match(probe, /data-probe-reading/);
  assert.match(canvas, /data-probe-reading/);
  // …but it is still measured for everything else, which is what caught the
  // bubble growing up onto the filter bar on a landscape phone.
  assert.match(canvas, /data-probe-chrome="bubble"/);
});

/* "No violations" over four surfaces on an empty board is not the same answer
   as "no violations" over eleven, and only one of them is worth anything. This
   is the same failure the exit codes guard at the other end — and it is what
   caught the hash-router reload bug above, on the probe's own first run. */
test('every run says what it COVERED, clean or not', () => {
  assert.match(probe, /covered: \${measured} measurement\(s\)/);
  assert.match(probe, /NOT covered/);
  assert.match(probe, /skipped\.push/);
});
