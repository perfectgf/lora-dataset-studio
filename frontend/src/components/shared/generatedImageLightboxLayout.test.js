import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import {
  FACTS_PANEL_CLASS, IMAGE_CLASS, IMAGE_CLASS_BARE, IMAGE_PANE_CLASS,
  IMAGE_PANE_CLASS_BARE, MAX_PANEL_PX, MIN_PANEL_PX,
  SHELL_CLASS, SPLIT_MIN_WIDTH_PX, SPLIT_VARIANT,
} from './generatedImageLightboxLayout.js';

const lightbox = fs.readFileSync(
  new URL('./GeneratedImageLightbox.jsx', import.meta.url), 'utf8');

/* THE REGRESSION THESE TESTS EXIST FOR, stated once so a future edit knows what
 * it is undoing: the split used to start at `lg` (1024 px). A tablet in
 * LANDSCAPE reports ~900-960 CSS px — under that floor — so the widest screen
 * this viewer is opened on got the phone layout, with the picture drawn at
 * roughly a tenth of the viewport and every action pushed below the fold.
 * Raising the floor back to `lg`, or dropping it to `sm`, both reintroduce a
 * measured failure; the tests below fail rather than let either happen quietly.
 */

/** Every `x:` variant chain in a Tailwind class string, deduplicated. */
function variantsOf(classString) {
  return new Set(classString.split(/\s+/).filter(Boolean)
    .map((c) => {
      const i = c.lastIndexOf(':');
      return i === -1 ? '' : `${c.slice(0, i + 1)}`;
    })
    .filter(Boolean));
}

/** The px width behind a `w-[NNrem]` utility, root font size 16. */
function remWidth(utility) {
  const m = /w-\[([\d.]+)rem\]$/.exec(utility);
  return m ? Number(m[1]) * 16 : null;
}

test('the split starts at md, not lg — a landscape tablet is ~900 CSS px', () => {
  assert.equal(SPLIT_MIN_WIDTH_PX, 768);
  assert.ok(SPLIT_VARIANT.startsWith('md:'),
    `split variant must be gated on md, got ${SPLIT_VARIANT}`);
  assert.ok(!SPLIT_VARIANT.startsWith('lg:'));
});

test('the split also requires landscape — a portrait tablet keeps the stack', () => {
  // Width alone said "split" for a 900x2000 tablet stood upright, where the
  // side column costs the one axis that screen is short of: measured, the
  // picture went from 832x1216 stacked to 556x813 split.
  assert.ok(SPLIT_VARIANT.includes('landscape:'),
    `split variant must include the landscape variant, got ${SPLIT_VARIANT}`);
});

test('shell and facts panel change shape on the SAME condition', () => {
  // A panel that moves to the side while the shell is still a column, or the
  // reverse, is not a degraded layout — it is a blank screen with a stripe.
  const shellSplit = [...variantsOf(SHELL_CLASS)];
  const panelSplit = [...variantsOf(FACTS_PANEL_CLASS)];
  assert.deepEqual(shellSplit, [SPLIT_VARIANT]);
  for (const variant of panelSplit) {
    assert.ok(variant.endsWith('landscape:') && /^(md|lg|xl):/.test(variant),
      `facts panel variant ${variant} does not follow the split condition`);
  }
});

test('the facts column stays a reading width at every breakpoint', () => {
  const widths = FACTS_PANEL_CLASS.split(/\s+/)
    .map(remWidth).filter((n) => n !== null);
  assert.ok(widths.length >= 2, 'expected several responsive panel widths');
  for (const px of widths) {
    assert.ok(px >= MIN_PANEL_PX,
      `${px}px is under the ${MIN_PANEL_PX}px floor — action labels would wrap to stubs`);
    assert.ok(px <= MAX_PANEL_PX,
      `${px}px is over the ${MAX_PANEL_PX}px cap — the prompt stops being a paragraph`);
  }
  // Widths only ever grow with the breakpoint: a panel that narrows on a bigger
  // screen is a bug that reads as a rendering glitch.
  assert.deepEqual(widths, [...widths].sort((a, b) => a - b));
});

test('stacked, the facts band is bounded so the image keeps the screen', () => {
  // Unprefixed = the narrow/portrait case. Without a cap the panel takes the
  // height it wants and the picture is what is left over.
  assert.ok(FACTS_PANEL_CLASS.split(/\s+/).includes('max-h-[45vh]'));
  assert.ok(FACTS_PANEL_CLASS.split(/\s+/).includes('overflow-y-auto'));
  assert.ok(FACTS_PANEL_CLASS.split(/\s+/).includes('w-full'));
});

test('the image pane can shrink on both axes', () => {
  // `min-h-0`/`min-w-0`: a flex child defaults to min-content, so without these
  // a tall or wide picture pushes the facts column off the viewport instead of
  // scaling itself down.
  assert.ok(IMAGE_PANE_CLASS.includes('min-h-0'));
  assert.ok(IMAGE_PANE_CLASS.includes('min-w-0'));
  assert.ok(IMAGE_PANE_CLASS.includes('flex-1'));
  assert.ok(IMAGE_CLASS.includes('object-contain'));
  assert.ok(IMAGE_CLASS.includes('max-h-full') && IMAGE_CLASS.includes('max-w-full'));
});

test('the lightbox uses these classes rather than restating them', () => {
  for (const name of ['SHELL_CLASS', 'FACTS_PANEL_CLASS']) {
    assert.ok(lightbox.includes(`{${name}}`),
      `GeneratedImageLightbox.jsx should render {${name}}`);
  }
  // The pane and the picture now depend on one state — are the facts on screen —
  // and they read it from ONE derived boolean. Not from helper functions: this
  // module must export nothing but constants or the opacity contract goes blind
  // to it (see the note in the module itself).
  assert.match(lightbox, /const showFacts = facts && factsOpen;/);
  assert.match(lightbox, /showFacts \? IMAGE_PANE_CLASS : IMAGE_PANE_CLASS_BARE/);
  assert.match(lightbox, /showFacts \? IMAGE_CLASS : IMAGE_CLASS_BARE/);
  // No second copy of the breakpoint anywhere in the component: that is how the
  // two halves drift apart.
  assert.ok(!/lg:flex-row|lg:w-\[\d/.test(lightbox),
    'a leftover lg: layout utility is still in the component');
});

/* 📏 Measured on a real render, the panel open: at 412x780 the picture is drawn
   388x290 — 35 % of the screen — and at 904x750 held sideways it is 560x418,
   the same 35 %, the rail spending 320 of 904 px instead of 45 vh. The facts
   are what this viewer is for and they are not what you want while you are
   looking at the render; both have to be reachable in one gesture. */
test('the facts fold away and the picture takes the screen', () => {
  // Both states exist, and the bare one drops the frame with the panel: 12 px a
  // side is a considered margin in a layout and 24 px of a 412-px screen when
  // the picture IS the layout.
  assert.ok(IMAGE_PANE_CLASS.includes('p-3'));
  assert.ok(IMAGE_PANE_CLASS_BARE.includes('p-0'));
  assert.ok(!IMAGE_CLASS_BARE.includes('rounded') && !IMAGE_CLASS_BARE.includes('shadow'));
  // …and everything load-bearing survives the fold.
  for (const c of ['min-h-0', 'min-w-0', 'flex-1']) assert.ok(IMAGE_PANE_CLASS_BARE.includes(c));
  for (const c of ['object-contain', 'max-h-full', 'max-w-full']) assert.ok(IMAGE_CLASS_BARE.includes(c));
  // Every export is a constant — the shape this module must keep, so the
  // opacity contract can still read the panel's `bg-app` through the import.
  const src = fs.readFileSync(new URL('./generatedImageLightboxLayout.js', import.meta.url), 'utf8');
  assert.doesNotMatch(src, /export function /);
});

test('the fold has a button AND the gesture, and never strands the reader', () => {
  // Discoverable: a tap-to-hide with no affordance is folklore, not a feature.
  assert.match(lightbox, /data-testid="lightbox-facts-toggle"/);
  assert.match(lightbox, /aria-expanded=\{factsOpen\}/);
  // An emoji is not an accessible name, and the name has to say which way it goes.
  assert.match(lightbox, /aria-label=\{factsOpen \? 'Hide the image details' : 'Show the image details'\}/);
  // Pinned to the OVERLAY beside ✕, never inside the column it folds.
  const toggleIdx = lightbox.indexOf('data-testid="lightbox-facts-toggle"');
  const asideIdx = lightbox.indexOf('<aside');
  assert.ok(toggleIdx > 0 && toggleIdx < asideIdx);
  assert.match(lightbox.slice(toggleIdx - 200, toggleIdx + 700), /absolute right-14 top-3/);
  // Tapping the picture does it too. The tap now arrives through the zoom hook
  // rather than from an onClick, because a single tap and the first half of a
  // double tap are the same event — see useImageZoomPan. It still cannot close
  // the viewer: the press never reaches the backdrop.
  assert.match(lightbox, /onTap: useCallback\(\(\) => \{ if \(facts\) setFactsOpen\(\(v\) => !v\); \}, \[facts\]\)/);
  assert.match(lightbox, /onClick=\{\(e\) => e\.stopPropagation\(\)\}/);
  // A picture with no facts (a 🪪 reference face passes facts={false}) gets no
  // toggle: a control that folds nothing is a control that reads as broken.
  const toggleBlock = lightbox.slice(lightbox.indexOf('{facts && ('), toggleIdx + 60);
  assert.match(toggleBlock, /data-testid="lightbox-facts-toggle"/);
  assert.match(lightbox, /\{showFacts && \(/);
});

test('the close button is pinned to the overlay, above both halves', () => {
  // It must not live inside the scrolling facts column — scrolling the prompt
  // would carry the only way out of the viewer off screen.
  const closeIdx = lightbox.indexOf('aria-label="Close image"');
  const asideIdx = lightbox.indexOf('<aside');
  assert.ok(closeIdx > 0 && asideIdx > 0 && closeIdx < asideIdx);
  assert.match(lightbox.slice(closeIdx - 400, closeIdx + 400), /absolute right-3 top-3/);
});
