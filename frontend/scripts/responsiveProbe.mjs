#!/usr/bin/env node
/* 📐 The guard rail the responsive work never had.
 *
 * ── Why this exists ─────────────────────────────────────────────────────────
 *
 * `canvasResponsive.test.js` and its siblings read the JSX as TEXT and match
 * class names against it. That was an honest compromise — `node --test` cannot
 * parse JSX and the suite renders nothing — but it means every one of those
 * assertions can be green while the screen is unusable. They can prove `h-10`
 * is WRITTEN. They cannot see a panel that is mostly air, a box wider than the
 * screen, a control overlapping another, a label cut in half, or a bar eating
 * half the fold.
 *
 * Three responsive regressions shipped through that gap in one week, each found
 * by a person looking at a phone. This script is the thing that should have
 * found them: it RENDERS the app at real device sizes, in each of the states a
 * surface can be in, and measures what came out.
 *
 * ── What it checks ──────────────────────────────────────────────────────────
 *
 *   overflow    nothing sticks out past the right edge of the viewport.
 *   budget      the fixed chrome may not eat the fold.
 *   fill        a panel's ROWS may not be mostly empty. This is what catches
 *               "technically it fits": a shelf whose rows each hold one small
 *               chip in a wide box is rows of air, and it reads as broken even
 *               though every box is inside every other box.
 *   truncated   a label cut by its own box is not a label. A box that fits
 *               BECAUSE its text was chopped passed every other check here.
 *   targets     every interactive control keeps its finger-sized height below
 *               the desktop breakpoint.
 *   overlap     no two elements of the fixed chrome may cover each other.
 *
 * Surfaces are found by attribute, never by class: `data-probe-chrome="<name>"`
 * (the fixed chrome, budgeted), `data-probe-panel` (rows measured for fill),
 * `data-probe-world` (a pannable surface whose contents are not overflow),
 * `data-probe-reading` (opened to READ — not budgeted) and `data-probe-layer`
 * (a lightbox or dialog that covers the page BY DESIGN — not budgeted, paired
 * with nothing in the overlap check). Pages: #/canvas, #/bank, #/datasets and
 * #/dataset/studio/<id>; anything else is measured at rest only.
 *
 * ── Running it ──────────────────────────────────────────────────────────────
 *
 *   npm run probe:responsive -- --url http://127.0.0.1:5173/#/canvas
 *   npm run probe:responsive -- --url http://127.0.0.1:5173/#/bank
 *   npm run probe:responsive -- --url http://127.0.0.1:5173/#/datasets
 *   npm run probe:responsive -- --url http://127.0.0.1:5173/#/dataset/studio/<id>
 *
 * Options: --viewports 360x800,844x390,1280x800   --states shelf,layouts
 *          --json   --quiet
 *
 * Exit codes are THREE, on purpose: 0 clean, 1 violations found, 2 could not
 * run. A probe that cannot run must never look like a probe that passed — the
 * silent skip is its own bug (it is how a privacy guard once passed for weeks
 * while checking nothing), so 2 is loud and separate from 0. For the same
 * reason every run prints what it COVERED: "no violations" over four surfaces
 * on an empty board is not the same answer as "no violations" over eleven.
 */

import { createRequire } from 'node:module';
import process from 'node:process';

const require = createRequire(import.meta.url);

// ── the contract, in numbers ────────────────────────────────────────────────
// Each of these was a real symptom before it was a threshold. Change one only
// with a measurement in hand, and say which screen you measured on.

/** Share of the fold the fixed chrome may occupy — TWO budgets, because they
 *  answer two different questions, and one number for both is how a regression
 *  hides. RESTING is what the board costs you before you touch anything: it is
 *  not negotiable, since you did not ask for any of it. OPEN is what it costs
 *  with something deliberately unfolded on it — you did ask for that, so it may
 *  cost more, but a shelf that has taken the fold is a shelf you cannot use
 *  against the board it describes.
 *
 *  Measured at 400×800: the shelf with the gesture manual printed in it reached
 *  51 % and the board vanished behind its own instructions.
 *
 *  ⚠️ `open` is 50 and not 45 because 360×800 measures 49 % with a shelf whose
 *  every row PASSES the fill check below — the action chips and the machine
 *  readouts, wrapping three deep because the screen is 360 px wide. That is
 *  content, not waste, and `fill` is the check that tells the two apart. So if
 *  this number ever wants raising again, the answer is to take something OUT of
 *  the panel, not to move the line. */
const MAX_CHROME_SHARE = { resting: 0.28, open: 0.50 };

/** How full a panel row has to be before it earns its own line. A row holding
 *  one 200-px chip in a 900-px panel is 22 % full, and two of those stacked are
 *  what "the box is huge and empty" looks like. Rows that are the panel's only
 *  row are exempt: there is nothing to merge them with. */
const MIN_ROW_FILL = 0.35;

/** …unless the panel is narrow enough that the row genuinely could not share.
 *  Below this width a lone chip per row is the honest layout, not waste. */
const NARROW_PANEL = 420;

/** Finger-sized, below `lg`. A 36-px control is under the ~40 px a fingertip
 *  actually lands on, and a miss goes to whatever is behind it. */
const MIN_TOUCH_PX = 40;
const DESKTOP_BREAKPOINT = 1024;

/** Sub-pixel slack before a clipped text counts as truncated. Measured: a
 *  "Loading…" span reports 47 vs 45 at some zoom levels with nothing wrong. */
const TRUNCATION_SLACK = 4;

/** …and a text box narrower than this is not truncated, it is FOLDED — the
 *  canvas search input collapses to 1 px wide and would otherwise report a
 *  149-px overflow on every single run. */
const MIN_TEXT_WIDTH = 24;

/** REAL devices, width × height, because the fold has two dimensions and this
 *  script only ever asked about one. Measured the day the second was added:
 *  740×390 — a phone held sideways — put the ⋯ shelf at 67 % of the fold while
 *  the very same shelf measured 27 % at 904×800 and passed. A viewport list of
 *  widths is a list that cannot see a landscape phone. */
const DEFAULT_VIEWPORTS = [
  [360, 800],   // small Android, portrait
  [412, 915],   // common Android, portrait
  [844, 390],   // phone held SIDEWAYS — the one that was missing
  [768, 1024],  // tablet, portrait
  [1280, 800],  // laptop
];

/** What a page can be put INTO, because a surface at rest is not the surface
 *  people complain about. Every state is measured on a freshly loaded page, so
 *  one state can never leak into the next — an opened menu that fails to close
 *  would otherwise silently join every measurement after it.
 *
 * `open` is a list of selectors clicked in order. A selector that is absent or
 *  invisible SKIPS the state rather than failing it: a board with no runs has
 *  no Layouts menu to open, and that is a fact about the fixture, not a defect
 *  — which is exactly why the coverage report at the bottom names what it
 *  actually reached. */
const PAGES = {
  '#/canvas': {
    label: 'Canvas',
    states: [
      { name: 'resting', open: [] },
      { name: 'shelf', open: ['[aria-label="More board tools"]'] },
      { name: 'shelf+help',
        open: ['[aria-label="More board tools"]', '[data-testid="canvas-gestures-info"]'] },
      { name: 'layouts',
        open: ['[aria-label="More board tools"]', '[data-testid="canvas-layout-presets"] summary'] },
      { name: 'datasets-menu', open: ['[data-testid="canvas-filter-datasets"]'] },
      { name: 'search', open: ['[data-testid="canvas-filter-search-toggle"]'] },
    ],
  },

  /* The three pages people actually live in. Each opens on a LIST (banks,
     datasets) or needs an id (the Studio), and the workspace behind the list is
     what the complaints are about — so `prime` opens the first item once per
     viewport, and the app's own localStorage keeps it open for every state
     after that (each state still gets a fresh load). A prime whose control is
     absent is not a failure: the workspace is already open, or the instance
     holds nothing to open, and the coverage line says which. */
  '#/bank': {
    label: 'Bank',
    prime: ['[aria-label^="Open the bank"]'],
    states: [
      { name: 'resting', open: [] },
      // The ☰ button exists only where the rail cannot sit beside the grid, so
      // this state measures the DRAWER on a phone and is skipped on a desktop.
      { name: 'rail', open: ['[aria-controls="bank-filter-rail"]'] },
      { name: 'passes', open: ['[aria-controls="bank-passes-panel"]'] },
      { name: 'auto-reject', open: ['button:has-text("Auto-reject")'] },
      { name: 'review', open: ['[aria-label^="Review from"]'] },
    ],
  },
  '#/datasets': {
    label: 'Datasets',
    prime: ['[aria-label^="Open the dataset"]'],
    states: [
      { name: 'resting', open: [] },
      { name: 'more', open: ['summary:has-text("More")'] },
      // Two navs carry this label (the phone chip rail and the desktop rail);
      // `:visible` picks whichever this viewport shows.
      // `text=Images` is a SUBSTRING match and lands on "Add images" first; the
      // button is named exactly, and the hidden twin rail is excluded by :visible.
      { name: 'images', open: ['nav[aria-label="Dataset sections"]:visible >> button:has-text("Images"):not(:has-text("Add"))'] },
      { name: 'training', open: ['nav[aria-label="Dataset sections"]:visible >> button:has-text("Training")'] },
      { name: 'lightbox', open: ['nav[aria-label="Dataset sections"]:visible >> button:has-text("Images"):not(:has-text("Add"))',
        '[aria-label^="Inspect"]'] },
    ],
  },
  '#/dataset/studio': {
    label: 'Test Studio',
    states: [
      { name: 'resting', open: [] },
      // The bottom bar's first shortcut: it reveals and scrolls to a section —
      // the page in the state a shortcut leaves it in.
      { name: 'shortcut', open: ['[data-probe-chrome="action-bar"] button'] },
    ],
  },
};

/** A page the map says nothing about is still worth measuring — at rest, and
 *  with nothing opened. Better a thin answer than no answer, as long as the
 *  coverage report says which one you got. */
const UNKNOWN_PAGE = { label: 'page', states: [{ name: 'resting', open: [] }] };

// ── argument plumbing ───────────────────────────────────────────────────────

function parseViewports(text) {
  return text.split(',').map((pair) => {
    const [w, h] = pair.trim().toLowerCase().split('x').map(Number);
    return [w, h || 800];
  }).filter(([w]) => Number.isFinite(w) && w > 0);
}

function parseArgs(argv) {
  const out = {
    url: 'http://127.0.0.1:5173/#/canvas',
    viewports: DEFAULT_VIEWPORTS, states: null, json: false, quiet: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === '--url') out.url = argv[++i];
    else if (a === '--viewports') out.viewports = parseViewports(argv[++i]);
    // Kept so the older invocation still works; a bare width means "at 800
    // tall", which is precisely the blind spot --viewports exists to close.
    else if (a === '--widths') out.viewports = parseViewports(argv[++i]);
    else if (a === '--height') {
      const h = Number(argv[++i]);
      out.viewports = out.viewports.map(([w]) => [w, h]);
    } else if (a === '--states') out.states = argv[++i].split(',').map((s) => s.trim());
    else if (a === '--json') out.json = true;
    else if (a === '--quiet') out.quiet = true;
  }
  return out;
}

function cannotRun(why, how) {
  console.error('');
  console.error('  ⚠️  RESPONSIVE PROBE DID NOT RUN — this is NOT a pass.');
  console.error(`      ${why}`);
  if (how) console.error(`      ${how}`);
  console.error('');
  process.exit(2);
}

/** chrome-headless-shell has no windowed mode at all, so no window can appear
 *  on anyone's desktop while this runs. Its version directory changes with the
 *  Playwright release, so it is discovered rather than pinned. */
function findHeadlessShell(fs, path) {
  const roots = [
    process.env.PLAYWRIGHT_BROWSERS_PATH,
    process.env.LOCALAPPDATA && path.join(process.env.LOCALAPPDATA, 'ms-playwright'),
    process.env.HOME && path.join(process.env.HOME, '.cache', 'ms-playwright'),
  ].filter(Boolean);
  for (const root of roots) {
    let entries;
    try { entries = fs.readdirSync(root); } catch { continue; }
    const dir = entries.filter((n) => n.startsWith('chromium_headless_shell-')).sort().pop();
    if (!dir) continue;
    for (const rel of [
      'chrome-headless-shell-win64/chrome-headless-shell.exe',
      'chrome-headless-shell-linux/chrome-headless-shell',
      'chrome-headless-shell-mac/chrome-headless-shell',
    ]) {
      const exe = path.join(root, dir, rel);
      if (fs.existsSync(exe)) return exe;
    }
  }
  return null;
}

// ── the measurements, taken inside the page ─────────────────────────────────
/* One evaluate() rather than a call per element: every round trip is a chance
   for the layout to have moved between two questions about it. */
const MEASURE = ({ minRowFill, narrowPanel, minTouch, desktopBreakpoint,
  truncationSlack, minTextWidth, expectChrome }) => {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const violations = [];
  const box = (el) => {
    const r = el.getBoundingClientRect();
    return { x: r.left, y: r.top, w: r.width, h: r.height, right: r.right, bottom: r.bottom };
  };
  const name = (el) => {
    const id = el.getAttribute('data-testid');
    if (id) return `[${id}]`;
    const label = el.getAttribute('aria-label');
    if (label) return `"${label}"`;
    const text = (el.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 28);
    return text ? `${el.tagName.toLowerCase()} "${text}"` : el.tagName.toLowerCase();
  };

  /* ⚠️ If the markers are gone the probe measures NOTHING and reports a clean
     run — the exact shape of the silent skip this script's exit codes exist to
     avoid. So their absence is itself a violation, loudly. */
  /* VISIBLE chrome only. A marker inside a closed <details> (the ⋯ More menu)
     keeps a measurable box — Chrome parks closed details content under
     content-visibility: hidden, whose boxes still answer getBoundingClientRect
     — and it was billed to the fold at rest while nobody could see it. */
  const visible = (el) => (typeof el.checkVisibility === 'function'
    ? el.checkVisibility({ contentVisibilityAuto: true, visibilityProperty: true })
    : true);
  const chromeEls = [...document.querySelectorAll('[data-probe-chrome]')].filter(visible);
  const chrome = chromeEls.map((el) => ({ el, b: box(el) }));
  // Only a page that is SUPPOSED to carry markers may alarm on their absence —
  // that alarm exists to catch a mapped page whose markers were refactored
  // away. A page outside PAGES has no chrome by design and is measured at rest
  // (overflow / targets / truncation over the whole DOM, zero chrome surfaces);
  // its coverage line says so instead.
  if (!chrome.length && expectChrome) {
    violations.push({ kind: 'unmarked', el: 'data-probe-chrome',
      detail: 'no chrome markers found — the probe measured nothing, which is NOT a pass' });
  }

  // What this run actually looked at, so a thin answer cannot pass for a full
  // one. Names, not just counts: a missing surface is the interesting part.
  const coverage = {
    chrome: chromeEls.map((el) => el.getAttribute('data-probe-chrome')),
    panels: 0, rows: 0, controls: 0, texts: 0,
  };

  // ── overflow ──────────────────────────────────────────────────────────────
  /* The right edge, not scrollWidth: `overflow-x: hidden` on an ancestor cuts a
     too-wide box off without ever growing the scroll width, so the page looks
     fine to scrollWidth while the content is sliced. */
  const exemptFromOverflow = (el) => {
    for (let p = el.parentElement; p; p = p.parentElement) {
      // A pannable/zoomable surface holds a world larger than its frame BY
      // DESIGN — that is what panning is for — so its contents are not a
      // layout defect and the frame's own clip is the intended behaviour.
      if (p.hasAttribute('data-probe-world')) return true;
      const s = getComputedStyle(p);
      if (s.overflowX === 'auto' || s.overflowX === 'scroll') return true;
    }
    return false;
  };
  for (const el of document.querySelectorAll('body *')) {
    const b = box(el);
    if (b.w === 0 || b.h === 0) continue;
    if (b.right <= vw + 1) continue;
    if (exemptFromOverflow(el)) continue;
    violations.push({ kind: 'overflow', el: name(el),
      detail: `right edge ${Math.round(b.right)} > viewport ${vw}` });
  }

  // ── truncation ────────────────────────────────────────────────────────────
  /* A box that fits BECAUSE its text was chopped passes every other check in
     this file. "⏏ Undeploy…" is inside its container, inside the viewport,
     overlapping nothing — and unreadable.
   *
   * Three exemptions, each earned:
   *  • a text narrower than `minTextWidth` is FOLDED, not truncated. The canvas
   *    search input collapses to 1 px and would report a 149-px overflow on
   *    every run — one false positive per run is how a check gets switched off.
   *  • no clip, no truncation: an element whose overflow is visible SPILLS, and
   *    spilling is the overflow check's business, not this one's.
   *  • a `title` or `aria-label` carrying the full string is this codebase's
   *    OWN convention for a deliberate ellipsis (`truncate` + `title` on lane
   *    names). The words are still reachable, so it is a layout choice, not a
   *    loss. Ignoring that convention would flood the report with decisions
   *    somebody already made on purpose. */
  const fullTextRescue = (el, text) => {
    const needle = text.slice(0, 12).toLowerCase();
    for (let p = el; p; p = p.parentElement) {
      const t = `${p.getAttribute?.('title') || ''} ${p.getAttribute?.('aria-label') || ''}`;
      if (t.toLowerCase().includes(needle)) return true;
      if (p.hasAttribute?.('data-probe-chrome')) break;
    }
    return false;
  };
  for (const el of document.querySelectorAll('[data-probe-chrome] *')) {
    if (el.children.length) continue;
    const text = (el.textContent || '').trim();
    if (!text) continue;
    const b = box(el);
    if (b.w < minTextWidth || b.h === 0) continue;
    coverage.texts += 1;
    const s = getComputedStyle(el);
    const clips = s.overflowX === 'hidden' || s.overflowX === 'clip'
      || s.textOverflow === 'ellipsis';
    if (!clips) continue;
    if (el.scrollWidth <= el.clientWidth + truncationSlack) continue;
    if (fullTextRescue(el, text)) continue;
    violations.push({ kind: 'truncated', el: name(el),
      detail: `"${text.slice(0, 30)}" needs ${el.scrollWidth}px, has ${el.clientWidth}px `
        + '— widen it, shorten the words, or give it a title with the full string' });
  }

  /* ── the fixed chrome: what is permanently in front of the content ────────
   *
   * ⚠️ A surface marked `data-probe-reading` is NOT counted against the budget,
   * and the distinction is the whole point rather than an exemption. The budget
   * asks "how much of the fold is IN YOUR WAY". A toolbar is in your way — you
   * did not ask for it and you cannot read the board through it. A dialog you
   * opened in order to READ is not in your way: it IS the thing you asked for,
   * one tap from being gone, and on a phone held sideways a 500-character
   * manual is half a screen of text by arithmetic, not by bad layout.
   *
   * Measured, which is why this exists: at 844×390 the ⓘ bubble put the total
   * at 100 % of the fold and the report said "over budget" about a paragraph
   * somebody had deliberately opened. Charging it to the chrome budget would
   * have forced the manual to be cut for a reason that was never true.
   *
   * It stays a measured surface for everything else — overlap, truncation and
   * target size all still apply to it, and the overlap check is what caught it
   * growing up into the filter bar on the very same screen. */
  /* `data-probe-layer` is the other kind of surface that is not IN YOUR WAY: a
     lightbox or a dialog that covers the page BY DESIGN. You opened it, it is the
     whole screen on purpose, and one tap puts it away. It is measured like any
     chrome — targets, truncation, fill — but charged to no budget, and the
     overlap check below pairs it with nothing: a layer overlapping the header
     is what a layer is, not a defect. */
  let chromeH = 0;
  const chromeParts = [];
  for (const { el, b } of chrome) {
    const exempt = el.hasAttribute('data-probe-reading') ? 'reading'
      : el.hasAttribute('data-probe-layer') ? 'layer' : null;
    chromeParts.push({ name: el.getAttribute('data-probe-chrome'), h: Math.round(b.h), exempt });
    if (exempt) continue;
    chromeH += b.h;
  }

  // ── overlap, among the chrome only ────────────────────────────────────────
  for (let i = 0; i < chrome.length; i += 1) {
    for (let j = i + 1; j < chrome.length; j += 1) {
      if (chrome[i].el.hasAttribute('data-probe-layer')
        || chrome[j].el.hasAttribute('data-probe-layer')) continue;
      const a = chrome[i].b; const c = chrome[j].b;
      const ox = Math.min(a.right, c.right) - Math.max(a.x, c.x);
      const oy = Math.min(a.bottom, c.bottom) - Math.max(a.y, c.y);
      if (ox > 1 && oy > 1) {
        violations.push({ kind: 'overlap', el: `${name(chrome[i].el)} × ${name(chrome[j].el)}`,
          detail: `${Math.round(ox)}×${Math.round(oy)} px of overlap` });
      }
    }
  }

  // ── fill: a panel's rows may not be mostly air ────────────────────────────
  for (const panel of document.querySelectorAll('[data-probe-panel]')) {
    const pb = box(panel);
    if (pb.w < narrowPanel) continue;
    const style = getComputedStyle(panel);
    const inner = pb.w - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
    const rows = [...panel.children].filter((el) => box(el).h > 0);
    if (rows.length < 2) continue;
    coverage.panels += 1;
    coverage.rows += rows.length;
    for (const row of rows) {
      /* ATOMS, not direct children, and not the row's own box either. The row is
         usually a full-width flex container, so its box always fills and would
         never flag; its direct children can be one <span> holding an icon
         inside a wide button, which flags at 2 % and reads like a bug in the
         probe. What is actually being measured is INK: the controls and the
         text runs, wherever they sit in the tree. */
      const atoms = [];
      const walk = (el) => {
        const b = box(el);
        if (b.h === 0 || b.w === 0) return;
        const tag = el.tagName.toLowerCase();
        const isControl = tag === 'button' || tag === 'summary' || tag === 'input'
          || tag === 'select' || (tag === 'a' && el.hasAttribute('href'));
        if (isControl || !el.children.length) { atoms.push(b); return; }
        for (const kid of el.children) walk(kid);
      };
      walk(row);
      const rb = box(row);
      const spans = atoms.length ? atoms : [rb];
      const left = Math.min(...spans.map((b) => b.x));
      const right = Math.max(...spans.map((b) => b.right));
      const fill = (right - left) / inner;
      if (fill < minRowFill) {
        violations.push({ kind: 'fill', el: `${name(panel)} → ${name(row)}`,
          detail: `row is ${Math.round(fill * 100)}% full of ${Math.round(inner)} px `
            + `(${Math.round(rb.h)} px tall) — merge it with a neighbour` });
      }
    }
  }

  // ── targets ───────────────────────────────────────────────────────────────
  const controls = [...document.querySelectorAll(
    '[data-probe-chrome] button, [data-probe-chrome] summary, [data-probe-chrome] a[href]')]
    .filter((el) => box(el).h > 0);
  coverage.controls = controls.length;
  if (vw < desktopBreakpoint) {
    for (const el of controls) {
      const b = box(el);
      if (b.h + 0.5 < minTouch) {
        violations.push({ kind: 'target', el: name(el),
          detail: `${Math.round(b.h)} px tall, needs ${minTouch}` });
      }
    }
  }

  return { vw, vh, chromeH, chromeShare: chromeH / vh, chromeParts, violations, coverage };
};

// ── driving the page ────────────────────────────────────────────────────────

async function main() {
  const args = parseArgs(process.argv.slice(2));

  let fs; let path; let chromium;
  try {
    fs = require('node:fs'); path = require('node:path');
    ({ chromium } = require('playwright-core'));
  } catch {
    cannotRun('playwright-core is not installed.',
      'npm i -D playwright-core   (the browser itself is found below, not downloaded here)');
  }
  const executablePath = findHeadlessShell(fs, path);
  if (!executablePath) {
    cannotRun('no chrome-headless-shell found in the Playwright browser cache.',
      'npx playwright install chromium   (or set PLAYWRIGHT_BROWSERS_PATH)');
  }

  // The hash path without its query, matched against PAGES by LONGEST prefix:
  // `#/dataset/studio/7` is the Studio page, `#/datasets` is not `#/dataset`.
  const hashPath = ((args.url.split('#')[1] || '/').split('?')[0]);
  const route = Object.keys(PAGES)
    .filter((k) => {
      const p = k.slice(1);
      return hashPath === p || hashPath.startsWith(p + '/');
    })
    .sort((a, b) => b.length - a.length)[0] || null;
  const pageSpec = (route && PAGES[route]) || UNKNOWN_PAGE;
  let states = pageSpec.states;
  if (args.states) states = states.filter((s) => args.states.includes(s.name));
  if (!states.length) cannotRun(`no state named "${args.states?.join(', ')}" on ${route}`);

  const browser = await chromium.launch({ executablePath });
  const findings = [];
  const rows = [];
  const seenSurfaces = new Set();
  const skipped = [];
  let measured = 0;

  try {
    for (const [width, height] of args.viewports) {
      const ctx = await browser.newContext({
        // Emulation.setDeviceMetricsOverride, which reports the size it was
        // asked for. A --window-size below ~500 px does NOT: Chrome floors the
        // window and renders wider than requested, which turns a "400 px proof"
        // into a cropped 500 px one.
        viewport: { width, height }, deviceScaleFactor: 1,
      });
      const page = await ctx.newPage();

      /* Open the workspace behind the list, ONCE per viewport. The app keeps the
         opened item in localStorage, which lives in this context, so every state
         after this still gets its fresh load and lands straight in the workspace.
         Absence of the control is reported, never failed: the coverage line is
         where "nothing to open on this instance" has to show up. */
      if (pageSpec.prime?.length) {
        try {
          await page.goto('about:blank');
          await page.goto(args.url, { waitUntil: 'domcontentloaded', timeout: 20000 });
          await page.waitForTimeout(900);
          for (const selector of pageSpec.prime) {
            const el = page.locator(selector).first();
            if ((await el.count()) && (await el.isVisible())) {
              await el.click({ timeout: 4000 });
              await page.waitForTimeout(600);
            } else {
              skipped.push(`${width}×${height} prime (${selector} absent)`);
            }
          }
        } catch (e) {
          skipped.push(`${width}×${height} prime (${e.message.split('\n')[0]})`);
        }
      }

      for (const state of states) {
        /* A FRESH load per state, deliberately. Closing a menu and opening the
           next is one more thing that can silently not happen, and a state that
           leaked into the next would corrupt every measurement after it without
           failing anything. */
        let reachable = true;
        try {
          /* ⚠️ about:blank FIRST, and it is not belt-and-braces. The app is a
             HASH router, so `goto('…#/canvas')` while already on `#/canvas` is
             a same-document navigation: the page does NOT reload and every bit
             of React state survives. Measured on this script's first run — the
             shelf left open by one state was toggled SHUT by the next state's
             first click, so every other state silently found nothing to open
             and was skipped. Only the coverage report caught it. */
          await page.goto('about:blank');
          await page.goto(args.url, { waitUntil: 'domcontentloaded', timeout: 20000 });
          // Wait for the chrome rather than for a fixed sleep: a fixed sleep is
          // either too short on a cold load or wasted on a warm one, and this
          // now runs thirty times instead of six.
          await page.waitForSelector('[data-probe-chrome]', { timeout: 15000 });
          await page.waitForTimeout(900);
        } catch (e) {
          // A page with NO probe markers (anything outside PAGES — Settings,
          // Guide…) never grows a [data-probe-chrome], so waiting only for the
          // chrome reported every unknown page as unreachable — breaking the
          // header's promise that such pages are measured at rest. Fall back to
          // "the app painted something": overflow/targets/truncation still
          // measure the DOM with zero chrome surfaces.
          try {
            await page.waitForFunction(
              () => ((document.getElementById('root') || {}).childElementCount || 0) > 0,
              { timeout: 5000 });
            await page.waitForTimeout(900);
          } catch {
            reachable = false;
            findings.push({ width, height, state: state.name, kind: 'load',
              el: args.url, detail: e.message.split('\n')[0] });
          }
        }
        if (!reachable) continue;

        let opened = true;
        for (const selector of state.open) {
          const el = page.locator(selector).first();
          try {
            if (!(await el.count()) || !(await el.isVisible())) { opened = false; break; }
            await el.click({ timeout: 4000 });
            await page.waitForTimeout(350);
          } catch { opened = false; break; }
        }
        if (!opened) {
          // NOT a violation: a board with no runs has no Layouts menu to open.
          // It is reported all the same, because "clean over four states" and
          // "clean over six" are different answers and only one is worth much.
          skipped.push(`${width}×${height} ${state.name}`);
          continue;
        }

        const result = await page.evaluate(MEASURE, {
          minRowFill: MIN_ROW_FILL, narrowPanel: NARROW_PANEL,
          minTouch: MIN_TOUCH_PX, desktopBreakpoint: DESKTOP_BREAKPOINT,
          truncationSlack: TRUNCATION_SLACK, minTextWidth: MIN_TEXT_WIDTH,
          expectChrome: pageSpec !== UNKNOWN_PAGE,
        });
        measured += 1;
        for (const s of result.coverage.chrome) seenSurfaces.add(s);

        const budget = MAX_CHROME_SHARE[state.name === 'resting' ? 'resting' : 'open'];
        rows.push({ width, height, state: state.name, chromeH: Math.round(result.chromeH),
          share: result.chromeShare, budget, cov: result.coverage, parts: result.chromeParts });
        if (result.chromeShare > budget) {
          // Name the surfaces, tallest first, so the report says WHAT to shrink.
          const parts = result.chromeParts.filter((p) => !p.exempt)
            .sort((a, b) => b.h - a.h).map((p) => `${p.name} ${p.h}px`).join(', ');
          findings.push({ width, height, state: state.name, kind: 'budget', el: 'fixed chrome',
            detail: `${Math.round(result.chromeH)} px = ${Math.round(result.chromeShare * 100)}% `
              + `of the ${result.vh}px fold, over the ${Math.round(budget * 100)}% budget `
              + `(${parts})` });
        }
        for (const v of result.violations) {
          findings.push({ width, height, state: state.name, ...v });
        }
      }
      await ctx.close();
    }
  } finally {
    await browser.close();
  }

  /* A clean run STAMPS the tree, which is what closes the loop: layoutGuard's
     Stop hook compares that stamp against the layout-dirty mark and refuses to
     let a layout change be called done until this file has been written on the
     tree that change is in. Only on a clean run, and only when the probe
     actually measured something — `unmarked` is itself a violation, so a tree
     whose markers were deleted can never stamp itself green. */
  if (!findings.length && measured > 0) {
    try {
      const { execFileSync } = require('node:child_process');
      const { fileURLToPath } = require('node:url');
      const guard = path.join(path.dirname(fileURLToPath(import.meta.url)), 'layoutGuard.mjs');
      execFileSync(process.execPath, [guard, 'green'], { stdio: 'ignore' });
    } catch { /* the guard is optional; the probe's own verdict is not */ }
  }

  if (args.json) {
    console.log(JSON.stringify({ rows, findings, measured, skipped,
      surfaces: [...seenSurfaces] }, null, 2));
    process.exit(findings.length ? 1 : 0);
  }

  console.log('');
  if (!args.quiet) {
    console.log('  viewport      state         chrome   share   budget');
    for (const r of rows) {
      console.log(`  ${`${r.width}×${r.height}`.padStart(10)}    ${r.state.padEnd(12)}  `
        + `${String(r.chromeH).padStart(5)}px   ${String(Math.round(r.share * 100)).padStart(3)}%   `
        + `${String(Math.round(r.budget * 100)).padStart(4)}%`
        + (r.share > r.budget ? '   ← over' : ''));
    }
    console.log('');
  }

  /* ── coverage, ALWAYS, clean run or not ──────────────────────────────────
     "No violations" over four surfaces on an empty board is not the same
     answer as "no violations" over eleven, and only one of them is worth
     anything. Printing this is what stops a thin run reading as a full one —
     the same failure the exit codes guard at the other end. */
  const totals = rows.reduce((acc, r) => ({
    panels: acc.panels + r.cov.panels, rows: acc.rows + r.cov.rows,
    controls: acc.controls + r.cov.controls, texts: acc.texts + r.cov.texts,
  }), { panels: 0, rows: 0, controls: 0, texts: 0 });
  console.log(`  covered: ${measured} measurement(s) — up to ${args.viewports.length} viewport(s) `
    + `× ${states.length} state(s) of ${pageSpec.label}`);
  console.log(`           surfaces: ${[...seenSurfaces].sort().join(', ') || 'NONE'}`);
  console.log(`           ${totals.panels} panel(s), ${totals.rows} row(s), `
    + `${totals.controls} control(s), ${totals.texts} text run(s)`);
  if (skipped.length) {
    console.log(`  NOT covered: ${skipped.length} state(s) whose control was absent — `
      + `${skipped.slice(0, 5).join('; ')}${skipped.length > 5 ? '; …' : ''}`);
  }
  console.log('');

  if (!findings.length) {
    console.log('  ✅ no responsive violations');
  } else {
    for (const f of findings) {
      console.log(`  ✗ ${`${f.width}×${f.height}`.padStart(10)} ${(f.state || '-').padEnd(12)} `
        + `${f.kind.padEnd(9)} ${f.el}`);
      console.log(`               ${f.detail}`);
    }
    console.log('');
    console.log(`  ${findings.length} violation(s)`);
  }
  console.log('');
  process.exit(findings.length ? 1 : 0);
}

main().catch((e) => cannotRun(e.message, e.stack?.split('\n')[1]?.trim()));
