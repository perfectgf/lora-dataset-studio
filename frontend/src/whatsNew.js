// ============================================================================
//  🎁 What's new — in-app changelog feed (source of truth)
// ============================================================================
//
//  WHY THIS FILE EXISTS
//  --------------------
//  The update banner only fires on TAGGED releases. Between releases, features
//  ship silently after an "Update & restart" and users never learn they exist.
//  This file backs the in-app "What's new" panel: a short, benefit-oriented feed
//  of what changed, surfaced in the header with an unseen badge.
//
//  This is a FLOW OF NOVELTIES, not documentation. The Guide/Help registry owns
//  docs — from here, point at it with a plain URL if you want to explain rather
//  than jump. Do NOT grow a second help surface in this file.
//
//  ── HOW TO ADD AN ENTRY (do this at the tail of EVERY shipping wave) ─────────
//  Prepend a new object to the TOP of WHATS_NEW (newest first). Shape:
//
//    {
//      id:    'YYYY-MM-DD-short-slug',  // unique, stable, NEVER reused or edited
//      date:  'YYYY-MM-DD',            // ship date (drives ordering + display)
//      title: 'Benefit-first headline', // short, like a Discord announcement
//      blurb: 'One or two sentences, English, oriented on what the user gets.',
//      to:    '/settings/engines',     // OPTIONAL in-app target for "Try it →"
//    }
//
//  RULES
//  -----
//  • Write like the Discord #announcements posts: benefit-first, plain English,
//    no changelog jargon ("Added --allow-crop flag" → "Clean watermarks without
//    ever cropping the shot").
//  • `id` is a PERMANENT handle. Never change or reuse one: the "seen" marker
//    (localStorage) and the unseen badge are keyed on it. Editing an id would
//    re-flag that entry as unseen for everyone who had already read it.
//  • `date` is `YYYY-MM-DD` (zero-padded). Ordering is by date desc, then id
//    desc — so same-day entries stay stable regardless of array position.
//  • `to` is OPTIONAL. Omit it for reliability/plumbing changes with nothing to
//    click. When present it MUST be a valid in-app target (see isValidTarget):
//    a top-level route ('/studio', '/cloud', '/settings/<id>') or a dataset
//    deep-link ('/datasets?section=<id>&panel=<id>'). The section/panel ids are
//    validated against the LIVE navigation registries by whatsNew.test.js, so a
//    stale target fails the test the moment a section is renamed.
//  • Keep the list tidy: tail entries older than a couple of months can be
//    pruned once everyone has cycled through an update or two.
// ============================================================================

import { SETTINGS_SECTIONS } from './components/settings/registry.js';
import { WORKSPACE_SECTIONS } from './components/dataset/workspaceSections.js';

// Newest first. Prepend new waves at the top.
export const WHATS_NEW = [
  {
    id: '2026-07-17-help-mode',
    date: '2026-07-17',
    title: 'A two-way Help mode + a full Settings reference',
    blurb:
      'Flip the ? toggle in the header and help badges appear across the app, each opening the Guide at the exact section that explains that control — and Guide sections link back with "Open this screen →". A new Settings reference chapter documents every setting (role, default, traps), and the Settings search now finds individual settings, not just sections.',
    to: '/guide/settings-reference',
  },
  {
    id: '2026-07-17-watermark-engine',
    date: '2026-07-17',
    title: 'Watermark cleanup that actually restores the image',
    blurb:
      'The Klein-powered clean now prefills the mark with LaMa and refines it, so logos and text vanish instead of smearing. Pick clean-in-place or crop per image, allow auto-crop as a fallback, and restore the original in one click if you do not like a result.',
    to: '/datasets?section=curation&panel=watermarks',
  },
  {
    id: '2026-07-17-scrape-section',
    date: '2026-07-17',
    title: 'A dedicated 🕸 Scrape section',
    blurb:
      'Scanning a gallery is now its own step in every dataset. Paste a gallery URL, pick the images you want, and import them full-frame — then crop each one afterwards right on its tile.',
    to: '/datasets?section=scrape&panel=scan',
  },
  {
    id: '2026-07-17-generation-lora-presets',
    date: '2026-07-17',
    title: 'Generation LoRAs are now named presets',
    blurb:
      'Save the extra LoRAs you generate with as reusable, named presets — no more re-typing filenames and weights, and no automatic NSFW gating getting in your way.',
    to: '/settings/engines',
  },
  {
    id: '2026-07-17-prompt-suffixes',
    date: '2026-07-17',
    title: 'Steer generation with prompt suffixes',
    blurb:
      "Add a reusable creative suffix to every generated variation — globally or per framing — from a dataset's ⚙ Settings. Great for locking in a lighting mood or a lens look across a whole dataset.",
    to: '/datasets?section=add',
  },
  {
    id: '2026-07-17-targeted-recaption',
    date: '2026-07-17',
    title: 'Re-caption only the images you pick',
    blurb:
      'Select a handful of images and re-run captioning on just those, instead of the whole dataset. Fixing a few bad captions no longer means redoing all the good ones.',
    to: '/datasets?section=captions&panel=tools',
  },
  {
    id: '2026-07-17-library-taxonomy',
    date: '2026-07-17',
    title: 'A dataset library sorted by status and size',
    blurb:
      'The datasets page now groups your work by Trained vs Not-trained and tags each one S / M / L by image count — so you can spot at a glance what is ready to train and what still needs images.',
    to: '/datasets',
  },
  {
    id: '2026-07-17-studio-lightbox-nav',
    date: '2026-07-17',
    title: 'Arrow through results in the Test Studio',
    blurb:
      'Open any result in the Test Studio lightbox and step through the whole grid with the arrow keys — compare epochs and strengths without closing and reopening each image.',
    to: '/studio',
  },
  {
    id: '2026-07-17-slider-lora-cloud',
    date: '2026-07-17',
    title: 'Train slider LoRAs in the cloud',
    blurb:
      'Concept-slider training is unlocked on the cloud GPU path, so you can build strength sliders (age, expression, style intensity…) without tying up your local card.',
    to: '/cloud',
  },
  {
    id: '2026-07-17-pillow-self-heal',
    date: '2026-07-17',
    title: 'A smoother, self-healing first launch',
    blurb:
      'Setup now repairs a mixed Pillow install on boot and keeps incompatible ML extras out of the Flask environment — fewer cryptic image errors the first time you run the app.',
    // No `to`: a reliability fix with nothing to click.
  },
];

// ── Ordering ────────────────────────────────────────────────────────────────

// Canonical newest-first order: by date desc, then id desc as a stable
// tiebreaker. Never trust raw array order for "unseen" — sort defensively.
export function sortedEntries(entries = WHATS_NEW) {
  return [...entries].sort((a, b) => {
    if (a.date !== b.date) return a.date < b.date ? 1 : -1;
    if (a.id === b.id) return 0;
    return a.id < b.id ? 1 : -1;
  });
}

export function latestEntryId(entries = WHATS_NEW) {
  const s = sortedEntries(entries);
  return s.length ? s[0].id : null;
}

// ── Unseen logic (drives the badge) ──────────────────────────────────────────
//
//  `lastSeenId` is the id of the newest entry the user has already read.
//    • null / unknown id  → everything is unseen (first visit, or a pruned id:
//      over-notify rather than silently hide new work)
//    • === latest id      → nothing unseen
//    • an older id        → every entry strictly newer than it

export function unseenEntries(lastSeenId, entries = WHATS_NEW) {
  const s = sortedEntries(entries);
  if (!lastSeenId) return s;
  const idx = s.findIndex((e) => e.id === lastSeenId);
  if (idx === -1) return s;
  return s.slice(0, idx);
}

export function unseenCount(lastSeenId, entries = WHATS_NEW) {
  return unseenEntries(lastSeenId, entries).length;
}

export function hasUnseen(lastSeenId, entries = WHATS_NEW) {
  return unseenCount(lastSeenId, entries) > 0;
}

// ── localStorage marker ──────────────────────────────────────────────────────

export const WHATS_NEW_SEEN_KEY = 'lds_whatsNewSeenId';

// DOM CustomEvent names — mirror the codebase's lightweight event bus
// (see App.jsx: 'lds:home', 'lds:update-available'). One modal, many buttons.
export const WHATS_NEW_OPEN_EVENT = 'lds:open-whats-new';
export const WHATS_NEW_SEEN_EVENT = 'lds:whats-new-seen';

function resolveStorage(storage) {
  if (storage) return storage;
  return typeof localStorage !== 'undefined' ? localStorage : null;
}

export function readSeenId(storage) {
  const s = resolveStorage(storage);
  if (!s) return null;
  try {
    return s.getItem(WHATS_NEW_SEEN_KEY);
  } catch {
    return null;
  }
}

// Mark the whole feed as read by pinning the newest id. Returns the id written
// (or null when the feed is empty). Swallows storage failures (private mode /
// denied quota) — the badge simply stays until next time.
export function markAllSeen(storage, entries = WHATS_NEW) {
  const s = resolveStorage(storage);
  const id = latestEntryId(entries);
  if (!s || !id) return id;
  try {
    s.setItem(WHATS_NEW_SEEN_KEY, id);
  } catch {
    /* ignore */
  }
  return id;
}

// ── Navigation targets ("Try it →") ──────────────────────────────────────────

// Param-less top-level routes (mirror App.jsx <Routes>).
const TOP_LEVEL_ROUTES = new Set([
  '/datasets', '/studio', '/cloud', '/guide', '/help', '/setup',
]);

const SETTINGS_IDS = new Set(SETTINGS_SECTIONS.map((s) => s.id));

// Split a target string into { path, section, panel }. Returns null for
// anything that is not an in-app absolute path.
export function parseTarget(to) {
  if (typeof to !== 'string' || !to.startsWith('/')) return null;
  const [path, query = ''] = to.split('?');
  const params = new URLSearchParams(query);
  return { path, section: params.get('section'), panel: params.get('panel') };
}

// Is `to` a target the app can actually navigate to? Validated against the LIVE
// settings + workspace registries so a renamed section is caught by the tests.
export function isValidTarget(to) {
  const t = parseTarget(to);
  if (!t) return false;
  const { path, section, panel } = t;

  // /settings and /settings/<id> — never carry section/panel query params.
  if (path === '/settings') return !section && !panel;
  if (path.startsWith('/settings/')) {
    const id = path.slice('/settings/'.length);
    return SETTINGS_IDS.has(id) && !section && !panel;
  }

  // /datasets with an optional ?section=<id>&panel=<id> workspace deep-link.
  if (path === '/datasets') {
    if (!section) return !panel; // plain /datasets, no orphan panel
    const ws = WORKSPACE_SECTIONS.find((s) => s.id === section);
    if (!ws) return false;
    if (!panel) return true;
    return ws.panels.some((p) => p.id === panel);
  }

  // /guide/<slug> — the Guide owns its own section slugs; any non-empty one is fine.
  if (path.startsWith('/guide/')) {
    return path.length > '/guide/'.length && !section && !panel;
  }

  // Everything else must be a bare, param-less top-level route.
  return TOP_LEVEL_ROUTES.has(path) && !section && !panel;
}
