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
    id: '2026-07-18-runs-show-base-model',
    date: '2026-07-18',
    title: 'Run cards now name the exact base model each LoRA trained on',
    blurb:
      "The Runs hub cards used to show only the family and dataset version — now each one spells out the real base it trained on: the official base by name (e.g. “Z-Image Turbo”, “Krea 2 Raw”), or, when you trained on a custom checkpoint, that file's name (e.g. “bigLove_zt3.safetensors”). Handy when several runs of the same family used different bases. Older runs that never recorded their base just keep the family badge, as before. The “⎘ Share config” export names the base the same way.",
    to: '/cloud',
  },
  {
    id: '2026-07-18-help-mode-rounder',
    date: '2026-07-18',
    title: 'Help mode lands on the exact field — even a folded one',
    blurb:
      "Open a setting from Help search or a Guide's “Open this screen →” and it now reveals the field before highlighting it: a control tucked inside a collapsed “Advanced” panel is opened first, and a field that only appears once a switch is on — like the access token behind LAN access — now points you at that switch instead of scrolling to nothing. New “?” help badges also cover the ▶ Continue dialog and the Dual captions option.",
    to: '/settings',
  },
  {
    id: '2026-07-18-back-up-everything',
    date: '2026-07-18',
    title: 'Back up your whole library — and your settings — in one click',
    blurb:
      "A new “💾 Back up everything” button on the Datasets library packs every dataset (images, captions, statuses, references) plus your settings into a single file, so you can move to a new machine or recover from one without losing anything. It runs in the background with a live progress bar — a big library can be gigabytes — then hands you a download and an “Open folder”. Your API keys and tokens are deliberately left out, so the file is safe to keep around; re-enter them once on the new install. Restoring is the same “📦 Import backup” button: it now accepts the master archive too, rebuilds every dataset without ever overwriting one (name clashes get a “(restored)” suffix) and gives you an honest report of what came back.",
    to: '/datasets',
  },
  {
    id: '2026-07-18-continue-anyway',
    date: '2026-07-18',
    title: 'Train a not-quite-ready dataset on purpose, with your eyes open',
    blurb:
      "When the readiness panel shows a red blocker that's really just a quality warning — too few images for the family, for instance — a “Continue anyway” checkbox now appears under the list. Tick it and the Train button unlocks, with an honest one-line note about the concrete risk (e.g. “7 images will likely overfit; the minimum exists because Z-Image needs variety”). It only ever covers quality guard-rails: genuine impossibilities that would just crash the trainer — zero kept images, a slider with no prompt pair — are never offered the option. The box also un-ticks itself whenever the blockers change, and the run is quietly tagged “acknowledged not-ready” in its saved config.",
    to: '/datasets',
  },
  {
    id: '2026-07-18-flexible-continue',
    date: '2026-07-18',
    title: 'Continue a run from any epoch, for as many steps as you want',
    blurb:
      "The “▶ Continue training” button is now a small dialog: choose how many more steps to train, WHICH checkpoint to resume from — including an earlier, less-cooked epoch (the classic case where step 750 beat the over-cooked 1000) — and optionally adjust the few settings a resume can safely change: save/preview cadence, preview prompts, and the timestep weighting (the two-phase recipe: train balanced, then continue low-noise-leaning to polish texture). Restarting from an earlier checkpoint never touches the run's later saves: they're set aside intact and the continuation writes its own. Works for both local and cloud runs from the Runs hub.",
    to: '/cloud',
  },
  {
    id: '2026-07-18-krea-studio-unblocked',
    date: '2026-07-18',
    title: 'The Krea 2 Turbo Test Studio launches again',
    blurb:
      "The Krea grid was refusing to start for everyone with a “custom node missing” error, because the app asked ComfyUI for a node under the wrong name. Fixed — and when a Studio node really is missing, the message now names exactly which pack to install (ComfyUI-Manager → search “Krea 2 Conditioning”) with a link, instead of just showing a raw class name. The Krea rebalance strength you set is now honored no matter which version of that node pack you installed.",
    to: '/studio',
  },
  {
    id: '2026-07-18-dual-long-short-captions',
    date: '2026-07-18',
    title: 'Train each image with both a long and a short caption',
    blurb:
      "A new Advanced option, “Dual captions (long + short)”, turns on ai-toolkit's native long+short captioning: every image trains with a full caption AND a brief one, so the LoRA leans less on any single wording. The short variant is written for you from the long one when you caption — same rules, no trigger, the identity/concept/aesthetic still kept out — and you can tweak it per image in the ⛶ caption editor. Off by default; local training only for now (cloud runs use the long caption).",
    to: '/datasets',
  },
  {
    id: '2026-07-18-watermark-install-verified',
    date: '2026-07-18',
    title: 'Watermark inpainting turns green the moment it finishes installing',
    blurb:
      "After the one-click install, the feature now reliably switches on right away — no more '✗ Watermark inpainting' lingering on a fresh machine seconds after a successful install. The installer confirms the package actually loads before calling itself done (and warms that first, heavy load so the check is instant), and if an environment is genuinely broken it now tells you why instead of failing silently.",
    to: '/setup',
  },
  {
    id: '2026-07-18-sdxl-studio-without-dmd2',
    date: '2026-07-18',
    title: 'The SDXL Test Studio runs even without the DMD2 accelerator on disk',
    blurb:
      "The SDXL grid used to refuse to launch unless one specific accelerator LoRA (the 4-step DMD2 file) sat in one exact folder — a file plenty of ComfyUI setups don't have. Now the Studio finds that LoRA wherever you keep it, and simply runs without it when it's absent: distilled checkpoints look identical, a full SDXL checkpoint just renders a touch softer, instead of the whole grid refusing to start.",
    to: '/studio',
  },
  {
    id: '2026-07-18-change-dataset-kind',
    date: '2026-07-18',
    title: 'Change a dataset from Character, Concept or Style — after creation',
    blurb:
      "Picked the wrong kind when you started, or want to repurpose a set you already built? The ⚙ Dataset settings modal now lets you switch a dataset between Character, Concept and Style at any time. It's honest, not magic: a confirmation spells out exactly what changes (caption strategy, which panels show, the trigger's role) and what's kept — your images, captions, face scores and training history are never touched. Existing captions keep their old style until you Re-caption.",
    to: '/datasets',
  },
  {
    id: '2026-07-18-one-click-lama-and-queued-installs',
    date: '2026-07-18',
    title: 'Watermark inpainting installs itself — and Setup installs never collide',
    blurb:
      "The Install button for watermark inpainting (LaMa) now sets everything up by itself: it finds a Python 3.10-3.12 on your machine, builds a dedicated environment, installs it, and switches the feature on — no venv to create, no setting to paste. And clicking several Install buttons in a row no longer breaks them: installs now run one at a time in the order you click, so two of them can't corrupt each other's packages. A stray antivirus lock is retried automatically.",
    to: '/datasets?section=curation&panel=watermarks',
  },
  {
    id: '2026-07-18-comfyui-setup-guardrails',
    date: '2026-07-18',
    title: 'Setup tells you straight away if the ComfyUI folder is wrong',
    blurb:
      'The ComfyUI directory field now checks your path as you type: a wrong or empty folder gets a clear reason, and if you point at the launcher/parent folder it offers the real ComfyUI inside it in one click. Leaving it blank is now a conscious choice — Setup shows exactly what you give up (local Klein generation, Test Studio, custom-base training) and what still works before you continue without it.',
    to: '/setup',
  },
  {
    id: '2026-07-17-lora-autocomplete',
    date: '2026-07-17',
    title: 'Pick preset LoRAs from what is actually on disk',
    blurb:
      'Each row of a Klein LoRA preset is now a searchable dropdown of the LoRAs found in your ComfyUI (all folders, extra_model_paths included), with Klein-compatible ones listed first and every file badged by architecture. Free text still works for files not downloaded yet.',
    to: '/settings/engines',
  },
  {
    id: '2026-07-17-suffixes-per-batch',
    date: '2026-07-17',
    title: 'Tweak prompt suffixes between batches, right in the panel',
    blurb:
      'The generation panel now has a ✨ Prompt suffixes accordion — same per-dataset suffixes as the ⚙ Settings modal, editable without leaving the workspace. Adjust the mood, hit Generate, adjust again.',
    to: '/datasets?section=add',
  },
  {
    id: '2026-07-17-captions-uncapped',
    date: '2026-07-17',
    title: 'Captions finish their sentences',
    blurb:
      'Generated captions were silently cut at 800 characters, often mid-word. The cap is gone — JoyCaption and the vision fallback now store their full text, and captions that were truncated in the past get an amber note in the editor pointing at targeted re-captioning.',
    to: '/datasets?section=captions&panel=tools',
  },
  {
    id: '2026-07-17-klein-kv-default',
    date: '2026-07-17',
    title: 'Faster Klein editing — and no Hugging Face token needed',
    blurb:
      'New installs now download the public Klein 9B KV build: up to 2.5× faster multi-reference editing at identical quality, and no license gate to click through. Existing installs keep their current file — nothing re-downloads.',
  },
  {
    id: '2026-07-17-model-file-integrity',
    date: '2026-07-17',
    title: 'Broken model files are caught at Setup, not at generate time',
    blurb:
      'A .safetensors that is really an HTML page (a license-gated download gone wrong), a truncated file or a dead symlink is now detected from its header and explained in plain words — delete and re-download — instead of failing cryptically minutes later.',
  },
  {
    id: '2026-07-17-dataset-delete-fix',
    date: '2026-07-17',
    title: 'Deleting datasets now works on every install',
    blurb:
      'On databases created by older versions, deleting a dataset with Test Studio history could fail with a server error. Fixed for every vintage — deletions land in the app trash as usual, nothing is lost by accident.',
    to: '/datasets',
  },
  {
    id: '2026-07-17-canvas-lora-chain',
    date: '2026-07-17',
    title: 'Dropped images rebuild the full LoRA chain in ComfyUI',
    blurb:
      'Drag a generated image onto the ComfyUI canvas and the reconstructed workflow now shows every LoRA of your preset, not just the last one. (Generation itself was always correct — all LoRAs were applied.)',
  },
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
