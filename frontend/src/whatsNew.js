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
    id: '2026-07-19-bank-sort-resolution',
    date: '2026-07-19',
    title: '📐 Sort your Bank by resolution',
    blurb:
      "The 🗃️ Bank grid gains a Sort control next to the tiles: order every image by resolution, biggest or smallest first. It ranks by megapixels (width×height), so a crisp 900×900 outranks a stretched 1200×300 — the right way to skim a mixed dump for the sharpest, most trainable shots. It stacks on top of every filter and search you already have, and 'Select all in filter' follows the same order. Images not scanned yet sink to the end.",
    to: '/bank',
  },
  {
    id: '2026-07-19-caption-lab',
    date: '2026-07-19',
    title: '🧪 Caption Lab — try caption models side by side before you commit',
    blurb:
      "Open any image's caption editor and switch to the new 🧪 Caption Lab tab: line up to four caption configs — engine (JoyCaption or an Ollama vision model), which model, and the nude/sexual vocabulary register (Explicit / Clinical / Safe) — and run them on THIS image. They generate one after another (the GPU stays serialized, never fighting a training run), then land as cards side by side with the caption, its length and how long it took, next to your current caption for reference. A/B your NSFW captioners without guessing. When one wins, ✓ Keep this one drops it straight into the editor, or ⚙️ Make default stores that config as the dataset's caption method. Nothing is saved until you pick — it's a bench, not a batch.",
    to: '/datasets',
  },
  {
    id: '2026-07-19-graph-modal-visible-from-checkpoints',
    date: '2026-07-19',
    title: '◉ Graph now opens from the Checkpoints panel',
    blurb:
      "Opening ◉ Graph from the Checkpoints & LoRAs section did nothing — no window, no error. The run-and-checkpoints graph was being drawn inside the hidden Training section, so it never showed. It now pops up over the page from wherever you open it, with your dataset's runs and their saved checkpoints.",
    to: '/datasets?section=checkpoints',
  },
  {
    id: '2026-07-19-continue-lr-factor',
    date: '2026-07-19',
    title: 'Finish a run gentler with a lower learning rate',
    blurb:
      "The ▶ Continue training dialog gains one more safe knob under “Adjust settings”: the learning rate. Resume the epoch that held up best, then finish at half (polish) or a tenth (gentle finish) of the current rate — a smaller rate polishes fine texture without moving the identity, the learning-rate pendant of the low-noise timestep recipe. The values are factors of this run's rate, and the dialog shows the resulting number (a 1e-4 run → 5e-5 or 1e-5). Works for local and cloud runs; hidden for Prodigy, which adapts its own rate.",
    to: '/cloud',
  },
  {
    id: '2026-07-19-bank-stop-keeps-progress',
    date: '2026-07-19',
    title: '⏹ Stopping a Bank face or score pass no longer loses your progress',
    blurb:
      "Stopping the Image bank's 👥 Group by person or ✨ Score pass mid-run used to feel like it threw everything away and left the bar blank. It never actually lost the finished work — the embeddings were cached — but nothing said so. Now Stop asks the pass to finish the image it's on, flush its cache and bow out cleanly, then tells you exactly where it landed: “Stopped — 1 240 face embeddings cached (760 remaining); relaunch to finish and cluster.” Relaunch and it picks up from the cache — the detail even reads “resuming — 1 240 of 2 000 already cached” so you can see it's continuing, not starting over. Same for the passes inside 🚀 Launch all.",
    to: '/bank',
  },
  {
    id: '2026-07-19-caption-stop-actually-stops',
    date: '2026-07-19',
    title: '⏹ Stop now stops captioning right away',
    blurb:
      "Hitting Stop during a caption run used to flip the button to “Stopping…” but the JoyCaption pass kept churning through every remaining image before it actually halted. Now Stop is honoured the moment the current image finishes: what's already captioned is kept, the rest is left untouched, and the GPU is handed straight back to ComfyUI — on character and concept datasets alike.",
    to: '/datasets',
  },
  {
    id: '2026-07-19-explicit-vocabulary-on-concepts',
    date: '2026-07-19',
    title: '🔞 Explicit captions now work on concept datasets too',
    blurb:
      "The Captions ⚙️ Options “Explicit” vocabulary preset was reaching the first captioning pass but not the refine step that concept datasets rely on, so crude terms got quietly smoothed back out. That path now carries your chosen register end to end — pick Explicit (with an uncensored vision model) and the words stay in, while the recurring concept is still left unspoken so it binds to your trigger.",
    to: '/datasets',
  },
  {
    id: '2026-07-19-bank-semantic-dedup',
    date: '2026-07-19',
    title: '✂ Catch the same shot in a dozen crops',
    blurb:
      "The Image bank already grouped exact and resized copies with a perceptual hash. Now a second pass catches what that misses: the same photo re-cropped, re-compressed or lightly re-touched — the “same shot, different crop” that fills a Telegram export. After you run ✨ Score, hit ✂ Find crops & variants (it reuses Score's embeddings, so it costs no extra GPU time) and the near-duplicate variants group up under their own ✂ chip, with the same keep-best / keep-first / pick-one resolution you already know — losers are rejected, never deleted. It also rides along in 🚀 Launch all, right after Score. Tune how close counts as a match in Settings ▸ Captioning & quality; re-running re-sorts instantly from the cached embeddings.",
    to: '/bank',
  },
  {
    id: '2026-07-19-runs-lineage-tree',
    date: '2026-07-19',
    title: '🌳 See how your runs descend from each other — down to every checkpoint',
    blurb:
      "When you continue a training — from its last checkpoint or an earlier, less-cooked epoch — a lineage is born: the original run, its continuation, the re-continuation, and any branch you forked off. The Runs page draws it, two ways: a compact ☰ List and a ◉ Graph — a left-to-right family tree with flowing connectors, the path to the run you're looking at lit up, and forks branching off. Now the graph also shows each run's checkpoints as sober pills beneath it — one run can hold a dozen epochs, all worth a look — and a continuation's connector starts from the exact checkpoint it resumed, so you can see at a glance that “this run began from THAT save”. Click any checkpoint for its actions: ⬇ download it, or ▶ continue from here (the resume dialog opens already set to that step). The graph now opens for a single run too, the moment it has one saved checkpoint — and you can open it straight from a dataset's Checkpoints & LoRAs panel with the new ◉ Graph button. Either view still shows family, steps, dataset version and whether a LoRA is on disk, highlights the current run, and greys a branch resumed from an earlier step (its later saves were set aside, never deleted). Older continuations are reconnected automatically — chains you trained before this shipped now show as one lineage instead of scattered roots, and anything too ambiguous to be sure of is left as a root, never invented.",
    to: '/cloud',
  },
  {
    id: '2026-07-19-training-recipe-tuning',
    date: '2026-07-19',
    title: '🎓 Sharper training recipes from verified community research',
    blurb:
      "Two training defaults were re-tuned from a fact-checked sweep of recent community results. A FLUX.2 Klein STYLE LoRA now trains the winning 128/64/64/32 network (a linear + Conv2d LoRA) that a 64-run sweep and Black Forest Labs' own example converge on — noticeably better at capturing a look. And Slider LoRAs now default to alpha 4 (scale 0.5), matching the Ostris slider notebook (\"bigger is not always better, especially for sliders\") for a cleaner ± sweep. Both are just smarter defaults: your other Klein LoRAs are unchanged, existing runs aren't touched, and Advanced options still lets you set the network alpha back to 8 if you're reproducing an older slider.",
    to: '/datasets?section=training&panel=advanced',
  },
  {
    id: '2026-07-19-bank-launch-all',
    date: '2026-07-19',
    title: '🚀 Launch all — clean a whole bank while you sleep',
    blurb:
      "One button now runs the entire Image bank triage end to end: quality scan → auto-reject the flagged and duplicate shots → ✨ score → 🚩 find watermarks → 👥 group by person → (optionally) 🏷️ caption. Hit “🚀 Launch all”, tick which passes run and how auto-reject behaves, and walk away — a pass whose tool isn't installed (or a busy GPU) is simply skipped with a reason instead of failing the run, and the heavy passes only touch the survivors, never the images you just rejected. You can Stop it any time, and when you come back a saved report tells you exactly what ran, what was skipped and why, with the headline counts.",
    to: '/bank',
  },
  {
    id: '2026-07-19-bank-face-pass-gpu',
    date: '2026-07-19',
    title: 'The Image bank face pass can run on your GPU',
    blurb:
      "The bank's subject (face) pass now uses your GPU automatically when it can — much faster on a big bank — and quietly falls back to CPU when it can't, so nothing breaks. It only takes the GPU when nothing else is using it, never competing with a training run. (GPU needs onnxruntime-gpu in the face-scoring interpreter; without it the pass keeps running on CPU exactly as before.) The “No face” filter is also sharper now: it shows only photos where no face was found — pictures with a small, low-confidence or side-profile face no longer slip into that list.",
    to: '/bank',
  },
  {
    id: '2026-07-19-bank-captions-search',
    date: '2026-07-19',
    title: '🗃️ Caption images inside the Bank and search a big dump by what’s in it',
    blurb:
      "The Image bank can now caption its images with the same engines your datasets use (JoyCaption / Ollama vision, your Settings). Hit “🏷️ Caption” to describe every not-yet-captioned image, or select some first to caption just those — it runs in the background, is Stop-able mid-run, and never races your GPU. The captions then power a new 🔍 search bar: type “red dress” and the grid filters to matching images (it matches file names too), combinable with every existing filter — the fast way to find shots in a 9,000-image Telegram export. Best of all, captions follow the images: promote a captioned selection and the dataset starts already captioned for them.",
    to: '/bank',
  },
  {
    id: '2026-07-19-folder-browse-button',
    date: '2026-07-19',
    title: 'Browse for a folder instead of typing its path',
    blurb:
      "Pointing the Image bank (or a dataset folder-import) at a folder no longer means typing a path by hand. Hit “📂 Browse…” and the app opens your computer's own folder dialog — pick the folder and the field fills itself in. On a phone or a remote/Linux server where that native dialog can't show, a built-in folder browser opens instead. Pasting a path still works too.",
    to: '/bank',
  },
  {
    id: '2026-07-19-bank-scoring-passes',
    date: '2026-07-19',
    title: '🗃️ Image bank now rates looks, flags NSFW, groups by style and finds watermarks',
    blurb:
      "The Bank gains three new triage passes for a big mixed dump. “✨ Score” rates every image for aesthetics (1–10) with the LAION predictor, flags NSFW, and groups shots by visual STYLE (screenshots and memes cluster apart from photoreal) — and “keep best” on a duplicate group now keeps the nicest-looking copy, not just the biggest. “🚩 Find watermarks” reuses the same Qwen3-VL detector the datasets use to flag overlaid logos/URLs (detection only — your files are never touched). New filter chips, style groups and a per-subfolder scope let you slice a Telegram export by chat; every threshold lives in Settings → Captioning & quality and re-sorts the bank with no rescan. The scoring model installs on demand from Setup ▸ Quality tools; without it the button explains what to install rather than failing silently.",
    to: '/bank',
  },
  {
    id: '2026-07-19-stop-captioning-batch',
    date: '2026-07-19',
    title: 'Stop a captioning batch mid-run',
    blurb:
      "Launched a big caption pass and realized it's captioning badly, or you mis-set an option? A ⏹ Stop button now sits in the captioning progress banner. It finishes the image currently being written — never cuts an inference off mid-way — then stops cleanly: everything captioned so far is kept, the rest is left untouched, and the GPU is freed exactly as on a normal finish. You get an honest \"stopped — X captioned\" summary. No more waiting out a 100-image run you already know is wrong.",
    to: '/datasets?section=captions',
  },
  {
    id: '2026-07-19-caption-method-options',
    date: '2026-07-19',
    title: 'Choose your caption engine, model and instructions — per dataset',
    blurb:
      "The Captions area has a new ⚙️ Options button. Pick which engine writes this dataset's captions (Auto, JoyCaption, or Ollama vision), choose which pulled Ollama vision model runs — or pull a new one by name right there, with a live progress readout. A Vocabulary preset sets how the model names nude or sexual content — Explicit (crude, uncensored — pair it with an abliterated vision model), Clinical, or Safe — and you can still add your own extra instructions to steer the wording (e.g. “always name the visible clothing colors”). Presets and instructions ride on top of the built-in prompt, so the identity / concept / style guardrails and the leak cleaners still apply — they change wording, never what binds to the trigger. Everything is remembered on the dataset and used by the next caption or re-caption run; leave any field on “default” to keep following Settings.",
    to: '/datasets?section=captions',
  },
  {
    id: '2026-07-19-setup-install-everything',
    date: '2026-07-19',
    title: 'Setup: an install step with one-click Install everything — and reinstall per item',
    blurb:
      "After you've configured your services, Setup has a dedicated install step. One Install everything button queues every component the app can install for you — the ML extras (face scoring, person masks, watermark inpainting), the Ollama vision model when Ollama is running, and the Klein weights when a valid ComfyUI is set — with a live “X / N” progress bar. Heavy installs still run one at a time so they never clash, and the big model downloads run in parallel. Below it, a menu lets you install each component on its own — and it stays there even once everything is in, with a ↻ Reinstall button per item to repair a broken install (a corrupted environment) without redoing the rest.",
    to: '/setup',
  },
  {
    id: '2026-07-19-zip-install-in-app-update',
    date: '2026-07-19',
    title: '“Update & restart” now works even if you installed from a ZIP',
    blurb:
      "If you downloaded the app as a ZIP from the releases page (no Git), the “Update & restart” button used to only send you off to download the new version by hand. Now it does it for you, and the button adapts to how you installed: on a ZIP install it names the release and its size (“Update to v2026.07.19 — download ~42 MB”) and shows a live progress bar while it downloads and installs, since that takes longer than a git pull. It backs up your current files, swaps in the new ones — keeping your datasets, settings, .env and Python environment fully intact — then restarts. If anything goes wrong mid-way it rolls back automatically, so a failed update never leaves you with a broken install. Git clones keep updating exactly as before.",
    to: '/settings/maintenance',
  },
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
    title: 'Back up your whole library — datasets, training history and settings — in one click',
    blurb:
      "A new “💾 Back up everything” button on the Datasets library packs every dataset (images, captions, statuses, references), its training history, plus your settings into a single file, so you can move to a new machine or recover from one without losing anything. It runs in the background with a live progress bar — a big library can be gigabytes — then hands you a download and an “Open folder”. Your API keys and tokens are deliberately left out, so the file is safe to keep around; re-enter them once on the new install. Restoring is the same “📦 Import backup” button: it now accepts the master archive too, rebuilds every dataset without ever overwriting one (name clashes get a “(restored)” suffix), and — new — brings back each dataset’s training runs so it lands under “Trained” again instead of “Not trained yet”, with its history in the Runs hub. Tick “Include trained LoRAs” before backing up to bundle the trained .safetensors themselves (a much larger file); leave it off and the light training history still restores your “Trained” status. You always get an honest report of exactly what came back and what was skipped.",
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
    id: '2026-07-18-image-bank-triage',
    date: '2026-07-18',
    title: 'New (Beta): 🗃️ Image bank — turn a 9 000-image dump into a dataset',
    blurb:
      "Exported thousands of unsorted images from Telegram (or anywhere)? Point the new Bank tab at the folder: a background quality scan flags the blurry, noisy, flat and too-small shots and groups near-duplicates (resolve a whole bank with one “keep best” click); the face pass then sorts everything by PERSON — no reference photo needed. Keep the good ones and promote them straight into a dataset. Your folder is never modified, rejections are just reversible statuses, and the thresholds are tunable in Settings → Captioning & quality without rescanning.",
    to: '/bank',
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
  '/datasets', '/bank', '/studio', '/cloud', '/guide', '/help', '/setup',
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
