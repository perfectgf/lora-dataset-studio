// =====================================================================
//  🎁 What's new — in-app changelog feed (source of truth)
// =====================================================================
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
//  • `image` is OPTIONAL: a repo-relative path to a screenshot
//    ('docs/screenshots/canvas/board.png'). It is shown in the GITHUB RELEASE
//    body only — the in-app panel does not render it — under the prose, pinned
//    to the released tag so an old release keeps showing what actually shipped.
//    Three rules, and they are not style preferences:
//      1. ONE per release, on the headline change. A wall of screenshots reads
//         as a brochure; one picture reads as evidence.
//      2. NEVER from a real bank or dataset. The maintainer's own images are
//         NSFW and are out of bounds for anything public, cropped or not — use
//         the showcase instance (scripts/seed_showcase.py) whose data is
//         generated.
//      3. COMMIT THE PICTURE BEFORE YOU TAG. The URL is pinned to the tag,
//         so the file has to exist inside it — a screenshot added after the
//         release is a dead link, and a published release cannot be fixed
//         by attaching one. Order: shoot, commit, tag, release. (Measured:
//         v2026.08.22 and v2026.08.18 both carry no docs/screenshots/release/
//         folder at all, so neither could be retrofitted.)
//      4. Prefer a screen that shows the CHANGE, not the app. A settings panel,
//         a toolbar, a queue — those photograph without any dataset image at
//         all, which is why most entries can carry one cheaply.
//      5. WIRE THE `image:` FIELD BEFORE THE TAG, on an entry that is NEW in
//         that tag. The notes select new ids only, so a picture wired onto an
//         already-shipped entry never appears anywhere. This is not
//         hypothetical: v2026.08.23 went out imageless with the mechanism
//         landed AND the PNG committed — nobody had written the one field
//         line. release-notes-contract.test.mjs now fails on an orphan
//         screenshot, so the miss is loud instead of silent.
//  • Keep the list tidy: entries whose release has LONG shipped move to
//    frontend/src/whatsNewArchive.js — their own lazy chunk, loaded by the
//    panel's "Show older updates". NEVER archive an entry that has not been
//    in a tagged release yet: release notes are built from THIS file's id
//    diff, so archiving an unshipped entry silently costs it its notes.
//    `id`/`date` move unchanged (the seen-marker keys on ids); `to:` is
//    dropped on the way (months-old targets go stale, and no `to` keeps the
//    archive import-free). whatsNewArchive.test.js holds the pairing.
// =====================================================================
import { SETTINGS_SECTIONS } from './components/settings/registry.js';
import { WORKSPACE_SECTIONS } from './components/dataset/workspaceSections.js';
import { SETUP_DEEP_LINK_STEPS } from './hooks/useSetupSteps.js';

// Newest first. Prepend new waves at the top.
export const WHATS_NEW = [
  {
    id: '2026-09-02-comparison-prompt-batch',
    date: '2026-09-02',
    title: 'The multi-LoRA comparison can replay a batch of prompts too',
    blurb:
      'Ticking several prompts to replay them in one run worked in the Test '
      + 'Studio and on the canvas, and did nothing at all on the comparison '
      + 'screen — its launch simply never carried them. It does now, so saved '
      + 'prompts, 🎬 scenes and 🌐 Civitai picks all build a batch there as '
      + 'well: one image set per prompt, across every LoRA you are comparing, '
      + 'same seed and settings. The cost counter multiplies by the batch '
      + 'before you launch instead of surprising you afterwards.',
    to: '/studio',
  },
  {
    id: '2026-09-02-free-memory-button',
    date: '2026-09-02',
    title: 'A 🧹 button beside the machine-load numbers gives the RAM back',
    blurb:
      'ComfyUI keeps every model of the day cached in RAM after it leaves the '
      + 'card (measured: 34 GB on an idle ComfyUI), and the vision model stays '
      + 'warm for captioning — neither returns it by itself. 🧹 next to the '
      + 'CPU · GPU · VRAM · RAM readout (top bar and Canvas toolbar) unloads '
      + 'both and re-reads the machine; the toast says what actually came back. '
      + 'Refused while something is rendering or training, and a model another '
      + 'tool loaded is never touched.',
    to: '/canvas',
  },
  {
    id: '2026-09-02-civitai-prompts-in-the-batch',
    date: '2026-09-02',
    title: 'Tick several Civitai prompts straight into the batch',
    blurb:
      'In the 🌐 Civitai browser every prompt-bearing card now has a ☐ Batch '
      + 'box: tick as many as you like without leaving the browser, and the '
      + 'next Run test replays them all — one pass per prompt, same checkpoints, '
      + 'same settings, same seed, alongside the saved prompts you ticked. The '
      + 'count shows under the prompt field and on the 🌐 button; ⤵ Use prompt '
      + 'still drops a single one into the field. On the Test Studio and the '
      + 'board’s 🎨 Generate alike.',
    to: '/studio',
  },
  {
    id: '2026-09-02-video-checkpoints-and-loras',
    date: '2026-09-02',
    title: 'A video set gets its Checkpoints & LoRAs section — deploy, continue, clear, step by step',
    blurb:
      'Every save a video training brought back — from this PC or from a rented '
      + 'pod — now has its own section in the video workspace, listed by step so '
      + 'both experts of a Wan 2.2 pair travel together. Each step offers what an '
      + 'image dataset’s does: ⬇ download, 📦 deploy into ComfyUI’s loras folder '
      + '(the Video Test Studio lists it as deployed right away), ⏏ undeploy, '
      + '▶ continue training from that exact step on a fresh pod, ⓘ the run’s '
      + 'details, and 🗑 delete — to the app’s Trash, recoverable. A Studio section '
      + 'opens the Video tab of the Test Studio next door.',
    to: '/datasets',
  },
  {
    id: '2026-09-02-publish-to-civitai',
    date: '2026-09-02',
    title: 'Publish a checkpoint and its images to Civitai without leaving the app',
    blurb:
      'A checkpoint pill now has a 📤 Civitai row: create its model page from '
      + 'here (name, trigger words, base model and description pre-filled, the '
      + '.safetensors uploaded, as a draft to finish on Civitai or published at '
      + 'once), or mark the page it already has by pasting its address. Once a '
      + 'checkpoint is linked, the image viewer’s 📤 button posts a picture '
      + 'under that page in one press, with its prompt, seed, sampler and LoRA '
      + 'weight as Civitai generation data. Every image leaves as a fresh PNG '
      + 'with no embedded metadata, and the checkpoint’s own metadata is '
      + 'checked for a machine path before anything is sent. Same key as the '
      + 'scraper and the 🌐 prompt browser.',
    to: '/gallery',
  },
  {
    id: '2026-09-02-video-cloud-launch-window',
    date: '2026-09-02',
    title: 'See the price, the time and the total before renting a GPU for a video run',
    blurb:
      'A video set used to rent a pod on one click, with the cost hidden inside a '
      + 'closed dropdown. ☁ Train in the cloud now opens the same window as an image '
      + 'dataset: every GPU class with its price per hour, a rough duration and total '
      + 'for this set, a warning when the run would outlive the runtime cap, and this '
      + 'month’s spend against your budget. A readiness card above the launch says what '
      + 'still stands in the way — no clips, no references, a missing key — before '
      + 'anything is spent, and “second pod, billed separately?” is a question you can '
      + 'answer instead of an error.',
    to: '/datasets',
  },
  {
    id: '2026-09-02-fence-one-click-one-answer',
    date: '2026-09-02',
    title: 'A click that waited for a busy local model runs once — and the notice names your server',
    blurb:
      'When another tool held your local model, LDS waited and replayed the click '
      + 'the moment it was free — and kept that replay armed after you had already '
      + 'clicked again, so ✨ Enhance could write two answers into the field, or '
      + 'report one failure twice. One click is one answer now, on every surface '
      + 'the fence guards — and an answer that arrives after you moved on (a new '
      + 'frame, another mode or length, a newer click) is set aside, never '
      + 'written into the field you are now looking at; the ✨ writers say so '
      + 'with a note. And the notice names the server you actually run: on LM '
      + 'Studio it no longer sends you to look in Ollama.',
  },
  {
    id: '2026-09-02-krea-rebalance-and-enhancer-retired',
    date: '2026-09-02',
    title: 'Krea grids start on a bare ComfyUI — the rebalance and the enhancer are gone',
    blurb:
      'The Krea Studio graph now uses core ComfyUI nodes only: nothing to install, '
      + 'and no more "custom node missing" at launch on a fresh setup. The NSFW / '
      + 'texture rebalance toggle is retired — measured at a fixed seed, x4 did not '
      + 'refine skin, it re-decided the whole picture (94% of pixels moved) — and so '
      + 'is the experimental Krea2T Enhancer, which nobody used. Cells rendered with '
      + 'either keep their record in the database; a resume renders them without.',
    to: '/studio',
  },
  {
    id: '2026-09-02-krea-hires-fix-and-finishing',
    date: '2026-09-02',
    title: 'A second pass for Krea, and a finishing touch after Upscale & improve',
    blurb:
      'Krea can now sample small and re-sample an upscaled latent — the model draws '
      + 'the detail instead of interpolating it. Set the default in Settings ▸ Image '
      + 'engines ▸ Krea 2, or pick it per run from the Studio\'s Sampling section. '
      + 'And ✨ Upscale & improve gets a finishing pass the app runs itself: put the '
      + 'source\'s colours back after a Klein pass, sharpen the finest detail, add a '
      + 'touch of film grain — nothing to install, all off until you turn them on. '
      + 'Sharpen and grain are also per run, in the Studio\'s Engine section.',
    to: '/settings/engines',
  },
  {
    id: '2026-09-01-video-dataset-workspace',
    date: '2026-09-01',
    title: 'Your video training sets finally have a screen of their own',
    blurb:
      'A video set used to be a card at the bottom of the library: a list of file '
      + 'names, one caption box each, and nothing else. Opening one now opens a full '
      + 'workspace — a grid of every clip with the rush and timecode it came from, a '
      + 'player you can step through with the arrow keys, a search, and caption tools '
      + 'that rewrite the .txt files in bulk. Same shape as an image dataset, because '
      + 'it is the same job.',
    to: '/datasets',
  },
  {
    id: '2026-09-01-remove-a-clip-from-a-video-set',
    date: '2026-09-01',
    title: 'Drop a bad clip without re-cutting the whole set',
    blurb:
      'The three-frame clip of somebody’s hand is only visible after the encode, and '
      + 'until now the only way out was deleting the dataset and promoting it again. '
      + 'Remove it from the set instead: its .mp4 and .txt go, and the bank keeps the '
      + 'shot, its bounds and every decision — so you can re-cut and promote it again '
      + 'with no triage to redo.',
    to: '/datasets',
  },
  {
    id: '2026-09-02-sliders-locked-against-mistaps',
    date: '2026-09-02',
    title: 'Dials no longer move when you scroll past them on a phone',
    blurb:
      'A slider claims the touch that merely crosses it, so scrolling the '
      + 'render rail with a thumb dragged whichever dial was under it — '
      + 'silently, and the next clip rendered on a length nobody chose. Every '
      + 'slider in the app now hands vertical swipes back to the page, and the '
      + 'Video Test Studio’s dials (steps, length, resolution, LoRA '
      + 'strength) carry the padlock the image side already had: locked by '
      + 'default, one tap to open, and each remembers whether you left it open.',
    to: '/studio',
  },
  {
    id: '2026-09-02-start-frame-clip-tab-fixed',
    date: '2026-09-02',
    title: 'Picking a start frame from a training clip no longer takes the page down',
    blurb:
      'In the Video Test Studio, opening the “Dataset clip” tab and choosing a '
      + 'training set blanked the screen: the clip list was read from the wrong '
      + 'field and a count arrived where the clips were meant to be. It lists '
      + 'them properly now, and says so plainly when a set holds none.',
    to: '/datasets',
  },
  {
    id: '2026-09-01-motion-auto-instructed',
    date: '2026-09-01',
    title: '✨ Auto obeys what you type, and you choose the model behind it',
    blurb:
      'Type what should happen — "make her jump twice", "slower" — and ✨ Auto '
      + 'follows it while keeping the people your start frame actually shows; '
      + 'leave the field empty and it proposes freely. ✨ Enrich picks the same '
      + 'two modes by itself. The ⚙ beside them opens the model window: the '
      + 'motion writer is its own setting, so tuning it never re-points your '
      + 'image passes.',
  },
  {
    id: '2026-09-01-motion-auto-and-enrich',
    date: '2026-09-01',
    title: 'The Motion field writes itself, in H3’s own words and paced to your clip',
    blurb:
      '✨ Auto reads your start frame and proposes the movement; ✨ Enrich '
      + 'rewrites what you wrote with more of the detail a sampler can use — '
      + 'both leave the text yours to edit. And a toggle enriches at launch, '
      + 'recording on the clip the prompt that actually ran. All three write '
      + 'the official three-field prompt the model was trained on, paced to '
      + 'the length you set (a 1 s clip and a 15 s one are no longer given the '
      + 'same beat), and — when there is a start frame — name it so the subject '
      + 'stays who it is; a prompt you typed yourself gets that first-frame line '
      + 'at launch, once. They go through the local model you already run for '
      + 'the image passes, or another one if you pick it under ⚙.',
  },
  {
    id: '2026-09-01-bank-improve-carries-the-dials',
    date: '2026-09-01',
    title: 'The bank’s ✨ improve shows the dials it obeys, instead of naming them',
    blurb:
      'Improving a whole bank ran on the same instruction, LoRA preset, '
      + 'strengths and output size as a dataset improve — and its launch window '
      + 'listed them as things to go and change in Settings. They are in the '
      + 'window now, exactly as in the dataset one, whenever Klein is the engine.',
    to: '/bank',
  },
  {
    id: '2026-09-01-saved-prompts-browser',
    date: '2026-09-01',
    title: 'Your saved prompts, big enough to recognise and searchable',
    blurb:
      'The list of prompts you have launched a test with was a wall of 32-pixel '
      + 'thumbnails showing the first thirty characters — and test prompts run '
      + 'to hundreds of characters that all start the same way, so most cards '
      + 'read alike and the picture that told them apart was too small to see. '
      + 'The strip now keeps the last few at a size you can actually read, and '
      + '📚 Browse all opens the whole history: search it by any words you '
      + 'remember, read each prompt in full, tick them for a batch, delete the '
      + 'ones you are done with. Same panel on the dataset Test Studio and on '
      + '“Generate from the board”.',
    to: '/studio',
  },
  {
    id: '2026-09-01-improve-panel-lora-strengths',
    date: '2026-09-01',
    title: 'Tune a preset’s LoRAs from the picture they apply to',
    blurb:
      'The ✨ Upscale & improve window named which LoRA preset it chains and '
      + 'then said nothing about what was in it, so the one number you actually '
      + 'change — how hard a LoRA pulls — still meant a trip to Settings. The '
      + 'window now lists the preset’s LoRAs with a slider each, saved as you '
      + 'drag. Building the presets themselves (adding, removing, reordering) '
      + 'stays in Settings ▸ Engines: those change what a preset IS, for every '
      + 'surface that runs Klein.',
    to: '/gallery',
  },
  {
    id: '2026-09-01-improve-result-zoom',
    date: '2026-09-01',
    title: 'Zoom into the improved picture without leaving the window',
    blurb:
      'An upscale is judged on detail that fit-to-window hides. The result now '
      + 'takes the wheel, a pinch on a touchscreen and a double-tap to fit '
      + 'again — the same gestures the image viewer has always had, and never '
      + 'past the picture’s own pixels.',
    to: '/gallery',
  },
  {
    id: '2026-09-01-gallery-refreshes-itself',
    date: '2026-09-01',
    title: 'The Gallery shows a new render without a page reload',
    blurb:
      'Generate or improve something with the Gallery open and the image only '
      + 'appeared after refreshing the page by hand. The feed now watches the '
      + 'shared generation queue and slips whatever finished in at the top — '
      + 'keeping your scroll, your selection and an open image exactly where '
      + 'they were.',
    to: '/gallery',
  },
  {
    id: '2026-09-01-video-studio-smooth-vfi',
    date: '2026-09-01',
    title: 'Smooth a test clip to twice its frame rate',
    blurb:
      'Every finished clip gains a ↗ Smooth button: RIFE frame interpolation, '
      + 'the same recipe the image generator runs (rife49, ×2, ensemble), so a '
      + 'clip smoothed here is the clip smoothed there. It makes a NEW clip at '
      + 'double the rate and the same duration — the original stays, because '
      + 'comparing the two is the point.',
  },
  {
    id: '2026-09-01-reuse-brings-the-start-frame-back',
    date: '2026-09-01',
    title: '↻ Reuse gives the start frame back, and any LoRA can be imported',
    blurb:
      'Reusing an image-to-video clip restored every dial — model, steps, '
      + 'length, seed — and left the start frame empty, so Generate stayed '
      + 'blocked. It comes back now. And the LoRA picker gained an import: give '
      + 'it a path on this machine or choose the file, and it lands in '
      + 'ComfyUI’s folder ready to test — no more moving files by hand.',
  },
  {
    id: '2026-09-01-video-studio-length-to-15s',
    date: '2026-09-01',
    title: 'Clips up to 15 seconds, on a slider instead of a 21-row list',
    blurb:
      'The length list stopped at 209 frames (8.7s) because it was reading the '
      + 'TRAINING catalogue — the model renders to 15s and the server always '
      + 'accepted it. Every legal length from 0.88s to 15.04s is now on one '
      + 'slider that snaps to what the VAE accepts, with the seconds and the '
      + 'frame count above it and both ends of the range in view.',
  },
  {
    id: '2026-09-01-video-studio-steps-dial',
    date: '2026-09-01',
    title: 'The sampling steps are a dial now, not a decision made for you',
    blurb:
      'The Video Test Studio ran 6 steps with Turbo and 20 without, and nothing '
      + 'on screen let you move that — the one number that plainly trades time '
      + 'for fidelity. There is now a Sampling steps slider (4 to 40) that says '
      + 'what auto resolves to, and an explicit count wins over Turbo’s own. '
      + '↻ Reuse replays the count a clip really ran.',
  },
  {
    id: '2026-09-01-canvas-lanes-move-and-resize',
    date: '2026-09-01',
    title: 'Move a dataset’s block on the Canvas, and give it the room it needs',
    blurb:
      'Pin a run’s images and the contact sheet hangs below the tree — but the '
      + 'board never counted it, so it landed on top of the next dataset. Each '
      + 'lane now has its own two grips: drag its title strip to move the whole '
      + 'block, drag its bottom edge to set how much room it keeps, and the '
      + 'datasets below move with it. The edge turns amber when a lane draws '
      + 'past its own room — double-click it to fit. ✦ Tidy up still hands '
      + 'everything back to the automatic layout.',
    to: '/canvas',
  },
  {
    id: '2026-09-01-start-frame-from-the-gallery',
    date: '2026-09-01',
    title: 'Animate an image straight from the Gallery',
    blurb:
      'The Video Test Studio’s start frame took an upload, a Bank image or '
      + 'a dataset clip — but not the picture this app had just generated, which '
      + 'meant exporting it to disk to feed it back in. The Gallery is now a '
      + 'fourth source: pick any generated image and it is staged at full size.',
    to: '/datasets',
  },
  {
    id: '2026-09-01-video-prep-in-one-button',
    date: '2026-09-01',
    title: '▶ Run everything now really runs everything',
    blurb:
      'The chain stopped after thumbnails while its own tooltip promised more. '
      + 'It now offers every preparation pass — measure, embeddings, duplicates '
      + 'and camera — in a launch window where you tick what you want, in the '
      + 'order each one needs the previous. Describe shots stays its own button: '
      + 'its wording changes what the captions say.',
  },
  {
    id: '2026-09-01-slice-long-shots',
    date: '2026-09-01',
    title: 'Long shots can give several training clips instead of one',
    blurb:
      'A 15-second shot built at 209 frames used to train on its first 8.7 '
      + 'seconds and the rest was never used. Tick “Slice shots longer than one '
      + 'clip” when building a set and it gives whole clips end to end instead — '
      + 'up to 8 per shot. Each slice carries its shot’s caption, so the window '
      + 'says it plainly.',
  },
  {
    id: '2026-09-01-max-shot-length-cut',
    date: '2026-09-01',
    title: 'A maximum shot length, next to the minimum',
    blurb:
      'The Quality cuts had a floor and no ceiling, so the shots your target '
      + 'will truncate were invisible. Set a maximum and they are flagged '
      + '“Longer than a clip” — filter them, cut them by hand, or slice them at '
      + 'build time. It flags and sorts; it never rejects anything.',
  },
  {
    id: '2026-09-01-clip-length-suggestion',
    date: '2026-09-01',
    title: 'The clip-length picker tells you what each length costs',
    blurb:
      'Building a set now says how long your kept shots actually run and how '
      + 'many of them each length keeps whole — “141 frames keeps 87% of them”. '
      + 'It never changes your choice, it just stops the default from being a '
      + 'guess.',
  },
  {
    id: '2026-09-01-recut-keeps-what-did-not-move',
    date: '2026-09-01',
    title: 'Changing the shot threshold no longer throws away your triage',
    blurb:
      'A re-cut used to replace every shot of a file — decisions, captions and '
      + 'measurements with them — so trying a different threshold cost an '
      + 'afternoon of work. Now a shot whose bounds do not change keeps its row: '
      + 'its Keep/Reject, its caption and its scores stay. Only genuinely new or '
      + 'merged shots start clean, and the result line says how many were kept.',
  },
  {
    id: '2026-09-01-video-bank-wears-the-bank-shell',
    date: '2026-09-01',
    title: 'The video bank now looks and works like the image bank',
    blurb:
      'Same shell, same gestures: filters live in a rail beside the shot grid '
      + '(a drawer on a phone), the analysis passes open on demand from the ⚙ '
      + 'button, the two decisive actions — ▶ run the pipeline and 🎬 build a '
      + 'training set — sit in the top bar next to the same stats strip, and '
      + 'every chip and button is the one the image bank already taught you. '
      + 'Nothing moved in what the passes do — only where you reach them.',
  },
  {
    id: '2026-09-01-video-captions-follow-what-you-installed',
    date: '2026-09-01',
    title: 'Video captions now run on what your machine has — Ollama and LM Studio included',
    blurb:
      'No torch Python? If Ollama or LM Studio is running, 🗣 Describe shots '
      + 'captions through it — the same local server and vision model your image '
      + 'passes already use — instead of showing a dead ✗. LDS\u2019s own '
      + 'transformers worker stays the default when available (it feeds the '
      + 'model real frame timestamps and measures captions in the encoder\u2019s '
      + 'own tokens), the launch window says which engine will run, and every '
      + 'caption records which engine wrote it.',
  },
  {
    id: '2026-09-01-captions-never-swap-the-scene',
    date: '2026-09-01',
    title: 'NSFW captions can no longer be quietly swapped for an invented scene',
    blurb:
      'A measured failure, not a theory: asked politely, caption models do not '
      + 'soften explicit footage — they replace it with a harmless invented one. '
      + 'The Plain wording now forbids sanitizing, softening or replacing the '
      + 'scene outright, and the standard wording gains the neutral half: '
      + 'describe the scene that is shown, never a substitute for it.',
  },
  {
    id: '2026-09-01-video-captions-fit-the-encoder',
    date: '2026-09-01',
    title: 'Video captions that fit the model — measured in its own tokens',
    blurb:
      '🗣 Describe shots now ends each caption with a short structured tail '
      + '(Subject, Motion, Setting, Style) and, when umT5\'s tokenizer is on your '
      + 'machine, counts the caption in the Wan encoder\'s own tokens instead of '
      + 'guessing from words. Building a training set uses both: a prompt that '
      + 'would overrun the encoder window (512 tokens on Wan, which cuts in '
      + 'silence) is written in its short form instead of being truncated '
      + 'mid-sentence, and the export tells you how many.',
  },
  {
    id: '2026-09-01-video-captions-work-under-transformers-5',
    date: '2026-09-01',
    title: 'Describe shots works again on machines whose Python carries transformers 5',
    blurb:
      'Every shot of a caption pass was failing there — with no reason shown '
      + 'anywhere. The pass now runs on transformers 4 and 5 alike, and when a '
      + 'shot is refused the reason lands in the log instead of vanishing.',
  },
  {
    id: '2026-08-31-watermark-zones-whole-mark',
    date: '2026-08-31',
    title: 'Watermark zones that cover the whole mark — thumbnails included',
    blurb:
      'A logo is usually an emblem above a line of text, and the detector was '
      + 'boxing only the text: the clean erased the words and re-rendered the '
      + 'emblem as a ghost. Zones now reach the whole mark, so a clean has '
      + 'nothing left to put back. And small stock thumbnails — the 474px '
      + 'previews with the brand stamped across them — no longer come back '
      + '“watermarked, position unknown”: the word gets a zone you can crop, '
      + 'mask or clean like any other.',
    to: '/datasets',
  },
  {
    id: '2026-08-31-klein-clean-prompt-and-size',
    date: '2026-08-31',
    title: 'See — and change — what the Klein watermark clean actually does',
    blurb:
      'Cleaning a watermark with Klein had one option: which model. The prompt it '
      + 'sends was a constant in the code, so a mark that survived left you nothing '
      + 'to turn. Pick Klein on the Bank panel or the dataset Clean bar and you now '
      + 'see the exact instruction being sent — “remove watermark” — in an editable '
      + 'box with a Reset to default beside it, plus the processing size (1 to 4 MP, '
      + 'default 2: higher regenerates finer detail and costs more VRAM and time, and '
      + 'a photo already smaller is never enlarged) and whether the cleaned file keeps '
      + 'your original dimensions or is written at the render size, which changes the '
      + 'file dimensions. One stored choice, so setting it on either surface arms both '
      + '— and every clean now logs the prompt it used, so you can tell afterwards '
      + 'what ran.',
    to: '/datasets',
  },
  {
    id: '2026-08-31-video-test-studio',
    date: '2026-08-31',
    title: 'Play your video LoRA back, without leaving the app (beta)',
    blurb:
      'Training a video LoRA gave you a file and a loss curve, and judging it '
      + 'meant wiring a graph in ComfyUI by hand. The Test Studio now has a '
      + 'Video tab: pick a LoRA you trained (it is copied into ComfyUI for you '
      + 'the first time), give it a start frame — uploaded, from a bank, or the '
      + 'first frame of a clip in a training set — or none at all for '
      + 'text-to-video, describe the motion, and get a clip. ⚡ Turbo renders in '
      + 'minutes instead of tens of minutes, sparse attention and the latent '
      + 'upscale trade a little fidelity for speed, and every clip keeps the '
      + 'settings that made it so Reuse can rerun the same seed with one dial '
      + 'moved. New here? Setup ▸ 🎬 Video Test Studio downloads the engine '
      + '(about 39.5 GB); the clip itself needs no ComfyUI add-on at all, and '
      + 'the three optional accelerators are named and linked for you to '
      + 'install on the ComfyUI side. Marked beta while the first clips come '
      + 'back from real machines.',
    to: '/studio?lane=video',
  },
  {
    id: '2026-08-31-vision-model-in-the-scan-window',
    date: '2026-08-31',
    title: 'Watermark scans on the vision route: pick — or pull — the model right there',
    blurb:
      'When Find watermarks runs on your local LLM, the scan window now names the '
      + 'exact model that will judge your images, lists the ones installed in Ollama '
      + 'or LM Studio to switch in one click, and pulls a new one without leaving the '
      + 'window — a finished pull is selected for the next scan. Stored, so the bank, '
      + 'the dataset and Settings ▸ Local tools all read the same choice.',
    to: '/datasets',
  },
  {
    id: '2026-08-31-klein-cleans-the-whole-photo',
    date: '2026-08-31',
    title: 'Klein now cleans watermarks it could never reach before',
    blurb:
      'Pick Klein on 🧽 Clean and it now erases the zones it found, then hands the '
      + 'whole photo to the model with one instruction — remove the watermarks — '
      + 'instead of repainting a crop around each box. So it clears the marks the '
      + 'scan missed as well: a stock photo tiled with a logo, the case that used '
      + 'to be hopeless because there was no clean area to copy from, comes back '
      + 'clear, and so does a mark on the subject or one boxed in the wrong place. '
      + 'The trade is that the picture is re-rendered rather than patched, so '
      + 'details shift outside the marks too, and a mark nobody detected can '
      + 'survive — look at the result, and ↩ Restore original brings your file '
      + 'back. LaMa is unchanged, and so is ✦ Repair: a repair you aim at a drawn '
      + 'box still leaves everything outside it untouched.',
    to: '/datasets?section=curation&panel=watermarks',
  },
  {
    id: '2026-08-31-deep-zone-hunt',
    date: '2026-08-31',
    title: 'Watermark zones: the detector now finds the small and repeated marks',
    blurb:
      'The zone hunt sweeps each flagged image at up to three scales (full '
      + 'frame plus tiles), so a logo stamped seven times across a large photo '
      + 'comes back with all seven zones instead of four — and a stock-style '
      + 'tiled watermark now shows the dozen zones it pinned instead of none. '
      + 'Every zone is double-checked before it is kept, so rocks and icicles '
      + 'stop being boxed as logos. Slower per flagged image (a few seconds), '
      + 'unchanged on clean ones.',
    to: '/datasets',
  },
  {
    id: '2026-08-31-not-duplicates',
    date: '2026-08-31',
    title: 'Tell the bank a group is NOT duplicates — once, and for good',
    blurb:
      'A burst, a tripod series, two crops a threshold called one picture: the '
      + 'duplicate panel could only be answered by rejecting a photo you wanted '
      + 'to keep, and Skip wrote nothing so the group came back on every run. '
      + '≠ Not duplicates (N in ⤢ Compare) keeps every copy, rejects nothing, '
      + 'and stops proposing the group. It is remembered as the pairs you ruled '
      + 'on, so it survives the renumbering each pass does — and a group that '
      + 'later gains a new copy asks you again, because that copy is a new '
      + 'question. One line above the list puts them all back.',
    to: '/bank',
  },
  {
    id: '2026-08-30-every-watermark-zone-survives',
    date: '2026-08-30',
    title: 'Multi-logo watermarks: every zone survives the scan',
    blurb:
      'An image stamped with several logos used to come out of Find watermarks '
      + 'with a single box — Clean repainted one logo and left the rest. The '
      + 'detector now keeps every zone it finds (Review shows them all, Clean '
      + 'repaints them all), on datasets and banks alike. Single-mark images '
      + 'behave exactly as before, so border marks stay croppable.',
    to: '/datasets',
  },
  {
    id: '2026-08-30-watermark-scan-honesty',
    date: '2026-08-30',
    title: 'Watermark scans stop hiding their misses',
    blurb:
      'Three fixes from one real test session. A vision scan whose model never '
      + 'answered used to show a green "0 found (of 0)" — it now says plainly '
      + 'that nothing was scanned and names the server to check. Marks tiled '
      + 'across the WHOLE image no longer shrink to one corner box: the image '
      + 'is flagged for 🔍 Review instead, where you can judge it honestly. And '
      + 'when nothing crosses the detector threshold, the toast tells you the '
      + 'highest score it saw — so "lower the threshold" stops being a guess.',
    to: '/datasets',
  },
  {
    id: '2026-08-30-compare-duplicate-copies',
    date: '2026-08-30',
    title: 'See which duplicate you are keeping, before you keep it',
    blurb:
      'Duplicate and "same shot" groups get a ⤢ Compare button that opens their '
      + 'copies full screen — side by side, or one at a time in the same frame '
      + 'so ← → flips between them and the difference lands on the same pixels. '
      + 'Resolution, sharpness, score and weight sit under each copy with the '
      + "group's best value lit, byte-identical copies are marked as such, and "
      + 'K keeps the one you are looking at while R throws out just that one. '
      + 'Keep best and keep first are still one click away — now you can check '
      + 'them first.',
    to: '/bank',
  },
  {
    id: '2026-08-30-delete-video-runs',
    date: '2026-08-30',
    title: 'Delete old video training runs — and a Beta label that says so',
    blurb:
      'Every run a video dataset ever made stayed on its card forever — '
      + 'smoke tests, superseded step counts, all of it. Each checkpoint '
      + 'group now has a 🗑 that removes that run’s LoRA files and its '
      + 'history line (with a confirmation that counts the files; an active '
      + 'run must be stopped first, and the dataset itself is never touched). '
      + 'And the training block now wears a Beta chip — the rail is proven '
      + 'end to end, but it is days old, and the label says exactly that. '
      + 'The whole Video training sets section also folds away now, like the '
      + 'two dataset sections above it.',
    to: '/datasets',
  },
  {
    id: '2026-08-30-watermark-engine-choice',
    date: '2026-08-30',
    title: 'Choose which engine finds your watermarks',
    blurb:
      'Both Find-watermarks windows now carry a Detection engine selector: the '
      + 'dedicated detector (SigLIP2 + Grounding DINO, ~10x faster, scored '
      + 'threshold) or your local vision model — with a line naming exactly '
      + 'what the next scan will run. The choice was always honoured by the '
      + "backend; now there's a control for it, stored once for both surfaces. "
      + 'Pair it with "Try on a sample first" to judge the two engines on the '
      + 'same images.',
    to: '/datasets',
  },
  {
    id: '2026-08-30-caption-budget-and-audio',
    date: '2026-08-30',
    title: 'The video export counts your words, and tells the truth about sound',
    blurb:
      'Two silent failures now speak up at Build the dataset. If captions run '
      + 'past the target model’s own published prompt budget (Wan caps at '
      + '200 words, 100 for I2V), the export says how many and how long the '
      + 'longest is — because the trainer would cut them mid-sentence without '
      + 'a word. And for targets that keep their audio (MiniMax H3), each '
      + 'clip’s prompt gains a measured Audio line when the numbers prove one '
      + '— a missing track, or near-total silence. Audible audio gets no '
      + 'invented description: only what was measured gets written.',
    to: '/video-bank',
  },
  {
    id: '2026-08-30-klein-clean-compare',
    date: '2026-08-30',
    title: 'Try your Klein models before the clean commits',
    blurb:
      'Watermark clean, Klein engine: a new ⚖ Compare models window runs each '
      + 'of your Klein checkpoints on the same flagged image — same zones, same '
      + 'seed — so the only difference between the results is the model. Pick '
      + "the winner: on a dataset it becomes the dataset's Klein model, on a "
      + 'bank it applies to that run. The original image is never touched.',
    to: '/datasets',
  },
  {
    id: '2026-08-30-sota-video-captions',
    date: '2026-08-30',
    title: 'Video captions grow up: a full paragraph, built like the measurements say',
    blurb:
      'Describe shots now writes 150-200 words per shot instead of a sentence '
      + 'or two — the length the published ablations converge on, where the '
      + 'whole gain lands on MOTION, exactly what a video LoRA learns. It '
      + 'watches 16 frames instead of 8 so that motion is actually visible, '
      + 'and the token budget follows. The camera line is no longer asked of '
      + 'the caption model (none describes it reliably — that is measured): '
      + 'the 🎥 Camera pass’s own classifier writes it into the exported '
      + 'prompt, in words it can prove, labeled the way MiniMax H3’s own '
      + 'prompts label their blocks. Expect the pass to take longer per shot '
      + '— it is reading twice the frames and writing four times the words.',
    to: '/video-bank',
  },
  {
    id: '2026-08-30-pass-info-dots',
    date: '2026-08-30',
    title: 'Every video pass button now explains itself',
    blurb:
      'A small ⓘ sits beside each pass of the video bank — Safe zone, Defects, '
      + 'AI check and the rest. It opens the guide’s own explanation right '
      + 'there, in a window, without leaving the page or losing your scroll: '
      + 'what the pass does, what it flags, and what to do with the result. '
      + 'Same text as the guide, so it can never drift out of date.',
    to: '/video-bank',
  },
  {
    id: '2026-08-30-describe-shots-window',
    date: '2026-08-30',
    title: 'Describe shots now asks its questions before it runs',
    blurb:
      'The video bank’s 🗣 Describe button opens a launch window instead of '
      + 'firing blind. Pick the wording there — Standard, or Plain, which '
      + 'names explicit content instead of describing around it (measured on '
      + 'real adult footage: the prompt matters more than the model). Pick '
      + 'the model too: the proven 4B default, or Qwen3-VL 8B for better '
      + 'motion writing — each saying whether it is already on your machine '
      + 'or downloads first. And choose what it covers: only the shots still '
      + 'missing a caption, or a rewrite of the whole bank in the new wording '
      + '— captions you edited by hand are never touched unless you '
      + 'explicitly say so.',
    to: '/video-bank',
  },
  {
    id: '2026-08-30-video-triage-exits',
    date: '2026-08-30',
    title: 'Un-decide video shots, and always reach the ✕',
    blurb:
      'Two dead ends gone from the video bank. Selected shots can now go '
      + '↩ back to triage — until now a mis-kept shot could only switch to the '
      + 'other verdict, never to “undecided”. And the shot player’s header '
      + 'stays pinned while you scroll, so the ✕ is always on screen — on a '
      + 'phone, where Esc does not exist, it was possible to scroll the only '
      + 'way out off the top of the page.',
    to: '/video-bank',
  },
  {
    id: '2026-08-30-lmstudio-download',
    date: '2026-08-30',
    title: 'Download LM Studio models without leaving the app',
    blurb:
      'The LM Studio card in Settings ▸ Local tools (and the Setup step) now '
      + 'downloads models — give it a model id like qwen/qwen3-vl-4b, or paste a '
      + 'huggingface.co model URL, and watch the progress. The download runs '
      + 'inside LM Studio itself, so reloading the page or restarting LDS does '
      + 'not stop it. The same parity Ollama has always had with its pull.',
    to: '/settings/local-tools',
  },
  {
    id: '2026-08-30-lmstudio-loads-itself',
    date: '2026-08-30',
    title: 'LM Studio models now load themselves',
    blurb:
      'No more opening LM Studio just to load the model: LDS loads it for you — '
      + 'automatically the first time captioning or framing needs it, or from the '
      + 'new ⏬ Load button in Setup and Settings ▸ Local tools. A model LDS '
      + 'loads is also one it can unload later to hand the GPU to ComfyUI; one '
      + 'YOU loaded is never touched.',
    to: '/settings/local-tools',
  },
  {
    id: '2026-08-30-promote-window-knows-the-numbers',
    date: '2026-08-30',
    title: 'The video promote window now tells you what the numbers mean',
    blurb:
      'Building a first video training set means guessing a target, a size and '
      + 'a clip count — so the window stops making you guess. Each target '
      + 'carries a one-line hint (which one is proven locally, which trains in '
      + 'the cloud, which needs reference photos). The size menu says which '
      + 'sizes train exactly as cut and which of the model’s stated sizes get '
      + 'rescaled a little. And a line under the clip count tells you where '
      + 'your dataset sits: a dozen clips proves the pipeline, strong LoRAs '
      + 'are typically trained on 50–200. All of it measured, none of it '
      + 'blocking — every field stays yours to set.',
    to: '/video-bank',
  },
  {
    id: '2026-08-30-start-lm-studio',
    date: '2026-08-30',
    title: 'Start LM Studio without leaving the app',
    blurb:
      'A stopped LM Studio server now has a ▶ Start button, in Settings ▸ Local '
      + 'tools and on the Setup step — the one Ollama has always had. Whatever '
      + 'model you had loaded stays loaded, and the server comes up on the port '
      + 'your settings name, not whichever one it used last. The button only '
      + 'appears once LM Studio has been opened at least once on this machine.',
    to: '/settings/local-tools',
  },
  {
    id: '2026-08-30-lm-studio-provider',
    date: '2026-08-30',
    title: 'Use LM Studio instead of Ollama, if that is what you run',
    blurb:
      'Captioning, framing, head-crop, Describe and Enhance can now run on '
      + 'LM Studio. The Setup wizard asks which one you run, and Settings ▸ Local '
      + 'tools switches it any time — the whole '
      + 'app follows — both the Dataset and the Bank pickers, and the GPU '
      + 'arbitration that keeps a vision model and ComfyUI from fighting over '
      + 'the card. Ollama stays the default and nothing changes unless you '
      + 'switch. LM Studio only serves a model you have loaded, so the app says '
      + 'so plainly when none is.',
    to: '/settings/local-tools',
  },
  {
    id: '2026-08-30-setup-without-ollama',
    date: '2026-08-30',
    title: 'Setup no longer stops at Ollama',
    blurb:
      'Ollama is optional, and the wizard finally treats it that way. With '
      + 'JoyCaption installed, captioning already works without it — JoyCaption '
      + 'writes the same captions the vision model would, prose or booru tags '
      + 'depending on what you train — so the step is a recommendation, not a '
      + 'gate. With neither installed you get an explicit "Continue without '
      + 'Ollama" that lists exactly what turns off first: auto-framing, '
      + 'head-crop, Describe & Enhance, the bank’s natural-language filter. '
      + 'Start Ollama later and everything switches back on by itself.',
    to: '/setup',
  },
  {
    id: '2026-08-30-h3-ref2va-training',
    date: '2026-08-30',
    title: 'Train MiniMax H3 Ref2V LoRAs — identity from reference images',
    blurb:
      'The Ref2V flavour of H3 generates from reference images of a subject, '
      + 'and now you can train for it: pick the MiniMax H3 Ref2V target when '
      + 'promoting clips, attach 1–4 reference images on the dataset card, and '
      + 'train locally or in the cloud with the same recipe H3 uses. The app '
      + 'refuses to launch without references on purpose — without them the '
      + 'trainer silently learns nothing of the identity, which is a paid run '
      + 'wasted. Local training needs an ai-toolkit from 2026-08-13 or newer; '
      + 'the app checks yours and says so instead of failing mid-run.',
    to: '/datasets',
  },
  {
    id: '2026-08-30-video-cloud-gpu-picker',
    date: '2026-08-30',
    title: 'Pick the GPU for a cloud video run — with prices in front of you',
    blurb:
      'Cloud video training used to rent the cheapest suitable card without '
      + 'showing you a number. The panel now lists one offer per GPU class — '
      + 'price per hour, VRAM, and a rough time and cost for your dataset — '
      + 'and lets you pick, or leave it on “cheapest suitable” as before. '
      + 'Estimates are honest about being rough: they come from one measured '
      + 'run, and a clip length nothing was measured at shows no estimate '
      + 'rather than an invented one.',
    to: '/datasets',
  },
  {
    id: '2026-08-30-video-trigger-word',
    date: '2026-08-30',
    title: 'Video datasets get a trigger word',
    blurb:
      'Set it once when promoting clips, and it is prepended to every clip’s '
      + 'caption file at export — exactly once, in one place. Your captions '
      + 'stay clean on screen, editing one never loses the trigger, and a '
      + 'caption that already starts with it is left alone: a doubled trigger '
      + 'measurably hurts prompt adherence, so the app makes doubling '
      + 'impossible. Optional — a style set legitimately has none.',
    to: '/video-bank',
  },
  {
    id: '2026-08-30-h3-i2v-training',
    date: '2026-08-30',
    title: 'Train MiniMax H3 LoRAs for image-to-video',
    blurb:
      'If you animate still images, train the way you generate: one checkbox '
      + 'on the video training panels switches an H3 run to first-frame '
      + 'conditioning, so the LoRA learns under the same setup your i2v '
      + 'generations use. Works locally and in the cloud, on any ai-toolkit '
      + 'that trains H3 at all.',
    to: '/datasets',
  },
  {
    id: '2026-08-30-h3-stills-training',
    date: '2026-08-30',
    title: 'Train an H3 video LoRA from your image datasets — no clips needed',
    blurb:
      'MiniMax H3 trains on still images too, and your image datasets already '
      + 'have everything that needs: curated pictures, edited captions, a '
      + 'trigger. One button in the Video training sets section turns an image '
      + 'dataset into a ready-to-train stills set — people have trained H3 '
      + 'character LoRAs this way on 12 GB cards. The promotion window also '
      + 'now counts clips coming from 48+ fps sources, which are often '
      + 'slow-motion footage that teaches floaty movement.',
    to: '/datasets',
  },
  {
    id: '2026-08-30-video-steps-sized-to-the-dataset',
    date: '2026-08-30',
    title: 'Video training steps now start from your dataset, not a constant',
    blurb:
      'A 12-clip set and a 176-clip set used to get the same step count. The '
      + 'Steps field on a video dataset now starts from a suggestion sized to '
      + 'the clips it actually holds — about 28 steps per clip, taken from '
      + 'measured runs, never below the old default and never past what the '
      + 'measurements support. And there is exactly ONE such field per dataset: '
      + 'the training block asks for the settings once, and training on this PC '
      + 'or on a rented GPU is just the button you press — no run starts on a '
      + 'number you never saw. Type over it freely: what you enter is what trains.',
    to: '/datasets',
  },
  {
    id: '2026-08-30-h3-trains-the-way-h3-is-trained',
    date: '2026-08-30',
    title: 'MiniMax H3 video LoRAs now train the way the model expects',
    blurb:
      'H3 ships guidance-distilled, and training a LoRA on it without accounting '
      + 'for that quietly degrades the result. ai-toolkit answered with a '
      + 'contrastive guidance loss and a small training adapter, and made the '
      + 'pair its default for H3 — video training here now uses both, wherever '
      + 'the ai-toolkit it is driving can actually run them: your installed copy '
      + 'is read for the capability, and a rented pod is judged on the image it '
      + 'boots, so an older setup quietly skips the recipe instead of failing. '
      + 'Alongside it, a clip now defaults to 39 frames instead of 107 — the '
      + 'length the trainer itself trains at, and about a third of the work per '
      + 'step — with every other length still on the menu, 22 included.',
    to: '/datasets',
  },
  {
    id: '2026-08-29-video-lora-in-the-cloud',
    date: '2026-08-29',
    title: 'Train a video LoRA on a rented GPU — MiniMax H3 included',
    blurb:
      'Promote clips out of a 🎬 video bank, pick a target model, and '
      + '☁️ Train in cloud now rents a pod that can actually run it: the '
      + 'right ai-toolkit, enough VRAM, and enough disk for a base that weighs '
      + '42 GB — all settled before the rental instead of discovered after it. '
      + 'MiniMax H3 has now been trained end to end that way, on clips cut to '
      + 'its own geometry (107 frames at 24 fps, audio kept at 32 kHz stereo), '
      + 'and the checkpoints come back to the dataset. Check your territory '
      + 'first: H3’s licence grants no rights in the EU, the UK, South Korea '
      + 'or the USA without MiniMax’s free authorisation, and it covers what '
      + 'you generate as well as the model.',
    to: '/datasets',
  },
  {
    id: '2026-08-29-krea-preset-sampler',
    date: '2026-08-29',
    title: 'A second way to sample Krea renders, built for its 8-step setting',
    blurb:
      'Krea 2 Turbo runs at eight steps, where the sampler has to make every '
      + 'one of them count. The Studio’s Sampler menu now offers five presets '
      + 'that change how those steps are taken — more texture and finer detail '
      + 'as you go up the scale, at no extra generation time. Pick “neutral” '
      + 'to render exactly as before, so you can judge the others against it at '
      + 'the same seed. It is optional: install it from the Krea card on the '
      + 'Setup screen, and everything works as it always did if you do not.',
    to: '/studio',
  },
  {
    id: '2026-08-29-caption-lab-on-a-bank',
    date: '2026-08-29',
    title: 'Try caption models on a bank before captioning thousands of images',
    blurb:
      'The 🧪 Caption Lab now runs on an image bank too: open the 🏷️ Caption '
      + 'window, press Caption Lab, pick one image, and line up to four configs — '
      + 'engine, vision model, vocabulary register and length — side by side. '
      + 'Nothing is written until you choose, and the winning config loads straight '
      + 'into the dials the next pass will use. A bank caption can also be edited by '
      + 'hand for the first time, and what you write is protected from a later '
      + 're-caption exactly as it is on a dataset.',
    to: '/bank',
  },
  {
    id: '2026-08-29-queue-hold-has-an-answer',
    date: '2026-08-29',
    title: 'A queue held by another app now has a way out',
    blurb:
      'When something outside LDS is holding a model on your graphics card, '
      + 'generations wait rather than evict it — and that wait could last as '
      + 'long as the other app did. The queue dock now tells you how long it '
      + 'has been waiting and offers Run anyway, which starts generating next '
      + 'to the other model instead (nothing of yours is unloaded, it can be '
      + 'slower, and the guard returns after fifteen minutes). An Ollama URL '
      + 'the app cannot use no longer stops image generation at all.',
  },
  {
    id: '2026-08-29-one-run-number',
    date: '2026-08-29',
    title: 'One number per training run, on every screen',
    blurb:
      'A cloud run used to wear two ids — its run number on the lineage '
      + 'cards and a different cloud number on the Checkpoints and Runs '
      + 'chips — and ⚙ Details on a checkpoints card could answer "not in '
      + 'the lineage tree" because of it. Every card, chip and tree row now '
      + 'prints the same single run number; the cloud id moved into the '
      + 'tooltip for when support asks. Deep links and stored data are '
      + 'untouched.',
  },
  {
    id: '2026-08-29-improve-settings-window',
    date: '2026-08-29',
    title: '✨ Improve now opens its settings in a window — and shows you the result',
    blurb:
      'Press ✨ Improve via Klein anywhere — dataset, Gallery, Canvas, a '
      + 'checkpoint gallery — and a window opens with the instruction '
      + '(editable in place), model, LoRA preset and output size, then a '
      + 'Generate button. Stay and the finished picture appears right there; '
      + 'leave early and it lands where it always did, with a toast saying '
      + 'where. The image viewers keep their action bars short — the batch '
      + 'toolbar keeps its inline note, since a batch should show its '
      + 'instruction before launching a lot.',
  },
  {
    id: '2026-08-29-run-details-from-checkpoints',
    date: '2026-08-29',
    title: 'Run details and run-vs-run compare, right on the checkpoint cards',
    blurb:
      'Every cloud-run card in a dataset’s checkpoints now has ⚙ Details '
      + '— the full recipe that trained it (rank, learning rate, optimizer, '
      + 'resolution, notes) — and ⇄ Compare: pick two runs to see exactly '
      + 'what changed between them, including the frozen dataset (images '
      + 'added, removed or re-captioned) and the machine. Same panels as the '
      + 'Lineage graph, one click closer.',
    to: '/datasets?section=training',
  },
  {
    id: '2026-08-29-dataset-made-with',
    date: '2026-08-29',
    title: 'Generated dataset images now remember what made them',
    blurb:
      'Every image a dataset generates — variations, ✨ improve, 📷 camera '
      + 'views, small-image rescues — is stamped with what actually ran: '
      + 'engine, base model, chained LoRAs, steps, seed. A folded ⚙ Made '
      + 'with block in the image actions panel shows it, in the same words '
      + 'as the Gallery viewer. Older images and imports simply show '
      + 'nothing — the stamp never guesses.',
    to: '/datasets?section=images',
  },
  {
    id: '2026-08-29-studio-viewer-facts',
    date: '2026-08-29',
    title: 'The Test Studio viewer now tells you everything about a render',
    blurb:
      'Open an image in the Test Studio — or in a comparison of two training '
      + 'runs — and you get the same full viewer as the Gallery: prompt, '
      + 'seed, checkpoint, extra LoRAs, base model, sampler, and the same '
      + 'verbs (download, improve, repair, camera angles), with the 👍/👎 '
      + 'vote kept right there. Comparing two runs no longer shows less '
      + 'about an image than the Gallery knows about the very same file.',
    to: '/studio',
  },
  {
    id: '2026-08-29-same-verbs-every-viewer',
    date: '2026-08-29',
    title: 'Every generated-image viewer now offers the same verbs',
    blurb:
      'Open a render anywhere — the Gallery, the ◉ Canvas, a checkpoint '
      + 'gallery — and the same footer is there: ⬇ Download, ✨ Improve, '
      + '✦ Repair and 📷 Camera angles. The Canvas used to lack the camera '
      + 'button and only the Canvas had Repair; now the viewer itself owns '
      + 'its verbs, so a picture has the same powers wherever you meet it.',
    to: '/gallery',
  },
  {
    id: '2026-08-28-krea-base-pick-saves',
    date: '2026-08-28',
    title: 'The Krea 2 base-model pick actually saves now',
    blurb:
      'Picking a Krea 2 base model from the variation catalog looked saved '
      + 'but quietly forgot the choice on the next reload — the save request '
      + 'was shaped wrong and the server ignored it politely. Fixed; your '
      + 'pick now survives, and the camera panel’s new Model row uses '
      + 'the same repaired path.',
    to: '/datasets?section=images',
  },
  {
    id: '2026-08-28-camera-model-choice',
    date: '2026-08-28',
    title: 'Camera angles can run on your own Qwen build',
    blurb:
      '📷 The camera-angles panel now has a Model row: pick any '
      + 'Qwen-Image-Edit build on your disk — a finetune, an NSFW merge — '
      + 'and every camera run uses it, on the Gallery and in datasets alike. '
      + 'Empty keeps the installed 2511 default. The angle grammar comes from '
      + 'the LoRA, so a different build changes the look, not the camera.',
    to: '/gallery',
  },
  {
    id: '2026-08-28-enhance-model-choice',
    date: '2026-08-28',
    title: 'Pick which Ollama model runs ✨ Enhance',
    blurb:
      'A ⚙️ next to ✨ Enhance — in the Test Studio and in the Canvas run '
      + 'panel alike — lets you pick which pulled Ollama model enriches your '
      + 'test prompt, instead of always the captioning model. The pick is '
      + 'remembered and applies to both surfaces at once; leave it on the '
      + 'default and nothing changes. A vanilla model can refuse NSFW '
      + 'prompts — the abliterated captioning default stays the safe choice '
      + 'there.',
    to: '/studio',
  },
  {
    id: '2026-08-28-studio-trigger-toggle',
    date: '2026-08-28',
    title: 'Test a prompt without the trigger word',
    blurb:
      'A new "Trigger word" checkbox next to the Studio test prompt (also in '
      + 'Compare and the canvas panel) controls whether the dataset\'s trigger '
      + 'is prefixed to what you type. Untick it to send the prompt exactly as '
      + 'written — handy when a render keeps typing the trigger back into '
      + 'speech bubbles or signs, or for pure style and scene tests. Ticked '
      + 'stays the default, the choice is remembered in this browser, and '
      + 'images generated without it say "no trigger" in their details.',
    to: '/studio',
  },
  {
    id: '2026-08-28-watermark-scan-window',
    date: '2026-08-28',
    title: 'Find watermarks gets the Find-text launch window',
    blurb:
      '🚩 Find watermarks now opens the same kind of window as 🔤 Find text, '
      + 'on both surfaces (the dataset button used to fire straight from the '
      + 'click): try a sample first — deterministic, so a re-run re-judges '
      + 'the same images — tune the detector threshold where its effect is '
      + 'judged (one stored value, both surfaces), and watch the flagged '
      + 'pages appear below the dials with their boxes drawn on them while '
      + 'the scan runs.',
    to: '/bank',
  },
  {
    id: '2026-08-28-header-machine-load',
    date: '2026-08-28',
    title: 'A resource monitor on every page — now with GPU temperature',
    blurb:
      'The 📊 machine-load readout is no longer Canvas-only: click 📊 in the '
      + 'header and every page — the Test Studio above all — shows live '
      + 'CPU · GPU · VRAM · RAM numbers, now joined by the GPU temperature, '
      + 'so you can watch a generation or a training work without keeping '
      + 'Task Manager or a ComfyUI monitor open. It polls only while the tab '
      + 'is visible, folds away with ▾, and remembers your choice. '
      + 'Suggested by Sam Exit (Discord).',
  },
  {
    id: '2026-08-28-clean-text-or-watermarks',
    date: '2026-08-28',
    title: 'Clean text and watermarks separately',
    blurb:
      'Once 🔤 Find text has flagged something, the repaint level grows a '
      + '“What to clean” switch — Both, 🔤 Text, 🚩 Marks — next to the '
      + 'LaMa/Klein toggle, on the bank panel and the dataset Clean row '
      + 'alike, and the button’s count follows the choice. The split is by '
      + 'page: a page carrying both counts as text and is repainted whole, '
      + 'so one page is never split between two runs.',
    to: '/bank',
  },
  {
    id: '2026-08-28-find-text-results-in-window',
    date: '2026-08-28',
    title: 'Find text shows its result in the launch window',
    blurb:
      'Launching 🔤 Find text no longer closes the window: the flagged '
      + 'pages appear right below the dials with every zone drawn on them, '
      + 'filling in live while the scan runs, and each tile opens the '
      + 'full-size page. Try a sample, judge the zones where you launched '
      + 'them, adjust, re-run: the whole loop happens in one window, on '
      + 'both surfaces.',
    to: '/bank',
  },
  {
    id: '2026-08-28-similar-add-more',
    date: '2026-08-28',
    title: 'Similar to selected: ask for the next batch without starting over',
    blurb:
      'After “Select 60 most similar” your selection holds 60 '
      + 'images — and the one-reference rule used to lock the panel '
      + 'shut right when you wanted more. It now remembers the last '
      + 'ranking: reopen Similar to selected and “Add N more” '
      + 'extends the SAME ranking by the next closest images — no '
      + 'unselecting, no hunting the reference down again.',
    to: '/bank',
  },
  {
    id: '2026-08-28-text-fill-outline-safe',
    date: '2026-08-28',
    title: 'Text cleaning stops breaking speech bubbles',
    blurb:
      'Repainting a 🔤 text zone used to hand the WHOLE rectangle to the '
      + 'repaint model, which kept eating balloon outlines and cartouche '
      + 'borders. Zones found by Find text now go through an outline-safe '
      + 'filler first: the letters are emptied with the bubble’s own '
      + 'background (instant, on the CPU), anything drawn across the zone '
      + 'edge — the outline, the art — is untouched by '
      + 'construction, and the repaint model only ever sees the leftover '
      + 'lettering on busy art. Both surfaces; ↩ Undo then Clean again '
      + 'upgrades pages you already cleaned.',
    to: '/bank',
  },
  {
    id: '2026-08-27-civitai-prompt-browser',
    date: '2026-08-27',
    title: 'Borrow a prompt from Civitai’s top images',
    blurb:
      'A new 🌐 Civitai button next to the test-prompt field (Test Studio, '
      + 'multi-LoRA comparison and the canvas alike) browses the most-reacted '
      + 'images of the day, week or month — each one shown right next to the '
      + 'prompt it was generated with, when the poster published it. One click '
      + 'copies it or drops it into your prompt field. Reading prompts uses '
      + 'the free Civitai API key from Settings → Scraping & sources.',
    to: '/studio',
  },
  {
    id: '2026-08-27-find-text-sample',
    date: '2026-08-27',
    title: 'Find text: try a sample, tune the sensitivity, then commit',
    blurb:
      'The 🔤 Find text launch window now carries two dials — on BOTH '
      + 'surfaces. “Try on a sample first” reads only the first N pages — '
      + 'judge the zones in the flagged review, then launch the rest, or '
      + 're-read the SAME sample after moving the new Sensitivity slider '
      + '(lower catches fainter lettering, at the cost of false zones — one '
      + 'stored value, moved from either side). No more committing a '
      + '9 000-page bank to find out.',
    to: '/bank',
  },
  {
    id: '2026-08-27-find-text-clean',
    date: '2026-08-27',
    title: 'Erase burned-in text — speech bubbles, subtitles, captions',
    blurb:
      'A comic page carries its dialogue, a screencap its subtitle — and a '
      + 'LoRA learns the lettering along with the subject. 🔤 Find text reads '
      + 'the text (Latin or CJK alike) and turns each block into a mask zone, '
      + 'so the same 🧽 Repaint that clears watermarks erases it — one funnel, '
      + 'one ↩ Undo, and ✂ Auto-crop never touches a bubble. On banks and '
      + 'datasets both, CPU-only, powered by the same small offline OCR the '
      + 'Video bank already uses. Very stylised sound-effect lettering can '
      + 'still escape the reader — the mask editor covers those.',
    to: '/bank',
  },
  {
    id: '2026-08-27-scene-custom-prompt',
    date: '2026-08-27',
    title: 'Scenes take a custom prompt of their own',
    blurb:
      'In the Test Studio\'s 🎬 Scenes panel, every picked scene now carries an '
      + 'optional ✏️ text field. Whatever you type there is appended to that '
      + 'scene\'s caption at launch — swap an outfit, set the time of day, add '
      + 'your trigger word — without touching the caption itself or the other '
      + 'scenes. Leave it empty and the caption runs exactly as before.',
    to: '/studio',
  },
  {
    id: '2026-08-26-klein-enhancement-repair-row',
    date: '2026-08-26',
    title: 'The Klein enhancement LoRA can finally be repaired from Setup',
    blurb:
      'The detail LoRA behind ✨ Upscale & improve installed itself on demand '
      + 'but appeared nowhere on Setup — so a broken or deleted file could '
      + 'only be fixed by triggering an improve and hoping. It now has its own '
      + 'row in the Install screen\'s repair menu, like every other weight. '
      + 'Found by a new internal guard that walks every install the app can '
      + 'run and fails when one is offered on no screen — so the next engine '
      + 'cannot ship half-visible the way three lanes did before it.',
    to: '/setup',
  },
  {
    id: '2026-08-26-camera-angles-dataset',
    date: '2026-08-26',
    title: 'Camera angles inside a dataset — with the angle already captioned',
    blurb:
      'Open any kept image of a dataset and press 📷 Camera angles: the views '
      + 'arrive as pending candidates in the ordinary keep/reject cycle, and '
      + 'each one is born knowing its own caption fragment — "seen from '
      + 'behind, low camera angle" — which the captioner then completes and '
      + 're-injects on every later pass. That phrase is the point: an angle '
      + 'left undescribed binds to the trigger word, and the angle is the one '
      + 'fact a vision model cannot reliably see while the app knows it '
      + 'exactly, because you asked for it. Imports and ✨ results are valid '
      + 'sources; camera views are not re-shot from camera views. The Bank '
      + 'deliberately does not carry the button — it is the reservoir of real '
      + 'photos; promote to a dataset first, and the views are born as '
      + 'candidates, never filed as real.',
    to: '/datasets?section=images',
    image: 'docs/screenshots/release/camera-angles-picker.png',
  },
  {
    id: '2026-08-26-camera-setup-card',
    date: '2026-08-26',
    title: 'Install 📷 Camera angles from Setup, before the first click',
    blurb:
      'The camera weights used to install only when you pressed 📷 with them '
      + 'missing. Setup now shows the lane properly: a one-click install card '
      + 'on the Install screen (~21.6 GB, shared parts skipped when another '
      + 'engine already brought them), a row per weight in the repair menu so '
      + 'a broken download can be fixed alone, and Camera angles is counted on '
      + 'the readiness screen — a machine without it reads "not ready, here is '
      + 'the install", never a shorter list that certifies completeness by '
      + 'leaving it out.',
    to: '/setup',
  },
  {
    id: '2026-08-26-gallery-download-files',
    date: '2026-08-26',
    title: 'Download a Gallery selection as plain files — no ZIP to unpack',
    blurb:
      'Select images in the Gallery and press ⬇ Files: each one saves to your '
      + 'Downloads as its own file, under the same lineage name the ZIP would '
      + 'have used — dataset, run, step and seed. Built for the places an '
      + 'archive is a chore: grabbing three pictures, a phone, or a training '
      + 'tool watching a folder. Files save one at a time (your browser may ask '
      + 'once to allow multiple downloads), a picture whose file was cleaned '
      + 'off the disk is skipped and counted rather than stopping the rest, '
      + 'and leaving Select mode stops the run. The camera picker also lost '
      + 'its 12-view cap: pick as many angles as you want — the button states '
      + 'the cost, and long runs warn instead of being blocked.',
    to: '/gallery',
  },
  {
    id: '2026-08-26-camera-angles',
    date: '2026-08-26',
    title: 'Walk around your subject: re-shoot any picture from another camera position',
    blurb:
      'Open a picture in the Gallery and press 📷 Camera angles: pick where the '
      + 'camera stands on the dial, how high it is and how close, and the app '
      + 'renders that scene from there — the subject stays put and the '
      + 'background moves with the camera, so what was behind them comes into '
      + 'view. This is not the "profile view" shot the catalog already had: '
      + 'that one turns the person and leaves the room where it was. Eight '
      + 'sides of a subject at eye level is one gesture and about two minutes. '
      + 'First use downloads the weights from Setup ▸ ComfyUI. The Gallery '
      + 'carries a Beta chip while this settles: distance is a hint the model '
      + 'mostly honours, and whatever the original photo never showed is '
      + 'plausible rather than real.',
    to: '/gallery',
  },
  {
    id: '2026-08-26-klein-in-the-test-studio',
    date: '2026-08-26',
    title: 'Test Studio can finally generate with your FLUX.2 Klein LoRAs',
    blurb:
      'Klein LoRAs trained and deployed, but the Test Studio had no Klein '
      + 'generation lane, so clicking Generate from the board answered "no '
      + 'Z-Image model available" about a family you never picked. Klein now '
      + 'has a lane of its own: fixed-seed checkpoint and strength grids, the '
      + 'same as Krea and Z-Image, built on your configured Klein model, text '
      + 'encoder and VAE. Guidance is pinned where a distilled model wants it, '
      + 'so a swept CFG cannot burn a cell and make a good checkpoint look bad. '
      + 'FLUX.1 and Anima still have no lane, and now say so plainly instead of '
      + 'blaming Z-Image. Reported by lunchingfriar.',
    to: '/studio',
  },
  {
    id: '2026-08-26-klein-loras-report-as-deployed',
    date: '2026-08-26',
    title: 'Klein, FLUX.1 and Anima LoRAs finally report as deployed',
    blurb:
      'Deploying a FLUX.2 Klein LoRA worked, and then nothing believed it: the '
      + 'button never flipped to Deployed, Generate said the checkpoint was not '
      + 'deployed yet, and "deploy then generate" went green and instantly red. '
      + 'The file was written to the Klein folder and looked for in the Z-Image '
      + 'one, because the studio only knew three families while the app deploys '
      + 'six. Klein, FLUX.1 and Anima now read their own folders, and they also '
      + 'appear in the Test Studio family picker, where they were missing for '
      + 'the same reason. Reported by lunchingfriar.',
    to: '/canvas',
  },
  {
    id: '2026-08-25-selection-stops-tinting',
    date: '2026-08-25',
    title: 'Selecting an image no longer puts a colour film over it',
    blurb:
      'A selected image in the Bank used to be covered by a coloured overlay — '
      + 'which is a problem when the thing you are deciding on is the colour of '
      + 'the photo. Selection is now marked the way an editor marks a contact '
      + 'sheet: a pencil stroke in the corner and a ring around the frame, with '
      + 'the picture left alone. Same mark in the dataset grid, so the gesture '
      + 'reads the same on both.',
  },
  {
    id: '2026-08-25-review-resumes-where-you-clicked',
    date: '2026-08-25',
    title: 'Review picks up where you left off',
    blurb:
      'Starting ▶ Review from a tile halfway down a bank used to review that '
      + 'one shot and then jump back to image #1, so the only way to resume a '
      + 'triage was to decide on everything in between. It now continues from '
      + 'the shot you clicked, and the counter says where you actually are — '
      + '← still steps back over what you passed. Reported by nofaceman.',
    to: '/bank',
  },
  {
    id: '2026-08-25-safelight-icons',
    date: '2026-08-25',
    title: 'Real icons everywhere, instead of emoji',
    blurb:
      'Buttons, the navigation, the settings and workspace rails, the review '
      + 'lightboxes and the tile badges now use one drawn icon set instead of '
      + 'emoji. They look the same on every machine — emoji were rendered by '
      + 'your operating system, so the app never looked twice alike — and they '
      + 'stay legible at small sizes. Nothing moved: same buttons, same words, '
      + 'same places.',
  },
  {
    id: '2026-08-25-safelight-look',
    date: '2026-08-25',
    title: 'A new look: neutral darkroom greys, one amber accent',
    blurb:
      'The whole app moves to "Safelight": neutral, opaque greys inspired by '
      + 'photo-editing darkrooms — so the interface never tints the images you '
      + 'are judging — with a single amber accent replacing the old purple '
      + 'gradient, and a crisper typeface (Archivo). Same layout, same '
      + 'controls, calmer room.',
    image: 'docs/screenshots/release/2026-08-25-safelight.png',
  },
  {
    id: '2026-08-25-comfyui-submit-patience',
    date: '2026-08-25',
    title: 'Far fewer false "paused ComfyUI job" banners',
    blurb:
      'Sending a job to a busy ComfyUI used to give up after 10 seconds — and '
      + 'because the app could no longer tell whether the job had been accepted, '
      + 'it raised the paused-job banner that asks you to restart ComfyUI, often '
      + 'again on the very next try (reported by charlesangus on GitHub). The app '
      + 'now waits up to two minutes for a busy server to answer, and a ComfyUI '
      + 'that is simply not running fails the job cleanly so you can just retry — '
      + 'the banner is reserved for jobs whose outcome is genuinely unknown.',
    to: '/datasets',
  },
  {
    id: '2026-08-25-repair-lanpaint',
    date: '2026-08-25',
    title: '✦ Repair stops smearing — a real inpainting sampler under the mask',
    blurb:
      'Masked Repair used to condition Klein like an inpaint-trained model, which '
      + 'it is not, and the painted area came back a smeary mess (reported by '
      + 'charlesangus on GitHub). It now runs on LanPaint, a training-free '
      + 'inpainting sampler; the mask grows a few pixels so edges rebuild '
      + 'cleanly, and a localized repair travels as a native-resolution crop '
      + 'instead of a scaled-down full frame. One small install in Setup — the '
      + 'new "LanPaint sampler" row — then restart ComfyUI.',
    to: '/setup',
  },
  {
    id: '2026-08-24-crop-clears-dataset-framing',
    date: '2026-08-24',
    title: 'Crop a shot, and Composition forgets the old framing',
    blurb:
      'Cropping a body shot into a face used to leave Composition still counting '
      + 'it as a body. A crop now clears that shot type — the same way the Bank '
      + 'already does — so the image drops out of the mix until 📐 Classify framing '
      + 're-reads just the ones you cut. Untouched images stay as they were.',
    to: '/datasets?section=add',
  },
  {
    id: '2026-08-24-gallery-page',
    date: '2026-08-24',
    title: 'A Gallery of everything you ever generated',
    blurb:
      '🖼 Gallery in the top bar is one feed of every image the app made — Test '
      + 'Studio cells, Canvas previews, comparison runs and ✨ improvements — '
      + 'across every dataset, newest first, with filters (dataset, renders vs '
      + 'improved, 👍 liked). The viewer walks the feed with ‹ › or the arrow '
      + 'keys, shows everything a picture was made from, and carries the '
      + 'actions you already know: ⬇ Download under its lineage name, ✨ '
      + 'Upscale & improve (result lands at the top of the feed), and a Select '
      + 'mode to 🗑 delete misses or ⬇ ZIP a pick. The feed loads itself as '
      + 'you scroll towards its end. Built for the phone as much as the '
      + 'desktop.',
    to: '/gallery',
  },
  {
    id: '2026-08-24-settings-groups-everywhere',
    date: '2026-08-24',
    title: 'The rest of Settings gets the same organised layout',
    blurb:
      'Local tools, Captioning & quality, Training and Storage now open on the '
      + 'same clickable summary of collapsible groups Image engines got — the '
      + 'vast.ai key finally sits WITH the cloud-training guardrails, the '
      + 'movable folders stand apart from the cleanup tools, and which groups '
      + 'you keep open is remembered per section. Every Settings link and '
      + 'search result still lands on its field: a collapsed group opens '
      + 'itself on the way. Scraping, Server and Maintenance keep their one '
      + 'or two cards flat — a summary there would just be noise.',
    to: '/settings/local-tools',
  },
  {
    id: '2026-08-24-use-these-improve-settings',
    date: '2026-08-24',
    title: 'Like a ✨ result? Make the next improves run the same way',
    blurb:
      'Open any Upscale & improve result (Gallery, checkpoint gallery, or '
      + 'pinned on the Canvas) and press ↩ Use these improve settings: the '
      + 'instruction, LoRA preset, strength, steps, output size and Klein '
      + 'model this image was made with become the app-wide improve settings '
      + 'again. New improvements record all of it from now on; older images '
      + 'restore what they carry (instruction + preset), and the toast names '
      + 'exactly which halves happened — a preset renamed or deleted since '
      + 'is said out loud instead of silently dropped. SeedVR2 results have '
      + 'no settings to restore and show no button.',
    to: '/gallery',
  },
  {
    id: '2026-08-24-engines-settings-groups',
    date: '2026-08-24',
    title: 'Image engines settings, organised into groups',
    blurb:
      'The section had grown into a wall of eleven cards — API keys next to '
      + 'Klein pins next to the improve prompt. It now opens on a clickable '
      + 'summary of six groups (Engines & API keys · Klein · Krea 2 · '
      + 'Generation LoRA presets · SeedVR2 · Prompts & improve tuning), each '
      + 'collapsible and remembered across visits. Every Settings link and '
      + 'search result still lands on its field — a collapsed group opens '
      + 'itself on the way.',
    to: '/settings/engines',
  },
  {
    id: '2026-08-24-improve-output-size-in-note',
    date: '2026-08-24',
    title: 'Pick the improve output size right under the ✨ button',
    blurb:
      'The ✨ Upscale & improve note now carries the Output size (MP) box — '
      + 'the same 0.5–8 MP value Settings edits (klein.improve_megapixels), '
      + 'changeable without leaving your images. App-wide, like the '
      + 'instruction and the LoRA preset beside it.',
    to: '/datasets?section=images',
  },
  {
    id: '2026-08-24-improve-lora-preset',
    date: '2026-08-24',
    title: 'Upscale & improve can now chain your LoRA presets',
    blurb:
      'The ✨ pass ran with its instruction and nothing else — your generation '
      + 'LoRA presets (Settings ▸ Engines) never applied to it. The improve '
      + 'note now has a LoRA preset picker next to the instruction editor: '
      + 'pick one and every Klein improve chains it after the consistency '
      + 'LoRA — the single pass, the 🔄 re-run and the batch alike, in every '
      + 'dataset (it is app-wide, like the instruction, and the panel says '
      + 'so). SeedVR2 stays a pure restoration. A renamed or deleted preset '
      + 'quietly runs as None, never a blocked pass — and the result records '
      + 'which LoRAs actually ran in its details.',
    to: '/datasets?section=images',
  },
  {
    id: '2026-08-24-bank-forgets-missing-images',
    date: '2026-08-24',
    title: 'A bank can now let go of images that are really gone',
    blurb:
      'The "no longer in the folder" warning used to have one remedy — Move '
      + 'folder… — which did not help when the files were really deleted (a '
      + 'downloader that cleans up after itself, a by-hand tidy): the ghost rows '
      + 'failed to load for ever and kept counting against the bank\'s ceiling. '
      + 'The warning now also offers 🧹 Forget missing: after a fresh check and '
      + 'a confirmation with the exact count, the bank drops just those rows. '
      + 'Files on disk are never touched, and a disconnected drive is refused '
      + 'outright — it can never erase your triage.',
    to: '/bank',
  },
  {
    id: '2026-08-23-the-app-opens-lighter',
    date: '2026-08-23',
    title: 'The app opens on a bundle seven times lighter',
    blurb:
      'Every screen used to ship in one 3.4 MB file your browser had to fetch and parse '
      + 'before anything painted. Each page now loads its own, much smaller piece the first '
      + 'time you visit it — the initial load carries 0.5 MB — and this 🎁 panel keeps only '
      + 'recent news up front, with a "Show older updates" button that fetches the 500-entry '
      + 'history on demand. If a tab is open across an "Update & restart", the first click '
      + 'after the update reloads once by itself instead of dying on files that moved.',
  },
  {
    id: '2026-08-23-setup-model-check-reads-the-pickers-truth',
    date: '2026-08-23',
    title: 'The Setup model check now reads exactly what the pickers read',
    blurb:
      'Setup had its own way of scanning your model folders — one level deep, with its own '
      + 'family rules — while the pickers and the generate path had long moved to the full '
      + 'recursive scan. So a model filed two folders down generated fine but showed ✗ in '
      + 'Setup, and a Z-Image build in a hyphenated Z-Image folder showed ✓ in Setup while '
      + 'the Test Studio picker could not see it. One scan now feeds all three — what Setup '
      + 'checks is what the pickers offer is what Generate loads — and the hyphen spelling '
      + 'is recognised everywhere.',
    to: '/setup',
  },
  {
    id: '2026-08-23-drop-in-extensions',
    date: '2026-08-23',
    title: 'Drop a local extension in, and it loads at boot',
    blurb:
      'A new backend/extensions/ folder loads optional local packages at start: each one '
      + 'registers its own API routes and can mount its own UI script, and the app lists what '
      + 'loaded. Nothing changes on a normal install — the folder does not exist, and it can '
      + 'never reach a release bundle (three separate locks, each pinned by its own test). An '
      + 'extension is code you place on your own machine, trusted like the app itself; it loads '
      + 'behind the access-token gate, a broken one is skipped instead of taking the app down, '
      + 'and LDS_EXTENSIONS=0 turns the whole mechanism off. The contract is documented in '
      + 'docs/guide/extensions.md.',
  },
  {
    id: '2026-08-23-preview-steps-and-cfg',
    date: '2026-08-23',
    title: 'Set how your training previews are rendered — steps and CFG',
    blurb:
      'Advanced options now carries a Preview quality pair: how many steps a preview image '
      + 'gets, and at what guidance. Until now both were fixed by the base you picked, which '
      + 'is right for the models the studio ships and wrong the moment you train on something '
      + 'else — a distilled model wants 8 steps and an undistilled one wants 25, and at the '
      + 'wrong number your previews come back either as unfinished sketches or slower than the '
      + 'training they interrupt. Leave the boxes empty and nothing changes: they show the '
      + 'default your base resolves to. They also work when continuing a run, full-state resume '
      + 'included, because they only touch the picture and never the weights. '
      + 'Suggested by charlesangus (GitHub #46).',
    to: '/datasets',
  },
  {
    id: '2026-08-23-runpod-pod',
    date: '2026-08-23',
    title: 'Run the whole studio on a rented RunPod GPU',
    blurb:
      'Point a RunPod pod at the GPU image and reach the studio, the Image Bank and ComfyUI generation '
      + 'from any browser, with your datasets on a network volume that survives restarts. Training still '
      + 'runs on vast.ai, and the guide is honest about what has not been measured on real hardware yet — '
      + 'it also turns a proxy 404 into the exact log line that explains it. Along the way the studio '
      + 'learned to create its data folder instead of only checking it, which fixes any install pointing '
      + 'LDS_DATA_DIR at a location that does not exist yet. (Contributed by @Cyberschorsch.)',
  },
  {
    id: '2026-08-23-public-bind-token',
    date: '2026-08-23',
    title: 'Reaching the studio over the internet now always asks for a token',
    blurb:
      'Running the studio on a public address — a rented GPU box, a tunnel — used to be open to anyone '
      + 'who found the URL, because the token gate is off by default for trusted home networks. Set '
      + 'LDS_PUBLIC=1 and the gate is forced on, a token is generated for you, and Settings shows it '
      + 'instead of a switch that does nothing. (Contributed by @Cyberschorsch.)',
    to: '/settings/server',
  },
  {
    id: '2026-08-23-studio-guest-checkpoints',
    date: '2026-08-23',
    title: 'Test their LoRA next to yours, same prompt and seed',
    blurb:
      'The Test Studio could only tick epochs you trained here, so a character LoRA from another trainer had nowhere to sit in the grid. Compare with other LoRAs (under Checkpoints to test) adds files from your ComfyUI loras folder as their own rows — not stacked on yours — with the same prompt, seed and strengths. Tick any of yours and any of theirs; the counts do not have to match. Their trigger is not injected, so the comparison is the prompt you typed. (Contributed by @OneCodingDude.)',
    to: '/studio',
  },
  {
    id: '2026-08-23-bank-datasets-and-studio-fit-a-phone',
    date: '2026-08-23',
    title: 'The Bank, the Datasets and the Test Studio now fit a phone',
    blurb:
      'The same measuring pass that fixed the Canvas has been run over the three pages you '
      + 'actually live in, at five real screen sizes, and what it found is fixed. Every button, '
      + 'chip and menu item on those pages is finger-sized below desktop widths. The Bank header '
      + 'gives the screen back on a phone: the counters and the action row scroll on one line '
      + 'instead of stacking, and a phone held sideways gets a one-row header. The passes panel '
      + 'no longer opens itself on a phone (it was 1 500 px tall there) and, below desktop '
      + 'widths, folds everything that is not a pass button so the passes stay one tap away. '
      + 'In a dataset, the two chip rails no longer touch, and the in-section shortcuts fold on '
      + 'a phone held sideways — the section buttons still reach every panel. Nothing changes on '
      + 'a desktop.',
    to: '/bank',
  },
  {
    id: '2026-08-22-a-base-model-filed-deeper-than-one-folder-is-found',
    date: '2026-08-22',
    title: 'A Krea or Klein base model filed two folders deep is found, like ComfyUI finds it',
    blurb:
      'If you named a “Base model file” that sat more than one folder deep — a folder inside '
      + 'your Krea folder — the app looked only one level down, did not find it, and refused to '
      + 'run with “not on disk” about a file that was right there (and before that guard '
      + 'existed, it quietly generated with a different Krea build instead). The Test Studio '
      + 'already saw those files; Generate did not. Both now look as deep as you have filed '
      + 'things, in the same folders and the same order ComfyUI itself searches — so what the '
      + 'Studio lists is what Generate can load, and a name you type resolves wherever it sits. '
      + 'When two files share a name, the shallower one wins, and a `.git` folder is skipped '
      + 'exactly as ComfyUI skips it.',
    to: '/settings/engines',
  },
  {
    id: '2026-08-22-scene-prompts-from-a-dataset',
    date: '2026-08-22',
    title: 'Replay your own dataset’s captions as scenes, not just a bank’s',
    blurb:
      '🎬 Scenes could run a BANK’s captions in order — which meant the sequence you most '
      + 'wanted to replay, the one in the dataset you captioned and curated yourself, was the '
      + 'one place it would not read from. The section under the prompt (Test Studio and the '
      + 'board’s 🎨 Generate) now starts with two buttons, 🗃 Bank and 📁 Dataset: pick either, '
      + 'load its captions in order — each card showing the image it came from — tick the ones '
      + 'you want, and every ticked scene becomes one pass of the same run. Everything else is '
      + 'unchanged, deliberately: same checkpoints, same settings, same seed, and an image with '
      + 'no caption is still skipped and counted rather than guessed. A dataset reads its KEPT '
      + 'and pending images only — the ones you rejected stay out, because you already answered '
      + 'that question. And the captions ride without the trigger word, exactly as they are '
      + 'stored, so the run prepends the trigger of the LoRA you are actually testing: scenes '
      + 'written for one character replay against another.',
    to: '/studio',
  },
  {
    id: '2026-08-22-typed-captions-survive-a-forced-pass',
    date: '2026-08-22',
    title: 'A forced re-caption no longer overwrites what you typed',
    blurb:
      'The caption editor has promised since the day it shipped that a hand-written caption '
      + 'survives a forced 🔄 Re-caption. The Bank kept that promise; the dataset overwrote '
      + 'everything on every forced batch, your own words included. Both surfaces spare them '
      + 'now — and a caption you type WHILE an image is still being captioned wins over the '
      + 'answer that comes back for it. Naming images explicitly stays the way to re-caption '
      + 'them anyway, which is what the identity-leak panel does.',
    to: '/datasets?section=captions',
  },
  {
    id: '2026-08-22-dataset-watermark-scan-runs-again',
    date: '2026-08-22',
    title: '🧽 Find watermarks runs again on a dataset',
    blurb:
      'The dataset scan stopped on its very first image when it ran through the detector: it '
      + 'was reading one field fewer than the scan hands back, while the Bank read them all. '
      + 'Fixed — and pinned by a test that reads BOTH surfaces, so the two cannot drift apart '
      + 'again without something going red.',
  },
  {
    id: '2026-08-22-bank-stops-filing-distant-faces-too-small',
    date: '2026-08-22',
    title: 'The Bank stops filing distant faces as “too small”',
    blurb:
      'A head in a full-body shot reaches the face model a few pixels wide however large the '
      + 'file is, because the detector fits the whole frame into its window before it looks. '
      + 'The dataset scorer already rescued those by looking again at a crop around the head; '
      + 'the Bank did not, and left them unscored. Same rescue on both surfaces now, with the '
      + 'same numbers, held together by one test.',
  },
  {
    id: '2026-08-22-generation-queue',
    date: '2026-08-22',
    title: 'Line your work up instead of waiting on it',
    blurb:
      'Starting a generation no longer switches off the others. Fire an ✨ Upscale & improve '
      + 'batch, then launch a ⚡ Generate, then retry a tile — they queue behind each other and '
      + 'run in turn, instead of greying out the whole workspace until the first one finished. '
      + 'A new dock in the bottom-left corner shows that queue for the first time: what the GPU '
      + 'is working on right now, what is waiting behind it and where each job came from — the '
      + 'dataset, the Test Studio, the Canvas or the Bank. You can send one job to the front, or '
      + 'cancel it, without stopping the batch it belongs to. The dock stays out of sight while '
      + 'the queue is empty. Suggested by charlesangus (GitHub #44).',
    // Wired AFTER this entry's release went out (its id shipped in v2026.08.22,
    // the screenshot mechanism landed the morning after, and v2026.08.23 was cut
    // without anyone adding the field) — so this picture missed its one train:
    // release notes select NEW ids only, and an image added to an already-shipped
    // entry never appears anywhere. Kept referenced so the file is not an orphan,
    // and so the wiring test below the mechanism can hold the pairing.
    image: 'docs/screenshots/release/generation-queue-dock.png',
  },
  {
    id: '2026-08-20-caption-appearance-policy',
    date: '2026-08-20',
    title: 'Choose whether hair, makeup, facial hair and glasses bind to the trigger',
    blurb:
      'What a caption does not name binds to the trigger, and Extra instructions could not '
      + 'change that: hair was always forbidden, makeup was never asked for, and mascara still '
      + 'baked in. Captions ⚙️ Options on a character dataset now has Appearance in captions — '
      + 'Omit or Describe for hair, makeup and nails, facial hair, and glasses. Face, eyes, '
      + 'skin, age, gender and ethnicity stay omitted. Untouched datasets keep the classic lock '
      + 'until you flip a row; then makeup defaults to Describe so it cannot silently bind. '
      + 'Re-caption to apply. Suggested by Sam Exit and Meeseeks (Discord).',
    to: '/datasets?section=captions',
  },
  {
    id: '2026-08-20-viewer-pinch-zoom',
    date: '2026-08-20',
    title: 'Zoom into a render to see whether it actually got the detail right',
    blurb:
      'Folding the details away gave the picture the window on a tablet and a desktop, and '
      + 'barely moved on a phone held upright — measured, 35 % of the screen became 39 %. The '
      + 'panel was never the limit there: a 4:3 render on a 412-px screen already has the whole '
      + 'width, so seeing more means magnifying, not folding. The image viewer now zooms. Pinch '
      + 'it, double-tap it, or roll the wheel on a desktop, then drag to move around; a second '
      + 'double-tap, Esc, or the ⤾ chip that appears puts it straight back. It zooms around your '
      + 'fingers, so pinching on a face makes that face bigger instead of the middle of the '
      + 'picture, and it stops exactly where the file does — one screen pixel per stored pixel, '
      + 'never magnified guesswork you could mistake for detail. The picture can never be '
      + 'dragged off the screen either: it always covers the window, so there is no way to end '
      + 'up looking at black with no way back. Every render opens at fit, including the next one '
      + 'you flip to.',
  },
  {
    id: '2026-08-20-lightbox-hide-the-details',
    date: '2026-08-20',
    title: 'Put the details away and give the render the screen',
    blurb:
      'The image viewer shows you what a render was made from — seed, settings, prompt — and '
      + 'that panel is the point of it. It is also not what you want on screen while you are '
      + 'actually looking at the picture: measured on a phone the render was drawn at 35 % of '
      + 'the screen, and on a tablet held sideways the same 35 %, with the panel taking the rest. '
      + 'There is a new ⤢ button beside the ✕ that folds the whole panel away — and tapping the '
      + 'picture does it too, the gesture every photo viewer already has. The picture then takes '
      + 'the entire window, frame and padding included: 90 % of a tablet held sideways, 84 % of a '
      + 'desktop window. Tap again, or press ⓘ, and everything comes back exactly where it was — '
      + 'including while you flip from one render to the next, so comparing two crops does not '
      + 'mean re-hiding the panel each time.',
  },
  {
    id: '2026-08-20-canvas-room-to-work',
    date: '2026-08-20',
    title: 'The ◉ Canvas gets its screen back on a phone',
    blurb:
      'The board is the whole point of the page, and on a phone it was getting half the screen: '

      + 'the filter bar wrapped onto two rows, the toolbar under it onto two more, and the page '

      + 'title repeated a word the nav bar was already highlighting. Measured on a 412-px phone, '

      + 'the board had 297 px to work in — 50 % of the page. It now has 451, which is 76 %, and on '

      + 'a folding phone opened out it goes from 57 % to 72 %. Nothing was taken away to get there. '

      + 'The toolbar is ranked instead: zoom, Fit and 🎨 Generate stay where your thumb is, and a new '

      + '⋯ button holds ✦ Tidy up, 💾 Layouts, 📷 PNG, 🔌 external LoRAs, ⏏ Undeploy, the colour key, '

      + 'the machine load and the full list of board gestures — as a sheet that floats over the '

      + 'board, so opening it never pushes the board down. ⋯ shows a badge when an external LoRA is '

      + 'on the board, so nothing it holds can go quietly. The chips in there carry their words '

      + 'again, and CPU/GPU/VRAM now reads from a phone too — which is the screen you check the '

      + 'machine from when you are not sitting at it. The desktop toolbar drops to a single row as '

      + 'well, because that ~500-character gesture line had been giving it a second one at every '

      + 'width, 1920 included.',
    to: '/canvas',
  },
  {
    id: '2026-08-20-scene-prompts-from-a-bank',
    date: '2026-08-20',
    title: 'Run a bank’s captions in order, as one batch',
    blurb:
      'The 🎲 shortcut draws ONE caption at random — the right tool for a bag of images, the wrong one when the ORDER is the point: a storyboard, a shoot, a chapter read page by page. Both generation panels (the Test Studio and the board’s 🎨 Generate) now have 🎬 Scenes from a bank under the prompt: pick a bank, load its captions in bank order — each shown with the image it came from — tick the ones you want, and every ticked scene becomes one pass of the same run, in order, alongside anything you ticked in the prompt history. Same checkpoints, same settings, same seed, so the scenes stay comparable. An image with no caption is skipped and counted rather than guessed, and the button and the counter say how many passes before you click.',
    to: '/canvas',
  },
  {
    id: '2026-08-20-coverage-chips-show-their-images',
    date: '2026-08-20',
    title: 'Click a coverage chip to see exactly those images',
    blurb:
      'The 🔍 Coverage panel could tell you that three captions mention a profile. It could not tell you WHICH three, so acting on it meant scrolling a grid of two hundred looking for them — the panel was easy to read and hard to use. Every chip with a count is now a button: click frontal 35, or nude 7, or backlit 1, and the grid opens showing exactly those images, with 🔍 profile — camera view in the filter bar and clear all beside it. It composes with everything already there, so filter to the profiles and Sort ▸ Shot type puts what is left in order. The images you get are the ones the number counted and no others: rejected and failed pictures are outside the panel, so they stay outside its filter. A chip showing zero stays a plain chip, because there is nothing to show you and the answer to that gap is generating, not filtering. Still advice only: it changes what you are looking at, never what your images are. (Asked for by .samexit on Discord.)',
    to: '/datasets?section=add&panel=generate',
  },
  {
    id: '2026-08-20-group-the-grid-by-shot-type',
    date: '2026-08-20',
    title: 'Compare like with like: group the grid by shot type',
    blurb:
      'The grid shows your images in the order they arrived, which means a face shot, then a back shot, then two bodies, then another face — and every question you actually ask at that point is about ONE kind at a time: do I have too many of these, not enough of those, and which of these near-identical ones do I keep? The Sort menu above the grid has two new entries. Shot type puts every face shot in one run, then the busts, then the bodies, then the backs, in the same order the Composition bar counts them; images the 📐 Classify framing pass never reached gather at the end rather than in the middle. Shot type, then face similarity ↓ is the same grouping with the closest to your reference at the head of each run, so you walk down a kind and the ones to cut are waiting at its end. Like every sort here it only reorders: the filters still decide which images are shown, the counts do not move, and select-all and the ⟨ ⟩ arrows follow what is on screen. (Asked for by .samexit on Discord.)',
    to: '/datasets?section=images&panel=review',
  },
  {
    id: '2026-08-20-edit-a-custom-shot',
    date: '2026-08-20',
    title: 'Edit a custom shot instead of retyping it',
    blurb:
      'A shot card you wrote is a whole sentence — outfit, pose, setting, light — and until now the only way to change one word of it was to delete the card and type the other forty again. Worse, the card that came back was a different card: it landed at the end of the row, unselected, so a typo cost you your place in a selection you had spent minutes building. Every card you authored now has an ✏️ next to its ✕, in the ✨ Custom group and in the 📥 Imported one alike, so saving a card for good with ⇪ Keep no longer takes its pencil away. Press it and the words come back into the ✨ Custom shot box below, with the framing you picked; change what you want and Save puts the card back exactly where it was, still selected. Cancel leaves it untouched. Two things do not carry over, both on purpose: the ✓×N tally on the card, because those images were generated from the words you just replaced, and a name you wrote yourself in an imported catalog, which is kept exactly as you typed it while the auto-named cards follow their prompt. (Asked for by .samexit on Discord, twice: the second time to say that ⇪ Keep was hiding the button.)',
    to: '/datasets?section=add&panel=generate',
  },
  {
    id: '2026-08-19-video-burst-triage',
    date: '2026-08-19',
    title: 'Triage a video bank one keystroke per shot',
    blurb:
      'A two-hour rush becomes three hundred shots, and until now judging them meant three gestures each: click a tile, click ✓ or ✕, come back to the grid. The 🎬 Video bank has a new ⌨ Burst mode above the gallery. Turn it on and one tile carries a cursor — K keeps it, R rejects it, P puts it back to untriaged, S or → moves on without deciding, ← steps back. They are the same keys as the image bank\'s ▶ Review, so the reflex you already have works here. The cursor then jumps to the next shot you have NOT judged yet, which on a half-triaged bank is most of the speed; untick Auto-advance and it stays put so K then R corrects the same shot. It never wraps silently: when nothing untriaged is left ahead, the bar says how many are still behind you and Home goes back to the first. U undoes the last decision and moves the cursor onto that shot so you can see what it fixed, ten steps deep, always restoring what the shot actually was before — undoing a reject on a shot you had kept puts the keep back. The offer sits in the bar rather than in a toast, because at one keystroke a second a toast is replaced before it can be read. Your keys never wait for the network either: the tile flips at once and the decisions are sent behind you, one request at a time, with a run of identical verdicts going out as a single batch and a "saving N…" counter so a run that has ended is never mistaken for a run that is saved. Press ? for the full list, and nothing fires while you are typing in the search box or a threshold field.',
  },
  {
    id: '2026-08-19-video-temporal-coherence',
    date: '2026-08-19',
    title: 'Find the shots that are secretly two shots',
    blurb:
      'Shot detection cuts on a change big enough to see. The ones it misses are the soft ones — a dissolve, a match cut, a new angle inside the same room — and each one leaves behind a “shot” that is really two scenes. It is the worst kind of training example, because it teaches the model a transition nobody asked for, and you cannot catch it by scrolling: the thumbnail is one of the two halves and looks perfectly fine. The 🎬 Video bank now checks every shot for this by itself, at the end of 🔎 Find scenes, and it costs nothing at all — no decoding, no model, no GPU, no button. It compares a shot’s first frame to its last using vectors that pass already cached, so a bank you embedded weeks ago gets its reading by clicking 🔎 Find scenes again. Each shot gains a scene coherence number (1.00 means its ends are the same picture), and 🎚 Quality cuts gains a Scene coherence floor that flags anything below it as “Cut inside the shot” — then ✂ Split here does the repair. Empty by default, and the Guide is blunt about why: this is a ranking, not a verdict. Measured against shots of the same length, a cut at 0.80 catches about a third of the double shots and flags about one honest shot in seven, so use it to choose what to look at first. Long takes score lower whether or not anything was cut, which the panel says out loud.',
    to: '/video-bank',
  },
  {
    id: '2026-08-19-video-camera-motion',
    date: '2026-08-19',
    title: 'Sort your shots by what the camera did',
    blurb:
      'A video LoRA learns camera language along with everything else, and until now there was no way to see any of it: a bank of a thousand shots gave you no answer to “which of these are locked off” or “where are the handheld ones”. The 🎬 Video bank has a new 🎥 Camera pass. It tracks every frame of every shot and labels what the camera did — pan left, pan right, pan up, pan down, zoom in, zoom out, static shot, handheld shot — using the same words the video trainer itself uses, so a label here means the same thing there. Three more are ours: rolling, slideshow (a photograph panned across rather than filmed) and subject moves. The labels appear on each thumbnail and as a new 🎥 Camera row of filters above the gallery, which composes with the ⚑ flag chips — “shaky shots that also pan right” is one click each. Nothing is ever rejected: these are descriptions, not faults, because the wobble one person is filtering out is exactly what the next person is training on. If you do want to cut on it, 🎚 Quality cuts gains a Camera shake threshold, empty by default like the rest. The pass runs on the CPU at about fifteen times real time and needs only the video decode extra, which now installs OpenCV alongside PyAV — press Install on the video row in Setup if it shows a ✗. Honest limits, stated in the app too: a pivot and a slide look identical in a flat picture so both are called a pan, orbits are not detected at all, and a real pan across a wall or a horizon can read as a slideshow because it has no depth either.',
    to: '/video-bank',
  },
  {
    id: '2026-08-19-video-ai-check',
    date: '2026-08-19',
    title: 'See which shots may have been generated rather than filmed',
    blurb:
      'A scrape in 2026 brings back generated video mixed in with the real thing, and it is invisible at thumbnail size — a clean, well-lit, well-framed synthetic clip passes every other check in the bank. It is worth finding: published curation work reports that even under a tenth of a corpus being synthetic measurably degrades what a model learns from it. The 🎬 Video bank has a new 🤖 AI check: it looks at two contiguous seconds of each shot and measures how erratically the motion changes, because real footage is full of small irregularities and generated footage tends to be smoother than the world. Shots that come out suspiciously smooth get a “May be AI-generated” chip, against a new cut in 🎚 Quality cuts that is empty by default and applied as you move it with nothing rescanned. Read as a hint and never as a verdict: on re-compressed material — which anything scraped is — the best blind-evaluated detectors in the field are right about three times in four, and this one has never been measured against 2025-and-later generators. Nothing is ever rejected or deleted on it. Runs on the CPU, so it can check a bank while a training owns your card, and it needs the same ✨ Score interpreter the look score already uses.',
    to: '/video-bank',
  },
  {
    id: '2026-08-18-video-defect-sweep',
    date: '2026-08-18',
    title: 'Spot the footage that has already been through the mill',
    blurb:
      'A clip that was uploaded, re-encoded and re-uploaded three times carries damage a thumbnail cannot show you — and a LoRA learns it first, because it sits identically on every frame of every shot from that file. The 🎬 Video bank has a new 🩻 Defects pass: one sweep per source file finds frames that were simply delivered twice (what 24 fps material uploaded as 30 fps looks like), the macroblock grid showing through a hard squeeze, and edges that stay soft at FULL size. That last one is the important one — the sharpness floor measures a small analysis copy, where footage upscaled from 480p and the genuine 1080p are the same picture, so until now nothing in the app could tell them apart. Three new cuts in 🎚 Quality cuts, all empty by default, all applied as you move them with nothing rescanned. Each file card also shows how hard it was squeezed, in bits per pixel. Needs ffmpeg, which the video extra already installs.',
    to: '/video-bank',
  },
  {
    id: '2026-08-18-video-safe-zone',
    date: '2026-08-18',
    title: 'See the black bars and the subtitles before your LoRA learns them',
    blurb:
      'A subtitle sits in the same rectangle of every frame of every clip from the same source, so it is among the first things a LoRA learns to draw — and at thumbnail size you cannot see it. Neither are the letterbox bars on a vertical video somebody padded into 16:9. The 🎬 Video bank has a new 🔳 Safe zone pass: it looks at three frames of each shot, measures the flat bands on all four sides, reads any text that HOLDS STILL across those frames (a passing shop sign is scene content and is left alone), and works out how much of the frame a crop would leave you. Three new cuts in the thresholds panel — letterbox share, burned-in text share, usable frame floor — all empty by default, all applied as you move them with nothing rescanned. Reading text needs one small CPU package from Setup; without it the pass still measures the bands and says so rather than pretending it found none.',
    to: '/video-bank',
  },
  {
    id: '2026-08-18-canvas-one-grid-per-checkpoint',
    date: '2026-08-18',
    title: 'Pinned batches land as one grid per LoRA',
    blurb:
      'Generate a batch on the Canvas with several LoRAs selected and pin the results: each LoRA now gets its own grid on the board instead of everything fusing into one strip where the batches were indistinguishable. The epochs of one LoRA still share its grid — that side-by-side is the point — and prompts and separate launches keep their own grids too. Boards pinned before this keep drawing exactly what they drew.',
    to: '/canvas',
  },
  {
    id: '2026-08-18-the-app-gets-back-up-after-a-crash',
    date: '2026-08-18',
    title: 'The app gets itself back up after a crash',
    blurb:
      'Some deaths are not something the app can catch: an antivirus hook faulting inside an image library, or a native crash in one of the GPU extensions, kills the whole process outright — no error, no message, and until now it simply stayed down until you noticed and started it again. Launched from start.bat, it now comes back on its own, says in the console that it crashed rather than pretending nothing happened, and gives up after a few deaths in a row so an app that is broken at startup cannot loop forever. Set LDS_SUPERVISE=0 to run without it.',
  },
  {
    id: '2026-08-18-scrape-videos-into-a-video-bank',
    date: '2026-08-18',
    title: 'Fill a video bank straight from the web',
    blurb:
      'The scraper could already SEE videos — RedGifs, Erome, Picazor, TikTok, X, Civitai all list them — and the picker threw every one of them away, so the only way to triage a clip you found online was to download it by hand, drop it in a folder and point a bank at that folder. 🎬 Video bank now has its own 🕸 Scrape the web panel: paste a link, pick the clips, and they land in a bank ready to be cut into shots. Nothing is judged on the way in — length, motion and sharpness stay for the bank’s own passes, exactly like the image side. Send them to a brand new bank, which gets a folder of its own, or to any bank you already have — including one pointed at your own footage, where the clips are simply added to that folder. The picker names the folder before you start.',
    to: '/video-bank',
  },
  {
    id: '2026-08-18-video-look-score',
    date: '2026-08-18',
    title: 'Your shots now carry a look score — for free',
    blurb:
      'The video bank could tell you a shot was sharp, lit and moving; it could not tell you it was ugly. 🔎 Find scenes now also rates how each shot LOOKS, using the same LAION aesthetic model — and the same ~1–10 scale — the image Bank’s ✨ Score puts on a still. It costs nothing extra: the rating is read off the frame vectors that pass already caches, so no video is decoded twice and your GPU is never touched. Already embedded a bank? Click 🔎 Find scenes again and it rates the whole thing in seconds, without re-reading a single file. The new Aesthetic floor sits in 🎚 Quality cuts, empty by default — preview 4 against your own bank first; the published LAION cuts (4 casual, 4.75 strict) were set for filtering a web crawl, and real rushes sit well above them. A shot with no rating is never flagged.',
    to: '/video-bank',
  },
  {
    id: '2026-08-18-cleaning-keeps-the-zones-you-drew',
    date: '2026-08-18',
    title: 'The zones you draw by hand survive a clean',
    blurb:
      'When 🚩 Find watermarks missed a mark, you could draw the zone yourself — and a successful 🧽 Clean then deleted what you had drawn. Nothing said so, and it only cost you later: ↩ Restore original brings the watermarked picture back so you can clean it again, usually with the other engine, but the retry no longer had your zones and quietly fell back to the box the detector got wrong in the first place. Your zones now survive both steps, so a second attempt starts exactly where you left off. (The Bank already worked this way.)',
    to: '/datasets',
  },
  {
    id: '2026-08-18-face-analysis-can-use-the-gpu',
    date: '2026-08-18',
    title: 'Analyze faces can use your GPU — and says when it is working',
    blurb:
      'Two fixes to the same screen. 🎭 Analyze faces spent its first stretch fingerprinting every image before it told you anything, so a big dataset looked frozen and the banner fell back to claiming your GPU was busy and ComfyUI paused — neither of which was true. It now names itself and counts from the first second. And it can finally use the GPU: the Image bank’s face pass already could, this one was pinned to CPU. One setting now governs both (Settings ▸ face scoring device, `auto` by default), and a GPU run goes through the same exclusive window as every other GPU pass, so it can never compete with a training. Nothing changes unless you install `onnxruntime-gpu` into the face interpreter — the standard install ships the CPU build and stays on CPU, exactly as before.',
    to: '/datasets',
  },
  {
    id: '2026-08-18-auto-triage-says-why-its-empty',
    date: '2026-08-18',
    title: 'Auto-triage stops vanishing without a word',
    blurb:
      'The 🎯 Auto-triage bar used to disappear entirely whenever it had nothing to do — and four completely different situations looked identical: you had never run 🎭 Analyze faces, the pass could not score any of your images, you had already decided every one of them, or a decision filter was simply hiding the undecided ones. It now stays put and tells you which of the four it is, including how many scored images your current filter is hiding, so "nothing happens" is never left for you to guess at.',
    to: '/datasets',
  },
  {
    id: '2026-08-18-analyze-faces-in-wider-shots',
    date: '2026-08-18',
    title: 'Face scores for your full-body and bust shots, not just the close-ups',
    blurb:
      '🎭 Analyze faces used to skip almost every wide shot: it asked the head to fill 6% of the frame, which describes your camera rather than the face — the same head passed on a small photo and failed on a big one. It now judges the head in actual pixels, and when a head is small in frame it takes a second look zoomed in on it, at the photo\'s own resolution, instead of at the shrunk-down copy the detector normally sees. Full-body and bust shots get a real score, so 🎯 Auto-triage and the “Face similarity” sort finally cover them. True profiles are still left unscored on purpose — a turned head can\'t be compared honestly, so it stays your call. (Reported by .samexit on Discord.)',
    to: '/datasets',
  },
  {
    id: '2026-08-18-repair-with-a-brush',
    date: '2026-08-18',
    title: 'Paint over what should go, instead of boxing it',
    blurb:
      'A rectangle is the wrong shape for a necklace, a pair of glasses or a bra strap — it hands the model a square full of face it was never asked to touch. ✦ Repair now has a 🖌 Brush next to its ▭ Box: paint over the thing, say what should be there, and the whole picture goes to Klein with your painted mask, so it reconstructs while actually seeing the face around it. The box is still there and still the default — it is quicker, and better for a mark in a corner. Everything outside what you painted keeps its original bytes, exactly as before. (Contributed by OneCodingDude on GitHub.)',
    to: '/datasets',
  },
  {
    id: '2026-08-17-watermark-review-on-a-phone',
    date: '2026-08-17',
    title: 'Review watermarks from your phone without squinting',
    blurb:
      'Opened on a phone, the watermark review gave the photo about a third of the screen and spent the rest on controls — including a model picker and a permanent text field — so the one thing you were there to judge was the smallest thing on screen. The picture now gets the screen: 🧽 Clean, ✓ Not a watermark, ✕ Reject and the arrows stay put, and the setup controls (zone editor, crop-or-repaint, engine, model) fold behind one “Zones & engine” button. Nothing moves on a desktop, and anything explaining why a button is greyed out stays visible at every size.',
    to: '/datasets',
  },
  {
    id: '2026-08-17-undo-a-repair',
    date: '2026-08-17',
    title: 'Undo a repair, and try another description',
    blurb:
      'An inpaint is a dice roll, so the normal way to use ✦ Repair is: look at it, decide it is not right, change the sentence, go again. That was expensive — each attempt overwrote the image with no way back. Now the dialog stays open after a repair so you can repair again straight away, and ↩ Undo puts back the picture from just before it. One step deep, and it deliberately never touches the original kept for ↩ Undo cleaning, so undoing a repair cannot throw away a watermark clean you made earlier. (Suggested by a user on Discord the day ✦ Repair shipped.)',
    to: '/canvas',
  },
  {
    id: '2026-08-17-repair-a-generated-image',
    date: '2026-08-17',
    title: 'Fix one detail of a render instead of regenerating it',
    blurb:
      'A stray finger, an object you did not ask for — until now that meant throwing away the picture you liked and rolling the dice again, because the only prompted lane re-renders everything and gives you a different image. Open a generated image full size (on the Canvas, or from a checkpoint gallery) and press ✦ Repair next to ⬇ and ✨: draw the zone, say what should be there, and only that zone is repainted. Everything outside it comes back byte-identical, and your picture is preserved before anything is written, so a repair that fails costs you nothing. (Asked for by .samexit on Discord.)',
    to: '/canvas',
  },
  {
    id: '2026-08-17-repair-a-detail-free-prompt',
    date: '2026-08-17',
    title: 'Repaint one detail — and only that detail',
    blurb:
      'Until now the app had two halves of this and neither was the whole thing. 🧽 Clean repaints exactly the box you draw and leaves every pixel outside it byte-identical, but its instruction was frozen on watermark reconstruction. ✦ Edit takes any instruction but re-renders the whole image, drifting outside the part you cared about. ✦ Repair is the first lane with both: open an image, press ✦ Repair in the action bar, draw the zone, type what should be there ("remove the necklace"), and only that zone is repainted — the rest comes back to the byte. It stamps no watermark verdict, refuses an empty description rather than guessing, and preserves your original before writing anything, so a failed repair costs you nothing. (Asked for independently by mr.arrow and .samexit on Discord.)',
    to: '/datasets',
  },
  {
    id: '2026-08-17-vast-do-not-rent-this-machine-again',
    date: '2026-08-17',
    title: 'Stopping a cloud run can now blacklist the machine',
    blurb:
      'The app already refuses to re-rent a vast host when it can see the failure — a boot that never completes, a pod that stops progressing, a checkpoint it cannot serve. What it cannot see is a machine that boots perfectly and then simply trains at half speed: no failure happens, so nothing gets banned and the picker can rent it straight back. Stopping a run from the Runs page now offers "Do not rent this machine again", which puts that host on the same blacklist for the same few days. It is a tick box, not a rule: a stop can just as easily mean you changed your mind, and that says nothing about the machine. (Asked for by mr.arrow on Discord.)',
    to: '/cloud',
  },
  {
    id: '2026-08-17-canvas-bulk-undeploy',
    date: '2026-08-17',
    title: 'Undeploy a pile of LoRAs in one go',
    blurb:
      'Taking LoRAs back out of ComfyUI was a one-at-a-time errand buried in a checkpoint popover, and nothing anywhere told you how many were deployed. ⏏ Undeploy… at the top of the Canvas now opens the whole list — every LoRA the app has put into ComfyUI, across all your datasets and families, grouped by dataset. Tick what goes, press once, done; Select all is there for the clear-out. Only what the app deployed is listed, so a LoRA you downloaded into the same folder is never shown and never touched. Your training saves are kept — anything you undeploy can be deployed again from its checkpoint — and the removed copies go to the trash. The result is reported in three parts rather than a flat "done": removed, already gone, and refused (each one named).',
    to: '/canvas',
  },
  {
    id: '2026-08-17-caption-draw-from-a-bank',
    date: '2026-08-17',
    title: 'Draw a test prompt from a bank, not just a dataset',
    blurb:
      'The 🎲 Caption shortcut — the one that fills a test prompt with a real caption instead of something you invent — could only read datasets. But a bank is captioned by the 🏷️ Caption pass long before anything is promoted, so the biggest pile of real captions on your machine was the one it could not reach. The picker now lists your banks alongside your datasets, in their own section (a bank and the dataset it promotes into often share a name, so they are never mixed into one list). Your existing locked choice is untouched.',
    to: '/canvas',
  },
  {
    id: '2026-08-16-bank-crop-and-upscale',
    date: '2026-08-16',
    title: 'Crop and upscale without leaving the Bank',
    blurb:
      'Reframing or upscaling a shot used to mean taking it out of the Bank: promote it into a dataset, edit it there, export the result into a NEW bank, and start curating all over again. Both now happen in the Bank itself. ✂ Crop is in ▶ Review — press C, drag the box, done; it decides nothing, so you can frame an image and then judge it. Nothing is resampled, unlike a dataset crop: a Bank sits upstream of the training resolution, so the cut keeps its pixels and the dataset still decides the size when it imports. ✨ Upscale & improve is a proper pass on the new ✂ Edits panel, with a scope, a progress bar and ⏹ Stop, running on Klein or SeedVR2. Your own files are never touched: both edits land in a copy the app keeps, ↩ Revert throws it away, and every measurement taken from the old pixels is cleared so the analysis passes re-read the image you are actually keeping. (Asked for by nofaceman on Discord, backed by mr.arrow.)',
    to: '/bank',
  },
  {
    id: '2026-08-13-mark-a-watermark-the-scan-missed',
    date: '2026-08-13',
    title: 'Mark a watermark the scan missed, yourself',
    blurb:
      'The watermark detector is very good, but it is a classifier — some marks, especially the ones stock sites tile across a whole photo, score under any threshold you set. Until now that was a dead end: the mask editor only opened on images the scan had already flagged. Now you can open it on any image you are looking at, in a Dataset or a Bank, and the zones you draw become the flag — 🧽 Clean then repaints exactly what you drew. Changed your mind about an image you had ruled a false positive? Drawing on it takes that back too.',
    to: '/bank',
  },
  {
    id: '2026-08-11-bank-rail-status-curate',
    date: '2026-08-11',
    title: 'The Bank puts its main gestures where your eyes are',
    blurb:
      'The Status split (All / Undecided / Kept / Rejected) now leads the filter rail as four large colour-coded buttons carrying live counts, the Curate tools (Pick diverse, Balanced pick, Similar, Find by text) wear the size their role deserves, the measured filters under “More filters” start unfolded, and the ✨ Clean chip finally says how many images it holds. The Quality row also tells you when only part of the bank has been scanned — with a one-click way to scan the rest.',
    to: '/bank',
  },
  {
    id: '2026-08-11-single-instance-guard',
    date: '2026-08-11',
    title: 'Launching the app twice can no longer split it in two',
    blurb:
      'Starting the app while it was already running used to quietly boot a second server on the next port, sharing the same database — jobs launched in one were invisible in the other, with no bar and “pass is running” refusals that pointed at nothing. A second launch now says the app is already running, points at its address, and steps aside. Separate installs and test copies with their own data folder are untouched, and running two on the same data on purpose stays possible (LDS_ALLOW_SECOND_INSTANCE=1).',
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
  '/datasets', '/bank', '/video-bank', '/studio', '/cloud', '/canvas', '/gallery',
  '/guide', '/help', '/setup',
]);

const SETTINGS_IDS = new Set(SETTINGS_SECTIONS.map((s) => s.id));

// Split a target string into { path, section, panel }. Returns null for
// anything that is not an in-app absolute path.
export function parseTarget(to) {
  if (typeof to !== 'string' || !to.startsWith('/')) return null;
  const [path, query = ''] = to.split('?');
  const params = new URLSearchParams(query);
  return {
    path,
    section: params.get('section'),
    panel: params.get('panel'),
    step: params.get('step'),
  };
}

// Is `to` a target the app can actually navigate to? Validated against the LIVE
// settings + workspace registries so a renamed section is caught by the tests.
export function isValidTarget(to) {
  const t = parseTarget(to);
  if (!t) return false;
  const { path, section, panel, step } = t;

  // /setup with an optional ?step=<wizard step id> deep-link (the Settings
  // Overview capability rows use it to open the screen that installs them).
  if (path === '/setup') {
    if (section || panel) return false;
    return step === null || SETUP_DEEP_LINK_STEPS.includes(step);
  }
  if (step) return false; // ?step= is meaningless anywhere but the wizard

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
