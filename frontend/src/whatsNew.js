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
//  • Keep the list tidy: tail entries older than a couple of months can be
//    pruned once everyone has cycled through an update or two.
// =====================================================================
import { SETTINGS_SECTIONS } from './components/settings/registry.js';
import { WORKSPACE_SECTIONS } from './components/dataset/workspaceSections.js';
import { SETUP_DEEP_LINK_STEPS } from './hooks/useSetupSteps.js';

// Newest first. Prepend new waves at the top.
export const WHATS_NEW = [
  {
    id: '2026-08-09-canvas-external-lora-links',
    date: '2026-08-09',
    title: 'See which images an external LoRA actually touched',
    blurb:
      'A permanent cyan line now joins each 🔌 plugin node to every board image generated with it, and an image’s facts panel lists its External LoRAs on their own row instead of filing them under always-on.',
    to: '/canvas',
  },
  {
    id: '2026-08-09-preview-prompts-from-captions',
    date: '2026-08-09',
    title: 'Preview images that finally show YOUR subject',
    blurb:
      'The preview prompts a run renders every few hundred steps used to be generic defaults that describe nobody — so the images you judge an expensive run by showed a stranger. A new 🎲 Use dataset captions button under Preview prompts fills the field with up to five real captions drawn at random from this dataset’s kept images. Click it again for a different draw.',
    to: '/datasets?section=training&panel=advanced',
  },
  {
    id: '2026-08-09-canvas-phone-toolbar',
    date: '2026-08-09',
    title: 'The LoRA Canvas gives the board back to your phone',
    blurb:
      'On a phone the board’s bottom controls took a quarter of the screen, and tapping ☝ Gestures buried the board under its own instructions with no way to put them away. The row is icons-only below a tablet — two rows instead of five, every button still thumb-sized — and the gesture help is now a sheet that floats over the board and closes with its ×. Nothing changes on a desktop.',
    to: '/canvas',
  },
  {
    id: '2026-08-09-canvas-external-loras',
    date: '2026-08-09',
    title: 'Pin any LoRA onto the Canvas, even one you never trained here',
    blurb:
      'Pin any LoRA from your ComfyUI folder onto the LoRA Canvas as a 🔌 plugin node and stack it on your generations, with its own strength. It stacks on a run anchored by a checkpoint trained here — there is no solo generation from an external LoRA alone.',
    to: '/canvas',
  },
  {
    id: '2026-08-08-the-canvas-controls-live-on-the-board',
    date: '2026-08-08',
    title: 'The Canvas gives the screen back to the board',
    blurb:
      'Everything that steers the LoRA Canvas — the zoom row, Fit, Tidy up, Generate, the colour key, the gestures sheet, the dataset filter and the banner announcing finished images — used to be stacked above the board. On a phone that chrome cost most of the screen before a single card was drawn, which is why the board opened tiny and pinned under a wall of buttons. Those controls now float ON the board: what it is showing sits along the top, what you do to it sits along the bottom within thumb reach, and the board itself takes back the space they were using. Nothing was removed and nothing moved to another page — the same controls, on the surface they act on.',
    to: '/canvas',
  },
  {
    id: '2026-08-08-canvas-filter-bar',
    date: '2026-08-08',
    title: 'The canvas filters stopped eating half the screen',
    blurb:
      'The Datasets panel was a fold-out card: unfolded on a library of fourteen datasets it stood 389 px tall on a 720-px screen — 54 % of the window, directly above the board, for anyone who had ever left it open. It is now a single row of chips about 40 px tall. Datasets, Models and Status each open a small menu with the same checkboxes (the dataset menu gained a search of its own, which the three-column list never had), Pinned images and Reset stay in the row, and the run search keeps its full-size box. Every chip shows its count and lights up while it is filtering, so nothing can narrow your board without saying so — and the menus now open above the board instead of under a pinned image.',
    to: '/canvas',
  },
  {
    id: '2026-08-08-canvas-zoom-labels',
    date: '2026-08-08',
    title: 'Run cards stay readable when you zoom the board out',
    blurb:
      'A board holding a dozen datasets is read at 30-40 % zoom, and at that scale a run card\'s title renders at about four pixels: the canvas was showing you everything and telling you nothing, so finding a run meant zooming in on each one in turn. Below 55 % every card now carries its run number at a constant, readable size, and below 30 % it carries the dataset name too — because the lane headings have gone unreadable by then as well.',
    to: '/canvas',
  },
  {
    id: '2026-08-08-canvas-layout-presets',
    date: '2026-08-08',
    title: 'Keep a board arrangement instead of losing it to ✦ Tidy up',
    blurb:
      'Laying two datasets\' renders out side by side to judge a likeness takes twenty minutes, and until now the board only ever held ONE arrangement: the moment you needed it for something else, your only options were to leave it there forever or throw it away. 💾 Layouts in the board toolbar saves where every run card and every pinned picture sits — closed pictures included — under a name, and puts it back whenever you want. A run deleted since is simply not restored, and the app says how many, rather than leaving you to hunt for the card that is missing.',
    to: '/canvas',
  },
  {
    id: '2026-08-08-canvas-export-png',
    date: '2026-08-08',
    title: 'Save the whole board as a PNG',
    blurb:
      '📷 PNG in the board toolbar writes the entire canvas to one image file: every pinned picture at full size, every run card with its checkpoints, and the lines that join them. Useful for a comparison you want to keep, post, or look at next to something else. It is a redraw of the board, not a screenshot, so the buttons and badges are not in it — and a picture whose file has been cleaned off the disk comes out as a labelled placeholder instead of silently missing.',
    to: '/canvas',
  },
  {
    id: '2026-08-08-canvas-delete-image',
    date: '2026-08-08',
    title: 'Bin a bad render from the board itself',
    blurb:
      'The board is where you actually decide a render is a failure — and deleting it meant closing the node, opening the run, finding the checkpoint, opening its gallery, entering Select mode and finding the same picture again. Pinned images now carry a 🗑 next to their ✕. Press it once to arm it, again to delete: ✕ still only takes the picture off the board and remembers where it was, 🗑 deletes the image itself, through the same route (and the same recoverable-or-not setting) the gallery uses.',
    to: '/canvas',
  },
  {
    id: '2026-08-08-lora-strength-up-to-5',
    date: '2026-08-08',
    title: 'LoRA strength and blend weights now go up to 5',
    blurb:
      'The weight slider of a 🧬 Blend stopped at 2 and the strength sweep at 4 — both were comfort limits, not technical ones, and pushing an under-trained LoRA or an overwhelming style meant leaving the app for a hand-built workflow. The ceiling is 5 everywhere now: the blend slider, the strength chips behind « + », and the server that validates them. The weight next to each slider is also typeable, so getting to 3.35 no longer means eighty small drags.',
    to: '/canvas',
  },
  {
    id: '2026-08-08-launch-anyway-actually-asks',
    date: '2026-08-08',
    title: 'A second cloud run on one dataset now actually asks you',
    blurb:
      'Launching a second cloud training on a dataset that already had one was supposed to ask "launch anyway?" and go ahead if you said yes. The question never reached you: the refusal arrived as an error toast printing the question at you, with no way to answer it, so the second run was simply impossible however many times you tried. The confirmation now appears and the run starts when you accept. Nothing about what is allowed has changed — the guard, your fleet limit and your budget behave exactly as before; the answer you gave it was being thrown away. Reported by Arrow (Discord).',
  },
  {
    id: '2026-08-08-start-comfyui-from-the-banner',
    date: '2026-08-08',
    title: 'Start ComfyUI from the banner that tells you it is down',
    blurb:
      'When LDS cannot reach ComfyUI, the banner explaining it offered one button: "I restarted ComfyUI — clear it", which only lets you say someone else fixed it. On an install whose ComfyUI this app can launch, there is now a ▶ Start ComfyUI button right there, and it comes first: it is the one that ends the outage rather than declaring it over. The banner then clears itself once ComfyUI actually answers, so a start that spawns and dies cannot be mistaken for a fix, and your paused job resumes on its own. The button is deliberately absent when your ComfyUI is not ours to launch — ComfyUI Desktop, your own launcher script, or one running on another machine — because on the one screen whose job is to unblock you, a button that fails costs more than no button at all.',
  },
  {
    id: '2026-08-08-klein-stops-switching-itself-off',
    date: '2026-08-08',
    title: 'Picking a Klein model no longer switches Klein off',
    blurb: 'Choosing a model from the workspace dropdown stored its name without the folder it lives in — and the check that guards that setting only recognised the full path. So the app wrote a value its own guard then called missing, the engine went dark, and the message said no file was missing or broken. Both spellings are now accepted, so nothing you picked was ever wrong. And an unresolvable model no longer takes the whole engine hostage while usable Klein builds sit on your disk: the card asks you to pick one instead. What has not changed is the promise that made that guard exist — a run never quietly loads a file other than the one shown.',
    to: '/settings/engines',
  },
  {
    id: '2026-08-08-krea-ships-identity-first',
    date: '2026-08-08',
    title: 'Krea 2 Edit now ships tuned for likeness, not for variety',
    blurb: 'Its two identity dials shipped unpaired — reference grounding at 512 with a reference pull of 0.25, a quarter of what the calibrated 512 profile used. A benchmark scoring generated faces against the reference measured the 1024 / 4.0 pair ahead by a clear margin on bust shots, and it got there while pulling LESS hard on the reference, so the likeness is not bought by recopying the pose you asked to change. Both dials, and sampler steps, now ship at those values. If you had never touched them, your next generations will resemble your reference more; if you prefer the looser, more varied restaging, set grounding back to 512 and the pull to 1.0. Anything you had already changed yourself is left exactly as you set it.',
    to: '/settings/engines',
  },
  {
    id: '2026-08-08-one-model-setting-wherever-you-change-it',
    date: '2026-08-08',
    title: 'Change the model where you are working, not in Settings',
    blurb:
      'The Krea 2 base could only be set in Settings, so the one thing you reach for when a whole run looks wrong meant leaving the screen you were judging on. It is now in the Krea 2 Edit panel of the work screen, using the same searchable list and writing the same value Settings shows. The Klein model works the same way, and there is now exactly one of each: change it from a dataset screen or from Settings and it is the same setting either way, applying to every run from then on rather than to the batch in front of you. Previously the Klein model was remembered per dataset, so two datasets could disagree with Settings about which model would actually run.',
  },
  {
    id: '2026-08-08-every-krea-build-you-own-is-offered',
    date: '2026-08-08',
    title: 'Every Krea build on your disk is offered, warning and all',
    blurb:
      'A checkpoint measured to render noise under the identity-edit LoRA used to be removed from every Krea list, so a file sitting in your own Krea folder was simply absent, with nothing on screen saying it existed or why it had gone. Choosing is yours: those builds are listed again and you can select them. What the app still will not do is pick one FOR you when nothing is pinned, because a base chosen in silence has already sent a run onto the wrong model without anyone noticing until the output was wrong. It is only elected automatically when it is the only Krea build you have, which beats refusing to run at all.',
  },
  {
    id: '2026-08-08-phone-lightbox-image-first',
    date: '2026-08-08',
    title: 'On a phone, the image you opened finally gets the screen',
    blurb:
      'Opening a dataset image on a phone showed you the buttons, not the picture: '
      + 'Crop, Mirror, two Rotates, two Improves and the Klein note stacked into six '
      + 'full-width rows, leaving the photo about 100 px tall — and side-by-side '
      + 'comparison, the one place where size is the whole point, gave each pane even '
      + 'less. Below a phone-sized width the lightbox now flips: the image (or both '
      + 'comparison panes) takes the screen, and every action — compare, crop, mirror, '
      + 'rotate, improve, upscale, the Klein instruction and its editor — moves behind '
      + 'one ☰ Actions button that opens a panel over the bottom of the picture. '
      + 'Measured at 400 px: a single image goes from 96 to 538 px tall with the Klein '
      + 'editor open, and each comparison pane from 144 to 354 px. Esc closes the panel '
      + 'first and the image second, and ⟨ / ⟩ still walk the grid. On a desktop nothing '
      + 'changes — the side rail was already doing this job with space the image cannot use.',
    to: '/datasets?section=images',
  },
  {
    id: '2026-08-08-a-just-deployed-model-generates-right-away',
    date: '2026-08-08',
    title: 'A model you just deployed works right away, not a minute later',
    blurb:
      'Deploy a checkpoint, hit Generate, and the job died: "Your ComfyUI does not offer a model file this workflow requires", followed by advice to go and check your ComfyUI address — which was right all along. The app reads the list of models ComfyUI offers once a minute and reuses it, so a model deployed seconds ago was judged against a list drawn up before it existed, and the refusal it produced is final: the job was never retried. Now, before a job is refused over a missing model, the app asks ComfyUI again — once, on the spot, ignoring what it had cached — and the model it finds is used immediately, spelled exactly the way that ComfyUI spells it. When the file really is absent the job still stops before queuing, but the explanation has stopped guessing: it now says the list was just re-read, so a fresh deploy is not what you are looking at, which leaves a second ComfyUI install as the thing to check.',
  },
  {
    id: '2026-08-08-krea-tuning-same-four-dials-both-screens',
    date: '2026-08-08',
    title: 'Every Krea 2 Edit dial, on both screens that talk about it',
    blurb:
      'Krea\'s four calibration dials used to be split three ways: reference grounding was a slider in Settings but only a read-out in the workspace panel, sampler steps existed in Settings alone, and reference pull and identity LoRA strength existed in the panel alone. "Where do I change this?" had a different answer per dial. Now all four — reference grounding, sampler steps, reference pull, identity LoRA strength — are sliders in BOTH places, so you can retune from wherever you happen to be judging the result. There is still exactly one value per dial: every control writes the same global setting, so nothing can disagree with anything. Each slider says in words what its current number means, and offers Reset to default the moment you leave it. The two file-path fields stay in Settings alone — you fill those once at install, not while looking at an image.',
    to: '/settings/engines',
  },
  {
    id: '2026-08-07-compare-any-image-with-the-reference',
    date: '2026-08-07',
    title: 'Check any image against your reference photo, side by side',
    blurb:
      'The question you ask of a generated shot is "is this still the same person?" — and until now the only way to answer it was to remember the reference photo from another panel. Open any image in the dataset full screen and press ◐ Compare with reference: the reference and the image sit side by side, each named, each filling its own pane. It works on every image, generated or imported, not just on improve candidates — and on an improved one both buttons are there, so you can flip between "is it sharper?" and "is it still them?". The two never open at once, and neither follows you: press ⟩ and the next picture starts clean. Each pane fits its own image and says so — a square head crop next to a full-body plan has no shared scale to promise, unlike the comparison against the original.',
    to: '/datasets?section=images',
  },
  {
    id: '2026-08-07-dataset-lightbox-next-previous',
    date: '2026-08-07',
    title: 'Walk a dataset image by image, without closing the picture each time',
    blurb:
      'Reviewing image 41 of 340 full screen used to mean closing it, hunting tile 42 on the wall, and opening that — for every single image. The inspection view now has ⟨ and ⟩ on the edges of the picture, ← and → on the keyboard, and a 12 / 340 badge telling you where you are. It walks what the grid is SHOWING: chip it down to "awaiting ✓/✕", sort by face similarity, and the arrows follow that list in that order, crossing pages as they go — close the view and you are on the page holding the image you were looking at. The ends do not wrap: on the first image ⟨ goes grey and says so, because a loop that silently restarts makes "have I seen everything?" unanswerable. Zoom, an open comparison pane and a running improvement stay behind with the image they belong to, so the pane labelled "original" is always the parent of the picture in front of you. And since moving is only reading, the arrows keep working while a generation or a captioning pass holds the dataset.',
    to: '/datasets?section=images',
  },
  {
    id: '2026-08-07-lora-preset-applies-itself',
    date: '2026-08-07',
    title: 'Your LoRA preset can now apply itself, every run',
    blurb:
      'Building a generation-LoRA preset in Settings was only half the job: the tuning panel in the workspace opened on "None" every single time you came back to a dataset, so the preset applied only if you remembered to re-pick it — and a run that forgot showed no LoRA at all in the finished image\'s metadata, which looks exactly like the app ignoring your settings. Each local engine now has its own "Preset selected by default" in Settings → Image engines: pick one and every run starts there. It stays a starting point, not a lock — you can still choose another preset, or None, for a single run without touching the setting. Shipped as None, so nothing changes until you choose. Preset mechanism by @waltm (Discord).',
    to: '/settings/engines',
  },
  {
    id: '2026-08-07-lora-row-that-would-be-ignored-says-so',
    date: '2026-08-07',
    title: 'A LoRA line that would be ignored now tells you, in the editor',
    blurb:
      'Both local engines already load one LoRA outside your presets — Klein its consistency LoRA, Krea 2 Edit its identity edit LoRA — and a preset row naming that same file is deliberately dropped: chaining it twice adds both strengths together, well past what the file was trained for, and the result goes blocky. That drop happened in complete silence, one line in a server log nobody reads. Write such a row now and the preset editor says so on that exact line, explains why, and points at the strength setting to change instead. The check spells the path the same way the server does, so a slash the other way round or a different capitalisation cannot slip past it.',
    to: '/settings/engines',
  },
  {
    id: '2026-08-07-krea-base-model-is-named',
    date: '2026-08-07',
    title: 'Settings now names the Krea base model your runs actually load',
    blurb:
      'The "Base model file" field said "auto — finds a Krea 2 Turbo/Raw build" and stopped there. If your ComfyUI holds more than one candidate — the official Turbo build next to a community finetune whose filename also reads as turbo — the tie-break picked one and never told you which, so the only way to find out was to open a finished PNG and read its metadata. Every judgement about quality in between was about a model you had not chosen. The field now names the file the next run will load — and, when a filename you pinned yourself was not found under any krea folder, says so and names what is being loaded instead.',
    to: '/settings/engines',
  },
  {
    id: '2026-08-08-improve-instruction-editable-in-place',
    date: '2026-08-08',
    title: 'Fix the improve instruction where it goes wrong, not in Settings',
    blurb:
      'The note under ✨ Upscale & improve already told you what the pass was about to ask Klein for — "add detailed texture, add sharp details…" — and then sent you to Settings to change it. Now you can change it right there: ✎ Edit this instruction here opens the box under the button, in the lightbox and in the bulk toolbar, already filled with the exact text in force. Rewrite it for a drawing, or untick it and let the pass upscale with no instruction at all; both take effect on your next improve, with nothing to save. It edits the app-wide setting — the same value Settings shows, applying to every dataset — and the panel says so before you touch it. Reset to default appears only once you have actually overridden something, and puts you back on the shipped text rather than on a frozen copy of it, so later improvements to that text still reach you.',
    to: '/datasets?section=images',
  },
  {
    id: '2026-08-07-bank-stop-answers-the-click',
    date: '2026-08-07',
    title: 'Stop on a Bank pass answers you the moment you press it',
    blurb:
      'Press Stop while ✨ Score is writing thirty-six thousand rows and nothing appeared to happen for about three seconds — so everyone pressed it again, several times. The click was always registered instantly; it was the banner around it that took that long to refresh, and the button looked exactly the same before and after. It now changes to "Stopping…" the instant you press it and stops taking clicks, without waiting for anything from the server. It also says what it is waiting for, in the words of the step actually running — "finishing the current batch of 200 rows, then saving" is why the counter keeps moving for a moment after you press. And it tells you the price BEFORE you press: while scores are being written, everything already written stays and only the style grouping has to be redone in full, while during the style grouping itself Stop costs nothing because that step is written whole either way.',
  },
  {
    id: '2026-08-07-passes-survive-a-locked-cache',
    date: '2026-08-07',
    title: 'A pass no longer dies because something else touched its cache file',
    blurb:
      'A ✨ Score over 37,000 images stopped at image 1849 with "Access denied" on its own cache — not a permissions problem, but an antivirus holding the freshly written file for a fraction of a second while the pass tried to swap it into place. On Windows that is enough to refuse the swap, and the work the pass had already computed and written was thrown away with it. Now the swap waits and retries for a few seconds, and if something really is holding the file the message says so instead of sending you into folder permissions. Better still, nothing is lost either way: work a previous run finished but could not file away is picked up automatically when the pass next starts, so a run interrupted by a lock, a crash or a power cut resumes from where it got to. Recovered work is checked before it is trusted — a half-written file is refused and removed rather than quietly becoming your cache. Applies to ✨ Score, faces, the semantic index and the video search vectors.',
  },
  {
    id: '2026-08-07-image-size-budget-is-a-setting',
    date: '2026-08-07',
    title: 'Big panoramas and camera masters import now — and the size limit is yours to set',
    blurb:
      'A 10418×2100 panorama used to be refused with "reduce the image before import", which was the only advice possible while the limit was welded into the code. The limit is now a setting, and its default has moved from 16 Mi-pixels / 8192 px per side to 64 Mi-pixels / 16384 px — room for a 61 MP camera master or a stitched panorama without a second thought. Settings ▸ Captioning & quality ▸ Image size budget lets you raise it further, or switch it off entirely; the choices are labelled in the memory each one commits (a decoded pixel costs 3 bytes, and an edit can hold a second copy at once), and "No limit" says plainly what it disarms. One number governs the whole app — import, ZIP and scrape ingest, Bank scan and thumbnails, edits, ComfyUI staging and vision captioning — so an image you can import is an image you can look at. And when something is refused, the message now tells you where to change the budget instead of just telling you to shrink the file.',
    to: '/settings/captioning',
  },
  {
    id: '2026-08-07-dataset-stays-readable-while-it-works',
    date: '2026-08-07',
    title: 'A dataset you can still look through while it works',
    blurb:
      'Starting a generation, a captioning run or a watermark scan used to freeze the whole grid: you could not open an image full screen, zoom into a face, or even tick a box, on a pass that might run for an hour. Inspecting and selecting are reads — they change nothing — so they now stay available throughout. Editing does not: crop, mirror, rotate, keep/reject, captions and delete still wait for the pass to finish, because a second writer would race it. What changed there is that they stop being silently grey — each one now says which pass is holding it, where that pass has got to and how long it still needs ("⚡ Variation generation is running on this dataset — 12 / 64"), and a line above the grid states the rule once: edits wait, looking and ticking do not.',
    to: '/datasets?section=images&panel=review',
  },
  {
    id: '2026-08-07-pick-the-model-file-you-actually-have',
    date: '2026-08-07',
    title: 'Choose your Klein and Krea model files from a list instead of typing a filename',
    blurb:
      'The Klein model-file fields and the Krea 2 Edit base model and identity LoRA were blank boxes you had to fill from memory — one typo and the engine quietly used a different file. Each is now a searchable list of the model files actually found in your ComfyUI (extra_model_paths.yaml roots included), with a ↻ to rescan after you drop a new one in, and a plain sentence naming the folder when nothing is there. The Krea base list shows only what the app would really elect, so a checkpoint it refuses is never offered. Typing still works for a file that is not on disk yet, or an absolute path from anywhere.',
    to: '/settings/engines',
  },
  {
    id: '2026-08-07-a-pinned-model-that-is-missing-stops-the-run',
    date: '2026-08-07',
    title: 'A model file you chose that is no longer there now stops the engine instead of being swapped',
    blurb:
      'If you pinned a Klein or Krea model file and that file is gone or renamed, the app used to fall back to picking one for you and write a line in a log nobody reads — so the graph loaded a file other than the one on screen, and you only found out from the images. Now the pinned name stays in the field, marked "not found", the engine refuses to start and says which file it is looking for. Clearing the field is the explicit way back to automatic detection. The one exception is the shipped identity/consistency LoRA name, which nobody typed: at its default it still recovers a renamed download on its own.',
    to: '/settings/engines',
  },
  {
    id: '2026-08-07-krea-likeness-dials-reachable',
    date: '2026-08-07',
    title: 'When Krea 2 Edit does not look enough like your reference, you can now fix it',
    blurb:
      'Krea has two dials that decide how strongly it holds on to your reference photo — the reference pull and the identity LoRA\'s own strength — and neither had a control anywhere in the app: the only way to move them was to hand-edit config.json. They are now sliders in the "🧬 Krea 2 Edit tuning" panel of Generate variations, right where you judge the result. Reference pull is the one to reach for when the face is too loose (it used to ship paired with reference grounding, so raising grounding alone left you on an uncalibrated mix), and the identity LoRA strength has 50% of headroom above its shipped 1.0. Both write your global Krea settings — the panel says so, and each has a one-click way back to the shipped value.',
    to: '/datasets?section=add&panel=generate',
  },
  {
    id: '2026-08-07-passes-say-how-long-is-left',
    date: '2026-08-07',
    title: 'Every pass now tells you how much longer it needs',
    blurb:
      '"12939 / 37800" told you where a pass was, never whether that meant twenty minutes or four hours — so the only way to find out was to sit and watch. Scan, ✨ Score, faces, framing, medium, captioning, watermark, crops & variants, the semantic index and the video passes now add "about 2 hours left" beside their counter, and a refusal to start a second pass tells you how long the first one still needs. The number is measured over the last minute of real work, not averaged since the pass began: a ✨ Score resuming from cache swallows twenty thousand rows in two seconds and an average would have promised "a few seconds" for the next hour and a half. It stays quiet until it can hold still — you get "estimating time left…" rather than a figure that changes every poll — it says "in this step" once a pass has moved on to a different kind of work, and a step with nothing to count (grouping styles, comparing shots) gets no estimate at all rather than an invented one.',
  },
  {
    id: '2026-08-07-reference-edit-survives-a-restart',
    date: '2026-08-07',
    title: 'An edited reference photo is no longer lost if the app restarts',
    blurb:
      'Editing your reference photo is a paid call that takes one to three minutes. If the app restarted before you pressed Keep, the finished image was still sitting in your dataset folder — but nothing could reach it any more: the modal came back empty, Keep refused, and the result you paid for was deleted half an hour later. It now comes back waiting for your Keep or Discard. If a second engine was still working when the app went down, it says so plainly instead of spinning forever on a result that is not coming.',
    to: '/datasets',
  },
  {
    id: '2026-08-07-video-install-checks-the-encoder',
    date: '2026-08-07',
    title: 'Installing the video extra no longer claims success when clips still cannot be cut',
    blurb:
      'The video extra delivers two things — reading your files, and encoding the clips you keep — and Setup only ever checked the first. So on a machine where the bundled ffmpeg never finished downloading (or an antivirus emptied it), the install said "✓ installed successfully" while the "Video bank — clip encoding" row stayed ✗ right underneath, behind the very same ↻ button: you reinstalled the half that already worked. That install now fails honestly and tells you which half is missing and what repairs it. The Setup row got stricter too: it runs ffmpeg instead of trusting that a file exists at the right path, so a truncated or quarantined binary is caught in Setup rather than in the middle of an export.',
    to: '/setup',
  },
  {
    id: '2026-08-07-face-mask-preview-survives-a-restart',
    date: '2026-08-07',
    title: 'The face-mask preview no longer disappears when you restart',
    blurb:
      'Looking for faces across a concept dataset is the slow part of the training panel — the detector alone takes seconds to load before the first image, then every kept photo on top. That whole pass used to live only in the app\'s memory, so restarting the server threw it away and the panel offered to run it all over again. It is now saved beside your images and comes back with them. Stopping a pass keeps its credit too: the faces already found survive the restart, so "Resume — 47 of 153 already analyzed" is still waiting for you. If you changed your kept images while the app was down, the preview comes back labelled out of date rather than pretending to be current.',
    to: '/datasets',
  },
  {
    id: '2026-08-07-parallel-cloud-runs',
    date: '2026-08-07',
    title: 'Train the same dataset twice at once',
    blurb: 'Launch a second cloud run on a dataset that is already training to compare toolkit settings side by side — confirm the extra pod, then follow each run from its own chip on the Training panel. The runs warn you if the dataset changed between launches.',
    to: '/cloud',
  },
  {
    id: '2026-08-03-phone-layout-full-width',
    date: '2026-08-03',
    title: 'The dataset page uses the whole screen again on a phone',
    blurb:
      'On a phone (and in the installed app), the dataset page drew everything at about three quarters of the screen width — the header bar, the section chips, every card — with a dead strip down the right-hand side. Nothing was cut off and nothing looked broken up close, which is what made it hard to place: the culprit was a one-pixel, invisible label that the horizontal section rail was failing to keep inside itself, which stretched the page wider than the screen and made the browser shrink the whole thing to fit. The rail now holds its contents, so the page measures the width of your screen and uses it. The Settings and Guide chip rails got the same treatment before they could catch it. Diagnosed and fixed by socrasteeze (GitHub).',
    to: '/datasets',
  },
  {
    id: '2026-08-07-captions-say-who-wrote-them',
    date: '2026-08-07',
    title: 'Every caption now says who wrote it',
    blurb: 'The default captioning backend is a chain, not a choice: JoyCaption writes what it can, then the Ollama vision model covers the images it missed — and both halves used to land in the same box looking identical. Captions now carry their author everywhere you read one (bank tile, dataset tile, the expanded editor, the identity-leak list), your own hand-written captions stay marked as yours, and a bank caption run ends with "340 by JoyCaption, 87 by the Ollama vision model" instead of a bare total. On a bank that ✨ Score has found to be mostly NSFW, the caption window now also says what was measured on that kind of image before you start.',
    to: '/bank',
  },
  {
    id: '2026-08-07-unlocked-rows-lead-to-their-install',
    date: '2026-08-07',
    title: 'Every "What\'s unlocked" row now takes you to its install',
    blurb: 'The Krea 2 Edit row told you it was missing and did nothing when clicked — it now opens the install screen where its one-click installer lives (or, when the files are already there, tells you to restart ComfyUI). Clip encoding joins the list too: a machine with no ffmpeg used to be certified complete while it could not export a single clip. And the scraper install warning finally names the packages YOUR machine is missing, instead of reciting three from memory.',
    to: '/setup?step=install',
  },
  {
    id: '2026-08-07-setup-summary-lists-every-installable-engine',
    date: '2026-08-07',
    title: 'Setup now lists every engine you can install',
    blurb: "Bank scoring, the SigLIP2 semantic engine, the fast watermark detector and the scraping extras (gallery links & keyless web search) could already be installed with one click, but never showed up on the wizard's \"What's unlocked\" screen — so finishing Setup could look complete on a machine still missing four of them. All four now get their own row, their own install card, and their own ✓/✗.",
    to: '/setup?step=quality',
  },
  {
    id: '2026-08-07-web-image-search-install-check-fixed',
    date: '2026-08-07',
    title: 'Web image search now flags a missing install correctly',
    blurb: 'If your scraper packages were installed before web image search (DuckDuckGo-style keyword search) shipped, the app used to say the scraper was ready and only fail once you actually searched. The Web images tab now correctly flags it as needing a reinstall, with a one-click "Install scraper extras" button to fix it.',
  },
  {
    id: '2026-08-06-reddit-load-more-no-longer-drops-posts',
    date: '2026-08-06',
    title: 'Reddit "Load more" no longer skips posts',
    blurb: 'On image-heavy subreddits, a batch of posts could hit the per-scan cap mid-way through — and the posts after that point were gone for good, on every page. "Load more" now picks up exactly where the cap cut off, so nothing is left unreachable.',
  },
  {
    id: '2026-08-06-cap-hit-scans-say-so',
    date: '2026-08-06',
    title: 'A capped scan says there might be more',
    blurb: 'Picazor, RedGifs and Instagram scans stop at a built-in limit to stay fast. Hitting that limit used to look exactly like a profile that genuinely had that few posts. Now the picker flags it, the same "some images may be missing" banner used for a scan cut short by a block or a timeout.',
  },
  {
    id: '2026-08-06-scan-truncation-and-blocked-pages',
    date: '2026-08-06',
    title: 'A blocked page says so — even on a slow scan',
    blurb: 'A gallery scan that hit both a login wall and the time budget used to report "nothing here" instead of the real reason. It now shows the actual block. And when a scan is cut short by the time limit, the picker tells you some images may be missing instead of quietly presenting a partial list as the whole thing.',
  },
  {
    id: '2026-08-06-bank-scrape-keeps-provenance',
    date: '2026-08-06',
    title: 'Images scraped into a bank keep their source',
    blurb: 'Scraping straight into a bank used to drop the Pexels credit or the page a web-search image was found on. It now survives the trip — and shows up again if you later promote that image into a dataset.',
  },
  {
    id: '2026-08-06-web-search-image-origin-link',
    date: '2026-08-06',
    title: 'See where a web-search image came from',
    blurb: 'Images you pull in from web search now carry a "Source ↗" link back to the page they were found on, right in the dataset — so you can credit or double-check the origin without hunting for it again.',
  },
  {
    id: '2026-08-06-any-gallery-site',
    date: '2026-08-06',
    title: 'Paste a gallery from almost any site',
    blurb: 'Sites without built-in support used to return a single image. They now list the whole page, using the same engine that already powers PornPics and Civitai. A site the engine does not know shows "No images found" instead.',
  },
  {
    id: '2026-08-06-web-image-search',
    date: '2026-08-06',
    title: 'Search the web for images by keyword',
    blurb: 'Type what you are looking for instead of hunting for a gallery URL first. No account, no API key. Results come from third-party sites, so check the licence before using one.',
  },
  {
    id: '2026-08-06-pinned-runs-keep-one-grid-per-prompt',
    date: '2026-08-06',
    title: 'Pinning a run to the canvas gives each prompt its own grid',
    blurb:
      'Pinning everything a run produced onto the ◉ LoRA Canvas used to drop the pictures of several prompts into one another, so a board carrying two or three runs read as one scatter and you could not tell at a glance which image came from which prompt. Each prompt now keeps its own grid, and a fresh grid is moved aside rather than laid over anything already on the board — including the cards, the checkpoint pills and the pictures you placed by hand, which are never touched. The reason for putting a run on the board at all is to SEE which prompt gave what; that is now what you get.',
    to: '/canvas',
  },
  {
    id: '2026-08-06-duplicates-headline-counts-what-is-left',
    date: '2026-08-06',
    title: 'The duplicates figure counts what you still have to decide',
    blurb:
      'The ≈ Duplicates figure on the bank overview counted every group the last analysis found, including the ones you had already resolved — so it sat at the same number no matter how much of the queue you cleared, and there was no way to tell progress from a stuck pass. It now counts what is left to decide, and says so: “groups remaining to resolve”. A bank whose duplicates are all sorted reads zero, which is the whole point of a headline number.',
    to: '/bank',
  },
  {
    id: '2026-08-06-face-mask-preview-stops-and-resumes',
    date: '2026-08-06',
    title: 'The face-mask preview can be stopped — and picks up where it left off',
    blurb:
      '"Looking for faces… analyzing image 4 of 153" had no way out: the only choice was to wait for the whole pass or leave the page and lose it. There is a Stop button now, and it keeps what the pass already found — start it again and it resumes at image 5 instead of image 1. The button says what stopping costs at the moment you press it, and the cost changes as the pass runs: during the model load nothing has been analyzed yet, so only the load is given up; once images are being analyzed, every face already found is kept. Change your kept images and the saved work is dropped rather than reused, because boxes from photos that left the set would describe a run that no longer exists.',
    to: '/datasets?section=training',
  },
  {
    id: '2026-08-06-setup-counts-the-video-pieces',
    date: '2026-08-06',
    title: 'Setup now counts the video pieces — and its repair menu can reach them',
    blurb:
      'The setup wizard could certify "12 of 12 capabilities ready" on a machine whose Video bank could not open a single file: the two video pieces were not counted, not listed in the Install-or-repair menu, and the wizard skipped its own install screen because everything it DID count was green. The summary now counts 14 — reading video files and shot detection included, each ✗ row clickable to where it installs — and 🎬 Video decoding and 🎞️ Shot detection sit in the Install or repair individually menu like every other component.',
    to: '/setup',
  },
  {
    id: '2026-08-06-shot-detection-installs-its-decoder',
    date: '2026-08-06',
    title: 'Shot detection no longer fails every file right after a clean install',
    blurb:
      'The 🎞️ Shot detection install put the model in place but not the decoder it reads files with — so the install reported success, the readiness badge turned green, and then every single file answered "failed shot detection". The install now carries PyAV into the same environment, the badge only turns green when the worker can actually open a file, and a contract test holds the three ends (worker, installer, probe) to the same list. If you hit this: Setup → 🎞️ Shot detection → ↻ Reinstall, then run Find shots again — your files were never the problem.',
    to: '/setup',
  },
  {
    id: '2026-08-06-bank-page-opens-without-rescanning-everything',
    date: '2026-08-06',
    title: 'The Bank page opens in a blink instead of a second',
    blurb:
      'Every time you opened 🗃️ Bank, the app re-listed every image in every bank\'s source folder before drawing the page — on a real library of 86 000 images that was a full second of disk work, paid even when you were only passing through. It no longer does: the page opens on what it already knows. A folder is still re-checked automatically when you OPEN its bank, so images you dropped in it show up as before, and a new 🔄 Rescan folders button on the list checks them all at once. The page says how fresh its counts are, and a folder that went missing (unplugged drive, renamed folder) is still flagged straight from the list.',
  },
  {
    id: '2026-08-06-semantic-index-borrows-a-gpu-python',
    date: '2026-08-06',
    title: 'Build the SigLIP 2 index on the GPU you already have',
    blurb:
      '✨ Score could already borrow a CUDA Python from ai-toolkit or ComfyUI instead of crawling on the CPU. The SigLIP 2 semantic index could not, so it kept indexing in the app’s own CPU-only environment even on a machine with a 4090 sitting idle. The Semantic engine panel now says which device the index will really use, and offers the same picker. It checks each interpreter dependency by dependency — and the list is the right one: SigLIP 2 never needs OpenCLIP or timm, so a ComfyUI venv Score refuses can be perfectly good here. Nothing is ever installed into an environment the app did not build: the pinned weights live in your data folder, and Setup ▸ Quality tools keeps installing into the app’s own Python even after you borrow one.',
    to: '/bank',
  },
  {
    id: '2026-08-06-video-extras-installable-from-setup',
    date: '2026-08-06',
    title: 'The video extras can now be installed where the app said they were',
    blurb:
      'The Video bank\'s banner told you "Install the video extra from Setup" — and Setup had no such button: both installs existed, but only for the API. Setup\'s optional-helpers step now carries the two missing cards: 🎬 Video decoding (PyAV + bundled ffmpeg into the app\'s own Python, no torch) and 🎞️ Shot detection (TransNetV2 into the scoring Python, CPU is fine). One click each, live progress, and the banner clears without a restart. A new test holds Setup to every install the video banners promise, so a pass can no longer point at a button that does not exist.',
    to: '/setup',
  },
  {
    id: '2026-08-06-video-watermark-flag',
    date: '2026-08-06',
    title: 'Spot the watermarked shots before they teach your LoRA a logo',
    blurb:
      'Rushes come off stock sites and other people\'s uploads, and a logo sitting in the same corner of every frame is the most consistent thing in your dataset — so it is the first thing a LoRA learns to draw. You cannot catch that by scrolling 90-pixel thumbnails. The new 🔖 Watermarks pass runs the same detector the image bank uses over each shot\'s sharpest frame and flags what it finds. Nothing is deleted: it is an amber flag you can filter on and act on. Needs the watermark detector from Setup; the cut sits in 🎚 Quality cuts at the measured 0.94, and a shot the pass has not judged is never called clean.',
    to: '/video-bank',
  },
  {
    id: '2026-08-06-video-duplicate-shots',
    date: '2026-08-06',
    title: 'Find the takes you already have, without watching them twice',
    blurb:
      'Ten near-identical takes of one gesture do not teach a model ten things — they teach it one thing ten times as loudly, and that is how a LoRA ends up unable to do anything else. The new ✂ Duplicates pass compares your shots to each other and groups the near-identical ones, keeping the sharpest of each pile unflagged so you know which one to keep. It costs no GPU and no waiting: it reuses the frame vectors 🔎 Find scenes already cached, so it is dot products over a file you already have. Flags only — nothing is rejected or deleted for you.',
    to: '/video-bank',
  },
  {
    id: '2026-08-06-video-flag-chips',
    date: '2026-08-06',
    title: 'Act on a quality flag instead of just reading it',
    blurb:
      'The amber flags in a video bank could be read one shot at a time and nothing more. There is now a row of chips above the gallery — "Barely moves (14)", "Same as another shot (31)" — and pressing one narrows the grid to exactly those shots, so you can select them and reject the lot in one gesture. The counts cover the shots currently loaded and the row says so when there are more to load.',
    to: '/video-bank',
  },
  {
    id: '2026-08-04-video-minimum-length',
    date: '2026-08-04',
    title: 'Half-second flash cuts stop cluttering your triage',
    blurb:
      'Shot detection deliberately keeps very short cuts — a real flash cut is a real shot, and a detector that hides them also hides genuine boundaries. The price was a grid full of half-second shots you scrolled past over and over, and that could never reach a dataset anyway. 🎚 Quality cuts now has a "Minimum length" field: type 1 second and every shorter shot wears an amber flag you can see and sort by. It is the one cut that works straight after detection — it reads the shot bounds, so you do not have to run the measuring pass first — and Preview tells you how many it would flag before you apply it. Nothing is deleted: it is a flag, like every other cut in that panel.',
    to: '/video-bank',
  },
  {
    id: '2026-08-04-video-train-local',
    date: '2026-08-04',
    title: 'Your video sets can now be trained here, without leaving the app',
    blurb:
      'A promoted video set now carries a ▶ Train this dataset button, and it hands the clips straight to the ai-toolkit already installed on your machine — no export, no copy, no config to write by hand. It shares the GPU with everything else honestly: a captioning pass or a ComfyUI render in flight refuses the launch instead of fighting over the card, and an image training already running blocks it exactly as another video run would. MiniMax H3 is wired in alongside Wan, with the quantisation, the noise schedule, the audio flags and the guidance its own trainer actually expects — a mismatch there does not crash, it just trains a slightly wrong model, which is why each value was read in the installed trainer rather than guessed. Two things it refuses to do quietly. H3 needs about 43 GB of weights: if they are not on your disk the button says so, names the repository and the size, and waits for a yes rather than turning into a silent overnight download. And a set re-promoted to a different target is refused rather than resumed, because the run folder still holds the previous model’s LoRA. Wan 2.2 is the one target a finished run has been through here; the card says plainly which of the others are wired but not yet proven.',
    to: '/datasets',
  },
  {
    id: '2026-08-04-video-cloud-training',
    date: '2026-08-04',
    title: 'A promoted video dataset can now be trained on a rented GPU without leaving the app',
    blurb:
      'Open a video training set in your library and it has a Cloud training panel: pick a step count, click Train in the cloud, and watch the run — the GPU it landed on and what it costs per hour are on screen as soon as the pod has them. When it finishes, the checkpoints are listed by step and downloadable from the same panel. Wan 2.2 saves TWO files per step, the high-noise and low-noise experts, and either one alone is a LoRA nothing can load — so a step is offered as a step, with both of its files, never as a loose file list. A run that failed has a Retry, a run that finished has Train further, and continuing a Wan run sends both experts back to the new pod rather than resuming half of it. Before the job starts, the pod is asked to decode one of the clips you just uploaded with the same decoder the trainer uses; a pod that cannot read them fails in the first minute instead of billing you for hours of training on nothing.',
    to: '/datasets',
  },
  {
    id: '2026-08-04-video-caption-wording',
    date: '2026-08-04',
    title: 'Captions can now speak plainly instead of describing around the subject',
    blurb:
      'Next to 🗣 Describe shots there is now a Caption wording choice. Standard is the wording that shipped and stays the default. Plain gives the model explicit permission to name what is actually on screen rather than reaching for vague stand-ins. This came out of a measurement, not a hunch: on real footage, four combinations were compared and the WORDING mattered more than the model did — the stock model asked plainly named things precisely and wrote the best action description of the four, while an uncensored model asked the old way still described around the subject. That matters because a caption that talks around its footage is a dataset defect you cannot see: the text reads perfectly well, the training set looks complete, and the LoRA learns the evasion. Every caption now records which wording produced it, so a bank captioned across a change is still one you can reason about. Pick it per run, or set it once in your config.',
    to: '/video-bank',
  },
  {
    id: '2026-08-04-video-caption-model-choice',
    date: '2026-08-04',
    title: 'Captions can now speak plainly — the model that writes them is yours to choose',
    blurb:
      'The 🗣 Describe shots pass had one model wired in. On a real corpus that turned out to be a dataset problem rather than a matter of taste: a captioner that describes what it sees in evasive terms produces captions that are about something slightly other than your footage — and a LoRA trained on those learns to look away too, with nothing in the output to reveal it. The captions read perfectly well; they are just not about the shot. So the checkpoint is now a setting, `video_caption.model`. Leave it empty and nothing changes: the same model as before, the same captions. Point it at any checkpoint of the same architecture and the pass uses that instead. Two things come with it. If the model is not on your machine yet, the pass SAYS so in its own progress line before it starts, because the first run downloads it and that should never be a silent twenty-minute wait. And every caption now records which model wrote it, so a bank captioned half before the change and half after is still a bank you can reason about.',
    to: '/video-bank',
  },
  {
    id: '2026-08-04-video-describe-shots',
    date: '2026-08-04',
    title: 'Your shots get described — so you can search for what HAPPENS, and so they train on words',
    blurb:
      'There is a new 🗣 Describe shots pass. It watches eight frames spread across each shot and writes what happens in it — “a woman turns and walks away”, not an inventory of objects — and that one line does two jobs. It becomes the clip’s .txt sidecar at promotion, which IS the prompt it trains on: until now every promoted clip shipped with an EMPTY prompt, and the trainer accepts that in silence. And it makes 🔎 Find scenes able to answer a question it structurally could not before. CLIP looks at frames, so it finds what a moment LOOKS like; an action is a fact about time and no single frame carries it. With captions the search reads both, and the panel says which halves are running so an empty result can be read correctly. Captions are drafts: open any shot and edit it, and a bulk re-run will not overwrite what you wrote. The promotion now also tells you how many clips are about to ship with no caption at all, before it encodes anything.',
    to: '/video-bank',
  },
  {
    id: '2026-08-04-video-source-cap-knob',
    date: '2026-08-04',
    title: 'The per-source cap you could already read about now has a knob',
    blurb:
      'The build dialog gains a “Max clips per source” field. The cap itself is not new — it has capped nothing so far because nothing could send it: the setting was implemented and reachable from neither the dialog nor the API. It matters because a 50-clip set that is three videos over-represented looks exactly like a diverse one on disk. Leave it empty for no cap. Each source keeps its EARLIEST clips, so promoting the same bank twice gives the same dataset, and a source with fewer clips than the cap keeps all of them — it trims dominance without punishing scarcity. And when a finished set turns out to lean on one file anyway, the result now says so with the real share instead of leaving you to notice.',
    to: '/video-bank',
  },
  {
    id: '2026-08-04-video-edge-trim',
    date: '2026-08-04',
    title: 'Trim the dissolve off both ends of every clip you export',
    blurb:
      'A shot boundary is where a cut just happened, so the first and last frames of a detected shot are disproportionately dissolves, fades and leftovers of a transition — and a dataset whose clips all open on half a dissolve teaches the model to open on half a dissolve. The build dialog now has a “Trim each end” field: a number of seconds taken off BOTH bounds of every clip. 0.25 is the common figure; the default is 0, so an existing recipe exports exactly what it exported before. What it will NOT do is hand you a short clip. Frame counts are a property of the target model’s VAE, and ffmpeg happily writes a 32-frame file and exits 0 when asked for 81 — so a clip that no longer supplies the count is dropped rather than exported short. The dialog tells you how many clips the trim will cost before you press the button, and counts them separately from clips that were never long enough: only the first kind is fixed by lowering the trim, and reporting them as one number is how a setting quietly halves a dataset while the material looks to blame.',
    to: '/video-bank',
  },
  {
    id: '2026-08-04-video-audio-metrics',
    date: '2026-08-04',
    title: 'Your shots are now listened to, not just looked at',
    blurb:
      'For LTX and MiniMax H3 the source’s audio is muxed into every clip you export — and until today nothing had ever listened to it. A shot whose track is a silent stretch, a dropout or a muted camera passed exactly like a shot with sound, because the file on disk is the right length, the right sample rate, and mute. A dataset of silent clips teaches the model to be silent. Measure now reports, per shot, how much of it is silence and its overall level in dBFS, with two new cuts to go with them — Silent share and Loudness floor — raising two different flags on purpose, since a quiet clip can be normalised and a silent one cannot be rescued. Three states are kept strictly apart, because collapsing any two of them makes the bank lie: a file with NO sound track is never flagged (Wan datasets are supposed to look like that), a track that is there and carries nothing is the actual defect, and shots measured before this shipped have no sound reading at all — an audio cut will never flag those, and Measure with re-measure is what fills them in.',
    to: '/video-bank',
  },
  {
    id: '2026-08-04-video-find-scenes',
    date: '2026-08-04',
    title: 'Type a word and find the scene, in a folder of rushes with no names',
    blurb:
      'A video bank is a haystack whose needles have no names: quality cuts tell you which shots are sharp and which move, and nothing tells you which one has the red car in it. There is now a 🔎 Find scenes box above the gallery. Run the pass once — it looks at a few frames of every shot — then type “a woman walking on a beach” and the gallery is replaced by the shots that look most like it, best first, instantly. Several frames per shot on purpose: a car that only drives into view in the last second would be invisible to a search that had looked at the opening frame, and you would get no hint it had been missed. So every shot contributes a frame near its start, its sharpest frame and one near its end — and every result tells you WHICH SECOND matched, with the player opening right there. Two things it says out loud rather than hiding. It is a ranking, not a filter: every shot scores something against every phrase, so the results always come back full and the line above the gallery tells you how strong the top and the tail really are, plus how many shots have not been looked at yet and could not be searched at all. And “without” does not work — ask for a street without cars and you get cars, because the model ignores the word rather than honouring it. Type “-cars” instead: that pushes them down the ranking, which is a promise the app can actually keep.',
    to: '/video-bank',
  },
  {
    id: '2026-08-04-video-clip-retouch',
    date: '2026-08-04',
    title: 'A badly cut shot is no longer a shot you have to throw away',
    blurb:
      'Until today, a shot the detector cut one second too early — or one holding a frozen tail — had exactly one available gesture: ✕ Reject, which threw away the eight good seconds to be rid of the bad one. Open any shot and there is now a ✂ Trim & split panel under the player. Nudge either bound by one second or by one frame (one frame OF YOUR FILE, at its own rate), snap a bound to wherever the playhead sits, split a shot in two at the playhead, or draw a shot the detector missed entirely — scrub anywhere in the rush and press ＋ New shot from here. Splitting keeps the decision you were making: split a kept shot and both halves stay kept, so you do not have to find them again among hundreds. One thing worth knowing before you trim: for image-to-video targets, the trainer conditions on the clip’s FIRST frame. Moving a start is therefore not trimming — it is choosing the exact image the model learns to animate from, and the panel says so where the buttons are. Re-cut shots lose their thumbnail and their quality scores on purpose: a thumbnail of a frame the shot no longer contains is not stale, it is wrong. Run Make thumbnails again when you are done cutting. And re-detecting a file no longer destroys the cuts you made by hand.',
    to: '/video-bank',
  },
  {
    id: '2026-08-04-video-quality-flags',
    date: '2026-08-04',
    title: 'Your video bank now measures every shot — and tells you which ones to look at',
    blurb:
      'One pass reads every frame of every shot and scores what quietly ruins a video dataset: shots that barely move, shots that are all blur, black moments, frozen stretches. Nothing is rejected for you — flagged shots get an amber mark in the grid, and the verdict stays yours. The cuts are yours too: there are deliberately NO default thresholds, because the same number that flags 2% of one bank flags 12% of another. Open 🎚 Quality cuts, set a value, and Preview shows exactly how many shots each cut would flag — per rule, before anything is applied. On a real 4.5-hour test bank the most valuable filter turned out to be the frozen-stretch one: 15% of shots carried a freeze the average could never see. Bonus: thumbnails now come from the SHARPEST measured frame instead of the middle guess.',
    to: '/video-bank',
  },
  {
    id: '2026-08-04-video-bank',
    date: '2026-08-04',
    title: 'Your folder of rushes is now a training set — and a .mp4 is no longer ignored in silence',
    blurb:
      'Drop a video into an image bank and until today it was skipped without a word: no row, no warning, nothing to click. Videos now get their own bank. Point it at a folder of rushes and it cuts every file at its shot boundaries, so you triage SHOTS instead of files — a two-hour rush becomes three hundred things you can judge in an afternoon. Click any shot to watch exactly that moment; the grid stays thumbnails, so a bank of hundreds of shots stays as light as a page of photos. Nothing is copied and nothing is re-encoded while you triage: a bank stores where each shot starts and ends, and only the ones you keep are ever encoded. When you build the set, the length menu offers only the frame counts your target model can actually ingest — 29 frames is legal for Wan and illegal for LTX, and no trainer tells you, they just quietly round it down. Two things are written next to the target you pick, because they are what costs a wasted week: whether a LoRA trainer for it is known to exist at all (exactly one of the four), and MiniMax H3’s licence, which grants no rights in the EU, the UK, South Korea or the USA — outputs included. And if a piece is missing, the app names which one: with no ffmpeg you can still scan, cut, watch and triage everything — only the final encode waits.',
    to: '/video-bank',
  },
  {
    id: '2026-08-06-aim-the-watermark-cleaning-levels',
    date: '2026-08-06',
    title: 'Choose which images get cropped or repainted — the two watermark levels no longer fire on one click',
    blurb:
      '🚩 Watermarks offered “✂ Auto-crop (16 052)” and “🧽 Inpaint (16 507)” as single buttons: press one and thousands of images were cleaned, with no way to say which. Both now open the same launch window the other passes use — run them on your kept images only, on the undecided ones, on the bin, on all three, or on exactly the images you ticked in the grid. Each line carries the real number for that pile, a run that would touch nothing is refused instead of reporting success, and the window states up front what is reversible: your own files are never written to, the cleaned pixels live in the bank’s own copy, and ↩ Undo cleaning throws them away and re-flags the images — what it cannot reach is an image you already promoted into a dataset, or one whose source file changed on disk since the clean. A run that only covered part of your bank now names what it left out, and flagged images sitting in the bin are reachable at last instead of leaving a dead button. Leave the window alone and both levels behave exactly as before.',
  },
  {
    id: '2026-08-06-big-banks-open-instantly-and-stop-answers-again',
    date: '2026-08-06',
    title: 'Big Banks open in a second again — and Stop answers while a pass runs',
    blurb:
      'A Bank of 50 000 images took over 12 seconds to load its workspace, and more than 28 while a pass was running — long enough that the progress bar and its Stop button never arrived, so a running scan could not be stopped. The workspace was re-checking the location of every single image on disk, twice each, on every 2-second refresh; it now remembers what it already resolved. The progress bar has also moved to its own tiny request, so it keeps ticking and Stop keeps working no matter how large the Bank or how busy the pass. Measured on a 50 397-image Bank: 18.5 s to 1.5 s, and the progress bar answers in 1 ms.',
    to: '/bank',
  },
  {
    id: '2026-08-06-each-bank-can-choose-clip-or-siglip2-without-losing-work',
    date: '2026-08-06',
    title: 'Each Bank can choose CLIP or SigLIP 2 — switching keeps every result',
    blurb:
      'A Bank now chooses the semantic engine behind Find by text, Similar, diverse and balanced picks, coverage, and same-shot grouping. CLIP remains the zero-change default and still owns Score; SigLIP 2 is an explicit optional install with its own pinned, resumable index. The two caches and the two same-shot partitions are stored separately, restored when you switch back, and carried through Bank → Dataset, Dataset → Bank and Bank → Bank transfers. Live readiness counts now follow the images that are actually in the Bank, so deleting or rejecting the last group cannot leave an old “65 ready” graph behind.',
    to: '/bank',
  },
  {
    id: '2026-08-05-describe-the-set-you-want-and-the-filters-move',
    date: '2026-08-05',
    title: 'Say what you want in a sentence, and the bank sets its own filters',
    blurb:
      'Type "an amateur photo set, least polished first" above the triage chips and the bank moves ITS controls — medium, flags, resolution, sort — then the counters you already had say, measured, how many images that lands on. The model never looks at your pictures and never picks any: it only reads the sentence, so a wrong reading costs one glance at chips you can edit rather than a silent selection of a few thousand images you would have to trust. It answers over what your bank has actually measured, with the real per-value counts in front of it, so it cannot reach for a bucket that holds nothing. When part of a request has nowhere to land — asking for what is IN the pictures on a bank whose captions cover 1% of it — it says that part is not expressible here instead of inventing a filter that would return three thousand convincing, unrelated images. And it will not turn an exclusion into a search phrase: the ranker in this app returns MORE of a negated thing, not less, so "without a watermark" is reported rather than quietly sent.',
    to: '/bank',
  },
  {
    id: '2026-08-05-bank-passes-say-what-they-left-out',
    date: '2026-08-05',
    title: 'Bank passes now say what they were doing, what they grouped, and what they left out',
    blurb:
      '✨ Score used to go quiet for minutes once its counter reached the last image, bar full, Stop still offered — on a 21 000-image bank that silence read as a freeze. It was writing a 70 MB cache and then comparing every embedding against every other; both steps now name themselves while they run, and the ones where Stop costs something different say what it costs before you press it. The style grouping stopped hiding behind a number too: "1 style group of 2+" was printed both when almost nothing grouped and when a single group had swallowed 24 928 of 24 931 images. It now states the size of the biggest group against the total, and names the threshold when the grouping is too loose to separate anything or too tight to join anything. And every pass that skips part of your bank says so at the end: 🎨 Classify medium answering "0 classified" on a 50 000-image bank was exact — 25 464 rejected images had been dropped by the scope before the pass even began, and that figure appeared nowhere. It does now, on 🔎 Scan, 🏷️ Caption, 📐 Framing, 💧 Watermark and 🎨 Medium alike, and stays silent when a run really did reach everything. 🎨 Medium\'s launch window also counts, per scope, the images it cannot answer for because ✨ Score never reached them — and refuses to start a run where that is all of them.',
    to: '/bank',
  },
  {
    id: '2026-08-05-every-bank-pass-asks-where-to-run-before-it-runs',
    date: '2026-08-05',
    title: 'Every bank pass now asks where to run — kept, undecided, unkept, or your selection',
    blurb:
      'Pressing 🔎 Scan, 🏷️ Caption, 📐 Classify framing or any other pass used to fire immediately on whatever the app had decided was its pool. Each of them now opens a window first, and that window says three separate things. Where the run applies: kept only, undecided only, unkept (the bin), all three, or the images you have selected — each line quoting the number of images IT would actually touch, so a scope with nothing to do is refused before the click rather than reported as a success afterwards. What the pass reads: only the settings its calculation consumes, with where they live, so you stop guessing whether nudging a threshold costs you a rescan. And what is NOT decided there: the knobs that only re-sort and re-flag the grid, retunable any time with no pass at all. "Rescan all" and "Rescore all" were never separate actions and no longer pretend to be — they are a tick box in their pass\'s window, next to the pool they re-run and with their price written next to them. Three passes refuse a partial scope, and now say so instead of quietly not offering it: ✨ Score, 👥 Group by person and ✂ Find crops & variants each produce one numbering of the whole bank, and handed a slice they would land fresh group ids on top of unrelated groups already saved. Aiming a pass at the bin is offered everywhere it makes sense, always with what that run costs, and never as a default.',
    to: '/bank',
  },
  {
    id: '2026-08-05-the-launch-all-report-names-every-step',
    date: '2026-08-05',
    title: 'The Launch-all report calls every step by the name on its button',
    blurb:
      'Two of the eight steps in the "Last Launch-all run" summary printed their internal identifier instead of their name — you would read "semantic_dedup" and "framing" in your own report and have no way to tell which button that was. They are now "✂ Find crops & variants" and "📐 Classify framing", the exact words on the buttons you pressed. The fallback that made the omission invisible is now guarded by a test that reads the server\'s own step list, so a ninth step cannot ship nameless.',
    to: '/bank',
  },
  {
    id: '2026-08-05-select-images-and-see-their-tags-with-how-often-each-is-cited',
    date: '2026-08-05',
    title: 'Select images and their tags are right there, with how often each is cited',
    blurb:
      'Once a bank is captioned, selecting an image now shows its tags immediately — no extra click, no hunting for a badge. Select several and you get the tags across the whole selection, each with how many of them mention it: "red dress 7 / 12" tells you at a glance that over half of what you picked is that, which is the judgement you were making anyway. Ticking a chip filters the bank exactly as before, and the row holds still while it does. It is honest about its own reach: images with no caption yet are counted apart and named, so are captions whose every word was a grammar word, and a selection too large to read in one go says how many it left out instead of quietly shrinking the denominator. The 🏷️ button on a tile has also moved out of the badge cluster in the top-left corner — that corner is a state readout where nothing is clickable — down to the bottom-right next to ▶ and ⛶, where the tile\'s other actions live. When an image has no caption, or a caption with no word worth filtering on, the button stays visible and says which of the two it is, instead of disappearing and reading as a feature that does not exist.',
    to: '/bank',
  },
  {
    id: '2026-08-05-bank-chip-counts-follow-the-filters',
    date: '2026-08-05',
    title: 'The numbers on the bank filter chips now match what they open',
    blurb:
      'Every chip in the bank — 🌫 Blurry, 📺 Noisy, the resolution tiers, 🔎 Origin, 📐 Framing, 🎨 Medium, ⤢ Angle — used to print a total for the whole bank whatever else you had filtered. Pick ✕ Rejected, read "📺 Noisy 4 043", click it, get twelve images. The counts now describe the bank you are actually looking at: each one is measured with your other filters applied. The value you already picked is deliberately left out of its own count, so choosing 🌫 Blurry never zeroes 📺 Noisy next to it and you can always switch to a neighbour without clearing everything first. Chips stay on screen even when they hold nothing under the current filter, for the same reason. Two smaller things went with it: clicking a ✨ Score or ≈ Duplicates chip no longer silently drops the person or style group you were in, and the "showing N of M" line now notices 🔎 Origin, the 🚫 exclude box and the 🏷️ tag chips, which it used to ignore.',
  },
  {
    id: '2026-08-05-a-scoring-pass-that-came-back-empty-now-says-why',
    date: '2026-08-05',
    title: 'A ✨ Score pass that could not load its models tells you why',
    blurb:
      'A scoring pass could finish, announce "done", and leave every image without a single number — with sorting by aesthetics, find crops and variants all greyed out because there was nothing to sort on. The closing line named the head that had gone quiet and stopped there: "(aesthetic + NSFW head unavailable)". Both models are fetched over the internet the first time they are used — the aesthetic predictor from GitHub, the NSFW classifier from Hugging Face — so on a machine that cannot reach them, typically a container without outbound network, both go down together and the whole pass comes back empty for one reason nobody was told. That reason was known all along, in the scoring log, and never made it to the screen. It does now: the sentence carries the actual error, so "no scores" reads as a network problem you can fix rather than a feature that looks broken. Reported by @_nofaceman on Discord.',
    to: '/bank',
  },
  {
    id: '2026-08-05-upscale-improve-finds-its-lora-on-a-docker-install',
    date: '2026-08-05',
    title: '✨ Upscale & improve stops re-downloading a LoRA you already have',
    blurb:
      'On a Docker or Linux install, raising the enhancement-LoRA strength made every ✨ Upscale & improve answer “Klein needs klein_enhancement_lora — I’ve started downloading it”, then download the file it already had, then say it again. The file was never missing: the improve workflow was exported from a Windows ComfyUI and names that LoRA klein\\realistic.safetensors, and on Linux a backslash is part of a filename rather than a folder separator — so the app looked for one file with a strange name instead of realistic.safetensors inside klein/. Setup, which spelled it correctly, kept showing it installed, which is why the two screens disagreed and only that one file looked broken while every other model loaded. The name is now respelled for whichever system opens it, on the way in and on the way out to ComfyUI. Nothing to reinstall — the LoRA already on your disk is picked up after an Update & restart. Reported by @_nofaceman on Discord.',
    to: '/settings/engines',
  },
  {
    id: '2026-08-05-stopping-score-no-longer-throws-away-what-it-computed',
    date: '2026-08-05',
    title: 'Stopping ✨ Score keeps the scores it had already computed',
    blurb:
      'Pressing Stop during a scoring pass used to leave your images exactly as they were: the work reached a cache on disk, but not a single image got its aesthetic or NSFW number. Now everything it computed is written to your images before the pass ends, and the closing line tells you how many were scored, how many are left, and how many were reused from the cache instead of recomputed — so a relaunch is visibly paying only for the rest. The 🎨 style groups are the one thing that still needs a full pass to land, because those ids are a single numbering of the whole bank and half of one would mix two unrelated styles under the same chip; a stopped pass now leaves the previous grouping untouched and says so, rather than writing a contradiction. Two more things that were quietly wrong: a run where the aesthetic or NSFW model could not be downloaded no longer wipes the scores your bank already had, and the images scored during such a window are picked up again once the missing piece is back, instead of staying half-scored forever. Finally, "Rescore all" is available for when you do want everything recomputed from scratch — it is the last line of ✨ Score\'s launch window, unticked, with its price next to it.',
    to: '/bank',
  },
  {
    id: '2026-08-05-reject-every-flagged-watermark-in-one-click',
    date: '2026-08-05',
    title: 'Drop every watermarked image at once — and get them back if you change your mind',
    blurb:
      'Reviewing flagged images one by one is still the careful way through a watermark scan, but it is no longer the only way: a dataset\'s curation row now offers ✕ Reject all flagged next to 🔍 Review flagged. The number on the button is what it will really reject, not how many are flagged — small-image rescue pairs and failed rows are left out, because including even one rescue row made the server refuse the whole batch and reject nothing at all. It asks before it acts, tells you exactly what it is about to do, and then tells you the way back: rejected images stay on disk, and Show ▸ Rejected in the grid followed by ✓ Keep brings any of them home. The one thing it does destroy is said out loud beforehand — rejecting clears the watermark flags, so 🔍 Review flagged will be empty afterwards and only a new scan can flag them again.',
    to: '/datasets?section=curation&panel=reject-flagged',
  },
  {
    id: '2026-08-05-choose-which-engine-finds-your-watermarks',
    date: '2026-08-05',
    title: 'Datasets and banks now find watermarks the same way — and you pick which way',
    blurb:
      'The two screens had quietly drifted apart: a bank used the fast watermark detector whenever its extra was installed, while a dataset always asked the vision model, one slow question per image, with no way to change it. Both now read one setting — Settings ▸ Captioning & quality ▸ Watermark detection — with three values: Auto, Watermark detector, or Vision model. Auto is the default and behaves exactly as before, so an untouched install changes nothing. Pin the detector without its extra installed and the scan still runs on the vision model rather than failing, and the app says so, with the link to install the extra. Where the two engines genuinely differ, the screen says that too: only the detector can flag an image without knowing where the mark is, and those are now counted apart and left for 🔍 Review instead of being marked failed. A ⟲ Rescan incl. dismissed button appears when you have ruled some flags false positives — the only way to have them judged again by a different engine.',
    to: '/settings/captioning',
  },
  {
    id: '2026-08-05-stop-a-watermark-scan-without-losing-it',
    date: '2026-08-05',
    title: 'A watermark scan can be stopped, and keeps everything it already found',
    blurb:
      'Starting 🧽 Find watermarks on a large dataset used to be a commitment: there was no Stop button anywhere, and the only way out was closing the tab and hoping. The progress banner now carries the same ⏹ Stop the captioning pass has. It stops after the image in flight — never mid-inference — and every watermark already found is kept, so running 🧽 Find watermarks again simply picks up where you left off. The live "Scanning… 12/340" counter now actually moves in the tab that started the scan, too; it only ever updated after a page reload before.',
    to: '/datasets?section=curation&panel=watermarks',
  },
  {
    id: '2026-08-04-the-quality-scan-no-longer-takes-the-app-away-with-it',
    date: '2026-08-04',
    title: 'The 🔎 quality scan leaves the rest of the bank usable',
    blurb:
      'On a big bank, pressing 🔎 Scan quality could take the whole app away for a couple of minutes: the bar reached 100 %, said “grouping duplicates”, and nothing else answered — sorting, renaming, promoting, all of it waited. Three things were behind that, and all three are fixed. The duplicate grouping no longer re-runs over your whole bank at the end of a scan that had nothing to scan — it runs when the hashes actually changed, and when you ask for it with “↻ Re-group duplicates”, which still applies a new duplicate distance without decoding a single file. When it does run it now shows its own progress and stops when you press Stop, instead of leaving you in front of a frozen-looking bar with no way out. And it no longer holds the database while it works, so the rest of the bank keeps answering: on a 50 000-image bank the grouping went from up to two minutes of dead application to a phase you can watch and interrupt, and the worst wait a click had to sit through fell from over five seconds to a few dozen milliseconds.',
    to: '/bank',
  },
  {
    id: '2026-08-04-full-model-training-on-turbo-and-your-own-krea-checkpoint',
    date: '2026-08-04',
    title: 'Full-model training now accepts Krea 2 Turbo — and a checkpoint of your own',
    blurb:
      'Full-model training used to accept exactly one thing: the official Krea 2 Raw base. Choosing Turbo did not just refuse — it silently switched the dataset back to LoRA, which is why the option looked like it did not exist. Both refusals are gone, and the reason they were there has been fixed rather than waived: the configuration sent to the rented GPU used to name Raw no matter what you had selected, so simply lifting the ban would have produced a run labelled Turbo that trained Raw for hours. The base is now resolved from your selection everywhere it appears — the job sent to the pod, the run’s provenance, the model card, the recipe summary — and the full-model panel finally has its own Raw/Turbo switch and base picker. You can also fine-tune a Krea 2 checkpoint from your own disk: it rides to the GPU through a private repository on your Hugging Face account, exactly as it already did for LoRA runs, and the launch checks the pod’s credential can really read it before renting anything. What has not changed: a ComfyUI scaled-fp8 export still cannot be trained at all — the loader refuses it — and the app says so when you pick it, while a plain fp8 cast stays allowed. Turbo comes with a plain warning shown before you spend anything: Krea recommends training a LoRA on Raw and applying it to Turbo, nobody has measured full-model training on a distilled base, and it may cost the model its few-step behaviour. It is a warning, not a wall.',
    to: '/datasets?section=training',
  },
  {
    id: '2026-08-05-re-caption-keeps-the-captions-you-wrote-yourself',
    date: '2026-08-05',
    title: 'Re-caption no longer destroys the captions you wrote yourself',
    blurb:
      '🔄 Re-caption used to warn you, honestly, that it could not tell a caption you had typed from one a model had written — one column, one string, no author. It can now: every caption records who wrote it, whether that is JoyCaption, Ollama, or you (typing one in a dataset, running a find/replace, or bringing .txt files back from another tool), and that record travels with the text through dataset → bank, bank → bank, promotion and backups. So a forced run keeps your own words and says so, the way the face pass already skips a person you declared yourself. The button now tells you three separate things instead of two: how many images it rewrites, how many captions it keeps because you wrote them, and how many it overwrites whose author was never recorded — captions written before this existed, which cannot be recovered and are re-captioned, so you learn that before the click and not after. If you actually want your own captions redone by a better model there is a tick box for it, offered only when you have something to lose.',
    to: '/bank',
  },
  {
    id: '2026-08-04-the-app-stops-claiming-a-full-model-it-has-not-checked',
    date: '2026-08-04',
    title: 'Your full models say where they really are, instead of where they were',
    blurb:
      'A delivered full model was announced as “available” on Hugging Face for ever after, because that status is written once, when the run ends, and was never asked again — so a repository you deleted last night still offered a link that answered 404, and one card managed to print “missing — the model is there” in the same sentence. The Checkpoints panel and the Runs page now ask Hugging Face when you open them, and say one of three things: it is still there, it is gone, or the check itself failed (no token, offline, an outage) — which is never reported as a loss. Until an answer arrives they date the delivery instead of claiming the present. When a repository really is gone they tell you what is left — the master on your disk, the fp8 twin, or nothing — and the buttons that could only have failed stop being offered: the dead links go, “Quantize to fp8” is disabled with its reason, and ▶ Continue is disabled only when the run left nothing on this computer to send to a pod instead.',
    to: '/datasets?section=checkpoints',
  },
  {
    id: '2026-08-04-choose-how-a-full-model-reaches-the-pod',
    date: '2026-08-04',
    title: 'Continue a full model from your own copy — and see what each way costs',
    blurb:
      '☁ Continuing a 12B full model used to be possible from exactly one place: its Hugging Face copy. If a run had been delivered to this computer only, or its Hub copy had been deleted, there was nothing to do — the app refused, and the reason was a limit inside one upload route, not a fact about the world. ▶ Continue now offers both roads and prices them before you click. Hugging Face is minutes, because the pod downloads over a datacenter link; sending the copy from this computer is autonomous and costs your upload speed. The figure that actually decides it has never been shown anywhere until now: the pod is rented and billed the entire time it waits for its file, so three hours of upload at $1.40/h is $4.20 of GPU computing nothing. You see that number, the file size, and how long it should take — measured on your own past transfers once you have made one, and openly labelled an estimate before that. A long upload is no longer all-or-nothing either: the file goes up in slices, and every slice that landed stays on the pod, so a dropped link, a closed app or a reboot resumes where it stopped instead of starting over. When a road is closed the dialog says which one and why, and what would reopen it. The Hugging Face road is checked rather than remembered, too: the app knew only that a delivery had once succeeded, so a repository you deleted last night still offered itself with a duration and a price — and picking it rented a pod that then found nothing to download. Opening the dialog now asks whether the repository still answers. A confirmed deletion closes that road; a check that could not be made (offline, no token, an outage) leaves it open, because losing your fast road to a dropped connection would be worse than the problem being avoided.',
    to: '/cloud',
  },
  {
    id: '2026-08-04-auto-reject-shows-the-number-it-will-actually-reject',
    date: '2026-08-04',
    title: 'Auto-reject stops promising more than it can do',
    blurb:
      '🧹 Auto-reject offered “5,930 flagged” for blurry shots, and rejecting them did nothing — because those 5,930 had already been rejected by an earlier run, and the pass deliberately never re-flips a decision. The count announced was not the count the button acted on, so the honest conclusion was that the feature was broken. It was not: the number was. Each checkbox now shows how many still-undecided images that exact click would reject, so a second run says “0 to reject” instead of advertising work it will not do — and running it really does reject that many. The filter chips keep counting every image carrying the flag, rejected ones included, which is what you want when you click one to look at them. Two things that used to hide behind an identical 0 now say which they are: a flag whose pass never ran tells you to run it first, and the panel names how many images have never been scanned at all — invisible to every quality flag until 🔎 Scan reaches them, which is very different from being clean. 🚀 Launch all shows the same numbers, and says plainly that the scan runs first so they will grow.',
    to: '/bank',
  },
  {
    id: '2026-08-04-the-merge-form-keeps-what-you-typed',
    date: '2026-08-04',
    title: 'The merge tool stops emptying itself when the window changes shape',
    blurb:
      'Turning a phone to landscape — or anything else that reshaped the page — folded “Merge a LoRA into a base checkpoint” shut and threw away the checkpoint path and the LoRA rows you had just typed, with nothing to undo. The panel it lives in is rebuilt when the layout moves, and it was taking the form down with it. What you type is now kept: the tool stays open where you left it and comes back filled in, after a resize, a rotation or a reload alike. It is cleared the moment a merge actually starts, so a form you already sent never comes back looking like unfinished work.',
    to: '/datasets?section=checkpoints',
  },
  {
    id: '2026-08-04-converting-a-custom-zimage-base-stops-hitting-the-paging-file',
    date: '2026-08-04',
    title: 'Converting a custom Z-Image base no longer dies on a “paging file” error',
    blurb:
      'If preparing your own Z-Image merge for training ended in “the paging file is too small to complete this operation”, nothing was wrong with your disk, your memory or your model — and buying more disk would not have helped. Opening the checkpoint reserved its whole size up front: on a 11.5 GB base that was 11.7 GB claimed in a tenth of a second, before a single number had been read, and about 15 GB by the time the conversion was under way. Big custom bases are exactly where this bit, because they are the ones people convert. The file is now read one tensor at a time and nothing is mapped: the same 11.5 GB conversion peaks at 0.19 GB and finishes in about a minute, so the size of your checkpoint no longer has anything to do with whether it can be opened. The check that runs before the conversion got faster too — it only ever compared shapes, which are in the file header, and it was loading all 11.5 GB to read them.',
    to: '/datasets?section=training',
  },
  {
    id: '2026-08-04-typed-base-path-is-checked-while-you-type',
    date: '2026-08-04',
    title: 'A base you type by hand is checked before you launch, not after',
    blurb:
      'Pick a base from the dropdown and the panel tells you immediately if it is a packed export the trainer cannot load, or an fp8 cast that trains from already-degraded weights. Type the path yourself under “Custom weights…” and, until now, you got that same verdict only when you saved or launched — which on the cloud lane meant after the dataset had been exported and a GPU had been rented. The typed path is now read the moment you stop typing: same check, same sentence, same red box, and the Train button stays disabled if the file cannot be loaded at all. A path that is not there, or that is not a .safetensors, says so instead of letting the run find out.',
    to: '/datasets?section=training',
  },
  {
    id: '2026-08-04-quantizing-no-longer-fails-on-the-paging-file',
    date: '2026-08-04',
    title: 'Quantizing a big model no longer dies on a “paging file” error',
    blurb:
      'If you tried to turn a full-precision model into its fp8 file and got “the paging file is too small to complete this operation”, nothing was wrong with your disk, your memory or your model — and adding disk space would not have helped. Opening the checkpoint reserved its entire size, 26 GB of it, before reading a single number. The app now reads big checkpoints one tensor at a time instead, so the size of the file no longer has anything to do with whether it opens: a 25.6 GB model that could not be opened at all now quantizes in about a minute, and the read-back check at the end works the same way, so it can no longer fail on the last step after twenty minutes of work.',
    to: '/datasets?section=checkpoints',
  },
  {
    id: '2026-08-04-studio-krea-base-is-chosen-not-hardcoded',
    date: '2026-08-04',
    title: 'The Test Studio picks its Krea base instead of hoping one file is there',
    blurb:
      'The Krea 2 base the Studio rendered on was a filename frozen into a workflow file — and not the one Setup installs. On a machine that simply followed Setup, that name matched nothing ComfyUI publishes, and the whole run was refused before a single step: it only ever worked if you happened to own that one community repack. The base is now chosen from what is actually on your disk, in an order you can check: the file Setup installs first, then a Turbo build (these graphs run at CFG 1 and a handful of steps, which only a distilled build can do), then whichever file the header says is in better shape — full precision ahead of a quantized cast, ahead of a packed export. Last of all comes anything carrying tensors the model family never declares: the repack that used to be the default hides about 75 MB of image inside itself, 432 tensors where Krea 2 has 430, announced in its own metadata. It stays usable — if it is the only Krea file you own it is still the default — but the Studio now tells you which file it picked and why, and the sampler numbers follow that file instead of assuming Turbo.',
    to: '/studio',
  },
  {
    id: '2026-08-04-merge-a-lora-into-a-base-checkpoint',
    date: '2026-08-04',
    title: 'Turn your LoRA into a full model you can publish',
    blurb:
      'Most of the checkpoints you download were not trained — they were merged: a LoRA folded into somebody’s base, quantized, uploaded. LDS could train the LoRA and could quantize the result, and could not do the step in between, so you could not reproduce what everyone else was doing. Now you can: pick a base, add one or more LoRAs with a weight each, and get a complete checkpoint. It also unlocks the speed problem — a full model trained here targets Raw, which is slow, and merging in the re-distillation LoRA Krea publishes for Turbo is the published route to getting few-step generation back (we have not tested that one ourselves, and the screen says so). Nothing starts on one click: the plan tells you how many tensors change, exactly how big the output is, which drive it lands on and how long it takes — about two minutes on a 26 GB base — and nothing is ever overwritten. And it calls the result what it is. A merged model is not a trained model, however often the model sites say “finetune” for it, so the file records the base, every LoRA and its weight, and the date, in its own metadata — which is what still identifies it in six months, after the name has changed.',
    to: '/datasets?section=checkpoints',
  },
  {
    id: '2026-08-04-turbo-dense-untested-not-impossible',
    date: '2026-08-04',
    title: 'Correction: full-model training on Turbo is untested, not impossible',
    blurb:
      'The Guide and the app both told you that training a full model on Krea 2 Turbo was mechanically impossible and would deliver a broken file. That was overstated, and we are correcting it. On the distilled models anyone has actually measured, full training leaves a perfectly valid checkpoint that simply stops being fast — and for Krea 2 nobody has published that measurement at all. Full-model runs still target Raw, and nothing about how they behave has changed; the refusal now says what it really is — a lane we have not tested — and the Guide explains what is known, including the published route people use to get the speed back afterwards.',
  },
  {
    id: '2026-08-04-train-krea-on-a-checkpoint-you-already-have',
    date: '2026-08-04',
    title: 'Train Krea 2 on a model you already have — including the one your last run delivered',
    blurb:
      'The Krea 2 base selector offered exactly one thing: the official base. So a full model you had just trained, or any Krea 2 build sitting in your ComfyUI folders, could not be used as the starting point of the next run — while a Z-Image merge could. It now lists every Krea 2 checkpoint on your disk, the yaml-declared folders included, and hands the real file to the trainer (local runs load it directly; a cloud run offers to push it to your private repo first). Each entry also states its format before you pick it: an ordinary fp8 file trains — the trainer up-casts it as it loads — and now says with numbers how much precision the cast dropped, instead of being refused outright as it used to be. Only a packed ComfyUI export is still refused, because it carries decompression tables a trainer literally cannot load; the message says that, and points at the bf16 master your run kept next to it.',
    to: '/datasets?section=training',
  },
  {
    id: '2026-08-04-canvas-says-which-run-trained-a-full-model',
    date: '2026-08-04',
    title: 'On the board, a run that trained the whole model now says so',
    blurb:
      'A full-model run and a LoRA run of the same family printed exactly the same two words on their card — “Krea 2 · Raw” — while being completely different things: one produces a large checkpoint you load instead of the base, the other a small adapter you load on top of it. On a board holding both, nothing told you which was which. Those cards now carry a “full model” badge, in the graph and in the list alike.',
    to: '/canvas',
  },
  {
    id: '2026-08-04-full-model-is-selectable-as-a-studio-base',
    date: '2026-08-04',
    title: 'The fp8 file of a full model you trained is now offered as a base in the Test Studio — with its own sample settings already filled in',
    blurb:
      'One line decided it: the base picker only accepted a Krea checkpoint if it sat in a folder whose name carried “krea”. The fp8 twin of a full-model run is written to the root of ComfyUI’s diffusion_models folder — deliberately, because that is a folder ComfyUI reads — so the one file the whole full-model lane exists to produce was invisible to the one screen meant to try it, and the only way to test a model you had paid hours of GPU for was to open ComfyUI by hand. The picker now also accepts a file whose NAME carries “krea”, which is the rule the Generate side has always used, so every twin already on your disk appears without moving a byte. And because a full model trained here is undistilled, selecting it fills in CFG 4 / 25 steps rather than the family’s few-step Turbo defaults, which render a blurry sketch on it. Those per-base settings now also reach the comparison and blend screen, which never received them.',
    to: '/studio',
  },
  {
    id: '2026-08-04-full-models-appear-in-the-checkpoints-panel',
    date: '2026-08-04',
    title: 'A full model you trained now appears in 📦 Checkpoints & LoRAs, with the two files it leaves and what each one is for',
    blurb:
      'Until now a full-model run existed in the app as a banner and a Hugging Face link. It was in none of the places a trained thing lives, so testing one meant going into ComfyUI and finding the file yourself. There is now a 🧱 Full models block in the same panel, and it keeps apart the two files a run leaves — because they are not interchangeable. The full-precision master (~26 GB) is the only one you can train again or resume from, and it is never sent to ComfyUI: it would fill a model folder to do a job the smaller file does better. The fp8 twin (~13 GB) is the inference format, and → Send to ComfyUI puts it where ComfyUI looks — on the same drive that is a hard link, so it is instant and costs no extra disk space. Each card names the exact file it chose when a run left several 26 GB saves, the sampler settings the model wants, the machine it trained on, and where the Hugging Face backup is. ✨ Quantize works whether the master is on this computer or still only in the repository, and 🗑 Trash is the app trash, so a mis-click on hours of GPU is recoverable.',
    to: '/datasets?section=checkpoints',
  },
  {
    id: '2026-08-04-comfyui-unreachable-is-said-out-loud',
    date: '2026-08-04',
    title: 'When LDS cannot reach ComfyUI, the banner now says exactly that — instead of blaming a paused job',
    blurb:
      'On a brand-new install, the very first Generate could answer “A paused ComfyUI job is blocking new generations” — on a machine that had never generated anything, while ComfyUI logged no incoming connection at all. The pause was real, but it was the consequence, not the cause: LDS was knocking at an address nobody was behind, and nothing on screen said so. The banner now checks whether the two programs are in touch and leads with the answer: “LDS cannot reach ComfyUI at <your URL>”, followed by the three things that are actually wrong when that happens — the address of the ComfyUI window you really use, a ComfyUI started without --listen (it only answers on its own machine), and LDS in Docker needing host.docker.internal instead of 127.0.0.1. The paused job is still held, and still clears the same way; it just stops being the accusation. Reported by jerkyjunky (Discord).',
    to: '/settings/local-tools',
  },
  {
    id: '2026-08-04-prompt-batch-grid-shows-every-prompt',
    date: '2026-08-04',
    title: 'A batch of prompts now shows all of its images — one labelled grid per prompt, instead of a grid with a single picture',
    blurb:
      'Ticking several saved prompts generated every one of them, but the results view only ever showed one: it identified a run by its seed AND its prompt, so a single launch of five prompts arrived as five separate “runs” in the run picker, and you were looking at one of them. Nothing was lost — the images were all there, behind a dropdown that should never have had them. A run is now identified by the launch itself, so the batch stays whole, and each prompt gets its own grid under the prompt that produced it (shortened to fit, in full when you hover). Runs from before this change are grouped exactly as they were.',
    to: '/studio',
  },
  {
    id: '2026-08-04-full-model-run-no-longer-reads-as-gone',
    date: '2026-08-04',
    title: 'A full-model run whose model is on Hugging Face no longer shows up as “gone” — and the app no longer offers to delete it',
    blurb:
      'The canvas asked one question to decide whether a run still had anything: is there a checkpoint file on this disk? That question has no good answer for a full model delivered to a private Hugging Face repository — there is no local file, and there never was. So those runs were drawn dimmed, badged “gone”, and given a “Remove this run” button under the words “No checkpoints left on disk”, for a model that was perfectly fine and had cost hours of GPU. Removing one threw away the lineage, the notes and the only record of which repository the model was in. A full-model run is now asked about both of the addresses it can have: it shows “💾 full model here” when the weights are on this computer, “☁ on Hugging Face” when they are in its repository, and it is only offered for removal when the model is genuinely gone from both. If you try anyway, the app now says where the model still is instead of deleting the trail to it.',
    to: '/canvas',
  },
  {
    id: '2026-08-04-person-preflight-redraws-unreadable-samples',
    date: '2026-08-04',
    title: 'The folder check before 👤 Group by person no longer gives up when its sample lands on faceless images — it draws new ones',
    blurb:
      'On a real bank, four folders out of six came back with no verdict at all: “only 0 of 15 sampled images had a usable face — analyzed in full”. Scraped folders are full of crops, backs and blur, so a sample of fifteen can land entirely on images no face detector can read — and the biggest folder then spent fifteen embeddings for nothing before analysing all 3 546 of its images anyway, which is exactly the cost the check exists to avoid. A draw that cannot be read is now replaced: the check keeps drawing new images (never one it already tried, still spread across the whole folder) until it has about fifteen usable faces or hits a budget — at most 60 images per folder, or a quarter of it, whichever is smaller, and the dialog prints that ceiling next to the typical cost before you start. Three honest endings replace the old silence: a normal verdict; “looks like one person, on thin evidence — only 6 usable faces in 60 images tried”, still offered because a weak verdict that says how weak beats none; and, when almost nothing is readable, a fact about the folder rather than a promise the full pass would do better — it would not, it uses the same detector on the same images and re-reads the answers the check already cached.',
    to: '/bank',
  },
  {
    id: '2026-08-04-fp8-quantize-runs-where-torch-lives',
    date: '2026-08-04',
    title: 'Quantizing to fp8 now actually runs — it uses an environment that has torch, and says so before you click if none does',
    blurb:
      'On a real install the conversion could not run at all: it ended on “No module named ‘safetensors’”, because it tried to do the work inside the app’s own Python — which ships without torch on purpose, since torch is gigabytes and nothing else here needs it. It now runs the conversion in a separate interpreter that has the dependencies, exactly like ✨ Score and the masking passes already do: the one ✨ Score uses, ai-toolkit’s, or whichever you set as `quantize.python`. And because “can this machine do it at all” is something you should learn before committing, it is checked while the plan is drawn: an environment without torch disables the button and tells you which environments would work and what to install, instead of failing thirty seconds in — or, worse, after a 26 GB download.',
    to: '/datasets?section=training',
  },
  {
    id: '2026-08-04-fp8-disk-guard-says-yes-when-it-fits',
    date: '2026-08-04',
    title: 'The fp8 quantizer no longer refuses a conversion that fits — and a refusal now shows its arithmetic',
    blurb:
      'Two things were wrong with the disk check, and both showed up on a real 25.6 GB model. The panel said the conversion was fine, and the click that followed refused it: the threshold was only applied when starting, never when planning, so the button stayed enabled right up to the moment it was too late. And that threshold was a flat 30 GB, while the file being written was 12.8 GB and the drive had 17.6 GB free — an operation that fit twice over, refused by a number of our own. The budget is now derived from the job itself (what is left to download, the fp8 file’s own ceiling, and 2 GB of working headroom), every term is named in the refusal so you can check it, and whatever the plan accepts the start no longer rejects. Free space is also measured on the volume that really holds the folder, which matters because a ComfyUI models folder is very often a junction onto another drive. And when a drive genuinely is too full, the refusal offers to write the file to another folder instead of ending there.',
    to: '/datasets?section=training',
  },
  {
    id: '2026-08-04-quantize-to-fp8-in-one-click',
    date: '2026-08-04',
    title: 'One click turns your full model into the fp8 file ComfyUI loads — nothing to type, and it works on a model that is only on Hugging Face',
    blurb:
      'The fp8 quantizer asked for an absolute path to a file on this machine, and the model most people want to shrink has no such path: a full-model run delivers its 26 GB master into your private Hugging Face repository and never downloads it. So the one full model you own was the one thing the tool could not touch. Now “✨ Quantize to fp8” sits right there in the full-model recipe, already aimed at the model your run delivered, and does the whole chain: fetches the master, converts it, and leaves the fp8 file in ComfyUI’s own models folder, ready to load. Before it starts it tells you which checkpoint it takes (a repository often holds the final save AND several 26 GB step snapshots whose names differ by a number — one rule now decides, and it is the same one that names the file on the card), which folder the file lands in, and what it costs in disk. The download reports its gigabytes, can be stopped, and resumes where it left off. Afterwards the master is kept by default, because it is the only copy you can train from again; deleting it is one radio button away with its size on it. The path field is still there for a file nothing in the app points at — and it now pre-fills itself with your custom training base.',
    to: '/datasets?section=training',
  },
  {
    id: '2026-08-04-dense-quality-levers',
    date: '2026-08-04',
    title: 'Full-model training: choose how many images each step learns from, and how the learning rate moves',
    blurb:
      'A full-model Krea 2 run was training on ONE image per step — over a dataset of thousands, that is a very noisy idea of the right direction — at a flat learning rate from the first step to the last. Three settings in the recipe card change that. “Images per step” averages several images into each update, which steadies training on a big set; it needs no extra VRAM, only time, so the card tells you straight out that 4 images per step means a run about 4× longer and a rented GPU that costs about 4× as much. The learning-rate schedule can now warm up over the first steps instead of hitting a 12B model at full rate immediately, or fade to zero by the end to settle fine detail. The noise schedule picks which noise levels the run trains on. Leave all three alone and your run is byte-for-byte the recipe that shipped before — the defaults did not move. Two settings people ask for are deliberately absent, and the guide says why: on this model EMA would run the pod out of memory at its first checkpoint, and min-SNR weighting would crash the job an hour in, because a flow-matching model has none of the numbers it needs.',
    to: '/datasets',
  },
  {
    id: '2026-08-04-image-grid-pages-big-datasets',
    date: '2026-08-04',
    title: 'A dataset of thousands of images no longer bogs the Images screen down',
    blurb:
      'The Images grid used to draw every photo of the dataset at once. On a 6 211-image dataset that is about 148 000 elements on one page — 6 211 thumbnails, 6 211 caption boxes, 60 000 buttons — and it showed: scrolling ran at roughly 20 frames a second on a desktop and 12 on a phone, and a single keystroke in a caption took a tenth of a second to appear. The grid now shows 500 images at a time with a ← Prev / Next → pager above and below it, the same way the Bank has always handled 24 000-image folders. Measured on that same 6 211-image dataset: scrolling back at full speed, typing in a caption instant again, switching a filter about seven times faster. Nothing about curation changed — “select all” still takes every image the current filters show across all pages (its tooltip now says so), a selection you started on one page is still there on the next, the counters, sort, filters and auto-triage all still read the whole dataset, and captions are still edited right on the tile. The pager only appears when there is more than one page.',
    to: '/datasets?section=images',
  },
  {
    id: '2026-08-04-full-model-lands-on-your-computer',
    date: '2026-08-04',
    title: 'A finished full model now lands on YOUR computer — and a full Hugging Face quota can no longer end a training',
    blurb:
      '🖥 Until now a full-model (dense) run had exactly one address: a private Hugging Face repository the pod pushed to while it trained. That address has a ceiling nobody controls, and it collected: a run died 250 steps from the end on “403 private repository storage limit reached”, after eight hours of paid GPU, and only survived because 50 GB were deleted by hand. So the order is reversed. The finished model is downloaded to your checkpoint folder FIRST, the ~10 GB fp8 file for ComfyUI with it, and the pod is destroyed only once the file here is proven — the byte count has to match what the pod advertised, and the safetensors header has to re-read. Only then is the master uploaded to Hugging Face as a backup, and that upload is now allowed to fail: it costs the ability to continue that model later, nothing else. Nothing is pushed while the run trains, so the quota can no longer reach the training at all. The transfer is tens of minutes of 26 GB, so it shows its progress, survives an app restart, and can be stopped and resumed without losing what already landed — and if it fails, the machine is kept and the Runs page offers “Fetch to this computer”. A launch also checks this machine’s disk before renting anything, and refuses (confirmably, like every other estimate) when the drive plainly has no room. Choose the delivery in Settings ▸ Storage ▸ Full-model delivery; runs made before today keep their Hugging-Face-only behaviour exactly as it was.',
    to: '/settings/storage',
  },
  {
    id: '2026-08-04-continue-a-full-model',
    date: '2026-08-04',
    title: 'Continue a full model instead of paying for its first 3000 steps again',
    blurb:
      '▶ A full-model run that stopped at step 3000 can now be continued to 4000 — the same ▶ Continue as a LoRA, with the same guardrails, cost estimate and “from which step” choice. The fresh machine downloads the checkpoint from the run’s Hugging Face copy itself, over a datacenter link, and training picks up at the step written inside the file. Two honest limits, said in the app rather than discovered: the copy on your computer cannot be used for this (the only channel that puts a file on a pod builds its whole request in memory, which 26 GB cannot survive), so a run delivered to this computer ONLY is not resumable — the default delivery keeps a Hugging Face copy precisely to leave that door open.',
    to: '/cloud',
  },
  {
    id: '2026-08-04-cloud-quantize-rents-a-machine-that-fits',
    date: '2026-08-04',
    title: 'Cloud quantization picks a machine that can hold the model (server side — this lane has no button)',
    blurb:
      'A correction first: this lane has no interface, so there is no click to make. The fix below is real and lives in the server half; the reachable way to shrink a model is ✨ Quantize to fp8, locally, from a full-model card or Settings ▸ Storage. Cloud quantization used to give up one second after being asked with “create_instance failed: HTTP 400 {}” — no machine, no money spent, and no reason. Two things were wrong. It rented the cheapest offer on the market, and cheap is exactly where free disk runs out: a 26 GB model needs about 86 GB on the pod for the master, its fp8 twin and the download cache, while the top offer of a live search had 57 GB — an ask vast refuses outright. And a single refusal ended the job, even though the next machine would have taken it. Now the search only considers machines with the disk this job will claim, the offer is chosen by the same rule a training launch uses (bad hosts skipped, suspiciously cheap listings ignored), and a refusal moves to the next candidate instead of ending everything. When a rental really is impossible, the error quotes what vast said rather than an empty “{}”. The estimate stays an estimate — but if the market moved and the only machine left costs materially more than the price you agreed to, it tells you and rents nothing.',
  },
  {
    id: '2026-08-04-quantize-to-fp8-from-settings-storage',
    date: '2026-08-04',
    title: 'Shrink a model to fp8 from Settings ▸ Storage — no dataset, no training run',
    blurb:
      'The fp8 quantizer that shipped yesterday had exactly one door: the full-model recipe card, which only exists inside a dense dataset. So the person it was written for — someone who downloaded a 26 GB full-precision model from Hugging Face and cannot load it — had no dataset, and never found it. It is now also in Settings ▸ Storage, beside the folder sizes and the trash, because “this file is too big” is a disk question. It is the same tool, not a second copy: point it at any full-precision .safetensors on this machine and it writes the ~10 GB version ComfyUI loads directly, next to the original. Same refusals before you click (a file that is already quantized, a LoRA/adapter), same promise that your source file is never modified and never overwritten, same read-back of the result before it reports success. It runs on the CPU, one at a time, so it never takes VRAM from ComfyUI or a training run. One correction landed on the same tab: the Hugging Face storage card counts the fp8 export in what a full-model run needs, but did not name it, so a ~60 GB forecast explained itself as 46 GB — the breakdown now lists every term it adds up.',
    to: '/settings/storage',
  },
  {
    id: '2026-08-04-gallery-lightbox-upscale-improve',
    date: '2026-08-04',
    title: 'Upscale & improve is now in the checkpoint gallery — the screen the result lands on',
    blurb:
      'Yesterday ✨ Upscale & improve arrived on the ◉ LoRA Canvas lightbox. It was missing from the one place an improvement actually appears: a checkpoint’s gallery. Open any picture from a pill’s 🖼 gallery or from a run card and the button is there, next to ⬇ Download — the same pass, the same choice between Klein (re-renders detail and texture) and SeedVR2 (resolves detail, keeps the look), and the same quote of the instruction Klein is about to send. It is the same action on the same picture as on the board, wired once rather than twice, so the two screens can never start behaving differently. The original is never touched: the improvement arrives as its own image in that very gallery, beside the picture it came from, ready to compare, download or pin onto the board. One honest limit — the pass takes minutes and a gallery left open does not refresh by itself, so close it and open it again to find the new picture waiting at the top.',
    to: '/canvas',
  },
  {
    id: '2026-08-04-canvas-drop-keeps-your-view',
    date: '2026-08-04',
    title: 'Arranging the canvas no longer throws your framing away',
    blurb:
      'Park a render up beside another dataset’s lane, let go — and the whole board zoomed out from under you, because it had just become bigger. Every time you tidied, the canvas re-framed the thing you were tidying, and the further you placed something the harder it kicked. From now on, moving anything — a pinned picture or a run card — means you have taken the view over: the board keeps the zoom and the position you chose and never re-frames itself again. ✦ Fit is still one click away for when you do want the whole board back, which is the difference between an offer and an interruption. A board you have never arranged still opens fitted to your screen, exactly as before.',
    to: '/canvas',
  },
  {
    id: '2026-08-03-krea-edit-takes-an-extra-angle',
    date: '2026-08-03',
    title: 'Krea 2 Edit takes a second image — add it in the edit dialog, and use it to compose',
    blurb:
      'Editing your reference with Krea 2 Edit used the main photo and nothing else. You can now add a second image, with the “+” inside the ✦ Edit reference dialog. What it is for matters more than the fact it exists: that slot was trained on two-input edits where the second image is a *different* subject — another person, or a scene to place yours in. So it composes (“put her in this room”, “next to him”). Another angle of the same face is off-label there and can come back duplicated, which is why it deliberately does NOT read the dataset’s extra reference photos: those are angles of one person by definition. Extra angles keep doing what they always did, on the engines built for them — Klein chains every one of them from the reference card and locks identity across every generation, and the API engines use them too. Two different jobs, two different places, and the dialog now says which is which before you press Generate.',
  },
  {
    id: '2026-08-03-canvas-images-go-anywhere',
    date: '2026-08-03',
    title: 'Pinned images go anywhere on the canvas, not just below and right of their run',
    blurb:
      'A picture pinned onto the ◉ LoRA Canvas could be dragged down and right as far as you liked, but never up and never left: its own lane\'s corner was a wall, so you could not park a render above its run, in the free margin beside the board, or next to another dataset\'s lane to compare across datasets. That wall is gone — the mouse and the arrow keys both reach everywhere now, and ✦ Fit grows to include a picture wherever you put it, so it is always one click from being back on screen. Nothing about where an image came from changes: the line to the checkpoint that made it follows it, because that link is read off the image itself rather than off its position. Three things came with it. ✦ Tidy up is the way home — it brings every picture on the board back beside the run that made it, side-by-side strips included, moved in one piece and never taken apart — and it is no longer greyed out on a board where only pictures have been moved, which is exactly when you need it. The board no longer re-zooms under your finger while you drag something past its edge — nor when you let go of it (see the entry above). And a strip of grouped pictures now draws ONE line back to each checkpoint it came from instead of one per picture, so a long link stays readable.',
    to: '/canvas',
  },
  {
    id: '2026-08-03-canvas-group-drag-out-crash',
    date: '2026-08-03',
    title: 'Pulling a picture out of a group on the canvas no longer blanks the board',
    blurb:
      'Dragging one image off a strip of grouped images showed the error screen instead of the picture coming loose — the board went blank and the only way back was a reload. The hint that appears while you pull ("Drag it off the group to take it out") was reading a size that had moved to another file when the group\'s title bar was split out earlier today, so the very gesture it exists to explain was the one that crashed. It is back, at the same size as the bar\'s own label at every zoom. Nothing you had pinned was lost — the board reloads exactly as you left it.',
  },
  {
    id: '2026-08-03-canvas-lightbox-upscale-improve',
    date: '2026-08-03',
    title: 'Upscale a picture without leaving the canvas',
    blurb:
      'Open a picture on the ◉ LoRA Canvas and it now carries ✨ Upscale & improve next to ⬇ Download — the same pass, and the same choice between Klein (re-renders detail and texture) and SeedVR2 (resolves detail and keeps the look) you already had in the dataset lightbox, with the same live quote of the instruction Klein is about to send. Until now the only way to improve a render you liked on the board was to go and find it somewhere else. The picture on the board is never touched: the result arrives as its own image in that checkpoint’s gallery, right next to the original, so you can compare the two and pin the better one. SeedVR2 offers to install itself if it is not there yet, an improvement cannot be improved again, and these upscales stay out of the Test Studio — they never count as a run in progress and never enter a checkpoint’s 👍/👎 ranking.',
    to: '/canvas',
  },
  {
    id: '2026-08-03-checkpoint-store-survives-cleanup',
    date: '2026-08-03',
    title: 'A cleanup can no longer delete a checkpoint you never deployed',
    blurb:
      '“Clean finished runs” said it was trashing “checkpoint duplicates already imported”. For a checkpoint you had never sent to ComfyUI there was no duplicate: the run’s staging folder was its only copy, and emptying the trash destroyed it. Trained files now land in their own checkpoint store the moment they are downloaded, and no cleanup path can reach them — cleaning a run moves its dataset copy, its sample images and its logs, and nothing else. Anything still sitting in an old staging folder is swept into the store at startup, and a button in Settings › Storage re-runs that sweep whenever you want. Two more honest answers while we were there: a run folder nothing points at is now listed with its size instead of being answered “already clean”, and a kept pod stops sparing tens of gigabytes once its recovery window has actually closed.',
    to: '/settings/storage',
  },
  {
    id: '2026-08-03-settings-storage-tab',
    date: '2026-08-03',
    title: 'One tab that says where everything lives — and moves it to another drive',
    blurb:
      'When C: fills up, the answer used to be a config.json edit and a manual copy. Settings › Storage now shows every folder the app writes to — datasets, banks, cloud run staging, the checkpoint store, the trash, the archive, the Hugging Face cache — with its real path, what it holds and the free space on its drive. Press 📏 Measure everything to see the sizes (never on open: walking a hundred gigabytes while you read is not a thing a page should do). Three of those roots can be pointed anywhere: type a path, the app proves it can really write there, then you choose out loud — move what is already there, with a progress bar and the old folder removed only once the last byte has landed, or start using the new folder empty and leave the old files exactly where they are. Nothing is ever moved silently. The trash, the run image archive and the Hugging Face allowance moved here too, so the disk questions are answered together instead of one card per screen.',
    to: '/settings/storage',
  },
  {
    id: '2026-08-03-prompt-batch-no-cap',
    date: '2026-08-03',
    title: 'The prompt batch no longer caps at 24 — and the time estimate is now yours',
    blurb:
      'Ticking more than 24 saved prompts was refused. That number was a guess, not a limit: nothing breaks past it, and it governed the wrong thing — 24 prompts across eight checkpoints went through, while 25 on a single one did not, even though the second run is far shorter. Tick as many as you like. What you get instead is the real cost, before the click: the panel counts every generation the run will queue and tells you how long that takes at the pace your own machine has actually been running at, measured from your recent tests rather than assumed. Past about an hour it asks once whether you meant it, and reminds you that the queue is serial — you can stop it whenever you like and keep what is already done. Every duration in the Studio and on the canvas now comes from that same measured pace, so a slower card stops being told it will take twenty minutes when it will take two hours.',
    to: '/studio',
  },
  {
    id: '2026-08-03-full-model-fp8-export-for-comfyui',
    date: '2026-08-03',
    title: 'A finished full model now also arrives as a ~10 GB file ComfyUI can just load',
    blurb:
      'Full-model training delivers a 26 GB bf16 checkpoint, and nobody generates with a file that size — everyone hunts for a community fp8 repack instead. Now the run makes one itself, on the pod, in the minutes before the machine is released: a scaled fp8 export (~10 GB) pushed next to the master in your private Hugging Face repo, loadable with the standard Load Diffusion Model node, no extra setup. The run card lists both files and says which to download for ComfyUI and which one is the master. “Keep the bf16 master” stays on by default on purpose: fp8 is a one-way export, and the master is the only file you can ever continue, merge or re-quantize from — turning it off halves your storage and closes that door. If the export fails the run is still a success: the master was delivered before it ever ran, and the card says exactly that instead of reporting a failure.',
    to: '/cloud',
  },
  {
    id: '2026-08-03-quantize-in-the-cloud-without-downloading',
    date: '2026-08-03',
    title: 'Cloud quantization: built on the server, never given a button — corrected',
    blurb:
      'This entry announced a “☁ Quantize to fp8 in the cloud” button, and that button was never wired: the service, its three endpoints and their tests are in the app, but nothing in the interface calls them, so there has never been anything to click. We are correcting the claim rather than quietly leaving it: an app that announces what it cannot do is worse than one that shipped less. What DOES work, and always did, is the local conversion — ✨ Quantize to fp8, from a full-model recipe card or from Settings ▸ Storage. It runs on your CPU in about a minute, so the arithmetic was never the expensive part. The cost the cloud lane was meant to save is bandwidth: a master that lives only in your Hugging Face repo has to come down (26 GB) and go back up (10 GB) for the local path to touch it. If that round trip is your problem, say so on the Discord — the server half is already written and only the door is missing.',
    to: '/settings',
  },
  {
    id: '2026-08-03-quantize-an-existing-model-to-fp8',
    date: '2026-08-03',
    title: 'Turn any full-precision model you already have into its ~10 GB ComfyUI version',
    blurb:
      'The same conversion the cloud runs at the end of a full-model training is now available by hand, on this machine: give it the path to any full-precision .safetensors — a 26 GB model downloaded from Hugging Face, a checkpoint from an earlier run — and it writes the fp8 version next to it. The source file is never modified and an existing output is never silently overwritten. It runs on the CPU, so nothing competes with ComfyUI or a training run, and when it finishes it re-opens the file it wrote to check the scales and dtypes are what ComfyUI expects. It refuses a file that is already quantized, and it refuses a LoRA — neither has anything to gain. Worth saying plainly, because it is constantly confused: this is NOT the “quantize” option in Advanced training, which only shrinks the model in memory while it trains and writes no file at all.',
    to: '/datasets?section=training',
  },
  {
    id: '2026-08-03-full-model-recipe-unlocked-four-settings',
    date: '2026-08-03',
    title: 'The full-model recipe finally lets you change the four things that matter',
    blurb:
      'Full-model training only exposed “steps”, so the mid-run preview sheet was four generic prompts that showed nothing about your actual dataset — the one thing you look at while the GPU is billing. Preview prompts are now editable, along with the learning rate (1e-7 to 5e-6), the resolution (768 or 1024) and the checkpoint cadence (every ≥100 steps, keep 1 to 3). Defaults are unchanged, so nothing moves unless you move it. Keeping 3 checkpoints means about 78 GB of private Hugging Face storage, and the panel now says the total before you launch instead of letting the last push fail. Everything still locked — batch size, Adafactor, bf16, gradient checkpointing — now says WHY: it is the geometry that makes a 12B model fit on one 80 GB card, and changing it turns a working run into an out-of-memory crash an hour in.',
    to: '/datasets?section=training',
  },
  {
    id: '2026-08-03-raw-full-model-test-settings',
    date: '2026-08-03',
    title: 'Testing a full model no longer looks like a failed training',
    blurb:
      'A full model trained on Krea 2 Raw is an undistilled checkpoint: it needs a real CFG and a real step count. The app was handing it Turbo’s defaults — CFG 1 and 8 steps — which render a blurry sketch and read as “the fine-tune did not work”, when nothing had gone wrong. The Test Studio now pre-fills CFG 4 and 25 steps when the selected base looks like a Raw / full / fp8 checkpoint (the exact settings the run’s own preview sheet used), and the training panel and run card say it in words: test at CFG ~4 and 20-30 steps.',
    to: '/datasets?section=training',
  },
  {
    id: '2026-08-03-quantized-base-refused-at-selection',
    date: '2026-08-03',
    title: 'Picking a quantized checkpoint as a training base is refused before it costs you a run',
    blurb:
      'The fp8/int8 exports everyone keeps on disk for generation cannot be trained on — the weights no longer carry the precision a gradient step needs. Until now the app took one happily and the run died deep inside ai-toolkit, after the dataset export and, in the cloud, after a GPU had been rented. Choosing one as Custom weights is now refused the moment you pick it, with a sentence that says what to do: “This is an inference-only quantized export — training needs the bf16/fp16 version of this model.” It reads a few kilobytes of file header, never the 10 GB body, and a header it cannot read is let through rather than guessed at.',
    to: '/datasets?section=training',
  },
  {
    id: '2026-08-03-person-pass-checks-folders-first',
    date: '2026-08-03',
    title: 'The person pass now checks your folders first — and asks once',
    blurb:
      'Scraped material arrives one folder per person, and 👥 Group by person used to pay a face embedding per image to rediscover that. The bank could already sample folders and offer the obvious ones, but only from a button in the Subfolder panel — and the first thing anyone does with a new bank is press 🚀 Launch all, so nobody ever saw it. Now the sampling runs by itself the moment you launch the person pass, standalone or inside Launch all: about fifteen images per folder, then one dialog saying “12 folders look like a single person — treat each as one person and skip their full analysis”, with those folders already ticked. One click accepts them all; untick what you disagree with; “👥 Analyze everything anyway” is right there and says what it costs. Folders where the sample showed several faces say so and go to the full analysis, and a ceiling that was not reached is named rather than assumed away. Nothing is ever grouped without your click — and whatever you accept is a normal folder assertion you can undo any time.',
    to: '/bank',
  },
  {
    id: '2026-08-03-hf-storage-precheck-and-cleanup',
    date: '2026-08-03',
    title: 'Full-model cloud runs now check your Hugging Face space before renting a GPU',
    blurb:
      'A full-model (dense) Krea run delivers each ~26 GB checkpoint straight into a private Hugging Face repo — and that push happens at the very end. A run died at step 2750 of 3000 on “private repository storage limit reached”, hours of paid GPU gone, because the account\'s private space was full of custom-base caches nothing in the app ever showed you. Now the launch measures your private storage first and refuses before a pod is rented, saying how much is missing and what is taking the room — with Train anyway always available, because Hugging Face publishes no quota endpoint and the ceiling is an estimate. Settings ▸ Training gained a Hugging Face storage card that lists every lds-base-* cache with its size, the local file it mirrors and the run that last used it, and deletes them one by one or all at once — warning you when a cache is the last copy left. And if a run hits the wall anyway, it now says so in plain words and keeps the pod so the checkpoint is recoverable.',
    to: '/settings/training',
  },
  {
    id: '2026-08-03-seedvr2-tile-and-vae-settings',
    date: '2026-08-03',
    title: 'SeedVR2 upscaling now fits smaller cards — tile size is a setting',
    blurb:
      'The fidelity upscaler used to hold one 1024 px tile at a time whatever your GPU, which is where a large upscale ran out of memory on an 8 GB card. Settings ▸ Image engines now has a Tile size: lower it to 768 or 512 and the same 4K upscale fits, at the cost of a few more seams. It also sizes the model\'s own tiled encode/decode, so it lowers memory use even without the optional tiling node pack. Two more dials came with it — where automatic tiling switches over, and which VAE file to load when yours is named something the automatic search cannot recognise. Defaults are unchanged, so nothing moves unless you touch it. Thanks to SurpassHR (GitHub) for asking for these knobs alongside the engine itself.',
    to: '/settings/engines',
  },
  {
    id: '2026-08-03-canvas-usable-on-a-phone',
    date: '2026-08-03',
    title: 'The LoRA Canvas is finally usable on a phone',
    blurb:
      'The board had never had a small-screen pass, and it showed. Opening 🎨 Generate on a tablet-width window turned it into a fixed side drawer that took more than half the screen and left a sliver of the very board you were picking checkpoints from — so that panel, the run details, the compare view and the image gallery now stay full-width sheets right up to a real desktop, and each one closes with a thumb-sized ✕ instead of a 14-pixel glyph. The zoom, Fit and Tidy up buttons are 40 px on touch, where a miss used to land on the board and pan it. The ✓ box that adds a checkpoint to a run no longer shrinks with the zoom — at the level the board opens on it had become a five-pixel square, on the one control the whole generate flow goes through. And the list of what the board can be told to do, which was simply hidden below laptop width, is now one tap away with the touch gestures spelled out.',
    to: '/canvas',
  },
  {
    id: '2026-08-03-multilora-steps-and-cfg',
    date: '2026-08-03',
    title: 'Set the steps and CFG with several LoRAs selected — compare or blend',
    blurb:
      'With two or more LoRAs checked, the Test Studio gave you no way to choose the number of steps or the CFG: the setting was not greyed out, it simply was not there, and every image ran on the family default no matter what you wanted. It is there now, in the same panel as the strengths, with the same ladders the single-LoRA studio and the canvas already used — including the SDXL second pass. It stays available in 🧬 Blend, where the strength sweep disappears but the render settings do not, and the cell counter includes them so you can see what a sweep will cost before you launch it.',
    to: '/studio',
  },
  {
    id: '2026-08-03-studio-prompt-batch',
    date: '2026-08-03',
    title: 'Run several saved prompts in one go, on both generation screens',
    blurb:
      'The list of saved prompts could only be replayed one at a time: pick a prompt, launch, wait, pick the next one. Every card now has a tick box. Tick three and the launch renders all three with the same checkpoints, the same settings and the same seed — which is what makes them comparable — in a single run the GPU works through by itself. The counter and the button say how many images that is before you click, and ticking nothing leaves the screen exactly as it was. It works the same way in the dataset Test Studio and in "Generate from the board" on the canvas, because both screens show the same list.',
    to: '/studio',
  },
  {
    id: '2026-08-03-restart-no-longer-kills-a-live-cloud-run',
    date: '2026-08-03',
    title: 'Restarting the app no longer kills a cloud run that is training fine',
    blurb:
      'When the app restarted, it picked the run back up and asked vast.ai whether the pod still existed. If that one answer came back without the pod in it — which happens, and means nothing — the run was declared dead about ten seconds later, and the "stop" that followed reached the pod that was still training and ended it. A run at step 825 of 3000 was lost that way, with the hour already paid. Now the pod itself is asked: a pod that answers is a pod that exists, whatever the marketplace says, and silence has to last minutes before the run is given up. If it truly cannot be reached, no stop is sent to a machine we could not talk to, and the pod is kept so the result stays recoverable.',
    to: '/cloud',
  },
  {
    id: '2026-08-03-canvas-opens-again-hotfix',
    date: '2026-08-03',
    title: 'The LoRA Canvas opens again — v2026.08.03 broke it for a few minutes',
    blurb:
      'If you updated to v2026.08.03 in the short window it was live, the Canvas page crashed on load (“An unexpected error occurred”). The provenance-edges feature read a value before it existed. Fixed — nothing else in that release was affected, and no data was touched.',
    to: '/canvas',
  },
  {
    id: '2026-08-03-canvas-blend-provenance',
    date: '2026-08-03',
    title: '🧬 A blended picture now shows every checkpoint it came from',
    blurb:
      'A picture pinned on the LoRA Canvas was linked back to one checkpoint — but a 🧬 Blend loads several, often from different datasets, so the board was showing one parent out of two or three. Violet lines now join a blended picture to every source it was made from, across lanes, next to the existing indigo training lineage. When a source is no longer on the board — its run deleted, its dataset unticked — no line is invented: the picture says “1 of 2 sources is not on the board” instead. Blends made before this update keep their images and simply have no lines to show.',
    to: '/canvas',
  },
  {
    id: '2026-08-03-seedvr2-tiling-is-a-choice',
    date: '2026-08-03',
    title: 'Tiled upscaling is now the default — it keeps more detail, not just less VRAM',
    blurb:
      'Yesterday tiling only kicked in when a frame would not fit on your card. SurpassHR (GitHub #32) re-tested it and sent the source renders: side by side, the full-frame result does not just soften fine texture, it rewrites it — short dense stubble comes back as long smeared strands. A tile is upscaled at the size the model works well at, while a whole 4K frame spreads its capacity over four times the surface. That made the old rule backwards: the bigger your GPU, the less often you got the better picture. So with the tiling node pack installed, large upscales are now tiled by default, and Settings ▸ Image engines lets you choose — tile when it helps (recommended), always tile large frames, or never. Nothing is tiled below roughly 1536 px on the short edge, where the model already works at a good size and a grid would only add seams.',
    to: '/settings/engines',
  },
  {
    id: '2026-08-03-bank-suggests-one-person-folders',
    date: '2026-08-03',
    title: '👤 The bank now spots your one-person folders — suggested automatically, you confirm',
    blurb:
      'Telling the bank a subfolder holds a single person saves the whole face pass on it, but you had to know which folders those were. It now samples them and asks: a folder that looks consistent gets a 👤? in the Subfolder picker and offers "Looks like one person (15/15 sampled) — assert?", and a folder holding several people says that too. Nothing is ever grouped on its own — the suggestion is an offer, confirming stays your click, because a wrong grouping made silently is one you would never think to look for. It runs free at the end of 👤 Group by person (the embeddings are already cached, so covering every folder costs no inference), or on demand with 🔎 Scan folders, which tells you what it will cost and what it did not reach. A suggestion expires by itself when the folder gains or loses images.',
    to: '/bank',
  },
  {
    id: '2026-08-03-medium-runs-itself-after-score',
    date: '2026-08-03',
    title: '🎨 Medium classifies itself at the end of ✨ Score',
    blurb:
      'Sorting a bank into photo / anime / 3D render / illustration reads the embeddings ✨ Score already computed — no GPU, seconds for 23 000 images. It was still a second button you had to know about, so banks sat there holding the data for the answer without the answer. Score now finishes the job and says so on its own line ("· 🎨 Medium: 812 classified"), or names the reason it could not. The 🎨 Classify medium button stays for re-running it alone, and the automatic pass never re-judges an image that already has a verdict.',
    to: '/bank',
  },
  {
    id: '2026-08-03-export-keeps-original-bytes',
    date: '2026-08-03',
    title: 'Big datasets stop bloating on their way to training',
    blurb:
      'Preparing a dataset for training used to re-encode every picture to a lossless PNG, even when nothing needed changing. On a 6 200-image style dataset that turned 3.6 GB of photos into 24 GB and took half an hour before training could even start — long enough to fill a disk and fail with a bare "no space left on device", and long enough that any restart in between killed the run. Pictures the trainer already reads as they are now go through untouched: same dataset, 3.6 GB, a couple of minutes. Anything that genuinely needs converting — a photo with a rotation tag to bake in, an unusual colour mode, a format the trainer does not read — still is. And if the disk really cannot hold the export, you are now told before it starts, with the size it needs and the space you have, instead of finding out halfway through.',
  },
  {
    id: '2026-08-03-canvas-group-bar-reachable',
    date: '2026-08-03',
    title: 'Groups of pinned images can be moved and closed again',
    blurb:
      'A side-by-side group on the LoRA Canvas could end up impossible to move AND impossible to close, with no way to tell why. Its title bar — which holds the ⠿ grip, Export grid and ✕ — is drawn just above the strip, and any picture pinned over that space silently took the clicks meant for it. The bars are now drawn above every picture, so they always answer; and ✦ Tidy up and 📌 Pin all know that space is taken, so they stop dropping a picture there. It showed up most on a zoomed-out board, where the bar is twice as tall.',
    to: '/canvas',
  },
  {
    id: '2026-08-03-watermark-detector-extra',
    date: '2026-08-03',
    title: 'Finding watermarks in a big bank now takes minutes instead of hours',
    blurb:
      'Until now, 🚩 Find watermarks asked the vision model, in words, "is there a watermark here?" once per image — about 1.7 seconds each, which is fifteen hours on a 30 000-image bank for a question whose answer is usually no. There is now an optional extra (Setup ▸ Quality tools ▸ Watermark detector) that answers it with a small classifier instead, in about 0.14 seconds, and a second model marks where the logo sits so ✂ Crop and 🧽 Inpaint still have something to work on. It does not need Ollama at all, so a machine with no vision model can scan too. On a hand-labelled sample of 110 images from a real bank it flagged none of the 55 clean ones and 54 of the 55 marked ones — slightly better than the vision model on the same images, which also missed one. Install nothing and nothing changes: the vision model keeps doing exactly the same job, slower. The flag score is tunable in Settings ▸ Captioning & quality.',
    to: '/setup',
  },
  {
    id: '2026-08-03-canvas-image-controls-see-through',
    date: '2026-08-03',
    title: 'Pinned pictures are no longer hidden by their own buttons',
    blurb:
      'Hovering a picture pinned on the LoRA Canvas dropped an opaque black band across its top and a black block over its corner — the controls covered the very thing you were pointing at. 🔍, ✕ and ⬇ are now separate rounded pills over a blur, and the step label is a small tag instead of a full-width band, so the image shows through between them. The glyphs went white too: they were mid-grey, which reads on the app’s dark chrome but vanishes on a bright render.',
    to: '/canvas',
  },
  {
    id: '2026-08-03-canvas-filter-opens-folded',
    date: '2026-08-03',
    title: 'The LoRA Canvas opens on the board, not on its filter',
    blurb:
      'The Datasets filter opened expanded every time you loaded the canvas, and on a library of any size its checkbox list pushed the board itself below the fold — so the first thing you did on the page you came to look at was scroll past a filter. It now opens folded at every width, with the same summary on the button (“3 of 7 · 12 runs shown”), so nothing is hidden. If you unfold it, it stays unfolded next time.',
    to: '/canvas',
  },
  {
    id: '2026-08-03-blend-weight-sweep',
    date: '2026-08-03',
    title: '🧬 Tick several weights and blend every combination in one run',
    blurb:
      'Finding the right balance between two LoRAs meant launching, looking, moving a slider, launching again. Each LoRA in a 🧬 Blend now has weight boxes next to its slider: tick 0.6 and 0.8 on one, 0.4 and 1.0 on the other, and the launch renders all four combinations in a single run — each image labelled with the pair that made it, and lined up side by side in the stack view so you can just pick. The button says the cost before you commit (“4 weight combinations → 4 images, about 1 min”) and warns past 24, without ever refusing: it is your machine. Tick nothing and the slider governs, exactly as before. Works in the Test Studio and from the LoRA Canvas board.',
    to: '/studio',
  },
  {
    id: '2026-08-03-concept-sdxl-refusal-tells-the-truth',
    date: '2026-08-03',
    title: 'A Concept dataset on SDXL is told the truth instead of sent hunting',
    blurb:
      'Concept captions are only ever written as prose, and the prose/booru selector is hidden on concept datasets — so a Concept dataset trained on a booru-native SDXL checkpoint was always stopped, and then told to “re-caption in Booru tags mode”: a control that does not exist for it. You could look for it forever. The refusal now says what is actually true — no booru concept captioner exists yet — and names the two paths that work: train the concept on a prose family (Z-Image, Krea 2, FLUX.1, FLUX.2 Klein, Anima), or force the launch knowing what a booru-native base loses on prose. Character and Style datasets are untouched; both reach booru captions normally.',
    to: '/datasets?section=training&panel=launch',
  },
  {
    id: '2026-08-03-pinokio-update-hands-back',
    date: '2026-08-03',
    title: 'Installed from Pinokio? Updates now tell you the right three clicks',
    blurb:
      'Pinokio starts and stops the app itself, so pressing "Update & restart" in Settings would have relaunched the server in a window Pinokio no longer follows — it would show the app as stopped while the old one still held the port, and Start would have opened a second one on the same datasets. The Updates card and the update banner now show Stop, Update, Start instead of that button, while still telling you how many commits behind you are. Every other install shape is unchanged.',
    to: '/settings/maintenance',
  },
  {
    id: '2026-08-03-seedvr2-tiled-highres',
    date: '2026-08-03',
    title: 'Big SeedVR2 upscales no longer have to fit on your card in one piece',
    blurb:
      'Upscaling a whole frame at once needs the whole frame in VRAM, and past a certain size that simply fails — with a CUDA out-of-memory error in a log, which is a terrible way to find out. Two things change. The app now tells you, before it starts, roughly how many megapixels your GPU is good for in one pass. And if you install the Comfyui_TTP_Toolset node pack in ComfyUI, anything bigger is automatically cut into overlapping tiles, upscaled tile by tile and blended back together — so a 4K result works on a card that could not hold it whole. Without the pack nothing breaks: upscales still run, they are just capped. Tiled workflow and the measurement behind it contributed by SurpassHR (GitHub #32).',
    to: '/setup',
  },
  {
    id: '2026-08-03-dataset-passes-survive-a-deleted-image',
    date: '2026-08-03',
    title: 'A long pass no longer dies if you delete an image while it is running',
    blurb:
      'Captioning, watermark detection and cleaning, framing, short captions — these walk your whole dataset for minutes or hours, and the grid stays clickable the whole time. Delete a bad tile, press ⏹ Stop, or start an improve while one of them is working, and the pass could die on the spot with a cryptic database error, losing the work it had already done on every other image. It now skips whatever disappeared and carries on. ⏹ Stop also reports the number of generations it really cancelled rather than the number it attempted, so the count and the grid can no longer disagree.',
    to: '/datasets',
  },
  {
    id: '2026-08-03-bank-coverage-sees-more',
    date: '2026-08-03',
    title: 'The bank’s Coverage advice can now see that your images all look the same',
    blurb:
      'Coverage advice used to read only labels — framing, person and style clusters, resolution — and labels have a blind spot: they cannot tell two hundred near-identical shots from two hundred different ones, and they say nothing about outfits, lighting or camera angle. It now also reads two things you may already have on disk. Visual spread uses the embeddings ✨ Score cached to report how alike the pool actually looks ("91% average similarity — a set this repetitive teaches one look"); the bands come from measuring real banks, where an ordinary one sits near 65% and an image plus its nearest neighbours lands at 79-90%. Caption variety uses the captions 🏷️ wrote to say which camera views, lightings, settings, outfits and expressions your set never mentions — the same reading the dataset Coverage panel does. Both are honest about their limits on the panel itself: without ✨ Score the spread says "Not measured" rather than "varied", and the caption read looks at words rather than pixels, so a profile nobody described is invisible and "not smiling" still counts as a smile. Still advice only — nothing is kept or rejected.',
    to: '/bank',
  },
  {
    id: '2026-08-03-canvas-blend',
    date: '2026-08-03',
    title: '🧬 Blend two LoRAs into one image, straight from the board',
    blurb:
      'Ticking several checkpoints on the LoRA Canvas used to mean one pass each. A new ⚖ Compare / 🧬 Blend toggle lets you load them all into the SAME generation instead, each on its own weight slider, with every dataset\'s trigger word listed before you launch rather than injected behind your back. Identity + style and identity + concept are where it pays off — two identities blend into a hybrid person, which the panel now tells you up front. The mode is called Blend everywhere now: the Test Studio\'s 🧬 Combine toggle is the same thing and now says Blend too. Nothing you saved changes — only the word.',
    to: '/canvas',
  },
  {
    id: '2026-08-03-bank-single-person-folder',
    date: '2026-08-03',
    title: '👤 Tell the bank a folder is already one person — and skip the pass',
    blurb:
      'Scraped folders are usually one person each, and 👤 Group by person did not know it: one face embedding per image to rediscover what the folder name already said. Scope the grid to a subfolder and click "Single person here" — every image is grouped instantly, with no pass at all, and the next Group by person run skips them entirely. It survives re-scans, adopts images that land in the folder later, and one click undoes it. Not sure? "Check a sample" embeds about fifteen images spread across the folder and tells you whether they look like the same person, at the same threshold the clustering uses — fifteen inferences instead of thousands, and whatever it finds, your call stands.',
    to: '/bank',
  },
  {
    id: '2026-08-03-caption-mismatch-names-your-family',
    date: '2026-08-03',
    title: 'The caption-mismatch warning names YOUR model family, not Z-Image',
    blurb:
      'If your captions did not match the family you picked, the app refused the launch with a message that always said “this Z-Image dataset… but Z-Image expects prose” — even when you were training Krea 2, FLUX.1 or FLUX.2 Klein. The sentence was frozen into the code while the truth depends on the family. It now names the one you actually chose, so the advice matches what you are looking at.',
    to: '/datasets?section=training&panel=launch',
  },
  {
    id: '2026-08-03-anima-hybrid-captions',
    date: '2026-08-03',
    title: 'Anima takes booru tags too — the app stops calling them a mistake',
    blurb:
      'Anima is a hybrid-prompting model: booru tags and natural language are both first-class on it. The app only knew half of that and treated a booru-captioned Anima dataset as a caption mismatch, so a perfectly valid dataset had to be force-launched past a red warning. Both forms now train without a warning and without forcing, the caption-style selector says so on Anima, and prose stays the preselected default. Every other family keeps its guard exactly as it was — SDXL still refuses prose. Prompted by a correction from Witty_Mycologist_995 (Reddit).',
    to: '/datasets?section=captions&panel=generate',
  },
  {
    id: '2026-08-03-improve-note-cites-the-setting',
    date: '2026-08-03',
    title: 'The amber “drawn dataset” note now names the setting it came from',
    blurb:
      'Next to Improve, a caution used to announce “This dataset is drawn.” — a verdict the app never actually reached, because it only ever read the subject type you picked. On a photoreal dataset left marked Anime the sentence was simply wrong, with nothing to tell you where it came from. It now says the subject type is set to anime, so when the setting and your images disagree you can see which one to change. The advice itself is unchanged.',
    to: '/datasets?section=images&panel=review',
  },
  {
    id: '2026-08-03-folder-picker-paste-a-path',
    date: '2026-08-03',
    title: 'You can finally paste a path when choosing a folder',
    blurb:
      'Browse… opened the Windows folder tree from the XP era: no address bar, no Quick Access, and no way to paste — so a path someone sent you, or one you had just copied out of Explorer, had to be clicked down to one folder at a time. It now opens the modern Windows folder picker, with an address bar you can paste into and your usual shortcuts down the side. The in-app browser — the one you get over the LAN, on a tablet, or on Linux, where no Windows dialog exists — gained the same thing: a path box at the top, Enter to jump. If the modern dialog cannot be used on your machine, the old one still opens rather than nothing at all.',
    to: '/datasets',
  },
  {
    id: '2026-08-03-bank-tag-chips',
    date: '2026-08-03',
    title: 'Click an image’s 🏷️ tags to find the other images that share them',
    blurb:
      'Every captioned tile now has a clickable 🏷️ badge. It opens that image’s caption as chips — woman, red, dress, balcony — and ticking them narrows the grid to the images whose captions mention them. It is the readable cousin of 🎯 Similar to selected: that one matches the overall look and cannot say why, this one matches attributes you picked and shows you exactly which. Several chips mean AND, so each tick narrows further, and chips match as whole words — “car” will not drag back “scarf”.',
    to: '/bank',
  },
  {
    id: '2026-08-03-bank-medium-chips',
    date: '2026-08-03',
    title: '🎨 Split a mixed dump into photos, anime, 3D renders and illustrations',
    blurb:
      'A new chip row on the Bank sorts your images by what they are MADE of — a photograph, an anime drawing, a 3D render, a painting — and it costs nothing to run: it reads the embeddings ✨ Score already computed, so no image is looked at twice and the GPU stays free. It is not the same question as 🔎 Origin: a photorealistic AI portrait is 🤖 AI and 📷 Photo at once. Where it cannot tell, it says “Unsure” instead of guessing, and the row prints how big that pile is — measured on a real 23 500-image bank, its photograph verdicts were right 90 out of 90 times, and it refused to call every 3D render it saw rather than invent one. A photo of somebody cosplaying an anime character is the confusion that forced that caution.',
    to: '/bank',
  },
  {
    id: '2026-08-03-bank-angle-chips',
    date: '2026-08-03',
    title: '⤢ Filter a bank by head angle — frontal, three-quarter, profile, from behind',
    blurb:
      'The 🎭 Person groups pass has always estimated where a head is pointing, used the number once and thrown it away. It now keeps it, and a new chip row lets you pull out the frontal shots, the three-quarters, the profiles and the back views — the fastest way to see that a set is fifty selfies facing the camera. You can sort by it too. Two limits are printed rather than hidden: profiles are under-counted (a head turned that far often defeats the face detector), and “from behind” needs the 📐 Framing pass as well, because “no face” on its own is also what an empty landscape looks like. Banks scanned before this can measure their missing angles from the row itself — it tells you how many and roughly how long, and does nothing until you click.',
    to: '/bank',
  },
  {
    id: '2026-08-03-caption-length-preset',
    date: '2026-08-03',
    title: 'Ask for shorter — or longer — captions in one click',
    blurb:
      'Captions came out at whatever length the vision model felt like, and the only way to steer that was to hand-write it into Extra instructions on every dataset. ⚙️ Options on the Captions panel now has a Caption length dial: Concise aims for one short sentence, Detailed for several, Standard leaves the prompt exactly as it was. On 18 real photos with the shipped default vision model that came out at a median of 24 words for Concise, 88 for Standard and 126 for Detailed — a target the model follows loosely, not a word cap, and your own numbers will differ. Concise stays full prose, so a short-captioned dataset still trains on prose families without forcing past the caption-style check, and it is a different axis from the long + short dual captions — use both if you want. Saved per dataset, offered per run on the image bank\'s caption pass too, and your Extra instructions still get the last word. Suggested by djpraxis (Reddit).',
    to: '/datasets?section=captions&panel=generate',
  },
  {
    id: '2026-08-03-review-opens-instantly',
    date: '2026-08-03',
    title: '▶ Review opens instantly, even on a 20 000-image bank',
    blurb:
      'Pressing ▶ on a tile used to sit there for seconds on a big bank — longer still with a sort active. It was building its list of images the slow way: asking the server for the whole grid, 500 rows at a time, and keeping only the identifiers. On a 22 940-image bank that was 46 requests and 16 MB to end up with 23 000 numbers. It now asks once, for the numbers alone: 3.8 s down to 44 ms, on the same bank. "Select all in filter" took the same shortcut and got the same gain.',
    to: '/bank',
  },
  {
    id: '2026-08-02-klein-model-paths',
    date: '2026-08-02',
    title: 'Point the app at your models with a full path, from anywhere',
    blurb:
      'Every Klein model field — diffusion model, text encoder, VAE, and now the consistency LoRA, which was config-only until today — takes a full absolute path as well as a ComfyUI-relative name, and generation-LoRA preset rows do too. A path inside one of ComfyUI\'s model folders is converted to what the loader needs; a path from anywhere else (Downloads, a Hugging Face cache, another drive) is hardlinked into an lds-pinned/ folder for you, so a 10 GB weight stays where it is instead of being copied or moved. A native bf16 model also runs at full precision at last: both Klein graphs hardcoded an FP8 loader setting that quantized full-precision weights on load without telling anyone — the canonical download is FP8, so a stock install renders exactly as before. Contributed by socrasteeze (GitHub).',
    to: '/settings/engines',
  },
  {
    id: '2026-08-02-klein-model-file-pins',
    date: '2026-08-02',
    title: 'Tell the app exactly which Klein model files to load',
    blurb:
      'Settings → Image engines has three new optional fields — diffusion model (UNET), text encoder, VAE — that name the exact files the Klein graph loads. Empty keeps today\'s automatic detection. This is the way out of "the app says the model is missing and I am looking right at it": detection is deliberately narrow, so it declines any file it cannot confidently name — a UNET outside a "klein" folder, a weight renamed on download — and a declined file was reported missing. A pinned file resolves, and you get the real verdict on it, including "present but unreadable" for a corrupt download, with the delete-and-re-download action attached. A pin that is not on disk falls back to detection with a badge instead of blocking generation. Contributed by socrasteeze (GitHub).',
    to: '/settings/engines',
  },
  {
    id: '2026-08-02-bank-passes-survive-a-deleted-image',
    date: '2026-08-02',
    title: 'Deleting a bank image while a long pass is running no longer throws the whole pass away',
    blurb:
      'Score, watermark, framing, faces, captions and the quality scan all walk thousands of images for minutes or hours. If an image disappeared underneath one of them — most often because you deleted the bank, or its rejected images, while it was still working — the pass died on the spot with a cryptic error, and every image it had already analysed was lost with it. Now the vanished image is simply skipped and the pass finishes; the line at the end tells you how many were skipped, so a pass never quietly claims it analysed more than it did.',
  },
  {
    id: '2026-08-02-dataset-coverage-panel',
    date: '2026-08-02',
    title: 'Your dataset can hit a green composition target and still be twenty-five copies of the same shot — a new panel says so',
    blurb:
      'The Composition bar counts face / bust / body / back against a target, so it goes green on a set that is entirely front-on, in one outfit, under one light. Those are not things it counts, and that set trains a LoRA that can only reproduce that one look. The new 🔍 Coverage panel, folded under that bar, is the second question: it reads the captions you already generated and reports what your set never shows — no profile views, only one outfit, eye level only, never outdoors — with a plain "generate or import more: profile, three-quarter" line under it. It is honest about being a keyword read of your captions rather than a look at the pixels: with no captions it says so instead of drawing empty bars, under five captions it refuses to judge, and it never keeps, rejects or changes anything. Style and concept datasets are judged on their own axes — "one outfit" is not a defect when the outfit is not what you are teaching.',
    to: '/datasets?section=add',
  },
  {
    id: '2026-08-02-pinokio-one-click-install',
    date: '2026-08-02',
    title: 'Install LDS in one click from Pinokio',
    blurb:
      'LDS now ships a Pinokio launcher. In Pinokio, paste the repository URL under "Download from URL", click Install, then Start — the Python environment, the requirements and the browser tab are handled for you, with no terminal, no Python install and nothing added to your PATH. Only the core app is set up that way: ComfyUI, Ollama, ai-toolkit and the ML helpers are still connected from Setup, so nothing heavy is downloaded behind your back. Update from Pinokio\'s own Update tab — it fast-forwards the same checkout the in-app updater uses.',
  },
  {
    id: '2026-08-02-terminal-prints-the-address',
    date: '2026-08-02',
    title: 'The terminal finally tells you where the app is',
    blurb:
      'Starting the server by hand (python backend/run.py) printed no address at all — the usual "Running on http://..." line was being swallowed into data/app.log — so you had to guess the port, which is exactly the case where it may have moved because 5050 was taken. The console now prints "[LDS] Ready on http://…" once the app really answers, whichever way you launched it.',
  },
  {
    id: '2026-08-02-ref-edit-keep-error',
    date: '2026-08-02',
    title: 'Fixing a failed reference edit no longer errors twice',
    blurb:
      'When keeping an edited reference failed, the app hit a second error while reporting the first, so you got a blank failure instead of the reassurance that your previous reference was left untouched. It now says exactly that — thanks to socrasteeze (GitHub) for spotting it.',
  },
  {
    id: '2026-08-02-bank-text-search-push-down',
    date: '2026-08-02',
    title: 'Text search can finally get rid of what you do not want',
    blurb:
      'Typing "a woman without a hat" in 🔤 Find by text never worked — the search engine ignores the word "without", so you got hats, with nothing to tell you it had misfired (measured on a real bank: "without a bikini" returned 60% bikinis against a 10% base rate). There is now a Push down field for the trait you are trying to avoid, plus a -hat shorthand in the query itself, and typing "without" offers to move it there for you. It re-ranks rather than filters: matching images sink instead of disappearing, and after each search the panel tells you what it actually achieved on your bank — including "this changed nothing", which is the one outcome you could never have spotted.',
    to: '/bank',
  },
  {
    id: '2026-08-02-upscale-candidate-visible',
    date: '2026-08-02',
    title: 'You can finally see that an upscale is waiting for you',
    blurb:
      'An upscale never touches your original: it arrives as a separate tile you keep or reject. Which also meant that from the image you had just sent, nothing appeared to happen — so the pass got re-run on images that already had a result waiting, paying GPU time for a duplicate. The source tile now says it, both while the result is rendering and once it is ready to review. And the candidate names the engine that actually made it, instead of always crediting Klein.',
    to: '/datasets',
  },
  {
    id: '2026-08-02-seedvr2-in-the-lightbox',
    date: '2026-08-02',
    title: 'Pick your upscaler while you are looking at the image',
    blurb:
      'The full-screen inspector only ever offered the Klein pass. That is the one place where the choice matters most: on a drawn dataset the panel already warns you that Klein’s instruction pulls anime skin towards realism, and the pass that does not do that was two screens away in the selection toolbar. Both engines are now side by side in the inspector, each saying what it does to the original — and that warning stays under Klein alone, because SeedVR2 sends no instruction at all.',
    to: '/datasets',
  },
  {
    id: '2026-08-02-bank-exclude-words',
    date: '2026-08-02',
    title: 'Hide the bank images you have already handled, and work the rest as a checklist',
    blurb:
      'The bank search box narrows the grid TO a word. Next to it there is now a 🚫 Exclude box that does the opposite: it hides every image whose caption or file name carries one of the words you type (comma-separated, so "logo, watermark" is one gesture). It composes with the search and every chip — "dresses that are not red" is one line — and it travels with the filter, so Select all, ▶ Review and the curation picks never hand back an image you just hid. Images with no caption are never hidden: they are exactly what the checklist is looking for.',
    to: '/bank',
  },
  {
    id: '2026-08-02-bank-sort-every-measure',
    date: '2026-08-02',
    title: 'Sort a bank by anything its passes measured — and it remembers the order you chose',
    blurb:
      'The Sort menu covered resolution, aesthetics and sharpness. It now covers every figure the passes persist, both ways: file size, NSFW likelihood, noise, contrast, detail, letterbox bars, JPEG quality and face confidence — grouped by the pass that produces them, so a greyed-out section tells you which pass to run. A chip only ranks what crosses its threshold, so "the noisiest of the ones I am keeping" was a question nothing could answer. Each bank also remembers its own order now, so the dump you review by sharpness opens that way tomorrow.',
    to: '/bank',
  },
  {
    id: '2026-08-02-writes-refuse-an-unreadable-body',
    date: '2026-08-02',
    title: 'A save that cannot be read now says so instead of answering OK and storing nothing',
    blurb:
      'If you drive the API yourself — a script, a curl command, your own tooling — a request body the app could not parse used to be treated as an empty one: it answered 200 OK, rewrote your settings, and the value you sent was simply gone, with nothing in the log. The most common cause is a raw Windows path, since a lone backslash is not valid JSON. Those requests are now refused with the real parse error and the escaping to use, and nothing on disk is touched. The app\'s own screens were never affected.',
  },
  {
    id: '2026-08-02-ollama-fence-waits-and-offers-the-unload',
    date: '2026-08-02',
    title: 'When another tool is holding the Ollama model, LDS now waits for it — or takes it, if you say so',
    blurb:
      'If you run another AI tool on the same machine, ✨ Enhance and 🔎 Describe would sometimes refuse with "a local Ollama model is already in use outside LDS". That refusal was right — LDS must never unload a model another app is using — but it was also a dead end: it told you to go unload it by hand, then forgot what you had asked for. Now the refusal carries its own way out. LDS keeps watching, and the moment the model is free (Ollama unloads an idle one by itself after a few minutes) your Enhance or Describe starts again on its own, with the prompt and the image you had. If you would rather not wait, one click on "Unload it and continue" evicts the other model and resumes — and nothing evicts it without that click. One more thing fixed while we were here: restarting LDS while its OWN model was still warm used to make it mistake that model for a stranger\'s and block itself out of it.',
    to: '/studio',
  },
  {
    id: '2026-08-02-anima-shows-its-own-base',
    date: '2026-08-02',
    title: 'Anima no longer says it trains on a Z-Image base',
    blurb:
      'Picking the Anima family left the Base selector — and the summary line under the Train button — announcing "Official - Z-Image-Turbo", and offered your Z-Image merges as if an Anima run could load them. Anima now names its own base, Anima-Base, and offers nothing else. Your runs were never affected: the launcher always used the official Anima base and ignored the foreign selection, so this was the panel lying about a run it was going to get right. The cloud button on Anima also stops pretending: it says Anima is local-only until the pod image is verified, instead of enabling itself and being refused after the click.',
    to: '/datasets?section=training&panel=advanced',
  },
  {
    id: '2026-08-02-seedvr2-results-come-back',
    date: '2026-08-02',
    title: 'SeedVR2 upscales now actually appear — and the ones you already ran are recovered',
    blurb:
      'The first SeedVR2 build rendered correctly and then dropped the result on the floor: ComfyUI finished the image, the job was marked done, and the candidate stayed blank forever with nothing in the log, because nothing had failed. The finished image is now attached to its tile. Any upscale you already ran and never saw is picked up automatically the next time the app starts — the image is still there, it just never made it home. Caught on a real run the day it shipped.',
    to: '/datasets',
  },
  {
    id: '2026-08-02-comfyui-recovery-anywhere',
    date: '2026-08-02',
    title: 'A ComfyUI job that died in a past session no longer blocks you forever',
    blurb:
      'When a generation stops without a known outcome — ComfyUI killed, the machine out of disk — LDS pauses everything on purpose, so it never abandons work that might still be running on the GPU. The catch: there was no way to resolve it. Opening the dataset that owned the job showed nothing either; you were simply refused, everywhere, for as long as the record lasted. Now LDS checks first: if ComfyUI is back and no longer knows that job, it clears it on its own and just tells you. When it cannot prove that — ComfyUI unreachable, the job still running there, or a submission it cannot identify — a banner appears wherever you are, names the dataset and the variation and how long it has been stuck, and clears it in one click once you confirm the restart. A job that is genuinely still running is never touched.',
  },
  {
    id: '2026-08-02-seedvr2-upscaling',
    date: '2026-08-02',
    title: 'A second way to upscale — one that does not repaint your images',
    blurb:
      'Klein’s ✨ Upscale & improve re-renders detail from a prompt: it rescues a soft photo, and it can move skin tone and colour along the way — which is the wrong trade when the exact look is what you are training on. SeedVR2 is now the other option: it resolves detail at a higher resolution and leaves the content alone. Pick either one straight from the bulk actions on a selection (each button says what it does to the original), or set your default for the single-image pass in Settings ▸ Image engines. Setup ▸ ComfyUI downloads the two models (~3.9 GB) on a click and tells you how to add the node pack. Requested by SurpassHR (GitHub #32).',
    to: '/settings/engines',
  },
  {
    id: '2026-08-02-docker-first-boot-opens-setup',
    date: '2026-08-02',
    title: 'A brand-new Docker install now opens Setup instead of hiding it',
    blurb:
      'On the Docker GPU install, ComfyUI comes bundled — which made the app look fully configured on its very first boot, so it dropped you straight into Datasets and never offered the wizard. Meanwhile the launcher window sat waiting up to fifteen minutes for an Ollama choice that only Setup can ask for. A first boot that still owes that choice now opens Setup. Nothing changes for an install that has been seen working: it is still never interrupted, and once you have picked an option you are not asked again.',
    to: '/setup',
  },
  {
    id: '2026-08-02-dataset-upload-is-visible',
    date: '2026-08-02',
    title: 'You can now watch the dataset go up to the rented machine — and it stops billing if it does not',
    blurb:
      'Sending a dataset to a cloud pod was the one step that reported nothing at all: a big set could be uploading perfectly, or wedged and going nowhere, and both looked exactly the same for hours while the machine billed. The launch card now shows the files and gigabytes as they land, so a slow upload is visibly a slow upload. A large dataset is still allowed to take as long as it needs — nothing was made stricter — but if NO data at all reaches the machine for 25 minutes the run is given up and the pod released, with a message that says the dataset never arrived rather than leaving you to guess the training crashed. The delay is yours to set in Settings → Training, and full-model runs no longer claim to be "uploading the model" while they are still uploading the dataset.',
    to: '/cloud',
  },
  {
    id: '2026-08-02-docker-install-and-update-actually-work',
    date: '2026-08-02',
    title: 'The Windows Docker install now starts, restarts and updates',
    blurb:
      'A real end-to-end run of the one-click Docker install found four faults that made it unusable, and all four are fixed. The launcher now finds a host port that is genuinely free instead of trusting Docker to skip a busy one, so an existing ComfyUI on 8188 (or anything else in the way) no longer stops the install dead — and because it publishes one fixed port, the address stays the same across restarts instead of quietly moving. Launching a second time works: rewriting the small settings marker used to fail on every run after the first, which also meant one-click update could never finish. On Windows the bundled ComfyUI no longer restart-loops on its own folders. When Docker does refuse something you now get the actual reason instead of a PowerShell error about a missing property, and the updater speaks English like the rest of the app.',
  },
  {
    id: '2026-08-02-cloud-launch-is-observable',
    date: '2026-08-02',
    title: 'A cloud launch now tells you what it is doing, and for how long',
    blurb:
      'Renting a GPU takes minutes, and the button used to say "Launching…" for all of them — impossible to tell a normal wait from a dead one. The launch now shows its steps as it walks them (preparing the dataset, searching for an offer, renting and booting the pod, uploading, starting the job) with the time elapsed, on the dataset panel and on the Runs page. While a pod boots you also see how long it is allowed to take, so a machine that never starts ends with a plain explanation instead of a frozen screen — it is released, it stops billing, and launching again picks a different host. A launch can be cancelled from the Runs page like any run.',
    to: '/cloud',
  },
  {
    id: '2026-08-02-enhance-says-why-it-refused',
    date: '2026-08-02',
    title: '✨ Enhance now tells you what to fix when it cannot run',
    blurb:
      'When Enhance could not reach your local model it always said the same thing — that the model returned nothing, and that you should check the model configured in Settings — even when that setting was perfectly correct. The most common cause is different: another tool is already holding a model in Ollama, and the app will not evict it. Enhance now repeats the real reason and the action that clears it, so you stop hunting through Settings for a problem that was never there.',
    to: '/studio',
  },
  {
    id: '2026-08-02-studio-stack-results-view',
    date: '2026-08-02',
    title: 'Find the right weights for a 🧬 stack by comparing your runs side by side',
    blurb:
      'Testing two LoRAs together used to end in a single anonymous column: you could see the image, not what was in it. The Test Studio now shows a stacked run for what it is — every LoRA of the stack with its weight and its trigger word, and one column per run of that same stack, so the version at 1.00 / 0.55 sits next to the one at 1.00 / 0.80 with the changed weights highlighted and the votes counted per column. Reload any variant’s weights into the sliders in one click, and save the winning set with ★: the best setting of a stack is now its weights, not one lone checkpoint.',
    to: '/studio',
  },
  {
    id: '2026-08-02-cloud-only-installs-see-their-checkpoints',
    date: '2026-08-02',
    title: 'Trained in the cloud without ai-toolkit? Your checkpoints show up now',
    blurb:
      'The checkpoint list refused to answer at all unless local training was set up, so an install that only ever trains in the cloud saw an empty panel — its own paid-for cloud saves were there on disk, just never displayed. The list now always answers: cloud saves appear whether or not ai-toolkit is configured, and the local half simply stays empty when there is no local trainer. Deleting a cloud-trained LoRA you had deployed to ComfyUI works from the same install too, instead of being listed but undeletable.',
  },
  {
    id: '2026-08-02-quiet-polls-when-the-server-is-unhappy',
    date: '2026-08-02',
    title: 'A server that is still starting up no longer buries the app in error toasts',
    blurb:
      'Background checks — the progress polls, the setup re-check, the "is ComfyUI up yet?" watcher — already stayed quiet when they could not reach the server at all. They now stay just as quiet when the server answers badly: a container still booting replies "not ready" every few seconds, and that used to become one "Server error. Please try again later." toast per reply until it finished. The screens that depend on those checks still show what went wrong where it belongs, and the moment the server comes back you still get the single "Back online" line.',
  },
  {
    id: '2026-08-01-bank-image-ceiling-raised',
    date: '2026-08-01',
    title: 'A bank now takes 200,000 images, and stops miscounting the ones you deleted',
    blurb:
      'The per-bank ceiling went from 50,000 to 200,000 images. It is also counted correctly at last: images you had deleted from the folder still counted against it, so a folder that once hit the limit could refuse every new file for ever — with a message claiming it "now holds more than 50000 images" when it held a fraction of that. Reaching the ceiling no longer refuses the whole batch either: everything that fits is added and the app tells you exactly how many were left out.',
    to: '/bank',
  },
  {
    id: '2026-08-01-delete-rejected-shows-progress',
    date: '2026-08-01',
    title: 'Watch 🗑 Delete rejected work, and stop it whenever you like',
    blurb:
      'Deleting thousands of rejected files used to freeze the dialog on "Deleting…" for minutes with no count, no way to stop, and no way to tell a slow run from a crashed one. It is an ordinary bank pass now: the progress bar at the top of the bank counts the files as they go, Stop works, and the finished run reports how many left the disk and where they went.',
    to: '/bank',
  },
  {
    id: '2026-08-01-verified-cloud-token-stops-nagging',
    date: '2026-08-01',
    title: 'Full-model training confirms your Hugging Face token instead of re-asking for it',
    blurb:
      'The 80 GB GPU picker used to show the same “configure a token before renting the GPU” notice even when your saved token had just been verified. It now reports the verified delivery namespace, keeps a distinct amber note for a global write token, and shows the setup instructions only when something is genuinely missing.',
    to: '/settings/local-tools',
  },
  {
    id: '2026-08-01-progress-heartbeat-cannot-fail-a-run',
    date: '2026-08-01',
    title: 'A busy database no longer marks a healthy cloud run as failed',
    blurb:
      'A cloud run whose progress text could not be saved — because another long operation held the local database — was recorded as failed with “database is locked” while the rented GPU was still training normally. The progress refresh is now allowed to be skipped and retried on the next poll; only real status changes still stop a run.',
    to: '/cloud',
  },
  {
    id: '2026-08-01-klein-edits-follow-the-instruction',
    date: '2026-08-01',
    title: 'Klein edits follow what you asked again',
    blurb:
      'Every local Klein edit — the ✦ reference edit, variations, regenerations and the small-image rescue — was quietly mixing in a detail LoRA at 0.8 strength, which pulled results away from the instruction you typed. It only started happening once that LoRA became an automatic download, and nothing showed it or could turn it down. Edits now run without it by default, and a new "Enhancement LoRA on edits" setting lets you add it back on purpose.',
    to: '/settings/engines',
  },
  {
    id: '2026-08-01-studio-combine-loras',
    date: '2026-08-01',
    title: 'Load two of your LoRAs in the same test image',
    blurb:
      'Test Studio can now COMBINE the LoRAs you check instead of only comparing them side by side. Switch to 🧬 Combine, give each LoRA its own weight, and they render together in one image with every trigger word injected for you. Mixing families stays blocked — Krea and SDXL LoRAs need different base models — and the message now names which two you picked.',
    to: '/studio',
  },
  {
    id: '2026-08-01-studio-enhance-prompt',
    date: '2026-08-01',
    title: 'Turn a one-line test prompt into a full one',
    blurb:
      'A ✨ Enhance button next to the Test Studio prompt rewrites what you typed into a richer prompt using your local Ollama model — framing, pose, lighting and mood added, identity and trigger words left alone. Without Ollama the button is simply greyed out and tells you what is missing.',
    to: '/studio',
  },
  {
    id: '2026-08-01-setup-checks-in-the-background',
    date: '2026-08-01',
    title: 'Stop re-running Setup every time you come back',
    blurb:
      'Once LDS has seen your install working, coming back — a new tab, a new browser, another machine, a restarted server — drops you straight into the app instead of the wizard. The same checks now run in the background while you work, with one small line in the corner that fades on its own. You are only interrupted when something that used to work has stopped: the warning names what broke, ignores ComfyUI or Ollama simply not running, and takes "that was on purpose" for an answer.',
    to: '/setup',
  },
  {
    id: '2026-08-01-canvas-runs-stay-separate-and-in-epoch-order',
    date: '2026-08-01',
    title: 'Pin two generation runs to the canvas and compare them side by side',
    blurb:
      'Pinning a second run at the same checkpoint no longer folds its images into the first run’s strip — each generation keeps its own strip on the board, so two runs stay two runs. Every strip now reads left to right in training order (500, 1000, 1500…) instead of alphabetically, and an over-cap batch keeps the early epochs rather than an arbitrary slice.',
    to: '/canvas',
  },
  {
    id: '2026-08-01-canvas-shows-the-dataset-reference-face',
    date: '2026-08-01',
    title: 'See who the renders are meant to be, right on the canvas',
    blurb:
      'Each character dataset’s lane on the LoRA Canvas now opens with its reference image, next to the dataset name. Click it to open it full size against the pinned renders. Concept and style datasets are unaffected — they have no reference face.',
    to: '/canvas',
  },
  {
    id: '2026-08-01-choose-how-docker-uses-ollama',
    date: '2026-08-01',
    title: 'Choose how your Docker install uses Ollama — and never get a surprise download',
    blurb:
      'The Docker Setup page now asks one question instead of making you edit files: no Ollama, the one already running on your computer, or an isolated companion container. Nothing large is downloaded to make that choice — the vision model stays absent until you press Pull, which now streams its real progress and can be cancelled mid-download and resumed later. Switching modes keeps your models: only the container LDS itself created is removed, and its data volume survives.',
    to: '/setup?step=ollama',
  },
  {
    id: '2026-08-01-docker-with-your-existing-comfyui',
    date: '2026-08-01',
    title: 'Run the Docker build against the ComfyUI you already installed',
    blurb:
      'start-docker.bat now reuses your existing Windows ComfyUI instead of building a second one: the first launch opens a folder picker, accepts either the ComfyUI folder or the portable folder above it, and reuses your models and outputs in place. start-docker-gpu.bat still builds the fresh isolated ComfyUI for people who want one. Both pick free ports on their own and open the browser while the stack is still warming up.',
  },
  {
    id: '2026-08-01-one-click-docker-update',
    date: '2026-08-01',
    title: 'Update the Docker install by double-clicking one file',
    blurb:
      'update-docker.bat follows the latest published release (pass "main" for the preview channel), downloads that exact commit, and swaps the code in one transaction. Your .env, config, datasets, run and basedir folders, bank images and Ollama data are never touched. If the new version fails to come up healthy, the previous code is restored and restarted automatically — and a git checkout is never reset or merged behind your back, it just tells you to run git pull --ff-only.',
  },
  {
    id: '2026-07-31-community-tested-krea-zimage-presets',
    date: '2026-07-31',
    title: 'Start Krea 2 Raw and Z-Image Turbo from five community-tested recipes',
    blurb:
      'Training now includes five source-linked community presets for Krea 2 Raw character, fast LoKr character, compact style, a reported 16 GB concept setup, and Z-Image Turbo character. Each recipe carries its own image-count, step, optimizer, memory and network settings; switching family, dataset kind or variant can no longer leave its hidden settings active, and Conv checkpoints refuse an incompatible continuation instead of failing deep inside training.',
    to: '/datasets?section=training&panel=advanced',
  },
  {
    id: '2026-07-31-generation-stop-recovers-gpu-lock',
    date: '2026-07-31',
    title: 'Stop generation without stranding the GPU',
    blurb:
      'Stopping a local generation now distinguishes a finished job, a known ComfyUI prompt that can be retried, and an unknown submission that requires a confirmed ComfyUI restart. LDS keeps the exact recovery card until it is safe, removes terminal cards cleanly, and no longer leaves the whole app stuck behind an orphaned “GPU busy” lock.',
    to: '/datasets?section=add&panel=generate',
  },
  {
    id: '2026-07-31-generation-lora-double-stack-guard',
    date: '2026-07-31',
    title: 'A generation-LoRA preset can no longer double-apply your identity LoRA',
    blurb: 'A preset row pointing at the same file already loaded as the identity/consistency LoRA used to chain it a second time — summing both strengths into one delta well past what the file was trained for, which showed up as blocky, posterized dataset images. Krea 2 Edit and Klein now skip that row (the rest of the chain still applies), with a log line saying why. Fixed by Cyberschorsch (GitHub), reported by waltm (Discord).',
    to: '/settings/engines',
  },
  {
    id: '2026-07-30-retry-reference-edit-show-engine',
    date: '2026-07-30',
    title: 'Retry a reference edit exactly as it ran',
    blurb:
      'An Edit reference candidate now names the engine/API that actually produced it. Retry repeats the same instruction, selected engine and temporary reference files; choose Try another prompt only when you want to change the edit.',
    to: '/datasets?section=add',
  },
  {
    id: '2026-07-30-krea-generation-lora-presets',
    date: '2026-07-30',
    title: 'Your own LoRAs now apply when Krea builds a dataset',
    blurb: 'Krea 2 Edit can chain your own LoRA files while it generates dataset images — not just in the Test Studio. Build named combinations in Settings › Image engines, then pick one per run in the workspace\'s 🧬 Krea 2 Edit tuning panel; "None" stays the default. The LoRA has to be trained for Krea 2, only the model side is patched, and a row whose file you have moved is skipped with the rest of the chain still applied. Built by Cyberschorsch (GitHub), idea by waltm (Discord).',
    to: '/settings/engines',
  },
  {
    id: '2026-07-30-krea-shot-card-adherence',
    date: '2026-07-30',
    title: 'Krea 2 now follows the dataset shots you selected',
    blurb:
      'Krea 2 Edit used to let the reference photo dominate a dataset run, so distinct cards could repeat its pose. Its calibrated Krea-only profile now gives the selected card priority for angle, expression, pose and scene, while keeping identity from the reference. Face cards render 1:1 and bust/body/back cards 3:4 through Krea Fit v1.2, so a square reference no longer squeezes a full-body or sitting card into a bust crop. ChatGPT, Nano Banana, OpenRouter and Klein keep their own generation paths unchanged.',
    to: '/datasets?section=add',
  },
  {
    id: '2026-07-30-studio-safe-pause-comfyui-start',
    date: '2026-07-30',
    title: '🛟 A Test Studio batch now pauses safely for ComfyUI recovery',
    blurb:
      'If ComfyUI goes away mid-batch, Studio now pauses with a paste-safe reason and submits no later prompt. Recover or restart ComfyUI, then cancel and resume the batch. Setup’s “Start ComfyUI” also works from a phone already allowed by LDS and uses the app’s fixed safe profile: it never reads, changes or runs a .bat file, so your own launcher stays untouched.',
    to: '/studio',
  },
  {
    id: '2026-07-30-runs-test-in-studio',
    date: '2026-07-30',
    title: 'Open the right Test Studio straight from a training run',
    blurb:
      'Every run that still belongs to a dataset now carries 🧪 Test in Studio in Runs — active local and cloud runs, recent cards, and even a folded dataset group. One click opens Test Studio with that run’s dataset already selected, so you can compare its checkpoints without first hunting through the library.',
    to: '/cloud',
  },
  {
    id: '2026-07-30-keep-an-improvement-without-training-on-both',
    date: '2026-07-30',
    title: '✓ Keep an improved image without accidentally training on both',
    blurb:
      'Keeping a completed ✨ Upscale & improve candidate now returns its original to Undecided automatically — from one tile or a bulk Keep, even when both were selected. Nothing is deleted: both files and the comparison remain, and you can keep the original again if you deliberately want both in training.',
    to: '/datasets?section=images',
  },
  {
    id: '2026-07-30-preserve-imported-photo-files',
    date: '2026-07-30',
    title: 'Keep the photo you imported, not an automatic WebP copy',
    blurb:
      'New un-cropped JPG, PNG, WebP and BMP imports now stay byte-for-byte in their original format by default. Training still gets disposable PNG pairs only when it starts, so the dataset keeps its master files. WebP normalization remains available as an opt-in policy; Auto head-crop deliberately creates a derived WebP, and older WebPs cannot be reversed into originals.',
    to: '/settings/captioning',
  },
  {
    id: '2026-07-30-krea-raw-lokr-likeness-starter',
    date: '2026-07-30',
    title: 'Start a Krea 2 Raw LoKr likeness run from a named recipe',
    blurb:
      'A new Character-only Krea 2 Raw · LoKr likeness preset puts the reported community starting point in one place: LoKr factor 16, 32/32, 768 px, Automagic2, Sigmoid, Balanced and differential guidance 3. Those Krea-only controls are now visible in Expert options and every run records them for comparison. It is a starting point, not a likeness promise: inspect your own checkpoints, and type 3000 in Steps only when you deliberately want that target instead of the adaptive policy.',
    to: '/datasets?section=training&panel=advanced',
  },
  {
    id: '2026-07-30-dataset-to-bank-keeps-useful-context',
    date: '2026-07-30',
    title: '🗃️ Turn a dataset back into a bank without losing its useful context',
    blurb:
      '↑ Import to bank now carries captions, keep/reject curation, framing, watermark and provenance into the copied bank, whichever analysis choice you make. The default restores compatible final-file technical analysis; Start fresh skips only reuse of prior analysis. Face and Score AI results are deliberately not reused after normalization, and the original dataset is untouched. Suggested by the owner.',
    to: '/datasets?section=export&panel=to-bank',
  },
  {
    id: '2026-07-29-test-studio-random-dataset-captions',
    date: '2026-07-29',
    title: 'Give Test Studio a real dataset caption in one click',
    blurb:
      '🎲 Caption can now pull a random nonblank caption from a kept image in a dataset you choose, so a useful Studio test prompt is never far from the work you already curated. Pick the source once and it stays locked in this browser; use ▾ to change it. If you have typed a prompt, Studio asks before replacing it. Suggested by the owner.',
    to: '/studio',
  },
  {
    id: '2026-07-29-deploy-follows-extra-model-paths',
    date: '2026-07-29',
    title: 'Deployed LoRAs land where your extra_model_paths.yaml says',
    blurb:
      'If your ComfyUI keeps its LoRAs outside the install folder, deploying one — and the "open LoRA folder" button — used the default folder anyway. Both now follow your extra_model_paths.yaml, the LoRA override still wins when you set one, and Settings shows the exact folder that will receive the file. LoRAs you deployed before stay listed and deletable where they are. Thanks to Geekswordsman (GitHub #25).',
    to: '/settings/local-tools',
  },
  {
    id: '2026-07-29-training-bases-follow-extra-model-paths',
    date: '2026-07-29',
    title: 'Train on a base that lives outside ComfyUI\'s models folder',
    blurb:
      'If your ComfyUI keeps its weights elsewhere through extra_model_paths.yaml (portable builds, Stability Matrix, a shared A1111 tree), your SDXL checkpoints and Z-Image merges now show up in the training base picker, launch, and convert — the last two places that still only looked in models/. When the same file name exists in two roots, the app picks the one ComfyUI itself would load, so you train on the weights you generate with. And a base that really is missing is now named here, instead of failing later inside ai-toolkit with a path you never typed.',
    to: '/datasets',
  },
  {
    id: '2026-07-29-chatgpt-failures-name-their-cause',
    date: '2026-07-29',
    title: 'A failed ChatGPT generation now says what went wrong',
    blurb:
      'On a ChatGPT subscription, a dropped connection, a timeout, an OpenAI outage and a lane OpenAI had closed all came back as the same blank tile worded as if the provider had simply produced nothing. Each now names itself, and the two that stop a run say so: a refused connection asks you to reconnect, and an endpoint OpenAI no longer serves tells you to switch to API-key mode instead of leaving you searching your own settings. On the API key, a safety refusal is now told apart from a reference photo OpenAI could not read — opposite problems that used to share one sentence. OpenRouter does the same with the moderation reasons it returns inside a successful response. Where the cause genuinely cannot be read, the tile still says so rather than guessing.',
    to: '/settings/engines',
  },
  {
    id: '2026-07-29-every-klein-screen-names-its-model',
    date: '2026-07-29',
    title: 'Every screen that runs Klein now says which model it runs',
    blurb:
      'Three more places started Klein work on a model nothing named: the reference edit, the rescue of scraped images under 768 px, and the 🧽 watermark clean. They follow your dataset\'s Klein model now, like improve and generation already did, and each screen states which model will run — including when there is only one and nothing to pick. A model you chose that has since been moved or deleted is refused by name instead of being swapped for a neighbour. A bank has no dataset to follow, so its Klein inpaint keeps resolving the model itself — and now tells you which. Datasets that never chose a model are untouched.',
    to: '/datasets',
  },
  {
    id: '2026-07-28-memory-savers-and-timestep-per-family',
    date: '2026-07-28',
    title: 'Switching model family no longer changes your run behind your back',
    blurb:
      'Turning quantisation or low-VRAM streaming off on a small family (Anima, SDXL — where off is the normal setting) and then switching to Krea 2, FLUX or Z-Image quietly built an unquantised 12B run: no warning, and hours of GPU — or rented cloud time — before anything went wrong. Advanced options and the pre-launch check now both say which saver is off, what that family actually needs, and what your card has. Timestep weighting is remembered per family instead, so a value picked for Z-Image stops overwriting the recipe of FLUX.2 Klein or Anima. Nothing changes for a dataset that stays on one family.',
    to: '/datasets',
  },
  {
    id: '2026-07-28-mask-faces-says-why-it-is-off',
    date: '2026-07-28',
    title: '"Mask faces" tells you why it does not apply instead of vanishing',
    blurb:
      'On a character or style dataset the option disappeared entirely, so it read as a missing feature rather than one that does not apply — people went looking for it. It is now shown, greyed out, with the reason: a character LoRA has to learn the face, so weighing faces down would erase what you are training.',
    to: '/datasets',
  },
  {
    id: '2026-07-28-comfyui-input-copies-cleaned-up',
    date: '2026-07-28',
    title: 'Improve and Edit stop filling your ComfyUI input folder',
    blurb:
      'Every Klein or Krea job copies your source image into ComfyUI\'s input folder, and nothing ever deleted those copies: a three-month-old install had 3,896 of them sitting there, 0.67 GB. Each job now removes its own copies the moment it ends — finished, failed or stopped — and a sweep at startup clears what earlier versions left behind. Images you dropped in that folder yourself are never touched.',
  },
  {
    id: '2026-07-28-training-base-is-per-family',
    date: '2026-07-28',
    title: 'Your training base now belongs to its model family',
    blurb:
      'Picking a custom base for Z-Image and then switching the LoRA type to Krea 2 left that Z-Image file attached: the selector said Krea 2, the line below it said the Z-Image file, and the only cure was changing the family and coming back — which fixed the screen but nothing else, so it was back on the next reload. The base and the variant are now remembered per family: switching hands you that family\'s own base, coming back finds yours exactly where you left it, and nothing is thrown away. The cloud dialog stops offering to upload another family\'s weights to your Hugging Face account, and no longer calls a file "missing" when it is sitting safely on your disk.',
    to: '/datasets',
  },
  {
    id: '2026-07-28-lightbox-actions-use-the-side-space',
    date: '2026-07-28',
    title: 'A portrait photo now fills the screen, with the actions beside it',
    blurb:
      'Opening a portrait image full-screen on a wide monitor left two thirds of the width black while Crop, Mirror, Rotate and Upscale & improve queued on one line underneath — on the one axis the photo was short of. Those actions now move into a labelled rail in that empty space, and the photo gets the height back. Landscape images keep the bar at the bottom, where there is no side space to take, and phones are unchanged. The rail keeps full wording, not mute icons.',
  },
  {
    id: '2026-07-28-choose-the-klein-model-improve-runs-on',
    date: '2026-07-28',
    title: 'Choose which Klein model ✨ Upscale & improve runs on',
    blurb:
      'Improve never asked which model to use: it picked one for you, silently, and nothing on the screen said which. It now names the model it will run — even when there is only one — and lets you choose it when your ComfyUI has several. The choice is saved on the dataset (not in one browser), it is the same model Klein generation uses, and it applies to the single pass, the 🔄 re-run and the whole batch alike. Models are detected automatically wherever ComfyUI can load them, and if the one you chose is later moved away the run says so by name instead of quietly swapping in another.',
    to: '/datasets',
  },
  {
    id: '2026-07-28-refused-save-keeps-what-you-typed',
    date: '2026-07-28',
    title: 'A refused save no longer throws away what you just typed',
    blurb:
      'Four dialogs closed themselves before the server had answered, so a refusal deleted your work: the expanded caption editor lost the long AND short caption you had just written, the ✏️ edit-prompt bubble lost a rewritten prompt, 🚀 Launch all reset its seven pass checkboxes, and the folder browser dropped you back to the drive list. They now stay open, keep every field exactly as you left it, and show the reason next to the input that caused it. Escape and clicking outside still close them — only the server keeps them open.',
  },
  {
    id: '2026-07-28-nano-banana-says-when-google-refused',
    date: '2026-07-28',
    title: 'Missing images from Nano Banana now say who refused them',
    blurb:
      'Google screens every image Gemini returns, and when it blocks one the API answers "success" with nothing in it. LDS used to call that an empty response and suggest retrying — so a refused request looked like a broken app. Each refused tile now names the cause and relays Google\'s own reason code, a run tells you how many were refused versus how many actually failed, and real problems (key, quota, connection) keep their own separate message. A batch never stops on a refusal. No promises attached: that filter is not configurable, it refuses ordinary requests, and the same prompt can pass one time and not the next.',
    to: '/datasets?section=add',
  },
  {
    id: '2026-07-28-bank-watermark-mask-editing',
    date: '2026-07-28',
    title: 'Fix a wrong watermark box without leaving the bank',
    blurb:
      'The watermark detector draws one box, and it guesses: it misses a second logo, or lands beside the mark. Correcting it was only possible inside a dataset, so in a bank a bad box meant rejecting the image. Open ▶ Review on a flagged image and press 🚩 Edit mask: draw the zones yourself, and 🧽 Inpaint repaints exactly those — including a mark on the subject. Auto-crop deliberately skips a hand-masked image, and an emptied mask cleans nothing, on purpose. Thanks to Qeeyana (Reddit) for reporting it.',
    to: '/bank',
  },
  {
    id: '2026-07-28-improve-says-what-it-is-about-to-ask-klein',
    date: '2026-07-28',
    title: 'Upscale & improve now shows the instruction it is about to send',
    blurb:
      'The improve pass sends Klein a fixed instruction — and the built-in one asks for photographic texture and sharp detail, which is why anime and illustrated datasets came back looking realistic. That instruction is now quoted right next to the ✨ button, with one click to rewrite it or turn it off entirely, and a drawn dataset gets an explicit warning. Thanks Qeeyana (Reddit).',
    to: '/settings/engines',
  },
  {
    id: '2026-07-28-klein-model-needs-no-symlink',
    date: '2026-07-28',
    title: 'Your Klein model can stay where it is — no copy, no symlink',
    blurb:
      'models/unet/klein/ is only where Setup downloads to; it was never required. Any klein-named sub-folder of models/unet or models/diffusion_models works, as does either folder\'s top level, plus every extra_model_paths.yaml root and a relocated models folder. The Setup screen, the README and the Guide now say so, and a test keeps that list honest. Thanks CyberTod (Reddit).',
  },
  {
    id: '2026-07-28-import-resolution-is-yours-to-choose',
    date: '2026-07-28',
    title: 'You choose what resolution your imported photos are stored at',
    blurb:
      'Every image entering a dataset was resampled to 1024 px and re-encoded as WebP quality 92 — with no setting anywhere and nothing on screen saying why. Settings ▸ Captioning & quality ▸ Dataset import now lets you pick 1024 to 4096 px, or the original size, and the encoding alongside it (quality 92, quality 100, or fully lossless) — because the resolution was only half the loss. The default is unchanged at 1024/q92, so nothing already imported moves, and the import dropzone now states what it is about to store and why, with a link to change it. Original size still stops at 8192 px: WebP itself refuses past 16383. Thanks to Qeeyana (Reddit) for asking why the app was deciding this for you.',
    to: '/settings/captioning',
  },
  {
    id: '2026-07-28-caption-elsewhere-round-trip',
    date: '2026-07-28',
    title: 'Caption your images in another tool, and bring the captions back',
    blurb:
      'A Style dataset refused to export its ZIP until every kept image was captioned, with no way past it — which blocked something perfectly reasonable: getting the bare images out to caption them in your own tool. That refusal is now a confirmation that explains what an empty caption does to a Style LoRA, and cancelling still takes you to the captions. The return trip works too, and that was the real gap: re-importing those images with their new .txt files used to drop every one of them as a duplicate, captions included. Their captions now land on the images already in the dataset, a caption you wrote here is never overwritten, and the Import & export panel finally says the round trip exists. Thanks to Qeeyana (Reddit).',
    to: '/datasets?section=export',
  },
  {
    id: '2026-07-28-installs-that-do-not-work-no-longer-report-success',
    date: '2026-07-28',
    title: 'An install that did not work no longer says it worked',
    blurb:
      'Installing Person masks could report success — every requirement "already satisfied" — while the capability stayed ✗ Not installed and masked training quietly fell back to unmasked, so a whole run learned the background you meant to exclude. The missing piece (onnxruntime, which rembg needs but no longer declares) is now installed, an existing GPU build is left alone, and every scoped install re-runs the capability check afterwards and names the missing module when it fails. Launching a masked run without it now asks you first, instead of finding out from the result. Thanks to 1Tomber (GitHub #24).',
    to: '/settings/local-tools',
  },
  {
    id: '2026-07-28-cloud-boot-waits-for-a-pod-that-is-still-working',
    date: '2026-07-28',
    title: 'Cloud launches survive a slow host pulling its image',
    blurb:
      'A pod that took more than 25 minutes to boot was terminated even when it was honestly downloading its multi-gigabyte image — and its host was quietly skipped for the next three days. The boot wait now restarts its clock whenever the pod shows real progress, keeps an absolute ceiling so a dead pod still dies fast, tells you where the boot actually got to, and only exiles a slow host for a few hours.',
  },
  {
    id: '2026-07-28-continue-training-keeps-your-choices-when-refused',
    date: '2026-07-28',
    title: 'A refused ▶ Continue training no longer wipes what you picked',
    blurb:
      'Continue training closed the moment you clicked it, so a refusal left you with an error and an empty form: the lane, the checkpoint to resume from, the extra steps and every adjusted setting had to be typed again, with no clue which of them was refused. The form now stays open, the reason appears inside it next to those choices, and only a launch that actually starts closes it.',
  },
  {
    id: '2026-07-28-a-bank-can-no-longer-be-created-on-a-dataset-folder',
    date: '2026-07-28',
    title: 'A bank can no longer be pointed at a dataset’s own image folder',
    blurb:
      'Nothing stopped you from pasting a dataset’s storage folder into “Create bank”. The bank then listed the dataset’s live files — and its 🗑 Delete rejected deleted images out of the dataset, with no warning at all. A bank and a dataset only ever pass images to each other by copy, so that folder is now refused at creation and when moving a bank, and the refusal names the dataset and sends you to 🗃 Import to bank instead (which copies). The check sees through subfolders, the folder holding all datasets, a different case, other separators, and symlinks or Windows junctions. If you already have such a bank, nothing is deleted or repaired behind your back: opening it says so, and only the destructive button is refused.',
    to: '/bank',
  },
  {
    id: '2026-07-28-a-dataset-now-shows-where-its-images-are-on-disk',
    date: '2026-07-28',
    title: 'A dataset now shows where its images are on disk',
    blurb:
      'The folder holding a dataset’s images was displayed nowhere, so finding it meant digging through the app’s data directory by hand. It is now at the top of the dataset with a copy button — next to the one line worth knowing: that folder belongs to the dataset, and it must not be used as an image bank’s source.',
    to: '/datasets',
  },
  {
    id: '2026-07-28-masked-training-is-saved-on-the-dataset',
    date: '2026-07-28',
    title: 'Masked training is saved on the dataset, not in one browser',
    blurb:
      'The 🎭 Masked toggle used to live in the browser you set it in: open the app from your phone and it quietly reverted to the default, and no run recorded which way it was set. It is now a dataset setting — shared across your devices, stamped into every run so two runs that differ only by masking no longer look identical, and read by the readiness badge, which can finally warn you that a dataset set to masked will train unmasked because rembg is missing. Existing datasets keep today’s behaviour; a browser that had turned masking off is asked once what to do with it.',
    to: '/datasets?section=training',
  },
  {
    id: '2026-07-28-docker-image-with-comfyui-and-your-gpu',
    date: '2026-07-28',
    title: 'A Docker image that brings its own ComfyUI — and uses your GPU',
    blurb:
      'The Docker image could not do the ComfyUI half of the app at all: no Klein or Z-Image generation, no Test Studio, no deploying a trained LoRA, because ComfyUI was a host-native tool the container could not see. A second image now runs ComfyUI inside the same container on your NVIDIA GPU, with the folder paths already filled in — start it with "docker compose -f docker-compose.gpu.yml up --build". The API-only image is unchanged for machines without a GPU. Two honest limits: it is a large download, about 20 GB before you download a single model, and local training still needs ai-toolkit on the host.',
  },
  {
    id: '2026-07-28-notifications-are-no-longer-hidden-behind-dialogs',
    date: '2026-07-28',
    title: 'Notifications no longer disappear behind an open dialog',
    blurb:
      'Every message the app raises — a refusal, a confirmation, an error — was drawn underneath any open dialog or full-screen viewer, so it simply never reached you: the app answered, and the answer was covered up. Notifications now sit above everything, and a check makes sure no future panel can climb over them again.',
  },
  {
    id: '2026-07-28-cloud-watchdog-counts-a-downloading-pod-as-progress',
    date: '2026-07-28',
    title: 'A cloud run is no longer killed while its pod is downloading normally',
    blurb:
      'The run card now shows the bytes a pod is fetching — but the watchdog guarding that phase was still only watching the training step counter, so a run on a slow host was killed at 45 minutes for "no progress" while the card beside it showed the download working perfectly (a 26.3 GB model at the 2.6 MB/s some hosts give you takes nearly 3 hours). The watchdog now reads the same counter the card does: bytes moving is progress, and the clock restarts. A pod that reports no bytes at all still dies as fast as before, a hard ceiling still stops a host that will never finish, and the failure message finally says what was measured instead of guessing. The idle budget and that ceiling are now in Settings → Training. Thanks to j_o_e_l. (Discord) for the report.',
    to: '/settings/training',
  },
  {
    id: '2026-07-28-retry-asks-instead-of-doing-nothing',
    date: '2026-07-28',
    title: 'Retry no longer looks dead when a run needs your confirmation',
    blurb:
      'On the Runs page, ↻ Retry could do nothing at all: no job, no error, no toast. It happened whenever the run needed a confirmation Start had already asked for — an image with no caption, a dataset under the image floor, captions in the wrong style — because Retry never carried your answer, and the refusal that came back was thrown away before it reached the screen. Retry now asks the same question Start asks, relaunches once you confirm, and says out loud why it stopped when it stops. Stop and 🧹 Clean finished runs on the same page were silent in the same way and now speak too. Reported by 1Tomber (GitHub #23).',
    to: '/cloud',
  },
  {
    id: '2026-07-28-download-canvas-images-one-or-the-whole-gallery',
    date: '2026-07-28',
    title: 'Download your generated images — one, or a whole run as a ZIP',
    blurb:
      'The board can now hand the pictures over: ⬇ on a pinned image and in the full-screen viewer saves that one, and ⬇ ZIP in a gallery saves the lot (turn on Select first to take only the ones you tick). Every file keeps its lineage in its NAME — dataset, run, step and seed — so a render is still identifiable a month later instead of becoming another out_00042_.png. Big galleries say up front how many the archive holds, and a file that has left the disk is named rather than quietly dropped.',
  },
  {
    id: '2026-07-28-canvas-fuse-pinned-images',
    date: '2026-07-28',
    title: 'Drop one pinned image onto another and compare them edge to edge',
    blurb:
      'Comparing two checkpoints on the canvas meant lining two pinned pictures up by hand and squinting at the gap between their frames. Now dropping one onto another fuses them into a single node: the pictures sit side by side with nothing drawn between them, and there is no limit — add a third, a tenth. Drag the title bar to move the whole strip, hover a picture for its own 🔍 and ✕, and drag one off the group to take it back out at the size it had before.',
    to: '/canvas',
  },
  {
    id: '2026-07-28-scrape-straight-into-a-bank',
    date: '2026-07-28',
    title: 'Scrape the web straight into a bank — no throwaway dataset first',
    blurb:
      'The scraper had one outlet: straight into a dataset, through filters made for training — anything under 768 px, anything wider than 3:1 and anything it judged a near-duplicate was dropped before you ever saw it. Getting a scrape into the Image bank meant building a dataset you did not want, then importing it back, having already lost the images the triage passes exist to judge. The Image bank page now has its own scrape section: same scan, same picking, you just choose which bank receives them — a new one, or more into a bank you are already triaging. Nothing is filtered on the way in; the quality, duplicate and framing passes rule on the pile, and you promote the keepers into a dataset as usual.',
    to: '/bank',
  },
  {
    id: '2026-07-28-cloud-run-download-bytes-and-durable-freeze-clock',
    date: '2026-07-28',
    title: 'A downloading cloud run now shows the bytes, not a frozen sentence',
    blurb:
      'While a pod fetches its base weights — 26 GB for Krea — the run card used to show one motionless line, "fetching transformer weights", for as long as it took. Nothing told a healthy download from a dead pod short of opening the vast.ai console, and people waited hours to find out. The card now reads the pod\'s own counter: how much has landed, of how much, at what speed, with the ETA. And the "no progress" warning is finally reliable — it is measured on what the pod does, so restarting the app no longer resets it, and a monitor repeating the same sentence no longer hides a frozen run.',
    to: '/cloud',
  },
  {
    id: '2026-07-28-generation-works-on-linux-and-across-wsl',
    date: '2026-07-28',
    title: 'Generation works on Linux — and when ComfyUI runs in WSL or a container',
    blurb:
      'On Linux, nothing generated at all: every model kept in a subfolder (Krea, Klein, Z-Image, your trained LoRAs — which is all of them) was handed to ComfyUI with Windows-style backslashes, and ComfyUI rejected the whole workflow before the first step. Model names are now spelled the way the ComfyUI you are actually talking to spells them, read from that install itself — so it also works the other way round, when the app runs on Windows and ComfyUI lives in WSL, Docker or on another machine. Found and diagnosed by 1Tomber (GitHub #21).',
  },
  {
    id: '2026-07-28-pick-diverse-and-balanced-are-fast-again',
    date: '2026-07-28',
    title: '🎨 Pick diverse and ⚖️ Balanced pick answer in about a second',
    blurb:
      'On a large bank these two buttons took over half a minute, almost all of it spent computing the same thing over and over: how crowded each image\'s neighbourhood is, plus one filesystem lookup per image that had already been done. The maths now runs on an optimised BLAS, the bank folder is resolved once instead of once per image, and a second click on an unchanged bank reuses the scores it just read. Measured on a real 9 500-image pool: 32 seconds down to roughly two. The images picked are exactly the same ones — this is speed, not a different selection.',
  },
  {
    id: '2026-07-28-dual-captions-no-longer-crash-krea-and-anima',
    date: '2026-07-28',
    title: 'Dual captions no longer crash a Krea 2 or Anima run',
    blurb:
      'Turning on dual captions before a Krea 2 or Anima run made training die at the first step with a NoneType error — after the weights download and the whole caching pass. Those two families pre-cache their text embeddings and unload the text encoder to fit in VRAM, so there is no encoder left to read a second caption. The app now says so on the toggle and in the pre-launch check, and trains on the long caption alone instead of building a config that cannot run. Reported by 1Tomber (GitHub #22).',
    to: '/datasets?section=training',
  },
  {
    id: '2026-07-28-continue-training-from-the-canvas',
    date: '2026-07-28',
    title: 'Continue training straight from a checkpoint on the Canvas',
    blurb:
      'Click any checkpoint on the LoRA Canvas and ▶ Continue from here now opens the real launch dialog on that exact save — where to run it (local or cloud), how many extra steps, cadence, preview prompts, timestep weighting and learning rate. It used to be a greyed line telling you to go find the run on another page, and which page depended on whether it had trained locally or in the cloud. The board is also the first place that offers BOTH lanes for BOTH kinds of run: a local checkpoint can be finished on a rented GPU, and a cloud epoch can be finished on your machine. A lane that cannot be used says why instead of disappearing.',
    to: '/canvas',
  },
  {
    id: '2026-07-28-busy-database-no-longer-strands-a-paid-cloud-run',
    date: '2026-07-28',
    title: 'A busy database no longer strands a paid cloud run — for real this time',
    blurb:
      'Cloud runs are watched by a monitor that writes progress to the database every few seconds. When something else was writing heavily at that moment (a caption batch, a Bank import), that write could lose the lock — and the retry meant to absorb it crashed on its own error message instead, killing the monitor. The run then sat at "TRAINING" with no error, no progress and a rented GPU still billing, until the freeze watchdog or an app restart caught it up to 45 minutes later. The retry now works, and a run whose monitor does die is closed properly with its pod terminated instead of being left open.',
    to: '/cloud',
  },
  {
    id: '2026-07-28-comfyui-slow-is-not-comfyui-stopped',
    date: '2026-07-28',
    title: 'A busy ComfyUI is no longer reported as a stopped one',
    blurb:
      'The app gave ComfyUI 8 seconds to list its nodes and model files — and that list grows with every custom-node pack and every weight you install, so the richer your ComfyUI, the more likely it ran out of time. Krea 2 generations then refused with "ComfyUI isn\'t running" at a ComfyUI that was running perfectly. The budget is now 45 seconds and adjustable (Settings ▸ Local tools ▸ ComfyUI ▸ "ComfyUI response timeout"), and a slow ComfyUI and a stopped one no longer share one message: one tells you to raise the timeout, the other to start ComfyUI. A ComfyUI that is genuinely off is still detected in seconds, so nothing waits 45 seconds for nothing. Found, measured (~15 s on his install) and fixed by j_o_e_l. (Discord).',
    to: '/settings/local-tools',
  },
  {
    id: '2026-07-28-bank-undo-last-bulk-decision',
    date: '2026-07-28',
    title: 'Marked 400 bank images by mistake? Take it back.',
    blurb:
      'A bank\'s bulk actions — ✓/✕ over a whole filter, auto-reject at a threshold, collapsing duplicate groups, 🚀 Launch all — now leave an ↩ Undo bar above the grid. One press puts every image back exactly as it was, its state and its reason, without touching the images the action never moved. The bar waits for you instead of vanishing on a timer, and it survives a page reload. Limits stated on the bar itself: one step back, and only until the app restarts. 🗑 Delete rejected and ⬆ Promote deliberately offer nothing, because neither can be undone honestly — and if an undo cannot restore everything, it says how many it restored and names what it left alone.',
    to: '/bank',
  },
  {
    id: '2026-07-28-balanced-pick',
    date: '2026-07-28',
    title: 'Pick a set that covers your framings, not just the top of a ranking',
    blurb:
      'Asking for "the 60 most varied" of a bank that is mostly full-body shots gives you mostly full-body shots — on a realistic test pool it returned 0 face shots and 0 back views out of 20, and nothing said so. ⚖️ Balanced pick spreads the same sampling evenly over face / bust / body / back (optionally per person), tells you exactly what you got — "5 face, 5 bust, 5 body, 5 back" — and names any framing it could not fill instead of quietly padding with something else. It sits in the Curate row and at the bottom of 📊 Coverage advice, where the advice finally becomes a gesture.',
    to: '/bank',
  },
  {
    id: '2026-07-28-crops-keep-their-own-resolution',
    date: '2026-07-28',
    title: 'Cropping in no longer blows the crop up to 1024 px',
    blurb:
      'A crop used to be stretched to a 1024 px long side whatever its real size, so a 240×180 selection was stored as 1024×768. Those extra pixels carried nothing — shrinking such a file back recovers the real crop almost exactly — while costing about 6× the bytes now that crops are stored losslessly. A crop is now never enlarged beyond what you actually selected; anything LARGER than 1024 px is still normalised down to 1024, exactly as before. Two things to know: new crops are smaller images than old ones, so a dataset can end up mixing sizes — training handles that (it buckets by size), but the smallest tiles genuinely carry less detail. That is what the composition meter now says out loud: the old "⚠ Upscaled" line is now "⚠ Under training resolution", and it still flags a framing bucket you filled by cropping far into a photo instead of adding native shots. Images cropped BEFORE this keep the enlarged pixels they already have.',
    to: '/datasets',
  },
  {
    id: '2026-07-28-broken-model-replaced-only-once-the-new-one-arrives',
    date: '2026-07-28',
    title: 'A corrupted model file is only deleted once its replacement has landed',
    blurb:
      'Setup can now spot a model file that cannot be loaded (a login page saved under the model name, a truncated download) and re-fetch it. It used to delete the broken file first and download after — so an expired token, a re-gated repo or a host that was simply down left you with nothing at all instead of something broken. Now the download is opened and checked first, and the old file is only removed once real weights are actually on disk. Nothing is ever thrown away for a download that did not happen.',
    to: '/setup',
  },
  {
    id: '2026-07-28-bank-rerun-buttons-say-what-is-happening',
    date: '2026-07-28',
    title: 'The bank\'s ↻ re-run buttons finally tell you what is going on',
    blurb:
      'A bank runs one pass at a time, so pressing ↻ Re-group duplicates while a ✨ Score was still walking the bank could only ever produce a red "a scan job is already running on this bank" — a sentence with no progress and no way out. Those buttons are now disabled while another pass owns the bank, and each says which one and how far it has got ("✨ Score pass is running — 137 / 412"), pointing at Stop. When a re-run does go through, it reports what it produced right where you pressed it — "Done — 12 duplicate groups · 34 images (was 9 · 26)" — and says "unchanged" when your new value groups exactly the same images, instead of leaving you unable to tell a no-op from a pass that never ran. Every other occupied-bank refusal in the bank (Promote, 🗑 Delete rejected, 🚀 Launch all, the watermark passes) is now worded the same way.',
    to: '/bank',
  },
  {
    id: '2026-07-28-cropping-no-longer-recompresses',
    date: '2026-07-28',
    title: 'Cropping no longer quietly re-compresses your image',
    blurb:
      'Every crop was re-encoded to lossy WebP, so cropping a PNG degraded it — and left a .png file holding WebP bytes. Crop and the watermark cleaners now keep the file\'s own format and write it back without losing pixels, like ✂ Mirror and ↺ Rotate already did: crop the same shot ten times and the tenth is identical to the first. JPEG has no lossless mode, so it is re-saved at the highest practical quality instead of being converted to something heavier. Cropped files are noticeably bigger now — that is the price of keeping the pixels. Two honest limits: a box longer than 1024 px is still rescaled down to it, and that resampling can never be lossless (only the watermark ✂ auto-crop, which never resizes, is); and images you cropped BEFORE this keep the pixels they have — nothing is re-processed retroactively.',
    to: '/datasets',
  },
  {
    id: '2026-07-28-rotate-images-in-the-dataset-and-the-bank',
    date: '2026-07-28',
    title: 'Straighten a sideways photo — rotate 90° in the dataset and in the bank',
    blurb:
      'Idea by 1Tomber (GitHub #17). ↺ / ↻ turn an image a quarter turn: in the dataset from the image inspector, next to Mirror, and in the bank from the selection bar or straight inside ▶ Review ([ and ]). It costs the image nothing it does not have to: a dataset PNG or WEBP comes back pixel-for-pixel identical after four turns, and in the bank your own files are never rewritten at all — the turn is remembered and applied to what you see and to what gets promoted.',
    to: '/datasets',
  },
  {
    id: '2026-07-28-offline-is-not-empty',
    date: '2026-07-28',
    title: 'Losing the connection no longer looks like your job stopped',
    blurb:
      'Leaving a running pass and coming back on a phone used to greet you with ten stacked "Connection lost" banners over the whole app — and no progress bar, because the poll behind it had failed. The banners are now one line that counts repeats, automatic polls fail silently, and a single "Offline — reconnecting…" strip takes over: your progress stays on screen, marked as the last thing we heard. Passes always kept running on the server; now the screen says so, and says it again when the connection is back.',
    to: '/bank',
  },
  {
    id: '2026-07-28-bank-filter-thresholds-in-place',
    date: '2026-07-28',
    title: 'Tune every Bank filter without leaving the bank',
    blurb:
      'The twelve numbers behind the filter chips — blurry, small, duplicate, NSFW — were only editable in Settings, three screens from the bank you were triaging. They are now under the chips too, in 🎚 Filter thresholds: grouped by what they answer, each one saying which way catches MORE images (the duplicate distance and the semantic similarity move opposite ways), when it takes effect, and how many images the value you are typing would flag — before you save. Reset any one, or all of them, to the shipped defaults. Same setting as Settings, so it applies to every bank.',
    to: '/bank',
  },
  {
    id: '2026-07-28-canvas-pin-all-generated-images',
    date: '2026-07-28',
    title: 'One click puts every image a canvas run made onto the board',
    blurb:
      'A finished generation said “5 images ready” and left you to open each checkpoint’s gallery and pin the pictures one by one. The green bar now carries 📌 Pin all — the whole lot lands on the board in one go, each image in its own column under the checkpoint that made it, and nothing is ever placed on top of anything else. It says how many it put down, names anything it left out, and ↩ Undo takes them straight back off.',
    to: '/canvas',
  },
  {
    id: '2026-07-28-canvas-node-buttons-reachable-on-a-phone',
    date: '2026-07-28',
    title: 'The ✕ on a pinned image can be tapped again',
    blurb:
      'Closing a picture pinned on the LoRA Canvas did not work on a phone. The buttons were drawn at the board’s zoom, so on a board read at 65 % the cross was about ten pixels wide with the 🔍 right beside it — a near miss opened the full-screen view instead of closing the node. The ✕, the 🔍 and the resize corner now keep a real finger-sized target at every zoom level.',
    to: '/canvas',
  },
  {
    id: '2026-07-28-preflight-before-cloud-and-continue',
    date: '2026-07-28',
    title: 'The pre-training check now runs before a cloud run and before ▶ Continue',
    blurb:
      'Launching in the cloud went straight to a paid GPU without ever showing the pre-training review — so leaking captions, near-duplicate images and pictures still waiting on a ✓/✕ shipped to the pod unnoticed. ▶ Continue skipped it too, on both lanes. Both now open the same review, with its editable caption list and its reject-one-of-each pairs. Checks about your own GPU (memory, PyTorch build) are left out of a cloud launch: that card is not the one running the job.',
    to: '/datasets',
  },
  {
    id: '2026-07-28-zimage-finds-its-own-encoder-and-vae',
    date: '2026-07-28',
    title: 'Z-Image no longer asks you to rename your files',
    blurb:
      'The Test Studio demanded one exact spelling for the Z-Image text encoder and VAE — “z ae.safetensors” with a space, inside a folder capitalised “Z image”. Anything else, including ComfyUI’s own documented names, read as missing. The app now finds them itself: any capitalisation, any separator (z_ae, z ae, z-ae), any sub-folder, across every extra_model_paths root. If nothing is there it still tells you exactly what to place and where. Thanks to bobba84 (GitHub #18).',
  },
  {
    id: '2026-07-28-zimage-base-keeps-its-own-settings',
    date: '2026-07-28',
    title: 'Z-Image Base starts on Base settings, not Turbo’s',
    blurb:
      'Selecting a non-distilled Z-Image Base in the Test Studio opened on CFG 1 and 8 steps — correct for the distilled Turbo build and ruinous for Base, which needs real guidance and far more steps. Each base model now proposes its own starting CFG and step count, and the pickers reach the values Base needs. Anything you had already chosen yourself is left exactly as it was. Thanks to bobba84 (GitHub #18).',
  },
  {
    id: '2026-07-28-training-names-the-python-it-uses',
    date: '2026-07-28',
    title: 'Training now tells you WHICH Python it is about to run',
    blurb:
      'A path that exists, runs, and has no torch used to pass every check — then every run died on "No module named \'torch\'" while the panel suggested a missing base model or a Hugging Face token, two dead ends. The app now tries `import torch` on the interpreter you configured before it launches anything, and if it fails it refuses with the path on screen, points out a Windows Store python.exe when that is what you picked, and offers the working venv sitting next to run.py. The Test button in Settings ▸ Local tools checks the same thing, and a torch failure never mentions Hugging Face again. Reported in detail by strouder (GitHub #19).',
    to: '/settings/local-tools',
  },
  {
    id: '2026-07-28-a-failed-download-is-not-always-your-network',
    date: '2026-07-28',
    title: 'A download that dies is no longer blamed on your connection',
    blurb:
      'The optional Hugging Face fast-download accelerator (HF_HUB_ENABLE_HF_TRANSFER) needs the hf_xet package, and without it transfers abort with something that reads exactly like a network fault — so people go and check their firewall. The training failure panel now recognises it and names both fixes: set the variable to 0, or install hf_xet. The app never sets that variable itself; it comes from your shell or another tool. Reported by bobba84 (GitHub #18).',
  },
  {
    id: '2026-07-28-guide-explains-the-two-folders',
    date: '2026-07-28',
    title: 'The Guide finally explains which folder does what',
    blurb:
      'A full-local install is three programs, two folders, two ports and two Python environments — and nothing said so, which cost one user hours of patching ai-toolkit\'s own web UI (port 8675) while the real problem was one setting here. Getting started now has a short table: the Studio and its .venv drive training and read config.json, ai-toolkit\'s venv is the one that needs torch, and its Next.js UI is unrelated. It also documents the Python versions that actually work — 3.11 for ai-toolkit, and 3.11.9 on Windows because later 3.11 releases ship no installer. Reported by strouder (GitHub #19).',
    to: '/guide',
  },
  {
    id: '2026-07-28-pin-to-canvas-from-the-thumbnail',
    date: '2026-07-28',
    title: 'Put a generated image on the board without opening it first',
    blurb:
      'Pinning a render onto the lineage board was only offered once you had opened it full-screen, so most people never learned the board could hold images at all. Every thumbnail in a run or checkpoint gallery now carries a 📌 of its own — one tap and it lands on the board next to the checkpoint that made it. It stays out of the way while you are selecting images to delete, so nothing new can be tapped by mistake.',
  },
  {
    id: '2026-07-28-klein-refusals-name-the-cause',
    date: '2026-07-28',
    title: 'When Klein is greyed out, the app now tells you which thing is wrong',
    blurb:
      'The watermark cleaner and the small-image rescue used to answer every Klein refusal the same way — "needs ComfyUI running and the Klein models" — even when ComfyUI was running and the models were right there. They now show the same precise sentence the generation page does: the exact file that is missing, the file that is present but corrupted, the widget value your ComfyUI does not offer, or the engine being switched off in Settings. The cleaner also stops treating a broken weight as usable and silently handing it to ComfyUI.',
  },
  {
    id: '2026-07-28-setup-checks-every-file-it-skips',
    date: '2026-07-28',
    title: 'Setup checks every file it decides not to re-download',
    blurb:
      'A download can be skipped because some other file already covers it — an earlier build under its old name, a copy in a folder you added through extra_model_paths, or a model you placed by hand. None of those were being opened before being vouched for, so a corrupted one sent you back into the same dead end by a different door. Each of them is now validated, and a file that cannot be loaded no longer counts as installed.',
    to: '/setup',
  },
  {
    id: '2026-07-28-setup-stops-certifying-broken-models',
    date: '2026-07-28',
    title: 'Setup no longer says a model is installed when it cannot be loaded',
    blurb:
      'A download that stops halfway leaves a file of plausible size that no loader can open. Setup used to tick it as "✓ Installed" purely because it was there, while the Generate page kept the engine greyed out and blamed a "missing model" — for a file sitting in the right folder. Setup now asks the same question the generation page does, and both give the same answer: the file is named, the fault is named (cut short, corrupted, or a licence page saved as weights), and the fix is one button that replaces it instead of reporting "already present" and doing nothing. The same goes for a widget value your ComfyUI does not offer, and for ComfyUI being unreachable — which now reads as "not checked" rather than a tick nobody earned. Reported by zigzag4794.',
    to: '/setup',
  },
  {
    id: '2026-07-27-face-mask-preview-progress',
    date: '2026-07-27',
    title: 'The face-mask preview now shows what it is doing, and you can walk away from it',
    blurb:
      'Previewing what "Mask faces" would cover used to say "Looking for faces…" and nothing else, for as long as it took — and the longest part happens before the first image is even looked at, while the face detector loads (or, on a fresh install, downloads a few hundred megabytes). You now get the stage by name and a counter that climbs image by image, so a slow run no longer looks like a crashed one. If it does fail — detector missing, model that will not load, the pass dying — it says so instead of spinning forever, and finding no face at all is reported as the ordinary result it is. The detection also runs on the server now: leaving the training panel and coming back picks the same pass back up rather than starting a second one, and the last preview is still on screen when you return. If your kept images changed in the meantime, it tells you the preview is out of date instead of passing it off as current.',
    to: '/datasets',
  },
  {
    id: '2026-07-27-delete-a-run-and-everything-it-produced',
    date: '2026-07-27',
    title: 'Delete a whole run — checkpoints, images and all — from the run panel',
    blurb:
      'Getting rid of an abandoned run used to mean deleting its checkpoints one by one, then its images, then the run entry. Open a run on the LoRA Canvas and the panel now ends with a Danger zone: one button deletes the run, its checkpoints and the images it produced. It tells you exactly what goes first — "14 checkpoints · 24.0 GB, 37 images" — and what stays: runs that continued from it are kept as their own roots, images you marked 👍 survive, and LoRAs already deployed into ComfyUI are left alone so your workflows keep working. Files go to the recycle bin. A run that is training right now cannot be deleted, and if a file refuses to move the run is kept rather than half-deleted.',
    to: '/canvas',
  },
  {
    id: '2026-07-27-mask-faces-installs-its-own-detector',
    date: '2026-07-27',
    title: 'Mask faces now installs what it needs, from where you tick it',
    blurb:
      'Face masking needs a face detector (InsightFace), and on most installs it simply is not there. Until now the option greyed itself out and sent you to the Setup tab to install something called "Face-similarity scoring" — which nobody ticking Mask faces would ever go looking for. The option now names the missing piece, says what it costs before you click (~400 MB, a few minutes), and installs it in place with a progress bar. It stays entirely optional: nothing downloads on its own, and declining leaves the app exactly as it is with just that one option off. If your Python is outside 3.10–3.12 it says so instead of offering an install that could only fail. And launching a run with Mask faces on while the detector is missing no longer silently trains unmasked — the pre-launch report tells you, and lets you install or continue on purpose.',
    to: '/datasets',
  },
  {
    id: '2026-07-27-canvas-pinned-images',
    date: '2026-07-27',
    title: 'Put generated images ON the canvas, next to the checkpoint that made them',
    blurb:
      'Comparing two checkpoints meant opening their images one at a time in a modal — never side by side. Open any generated image and hit 📌 Pin to canvas: it becomes a node on the board, joined to its checkpoint by the same connector the board already uses for "this continued from that". Drag it, resize it from its corner, close it with ✕. Closing does not forget anything: pin the same image again and it comes back exactly where you left it, at exactly the size you left it — stored with your card positions, so it follows the dataset from one machine to the next. Arrow keys move a focused image and +/− resize it, so a mouse is not required.',
    to: '/canvas',
  },
  {
    id: '2026-07-27-generated-image-facts',
    date: '2026-07-27',
    title: 'A generated image now tells you what it was made with — readably',
    blurb:
      'The full-screen view used to print step, seed, strength and the whole prompt as one paragraph stretched across your entire screen, with the three numbers you were looking for buried at the front of it. Now the facts are chips, the settings that decided the picture are a table — sampler, scheduler, CFG, steps, base model, LoRA file, always-on LoRAs, format, face similarity, all of it recorded per image and never shown until today — and the prompt is last, folded when it is long. The seed and the prompt copy in one click, because that is what you do with them.',
  },
  {
    id: '2026-07-27-canvas-deployed-at-a-glance',
    date: '2026-07-27',
    title: 'See at a glance which checkpoints you can generate from',
    blurb:
      'On the LoRA Canvas, whether a checkpoint is deployed to ComfyUI — that is, usable right now — only showed up as small print AFTER you had picked it. Every pill now carries it on its left edge: a solid sky bar means deployed, a dashed grey bar means the file is on your disk but not deployed yet (the 🎨 Generate button deploys it for you). The shape carries the message as much as the colour, a legend sits above the board, and hovering a pill spells it out in words.',
    to: '/canvas',
  },
  {
    id: '2026-07-27-bank-diverse-skips-the-odd-ones-out',
    date: '2026-07-27',
    title: '🎨 Pick diverse stops spending your first picks on memes and strangers',
    blurb:
      '"The 60 most diverse" was computed as "the 60 most isolated", and those are not the same thing: the image that is farthest from everything else in a collected bank is usually the botched frame, the meme, or the one photo of somebody else — so the first picks went to exactly what you would have rejected. A new "Skip the odd ones out" slider in the 🎨 Pick diverse popover discounts an image for being alone in the bank, while leaving variety inside your subject completely untouched. HEADS UP: it is ON at 50% by default, so this selection is no longer the same set of images it used to be — set the slider to 0 for the exact previous behaviour, or push it to 100% to be ruthless. No rescan, no GPU: it reuses the ✨ Score embeddings you already have.',
    to: '/bank',
  },
  {
    id: '2026-07-27-bank-find-by-text',
    date: '2026-07-27',
    title: 'Find images in a bank by describing them',
    blurb:
      'A new 🔤 Find by text button in the Bank\'s Curate row ranks the images you are currently looking at by how close they are to a phrase — "brunette outdoors, wide shot". It reuses the embeddings ✨ Score already computed, so there is no new model, no download and no GPU work; it even runs while a LoRA trains. Results come back closest-first, and the panel says how far the last result sits from the best, because this is a ranking and not a filter: every image scores something against every phrase. Images that were never scored cannot be found by any phrase, so they are counted and named rather than quietly missing. The first search loads the search model (about ten seconds on the CPU); after that searches are instant, and a phrase you have already used stays free even after a restart. One thing worth knowing before you trust it: it is good at subjects, settings, styles and framing, but it cannot count, it ignores the word "without" — ask for "woman without glasses" and you get glasses — and left/right means nothing to it. Describe what IS in the shot rather than what is missing.',
    to: '/bank',
  },
  {
    id: '2026-07-27-gallery-select-moved-to-the-bottom-bar',
    date: '2026-07-27',
    title: 'Cleaning up a run\'s images no longer means reaching for the top of the panel',
    blurb:
      'In a checkpoint or run gallery, Select opened the picking mode from the panel header, while everything it leads to — Select all, 🗑 Delete, the count — sat in a bar at the bottom. On a phone that was the most expensive reach in the panel. Select now lives in that same bottom bar, in indigo rather than grey so it is actually findable, and the bar is there from the moment the gallery has images. Deleting is no harder to reach by accident than before: Select sits at one end of the bar and Delete at the other, Delete stays greyed out until you have tapped at least one image, and the confirmation still spells out what leaves and where it goes. An empty gallery shows no bar at all.',
  },
  {
    id: '2026-07-27-model-not-in-comfyui-list',
    date: '2026-07-27',
    title: 'A model ComfyUI cannot load now says so, instead of a cryptic "value not in list"',
    blurb:
      'Picking a model your ComfyUI does not actually accept used to fail with a raw ComfyUI error listing other filenames — no cause, no fix. The app now checks your models against what your ComfyUI publishes, before starting anything, and names the reason. Two cases it can finally explain: a .gguf (quantised) model, which ComfyUI cannot read at all without the ComfyUI-GGUF pack and which the standard loader will never open, so no amount of moving it between folders helps — use a .safetensors build; and a model that is on your disk but belongs to a DIFFERENT ComfyUI install, which happens easily with ComfyUI Desktop since it keeps a shared models folder as well as one in its install directory. Thanks to naniii2352 (Discord) for the report and the digging.',
    to: '/settings/local-tools',
  },
  {
    id: '2026-07-27-mask-faces-on-concept-loras',
    date: '2026-07-27',
    title: 'Concept LoRAs can now leave the faces out',
    blurb:
      'A Concept LoRA quietly learns the faces of the people in its dataset, so stacking it with a Character LoRA left the two fighting over whose face to render. Concept datasets now have a Mask faces option in Advanced training options: the detected faces are weighed down in the training loss, so the concept binds to the act instead of to your models. Your images are never altered — nothing is blurred or painted over, which matters, because a blurred face would be exactly what the model learns to reproduce. Preview it on your own shots before training: the mask is drawn over the photos, the head coverage redraws live as you drag it, and images where no face was found are shown first so you can see what would slip through. Two knobs in Settings ▸ Training let you tune how much of the head is covered and how hard identity is pushed out. Off by default, and existing datasets are untouched. Reported by shivdbz2010 (GitHub).',
    to: '/datasets?section=training',
  },
  {
    id: '2026-07-27-edit-the-reference-for-free-on-your-own-gpu',
    date: '2026-07-27',
    title: 'Edit your reference photo for free — Klein and Krea 2 Edit joined ✦ Edit reference',
    blurb:
      '✦ Edit reference offered three paid API engines and nothing else, on the one gesture you naturally repeat until it looks right — every "add glasses, no, warmer light, no, plain grey background" cost another API call. Klein and Krea 2 Edit now sit at the top of that list. They render on your own ComfyUI: no key, no bill, nothing leaving your machine, so you can try a prompt ten times. They run as a job on the same queue as your generations, so you can still close the tab and come back to the Before/After. Three things are now said BEFORE you press Generate instead of after a render: what the edit costs (the "paid API call" line no longer appears on a free engine), which reference photos that engine actually uses (Krea edits the main reference only; Klein also takes the dataset\'s extra refs), and — when an engine is not ready on your machine — the one action that fixes it, in the same words the generation panel uses. If you have no ComfyUI at all, the two local engines simply do not appear rather than sitting there greyed out forever.',
    to: '/datasets?section=add',
  },
  {
    id: '2026-07-27-rerun-upscale-and-improve',
    date: '2026-07-27',
    title: 'Tuned the Upscale & improve settings? Re-run the pass on a tile in one click',
    blurb:
      'An image made by ✨ Upscale & improve had no regenerate button, and that was on purpose: the normal 🔄 restarts from your dataset\'s reference photo, so on an improved image it would have quietly produced something unrelated instead of a better version of that shot. But the improve settings became editable (steps, megapixels, base and consistency strength, and the instruction itself), and until now the only way to see a new value take effect was to delete the result and click ✨ again on the original. Those tiles now carry their own 🔄✨ button: it re-runs the improve pass on the SAME source image, with your settings as they are right now, and replaces the result in place — old file to the Trash, your typed caption kept. Images you improved with earlier versions get it too. If the source image was deleted since, the button says so instead of improving the wrong thing.',
    to: '/settings/engines',
  },
  {
    id: '2026-07-27-compare-an-improvement-with-its-original',
    date: '2026-07-27',
    title: 'Improved an image? Now you can see it next to the original before deciding',
    blurb:
      'Klein\'s ✨ Upscale & improve never touches your image: it adds a candidate beside it and waits for your verdict. But the full-screen viewer only ever showed ONE image, so judging that candidate meant memorising the original and bouncing back and forth in the grid. Open a candidate now and it carries ⧉ Compare with original: the view splits into two named panes — Original and Improved — side by side on a wide screen, stacked on a phone, where two half-width thumbnails would have proved nothing. Both panes are the same size and both images are fitted inside them, so you are looking at the same scale and the same framing even though the improved one has four times the pixels; comparing at different scales proves nothing either. For the same reason click-to-zoom is deliberately off inside the comparison and the hint says so — at 100 % the two images cover different parts of the subject. Leave the comparison and 100 % is back exactly as before. The automatic small-image rescue of scraped photos gets the same button, since it is the same question. And when the original has been deleted or purged there is no dead button: a short note says why.',
    to: '/datasets?section=images',
  },
  {
    id: '2026-07-27-canvas-run-card-opens-everything',
    date: '2026-07-27',
    title: 'Click a run on the Canvas to see everything it made, step by step — with its notes and settings',
    blurb:
      'On the LoRA Canvas, the only way to look at a run\'s images was one checkpoint at a time: click a pill, look, close, click the next pill. Clicking the run card itself did almost nothing — it opened a little menu with a single "Details" row. It now opens the gallery for the WHOLE run: every image it ever generated, grouped by the checkpoint that made it, most-trained first, so you can judge where the LoRA stopped improving without hopping between pills. The run\'s note and its checkpoint notes are right there under the images, and so are the settings it trained with. It is the same panel the pills open — same Select mode, same real delete to the recycle bin — so nothing you already knew changes. Big runs stay quick: the three most-trained steps open, the rest fold behind their counts, and if a run has more images than one panel should hold it says so instead of pretending to be complete. Two bonuses: dragging a card to rearrange the board still opens nothing, and old test images whose file name names the run but not the step now show up in a "Step unknown" group instead of being counted as untraceable.',
    to: '/canvas',
  },
  {
    id: '2026-07-27-reset-any-setting-to-its-default',
    date: '2026-07-27',
    title: 'Changed a setting and want it back? Every field now has “Reset to default”',
    blurb:
      'Until now only the prompt boxes could be put back to how they shipped. Set “Upscale & improve ▸ Steps” to 43 and there was no way home unless you happened to know the answer was 4 — the app knew, and never told you. Every editable number, path, dropdown and the enabled-engine list now carries a small ↺ Reset to default button, across Image engines, Captioning & quality, Training, Local tools, Server and Maintenance. It only shows up when the field is actually off its default, so it never adds a row of dead buttons to an already busy page. Two details worth knowing: the value it restores comes from the app itself rather than a copy baked into the screen, so when we improve a default in a later release Reset hands you the new one, not the one your version was built with; and on the fields where blank means “work it out yourself” — the engine model slugs, the Krea base model, the dataset images root — Reset empties the box instead of typing today’s answer in, so you keep following our improvements instead of freezing one. Nothing is written until you Save. Reported by the owner.',
  },
  {
    id: '2026-07-27-settings-links-land-on-the-setting',
    date: '2026-07-27',
    title: 'Links that promise one setting now drop you on that setting',
    blurb:
      'The little "Adjust improve strength →" under Upscale & improve used to open Image engines at the top and leave you scrolling a long page looking for four number boxes. It now lands directly on them, highlighted — and opens the collapsed block around a setting when there is one. Same for "Which model writes them, and how" (the captioning backend), "Source credentials" (the scraper keys card) and the Setup wizard\'s "Set the interpreter in Settings ▸ Local tools" (that exact field). One link still points at a whole section on purpose: "Defaults & cloud limits" names two settings that live in two different cards, and picking one would send half of you to the wrong one.',    to: '/settings/engines',  },
  {
    id: '2026-07-27-klein-workflow-runs-on-a-stock-comfyui',
    date: '2026-07-27',
    title: 'Klein generation now works on a normal ComfyUI — and its images will look slightly different',
    blurb:
      'If Klein variations or watermark cleaning never worked for you, this was almost certainly why: our Klein workflows asked ComfyUI for a sampling scheduler called "beta57", which ComfyUI does not have. It comes from a community node pack (RES4LYF) that quietly adds it to ComfyUI\'s own list — so the graph ran on the machine it was built on and refused everywhere else, with the real reason buried in ComfyUI\'s console ("Value not in list: scheduler"). Nothing in the workflow hinted at the dependency. Both Klein workflows now use "simple", which every ComfyUI ships and which four of our other workflows already used. Being straight with you: this is a genuine change to how Klein images render, not just a fix — if you were among the few who could already generate, your results will shift a little. That is the deliberate price of everyone running the same pipeline instead of a lucky minority running one nobody else can reproduce. Two safety nets came with it: the app now checks our graphs against what YOUR ComfyUI actually offers and names any missing value (and the pack it comes from) on the Setup screen and the engine card, instead of letting you discover it mid-batch; and a test blocks any future workflow that depends on someone\'s custom nodes without saying so. Reported by IndependentProcess0 (Reddit).',
    to: '/setup',
  },
  {
    id: '2026-07-27-sort-grids-by-score-and-similarity',
    date: '2026-07-27',
    title: 'Sort a bank or a dataset by score, sharpness or face similarity — reviewing gets a lot faster',
    blurb:
      'Both grids already measured plenty (aesthetic rating, sharpness, face similarity to your reference) and let you filter on it, but nothing could put the best — or the worst — in front of you first. The bank\'s Sort menu now offers Aesthetic ↓/↑ and Sharpness ↓/↑ next to Resolution, and the dataset grid gains its own Sort with Face similarity ↓/↑. Sorting only reorders: it composes with every filter and chip you already had, and in a bank it is done over the whole filter rather than the page on screen, so "Select all in filter" and ▶ Review walk the same order you see. Images a pass never reached always sink to the end, in both directions — a "worst first" sort that opened on un-analysed images would hide exactly what you asked for. And a sort with no data behind it is greyed out naming the pass to run, instead of silently doing nothing. Suggested by nofaceman (Discord).',
    to: '/datasets?section=images',
  },
  {
    id: '2026-07-27-comfyui-input-folder-failures-explained',
    date: '2026-07-27',
    title: 'ComfyUI in Docker: the blank 500 on Generate now tells you what went wrong',
    blurb:
      'Running ComfyUI in a separate container (or in WSL, or on another machine) could pass every setup check and then fail at the first generation with a bare "500" and no detail. The reason is that the app talks to ComfyUI over TWO channels, and only one is the network: the URL you configure, and the FILESYSTEM — every local engine hands its source image over by copying it into ComfyUI\'s input folder. That folder is not shared between containers by default, so the copy failed and nothing said so. Now that failure names the operation, the folder and the cause, and says what it needs: input/ and output/ must be visible to both sides at the same path. Settings shows the same warning on the folder overrides, and the Setup wizard checks it while you are configuring instead of letting you find out an hour later — as a warning, never a blocker, since mounting the volumes afterwards is perfectly normal. Reported by nofaceman (Discord).',
    to: '/settings/local-tools',
  },
  {
    id: '2026-07-27-promote-a-bank-selection-into-a-new-bank',
    date: '2026-07-27',
    title: 'Pull a shortlist out of a huge bank — into a new bank, not a dataset',
    blurb:
      'Promoting a selection had exactly one destination: a dataset. But when you dump 9 000 scraped images into a bank and isolate 200 candidates, a dataset is the wrong container — it is the training end of the funnel, and you are not there yet. ⬆ Promote now asks where to send the selection: an existing dataset, or a brand-new image bank you name on the spot. The new bank arrives un-triaged with every bank tool available again (scan, duplicates, framing, captions, review), and the bank you came from keeps all its images, marked as promoted. The files are COPIED on purpose — banks never share, so curating one can never mutate the other — and the dialog tells you how many megabytes that costs for your exact selection before you click, measured, not guessed. If the disk fills up mid-copy the new bank is discarded rather than left half-full and looking finished.',
    to: '/bank',
  },
  {
    id: '2026-07-27-hugging-face-token-reaches-training',
    date: '2026-07-27',
    title: 'Gated models train again — your Hugging Face login is no longer lost on the way',
    blurb:
      'Training on a license-gated base (Krea 2, FLUX.1-dev, FLUX.2 Klein) could die on "401 — you must have access to it and be authenticated", even for people who were signed in and could download the very same weights by hand. Cause: training runs with its own Hugging Face cache folder, and that override also hid the login `hf auth login` had written — so the download went out with no token at all. Now the token from Settings ▸ API keys is handed to the trainer explicitly, and if you have none saved there, the login already on your machine is found and used instead of being shadowed. And when Hugging Face does refuse, the failure block finally tells you which of the two problems you have: 401 means it saw no valid token (paste one in Settings), 403 means your token is fine but the model licence has not been accepted yet (open the model page and accept it). Those have opposite fixes and the raw error text conflates them. Reported by SurpassHR (GitHub).',
    to: '/settings/local-tools',
  },
  {
    id: '2026-07-27-stop-responds-while-a-training-starts',
    date: '2026-07-27',
    title: 'Stop answers immediately, even in the seconds a training is starting',
    blurb:
      'Starting a run hands the vision model\'s VRAM back to Ollama first. That handover is a network call, and it was made while the training queue was locked — so if Ollama was slow to answer (busy loading a model, for instance), Stop, queueing and un-queueing all sat waiting behind it, for up to a minute and a half in the worst case. Pressing Stop looked like nothing happened. The handover now runs before the queue is locked: it still frees the card before the trainer claims it, but it can no longer freeze the buttons while it waits.',
    to: '/datasets',
  },
  {
    id: '2026-07-27-install-krea-in-one-click',
    date: '2026-07-27',
    title: 'Krea 2 Edit now installs itself — no more five manual steps',
    blurb:
      'Klein has always downloaded itself; Krea 2 Edit asked you to find a GitHub repo, clone it into custom_nodes and hunt down four model files by hand. It does not any more. Setup ▸ Install now has an “Install Krea 2 Edit” button that fetches the comfyui-krea2edit node pack straight into YOUR ComfyUI (git, or a ZIP when git is not installed) plus the base model, the text encoder, the VAE and the identity LoRA — and picking Krea in the workspace and pressing Generate starts the same install for you. Files you already placed yourself are detected and never re-downloaded, and a download that turns out to be a login page instead of weights is now caught and deleted instead of crashing ComfyUI hours later. It stays out of “Install everything” on purpose: it is ~20 GB and Klein alone builds datasets. Setup also stops pretending Krea does not exist: it now has its own rows in the install list and counts as a capability, so the last screen says “11 of 12 ready” instead of congratulating you with “11 of 11” on a machine missing a whole engine. One thing no installer can do for you — ComfyUI only loads custom nodes when it starts, so the node pack shows “⟳ Restart ComfyUI” until you do, then the engine card turns green by itself.',
    to: '/setup?step=install',
  },
  {
    id: '2026-07-27-tile-engine-badge-readable',
    date: '2026-07-27',
    title: 'You can read which engine made a photo again, phone and tablet included',
    blurb:
      'In the dataset grid, the little "generated · face · Nano Banana Pro" label sat in the same top corner as the 🔄 ✏️ ⇆ ✂ 🗑 buttons. On a narrow tile — a phone, a tablet, or simply the S/M thumbnail sizes — the buttons covered it and the engine name was cut in half, which is exactly the part you need when a batch ran on several engines at once. The label now drops to the bottom-right corner of the thumbnail whenever the top row is too tight, and stays in its usual top-left spot when there is room. It reacts to the THUMBNAIL width, not the window, so large tiles on a phone keep the label at the top and small tiles on a big screen move it down. It never covers the ✓/✕ buttons or the selection tickbox, and hovering it still shows the full text.',
    to: '/datasets',
  },
  {
    id: '2026-07-27-krea-download-links-that-work',
    date: '2026-07-27',
    title: 'Setting up Krea 2 Edit: the links now lead to the actual files',
    blurb:
      'If you tried to install the Krea 2 Edit engine, two of the three download links the app handed you were dead ends: one asked Hugging Face for an account and refused, the other opened a repository that does not contain the text encoder this engine needs — so you downloaded the wrong Qwen file and the app still said "text encoder missing". All three now point straight at the public Comfy-Org/Krea-2 repository, which holds the base model, the text encoder and the VAE together, with no account and no licence to accept. The Guide (Settings ▸ Image engines ▸ Krea 2 Edit) lists every path and filename in one place, and the engine card sends you there instead of to a Setup page that never mentioned Krea.',
    to: '/settings/engines',
  },
  {
    id: '2026-07-27-see-and-edit-the-whole-prompt',
    date: '2026-07-27',
    title: 'See the prompt your engine actually receives — and edit every sentence of it',
    blurb:
      'A Klein or Krea prompt is about a thousand characters assembled from six sources, and until now you could see none of it. Settings ▸ Image engines now ends with a live preview of the COMPOSED prompt: pick an engine, a framing and SFW/uncensored, and read the exact text a real shot would be sent — including edits you have not saved yet. It generates nothing and costs nothing. Five more parts became editable next to it, with Restore default on each: the “hold the skin” order Krea sends with every shot, the outfit and expression directives injected into every human shot, the list of concrete garments Krea dresses each shot in, the rendering tail (“Professional realistic photograph” — an illustration on Anime datasets), and the per-framing shot detail, which is the box to open when your full-body shots keep coming back cropped. Leave a box alone and nothing changes: blank still means the shipped text, byte for byte.',
    to: '/settings/engines',
  },
  {
    id: '2026-07-27-delete-images-from-checkpoint-gallery',
    date: '2026-07-27',
    title: 'Throw away the misses without leaving the board',
    blurb:
      'The 🖼 gallery under a checkpoint could only show what that epoch produced — and a checkpoint you keep testing ends up holding thirty-odd renders, most of them tries you never want to see again. It deletes now: hit Select in the panel header, tap the misses (several at once), then 🗑 Delete. Nothing is destroyed on a stray tap — outside Select mode a tap still just zooms — and nothing is destroyed for good either: the files go to your system Recycle Bin, or to the app’s own Trash under Settings ▸ Storage when that is not available, and the confirmation tells you which one BEFORE you click. It also tells you the part that would otherwise be a nasty surprise: these are the same rows as the Test Studio grid, so they leave both places at once.',
    to: '/canvas',
  },
  {
    id: '2026-07-27-remove-run-takes-everything',
    date: '2026-07-27',
    title: 'Removing a run now clears everything it left behind — and tells you what that is first',
    blurb:
      'Deleting a leftover run used to drop the run entry and its notes, and quietly leave the rest: its checkpoint previews, its card position on the canvas, and the provenance link on every image you generated from it. Those strays stayed in the database forever. A removal now clears all of them in one go, and the confirmation counts what it takes before you click — "12 checkpoint notes, 8 preview links, 6 archived source images". Two things it does NOT take: the images you generated stay in the Test Studio (they only lose the link to the run), and an archived source image is freed only when no other run still uses it, so cleaning up one run can never blank another run\'s comparison.',
  },
  {
    id: '2026-07-27-klein-outfits-and-skin-hold',
    date: '2026-07-27',
    title: 'Klein datasets stop landing in the same jeans — and stop redrawing your tattoos',
    blurb:
      'Two fixes that Krea 2 Edit already had are now measured on Klein and shipped there too. Every shot gets a named garment instead of "a different outfit": asked that way, Klein answered three wide shots with three different tops but the same blue jeans and pale sneakers every time — a LoRA trained on that learns the jeans. And the shot keeps its skin: on the outdoor bust, a forehead tattoo simply vanished without the hold order and is fully there with it, same seed. Checked in both directions — on a subject with no markings at all, the hold order invents none. The wardrobe grew from 12 garments to 25, so a 40-shot dataset now spreads over 23 of them instead of repeating one six times. Nothing changes on Nano Banana, ChatGPT or OpenRouter: nothing was measured there.',
    to: '/datasets?section=add',
  },
  {
    id: '2026-07-27-memory-saving-levers',
    date: '2026-07-27',
    title: 'A bigger card no longer pays a 24 GB tax — quantisation and low-VRAM streaming are now yours to switch off',
    blurb:
      'Every training recipe was tuned so a 12B model fits in 24 GB: the base model and the text encoder are quantised and the trainer streams blocks between CPU and GPU. That is what makes training fit on most cards — and it is pure loss on a card that never needed it, costing precision and a lot of speed. Advanced options → Memory saving now exposes the three switches, and the help line is indexed on YOUR card: a 32 GB GPU is told it can turn them off and roughly what the family needs without them, a 12 GB one is told to leave them on. Nothing changed for anyone who does not touch it — the defaults are exactly what they were, on every family. One honest warning if you go too far: on Windows there is no clean out-of-memory error, the run just crawls for hours while the driver pages to system RAM. Thanks to bobba84 (GitHub) for asking.',
    to: '/datasets?section=training',
  },
  {
    id: '2026-07-27-lineage-panels-fit-a-phone',
    date: '2026-07-27',
    title: 'The run details panel no longer swallows the graph on a phone',
    blurb:
      'Tapping a run in the lineage graph opened a fixed-width drawer that covered most of a 400-px screen: you could read the run\'s settings, but not see the run they belonged to. It is now a bottom sheet on a phone — capped at 70% of the height, so the graph stays visible above it — and the same side drawer as before from a tablet up. The checkpoint image gallery already worked this way; the two now match.',
    to: '/canvas',
  },
  {
    id: '2026-07-27-face-scoring-shows-its-progress',
    date: '2026-07-27',
    title: 'Face scoring counts up instead of sitting at zero',
    blurb:
      'The scorer had been announcing every image it finished all along — nothing was reading it, so a pass over a few hundred images showed 0 for several minutes and then jumped straight to done. That is indistinguishable from a hang, and more than one run got killed for looking stuck. The indicator now moves image by image, so you can tell a slow pass from a dead one.',
    to: '/datasets?section=curation&panel=face-analysis',
  },
  {
    id: '2026-07-27-krea-names-a-broken-download',
    date: '2026-07-27',
    title: 'A half-downloaded Krea model now says so, instead of crashing ComfyUI',
    blurb:
      'The Krea 2 base sits behind a Hugging Face licence gate and the identity LoRA behind a Civitai login. Download either from a browser without going through that step and you get the web page saved as a .safetensors file — the right name, the right extension, no weights inside. Setup went green and the first generation died on a raw "Expecting value: line 1 column 1". Krea now checks each file it is about to load, names the broken one and tells you to re-download it. Truncated downloads are caught the same way. Klein has worked like this for a while; Krea now does too.',
    to: '/datasets?section=add',
  },
  {
    id: '2026-07-27-krea-reference-shape-notice',
    date: '2026-07-27',
    title: 'Krea told you why your full-body shots came back as busts — before you generate them',
    blurb:
      'Krea 2 Edit reproduces the shape of your reference photo; that is how the identity LoRA was trained, and no prompt overrides it. So a square reference cannot hold a standing figure, and the model resolves the conflict by moving in — measured: the same shot, same seed, came back a bust from a 1024×1024 reference and a full figure from an 835×1024 one. The generation panel now says so the moment you tick Krea: "18 of your 20 selected shots are body or back framings", what will happen to them, and a ✂ Crop reference to 3:4 button that opens the usual crop editor already set to a portrait ratio. It is a heads-up, not a wall — those shots still generate, they just land closer in. It only shows up for Krea (Klein and the API engines follow each shot), only when your reference really is square or wide, and only when wide shots are actually selected.',
    to: '/datasets?section=add',
  },
  {
    id: '2026-07-27-face-score-the-triage-pile',
    date: '2026-07-27',
    title: '\u{1F3AD} Analyze faces now scores the images waiting for your ✓/✕ — the ones you cannot judge by eye',
    blurb:
      'On a face with no distinctive marking, restaged from a bad photo, "is this still her?" is not a question your eye can answer — it needs a number. The face pass had one, and it was pointed at the wrong pile: it only scored images you had already kept, so freshly generated variations, the exact set you are trying to triage, never got a score. Now they do, and the 🎯 Auto-triage slider that was built for them finally has something to act on: set a threshold, keep what resembles the reference, reject what drifted. The button also tells you what it is about to do ("🎭 Analyze faces (42 · 7 new)") instead of running a mystery pass. Still on demand, still CPU-only — it never runs by itself and never touches your GPU. And if InsightFace is not installed, the button now says so with a link to Setup instead of failing after the click.',
    to: '/datasets?section=curation',
  },
  {
    id: '2026-07-27-run-freeze-and-real-compare',
    date: '2026-07-27',
    title: 'Comparing two runs now shows what the captions actually said — and which image you deleted',
    blurb:
      'Every launch, local or cloud, now freezes the whole thing: the text of every caption, a real content hash of every image, the dataset’s kind and reference photo, and the machine itself — ai-toolkit revision, PyTorch/CUDA, GPU, and the identity of the base-model file. Shift-click two runs and the compare drawer answers in full: which images arrived, which left, which captions were edited with the changed words highlighted word by word, and which images were quietly re-cropped or re-masked behind an unchanged id. Deleted images stay lookable — a deduplicated copy is kept at launch, so "+2 images, 3 captions edited" finally comes with pictures. The recipe table grew up too: steps, base model, masked training, EMA, network type, scheduler and warmup are all compared now, where before three of the most important rows silently matched nothing. Runs trained before today say so instead of pretending nothing changed.',
    to: '/canvas',
  },
  {
    id: '2026-07-27-bank-real-detail-and-origin',
    date: '2026-07-27',
    title: '\u{1F5C3}️ The bank now tells you when an image’s size is a bluff, and where it came from',
    blurb:
      'A picture enlarged from 512 to 2048 walks into a dataset claiming 2048, and the LoRA learns interpolated mush. The quality scan now also measures how far real detail actually goes and says so in plain pixels — "2048 px stored · ~512 px of real detail" — with a \u{1F9C7} Soft detail filter for the worst of them. It is a score, not an accusation: a soft focus or a heavy denoise reads the same way, so it points you at images to look at rather than deciding for you. The same pass reads the file’s own metadata and sorts the bank by \u{1F50E} Origin into \u{1F916} AI, \u{1F4F7} Camera and ❔ Unknown — three answers, never two, because scrapers and chat apps strip metadata and a silent file is genuinely unknown, not "definitely a real photo". Two more free filters come along: \u{1F39E} Black bars for video screenshots, and the JPEG quality of the last save. All of it is plain CPU work with no extra install, and a bank you already scanned picks the new numbers up on its next \u{1F50D} Scan — no full rescan.',
    to: '/settings/captioning',
  },
  {
    id: '2026-07-27-canvas-checkpoint-actions',
    date: '2026-07-27',
    title: 'Click a checkpoint on the LoRA Canvas and act on it — download, deploy, undeploy, delete',
    blurb:
      'On the board a checkpoint could only be ticked. It now opens the same actions the graph inside a run card has always had: ⬇ Download, 📦 Deploy → loras/…, ⏏ Undeploy, and the 🗑 delete that names exactly which file it removes. It is literally the same popover, so the two screens can never drift apart. When an action is not possible the reason is written where the button would be — a save that left the disk, a cloud run this machine has no link to — instead of a button that does nothing.',
    to: '/canvas',
  },
  {
    id: '2026-07-27-canvas-details-on-demand',
    date: '2026-07-27',
    title: 'The run details drawer waits to be asked',
    blurb:
      'Touching a run on the canvas used to throw the configuration drawer open, so glancing at the board meant closing a panel. Clicking a run — or a checkpoint — now opens its actions, and the drawer is one of them: ⓘ Details, filed with deploy and the rest. Shift-click still compares two runs, and dragging a card still just moves it.',
    to: '/canvas',
  },
  {
    id: '2026-07-27-canvas-generation-visible',
    date: '2026-07-27',
    title: 'A generation launched from the board can be found again — and it says where the images went',
    blurb:
      'Launch from the canvas and the progress now lives on the board itself: "1 generating · 0 queued", with its Stop. Close the settings panel, change page, reload — it is still there when you come back, instead of showing you an empty form while ComfyUI was still working. When it finishes it names the checkpoints it filled, and each one opens its gallery in a click. The board also refreshes itself as the images land, so the count on the checkpoint appears without a reload.',
    to: '/canvas',
  },
  {
    id: '2026-07-27-checkpoint-pill-readable',
    date: '2026-07-27',
    title: 'Checkpoints now say how many images they made, instead of showing an unreadable one',
    blurb:
      'A checkpoint carried a 14-pixel copy of its last image. At that size a picture tells you nothing — not the framing, not the outfit, not whether the face holds — and the little counter next to it overlapped the neighbouring checkpoint\'s. Both are gone. A checkpoint now carries a clean 🖼 N chip: how many images it has produced, and one click to open them at a size where they can actually be judged. Turn on 🔍 Big previews in a run\'s graph when you want the images on the board itself.',
    to: '/canvas',
  },
  {
    id: '2026-07-26-classify-framing-button',
    date: '2026-07-26',
    title: '\u{1F4D0} Classify framing is now a button, under the Composition bar it fixes',
    blurb:
      'Images dropped into a dataset without the head-crop option keep no shot type — which is the default on body-fidelity datasets — and the Composition bar only counts images whose shot type is known. A whole import could therefore read "Composition (0)" with every shot sitting right there, and the pass that sorts them existed with nothing to click. It is now a button in \u{1F4F8} Add images, directly under that bar, and it says how many it will treat: "\u{1F4D0} Classify framing (42)". It only appears when there is something to classify, shows its progress while it runs (a reload finds it again), and when the local vision model is not available it says which part is missing — Ollama not installed, not running, or its model not pulled — with a link straight to Local tools, instead of failing silently. Images it cannot read stay unclassified, so running it again just retries those.',
    to: '/datasets?section=add',
  },
  {
    id: '2026-07-26-canvas-generate-from-the-board',
    date: '2026-07-26',
    title: 'Test your checkpoints straight from the LoRA Canvas',
    blurb:
      'Tick the ✓ on any checkpoint on the board and the Test Studio opens right there — the same prompt, seed, format, steps and engine settings, because it is the same panel, not a copy. The new part: your picks can come from several datasets at once, so you can put three LoRAs side by side on one prompt and one seed without leaving the board. Picked a checkpoint that is not in ComfyUI yet? The button says so before it does anything: "Deploy 2 checkpoints, then generate". Picked two families by mistake? It tells you Krea and Z-Image have no engine in common instead of going quietly dead.',
    to: '/canvas',
  },
  {
    id: '2026-07-26-checkpoint-keeps-every-image-it-made',
    date: '2026-07-26',
    title: 'Every checkpoint keeps a gallery of everything it made',
    blurb:
      'Generating a second preview on a checkpoint used to replace the first one — the picture stayed on disk but you could not reach it again. They now pile up: a × N badge appears on the checkpoint and opens the whole set, newest first, whatever produced it (the canvas, the Test Studio, a comparison run). Which image belongs to which checkpoint is now recorded when it is generated instead of being worked out from the file name each time, so multi-word triggers can no longer scramble it. Older images are matched back where the evidence allows it; the ones that cannot be traced are counted and left out rather than filed under a checkpoint they may not belong to.',
    to: '/canvas',
  },
  {
    id: '2026-07-26-krea-2-identity-edit-local-engine',
    date: '2026-07-26',
    title: 'A second free engine that keeps a face from ONE photo — Krea 2 Edit',
    blurb:
      'Krea 2 Identity Edit joins Klein as a local engine: it holds the face, the body and the permanent markings from a single reference photo, with no character LoRA — which is exactly the problem you have before a LoRA exists. Tick it next to (or instead of) the others; it costs nothing but your GPU time and accepts 🔞 shots. One dial, Reference grounding, decides whether a shot follows your description or resembles the photo more closely. It runs on your own ComfyUI and needs a node pack plus four model files, so if anything is missing the engine card names it — which file, where it goes, and where to get it — instead of greying out.',
    to: '/settings/engines',
  },
  {
    id: '2026-07-26-existing-promoted-images-recover-their-framing',
    date: '2026-07-26',
    title: 'Images already promoted from a Bank take their framing back',
    blurb:
      'Carrying the framing across a promotion fixes what you promote from now on — it does nothing for what is already in your datasets. So the next start repairs those too, where it honestly can: an image that still carries the link to the Bank image it came from takes back that verdict and starts counting in the Composition bar. That link only exists for promotions made since the 25 July update, so older datasets will not change — those images stay blank on purpose, and 📐 Classify framing is what fills them in. It is a recovery, never a guess: an image you classified yourself is never touched, and a Bank verdict that was never worked out (or came back "unknown") leaves the image blank rather than inventing a bucket. Nothing to click, it runs once. Reported by axelf_ (Discord).',
    to: '/datasets',
  },
  {
    id: '2026-07-26-promoted-images-count-in-composition',
    date: '2026-07-26',
    title: 'Images promoted from a Bank now count in the dataset Composition',
    blurb:
      'The Composition bar in Add images only tallies images whose framing is known, and a promotion left that field empty \u2014 so a dataset built from a Bank read "Composition (0)" even with the shots sitting right there, then briefly showed real numbers while a generation was in flight (those images do carry a framing) and dropped back to 0 when it stopped. A promotion now carries over the framing the Bank\u0027s own \u{1F4D0} Classify framing pass already worked out, so the counts and the "missing" advice are right the moment the images land. Reported by axelf_ (Discord).',
    to: '/datasets',
  },
  {
    id: '2026-07-26-resume-keeps-its-lora',
    date: '2026-07-26',
    title: '▶ Continue now resumes the LoRA you picked — at the rank it was trained with',
    blurb:
      'One dataset can hold several runs whose saves sit side by side in the same folder, and a continuation used to be filed under whichever run was most recent rather than the one whose file it actually loaded. The lineage then showed a continuation of a run that was never touched — and, worse, the new run took its rank from the dataset\'s current settings instead of the checkpoint\'s. Rank and alpha size a LoRA\'s matrices, so they are a property of the weights, not a setting to re-pick: a cloud continuation now inherits them from the checkpoint, and a local one refuses to start rather than quietly train a different LoRA you would believe was a continuation.',
    to: '/cloud',
  },
  {
    id: '2026-07-26-lineage-edge-and-numbers',
    date: '2026-07-26',
    title: '🌳 Lineage: the missing links are back, and a run has one number',
    blurb:
      'A run linked to its parent in a straight line drew no connector at all — the most common shape in the graph was the one that vanished. It is drawn again, and a branch that continued from before its parent\'s end now reads clearly even on a phone at low brightness. Cards, the tree and the inspector also agree on the run number at last: the run\'s own number everywhere, with its cloud run spelled out next to it ("Run #107 · cloud #103") instead of two numbers that never matched.',
    to: '/cloud',
  },
  {
    id: '2026-07-26-runs-indicator',
    date: '2026-07-26',
    title: '🏋️ A live dot on Runs tells you something is still training',
    blurb:
      'A training holds your graphics card for hours when it runs here, and bills by the minute when it runs on a rented pod — but from any other page nothing said it was still going. A small pulsing dot now sits next to Runs whenever anything is training, and hovering it (or long-pressing on a phone) says where: on this machine, in the cloud, or both, with how many. A cloud run counts from the moment its pod starts provisioning, not from its first step, because that is when it starts costing you. The check is deliberately free — one flag and one count, no scan — and it pauses while the tab is in the background.',
    to: '/cloud',
  },
  {
    id: '2026-07-26-canvas-move-cards',
    date: '2026-07-26',
    title: 'Arrange the Canvas the way you think about your runs',
    blurb:
      'Run cards on the Canvas can now be dragged, and they stay where you put them — after a reload, and after the next training finishes. That second part is the real change: the automatic tree centres every run over its continuations, so a new branch used to re-flow the whole lane and quietly undo any layout you had in mind. Once you have moved something in a lane, a run that finishes later lands in free space beside your arrangement and nothing else moves. On a phone, hold a card for a moment to pick it up (a finger that slides straight away still scrolls the board). Changed your mind? ✦ Tidy up forgets every card you moved on the lanes on screen and rebuilds the automatic tree — positions are only ever a display preference, never provenance.',
    to: '/canvas',
  },
  {
    id: '2026-07-26-lora-canvas',
    date: '2026-07-26',
    title: '◉ Your whole training history on one board',
    blurb:
      'Lineage graphs used to be locked inside a single run’s card — one dataset at a time, in a fixed frame. The new Canvas tab puts every dataset you have trained on one surface you can zoom and pan: each dataset gets a lane, each run a card, and a continuation is joined to the exact checkpoint it resumed from. Untick the datasets you do not want to see, click a run to inspect what it trained with, and shift-click two runs to compare their settings — across different datasets now, which was never possible before. The graph inside your dataset panel is unchanged and keeps all of its checkpoint actions.',
    to: '/canvas',
  },
  {
    id: '2026-07-26-test-studio-starts-a-grid-faster',
    date: '2026-07-26',
    title: '⚡ "Run the test" starts your grid in a fraction of the time',
    blurb:
      'Launching a Test Studio grid used to re-read the workflow template, re-scan your LoRA folder and write to the database three separate times for every single cell — 150 database writes for a 50-cell grid — and it asked ComfyUI for its full node list twice (that answer weighs about 9 MB here: 4.8 seconds each time). A cell is now one single write, the folder is scanned once, and the node list is fetched once and reused. Measured on a 50-cell grid: the database work dropped from 150 writes to 50 and the launch itself from 129 ms to 56 ms, on top of the ~4.8 s saved on the duplicate ComfyUI probe. Fewer rapid-fire writes also means a grid launch no longer competes with a cloud run recording its progress.',
    to: '/studio',
  },
  {
    id: '2026-07-26-cloud-run-survives-a-busy-database',
    date: '2026-07-26',
    title: '💾 A busy database no longer abandons a cloud run you are paying for',
    blurb:
      'A cloud run records its progress in the local database as it goes. When something else was writing heavily at the same time — a captioning batch, a large import — that write could be refused, and the run died on the spot, three minutes in, while the rented GPU kept billing until someone noticed. Those writes now wait their turn and retry instead of killing the run.',
    to: '/cloud',
  },
  {
    id: '2026-07-26-vision-model-stays-warm-when-nothing-else-needs-the-gpu',
    date: '2026-07-26',
    title: '⚡ One-off vision jobs stop paying a 13-second model load every time',
    blurb:
      'Loading the vision model takes about 13 seconds; describing an image once it is loaded takes half a second. One-off jobs — the automatic head crop when you add a reference photo, Describe in Test Studio — used to unload it the instant they finished, so cropping five references in a row paid that load five times. The model now stays loaded for a couple of minutes, but only while nothing else wants the graphics card: with a generation queued or a training run going, it still unloads immediately, and a lease already granted is handed back the moment a generation or a training starts. New setting in Settings → Local tools if you want it longer, shorter, or off.',
    to: '/settings/local-tools',
  },
  {
    id: '2026-07-26-training-refused-during-a-vision-pass',
    date: '2026-07-26',
    title: '🛑 Starting a training during a captioning or watermark pass now tells you, instead of quietly crawling',
    blurb:
      'A vision pass (captioning, watermark or framing) holds the GPU for as long as it runs. Queued trainings already waited their turn, but hitting Train directly went ahead anyway — and because the two do not fail loudly when they overlap, nothing crashed: the graphics card simply ran out of room, part of the vision model spilled onto the processor, and both jobs slowed to a fraction of their normal speed for hours with no error to explain it. Training now refuses with a clear message while a pass is running, and points you at the queue — add the dataset there and it starts on its own the moment the pass finishes.',
    to: '/datasets',
  },
  {
    id: '2026-07-26-cloud-run-survives-a-restart-after-submit',
    date: '2026-07-26',
    title: '☁️ Restarting the app no longer destroys a cloud run that was training',
    blurb:
      'A cloud run submits its job to the pod, then records the job id. If the app restarted in the sliver of time between those two steps, the run came back not knowing it had already submitted anything — so it submitted again, the pod refused the duplicate name, and the run died as FAILED with the GPU hour already paid for. The id is now written the instant the pod accepts the job, and if a duplicate is ever refused anyway, the run reattaches to the job already on the pod and keeps polling it instead of failing. A job that was created but never actually launched is recognised as such and started for real, rather than being read as "stopped" and buried. When nothing can be salvaged, the error now tells you what to do next and says plainly that the pod is being terminated so it stops costing money.',
    to: '/cloud',
  },
  {
    id: '2026-07-26-face-scoring-off-for-anime',
    date: '2026-07-26',
    title: '🎭 Face scoring no longer pretends to read a drawn face',
    blurb:
      "Analyze faces uses InsightFace, a model trained on photographs. Run it on an anime dataset and it mostly detects nothing — so the pass finished quietly, sprayed grey \"no face detected\" tiles across the grid, and left you wondering what you had done wrong. On a dataset whose subject type is Anime the pass now stands down and says why, in place: face similarity needs a photographic face, it cannot read a drawn one. Same story for Best epoch and for scoring Test Studio cells, which were ranking checkpoints on a number nobody could measure. Auto-triage also steps aside there, because it batch-flips keep/reject from those scores. Nothing is deleted: any score from an earlier run is still in the database, and setting the subject type back to Human brings the whole thing back exactly as it was.",
    to: '/datasets?section=curation',
  },
  {
    id: '2026-07-26-anima-note-on-anime-datasets',
    date: '2026-07-26',
    title: '🎌 A quiet pointer to Anima when you train an anime character',
    blurb:
      "If your dataset's subject type is Anime and you are about to train on another family, the training panel now mentions in passing that Anima trains on an anime base. That is all it does. Nothing is preselected for you, nothing is greyed out, and no launch is blocked — training an anime character on SDXL or Z-Image is a perfectly reasonable thing to want, and Anima is local-only and needs an up-to-date ai-toolkit, so making it the forced answer would just break launches for everyone else. The line also stays away entirely if your ai-toolkit cannot run Anima, rather than recommending something you would not be able to start.",
    to: '/datasets?section=training',
  },
  {
    id: '2026-07-26-edit-the-reference-with-openrouter',
    date: '2026-07-26',
    title: '✦ Edit your reference photo with OpenRouter too',
    blurb:
      "The ✦ Edit button on the reference card only offered ChatGPT and Nano Banana Pro, so if OpenRouter was the account you actually pay for, retouching your reference meant opening a second one. OpenRouter is now a third choice in the modal, using the model you set in Settings › Image engines — the same one your variations run on. Everything else is unchanged: your reference and any extra images you drop in are all sent along so the face stays the same person, you get the Before/After, and you Keep or Discard. If the model you configured does not accept reference images, the failure now says so in OpenRouter's own words instead of looking like a refused prompt.",
    to: '/datasets?section=add',
  },
  {
    id: '2026-07-26-blank-page-on-windows-fixed',
    date: '2026-07-26',
    title: '⬜ Fixed: the app opening on a completely blank page on Windows',
    blurb:
      "On some Windows machines the server started fine, the browser opened, and you got a white screen — in Firefox and in Chrome alike, with nothing in the log to go on. The cause was outside the app: Windows keeps a registry of file types that any installed program may overwrite, and once it claims the app's script bundle is plain text, the browser refuses to run it. The app no longer asks Windows what its own files are — it states the correct type for every file it serves, so the page loads whatever else is installed on your PC. Found, diagnosed and fixed by gessyoo (GitHub #12).",
  },
  {
    id: '2026-07-26-anime-subject-type',
    date: '2026-07-26',
    title: '🎌 Anime characters get their own subject type',
    blurb:
      "Subject type offered Human, Animal, Creature, Object and Other — so a drawn character had to call itself Human, which handed it an identity lock written for photography (\"same skin tone and texture\", \"realistic photographic portrait\") and a shot list of camera-lens conventions. Pick Anime instead and the whole chain changes: the identity lock now protects what actually makes a character recognisable — hair colour and hairstyle, eye shape and iris colour, the signature outfit, the accessories, the distinctive marks — and it protects the art style itself, explicitly refusing to turn your character into a photograph or a 3D render. The shot catalogue is 55 drawn-media shots (bust-up, cowboy shot, expression sheet, and a front/side/back character-sheet turnaround nothing else offered), with four presets including Character sheet. Your existing datasets are untouched, Human included.",
    to: '/datasets?section=add',
  },
  {
    id: '2026-07-26-comfyui-custom-folders',
    date: '2026-07-26',
    title: '📂 A ComfyUI running on custom input/output folders is no longer ignored',
    blurb:
      "If you start ComfyUI with --input-directory or --output-directory, the app looked like it never got the message: it kept reading and writing under the install directory, and there was nowhere on screen to tell it otherwise. Settings › Local tools › ComfyUI now has an \"Advanced: ComfyUI folder overrides\" block with the four folders — output, input, models, LoRAs. Each empty field shows the exact path it falls back to, so you can see what the app will use instead of guessing, and a path that isn't on disk is called out in amber rather than failing silently mid-generation. Better still, if ComfyUI is running the app asks it which folders it was launched with and offers them in one click. Thanks to vykas22 (Discord) for the report.",
    to: '/settings/local-tools',
  },
  {
    id: '2026-07-26-capability-rows-are-doors',
    date: '2026-07-26',
    title: '🚪 Every line of the Capabilities list now takes you to the thing that turns it on',
    blurb:
      "Settings › Overview told you what was missing and then left you to find it: \"✗ Person masks\" was a dead end, and the four generic links underneath sent you to the top of a screen to hunt. Each of the eleven rows is now a link that lands you ON the control — the OpenRouter key field, the ComfyUI URL, the button that installs person masks — with the field scrolled to and highlighted. Rows that only need ComfyUI running now say so in amber instead of showing a red cross, and point at the connection test rather than at an install you have already done.",
    to: '/settings/overview',
  },
  {
    id: '2026-07-26-pick-the-model-of-every-api-engine',
    date: '2026-07-26',
    title: '🎛️ Pick the model for Nano Banana and ChatGPT too, not just OpenRouter',
    blurb:
      "OpenRouter let you type any model you liked, while Nano Banana and ChatGPT were stuck on whatever the release hardcoded — a newer, cheaper or better model meant waiting for an update. All three now have a plain text field, side by side in Settings › Image engines › Image models. Leave a field blank and nothing changes: that engine keeps the exact model it has always used. And when a model does not work out, the failed tile now says why in the provider's own words — unknown model, key refused, a model that will not take your reference photos — instead of the old catch-all about a content-policy refusal, and the run stops on the first one rather than paying for the same refusal once per image. Two things worth knowing before you type: every model here must accept reference images, because the generator always sends your reference photos with the prompt; and on OpenAI, gpt-image-2 is the only model that works without organization verification — a newer slug answers 403 and that is the model talking, not your key.",
    to: '/settings/engines',
  },
  {
    id: '2026-07-26-new-engines-reach-existing-installs',
    date: '2026-07-26',
    title: '🆕 An engine added by an update now shows up even if you already saved your settings',
    blurb:
      "If you had ever opened Settings and hit Save, your list of image engines was frozen at whatever existed that day — so OpenRouter shipped and simply never appeared for you, with nothing on screen hinting it was there. The longer you had used the app, the more you missed. Engines added by an update are now offered to everyone, existing installs included: open Settings › Image engines and OpenRouter is waiting, unticked engines still unticked. An engine you deliberately switched off stays off — the app remembers which engines you were shown when you made that choice, so only genuinely new ones are added, and nothing you set is overwritten.",
    to: '/settings/engines',
  },
  {
    id: '2026-07-26-ai-toolkit-without-a-venv-is-a-supported-install',
    date: '2026-07-26',
    title: '🐍 An ai-toolkit installed without a venv is set up in one click',
    blurb:
      "Plenty of ai-toolkit installs have no venv at all — the popular easy-install script ships a python_embeded folder instead, and conda, uv and system-Python setups have nothing to find either. Setup used to answer those with \"set up its Python venv per the README\", which named a cause it had never checked and a fix those installs can never follow; more than one person concluded the app required a venv. It now says what it actually found — no Python interpreter in that folder — and offers both real ways out: create a venv, or keep the Python you already run ai-toolkit with. Better still, when an interpreter is sitting in that folder the wizard spots it and applies it with a single button. Thanks to Psyko_2000 (Reddit) for reporting it.",
    to: '/settings/local-tools',
  },
  {
    id: '2026-07-26-openrouter-image-engine',
    date: '2026-07-26',
    title: '🔀 OpenRouter is now an image engine — one key instead of one per provider',
    blurb:
      'Generating a dataset meant an account at Google AND at OpenAI, one key each. If you already pay for OpenRouter — a single balance in front of every provider — there was no way in at all. There is now: paste your OpenRouter key in Settings › Image engines, tick the OpenRouter card in the generator, and it renders alongside (or instead of) the others. It reaches the SAME upstream models, so this changes who bills you, not what the images look like: the default is google/gemini-3-pro-image, exactly the weights the Nano Banana engine calls. The model is a plain text field, so you can point it at gpt-image-2, Seedream, FLUX or anything else OpenRouter serves that accepts reference images, without waiting for an update. When something goes wrong it says which thing — no key, key refused, out of credits, unknown model — and a run that cannot possibly succeed stops instead of paying for the same refusal once per image. Nothing about the existing engines changed. Suggested by jqs (GitHub #13).',
    to: '/settings/engines',
  },
  {
    id: '2026-07-26-exported-shots-keep-their-nsfw-flag',
    date: '2026-07-26',
    title: '🔞 Exporting your shot catalog no longer strips the explicit flag',
    blurb:
      'The export is both your backup and the file you hand to an LLM to write more shots — so it has to come back exactly as it left. It did not: a shot marked explicit was written out without that mark, and re-importing it produced a safe shot with the same name, quietly sent to a different image engine the next time it was regenerated. The flag now travels with the shot. Safe shots are written exactly as before, and re-importing an older export is unaffected.',
    to: '/datasets',
  },
  {
    id: '2026-07-26-interpreter-search-says-when-it-broke',
    date: '2026-07-26',
    title: '🔎 "No Python found" now means it looked — not that the search broke',
    blurb:
      'The ⚡ picker that lets ✨ Score borrow a Python you already have showed the same empty screen in two very different situations: this machine genuinely has nothing to borrow, and the search itself failed. The second one reads as the first, so nobody had any reason to press ↻ Check again — which is exactly what would have fixed it. A failed search now says so, shows what went wrong, and points at the retry.',
    to: '/bank',
  },
  {
    id: '2026-07-26-score-gpu-check-stops-guessing-and-stops-repeating',
    date: '2026-07-26',
    title: '⚡ ✨ Score stops mistaking a slow answer for "no GPU here"',
    blurb:
      'To know whether a scoring pass will use your card, the app starts your scoring Python and asks it. On a cold machine — a big PyTorch, an antivirus reading every DLL — that question can take longer than the minute it was given, and the app filed the silence as "this Python has no CUDA". Two costs followed: the pass took the card without reserving it, so ComfyUI stayed loaded and a training start was still allowed against it; and because a non-answer was never remembered, opening a bank re-asked the same slow question every couple of seconds. The check now gets the same 90 seconds the interpreter picker already used, remembers "did not answer" for a minute instead of re-asking, and treats it as "assume the card is in use" on a machine that has one — rather than as a no.',
    to: '/bank',
  },
  {
    id: '2026-07-26-reinstall-never-writes-into-a-borrowed-python',
    date: '2026-07-26',
    title: '🔒 Reinstall no longer writes into the Python you only lent us',
    blurb:
      'When you point ✨ Score at a Python you already have — ai-toolkit\'s, ComfyUI\'s, a conda env — the picker promises those environments are checked, never changed. The Install / ↻ Reinstall button in Setup ▸ Quality tools did not keep that promise: it took your borrowed interpreter as its install target and pip-installed torch, OpenCLIP, Transformers and timm straight into the environment that runs your training. It now refuses, installs nothing, and prints the exact command if you do want those packages there. Clearing the setting ("Back to the app default") makes the button build the app\'s own environment again, as before.',
    to: '/settings/local-tools',
  },
  {
    id: '2026-07-26-delete-rejected-waits-for-its-own-check',
    date: '2026-07-26',
    title: '🗑 Delete rejected now waits for its own safety check before it will run',
    blurb:
      'Before that button deletes anything, the app asks where the files would go and whether another bank shares them — that is where the "⚠ Another bank uses these files" warning comes from. If that question failed, or simply had not come back yet, the warning quietly did not appear, the dialog claimed the files were "deleted for good", and the button armed anyway: the protection vanished exactly when it could not do its job. The delete button now stays disabled until the check has answered, says which of the two it is waiting on, and never states a destination it has not verified.',
    to: '/bank',
  },
  {
    id: '2026-07-26-stop-during-a-watermark-pass-loses-nothing',
    date: '2026-07-26',
    title: '🛑 Stopping a watermark pass no longer costs you a cleaned image',
    blurb:
      'A watermark re-scan always looks at the ORIGINAL pixels, so it throws away the cleaned version of an image just before analysing it again. If you had set the vision passes back to one image at a time (Settings ▸ Local tools ▸ Ollama), pressing Stop took one image further than it analysed: that last image lost its cleaned file and got no new verdict in exchange. Stop now lets go before touching the next image, at any speed setting — so a cleaned image is only ever given up in return for a fresh answer. Everything already analysed still stays analysed, exactly as before.',
    to: '/settings/local-tools',
  },
  {
    id: '2026-07-26-move-folder-accepts-a-pasted-path',
    date: '2026-07-26',
    title: '📦 Move folder…: a pasted path is accepted the way you paste it',
    blurb:
      'Right-clicking a folder in Windows and choosing "Copy as path" wraps it in quotes — the most natural way there is to hand the app a folder. The Move folder… dialog checked it happily, then dropped the whole verdict off the screen and left "Repoint this bank" greyed out for good, with nothing said. It was comparing your text to the tidied-up path it had resolved, and those two are never identical. Quotes, a trailing backslash, forward slashes, a junction — all accepted now, and once the check has run the field shows the folder the app actually resolved, so the number you confirm belongs to the folder you can see.',
    to: '/bank',
  },
  {
    id: '2026-07-26-cloud-checkpoint-rescue-is-never-cut-short',
    date: '2026-07-26',
    title: '💾 A cloud checkpoint being brought home can no longer be lost on the way',
    blurb:
      'The safety net that shuts down a silent cloud run had one blind spot, and it was the worst one possible: the very end, when the training has succeeded and the app is pulling the finished LoRA off the pod. Some hosts serve that file in fits and starts — a big checkpoint can take a long while — and for all that time the run reported nothing, so it looked exactly like a run that had died. The pod could be terminated with the result still on it: the work done, the money spent, and nothing to show for it. The transfer now reports itself. The run card says "Downloading" and shows the megabytes climbing, so you can see it is working rather than guess, and no watchdog treats a live transfer as silence — including after you press Stop, where rescuing the checkpoint is the whole point. A transfer that genuinely dies is still caught, just as before.',
    to: '/cloud',
  },
  {
    id: '2026-07-26-continue-training-appears-and-resumes-the-final-save',
    date: '2026-07-26',
    title: '▶ Continue training: the dialog shows up, and the last checkpoint can be resumed',
    blurb:
      'Two things made "Continue training" look broken from the Checkpoints & LoRAs page. The dialog did open — invisibly, in the section you were not looking at — so the click produced nothing at all until you happened to walk over to Training and found it waiting there. It now opens where you clicked, in any section. And the final checkpoint of a finished run is resumable again: that file carries no step number in its name, so the graph called it "3k" while the list quietly filed it under the previous save (2750) and refused the pill you had just clicked, blaming a family/base/variant selection that was perfectly correct. Both views now agree on the run\'s real last step, so "Continue from here" on the last pill resumes the run\'s true end — and when a save really is missing here, the message says the true reason (it lives in its cloud run: continue it from the Runs page) instead of sending you to change a setting that was never in cause. A greyed Continue button also states why, rather than sitting there silently.',
    to: '/datasets?section=checkpoints',
  },
  {
    id: '2026-07-26-stop-training-button-says-what-it-does',
    date: '2026-07-26',
    title: '⏹ The button that stops your training now says so',
    blurb:
      'While a training ran, the red button beside Train read “Finish / re-enable ComfyUI” — which sounds like tidying up, and instead killed the run. People lost hours-long trainings to one click, and at least one just stopped touching it rather than find out what it did. It now reads ⏹ Stop training, hovering it tells you what survives, and it asks for confirmation before ending a run — because a training you meant to keep is worth one extra click. What you keep is unchanged and now stated up front: every checkpoint already saved stays, testable in the Studio and resumable with ▶ Continue, and ComfyUI still gets the GPU back. Reported by wannadecryptor (Discord).',
    to: '/datasets?section=training&panel=launch',
  },
  {
    id: '2026-07-26-cloud-stop-that-cannot-lie',
    date: '2026-07-26',
    title: '🛑 Stop really stops the pod — and a frozen cloud run stops billing you',
    blurb:
      'A rented GPU bills by the hour whether or not anything is happening, so two things had to become impossible. First: Stop can no longer answer "ok" without doing anything. If nothing is left in a state to wind the run down — the app was restarted, the connection to the pod wedged — the pod is now terminated on the spot, and if even that fails you get an error naming the instance to destroy in the vast.ai console instead of a reassuring message. Second: a run that goes completely silent is caught from outside itself. The run card warns as soon as a training run stops reporting, and after 45 minutes of total silence the pod is shut down automatically — checkpoints already downloaded are kept. The runtime cap is enforced from that same place, so it holds even if the run\'s own supervision died. Phases that are quiet by design — booting, uploading, downloading the result — are never cut. You can change the delay, or set it to warn only, under Settings ▸ Training ▸ Cloud training.',
    to: '/cloud',
  },
  {
    id: '2026-07-25-score-borrows-your-gpu-python',
    date: '2026-07-25',
    title: '⚡ ✨ Score can borrow the CUDA Python you already have',
    blurb:
      'Score ships CPU-only PyTorch on purpose — a first install stays small instead of pulling 2.5 GB on people with no card. On a machine that has one, that meant hours: nearly three of them on a 30 000-image bank. But if you train LoRAs or run ComfyUI, this machine already has a proven CUDA PyTorch sitting right there. Score can now use it — no download, no second install. Open a bank, and where the CPU warning appears click "Use a GPU Python I already have": the app checks the interpreters it knows about (ai-toolkit, ComfyUI, its own) and tells you, package by package, which ones can really run the pass. One that has CUDA but is missing OpenCLIP is refused by name rather than accepted and crashed an hour in — and nothing is ever installed into an environment the app did not create: it shows you the command and lets you decide. Typing a path yourself is a first-class route, not a fallback — point it at an interpreter or at the environment folder holding it (venv, conda, uv, a portable bundle, the system Python, another disk; spaces and accents are fine), and the app works out the rest instead of assuming a layout. No particular PyTorch or CUDA version is demanded, so an old card and a brand-new one are equally welcome. If this machine has no NVIDIA card, it is told so plainly rather than sold a CUDA install it could not use — borrowing is still offered there, purely to avoid installing the same packages twice. And if you never open the dialog, nothing changes: it is an offer, never a prerequisite. Re-check after installing something, and go back to the app default whenever you like.',
    to: '/bank',
  },
  {
    id: '2026-07-25-bank-vision-passes-twice-as-fast',
    date: '2026-07-25',
    title: '⚡ The bank passes that took all night now take half of it',
    blurb:
      'Watermark scan, framing and captions ask the vision model about every single image in the bank, and they used to ask one image at a time — on a 30 000-image bank that is most of a night, and almost all of it was spent waiting on the round-trip rather than on your GPU. Those calls now overlap. Measured on the real thing, the same set of images finishes in half the time (2× faster), so a pass that ran from midnight to lunchtime lands before breakfast. Nothing about the results changes, Stop still stops within seconds, and everything already analysed when you stop stays analysed — a re-run picks up where it left off instead of starting over. You can tune it, or put it back to strictly one at a time, under Settings ▸ Local tools ▸ Ollama.',
    to: '/settings/local-tools',
  },
  {
    id: '2026-07-25-bank-move-folder',
    date: '2026-07-25',
    title: '📦 Move a bank to another disk without losing a single analysis',
    blurb:
      'A bank was nailed to the folder path you gave it: move that folder to a bigger drive and the bank had no way to follow — and worse, running a pass while the files were away marked every image "unreadable" and auto-rejected it, quietly wiping the triage of a 30 000-image bank. Both are fixed. Move the folder, then press 📦 on the bank\'s card: the app checks the new location FIRST and tells you how many of this bank\'s images are actually in there and how many are not, before anything is written. Confirm and every score, duplicate group, face cluster, caption and keep/reject decision is still there. A folder holding none of your images is refused instead of accepted in silence, nothing is ever deleted — and a pass that finds the files missing now stops and says the folder appears to have moved, rather than grading absent files as broken ones.',
    to: '/bank',
  },
  {
    id: '2026-07-25-crashed-run-log-is-findable',
    date: '2026-07-25',
    title: '🪵 A training that dies in seconds now hands you its log',
    blurb:
      'When a run crashed we told you to open training.log via "📂 Run folder" — and that button opened the folder of the checkpoints, one level BELOW the folder the log is written in. A run that died at boot has no checkpoints, so the button silently created that empty folder and showed you exactly nothing. You could search the disk and never find a log that had been there all along. The button now opens the run\'s own folder, log included (the checkpoints are one click deeper), and a failed run gets its own 📂 Open run folder button right next to the error — no more digging under a collapsed "Checkpoints" section for it. Reported by wannadecryptor (Discord).',
    to: '/datasets',
  },
  {
    id: '2026-07-25-bank-score-frees-the-gpu',
    date: '2026-07-25',
    title: '🖥️ ✨ Score no longer locks your GPU to compute on the CPU',
    blurb:
      'The scoring pass took the GPU-exclusive lock every time — unloading ComfyUI and blocking any training start for its whole run — even though it installs CPU-only PyTorch and was computing on the processor anyway. The worst of both worlds: the card idle and unusable, the pass slow regardless. It now takes that lock only when it really runs on the card, so a scoring pass on the CPU leaves you free to train or generate at the same time. It also finally tells you which of the two is happening, how long the images left would take, and — only if you actually have an NVIDIA card — what a CUDA install into the scoring environment would cost you in download size.',
    to: '/bank',
  },
  {
    id: '2026-07-25-bank-promote-is-rechecked',
    date: '2026-07-25',
    title: '🗃️ Delete a promoted image in a dataset and the bank offers it again',
    blurb:
      'Promoting marked a bank image as done and that was final: delete the copy in the dataset and the bank still refused to offer it, so a bank you had emptied into a dataset could end up announcing nothing left to import — as if it had lost your images. The bank now CHECKS instead of remembering. It reads whether the dataset really still holds each image, so deleting one there puts exactly that one back on offer, the ⬆ promoted badge disappears with the copy, and the counter matches what you can see. Nothing was ever deleted from your bank — but it looked like it, and that was our fault.',
    to: '/bank',
  },
  {
    id: '2026-07-25-bank-scan-skips-rejected',
    date: '2026-07-25',
    title: '⚡ The quality scan stops re-analysing images you already threw away',
    blurb:
      'Every other bank pass skipped rejected images; the quality scan did not. On a 30 000-image bank with two thirds rejected, running it again spent two thirds of its time on shots you had discarded. It now leaves them alone, so a rescan takes a fraction of the time and the progress bar counts only real work. Un-reject an image and it comes straight back into the pool — and a first scan still flags unreadable files exactly as before.',
    to: '/bank',
  },
  {
    id: '2026-07-25-bank-shared-folders-warning',
    date: '2026-07-25',
    title: '🛡️ Two banks over the same folder can no longer amputate each other',
    blurb:
      'Nothing stops a bank pointing at a folder that sits inside another bank\'s — and that is fine until 🗑 Delete rejected, the one action that removes real files: the other bank simply finds them gone, along with every decision you made on them. Creating such a bank now says so up front, and the delete confirmation names the other bank and how many of its files are about to disappear. It also tells you where the files GO before you click. And when the optional send2trash package is missing — which is most installs — deleted photos now land in the app\'s own Trash instead of being erased for good.',
    to: '/bank',
  },
  {
    id: '2026-07-25-keep-custom-shot',
    date: '2026-07-25',
    title: '⇪ Keep a custom shot you wrote — it stops dying with your browser cache',
    blurb:
      'A shot you typed into the ✨ Custom shot box was stored in the browser and nowhere else, so clearing its data quietly took your prompts with it — and nothing on screen ever said so. Every custom card now has a ⇪ button: press it and the card moves into the 📥 Imported group, saved with the app instead of the browser. It survives a cache wipe, shows up on your other devices, and rides along in the backup. The card keeps its identity, so a shot preset that had it selected still works, and if its name clashes with a built-in shot the app says which one and leaves the card untouched rather than making a duplicate. Follow-up to the shot catalog import — idea by ashish.sinha on Discord.',
    to: '/datasets?section=add',
  },
  {
    id: '2026-07-25-shot-catalog-json',
    date: '2026-07-25',
    title: '📥 Bring your own shots — import a catalog written by an LLM',
    blurb:
      'Typing 30-40 shot prompts by hand was the only way to go beyond the built-in catalog. Under the shot grid, 📥 Shot catalog (JSON) now exports the current catalog as a template, so you can ask a chat assistant for forty more shots in the same shape and import the file it writes. Bad files do not get through: an unknown framing, a missing prompt or a label that clashes with an existing shot is refused by name, and nothing at all is saved until you have seen the summary — a forty-shot file with one broken entry can never leave you with thirty-nine and a mystery. Imported shots live in their own group per subject type, never replace a built-in, can be removed one by one, and are stored with the app rather than in the browser, so they survive a cache wipe and follow you to your phone. Idea by ashish.sinha on Discord.',
    to: '/datasets?section=add',
  },
  {
    id: '2026-07-25-non-human-catalogs-deeper',
    date: '2026-07-25',
    title: '🐕 Animals, creatures and objects get a real shot catalog',
    blurb:
      'The non-human catalogs shipped as first drafts: 16 shots for an animal against 53 for a human, which is not enough to build a varied dataset. Animal now offers around 59 shots (head angles and expressions, light directions, poses from sleeping to jumping, snow, water, forest, city, plus coat, paw and tail details), Creature 40, Object 30 and Other 22. Their presets are curated instead of "select everything", so one click no longer queues — or bills — the whole catalog: each type has a balanced spread plus focused sets (head, full body, studio, in context).',
    to: '/datasets?section=add',
  },
  {
    id: '2026-07-25-training-failure-real-cause',
    date: '2026-07-25',
    title: '🔎 A failed run now tells you what actually killed it',
    blurb:
      'When a local run died, the red box printed the last lines of the log — and ai-toolkit\'s last lines are usually a harmless huggingface_hub FutureWarning about HF_HUB_ENABLE_HF_TRANSFER. People spent hours chasing a deprecation notice. The box now quotes the real cause: the last traceback, or the last genuine error line. A warning is never shown in red as if it were the reason, and when the log truly holds no error we say exactly that instead of pretending. RTX 50-series owners get more: the pre-flight and the failure box both detect the Blackwell trap — CUDA reports your card, training starts, then dies at the first computation because the PyTorch in your ai-toolkit venv only ships kernels up to sm_90 — and hand you the one pip command that fixes it. Reported by wannadecryptor on Discord.',
    to: '/datasets?section=training',
  },
  {
    id: '2026-07-25-identity-prompts-per-subject',
    date: '2026-07-25',
    title: '🐾 Identity prompts no longer leak between subject types',
    blurb:
      'Tweak the identity instruction on an Animal dataset and your Human datasets used to inherit it — which is exactly how variations of a person came back with tails, extra limbs and odd footwear. Each subject type (Human, Animal, Creature, Object, Other) now keeps its OWN set of identity prompts, and both places you can edit them say which subject you are editing: the ✎ button next to Extra refs edits the prompts of the dataset you have open, and Settings ▸ Image engines has a Subject type picker with a dot on every type you have customised. Anything you had already written stays where it was, on the Human set. Reported by ashish.sinha on Discord.',
    to: '/settings/engines',
  },
  {
    id: '2026-07-25-klein-generation-steps',
    date: '2026-07-25',
    title: '🎚️ Klein generation steps are yours to set',
    blurb:
      'The local Klein engine always spent exactly 5 sampler steps on each variation, with no way to change it. Settings ▸ Image engines ▸ Klein generation quality now exposes that number (1–50). It still starts at 5, so nothing changes until you raise it; more steps render more cleanly and cost proportionally more time. It is a rendering knob, not a fix for anatomy — extra limbs come from the identity prompt, not from the step count. Raised by ashish.sinha on Discord.',
    to: '/settings/engines',
  },
  {
    id: '2026-07-24-bank-watermark-two-level-cleaning',
    date: '2026-07-24',
    title: '🚩 Banks can now REMOVE the watermarks they find — in two safe steps',
    blurb:
      'Finding the marked images in a bank was all you could do: cleaning them meant promoting the watermark into your dataset first, then cleaning there. A bank now cleans them itself, in two steps you launch by hand. Step 1 crops off the marks sitting in a border — no model, no GPU, and not a single invented pixel. Step 2 repaints whatever a crop can\'t remove, with LaMa (fast) or Klein (slower, and the only one that clears a mark on the subject). Each step shows how many images it still has to work on, so you can see how far down the funnel you are. Your own files are never modified: the cleaned version is a copy the app keeps, promoting sends that cleaned copy to the dataset, and ↩ Undo simply throws it away.',
  },
  {
    id: '2026-07-24-bank-review-one-by-one',
    date: '2026-07-24',
    title: '▶ Triage a bank at speed — one full-size image, Keep / Reject / Skip',
    blurb:
      'Judging a photo from a 140-pixel thumbnail was never really possible, so the last call always meant opening files by hand. The bank now has a review mode: hit "▶ Review one by one" above the grid and the images of your current filter come up full size, one after the other. ✓ Keep, ✕ Reject and ⏭ Skip each save and jump straight to the next — K, R and S on the keyboard, ←/→ to move without deciding, Esc to leave. Skip means "not now": the image stays undecided, and doesn\'t come back in that run. Tick 🎲 Random order and it walks what\'s left shuffled instead of in folder order — on a scraped dump of 3 000 shots that\'s the difference between 200 near-identical frames in a row and a representative sample from the first click; nothing you have already seen is ever shown twice. Each decision is saved on the spot, so closing after fifty of them keeps all fifty, and the counters at the top follow along.',
    to: '/bank',
  },
  {
    id: '2026-07-24-bank-folder-auto-refresh',
    date: '2026-07-24',
    title: '🗃️ Images you add to a bank\'s folder now show up on their own',
    blurb:
      'A bank used to be a snapshot: whatever was in the folder the day you created it, forever. Keep scraping into that folder and the new shots simply never appeared — the only way in was to rebuild the bank and lose your triage. The folder is now re-walked automatically when you open the bank list or a bank, and anything new joins it as undecided ("42 new image(s) found in the folder"), ready for the next scan. Strictly additive: not one keep, reject, score or caption is touched. Files you removed from the folder are reported, never deleted from the bank — a disconnected drive can\'t erase your work.',
    to: '/bank',
  },
  {
    id: '2026-07-24-checkpoints-panel-deployed-state',
    date: '2026-07-24',
    title: '✓ The Checkpoints panel now says which LoRAs are already in ComfyUI',
    blurb:
      'Every checkpoint used to offer "Import → loras/…", even the ones already deployed — and the only way back out lived in a separate list under a red 🗑 that read like destruction. A deployed checkpoint now shows "✓ Deployed" with an ⏏ Undeploy right there, exactly like the run graph: reversible, your training save is kept and you can deploy it again. The list below keeps only the LoRAs no checkpoint on the page explains (imported before run tagging, or dropped in by hand).',
    to: '/datasets?section=checkpoints',
  },
  {
    id: '2026-07-24-per-run-staging-cleanup',
    date: '2026-07-24',
    title: '🧹 See what each training run weighs on disk — and clean just that one',
    blurb:
      'Every finished run on the Runs page now shows how much disk its staging folder still holds ("8.2 GB on disk"), with a 🧹 button to move just that run to the trash — no more all-or-nothing cleanup of a whole history. Runs still training, and pods kept for manual recovery, are left alone exactly as before. The messages are honest too: the trash lives on the same disk, so they now tell you to empty it in Settings to actually reclaim the space, and "nothing to clean" no longer looks like a failed click.',
    to: '/cloud',
  },
  {
    id: '2026-07-24-explicit-undeploy',
    date: '2026-07-24',
    title: '⏏ Undeploy a LoRA from ComfyUI without fearing you are deleting it',
    blurb:
      'In the run graph, a deployed checkpoint used to be a dead end: a "✓ Deployed" badge, and the only way back was a discreet 🗑 that read like destruction. It now offers ⏏ Undeploy right next to that badge — the exact counterpart of 📦 Import, and reversible: only the ComfyUI copy is removed, your training save stays and can be deployed again any time. The 🗑 stays where it belongs, for deleting the training save itself.',
    to: '/cloud',
  },
  {
    id: '2026-07-24-subject-type-selector',
    date: '2026-07-24',
    title: '🐾 Build LoRAs of animals, objects and creatures — not just people',
    blurb:
      'The generation panel has a new Subject type selector: Human, Animal, Creature, Object or Other. Pick anything but Human and the shot list stops assuming a person — the prompts and the identity lock switch to that subject (a dog keeps its breed and markings, a product keeps its shape and logo), the shot cards become head/full-body/detail/rear instead of face/bust, and you get a preset tuned for it. Existing datasets stay exactly as they were (Human). Suggested by ashish.sinha on Discord.',
    to: '/datasets?section=add',
  },
  {
    id: '2026-07-24-caption-replace-case-insensitive',
    date: '2026-07-24',
    title: '🏷️ Find & Replace in captions now ignores case',
    blurb:
      'Clicking a frequent word like "bulldog ×41" and stripping it used to update 0 captions — because the text replace was case-sensitive while your captions said "Bulldog". Text mode now matches whole words regardless of case, the same rule the filter and the word counts already used, so stripping a word actually removes all of them. Whole-word too, so "red" never eats the "red" inside "colored".',
    to: '/datasets?section=captions',
  },
  {
    id: '2026-07-24-test-studio-all-recent-prompts',
    date: '2026-07-24',
    title: '🧪 The Test Studio keeps all your recent prompts, not just ten',
    blurb:
      'The "Recent prompts" strip in the Test Studio used to stop at the ten most recent — older ones you wanted to reload were simply gone. The cap is removed: every distinct prompt from your recent test history is there to click and reload, across all your datasets. (It still scans your latest activity for speed, so truly ancient prompts eventually roll off.)',
    to: '/studio',
  },
  {
    id: '2026-07-24-edit-reference-photo',
    date: '2026-07-24',
    title: '✦ Edit the reference photo with a prompt',
    blurb:
      'The reference card has a new ✦ Edit button. Type what to change — "plain studio-grey background", "add glasses", "warmer lighting" — optionally drop in a few extra reference images, and ChatGPT or Nano Banana edits your reference. You get a Before/After side by side, then Keep it or Discard it. The edit runs on the server, so you can close the tab (or lock your phone) and come back — the Before/After is waiting for you, and a slow render never gets lost. Editing replaces the reference and only affects future variations, never images you already generated. Each edit is a paid API call.',
    to: '/datasets?section=add',
  },
  {
    id: '2026-07-24-anima-training-family',
    date: '2026-07-24',
    title: '🎨 Train Anima LoRAs (anime model)',
    blurb:
      'Anima — the open anime image model from circlestone-labs — is now a first-class training target, right next to Z-Image, SDXL, Krea 2, FLUX.1 and FLUX.2 Klein. Pick "Anima" as your dataset\'s target model and it trains on the official public base (no gated download, no Hugging Face licence to accept), with prose captions and researched defaults (rank 32, weighted timesteps, the anime-tuned preview negative) already dialled in — plus one-click Character and Concept presets. Training runs locally on an up-to-date ai-toolkit; cloud training for Anima is coming once the GPU image is verified.',
    to: '/datasets?section=add',
  },
  {
    id: '2026-07-23-multi-engine-generation',
    date: '2026-07-23',
    title: '⚡ Generate with several engines at once',
    blurb:
      'The engine cards are now checkboxes: tick Klein, Nano Banana and ChatGPT together instead of picking one and re-running the batch to try another. Two modes decide what that means — "Split across engines" shares your shots between them, so you get a more varied dataset for exactly the same number of images and the same cost, while "All engines" sends every shot to every engine so you can compare the renders side by side and keep the best. The image count and the estimated price update live as you tick, the button carries the real total, and each generated tile is now labelled with the engine that made it. Every engine also has its own colour — green went back to meaning "kept".',
    to: '/datasets?section=add',
  },
  {
    id: '2026-07-23-crop-extra-references',
    date: '2026-07-23',
    title: '✂ Crop your extra reference photos too',
    blurb:
      'Extra references could be added and removed, but never reframed: a great side-angle shot with half a living room in it stayed that way. Each extra-ref thumbnail now has its own ✂ button, opening the same crop editor as the main reference. The full frame is kept behind the scenes, so you can widen a crop back out later instead of only tightening it — and that also works on the extra refs you imported before today.',
    to: '/datasets?section=add',
  },
  {
    id: '2026-07-23-stop-generation-works-again',
    date: '2026-07-23',
    title: '⏹ Stop generation works again',
    blurb:
      'Since July 14th the ⏹ Stop generation button was greyed out for the whole time a batch was running — the one moment you actually need it. Clicking did nothing, on all three engines (Klein, Nano Banana and ChatGPT), so a batch you already regretted had to run to the very end. It is clickable again, it says "Stopping…" while it works, and it keeps the promise printed next to it: the generations that had not started yet are skipped and never sent to the API, so they cost you nothing.',
    to: '/datasets',
  },
  {
    id: '2026-07-23-interface-fully-in-english',
    date: '2026-07-23',
    title: '🔤 The last French labels are gone',
    blurb:
      'A few corners of the app were still speaking French. The 🗃 Bank\'s four zones read ① Analyze, ② Triage, ③ Curate, ④ Promote instead of their French names, the Pexels language picker offers "French" rather than "Français", and the quotes wrapping model and checkpoint names in dialogs and banners are no longer French guillemets. Nothing moved and nothing was renamed under the hood — same zones, same buttons, same saved settings, just read in one language.',
    to: '/bank',
  },
  {
    id: '2026-07-23-bank-cards-show-their-first-images',
    date: '2026-07-23',
    title: '🗃 Tell your banks apart at a glance',
    blurb:
      'The 🗃 Bank list used to be a wall of names and folder paths — you had to open a bank to remember what was in it. Each card now shows a strip of its first five images, with a "+N" badge for the rest. It works on banks you never scanned too (the thumbnails are made on the spot), and clicking one opens the bank.',
    to: '/bank',
  },
  {
    id: '2026-07-23-one-box-per-editable-prompt',
    date: '2026-07-23',
    title: '✎ One box per editable prompt — the real text, ready to edit',
    blurb:
      'Each editable prompt in Settings › Image engines showed you two things: an empty field, and a grey read-only copy of the built-in text you had to click "Load default to edit" to use. Now there is a single box, already holding the exact prompt in use — put your cursor in it and change a word. Nothing is stored while the text still matches the built-in one, so you keep receiving improvements to that prompt instead of being frozen on today\'s wording, and "Reset to default" puts you back there in one click. The line under the box always says which of the two you are on.',
    to: '/settings/engines',
  },
  {
    id: '2026-07-23-edit-identity-instruction-from-extra-refs',
    date: '2026-07-23',
    title: '✎ Tune the identity instruction where you add the extra refs',
    blurb:
      'Adding "Extra refs" is how you ask for a stronger identity lock — but the instruction those photos actually ride on lived three clicks away in Settings. A small ✎ next to the + now opens it right there, with the shipped text ready to edit and a reset. It shows both instructions, because the engines do not share one: Nano Banana and ChatGPT read the multi-reference identity lock, Klein reads its own restage block whatever the number of references. The one your currently selected engine uses is badged, so you can no longer spend ten minutes rewriting a text your generations ignore.',
    to: '/datasets?section=add',
  },
  {
    id: '2026-07-23-one-backup-menu-in-the-library',
    date: '2026-07-23',
    title: '💾 One Backup menu instead of three loose controls',
    blurb:
      'The Datasets header used to line up "Back up everything", a bare "Include trained LoRAs" checkbox and "Import backup" side by side — a checkbox floating next to a button it silently belonged to. They are now one 💾 Backup menu, with the LoRAs option sitting right under the action it changes, so it is obvious what it applies to. "+ New dataset" stays where it was. A backup in progress is still impossible to miss: the button itself reads "Backing up…" and the progress window keeps running whether the menu is open or closed.',
    to: '/datasets',
  },
  {
    id: '2026-07-23-import-from-a-bank-in-add-images',
    date: '2026-07-23',
    title: '🗃 Pull images straight from a bank, without leaving the dataset',
    blurb:
      'Triaging a big folder in the 🗃 Bank and then feeding a dataset with it meant going back to the Bank page and promoting from there. "Add images" now offers "Import from a bank" right next to the photo dropzone and the scraper: choose which bank, see how many of its kept images would actually land in THIS dataset (near-duplicates and images already here are excluded from the count), and start. It copies in the background, so the grid fills in on its own. When a bank offers nothing, it says which kind of nothing: a bank you never triaged tells you how many images are still undecided and offers to open it, while a bank whose kept images are all already here simply says so.',
    to: '/datasets?section=add',
  },
  {
    id: '2026-07-23-delete-a-save-from-the-graph',
    date: '2026-07-23',
    title: '🗑 Undo a checkpoint step by step, straight from the lineage graph',
    blurb:
      'A run that saved every epoch fills the disk fast, and until now the only way to get rid of one was to leave the graph, switch to the flat list and hunt the filename down. Click any checkpoint pill and its actions now end with a quiet delete row that walks backwards through what you did: while the checkpoint is deployed it reads "Remove from ComfyUI" and takes out only the imported copy — the training save stays, so nothing is lost. Once it is no longer deployed the same row reads "Delete the training save" and clears the run file itself. The confirmation always names which of the two you are about to delete, says what survives, and reminds you it goes to the trash (recoverable until you empty it in Settings). If the checkpoint is the one pinned as the dataset\'s ★ best settings in the Test Studio, the warning comes first. Cloud saves are deleted on the right run, and a run still training keeps its saves.',
    to: '/datasets?section=training',
  },
  {
    id: '2026-07-23-grid-filter-by-decision',
    date: '2026-07-23',
    title: '🔎 Show only what still needs a ✓/✕',
    blurb:
      'The Images header already told you "254 awaiting ✓/✕" — but nothing could pull those 254 out of a 508-image grid, and "select all" always took all 508. A new Show row above the grid filters by decision: All, Undecided, Kept, Rejected, or Improve candidates, each with its live count. Everything downstream follows the visible list, so "select all" now grabs exactly the subset you are looking at — pick Improve candidates and one click reviews the whole batch. It stacks with the caption tag filter, it is remembered between visits, and whenever a filter is on, a banner above the grid says "showing 254 of 508" so a narrowed view can never be mistaken for lost images.',
    to: '/datasets?section=images',
  },
  {
    id: '2026-07-23-bulk-improve-is-a-server-job',
    date: '2026-07-23',
    title: '✨ Improve 250 images at once — and Stop really stops',
    blurb:
      'Selecting a big batch for "Improve via Klein" used to hit a wall: only the first 60 were accepted, the rest were refused one by one, and ⏹ Stop generation had no effect because the batch was a loop running in your browser tab — cancel the images in flight and the tab queued the next ones. The batch now runs on the server. It works through the whole selection a few at a time, waiting for a free slot instead of being refused, shows honest progress (how many queued out of how many), survives a page reload, and keeps going if you close the tab. And ⏹ Stop generation ends the batch itself, not just what happened to be generating at that instant.',
    to: '/datasets?section=images',
  },
  {
    id: '2026-07-23-continue-lane-picker-on-runs',
    date: '2026-07-23',
    title: '▶ Continue a cloud run on your own GPU, straight from the Runs page',
    blurb:
      'The Continue dialog already let you pick Local or Cloud — but only inside a dataset panel. Opened from the Runs page it silently relaunched a pod, even when your own GPU was free. It now offers the same choice: finish that epoch on this machine (its checkpoint was already mirrored here) or on a fresh pod. A lane you can’t use stays visible with the reason, and it is the RIGHT reason for that run — the cloud one counts the runs of that run’s own dataset, not the whole page. Refusals now speak up too: a busy GPU or a caption change is a toast, not a click that seems to do nothing.',
    to: '/cloud',
  },
  {
    id: '2026-07-22-hf-gate-checked-before-renting',
    date: '2026-07-22',
    title: '☁ A locked model no longer costs you a rented GPU',
    blurb:
      'Some base models are gated on Hugging Face: you must accept their licence once, and a repository can become gated overnight — three runs failed that way on a config that had worked the day before. The failure happened on the pod, after renting: you paid for a GPU that downloaded nothing. The launch now checks that access first and refuses before reserving anything, naming the model and the page to open. If Hugging Face is simply unreachable the launch still goes ahead — an outage must not ground a run that would have worked. And the run card no longer shows only "403 Client Error": these messages carry their explanation on the second line, which used to be visible only by hovering.',
    to: '/cloud',
  },
  {
    id: '2026-07-22-style-rename-actually-renames',
    date: '2026-07-22',
    title: '✎ Renaming a Style dataset really renames its files now',
    blurb:
      'Two bugs made it change the label and nothing else. A Style has no trigger field, so the settings dialog sends the stored token back unchanged — and that echo overwrote the token just derived from the new name, so nothing on disk ever moved. And the rename only touched the outer run folder, while training stamps the name at three levels: the run folder, a subfolder inside it, and every checkpoint file. Since importing a checkpoint deploys it under the file\'s own name, the LoRA kept arriving in ComfyUI under the old one. Both fixed. If you renamed a Style before this, rename it once more for its files to catch up.',
    to: '/datasets',
  },
  {
    id: '2026-07-22-version-label-names-the-commit',
    date: '2026-07-22',
    title: '🔢 The version shown is the version you are running',
    blurb:
      "The version number only moves when a release is cut, so anyone following the project on a git checkout was told they were on the last release — even sitting twenty commits past it. Being told “you're up to date” under an older number than the code you are running reads as a contradiction, and made it impossible to tell what was actually live. The update check already knew the branch and the commit; it now says them. Packaged installs are unchanged: there the release number really is the truth.",
    to: '/settings/maintenance',
  },
  {
    id: '2026-07-22-export-links-open-the-disclosure',
    date: '2026-07-22',
    title: '🎯 "Import to bank", "Portable backup" and "Publish to Hugging Face" show up again',
    blurb:
      'Those three ways out of a dataset live behind the "More ways out" fold, and clicking their link in the Import & export menu highlighted the link while the button stayed hidden inside the closed fold. Jumping to a panel now opens whatever fold it sits in, so the button you asked for is the one you land on.',
    to: '/datasets?section=export&panel=to-bank',
  },
  {
    id: '2026-07-22-improve-tuned-profile-and-loud-missing-lora',
    date: '2026-07-22',
    title: '✨ A better "Upscale & improve" out of the box — and it speaks up now',
    blurb:
      'The pass now ships with a high consistency strength by default. That setting resists redrawing the shot, which is a drawback when you are restaging an image and exactly the point when you are only adding detail — so an improve keeps your composition instead of quietly reinventing it. And a LoRA strength you raised is never silently ignored any more: if its weights file is missing, the pass says so (which is what fetches it) rather than running unchanged and leaving you guessing. At strength 0 nothing changes — a LoRA you did not ask for is still skipped quietly.',
    to: '/settings/engines',
  },
  {
    id: '2026-07-22-enhancement-lora-installed-automatically',
    date: '2026-07-22',
    title: '⬇ The improve detail LoRA installs itself now',
    blurb:
      'The "Upscale & improve" enhancement strength depends on a weights file the app never shipped or fetched — and when it is missing, that node is skipped entirely, so the slider moved nothing at all and said nothing about it. It is now downloaded with the other Klein assets by Setup ▸ Install everything, straight into the right ComfyUI folder. Fetched from its original public source (dx8152, Apache-2.0), never re-hosted.',
    to: '/setup',
  },
  {
    id: '2026-07-22-settings-links-where-you-act',
    date: '2026-07-22',
    title: '⚙️ "This is adjustable" — said where you are, not where the setting lives',
    blurb:
      'Several things were configurable without anything saying so: the Upscale & improve strength, which model writes your captions, the credentials a scraper source needs, the default LoRA family and the cloud GPU limits. Each of those places now carries a small link straight to the right Settings section, so you never have to go hunting for a page you did not know existed. The cloud banner also lands on the section that actually holds the vast.ai key instead of the Settings landing page.',
    to: '/settings/engines',
  },
  {
    id: '2026-07-22-checkpoints-refresh-when-a-run-ends',
    date: '2026-07-22',
    title: '🔄 Freshly trained LoRAs appear on their own',
    blurb:
      'When a run finished, its checkpoints were on disk but the list never re-read them — so the LoRA you had just trained stayed invisible until you changed the browse filter or reloaded the page. The panel now watches the run that concerns this dataset end, local or cloud, and re-reads what it produced; the lineage graph refreshes with it so the two views cannot disagree.',
    to: '/datasets?section=checkpoints',
  },
  {
    id: '2026-07-22-export-more-ways-out',
    date: '2026-07-22',
    title: '📦 Import & export: one clear action, the rest tucked away',
    blurb:
      'Export ZIP is what you reach for; Import to bank, Backup and Publish to Hugging Face are occasional. They now sit behind a single “More ways out” disclosure instead of four buttons competing for the same glance. Sidebar links still jump straight to them — the panel opens itself.',
    to: '/datasets?section=export',
  },
  {
    id: '2026-07-22-update-survives-history-rewrite',
    date: '2026-07-22',
    title: '🔄 Updating no longer breaks if the project history is rewritten',
    blurb:
      'In-app updates used to depend on every commit keeping its identity forever. If the project history was ever rewritten, every commit got a new id, no fast-forward was possible, and “Update & restart” failed for good — on a checkout that was otherwise perfectly healthy. The updater now recognises that case and resyncs, but only after proving nothing would be lost: it refuses if you have uncommitted changes to tracked files, or local commits of your own. Untracked files are never touched. The “commits behind” count is measured by content too, so a rewrite no longer reads as hundreds of pending commits when you are already up to date.',
    to: '/setup',
  },
  {
    id: '2026-07-22-improve-strength-settings',
    date: '2026-07-22',
    title: '🔧 "Upscale & improve" is now adjustable, not a fixed profile',
    blurb:
      'Its instruction was editable, but everything deciding what the pass produces was hardcoded — the output size at 2 MP whatever your source was worth, and both LoRA strengths at 0, which meant the enhancement LoRA built into the workflow never applied at all. Settings ▸ Image engines now exposes the output size, the enhancement LoRA, the consistency LoRA (it anchors composition, not identity) and the step count. All four start at exactly the values the action used before, so leaving them alone changes nothing. One caveat worth knowing: the enhancement LoRA reads a file that ships with neither the app nor the Klein install, and when it is missing its node is skipped entirely — so that one slider does nothing until you have it.',
    to: '/settings/engines',
  },
  {
    id: '2026-07-22-import-dataset-to-bank',
    date: '2026-07-22',
    title: '↑ Import to bank — send a dataset back the other way',
    blurb:
      "The bank could feed datasets, but nothing went the other way. Import & export now has ↑ Import to bank: the dataset's kept images are copied into a brand-new bank under a name you choose, so you can re-triage them with the bank tools — duplicate detection (perceptual and semantic), framing, quality and face scores — and promote a cleaner selection back out. They are COPIED, not shared, so nothing you do in the bank can disturb the dataset. Deleting such a bank takes its copy to Trash with it, so it never lingers on disk.",
    to: '/datasets?section=export',
  },
  {
    id: '2026-07-22-continue-says-why-it-is-off',
    date: '2026-07-22',
    title: '▶ "Continue training" now tells you why it is greyed out',
    blurb:
      'When the button was disabled, the reason was only in its hover tooltip — so it read as a button that simply does nothing. The reason is now written in the panel itself, in the same amber line the epoch tools use: either the checkpoints come from a different LoRA family, base or variant than the one selected in Training, or a training is already running on this machine and no cloud lane is available.',
    to: '/datasets?section=checkpoints',
  },
  {
    id: '2026-07-22-trigger-rename-follows-on-disk',
    date: '2026-07-22',
    title: '✎ Renaming the trigger word now renames the LoRAs it already made',
    blurb:
      "The trigger word is what names everything a dataset produces — the deployed LoRA, the training run folder, the export. Changing it used to leave all of that behind under the old name, orphaned from the dataset that made it. Now the files follow: LoRAs, run folder, export and job config are renamed together, and the Test Studio history and cloud runs keep pointing at them. If the new name is already taken on disk nothing is moved at all (never half), and the edit is refused while a run is live, since that folder is what training resumes from. Style datasets have no visible trigger — they are always-on — so there it is the dataset NAME that renames them.",
    to: '/datasets',
  },
  {
    id: '2026-07-22-install-everything-covers-scraper',
    date: '2026-07-22',
    title: '⬇ "Install everything" now repairs the scraper too',
    blurb:
      "The scraper packages were the one component Install everything never touched: it reported everything was already in place while a source kept failing on a missing package. They are now part of the plan, and the check looks at every package the scraper imports — so a package added by an update (instaloader, for Instagram) is picked up instead of staying invisible until you found the per-tile Reinstall button.",
    to: '/setup',
  },
  {
    id: '2026-07-22-continue-choose-local-or-cloud',
    date: '2026-07-22',
    title: '▶ Continue: choose where it runs — this GPU or a rented one',
    blurb:
      "A checkpoint is just a file, so where it was trained no longer decides where it can be finished. The Continue dialog now has a Local / Cloud switch: send a run trained on your machine to a rented GPU (the checkpoint is uploaded and training picks up from it), or finish a cloud epoch here. A lane you can't use says why instead of vanishing — and a local training in progress no longer blocks the cloud one.",
    to: '/datasets?section=checkpoints',
  },
  {
    id: '2026-07-22-final-checkpoint-previewable',
    date: '2026-07-22',
    title: '☑ The final checkpoint can be previewed too',
    blurb:
      'Importing the last save of a run left its pill in the lineage graph without a tick-box, so the one checkpoint you most want to look at was the only one you could not preview. The final save is deployed without a step number in its name; it is now matched back to its own run, and ticking it generates a preview like any other epoch.',
    to: '/datasets?section=checkpoints',
  },
  {
    id: '2026-07-22-continue-from-any-graph-checkpoint',
    date: '2026-07-22',
    title: '▶ Continue from any checkpoint straight from the graph — in your dataset too',
    blurb:
      "Clicking a checkpoint in the lineage graph used to offer “Continue from here” only on the Runs page. In a dataset's Checkpoints & LoRAs graph the same click now opens the Continue dialog already set on THAT step — including for runs trained on this machine — so resuming from the epoch that held up best is one click, not a dropdown hunt.",
    to: '/datasets?section=checkpoints',
  },
  {
    id: '2026-07-22-lineage-preview-checkbox-visible',
    date: '2026-07-22',
    title: '☑ The preview tick-box on checkpoints is finally visible',
    blurb:
      "In the lineage graph, the little corner box that picks a checkpoint for an inline preview was a 14px near-invisible square — easy to miss and fiddly to tap on a phone. It's bigger now with a clear outline, and the hint spells out that only an imported (📦 deployed) checkpoint can be ticked.",
    to: '/datasets?section=checkpoints',
  },
  {
    id: '2026-07-22-aitoolkit-readiness-honest',
    date: '2026-07-22',
    title: '🎓 Clearer guidance when ai-toolkit isn\'t ready to train',
    blurb:
      "If training can't use ai-toolkit, the hint now points you at the real fix — set its venv Python (venv/Scripts/python.exe) in Settings › Local tools — instead of a setup script that doesn't exist. And the diagnostic no longer reports ai-toolkit as ready when its interpreter isn't actually a usable file, so \"ai-toolkit=yes\" and the training gate finally agree. Thanks to sylvie for the report.",
    to: '/settings/local-tools',
  },
  {
    id: '2026-07-21-instagram-scrape-and-english-messages',
    date: '2026-07-21',
    title: '📸 Instagram scraping is back — and every scraper speaks English',
    blurb:
      "Instagram scraping works again: the missing 'instaloader' dependency now ships with the scrape extras (Setup › Install everything). Every scraper error message — Instagram, Civitai, Pexels, Reddit, RedGifs, Picazor, Erome and more — now reads in clear English, and the \"missing dependency\" ones tell you exactly which extra to install.",
  },
  {
    id: '2026-07-21-load-default-prompt-to-edit',
    date: '2026-07-21',
    title: '✎ Tweak a built-in prompt instead of retyping it',
    blurb:
      "The editable identity & Klein prompts in Settings › Image engines now put a \"Load default to edit\" button right under each field: one tap drops the shipped default into the box so you can adjust a word or two, instead of typing a whole new prompt from scratch.",
    to: '/settings/engines',
  },
  {
    id: '2026-07-21-zimage-style-preset-all-variants',
    date: '2026-07-21',
    title: '🎨 The Z-Image style preset now works on Turbo too',
    blurb:
      "The built-in \"Z-Image · Style\" preset used to be scoped to the Base variant only, so a Z-Image Turbo style dataset saw no style preset at all. Its recipe (rank 32/32, weighted timesteps) is the Z-Image architecture default, so it now applies to every Z-Image variant — Turbo, Base and De-Turbo.",
  },
  {
    id: '2026-07-21-studio-resolution-multiplier',
    date: '2026-07-21',
    title: '🔍 Push Test Studio renders larger',
    blurb:
      "A new resolution multiplier slider (1.0×–1.9×) sits under the Fast/Standard/HQ/Max presets in the Test Studio: keep your preset's aspect and just scale it up, both sides at once (e.g. a 1008px Standard square becomes ~1915px at 1.9×). Each preset chip and the live readout show the final W×H as you slide. Default 1.0× leaves everything exactly as before. Past ~1.5× it warns that Krea/Z-Image can soften, duplicate or run out of GPU memory — it warns, never blocks.",
    to: '/studio',
  },
  {
    id: '2026-07-21-tidier-top-bar',
    date: '2026-07-21',
    title: '🧭 A tidier top bar',
    blurb:
      "Your workspaces (Datasets, Bank, Runs, Test Studio) now sit together on the left, while Guide and Help tuck into a ? menu and Setup and Settings into a ⚙ menu on the right. Less clutter, same one-click reach.",
  },
  {
    id: '2026-07-21-toggle-thumb-alignment',
    date: '2026-07-21',
    title: '🎚️ Toggle switches sit flush again',
    blurb:
      "The little sliding knob on on/off switches now rests with an even 1-2px gap on both ends instead of floating short of the right edge when on. Purely cosmetic, but it looks right now. Thanks to bbsorry for the pixel-perfect report.",
  },
  {
    id: '2026-07-21-zimage-convert-cross-drive',
    date: '2026-07-21',
    title: '🗜️ Convert a custom Z-Image base even when your models live on another drive',
    blurb:
      "Clicking \"Convert the base\" no longer fails with a red \"Paths don't have the same drive\" toast when your ComfyUI models folder is a junction to a second drive (a common setup — big weights rarely fit the system disk). The conversion now follows the junction across drives while still refusing any base path that tries to escape your models folder.",
  },
  {
    id: '2026-07-21-cloud-unreachable-grace',
    date: '2026-07-21',
    title: '☁️ Fewer cloud runs lost to a passing network blip',
    blurb:
      "A rented pod that briefly drops off the network (a vast.ai proxy hiccup mid-training) is no longer given up so quickly: the grace before a run is declared \"pod unreachable\" is now measured as real consecutive silence, not polluted by slow log/checkpoint mirroring — and it defaults to a more forgiving 6 minutes. Too twitchy or too patient for your hosts? Tune it under Settings ▸ Training ▸ Unreachable grace. Also: a transient rental refusal at pod creation now retries on a fresh offer instead of failing the launch outright.",
    to: '/settings/training',
  },
  {
    id: '2026-07-20-bank-guided-zones',
    date: '2026-07-20',
    title: '🧭 The Bank top is now a guided path, not a wall of buttons',
    blurb:
      "The 🗃️ Bank's controls are now grouped into four ordered, labeled zones — ① Analyze, ② Triage, ③ Curate, ④ Promote — that follow the natural workflow, and a subtle amber marker points at the recommended next step based on where your bank is (nothing scanned → Analyse; scored with images kept → Promote). Nothing is hidden — every control stays where you can reach it — it just finally reads as a path instead of a pile.",
    to: '/bank',
  },
  {
    id: '2026-07-20-bank-explicit-caption',
    date: '2026-07-20',
    title: '🏷️ Caption your bank crude — explicit lane, right at triage',
    blurb:
      "The 🏷️ Caption pass in the Bank now has a vocabulary picker, the same one the datasets use: Explicit, Clinical or Safe. Pick Explicit (paired with an uncensored/abliterated Ollama vision model) and captions name nude and sexual content plainly instead of tip-toeing around it — so you capture what's really there the moment you triage. Bonus: richer, more explicit captions also give the 🔍 Bank search far more to match on. Leave it on default and nothing changes.",
    to: '/bank',
  },
  {
    id: '2026-07-20-graph-hub',
    date: '2026-07-20',
    title: '◉ The lineage graph is now home for your checkpoints',
    blurb:
      "Open Checkpoints & LoRAs and you now land on the ◉ Graph — your dataset's runs and every checkpoint they made, at a glance (the flat ☰ List is one click away). Deploy any checkpoint straight from its pill with 📦 Import → loras/…, generate a preview per checkpoint, then click a preview thumbnail to see it LARGE and compare epochs like in ComfyUI. See it, deploy it, judge it — all without leaving the graph.",
    to: '/datasets?section=checkpoints',
  },
  {
    id: '2026-07-20-graph-big-previews',
    date: '2026-07-20',
    title: '🔍 Big-preview mode — compare checkpoints like a ComfyUI grid',
    blurb:
      "The lineage graph gets a 🔍 Big previews toggle: the generated thumbnails on each checkpoint blow up into large tiles laid out like a ComfyUI grid, so you can eyeball several epochs side by side and pick the sweet spot without clicking into each one. It's remembered between visits; leave it off for the compact pill view. Click any tile to still open it full-screen.",
    to: '/datasets?section=checkpoints',
  },
  {
    id: '2026-07-20-lineage-inline-generation',
    date: '2026-07-20',
    title: '🎨 Generate a preview per checkpoint, right in the lineage graph',
    blurb:
      "Turn the ◉ Graph into an experiment lab: tick the checkpoints you want (across any runs), type ONE shared prompt and seed, and hit Generate. Each selected checkpoint renders a strength-1.0 image under the exact same conditions — reusing the Test Studio engine — so you can see at a glance how a LoRA evolves epoch by epoch and pick the sweet spot before it overcooks. The thumbnail lands on its checkpoint pill (◌ while it renders, ⚠ if it fails). It shares the GPU politely: a checkpoint that isn't deployed yet can't be picked (with a clear hint), and while a training is running the previews wait rather than fight it.",
    to: '/cloud',
  },
  {
    id: '2026-07-20-identity-prompts-show-defaults',
    date: '2026-07-20',
    title: '👁️ See the built-in identity prompts, not just an empty box',
    blurb:
      "The editable identity & Klein prompts used to show a blank field with a generic \"leave blank\" note, so you could never see the actual default text that was being applied. Each field now displays its real built-in default, and a \"Load default to edit\" button drops that exact text into the box so you can tweak it instead of writing one from scratch. Leaving a field blank still uses the shipped default, byte-for-byte. Completes @bbsorry (雨田壹)'s request to see and adjust these prompts.",
    to: '/settings/engines',
  },
  {
    id: '2026-07-20-runs-history-load-more',
    date: '2026-07-20',
    title: '📜 See more than the last 15 runs — “Load older runs”',
    blurb:
      "The Runs page only kept the 15 most recent runs in its history, so anything older dropped off the list. The live view still refreshes lightly on those recent runs, but a new “Load older runs” button now pulls the rest on demand (up to 100), so a long training history stays reachable without slowing the page down.",
    to: '/cloud',
  },
  {
    id: '2026-07-20-lineage-diff',
    date: '2026-07-20',
    title: '⚖️ Compare two runs and see exactly what changed',
    blurb:
      "Shift-click two runs in the ◉ Graph and a side-by-side panel shows their settings, with the ones that DIFFER highlighted (and the identical ones dimmed and foldable). Stop eyeballing two panels to answer \"what did I change between v2 and v3\" — the diff spells it out: rank, learning rate, optimizer, steps and the rest. Older runs that never recorded their settings say so honestly instead of faking a comparison.",
    to: '/cloud',
  },
  {
    id: '2026-07-20-delete-gone-runs',
    date: '2026-07-20',
    title: '🗑 Tidy up the lineage graph — remove runs whose checkpoints are gone',
    blurb:
      "Runs whose checkpoints are no longer on disk used to pile up in the ◉ Graph as \"gone\" cards you couldn't clear. Click a gone run and the inspector now offers “Remove this run” — it clears the leftover entry and its notes (no files are touched, they're already gone). A run that still has checkpoints on disk is protected: it shows no remove button, and the server refuses to delete it. A removed run that others continued from doesn't break the tree — its children stay, re-rooted.",
    to: '/cloud',
  },
  {
    id: '2026-07-20-bank-promote-per-target',
    date: '2026-07-20',
    title: '⬆ Promote the same Bank picks into more than one dataset',
    blurb:
      "Promoting kept images into a dataset used to lock them out of every OTHER dataset — a second promote showed \"nothing to promote\" even though the dialog had just offered to copy them. Now “promote all kept” is per-target: images already sitting in another dataset stay promotable to a new one, and the dialog’s count reflects exactly what will be copied into the dataset you picked. Near-duplicates already in the target are still skipped on import.",
    to: '/bank',
  },
  {
    id: '2026-07-20-resolve-rejects-losers',
    date: '2026-07-20',
    title: '✂ “Keep best” on same-shot groups now actually rejects the extras',
    blurb:
      "Resolving a duplicate or same-shot group with “Keep best”, “Keep first” or “Resolve ALL” used to report “0 duplicate(s) rejected” whenever the group’s images were already kept — so the near-identical shots you meant to thin all stayed in. Now an explicit resolve keeps the chosen one and rejects the rest, kept or not. The automatic pipeline pass is unchanged: a mass auto-reject still never un-keeps an image you picked by hand.",
    to: '/bank',
  },
  {
    id: '2026-07-20-editable-identity-prompts',
    date: '2026-07-20',
    title: '🪪 Edit the hidden identity prompts that keep a face consistent',
    blurb:
      "The prompts that lock a subject's facial identity across every generated variation used to be baked in and invisible. They're now editable in Settings → Image engines: the API-engine identity locks, the Klein restage block, and the “Klein upscale & improve” instruction — each with a one-line explanation and a Restore default. You can also switch the improve prompt off entirely for a pure upscale. Leave everything blank and generation is exactly as before. Feature request by @bbsorry (雨田壹).",
    to: '/settings/engines',
  },
  {
    id: '2026-07-20-lineage-inspect-notes',
    date: '2026-07-20',
    title: '🔬 Inspect any run’s settings and take notes, right on the graph',
    blurb:
      "Click a run in the ◉ Graph and a panel now shows the exact settings it trained with — rank, alpha, learning rate, optimizer, timestep, base model, steps. Jot a note on any run or checkpoint (\"step 1500 = best face\", \"3000 overcooks\") and a dot marks the ones you've annotated, so a lineage becomes a lab notebook instead of a list. Older runs that never recorded their settings simply say so.",
    to: '/cloud',
  },
  {
    id: '2026-07-20-bank-show-selected-view',
    date: '2026-07-20',
    title: '🔎 See your curated picks together — 🎯 Similar & 🎨 Diverse now show their results',
    blurb:
      "Picking the images that look like one reference, or the most varied of a big dump, used to just tick boxes — on a 20 000-image bank those picks were scattered across pages you'd never scroll to, so it felt like nothing happened. Now 🎯 Similar to selected and 🎨 Pick diverse drop the grid straight into a “selected” view that shows ONLY your picks — and 🎯 Similar orders them closest-first, reference at the top. A new “🔎 Show selected” toggle flips any selection into that view (and “↩ Show all” takes you back). Keep, Reject and Promote still act on the selection exactly as before — this is just a way to look at it.",
    to: '/bank',
  },
  {
    id: '2026-07-20-bank-framing-filter',
    date: '2026-07-20',
    title: '📐 Sort a Bank by shot type — face, bust, body, back',
    blurb:
      "The 🗃️ Bank can now classify every image by framing — face close-up, bust, full body or back view — with the same detector the datasets use. New 📐 Framing filter chips slice the grid one shot type at a time (and compose with every other filter and search), so balancing a character set's angles is a couple of clicks. Run 📐 Classify framing, or just add it to your 🚀 Launch all overnight run.",
    to: '/bank',
  },
  {
    id: '2026-07-20-bank-coverage-advice',
    date: '2026-07-20',
    title: '📊 Coverage advice — what your kept set is missing',
    blurb:
      "A new 📊 Coverage advice panel in the 🗃️ Bank reads what you've kept and tells you, in plain sentences, what leans and what's thin for a good LoRA — '70% face shots, add body/back', 'person #1 is 60% of the set — one subject or a mix?', 'only 8 kept, most families want 20+'. It's advice only (nothing is kept or rejected) and pure maths on data the passes already computed, so it costs no GPU. Idea by @antonp.",
    to: '/bank',
  },
  {
    id: '2026-07-20-bank-curation-diverse-similar',
    date: '2026-07-20',
    title: '🎨 Curate a big Bank down to the images that actually train well',
    blurb:
      "Curation is 90% of a good LoRA, so the 🗃️ Bank gets two selectors that turn a huge dump into the right subset — both reuse the ✨ Score embeddings, so they cost no extra GPU time. 🎨 Pick diverse selects the N images that best COVER the variety (angles, outfits, scenes) instead of N near-identical shots — the antidote to '4000 photos of the same pose'. 🎯 Similar to selected ranks the bank by how much it looks like ONE image you pick and selects the closest, to pull one person or look out of a mixed export. Both compose with your filters and search ('60 most diverse of this subfolder'), and land as a normal selection you review before ✓ Keep or ⬆ Promote — nothing is auto-kept or deleted. Run ✨ Score once to unlock them.",
    to: '/bank',
  },
  {
    id: '2026-07-20-bank-workspace-tidy',
    date: '2026-07-20',
    title: '🗃️ A calmer, clearer Bank workspace',
    blurb:
      "The 🗃️ Bank toolbar is reorganized around what you actually do: Launch all and Promote stand out as the two outcomes, the individual analysis passes (Scan, Score, Watermarks, Person, Crops, Caption) sit together below them, and the flag filters are now grouped by Status, Quality, Score, Groups and 📐 Resolution with a live \"N shown of total\" count. Same tools, nothing removed — just far easier to read on a wide screen or a phone.",
  },
  {
    id: '2026-07-20-bank-delete-rejected',
    date: '2026-07-20',
    title: '🗑 Delete rejected images from your disk',
    blurb:
      "Done triaging a 🗃️ Bank? A new 'Delete rejected from disk' button next to Promote clears every image you marked ✕ rejected straight off your drive — the one Bank action that touches your source files. It asks you to type DELETE first, and sends the files to your OS trash when possible (a hard delete otherwise). Heads up: this is irreversible — the app's own trash can't bring them back. Kept and undecided images are never touched.",
    to: '/bank',
  },
  {
    id: '2026-07-19-bank-sort-resolution',
    date: '2026-07-19',
    title: '📐 Sort AND filter your Bank by resolution',
    blurb:
      "The 🗃️ Bank grid gains a Sort control next to the tiles: order every image by resolution, biggest or smallest first. It ranks by megapixels (width×height), so a crisp 900×900 outranks a stretched 1200×300 — the right way to skim a mixed dump for the sharpest, most trainable shots. New: a 📐 Resolution row of tier chips (< 0.25 MP · 0.25–1 · 1–2 · 2–4 · > 4 MP), each showing its count — click a tier to see just those images, then 'Select all in filter' + reject to clear out the tiny thumbnails of a 20k-image Telegram dump in seconds. Both stack on top of every filter and search you already have. Images not scanned yet sink to the end and count toward no tier.",
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
  {
    id: '2026-08-04-text-encoder-tolerates-a-chatty-worker',
    date: '2026-08-04',
    title: '🎨 Medium and 🔎 text search stop failing over a line of chatter',
    blurb:
      'On some installs 🎨 Medium and 🔎 text search died with “the text encoder produced no result — check the ✨ Score interpreter”, on machines whose ✨ Score interpreter was fine — it had just produced the embeddings both features read. The cause was one line: the first thing the text encoder printed had to be its answer, so a first-run banner, a weights download or any greeting from the ML environment ahead of it was read as a failure. Both features now step over anything that is not an answer, the way the ✨ Score pass already did. And when the encoder really does fail, the message quotes what it actually printed — including the error output, which used to be discarded — with home-folder paths stripped, so it is safe to paste into a help thread and it no longer sends you to check a component it never looked at. Reported by shinybtw (Discord), with the failing traceback.',
    to: '/bank',
  },
  // Appended rather than prepended ON PURPOSE: several waves are editing the
  // head of this array at the same time and ordering is by date desc then id
  // desc, so position here changes nothing and a tail entry is a conflict less.
  {
    id: '2026-08-04-seedvr2-settings-say-which-lane-your-target-takes',
    date: '2026-08-04',
    title: 'SeedVR2 settings tell you whether your target will be tiled',
    blurb:
      'Tiling starts strictly above the crossover, and the crossover is 1.5× your tile size — so it lands exactly on the round numbers people type. Ask for 1536 px with the default 1024 px tile (or 768 px with a 512 px tile) and the upscale ran whole, with nothing anywhere saying why: no tiles, no warning, no line in the panel. The SeedVR2 card now names the lane your configured target will actually take, and when it sits on the crossover it says so and gives you the three ways to change it.',
    to: '/settings/engines',
  },
  {
    id: '2026-08-04-fp8-delivery-names-the-running-job-before-the-disk',
    date: '2026-08-04',
    title: 'A busy fp8 conversion says so, instead of blaming your disk',
    blurb:
      'Starting a second fp8 conversion while one was already running could answer “not enough disk space” — true, but not the reason it refused, and it sent you deleting gigabytes for a job the disk was never blocking. The refusal now names what is actually in the way: something is already being prepared, wait for it or stop it. It also answers instantly, because it no longer reads the Hugging Face repository just to say no.',
    to: '/datasets?section=checkpoints',
  },
  {
    id: '2026-08-04-setup-tells-a-broken-optional-file-from-a-dead-engine',
    date: '2026-08-04',
    title: 'Setup stops calling a damaged optional file a broken engine',
    blurb:
      'A corrupted Klein consistency LoRA lit up Setup in red with “⚠ On disk, unreadable”, exactly like a dead model file — while generation carried on working perfectly, because that LoRA has never been required. The alarm said the engine was down when the engine was fine. It now says what is true: an unreadable file that nothing waits on is flagged in amber as optional, with a line telling you Klein still generates without it and a button to download it again; only a file the engine really needs keeps the red badge. The same file also stopped hiding on the other screen — the download buttons used to print “✓ Installed” over it, because it was not on the required list, so the one place offering to fix it was the one place claiming nothing was wrong.',
    to: '/setup?step=install',
  },
  {
    id: '2026-08-04-captions-say-which-engine-wrote-them',
    date: '2026-08-04',
    title: 'See which engine actually wrote your captions',
    blurb:
      'The captioning engine is set to Auto by default, and Auto is a chain, not a choice: JoyCaption writes what it can, the Ollama vision model writes the rest, and on a Concept dataset Ollama rewrites JoyCaption’s drafts. Those engines do not write alike — so captions could come back in two different voices, or in a different voice than last week, with nothing anywhere to explain it. Every pass now reports who wrote what, in the toast and on a line under the caption buttons: “Written by JoyCaption.”, “Drafted by JoyCaption, rewritten by the Ollama vision model.”, or “8 by JoyCaption · 4 by Ollama” when the batch was shared. If you want one consistent voice, ⚙️ Options lets you name a single engine — and now you can tell whether it obeyed.',
    to: '/datasets?section=captions&panel=generate',
  },
  // Appended at the TAIL on purpose. Ordering is by date desc then id desc
  // (sortedEntries), never by array position, so a new entry does not have to be
  // prepended — and not prepending is what keeps two branches editing this file
  // from merging into one mangled object.
  {
    id: '2026-08-04-bank-caption-engine-and-scope',
    date: '2026-08-04',
    title: 'Choose who writes your bank captions, and which pile gets them',
    blurb:
      'The 🏷️ Caption pass used to take whatever engine and vision model your Settings held, and always ran over everything you had not rejected. It now has its own row: pick the engine and the Ollama vision model for THIS run without touching your Settings, and aim the pass at the kept images only, the undecided only, or both. Which model writes a caption is not a matter of taste — one that describes things in evasive terms produces captions that are about something slightly other than your images, and a LoRA trained on them learns to look away too. The button also stops saying “Caption all” and starts saying how many images it is really about to write, because already-captioned ones are skipped and the count you could see was never the count it acted on. Rejected images stay out of reach, whatever you pick.',
    // No `to`: the control lives inside a bank workspace, which has no deep link.
  },
  {
    id: '2026-08-04-bank-recaption-button',
    date: '2026-08-04',
    title: 'Redo a bank’s captions with a better model — and see what that costs first',
    blurb:
      'Once every image in a bank had a caption, 🏷️ Caption reached zero and went grey — taking the engine and model selects beside it out of reach, on exactly the bank whose captions you wanted to redo. A 🔄 Re-caption button now sits at the end of the Caption options row and runs the same pass over the pile you chose, with the engine, model, register and length you picked. It overwrites, so it tells you the numbers before you click: the button quotes how many images it will rewrite, an amber line quotes how many of those already carry a caption, and the confirmation repeats both. It also says the part we cannot fix: this app stores one caption per image and records nothing about who wrote it, so a caption you corrected by hand looks exactly like a generated one and is overwritten too — and no undo covers captions. With images selected the button stays inert and says why, because a selection can span pages that were never loaded and the count would be a guess.',
  },
  {
    id: '2026-08-07-framing-stops-locking-itself-out-of-the-model',
    date: '2026-08-07',
    title: '📐 Framing classifies your whole bank instead of a handful of images',
    blurb:
      'On a bank of any size, 📐 Framing could classify a few images and then quietly stop classifying anything at all — the bar kept moving, the log filled with “vision GPU window renewal failed / database is locked”, and stopping the pass reported four images done out of the twelve it had walked. The pass was starving itself: it saved its results in batches of 25 and held the database write lock between them, across model calls that take seconds each, while the check that lets the app talk to the vision model needs that same database every single call. Failing that check is fail-safe by design, so every remaining image was refused rather than mislabelled — correct, and completely invisible. It now saves each image as it goes (measured at 0.17 ms per save) and never holds the lock across a model call, so a pass classifies what you gave it. The 🔎 Watermark scan and the watermark repaint level got the same treatment before they could hit it. And when something does go wrong, every pass now ends the same way whether it finished or you stopped it: what it classified, what it could not read, what was deleted under it, what changed while it was being analysed, and what the model was never shown. Reported by _mr.arrow_ (Discord).',
    // No `to`: the passes live inside a bank workspace, which has no deep link.
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
  '/datasets', '/bank', '/video-bank', '/studio', '/cloud', '/canvas', '/guide',
  '/help', '/setup',
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
