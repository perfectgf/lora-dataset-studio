import { registeredDescriptors, routes as pluginRoutes } from './plugins/registry.js';
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
import { WORKSPACE_SECTIONS, sectionPanels } from './components/dataset/workspaceSections.js';
import { SETUP_DEEP_LINK_STEPS } from './hooks/useSetupSteps.js';

// Newest first. Prepend new waves at the top.
export const WHATS_NEW = [
  {
    id: '2026-09-26-windows-dataset-forge',
    date: '2026-09-26',
    title: 'Dataset Forge is available to Windows ZIP installations',
    blurb: 'Update LDS, then install the free Dataset Forge plugin from Plugins to generate dataset variations locally with Qwen-Image 2.1. Prepare its models in the plugin settings and choose Qwen-Image 2.1 in Generate variations. This release brings the required plugin catalog and engine support to the Windows ZIP update path. Thanks to vitokorn (GitHub #71) for reporting the release gap.',
    to: '/plugins',
  },
  {
    id: '2026-09-26-cloud-video-release-compatibility',
    date: '2026-09-26',
    title: 'Start cloud video training with installed plugins',
    blurb: 'Cloud video launches work with the default rank on existing Cloud training installations. Update Cloud training from Plugins to use another rank and preserve it when retrying or continuing a run. Released rental records can be removed from history while pending cleanup remains protected.',
    to: '/plugins',
  },
  {
    id: '2026-09-23-zzzzzzzz-plugin-engine-settings-save',
    date: '2026-09-23',
    title: 'Save your enabled image engines from plugin settings',
    blurb: 'Engine checkboxes now show your saved selection and save alongside API keys and model choices without an ownership error. Changing a plugin\'s engines preserves your other engine selections.',
  },
  {
    id: '2026-09-23-zzzzzzz-studio-node-repair',
    date: '2026-09-23',
    title: 'Resolve missing ComfyUI nodes from your test setup',
    blurb: 'Studio and model settings now offer installation-specific repair steps, direct ComfyUI links and a fresh node check. Missing nodes stay visible even when all model files are present, and your prompt and checkpoint choices survive the repair.',
    to: '/studio',
  },
  {
    id: '2026-09-23-video-reference-text-writer',
    date: '2026-09-23',
    title: 'Auto and Enrich reach your chosen motion writer',
    blurb: 'Reference prompts can now continue from JoyCaption to Ollama without a server error. Text generation also keeps the selected local provider when settings change during a request.',
    to: '/studio',
  },
  {
    id: '2026-09-23-zzzzzz-studio-model-downloads',
    date: '2026-09-23',
    title: 'Download missing test models without leaving Studio',
    blurb: 'Missing Qwen-Image 2.1, FLUX.1 or Anima files now come with download buttons, file sizes and progress. LDS installs them in the correct folders and refreshes the model list while keeping your prompt and checkpoints.',
    to: '/studio',
  },
  {
    id: '2026-09-23-generation-preset-library',
    date: '2026-09-23',
    title: 'Keep every named LoRA preset',
    blurb: 'Klein and Krea settings let you save and duplicate more than twelve named presets. Every preset stays available in generation selectors instead of being silently dropped.',
    to: '/settings/engines',
  },
  {
    id: '2026-09-23-zzzzz-test-every-image-family',
    date: '2026-09-23',
    title: 'Test and compare every image LoRA family',
    blurb: 'Generate test images with FLUX.1, Anima and Qwen-Image 2.1 alongside Z-Image, SDXL, Krea 2 and FLUX.2 Klein. Compare checkpoints or blend compatible LoRAs with family-specific models, defaults and negative prompts where supported.',
    to: '/studio',
  },
  {
    id: '2026-09-23-zzzzz-joycaption-video-enrich',
    date: '2026-09-23',
    title: 'Use Auto and Enrich with JoyCaption again',
    blurb: 'Using JoyCaption to read a reference image no longer makes Auto or Enrich fail with a server error. If the image reader fails or times out, the Video plugin can show the reason.',
  },
  {
    id: '2026-09-23-zzzz-mobile-plugin-navigation',
    date: '2026-09-23',
    title: 'Reach every plugin from your phone',
    blurb: 'The mobile navigation menu now scrolls within the screen, so Plugins, Settings and other entries stay reachable even with many plugins enabled.',
    to: '/plugins',
  },
  {
    id: '2026-09-23-zzz-qwen-image-21-training',
    date: '2026-09-23',
    title: 'Train Qwen-Image 2.1 LoRAs',
    blurb: 'Choose Qwen-Image 2.1 for image-and-caption datasets, with its own training recipe and checkpoint folder. Train locally with an up-to-date AI Toolkit, or use Cloud Training with its dedicated pod image. Image-edit pairs and RGBA datasets are not available for this family.',
    to: '/datasets',
  },
  {
    id: '2026-09-23-zz-plugin-update-all', date: '2026-09-23',
    title: 'Update all your plugins with one restart',
    blurb: 'Use Update all to review available updates together, including connected private catalogs. Confirm once to download, prepare and restart automatically when your launcher supports it. Disabled plugins stay disabled. A retained cloud pod no longer blocks a restart that keeps its plugin enabled.',
    to: '/plugins?filter=updates',
  },
  {
    id: '2026-09-23-studio-resume-feedback',
    date: '2026-09-23',
    title: 'A clear answer when Resume cannot start',
    blurb: 'Canvas and Studio now show why a run could not resume and keep its saved settings ready to retry. Generation checks use a small ComfyUI status request and distinguish a slow server from an unreachable one.',
  },
  {
    id: '2026-09-22-zzzzzzz-plugin-restart-session',
    date: '2026-09-22',
    title: 'Apply plugin changes without a session-token interruption',
    blurb: 'Keeping another local app or LDS instance open no longer disrupts your session. Apply and restart automatically recovers the correct session token without asking you to refresh the page.',
    to: '/plugins',
  },
  {
    id: '2026-09-22-zzzzzzz-fast-tool-detection',
    date: '2026-09-22',
    title: 'Find ComfyUI and ai-toolkit sooner after a restart',
    blurb: 'LDS checks your configured ComfyUI server and ai-toolkit folder immediately, while optional tools finish their checks in the background. Training and Studio become visible as soon as their tools are found, and checking ComfyUI no longer downloads its entire generation history.',
  },
  {
    id: '2026-09-22-zzzzzzz-english-diagnostics',
    date: '2026-09-22',
    title: 'Read consistent English diagnostics throughout LDS',
    blurb: 'Remaining French messages are now in English, including scraping and training diagnostics and public plugin messages.',
  },
  {
    id: '2026-09-22-zzzzzz-private-lds-plugin-updates',
    date: '2026-09-22',
    title: 'Receive updates for your private LDS plugins',
    blurb: 'Trusted private catalogs can now update previously installed LDS plugins alongside the public catalog. Keep using your usual launcher.',
    to: '/plugins',
  },
  {
    id: '2026-09-22-zzzzzz-optional-usage-statistics',
    date: '2026-09-22',
    title: 'Choose whether to help improve LDS with usage statistics',
    blurb: 'Optional usage sharing helps prioritize features and reliability improvements. It stays off until you agree, excludes your images and text, and can be turned off immediately in Settings → Maintenance.',
    to: '/settings/maintenance',
  },
  {
    id: '2026-09-22-zzzzzz-checkpoint-graph-refresh',
    date: '2026-09-22',
    title: 'Find your saved LoRAs without waiting for training setup',
    blurb: 'Checkpoints load using the dataset’s saved family, base and variant while training tools are still being detected. Refresh checkpoints also reloads the visible graph, and an older response can no longer hide newer results.',
    to: '/datasets?section=checkpoints',
  },
  {
    id: '2026-09-22-zzzzz-plugin-restart-without-comfyui',
    date: '2026-09-22',
    title: 'Apply plugin changes even when ComfyUI is offline',
    blurb: 'ComfyUI connection and queue checks now show a warning while LDS restarts. Tasks still running in LDS remain protected.',
    to: '/plugins',
  },
  {
    id: '2026-09-22-private-plugin-updates',
    date: '2026-09-22',
    title: 'Update private plugins alongside your public plugins',
    blurb: 'Configured private catalogs can offer signed updates for their approved plugins while the public catalog remains available. Install from a ZIP is also available at the top of Plugins.',
    to: '/plugins',
  },
  {
    id: '2026-09-22-zzzz-plugin-update-button',
    date: '2026-09-22',
    title: 'Spot plugin updates immediately',
    blurb: 'An orange Update button now sits beside the available version at the top of each plugin card. Review the change before confirming installation.',
    to: '/plugins',
  },
  {
    id: '2026-09-22-zzz-plugin-columns',
    date: '2026-09-22',
    title: 'See more plugins at a glance',
    blurb: 'The plugin catalog uses up to four columns on wide screens and adapts to three, two or one as the window gets narrower.',
    to: '/plugins',
  },
  {
    id: '2026-09-22-zzz-local-llm-model-lists',
    date: '2026-09-22',
    title: 'Choose a detected Ollama or LM Studio model',
    blurb: 'Settings now lists the models seen by each local server instead of asking you to type a name. Refresh after downloading a model. Your existing choice is preserved if the server is offline, and LM Studio still offers automatic selection.',
    to: '/settings/local-tools?focus=ollama-vision-model',
  },
  {
    id: '2026-09-22-zz-unified-plugins',
    date: '2026-09-22',
    title: 'Discover and manage plugins in one place',
    blurb: 'Browse all plugins, filter installed tools or available updates, and open, configure or turn plugins on and off from the same cards. Plugins installed outside the store stay in the list, even when the catalog is offline.',
    to: '/plugins',
  },
  {
    id: '2026-09-22-zz-processing-network-timeouts',
    date: '2026-09-22',
    title: 'Give repairs and slow hardware more time',
    blurb: 'Settings → Local tools → Time limits now controls the five-minute Klein repair limit, Bank upscaling waits, and processing and network timeout multipliers. All defaults stay the same. Longer repair limits also reach the generation worker.',
    to: '/settings/local-tools?focus=klein-repair-timeout',
  },
  {
    id: '2026-09-22-z-plugin-manager-cards',
    date: '2026-09-22',
    title: 'Find your way around installed plugins',
    blurb: 'Each plugin has its own card with a clear name, status and grouped actions. Expand details when you need them. Store cards keep plugin names above their previews and retry screenshots when the catalog refreshes.',
    to: '/plugins?tab=installed',
  },
  {
    id: '2026-09-22-working-plugin-tool-installs',
    date: '2026-09-22',
    title: 'Prepare plugin tools without a dependency error',
    blurb: 'Fixed the “Constraints cannot have extras” error when preparing plugin tools. LDS keeps its required dependency versions while installing the selected tools. Retry the failed component from the plugin settings.',
  },
  {
    id: '2026-09-22-video-lora-collapse',
    date: '2026-09-22',
    title: 'Fold the video LoRA list whenever you need more room',
    blurb: 'Use Collapse to close the LoRA list without changing your selection. The selected LoRA and its strength stay visible; Change opens the list again.',
    to: '/studio?lane=video',
  },
  {
    id: '2026-09-22-long-local-generation',
    date: '2026-09-22',
    title: 'Queue local generations for an unattended run',
    blurb: 'Klein and Krea can now queue up to 1,000 unfinished images per dataset by default, with a configurable limit in Settings → Local tools → ComfyUI. The shot multiplier now goes up to 20. Images still run one at a time, and API runs keep their separate limit. Thanks to lucasofff for requesting longer overnight runs.',
    to: '/settings/local-tools',
  },
  {
    id: '2026-09-22-comfyui-generation-time-limit',
    date: '2026-09-22',
    title: 'Give slow ComfyUI generations more time',
    blurb: 'Settings → Local tools → ComfyUI now lets you increase the 15-minute generation time limit, or set it to 0 for unlimited waiting. Stop controls and worker health checks stay active. Thanks to unteins for reporting the interruption on slow machines.',
    to: '/settings/local-tools',
  },
  {
    id: '2026-09-22-install-selected-plugins',
    date: '2026-09-22',
    title: 'Install several plugins with one restart',
    blurb: 'Select the plugins you want in the Store, review their shared plan and confirm once. LDS verifies the whole selection before preparing it, then applies all changes in one restart. Thanks to _nofaceman for reporting the repeated restarts in Docker.',
    to: '/plugins',
  },
  {
    id: '2026-09-21-docker-v2-plugins',
    date: '2026-09-21',
    title: 'Install V2 plugins in Docker',
    blurb: 'Docker images now include the public plugin Store configuration. GPU installs can apply plugin changes by restarting only LDS, and update instructions follow your Docker setup. The Docker updater also follows the maintained V2 branch. Thanks to _nofaceman for reporting the missing catalog.',
  },
  {
    id: '2026-09-20-plugin-install-review-focus',
    date: '2026-09-20',
    title: 'Plugin installation review stays in view',
    blurb: 'Installing a plugin from lower in the catalog now brings its review panel into view and moves keyboard focus there, so the confirmation step is easy to find. Thanks to @strichinina for reporting and diagnosing this in #70.',
  },
  {
    id: '2026-09-19-vast-referral-links',
    date: '2026-09-19',
    title: 'Consistent Vast.ai referral links',
    blurb: 'Vast.ai links in setup, settings and documentation consistently use the project referral link. The affiliate disclosure remains visible beside account-creation instructions.',
  },
  {
    id: '2026-09-19-plugin-installation-unlock',
    date: '2026-09-19',
    title: 'A clear way to unlock plugin installation',
    blurb: 'The plugin store now explains restricted installation and guides you through local access or an admin token for another computer. Locked install buttons lead to the unlock form, and an incorrect token gets a clear explanation. Thanks to lucasofff for reporting the confusing grey buttons.',
    to: '/plugins',
  },
  {
    id: '2026-09-19-installed-plugin-compatibility',
    date: '2026-09-19',
    title: 'Keep your installed plugins when updating V2',
    blurb: 'The public V2 now supports the shared interfaces used by newer installed plugins, including Video reference frames. Existing plugin installations and their saved data stay in place.',
  },
  {
    id: '2026-09-19-caption-lab-dataset-prompt',
    date: '2026-09-19',
    title: 'Caption Lab uses your dataset’s caption prompt',
    blurb: 'The Lab now starts with your saved caption method and uses the same character, style or concept base prompt as the dataset pass, including appearance rules and extra instructions. Expand “Prompt sent” to inspect each result’s instructions. Concept previews show the initial caption; the batch’s later refinement and omission passes are indicated separately. Thanks to adamslowe for reporting the mismatch (#68).',
    to: '/datasets?section=captions&panel=lab',
  },
  {
    id: '2026-09-15-git-update-banner',
    date: '2026-09-15',
    title: 'Git updates: one consistent answer',
    blurb: 'The banner, navigation badge and Settings now check the same update source. Git installations follow their configured branch, and checking an up-to-date branch clears an outdated banner.',
    to: '/settings/maintenance',
  },
  {
    id: '2026-09-15-studio-lower-step-counts',
    date: '2026-09-15',
    title: 'Image Studio: try fewer steps with compact controls',
    blurb: 'Try 1–5 sampling steps alongside the existing presets. Three choices stay visible; use − and + to browse lower or higher values. Your selected step counts stay listed, including in comparison and blend runs.',
    to: '/studio',
  },
  {
    id: '2026-09-15-bank-python-calculation-check',
    date: '2026-09-15',
    title: 'Recover Bank scoring when a detected GPU Python fails',
    blurb: 'Manage Score and SigLIP 2 Python from Bank ▸ Passes even when CUDA is detected. '
      + 'See the Python actually used, explicitly select the managed environment after a repair, '
      + 'and test a small calculation before starting a long pass. CUDA detection no longer '
      + 'claims that calculations are verified. Reported by perv0839 (Discord).',
    to: '/bank',
  },
  {
    id: '2026-09-14-v2-choose-your-plugins',
    date: '2026-09-14',
    title: 'Start with the LDS core and add the plugins you need',
    image: 'docs/screenshots/plugins/public-store-catalog.png',
    blurb: 'V2 separates the core setup from optional features. Start importing and organising '
      + 'datasets, then choose plugins from the Store. Each plugin brings its own screens, '
      + 'settings and preparation steps, including required ComfyUI custom nodes. '
      + 'Plugin updates are listed with their plugin; this feed covers the LDS core.',
    to: '/plugins',
  },
  {
    id: '2026-09-14-bank-score-retry-errors',
    date: '2026-09-14',
    title: 'Bank scoring recovers from failed images',
    blurb: 'Run Score again to retry failed images while keeping successful cached work. '
      + 'A failing aesthetic or NSFW scorer now preserves the CLIP index and the other scores. '
      + 'Failed images are reported clearly, and an empty semantic index no longer looks like '
      + 'an incomplete installation. Reported by perv0839 (Discord).',
    to: '/bank',
  },
  {
    id: '2026-09-07-training-speed-levers',
    date: '2026-09-07',
    title: 'Trade memory back for time: batch size, quantisation backend, checkpointing and compile',
    blurb:
      'Local training was tuned to FIT — a 12B model in 24 GB — and four of those choices were '
      + 'frozen where you could not reach them. Advanced options gains a Speed group: train 2 or 4 '
      + 'images per step, switch gradient checkpointing off, pick the quantisation backend (the '
      + 'default saves memory but cannot go faster; convrot8 quantises the activations too and does '
      + 'the maths in int8), and try compiling the model. Defaults are unchanged, and each control '
      + 'says what it costs.',
    to: '/datasets?section=training',
  },


  {
    id: '2026-09-05-zzz-video-reuse-reliability',
    date: '2026-09-05',
    title: 'Keep your clip settings when you reuse or finish a video',
    blurb: 'Reuse keeps the canvas shape of text-to-video clips, and Smooth and Neural rendering preserve the selected accelerator. Last-frame extraction also recovers cleanly if an export is interrupted.',
    to: '/studio?lane=video',
  },
  {
    id: '2026-09-05-zzz-training-cleanup-ownership',
    date: '2026-09-05',
    title: 'Keep your quantization GPU safe from training cleanup',
    blurb: 'Training cleanup now checks which task owns a rented instance, so it leaves quantization instances alone.',
  },
  {
    id: '2026-09-05-zzz-small-screen-video-controls',
    date: '2026-09-05',
    title: 'Reach video graph actions on a small screen',
    blurb: 'Checkpoint menus stay inside the screen and offer comfortable touch targets. Long render-option labels wrap so you can read the whole setting.',
    to: '/datasets',
  },
  {
    id: '2026-09-05-zzz-comfyui-input-check',
    date: '2026-09-05',
    title: 'Know when ComfyUI cannot see the image you selected',
    blurb: 'A mismatched input folder is caught before a render starts, with an explanation of which folder needs checking instead of an unexplained generation error.',
  },
  {
    id: '2026-09-05-zzz-cancelled-reads',
    date: '2026-09-05',
    title: 'Closing a picker no longer looks like a lost connection',
    blurb: 'A cancelled read no longer triggers the offline banner or a connection-error notification.',
  },

  {
    id: '2026-09-05-zz-setup-scan-startup',
    date: '2026-09-05',
    title: 'Spend less time waiting for the Setup scan',
    blurb: 'Setup checks your installed tools sooner after an app restart. Opening it reuses the current machine scan; Re-scan still checks again when you need it.',
    to: '/setup',
  },
  {
    id: '2026-09-05-setup-adopts-comfyui-input-folder',
    date: '2026-09-05',
    title: "Setup points you at the input folder ComfyUI really reads",
    blurb: "If your ComfyUI keeps its input folder somewhere else (Comfy Desktop's shared folder, a --input-directory launch), the Setup wizard's ComfyUI card now says so and offers that folder in one click, instead of leaving the fix inside an Advanced fold of Settings. Reported on GitHub (#64).",
    to: '/setup?step=comfyui',
  },
  {
    id: '2026-09-04-zzzzzzzz-workspace-header-refresh',
    date: '2026-09-04',
    title: 'Find your workspace at a glance',
    blurb: 'The header now separates workspace navigation from machine readings and tools, with matching icons, a clearer active tab and more readable resource values. Gallery no longer carries a Beta badge.',
  },
  {
    id: '2026-09-04-zzzzz-no-setup-detour-after-a-dropped-request',
    date: '2026-09-04',
    title: 'Coming back to the app no longer lands you on Setup when a request dropped on the way',
    blurb:
      'Opening the app again on a phone — after a video had rendered, say — '
      + 'sometimes showed the Setup wizard as if the machine had never been '
      + 'configured. It had: the first two requests of the page load had been '
      + 'dropped by a link that was reconnecting, and “I could not ask” was '
      + 'being read as “never set up”. A dropped request is now retried once '
      + 'and otherwise ignored; only the server’s own answer can open the '
      + 'wizard, and an install it has already seen working is never bounced.',
  },
  {
    id: '2026-09-04-per-picture-prompts-in-one-pass',
    date: '2026-09-04',
    title: 'Writing a prompt per picture no longer reloads the video model between each one',
    blurb:
      '✨ Written per picture asked the vision model once per picture — and every '
      + 'one of those asks makes ComfyUI let go of its models, so the next clip '
      + 'reloaded MiniMax H3, tens of gigabytes, once per picture. A twelve-frame '
      + 'strip paid that eleven times over for nothing. The prompts are now all '
      + 'written in a single hold of the GPU, before the first clip is queued: '
      + 'the model comes back once. Same screen, same fallbacks, same notices — '
      + 'the batch is simply not spending its time swapping weights any more.',
    to: '/studio',
  },
  {
    // Same-day ids sort the feed (date, then id): an id below the day's
    // others never reaches the top, and the badge counts nothing new.
    id: '2026-09-03-what-each-capability-unlocks',
    date: '2026-09-03',
    title: 'Capabilities: every row says what it unlocks — and the Video lane’s three doors are counted',
    blurb:
      'Settings ▸ Overview (and the wizard’s last screen) put one line under '
      + 'each capability saying what it does — "Test Studio (images)" is where '
      + 'test images are generated with a LoRA, "Video Test Studio" tests one '
      + 'in motion. Three rows join the count: ✨ DLSS 5 neural rendering, '
      + '↗ Smooth (frame interpolation) and the 🔴 Live lane — each its own '
      + 'install, each shown not-ready with its door instead of hiding behind '
      + 'a green video row.',
    to: '/settings/overview',
  },















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
    id: '2026-08-18-the-app-gets-back-up-after-a-crash',
    date: '2026-08-18',
    title: 'The app gets itself back up after a crash',
    blurb:
      'Some deaths are not something the app can catch: an antivirus hook faulting inside an image library, or a native crash in one of the GPU extensions, kills the whole process outright — no error, no message, and until now it simply stayed down until you noticed and started it again. Launched from start.bat, it now comes back on its own, says in the console that it crashed rather than pretending nothing happened, and gives up after a few deaths in a row so an app that is broken at startup cannot loop forever. Set LDS_SUPERVISE=0 to run without it.',
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
  if (idx === -1) {
    // A published entry may now belong to a plugin (or the archive). Its stable
    // dated id still marks what was read; moving it must not relabel old news.
    const seenDate = /^(\d{4}-\d{2}-\d{2})-[a-z0-9-]+$/.exec(lastSeenId)?.[1];
    return seenDate ? s.filter((e) => e.date > seenDate || (e.date === seenDate && e.id > lastSeenId)) : s;
  }
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
  '/guide', '/help', '/setup', '/plugins',
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

  const pluginSettings = path.match(/^\/plugins\/([^/]+)\/settings$/);
  if (pluginSettings) return !section && !panel && registeredDescriptors()
    .some(({ descriptor }) => descriptor.id === pluginSettings[1]);

  // /datasets with an optional ?section=<id>&panel=<id> workspace deep-link.
  if (path === '/datasets') {
    if (!section) return !panel; // plain /datasets, no orphan panel
    const ws = WORKSPACE_SECTIONS.find((s) => s.id === section);
    if (!ws) return false;
    if (!panel) return true;
    return sectionPanels(ws).some((p) => p.id === panel);
  }

  // /guide/<slug> — the Guide owns its own section slugs; any non-empty one is fine.
  if (path.startsWith('/guide/')) {
    return path.length > '/guide/'.length && !section && !panel;
  }

  // Everything else must be a bare, param-less top-level route.
  return (TOP_LEVEL_ROUTES.has(path) || pluginRoutes().some(route => route.path === path))
    && !section && !panel;
}
