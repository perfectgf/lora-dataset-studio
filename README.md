# LoRA Dataset Studio

[![CI](https://github.com/perfectgf/lora-dataset-studio/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/perfectgf/lora-dataset-studio/actions/workflows/ci.yml) [![Join our Discord](https://img.shields.io/discord/1525908170331914411?logo=discord&logoColor=white&label=Discord&color=5865F2)](https://discord.gg/j6hnJBFtXE) [![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/perfectgf) [![Support on Ko-fi](https://img.shields.io/badge/Ko--fi-Buy%20me%20a%20coffee-FF5E5B?logo=kofi&logoColor=white)](https://ko-fi.com/perfectgf)

**A complete, self-hosted LoRA workflow in one browser tab:** source or generate a Character, Concept or Style dataset, curate it, caption it, clean watermarks, train locally or in the cloud, then compare checkpoints before export.

No account, paid tier or telemetry. API engines and rented GPUs are optional; local and manual workflows remain available.

> New here? Start with [Setup & install](#setup--install), then follow the [end-to-end workflow](docs/guide/workflow.md). The [documentation index](docs/README.md) links every guide. Project news and current development live on [Discord](https://discord.gg/j6hnJBFtXE).

### 📖 [The complete guide — every feature, screen by screen →](docs/guide/using-the-app.md)

Everything the app can do, in one long read: [getting started](docs/guide/getting-started.md) · [the full workflow](docs/guide/workflow.md) · [every setting explained](docs/guide/settings-reference.md) · [Docker](docs/guide/docker.md) · [troubleshooting](docs/guide/troubleshooting.md).

### ▶️ Watch the whole thing, start to finish

A real Character LoRA built end to end in seven minutes, unedited and without narration:

https://github.com/user-attachments/assets/d51ff89c-34e9-41a9-b47d-08939a8c867b

<table>
  <tr>
    <td width="45%" valign="top">
      <a href="docs/screenshots/generate/generate-variations.png"><img src="docs/screenshots/generate/generate-variations.png" alt="Generate variations: the subject type, and the five engines side by side with what each one costs per image, whether it runs on your GPU or bills an API, and which ones refuse adult content" width="100%"></a>
    </td>
    <td width="55%" valign="top">
      <a href="docs/screenshots/02-workspace.png"><img src="docs/screenshots/02-workspace.png" alt="Guided dataset workspace: a progress rail mapping reference, generation, curation, captioning and training, next to the curation grid and its bulk actions" width="100%"></a>
    </td>
  </tr>
  <tr>
    <td valign="top"><sub><strong>Pick where the images come from</strong> — the five engines side by side, each stating its price per image, whether it runs free on your GPU or bills an API, and which ones refuse adult content. The first question anyone asks of this app, answered before you commit to it.</sub></td>
    <td valign="top"><sub><strong>Then one workspace for the whole route</strong> — a progress rail from reference to Studio, beside the grid where you keep, reject, re-caption or send a selection back through an engine in bulk.</sub></td>
  </tr>
</table>

<p align="center"><em>Every person shown in these screenshots was produced by the app's own generation engines; no real individual is depicted.</em></p>

## What it does

### Build any dataset

| Capability | What it provides |
|---|---|
| **Character / Concept / Style** | Kind-aware captioning, masking, readiness checks and training policies rather than three cosmetic labels |
| **Human / Animal / Creature / Object / Other / Anime subjects** | Subject-specific identity wording and shot catalogs; Anime protects the character design and illustrated rendering instead of forcing photorealism |
| **Five generation engines** | Nano Banana Pro (Gemini), ChatGPT (`gpt-image-2`), OpenRouter, or local Klein and Krea 2 Edit through ComfyUI |
| **Several engines in one batch** | Tick multiple engines and split one shot list across them; every result remains labelled with the engine that produced it |
| **Krea 2 Edit** | Restage a single reference while preserving identity, without needing a character LoRA first; the selected card controls output framing |
| **Variation catalog** | Balanced expression, angle, lighting, framing, outfit and background shots; import/export the catalog as JSON and keep custom entries |
| **Reference editing and exact retry** | Edit the main reference through any available engine, compare before/after, then retry with the exact prompt, engine and temporary references |
| **Import or scrape** | Drag in images, merge ZIP/folder datasets, search Reddit, Pexels or the open web by keyword, or scan a gallery/direct-media URL — a dozen sites have a dedicated handler that enumerates a profile or a gallery properly, several of them needing your own credentials entered once in **Settings → Source credentials**, and anything else goes through gallery-dl and whatever its bundled extractors cover. A site gallery-dl has no extractor for shows "No images found on this page" (the fallback item exists internally but is video-typed, so the image picker filters it out); a URL the app refuses outright — a retired source, a non-public host — says so as an error instead |

API generation follows each provider's billing and content policy. Read the direct notes for [Gemini](docs/guide/settings-reference.md#what-the-gemini-engine-will-and-will-not-do), [ChatGPT subscription mode](docs/guide/settings-reference.md#chatgpt-subscription-experimental), [OpenRouter and image-engine settings](docs/guide/settings-reference.md#image-engines), and [Pexels authorization](docs/guide/workflow.md#the-built-in-web-scraper) before using those lanes. The local engines do not send reference images to an API.

### Image Bank

A dataset is the thirty images you train on. A **bank** is the three thousand you had to look at to find them — and looking at three thousand images by hand is where most datasets die.

Point a bank at a folder, or scrape straight into one. It reads what is there **in place**: your files are never modified, moved or renamed, and the single action that does touch the source folder announces itself in capitals before it runs. Then **one pass measures the whole pile**, and every question afterwards is answered against those measurements instead of against your eyes — what is blurry, what is a duplicate of what, who is in it, how it is framed, whether it is a photograph or a render, and what it actually shows. You keep, reject and shortlist; a kept selection graduates into a dataset with its analysis attached, and can come back the other way.

The cuts are measured rather than guessed: the aesthetic and near-duplicate thresholds were calibrated on a real bank of **7,316 images**, and every measure that cannot answer says "unsure" or "not measured" instead of inventing a verdict. The image lane is out of Beta; the **video** lane still carries the chip, and says why below.

<table>
  <tr>
    <td width="62%" valign="top">
      <a href="docs/screenshots/bank/bank-analyze-and-overview.png"><img src="docs/screenshots/bank/bank-analyze-and-overview.png" alt="The Bank workspace: the Analyze panel with every pass, the three-level watermark cleaning, and a Bank overview reporting coverage, resolution, framing, medium and structure across 50,461 images" width="100%"></a>
    </td>
    <td width="38%" valign="top">
      <a href="docs/screenshots/bank/bank-launch-all.png"><img src="docs/screenshots/bank/bank-launch-all.png" alt="The Launch all dialog: eight passes ticked, each quality flag quoting how many images it would reject, and a warning that unscanned images will change those counts" width="100%"></a>
    </td>
  </tr>
  <tr>
    <td valign="top"><sub><strong>The workspace</strong> — every pass on the left, and on the right what the bank actually <em>is</em>: how much of it each pass has covered, its resolutions, framings, mediums, and how many duplicate and person groups are still unresolved. 50,461 images here, 93% measured for quality, 49% scored.</sub></td>
    <td valign="top"><sub><strong>Launch all</strong> — the whole triage in one go. Every flag quotes what it would reject <em>today</em>, and says out loud that 3,602 images have not been scanned yet, so those counts will grow. Stop it any time; a pass whose tool is missing is skipped, never failed.</sub></td>
  </tr>
</table>

| Capability | What it provides |
|---|---|
| **Folder or web scrape → bank** | Inventory a live folder in place, or scrape into a new/existing bank without applying dataset filters on the way in |
| **One scoring pass, many answers** | A single pass produces aesthetic, NSFW and style rankings, groups, and embeddings every later feature reuses instead of re-inferring |
| **Quality and similarity passes** | Flag blur, noise, flat frames, small/soft-detail images and black bars; group duplicates, crops and recompressions — then, in a second pass, the near-duplicates only an embedding sees, with a per-run threshold and its own groups kept apart from the pixel ones |
| **Auto-reject by flag** | Turn any set of quality flags into a bulk rejection in one click, on the number it will really reject rather than the number flagged — and undo it as one decision |
| **People, framing and captions** | Cluster faces without a reference, classify face/bust/body/back, and caption the bank for full-text search — choosing the engine, the vision model and the pile per run, without changing your settings — kept + undecided by default, or kept, undecided, unkept (the bin) or all. Aiming a pass at the bin is never the default and states what it costs, and the button quotes the number it will actually write rather than the size of the pile. Every caption records **who wrote it** — the engine, or you — so once a pile is fully captioned, **Re-caption** redoes it with another model while keeping the ones you wrote or corrected by hand, unless you tick the box that rewrites those too. It counts the three cases before you click, including the captions written before the app tracked authorship: those cannot be told apart, and no undo covers captions |
| **Medium and head angle** | Split a mixed dump into photographs, anime, 3D renders and illustrations (reusing the scoring embeddings, no new inference), and filter by frontal / three-quarter / profile / back view. Both answer "unsure" or "not measured" instead of guessing, and both say so on screen: non-photo verdicts are rare by design, and profiles are under-counted because a hard-turned head often defeats face detection |
| **Find and shortlist** | Find by text, pick diverse, make framing-balanced picks, find similar images, or promote a shortlist into a new bank. You can also describe the set you want in a sentence and let the app set its own filters from it: the model reads the words only, never your images, so it moves the same chips you can edit and the counts beside them stay the measured ones. It says when a request has nowhere to land rather than inventing a filter, and it will not turn an exclusion into a search phrase, because the ranker returns more of a negated thing, not less |
| **Coverage advice** | A green readiness meter says the set is big enough; it does not say the set is varied. Coverage reads the labels, the scoring embeddings and the captions to name what the pool never shows — no profile views, one outfit, eye level only — because twenty-five versions of one photograph pass every other check. Advice only: nothing is kept or rejected, and what could not be measured says so instead of drawing an empty bar |
| **Fast review tools** | Filter, sort, review one by one, rotate without rewriting the source, compare an improved candidate with its original, and re-run only eligible passes |
| **Editable watermark masks** | Detect marks, redraw several mask zones, then crop or repaint into a separate clean derivative. Your source file is never written to: the clean copy lives beside it, and each level asks which pile it should run on (kept, undecided, unkept or all) before touching anything. Undo drops the clean copy and restores the original as the one in use, for the whole bank rather than a single run, and it skips any image whose source changed on disk since; an image already promoted into a dataset keeps the cleaned copy that went with it |
| **Dataset ↔ bank round trip** | Promote bank keepers into a dataset or copy dataset keepers into a bank while retaining compatible metadata and provenance |
| **Safe bulk work** | Undo the last bulk decision, tune thresholds where you work, move a bank without losing analysis, or run the full chain overnight |
| **The one destructive action** | Everything above leaves your files alone — 🗑 **Delete rejected** is the single bank action that touches the source folder. It sends the rejected files to the OS trash (or the app's own, or deletes them) behind a type-DELETE confirmation that first states how many files, where they go, and which other banks share that folder. It refuses outright when the folder is also a dataset's |

### Video Bank *(Beta — first release, read the limits)*

Turns long source videos into a **video training set**: a flat folder of `.mp4`
clips with matching `.txt` captions, cut to the exact frame count and frame rate
the target model accepts.

| Capability | What it provides |
|---|---|
| **Folder → video bank** | Point a bank at a folder of videos. It is referenced **in place** and never written to, like the image bank |
| **Automatic shot detection** | Finds the cuts with TransNetV2, so a long file becomes individually reviewable shots instead of one blob |
| **Review without waiting** | The grid shows thumbnails; a click plays that shot from the source, so nothing is encoded before you have decided |
| **Target-aware cutting** | Pick the model you are building for and the clip length offers **only counts that model can actually ingest** — Wan wants 4n+1 frames, LTX 8n+1, MiniMax H3 five modulo seventeen, and none of them will tell you if you get it wrong |
| **Encode only what you keep** | Cutting a clip means re-encoding it, so that is paid once, at promotion, for the clips you kept. A bank of 400 shots you triage down to 120 encodes 120 files, not 400 |
| **Fix a bad cut instead of rejecting it** | Trim either bound (by 1 s or one frame *of your source*), split a shot at the playhead, or draw a shot the detector missed. Bounds only — there is no scrubbable timeline. For image-to-video targets the first frame is the conditioning image, so moving a start picks what the model animates from, and the panel says so |
| **Measure every shot, choose your own cuts** | One pass reads every frame and scores stillness, blur, black moments and frozen stretches. Flags mark shots to *look at* — nothing is auto-rejected — and there are **no default thresholds**: a preview shows how many shots each cut would flag against *your* bank's own distribution before you apply it |
| **Sound measured, not assumed** | For the targets that keep an audio track (LTX, MiniMax H3), every shot is scored for **how much of it is silence** and its **level in dBFS** — because a dataset of silent clips teaches the model to be silent and the file on disk gives nothing away. "No track", "silent" and "not measured yet" stay three different answers |
| **Cap one source's share** | Optional cap on how many clips a single file contributes, so a 50-clip set is not quietly three videos over-represented. Keeps each source's earliest clips (same bank, same dataset — not a random sample), and the result reports the share it ended up with |
| **Trim the transition off both ends** | Optional per-export trim of both bounds (0 by default). A clip the trim makes too short for the target's frame count is **dropped, never exported short** — and counted separately from clips that were never long enough, since only one of the two is fixed by lowering the trim |
| **Train it without leaving the app** | A promoted set gets a ▶ Train button that runs it through the ai-toolkit installed here — no export, no hand-written config. It queues behind the same GPU as everything else: a captioning pass, a ComfyUI render or an image training in flight refuses the launch instead of racing it |
| **Shots described in words** | A pass writes what HAPPENS in each shot ("a woman turns and walks away"), which becomes the clip's `.txt` — the prompt it trains on. Captions are drafts: editable per shot, and a re-run never overwrites what you wrote |
| **Spot the shot you already have** | A pass compares every shot to every other and groups the near-identical takes — ten copies of one gesture do not teach a model ten things. Each pile keeps its **sharpest** member unflagged, so you know which one to keep, and flagged shots can be selected and rejected in one gesture. It costs no GPU and no new decode: it reuses the frame vectors *Find a scene* already cached |
| **Spot the watermarked shots** | A logo burned into the same corner of every frame is the most consistent thing in a dataset, so it is the first thing a LoRA learns to draw — and it is invisible at thumbnail size. An optional pass runs the same detector the image bank uses over each shot's sharpest frame and flags what it finds. Needs the watermark detector from Setup; a shot it could not judge is counted apart and reported as one it **could not judge**, never folded into the clean ones |
| **Find a scene by typing a word** | One pass looks at a few frames of every shot; after it, typing *a woman walking on a beach* ranks the bank instantly and tells you **which second** of each shot matched. Several frames per shot, so a subject that only appears at the end is still findable. It is a **ranking, not a filter** — every shot scores something against every phrase — and the model **ignores "without"**, so `-word` pushes something down instead |

**What it does NOT do yet**, plainly:

- **No aesthetic scoring.** The technical measures above say what is broken, not
  what is beautiful — quality ranking and "most varied" are still to come.
  Searching by words ranks shots by what they LOOK like, which is a different
  question from whether they are any good.
- **Near-duplicates are found, but the threshold is inherited, not measured on
  video.** ✂ Duplicates groups shots at a cosine cut carried over from the image
  bank's own calibration over the same CLIP space; no video-pair calibration
  exists yet. It also compares two shots at their *closest* pair of frames, which
  reaches any given cut more easily than a single-image comparison — so on a bank
  of similar-looking material, expect to raise it.
- **No audio captioning, and no audio in the search.** The sound is measured
  (silence and level) but never described, and 🔎 Find scenes reads frames only —
  "a door slamming" describes nothing it can see.
- **Captioning is per-shot prose, not tags.** Every promoted clip gets a `.txt`:
  its caption when it has one, and an **empty** file when it does not. The file is
  always written, because a missing one crashes one trainer and makes another drop
  the clip silently — and an empty one trains uncaptioned, which is why the build
  dialog counts them out loud before encoding.
- **Training starts here, but only one target is proven here.** A promoted set
  has a **▶ Train this dataset** button that hands the clips to the ai-toolkit
  installed on your machine, and the cloud lane rents a pod for the same set. The
  eight offered targets (Wan 2.1 T2V and I2V, Wan 2.2 T2V and I2V A14B, Wan 2.2
  TI2V-5B, LTX-2 and 2.3, MiniMax H3) are exactly the video architectures that
  ai-toolkit ships, and each one's settings were read in its code — plus a
  "Generic / other" escape hatch that imposes no frame rule at all. But **Wan 2.2
  14B is the only one a finished run has been through here**, and the card says so
  on the others. Measured on that run: 24 GB was full, at 170-185 s per step, with
  the CPU offload that makes 24 GB possible at all. Only three bases are stated by
  anything installed locally, so **five of the eight need you to name a base
  repository** — both I2V Wan variants, Wan 2.2 TI2V-5B, and both LTX-2 versions.
- **The cloud lane names what it is billing.** The **☁ Train in the cloud** panel
  shows the GPU and its hourly price as soon as the pod reports them — while the
  run is alive, not before you click — and before the job starts it asks the pod
  to decode one of the clips you just uploaded, so a machine that cannot read them
  fails in the first minute instead of billing hours of training on nothing. A
  target with no verified base repository is **refused with the reason** rather
  than launched on a guessed repo id; the base can only be supplied through the
  API today, not from the panel. Wan 2.2 **A14B** saves each checkpoint as a
  **pair** (high-noise and low-noise experts) — the 5B does not — and both files
  are offered together, because either one alone is a LoRA nothing can load.
- **MiniMax H3 needs about 43 GB of weights, and will say so rather than fetch
  them.** They come from `Comfy-Org/MiniMax-H3`. If they are not on your disk the
  button names the repository and the size and waits for a yes — a first run that
  quietly downloaded 43 GB would look like a training that had hung.
- **MiniMax H3 is licence-restricted.** Its community licence grants no rights in
  the EU, the UK, South Korea or the USA, and the restriction covers the model's
  outputs, not only the model. Check your own territory before using that profile.

### Curate, caption and clean

| Capability | What it provides |
|---|---|
| **Curation grid** | Keep/reject, crop, mirror, rotate, zoom, resize, multi-select and non-destructive upscale candidates from either engine — Klein re-renders detail (sharper, but skin and colour can shift), SeedVR2 resolves detail and leaves the original look alone |
| **Identity and composition checks** | InsightFace similarity, score-based auto-triage, framing badges and a live Character composition meter |
| **Model-matched captions** | Prose or booru form selected by target family, with kind-aware Concept leak checks and content-only Style rules |
| **Caption Lab and recovery** | Find/replace, tag frequencies, expanded editing, targeted re-captioning, stoppable batches and reload-proof recovery |
| **External caption round trip** | Export ordinary image/`.txt` pairs, caption them in any tool, then re-import without duplicating images or overwriting non-empty LDS captions |
| **Dual long + short captions** | ai-toolkit text-side augmentation for supported local families; both wordings remain editable per image |
| **Watermark review** | Detect, review and edit masks; choose crop or LaMa/Klein inpaint; every edit keeps an `.orig` backup and **Restore original** supports another attempt |

### Train, compare and continue

| Capability | What it provides |
|---|---|
| **Guided local training** | ai-toolkit underneath, family-scoped starters, adaptive step policies, launch guards, queueing and advanced controls |
| **Slider LoRA (Beta)** | Train a bipolar conceptual slider from positive and negative prompt poles, so LoRA strength moves the learned trait in either direction and Test Studio can sweep both sides |
| **Cloud training** | Rent a vast.ai GPU from the same launch flow, stream progress and checkpoints home, and terminate pods automatically |
| **Parallel cloud runs** | Run several cloud trainings on one dataset at once to A/B toolkit settings — each run rents its own pod (billed separately), capped by the concurrent-runs ceiling in Settings |
| **Full-model training (Krea 2)** | Train the whole transformer instead of an adapter. The finished master lands on your own disk and is verified before the pod is destroyed, appears in 📦 Checkpoints & LoRAs next to the ~10 GB fp8 twin ComfyUI loads, and can be continued from the step written inside it. The lane is cloud-only and Krea 2 only, and it accepts Raw, Turbo or a Krea 2 checkpoint of your own. Turbo is allowed with a warning nobody can honestly skip — a full-model run on a distilled base has not been measured, by us or by anyone, and it may cost the model its few-step behaviour. A ComfyUI scaled-fp8 export is refused outright as a base: the trainer cannot load one |
| **Merge a LoRA into a checkpoint** | Fold one or more of your LoRAs into a base, each at its own weight, and get a complete model you can publish. A plan answers first, from the file headers alone: how many tensors change, how big the output is, which drive it lands on, how long it takes, and what a half-way failure leaves. What comes out is a **merged** model, not a trained one — the file's own metadata records the base, every LoRA and its weight, so it stays true after a rename. It is also the published route to getting few-step speed back on a Raw full model, by folding in the re-distillation LoRA Krea publishes for Turbo; that one we have not tested ourselves, and the screen says so before you start it |
| **Custom bases and continuation** | Train compatible custom weights, continue from any saved epoch, or use verified full-state resume where available |
| **Runs** | Local and cloud runs together with progress, logs, stop/retry/continue/download actions and paste-safe config sharing |
| **Experiment lineage** | Inspect, annotate and diff the exact tree of runs and the checkpoint each continuation resumed from |
| **LoRA Canvas** | Put every dataset's lineage on one pan/zoom board, rearrange cards, compare runs across datasets, generate from same-family checkpoints — including 🧬 blending several checkpoints into one image, with purple provenance edges joining a blended picture to every pill it came from (blends made before this feature show a badge instead) — pin/fuse outputs and continue training from a pill; each generation run keeps its own strip in training-step order, with the character dataset's reference face on its lane. A 🔌 + LoRA button pins any LoRA from your ComfyUI folder onto the board as its own plugin node, with its own strength — it stacks onto a run anchored by a checkpoint trained here, not as a solo generation on its own |
| **Test Studio** | Fixed-seed checkpoint × strength grids, multi-LoRA comparisons or 🧬 combined stacks (several of your LoRAs in one image, each at its own weight, weight variants compared side by side), a ✨ Enhance button that enriches your prompt through your local Ollama, votes, Wilson ranking, face ranking and shareable exports |
| **Studio shortcuts and recovery** | Open Studio directly from a run, draw prompts from kept dataset captions, and pause safely when ComfyUI drops instead of launching later cells against changed state |

### Keep control of the files

| Capability | What it provides |
|---|---|
| **Training ZIP and sidecars** | Standard kept image + same-stem `.txt` pairs for ai-toolkit/Kohya-compatible tools |
| **Portable backup and restore** | Datasets, decisions, captions, settings and run history in one file; API keys stay out |
| **Hugging Face publishing** | Publish kept pairs to a dataset repository, private by default and gated by an explicit rights confirmation |
| **ComfyUI deployment** | Deploy individual LoRA checkpoints or downloaded cloud results into the configured LoRA tree; a full model's fp8 twin goes to ComfyUI's own diffusion-models folder instead, hard-linked when it sits on the same drive so it costs no second copy. The full-precision master is never sent — it is the only file you can train from again |
| **Recoverable deletion** | Deleted app data goes to Trash; destructive Image Bank actions state their destination before confirmation |
| **Storage you can see and move** | Settings › Storage lists every folder the app writes to with its path and (on request) its size, and can point the dataset root, the cloud run staging and the checkpoint store at another drive — moving what is already there, or adopting the new folder empty, never silently. Trained checkpoints live in their own store that no cleanup touches; the trash sits on the same disk, so space returns only when you empty it. The same tab shrinks any full-precision `.safetensors` on this machine to the ~10 GB fp8 file ComfyUI loads, and chooses where a finished full model is delivered. |

### A quick visual tour

<table>
  <tr>
    <td align="center" width="50%">
      <a href="docs/screenshots/bank/bank-overview.png"><img src="docs/screenshots/bank/bank-overview.png" alt="Image Bank overview with scoring, filters and review controls" width="380"></a><br>
      <sub><strong>Image Bank</strong> — score, search and shortlist large collections.</sub>
    </td>
    <td align="center" width="50%">
      <a href="docs/screenshots/03-curate.png"><img src="docs/screenshots/03-curate.png" alt="Dataset image grid with keep/reject decisions, face-similarity scores and per-tile caption fields" width="380"></a><br>
      <sub><strong>Curate</strong> — review, repair and balance the training set.</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="docs/screenshots/training/runs-hub.png"><img src="docs/screenshots/training/runs-hub.png" alt="Training Runs hub showing local and cloud experiment progress" width="380"></a><br>
      <sub><strong>Runs</strong> — follow local and cloud experiments together.</sub>
    </td>
    <td align="center" width="50%">
      <a href="docs/screenshots/studio/studio-grid.png"><img src="docs/screenshots/studio/studio-grid.png" alt="Test Studio grid comparing checkpoints and LoRA strengths" width="380"></a><br>
      <sub><strong>Test Studio</strong> — compare checkpoints at fixed seeds and strengths.</sub>
    </td>
  </tr>
</table>

<p align="center">
  <a href="docs/screenshots/canvas/canvas-board.png"><img src="docs/screenshots/canvas/canvas-board.png" alt="LoRA Canvas board with run cards, checkpoint pills and the run inspector showing the frozen training settings" width="780"></a><br>
  <sub><strong>LoRA Canvas</strong> — every run on one board, and the exact recipe behind each one.</sub>
</p>

The detailed journey, screenshots and operational notes now live in the [workflow guide](docs/guide/workflow.md).

### Roadmap

Directions, not dates. These are discussed openly on the project's Discord, and the most-requested ideas move up the list.

- **🧬 Merge Lab** *(partly shipped)* — baking your LoRAs into a standalone checkpoint has landed, and so has full-model training on Krea 2. What is left is the *lab* part: **model ↔ model** merges with guided recipes, judged side by side in the Test Studio (same seeds, A/B grids), and a full-model lane that runs somewhere other than a rented pod.
- **🎬 Video LoRAs** — *the dataset half exists and training now launches from the app* (see **Video Bank** above): shot detection, quality measures (motion, exposure, freeze, audio), captions that describe the action, keyword search across shots, target-aware cutting into a trainable folder, and a ▶ Train button that runs the set through your local ai-toolkit or a rented pod. What remains is proving the targets beyond Wan 2.2 with a finished run each, and testing the resulting LoRAs in-app. Community-driven.
- **🧠 Watermark cleaning during import** — cleaning that happens **during import** instead of as a separate errand, and automation you can trust unattended. *(Detection has caught up: a dedicated detector that needs no vision model now ships alongside the Ollama path, and manual two-pass cleaning already works in datasets and in the Image Bank.)*
- **🧩 More base models** — additional Flux-family bases (Chroma, Qwen-Image…) with the same one-click flow as Krea 2.

## Why this instead of ai-toolkit?

"Instead of" is the wrong frame: this app is **not a competitor to [ai-toolkit](https://github.com/ostris/ai-toolkit) — it orchestrates it**. ai-toolkit is the training engine; LoRA Dataset Studio adds the work before, around and after a run.

| Stage | ai-toolkit alone | LoRA Dataset Studio |
|---|---|---|
| Build from references | ❌ bring your own images | ✅ five engines, simultaneous multi-engine batches, subject-aware catalogs including Anime, reference edits and exact retries |
| Build from the web | ❌ none | ✅ Reddit, Pexels, keyword search across the open web, and gallery/direct-media URL scans (through gallery-dl, which covers several hundred sites) into a dataset or Image Bank, with deduplication and explicit provider warnings |
| Triage a large dump | ❌ none | ✅ Image Bank scans, scores, search, filters, sorts, balanced/diverse shortlists, watermark masks and dataset round trips |
| Curate and repair | ❌ external file tools | ✅ keep/reject, crop/mirror/rotate, InsightFace scoring, composition guidance, improve/compare and recoverable originals |
| Captions | ❌ write or prepare them yourself | ✅ JoyCaption/Ollama, kind/family rules, Caption Lab, external `.txt` round trip and dual-caption support |
| Masked training | ⚙️ consumes masks you supply | ✅ generates Character masks, supports Concept face masks and disables unsafe kind combinations |
| Training | ✅ **it is the engine** — direct YAML/config control | ⚙️ guided/scoped recipes, preflight guards, advanced controls, queueing, local/cloud lanes and continuation |
| Track experiments | ⚙️ inspect outputs manually | ✅ Runs hub, lineage graphs and a cross-dataset LoRA Canvas with notes, diffs, galleries and actions |
| Pick a checkpoint | ❌ samples + your eye | ✅ Test Studio grids, multi-LoRA comparison, dataset-caption prompts, votes/rankings, outage-safe pause and export |
| Move or publish | ⚙️ manual file handling | ✅ ZIP/sidecars, portable backup/restore, folder merge, ComfyUI deployment and optional Hugging Face publishing |

**Honest verdict:** the studio is strongest when you want one guided path from raw images to a reviewed LoRA. A raw ai-toolkit config still exposes the widest surface for unsupported architectures and experimental keys. Standard ZIP/sidecars keep both workflows interoperable.

## Feature matrix by backend

Missing dependencies are shown in Setup/Settings and gated features stay unavailable until their requirements are satisfied. Setup's closing screen lists the installable capabilities — including bank scoring, the optional SigLIP 2 engine, the watermark detector and the scraping extras — and each row that is not ready leads to the step that installs it. The SeedVR2 upscaler is the exception: it installs from its own Setup ▸ ComfyUI card and is not counted on that screen.

| Feature | Requires |
|---|---|
| Nano Banana Pro generation | `GEMINI_API_KEY` |
| ChatGPT / `gpt-image-2` generation | `OPENAI_API_KEY`, or the separate experimental ChatGPT-subscription connection |
| OpenRouter generation | `OPENROUTER_API_KEY` plus an image-capable model slug; OpenRouter billing and the upstream provider's policy still apply |
| Klein generation / improvement | ComfyUI reachable + Klein model stack |
| SeedVR2 upscaling | ComfyUI reachable + the `ComfyUI-SeedVR2_VideoUpscaler` node pack (installed from ComfyUI, not by this app — it has its own Python dependencies) + two model files the Setup step downloads (~3.9 GB); big frames are upscaled in overlapping tiles by default when the optional `Comfyui_TTP_Toolset` pack is present (a `tiling` setting keeps `always`/`never` available); [exact files](docs/guide/settings-reference.md#seedvr2-upscaling-local) |
| Krea 2 Edit generation | ComfyUI reachable + `comfyui-krea2edit`, a Krea 2 base, Identity Edit LoRA, Qwen3-VL encoder and Qwen Image VAE; [exact files](docs/guide/settings-reference.md#krea-2-edit-local) |
| Captioning | Ollama **or** ai-toolkit (JoyCaption) |
| Dual long + short captions | ai-toolkit + local vision caption derivation; local training only, and unavailable for Krea 2 / Anima |
| Auto-framing / auto head-crop | Ollama with a vision model |
| Face similarity / auto-triage | `backend/requirements-ml.txt` (InsightFace + ONNX Runtime) |
| Character person masks | `backend/requirements-ml.txt` (rembg); Concept/Style intentionally disable them |
| Image Bank scoring, crops and semantic tools | The Bank scoring extra provides CLIP and ✨ Score. Each Bank can instead select the optional pinned SigLIP 2 engine from Setup; it builds a separate index, while aesthetic/NSFW/style/medium remain on CLIP. Balanced picks also need Framing. Both ship **CPU-only PyTorch** on purpose; on a machine that already has a CUDA Python (ai-toolkit's, ComfyUI's) each can be pointed at it instead — checked package by package, never installed into, and separately for ✨ Score and for SigLIP 2. |
| Watermark detection | Ollama with a vision model, **or** the dedicated detector (torch + transformers — the bank-scoring extra's environment is reused when present — plus ~0.9 GB of model downloads at first use) |
| Watermark inpainting | LaMa extra from `backend/requirements-ml.txt`, or ComfyUI + Klein for the refine lane; crop remains model-free |
| Scraping | `backend/requirements-scrape.txt`; Pexels also needs `PEXELS_API_KEY` and explicit authorization. Gallery/URL scanning goes through gallery-dl for any site it recognizes, whatever its bundled extractors cover; an unrecognized site returns "No images found" in the picker (the single item gallery-dl's yt-dlp fallback can still fetch is video-typed, so it never reaches the image list), and a listing of albums returns one cover per album unless **Scan full albums** is ticked. A scan that was cut short — by the time budget, a result cap, or a source that blocked or rate-limited it — now says so under the results ("this scan stopped before the end of the listing"), instead of presenting a partial list as the whole thing. Web image search needs no key — it queries a metasearch layer over several backends and asks for photos, but the filter is not honored uniformly, so some non-photo results can still come through; results are capped per search rather than guaranteed — a request for the 120 maximum routinely comes back with far fewer — come from third-party sites whose licence is your responsibility, and a few links — mainly stock-photo CDNs that redirect to the actual file — are refused by the hardened fetch that protects every import |
| Video Bank — reading and triaging | `backend/requirements-ml.txt` (PyAV). Shot detection additionally needs `transnetv2-pytorch` (weights bundled, nothing to download), which rides the bank-scoring environment because it pulls torch. The three pieces install and fail **apart**, and Setup reports them as three separate rows |
| Video Bank — cutting clips into a dataset | An ffmpeg binary: `imageio-ffmpeg` ships one, or any ffmpeg on PATH. Needed **only to promote** — without it you can still scan, detect shots, watch and triage a whole bank |
| Video Bank — shot captions and scene search | The Bank scoring extra's environment (torch + `transformers` ≥ 4.57) plus a Qwen3-VL checkpoint downloaded at first use; the model is a setting, and the same environment serves ✨ Score, SigLIP 2 and the watermark detector |
| Civitai scanning | `backend/requirements-scrape.txt`; without `CIVITAI_API_KEY` the scan runs but returns SFW results only |
| Local LoRA training: Z-Image / Krea 2 / FLUX.1 / FLUX.2 Klein / Anima | ai-toolkit; no ComfyUI is needed for official Hugging Face bases. Krea 2 can start from any Krea 2 checkpoint already on your disk instead — including one a full-model run delivered — discovered through ComfyUI's model tree; an ordinary fp8 build trains (the trainer up-casts it, and the app says with numbers how much precision that cast dropped), while a packed ComfyUI export is refused because it carries decompression tables a trainer cannot load |
| Local SDXL training | ai-toolkit + a base checkpoint discoverable in ComfyUI's model tree |
| Cloud training | `VAST_API_KEY`; supported families are shown in the launch UI. Full-model Krea 2 also needs `HF_CLOUD_TOKEN` with Krea base read and repository write access; fine-grained is recommended, global `role=write` is accepted with a warning, and read-only is rejected. A finished full model (~26 GB, plus its ~10 GB fp8 twin) is downloaded **to your machine** and verified before the rented pod is released; a copy of the master is then pushed to a private Hugging Face repository as a backup, and it can be turned off. Either copy can seed a fresh pod when you continue the run: the Hugging Face one is minutes over a datacenter link, the one on your machine costs your upload speed — and the ▶ Continue dialog shows both durations and what each one costs in rented GPU time before you pick. So you need **room on the checkpoint drive** (checked before anything is rented), and Hugging Face room only for the backup; Settings ▸ Storage lists what is taking that space |
| Quantizing a model to fp8 (Settings ▸ Storage, or a full model's card) | A Python with `torch`; the interpreter is probed before the button is enabled, so a missing package is a refusal with its pip line, not a crash thirty seconds in. Runs on the CPU, one at a time, so it never takes VRAM from ComfyUI or a training run. Your source file is never modified or overwritten; an already-quantized file, or an adapter, is refused |
| Merging a LoRA into a base checkpoint (produces a full model) | A Python with `torch` (the same one fp8 quantization uses) and room for a second copy of the base — a 26 GB Krea 2 base takes about two minutes and writes 26 GB. Refused on an already-quantized base: merge into the full-precision file, then quantize. LoRAs must name their modules the way the base names its weights (the ai-toolkit/diffusion-model convention); kohya's flattened `lora_unet_…` SDXL exports do not, and are refused by name before anything is written. The result is a **merged** model, not a trained one, and its metadata says so |
| LoRA Canvas browsing, layout, notes and diffs | No external service; generating needs ComfyUI and same-family checkpoints, continuing needs the chosen local/cloud training lane |
| Test Studio | ComfyUI reachable + assets for a supported Studio family |
| Backup/restore and ZIP/folder merge | No external service |
| Hugging Face publishing | Write-enabled `HF_TOKEN`; repositories are private by default |

## Run it your way

| Mode | Good for | What is optional or unavailable |
|---|---|---|
| **Docker + existing ComfyUI** | Run LDS in Docker while keeping the ComfyUI already installed on the host | The launcher asks for the ComfyUI folder once; local training still uses host ai-toolkit or the cloud |
| **Docker GPU + fresh ComfyUI** | Run LDS and a new isolated ComfyUI together on an NVIDIA GPU | Existing ComfyUI/models stay untouched; local training still uses host ai-toolkit or the cloud |
| **Full local** | Local engines, ML helpers, ai-toolkit training, Canvas generation and Test Studio | Install/connect only the tools you need; each capability degrades independently |

## Setup & install

On first launch, **Setup** scans the machine and links every missing capability to its install/configuration step. You can skip optional tools and begin with imported images immediately.

### Option 1 — release ZIP + start.bat (Windows)

Download **`LoRA-Dataset-Studio-windows.zip`** from the [latest release](https://github.com/perfectgf/lora-dataset-studio/releases/latest) when that asset is present; otherwise use GitHub's **Source code (zip)**. Extract the entire archive, then double-click:

```text
start.bat
```

`start.bat` uses Python 3.10–3.12 if available. If none is installed, it downloads a self-contained CPython 3.12 into `.python\`, creates `.venv`, installs the core requirements, opens `http://127.0.0.1:5050/`, and starts the server. It requires no admin rights and changes no system PATH.

A ZIP install updates from inside the app too: **Update & restart** downloads the next **release** and swaps it in, keeping `data/`, `config.json`, `.env`, `.venv` and `.python` untouched. A git checkout follows every commit instead — and needs `git` on your PATH, which an install made through a desktop Git client does not always provide.

From a git checkout, the same launcher works and **Update & restart** can pull fixes directly:

```bash
git clone https://github.com/perfectgf/lora-dataset-studio.git
cd lora-dataset-studio
start.bat
```

### Option 2 — manual venv (any OS)

Clone/download the source, open a terminal in its root, then run:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
# optional local ML capabilities:
pip install -r backend/requirements-ml.txt
python backend/run.py
```

Only rebuild the frontend when changing `frontend/src`:

```bash
cd frontend
npm install
npm run build
```

### Option 3 — Docker + your existing ComfyUI

**Beginner Windows flow:** download/extract the **source** ZIP (GitHub ▸ **Code → Download ZIP**) — the release asset `LoRA-Dataset-Studio-windows.zip` does not carry the Docker launchers — start Docker Desktop, then double-click **`start-docker.bat`**. On the first run, select either the ComfyUI folder containing `main.py` and `models`, or its portable parent containing `ComfyUI\main.py`. LDS validates the folder and remembers it for this checkout.

Start your usual ComfyUI on the host. LDS uses `http://host.docker.internal:8188` from its container and mounts the selected folder at `/external-comfyui`. If the folder later moves, double-click **`configure-docker.bat`**. The launcher chooses a free Studio port and opens the browser automatically.

### Option 4 — Docker (GPU + ComfyUI)

**Beginner Windows flow:**

1. On GitHub, choose **Code → Download ZIP**, then extract the complete folder.
2. Start **Docker Desktop** and wait until it reports that Docker is running.
3. Double-click **`start-docker-gpu.bat`** in the extracted folder.
4. Leave the first build/start running; it downloads the image and ComfyUI environment. The launcher prints both actual addresses and opens Studio as soon as Studio responds, while its batch window stays open until ComfyUI finishes its first boot. You do not need to open a second ComfyUI window.

This creates a **fresh, isolated, repo-local** Docker setup: its own ComfyUI, models, application data and Image Bank folder live beside this checkout. **It never touches an existing ComfyUI by default.**

For either Docker launcher, choose Ollama only inside **LDS Setup**: **No Ollama**, **Existing host Ollama**, or **Docker Ollama**. The Docker companion is started only after that explicit choice, and no vision model is downloaded automatically. Pull the selected model from the LDS Ollama card to see progress and cancel it if needed.

The double-click launcher allocates free host ports atomically: Studio uses the first available port in `5050-5149`, and ComfyUI the first available port in `8188-8287`. If `5050` or `8188` is already occupied, the existing service is left running and another port is chosen automatically. Re-running the launcher from the same checkout reopens its current mapped ports without recreating the running container; a conflicting container owned by another checkout is reported and left untouched. The launcher does not edit `.env`.

Advanced CLI:

```bash
cp .env.example .env
mkdir -p run basedir data-docker-gpu bank-images
docker compose -f docker-compose.gpu.yml up --build
```

For the advanced CLI, the default addresses remain `http://127.0.0.1:5050/` for Studio and `http://127.0.0.1:8188/` for ComfyUI; `.env` can override them. This lane requires an NVIDIA GPU, a compatible driver and NVIDIA Container Toolkit support. Storage relocation, ports, existing-ComfyUI adoption, UID/GID, DNS, update commands, resource caps and operational limits are documented in the dedicated [Docker guide](docs/guide/docker.md).

### Option 4b — Docker, API-only (no GPU, any OS)

For a machine with no NVIDIA GPU: generation through Gemini/ChatGPT/OpenRouter, import and scraping, curation, manual captions, export and backup. ComfyUI and ai-toolkit stay out of this image, so local generation, Test Studio and local training are unavailable in this lane.

```bash
cp .env.example .env
mkdir -p data-docker
docker compose up --build          # docker-compose.yml, the default file
```

Studio answers on `http://127.0.0.1:5050/` and its data lives in `./data-docker`. This is the only Docker lane that needs no NVIDIA support at all.

To update any Docker install, double-click **`update-docker.bat`** (latest stable release; pass `main` for the preview channel) — it rebuilds transactionally and rolls back if the container does not come up healthy. Both `start-docker.bat` and `start-docker-gpu.bat` also accept `--rebuild` and `--update-rebuild`; `start-docker.bat` additionally accepts `--configure`, which is what `configure-docker.bat` calls.

### Option 5 — Pinokio (one click, any OS)

In [Pinokio](https://pinokio.computer), open **Discover → Download from URL** and paste `https://github.com/perfectgf/lora-dataset-studio.git`, then click **Install** and **Start**. Pinokio builds the Python environment, installs the core requirements and opens Studio; **Update** fast-forwards the same checkout the in-app updater uses.

Only the core app is installed this way — ComfyUI, Ollama, ai-toolkit and the optional ML helpers are still connected from the app's own **Setup** screen. Updates go through Pinokio's **Update** tab: because Pinokio starts and stops the server, the app detects this install shape and shows *Stop → Update → Start* instead of its own **Update & restart** button, which would relaunch the server outside Pinokio's control.

### External tools (install once, connect in Settings)

| Tool | Unlocks | Connect it |
|---|---|---|
| [ai-toolkit](https://github.com/ostris/ai-toolkit) | Local LoRA training and JoyCaption | Set its directory and Python interpreter in **Settings → Local tools**; conda, uv, venv and portable Python installs are supported |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | Klein/Krea local generation, Studio, Canvas generation and deployment; SDXL base discovery | Keep its API reachable and set the install/models paths in **Settings → Local tools** |
| [Ollama](https://ollama.com) | Auto-captioning, framing, head-crop and watermark detection | In Docker, choose none/host/companion in **Setup**, then pull the model explicitly from LDS; native installs can use their configured URL |

The full path rules, model layouts and Ollama deployment/model states are in the [settings reference](docs/guide/settings-reference.md#local-tools). If a tool remains unavailable, use the [troubleshooting guide](docs/guide/troubleshooting.md).

### Getting API keys

| Service | Used for | Where to create it |
|---|---|---|
| Gemini | Nano Banana Pro | [Google AI Studio](https://aistudio.google.com) |
| OpenAI | ChatGPT / `gpt-image-2` | [OpenAI API keys](https://platform.openai.com/api-keys) |
| OpenRouter | Image models through OpenRouter | [OpenRouter keys](https://openrouter.ai/keys) |
| Pexels | Optional official-API image search | [Pexels API key](https://www.pexels.com/api/key/) |
| Hugging Face | Gated weights and optional publishing | [Hugging Face tokens](https://huggingface.co/settings/tokens) |
| vast.ai | Optional cloud training | [vast.ai console](https://cloud.vast.ai/) |

Secrets saved in Settings live in the git-ignored `.env`, never in `config.json` or a commit. Full-model Krea 2 cloud runs use a separate `HF_CLOUD_TOKEN`; a narrowly scoped fine-grained token is recommended, while a global `role=write` token is accepted with a broad-access warning and read-only is rejected. Follow the [cloud-token instructions](docs/guide/settings-reference.md#cloud-training).

> **Pexels authorization required:** An API key alone does not authorize dataset or machine-learning use. Configure this integration only if Pexels has explicitly authorized this use case, and keep the attribution LDS displays. Read the [official Pexels terms and conditions](https://help.pexels.com/hc/en-us/articles/900005880463-What-are-the-Terms-and-Conditions/).

## Minimum requirements

The app scales from "no GPU at all" to a full local training rig — each capability has its own floor, and missing pieces are hidden or guided through Setup.

| Mode / capability | GPU (NVIDIA) | Disk | Notes |
|---|---|---|---|
| **API-only** (Gemini/ChatGPT/OpenRouter generation, import/scrape, curate, manual captions, export/backup) | none | ~2 GB | Any machine with Python 3.10+ (3.13/3.14 run the core app fine — the 3.10–3.12 window is an ML-extras constraint); Docker image available |
| **Auto-captioning & framing** (Ollama vision, 8B model) | ~8 GB VRAM | ~7 GB | Runs alongside generation, not concurrently |
| **Local generation** (Klein 9B **KV** fp8 via ComfyUI) | ~16 GB VRAM | ~30 GB (model + text encoder + VAE) | Free, local and NSFW-capable; Setup downloads the models. The KV build is up to **2.5× faster on multi-reference edits** at the same quality. Available in Docker GPU mode |
| **LoRA training — Z-Image / SDXL** (ai-toolkit) | 16 GB+ recommended | 10 GB+ free enforced per run | Quantized (qfloat8) + low-VRAM mode |
| **LoRA training — Krea 2** (ai-toolkit) | **24 GB VRAM** at 1024 px (enforced warning) | ~24 GB base download (Raw), or none if you start from a Krea 2 checkpoint you already have, + 10 GB+ free | Under 24 GB, select **Resolution → 768 only** in Advanced options |
| **LoRA training — FLUX.2 Klein** (ai-toolkit) | 4B: **16–24 GB VRAM** · 9B: **32–48 GB** | base download + 10 GB+ free | Both bases are gated on Hugging Face; the cloud lane is practical for 9B |
| **LoRA training — FLUX.1 / Anima** (ai-toolkit) | ~24 GB VRAM (both are 12B-class families) | base download + 10 GB+ free | **Local only — neither has a cloud lane.** FLUX.1 is gated on Hugging Face; Anima's base is public and reads booru tags natively |
| **Full-model (dense) training — Krea 2** (experimental, cloud only) | **80 GB VRAM** — there is no local lane | 200 GB on the pod, plus a private Hugging Face repo for the ~26 GB master | Not the recommended path: Krea's own advice is a LoRA on Raw applied to Turbo. Needs a separate `HF_CLOUD_TOKEN` |
| **Face scoring / person masks / watermark inpaint** (ML extras) | none (CPU) | ~3 GB (+ CPU torch for LaMa) | Python **3.10–3.12 required** for wheels; installable per capability from Setup |

- **OS:** Windows 10/11 for the full local stack (`start.bat`). Linux/macOS work for API-only + manual venv; GPU Docker depends on host NVIDIA support.
- **Python:** 3.10–3.12, but not required up front: `start.bat` fetches a self-contained CPython 3.12 when none is installed. Python 3.13+ can run the core app but not the ML extras.
- **RAM:** 16 GB+ recommended for local training. Unlike VRAM and free disk, this one is a recommendation the app never measures — a run that dies for want of system memory has no guard-rail in front of it.
- **Dataset size:** a launch is gated on a per-family floor — 12 images for Z-Image, 15 for Krea 2 / FLUX.1 / FLUX.2 Klein, 20 for SDXL, 4 for a slider LoRA — with 20-30 recommended. Below the floor the app asks you to confirm and warns about overfitting rather than refusing outright.
- Reference development rig: RTX 4090 (24 GB); every number above was measured or enforced there.

## Configuration & network access

Use **Settings** for normal configuration. The complete defaults, `config.json` keys, model locations and environment overrides live in [docs/guide/settings-reference.md](docs/guide/settings-reference.md).

The server binds to `127.0.0.1` by default. Before enabling LAN access or publishing a port, read [SECURITY.md](SECURITY.md#the-default-threat-model) and configure the access-token/VPN/reverse-proxy boundary that fits your network. The whole interface also works on a phone or tablet on your own network, so checking a run or triaging a bank does not need the machine that is training.

**What leaves this machine.** There is no telemetry and no analytics: nothing about you, your images or your datasets is sent anywhere. The app does reach the internet in four situations:

- **Update check** — on load and once an hour, it asks GitHub whether a newer version exists (a `git fetch` on a checkout, the releases API on a packaged install). It sends nothing about you, and there is currently **no setting to turn it off** — block the process at the firewall if you need it silent.
- **Model downloads you start** — Setup and the Install buttons stream weights from Hugging Face, Civitai, Ollama and pytorch.org. Two extras also fetch their own weights the first time you use them: the aesthetic head (~13 MB, from GitHub) and the NSFW classifier plus SigLIP 2 (Hugging Face).
- **API engines and cloud training you configure** — only the providers whose keys you entered, and only when you press the button. OpenRouter additionally receives this project's public name and repository URL as attribution headers.
- **The built-in scraper** — the sites you ask it to scan, and nothing else.

## Known limitations

Current boundaries and environment-specific caveats are tracked in [docs/guide/known-limitations.md](docs/guide/known-limitations.md).

## Troubleshooting

The symptom-first fixes — including Windows blank pages, RTX 50-series PyTorch, slow/unreachable ComfyUI and Ollama's three detection states — are in [docs/guide/troubleshooting.md](docs/guide/troubleshooting.md).

Still stuck? **Guide → Getting help** generates a paste-safe diagnostic report, then [Discord](https://discord.gg/j6hnJBFtXE) and [GitHub issues](https://github.com/perfectgf/lora-dataset-studio/issues) are the best places to share it.

## Support the project

LoRA Dataset Studio is free, open source, and has no paid tier, no telemetry and
no upsell. It is built and maintained by one person, on personal time — every
feature in the list above came out of somebody's evenings.

If the app saves you an afternoon of sorting, captioning and re-running failed
trainings, consider giving a little of that time back:

- [**Ko-fi**](https://ko-fi.com/perfectgf) — one-off, no account needed, from the price of a coffee.
- [**GitHub Sponsors**](https://github.com/sponsors/perfectgf) — one-off or monthly, and 100% reaches the project (GitHub takes no platform fee).

**Where it goes.** Not into anyone's pocket: the API credits used to test the
paid generation engines, the rented cloud GPUs used to verify the training lanes
on hardware most people actually have, and the hours that turn a working script
into something you can hand to a stranger — the docs, the guard-rails, the error
messages that tell you what to do next.

**Not able to chip in? These help just as much**, honestly:

- ⭐ **Star the repo** — it is the single biggest driver of new users finding it.
- 🐛 **Report a bug** with the app's built-in diagnostic report (Guide → Getting help). A precise report is worth more than a donation.
- 💡 **Bring an idea** to [Discord](https://discord.gg/j6hnJBFtXE) — several features shipped this year started as somebody's message there, and contributors are credited in the commit and in the app.
- 📣 **Tell someone** who is fighting with datasets by hand.

Nothing here is gated, and nothing ever will be: paying changes nothing about
what you can do with the app. It only decides how much time there is to keep
making it better.

## Legal & responsible use

> **Short version:** this software is a neutral tool. What you feed it and what you do with the result is entirely your responsibility. Some of its features can build a LoRA of a *real, identifiable person* — doing that without that person's consent may be illegal where you live, and is explicitly outside the intended use of this project.

*This section is not legal advice. Laws differ by country, state, and platform, and they change. If you are unsure whether a particular use is lawful, consult a qualified lawyer before proceeding — not this README.*

### What this project is for

LoRA Dataset Studio is intended for building datasets from imagery **you have the right to use**, specifically:

- **Yourself**, or
- **Synthetic / AI-generated people** who do not exist (the demo person shown throughout this README is one such synthetic identity), or
- **Real adults who have given you explicit, informed consent** to train and generate their likeness.

Any other use — in particular training a look-alike model of a real person from photos scraped, downloaded, or otherwise obtained without their consent — is **not** a use this project endorses or supports.

### Your responsibilities as the operator

Because the app runs entirely on your machine, under your control, **you** are the data controller and the sole party responsible for every dataset you build and every image you generate. That includes ensuring you have the necessary rights and that your use complies with all applicable law, which may include (non-exhaustively):

- **Likeness, publicity & personality rights** — many jurisdictions give people control over the commercial and non-commercial use of their face, name, and likeness.
- **Biometric-data law** — a face-recognition/similarity model of an identifiable person can constitute biometric personal data under regimes such as the EU/UK **GDPR**, Illinois **BIPA**, and similar state and national statutes, with consent and disclosure obligations attached.
- **Non-consensual intimate imagery & deepfake statutes** — a growing number of countries and U.S. states criminalize creating or sharing sexual or intimate deepfakes of real people without consent. Do not use this tool to make them.
- **Child protection law** — generating sexual or exploitative imagery of minors, real or synthetic, is a serious crime effectively everywhere. This is an absolute prohibition, without exception.
- **Copyright & platform terms** — source images may themselves be copyrighted, and scraping may violate a site's terms of service. The built-in scraper is a convenience for collecting material you are entitled to use; respect each site's terms, `robots` directives, rate limits, and the copyright of the images you download.

### Prohibited uses

Do not use this software to:

- Create a model or imagery of **any real person without their consent**;
- Produce **sexual, intimate, defamatory, harassing, or misleading** content depicting a real person without consent;
- Produce **any** sexual or exploitative content involving **minors**, real or synthetic;
- Impersonate a real person or organization, commit fraud, or otherwise deceive;
- Violate the terms of service, copyright, or rate limits of any site the scraper touches.

### No warranty & limitation of liability

This software is provided **"as is", without warranty of any kind**, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement (see the [PolyForm Noncommercial License 1.0.0](LICENSE) for the full terms). As far as the law allows, **the licensor accepts no liability** for damages — including any legal consequence arising from datasets, models, or images you create with it. By using this software you accept that responsibility yourself.

## Contributing

Issues, ideas and pull requests are welcome. For anything bigger than a small fix, say hello first — on [Discord](https://discord.gg/j6hnJBFtXE) (**#help** for questions, **#roadmap** for ideas) or in a [GitHub issue](https://github.com/perfectgf/lora-dataset-studio/issues). See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, tests, and PR conventions, and the [Code of Conduct](CODE_OF_CONDUCT.md) for how we treat each other. Found a security issue? Report it privately — see [SECURITY.md](SECURITY.md).

## License

Licensed under the **PolyForm Noncommercial License 1.0.0** — see [LICENSE](LICENSE). Noncommercial use is permitted; commercial use requires separate permission from the licensor.
