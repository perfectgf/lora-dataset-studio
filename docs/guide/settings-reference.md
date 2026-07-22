# Settings reference

Every setting in LoRA Dataset Studio, explained: what it does, its default, when to change it, and the traps to avoid.

## How settings work

Open **Settings** from the top nav. Each rail entry on the left is a section (Overview, Image engines, Scraping & sources, Local tools, Captioning & quality, Training, Server & access, Maintenance); its little LED shows live health at a glance — **green** when the section is fully configured, **amber** when it's partly set up, **off** when nothing is configured yet.

A few things hold true everywhere:

- **Nothing saves until you say so.** Change any field and a floating **Unsaved changes** bar appears with **Save** and **Discard**. Navigate away with changes pending and they're kept in the bar, not written.
- **Where values live.** Ordinary settings are written to `config.json` (git-ignored, in your data directory). Secrets — API keys and tokens — go to a separate `.env` file and are never written to `config.json` or committed.
- **Secret fields are write-only.** An API-key box is always blank, even when a key is saved (a ✓ *Configured* badge tells you it's there). Typing a new value replaces the old one; **leaving a field blank never erases a saved key** — that would be too easy to do by accident. To actually remove a key, use its **Remove** button.
- **Test buttons probe what's saved, not what's typed.** Hitting **Test** first persists whatever you've typed, then tests the *saved* setting end-to-end. So a Test result always reflects the value the app will really use.
- **Server changes need a restart.** Host, port and the access token only take effect when the server process starts. Those fields show a **Running vs Saved** comparison and a **Save & restart to apply** button. Everything else — including scraping credentials — applies immediately, no restart.
- **Search finds settings, not just sections.** The search box at the top of the page matches both section names *and* individual settings, so typing "budget" or "vision model" jumps you straight to the right field.

### Advanced: environment overrides

For containerized or scripted setups, a handful of environment variables override paths and binds before `config.json` is even read. You rarely need these — the UI covers the normal cases.

| Variable | Overrides |
|---|---|
| `LDS_DATA_DIR` | Runtime data directory (where `config.json`, datasets and trash live). |
| `LDS_CONFIG` | Path to `config.json`. |
| `LDS_ENV` | Path to the `.env` secrets file. |
| `LDS_HOST` | Bind host — takes priority over `server.host`. |
| `FLASK_DEBUG` | `1` enables Flask debug mode. |

## Overview

The Overview section has **no settings of its own** — it's the at-a-glance dashboard for the rest of the page. If nothing is configured yet, it opens with a *Let's get you set up* banner. Below that, a **Capabilities** grid marks each feature ✓ or ✗ depending on what the app can currently see (a key, a reachable tool, an installed extra), and a **Where to fix it** list links straight to the section that turns each one on. Use it as your first stop to answer "why is this feature greyed out?" — then follow the link to the section that fixes it.

## Image engines

This is where you connect the services that *generate* dataset images. The app has three engines: **Nano Banana** (Google Gemini), **ChatGPT** (`gpt-image-2`), and **Klein** (local, via ComfyUI). Klein is configured under **Local tools**; the two API engines are configured here.

### API keys

- **Gemini API key** — powers the Nano Banana engine. Paste it here and hit **Test** to confirm the key works. Get one from [aistudio.google.com](https://aistudio.google.com) → *Get API key*.
- **OpenAI API key** — powers the ChatGPT engine (`gpt-image-2`). **Test** confirms it. This key is **optional if you connect a ChatGPT subscription** below — the subscription lane can run the ChatGPT engine on your plan's image quota instead.

Both are write-only secrets: blank once saved, replaced by typing a new value, cleared only via **Remove**.

### ChatGPT subscription (experimental)

If you have a ChatGPT Plus/Pro plan, you can run the ChatGPT engine on your subscription's image quota instead of a pay-per-use API key. This uses the same sign-in lane as OpenAI's Codex CLI — it is **not a documented API and may stop working at any time**; you connect your own account at your own risk.

- **Connect with ChatGPT** — starts an OAuth device-code sign-in; the badge then shows the connected account's email.
- **Import from Codex CLI** — appears only if the app detects an existing `codex login` on this machine, and reuses that session.
- **Disconnect** — signs out of the subscription lane.
- **ChatGPT engine auth** → `engines.chatgpt_auth`. Chooses which credential the ChatGPT engine uses. Default **`auto`**.

| Value | Behaviour |
|---|---|
| `auto` *(default)* | Use the subscription when connected, otherwise fall back to the API key. |
| `api` | API key only — ignore the subscription. |
| `subscription` | Subscription only — never touch the API key. |

Good to know: in subscription mode you get up to **5 reference images** per generation (versus 16 on the API), your plan's image cap applies, and when the quota runs out mid-batch the remaining rows fail with a clear message — **the app never silently switches to your paid API key**.

### Engines

- **Default engine** → `engines.default`. Which engine is preselected in the workspace. One of `nanobanana`, `chatgpt`, `klein`. Default **`chatgpt`**.
- **Enabled engines** → `engines.enabled`. Checkboxes deciding which engines appear as options at all. Default: **all three** enabled. Untick an engine you never use to declutter the generator picker.

### Klein generation LoRA presets (optional)

*Idea from @waltm on Discord.* Named combinations of generation LoRAs that stack on top of the local Klein edit graph. Stored in `klein.generation_lora_presets` (default: empty — no presets).

Each preset has a **name** and an **ordered list of LoRAs**, and each LoRA row has:

- a **file** — a name relative to your ComfyUI `models/loras` folder (e.g. `klein/my-lora.safetensors`), exactly like the consistency LoRA. The field is a **searchable dropdown of the LoRAs actually on disk** (every folder, `extra_model_paths.yaml` included), with Klein-compatible files listed first and each one badged by architecture; free text still works for a file you haven't downloaded yet;
- a **strength** — `0`–`1.5`, default **`0.6`**.

Use **＋ New preset**, **Duplicate**, **Delete** and rename to manage them, and the up/down controls to set chain order. **Caps: 8 LoRAs per preset, 12 presets.**

How presets are used matters:

- A preset is **chosen per run** in the **🖥️ Klein tuning** panel of the workspace, and it **defaults to *None* every visit** — presets never apply on their own.
- Resolution happens **by name** on the server, and it's **fail-closed**: if a run references a preset name that no longer exists, it runs **with no extra LoRAs** rather than erroring.
- **Trap:** *renaming* a preset does **not** follow a run that referenced it by the old name — that run silently falls back to no extra LoRAs. Rename before you queue, or re-pick the preset on the run.
- There is deliberately **no automatic NSFW gating** on individual LoRAs — the preset you pick carries the intent. If you want an "NSFW full" stack, make it a preset.

### Identity & Klein prompts (advanced)

*Feature request by @bbsorry (雨田壹).* Every generated variation is prefixed by a hidden **identity lock** — a block of text that tells the engine to keep the subject's exact face and take the outfit and expression from the description, not the reference photo. These used to be baked in and invisible; now you can read and edit them. All are **global** (they apply to every dataset) and stored under `identity_prompts.*`.

**Reproducibility guarantee:** every field **defaults to blank**, and blank means *use the shipped default*. With nothing overridden, generation is **byte-identical** to before this setting existed — so you only change behaviour if you deliberately type an override. **Restore default** simply clears the field back to blank.

- **API engine — identity lock (single reference)** → `identity_prompts.face_single`. Prepended to Nano Banana / ChatGPT variations built from **one** reference photo.
- **API engine — identity lock (multiple references)** → `identity_prompts.face_multi`. The same, for variations built from **several** reference photos — it tells the model every reference is the same person and to use them together.
- **Klein — restage & face-identity block** → `identity_prompts.klein_identity`. The instruction block the local **Klein** engine uses to restage the shot (pose, framing, outfit, expression) while keeping the face identical.
- **Klein upscale & improve prompt** → `identity_prompts.klein_improve`, with an on/off toggle `identity_prompts.klein_improve_enabled` (default **on**). The fixed instruction the manual **Klein upscale & improve** action sends to add texture and detail. **Turn the toggle off** to run that action with **no prompt at all** — a pure upscale with no added styling.
- **Upscale & improve — strength** → `klein.improve_megapixels`, `klein.improve_base_lora_strength`, `klein.improve_consistency_strength`, `klein.improve_steps`. The output resolution, and how much that pass is allowed to change the image. Until these were exposed the whole profile was hardcoded — **both LoRA strengths pinned to 0**, so the *enhancement* LoRA baked into the workflow never applied at all, and the size was fixed at 2 MP whatever the source was worth. Defaults are those same historical values (**2 MP / 0 / 0 / 4 steps**), so leaving them alone keeps today's result exactly.
  - **Output size (MP)** (0.5–8, default **2**) — the source is rescaled to this pixel budget before sampling, so it *is* the result's resolution. This is the knob that makes "Upscale" actually upscale.
  - **Enhancement LoRA** (0–2, default **0**) — the workflow's own detail LoRA. At 0 it does nothing; try **0.5–0.8**. It needs its weights file (`klein/realistic.safetensors`): when that file is missing the node is skipped entirely and this value changes nothing. **Setup ▸ Install everything downloads it** with the other Klein assets (from [dx8152/Flux2-Klein-9B-Enhanced-Details](https://huggingface.co/dx8152/Flux2-Klein-9B-Enhanced-Details), Apache-2.0) — run it first if the slider seems inert.
  - **Consistency LoRA** (0–1.5, default **0**) — anchors the **composition and background**, not identity. High values resist the edit. (Shipped briefly as `improve_character_lora_strength`, a misnomer; a value saved under that name is still honoured.)
  - **Steps** (1–50, default **4**) — more steps is slower and usually cleaner.
  - Out-of-range or malformed values are **clamped**, never rejected: a bad config weakens the pass instead of failing your click.

Each field is a plain textarea; there's no Test button — you see the effect on your next generation. If an override ever makes results worse, hit **Restore default**.

## Scraping & sources

Credentials for the built-in web scraper. **All of these apply immediately — no restart** — because sources read their key at request time.

### Source credentials

None of these has a Test button; you find out they work on your next scan.

- **Reddit client ID** → `REDDIT_CLIENT_ID` (secret). Optional. Reddit scans work out of the box using a shared public client ID, but that ID is rate-limited across everyone who uses it, so you can hit *"rate limiting requests, retry in Ns"* (429) before your first scan of the day. Your own free ID gives you a private quota and clears those. **Trap:** on reddit.com/prefs/apps, create the app as type **installed app** — a *web app* or *script* comes with a client secret, and Reddit then rejects the anonymous login this app uses (every scan fails with **401**). The field has a built-in step-by-step guide.
- **Civitai API key** → `CIVITAI_API_KEY` (secret). Optional. Without it, Civitai scans return **SFW results only**; add a key to reach adult content you're entitled to use.
- **Pexels API key (required for Pexels)** → `PEXELS_API_KEY` (secret). **Required** for any Pexels search — there's no shared fallback. The free quota is **200 requests/hour and 20,000/month**. [Create one here](https://www.pexels.com/api/key/). Note the standing warning: an API key alone does **not** authorize dataset or machine-learning use — configure this only if Pexels has explicitly authorized your use case.

### Klein rescue — small scraped images

- **Small-image rescue instruction** → `klein.small_image_prompt`. An optional free-text instruction for **one flow only**: the automatic Klein **rescue** of scraped images under 768 px. Default **empty** — and empty is intentional: with nothing here the app improves from the reference image alone rather than inventing a restoration prompt on your behalf. Add an instruction only if you want to steer that rescue (e.g. "sharpen skin texture, keep natural tones"). The manual **"Klein upscale & improve"** action in the lightbox does **not** use this field — it has its own editable prompt under Settings ▸ Engines ▸ **Identity & Klein prompts** (`identity_prompts.klein_improve`), which can also be turned off for a pure upscale.

## Local tools

Where you point the app at the local programs that unlock the full pipeline: **ComfyUI** (Klein generation and Test Studio), **Ollama** (the vision model behind captioning and framing) and **ai-toolkit** (training and JoyCaption). Each card has a **Test** button that tells you immediately whether the app can see the tool.

### ComfyUI

- **ComfyUI API URL** → `comfyui.api_url`. The HTTP endpoint of your running ComfyUI. Default **`http://127.0.0.1:8188`**. **Test** confirms it answers.
- **ComfyUI install directory** → `comfyui.base_dir`. The folder that contains `models/`, `output/`, `input/`. Default **empty**. This is what lets the app scan your checkpoints and LoRAs — set the API URL alone and there's nothing to scan. If you point it at a `..._windows_portable` folder, the app auto-corrects to the `ComfyUI` sub-folder inside it. In the **Setup wizard** this field is checked as you type: a wrong, empty or missing folder gets a specific reason, and pointing at the launcher/parent folder offers the real ComfyUI inside it in one click.
- **Hugging Face token** → `HF_TOKEN` (secret, no Test button). Only needed to auto-download **license-gated** models — notably the Klein fp8 weights. Read access is enough for accepted gated models.

**Continuing without ComfyUI.** Leaving the install directory empty in the Setup wizard is a deliberate choice: it shows what turns off (local Klein generation including the NSFW lane, Klein watermark cleaning, the Test Studio, training on your own ComfyUI base models, and the on-disk LoRA preset picker) versus what stays on (scraping, curation, captioning, the API image engines, ai-toolkit/cloud training, Hugging Face publishing), then remembers the skip (`comfyui.setup_skipped`) so it stops nagging. Entering a directory at any point cancels the skip automatically and turns those features back on — the flag never hides a real problem with a ComfyUI you *have* configured.

**Models outside `models/`?** If your ComfyUI uses an `extra_model_paths.yaml` (portable builds and Stability Matrix installs commonly do), the app reads it the same way ComfyUI does, so bases that live elsewhere are found. This isn't a setting — it follows automatically from your install directory. Without such a file, nothing changes.

### Ollama

The card shows Ollama's live state and, when the binary is installed but the server isn't running, a **▶ Start Ollama** button that launches it for you — no terminal needed.

- **Ollama URL** → `ollama.url`. Where Ollama is listening. Default **`http://127.0.0.1:11434`**.
- **Ollama vision model** → `ollama.vision_model`. The vision model used for auto-captioning, framing auto-classify, head-crop and watermark detection. Default **`huihui_ai/qwen3-vl-abliterated:8b-instruct`** — the **abliterated** (uncensored) build, so it captions adult datasets instead of refusing them. **Trap:** keep the **`-instruct`** tag. The plain `:8b` tag is the *Thinking* variant, which reasons out loud instead of captioning and produces garbage here.

**Test** checks end-to-end: that Ollama is reachable *and* the configured model is actually pulled.

### ai-toolkit

- **ai-toolkit directory** → `aitoolkit.dir`. The folder containing ai-toolkit's `run.py`. Default **empty**. **Test** validates it and unlocks training + JoyCaption captioning.
- **Python interpreter (optional)** → `aitoolkit.python`. Default **empty = auto-detect** a `venv/` or `.venv/` next to `run.py`. Fill this with the full path to the right interpreter only if you installed ai-toolkit with **conda, uv or the system Python** (no venv folder for the app to find), e.g. `C:\miniconda3\envs\aitk\python.exe`.

Under **Advanced: ai-toolkit overrides**, three optional path overrides (all default empty → derived from the ai-toolkit directory):

- **Datasets directory override** → `aitoolkit.datasets_dir` (defaults to `<dir>/datasets`).
- **Output directory override** → `aitoolkit.output_dir` (defaults to `<dir>/output`).
- **Hugging Face cache override** → `aitoolkit.hf_home` (defaults to a cache under the ai-toolkit folder). Point this at an existing HF cache to avoid re-downloading base models.

## Captioning & quality

Settings for how captions are produced and how the quality tools behave.

### Captioning

- **Captioning backend** → `captioning.backend`. Which captioner writes your captions. Default **`auto`**.

| Value | Behaviour |
|---|---|
| `auto` *(default)* | Prefer JoyCaption (via ai-toolkit), fall back to the Ollama vision model. |
| `joycaption` | JoyCaption only. |
| `ollama` | Ollama vision model only. |
| `none` | No auto-captioning — you write them yourself. |

### Watermark inpainting

- **Processing device** → `watermark.device`. Where LaMa inpainting runs. Default **`auto`**. Options: `auto` (GPU when available, otherwise CPU), `cuda` (force GPU — pauses ComfyUI while cleaning), `cpu` (keep the GPU free). This only affects the **LaMa** engine.
- **Allow automatic crop** → `watermark.allow_crop`. Default **on**. When on, a watermark sitting in an outer border band is **cropped off** (a pure pixel crop — it invents nothing). Turn it **off** and such a mark is **repainted instead** (with LaMa or Klein per the chosen engine) rather than cropped. The exact same preference is editable inline in the workspace's **Clean** bar — it's one shared value, so changing it in either place changes both.

**Honest note on engine choice.** The **LaMa (fast) vs Klein (quality)** engine is *not* a Settings toggle — it's a per-batch picker in the Clean bar and a per-image choice in the review lightbox. `watermark.device` above governs LaMa only; Klein cleaning runs through ComfyUI.

### Face similarity

Two thresholds on the 0–1 face-similarity score (InsightFace), which badge each image against your reference.

- **Face score — green threshold** → `face_scoring.green`. At or above this, the image is a **strong match** (green). Default **`0.50`**.
- **Face score — orange threshold** → `face_scoring.orange`. At or above this but below green, it's **borderline** (orange); below it, red. Default **`0.45`**.

Raise them for a stricter set, lower them if good shots are being flagged too harshly.

### Image bank triage

Thresholds for the **🗃️ Bank** quality flags. Every scanned image stores its
**raw scores**, and the flags are recomputed against these values on every
read — so changing a threshold re-sorts an already-scanned bank instantly,
with **no rescan**. (The two exceptions are noted below.)

- **Sharpness minimum** → `bank.sharpness_min`. Variance of the Laplacian (the classic focus measure) under this = flagged **🌫 blurry**. Default **`100`**. Raise it to be stricter about focus, lower it if artistic soft shots get flagged.
- **Noise maximum** → `bank.noise_max`. High-frequency residual (RMS vs a Gaussian blur) over this = flagged **📺 noisy**. Default **`15`**. Heavily textured images (foliage, fabric) score high by nature — this is a flag to review, not a verdict.
- **Uniformity minimum** → `bank.uniformity_min`. Grayscale spread under this = flagged **⬜ flat** (solid colors, black frames, empty screenshots). Default **`12`**.
- **Minimum side (px)** → `bank.min_side`. Smaller image side under this = flagged **📐 small**. Default **`768`** — the same bar as the dataset import guard, because trainers only ever *downscale*.
- **Duplicate distance** → `bank.dup_distance`. How many of the 64 perceptual-hash bits two images may differ by and still be grouped as **≈ near-duplicates**. Default **`8`** (the same hash and distance the dataset import dedup uses). *Applies at the next quality scan* (groups are rebuilt then).
- **Same-person similarity** → `bank.face_threshold`. Cosine similarity at or above which two faces cluster as the same person in **👥 Group by person**. Default **`0.45`**. Raise it if different people get merged into one cluster; lower it if the same person splits into several. *Applies at the next face pass* (embeddings are cached, so re-clustering is fast).
- **Aesthetic minimum** → `bank.aesthetic_min`. LAION aesthetic score (~1–10) under which an image is flagged **💔 low aesthetic** — the "keep the nice ones" cut. Default **`5`**. Only images the **✨ Score** pass reached carry a score; an unscored image is never flagged. The score also drives "keep best" on duplicate groups (the nicest-looking copy wins).
- **NSFW maximum** → `bank.nsfw_max`. NSFW probability (0–1) over which an image is flagged **🔞 NSFW**, to split a mixed SFW/NSFW dump. Default **`0.5`**. Set by the **✨ Score** pass; a review flag, not a verdict.
- **Same-style similarity** → `bank.style_threshold`. Cosine similarity on the CLIP image embeddings at or above which two images share a visual **🎨 style** (screenshots/memes cluster apart from photoreal) in the **✨ Score** pass. Default **`0.6`**. *Applies at the next scoring pass* (embeddings are cached, so re-clustering at another threshold is fast).
- **Semantic duplicate similarity** → `bank.semantic_dup_threshold`. Cosine similarity on the *same* CLIP embeddings at or above which two scored images are grouped as a **✂ semantic near-duplicate** — a crop or re-compressed variant of the *same shot* that the perceptual-hash **≈ Duplicates** (stage 1) misses. Default **`0.96`** (much higher than the style threshold: a crop is far closer than merely "same style"). Needs the **✨ Score** pass first (it reuses those embeddings — no extra GPU work). *Re-running at another threshold re-sorts instantly* from the cached embeddings, no re-scan.

The **✨ Score** pass (aesthetic · NSFW · style) needs the **Bank scoring** extra (Setup ▸ Quality tools); **🚩 Find watermarks** reuses the vision model from **Captioning**. Both are GPU passes, serialized against training and captioning, and detection-only — the bank never edits your source files.

## Training

Defaults for new runs, plus everything about the optional cloud training lane.

### Defaults

- **Default training family** → `training.default_family`. The model family preselected when you start a new run. One of `zimage`, `sdxl`, `krea`, `flux`, `flux2klein`. Default **`zimage`**. Purely a starting point — you can switch family per run.

### Cloud GPU (vast.ai)

- **vast.ai API key** → `VAST_API_KEY` (secret). Add it to unlock **☁️ Train in cloud**. **Test** validates it (and auto-saves it first). The card includes a step-by-step guide to getting the key from [cloud.vast.ai](https://cloud.vast.ai/).

### Cloud training

Guard-rails on cost and host quality for rented pods. The card also shows a live **Spent this month** line. Everything here has a sane default — you can leave it all alone and just add the key.

| Setting | Key | Default | Range | What it does |
|---|---|---|---|---|
| **Max simultaneous cloud runs** | `cloud.max_concurrent_runs` | `1` | 1–10 | How many cloud pods may train at once. |
| **Max price per hour ($)** | `cloud.max_price_per_hour` | `0.80` | 0.1–5 | A safety cap on the hourly offer price; pricier hosts are skipped before launch. |
| **Monthly budget ($, 0 = unlimited)** | `cloud.monthly_budget_usd` | `0` | ≥0 | A hard spend ceiling for the month; new launches are **blocked** once you pass it. `0` means no limit. |
| **Stall timeout (minutes)** | `cloud.stall_timeout_minutes` | `30` | 5–240 | If no training step progresses for this long, the watchdog rescues the logs and kills the pod. |
| **Unreachable grace (minutes)** | `cloud.unreachable_grace_minutes` | `6` | 1–60 | How long a running pod may stay unreachable (a vast.ai network blackout) before the run is given up and auto-retried on a fresh host. Raise it if healthy runs die with *pod unreachable*. |
| **Min host reliability** | `cloud.min_reliability` | `0.98` | 0.9–0.999 | vast.ai reliability floor. Lowering toward 0.95 surfaces cheaper hosts at a higher boot-failure risk. |
| **Verified hosts only** | `cloud.verified_only` | **on** | toggle | Restrict to vast.ai's verified hosts (the historical, safer behaviour). |
| **Secure Cloud only** | `cloud.secure_cloud_only` | **off** | toggle | Restrict to vast.ai's *datacenter* (Secure Cloud) tier — usually narrows the market and raises the price, so it's opt-in. |

### Advanced options (per run)

These live under **⚙️ Advanced options** in a dataset's training panel — rank, resolution, save/sample cadence, optimizer, scheduler, EMA, LoKr and more. Each carries its own inline **Why/How** note, so they aren't repeated here. One is worth calling out because of a caveat:

- **Dual captions (long + short)** — off by default. When on, the run uses ai-toolkit's native `short_and_long_captions`: every image trains with **both** its full caption and a short one (text-side augmentation, so the LoRA leans less on any single wording). The short variant is **derived from the long caption** the next time you (re-)caption — text-only, via the local vision model, honouring the same kind rules (no trigger; the identity/concept/aesthetic stays omitted) — and you can edit it per image in the **⛶** caption editor. **Local training only for now:** the cloud pod's dataset upload doesn't carry the JSON caption file the short is read from, so cloud runs train on the long caption alone.

## Server & access

How the app binds and who can reach it. **These are the settings that need a restart** — the card shows a **Running vs Saved** banner and a **Save & restart to apply** button that does it in one click.

- **Port** → `server.port`. The port the app listens on. Default **`5050`**. Change it if something else owns the port (on macOS, port 5000 is taken by AirPlay Receiver).
- **Available on the local network** — a toggle that flips the bind host between `127.0.0.1` (this machine only, the default) and `0.0.0.0` (reachable from your LAN — phone, tablet, another PC). The token and phone controls below only appear once this is on.
- **Require an access token** → `server.require_token`. Default **off** — a home LAN is treated as trusted, so LAN access is open and there's no token to type on a phone. Turn it **on** to demand a token from remote devices; requests from localhost never need one.
- **Access token** → `server.access_token`. Shown only when the token gate is on: a read-only field with **Generate new token** and **Copy**. It's persisted, so it survives restarts. Open `http://<machine>:<port>/?token=<token>` once from the remote device and a signed session cookie takes over.
- **Open it on your phone** — a card with a scannable **QR code** and copyable URLs built from this machine's real LAN IP (and Tailscale IP, if present). No guessing which address to type.

**Trap:** if you launched via `start.bat` with `LDS_PORT` set, that variable can override the port in your config. The in-app **Save & restart** pins host and port for the relaunch, precisely so the restart lands on the port you chose rather than the one the script forced.

## Maintenance

Housekeeping and diagnostics. Only one true setting lives here; the rest are actions.

- **Updates** — **Check for updates** and **Update & restart**, plus a *see what's in this update* compare link. **The button adapts to how you installed.** A **git checkout** fast-forwards to the latest commits. A **packaged (ZIP) install** announces the release and its size (*Update to vX — download ~XX MB*) and shows a **live progress bar** while it downloads and installs (a release ZIP is far larger than a git pull), then backs up the current files and swaps in the new ones — keeping `data/`, `config.json`, `.env` and your `.venv` untouched — and restarts. A mid-way failure rolls back automatically, so a broken download never leaves you with a half-updated install. If the app can't identify a downloadable release (no ZIP asset, or offline), the button steps aside and links to the releases page instead of promising an update it can't perform.
- **Trash** — **Open folder** and **Empty trash**. Everything the app deletes goes here first; emptying is the one destructive action, and it asks for confirmation.
- **Back up everything** — not on this page but on the **Datasets library**: one button archives every dataset, its **training history** and your settings into a single file (⬇ download or 📂 open folder), and the library's **Import backup** restores it — datasets come back under **Trained**, not "Not trained yet". Tick **Include trained LoRAs** to bundle the (large) trained `.safetensors` too. **API keys and tokens are never included** — re-enter them on the new install. See *Using the app → Back up everything*.
- **Dataset images root** → `paths.dataset_images_root`. Where dataset images are stored. Default **empty → `<data dir>/datasets`**. Point it at a bigger or faster drive if your default data directory is tight on space.
- **Diagnostic report** — a one-click, **paste-safe** report for bug reports: it carries the version, capability status and a log tail, with **no secrets** and file paths reduced to booleans (present/absent). Safe to drop into Discord or a GitHub issue.
- **Server log** — a live tail of the server log, with **Copy all**, for when you need to see what just happened.

## Per-dataset settings

Separate from everything above: these live **per dataset**, in the **⚙ Dataset settings** modal you open from the workspace. They travel with that one dataset and don't touch the global Settings page.

- **Name** — the dataset's display name. **Display only** for Character and Concept datasets: it never appears in a file name, so changing it touches nothing on disk (the *trigger word* names produced files — see below). **On a Style dataset it means more**: a Style is always-on and has no visible trigger, so its name is its only editable identity — renaming it also renames the LoRAs, run folder and export it already produced (and is refused while a run is live, same as a trigger change).
- **Dataset kind** *(🧑 Character / 💡 Concept / 🎨 Style)* — the nature of the LoRA, chosen at creation but changeable here. It is the disruptive setting, so picking a different pill reveals a confirmation block that spells out **what changes** and **what is kept** before you save:
  - *What changes* — the **caption strategy** (Character leaves out identity; Concept leaves out the recurring concept; Style leaves out the aesthetic), which **panels show** (Reference photo, Generate variations and Face analysis are Character-only — they appear when you become a Character and are hidden otherwise), the **trigger's role** (Style has none; switching to Character/Concept brings the field back, prefilled), and Character-only settings such as **face/body fidelity**. Switching **to Concept** requires a concept description.
  - *What is kept* — **nothing is deleted.** Every image, its caption text, keep/reject status, face scores, watermark work and **training history** stay exactly as they are (past runs are named by the model family and trigger, never the kind). A concept description is remembered so switching back restores it.
  - Existing captions were written for the **old** kind and are **not** rewritten automatically — use **🔄 Re-caption** in the Captions section to apply the new strategy. The switch is refused while the dataset has work in progress (generation, captioning or a quality pass) — wait for it to finish.
- **Trigger word** — the word you put in prompts to summon this LoRA (Character and Concept datasets). Safe to change anytime — it's added at export, so existing captions don't need redoing. It is also **the name everything this dataset produces carries** (the deployed LoRA, the training run folder, the export, the job config), so changing it **renames all of them to match** and repoints the Test Studio history and cloud runs at the new names — a toast tells you how many files moved. Two guards: if the new trigger is already used on disk by another dataset, **nothing** is renamed (never half a set) and the old names are kept; and the change is **refused while a training run is live**, because that run folder is what training resumes from — stop it or let it finish first. **Style datasets don't have one**: Style is always-on, and the modal shows a note reminding you to control the effect with the LoRA weight instead.
- **Concept description** *(Concept datasets only)* — the thing the LoRA learns, i.e. exactly what captions must **omit**. Editing it rebuilds the caption avoid-list, so **re-caption** afterwards to apply the new list to images already captioned.
- **Prompt suffixes** *(collapsible — optional creative direction)* — free text appended to **generated** variations at generation time, to steer a global look without rewriting anything:
  - **All shots** → `prompt_suffix` — one global suffix (e.g. *"shot on 35mm film, warm tones"*), up to **300 characters**.
  - **Face / Bust / Body / Back shots** → `prompt_suffixes` — one suffix per framing, up to **300 characters** each. A framing suffix applies to that shot type first, then the global one.

  Key behaviours: these are **applied at generation time and never stored into a tile's own prompt** (so a regenerate can't double-apply them), the **identity lock always comes first** — a suffix can't override it, clearing a field removes that suffix, and existing images stay as they are until you **regenerate** to apply.

  You can also edit the very same suffixes **inline in the generation panel** (the collapsible *✨ Prompt suffixes* row under the shot picker), which is handy for tuning them **per batch** without opening this modal — both surfaces read and write the one dataset value, and an edit made there is saved the moment you press **Generate**.

## Config-file-only settings

These have no UI control — they're for advanced users editing `config.json` by hand (copy `config.example.json` to `config.json` first). Most people never touch them; the defaults are tuned. Values below are the shipped defaults.

**ComfyUI folder overrides** — explicit paths that override the folders otherwise derived from `comfyui.base_dir`:

| Key | Default | Role |
|---|---|---|
| `comfyui.output_dir` | `''` | Override ComfyUI's output folder. |
| `comfyui.input_dir` | `''` | Override ComfyUI's input folder. |
| `comfyui.models_dir` | `''` | Override the models folder scanned for checkpoints/UNETs. |
| `comfyui.loras_dir` | `''` | Override the LoRA folder. |

**Engines:**

| Key | Default | Role |
|---|---|---|
| `engines.chatgpt_subscription_model` | `gpt-5.4-mini` | The Codex **router** model used by the subscription lane. The image model stays `gpt-image-2` regardless — this is not the image model. |

**Cloud (vast.ai) internals** — knobs for after the real-world smoke test; the UI-exposed cloud settings above are the ones you'll normally want:

| Key | Default | Role |
|---|---|---|
| `cloud.template_hash` | `471ed5903d8cdb8e63b0d0e50f6cd519` | The official vast.ai "Ostris AI Toolkit" template. Clearing it falls back to a raw-image launch. |
| `cloud.ui_port` | `18675` | Container port the pod UI is proxied on. |
| `cloud.image` | `vastai/ostris-ai-toolkit:…` | Raw-image fallback (used only when the template is cleared). |
| `cloud.offer_scan_limit` | `100` | How many offers are fetched when listing GPU speed tiers. |
| `cloud.pod_overhead_minutes` | `35` | Boot + model download + quantize time built into cost estimates. |
| `cloud.min_inet_down_mbps` | `400` | Skip hosts too slow to pull the image. |
| `cloud.min_disk_bw_mbps` | `500` | Skip hosts too slow to extract it. |
| `cloud.host_blacklist_days` | `3` | How long to skip a host whose pod never became ready. |
| `cloud.ready_timeout_minutes` | `25` | Boot budget: image pull + services up. |
| `cloud.max_runtime_minutes` | `480` | Hard stop past this (the stall watchdog is the first line of defence). |
| `cloud.disk_gb` | `60` | Instance disk (base model + dataset + checkpoints). |
| `cloud.min_vram_gb` | `{zimage:24, sdxl:16, krea:24, flux2klein:32}` | Minimum VRAM **per family**. flux2klein uses 32 (the 9B is the cloud-first lane; a 32 GB pod also trains the 4B). |
| `cloud.onstart` | `''` | Optional startup command for the raw-image fallback. |

**Quality-tool interpreters and models:**

| Key | Default | Role |
|---|---|---|
| `face_scoring.python` | `''` | Interpreter for the InsightFace subprocess (empty = current interpreter). |
| `face_scoring.models_root` | `''` | Where InsightFace weights are stored/downloaded. |
| `face_scoring.device` | `'auto'` | Device for the Image-bank face pass. `auto` uses the GPU when the face interpreter exposes CUDA (needs `onnxruntime-gpu` installed in it) and falls back to CPU otherwise; `cpu` forces CPU (never touches the GPU); `cuda` requests the GPU but still falls back to CPU when unavailable. A GPU run is serialized through the GPU-exclusive window so it never competes with a training/scoring pass. |
| `masks.python` | `''` | Interpreter for the rembg (person-mask) subprocess. |
| `watermark.python` | `''` | Interpreter for the LaMa watermark subprocess. **Auto-managed:** leave it empty and the **Install inpainting** button builds a dedicated Python 3.10-3.12 environment for you (`simple-lama-inpainting` needs Pillow&lt;10, so it can't share the app's own Python) and fills this in automatically. Set it yourself only to point at an environment you already have — a manual value is always respected and never overwritten. |

**Klein consistency LoRA:**

| Key | Default | Role |
|---|---|---|
| `klein.consistency_lora` | `klein/Flux2-Klein-9B-consistency-V2.safetensors` | The structure-anchoring LoRA on the Klein edit graph, relative to ComfyUI's LoRA folder. |
| `klein.consistency_strength` | `0.5` | Its strength (0–1). Its own guide warns 0.8–1.0 can stop edits applying; `0` disables it entirely. |

**Updates:**

| Key | Default | Role |
|---|---|---|
| `updates.repo` | `perfectgf/lora-dataset-studio` | The GitHub repo the update checker reads its release feed from. |

## config.json key reference (all keys)

A flat cheat-sheet of the main `config.json` keys, for quick lookup or hand-editing (copy `config.example.json` to `config.json` first — it's git-ignored, in your data directory). Every key here is documented in full, with defaults and traps, in the sections above; this table is the index. **Secrets** (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `HF_TOKEN`, `VAST_API_KEY`, optional scraper credentials) live in `.env`, not here.

| Key | Meaning |
|---|---|
| `server.host` | Interface the Flask server binds to (default `127.0.0.1`, local-only). |
| `server.port` | Port the server listens on (default `5050`). |
| `server.require_token` | On a non-loopback bind, require remote clients to present an access token (default `false` — a trusted LAN needs none). Toggle and token also live in Settings → Server & access. |
| `paths.dataset_images_root` | Where dataset images are stored. Empty string defaults to `<data dir>/datasets`. |
| `comfyui.api_url` | Base URL of your ComfyUI instance (default `http://127.0.0.1:8188`). |
| `comfyui.base_dir` | ComfyUI install directory, used to derive `output`/`input`/`models`/`loras` dirs if those aren't set explicitly. |
| `comfyui.output_dir` | Explicit override for ComfyUI's output folder. |
| `comfyui.input_dir` | Explicit override for ComfyUI's input folder. |
| `comfyui.models_dir` | Explicit override for ComfyUI's models folder (used to scan available checkpoints/UNETs). |
| `comfyui.loras_dir` | Explicit override for ComfyUI's LoRA folder. |
| `ollama.url` | Base URL of your Ollama instance (default `http://127.0.0.1:11434`). |
| `ollama.vision_model` | Ollama vision model used for auto-classify and auto head-crop (default `huihui_ai/qwen3-vl-abliterated:8b-instruct`, the uncensored **abliterated** build — use the Instruct, not Thinking, variant). |
| `aitoolkit.dir` | ai-toolkit install directory. |
| `aitoolkit.datasets_dir` | Override for ai-toolkit's datasets folder (defaults to `<aitoolkit.dir>/datasets`). |
| `aitoolkit.output_dir` | Override for ai-toolkit's output folder (defaults to `<aitoolkit.dir>/output`). |
| `aitoolkit.hf_home` | Override for the Hugging Face cache directory ai-toolkit uses. |
| `aitoolkit.python` | Full path to the Python interpreter to run ai-toolkit with. Empty = auto-detect a `venv/`/`.venv/` next to `run.py`; set it for conda/uv/system-Python installs that have no venv folder. |
| `engines.default` | Default image-generation engine selected in the UI (`nanobanana`, `chatgpt`, or `klein`). |
| `engines.enabled` | List of engines shown as options in the UI. |
| `engines.chatgpt_auth` | Which credential the ChatGPT engine uses: `auto` (subscription when connected, else API key), `api`, or `subscription`. |
| `engines.chatgpt_subscription_model` | Codex **router** model for the subscription lane (default `gpt-5.4-mini`); the image model stays `gpt-image-2` regardless. |
| `captioning.backend` | Caption backend: `auto` (prefer JoyCaption, fall back to Ollama), `joycaption`, `ollama`, or `none`. |
| `training.default_family` | Default model family preselected for new training runs (`zimage`, `sdxl`, `krea`, `flux`, or `flux2klein`). |
| `cloud.max_concurrent_runs` | Simultaneous cloud pods allowed (default `1`, 1–10). Also in Settings → Training. |
| `cloud.max_price_per_hour` | Safety cap on the hourly offer price in $ (default `0.80`); pricier hosts are skipped before launch. |
| `cloud.monthly_budget_usd` | Hard monthly spend ceiling in $ (default `0` = unlimited); launches are blocked past it. |
| `cloud.stall_timeout_minutes` | Kill + rescue a cloud run after this many minutes without step progress (default `30`, 5–240). |
| `cloud.min_reliability` | vast.ai host-reliability floor (default `0.98`, 0.9–0.999); lower surfaces cheaper, riskier hosts. |
| `cloud.verified_only` | Restrict to vast.ai verified hosts (default `true`). |
| `cloud.secure_cloud_only` | Restrict to vast.ai's Secure Cloud (datacenter) tier (default `false`; narrows the market, raises price). |
| `face_scoring.python` | Python interpreter used to run the InsightFace subprocess (empty = current interpreter). |
| `face_scoring.models_root` | Directory where InsightFace model weights are stored/downloaded. |
| `face_scoring.green` | Similarity score threshold (0–1) above which an image is flagged "green" (strong match). |
| `face_scoring.orange` | Similarity score threshold (0–1) above which an image is flagged "orange" (borderline match). |
| `masks.python` | Python interpreter used to run the rembg subprocess (empty = current interpreter). |
| `watermark.python` | Python interpreter used to run the LaMa watermark-inpainting subprocess (empty = reuse `masks.python`, then the current interpreter). |
| `watermark.device` | LaMa processing device: `auto` (CUDA when available, otherwise CPU), `cuda`, or `cpu`. |
| `watermark.allow_crop` | When `true` (default), a border watermark is cropped off; when `false`, it is repainted instead. Also editable in the Clean bar. |
| `klein.consistency_lora` | Filename of the Klein consistency LoRA, relative to ComfyUI's LoRA folder. |
| `klein.consistency_strength` | Strength (0–1) applied to the Klein consistency LoRA. |
| `klein.generation_lora_presets` | Named generation-LoRA stacks (default empty) picked per run in Klein tuning; each has a name and up to 8 `{file, strength}` rows. Managed in Settings → Image engines. |
| `klein.small_image_prompt` | Optional shared instruction for scraper rescue and single/bulk image improvement (empty = reference image only). |
| `updates.repo` | GitHub repo the update checker reads its release feed from (default `perfectgf/lora-dataset-studio`). |

Additional config-file-only keys (ComfyUI folder overrides, cloud internals, quality-tool interpreters, Klein consistency LoRA) are documented in [Config-file-only settings](#config-file-only-settings) above.
