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

- a **file** — a name relative to your ComfyUI `models/loras` folder (e.g. `klein/my-lora.safetensors`), exactly like the consistency LoRA;
- a **strength** — `0`–`1.5`, default **`0.6`**.

Use **＋ New preset**, **Duplicate**, **Delete** and rename to manage them, and the up/down controls to set chain order. **Caps: 8 LoRAs per preset, 12 presets.**

How presets are used matters:

- A preset is **chosen per run** in the **🖥️ Klein tuning** panel of the workspace, and it **defaults to *None* every visit** — presets never apply on their own.
- Resolution happens **by name** on the server, and it's **fail-closed**: if a run references a preset name that no longer exists, it runs **with no extra LoRAs** rather than erroring.
- **Trap:** *renaming* a preset does **not** follow a run that referenced it by the old name — that run silently falls back to no extra LoRAs. Rename before you queue, or re-pick the preset on the run.
- There is deliberately **no automatic NSFW gating** on individual LoRAs — the preset you pick carries the intent. If you want an "NSFW full" stack, make it a preset.

## Scraping & sources

Credentials for the built-in web scraper. **All of these apply immediately — no restart** — because sources read their key at request time.

### Source credentials

None of these has a Test button; you find out they work on your next scan.

- **Reddit client ID** → `REDDIT_CLIENT_ID` (secret). Optional. Reddit scans work out of the box using a shared public client ID, but that ID is rate-limited across everyone who uses it, so you can hit *"rate limiting requests, retry in Ns"* (429) before your first scan of the day. Your own free ID gives you a private quota and clears those. **Trap:** on reddit.com/prefs/apps, create the app as type **installed app** — a *web app* or *script* comes with a client secret, and Reddit then rejects the anonymous login this app uses (every scan fails with **401**). The field has a built-in step-by-step guide.
- **Civitai API key** → `CIVITAI_API_KEY` (secret). Optional. Without it, Civitai scans return **SFW results only**; add a key to reach adult content you're entitled to use.
- **Pexels API key (required for Pexels)** → `PEXELS_API_KEY` (secret). **Required** for any Pexels search — there's no shared fallback. The free quota is **200 requests/hour and 20,000/month**. [Create one here](https://www.pexels.com/api/key/). Note the standing warning: an API key alone does **not** authorize dataset or machine-learning use — configure this only if Pexels has explicitly authorized your use case.

### Klein image improvement

- **Klein instruction** → `klein.small_image_prompt`. An optional free-text instruction shared by two flows: the automatic Klein **rescue** of scraped images under 768 px, and the manual **single/bulk 2 MP improvement**. Default **empty** — and empty is intentional: with nothing here the app improves from the reference image alone rather than inventing a restoration prompt on your behalf. Add an instruction only if you want to steer that improvement (e.g. "sharpen skin texture, keep natural tones").

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
| **Min host reliability** | `cloud.min_reliability` | `0.98` | 0.9–0.999 | vast.ai reliability floor. Lowering toward 0.95 surfaces cheaper hosts at a higher boot-failure risk. |
| **Verified hosts only** | `cloud.verified_only` | **on** | toggle | Restrict to vast.ai's verified hosts (the historical, safer behaviour). |
| **Secure Cloud only** | `cloud.secure_cloud_only` | **off** | toggle | Restrict to vast.ai's *datacenter* (Secure Cloud) tier — usually narrows the market and raises the price, so it's opt-in. |

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

- **Updates** — **Check for updates** and, on a git checkout, **Update & restart**, plus a *see what's in this update* compare link.
- **Trash** — **Open folder** and **Empty trash**. Everything the app deletes goes here first; emptying is the one destructive action, and it asks for confirmation.
- **Dataset images root** → `paths.dataset_images_root`. Where dataset images are stored. Default **empty → `<data dir>/datasets`**. Point it at a bigger or faster drive if your default data directory is tight on space.
- **Diagnostic report** — a one-click, **paste-safe** report for bug reports: it carries the version, capability status and a log tail, with **no secrets** and file paths reduced to booleans (present/absent). Safe to drop into Discord or a GitHub issue.
- **Server log** — a live tail of the server log, with **Copy all**, for when you need to see what just happened.

## Per-dataset settings

Separate from everything above: these live **per dataset**, in the **⚙ Dataset settings** modal you open from the workspace. They travel with that one dataset and don't touch the global Settings page.

- **Name** — the dataset's display name.
- **Trigger word** — the word you put in prompts to summon this LoRA (Character and Concept datasets). Safe to change anytime — it's added at export, so existing captions don't need redoing. **Style datasets don't have one**: Style is always-on, and the modal shows a note reminding you to control the effect with the LoRA weight instead.
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
