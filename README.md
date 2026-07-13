# LoRA Dataset Studio

[![Join our Discord](https://img.shields.io/discord/1525908170331914411?logo=discord&logoColor=white&label=Discord&color=5865F2)](https://discord.gg/j6hnJBFtXE)

**Turn one reference photo into a trained, ranked LoRA — curation, captioning, face-scoring and training behind a single browser tab, on your own machine.**

The useful part of LoRA training isn't the training — it's building a clean, balanced, well-captioned image set. That job is normally scattered across a scraper, an image editor, a captioning script, and a training config someone hand-tunes per run. LoRA Dataset Studio puts the whole pipeline behind one UI: generate variations from a reference photo, curate them against a live composition meter, caption them automatically, score them for face fidelity, train the LoRA, and rank the resulting checkpoints — without leaving the page.

<p align="center">
  <img src="docs/screenshots/03-curate.png" alt="Curation grid: framing badges, face-similarity scores, per-image captions, keep/reject" width="900">
</p>
<p align="center"><em>The curation grid — every image tagged by framing (face / bust / body), scored against the reference face, captioned, and one click from keep or reject.<br>All screenshots in this README use a synthetic, AI-generated demo person — no real individual is depicted.</em></p>

---

## Everything it does, at a glance

The whole pipeline, grouped by stage — every item links to the section that details it.

| Stage | What you get |
| :-- | :-- |
| 🏗️ **Build** | 🎭 **[3 dataset types](#1-three-dataset-types-character--concept--style)** — character, concept or style; each rewires captioning, masking and step-scaling to match.<br>🖼️ **[3 image sources](#2-three-ways-to-source-images)** — generate from a reference photo, import your own, or scrape the web.<br>🧭 **[Guided workspace](#3-the-guided-workspace)** — a progress rail unlocks each step and shows what's blocking Train.<br>✏️ **[Edit & regenerate](#7-edit-the-prompt-regenerate-the-shot)** — tweak any tile's prompt in place and re-shoot it, identity preserved. |
| 🎯 **Curate & caption** | 📐 **[Auto-framing + meter](#5-auto-framing-classification)** — auto-tags each shot face/bust/body and scores the set against a 12/6/6/1 target.<br>👤 **[Face scoring](#4-face-similarity-scoring)** — InsightFace flags off-identity shots before they poison training.<br>📝 **[Model-matched captions](#6-captioning-that-matches-the-model)** — prose or booru tags, picked for the model and written by JoyCaption or Ollama. |
| 🎓 **Train** | 🎛️ **[No-hand-tune training](#8-training-you-dont-hand-tune)** — click Train: adaptive steps, a GPU queue and auto rembg masks, no config file.<br>🧬 **[5 model families](#8-training-you-dont-hand-tune)** — Z-Image, SDXL, Krea 2, FLUX.1 and FLUX.2 Klein, presets built in.<br>📑 **[Training presets](#8-training-you-dont-hand-tune)** — save named recipes (3 ship read-only), import/export as shareable JSON.<br>☁️ **[Cloud training](#cloud-training-vastai--experimental)** — no GPU? rent a vast.ai pod (~$1–2/run) with retry and continue.<br>🏋️ **[Runs hub](#8-training-you-dont-hand-tune)** — cloud and local runs in one tab: live progress, checkpoint trash and cap. |
| 🚀 **Test & ship** | 🧪 **[Test Studio](#9-test-studio--pick-the-best-checkpoint)** — grid-test checkpoint × strength, vote, and rank epochs by face match.<br>📦 **[Export ZIP](#10-export)** — leave with image + `.txt` caption pairs that train in any ai-toolkit. |
| 🌐 **Comfort & access** | 📱 **[Phone access](#exposing-the-app-beyond-localhost)** — scan a QR to open the app on your phone over LAN or Tailscale.<br>🧰 **[Setup wizard](#setup--install)** — scans your machine and installs only what's missing.<br>📖 **[Guide + diagnostics](#troubleshooting)** — a 5-chapter in-app manual and a one-click, paste-safe diagnostic report. |

---

## Table of contents

- [Everything it does, at a glance](#everything-it-does-at-a-glance)
- [How it works, in one pass](#how-it-works-in-one-pass)
- [Features, one at a time](#features-one-at-a-time)
  - [1. Three dataset types](#1-three-dataset-types-character--concept--style)
  - [2. Three ways to source images](#2-three-ways-to-source-images)
  - [3. The guided workspace](#3-the-guided-workspace)
  - [4. Face-similarity scoring](#4-face-similarity-scoring)
  - [5. Auto-framing classification](#5-auto-framing-classification)
  - [6. Captioning that matches the model](#6-captioning-that-matches-the-model)
  - [7. Edit the prompt, regenerate the shot](#7-edit-the-prompt-regenerate-the-shot)
  - [8. Training you don't hand-tune](#8-training-you-dont-hand-tune)
  - [9. Test Studio — pick the best checkpoint](#9-test-studio--pick-the-best-checkpoint)
  - [10. Export](#10-export)
- [Why this instead of driving ai-toolkit directly?](#why-this-instead-of-driving-ai-toolkit-directly)
- [Feature matrix by backend](#feature-matrix-by-backend)
- [Two run modes](#two-run-modes)
- [Cloud training (vast.ai)](#cloud-training-vastai--experimental)
- [Setup & install](#setup--install)
- [Minimum requirements](#minimum-requirements)
- [Configuration reference](#configuration-reference)
- [Exposing the app beyond localhost](#exposing-the-app-beyond-localhost)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Legal & responsible use](#legal--responsible-use)
- [License](#license)

---

## How it works, in one pass

The app is a **guided flow**: each stage stays folded until the one before it is done, and a progress rail tells you exactly where you are and what's blocking the next step. The character path looks like this (a **concept** or **style** dataset skips the reference photo and sources images by import/scrape instead):

<p align="center">
  <img src="docs/screenshots/02-workspace.png" alt="Guided dataset workspace with a progress rail and the training panel expanded" width="820">
</p>
<p align="center"><em>The workspace walks you from reference photo → generate → curate → caption → train, one unlocked step at a time.</em></p>

1. **Create a dataset** — pick the type, name it, set a trigger word.
2. **Upload a reference photo** (+ up to 3 extra angles for multi-view consistency).
3. **Generate variations** via Nano Banana Pro (Gemini), ChatGPT (`gpt-image-2`), or Klein (local ComfyUI).
4. **Import** with automatic head-crop.
5. **Auto-classify framing** (face / bust / body / back) via a local vision model.
6. **Curate** — keep / reject / crop, guided by a live meter targeting **12 face · 6 bust · 6 body · 1 back**.
7. **Caption** — prose for Z-Image, booru tags for SDXL, generated for you.
8. **Score face similarity** against the reference (InsightFace, green/orange thresholds).
9. **Generate person masks** (rembg) for masked training.
10. **Train a LoRA** via ai-toolkit — adaptive step counts, a queue, scheduling.
11. **Test Studio** — grid-test checkpoint × strength, vote, and rank checkpoints by face similarity.
12. **Export** the curated, captioned set as a ZIP.

> 📖 **New here?** The **Guide** tab inside the app is a 5-chapter manual: getting started, day-to-day usage, dataset quality (also readable as [docs/DATASET_GUIDE.md](docs/DATASET_GUIDE.md)), troubleshooting, and how to report problems — with a one-click diagnostic report. The chapters live in [docs/guide/](docs/guide/) if you prefer reading on GitHub.

---

## Features, one at a time

### 1. Three dataset types (Character · Concept · Style)

One shared rule runs through all three — *what you caption stays promptable, what you omit gets absorbed into the trigger* — but each type flips the machinery to match what you're actually teaching.

<p align="center">
  <img src="docs/screenshots/01-create.png" alt="New-dataset panel: Character / Concept / Style tabs, name, trigger word, target model, fidelity" width="820">
</p>
<p align="center"><em>Pick the type and the app reconfigures captioning, masking, and step-scaling behind the scenes.</em></p>

- **Character** — pin an identity from one reference photo. The app fans out a **45-shot variation catalog** (expression / angle / lighting / framing / outfit / background) so the set spans close-up to full-body without you writing a single prompt.
- **Concept** — train an *object or action* instead of a person. Captioning **inverts**: it describes everything *except* the concept (with an identity-leak check), so the concept is what binds to the trigger — and masked training turns itself off so it can't erase what you're teaching.
- **Style** — train a *global aesthetic* that tints every image once the LoRA is loaded. Captions describe **content only** (never the rendering), there is **no trigger word** in the training config, caption dropout rises to 30%, and the step count switches to a **sublinear √n scale** built for the large (hundreds-of-images) sets style LoRAs want. Captions are optional.

### 2. Three ways to source images

- **Generate** — from your reference photo, through Nano Banana Pro, ChatGPT (`gpt-image-2`), or a local Klein/ComfyUI model. An identity guard is wrapped around every request so the face stays *the same person* across expressions, angles, and lighting.
- **Import** — drag in your own photos; each one is auto-cropped to the face on the way in (or centered-cropped if no vision model is available).
- **Scrape** — collect real images from the web straight into a concept dataset. This is its own panel, covered next.

#### Using a ChatGPT subscription instead of an API key (experimental)

If you have a ChatGPT Plus/Pro subscription you can run the ChatGPT engine on your plan's image quota instead of a pay-per-use API key: **Settings → ChatGPT subscription → Connect with ChatGPT** (or **Import from Codex CLI** if you already use `codex login`).

Good to know:

- **Experimental.** This uses the same subscription lane as OpenAI's Codex sign-in. It is not a documented API and may stop working at any time; you connect your own account at your own risk. The API-key mode is unaffected.
- **Limits vs API mode:** up to 5 reference images per generation (instead of 16), and your plan's image cap applies. When the quota runs out mid-batch, the remaining rows fail with a clear message — the app never silently switches to your paid API key.
- Auth mode is configurable (**Settings → ChatGPT engine auth**): Auto (subscription when connected, otherwise API key), API key only, or Subscription only.

#### Built-in web scraper

Concept and style LoRAs learn from *real* images, so those datasets swap the face tooling for a scraper. Paste an **image-gallery / album URL**, or run a **Reddit keyword search** — with an optional community (subreddit) scope for cleaner, on-topic results — and the app turns the results into a pick-and-import grid. Tick the frames you want and they download **directly into this dataset**; nothing touches a shared pool.

<p align="center">
  <img src="docs/screenshots/06-scraper.png" alt="Scraper panel: gallery URL field, Reddit keyword + subreddit search, Scan and Import" width="900">
</p>
<p align="center"><em>Scrape a gallery URL or search Reddit by keyword (optionally scoped to a community), then pick frames straight into the dataset.</em></p>

What it does on your behalf:

- **SSRF-hardened** — the fetcher refuses internal/loopback/link-local targets, so a hostile URL can't turn the scraper into a request proxy into your network.
- **Perceptual de-duplication** — near-identical frames are dropped so the same shot doesn't get counted five times.
- **Quality filters at import** — anything under 768px on the short side, or wider than a 3:1 ratio, is rejected before it lands.
- **Dead-link hygiene** — source links whose thumbnails fail to load are hidden from the grid, so you only ever pick live images.
- **Sensible guidance baked in** — the panel nudges you toward 20–50 varied images, at most ~10 per gallery (one gallery ≈ one shoot), which is what actually trains well.

Some sources take **optional credentials** in **Settings → Scraping & sources**: your own free **Reddit client ID** (the built-in shared one is rate-limited — a personal id gives you a private quota and clears the "retry in Ns" 429s) and a **Civitai API key** (Civitai scans return SFW results only without one). Everything works without them — they just lift per-source limits.

The scraper can reach adult communities as well — this is an NSFW-capable tool — so use it only for material you have the right to train on. See [Legal & responsible use](#legal--responsible-use). The scraping extras (`gallery-dl`, `curl_cffi`, …) install with one click from the panel when they're missing.

### 3. The guided workspace

The composition meter is the quiet workhorse: as you keep and reject, it tracks your framing mix against the **12 / 6 / 6 / 1** target and tells you what the set is still missing (*"needs more full-body shots"*) — the difference between a dataset that renders faces well and one that also knows the body. The progress rail on the left keeps the whole pipeline legible: what's done, what's next, what's blocking Train.

### 4. Face-similarity scoring

Before an off-identity shot can poison training, **InsightFace** scores every image against your reference and badges it — green for a strong match, orange for borderline — with thresholds you set in Settings. In the curation grid above, the badges (e.g. `0.63` green, `0.47 to review`) are exactly this: a numeric, sortable signal for *"is this even the right person?"* that your eye alone will miss on shot 40.

### 5. Auto-framing classification

A local vision model classifies each image as **face / bust / body / back** and stamps a badge on the tile. That's what feeds the composition meter — and it's why the app can tell you the set is close-up-heavy without you tagging anything by hand.

### 6. Captioning that matches the model

Captions are what training actually reads, and the right *form* depends on the base model:

- **Prose** sentences for Z-Image / Krea 2 / FLUX.1 / FLUX.2 Klein, **booru-style tags** for SDXL — selected automatically from the dataset's target model.
- Generated by **JoyCaption** (via ai-toolkit) or an **Ollama** vision model.
- **Concept datasets invert** the caption: it names everything *but* the concept, and runs an **identity-leak check** so a stray "a woman with brown hair" doesn't quietly compete with the trigger.
- A **find/replace + tag-frequency** panel lets you sweep the whole set at once.

### 7. Edit the prompt, regenerate the shot

Every generated tile carries a ✏️ button next to crop and delete. Click it and the exact prompt that produced the image opens in an inline bubble — tweak the wording (*"soft window light,"* *"three-quarter view"*), hit **OK**, and the tile regenerates through the same engine with your edit, re-wrapped in the identity guard so the face is preserved. The edited prompt is saved with the image, so the next regenerate starts from where you left off.

<p align="center">
  <img src="docs/screenshots/04-editprompt.png" alt="A generated tile with the edit-prompt bubble open, showing the editable prompt and OK/Cancel" width="900">
</p>
<p align="center"><em>Fix a shot's framing or lighting by editing its prompt in place — no re-typing, no losing the rest of the set.</em></p>

### 8. Training you don't hand-tune

Click **Train** and ai-toolkit runs underneath — but you don't touch a config file:

- **Adaptive step counts** scaled to image count and clamped to a sane range (√n scaling for concept/style sets).
- A **training queue** with scheduling, so runs line up instead of colliding on the GPU.
- **Masked training** from **auto-generated rembg masks** — the app makes the masks and writes the `mask_path` config for you.
- **Continue +N steps** to extend a run, and **auto-import** of the finished LoRA into ComfyUI's `models/loras/<family>` so it's ready to test immediately.
- **Named presets** — save the whole ⚙️ Advanced panel as a named recipe, apply it to any dataset, and import/export it as a shareable JSON. Three recommended presets ship read-only (★): *Krea character*, *Concept*, and *Style*.
- **Checkpoint housekeeping** — a **Saves kept** cap lets ai-toolkit trim older intermediate checkpoints during the run (default 4, so a long Krea run no longer piles up ~10 GB of snapshots), and everything the app deletes goes to an app-wide **Trash** (Settings → Maintenance) that you empty on your own terms.
- **One place for every run** — a **🏋️ Runs** tab collects all training, cloud *and* local: live progress, the exact settings each launch used (and which dataset version, v1/v2/…, it trained on), **↻ Retry** a failed run on a fresh pod, **▶ Continue** a finished cloud run from its last checkpoint, and a one-click download of the resulting LoRA.
- Model families: **Z-Image**, **SDXL**, **Krea 2**, **FLUX.1**, **FLUX.2 Klein** — each with its own base/variant presets.

### 9. Test Studio — pick the best checkpoint

A LoRA that's trained isn't a LoRA that's *good*. Test Studio grid-tests **checkpoint × strength** through ComfyUI, lets you **vote** on the outputs (Wilson-ranked so a few votes don't overfit), and **ranks checkpoints by face similarity** — so you can pick the epoch that nails the identity *before* it overcooks, instead of guessing from sample images.

### 10. Export

At any point, **Export ZIP** gives you the curated, captioned set as a standard ai-toolkit dataset — pairs of `image` + `.txt` caption — that you can train anywhere. Nothing here locks your data in.

---

## Why this instead of driving ai-toolkit directly?

"Instead of" is the wrong frame: this app is **not a competitor to [ai-toolkit](https://github.com/ostris/ai-toolkit) — it orchestrates it**. When you click Train, ai-toolkit is the engine running underneath. The real question is whether to drive it through this studio or by hand (its own UI and config files):

| Stage of the job | ai-toolkit alone | LoRA Dataset Studio |
|---|---|---|
| Build the dataset from one photo | ❌ none — you arrive with your images | ✅ 3-engine fan-out, 45-shot variation catalog, 12/6/6/1 composition target |
| Build the dataset from the web | ❌ none | ✅ scrape a Reddit keyword search / gallery URL straight into a concept dataset (dedup + quality filters) |
| Curate | ❌ your file explorer | ✅ keep/reject, crop, composition meter, **InsightFace scoring** to drop off-identity shots *before* training |
| Captions | ❌ write them yourself | ✅ JoyCaption/Ollama, prose vs booru by family, identity-leak detection |
| Masked training | ⚙️ consumes `mask_path` if you supply masks | ✅ generates rembg masks and writes the config for you |
| Training | ✅ **it is the engine** — full control (rank, lr, optimizer…) | ⚙️ orchestrates: adaptive steps, queue + scheduling, continue +N, auto-import into ComfyUI |
| Pick the best checkpoint | ❌ its sample images + your eye | ✅ Test Studio: checkpoint × strength grids, Wilson-ranked voting, **face-similarity ranking** |

**Honest verdict:** this studio is the better tool when your goal is a **character LoRA built from a single reference photo** — roughly 80% of that job (dataset, curation, captions, epoch selection) happens *outside* training, and that 80% is exactly what ai-toolkit doesn't cover. It is *not* the better tool if you already have prepared datasets and want fine-grained hyperparameter tuning (the studio exposes type/base/variant/steps/masked, not rank or optimizer — use ai-toolkit directly for that), or for anything that isn't an image character LoRA. The two coexist cleanly: the studio's ZIP export is a standard ai-toolkit dataset you can always pick up by hand.

---

## Feature matrix by backend

Not every feature needs every backend. The app degrades gracefully — API keys show a Configured/Not-set status in Settings, endpoint reachability can be tested via the "Test" button, and gated features simply don't appear until their dependency is satisfied.

| Feature | Requires |
|---|---|
| API image generation (Nano Banana Pro) | `GEMINI_API_KEY` |
| API image generation (ChatGPT / `gpt-image-2`) | `OPENAI_API_KEY` |
| Klein image generation | ComfyUI reachable + Klein model installed |
| Captioning | Ollama **or** ai-toolkit (JoyCaption) |
| Auto-classify framing / auto head-crop | Ollama (vision model) |
| Face-similarity scoring | `backend/requirements-ml.txt` (insightface + onnxruntime) |
| Person masks | `backend/requirements-ml.txt` (rembg) |
| Scrape images into a concept dataset (Reddit keyword search + gallery URLs) | `backend/requirements-scrape.txt` (gallery-dl + curl_cffi) |
| Concept-caption inversion (identity-leak-aware) | Ollama **or** ai-toolkit (JoyCaption) |
| LoRA training | ai-toolkit installed and configured |
| Test Studio (checkpoint testing) | ComfyUI reachable |

## Two run modes

**API-only** — dataset creation, generation via Gemini/ChatGPT, curation, and export. Runs on any machine with Python and no GPU; this is what the Docker image ships. No ComfyUI, no ai-toolkit, no local ML extras required.

**Full local** — everything above plus Klein/Z-Image generation, captioning via JoyCaption, face scoring, masks, training, and Test Studio. Requires ComfyUI and/or ai-toolkit running on the same host (or reachable over the network) and an NVIDIA GPU with 12 GB+ VRAM for Klein/Z-Image inference. Training VRAM depends on the model family (Z-Image, SDXL, Krea 2, FLUX.1 and FLUX.2 Klein have different footprints) — check the family's ai-toolkit preset before queuing a run. The face-scoring and masking helpers (`requirements-ml.txt`) run fine on CPU; they don't need the GPU.

## Cloud training (vast.ai) — experimental

No local GPU? Add a **vast.ai API key** (Settings → Secrets, or the setup
wizard) and use **☁️ Train in cloud** in the Training panel. The app rents a
verified-datacenter GPU, uploads your dataset, trains with the exact same
ai-toolkit configuration as a local run, downloads the resulting
`.safetensors`, and terminates the pod automatically.

- Cost: you pay vast.ai directly (typical Z-Image run: **~$1–2**). A price cap
  (`cloud.max_price_per_hour`) and a hard runtime cap
  (`cloud.max_runtime_minutes`, default 4 h) are enforced.
- Supported families: **Z-Image, Krea and FLUX.2 Klein** (official Hugging Face
  bases; Klein 9B — 32-48 GB VRAM — is the cloud-first lane of its family).
  SDXL and custom converted bases require local training.
- Manage it from the **🏋️ Runs** tab (top nav): retry a failed run (↻), continue
  a finished run for more steps (▶), stop a run, and download the LoRA — cloud and
  local runs listed side by side, each showing the exact settings it used.
- Safety: pods are labeled `lds-<run-id>`; on every app start, orphaned pods
  are destroyed automatically. If the app is closed mid-run, the pod keeps
  training and the app resumes monitoring on restart.
- Privacy note: the pod belongs to your vast.ai account; dataset images and
  checkpoints transit through it and are destroyed with the pod.

---

## Setup & install

On first launch the **Setup** wizard scans your machine, tells you what's already installed, and walks you through the rest — but you can skip it and start building a dataset from your own photos right now, no setup required.

<p align="center">
  <img src="docs/screenshots/05-setup.png" alt="Setup wizard scanning the machine for ComfyUI, Ollama, and ai-toolkit" width="820">
</p>
<p align="center"><em>Setup detects ComfyUI (optional), an Ollama vision model, and ai-toolkit — and helps you install whatever's missing.</em></p>

First, get the code (every option below starts from here):

```bash
git clone https://github.com/perfectgf/lora-dataset-studio.git
cd lora-dataset-studio
```

### Option 1 — start.bat (Windows, one command)

The repo ships with the frontend prebuilt (`frontend/dist/`), so this doesn't need Node.js:

```
start.bat
```

No Python needed up front: `start.bat` looks for a compatible interpreter (`py -3.12/3.11/3.10` — the range with prebuilt wheels for the optional ML extras) and, if it finds none, **downloads a self-contained CPython 3.12** into a local `.python\` folder (~44 MB, once — no system install, no admin, nothing added to PATH). It then creates a `.venv`, installs `backend/requirements.txt`, opens `http://127.0.0.1:5050/` in your browser, and starts the server. So the whole flow is just: **unzip → double-click `start.bat`**. (Already have Python 3.10–3.12? It's used as-is and nothing is downloaded. On 3.13+ only, the core app still runs but the ML extras can't install.) Override the port with `set LDS_PORT=<port>` before running.

### Option 2 — manual venv (any OS)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
# optional, for face scoring + masks:
pip install -r backend/requirements-ml.txt
python backend/run.py
```

If you need to rebuild the frontend (e.g. you changed something under `frontend/src`):

```bash
cd frontend
npm install
npm run build
```

### Option 3 — Docker (API-only)

Copy `.env.example` to `.env` first — the compose file bind-mounts `./.env`, and Docker will otherwise create an empty directory in its place:

```bash
cp .env.example .env
```

Then build and run:

```bash
docker compose up --build
```

This builds and runs the API-only mode (see `Dockerfile` / `docker-compose.yml`) — ComfyUI and ai-toolkit are host-native tools and out of scope for the container. Data persists to `./data-docker` on the host, and your API keys are mounted in from `.env`.

### External tools (install once, connect in Settings)

None of these are bundled — each one is optional, installed separately, and then simply pointed to from the app's Settings page. Features light up automatically once their tool is detected (the "Test" button next to each field tells you immediately whether the app can see it).

| Tool | Unlocks | Get it |
|---|---|---|
| [ai-toolkit](https://github.com/ostris/ai-toolkit) (Ostris) | LoRA **training**, JoyCaption **captioning** | Follow its README install (clone + its installer creates a `venv`) |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | **Klein** local generation, **Test Studio** | Windows portable build, git install, or the ComfyUI Desktop app; keep it running on `http://127.0.0.1:8188` |
| [Ollama](https://ollama.com) | Auto-captioning, framing auto-classify, head-crop | Install, then `ollama pull qwen3-vl:8b-instruct` (use the **-instruct** tag, not the Thinking one — or set your own vision model in Settings) |

**ai-toolkit** — install it anywhere (e.g. `C:\ai-toolkit`), following [its own instructions](https://github.com/ostris/ai-toolkit#installation). Paste the folder path into **Settings → Local tools → ai-toolkit directory** and hit Test — training and JoyCaption captioning appear once it's valid. The app looks for `<folder>/run.py` and auto-detects the interpreter from a `venv/` **or** `.venv/` next to it (Scripts\python.exe on Windows, bin/python on Linux). Installed with conda, uv, or system Python and have **no venv folder**? Leave the directory pointing at the ai-toolkit folder and fill the optional **Python interpreter** field with the full path to the python that has ai-toolkit's dependencies. Job configs, datasets, and outputs live under that same folder by default (overridable under "Advanced").

**ComfyUI** — this app talks to a running ComfyUI over its HTTP API and scans its `models/` folders to list checkpoints and LoRAs. Set **Settings → ComfyUI API URL** (default `http://127.0.0.1:8188`) and **ComfyUI install directory** (the folder containing `models/`, `output/`, `input/`). Each family's base model goes in the layout its scanner expects:

- **Z-Image** → a sub-folder whose name contains **`z image`** (or `zimage`) under `models/unet` (or `models/diffusion_models`) — e.g. `models/unet/z image/bigLove_zt3.safetensors`. A file dropped **loose** in `models/unet` is *not* detected. The text encoder and VAE go at `models/text_encoders/Z image/qwen_3_4b.safetensors` and `models/vae/z ae.safetensors`.
- **SDXL** → `models/checkpoints` (a `Biglove/` sub-folder is also scanned).
- **Krea 2** → the default UNET at the root of `models/unet`; any extra Krea checkpoints under a `krea` sub-folder.

Trained LoRAs land in `models/loras/<family>` automatically after training. Generated images are pulled back over the ComfyUI API, so a custom ComfyUI output directory is fine — it doesn't need to match the install dir.

**Ollama** — used as the lightweight local vision backend. Any vision-capable model works; the default the app looks for is `qwen3-vl:8b-instruct` (the **Instruct** variant — the *Thinking* variant reasons out loud instead of captioning, so avoid it). If you run a different one, set its exact tag in **Settings → Ollama vision model**. If Ollama (or the model) is missing, the app degrades gracefully: imports fall back to a centered crop and captioning falls back to JoyCaption or manual captions.

### Getting API keys

- **Gemini** (for Nano Banana Pro): go to [aistudio.google.com](https://aistudio.google.com), click **Get API key**, and paste it into the app's Settings page.
- **OpenAI** (for ChatGPT / `gpt-image-2`): go to [platform.openai.com](https://platform.openai.com) → **API keys**, create a key, and paste it into Settings.

Both keys are stored in a git-ignored `.env` file (see `.env.example`) — they're never written to `config.json` and never committed.

---

## Minimum requirements

The app scales from "no GPU at all" to a full local training rig — each capability has its own floor, and everything degrades gracefully (missing pieces are simply hidden or guided through Setup).

| Mode / capability | GPU (NVIDIA) | Disk | Notes |
|---|---|---|---|
| **API-only** (generate via Gemini/ChatGPT, curate, caption via API, export ZIP) | none | ~2 GB | Any machine with Python 3.10–3.12; Docker image available |
| **Auto-captioning & framing** (Ollama vision, 8B model) | ~8 GB VRAM | ~7 GB | Runs alongside generation, not concurrently |
| **Local generation** (Klein 9B fp8 via ComfyUI) | ~16 GB VRAM | ~30 GB (model + text encoder + VAE) | Free, NSFW-capable; Setup downloads the models |
| **LoRA training — Z-Image / SDXL** (ai-toolkit) | 16 GB+ recommended | 10 GB+ free enforced per run | Quantized (qfloat8) + low-VRAM mode |
| **LoRA training — Krea 2** (ai-toolkit) | **24 GB VRAM** at 1024px (enforced warning) | ~24 GB base download (Raw) + 10 GB+ free | 12B model. Under 24 GB, set **Resolution → 768 only** in ⚙️ Advanced options — the main VRAM lever |
| **LoRA training — FLUX.2 Klein** (ai-toolkit) | 4B: **16–24 GB VRAM** · 9B: **32–48 GB** (cloud lane) | base download + 10 GB+ free | Both bases gated on Hugging Face (HF token required). Train the 9B via ☁️ cloud |
| **Face scoring / person masks** (ML extras) | none (CPU) | ~3 GB | Python **3.10–3.12 required** (no wheels beyond) |

- **OS**: Windows 10/11 for the full local stack (`start.bat`). Linux/macOS work for API-only + manual venv.
- **Python**: 3.10–3.12 — but not required up front: `start.bat` fetches a self-contained CPython 3.12 if your machine has none. 3.13+ (already installed) runs the core app but can't install the ML extras.
- **RAM**: 16 GB+ recommended when training locally.
- Reference rig used for development: RTX 4090 (24 GB) — every number above was measured or enforced there.

## Configuration reference

Copy `config.example.json` to `config.json` (git-ignored) and adjust. Every key:

| Key | Meaning |
|---|---|
| `server.host` | Interface the Flask server binds to (default `127.0.0.1`, local-only). |
| `server.port` | Port the server listens on (default `5000`). |
| `server.require_token` | On a non-loopback bind, require remote clients to present an access token (default `false` — a trusted LAN needs none). Toggle and token also live in Settings → Server & access. |
| `paths.dataset_images_root` | Where dataset images are stored. Empty string defaults to `<data dir>/datasets`. |
| `comfyui.api_url` | Base URL of your ComfyUI instance (default `http://127.0.0.1:8188`). |
| `comfyui.base_dir` | ComfyUI install directory, used to derive `output`/`input`/`models`/`loras` dirs if those aren't set explicitly. |
| `comfyui.output_dir` | Explicit override for ComfyUI's output folder. |
| `comfyui.input_dir` | Explicit override for ComfyUI's input folder. |
| `comfyui.models_dir` | Explicit override for ComfyUI's models folder (used to scan available checkpoints/UNETs). |
| `comfyui.loras_dir` | Explicit override for ComfyUI's LoRA folder. |
| `ollama.url` | Base URL of your Ollama instance (default `http://127.0.0.1:11434`). |
| `ollama.vision_model` | Ollama vision model used for auto-classify and auto head-crop (default `qwen3-vl:8b-instruct` — use the Instruct, not Thinking, variant). |
| `aitoolkit.dir` | ai-toolkit install directory. |
| `aitoolkit.datasets_dir` | Override for ai-toolkit's datasets folder (defaults to `<aitoolkit.dir>/datasets`). |
| `aitoolkit.output_dir` | Override for ai-toolkit's output folder (defaults to `<aitoolkit.dir>/output`). |
| `aitoolkit.hf_home` | Override for the Hugging Face cache directory ai-toolkit uses. |
| `aitoolkit.python` | Full path to the Python interpreter to run ai-toolkit with. Empty = auto-detect a `venv/`/`.venv/` next to `run.py`; set it for conda/uv/system-Python installs that have no venv folder. |
| `engines.default` | Default image-generation engine selected in the UI (`nanobanana`, `chatgpt`, or `klein`). |
| `engines.enabled` | List of engines shown as options in the UI. |
| `captioning.backend` | Caption backend: `auto` (prefer JoyCaption, fall back to Ollama), `joycaption`, `ollama`, or `none`. |
| `training.default_family` | Default model family preselected for new training runs (`zimage`, `sdxl`, `krea`, `flux`, or `flux2klein`). |
| `face_scoring.python` | Python interpreter used to run the InsightFace subprocess (empty = current interpreter). |
| `face_scoring.models_root` | Directory where InsightFace model weights are stored/downloaded. |
| `face_scoring.green` | Similarity score threshold (0–1) above which an image is flagged "green" (strong match). |
| `face_scoring.orange` | Similarity score threshold (0–1) above which an image is flagged "orange" (borderline match). |
| `masks.python` | Python interpreter used to run the rembg subprocess (empty = current interpreter). |
| `klein.consistency_lora` | Filename of the Klein consistency LoRA, relative to ComfyUI's LoRA folder. |
| `klein.consistency_strength` | Strength (0–1) applied to the Klein consistency LoRA. |

Secrets (`GEMINI_API_KEY`, `OPENAI_API_KEY`) live in `.env`, not `config.json` — copy `.env.example` to `.env`, or paste keys into Settings and let the app write them for you.

A few environment variables override paths for advanced/containerized setups: `LDS_DATA_DIR` (runtime data directory), `LDS_CONFIG` (path to `config.json`), `LDS_ENV` (path to `.env`), `LDS_HOST` (bind host, takes priority over `server.host`), `FLASK_DEBUG` (`1` to enable Flask debug mode).

## Exposing the app beyond localhost

The simplest path is the UI. **Settings → Server & access** has an *Available on the local network* toggle (flips the bind between `127.0.0.1` and `0.0.0.0`), an optional *Require an access token* switch (off by default — a home LAN is trusted), and an **Open it on your phone** card that shows a scannable **QR code** plus copyable URLs built from this machine's real LAN IP (and Tailscale IP, if present) — no guessing which address to type. Changing the port or the LAN toggle needs a restart; the card does it in one click.

Under the hood: the app has **no user accounts**, so on `127.0.0.1` (the default) that's fine, but any other bind would hand the whole network your API keys, GPU and datasets. On a non-loopback bind you can require an **access token**: with the token gate on, `run.py` generates one at boot (printed to the console with a ready-to-open URL) unless you set `LDS_ACCESS_TOKEN` yourself. Open `http://<machine>:<port>/?token=<token>` once from the remote device — a signed session cookie takes over from there. Requests from localhost never need the token. If your network is already locked down (VPN, authenticated reverse proxy), `LDS_ALLOW_UNAUTHENTICATED=1` disables the guard explicitly.

## Known limitations

- Krea 2's img2img workflow (`backend/workflows/krea2_turbo_img2img.json`) ships in the repo but isn't wired into a Test Studio mode yet — only the text-to-image Krea 2 workflow is currently reachable from the UI.
- ComfyUI-dependent code paths (Klein generation, Test Studio, the consistency-LoRA path normalization for Windows ComfyUI) are covered by unit tests against a mocked ComfyUI API; they haven't all been exercised against a live ComfyUI instance yet. If something looks wrong when wiring up your own ComfyUI, check Settings → the "Test" button next to each endpoint.
- The dataset workspace remembers your last-used generator (`localStorage`) and defaults to Nano Banana Pro on a first visit. If you've only configured an OpenAI key, the Nano Banana card shows disabled and the Generate button stays greyed out until you explicitly click the ChatGPT card — a one-click step that's easy to miss right after onboarding.

## Troubleshooting

**`npm install` fails with `Cannot find module @rollup/rollup-<platform>-...`**
A known npm bug ([npm/cli#4828](https://github.com/npm/cli/issues/4828)) can make `package-lock.json` "remember" the platform it was generated on. Fix: run `npm i -D @rollup/rollup-<your-platform>` for your OS/arch, or delete `frontend/node_modules` and `frontend/package-lock.json` and run `npm install` again on the target platform.

**Training log looks frozen for several minutes**
This is normal — ai-toolkit's stdout is block-buffered during model load and latent caching, so nothing prints for a while even though it's working. Check GPU utilization or watch for new files under the ai-toolkit output directory to confirm it's alive; a "warming up" state before the first logged step is expected.

**ComfyUI shows as unreachable**
Check `comfyui.api_url` in Settings, confirm ComfyUI is actually running, and check that nothing (firewall, a different bind interface) is blocking the connection between this app and ComfyUI.

**Port 5000 conflicts with AirPlay Receiver on macOS**
macOS reserves port 5000 for AirPlay Receiver by default. Change `server.port` in `config.json` to something else (e.g. `5050`) and restart.

**Windows console shows garbled characters (mojibake) from `start.bat`**
Cosmetic only — some UTF-8 text (em dashes, accents) renders incorrectly on the legacy Windows console codepage. It doesn't affect functionality.

Still stuck? Open the app's **Guide → Getting help** for the one-click **diagnostic report** (version, capability status, log tail — no keys, no paths), then post it on [Discord](https://discord.gg/j6hnJBFtXE) or in a [GitHub issue](https://github.com/perfectgf/lora-dataset-studio/issues).

---

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

This software is provided **"as is", without warranty of any kind**, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement (see the [MIT License](LICENSE) for the full text). To the maximum extent permitted by law, **the authors and contributors accept no liability** for any claim, damage, or other liability — including any legal consequence arising from datasets, models, or images you create with it. By using this software you accept that responsibility yourself.

## License

MIT — see [LICENSE](LICENSE).
