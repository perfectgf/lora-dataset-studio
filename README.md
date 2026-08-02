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

<p align="center">
  <img src="docs/screenshots/02-workspace.png" alt="Guided dataset workspace: a progress rail mapping reference, generation, curation, captioning and training, next to the curation grid and its bulk actions" width="820">
</p>
<p align="center"><em>One workspace for the full route. Every person shown in these screenshots was produced by the app's own generation engines; no real individual is depicted.</em></p>

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
| **Import or scrape** | Drag in images, merge ZIP/folder datasets, search Reddit or Pexels, or scan supported galleries and direct-media URLs |

API generation follows each provider's billing and content policy. Read the direct notes for [Gemini](docs/guide/settings-reference.md#what-the-gemini-engine-will-and-will-not-do), [ChatGPT subscription mode](docs/guide/settings-reference.md#chatgpt-subscription-experimental), [OpenRouter and image-engine settings](docs/guide/settings-reference.md#image-engines), and [Pexels authorization](docs/guide/workflow.md#the-built-in-web-scraper) before using those lanes. The local engines do not send reference images to an API.

### Image Bank

| Capability | What it provides |
|---|---|
| **Folder or web scrape → bank** | Inventory a live folder in place, or scrape into a new/existing bank without applying dataset filters on the way in |
| **Quality and similarity passes** | Flag blur, noise, flat frames, small/soft-detail images and black bars; group duplicates, crops and recompressions |
| **Aesthetic, NSFW and style scoring** | One scoring pass produces rankings, groups and reusable embeddings |
| **People, framing and captions** | Cluster faces without a reference, classify face/bust/body/back, and caption the bank for full-text search |
| **Medium and head angle** | Split a mixed dump into photographs, anime, 3D renders and illustrations (reusing the scoring embeddings, no new inference), and filter by frontal / three-quarter / profile / back view. Both answer "unsure" or "not measured" instead of guessing, and both say so on screen: non-photo verdicts are rare by design, and profiles are under-counted because a hard-turned head often defeats face detection |
| **Find and shortlist** | Find by text, pick diverse, make framing-balanced picks, find similar images, or promote a shortlist into a new bank |
| **Fast review tools** | Filter, sort, review one by one, rotate without rewriting the source, compare an improved candidate with its original, and re-run only eligible passes |
| **Editable watermark masks** | Detect marks, redraw several mask zones, then crop or repaint into a separate clean derivative; undo returns to the untouched source |
| **Dataset ↔ bank round trip** | Promote bank keepers into a dataset or copy dataset keepers into a bank while retaining compatible metadata and provenance |
| **Safe bulk work** | Undo the last bulk decision, tune thresholds where you work, move a bank without losing analysis, or run the full chain overnight |

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
| **Custom bases and continuation** | Train compatible custom weights, continue from any saved epoch, or use verified full-state resume where available |
| **Runs hub** | Local and cloud runs together with progress, logs, stop/retry/continue/download actions and paste-safe config sharing |
| **Experiment lineage** | Inspect, annotate and diff the exact tree of runs and the checkpoint each continuation resumed from |
| **LoRA Canvas** | Put every dataset's lineage on one pan/zoom board, rearrange cards, compare runs across datasets, generate from same-family checkpoints — including 🧬 blending several checkpoints into one image, with purple provenance edges joining a blended picture to every pill it came from (blends made before this feature show a badge instead) — pin/fuse outputs and continue training from a pill; each generation run keeps its own strip in training-step order, with the character dataset's reference face on its lane |
| **Test Studio** | Fixed-seed checkpoint × strength grids, multi-LoRA comparisons or 🧬 combined stacks (several of your LoRAs in one image, each at its own weight, weight variants compared side by side), a ✨ Enhance button that enriches your prompt through your local Ollama, votes, Wilson ranking, face ranking and shareable exports |
| **Studio shortcuts and recovery** | Open Studio directly from a run, draw prompts from kept dataset captions, and pause safely when ComfyUI drops instead of launching later cells against changed state |

### Keep control of the files

| Capability | What it provides |
|---|---|
| **Training ZIP and sidecars** | Standard kept image + same-stem `.txt` pairs for ai-toolkit/Kohya-compatible tools |
| **Portable backup and restore** | Datasets, decisions, captions, settings and run history in one file; API keys stay out |
| **Hugging Face publishing** | Publish kept pairs to a dataset repository, private by default and gated by an explicit rights confirmation |
| **ComfyUI deployment** | Deploy individual checkpoints or downloaded cloud results into the configured LoRA tree |
| **Recoverable deletion** | Deleted app data goes to Trash; destructive Image Bank actions state their destination before confirmation |
| **Storage you can see and move** | Settings › Storage lists every folder the app writes to with its path and (on request) its size, and can point the dataset root, the cloud run staging and the checkpoint store at another drive — moving what is already there, or adopting the new folder empty, never silently. Trained checkpoints live in their own store that no cleanup touches; the trash sits on the same disk, so space returns only when you empty it. |

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
      <sub><strong>Runs hub</strong> — follow local and cloud experiments together.</sub>
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

- **🧬 Merge Lab** *(next big one)* — bake your trained LoRAs into a standalone, shareable checkpoint and merge models with guided recipes, judged side by side in the Test Studio (same seeds, A/B grids). Full model fine-tuning on large curated datasets comes later on the same path.
- **🎬 WAN 2.1 / 2.2 video LoRAs** — ai-toolkit already trains WAN and the scraper can already pull video, so the whole pipeline (scrape, curate, caption, train, test) extends naturally to motion. Community-driven.
- **🧠 Watermark cleaning during import** — cleaning that happens **during import** instead of as a separate errand, and automation you can trust unattended. *(Detection has caught up: a dedicated detector that needs no vision model now ships alongside the Ollama path, and manual two-pass cleaning already works in datasets and in the Image Bank.)*
- **🧩 More base models** — additional Flux-family bases (Chroma, Qwen-Image…) with the same one-click flow as Krea 2.

## Why this instead of ai-toolkit?

"Instead of" is the wrong frame: this app is **not a competitor to [ai-toolkit](https://github.com/ostris/ai-toolkit) — it orchestrates it**. ai-toolkit is the training engine; LoRA Dataset Studio adds the work before, around and after a run.

| Stage | ai-toolkit alone | LoRA Dataset Studio |
|---|---|---|
| Build from references | ❌ bring your own images | ✅ five engines, simultaneous multi-engine batches, subject-aware catalogs including Anime, reference edits and exact retries |
| Build from the web | ❌ none | ✅ Reddit, Pexels and supported URL scans into a dataset or Image Bank, with deduplication and explicit provider warnings |
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

Missing dependencies are shown in Setup/Settings and gated features stay unavailable until their requirements are satisfied.

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
| Image Bank scoring, crops and semantic tools | Bank scoring extra from `backend/requirements-ml.txt`; semantic search/diversity reuse an existing Score pass, and balanced picks also need Framing |
| Watermark detection | Ollama with a vision model, **or** the dedicated detector (torch + transformers — the bank-scoring extra's environment is reused when present — plus ~0.9 GB of model downloads at first use) |
| Watermark inpainting | LaMa extra from `backend/requirements-ml.txt`, or ComfyUI + Klein for the refine lane; crop remains model-free |
| Scraping | `backend/requirements-scrape.txt`; Pexels also needs `PEXELS_API_KEY` and explicit authorization |
| Local LoRA training: Z-Image / Krea 2 / FLUX.1 / FLUX.2 Klein / Anima | ai-toolkit; no ComfyUI is needed for official Hugging Face bases |
| Local SDXL training | ai-toolkit + a base checkpoint discoverable in ComfyUI's model tree |
| Cloud training | `VAST_API_KEY`; supported families are shown in the launch UI. Full-model Krea 2 also needs `HF_CLOUD_TOKEN` with Krea base read and repository write access; fine-grained is recommended, global `role=write` is accepted with a warning, and read-only is rejected. A finished full model (~26 GB, plus its ~10 GB fp8 twin) is downloaded **to your machine** and verified before the rented pod is released; a copy of the master is then pushed to a private Hugging Face repository as a backup — that copy is what makes the run resumable later, and it can be turned off. So you need **room on the checkpoint drive** (checked before anything is rented), and Hugging Face room only for the backup; Settings ▸ Storage lists what is taking that space |
| Merging a LoRA into a base checkpoint (produces a full model) | A Python with `torch` (the same one fp8 quantization uses) and room for a second copy of the base — a 26 GB Krea 2 base takes about two minutes and writes 26 GB. Refused on an already-quantized base: merge into the full-precision file, then quantize. The result is a **merged** model, not a trained one, and its metadata says so |
| LoRA Canvas browsing, layout, notes and diffs | No external service; generating needs ComfyUI and same-family checkpoints, continuing needs the chosen local/cloud training lane |
| Test Studio | ComfyUI reachable + assets for a supported Studio family |
| Backup/restore and ZIP/folder merge | No external service |
| Hugging Face publishing | Write-enabled `HF_TOKEN`; repositories are private by default |

## Run it your way

| Mode | Good for | What is optional or unavailable |
|---|---|---|
| **Docker + existing ComfyUI** | Run LDS in Docker while keeping the ComfyUI already installed on the host | The launcher asks for the ComfyUI folder once; local training still uses host ai-toolkit or the cloud |
| **Docker GPU + fresh ComfyUI** | Run LDS and a new isolated ComfyUI together on an NVIDIA GPU | Existing ComfyUI/models stay untouched; local training still uses host ai-toolkit or the cloud |
| **Rented GPU pod (RunPod)** | Reach the studio, Image Bank and ComfyUI generation from any browser, on a GPU you do not own | Training still rents a vast.ai instance; ai-toolkit is not in the image, so local training is unavailable. Large ZIP exports can hit the pod proxy's 100-second timeout. See the [RunPod guide](docs/guide/runpod.md) |
| **Full local** | Local engines, ML helpers, ai-toolkit training, Canvas generation and Test Studio | Install/connect only the tools you need; each capability degrades independently |

## Setup & install

On first launch, **Setup** scans the machine and links every missing capability to its install/configuration step. You can skip optional tools and begin with imported images immediately.

### Option 1 — release ZIP + start.bat (Windows)

Download **`LoRA-Dataset-Studio-windows.zip`** from the [latest release](https://github.com/perfectgf/lora-dataset-studio/releases/latest) when that asset is present; otherwise use GitHub's **Source code (zip)**. Extract the entire archive, then double-click:

```text
start.bat
```

`start.bat` uses Python 3.10–3.12 if available. If none is installed, it downloads a self-contained CPython 3.12 into `.python\`, creates `.venv`, installs the core requirements, opens `http://127.0.0.1:5050/`, and starts the server. It requires no admin rights and changes no system PATH.

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

**Beginner Windows flow:** download/extract the GitHub ZIP, start Docker Desktop, then double-click **`start-docker.bat`**. On the first run, select either the ComfyUI folder containing `main.py` and `models`, or its portable parent containing `ComfyUI\main.py`. LDS validates the folder and remembers it for this checkout.

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
| **API-only** (Gemini/ChatGPT/OpenRouter generation, import/scrape, curate, manual captions, export/backup) | none | ~2 GB | Any machine with Python 3.10–3.12; Docker image available |
| **Auto-captioning & framing** (Ollama vision, 8B model) | ~8 GB VRAM | ~7 GB | Runs alongside generation, not concurrently |
| **Local generation** (Klein 9B **KV** fp8 via ComfyUI) | ~16 GB VRAM | ~30 GB (model + text encoder + VAE) | Free, local and NSFW-capable; Setup downloads the models. The KV build is up to **2.5× faster on multi-reference edits** at the same quality. Available in Docker GPU mode |
| **LoRA training — Z-Image / SDXL** (ai-toolkit) | 16 GB+ recommended | 10 GB+ free enforced per run | Quantized (qfloat8) + low-VRAM mode |
| **LoRA training — Krea 2** (ai-toolkit) | **24 GB VRAM** at 1024 px (enforced warning) | ~24 GB base download (Raw) + 10 GB+ free | Under 24 GB, select **Resolution → 768 only** in Advanced options |
| **LoRA training — FLUX.2 Klein** (ai-toolkit) | 4B: **16–24 GB VRAM** · 9B: **32–48 GB** | base download + 10 GB+ free | Both bases are gated on Hugging Face; the cloud lane is practical for 9B |
| **Face scoring / person masks / watermark inpaint** (ML extras) | none (CPU) | ~3 GB (+ CPU torch for LaMa) | Python **3.10–3.12 required** for wheels; installable per capability from Setup |

- **OS:** Windows 10/11 for the full local stack (`start.bat`). Linux/macOS work for API-only + manual venv; GPU Docker depends on host NVIDIA support.
- **Python:** 3.10–3.12, but not required up front: `start.bat` fetches a self-contained CPython 3.12 when none is installed. Python 3.13+ can run the core app but not the ML extras.
- **RAM:** 16 GB+ recommended for local training.
- Reference development rig: RTX 4090 (24 GB); every number above was measured or enforced there.

## Configuration & network access

Use **Settings** for normal configuration. The complete defaults, `config.json` keys, model locations and environment overrides live in [docs/guide/settings-reference.md](docs/guide/settings-reference.md).

The server binds to `127.0.0.1` by default. Before enabling LAN access or publishing a port, read [SECURITY.md](SECURITY.md#the-default-threat-model) and configure the access-token/VPN/reverse-proxy boundary that fits your network.

When the app is served on an address the public internet can reach — a rented pod's proxy hostname, a tunnel — set `LDS_PUBLIC=1`. That forces the access token on whatever the setting says, so the switch cannot be turned off into an open door, and generates a token at boot if none exists. It applies to non-loopback binds only, and `LDS_ALLOW_UNAUTHENTICATED=1` still overrides it for setups that authenticate elsewhere.

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
