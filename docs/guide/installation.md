# Installation

[Documentation](../README.md) · [Requirements](requirements.md) · [First launch](getting-started.md)

On first launch, **Setup** prepares the core. No plugin is needed to import and organise images. Afterwards, open **Plugins → Store**, select the features you want and install them together with one LDS restart. Each plugin has its own settings and preparation steps; required ComfyUI custom nodes are installed through that plugin's preparation flow.

The Store offers **free public plugins** for generation, editing, training, publishing and other optional features. Each listing describes the plugin's capabilities and preparation requirements. Updates are delivered through **Plugins → Updates**.

## Windows

Download **`LoRA-Dataset-Studio-windows.zip`** from the [latest release](https://github.com/perfectgf/lora-dataset-studio/releases/latest). For a new installation, extract the entire archive into a new folder, then double-click:

```text
start.bat
```

`start.bat` uses Python 3.11–3.12 if available. If none is installed, it downloads a self-contained CPython 3.12 into `.python\`, creates `.venv`, installs the core requirements, opens `http://127.0.0.1:5050/`, and starts the server. It requires no admin rights and changes no system PATH.

On an existing ZIP installation, **Update & restart** downloads the next release and swaps the core in, keeping `data/`, `config.json`, `.env`, `.venv` and `.python` untouched. Install and update optional plugins separately from the Store. A git checkout follows its configured branch instead and needs `git` on your PATH, which an install made through a desktop Git client does not always provide.

The default `v2` branch carries the maintained version of LDS. Clone it to follow its commits with **Update & restart**:

```bash
git clone https://github.com/perfectgf/lora-dataset-studio.git
cd lora-dataset-studio
start.bat
```

The former `main` branch is now [`v1`](https://github.com/perfectgf/lora-dataset-studio/tree/v1). It is read-only and no longer maintained; updates and contributions go to `v2`.

**Want a guided migration?** [Download the V2 migration helper](https://github.com/perfectgf/lora-dataset-studio/releases/download/v2026.09.14.4/LDS-Migrate-to-V2.zip), extract it outside your installation and run `migrate-to-v2.bat`. It checks the Git installation, backs up the database/settings and switches branches while leaving media in place. [Instructions, backup scope and supported installations](migrate-to-v2.md).

For an existing git installation still on `main`, stop LDS and run these commands from its folder, then start LDS again:

```bash
git fetch origin
git switch v2
git branch --set-upstream-to=origin/v2 v2
```

## Manual installation

Clone the default branch as above or download its [source archive](https://github.com/perfectgf/lora-dataset-studio/archive/refs/heads/v2.zip), open a terminal in its root, then run:

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

## Docker with an existing ComfyUI

**Beginner Windows flow:** download/extract the [**source ZIP**](https://github.com/perfectgf/lora-dataset-studio/archive/refs/heads/v2.zip) — the release asset `LoRA-Dataset-Studio-windows.zip` does not carry the Docker launchers — start Docker Desktop, then double-click **`start-docker.bat`**. On the first run, select either the ComfyUI folder containing `main.py` and `models`, or its portable parent containing `ComfyUI\main.py`. LDS validates the folder and remembers it for this checkout.

Start your usual ComfyUI on the host. LDS uses `http://host.docker.internal:8188` from its container and mounts the selected folder at `/external-comfyui`. If the folder later moves, double-click **`configure-docker.bat`**. The launcher chooses a free Studio port and opens the browser automatically.

## Docker with a fresh ComfyUI

**Beginner Windows flow:**

1. Download the [source ZIP](https://github.com/perfectgf/lora-dataset-studio/archive/refs/heads/v2.zip), then extract the complete folder.
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

For the advanced CLI, the default addresses remain `http://127.0.0.1:5050/` for Studio and `http://127.0.0.1:8188/` for ComfyUI; `.env` can override them. This lane requires an NVIDIA GPU, a compatible driver and NVIDIA Container Toolkit support. Storage relocation, ports, existing-ComfyUI adoption, UID/GID, DNS, update commands, resource caps and operational limits are documented in the dedicated [Docker guide](docker.md).

## Docker without a GPU

For a machine with no NVIDIA GPU: generation through Gemini/ChatGPT/OpenRouter, import and scraping, curation, manual captions, export and backup. ComfyUI and ai-toolkit stay out of this image, so local generation, Test Studio and local training are unavailable in this lane.

```bash
cp .env.example .env
mkdir -p data-docker
docker compose up --build          # docker-compose.yml, the default file
```

Studio answers on `http://127.0.0.1:5050/` and its data lives in `./data-docker`. This is the only Docker lane that needs no NVIDIA support at all.

To update a Docker install, double-click **`update-docker.bat`** for the latest stable release, or pass `v2` to follow the maintained branch (`main` remains a compatibility alias for `v2`). It rebuilds transactionally and rolls back if the container does not come up healthy. Both `start-docker.bat` and `start-docker-gpu.bat` accept `--rebuild` and `--update-rebuild`; `start-docker.bat` also accepts `--configure`, which is what `configure-docker.bat` calls. After upgrading from V1, install the optional features you use from **Plugins → Store**. Both images include the public Store configuration; [Docker plugin administration and restarts](docker.md#v2-plugins) explains how to unlock installation and apply changes.

## Pinokio

In [Pinokio](https://pinokio.computer), open **Discover → Download from URL** and paste `https://github.com/perfectgf/lora-dataset-studio.git`, then click **Install** and **Start**. Pinokio builds the Python environment, installs the core requirements and opens Studio; **Update** fast-forwards the same checkout the in-app updater uses.

Only the core app is installed this way. Complete **Setup**, then choose your optional features in **Plugins → Store**; each plugin carries its own settings and preparation steps. Updates go through Pinokio's **Update** tab: because Pinokio starts and stops the server, the app detects this install shape and shows *Stop → Update → Start* instead of its own **Update & restart** button, which would relaunch the server outside Pinokio's control.

The app needs Python 3.11 or newer. Pinokio creates `env/` on Python 3.12 (`venv_python`), which requires **Pinokio 3.8.27 or newer**: update Pinokio first if it is older. An installation made on Pinokio's bundled Python 3.10 needs **Update twice** when it reaches this version: the first Update still runs the previous installer, which cannot install the new requirements, and Start then says so; the second one moves the old `env/` aside and rebuilds it on Python 3.12. Datasets, settings and the Image Bank are not touched.

## External tools

| Tool | Unlocks | Connect it |
|---|---|---|
| [ai-toolkit](https://github.com/ostris/ai-toolkit) | Local LoRA training and JoyCaption | Set its directory and Python interpreter in **Settings → Local tools**; conda, uv, venv and portable Python installs are supported |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | Klein/Krea local generation, Studio, Canvas generation and deployment; SDXL base discovery | Keep its API reachable and set the install/models paths in **Settings → Local tools** |
| [Ollama](https://ollama.com) | Auto-captioning, framing, head-crop and watermark detection | In Docker, choose none/host/companion in **Setup**, then pull the model explicitly from LDS; native installs can use their configured URL |
| [LM Studio](https://lmstudio.ai) | The same, if that is the local model server you already run | Pick it in **Settings ▸ Local tools**. It only serves a model you have loaded (no JIT by default), and LDS cannot start it for you — its Developer tab has the switch |

Which of the two serves those features is a single setting (**Settings ▸ Local tools ▸ Local LLM provider**); Ollama stays the default. The full path rules, model layouts and provider states are in the [settings reference](settings-reference.md#local-tools). If a tool remains unavailable, use the [troubleshooting guide](troubleshooting.md).

## API keys

| Service | Used for | Where to create it |
|---|---|---|
| Gemini | Nano Banana Pro | [Google AI Studio](https://aistudio.google.com) |
| OpenAI | ChatGPT / `gpt-image-2` | [OpenAI API keys](https://platform.openai.com/api-keys) |
| OpenRouter | Image models through OpenRouter | [OpenRouter keys](https://openrouter.ai/keys) |
| Pexels | Optional official-API image search | [Pexels API key](https://www.pexels.com/api/key/) |
| Hugging Face | Gated weights and optional publishing | [Hugging Face tokens](https://huggingface.co/settings/tokens) |
| vast.ai | Optional cloud training | [vast.ai console](https://cloud.vast.ai/?ref_id=683073) (referral link; disclosed below) |

> **Affiliate disclosure.** The vast.ai links in this README, in the guides and in the app are referral links. If you create an account through one of them, vast.ai pays this project 3% of what you spend on their platform, for as long as your account lives. It costs you nothing extra — vast.ai's prices are the same either way. Cloud training runs on your own API key, and vast.ai bills you directly. Optional usage statistics are separate from these referral links and remain off unless you choose to share them; see [Network access and privacy](network-access.md).

Secrets saved in Settings live in the git-ignored `.env`, never in `config.json` or a commit. Full-model Krea 2 cloud runs use a separate `HF_CLOUD_TOKEN`; a narrowly scoped fine-grained token is recommended, while a global `role=write` token is accepted with a broad-access warning and read-only is rejected. Configure it in **Plugins → Cloud training → Settings**; see the [cloud requirements](requirements.md#dependencies-by-feature).

> **Pexels authorization required:** An API key alone does not authorize dataset or machine-learning use. Configure this integration only if Pexels has explicitly authorized this use case, and keep the attribution LDS displays. Read the [official Pexels terms and conditions](https://help.pexels.com/hc/en-us/articles/900005880463-What-are-the-Terms-and-Conditions/).

## Hosted and local options

| Mode | Good for | What is optional or unavailable |
|---|---|---|
| **Docker + existing ComfyUI** | Run LDS in Docker while keeping the ComfyUI already installed on the host | The launcher asks for the ComfyUI folder once; local training still uses host ai-toolkit or the cloud |
| **Docker GPU + fresh ComfyUI** | Run LDS and a new isolated ComfyUI together on an NVIDIA GPU | Existing ComfyUI/models stay untouched; local training still uses host ai-toolkit or the cloud |
| **Rented GPU pod (RunPod)** | Reach the studio, Image Bank and ComfyUI generation from any browser, on a GPU you do not own | Training still rents a [vast.ai](https://cloud.vast.ai/?ref_id=683073) instance; ai-toolkit is not in the image, so local training is unavailable. Large ZIP exports can hit the pod proxy's 100-second timeout. See the [RunPod guide](runpod.md) |
| **Full local** | Local engines, ML helpers, ai-toolkit training, Canvas generation and Test Studio | Install/connect only the tools you need; each capability degrades independently |
