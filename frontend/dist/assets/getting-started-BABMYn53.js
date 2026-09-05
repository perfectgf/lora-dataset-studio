const n=`# Getting started

LoRA Dataset Studio turns one reference photo into a trained, ranked LoRA —
curation, captioning, face-scoring and training behind a single browser tab, on
your own machine. The useful part of LoRA training isn't the training; it's
building a clean, balanced, well-captioned image set. This app puts that whole
pipeline behind one UI.

> **In a hurry?** Launch the app, let the **Setup** wizard scan your machine,
> and create your first dataset from your own photos — no API key, no GPU, no
> external tool required for that first step.

---

## Two ways to run it

| | API-only | Full local |
|---|---|---|
| **What works** | Create datasets, generate via Gemini/ChatGPT, curate, caption via API, export ZIP | Everything — plus local (Klein) generation, JoyCaption, face scoring, masks, training, Test Studio |
| **Needs** | Python 3.10–3.12, an API key | ComfyUI and/or ai-toolkit + an NVIDIA GPU (12 GB+ for local generation) |
| **Good for** | Laptops, first try, cloud training | The full pipeline on a training rig |

You can start API-only and add the local tools later — features light up
automatically when their tool is detected.

## First launch

**Windows (one command):** download \`LoRA-Dataset-Studio-windows.zip\` from the
[latest release](https://github.com/perfectgf/lora-dataset-studio/releases/latest),
extract it, then double-click \`start.bat\`. Releases contain an archive/source, not
a prebuilt executable launcher. \`start.bat\` finds or downloads a compatible Python
(3.10–3.12), creates \`.venv\`, installs the requirements, and opens the app at
\`http://127.0.0.1:5050/\`.

**Any OS (manual venv):**

\`\`\`
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r backend/requirements.txt
python backend/run.py
\`\`\`

**Pinokio (one click, any OS):** in [Pinokio](https://pinokio.computer), use
**Discover → Download from URL** with
\`https://github.com/perfectgf/lora-dataset-studio.git\`, then **Install** and
**Start**. Pinokio creates the environment (\`env/\`), installs the core
requirements and opens Studio on the port it really bound. Two things to know:
the optional tools are still connected from **Setup**, and updates go through
Pinokio's **Update** tab — it runs the same \`git pull --ff-only\` as the in-app
updater. You do not have to remember that last part: the app recognises a
Pinokio launch and its Updates card shows *Stop → Update → Start* (with how many
commits behind you are) instead of the **Update & restart** button, which would
relaunch the server in a window Pinokio no longer tracks.

**Docker (API-only):** \`cp .env.example .env\`, then \`docker compose up --build\`.

**Docker (GPU, with ComfyUI inside):** \`cp .env.example .env\`, then
\`mkdir -p run basedir data-docker-gpu bank-images\` — create those yourself, or Docker creates
them as \`root\` and the app cannot write to them — then
\`docker compose -f docker-compose.gpu.yml up --build\`. Needs an NVIDIA GPU and the
NVIDIA Container Toolkit; this is the only Docker option that can do Klein/Z-Image
generation and the Test Studio.

The CUDA 12.x compatibility minimum is driver **525.60.13 on Linux** or **528.33
on Windows/WSL**; for the default CUDA 12.9 image, use at least **575.51.03 on
Linux** or **576.02 on Windows**. The optional CUDA 13 image needs **R580+**.
If you reuse an existing ComfyUI tree, \`LDS_COMFY_BASEDIR\` must name the parent
that directly contains \`models/\`, \`input/\`, \`output/\` and \`custom_nodes/\`, not
the \`models/\` folder itself. Compose actively defaults DNS to \`1.1.1.1\`; set
\`LDS_DNS\` to your router or Pi-hole when internal hostnames matter.

Both ports are published on the host and remain reachable from the LAN. A
Settings restart is handled by the container supervisor and keeps the fixed
container bind. To update this Docker flavour, update and rebuild the image rather
than trying to replace \`/app\` in place:

\`\`\`
git pull
docker compose -f docker-compose.gpu.yml up -d --build
\`\`\`

If \`/data\` or existing contents beneath it are not writable, \`LDS_FORCE_CHOWN=true\`
is the last resort: it recursively changes ownership only for the \`LDS_DATA\` mount,
never the ComfyUI or bank mounts.

The full install matrix (Windows release ZIP, GPU requirements, external tools)
lives in the README on GitHub.

## What is running on your machine {#architecture}

A full-local install is **three separate programs**, in **two folders**, on
**two ports**, with **two Python environments**. Nothing in the app used to say
so, and the cost of guessing is real: a user spent hours patching ai-toolkit's
own web UI — a component this app never talks to — while the actual problem was
one line of \`config.json\` (reported by strouder, GitHub #19).

| Component | Where | What it does | Port |
|---|---|---|---|
| **LoRA Dataset Studio** (Flask) | the folder you extracted, **its own \`.venv\`** | The UI you are reading this in. Curation, captioning, and the thing that **starts and stops training**. | **5050** |
| **ai-toolkit** (\`run.py\`) | the folder you point at in Settings, **its own venv — the one with \`torch\`** | The training **engine**. It is run as a command-line process; it has no UI of its own. | — |
| ai-toolkit's Next.js UI (\`ui/\`) | inside the ai-toolkit folder | An **unrelated** web interface that ships with ai-toolkit. This app never launches it, never reads it, and never writes to it. | 8675 |

Two consequences worth remembering:

- **The Studio reads \`config.json\` and drives training.** That file belongs to
  this app (it sits in its data folder, and every key in it has a field in
  **Settings**). Editing anything inside ai-toolkit's \`ui/\` folder changes
  nothing here — if you ever find yourself editing \`ui/dist/…\`, you are in the
  wrong project.
- **The two Python environments are not interchangeable.** The Studio's \`.venv\`
  runs the web app; ai-toolkit's venv is the one that must have \`torch\` and the
  training dependencies. Settings ▸ Local tools ▸ **Python interpreter** is how
  you tell the app which interpreter is ai-toolkit's — and its **Test** button
  now checks that the interpreter can really \`import torch\`, not just that the
  file exists.

There is no process stacking to worry about on our side: the Studio is a single
Flask process, and starting a second one on the same port fails loudly instead
of running invisibly alongside the first.

### Supported Python versions {#python-versions}

- **LoRA Dataset Studio: Python 3.10 – 3.12.** \`start.bat\` finds or downloads
  one for you; the optional ML extras (insightface, onnxruntime, \`numpy<2\`)
  only publish wheels for those versions.
- **ai-toolkit: Python 3.11 is the safe choice.** On **3.13** its pinned
  \`scipy==1.12.0\` has no wheel, pip falls back to building from source and dies
  on a missing Fortran compiler (measured and reported by strouder, GitHub #19).
  On Windows, install **3.11.9** — it is the last 3.11 with a binary installer;
  later 3.11.x are source-only security releases.
- The two do **not** have to match. They are separate environments on purpose.

### If Hugging Face downloads fail {#hf-downloads}

If a base-model download dies with something that reads like a network error,
check whether \`HF_HUB_ENABLE_HF_TRANSFER=1\` is set in your environment. That
turns on an optional fast-download accelerator which needs the \`hf_xet\` (or
\`hf_transfer\`) package installed **in the environment doing the downloading**;
without it, transfers abort with a misleading message. This app never sets that
variable — it comes from your shell, ai-toolkit's \`.env\`, or a ComfyUI launcher.
Either fix works: set \`HF_HUB_ENABLE_HF_TRANSFER=0\`, or \`pip install hf_xet\`.
The training failure panel now recognises this case and says so.
*(Reported by bobba84, GitHub #18.)*

## The Setup wizard

On first launch you land in **Setup**. It scans your machine automatically and
walks through five steps — each one unlocks a set of features:

1. **Image generation** — add a Gemini or OpenAI API key (or point at a local
   Klein model) so the app can generate dataset images.
2. **ComfyUI** — unlocks local (Klein) generation and the Test Studio.
3. **Ollama** — the local vision model behind auto-captioning, framing
   auto-classify and head-crop.
4. **Quality tools** — face-similarity scoring, person masks, watermark
   inpainting and bank scoring (a one-click \`pip install\`).
5. **ai-toolkit** — the training engine.

Each optional helper says what it unlocks and what still works without it, and
installs on its own — or all at once, which is usually what you want on a fresh
machine.

<p align="center">
  <img src="../screenshots/setup/install-everything.png" alt="Setup step 4 listing each ML helper with its own Install button and an Install all option" width="760">
</p>

Nothing is mandatory: **Skip setup** is always available, and every step can be
revisited later from **Settings**, where each tool has a Test button that tells
you immediately whether the app can see it.

**Setup is a first run, not a toll gate.** Once the app has seen your install
working — configured, with at least one image engine answering — it stops
sending you to the wizard. Coming back later (a new tab, a new browser, another
machine on your network, or a restarted server) drops you straight into the app,
and the same checks the wizard runs happen in the background while you work. A
short line in the corner says so and then fades.

You are only interrupted when something that *used to* work has stopped —
an API key that no longer answers, an ML helper that no longer imports. The
warning names what broke and links to Setup. It does **not** fire because
ComfyUI or Ollama simply isn't running (you start those on demand), and it does
not fire because something was never installed in the first place. If you
removed a component deliberately, **That was on purpose** stops the app
mentioning it again.

## Around the app

- **Datasets** — the home tab and your **library**: photo tiles of every
  dataset, grouped by model family, with a search box and a badge for each
  family you've already trained. Create one and work it through the guided
  flow (source → curate → caption → train).
- **🏋️ Runs** — every training in one place, cloud *and* local: live progress,
  the settings each launch used, retry a failed run (↻), continue a finished
  one (▶), and download the LoRA (appears once ai-toolkit or a [vast.ai](https://cloud.vast.ai/?ref_id=683073) key is set).
- **Test Studio** — grid-test a trained LoRA across checkpoints and strengths,
  vote, and rank (appears once ComfyUI is reachable).
- **Guide** — this manual.
- **Setup** — the guided wizard, re-runnable anytime.
- **Settings** — everything the wizard configures, plus server, updates,
  maintenance and the diagnostic report.

Next chapter: **Using the app** — the full walkthrough, dataset type by dataset
type.
`;export{n as default};
