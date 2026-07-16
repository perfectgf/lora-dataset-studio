# Getting started

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

**Windows (one command):** download `LoRA-Dataset-Studio-windows.zip` from the
[latest release](https://github.com/perfectgf/lora-dataset-studio/releases/latest),
extract it, then double-click `start.bat`. Releases contain an archive/source, not
a prebuilt executable launcher. `start.bat` finds or downloads a compatible Python
(3.10–3.12), creates `.venv`, installs the requirements, and opens the app at
`http://127.0.0.1:5050/`.

**Any OS (manual venv):**

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
python backend/run.py
```

**Docker (API-only):** `cp .env.example .env`, then `docker compose up --build`.

The full install matrix (Windows release ZIP, GPU requirements, external tools)
lives in the README on GitHub.

## The Setup wizard

On first launch you land in **Setup**. It scans your machine automatically and
walks through five steps — each one unlocks a set of features:

1. **Image generation** — add a Gemini or OpenAI API key (or point at a local
   Klein model) so the app can generate dataset images.
2. **ComfyUI** — unlocks local (Klein) generation and the Test Studio.
3. **Ollama** — the local vision model behind auto-captioning, framing
   auto-classify and head-crop.
4. **Quality tools** — face-similarity scoring and person masks (a one-click
   `pip install`).
5. **ai-toolkit** — the training engine.

Nothing is mandatory: **Skip setup** is always available, and every step can be
revisited later from **Settings**, where each tool has a Test button that tells
you immediately whether the app can see it.

## Around the app

- **Datasets** — the home tab and your **library**: photo tiles of every
  dataset, grouped by model family, with a search box and a badge for each
  family you've already trained. Create one and work it through the guided
  flow (source → curate → caption → train).
- **🏋️ Runs** — every training in one place, cloud *and* local: live progress,
  the settings each launch used, retry a failed run (↻), continue a finished
  one (▶), and download the LoRA (appears once ai-toolkit or a vast.ai key is set).
- **Test Studio** — grid-test a trained LoRA across checkpoints and strengths,
  vote, and rank (appears once ComfyUI is reachable).
- **Guide** — this manual.
- **Setup** — the guided wizard, re-runnable anytime.
- **Settings** — everything the wizard configures, plus server, updates,
  maintenance and the diagnostic report.

Next chapter: **Using the app** — the full walkthrough, dataset type by dataset
type.
