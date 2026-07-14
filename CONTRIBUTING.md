# Contributing

Thanks for wanting to make LoRA Dataset Studio better. This is a small, self-hosted, volunteer project — issues, ideas and pull requests are all welcome.

## Before you write code

For anything bigger than a typo or a one-line fix, **talk about it first**. It saves you from building something that's already in progress or that doesn't fit the direction.

- **Discord** ([join](https://discord.gg/j6hnJBFtXE)) — usually the fastest way. Ask in **#help**; float feature ideas in **#feature-requests**; talk implementation in **#dev-chat**. The curated **#community-ideas** board shows what people voted for (the roadmap follows it), and **#roadmap** shows what's shipped and coming.
- **[GitHub issues](https://github.com/perfectgf/lora-dataset-studio/issues)** — bug reports and feature requests. There are templates for both; for a bug, the app can write most of the report for you (**Guide → Getting help → Copy diagnostic report** — it includes version, OS and a log tail, no keys, no paths).

A quick "I'm going to look at X" in an issue or on Discord means nobody duplicates your work.

## Dev setup

You only need the backend to work on backend code. You only need Node to change the frontend.

### Backend

Use **CPython 3.10–3.12**. This matters: the optional ML extras (`insightface`, `onnxruntime`, `numpy<2`, …) publish no wheels for 3.13+, so a venv built on a newer Python (a bare `python`/`py -3` often grabs 3.13/3.14) can't install them. Pick the version explicitly.

```bash
git clone https://github.com/perfectgf/lora-dataset-studio.git
cd lora-dataset-studio

python -m venv .venv                 # on Windows: py -3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt

# optional, only if you're touching face scoring / masks / watermark inpainting:
pip install -r backend/requirements-ml.txt
# optional, only if you're touching the scraper:
pip install -r backend/requirements-scrape.txt

python backend/run.py
```

`run.py` re-execs itself into `.venv` if it exists, so every launch method converges on the same interpreter. On Windows you can instead just double-click **`start.bat`**, which finds (or downloads) a suitable Python, builds the venv, and starts the server on port **5050**.

### Frontend

The repo **ships the frontend prebuilt** in `frontend/dist/` (that folder is committed on purpose — `start.bat` and the Docker/portable builds serve it directly and never run Node). So:

```bash
cd frontend
npm install
npm run dev      # live-reload dev server; proxies /api to the running backend (see frontend/README.md)
npm run build    # writes frontend/dist/
```

**If you change anything under `frontend/src`, run `npm run build` and commit the regenerated `frontend/dist/` in the same PR** — otherwise people running from source won't see your change. There's no TypeScript/ESLint step; a clean `npm run build` is the bar.

## Tests

The backend has a large test suite (950+ tests) and it must stay green:

```bash
python -m pytest backend/tests -q
```

This is exactly what CI runs on a release tag, so run it locally before you open a PR. If you add or change behavior, add or update a test for it. The suite mocks external tools (ComfyUI, ai-toolkit, Ollama), so it runs without a GPU or any of those installed.

For the frontend, a successful `npm run build` is the check.

## Pull requests

- **Keep PRs small and focused** — one change per PR reviews faster and reverts cleanly.
- **Explain the *why*, not just the what.** What problem does this solve? Link the issue or Discord thread if there is one. The feature-request template's framing applies here too: the job you're doing matters more than the mechanism.
- **UI text is English.** Match the tone of what's already there — concrete and direct.
- **Screenshots for any UI change** (before/after if you're changing something that exists).
- **No secrets or local paths.** Don't commit API keys, tokens, `config.json`, `.env`, or absolute paths from your machine — and scrub them from screenshots, logs and PR descriptions too. `.gitignore` already covers the usual suspects; the diagnostic report is built to be paste-safe for the same reason.
- Be kind. See the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting oriented

The product docs double as a map of what the app does and why:

- [`README.md`](README.md) — architecture in one pass, the feature list, run modes, and the **Legal & responsible use** section (please read it — this project has hard lines around real people and consent).
- [`docs/guide/`](docs/guide/) — the in-app manual (getting started, using the app, troubleshooting, getting help).
- [`docs/DATASET_GUIDE.md`](docs/DATASET_GUIDE.md) — how good datasets are built, which informs a lot of the product decisions.

## Security

Found a vulnerability? Please **don't** open a public issue — see [`SECURITY.md`](SECURITY.md) for how to report it privately.
