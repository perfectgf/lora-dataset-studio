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

Use **Node.js 22.22 or newer**. The repo **ships the frontend prebuilt** in `frontend/dist/` (that folder is committed on purpose — `start.bat` and the Docker/portable builds serve it directly and never run Node). So:

```bash
cd frontend
npm install
npm run dev      # live-reload dev server; proxies /api to the backend on :5050
npm run build    # writes frontend/dist/
```

⚠ **`npm run dev` drives a real backend.** `/api` is proxied to `http://127.0.0.1:5050` — your actual install — and write requests go through. Set `LDS_DEV_API_TARGET` (shell, or `frontend/.env.local`) to point it at a throwaway instance instead; see [frontend/README.md](frontend/README.md).

**If you change anything under `frontend/src`, run `npm run build` and commit the regenerated `frontend/dist/` in the same PR** — otherwise people running from source won't see your change. There's no TypeScript/ESLint step; a clean `npm run build` is the bar.

## Tests

The backend has a large test suite (7000+ tests) and it must stay green:

```bash
pip install -r backend/requirements-dev.txt   # pytest + the two test-only ML extras
# optional: execute the 16 Torch/ai-toolkit bridge runtime tests instead of skipping them
pip install -r backend/requirements-torch-tests.txt
python -m pytest backend/tests -q -n 8 --dist loadfile
```

**Run it in parallel — `-n` is not an optimisation, it is the normal way to
run this suite.** Sequentially it takes about 30 minutes; on 8 workers it takes
about 3-4 minutes. The cost is flat rather than concentrated (the ten slowest
tests are 7.5% of the total; the long tail is 74%), so there is no subset worth
running instead. Keep `--dist loadfile`: it keeps each file's tests on one
worker, which several files depend on. Drop to `-n0` when you want a debugger,
live output, or a readable single-file run.

Prefer a fixed worker count over `-n auto`. A few tests assert on elapsed time
rather than on a result — `test_bank_scan_no_db_lock` gives a concurrent writer
a 400 ms budget — and those get less true the busier the machine is. On a
many-core box `-n auto` saturates every core and they start reporting on the
machine's load instead of on the code. (They can fail sequentially too, on a
loaded machine; parallelism is not what makes them fragile.)

⚠ Pass the `backend/tests` path. A bare `python -m pytest` from the repo root
finds no `pytest.ini` at all (the config lives in `backend/`), so it silently
runs *without* `-p no:flask` — the exact configuration that once made nine
tests pass locally and fail on the release tag.

`requirements-dev.txt` is the common CI/release baseline. CI adds the CPU-only `requirements-torch-tests.txt` overlay when training-state code changes (and for manual runs), while the release job always adds it. A suite green against a different set of packages is not evidence about CI: a stray `pytest-flask` on one dev machine once made nine tests pass locally and fail on the release tag. (The suite runs with `-p no:flask` for that reason; you do not need to uninstall anything.)

This is exactly what CI runs on a release tag, so run it locally before you open a PR. If you add or change behavior, add or update a test for it. The suite mocks external tools (ComfyUI, ai-toolkit, Ollama), so it runs without a GPU or any of those installed.

For the frontend, a successful `npm run build` is the check.

## Before you start: see what is already in flight

This is the part we got wrong for a long time, and a contributor was right to
say so. Work used to land on `main` in one jump, so there was no way to tell
what was already being built — twice, two people built the same feature without
knowing about each other.

So anything non-trivial now lives on a **pushed branch** before it lands, ours
included. Before you start something bigger than a one-file fix:

```
git ls-remote --heads origin      # what is being worked on right now
gh pr list                        # and what is already up for review
```

A branch named for the job means someone is on it. If it looks like yours, say
so in the issue or on Discord — overlapping is easy to sort out beforehand and
miserable to discover afterwards. If nothing matches, it is free.

`main` stays releasable at all times. A branch may be broken while it cooks;
`main` may not.

## Pull requests

- **Keep PRs small and focused** — one change per PR reviews faster and reverts cleanly.
- **Explain the *why*, not just the what.** What problem does this solve? Link the issue or Discord thread if there is one. The feature-request template's framing applies here too: the job you're doing matters more than the mechanism.
- **UI text is English.** Match the tone of what's already there — concrete and direct.
- **Screenshots for any UI change** (before/after if you're changing something that exists).
- **No secrets or local paths.** Don't commit API keys, tokens, `config.json`, `.env`, or absolute paths from your machine — and scrub them from screenshots, logs and PR descriptions too. `.gitignore` already covers the usual suspects; the diagnostic report is built to be paste-safe for the same reason.
- Be kind. See the [Code of Conduct](CODE_OF_CONDUCT.md).

## Already running your own fork?

Several people run a modified copy, and that is exactly what the MIT licence is
for — take it wherever you need it, no permission required, no obligation to
send anything back.

But if you have fixed something along the way that would help everyone, we would
genuinely like it. A fork usually carries two very different kinds of change:

- **Your product decisions** — engines you removed, defaults you disagree with,
  behaviour you shaped around your own workflow. Keep them. They are why you
  forked, and upstreaming them would only make your next merge harder.
- **The fixes underneath them** — a path that only resolved on your machine, a
  message that named the wrong cause, a crash on a layout we never tested. Those
  are ours too, and they are usually a small diff sitting inside a much larger
  branch.

You do not have to disentangle them yourself. **Open an issue describing what you
fixed and point at the commit in your fork** — that is enough. If it is a
one-file change you would rather just send, open the PR straight from your fork
(GitHub keeps the base repo correct by default).

Worth knowing: fixes reported this way get credited to you by name in the commit
and in the release notes, same as anyone else.

## Getting oriented

The product docs double as a map of what the app does and why:

- [`README.md`](README.md) — architecture in one pass, the feature list, run modes, and the **Legal & responsible use** section (please read it — this project has hard lines around real people and consent).
- [`docs/guide/`](docs/guide/) — the in-app manual (getting started, using the app, troubleshooting, getting help).
- [`docs/DATASET_GUIDE.md`](docs/DATASET_GUIDE.md) — how good datasets are built, which informs a lot of the product decisions.

## Security

Found a vulnerability? Please **don't** open a public issue — see [`SECURITY.md`](SECURITY.md) for how to report it privately.
