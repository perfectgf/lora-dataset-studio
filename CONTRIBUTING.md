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

**If you change anything under `frontend/src`, run `npm run build` and commit the regenerated `frontend/dist/` in the same PR** — otherwise people running from source won't see your change. There's no TypeScript step. Lint the relevant source files with ESLint (config in `frontend/eslint.config.mjs`); CI runs the frontend build and lint.

### Lint

Both halves have a small, correctness-only lint gate that CI runs on every push and PR (the `lint` job). It does not police style — only things that cannot be intended: unused imports and variables, undefined names, a hook called under a condition, a dict literal that writes the same key twice, a bare `except:`.

```bash
pip install -r backend/requirements-dev.txt   # brings the pinned ruff
ruff check .                                  # backend, scripts, packaging — rules in ruff.toml
ruff check . --fix                            # applies the safe fixes

cd frontend && npm run lint                   # ESLint — rules in frontend/eslint.config.mjs
```

## Tests

Use the [verification policy](CLAUDE.md#scope-and-verification): choose the
smallest check that can detect a problem introduced by your change. A docs-only
edit needs a diff/link review. For a trivial code change whose effect is clear
from the diff, review can also suffice. Otherwise choose an existing test for
that behavior, direct execution or a relevant UI interaction.

```bash
# From the repository root: select the relevant file or case.
python -m pytest backend/tests/test_<area>.py -q

# From frontend/: select the relevant module or feature test.
node --test <path-to-test>
```

Always pass a backend test path: it loads `backend/pytest.ini`, including the
plugin exclusions used by CI. Use the existing environment; test dependencies
are in `backend/requirements-dev.txt` if you need to prepare one.

Add a durable test for a meaningful regression or contract. Avoid tests that
only repeat a label, class name or constant without checking a useful behavior.
Frontend module tests live beside the module as `*.test.js`; contracts spanning
several modules live in `frontend/tests/*.test.mjs`.

Do not run both complete suites, all contract tests or both linters for every
commit or PR. Lint relevant source files and build the frontend when needed for
delivery. State what you checked, including any meaningful gaps.

The full suites and linters belong to the
[public release gate](.claude/rules/release-mechanics.md), on the final public
tree. That document contains the full commands and environment requirements.
Reuse valid results through a wave instead of rerunning them at each step.

## Work already in progress

Check relevant open PRs and issues before starting substantial work. Discuss
overlap in the existing thread so contributors can coordinate.

Maintainer work branches and `nightly` live in a private repository. Public
release refs do not show unfinished maintainer work. Contributors use their fork
and a focused PR; maintainers qualify approved changes for public `v2`.
Keep private product history out of public merges and keep `v2` releasable.
The former `main` branch is now `v1`, read-only and no longer maintained.

## Pull requests

- **Keep PRs small and focused** — one change per PR reviews faster and reverts cleanly.
- **Explain the *why*, not just the what.** What problem does this solve? Link the issue or Discord thread if there is one. The feature-request template's framing applies here too: the job you're doing matters more than the mechanism.
- **UI text is English.** Match the tone of what's already there — concrete and direct.
- **Screenshots when useful for review.** One relevant view usually suffices for a small visual change; check responsive widths when the change affects adaptation or overflow.
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
