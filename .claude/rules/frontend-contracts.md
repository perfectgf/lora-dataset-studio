---
paths:
  - "frontend/**"
---

# Frontend contracts (enforced by tests)

## 🎁 What's new (`frontend/src/whatsNew.js`)

Prepend one **benefit-first** entry per user-visible feature or fix. Between
releases this panel is the ONLY way users learn something shipped. Plumbing and
refactors don't need one — bugfixes of unreleased work don't either.

Release notes are built from the entries this file gained since the previous
tag (`frontend/scripts/releaseNotes.mjs`, git diff of the file — not entry
`date`). Skipping an entry costs a release, not just a panel line.

## Help registry (`frontend/src/help/helpRegistry.js`)

Any new setting, section, page or big button needs a topic (and its Guide
anchor), or the contract test fails.

## Responsive: the source tests cannot see a layout

`canvasResponsive.test.js` and its siblings read the JSX as **text** and match
class names. That is all `node --test` can do — it parses no JSX and renders
nothing — so those assertions prove a class is WRITTEN, never that the screen
works. Three responsive regressions shipped through that gap in one week, each
one found by a person holding a phone.

**So a change that touches layout is not verified until the probe has run** —
on the page the change touches, against an instance that holds something to
open (a bank, a dataset, LoRA files for the Studio):

```
cd frontend && npm run probe:responsive -- --url http://127.0.0.1:5173/#/canvas
cd frontend && npm run probe:responsive -- --url http://127.0.0.1:5173/#/bank
cd frontend && npm run probe:responsive -- --url http://127.0.0.1:5173/#/datasets
cd frontend && npm run probe:responsive -- --url http://127.0.0.1:5173/#/dataset/studio/<id>
```

It renders the page at five REAL device sizes (360×800, 412×915, a phone held
sideways at 844×390, 768×1024, 1280×800), opens what can be opened — each
state on a fresh load — and measures: **overflow** (nothing past the right
edge), **budget** (the fixed chrome may not eat the fold — 28 % at rest, 50 %
with something deliberately open), **fill** (a panel row must use ≥ 35 % of its
width, which is what catches "the box is huge and empty"), **truncation** (a
label cut by its own box), **targets** (40 px below `lg`) and **overlap**.

- Exit codes are three: `0` clean, `1` violations, `2` could not run. A probe
  that cannot run must never read as a pass. The coverage report at the bottom
  names what was actually reached — a state whose control is absent on this
  instance is SKIPPED and said so, never counted as clean.
- Surfaces are found by attribute, never by class — restyling cannot silently
  take a pill out of scope:
  - `data-probe-chrome="<name>"` — fixed chrome, budgeted and paired for
    overlap; only VISIBLE chrome counts (a marker inside a closed `<details>`
    still has a box). The budget detail names each surface's height.
  - `data-probe-panel="<name>"` — rows measured for fill.
  - `data-probe-world` — a pannable surface whose contents are not overflow.
  - `data-probe-reading` — opened to READ (the Bank's passes panel): chrome in
    the overlap check, NOT budgeted.
  - `data-probe-layer` — a lightbox, drawer or sheet that covers the page BY
    DESIGN: not budgeted, paired with nothing in the overlap check.
- Bank and Datasets open on a LIST; the probe's `prime` clicks the first item
  once per viewport (`aria-label="Open the bank …"` / `"Open the dataset …"`)
  and the app's own localStorage keeps the workspace open for every state
  after. Rename those labels and the probe measures an empty list.
- `canvasProbeMarkers.test.js`, `bankProbeMarkers.test.js`,
  `datasetProbeMarkers.test.js` and `studioProbeMarkers.test.js` fail if a
  marker, a prime selector or a threshold goes.
- A breach of the budget is fixed by taking something OUT of the panel, not by
  raising the number. A target under 40 px is fixed with `min-h-10 lg:min-h-0`
  (finger-sized below `lg`, unchanged on a desktop), not by exempting it.

## Commits & dist

- **Source-only commits.** Never commit `frontend/dist/**` alongside sources;
  the dist rebuild is a separate consolidated `build(frontend):` commit at the
  end of the wave.
- Frontend tests: `node --test` from `frontend/` — includes the help-registry
  and what's-new contract tests.

## Stable identifiers

Never rename catalog labels, config keys or What's-new ids without an alias
path — several are stored in user databases and localStorage.
