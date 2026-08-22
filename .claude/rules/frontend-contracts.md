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

**So a change that touches layout is not verified until the probe has run:**

```
cd frontend && npm run probe:responsive -- --url http://127.0.0.1:5173/#/canvas
```

It renders the page at 360/412/768/904/1024/1280 px, opens what can be opened,
and measures: **overflow** (nothing past the right edge), **budget** (the fixed
chrome may not eat the fold — 28 % at rest, 50 % with the ⋯ shelf open),
**fill** (a panel row must use ≥ 35 % of its width, which is what catches "the
box is huge and empty"), **targets** (40 px below `lg`) and **overlap**.

- Exit codes are three: `0` clean, `1` violations, `2` could not run. A probe
  that cannot run must never read as a pass.
- Surfaces are found by `data-probe-chrome` / `data-probe-panel` /
  `data-probe-world`, not by class — restyling cannot silently take a pill out
  of scope. `canvasProbeMarkers.test.js` fails if a marker or a threshold goes.
- A breach of the budget is fixed by taking something OUT of the panel, not by
  raising the number.

## Commits & dist

- **Source-only commits.** Never commit `frontend/dist/**` alongside sources;
  the dist rebuild is a separate consolidated `build(frontend):` commit at the
  end of the wave.
- Frontend tests: `node --test` from `frontend/` — includes the help-registry
  and what's-new contract tests.

## Stable identifiers

Never rename catalog labels, config keys or What's-new ids without an alias
path — several are stored in user databases and localStorage.
