---
paths:
  - ".github/workflows/**"
  - "frontend/scripts/**"
---

# Release mechanics

Releases are cut on validated waves/milestones only — never per commit, and
always from `main` — never from `nightly`. Development lands on `nightly`;
`main` receives it when the work is done, and a tag is cut from there (see
CLAUDE.md, "Where work lands"). Announcements tell users to "Update & restart".

- The dist-freshness check runs at release time (`release.yml`).
- CI on push gates heavy jobs on big changes (≥5 source files or ≥100 lines —
  see `.github/workflows/ci.yml`).
- `frontend/scripts/releaseNotes.mjs` builds the release body from the
  What's-new entries `frontend/src/whatsNew.js` gained since the previous tag
  (git diff of that file, not entry `date` — several releases can be cut on
  one day). A tag whose body would announce NOTHING fails the release job in
  seconds. A genuine plumbing-only release says so on purpose by carrying
  `[no-notes]` in its annotated tag message.
