---
paths:
  - ".github/workflows/**"
  - "frontend/scripts/**"
---

# Public release mechanics

Read this when preparing a public publication or changing release tooling.
These gates do not apply to ordinary private development.

## Qualify the public tree

- Publish validated waves from public `v2`, never from private `nightly`.
  Verify approved provenance first: private product history cannot be merged
  into public history, even after deleting its files.
- On the final qualified public tree, require both full suites and both linters
  before pushing. Reuse results that still cover this tree; recheck changed
  behavior after a conflict or fix. Do not repeat a valid run for a new SHA alone.
- From the repository root: `python -m pytest backend/tests -q -n 8 --dist loadfile`
  and `ruff check backend scripts packaging`.
- From `frontend/`: `node --test` and `npx eslint .`.
  Rebuild only if changed frontend inputs are not already represented in dist.
- Use the repository test dependencies (`backend/requirements-dev.txt`).
  Release validation also uses the CPU Torch overlay
  `backend/requirements-torch-tests.txt`, as defined in `release.yml`.
  Keep the explicit `backend/tests` path to load `backend/pytest.ini`.
- Use a short task-owned `--basetemp` if needed, and a fixed worker count
  rather than `-n auto` locally. Diagnose a failed test in isolation; do not
  dismiss a worker crash as a pass or automatically replay all suites.

## Tag and release

- A tag must match `APP_VERSION`. `release.yml` runs on the tag, executes its
  own suites and rebuilds the packaged frontend. Do not add another manual
  full run at tagging time.
- The workflow reports tracked-dist/toolchain drift; the ZIP uses its fresh
  build. Check the current workflow when changing packaging or gates.
- `frontend/scripts/releaseNotes.mjs` uses What's-new entries added since the
  last published release.
- The release tag is lightweight, on the qualified commit: the pre-push guard
  approves exact commit IDs and refuses an annotated tag object ("without tag
  indirection"). A deliberately notes-free plumbing release therefore puts
  `[no-notes]` in the tagged commit's message: `release.yml` reads
  `git tag -l --format='%(contents)'`, which returns that message for a
  lightweight tag.
- Wait for successful release publication before announcing availability.
  A pushed tag alone is not a published release.
