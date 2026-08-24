# Packaging — Windows release ZIP

The supported Windows release is a source archive named
`LoRA-Dataset-Studio-windows.zip`. It contains the tracked frontend build,
backend, `start.bat`, and the small Python bootstrap script. It does not ship a
prebuilt launcher or an embedded runtime.

Users extract the archive and double-click **`start.bat`**. The launcher finds a
compatible Python already installed or downloads a standalone CPython into the
extracted folder on first launch, creates `.venv`, installs the core requirements,
and opens the browser. Nothing is installed system-wide and administrator rights
are not required.

## Build the release archive

On Windows, either double-click `packaging\build.bat` or run:

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_release_zip.ps1
# -> packaging\dist\LoRA-Dataset-Studio-windows.zip
```

The release workflow runs the same script after the backend/frontend test suites.
It then runs `scripts/check_release_artifacts.py` against the ZIP before uploading
the explicitly named archive. GitHub also supplies its normal source-code archives.

## What the archive check looks for

`scripts/check_release_artifacts.py <zip>` reads every textual member of the ZIP
and reports machine paths, personal e-mail addresses, API tokens and tailnet
addresses, using the same pattern table as `backend/tests/test_no_personal_data.py`
(`scripts/privacy_patterns.py` — one table, imported twice, never copied).

It is aimed at the archive rather than the repo on purpose. The test walks
`git ls-files` and skips `frontend/dist/`, so untracked files, ignored files and
the compiled bundle are invisible to it — that is how `backend/.pytest_cache/`
reached a published release. The ZIP has no such blind spot.

**False positives.** A guard that fails a release over a placeholder gets turned
off, so tolerated shapes are handled by name: UI placeholders such as
`C:\path\to\...` (no `\Users\` segment, so nothing fires), `…\Users\<account>`
and documented stand-in accounts, RFC 2606 documentation domains, and
token-shaped strings that say they are fake. If a real release ever fails, judge
the string first, then add an entry to `ARCHIVE_EXCEPTIONS` at the top of the
script — member glob + label + exact text, with a comment saying why. It is
empty today, and a full build of the current bundle scans all 189 files clean.

The forbidden-*name* half needs a list kept out of the repo (`LDS_PRIVACY_NAMES`
or a gitignored `.privacy-names`). Without one the script says so out loud
rather than passing quietly.

## Release policy

- Publish archives/source only; do not attach an executable launcher.
- Never replace the explicit ZIP path in the workflow with a broad `dist/*` glob.
- Run `python scripts/check_release_artifacts.py` after changing a release workflow.
- Test the extracted archive by double-clicking `start.bat` on a clean Windows VM.

`make_icon.py` is the generator for `packaging/icon.ico` (launcher exe and
window) and the repo-root `icon.png` (the Pinokio tile). Nothing invokes it at
build time - both outputs are committed so the build is reproducible; run
`python packaging/make_icon.py` to regenerate them from the same render when
the tile design changes.

`build_portable.ps1` and `launcher.py` remain in the repository only as a legacy
local developer experiment. They are not invoked by CI, `build.bat`, or the release
workflow, and their output is unsupported and must never be published.
