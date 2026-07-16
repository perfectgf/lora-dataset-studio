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

## Release policy

- Publish archives/source only; do not attach an executable launcher.
- Never replace the explicit ZIP path in the workflow with a broad `dist/*` glob.
- Run `python scripts/check_release_artifacts.py` after changing a release workflow.
- Test the extracted archive by double-clicking `start.bat` on a clean Windows VM.

`build_portable.ps1` and `launcher.py` remain in the repository only as a legacy
local developer experiment. They are not invoked by CI, `build.bat`, or the release
workflow, and their output is unsupported and must never be published.
