# Packaging — portable Windows bundle

A one-double-click distribution for non-technical users: **no Python install, no
terminal**. They download a `.zip`, extract it, and run **`LoRA Dataset Studio.exe`**.

## What's in the bundle

```
LoRA Dataset Studio.exe   ← launcher (Tkinter status window: Open / Quit)
python/                   ← standalone CPython + core deps, WITH pip
backend/                  ← Flask app
frontend/dist/            ← prebuilt UI (shipped in the repo)
icon.ico  README.md
data/                     ← created on first run: config.json, .env, datasets, server.log
```

The launcher starts `python\python.exe backend\run.py` (no console), waits for
`/api/health`, opens the browser, and shows a small window with **Open** / **Quit**.
Everything writable lives under `data/` next to the exe, so the bundle is fully
portable — copy the folder anywhere, nothing touches `%APPDATA%` or the registry.

## Why a standalone Python and not a single frozen .exe

The in-app **Setup wizard** installs the optional ML extras at runtime with
`pip install -r backend/requirements-ml.txt` (face scoring, background masks). A frozen
PyInstaller single-exe has **no pip**, so that step would break. Shipping a real
standalone CPython keeps it working: the app runs under `python/python.exe`, so the
wizard's `pip` targets the bundle. The heavy externals stay outside the bundle and are
guided by the wizard:

| In the bundle | Installed later via the Setup wizard |
| --- | --- |
| Flask app + core deps (light) | ML extras (insightface, onnxruntime, rembg, opencv) |
| Prebuilt UI | ComfyUI · ai-toolkit · Ollama + a vision model |

## Build it

On a **build machine** (Windows 10+, a host `python` 3.9–3.12 on PATH, internet):

```powershell
python packaging\make_icon.py        # (re)generate icon.ico — optional, it's committed
powershell -ExecutionPolicy Bypass -File packaging\build_portable.ps1
# -> packaging\dist\LoRA-Dataset-Studio-win64.zip
```

The script fetches the latest [`python-build-standalone`](https://github.com/astral-sh/python-build-standalone)
CPython (no pinned tag, so it never goes stale), installs the core deps into it, stages
the app, builds the launcher exe with PyInstaller, and zips the result. Build artifacts
(`packaging/build/`, `packaging/dist/`) are git-ignored; only the sources
(`launcher.py`, `build_portable.ps1`, `make_icon.py`, `icon.ico`) are committed.

## Distribute

Attach the `.zip` to a **GitHub Release**. Note for users:

- **SmartScreen** may say *"Windows protected your PC — unknown publisher."* That's
  expected for an unsigned exe: **More info → Run anyway**. A code-signing certificate
  (paid) removes the warning — a later add-on, not required to ship.
- Some antivirus engines occasionally false-positive on PyInstaller exes; the launcher
  source is right here (`launcher.py`) for anyone who wants to audit or rebuild it.

## Verified vs. not

`launcher.py` and `build_portable.ps1` are committed and parse-checked. The **full
end-user flow** (extract → double-click → wizard) should be validated on a **clean
Windows machine without Python** before publishing a release.
