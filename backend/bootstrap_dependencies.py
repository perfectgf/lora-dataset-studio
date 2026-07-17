"""Repair dependency installs that are unsafe to import.

The in-app updater -- or a partial ML-extras install that downgrades Pillow to
satisfy a pillow<10 dependency -- can leave Pillow HALF-swapped on Windows: the
core (``Image.py``) at one version while an image plugin is at another. Pillow 12's
``Image.mode`` is a read-only property, so an old plugin that still does
``self.mode = ...`` raises ``property 'mode' has no setter`` the first time it
decodes an image (e.g. PngImagePlugin).

Detect that mix by INSPECTING THE FILES ON DISK, never by trusting pip's metadata:
a rolled-back / interrupted downgrade leaves the ``.dist-info`` reading an OLDER
version than ``Image.py`` actually is, and the previous metadata-gated check
(``version < 12 -> assume fine``) returned early and MISSED exactly that case. The
decision is made from the relationship between ``Image.py`` (read-only ``mode``
property?) and the plugins (``self.mode =``?) -- independent of the version string.
Then force-reinstall the PINNED Pillow before the application imports any PIL module.

``importlib.metadata`` + plain file reads only: importing PIL here would lock its
binaries on Windows and recreate the very failure this bootstrap is meant to fix.
"""
from __future__ import annotations

from importlib import metadata
from pathlib import Path
import re
import subprocess
import sys


# An old-style plugin writes the image mode directly (`self.mode = ...`). The
# negative lookahead is load-bearing: WITHOUT it `\bself\.mode\s*=` also matches
# `self.mode == "P"` -- a COMPARISON that every healthy Pillow 12 plugin (Bmp/Gif/
# Jpeg/...) contains -- which would flag a coherent install and reinstall Pillow on
# every boot. `(?!=)` requires a single `=` (assignment), never `==`.
_OLD_MODE_ASSIGNMENT = re.compile(r"\bself\.mode\s*=(?!=)")
# ...which crashes once Image.mode is a READ-ONLY property (Pillow >= 10.1 / 12):
# a `@property def mode` with NO matching `@mode.setter`.
_MODE_PROPERTY = re.compile(r"@property\s+def\s+mode\b")
_MODE_SETTER = re.compile(r"@mode\.setter\b")

# The Pillow version the app requires (single source of truth). A repair reinstalls
# THIS -- not the possibly-lying metadata version -- so a half-downgraded venv
# converges on what requirements.txt pins instead of whatever the mix left behind.
_REQUIREMENTS = Path(__file__).with_name('requirements.txt')


def _major_version(version: str) -> int:
    match = re.match(r"\s*(\d+)", version or '')
    return int(match.group(1)) if match else 0


def _pinned_pillow_version() -> str | None:
    """The Pillow version pinned in requirements.txt (e.g. '12.2.0'), or None when
    the file can't be read (frozen build / moved layout). Plain file read only."""
    try:
        for raw in _REQUIREMENTS.read_text(encoding='utf-8').splitlines():
            line = raw.split('#', 1)[0].strip()
            m = re.match(r'(?i)pillow\s*==\s*([0-9][^\s;]*)', line)
            if m:
                return m.group(1)
    except OSError:
        pass
    return None


def _image_mode_is_readonly(pil_dir: Path) -> bool | None:
    """Does ``PIL/Image.py`` define ``mode`` as a READ-ONLY property (Pillow
    >= 10.1 / 12)? True/False from the file itself; None when Image.py can't be read
    (the caller then falls back to the metadata version). File read only -- never
    imports PIL, so it can't lock the binaries the repair may need to replace."""
    try:
        src = (pil_dir / 'Image.py').read_text(encoding='utf-8', errors='ignore')
    except OSError:
        return None
    return bool(_MODE_PROPERTY.search(src)) and not _MODE_SETTER.search(src)


def incompatible_pillow_plugins(distribution=None) -> tuple[str | None, list[Path]]:
    """Return Pillow's installed version and any legacy plugins unsafe to import
    against the ``Image.py`` actually on disk.

    The mix is decided by the FILES, not by pip's metadata: if ``Image.py`` exposes
    a read-only ``mode`` property, every plugin still doing ``self.mode = ...`` is
    incompatible -- even when the metadata claims an old (< 10) version, which is
    exactly what a partial downgrade leaves behind (Image.py at 12, ``.dist-info``
    at 9) and what the old version-gated check silently skipped.
    """
    try:
        dist = distribution or metadata.distribution('Pillow')
    except metadata.PackageNotFoundError:
        return None, []

    version = str(dist.version)
    pil_dir = Path(dist.locate_file('PIL'))
    if not pil_dir.is_dir():
        return version, []

    readonly = _image_mode_is_readonly(pil_dir)
    if readonly is None:
        # No readable Image.py -> fall back to the metadata version: a pre-10 Pillow
        # can legitimately assign self.mode, so only inspect plugins from >= 12.
        if _major_version(version) < 12:
            return version, []
        readonly = True
    if not readonly:
        return version, []          # writable-mode Image.py -> `self.mode =` is fine

    incompatible = []
    for plugin in pil_dir.glob('*ImagePlugin.py'):
        try:
            source = plugin.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            continue
        if _OLD_MODE_ASSIGNMENT.search(source):
            incompatible.append(plugin)
    return version, incompatible


def ensure_pillow_consistent(*, distribution=None, runner=None) -> bool:
    """Force-reinstall a mixed Pillow install, returning whether a repair ran."""
    version, plugins = incompatible_pillow_plugins(distribution)
    if not version or not plugins:
        return False

    # Reinstall the PINNED version (what the app actually needs), falling back to the
    # on-disk metadata version only when the pin can't be read. Never reinstall the
    # metadata version blindly: in a mix it can read older than Image.py really is,
    # so `Pillow=={metadata}` would rebuild the WRONG (old) version.
    target = _pinned_pillow_version() or version
    names = ', '.join(path.name for path in plugins)
    print(f'[LDS] mixed Pillow install detected (metadata {version}; {names}); '
          f'repairing to Pillow {target}...', flush=True)
    run = runner or subprocess.run
    result = run(
        [sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-deps',
         f'Pillow=={target}'],
        check=False,
        timeout=900,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f'Pillow repair failed with exit code {result.returncode}. '
            f'Run: {sys.executable} -m pip install --force-reinstall Pillow=={target}'
        )
    print(f'[LDS] Pillow {target} repaired successfully.', flush=True)
    return True
