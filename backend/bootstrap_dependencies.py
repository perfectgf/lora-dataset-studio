"""Repair dependency installs that are unsafe to import.

The in-app updater used to run pip while Flask (and therefore Pillow) was still
loaded.  On Windows, locked Pillow files could then be left half old and half
new: Pillow 12's ``Image.mode`` is read-only, while an old image plugin still
tries to assign ``self.mode``.  Detect that exact mixed install without ever
importing :mod:`PIL`, then reinstall the already-selected Pillow version before
the application imports any of its modules.
"""
from __future__ import annotations

from importlib import metadata
from pathlib import Path
import re
import subprocess
import sys


_OLD_MODE_ASSIGNMENT = re.compile(r"\bself\.mode\s*=")


def _major_version(version: str) -> int:
    match = re.match(r"\s*(\d+)", version or '')
    return int(match.group(1)) if match else 0


def incompatible_pillow_plugins(distribution=None) -> tuple[str | None, list[Path]]:
    """Return Pillow's installed version and legacy plugins mixed into Pillow 12+.

    ``importlib.metadata`` and plain file reads are intentional here. Importing
    PIL before the repair would lock its binaries on Windows and recreate the
    failure this bootstrap is designed to fix.
    """
    try:
        dist = distribution or metadata.distribution('Pillow')
    except metadata.PackageNotFoundError:
        return None, []

    version = str(dist.version)
    if _major_version(version) < 12:
        return version, []

    pil_dir = Path(dist.locate_file('PIL'))
    if not pil_dir.is_dir():
        return version, []

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

    names = ', '.join(path.name for path in plugins)
    print(f'[LDS] mixed Pillow {version} install detected ({names}); repairing…',
          flush=True)
    run = runner or subprocess.run
    result = run(
        [sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-deps',
         f'Pillow=={version}'],
        check=False,
        timeout=900,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f'Pillow repair failed with exit code {result.returncode}. '
            f'Run: {sys.executable} -m pip install --force-reinstall Pillow=={version}'
        )
    print(f'[LDS] Pillow {version} repaired successfully.', flush=True)
    return True
