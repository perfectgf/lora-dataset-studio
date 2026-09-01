"""Where the InsightFace (antelopev2) weights live.

Every other engine in this app places its own weights: ``bank_semantic`` and
``watermark_detect`` both resolve an empty ``models_root`` to a folder under the
data directory. Face work was the exception — it passed ``root=`` only when the
user had configured one, so insightface fell back to its OWN default,
``~/.insightface``, and downloaded ~350 MB there.

That default is invisible on a native install, where the home directory is
permanent, and fatal in Docker: no Compose file mounts the container user's home
(``/root`` for the API-only image, ``/home/comfy`` for the GPU one — upstream's
``useradd -d /home/comfy``), and the Windows launcher restarts a STOPPED
container with ``--force-recreate`` (scripts/docker-launch.ps1), which replaces
the container and discards its writable layer. So the pack was re-downloaded on
every restart, while the ML venvs — which live under ``data/envs`` — survived.

An install that ALREADY holds the pack under ``~/.insightface`` keeps using it.
Moving those files could break another tool sharing that folder, and copying
them would spend 350 MB fixing a path that was never broken there: on a native
install the home directory persists, which is the only property this module
cares about.
"""
from __future__ import annotations

import glob
import os

from .. import config as cfg

# The pack every face path asks FaceAnalysis for (detection + recognition +
# landmarks + genderage). One name, because one absent pack is what makes the
# difference between "cached" and "download 350 MB again".
PACK = 'antelopev2'


def _pack_present(root) -> bool:
    """True when ``root`` already holds the pack, in EITHER layout.

    insightface 0.7.3 ships antelopev2.zip with a root folder inside, so a fresh
    auto-download lands nested one level too deep; the workers flatten it on load
    (infer/face_score_infer._repair_nested_antelopev2). Both layouts count as
    present here — the flattening happens after this resolver has already chosen
    a root, and a nested pack is a downloaded pack. The test is .onnx FILES, not
    the folder: insightface skips the download whenever the directory exists,
    which is exactly how a half-unzipped pack survives forever.
    """
    outer = os.path.join(str(root), 'models', PACK)
    return bool(glob.glob(os.path.join(outer, '*.onnx'))
                or glob.glob(os.path.join(outer, PACK, '*.onnx')))


def legacy_root() -> str:
    """insightface's own default — where every install made before this module
    put its pack, and where a native install may still legitimately keep it."""
    return os.path.join(os.path.expanduser('~'), '.insightface')


def models_root() -> str:
    """The root handed to FaceAnalysis, never empty.

    A configured value wins and is passed through VERBATIM — it is the user's
    path, and normalising it (``str(Path(...))`` turns ``C:/x`` into ``C:\\x``)
    would change what reaches the child for no benefit. Otherwise the data
    directory, unless the pack already sits in insightface's default and not in
    ours. A string rather than a Path for that same reason.
    """
    configured = str(cfg.get('face_scoring.models_root') or '').strip()
    if configured:
        return configured
    managed = str(cfg.data_dir() / 'models' / 'insightface')
    if not _pack_present(managed) and _pack_present(legacy_root()):
        return legacy_root()
    return managed
