"""Where the InsightFace (antelopev2) weights live.

Every other engine in this app places its own weights: ``bank_semantic`` and
``watermark_detect`` both resolve an empty ``models_root`` to a folder under the
data directory. Face work was the exception — it passed ``root=`` only when the
user had configured one, so insightface fell back to its OWN default,
``~/.insightface``, and downloaded ~350 MB there (~750 MB on disk: the line that
would delete the zip after extracting it is commented out upstream).

That default is invisible on a native install, where the home directory is
permanent, and fatal in Docker: no Compose file mounts the container user's home
(``/root`` for the API-only image, ``/home/comfy`` for the GPU one — upstream's
``useradd -d /home/comfy``), and the Windows launcher restarts a STOPPED
container with ``--force-recreate`` (scripts/docker-launch.ps1), which replaces
the container and discards its writable layer. So the pack was re-downloaded on
every restart, while the ML venvs — which live under ``data/envs`` — survived.

An install that ALREADY holds the pack under ``~/.insightface`` keeps using it.
Moving those files could break another tool sharing that folder, and copying
them would spend 750 MB fixing a path that was never broken there: on a native
install the home directory persists, which is the only property this module
cares about.
"""
from __future__ import annotations

import os

from .. import config as cfg

# The pack every face path asks FaceAnalysis for, and the five models it ships:
# detection, recognition, both landmark models and genderage.
PACK = 'antelopev2'
PACK_MODELS = ('1k3d68', '2d106det', 'genderage', 'glintr100', 'scrfd_10g_bnkps')


def _pack_present(root) -> bool:
    """True when ``root`` holds the COMPLETE pack, in either layout.

    Complete, not "has an .onnx". ``extractall`` writes the five models one by
    one, so a run killed mid-extraction leaves a folder with some of them — and
    insightface skips the download whenever that folder EXISTS, which is how a
    half-unzipped pack survives forever. Counting one file would let such a
    folder outrank a complete pack sitting elsewhere, and answer "installed"
    about a root where FaceAnalysis raises ``'detection' in self.models``.

    Either layout: antelopev2.zip carries a root folder, so a fresh download
    lands nested one level too deep and the workers flatten it on load
    (infer/face_score_infer._repair_nested_antelopev2). Nested is downloaded.
    """
    outer = os.path.join(str(root), 'models', PACK)
    for base in (outer, os.path.join(outer, PACK)):
        if all(os.path.isfile(os.path.join(base, f'{m}.onnx')) for m in PACK_MODELS):
            return True
    return False


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
    try:
        managed = str(cfg.data_dir() / 'models' / 'insightface')
    except OSError:
        # data_dir() creates the folder on demand and can fail (permissions, a
        # full disk, a volume that went away). watermark_detect answers None
        # here; we cannot — None is the very default that sent the pack into an
        # unmounted home. The home directory is the one root that always exists.
        return legacy_root()
    if not _pack_present(managed) and _pack_present(legacy_root()):
        return legacy_root()
    return managed
