"""LanPaint node-pack awareness, for the masked Repair lane.

LanPaint (github.com/scraed/LanPaint, GPL-3.0 — installed into the USER'S
ComfyUI like every custom node there, ComfyUI itself being GPL; this app only
names its node class in a workflow sent over HTTP) is a training-free
inpainting SAMPLER (TMLR 2025): instead of
asking the model to honour a mask it was never trained on, the sampler itself
runs a few "thinking" iterations of Langevin dynamics per step, which keeps the
unmasked region consistent and generates genuinely new content inside the mask.
That is what replaced InpaintModelConditioning in klein_mask_inpaint.json:
Klein is an EDIT model, not an inpaint-trained checkpoint, and conditioning it
the Fill-model way produced the smeared results GitHub #43 reported.

Shape and caching mirror krea_edit_helper's node probes on purpose — one
convention, three packs (krea, seedvr2, lanpaint), so the Setup screen and the
studio preflight can treat them alike. The success-TTL exists because
/object_info is an expensive call on a busy server and a pack that was present
a minute ago is overwhelmingly still present; only a POSITIVE result is cached,
so a missing pack is always re-checked and an install is noticed at once.
"""
from __future__ import annotations
import os
import time

from .. import config as cfg

# The ONE node class the masked-inpaint graph uses from the pack. Listing only
# what the shipped graph references keeps the probe honest: a pack update that
# renames other nodes must not fail a graph that never used them.
LANPAINT_NODE_CLASSES = ('LanPaint_KSampler',)

# Same contract as setup_installer._NODE_PACKS['lanpaint_nodes'] (a test pins
# the two together): constant URL, clone-or-zip, installed into
# <validated ComfyUI>/custom_nodes/<folder>. The pack declares zero pip
# dependencies (pyproject checked 2026-08-25, v2.1.0), so a clone is enough.
LANPAINT_NODE_PACK = {
    'pack': 'LanPaint',
    'url': 'https://github.com/scraed/LanPaint',
    'search': 'LanPaint',
}

_NODES_OK_TTL_S = 300
_nodes_ok_until = 0.0


def lanpaint_missing_nodes():
    """[class_type] of the LanPaint nodes the target ComfyUI does not expose.
    [] when they are all present OR when /object_info is unreachable."""
    global _nodes_ok_until
    if time.time() < _nodes_ok_until:
        return []
    from ..utils.comfyui import fetch_object_info_classes
    available = fetch_object_info_classes()
    if available is None:
        return []
    out = sorted(c for c in LANPAINT_NODE_CLASSES if c not in available)
    if not out:
        _nodes_ok_until = time.time() + _NODES_OK_TTL_S
    return out


def clear_nodes_cache():
    """Drop the success-TTL so the next probe re-asks /object_info. Called right
    after the node pack is installed: the cache only ever holds a POSITIVE
    result, but a stale positive would hide a pack the user removed, and
    clearing costs one probe."""
    global _nodes_ok_until
    _nodes_ok_until = 0.0


def lanpaint_node_pack_installed():
    """Is the pack's folder present in this ComfyUI's custom_nodes? Disk-only.

    Separates "you have to install the pack" from "the pack is installed,
    ComfyUI just hasn't been restarted yet" — ComfyUI registers nodes at
    STARTUP, so /object_info keeps reporting them missing until then, and
    without this distinction the app would tell someone to install what they
    just installed. False whenever ComfyUI's folder isn't configured/valid: we
    then genuinely do not know."""
    from .. import capabilities
    r = capabilities.resolve_comfyui_base(cfg.get('comfyui.base_dir') or '')
    if not r['valid']:
        return False
    folder = os.path.join(r['resolved'], 'custom_nodes', LANPAINT_NODE_PACK['pack'])
    try:
        return os.path.isdir(folder) and any(os.scandir(folder))
    except OSError:
        return False
