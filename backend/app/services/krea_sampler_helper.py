"""Presence of the Krea 2 preset sampler node, for the optional sampling lane.

The node itself is `backend/comfy_nodes/lds_krea_sampler` — code this app ships
and copies into the user's ComfyUI (`setup_installer` action
`krea_sampler_nodes`), rather than a pack fetched from somebody's repository.

Shape and caching mirror `lanpaint_helper` and `krea_edit_helper` on purpose —
one convention across every node the Setup screen can talk about, so the engine
cards and the studio preflight treat them alike. The success-TTL exists because
/object_info is an expensive call on a busy server and a node that was present a
minute ago overwhelmingly still is; only a POSITIVE result is cached, so a
missing node is always re-checked and an install is noticed at once.

WHAT IS DIFFERENT HERE, AND WHY IT MATTERS
------------------------------------------
For a third-party pack, "the folder is there and not empty" is the whole disk
answer. For one we ship, it is not: a folder can be there and hold LAST
version's node. So the disk answer comes from the installer's stamp
(`_bundled_pack_state`), and `krea_sampler_pack_installed()` is true only for a
copy that is CURRENT. A stale copy reports as not-installed on purpose — the
Setup card should offer the button that repairs it, and the boot-time refresh
normally means a user never sees that state at all.

ABSENCE IS NOT AN ERROR
-----------------------
Nothing here gates the Krea lane. The preset sampler is opt-in: with no preset
selected the app's graph never names this class, so an install that never copied
the folder is not broken, it is an install that has not opted in. These probes
exist to answer "can I offer the option?", never "may this render run?".
"""
from __future__ import annotations
import time

# The ONE class the swapped-in graph references. Listing only what the graph
# actually names keeps the probe honest if the folder ever grows a second node.
KREA_SAMPLER_NODE_CLASSES = ('LDSKrea2PresetSampler',)

# The Setup action that installs it. Named here so a caller can turn "missing"
# into a button without importing the installer's private registry; pinned
# against setup_installer._BUNDLED_NODE_PACKS by the contract test.
KREA_SAMPLER_INSTALL_ACTION = 'krea_sampler_nodes'

# What a missing-node hint says. No `url`: unlike the third-party packs, there is
# nowhere to send someone — the code is already on their disk, it just has to be
# copied into ComfyUI, which is what the action above does.
KREA_SAMPLER_NODE_PACK = {
    'pack': 'Krea 2 preset sampler',
    'action': KREA_SAMPLER_INSTALL_ACTION,
}

_NODES_OK_TTL_S = 300
_nodes_ok_until = 0.0


def krea_sampler_missing_nodes():
    """[class_type] the target ComfyUI does not expose. [] when all present OR
    when /object_info is unreachable (fail open — a probe failure is not a verdict)."""
    global _nodes_ok_until
    if time.time() < _nodes_ok_until:
        return []
    from ..utils.comfyui import fetch_object_info_classes
    available = fetch_object_info_classes()
    if available is None:
        return []
    out = sorted(c for c in KREA_SAMPLER_NODE_CLASSES if c not in available)
    if not out:
        _nodes_ok_until = time.time() + _NODES_OK_TTL_S
    return out


def clear_nodes_cache():
    """Drop the success-TTL so the next probe re-asks /object_info. Called right
    after the node is deployed: the cache only ever holds a POSITIVE result, but a
    stale positive would hide a copy the user removed, and clearing costs one probe."""
    global _nodes_ok_until
    _nodes_ok_until = 0.0


def krea_sampler_pack_installed():
    """Is a CURRENT copy of the node in this ComfyUI's custom_nodes? Disk-only.

    This is what separates "install it" from "it is installed, ComfyUI just hasn't
    been restarted yet" — ComfyUI registers nodes at startup, so /object_info keeps
    reporting the class missing until then, and without this distinction the app
    would tell someone to install what they just installed.

    False when ComfyUI's folder isn't configured (we genuinely do not know), when
    nothing is deployed, when what is deployed is a previous version, and when the
    folder carries no stamp of ours (see `_bundled_pack_state`: the installer will
    not overwrite a directory it cannot prove it wrote)."""
    from .. import setup_installer
    try:
        return setup_installer._bundled_pack_state(
            KREA_SAMPLER_INSTALL_ACTION) == 'current'
    except Exception:       # noqa: BLE001 — an unreadable ComfyUI folder is "unknown"
        return False
