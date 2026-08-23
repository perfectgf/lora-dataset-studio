"""Helpers shared by more than one route blueprint."""
import json
import logging

from flask import jsonify, request
from sqlalchemy.exc import IntegrityError

from .. import capabilities
from ..gpu_window import GpuBusyError
from ..job_queue import (ComfyUIRecoveryRequired, auto_resolve_comfyui_barrier,
                         require_comfyui_enqueue_ready)
from ..services.vision_ollama import LocalOllamaFenceError

logger = logging.getLogger(__name__)


# Every write route reads its body with `get_json(silent=True) or {}`: a body that
# does not parse becomes an EMPTY one, so the route runs with no input, saves, and
# answers 200 while the value the caller sent is simply absent. The natural way to
# hit it is a raw Windows path in a hand-written body (`"C:\ai-toolkit"` — `\a` is
# not a JSON escape), which is exactly how it was reported: a settings save that
# said OK and stored nothing.
#
# Rather than convert ~110 call sites (and depend on the next one remembering),
# the parse is made strict once, at the request boundary: an unreadable body never
# reaches a view, so it can never be mistaken for an empty one. The call sites keep
# their `or {}` — past this guard it only means "no body", which stays legitimate.
_STRICT_JSON_METHODS = frozenset({'POST', 'PUT', 'PATCH', 'DELETE'})

# The two previews that DELIBERATELY degrade instead of refusing: both carry
# unsaved editor state and answer while the user is mid-keystroke, where showing
# the currently-effective value beats showing 'error' (see their docstrings).
# Anything that persists or starts work is strict — no exceptions.
_TOLERANT_JSON_ENDPOINTS = frozenset({
    'settings.post_prompt_preview',      # composed prompt for one shot
    'bank.bank_flag_preview',            # flag counts for unsaved thresholds
})


def reject_unparsable_json_body():
    """Refuse an /api write whose body is present but is not valid JSON (400).

    Registered as an app-wide ``before_request``. Multipart uploads are skipped
    (their body is files, not JSON) and an ABSENT body is skipped — many POSTs
    legitimately carry none. Everything else that arrives on a write method is
    body the caller meant us to read, so failing to read it must be said out loud
    and must leave every file untouched.
    """
    if request.method not in _STRICT_JSON_METHODS:
        return None
    if not request.path.startswith('/api/'):
        return None
    if request.endpoint in _TOLERANT_JSON_ENDPOINTS:
        return None
    if (request.mimetype or '').lower().startswith('multipart/'):
        return None
    raw = request.get_data(cache=True).strip()
    if not raw:
        return None
    if request.mimetype != 'application/json' and not raw.startswith((b'{', b'[')):
        # Not declared as JSON and not shaped like it: a genuine form body
        # (`engine=flux&batch_id=3` — /dataset/<id>/ref/edit/keep still accepts
        # one, and studio's uploads fall back to multipart). Accusing it of being
        # invalid JSON would be a fresh bug, not a fix. A body that DOES start
        # like JSON was meant as JSON whatever header carried it — `curl -d`
        # sends form-urlencoded by default, which is precisely how a
        # hand-written body arrives here.
        return None
    try:
        json.loads(raw.decode('utf-8'))
    except UnicodeDecodeError:
        return jsonify({'error': 'Invalid JSON body: it is not valid UTF-8 text.'}), 400
    except ValueError as e:
        msg = f'Invalid JSON body: {e}'
        if b'\\' in raw:
            # The overwhelmingly common cause on this app's audience, and the one
            # the error alone never explains: a pasted Windows path.
            msg += (' — backslashes in Windows paths must be escaped '
                    r'("C:\\ai-toolkit", not "C:\ai-toolkit").')
        return jsonify({'error': msg}), 400
    return None


def _map_error(e: Exception):
    """Map a service/vision exception to a Flask (body, status) tuple.
    Unrecognized exceptions are re-raised (-> 500, a real bug)."""
    if isinstance(e, GpuBusyError):
        return jsonify({'error': 'GPU busy', 'detail': str(e)}), 503
    if isinstance(e, ValueError):
        return jsonify({'error': str(e)}), 400
    if isinstance(e, LocalOllamaFenceError):
        # The one refusal that carries its own remedy: the model in the way can
        # be unloaded with the user's consent, and it often goes away on its own
        # (Ollama's idle unload). A code — not a sentence to string-match —
        # tells the UI it may offer both instead of ending the story here.
        return jsonify({'error': str(e), 'code': 'ollama_fence_blocked'}), 409
    if isinstance(e, RuntimeError):
        return jsonify({'error': str(e)}), 409
    if isinstance(e, IntegrityError):
        # A referential-integrity conflict (e.g. deleting a row a legacy DB still
        # references without ON DELETE CASCADE). The service belt should keep this
        # from happening, but map it to a clear 409 rather than a bare 500 as a
        # last resort — and log the raw error so a real defect is never swallowed.
        logger.warning('database integrity error: %s', e, exc_info=True)
        return jsonify({'error': 'This action conflicts with related data that '
                                 'still references it. Please retry; if it keeps '
                                 'failing, restart the app and report it.'}), 409
    raise e


def _require_comfyui(*, force=False):
    """None if ComfyUI is reachable, else the (body, status) 409 to return.
    Shared by studio.py and datasets.py's lora-test routes that actually enqueue
    a ComfyUI job (run/resume) — read-only/history/DB-only routes stay ungated."""
    comfy = capabilities.probe(force=force)['comfyui']
    if not comfy['reachable']:
        # Two causes, two sentences: "not reachable / check the URL" was returned
        # for a ComfyUI that was up and merely slow to enumerate itself, which sent
        # the user to re-check a URL that was correct. capabilities publishes WHICH
        # it is; the wording lives there so this 409 and the engine cards agree.
        slow = comfy.get('status') == 'slow'
        return jsonify({'error': ('ComfyUI is answering too slowly' if slow
                                  else 'ComfyUI is not reachable'),
                        'hint': comfy.get('hint') or 'Check the URL in Settings'}), 409
    return None


def _require_no_stalled_comfyui():
    """Structured route guard for the durable ComfyUI recovery barrier.

    The moment of refusal is also the best moment to check whether the barrier
    is still real: the user is here, ComfyUI is presumably back (that is what
    the message told them to do), and a prompt id ComfyUI no longer knows is
    proof the remote job is gone. So try the provable clear once before
    refusing — the barrier survives every unprovable case untouched, which is
    what keeps this safe.
    """
    try:
        require_comfyui_enqueue_ready()
    except ComfyUIRecoveryRequired as first:
        e = first
        if auto_resolve_comfyui_barrier() is not None:
            try:
                require_comfyui_enqueue_ready()
                return None
            except ComfyUIRecoveryRequired as retried:
                e = retried  # another barrier appeared meanwhile: still refuse
        return jsonify({
            'ok': False,
            'code': 'comfyui_recovery_required',
            'error': str(e),
        }), 409
    return None


_STUDIO_FAMILY_LABELS = {'zimage': 'Z-Image', 'sdxl': 'SDXL', 'krea': 'Krea 2 Turbo',
                         'flux': 'FLUX.1', 'flux2klein': 'FLUX.2 Klein'}


def _studio_missing_response(e):
    """Turn a StudioAssetsMissing into a structured 409 (same spirit as Klein's
    missing-models 409): a human message + the itemized file/node lists the front
    lists in a banner, so the user knows WHY the grid can't run instead of watching
    every tile fail silently.

    No auto-download: unlike Klein's public assets, Studio bases / VAEs / text
    encoders are large and often license-gated, and the missing custom nodes aren't
    files at all — a clear 'place X here / install node Y' is the P0 contract.
    Shared by the per-dataset run and the comparison run."""
    fam = _STUDIO_FAMILY_LABELS.get(e.family, e.family)
    invalid = getattr(e, 'invalid_files', None) or []
    msg = f"The {fam} test pipeline can't run. "
    if e.missing_files:
        msg += (f"{len(e.missing_files)} required model file(s) are missing — place "
                f"them at the shown path(s) inside your ComfyUI folder. ")
    if invalid:
        # Present-but-unloadable: the file is on disk but is not real weights (e.g. an
        # HTML licence-gate page saved as .safetensors, or a truncated download).
        msg += (f"{len(invalid)} model file(s) are present but not real weights "
                f"(an HTML download page saved as .safetensors, or a truncated "
                f"download) — delete and re-download them. ")
    node_packs = []
    if e.missing_nodes:
        # Name WHAT to install for the nodes we recognise (pack + ComfyUI-Manager
        # search term + URL), so the user doesn't have to reverse-map a class_type.
        from ..services.lora_test_studio import studio_missing_node_hints
        node_packs = studio_missing_node_hints(e.missing_nodes)
        msg += f"{len(e.missing_nodes)} custom node(s) are missing — install them into ComfyUI. "
        for h in node_packs:
            msg += (f"For “{h['class_type']}”, install “{h['pack']}” via ComfyUI-Manager "
                    f"(search “{h['search']}”: {h['url']}). ")
    msg += "Then relaunch the test."
    return jsonify({'ok': False, 'error': msg,
                    'studio_missing': {'family': e.family,
                                       'files': e.missing_files,
                                       'invalid': invalid,
                                       'nodes': e.missing_nodes,
                                       'node_packs': node_packs}}), 409


def _studio_arch_mismatch_response(e):
    """Turn a StudioArchMismatch into a structured 409 (same spirit as
    _studio_missing_response): a selected checkpoint's REAL architecture, read
    from its header, is not the Studio family's — ComfyUI would silently drop it
    and render every tile as if the LoRA were off. Tell the user WHICH Studio /
    family the file actually belongs to instead of letting the grid run blank."""
    fam = _STUDIO_FAMILY_LABELS.get(e.family, e.family)
    det = _STUDIO_FAMILY_LABELS.get(e.detected, e.detected)
    name = (e.checkpoint or '').replace('\\', '/').rsplit('/', 1)[-1]
    msg = (f"“{name}” is a {det} LoRA, but this is the {fam} Studio — "
           f"ComfyUI would silently drop it and every tile would render as if the "
           f"LoRA were off. Test it in the {det} Studio, or re-deploy it under the "
           f"{det} family.")
    return jsonify({'ok': False, 'error': msg,
                    'studio_arch_mismatch': {'family': e.family,
                                             'detected': e.detected,
                                             'checkpoint': e.checkpoint}}), 409
