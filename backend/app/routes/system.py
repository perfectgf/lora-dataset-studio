"""🖥️ System pickers — server-side folder selection for the Browse… field.

The folders the app works on live on the machine RUNNING the server (not the
browser's), so these two endpoints do the selecting there:

  POST /api/system/pick-folder   pops the OS-native dialog on the server desktop.
  GET  /api/system/list-folders  read-only in-app browser (drives + subfolders).

It also hosts the app-wide ComfyUI recovery surface (GET/POST
/api/system/comfyui-recovery…), which belongs here for the same reason: it is
about the machine running the server, not about any one dataset.

pick-folder never 500s on the expected "no desktop here" case — it answers 200
with {available:false} so the UI silently falls back to the in-app browser
(LAN/tablet/Linux) instead of flashing a scary error toast.
"""
import json
import logging

from flask import Blueprint, jsonify, request

from ..config import LOCAL_USER
from ..services import folder_picker
from ._common import _map_error, _require_comfyui

logger = logging.getLogger(__name__)

bp = Blueprint('system', __name__, url_prefix='/api/system')


@bp.post('/pick-folder')
def pick_folder():
    data = request.get_json(silent=True) or {}
    initial = (data.get('initial') or '').strip() or None
    try:
        path = folder_picker.open_native_folder_dialog(initial)
    except folder_picker.NativePickerUnavailable as e:
        # Expected on a headless / Linux / service-session server: 200 so the
        # front falls back to the in-app browser without an error toast.
        return jsonify({'available': False, 'reason': str(e)})
    if path is None:
        return jsonify({'available': True, 'cancelled': True})
    return jsonify({'available': True, 'path': path})


@bp.get('/list-folders')
def list_folders():
    path = request.args.get('path') or None
    try:
        return jsonify(folder_picker.list_subfolders(path))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except PermissionError:
        return jsonify({'error': 'Permission denied for this folder.'}), 403


# --- ComfyUI recovery, app-wide ---------------------------------------------
# The durable recovery barrier is global: one stalled prompt blocks EVERY local
# generation in the app. Its resolution used to be reachable from exactly one
# place — the Stop button of the dataset that happened to own the job — so a
# user working anywhere else met a refusal with no visible way out. These two
# endpoints are the app-wide surface: what is stuck, and one confirmed action
# to clear it, wherever the user happens to be standing.


def _stalled_job_facts(job_id):
    """(stalled_since, metadata) for the barrier's queue row, best effort."""
    from ..models import ImageGenerationQueue
    row = (ImageGenerationQueue.query
           .filter_by(job_id=str(job_id))
           .with_entities(ImageGenerationQueue.last_heartbeat,
                          ImageGenerationQueue.started_at,
                          ImageGenerationQueue.created_at,
                          ImageGenerationQueue.job_metadata).first())
    if row is None:
        return None, {}
    when = row.last_heartbeat or row.started_at or row.created_at
    try:
        metadata = json.loads(row.job_metadata or '{}')
    except (TypeError, ValueError):
        metadata = {}
    return (when.isoformat() + 'Z' if when else None,
            metadata if isinstance(metadata, dict) else {})


def _dataset_name(dataset_id):
    if dataset_id is None:
        return None
    from ..services import face_dataset_service as fds
    try:
        ds = fds.get_dataset(LOCAL_USER, int(dataset_id))
    except (TypeError, ValueError):
        return None
    return getattr(ds, 'name', None) if ds else None


def _comfyui_connection(link):
    """Can LDS talk to ComfyUI at all, and at which address — for the banner.

    A barrier says a job is paused. It says nothing about whether LDS and
    ComfyUI are even in touch, and that silence is what made a fresh install
    unreadable: the URL was auto-detected, the files were found, the first
    Generate answered "a paused ComfyUI job is blocking new generations", and
    ComfyUI logged no incoming connection at all (jerkyjunky, Discord). The
    person went looking for a flag they had to pass; the truth was that LDS was
    knocking on a door nobody was behind.

    `link` is the verdict the recovery probe ALREADY produced this poll (it had
    to read /queue to decide whether to auto-clear), so the common case costs
    nothing extra. Only a barrier it could not ask about — an unconfirmed
    submission, which has no prompt id — falls back to the app's shared ComfyUI
    reachability read, the same one the engine cards and the enqueue 409 use, so
    the two surfaces cannot contradict each other.
    """
    from .. import capabilities, config as cfg
    from ..job_queue import COMFYUI_LINK_REACHABLE, COMFYUI_LINK_UNREACHABLE
    from ..utils.redact import redact_url_secrets
    url = redact_url_secrets((cfg.get('comfyui.api_url') or '').rstrip('/')) or ''
    if link == COMFYUI_LINK_REACHABLE:
        return {'reachable': True, 'url': url, 'status': 'ok', 'hint': None}
    if link == COMFYUI_LINK_UNREACHABLE:
        return {'reachable': False, 'url': url, 'status': 'unreachable', 'hint': None}
    comfy = capabilities.probe_comfyui()
    status = comfy.get('status')
    # 'slow' is NOT a broken link: the server accepted the connection and is
    # merely slow to answer. Sending that user to re-check a correct URL is the
    # exact mistake `comfyui_down_message` exists to prevent.
    reachable = status in ('ok', 'slow')
    return {'reachable': reachable, 'url': url, 'status': status,
            'hint': None if reachable else (comfy.get('hint') or None)}


def _recovery_snapshot():
    """What is blocking generation right now, or None when nothing is.

    `connection` is left None here and filled by the polling GET alone: the POST
    below needs only the barrier's identity and must not pay for a probe to get
    it. None therefore means "nobody asked", never "unreachable".
    """
    from ..job_queue import (COMFYUI_RECOVERY_REQUIRED_MESSAGE, queue_manager)
    owner = queue_manager.get_comfyui_stalled_barrier()
    if owner is None:
        if not queue_manager.has_comfyui_stalled_barrier():
            return None
        # Present but unreadable. It still blocks every generation, and no code
        # may guess its way out of it — but hiding it is how a user ends up
        # thinking the app is simply broken.
        return {'kind': 'unreadable', 'job_id': None, 'can_confirm_restart': False,
                'message': COMFYUI_RECOVERY_REQUIRED_MESSAGE,
                'connection': None,
                'detail': ('LDS found an invalid ComfyUI recovery record. Restart LDS '
                           'and check the server log before starting new generations.')}
    stalled_since, metadata = _stalled_job_facts(owner.get('job_id'))
    dataset_id = owner.get('dataset_id') or metadata.get('dataset_id')
    try:
        dataset_id = int(dataset_id) if dataset_id is not None else None
    except (TypeError, ValueError):
        dataset_id = None
    return {
        'kind': owner.get('kind'),
        'job_id': owner.get('job_id'),
        'dataset_id': dataset_id,
        'dataset_name': _dataset_name(dataset_id),
        'variation_label': metadata.get('variation_label'),
        'run_id': owner.get('run_id'),
        'cell_id': owner.get('cell_id'),
        'stalled_since': stalled_since,
        'message': COMFYUI_RECOVERY_REQUIRED_MESSAGE,
        'detail': owner.get('reason'),
        'connection': None,
        # Every readable barrier can be resolved from here once the user
        # confirms the restart; the backend still refuses anything it cannot
        # prove or identify, with its own message.
        'can_confirm_restart': True,
    }


@bp.get('/comfyui-recovery')
def comfyui_recovery_state():
    """Poll target for the app-wide banner.

    Reading this also attempts the provable automatic clear, so an install left
    blocked overnight heals as soon as any page is open and ComfyUI is back —
    without the user having to click the thing that was refused.

    That same attempt is where the reachability verdict comes from: the banner
    must be able to say "LDS cannot reach ComfyUI at <url>" instead of blaming a
    paused job for a connection that never existed.
    """
    from ..job_queue import peek_auto_recovery_notice, probe_comfyui_barrier
    probe = probe_comfyui_barrier()
    if probe.resolved is not None:
        logger.info('system: ComfyUI recovery barrier cleared automatically on poll')
    recovery = _recovery_snapshot()
    if recovery is not None:
        # Only ever paid for while something is actually blocking: a healthy app
        # polls this route forever and must not probe ComfyUI for nothing.
        recovery['connection'] = _comfyui_connection(probe.link)
        # Whether the banner may offer to START ComfyUI itself, rather than only
        # ask the user to confirm they restarted it. Same check the start route
        # runs before spawning, so the button can never appear over an endpoint
        # that would refuse. False on every install whose ComfyUI is not ours to
        # run — Desktop, a hand-written .bat, another machine — where the honest
        # offer is the confirmation, not a launch that would fail. Computed only
        # inside this branch: an unblocked app must not pay for it.
        from ..services import comfyui_control
        recovery['can_start_comfyui'] = comfyui_control.can_start()
    return jsonify({'ok': True,
                    'recovery': recovery,
                    'auto_cleared': peek_auto_recovery_notice()})


@bp.post('/comfyui-recovery/resolve')
def comfyui_recovery_resolve():
    """"I restarted ComfyUI — clear it", from anywhere in the app."""
    data = request.get_json(silent=True) or {}
    if data.get('confirmed_comfyui_restart') is not True:
        return jsonify({'error': 'Confirm that you restarted ComfyUI before '
                                 'clearing this paused job.'}), 400
    state = _recovery_snapshot()
    if state is None:
        return jsonify({'ok': True, 'cleared': 0, 'already_clear': True})
    # A confirmation is only meaningful when the replacement process answers
    # NOW; a cached green probe is not a restart gate.
    gate = _require_comfyui(force=True)
    if gate:
        return gate
    if state['kind'] == 'unreadable':
        return jsonify({'ok': False, 'error': state['detail']}), 409

    from ..job_queue import auto_resolve_comfyui_barrier, queue_manager
    if state['kind'] == 'prompt':
        # A known prompt id is checkable, so the user's word is not the
        # authority here — ComfyUI's answer is. Still queued/running means the
        # job is alive and must not be cancelled behind the user's back.
        if auto_resolve_comfyui_barrier() is not None:
            return jsonify({'ok': True, 'cleared': 1})
        return jsonify({'ok': False, 'error': (
            'ComfyUI still reports this generation, or it did not answer. If you '
            'just restarted it, wait a few seconds and try again; if the job is '
            'still running there, let it finish.')}), 409

    cleared = 0
    try:
        if state.get('run_id'):
            from ..services import lora_test_studio as lts
            cleared = lts.confirm_unknown_comfyui_restart(
                LOCAL_USER, run_id=state['run_id'], restart_confirmed=True)
        elif state.get('cell_id') and state.get('dataset_id') is not None:
            from ..services import lora_test_studio as lts
            cleared = lts.confirm_unknown_comfyui_restart(
                LOCAL_USER, dataset_id=state['dataset_id'], restart_confirmed=True)
        elif state.get('dataset_id') is not None:
            from ..services import face_dataset_service as fds
            cleared = fds.confirm_unknown_generation_restart(
                LOCAL_USER, state['dataset_id'], restart_confirmed=True)
    except Exception as e:
        return _map_error(e)
    if cleared:
        return jsonify({'ok': True, 'cleared': cleared})
    # The services match on a live queue row and its card, and a barrier with no
    # identity at all reaches here having called none of them. Either way, if the
    # row is already gone the barrier guards nothing and NO resolver can ever
    # lift it — with a confirmed restart and a ComfyUI that answers, dropping it
    # is the difference between a working button and a hand-written SQL DELETE.
    if queue_manager.discard_orphan_comfyui_barrier():
        return jsonify({'ok': True, 'cleared': 1, 'discarded_orphan': True})
    return jsonify({'ok': False, 'error': (
        'The paused job could not be cleared. Refresh the page and try again; '
        'if it persists, check the server log.')}), 409


@bp.get('/stats')
def machine_stats():
    """📊 CPU / RAM / GPU / VRAM of the machine RUNNING the server.

    Feeds the small load readout on the Canvas, polled every few seconds while
    the tab is visible. Two properties make that polling harmless: the reading
    is cached ~3 s and SHARED (N tabs cost one `nvidia-smi`, not N), and every
    field is optional — a machine with no NVIDIA card simply answers without
    the GPU keys instead of answering zeros the widget would draw as "idle".

    Always 200: this is a glance, never a gate. A machine that cannot answer
    anything answers `{}`, and the widget draws nothing.
    """
    from ..services import system_stats
    return jsonify(system_stats.machine_stats())


@bp.get('/ollama-fence')
def ollama_fence_state():
    """Is the local Ollama fence standing in the way, and because of what?

    Polled by the surfaces the fence refused. It is a read-only /api/ps probe
    on the configured local endpoint, which is what makes waiting worth doing:
    Ollama unloads an idle model by itself after a few minutes, so most of
    these blocks end without anyone touching anything.
    """
    from ..services import ollama_gpu_fence
    return jsonify({'ok': True, **ollama_gpu_fence.fence_status()})


@bp.post('/ollama-fence/unload')
def ollama_fence_unload():
    """"Unload it and continue" — the ONE place LDS evicts a model it does not own.

    The default everywhere else is to refuse, because on a machine running
    another AI tool the resident model is usually that tool's legitimate work.
    The consent flag is what lifts that refusal, and it is required: this route
    is never reached by a retry, a poll or a fallback.
    """
    data = request.get_json(silent=True) or {}
    if data.get('confirmed_unload_external') is not True:
        return jsonify({'error': 'Confirm the unload of the external Ollama model '
                                 'before LDS touches it.'}), 400
    from ..services import ollama_gpu_fence
    result = ollama_gpu_fence.unload_foreign_models()
    if result['ok']:
        return jsonify({'ok': True, **result})
    reasons = {
        'not-local': 'LDS is configured with a remote Ollama endpoint, so there is '
                     'no local model for it to unload.',
        'unreachable': 'Ollama did not answer. Check that it is running, then try again.',
        'still-loaded': 'Ollama still reports a model in memory — a request may still '
                        'be running there. Wait a moment and try again.',
    }
    return jsonify({'ok': False, 'error': reasons.get(result['reason'],
                                                      'The model could not be unloaded.'),
                    **result}), 409


# --- The generation queue, app-wide -----------------------------------------
# One ComfyUI, one serialized queue, fed by the dataset workspace, the Test
# Studio, the ◉ Canvas and the Bank. It lives here for the same reason the
# recovery surface above does: it is about the machine running the server, not
# about any one dataset. See services/queue_view.py for what a job is allowed to
# claim about itself, and why two families are listed but not cancellable.


def _queue_dataset_names(jobs):
    """{dataset_id: name} for the datasets the queue mentions, in ONE pass.

    Resolved here rather than in `queue_view` so that module keeps out of the
    services that feed the queue.
    """
    from ..services import face_dataset_service as fds
    from ..services import queue_view
    names = {}
    for dataset_id in queue_view.dataset_ids(jobs):
        ds = fds.get_dataset(LOCAL_USER, dataset_id)
        if ds is not None:
            names[dataset_id] = ds.name
    return names


@bp.get('/queue')
def generation_queue():
    """Everything still owing GPU time, in the order the worker will take it."""
    from ..services import queue_view
    listing = queue_view.list_queue()
    names = _queue_dataset_names(listing['jobs'])
    for job in listing['jobs']:
        job['dataset_name'] = names.get(job['dataset_id'])
    return jsonify({'ok': True, **listing})


@bp.post('/queue/<job_id>/next')
def generation_queue_promote(job_id):
    """Send one WAITING job to the front of the queue.

    Writes only `priority`, which is what the worker has always ordered by
    (`priority DESC, created_at ASC`) — nothing about the job itself changes.
    409 when it already started: at that point it can be cancelled, not
    re-ordered, and pretending otherwise would be a button that does nothing.
    """
    from ..services import queue_view
    result = queue_view.promote(job_id)
    if result.get('ok'):
        return jsonify({'ok': True})
    return jsonify({'error': result['error']}), result.get('status', 409)


@bp.post('/queue/<job_id>/cancel')
def generation_queue_cancel(job_id):
    """Cancel ONE queued or running job, and settle whatever row owns it.

    Deliberately NOT the same thing as ⏹ Stop generation, and it does not wear
    that label: Stop ends a whole batch and removes the tiles it had not
    produced yet, while this drops one job and leaves its tile marked failed —
    which is recoverable (Retry re-queues it) and is exactly what already
    happens when a stalled job is auto-resolved.

    Refused for the two families a pass is BLOCKED on (the watermark inpaint,
    the reference edit): cancelling those from here would leave that pass
    waiting on a result that will never come, and each has its own Stop.
    """
    from ..job_queue import _dispatch_auto_resolved_cancellation, queue_manager
    from ..models import ImageGenerationQueue
    from ..services import queue_view
    row = ImageGenerationQueue.query.filter_by(job_id=str(job_id)).first()
    if row is None or row.status not in queue_view.LIVE_STATUSES:
        return jsonify({'error': 'this job is no longer in the queue'}), 404
    job = queue_view.describe(row)
    if not job['cancellable']:
        return jsonify({'error': f"This job belongs to {job['blocked_by']} — "
                                 'stop it from there so the pass waiting on it '
                                 'is not left hanging.'}), 409
    try:
        outcome = queue_manager.cancel_job_outcome(job_id, LOCAL_USER, 'image')
    except Exception:
        logger.exception('could not safely cancel queued job %s', job_id)
        outcome = 'retry'
    if outcome == 'cancelled':
        _dispatch_auto_resolved_cancellation(job_id)
        return jsonify({'ok': True, 'outcome': outcome})
    # Everything else is a recovery state the app already has words for; say
    # which one rather than reporting a cancellation that did not happen.
    messages = {
        'restart_required': 'ComfyUI must be restarted before this job can be '
                            'resolved — LDS cannot safely identify its remote prompt.',
        'barrier_corrupt': 'A ComfyUI recovery is pending. Resolve it from the banner '
                           'first, then try again.',
        'retry': 'The job could not be cancelled cleanly. Try again in a moment.',
        'terminal': 'This job already finished.',
        'missing': 'This job is no longer in the queue.',
    }
    status = 409 if outcome not in ('terminal', 'missing') else 404
    return jsonify({'error': messages.get(outcome, 'The job could not be cancelled.'),
                    'outcome': outcome}), status
