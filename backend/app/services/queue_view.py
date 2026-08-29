"""👁️ The generation queue, made visible.

`job_queue` has always been a queue: a FIFO worker over `image_generation_queue`
that takes one job at a time and orders what waits by `priority DESC,
created_at ASC`. Nothing ever showed it. Every surface displayed only its OWN
slice — the tiles a dataset is waiting on, the cells of a Studio run — so work
queued from one screen was invisible from every other, and "is it stuck, or is
something else simply in front of it?" had no answer anywhere in the app.

This module is the read model for that queue, plus the two actions that make it
worth looking at (GitHub #44, charlesangus).

WHAT A JOB IS is not stored as such: the queue row carries a workflow and a
metadata blob, and `job_queue._dispatch_completion` decides at completion time
which service owns it by reading that blob. `describe` reads the SAME keys in
the SAME order, so the panel can never name a job something other than what the
completion callback will hand it to. That ordering is the contract, and
`test_queue_view.py` pins it against the dispatcher.

CANCELLING IS NOT ALWAYS THE PANEL'S TO OFFER. Two families are waited on
SYNCHRONOUSLY by the pass that queued them (`watermark_klein._wait_for_job`, and
the reference edit, which produces a Before/After the user still has to resolve).
Cancelling those from here would break a caller that is blocked on the result and
whose own Stop lives two screens away. They are listed — seeing why the GPU is
busy is the whole point — with `cancellable: False` and the sentence saying where
their Stop really is.
"""
from __future__ import annotations

import json
import logging

from ..extensions import db
from ..models import ImageGenerationQueue

logger = logging.getLogger(__name__)

# Rows that are still going to consume the GPU. Everything else is history and
# has no business in a queue view.
LIVE_STATUSES = ('pending', 'processing', 'sent_to_comfy', 'cancel_requested', 'stalled')

# The vocabulary the Test Studio already shows for these very rows
# (`lora_test_studio._queue_activity`). Two surfaces describing one queue must
# not invent two words for one state.
DISPLAY_STATUS = {
    'pending': 'queued',
    'processing': 'generating',
    'sent_to_comfy': 'generating',
    'cancel_requested': 'stalled',
    'stalled': 'stalled',
}

# Priority given to a job the user sends to the front. The worker orders by
# `priority DESC, created_at ASC` and every enqueue helper leaves the default 10,
# so one band above it is enough; promoting again bumps past whatever was
# promoted before, so "run next" keeps meaning next.
PRIORITY_NORMAL = 10
PRIORITY_NEXT = 20

# Passes that BLOCK on their own job. The panel shows them and keeps its hands
# off — each names the control that really ends it.
_SYNCHRONOUS_OWNERS = {
    'watermark_klein': 'the 🧽 Clean watermarks pass',
    'watermark_klein_mask': 'the 🧽 Clean watermarks pass',
}
_REFERENCE_EDIT_OWNER = 'the ✦ Edit reference panel'


def _metadata(row) -> dict:
    try:
        md = json.loads(row.job_metadata or '{}')
    except (TypeError, ValueError):
        return {}
    return md if isinstance(md, dict) else {}


def _engine_label(md):
    engine = md.get('improve_engine')
    if engine == 'seedvr2':
        return 'SeedVR2'
    if engine == 'klein':
        return 'Klein'
    return {'klein_edit_dataset': 'Klein',
            'krea_identity_edit_dataset': 'Krea 2',
            'seedvr2_upscale': 'SeedVR2'}.get(md.get('model_name'))


def describe(row) -> dict:
    """One queue row, as the panel shows it.

    The branch order mirrors `job_queue._dispatch_completion` exactly: the same
    metadata key wins here as there, so what the panel calls a job is what the
    completion callback will treat it as.
    """
    md = _metadata(row)
    dataset_id = md.get('dataset_id')
    try:
        dataset_id = int(dataset_id) if dataset_id is not None else None
    except (TypeError, ValueError):
        dataset_id = None

    cancellable, blocked_by = True, None
    if md.get('is_lora_test'):
        if md.get('derivation_kind') == 'canvas_image_improve':
            title, surface = 'Upscale & improve', '◉ Canvas'
        else:
            title, surface = 'Test Studio image', '🧪 Test Studio'
    elif md.get('is_reference_edit'):
        title, surface = 'Reference edit', '✦ Edit reference'
        cancellable, blocked_by = False, _REFERENCE_EDIT_OWNER
    elif md.get('is_bank_improve'):
        title, surface = 'Upscale & improve', '🗃️ Bank'
    elif md.get('model_name') in _SYNCHRONOUS_OWNERS:
        title, surface = 'Watermark inpaint', '🧽 Clean watermarks'
        cancellable, blocked_by = False, _SYNCHRONOUS_OWNERS[md['model_name']]
    elif md.get('action') == 'upscale_improve':
        title, surface = 'Upscale & improve', '📁 Dataset'
    else:
        title, surface = 'Generation', '📁 Dataset'

    when = row.started_at or row.created_at
    return {
        'job_id': row.job_id,
        'status': DISPLAY_STATUS.get(row.status, row.status),
        # The RAW status too: 'stalled' and 'cancel_requested' both display as
        # stalled, but only one of them is the durable ComfyUI barrier, and the
        # app-wide recovery banner keys off that difference.
        'raw_status': row.status,
        'title': title,
        'surface': surface,
        'engine': _engine_label(md),
        'dataset_id': dataset_id,
        'created_at': (row.created_at.isoformat() + 'Z') if row.created_at else None,
        'since': (when.isoformat() + 'Z') if when else None,
        'promoted': (row.priority or PRIORITY_NORMAL) > PRIORITY_NORMAL,
        # A job already on the GPU cannot be re-ordered — only cancelled.
        'promotable': row.status == 'pending',
        'cancellable': cancellable,
        'blocked_by': blocked_by,
    }


# What the dock says when nothing is starting. One sentence per hold, written for
# a reader who may be on ANY screen — the queue is app-wide and so is its pause.
#
# It used to relay `lora_test_studio.gpu_busy_reason()` verbatim, whose three
# sentences are written for the Test Studio: someone in the dataset workspace
# was told "the studio is unavailable", about a screen they were not on, and
# pointed at a paused test they had never opened. Same state, wrong address.
_HOLD_SENTENCES = {
    'training': 'Nothing is starting — a LoRA training run has the GPU. '
                'The queue resumes on its own when it ends.',
    'vision': 'Nothing is starting — a vision pass has the GPU (captions, framing '
              'or face analysis). The queue resumes on its own when it ends.',
    'comfyui_recovery': 'Nothing is starting — a paused ComfyUI job is blocking the '
                        'queue. Clear it from the banner at the top of the screen.',
}


def paused_reason() -> str | None:
    """Why the whole queue is standing still, or None when it is not.

    Reads the WORKER's own answer (`queue_manager.gpu_hold`) rather than the DB
    flags: training and the vision pass hold the GPU outside this queue, and one
    of the four conditions the worker checks — the in-process vision window —
    does not appear in those flags at all. That was the single pause with no
    explanation anywhere in the app, which is precisely the one a queue view
    owes the user.
    """
    from ..job_queue import queue_manager
    try:
        hold = queue_manager.gpu_hold()
    except Exception:   # noqa: BLE001 — a listing must not die on a state read
        logger.exception('queue_view: could not read why the queue is held')
        return None
    if hold == 'ollama_fence':
        return _ollama_fence_sentence()
    return _HOLD_SENTENCES.get(hold)


def _ollama_fence_sentence() -> str:
    """The fence hold, worded from what the fence actually saw — this is the one
    hold whose remedy depends on WHO is squatting on the endpoint. KoboldCPP
    self-identifies in /api/ps and can never unload (that is its design), so
    telling its user to "unload the model" would be a dead end; the honest
    remedy there is the Ollama URL itself."""
    try:
        from .ollama_gpu_fence import last_block
        blk = last_block() or {}
    except Exception:   # noqa: BLE001 — same contract as the caller: never die
        blk = {}
    if 'koboldcpp' in (blk.get('families') or ()):
        return ('Nothing is starting — the Ollama URL in Settings points at KoboldCPP, '
                'which never unloads its model, so LDS cannot hand the GPU to ComfyUI. '
                'Point it at a real Ollama (or close KoboldCPP) and the queue resumes.')
    models = ', '.join(blk.get('models') or ())
    if models:
        return (f'Nothing is starting — a local model outside LDS ({models}) is holding '
                'the GPU at the configured Ollama endpoint. Unload it there, or close '
                'the app that loaded it, and the queue resumes.')
    return ('Nothing is starting — the configured Ollama endpoint cannot prove the GPU '
            'is free (it does not answer the way Ollama does). Check the Ollama URL in '
            'Settings ▸ Local tools, and the queue resumes.')


def _live_rows():
    return (ImageGenerationQueue.query
            .filter(ImageGenerationQueue.status.in_(LIVE_STATUSES))
            .order_by(ImageGenerationQueue.priority.desc(),
                      ImageGenerationQueue.created_at.asc()).all())


def list_queue(dataset_names=None) -> dict:
    """Everything still owing GPU time, in the order the worker will take it.

    `dataset_names` is an optional {id: name} lookup; the caller resolves it,
    because this module deliberately does not import the dataset service (the
    queue sits upstream of every surface that feeds it).
    """
    names = dataset_names or {}
    described = [describe(row) for row in _live_rows()]
    # What is HAPPENING comes first, then the line waiting behind it. The worker
    # order alone put the running job wherever its created_at fell — in the
    # middle of the jobs it is holding up — which reads as a bug in the queue
    # rather than as the answer to "what is the GPU doing right now?".
    # `sorted` is stable, so the waiting half keeps the worker's exact order.
    jobs = sorted(described, key=lambda job: job['status'] == 'queued')
    position = 0
    for job in jobs:
        if job['status'] == 'queued':
            position += 1
            job['position'] = position
        else:
            job['position'] = 0
        job['dataset_name'] = names.get(job['dataset_id'])
    return {
        'jobs': jobs,
        'queued': sum(1 for j in jobs if j['status'] == 'queued'),
        'generating': sum(1 for j in jobs if j['status'] == 'generating'),
        'stalled': sum(1 for j in jobs if j['status'] == 'stalled'),
    }


def dataset_ids(jobs) -> list:
    return sorted({j['dataset_id'] for j in jobs if j['dataset_id'] is not None})


def promote(job_id) -> dict:
    """Send one waiting job to the front. Returns {'ok'} or {'ok': False, ...}.

    Only touches `priority`, which is what the worker already orders by — the row
    keeps its place in the table and its identity everywhere else. A job that is
    no longer 'pending' is refused rather than silently ignored: it is either on
    the GPU (nothing left to re-order) or finished.
    """
    row = ImageGenerationQueue.query.filter_by(job_id=str(job_id)).first()
    if row is None:
        return {'ok': False, 'status': 404,
                'error': 'This job is no longer in the queue.'}
    if row.status != 'pending':
        return {'ok': False, 'status': 409,
                'error': 'This job already started — it can be cancelled, '
                         'but no longer re-ordered.'}
    top = (db.session.query(db.func.max(ImageGenerationQueue.priority))
           .filter(ImageGenerationQueue.status == 'pending').scalar()) or PRIORITY_NORMAL
    row.priority = max(PRIORITY_NEXT, int(top) + 1)
    db.session.commit()
    return {'ok': True}
