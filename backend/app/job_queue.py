"""Slim image job queue: a FIFO worker over `ImageGenerationQueue` plus a tiny
JSON key/value store (`SystemState`) used by lifted services for cross-request
flags (e.g. training locks, test-studio run state).

Replaces the source app's ~181 KB queue_manager. `_submit`/`_poll_outputs` are
the only two functions that talk to ComfyUI; both lazy-import
`app.utils.comfyui` (not lifted until Task 13) so a missing module fails the
job cleanly instead of crashing the worker thread. `_dispatch_completion`
lazy-imports the owning service (routing on job metadata) so this module never
imports the services that create jobs (avoids import cycles).
"""
from __future__ import annotations
import json
import logging
import threading
import time
import uuid
from datetime import datetime

from .extensions import db
from .models import ImageGenerationQueue, SystemState

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 2
POLL_TIMEOUT_SECONDS = 15 * 60
STUCK_TIMEOUT_MINUTES = 10
IDLE_SLEEP_SECONDS = 1


def _submit(workflow, client_id):
    """Queue a workflow on ComfyUI. Raises on failure; the caller fails the job."""
    from .utils.comfyui import queue_prompt_to_comfyui
    return queue_prompt_to_comfyui(workflow, client_id)


def _poll_outputs(prompt_id, timeout=POLL_TIMEOUT_SECONDS):
    """Poll ComfyUI history for `prompt_id` until it has an output image, an
    error, or `timeout` elapses. Returns (filename, failed). Heartbeats the
    owning job row on every poll so `is_stuck()` sees this job as alive.

    NOTE: the exact history JSON shape is ComfyUI's own (an `outputs` dict of
    node_id -> {images: [{filename, subfolder, ...}]}, plus a `status` dict).
    `app.utils.comfyui` isn't lifted yet (Task 13) so this is a best-effort
    reading of that shape; any exception here degrades to a failed job rather
    than raising.
    """
    from .utils.comfyui import get_comfyui_history
    deadline = time.monotonic() + timeout
    while True:
        try:
            history = get_comfyui_history(prompt_id) or {}
            entry = history.get(prompt_id, history) if isinstance(history, dict) else {}
        except Exception:
            entry = {}
        outputs = (entry or {}).get('outputs') or {}
        for node_output in outputs.values():
            images = (node_output or {}).get('images') or []
            if images:
                return images[0].get('filename'), False
        status = (entry or {}).get('status') or {}
        if status.get('completed') and not outputs:
            return None, True  # ComfyUI finished the prompt with no image -> failure

        job = ImageGenerationQueue.query.filter_by(comfyui_prompt_id=prompt_id).first()
        if job:
            job.last_heartbeat = datetime.utcnow()
            db.session.commit()

        if time.monotonic() >= deadline:
            return None, True
        time.sleep(POLL_INTERVAL_SECONDS)


def _dispatch_completion(job, filename, failed):
    """Route a finished job to whichever service created it, per its metadata.
    A callback crash must never take down the worker thread."""
    try:
        md = json.loads(job.job_metadata or '{}')
    except (TypeError, ValueError):
        md = {}
    try:
        if md.get('is_lora_test'):
            from .services import lora_test_studio
            lora_test_studio.link_completed_test_image(job.job_id, filename, failed=failed)
        elif md.get('model_name') == 'klein_edit_dataset':
            from .services import face_dataset_service
            face_dataset_service.link_completed_dataset_image(job.job_id, filename, failed=failed)
    except Exception:
        logger.exception('job_queue: completion dispatch failed for job %s', job.job_id)


class JobQueueManager:
    """Singleton worker: one background thread picks the oldest pending image
    job, submits it to ComfyUI, polls for its output, and dispatches
    completion — all synchronously per job via `process_one()`."""

    def __init__(self):
        self._app = None
        self._thread = None
        self._running = False

    def init_app(self, app):
        self._app = app

    # -- lifecycle ------------------------------------------------------
    def start(self):
        """Idempotent: a no-op if the worker thread is already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        with self._app.app_context():
            self._recover_stuck_jobs()
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name='job-queue-worker', daemon=True)
        self._thread.start()

    def stop(self, timeout=5):
        """For tests: stop the loop and wait for the thread to exit."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run_loop(self):
        while self._running:
            try:
                with self._app.app_context():
                    worked = self.process_one()
            except Exception:
                logger.exception('job_queue: worker loop error')
                worked = False
            time.sleep(0 if worked else IDLE_SLEEP_SECONDS)

    def _recover_stuck_jobs(self):
        """Boot recovery: rows left in processing/sent_to_comfy past the
        timeout (a prior crash) are failed and their callback dispatched."""
        stuck = [j for j in ImageGenerationQueue.query
                 .filter(ImageGenerationQueue.status.in_(('processing', 'sent_to_comfy'))).all()
                 if j.is_stuck(STUCK_TIMEOUT_MINUTES)]
        for job in stuck:
            job.update_status('failed', error_message='stale job recovered at boot')
            db.session.commit()
            _dispatch_completion(job, None, True)

    # -- worker -----------------------------------------------------------
    def process_one(self) -> bool:
        """Run one pending job end-to-end, synchronously. Returns True if a
        job was processed, False if the queue was empty (caller should back
        off). Assumes an active app context (pushed by the caller)."""
        job = (ImageGenerationQueue.query
               .filter_by(status='pending')
               .order_by(ImageGenerationQueue.priority.desc(), ImageGenerationQueue.created_at.asc())
               .first())
        if job is None:
            return False

        job.update_status('processing')
        db.session.commit()

        try:
            workflow = json.loads(job.workflow_data or '{}')
            prompt_id = _submit(workflow, job.job_id)
            job.update_status('sent_to_comfy', comfyui_prompt_id=prompt_id)
            db.session.commit()
            filename, failed = _poll_outputs(prompt_id, POLL_TIMEOUT_SECONDS)
        except Exception as exc:
            logger.warning('job_queue: job %s failed: %s', job.job_id, exc)
            filename, failed = None, True

        db.session.refresh(job)
        if job.status == 'cancelled':  # cancelled by another request while in flight
            _dispatch_completion(job, filename, True)
            return True

        job.update_status('failed' if failed else 'completed',
                          result_filename=filename,
                          error_message=None if not failed else 'generation failed')
        db.session.commit()
        _dispatch_completion(job, filename, failed)
        return True

    # -- public API (verbatim surface; lifted services call these) --------
    def add_job(self, job_type='image', user_id='local', workflow_data=None, prompt='',
               job_id=None, metadata=None, priority=10) -> str:
        if job_type != 'image':
            raise ValueError(f'unsupported job_type: {job_type!r}')
        if not workflow_data:
            raise ValueError('workflow_data is required')
        job_id = job_id or str(uuid.uuid4())
        job = ImageGenerationQueue(
            job_id=job_id,
            user_id=str(user_id),
            status='pending',
            workflow_data=json.dumps(workflow_data),
            prompt=prompt,
            priority=priority,
            job_metadata=json.dumps(metadata) if metadata else None,
        )
        db.session.add(job)
        db.session.commit()
        return job_id

    def cancel_job(self, job_id, user_id=None, job_type='image') -> bool:
        """pending -> cancelled directly; processing/sent_to_comfy -> best-effort
        (marks the row; `process_one` checks status before finalizing)."""
        if job_type != 'image':
            return False
        query = ImageGenerationQueue.query.filter_by(job_id=job_id)
        if user_id is not None:
            query = query.filter_by(user_id=str(user_id))
        job = query.first()
        if job is None or job.status in ('completed', 'failed', 'cancelled'):
            return False
        job.update_status('cancelled')
        db.session.commit()
        return True

    # -- system-state KV (underscore names required verbatim) -------------
    def _set_system_state(self, key, value, ttl_seconds=None):
        if value is None:
            SystemState.query.filter_by(key=key).delete()
            db.session.commit()
            return
        exp = time.time() + ttl_seconds if ttl_seconds is not None else None
        encoded = json.dumps({'v': value, 'exp': exp})
        row = SystemState.query.get(key)
        if row is None:
            db.session.add(SystemState(key=key, value=encoded))
        else:
            row.value = encoded
        db.session.commit()

    def _get_system_state(self, key, default=None):
        row = SystemState.query.get(key)
        if row is None or row.value is None:
            return default
        try:
            payload = json.loads(row.value)
        except (TypeError, ValueError):
            return default
        exp = payload.get('exp')
        if exp is not None and time.time() >= exp:
            db.session.delete(row)
            db.session.commit()
            return default
        return payload.get('v', default)


queue_manager = JobQueueManager()
