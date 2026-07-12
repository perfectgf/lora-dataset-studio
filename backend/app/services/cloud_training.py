"""Cloud LoRA training orchestrator (vast.ai ephemeral pod).

State machine (CloudTrainingRun.status):
  preparing -> provisioning -> uploading -> training -> downloading
  -> terminating -> done | stopped | error | error_pod_kept

Leak-safety invariant: every path between create_instance() and run
completion must end in destroy_instance() -- enforced here (provision
try/except), by the max-runtime cap (monitor, Task 6) and by boot
reconciliation. The local training path is untouched: a cloud run never sets
'training_in_progress', so local generation/captioning stay available."""
import json
import logging
import secrets as pysecrets
import threading
from datetime import datetime
from pathlib import Path

from .. import config as cfg
from ..extensions import db
from ..models import CloudTrainingRun
from . import face_dataset_service as fds
from . import lora_training as lt
from . import vast_client

logger = logging.getLogger(__name__)

ACTIVE_STATES = ('preparing', 'provisioning', 'uploading', 'training',
                 'downloading', 'terminating')

_stop_event = threading.Event()
_monitor_thread = None


def _staging_root() -> Path:
    # face_dataset_service has no DATA_DIR of its own (its image root is
    # cfg.dataset_images_root() = DATA_DIR/datasets); the actual data root
    # lives in config.py as the private _data_dir() (same convention already
    # used by services.chatgpt_oauth). cloud_runs is a sibling of datasets/.
    root = cfg._data_dir() / 'cloud_runs'
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_active_run():
    return (CloudTrainingRun.query
            .filter(CloudTrainingRun.status.in_(ACTIVE_STATES))
            .order_by(CloudTrainingRun.id.desc()).first())


def _set(run, **fields):
    for k, v in fields.items():
        setattr(run, k, v)
    run.updated_at = datetime.utcnow()
    db.session.commit()


def launch_cloud_training(user_id, dataset_id, steps=None, base_model='',
                          variant='turbo', train_type=None, masked=True,
                          allow_caption_mismatch=False) -> dict:
    if not cfg.secret('VAST_API_KEY'):
        raise RuntimeError('vast.ai API key is not configured — add it in Settings')
    if base_model:
        raise ValueError('Custom base models are local-only — cloud training '
                         'uses the official Hugging Face bases')
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    fam = fds.normalize_train_type(train_type or getattr(ds, 'train_type', None))
    if fam == 'sdxl':
        raise ValueError('SDXL training needs a local base checkpoint — '
                         'cloud training supports Z-Image and Krea for now')
    if get_active_run():
        raise RuntimeError('a cloud training run is already active')

    # Same caption-mismatch preflight as launch_training (MISMATCH_CAPTION
    # contract): assert_trainable is ALREADY a standalone helper in
    # lora_training.py (called from launch_training, not inlined there), so
    # no extraction was needed -- just match its real signature:
    # assert_trainable(dataset_id, train_type=None, allow_caption_mismatch=False).
    lt.assert_trainable(dataset_id, train_type=fam,
                        allow_caption_mismatch=allow_caption_mismatch)

    run = CloudTrainingRun(dataset_id=dataset_id, status='preparing',
                           run_name=lt._run_name(ds, family=fam))
    db.session.add(run)
    db.session.commit()
    try:
        # Anything failing past this point (export, staging, thread start) must
        # not strand the 'preparing' row forever — that would deadlock the
        # single-active-run guard above. Flip it to 'error' and re-raise.
        _set(run, vast_label=f'lds-{run.id}',
             job_name=f'lds{run.id}_{lt._run_name(ds, family=fam)}')

        staging = _staging_root() / f'run_{run.id}'
        dataset_dir = staging / 'dataset'
        (staging / 'samples').mkdir(parents=True, exist_ok=True)
        lt.export_dataset_to_aitoolkit(user_id, dataset_id, masked=masked,
                                       dest_dir=str(dataset_dir))
        n_steps = int(steps) if steps else lt.default_steps(ds)
        _set(run, staging_dir=str(staging),
             train_params=json.dumps({'steps': n_steps, 'variant': variant,
                                      'train_type': fam, 'masked': bool(masked)}))
        _stop_event.clear()
        _start_monitor(run.id)
    except Exception as e:
        _set(run, status='error', error=f'launch failed: {e}',
             finished_at=datetime.utcnow())
        raise
    return {'run_id': run.id, 'status': run.status,
            'job_name': run.job_name, 'steps': n_steps}


def _register_instance(run, instance_id, offer, token):
    """Isolated so provisioning tests can inject a post-create failure."""
    _set(run, vast_instance_id=str(instance_id), auth_token=token,
         gpu_name=offer.get('gpu_name'), price_per_hour=offer.get('dph_total'),
         status='provisioning', phase_detail='Instance created — booting')


def _provision(run):
    """Search the cheapest suitable offer and create the instance.
    LEAK-SAFE: any failure after create_instance destroys the instance."""
    c = cfg.get('cloud') or {}
    params = json.loads(run.train_params or '{}')
    fam = params.get('train_type') or 'zimage'
    min_vram = (c.get('min_vram_gb') or {}).get(fam, 24)
    offers = vast_client.search_offers(min_vram_gb=min_vram,
                                       max_dph=c.get('max_price_per_hour', 0.80))
    if not offers:
        raise RuntimeError(
            f'no vast.ai offer matches (>= {min_vram} GB VRAM, '
            f'<= ${c.get("max_price_per_hour", 0.80)}/h) — raise the price cap in Settings')
    offer = offers[0]
    token = pysecrets.token_urlsafe(24)
    port = int(c.get('ui_port') or 8675)
    env = {'AI_TOOLKIT_AUTH': token, f'-p {port}:{port}': '1'}
    hf = cfg.secret('HF_TOKEN')
    if hf:
        env['HF_TOKEN'] = hf
    instance_id = vast_client.create_instance(
        offer['offer_id'], image=c.get('image'), env=env,
        disk_gb=int(c.get('disk_gb') or 60), label=run.vast_label,
        onstart=(c.get('onstart') or None))
    try:
        _register_instance(run, instance_id, offer, token)
    except Exception:
        # the pod exists but we failed to remember it -> kill it NOW, and make
        # the outcome observable (destroy_instance returns False on failure)
        try:
            if not vast_client.destroy_instance(instance_id):
                logger.warning('leak-safe destroy of %s FAILED — instance may still '
                               'be running; boot reconciliation will retry', instance_id)
        except Exception:
            logger.exception('leak-safe destroy of %s raised', instance_id)
        raise


def request_stop() -> bool:
    run = get_active_run()
    if not run:
        return False
    _stop_event.set()
    return True


def reconcile_orphans(app) -> int:
    """Boot-time safety net: destroy every 'lds-*' vast instance that no
    active run owns. GENUINELY never raises (boot must not be blocked): the
    whole body — app_context included — sits under a blanket except, so an
    unexpected failure outside the vast_client calls (db not ready, config
    error...) is logged and returns the count destroyed so far."""
    destroyed = 0
    try:
        with app.app_context():
            if not cfg.secret('VAST_API_KEY'):
                return 0
            try:
                instances = vast_client.list_instances()
            except Exception as e:
                logger.warning('reconcile: cannot list vast instances: %s', e)
                return 0
            active = get_active_run()
            keep = str(active.vast_instance_id) if active and active.vast_instance_id else None
            for inst in instances:
                label = inst.get('label') or ''
                if not label.startswith('lds-'):
                    continue
                if keep and inst['instance_id'] == keep:
                    continue
                try:
                    if vast_client.destroy_instance(inst['instance_id']):
                        destroyed += 1
                        logger.warning('reconcile: destroyed orphan pod %s (%s)',
                                       inst['instance_id'], label)
                except Exception as e:
                    logger.warning('reconcile: destroy %s failed: %s',
                                   inst['instance_id'], e)
    except Exception:
        logger.exception('reconcile failed')
    return destroyed


def _start_monitor(run_id):
    """Replaced with the full monitor loop in the next task."""
    global _monitor_thread
    from flask import current_app
    app = current_app._get_current_object()
    _monitor_thread = threading.Thread(
        target=_monitor, args=(app, run_id), daemon=True, name=f'cloud-train-{run_id}')
    _monitor_thread.start()


def _monitor(app, run_id):     # pragma: no cover — implemented in Task 6
    pass
