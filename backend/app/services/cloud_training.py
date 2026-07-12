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
import os
import secrets as pysecrets
import threading
import time
from datetime import datetime
from pathlib import Path

from .. import config as cfg
from ..extensions import db
from ..models import CloudTrainingRun
from . import face_dataset_service as fds
from . import lora_training as lt
from . import vast_client
from .aitoolkit_remote import RemoteAiToolkit

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
    global _monitor_thread
    from flask import current_app
    app = current_app._get_current_object()
    _monitor_thread = threading.Thread(
        target=_monitor, args=(app, run_id), daemon=True, name=f'cloud-train-{run_id}')
    _monitor_thread.start()


POLL_SECONDS = 10
READY_TIMEOUT_SECONDS = 900          # 15 min: boot + image pull
UNREACHABLE_GRACE_SECONDS = 180      # tolerated mid-run network blackout
_sleep = time.sleep


def _now():
    return time.time()


def _make_remote(run) -> RemoteAiToolkit:
    return RemoteAiToolkit(run.base_url, run.auth_token)


def _cloudify_job_config(job_config: dict, job_name: str,
                         staging_dataset: str, pod_settings: dict) -> dict:
    """Rewrite the locally-built config for the pod: remote paths, remote
    trainer type (DB status updates), and the job name the pod's routes key
    on. The staging->pod path swap is done on the JSON text so every field
    referencing the staging dir (folder_path, mask_path) is rewritten at
    once, backslash-escaping included."""
    pod_ds = pod_settings['DATASETS_FOLDER'].rstrip('/') + '/' + job_name
    text = json.dumps(job_config)
    needle = json.dumps(str(staging_dataset))[1:-1]     # JSON-escaped form
    text = text.replace(needle, pod_ds)
    out = json.loads(text)
    conf = out['config']
    conf['name'] = job_name
    proc = conf['process'][0]
    proc['type'] = 'diffusion_trainer'
    proc['training_folder'] = pod_settings['TRAINING_FOLDER']
    proc['device'] = 'cuda:0'
    return out


def _finish(run, status, detail='', error=None, destroy=True):
    if destroy and run.vast_instance_id:
        try:
            vast_client.destroy_instance(run.vast_instance_id)
        except Exception as e:
            logger.warning('terminate %s failed: %s', run.vast_instance_id, e)
    _set(run, status=status, phase_detail=detail, error=error,
         finished_at=datetime.utcnow())


def _monitor(app, run_id):
    """Full run lifecycle. Runs in a daemon thread; every exit path goes
    through _finish() so the pod cannot be leaked by this thread."""
    with app.app_context():
        run = CloudTrainingRun.query.get(run_id)
        if not run:
            return
        c = cfg.get('cloud') or {}
        max_seconds = int(c.get('max_runtime_minutes') or 240) * 60
        started = _now()
        try:
            # -- provision (if resuming, the instance may already exist) ----
            if not run.vast_instance_id:
                _provision(run)

            # -- wait until the pod's UI answers ----------------------------
            # Readiness is checked BEFORE the elapsed-time read: an
            # already-booted pod (the common case, and every resumed run)
            # must be able to break out on the very first iteration without
            # ever touching _now() -- a test clock that jumps in large
            # strides per call must not misfire this boot-timeout on a pod
            # that was, in fact, instantly ready.
            _set(run, phase_detail='Waiting for the pod to boot')
            port = int(c.get('ui_port') or 8675)
            while True:
                inst = vast_client.get_instance(run.vast_instance_id)
                base = vast_client.derive_base_url(inst, port) if inst else None
                if base:
                    if run.base_url != base:
                        _set(run, base_url=base)
                    if _make_remote(run).is_ready():
                        break
                if _now() - started > READY_TIMEOUT_SECONDS:
                    raise RuntimeError('pod did not become ready in time')
                _sleep(POLL_SECONDS)

            remote = _make_remote(run)

            # -- resume contract: an already-submitted job (app restarted
            # mid-run) skips settings/upload/create/start entirely and goes
            # straight to polling the existing remote job. ------------------
            if not run.remote_job_id:
                pod_settings = remote.ensure_settings(hf_token=cfg.secret('HF_TOKEN'))

                # -- upload dataset (+ masks folder if present) --------------
                _set(run, status='uploading', phase_detail='Uploading dataset')
                staging_dataset = os.path.join(run.staging_dir, 'dataset')
                remote.upload_dataset(run.job_name, staging_dataset)
                masks_dir = staging_dataset + '_masks'
                if os.path.isdir(masks_dir) and os.listdir(masks_dir):
                    remote.upload_dataset(run.job_name + '_masks', masks_dir)

                # -- build + submit the job -----------------------------------
                params = json.loads(run.train_params or '{}')
                ds = fds.get_dataset('local', run.dataset_id)
                job_config = lt.build_job_config(
                    ds, staging_dataset, steps=params.get('steps') or 3000,
                    training_folder='__POD__')
                job_config = _cloudify_job_config(job_config, run.job_name,
                                                  staging_dataset, pod_settings)
                job_id = remote.create_job(run.job_name, job_config)
                remote.start_job(job_id)
                _set(run, remote_job_id=job_id, status='training',
                     phase_detail='Job queued on the pod')
            else:
                job_id = run.remote_job_id
                _set(run, phase_detail='Resuming — reattaching to running job')

            # -- poll until terminal ------------------------------------------
            last_ok = _now()
            while True:
                if _now() - started > max_seconds:
                    try:
                        remote.stop_job(job_id)
                    except Exception:
                        pass
                    _try_download_checkpoint(run, remote)
                    _finish(run, 'stopped',
                            detail='Max runtime reached — pod terminated',
                            error='max runtime cap hit')
                    return
                if _stop_event.is_set():
                    _stop_event.clear()
                    _set(run, phase_detail='Stopping on user request')
                    try:
                        remote.stop_job(job_id)
                    except Exception:
                        pass
                    _try_download_checkpoint(run, remote)
                    _finish(run, 'stopped', detail='Stopped by user')
                    return
                try:
                    job = remote.get_job(job_id)
                    last_ok = _now()
                except Exception as e:
                    if _now() - last_ok > UNREACHABLE_GRACE_SECONDS:
                        raise RuntimeError(f'pod unreachable: {e}')
                    _sleep(POLL_SECONDS)
                    continue

                _pull_log_and_samples(run, remote, job_id)
                status = job.get('status')
                info = job.get('info') or ''
                _set(run, phase_detail=f"{status}: {info}"[:500])

                if status == 'completed':
                    ok = _try_download_checkpoint(run, remote)
                    if not ok:
                        # LoRA > a few minutes of pod time: keep the pod for
                        # manual recovery; max-runtime/reconcile will reap it.
                        _set(run, status='error_pod_kept',
                             error='checkpoint download failed — pod kept, '
                                   f'recover manually at {run.base_url}',
                             finished_at=datetime.utcnow())
                        return
                    _import_result(run)
                    _finish(run, 'done', detail='Training complete')
                    return
                if status in ('error', 'stopped'):
                    _try_download_checkpoint(run, remote)
                    _finish(run, 'error' if status == 'error' else 'stopped',
                            detail=f'Remote job {status}', error=info or status)
                    return
                _sleep(POLL_SECONDS)
        except Exception as e:
            logger.exception('cloud run %s failed', run_id)
            _finish(run, 'error', detail='Run failed', error=str(e)[:500])


def _pull_log_and_samples(run, remote, job_id):
    """Mirror remote log + new samples into staging so cloud_progress reuses
    the exact local parsing/serving machinery. Never raises."""
    try:
        text = remote.get_log(job_id)
        with open(os.path.join(run.staging_dir, 'training.log'), 'w',
                  encoding='utf-8', errors='replace') as fh:
            fh.write(text)
    except Exception:
        pass
    try:
        samples_dir = os.path.join(run.staging_dir, 'samples')
        have = set(os.listdir(samples_dir))
        for remote_path in remote.get_samples(job_id):
            name = os.path.basename(remote_path.replace('\\', '/'))
            if name and name not in have:
                remote.download_sample(remote_path,
                                       os.path.join(samples_dir, name))
    except Exception:
        pass


def _try_download_checkpoint(run, remote) -> bool:
    """Download the newest .safetensors into staging. False on failure."""
    try:
        files = [f for f in remote.list_files(run.remote_job_id)
                 if f.get('path', '').endswith('.safetensors')]
        if not files:
            return False
        newest = sorted(files, key=lambda f: f['path'])[-1]
        name = os.path.basename(newest['path'].replace('\\', '/'))
        dest = os.path.join(run.staging_dir, name)
        remote.download_public_file(newest['path'], dest)
        _set(run, status='downloading', checkpoint_local_path=dest,
             phase_detail=f'Downloaded {name}')
        return True
    except Exception as e:
        logger.warning('checkpoint download failed: %s', e)
        return False


def _import_result(run):
    """Copy the downloaded checkpoint into the ComfyUI loras folder when one
    is configured; otherwise it stays in staging (served by the download
    route). Import failure must not fail the run."""
    try:
        if not run.checkpoint_local_path:
            return
        if not (cfg.get('comfyui.base_dir') or cfg.get('comfyui.loras_dir')):
            return
        params = json.loads(run.train_params or '{}')
        lt.import_checkpoint('local', run.dataset_id,
                             os.path.basename(run.checkpoint_local_path),
                             family=params.get('train_type'),
                             src_dir=run.staging_dir)
    except Exception as e:
        logger.warning('cloud import into ComfyUI failed: %s', e)


def _cost_estimate(run) -> float:
    if not run.price_per_hour:
        return 0.0
    end = run.finished_at or datetime.utcnow()
    hours = max(0.0, (end - run.created_at).total_seconds() / 3600.0)
    return round(run.price_per_hour * hours, 2)


def _run_payload(run) -> dict:
    return {'run_id': run.id, 'dataset_id': run.dataset_id, 'status': run.status,
            'phase_detail': run.phase_detail, 'gpu': run.gpu_name,
            'price_per_hour': run.price_per_hour,
            'cost_estimate': _cost_estimate(run), 'error': run.error,
            'checkpoint_ready': bool(run.checkpoint_local_path),
            'created_at': run.created_at.isoformat() if run.created_at else None}


def cloud_status() -> dict:
    active = get_active_run()
    last = (CloudTrainingRun.query
            .order_by(CloudTrainingRun.id.desc()).first())
    return {'configured': bool(cfg.secret('VAST_API_KEY')),
            'active': _run_payload(active) if active else None,
            'last': _run_payload(last) if last else None}


def cloud_progress(user_id, dataset_id) -> dict:
    """Same shape as lt.training_progress + cloud phase/cost fields, built
    from the staging mirror (log + samples) written by the monitor."""
    run = (CloudTrainingRun.query.filter_by(dataset_id=dataset_id)
           .order_by(CloudTrainingRun.id.desc()).first())
    empty = {'step': None, 'total': None, 'loss': None, 'speed': None,
             'eta': None, 'loss_curve': []}
    if not run:
        return {'active': False, 'log_exists': False, **empty, 'samples': [],
                'phase': None, 'phase_detail': None, 'cost_estimate': 0.0,
                'gpu': None, 'price_per_hour': None, 'checkpoint_ready': False}
    log_path = os.path.join(run.staging_dir or '', 'training.log')
    parsed = dict(empty)
    log_exists = bool(run.staging_dir) and os.path.isfile(log_path)
    if log_exists:
        try:
            with open(log_path, encoding='utf-8', errors='replace') as fh:
                parsed.update(lt._parse_training_log(fh.read()))
        except OSError:
            pass
    samples = []
    samples_dir = os.path.join(run.staging_dir or '', 'samples')
    if os.path.isdir(samples_dir):
        for f in os.listdir(samples_dir):
            m = lt._SAMPLE_RE.search(f)
            if m:
                samples.append({'filename': f, 'step': int(m.group(1)),
                                'prompt_idx': int(m.group(2))})
        samples.sort(key=lambda s: s['step'], reverse=True)
    return {'active': run.status in ACTIVE_STATES, 'log_exists': log_exists,
            **parsed, 'samples': samples, **_run_payload(run),
            'phase': run.status}
