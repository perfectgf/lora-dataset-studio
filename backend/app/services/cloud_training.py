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
import re
import secrets as pysecrets
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

from .. import config as cfg
from ..extensions import db
from ..models import CloudTrainingRun
from . import face_dataset_service as fds
from . import gpu_speed
from . import lora_training as lt
from . import vast_client
from .aitoolkit_remote import RemoteAiToolkit

logger = logging.getLogger(__name__)

ACTIVE_STATES = ('preparing', 'provisioning', 'uploading', 'training',
                 'downloading', 'terminating')

_stop_events = {}        # run_id -> threading.Event
_monitor_threads = {}    # run_id -> threading.Thread


def _stop_event_for(run_id):
    return _stop_events.setdefault(int(run_id), threading.Event())


def _staging_root() -> Path:
    # face_dataset_service has no DATA_DIR of its own (its image root is
    # cfg.dataset_images_root() = DATA_DIR/datasets); the actual data root
    # lives in config.py as the private _data_dir() (same convention already
    # used by services.chatgpt_oauth). cloud_runs is a sibling of datasets/.
    root = cfg._data_dir() / 'cloud_runs'
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_active_runs():
    return (CloudTrainingRun.query
            .filter(CloudTrainingRun.status.in_(ACTIVE_STATES))
            .order_by(CloudTrainingRun.id.asc()).all())


def get_active_run():
    """Compat alias for single-run callers/tests: the first of the active
    runs (or None). Multi-run-aware code uses get_active_runs()."""
    actives = get_active_runs()
    return actives[0] if actives else None


def _run_param(run, key):
    """One key of the run's train_params JSON. None when the params are absent
    or corrupted — a pre-feature row, or the 'preparing' window before launch
    stamps them. Valid-but-non-dict JSON ('"x"', '[1]', '3') must degrade to
    None too, not AttributeError — one corrupt row would 500 cloud_status."""
    try:
        parsed = json.loads(run.train_params or '{}')
        return parsed.get(key) if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        return None


def _run_family(run):
    """Family ('zimage'/'krea'/...) stamped in the run's train_params."""
    return _run_param(run, 'train_type')


def latest_run_for(dataset_id, train_type=None):
    """Newest run of the dataset; with train_type, the newest run OF THAT
    FAMILY. Falls back to the plain newest when none matches (or the filter
    is absent) so rows without a stamped family stay reachable."""
    q = (CloudTrainingRun.query.filter_by(dataset_id=dataset_id)
         .order_by(CloudTrainingRun.id.desc()))
    newest = q.first()
    if not train_type:
        return newest
    fam = fds.normalize_train_type(train_type)
    for r in q.all():
        if _run_family(r) == fam:
            return r
    return newest


def _set(run, **fields):
    for k, v in fields.items():
        setattr(run, k, v)
    run.updated_at = datetime.utcnow()
    db.session.commit()


def _reconcile_before_launch(app):
    """Seam around the launch-time reconcile_orphans() call (defined below).
    A thin indirection rather than calling reconcile_orphans directly so
    tests can no-op launch's reconcile call without also neutering tests
    that exercise reconcile_orphans() itself -- both are the same module-level
    name, so patching that name would silence both call sites at once."""
    reconcile_orphans(app)


def launch_cloud_training(user_id, dataset_id, steps=None, base_model='',
                          variant='turbo', train_type=None, masked=True,
                          allow_caption_mismatch=False, gpu_name=None) -> dict:
    if not cfg.secret('VAST_API_KEY'):
        raise RuntimeError('vast.ai API key is not configured — add it in Settings')
    # A user launching after days away is exactly when an expired
    # error_pod_kept pod (past its recovery window) should be reaped, not
    # just at boot. reconcile_orphans() never raises, so this is safe; routed
    # through the _reconcile_before_launch seam (rather than calling
    # reconcile_orphans directly) so tests can no-op *this* call site without
    # also neutering tests that exercise reconcile_orphans() itself.
    from flask import current_app
    # Fire-and-forget: reconcile_orphans never raises and reaping an expired
    # pod does not need to finish before THIS launch — inline it cost the
    # launch click a vast list_instances round-trip.
    threading.Thread(
        target=_reconcile_before_launch,
        args=(current_app._get_current_object(),), daemon=True,
        name='cloud-reconcile-prelaunch').start()
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
    if fam == 'flux':
        raise ValueError('FLUX.1 training is local-only for now — '
                         'cloud training supports Z-Image and Krea')
    actives = get_active_runs()
    limit = max(1, int((cfg.get('cloud.max_concurrent_runs') or 1)))
    # Uniqueness is per (dataset, family): a zimage run and a krea run may
    # train the same dataset in parallel. An active run whose family is
    # unknown (pre-feature row, or the 'preparing' window before the params
    # are stamped) blocks every family of its dataset, out of caution.
    if any(r.dataset_id == dataset_id and (_run_family(r) or fam) == fam
           for r in actives):
        raise RuntimeError(f'this dataset already has an active {fam} cloud run')
    if len(actives) >= limit:
        raise RuntimeError(
            f'cloud run limit reached ({len(actives)}/{limit} active) — '
            'raise cloud.max_concurrent_runs in Settings')
    # Monthly budget: block LAUNCHES only — a running pod is NEVER killed
    # over budget (that would waste the money already spent on its training).
    budget = float(cfg.get('cloud.monthly_budget_usd') or 0)
    if budget > 0:
        spent = month_spend_usd()
        if spent >= budget:
            raise RuntimeError(
                f'monthly cloud budget reached (${spent:.2f} of ${budget:.2f}) — '
                'raise cloud.monthly_budget_usd in Settings')

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
        # Anything failing past this point (params, thread start) must not
        # strand the 'preparing' row forever — that would deadlock the
        # single-active-run guard above. Flip it to 'error' and re-raise.
        # NOTE: the heavy dataset EXPORT (rembg masks: ~1-2 s/image) happens in
        # the MONITOR thread (_prepare_staging), not here — this call must
        # return in well under a second or the launch dialog sits on
        # 'Launching…' for a minute (user-observed).
        _set(run, vast_label=f'lds-{run.id}',
             job_name=f'lds{run.id}_{lt._run_name(ds, family=fam)}')
        n_steps = int(steps) if steps else lt.default_steps(ds)
        # requested_gpu (from the launch-time speed picker) is a PREFERENCE, not
        # a lock: _provision re-searches live offers and rents the cheapest one
        # of this class, falling back to the cheapest overall if the class has
        # since sold out (vast offers are ephemeral).
        params = {'steps': n_steps, 'variant': variant,
                  'train_type': fam, 'masked': bool(masked)}
        if gpu_name:
            params['requested_gpu'] = str(gpu_name)
        # Provenance registry (same as local launches): dataset version at
        # launch time, stamped into the params so payloads can expose it.
        from . import checkpoint_registry
        rec = checkpoint_registry.register_launch(
            user_id, dataset_id, family=fam, source='cloud',
            variant=variant, masked=bool(masked), steps=n_steps,
            cloud_run_id=run.id)
        if rec is not None:
            params['version'] = rec.version
        _set(run, train_params=json.dumps(params))
        _stop_event_for(run.id).clear()
        _start_monitor(run.id)
    except Exception as e:
        _set(run, status='error', error=f'launch failed: {e}',
             finished_at=datetime.utcnow())
        raise
    return {'run_id': run.id, 'status': run.status,
            'job_name': run.job_name, 'steps': n_steps}


def _prepare_staging(run):
    """Heavy part of the launch, run from the MONITOR thread: staging dirs +
    dataset export (rembg masks — ~1-2 s/image). No-op when staging already
    exists (resume). A failure propagates to the monitor's generic error
    handler (run flips to 'error', slot freed)."""
    if run.staging_dir:
        return
    _set(run, phase_detail='Preparing dataset (masks)…')
    params = json.loads(run.train_params or '{}')
    staging = _staging_root() / f'run_{run.id}'
    (staging / 'samples').mkdir(parents=True, exist_ok=True)
    lt.export_dataset_to_aitoolkit('local', run.dataset_id,
                                   masked=bool(params.get('masked', True)),
                                   dest_dir=str(staging / 'dataset'))
    _set(run, staging_dir=str(staging))


def _register_instance(run, instance_id, offer, token):
    """Isolated so provisioning tests can inject a post-create failure."""
    _set(run, vast_instance_id=str(instance_id), auth_token=token,
         gpu_name=offer.get('gpu_name'), price_per_hour=offer.get('dph_total'),
         status='provisioning', phase_detail='Instance created — booting')


# --- Offer quality layer (2026-07-13, after a dead-cheap 5090 host froze in
# --- 'loading'): the absolute cheapest host of a class is adversely selected
# --- more often than not. Bait prices are excluded, recently-failed hosts are
# --- blacklisted, and at similar price the more RELIABLE host wins. ----------

_PRICE_BAIT_RATIO = 0.60      # offers < 60% of their class median are suspect
_SIMILAR_PRICE_WINDOW = 1.10  # within +10% of cheapest -> reliability decides


def _bad_hosts_path() -> Path:
    return _staging_root() / 'bad_hosts.json'


def _run_machine_id(run):
    """machine_id stamped by _provision into train_params. Defensive like
    _run_family: absent/corrupt params -> None, never an exception (this is
    called from stop/timeout paths that must not fail)."""
    try:
        parsed = json.loads(run.train_params or '{}')
        return parsed.get('machine_id') if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        return None


def _load_bad_hosts() -> dict:
    """{machine_id(str): {'ts': epoch, 'reason': str}} — expired entries are
    dropped on read (TTL cloud.host_blacklist_days). Corrupt file -> empty."""
    try:
        raw = json.loads(_bad_hosts_path().read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    ttl = float(cfg.get('cloud.host_blacklist_days') or 3) * 86400
    now = _now()
    live = {k: v for k, v in raw.items()
            if isinstance(v, dict) and now - float(v.get('ts') or 0) <= ttl}
    if len(live) != len(raw):
        try:
            _bad_hosts_path().write_text(json.dumps(live), encoding='utf-8')
        except OSError:
            pass
    return live


def _blacklist_host(machine_id, reason):
    """Remember a host whose pod never became ready so the next launch (and the
    tier list) skips it for a few days. Best-effort: never raises."""
    if not machine_id:
        return
    try:
        hosts = _load_bad_hosts()
        hosts[str(machine_id)] = {'ts': _now(), 'reason': str(reason)[:200]}
        _bad_hosts_path().write_text(json.dumps(hosts), encoding='utf-8')
        logger.warning('blacklisted vast host machine_id=%s for %s day(s): %s',
                       machine_id, cfg.get('cloud.host_blacklist_days') or 3, reason)
    except Exception:
        logger.exception('could not blacklist host %s', machine_id)


def _filter_offers(offers) -> list:
    """Drop blacklisted hosts and bait-priced offers (< 60% of their GPU
    class's median price when the class has >= 3 offers — with fewer there is
    no reliable median). Never returns [] when the input wasn't: if every
    offer got filtered, fall back to the input minus blacklisted hosts only
    (renting a suspect host beats failing the run outright)."""
    bad = _load_bad_hosts()
    not_blacklisted = [o for o in offers
                       if str(o.get('machine_id') or '') not in bad]
    by_class = {}
    for o in not_blacklisted:
        by_class.setdefault(o.get('gpu_name') or '', []).append(o)
    kept = []
    for name, group in by_class.items():
        prices = sorted(o['dph_total'] for o in group
                        if o.get('dph_total') is not None)
        if len(prices) >= 3:
            median = prices[len(prices) // 2]
            floor = median * _PRICE_BAIT_RATIO
            group = [o for o in group
                     if o.get('dph_total') is None or o['dph_total'] >= floor]
        kept.extend(group)
    kept.sort(key=lambda o: o.get('dph_total')
              if o.get('dph_total') is not None else 9e9)
    return kept or not_blacklisted


def _best_of(group):
    """Most reliable offer among those within +10% of the group's cheapest —
    a hair more money for a host that actually boots is the right trade."""
    priced = [o for o in group if o.get('dph_total') is not None]
    if not priced:
        return group[0]
    cheapest = min(o['dph_total'] for o in priced)
    window = [o for o in priced if o['dph_total'] <= cheapest * _SIMILAR_PRICE_WINDOW]
    # reliability first; at equal (or absent) reliability the CHEAPEST wins —
    # offers without the field must not silently cost +10%.
    return max(window, key=lambda o: ((o.get('reliability') or 0), -o['dph_total']))


def _pick_offer(offers, requested_gpu):
    """Best offer of the requested GPU class if the user picked a speed tier
    and that class is still on the market; otherwise best overall. 'Best' =
    most reliable within +10% of the cheapest (see _best_of), on offers
    already stripped of blacklisted hosts and bait prices by _filter_offers."""
    if requested_gpu:
        matches = [o for o in offers if (o.get('gpu_name') or '') == requested_gpu]
        if matches:
            return _best_of(matches)
    return _best_of(offers)


def _provision(run):
    """Search offers and create the instance, honoring the launch-time GPU
    choice when the picked class is still available.
    LEAK-SAFE: any failure after create_instance destroys the instance."""
    c = cfg.get('cloud') or {}
    params = json.loads(run.train_params or '{}')
    fam = params.get('train_type') or 'zimage'
    min_vram = (c.get('min_vram_gb') or {}).get(fam, 24)
    offers = vast_client.search_offers(
        min_vram_gb=min_vram, max_dph=c.get('max_price_per_hour', 0.80),
        min_inet_down_mbps=int(c.get('min_inet_down_mbps') or 0),
        min_reliability=float(c.get('min_reliability') or 0.98),
        min_disk_bw_mbps=int(c.get('min_disk_bw_mbps') or 0))
    if not offers:
        raise RuntimeError(
            f'no vast.ai offer matches (>= {min_vram} GB VRAM, '
            f'<= ${c.get("max_price_per_hour", 0.80)}/h) — raise the price cap in Settings')
    offer = _pick_offer(_filter_offers(offers), params.get('requested_gpu'))
    # Stamp the host identity so a boot failure can blacklist THIS machine.
    if offer.get('machine_id') is not None:
        params['machine_id'] = offer['machine_id']
        _set(run, train_params=json.dumps(params))
    template_hash = (c.get('template_hash') or '').strip()
    if template_hash:
        # Preferred path (smoke-validated 2026-07-12): the official template
        # publishes the UI behind the pod's Caddy proxy on ui_port and vast
        # generates the per-instance auth token (picked up from the instance
        # record during boot-wait). HF_TOKEN reaches the pod later via
        # ensure_settings(), not env.
        token = ''
        instance_id = vast_client.create_instance(
            offer['offer_id'], disk_gb=int(c.get('disk_gb') or 60),
            label=run.vast_label, template_hash=template_hash,
            image=(c.get('image') or None))
    else:
        # Raw-image fallback (config escape hatch): direct port publish +
        # our own bearer token on the UI itself.
        token = pysecrets.token_urlsafe(24)
        port = int(c.get('ui_port') or 18675)
        env = {'AI_TOOLKIT_AUTH': token, f'-p {port}:{port}': '1'}
        hf = cfg.secret('HF_TOKEN')
        if hf:
            env['HF_TOKEN'] = hf
        instance_id = vast_client.create_instance(
            offer['offer_id'], disk_gb=int(c.get('disk_gb') or 60),
            label=run.vast_label, image=c.get('image'), env=env,
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


def request_stop(run_id=None) -> bool:
    if run_id is not None:
        run = CloudTrainingRun.query.get(int(run_id))
        if not run or run.status not in ACTIVE_STATES:
            return False
        _stop_event_for(run.id).set()
        return True
    actives = get_active_runs()
    for run in actives:
        _stop_event_for(run.id).set()
    return bool(actives)


def reconcile_orphans(app) -> int:
    """Boot-time safety net: destroy every 'lds-*' vast instance that no
    active run owns. GENUINELY never raises (boot must not be blocked): the
    whole body — app_context included — sits under a blanket except, so an
    unexpected failure outside the vast_client calls (db not ready, config
    error...) is logged and returns the count destroyed so far.

    error_pod_kept policy: a run in that status deliberately kept its pod
    alive (checkpoint download failed at run completion) so the user can
    recover the checkpoint manually. That pod must NOT be destroyed like a
    plain orphan -- it is spared while `run.finished_at` is within
    cloud.max_runtime_minutes of now, and only reaped past that window (with
    the run annotated, status left untouched -- terminal states stay
    terminal)."""
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
            keep = {str(r.vast_instance_id) for r in get_active_runs() if r.vast_instance_id}
            c = cfg.get('cloud') or {}
            max_seconds = int(c.get('max_runtime_minutes') or 480) * 60
            now = datetime.utcnow()
            kept_by_instance = {
                str(r.vast_instance_id): r
                for r in CloudTrainingRun.query.filter_by(status='error_pod_kept').all()
                if r.vast_instance_id}
            for inst in instances:
                label = inst.get('label') or ''
                if not label.startswith('lds-'):
                    continue
                iid = str(inst['instance_id'])
                if iid in keep:
                    continue
                kept_run = kept_by_instance.get(iid)
                if kept_run is not None:
                    # No finished_at (shouldn't happen -- every writer stamps it) means
                    # the recovery window can't be established: fail toward the leak-safety
                    # invariant (reap) rather than sparing an unbounded pod.
                    if kept_run.finished_at and \
                            (now - kept_run.finished_at).total_seconds() <= max_seconds:
                        continue    # still within the manual-recovery window -> spare
                    try:
                        if vast_client.destroy_instance(inst['instance_id']):
                            destroyed += 1
                            logger.warning('reconcile: reaped expired error_pod_kept '
                                           'pod %s (%s)', inst['instance_id'], label)
                            _set(kept_run, error=(kept_run.error or '') +
                                 ' — pod reaped after the recovery window')
                    except Exception as e:
                        logger.warning('reconcile: destroy %s failed: %s',
                                       inst['instance_id'], e)
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


def _start_monitor_for_app(app, run_id):
    """Like _start_monitor but usable outside a request context (boot)."""
    t = threading.Thread(
        target=_monitor, args=(app, run_id), daemon=True, name=f'cloud-train-{run_id}')
    _monitor_threads[int(run_id)] = t
    t.start()


def _start_monitor(run_id):
    from flask import current_app
    _start_monitor_for_app(current_app._get_current_object(), run_id)


def boot_recover(app):
    """Called once at startup (daemon thread). Never raises: a boot recovery
    bug must not prevent the app from serving requests. (1) reconcile any
    'lds-*' pod the DB no longer accounts for; (2) if a run was active when
    the app last closed and its pod was already created, resume monitoring
    it (the pod kept training/uploading in our absence); (3) if it never got
    a pod (crashed during 'preparing'), there is nothing to resume -> flip
    it to 'error' so its slot is freed. Iterates every active run (not just
    one) so a restart with several concurrent runs resumes all of them."""
    try:
        reconcile_orphans(app)
        with app.app_context():
            if not cfg.secret('VAST_API_KEY'):
                return
            for run in get_active_runs():
                if run.vast_instance_id:
                    logger.info('resuming cloud run %s (pod %s kept training)',
                                run.id, run.vast_instance_id)
                    _start_monitor_for_app(app, run.id)
                else:
                    _set(run, status='error', finished_at=datetime.utcnow(),
                         error='app restarted before the pod was created')
    except Exception:
        logger.exception('cloud boot recovery failed')


POLL_SECONDS = 10
_CKPT_SYNC_EVERY_POLLS = 12          # mid-run checkpoint mirror every ~2 min
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
            _stop_events.pop(int(run_id), None)
            _monitor_threads.pop(int(run_id), None)
            return
        stop_event = _stop_event_for(run_id)
        c = cfg.get('cloud') or {}
        max_seconds = int(c.get('max_runtime_minutes') or 480) * 60
        # The runtime cap must survive restarts: anchor it to the run's durable
        # created_at (backdate the local clock by the run's age), not to this
        # thread's start. The boot-wait timeout keeps its own fresh anchor —
        # a resumed monitor legitimately gets a new 15-min readiness window.
        run_age = max(0.0, (datetime.utcnow() - (run.created_at or datetime.utcnow())).total_seconds())
        cap_anchor = _now() - run_age
        boot_started = _now()
        try:
            # -- heavy launch work, moved off the HTTP path (see launch) ----
            _prepare_staging(run)
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
            template_mode = bool((c.get('template_hash') or '').strip())
            ready_timeout = (int(c.get('ready_timeout_minutes') or 0) * 60
                             or READY_TIMEOUT_SECONDS)
            _set(run, phase_detail='Waiting for the pod to boot')
            port = int(c.get('ui_port') or 18675)
            if template_mode and port == 8675:
                # 8675 is the pre-template default that Settings saves may have
                # baked into config.json; the official template only publishes
                # the UI behind the pod proxy on 18675 — a stale 8675 makes the
                # boot-wait spin for its whole budget (observed live 2026-07-12).
                logger.warning('cloud.ui_port=8675 is stale for template mode — using 18675')
                port = 18675
            while True:
                # A transient vast API hiccup is just "not ready yet" -- only
                # READY_TIMEOUT_SECONDS may fail the boot wait, never a single
                # 502 that would destroy a pod about to come up fine.
                try:
                    inst = vast_client.get_instance(run.vast_instance_id)
                except vast_client.VastError as e:
                    logger.warning('boot-wait: vast API hiccup (%s) — retrying', e)
                    inst = None
                # Template launches authenticate with the vast-generated
                # per-instance token (the pod's Caddy proxy accepts it as a
                # Bearer header) — pick it up as soon as the record shows it.
                if inst and not run.auth_token and inst.get('jupyter_token'):
                    _set(run, auth_token=inst['jupyter_token'])
                base = vast_client.derive_base_url(inst, port) if inst else None
                ready = False
                if base:
                    if run.base_url != base:
                        _set(run, base_url=base)
                    ready = _make_remote(run).is_ready()
                    if ready:
                        break
                # Honor "Stop run" DURING boot too — but only on a pod that is
                # NOT ready yet (a ready pod breaks out above and the training
                # loop handles the stop normally). Without this, the boot-wait
                # spun its whole 25-min budget on a dead host while the stop
                # button silently did nothing (observed live 2026-07-13, a
                # 5090 stuck in 'loading'). No job exists yet -> terminate.
                if stop_event.is_set():
                    stop_event.clear()
                    # A user killing a boot this late is almost always a stuck
                    # host — blacklist it like a timeout would. An early stop
                    # (changed their mind) says nothing about the host.
                    if _now() - boot_started > 8 * 60:
                        _blacklist_host(_run_machine_id(run),
                                        'user stopped a boot stuck past 8 min')
                    _finish(run, 'stopped', detail='Stopped by user during boot')
                    return
                # Live telemetry: surface WHERE the boot is stuck (image pull,
                # port publication, UI warm-up) in the UI phase line and the
                # log — runs #3/#4 died blind on 'Waiting for the pod to boot'.
                st = (inst or {}).get('actual_status') or 'not listed yet'
                has_ports = bool(((inst or {}).get('ports') or {}).get(f'{port}/tcp'))
                stage = (f'pod {st}' if not has_ports
                         else 'pod up — waiting for the UI to answer')
                detail = f'Waiting for the pod to boot — {stage}'
                if run.phase_detail != detail:
                    logger.info('boot-wait run %s: status=%s port_%s_published=%s '
                                'base=%s ready=%s', run.id, st, port, has_ports,
                                base or '-', ready)
                    _set(run, phase_detail=detail)
                if _now() - boot_started > ready_timeout:
                    # This host burned the whole boot budget — skip it for the
                    # next few days so a relaunch can't land on it again.
                    _blacklist_host(_run_machine_id(run),
                                    'pod did not become ready in time')
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
            # Stall watchdog state: armed only once training has produced its
            # first step (before that — base download, quantization, latent
            # caching — the watchdog stays INACTIVE; those phases are covered
            # by ready_timeout + max_runtime).
            stall_seconds = int(c.get('stall_timeout_minutes') or 30) * 60
            last_step = -1
            last_progress_ts = _now()
            last_ok = _now()
            polls = 0
            while True:
                if _now() - cap_anchor > max_seconds:
                    try:
                        remote.stop_job(job_id)
                    except Exception:
                        pass
                    _try_download_checkpoint(run, remote, allow_stale=True)
                    _finish(run, 'stopped',
                            detail='Max runtime reached — pod terminated',
                            error='max runtime cap hit')
                    return
                if stop_event.is_set():
                    stop_event.clear()
                    _set(run, phase_detail='Stopping on user request')
                    try:
                        remote.stop_job(job_id)
                    except Exception:
                        pass
                    _try_download_checkpoint(run, remote, allow_stale=True)
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
                # Mid-run checkpoint mirror, throttled (~2 min at 10 s polls):
                # list_files is cheap, but no need to hammer it every poll —
                # the pod only writes a new save every save_every steps.
                polls += 1
                if polls % _CKPT_SYNC_EVERY_POLLS == 0:
                    _sync_latest_checkpoint(run, remote)
                status = job.get('status')
                info = job.get('info') or ''
                _set(run, phase_detail=f"{status}: {info}"[:500])

                if status == 'completed':
                    ok = _try_download_checkpoint(run, remote)
                    if not ok:
                        # A host that cannot DELIVER its result (even through
                        # the resume loop) is a bad host — skip it next time.
                        _blacklist_host(_run_machine_id(run),
                                        'could not serve the final checkpoint')
                        # LoRA > a few minutes of pod time: keep the pod for
                        # manual recovery; max-runtime/reconcile will reap it.
                        _set(run, status='error_pod_kept',
                             error='checkpoint download failed — pod kept, '
                                   f'recover manually at {run.base_url}',
                             finished_at=datetime.utcnow())
                        return
                    _import_result(run)
                    _mirror_into_local_run(run)
                    _finish(run, 'done', detail='Training complete')
                    return
                if status in ('error', 'stopped'):
                    _try_download_checkpoint(run, remote, allow_stale=True)
                    _finish(run, 'error' if status == 'error' else 'stopped',
                            detail=f'Remote job {status}', error=info or status)
                    return
                # -- stall watchdog: guiding rule — NEVER kill a run that
                # progresses. The elif keeps a progressing poll from ever
                # evaluating the stall clock (a coarse test clock jumping in
                # large strides per call must not misfire on a healthy run).
                step = job.get('step') or 0
                if step > last_step:
                    last_step = step
                    last_progress_ts = _now()
                elif last_step > 0 and (_now() - last_progress_ts) > stall_seconds:
                    try:
                        remote.stop_job(job_id)
                    except Exception:
                        pass
                    _try_download_checkpoint(run, remote, allow_stale=True)
                    _finish(run, 'error',
                            detail='Stalled — no step progress for '
                                   f'{stall_seconds // 60} min; pod terminated',
                            error='stall watchdog')
                    return
                _sleep(POLL_SECONDS)
        except Exception as e:
            logger.exception('cloud run %s failed', run_id)
            # A pod that died mid-run is a HOST-quality signal (live
            # 2026-07-13: a krea run lost at ~$0.93 when its pod went
            # unreachable) — skip this machine for the next launches.
            if 'unreachable' in str(e).lower():
                _blacklist_host(_run_machine_id(run), 'pod died mid-run (unreachable)')
            _finish(run, 'error', detail='Run failed', error=str(e)[:500])
        finally:
            # This run's slot in the module maps is done with — drop it so
            # they cannot grow unbounded across the app's lifetime with many
            # concurrent runs coming and going.
            _stop_events.pop(int(run_id), None)
            _monitor_threads.pop(int(run_id), None)
            _sync_state.pop(int(run_id), None)


def _pull_log_and_samples(run, remote, job_id):
    """Mirror remote log + new samples into staging so cloud_progress reuses
    the exact local parsing/serving machinery. Never raises."""
    try:
        text = remote.get_log(job_id)
        with open(os.path.join(run.staging_dir, 'training.log'), 'w',
                  encoding='utf-8', errors='replace') as fh:
            fh.write(text)
    except Exception as e:
        logger.debug('log mirror failed: %s', e)
    try:
        samples_dir = os.path.join(run.staging_dir, 'samples')
        have = set(os.listdir(samples_dir))
        for remote_path in remote.get_samples(job_id):
            name = os.path.basename(remote_path.replace('\\', '/'))
            if name and name not in have:
                remote.download_sample(remote_path,
                                       os.path.join(samples_dir, name))
    except Exception as e:
        logger.debug('sample mirror failed: %s', e)


def _newest_remote_checkpoint(remote, job_id):
    """The newest .safetensors file entry ({'path', 'size'}), or None.
    ai-toolkit zero-pads step numbers, so lexicographic order IS step order."""
    files = [f for f in remote.list_files(job_id)
             if f.get('path', '').endswith('.safetensors')]
    if not files:
        return None
    return sorted(files, key=lambda f: f['path'])[-1]


def _fetch_checkpoint(run, remote, ckpt, timeout=None, attempts=3) -> str:
    """Download the checkpoint entry ({'path','size'}) into staging and return
    the local path. Skips the transfer when this exact save is already local
    (the mid-run sync usually got there first). Two integrity layers:
    - a KILLED transfer never lands at dest (RemoteAiToolkit._download's own
      .part-then-rename; no second layer here — it produced '.part.part');
    - a TRUNCATED transfer that ends with a clean EOF (observed live
      2026-07-13: pods closing the stream after a few chunks while training)
      is caught by comparing the byte size against list_files' size — a short
      file is deleted and the fetch fails rather than registering garbage."""
    remote_path = ckpt['path']
    name = os.path.basename(remote_path.replace('\\', '/'))
    dest = os.path.join(run.staging_dir, name)
    if run.checkpoint_local_path and os.path.isfile(dest) \
            and os.path.basename(run.checkpoint_local_path) == name:
        return dest
    remote.download_public_file(remote_path, dest, timeout=timeout,
                                expected_size=ckpt.get('size'), attempts=attempts)
    want = int(ckpt.get('size') or 0)
    got = os.path.getsize(dest)
    if want and got != want:
        try:
            os.remove(dest)
        except OSError:
            pass
        raise RuntimeError(f'truncated download of {name}: {got}/{want} bytes')
    return dest


_SYNC_DL_TIMEOUT = 60      # opportunistic pull: fail fast, the loop must not hang
_SYNC_MAX_FAILS = 3        # give up on a save after this; a NEWER save retries
_sync_state = {}           # run_id -> {'name': save filename, 'fails': int}


def _sync_latest_checkpoint(run, remote):
    """Mid-run mirror of the pod's newest SAVE: if the host dies at step 3000
    the local copy of the step-2750 save survives, instead of everything being
    lost because downloads only happened at run end (user-observed gap,
    2026-07-13). Never raises, never flips the run's status; keeps only the
    newest save locally (a LoRA is 100-300 MB, saves accumulate).

    Some pods cannot serve big files WHILE training (observed live: streams
    die after a few chunks) — after _SYNC_MAX_FAILS failed attempts on the
    same save we stop retrying it (a newer save resets the counter), and each
    attempt is capped at _SYNC_DL_TIMEOUT so a trickling stream cannot hold
    the monitor loop — and with it the stop button — for minutes."""
    try:
        ckpt = _newest_remote_checkpoint(remote, run.remote_job_id)
        if not ckpt:
            return
        name = os.path.basename(ckpt['path'].replace('\\', '/'))
        st = _sync_state.get(run.id)
        if st and st.get('name') == name and st.get('fails', 0) >= _SYNC_MAX_FAILS:
            return
        prev = run.checkpoint_local_path
        try:
            dest = _fetch_checkpoint(run, remote, ckpt,
                                     timeout=_SYNC_DL_TIMEOUT)
        except Exception as e:
            st = _sync_state.setdefault(run.id, {'name': name, 'fails': 0})
            if st.get('name') != name:
                st.update(name=name, fails=0)
            st['fails'] += 1
            # First failure at WARNING so it is visible in the log viewer;
            # repeats at DEBUG (the give-up cap bounds them anyway).
            log = logger.warning if st['fails'] == 1 else logger.debug
            log('mid-run checkpoint sync of %s failed (attempt %s/%s): %s',
                name, st['fails'], _SYNC_MAX_FAILS, e)
            return
        _sync_state.pop(run.id, None)
        if dest != prev:
            _set(run, checkpoint_local_path=dest)
            if prev and os.path.isfile(prev):
                try:
                    os.remove(prev)
                except OSError:
                    pass
    except Exception as e:
        logger.debug('mid-run checkpoint sync failed: %s', e)


def _try_download_checkpoint(run, remote, allow_stale=False) -> bool:
    """Download the newest .safetensors into staging. False on failure.
    allow_stale (rescue paths — stop/stall/cap): when the pod can't serve the
    newest save anymore, an already-synced OLDER save still counts as success.
    The COMPLETION path must stay strict (allow_stale=False): falling back to
    an older save there would silently discard the final training steps —
    error_pod_kept keeps the pod so the user can recover the real result."""
    try:
        ckpt = _newest_remote_checkpoint(remote, run.remote_job_id)
        if ckpt:
            # Large attempts budget: a sick-proxy host cutting the stream
            # every ~0.5-2 MB still delivers an 85 MB file via ~100 resumed
            # connections (validated live 2026-07-13, run #7's manual rescue).
            dest = _fetch_checkpoint(run, remote, ckpt, attempts=400)
            _set(run, status='downloading', checkpoint_local_path=dest,
                 phase_detail=f'Downloaded {os.path.basename(dest)}')
            return True
    except Exception as e:
        logger.warning('checkpoint download failed: %s', e)
    return bool(allow_stale and run.checkpoint_local_path
                and os.path.isfile(run.checkpoint_local_path))


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
                             src_dir=run.staging_dir,
                             version=params.get('version'))
    except Exception as e:
        logger.warning('cloud import into ComfyUI failed: %s', e)


def _mirror_into_local_run(run):
    """Copy the downloaded cloud checkpoint into the LOCAL ai-toolkit run dir,
    renamed to the local convention (`lora_<trigger>[_<step>].safetensors`), so
    cloud results behave exactly like local ones everywhere downstream: the
    panel's checkpoint list, the Resume-or-Fresh prompt, Continue training.
    (The user looked for the result in ai-toolkit/output and found it empty —
    2026-07-13.) No-op when ai-toolkit isn't configured locally; best-effort,
    never fails the run."""
    try:
        if not run.checkpoint_local_path or not os.path.isfile(run.checkpoint_local_path):
            return
        params = json.loads(run.train_params or '{}')
        # cloud trains on the OFFICIAL base only -> base_model=''
        run_dir = lt._run_dir('local', run.dataset_id, base_model='',
                              family=params.get('train_type'))
        os.makedirs(run_dir, exist_ok=True)
        src_name = os.path.basename(run.checkpoint_local_path)
        m = re.search(r'_(\d{6,})\.safetensors$', src_name)
        base = os.path.basename(os.path.normpath(run_dir))     # lora_<trigger>
        dest_name = f'{base}_{m.group(1)}.safetensors' if m else f'{base}.safetensors'
        dest = os.path.join(run_dir, dest_name)
        if os.path.exists(dest):
            # A LOCAL run of the same dataset+family already produced this
            # exact name (the unsuffixed FINAL collides whenever both worlds
            # completed a run) — never clobber local work. The cloud result
            # stays available in staging, ComfyUI and the hub's ⬇ button.
            logger.warning('local run dir already has %s — cloud mirror skipped '
                           '(local checkpoint left untouched)', dest_name)
            return
        shutil.copy2(run.checkpoint_local_path, dest)
        logger.info('mirrored cloud checkpoint into local run dir: %s/%s',
                    run_dir, dest_name)
    except Exception as e:
        # RuntimeError from _run_dir = ai-toolkit not configured -> fine
        logger.debug('local run-dir mirror skipped: %s', e)


def _cost_estimate(run) -> float:
    if not run.price_per_hour:
        return 0.0
    end = run.finished_at or datetime.utcnow()
    hours = max(0.0, (end - run.created_at).total_seconds() / 3600.0)
    return round(run.price_per_hour * hours, 2)


def month_spend_usd() -> float:
    """Total cost of the runs STARTED since the 1st of the current month
    (UTC). A run's cost = price_per_hour x (finished_at or now - created_at);
    runs that never got a priced pod (price_per_hour NULL) count for $0."""
    now = datetime.utcnow()
    month_start = datetime(now.year, now.month, 1)
    total = 0.0
    for r in (CloudTrainingRun.query
              .filter(CloudTrainingRun.created_at >= month_start).all()):
        if not r.price_per_hour or not r.created_at:
            continue
        end = r.finished_at or now
        total += r.price_per_hour * max(0.0, (end - r.created_at).total_seconds() / 3600.0)
    return total


def _dataset_name(dataset_id):
    """Human-readable dataset name for the cloud-runs hub — the run only stores
    dataset_id. Best-effort: a since-deleted dataset yields None, never a crash."""
    try:
        from ..models import FaceDataset
        ds = FaceDataset.query.get(dataset_id)
        return ds.name if ds is not None else None
    except Exception:
        return None


def _run_payload(run) -> dict:
    return {'run_id': run.id, 'dataset_id': run.dataset_id, 'status': run.status,
            'run_name': run.run_name, 'dataset_name': _dataset_name(run.dataset_id),
            'vast_instance_id': run.vast_instance_id,   # for the per-run "console ↗" tooltip
            'phase_detail': run.phase_detail, 'gpu': run.gpu_name,
            'price_per_hour': run.price_per_hour,
            'cost_estimate': _cost_estimate(run), 'error': run.error,
            # isfile, not just a stored path: the user may delete staging
            # files by hand (Explorer) — a ready flag pointing at a missing
            # file yields a download button that 404s.
            'checkpoint_ready': bool(run.checkpoint_local_path
                                     and os.path.isfile(run.checkpoint_local_path)),
            'train_type': _run_family(run),
            'version': _run_param(run, 'version'),
            'created_at': run.created_at.isoformat() if run.created_at else None,
            'finished_at': run.finished_at.isoformat() if run.finished_at else None}


def cloud_status() -> dict:
    actives = get_active_runs()
    c = cfg.get('cloud') or {}
    limit = max(1, int((c.get('max_concurrent_runs') or 1)))
    last = (CloudTrainingRun.query
            .order_by(CloudTrainingRun.id.desc()).first())
    return {'configured': bool(cfg.secret('VAST_API_KEY')), 'limit': limit,
            'actives': [_run_payload(r) for r in actives],
            # compat: single 'active' field for old frontend/tests, first of actives
            'active': _run_payload(actives[0]) if actives else None,
            'total_price_per_hour': round(sum(r.price_per_hour or 0 for r in actives), 4),
            # budget guardrails: what this month already cost, the configured
            # ceiling (0 = unlimited), and the runtime cap the frontend uses
            # for its worst-case cost estimate.
            'month_spend': round(month_spend_usd(), 2),
            'monthly_budget': float(c.get('monthly_budget_usd') or 0),
            'max_runtime_minutes': int(c.get('max_runtime_minutes') or 480),
            'last': _run_payload(last) if last else None}


def all_runs(limit: int = 20) -> dict:
    """Everything the dedicated Cloud-runs hub needs in one call: the active
    runs (manage/watch) plus the most recent TERMINAL runs (history: outcome +
    checkpoint download), newest first, and the same budget summary as
    cloud_status(). Kept separate from cloud_status() so the per-dataset panel
    poll stays lean (it never needs the history)."""
    actives = get_active_runs()
    c = cfg.get('cloud') or {}
    limit = max(1, min(int(limit or 20), 100))
    recent = (CloudTrainingRun.query
              .filter(CloudTrainingRun.status.notin_(ACTIVE_STATES))
              .order_by(CloudTrainingRun.id.desc()).limit(limit).all())
    return {'configured': bool(cfg.secret('VAST_API_KEY')),
            'limit': max(1, int((c.get('max_concurrent_runs') or 1))),
            'actives': [_run_payload(r) for r in actives],
            'recent': [_run_payload(r) for r in recent],
            'total_price_per_hour': round(sum(r.price_per_hour or 0 for r in actives), 4),
            'month_spend': round(month_spend_usd(), 2),
            'monthly_budget': float(c.get('monthly_budget_usd') or 0)}


def gpu_tiers(user_id, dataset_id, train_type=None, steps=None) -> dict:
    """Live vast.ai offers for THIS dataset+family, grouped by GPU class
    (cheapest offer per class), ranked slowest -> fastest, each annotated with
    an approximate training time and total run cost. Read-only: rents nothing.
    The launch then re-searches and rents the cheapest live offer of the chosen
    class. Raises the same guards as launch (no key / dataset / SDXL)."""
    if not cfg.secret('VAST_API_KEY'):
        raise RuntimeError('vast.ai API key is not configured — add it in Settings')
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    fam = fds.normalize_train_type(train_type or getattr(ds, 'train_type', None))
    if fam == 'sdxl':
        raise ValueError('SDXL training needs a local base checkpoint — '
                         'cloud training supports Z-Image and Krea for now')
    if fam == 'flux':
        raise ValueError('FLUX.1 training is local-only for now — '
                         'cloud training supports Z-Image and Krea')
    n_steps = int(steps) if steps else lt.default_steps(ds)
    c = cfg.get('cloud') or {}
    min_vram = (c.get('min_vram_gb') or {}).get(fam, 24)
    price_cap = c.get('max_price_per_hour', 0.80)
    overhead_min = float(c.get('pod_overhead_minutes') or 0)
    # A wider scan than the launch default so several GPU classes surface (the
    # user is choosing between them, not taking the single cheapest). Same
    # quality filters as the launch so the shown tiers match what gets rented.
    offers = _filter_offers(vast_client.search_offers(
        min_vram_gb=min_vram, max_dph=price_cap,
        limit=int(c.get('offer_scan_limit') or 100),
        min_inet_down_mbps=int(c.get('min_inet_down_mbps') or 0),
        min_reliability=float(c.get('min_reliability') or 0.98),
        min_disk_bw_mbps=int(c.get('min_disk_bw_mbps') or 0)))
    cheapest_by_gpu = {}
    for o in offers:
        name = o.get('gpu_name') or 'GPU'
        cur = cheapest_by_gpu.get(name)
        dph = o.get('dph_total')
        if cur is None or (dph is not None and (cur.get('dph_total') is None
                           or dph < cur['dph_total'])):
            cheapest_by_gpu[name] = o
    max_runtime = int(c.get('max_runtime_minutes') or 480)
    tiers = []
    for name, o in cheapest_by_gpu.items():
        dph = o.get('dph_total')
        est_min = gpu_speed.estimate_minutes(name, fam, n_steps)
        # Cost bills the whole pod life: training + boot/download/quantize.
        est_cost = (round(dph * (est_min + overhead_min) / 60.0, 2)
                    if dph is not None else None)
        tiers.append({
            'gpu_name': name, 'offer_id': o.get('offer_id'),
            'dph_total': round(dph, 4) if dph is not None else None,
            'gpu_ram_gb': o.get('gpu_ram_gb'),
            'speed': round(gpu_speed.speed_factor(name), 2),
            'est_minutes': int(round(est_min)), 'est_cost': est_cost,
            # A tier slower than the runtime cap would be KILLED mid-training
            # (checkpoint rescued, but steps lost) — warn at pick time.
            'exceeds_cap': (est_min + overhead_min) > max_runtime,
        })
    # slowest -> fastest (matches the launch dialog); ties broken by price.
    tiers.sort(key=lambda t: (t['speed'], t['dph_total']
                              if t['dph_total'] is not None else 9e9))
    return {'tiers': tiers, 'steps': n_steps, 'family': fam,
            'max_price_per_hour': price_cap,
            'max_runtime_minutes': max_runtime}


def cloud_checkpoints(dataset_id, train_type=None) -> list:
    """Locally-synced cloud checkpoints of this dataset (+family filter), one
    per run, newest run first — INCLUDING an in-progress run's latest synced
    save (user-observed gap: step 1000 reached, save synced to staging, panel
    list empty). Only files that actually exist are listed (hand-deleting in
    Explorer must not yield 404 buttons)."""
    fam = fds.normalize_train_type(train_type) if train_type else None
    out = []
    for run in (CloudTrainingRun.query.filter_by(dataset_id=dataset_id)
                .order_by(CloudTrainingRun.id.desc()).all()):
        p = run.checkpoint_local_path
        if not p or not os.path.isfile(p):
            continue
        if fam and (_run_family(run) or fam) != fam:
            continue
        name = os.path.basename(p)
        m = re.search(r'_(\d{6,})\.safetensors$', name)
        step = int(m.group(1)) if m else int(_run_param(run, 'steps') or 0)
        out.append({'filename': name, 'step': step, 'cloud': True,
                    'run_id': run.id, 'version': _run_param(run, 'version'),
                    'final': bool(not m and run.status == 'done'),
                    'active': run.status in ACTIVE_STATES,
                    'trained_at': run.created_at.isoformat() if run.created_at else None})
    return out


def delete_cloud_checkpoint(dataset_id, run_id, filename) -> str:
    """Move a cloud run's synced checkpoint to the trash. The run must belong
    to the dataset and be TERMINAL (deleting an active run's save is pointless
    — the sync re-downloads it). Clears checkpoint_local_path when it pointed
    at the trashed file."""
    run = CloudTrainingRun.query.get(int(run_id))
    if not run or run.dataset_id != int(dataset_id) or not run.staging_dir:
        raise ValueError('unknown cloud run')
    if run.status in ACTIVE_STATES:
        raise ValueError('this cloud run is still active — its save would just '
                         'be re-synced; stop the run first')
    allowed = {f for f in os.listdir(run.staging_dir)
               if f.lower().endswith('.safetensors')}
    if filename not in allowed:
        raise ValueError('unknown checkpoint')
    from . import trash
    trash.send_to_trash(os.path.join(run.staging_dir, filename),
                        context=f'cloudckpt_run{run.id}')
    if run.checkpoint_local_path \
            and os.path.basename(run.checkpoint_local_path) == filename:
        _set(run, checkpoint_local_path=None)
    return filename


def purge_finished_runs() -> dict:
    """Hub 'Clean finished runs': move the staging dirs of TERMINAL runs to the
    trash — dataset copies, samples and checkpoint duplicates of results that
    are already imported/mirrored. Active runs and error_pod_kept (manual
    recovery may still be under way) are spared. DB rows stay (history)."""
    from . import trash
    purged = 0
    freed = 0
    for run in CloudTrainingRun.query.all():
        if run.status in ACTIVE_STATES or run.status == 'error_pod_kept':
            continue
        sd = run.staging_dir
        if not sd or not os.path.isdir(sd):
            continue
        try:
            freed += lt._dir_size(sd)
            trash.send_to_trash(sd, context=f'staging_run{run.id}')
            purged += 1
            if run.checkpoint_local_path:
                _set(run, checkpoint_local_path=None)
        except OSError as e:
            logger.warning('purge: could not trash %s: %s', sd, e)
    return {'purged_runs': purged, 'freed_bytes': freed}


def cloud_progress(user_id, dataset_id, train_type=None) -> dict:
    """Same shape as lt.training_progress + cloud phase/cost fields, built
    from the staging mirror (log + samples) written by the monitor. With
    train_type, reads THAT family's newest run (several families may train
    the same dataset in parallel)."""
    run = latest_run_for(dataset_id, train_type)
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
