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
_auto_retry_lock = threading.Lock()
# Flask serves requests from multiple threads in the portable app.  SQLite
# cannot express the two launch invariants (global active-run cap and
# per-dataset/family uniqueness) as a simple UNIQUE constraint because both
# depend on a set of non-terminal statuses.  Serialize the final guardrail
# re-check and reservation row instead, before any monitor can rent a pod.
_launch_reservation_lock = threading.Lock()
_UNSET = object()
_TRAIN_SETTINGS_SNAPSHOT = 'train_settings_snapshot'
# Slider LoRA mode (Beta) settings live in a DEDICATED column (train_slider),
# not train_settings, so they need their own per-run snapshot: the pod job is
# built minutes after launch (and later on retry/continue), and a toggle-off or
# prompt edit in that window must not retarget an already-launched slider run
# into a plain LoRA (same immutability contract as _TRAIN_SETTINGS_SNAPSHOT —
# see _RunConfigDataset / incident 2026-07-14).
_TRAIN_SLIDER_SNAPSHOT = 'train_slider_snapshot'
_CONFIRMATION_FLAGS = (
    'allow_caption_mismatch',
    'allow_uncaptioned',
    'allow_caption_quality',
    # Custom-weights arch confirm (CUSTOM_WEIGHTS_UNVERIFIED contract): replayed
    # on retry/continue like the caption flags — the base_model is replayed
    # verbatim too, so a confirmed file stays confirmed.
    'allow_unverified_weights',
    # « Continue anyway » ack (readiness floor blocker): replayed on retry/continue
    # like the caption flags, and stamped into train_params so a thin cloud run is
    # honestly explainable in the Runs hub.
    'allow_not_ready',
)


def _confirmation_flags(params) -> dict:
    """Replay only booleans explicitly stamped by the original launch.

    Missing/corrupt legacy values are False: retrying or continuing re-exports
    the mutable current dataset, so an old successful run is never authority to
    waive today's caption guardrails.
    """
    source = params if isinstance(params, dict) else {}
    return {key: source.get(key) is True for key in _CONFIRMATION_FLAGS}


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


def active_runs_for(dataset_id):
    """Non-terminal cloud runs of ONE dataset (empty list when none). Scoped
    twin of get_active_runs(); delete_dataset uses it to refuse deleting a
    dataset out from under a running pod."""
    return (CloudTrainingRun.query
            .filter_by(dataset_id=int(dataset_id))
            .filter(CloudTrainingRun.status.in_(ACTIVE_STATES))
            .order_by(CloudTrainingRun.id.asc()).all())


def _assert_official_base_reachable(repo_id, token, timeout=8):
    """Fail the launch when the account cannot actually download `repo_id`.

    Hugging Face answers **200 on the model's metadata** for a gated repo you have
    not been granted — only fetching a FILE returns 403. So this asks for the file
    listing under auth, which is subject to the same gate, and reads the status.

    FAIL-OPEN on anything that is not an outright refusal: a timeout, DNS failure or
    HF outage must never block a launch that would have worked. The pod remains the
    real authority; this only converts the ONE failure we can predict — a gate the
    user has never accepted — into a message that arrives before the bill."""
    if not repo_id:
        return
    import urllib.error
    import urllib.request
    req = urllib.request.Request(
        f'https://huggingface.co/api/models/{repo_id}/tree/main',
        headers={'Authorization': f'Bearer {token}'} if token else {})
    try:
        urllib.request.urlopen(req, timeout=timeout).read(1)
    except urllib.error.HTTPError as e:
        if e.code not in (401, 403):
            return                          # 404 / 5xx: not our call to make
        raise ValueError(
            f'Hugging Face refuses access to {repo_id}, which the rented GPU has to '
            f'download. Open https://huggingface.co/{repo_id} while signed in with '
            'the account your HF token belongs to, accept the licence ("Agree and '
            'access repository"), then launch again. Approval is usually instant. '
            'Nothing was rented, so this run cost nothing.') from None
    except Exception:                        # noqa: BLE001 — offline/outage: fail open
        return


def _assert_launch_guardrails(dataset_id, fam):
    """Raise when a cloud launch cannot reserve an active slot.

    Callers may use this once as a cheap fast-fail before expensive preflight,
    but the authoritative call must happen while ``_launch_reservation_lock``
    is held and immediately before inserting the ``preparing`` row.
    """
    actives = get_active_runs()
    limit = max(1, int((cfg.get('cloud.max_concurrent_runs') or 1)))
    # Uniqueness is per (dataset, family): a zimage run and a krea run may
    # train the same dataset in parallel. An active run whose family is
    # unknown (pre-feature row) blocks every family of its dataset, out of
    # caution.
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


class _RunConfigDataset:
    """Read-only view of a dataset whose config inputs are forced to the values
    stamped for this run; every other attribute delegates to the real dataset.

    The cloud monitor builds the pod job through this view so the job's
    architecture/variant come from what the run was LAUNCHED with — never from
    the dataset's *current* row. Each launch persists ds.train_type /
    ds.train_variant (last writer wins) and the monitor rebuilds the config
    minutes later, at pod boot; a second launch on the same dataset (or a
    /train-type change) between this run's launch and its boot would otherwise
    retarget its architecture. Incident 2026-07-14: a Krea run launched first, a
    Z-Image run 28 min later persisted 'zimage', and the Krea pod — booting after
    that — would have been rebuilt as Z-Image under a Krea name (wrong arch on a
    rented GPU). build_job_config only READS the dataset, so a view is enough:
    no DB mutation, nothing to restore, and both concurrent runs stay isolated."""

    def __init__(self, ds, train_type, train_variant, train_base_model='',
                 train_settings_snapshot=_UNSET, train_slider_snapshot=_UNSET):
        self._ds = ds
        self._train_type = train_type
        self._train_variant = train_variant
        self._train_base_model = train_base_model
        self._train_settings_snapshot = train_settings_snapshot
        self._train_slider_snapshot = train_slider_snapshot

    @property
    def train_type(self):
        return (self._train_type if self._train_type is not None
                else getattr(self._ds, 'train_type', None))

    @property
    def train_variant(self):
        return (self._train_variant if self._train_variant is not None
                else getattr(self._ds, 'train_variant', None))

    @property
    def train_base_model(self):
        # Cloud runs always stamp their launch-time selection.  In particular,
        # an empty string means the official Hugging Face base and must not
        # fall through to a base subsequently persisted on the dataset row.
        return self._train_base_model

    @property
    def train_settings(self):
        # ``None`` is a meaningful snapshot: the run was launched with the
        # family defaults. Only _UNSET means a legacy run without a snapshot.
        return (getattr(self._ds, 'train_settings', None)
                if self._train_settings_snapshot is _UNSET
                else self._train_settings_snapshot)

    @property
    def train_slider(self):
        # Slider mode (Beta) is read off this column by build_job_config
        # (slider_mode_enabled -> _apply_slider_overrides). Frozen per run like
        # train_settings: ``None`` means "not a slider run at launch"; only
        # _UNSET (a legacy row that predates the snapshot) falls back to the
        # live dataset column. A @property is resolved by normal lookup, so it
        # deliberately shadows the __getattr__ delegation below.
        return (getattr(self._ds, 'train_slider', None)
                if self._train_slider_snapshot is _UNSET
                else self._train_slider_snapshot)

    def __getattr__(self, name):
        # Reached only for attributes not resolved normally (i.e. everything
        # except _ds / _train_* / the two properties) -> delegate to the real ds.
        return getattr(self._ds, name)


def _run_config_dataset(ds, params):
    """Wrap ``ds`` so build_job_config reads THIS run's stamped recipe.

    Advanced settings must be immutable per run: the pod job is built minutes
    after launch and an automatic retry even later. Dataset edits in between
    affect future launches only. Legacy rows without a settings snapshot retain
    their historical DB fallback.
    """
    fam = params.get('train_type')
    var = params.get('variant')
    base = params.get('base_model', '')
    advanced = params.get(_TRAIN_SETTINGS_SNAPSHOT, _UNSET)
    slider = params.get(_TRAIN_SLIDER_SNAPSHOT, _UNSET)
    return _RunConfigDataset(ds, fam, var, base, advanced, slider)


def _recipe_replay_diagnostic(params):
    """Safety diagnosis for retry/continue without mutating the source run."""
    if not isinstance(params, dict):
        return None
    return lt.zimage_recipe_diagnostic(
        params.get('train_type'), params.get('variant'),
        params.get('effective_base'), params.get('training_adapter'),
        params.get('recipe_version'))


def _assert_recipe_replayable(params, action):
    diag = _recipe_replay_diagnostic(params)
    if diag and diag.get('status') in ('legacy_incompatible', 'incompatible'):
        raise ValueError(
            f'cannot {action} this run safely: {diag.get("warning")} Start a fresh '
            'run with the validated Z-Image recipe instead.')


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


def retry_cloud_run(user_id, run_id) -> dict:
    """Relance un run TERMINÉ EN ERREUR avec les paramètres exacts persistés au
    lancement d'origine (train_params) — le bouton ↻ Retry de la page Cloud.
    C'est un VRAI launch_cloud_training (pod frais, mêmes garde-fous : limite
    de runs actifs, budget, unicité par famille), pas une réanimation du pod
    mort. Les confirmations ne sont rejouées que si le lancement d'origine les
    avait explicitement enregistrées."""
    run = db.session.get(CloudTrainingRun, int(run_id))
    if not run:
        raise ValueError('unknown cloud run')
    if run.status != 'error':
        raise ValueError('only a failed run can be retried')
    try:
        p = json.loads(run.train_params or '{}')
    except ValueError:
        p = {}
    if not isinstance(p, dict):
        p = {}
    _assert_recipe_replayable(p, 'retry')
    return launch_cloud_training(
        user_id, run.dataset_id,
        steps=p.get('steps'),
        base_model=p.get('base_model', ''),
        variant=p.get('variant'),
        train_type=p.get('train_type'),
        masked=p.get('masked', True),
        **_confirmation_flags(p),
        gpu_name=p.get('requested_gpu'),
        resume_ckpt_path=p.get('resume_ckpt_path'),
        resume_step=p.get('resume_step'),
        train_settings_snapshot=p.get(_TRAIN_SETTINGS_SNAPSHOT, _UNSET),
        train_slider_snapshot=p.get(_TRAIN_SLIDER_SNAPSHOT, _UNSET))


def _run_staging_checkpoints(run) -> list:
    """This run's HARVESTED checkpoints that still live in staging (NOT the
    trash — trashed saves are moved out of staging_dir): list of
    {'filename', 'step', 'path'}, step-sorted ascending. Mirrors
    cloud_checkpoints' step extraction so 'continue' resumes from the exact same
    checkpoint the hub lists. The unsuffixed FINAL save (no _<step> suffix) is
    the run's target step count."""
    sd = run.staging_dir
    if not sd or not os.path.isdir(sd):
        return []
    target = int(_run_param(run, 'steps') or 0)
    out = []
    for name in os.listdir(sd):
        if not name.lower().endswith('.safetensors'):
            continue
        m = re.search(r'_(\d{6,})\.safetensors$', name)
        out.append({'filename': name,
                    'step': int(m.group(1)) if m else target,
                    'path': os.path.join(sd, name)})
    # step asc; a suffixed save wins ties over the unsuffixed final (deterministic).
    out.sort(key=lambda e: (e['step'], bool(re.search(r'_(\d{6,})\.safetensors$',
                                                       e['filename']))))
    return out


def _merge_resume_overrides(snapshot, patch):
    """Fold a validated safe-override patch into a per-run train_settings snapshot
    (JSON string, None, or _UNSET) for a cloud continue. Mirrors the local path,
    where update_train_settings persists the same keys — but a cloud run carries a
    frozen snapshot instead of the live dataset column, so we merge into a COPY and
    never touch the dataset. A None value drops the key (reset to default), matching
    update_train_settings' semantics. Returns a JSON string (or None if empty)."""
    if snapshot in (_UNSET, None):
        base = {}
    else:
        try:
            base = json.loads(snapshot)
        except (ValueError, TypeError):
            base = {}
        if not isinstance(base, dict):
            base = {}
    for k, v in patch.items():
        if v is None:
            base.pop(k, None)
        else:
            base[k] = v
    return json.dumps(base) if base else None


def continue_cloud_run(user_id, run_id, extra_steps=1000, from_step=None,
                       overrides=None) -> dict:
    """Reprend un run cloud TERMINAL (done OU en échec) depuis un checkpoint
    harvesté et vise step_de_reprise + extra_steps — le pendant cloud de
    lora_training.continue_training. C'est un VRAI launch_cloud_training (pod
    frais, mêmes garde-fous : limite de runs actifs, budget, unicité par
    famille) avec les paramètres persistés du run source (variante/famille/
    masked/GPU class, comme retry_cloud_run) ; son monitor, AVANT de démarrer le
    job, dépose le checkpoint dans le save_root du job sur le pod pour déclencher
    l'auto-resume d'ai-toolkit.

    ``from_step`` absent → dernier checkpoint (défaut). Fourni → CE step précis, y
    compris un checkpoint plus ancien : le seed d'un checkpoint arbitraire sur un
    pod NEUF est le même canal que le seed du dernier, et le staging du run source
    n'est jamais touché — repartir d'un step inférieur est donc gratuit côté cloud.
    ``overrides`` = mêmes réglages sûrs que le local (cadence/preview prompts),
    fusionnés dans le snapshot du run (jamais dans le dataset). register_launch
    reste un launch cloud normal — le resume est un détail d'exécution."""
    run = db.session.get(CloudTrainingRun, int(run_id))
    if not run:
        raise ValueError('unknown cloud run')
    # Continue from any TERMINAL run — a run that failed at pod teardown
    # ('pod did not become ready in time') can still have harvested, complete
    # checkpoints in its staging, and resuming from one is valid. Only a run
    # that is STILL RUNNING is blocked; the `no harvested checkpoint` check
    # below is the real gate for a terminal run whose staging was cleaned.
    if run.status in ACTIVE_STATES:
        raise ValueError('a run that is still running cannot be continued — '
                         'wait for it to finish or fail')
    try:
        p = json.loads(run.train_params or '{}')
    except ValueError:
        p = {}
    if not isinstance(p, dict):
        p = {}
    _assert_recipe_replayable(p, 'continue')
    override_patch = lt.validate_resume_overrides(overrides)
    cks = _run_staging_checkpoints(run)
    if not cks:
        raise ValueError('no harvested checkpoint to continue from — its staging '
                         'was cleaned; relaunch a fresh cloud run instead')
    # Which harvested checkpoint to resume from. Default = the latest; a specific
    # step restarts from an earlier epoch (seeding it onto the fresh pod is the same
    # channel, and the source run's staging is read-only here — nothing destroyed).
    if from_step is None:
        chosen = cks[-1]
    else:
        try:
            want = int(from_step)
        except (TypeError, ValueError):
            raise ValueError('from_step must be an integer step')
        matches = [c for c in cks if c['step'] == want]
        if not matches:
            avail = sorted({c['step'] for c in cks})
            raise ValueError(
                f'no harvested checkpoint at step {want} for this run (available: {avail})')
        # Prefer a suffixed save over the unsuffixed final when steps tie.
        chosen = min(matches, key=lambda c: not re.search(
            r'_(\d{6,})\.safetensors$', c['filename']))
    try:
        extra = max(100, int(extra_steps))
    except (TypeError, ValueError):
        extra = 1000
    snapshot = p.get(_TRAIN_SETTINGS_SNAPSHOT, _UNSET)
    # Resolve the LR factor against the SOURCE run's frozen settings (never the live
    # dataset): a 1e-4 run continues at 5e-5 / 1e-5. Refused loudly on a Prodigy run
    # before any launch. The resulting learning_rate merges into the per-run snapshot
    # exactly like cadence/timestep — the pod's _lr_eff then reads it.
    lr_factor = override_patch.pop('lr_factor', None)
    if lr_factor is not None:
        run_settings = {}
        if snapshot not in (_UNSET, None):
            try:
                parsed = json.loads(snapshot)
                run_settings = parsed if isinstance(parsed, dict) else {}
            except (ValueError, TypeError):
                run_settings = {}
        override_patch['learning_rate'] = lt.resolve_resume_lr(run_settings, lr_factor)
    if override_patch:
        snapshot = _merge_resume_overrides(snapshot, override_patch)
    # Lineage: the parent is the provenance record of the cloud run being
    # continued (source-of-truth edge for the Runs-hub tree). Legacy cloud runs
    # that predate the registry have no record -> the child is a root (NULL).
    from ..models import TrainingRunRecord
    _parent = (TrainingRunRecord.query
               .filter_by(cloud_run_id=run.id)
               .order_by(TrainingRunRecord.id.desc()).first())
    res = launch_cloud_training(
        user_id, run.dataset_id,
        steps=chosen['step'] + extra,
        base_model=p.get('base_model', ''),
        variant=p.get('variant'),
        train_type=p.get('train_type'),
        masked=p.get('masked', True),
        **_confirmation_flags(p),
        gpu_name=p.get('requested_gpu'),
        resume_ckpt_path=chosen['path'], resume_step=chosen['step'],
        train_settings_snapshot=snapshot,
        train_slider_snapshot=p.get(_TRAIN_SLIDER_SNAPSHOT, _UNSET),
        parent_record_id=(_parent.id if _parent else None),
        resumed_from=chosen['step'])
    res['resumed_from'] = chosen['step']
    res['target_steps'] = chosen['step'] + extra
    return res


def continue_local_run_in_cloud(user_id, dataset_id, extra_steps=1000,
                                from_step=None, overrides=None,
                                base_model=_UNSET, variant=None, train_type=None,
                                masked=True, allow_caption_mismatch=False,
                                allow_uncaptioned=False, allow_caption_quality=False,
                                allow_unverified_weights=False, allow_not_ready=False,
                                gpu_name=None) -> dict:
    """▶ Continue a LOCAL run's checkpoint IN THE CLOUD — the mirror of
    continue_cloud_run, and the other half of "pick your lane" in the ▶ Continue
    dialog. Nothing new is invented: the pod-side resume is the SAME seam
    (`resume_ckpt_path`), which launch_cloud_training's monitor drops into the
    job's save_root on a FRESH pod before start_job so ai-toolkit auto-resumes
    from it. The only difference with the cloud→cloud continue is where the file
    comes from: this one reads the ai-toolkit RUN DIR on disk instead of a cloud
    run's harvested staging.

    ``from_step`` absent → the newest local save. Provided → THAT step, including
    an earlier epoch: unlike the local lane (which archives the run aside and
    re-seeds it), seeding an arbitrary checkpoint onto a fresh pod touches
    NOTHING on disk — the local run dir is read-only here.

    Every guard of a normal cloud launch applies unchanged (vast.ai key, budget,
    active-run limit, per-family uniqueness, dataset export/captions): this IS a
    launch_cloud_training call, the resume is an execution detail. ``overrides``
    = the same safe subset as everywhere (cadence / preview prompts / timestep /
    lr_factor), merged into THIS run's settings snapshot — the dataset's own
    persisted settings are never touched (the local lane's update_train_settings
    is a local-lane behaviour, not something to replicate on a cloud launch)."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    fam = lt._train_type(ds, train_type)
    var = variant or getattr(ds, 'train_variant', None) or lt._default_variant_for(fam)
    # base_model _UNSET = the dataset's persisted base (the queue's behaviour);
    # an explicit value (the UI's checkpoint selection) targets THAT lane.
    lane = {} if base_model is _UNSET else {'base_model': base_model}
    base = (getattr(ds, 'train_base_model', None) or '') if base_model is _UNSET \
        else (base_model or '')
    # Validate the safe-subset overrides BEFORE anything else — a forbidden key
    # must fail with nothing launched (same contract as both other lanes).
    override_patch = lt.validate_resume_overrides(overrides)
    # The run dir lives under ai-toolkit's output: without it configured there is
    # no local save to send anywhere. Say that, rather than leaking the raw
    # 'ai-toolkit is not configured' from a lane the user asked to run in the CLOUD.
    try:
        cks = lt.list_checkpoints(user_id, dataset_id, family=fam, variant=var, **lane)
    except RuntimeError:
        raise ValueError('no local checkpoint to continue from — ai-toolkit is not '
                         'configured, so this machine has no local run folder')
    if not cks:
        raise ValueError('no local checkpoint to continue from for this base — '
                         'the run folder holds no save (a cloud run\'s epochs '
                         'live in its own staging: continue THAT run instead)')
    if from_step is None:
        chosen = cks[-1]
    else:
        try:
            want = int(from_step)
        except (TypeError, ValueError):
            raise ValueError('from_step must be an integer step')
        matches = [c for c in cks if c['step'] == want]
        if not matches:
            avail = sorted({c['step'] for c in cks})
            raise ValueError(
                f'no local checkpoint at step {want} for this run (available: {avail})')
        # Ties (a numbered save and the bare final at the same step): prefer the
        # numbered file — same rule as the local lane.
        chosen = min(matches, key=lambda c: bool(c.get('final')))
    # Resolve through the whitelisting helper (never os.path.join on a name from
    # the wire): it only returns a path that IS a save of this exact run.
    path = lt.checkpoint_file_path(user_id, dataset_id, chosen['filename'],
                                   family=fam, variant=var, **lane)
    if not path:
        raise ValueError(f"local checkpoint '{chosen['filename']}' is no longer on disk")
    try:
        extra = max(100, int(extra_steps))
    except (TypeError, ValueError):
        extra = 1000
    # LR factor → an absolute rate, resolved against the DATASET's live settings
    # (a local run trains from those, there is no per-run snapshot), and refused
    # loudly on a Prodigy run before any launch.
    lr_factor = override_patch.pop('lr_factor', None)
    if lr_factor is not None:
        override_patch['learning_rate'] = lt.resolve_resume_lr(lt._train_settings(ds), lr_factor)
    snapshot = _UNSET      # _UNSET → launch stamps the dataset's live settings
    if override_patch:
        snapshot = _merge_resume_overrides(getattr(ds, 'train_settings', None),
                                           override_patch)
    # Lineage: the parent is the newest record of THIS local lane, exactly like
    # lora_training.continue_training resolves it. Best-effort — a failure leaves
    # the edge NULL and never blocks the launch.
    from . import checkpoint_registry
    try:
        _parent = checkpoint_registry.newest_record_for(dataset_id, fam, base, var)
    except Exception:
        _parent = None
    res = launch_cloud_training(
        user_id, dataset_id,
        steps=chosen['step'] + extra,
        base_model=base, variant=var, train_type=fam, masked=masked,
        allow_caption_mismatch=allow_caption_mismatch,
        allow_uncaptioned=allow_uncaptioned,
        allow_caption_quality=allow_caption_quality,
        allow_unverified_weights=allow_unverified_weights,
        allow_not_ready=allow_not_ready,
        gpu_name=gpu_name,
        resume_ckpt_path=path, resume_step=chosen['step'],
        train_settings_snapshot=snapshot,
        parent_record_id=(_parent.id if _parent else None),
        resumed_from=chosen['step'])
    res['resumed_from'] = chosen['step']
    res['target_steps'] = chosen['step'] + extra
    return res


def launch_cloud_training(user_id, dataset_id, steps=None, base_model=_UNSET,
                          variant=None, train_type=None, masked=True,
                          allow_caption_mismatch=False, allow_uncaptioned=False,
                          allow_caption_quality=False,
                          allow_unverified_weights=False, allow_not_ready=False,
                          gpu_name=None, resume_ckpt_path=None, resume_step=None,
                          auto_retry_count=0, auto_retry_of=None,
                          strict_gpu=False, train_settings_snapshot=_UNSET,
                          train_slider_snapshot=_UNSET,
                          parent_record_id=None, resumed_from=None) -> dict:
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
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    # Slider LoRA mode (Beta) rides the SAME pod as every other family: the
    # pod's ai-toolkit registers `concept_slider` as a built-in trainer uid
    # (extends DiffusionTrainer — the very path _cloudify_job_config targets),
    # build_job_config already emits the full concept_slider process for the
    # cloud families, and the prompt-pair/substrate preflight below is family-
    # agnostic. The launch-time slider settings are frozen into the run params
    # (train_slider snapshot) exactly like train_settings, so a later toggle or
    # prompt edit cannot retarget an in-flight run.
    fam = fds.normalize_train_type(train_type or getattr(ds, 'train_type', None))
    # ``base_model`` is an explicit launch selection on the HTTP path.  Keep a
    # compatibility fallback for older internal callers that omitted it: use
    # the persisted value only while staying on the dataset's persisted family.
    # If the caller explicitly switches family, that old value belongs to the
    # previous family and must not make an official Krea/Klein launch fail.
    if base_model is _UNSET:
        persisted_fam = fds.normalize_train_type(getattr(ds, 'train_type', None))
        selected_base = (getattr(ds, 'train_base_model', None)
                         if not train_type or fam == persisted_fam else '')
    else:
        selected_base = base_model
    base_model = str(selected_base or '').strip()
    # Custom weights ride the cloud through a PRIVATE Hugging Face repo on the
    # user's account (one-time push, hf_base_push) — but only for the three
    # cloud families. Everything else keeps the historical refusal verbatim.
    from . import hf_base_push
    if base_model and fam not in hf_base_push.CLOUD_CUSTOM_BASE_FAMILIES:
        raise ValueError('custom weights are local-only — cloud training '
                         'uses the official Hugging Face bases')
    # These fields are local-only.  Unlike ``train_base_model`` they have no
    # supported cloud-family meaning (SDXL itself is rejected below), so retain
    # the historical fail-fast instead of silently accepting a selected override.
    if getattr(ds, 'train_vae_path', None) or getattr(ds, 'train_te_path', None):
        raise ValueError('custom VAE/text-encoder overrides are local-only — '
                         'cloud training uses the official Hugging Face bases')
    if fam == 'sdxl':
        raise ValueError('SDXL training needs a local base checkpoint — '
                         'cloud training supports Z-Image, Krea and FLUX.2 Klein')
    # flux2klein n'est PAS bloqué (contrairement à flux) : ses bases sont des repos
    # HF officiels que le pod télécharge lui-même — le 9B (32-48 GB VRAM) est même
    # la voie cloud principale de la famille.
    if fam == 'flux':
        raise ValueError('FLUX.1 training is local-only for now — '
                         'cloud training supports Z-Image, Krea and FLUX.2 Klein')
    variant = (variant or '').strip().lower()
    confirmations = {
        'allow_caption_mismatch': bool(allow_caption_mismatch),
        'allow_uncaptioned': bool(allow_uncaptioned),
        'allow_caption_quality': bool(allow_caption_quality),
        'allow_unverified_weights': bool(allow_unverified_weights),
        'allow_not_ready': bool(allow_not_ready),
    }
    recipe = None
    if fam == 'zimage':
        # Authoritative recipe validation happens before the reservation row and
        # therefore before a monitor can provision/rent a GPU.  build_job_config
        # validates again when the pod job is assembled. A custom base resolves
        # to the custom recipe (extras from the official Turbo pipeline), same
        # as the local path.
        recipe = lt.zimage_training_recipe(
            variant or lt._default_variant_for(fam), base_model=base_model or None)
        variant = recipe['variant']
    elif variant not in lt._valid_variants_for(fam):
        variant = lt._default_variant_for(fam)
    # Custom base: local-parity guardrails first (the confirmable arch sniff on
    # a still-present file; the distillation confirm for a custom Z-Image
    # declared Base/De-Turbo), then the pre-rent repo check — the pod downloads
    # the base from a PRIVATE repo on the user's HF account (hf_base_push), so
    # the launch fails HERE, with an actionable message, never after renting.
    base_repo = None
    if base_model:
        if fam == 'zimage':
            lt.assert_zimage_custom_recipe_confirmed(
                fam, base_model, variant, allow_unverified_weights)
        elif os.path.isfile(base_model):
            lt.preflight_custom_paths(
                fam, weights=base_model,
                allow_unverified_weights=allow_unverified_weights)
        base_repo = hf_base_push.require_base_repo(
            ds, fam, variant, base_model, cfg.secret('HF_TOKEN'))
    else:
        # OFFICIAL base: the pod downloads it from Hugging Face. Several are GATED
        # (Krea, FLUX, FLUX.2 Klein) and a gate the account never accepted answers
        # 403 — on the pod, after renting. Three runs were paid for and lost that
        # way, and the card only showed "403 Client Error (Request ID…)", hiding the
        # sentence that named the repo. One HEAD here costs nothing and turns that
        # into a message before a GPU is reserved.
        _assert_official_base_reachable(
            lt.official_base_repo(ds, fam, variant), cfg.secret('HF_TOKEN'))
    # Cheap fast-fail before the image/caption preflight below. This read is
    # intentionally advisory: another Flask request can reserve a slot after
    # it, so the same checks are repeated atomically at reservation time.
    _assert_launch_guardrails(dataset_id, fam)

    # Same caption-mismatch preflight as launch_training (MISMATCH_CAPTION
    # contract): assert_trainable is ALREADY a standalone helper in
    # lora_training.py (called from launch_training, not inlined there), so
    # no extraction was needed -- just match its real signature:
    # assert_trainable(dataset_id, train_type=None, allow_caption_mismatch=False).
    lt.assert_trainable(dataset_id, train_type=fam,
                        allow_caption_mismatch=allow_caption_mismatch,
                        allow_uncaptioned=allow_uncaptioned,
                        allow_caption_quality=allow_caption_quality,
                        allow_not_ready=allow_not_ready,
                        variant=variant)

    # The explicit launch base (''=official) rides into the run name so a
    # custom-base run keeps its own folder/prefix (combo-hash suffix, exactly
    # like local runs) and Base/De-Turbo cannot share Turbo's run path.
    run_name = lt._run_name(ds, base_model=base_model, family=fam, variant=variant)
    with _launch_reservation_lock:
        # Authoritative re-check + insert. Keeping the commit inside this
        # process-wide critical section means a second request always sees the
        # first request's preparing row before it can reserve or rent a pod.
        _assert_launch_guardrails(dataset_id, fam)
        run = CloudTrainingRun(
            dataset_id=dataset_id, status='preparing', run_name=run_name,
            # Stamp the family in the reservation itself. Without this, the
            # short window before the complete params are saved would make a
            # legitimate second-family launch look like an unknown-family run.
            train_params=json.dumps({'train_type': fam, 'variant': variant,
                                     'base_model': base_model, **confirmations}))
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
             job_name=f'lds{run.id}_{run_name}')
        # Mirror the LOCAL launch: persist this dataset's family/variant as its
        # remembered selection (launch_training does the same; two launch tests
        # assert it). This is now ONLY the dataset's default selection — the
        # monitor builds the pod job from the run's STAMPED params (see
        # _run_config_dataset at the build site), so a later launch overwriting
        # this row can no longer retarget an already-provisioning run's arch.
        ds.train_type = fam
        ds.train_variant = variant
        db.session.commit()
        # Same floor as the local path — a sub-500 target produces a run with
        # zero usable snapshots.
        n_steps = (max(500, int(steps)) if steps else lt.default_steps(
            ds, train_type=fam, variant=variant))
        # requested_gpu (from the launch-time speed picker) is a PREFERENCE, not
        # a lock: _provision re-searches live offers and rents the cheapest one
        # of this class, falling back to the cheapest overall if the class has
        # since sold out (vast offers are ephemeral).
        params = {'steps': n_steps, 'variant': variant, 'base_model': base_model,
                  'train_type': fam, 'masked': bool(masked), **confirmations}
        if base_repo:
            # The monitor's rebuild (and any retry/continue replay) must route
            # the pod's name_or_path to the PRIVATE repo without recomputing
            # anything: stamp the repo id and the remote weight size (drives
            # the pod's disk_gb sizing in _provision).
            params['base_repo_id'] = base_repo['repo_id']
            params['base_size_bytes'] = int(base_repo.get('size_bytes') or 0)
        if recipe:
            params.update({'recipe_version': recipe['recipe_version'],
                           'effective_base': recipe['effective_base'],
                           'training_adapter': recipe['training_adapter']})
        # Freeze the RAW JSON, not only the compact provenance summary: it also
        # carries custom preview prompts and explicit family defaults. ``None``
        # deliberately means "family defaults at launch".
        if train_settings_snapshot is _UNSET:
            train_settings_snapshot = getattr(ds, 'train_settings', None)
        params[_TRAIN_SETTINGS_SNAPSHOT] = train_settings_snapshot
        # Freeze the slider column the same way: a fresh launch snapshots the
        # dataset's current train_slider blob; retry/continue replay the source
        # run's snapshot (passed in) so a slider run stays a slider run even if
        # the dataset's mode was toggled off in between. ``None`` = not a slider
        # run at launch (build_job_config then emits the normal process).
        if train_slider_snapshot is _UNSET:
            train_slider_snapshot = getattr(ds, 'train_slider', None)
        params[_TRAIN_SLIDER_SNAPSHOT] = train_slider_snapshot
        if gpu_name:
            params['requested_gpu'] = str(gpu_name)
        if auto_retry_count:
            params['auto_retry_count'] = max(0, int(auto_retry_count))
        if auto_retry_of is not None:
            params['auto_retry_of'] = int(auto_retry_of)
        if strict_gpu:
            params['strict_gpu'] = True
        # Continue-in-cloud: the monitor seeds this checkpoint into the pod job's
        # save_root before start_job so ai-toolkit auto-resumes from it. Absent
        # on a normal launch (the seed step is then a no-op).
        if resume_ckpt_path:
            params['resume_ckpt_path'] = str(resume_ckpt_path)
            if resume_step is not None:
                params['resume_step'] = int(resume_step)
        # Provenance registry (same as local launches): dataset version at
        # launch time, stamped into the params so payloads can expose it.
        from . import checkpoint_registry
        rec = checkpoint_registry.register_launch(
            user_id, dataset_id, family=fam, source='cloud',
            variant=variant, masked=bool(masked), steps=n_steps,
            cloud_run_id=run.id,
            settings=lt.launch_settings_snapshot(
                _run_config_dataset(ds, params), fam),
            parent_record_id=parent_record_id, resumed_from=resumed_from)
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


_AUTO_RETRY_LIMIT = 1
_AUTO_RETRY_MARKERS = (
    'pod did not become ready',
    'pod unreachable',
    'connection aborted',
    'connection reset',
    'connectionreseterror',
    'remote end closed connection',
    'failed to establish a new connection',
    'max retries exceeded',
    'read timed out',
    'readtimeout',
    'connect timeout',
    'connecttimeout',
    'connection refused',
    'connectionrefusederror',
)


def _is_retryable_pod_failure(error) -> bool:
    """True only for transient pod/transport failures worth paying to retry."""
    text = str(error or '').lower()
    return any(marker in text for marker in _AUTO_RETRY_MARKERS)


_TRANSIENT_CREATE_CODES = ('400', '408', '409', '429', '500', '502', '503', '504')


def _is_transient_create_error(err) -> bool:
    """A vast create_instance refusal worth retrying with a FRESH offer: the
    offer was just taken (vast answers 400/409 — run #80's 'HTTP 400 {}'), a
    rate limit (429), or a vast-side hiccup (5xx). NOT an auth/quota rejection
    (401/403) or a genuinely-missing offer (404) that a retry cannot fix."""
    m = re.search(r'HTTP\s+(\d{3})', str(err or ''))
    return bool(m and m.group(1) in _TRANSIENT_CREATE_CODES)


def _auto_retry_child(parent_id):
    """Existing child, including the crash window before its id reached parent."""
    for child in CloudTrainingRun.query.order_by(CloudTrainingRun.id.desc()).all():
        if _run_param(child, 'auto_retry_of') == int(parent_id):
            return child
    return None


def _maybe_auto_retry(run, error):
    """Rent at most one fresh pod after a transient failure of an existing pod."""
    if (run.status != 'error' or not run.vast_instance_id
            or not _is_retryable_pod_failure(error)):
        return None

    with _auto_retry_lock:
        db.session.refresh(run)
        try:
            params = json.loads(run.train_params or '{}')
        except (TypeError, ValueError):
            return None
        if not isinstance(params, dict):
            return None
        replay_diag = _recipe_replay_diagnostic(params)
        if replay_diag and replay_diag.get('status') in (
                'legacy_incompatible', 'incompatible'):
            logger.warning('automatic retry blocked for unsafe legacy recipe on run %s',
                           run.id)
            return None
        try:
            retry_count = max(0, int(params.get('auto_retry_count') or 0))
        except (TypeError, ValueError):
            return None

        existing = _auto_retry_child(run.id)
        if existing is not None:
            params['auto_retry_scheduled'] = True
            params['auto_retry_pending'] = False
            params['auto_retry_run_id'] = existing.id
            _set(run, train_params=json.dumps(params),
                 phase_detail='Run failed — automatic retry launched')
            return {'run_id': existing.id, 'status': existing.status}

        pending_recovery = bool(params.get('auto_retry_scheduled')
                                and params.get('auto_retry_pending'))
        if retry_count >= _AUTO_RETRY_LIMIT:
            return None
        if params.get('auto_retry_scheduled') and not pending_recovery:
            return None

        # Commit the claim before renting. boot_recover resumes this exact
        # pending state if the app stops between the claim and child creation.
        params['auto_retry_scheduled'] = True
        params['auto_retry_pending'] = True
        _set(run, train_params=json.dumps(params),
             phase_detail='Run failed — automatic retry starting…')

        # Reuse the GPU class actually rented. requested_gpu may have fallen
        # back on the initial launch, so it is not necessarily the effective GPU.
        gpu_name = run.gpu_name or params.get('requested_gpu')
        try:
            result = launch_cloud_training(
                'local', run.dataset_id,
                steps=params.get('steps'),
                base_model=params.get('base_model', ''),
                variant=params.get('variant'),
                train_type=params.get('train_type'),
                masked=params.get('masked', True),
                **_confirmation_flags(params),
                gpu_name=gpu_name,
                resume_ckpt_path=params.get('resume_ckpt_path'),
                resume_step=params.get('resume_step'),
                auto_retry_count=retry_count + 1,
                auto_retry_of=run.id,
                strict_gpu=bool(gpu_name),
                train_settings_snapshot=params.get(
                    _TRAIN_SETTINGS_SNAPSHOT, _UNSET))
        except Exception as retry_error:
            params['auto_retry_pending'] = False
            params['auto_retry_error'] = str(retry_error)[:300]
            prior = str(run.error or error or '')
            _set(run, train_params=json.dumps(params),
                 phase_detail='Run failed — automatic retry could not start',
                 error=f'{prior} | automatic retry: {retry_error}'[:1000])
            logger.exception('automatic retry for cloud run %s could not start',
                             run.id)
            return None

        params['auto_retry_pending'] = False
        params['auto_retry_run_id'] = result.get('run_id')
        _set(run, train_params=json.dumps(params),
             phase_detail='Run failed — automatic retry launched')
        logger.warning('cloud run %s automatically retried as run %s on %s',
                       run.id, result.get('run_id'), gpu_name)
        return result


def _recover_pending_auto_retries():
    """Complete the persisted claim-to-child crash window at app boot."""
    parents = CloudTrainingRun.query.filter_by(status='error').all()
    for parent in parents:
        if _run_param(parent, 'auto_retry_pending'):
            _maybe_auto_retry(parent, parent.error)


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


def _pick_offer(offers, requested_gpu, strict=False):
    """Best offer of the requested GPU class if the user picked a speed tier
    and that class is still on the market; otherwise an offer of a
    SIMILAR-OR-BETTER speed tier (≥75% of the requested class's throughput,
    per gpu_speed). 'Best' = most reliable within +10% of the cheapest (see
    _best_of), on offers already stripped of blacklisted hosts and bait
    prices by _filter_offers.

    The historical fallback — cheapest offer of ANY class — handed a $0.13/h
    RTX 3090 to a 12B Krea run when the requested RTX PRO 6000 S sold out
    between the picker and the launch (retry path, user-reported): the
    bottom-barrel is exactly where the flaky hosts live, and the run would
    have been ~3x slower. No similar tier on the market -> actionable error,
    never a silent downgrade."""
    if requested_gpu:
        matches = [o for o in offers if (o.get('gpu_name') or '') == requested_gpu]
        if matches:
            return _best_of(matches)
        if strict:
            raise RuntimeError(
                f'no {requested_gpu} offer is available for the automatic retry')
        floor = gpu_speed.speed_factor(requested_gpu) * 0.75
        similar = [o for o in offers
                   if gpu_speed.speed_factor(o.get('gpu_name')) >= floor]
        if similar:
            return _best_of(similar)
        raise RuntimeError(
            f'no offers similar to {requested_gpu} right now — open the GPU '
            'picker and choose another speed tier')
    return _best_of(offers)


def _disk_gb_for(cloud_cfg, params) -> int:
    """Pod disk size: the configured default, bumped when the run trains on a
    LARGE custom base (stamped remote size). The pod holds the raw download
    plus its quantized working copy plus dataset/checkpoints/HF cache, so the
    bump budgets twice the base size + 30 GB of headroom. Official-base runs
    (no stamp) keep the configured value bit-for-bit."""
    disk_gb = int(cloud_cfg.get('disk_gb') or 60)
    try:
        base_bytes = int(params.get('base_size_bytes') or 0)
    except (TypeError, ValueError):
        base_bytes = 0
    if base_bytes:
        needed = int(base_bytes / 1e9 * 2) + 30
        if needed > disk_gb:
            logger.info('custom base is %.1f GB — pod disk bumped %s -> %s GB',
                        base_bytes / 1e9, disk_gb, needed)
            disk_gb = needed
    return disk_gb


def _provision(run):
    """Search offers and create the instance, honoring the launch-time GPU
    choice when the picked class is still available.
    LEAK-SAFE: any failure after create_instance destroys the instance."""
    c = cfg.get('cloud') or {}
    params = json.loads(run.train_params or '{}')
    fam = params.get('train_type') or 'zimage'
    min_vram = (c.get('min_vram_gb') or {}).get(fam, 24)
    disk_gb = _disk_gb_for(c, params)
    template_hash = (c.get('template_hash') or '').strip()
    # A transient create refusal (offer just taken -> HTTP 400/409, rate limit,
    # vast 5xx — run #80's 'HTTP 400 {}' died here with no retry) gets a bounded
    # re-search: the failed offer is likely gone, so each attempt excludes the
    # offers already tried and picks a fresh one. A non-transient refusal (auth,
    # 404) or an exhausted budget raises immediately.
    tried_offers = set()
    offer = instance_id = None
    token = ''
    for attempt in range(1, _CREATE_INSTANCE_ATTEMPTS + 1):
        offers = vast_client.search_offers(
            min_vram_gb=min_vram, max_dph=c.get('max_price_per_hour', 0.80),
            min_inet_down_mbps=int(c.get('min_inet_down_mbps') or 0),
            min_reliability=float(c.get('min_reliability') or 0.98),
            min_disk_bw_mbps=int(c.get('min_disk_bw_mbps') or 0),
            verified_only=bool(c.get('verified_only', True)),
            secure_cloud_only=bool(c.get('secure_cloud_only', False)))
        pool = [o for o in offers if o.get('offer_id') not in tried_offers]
        if not pool:
            if tried_offers:
                raise RuntimeError(
                    'no fresh vast.ai offer left after a transient create refusal '
                    f'(tried {len(tried_offers)})')
            raise RuntimeError(
                f'no vast.ai offer matches (>= {min_vram} GB VRAM, '
                f'<= ${c.get("max_price_per_hour", 0.80)}/h) — raise the price cap in Settings')
        offer = _pick_offer(_filter_offers(pool), params.get('requested_gpu'),
                            strict=bool(params.get('strict_gpu')))
        tried_offers.add(offer['offer_id'])
        # Stamp the host identity so a boot failure can blacklist THIS machine.
        if offer.get('machine_id') is not None:
            params['machine_id'] = offer['machine_id']
            _set(run, train_params=json.dumps(params))
        try:
            if template_hash:
                # Preferred path (smoke-validated 2026-07-12): the official
                # template publishes the UI behind the pod's Caddy proxy on
                # ui_port and vast generates the per-instance auth token (picked
                # up from the instance record during boot-wait). HF_TOKEN reaches
                # the pod later via ensure_settings(), not env.
                token = ''
                instance_id = vast_client.create_instance(
                    offer['offer_id'], disk_gb=disk_gb,
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
                    offer['offer_id'], disk_gb=disk_gb,
                    label=run.vast_label, image=c.get('image'), env=env,
                    onstart=(c.get('onstart') or None))
            break
        except vast_client.VastError as e:
            if attempt >= _CREATE_INSTANCE_ATTEMPTS or not _is_transient_create_error(e):
                raise
            logger.warning('create_instance attempt %s/%s failed (%s) — retrying '
                           'with a fresh offer', attempt, _CREATE_INSTANCE_ATTEMPTS, e)
            _sleep(_CREATE_INSTANCE_BACKOFF)
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
            _recover_pending_auto_retries()
    except Exception:
        logger.exception('cloud boot recovery failed')


POLL_SECONDS = 10
_CKPT_SYNC_EVERY_POLLS = 12          # mid-run checkpoint mirror every ~2 min
READY_TIMEOUT_SECONDS = 900          # 15 min: boot + image pull
UNREACHABLE_GRACE_SECONDS = 360      # default tolerated mid-run network blackout
                                     # (overridable: cloud.unreachable_grace_minutes)
_CREATE_INSTANCE_ATTEMPTS = 3        # bounded retry on a transient vast create refusal
_CREATE_INSTANCE_BACKOFF = 5         # seconds between create attempts
_sleep = time.sleep


def _now():
    return time.time()


def _make_remote(run) -> RemoteAiToolkit:
    return RemoteAiToolkit(run.base_url, run.auth_token)


def _cloudify_job_config(job_config: dict, job_name: str,
                         staging_dataset: str, pod_settings: dict,
                         run_params: dict | None = None) -> dict:
    """Rewrite the locally-built config for the pod: remote paths, remote
    trainer type (DB status updates), and the job name the pod's routes key
    on. The staging->pod path swap is done on the JSON text so every field
    referencing the staging dir (folder_path, mask_path) is rewritten at
    once, backslash-escaping included.

    Custom base (run_params carries base_repo_id): the symmetric seam to the
    dataset swap — build_job_config emitted the LOCAL custom path (single file
    or converted diffusers dir), which means nothing on the pod; route
    model.name_or_path to the PRIVATE HF repo the base was pushed to, and the
    pod downloads it with the user's HF_TOKEN exactly like the gated official
    bases. Krea additionally pins model_kwargs.checkpoint_filename (its loader
    fetches ONE file from the repo); Klein derives its hardcoded per-size
    filename from the arch; Z-Image loads the repo's transformer/ subfolder."""
    pod_ds = pod_settings['DATASETS_FOLDER'].rstrip('/') + '/' + job_name
    text = json.dumps(job_config)
    needle = json.dumps(str(staging_dataset))[1:-1]     # JSON-escaped form
    text = text.replace(needle, pod_ds)
    out = json.loads(text)
    conf = out['config']
    conf['name'] = job_name
    proc = conf['process'][0]
    # The local build emits the legacy 'sd_trainer' uid; the pod's ai-toolkit
    # runs the modern 'diffusion_trainer' path (the universal ui/api trainer
    # whose progress events the pod UI's DB understands), so retype it here. A
    # slider run already emits 'concept_slider' — a first-class built-in
    # extension that itself extends DiffusionTrainer — which the pod runs as-is;
    # flattening it to diffusion_trainer would silently drop the slider loss and
    # train an ordinary LoRA. Only the standard trainer is retyped.
    if proc.get('type') == 'sd_trainer':
        proc['type'] = 'diffusion_trainer'
    proc['training_folder'] = pod_settings['TRAINING_FOLDER']
    proc['device'] = 'cuda:0'
    # Dual captions are LOCAL-ONLY for now: remote.upload_dataset only ships the images
    # and their .txt sidecars (its extension filter skips the JSON caption file), so a
    # pod folder_path pointing at that missing JSON would find zero images. Revert to the
    # historical folder + .txt sidecars — the run trains with long captions only. The
    # earlier blanket path-swap already mangled the JSON folder_path; overwrite it cleanly.
    train = proc.get('train') or {}
    if train.pop('short_and_long_captions', None):
        datasets = proc.get('datasets') or []
        if datasets:
            datasets[0]['folder_path'] = pod_ds
            datasets[0].setdefault('caption_ext', 'txt')
    base_repo = (run_params or {}).get('base_repo_id')
    if base_repo:
        from . import hf_base_push
        fam = (run_params or {}).get('train_type')
        model = proc.get('model') or {}
        model['name_or_path'] = base_repo
        fname = hf_base_push.weight_filename(
            fam, (run_params or {}).get('variant'), base_repo.split('/')[-1])
        if fam == 'krea' and fname:
            kwargs = dict(model.get('model_kwargs') or {})
            kwargs['checkpoint_filename'] = fname
            model['model_kwargs'] = kwargs
        proc['model'] = model
    return out


def _finish(run, status, detail='', error=None, destroy=True):
    # A paid retry must never overlap the failed pod. Return whether there is
    # confirmed to be no old pod left; callers that do not retry ignore it.
    pod_gone = not bool(run.vast_instance_id)
    if destroy and run.vast_instance_id:
        try:
            pod_gone = bool(vast_client.destroy_instance(run.vast_instance_id))
            if not pod_gone:
                logger.error('terminate %s returned false', run.vast_instance_id)
        except Exception as e:
            pod_gone = False
            logger.warning('terminate %s failed: %s', run.vast_instance_id, e)
    _set(run, status=status, phase_detail=detail, error=error,
         finished_at=datetime.utcnow())
    return pod_gone


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
        # thread's start.
        run_age = max(0.0, (datetime.utcnow() - (run.created_at or datetime.utcnow())).total_seconds())
        cap_anchor = _now() - run_age
        # Whether we ENTER the monitor already owning a pod (app restarted while
        # it was still booting) — captured BEFORE _provision, which sets
        # vast_instance_id on a fresh launch. It decides the boot-readiness
        # anchor below.
        resuming_existing_pod = bool(run.vast_instance_id)
        try:
            # -- heavy launch work, moved off the HTTP path (see launch) ----
            _prepare_staging(run)
            # -- provision (if resuming, the instance may already exist) ----
            if not run.vast_instance_id:
                _provision(run)
            # Boot-readiness timeout anchor. A FRESH launch measures from now
            # (post-provision) so dataset staging / offer search never eat into
            # the pod's boot budget. A RESUME must NOT get a brand-new window on
            # every restart: that let a pod whose UI never answered survive
            # 37 min across two restarts instead of the 15-min READY_TIMEOUT
            # (incident 2026-07-14). On resume we anchor to the DURABLE
            # created_at (cap_anchor), so readiness measures the TOTAL time since
            # launch across every restart — the intended behaviour even for a pod
            # that was honestly still booting.
            boot_started = cap_anchor if resuming_existing_pod else _now()

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
                # Build from the run's STAMPED family/variant, NEVER the dataset's
                # current train_type/train_variant: a later launch on the same
                # dataset (or a /train-type change) may have moved that row since
                # this run launched, and this rebuild happens minutes later at pod
                # boot. _run_config_dataset presents the run's own launch params
                # so two concurrent multi-family runs each get their own arch
                # (incident 2026-07-14 — see _RunConfigDataset).
                params = json.loads(run.train_params or '{}')
                ds = fds.get_dataset('local', run.dataset_id)
                job_config = lt.build_job_config(
                    _run_config_dataset(ds, params),
                    staging_dataset, steps=params.get('steps') or 3000,
                    training_folder='__POD__')
                job_config = _cloudify_job_config(job_config, run.job_name,
                                                  staging_dataset, pod_settings,
                                                  run_params=params)
                job_id = remote.create_job(run.job_name, job_config)
                # Continue-in-cloud: drop the source checkpoint into the job's
                # save_root BEFORE start so ai-toolkit auto-resumes from it.
                _seed_resume_checkpoint(run, remote, pod_settings)
                remote.start_job(job_id)
                _set(run, remote_job_id=job_id, status='training',
                     phase_detail='Job queued on the pod')
            else:
                job_id = run.remote_job_id
                _set(run, phase_detail='Resuming — reattaching to running job')

            # -- poll until terminal ------------------------------------------
            # Two watchdogs share one progress clock (last_progress_ts):
            #  * stall — once training has produced a step, kill if the step
            #    counter freezes past stall_timeout_minutes.
            #  * first-step — BEFORE the first step (base download, quantize,
            #    latent caching) kill if step 1 is never reached in time. Only
            #    the runtime cap used to bound this phase, so a pod whose base
            #    download collapsed to a crawl burned the WHOLE cap for zero
            #    steps (run #75: 26.3 GB base at ~12 kB/s, 10h45 / 7 € / 0 saves
            #    — 2026-07-19). A healthy Krea-2-Raw run reaches step 1 in a few
            #    minutes (its full 2000-step run was ~84 min), so the default is
            #    generous enough to survive an honestly slow download.
            stall_seconds = int(c.get('stall_timeout_minutes') or 30) * 60
            first_step_seconds = int(c.get('first_step_timeout_minutes') or 45) * 60
            grace_seconds = (int(c.get('unreachable_grace_minutes') or 0) * 60
                             or UNREACHABLE_GRACE_SECONDS)
            last_step = -1
            last_progress_ts = _now()
            # Time of the FIRST failure of the current unreachable streak (None
            # while the pod answers). The grace must measure CONSECUTIVE get_job
            # failure time, not time-since-last-success: the per-poll log/sample
            # mirror and checkpoint sync can each block for tens of seconds on a
            # degrading vast proxy, and anchoring to the last success would let
            # that non-probe time silently eat the grace and declare a still-live
            # pod 'unreachable' on its very first failed probe.
            unreachable_since = None
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
                    unreachable_since = None
                except Exception as e:
                    now = _now()
                    if unreachable_since is None:
                        unreachable_since = now
                    if now - unreachable_since > grace_seconds:
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
                    _download_intermediates(run, remote)
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
                elif last_step <= 0 and (_now() - last_progress_ts) > first_step_seconds:
                    # No training step in first_step_timeout_minutes — the pod is
                    # wedged before step 1 (typically the base-model download
                    # crawling; nothing to rescue since no checkpoint exists).
                    try:
                        remote.stop_job(job_id)
                    except Exception:
                        pass
                    _finish(run, 'error',
                            detail='No training step reached in '
                                   f'{first_step_seconds // 60} min — pod likely '
                                   'stuck downloading the base model; terminated',
                            error='first-step watchdog')
                    return
                _sleep(POLL_SECONDS)
        except Exception as e:
            logger.exception('cloud run %s failed', run_id)
            error_text = str(e)[:500]
            retryable = _is_retryable_pod_failure(error_text)
            # Exclude the failed host before selecting the fresh pod.
            if retryable:
                _blacklist_host(_run_machine_id(run),
                                f'transient pod failure: {error_text[:160]}')
            pod_gone = _finish(run, 'error', detail='Run failed',
                               error=error_text)
            if retryable and pod_gone:
                _maybe_auto_retry(run, error_text)
            elif retryable:
                _set(run, phase_detail='Run failed — automatic retry withheld '
                                       'because pod termination was not confirmed')
        finally:
            # This run's slot in the module maps is done with — drop it so
            # they cannot grow unbounded across the app's lifetime with many
            # concurrent runs coming and going.
            _stop_events.pop(int(run_id), None)
            _monitor_threads.pop(int(run_id), None)
            _sync_state.pop(int(run_id), None)


def _seed_resume_checkpoint(run, remote, pod_settings):
    """Continue-in-cloud: place the source run's harvested checkpoint into THIS
    job's save_root on the pod so ai-toolkit's auto-resume finds it — it globs
    <TRAINING_FOLDER>/<job_name>/<job_name>*.safetensors, takes the newest by
    ctime, and reads the resume step from the safetensors metadata. The file is
    renamed to THIS job's prefix so the glob matches (the save the trainer would
    itself write). No resume checkpoint stamped in train_params -> no-op (a
    normal launch). A missing/failed seed RAISES: a 'continue' that cannot
    resume must fail loudly, never silently train from scratch."""
    src = _run_param(run, 'resume_ckpt_path')
    if not src:
        return
    if not os.path.isfile(src):
        raise RuntimeError(f'resume checkpoint vanished before upload: {src}')
    step = int(_run_param(run, 'resume_step') or 0)
    remote_name = f'{run.job_name}_{step:09d}.safetensors'
    training_folder = pod_settings['TRAINING_FOLDER'].rstrip('/')
    dest_dir = f'{training_folder}/{run.job_name}'
    _set(run, phase_detail='Seeding checkpoint for resume…')
    remote.seed_checkpoint(pod_settings['DATASETS_FOLDER'], dest_dir,
                           remote_name, src)
    logger.info('run %s: seeded resume checkpoint %s -> %s',
                run.id, os.path.basename(src), dest_dir)


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
    2026-07-13). Never raises, never flips the run's status. EVERY synced save
    is KEPT (user ask: harvest ALL trained epochs) — the pod prunes its own
    saves to max_step_saves, so grabbing each one as it appears is the only
    way to collect the full epoch history; disk is reclaimed via the 🗑/🧹
    tools and the trash.

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
            # checkpoint_local_path tracks the NEWEST save; earlier synced
            # saves stay on disk (full epoch harvest).
            _set(run, checkpoint_local_path=dest)
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
                             base_model=params.get('base_model', ''),
                             family=params.get('train_type'),
                             src_dir=run.staging_dir,
                             version=params.get('version'),
                             variant=params.get('variant'),
                             run_id=run.id, run_source='cloud')
    except Exception as e:
        logger.warning('cloud import into ComfyUI failed: %s', e)


def _download_intermediates(run, remote):
    """After the FINAL checkpoint landed (strict path), also pull the pod's
    remaining intermediate saves — WITHOUT them a cloud run offered only its
    last epoch while a local run offers max_step_saves of them to pick the
    least-overfit one (user-observed parity gap, 2026-07-13). Best-effort per
    file: a failed intermediate never degrades the run's outcome."""
    try:
        files = [f for f in remote.list_files(run.remote_job_id)
                 if f.get('path', '').endswith('.safetensors')]
    except Exception as e:
        logger.warning('intermediate listing failed: %s', e)
        return
    have = os.path.basename(run.checkpoint_local_path or '')
    for f in files:
        name = os.path.basename(f['path'].replace('\\', '/'))
        if name == have:
            continue
        dest = os.path.join(run.staging_dir, name)
        want = int(f.get('size') or 0)
        try:
            if os.path.isfile(dest) and (not want or os.path.getsize(dest) == want):
                continue
            remote.download_public_file(f['path'], dest,
                                        expected_size=want or None, attempts=50)
        except Exception as e:
            logger.warning('intermediate %s not retrieved: %s', name, e)


def _mirror_into_local_run(run):
    """Copy the downloaded cloud checkpoints (final + retrieved intermediates)
    into the LOCAL ai-toolkit run dir, renamed to the local convention
    (`lora_<trigger>[_<step>].safetensors`), so cloud results behave exactly
    like local ones everywhere downstream: the panel's checkpoint list, the
    Resume-or-Fresh prompt, Continue training. No-op when ai-toolkit isn't
    configured locally; best-effort, never fails the run."""
    try:
        if not run.staging_dir or not os.path.isdir(run.staging_dir):
            return
        params = json.loads(run.train_params or '{}')
        # Mirror into THIS run's local dir: the stamped base ('' = official,
        # else the custom selection whose combo hash isolates the folder).
        run_dir = lt._run_dir('local', run.dataset_id,
                              base_model=params.get('base_model', ''),
                              family=params.get('train_type'),
                              variant=params.get('variant'))
        os.makedirs(run_dir, exist_ok=True)
        base = os.path.basename(os.path.normpath(run_dir))     # lora_<trigger>
        for src_name in sorted(os.listdir(run.staging_dir)):
            if not src_name.lower().endswith('.safetensors'):
                continue
            _mirror_one(run, run_dir, base, src_name)
    except Exception as e:
        # RuntimeError from _run_dir = ai-toolkit not configured -> fine
        logger.debug('local run-dir mirror skipped: %s', e)


def _mirror_one(run, run_dir, base, src_name):
    try:
        m = re.search(r'_(\d{6,})\.safetensors$', src_name)
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
        shutil.copy2(os.path.join(run.staging_dir, src_name), dest)
        logger.info('mirrored cloud checkpoint into local run dir: %s/%s',
                    run_dir, dest_name)
    except (OSError, re.error) as e:
        logger.warning('mirror of %s skipped: %s', src_name, e)


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


def _staging_save_count(run) -> int:
    """How many checkpoints (.safetensors) this run's staging currently holds —
    the 'saves' figure on the Runs-hub cards. 0 when staging is gone (purged /
    hand-deleted): the card simply drops the metric, never crashes."""
    if not run.staging_dir:
        return 0
    try:
        return sum(1 for f in os.listdir(run.staging_dir)
                   if f.lower().endswith('.safetensors'))
    except OSError:
        return 0


def _latest_sample_name(samples_dir):
    """Filename of the NEWEST sample image in a run's samples dir (highest
    step, then prompt index) or None. ai-toolkit names samples
    <ts>__<step>_<promptidx>.<ext> — lt._SAMPLE_RE parses that."""
    try:
        names = os.listdir(samples_dir)
    except OSError:
        return None
    best = None
    for f in names:
        m = lt._SAMPLE_RE.search(f)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)))
        if best is None or key > best[0]:
            best = (key, f)
    return best[1] if best else None


def _run_samples_dir(crun, rec):
    """Samples dir of a run — cloud: its staging download; local: the
    ai-toolkit run dir stamped on its registry row. None when unresolvable
    (purged staging, ai-toolkit not configured, deleted dataset)."""
    if crun is not None and crun.staging_dir:
        return os.path.join(crun.staging_dir, 'samples')
    if rec is not None and rec.source == 'local':
        try:
            return os.path.join(
                lt._run_dir(cfg.LOCAL_USER, rec.dataset_id,
                            base_model=rec.base_model or '',
                            family=rec.family, variant=rec.variant),
                'samples')
        except Exception:
            return None
    return None


def run_preview_path(run_key):
    """Absolute path of a run's newest sample image, or None — the Runs-hub
    card thumbnail. `run_key` is the run's share key ('cloud-<id>'/'rec-<id>',
    same addressing as the Share-config download). Resolved entirely
    server-side from the run's own rows: the client never sends a path."""
    from .run_share import resolve_run
    crun, rec = resolve_run(run_key)
    if crun is None and rec is None:
        return None
    d = _run_samples_dir(crun, rec)
    if not d:
        return None
    name = _latest_sample_name(d)
    if not name:
        return None
    p = os.path.join(d, name)
    return p if os.path.isfile(p) else None


def _annotate_preview(row, crun, rec):
    """Stamp `preview_url` on a Runs-hub history row when the run left at
    least one sample on disk — the frontend shows it as the card thumbnail
    (and falls back to a family tile when absent)."""
    d = _run_samples_dir(crun, rec)
    if d and _latest_sample_name(d):
        row['preview_url'] = f"/api/dataset/train/runs/{row['share_key']}/preview"


def _run_payload(run) -> dict:
    family = _run_family(run)
    variant = _run_param(run, 'variant')
    effective_base = _run_param(run, 'effective_base')
    training_adapter = _run_param(run, 'training_adapter')
    recipe_version = _run_param(run, 'recipe_version')
    # The base a run trained on ('' = official family base, else a custom
    # checkpoint/repo) — so the Runs-hub card can name it. Only merged when the
    # pod actually stamped it: absent on a very old pod stays out of the payload
    # (the card degrades to the family badge, never a wrong "official" claim),
    # and a registry-backed row keeps its own base_model through _run_payload's
    # enrichment update.
    base_model = _run_param(run, 'base_model')
    diagnostic = lt.zimage_recipe_diagnostic(
        family, variant, effective_base, training_adapter, recipe_version)
    payload = {'run_id': run.id, 'dataset_id': run.dataset_id, 'status': run.status,
            # Stable id for the per-run "Share configuration" download. Every
            # cloud row (active/finished/legacy) addresses by its pod row id;
            # local rows use 'rec-<record id>' (set in all_runs).
            'share_key': f'cloud-{run.id}',
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
            # card metrics: target steps (stamped launch param) + how many
            # checkpoints the pod saved (live count of the staging downloads)
            'steps': _run_param(run, 'steps'),
            'saves': _staging_save_count(run),
            # Distinct steps of the harvested checkpoints still in staging — the
            # ▶ Continue dialog offers them so a finished run can resume from an
            # EARLIER epoch, not only its last (empty when staging was purged).
            'resume_steps': (sorted({c['step'] for c in _run_staging_checkpoints(run)})
                             if run.status == 'done' else []),
            'train_type': family, 'variant': variant,
            'effective_base': effective_base,
            'training_adapter': training_adapter,
            'recipe_version': recipe_version,
            'recipe_status': diagnostic and diagnostic.get('status'),
            'recipe_warning': diagnostic and diagnostic.get('warning'),
            'version': _run_param(run, 'version'),
            'auto_retry_count': int(_run_param(run, 'auto_retry_count') or 0),
            'auto_retry_of': _run_param(run, 'auto_retry_of'),
            'auto_retry_run_id': _run_param(run, 'auto_retry_run_id'),
            'created_at': run.created_at.isoformat() if run.created_at else None,
            'finished_at': run.finished_at.isoformat() if run.finished_at else None}
    if base_model is not None:
        payload['base_model'] = base_model
    return payload


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
    """Everything the unified Runs hub needs in one call: the active cloud
    runs (manage/watch), the LIVE local training if any, and a history of
    EVERY launch — local AND cloud — from the provenance registry (each row
    carries the settings snapshot the launch actually sent to ai-toolkit).
    Cloud rows are enriched from their CloudTrainingRun (status/cost/
    checkpoint); cloud runs that predate the registry still appear via a
    fallback union, so history never shrinks."""
    from ..models import TrainingRunRecord
    actives = get_active_runs()
    c = cfg.get('cloud') or {}
    limit = max(1, min(int(limit or 20), 100))
    recs = (TrainingRunRecord.query
            .order_by(TrainingRunRecord.id.desc()).limit(limit).all())
    cloud_ids = {r.cloud_run_id for r in recs if r.cloud_run_id}
    cloud_by_id = ({r.id: r for r in CloudTrainingRun.query
                    .filter(CloudTrainingRun.id.in_(cloud_ids)).all()}
                   if cloud_ids else {})
    # Local runs have no status column — the failed one (at most a single row,
    # local training is single-flight) is derived from the transient crash state
    # so its row can carry status='error' + a ↻ Retry affordance.
    _failed_local = lt.failed_local_run()
    failed_local_id, failed_local_msg = _failed_local or (None, None)
    recent = []
    for rec in recs:
        crun = cloud_by_id.get(rec.cloud_run_id)
        if crun is not None and crun.status in ACTIVE_STATES:
            continue                      # already shown in the actives section
        try:
            settings = json.loads(rec.settings) if rec.settings else None
        except ValueError:
            settings = None
        row = {'source': 'cloud' if rec.source == 'cloud' else 'local',
               'dataset_id': rec.dataset_id,
               'dataset_name': _dataset_name(rec.dataset_id),
               'train_type': rec.family, 'version': rec.version,
               'steps': rec.steps, 'masked': bool(rec.masked),
               'variant': rec.variant, 'base_model': rec.base_model or '',
               'settings': settings,
               # Lineage edge (genealogy tree): the record this launch resumed
               # from, NULL on a fresh run / root. `lineage` (below) then flags
               # rows that open into a ≥2-node tree.
               'parent_record_id': rec.parent_record_id,
               'resumed_from': rec.resumed_from,
               # Stable local-run identity for the 💻 #N chip + Checkpoints
               # deep-link. Cloud rows show ☁ #<cloud run id> (run_id, below).
               'record_id': rec.id,
               # local rows live only in the registry -> addressed by record id;
               # a cloud row overrides this with 'cloud-<id>' via _run_payload.
               'share_key': f'rec-{rec.id}',
               # Stable retry target for a LOCAL run (cloud rows retry by run_id).
               'record_id': rec.id,
               'created_at': rec.created_at.isoformat() if rec.created_at else None}
        if rec.source == 'local' and rec.id == failed_local_id:
            row['status'] = 'error'
            row['error'] = failed_local_msg
        if rec.family == 'zimage':
            safe_settings = settings if isinstance(settings, dict) else {}
            diag = lt.zimage_recipe_diagnostic(
                rec.family, rec.variant,
                safe_settings.get('effective_base'),
                safe_settings.get('training_adapter'),
                safe_settings.get('recipe_version'))
            row.update({'effective_base': safe_settings.get('effective_base'),
                        'training_adapter': safe_settings.get('training_adapter'),
                        'recipe_version': safe_settings.get('recipe_version'),
                        'recipe_status': diag and diag.get('status'),
                        'recipe_warning': diag and diag.get('warning')})
        if crun is not None:
            # cloud enrichment wins on shared keys (status/cost/checkpoint/...)
            # — except steps, where the registry row must survive a pod row
            # whose train_params never stamped them (payload steps = None).
            registry_steps = row.get('steps')
            row.update(_run_payload(crun))
            if row.get('steps') is None:
                row['steps'] = registry_steps
            row['settings'] = settings
            row['source'] = 'cloud'
        _annotate_preview(row, crun, rec)
        recent.append(row)
    # Legacy cloud runs that predate the provenance registry (no record row).
    seen_cloud = {r.get('run_id') for r in recent if r.get('run_id')}
    for crun in (CloudTrainingRun.query
                 .filter(CloudTrainingRun.status.notin_(ACTIVE_STATES))
                 .order_by(CloudTrainingRun.id.desc()).limit(limit).all()):
        if crun.id in seen_cloud:
            continue
        row = {'source': 'cloud', 'settings': None, **_run_payload(crun)}
        _annotate_preview(row, crun, None)
        recent.append(row)
    # Lineage flag: a row opens the 🌳 tree when it has a parent OR is itself a
    # parent (a continuation branched off it). `records_with_children` is one
    # query over the shown record ids, so a parent still flags even when its
    # child sits outside this window.
    from . import checkpoint_registry
    _rec_ids = [r['record_id'] for r in recent if r.get('record_id')]
    _parents = checkpoint_registry.records_with_children(_rec_ids)
    for r in recent:
        r['lineage'] = bool(r.get('parent_record_id')
                            or (r.get('record_id') in _parents))
    recent.sort(key=lambda r: r.get('created_at') or '', reverse=True)
    recent = recent[:limit]
    # Live LOCAL training: shown as its own card next to the cloud actives;
    # its freshly-registered history row is dropped to avoid the double.
    local = lt.training_status()
    local_active = local if local.get('in_progress') else None
    if local_active and (local.get('current') or {}).get('dataset_id') is not None:
        cur_ds = local['current']['dataset_id']
        for i, r in enumerate(recent):
            if r['source'] == 'local' and r['dataset_id'] == cur_ds:
                # its freshly-registered history row is dropped to avoid the
                # double — carry its share_key (Share config) AND record_id
                # (💻 #N chip) onto the live card.
                dropped = recent.pop(i)
                local_active['share_key'] = dropped.get('share_key')
                local_active['record_id'] = dropped.get('record_id')
                break
    return {'configured': bool(cfg.secret('VAST_API_KEY')),
            'limit': max(1, int((c.get('max_concurrent_runs') or 1))),
            'actives': [_run_payload(r) for r in actives],
            'local_active': local_active,
            'recent': recent,
            'total_price_per_hour': round(sum(r.price_per_hour or 0 for r in actives), 4),
            'month_spend': round(month_spend_usd(), 2),
            'monthly_budget': float(c.get('monthly_budget_usd') or 0)}


def _checkpoint_download_url(dataset_id, source, run_id, filename,
                            family=None, variant=None, base_model=None):
    """The EXISTING browser-download endpoint for one saved checkpoint, so the
    ◉ Graph's per-checkpoint ⬇ reuses the same serving path as the Runs hub /
    Checkpoints panel instead of a parallel one. Cloud saves stream from the
    run's staging dir (train/cloud/checkpoint, extended with ?filename); local
    saves from the run dir (train/checkpoint/file). Query values are url-encoded
    so a trigger with odd characters can't break the link."""
    from urllib.parse import quote
    if source == 'cloud':
        return (f'/api/dataset/{dataset_id}/train/cloud/checkpoint'
                f'?run_id={run_id}&filename={quote(filename)}')
    qs = [f'filename={quote(filename)}']
    if family:
        qs.append(f'train_type={quote(str(family))}')
    if variant:
        qs.append(f'variant={quote(str(variant))}')
    if base_model:
        qs.append(f'base_model={quote(str(base_model))}')
    return f'/api/dataset/{dataset_id}/train/checkpoint/file?' + '&'.join(qs)


def _node_checkpoints(rec, crun):
    """Every save this ONE run produced, as compact nodes for the ◉ Graph: a
    step-sorted list of {step, filename, final, present, download_url}. Cloud
    runs read their harvested staging saves; local runs read the run-dir files
    list_checkpoints already attributes to this record. `present` is True for a
    listed file (it is on disk); a run whose saves are all gone simply lists
    none. Best-effort — a failed scan yields [] (the node shows no pills), never
    a wrong claim. Mirrors the Runs-hub / Checkpoints-panel step extraction so a
    pill's step is exactly what 'continue from here' resumes."""
    out = []
    if crun is not None:
        for c in _run_staging_checkpoints(crun):
            final = not re.search(r'_(\d{6,})\.safetensors$', c['filename'])
            out.append({
                'step': c['step'], 'filename': c['filename'],
                'final': bool(final and crun.status == 'done'), 'present': True,
                'download_url': _checkpoint_download_url(
                    rec.dataset_id, 'cloud', crun.id, c['filename'])})
        return out
    # Local: list_checkpoints may raise if ai-toolkit isn't configured — let it
    # propagate so the caller can tell "scan failed" (None) from "no saves" ([]).
    cks = lt.list_checkpoints(cfg.LOCAL_USER, rec.dataset_id,
                              rec.base_model or '', rec.family, rec.variant)
    for c in cks:
        if c.get('run_source') != 'local' or c.get('run_id') != rec.id:
            continue
        out.append({
            'step': c['step'], 'filename': c['filename'],
            'final': bool(c.get('final')), 'present': True,
            'download_url': _checkpoint_download_url(
                rec.dataset_id, 'local', rec.id, c['filename'],
                family=rec.family, variant=rec.variant, base_model=rec.base_model)})
    return out


def _rec_config(rec):
    """The settings the run actually used, for the Lab inspector. None (not {})
    when a run recorded nothing (legacy) so the UI can say 'config not recorded'
    instead of showing an empty table."""
    if not rec.settings:
        return None
    try:
        cfg = json.loads(rec.settings)
        return cfg if isinstance(cfg, dict) else None
    except (ValueError, TypeError):
        return None


def set_run_note(record_id, text):
    """Save the free-form Lab note on a run. False (no-op) if the record is gone."""
    from ..models import TrainingRunRecord
    from ..extensions import db
    rec = TrainingRunRecord.query.get(record_id)
    if rec is None:
        return False
    rec.note = text or ''
    db.session.commit()
    return True


def set_checkpoint_note(record_id, step, text):
    """Save the free-form Lab note on one checkpoint (record_id, step). False if
    the owning run is gone."""
    from ..models import TrainingRunRecord, CheckpointNote
    from ..extensions import db
    if TrainingRunRecord.query.get(record_id) is None:
        return False
    row = CheckpointNote.query.filter_by(record_id=record_id, step=step).first()
    if row is None:
        row = CheckpointNote(record_id=record_id, step=step)
        db.session.add(row)
    row.note = text or ''
    db.session.commit()
    return True


def checkpoint_notes_for(record_id):
    """{step: note} for a run's annotated checkpoints (empty notes omitted)."""
    from ..models import CheckpointNote
    return {r.step: r.note for r in
            CheckpointNote.query.filter_by(record_id=record_id).all()
            if r.note}


def training_in_progress() -> bool:
    """True while a LoRA training holds the GPU — the Lab's inline generation is
    refused with a 409 in that window (a training and a generation must never
    share the GPU). Reads the same persisted flag the queue and the vision window
    check, so all three agree on 'GPU held by training'."""
    from ..job_queue import queue_manager
    return bool(queue_manager._get_system_state('training_in_progress', False))


# --- Lab inline previews (D) -------------------------------------------------
# The flagship: render ONE same-prompt/same-seed image per selected lineage
# checkpoint, reusing the EXISTING Test-Studio ComfyUI engine pinned to those
# checkpoints at strength 1.0 (not a checkpoint×strength grid). A checkpoint is
# "testable" only when its step has a matching DEPLOYED LoRA in the family pool
# (the same pool the Studio tests) — otherwise there is nothing ComfyUI can load,
# so we say "not deployed" instead of launching a silent no-op.

def _step_of_testable(filename) -> int | None:
    """The training step embedded in a deployed testable LoRA filename, so a
    lineage pill (which carries its own step) can be joined to the deployed LoRA
    of the same step. ai-toolkit deploys the step ZERO-PADDED IN THE MIDDLE, e.g.
    'lora_morgot_cv_000001500_Krea-2-Raw_rc74_v3' (step 1500) — not at the end, so
    an end-anchored match misses every real checkpoint. Match the zero-padded run
    first (unambiguous: base/rc/version tokens aren't zero-padded), then fall back
    to a plain step at the very end ('<trigger>-<step>' / 'lora_<trigger>_<step>').
    A final save with no number yields None (matched by step only)."""
    stem = os.path.basename(str(filename or '')).rsplit('.', 1)[0]
    if stem.lower().startswith('lora_'):
        stem = stem[5:]
    # Zero-padded step anywhere in the name (leading 0, ≥4 digits) — this is the
    # ai-toolkit convention and never collides with 'Krea-2-Raw'/'rc74'/'v3'.
    m = re.search(r'[-_](0\d{3,})(?=[-_]|$)', stem)
    if not m:
        m = re.search(r'[-_](\d+)$', stem)   # legacy: plain step at the end
    return int(m.group(1)) if m else None


def _deployed_run_tag(rec):
    """(source, run_id) as it appears in the DEPLOYED names of THIS record's
    saves. Mirrors import_checkpoint's own rule: a cloud launch is tagged with
    its pod-run id (`_rc<id>`, the ☁ #N chip), everything else with its
    TrainingRunRecord id (`_rl<id>`). (None, None) when a cloud record lost its
    pod-run id — no tag to match, so nothing is claimed."""
    if rec.source == 'cloud':
        return ('cloud', rec.cloud_run_id) if rec.cloud_run_id else (None, None)
    return 'local', rec.id


def _final_step_of(checkpoints) -> int | None:
    """The step of a run's FINAL save among its listed pills: the one flagged
    `final`, else the largest step (a run whose final save exists but isn't
    flagged — cloud runs only flag it once the run is 'done'). None when the run
    lists no save."""
    for c in (checkpoints or []):
        if c.get('final') and c.get('step') is not None:
            return c['step']
    return max((c['step'] for c in (checkpoints or [])
                if c.get('step') is not None), default=None)


def _deploy_version(filename) -> int:
    """The `_v<N>` dataset-version suffix of a deployed name (0 when absent) —
    used only to pick deterministically between several step-less deploys of the
    SAME run (the newest version wins; a plain name sort would rank `_v10` under
    `_v9`)."""
    stem = os.path.basename(str(filename or '')).rsplit('.', 1)[0]
    m = re.search(r'_v(\d+)$', stem)
    return int(m.group(1)) if m else 0


def _testable_by_step(dataset_id, family, run_tag=None, final_step=None) -> dict:
    """{step: deployed_lora_filename} for this dataset+family — the checkpoints
    the Lab can actually generate a preview for. Best-effort: no dataset / no
    deployed LoRA → {} (every pill reads as not-testable, the Generate button
    stays disabled with the app's usual 'needs setup' hint).

    `run_tag`/`final_step` (one run's identity + its last step) additionally join
    that run's STEP-LESS deploy: ai-toolkit names the final save without a step
    (`lora_nova_Krea-2-Raw_rc90_v2`), so it has no number to be matched by and the
    final pill never gained a tick-box even once imported (user-reported on two
    runs). The run tag baked into the deployed name (`_rc<id>`/`_rl<id>`, see
    lt.parse_deployed_run) is what attaches it to its run. A legacy untagged file
    stays unmatched — better no tick-box than one on another run's checkpoint."""
    ds = fds.get_dataset(cfg.LOCAL_USER, dataset_id)
    if not ds:
        return {}
    from . import lora_test_studio as studio
    out, stepless = {}, []
    try:
        cands = studio.list_test_checkpoints(ds, family)
    except Exception:
        return {}
    for c in cands:
        s = _step_of_testable(c.get('filename'))
        if s is not None:
            out[s] = c['filename']
        else:
            stepless.append(c['filename'])
    # Deterministic on collision: a file that NAMES the step always wins over a
    # step-less one claiming the same step, and among several step-less deploys
    # of the same run the highest `_v<N>` wins.
    if stepless and run_tag and run_tag[1] and final_step is not None \
            and final_step not in out:
        mine = [f for f in stepless if lt.parse_deployed_run(f) == tuple(run_tag)]
        if mine:
            out[final_step] = max(mine, key=lambda f: (_deploy_version(f), f))
    return out


def _testable_for_record(dataset_id, family, record_id) -> dict:
    """{step: deployed filename} as seen FROM one run — the dataset+family map
    plus that run's own step-less final save (see _testable_by_step). Degrades to
    the plain map when the record is unknown or its save list can't be read."""
    from ..models import TrainingRunRecord
    rec = db.session.get(TrainingRunRecord, record_id)
    if rec is None:
        return _testable_by_step(dataset_id, family)
    crun = (db.session.get(CloudTrainingRun, rec.cloud_run_id)
            if rec.cloud_run_id else None)
    try:
        cks = _node_checkpoints(rec, crun)
    except Exception:
        cks = []
    return _testable_by_step(dataset_id, family, run_tag=_deployed_run_tag(rec),
                             final_step=_final_step_of(cks))


def checkpoint_previews_for(record_id) -> dict:
    """{step: {status, url, seed}} for a run's inline-generated previews. Each
    stored pointer resolves LIVE to its reused LoraTestImage: 'done' with a served
    url once the file exists, 'failed' if the cell failed, else 'pending' (the job
    is still in the serial queue). A dangling pointer (image row gone) is dropped
    so the node never claims a preview it can't show."""
    from ..models import CheckpointPreview, LoraTestImage
    rows = CheckpointPreview.query.filter_by(record_id=record_id).all()
    if not rows:
        return {}
    img_ids = [r.lora_test_image_id for r in rows if r.lora_test_image_id]
    imgs = ({i.id: i for i in LoraTestImage.query
             .filter(LoraTestImage.id.in_(img_ids)).all()} if img_ids else {})
    out = {}
    for r in rows:
        img = imgs.get(r.lora_test_image_id)
        if img is None:
            continue
        status = img.status if img.status in ('pending', 'done', 'failed') else 'pending'
        url = (f'/api/dataset/{r.dataset_id}/img/{img.filename}'
               if status == 'done' and img.filename else None)
        out[r.step] = {'status': status, 'url': url, 'seed': r.seed}
    return out


def generate_checkpoint_previews(user_id, dataset_id, checkpoints, prompt=None,
                                 seed=None, family=None) -> dict:
    """Point the EXISTING Test-Studio engine at the selected lineage checkpoints,
    each at strength 1.0, with ONE shared prompt+seed — a same-conditions look at
    how the LoRA evolves epoch by epoch. `checkpoints` = [{record_id, step}].

    Each is resolved to the deployed LoRA of its step; checkpoints with no deployed
    LoRA are SKIPPED (reported, never a silent no-op). None resolvable → needs_setup
    True so the route answers an actionable 409 instead of launching an empty run.
    GPU serialization is the engine's own (each cell rides the serial image queue,
    which the queue leaves pending while a training/vision pass holds the GPU); the
    training→409 guard sits in the route. Stores/refreshes a CheckpointPreview per
    rendered checkpoint pointing at the reused LoraTestImage row (regeneration just
    re-points it). Returns {queued, skipped:[{record_id,step,reason}], needs_setup,
    seed}."""
    from ..models import CheckpointPreview
    fam = (family or '').strip().lower() or None
    if fam is None:
        ds = fds.get_dataset(cfg.LOCAL_USER, dataset_id)
        fam = (getattr(ds, 'train_type', None) or 'zimage').lower() if ds else 'zimage'
    # Per RUN: the map is the same dataset+family one, plus that run's own
    # step-less final deploy mapped onto its final step (cached per record so a
    # multi-checkpoint selection scans its runs once).
    by_run = {}
    resolved, skipped = [], []
    for c in (checkpoints or []):
        try:
            rid, step = int(c['record_id']), int(c['step'])
        except (KeyError, TypeError, ValueError):
            continue
        if rid not in by_run:
            by_run[rid] = _testable_for_record(dataset_id, fam, rid)
        fn = by_run[rid].get(step)
        if not fn:
            skipped.append({'record_id': rid, 'step': step, 'reason': 'not_deployed'})
            continue
        resolved.append((rid, step, fn))
    if not resolved:
        return {'queued': 0, 'skipped': skipped, 'needs_setup': True, 'seed': None}

    from . import lora_test_studio as studio
    # Pin the engine to EXACTLY these checkpoints at strength 1.0, one image each
    # (count=1). The Studio validates/preflights + enqueues; a GpuBusyError (vision
    # holding the GPU) or a StudioAssetsMissing (ComfyUI not set up) propagates and
    # the route maps it to the same structured error the Studio already returns.
    result = studio.create_run(
        user_id, dataset_id, checkpoints=[fn for _, _, fn in resolved],
        strengths=[1.0], seed=seed, prompt=prompt, family=fam, count=1)
    ids = result.get('ids') or []
    run_seed = result.get('seed', seed)
    # Single base model × single strength × count=1 → cells are 1:1 with `resolved`
    # in order; zip guards against any engine-side short count (never a wrong link).
    for (rid, step, _fn), img_id in zip(resolved, ids):
        row = CheckpointPreview.query.filter_by(record_id=rid, step=step).first()
        if row is None:
            row = CheckpointPreview(record_id=rid, step=step, dataset_id=dataset_id)
            db.session.add(row)
        row.lora_test_image_id = img_id
        row.prompt = prompt or ''
        row.seed = run_seed
    db.session.commit()
    return {'queued': len(resolved), 'skipped': skipped, 'needs_setup': False,
            'seed': run_seed}


def _record_checkpoints_on_disk(rec) -> int:
    """How many of this run's checkpoints are still on disk RIGHT NOW — the guard
    the "remove a gone run" action checks. Mirrors the graph badge: a cloud run
    counts its staging saves (plus a harvested final LoRA), a local run counts the
    run-dir files the checkpoint scan attributes to this record. Best-effort: a
    scan we can't run reports 0 — an unprovable presence must never block removing
    a run the graph already shows as gone, and never raise (no 500)."""
    try:
        if rec.source == 'cloud':
            crun = (db.session.get(CloudTrainingRun, rec.cloud_run_id)
                    if rec.cloud_run_id else None)
            if crun is None:
                return 0
            n = _staging_save_count(crun)
            if not n and crun.checkpoint_local_path \
                    and os.path.isfile(crun.checkpoint_local_path):
                n = 1
            return n
        return len(_node_checkpoints(rec, None))
    except Exception:
        return 0


def delete_run_record(record_id) -> str:
    """Remove a GONE run (no checkpoints on disk) from the lineage graph — its
    TrainingRunRecord, its checkpoint notes, and (by detaching) its lineage edge.
    METADATA ONLY: the checkpoints are already gone, so nothing on disk is touched.

    Guards instead of deleting silently:
      • a run whose checkpoints are still on disk is REFUSED ('has_saves') so a
        recoverable run is never discarded from under the user;
      • children that resumed FROM this run are DETACHED (parent_record_id → NULL),
        keeping them in the graph as honest "origin unknown" roots rather than
        breaking the tree on a dangling edge.

    Returns 'not_found' | 'has_saves' | 'deleted' | 'conflict'. The FK children
    (CheckpointNote — no relationship cascade in this schema) are deleted and
    FLUSHED before the parent row so SQLite never raises the repo's "delete 500"
    IntegrityError; a stray one is caught and reported as 'conflict', never a 500."""
    from ..models import TrainingRunRecord, CheckpointNote
    from sqlalchemy.exc import IntegrityError
    rec = db.session.get(TrainingRunRecord, int(record_id))
    if rec is None:
        return 'not_found'
    if _record_checkpoints_on_disk(rec) > 0:
        return 'has_saves'
    try:
        # Detach any run that resumed from this one BEFORE deleting it: the child
        # stays displayed (as a root), the parent edge just disappears.
        (TrainingRunRecord.query
         .filter_by(parent_record_id=rec.id)
         .update({'parent_record_id': None}, synchronize_session=False))
        # Delete FK children first and flush, so deleting the parent row can't hit
        # an IntegrityError (the "delete 500" trap — no cascade on these tables).
        CheckpointNote.query.filter_by(record_id=rec.id).delete(
            synchronize_session=False)
        db.session.flush()
        db.session.delete(rec)
        db.session.commit()
        return 'deleted'
    except IntegrityError:
        db.session.rollback()
        return 'conflict'


def _lineage_node(rec, crun, requested_id, failed_local_id):
    """One genealogy-tree node from a provenance record, enriched with what the
    card badge shows: family/variant/base/version/date, run status, and whether
    its LoRA/checkpoints are still ON DISK vs gone (superseded aside or deleted).
    Each node also carries its own `checkpoints` (the ◉ Graph draws them as pills
    under the run and anchors a continuation's edge on the exact one it resumed).
    Checkpoint presence is best-effort — a disk scan that fails degrades to
    None (the UI shows nothing) rather than a wrong "available" claim."""
    node = {
        'record_id': rec.id,
        'parent_record_id': rec.parent_record_id,
        'resumed_from': rec.resumed_from,
        'source': 'cloud' if rec.source == 'cloud' else 'local',
        'dataset_id': rec.dataset_id,
        'dataset_name': _dataset_name(rec.dataset_id),
        'train_type': rec.family,
        'variant': rec.variant,
        'base_model': rec.base_model or '',
        'version': rec.version,
        'steps': rec.steps,
        'config': _rec_config(rec),
        'note': rec.note or '',
        'has_note': bool((rec.note or '').strip()),
        'created_at': rec.created_at.isoformat() if rec.created_at else None,
        'is_current': rec.id == requested_id,
        # A record with a resume step but no resolvable parent (legacy: the edge
        # was never persisted) is an honest ROOT with a discreet "origin unknown".
        'origin_unknown': bool(rec.resumed_from and not rec.parent_record_id),
    }
    if crun is not None:
        node['run_id'] = crun.id
        node['status'] = crun.status
        node['checkpoint_ready'] = bool(
            crun.checkpoint_local_path and os.path.isfile(crun.checkpoint_local_path))
        node['checkpoints'] = _node_checkpoints(rec, crun)
        node['saves'] = _staging_save_count(crun)
    else:
        node['status'] = ('error' if (rec.source == 'local'
                                       and rec.id == failed_local_id) else None)
        # Local checkpoints still on disk that list_checkpoints attributes to
        # THIS record (record_for_mtime). Superseded/deleted saves simply don't
        # appear — honest "what's recoverable now", never invented.
        try:
            cks = _node_checkpoints(rec, None)
            node['checkpoints'] = cks
            node['saves'] = len(cks)
            node['checkpoint_ready'] = len(cks) > 0
        except Exception:
            node['checkpoints'] = []
            node['saves'] = None
            node['checkpoint_ready'] = None
    _cnotes = checkpoint_notes_for(rec.id)
    _cprev = checkpoint_previews_for(rec.id)
    # A pill is `testable` when its step maps to a deployed LoRA the Studio engine
    # can load — the front enables Generate only for testable selections and shows
    # the app's usual 'needs setup' hint otherwise. `preview_*` render the inline
    # thumbnail (or its pending/failed state) in the node card.
    # Scoped to THIS run so its step-less final deploy (`..._rc90_v2`, no step in
    # the name) joins its own final pill instead of going unmatched.
    _testable = _testable_by_step(rec.dataset_id, rec.family,
                                  run_tag=_deployed_run_tag(rec),
                                  final_step=_final_step_of(node.get('checkpoints')))
    for _ck in (node.get('checkpoints') or []):
        _step = _ck.get('step')
        _ck['note'] = _cnotes.get(_step, '')
        _ck['testable'] = _step in _testable
        _pv = _cprev.get(_step)
        if _pv:
            _ck['preview_url'] = _pv.get('url')
            _ck['preview_status'] = _pv.get('status')
    return node


def run_lineage(record_id) -> dict:
    """The genealogy tree for the lineage that record_id belongs to: every
    launch (local AND cloud) linked by continuations, as nodes + parent→child
    edges. `single` is True for a lone run with no parent and no children (the
    UI then offers no tree). Superseded branch: an edge whose child resumed
    BELOW where its parent ended means the parent has saves set aside — flagged
    on the edge (superseded) and on the parent node (has_superseded_tail)."""
    from . import checkpoint_registry as reg
    from ..models import TrainingRunRecord
    records = reg.resolve_lineage(record_id)
    if not records:
        return {'nodes': [], 'edges': [], 'root_id': None,
                'current_id': None, 'single': True}
    requested_id = int(record_id)
    cloud_ids = {r.cloud_run_id for r in records if r.cloud_run_id}
    cloud_by_id = ({r.id: r for r in CloudTrainingRun.query
                    .filter(CloudTrainingRun.id.in_(cloud_ids)).all()}
                   if cloud_ids else {})
    failed_local_id = (lt.failed_local_run() or (None, None))[0]
    nodes = [_lineage_node(rec, cloud_by_id.get(rec.cloud_run_id),
                           requested_id, failed_local_id)
             for rec in records]
    steps_by_id = {r.id: (r.steps or 0) for r in records}
    edges, superseded_parents = [], set()
    for rec in records:
        pid = rec.parent_record_id
        if pid and pid in steps_by_id:
            superseded = (rec.resumed_from is not None
                          and rec.resumed_from < steps_by_id[pid])
            if superseded:
                superseded_parents.add(pid)
            edges.append({'parent': pid, 'child': rec.id,
                          'resumed_from': rec.resumed_from,
                          'superseded': superseded})
    for node in nodes:
        node['has_superseded_tail'] = node['record_id'] in superseded_parents
    return {'root_id': records[0].id, 'current_id': requested_id,
            'nodes': nodes, 'edges': edges, 'single': len(nodes) < 2}


def _lineage_edges(records):
    """parent→child edges over a SET of records (the genealogy the ◉ Graph draws),
    with the superseded flag (a child resumed BELOW where its parent ended) — the
    same rule run_lineage uses, factored so a dataset-wide forest reuses it."""
    steps_by_id = {r.id: (r.steps or 0) for r in records}
    ids = set(steps_by_id)
    edges, superseded_parents = [], set()
    for rec in records:
        pid = rec.parent_record_id
        if pid and pid in ids:
            superseded = (rec.resumed_from is not None
                          and rec.resumed_from < steps_by_id[pid])
            if superseded:
                superseded_parents.add(pid)
            edges.append({'parent': pid, 'child': rec.id,
                          'resumed_from': rec.resumed_from,
                          'superseded': superseded})
    return edges, superseded_parents


def dataset_lineage(dataset_id, train_type=None, variant=None) -> dict:
    """Every run this dataset produced, as ONE genealogy forest for the ◉ Graph
    the Checkpoints & LoRAs manager opens — not a single record's lineage but all
    of them (several independent trees stack; the graph layout already handles
    multiple roots). Optionally scoped to the family/variant the panel is showing,
    so it matches the checkpoints listed there. Nodes carry their checkpoints
    exactly like run_lineage; there is no single 'current' run here (root_id /
    current_id are None). Empty dataset → an empty, safe shape."""
    from ..models import TrainingRunRecord
    fam = fds.normalize_train_type(train_type) if train_type else None
    q = TrainingRunRecord.query.filter_by(dataset_id=dataset_id)
    if fam:
        q = q.filter_by(family=fam)
    if variant:
        q = q.filter_by(variant=str(variant).strip().lower())
    records = q.order_by(TrainingRunRecord.id.asc()).all()
    if not records:
        return {'nodes': [], 'edges': [], 'root_id': None,
                'current_id': None, 'single': True}
    cloud_ids = {r.cloud_run_id for r in records if r.cloud_run_id}
    cloud_by_id = ({r.id: r for r in CloudTrainingRun.query
                    .filter(CloudTrainingRun.id.in_(cloud_ids)).all()}
                   if cloud_ids else {})
    failed_local_id = (lt.failed_local_run() or (None, None))[0]
    nodes = [_lineage_node(rec, cloud_by_id.get(rec.cloud_run_id), None,
                           failed_local_id) for rec in records]
    edges, superseded_parents = _lineage_edges(records)
    for node in nodes:
        node['has_superseded_tail'] = node['record_id'] in superseded_parents
    return {'root_id': None, 'current_id': None,
            'nodes': nodes, 'edges': edges, 'single': len(nodes) < 2}


def gpu_tiers(user_id, dataset_id, train_type=None, steps=None,
              variant=None) -> dict:
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
    # Slider mode rides the same offers/pods as its family (see
    # launch_cloud_training) — no separate refusal here.
    if fam == 'sdxl':
        raise ValueError('SDXL training needs a local base checkpoint — '
                         'cloud training supports Z-Image, Krea and FLUX.2 Klein')
    # flux2klein passe (cf. launch_cloud_training) — seul flux reste local-only.
    if fam == 'flux':
        raise ValueError('FLUX.1 training is local-only for now — '
                         'cloud training supports Z-Image, Krea and FLUX.2 Klein')
    selected_variant = str(
        variant or getattr(ds, 'train_variant', None)
        or lt._default_variant_for(fam)).strip().lower()
    if selected_variant not in lt._valid_variants_for(fam):
        selected_variant = lt._default_variant_for(fam)
    n_steps = (int(steps) if steps else lt.default_steps(
        ds, train_type=fam, variant=selected_variant))
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
        min_disk_bw_mbps=int(c.get('min_disk_bw_mbps') or 0),
        verified_only=bool(c.get('verified_only', True)),
        secure_cloud_only=bool(c.get('secure_cloud_only', False))))
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
            'variant': selected_variant,
            'max_price_per_hour': price_cap,
            'max_runtime_minutes': max_runtime}


def cloud_checkpoint_groups(dataset_id, train_type=None, variant=None) -> list:
    """Locally-synced cloud checkpoints GROUPED by their source run (newest run
    first, step-sorted within a run). Each group carries the run's identity and
    outcome — id / status / GPU / cost / timing, the SAME facts the Runs hub
    shows — so the Checkpoints panel can label WHICH run produced which epochs
    and deep-link back to its row, instead of listing several indistinguishable
    step-500→final sets. Runs whose saves were all hand-deleted are omitted."""
    fam = fds.normalize_train_type(train_type) if train_type else None
    wanted_variant = str(variant).strip().lower() if variant else None
    groups = []
    for run in (CloudTrainingRun.query.filter_by(dataset_id=dataset_id)
                .order_by(CloudTrainingRun.id.desc()).all()):
        if fam and (_run_family(run) or fam) != fam:
            continue
        if wanted_variant and (
                str(_run_param(run, 'variant') or wanted_variant).lower()
                != wanted_variant):
            continue
        if not run.staging_dir or not os.path.isdir(run.staging_dir):
            continue
        run_variant = _run_param(run, 'variant')
        entries = []
        for name in os.listdir(run.staging_dir):
            if not name.lower().endswith('.safetensors'):
                continue
            m = re.search(r'_(\d{6,})\.safetensors$', name)
            step = int(m.group(1)) if m else int(_run_param(run, 'steps') or 0)
            entries.append({'filename': name, 'step': step, 'cloud': True,
                            'run_id': run.id, 'version': _run_param(run, 'version'),
                            'variant': run_variant,
                            'final': bool(not m and run.status == 'done'),
                            'active': run.status in ACTIVE_STATES,
                            'trained_at': run.created_at.isoformat()
                                          if run.created_at else None})
        if not entries:
            continue
        entries.sort(key=lambda e: (e['step'], e['final']))
        groups.append({
            'run_id': run.id, 'source': 'cloud', 'status': run.status,
            'active': run.status in ACTIVE_STATES, 'gpu': run.gpu_name,
            'price_per_hour': run.price_per_hour,
            'cost_estimate': _cost_estimate(run),
            'version': _run_param(run, 'version'), 'variant': run_variant,
            'train_type': _run_family(run),
            'created_at': run.created_at.isoformat() if run.created_at else None,
            'finished_at': run.finished_at.isoformat() if run.finished_at else None,
            'checkpoints': entries,
        })
    return groups


def cloud_checkpoints(dataset_id, train_type=None, variant=None) -> list:
    """Flat view of cloud_checkpoint_groups — every synced save of this dataset
    (+family/variant filters), newest run first, step-sorted within a run. Kept
    for callers that reason on individual saves (import / delete / continue);
    the panel groups them per run via cloud_checkpoint_groups."""
    out = []
    for g in cloud_checkpoint_groups(dataset_id, train_type, variant):
        out.extend(g['checkpoints'])
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
