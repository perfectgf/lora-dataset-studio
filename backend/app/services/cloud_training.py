"""Cloud LoRA training orchestrator (vast.ai ephemeral pod).

State machine (CloudTrainingRun.status):
  preparing -> provisioning -> uploading -> training -> downloading
  -> terminating -> done | stopped | error | error_pod_kept

Lifecycle invariant: ordinary LoRA exits, explicit user stops and max-runtime
caps destroy the instance. Once dense Krea training has started, unexpected
failure keeps the pod recoverable until its direct Hugging Face delivery and
licence metadata are verified; only then may completion destroy it. The local
training path is untouched: a cloud run never sets 'training_in_progress', so
local generation/captioning stay available."""
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

from sqlalchemy import func

from .. import config as cfg
from ..extensions import db
from ..models import CloudTrainingRun, SystemState
from . import checkpoint_registry
from . import dataset_activity
from . import dense_local_delivery as dld
from . import dense_weights
from . import face_dataset_service as fds
from . import gpu_speed
from . import lora_training as lt
from . import cloud_run_dataset as crd
from . import vast_client
# Imported for its checkpoint-name parser, which is not video-specific: it is the
# one place that knows a save may carry a multistage suffix, and every family's
# step is read through it so the single-file and the paired case cannot drift.
# The module is pure (no torch, no ffmpeg, no database), so this costs nothing.
from . import video_run_lineage
from . import video_targets
from . import video_training
from .aitoolkit_remote import RemoteAiToolkit, TransferCancelled

logger = logging.getLogger(__name__)

ACTIVE_STATES = ('preparing', 'provisioning', 'uploading', 'training',
                 'downloading', 'terminating')

_stop_events = {}        # run_id -> threading.Event
_monitor_threads = {}    # run_id -> threading.Thread
_supervisor_thread = None    # the one out-of-monitor watchdog (start_supervisor)
_auto_retry_lock = threading.Lock()

# -- watchdog / stop authority ------------------------------------------------
# A pod bills by the hour, so every guarantee below is anchored on the DATABASE
# (and on the vast API), never on in-process state: a threading.Event does not
# survive a restart and is worthless when the thread meant to observe it is
# dead or wedged.
SUPERVISOR_INTERVAL_SECONDS = 60
# Listing the whole vast.ai account every watchdog minute is unnecessary, but
# terminal kept pods still need reaping while the desktop app stays open.
_ORPHAN_RECONCILE_INTERVAL_SECONDS = 5 * 60
# Database silence past which a monitor thread is no longer trusted to carry
# out a stop -- it writes phase_detail every poll (~10 s), so two minutes of
# nothing means it is not coming back in time to save a paid pod.
STOP_HANDOFF_SECONDS = 120
# ... and how long a stop handed to a (then) responsive monitor may stay
# unfinished before the supervisor terminates the pod itself. Generous enough
# to cover the graceful path: stop the remote job, pull the last checkpoint.
STOP_DEADLINE_SECONDS = 15 * 60
# The supervisor defers to a live monitor on the runtime cap (the monitor
# rescues the checkpoint first); it only acts if the monitor did not.
_SUPERVISOR_MARGIN_SECONDS = 120
# Floor for phases that are legitimately silent (staging, boot, final
# download). The runtime cap stays their real backstop.
_SILENT_PHASE_FREEZE_SECONDS = 120 * 60
_FREEZE_WATCHDOG_MINUTES = 45   # default when config carries no value
# ... except the dataset upload, which stopped being one of those silent phases
# the moment it started reporting bytes (_write_upload_progress). Judged on
# BYTES and not on wall time, so this is not a budget for the transfer — a
# 24 GB dataset may legitimately take hours — but the answer to "how long may
# nothing at all reach the pod before we stop paying for it?". Config key
# `cloud.upload_stall_minutes`; 0 (like freeze_watchdog_minutes) turns it off.
_UPLOAD_STALL_MINUTES = 25
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
_FULL_TRANSFORMER_ARTIFACT = 'full_transformer'
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
    # « Train anyway » on the Hugging Face private-storage pre-check. Replayed
    # like the others: the ceiling the check compares against is an ESTIMATE, so
    # a user who already judged their own account must not have an automatic
    # retry blocked by the same guess.
    'allow_hf_storage',
    # ... and its twin on THIS machine's disk, for a dense run that delivers
    # locally. Same reasoning, same replay: the size of a checkpoint that does
    # not exist yet is an estimate too.
    'allow_local_disk',
    # « Launch anyway » on the same-(dataset, family) sibling guard: a second
    # run rents a second pod, so a double-click must stay free — but a
    # deliberate settings A/B must not wait hours. Stamped so the Runs hub can
    # explain why two pods exist; replayed like the others.
    'allow_parallel_run',
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
    # Working area only (dataset copy, samples, log). Relocatable through
    # Settings › Storage; '' means DATA_DIR/cloud_runs, the historical place.
    return cfg.cloud_runs_root()


# ── The durable checkpoint store ──────────────────────────────────────────────
# Written after an incident: "Clean finished runs" said it was trashing
# "checkpoint duplicates already imported", but a checkpoint that was never
# deployed to ComfyUI had NO duplicate — staging was its only copy. Emptying the
# trash destroyed it. Weights now land in their own store, which no cleanup path
# touches, and staging goes back to being genuinely disposable.

def checkpoint_store_dir(run, create=False) -> str | None:
    """Where THIS run's ``.safetensors`` live for good: ``<store>/run_<id>``.
    None for a row that has no id yet (never flushed) — a run with no identity
    has no store, and inventing one would scatter files under ``run_None``."""
    if getattr(run, 'id', None) is None:
        return None
    d = cfg.checkpoints_root(create=create) / f'run_{int(run.id)}'
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _checkpoint_dirs(run) -> list:
    """Directories that may hold this run's saves, in PRECEDENCE order: the
    store first, then the legacy staging dir. The legacy read is what keeps an
    install that trained before the store existed from losing its checkpoint
    list before the retrofit pass has moved anything."""
    dirs = []
    if getattr(run, 'id', None) is not None:
        dirs.append(str(cfg.checkpoints_root(create=False) / f'run_{int(run.id)}'))
    if run.staging_dir:
        dirs.append(run.staging_dir)
    return dirs


def run_checkpoint_files(run) -> dict:
    """``{filename: absolute path}`` of every save of this run, store winning
    over staging on a duplicate name. Empty dict when the run produced none —
    never raises on a directory that is gone (purged, hand-deleted, relocated)."""
    out = {}
    for d in _checkpoint_dirs(run):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            if name.lower().endswith('.safetensors') and name not in out:
                out[name] = os.path.join(d, name)
    return out


def run_checkpoint_path(run, filename) -> str | None:
    """Resolve ONE save of this run by basename, or None. Basename-only by
    construction, so a caller can hand it a filename straight from a request."""
    if not filename or os.path.basename(filename) != filename:
        return None
    return run_checkpoint_files(run).get(filename)


def _adopt_checkpoints_into_store(run) -> int:
    """Move this run's ``.safetensors`` out of staging and into the store.

    Runs after every sync and again from the retrofit pass, so a weight is in
    the store BEFORE any cleanup can look at staging. Best-effort per file: a
    move that fails leaves the file where it is (still readable through the
    legacy path) rather than losing it. Returns how many files moved."""
    sd = run.staging_dir
    if not sd or not os.path.isdir(sd):
        return 0
    moved = 0
    dest_dir = None
    try:
        names = sorted(os.listdir(sd))
    except OSError:
        return 0
    for name in names:
        if not name.lower().endswith('.safetensors'):
            continue
        if dest_dir is None:
            dest_dir = checkpoint_store_dir(run, create=True)
            if dest_dir is None:
                return moved
        src = os.path.join(sd, name)
        dest = os.path.join(dest_dir, name)
        try:
            if os.path.exists(dest):
                # Already in the store (a previous pass, or a re-sync): the
                # staging copy is the redundant one and may go.
                if os.path.getsize(dest) == os.path.getsize(src):
                    os.remove(src)
                    continue
                # Same name, different bytes — keep both rather than pick.
                continue
            shutil.move(src, dest)
            moved += 1
        except OSError as e:
            logger.warning('checkpoint %s not moved into the store: %s', name, e)
    if moved and run.checkpoint_local_path:
        base = os.path.basename(run.checkpoint_local_path)
        landed = os.path.join(dest_dir or '', base)
        if os.path.isfile(landed):
            _set(run, checkpoint_local_path=landed)
    return moved


_STORE_MIGRATION_KEY = 'cloud_checkpoint_store'
_STORE_MIGRATION_VERSION = 1


def migrate_checkpoints_into_store(force=False) -> dict:
    """Retrofit: sweep every known run's staging dir into the checkpoint store.

    Called once at boot (guarded by a persisted version flag) and on demand from
    Settings › Storage, because an install that has not booted since the store
    landed still keeps its only copies in a directory the cleanup may trash.
    Never raises — a failed retrofit must not keep the app from starting."""
    state = None
    try:
        state = SystemState.query.filter_by(key=_STORE_MIGRATION_KEY).first()
        if not force and state is not None \
                and str(state.value or '') == str(_STORE_MIGRATION_VERSION):
            return {'ran': False, 'runs': 0, 'moved': 0}
    except Exception as e:                       # pragma: no cover - legacy DB
        logger.debug('checkpoint store migration flag unavailable: %s', e)
    runs = 0
    moved = 0
    try:
        for run in CloudTrainingRun.query.all():
            n = _adopt_checkpoints_into_store(run)
            if n:
                runs += 1
                moved += n
    except Exception as e:
        logger.warning('checkpoint store migration incomplete: %s', e)
    try:
        if state is None:
            state = SystemState(key=_STORE_MIGRATION_KEY,
                                value=str(_STORE_MIGRATION_VERSION))
            db.session.add(state)
        else:
            state.value = str(_STORE_MIGRATION_VERSION)
        db.session.commit()
    except Exception as e:                       # pragma: no cover - legacy DB
        logger.debug('checkpoint store migration flag not persisted: %s', e)
        db.session.rollback()
    if moved:
        logger.info('checkpoint store: moved %s save(s) of %s run(s) out of staging',
                    moved, runs)
    return {'ran': True, 'runs': runs, 'moved': moved}


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


def _assert_dense_custom_base_readable(repo_id, token, timeout=8):
    """Fail a DENSE launch whose pod credential cannot read the custom base.

    A custom base rides to the pod through a private repository pushed with the
    general ``HF_TOKEN``; a dense pod is deliberately cut off from that
    credential and receives ``HF_CLOUD_TOKEN`` instead (``_hf_token_for_mode``).
    When both belong to the same account, the delivery-namespace scope already
    covers the base repo and this check passes silently. When they do not — a
    delivery org, a second account — the download 403s ON THE POD, after the
    GPU is paid for. Same fail-open contract as the official-base gate: only an
    outright 401/403 blocks."""
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
            return
        raise ValueError(
            f'HF_CLOUD_TOKEN cannot read {repo_id}, the private repository the '
            'rented GPU downloads your custom base from. Full-model runs use '
            'HF_CLOUD_TOKEN only, so its delivery namespace must be the same '
            'Hugging Face account that holds this base repository — or use the '
            'official Krea 2 base. Nothing was rented, so this run cost '
            'nothing.') from None
    except Exception:                        # noqa: BLE001 — offline/outage: fail open
        return


def _make_hf_api(token):
    """Small Hugging Face seam kept injectable for offline unit tests."""
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise RuntimeError('huggingface-hub is required for full_transformer '
                           'cloud delivery') from e
    return HfApi(token=token)


_KREA_BASE_REPO = 'krea/Krea-2-Raw'
# Both official Krea 2 repositories a dense run can be pointed at. The token
# audit tolerates a read scope on either and REQUIRES the one this run needs —
# a Turbo dense run whose token can only read Raw would 403 on the pod, after
# the GPU is paid for.
_KREA_BASE_REPOS = (_KREA_BASE_REPO, lt.KREA_TURBO_BASE)
_KREA_BASE_REPOS_LOWER = {repo.lower() for repo in _KREA_BASE_REPOS}
_KREA_LICENSE_FILENAME = 'LICENSE.pdf'
_KREA_LICENSE_LINK = (
    'https://huggingface.co/krea/Krea-2-Raw/blob/main/LICENSE.pdf')
_KREA_REQUIRED_ATTRIBUTION = (
    'Krea 2 is licensed under the Krea 2 Community License Agreement. '
    'For more information, visit https://krea.ai/krea-2-licensing.')
_KREA_NOTICE = (
    f'{_KREA_REQUIRED_ATTRIBUTION}\n\n'
    'This repository contains a modified derivative of Krea 2. The training '
    'dataset and resulting weights differ from the official model.\n\n'
    'This modified derivative is unofficial, is not an official Krea product, '
    'and is not endorsed by Krea.\n')

# A Krea 2 full-transformer save is measured in tens of gigabytes.  Eight GiB
# is deliberately conservative (well below the expected bf16 checkpoint) while
# still rejecting an LFS pointer, empty upload, partial multipart upload, or a
# small unrelated safetensors file.  Verification only reads Hub metadata; the
# application never downloads the dense checkpoint.
_FULL_TRANSFORMER_MIN_WEIGHT_BYTES = 8 * 1024 ** 3


def _hf_token_for_mode(training_mode: str):
    """Return the least-privilege token for one execution mode.

    Dense Krea runs must never inherit the general-purpose ``HF_TOKEN``.  The
    remote still calls its environment/settings key HF_TOKEN, but the value we
    put there is selected here and comes exclusively from HF_CLOUD_TOKEN.
    """
    return cfg.secret('HF_CLOUD_TOKEN' if training_mode == 'full_transformer'
                      else 'HF_TOKEN')


def _hf_token_for_run(run):
    return _hf_token_for_mode(_run_training_mode(run))


def _permission_values(raw) -> set:
    """Normalize one Hub permission list without guessing malformed values."""
    if not isinstance(raw, (list, tuple)):
        return set()
    return {value.strip().lower() for value in raw
            if isinstance(value, str) and value.strip()}


def _full_transformer_delivery_namespace(who, required_base_repo=None) -> str:
    """Validate and return the token's single delivery-only namespace.

    ``required_base_repo`` is the OFFICIAL Krea repository this particular run
    needs the pod to download — Raw or Turbo. ``None`` means the run trains from
    a custom base living in a private repository inside the delivery namespace,
    so no official read scope is required. A read scope on either official
    repository is tolerated in every case: it is the recommended token shape and
    a user who trains both variants should not have to re-issue a token between
    runs.

    Hugging Face cannot grant write access to a repository that does not exist
    yet.  Dense runs create a private repository per run, so the narrowest
    technically usable contract is one *dedicated* user/org namespace scope.
    Everything else is fail-closed: no global permission, exact read scope for
    the gated base, exactly one namespace with read+write, and no unrelated
    scoped permissions.  The UI/docs explicitly require that namespace to
    contain only LDS delivery repositories.
    """
    access = ((who or {}).get('auth') or {}).get('accessToken') or {}
    fine = access.get('fineGrained')
    if not isinstance(fine, dict):
        raise ValueError('HF_CLOUD_TOKEN has no inspectable fine-grained scopes')

    raw_global = fine.get('global')
    if not isinstance(raw_global, list) or any(
            not isinstance(value, str) for value in raw_global):
        raise ValueError(
            'HF_CLOUD_TOKEN global permissions are not safely inspectable')
    global_permissions = _permission_values(raw_global)
    can_read_gated = fine.get('canReadGatedRepos', False)
    if global_permissions or can_read_gated not in (False, None):
        raise ValueError(
            'HF_CLOUD_TOKEN must not have global or broad permissions; grant '
            'an exact read scope for krea/Krea-2-Raw instead')

    scopes = fine.get('scoped')
    if not isinstance(scopes, list):
        raise ValueError('HF_CLOUD_TOKEN scoped permissions are not inspectable')

    base_reads = set()
    delivery_scopes = []
    for scope in scopes:
        if not isinstance(scope, dict):
            raise ValueError('HF_CLOUD_TOKEN contains a malformed scope')
        raw_permissions = scope.get('permissions')
        if not isinstance(raw_permissions, list) or any(
                not isinstance(value, str) for value in raw_permissions):
            raise ValueError(
                'HF_CLOUD_TOKEN contains permissions that are not safely inspectable')
        permissions = _permission_values(raw_permissions)
        if not permissions:
            continue
        entity = scope.get('entity')
        if not isinstance(entity, dict):
            raise ValueError(
                'HF_CLOUD_TOKEN permissions must identify their scoped resource')
        entity_type = str(entity.get('type') or '').strip().lower()
        entity_name = str(entity.get('name') or '').strip()

        if entity_type == 'model' and entity_name.lower() in _KREA_BASE_REPOS_LOWER:
            if permissions != {'repo.content.read'}:
                raise ValueError(
                    f'{entity_name} must have exact repo.content.read access only')
            base_reads.add(entity_name.lower())
            continue

        if 'repo.write' in permissions:
            if entity_type not in {'user', 'org'} or not entity_name:
                raise ValueError(
                    'HF_CLOUD_TOKEN write access must target one dedicated user '
                    'or organization delivery namespace')
            if permissions != {'repo.content.read', 'repo.write'}:
                raise ValueError(
                    'the delivery namespace must have only repo.content.read and '
                    'repo.write permissions')
            delivery_scopes.append((entity_type, entity_name))
            continue

        # Even read-only access to unrelated private resources contradicts the
        # delivery-only credential promise and increases the token's blast
        # radius if the paid pod is compromised.
        raise ValueError(
            'HF_CLOUD_TOKEN contains an unrelated scope; keep only exact Krea '
            'base read and one dedicated delivery namespace')

    if required_base_repo and required_base_repo.lower() not in base_reads:
        raise ValueError(
            'HF_CLOUD_TOKEN needs exact repo.content.read access to '
            f'{required_base_repo}')
    if len(delivery_scopes) != 1:
        raise ValueError(
            'HF_CLOUD_TOKEN needs exactly one dedicated delivery namespace '
            'with repo.content.read and repo.write access')

    entity_type, namespace = delivery_scopes[0]
    identity = str((who or {}).get('name') or '').strip()
    if entity_type == 'user':
        if not identity or namespace.lower() != identity.lower():
            raise ValueError(
                'HF_CLOUD_TOKEN user scope does not match its authenticated namespace')
    else:
        orgs = (who or {}).get('orgs')
        if not isinstance(orgs, list):
            raise ValueError(
                'HF_CLOUD_TOKEN organization membership cannot be verified')
        org_names = {
            str(org.get('name') if isinstance(org, dict) else org).strip().lower()
            for org in orgs
        }
        if namespace.lower() not in org_names:
            raise ValueError(
                'HF_CLOUD_TOKEN organization scope is not owned by this identity')
    return namespace


_BROAD_HF_TOKEN_WARNING = (
    'This Hugging Face token has global write access to every repository the '
    'account can modify. It is accepted, but a dedicated fine-grained token '
    'limited to Krea 2 reads and one LDS delivery namespace is strongly '
    'recommended.')


def _validate_full_transformer_token(token, _api=None,
                                     required_base_repo=_KREA_BASE_REPO):
    """Require real Krea read rights and usable delivery write rights.

    ``whoami`` proves the token type and advertised scopes; listing the gated
    official base proves that the token/account can actually read it.  Private
    repository creation and compliance uploads later provide the real write
    check before a GPU is ever rented.

    ``required_base_repo`` names the repository THIS run needs the pod to
    download: ``krea/Krea-2-Raw`` (the default, and what Settings shows with no
    run in hand), ``krea/Krea-2-Turbo``, or ``None`` for a custom base, which
    lives in a private repository covered by the delivery-namespace scope
    instead. Hardcoding Raw here used to be free — it was the only base a dense
    run could have.
    """
    if not token:
        raise ValueError(
            'full_transformer cloud training requires HF_CLOUD_TOKEN with '
            'repository write access (fine-grained recommended; global write '
            'accepted with a warning)')
    api = _api or _make_hf_api(token)
    try:
        who = api.whoami() or {}
    except Exception:
        raise ValueError(
            'HF_CLOUD_TOKEN could not be authenticated; verify the token and '
            'try again (fine-grained recommended; global write accepted)') from None
    access = ((who.get('auth') or {}).get('accessToken') or {})
    role = re.sub(r'[^a-z]', '', str(access.get('role') or '').lower())
    if role == 'finegrained':
        namespace = _full_transformer_delivery_namespace(who, required_base_repo)
        broad_access = False
    elif role == 'write':
        namespace = str((who or {}).get('name') or '').strip()
        if not namespace:
            raise ValueError(
                'HF_CLOUD_TOKEN authenticated identity has no usable delivery namespace')
        broad_access = True
    else:
        raise ValueError(
            'HF_CLOUD_TOKEN requires write access to create and upload the '
            'private delivery repository; read-only tokens cannot be used')
    if required_base_repo:
        try:
            api.list_repo_files(repo_id=required_base_repo, repo_type='model')
        except Exception:
            raise ValueError(
                f'HF_CLOUD_TOKEN cannot read {required_base_repo}; accept its '
                'licence with the same Hugging Face account and grant this '
                'token access') from None
    return api, str(namespace), broad_access


def _dense_whoami(api) -> dict:
    """``whoami`` as a plain dict, or {} — used only to read ``isPro`` for the
    storage-allowance ESTIMATE. Never fatal: a missing plan just means the
    forecast falls back to the free-tier figure."""
    try:
        who = api.whoami()
    except Exception:
        return {}
    return who if isinstance(who, dict) else {}


# The Hub's refusal when an account's PRIVATE allowance is exhausted. Matched on
# the pod's job info and its log because that 403 arrives mid-training, on the
# checkpoint push — the run is otherwise perfectly healthy, and the generic
# "remote job error" it used to produce sent people looking at the GPU.
_HF_STORAGE_FULL_SIGNS = (
    'private repository storage limit',
    'private storage limit',
    'storage limit reached',
    'storage quota exceeded',
    'exceeded your storage quota',
)


def _hf_storage_full(text) -> bool:
    low = str(text or '').lower()
    return any(sign in low for sign in _HF_STORAGE_FULL_SIGNS)


def _assert_dense_storage_headroom(namespace, dense_api, allow_override,
                                   keeps=None, fp8_export=True, required=True):
    """Refuse a dense launch whose delivery plainly will not fit in the
    namespace's PRIVATE Hugging Face storage — before anything is rented.

    Measured with HF_CLOUD_TOKEN and ONLY with it. Falling back to the general
    HF_TOKEN would measure more accounts, and it was tried — but dense runs are
    deliberately cut off from that credential (see _hf_token_for_mode, and the
    launch test that asserts no dense path ever touches it), and a read that
    widens a least-privilege boundary to improve an ESTIMATE is a bad trade. The
    quota that matters is the delivery namespace's anyway, which is what this
    token is scoped to. The cost is a real blind spot: a token that cannot list
    that namespace makes the forecast unknown, and unknown never blocks. The
    Settings ▸ Training storage card, which manages the caches rather than the
    delivery, does use HF_TOKEN and stays fully sighted.

    ``required=False`` measures without refusing: the caller then only wants the
    forecast, because the Hub copy is a backup taken AFTER a verified local one
    and can no longer cost the run anything.
    """
    from . import hf_storage
    forecast = hf_storage.dense_storage_forecast(
        namespace, None,
        keeps=(lt.FULL_TRANSFORMER_MAX_STEP_SAVES if keeps is None else keeps),
        who=_dense_whoami(dense_api), _api=dense_api, fp8_export=fp8_export)
    if required and forecast.get('fits') is False and not allow_override:
        raise ValueError(hf_storage.storage_refusal_message(forecast))
    return forecast


def _dense_remote_failure(status, info, log_text) -> tuple:
    """(phase_detail, error) for a dense run whose remote job ended badly.

    Every branch here is RECOVERABLE by construction — the caller is
    _keep_full_transformer_pod, which closes as ``error_pod_kept`` and does not
    destroy the instance. Split out of the poll loop so the storage verdict is
    testable without driving a whole monitor.
    """
    if _hf_storage_full(f'{info}\n{log_text}'):
        return ('HF private storage full — free space then resume from the '
                'kept pod',
                'Hugging Face refused the checkpoint push: your PRIVATE storage '
                'allowance is full. Free space in Settings ▸ Training ▸ Hugging '
                'Face storage (the lds-base-* caches are re-pushable), then '
                'recover the checkpoint from the kept pod — it is held for the '
                'recovery window, not destroyed.')
    return (f'Remote dense job unexpectedly {status}; pod kept',
            f'remote job {status}; pod kept for recovery')


def full_transformer_token_status(token, _api=None,
                                  required_base_repo=_KREA_BASE_REPO) -> dict:
    """Return a secret-free readiness state for one prospective cloud token.

    This intentionally performs the same authenticated scope/read checks as
    launch.  Launch calls the validator again as the authoritative TOCTOU-safe
    gate; callers may cache this serializable advisory response if desired.
    """
    base = {
        'configured': bool(token),
        'namespace': None,
        'settings_focus': 'HF_CLOUD_TOKEN',
        'warning': None,
    }
    if not token:
        return {
            **base, 'ok': False, 'code': 'missing', 'severity': 'error',
            'error': ('Full-model Krea 2 cloud training requires a dedicated '
                      'HF_CLOUD_TOKEN.'),
        }
    try:
        _api_obj, namespace, broad_access = _validate_full_transformer_token(
            token, _api=_api, required_base_repo=required_base_repo)
    except Exception as exc:
        # The validator deliberately raises only generic, token-free messages.
        # Still scrub both the exact candidate and common token forms in case a
        # future local seam regresses.
        error = str(exc).replace(str(token), '[redacted]')
        error = re.sub(r'\bhf_[A-Za-z0-9_-]{8,}\b', '[redacted]', error)
        return {
            **base, 'ok': False, 'code': 'invalid', 'severity': 'error',
            'error': error,
        }
    if broad_access:
        return {
            **base, 'ok': True, 'code': 'broad_access',
            'severity': 'warning', 'namespace': namespace, 'error': None,
            'warning': _BROAD_HF_TOKEN_WARNING,
        }
    return {
        **base, 'ok': True, 'code': 'ready', 'namespace': namespace,
        'severity': 'success', 'warning': None, 'error': None,
    }


def full_transformer_token_preflight(_api=None,
                                     required_base_repo=_KREA_BASE_REPO) -> dict:
    """Check the saved dense-training token without exposing its value."""
    return full_transformer_token_status(
        cfg.secret('HF_CLOUD_TOKEN'), _api=_api,
        required_base_repo=required_base_repo)


def _full_transformer_repo_name(run) -> str:
    """License-compliant, per-run model-repository segment.

    The Krea 2 Community License requires derivative model names to start with
    ``Krea``.  The database id makes this repository one-to-one with a cloud
    run; a retry is a new run and intentionally receives a new repository.
    """
    stem = re.sub(r'[^A-Za-z0-9._-]+', '-', str(run.run_name or '')).strip('-.')
    stem = stem[:48] or 'model'
    return f'Krea-2-full-{int(run.id)}-{stem}'


def _dense_base_repo_for(params) -> str:
    """The base a dense RUN was launched against, for the model card and for the
    licence source. A custom base lives in a private repository of the user's
    own (``base_repo_id``); the official lane resolves Raw or Turbo from the
    stamped variant, exactly like ``lt.official_base_repo``."""
    params = params or {}
    custom = str(params.get('base_repo_id') or '').strip()
    if custom:
        return custom
    variant = str(params.get('variant') or 'base').strip().lower()
    return _KREA_BASE_REPO if variant in ('base', 'raw') else lt.KREA_TURBO_BASE


def _krea_license_source(base_repo) -> str:
    """Which Krea repository to copy LICENSE.pdf from. Both official
    repositories carry the same Krea 2 Community License; a custom base is
    itself a Krea derivative, and its private repo carries no licence file, so
    Raw remains the source there."""
    return (base_repo if str(base_repo or '').lower() in _KREA_BASE_REPOS_LOWER
            else _KREA_BASE_REPO)


def _krea_license_bytes(api, base_repo) -> bytes:
    """LICENSE.pdf for the derivative, from whichever official Krea repository
    this token can actually read.

    The licence obligation does not depend on the variant, but the token's read
    scope does: a custom-base run needs no official scope at all (its weights
    live in the delivery namespace), so the token legitimately may be scoped to
    Turbo, to Raw, or — for a custom base — to neither. Trying the preferred
    source and then the other one turns a scope mismatch into a working launch
    instead of a refusal nobody could act on. When neither answers, the caller
    fails BEFORE any GPU is rented, which is the honest outcome: a Krea
    derivative must not reach the Hub without its licence."""
    preferred = _krea_license_source(base_repo)
    sources = [preferred] + [r for r in _KREA_BASE_REPOS if r != preferred]
    last = None
    for repo in sources:
        try:
            return _download_hf_file(api, repo, _KREA_LICENSE_FILENAME)
        except Exception as exc:               # noqa: BLE001 — try the next one
            last = exc
    raise last or RuntimeError('no Krea 2 licence source is readable')


def _full_transformer_readme(repo_id: str, base_repo=_KREA_BASE_REPO) -> str:
    model_name = repo_id.rsplit('/', 1)[-1]
    base_repo = str(base_repo or _KREA_BASE_REPO)
    return (
        '---\n'
        'license: other\n'
        'license_name: krea-2-community-license\n'
        f'license_link: {_KREA_LICENSE_LINK}\n'
        f'base_model: {base_repo}\n'
        'pipeline_tag: text-to-image\n'
        'tags:\n'
        '- krea-2\n'
        '- full-transformer\n'
        '- diffusers\n'
        '---\n\n'
        f'# {model_name}\n\n'
        f'{_KREA_REQUIRED_ATTRIBUTION}\n\n'
        'This repository contains a **modified derivative** of '
        f'`{base_repo}`, trained on a user-provided dataset. Its weights '
        'differ from the official model.\n\n'
        'This derivative is unofficial, is not an official Krea product, and '
        'is not endorsed by Krea. See `NOTICE` and `LICENSE.pdf` in this '
        'repository.\n')


def _download_hf_file(api, repo_id: str, filename: str) -> bytes:
    path = api.hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type='model')
    return Path(path).read_bytes()


def _full_transformer_compliance_files(api, repo_id: str,
                                       base_repo=_KREA_BASE_REPO) -> dict:
    """Return exact licence/notice/model-card bytes for a dense derivative."""
    return {
        _KREA_LICENSE_FILENAME: _krea_license_bytes(api, base_repo),
        'NOTICE': _KREA_NOTICE.encode('utf-8'),
        'README.md': _full_transformer_readme(repo_id, base_repo).encode('utf-8'),
    }


def _apply_full_transformer_compliance(api, repo_id: str, *, validate=True,
                                       base_repo=_KREA_BASE_REPO):
    """(Re)apply files ai-toolkit may overwrite, then optionally read back."""
    expected = _full_transformer_compliance_files(api, repo_id, base_repo)
    for filename, payload in expected.items():
        api.upload_file(
            path_or_fileobj=payload, path_in_repo=filename, repo_id=repo_id,
            repo_type='model',
            commit_message=f'Apply Krea 2 derivative compliance: {filename}')
    if validate:
        for filename, payload in expected.items():
            if _download_hf_file(api, repo_id, filename) != payload:
                raise RuntimeError(f'compliance validation failed for {filename}')


def _create_full_transformer_repo(run, token, _api=None,
                                  base_repo=_KREA_BASE_REPO) -> dict:
    """Create the private direct-delivery repository before a pod is rented.

    ``base_repo`` is what this run actually trains from: it decides which Krea
    repository the token must be able to read, which one LICENSE.pdf is copied
    from, and what the model card names as the base. Pinning Raw here would
    reject a token legitimately scoped to Turbo, and would print a base the run
    never used on a public-facing card.

    No exception text from the SDK is persisted: authentication/network
    errors can include request diagnostics, and secrets never belong in the
    run JSON or application log.
    """
    api, namespace, _broad_access = _validate_full_transformer_token(
        token, _api=_api,
        required_base_repo=(base_repo if str(base_repo or '').lower()
                            in _KREA_BASE_REPOS_LOWER else None))
    repo_id = f'{namespace}/{_full_transformer_repo_name(run)}'
    try:
        api.create_repo(repo_id=repo_id, repo_type='model', private=True,
                        exist_ok=False)
    except Exception:
        raise RuntimeError('could not create the private Hugging Face repository '
                           'for full_transformer delivery; verify HF_CLOUD_TOKEN has '
                           'write access and try again') from None
    hf_url = f'https://huggingface.co/{repo_id}'
    try:
        # Persist immediately: if this process dies between creation and the
        # completed launch params, the repository is still discoverable.
        _persist_artifact_state(
            run, 'preparing_metadata', hf_repo_id=repo_id, hf_url=hf_url,
            artifact_status_detail='Preparing Krea 2 licence and model card')
        _apply_full_transformer_compliance(api, repo_id, validate=True,
                                           base_repo=base_repo)
    except Exception:
        cleaned = False
        try:
            api.delete_repo(repo_id=repo_id, repo_type='model')
            cleaned = True
        except Exception:
            pass
        try:
            _persist_artifact_state(
                run, 'repository_preparation_failed', hf_repo_id=repo_id,
                hf_url=hf_url,
                artifact_status_detail=(
                    'Repository preparation failed; empty repository was deleted'
                    if cleaned else
                    'Repository preparation failed; repository cleanup must be checked'))
        except Exception:
            pass
        raise RuntimeError(
            'could not prepare the Krea 2 licence and model card in the private '
            'Hugging Face repository; no GPU was rented') from None
    return {'hf_repo_id': repo_id,
            'hf_url': hf_url}


def _updated_artifact_params(run, status, **extra) -> dict:
    """Build non-secret delivery metadata without committing it yet."""
    try:
        params = json.loads(run.train_params or '{}')
    except (TypeError, ValueError):
        params = {}
    if not isinstance(params, dict):
        params = {}
    params['artifact_status'] = status
    params.update(extra)
    return params


def _persist_artifact_state(run, status, **extra) -> dict:
    """Persist non-secret delivery metadata and return the updated params."""
    params = _updated_artifact_params(run, status, **extra)
    _set(run, train_params=json.dumps(params))
    return params


def _export_full_transformer_fp8(run, remote) -> dict:
    """Post-training fp8 export, stamped on the run. NEVER raises.

    Disabled by the dataset's own setting, or by config
    (``cloud.full_transformer.fp8_export``) for an install that would rather not
    spend the extra pod minutes. A disabled export leaves NO status, so the UI
    shows nothing rather than an unexplained absence.
    """
    from . import dense_fp8_delivery
    dense = ((cfg.get('cloud') or {}).get('full_transformer') or {})
    if dense.get('fp8_export') is False:
        return {'state': 'skipped', 'detail': 'disabled in configuration'}
    ds = None
    try:
        from . import face_dataset_service as fds
        ds = fds.get_dataset(cfg.LOCAL_USER, run.dataset_id)
    except Exception:
        ds = None
    enabled = (lt.dense_fp8_export_enabled(ds) if ds is not None
               else bool(_run_param(run, 'fp8_export') is not False))
    if not enabled:
        return {'state': 'skipped', 'detail': 'fp8 export is off for this dataset'}
    keep_bf16 = (lt.dense_keep_bf16_master(ds) if ds is not None
                 else _run_param(run, 'keep_bf16_master') is not False)
    # A run that brings its files home exports the twin ON the pod and fetches
    # it like the master: pushing a second ~10 GB object through a private
    # quota to download it back would be paying twice for a file that is
    # regenerated from the master in seconds.
    upload = not _dense_delivers_local(run)
    outcome = dense_fp8_delivery.run_pod_fp8_export(
        run, remote,
        instance_id=run.vast_instance_id,
        repo_id=_run_param(run, 'hf_repo_id'),
        hf_token=cfg.secret('HF_CLOUD_TOKEN'),
        keep_bf16=keep_bf16, upload=upload,
        budget_seconds=int(dense.get('fp8_export_budget_seconds')
                           or dense_fp8_delivery.DEFAULT_BUDGET_SECONDS),
        on_state=lambda detail: _set_soft(run, phase_detail=detail[:500]))
    result = outcome.get('result') or {}
    try:
        _persist_run_params(
            run,
            fp8_export_status=outcome['state'],
            fp8_export_detail=outcome['detail'],
            fp8_keep_bf16=bool(keep_bf16),
            fp8_weight_filename=(os.path.basename(str(result.get('path')))
                                 if result.get('path') else None),
            fp8_size_bytes=result.get('bytes_after'))
    except Exception:
        logger.debug('could not stamp the fp8 export state of run %s', run.id)
    return outcome


def _persist_run_params(run, **extra) -> dict:
    """Merge non-secret keys into train_params and commit. Same shape as
    _persist_artifact_state, without claiming an artifact status."""
    try:
        params = json.loads(run.train_params or '{}')
    except (TypeError, ValueError):
        params = {}
    if not isinstance(params, dict):
        params = {}
    params.update(extra)
    _set(run, train_params=json.dumps(params))
    return params


def _metadata_value(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _full_transformer_weight_proof(sibling) -> dict | None:
    """Return a non-secret integrity proof from Hub metadata, never file bytes."""
    direct_size = _metadata_value(sibling, 'size')
    lfs = _metadata_value(sibling, 'lfs')
    lfs_size = _metadata_value(lfs, 'size') if lfs is not None else None
    sizes = [value for value in (direct_size, lfs_size)
             if isinstance(value, int) and not isinstance(value, bool)]
    if not sizes or any(value < _FULL_TRANSFORMER_MIN_WEIGHT_BYTES
                        for value in sizes):
        return None
    if len(set(sizes)) > 1:
        return None

    sha256 = (_metadata_value(lfs, 'sha256') if lfs is not None else None)
    if not sha256 and lfs is not None:
        sha256 = _metadata_value(lfs, 'oid')
    sha256 = str(sha256 or '').strip().lower()
    if not re.fullmatch(r'[0-9a-f]{64}', sha256):
        sha256 = None

    blob_id = str(_metadata_value(sibling, 'blob_id') or '').strip().lower()
    if not re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', blob_id):
        blob_id = None
    if not sha256 and not blob_id:
        return None
    return {
        'size_bytes': sizes[0],
        'sha256': sha256,
        'blob_id': blob_id,
        'metadata_source': 'huggingface_repo_info_files_metadata',
    }


def _verify_full_transformer_artifact(run, _api=None) -> str:
    """Verify direct HF delivery and return its explicit persisted status.

    ``available`` is written only after the private repo exposes a credibly
    sized ``.safetensors`` object plus an immutable Hub blob/LFS identifier.
    ``repo_info(files_metadata=True)`` provides that proof without downloading
    the ~26 GB checkpoint.  A network/API failure remains distinct from a repo
    that answered successfully but contains no intact dense checkpoint.
    """
    if not _is_full_transformer_run(run):
        return 'not_applicable'
    repo_id = _run_param(run, 'hf_repo_id')
    if not repo_id:
        _persist_artifact_state(
            run, 'missing', artifact_status_detail='Hugging Face repository was not recorded')
        return 'missing'
    token = cfg.secret('HF_CLOUD_TOKEN')
    if not token and _api is None:
        _persist_artifact_state(
            run, 'verification_pending',
            artifact_status_detail='HF_CLOUD_TOKEN is unavailable; delivery is not verified')
        return 'verification_pending'
    try:
        api = _api or _make_hf_api(token)
        info = api.repo_info(
            repo_id=repo_id, repo_type='model', files_metadata=True)
        siblings = _metadata_value(info, 'siblings')
        if not isinstance(siblings, (list, tuple)):
            raise RuntimeError('Hugging Face did not return file metadata')
    except Exception:
        # Do not log the SDK exception: request diagnostics must never risk
        # echoing an authorization header.  The explicit state is actionable.
        logger.warning('run %s: Hugging Face artifact verification unavailable', run.id)
        _persist_artifact_state(
            run, 'verification_pending',
            artifact_status_detail='Hugging Face verification is temporarily unavailable')
        return 'verification_pending'
    expected_prefix = (run.job_name if str(run.job_name or '').startswith('Krea')
                       else 'Krea')
    matching = []
    for sibling in siblings:
        path = str(_metadata_value(sibling, 'rfilename') or '')
        if (path.lower().endswith('.safetensors')
                and Path(path).name.startswith(expected_prefix)):
            matching.append((path, _full_transformer_weight_proof(sibling)))
    valid = sorted((path, proof) for path, proof in matching if proof is not None)
    checked_at = datetime.utcnow().isoformat()
    # Which of them is "the model" is dense_weights' single rule, shared with
    # every lane that acts on this file. Sorting and taking the last one used to
    # land on `…_000002750.safetensors` (`.` sorts before `_`), i.e. a step
    # snapshot, while the quantizer offered on the same card took the final —
    # the card named one file and the button next to it would have taken another.
    if not valid:
        reason = ('has no matching dense checkpoint' if not matching else
                  'has only empty, truncated, or unverifiable matching checkpoints')
        _persist_artifact_state(
            run, 'missing',
            artifact_status_detail=(
                f'Repository {reason} ({expected_prefix}*.safetensors)'),
            delivery_last_checked_at=checked_at)
        return 'missing'
    proofs = dict(valid)
    chosen = dense_weights.pick_master(
        [(path, (proof or {}).get('size_bytes') or 0) for path, proof in valid])
    weight_path = chosen or valid[-1][0]
    proof = proofs[weight_path]
    try:
        # ai-toolkit writes its own README while pushing. Reapply and read back
        # every compliance file before the result can become available — with
        # the base this RUN used, not a constant.
        _apply_full_transformer_compliance(
            api, repo_id, validate=True,
            base_repo=_dense_base_repo_for({
                'variant': _run_param(run, 'variant'),
                'base_repo_id': _run_param(run, 'base_repo_id')}))
    except Exception:
        logger.warning('run %s: Krea repository metadata verification unavailable',
                       run.id)
        _persist_artifact_state(
            run, 'verification_pending',
            artifact_status_detail=(
                'Dense checkpoint exists, but Krea licence/model-card metadata '
                'could not be reapplied and verified'),
            delivery_last_checked_at=checked_at)
        return 'verification_pending'
    verified_at = datetime.utcnow().isoformat()
    _persist_artifact_state(
        run, 'available', hf_weight_filename=weight_path,
        hf_artifact_proof=proof,
        verified_at=verified_at, artifact_verified_at=verified_at,
        delivery_last_checked_at=verified_at,
        artifact_status_detail='Dense checkpoint and compliance metadata verified')
    return 'available'


def _verify_full_transformer_artifact_with_retries(run, _api=None) -> str:
    """Bound transient HF/metadata verification without ever failing open."""
    dense = ((cfg.get('cloud') or {}).get('full_transformer') or {})
    attempts = max(1, int(dense.get('verification_attempts') or 3))
    delay = max(0, int(dense.get('verification_retry_seconds') or 0))
    state = 'verification_pending'
    for attempt in range(1, attempts + 1):
        try:
            state = _verify_full_transformer_artifact(run, _api=_api)
        except Exception:
            # Persistence/SDK edge cases must remain fail-closed, and the
            # exception is intentionally not interpolated (it may contain
            # request diagnostics from an authenticated call).
            logger.warning('run %s: dense artifact verification attempt %s/%s failed',
                           run.id, attempt, attempts)
            state = 'verification_pending'
        if state != 'verification_pending':
            return state
        if attempt < attempts and delay:
            _sleep(delay)
    _persist_artifact_state(
        run, 'verification_pending',
        artifact_status_detail=(
            f'Hugging Face delivery/compliance verification remained unavailable '
            f'after {attempts} attempts'))
    return state


def _assert_launch_guardrails(dataset_id, fam, dataset_table=crd.FACE,
                              allow_parallel_run=False):
    """Raise when a cloud launch cannot reserve an active slot.

    Callers may use this once as a cheap fast-fail before expensive preflight,
    but the authoritative call must happen while ``_launch_reservation_lock``
    is held and immediately before inserting the ``preparing`` row.

    `dataset_table` is part of the per-dataset uniqueness key, not decoration:
    face and video datasets share one integer space, so without it an active
    video run of id 3 would refuse every launch on FACE dataset 3 — a button
    locked by a run on someone else's data, with no explanation available. The
    fleet-wide limit and the budget below are deliberately NOT scoped: they are
    about the account's pods and its money, which one lane cannot claim.
    """
    actives = get_active_runs()
    limit = max(1, int((cfg.get('cloud.max_concurrent_runs') or 1)))
    # Uniqueness is per (dataset, table, family): a zimage run and a krea run may
    # train the same dataset in parallel. An active run whose family is
    # unknown (pre-feature row) blocks every family of its dataset, out of
    # caution — and so does one whose TABLE cannot be read, for the same
    # reason. `crd.owns` answers False there, which is right for a route
    # deciding whether to serve a file and wrong for a guard deciding whether
    # to spend money, so this one asks the question itself. Both are
    # AMBIGUOUS siblings, not same-family ones: allow_parallel_run answers
    # "yes, another <fam> run" and cannot cover a case where the guard does
    # not even know what it would be confirming past — those stay hard blocks.
    def _sibling_kind(r):
        """None (unrelated), 'ambiguous' (unreadable table or unknown family —
        never waivable) or 'same_family' (the only confirmable case) /
        'other_family' (no block at all)."""
        if int(r.dataset_id or 0) != int(dataset_id):
            return None
        try:
            same_table = crd.table_of(r) == dataset_table
        except ValueError:
            return 'ambiguous'   # unreadable table -> block, never spend twice
        if not same_table:
            return None
        rfam = _run_family(r)
        if rfam is None:
            return 'ambiguous'   # unknown family -> block, never spend twice
        return 'same_family' if rfam == fam else None

    ambiguous = next((r for r in actives if _sibling_kind(r) == 'ambiguous'),
                     None)
    if ambiguous is not None:
        raise RuntimeError(
            'this dataset already has an active cloud run that cannot be '
            'attributed to a family or table — refusing to rent a second '
            'pod on ambiguity')
    sibling = next((r for r in actives if _sibling_kind(r) == 'same_family'),
                   None)
    if sibling is not None and not allow_parallel_run:
        # Confirmable (PARALLEL_RUN: contract, mirrors MISMATCH_CAPTION:):
        # the UI strips the marker, window.confirm IS the answer, and the
        # retry carries allow_parallel_run.
        raise RuntimeError(
            f'PARALLEL_RUN: this dataset already has an active {fam} cloud '
            f'run (#{sibling.id}) — launching another one rents a second '
            'pod, billed separately. Launch anyway?')
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


def _run_training_mode(run) -> str:
    """Frozen execution mode for a run; legacy/corrupt rows stay LoRA-safe."""
    return ('full_transformer'
            if _run_param(run, 'training_mode') == 'full_transformer'
            else 'lora')


def _is_full_transformer_run(run) -> bool:
    return _run_training_mode(run) == 'full_transformer'


def _dense_delivery(run) -> str:
    """Where THIS run's dense artifact goes: 'local', 'hub' or 'both'.

    Read from the run's OWN stamp, never from the live config — the delivery is
    frozen at launch for the same reason the family and the settings snapshot
    are (see _RunConfigDataset): the monitor decides what to do with a ~26 GB
    file hours later, and a setting changed in between must not retarget a run
    already in flight. An unstamped row predates the choice and delivered to
    Hugging Face only, which is exactly what dense_local_delivery.run_mode
    answers for it."""
    try:
        params = json.loads(run.train_params or '{}')
    except (ValueError, TypeError):
        params = {}
    return dld.run_mode(params if isinstance(params, dict) else {})


def _dense_delivers_local(run) -> bool:
    return _is_full_transformer_run(run) and dld.delivers_local(_dense_delivery(run))


def _dense_delivers_hub(run) -> bool:
    return _is_full_transformer_run(run) and dld.delivers_hub(_dense_delivery(run))


# Where a full model still IS. STORED nowhere — computed — but published in the
# lineage payload, so the strings are part of the frontend contract.
DENSE_ON_DISK = 'local'
DENSE_ON_HUB = 'hub'
DENSE_GONE = 'none'


def dense_artifact_state(run) -> str:
    """Where THIS run's full model still lives: on this disk, on Hugging Face,
    or nowhere.

    WHY THIS CANNOT BE ``checkpoint_local_path``
    --------------------------------------------
    That column belongs to the LoRA lane: the cloud sync writes it for the
    adapter it downloaded. A dense run only ever fills it when the LOCAL
    delivery lands — a feature younger than the dense lane itself — so every run
    that delivered to Hugging Face only, which is every dense run trained before
    it, reads as holding nothing.

    The canvas believed that literally. It dimmed those cards, badged them
    ``gone``, and offered "Remove this run" under the words "No checkpoints left
    on disk" — for a model sitting in a private repository that cost eight hours
    of GPU. Removing one discards the lineage, the notes, the version and the
    only recorded pointer to that repository.

    So presence is asked of BOTH addresses a full model can have, and the
    Hugging Face half is deliberately generous: only an explicitly verified
    ``missing`` counts as absent. An upload still pending, or a verification that
    could not run, is not proof of a lost model — and the two mistakes do not
    cost the same. A button that fails to appear is an annoyance; a training
    thrown away is not recoverable.
    """
    if not _is_full_transformer_run(run):
        return DENSE_GONE
    try:
        if run_checkpoint_files(run):
            return DENSE_ON_DISK
    except Exception:                                   # noqa: BLE001
        # A scan that cannot run is not an absence. Fall through to the Hub
        # answer rather than report a model gone on the strength of an OSError.
        logger.debug('dense artifact scan failed', exc_info=True)
    if not _run_param(run, 'hf_repo_id'):
        return DENSE_GONE
    return (DENSE_GONE if _run_param(run, 'artifact_status') == 'missing'
            else DENSE_ON_HUB)


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
                 train_settings_snapshot=_UNSET, train_slider_snapshot=_UNSET,
                 training_mode='lora'):
        self._ds = ds
        self._train_type = train_type
        self._train_variant = train_variant
        self._train_base_model = train_base_model
        self._train_settings_snapshot = train_settings_snapshot
        self._train_slider_snapshot = train_slider_snapshot
        self._training_mode = lt.normalize_training_mode(training_mode)

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
    def training_mode(self):
        # Dense-vs-LoRA changes the tensors being optimized and the artifact
        # type.  It is therefore provenance, not a mutable dataset preference.
        return self._training_mode

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
    mode = params.get('training_mode', 'lora')
    return _RunConfigDataset(ds, fam, var, base, advanced, slider, mode)


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


def latest_run_for(dataset_id, train_type=None, dataset_table=crd.FACE):
    """Newest run of the dataset; with train_type, the newest run OF THAT
    FAMILY. Falls back to the plain newest when none matches (or the filter
    is absent) so rows without a stamped family stay reachable.

    Scoped to ONE table. The progress and sample routes call this with nothing
    but the URL's integer and no ownership check of their own, so an unscoped
    query would let a face dataset's progress endpoint return a video run's
    phase, cost and preview files the moment a video run of the same id became
    the newest — the single most exposed mis-attribution in this file.

    A row whose table cannot be read is EXCLUDED rather than raised on: this
    feeds a progress poll, and a 500 on every poll would be a worse answer than
    omitting one unattributable run. The launch guard makes the opposite choice,
    deliberately — there, ambiguity must block rather than spend."""
    rows = [r for r in CloudTrainingRun.query.filter_by(dataset_id=dataset_id)
            .order_by(CloudTrainingRun.id.desc()).all()
            if crd.owns(r, dataset_id, dataset_table)]
    newest = rows[0] if rows else None
    if not train_type:
        return newest
    fam = fds.normalize_train_type(train_type)
    for r in rows:
        if _run_family(r) == fam:
            return r
    return newest


def run_for(dataset_id, run_id=None, train_type=None, dataset_table=crd.FACE):
    """ONE resolution point for "which run of this dataset".

    With run_id: that run, only if the (id, table) ownership holds — the same
    barrier the checkpoint download applies, because face and video datasets
    share one integer space. None on a miss, NEVER a fallback to the newest:
    a poll quietly answering for a different run is the mis-attribution
    latest_run_for's docstring warns about. Without run_id: exactly
    latest_run_for, so legacy callers keep their behaviour."""
    if run_id is not None:
        run = db.session.get(CloudTrainingRun, int(run_id))
        if run is None or not crd.owns(run, dataset_id, dataset_table):
            return None
        return run
    return latest_run_for(dataset_id, train_type, dataset_table=dataset_table)


# A monitor state write that loses a race for the SQLite write lock must not
# kill a run that is burning rented GPU time. The DB is opened WAL with
# busy_timeout=5000 (app/__init__.py), so a writer only ever sees 'database is
# locked' when another writer held the lock for more than five seconds — a
# captioning batch, a bank import, a big dataset write. That happened on
# 2026-07-26: two monitors had just created their job on the pod and died on
# `_set(status='training')` with `sqlite3.OperationalError: database is locked`,
# three minutes into runs that then sat abandoned for an hour of paid 5090 time
# (runs #106 and #107). The lock is transient by nature, so the commit is
# retried instead of being fatal.
_COMMIT_RETRIES = 4
_COMMIT_RETRY_BASE_SECONDS = 0.5


def _is_locked_error(exc):
    return 'database is locked' in str(exc).lower()


def _set(run, **fields):
    """Write monitor state, surviving a transient SQLite write-lock loss.

    A failed commit leaves the session with a pending rollback and — once
    rolled back — the instance reverted to its stored values, so the fields are
    re-applied on every attempt rather than set once up front.

    Rolling back is therefore the FIRST thing the failure path does, before
    anything reads the instance and before the raise. Until a rollback happens,
    the session refuses every operation with PendingRollbackError, and a
    persistent instance whose attributes were expired by the previous commit
    cannot even be read: touching `run.id` fires a lazy load, which needs the
    session, which raises. That is not theoretical — it defeated this very
    retry loop on 2026-07-28. The lock was hit, the failure path formatted its
    log line, reading `run.id` raised PendingRollbackError out of _set, and the
    monitor thread died in the exact way the retry was written to prevent. Run
    #121 then sat at 'training' with no error and a live rented 5090, because
    the caller's recovery path (_finish) inherited the same poisoned session and
    failed too. The run id is read up front, off the healthy session, so the
    log line cannot resurrect that failure."""
    run_id = getattr(run, 'id', '?')
    for attempt in range(_COMMIT_RETRIES):
        for k, v in fields.items():
            setattr(run, k, v)
        run.updated_at = datetime.utcnow()
        try:
            db.session.commit()
            return
        except Exception as e:                    # noqa: BLE001 - re-raised below
            db.session.rollback()
            if attempt == _COMMIT_RETRIES - 1 or not _is_locked_error(e):
                raise
            logger.warning('run %s: SQLite write lock busy (attempt %s/%s) — '
                           'retrying the state write', run_id,
                           attempt + 1, _COMMIT_RETRIES)
            _sleep(_COMMIT_RETRY_BASE_SECONDS * (2 ** attempt))


def _set_soft(run, **fields) -> bool:
    """Write purely informational monitor state; never fail the run over it.

    ``_set`` is the authoritative writer and must keep raising: a lost status
    transition would leave a rented pod misrepresented. The per-poll progress
    heartbeat is different — it only refreshes cosmetic ``phase_detail`` text.
    On 2026-08-01 a local write lock outlived the retry budget while run #137
    was training normally on a rented 5090; the heartbeat commit raised out of
    the monitor thread and the run was recorded as failed with
    'database is locked', GPU time and all. A cosmetic refresh is allowed to be
    skipped; the next poll writes the same text a few seconds later.

    Returns whether the write landed, so callers can log the miss.
    """
    try:
        _set(run, **fields)
        return True
    except Exception as e:                        # noqa: BLE001 - deliberate
        if not _is_locked_error(e):
            raise
        logger.warning('run %s: progress heartbeat skipped — the database '
                       'stayed write-locked; training is unaffected',
                       getattr(run, 'id', '?'))
        return False


def _reconcile_before_launch(app):
    """Seam around the launch-time reconcile_orphans() call (defined below).
    A thin indirection rather than calling reconcile_orphans directly so
    tests can no-op launch's reconcile call without also neutering tests
    that exercise reconcile_orphans() itself -- both are the same module-level
    name, so patching that name would silence both call sites at once."""
    reconcile_orphans(app)


def _video_lane(run):
    """The video lane's own relauncher for this run, or None when it is a face
    run and this module's own path applies.

    `retry_cloud_run` and `continue_cloud_run` both rebuild their arguments from
    a run's stamped params and call `launch_cloud_training`, which resolves
    `dataset_id` as a FACE dataset. Handed a video run they would either 404 on
    a dataset that is not there or — on a colliding id — launch a face training
    on someone else's data and charge for it. Both used to refuse for exactly
    that reason; what was actually missing was the video-side rebuild, which now
    exists, so they DISPATCH instead.

    `crd.is_video` still raises on a run naming a table this build does not know
    — that refusal was never about the video lane, it is about a row that cannot
    say which dataset it trained, and guessing there is the silent
    mis-attribution the whole column exists to prevent.

    Imported inside the function: `cloud_video_training` imports this module at
    its top, and the video lane reusing the shared monitor is the point."""
    if not crd.is_video(run):
        return None
    from . import cloud_video_training
    return cloud_video_training


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
    video = _video_lane(run)
    if video:
        return video.retry_cloud_video_run(user_id, run.id)
    if run.status != 'error':
        raise ValueError('only a failed run can be retried')
    try:
        p = json.loads(run.train_params or '{}')
    except ValueError:
        p = {}
    if not isinstance(p, dict):
        p = {}
    if (p.get('training_mode') == 'full_transformer'
            and p.get('resume_ckpt_path')
            and not os.path.isfile(p['resume_ckpt_path'])):
        # A dense retry replays its seed verbatim, so the seed has to still be
        # there. This used to refuse EVERY local dense seed, because the pod's
        # upload seam could not carry 26 GB at all; now the only thing that
        # stops a replay is the file being gone — and saying "gone" when it is
        # merely large would be the same missing choice this lane just gained.
        raise ValueError('the full model this run resumed from is no longer on '
                         'this computer; continue it from its Hugging Face copy '
                         'instead')
    _assert_recipe_replayable(p, 'retry')
    snapshot = p.get(_TRAIN_SETTINGS_SNAPSHOT, _UNSET)
    if p.get('resume_ckpt_path'):
        # A retry with a seed is still a continuation: its checkpoint topology
        # comes from the original parent record, not a modern default inferred
        # from this failed child's raw snapshot.
        from . import checkpoint_registry
        topology = checkpoint_registry.network_geometry(_resume_parent_record(run))
        snapshot = _resume_snapshot_with_recorded_topology(
            user_id, run.dataset_id, snapshot, topology)
    return launch_cloud_training(
        user_id, run.dataset_id,
        steps=p.get('steps'),
        base_model=p.get('base_model', ''),
        variant=p.get('variant'),
        train_type=p.get('train_type'),
        training_mode=p.get('training_mode', 'lora'),
        masked=p.get('masked', True),
        **_confirmation_flags(p),
        gpu_name=p.get('requested_gpu'),
        resume_ckpt_path=p.get('resume_ckpt_path'),
        # A dense continuation replays its Hub seed verbatim: the retry is the
        # same run, on a fresh pod, and it must resume from the same weights.
        resume_hf=({'repo_id': p.get('resume_hf_repo_id'),
                    'filename': p.get('resume_hf_filename')}
                   if p.get('resume_hf_repo_id') else None),
        resume_step=p.get('resume_step'),
        train_settings_snapshot=snapshot,
        train_slider_snapshot=p.get(_TRAIN_SLIDER_SNAPSHOT, _UNSET))


_CLOUD_FULL_STATE_REASON = (
    'This cloud image does not run the LDS state bridge; only the LoRA weights '
    'were harvested. Optimizer, scheduler, RNG and dataloader state are unavailable.')


def _cloud_resume_state() -> dict:
    """Truthful capability stamp for a checkpoint produced by today's pod image."""
    return {
        'bundle_id': None,
        'status': 'unsupported',
        'integrity': 'unchecked',
        'state_level': 'weights',
        'size_bytes': 0,
        'capabilities': {'weights': True, 'exact': False},
        'reason': _CLOUD_FULL_STATE_REASON,
    }


def _run_staging_checkpoints(run) -> list:
    """This run's HARVESTED checkpoints that are still on disk (NOT the trash —
    a trashed save is moved out of its folder): list of {'filename', 'step',
    'path'}, step-sorted ascending. Mirrors cloud_checkpoints' step extraction so
    'continue' resumes from the exact same checkpoint the hub lists. The
    unsuffixed FINAL save (no _<step> suffix) is the run's target step count.

    Reads the durable store (falling back to the legacy staging dir) — kept
    under its historical name because "continue from an earlier epoch" is wired
    to it everywhere; what changed is only WHERE the files live."""
    if _is_full_transformer_run(run):
        return []
    saves = run_checkpoint_files(run)
    if not saves:
        return []
    target = int(_run_param(run, 'steps') or 0)
    out = []
    for name, path in saves.items():
        step, _stage = video_training.split_checkpoint_name(name)
        out.append({'filename': name,
                    'step': step if step is not None else target,
                    'path': path,
                    'resume_state': _cloud_resume_state()})
    # step asc; a suffixed save wins ties over the unsuffixed final (deterministic).
    out.sort(key=lambda e: (
        e['step'],
        video_training.split_checkpoint_name(e['filename'])[0] is not None,
        e['filename']))
    return out


def _dense_resume_candidates(run) -> list:
    """What a FULL-MODEL run can be continued from, and BY WHICH ROAD.

    TWO sources now, and the difference between them is the point:

    * ``'hub'`` — the Hugging Face copy. The pod pulls it over a datacenter
      link, so it is minutes; it needs that copy to exist, which means the run
      was delivered with a Hub leg, and it means the weights travel through a
      third party.
    * ``'local'`` — the master sitting on this computer. Nothing outside the
      machine is involved, and it costs the user's uplink: ~26 GB of upload
      while a rented GPU is being billed for doing nothing.

    The local road used to be absent from this list on purpose, because the
    pod's only write seam built its whole request in memory and 26 GB could not
    survive that. That is fixed (``pod_checkpoint_push``), so hiding the road is
    no longer honesty, it is a missing choice — and it was the ONLY road for a
    run delivered without a Hub leg.

    Same shape as _run_staging_checkpoints (step / filename / resume_state) so
    the selection logic in continue_cloud_run stays ONE piece of code. Sorted
    step-ascending with 'hub' last on a tie, which is what keeps the historical
    default: continue_cloud_run's no-argument pick stays the Hub copy."""
    if not _is_full_transformer_run(run):
        return []
    out = []
    for name, path in (run_checkpoint_files(run) or {}).items():
        if dld.is_fp8_name(name):
            # A quantized twin cannot be trained further — its weights are fp8
            # with per-tensor scales, not the bf16 the trainer loads.
            continue
        try:
            size = os.path.getsize(path)
        except OSError:
            continue                     # vanished between listing and stat
        out.append({'step': int(dld.step_of(
                        name, default=int(_run_param(run, 'steps') or 0)) or 0),
                    'filename': name, 'source': 'local', 'path': path,
                    'size_bytes': size, 'repo_id': None, 'hf_filename': None,
                    'resume_state': _cloud_resume_state()})
    repo_id = _run_param(run, 'hf_repo_id')
    filename = _run_param(run, 'hf_weight_filename')
    # Unverified is not a source: seeding a checkpoint that may be truncated
    # would spend a fresh pod to train from garbage.
    if repo_id and filename and _run_param(run, 'artifact_status') == 'available':
        name = os.path.basename(str(filename))
        if not dld.is_fp8_name(name):
            step = dld.step_of(name, default=int(_run_param(run, 'steps') or 0))
            # The size comes from the verification proof, which measured the
            # file ON the Hub. A local twin of the same name would be a good
            # guess and a bad fact — the two can differ, and this number ends
            # up in a forecast the user is asked to trust.
            proof = _run_param(run, 'hf_artifact_proof')
            out.append({'step': int(step or 0), 'filename': name,
                        'source': 'hub', 'repo_id': str(repo_id),
                        'hf_filename': str(filename), 'path': None,
                        'size_bytes': int((proof or {}).get('size_bytes') or 0)
                        if isinstance(proof, dict) else 0,
                        'resume_state': _cloud_resume_state()})
    out.sort(key=lambda c: (c['step'], c['source'] == 'hub'))
    return out


def dense_resume_transport(candidates, transport=None) -> str:
    """Which road a dense continue takes: ``'hub'`` or ``'local'``.

    ``transport`` is what the user picked ('hub' / 'direct'), or None for "no
    opinion". No opinion means the Hugging Face copy WHENEVER ONE EXISTS, at any
    step — never "whichever checkpoint is newest". Those two rules differ when
    the local disk holds a later save than the Hub does, and picking the newest
    would silently switch the user onto a road that takes hours and bills a GPU
    the whole time. A lane that expensive is a decision, not a default."""
    want = {'hub': 'hub', 'direct': 'local', 'local': 'local'}.get(
        str(transport or '').strip().lower())
    if want:
        return want
    return 'hub' if any(c.get('source') == 'hub' for c in candidates) else 'local'


def dense_resume_plan(user_id, run_id, from_step=None) -> dict:
    """The two roads a full model can take back to a pod, PRICED — the answer
    behind the ▶ Continue dialog's choice, before anything is rented.

    Always returns; a road that cannot be taken comes back with
    ``available: False`` and a ``reason`` that names what would make it
    available. A greyed-out option with no explanation is how a user ends up
    reading source code to find out that a trade-off exists at all.

    The GPU cost is the number this whole panel exists for. A pod is rented and
    billed while it is being handed its checkpoint, so the road that takes three
    hours costs three hours of a graphics card doing nothing — and until this
    existed, nothing in the app said so.
    """
    from . import pod_transfer_plan as ptp
    run = db.session.get(CloudTrainingRun, int(run_id))
    if not run:
        raise ValueError('unknown cloud run')
    candidates = _dense_resume_candidates(run)
    if from_step is not None:
        try:
            want = int(from_step)
        except (TypeError, ValueError):
            raise ValueError('from_step must be an integer step')
        candidates = [c for c in candidates if c['step'] == want]
    # What the hour will cost. The source run's own rate is the best evidence
    # there is — the child asks for the same GPU class — and the configured cap
    # is the honest worst case when there is none.
    price = float(run.price_per_hour or 0)
    price_source = 'this run'
    if not price:
        price = float((cfg.get('cloud') or {}).get('max_price_per_hour') or 0.80)
        price_source = 'the price cap in Settings'

    def _lane(source):
        picked = [c for c in candidates if c.get('source') == source]
        return picked[-1] if picked else None

    local = _lane('local')
    hub = _lane('hub')
    delivery = dld.run_mode(_run_params(run))
    options = []

    # Does the repository still ANSWER? `_dense_resume_candidates` can only read
    # the registry, and the registry's `artifact_status` is stamped once at
    # delivery and never revisited. A repository its owner deleted last night
    # therefore still reads 'available' — so this lane was offered, priced, and
    # given an ETA, and choosing it RENTED A POD that then took a 404. The road
    # has to be measured here, not remembered.
    #
    # Only a proven 'gone' closes it. `unknown` — no token, offline, a 5xx —
    # leaves it OPEN: refusing someone's fast road because their Wi-Fi dropped
    # would be a worse failure than the one being fixed, and hub_presence is
    # built to never say 'gone' without proof.
    presence = None
    if hub:
        from . import hub_presence
        presence = hub_presence.check(hub['repo_id'])
        if presence.get('state') == hub_presence.GONE:
            hub = None

    if hub:
        est = ptp.estimate_hub(hub.get('size_bytes') or 0, price)
        options.append({**est, 'transport': 'hub', 'available': True,
                        'filename': hub['filename'], 'step': hub['step'],
                        'repo_id': hub['repo_id'], 'reason': None,
                        'presence': (presence or {}).get('state')})
    else:
        # Say WHICH reason it is, in the order that makes each answer TRUE
        # rather than merely first. "Unavailable" alone sends the user to change
        # a setting that was never the problem; "not verified" for a run that
        # never had a copy sends them to re-verify something that does not
        # exist; and "gone" for a run whose only Hub file is an fp8 twin blames
        # a deletion that never happened.
        status = _run_param(run, 'artifact_status')
        name = os.path.basename(str(_run_param(run, 'hf_weight_filename') or ''))
        has_copy = bool(_run_param(run, 'hf_repo_id') and name)
        if not dld.delivers_hub(delivery):
            reason = ('this run was delivered to this computer only, so no '
                      'Hugging Face copy of it was ever made. Set the delivery '
                      'to "This computer + Hugging Face" (Settings ▸ Training) '
                      'and future runs keep this road open.')
        elif not has_copy:
            reason = ('this run has no Hugging Face copy on record — it was '
                      'never delivered there, or the copy is gone.')
        elif status != 'available':
            reason = ('the Hugging Face copy of this run is not verified '
                      f"({status or 'unknown'}) — resuming from a file that may "
                      'be truncated would spend a pod training from garbage.')
        elif presence is not None:
            # Measured gone, just now. This is the ONLY branch entitled to say
            # a deletion happened, because it is the only one that looked.
            reason = ('the Hugging Face copy of this run is gone — checked just '
                      'now, the repository does not answer. Send the copy on '
                      'this computer instead, if there is one.')
        elif dld.is_fp8_name(name):
            reason = ('the only Hugging Face file of this run is its quantized '
                      'fp8 twin, which cannot be trained further — its weights '
                      'are fp8 with per-tensor scales, not the bf16 the trainer '
                      'loads.')
        else:
            reason = ('this run has no full-precision Hugging Face checkpoint '
                      'to resume from.')
        options.append({'transport': 'hub', 'available': False,
                        'reason': reason, 'bytes': 0, 'seconds': 0,
                        'gpu_cost': 0, 'price_per_hour': round(price, 4),
                        'presence': (presence or {}).get('state')})

    if local:
        est = ptp.estimate_direct(local.get('size_bytes') or 0, price)
        options.append({**est, 'transport': 'direct', 'available': True,
                        'filename': local['filename'], 'step': local['step'],
                        'repo_id': None, 'reason': None})
    else:
        options.append({
            'transport': 'direct', 'available': False, 'bytes': 0,
            'seconds': 0, 'gpu_cost': 0, 'price_per_hour': round(price, 4),
            'reason': ('this computer no longer holds a full-precision copy of '
                       'this run. A quantized (fp8) twin cannot be trained '
                       'further, so it does not count.')})

    return {
        'run_id': run.id,
        'default_transport': ('hub' if dense_resume_transport(candidates) == 'hub'
                              else 'direct'),
        'price_per_hour': round(price, 4),
        'price_source': price_source,
        'delivery': delivery,
        'options': options,
    }


def _run_params(run) -> dict:
    try:
        parsed = json.loads(run.train_params or '{}')
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


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


def _cloud_run_record(run):
    """The provenance row stamped for one cloud run, if it still exists."""
    from ..models import TrainingRunRecord
    return (TrainingRunRecord.query
            .filter_by(cloud_run_id=run.id)
            .order_by(TrainingRunRecord.id.desc()).first())


def _resume_parent_record(run):
    """The record that made a seeded checkpoint replayed by a retry.

    A normal cloud→cloud continuation records this edge on its child.  Do not
    trust that child's own emitted settings as evidence for an older checkpoint:
    before full-rank provenance existed it may already contain a forced modern
    default.  Without a parent edge, the raw snapshot remains the only fact.
    """
    record = _cloud_run_record(run)
    if record is None or not record.parent_record_id:
        return None
    from ..models import TrainingRunRecord
    return db.session.get(TrainingRunRecord, record.parent_record_id)


def _resume_snapshot_with_recorded_topology(user_id, dataset_id, snapshot, topology):
    """Freeze known checkpoint topology or reject an ambiguous legacy LoKr seed."""
    fallback = (getattr(fds.get_dataset(user_id, dataset_id), 'train_settings', None)
                if snapshot is _UNSET else snapshot)
    error = lt.legacy_lokr_resume_error(topology, fallback)
    if error:
        raise ValueError(error)
    return _merge_resume_overrides(fallback, topology) if topology else snapshot


def _require_cloud_weights_only(resume_mode='weights_only', state_bundle_id=None):
    """Validate the resume contract before any cloud-side effect.

    The current pod image exposes only ai-toolkit's checkpoint upload seam. It
    cannot activate LDS's in-process state bridge, so sending optimizer/RNG
    artifacts would still replay a weights-only run. Refuse that lie explicitly
    until the remote runtime advertises the bridge capability.
    """
    mode = str(resume_mode or '').strip().lower()
    if mode not in ('weights_only', 'full_state'):
        raise ValueError("resume_mode must be 'weights_only' or 'full_state'")
    if mode == 'full_state':
        raise ValueError(
            'full-state resume is not supported by the current cloud runtime — '
            'choose weights only, or continue this verified bundle locally')
    if state_bundle_id is not None:
        raise ValueError('state_bundle_id is only valid with resume_mode=full_state')


def continue_cloud_run(user_id, run_id, extra_steps=1000, from_step=None,
                       overrides=None, resume_mode='weights_only',
                       state_bundle_id=None, transport=None,
                       allow_parallel_run=False) -> dict:
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
    _require_cloud_weights_only(resume_mode, state_bundle_id)
    run = db.session.get(CloudTrainingRun, int(run_id))
    if not run:
        raise ValueError('unknown cloud run')
    video = _video_lane(run)
    if video:
        # The video lane's own continue: its checkpoints come in steps that may
        # hold TWO files, and its launcher is the one that resolves a video
        # dataset id. `overrides` / `transport` / state bundles are face-lane
        # concepts with no video counterpart yet, and are not silently dropped —
        # `_require_cloud_weights_only` above already refused a state bundle.
        return video.continue_cloud_video_run(
            user_id, run.id, extra_steps=extra_steps, from_step=from_step)
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
    dense = p.get('training_mode') == 'full_transformer'
    _assert_recipe_replayable(p, 'continue')
    override_patch = lt.validate_resume_overrides(overrides)
    # The raw cloud snapshot historically omitted LoKr's full-rank bit, while
    # the provenance record for newer launches has the complete emitted
    # topology.  Prefer that recorded fact and freeze it into the child.  A
    # legacy LoKr source with neither fact must stop: emitting today's False
    # would silently change which checkpoint tensors can load.
    from . import checkpoint_registry
    _parent = _cloud_run_record(run)
    _parent_topology = checkpoint_registry.network_geometry(_parent)
    snapshot = _resume_snapshot_with_recorded_topology(
        user_id, run.dataset_id, p.get(_TRAIN_SETTINGS_SNAPSHOT, _UNSET),
        _parent_topology)
    cks = _dense_resume_candidates(run) if dense else _run_staging_checkpoints(run)
    if not cks and dense:
        raise ValueError(
            'this full model has nothing left to continue from: no copy on this '
            'computer, and no verified Hugging Face copy either. Give future '
            'runs the "This computer + Hugging Face" delivery (Settings ▸ '
            'Training) so a run stays resumable even after its local file is '
            'deleted.')
    if dense:
        # Filter to the chosen ROAD before choosing the step: at the same step
        # the same checkpoint can exist on both, and letting the step tie-break
        # decide the lane would pick the price by accident.
        want_source = dense_resume_transport(cks, transport)
        on_road = [c for c in cks if c.get('source') == want_source]
        if not on_road:
            raise ValueError(
                'this full model has no Hugging Face copy to continue from — '
                'send the copy on this computer to the pod instead, or give '
                'future runs the "This computer + Hugging Face" delivery '
                '(Settings ▸ Training) so the fast road exists next time.'
                if want_source == 'hub' else
                'the copy of this full model on this computer is gone — '
                'continue from its Hugging Face copy instead.')
        if want_source == 'hub':
            # The plan showed a price for this road; THIS is where the money is
            # actually committed, so the repository is checked again here rather
            # than trusted from a forecast that may be minutes old. Without it a
            # deleted repository still reads 'available' in the registry and the
            # pod is rented before anyone discovers the 404.
            #
            # Only a proven 'gone' refuses. `unknown` (offline, no token, a 5xx)
            # proceeds: blocking a resume because a check could not be made would
            # be a worse failure than the one this prevents.
            from . import hub_presence
            probe = hub_presence.check(on_road[-1]['repo_id'])
            if probe.get('state') == hub_presence.GONE:
                raise ValueError(
                    'the Hugging Face copy of this run is gone — the repository '
                    'does not answer, checked just now. Renting a pod to fetch '
                    'it would spend money on a download that cannot succeed. '
                    'Send the copy on this computer instead, if there is one.')
        cks = on_road
    if not cks:
        raise ValueError('no harvested checkpoint to continue from — this run '
                         'has none left on disk; relaunch a fresh cloud run instead')
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
        chosen = min(matches, key=lambda c: (
            video_training.split_checkpoint_name(c['filename'])[0] is None,
            c['filename']))
    try:
        extra = max(100, int(extra_steps))
    except (TypeError, ValueError):
        extra = 1000
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
    # Lineage: the source record above is also the parent edge. Legacy cloud
    # runs that predate the registry have no record -> the child is a root.
    # Where the fresh pod will take its checkpoint FROM — the one difference
    # between continuing a LoRA and continuing a full model, and it is worth
    # saying out loud in the answer: 'local' is uploaded from here, 'hub' is
    # pulled by the pod itself.
    from_hub = chosen.get('source') == 'hub'
    # The source run's stamped answers replay as-is — except the sibling
    # confirm, which the ▶ Continue dialog may have just answered FRESH: a run
    # launched alone (stamped False) must still be continuable while a sibling
    # trains, without the refusal looping on a flag nobody could carry.
    flags = _confirmation_flags(p)
    if allow_parallel_run:
        flags['allow_parallel_run'] = True
    res = launch_cloud_training(
        user_id, run.dataset_id,
        steps=chosen['step'] + extra,
        base_model=p.get('base_model', ''),
        variant=p.get('variant'),
        train_type=p.get('train_type'),
        masked=p.get('masked', True),
        training_mode=p.get('training_mode') or 'lora',
        **flags,
        gpu_name=p.get('requested_gpu'),
        resume_ckpt_path=(None if from_hub else chosen['path']),
        resume_hf=({'repo_id': chosen['repo_id'],
                    'filename': chosen['hf_filename']} if from_hub else None),
        resume_step=chosen['step'],
        train_settings_snapshot=snapshot,
        train_slider_snapshot=p.get(_TRAIN_SLIDER_SNAPSHOT, _UNSET),
        parent_record_id=(_parent.id if _parent else None),
        resumed_from=chosen['step'])
    res['resumed_from'] = chosen['step']
    res['target_steps'] = chosen['step'] + extra
    res['resume_source'] = 'hub' if from_hub else 'local'
    res['resume_transport'] = 'hub' if from_hub else 'direct'
    res['resume_from'] = (f"{chosen['repo_id']}/{chosen['filename']}" if from_hub
                          else chosen['filename'])
    return res


def continue_local_run_in_cloud(user_id, dataset_id, extra_steps=1000,
                                from_step=None, overrides=None,
                                base_model=_UNSET, variant=None, train_type=None,
                                masked=None, allow_caption_mismatch=False,
                                allow_uncaptioned=False, allow_caption_quality=False,
                                allow_unverified_weights=False, allow_not_ready=False,
                                allow_parallel_run=False,
                                gpu_name=None, training_mode='lora',
                                resume_mode='weights_only',
                                state_bundle_id=None) -> dict:
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
    _require_cloud_weights_only(resume_mode, state_bundle_id)
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if lt.normalize_training_mode(training_mode) == 'full_transformer':
        # Still refused, and the reason CHANGED. It used to say a 26 GB file
        # could not reach a pod, which is no longer true (pod_checkpoint_push).
        # The real and older reason is upstream of transport: full-model
        # training is cloud-only (lora_training._assert_local_mode), so there
        # is no local full-model run for this lane to continue in the first
        # place. Leaving the transport wording here would have advertised a
        # limit that no longer exists to explain one that does.
        raise ValueError('full models are only trained in the cloud, so this '
                         'computer has no full-model run to continue — continue '
                         'the cloud run itself, in the Runs hub')
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
    # Lineage: the parent is the record that PRODUCED the file being seeded — the
    # `record_id` list_checkpoints stamps on every save — NOT the newest record of
    # the lane. A lane holds several runs whose saves share one run dir, so "newest
    # record" pointed the edge at a run whose weights were never loaded: the graph
    # claimed a continuation of a rank-32 run while a rank-64 file went up the wire.
    # Falls back to the lane's newest record for a pre-registry save. Best-effort —
    # a failure leaves the edge NULL and never blocks the launch.
    from . import checkpoint_registry
    try:
        _parent = checkpoint_registry.record_by_id(chosen.get('record_id'))
        if _parent is None:
            _parent = checkpoint_registry.newest_record_for(dataset_id, fam, base, var)
    except Exception:
        _parent = None
    # The checkpoint's complete known topology belongs to its weights, not to
    # today's dataset settings: rank/alpha, adapter type, and LoKr's factor /
    # full-rank mode all change tensors. This lane carries a PER-RUN snapshot, so
    # it inherits those recorded facts without touching the dataset. A legacy
    # LoKr parent without full-rank provenance is deliberately blocked rather
    # than emitting today's default and changing its topology.
    topology = checkpoint_registry.network_geometry(_parent)
    _legacy_lokr_error = lt.legacy_lokr_resume_error(
        topology, getattr(ds, 'train_settings', None))
    if _legacy_lokr_error:
        raise ValueError(_legacy_lokr_error)
    snapshot = _UNSET      # _UNSET → launch stamps the dataset's live settings
    if override_patch or topology:
        snapshot = _merge_resume_overrides(getattr(ds, 'train_settings', None),
                                           {**override_patch, **topology})
    res = launch_cloud_training(
        user_id, dataset_id,
        steps=chosen['step'] + extra,
        base_model=base, variant=var, train_type=fam, masked=masked,
        allow_caption_mismatch=allow_caption_mismatch,
        allow_uncaptioned=allow_uncaptioned,
        allow_caption_quality=allow_caption_quality,
        allow_unverified_weights=allow_unverified_weights,
        allow_not_ready=allow_not_ready,
        allow_parallel_run=allow_parallel_run,
        gpu_name=gpu_name,
        resume_ckpt_path=path, resume_step=chosen['step'],
        train_settings_snapshot=snapshot,
        parent_record_id=(_parent.id if _parent else None),
        resumed_from=chosen['step'])
    res['resumed_from'] = chosen['step']
    res['target_steps'] = chosen['step'] + extra
    return res


# Module-local seams for the retry loop below, patched by tests instead of
# stdlib `time` (which other threads — the heartbeat, SQLAlchemy — also read).
_wait_sleep = time.sleep
_wait_clock = time.monotonic


def _with_frozen_dataset_generation(user_id, dataset_id, detail, operation,
                                    wait_seconds=0, on_wait=None,
                                    should_abort=None):
    """Run ``operation`` while every LDS Dataset mutation is excluded.

    ``wait_seconds`` bounds a retry on a BUSY lease (default 0 = today's
    fail-fast: a single blocking acquire, byte-identical to before). Two
    parallel launches are the case that needs it: the first run's monitor
    exports the dataset for minutes, and the second launch's freeze colliding
    with that is expected traffic, not an error. With ``wait_seconds > 0``
    the ingest lock is acquired with a short timeout instead of blocking
    unboundedly, so a sibling run holding it for its whole export cannot eat
    the deadline before a single retry is even attempted. ``on_wait`` fires on
    EVERY iteration that goes on to sleep — not once — the monitor uses it as
    a heartbeat: with ``wait_seconds`` running to the better part of an hour, a
    single write at the start of the wait leaves the run's ``updated_at``
    (and so ``_monitor_is_responsive``) stale long before the wait ends, and a
    Stop pressed then takes the false "monitor was not responding" path even
    though this loop is alive and simply waiting. ``should_abort`` is polled
    every iteration, before sleeping — a Stop during the wait must not leave
    this call waiting up to ``wait_seconds`` for a dataset it will export for
    a run that no longer wants it."""
    lock = fds._dataset_ingest_lock(user_id, dataset_id)
    bounded = bool(wait_seconds and wait_seconds > 0)
    deadline = _wait_clock() + max(0.0, float(wait_seconds or 0))

    def _try_once():
        token = dataset_activity.begin_exclusive(
            dataset_id, 'training_export', detail=detail)
        if token is None:
            return False, None
        stop = threading.Event()

        def heartbeat():
            while not stop.wait(30.0):
                dataset_activity.progress(token)

        lease = threading.Thread(
            target=heartbeat, daemon=True,
            name=f'dataset-{dataset_id}-cloud-freeze-heartbeat')
        lease.start()
        try:
            return True, operation()
        finally:
            stop.set()
            lease.join(timeout=1.0)
            dataset_activity.end(token)

    while True:
        if bounded:
            remaining = max(0.0, deadline - _wait_clock())
            if lock.acquire(timeout=min(2.0, remaining)):
                try:
                    got, result = _try_once()
                finally:
                    lock.release()
                if got:
                    return result
        else:
            with lock:
                got, result = _try_once()
            if got:
                return result
        if should_abort is not None and should_abort():
            raise _WaitAborted(
                'stop requested while waiting for the dataset')
        if _wait_clock() >= deadline:
            raise dataset_activity.DatasetActivityBusy(
                'This dataset already has work in progress. Wait for it to '
                'finish before launching cloud training.')
        if on_wait is not None:
            on_wait()
        _wait_sleep(2.0)


def _prepare_cloud_generation(user_id, dataset_id, base_model):
    def prepare():
        frozen = checkpoint_registry.prepare_launch(
            user_id, dataset_id, base_model=base_model)
        if checkpoint_registry.prepared_generation_identity(frozen) is None:
            raise RuntimeError(
                'could not freeze the Dataset provenance for cloud training; '
                'no run was started — retry after checking the backend log')
        return frozen

    return _with_frozen_dataset_generation(
        user_id, dataset_id, 'freezing the Dataset for cloud training', prepare,
        wait_seconds=120)


def launch_cloud_training(user_id, dataset_id, steps=None, base_model=_UNSET,
                          variant=None, train_type=None, masked=None,
                          allow_caption_mismatch=False, allow_uncaptioned=False,
                          allow_caption_quality=False,
                          allow_unverified_weights=False, allow_not_ready=False,
                          allow_hf_storage=False, allow_local_disk=False,
                          allow_parallel_run=False,
                          gpu_name=None, resume_ckpt_path=None, resume_step=None,
                          resume_hf=None,
                          auto_retry_count=0, auto_retry_of=None,
                          strict_gpu=False, train_settings_snapshot=_UNSET,
                          train_slider_snapshot=_UNSET,
                          parent_record_id=None, resumed_from=None,
                          training_mode='lora') -> dict:
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
    mode = lt.normalize_training_mode(training_mode)
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
    # Anima is LOCAL-ONLY for this wave: a pod would need ai-toolkit with the
    # 'anima' arch (PR #860, 2026-07-15) + a recent diffusers, which current pod
    # images predate — renting one would burn a GPU on an unknown arch. Refuse
    # BEFORE any reservation. Lift once the pod image is verified.
    if fam == 'anima':
        raise ValueError('Anima cloud training is coming once the pod image is '
                         'verified — train it locally for now')
    variant = (variant or '').strip().lower()
    if mode == 'full_transformer':
        if fam != 'krea':
            raise ValueError('full_transformer cloud training is supported only '
                             'for Krea 2')
        # Raw, Turbo, or a custom checkpoint — all three now reach the pod with
        # the base they name (lt._krea_name_or_path). The variant is NO LONGER
        # overwritten with 'base' here: that line is what would have turned a
        # lifted refusal into a run labelled Turbo and trained on Raw.
        if variant and variant not in lt._valid_variants_for(fam):
            variant = lt._default_variant_for(fam)
        # Continuing a dense run from the copy on THIS COMPUTER is supported.
        # It used to be refused here, and the refusal was honest about its
        # reason: the pod's dataset-upload route was driven with a multipart
        # body built ENTIRELY in memory (an 85 MB LoRA is fine; 26 GB is an
        # OOM) under a 300 s timeout. Both halves of that are gone —
        # `upload_file_slice` produces the body as it sends it, and
        # `pod_checkpoint_push` cuts the file into resumable slices with a
        # per-slice timeout. What remains is a COST, not a wall: the user's
        # uplink, billed at the pod's hourly rate the whole way up. That is a
        # number to show (pod_transfer_plan), not a reason to refuse.
        if resume_ckpt_path:
            if not os.path.isfile(resume_ckpt_path):
                raise ValueError(
                    'the full model to continue from is no longer on this '
                    'computer — continue from its Hugging Face copy instead, '
                    'or pick another checkpoint')
            if os.path.getsize(resume_ckpt_path) <= 0:
                raise ValueError(
                    'the full model to continue from is an empty file on this '
                    'computer — it cannot be trained further')
        slider_value = (getattr(ds, 'train_slider', None)
                        if train_slider_snapshot is _UNSET
                        else train_slider_snapshot)
        slider_view = _RunConfigDataset(
            ds, fam, variant, base_model, train_slider_snapshot=slider_value,
            training_mode=mode)
        if lt.slider_mode_enabled(slider_view):
            raise ValueError('full_transformer cloud training is incompatible '
                             'with Slider LoRA mode')
        # Recipe validation with the LAUNCH's selection, not the stored one:
        # family, Slider, and the mechanical fp8-export refusal on a custom base.
        lt._assert_full_transformer_recipe(slider_view)
        # Validate token type/scopes and real Krea-base readability before the
        # reservation row exists. Required whatever the delivery is: the Krea 2
        # base itself is GATED, so the pod needs this credential to read it.
        # Repository creation later proves write access before any pod is rented.
        # The repository asked for is the one THIS run needs (Raw or Turbo), and
        # None for a custom base — whose private repo is covered by the delivery
        # namespace scope, and is separately proven readable below.
        dense_api, delivery_namespace, _broad = _validate_full_transformer_token(
            cfg.secret('HF_CLOUD_TOKEN'),
            required_base_repo=lt.official_base_repo(slider_view, fam, variant))
        dense_delivery = dld.configured_mode()
        dense_keep_bf16 = lt.dense_keep_bf16_master(ds)
        dense_fp8 = lt.dense_fp8_export_enabled(ds)
        if dld.delivers_local(dense_delivery):
            # THIS machine's disk, checked with the same care as the Hub's:
            # the drive filled up twice in one week, and a delivery that runs
            # out of room lands after the money is spent. Confirmable, and an
            # unmeasurable volume never blocks.
            dld.assert_local_disk_headroom(
                keeps=lt.dense_max_step_saves_for(ds), fp8_export=dense_fp8,
                keep_bf16=dense_keep_bf16,
                allow_override=bool(allow_local_disk))
        # ...and then the OTHER wall run #146 hit, after hours of paid GPU: the
        # delivery namespace's private storage. One dense save is ~26 GB.
        # It is no longer the same bet, and that is the point of the ordering:
        #  * hub-only — the repository is the ONLY address the artifact will
        #    ever have, so a forecast that does not fit is a refusal (still
        #    confirmable: the ceiling is an estimate).
        #  * anything with a local copy — the Hub copy is a BACKUP taken after
        #    the local file is harvested and proven, so it cannot cost the run
        #    any more. It only buys resumability, so a bad forecast is a WARNING
        #    said before the GPU, not a wall.
        # The Hub copy is the master alone (one file, once), never the fp8 twin:
        # the twin is regenerated from the master in seconds and would eat the
        # private quota twice as fast.
        hub_only = not dld.delivers_local(dense_delivery)
        dense_storage_forecast = _assert_dense_storage_headroom(
            delivery_namespace, dense_api, bool(allow_hf_storage),
            keeps=(lt.dense_max_step_saves_for(ds) if hub_only else 1),
            fp8_export=dense_fp8 if hub_only else False,
            required=hub_only) if dld.delivers_hub(dense_delivery) else None
        dense_hub_warning = (dld.hub_backup_warning(dense_storage_forecast)
                             if not hub_only else None)
    else:
        dense_delivery = None
        dense_hub_warning = None
    confirmations = {
        'allow_caption_mismatch': bool(allow_caption_mismatch),
        'allow_uncaptioned': bool(allow_uncaptioned),
        'allow_caption_quality': bool(allow_caption_quality),
        'allow_unverified_weights': bool(allow_unverified_weights),
        'allow_not_ready': bool(allow_not_ready),
        'allow_hf_storage': bool(allow_hf_storage),
        'allow_local_disk': bool(allow_local_disk),
        'allow_parallel_run': bool(allow_parallel_run),
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
        if mode == 'full_transformer':
            # The private base repo is created with the GENERAL HF_TOKEN, but a
            # dense pod authenticates with HF_CLOUD_TOKEN only
            # (_hf_token_for_mode). Those are two different credentials and
            # nothing guarantees the second can read what the first pushed —
            # a delivery namespace on another account or an org would 403 on
            # the pod, hours of GPU later. Fail-open on anything but an
            # outright refusal, exactly like the official-base gate check.
            _assert_dense_custom_base_readable(
                (base_repo or {}).get('repo_id'), cfg.secret('HF_CLOUD_TOKEN'))
    else:
        # OFFICIAL base: the pod downloads it from Hugging Face. Several are GATED
        # (Krea, FLUX, FLUX.2 Klein) and a gate the account never accepted answers
        # 403 — on the pod, after renting. Three runs were paid for and lost that
        # way, and the card only showed "403 Client Error (Request ID…)", hiding the
        # sentence that named the repo. One HEAD here costs nothing and turns that
        # into a message before a GPU is reserved.
        # Through the launch VIEW, not the dataset row: a base persisted on the
        # dataset after (or between) launches would make official_base_repo
        # answer None and silently skip the gate check for a run that does use
        # the official base.
        _assert_official_base_reachable(
            lt.official_base_repo(
                _RunConfigDataset(ds, fam, variant, base_model or '',
                                  training_mode=mode),
                fam, variant),
            _hf_token_for_mode(mode))
    # Cheap fast-fail before the image/caption preflight below. This read is
    # intentionally advisory: another Flask request can reserve a slot after
    # it, so the same checks are repeated atomically at reservation time. The
    # confirmation flag must ride along here too — otherwise a confirmed retry
    # would still die on this early call before ever reaching the ceiling/budget
    # checks it is meant to fall through to.
    _assert_launch_guardrails(dataset_id, fam, allow_parallel_run=allow_parallel_run)

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
    # Freeze the dataset (manifest + caption text + image content hashes +
    # environment) BEFORE the reservation lock, exactly like the local path: the
    # only file I/O of the registration happens here, so the registration itself
    # stays one short write and neither the reservation window nor the launch
    # response grows a second writer competing for the database lock.
    _prepared = _prepare_cloud_generation(
        user_id, dataset_id, base_model)
    with _launch_reservation_lock:
        # Authoritative re-check + insert. Keeping the commit inside this
        # process-wide critical section means a second request always sees the
        # first request's preparing row before it can reserve or rent a pod.
        _assert_launch_guardrails(dataset_id, fam, allow_parallel_run=allow_parallel_run)
        run = CloudTrainingRun(
            dataset_id=dataset_id, status='preparing', run_name=run_name,
            # Stamp the family in the reservation itself. Without this, the
            # short window before the complete params are saved would make a
            # legitimate second-family launch look like an unknown-family run.
            train_params=json.dumps({
                'train_type': fam, 'variant': variant,
                'base_model': base_model, 'training_mode': mode,
                'artifact_kind': (_FULL_TRANSFORMER_ARTIFACT
                                  if mode == 'full_transformer' else 'lora'),
                # Stamped in the RESERVATION row, not only in the full params:
                # everything downstream reads the run's own delivery, and the
                # window between these two writes is one an app restart can
                # land in. Absent = the legacy Hugging-Face-only meaning.
                **({'dense_delivery': dense_delivery}
                   if mode == 'full_transformer' else {}),
                'artifact_status': (
                    ('creating_repository' if dld.delivers_hub(dense_delivery)
                     else 'not_requested')
                    if mode == 'full_transformer' else None),
                **confirmations,
            }))
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
        job_prefix = 'Krea_' if mode == 'full_transformer' else ''
        _set(run, vast_label=f'lds-{run.id}',
             job_name=f'{job_prefix}lds{run.id}_{run_name}')
        artifact = {}
        if mode == 'full_transformer' and dld.delivers_hub(dense_delivery):
            # Created up front even though the push now happens at the END: the
            # repository carries the Krea 2 licence, NOTICE and model card, and
            # a derivative that reaches the Hub without them is a compliance
            # problem, not a missing nicety.
            artifact = _create_full_transformer_repo(
                run, cfg.secret('HF_CLOUD_TOKEN'),
                base_repo=_dense_base_repo_for(
                    {'variant': variant,
                     'base_repo_id': (base_repo or {}).get('repo_id')}))
        # Mirror the LOCAL launch: persist this dataset's family/variant as its
        # remembered selection (launch_training does the same; two launch tests
        # assert it). This is now ONLY the dataset's default selection — the
        # monitor builds the pod job from the run's STAMPED params (see
        # _run_config_dataset at the build site), so a later launch overwriting
        # this row can no longer retarget an already-provisioning run's arch.
        ds.train_type = fam
        ds.train_variant = variant
        ds.training_mode = mode
        db.session.commit()
        # Same floor as the local path — a sub-500 target produces a run with
        # zero usable snapshots.
        n_steps = (max(500, int(steps)) if steps else lt.default_steps(
            ds, train_type=fam, variant=variant))
        # requested_gpu (from the launch-time speed picker) is a PREFERENCE, not
        # a lock: _provision re-searches live offers and rents the cheapest one
        # of this class, falling back to the cheapest overall if the class has
        # since sold out (vast offers are ephemeral).
        # `masked` resolved ONCE, here, then stamped into the run params. That
        # stamp is what _prepare_staging reads to decide whether rembg generates
        # the person masks that get UPLOADED with the dataset — the cloud lane's
        # only source of truth for masking. `None` (a fresh launch) = the
        # dataset's stored setting; an explicit bool = a retry/continue replaying
        # the source run's own frozen flag.
        masked = lt.resolve_masked(ds, masked)
        params = {'steps': n_steps, 'variant': variant, 'base_model': base_model,
                  'train_type': fam, 'training_mode': mode,
                  'artifact_kind': (_FULL_TRANSFORMER_ARTIFACT
                                    if mode == 'full_transformer' else 'lora'),
                  'masked': bool(masked), **confirmations}
        if mode == 'full_transformer':
            params['dense_delivery'] = dense_delivery
            if dld.delivers_local(dense_delivery):
                params['local_artifact_status'] = 'pending'
                params['fp8_keep_bf16'] = bool(dense_keep_bf16)
            if dense_hub_warning:
                params['hub_backup_warning'] = dense_hub_warning
        if artifact:
            params.update(artifact)
            params['artifact_status'] = 'pending'
        elif mode == 'full_transformer':
            params['artifact_status'] = 'not_requested'
            params['artifact_status_detail'] = (
                'This full model is delivered to this computer only — no '
                'Hugging Face copy was requested.')
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
            params['resume_source'] = 'local'
            # Frozen here because _disk_gb_for reads it when the pod is RENTED,
            # and because it is what the run card quotes while the push runs. A
            # file that vanishes between launch and seeding then fails on the
            # missing file, not on a size nobody can recover.
            try:
                params['resume_ckpt_bytes'] = os.path.getsize(resume_ckpt_path)
            except OSError:
                params['resume_ckpt_bytes'] = 0
            if resume_step is not None:
                params['resume_step'] = int(resume_step)
        elif resume_hf:
            # The other seed channel: the POD downloads the checkpoint from the
            # Hub itself. Same contract as the local seed — it lands in the
            # job's save_root before start_job and ai-toolkit auto-resumes from
            # it — but the 26 GB never crosses the user's uplink.
            params['resume_hf_repo_id'] = str(resume_hf.get('repo_id') or '')
            params['resume_hf_filename'] = str(resume_hf.get('filename') or '')
            params['resume_source'] = 'hub'
            if resume_step is not None:
                params['resume_step'] = int(resume_step)
        # Provenance registry (same as local launches): dataset version at
        # launch time, stamped into the params so payloads can expose it.
        rec = checkpoint_registry.register_launch(
            user_id, dataset_id, family=fam, source='cloud',
            variant=variant, masked=bool(masked), steps=n_steps,
            cloud_run_id=run.id,
            settings=lt.launch_settings_snapshot(
                _run_config_dataset(ds, params), fam, masked=masked),
            prepared=_prepared,
            parent_record_id=parent_record_id, resumed_from=resumed_from)
        if rec is None:
            raise RuntimeError(
                'could not persist the Dataset provenance for cloud training; '
                'the run was not started')
        params['version'] = rec.version
        params['record_id'] = rec.id
        _set(run, train_params=json.dumps(params))
        _stop_event_for(run.id).clear()
        _start_monitor(run.id)
    except Exception as e:
        _set(run, status='error', error=f'launch failed: {e}',
             finished_at=datetime.utcnow())
        raise
    result = {'run_id': run.id, 'status': run.status,
              'job_name': run.job_name, 'steps': n_steps,
              'training_mode': mode}
    if mode == 'full_transformer':
        result.update({'artifact_kind': _FULL_TRANSFORMER_ARTIFACT,
                       'dense_delivery': dense_delivery,
                       'hf_repo_id': params.get('hf_repo_id'),
                       'hf_url': params.get('hf_url')})
        if dense_hub_warning:
            # Said at LAUNCH, in the launch response, because "this model will
            # not be resumable" is worth knowing before eight hours of GPU.
            result['warning'] = dense_hub_warning
    return result


_AUTO_RETRY_LIMIT = 1
_AUTO_RETRY_MARKERS = (
    'pod did not become ready',
    'pod unreachable',
    # A pod that vanished across an app restart used to say 'did not become
    # ready' and earn the same single retry; it now has its own wording, and
    # dropping it from this list would have silently retired that retry.
    'could not be reached again after the app restarted',
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
        resume_snapshot = params.get(_TRAIN_SETTINGS_SNAPSHOT, _UNSET)
        if params.get('resume_ckpt_path'):
            # A pod retry that re-seeds a checkpoint is a continuation, not a
            # fresh run. Refuse an ambiguous legacy LoKr source before claiming
            # a retry or renting another GPU; newer parent provenance is folded
            # into the child's frozen snapshot.
            from . import checkpoint_registry
            topology = checkpoint_registry.network_geometry(_resume_parent_record(run))
            try:
                resume_snapshot = _resume_snapshot_with_recorded_topology(
                    'local', run.dataset_id, resume_snapshot, topology)
            except ValueError as e:
                logger.warning('automatic retry blocked for unsafe resume topology on run %s: %s',
                               run.id, e)
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
        # An auto-retry replaces a run whose pod is already dead, so a live
        # sibling's same-family guard must never block it — the fleet ceiling
        # and monthly budget still guard the spend either way. Overridden
        # AFTER _confirmation_flags(params) replays this run's own stamped
        # answer (which may be False — this run never confirmed it).
        retry_flags = _confirmation_flags(params)
        retry_flags['allow_parallel_run'] = True
        try:
            result = launch_cloud_training(
                'local', run.dataset_id,
                steps=params.get('steps'),
                base_model=params.get('base_model', ''),
                variant=params.get('variant'),
                train_type=params.get('train_type'),
                training_mode=params.get('training_mode', 'lora'),
                masked=params.get('masked', True),
                **retry_flags,
                gpu_name=gpu_name,
                resume_ckpt_path=params.get('resume_ckpt_path'),
                resume_step=params.get('resume_step'),
                auto_retry_count=retry_count + 1,
                auto_retry_of=run.id,
                strict_gpu=bool(gpu_name),
                train_settings_snapshot=resume_snapshot,
                train_slider_snapshot=params.get(_TRAIN_SLIDER_SNAPSHOT, _UNSET))
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
    handler (run flips to 'error', slot freed) — except a Stop fired while
    waiting for a sibling's export lease, which the monitor lands as
    'stopped' (see ``_WaitAborted``)."""
    if run.staging_dir:
        return
    run_id = run.id           # captured once: should_abort polls it up to
    ev = _stop_event_for(run_id)   # ~1800 times and must not re-read the ORM
    staging = _staging_root() / f'run_{run.id}'
    if crd.is_video(run):
        # Nothing to export, and none of the face-dataset generation checks
        # below apply: the checkpoint registry resolves FACE datasets, and a
        # video dataset's folder is ALREADY the flat mp4 + homonym .txt shape
        # ai-toolkit wants — that is what the builder writes — so the staging
        # copy the image lane needs (rembg masks, ~1-2 s an image) would only
        # duplicate gigabytes of clips to no end. The staging dir still exists
        # for the samples the pod sends back; _staging_dataset_dir is what
        # points the upload at the real folder.
        (staging / 'samples').mkdir(parents=True, exist_ok=True)
        _set(run, staging_dir=str(staging))
        return
    _set(run, phase_detail='Preparing dataset (masks)…')
    params = json.loads(run.train_params or '{}')

    def verify_and_export():
        # Re-stamp in case on_wait overwrote it with the "waiting" text below
        # — this only runs once the lease is actually held.
        _set(run, phase_detail='Preparing dataset (masks)…')
        record = checkpoint_registry.record_by_id(params.get('record_id'))
        expected = checkpoint_registry.record_generation_identity(record)
        current = checkpoint_registry.prepare_launch(
            'local', run.dataset_id,
            base_model=params.get('base_model') or None)
        observed = checkpoint_registry.prepared_generation_identity(current)
        if expected is None or observed is None or observed != expected:
            raise RuntimeError(
                'The Dataset changed after this cloud run was requested. '
                'Nothing was uploaded or trained; launch a new run from the '
                'current Dataset.')
        current_ds = current['ds']
        if (getattr(current_ds, 'train_settings', None)
                != params.get(_TRAIN_SETTINGS_SNAPSHOT)
                or getattr(current_ds, 'train_slider', None)
                != params.get(_TRAIN_SLIDER_SNAPSHOT)):
            raise RuntimeError(
                'The Dataset training options changed after this cloud run was '
                'requested. Nothing was uploaded or trained; launch a new run.')
        (staging / 'samples').mkdir(parents=True, exist_ok=True)
        return lt.export_dataset_to_aitoolkit(
            'local', run.dataset_id,
            masked=bool(params.get('masked', True)),
            dest_dir=str(staging / 'dataset'))

    # on_wait now ticks every ~2 s of the wait (see _with_frozen_dataset_
    # generation) instead of firing once — a wait_seconds=3600 export lease
    # would otherwise leave run.updated_at stale past STOP_HANDOFF_SECONDS,
    # and a Stop pressed mid-wait would find _monitor_is_responsive False and
    # force-terminate a pod that does not exist yet, with a message claiming
    # the monitor had died. Throttled to one write per ~30 s: still far under
    # the 120 s handoff window, without hammering the DB every tick.
    last_heartbeat = [float('-inf')]   # first tick always fires

    def _wait_tick():
        now = _wait_clock()
        if now - last_heartbeat[0] < 30.0:
            return
        last_heartbeat[0] = now
        # _set_soft, not _set: this write is purely cosmetic (see its own
        # docstring). A sibling run hammering the DB with its export can
        # outlive the write-lock retry budget, and _set would then record
        # this WAITING run as 'Run failed — database is locked' — exactly
        # the outcome this heartbeat exists to prevent. A skipped tick is
        # harmless: the next one lands in 30 s, well inside the 120 s window.
        _set_soft(run, phase_detail='Waiting for the dataset — another run '
                                    'is exporting…')

    _with_frozen_dataset_generation(
        'local', run.dataset_id,
        'verifying and exporting the Dataset for cloud training',
        verify_and_export, wait_seconds=3600,
        on_wait=_wait_tick,
        should_abort=ev.is_set)
    _set(run, staging_dir=str(staging))


def _build_pod_job_config(run, staging_dataset: str, pod_settings: dict) -> dict:
    """The ai-toolkit job config this run's pod will receive.

    Extracted from the monitor's boot path so the video branch is one `if` in one
    place rather than a second copy of the build-then-cloudify pair — and so a
    test can assert what the pod gets without provisioning anything.

    Built from the run's STAMPED family/variant, NEVER the dataset's current
    train_type/train_variant: a later launch on the same dataset (or a
    /train-type change) may have moved that row since this run launched, and this
    rebuild happens minutes later at pod boot. `_run_config_dataset` presents the
    run's own launch params so two concurrent multi-family runs each get their
    own arch (incident 2026-07-14 — see _RunConfigDataset)."""
    params = json.loads(run.train_params or '{}')
    if crd.is_video(run):
        vds = crd.dataset_row(run)
        if vds is None:
            raise RuntimeError(f'run {run.id} trained a video dataset that is gone')
        job_config = video_training.build_job_config(
            vds, staging_dataset, steps=params.get('steps') or 1000,
            training_folder='__POD__', base_model=params.get('base_model') or None,
            # The measured reason this run is on a rented GPU at all: low_vram
            # cost 170-185 s a step on 24 GB by shuttling the idle expert over
            # PCIe. Stamped at launch so the pod cannot be re-decided later.
            low_vram=bool(params.get('low_vram', False)))
    else:
        ds = fds.get_dataset('local', run.dataset_id)
        job_config = lt.build_job_config(
            _run_config_dataset(ds, params),
            staging_dataset, steps=params.get('steps') or 3000,
            training_folder='__POD__')
    return _cloudify_job_config(job_config, run.job_name, staging_dataset,
                                pod_settings, run_params=params)


def _staging_dataset_dir(run) -> str:
    """The folder whose contents get uploaded to the pod as this run's dataset.

    For a face run that is the exported copy under staging. For a video run it is
    the dataset's OWN output_dir: it already has the right shape, and a dataset of
    81-frame clips is gigabytes — copying it would double the disk and the wait to
    produce a byte-identical folder."""
    if crd.is_video(run):
        row = crd.dataset_row(run)
        if row is None or not row.output_dir:
            raise RuntimeError(f'run {run.id} has no video dataset folder left')
        return str(row.output_dir)
    return os.path.join(run.staging_dir, 'dataset')


def _assert_pod_can_decode(run, remote, pod_settings):
    """Before the job starts: can this pod READ the clips it was just sent?

    Only for a video run — a face run uploads jpegs and would gain a new way to
    fail for a decoder it never calls.

    The placement is the design. A pod is billed from boot, and whether its image
    can decode these mp4s is genuinely unknown: OpenCV's bundled ffmpeg has no
    software AV1 decoder, PyAV is absent from some images, and an image without
    libGL cannot import cv2 at all. Each of those ends the same way — a job that
    runs and yields nothing. Asked here, the answer costs seconds of an
    already-booted pod; discovered later it costs the run. This is run #138's
    lesson one step further along: the phase you do not observe is the phase that
    bills you.

    A refusal RAISES, and the monitor's generic handler turns it into a failed
    run with the pod released. Standing down "just in case the probe is wrong"
    would restore exactly the blind launch this exists to remove."""
    from . import pod_video_probe
    if not crd.is_video(run):
        return None
    profile = video_targets.get(_run_param(run, 'target_profile')
                                or getattr(crd.dataset_row(run),
                                           'target_profile', None)) or {}
    # Only when the target actually trains on the track. MiniMax H3 is a joint
    # video+audio model on muxed clips: a pod that decodes the picture and finds
    # no audio stream trains a video-only LoRA under an audio target's name.
    want_audio = bool((profile.get('audio') or {}).get('muxed'))
    pod_dir = (pod_settings['DATASETS_FOLDER'].rstrip('/') + '/' + run.job_name)
    _set(run, phase_detail='Checking the pod can read the clips…')
    verdict = pod_video_probe.probe_decoder(
        remote, instance_id=run.vast_instance_id, pod_dataset_dir=pod_dir,
        want_audio=want_audio, tmp_dir=run.staging_dir or str(_staging_root()),
        should_cancel=lambda: _stop_event_for(run.id).is_set())
    logger.info('run %s: the pod decoded %s with %s (%s frames)', run.id,
                verdict.get('clip'), verdict.get('decoder'), verdict.get('frames'))
    _set(run, phase_detail=f'Pod reads the clips with '
                           f'{verdict.get("decoder") or "its decoder"}')
    return verdict


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


def _run_host_ip(run):
    """Public address of the host this run rented, or None.

    Two sources, in order of trust: the address stamped from the RENTED
    instance (measured — it is the same field the pod's base_url is built
    from), then the one the offer advertised. A bad host that re-registers
    under a new machine_id keeps its address, which is the only reason this
    exists (2026-07-28)."""
    try:
        parsed = json.loads(run.train_params or '{}')
        if not isinstance(parsed, dict):
            return None
        return parsed.get('host_ip') or parsed.get('offer_ip') or None
    except (ValueError, TypeError):
        return None


def _load_bad_hosts() -> dict:
    """{machine_id(str): {'ts': epoch, 'reason': str, 'ip': str|None,
    'ttl': seconds|None}} — expired entries are dropped on read. The default TTL
    is cloud.host_blacklist_days; an entry may carry its OWN shorter 'ttl' when
    the failure said "slow", not "broken" (see _blacklist_host).
    Corrupt file -> empty. Legacy files (entries without 'ip'/'ttl') load
    unchanged; they simply ban one machine_id on the default TTL, as always."""
    try:
        raw = json.loads(_bad_hosts_path().read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    default_ttl = float(cfg.get('cloud.host_blacklist_days') or 3) * 86400
    now = _now()

    def _alive(v):
        if not isinstance(v, dict):
            return False
        try:
            ttl = float(v.get('ttl')) if v.get('ttl') is not None else default_ttl
        except (TypeError, ValueError):
            ttl = default_ttl
        return now - float(v.get('ts') or 0) <= ttl

    live = {k: v for k, v in raw.items() if _alive(v)}
    if len(live) != len(raw):
        try:
            _bad_hosts_path().write_text(json.dumps(live), encoding='utf-8')
        except OSError:
            pass
    return live


def _blacklist_host(machine_id, reason, ip=None, ttl_seconds=None):
    """Remember a host whose pod never became ready so the next launch (and the
    tier list) skips it for a few days. Best-effort: never raises.

    The entry is still KEYED by machine_id (legacy files keep working), but it
    also records the host's public address when one is known, because
    machine_id alone is defeatable: run #120 failed on a machine, that machine
    was blacklisted, and run #121 was rented three minutes later on a DIFFERENT
    machine_id at the same address — the same box, re-registered (a vast
    machine_id is a file on the host; reinstalling the daemon mints a new one).

    The address is a WEAKER identity than the machine id — several machines can
    sit behind one NAT — so it only ever widens a ban that a real failure
    already justified, it expires on the same TTL, and _filter_offers refuses
    to let it starve a launch.

    ttl_seconds overrides the default TTL for THIS entry — a host killed while
    it was still visibly booting is slow, not broken, and a three-day exile is
    the wrong price for that. A later, generic ban on the same host (the
    retry path re-bans every failure it classifies as transient) INHERITS the
    explicit ttl instead of silently upgrading it back to the default: the
    specific classification of a failure outranks the generic one."""
    if not machine_id and not ip:
        return
    try:
        hosts = _load_bad_hosts()
        key = str(machine_id) if machine_id else f'ip:{ip}'
        prev = hosts.get(key) if isinstance(hosts.get(key), dict) else {}
        ttl = ttl_seconds if ttl_seconds is not None else prev.get('ttl')
        hosts[key] = {'ts': _now(), 'reason': str(reason)[:200], 'ip': ip or None,
                      'ttl': float(ttl) if ttl is not None else None}
        _bad_hosts_path().write_text(json.dumps(hosts), encoding='utf-8')
        logger.warning('blacklisted vast host machine_id=%s ip=%s for %s: %s',
                       machine_id, ip or '?',
                       f'{float(ttl) / 3600:.0f} h' if ttl is not None
                       else f"{cfg.get('cloud.host_blacklist_days') or 3} day(s)",
                       reason)
    except Exception:
        logger.exception('could not blacklist host %s', machine_id)


def _stamp_host_ip(run, ip):
    """Record the rented pod's public address in train_params (once). Silent on
    any failure: this is bookkeeping for a future ban, never a reason to fail
    a boot that is otherwise going fine."""
    try:
        parsed = json.loads(run.train_params or '{}')
        if not isinstance(parsed, dict) or parsed.get('host_ip') == str(ip):
            return
        parsed['host_ip'] = str(ip)
        _set(run, train_params=json.dumps(parsed))
    except Exception:
        logger.debug('could not stamp the host address of run %s', run.id)


def _stamp_pod_image(run, image):
    """Record the image the rented pod is ACTUALLY running, once.

    Not the same fact as the image we asked for. The default launch path is a
    vast.ai template published by a third party: its contents can change without
    a line of this repo changing, and `cloud.image` is only the raw-image
    fallback. So until now nothing anywhere could answer "which trainer produced
    these weights?" — the answer lived in a config we BELIEVED was in force.

    That question is not academic for dense runs: which ai-toolkit ran decides
    whether a setting in the recipe was honoured, ignored, or silently
    mis-calibrated, and a run that goes wrong six months from now has to be able
    to say it itself rather than have it re-derived from today's config. Same
    principle as reading a checkpoint's own header instead of trusting its
    filename.

    Silent on failure, like `_stamp_host_ip`: bookkeeping never fails a boot."""
    try:
        parsed = json.loads(run.train_params or '{}')
        if not isinstance(parsed, dict) or parsed.get('pod_image') == str(image):
            return
        parsed['pod_image'] = str(image)
        _set(run, train_params=json.dumps(parsed))
    except Exception:
        logger.debug('could not stamp the pod image of run %s', run.id)


def _blacklist_run_host(run, reason, ttl_seconds=None):
    """Blacklist the host a RUN was on, with every identity it left behind."""
    _blacklist_host(_run_machine_id(run), reason, ip=_run_host_ip(run),
                    ttl_seconds=ttl_seconds)


def _banned_ips(bad) -> set:
    return {str(v.get('ip')) for v in bad.values()
            if isinstance(v, dict) and v.get('ip')}


def _offer_ip(offer) -> str:
    """Public address advertised by an OFFER. Documented on the vast offer
    object; treated as optional because nothing guarantees it is populated for
    every offer — when it is absent the address ban simply does not apply to
    that offer, and the machine_id ban still does."""
    return str(offer.get('public_ipaddr') or '')


def _filter_offers(offers) -> list:
    """Drop blacklisted hosts and bait-priced offers (< 60% of their GPU
    class's median price when the class has >= 3 offers — with fewer there is
    no reliable median). Never returns [] when the input wasn't: if every
    offer got filtered, fall back to the input minus blacklisted hosts only
    (renting a suspect host beats failing the run outright)."""
    bad = _load_bad_hosts()
    banned_ips = _banned_ips(bad)
    by_machine = [o for o in offers
                  if str(o.get('machine_id') or '') not in bad]
    not_blacklisted = [o for o in by_machine
                       if not (_offer_ip(o) and _offer_ip(o) in banned_ips)]
    if not not_blacklisted and by_machine:
        # The address ban is the wide one; it must never be the reason a launch
        # finds nothing. Fall back to the narrow machine_id ban and say so.
        logger.warning('every remaining offer sits on a blacklisted address — '
                       'falling back to the machine-id blacklist only')
        not_blacklisted = by_machine
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
    if params.get('training_mode') == 'full_transformer':
        dense = cloud_cfg.get('full_transformer') or {}
        # Safety floor, even when an old/user-edited config carries a smaller
        # number: base + working weights + one ~26 GB save do not fit below it.
        disk_gb = max(200, int(dense.get('disk_gb') or 200))
    else:
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
    # A checkpoint pushed from this computer lands on the pod IN SLICES that are
    # then assembled, so it briefly occupies itself plus one slice on top of
    # everything above. Asked for at RENTAL time on purpose: discovering the
    # shortfall when the file is already half sent means the money is spent and
    # the pod has to be thrown away. The Hub road needs nothing extra here — the
    # pod writes the file straight to its destination.
    try:
        seed_bytes = int(params.get('resume_ckpt_bytes') or 0)
    except (TypeError, ValueError):
        seed_bytes = 0
    if seed_bytes:
        from . import pod_checkpoint_push
        needed = disk_gb + int(
            (seed_bytes + min(pod_checkpoint_push.DEFAULT_SLICE_BYTES,
                              seed_bytes)) / 1e9) + 1
        logger.info('a %.1f GB checkpoint is being pushed to this pod — disk '
                    'bumped %s -> %s GB', seed_bytes / 1e9, disk_gb, needed)
        disk_gb = needed
    return disk_gb


def rent_with_fresh_offers(*, search, create, pick=None, on_offer=None,
                           no_offer_message=None, attempts=None, sleep=None):
    """Rent the first offer vast actually accepts, re-searching between tries.

    Vast's offer index is a CACHE. An offer it hands back can already be sold,
    or sit on a host that refuses the ask for a reason the listing does not
    carry, and the refusal arrives as a bare ``HTTP 400`` at create time — run
    #80 died there, and so did the first cloud quantization. One shot at one
    offer therefore loses a launch that a second offer would have won. Each
    attempt re-searches live, skips what it already tried, and re-picks.

    Shared with the quantization lane on purpose: a second copy of this loop
    would drift, and the blacklist and bait-price filter must cover both.

    ``search`` returns live offers (the caller owns the resource predicates),
    ``pick`` chooses among the already-filtered survivors — it may raise its own
    refusal, which is final and never retried — and ``create`` rents the chosen
    one. ``on_offer`` sees the offer just before it is rented (host stamping).
    Returns ``(instance_id, offer)``.
    """
    attempts = int(attempts or _CREATE_INSTANCE_ATTEMPTS)
    sleep = sleep or _sleep
    pick = pick or (lambda offers: _pick_offer(offers, None))
    tried = set()
    last_error = None
    for attempt in range(1, attempts + 1):
        pool = [o for o in (search() or []) if o.get('offer_id') not in tried]
        if not pool:
            if tried:
                # Carry vast's own words out: a marketplace that refuses every
                # machine is diagnosable only through the last refusal, and
                # dropping it here is how 'HTTP 400 {}' became an hour of guessing.
                raise RuntimeError(
                    f'no vast.ai offer left after {len(tried)} refused attempt(s) — '
                    f'last refusal: {last_error}')
            raise RuntimeError(no_offer_message or 'no vast.ai offer matches right now')
        offer = pick(_filter_offers(pool))
        tried.add(offer['offer_id'])
        if on_offer:
            on_offer(offer)
        try:
            return create(offer), offer
        except vast_client.VastError as e:
            if attempt >= attempts or not _is_transient_create_error(e):
                raise
            last_error = e
            logger.warning('create_instance attempt %s/%s failed (%s) — retrying '
                           'with a fresh offer', attempt, attempts, e)
            sleep(_CREATE_INSTANCE_BACKOFF)


def _provision(run):
    """Search offers and create the instance, honoring the launch-time GPU
    choice when the picked class is still available.
    LEAK-SAFE: any failure after create_instance destroys the instance."""
    c = cfg.get('cloud') or {}
    params = json.loads(run.train_params or '{}')
    fam = params.get('train_type') or 'zimage'
    if params.get('training_mode') == 'full_transformer':
        dense = c.get('full_transformer') or {}
        min_vram = max(80, int(dense.get('min_vram_gb') or 80))
    else:
        min_vram = (c.get('min_vram_gb') or {}).get(fam, 24)
    disk_gb = _disk_gb_for(c, params)
    template_hash = (c.get('template_hash') or '').strip()
    # A transient create refusal (offer just taken -> HTTP 400/409, rate limit,
    # vast 5xx — run #80's 'HTTP 400 {}' died here with no retry) gets a bounded
    # re-search; a non-transient one (auth, 404) raises immediately. The loop
    # itself is rent_with_fresh_offers, shared with the quantization lane.
    # `token` is set by _create below and read after the rental — the raw-image
    # branch mints the UI bearer, the template branch has none.
    token = ''
    # The offer search runs under the 'preparing' status and used to keep the
    # staging sentence, so a search that found nothing looked like a dataset
    # export that had hung. It is a distinct launch step and now says so.
    _set(run, phase_detail=_OFFER_SEARCH_DETAIL)

    def _search():
        return vast_client.search_offers(
            min_vram_gb=min_vram, max_dph=c.get('max_price_per_hour', 0.80),
            min_inet_down_mbps=int(c.get('min_inet_down_mbps') or 0),
            min_reliability=float(c.get('min_reliability') or 0.98),
            min_disk_bw_mbps=int(c.get('min_disk_bw_mbps') or 0),
            verified_only=bool(c.get('verified_only', True)),
            secure_cloud_only=bool(c.get('secure_cloud_only', False)),
            # Ask only machines that HAVE the disk this pod is about to claim —
            # a dense run asks for 200 GB and the market is full of 60 GB boxes.
            min_disk_gb=disk_gb)

    def _stamp(offer):
        # Stamp the host identity so a boot failure can blacklist THIS machine —
        # by its id AND by the address it answers on, since the id alone was
        # re-minted around a ban (see _blacklist_host). offer_ip is whatever the
        # offer advertised; host_ip (stamped during boot-wait, below) is the
        # address of the pod actually rented and is the one to trust.
        if offer.get('machine_id') is not None or _offer_ip(offer):
            if offer.get('machine_id') is not None:
                params['machine_id'] = offer['machine_id']
            if offer.get('host_id') is not None:
                params['host_id'] = offer['host_id']
            if _offer_ip(offer):
                params['offer_ip'] = _offer_ip(offer)
            _set(run, train_params=json.dumps(params))

    def _create(offer):
        nonlocal token
        if template_hash:
            # Preferred path (smoke-validated 2026-07-12): the official
            # template publishes the UI behind the pod's Caddy proxy on
            # ui_port and vast generates the per-instance auth token (picked
            # up from the instance record during boot-wait). HF_TOKEN reaches
            # the pod later via ensure_settings(), not env.
            token = ''
            return vast_client.create_instance(
                offer['offer_id'], disk_gb=disk_gb,
                label=run.vast_label, template_hash=template_hash,
                image=(c.get('image') or None))
        # Raw-image fallback (config escape hatch): direct port publish +
        # our own bearer token on the UI itself.
        token = pysecrets.token_urlsafe(24)
        port = int(c.get('ui_port') or 18675)
        env = {'AI_TOOLKIT_AUTH': token, f'-p {port}:{port}': '1'}
        hf = _hf_token_for_mode(params.get('training_mode') or 'lora')
        if hf:
            env['HF_TOKEN'] = hf
        return vast_client.create_instance(
            offer['offer_id'], disk_gb=disk_gb,
            label=run.vast_label, image=c.get('image'), env=env,
            onstart=(c.get('onstart') or None))

    instance_id, offer = rent_with_fresh_offers(
        search=_search, create=_create, on_offer=_stamp,
        pick=lambda offers: _pick_offer(offers, params.get('requested_gpu'),
                                        strict=bool(params.get('strict_gpu'))),
        no_offer_message=(
            f'no vast.ai offer matches (>= {min_vram} GB VRAM, >= {disk_gb} GB disk, '
            f'<= ${c.get("max_price_per_hour", 0.80)}/h) — raise the price cap in Settings'))
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


def _idle_seconds(run, now=None) -> float:
    """How long this run has been silent in the DATABASE — the MONITOR's
    heartbeat, and nothing more.

    Every monitor poll writes phase_detail through _set(), which bumps
    updated_at, so a frozen updated_at means the monitor stopped completing
    iterations (dead, wedged in a socket read, or gone with a restart). That
    makes this the right question for "can this thread still be trusted with a
    stop?" — and the WRONG one for "is the run getting anywhere?", which is
    what _silent_seconds answers: a monitor happily re-writing the same
    sentence every 10 s keeps this at zero forever. Do not merge the two."""
    now = now or datetime.utcnow()
    ref = run.updated_at or run.created_at or now
    return max(0.0, (now - ref).total_seconds())


# -- durable progress clock ---------------------------------------------------
# updated_at cannot answer "is this run getting anywhere?" for two independent
# reasons, both measured:
#  * it is re-stamped when the app RE-ADOPTS a run after a restart, so the
#    silence counter restarts with the process. Three restarts in one hour kept
#    a dead pod under the 45 min threshold for good (2026-07-28) — on the
#    machine of someone who tinkers, the watchdog is off by construction;
#  * it is bumped by the monitor's own writes, so a pod frozen at
#    'running: - fetching transformer weights' looks perfectly alive.
# The fix is to timestamp REMOTE evidence instead, and to keep that timestamp
# in the database (SystemState) so it outlives the process. The fingerprint is
# deliberately narrow: the run's phase, how many checkpoints landed, and the
# byte/step counters the pod printed. Not the raw log — tqdm re-prints the same
# bar with a bumped elapsed while the byte counter is frozen (measured: 1.95G
# at 15:11 and again at 15:30), so hashing the text would call a stuck download
# "progress". Not phase_detail either — re-adoption rewrites it, which is the
# very reset being fixed.
_PROGRESS_STATE_PREFIX = 'cloud_progress_watch:'


def _progress_state_key(run_id) -> str:
    return f'{_PROGRESS_STATE_PREFIX}{int(run_id)}'


def _log_tail(run, max_bytes=64 * 1024) -> str:
    """Tail of the run's mirrored pod log ('' when there is none). Bounded:
    this is read on every card render and every supervisor tick."""
    path = os.path.join(run.staging_dir or '', 'training.log')
    try:
        with open(path, 'rb') as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - max_bytes))
            return fh.read().decode('utf-8', errors='replace')
    except OSError:
        return ''


# The dataset upload's own durable evidence, written next to the mirrored pod
# log and for the same reason: the supervisor judges a run from files and
# database rows, never from the state of a thread it does not own. Until this
# existed, `uploading` produced NO observable evidence at all — run #138 pushed
# a 24 GB / 12 422-file dataset through 1 553 sequential POSTs while every
# watchdog input stayed frozen (see _freeze_limit_seconds).
_UPLOAD_PROGRESS_FILE = 'upload_progress.json'


def _upload_progress_path(run) -> str:
    return os.path.join(run.staging_dir or '', _UPLOAD_PROGRESS_FILE)


def _read_upload_bytes(run):
    """Bytes the dataset upload has pushed to the pod, or None when no upload
    has reported any. Never raises: the supervisor reads this every tick."""
    if not run.staging_dir:
        return None
    try:
        with open(_upload_progress_path(run), encoding='utf-8') as fh:
            return int(json.load(fh).get('bytes') or 0)
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _write_upload_progress(run, files, files_total, sent, total) -> None:
    """Record one upload observation. Never raises, for the same reason
    _set_soft exists: a progress write must not be able to sink the transfer
    it is only describing.

    A run with no staging_dir writes NOTHING rather than dropping the file in
    the process's working directory — a stray upload_progress.json there would
    be read back for every such run and make one run's bytes look like
    another's progress."""
    if not run.staging_dir:
        return
    try:
        with open(_upload_progress_path(run), 'w', encoding='utf-8') as fh:
            json.dump({'files': int(files), 'files_total': int(files_total),
                       'bytes': int(sent), 'bytes_total': int(total)}, fh)
    except (OSError, TypeError, ValueError):
        logger.debug('could not record upload progress for run %s',
                     getattr(run, 'id', '?'), exc_info=True)


def _record_uplink(run, folder, seconds) -> None:
    """File one measured upload speed away for the next forecast. Never raises:
    a transfer that LANDED must not become an error because a statistic about
    it could not be written.

    Filed as BULK, not as the line's throughput. A dataset is thousands of small
    files at eight per POST, so what this timed is dominated by per-request
    latency; a checkpoint push is one continuous stream. Letting this number
    forecast that one would describe neither."""
    try:
        total = 0
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                total += os.path.getsize(path)
        from . import pod_transfer_plan
        pod_transfer_plan.record_uplink_sample(
            total, seconds, kind=pod_transfer_plan.KIND_BULK)
    except Exception:
        logger.debug('run %s: uplink sample not recorded',
                     getattr(run, 'id', '?'), exc_info=True)


def _download_progress(run):
    """Byte-counter progress of whatever the pod is currently downloading, or
    None. Never raises: a card must never fail because a third-party bar
    changed shape."""
    try:
        return lt.parse_download_progress(_log_tail(run))
    except Exception:
        logger.debug('download progress parse failed for run %s', run.id)
        return None


def _progress_fingerprint(run) -> str:
    """What "something actually happened on the pod" means, as a short string.

    The download part is the SUM of every bar's bytes, not the last bar. Those
    are not the same reading: huggingface_hub fetches several files at once, so
    two consecutive tails end on different bars, and a fingerprint built from
    the last one alone flips A(1.0G) → B(2.0G) → A(1.0G) forever while BOTH
    files sit frozen. It would report movement on a dead pod — the freeze
    watchdog would then never fire on the one case it exists for. Summing per
    label means only a file that genuinely advanced can move the total.

    The upload part is the byte counter the dataset transfer writes as it goes.
    It belongs here for exactly the reason the download bytes do: it is the
    only reading that separates a 24 GB upload that is merely slow from one
    that has stopped moving. A restart restarts the upload from zero, which
    reads as a CHANGE (progress) — the same direction of error the paragraph
    above chooses, and the reason a re-adopted run is never killed on its
    predecessor's byte count.

    Changing this string re-anchors the clock once per open run on upgrade (an
    unseen fingerprint reads as progress). That costs one watchdog period on
    runs alive at that moment, and it errs toward NOT killing — the right side
    to be wrong on."""
    tail = _log_tail(run)
    parsed = {}
    try:
        parsed = lt._parse_training_log(tail) or {}
    except Exception:
        parsed = {}
    try:
        downloaded = lt.download_bytes_seen(tail)
    except Exception:
        downloaded = None
    uploaded = _read_upload_bytes(run)
    return '|'.join(str(x) for x in (
        run.status or '', _staging_save_count(run), parsed.get('step'),
        '' if downloaded is None else downloaded,
        '' if uploaded is None else uploaded))


def _read_progress_watch(run):
    row = db.session.get(SystemState, _progress_state_key(run.id))
    if row is None or not row.value:
        return None
    try:
        data = json.loads(row.value)
        return (data['fp'], datetime.fromisoformat(data['ts']))
    except (ValueError, KeyError, TypeError):
        return None


def note_progress(run, now=None) -> datetime:
    """Observe the run and return WHEN it last actually moved.

    Writes only when the fingerprint changed, so a frozen run does not touch
    the database at all (and a stuck run's own clock cannot be reset by the
    act of watching it). The first observation of a run seeds the timestamp
    with updated_at rather than `now`: at the first tick after a restart, the
    last thing the previous process wrote is a much better estimate of "last
    seen alive" than the instant the new process happened to start — seeding
    with `now` would re-create the very reset this exists to remove."""
    now = now or datetime.utcnow()
    fp = _progress_fingerprint(run)
    prev = _read_progress_watch(run)
    if prev and prev[0] == fp:
        return prev[1]
    ts = now if prev else min(run.updated_at or now, now)
    key = _progress_state_key(run.id)
    try:
        row = db.session.get(SystemState, key)
        if row is None:
            row = SystemState(key=key)
            db.session.add(row)
        row.value = json.dumps({'fp': fp, 'ts': ts.isoformat()})
        db.session.commit()
    except Exception:
        db.session.rollback()
        logger.debug('could not record the progress clock of run %s', run.id)
    return ts


def _clear_progress_watch(run_id):
    """Drop a finished run's progress clock — history rows never consult it."""
    try:
        row = db.session.get(SystemState, _progress_state_key(run_id))
        if row is not None:
            db.session.delete(row)
            db.session.commit()
    except Exception:
        db.session.rollback()


def _silent_seconds(run, now=None) -> float:
    """How long the run has made no OBSERVABLE progress. Read-only: falls back
    to _idle_seconds when nothing has been recorded yet (a run younger than the
    first supervisor tick), so this is never worse than what it replaces."""
    now = now or datetime.utcnow()
    prev = _read_progress_watch(run)
    if not prev:
        return _idle_seconds(run, now)
    return max(0.0, (now - prev[1]).total_seconds())


def _monitor_is_responsive(run) -> bool:
    """Can this run's monitor thread be TRUSTED to carry out a stop?

    Both halves matter. A registered thread object proves nothing (the run-103
    monitor was still alive, blocked forever inside one HTTP call), and a fresh
    updated_at alone would be satisfied by a monitor that has just died. Only a
    live thread that is also still writing gets the graceful path."""
    thread = _monitor_threads.get(int(run.id))
    if thread is None or not thread.is_alive():
        return False
    return _idle_seconds(run) <= STOP_HANDOFF_SECONDS


def _force_stop(run, detail, error=None) -> dict:
    """Terminate the pod HERE, without asking the monitor thread.

    The pod is the thing that costs money, and the vast API is the only
    authority on whether it is gone: a successful destroy closes the run as
    'stopped'; a refused or failing destroy must NEVER be reported as a
    success. In that case the run is parked in 'error_pod_kept' — the existing
    status meaning "a pod may still be alive out there" — so boot/launch
    reconciliation reaps it later, and the caller gets the instance id to
    destroy by hand in the meantime."""
    iid = run.vast_instance_id
    _stop_event_for(run.id).set()   # a still-living monitor stands down too
    _clear_progress_watch(run.id)   # every path below closes the run
    if not iid:
        _set(run, status='stopped', phase_detail=detail,
             error=error, finished_at=datetime.utcnow())
        return {'ok': True, 'run_id': run.id, 'mode': 'forced',
                'message': detail, 'instance_id': None}
    gone = False
    failure = ''
    try:
        gone = bool(vast_client.destroy_instance(iid))
        if not gone:
            failure = 'the vast.ai API refused the termination'
    except Exception as e:
        failure = str(e)[:200]
        logger.warning('forced stop of run %s: destroy %s failed: %s',
                       run.id, iid, failure)
    if gone:
        _set(run, status='stopped', phase_detail=detail,
             error=error, finished_at=datetime.utcnow())
        logger.warning('forced stop of run %s: pod %s terminated (%s)',
                       run.id, iid, error or detail)
        return {'ok': True, 'run_id': run.id, 'mode': 'forced',
                'message': detail, 'instance_id': iid}
    message = (f'Could not terminate instance {iid} ({failure}). It may still '
               f'be running and billing — destroy it in the vast.ai console.')
    _set(run, status='error_pod_kept', phase_detail=detail[:500],
         error=message, finished_at=datetime.utcnow())
    return {'ok': False, 'run_id': run.id, 'mode': 'failed',
            'error': message, 'instance_id': iid}


def _stop_one(run) -> dict:
    # Decide BEFORE writing anything: stamping stop_requested_at bumps
    # updated_at, which would make a frozen run look freshly alive.
    responsive = _monitor_is_responsive(run)
    _stop_event_for(run.id).set()
    if not run.stop_requested_at:
        _set(run, stop_requested_at=datetime.utcnow())
    if responsive:
        # Graceful: the monitor stops the remote job and rescues the latest
        # checkpoint before terminating. The stamped stop_requested_at arms the
        # supervisor's deadline in case it wedges on the way.
        return {'ok': True, 'run_id': run.id, 'mode': 'graceful',
                'message': 'Stopping the run — the pod is winding down…',
                'instance_id': run.vast_instance_id}
    return _force_stop(
        run,
        detail='Stopped by user — the run monitor was not responding, so the '
               'pod was terminated directly (checkpoints already downloaded '
               'are kept)',
        error='stopped by user without a responsive monitor')


def request_stop(run_id=None) -> dict:
    """Stop one run (or every active run when run_id is None) and report what
    ACTUALLY happened.

    Historically this only set an in-process threading.Event and returned True
    as long as the row was active — so when the monitor thread was dead or
    wedged, the button answered "ok" and the pod kept billing for hours
    (incident 2026-07-25). A stop now either terminates the pod or says it
    could not, naming the instance."""
    if run_id is not None:
        run = CloudTrainingRun.query.get(int(run_id))
        runs = [run] if run and run.status in ACTIVE_STATES else []
    else:
        runs = get_active_runs()
    if not runs:
        return {'ok': False, 'mode': 'none', 'runs': [],
                'error': 'No active cloud run to stop — it may have already '
                         'finished.'}
    results = [_stop_one(run) for run in runs]
    failed = [r for r in results if not r['ok']]
    modes = {r['mode'] for r in results}
    return {'ok': not failed,
            'mode': modes.pop() if len(modes) == 1 else 'mixed',
            'runs': results,
            'message': results[0].get('message', ''),
            'error': failed[0]['error'] if failed else None}


def supervise_active_runs() -> list:
    """One supervisor tick: enforce, from OUTSIDE any monitor thread, the
    guarantees a monitor can no longer make once it is dead or wedged.

    Three rules, all anchored on durable database state:
      * runtime cap  — the configured ceiling used to be a deadline computed
        inside the monitor itself, so the net died with what it protected;
      * stop deadline — a stop handed to a monitor that never carries it out
        (a monitor still streaming the checkpoint down is exempt while it
        keeps writing — see _rescuing_checkpoint);
      * freeze watchdog — no database progress for longer than the phase
        allows (see _freeze_limit_seconds).
    A margin is deliberately left on the first two so a HEALTHY monitor always
    gets to act first: its own paths rescue the last checkpoint from the pod,
    while a forced stop can only keep what mid-run mirroring already pulled.
    Never raises — the whole point is a net that cannot die."""
    acted = []
    try:
        c = cfg.get('cloud') or {}
        max_seconds = int(c.get('max_runtime_minutes') or 480) * 60
        now = datetime.utcnow()
        for run in get_active_runs():
            try:
                age = (now - (run.created_at or now)).total_seconds()
                if age > max_seconds + _SUPERVISOR_MARGIN_SECONDS \
                        and not _rescuing_checkpoint(run, now):
                    # ... unless the monitor is, right now, pulling the result
                    # off the pod and still writing while it does. The cap is
                    # there so a run stops COSTING, and cutting the rescue would
                    # spend the whole run and keep nothing — the same reasoning
                    # the stop deadline already applies, and it matters far more
                    # for a 26 GB dense master than for an 85 MB LoRA.
                    res = _force_stop(
                        run, detail='Max runtime reached — pod terminated by '
                                    'the supervisor', error='max runtime cap hit')
                    acted.append({'run_id': run.id, 'reason': 'runtime_cap',
                                  'ok': res['ok']})
                    continue
                stop_age = ((now - run.stop_requested_at).total_seconds()
                            if run.stop_requested_at else 0)
                if stop_age > STOP_DEADLINE_SECONDS \
                        and not _rescuing_checkpoint(run, now):
                    res = _force_stop(
                        run, detail='Stopped by user — the run monitor never '
                                    'completed the stop, so the pod was '
                                    'terminated by the supervisor',
                        error='stop request not honoured in time')
                    acted.append({'run_id': run.id, 'reason': 'stop_deadline',
                                  'ok': res['ok']})
                    continue
                # The progress clock is advanced HERE, from outside every
                # monitor: the tick that judges the run is also the one that
                # observes it, so the watchdog cannot be starved by a monitor
                # that stopped looking.
                note_progress(run, now)
                limit = _freeze_limit_seconds(run, c)
                if limit and _silent_seconds(run, now) > limit:
                    if (_is_full_transformer_run(run)
                            # 'downloading' joined 'training' when the dense
                            # master started coming home: a monitor that dies
                            # mid-harvest leaves the ONLY copy on the pod, and
                            # destroying it there would be the loss this whole
                            # lane exists to prevent.
                            and run.status in ('training', 'downloading')
                            and run.remote_job_id):
                        _keep_full_transformer_pod(
                            run,
                            detail=(f'Frozen — no progress for {limit // 60} min; '
                                    'remote job stopped if possible and pod kept '
                                    'for dense-checkpoint recovery'),
                            error='freeze watchdog; dense pod kept',
                            stop_remote=True)
                        acted.append({'run_id': run.id, 'reason': 'freeze',
                                      'ok': True, 'pod_kept': True})
                    elif run.status == 'uploading':
                        # Nothing has been trained and nothing is on the pod
                        # worth keeping, so this is a plain teardown — but it
                        # gets its OWN error string: 'the dataset never
                        # reached the machine' and 'the run froze' send the
                        # user to completely different places, and only the
                        # first one is about their upload.
                        res = _force_stop(
                            run,
                            detail=('Dataset upload stalled — nothing reached '
                                    f'the pod for {limit // 60} min; pod '
                                    'terminated by the supervisor'),
                            error='upload stall watchdog')
                        acted.append({'run_id': run.id, 'reason': 'upload_stall',
                                      'ok': res['ok']})
                    else:
                        res = _force_stop(
                            run,
                            detail=f'Frozen — no progress for {limit // 60} min; '
                                   'pod terminated by the supervisor',
                            error='freeze watchdog')
                        acted.append({'run_id': run.id, 'reason': 'freeze',
                                      'ok': res['ok']})
            except Exception:
                logger.exception('supervisor: run %s could not be judged', run.id)
    except Exception:
        logger.exception('cloud supervisor tick failed')
    return acted


def _rescuing_checkpoint(run, now=None) -> bool:
    """Is this run, right now, pulling its checkpoint off the pod — and still
    writing while it does?

    The stop deadline exists for a monitor that WEDGED after being handed a
    stop. A monitor that is downloading the result is the opposite: it is doing
    the single most valuable part of the stop, and cutting it there throws away
    a checkpoint the user already paid for. The exemption is deliberately
    narrow — it needs the 'downloading' status AND a row written inside the
    handoff window (the transfer heartbeats far more often than that), so a
    monitor that dies mid-transfer stops being spared within a couple of
    minutes and falls back to the freeze watchdog and the runtime cap."""
    return (run.status == 'downloading'
            and _idle_seconds(run, now) <= STOP_HANDOFF_SECONDS)


def _freeze_limit_seconds(run, c=None) -> int:
    """Seconds of database silence tolerated in the run's CURRENT phase (0 =
    watchdog off).

    Only 'training' is judged on the configured value: there the monitor writes
    phase_detail on every poll (~10 s), so silence is unambiguous. Every other
    phase is silent by design for long stretches — staging a big dataset,
    renting and booting a pod, pulling the final checkpoint — and killing a run
    that is merely starting up would be worse than the leak we are closing.
    They get a fixed, very generous floor; the runtime cap remains their real
    backstop.

    'uploading' left that group on 2026-08-02. It was the worst of both: a
    phase that can run for hours AND the one with no evidence of its own, so
    the two-hour floor was the only thing standing between a wedged transfer
    and a pod billing until the runtime cap. Run #138 spent 2 h 07 there — 93
    min of it in total database silence — pushing a 24 GB dataset that never
    reached the pod, and was still under the floor when its owner cancelled by
    hand. Now that the transfer reports bytes, silence in this phase means
    'nothing arrived', which deserves a far shorter answer than 'nobody has
    written a row'."""
    c = c if c is not None else (cfg.get('cloud') or {})
    raw = c.get('freeze_watchdog_minutes')
    minutes = _FREEZE_WATCHDOG_MINUTES if raw is None else int(raw or 0)
    if minutes <= 0:
        # The watchdog is off by explicit configuration; the upload's shorter
        # limit is a tightening of it, never a way around it.
        return 0
    if run.status == 'training':
        return minutes * 60
    if run.status == 'uploading':
        raw_upload = c.get('upload_stall_minutes')
        upload_minutes = (_UPLOAD_STALL_MINUTES if raw_upload is None
                          else int(raw_upload or 0))
        return max(0, upload_minutes * 60)
    return max(minutes * 60, _SILENT_PHASE_FREEZE_SECONDS)


def _supervisor_tick(app, *, reap_orphans=False):
    """Run one watchdog pass; optionally perform the throttled account reap."""
    with app.app_context():
        supervise_active_runs()
        reconcile_full_transformer_deliveries()
    if reap_orphans:
        # This helper owns its own app context and never raises.
        reconcile_orphans(app)


def _supervisor_loop(app):
    last_orphan_reconcile = None
    while True:
        try:
            now = time.monotonic()
            reap_orphans = (
                last_orphan_reconcile is None
                or now - last_orphan_reconcile
                >= _ORPHAN_RECONCILE_INTERVAL_SECONDS)
            _supervisor_tick(app, reap_orphans=reap_orphans)
            if reap_orphans:
                last_orphan_reconcile = now
        except Exception:
            logger.exception('cloud supervisor loop failed')
        _sleep(SUPERVISOR_INTERVAL_SECONDS)


def start_supervisor(app):
    """Start the single watchdog thread (idempotent). Deliberately independent
    of boot_recover and of every per-run monitor: it owns nothing, blocks on
    nothing but its own sleep, and therefore survives what they cannot."""
    global _supervisor_thread
    if _supervisor_thread is not None and _supervisor_thread.is_alive():
        return _supervisor_thread
    _supervisor_thread = threading.Thread(
        target=_supervisor_loop, args=(app,), daemon=True, name='cloud-supervisor')
    _supervisor_thread.start()
    return _supervisor_thread


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
    cloud.max_runtime_minutes of now, and only reaped past that window. A dense
    artifact already verified becomes ``done`` when that reap confirms cleanup;
    unverified/manual-recovery rows remain terminal and are annotated."""
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
            # Absence is useful cleanup evidence only when the account listing
            # itself is complete and structurally inspectable.  Never turn a
            # partial/malformed response into a false "pod is gone" result.
            if (not isinstance(instances, list)
                    or any(not isinstance(inst, dict)
                           or inst.get('instance_id') in (None, '')
                           for inst in instances)):
                logger.warning(
                    'reconcile: vast instance listing is not safely inspectable')
                return 0
            live_instance_ids = {
                str(inst['instance_id']) for inst in instances}
            keep = {str(r.vast_instance_id) for r in get_active_runs() if r.vast_instance_id}
            c = cfg.get('cloud') or {}
            max_seconds = int(c.get('max_runtime_minutes') or 480) * 60
            now = datetime.utcnow()
            kept_by_instance = {
                str(r.vast_instance_id): r
                for r in CloudTrainingRun.query.filter_by(status='error_pod_kept').all()
                if r.vast_instance_id}

            # A DELETE may have succeeded while its response or the following
            # DB commit failed.  A successful full account listing that no
            # longer contains the owned instance is authoritative confirmation
            # that billing cleanup is complete.  This is intentionally limited
            # to already-verified dense artifacts whose only pending concern is
            # cleanup; unverified/manual-recovery runs keep their old policy.
            for iid, kept_run in kept_by_instance.items():
                if iid in live_instance_ids:
                    continue
                if (_is_full_transformer_run(kept_run)
                        and _run_param(kept_run, 'artifact_status') == 'available'
                        and _run_param(
                            kept_run, 'artifact_cleanup_status') != 'complete'):
                    _mark_verified_full_transformer_cleanup_complete(
                        kept_run,
                        cleanup_detail=(
                            'vast.ai account listing confirmed the pod is absent'),
                        phase_detail=(
                            'Dense checkpoint available — pod absence confirmed'))
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
                            if (_is_full_transformer_run(kept_run)
                                    and _run_param(
                                        kept_run, 'artifact_status') == 'available'):
                                _mark_verified_full_transformer_cleanup_complete(
                                    kept_run,
                                    cleanup_detail=(
                                        'vast.ai pod termination confirmed by '
                                        'expired-run reconciliation'),
                                    phase_detail=(
                                        'Dense checkpoint available — expired '
                                        'pod cleanup confirmed'))
                            else:
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


# --- Boot evidence (2026-07-28) ------------------------------------------------
# The boot wait used to read the pod's remote state and spend it on the phase
# line only, then decide on elapsed time alone: a host honestly pulling a 26 GB
# image died at 25 min and was exiled from the marketplace for three days. Same
# guiding rule as the first-step watchdog: a pod whose remote evidence advances
# is a pod that progresses.
#
# "Advances" is the whole difficulty. A boot fact that keeps being TRUE is not
# progress (a pod frozen in 'loading' reports 'loading' forever), and a line
# whose only moving part is its clock is not progress either. So evidence is
# modelled as a growing SET of facts: only a fact never observed before rearms
# the clock, which makes a frozen pod rearm exactly zero times.
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
# Clock-ish runs (12:34, 1:02:03, 2026-07-28T10:11) are stripped before a host
# progress line is compared: an advancing elapsed time is the exact false
# evidence that nearly got shipped into the first-step watchdog.
_CLOCKISH_RE = re.compile(r'\d{1,2}:\d{2}(?::\d{2})?'
                          r'|\d{4}-\d{2}-\d{2}[t ]?[\d:.]*', re.I)


def _boot_status_message(inst) -> str:
    """The host's free-text boot progress line, stripped of colour codes and of
    anything that only measures time. Empty when vast publishes nothing — the
    field is optional, so this is a bonus signal, never a requirement."""
    raw = (inst or {}).get('status_msg')
    if not isinstance(raw, str) or not raw.strip():
        return ''
    text = _CLOCKISH_RE.sub('', _ANSI_RE.sub('', raw))
    return ' '.join(text.split())[:200]


def _boot_facts(inst, port, base) -> set:
    """Everything the pod is provably showing RIGHT NOW, as comparable facts."""
    inst = inst or {}
    facts = {'status:' + str(inst.get('actual_status') or 'unlisted')}
    if ((inst.get('ports') or {}).get(f'{port}/tcp')):
        facts.add('port-published')
    if base:
        facts.add('base-url')
    if inst.get('jupyter_token'):
        facts.add('auth-token')
    msg = _boot_status_message(inst)
    if msg:
        facts.add('msg:' + msg)
    return facts


def _boot_stage_label(inst, port, base) -> str:
    """What was MEASURED about this boot, for the failure message. 'Pod never
    became ready in 25 min' is a guess; where it actually got to is not."""
    inst = inst or {}
    bits = ['vast status "%s"' % (inst.get('actual_status') or 'not listed yet')]
    bits.append(f'port {port} '
                + ('published' if ((inst.get('ports') or {}).get(f'{port}/tcp'))
                   else 'not published yet'))
    if base:
        bits.append('UI not answering')
    msg = _boot_status_message(inst)
    if msg:
        bits.append(f'host reported "{msg[:120]}"')
    return ', '.join(bits)


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
    if (run_params or {}).get('training_mode') == 'full_transformer':
        delivery = dld.run_mode(run_params or {})
        repo_id = (run_params or {}).get('hf_repo_id')
        if proc.get('network') is not None:
            raise RuntimeError('full_transformer job unexpectedly contains a '
                               'LoRA network block')
        if dld.delivers_local(delivery):
            # The trainer pushes NOTHING. This is the change run #146 paid for:
            # its 403 arrived on a mid-training push at step 2750/3000, and a
            # storage ceiling on a service we do not control must never be able
            # to end a training. The saves stay on the pod and are harvested at
            # the end; the Hub backup, when asked for, is a separate step taken
            # AFTER the local copy is proven.
            proc['save'] = dict(proc.get('save') or {})
        else:
            if not repo_id:
                raise RuntimeError('full_transformer run has no Hugging Face '
                                   'delivery repository')
            save = dict(proc.get('save') or {})
            save.update({'push_to_hub': True,
                         'hf_repo_id': repo_id,
                         'hf_private': True})
            proc['save'] = save
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
    _clear_progress_watch(run.id)
    return pod_gone


class _RunClosedExternally(Exception):
    """The run row left ACTIVE_STATES while this monitor was working — a forced
    stop or the supervisor closed it. The monitor must stand down instead of
    resurrecting the row (or renting a pod for a run nobody waits for)."""


class _WaitAborted(RuntimeError):
    """A user Stop fired while ``_with_frozen_dataset_generation`` was
    retrying a busy export lease (another run's export can take minutes).
    Carried as its own type so the monitor can land the run as 'stopped',
    the same treatment the boot-wait's Stop check gets — the generic
    ``except Exception`` below would otherwise report this as a red 'Run
    failed', which is not what happened."""


class _ReattachFailed(RuntimeError):
    """A pod whose job was ALREADY running could not be reached again after an
    app restart, for the whole reconnect window.

    Carried as its own type for one reason: there is nothing to stop. A job we
    could not contact for minutes cannot receive a stop request, and a job we
    DID contact never produces this failure — so the recovery path must not
    send one. Run #146 (2026-08-03) died the other way round: the run was
    condemned on a vast listing gap while the pod was perfectly reachable, and
    the 'best effort' stop that followed reached the live trainer and killed
    825 healthy steps."""


def _assert_run_open(run):
    db.session.refresh(run)     # another thread may have committed a close
    if run.status not in ACTIVE_STATES:
        raise _RunClosedExternally(run.status)


def _finish_if_open(run, status, detail='', error=None, destroy=True):
    """_finish(), but only for a run that is still ours to close.

    Checking ONCE at the top of the poll loop is not enough. Every terminal
    branch does minutes of work after that check — stopping the remote job,
    pulling the final checkpoint, importing it, mirroring it locally — and the
    supervisor is a different thread on a different session: it can force-stop
    the run, destroy the pod and write the row inside that window. The monitor
    would then rewrite a closed row and announce a pod it 'kept' that no longer
    exists. Re-asserting immediately before the write means a run closed behind
    our back raises _RunClosedExternally and takes the stand-down path instead
    (the work done up to here — the downloaded checkpoint — is kept on disk)."""
    _assert_run_open(run)
    return _finish(run, status, detail=detail, error=error, destroy=destroy)


def _stop_remote_job_best_effort(run):
    """Freeze a recoverable dense job without risking pod destruction."""
    if not run.remote_job_id or not run.base_url:
        return False
    try:
        _make_remote(run).stop_job(run.remote_job_id)
        return True
    except Exception:
        # Never interpolate an authenticated remote exception into logs.
        logger.warning('run %s: could not stop dense remote job before keeping pod',
                       run.id)
        return False


def _ensure_remote_settings_without_secret(run, remote) -> dict:
    """Configure pod auth, then immediately discard the echoed credential."""
    raw = remote.ensure_settings(hf_token=_hf_token_for_run(run))
    if not isinstance(raw, dict):
        raise RuntimeError('remote settings response is invalid')
    settings = dict(raw)
    # RemoteAiToolkit returns the freshly saved settings, including HF_TOKEN.
    # Training config construction only needs the folder paths; carrying the
    # credential forward makes accidental JSON/log persistence possible.
    settings.pop('HF_TOKEN', None)
    return settings


def _redacted_error_text(error) -> str:
    """Persist an actionable error without ever echoing an HF credential."""
    value = str(error or '')
    for key in ('HF_CLOUD_TOKEN', 'HF_TOKEN'):
        token = cfg.secret(key)
        if token:
            value = value.replace(token, '[redacted]')
    # Defense in depth for SDK messages that render a token unknown to the
    # current process (for example, a token rotated while the pod was alive).
    value = re.sub(r'\bhf_[A-Za-z0-9_-]{8,}\b', '[redacted]', value)
    value = re.sub(r'(?i)(authorization\s*:\s*bearer\s+)\S+',
                   r'\1[redacted]', value)
    return value[:500]


def _keep_full_transformer_pod(run, detail, error, *, stop_remote=False,
                               require_open=True):
    """Close a dense run as recoverable while preserving the paid pod."""
    if require_open:
        _assert_run_open(run)
    if stop_remote:
        _stop_remote_job_best_effort(run)
    if run.status in ACTIVE_STATES:
        _stop_event_for(run.id).set()
    return _finish(run, 'error_pod_kept', detail=detail, error=error,
                   destroy=False)


def _mark_verified_full_transformer_cleanup_complete(
        run, *, cleanup_detail, phase_detail):
    """Atomically expose completion after compute cleanup is proven."""
    if run.status in ACTIVE_STATES:
        _stop_event_for(run.id).set()
    _clear_progress_watch(run.id)
    params = _updated_artifact_params(
        run, 'available', artifact_cleanup_status='complete',
        artifact_cleanup_detail=cleanup_detail,
        artifact_status_detail=(
            'Dense checkpoint and compliance metadata verified; pod '
            'termination confirmed'))
    _set(
        run, status='done', phase_detail=phase_detail, error=None,
        finished_at=datetime.utcnow(), train_params=json.dumps(params))


def _destroy_dense_pod(run) -> bool:
    """Terminate a dense run's pod and say — truthfully — whether it is gone.

    Shared by both dense finalizers (a verified Hugging Face delivery, and a
    verified local one) because the rule they enforce is the same one: the pod
    is the thing that costs money, the vast API is the only authority on
    whether it is gone, and an ambiguous answer must never be published as a
    clean ending.
    """
    instance_id = run.vast_instance_id
    if not instance_id:
        return True
    try:
        pod_gone = bool(vast_client.destroy_instance(instance_id))
        if not pod_gone:
            logger.warning(
                'run %s: verified dense delivery, but pod termination was '
                'not confirmed', run.id)
        return pod_gone
    except Exception:
        # Never interpolate authenticated vast.ai diagnostics.
        logger.warning(
            'run %s: verified dense delivery pod termination raised; '
            'cleanup remains pending', run.id)
        return False


def _finalize_verified_full_transformer(run, *, require_open=False) -> bool:
    """Destroy the paid pod first; publish ``done`` only after confirmation.

    Artifact availability and compute cleanup are separate facts.  A failed or
    ambiguous vast.ai termination must leave the verified model visible while
    keeping the run retryable, otherwise reconciliation would ignore a billing
    pod merely because Hugging Face delivery succeeded.
    """
    if require_open:
        _assert_run_open(run)
    instance_id = run.vast_instance_id
    pod_gone = _destroy_dense_pod(run)

    if pod_gone:
        _mark_verified_full_transformer_cleanup_complete(
            run, cleanup_detail='vast.ai pod termination confirmed',
            phase_detail=(
                'Training complete — dense checkpoint available on Hugging Face'))
        return True

    if run.status in ACTIVE_STATES:
        _stop_event_for(run.id).set()
    _clear_progress_watch(run.id)
    params = _updated_artifact_params(
        run, 'available', artifact_cleanup_status='pending',
        artifact_cleanup_detail=(
            'Dense model is available; vast.ai pod termination is not yet '
            'confirmed and will be retried automatically'),
        artifact_status_detail=(
            'Dense checkpoint and compliance metadata verified; pod cleanup '
            'pending'))
    _set(
        run, status='error_pod_kept',
        phase_detail='Dense checkpoint available — pod cleanup pending',
        error=(
            f'Dense model is available on Hugging Face, but termination of '
            f'instance {instance_id} was not confirmed. Cleanup will retry '
            'automatically; the pod may still be billing.'),
        # The first failure starts the bounded recovery window; retries must not
        # extend it indefinitely.
        finished_at=run.finished_at or datetime.utcnow(),
        train_params=json.dumps(params))
    return False


def _complete_full_transformer_delivery(run, _api=None):
    """Only verified weights+metadata may become done and destroy the pod."""
    delivery = _verify_full_transformer_artifact_with_retries(run, _api=_api)
    if delivery == 'available':
        return _finalize_verified_full_transformer(run, require_open=True)
    if delivery == 'missing':
        detail = ('Training complete — compliant Krea dense checkpoint missing; '
                  'pod kept')
        error = ('Hugging Face repository contains no checkpoint matching the '
                 'Krea job name; pod kept for recovery')
    else:
        detail = ('Training complete — Hugging Face delivery/compliance could '
                  'not be verified; pod kept')
        error = ('Hugging Face verification remained unavailable after bounded '
                 'retries; pod kept for recovery')
    return _keep_full_transformer_pod(run, detail, error)


# ── Dense delivery to THIS computer ───────────────────────────────────────────
# The pod already knows how to hand its files back — it is how every LoRA
# checkpoint reaches its user. Dense was excluded from that path by a handful of
# early returns, not by anything technical. What follows is the dense half of
# the same harvest, with two rules the LoRA lane does not need:
#   * a 26 GB transfer must survive an app restart (resume from .part) and be
#     interruptible (should_cancel), because restarting it is an evening;
#   * the pod is destroyed only after the local file is PROVEN — size, then a
#     real safetensors header read. Destroying it before that is how a run is
#     lost, and the whole point of bringing the file home is not to lose runs.

def _dense_local_dir(run) -> str:
    return checkpoint_store_dir(run, create=True) or run.staging_dir


def _dense_free_space_error(entries) -> str | None:
    """Refuse to start a transfer the disk plainly cannot hold. Best-effort: a
    volume we cannot measure never blocks (same rule as every other forecast)."""
    from . import storage_locations
    need = sum(int((e or {}).get('size') or 0) for e in entries)
    if not need:
        return None
    root = str(cfg.checkpoints_root(create=False))
    info = storage_locations.free_space(root) or {}
    free = info.get('free_bytes')
    if not isinstance(free, int) or isinstance(free, bool):
        return None
    need += dld.disk_margin_bytes()
    if free >= need:
        return None
    from .hf_storage import fmt_bytes
    # No machine path in the sentence: it lands in run.error, which is exactly
    # the text people paste into a bug report.
    return (f'not enough room on this computer: the full model needs '
            f'{fmt_bytes(need)} in the checkpoint folder and {fmt_bytes(free)} '
            f'is free. Free space (or point it at a bigger drive in '
            f'Settings ▸ Storage) and fetch it again — the pod is kept '
            f'until then.')


def _harvest_dense_artifacts(run, remote, should_cancel=None) -> dict:
    """Download this dense run's files from the pod and prove them.

    Returns ``{'ok', 'detail', 'error', 'cancelled', 'master', 'files'}`` and
    never raises: every caller has a pod to keep alive on failure.
    """
    out = {'ok': False, 'detail': '', 'error': None, 'cancelled': False,
           'master': None, 'files': []}
    try:
        files = remote.list_files(run.remote_job_id)
    except Exception as e:
        out['error'] = (f'the pod would not list its files ({e}); nothing was '
                        'downloaded and the pod is kept')
        return out
    keep_bf16 = _run_param(run, 'fp8_keep_bf16') is not False
    picked = dld.select_pod_artifacts(files, keep_bf16=keep_bf16)
    wanted = [entry for entry in (picked['master'], picked['fp8']) if entry]
    if not wanted:
        out['error'] = ('the pod holds no full-model checkpoint to fetch — '
                        'check the run log before deleting anything')
        return out
    room = _dense_free_space_error(wanted)
    if room:
        out['error'] = room
        return out
    proofs = []
    for kind, entry in (('master', picked['master']), ('fp8', picked['fp8'])):
        if not entry:
            continue
        name = os.path.basename(str(entry['path']).replace('\\', '/'))
        label = 'full model' if kind == 'master' else 'fp8 file'
        _set(run, status='downloading',
             phase_detail=f'Fetching the {label} to this computer — {name}'[:500])
        try:
            dest = _fetch_checkpoint(
                run, remote, entry, attempts=_DENSE_FETCH_ATTEMPTS,
                on_progress=_download_heartbeat(run, name),
                resume=True, should_cancel=should_cancel)
            proof = dld.verify_local_file(dest, entry.get('size'))
        except TransferCancelled as e:
            out['cancelled'] = True
            out['error'] = str(e)
            out['detail'] = ('Transfer stopped — what was already downloaded is '
                             'kept and the next attempt continues from there')
            return out
        except Exception as e:
            out['error'] = (f'the {label} could not be brought home ({e}); the '
                            'pod is kept so it can be fetched again')
            return out
        proof['kind'] = kind
        proofs.append(proof)
        if kind == 'master':
            out['master'] = entry
            # Point the run at the master: every disk-accounting and cleanup
            # path reads this, and a weight nothing points at is a weight the
            # next 'clean finished runs' is free to consider spare.
            _set(run, checkpoint_local_path=proof['path'])
    out['files'] = proofs
    master = next((p for p in proofs if p['kind'] == 'master'), None)
    fp8 = next((p for p in proofs if p['kind'] == 'fp8'), None)
    _persist_run_params(
        run,
        local_artifact_status='available',
        local_artifact_dir=_dense_local_dir(run),
        local_weight_filename=(master or {}).get('name'),
        local_weight_bytes=(master or {}).get('size_bytes'),
        local_fp8_filename=(fp8 or {}).get('name'),
        local_fp8_bytes=(fp8 or {}).get('size_bytes'),
        local_verified_at=datetime.utcnow().isoformat(),
        local_artifact_detail=('Downloaded from the pod and verified '
                               '(byte count and safetensors header).'))
    out['ok'] = True
    names = ' + '.join(p['name'] for p in proofs)
    out['detail'] = f'Full model on this computer — {names}'
    return out


def _dense_hub_backup(run, remote, master_entry) -> dict:
    """Best-effort Hugging Face copy of the master, pushed BY THE POD.

    Runs only once the local copy is verified, which is what makes it safe to
    be best-effort: a refused push (a full private quota — the exact wall run
    #146 hit) costs the ability to CONTINUE this model later, and nothing else.
    The fp8 twin is deliberately not pushed: it is regenerated from the master
    in seconds and would consume the quota twice as fast.
    """
    from . import dense_pod_hub
    repo_id = _run_param(run, 'hf_repo_id')
    path = str((master_entry or {}).get('path') or '')
    if not repo_id:
        outcome = {'state': 'skipped',
                   'detail': 'No Hugging Face copy was requested for this run.'}
    elif not path:
        # "Keep the bf16 master" is off, so there is no master to back up — and
        # the fp8 twin is deliberately never pushed. Say which, because the
        # consequence (this run cannot be continued) is the same either way.
        outcome = {'state': 'skipped',
                   'detail': ('No Hugging Face backup: this run kept only its '
                              'fp8 file, and a quantized twin cannot be trained '
                              'again. Keep the bf16 master to stay resumable.')}
    else:
        dense = ((cfg.get('cloud') or {}).get('full_transformer') or {})
        outcome = dense_pod_hub.push_master(
            remote, instance_id=run.vast_instance_id, src_path=path,
            repo_id=repo_id, path_in_repo=os.path.basename(path),
            hf_token=cfg.secret('HF_CLOUD_TOKEN'),
            tmp_dir=run.staging_dir or _dense_local_dir(run),
            budget_seconds=int(dense.get('hub_push_budget_seconds')
                               or dense_pod_hub.DEFAULT_PUSH_BUDGET_SECONDS),
            on_state=lambda detail: _set_soft(run, phase_detail=detail[:500]))
    try:
        _persist_run_params(run, hub_backup_status=outcome['state'],
                            hub_backup_detail=outcome['detail'])
    except Exception:
        logger.debug('could not stamp the hub backup state of run %s', run.id)
    if outcome['state'] == 'done':
        # Prove what landed with the reader that already exists — repo metadata
        # only, never a 26 GB download.
        try:
            _verify_full_transformer_artifact(run)
        except Exception:
            logger.warning('run %s: hub backup verification unavailable', run.id)
    else:
        _persist_artifact_state(
            run, 'missing' if repo_id else 'not_requested',
            artifact_status_detail=outcome['detail'])
    return outcome


def _finalize_dense_local_delivery(run, *, require_open=False) -> bool:
    """Release the pod once the local copy is proven, and publish the result.

    Same contract as its Hugging Face twin: an ambiguous termination leaves the
    (already verified, already local) model visible while the run stays
    recoverable, so a billing pod is never hidden by a successful delivery.
    """
    if require_open:
        _assert_run_open(run)
    instance_id = run.vast_instance_id
    pod_gone = _destroy_dense_pod(run)
    detail = _run_param(run, 'local_artifact_detail') or 'Full model downloaded'
    if pod_gone:
        if run.status in ACTIVE_STATES:
            _stop_event_for(run.id).set()
        _clear_progress_watch(run.id)
        params = _updated_artifact_params(
            run, _run_param(run, 'artifact_status') or 'not_requested',
            artifact_cleanup_status='complete',
            artifact_cleanup_detail='vast.ai pod termination confirmed')
        _set(run, status='done', error=None, finished_at=datetime.utcnow(),
             phase_detail=('Training complete — full model on this computer '
                           f'({_run_param(run, "local_weight_filename") or "verified"})')[:500],
             train_params=json.dumps(params))
        return True
    if run.status in ACTIVE_STATES:
        _stop_event_for(run.id).set()
    _clear_progress_watch(run.id)
    params = _updated_artifact_params(
        run, _run_param(run, 'artifact_status') or 'not_requested',
        artifact_cleanup_status='pending',
        artifact_cleanup_detail=(
            'The full model is on this computer; vast.ai pod termination is '
            'not yet confirmed and will be retried automatically'))
    _set(run, status='error_pod_kept',
         phase_detail=f'{detail} — pod cleanup pending'[:500],
         error=(f'The full model is on this computer, but termination of '
                f'instance {instance_id} was not confirmed. Cleanup will retry '
                'automatically; the pod may still be billing.'),
         finished_at=run.finished_at or datetime.utcnow(),
         train_params=json.dumps(params))
    return False


def _deliver_dense_locally(run, remote, *, should_cancel=None,
                           require_open=False) -> bool:
    """Harvest → prove → (optional) Hub backup → release the pod. False keeps it."""
    report = _harvest_dense_artifacts(run, remote, should_cancel=should_cancel)
    if not report['ok']:
        # Stamp WHY on the run, not only in the error line: the delivery card
        # reads this, and "not downloaded" without a reason is what sends
        # someone hunting a fault that a sentence would have named.
        try:
            _persist_run_params(
                run, local_artifact_status=('cancelled' if report['cancelled']
                                            else 'failed'),
                local_artifact_detail=(report['detail'] or report['error']))
        except Exception:
            logger.debug('could not stamp the failed local delivery of run %s',
                         run.id)
        _keep_full_transformer_pod(
            run,
            detail=(report['detail'] or 'Full model not downloaded; pod kept'),
            error=report['error'],
            require_open=require_open)
        return False
    if _dense_delivers_hub(run):
        _set_soft(run, phase_detail='Backing the full model up to Hugging Face…')
        _dense_hub_backup(run, remote, report['master'])
    _finalize_dense_local_delivery(run, require_open=require_open)
    return True


# Fetches started from the Runs hub, one per run. In-process on purpose: the
# transfer belongs to the app that is showing its progress, and a run whose app
# restarts mid-fetch keeps its .part file and is simply fetched again.
_dense_fetch_threads = {}


def _can_fetch_dense_locally(run) -> bool:
    """Whether "Fetch to this computer" applies to this run right now.

    Deliberately narrow: a KEPT pod (the state every recoverable dense failure
    ends in), a delivery that wants a local copy, a pod we can still address,
    and no verified local file yet. Computed server-side so the button and the
    endpoint can never disagree."""
    return bool(_dense_delivers_local(run)
                and run.status == 'error_pod_kept'
                and run.vast_instance_id and run.remote_job_id and run.base_url
                and _run_param(run, 'local_artifact_status') != 'available')


def _dense_fetch_worker(app, run_id):
    with app.app_context():
        run = db.session.get(CloudTrainingRun, int(run_id))
        if run is None:
            _dense_fetch_threads.pop(int(run_id), None)
            _monitor_threads.pop(int(run_id), None)
            return
        # The recovery window is anchored on finished_at, and a kept pod bills
        # until it closes. Retrying a transfer must never push that deadline
        # back — the same rule recheck_full_transformer_delivery follows.
        finished = run.finished_at
        try:
            _deliver_dense_locally(
                run, _make_remote(run),
                should_cancel=_stop_event_for(run.id).is_set,
                require_open=False)
        except Exception as e:
            logger.warning('run %s: local dense fetch failed (%s)', run_id, e)
            try:
                _set(run, phase_detail=f'Fetch failed — {e}'[:500])
            except Exception:
                logger.debug('could not record the failed fetch', exc_info=True)
        finally:
            try:
                if finished and run.finished_at != finished:
                    _set(run, finished_at=finished)
            except Exception:
                logger.debug('could not restore the recovery deadline',
                             exc_info=True)
            _dense_fetch_threads.pop(int(run_id), None)
            _monitor_threads.pop(int(run_id), None)


def fetch_dense_locally(run_id, cancel=False) -> dict:
    """Start (or stop) bringing ONE kept dense run's files home.

    Returns immediately: a 26 GB transfer cannot be an HTTP request. Progress is
    the run's own phase line, which is what the hub already polls; cancelling
    keeps every byte already downloaded, so a resumed attempt continues from
    there instead of starting over.
    """
    run = db.session.get(CloudTrainingRun, int(run_id))
    if not run:
        raise ValueError('unknown cloud run')
    if not _is_full_transformer_run(run):
        raise ValueError('fetching to this computer is a full-model action')
    running = _dense_fetch_threads.get(int(run.id))
    if cancel:
        if not running:
            return {'ok': True, 'state': 'idle', 'run': _run_payload(run)}
        _stop_event_for(run.id).set()
        return {'ok': True, 'state': 'cancelling', 'run': _run_payload(run)}
    if running:
        return {'ok': True, 'state': 'fetching', 'run': _run_payload(run)}
    if not _can_fetch_dense_locally(run):
        raise ValueError(
            'this run has nothing to fetch: it needs a kept pod, a full-model '
            'delivery that includes this computer, and no verified local copy '
            'yet')
    from flask import current_app
    _stop_event_for(run.id).clear()
    _set(run, phase_detail='Fetching the full model to this computer…')
    thread = threading.Thread(
        target=_dense_fetch_worker,
        args=(current_app._get_current_object(), int(run.id)),
        daemon=True, name=f'dense-fetch-{run.id}')
    _dense_fetch_threads[int(run.id)] = thread
    # Registered as THE thread of this run, not only as a fetch: the transfer
    # flips the row to 'downloading', which makes it active again, and a Stop
    # pressed during it asks _monitor_is_responsive who is in charge. With no
    # thread registered the answer is "nobody", and a stop that finds nobody
    # DESTROYS the pod — mid-transfer, with the model still on it. Registered,
    # the stop takes the graceful path: the event is set, the transfer stops on
    # the next chunk, every byte already written is kept and so is the pod.
    _monitor_threads[int(run.id)] = thread
    thread.start()
    return {'ok': True, 'state': 'fetching', 'run': _run_payload(run)}


def _full_transformer_recovery_open(run, now=None) -> bool:
    """Whether a kept dense pod remains inside its bounded recovery window."""
    if not run.finished_at:
        return False
    now = now or datetime.utcnow()
    max_seconds = int((cfg.get('cloud.max_runtime_minutes') or 480)) * 60
    return (now - run.finished_at).total_seconds() <= max_seconds


def recheck_full_transformer_delivery(run_id, _api=None) -> dict:
    """Explicitly recheck one kept dense delivery without downloading weights.

    This is the service contract behind the UI's "Verify HF delivery" action.
    A late Hub propagation can therefore finish the run and release the pod;
    pending/missing/unverifiable results remain recoverable and never pretend
    that a checkpoint is available.
    """
    run = db.session.get(CloudTrainingRun, int(run_id))
    if not run:
        raise ValueError('unknown cloud run')
    if not _is_full_transformer_run(run):
        raise ValueError('delivery recheck is only available for full_transformer runs')
    if not _dense_delivers_hub(run):
        raise ValueError('this full model is delivered to this computer only — '
                         'there is no Hugging Face delivery to verify')
    if run.status == 'done' and _run_param(run, 'artifact_status') == 'available':
        return {
            'ok': True, 'delivery': 'available', 'cleanup_pending': False,
            'run': _run_payload(run),
        }
    if run.status != 'error_pod_kept':
        raise ValueError('only a kept full_transformer delivery can be rechecked')

    if _run_param(run, 'artifact_status') == 'available':
        # Integrity/compliance proof is already durable; this retry is only
        # about releasing compute and must not depend on HF availability.
        delivery = 'available'
    else:
        delivery = _verify_full_transformer_artifact(run, _api=_api)
    if delivery == 'available':
        _finalize_verified_full_transformer(run)
    else:
        detail = (
            'Hugging Face delivery is not visible or intact yet; pod kept'
            if delivery == 'missing' else
            'Hugging Face delivery verification is temporarily unavailable; pod kept')
        _set(
            run, status='error_pod_kept', phase_detail=detail,
            error=('Dense delivery is still unverified; use Verify HF delivery '
                   'again or recover from the kept pod.'),
            # Preserve the original bounded recovery deadline.  A click or a
            # periodic check must never extend paid-pod lifetime indefinitely.
            finished_at=run.finished_at)
    payload = _run_payload(run)
    return {
        'ok': True, 'delivery': delivery,
        'cleanup_pending': bool(
            delivery == 'available' and payload['status'] != 'done'),
        'run': payload,
    }


def reconcile_full_transformer_deliveries(_api=None, now=None) -> list:
    """One periodic late-propagation pass for recoverable dense deliveries.

    Terminal ``error_pod_kept`` rows are intentionally outside ACTIVE_STATES,
    so their original monitor is gone.  The independent supervisor invokes
    this once per tick until the existing bounded recovery deadline.  No call
    downloads the checkpoint; successful integrity+compliance verification is
    the only path that marks done and destroys the pod.
    """
    acted = []
    now = now or datetime.utcnow()
    try:
        runs = CloudTrainingRun.query.filter_by(status='error_pod_kept').all()
        for run in runs:
            if (not _is_full_transformer_run(run)
                    or not _run_param(run, 'hf_repo_id')
                    or not _full_transformer_recovery_open(run, now)):
                continue
            try:
                result = recheck_full_transformer_delivery(run.id, _api=_api)
                acted.append({
                    'run_id': run.id,
                    'delivery': result['delivery'],
                    'completed': result['run']['status'] == 'done',
                })
            except Exception:
                # Authenticated SDK diagnostics are deliberately omitted.
                logger.warning(
                    'run %s: periodic dense delivery verification failed', run.id)
    except Exception:
        logger.exception('periodic dense delivery reconciliation failed')
    return acted


def _monitor(app, run_id):
    """Full run lifecycle in a daemon thread.

    Destructive exits remain mandatory for ordinary LoRA runs, explicit user
    stops and the max-runtime cap. Once a dense job has started, unexpected
    failures deliberately end as ``error_pod_kept`` so the only ~26 GB result
    is recoverable; only verified Hugging Face delivery permits destruction.
    """
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
        job_started = bool(run.remote_job_id
                           and run.status in _JOB_STARTED_STATES)

        def mark_job_start_attempt():
            # A start POST can take effect remotely and then time out locally.
            # Mark BEFORE the call so dense recovery fails safe in that split-
            # brain window; LoRA's exception path does not consult this flag.
            nonlocal job_started
            job_started = True

        try:
            # -- heavy launch work, moved off the HTTP path (see launch) ----
            _prepare_staging(run)
            _assert_run_open(run)       # never rent for an already-stopped run
            # -- provision (if resuming, the instance may already exist) ----
            if not run.vast_instance_id:
                _provision(run)
            _assert_run_open(run)
            # Boot-readiness timeout anchor. A FRESH launch measures from now
            # (post-provision) so dataset staging / offer search never eat into
            # the pod's boot budget. A RESUME must NOT get a brand-new window on
            # every restart: that let a pod whose UI never answered survive
            # 37 min across two restarts instead of the 15-min READY_TIMEOUT
            # (incident 2026-07-14). On resume we anchor to the DURABLE
            # created_at (cap_anchor), so readiness measures the TOTAL time since
            # launch across every restart — the intended behaviour even for a pod
            # that was honestly still booting.
            #
            # ... unless the pod is not booting at all. A run we ENTER with a
            # started remote job is REATTACHING: the pod booted long ago, its
            # trainer is running, and the durable anchor has therefore already
            # eaten the whole boot budget. Run #146 (2026-08-03) is what that
            # costs: adopted at 16:01:58 at step 825/3000, one poll where the
            # vast API simply did not list the instance, and 10 s later the
            # budget check condemned it — pod alive, job training, money spent.
            # A reattach gets its own short window instead (see below).
            reattaching = resuming_existing_pod and job_started
            boot_started = (cap_anchor if resuming_existing_pod and not reattaching
                            else _now())

            # -- wait until the pod's UI answers ----------------------------
            # Readiness is checked BEFORE the elapsed-time read: an
            # already-booted pod (the common case, and every resumed run)
            # must be able to break out on the very first iteration without
            # ever touching _now() -- a test clock that jumps in large
            # strides per call must not misfire this boot-timeout on a pod
            # that was, in fact, instantly ready.
            template_mode = bool((c.get('template_hash') or '').strip())
            # Two clocks, exactly like the pre-step-1 phase further down:
            #  * ready_timeout is IDLE time, rearmed by any boot fact the pod
            #    had never shown before. Judging on elapsed time alone killed
            #    honest 26 GB image pulls at 25 min — while the evidence that
            #    they were progressing was already read, one line above, and
            #    shown to the user in the phase line.
            #  * boot_budget is the ABSOLUTE ceiling, evaluated BEFORE the
            #    rearm so a host that dribbles one new fact per poll cannot
            #    rearm its way past it. Raising ready_timeout instead would
            #    have been a cover-up: a pod that shows nothing is money
            #    burning and must still die in 25 minutes.
            ready_timeout = (int(c.get('ready_timeout_minutes') or 0) * 60
                             or READY_TIMEOUT_SECONDS)
            raw_boot_budget = c.get('boot_budget_minutes')
            boot_budget = int(90 if raw_boot_budget is None
                              else (raw_boot_budget or 0)) * 60
            slow_ban_seconds = float(
                cfg.get('cloud.slow_boot_blacklist_hours') or 6) * 3600
            if reattaching:
                # Not a boot: a reconnection. Both clocks become the SAME
                # tolerance the poll loop already grants a pod that stops
                # answering mid-run (cloud.unreachable_grace_minutes, 6 min by
                # default) — measured from this attempt, not from launch. That
                # is minutes of consecutive negative evidence instead of the
                # single unlucky poll that killed #146, and it stays bounded:
                # a pod that is really gone is still given up in 6 minutes.
                reconnect_seconds = (
                    int(c.get('unreachable_grace_minutes') or 0) * 60
                    or UNREACHABLE_GRACE_SECONDS)
                ready_timeout = boot_budget = reconnect_seconds
            # None until the first observation: the state a monitor INHERITS
            # (every fact a resumed pod already shows) is a baseline, not
            # progress — otherwise every app restart would hand a dead pod a
            # brand-new window, the 2026-07-14 regression all over again.
            boot_facts = None
            boot_progress_ts = boot_started
            boot_rearms = 0
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
                _assert_run_open(run)
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
                # The address of the pod we are actually paying for — the only
                # host identity that a machine_id re-registration cannot shed.
                if inst and inst.get('public_ipaddr'):
                    _stamp_host_ip(run, inst['public_ipaddr'])
                # ...and WHICH TRAINER it booted, which a template launch can
                # otherwise change under us without any local change.
                if inst and inst.get('image_uuid'):
                    _stamp_pod_image(run, inst['image_uuid'])
                derived = vast_client.derive_base_url(inst, port) if inst else None
                # The vast API is not the authority on whether the pod exists —
                # the pod is. Its listing has gaps (an answer without our
                # instance in it, indistinguishable from a destroyed pod), and
                # #146 was condemned inside one. When the row already carries an
                # address, that gap costs exactly one HTTP probe to settle: a
                # pod that answers its own URL is a pod that exists, whatever
                # the marketplace API is currently saying about it.
                base = derived or (run.base_url or None)
                ready = False
                if base:
                    if derived and run.base_url != derived:
                        _set(run, base_url=derived)
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
                        _blacklist_run_host(run, 'user stopped a boot stuck past 8 min')
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
                facts = _boot_facts(inst, port, base)
                if boot_facts is None:
                    boot_facts = set(facts)      # baseline, not progress
                elif (reattaching and boot_budget
                        and _now() - boot_started > boot_budget):
                    # A reattach that never got an answer. The host is not at
                    # fault (it was training minutes ago), so it is not banned,
                    # and the job is not stoppable, so no stop is sent.
                    raise _ReattachFailed(
                        'the pod could not be reached again after the app '
                        f'restarted — no answer for {boot_budget // 60} min: '
                        f'{_boot_stage_label(inst, port, base)}')
                elif boot_budget and _now() - boot_started > boot_budget:
                    # Ceiling first, so advancing evidence can never buy an
                    # unbounded boot. This host was still visibly working —
                    # slow, not broken — so it is skipped for HOURS, not days:
                    # a saturated uplink is a condition of the night, and a
                    # three-day exile the user never sees is the wrong price
                    # for it. (A host that shows nothing takes the full ban
                    # below — that mechanism has already saved real money.)
                    _blacklist_run_host(
                        run, 'pod was still booting past the boot budget',
                        ttl_seconds=slow_ban_seconds if boot_rearms else None)
                    raise RuntimeError(
                        'pod did not become ready in time — still booting after '
                        f'{boot_budget // 60} min: '
                        f'{_boot_stage_label(inst, port, base)}')
                elif facts - boot_facts:
                    boot_facts |= facts
                    boot_progress_ts = _now()
                    boot_rearms += 1
                elif _now() - boot_progress_ts > ready_timeout:
                    if reattaching:
                        raise _ReattachFailed(
                            'the pod could not be reached again after the app '
                            f'restarted — no answer for {ready_timeout // 60} '
                            f'min: {_boot_stage_label(inst, port, base)}')
                    # Nothing about this pod changed for the whole idle budget:
                    # a dead or frozen host. Full ban, as before.
                    _blacklist_run_host(run, 'pod stopped making boot progress')
                    raise RuntimeError(
                        'pod did not become ready in time — no boot progress '
                        f'for {ready_timeout // 60} min: '
                        f'{_boot_stage_label(inst, port, base)}')
                _sleep(POLL_SECONDS)

            remote = _make_remote(run)

            # -- resume contract: an already-submitted job (app restarted
            # mid-run) skips settings/upload/create/start entirely and goes
            # straight to polling the existing remote job. ------------------
            if not run.remote_job_id:
                pod_settings = _ensure_remote_settings_without_secret(run, remote)

                # -- upload dataset (+ masks folder if present) --------------
                _set(run, status='uploading', phase_detail='Uploading dataset')
                staging_dataset = _staging_dataset_dir(run)
                # Timed, because this is the app's ONLY regular observation of
                # how fast this machine can push bytes to a pod, and a
                # checkpoint-push forecast built on a guess is a forecast the
                # user is right not to believe. A dataset upload is the same
                # link, the same protocol and the same route.
                _upload_started = time.monotonic()
                remote.upload_dataset(
                    run.job_name, staging_dataset,
                    on_progress=_upload_heartbeat(run, 'Uploading the dataset'))
                _record_uplink(run, staging_dataset,
                               time.monotonic() - _upload_started)
                masks_dir = staging_dataset + '_masks'
                if os.path.isdir(masks_dir) and os.listdir(masks_dir):
                    remote.upload_dataset(
                        run.job_name + '_masks', masks_dir,
                        on_progress=_upload_heartbeat(run, 'Uploading the masks'))

                # A rented pod that cannot decode these clips is a job that runs
                # and yields nothing. Asked here, one command after the bytes
                # landed and before the GPU starts.
                _assert_pod_can_decode(run, remote, pod_settings)

                # -- build + submit the job -----------------------------------
                # Built from the run's own STAMPED params, and from the right
                # dataset table — see _build_pod_job_config, which now carries
                # the why (incident 2026-07-14, and the face/video split).
                job_config = _build_pod_job_config(run, staging_dataset,
                                                   pod_settings)
                job_id, adopted = _create_or_adopt_job(run, remote, job_config)
                # Persist the id THE INSTANT the job exists on the pod, before
                # the (slow) seeding and the start. Recording it only after
                # start_job left a window in which the pod already held the job
                # but our row still said remote_job_id=NULL — an app restart
                # inside that window sent the resume straight back into this
                # branch, where the pod refused the duplicate name with
                # 409 "Job name already exists" and the run died with the money
                # already spent (run #107, ~1 h of 5090 time). The run is NOT
                # yet 'training' here: only start_job earns that status, and the
                # resume branch relies on the distinction.
                _set(run, remote_job_id=job_id,
                     phase_detail='Job created on the pod')
                if adopted:
                    # The pod already had this job (this run's earlier attempt).
                    # Never blind-start it: it may be mid-training.
                    _ensure_remote_job_started(
                        run, remote, job_id, pod_settings,
                        on_start_attempt=mark_job_start_attempt)
                    # A pre-existing job whose status could not be read is
                    # conservatively treated as started. Destroying the pod on
                    # that uncertainty could erase a live dense checkpoint.
                    job_started = True
                else:
                    # Continue-in-cloud: drop the source checkpoint into the
                    # job's save_root BEFORE start so ai-toolkit auto-resumes.
                    _seed_resume_checkpoint(run, remote, pod_settings)
                    mark_job_start_attempt()
                    remote.start_job(job_id)
                    _set(run, status='training',
                         phase_detail='Job queued on the pod')
            else:
                job_id = run.remote_job_id
                _set(run, phase_detail='Resuming — reattaching to running job')
                _ensure_remote_job_started(
                    run, remote, job_id,
                    on_start_attempt=mark_job_start_attempt)
                job_started = True

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
            #    The step counter is NOT the only progress signal in that phase:
            #    the pod's log carries the base-model download's byte counter,
            #    and a pod whose bytes advance is a pod that progresses. Judging
            #    the phase on steps alone killed a paid run whose download was
            #    perfectly healthy (reported by j_o_e_l. on Discord 2026-07-27:
            #    KREA-2 RAW on a 5090, FAILED at 59 min, never past step 0 —
            #    26.3 GB at the 2.58 MB/s measured on another pod is ~2 h 50, so
            #    the 45-min budget GUARANTEED the failure). Advancing bytes now
            #    rearm this clock, exactly as a step rearms the stall clock.
            #    Raising the timeout instead would have been a cover-up: a
            #    genuinely wedged pod must still die fast, because it is money
            #    burning. Hence the second, ABSOLUTE ceiling below.
            #  * download budget — the ceiling that keeps the rearm honest. A
            #    host at 200 kB/s advances its bytes at every poll for 36 h;
            #    rearming alone would let it ride the whole runtime cap for zero
            #    steps, which IS the run-#75 failure the first-step watchdog was
            #    built to stop. The default (180 min) clears the measured 2 h 50
            #    worst case and stays far under the 480-min runtime cap; 0 turns
            #    the ceiling off and leaves the runtime cap as sole backstop.
            stall_seconds = int(c.get('stall_timeout_minutes') or 30) * 60
            first_step_seconds = int(c.get('first_step_timeout_minutes') or 45) * 60
            raw_budget = c.get('first_step_download_budget_minutes')
            dl_budget_seconds = int(180 if raw_budget is None else (raw_budget or 0)) * 60
            grace_seconds = (int(c.get('unreachable_grace_minutes') or 0) * 60
                             or UNREACHABLE_GRACE_SECONDS)
            last_step = -1
            last_progress_ts = _now()
            # Peak bytes the pod has reported downloading, and the anchor of the
            # absolute pre-step-1 ceiling. `downloaded_bytes` only ever grows:
            # a bar that restarts lower is treated as no progress, which is the
            # conservative side of the choice.
            downloaded_bytes = 0.0
            first_step_anchor = last_progress_ts
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
                _assert_run_open(run)
                if _now() - cap_anchor > max_seconds:
                    try:
                        remote.stop_job(job_id)
                    except Exception:
                        pass
                    if _dense_delivers_local(run):
                        # The cap is about not paying for ever, not about
                        # throwing the result away: a dense master that only
                        # exists on this pod dies with it. Bring it home first —
                        # the supervisor leaves a monitor that is actively
                        # writing alone (see _rescuing_checkpoint) — then the
                        # pod goes, exactly as the cap intends.
                        _deliver_dense_locally(
                            run, remote, should_cancel=stop_event.is_set,
                            require_open=True)
                        return
                    _try_download_checkpoint(run, remote, allow_stale=True)
                    _finish_if_open(run, 'stopped',
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
                    if _dense_delivers_local(run):
                        # Stopping the TRAINING is not abandoning the result:
                        # the LoRA lane rescues its checkpoint here too. A
                        # second press of Stop cancels the transfer itself
                        # (should_cancel), keeping what already landed.
                        _deliver_dense_locally(
                            run, remote, should_cancel=stop_event.is_set,
                            require_open=True)
                        return
                    _try_download_checkpoint(run, remote, allow_stale=True)
                    _finish_if_open(run, 'stopped', detail='Stopped by user')
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

                log_text = _pull_log_and_samples(run, remote, job_id)
                # Mid-run checkpoint mirror, throttled (~2 min at 10 s polls):
                # list_files is cheap, but no need to hammer it every poll —
                # the pod only writes a new save every save_every steps.
                polls += 1
                if polls % _CKPT_SYNC_EVERY_POLLS == 0:
                    _sync_latest_checkpoint(run, remote)
                status = job.get('status')
                info = job.get('info') or ''
                _set_soft(run, phase_detail=f"{status}: {info}"[:500])

                if status == 'completed':
                    if _is_full_transformer_run(run):
                        # LAST use of the pod, and the only moment the ~26 GB
                        # master and a GPU are in the same place: turn it into
                        # the ~10 GB file people actually load in ComfyUI.
                        # Fail-open by construction — the master exists either
                        # way, on the pod and (for a hub run) on the Hub.
                        _export_full_transformer_fp8(run, remote)
                        if _dense_delivers_local(run):
                            # Local FIRST, and the pod stays until the file on
                            # this computer is proven. Everything after that
                            # point — the Hub backup, the pod cleanup — can
                            # fail without costing the run.
                            _deliver_dense_locally(
                                run, remote, should_cancel=stop_event.is_set,
                                require_open=True)
                            return
                        _set(run, phase_detail='Verifying Hugging Face delivery…')
                        _complete_full_transformer_delivery(run)
                        return
                    ok = _try_download_checkpoint(run, remote)
                    if not ok:
                        # A host that cannot DELIVER its result (even through
                        # the resume loop) is a bad host — skip it next time.
                        _blacklist_run_host(run, 'could not serve the final checkpoint')
                        # LoRA > a few minutes of pod time: keep the pod for
                        # manual recovery; max-runtime/reconcile will reap it.
                        # Same guard as _finish_if_open: announcing a kept pod
                        # for a run the supervisor just force-stopped would
                        # point the user at an instance that is already gone.
                        _assert_run_open(run)
                        _set(run, status='error_pod_kept',
                             error='checkpoint download failed — pod kept, '
                                   f'recover manually at {run.base_url}',
                             finished_at=datetime.utcnow())
                        return
                    _download_intermediates(run, remote)
                    _import_result(run)
                    _mirror_into_local_run(run)
                    # The video lane's provenance, written beside the weights —
                    # the face lane's registry cannot hold it (its manifest is
                    # face IMAGES, its dataset_id a face id). No-op for a face
                    # run, and best-effort: bookkeeping never fails a run.
                    video_run_lineage.record(run)
                    _finish_if_open(run, 'done', detail='Training complete')
                    return
                if status in ('error', 'stopped'):
                    if _is_full_transformer_run(run):
                        # The 403 that killed run #146 at step 2750/3000: the
                        # training was fine, the PUSH was refused. Naming it is
                        # what turns "rent another GPU" into "click Settings".
                        detail, error = _dense_remote_failure(status, info, log_text)
                        _keep_full_transformer_pod(
                            run, detail=detail, error=error,
                            stop_remote=(status == 'error'))
                    else:
                        _try_download_checkpoint(run, remote, allow_stale=True)
                        _finish_if_open(
                            run, 'error' if status == 'error' else 'stopped',
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
                    if _is_full_transformer_run(run):
                        _keep_full_transformer_pod(
                            run,
                            detail='Stalled — no step progress for '
                                   f'{stall_seconds // 60} min; pod kept for '
                                   'dense-checkpoint recovery',
                            error='stall watchdog; dense pod kept')
                    else:
                        _try_download_checkpoint(run, remote, allow_stale=True)
                        _finish_if_open(
                            run, 'error',
                            detail='Stalled — no step progress for '
                                   f'{stall_seconds // 60} min; pod terminated',
                            error='stall watchdog')
                    return
                elif last_step <= 0:
                    # -- before step 1: the same guiding rule, applied to the
                    # signal this phase actually has. Nothing to rescue here
                    # either way — no checkpoint exists yet.
                    if dl_budget_seconds and \
                            (_now() - first_step_anchor) > dl_budget_seconds:
                        # Checked BEFORE the rearm on purpose: a pod that
                        # advances a handful of bytes every poll would otherwise
                        # rearm its way past every ceiling.
                        try:
                            remote.stop_job(job_id)
                        except Exception:
                            pass
                        _finish_if_open(
                            run, 'error',
                            detail='Still not training after '
                                   f'{dl_budget_seconds // 60} min '
                                   f'(base model fetched: {_fetched_label(downloaded_bytes)}) '
                                   '— pod terminated before it could burn the '
                                   'whole runtime cap',
                            error='first-step download budget')
                        return
                    # download_bytes_seen, not parse_download_progress: the
                    # card's parser reports the LAST bar, and with several
                    # files in flight consecutive tails end on different bars,
                    # so its `done` alternates between two frozen files and
                    # would read as endless movement. A kill decision needs the
                    # total, which only a file that really advanced can raise.
                    seen = lt.download_bytes_seen(log_text)
                    if seen is not None and seen > downloaded_bytes:
                        downloaded_bytes = seen
                        last_progress_ts = _now()
                    elif (_now() - last_progress_ts) > first_step_seconds:
                        # Say what was MEASURED. The old wording ("pod likely
                        # stuck downloading the base model") is exactly what
                        # j_o_e_l. read while his pod downloaded normally, and it
                        # sent him hunting a vast.ai fault that did not exist.
                        if downloaded_bytes > 0:
                            what = ('its base-model download stopped at '
                                    f'{_fetched_label(downloaded_bytes)}')
                        else:
                            what = ('the pod never reported a single downloaded '
                                    'byte')
                        try:
                            remote.stop_job(job_id)
                        except Exception:
                            pass
                        _finish_if_open(
                            run, 'error',
                            detail='No training step reached in '
                                   f'{first_step_seconds // 60} min and '
                                   f'{what}; pod terminated',
                            error='first-step watchdog')
                        return
                _sleep(POLL_SECONDS)
        except _RunClosedExternally as closed:
            # Someone with more authority than this thread (a forced stop, the
            # supervisor) already closed the run. Do NOT touch the row -- but a
            # pod we may have just rented is still ours to kill, unless the row
            # says it was deliberately kept for manual recovery.
            logger.warning('cloud run %s closed externally (%s) — monitor '
                           'standing down', run_id, closed)
            if run.vast_instance_id and run.status != 'error_pod_kept':
                try:
                    vast_client.destroy_instance(run.vast_instance_id)
                except Exception:
                    logger.exception('stand-down destroy of %s raised',
                                     run.vast_instance_id)
        except _WaitAborted:
            # Same gesture, same treatment as the boot-wait's Stop check
            # (~5384): no pod exists yet at this point in the launch, so
            # there is nothing to destroy — just land the row as 'stopped'
            # instead of falling through to the generic 'Run failed' below.
            stop_event.clear()
            _finish(run, 'stopped',
                    detail='Stopped by user while waiting for the dataset')
        except Exception as e:
            error_text = _redacted_error_text(e)
            if _is_full_transformer_run(run):
                # Do not emit the raw traceback of an authenticated HF/remote
                # request. Its exception may include request diagnostics.
                logger.error('dense cloud run %s failed (%s job start)', run_id,
                             'after' if job_started else 'before')
                if job_started:
                    # A pod we could not reach at all cannot be stopped — and
                    # asking anyway is how #146 lost its training: the stop was
                    # sent on a wrong verdict and the still-live trainer obeyed.
                    unreachable = isinstance(e, _ReattachFailed)
                    _keep_full_transformer_pod(
                        run,
                        detail=('Could not reach the pod again after restarting; '
                                'training left running and pod kept for recovery'
                                if unreachable else
                                'Dense run failed after job start; remote job '
                                'stopped if possible and pod kept for recovery'),
                        error=error_text or 'unexpected dense training failure',
                        stop_remote=not unreachable)
                else:
                    _finish(run, 'error', detail='Dense run failed before step 1',
                            error=error_text)
            else:
                logger.exception('cloud run %s failed', run_id)
                retryable = _is_retryable_pod_failure(error_text)
                # Exclude the failed host before selecting the fresh pod.
                if retryable:
                    _blacklist_run_host(
                        run, f'transient pod failure: {error_text[:160]}')
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


# A run that has reached one of these has provably had its remote job STARTED
# (only the post-start_job write sets 'training'). Anything earlier means the
# job may exist on the pod without ever having been launched.
_JOB_STARTED_STATES = ('training', 'downloading', 'terminating')


def _create_or_adopt_job(run, remote, job_config):
    """Submit this run's job, or ADOPT the one already on the pod.

    Returns (job_id, adopted). The pod's job `name` is unique, and ours is
    `lds<run.id>_<run_name>` — stable for the life of the run and derived from a
    primary key, so a 409 on submit can only mean THIS run already created THIS
    job on THIS pod (an earlier attempt whose id never reached our row). Killing
    the run over a duplicate of its own job wastes an already-paid hour, so the
    id is read back from the pod's job list and the run continues.

    If the list cannot resolve the name, the run still fails — but with an error
    that says what happens next and what becomes of the pod."""
    try:
        return remote.create_job(run.job_name, job_config), False
    except Exception as e:
        if 'HTTP 409' not in str(e):
            raise
        logger.warning('run %s: the pod already holds job %r (409) — adopting it '
                       'instead of failing the run', run.id, run.job_name)
        existing = None
        try:
            existing = remote.find_job_by_name(run.job_name)
        except Exception:
            logger.exception('run %s: could not list the pod jobs to adopt %r',
                             run.id, run.job_name)
        job_id = str((existing or {}).get('id') or '')
        if job_id:
            _set(run, phase_detail='Reattached to the job already on the pod')
            return job_id, True
        raise RuntimeError(
            f'this pod already holds a training job named "{run.job_name}" '
            'but would not say which one, so it cannot be reattached. The pod '
            'is being terminated so it stops costing money; any checkpoint it '
            'had already produced is lost. Use "Retry" on this run to relaunch '
            'on a fresh pod — a retry gets a new job name, so it cannot hit '
            'this again.')


def _ensure_remote_job_started(run, remote, job_id, pod_settings=None,
                               on_start_attempt=None):
    """Guarantee the remote job is actually RUNNING, not merely created.

    ai-toolkit creates a job with status 'stopped' and only `start` moves it to
    'queued'. The poll loop below reads 'stopped' as a terminal state, so a job
    that exists but was never started would kill the run at the first poll —
    exactly the bug traded in if the id were recorded early and nothing else
    changed. A run past `_JOB_STARTED_STATES` provably started its job; anything
    earlier asks the pod, and starts (after re-seeding any resume checkpoint,
    which must land before the first step) only a job still sitting at
    'stopped' with no step. Never blind-starts: re-queuing a live job would
    disturb a run that is training fine."""
    if run.status in _JOB_STARTED_STATES:
        return
    try:
        job = remote.get_job(job_id) or {}
    except Exception as e:
        # Not fatal here: the poll loop owns pod reachability and its grace
        # window. Guessing 'never started' on an unreachable pod could re-queue
        # a job that is training.
        logger.warning('run %s: could not read job %s to check whether it was '
                       'started (%s) — leaving it to the poll loop', run.id, job_id, e)
        return
    if (job.get('status') or 'stopped') != 'stopped' or (job.get('step') or 0) > 0:
        # Remote evidence is authoritative: once a live/advanced job is seen,
        # fail safe *before* any database write.  If the following _set raises
        # (lock/disk failure), the monitor's exception path must keep the dense
        # pod instead of misclassifying it as pre-start and destroying it.
        if on_start_attempt is not None:
            on_start_attempt()
        _set(run, status='training')     # already live (or finished) — poll it
        return
    logger.warning('run %s: job %s exists on the pod but was never started — '
                   'starting it now', run.id, job_id)
    _set(run, phase_detail='Resuming — the job was created but never started')
    _seed_resume_checkpoint(run, remote,
                            pod_settings if pod_settings is not None
                            else remote.get_settings())
    if on_start_attempt is not None:
        on_start_attempt()
    remote.start_job(job_id)
    _set(run, status='training', phase_detail='Job queued on the pod')


def _seed_resume_checkpoint(run, remote, pod_settings):
    """Continue-in-cloud: place the source run's harvested checkpoint into THIS
    job's save_root on the pod so ai-toolkit's auto-resume finds it — it globs
    <TRAINING_FOLDER>/<job_name>/<job_name>*.safetensors, takes the newest by
    ctime, and reads the resume step from the safetensors metadata. The file is
    renamed to THIS job's prefix so the glob matches (the save the trainer would
    itself write). No resume checkpoint stamped in train_params -> no-op (a
    normal launch). A missing/failed seed RAISES: a 'continue' that cannot
    resume must fail loudly, never silently train from scratch.

    TWO ROADS, one destination, and which one is taken is the user's choice —
    stamped at launch as a Hub repository (the pod pulls it itself, over a
    datacenter link) or a local path (this machine pushes it, over the user's
    uplink). A LoRA is small enough that the question never comes up: one
    request and it is there. A ~26 GB dense master is where the two roads have
    genuinely different prices, which is why they are both offered and both
    costed before the click."""
    src = _run_param(run, 'resume_ckpt_path')
    # The video lane stamps a LIST. A Wan 2.2 MoE checkpoint is two files at one
    # step, and ai-toolkit's auto-resume globs the save_root, takes the newest
    # match, and then `Wan2214bModel.load_lora` reads its SIBLING by rewriting
    # `_high_noise` into `_low_noise` — so both must land, under this job's
    # prefix, with their stage suffixes intact. Seeding one of them resumes one
    # expert and restarts the other from zero, and nothing raises.
    sources = list(_run_param(run, 'resume_ckpt_paths') or ())
    repo_id = _run_param(run, 'resume_hf_repo_id')
    if not src and not sources and not repo_id:
        return
    step = int(_run_param(run, 'resume_step') or 0)
    remote_name = f'{run.job_name}_{step:09d}.safetensors'
    training_folder = pod_settings['TRAINING_FOLDER'].rstrip('/')
    dest_dir = f'{training_folder}/{run.job_name}'
    if sources:
        _set(run, phase_detail='Seeding checkpoint for resume…')
        for one in sources:
            if not os.path.isfile(one):
                raise RuntimeError(f'resume checkpoint vanished before upload: {one}')
            _, stage = video_training.split_checkpoint_name(one)
            name = video_training.restage_checkpoint_name(run.job_name, step, stage)
            _push_resume_checkpoint(run, remote, pod_settings, one, dest_dir, name)
        logger.info('run %s: seeded %s resume file(s) -> %s',
                    run.id, len(sources), dest_dir)
        return
    if repo_id:
        from . import dense_pod_hub
        filename = _run_param(run, 'resume_hf_filename')
        dense = ((cfg.get('cloud') or {}).get('full_transformer') or {})
        _set(run, phase_detail='Fetching the checkpoint to resume from…')
        dense_pod_hub.fetch_checkpoint(
            remote, instance_id=run.vast_instance_id, repo_id=repo_id,
            filename=filename, dest_path=f'{dest_dir}/{remote_name}',
            hf_token=cfg.secret('HF_CLOUD_TOKEN'),
            tmp_dir=run.staging_dir or _staging_root(),
            budget_seconds=int(dense.get('hub_fetch_budget_seconds')
                               or dense_pod_hub.DEFAULT_FETCH_BUDGET_SECONDS),
            on_state=lambda detail: _set_soft(run, phase_detail=detail[:500]))
        logger.info('run %s: the pod fetched its resume checkpoint from the Hub '
                    '-> %s', run.id, dest_dir)
        return
    if not os.path.isfile(src):
        raise RuntimeError(f'resume checkpoint vanished before upload: {src}')
    _set(run, phase_detail='Seeding checkpoint for resume…')
    _push_resume_checkpoint(run, remote, pod_settings, src, dest_dir, remote_name)
    logger.info('run %s: seeded resume checkpoint %s -> %s',
                run.id, os.path.basename(src), dest_dir)


# Below this, one request is not the right shape any more. The threshold is not
# about memory (the streamed body costs a megabyte whatever the file weighs) —
# it is about what an interruption COSTS. Redoing a 200 MB LoRA is seconds;
# redoing a 26 GB master is the evening, and a slice that already landed is
# progress the next attempt can keep.
_SLICED_PUSH_THRESHOLD_BYTES = 1024 * 1024 * 1024


def _push_resume_checkpoint(run, remote, pod_settings, src, dest_dir, remote_name):
    """Put a checkpoint from THIS COMPUTER into the pod's save_root.

    Small file: one streamed request, as it always was. Big file: numbered
    slices, a probe that skips whatever already landed, and a pod-side assembly
    — see ``pod_checkpoint_push``. The byte counter is fed into the SAME
    upload-progress file the dataset transfer writes, so the freeze watchdog
    reads a multi-hour checkpoint push exactly the way it reads a multi-hour
    dataset push, with no new watchdog input to teach it."""
    total = os.path.getsize(src)
    if total < _SLICED_PUSH_THRESHOLD_BYTES:
        remote.seed_checkpoint(pod_settings['DATASETS_FOLDER'], dest_dir,
                               remote_name, src)
        return
    from . import pod_checkpoint_push, pod_transfer_plan
    dense = ((cfg.get('cloud') or {}).get('full_transformer') or {})
    started = time.monotonic()
    _write_upload_progress(run, 0, 1, 0, total)
    result = pod_checkpoint_push.push_checkpoint(
        remote, instance_id=run.vast_instance_id, local_path=src,
        dest_dir=dest_dir, remote_name=remote_name,
        datasets_folder=pod_settings['DATASETS_FOLDER'],
        job_name=run.job_name, tmp_dir=run.staging_dir or _staging_root(),
        slice_bytes=int(dense.get('push_slice_bytes')
                        or pod_checkpoint_push.DEFAULT_SLICE_BYTES),
        on_state=lambda detail: _set_soft(run, phase_detail=detail[:500]),
        on_progress=lambda done, want: _write_upload_progress(run, 0, 1, done, want),
        should_cancel=_stop_event_for(run.id).is_set)
    # The measurement that makes the NEXT forecast worth reading, and the only
    # KIND of transfer that may feed it: one continuous file, which is what a
    # checkpoint push forecasts. Only the bytes this attempt actually SENT are
    # timed — a resumed push that skipped 20 GB already on the pod would
    # otherwise report an uplink several times faster than the line has ever
    # been, and that number becomes a price.
    pod_transfer_plan.record_uplink_sample(
        result.get('sent_bytes'), time.monotonic() - started,
        kind=pod_transfer_plan.KIND_STREAM)


def _fetched_label(num_bytes) -> str:
    """Downloaded volume as a user-facing string. Watchdog messages quote what
    was measured, so the unit has to survive both '0 bytes' and '26.3 GB'."""
    n = float(num_bytes or 0)
    if n <= 0:
        return 'nothing'
    if n < 1e9:
        return f'{n / 1e6:.0f} MB'
    return f'{n / 1e9:.1f} GB'


def _pull_log_and_samples(run, remote, job_id):
    """Mirror remote log + new samples into staging so cloud_progress reuses
    the exact local parsing/serving machinery. Never raises.

    Returns the log text (''
    when it could not be fetched) — the first-step watchdog reads the
    base-model download's byte counter out of it, and re-reading the file we
    just wrote would only add a way for the two to disagree."""
    text = ''
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
    return text


def _newest_remote_checkpoint(remote, job_id):
    """The newest .safetensors file entry ({'path', 'size'}), or None.
    ai-toolkit zero-pads step numbers, so lexicographic order IS step order."""
    files = [f for f in remote.list_files(job_id)
             if f.get('path', '').endswith('.safetensors')]
    if not files:
        return None
    return sorted(files, key=lambda f: f['path'])[-1]


def _fetch_checkpoint(run, remote, ckpt, timeout=None, attempts=3,
                      on_progress=None, resume=False, should_cancel=None) -> str:
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
    # Straight into the durable store, never into staging: a weight that only
    # ever existed in a directory the cleanup may trash is how checkpoints were
    # lost. The .part-then-rename below still applies, one folder over.
    dest = os.path.join(checkpoint_store_dir(run, create=True) or run.staging_dir,
                        name)
    if run.checkpoint_local_path and os.path.isfile(dest) \
            and os.path.basename(run.checkpoint_local_path) == name:
        return dest
    remote.download_public_file(remote_path, dest, timeout=timeout,
                                expected_size=ckpt.get('size'), attempts=attempts,
                                on_progress=on_progress, resume=resume,
                                should_cancel=should_cancel)
    want = int(ckpt.get('size') or 0)
    got = os.path.getsize(dest)
    if want and got != want:
        try:
            os.remove(dest)
        except OSError:
            pass
        raise RuntimeError(f'truncated download of {name}: {got}/{want} bytes')
    return dest


# A dense master is ~26 GB on a proxy that can cut the stream every couple of
# MB. Each cut costs one attempt and resumes at the offset, so the budget has to
# be counted in thousands, not in hundreds — and it is bounded by the transfer's
# own no-progress rule (an attempt that adds zero bytes ends the loop).
_DENSE_FETCH_ATTEMPTS = 4000

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
    if _is_full_transformer_run(run):
        return
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


_DOWNLOAD_HEARTBEAT_SECONDS = 20


def _transfer_size(got, want) -> str:
    got_mb = (got or 0) / 1e6
    return f'{got_mb:.0f} / {want / 1e6:.0f} MB' if want else f'{got_mb:.0f} MB'


def _download_heartbeat(run, name):
    """Progress callback for a long checkpoint transfer.

    A transfer of tens of minutes must not LOOK like a dead monitor. Every
    safety net in this module reads database progress and nothing else, so a
    silent transfer is indistinguishable from a wedged thread — and the pod
    would be terminated exactly while we are rescuing the thing the run was
    for. Beating updated_at from inside the stream is what makes the two
    distinguishable; the user gets a moving figure out of it too.

    Throttled, and it never raises: a heartbeat that cannot write must not
    sink a transfer that is otherwise working."""
    state = {'ts': 0.0}

    def beat(got, want):
        now = _now()
        if now - state['ts'] < _DOWNLOAD_HEARTBEAT_SECONDS:
            return
        state['ts'] = now
        try:
            _set(run, phase_detail=f'Downloading {name} — '
                                   f'{_transfer_size(got, want)}'[:500])
        except Exception:
            logger.debug('download heartbeat could not write', exc_info=True)

    return beat


_UPLOAD_HEARTBEAT_SECONDS = 10


def _upload_size(sent, total) -> str:
    """Upload volume as a user-facing string. Datasets here span three orders
    of magnitude (12 files to 12 422), so the unit follows the TOTAL — a
    fixed unit would print either '0.0 GB' for a small set or '24000 MB' for a
    big one, and both read as a bug."""
    total = float(total or 0)
    sent = float(sent or 0)
    if total >= 1e9:
        return f'{sent / 1e9:.1f} of {total / 1e9:.1f} GB'
    return f'{sent / 1e6:.0f} of {total / 1e6:.0f} MB'


def _upload_heartbeat(run, label):
    """Progress callback for the dataset upload.

    Two jobs in one callback, and they are not the same job. The DURABLE write
    happens on every batch: it is the byte clock the supervisor judges the
    phase on (_progress_fingerprint), and throttling it would blunt the very
    watchdog it feeds. The phase_detail refresh is throttled and purely
    cosmetic, so it goes through _set_soft — a local write lock is allowed to
    skip a sentence, never to fail a run that is uploading normally (the
    lesson run #137 paid for on 2026-08-01).

    The driver disables a callback that raises, which is the right policy for
    the transfer but the wrong outcome for THIS callback: losing it would also
    stop the byte clock, and a healthy upload would then look stalled and be
    killed by the very watchdog these bytes feed. So the sentence half is made
    total here — any failure to describe the transfer is logged and dropped,
    and the recording half carries on."""
    state = {'ts': 0.0}

    def beat(files, files_total, sent, total):
        _write_upload_progress(run, files, files_total, sent, total)
        now = _now()
        last = files_total and files >= files_total
        if not last and now - state['ts'] < _UPLOAD_HEARTBEAT_SECONDS:
            return
        state['ts'] = now
        try:
            _set_soft(run, phase_detail=(
                f'{label} — {files}/{files_total} files, '
                f'{_upload_size(sent, total)}')[:500])
        except Exception:                       # noqa: BLE001 - deliberate
            logger.debug('upload heartbeat could not write', exc_info=True)

    return beat


def _try_download_checkpoint(run, remote, allow_stale=False) -> bool:
    """Download the newest .safetensors into staging. False on failure.
    allow_stale (rescue paths — stop/stall/cap): when the pod can't serve the
    newest save anymore, an already-synced OLDER save still counts as success.
    The COMPLETION path must stay strict (allow_stale=False): falling back to
    an older save there would silently discard the final training steps —
    error_pod_kept keeps the pod so the user can recover the real result."""
    if _is_full_transformer_run(run):
        return False
    try:
        ckpt = _newest_remote_checkpoint(remote, run.remote_job_id)
        if ckpt:
            name = os.path.basename(ckpt['path'].replace('\\', '/'))
            # The status flips BEFORE the transfer, not after it. This is the
            # end of a run that WORKED, and the transfer can take tens of
            # minutes on a pod proxy that cuts the stream every couple of MB.
            # While it was still labelled 'training' the freeze watchdog judged
            # it on the training threshold (45 min of database silence) with a
            # frozen updated_at — it would have destroyed the pod mid-rescue,
            # throwing away a checkpoint already paid for. 'downloading' is an
            # ACTIVE state judged on the silent-phase floor, and the heartbeat
            # below keeps even that from being needed.
            _set(run, status='downloading',
                 phase_detail=f'Downloading {name}…'[:500])
            # Large attempts budget: a sick-proxy host cutting the stream
            # every ~0.5-2 MB still delivers an 85 MB file via ~100 resumed
            # connections (validated live 2026-07-13, run #7's manual rescue).
            dest = _fetch_checkpoint(run, remote, ckpt, attempts=400,
                                     on_progress=_download_heartbeat(run, name))
            _set(run, checkpoint_local_path=dest,
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
    if _is_full_transformer_run(run):
        return
    # A video run's dataset_id names the video table, so this import would deploy
    # a Wan LoRA into the ComfyUI folder of the FACE dataset of the same id — and
    # a Wan 2.2 checkpoint is a high-noise/low-noise pair no loader here can take
    # anyway. Deploying video weights is a separate piece of work; until it
    # exists, standing down is the only correct answer. The weights stay in the
    # store and remain downloadable from the hub.
    if crd.is_video(run):
        logger.info('run %s trained a video dataset — ComfyUI import skipped '
                    '(no video deploy lane yet); weights kept in the store',
                    run.id)
        return
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
                             src_dir=os.path.dirname(run.checkpoint_local_path),
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
    if _is_full_transformer_run(run):
        return
    try:
        files = [f for f in remote.list_files(run.remote_job_id)
                 if f.get('path', '').endswith('.safetensors')]
    except Exception as e:
        logger.warning('intermediate listing failed: %s', e)
        return
    have = os.path.basename(run.checkpoint_local_path or '')
    store = checkpoint_store_dir(run, create=True) or run.staging_dir
    for f in files:
        name = os.path.basename(f['path'].replace('\\', '/'))
        if name == have:
            continue
        dest = os.path.join(store, name)
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
    if _is_full_transformer_run(run):
        return
    # `lt._run_dir` builds a path from a FACE dataset's folder. There is no local
    # video training lane, so a video run has no local run directory to mirror
    # into — and calling it anyway would write this run's checkpoints into the
    # run folder of the face dataset that happens to share its id.
    if crd.is_video(run):
        return
    try:
        saves = run_checkpoint_files(run)
        if not saves:
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
        for src_name in sorted(saves):
            _mirror_one(run, run_dir, base, saves[src_name])
    except Exception as e:
        # RuntimeError from _run_dir = ai-toolkit not configured -> fine
        logger.debug('local run-dir mirror skipped: %s', e)


def _mirror_one(run, run_dir, base, src_path):
    try:
        src_name = os.path.basename(src_path)
        # Step AND stage: a Wan 2.2 checkpoint is a high-noise/low-noise pair, and
        # a name rebuilt from the step alone is identical for both halves — the
        # second copy would then be refused as a collision with the first.
        step, stage = video_training.split_checkpoint_name(src_name)
        dest_name = video_training.restage_checkpoint_name(base, step, stage)
        dest = os.path.join(run_dir, dest_name)
        if os.path.exists(dest):
            # A LOCAL run of the same dataset+family already produced this
            # exact name (the unsuffixed FINAL collides whenever both worlds
            # completed a run) — never clobber local work. The cloud result
            # stays available in staging, ComfyUI and the hub's ⬇ button.
            logger.warning('local run dir already has %s — cloud mirror skipped '
                           '(local checkpoint left untouched)', dest_name)
            return
        shutil.copy2(src_path, dest)
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


def run_dataset_name(run):
    """Human-readable dataset name for ONE run, resolved in the table that run
    actually trained on.

    `_dataset_name(dataset_id)` above cannot do this: an id alone is ambiguous
    now that face and video datasets share an integer space, so it would name the
    face dataset of the same id for a video run — a label that is not merely
    unhelpful, but wrong about what was trained. Every payload that has the run
    in hand uses this instead; the id-only helper survives for the callers that
    genuinely only have a face dataset id."""
    try:
        return crd.display_name(run)
    except Exception:
        return None


def _staging_save_count(run) -> int:
    """How many checkpoints this run still HAS — the 'saves' figure on the
    Runs-hub cards. Reads the durable store (plus the legacy staging dir), so
    cleaning a run's staging no longer makes its saves vanish from the card."""
    return len(run_checkpoint_files(run))


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


# --- What the launch is doing, phase by phase --------------------------------
# Renting a pod takes minutes, and until now the only thing the user saw was a
# button stuck on 'Launching…' followed by one phase sentence. The monitor has
# always known where it was; this exposes THAT knowledge (status + the
# phase_detail it already writes) as an ordered checklist with a clock. No new
# state machine: every step below is decided from a row the monitor was already
# keeping, so a phase this function cannot place degrades to the first step of
# the current status rather than inventing one.
_LAUNCH_STEPS = (
    ('staging', 'Preparing the dataset'),
    ('offer', 'Searching for a GPU offer'),
    ('boot', 'Renting the machine and booting the pod'),
    ('upload', 'Uploading the dataset'),
    ('start', 'Starting the training job'),
)
_LAUNCH_STATES = ('preparing', 'provisioning', 'uploading')
_OFFER_SEARCH_DETAIL = 'Searching for a GPU offer…'


def _active_launch_step(status, phase_detail) -> str:
    detail = str(phase_detail or '')
    if status == 'provisioning':
        return 'boot'
    if status == 'uploading':
        # The job is created/queued on the pod while the row still reads
        # 'uploading' — only start_job earns 'training'.
        return 'start' if detail.startswith('Job ') else 'upload'
    return 'offer' if detail.startswith('Searching for a GPU offer') else 'staging'


def launch_view(run, *, now=None, cloud_cfg=None):
    """The pre-training part of a run, as an ordered checklist with elapsed
    time — or None once the job is queued (the step counter takes over) or the
    run is finished.

    ``boot_idle_limit_seconds`` / ``boot_budget_seconds`` are the two real
    deadlines the boot wait enforces, exposed so the card can say how long a
    pod that never boots is allowed to keep the user waiting (run #134 died on
    'no boot progress for 25 min' with nothing on screen announcing it)."""
    if run.status not in _LAUNCH_STATES:
        return None
    c = cloud_cfg if cloud_cfg is not None else (cfg.get('cloud') or {})
    active = _active_launch_step(run.status, run.phase_detail)
    order = [k for k, _ in _LAUNCH_STEPS]
    idx = order.index(active)
    started = run.created_at or datetime.utcnow()
    elapsed = ((now if now is not None else datetime.utcnow()) - started).total_seconds()
    raw_budget = c.get('boot_budget_minutes')
    return {
        'active_step': active,
        'detail': run.phase_detail or '',
        'elapsed_seconds': max(0, int(elapsed)),
        'steps': [{'key': key, 'label': label,
                   'state': ('done' if i < idx else
                             'active' if i == idx else 'pending')}
                  for i, (key, label) in enumerate(_LAUNCH_STEPS)],
        'boot_idle_limit_seconds': (int(c.get('ready_timeout_minutes') or 0) * 60
                                    or READY_TIMEOUT_SECONDS),
        'boot_budget_seconds': int(90 if raw_budget is None
                                   else (raw_budget or 0)) * 60,
        # The upload's deadline is on IDLE BYTES, not on the transfer's total
        # duration, so announcing it is what stops a legitimately long upload
        # from reading like a countdown to being killed.
        'upload_stall_limit_seconds': _freeze_limit_seconds(run, c)
        if run.status == 'uploading' else 0,
    }


def _run_payload(run) -> dict:
    family = _run_family(run)
    training_mode = _run_training_mode(run)
    full_transformer = training_mode == 'full_transformer'
    # What ▶ Continue may offer. A LoRA resumes from the saves harvested on this
    # disk; a full model resumes from its Hugging Face copy (see
    # _dense_resume_candidates) — including from a run whose pod was kept, which
    # is precisely the state a dense run ends in when something went wrong and
    # the reason continuing it matters.
    if full_transformer:
        resume_pool = (_dense_resume_candidates(run)
                       if run.status in ('done', 'error_pod_kept') else [])
    else:
        resume_pool = (_run_staging_checkpoints(run)
                       if run.status == 'done' else [])
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
            'run_name': run.run_name, 'dataset_name': run_dataset_name(run),
            # The frozen dataset generation this run trains on (provenance
            # registry). Two parallel runs whose fingerprints differ are NOT
            # an A/B of settings any more — the chips say so. None on a
            # pre-registry row.
            'dataset_fingerprint': getattr(_cloud_run_record(run),
                                           'fingerprint', None),
            'vast_instance_id': run.vast_instance_id,   # for the per-run "console ↗" tooltip
            'phase_detail': run.phase_detail, 'gpu': run.gpu_name,
            'price_per_hour': run.price_per_hour,
            'cost_estimate': _cost_estimate(run), 'error': run.error,
            # isfile, not just a stored path: the user may delete staging
            # files by hand (Explorer) — a ready flag pointing at a missing
            # file yields a download button that 404s.
            'checkpoint_ready': bool(not full_transformer
                                     and run.checkpoint_local_path
                                     and os.path.isfile(run.checkpoint_local_path)),
            # A dense artifact may well be on this disk now, but it is never
            # offered as a browser download: re-serving 26 GB through the app
            # would only write a second copy of a file the user already has.
            # `local_artifact_dir` below says where it is instead.
            'checkpoint_local_path': (None if full_transformer else
                                      (os.path.basename(run.checkpoint_local_path)
                                       if run.checkpoint_local_path else None)),
            # card metrics: target steps (stamped launch param) + how many
            # checkpoints the pod saved (live count of the staging downloads)
            'steps': _run_param(run, 'steps'),
            'saves': _staging_save_count(run),
            # Why the 🧹 must skip this run, or None. Computed server-side so the
            # hub button and the backend can never disagree about a kept pod
            # whose recovery window has since closed (the frontend cannot know).
            'staging_spare_reason': staging_spare_reason(run),
            # Distinct steps of the harvested checkpoints still on disk — the
            # ▶ Continue dialog offers them so a finished run can resume from an
            # EARLIER epoch, not only its last (empty when they are all gone).
            'resume_steps': sorted({c['step'] for c in resume_pool}),
            'resume_checkpoints': [
                {'step': c['step'], 'resume_state': c['resume_state'],
                 'source': c.get('source') or 'local'}
                for c in resume_pool],
            'train_type': family, 'variant': variant,
            'training_mode': training_mode,
            'artifact_kind': (_run_param(run, 'artifact_kind')
                              or ('full_transformer' if full_transformer else 'lora')),
            'artifact_status': _run_param(run, 'artifact_status'),
            'artifact_status_detail': _run_param(run, 'artifact_status_detail'),
            'hf_repo_id': _run_param(run, 'hf_repo_id'),
            'hf_url': _run_param(run, 'hf_url'),
            'hf_weight_filename': _run_param(run, 'hf_weight_filename'),
            'hf_artifact_proof': _run_param(run, 'hf_artifact_proof'),
            # Post-training fp8 twin: the file to actually download for ComfyUI.
            # Absent keys = an older run, or an install with the export off —
            # the panel then says nothing rather than implying a failure.
            'fp8_export_status': _run_param(run, 'fp8_export_status'),
            'fp8_export_detail': _run_param(run, 'fp8_export_detail'),
            'fp8_weight_filename': _run_param(run, 'fp8_weight_filename'),
            'fp8_size_bytes': _run_param(run, 'fp8_size_bytes'),
            'fp8_keep_bf16': _run_param(run, 'fp8_keep_bf16'),
            # How to TEST the delivered model: the sample settings the run
            # itself previewed with, carried to whatever generates from it. The
            # WORDING follows the run's own base — a Turbo-based artifact must
            # not be described as "a RAW (undistilled) model", which is the one
            # thing nobody has measured about it.
            **({'inference_hint': lt.dense_inference_hint(
                _RunConfigDataset(None, 'krea', _run_param(run, 'variant'),
                                  _run_param(run, 'base_model') or ''))}
               if full_transformer else {}),
            'artifact_cleanup_status': _run_param(
                run, 'artifact_cleanup_status'),
            'artifact_cleanup_detail': _run_param(
                run, 'artifact_cleanup_detail'),
            'delivery_last_checked_at': _run_param(
                run, 'delivery_last_checked_at'),
            'verified_at': (_run_param(run, 'verified_at')
                            or _run_param(run, 'artifact_verified_at')),
            # Where THIS run's full model goes, and what actually landed there.
            # Absent keys = a run from before the local delivery existed; every
            # surface then reads it exactly as it always did (Hugging Face only).
            **({'dense_delivery': _dense_delivery(run),
                'local_artifact_status': _run_param(run, 'local_artifact_status'),
                'local_artifact_detail': _run_param(run, 'local_artifact_detail'),
                'local_artifact_dir': _run_param(run, 'local_artifact_dir'),
                'local_weight_filename': _run_param(run, 'local_weight_filename'),
                'local_weight_bytes': _run_param(run, 'local_weight_bytes'),
                'local_fp8_filename': _run_param(run, 'local_fp8_filename'),
                'local_fp8_bytes': _run_param(run, 'local_fp8_bytes'),
                'local_verified_at': _run_param(run, 'local_verified_at'),
                'hub_backup_status': _run_param(run, 'hub_backup_status'),
                'hub_backup_detail': _run_param(run, 'hub_backup_detail'),
                'hub_backup_warning': _run_param(run, 'hub_backup_warning'),
                'dense_fetch_active': bool(_dense_fetch_threads.get(int(run.id))),
                'can_fetch_local': _can_fetch_dense_locally(run),
                'resume_source': _run_param(run, 'resume_source'),
                } if full_transformer else {}),
            'artifact_delivery': (
                ('Downloaded to this computer'
                 + (' and backed up to a private Hugging Face repository'
                    if _dense_delivers_hub(run) else '')
                 if _dense_delivers_local(run) else
                 'Private Hugging Face repository; no checkpoint_local_path is '
                 'created for full_transformer artifacts.')
                if full_transformer else 'Local LoRA checkpoint'),
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
            # How long the run has reported nothing OBSERVABLE (not just how
            # long the monitor has been quiet — see _silent_seconds), and how
            # long it is allowed to (0 = the freeze watchdog is off). The card
            # warns on its own from these two, so a silent run is visible even
            # when the watchdog is configured never to cut.
            'idle_seconds': int(_silent_seconds(run) if run.status in ACTIVE_STATES
                                else _idle_seconds(run)),
            'idle_limit_seconds': (_freeze_limit_seconds(run)
                                   if run.status in ACTIVE_STATES else 0),
            # Byte counter of whatever the pod is fetching right now (base
            # weights are 26 GB — the phase users could not tell from a hang).
            # None whenever nothing parsable is in the log: the card then keeps
            # showing phase_detail, exactly as before.
            'download': (_download_progress(run)
                         if run.status in ACTIVE_STATES else None),
            # Ordered launch checklist + elapsed time, None once the job is
            # queued: what the user watches instead of a mute 'Launching…'.
            'launch': launch_view(run),
            'stop_requested': bool(run.stop_requested_at),
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
            final = video_training.split_checkpoint_name(c['filename'])[0] is None
            out.append({
                'step': c['step'], 'filename': c['filename'],
                'final': bool(final and crun.status == 'done'), 'present': True,
                'resume_state': c['resume_state'],
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
            'resume_state': c.get('resume_state'),
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


def training_activity() -> dict:
    """🏋️ Is anything training RIGHT NOW — locally or on a rented pod.

    Deliberately the cheapest question the app can ask: one persisted flag plus
    one indexed COUNT. No capability probe, no disk, no network — the nav bar
    polls this from every page, so it has to stay free. `cloud` is a count
    because several pods can train at once; `local` is a boolean because local
    training is single-flight."""
    cloud = (CloudTrainingRun.query
             .filter(CloudTrainingRun.status.in_(ACTIVE_STATES)).count())
    local = training_in_progress()
    return {'local': local, 'cloud': cloud, 'running': bool(local or cloud)}


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


def _deletable_deploy_names(dataset_id, family) -> dict:
    """{lowercased basename: filename as delete_imported_checkpoint whitelists it}
    for this dataset+family. The testable map names the deployed LoRA as the
    ComfyUI POOL sees it, while the deployed-delete route whitelists the names
    `list_imported_checkpoints` scans off disk — the same files, but the two
    forms can differ (path separator, subfolder prefix). Joining on the basename
    gives the UI a delete target the route will actually accept, instead of a
    name that resolves to 'unknown checkpoint'. Best-effort: {} on any failure,
    so a pill simply offers no deployed-delete rather than a doomed button."""
    try:
        rows = lt.list_imported_checkpoints(cfg.LOCAL_USER, dataset_id, family=family)
    except Exception:
        return {}
    out = {}
    for r in rows:
        fn = str(r.get('filename') or '')
        if fn:
            out[os.path.basename(fn.replace('\\', '/')).lower()] = fn
    return out


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
        fn = c['filename']
        s = _step_of_testable(fn)
        if s is not None:
            # A NUMBERED save must belong to the run we are answering for.
            # Without this, two runs of the same dataset+family that both saved
            # at step 2500 shared one key: importing run A's 2500 lit up "✓
            # Deployed" on run B's 2500 too, and B's Undeploy would have removed
            # A's file. Reported after importing step 2500 of one run and seeing
            # every other run's 2500 turn green.
            # Untagged files (imported before run tagging) still match on the
            # step alone — refusing them would un-deploy every legacy import.
            if run_tag and run_tag[1]:
                tag = lt.parse_deployed_run(fn)
                if tag[1] is not None and tag != tuple(run_tag):
                    continue
            out[s] = fn
        else:
            stepless.append(fn)
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
    """{step: {status, url, seed, count}} for a run's checkpoints. Each stored
    pointer resolves LIVE to its reused LoraTestImage: 'done' with a served url
    once the file exists, 'failed' if the cell failed, else 'pending' (the job is
    still in the serial queue). A dangling pointer (image row gone) is dropped so
    the node never claims a preview it can't show.

    Previews ACCUMULATE, so a checkpoint can hold several rows: the NEWEST one
    that still resolves is the thumbnail. `count` is not that list's length — it
    counts every finished test image linked to the checkpoint, wherever it came
    from (Test Studio, canvas, comparison grid), which is what the gallery under
    the node opens on."""
    from ..models import CheckpointPreview, LoraTestImage
    # Newest first, so the first resolvable row per step is the one shown.
    rows = (CheckpointPreview.query.filter_by(record_id=record_id)
            .order_by(CheckpointPreview.id.desc()).all())
    counts = dict(db.session.query(LoraTestImage.step, func.count(LoraTestImage.id))
                  .filter(LoraTestImage.record_id == record_id,
                          LoraTestImage.status == 'done',
                          LoraTestImage.filename.isnot(None))
                  .group_by(LoraTestImage.step).all())
    if not rows and not counts:
        return {}
    img_ids = [r.lora_test_image_id for r in rows if r.lora_test_image_id]
    imgs = ({i.id: i for i in LoraTestImage.query
             .filter(LoraTestImage.id.in_(img_ids)).all()} if img_ids else {})
    out = {}
    for r in rows:
        if r.step in out:
            continue                      # an older preview of the same checkpoint
        img = imgs.get(r.lora_test_image_id)
        if img is None:
            continue
        status = img.status if img.status in ('pending', 'done', 'failed') else 'pending'
        url = (f'/api/dataset/{r.dataset_id}/img/{img.filename}'
               if status == 'done' and img.filename else None)
        out[r.step] = {'status': status, 'url': url, 'seed': r.seed,
                       'count': int(counts.get(r.step) or 0)}
    # A checkpoint generated from the Test Studio has images but never a preview
    # pointer. It still has a gallery — the node must say so.
    for step, n in counts.items():
        if step is None or step in out:
            continue
        out[step] = {'status': None, 'url': None, 'seed': None, 'count': int(n)}
    return out


def canvas_generate(user_id, selections, **knobs) -> dict:
    """◉ Launch from the LoRA Canvas: the EXACT Test-Studio engine, told which
    checkpoints to run by the pills the user ticked instead of by a picker.

    `selections` = [{dataset_id, checkpoint, record_id, step}] — possibly across
    SEVERAL datasets, which is the point of the canvas
    (``LoraTestImage.run_id`` has always grouped cells of different datasets).
    Every other setting rides through untouched to ``create_comparison_run``,
    because it IS the same call the comparison grid makes: no second engine, so
    no drift between the two screens.

    Mixing FAMILIES is refused by the engine itself (one run = one base + one
    workflow) and the message travels back to the button.

    After the cells are created, one ``CheckpointPreview`` per distinct
    (record, step) points the node at what it just launched, so the pill shows
    ◌ rendering and then the picture. It is an INSERT, not an update: previews
    accumulate, and the older ones stay in the checkpoint's gallery."""
    from ..models import CheckpointPreview, LoraTestImage
    from . import lora_test_studio as studio

    res = studio.create_comparison_run(user_id, selections, **knobs)
    ids = res.get('ids') or []
    if ids:
        rows = LoraTestImage.query.filter(LoraTestImage.id.in_(ids)).all()
        seen = set()
        # Ordered by id so "the preview" is the first cell of the launch, a
        # stable choice rather than whatever the query happened to return first.
        for row in sorted(rows, key=lambda r: r.id):
            if row.record_id is None or row.step is None:
                continue                  # an unattributed pick has no node to sit under
            key = (row.record_id, row.step)
            if key in seen:
                continue
            seen.add(key)
            db.session.add(CheckpointPreview(
                record_id=row.record_id, step=row.step, dataset_id=row.dataset_id,
                lora_test_image_id=row.id, prompt=row.prompt or '', seed=row.seed))
        db.session.commit()
    return res


def checkpoint_gallery(record_id, step, limit=120) -> dict:
    """Every finished image this checkpoint ever produced, newest first — the
    gallery the canvas opens under a node.

    The source is the LINK written at generation time
    (``lora_test_image.record_id`` / ``.step``), so it holds whatever made the
    image: an inline canvas preview, a Test-Studio grid cell, a comparison run.
    Nothing is parsed out of a filename here — that is the whole point of the
    columns.

    `unlinked` is the honest footnote: images that exist but carry no link (they
    predate the columns and their filename did not attribute itself). They are
    NOT shown under a checkpoint they might not belong to; the number is
    reported so the gap is stated instead of looking like an empty history."""
    from ..models import LoraTestImage
    q = (LoraTestImage.query
         .filter(LoraTestImage.record_id == record_id,
                 LoraTestImage.step == step,
                 LoraTestImage.status == 'done',
                 LoraTestImage.filename.isnot(None))
         .order_by(LoraTestImage.id.desc()))
    total = q.count()
    rows = q.limit(max(1, min(int(limit or 120), 500))).all()
    from .checkpoint_link_backfill import unlinked_count
    from . import trash
    return {
        'record_id': record_id, 'step': step, 'count': total,
        'unlinked': unlinked_count(),
        # Where a deleted image WOULD land, resolved the same way the deletion
        # resolves it, so the confirmation never promises the wrong thing.
        'delete_mode': trash.disposal_mode(),
        'images': [_gallery_image(r) for r in rows],
    }


def _gallery_image(r) -> dict:
    """One image row as the galleries publish it. Extracted so the checkpoint
    gallery and the run gallery can never drift into two shapes — the panel is
    ONE component and it reads these keys."""
    return {
        'id': r.id,
        'dataset_id': r.dataset_id,
        'url': f'/api/dataset/{r.dataset_id}/img/{r.filename}',
        'rating': r.rating,
        'prompt': r.prompt,
        'seed': r.seed,
        'strength': r.strength,
        'step': r.step,
        # WHICH checkpoint made it. Redundant inside a checkpoint gallery (the
        # scope already says so) and load-bearing outside one: a pinned canvas
        # node draws its link to the source pill from these two, so the link
        # cannot drift from the image.
        'record_id': r.record_id,
        # WHICH LAUNCH made it. Already grouped every cell of one "Generate"
        # (it is what a grid resumes from) and never left the database. The
        # canvas needs it: without it, two runs fired at the SAME checkpoint
        # were indistinguishable, so pinning the second one appended its
        # pictures to the first one's strip and the board showed one lot where
        # there were two. Null on images that predate the column.
        'run_id': r.run_id,
        'created_at': r.created_at.isoformat() if r.created_at else None,
        # ── What the image was actually MADE with ────────────────────────────
        # Every one of these was already persisted per cell (for a faithful
        # resume) and none of it reached the viewer, which on a board whose
        # whole job is comparing checkpoints is the wrong half to hide: two
        # renders that differ only by sampler or CFG look like a checkpoint
        # difference until you can read the settings. Nothing new is computed
        # here — these are columns, published.
        'checkpoint': r.checkpoint,
        'base_model': r.z_model,
        'negative': r.negative,
        'cfg': r.cfg,
        'steps': r.steps,
        'sampler': r.sampler,
        'scheduler': r.scheduler,
        'aspect': r.aspect,
        'extra_loras': r.extra_loras,
        'face_score': r.face_score,
        # ✨ Whether this row IS an Upscale & improve result, and of what. The
        # galleries and canvas_image_nodes read this table WITHOUT the studio's
        # `_cells()` filter — deliberately, because showing the improvement next
        # to its source and letting it be pinned is the point of the button. The
        # front needs the two keys for the same reason the dataset grid does:
        # to say so on the tile, and to refuse to improve an improvement before
        # the click rather than through a 400 after it.
        'derivation_kind': r.derivation_kind,
        'parent_image_id': r.parent_image_id,
    }


# How many images ONE step contributes to a run gallery, and how many the whole
# answer may carry. A run with 14 checkpoints × every render it ever got is a
# payload nobody reads and a grid that stutters on a phone; a per-step slice
# keeps every checkpoint represented instead of letting the newest one eat the
# whole budget. Both caps are REPORTED (`truncated`), never silent.
RUN_GALLERY_PER_STEP = 30
RUN_GALLERY_LIMIT = 240


def run_gallery(record_id, limit=RUN_GALLERY_LIMIT,
                per_step=RUN_GALLERY_PER_STEP) -> dict:
    """Everything ONE RUN ever produced, grouped by the checkpoint that made it —
    the gallery the ◉ Canvas opens on a run card.

    Same link, same rows and the same delete as ``checkpoint_gallery``: this is
    the checkpoint gallery with its scope widened, not a second reader. A run's
    images are the union of its checkpoints' images, so two independent readers
    would be two chances to disagree about what exists.

    Order: **steps descending**, the most-trained checkpoint first. The pill row
    on the board already reads left-to-right in training order; what the panel is
    for is judging where the LoRA stopped getting better, and that judgement
    starts at the end. Inside a step the app's rule is unchanged — newest first.

    ``step: null`` is a real group, LAST: images attributable to this RUN but not
    to a step (a step-less final save carries the run tag in its name and no
    step). They used to be counted as "not traced back to a checkpoint", which
    was true and useless — the run is knowable, so it is stated.

    Counts are exact; the IMAGES are capped per step and overall, and each group
    says whether it was cut (`truncated`). `count` is always the run's real
    total.

    The checkpoint NOTES ride along, read from ``checkpoint_note`` rather than
    from the pills: a note is keyed by (record_id, step) and outlives the file it
    describes, so a run whose saves have been cleaned off the disk would
    otherwise show its images and silently lose what was written about them."""
    from ..models import CheckpointNote, LoraTestImage
    from .checkpoint_link_backfill import unlinked_count
    from . import trash

    base = LoraTestImage.query.filter(
        LoraTestImage.record_id == record_id,
        LoraTestImage.status == 'done',
        LoraTestImage.filename.isnot(None))
    counted = (db.session.query(LoraTestImage.step, func.count(LoraTestImage.id))
               .filter(LoraTestImage.record_id == record_id,
                       LoraTestImage.status == 'done',
                       LoraTestImage.filename.isnot(None))
               .group_by(LoraTestImage.step).all())
    # Steps descending, the step-less group last — sorted here rather than in SQL
    # because NULL ordering is dialect-dependent and this list is a handful long.
    order = sorted(((s, n) for s, n in counted),
                   key=lambda t: (t[0] is None, -(t[0] or 0)))

    notes = {n.step: (n.note or '').strip()
             for n in CheckpointNote.query.filter_by(record_id=record_id).all()
             if (n.note or '').strip()}
    cap = max(1, min(int(limit or RUN_GALLERY_LIMIT), 800))
    per = max(1, min(int(per_step or RUN_GALLERY_PER_STEP), cap))
    groups, shown, total = [], 0, 0
    for step, n in order:
        total += n
        room = max(0, cap - shown)
        take = min(per, n, room)
        rows = []
        if take:
            q = base.filter(LoraTestImage.step.is_(None) if step is None
                            else LoraTestImage.step == step)
            rows = q.order_by(LoraTestImage.id.desc()).limit(take).all()
            shown += len(rows)
        groups.append({
            'step': step, 'count': n,
            'truncated': len(rows) < n,
            'note': notes.get(step) or '',
            'images': [_gallery_image(r) for r in rows],
        })
    return {
        'record_id': record_id, 'count': total, 'shown': shown,
        # Every checkpoint note of the run, including the steps that produced no
        # image at all — those have no group to hang off, and a note nobody can
        # read is a note that was lost.
        'checkpoint_notes': [{'step': s, 'note': notes[s]}
                             for s in sorted(notes, reverse=True)],
        'truncated': shown < total,
        'per_step': per, 'limit': cap,
        # The honest footnote stays: images that carry no link AT ALL, not even a
        # run. Widening the scope must not make that number disappear.
        'unlinked': unlinked_count(),
        'delete_mode': trash.disposal_mode(),
        'groups': groups,
    }


def delete_checkpoint_images(record_id, step, image_ids) -> dict:
    """🗑 Delete generated images from a checkpoint's gallery — file AND row.

    These rows ARE the Test Studio's cells: one image lives in exactly one place
    in the database and is shown by two surfaces. So "remove it from this
    gallery" and "delete it" cannot both be true, and hiding it here while it
    stays in the Studio grid would only move the confusion. The gallery deletes
    for real, and the confirmation says the Test Studio loses them too — the
    consequence is stated BEFORE the click, not discovered after it.

    Nothing is destroyed outright: the file goes through ``trash.dispose`` (OS
    recycle bin → the app's own trash → a permanent unlink only if both refuse),
    exactly like the bank's rejected sweep. ``checkpoint_gallery`` reports the
    mode up front and this returns the mode actually used.

    Scoped to the checkpoint: an id that is not linked to (record_id, step) is
    refused, so this route cannot be turned into "delete any image by id".

    ``step=None`` widens the scope to the WHOLE RUN — every step of it plus the
    step-less group — which is what the run gallery deletes with. It is the same
    function on purpose: two delete paths over the same rows would be two places
    to keep the recycle-bin promise, the shared-file rule and the "generating is
    never cancelled" rule true, and they would drift.

    Degrades instead of failing, because the gallery of a real install is never
    tidy:
      • a row still generating keeps its file (skipped 'generating') — cancelling
        someone's running job is not what a delete click asked for;
      • a row whose file is already gone from disk still loses its row;
      • a file another surviving row also points at is unlinked from THIS row but
        left on disk (no surface loses a picture it still lists);
      • a file that cannot be moved keeps its row, so it stays visible and
        retryable rather than vanishing from the UI while it sits on disk.

    ``CheckpointPreview`` rows pointing at a deleted image are removed in the
    same transaction. The reader already drops a dangling pointer, but leaving
    one behind means a checkpoint silently falls back to an older preview with no
    trace of why — and the row would outlive every image it could ever resolve.

    Returns {'mode', 'deleted', 'trashed', 'already_absent', 'rows_removed',
    'previews_removed', 'dataset_ids', 'skipped': [{'id', 'reason'}]}."""
    from ..models import CheckpointPreview, LoraTestImage
    from . import trash
    wanted = []
    for i in (image_ids or []):
        try:
            wanted.append(int(i))
        except (TypeError, ValueError):
            continue
    out = {'mode': None, 'deleted': 0, 'trashed': 0, 'already_absent': 0,
           'rows_removed': 0, 'previews_removed': 0, 'dataset_ids': [],
           'skipped': []}
    if not wanted:
        return out
    scoped = LoraTestImage.query.filter(LoraTestImage.record_id == record_id,
                                        LoraTestImage.id.in_(wanted))
    if step is not None:
        scoped = scoped.filter(LoraTestImage.step == step)
    rows = scoped.all()
    found = {r.id for r in rows}
    for i in wanted:
        if i not in found:
            # Either a stale gallery (already deleted elsewhere) or an id that
            # belongs to another checkpoint. Same answer: not ours to delete.
            out['skipped'].append({'id': i, 'reason': 'not_in_gallery'})

    # A file can be pointed at by more than one row (a preview reuses an existing
    # cell). Only unlink from disk what nothing else still lists.
    keys = {(r.dataset_id, r.filename) for r in rows if r.filename}
    still_used = set()
    if keys:
        others = (LoraTestImage.query
                  .filter(LoraTestImage.dataset_id.in_({k[0] for k in keys}),
                          LoraTestImage.filename.in_({k[1] for k in keys}),
                          LoraTestImage.id.notin_(list(found))).all())
        still_used = {(o.dataset_id, o.filename) for o in others}

    remove_ids, modes_used, datasets = [], set(), set()
    disposed = set()                       # (dataset_id, filename) done this pass
    for row in rows:
        if row.status not in ('done', 'failed', 'cancelled'):
            out['skipped'].append({'id': row.id, 'reason': 'generating'})
            continue
        datasets.add(row.dataset_id)
        if not row.filename:
            # A failed/cancelled cell never wrote a file: only the row goes.
            remove_ids.append(row.id)
            continue
        key = (row.dataset_id, row.filename)
        path = os.path.join(fds._dataset_path(row.dataset_id), row.filename)
        if key in still_used or key in disposed:
            remove_ids.append(row.id)      # someone else still shows this picture
            continue
        if not os.path.exists(path):
            out['already_absent'] += 1
            remove_ids.append(row.id)
            continue
        try:
            mode = trash.dispose(path, context=(f'run-{record_id}' if step is None
                                                else f'checkpoint-{record_id}-{step}'))
        except OSError as e:
            out['skipped'].append({'id': row.id, 'reason': str(e)})
            continue
        disposed.add(key)
        modes_used.add(mode)
        if mode == 'delete':
            out['deleted'] += 1
        else:
            out['trashed'] += 1            # OS trash or app trash — recoverable
        remove_ids.append(row.id)

    if remove_ids:
        out['previews_removed'] = CheckpointPreview.query.filter(
            CheckpointPreview.lora_test_image_id.in_(remove_ids)
        ).delete(synchronize_session=False)
        LoraTestImage.query.filter(
            LoraTestImage.id.in_(remove_ids)).delete(synchronize_session=False)
        out['rows_removed'] = len(remove_ids)
        db.session.commit()
    out['dataset_ids'] = sorted(datasets)
    # Report the WORST outcome: one permanently removed file makes the whole run
    # 'delete', whatever the rest did. The UI wording follows this.
    for mode in ('delete', 'app_trash', 'trash'):
        if mode in modes_used:
            out['mode'] = mode
            break
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
        strengths=[1.0], seed=seed, prompt=prompt, family=fam, count=1,
        # The caller KNOWS which lineage checkpoint each file is — it was just
        # resolved above — so every cell records it rather than the app deriving
        # it back from the filename later.
        origins={fn: {'record_id': rid, 'step': step} for rid, step, fn in resolved})
    ids = result.get('ids') or []
    run_seed = result.get('seed', seed)
    # Single base model × single strength × count=1 → cells are 1:1 with `resolved`
    # in order; zip guards against any engine-side short count (never a wrong link).
    #
    # A NEW row per generation: previews accumulate (the unique constraint that
    # forced one row per checkpoint was lifted — see
    # services.checkpoint_preview_migration). Regenerating an epoch used to
    # re-point the single row and the earlier image became unreachable.
    for (rid, step, _fn), img_id in zip(resolved, ids):
        db.session.add(CheckpointPreview(
            record_id=rid, step=step, dataset_id=dataset_id,
            lora_test_image_id=img_id, prompt=prompt or '', seed=run_seed))
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


def _record_removal_blocker(rec) -> str | None:
    """Why removing this run from the graph must be refused, or None.

    ``'has_saves'`` — its checkpoints are still on this disk.
    ``'has_model'`` — it is a full model that lives in its Hugging Face
    repository. The disk count cannot see that one (``checkpoint_local_path`` is
    a LoRA-lane column), which is exactly how a hub-delivered dense run came to
    be offered for removal under the words "No checkpoints left on disk".
    """
    if _record_checkpoints_on_disk(rec) > 0:
        return 'has_saves'
    try:
        crun = (db.session.get(CloudTrainingRun, rec.cloud_run_id)
                if rec.source == 'cloud' and rec.cloud_run_id else None)
        if crun is not None and dense_artifact_state(crun) == DENSE_ON_HUB:
            return 'has_model'
    except Exception:                                   # noqa: BLE001
        logger.debug('dense removal check failed for %s', rec.id, exc_info=True)
    return None


def _releasable_blob_sigs(rec) -> set:
    """Content hashes archived for `rec` that NO OTHER run references.

    `run_archive` is content-addressed and therefore shared on purpose: an
    unchanged dataset trained ten times stores its images ONCE, which is why a
    whole training history stays small. So "free this run's archived images"
    can only mean the blobs whose last referrer is this run — anything else
    would blank the image comparison of a run nobody asked to touch.

    The reference is the run snapshot (`run_snapshot.signatures`). Every path
    that cannot be accounted for cleanly returns an EMPTY set — nothing is
    deleted — rather than guessing:
      • this run has no snapshot (legacy run): its blobs are unattributable;
      • the `snapshot` column doesn't exist yet (a database whose boot migration
        hasn't run): the sweep would raise, so it is caught and yields nothing.
    Extra archived bytes are cheap; a hole in the archive is not."""
    from ..models import TrainingRunRecord
    from . import run_snapshot
    try:
        mine = run_snapshot.signatures(run_snapshot.loads(rec))
        if not mine:
            return set()
        rows = db.session.query(TrainingRunRecord.snapshot).filter(
            TrainingRunRecord.id != rec.id).all()
        for (raw,) in rows:
            mine -= run_snapshot.signatures_of_raw(raw)
            if not mine:
                break
        return mine
    except Exception:
        logger.debug('archive reference accounting failed — releasing nothing',
                     exc_info=True)
        return set()


def run_deletion_impact(record_id) -> dict | None:
    """What removing this run would actually take with it, COUNTED — the payload
    the confirmation dialog reads so a destructive action is announced before it
    happens, not discovered after.

    Every count degrades to 0 on its own rather than failing the preview: a
    fresh install with empty tables, a run that was never previewed or tested,
    an archive that doesn't exist. `None` when the run is unknown."""
    from ..models import (TrainingRunRecord, CheckpointNote, CheckpointPreview,
                          LoraTestImage, CanvasNodePosition)
    from . import run_archive
    rec = db.session.get(TrainingRunRecord, int(record_id))
    if rec is None:
        return None

    def _count(model):
        try:
            return int(model.query.filter_by(record_id=rec.id).count())
        except Exception:
            logger.debug('deletion impact count failed for %s', model, exc_info=True)
            return 0

    try:
        released = run_archive.stored_count(_releasable_blob_sigs(rec))
    except Exception:
        released = 0
    # The CASCADE half of the same preview: what a "delete everything" would take
    # that the conservative removal would not. Additive — the existing dialog
    # reads the flat keys and is untouched by it — and self-degrading to a
    # zeroed block, so a probe that cannot run never breaks the preview.
    try:
        from . import run_cascade_delete
        cascade = run_cascade_delete.cascade_impact(rec.id)
    except Exception:
        logger.debug('cascade impact preview failed', exc_info=True)
        cascade = None
    return {
        'record_id': rec.id,
        'has_saves': _record_checkpoints_on_disk(rec) > 0,
        # Why a removal would be refused, if it would ('has_saves' | 'has_model').
        # Additive: the dialog reads the flat keys and is untouched by it.
        'removal_blocker': _record_removal_blocker(rec),
        'cascade': cascade or {
            'checkpoints': 0, 'checkpoint_bytes': 0, 'images_deleted': 0,
            'images_kept_rated': 0, 'deployed_kept': 0, 'training_active': None},
        'notes': _count(CheckpointNote),
        'previews': _count(CheckpointPreview),
        'images_unlinked': _count(LoraTestImage),
        'canvas_positions': _count(CanvasNodePosition),
        'children_detached': _count_children(rec),
        'archived_images_released': released,
    }


def _count_children(rec) -> int:
    from ..models import TrainingRunRecord
    try:
        return int(TrainingRunRecord.query
                   .filter_by(parent_record_id=rec.id).count())
    except Exception:
        return 0


def delete_run_record(record_id, cascade=False) -> str:
    """Remove a GONE run from the lineage graph, with EVERYTHING that only
    existed because of it. Five tables carry a `record_id`; leaving three of
    them behind is how a "deleted" run keeps haunting the canvas and the
    checkpoint gallery.

    What goes, and why it goes that way:
      • `TrainingRunRecord` + its `CheckpointNote`s + its `CheckpointPreview`
        links (a checkpoint may hold several previews since the uniqueness
        constraint was lifted) + its `CanvasNodePosition` (a board coordinate
        for a card that no longer exists);
      • children that resumed FROM this run are DETACHED (parent_record_id →
        NULL), staying in the graph as honest "origin unknown" roots;
      • `LoraTestImage` rows are UNLINKED (`record_id`/`step` → NULL), never
        deleted. Those are real generated pictures that also live in the Test
        Studio and the canvas gallery; removing a run is a tidying of lineage,
        not an order to destroy images the user never said to destroy. They lose
        their provenance, and the confirmation dialog says so up front;
      • archived source blobs are released ONLY when this run was their last
        referrer (`_releasable_blob_sigs`) — the store is shared between runs and
        a naive delete would blank another run's comparison.

    Guards kept: a run whose checkpoints are still on disk is REFUSED
    ('has_saves') so a recoverable run is never discarded from under the user.
    A full model that lives in its Hugging Face repository is refused the same
    way ('has_model') — its files are not on this disk and never were, which is
    precisely why the disk count alone let it through.

    `cascade=True` is the ONE caller allowed past that guard:
    ``run_cascade_delete.delete_run_cascade`` has just moved those checkpoints to
    the trash itself, so re-asking "are they on disk?" would either refuse a
    deletion whose files are already gone or race a slow filesystem. It is a
    keyword with a False default so every existing caller keeps the conservative
    behaviour byte for byte — the cascade is an explicitly requested mode, never
    a new default that starts destroying files under code that never asked.

    Returns 'not_found' | 'has_saves' | 'has_model' | 'deleted' | 'conflict'.
    The FK children
    (no relationship cascade in this schema) are deleted and FLUSHED before the
    parent row so SQLite never raises the repo's "delete 500" IntegrityError; a
    stray one is caught and reported as 'conflict', never a 500. Blobs are
    touched only AFTER the commit succeeds — a filesystem hiccup must never roll
    back a database deletion, and vice versa."""
    from ..models import (TrainingRunRecord, CheckpointNote, CheckpointPreview,
                          LoraTestImage, CanvasNodePosition)
    from . import run_archive
    from sqlalchemy.exc import IntegrityError
    rec = db.session.get(TrainingRunRecord, int(record_id))
    if rec is None:
        return 'not_found'
    if not cascade:
        blocker = _record_removal_blocker(rec)
        if blocker:
            return blocker
    # Computed BEFORE the row is gone — the snapshot that names the blobs lives
    # on the record itself.
    releasable = _releasable_blob_sigs(rec)
    try:
        # Detach any run that resumed from this one BEFORE deleting it: the child
        # stays displayed (as a root), the parent edge just disappears.
        (TrainingRunRecord.query
         .filter_by(parent_record_id=rec.id)
         .update({'parent_record_id': None}, synchronize_session=False))
        # Generated images survive their run: only the provenance link is cut.
        (LoraTestImage.query
         .filter_by(record_id=rec.id)
         .update({'record_id': None, 'step': None}, synchronize_session=False))
        # Delete FK children first and flush, so deleting the parent row can't hit
        # an IntegrityError (the "delete 500" trap — no cascade on these tables).
        CheckpointNote.query.filter_by(record_id=rec.id).delete(
            synchronize_session=False)
        CheckpointPreview.query.filter_by(record_id=rec.id).delete(
            synchronize_session=False)
        CanvasNodePosition.query.filter_by(record_id=rec.id).delete(
            synchronize_session=False)
        db.session.flush()
        db.session.delete(rec)
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return 'conflict'
    try:
        run_archive.release(releasable)
    except Exception:
        logger.debug('archived blobs could not be released', exc_info=True)
    return 'deleted'


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
        node['training_mode'] = _run_training_mode(crun)
        if _is_full_transformer_run(crun):
            # A full model is not addressed by checkpoint_local_path (see
            # dense_artifact_state). `None` for a model that lives on the Hub is
            # the tri-state every existing reader already treats as "cannot say":
            # the chip stays quiet and `isRunDeletable` refuses — which is the
            # right behaviour on an older frontend too, without a version check.
            state = dense_artifact_state(crun)
            node['dense_artifact'] = state
            node['checkpoint_ready'] = {DENSE_ON_DISK: True, DENSE_ON_HUB: None,
                                        DENSE_GONE: False}[state]
        else:
            node['checkpoint_ready'] = bool(
                crun.checkpoint_local_path and os.path.isfile(crun.checkpoint_local_path))
        node['checkpoints'] = _node_checkpoints(rec, crun)
        node['saves'] = _staging_save_count(crun)
    else:
        # A local record is always a LoRA run: dense training is refused outside
        # the cloud lane (routes.training), so there is no local dense to detect.
        node['training_mode'] = 'lora'
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
    # Deployment (testable + the deployed copy's own name) comes from the SHARED
    # annotator, so the graph pills and the Checkpoints panel rows answer "is this
    # deployed, and which ComfyUI file is it?" with the same join. Scoped to THIS
    # run so its step-less final deploy (`..._rc90_v2`, no step in the name) joins
    # its own final pill instead of going unmatched.
    annotate_deployed_checkpoints(rec.dataset_id, rec.family,
                                  node.get('checkpoints') or [],
                                  run_tag=_deployed_run_tag(rec))
    for _ck in (node.get('checkpoints') or []):
        _step = _ck.get('step')
        _ck['note'] = _cnotes.get(_step, '')
        # `preview_*` render the inline thumbnail (or its pending/failed state);
        # `preview_count` is the SIZE of the checkpoint's gallery — every image it
        # ever produced, from any surface — which the pill shows as a × N badge.
        _pv = _cprev.get(_step)
        if _pv:
            _ck['preview_url'] = _pv.get('url')
            _ck['preview_status'] = _pv.get('status')
            _ck['preview_count'] = _pv.get('count') or 0
    return node


def annotate_deployed_checkpoints(dataset_id, family, checkpoints,
                                  run_tag=None) -> list:
    """Stamp `testable` and, when deployed, `deployed_filename` onto a flat list
    of a run's saves — IN PLACE, returning the same list.

    This is THE join between "a save on disk" and "its copy in ComfyUI", and it
    has exactly one implementation on purpose: the ◉ Graph pills and the
    Checkpoints & LoRAs rows must never disagree about which checkpoints are
    deployed, nor about which file an undeploy would remove. `testable` decides
    "✓ Deployed vs 📦 Import"; `deployed_filename` is the ONLY handle the UI has
    on the ComfyUI copy (without it the delete route answers "unknown checkpoint"
    and the action is withheld) — it is resolved to the form that route accepts.

    `run_tag` ((source, run_id), see _deployed_run_tag) additionally attaches a
    run's STEP-LESS final deploy to its final save. Best-effort throughout: an
    unreadable ComfyUI pool leaves every row "not deployed" rather than claiming
    a deployment that isn't there."""
    cks = list(checkpoints or [])
    testable = _testable_by_step(dataset_id, family, run_tag=run_tag,
                                 final_step=_final_step_of(cks))
    names = _deletable_deploy_names(dataset_id, family) if testable else {}
    for ck in cks:
        step = ck.get('step')
        ck['testable'] = step in testable
        dep = testable.get(step)
        if dep:
            ck['deployed_filename'] = names.get(
                os.path.basename(str(dep).replace('\\', '/')).lower())
    return cks


def annotate_deployed_by_run(dataset_id, family, checkpoints) -> list:
    """`annotate_deployed_checkpoints` for a MIXED list whose rows name their own
    source run (the Checkpoints panel's local list: several runs' saves in one
    flat list). Rows are grouped by (run_source, run_id) so each group is joined
    with ITS run tag — a step-less final save then attaches to the run that
    produced it, never to a neighbour. Rows with no recorded run (pre-registry
    files) are joined untagged, which still matches every step-named deploy."""
    groups = {}
    for ck in (checkpoints or []):
        rid = ck.get('run_id')
        src = ck.get('run_source')
        key = (src, rid) if rid and src else (None, None)
        groups.setdefault(key, []).append(ck)
    for (src, rid), rows in groups.items():
        annotate_deployed_checkpoints(dataset_id, family, rows,
                                      run_tag=(src, rid) if rid else None)
    return list(checkpoints or [])


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


def canvas_dataset_index(user_id) -> dict:
    """The LoRA Canvas' INDEX: every dataset of `user_id` that produced at least
    one training run, with its run count and the families it covers.

    Deliberately cheap — two grouped queries, no checkpoints, no disk. The canvas
    draws its dataset filter from this and then pulls each shown dataset's
    genealogy through the existing per-dataset lineage endpoint. Assembling the
    whole forest in one response would have to scan every run's saves on disk
    before ANYTHING could appear on screen, and a library of thirty datasets
    would stare at a spinner for it. Datasets with no run are omitted: there is
    nothing to draw for them, and offering them in the filter would only be a
    list of dead ends.

    Ordered newest-run-first, so the board opens on what was trained recently."""
    from sqlalchemy import func
    from ..models import TrainingRunRecord
    from . import lora_test_studio as studio
    datasets = {d.id: d for d in fds.list_datasets(user_id)}
    if not datasets:
        return {'datasets': []}
    ids = list(datasets)
    rows = (db.session.query(TrainingRunRecord.dataset_id,
                             func.count(TrainingRunRecord.id),
                             func.max(TrainingRunRecord.created_at))
            .filter(TrainingRunRecord.dataset_id.in_(ids))
            .group_by(TrainingRunRecord.dataset_id).all())
    fams = {}
    for ds_id, fam in (db.session.query(TrainingRunRecord.dataset_id,
                                        TrainingRunRecord.family)
                       .filter(TrainingRunRecord.dataset_id.in_(ids))
                       .distinct().all()):
        if fam:
            fams.setdefault(ds_id, set()).add(fam)
    out = []
    for ds_id, runs, last_at in rows:
        ds = datasets.get(ds_id)
        if ds is None:
            continue
        out.append({
            'id': ds_id,
            'name': ds.name,
            'runs': int(runs or 0),
            'families': sorted(fams.get(ds_id) or ()),
            'last_run_at': last_at.isoformat() if last_at else None,
            # The ★ pinned LoRA(s), one per family. Read off the dataset row we
            # ALREADY hold — no extra query, no disk. Without it a canvas delete
            # of the pinned checkpoint was confirmed with the plain wording, and
            # the ⚠ "this is your saved winning combo" line never appeared:
            # same route, same trash, but the user was not told what they were
            # about to break.
            'best_settings_loras': studio.best_settings_lora_filenames(ds),
            # 🪪 The dataset's REFERENCE face, and what kind of dataset it is.
            # Read off the row already in hand — no extra query, no disk, so
            # the "cheap by design" invariant above still holds. The canvas
            # draws it beside the lane's name: a board full of renders of a
            # person with the person nowhere on it made every comparison a
            # memory test. Only meaningful for a character dataset (a concept
            # or a style has no reference face); `kind` travels so the canvas
            # decides that instead of guessing from a filename.
            'ref_filename': ds.ref_filename,
            'kind': (ds.kind or '').lower() or 'character',
            # 🧬 The word this dataset's LoRA answers to. Read off the same row —
            # still no extra query, no disk. A blend launched from the board
            # injects the trigger of EVERY dataset it stacks, and a panel that
            # cannot name them can only promise it silently.
            'trigger_word': (ds.trigger_word or None),
        })
    out.sort(key=lambda d: (d['last_run_at'] or '', d['id']), reverse=True)
    return {'datasets': out}


# --- ◉ LoRA Canvas: remembered card positions -------------------------------
#
# The canvas draws its trees with the automatic layout and lets the user drag a
# card off it. These three functions are the whole persistence story: read the
# board's overrides, upsert some, drop a lane's. Everything about WHICH cards get
# a row and where they land is decided client-side by the pure placement layer
# (frontend/src/utils/canvasPlacement.js) — the server stores coordinates and
# asks no questions, so the geometry stays testable without a browser.
#
# Positions are a display preference. A failed write must never interrupt what
# the user is doing (design: "nothing about the canvas may block the canvas"),
# which is why these are plain, boring upserts with no side effects.

def canvas_positions(user_id, dataset_ids=None) -> dict:
    """Every remembered card position of `user_id`, grouped by dataset id.

    One request for the whole board on purpose: the canvas opens on N lanes and
    N round-trips for a handful of tiny rows would be slower than the genealogy
    fetches they have to be ready before. `dataset_ids` narrows it when the
    caller already knows the lanes it wants."""
    from ..models import CanvasNodePosition
    owned = {d.id for d in fds.list_datasets(user_id)}
    if dataset_ids is not None:
        owned &= {int(i) for i in dataset_ids}
    if not owned:
        return {'positions': {}}
    rows = (CanvasNodePosition.query
            .filter(CanvasNodePosition.dataset_id.in_(list(owned))).all())
    out = {}
    for r in rows:
        out.setdefault(str(r.dataset_id), []).append(
            {'record_id': r.record_id, 'x': float(r.x), 'y': float(r.y)})
    for lane in out.values():
        lane.sort(key=lambda p: p['record_id'])
    return {'positions': out}


def save_canvas_positions(user_id, dataset_id, positions) -> dict:
    """Upsert card positions for one lane. Returns how many rows the lane holds.

    Idempotent by (dataset_id, record_id) — the canvas re-sends a position on
    every drop and re-pins the same coordinates whenever a lane gains a run, so
    a second identical write must be a no-op rather than a duplicate row.
    Non-finite coordinates are rejected outright: one NaN stored here would make
    a card unreachable on every future load, and there is no UI to fix it."""
    from ..models import CanvasNodePosition
    if not fds.get_dataset(user_id, dataset_id):
        raise LookupError('dataset not found')
    wanted = {}
    for p in (positions or []):
        try:
            rid = int(p['record_id'])
            x, y = float(p['x']), float(p['y'])
        except (KeyError, TypeError, ValueError):
            continue
        if not (x == x and y == y and abs(x) != float('inf') and abs(y) != float('inf')):
            continue
        wanted[rid] = (x, y)
    if wanted:
        existing = {r.record_id: r for r in CanvasNodePosition.query.filter(
            CanvasNodePosition.dataset_id == dataset_id,
            CanvasNodePosition.record_id.in_(list(wanted))).all()}
        for rid, (x, y) in wanted.items():
            row = existing.get(rid)
            if row is None:
                db.session.add(CanvasNodePosition(
                    dataset_id=dataset_id, record_id=rid, x=x, y=y))
            else:
                row.x, row.y = x, y
        db.session.commit()
    return {'saved': len(wanted),
            'total': CanvasNodePosition.query.filter_by(dataset_id=dataset_id).count()}


def clear_canvas_positions(user_id, dataset_id) -> dict:
    """✦ Tidy up for one lane: drop every remembered position so the automatic
    tree takes over again. The escape hatch — an arrangement tangled over twenty
    runs has to have a way back that is not "edit the database"."""
    from ..models import CanvasNodePosition
    if not fds.get_dataset(user_id, dataset_id):
        raise LookupError('dataset not found')
    removed = CanvasNodePosition.query.filter_by(
        dataset_id=dataset_id).delete(synchronize_session=False)
    db.session.commit()
    return {'cleared': int(removed or 0)}


# 🖼 Bounds for a pinned image node, in the board's WORLD units (a run card is
# CARD_W = 264 wide, for scale). The floor keeps a node grabbable at any zoom;
# the ceiling is the one that matters — a node resized to 8 000 px would blow
# up its lane's extent, and ✦ Fit would then collapse the whole board to a scale
# where nothing else is readable. Enforced here as well as in the browser: the
# clamp protects the NEXT load, not just the gesture.
CANVAS_IMAGE_MIN = 96.0
CANVAS_IMAGE_MAX = 1400.0

# 🖼 How far from its lane's origin a pinned image may be parked, on either axis
# and in EITHER DIRECTION. Negative coordinates are legal: a picture is not a
# step of the lineage, and the wall at zero was what stopped a render from being
# dragged above its own lane or into the free margin beside the board. Mirrors
# IMG_REACH in frontend/src/utils/canvasImageNodes.js.
#
# A safety rail, not a design limit. The position axes had no ceiling at all
# until now — only the SIZE was bounded — so one corrupt row (1e9, a hand-edited
# database) could already blow a lane's extent up and collapse ✦ Fit to a scale
# where nothing on the board is readable. This bounds both directions at once.
CANVAS_IMAGE_REACH = 100000.0


def _clamp_image_box(x, y, w, h):
    """Lane-local geometry, clamped. Returns None for anything unusable —
    a NaN stored here would make a node unreachable on every future load and
    there is no UI to fix that.

    The position is bounded on both sides of zero rather than floored at it, so
    a picture the user parked above or left of its lane survives the round trip.
    Every row written before this read back and still reads back unchanged: the
    coordinates mean exactly what they always meant (lane-local world units) and
    every one of them is inside the rail."""
    try:
        x, y, w, h = float(x), float(y), float(w), float(h)
    except (TypeError, ValueError):
        return None
    for v in (x, y, w, h):
        if v != v or abs(v) == float('inf'):
            return None
    def reach(v):
        return min(CANVAS_IMAGE_REACH, max(-CANVAS_IMAGE_REACH, v))

    return (reach(x), reach(y),
            min(CANVAS_IMAGE_MAX, max(CANVAS_IMAGE_MIN, w)),
            min(CANVAS_IMAGE_MAX, max(CANVAS_IMAGE_MIN, h)))


def _clean_group(node) -> tuple:
    """🖼🖼 One row's group membership, sanitised: (group_id, group_pos).

    The id is an opaque client key (``g<image id>``, with a suffix when that one
    is taken); it is length-capped and stripped, never parsed. An empty or
    unusable id means "in no group", and then the position is meaningless and
    goes with it — a row carrying a position and no group would be a state the
    board cannot draw."""
    gid = node.get('group_id')
    gid = str(gid).strip()[:40] if gid not in (None, '') else None
    if not gid:
        return (None, None)
    try:
        pos = int(node.get('group_pos') or 0)
    except (TypeError, ValueError):
        pos = 0
    return (gid, max(0, min(10_000, pos)))


def canvas_image_nodes(user_id, dataset_ids=None) -> dict:
    """🖼 Every image pinned on the board, grouped by dataset id — geometry AND
    the image row itself, so a lane can draw its pinned pictures without a
    second round-trip per node.

    Rows whose image no longer exists are DELETED here rather than returned.
    That is the answer to the ghost node: an image deleted from a gallery (or
    with its whole dataset) leaves a row pointing at nothing, and a node that
    renders a broken picture forever is a bug that only shows up weeks later.
    The board simply loses it, silently, which is what "the picture is gone"
    should look like.

    ``visible: false`` rows ARE returned: that is the closed-but-remembered
    state, and the panel needs it to re-open an image exactly where it was."""
    from ..models import CanvasImageNode, LoraTestImage
    owned = {d.id for d in fds.list_datasets(user_id)}
    if dataset_ids is not None:
        owned &= {int(i) for i in dataset_ids}
    if not owned:
        return {'nodes': {}, 'pruned': 0}
    rows = (CanvasImageNode.query
            .filter(CanvasImageNode.dataset_id.in_(list(owned))).all())
    if not rows:
        return {'nodes': {}, 'pruned': 0}
    imgs = {i.id: i for i in LoraTestImage.query.filter(
        LoraTestImage.id.in_([r.image_id for r in rows]),
        LoraTestImage.status == 'done',
        LoraTestImage.filename.isnot(None)).all()}
    out, pruned = {}, 0
    for r in rows:
        img = imgs.get(r.image_id)
        if img is None:
            db.session.delete(r)
            pruned += 1
            continue
        out.setdefault(str(r.dataset_id), []).append({
            'image_id': r.image_id,
            'x': float(r.x), 'y': float(r.y),
            'w': float(r.w), 'h': float(r.h),
            'visible': bool(r.visible),
            # 🖼🖼 The side-by-side strip this picture belongs to, if any. Null
            # on every row of a board that has never grouped anything — and on
            # every row of a database that predates the columns.
            'group_id': r.group_id or None,
            'group_pos': None if r.group_pos is None else int(r.group_pos),
            'image': _gallery_image(img),
        })
    if pruned:
        db.session.commit()
    for lane in out.values():
        lane.sort(key=lambda n: n['image_id'])
    return {'nodes': out, 'pruned': pruned}


def save_canvas_image_nodes(user_id, dataset_id, nodes) -> dict:
    """Upsert pinned-image geometry for one lane.

    Body rows are {image_id, x, y, w, h, visible}. Idempotent by
    (dataset_id, image_id): a drag re-sends the node on every drop, and closing
    one re-sends it with ``visible: false`` — the row and its geometry survive,
    which is the entire point (re-opening restores where and how big it was).

    An image that does not belong to this dataset is refused, so a pinned node
    can never smuggle another dataset's render into this lane."""
    from ..models import CanvasImageNode, LoraTestImage
    if not fds.get_dataset(user_id, dataset_id):
        raise LookupError('dataset not found')
    wanted = {}
    for n in (nodes or []):
        try:
            iid = int(n['image_id'])
        except (KeyError, TypeError, ValueError):
            continue
        box = _clamp_image_box(n.get('x'), n.get('y'), n.get('w'), n.get('h'))
        if box is None:
            continue
        # 🖼🖼 Group membership travels with the row. A row that does not MENTION
        # the fields keeps whatever it had — a plain drag or resize sent by an
        # older client (or by any code path that only knows about geometry) must
        # never quietly dissolve a group.
        group = _clean_group(n) if ('group_id' in n or 'group_pos' in n) else None
        wanted[iid] = (box, bool(n.get('visible', True)), group)
    if not wanted:
        return {'saved': 0,
                'total': CanvasImageNode.query.filter_by(dataset_id=dataset_id).count()}
    legit = {i.id for i in LoraTestImage.query.filter(
        LoraTestImage.id.in_(list(wanted)),
        LoraTestImage.dataset_id == dataset_id).all()}
    existing = {r.image_id: r for r in CanvasImageNode.query.filter(
        CanvasImageNode.dataset_id == dataset_id,
        CanvasImageNode.image_id.in_(list(wanted))).all()}
    saved = 0
    for iid, ((x, y, w, h), visible, group) in wanted.items():
        if iid not in legit:
            continue
        gid, gpos = group if group is not None else (None, None)
        row = existing.get(iid)
        if row is None:
            db.session.add(CanvasImageNode(
                dataset_id=dataset_id, image_id=iid, x=x, y=y, w=w, h=h,
                visible=visible, group_id=gid, group_pos=gpos))
        else:
            row.x, row.y, row.w, row.h, row.visible = x, y, w, h, visible
            if group is not None:
                row.group_id, row.group_pos = gid, gpos
        saved += 1
    db.session.commit()
    return {'saved': saved,
            'total': CanvasImageNode.query.filter_by(dataset_id=dataset_id).count()}


def clear_canvas_image_nodes(user_id, dataset_id) -> dict:
    """Forget every pinned image of one lane — geometry included.

    NOT what ✦ Tidy up calls. Tidy up hands a lane back to the automatic tree,
    and there is no automatic position for a pinned image to fall back to, so
    "tidying" one could only mean throwing it away. This exists as the deliberate
    escape hatch, and nothing invokes it by accident."""
    from ..models import CanvasImageNode
    if not fds.get_dataset(user_id, dataset_id):
        raise LookupError('dataset not found')
    removed = CanvasImageNode.query.filter_by(
        dataset_id=dataset_id).delete(synchronize_session=False)
    db.session.commit()
    return {'cleared': int(removed or 0)}


# --- 💾 ◉ LoRA Canvas: named layout presets ---------------------------------
#
# A preset is a MEMORY of the board, never a second source of truth for it. It
# is written from what the board currently holds and restored THROUGH the same
# validated writers the live board uses — so everything those refuse (a dataset
# that is not yours, an image that belongs to another lane, unusable geometry)
# is refused on restore too, and a preset carried over from a database that has
# since lost a run simply puts back what is left.

# How many presets one user may keep. Not a technical limit: a picker with
# fifty entries in it is a picker nobody reads, and this is a board with a
# handful of useful arrangements, not an archive.
CANVAS_PRESET_MAX = 24
CANVAS_PRESET_NAME_MAX = 80


def _preset_payload(positions, images) -> dict:
    """The stored shape, sanitised on the way IN as well as on the way out.

    Sanitising here is belt and braces — the restore re-validates everything
    through save_canvas_positions / save_canvas_image_nodes — but it keeps a
    single fat row from being written at all, and it means the listing can
    report honest counts without parsing arbitrary client JSON."""
    out_pos, out_img = {}, {}
    for ds_id, rows in (positions or {}).items():
        lane = []
        for p in (rows or []):
            try:
                lane.append({'record_id': int(p['record_id']),
                             'x': float(p['x']), 'y': float(p['y'])})
            except (KeyError, TypeError, ValueError):
                continue
        if lane:
            out_pos[str(int(ds_id))] = lane
    for ds_id, rows in (images or {}).items():
        lane = []
        for n in (rows or []):
            try:
                iid = int(n['image_id'])
            except (KeyError, TypeError, ValueError):
                continue
            box = _clamp_image_box(n.get('x'), n.get('y'), n.get('w'), n.get('h'))
            if box is None:
                continue
            gid, gpos = _clean_group(n)
            lane.append({'image_id': iid, 'x': box[0], 'y': box[1],
                         'w': box[2], 'h': box[3],
                         'visible': bool(n.get('visible', True)),
                         'group_id': gid, 'group_pos': gpos})
        if lane:
            out_img[str(int(ds_id))] = lane
    return {'positions': out_pos, 'images': out_img}


def _preset_row(row) -> dict:
    try:
        payload = json.loads(row.payload or '{}')
    except (TypeError, ValueError):
        payload = {}
    positions = payload.get('positions') or {}
    images = payload.get('images') or {}
    return {
        'id': row.id,
        'name': row.name,
        # The counts are what the picker shows: "3 lanes · 12 cards · 8 pictures"
        # says whether this is the arrangement you meant far better than a name
        # typed in a hurry three weeks ago.
        'lanes': len(set(positions) | set(images)),
        'cards': sum(len(v) for v in positions.values()),
        'images': sum(len(v) for v in images.values()),
        'updated_at': row.updated_at.isoformat() if row.updated_at else None,
    }


def canvas_layout_presets(user_id) -> dict:
    """Every named arrangement this user has kept, newest first."""
    from ..models import CanvasLayoutPreset
    rows = (CanvasLayoutPreset.query.filter_by(user_id=user_id)
            .order_by(CanvasLayoutPreset.updated_at.desc(),
                      CanvasLayoutPreset.id.desc()).all())
    return {'presets': [_preset_row(r) for r in rows], 'max': CANVAS_PRESET_MAX}


def save_canvas_layout_preset(user_id, name, positions=None, images=None) -> dict:
    """Keep the board as it is, under a name.

    Saving under a name that already exists OVERWRITES it, deliberately: "save"
    on a board you have just adjusted means "this is the arrangement now", and
    a second entry with the same name would leave the user to guess which of
    the two the picker will hand back."""
    from ..models import CanvasLayoutPreset
    clean = (name or '').strip()[:CANVAS_PRESET_NAME_MAX]
    if not clean:
        raise ValueError('a preset needs a name')
    payload = _preset_payload(positions, images)
    if not payload['positions'] and not payload['images']:
        raise ValueError('there is nothing arranged on the board to save')
    row = CanvasLayoutPreset.query.filter_by(user_id=user_id, name=clean).first()
    if row is None:
        if CanvasLayoutPreset.query.filter_by(user_id=user_id).count() >= CANVAS_PRESET_MAX:
            raise ValueError(
                f'{CANVAS_PRESET_MAX} layout presets is the limit — delete one first')
        row = CanvasLayoutPreset(user_id=user_id, name=clean)
        db.session.add(row)
    row.payload = json.dumps(payload)
    db.session.commit()
    return {'preset': _preset_row(row)}


def apply_canvas_layout_preset(user_id, preset_id) -> dict:
    """Put a remembered arrangement back on the board.

    Every lane goes through the LIVE writers, so the ownership checks, the
    geometry clamps and the "this image is not in that lane" refusal all apply
    exactly as they do to a drag. What is missing (a run deleted since, a
    dataset the user no longer has) is simply not put back, and the counts say
    so rather than the restore failing on the first gap."""
    from ..models import CanvasLayoutPreset
    row = CanvasLayoutPreset.query.filter_by(user_id=user_id, id=preset_id).first()
    if row is None:
        raise LookupError('preset not found')
    try:
        payload = json.loads(row.payload or '{}')
    except (TypeError, ValueError):
        payload = {}
    cards = pictures = 0
    for ds_id, rows in (payload.get('positions') or {}).items():
        try:
            cards += save_canvas_positions(user_id, int(ds_id), rows).get('saved', 0)
        except (LookupError, ValueError):
            continue          # that lane is gone; the rest of the board still lands
    for ds_id, rows in (payload.get('images') or {}).items():
        try:
            pictures += save_canvas_image_nodes(user_id, int(ds_id), rows).get('saved', 0)
        except (LookupError, ValueError):
            continue
    return {'applied': {'cards': cards, 'images': pictures}, 'preset': _preset_row(row)}


def delete_canvas_layout_preset(user_id, preset_id) -> dict:
    from ..models import CanvasLayoutPreset
    removed = CanvasLayoutPreset.query.filter_by(
        user_id=user_id, id=preset_id).delete(synchronize_session=False)
    db.session.commit()
    if not removed:
        raise LookupError('preset not found')
    return {'deleted': int(removed)}


def gpu_tiers(user_id, dataset_id, train_type=None, steps=None,
              variant=None, training_mode='lora') -> dict:
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
    mode = lt.normalize_training_mode(training_mode)
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
    # Anima is LOCAL-ONLY for this wave: a pod would need ai-toolkit with the
    # 'anima' arch (PR #860, 2026-07-15) + a recent diffusers, which current pod
    # images predate — renting one would burn a GPU on an unknown arch. Refuse
    # BEFORE any reservation. Lift once the pod image is verified.
    if fam == 'anima':
        raise ValueError('Anima cloud training is coming once the pod image is '
                         'verified — train it locally for now')
    selected_variant = str(
        variant or getattr(ds, 'train_variant', None)
        or lt._default_variant_for(fam)).strip().lower()
    # Normalized BEFORE the dense block: the token check below resolves the Krea
    # repository from this value, and a stale foreign variant must not make it
    # demand read access to the wrong one.
    if selected_variant not in lt._valid_variants_for(fam):
        selected_variant = lt._default_variant_for(fam)
    if mode == 'full_transformer':
        if fam != 'krea':
            raise ValueError('full_transformer cloud training is supported only '
                             'for Krea 2')
        if lt.slider_mode_enabled(ds):
            raise ValueError('full_transformer cloud training is incompatible '
                             'with Slider LoRA mode')
        # The token has to be able to read the base THIS recipe needs — Raw or
        # Turbo — and nothing official at all when the base is a custom
        # checkpoint pushed to the user's own private repository.
        hf_cloud_token = full_transformer_token_preflight(
            required_base_repo=lt.official_base_repo(ds, fam, selected_variant))
    else:
        hf_cloud_token = None
    n_steps = (int(steps) if steps else lt.default_steps(
        ds, train_type=fam, variant=selected_variant))
    c = cfg.get('cloud') or {}
    if mode == 'full_transformer':
        dense = c.get('full_transformer') or {}
        min_vram = max(80, int(dense.get('min_vram_gb') or 80))
    else:
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
        secure_cloud_only=bool(c.get('secure_cloud_only', False)),
        # …including the disk floor, or the picker prices tiers that the launch
        # cannot rent (a custom base can push the real ask higher still).
        min_disk_gb=_disk_gb_for(c, {'training_mode': mode})))
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
        if mode == 'full_transformer':
            # The empirical speed model is LoRA-only. Presenting its estimate
            # for a dense 26 GB transformer would be fabricated precision.
            est_min = est_cost = exceeds_cap = None
            estimate_status = 'unavailable'
        else:
            est_min = gpu_speed.estimate_minutes(name, fam, n_steps)
            # Cost bills the whole pod life: training + boot/download/quantize.
            est_cost = (round(dph * (est_min + overhead_min) / 60.0, 2)
                        if dph is not None else None)
            exceeds_cap = (est_min + overhead_min) > max_runtime
            estimate_status = 'available'
        tiers.append({
            'gpu_name': name, 'offer_id': o.get('offer_id'),
            'dph_total': round(dph, 4) if dph is not None else None,
            'gpu_ram_gb': o.get('gpu_ram_gb'),
            'speed': round(gpu_speed.speed_factor(name), 2),
            'est_minutes': (int(round(est_min)) if est_min is not None else None),
            'est_cost': est_cost, 'estimate_status': estimate_status,
            # A tier slower than the runtime cap would be KILLED mid-training
            # (checkpoint rescued, but steps lost) — warn at pick time.
            'exceeds_cap': exceeds_cap,
        })
    # slowest -> fastest (matches the launch dialog); ties broken by price.
    tiers.sort(key=lambda t: (t['speed'], t['dph_total']
                              if t['dph_total'] is not None else 9e9))
    return {'tiers': tiers, 'steps': n_steps, 'family': fam,
            'variant': selected_variant,
            'training_mode': mode,
            'hf_cloud_token': hf_cloud_token,
            'disk_gb': _disk_gb_for(c, {'training_mode': mode}),
            'max_price_per_hour': price_cap,
            'max_runtime_minutes': max_runtime}


def cloud_checkpoint_groups(dataset_id, train_type=None, variant=None,
                            dataset_table=crd.FACE) -> list:
    """Locally-synced cloud checkpoints GROUPED by their source run (newest run
    first, step-sorted within a run). Each group carries the run's identity and
    outcome — id / status / GPU / cost / timing, the SAME facts the Runs hub
    shows — so the Checkpoints panel can label WHICH run produced which epochs
    and deep-link back to its row, instead of listing several indistinguishable
    step-500→final sets. Runs whose saves were all hand-deleted are omitted.

    Scoped to one dataset TABLE: this feeds a face dataset's checkpoints panel,
    and an id-only query would list a video run's Wan saves there as if they were
    that dataset's — offered for deployment into its ComfyUI folder."""
    fam = fds.normalize_train_type(train_type) if train_type else None
    wanted_variant = str(variant).strip().lower() if variant else None
    groups = []
    for run in (CloudTrainingRun.query.filter_by(dataset_id=dataset_id)
                .order_by(CloudTrainingRun.id.desc()).all()):
        if not crd.owns(run, dataset_id, dataset_table):
            continue
        if _is_full_transformer_run(run):
            # A dense run's saves may well be on this disk now, but they are
            # 26 GB FULL models: this panel deploys LoRA adapters to ComfyUI,
            # and a full checkpoint is not one. The Runs hub is where a full
            # model is named, located and quantized.
            continue
        if fam and (_run_family(run) or fam) != fam:
            continue
        if wanted_variant and (
                str(_run_param(run, 'variant') or wanted_variant).lower()
                != wanted_variant):
            continue
        # The store first, staging only as a legacy read (see _checkpoint_dirs):
        # a run whose staging was cleaned still lists every save it produced.
        saves = run_checkpoint_files(run)
        if not saves:
            continue
        run_variant = _run_param(run, 'variant')
        entries = []
        for name in saves:
            saved_step, _stage = video_training.split_checkpoint_name(name)
            step = (saved_step if saved_step is not None
                    else int(_run_param(run, 'steps') or 0))
            entries.append({'filename': name, 'step': step, 'cloud': True,
                            'run_id': run.id, 'version': _run_param(run, 'version'),
                            'variant': run_variant,
                            'resume_state': _cloud_resume_state(),
                            'final': bool(saved_step is None
                                          and run.status == 'done'),
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


def delete_cloud_checkpoint(dataset_id, run_id, filename,
                            dataset_table=crd.FACE) -> str:
    """Move a cloud run's synced checkpoint to the trash. The run must belong
    to the dataset and be TERMINAL (deleting an active run's save is pointless
    — the sync re-downloads it). Clears checkpoint_local_path when it pointed
    at the trashed file.

    Ownership is (id, table): the id alone stopped being a complete test the
    moment two dataset tables shared one integer space, and this one authorises
    a DELETE."""
    run = CloudTrainingRun.query.get(int(run_id))
    if not run or not crd.owns(run, dataset_id, dataset_table):
        raise ValueError('unknown cloud run')
    if run.status in ACTIVE_STATES:
        raise ValueError('this cloud run is still active — its save would just '
                         'be re-synced; stop the run first')
    path = run_checkpoint_path(run, filename)
    if not path:
        raise ValueError('unknown checkpoint')
    from . import trash
    trash.send_to_trash(path, context=f'cloudckpt_run{run.id}')
    if run.checkpoint_local_path \
            and os.path.basename(run.checkpoint_local_path) == filename:
        _set(run, checkpoint_local_path=None)
    return filename


# ── Staging cleanup (global 🧹 and per-run 🧹) ────────────────────────────────
# Both entry points share ONE sparing rule and ONE trashing step, so a run that
# the global purge spares can never be trashed by the per-run button (and back).

def staging_spare_reason(run) -> str | None:
    """Why this run's staging must NOT be trashed, or None when it is fair game.
    The single source of truth for both 🧹 buttons and for the per-run button's
    disabled state — duplicating it is how the two drift apart."""
    if run.status in ACTIVE_STATES:
        return 'this run is still active — its staging is being written to'
    if run.status == 'error_pod_kept':
        # Spared only while the recovery window is genuinely OPEN. A kept pod is
        # billed for at most cloud.max_runtime_minutes past the run's end; after
        # that the pod is gone and sparing its staging forever just froze tens of
        # GB on a full disk with no upside.
        if _full_transformer_recovery_open(run):
            return ('its pod was kept for manual recovery — clean it up after '
                    'you have retrieved what you need')
        return None
    return None


# What a staging dir is allowed to contain besides the working payload. The
# cleanup NEVER trashes an entry it does not recognise: an unexpected file gets
# left behind rather than destroyed.
_PURGEABLE_STAGING_DIRS = ('dataset', 'samples')
_PURGEABLE_STAGING_SUFFIXES = ('.log', '.txt', '.json', '.yaml', '.yml', '.part')


def _purgeable_staging_entries(staging_dir) -> list:
    """Names inside a staging dir the cleanup may throw away: the exported
    dataset copy, the sample images and the mirrored logs/progress files.

    Everything else stays — and `.safetensors` can never appear here anyway,
    because the caller rescues them into the store first. Keeping BOTH guards is
    deliberate: this is the function whose past over-reach destroyed weights."""
    out = []
    try:
        names = sorted(os.listdir(staging_dir))
    except OSError:
        return out
    for name in names:
        path = os.path.join(staging_dir, name)
        if os.path.isdir(path):
            if name in _PURGEABLE_STAGING_DIRS:
                out.append(name)
            continue
        if name.lower().endswith('.safetensors'):
            continue
        if name.lower().endswith(_PURGEABLE_STAGING_SUFFIXES):
            out.append(name)
    return out


def _trash_staging(run) -> int:
    """Clean ONE run's staging: rescue its checkpoints into the durable store,
    then move the dataset copy, the samples and the logs to the trash. Returns
    the bytes moved (0 when there was nothing). Callers own the sparing check.

    It used to trash the whole directory — including `.safetensors` that had
    never been deployed anywhere else. Emptying the trash then destroyed them."""
    from . import trash
    _adopt_checkpoints_into_store(run)
    sd = run.staging_dir
    if not sd or not os.path.isdir(sd):
        return 0
    freed = 0
    for name in _purgeable_staging_entries(sd):
        path = os.path.join(sd, name)
        try:
            freed += lt._dir_size(path) if os.path.isdir(path) \
                else os.path.getsize(path)
            trash.send_to_trash(path, context=f'staging_run{run.id}')
        except OSError as e:
            logger.warning('purge: could not trash %s: %s', name, e)
    _staging_size_cache.pop(run.id, None)
    return freed


# run_id -> (expires_at, bytes). A staging dir is a dataset copy + samples +
# logs — thousands of files per run, tens of thousands across a history.
# Walking them belongs to an EXPLICIT request, never to the hub's 5 s poll, and a
# short TTL keeps a re-open (or a second tab) from re-walking the same disk.
_staging_size_cache = {}
_STAGING_SIZE_TTL = 60.0


def staging_sizes(run_ids=None) -> dict:
    """{run_id: bytes on disk} for the runs whose staging dir still exists —
    what the per-run 🧹 needs to name the weight it is about to move. Runs with
    no staging (never launched, already purged, hand-deleted) are simply absent,
    which the UI reads as "nothing to clean here". Best-effort: a directory that
    cannot be walked is skipped rather than failing the whole request."""
    now = time.time()
    q = CloudTrainingRun.query
    if run_ids is not None:
        ids = [int(i) for i in run_ids]
        if not ids:
            return {}
        q = q.filter(CloudTrainingRun.id.in_(ids))
    out = {}
    for run in q.all():
        cached = _staging_size_cache.get(run.id)
        if cached and cached[0] > now:
            if cached[1]:
                out[run.id] = cached[1]
            continue
        sd = run.staging_dir
        size = 0
        if sd and os.path.isdir(sd):
            try:
                size = lt._dir_size(sd)
            except OSError as e:
                logger.warning('staging size: could not walk %s: %s', sd, e)
                continue
        _staging_size_cache[run.id] = (now + _STAGING_SIZE_TTL, size)
        if size:
            out[run.id] = size
    return out


def purge_run_staging(run_id) -> dict:
    """Per-run 🧹: trash THIS run's dataset copy, samples and logs (its
    checkpoints are rescued into the store first — see _trash_staging). Same
    sparing rule as the global purge (staging_spare_reason), so the two can't
    disagree; the DB row stays (history). Raises ValueError on an unknown or
    spared run — the caller turns it into a 400 with the reason."""
    run = CloudTrainingRun.query.get(int(run_id))
    if not run:
        raise ValueError('unknown cloud run')
    reason = staging_spare_reason(run)
    if reason:
        raise ValueError(f'this run\'s staging is spared: {reason}')
    if not run.staging_dir or not os.path.isdir(run.staging_dir) \
            or not _purgeable_staging_entries(run.staging_dir):
        _adopt_checkpoints_into_store(run)
        return {'purged': False, 'freed_bytes': 0, 'already_clean': True}
    try:
        freed = _trash_staging(run)
    except OSError as e:
        logger.warning('purge run %s: could not trash %s: %s',
                       run.id, run.staging_dir, e)
        raise RuntimeError(f'could not move this run\'s staging to the trash: {e}')
    return {'purged': True, 'freed_bytes': freed, 'already_clean': False}


def purge_finished_runs() -> dict:
    """Hub 'Clean finished runs': for every TERMINAL run, move the dataset copy,
    the sample images and the logs to the trash. Checkpoints are NOT part of the
    deal — they are moved into the durable store first and left alone. Active
    runs, and kept pods still inside their recovery window, are spared. DB rows
    stay (history).

    `already_clean` tells "there was nothing to purge" apart from "0 purged
    because every attempt failed" — the caller shows two different messages."""
    purged = 0
    freed = 0
    candidates = 0
    for run in CloudTrainingRun.query.all():
        if staging_spare_reason(run):
            continue
        if not run.staging_dir or not os.path.isdir(run.staging_dir):
            continue
        if not _purgeable_staging_entries(run.staging_dir):
            _adopt_checkpoints_into_store(run)
            continue
        candidates += 1
        try:
            freed += _trash_staging(run)
            purged += 1
        except OSError as e:
            logger.warning('purge: could not trash %s: %s', run.staging_dir, e)
    orphans = orphan_staging_dirs()
    return {'purged_runs': purged, 'freed_bytes': freed,
            'already_clean': candidates == 0,
            'orphans': orphans,
            'orphan_bytes': sum(o['size_bytes'] for o in orphans)}


# ── Orphaned run folders ─────────────────────────────────────────────────────
# A `run_<id>` folder on disk that NO row points at — left behind by a deleted
# database, a restored backup, a relocated data dir or an interrupted purge.
# The cleanup used to answer "already clean" while 25 GB sat right there, so it
# now names them instead and offers to trash them explicitly.

def orphan_staging_dirs() -> list:
    """`[{name, size_bytes}]` of the run folders under the cloud-runs root that
    no run row claims. Sizes are walked here on purpose: this list is only built
    by an explicit cleanup request, never by the hub's poll."""
    try:
        root = _staging_root()
    except OSError:
        return []
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return []
    claimed = set()
    for run in CloudTrainingRun.query.all():
        if run.staging_dir:
            claimed.add(os.path.normcase(os.path.abspath(run.staging_dir)))
    out = []
    for name in entries:
        path = root / name
        if not name.startswith('run_') or not path.is_dir():
            continue
        if os.path.normcase(os.path.abspath(str(path))) in claimed:
            continue
        try:
            size = lt._dir_size(str(path))
        except OSError:
            continue
        out.append({'name': name, 'size_bytes': size,
                    'checkpoints': len(_loose_checkpoints(str(path)))})
    return out


def _loose_checkpoints(path) -> list:
    """`.safetensors` sitting directly in a folder — what an orphan from before
    the store may still be the only home of."""
    try:
        return [n for n in sorted(os.listdir(path))
                if n.lower().endswith('.safetensors')]
    except OSError:
        return []


def purge_orphan_staging_dirs(names=None) -> dict:
    """Trash the named orphan run folders (all of them when `names` is None).

    Guarded twice: the name must be one this scan actually reported as an
    orphan, and it is resolved under the cloud-runs root — a caller can never
    aim this at an arbitrary path.

    Any `.safetensors` still loose in an orphan is RESCUED into the checkpoint
    store before the folder goes; an orphan is exactly the case where nobody can
    tell you whether that weight exists anywhere else."""
    from . import trash
    found = {o['name']: o['size_bytes'] for o in orphan_staging_dirs()}
    wanted = list(found) if names is None else [str(n) for n in names]
    root = _staging_root()
    purged = 0
    freed = 0
    rescued = 0
    skipped = []
    for name in wanted:
        if name not in found:
            skipped.append(name)
            continue
        path = root / name
        try:
            rescued += _rescue_loose_checkpoints(str(path), name)
            trash.send_to_trash(str(path), context=f'orphan_{name}')
            purged += 1
            freed += found[name]
        except OSError as e:
            logger.warning('orphan purge: could not trash %s: %s', name, e)
            skipped.append(name)
    return {'purged_dirs': purged, 'freed_bytes': freed,
            'rescued_checkpoints': rescued, 'skipped': skipped}


def _rescue_loose_checkpoints(path, folder_name) -> int:
    """Move an orphan folder's `.safetensors` into the checkpoint store, under
    the same `run_<id>` name. Returns how many were rescued."""
    names = _loose_checkpoints(path)
    if not names:
        return 0
    dest_dir = cfg.checkpoints_root() / folder_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for name in names:
        dest = dest_dir / name
        if dest.exists():
            continue
        try:
            shutil.move(os.path.join(path, name), str(dest))
            moved += 1
        except OSError as e:
            logger.warning('orphan rescue: %s not moved: %s', name, e)
    if moved:
        logger.info('orphan %s: rescued %s checkpoint(s) into the store',
                    folder_name, moved)
    return moved


def cloud_progress(user_id, dataset_id, train_type=None, run_id=None) -> dict:
    """Same shape as lt.training_progress + cloud phase/cost fields, built
    from the staging mirror (log + samples) written by the monitor. With
    train_type, reads THAT family's newest run (several families may train
    the same dataset in parallel). With run_id, addresses THAT run — unknown
    or foreign ids raise LookupError rather than silently answering for the
    newest run."""
    run = run_for(dataset_id, run_id=run_id, train_type=train_type)
    if run_id is not None and run is None:
        raise LookupError(f'no cloud run {int(run_id)} on this dataset')
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
    # `download` arrives through _run_payload (active runs only) — the same
    # field name the local training_progress payload uses, so the component
    # that renders it does not care which lane it is looking at.
    return {'active': run.status in ACTIVE_STATES, 'log_exists': log_exists,
            **parsed, 'samples': samples, **_run_payload(run),
            'phase': run.status}
