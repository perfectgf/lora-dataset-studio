"""What a dense (full-model) run actually LEFT on this computer, and what to do with it.

WHY THIS EXISTS
---------------
A LoRA lives in three places in this app: the Checkpoints panel lists it and
deploys it in one click, the Test Studio generates with it, the Canvas draws it
with its lineage. A full model — the thing a user pays eight hours of GPU for —
existed as a banner, a Hugging Face link, and a quantize block. To try one, you
had to open ComfyUI by hand and find the file yourself.

The Checkpoints panel could not be the answer as it stood, and the guard that
says so (``cloud_training.cloud_checkpoint_groups``) is right: that lane deploys
LoRA **adapters** into ``loras/<family>``, and a 26 GB transformer is not an
adapter. Widening it would have given a full model the adapter's verbs. So this
module is a SECOND lane in the same panel, with the verbs a full model actually
has.

THE DISTINCTION THIS MODULE EXISTS TO MAKE VISIBLE
--------------------------------------------------
A delivered dense run leaves up to two files, and they are not interchangeable:

* the **master** (bf16, ~26 GB) — the only one that can be trained again or
  resumed from. It is never deployed to ComfyUI: it would occupy 26 GB of a
  model folder to do a job the twin does in half the space.
* the **fp8 twin** (~13 GB) — the inference format. This is the one ComfyUI
  loads, and the only one worth putting in ``diffusion_models``.

Every field below serves that sentence. A panel that listed "2 files" and
offered "deploy" on both would be worse than the banner it replaces.

DISK IS THE TRUTH, PARAMS ARE THE STORY
---------------------------------------
What exists is read from the run's checkpoint store, never from the stamped
``local_weight_filename`` — a user who deletes a 26 GB file in Explorer must see
it disappear here, exactly as ``_run_payload`` already re-checks ``isfile``
before offering a download. The stamped params still answer the questions the
disk cannot: which Hugging Face repository holds the backup, what the pod ran,
what sampler settings the model wants.

WHICH FILE IS THE MASTER is ``dense_weights.pick_master`` — the one rule, shared
with the delivery verifier and the quantizer, so this panel can never name a
different file from the button underneath it.
"""
from __future__ import annotations
from ..extensions import db

import logging
import os
import shutil
import threading
import time

logger = logging.getLogger(__name__)

# Background-copy state, same shape and the same home as the fp8 delivery job's
# (queue_manager system state) so one status poll pattern covers both.
_STATE_KEY = 'dense_comfy_send'
_STATE_TTL = 6 * 3600
_lock = threading.Lock()

# What a "send to ComfyUI" really did. A hard link costs zero bytes and is the
# normal outcome (the checkpoint store and ComfyUI's models folder are very often
# the same volume); a copy is the cross-volume fallback. Telling them apart is
# not trivia: after a link, deleting one name leaves the other file intact, and
# users ask.
LINKED = 'linked'
COPIED = 'copied'


class DenseArtifactError(ValueError):
    """A refusal with a sentence for the user. Never a stack trace."""


# --- reading what is there --------------------------------------------------------

def _fmt_iso(value):
    return value.isoformat() if value else None


def _artifact_files(run) -> tuple[list, list]:
    """``(masters, twins)`` as ``(filename, abs_path, size)`` for ONE run.

    Reads the durable checkpoint store (plus the legacy staging dir) through the
    shared lister, so a run that trained before the store existed still shows its
    files, and a file deleted by hand disappears here on the next poll.
    """
    from . import cloud_training as ct
    from . import dense_local_delivery as dld

    masters, twins = [], []
    for name, path in (ct.run_checkpoint_files(run) or {}).items():
        try:
            size = os.path.getsize(path)
        except OSError:
            continue                                # vanished between list and stat
        (twins if dld.is_fp8_name(name) else masters).append((name, path, size))
    return masters, twins


def _master_of(masters) -> dict | None:
    """THE master among a run's full-precision saves, and what it was picked over."""
    from . import dense_weights

    if not masters:
        return None
    sizes = {name: size for name, _p, size in masters}
    paths = {name: path for name, path, _s in masters}
    choice = dense_weights.describe_choice([(n, s) for n, _p, s in masters])
    name = choice['name']
    if not name:
        return None
    return {
        'filename': name,
        'path': paths[name],
        'folder': os.path.dirname(paths[name]),
        'size_bytes': sizes.get(name, 0),
        'step': choice['step'],
        'is_final': choice['is_final'],
        # A run that saved every 250 steps leaves a dozen 26 GB files. Naming the
        # pick AND the count is what stops the card from silently disagreeing with
        # the quantize button underneath it.
        'total_candidates': choice['total'],
        'others': choice['others'],
        'others_bytes': sum(s for n, _p, s in masters if n != name),
    }


def comfy_index(folder_type) -> dict:
    """``{lower basename: (loader-relative name, absolute path)}`` of everything
    ComfyUI can load from one folder type.

    Built ONCE per listing and handed to every run. The obvious spelling —
    ``resolve_model_file`` per twin — walks every search root recursively on a
    MISS, and a miss is the normal state here (the whole point of the panel is
    the twin that is not in ComfyUI yet). A dataset with five full-model runs
    would therefore have walked those roots five times on every poll of a panel
    that refreshes itself. One walk, one dict, same answer.
    """
    from . import comfy_model_paths

    out = {}
    try:
        for rel, ab in comfy_model_paths.list_models(folder_type):
            out.setdefault(os.path.basename(rel).lower(), (rel, ab))
    except Exception:                               # noqa: BLE001 — never fatal
        logger.debug('ComfyUI model listing failed', exc_info=True)
    return out


def _twin_of(twins, family, index=None) -> dict | None:
    """The run's fp8 twin (newest wins), and the TWO facts about where it is.

    They are two facts and not one, and conflating them breaks one of the two
    things that read them:

    * ``in_comfyui`` — ComfyUI's own resolver can find it. This is what makes
      the Test Studio able to load it, so it alone may light the "test this"
      link. A name ComfyUI cannot resolve is a link to a screen where the model
      is absent.
    * ``delivered`` — it already sits where "Send to ComfyUI" would put it. When
      ComfyUI is not configured at all, that destination is the app's OWN models
      folder (``fp8_local_delivery.destination_folder`` says so out loud), which
      no ComfyUI scanner will ever list. Reading the button off ``in_comfyui``
      there would leave it offered for ever, and clicking it would answer "that
      file is already there" — a loop the panel would never leave.
    """
    from . import fp8_local_delivery

    if not twins:
        return None
    name, path, size = sorted(twins, key=lambda t: t[0])[-1]
    folder_type = fp8_local_delivery.folder_type_for(family)
    if index is None:
        index = comfy_index(folder_type)
    hit = index.get(os.path.basename(name).lower())
    deployed = hit[1] if hit else None
    # A file that resolves back to the store itself is NOT "in ComfyUI": that
    # happens when the checkpoint store sits inside a declared model root, and
    # reporting it as deployed would hide the one action that matters.
    if deployed and _same_file(deployed, path):
        deployed = None
    destination = os.path.join(fp8_local_delivery.destination_folder(family)['path'],
                               os.path.basename(name))
    delivered = bool(deployed) or (os.path.isfile(destination)
                                   and not _same_file(destination, path))
    return {
        'filename': name,
        'path': path,
        'folder': os.path.dirname(path),
        'size_bytes': size,
        'in_comfyui': bool(deployed),
        'delivered': delivered,
        'comfyui_path': deployed,
        # The loader-relative name the Test Studio's base picker publishes, so the
        # card can deep-link straight at it. None until it is really in a root.
        'comfyui_name': hit[0] if deployed else None,
        'folder_type': folder_type,
    }


def _same_file(a, b) -> bool:
    """Do these two paths name the same file on disk (junctions resolved)?"""
    try:
        return os.path.normcase(os.path.realpath(a)) == \
            os.path.normcase(os.path.realpath(b))
    except Exception:                               # noqa: BLE001
        return False


def _hub_of(run) -> dict | None:
    """The Hugging Face copy of this run AS RECORDED, or None when it never had one.

    Every field here is a fact about the PAST — ``status`` is stamped once, at
    delivery, and nothing rewrites it. Whether the repository is still there is
    a different question, answered live by ``hub_presence`` and never from these
    values. ``checked_at`` is what makes the difference sayable on screen: the
    panel can date the record instead of rendering it in the present tense.
    """
    from . import cloud_training as ct

    repo = ct._run_param(run, 'hf_repo_id')
    if not repo:
        return None
    return {
        'repo_id': repo,
        'url': ct._run_param(run, 'hf_url'),
        'weight_filename': ct._run_param(run, 'hf_weight_filename'),
        'status': ct._run_param(run, 'artifact_status'),
        'checked_at': (ct._run_param(run, 'delivery_last_checked_at')
                       or ct._run_param(run, 'artifact_verified_at')
                       or ct._run_param(run, 'verified_at')),
        'backup_status': ct._run_param(run, 'hub_backup_status'),
        'backup_detail': ct._run_param(run, 'hub_backup_detail'),
    }


def describe_run(run, index=None) -> dict:
    """One dense run as the Checkpoints panel needs it. Never raises on a run
    whose folders are gone — that run simply reports no local artifact.

    ``index`` is ``comfy_index(folder_type)``, shared across a listing so the
    ComfyUI folders are walked once rather than once per run."""
    from . import cloud_training as ct
    from . import fp8_local_delivery, lora_training as lt

    family = ct._run_family(run) or 'krea'
    masters, twins = _artifact_files(run)
    master = _master_of(masters)
    fp8 = _twin_of(twins, family, index)
    record = ct._cloud_run_record(run)
    active = run.status in ct.ACTIVE_STATES
    return {
        'run_id': run.id,
        'record_id': getattr(record, 'id', None),
        'dataset_id': run.dataset_id,
        'status': run.status,
        'active': active,
        'train_type': family,
        'variant': ct._run_param(run, 'variant'),
        'version': ct._run_param(run, 'version'),
        'steps': ct._run_param(run, 'steps'),
        'created_at': _fmt_iso(run.created_at),
        'finished_at': _fmt_iso(run.finished_at),
        # local | hub | both — frozen at launch; an unstamped row predates the
        # local delivery and means Hugging Face only.
        'delivery': ct._dense_delivery(run),
        # What actually ran the training. Stamped since the host blacklist landed
        # and shown nowhere until now; it is the first thing anyone needs when a
        # model behaves unlike its siblings.
        'trainer': ct._run_param(run, 'pod_image'),
        # The sample settings this model wants — and a sentence that follows the
        # base the run ACTUALLY used, so a Turbo-based artifact is not described
        # as an undistilled one.
        'inference_hint': lt.dense_inference_hint(
            ct._RunConfigDataset(None, 'krea', ct._run_param(run, 'variant'),
                                 ct._run_param(run, 'base_model') or '')),
        'master': master,
        'fp8': fp8,
        'hub': _hub_of(run),
        'comfyui': fp8_local_delivery.destination_folder(family),
        # A master with no twin is the whole reason the quantize block exists;
        # offering it while the run is still writing files is not.
        'can_quantize': bool(master and not fp8 and not active),
        # Only ever the twin. The master is never sent to ComfyUI — that is the
        # distinction this lane exists to make. Read off `delivered`, not
        # `in_comfyui`: on an install with no ComfyUI configured the file lands
        # in the app's own models folder, which no ComfyUI scan will ever list.
        'can_send_to_comfyui': bool(fp8 and not fp8['delivered'] and not active),
        'can_delete': bool((master or fp8) and not active),
    }


def list_dense_models(dataset_id, train_type=None) -> list:
    """Every full model this dataset produced, newest run first.

    Runs that delivered nothing locally are STILL listed when they have a Hugging
    Face copy: "trained, backed up, not on this computer" is a state the panel
    must be able to show — it is the only state every run from before the local
    delivery is in, and hiding those runs is what made the dense lane invisible
    in the first place. A run that has neither is omitted (nothing to say).
    """
    from ..models import CloudTrainingRun
    from . import cloud_training as ct
    from . import face_dataset_service as fds
    from . import fp8_local_delivery

    fam = fds.normalize_train_type(train_type) if train_type else None
    index, indexed_type = None, None
    out = []
    for run in (CloudTrainingRun.query.filter_by(dataset_id=int(dataset_id))
                .order_by(CloudTrainingRun.id.desc()).all()):
        if not ct._is_full_transformer_run(run):
            continue
        if fam and (ct._run_family(run) or fam) != fam:
            continue
        # Built lazily and only once — a dataset with no dense run must not pay
        # for a model-folder walk, and five dense runs must not pay five times.
        folder_type = fp8_local_delivery.folder_type_for(ct._run_family(run) or 'krea')
        if index is None or folder_type != indexed_type:
            index, indexed_type = comfy_index(folder_type), folder_type
        try:
            entry = describe_run(run, index)
        except Exception:                           # noqa: BLE001 — one bad row never 500s the panel
            logger.warning('dense artifact listing skipped run %s', run.id, exc_info=True)
            continue
        if not (entry['master'] or entry['fp8'] or entry['hub']):
            continue
        out.append(entry)
    return out


def _run_of(dataset_id, run_id):
    from ..models import CloudTrainingRun
    from . import cloud_training as ct

    run = db.session.get(CloudTrainingRun, int(run_id))
    if not run or run.dataset_id != int(dataset_id):
        raise DenseArtifactError('unknown cloud run')
    if not ct._is_full_transformer_run(run):
        raise DenseArtifactError('this run did not train a full model')
    return run


# --- sending the twin to ComfyUI --------------------------------------------------

def status() -> dict:
    from ..job_queue import queue_manager
    return queue_manager._get_system_state(_STATE_KEY, {}) or {}


def _set(state, info, **extra):
    from ..job_queue import queue_manager
    queue_manager._set_system_state(_STATE_KEY, {
        'status': state, 'run_id': info['run_id'], 'filename': info['filename'],
        'destination_dir': info['destination_dir'],
        'destination_dir_kind': info['destination_dir_kind'],
        'total_bytes': info['total_bytes'], **extra,
    }, ttl_seconds=_STATE_TTL)


def send_plan(dataset_id, run_id) -> dict:
    """What "Send to ComfyUI" will do, or the refusal — always renderable."""
    try:
        return {'ok': True, **_send_info(_run_of(dataset_id, run_id))}
    except DenseArtifactError as e:
        return {'ok': False, 'error': str(e)}
    except Exception as e:                          # noqa: BLE001 — reported, not raised
        logger.warning('dense send plan unavailable: %s', e)
        return {'ok': False, 'error': f'could not read the model: {e}'[:300]}


def _send_info(run) -> dict:
    entry = describe_run(run)
    fp8 = entry['fp8']
    if not fp8:
        raise DenseArtifactError(
            'this run has no fp8 file on this computer yet. Quantize the full '
            'model first — the fp8 twin is what ComfyUI loads.')
    if fp8['delivered']:
        raise DenseArtifactError(
            f"{fp8['filename']} is already where this app puts it for ComfyUI")
    if entry['active']:
        raise DenseArtifactError('this run is still working — wait for it to finish')
    dest = entry['comfyui']
    target = os.path.join(dest['path'], fp8['filename'])
    if os.path.exists(target):
        raise DenseArtifactError(
            f"{fp8['filename']} is already in {dest['path']} — delete or rename it "
            'first. This never overwrites a file you already have.')
    # A hard link is the normal outcome and costs nothing; only a genuinely
    # different volume falls back to a copy, and THAT is what needs a disk check.
    same_volume = _same_volume(fp8['path'], dest['path'])
    free = _free_bytes(dest['path'])
    needed = 0 if same_volume else fp8['size_bytes']
    return {
        'run_id': run.id, 'filename': fp8['filename'], 'source': fp8['path'],
        'destination': target, 'destination_dir': dest['path'],
        'destination_dir_kind': dest['kind'], 'destination_dir_note': dest['note'],
        'total_bytes': fp8['size_bytes'],
        'method': LINKED if same_volume else COPIED,
        'required_bytes': needed, 'free_bytes': free,
        'enough_space': free is None or free >= needed,
    }


def _same_volume(a, b) -> bool:
    """Whether a hard link between these two paths can even be attempted."""
    try:
        return os.path.splitdrive(os.path.realpath(a))[0].lower() == \
            os.path.splitdrive(os.path.realpath(b))[0].lower()
    except Exception:                               # noqa: BLE001
        return False


def _free_bytes(path):
    """Free space on the volume that REALLY holds this path (junctions resolved)."""
    probe = os.path.realpath(path)
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return None
        probe = parent
    try:
        return shutil.disk_usage(probe).free
    except Exception:                               # noqa: BLE001
        return None


def send_to_comfyui(app, dataset_id, run_id) -> dict:
    """Put the fp8 twin where ComfyUI loads it. Link when possible, else copy.

    Returns as soon as the answer is known: a hard link is instantaneous and
    reports ``done`` on the spot, a cross-volume copy runs in a thread and
    reports bytes. Refuses on the click exactly as ``send_plan`` said it would.
    """
    info = _send_info(_run_of(dataset_id, run_id))
    if not info['enough_space']:
        raise DenseArtifactError(
            f"not enough disk space in {info['destination_dir']}: "
            f"{(info['free_bytes'] or 0) / 1000 ** 3:.1f} GB free and this copy "
            f"needs {info['required_bytes'] / 1000 ** 3:.1f} GB. The file is on a "
            'different drive than ComfyUI, so it cannot be linked for free.')
    with _lock:
        if status().get('status') == 'sending':
            raise DenseArtifactError('a model is already being sent — wait for it')
        _set('sending', info, done_bytes=0, method=info['method'],
             started_at=time.time())
    os.makedirs(info['destination_dir'], exist_ok=True)
    if info['method'] == LINKED:
        try:
            os.link(info['source'], info['destination'])
            _set('done', info, done_bytes=info['total_bytes'], method=LINKED)
            logger.info('dense fp8 linked into ComfyUI: %s', info['filename'])
            return {**info, 'status': 'done'}
        except OSError as e:
            # Same drive is not the same as linkable (a junction onto another
            # volume, a filesystem that refuses links). Fall through to the copy
            # rather than fail an operation that plainly can succeed.
            logger.info('hard link refused (%s) — copying instead', e)
            info = {**info, 'method': COPIED}
            _set('sending', info, done_bytes=0, method=COPIED)

    def _run():
        with app.app_context():
            _copy(info)

    threading.Thread(target=_run, daemon=True, name='dense-comfy-send').start()
    return {**info, 'status': 'sending'}


def _copy(info):
    tmp = f"{info['destination']}.part"
    try:
        done = 0
        last = 0.0
        with open(info['source'], 'rb') as src, open(tmp, 'wb') as dst:
            while True:
                chunk = src.read(8 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
                done += len(chunk)
                now = time.monotonic()
                if now - last >= 1.0:               # a 13 GB copy, once a second
                    last = now
                    _set('sending', info, done_bytes=done, method=COPIED)
        os.replace(tmp, info['destination'])
        _set('done', info, done_bytes=done, method=COPIED)
        logger.info('dense fp8 copied into ComfyUI: %s', info['filename'])
    except Exception as e:                          # noqa: BLE001 — reported
        try:
            os.remove(tmp)
        except OSError:
            pass
        _set('error', info, done_bytes=0, error=str(e)[:300])
        logger.warning('dense fp8 send failed (%s): %s', info['filename'], e)
    finally:
        _invalidate_base_lists()


def _invalidate_base_lists():
    """Drop the 5-minute model-list caches so the new base shows up at once.

    Without this the file is there and every picker still says it is not, for up
    to five minutes — which reads exactly like the feature not working."""
    try:
        from ..utils.comfyui import clear_model_caches
        clear_model_caches()
    except Exception:                               # noqa: BLE001 — cosmetic
        logger.debug('model caches not cleared', exc_info=True)


# --- deleting -----------------------------------------------------------------------

def delete_artifact(dataset_id, run_id, filename) -> dict:
    """Move ONE of this run's dense files to the app trash.

    Whitelisted against what this run really holds, basename-only, and refused
    while the run is active (the monitor is still writing there). Recoverable on
    purpose: these files cost hours of GPU, and the trash is the difference
    between a mistake and a loss.
    """
    from . import cloud_training as ct
    from . import trash

    run = _run_of(dataset_id, run_id)
    if run.status in ct.ACTIVE_STATES:
        raise DenseArtifactError(
            'this run is still working — its files cannot be deleted yet')
    name = os.path.basename(str(filename or ''))
    path = (ct.run_checkpoint_files(run) or {}).get(name)
    if not path:
        raise DenseArtifactError('unknown file for this run')
    try:
        dest = trash.send_to_trash(path, context=f'dense-run{run.id}')
    except trash.TrashLockError as e:
        raise DenseArtifactError(
            f'{name} is open in another program (ComfyUI still has it loaded?) — '
            f'close it and try again') from e
    except FileNotFoundError as e:
        raise DenseArtifactError('that file is already gone') from e
    _invalidate_base_lists()
    return {'filename': name, 'trashed_to': dest}
