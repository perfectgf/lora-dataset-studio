"""Starting a video training on THIS machine, through ai-toolkit's own CLI.

This is the local sibling of `cloud_video_training`, and it exists for the same
reason that one does: `lora_training.launch_training` is a `FaceDataset` function
from end to end. It exports images, checks captions, resolves a trigger word,
names the run from a base-model tag and freezes an image manifest in the
provenance registry. A `video_dataset` row has none of those columns, and its
folder is ALREADY the flat `clip_0001.mp4` + homonym `clip_0001.txt` shape
ai-toolkit reads — there is nothing to export. Threading a second entity through
that function would put a `getattr(ds, ..., None)` at each of those steps, which
is not a branch but a config assembled from defaults nobody can see.

WHAT IS SHARED, DELIBERATELY
----------------------------
Everything that must have exactly one implementation:

  * the ai-toolkit paths and the interpreter readiness probe;
  * the GPU admission sequence — vision window, ComfyUI work, the Ollama fence —
    taken in the SAME lock order (launch transaction -> queue ownership -> GPU
    arbiter), because a second, weaker guard here would simply be the one that
    loses the race;
  * the durable ownership fence (`training_in_progress` and friends), so an image
    training in flight blocks a video launch and vice versa;
  * the crash watcher and the log parser: ai-toolkit writes the same lines
    whatever it trains.

THE FENCE IS ONE INTEGER, AND TWO TABLES LIVE IN IT
---------------------------------------------------
`training_dataset_id` has always meant a `face_dataset.id`. Face dataset #3 and
video dataset #3 both exist and both resolve, so this lane stamps
`training_dataset_table` beside it and every reader of the fence asks. Without
that stamp `training_status` names a face dataset for a video run — on the wrong
page, with a Stop button that works.

WHY WEIGHTS ARE ANNOUNCED AND NOT FETCHED
-----------------------------------------
MiniMax H3 loads ~43 GB from `Comfy-Org/MiniMax-H3`. ai-toolkit downloads what is
missing, silently, as part of "starting". A button that does that is not a
training button: it is a multi-hour download with a progress bar that says
`Starting up...`, and on a disk that cannot hold it, a crash an hour in. So the
size is named first and the download is a separate, explicit yes.
"""
import json
import logging
import os
import threading
from pathlib import Path

from flask import current_app

from ..job_queue import GPU_ARBITER_LOCK, queue_manager
from ..models import VideoDataset
from . import cloud_run_dataset as crd
from . import lora_training as lt
from . import video_targets
from . import video_training

logger = logging.getLogger(__name__)

# The only extension the promoter writes, and the only one ai-toolkit's video
# loader reads. Shared verbatim with the cloud lane's clip count.
_CLIP_EXT = '.mp4'
# A stills set (frames == 1) holds images instead — same flat layout,
# same trainer, counted by the same launch guard.
_MEDIA_EXTS = ('.mp4', '.png', '.jpg', '.jpeg', '.webp')

# Weight sets this build can PROVE are present or absent, keyed by ai-toolkit
# arch. The paths are relative to ai-toolkit's models folder and are the exact
# ones its H3 loader resolves; the size is the one its own model notes state.
#
# Only architectures whose weights are single files under that folder can appear
# here. Wan's base is a diffusers repository resolved through the Hugging Face
# cache, and this build cannot say whether it is there — so Wan gets no entry and
# no refusal, rather than a guess that would block the one lane proven end to end.
WEIGHT_FOOTPRINTS = {
    'minimax_h3': {
        'repo': 'Comfy-Org/MiniMax-H3',
        'gigabytes': 43,
        'files': (
            'diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors',
            'text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',
            'vae/minimax_h3_video_vae_fp16.safetensors',
            'vae/minimax_h3_audio_vae_fp32.safetensors',
        ),
    },
}

# Written into the run folder at launch. ai-toolkit resumes from whatever LoRA it
# finds there, and the folder is named after the dataset — so a dataset
# re-promoted to a different target would resume ACROSS architectures.
_RUN_MARKER = 'lds_video_run.json'


class VideoWeightsMissing(RuntimeError):
    """The model's weights are not on this machine, and downloading them is tens
    of gigabytes. Carries the repository, the size AND the free space on the drive
    they would land on, so the caller can state all three before asking for a yes.

    The free space is what turns the question from "do you want to wait?" into
    "this cannot happen here": 43 GB needed against 26 GB free is not a download
    to confirm, it is a destination to move. `free_gigabytes` is None when the
    drive could not be measured — which must be rendered as silence, never as
    zero and never as room."""

    def __init__(self, message, repo, gigabytes, free_gigabytes=None):
        super().__init__(message)
        self.repo = repo
        self.gigabytes = gigabytes
        self.free_gigabytes = free_gigabytes


def _models_dir() -> Path:
    """ai-toolkit's `MODELS_PATH`, resolved the way ai-toolkit resolves it: the
    environment variable when set, else `<toolkit root>/models`. The child
    inherits this process's environment, so reading it here and there cannot
    disagree."""
    env = (os.environ.get('MODELS_PATH') or '').strip()
    return Path(env) if env else Path(str(lt._aitoolkit_dir())) / 'models'


def supports_training_adapter(arch) -> bool:
    """Can the INSTALLED ai-toolkit load a training adapter for this arch?

    Read off the architecture's own source file, not a version string, because
    ai-toolkit has no version to read: it is a git checkout whose capabilities
    move commit by commit, and the thing we need arrived on 2026-08-06 without a
    release to point at. `load_training_adapter` is the method that consumes
    `model.assistant_lora_path` for this architecture; without it the generic
    loader takes over, and that one refuses everything that is not Flux.

    Answers False for anything it cannot read - a toolkit path that is not set, a
    folder that is not there, a file it cannot open, an arch with no adapter to
    begin with. The asymmetry is deliberate: a false negative costs the run a
    recipe it would have liked, a false positive costs the run.
    """
    if not video_training.training_adapter_for(arch):
        return False
    try:
        source = (Path(str(lt._aitoolkit_dir())) / 'extensions_built_in'
                  / 'diffusion_models' / str(arch) / f'{arch}.py')
        return 'def load_training_adapter' in source.read_text(
            encoding='utf-8', errors='ignore')
    except (OSError, TypeError, ValueError):
        return False


def weights_report(arch) -> dict | None:
    """`{repo, gigabytes, missing, present}` for an arch whose files this build can
    probe, or None when it cannot — which is a different answer from "absent" and
    must not be rendered as one.

    Each file is looked for at its repo-relative path AND flat at the root of the
    models folder, because that is where ai-toolkit's own loader looks. Probing
    only the first would announce a 43 GB download to someone who already has the
    weights sitting in a flat models folder."""
    footprint = WEIGHT_FOOTPRINTS.get(str(arch or ''))
    if not footprint:
        return None
    root = _models_dir()
    missing = [rel for rel in footprint['files']
               if not (root / rel).is_file()
               and not (root / os.path.basename(rel)).is_file()]
    return {
        'repo': footprint['repo'],
        'gigabytes': footprint['gigabytes'],
        'missing': missing,
        'present': not missing,
    }


# Below four fifths of the size a model states for itself, the pixel count is
# under two thirds of it — the point at which a difference stops being rounding
# and starts being a different training regime. Above it, saying anything would
# be the banner people learn to skip.
_RESOLUTION_NOTE_RATIO = 0.8


def resolution_note(video_ds) -> str | None:
    """A sentence when this dataset is well below the sizes its target states for
    itself, or None when there is nothing to say.

    A NOTE and never a refusal. A low-resolution run is a legitimate choice — it
    is faster, it fits on smaller cards, and someone may want exactly that — but
    it is not the regime the model was trained in, and discovering that after a
    night of GPU is the expensive way to learn it.

    Derived from the profile's own `recommended_sizes`, which is deliberately
    EMPTY for every Wan target because no local source states one. An empty list
    therefore yields no note: a threshold computed from a number we do not have
    would be exactly the dressed-up guess that field exists to avoid."""
    profile = video_targets.get(getattr(video_ds, 'target_profile', None))
    if not profile:
        return None
    sizes = profile.get('recommended_sizes') or ()
    width = getattr(video_ds, 'width', None)
    height = getattr(video_ds, 'height', None)
    if not sizes or not width or not height:
        return None
    native = min(min(w, h) for w, h in sizes)
    short = min(int(width), int(height))
    if short >= native * _RESOLUTION_NOTE_RATIO:
        return None
    return (f'These clips are {int(width)}x{int(height)}, and {profile["label"]} '
            f'states {native} px for its own shortest edge. Training at this size '
            'works, but it is well below what the model was trained on — expect '
            'less of it than a run at the stated size.')


def local_run_name(video_ds) -> str:
    """The run folder for this dataset's LOCAL training.

    NOT `video_training.job_name_for` alone: that name is built from the dataset's
    NAME, it is stored on cloud run rows, and two promotions called "surf clips"
    share it. Locally the folder IS the resume state — ai-toolkit picks up from
    whatever checkpoint it finds there — so the second dataset would continue the
    first one's LoRA and report it as a fresh run. The row id is what makes it
    one folder per dataset."""
    return f'{video_training.job_name_for(video_ds)}_ds{int(video_ds.id)}'


def _run_root(video_ds) -> Path:
    """The run's TOP folder — ai-toolkit's `training_folder` for this dataset.

    It holds `training.log` and the run marker, and ai-toolkit creates the save
    root one level below it (`<training_folder>/<config name>`). That is the image
    lane's layout too, and the reason for it is that a checkpoint scan of the save
    root must see checkpoints and nothing else."""
    return Path(str(lt._output_dir())) / local_run_name(video_ds)


def save_root(video_ds) -> Path:
    """Where ai-toolkit writes this run's checkpoints and samples."""
    return _run_root(video_ds) / local_run_name(video_ds)


def run_log_path(video_ds) -> str:
    """Where this run's `training.log` lives. Opened before the spawn so it exists
    from the first second — a crash before ai-toolkit's first line then reads as a
    crash rather than as a missing log."""
    return str(_run_root(video_ds) / 'training.log')


def _assert_run_folder_matches(video_ds, arch):
    """Refuse to resume into a folder built for another architecture.

    Re-promoting a dataset to a different target keeps its id, so it keeps this
    folder — and ai-toolkit would load, say, a Wan LoRA into an H3 transformer.
    What that produces, if anything, is a shape error deep inside a state dict
    load. Here it is a sentence naming both targets and the folder.

    A folder with no marker (one written before this guard existed) is allowed
    through and stamped: refusing on the unknown would strand runs that are almost
    certainly fine, and the marker makes every launch after this one checkable."""
    root = _run_root(video_ds)
    marker = root / _RUN_MARKER
    profile = str(getattr(video_ds, 'target_profile', '') or '')
    if marker.is_file():
        try:
            with open(marker, encoding='utf-8') as fh:
                previous = json.load(fh)
        except (OSError, ValueError):
            previous = None
        if isinstance(previous, dict):
            was = str(previous.get('target_profile') or '')
            if was and was != profile:
                raise ValueError(
                    f'this dataset was last trained here as {was!r} and is now '
                    f'{profile!r} — ai-toolkit would resume the old LoRA into a '
                    f'different architecture. Move or delete {root} to start '
                    'this target from scratch.')
    root.mkdir(parents=True, exist_ok=True)
    with open(marker, 'w', encoding='utf-8') as fh:
        json.dump({'dataset_id': int(video_ds.id), 'target_profile': profile,
                   'arch': arch}, fh, indent=2)


def _count_clips(folder) -> int:
    try:
        return sum(1 for n in os.listdir(folder)
                   if n.lower().endswith(_MEDIA_EXTS))
    except OSError:
        return 0


def _default_spawn(argv, cwd, env, stdout):
    import subprocess
    return subprocess.Popen(
        argv, cwd=cwd, env=env, shell=False, stdout=stdout,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))


@lt._serial_local_launch
def start_video_training(user_id, video_dataset_id, steps=1000, base_model=None,
                         low_vram=True, rank=16, sample_prompts=None,
                         accept_download=False, do_i2v=False, _spawn=None) -> dict:
    """Train a LoRA on a promoted video dataset, locally, through ai-toolkit.

    Everything that can refuse does so BEFORE the GPU fence is taken, in
    increasing order of what it costs to discover late: an unknown dataset, an
    empty folder, an uncatalogued target, a run folder belonging to another
    architecture, absent weights. Only then is the card claimed.

    `low_vram` defaults True for the measured reason the builder documents: a
    24 GB card cannot do without it on any of these models. `accept_download` is
    the explicit yes for a weight set this machine does not have; it is a
    parameter and not a setting, because accepting 43 GB once for MiniMax H3 is
    not accepting it forever, nor for the next architecture.

    Returns `{started, pid, config_path, log_path, run_name, steps, clips,
    downloading, run_token}`. `downloading` is True/False when this build can
    probe the weights and None when it cannot — a caller must not render the
    third case as "present".
    """
    if not lt.is_installed():
        raise RuntimeError('ai-toolkit is not configured')
    lt.assert_interpreter_ready()

    ds = VideoDataset.query.filter_by(id=int(video_dataset_id),
                                      user_id=user_id).first()
    if ds is None:
        raise ValueError('video dataset not found')

    folder = str(ds.output_dir or '')
    clips = _count_clips(folder)
    if not clips:
        raise ValueError(
            'this video dataset has no clips or stills on disk — there would '
            'be nothing to train on. Promote it again from the bank.')

    n_steps = max(100, int(steps or 1000))
    training_folder = str(_run_root(ds))
    # Built BEFORE anything is reserved: an uncatalogued target, an illegal frame
    # count or a missing base repo must surface here and not from a child process
    # that has already taken the card.
    # The recipe is offered to the toolkit that will actually run this, which
    # here is the one installed on this machine — and its capabilities are read
    # from its own source, not assumed. `video_targets` owns the arch name.
    target_arch = (video_targets.get(
        getattr(ds, 'target_profile', None)) or {}).get('aitk_arch')
    job_config = video_training.build_job_config(
        ds, folder, n_steps, training_folder=training_folder,
        base_model=base_model, low_vram=bool(low_vram), rank=rank,
        sample_prompts=sample_prompts,
        training_adapter=supports_training_adapter(target_arch),
        do_i2v=bool(do_i2v))
    proc_cfg = job_config['config']['process'][0]
    arch = proc_cfg['model']['arch']
    run_name = local_run_name(ds)
    job_config['config']['name'] = run_name

    weights = weights_report(arch)
    if weights and not weights['present'] and not accept_download:
        models = _models_dir()
        free = lt.free_disk_gb(models)
        room = (f' That drive has {free:.1f} GB free.' if free is not None else '')
        raise VideoWeightsMissing(
            f'{video_targets.get(ds.target_profile)["label"]} needs about '
            f'{weights["gigabytes"]} GB of weights that are not on this machine. '
            f'They would be downloaded from {weights["repo"]} into '
            f'{models}.{room} That folder follows the ai-toolkit folder set in '
            'Settings, so a drive with more room can host it. Confirm to '
            'download them.',
            weights['repo'], weights['gigabytes'], free)

    lt.assert_free_disk(lt._output_dir(), lt.MIN_FREE_GB_TRAIN, 'a training run')
    _assert_run_folder_matches(ds, arch)
    # Cheap refusal before the config is written; the authoritative copy of this
    # check runs under the GPU arbiter below.
    lt._assert_no_vision_pass_on_gpu()

    config_path = str(Path(str(lt._jobs_dir())) / f'{run_name}.json')
    with open(config_path, 'w', encoding='utf-8') as fh:
        json.dump(job_config, fh, indent=2)

    log_path = run_log_path(ds)
    env = lt.training_subprocess_env()
    spawn = _spawn or _default_spawn
    run_token = lt.secrets.token_urlsafe(16)
    dataset_id = int(ds.id)
    app = current_app._get_current_object()

    # The image lane's lock order, kept exactly: queue ownership first, GPU
    # admission second. The Ollama handoff stays INSIDE both, or a vision pass
    # could claim the card in the interval before the spawn.
    with lt._queue_lock, GPU_ARBITER_LOCK:
        if (queue_manager._get_system_state('training_in_progress', False)
                and not lt._training_process_is_definitely_dead(
                    queue_manager._get_system_state('training_pid', None))):
            raise ValueError(
                'a training is already in progress - wait for it to finish '
                'before starting this one')
        lt._assert_no_vision_pass_on_gpu()
        if queue_manager.has_comfyui_work():
            from ..gpu_window import GpuBusyError
            raise GpuBusyError(
                'ComfyUI has queued or active work, so local training cannot '
                'take the GPU. Wait for it to finish or cancel it safely first.')
        try:
            from .ollama_gpu_fence import ensure_released_for_comfy
            released = ensure_released_for_comfy()
        except Exception as exc:
            from ..gpu_window import GpuBusyError
            logger.exception('could not verify Ollama GPU release before video training')
            raise GpuBusyError(
                'Could not verify that Ollama released the GPU before local '
                'training. Check Ollama, then try again.') from exc
        if not released:
            from ..gpu_window import GpuBusyError
            raise GpuBusyError(
                'Ollama still owns the GPU, so local training cannot start '
                'safely. Wait for the vision task to finish or unload it.')

        queue_manager._set_system_state('training_error', None, ttl_seconds=1)
        identity = {
            'training_in_progress': True,
            'training_dataset_id': dataset_id,
            # The stamp that keeps this run out of the face lane's readers.
            'training_dataset_table': crd.VIDEO,
            'training_target_step': n_steps,
            'training_run_token': run_token,
            'training_train_type': 'video',
        }
        logf = None
        proc = None
        try:
            for key, value in identity.items():
                queue_manager._set_system_state(
                    key, value, ttl_seconds=lt._TRAIN_STATE_TTL)
            logf = open(log_path, 'w', encoding='utf-8')
            proc = spawn([str(lt._venv_python()), 'run.py', config_path],
                         str(lt._aitoolkit_dir()), env, logf)
        except Exception as exc:
            if logf is not None:
                try:
                    logf.close()
                except OSError:
                    pass
            try:
                lt._clear_training_identity(ttl_seconds=None)
            except Exception:
                logger.exception(
                    'could not clear the partial pre-spawn video training fence')
            if isinstance(exc, (FileNotFoundError, OSError)):
                raise ValueError(f'could not start training: {exc}') from exc
            raise
        # Past the spawn nothing may escape: the fence is now the only thing that
        # keeps another GPU owner off the card, and it must stay fail-closed even
        # if the richer PID identity cannot be persisted.
        try:
            lt._record_training_process_identity(proc.pid)
        except Exception:
            logger.exception(
                'could not persist the spawned video training identity; '
                'keeping the GPU fence fail-closed')

    threading.Thread(
        target=lt._watch_training,
        args=(app, proc, log_path, dataset_id),
        daemon=True).start()
    logger.info('local video run %s started: %s clips, %s steps, profile %s',
                run_name, clips, n_steps, ds.target_profile)
    return {
        'started': True,
        'pid': proc.pid,
        'config_path': config_path,
        'log_path': log_path,
        'run_name': run_name,
        'steps': n_steps,
        'clips': clips,
        'run_token': run_token,
        'downloading': None if weights is None else not weights['present'],
        # Things the run will not fail on but the user should hear once. Not
        # refusals: every one of these describes a legitimate choice that is
        # simply not the model's own regime.
        'warnings': [n for n in (resolution_note(ds),) if n],
    }


def video_training_progress(video_dataset_id, user_id=None) -> dict:
    """Live view of a local video run: the image lane's log parser, pointed at
    this dataset's own log.

    `active` asks the table as well as the id. Without that, a face training of
    the colliding id would drive this dataset's progress bar — the same steps, the
    same loss, for a run the user is not watching."""
    query = VideoDataset.query.filter_by(id=int(video_dataset_id))
    if user_id is not None:
        query = query.filter_by(user_id=user_id)
    ds = query.first()
    if ds is None:
        raise ValueError('video dataset not found')
    cur_id = queue_manager._get_system_state('training_dataset_id', None)
    cur_table = queue_manager._get_system_state('training_dataset_table', None)
    active = (bool(queue_manager._get_system_state('training_in_progress', False))
              and cur_table == crd.VIDEO
              and cur_id is not None and int(cur_id) == int(ds.id)
              and not lt._training_process_is_definitely_dead(
                  queue_manager._get_system_state('training_pid', None)))
    parsed = {'step': None, 'total': None, 'loss': None, 'speed': None,
              'eta': None, 'loss_curve': []}
    download = None
    log_path = run_log_path(ds)
    log_exists = os.path.isfile(log_path)
    if log_exists:
        try:
            size = os.path.getsize(log_path)
            with open(log_path, encoding='utf-8', errors='replace') as fh:
                if size > lt._PROG_LOG_MAX_BYTES:
                    fh.seek(size - lt._PROG_LOG_MAX_BYTES)
                text = fh.read()
            parsed = lt._parse_training_log(text)
            download = lt.parse_download_progress(text)
        except OSError:
            log_exists = False
    return {'active': active, 'log_exists': log_exists,
            'run_name': local_run_name(ds), 'download': download,
            # Carried here and not only on the launch result: a warning that
            # arrives after the click is a warning about a decision already
            # taken. The card polls this on mount.
            'resolution_note': resolution_note(ds), **parsed}


def list_run_checkpoints(video_dataset_id, user_id=None) -> list:
    """The saves this dataset's local run has written, newest step first.

    Uses `video_training.split_checkpoint_name` rather than the image lane's
    regex: a Wan 2.2 checkpoint is a PAIR (`_high_noise` / `_low_noise`), and the
    old anchor `_<digits>.safetensors$` reads no step at all from those names."""
    query = VideoDataset.query.filter_by(id=int(video_dataset_id))
    if user_id is not None:
        query = query.filter_by(user_id=user_id)
    ds = query.first()
    if ds is None:
        raise ValueError('video dataset not found')
    root = save_root(ds)
    out = []
    try:
        names = os.listdir(root)
    except OSError:
        return out
    for name in names:
        if not name.endswith('.safetensors'):
            continue
        step, stage = video_training.split_checkpoint_name(name)
        out.append({'filename': name, 'step': step, 'stage': stage,
                    'final': step is None,
                    'path': str(root / name)})
    out.sort(key=lambda c: (c['step'] is None, -(c['step'] or 0), c['filename']))
    return out
