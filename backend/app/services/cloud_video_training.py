"""Launching a cloud training run on a VIDEO dataset.

Its own module rather than a branch of `launch_cloud_training`, for the same
reason `video_training` is not a branch of `lora_training`: that function is
~250 lines of FACE preflight — image counts, caption quality, rembg masking,
custom VAE/TE overrides, slider prompt pairs, HF custom-base pushes, the
provenance registry — and a video dataset has none of those columns. Threading a
`VideoDataset` through it would mean a flag on every one of those blocks, in a
file two other sessions are editing.

What it DOES reuse is everything after the decision: the reservation lock, the
run row, the provisioning, the monitor thread, the stop path, the harvest. A
video run is an ordinary `cloud_training_run` row carrying
`dataset_table='video_dataset'`; from the monitor's point of view the only
differences are which folder gets uploaded and which builder writes the config,
and both of those are single seams in cloud_training (`_staging_dataset_dir`,
`_build_pod_job_config`).

EVERY REFUSAL HAPPENS BEFORE THE RESERVATION
--------------------------------------------
The point of raising in `video_training.build_job_config` was to fail before a
GPU is rented. That only holds if the config is built BEFORE the run row exists —
otherwise the refusal arrives from the monitor thread, minutes later, with a pod
already on the clock and a `preparing` row wedging the single-run guard. So this
function builds the config first, purely to see it raise, and throws the result
away; the monitor rebuilds it at pod boot from the stamped params, exactly like
the face lane (a rebuild is what keeps a launch from being retargeted mid-flight).
"""
import json
import logging
import os

from ..extensions import db
from ..models import CloudTrainingRun, VideoDataset
from . import cloud_run_dataset as crd
from . import cloud_training as ct
from . import video_training

logger = logging.getLogger(__name__)

# What the pod needs to see in the folder before renting anything. `.mp4` is the
# only extension the exporter writes and the only video extension the upload
# ships; a folder without one uploads captions alone and trains on nothing.
_CLIP_EXT = '.mp4'
# A stills set (frames == 1) holds images instead — same flat layout,
# same trainer, counted by the same launch guard.
_MEDIA_EXTS = ('.mp4', '.png', '.jpg', '.jpeg', '.webp')


def _count_clips(folder) -> int:
    try:
        return sum(1 for f in os.listdir(folder)
                   if f.lower().endswith(_MEDIA_EXTS))
    except OSError:
        return 0


def _start_pod(run):
    """Hand the run to the shared monitor thread, which rents the pod, uploads
    the folder, builds the job and harvests the saves. A named seam rather than
    an inline pair of calls so a route test can neutralise exactly this step —
    everything before it (the refusals, the reservation, the stamp) then runs for
    real instead of being mocked away with it."""
    ct._stop_event_for(run.id).clear()
    ct._start_monitor(run.id)


def video_gpu_tiers(user_id, video_dataset_id, steps=None) -> dict:
    """Live vast offers for THIS video dataset, one tier per GPU class, priced.

    The face lane has had this for weeks; the video lane launched without it,
    which meant "cheapest offer above the floors" was chosen for the user,
    silently, on a rented-by-the-minute decision. Read-only — rents nothing;
    the launch re-searches and takes the cheapest LIVE offer of the chosen
    class, exactly as the face lane does.

    Estimates are ROUGH and say so: the speed model behind them is one measured
    run (21 s/step at 107 frames on an A100), scaled by latent rows and by the
    shared per-GPU throughput table. A frame count off the 17n+5 grid gets no
    estimate at all rather than an invented one."""
    from . import gpu_speed
    from . import vast_client
    if not ct.cfg.secret('VAST_API_KEY'):
        raise RuntimeError('vast.ai API key is not configured — add it in Settings')
    ds = db.session.get(VideoDataset, int(video_dataset_id))
    if ds is None or str(ds.user_id) != str(user_id):
        raise ValueError('video dataset not found')
    n_steps = max(100, int(steps or 1000))
    c = ct.cfg.get('cloud') or {}
    min_vram = (c.get('min_vram_gb') or {}).get('video', 48)
    disk_gb = ct._disk_gb_for(c, {'train_type': 'video'})
    price_cap = c.get('max_price_per_hour', 0.80)
    overhead_min = float(c.get('pod_overhead_minutes') or 0)
    max_runtime = int(c.get('max_runtime_minutes') or 480)
    offers = ct._filter_offers(vast_client.search_offers(
        min_vram_gb=min_vram, max_dph=price_cap,
        limit=int(c.get('offer_scan_limit') or 100),
        min_inet_down_mbps=int(c.get('min_inet_down_mbps') or 0),
        min_reliability=float(c.get('min_reliability') or 0.98),
        min_disk_bw_mbps=int(c.get('min_disk_bw_mbps') or 0),
        verified_only=bool(c.get('verified_only', True)),
        secure_cloud_only=bool(c.get('secure_cloud_only', False)),
        min_disk_gb=disk_gb,
        # The same generation floor the launch applies — a picker that lists
        # cards the launch would refuse is a menu of dead ends.
        min_compute_cap=int((c.get('min_compute_cap') or {}).get('video', 0))))
    cheapest = {}
    for o in offers:
        name = o.get('gpu_name') or 'GPU'
        cur = cheapest.get(name)
        dph = o.get('dph_total')
        if cur is None or (dph is not None and (cur.get('dph_total') is None
                           or dph < cur['dph_total'])):
            cheapest[name] = o
    tiers = []
    for name, o in cheapest.items():
        dph = o.get('dph_total')
        est_min = gpu_speed.video_estimate_minutes(name, ds.frames, n_steps)
        est_cost = (round(dph * (est_min + overhead_min) / 60.0, 2)
                    if (est_min is not None and dph is not None) else None)
        tiers.append({
            'gpu_name': name, 'offer_id': o.get('offer_id'),
            'dph_total': round(dph, 4) if dph is not None else None,
            'gpu_ram_gb': o.get('gpu_ram_gb'),
            'speed': round(gpu_speed.speed_factor(name), 2),
            'est_minutes': int(round(est_min)) if est_min is not None else None,
            'est_cost': est_cost,
            'estimate_status': 'rough' if est_min is not None else 'unavailable',
            'exceeds_cap': ((est_min + overhead_min) > max_runtime
                            if est_min is not None else None),
        })
    tiers.sort(key=lambda t: (t['speed'], t['dph_total']
                              if t['dph_total'] is not None else 9e9))
    return {'tiers': tiers, 'steps': n_steps, 'frames': ds.frames,
            'disk_gb': disk_gb, 'min_vram_gb': min_vram,
            'max_price_per_hour': price_cap,
            'max_runtime_minutes': max_runtime}


def launch_cloud_video_training(user_id, video_dataset_id, steps=1000,
                                base_model=None, low_vram=False, gpu_name=None,
                                do_i2v=False,
                                resume_ckpt_paths=None, resume_step=None,
                                parent_run_id=None, _provision=None) -> dict:
    """Rent a pod and train a LoRA on a built video dataset.

    `low_vram` defaults to FALSE here and True in the builder, and the asymmetry
    is the whole point of this lane: on a 24 GB card the flag is mandatory and
    costs 170-185 s a step shuttling the idle expert over PCIe; on the 80 GB pod
    this function rents, leaving it on would pay cloud prices for the local
    machine's handicap.

    `_provision` overrides `_start_pod` for callers that drive provisioning
    themselves; leave it None and the shared monitor takes over, exactly as it
    does for a face run.

    `resume_ckpt_paths` is a LIST, and that is the one shape difference from the
    face lane's single `resume_ckpt_path`. A Wan 2.2 MoE checkpoint is two files
    — `_high_noise` and `_low_noise` — and seeding one of them onto a fresh pod
    resumes one expert while the other restarts from zero. Nothing raises; the
    LoRA simply comes back half as trained as its step count claims. So the
    continuation carries every file of the chosen step, and `_seed_resume_checkpoint`
    ships all of them.
    """
    ds = db.session.get(VideoDataset, int(video_dataset_id))
    if ds is None or str(ds.user_id) != str(user_id):
        raise ValueError('video dataset not found')

    profile = video_training.video_targets.get(ds.target_profile) or {}
    _ref_dirs = []
    if profile.get('requires_references'):
        from . import video_bank_service as _vbs
        _ref_dirs = _vbs.reference_dirs(ds)
        if not _ref_dirs:
            raise ValueError(
                f'{profile.get("label", ds.target_profile)} trains against '
                'reference images and this dataset has none attached — attach '
                '1-4 references first')
        c0 = ct.cfg.get('cloud') or {}
        image_tag = c0.get('video_image') or c0.get('image') or ''
        if not video_training.image_supports_ref2va(image_tag):
            raise ValueError(
                'the pinned pod image predates ref2va training '
                '(needs 2026-08-16 or later) — update cloud.video_image')
    clips = _count_clips(ds.output_dir or '')
    if not clips:
        raise ValueError(
            'this video dataset has no clips or stills on disk — there would '
            'be nothing to train on; rebuild it before launching')

    n_steps = max(100, int(steps or 1000))
    # Built HERE, before the reservation, purely so an unsupported target raises
    # now rather than from the monitor thread with a pod already running. The
    # result is deliberately discarded: the monitor rebuilds it from the stamped
    # params at pod boot.
    video_training.build_job_config(
        ds, str(ds.output_dir), n_steps, training_folder='__POD__',
        base_model=base_model, low_vram=low_vram, do_i2v=bool(do_i2v),
        # The validation build needs the SHAPE, not the pod paths — local dirs
        # prove the target's precondition; the monitor rebuilds with pod names.
        control_dirs=[str(d) for d in _ref_dirs] or None)

    fam = 'video'
    with ct._launch_reservation_lock:
        ct._assert_launch_guardrails(ds.id, fam, crd.VIDEO)
        run = CloudTrainingRun(
            dataset_id=ds.id, status='preparing',
            dataset_table=crd.VIDEO,
            run_name=video_training.job_name_for(ds),
            train_params=json.dumps({
                'train_type': fam,
                'steps': n_steps,
                'base_model': base_model or '',
                'low_vram': bool(low_vram),
                # Stamped like low_vram: the pod rebuild happens minutes later
                # and must not re-read a toggle the user may have moved since.
                'do_i2v': bool(do_i2v),
                'target_profile': ds.target_profile,
                'frames': ds.frames,
                'artifact_kind': 'lora',
                **({'requested_gpu': str(gpu_name)} if gpu_name else {}),
                # A continuation, and what it grew from. `resume_ckpt_paths`
                # (plural) is read by _seed_resume_checkpoint; `parent_run_id`
                # is the video lane's genealogy edge — a CloudTrainingRun id,
                # never a face TrainingRunRecord id.
                **({'resume_ckpt_paths': [str(p) for p in resume_ckpt_paths]}
                   if resume_ckpt_paths else {}),
                **({'resume_step': int(resume_step)}
                   if resume_step is not None else {}),
                **({'parent_run_id': int(parent_run_id)}
                   if parent_run_id is not None else {}),
            }))
        # Deliberately NO `version` key and no checkpoint_registry call. That
        # registry freezes a manifest of face-dataset IMAGES and their caption
        # hashes; filing a video run under face dataset #N would put it in that
        # dataset's lineage graph forever. A video run stays unversioned rather
        # than borrowing a number that is not its.
        db.session.add(run)
        db.session.commit()
    try:
        ct._set(run, vast_label=f'lds-{run.id}',
                job_name=f'lds{run.id}_{run.run_name}')
        (_provision or _start_pod)(run)
    except Exception as e:
        ct._set(run, status='error', error=f'launch failed: {e}')
        raise
    logger.info('cloud video run %s launched: %s clips, %s steps, profile %s',
                run.id, clips, n_steps, ds.target_profile)
    return {'run_id': run.id, 'status': run.status, 'job_name': run.job_name,
            'steps': n_steps, 'clips': clips}


# ── Relaunching: retry and continue, in the VIDEO lane ────────────────────────
# Both used to refuse. The refusal was right at the time: `retry_cloud_run` and
# `continue_cloud_run` rebuild their arguments from a run's stamped params and
# hand them to `launch_cloud_training`, which resolves `dataset_id` as a FACE
# dataset — so on a colliding id they would have launched a face training on
# someone else's data and billed for it. What was missing was not a guard, it
# was these two functions: the same rebuild, aimed at the video launcher.


def _params_of(run) -> dict:
    try:
        p = json.loads(run.train_params or '{}')
    except ValueError:
        return {}
    return p if isinstance(p, dict) else {}


def _relaunch_args(p) -> dict:
    """The launch arguments a video run replays. Read from the run's STAMPED
    params and never from the dataset row: the row may have been rebuilt at
    another length or retargeted since, and a replay that silently picked up
    today's target is a different training under the same name."""
    return {
        'base_model': p.get('base_model') or None,
        'low_vram': bool(p.get('low_vram', False)),
        # Found by self-review the day i2v shipped: a retry that drops this flag
        # replays an i2v run as t2v — the exact silent retarget this function's
        # docstring exists to forbid.
        'do_i2v': bool(p.get('do_i2v', False)),
        'gpu_name': p.get('requested_gpu'),
    }


def retry_cloud_video_run(user_id, run_id) -> dict:
    """↻ Relaunch a FAILED video run with the exact parameters of the original.
    A real launch on a fresh pod — same guardrails as any other — not a
    resurrection of the dead one."""
    run = db.session.get(CloudTrainingRun, int(run_id))
    if not run:
        raise ValueError('unknown cloud run')
    if run.status != 'error':
        raise ValueError('only a failed run can be retried')
    p = _params_of(run)
    return launch_cloud_video_training(
        user_id, run.dataset_id, steps=p.get('steps') or 1000, **_relaunch_args(p))


def harvested_steps(run) -> list:
    """This run's harvested saves GROUPED BY STEP, ascending.

    Grouping is not presentation here, it is correctness. A Wan 2.2 MoE
    checkpoint is two files at ONE step, and every operation that treats a save
    as a single file gets it wrong in a way that raises nothing — a download
    that serves one half, a continue that seeds one expert. So the unit this
    lane works in is the step, and its files travel together.

    The FINAL save carries no number in ai-toolkit's naming; it is reported at
    the run's total step count and flagged `final`."""
    saves = ct.run_checkpoint_files(run)
    if not saves:
        return []
    target = int(ct._run_param(run, 'steps') or 0)
    by_step = {}
    for name, path in saves.items():
        step, _stage = video_training.split_checkpoint_name(name)
        final = step is None
        key = (target if final else step, final)
        by_step.setdefault(key, []).append((name, path))
    out = []
    for (step, final), items in sorted(by_step.items()):
        items.sort()
        out.append({'step': int(step), 'final': final,
                    'files': [n for n, _ in items],
                    'paths': [p for _, p in items]})
    return out


def continue_cloud_video_run(user_id, run_id, extra_steps=1000,
                             from_step=None) -> dict:
    """▶ Resume a TERMINAL video run from one of its harvested steps and train
    `extra_steps` further. A fresh pod: the monitor drops every file of the
    chosen step into the new job's save_root before starting it, and ai-toolkit
    auto-resumes from what it finds there."""
    run = db.session.get(CloudTrainingRun, int(run_id))
    if not run:
        raise ValueError('unknown cloud run')
    if run.status in ct.ACTIVE_STATES:
        raise ValueError('a run that is still running cannot be continued — '
                         'wait for it to finish or fail')
    steps = harvested_steps(run)
    if not steps:
        raise ValueError('no harvested checkpoint to continue from — this run '
                         'has none left on disk; launch a fresh video run instead')
    if from_step is None:
        chosen = steps[-1]
    else:
        try:
            want = int(from_step)
        except (TypeError, ValueError):
            raise ValueError('from_step must be an integer step')
        matches = [s for s in steps if s['step'] == want]
        if not matches:
            raise ValueError(
                f'no harvested checkpoint at step {want} for this run '
                f'(available: {sorted({s["step"] for s in steps})})')
        # Prefer the numbered save over the unsuffixed final when they tie: the
        # numbered one states its step in its own name, the final one only
        # inherits it from the run's target.
        chosen = min(matches, key=lambda s: s['final'])
    try:
        extra = max(100, int(extra_steps))
    except (TypeError, ValueError):
        extra = 1000
    p = _params_of(run)
    return launch_cloud_video_training(
        user_id, run.dataset_id, steps=chosen['step'] + extra,
        resume_ckpt_paths=list(chosen['paths']), resume_step=chosen['step'],
        parent_run_id=run.id, **_relaunch_args(p))
