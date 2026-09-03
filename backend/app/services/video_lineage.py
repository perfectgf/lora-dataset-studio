"""The ◉ Graph of a VIDEO dataset's runs, and the previews behind its pills.

WHY THE SAME SHAPE AS THE IMAGE TREE. The image workspace draws its runs with
`utils/lineageGraph.buildLineageGraph` and the `GraphCard` / `CheckpointPill`
components — a pure layout over `{nodes, edges}` where a node is a run keyed by
`record_id`, `edges` carry `parent → child` with the step the child resumed
from, and each node holds its `checkpoints` as pills. That layout knows nothing
of `TrainingRunRecord`; it reads a dozen fields. So the video lane answers the
same fields for its own rows — a `CloudTrainingRun` per cloud launch, whose
genealogy is the `parent_run_id` its params stamp (see `video_run_lineage`),
plus ONE node for the local run — and the same drawing serves both surfaces.
Anything image-specific the tree carries (versions, notes, the Lab's generated
previews) is simply absent here: `version` is None because video runs are
unversioned by design (`cloud_video_training`), and a pill's preview is the
training SAMPLE ai-toolkit rendered at that step, not a Studio generation.

THE LOCAL NODE. A local video run has no row anywhere (the folder IS the run:
`video_training_local`), so its node borrows a `record_id` that can collide
with nothing a cloud run will ever have — the dataset id, NEGATED. The layout
only ever compares ids for equality and maps them; nothing sorts or ranks them
(the refuter's brief covers this premise).

THE PILL IS A STEP. As everywhere else on this lane: a Wan 2.2 save is two
files at one step, so a pill carries `files` and one download per file, and
`filename`/`download_url` (what the shared renderers read) name its FIRST file
only as a handle — never as "the" file.

SAMPLES. ai-toolkit writes a sample every `sample_every` steps into the run's
`samples/` folder (`<ts>__<step>_<promptidx>.<ext>`; `_SAMPLE_RE` accepts the
video containers as well as the image formats, because the extension a video
model writes is a fact of ai-toolkit's sampler, not of this app). Locally that
folder sits under the save root; a pod's samples are mirrored into staging by
the monitor (`cloud_training._pull_log_and_samples`). A poster (first frame)
is cut on demand with the video bank's own thumbnail writer and cached under
the app's data dir, so a pill can show a still without the page holding a
`<video>` per step.
"""
import json
import logging
import os
import re

from .. import config as cfg
from ..models import CloudTrainingRun
from . import cloud_run_dataset as crd
from . import cloud_training as ct
from . import cloud_video_training as cvt
from . import lora_training as lt
from . import video_checkpoints as vck
from . import video_targets
from . import video_training_local as vtl

logger = logging.getLogger(__name__)

# `<timestamp>__<step>_<promptidx>.<ext>` — the image lane's `_SAMPLE_RE` with
# the video containers added. Anything else in the folder is not a sample.
_SAMPLE_RE = re.compile(r'__(\d+)_(\d+)\.(mp4|webm|mov|gif|webp|png|jpe?g)$', re.IGNORECASE)
_VIDEO_EXTS = ('.mp4', '.webm', '.mov', '.gif')
_POSTER_CACHE = 'video_samples'


def _target_label(profile) -> str | None:
    """The card's variant chip: the target's catalogue LABEL ("Wan 2.2 T2V
    A14B"), not its key ("wan22_14b") — the shared chip uppercases whatever it
    is handed, and a key uppercased is a code, a label uppercased is words."""
    if not profile:
        return None
    try:
        return (video_targets.get(profile) or {}).get('label') or str(profile)
    except Exception:               # an unknown or retired profile key
        return str(profile)


def local_record_id(ds) -> int:
    """The local run's node id: the dataset id negated (see the module doc)."""
    return -int(ds.id)


def _cloud_runs(ds) -> list:
    """This dataset's cloud runs, oldest first — the order the layout sorts
    siblings in. Ownership is the (id, table) pair."""
    return [r for r in (CloudTrainingRun.query.filter_by(dataset_id=ds.id)
                        .order_by(CloudTrainingRun.id.asc()).all())
            if crd.owns(r, ds.id, crd.VIDEO)]


def _params(run) -> dict:
    try:
        return json.loads(run.train_params or '{}')
    except ValueError:
        return {}


# ── samples ─────────────────────────────────────────────────────────────────


def samples_dir(ds, run=None) -> str | None:
    """Where this lane's samples are, or None when the lane has no folder."""
    if run is None:
        try:
            return os.path.join(str(vtl.save_root(ds)), 'samples')
        except RuntimeError:            # no local trainer configured
            return None
    return os.path.join(run.staging_dir, 'samples') if run.staging_dir else None


def list_samples(ds, run=None) -> list:
    """``[{filename, step, prompt_idx, kind}]`` newest step first, then prompt
    order. `kind` is 'video' or 'image' by extension — the poster route serves
    an image sample as its own poster."""
    d = samples_dir(ds, run)
    if not d or not os.path.isdir(d):
        return []
    out = []
    for name in os.listdir(d):
        m = _SAMPLE_RE.search(name)
        if not m:
            continue
        out.append({'filename': name, 'step': int(m.group(1)),
                    'prompt_idx': int(m.group(2)),
                    'kind': 'video' if name.lower().endswith(_VIDEO_EXTS) else 'image'})
    out.sort(key=lambda s: (-s['step'], s['prompt_idx']))
    return out


def sample_path(ds, run, filename) -> str | None:
    """Resolve ONE sample by basename through the lane's own listing — a request
    names a file, never a path."""
    if not filename or os.path.basename(filename) != filename:
        return None
    if not any(s['filename'] == filename for s in list_samples(ds, run)):
        return None
    path = os.path.join(samples_dir(ds, run), filename)
    return path if os.path.isfile(path) else None


def poster_path(ds, run, filename) -> str | None:
    """A still of one sample: the image itself for an image sample, else the
    first frame cut once and cached under the data dir. None when the sample
    is not there or the frame cannot be cut (no PyAV, unreadable file) — the
    pill then simply shows no thumbnail."""
    from .video_bank_service import _write_thumbnail
    src = sample_path(ds, run, filename)
    if not src:
        return None
    if not filename.lower().endswith(_VIDEO_EXTS):
        return src
    lane = f'run_{int(run.id)}' if run is not None else f'local_{int(ds.id)}'
    cache = cfg.data_dir() / 'cache' / _POSTER_CACHE / lane
    dst = str(cache / (filename + '.jpg'))
    try:
        if os.path.isfile(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
            return dst
    except OSError:
        pass
    return dst if _write_thumbnail(src, 0.0, dst) else None


def _sample_urls(ds, run, sample) -> dict:
    q = f"filename={sample['filename']}"
    if run is not None:
        q = f'run_id={int(run.id)}&' + q
    base = f'/api/video-dataset/{int(ds.id)}/train/sample'
    return {'url': f'{base}?{q}', 'poster_url': f'{base}/poster?{q}'}


# ── the tree ────────────────────────────────────────────────────────────────


def _pills(ds, run, steps, paths, deployed, samples) -> list:
    """One pill per STEP, with the fields the shared renderers read, the
    video lane's own (`files`, `download_urls`) and the previews: the sample
    of prompt 0 at that step as the thumbnail, the count of samples at it."""
    by_step = {}
    for s in samples:
        by_step.setdefault(s['step'], []).append(s)
    out = []
    for s in steps:
        rows = vck._step_rows([s], paths.get, deployed)[0]
        files = rows['files']
        names = [f['filename'] for f in files]
        if run is not None:
            urls = [f'/api/video-dataset/{ds.id}/train/cloud/checkpoint?run_id={run.id}&filename={n}'
                    for n in names]
        else:
            urls = [f'/api/video-dataset/{ds.id}/train/checkpoint?filename={n}' for n in names]
        at_step = by_step.get(s['step']) if s['step'] is not None else None
        first = min(at_step, key=lambda x: x['prompt_idx']) if at_step else None
        pill = {
            'step': s['step'], 'final': bool(s['final']),
            'filename': names[0] if names else None, 'files': files,
            'download_url': urls[0] if urls else None, 'download_urls': urls,
            'present': bool(names),
            'testable': rows['deployed'],
            'deployed_filename': (files[0]['deployed_as'] if files and rows['deployed'] else None),
            'undeployable': bool(files) and all(f['undeployable'] for f in files),
            'preview_count': len(at_step or []),
        }
        if first is not None:
            pill.update({'preview_status': 'ready', **{k: v for k, v in
                         _sample_urls(ds, run, first).items() if k == 'poster_url'}})
            pill['preview_url'] = pill.pop('poster_url')
            pill['sample_url'] = _sample_urls(ds, run, first)['url']
        else:
            pill['preview_url'] = None
            pill['preview_status'] = None
        out.append(pill)
    return out


def local_total_steps(ds) -> int | None:
    """The step count the local run was launched with, read from the job
    config the launcher wrote (`<jobs_dir>/<run_name>.json`, ai-toolkit's own
    format). None when there is no such file — the final save then keeps
    `step: None` and the label says "Final" without inventing a number."""
    try:
        path = os.path.join(str(lt._jobs_dir()), vtl.local_run_name(ds) + '.json')
        with open(path, encoding='utf-8') as fh:
            job = json.load(fh)
        for proc in (job.get('config') or {}).get('process') or []:
            steps = (proc.get('train') or {}).get('steps')
            if steps:
                return int(steps)
    except (OSError, ValueError, TypeError, AttributeError, RuntimeError):
        return None
    return None


def _cloud_node(ds, run, run_ids, deployed) -> dict:
    p = _params(run)
    parent = p.get('parent_run_id')
    parent = int(parent) if parent is not None and int(parent) in run_ids else None
    resumed_from = p.get('resume_step')
    steps = cvt.harvested_steps(run)
    paths = ct.run_checkpoint_files(run)
    pills = _pills(ds, run, steps, paths, deployed, list_samples(ds, run))
    return {
        'record_id': run.id, 'run_id': run.id, 'source': 'cloud',
        'parent_record_id': parent,
        'resumed_from': int(resumed_from) if resumed_from is not None else None,
        'origin_unknown': bool(resumed_from is not None and parent is None),
        'dataset_id': ds.id, 'dataset_name': ds.name,
        'train_type': 'video',
        'variant': _target_label(p.get('target_profile') or ds.target_profile),
        'base_model': p.get('base_model') or '', 'version': None,
        'steps': p.get('steps'),
        'config': {k: p[k] for k in vck._PUBLIC_PARAMS if k in p},
        'note': '', 'has_note': False, 'is_current': False,
        'created_at': run.created_at.isoformat() if run.created_at else None,
        'finished_at': run.finished_at.isoformat() if run.finished_at else None,
        'status': run.status, 'active': run.status in ct.ACTIVE_STATES,
        'training_mode': 'lora', 'gpu': run.gpu_name, 'price_per_hour': run.price_per_hour,
        'checkpoints': pills, 'saves': len(paths), 'checkpoint_ready': bool(pills),
    }


def _local_node(ds, deployed) -> dict | None:
    saves = vck._local_saves(ds)
    if not saves:
        return None
    total = local_total_steps(ds)
    steps = cvt.group_saves_by_step(saves, target=total)
    active = bool(vtl.video_training_progress(ds.id, ds.user_id)['active'])
    pills = _pills(ds, None, steps, saves, deployed, list_samples(ds, None))
    return {
        'record_id': local_record_id(ds), 'run_id': None, 'source': 'local',
        'parent_record_id': None, 'resumed_from': None, 'origin_unknown': False,
        'dataset_id': ds.id, 'dataset_name': ds.name,
        'train_type': 'video', 'variant': _target_label(ds.target_profile),
        'base_model': '', 'version': None, 'steps': total,
        'config': {}, 'note': '', 'has_note': False, 'is_current': False,
        'created_at': None, 'finished_at': None,
        'status': 'training' if active else None, 'active': active,
        'training_mode': 'lora', 'run_name': vtl.local_run_name(ds),
        'checkpoints': pills, 'saves': len(saves), 'checkpoint_ready': bool(pills),
    }


def tree(user_id, dataset_id) -> dict:
    """The genealogy forest of this dataset's runs — the shape
    `cloud_training.dataset_lineage` answers, for the video lane. Cloud runs
    oldest first, then the local node; an edge per continuation whose parent
    is one of these runs (a parent that was deleted leaves an honest root with
    "origin unknown"). Empty → an empty, safe shape."""
    ds = vck._dataset(user_id, dataset_id)
    deployed = vck._deployed_index()
    runs = _cloud_runs(ds)
    run_ids = {r.id for r in runs}
    nodes = [_cloud_node(ds, r, run_ids, deployed) for r in runs]
    local = _local_node(ds, deployed)
    if local is not None:
        nodes.append(local)
    edges = [{'parent': n['parent_record_id'], 'child': n['record_id'],
              'resumed_from': n['resumed_from'], 'superseded': False}
             for n in nodes if n['parent_record_id'] is not None]
    return {'root_id': None, 'current_id': None, 'nodes': nodes, 'edges': edges,
            'single': len(nodes) < 2}


def resolve_run(ds, run_id):
    """The lane a `run_id` query names: None for the local run (absent, empty
    or 'local'), else one of THIS dataset's cloud runs or LookupError."""
    if run_id in (None, '', 'local', 'null'):
        return None
    return vck._cloud_run(ds, run_id)
