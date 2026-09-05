"""'Share configuration' for a training run — a paste-safe text recipe.

Every card on the 🏋️ Runs hub (cloud AND local) can produce a downloadable
`.txt` listing EVERYTHING that launch sent to ai-toolkit (family/variant/base,
rank/alpha/lr/optimizer/resolution/steps/save_every/timestep/masked/…) plus the
run's outcome, so it can be shared as a recipe or pasted verbatim into a
Discord/GitHub help thread.

Two run worlds share one addressing scheme (see resolve_run):
  - `cloud-<CloudTrainingRun.id>`  — every cloud run (active, finished, legacy)
  - `rec-<TrainingRunRecord.id>`   — local runs (they exist only in the registry)

Retrofit + graceful degradation: a run recorded before the settings snapshot
existed has no `settings` blob, so the "Training parameters" section reads
"not recorded on this run" instead of failing. NOTHING secret (HF/vast keys,
auth tokens) is ever emitted, and the whole text is run through the shared
home-path redaction so no local `C:\\Users\\<name>\\…` path leaks.
"""
import json
from ..extensions import db
from ..utils.timestamps import naive_utcnow
import re

from ..models import CloudTrainingRun, FaceDataset, TrainingRunRecord
from ..utils.redact import redact_user_paths
from ..version import APP_VERSION
from . import cloud_training as ct

_FAMILY_LABEL = {'zimage': 'Z-Image', 'krea': 'Krea 2', 'sdxl': 'SDXL',
                 'flux': 'FLUX.1', 'flux2klein': 'FLUX.2 Klein'}

_NOT_RECORDED = 'not recorded on this run'


def resolve_run(run_key):
    """(crun, rec) for a share key — either may be None; (None, None) = unknown.

    `cloud-<id>` resolves the CloudTrainingRun and back-links its registry row
    (for the settings snapshot) via cloud_run_id. `rec-<id>` resolves the
    TrainingRunRecord and its CloudTrainingRun (if it was a cloud launch)."""
    kind, _, sid = (run_key or '').partition('-')
    if not sid.isdigit():
        return None, None
    rid = int(sid)
    if kind == 'cloud':
        crun = db.session.get(CloudTrainingRun, rid)
        if crun is None:
            return None, None
        rec = (TrainingRunRecord.query
               .filter_by(cloud_run_id=crun.id)
               .order_by(TrainingRunRecord.id.desc()).first())
        return crun, rec
    if kind == 'rec':
        rec = db.session.get(TrainingRunRecord, rid)
        if rec is None:
            return None, None
        crun = (db.session.get(CloudTrainingRun, rec.cloud_run_id)
                if rec.cloud_run_id else None)
        return crun, rec
    return None, None


# --- value formatting --------------------------------------------------------

def _fmt_resolution(v):
    if isinstance(v, (list, tuple)):
        return ' + '.join(str(x) for x in v) + ' px'
    return f'{v} px'


def _fmt_lr(v):
    try:
        return f'{float(v):g}'
    except (TypeError, ValueError):
        return str(v)


def _fmt_steps(v):
    return f'{v} steps'


# (snapshot key, human label, formatter). Rendered in this order; any snapshot
# key NOT listed here is still emitted generically below, so future snapshot
# enrichment shows up in the file without a change here.
_SETTING_ROWS = [
    ('trigger', 'Trigger word', str),
    ('rank', 'LoRA rank', str),
    ('alpha', 'LoRA alpha', str),
    ('network_type', 'Network type', str),
    ('lokr_factor', 'LoKr factor', str),
    ('lokr_full_rank', 'LoKr full rank', lambda v: 'yes' if v else 'no'),
    ('resolution', 'Resolution', _fmt_resolution),
    ('save_every', 'Save every', _fmt_steps),
    ('max_step_saves', 'Max saved checkpoints', str),
    ('optimizer', 'Optimizer', str),
    ('lr', 'Learning rate', _fmt_lr),
    ('lr_scheduler', 'LR scheduler', str),
    ('warmup', 'Warmup steps', str),
    ('grad_accum', 'Gradient accumulation', str),
    ('timestep_type', 'Timestep type', str),
    ('dropout', 'LoRA dropout', str),
    ('ema', 'EMA decay', str),
    ('content_or_style', 'Content or style', str),
    ('do_differential_guidance', 'Differential guidance', lambda v: 'yes' if v else 'no'),
    ('differential_guidance_scale', 'Differential guidance scale', str),
    ('batch_size', 'Batch size', str),
    ('dual_captions', 'Dual captions', lambda v: 'yes' if v else 'no'),
    # Memory strategy: a quantised run and a full-precision one are different
    # experiments, so a shared config that omitted them could not be reproduced.
    ('quantize', 'Quantise base model', lambda v: 'yes' if v else 'no'),
    ('quantize_te', 'Quantise text encoder', lambda v: 'yes' if v else 'no'),
    ('low_vram', 'Low-VRAM loading', lambda v: 'yes' if v else 'no'),
    ('sample_every', 'Sample every', _fmt_steps),
]
_KNOWN_SETTING_KEYS = {k for k, _, _ in _SETTING_ROWS} | {'style_mode'}


def _snapshot_env(rec):
    """The 'machine at launch' block of a record's snapshot, or None. Never
    raises: a share file must render for a run recorded before snapshots too."""
    try:
        from . import run_snapshot
        return (run_snapshot.loads(rec) or {}).get('env')
    except Exception:
        return None


def _fmt_dt(dt):
    return dt.strftime('%Y-%m-%d %H:%M UTC') if dt else 'unknown'


def _slug(s):
    s = re.sub(r'[^\w.-]+', '_', (s or '').strip().lower())
    return s.strip('_') or 'dataset'


def _base_basename(value):
    """Trailing filename/tag of a custom base's path or repo id — split on both
    slash kinds so a Windows `…\\merges\\bigLove.safetensors` and an HF repo
    `owner/lds-base-hxxxx` both reduce to their leaf, leaking no parent path."""
    leaf = re.split(r'[\\/]', str(value or '').rstrip('\\/'))[-1]
    return leaf or str(value or '')


def _family_label(fam):
    return _FAMILY_LABEL.get(fam, fam or 'LoRA')


def _variant_label(variant):
    if not variant:
        return None
    return 'Raw' if variant == 'base' else variant


# --- rendering ---------------------------------------------------------------

def build_run_config_text(run_key):
    """{'filename', 'text'} for a run key, or None when the key is unknown
    (route -> 404). The text is markdown-light and fully redacted."""
    crun, rec = resolve_run(run_key)
    if crun is None and rec is None:
        return None

    # --- resolve fields from whichever source has them (rec preferred) ------
    is_cloud = crun is not None or (rec is not None and rec.source == 'cloud')
    dataset_id = (rec.dataset_id if rec is not None else crun.dataset_id)
    family = (rec.family if rec is not None else ct._run_family(crun)) or None
    variant = (rec.variant if rec is not None
               else ct._run_param(crun, 'variant'))
    base_model = (rec.base_model if rec is not None else '') or ''
    steps = (rec.steps if rec is not None else ct._run_param(crun, 'steps'))
    if rec is not None:
        masked = bool(rec.masked)
    else:
        mp = ct._run_param(crun, 'masked')
        masked = None if mp is None else bool(mp)
    version = (rec.version if rec is not None
               else ct._run_param(crun, 'version'))
    created = (rec.created_at if rec is not None else None) or \
        (crun.created_at if crun is not None else None)
    # Resolve in the table this run actually trained on. `FaceDataset.query.get`
    # on a video run returns the face dataset that merely shares its id, and the
    # exported config would then describe someone else's data — a shareable file
    # that is confidently wrong. A video run has no FaceDataset at all, so the
    # image-specific sections below simply have nothing to render.
    if crun is not None and rec is None:
        from . import cloud_run_dataset as crd
        dataset = crd.dataset_row(crun)
        dataset_name = getattr(dataset, 'name', None)
        if crd.is_video(crun):
            dataset = None
    else:
        dataset_name = ct._dataset_name(dataset_id)
        dataset = db.session.get(FaceDataset, dataset_id)

    settings = None
    if rec is not None and rec.settings:
        try:
            settings = json.loads(rec.settings)
        except ValueError:
            settings = None
    if not isinstance(settings, dict):
        settings = None
    # New snapshots explicitly stamp style_mode. Looking at the dataset kind as
    # well protects older Style rows that still carried the internal run id in
    # ``trigger`` before the always-on provenance fix.
    style_always_on = (
        (settings is not None and settings.get('style_mode') == 'always_on')
        or ((getattr(dataset, 'kind', None) or '').lower() == 'style')
    )

    image_count = None
    if rec is not None and rec.manifest:
        try:
            man = json.loads(rec.manifest)
            if isinstance(man, list):
                image_count = len(man)
        except ValueError:
            pass

    L = []
    L.append('# LoRA Dataset Studio — training configuration')
    L.append('')
    L.append(f'App version:   {APP_VERSION}')
    L.append(f'Run date:      {_fmt_dt(created)}')
    L.append(f'Source:        {"cloud (vast.ai)" if is_cloud else "local"}')
    fam_line = _family_label(family)
    vlabel = _variant_label(variant)
    if vlabel:
        fam_line += f'  (variant: {vlabel})'
    L.append(f'Model family:  {fam_line}')
    # A custom base is named by its leaf filename/tag only — never the parent
    # path (which redaction would keep as `~\merges\…`); an official base spells
    # out the family's canonical base so a reader knows exactly what it trained on.
    if base_model:
        base_line = _base_basename(base_model)
    else:
        base_line = f'{_family_label(family)} official base'
    L.append(f'Base model:    {base_line}')
    ds_bits = [dataset_name or f'#{dataset_id}']
    meta = []
    if version is not None:
        meta.append(f'version v{version}')
    if image_count is not None:
        meta.append(f'{image_count} image(s)')
    if meta:
        ds_bits.append('(' + ', '.join(meta) + ')')
    L.append(f'Dataset:       {" ".join(ds_bits)}')

    # --- training parameters -------------------------------------------------
    L.append('')
    L.append('## Training parameters (ai-toolkit)')
    L.append('')
    if settings is None:
        L.append(f'Training parameters are {_NOT_RECORDED} — it predates the '
                 'settings snapshot feature.')
    else:
        if style_always_on:
            L.append(f'{"Style mode:":<24}always-on (no activation trigger)')
        for key, label, fmt in _SETTING_ROWS:
            if style_always_on and key == 'trigger':
                continue
            if key in settings and settings[key] is not None:
                L.append(f'{label + ":":<24}{fmt(settings[key])}')
        # any enrichment key not in the known table -> generic line
        for key in sorted(settings):
            if key not in _KNOWN_SETTING_KEYS and settings[key] is not None:
                L.append(f'{key + ":":<24}{settings[key]}')
    # masked lives on the run row (not the snapshot) — always known for records.
    if masked is not None:
        L.append(f'{"Masked training:":<24}{"yes" if masked else "no"}')

    # --- machine -------------------------------------------------------------
    # The trainer's own version is half of any "why did this run come out
    # different" answer, and it is exactly what a help thread asks for first.
    # Already redacted at capture time; nothing here is secret.
    env = (_snapshot_env(rec) or {})
    if env:
        L.append('')
        L.append('## Machine at launch')
        L.append('')
        at = env.get('aitoolkit') or {}
        if at.get('commit'):
            when = f' ({at["committed_at"][:10]})' if at.get('committed_at') else ''
            L.append(f'{"ai-toolkit:":<24}{at["commit"]}{when}')
        if env.get('torch'):
            cu = f' / CUDA {env["cuda"]}' if env.get('cuda') else ''
            L.append(f'{"PyTorch:":<24}{env["torch"]}{cu}')
        if env.get('gpu'):
            drv = f' (driver {env["gpu_driver"]})' if env.get('gpu_driver') else ''
            L.append(f'{"Local GPU:":<24}{env["gpu"]}{drv}')
        bf = env.get('base_file') or {}
        if bf.get('name'):
            L.append(f'{"Base file:":<24}{bf["name"]} '
                     f'({bf.get("size", 0) / (1 << 30):.1f} GB, id {bf.get("sig", "?")})')

    # --- run outcome ---------------------------------------------------------
    L.append('')
    L.append('## Run outcome')
    L.append('')
    if steps is not None:
        L.append(f'{"Target steps:":<24}{steps}')
    if crun is not None:
        L.append(f'{"Status:":<24}{crun.status}')
        if crun.error:
            L.append(f'{"Error:":<24}{crun.error}')
        if crun.gpu_name:
            L.append(f'{"GPU:":<24}{crun.gpu_name}')
        if crun.price_per_hour:
            L.append(f'{"Cost:":<24}~${ct._cost_estimate(crun):.2f} '
                     f'(${crun.price_per_hour}/h)')
        if crun.finished_at:
            L.append(f'{"Finished:":<24}{_fmt_dt(crun.finished_at)}')
    else:
        L.append(f'{"Status:":<24}not recorded (local run history)')

    L.append('')
    L.append(f'Generated by LoRA Dataset Studio v{APP_VERSION} — '
             'paste-safe, no local paths or keys.')

    text = redact_user_paths('\n'.join(L) + '\n')
    date = created.strftime('%Y%m%d') if created else naive_utcnow().strftime('%Y%m%d')
    filename = (f'lds-config-{_slug(dataset_name)}-{_slug(family)}'
                f'-v{version if version is not None else "na"}-{date}.txt')
    return {'filename': filename, 'text': text}
