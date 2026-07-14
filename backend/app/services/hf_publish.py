"""Publish a dataset to the Hugging Face Hub as a `dataset` repo — EXPORT only.

Reuses the training-export SHAPE (kept images + a same-stem `.txt` caption with
the trigger word prepended, via `face_dataset_service._export_caption`) but with
three deliberate differences from the ZIP export:

  * files are kept **as-is** on disk (a `.webp` stays `.webp`) instead of the
    ZIP's PNG re-encode — smaller repo, and the HF viewer renders webp fine;
  * the **real reference photo** (`_000_ref`) is included ONLY when `include_ref`
    is set (default OFF) — it's the source face of a possibly real person;
  * a **`metadata.jsonl`** (`{"file_name", "text"}` per image) and a
    **`README.md`** dataset card are generated so the repo is readable by the HF
    Dataset Viewer and `load_dataset("imagefolder")`.

Upload goes through `huggingface_hub.HfApi`. A write-scope preflight (`whoami`)
refuses a read-only token BEFORE any upload. NOTHING secret is ever written into
the repo, and every line of the README passes through `redact_user_paths` so no
local `C:\\Users\\<name>\\…` path can leak. The HF token is the app's existing
`HF_TOKEN` secret (same one cloud training / model downloads read).

The long part is the upload of potentially hundreds of files: `start_publish`
runs the whole flow in a daemon thread (mirroring `setup_installer`) and
`publish_status` reports {state, repo_url, error, error_code} for the UI to poll.
`publish_to_hf` itself is fully synchronous and the single seam the tests drive
(HfApi mocked — no real network).
"""
import json
import os
import re
import shutil
import tempfile
import threading

from .. import config as cfg
from ..config import LOCAL_USER
from ..utils.redact import redact_user_paths
from ..version import APP_VERSION
from . import face_dataset_service as fds

# Licence choices offered in the modal — the usual dataset licences plus a
# catch-all. Validated server-side (never trust the client's dropdown).
LICENSE_CHOICES = ('cc0-1.0', 'cc-by-4.0', 'cc-by-nc-4.0', 'openrail', 'other')

_FAMILY_LABEL = {'zimage': 'Z-Image', 'krea': 'Krea 2', 'sdxl': 'SDXL',
                 'flux': 'FLUX.1', 'flux2klein': 'FLUX.2 Klein'}
_KIND_LABEL = {'concept': 'concept', 'style': 'style'}   # else -> 'character'

GITHUB_REPO = 'perfectgf/lora-dataset-studio'
GITHUB_URL = f'https://github.com/{GITHUB_REPO}'

READ_ONLY_MSG = ('your Hugging Face token is read-only — create a write token at '
                 'https://huggingface.co/settings/tokens')

# HF repo id: `<namespace>/<name>`, each segment [A-Za-z0-9._-], name up to 96.
_REPO_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$')


class HfPublishError(Exception):
    """A structured, user-facing failure. `code` lets the UI branch (notably on
    `read_only_token`, the most likely first-use failure); `message` is the human
    line shown as-is."""
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


# --- naming ------------------------------------------------------------------

def _slug(s) -> str:
    """A repo-name-safe slug: keep alphanumerics/._-, collapse the rest to a
    single dash, lowercase. Empty -> 'dataset'."""
    s = re.sub(r'[^A-Za-z0-9._-]+', '-', (s or '').strip())
    s = re.sub(r'-{2,}', '-', s).strip('-._').lower()
    return s or 'dataset'


def _safe_stem(ds) -> str:
    """File-name stem for the exported images (mirrors build_export_zip's `safe`)."""
    return ''.join(c for c in ds.name if c.isalnum() or c in ('-', '_')) or 'dataset'


def default_repo_id(username, ds) -> str:
    """`<username>/<slug-of-dataset-name>` — the modal's pre-filled default."""
    stem = _slug(ds.name) if ds is not None else 'dataset'
    return f'{username}/{stem}' if username else stem


def _valid_repo_id(repo_id) -> bool:
    return bool(_REPO_ID_RE.match(repo_id or ''))


# --- huggingface_hub seam ----------------------------------------------------

def _make_api(token):
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise HfPublishError(
            'hub_missing',
            'the huggingface-hub package is not installed — run '
            'pip install -r backend/requirements.txt') from e
    return HfApi(token=token)


def _http_status(e):
    """Best-effort HTTP status code out of a huggingface_hub error (its
    HfHubHTTPError carries `.response.status_code`); falls back to scraping a
    3-digit 4xx/5xx out of the message. None when nothing looks like a status."""
    resp = getattr(e, 'response', None)
    code = getattr(resp, 'status_code', None)
    if isinstance(code, int):
        return code
    m = re.search(r'\b([45]\d\d)\b', str(e))
    return int(m.group(1)) if m else None


def hf_namespace(token):
    """The token owner's username (for the default repo id), or None if the token
    is missing/invalid/unreachable. Never raises — the modal degrades to a free
    text field with a placeholder."""
    if not token:
        return None
    try:
        who = _make_api(token).whoami()
    except Exception:
        return None
    return (who or {}).get('name') or None


# --- write-scope preflight ---------------------------------------------------

def _fine_grained_can_write(who) -> bool:
    """A fine-grained token advertises its permissions under
    auth.accessToken.fineGrained (`global` list + `scoped[].permissions`). We
    accept it if ANY permission string mentions write (repo.write /
    repo.content.write / …) — anything narrower is treated as read-only."""
    fg = (((who.get('auth') or {}).get('accessToken') or {}).get('fineGrained')) or {}
    perms = list(fg.get('global') or [])
    for scope in (fg.get('scoped') or []):
        perms.extend(scope.get('permissions') or [])
    return any('write' in (p or '').lower() for p in perms)


def _require_write_scope(api):
    """whoami() then inspect auth.accessToken.role: 'write' passes, 'fineGrained'
    passes only with a write permission, 'read'/missing is refused BEFORE any
    upload. Returns the whoami dict on success."""
    try:
        who = api.whoami()
    except Exception as e:
        status = _http_status(e)
        if status in (401, 403):
            raise HfPublishError(
                'auth', 'your Hugging Face token was rejected (invalid or expired).') from e
        raise HfPublishError(
            'network', f'could not reach Hugging Face to verify the token: {e}') from e
    role = (((who or {}).get('auth') or {}).get('accessToken') or {}).get('role')
    if role == 'write':
        return who
    if role == 'fineGrained' and _fine_grained_can_write(who):
        return who
    raise HfPublishError('read_only_token', READ_ONLY_MSG)


# --- dataset folder build ----------------------------------------------------

def _write_text(dest_dir, name, text):
    with open(os.path.join(dest_dir, name), 'w', encoding='utf-8') as fh:
        fh.write(text)


def _target_family(ds) -> str:
    return (getattr(ds, 'train_type', None) or 'zimage').lower()


def _kind_label(ds) -> str:
    return _KIND_LABEL.get((getattr(ds, 'kind', None) or '').lower(), 'character')


def build_readme(ds, count, license, nfaa) -> str:
    """YAML front-matter (license / task_categories / tags) + a dataset card
    derived from the ⎘ Share-config pattern (kind, target family, image count,
    trigger, 'built with LoRA Dataset Studio' + repo link). Fully redacted."""
    fam = _FAMILY_LABEL.get(_target_family(ds), _target_family(ds))
    kind = _kind_label(ds)
    trigger = ds.trigger_word or ''
    tags = ['lora-dataset-studio']
    if nfaa:
        tags.append('not-for-all-audiences')

    fm = ['---', f'license: {license}', 'task_categories:', '- text-to-image', 'tags:']
    fm += [f'- {t}' for t in tags]
    fm += ['size_categories:', '- n<1K', '---', '']

    trig_line = (f'| Trigger word | `{trigger}` |' if trigger
                 else '| Trigger word | _(none — style dataset)_ |')
    body = [
        f'# {ds.name}',
        '',
        f'A LoRA training dataset built with '
        f'**[LoRA Dataset Studio]({GITHUB_URL})**.',
        '',
        '| | |',
        '|---|---|',
        f'| Dataset type | {kind} |',
        f'| Target model family | {fam} |',
        trig_line,
        f'| Images | {count} |',
        '',
        '## Contents',
        '',
        'Each image ships with a matching `.txt` caption (the trigger word is '
        'prepended), and a `metadata.jsonl` maps every image to its caption, so '
        'the dataset loads straight into the 🤗 Datasets library:',
        '',
        '```python',
        'from datasets import load_dataset',
        'ds = load_dataset("imagefolder", data_dir=".", split="train")',
        '```',
        '',
    ]
    if trigger:
        body += [f'Put the trigger word **`{trigger}`** in your prompts to summon '
                 'this concept once the LoRA is trained.', '']
    body += [
        '---',
        '',
        f'Exported automatically with [LoRA Dataset Studio]({GITHUB_URL}) — an '
        'open-source tool for building character / style / concept LoRA datasets. '
        'The uploader is responsible for the rights to these images and for the '
        'consent of any identifiable person shown.',
        '',
    ]
    return redact_user_paths('\n'.join(fm + body))


def build_publish_dir(user_id, dataset_id, dest_dir, include_ref, license, nfaa):
    """Populate `dest_dir` with the HF-ready dataset (images as-is + same-stem
    `.txt` + metadata.jsonl + README.md) and return {count, entries, readme,
    trigger}. `entries` is the metadata rows (also what tests assert on)."""
    from ..models import FaceDatasetImage
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise HfPublishError('dataset_not_found', 'dataset not found')
    license = (license or '').strip().lower()
    if license not in LICENSE_CHOICES:
        raise HfPublishError('invalid_license', f'unsupported license: {license or "(empty)"}')

    kept = (FaceDatasetImage.query.filter_by(dataset_id=dataset_id, status='keep')
            .order_by(FaceDatasetImage.id.asc()).all())
    entries = []
    stem = _safe_stem(ds)

    # Real reference photo: opt-in only. Caption = the bare trigger (an identity
    # anchor, not a described image).
    if include_ref and ds.ref_filename:
        ref_path = fds._ref_path(ds)
        if os.path.exists(ref_path):
            ext = os.path.splitext(ds.ref_filename)[1].lower() or '.webp'
            name = f'{stem}_000_ref{ext}'
            shutil.copyfile(ref_path, os.path.join(dest_dir, name))
            text = redact_user_paths(ds.trigger_word or '')
            _write_text(dest_dir, f'{stem}_000_ref.txt', text)
            entries.append({'file_name': name, 'text': text})

    for n, img in enumerate(kept, 1):
        if not img.filename:
            continue
        src = fds._img_path(img)
        if not os.path.exists(src):
            continue
        ext = os.path.splitext(img.filename)[1].lower() or '.webp'
        name = f'{stem}_{n:03d}{ext}'
        shutil.copyfile(src, os.path.join(dest_dir, name))
        # redact is a no-op on real captions; belt-and-suspenders so NOTHING in
        # the public repo can carry a stray home path.
        text = redact_user_paths(fds._export_caption(ds, img.caption))
        _write_text(dest_dir, f'{stem}_{n:03d}.txt', text)
        entries.append({'file_name': name, 'text': text})

    if not entries:
        raise HfPublishError('no_images', 'no kept images with files on disk to publish')

    with open(os.path.join(dest_dir, 'metadata.jsonl'), 'w', encoding='utf-8') as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + '\n')

    readme = build_readme(ds, len(entries), license, nfaa)
    _write_text(dest_dir, 'README.md', readme)
    return {'count': len(entries), 'entries': entries, 'readme': readme,
            'trigger': ds.trigger_word}


# --- the synchronous publish flow (the tested seam) --------------------------

def publish_to_hf(dataset_id, repo_id, private, nfaa, license, include_ref, token,
                  user_id=LOCAL_USER, _api=None):
    """Full synchronous publish: validate -> write-scope preflight -> build temp
    folder -> create_repo(exist_ok=False) -> upload_folder. Returns
    {ok, repo_url, repo_id, count} or raises HfPublishError. `_api` lets tests
    inject a mocked HfApi (no real network ever)."""
    if not token:
        raise HfPublishError('no_token', 'no Hugging Face token configured — '
                             'paste an HF_TOKEN in Settings ▸ API keys')
    repo_id = (repo_id or '').strip()
    if not _valid_repo_id(repo_id):
        raise HfPublishError('invalid_repo_id',
                             'invalid repo id — use the form "<username>/<name>"')
    license = (license or '').strip().lower()
    if license not in LICENSE_CHOICES:
        raise HfPublishError('invalid_license', f'unsupported license: {license or "(empty)"}')

    api = _api or _make_api(token)
    _require_write_scope(api)                      # refuse read-only BEFORE upload

    tmp = tempfile.mkdtemp(prefix='lds-hf-')
    try:
        info = build_publish_dir(user_id, dataset_id, tmp, include_ref, license, nfaa)
        try:
            api.create_repo(repo_id=repo_id, repo_type='dataset',
                            private=bool(private), exist_ok=False)
        except HfPublishError:
            raise
        except Exception as e:
            status = _http_status(e)
            if status == 409:
                raise HfPublishError(
                    'repo_exists',
                    f'the dataset repo "{repo_id}" already exists — pick a new '
                    'name (this tool never overwrites an existing repo)') from e
            if status in (401, 403):
                raise HfPublishError(
                    'auth', 'Hugging Face refused to create the repo (401/403) — '
                    'check the token has write access') from e
            raise HfPublishError('network', f'could not create the repo: {e}') from e

        try:
            api.upload_folder(folder_path=tmp, repo_id=repo_id, repo_type='dataset',
                              commit_message=f'Add dataset (LoRA Dataset Studio v{APP_VERSION})')
        except Exception as e:
            status = _http_status(e)
            if status in (401, 403):
                raise HfPublishError(
                    'auth', 'Hugging Face rejected the upload (401/403) — '
                    'check the token has write access') from e
            raise HfPublishError('network', f'the upload to Hugging Face failed: {e}') from e

        return {'ok': True, 'repo_id': repo_id, 'count': info['count'],
                'repo_url': f'https://huggingface.co/datasets/{repo_id}'}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --- background job (route seam; upload can exceed a request's window) --------
# The upload of a full dataset (dozens–hundreds of webp + txt, tens of MB) over a
# home uplink routinely runs past 30 s — too long for a synchronous request. So
# the route launches a daemon thread (same shape as setup_installer) and the UI
# polls `publish_status`. The DB reads happen inside the passed app's context.

_lock = threading.Lock()
_jobs = {}   # dataset_id -> {state, repo_url, repo_id, error, error_code, count}


def _job_snapshot(dataset_id):
    j = _jobs.get(dataset_id)
    if not j:
        return {'state': 'idle'}
    return dict(j)


def publish_status(dataset_id):
    with _lock:
        return _job_snapshot(dataset_id)


def start_publish(app, dataset_id, repo_id, private, nfaa, license, include_ref,
                  token, user_id=LOCAL_USER):
    """Launch the publish in the background. Returns the initial status; a second
    call while one is running is a no-op (`already=True`)."""
    with _lock:
        cur = _jobs.get(dataset_id)
        if cur and cur.get('state') == 'running':
            return {**cur, 'already': True}
        _jobs[dataset_id] = {'state': 'running', 'repo_url': None, 'repo_id': repo_id,
                             'error': None, 'error_code': None, 'count': None}
    threading.Thread(
        target=_run_publish_job,
        args=(app, dataset_id, repo_id, private, nfaa, license, include_ref, token, user_id),
        daemon=True).start()
    return {'state': 'running'}


def _run_publish_job(app, dataset_id, repo_id, private, nfaa, license, include_ref,
                     token, user_id):
    try:
        with app.app_context():
            res = publish_to_hf(dataset_id, repo_id, private, nfaa, license,
                                include_ref, token, user_id=user_id)
        result = {'state': 'done', 'repo_url': res['repo_url'], 'repo_id': res['repo_id'],
                  'error': None, 'error_code': None, 'count': res['count']}
    except HfPublishError as e:
        result = {'state': 'error', 'repo_url': None, 'repo_id': repo_id,
                  'error': e.message, 'error_code': e.code, 'count': None}
    except Exception as e:   # never let a background thread die silently
        result = {'state': 'error', 'repo_url': None, 'repo_id': repo_id,
                  'error': f'unexpected error: {e}', 'error_code': 'unknown', 'count': None}
    with _lock:
        _jobs[dataset_id] = result
