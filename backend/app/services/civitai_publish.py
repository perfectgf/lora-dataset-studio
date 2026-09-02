"""📤 Publish to Civitai — a trained checkpoint as a model page, and the images it
generated as posts on that page, without leaving the app.

Two gestures, one credential:

* **the checkpoint** (a pill on the ◉ Graph / LoRA Canvas) becomes a Civitai
  model page — model → version → the `.safetensors` uploaded in 25 MB parts →
  the file registered on the version. The page is created as a DRAFT unless
  the user ticks "publish now": a model page deserves a last look on Civitai
  (cover images, licence, description) before the world sees it;
* **a generated image** (the shared viewer's 📤 verb) becomes a post on the
  version its checkpoint is linked to, carrying the prompt, seed, sampler and
  the LoRA weight it was made with, so the picture files itself under the
  right model with its generation data. That post is published right away by
  default — that is what "post it" means — and can be left as a draft instead.

The two meet on the LINK: `civitai_link` remembers which Civitai model version
a SAVE is — keyed by `(record_id, step, filename)`, three columns because a
run that ends on a numbered save writes two files at its last step (the
numbered one and the step-less final) and the step alone cannot tell them
apart (see models.CivitaiLink). Marking the page first (paste its URL, or
create it from here) is what makes the image gesture a single press later.

A picture names its save without a file name: its row carries the checkpoint
stamped at generation (`record_id`, `step`) and the DEPLOYED LoRA name it ran
with (`checkpoint`, e.g. `lora_x_000001500_Krea-2-Raw_rc158_v1`). The
zero-padded step block that name carries is what tells the numbered save from
the final when both sit at one step, so every entry point takes that name as a
`hint` and resolves the save server-side — a picture can mark its page, and
create it, without ever being asked "which file".

That stamp is NULL for every picture made with a run's FINAL save (its deployed
name carries no step). So "pick one of this dataset's linked pages" is the
ordinary path for those, not a degraded one, and the link store is queried by
dataset as much as by checkpoint.

The Civitai side is the site's own tRPC + upload endpoints, authenticated by
the API key (Bearer, no cookie) — the same key the scraper and the 🌐 prompt
browser already read, so the app holds ONE Civitai credential. The call chain
(model.upsert → modelVersion.upsert → /api/upload → /api/upload/complete →
modelFile.upsert; /api/v1/image-upload → post.create → post.addImage) was
exercised end to end against the live site before being written here; the
publish mutations (`model.publish`, `post.update {publishedAt}`) follow the
site's own source and are optional at every step. `model.publish` is sent
WITHOUT a timestamp on purpose: the server stamps its own "now", and a PC
clock running ahead would otherwise file the page as *scheduled* instead of
published. A post's timestamp is taken from the server's `Date` header for
the same reason.

Privacy is a boundary, not a checkbox. Every image leaves as a fresh PNG with
no embedded metadata (the generation data travels EXPLICITLY in `meta`, never
as a ComfyUI workflow blob that can name local paths), every outgoing text is
run through the path/token redactors, and the checkpoint's own safetensors
metadata is scanned for a home path before a single byte is uploaded.

Uploads take minutes on a home uplink, so both flows run as background jobs
(`start_job` / `job_status`) the modal polls — the same shape hf_publish uses.
`publish_model` and `post_images` themselves are synchronous and are the seams
the tests drive, with the network replaced through `_transport`.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests
from PIL import Image

from .. import config as cfg
from ..config import LOCAL_USER
from ..utils.redact import redact_tokens, redact_user_paths
from ..version import APP_VERSION

logger = logging.getLogger(__name__)

# The API is one backend behind two domains (civitai.com and civitai.red serve
# the same account and database). Calls always go to civitai.com; the domain
# the user is SHOWN in links is a setting, because the mirror is where some
# people are signed in.
API_BASE = 'https://civitai.com'
LINK_HOSTS = ('civitai.com', 'civitai.red')
DEFAULT_LINK_HOST = 'civitai.com'

_UA = f'lora-dataset-studio/{APP_VERSION} (Civitai publisher)'
_TIMEOUT = (6.1, 60)          # (connect, read) for the JSON calls
_UPLOAD_TIMEOUT = (6.1, 600)  # one 25 MB part, or one image, over a home uplink
_CHUNK = 25 * 1024 * 1024     # Civitai's multipart part size
_PART_RETRIES = 3

GITHUB_URL = 'https://github.com/perfectgf/lora-dataset-studio'

# The Civitai `baseModel` strings this app can name for a training family. The
# site validates the value against its canonical list, so an unknown string is
# refused with a 400 rather than stored — which is why the modal offers a
# select over THESE and not a free text field. Civitai keeps a separate
# ecosystem per WEIGHTS, not per architecture ("a LoRA is trained against
# weights"): the FLUX.2 Klein base models this app trains on are the
# `-base` entries, the distilled ones are a different lineage. 'Other' is the
# honest catch-all.
CIVITAI_BASE_MODELS = (
    'ZImageTurbo', 'ZImageBase', 'Krea 2', 'SDXL 1.0', 'Pony', 'Illustrious',
    'NoobAI', 'Flux.1 D', 'Flux.1 Krea',
    'Flux.2 Klein 4B-base', 'Flux.2 Klein 9B-base', 'Flux.2 Klein 4B', 'Flux.2 Klein 9B',
    'Anima', 'Other',
)

# Licence defaults are Civitai's most permissive combination — the same the
# wizard pre-ticks — and the modal exposes each toggle.
_COMMERCIAL_ALL = ['Image', 'RentCivit', 'Rent', 'Sell']

_MODEL_FILE_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._ -]{0,120}\.safetensors$')
# The run tag `import_checkpoint` appends to a deployed name — `_rl<record>_v<N>`
# for a local run, `_rc<pod run>_v<N>` for a cloud one: internal ids that a
# public page has no use for.
_RUN_TAG_RE = re.compile(r'_r[lc]\d+_v\d+$', re.IGNORECASE)
# The zero-padded step block a save's name carries (`…_000002500…`), kept by
# the deployed copy (`lora_x_000002500_Krea-2-Raw_rc158_v1`): the one token
# that tells the numbered save from the step-less final on both sides.
_STEP_BLOCK_RE = re.compile(r'_(\d{6,})(?:[_.]|$)')


class CivitaiPublishError(Exception):
    """A structured, user-facing failure: `code` lets the UI branch (`no_key`,
    `auth`, `link_missing`…), `message` is the sentence shown as-is."""
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


# --- credential & hosts ------------------------------------------------------

def api_key():
    """The one Civitai credential (env > cookies dir > stored secret), through
    the prompt browser's resolver so the three Civitai features never disagree
    about which key they read."""
    from .civitai_browser import civitai_api_key
    return civitai_api_key()


def link_host() -> str:
    """The domain shown in every link this feature renders. Anything but the
    two known mirrors falls back to civitai.com — a typo in config.json must
    not produce links to nowhere."""
    host = str(cfg.get('civitai.link_host') or '').strip().lower()
    return host if host in LINK_HOSTS else DEFAULT_LINK_HOST


def model_url(model_id, version_id=None) -> str:
    url = f'https://{link_host()}/models/{int(model_id)}'
    if version_id:
        url += f'?modelVersionId={int(version_id)}'
    return url


def model_wizard_url(model_id) -> str:
    """A DRAFT has no public page (the pretty URL 404s until it is published);
    the wizard is the only address that opens it."""
    return f'https://{link_host()}/models/{int(model_id)}/wizard?step=1'


def post_url(post_id, published) -> str:
    base = f'https://{link_host()}/posts/{int(post_id)}'
    return base if published else base + '/edit'


# --- the network seam --------------------------------------------------------

def _transport(method, url, headers=None, data=None, json_body=None, timeout=_TIMEOUT):
    """(status, response headers, body bytes) — the ONE place bytes leave the
    process. Tests replace this function and nothing else; a network failure
    raises CivitaiPublishError('network')."""
    try:
        resp = requests.request(method, url, headers=headers, data=data,
                                json=json_body, timeout=timeout)
    except requests.RequestException as e:
        raise CivitaiPublishError(
            'network', 'Civitai did not answer - check your connection and try again.') from e
    return resp.status_code, {k.lower(): v for k, v in resp.headers.items()}, resp.content


def _headers(key, json_body=True):
    h = {'Authorization': f'Bearer {key}', 'User-Agent': _UA}
    if json_body:
        h['Content-Type'] = 'application/json'
    return h


def _decode(body):
    try:
        return json.loads(body.decode('utf-8')) if body else None
    except (ValueError, UnicodeDecodeError):
        return None


# Civitai's own clock, read off the `Date` header of the last answer, so a
# publish timestamp never depends on the PC's clock (a machine running ahead
# would file a post in the future).
_server_clock = {'date': None, 'at': 0.0}


def _note_server_date(headers):
    raw = (headers or {}).get('date')
    if not raw:
        return
    try:
        _server_clock['date'] = parsedate_to_datetime(raw)
        _server_clock['at'] = time.monotonic()
    except (TypeError, ValueError):
        pass


def _publish_stamp_iso() -> str:
    """A `publishedAt` that is "now" by Civitai's clock, minus a small margin:
    the site treats a timestamp in its future as *scheduled*."""
    base = _server_clock['date']
    if base is not None:
        now = base + timedelta(seconds=time.monotonic() - _server_clock['at'])
    else:
        now = datetime.now(timezone.utc)
    return (now - timedelta(seconds=15)).astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def _refuse(status, parsed, what):
    """Turn a non-200 into the sentence the user can act on."""
    if status == 401:
        raise CivitaiPublishError(
            'auth', 'Civitai refused the API key - check it in Settings > Scraping & sources.')
    if status == 403:
        raise CivitaiPublishError(
            'forbidden', f'Civitai refused this operation ({what}): the API key lacks the '
                         'permission it needs (a scoped key must allow "Media write" for posts '
                         'and "Models write" for model pages). A key created at '
                         'civitai.com/user/account with full access works.')
    detail = ''
    if isinstance(parsed, dict):
        try:
            detail = parsed['error']['json']['message']
        except (KeyError, TypeError):
            detail = parsed.get('error') or parsed.get('message') or ''
    detail = redact_tokens(str(detail))[:300]
    raise CivitaiPublishError(
        'civitai', f'Civitai answered HTTP {status} on {what}' + (f': {detail}' if detail else '.'))


def _post_json(path, obj, key, what):
    st, headers, body = _transport('POST', API_BASE + path, headers=_headers(key), json_body=obj)
    _note_server_date(headers)
    parsed = _decode(body)
    if st != 200:
        _refuse(st, parsed, what)
    return parsed


def _get_json(url, key, what):
    st, headers, body = _transport('GET', url, headers=_headers(key, json_body=False))
    _note_server_date(headers)
    parsed = _decode(body)
    if st != 200:
        _refuse(st, parsed, what)
    return parsed


# --- tRPC envelopes ----------------------------------------------------------
# Two response encodings coexist on the site: superjson (`{"json": value}`,
# every mutation this module sends) and devalue (a flattened, self-referencing
# array, used by the GET queries). Both are decoded here so a caller never
# sees the difference.

def _devalue_parse(arr):
    special = {-1: None, -2: None, -3: float('nan'), -4: math.inf, -5: -math.inf, -6: -0.0}
    hydrated = {}

    def hydrate(i):
        if isinstance(i, int) and i in special:
            return special[i]
        if i in hydrated:
            return hydrated[i]
        v = arr[i]
        if isinstance(v, list):
            if v and isinstance(v[0], str):
                tag = v[0]
                if tag == 'Date':
                    hydrated[i] = v[1]
                    return v[1]
                if tag == 'Set':
                    out = []
                    hydrated[i] = out
                    out.extend(hydrate(j) for j in v[1:])
                    return out
                if tag == 'Map':
                    out = {}
                    hydrated[i] = out
                    for k in range(1, len(v), 2):
                        out[hydrate(v[k])] = hydrate(v[k + 1])
                    return out
                if tag == 'BigInt':
                    hydrated[i] = int(v[1])
                    return hydrated[i]
                if tag == 'null':
                    hydrated[i] = None
                    return None
            out = []
            hydrated[i] = out
            out.extend(hydrate(j) for j in v)
            return out
        if isinstance(v, dict):
            out = {}
            hydrated[i] = out
            for k, j in v.items():
                out[k] = hydrate(j)
            return out
        hydrated[i] = v
        return v

    return hydrate(0)


def _unwrap_trpc(data):
    if isinstance(data, dict) and 'json' in data:
        return data['json']
    if isinstance(data, str):
        return _devalue_parse(json.loads(data))
    return data


def _superjson(input_obj, date_fields=()):
    """A superjson body. Dates travel as ISO strings plus a `meta` block naming
    them — without it a `z.date()` field arrives as a plain string and fails."""
    body = {'json': input_obj}
    present = [f for f in date_fields if f in input_obj]
    if present:
        body['meta'] = {'values': {f: ['Date'] for f in present}}
    return body


def trpc_mutation(proc, input_obj, key, date_fields=()):
    parsed = _post_json(f'/api/trpc/{proc}', _superjson(input_obj, date_fields), key, proc)
    try:
        return _unwrap_trpc(parsed['result']['data'])
    except (KeyError, TypeError, ValueError) as e:
        raise CivitaiPublishError(
            'civitai', f'Civitai answered an unexpected shape on {proc}.') from e


def trpc_query(proc, input_obj, key):
    from urllib.parse import quote
    payload = quote(json.dumps({'json': input_obj}))
    parsed = _get_json(f'{API_BASE}/api/trpc/{proc}?input={payload}', key, proc)
    try:
        return _unwrap_trpc(parsed['result']['data'])
    except (KeyError, TypeError, ValueError) as e:
        raise CivitaiPublishError(
            'civitai', f'Civitai answered an unexpected shape on {proc}.') from e


# --- account ------------------------------------------------------------------

_who_cache = {'key': None, 'at': 0.0, 'value': None}
_WHO_TTL = 300


def whoami(key):
    """The key owner's username, or None when the key is missing/refused/
    unreachable — best-effort, cached, never raises: it only decorates the
    modal's header."""
    if not key:
        return None
    now = time.monotonic()
    if _who_cache['key'] == key and now - _who_cache['at'] < _WHO_TTL:
        return _who_cache['value']
    try:
        me = _get_json(f'{API_BASE}/api/v1/me', key, 'me')
        value = (me or {}).get('username') or None
    except CivitaiPublishError:
        value = None
    _who_cache.update(key=key, at=now, value=value)
    return value


# --- the model page (an existing one, by URL) ---------------------------------

_MODEL_REF_RE = re.compile(
    r'(?:^|/models/)(\d+)(?:[^?\s]*)?(?:\?[^\s]*?modelVersionId=(\d+))?', re.IGNORECASE)


def parse_model_ref(text):
    """`(model_id, version_id|None)` out of a pasted model page URL — with or
    without the slug, on either domain, with an optional `?modelVersionId=` —
    or out of a bare numeric id. None when nothing looks like a model."""
    s = str(text or '').strip()
    if not s:
        return None
    if s.isdigit():
        return int(s), None
    if 'civitai.' not in s.lower() and '/models/' not in s:
        return None
    m = _MODEL_REF_RE.search(s)
    if not m:
        return None
    return int(m.group(1)), (int(m.group(2)) if m.group(2) else None)


def fetch_model_page(model_id, key):
    """What the modal needs to confirm a pasted page: its name, type, nsfw flag
    and the versions to pick from. The site's own query answers for the owner's
    DRAFTS too, which the public REST listing never lists; that listing is the
    fallback when the query is unavailable."""
    page = None
    try:
        page = trpc_query('model.getById', {'id': int(model_id)}, key)
    except CivitaiPublishError as e:
        if e.code in ('auth', 'forbidden'):
            raise
        logger.debug('model.getById failed for %s, trying the REST listing', model_id,
                     exc_info=True)
    if not isinstance(page, dict) or not page.get('id'):
        page = _get_json(f'{API_BASE}/api/v1/models/{int(model_id)}', key, 'model')
    if not isinstance(page, dict) or not page.get('id'):
        raise CivitaiPublishError('not_found', 'Civitai knows no model with that id.')
    versions = []
    for v in page.get('modelVersions') or []:
        if isinstance(v, dict) and v.get('id'):
            versions.append({'id': int(v['id']), 'name': str(v.get('name') or ''),
                             'base_model': v.get('baseModel'),
                             'status': v.get('status')})
    return {'id': int(page['id']), 'name': str(page.get('name') or ''),
            'type': page.get('type'), 'nsfw': bool(page.get('nsfw')),
            'status': page.get('status'), 'versions': versions}


# --- the checkpoint on disk ---------------------------------------------------

def _record(record_id):
    from ..extensions import db
    from ..models import TrainingRunRecord
    rec = db.session.get(TrainingRunRecord, int(record_id))
    if rec is None:
        raise CivitaiPublishError('run_missing', 'This training run is no longer registered.')
    return rec


def _step_block(name) -> str | None:
    """The zero-padded step a save's name carries, or None for a step-less
    final. A deployed name keeps its source's block, so this is what tells the
    two files of one step apart on both sides."""
    m = _STEP_BLOCK_RE.search(str(name or ''))
    return m.group(1) if m else None


def _run_saves(rec) -> list:
    """Every save of this run on this machine, `[{filename, step, path|None}]`,
    resolved the way the graph resolves them (never from a client-sent path):
    a cloud run's saves live in the checkpoint store, a local run's in
    ai-toolkit's run folder. Raises `checkpoint_missing` when they cannot be
    listed at all."""
    from ..extensions import db
    from ..models import CloudTrainingRun
    if rec.source == 'cloud' and rec.cloud_run_id:
        from . import cloud_training as ct
        from . import video_training
        run = db.session.get(CloudTrainingRun, int(rec.cloud_run_id))
        if run is None:
            raise CivitaiPublishError('checkpoint_missing',
                                      'This cloud run is not linked on this machine.')
        out = []
        for name, path in ct.run_checkpoint_files(run).items():
            saved_step, _stage = video_training.split_checkpoint_name(name)
            out.append({'filename': name, 'path': path,
                        'step': saved_step if saved_step is not None
                        else int(rec.steps or 0)})
        return out
    from . import lora_training as lt
    try:
        cks = lt.list_checkpoints(LOCAL_USER, rec.dataset_id, rec.base_model or '',
                                  rec.family, rec.variant)
    except Exception as e:  # ai-toolkit not configured, folder gone…
        raise CivitaiPublishError(
            'checkpoint_missing', f'The saves of this run cannot be listed: {e}') from e
    return [{'filename': c['filename'], 'step': int(c.get('step') or -1), 'path': None}
            for c in cks if c.get('run_source') == 'local' and c.get('run_id') == rec.id]


def _pick_save(candidates, step, filename=None, hint=None):
    """ONE save out of the run's files: by name when the caller names it (a
    pill always can); else by step — and when two files share the step (the
    numbered save and the final of a run that ended on it) the `hint` decides:
    the deployed LoRA name a picture ran with carries the numbered save's step
    block, or none for the final. Without a hint that can decide, the step is
    refused rather than guessed."""
    if filename:
        pick = next((c for c in candidates if c['filename'] == filename), None)
        if pick is None:
            raise CivitaiPublishError(
                'checkpoint_missing', f'{filename} is not a save of this run on this machine.')
        return pick
    at_step = [c for c in candidates if c['step'] == step]
    if not at_step:
        raise CivitaiPublishError('checkpoint_missing',
                                  f'No save of this run at step {step} is on this machine.')
    if len(at_step) > 1 and hint:
        block = _step_block(hint)
        same = [c for c in at_step if _step_block(c['filename']) == block]
        if len(same) == 1:
            return same[0]
    if len(at_step) > 1:
        names = ', '.join(c['filename'] for c in at_step)
        raise CivitaiPublishError(
            'ambiguous', f'Two saves of this run share step {step} ({names}) - open the '
                         'checkpoint\'s own pill on the Canvas or the run graph to name the file.')
    return at_step[0]


def checkpoint_file_for(record_id, step, filename=None, hint=None):
    """The `.safetensors` a checkpoint stands for — by file name, or by step
    with the deployed name as the tie-breaker. The file must be one the run
    really produced. Returns (path, record)."""
    rec = _record(record_id)
    pick = _pick_save(_run_saves(rec), int(step), filename, hint)
    path = pick.get('path')
    if not path:
        from . import lora_training as lt
        path = lt.checkpoint_file_path(LOCAL_USER, rec.dataset_id, pick['filename'],
                                       rec.base_model or '', rec.family, rec.variant)
    if not path:
        raise CivitaiPublishError('checkpoint_missing', 'This save is no longer on disk.')
    return path, rec


def resolve_save_filename(rec, step, hint=None):
    """The name of the run's save at `step`, for a caller that has none (a
    picture): the listing decides, the hint breaks a shared step. None when
    the saves cannot be listed or none sits at that step — marking a page
    needs no file on disk, only a name to remember it by, and '' is the honest
    name when there is none. A shared step the hint cannot settle stays a
    refusal: guessing which file would be a lie written into the store."""
    try:
        saves = _run_saves(rec)
    except CivitaiPublishError:
        return None
    try:
        return _pick_save(saves, int(step), None, hint)['filename']
    except CivitaiPublishError as e:
        if e.code == 'ambiguous':
            raise
        return None


_HOME_RE = re.compile(r'[A-Za-z]:\\{1,2}Users\\{1,2}[^\\/:*?"<>|\r\n]+|/(?:home|Users)/[^/\r\n]+',
                      re.IGNORECASE)
_EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
_DTYPE_FP = {'F16': 'fp16', 'BF16': 'bf16', 'F32': 'fp32'}


def inspect_checkpoint(path) -> dict:
    """Header-only look at the file about to be uploaded: is it a safetensors
    container, how many tensors, which precision, and does its own metadata
    carry anything that names the machine (a home path, an email). The
    multi-hundred-MB body is never read."""
    from .fp8_export import Fp8ExportError, read_header
    try:
        header = read_header(path)
    except Fp8ExportError as e:
        raise CivitaiPublishError('not_safetensors', str(e)) from e
    meta = header.get('__metadata__') or {}
    counts = {}
    for k, v in header.items():
        if k != '__metadata__' and isinstance(v, dict):
            counts[v.get('dtype')] = counts.get(v.get('dtype'), 0) + 1
    fp = 'fp16'
    if counts:
        top = max(counts, key=counts.get)
        fp = _DTYPE_FP.get(top, 'fp16')
    leaks = sorted(k for k, v in meta.items()
                   if _HOME_RE.search(str(v)) or _EMAIL_RE.search(str(v)))
    return {'size': os.path.getsize(path), 'tensors': sum(counts.values()),
            'fp': fp, 'leaks': leaks,
            'software': _meta_json(meta.get('software')).get('name'),
            'epoch': _meta_json(meta.get('training_info')).get('epoch')}


def _meta_json(value) -> dict:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value) if isinstance(value, str) else None
    except ValueError:
        parsed = None
    return parsed if isinstance(parsed, dict) else {}


# --- naming a training family the way Civitai does -----------------------------

def civitai_base_model(family, variant=None, base_model='') -> str:
    """The Civitai `baseModel` for an LDS training family + variant, or '' when
    the app cannot honestly say — and an empty prefill is what forces the
    choice in the form.

    Civitai keeps one ecosystem per WEIGHTS: FLUX.2 Klein trained on the
    official `FLUX.2-klein-base-*` weights is `Flux.2 Klein *-base`, not the
    distilled lineage; an SDXL LoRA trained on a Pony or Illustrious finetune
    belongs to that lineage, not to 'SDXL 1.0'. So a CUSTOM base (a ComfyUI
    checkpoint picked in the dropdown) leaves the answer open wherever the
    site distinguishes lineages — sdxl, zimage, flux2klein — and the user
    names it. Krea 2 and FLUX.1 have a single lineage on the site."""
    fam = str(family or '').strip().lower()
    var = str(variant or '').strip().lower()
    custom = bool(str(base_model or '').strip())
    if fam == 'zimage':
        if custom:
            return ''
        return 'ZImageBase' if var == 'base' else 'ZImageTurbo'
    if fam == 'krea':
        return 'Krea 2'
    if fam == 'sdxl':
        return '' if custom else 'SDXL 1.0'
    if fam == 'flux':
        return 'Flux.1 D'
    if fam == 'flux2klein':
        if custom:
            return ''
        return 'Flux.2 Klein 9B-base' if var == '9b' else 'Flux.2 Klein 4B-base'
    if fam == 'anima':
        return 'Anima'
    return 'Other'


def _family_label(family) -> str:
    from .lora_training import _FAMILY_LABEL
    return _FAMILY_LABEL.get(str(family or '').lower(), str(family or 'LoRA'))


def _kind_label(ds) -> str:
    kind = str(getattr(ds, 'kind', None) or '').lower()
    return {'concept': 'Concept', 'style': 'Style'}.get(kind, 'Character')


def _slug(s) -> str:
    s = re.sub(r'[^A-Za-z0-9._-]+', '-', (s or '').strip())
    s = re.sub(r'-{2,}', '-', s).strip('-._')
    return s or 'lora'


def _clean_stem(s) -> str:
    """A file stem reduced to what a human reads: no folder, no extension."""
    tail = str(s or '').replace('\\', '/').split('/')[-1]
    return re.sub(r'\.(safetensors|ckpt|pt|sft)$', '', tail, flags=re.IGNORECASE)


def _public_stem(s) -> str:
    """The stem without the app's deploy tag (`_rl<record>_v<N>` /
    `_rc<run>_v<N>`): internal ids have no business on a public page."""
    return _RUN_TAG_RE.sub('', _clean_stem(s))


def _manifest_count(rec):
    try:
        manifest = json.loads(rec.manifest or 'null')
    except (TypeError, ValueError):
        return None
    return len(manifest) if isinstance(manifest, list) else None


def build_description(ds, rec, step, inspection=None) -> str:
    """The model page's description, as simple HTML: what the LoRA is, the
    trigger to use, the training facts the run recorded. Redacted — a run note
    or a base-model path can carry a home directory."""
    from .checkpoint_registry import network_geometry
    fam = _family_label(rec.family)
    kind = _kind_label(ds)
    trigger = (getattr(ds, 'trigger_word', '') or '').strip()
    facts = [f'{fam} base', f'{int(step):,} steps'.replace(',', ' ')]
    epoch = (inspection or {}).get('epoch')
    if epoch:
        facts.append(f'{int(epoch)} epochs')
    geo = network_geometry(rec)
    if geo.get('rank'):
        facts.append(f'rank {geo["rank"]}' + (f' / alpha {geo["alpha"]}' if geo.get('alpha') else ''))
    n = _manifest_count(rec)
    if n:
        facts.append(f'{n} training images (dataset v{rec.version})')
    if getattr(rec, 'masked', None):
        facts.append('masked training')
    parts = [f'<p>{kind} LoRA for <strong>{fam}</strong>, trained with '
             f'<a href="{GITHUB_URL}">LoRA Dataset Studio</a>.</p>']
    if trigger:
        parts.append(f'<p>Trigger word: <code>{_html(trigger)}</code></p>')
    parts.append('<p>' + ' · '.join(_html(f) for f in facts) + '</p>')
    return redact_user_paths(''.join(parts))


def _html(s) -> str:
    return (str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            .replace('"', '&quot;'))


def default_tags(ds, rec) -> list:
    tags = [_kind_label(ds).lower(), _family_label(rec.family).lower()]
    subject = str(getattr(ds, 'subject_type', None) or '').lower()
    if subject and subject not in ('human',):
        tags.append(subject)
    return tags


def draft_defaults(record_id, step, filename=None, hint=None) -> dict:
    """Everything the "create a model page" form is pre-filled with, derived
    from the dataset and the run — plus the file's own facts, or the reason it
    cannot be uploaded (said, never a dead Publish button). `hint` is the
    deployed name a picture ran with, for the image door that has no file
    name of its own."""
    from . import face_dataset_service as fds
    rec = _record(record_id)
    ds = fds.get_dataset(LOCAL_USER, rec.dataset_id)
    if ds is None:
        raise CivitaiPublishError('dataset_missing', 'The dataset of this run is gone.')
    base = civitai_base_model(rec.family, rec.variant, rec.base_model)
    hint_text = None
    if not base:
        hint_text = (f'Trained on a custom base ({_public_stem(rec.base_model)}): pick the lineage '
                     'Civitai files it under (Pony, Illustrious, NoobAI, a Klein base…).')
    out = {
        'record_id': rec.id, 'step': int(step), 'dataset_id': rec.dataset_id,
        'filename': filename,
        'name': f'{ds.name} ({_family_label(rec.family)})',
        'version_name': f'v{rec.version} · step {int(step):,}'.replace(',', ' '),
        'base_model': base,
        'base_model_hint': hint_text,
        'base_model_choices': list(CIVITAI_BASE_MODELS),
        'trained_words': [w for w in [(ds.trigger_word or '').strip()] if w],
        'tags': default_tags(ds, rec),
        'nsfw': False,
        'steps': int(step),
        'file': None, 'file_error': None,
        'link': None,
    }
    try:
        path, _rec = checkpoint_file_for(rec.id, step, filename, hint)
        info = inspect_checkpoint(path)
        out['filename'] = os.path.basename(path)
        out['file'] = {
            'name': f'{_slug(ds.name)}_{_slug(_family_label(rec.family)).lower()}'
                    f'_v{rec.version}_step{int(step)}.safetensors',
            'source': os.path.basename(path),
            'size_mb': round(info['size'] / 1e6, 1), 'fp': info['fp'],
            'epoch': info['epoch'], 'leaks': info['leaks'],
        }
        out['description'] = build_description(ds, rec, step, info)
        if info['leaks']:
            out['file_error'] = ('the file\'s own metadata names this machine ('
                                 + ', '.join(info['leaks']) + ') - it cannot be uploaded as is')
    except CivitaiPublishError as e:
        out['file_error'] = e.message
        out['description'] = build_description(ds, rec, step)
    out['link'] = link_payload(link_for(rec.id, int(step), out['filename'] or None, hint))
    return out


# --- the link store --------------------------------------------------------------

def link_payload(link):
    if link is None:
        return None
    return {
        'id': link.id, 'record_id': link.record_id, 'step': link.step,
        'filename': link.filename or '',
        'dataset_id': link.dataset_id, 'model_id': link.model_id,
        'version_id': link.version_id, 'model_name': link.model_name,
        'version_name': link.version_name, 'base_model': link.base_model,
        'published': link.published,
        'model_url': model_url(link.model_id, link.version_id),
        'wizard_url': model_wizard_url(link.model_id),
    }


def _prefer(candidates, stem=None):
    """Among the links of ONE step, the save a picture was made with: the one
    whose file carries the SAME step block as the picture's deployed LoRA
    name; a name without a block (it was made with the final) prefers the
    step-less link. Without a stem (a bare step lookup): the numbered save's —
    a picture that carries a step was made with a numbered save — else the
    first. A prefix test cannot do this: the final's stem is a prefix of every
    numbered save's."""
    if not candidates:
        return None
    if stem:
        block = _step_block(stem)
        for c in candidates:
            if _step_block(c.filename) == block:
                return c
    for c in candidates:
        if _step_block(c.filename):
            return c
    return candidates[0]


def link_for(record_id, step, filename=None, hint=None):
    """The link of ONE save: by name when the caller names it (a pill always
    does), else the step's preferred one — the deployed name a picture ran
    with (`hint`) breaking a shared step (see `_prefer`)."""
    from ..models import CivitaiLink
    if record_id is None:
        return None
    q = CivitaiLink.query.filter_by(record_id=int(record_id), step=int(step))
    if filename:
        return q.filter_by(filename=filename).first()
    return _prefer(q.order_by(CivitaiLink.id.asc()).all(), stem=_clean_stem(hint) if hint else None)


def link_for_image(row):
    """The page a generated picture belongs to: the links of the checkpoint
    stamped on its row, the one whose file its deployed name was made from
    first. None when the row carries no stamp — the caller then offers the
    dataset's links, which is the ORDINARY path for a final save's pictures."""
    if row is None or row.record_id is None or row.step is None:
        return None
    return link_for(row.record_id, row.step, hint=row.checkpoint)


def links_for_dataset(dataset_id) -> list:
    from ..models import CivitaiLink
    rows = (CivitaiLink.query.filter_by(dataset_id=int(dataset_id))
            .order_by(CivitaiLink.record_id.desc(), CivitaiLink.step.desc(),
                      CivitaiLink.id.desc()).all())
    return [link_payload(r) for r in rows]


def links_for_record(record_id) -> dict:
    """{filename: link payload} — what the lineage payload stamps on each pill
    so the popover can say "on Civitai" without a request per pill."""
    from ..models import CivitaiLink
    return {(r.filename or ''): link_payload(r) for r in
            CivitaiLink.query.filter_by(record_id=int(record_id)).all()}


def save_link(record_id, step, filename, dataset_id, *, model_id, version_id,
              model_name=None, version_name=None, base_model=None, published=None):
    """Upsert the link of one save. A save IS one Civitai version; linking it
    again simply retargets it. `filename` '' is a save whose file could not
    be named (its run's saves are not on this machine) — one such row per
    step.

    `model_name` defaults to None, not '': the job-harvest guard
    (tests/test_dataset_job_harvest.py) reads every `model_name=<str>` default
    in this package as an engine stamping a job name, and '' read as an
    engine the dispatch does not route. It is a Civitai page's name here."""
    from ..extensions import db
    from ..models import CivitaiLink
    filename = str(filename or '')
    row = (CivitaiLink.query.filter_by(record_id=int(record_id), step=int(step),
                                       filename=filename).first())
    if row is None:
        row = CivitaiLink(record_id=int(record_id), step=int(step), filename=filename,
                          dataset_id=int(dataset_id))
        db.session.add(row)
    row.dataset_id = int(dataset_id)
    row.model_id = int(model_id)
    row.version_id = int(version_id)
    row.model_name = (model_name or '')[:255]
    row.version_name = (version_name or '')[:255]
    row.base_model = (base_model or None)
    row.published = published
    db.session.commit()
    return row


def delete_link(link_id) -> bool:
    from ..extensions import db
    from ..models import CivitaiLink
    row = db.session.get(CivitaiLink, int(link_id))
    if row is None:
        return False
    db.session.delete(row)
    db.session.commit()
    return True


def detach_links_of_run(record_id) -> int:
    """A run leaves the lineage graph: its links lose their record (the page
    on Civitai still exists) but keep their dataset, so the dataset's pictures
    can still be posted under them. Called inside the run's own transaction."""
    from ..models import CivitaiLink
    return int(CivitaiLink.query.filter_by(record_id=int(record_id))
               .update({'record_id': None}, synchronize_session=False))


def lookup_model_page(ref, key):
    """A pasted address or id → the page it names, plus the version the
    address itself points at (`?modelVersionId=`), or None. The modal shows
    this BEFORE linking, so the version is picked from a list rather than
    typed: a wrong id in a pasted address used to be the only way to learn
    what the page's versions were called."""
    parsed = parse_model_ref(ref)
    if not parsed:
        raise CivitaiPublishError(
            'bad_ref', 'Paste the address of a Civitai model page '
                       '(https://civitai.com/models/12345/...) or its numeric id.')
    model_id, url_version = parsed
    page = fetch_model_page(model_id, key)
    if page.get('type') and str(page['type']).upper() not in ('LORA', 'LOCON', 'DORA', 'LYCORIS'):
        raise CivitaiPublishError(
            'not_a_lora', f'That page is a {page["type"]}, not a LoRA - a checkpoint of this '
                          'app is a LoRA adapter.')
    if not page['versions']:
        raise CivitaiPublishError('no_version', 'That model page has no version to link to yet.')
    return page, url_version


def _versions_line(page) -> str:
    return ', '.join(f'{v["name"] or "unnamed"} (#{v["id"]})' for v in page['versions'][:12])


def link_checkpoint_to_page(record_id, step, ref, key, filename=None, version_id=None,
                            hint=None):
    """"Mark the page": resolve a pasted model URL/id against Civitai, pick the
    version (the caller's, the URL's, else the newest), and remember it for
    this save. A caller without a file name (a picture) gets it resolved from
    the run's saves, the deployed name it ran with breaking a shared step; a
    run whose saves are not on this machine is remembered under '' — marking
    needs no file, only a page. Returns (link, page)."""
    rec = _record(record_id)
    if not str(filename or '').strip():
        filename = resolve_save_filename(rec, step, hint) or ''
    page, url_version = lookup_model_page(ref, key)
    wanted = version_id or url_version
    version = None
    if wanted:
        version = next((v for v in page['versions'] if v['id'] == int(wanted)), None)
        if version is None:
            raise CivitaiPublishError(
                'no_version', f'Version {wanted} is not one of that model\'s versions - it has: '
                              f'{_versions_line(page)}. Look the page up and pick one.')
    else:
        version = page['versions'][0]
    link = save_link(record_id, step, filename, rec.dataset_id, model_id=page['id'],
                     version_id=version['id'], model_name=page['name'],
                     version_name=version['name'], base_model=version.get('base_model'),
                     published=(str(page.get('status') or '').lower() == 'published') or None)
    return link, page


# --- the model page (a new one, from a checkpoint) ------------------------------

def _validate_model_form(form) -> dict:
    name = redact_user_paths(str(form.get('name') or '').strip())
    if not name:
        raise CivitaiPublishError('invalid', 'The model needs a name.')
    base = str(form.get('base_model') or '').strip()
    if not base:
        raise CivitaiPublishError('invalid', 'Pick the base model as Civitai names it.')
    if base not in CIVITAI_BASE_MODELS:
        raise CivitaiPublishError(
            'invalid', f'"{base}" is not a base model this app can name on Civitai.')
    words = form.get('trained_words')
    if isinstance(words, str):
        words = [w.strip() for w in words.split(',')]
    words = [redact_user_paths(str(w).strip()) for w in (words or []) if str(w).strip()]
    tags = form.get('tags')
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(',')]
    tags = [redact_user_paths(str(t).strip().lower())[:60] for t in (tags or []) if str(t).strip()]
    file_name = str(form.get('file_name') or '').strip()
    if file_name and not _MODEL_FILE_RE.match(file_name):
        raise CivitaiPublishError(
            'invalid', 'The file name must end in .safetensors and use plain characters.')
    lic = form.get('license') if isinstance(form.get('license'), dict) else {}
    return {
        'name': name[:255],
        'version_name': redact_user_paths(str(form.get('version_name') or 'v1.0').strip())[:255] or 'v1.0',
        'description': redact_tokens(redact_user_paths(str(form.get('description') or '')))[:20000],
        'base_model': base,
        'trained_words': words[:20],
        'tags': tags[:30],
        'nsfw': bool(form.get('nsfw')),
        'file_name': file_name or None,
        'publish': bool(form.get('publish')),
        'license': {
            'allowNoCredit': bool(lic.get('allowNoCredit', True)),
            'allowCommercialUse': list(_COMMERCIAL_ALL) if lic.get('allowCommercialUse', True) else [],
            'allowDerivatives': bool(lic.get('allowDerivatives', True)),
            'allowDifferentLicense': bool(lic.get('allowDifferentLicense', True)),
        },
    }


def _upload_model_file(path, key, upload_name, progress=None):
    """Multipart upload of the checkpoint: ask for the part URLs, PUT each 25 MB
    part (retried), complete. Returns the storage URL the file must be
    registered under (the FULL location the site answers, not the key)."""
    size = os.path.getsize(path)
    up = _post_json('/api/upload', {'filename': upload_name, 'type': 'Model', 'size': size},
                    key, 'upload')
    urls = (up or {}).get('urls')
    if not isinstance(urls, list) or not urls:
        raise CivitaiPublishError('civitai', 'Civitai gave no upload slots for the file.')
    parts = []
    with open(path, 'rb') as fh:
        for i, u in enumerate(urls):
            chunk = fh.read(_CHUNK)
            etag = _put_part(u['url'], chunk)
            parts.append({'ETag': etag, 'PartNumber': u['partNumber']})
            if progress:
                progress('uploading', (i + 1) / len(urls))
    comp = _post_json('/api/upload/complete', {
        'bucket': up['bucket'], 'key': up['key'], 'type': 'Model',
        'uploadId': up['uploadId'], 'parts': parts}, key, 'upload/complete')
    location = comp if isinstance(comp, str) else (comp or {}).get('Location')
    if not location:
        raise CivitaiPublishError('civitai', 'Civitai did not confirm where the file landed.')
    return location, size / 1024.0


def _put_part(url, chunk):
    last = None
    for attempt in range(_PART_RETRIES):
        try:
            st, headers, _body = _transport('PUT', url, data=chunk, timeout=_UPLOAD_TIMEOUT)
            etag = headers.get('etag')
            if st in (200, 204) and etag:
                return etag
            last = CivitaiPublishError('civitai', f'a file part was refused (HTTP {st})')
        except CivitaiPublishError as e:
            last = e
        time.sleep(2 * (attempt + 1))
    raise last


def publish_model(record_id, step, form, key, filename=None, hint=None, progress=None,
                  user_id=LOCAL_USER):
    """Create the model page for one checkpoint — synchronous, the tested seam.

    model.upsert (Draft) → modelVersion.upsert → the file → modelFile.upsert →
    optionally model.publish → the link is saved. Returns the link payload plus
    the addresses to open."""
    from . import face_dataset_service as fds
    if not key:
        raise CivitaiPublishError(
            'no_key', 'No Civitai API key configured - paste one in Settings > Scraping & sources.')
    spec = _validate_model_form(form)
    path, rec = checkpoint_file_for(record_id, step, filename, hint)
    ds = fds.get_dataset(user_id, rec.dataset_id)
    if ds is None:
        raise CivitaiPublishError('dataset_missing', 'The dataset of this run is gone.')
    info = inspect_checkpoint(path)
    if info['leaks']:
        raise CivitaiPublishError(
            'file_metadata_leak',
            'The checkpoint\'s own metadata names this machine ('
            + ', '.join(info['leaks']) + ') - it was not uploaded.')
    if progress:
        progress('creating', 0.0)
    lic = spec['license']
    model = trpc_mutation('model.upsert', {
        'name': spec['name'], 'type': 'LORA', 'uploadType': 'Created', 'status': 'Draft',
        'description': spec['description'] or None, 'nsfw': spec['nsfw'], 'poi': False,
        'minor': False, 'sfwOnly': False,
        'allowNoCredit': lic['allowNoCredit'], 'allowCommercialUse': lic['allowCommercialUse'],
        'allowDerivatives': lic['allowDerivatives'],
        'allowDifferentLicense': lic['allowDifferentLicense'],
        'tagsOnModels': [{'name': t} for t in spec['tags']],
    }, key)
    model_id = int(model['id'])
    version = trpc_mutation('modelVersion.upsert', {
        'modelId': model_id, 'name': spec['version_name'], 'baseModel': spec['base_model'],
        'trainedWords': spec['trained_words'], 'steps': int(step),
        **({'epochs': int(info['epoch'])} if info.get('epoch') else {}),
    }, key)
    version_id = int(version['id'])
    upload_name = spec['file_name'] or os.path.basename(path)
    location, size_kb = _upload_model_file(path, key, upload_name, progress)
    if progress:
        progress('registering', 1.0)
    trpc_mutation('modelFile.upsert', {
        'name': upload_name, 'url': location, 'sizeKB': size_kb, 'type': 'Model',
        'modelVersionId': version_id,
        'metadata': {'format': 'SafeTensor', 'size': 'full', 'fp': info['fp']},
    }, key)
    published = False
    if spec['publish']:
        # No `publishedAt` on purpose: the server stamps its own "now". Sent
        # from a PC clock running ahead, the page would be filed as SCHEDULED.
        trpc_mutation('model.publish', {'id': model_id, 'versionIds': [version_id]}, key)
        published = True
    link = save_link(rec.id, step, os.path.basename(path), rec.dataset_id,
                     model_id=model_id, version_id=version_id,
                     model_name=spec['name'], version_name=spec['version_name'],
                     base_model=spec['base_model'], published=published)
    return {'link': link_payload(link), 'model_id': model_id, 'version_id': version_id,
            'published': published,
            'url': model_url(model_id, version_id) if published else model_wizard_url(model_id)}


# --- images → a post on the linked version ----------------------------------------

def _image_path(row) -> str:
    from . import face_dataset_service as fds
    return os.path.join(fds._dataset_path(row.dataset_id), row.filename or '')


def _prompt_as_generated(row, ds) -> str:
    """The prompt the workflow actually ran: the stored cell prompt is RAW and
    the trigger is prefixed at generation unless the box was unticked."""
    prompt = (row.prompt or '').strip()
    if row.inject_trigger is False or ds is None:
        return prompt
    from .lora_test_studio import _prompt_with_trigger
    return _prompt_with_trigger(prompt, (ds.trigger_word or '').strip())


def _extra_loras(row) -> list:
    try:
        entries = json.loads(row.extra_loras) if row.extra_loras else []
    except (TypeError, ValueError):
        return []
    return [e for e in entries if isinstance(e, dict) and e.get('filename')]


def image_meta(row, ds, link, width, height) -> dict:
    """The generation data Civitai shows under the image — every value a
    column the cell already persisted, so what is published is what ran.
    `civitaiResources` is what files the picture under the model on the site
    (its resource detection reads `modelVersionId` there, no hash needed);
    `resources` is the human-readable list."""
    meta = {'prompt': _prompt_as_generated(row, ds)}
    if row.negative:
        meta['negativePrompt'] = row.negative
    if row.cfg is not None:
        meta['cfgScale'] = row.cfg
    if row.steps is not None:
        meta['steps'] = row.steps
    if row.sampler:
        meta['sampler'] = row.sampler
    if row.scheduler:
        meta['scheduler'] = row.scheduler
    if row.seed is not None:
        meta['seed'] = int(row.seed)
    if width and height:
        meta['Size'] = f'{int(width)}x{int(height)}'
    if link is not None and link.base_model:
        meta['baseModel'] = link.base_model
    if row.z_model:
        meta['Model'] = _public_stem(row.z_model)
    lora_name = (link.model_name if link is not None and link.model_name
                 else _public_stem(row.checkpoint))
    resources = [{'type': 'lora', 'name': lora_name, 'weight': row.strength}]
    for e in _extra_loras(row):
        try:
            weight = float(e.get('strength'))
        except (TypeError, ValueError):
            weight = None
        entry = {'type': 'lora', 'name': _public_stem(e['filename'])}
        if weight is not None:
            entry['weight'] = weight
        resources.append(entry)
    meta['resources'] = resources
    if link is not None:
        meta['civitaiResources'] = [{'type': 'lora', 'weight': row.strength,
                                     'modelVersionId': int(link.version_id)}]
    return meta


def _staged_png(row, dest_dir) -> tuple:
    """A metadata-free PNG copy of the row's file: the disclosure boundary the
    HF publisher already draws, reused. Returns (path, width, height)."""
    from .hf_publish import HfPublishError, _write_sanitized_publish_png
    src = _image_path(row)
    if not row.filename or not os.path.isfile(src):
        raise CivitaiPublishError('image_missing', f'image {row.id} is no longer on disk')
    dst = os.path.join(dest_dir, f'lds_{row.id}.png')
    try:
        _write_sanitized_publish_png(src, dst)
    except HfPublishError as e:
        raise CivitaiPublishError('invalid_image', f'image {row.id}: {e.message}') from e
    with Image.open(dst) as im:
        return dst, im.width, im.height


def _upload_image(path, key):
    iu = _post_json('/api/v1/image-upload', {'filename': os.path.basename(path)}, key,
                    'image-upload')
    if not isinstance(iu, dict) or not iu.get('uploadURL') or not iu.get('id'):
        raise CivitaiPublishError('civitai', 'Civitai gave no upload slot for the image.')
    with open(path, 'rb') as fh:
        data = fh.read()
    st, _h, _b = _transport('PUT', iu['uploadURL'], headers={'Content-Type': 'image/png'},
                            data=data, timeout=_UPLOAD_TIMEOUT)
    if st not in (200, 204):
        raise CivitaiPublishError('civitai', f'the image bytes were refused (HTTP {st})')
    return iu['id']


def post_images(image_ids, link, key, title=None, publish=True, progress=None,
                user_id=LOCAL_USER):
    """Post generated images on the version a checkpoint is linked to —
    synchronous, the tested seam. post.create → per image: sanitized PNG,
    upload, post.addImage with its meta → optionally post.update {publishedAt}."""
    from ..models import LoraTestImage
    from . import face_dataset_service as fds
    if not key:
        raise CivitaiPublishError(
            'no_key', 'No Civitai API key configured - paste one in Settings > Scraping & sources.')
    if link is None:
        raise CivitaiPublishError(
            'link_missing', 'This checkpoint is not linked to a Civitai model page yet - mark '
                            'the page first (paste its address) or create it.')
    ids = []
    for i in (image_ids or []):
        try:
            ids.append(int(i))
        except (TypeError, ValueError):
            continue
    if not ids:
        raise CivitaiPublishError('invalid', 'No image selected.')
    rows = [r for r in LoraTestImage.query.filter(LoraTestImage.id.in_(ids)).all()
            if r.status == 'done' and r.filename]
    by_id = {r.id: r for r in rows}
    rows = [by_id[i] for i in ids if i in by_id]
    if not rows:
        raise CivitaiPublishError('image_missing', 'None of the selected images is finished and on disk.')
    datasets = {}
    tmp = tempfile.mkdtemp(prefix='lds-civitai-')
    try:
        if progress:
            progress('creating', 0.0)
        post = trpc_mutation('post.create', {
            'modelVersionId': int(link.version_id),
            **({'title': redact_user_paths(str(title).strip())[:255]} if title and str(title).strip() else {}),
        }, key)
        post_id = int(post['id'])
        added = []
        for index, row in enumerate(rows):
            if row.dataset_id not in datasets:
                datasets[row.dataset_id] = fds.get_dataset(user_id, row.dataset_id)
            path, w, h = _staged_png(row, tmp)
            uuid_ = _upload_image(path, key)
            meta = image_meta(row, datasets[row.dataset_id], link, w, h)
            img = trpc_mutation('post.addImage', {
                'postId': post_id, 'modelVersionId': int(link.version_id), 'url': uuid_,
                'index': index, 'width': w, 'height': h, 'mimeType': 'image/png',
                'type': 'image', 'meta': meta,
            }, key)
            added.append({'image_id': row.id, 'civitai_image_id': (img or {}).get('id')})
            if progress:
                progress('uploading', (index + 1) / len(rows))
        published = False
        if publish:
            trpc_mutation('post.update', {'id': post_id, 'publishedAt': _publish_stamp_iso()},
                          key, date_fields=('publishedAt',))
            published = True
        return {'post_id': post_id, 'published': published, 'count': len(added),
                'images': added, 'url': post_url(post_id, published),
                'model_url': model_url(link.model_id, link.version_id)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --- background jobs --------------------------------------------------------------
# Both flows move hundreds of MB over a home uplink: far past a request's
# window. The route launches the work in a daemon thread and the modal polls
# `job_status`. In-memory by design (as hf_publish): a restart forgets a job,
# and the modal says so when the id it polls is unknown.

_lock = threading.Lock()
_jobs = {}
_JOBS_MAX = 50


def start_job(app, kind, work):
    """Run `work(progress)` in the background under the app context. `progress`
    is a callable `(phase, fraction)`. Returns the job id."""
    job_id = uuid.uuid4().hex
    with _lock:
        if len(_jobs) >= _JOBS_MAX:
            for stale in list(_jobs)[:len(_jobs) - _JOBS_MAX + 1]:
                _jobs.pop(stale, None)
        _jobs[job_id] = {'id': job_id, 'kind': kind, 'state': 'running', 'phase': 'starting',
                         'progress': 0.0, 'result': None, 'error': None, 'error_code': None,
                         'started_at': time.time()}

    def progress(phase, fraction):
        with _lock:
            j = _jobs.get(job_id)
            if j:
                j['phase'] = phase
                j['progress'] = max(0.0, min(1.0, float(fraction or 0.0)))

    def run():
        try:
            with app.app_context():
                result = work(progress)
            outcome = {'state': 'done', 'phase': 'done', 'progress': 1.0, 'result': result}
        except CivitaiPublishError as e:
            outcome = {'state': 'error', 'error': e.message, 'error_code': e.code}
        except Exception as e:   # never let a background thread die silently
            logger.exception('civitai publish job failed')
            outcome = {'state': 'error', 'error': redact_user_paths(f'unexpected error: {e}'),
                       'error_code': 'unknown'}
        with _lock:
            j = _jobs.get(job_id)
            if j:
                j.update(outcome)

    threading.Thread(target=run, daemon=True, name=f'civitai-{kind}').start()
    return job_id


def job_status(job_id):
    with _lock:
        j = _jobs.get(str(job_id))
        return dict(j) if j else None
