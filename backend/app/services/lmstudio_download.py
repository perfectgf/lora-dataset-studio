"""Download an LM Studio model from inside LDS — the missing half of `ollama pull`.

The user's ask, verbatim: "model downloads only happen inside LM Studio — can't
we add that to the LLM settings, like Ollama pull?" Until this module the answer
in three different sentences of UI copy was "downloading stays in LM Studio".
That was a description of a gap, not of a constraint — measured on 0.4.23:

  · POST /api/v1/models/download {"model": ...} answers IMMEDIATELY with a job:
    {"status": "downloading", "job_id": ..., "total_size_bytes": ..., ...}.
    The download then runs inside LM Studio's own process — LDS holds no thread,
    no socket, nothing that dies with a page reload.
  · Re-POSTing the SAME model is idempotent and returns the job's CURRENT
    status; at completion it answers {"status": "already_downloaded"} — the same
    shape as a model that was already on disk, which is exactly the right
    meaning for both.
  · No byte counter in the response — but the file lands under
    ~/.lmstudio/models/<owner>/<repo>/ WHILE it downloads, so real progress is
    bytes-on-disk over total_size_bytes.
  · Identifier rules differ by origin, and the server says so: a catalog id
    ("qwen/qwen3-vl-4b") posts as-is; a community model is refused with
    "use the HuggingFace model URL instead" — so that exact refusal triggers ONE
    retry with the https://huggingface.co/ prefix, and the form that worked is
    remembered for the status polls.

The public shape mirrors ollama_control's pull snapshot ({state, model,
progress, log, error}) so every screen that already knows how to render a pull
renders this one unchanged.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_TIMEOUT = (10, 60)
# The server's own refusal for a community id posted bare — the ONE error that
# earns a retry with the HF-URL form, nothing else does.
_USE_HF_URL_MARKER = 'huggingface model url'

_lock = threading.Lock()
# The one download this process is watching: {model (as typed), url_form (what
# the server accepted), total, error, done}. LM Studio itself allows several
# concurrent jobs, but one at a time is what the Ollama pull offers, what the UI
# renders, and what a person actually does.
_current: dict | None = None


def _base_url() -> str:
    from . import vision_lmstudio
    return vision_lmstudio.base_url()


def _headers() -> dict:
    from . import vision_lmstudio
    return {'Content-Type': 'application/json', **vision_lmstudio._headers()}


def _post_download(ref: str) -> tuple[int, dict]:
    """POST the download/status request. (status_code, body-or-{}) — never raises."""
    try:
        resp = requests.post(f'{_base_url()}/api/v1/models/download',
                             json={'model': ref}, headers=_headers(),
                             timeout=_TIMEOUT)
    except requests.RequestException as exc:
        return 0, {'error': {'message': str(exc), 'type': 'unreachable'}}
    try:
        return resp.status_code, resp.json()
    except ValueError:
        return resp.status_code, {'error': {'message': (resp.text or '')[:300],
                                            'type': 'bad-json'}}


def _models_dir() -> Path:
    """Where LM Studio writes downloads. Overridable for tests; the default is
    the location every install uses (the app offers to move it, in which case
    progress degrades to None — the poll still ends on already_downloaded)."""
    override = os.environ.get('LDS_LMSTUDIO_MODELS_DIR')
    return Path(override) if override else Path.home() / '.lmstudio' / 'models'


def _bytes_on_disk(url_form: str) -> int | None:
    """Bytes already written for this model, or None when the folder is not
    derivable/visible. owner/repo comes from the tail of the HF URL or the id."""
    tail = url_form.rstrip('/').split('://')[-1]
    parts = [p for p in tail.split('/') if p]
    if len(parts) < 2:
        return None
    folder = _models_dir() / parts[-2] / parts[-1]
    try:
        if not folder.is_dir():
            return None
        return sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
    except OSError:
        return None


def _snapshot() -> dict:
    if _current is None:
        return {'state': 'idle', 'model': '', 'progress': None, 'log': [], 'error': None}
    c = _current
    return {'state': c['state'], 'model': c['model'], 'progress': c.get('progress'),
            'log': list(c.get('log') or []), 'error': c.get('error')}


def _friendly_error(body: dict) -> str:
    err = body.get('error') or {}
    kind = (err.get('type') or '').strip()
    msg = (err.get('message') or '').strip()
    if kind == 'model_not_found':
        return ('LM Studio does not know that model — check the spelling, or paste '
                'the model page URL from huggingface.co.')
    if kind == 'unreachable':
        return f'LM Studio is not answering — {msg}'
    return msg or 'LM Studio refused the download.'


def start_download(model: str) -> dict:
    """Begin (or re-attach to) a download. Returns the pull-shaped snapshot.

    Re-attaching matters more than it looks: the job lives in LM Studio, so an
    LDS restart mid-download loses nothing — POSTing the same model again simply
    answers with the running job's status.
    """
    global _current
    name = (model or '').strip()
    if not name or len(name) > 300 or any(ch in name for ch in '\r\n"\''):
        return {**_snapshot(), 'ok': False,
                'error': 'give a model id ("qwen/qwen3-vl-4b") or a huggingface.co URL'}
    with _lock:
        if _current is not None and _current['state'] == 'running' \
                and _current['model'] != name:
            return {'ok': False, **_snapshot(),
                    'error': (f'Already downloading "{_current["model"]}". Wait for it '
                              f'to finish before starting "{name}".')}

    forms = [name]
    if '://' not in name:
        forms.append(f'https://huggingface.co/{name}')
    last_body: dict = {}
    for i, ref in enumerate(forms):
        code, body = _post_download(ref)
        status = (body.get('status') or '').strip()
        if status in ('downloading', 'already_downloaded'):
            with _lock:
                _current = {'model': name, 'url_form': ref,
                            'total': body.get('total_size_bytes'),
                            'state': 'done' if status == 'already_downloaded' else 'running',
                            'progress': 100 if status == 'already_downloaded' else 0,
                            'log': [], 'error': None}
                return {'ok': True, **_snapshot()}
        last_body = body
        msg = ((body.get('error') or {}).get('message') or '').lower()
        # Only the server's own "use the HuggingFace model URL" earns the retry;
        # any other refusal on the bare form would repeat identically on the
        # prefixed one and burn a request to say the same thing twice.
        if i == 0 and len(forms) > 1 and _USE_HF_URL_MARKER not in msg:
            break
    with _lock:
        _current = {'model': name, 'url_form': name, 'total': None,
                    'state': 'error', 'progress': None, 'log': [],
                    'error': _friendly_error(last_body)}
        return {'ok': False, **_snapshot()}


def download_status() -> dict:
    """The poll: re-POST the accepted form, translate, measure the disk.

    `progress` is a 0-100 int when the disk is measurable, else None — the same
    contract as the Ollama pull, whose progress also goes missing sometimes and
    whose consumers already render that honestly.
    """
    with _lock:
        if _current is None or _current['state'] != 'running':
            return _snapshot()
        ref, total, name = _current['url_form'], _current.get('total'), _current['model']
    code, body = _post_download(ref)
    status = (body.get('status') or '').strip()
    with _lock:
        if _current is None or _current['model'] != name:
            return _snapshot()          # replaced concurrently; report the new truth
        if status == 'already_downloaded':
            _current.update(state='done', progress=100, error=None)
        elif status == 'downloading':
            _current['total'] = total = body.get('total_size_bytes') or total
            done = _bytes_on_disk(ref)
            _current['progress'] = (max(0, min(100, round(done * 100 / total)))
                                    if done is not None and total else None)
        else:
            _current.update(state='error', error=_friendly_error(body))
        return _snapshot()
