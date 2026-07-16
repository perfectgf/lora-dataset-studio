"""One-time push of a CUSTOM training base to a PRIVATE Hugging Face model repo.

Cloud pods train from Hugging Face repo ids: the official bases (and the gated
Klein ones) are downloaded pod-side with the user's HF_TOKEN. This service
extends that exact mechanism to CUSTOM weights: the user pushes their base ONCE
to a private repo on their own account, and every cloud launch afterwards points
``model.name_or_path`` at that repo — the pod downloads it at datacenter speed,
nothing is re-uploaded from the home uplink, and the pod is never billed while
the (one-time) upload runs.

Scope — the three cloud families only:
  * krea:       single ``.safetensors`` file (absolute local path);
  * flux2klein: single ``.safetensors`` file — the pod's loader HARDCODES the
                filename per size (``flux-2-klein-base-4b/9b.safetensors``), so
                the file is stored under the variant's exact expected name;
  * zimage:     the CONVERTED diffusers folder (``converted_dir``: transformer/
                config.json + weights) — uploaded as a folder; the pod loads it
                via ``from_pretrained(repo_id, subfolder='transformer')``.
SDXL and FLUX.1 stay local-only (out of scope, unchanged refusals).

Naming: ``lds-base-<combo-hash>`` where the hash is the SAME
``lora_training._custom_combo_hash`` that already isolates custom run folders —
one repo per distinct custom base, shared across datasets (the cache key).
Privacy is NON-NEGOTIABLE: the repo is created ``private=True``, there is no
toggle, and a pre-existing repo found public is flipped back to private before
any upload (refusing outright if that fails).

Reuses hf_publish's bricks: HfApi construction, the write-scope whoami
preflight (read-only tokens refused BEFORE any upload, with the actionable
message), HTTP status extraction, and the background-job + poll-status shape.
"""
import logging
import os
import threading

from ..config import LOCAL_USER
from . import face_dataset_service as fds
from . import lora_training as lt
from .hf_publish import (HfPublishError, _http_status, _make_api,
                         _require_write_scope)

logger = logging.getLogger(__name__)

# Families whose custom base can ride the cloud lane through a private HF repo.
# sdxl/flux are deliberately absent: their cloud refusals are unchanged.
CLOUD_CUSTOM_BASE_FAMILIES = ('zimage', 'krea', 'flux2klein')

# Mirror of ai-toolkit's Flux2Klein{4B,9B}Model.flux2_te_filename: the pod-side
# loader downloads EXACTLY this filename from the repo (no config override).
_KLEIN_FILENAMES = {'4b': 'flux-2-klein-base-4b.safetensors',
                    '9b': 'flux-2-klein-base-9b.safetensors'}

_ZIMAGE_REPO_FILES = ('transformer/config.json',
                      'transformer/diffusion_pytorch_model.safetensors')


def normalized_variant(family, variant) -> str:
    """Same per-family variant enum as the launch paths (foreign value -> family
    default) so the repo layout and the launch config can never disagree."""
    var = str(variant or '').strip().lower()
    if var not in lt._valid_variants_for(family):
        var = lt._default_variant_for(family)
    return var


def base_repo_name(ds, family, base_model) -> str:
    """Deterministic repo name for THIS custom base: ``lds-base-h<sha1[:8]>``,
    derived from the exact combo hash that already tags custom run folders.
    Raises HfPublishError when the selection carries nothing custom."""
    tag = lt._custom_combo_hash(ds, base_model, family)
    if not tag:
        raise HfPublishError('no_custom_base',
                             'this launch uses an official base — nothing to push')
    return 'lds-base-' + tag.lstrip('_')


def weight_filename(family, variant, repo_name) -> str | None:
    """Name of the single weight file inside the repo (None for zimage, whose
    payload is a diffusers folder).

    * flux2klein: the pod loader's HARDCODED per-size filename;
    * krea: ``<repo-name-tail>.safetensors`` — chosen to ALSO match ai-toolkit's
      derived-filename fallback (``repo.split('-')[-1] + '.safetensors'``), so
      even a pod predating ``model_kwargs.checkpoint_filename`` resolves it."""
    if family == 'flux2klein':
        return _KLEIN_FILENAMES['9b' if normalized_variant(family, variant) == '9b'
                                else '4b']
    if family == 'krea':
        return repo_name.split('-')[-1] + '.safetensors'
    return None


def expected_repo_files(family, variant, repo_name) -> list:
    """The repo paths the POD will actually fetch — what push uploads and what
    every readiness/launch check verifies."""
    if family == 'zimage':
        return list(_ZIMAGE_REPO_FILES)
    return [weight_filename(family, variant, repo_name)]


def _weight_repo_path(family, variant, repo_name) -> str:
    """Repo path of the big weights file (the one worth size-checking)."""
    if family == 'zimage':
        return _ZIMAGE_REPO_FILES[1]
    return weight_filename(family, variant, repo_name)


def local_base_payload(family, base_model) -> dict:
    """What must be uploaded for (family, base_model):
    {'kind': 'file'|'folder', 'path', 'size_bytes', 'weight_size_bytes'}.
    Raises HfPublishError with an actionable code when the local artifact is
    absent (weights_missing / not_converted / unsupported_family)."""
    if family not in CLOUD_CUSTOM_BASE_FAMILIES:
        raise HfPublishError(
            'unsupported_family',
            'custom weights are local-only for this family — the cloud custom '
            'lane covers Z-Image, Krea 2 and FLUX.2 Klein')
    base_model = str(base_model or '').strip()
    if not base_model:
        raise HfPublishError('no_custom_base',
                             'this launch uses an official base — nothing to push')
    if family == 'zimage':
        # Z-Image customs are ComfyUI merges converted ONCE to diffusers layout
        # (the same conversion local training uses) — the folder is the payload.
        from . import zimage_convert
        if not zimage_convert.is_converted(base_model):
            raise HfPublishError(
                'not_converted',
                'this Z-Image base has not been converted to the diffusers '
                'format yet — run "Convert" in the training panel first, then '
                'push it to your Hugging Face account')
        folder = zimage_convert.converted_dir(base_model)
        weights = os.path.join(folder, 'transformer',
                               'diffusion_pytorch_model.safetensors')
        return {'kind': 'folder', 'path': folder,
                'size_bytes': lt._dir_size(folder),
                'weight_size_bytes': os.path.getsize(weights)}
    if not lt._is_custom_weights(base_model) or not os.path.isfile(base_model):
        raise HfPublishError(
            'weights_missing',
            f'custom weights file not found on this machine: {base_model} — '
            'the one-time push needs the local file (already-pushed bases '
            'keep working without it)')
    size = os.path.getsize(base_model)
    return {'kind': 'file', 'path': base_model,
            'size_bytes': size, 'weight_size_bytes': size}


# --- remote readiness ----------------------------------------------------------

def _remote_check(api, repo_id, family, variant, repo_name,
                  local_weight_size=None) -> dict:
    """Inspect the repo the POD would download: existence, file presence, the
    private flag, and the weight file's size (compared against the local copy
    when one still exists). Returns {'ready', 'reason', 'private',
    'remote_size_bytes'} — never raises for a missing repo (that IS a reason)."""
    expected = expected_repo_files(family, variant, repo_name)
    try:
        info = api.repo_info(repo_id, repo_type='model')
    except Exception as e:
        status = _http_status(e)
        if status in (401, 403, 404):
            return {'ready': False, 'reason': 'not_pushed', 'private': None,
                    'remote_size_bytes': None}
        raise HfPublishError(
            'network', f'could not reach Hugging Face to check the repo: {e}') from e
    try:
        paths = api.get_paths_info(repo_id, expected, repo_type='model')
    except Exception as e:
        raise HfPublishError(
            'network', f'could not list the repo contents: {e}') from e
    by_path = {getattr(p, 'path', None): p for p in (paths or [])}
    if any(f not in by_path for f in expected):
        return {'ready': False, 'reason': 'file_missing',
                'private': getattr(info, 'private', None),
                'remote_size_bytes': None}
    weight_path = _weight_repo_path(family, variant, repo_name)
    entry = by_path.get(weight_path)
    remote_size = getattr(entry, 'size', None) \
        or getattr(getattr(entry, 'lfs', None), 'size', None)
    if local_weight_size and remote_size and int(remote_size) != int(local_weight_size):
        return {'ready': False, 'reason': 'size_mismatch',
                'private': getattr(info, 'private', None),
                'remote_size_bytes': int(remote_size)}
    return {'ready': True, 'reason': None,
            'private': getattr(info, 'private', None),
            'remote_size_bytes': int(remote_size) if remote_size else None}


def base_push_state(user_id, dataset_id, family, variant, base_model, token,
                    _api=None) -> dict:
    """Everything the launch dialog needs to decide between 'push first' and
    'reuse the pushed base': local artifact presence/size, the deterministic
    repo id, remote readiness, and the background push job's state. Degrades
    to ready=False + reason instead of raising (poll-friendly)."""
    family = fds.normalize_train_type(family)
    variant = normalized_variant(family, variant)
    out = {'supported': family in CLOUD_CUSTOM_BASE_FAMILIES,
           'family': family, 'variant': variant, 'ready': False,
           'reason': None, 'repo_id': None, 'repo_name': None,
           'local_size_bytes': None, 'local_available': False,
           'remote_size_bytes': None, 'job': {'state': 'idle'}}
    if not out['supported']:
        out['reason'] = 'unsupported_family'
        return out
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        out['reason'] = 'dataset_not_found'
        return out
    try:
        repo_name = base_repo_name(ds, family, base_model)
    except HfPublishError as e:
        out['reason'] = e.code
        return out
    out['repo_name'] = repo_name
    out['job'] = push_status(repo_name)
    local_weight_size = None
    try:
        payload = local_base_payload(family, base_model)
        out['local_size_bytes'] = payload['size_bytes']
        out['local_available'] = True
        local_weight_size = payload['weight_size_bytes']
    except HfPublishError as e:
        # A vanished local file is NOT fatal: an already-pushed repo keeps
        # working (that is the whole point). Report why push is unavailable.
        out['local_reason'] = e.code
    if not token:
        out['reason'] = 'no_token'
        return out
    api = _api or _make_api(token)
    try:
        who = api.whoami()
    except Exception:
        out['reason'] = 'token_invalid'
        return out
    namespace = (who or {}).get('name')
    if not namespace:
        out['reason'] = 'token_invalid'
        return out
    out['repo_id'] = f'{namespace}/{repo_name}'
    try:
        check = _remote_check(api, out['repo_id'], family, variant, repo_name,
                              local_weight_size=local_weight_size)
    except HfPublishError as e:
        out['reason'] = e.code
        return out
    out.update(ready=check['ready'], reason=check['reason'],
               remote_size_bytes=check['remote_size_bytes'],
               private=check['private'])
    return out


# --- launch-time guard (cloud_training calls this before renting anything) -----

def require_base_repo(ds, family, variant, base_model, token) -> dict:
    """The pre-rent guard: raises an ACTIONABLE ValueError unless the custom
    base is fully downloadable by the pod (repo exists on the user's account,
    expected files present, size matching the local copy when it still
    exists). Returns {'repo_id', 'size_bytes'} on success — size_bytes is the
    REMOTE weight size (what the pod will actually pull), for disk sizing."""
    family = fds.normalize_train_type(family)
    variant = normalized_variant(family, variant)
    if family not in CLOUD_CUSTOM_BASE_FAMILIES:
        raise ValueError('custom weights are local-only — cloud training '
                         'uses the official Hugging Face bases')
    if not token:
        raise ValueError(
            'a custom base trains from a PRIVATE repo on your Hugging Face '
            'account — add your HF_TOKEN in Settings ▸ API keys first')
    try:
        repo_name = base_repo_name(ds, family, base_model)
    except HfPublishError as e:
        raise ValueError(e.message) from e
    local_weight_size = None
    try:
        local_weight_size = local_base_payload(family, base_model)['weight_size_bytes']
    except HfPublishError:
        pass                       # pushed repo works even without the local file
    try:
        api = _make_api(token)
        who = api.whoami()
    except HfPublishError:
        raise
    except Exception as e:
        raise ValueError(
            f'could not verify your Hugging Face token before launch: {e}') from e
    namespace = (who or {}).get('name')
    if not namespace:
        raise ValueError('your Hugging Face token was rejected — paste a valid '
                         'HF_TOKEN in Settings ▸ API keys')
    repo_id = f'{namespace}/{repo_name}'
    try:
        check = _remote_check(api, repo_id, family, variant, repo_name,
                              local_weight_size=local_weight_size)
    except HfPublishError as e:
        raise ValueError(e.message) from e
    if check['reason'] == 'not_pushed':
        raise ValueError(
            f'your custom base is not on your Hugging Face account yet — use '
            f'"Push custom base" in the cloud dialog first (one-time upload '
            f'to the private repo {repo_id}, reused for every future run)')
    if check['reason'] == 'file_missing':
        needed = ', '.join(expected_repo_files(family, variant, repo_name))
        raise ValueError(
            f'the private repo {repo_id} exists but is missing {needed} — '
            'push the custom base again from the cloud dialog (the FLUX.2 '
            'Klein 4B and 9B lanes use different file names)')
    if check['reason'] == 'size_mismatch':
        raise ValueError(
            f'your local custom base differs from the copy pushed to '
            f'{repo_id} ({local_weight_size} bytes local vs '
            f'{check["remote_size_bytes"]} pushed) — push it again to update '
            'the private repo before training on it')
    if not check['ready']:
        raise ValueError(f'the pushed custom base at {repo_id} failed '
                         f'verification ({check["reason"]})')
    return {'repo_id': repo_id,
            'size_bytes': check['remote_size_bytes'] or local_weight_size or 0}


# --- the synchronous push flow (the tested seam) --------------------------------

def push_base_to_hf(dataset_id, family, variant, base_model, token,
                    user_id=LOCAL_USER, allow_unverified_weights=False,
                    _api=None) -> dict:
    """Full synchronous push: validate -> arch sniff (confirmable) -> write-scope
    preflight -> create PRIVATE repo (repair to private if it drifted public) ->
    upload file/folder -> verify what the pod would download. Cache-hit (repo
    already carries the exact files) skips the upload entirely."""
    family = fds.normalize_train_type(family)
    variant = normalized_variant(family, variant)
    if not token:
        raise HfPublishError('no_token', 'no Hugging Face token configured — '
                             'paste an HF_TOKEN in Settings ▸ API keys')
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise HfPublishError('dataset_not_found', 'dataset not found')
    payload = local_base_payload(family, base_model)   # raises when absent locally
    # Same architecture guardrail as a local launch (confirmable marker —
    # CUSTOM_WEIGHTS_UNVERIFIED — so the UI can confirm-and-retry). Z-Image
    # customs skip the sniff exactly like the local path: they are relative
    # ComfyUI merge names whose conversion already validated the architecture.
    if payload['kind'] == 'file':
        lt.preflight_custom_paths(family, weights=base_model,
                                  allow_unverified_weights=allow_unverified_weights)
    repo_name = base_repo_name(ds, family, base_model)

    api = _api or _make_api(token)
    who = _require_write_scope(api)          # refuse read-only BEFORE any upload
    namespace = (who or {}).get('name')
    if not namespace:
        raise HfPublishError('auth', 'could not resolve your Hugging Face '
                             'username from the token')
    repo_id = f'{namespace}/{repo_name}'
    repo_url = f'https://huggingface.co/{repo_id}'

    # Cache-hit: the whole point of the deterministic name — an already-pushed
    # base is reused forever, nothing is transferred again.
    check = _remote_check(api, repo_id, family, variant, repo_name,
                          local_weight_size=payload['weight_size_bytes'])
    if check['ready']:
        _ensure_private(api, repo_id, check['private'])
        return {'ok': True, 'repo_id': repo_id, 'repo_url': repo_url,
                'cached': True, 'size_bytes': payload['size_bytes']}

    try:
        # private=True is FORCED — never a parameter, never a toggle: these are
        # the user's personal model weights.
        api.create_repo(repo_id=repo_id, repo_type='model', private=True,
                        exist_ok=True)
    except Exception as e:
        status = _http_status(e)
        if status in (401, 403):
            raise HfPublishError(
                'auth', 'Hugging Face refused to create the repo (401/403) — '
                'check the token has write access') from e
        raise HfPublishError('network', f'could not create the repo: {e}') from e
    _ensure_private(api, repo_id, None)

    commit_message = f'Push custom {family} base (LoRA Dataset Studio)'
    try:
        if payload['kind'] == 'folder':
            api.upload_folder(folder_path=payload['path'], repo_id=repo_id,
                              repo_type='model', commit_message=commit_message)
        else:
            api.upload_file(path_or_fileobj=payload['path'],
                            path_in_repo=weight_filename(family, variant, repo_name),
                            repo_id=repo_id, repo_type='model',
                            commit_message=commit_message)
    except Exception as e:
        status = _http_status(e)
        if status in (401, 403):
            raise HfPublishError(
                'auth', 'Hugging Face rejected the upload (401/403) — '
                'check the token has write access') from e
        # Quota / storage errors carry their explanation in the HF message —
        # surface it verbatim, it is the actionable part.
        raise HfPublishError('network',
                             f'the upload to Hugging Face failed: {e}') from e

    # Verify what the pod will actually download (files present, size intact).
    check = _remote_check(api, repo_id, family, variant, repo_name,
                          local_weight_size=payload['weight_size_bytes'])
    if not check['ready']:
        raise HfPublishError(
            'verify_failed',
            f'the upload finished but the repo failed verification '
            f'({check["reason"]}) — retry the push')
    return {'ok': True, 'repo_id': repo_id, 'repo_url': repo_url,
            'cached': False, 'size_bytes': payload['size_bytes']}


def _ensure_private(api, repo_id, known_private):
    """The privacy invariant: a lds-base-* repo is NEVER public. If it drifted
    (user flipped it in the HF UI), flip it back; refuse to proceed when that
    fails — leaking personal model weights is worse than a failed push."""
    private = known_private
    if private is None:
        try:
            private = getattr(api.repo_info(repo_id, repo_type='model'),
                              'private', None)
        except Exception:
            return                      # freshly created private repo: fine
    if private is False:
        try:
            api.update_repo_settings(repo_id=repo_id, private=True,
                                     repo_type='model')
            logger.warning('repo %s was PUBLIC — flipped back to private', repo_id)
        except Exception as e:
            raise HfPublishError(
                'repo_public',
                f'the repo {repo_id} is PUBLIC and could not be made private '
                f'({e}) — make it private on huggingface.co, then retry') from e


# --- background job (multi-GB uploads outlive any request window) ---------------
# Keyed by the repo NAME (the combo-hash cache key), so two datasets pushing the
# same base share one job and one status.

_lock = threading.Lock()
_jobs = {}   # repo_name -> {state, repo_id, repo_url, error, error_code, cached}


def push_status(repo_name) -> dict:
    with _lock:
        j = _jobs.get(repo_name)
        return dict(j) if j else {'state': 'idle'}


def start_push(app, dataset_id, family, variant, base_model, token,
               user_id=LOCAL_USER, allow_unverified_weights=False) -> dict:
    """Launch the push in the background (daemon thread, hf_publish shape).
    Returns the initial status; a second call while one runs is a no-op."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise HfPublishError('dataset_not_found', 'dataset not found')
    family = fds.normalize_train_type(family)
    repo_name = base_repo_name(ds, family, base_model)
    # Cheap validations run SYNCHRONOUSLY so the request answers with the
    # actionable error (missing file, unconverted Z-Image base, and the
    # confirmable CUSTOM_WEIGHTS_UNVERIFIED arch sniff) instead of burying it
    # in a poll. The background job re-runs them harmlessly.
    payload = local_base_payload(family, base_model)
    if payload['kind'] == 'file':
        lt.preflight_custom_paths(family, weights=base_model,
                                  allow_unverified_weights=allow_unverified_weights)
    with _lock:
        cur = _jobs.get(repo_name)
        if cur and cur.get('state') == 'running':
            return {**cur, 'already': True}
        _jobs[repo_name] = {'state': 'running', 'repo_id': None, 'repo_url': None,
                            'error': None, 'error_code': None, 'cached': None}
    threading.Thread(
        target=_run_push_job,
        args=(app, repo_name, dataset_id, family, variant, base_model, token,
              user_id, allow_unverified_weights),
        daemon=True).start()
    return {'state': 'running', 'repo_name': repo_name}


def _run_push_job(app, repo_name, dataset_id, family, variant, base_model,
                  token, user_id, allow_unverified_weights):
    try:
        with app.app_context():
            res = push_base_to_hf(dataset_id, family, variant, base_model,
                                  token, user_id=user_id,
                                  allow_unverified_weights=allow_unverified_weights)
        result = {'state': 'done', 'repo_id': res['repo_id'],
                  'repo_url': res['repo_url'], 'error': None,
                  'error_code': None, 'cached': res.get('cached')}
    except HfPublishError as e:
        result = {'state': 'error', 'repo_id': None, 'repo_url': None,
                  'error': e.message, 'error_code': e.code, 'cached': None}
    except ValueError as e:
        # preflight_custom_paths raises the confirmable CUSTOM_WEIGHTS_UNVERIFIED
        # marker as a plain ValueError — surface it verbatim for the UI.
        result = {'state': 'error', 'repo_id': None, 'repo_url': None,
                  'error': str(e), 'error_code': 'unverified_weights'
                  if str(e).startswith(lt._UNVERIFIED_MARKER) else 'invalid',
                  'cached': None}
    except Exception as e:   # never let a background thread die silently
        result = {'state': 'error', 'repo_id': None, 'repo_url': None,
                  'error': f'unexpected error: {e}', 'error_code': 'unknown',
                  'cached': None}
    with _lock:
        _jobs[repo_name] = result
