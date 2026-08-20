"""Build/resume the versioned SigLIP2 Image Bank embedding cache.

The app sends every model/provenance value in the request.  This worker contains
no duplicated model id, revision, key, or embedding dimension and can therefore
never silently drift from ``app.services.bank_semantic_models``.

stdin (one JSON document)::

    {"images": [...], "cache": "...semantic_siglip2_cache.npz",
     "cancel_file": "...cancel", "device": "auto|cpu|cuda",
     "cache_version": "<from app>", "engine": "siglip2", "model_id": "...",
     "revision": "...", "model_key": "...", "dimension": "<from app>",
     "models_root": "...", "rescore": false}

stderr uses ``[semantic] i/N`` and ``[phase] ...`` lines understood by
``_drive_infer_subprocess``.  stdout ends with one JSON result line.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import sys
from typing import Any


CACHE_EVERY = 50
_CACHE_KEYS = frozenset({
    'version', 'engine', 'model_id', 'revision', 'model_key', 'dimension',
    'paths', 'states', 'embs', 'sigs', 'hashes',
})

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import import_report  # noqa: E402
from bank_image_guard import read_validated_bank_image  # noqa: E402
import npz_atomic  # noqa: E402


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _phase(message: str) -> None:
    _log(f'[phase] {message}')


def _emit(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def _contract_from_request(request: dict) -> dict:
    """Validate a flattened or nested app-provided model contract."""
    merged: dict[str, Any] = {}
    for key in ('contract', 'model'):
        nested = request.get(key)
        if isinstance(nested, dict):
            merged.update(nested)
    merged.update({key: value for key, value in request.items()
                   if key not in ('contract', 'model')})
    model_id = merged.get('model_id') or merged.get('pretrained_model_name_or_path')
    models_root = merged.get('models_root') or merged.get('cache_dir')
    contract = {
        'cache_version': merged.get('cache_version'),
        'engine': str(merged.get('engine') or '').strip(),
        'model_id': str(model_id or '').strip(),
        'revision': str(merged.get('revision') or '').strip(),
        'model_key': str(merged.get('model_key') or '').strip(),
        'models_root': str(models_root or '').strip(),
    }
    try:
        contract['cache_version'] = int(contract['cache_version'])
        contract['dimension'] = int(merged.get('dimension'))
    except (TypeError, ValueError) as exc:
        raise ValueError('cache_version and dimension must be integers') from exc
    if contract['cache_version'] != 1:
        raise ValueError('unsupported semantic cache version')
    if contract['engine'] != 'siglip2':
        raise ValueError('bank semantic worker requires engine=siglip2')
    for key in ('model_id', 'revision', 'model_key', 'models_root'):
        if not contract[key]:
            raise ValueError(f'missing semantic model contract field: {key}')
    if contract['dimension'] <= 0:
        raise ValueError('semantic embedding dimension must be positive')
    return contract


def _scalar(array: Any, key: str) -> Any:
    import numpy as np
    value = np.asarray(array)
    if value.shape != (1,):
        raise ValueError(f'{key} metadata must have shape (1,)')
    return value[0].item()


def _strings(array: Any, key: str, length: int | None = None) -> tuple[str, ...]:
    import numpy as np
    value = np.asarray(array)
    if value.ndim != 1 or value.dtype.kind not in ('U', 'S'):
        raise ValueError(f'{key} must be a string vector')
    if length is not None and len(value) != length:
        raise ValueError(f'{key} length mismatch')
    return tuple(str(item) for item in value)


# Cache tuple: (state, embedding, stat signature, raw SHA-256 bytes).
def _load_cache(path: str | None, contract: dict) -> dict:
    import numpy as np
    if not path or not os.path.isfile(path):
        return {}
    try:
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != _CACHE_KEYS:
                raise ValueError('cache keys do not match the versioned contract')
            version = np.asarray(z['version'])
            dimension = np.asarray(z['dimension'])
            # Shape (1,) is part of the transport contract; 0-D metadata used to
            # round-trip differently across a few NPZ consumers.
            if version.shape != (1,) or version.dtype != np.dtype('int32'):
                raise ValueError('invalid cache version representation')
            if dimension.shape != (1,) or dimension.dtype != np.dtype('int32'):
                raise ValueError('invalid cache dimension representation')
            expected = {
                'version': contract['cache_version'],
                'engine': contract['engine'],
                'model_id': contract['model_id'],
                'revision': contract['revision'],
                'model_key': contract['model_key'],
                'dimension': contract['dimension'],
            }
            for key, wanted in expected.items():
                value = np.asarray(z[key])
                if value.shape != (1,):
                    raise ValueError(f'invalid {key} metadata shape')
                if key not in ('version', 'dimension') and value.dtype.kind not in ('U', 'S'):
                    raise ValueError(f'invalid {key} metadata dtype')
                if _scalar(value, key) != wanted:
                    raise ValueError(f'cache {key} provenance mismatch')
            paths = _strings(z['paths'], 'paths')
            states = _strings(z['states'], 'states', len(paths))
            sigs = _strings(z['sigs'], 'sigs', len(paths))
            if len(set(paths)) != len(paths):
                raise ValueError('duplicate paths in semantic cache')
            if any(state not in ('ok', 'error') for state in states):
                raise ValueError('invalid semantic cache state')
            embs = np.asarray(z['embs'])
            hashes = np.asarray(z['hashes'])
            if (embs.dtype != np.dtype('float32')
                    or embs.shape != (len(paths), contract['dimension'])):
                raise ValueError('invalid semantic embedding array')
            if hashes.dtype != np.dtype('uint8') or hashes.shape != (len(paths), 32):
                raise ValueError('invalid semantic hash array')
            if not np.isfinite(embs).all():
                raise ValueError('non-finite semantic embedding')
            ok_indexes = [i for i, state in enumerate(states) if state == 'ok']
            if ok_indexes:
                norms = np.linalg.norm(embs[ok_indexes], axis=1)
                if not np.allclose(norms, 1.0, rtol=1e-3, atol=1e-4):
                    raise ValueError('semantic embeddings are not L2-normalised')
        return {
            path_value: (states[index], embs[index], sigs[index],
                         hashes[index].tobytes())
            for index, path_value in enumerate(paths)
        }
    except Exception as exc:  # corrupt/cross-engine cache => safe full rebuild
        _log(f'[semantic] cache rejected, recomputing: {exc}')
        return {}


def _cache_hash(entry: tuple) -> bytes:
    value = entry[3] if len(entry) > 3 else b''
    return value if isinstance(value, bytes) and len(value) == 32 else b''


def _save_cache(path: str | None, cache: dict, contract: dict) -> None:
    """Write all eleven NPZ keys atomically; metadata is shape ``(1,)``."""
    import numpy as np
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = list(cache)
    dimension = contract['dimension']
    if paths:
        embs = np.stack([cache[p][1] for p in paths]).astype('float32')
        hashes = np.frombuffer(b''.join(
            _cache_hash(cache[p]) or (b'\0' * 32) for p in paths),
            dtype='uint8').reshape(len(paths), 32)
    else:
        embs = np.empty((0, dimension), dtype='float32')
        hashes = np.empty((0, 32), dtype='uint8')
    npz_atomic.save_npz_atomic(destination, dict(
        version=np.asarray([contract['cache_version']], dtype='int32'),
        engine=np.asarray([contract['engine']]),
        model_id=np.asarray([contract['model_id']]),
        revision=np.asarray([contract['revision']]),
        model_key=np.asarray([contract['model_key']]),
        dimension=np.asarray([dimension], dtype='int32'),
        paths=np.asarray(paths, dtype=np.str_),
        states=np.asarray([cache[p][0] for p in paths], dtype=np.str_),
        embs=embs,
        sigs=np.asarray([cache[p][2] for p in paths], dtype=np.str_),
        hashes=hashes,
    ))


def _flush_cache(path: str | None, cache: dict, contract: dict) -> None:
    """Save, and never let a HELD cache file kill the pass.

    This writer used to delete its own temporary in a ``finally``, which meant a
    lost rename destroyed the finished archive on the way out. The temporary is
    now the fallback, not litter: it stays, and the next pass salvages it."""
    try:
        _save_cache(path, cache, contract)
    except npz_atomic.NpzReplaceLocked as error:
        _phase(str(error))


def _salvage_cache(path: str | None, contract: dict) -> None:
    """Promote a temporary left by an interrupted run, if it reads back clean.

    ``_load_cache`` is the validator, and here that matters twice: besides
    catching a file truncated mid-write, it rejects a temporary whose PROVENANCE
    (engine, model id, revision, dimension) is not the one this pass runs, so a
    cache from another semantic engine can never be promoted over this one."""
    if not path:
        return
    npz_atomic.salvage_orphan_tmp(
        path, lambda candidate: len(_load_cache(candidate, contract)), _log)


def _write_count(cache_path: str | None, count: int) -> None:
    if not cache_path:
        return
    destination = Path(str(cache_path) + '.count')
    temporary = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f'{destination.name}.{os.getpid()}-{secrets.token_hex(4)}.tmp')
        with open(temporary, 'w', encoding='utf-8') as target:
            target.write(str(max(0, int(count))))
            target.flush()
            os.fsync(target.fileno())
        os.replace(str(temporary), str(destination))
    except OSError:
        pass
    finally:
        try:
            if temporary is not None:
                temporary.unlink()
        except OSError:
            pass


def _cancel_requested(cancel_file: str | None) -> bool:
    return bool(cancel_file) and os.path.exists(str(cancel_file))


def _file_sig(path: str) -> str:
    try:
        stat = os.stat(path)
        return f'{stat.st_size}:{stat.st_mtime_ns}'
    except OSError:
        return ''


def _file_hash(path: str, expected_sig: str | None = None) -> bytes:
    before = _file_sig(path)
    if not before or (expected_sig and before != expected_sig):
        return b''
    digest = hashlib.sha256()
    try:
        with open(path, 'rb') as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return b''
    return digest.digest() if _file_sig(path) == before else b''


def _is_stale(path: str, entry: tuple) -> bool:
    # Errors are diagnostic checkpoints, never completed semantic work.  Retry
    # them on every launch so a transient decode/model failure can heal and the
    # Bank can eventually reach ``complete``.
    if not entry or entry[0] != 'ok':
        return True
    stored_sig = str(entry[2] or '')
    stored_hash = _cache_hash(entry)
    if not stored_sig or not stored_hash or stored_hash == b'\0' * 32:
        return True
    current_sig = _file_sig(path)
    return (not current_sig or current_sig != stored_sig
            or _file_hash(path, current_sig) != stored_hash)


def _move_to_device(inputs: Any, device: str) -> Any:
    if hasattr(inputs, 'to'):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {key: value.to(device) if hasattr(value, 'to') else value
                for key, value in inputs.items()}
    return inputs


def _pooled_features(output: Any) -> Any:
    """Tensor from old direct-return and new BaseModelOutput Transformers APIs."""
    pooled = getattr(output, 'pooler_output', None)
    if pooled is not None:
        return pooled
    if isinstance(output, (tuple, list)):
        # BaseModelOutputWithPooling(return_dict=False) is
        # ``(last_hidden_state, pooler_output, ...)``.
        if len(output) > 1:
            return output[1]
        if output:
            return output[0]
    return output


def _result_payload(contract: dict, images: list[str], cache: dict, *,
                    computed: int, reused: int, cancelled: bool = False) -> dict:
    cached = sum(1 for path in images if path in cache)
    ready = sum(1 for path in images
                if path in cache and cache[path][0] == 'ok')
    failed = sum(1 for path in images
                 if path in cache and cache[path][0] == 'error')
    payload = {
        'ok': True,
        'engine': contract['engine'],
        'model_id': contract['model_id'],
        'revision': contract['revision'],
        'model_key': contract['model_key'],
        'dimension': contract['dimension'],
        'computed': computed,
        'reused': reused,
        'cached': cached,
        'ready': ready,
        'failed': failed,
        'remaining': max(0, len(images) - ready),
    }
    if cancelled:
        payload['cancelled'] = True
    return payload


def main() -> int:
    raw = sys.stdin.read()
    try:
        request = json.loads(raw) if raw.strip() else {}
        if not isinstance(request, dict):
            raise ValueError('request must be a JSON object')
        contract = _contract_from_request(request)
    except (json.JSONDecodeError, ValueError) as exc:
        _emit({'ok': False, 'results': {}, 'error': f'bad request: {exc}'})
        return 1

    # Preserve order while refusing duplicate work/cache rows.
    images = list(dict.fromkeys(str(path) for path in (request.get('images') or [])))
    if not images:
        _emit({'ok': False, 'results': {}, 'error': 'no images'})
        return 1
    cache_path = str(request.get('cache') or '').strip() or None
    cancel_file = str(request.get('cancel_file') or '').strip() or None
    rescore = bool(request.get('rescore') or request.get('rescan'))
    device_pref = str(request.get('device') or 'auto').strip().lower()
    if device_pref not in ('auto', 'cpu', 'cuda'):
        _emit({'ok': False, 'results': {},
               'error': 'device must be auto, cpu or cuda'})
        return 1

    # Claim (or sweep) whatever a previous run left behind before deciding what
    # is still to embed — see _salvage_cache.
    _salvage_cache(cache_path, contract)
    loaded = {} if rescore else _load_cache(cache_path, contract)
    # A whole-Bank payload is authoritative.  Removed paths must not survive in
    # a cache and inflate coverage after a resume.
    cache = {path: loaded[path] for path in images if path in loaded}
    pruned = set(cache) != set(loaded)
    todo = []
    for path in images:
        entry = cache.get(path)
        if entry is None or _is_stale(path, entry):
            cache.pop(path, None)
            todo.append(path)
    reused = len(images) - len(todo)
    _write_count(cache_path, reused)
    _log(f'[semantic] {len(images)} image(s), {reused} cached')

    if _cancel_requested(cancel_file):
        if pruned and cache_path:
            _flush_cache(cache_path, cache, contract)
        _write_count(cache_path, reused)
        _emit(_result_payload(contract, images, cache, computed=0,
                              reused=reused, cancelled=True))
        return 0

    if not todo:
        if pruned and cache_path:
            _flush_cache(cache_path, cache, contract)
        _emit(_result_payload(contract, images, cache, computed=0, reused=reused))
        return 0

    try:
        import numpy as np
        import torch
        from PIL import Image
        from transformers import AutoModel, AutoProcessor
    except Exception as exc:  # clean JSON instead of an invisible traceback
        _emit({'ok': False, 'results': {},
               'error': import_report.import_failure(exc)})
        return 1

    cuda_available = bool(torch.cuda.is_available())
    if device_pref == 'cuda' and not cuda_available:
        _emit({'ok': False, 'results': {},
               'error': 'CUDA was requested but is unavailable'})
        return 1
    device = ('cuda' if (device_pref == 'cuda'
                         or device_pref == 'auto' and cuda_available) else 'cpu')
    model_kwargs = {
        'revision': contract['revision'],
        'cache_dir': contract['models_root'],
        'local_files_only': True,
    }
    _phase(f'loading {contract["model_key"]} on {device.upper()} (local files only)')
    try:
        processor = AutoProcessor.from_pretrained(contract['model_id'], **model_kwargs)
        # AutoModel, not Siglip2Model. The pinned checkpoint is a SigLIP 2 model —
        # SigLIP 2 training, SigLIP 2 weights — but the FIXED-RESOLUTION variants
        # declare `model_type: siglip` and reuse the SigLIP 1 architecture; only the
        # NaFlex variants carry `model_type: siglip2`. Naming the class by hand made
        # transformers build a siglip2 shell around a siglip checkpoint and refuse
        # the weights outright: "copying a param with shape [768, 3, 16, 16] from
        # checkpoint, the shape in current model is [768, 768]" — a 16x16 patch
        # convolution against a linear projection, two different image front-ends.
        # The config names the class; let it.
        model = AutoModel.from_pretrained(contract['model_id'], **model_kwargs)
        model.to(device).eval()
    except Exception as exc:
        _emit({'ok': False, 'results': {},
               'error': f'SigLIP2 load failed: {type(exc).__name__}: {exc}'})
        return 1

    computed = ready_added = saved_since = 0
    zero = np.zeros(contract['dimension'], dtype='float32')
    if _cancel_requested(cancel_file):
        _write_count(cache_path, reused)
        _emit(_result_payload(contract, images, cache, computed=0,
                              reused=reused, cancelled=True))
        return 0

    for index, path in enumerate(todo, 1):
        signature = _file_sig(path)
        payload_hash = b''
        state = 'error'
        try:
            payload = read_validated_bank_image(path)
            payload_hash = hashlib.sha256(payload).digest()
            if not signature or _file_sig(path) != signature:
                raise RuntimeError('image changed while it was read')
            with Image.open(io.BytesIO(payload)) as image:
                image = image.convert('RGB')
                inputs = processor(images=image, return_tensors='pt')
                inputs = _move_to_device(inputs, device)
                with torch.no_grad():
                    embedding = _pooled_features(model.get_image_features(**inputs))
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                vector = embedding.cpu().numpy()[0].astype('float32')
            if vector.shape != (contract['dimension'],) or not np.isfinite(vector).all():
                raise RuntimeError('model returned the wrong embedding dimension')
            if not np.isclose(np.linalg.norm(vector), 1.0, rtol=1e-3, atol=1e-4):
                raise RuntimeError('model returned an unnormalised embedding')
            if (_file_sig(path) != signature
                    or _file_hash(path, signature) != payload_hash):
                raise RuntimeError('image changed while it was embedded')
            state = 'ok'
            cache[path] = (state, vector, signature, payload_hash)
        except Exception as exc:  # one bad image never sinks the whole Bank pass
            if not payload_hash and signature:
                payload_hash = _file_hash(path, signature)
            if (signature and payload_hash and _file_sig(path) == signature
                    and _file_hash(path, signature) == payload_hash):
                cache[path] = ('error', zero.copy(), signature, payload_hash)
            else:
                cache.pop(path, None)
            _log(f'[semantic] {index}/{len(todo)} ERROR {exc}')
        finally:
            computed += 1
            saved_since += 1
            if state == 'ok':
                ready_added += 1
            if cache_path and saved_since >= CACHE_EVERY:
                _flush_cache(cache_path, cache, contract)
                _write_count(cache_path, reused + ready_added)
                saved_since = 0
        if state == 'ok':
            _log(f'[semantic] {index}/{len(todo)} ok')
        if _cancel_requested(cancel_file):
            if cache_path:
                _flush_cache(cache_path, cache, contract)
            _write_count(cache_path, reused + ready_added)
            _emit(_result_payload(contract, images, cache, computed=computed,
                                  reused=reused, cancelled=True))
            return 0

    if cache_path:
        _phase(f'saving the semantic cache ({len(cache)} image(s))…')
        _flush_cache(cache_path, cache, contract)
        _write_count(cache_path, reused + ready_added)
    _emit(_result_payload(contract, images, cache, computed=computed, reused=reused))
    return 0


if __name__ == '__main__':
    sys.exit(main())
