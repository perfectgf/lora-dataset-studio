"""Warm CPU SigLIP2 text tower paired with ``bank_semantic_infer.py``.

The first stdin line is the app-provided model/provenance handshake; subsequent
lines are ``{"text": "..."}`` queries.  Model assets are always resolved from
the pinned local cache and this worker never takes CUDA.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any
from _harness import _pooled_features

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import import_report  # noqa: E402


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _emit(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def _contract_from_handshake(request: dict) -> dict:
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
        'engine': str(merged.get('engine') or '').strip(),
        'model_id': str(model_id or '').strip(),
        'revision': str(merged.get('revision') or '').strip(),
        'model_key': str(merged.get('model_key') or '').strip(),
        'models_root': str(models_root or '').strip(),
    }
    try:
        contract['dimension'] = int(merged.get('dimension'))
    except (TypeError, ValueError) as exc:
        raise ValueError('dimension must be an integer') from exc
    if contract['engine'] != 'siglip2':
        raise ValueError('SigLIP2 text worker requires engine=siglip2')
    for key in ('model_id', 'revision', 'model_key', 'models_root'):
        if not contract[key]:
            raise ValueError(f'missing semantic model contract field: {key}')
    if contract['dimension'] <= 0:
        raise ValueError('semantic embedding dimension must be positive')
    return contract


def _move_to_cpu(inputs: Any) -> Any:
    if hasattr(inputs, 'to'):
        return inputs.to('cpu')
    if isinstance(inputs, dict):
        return {key: value.to('cpu') if hasattr(value, 'to') else value
                for key, value in inputs.items()}
    return inputs


def main() -> int:
    raw = sys.stdin.readline()
    try:
        request = json.loads(raw) if raw.strip() else {}
        if not isinstance(request, dict):
            raise ValueError('handshake must be a JSON object')
        contract = _contract_from_handshake(request)
    except (json.JSONDecodeError, ValueError) as exc:
        _emit({'ok': False, 'ready': False, 'error': f'bad handshake: {exc}'})
        return 1

    # Hide the card before torch is imported.  Encoding one query is tiny; using
    # CUDA here would only contend with training/vision work for no visible gain.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        import numpy as np
        import torch
        from transformers import AutoModel, AutoProcessor
    except Exception as exc:
        _emit({'ok': False, 'ready': False,
               'error': import_report.import_failure(exc)})
        return 1

    model_kwargs = {
        'revision': contract['revision'],
        'cache_dir': contract['models_root'],
        'local_files_only': True,
    }
    _log(f'[text] loading {contract["model_key"]} (CPU, local files only)…')
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
        model.to('cpu').eval()
    except Exception as exc:
        _emit({'ok': False, 'ready': False,
               'error': f'SigLIP2 load failed: {type(exc).__name__}: {exc}'})
        return 1

    def _encode(text: str):
        inputs = processor(
            text=[text], return_tensors='pt', padding='max_length',
            truncation=True, max_length=64)
        inputs = _move_to_cpu(inputs)
        with torch.no_grad():
            embedding = _pooled_features(model.get_text_features(**inputs))
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        vector = embedding.cpu().numpy()[0].astype('float32')
        if vector.shape != (contract['dimension'],) or not np.isfinite(vector).all():
            raise RuntimeError('model returned the wrong embedding dimension')
        if not np.isclose(np.linalg.norm(vector), 1.0, rtol=1e-3, atol=1e-4):
            raise RuntimeError('model returned an unnormalised embedding')
        return vector

    ready = {
        'ok': True,
        'ready': True,
        'engine': contract['engine'],
        'model_id': contract['model_id'],
        'revision': contract['revision'],
        'model_key': contract['model_key'],
        'dimension': contract['dimension'],
        # ``dim`` preserves the warm-worker protocol used by clip_text_infer.
        'dim': contract['dimension'],
    }
    _emit(ready)
    _log('[text] ready')

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError('query must be a JSON object')
        except (json.JSONDecodeError, ValueError) as exc:
            _emit({'ok': False, 'error': f'bad json: {exc}'})
            continue
        text = str(message.get('text') or '')
        if not text.strip():
            _emit({'ok': False, 'error': 'empty text'})
            continue
        try:
            vector = _encode(text)
        except Exception as exc:  # one bad query does not kill the warm worker
            _emit({'ok': False,
                   'error': f'text encoding failed: {type(exc).__name__}: {exc}'})
            continue
        _emit({
            'ok': True,
            'engine': contract['engine'],
            'model_key': contract['model_key'],
            'dimension': contract['dimension'],
            'vector': [float(value) for value in vector],
        })
    _log('[text] stdin closed — exiting')
    return 0


if __name__ == '__main__':
    sys.exit(main())
