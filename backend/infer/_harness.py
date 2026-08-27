"""Shared harness for the sibling infer scripts — stdlib ONLY, imported locally.

These scripts run in their own interpreters (the torch/ML venvs), are launched
as plain files (`python .../infer/xxx.py`), and never import `app` on purpose.
This module is their one shared layer and must stay importable everywhere they
run: standard library only — no torch, no numpy, no app.*, nothing a bare
interpreter lacks.

"Everywhere they run" includes interpreters that never put the script's own
directory on sys.path. `python script.py` normally makes this directory
sys.path[0] — but an embeddable Python (ComfyUI portable's python_embeded,
whose ._pth file pins sys.path and implies isolated mode) skips that step, and
a borrowed interpreter like that is exactly what the ✨ Score / semantic /
watermark pickers point passes at. So every sibling restores the directory
itself, right before its `from _harness import …` line
(test_infer_harness_contract.py pins the pattern). The parent cannot do it for
them: a ._pth interpreter ignores PYTHONPATH and every other environment
variable too.
(Under pytest and in the Flask process the resolution mechanism is
app/services/atomic_npz.py, which already appends this directory to sys.path —
the same door npz_atomic has always been imported through.)

Only zero-coupling helpers live here: each body reads no module globals and
calls no sibling helper, so `from _harness import x` keeps per-module
monkeypatching exact (a test that rebinds `mod._x` still intercepts every
internal caller in `mod`; nothing here calls back around a patch).

Deliberately NOT factored, measured before cutting (same doctrine as
app/services/video_pass_scaffold.py):
- the cache subsystems (`_file_sig`/`_file_hash`/`_is_stale`/`_salvage_cache` +
  `_load/_save/_flush_cache`): formats and signatures genuinely differ per
  script (bank_semantic's `_file_hash` takes the signature as an argument), and
  tests patch `_file_sig` per module expecting internal callers to see it — a
  harness-internal `_file_hash → _file_sig` call would dodge that patch.
- `_verdict` (face_embed/face_score): reads module-tuned thresholds
  (DET_MIN/FACE_PX_MIN/YAW_MAX).
- the divergent `_log`/`_emit` variants (mask/lama family, joycaption,
  bank_semantic/siglip2): their bodies really differ (stream, prefix, shape).
"""
import json
import os
import sys
from typing import Any


def _log(m):
    print(m, file=sys.stderr, flush=True)

def _emit(obj):
    print(json.dumps(obj), flush=True)

def _cancel_requested(cancel_file):
    """The parent drops this sentinel file to ask for a clean stop, so the pass
    flushes its cache and exits between images instead of being SIGKILLed
    mid-compute (which would lose up to CACHE_EVERY images)."""
    return bool(cancel_file) and os.path.exists(cancel_file)

def _write_count(cache_path, n):
    """Plain-text sidecar (``<cache>.count``) with how many images are scored so
    far. The Flask parent has no numpy to read the .npz, so this is how a stopped
    pass can still report an honest "N scored (M remaining)" — even in the rare
    case it had to be hard-killed before it could print its own cancel line."""
    if not cache_path:
        return
    try:
        with open(cache_path + '.count', 'w', encoding='utf-8') as f:
            f.write(str(int(n)))
    except OSError:
        pass

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
