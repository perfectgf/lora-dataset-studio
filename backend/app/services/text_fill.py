"""🔤 The text-zone filler's parent half — one batch child, per-image results.

The cleaning levels call this BEFORE LaMa on every row the text pass flagged:
the child empties speech bubbles in place (outline-safe, see
infer/text_fill_infer.py for the method and its measurements) and returns the
glyph-tight boxes of whatever sat on busy art — which is all LaMa still gets,
instead of the whole rectangle that used to eat balloon outlines.

Runs in the `video_text` interpreter: the child needs cv2 + numpy only, both
of which that capability's probe already imports — no new capability, no new
install, and a machine that can FIND text can always fill it.
"""
from __future__ import annotations
import json
import logging

from .. import config as cfg

logger = logging.getLogger(__name__)

_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'text_fill_infer.py')

# Per-image budget plus a cold-start floor. The work is thresholds and
# connected components — milliseconds per zone — so the floor dominates; it
# covers a slow disk re-reading a big staged page, never a model load.
_TIMEOUT_PER_ITEM_S = 4
_TIMEOUT_FLOOR_S = 120


def is_available() -> bool:
    """Gated on the SAME capability as the text scan: cv2 and numpy arrive with
    `video_text`, and a row can only be text-flagged if that extra ran."""
    from ..capabilities import probe_video_text
    return bool(probe_video_text().get('ok'))


def fill_batch(items, *, timeout=None, should_stop=None):
    """{image_path: {'ok', 'filled', 'busy_boxes', 'error'?}} for a batch.

    ``items`` is [{'image_path', 'regions'}] with normalised [x0,y0,x1,y1]
    zones — the mask channel's own shape. Images are rewritten IN PLACE, so
    callers hand STAGED copies, never user files. Raises RuntimeError with the
    child's own words when the child could not run at all; per-image failures
    come back as result rows instead, so one unreadable page never sinks a
    batch (the exact split read_text_boxes uses).
    """
    from .infer_stream import run_infer_script, stderr_tail
    from .video_safe_zone import _stop_plumbing
    if not items:
        return {}
    python = cfg.get('video_text.python') or _default_python()
    budget = timeout or max(_TIMEOUT_FLOOR_S,
                            _TIMEOUT_PER_ITEM_S * len(items))
    cancel_path, ask_stop, cleanup = _stop_plumbing()
    payload = json.dumps({'items': items, 'cancel_file': cancel_path}) + '\n'
    try:
        stdout, stderr_lines, rc, timed_out = run_infer_script(
            python, _SCRIPT, payload, budget,
            should_stop=should_stop, on_stop=ask_stop)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f'could not start the text filler: '
                           f'{type(e).__name__}: {e}') from None
    finally:
        cleanup()
    data = {}
    for text in reversed((stdout or '').strip().splitlines()):
        if text.lstrip().startswith('{'):
            try:
                data = json.loads(text)
            except ValueError:
                data = {}
            break
    if not data:
        logger.warning('text fill: no JSON from the filler (rc=%s, timed_out=%s) '
                       'stderr=%s', rc, timed_out, stderr_tail(stderr_lines))
        raise RuntimeError('the text filler produced no result — check the '
                           'burned-in text extra in Setup')
    if not data.get('ok'):
        raise RuntimeError(str(data.get('error') or 'unknown text-filler error'))
    return {path: dict(row) for path, row in (data.get('results') or {}).items()}


def _default_python():
    import sys
    return sys.executable
