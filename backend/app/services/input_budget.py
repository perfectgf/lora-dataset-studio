"""The app side of the shared image input budget.

`image_encoding` holds the guard but may not read config: it is loaded BY PATH
under the dedicated ML interpreter, where the Flask app package does not exist.
So the dependency runs the other way — this module reads the setting and
installs itself into `image_encoding` as a provider, at import. Anything that
imports an app service therefore gets the configured budget; anything that
loads `image_encoding.py` standalone keeps its shipped defaults, which is the
conservative side of the split and is stated in the Guide rather than implied.

The provider is a live read, not a snapshot: saving Settings changes the budget
for the next image, with no restart. That is what keeps import, Bank scan,
thumbnailing, edits, ComfyUI staging and Ollama vision quoting ONE number —
the failure this budget's original comment warned about is an image you can
import but not look at.
"""
from __future__ import annotations

import logging

from PIL import Image

from .. import config as cfg
from . import image_encoding

logger = logging.getLogger(__name__)

#: Pillow's own shipped decompression-bomb threshold, captured before anything
#: here can move it. Several call sites promote `DecompressionBombWarning` to an
#: exception, so a budget larger than Pillow's would be silently unreachable —
#: the user would raise the setting and still be refused, by a different guard
#: with a different message.
_PILLOW_DEFAULT_MAX_PIXELS = Image.MAX_IMAGE_PIXELS


def _coerce(raw, default: int, key: str) -> int:
    """A non-negative int, or the shipped default with a line in the log."""
    try:
        value = int(raw)
        if value < 0:
            raise ValueError(raw)
    except (TypeError, ValueError):
        logger.warning('ignoring unusable %s %r', key, raw)
        return int(default)
    return value


def input_image_budget() -> tuple[int, int]:
    """The configured `(max_side, max_pixels)`; 0 on either means no limit."""
    defaults = cfg.DEFAULTS['image_input']
    max_side = _coerce(cfg.get('image_input.max_side', defaults['max_side']),
                       defaults['max_side'], 'image_input.max_side')
    max_pixels = _coerce(cfg.get('image_input.max_pixels', defaults['max_pixels']),
                         defaults['max_pixels'], 'image_input.max_pixels')
    _align_pillow_bomb_threshold(max_pixels)
    return max_side, max_pixels


def _align_pillow_bomb_threshold(max_pixels: int) -> None:
    """Keep Pillow's process-wide bomb threshold from overruling our budget.

    Only ever RAISES it (or removes it, for an unlimited budget): a smaller
    configured budget is enforced by our own header check with our own message,
    and lowering Pillow's number would change behaviour for callers that never
    asked. This is a process-global write on purpose — the budget itself is
    process-global by design.
    """
    target = None if max_pixels == 0 else max(_PILLOW_DEFAULT_MAX_PIXELS or 0, max_pixels)
    if Image.MAX_IMAGE_PIXELS != target:
        Image.MAX_IMAGE_PIXELS = target


def infer_worker_env() -> dict:
    """The configured budget as ENVIRONMENT for a `backend/infer/*` subprocess.

    The provider trick at the bottom of this module only works in-process:
    `image_encoding` is loaded by path under the ML interpreter, but an infer
    WORKER is a different process entirely, and its own guard
    (`backend/infer/bank_image_guard.py`) cannot import the Flask app. Passing the
    two numbers down as environment is the only channel that crosses that line —
    without it the worker enforces its shipped 16 Mi-pixels / 8192 px while the
    app enforces the configured budget, and an image the user can import, view and
    train on is refused by the captioner alone.

    Callers merge this into the subprocess env; a worker launched without it keeps
    its own conservative defaults.
    """
    max_side, max_pixels = input_image_budget()
    return {'LDS_INFER_MAX_SIDE': str(max_side),
            'LDS_INFER_MAX_PIXELS': str(max_pixels)}


def unlimited_warning(max_side: int, max_pixels: int) -> str:
    """What a 0 actually disarms, in one sentence, said where it is chosen."""
    if max_side and max_pixels:
        return ''
    return ('No limit means a corrupt or hostile file can be decoded until it '
            'fills memory: a few hundred header bytes can claim billions of '
            'pixels, and the app has no second guard behind this one.')


image_encoding.set_input_budget_provider(input_image_budget)
