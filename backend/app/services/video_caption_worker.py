"""Drive the Qwen3-VL caption worker for the length of ONE pass.

The parent half of ``infer/video_caption_infer.py``, and deliberately the same
shape as ``clip_image_encoder.ImageEncoder``: lazily started, scoped to a
``with`` block, no cache, no idle reaper. A pass owns its worker and gives the
memory back when it ends — cancelled or not. The model is 8.3 GB, so holding it
past the job would be pure loss, and there is no "the user might caption again in
a minute" window to keep it warm for.

Started LAZILY on the first clip: a re-run that finds everything already
captioned — the common case — must not pay a model load to discover it has no
work.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading

from .. import config as cfg
from .clip_text_encoder import TextEncodeError, _readline_with_timeout

_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'video_caption_infer.py')

# 8.3 GB of weights off a spinning disk, through an antivirus, on a cold cache.
# Generous on purpose: a timeout here reads as "captioning is broken" about a
# machine that was merely slow.
START_TIMEOUT = 1800
# One shot's caption. Seconds on a GPU, minutes on a CPU with a 4B model —
# anything past this means the worker is wedged, not thinking.
CAPTION_TIMEOUT = 600


class CaptionError(TextEncodeError):
    """The worker could not produce a caption, carrying the child's own words.

    A subclass so one ``except TextEncodeError`` in a route keeps answering 503
    for every model worker in this lane: the user-facing meaning ("this install
    cannot run the model") is identical."""


def unavailable_reason():
    """None when captions CAN be produced here, else a sentence saying why not.

    The same interpreter probe ✨ Score and the CLIP towers use — one environment
    with torch and transformers serves all of them, and asking the user for a
    second copy is what scoring_python.py exists to avoid."""
    from .clip_text_encoder import unavailable_reason as text_reason
    reason = text_reason()
    if reason is None:
        return None
    return reason.replace('text search', 'video captioning')


class CaptionWorker:
    """A warm Qwen3-VL, alive for as long as the ``with`` block."""

    def __init__(self, *, use_gpu=False, models_root=None, max_new_tokens=96,
                 model=None):
        # None = let the child fall back to its own default. The parent normally
        # passes the configured id explicitly, so the two halves cannot disagree
        # about which checkpoint a caption came from.
        self.model = (model or '').strip() or None
        self.use_gpu = bool(use_gpu)
        self.models_root = models_root if models_root is not None else (
            cfg.get('bank_scoring.models_root') or None)
        self.max_new_tokens = int(max_new_tokens)
        self.device = None
        # What the child reports having ACTUALLY loaded, once it is up.
        self.loaded_model = None
        self._proc = None
        self._lock = threading.RLock()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        """Terminate the worker and give its 8.3 GB back. Idempotent."""
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()          # EOF = the child's clean exit path
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    def _start(self):
        python = cfg.get('bank_scoring.python') or sys.executable
        env = dict(os.environ)
        env['PYTHONUTF8'] = '1'
        if not self.use_gpu:
            # Belt and braces with the child, which hides CUDA again before it
            # imports torch. Two locks on the same door because the failure —
            # an hour-long pass taking the card from a training run — is silent.
            env['CUDA_VISIBLE_DEVICES'] = ''
        try:
            proc = subprocess.Popen(
                [python, _SCRIPT], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, encoding='utf-8',
                errors='replace', bufsize=1, env=env,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except Exception as e:  # noqa: BLE001
            raise CaptionError(f'could not start the caption model: '
                               f'{type(e).__name__}: {e}') from None
        try:
            proc.stdin.write(json.dumps({
                'models_root': self.models_root,
                'device': 'auto' if self.use_gpu else 'cpu',
                'max_new_tokens': self.max_new_tokens,
                'model': self.model}) + '\n')
            proc.stdin.flush()
            data = json.loads(_readline_with_timeout(proc, START_TIMEOUT))
        except TextEncodeError:
            _kill(proc)
            raise
        except Exception:  # noqa: BLE001
            _kill(proc)
            raise CaptionError('the caption model produced no result — check the '
                               '✨ Score interpreter') from None
        if not data.get('ok') or not data.get('ready'):
            _kill(proc)
            raise CaptionError(str(data.get('error') or 'unknown caption error'))
        self.device = data.get('device') or 'cpu'
        # What the child ACTUALLY loaded, echoed back — the authority on which
        # checkpoint wrote this run's captions, rather than what we asked for.
        self.loaded_model = data.get('model') or self.model
        self._proc = proc
        return proc

    def caption(self, frame_paths, prompt, span_s=None):
        """The caption for one shot, or '' when the model refused THIS shot.

        '' rather than an exception for a per-clip refusal, so the caller stores
        an 'error' state and moves on. An exception is reserved for the worker
        itself being gone or wedged — the one condition that must stop the pass
        rather than be absorbed 400 times in a row."""
        with self._lock:
            proc = self._proc if (self._proc is not None
                                  and self._proc.poll() is None) else None
            if proc is None:
                proc = self._start()
            try:
                req = {'frames': [str(p) for p in frame_paths],
                       'prompt': str(prompt)}
                if span_s:
                    # Seconds the frames span — lets the child stamp honest
                    # timestamps instead of transformers' 24 fps default.
                    req['span_s'] = float(span_s)
                proc.stdin.write(json.dumps(req) + '\n')
                proc.stdin.flush()
                data = json.loads(_readline_with_timeout(proc, CAPTION_TIMEOUT))
            except TextEncodeError:
                self.close()                # a wedged worker must not be reused
                raise
            except Exception:  # noqa: BLE001
                self.close()
                raise CaptionError('the caption model stopped responding') from None
        if not data.get('ok'):
            return ''
        return str(data.get('caption') or '')


def _kill(proc):
    try:
        proc.kill()
    except Exception:  # noqa: BLE001
        pass
