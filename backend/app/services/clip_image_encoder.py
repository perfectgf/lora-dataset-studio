"""Drive the CLIP image worker for the length of ONE pass, then give the RAM back.

The parent half of ``infer/clip_image_embed_infer.py``. Deliberately a much
smaller object than ``clip_text_encoder``, and the difference is the point:

* **No persistent cache.** A text query repeats constantly (people refine a
  phrase and re-type favourites), so caching vectors on disk pays forever. A
  frame is embedded exactly once, by the pass that owns it, and its vector is
  stored by the caller — a cache here would hold a copy of a file we already
  wrote and never read it again.
* **No idle reaper.** The worker is scoped to a pass with a context manager. It
  starts when the pass starts and dies when the pass ends, cancelled or not.
  There is no "the user might search again in a minute" window to keep it warm
  for, and the model is ~2.4 GB — holding it past the job would be pure loss.
* **The GPU is asked for.** Text encoding is microseconds of compute behind an
  8 s load, so ``clip_text_encoder`` forces the CPU and never races a training
  run. Embedding thousands of frames is the opposite: ~336 ms/frame on CPU
  against ~15 ms on a card. The caller decides, takes the GPU-exclusive window
  when it decides yes, and gets a child that hid CUDA from itself when it
  decides no.

The interpreter is the ✨ Score one (``bank_scoring.python``) — the very
environment that produced every image embedding this project owns, which is what
makes a video frame and a bank image land in the same space.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading

from .. import config as cfg
from . import infer_env
from .clip_text_encoder import TextEncodeError, _readline_with_timeout

_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'clip_image_embed_infer.py')

# Same generosity as the text worker: a cold `import torch` plus ~1.6 GB of
# weights on a machine that is also antivirus-scanning them is minutes, and a
# timeout here would read as "the pass is broken" about an interpreter that was
# merely slow.
START_TIMEOUT = 900
# Once warm, a frame is 15-350 ms depending on the device. Anything past this
# means the worker is wedged, not busy.
FRAME_TIMEOUT = 180


class ImageEncodeError(TextEncodeError):
    """The worker could not produce a vector, carrying the child's own words.

    A subclass rather than a sibling so one ``except TextEncodeError`` in a route
    keeps answering 503 for BOTH halves of CLIP — the user-facing meaning ("this
    install cannot run the model") is identical, and splitting it would only
    create a path where one of the two returns a 500."""


def unavailable_reason():
    """None when frames CAN be embedded here, else a sentence saying why not.

    The same probe ✨ Score and text search use, because it is the same
    interpreter and the same checkpoint: if one cannot run, none of them can."""
    from .clip_text_encoder import unavailable_reason as text_reason
    reason = text_reason()
    if reason is None:
        return None
    return reason.replace('text search', 'frame embedding')


class ImageEncoder:
    """A warm CLIP image tower, alive for as long as the ``with`` block.

    Started LAZILY on the first frame: a pass that finds nothing to do (every
    clip already embedded — the common case on a re-run) must not pay an 8 s
    model load to discover it has no work."""

    def __init__(self, *, use_gpu=False, models_root=None):
        self.use_gpu = bool(use_gpu)
        self.models_root = models_root if models_root is not None else (
            cfg.get('bank_scoring.models_root') or None)
        self.device = None
        self.dim = None
        self._proc = None
        self._lock = threading.RLock()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        """Terminate the worker and give its ~2.4 GB back. Idempotent."""
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
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    def _start(self):
        python = cfg.get('bank_scoring.python') or sys.executable
        # Isolated from the process owner's user site-packages, exactly like the
        # probe that said this interpreter was ready — see services/infer_env.
        # An unrelated package left there by another project is how this worker
        # failed to start 855 times in a row while every dependency it names was
        # correctly installed.
        env = infer_env.worker_env(python, PYTHONUTF8='1')
        if not self.use_gpu:
            # Belt and braces with the child, which hides CUDA again before it
            # imports torch. Two locks on the same door because the failure this
            # prevents — an hour-long pass stealing the card from a training run
            # — is silent and expensive.
            env['CUDA_VISIBLE_DEVICES'] = ''
        try:
            proc = subprocess.Popen(
                infer_env.worker_argv(python, _SCRIPT),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, encoding='utf-8',
                errors='replace', bufsize=1, env=env,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except Exception as e:  # noqa: BLE001
            raise ImageEncodeError(f'could not start the frame encoder: '
                                   f'{type(e).__name__}: {e}') from None
        try:
            proc.stdin.write(json.dumps({
                'models_root': self.models_root,
                'device': 'auto' if self.use_gpu else 'cpu'}) + '\n')
            proc.stdin.flush()
            data = json.loads(_readline_with_timeout(proc, START_TIMEOUT))
        except TextEncodeError:
            _kill(proc)
            raise
        except Exception:  # noqa: BLE001
            _kill(proc)
            raise ImageEncodeError('the frame encoder produced no result — check '
                                   'the ✨ Score interpreter') from None
        if not data.get('ok') or not data.get('ready'):
            _kill(proc)
            raise ImageEncodeError(str(data.get('error') or 'unknown encoder error'))
        self.device = data.get('device') or 'cpu'
        self.dim = data.get('dim') or None
        self._proc = proc
        return proc

    def encode(self, paths):
        """[vector | None] for `paths`, in order. None marks a frame the worker
        refused (an unreadable JPEG) — one bad frame costs that frame, never the
        pass, exactly like a bad file in every other pass of this lane.

        Raises ImageEncodeError only when the WORKER itself is gone or wedged,
        which is the one condition the caller must not paper over."""
        import numpy as np
        out = []
        with self._lock:
            proc = self._proc if (self._proc is not None
                                  and self._proc.poll() is None) else None
            if proc is None:
                proc = self._start()
            for path in paths:
                try:
                    proc.stdin.write(json.dumps({'image': str(path)}) + '\n')
                    proc.stdin.flush()
                    data = json.loads(_readline_with_timeout(proc, FRAME_TIMEOUT))
                except TextEncodeError:
                    self.close()             # a wedged worker must not be reused
                    raise
                except Exception:  # noqa: BLE001
                    self.close()
                    raise ImageEncodeError('the frame encoder stopped responding') \
                        from None
                if not data.get('ok'):
                    out.append(None)
                    continue
                vec = np.asarray(data.get('vector') or [], dtype='float32')
                out.append(vec if vec.size else None)
        return out


def _kill(proc):
    try:
        proc.kill()
    except Exception:  # noqa: BLE001
        pass
