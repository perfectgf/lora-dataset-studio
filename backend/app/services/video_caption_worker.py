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
import logging
import subprocess
import sys
import threading

logger = logging.getLogger(__name__)

from .. import config as cfg
from . import infer_env
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

    def __init__(self, *, use_gpu=False, models_root=None, max_new_tokens=400,
                 model=None, tokenizer_dir=None):
        # None = let the child fall back to its own default. The parent normally
        # passes the configured id explicitly, so the two halves cannot disagree
        # about which checkpoint a caption came from.
        self.model = (model or '').strip() or None
        self.use_gpu = bool(use_gpu)
        self.models_root = models_root if models_root is not None else (
            cfg.get('bank_scoring.models_root') or None)
        self.max_new_tokens = int(max_new_tokens)
        # Folder holding umT5's spiece.model, or None: with it the child measures
        # every caption in the encoder's own tokens (C12-C); without it the
        # parent estimates from words and says so.
        self.tokenizer_dir = (tokenizer_dir or '').strip() or None
        # Which counter the child managed to load ('sentencepiece',
        # 'transformers' or None), and the count for the LAST caption returned.
        self.token_counter = None
        self.last_tokens = None
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
        # No user site-packages, and the same for the probe - see infer_env.
        env = infer_env.worker_env(python, PYTHONUTF8='1')
        if not self.use_gpu:
            # Belt and braces with the child, which hides CUDA again before it
            # imports torch. Two locks on the same door because the failure —
            # an hour-long pass taking the card from a training run — is silent.
            env['CUDA_VISIBLE_DEVICES'] = ''
        try:
            proc = subprocess.Popen(
                infer_env.worker_argv(python, _SCRIPT),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
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
                'model': self.model,
                'tokenizer_dir': self.tokenizer_dir}) + '\n')
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
        self.token_counter = data.get('token_counter') or None
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
            # The reason used to die here: the pass stored caption_state='error'
            # and nobody could say why (a whole bench came back "0 words" with
            # nothing to read). Logged, never raised — a per-shot refusal must
            # still be absorbed, but it must be absorbed OUT LOUD.
            logger.warning('caption worker refused a shot: %s',
                           data.get('error') or 'no reason given')
            self.last_tokens = None
            return ''
        tokens = data.get('tokens')
        self.last_tokens = tokens if isinstance(tokens, int) and tokens >= 0 else None
        return str(data.get('caption') or '')


def frames_time_preamble(n_frames, span_s) -> str:
    """One sentence telling the model WHEN each frame sits in the shot, or ''.

    The transformers worker hands real VideoMetadata to the processor; an HTTP
    vision server has no such channel, so time rides in words. Measured
    (refutation bench, 2026-09-01): textual stamps are accepted and change
    nothing else — while with NO timing the model reads N stills and every
    judgement of speed and duration is a guess."""
    try:
        n = int(n_frames)
        span = float(span_s or 0)
    except (TypeError, ValueError):
        return ''
    if n < 2 or span <= 0:
        return ''
    stamps = ', '.join(f'{span * i / (n - 1):.1f}s' for i in range(n))
    return (f'The {n} frames are sampled evenly across a {span:.1f}-second '
            f'shot, at {stamps}. ')


class LocalLlmCaptionWorker:
    """CaptionWorker's contract, served by the local LLM the user already runs.

    Same seam, same rules: ``caption() -> str``, '' for a per-shot refusal
    (logged, absorbed by the pass as caption_state='error'), and the historic
    empty-response failure guarded the same way — an empty answer is never
    stored as a caption. ``last_tokens`` stays None: there is no umT5 tokenizer
    behind an HTTP server, so the export preflight falls back to its labelled
    estimate, exactly as designed.

    The engine and model come from the settings the image passes already obey
    (local_llm.provider, ollama.vision_model / lmstudio.vision_model) — which
    server and which tag are ONE fact each, said once."""

    def __init__(self, *, provider=None, model=None):
        from . import vision_llm
        self.provider = (provider or vision_llm.provider()).strip().lower()
        self.label = vision_llm.label(self.provider)
        self.model = (model or '').strip() or None
        # What caption_model records: engine-prefixed, so a bank captioned
        # across engines stays readable row by row.
        self.loaded_model = (f'{self.provider}:{self.model}' if self.model
                             else self.provider)
        self.device = self.label
        self.token_counter = None
        self.last_tokens = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        """Give the server's VRAM back, like the image batches do after a run.

        Scoped to OUR provider and OUR model: the bare call released every
        tracked model on the server, which took a concurrent image pass's
        checkpoint with it (review finding 8). The provider is the one pinned
        at construction — a config flip mid-pass must not reroute the unload."""
        try:
            if self.provider == 'lmstudio':
                from . import vision_lmstudio
                vision_lmstudio.unload_vision_model(model=self.model)
            else:
                from . import vision_ollama
                vision_ollama.unload_vision_model(model=self.model)
        except Exception:  # noqa: BLE001 — a failed unload must not fail the pass
            pass

    def caption(self, frame_paths, prompt, span_s=None):
        from . import vision_image, vision_llm
        frames = []
        for p in frame_paths:
            try:
                with open(p, 'rb') as fh:
                    raw = fh.read()
            except OSError as e:
                logger.warning('caption: frame unreadable (%s): %s', p, e)
                continue
            # The drivers' safety gate, applied HERE first: they drop a frame
            # that fails it in silence, and a time preamble built on the
            # pre-drop count then hands the model a FALSE grid — worse than no
            # preamble at all (review finding 9). Gated up front, the preamble
            # counts exactly what the model will see; the drivers re-gate the
            # already-safe JPEGs at negligible cost.
            safe = vision_image.ensure_vision_safe_jpeg(
                raw, provider='video_caption')
            if safe is None:
                logger.warning('caption: frame dropped as unsafe/unreadable (%s)', p)
                continue
            frames.append(safe)
        # One surviving frame is a still, not a shot — captioning it as a video
        # would describe motion nobody saw. Refused out loud, like every engine
        # refusal on this seam.
        if len(frames) < 2:
            logger.warning('caption worker refused a shot: only %d of %d frames '
                           'were readable', len(frames), len(frame_paths))
            self.last_tokens = None
            return ''
        full_prompt = frames_time_preamble(len(frames), span_s) + str(prompt)
        # 800, not 600: the paragraph and the labelled tail compete for one
        # generation cap and the tail is written LAST — at 600 a chatty model
        # hits the cap inside the tail, and a mutilated tail is what the
        # export's short-form floor exists to refuse (review finding,
        # 2026-09-01). A model that finishes early stops at its EOS; headroom
        # costs nothing.
        # provider= and model= pin THIS run's engine on every request: without
        # them the waist re-reads the config per shot, and half a bank could be
        # captioned by whatever the user loaded mid-pass while every row still
        # claimed the gate's model (review finding 7). The read timeout matches
        # the transformers lane's CAPTION_TIMEOUT — this lane runs on the
        # least-equipped machines, and 300 s cut off honest CPU-offload shots.
        text = str(vision_llm.describe_frames(
            frames, full_prompt, provider=self.provider, model=self.model,
            num_predict=800, keep_alive='5m',
            timeout=(10, CAPTION_TIMEOUT)) or '').strip()
        self.last_tokens = None
        if not text:
            # The reason was already logged by the driver; this line is the one
            # a caption_state='error' row leads back to.
            logger.warning('caption worker refused a shot: %s returned an '
                           'empty caption', self.label)
            return ''
        return text


def _kill(proc):
    try:
        proc.kill()
    except Exception:  # noqa: BLE001
        pass
