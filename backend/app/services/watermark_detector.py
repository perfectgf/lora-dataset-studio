"""The dedicated watermark detector — parent half.

WHY this exists at all. Finding watermarks used to mean asking a chat model, in
words, "is there a watermark in this picture?", once per image, and parsing the
sentence it wrote back. That works, and it is what still runs when this extra is
not installed — but it costs ~1.7 s per image for a question whose answer is "no"
the overwhelming majority of the time. On the reference bank of 29 759 images
that is roughly fifteen hours of GPU. A discriminative classifier answers the
same binary question in milliseconds, and it answers it with a NUMBER, which is
what makes a threshold (and therefore a measured calibration) possible.

The cascade is deliberately two models with two licences we checked at the source
(``backend/infer/watermark_detect_infer.py`` lists every dependency's licence):
SigLIP2 ranks, Grounding DINO locates only what the ranker flagged. The research
picked Florence-2 for that second stage and this build does NOT use it: its
modelling code ships in its own repo and no longer loads on current transformers,
and pinning transformers down would endanger the bank-scoring stack that shares
the environment — the full reason is recorded in ``watermark_detect_infer.py``,
where a test keeps it alive so the next reader does not "fix" it back.
`ultralytics` is never used — it claims AGPL-3.0 over trained weights and would
contaminate this public repository.

Contract with the caller: `scan()` is a GENERATOR yielding one result per image,
in input order, as they arrive. That shape is not decoration — the Bank pass
commits per image so a stopped scan keeps everything it already learned, and a
30 000-image batch that only reported at the end could not do that.

Fail-open, exactly like today's pass: no extra installed → `available()` is False
and the caller keeps using the vision model. A locator that cannot load →
images are still flagged, just without a box (the state the pre-box builds
shipped), and the summary says so.
"""
from __future__ import annotations
import json
import logging
import subprocess
import sys
import threading

from .. import config as cfg
from . import infer_env

logger = logging.getLogger(__name__)

_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'watermark_detect_infer.py')

# The two model repos, their licences, and the EXACT files worth downloading.
#
# Duplicated on purpose: the child script (which runs in another interpreter and
# may not import this package) declares the same ids, and
# test_watermark_detector.py asserts the two never drift. That is the same
# safeguard CAPABILITY_IMPORTS documents — an installer and a probe that disagree
# produce an install reporting success over a capability that stays ✗.
#
# `files` is not decoration either, it is MEASURED: a plain snapshot_download of
# the SigLIP2 repo pulls 2.4 GB because the author published the training
# checkpoints (checkpoint-712/, checkpoint-1424/, optimizer.pt) next to a model
# that weighs 371 MB, and the Grounding DINO repo would add a duplicate
# pytorch_model.bin. Naming the files brings the install down to ~0.9 GB.
MODEL_FILES = {
    'prithivMLmods/Watermark-Detection-SigLIP2': {
        'license': 'Apache-2.0',
        'role': 'ranks each image (watermarked / clean)',
        'files': ('config.json', 'preprocessor_config.json', 'model.safetensors'),
        'approx_mb': 372,
    },
    'IDEA-Research/grounding-dino-tiny': {
        'license': 'Apache-2.0',
        'role': 'locates the mark on the images the ranker flagged',
        'files': ('config.json', 'preprocessor_config.json', 'model.safetensors',
                  'tokenizer.json', 'tokenizer_config.json',
                  'special_tokens_map.json', 'added_tokens.json', 'vocab.txt'),
        'approx_mb': 690,
    },
}
MODEL_REPOS = tuple(MODEL_FILES)
DOWNLOAD_MB = sum(m['approx_mb'] for m in MODEL_FILES.values())

# How many images one child process handles before the parent starts a fresh one.
# The models load once per process (~10 s), so this must be big; it exists only so
# a leaked tensor or a fragmented allocator cannot accumulate over a bank of tens
# of thousands of images.
BATCH = 2000


def detector_python() -> str:
    """The interpreter that runs the detector.

    Dedicated key first, then the bank-scoring environment — which is not a
    fallback but the COMMON case: that venv already carries torch and
    transformers, and building a second one would cost the user another ~2.5 GB
    to hold a second copy of the same packages. Then the app's own Python, which
    will simply probe False on a normal install and keep the vision-model path.
    """
    return (cfg.get('watermark_detect.python')
            or cfg.get('bank_scoring.python')
            or sys.executable)


def available() -> bool:
    from ..capabilities import probe_watermark_detect
    return bool(probe_watermark_detect().get('ok'))


# The three values of `watermark_detect.backend`, in the shape the captioning
# backend setting already uses ('auto' first, then one id per engine). Same word,
# same order, same meaning — a second vocabulary for the same idea is how two
# screens end up obeying two rules, which is the very thing this setting fixes.
BACKENDS = ('auto', 'detector', 'vision')

# Where the extra is installed from, named in full because a fallback notice that
# does not say what to click is just an apology.
INSTALL_ROUTE = 'Setup ▸ Quality tools ▸ Watermark detector'


def configured_backend() -> str:
    """The persisted choice, normalised. Anything unknown (a hand-edited config)
    reads as 'auto' rather than aborting a scan."""
    value = str(cfg.get('watermark_detect.backend') or 'auto').strip().lower()
    return value if value in BACKENDS else 'auto'


def resolve_backend(requested: str | None = None) -> dict:
    """Which route a watermark scan will actually take, and whether that is what
    the user asked for.

    Returns ``{'requested', 'backend', 'fell_back', 'detail'}`` where ``backend``
    is 'detector' or 'vision' — never 'auto', because 'auto' is a question, not a
    route.

    The three cases, and why they differ:

    * ``auto`` (the default) — the detector when its extra probes ready, the
      vision model otherwise. This is byte-for-byte what the bank has always
      done, so leaving the setting alone changes nothing on either surface.
      ``fell_back`` stays False: nothing was overridden, the user asked for
      "whatever is fastest".
    * ``detector`` pinned, extra missing — the pass STILL RUNS, on the vision
      route, and ``fell_back`` is True with the probe's own sentence in
      ``detail``. Refusing would be worse (a scan the user wanted, not run) and
      running silently would be worse still: the user would change a detector and
      see nothing change.
    * ``vision`` pinned — the vision model, always. No probe, no surprise.
    """
    requested = (requested or configured_backend())
    requested = requested if requested in BACKENDS else 'auto'
    if requested == 'vision':
        return {'requested': requested, 'backend': 'vision',
                'fell_back': False, 'detail': ''}
    from ..capabilities import probe_watermark_detect
    probe = probe_watermark_detect()
    if probe.get('ok'):
        return {'requested': requested, 'backend': 'detector',
                'fell_back': False, 'detail': ''}
    if requested == 'auto':
        return {'requested': requested, 'backend': 'vision',
                'fell_back': False, 'detail': ''}
    reason = str(probe.get('detail') or 'the watermark detector is not installed')
    return {
        'requested': requested, 'backend': 'vision', 'fell_back': True,
        'detail': (f'the vision model ran this scan: {reason}. '
                   f'Install the detector from {INSTALL_ROUTE}, then run it again.'),
    }


def threshold() -> float:
    """The score at or above which an image counts as watermarked.

    Clamped rather than validated-and-refused: this is read on a hot path in the
    middle of a long pass, and a config someone hand-edited to 5 must not abort a
    scan — it must behave like "flag nothing", visibly."""
    try:
        value = float(cfg.get('watermark_detect.threshold'))
    except (TypeError, ValueError):
        return 0.60
    return min(1.0, max(0.0, value))


def models_root() -> str | None:
    """Where the weights are cached. The installer downloads into the app's data
    directory so an install is self-contained and the download is visible as
    disk, rather than disappearing into a per-user HuggingFace cache the app
    never mentions. None means "wherever huggingface_hub already puts things"."""
    root = (cfg.get('watermark_detect.models_root') or '').strip()
    if root:
        return root
    try:
        return str(cfg.data_dir() / 'models' / 'watermark_detect')
    except OSError:      # a data-dir hiccup is not a reason to fail a scan
        return None


class DetectorUnavailable(RuntimeError):
    """The child could not start or could not load its models — the caller
    should fall back to the vision model rather than fail the pass."""


def scan(paths, *, device=None, locate=True, should_cancel=None, cancel_file=None,
         info=None):
    """Yield ``(path, state, score, regions, fingerprint, error)`` per image.

    `state` is 'detected' | 'none' | 'error'. `regions` are normalised
    [x1,y1,x2,y2] boxes in 0..1 (the same contract the hand-drawn masks and the
    crop/inpaint router already use), empty when the locator found nothing or
    could not load. `error` is a short string for 'error' rows only.

    `info`, when given, is a dict this generator fills with facts about the run
    the per-image tuples cannot carry — today ``device``: 'cuda' or 'cpu', THE
    DEVICE THE CHILD ACTUALLY RANKED ON. The child has always reported it in its
    summary and this side always threw it away, which is how a machine with a
    4090 spent 15 hours scanning on the CPU without one line of UI or log ever
    saying so.

    Raises DetectorUnavailable if the child never got as far as loading its
    ranker — that is the one failure the caller must be able to tell apart from
    "ran and found nothing", because they lead to opposite decisions.
    """
    paths = [str(p) for p in paths]
    if not paths:
        return
    for start in range(0, len(paths), BATCH):
        chunk = paths[start:start + BATCH]
        produced = 0
        for item in _run_chunk(chunk, device=device, locate=locate,
                               should_cancel=should_cancel, cancel_file=cancel_file,
                               info=info):
            produced += 1
            yield item
        if should_cancel and should_cancel():
            return
        if produced < len(chunk):
            # The child stopped early without being cancelled. Everything it did
            # produce has already been yielded and will be committed; stopping
            # here rather than launching the next chunk keeps the caller's
            # "N scanned, M left" honest instead of silently skipping a block.
            logger.warning('watermark detector: child returned %d of %d results, '
                           'stopping this pass', produced, len(chunk))
            return


def _run_chunk(chunk, *, device, locate, should_cancel, cancel_file, info=None):
    job = {
        'images': chunk,
        'threshold': threshold(),
        'locate': bool(locate),
        'device': (device or cfg.get('watermark_detect.device') or 'auto'),
        'models_root': models_root(),
        'cancel_file': cancel_file or '',
    }
    python = detector_python()
    # The weights location travels as an explicit `cache_dir` (in the job), never
    # as HF_HOME. The two do NOT agree — HF_HOME appends a `hub/` level and
    # cache_dir does not — so setting both is how an app ends up downloading the
    # same 840 MB twice into two folders it never mentions to the user.
    env = infer_env.worker_env(python, PYTHONUTF8='1',
                           PYTHONIOENCODING='utf-8')
    try:
        proc = subprocess.Popen(
            infer_env.worker_argv(python, _SCRIPT),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace',
            env=env, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except OSError as e:
        raise DetectorUnavailable(f'could not start the watermark detector: {e}') from e

    tail = []

    def drain_stderr():
        for line in proc.stderr:
            line = line.rstrip()
            if line:
                tail.append(line)
                del tail[:-20]
    watcher = threading.Thread(target=drain_stderr, daemon=True)
    watcher.start()
    try:
        proc.stdin.write(json.dumps(job))
        proc.stdin.close()
    except OSError as e:
        proc.kill()
        raise DetectorUnavailable(f'the watermark detector closed immediately: {e}') from e

    index = 0
    summary = None
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue          # a stray print is noise, never a failed pass
            if 'summary' in payload:
                summary = payload['summary'] or {}
                if summary.get('device'):
                    # Both surfaced ON PURPOSE: the log line is the record an
                    # agent or a bug report can quote, the `info` dict is what
                    # the pass detail shows the user. Dropping this on success
                    # was how CPU-only scans stayed invisible for weeks.
                    logger.info('watermark detector ran on %s (%d images)',
                                summary['device'], len(chunk))
                    if info is not None:
                        info['device'] = summary['device']
                continue
            # The child answers in input order, but we key on the path it echoes
            # rather than on our own counter: a desynchronised index would attach
            # one image's verdict to a DIFFERENT image's database row, which is
            # the single worst thing this module could do.
            path = payload.get('path')
            if path != (chunk[index] if index < len(chunk) else None):
                logger.warning('watermark detector: out-of-order result for %r, '
                               'dropping it', path)
                continue
            index += 1
            yield (path, str(payload.get('state') or 'error'),
                   payload.get('score'), list(payload.get('regions') or []),
                   payload.get('fingerprint'),
                   payload.get('error'))
            if should_cancel and should_cancel():
                break
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        watcher.join(timeout=2)
    if summary is not None and not summary.get('ok') and index == 0:
        raise DetectorUnavailable(summary.get('error')
                                  or 'the watermark detector could not load its models')
    if summary is None and index == 0:
        raise DetectorUnavailable(
            _last_line(tail) or 'the watermark detector produced no output')


def _last_line(tail) -> str:
    return next((ln for ln in reversed(tail) if ln.strip()), '')
