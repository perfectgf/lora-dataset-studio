"""Turn a written query into a CLIP vector — and pay for it as rarely as possible.

The bank already holds a CLIP embedding for every ✨ Score-d image, so ranking a
bank against a sentence is one dot product per image: numpy does it in-process in
milliseconds. The ONE expensive step is producing the SENTENCE's vector, because
torch/open_clip are not in the Flask venv.

Measured on the reference machine (ai-toolkit venv, torch 2.9.1, CPU forced):

    import torch + open_clip .................. 5.4 s
    build model + load weights ................ 2.4 s
    encode_text, first call ................... 24-48 ms
    encode_text, subsequent calls ............. 20-25 ms
    whole one-shot subprocess, launch → exit ... 8.5-8.8 s
    peak RSS .................................. 2.43 GB

Those numbers decide the design. 8.5 s to encode three words, of which 7.8 s is
pure start-up, makes "one subprocess per search" untenable — and search is
iterative by nature, so the user pays it on every refinement. But the flip side
is 2.43 GB held for as long as a model stays resident, which on a modest machine
is a lot to spend on a feature used in bursts.

So the arbitration is three-layered, cheapest first:

1. **A persistent query cache.** A CLIP text vector depends only on the
   checkpoint, so it is written next to the banks and re-read forever, across
   restarts, app-wide. A re-typed or favourite phrase costs nothing and does not
   even wake the worker. This is the only layer that survives a restart.
2. **A lazily-started WARM worker.** The first uncached query of a session starts
   the child and pays the ~8.5 s load; every later one costs ~20 ms. Nothing is
   started at boot and nothing is started for a cache hit — an install that never
   searches never pays a byte.
3. **An idle reaper.** The worker exits after ``bank_scoring.text_search_idle_minutes``
   without a query (default 10). Ten minutes covers a realistic refine-and-retry
   session while guaranteeing the 2.43 GB is never held overnight or through an
   unattended training run. Setting it to 0 disables the warm layer entirely —
   the honest choice on a memory-tight machine, where paying 8.5 s per distinct
   phrase beats surrendering 2.4 GB. ``release()`` reaps it immediately when the
   user closes the search panel; the timer is the backstop for a closed tab.

**CPU, always.** The text tower is small and the cost is the load, not the maths,
so the GPU would buy nothing measurable and would make a search collide with a
training run. The child hides CUDA from itself, no GPU window is ever taken, and
a search works fine WHILE a LoRA trains — it competes for RAM, never for VRAM.

**No silent download.** open_clip fetches ~1.6 GB of ViT-L/14 weights on first
use. Text search never triggers that on its own: it refuses outright unless the
bank already HAS ✨ Score embeddings, which means this machine already downloaded
this exact checkpoint to produce them. The one residual case — the user re-pointed
``bank_scoring.python``/``models_root`` at an environment with a different cache
since scoring — is reported by ``weights_warning()`` so the UI can warn BEFORE the
click rather than stall for ten minutes.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections import OrderedDict, deque
import zipfile

from .. import config as cfg
from ..utils.redact import redact_tokens, redact_user_paths
from . import atomic_npz
from . import infer_env

logger = logging.getLogger(__name__)

_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'clip_text_infer.py')
_SIGLIP2_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'siglip2_text_infer.py')

# A cold `import torch` alone can run tens of seconds on a fresh machine (native
# DLLs + antivirus), and CLIP ViT-L/14 has ~1.6 GB of weights to read after that
# — more if they still have to be downloaded. Generous on purpose: a timeout here
# would read as "text search is broken" about an interpreter that was merely slow.
START_TIMEOUT = 900
# Once warm, a query is ~20 ms. Anything past this means the worker is wedged.
QUERY_TIMEOUT = 120
DEFAULT_IDLE_MINUTES = 10

# Text encoders truncate far below this in practice (SigLIP2 is explicitly 64
# tokens), so a multi-megabyte phrase can only consume memory/disk; it cannot
# improve retrieval.  The same ceiling is enforced before parsing ``-term``
# syntax and again at the encoder boundary.
MAX_QUERY_CHARS = 512
MAX_QUERY_UTF8_BYTES = 2048
MAX_CACHED_QUERIES = 512
_QUERY_CACHE_MAX_FILE_BYTES = 8 * 1024 * 1024
_QUERY_CACHE_MAX_UNCOMPRESSED_BYTES = 4 * 1024 * 1024
_CLIP_TEXT_DIMENSION = 768

# The child's stdout is ours by PROTOCOL, not by physics. An ML interpreter can
# greet its first run, a dependency can announce a download, a sitecustomize can
# print — and every one of those lands ahead of our JSON. A bare json.loads
# turned any such line into "the text encoder produced no result" on installs
# where the encoder was perfectly able to answer, and took 🎨 Medium and 🔎 text
# search down with it. So the reader steps over what is not JSON, exactly like
# the ✨ Score reader does, and keeps what it stepped over for the message.
_MAX_NOISE_LINES = 30       # past this the child is talking, not answering
_NOISE_KEEP = 6             # how many of those lines are quoted back
_NOISE_CHARS = 300          # and how much of them — a message, not a log
_STDERR_KEEP = 8

_lock = threading.RLock()
_memory: OrderedDict = OrderedDict()  # normalised query -> list[float]
_loaded = False             # has the on-disk cache been read into _memory yet?

_proc = None                # the warm worker (subprocess.Popen) or None
_last_used = 0.0
_reaper = None

# SigLIP2 is an independent vector space.  Keeping every layer separate is
# load-bearing: a query cached by CLIP, or a warm CLIP child, must never answer a
# SigLIP2 search merely because both spaces happen to be 768-dimensional.
_siglip2_memory: OrderedDict = OrderedDict()
_siglip2_loaded = False
_siglip2_proc = None
_siglip2_last_used = 0.0
_siglip2_reaper = None


def _semantic_engine(engine='clip') -> str:
    """Validate a public engine argument without importing an ML dependency."""
    from . import bank_semantic_engine
    try:
        return bank_semantic_engine.normalize_engine(engine)
    except ValueError as exc:
        raise TextEncodeError(str(exc)) from None


class TextEncodeError(RuntimeError):
    """The encoder ran but could not produce a vector. Carries the child's own
    words so the UI can show WHY instead of a generic failure."""


# --- the cache key ------------------------------------------------------------
_WS = re.compile(r'\s+')


def query_limit_error(text, label='query') -> str | None:
    """Return a stable validation sentence without allocating a NumPy key."""
    if text is None:
        return None
    if not isinstance(text, str):
        return f'{label} must be text'
    if len(text) > MAX_QUERY_CHARS:
        return f'{label} is too long (maximum {MAX_QUERY_CHARS} characters)'
    try:
        encoded_size = len(text.encode('utf-8'))
    except UnicodeError:
        return f'{label} contains invalid Unicode'
    if encoded_size > MAX_QUERY_UTF8_BYTES:
        return f'{label} is too large (maximum {MAX_QUERY_UTF8_BYTES} UTF-8 bytes)'
    return None


def normalize_query(text: str) -> str:
    """The cache key for a query: lowercased, whitespace collapsed, trimmed.

    Deliberately nothing more. Stripping punctuation or stop-words would change
    the vector CLIP produces (its tokenizer sees the raw string), so the key must
    only fold differences the ENCODER itself ignores. Kept in one named place
    because widening it later would silently invalidate every cached vector on
    every install."""
    return _WS.sub(' ', str(text or '').strip()).lower()


# A `-` that OPENS a whitespace-delimited token, followed by something. Anchored
# on the boundary on purpose: "close-up", "thigh-high", "2026-07" all carry an
# inner hyphen and must survive untouched — only a leading dash is a grammar.
_MINUS_TERM = re.compile(r'(?:^|\s)-(\S+)')


def split_query(text: str):
    """``("a woman in a car", "hat, sunglasses")`` from ``"a woman in a car -hat
    -sunglasses"`` — the positive phrase and the terms to push DOWN.

    The dash is a shorthand for the panel's second field, not a second search
    grammar: both ends produce one positive phrase and one excluded phrase, and
    the excluded terms are comma-joined into a single string because that is
    what CLIP handles best. Measured on 7,316 captioned bank images, excluding
    two attributes at once: a joined "blonde hair, a tattoo" left 8.3% of the
    top 60 carrying either attribute, exactly matching the mean of the two
    separate vectors (8.3%) and beating the per-term maximum (10.0%) — for one
    encode and one cache entry instead of two.

    Returns both halves un-normalised; the caller normalises what it encodes."""
    raw = str(text or '')
    # `--hat` is a typo, not a request to search for the string "-hat".
    terms = [t.strip('-') for t in _MINUS_TERM.findall(raw) if t.strip('-')]
    if not terms:
        return raw.strip(), ''
    return _MINUS_TERM.sub(' ', raw).strip(), ', '.join(terms)


def idle_minutes() -> float:
    """Configured warm window, clamped to something sane. 0 = never stay warm."""
    try:
        v = float(cfg.get('bank_scoring.text_search_idle_minutes'))
    except (TypeError, ValueError):
        return DEFAULT_IDLE_MINUTES
    if v != v or v < 0:           # NaN / negative -> the default, never a trap
        return DEFAULT_IDLE_MINUTES
    return min(v, 120.0)


# --- persistent cache ---------------------------------------------------------
def _cache_path():
    """App-wide, next to the banks: a text vector depends only on the CLIP
    checkpoint, never on which bank is being searched."""
    return cfg.banks_root() / 'text_query_cache.npz'


def _siglip2_cache_path():
    """Model-keyed cache path for the independent SigLIP2 text space."""
    from . import bank_semantic_engine
    model_key = bank_semantic_engine.engine_model_key('siglip2')
    safe_key = re.sub(r'[^a-zA-Z0-9._-]+', '-', model_key).strip('-').lower()
    return cfg.banks_root() / f'text_query_cache.{safe_key}.npz'


def _cache_archive_is_bounded(path, expected_names) -> bool:
    """Reject duplicate members and compressed bombs before NumPy allocates."""
    try:
        if path.stat().st_size > _QUERY_CACHE_MAX_FILE_BYTES:
            return False
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or set(names) != set(expected_names):
                return False
            return sum(info.file_size for info in infos) \
                <= _QUERY_CACHE_MAX_UNCOMPRESSED_BYTES
    except (OSError, ValueError, zipfile.BadZipFile):
        return False


def _remember_query(memory, key, vector) -> None:
    """Insert/move one phrase and evict the least recently used entries."""
    memory.pop(key, None)
    memory[key] = [float(value) for value in vector]
    while len(memory) > MAX_CACHED_QUERIES:
        memory.popitem(last=False)


def forget_memory_cache(engine=None) -> None:
    """Drop the in-memory layer (tests use it to simulate a restart). The file is
    untouched — that is the point.  No argument clears both independent spaces
    so test/app teardown cannot accidentally retain a SigLIP2 query cache."""
    global _loaded, _siglip2_loaded
    with _lock:
        selected = _semantic_engine(engine) if engine is not None else None
        if selected in (None, 'clip'):
            _memory.clear()
            _loaded = False
        if selected in (None, 'siglip2'):
            _siglip2_memory.clear()
            _siglip2_loaded = False


def _load_disk_cache() -> None:
    """Read the .npz into memory once. A missing or corrupt file is simply an
    empty cache — never an error; the worst case is re-encoding."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        import numpy as np
    except ModuleNotFoundError:
        # Text search is optional.  Merely rendering a Bank (or switching its
        # semantic engine) must stay safe on the lightweight base install.
        return
    path = _cache_path()
    try:
        if (not path.is_file()
                or not _cache_archive_is_bounded(
                    path, {'queries.npy', 'vecs.npy'})):
            return
        with np.load(str(path), allow_pickle=False) as z:
            if set(z.files) != {'queries', 'vecs'}:
                raise ValueError('CLIP text cache keys do not match')
            queries_array = np.asarray(z['queries'])
            vecs = np.asarray(z['vecs'])
            if (queries_array.ndim != 1
                    or queries_array.dtype.kind not in ('U', 'S')
                    or len(queries_array) > MAX_CACHED_QUERIES
                    or vecs.dtype != np.dtype('float32')
                    or vecs.shape != (len(queries_array), _CLIP_TEXT_DIMENSION)
                    or not np.isfinite(vecs).all()):
                raise ValueError('CLIP text cache arrays are invalid')
            queries = [str(query) for query in queries_array]
            if (len(set(queries)) != len(queries)
                    or any(not query or query_limit_error(query) for query in queries)):
                raise ValueError('CLIP text cache queries are invalid')
        for i, q in enumerate(queries):
            _remember_query(_memory, q, vecs[i])
    except Exception:  # noqa: BLE001 — a corrupt cache = recompute, never fatal
        _memory.clear()


def _sweep_orphan_text_caches(path) -> None:
    """Delete leftover ``.tmp.npz`` files for a TEXT cache — never promote one.

    The image-side caches salvage their orphans, because those hold hours of GPU
    work. These two hold query vectors: rebuilding one is a handful of seconds
    against a model that is loaded anyway, and both loaders enforce a strict
    provenance/normalisation contract that a promoted file would have to satisfy
    to be of any use. Promoting is therefore risk with no reward here.

    Sweeping, on the other hand, is a real fix: the SigLIP2 writer names its
    temporary with pid+thread+nanoseconds, so every process killed mid-write used
    to leave one more file behind, forever, with nothing that ever removed them.
    """
    for orphan in atomic_npz.orphan_temporaries(path):
        try:
            os.remove(orphan)
        except OSError:
            pass


def _save_disk_cache() -> None:
    import numpy as np
    if not _memory:
        return
    path = _cache_path()
    try:
        while len(_memory) > MAX_CACHED_QUERIES:
            _memory.popitem(last=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(_memory)
        atomic_npz.save_npz_atomic(path, dict(
            queries=np.array(keys),
            vecs=np.array([_memory[k] for k in keys], dtype='float32')))
    except Exception:  # noqa: BLE001 — losing the cache costs time, never data
        pass
    _sweep_orphan_text_caches(path)


def _load_siglip2_disk_cache() -> None:
    """Load only a cache with the exact current SigLIP2 text/image provenance."""
    global _siglip2_loaded
    if _siglip2_loaded:
        return
    _siglip2_loaded = True
    try:
        import numpy as np
    except ModuleNotFoundError:
        # SigLIP2 is installed explicitly from Setup.  Status/readiness calls
        # still have to degrade to an empty cache before that optional stack is
        # present instead of turning an engine switch into a 500 response.
        return
    from . import bank_semantic_engine
    path = _siglip2_cache_path()
    try:
        expected_members = {
            f'{name}.npy' for name in (
                'version', 'engine', 'model_id', 'revision', 'model_key',
                'dimension', 'queries', 'vecs')
        }
        if (not path.is_file()
                or not _cache_archive_is_bounded(path, expected_members)):
            return
        contract = bank_semantic_engine.semantic_contract('siglip2')
        expected_keys = {
            'version', 'engine', 'model_id', 'revision', 'model_key',
            'dimension', 'queries', 'vecs',
        }
        with np.load(str(path), allow_pickle=False) as z:
            if set(z.files) != expected_keys:
                raise ValueError('SigLIP2 text cache keys do not match')

            def scalar(name):
                value = np.asarray(z[name])
                if value.shape != (1,):
                    raise ValueError(f'{name} metadata must have shape (1,)')
                return value[0].item()

            if (np.asarray(z['version']).dtype != np.dtype('int32')
                    or int(scalar('version')) != 1
                    or scalar('engine') != contract['engine']
                    or scalar('model_id') != contract['model_id']
                    or scalar('revision') != contract['revision']
                    or scalar('model_key') != contract['model_key']
                    or np.asarray(z['dimension']).dtype != np.dtype('int32')
                    or int(scalar('dimension')) != contract['dimension']):
                raise ValueError('SigLIP2 text cache provenance mismatch')
            queries_array = np.asarray(z['queries'])
            vecs = np.asarray(z['vecs'])
            if (queries_array.ndim != 1
                    or queries_array.dtype.kind not in ('U', 'S')):
                raise ValueError('SigLIP2 text cache queries are invalid')
            queries = [str(query) for query in queries_array]
            if (len(queries) > MAX_CACHED_QUERIES
                    or len(set(queries)) != len(queries)
                    or any(not query or query_limit_error(query) for query in queries)
                    or vecs.dtype != np.dtype('float32')
                    or vecs.shape != (len(queries), contract['dimension'])
                    or not np.isfinite(vecs).all()):
                raise ValueError('SigLIP2 text cache vectors are invalid')
            if len(vecs):
                norms = np.linalg.norm(vecs, axis=1)
                if not np.allclose(norms, 1.0, rtol=1e-3, atol=1e-4):
                    raise ValueError('SigLIP2 text cache vectors are not normalised')
        for index, query in enumerate(queries):
            _remember_query(_siglip2_memory, query, vecs[index])
    except Exception:  # corrupt/cross-model cache = recompute, never reuse
        _siglip2_memory.clear()


def _save_siglip2_disk_cache() -> None:
    """Atomically persist the current SigLIP2 query space with provenance."""
    import numpy as np
    from . import bank_semantic_engine
    if not _siglip2_memory:
        return
    path = _siglip2_cache_path()
    try:
        while len(_siglip2_memory) > MAX_CACHED_QUERIES:
            _siglip2_memory.popitem(last=False)
        contract = bank_semantic_engine.semantic_contract('siglip2')
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(_siglip2_memory)
        atomic_npz.save_npz_atomic(path, dict(
            version=np.asarray([1], dtype='int32'),
            engine=np.asarray([contract['engine']]),
            model_id=np.asarray([contract['model_id']]),
            revision=np.asarray([contract['revision']]),
            model_key=np.asarray([contract['model_key']]),
            dimension=np.asarray([contract['dimension']], dtype='int32'),
            queries=np.asarray(keys),
            vecs=np.asarray(
                [_siglip2_memory[key] for key in keys], dtype='float32'),
        ))
    except Exception:  # losing the cache costs time, never data
        pass
    _sweep_orphan_text_caches(path)


def cached_queries(engine='clip') -> int:
    """How many phrases are already encoded — lets the UI say "instant" honestly."""
    with _lock:
        selected = _semantic_engine(engine)
        if selected == 'clip':
            _load_disk_cache()
            return len(_memory)
        _load_siglip2_disk_cache()
        return len(_siglip2_memory)


def is_cached(text: str, engine='clip') -> bool:
    with _lock:
        selected = _semantic_engine(engine)
        if selected == 'clip':
            _load_disk_cache()
            return normalize_query(text) in _memory
        _load_siglip2_disk_cache()
        return normalize_query(text) in _siglip2_memory


# --- availability -------------------------------------------------------------
def unavailable_reason(engine='clip'):
    """None when a text query CAN be encoded here, else a sentence explaining why
    not. Text search reuses the ✨ Score interpreter — the very one that produced
    the embeddings it ranks — so if that cannot run, neither can this, and the
    honest answer is to say so rather than to fail mid-request.

    Never raises: a probe that itself explodes reports "could not be verified",
    which is still an answer the UI can render."""
    selected = _semantic_engine(engine)
    if selected == 'siglip2':
        try:
            from ..capabilities import probe_bank_siglip2
            probe = probe_bank_siglip2()
            return None if probe.get('ok') else (
                probe.get('detail') or
                'SigLIP2 text search needs the Quality tools step in Setup')
        except Exception as e:  # noqa: BLE001
            return (f'the SigLIP2 environment could not be verified '
                    f'({type(e).__name__}) — text search needs it')
    try:
        from ..capabilities import probe_bank_scoring
        if probe_bank_scoring().get('ok'):
            return None
    except Exception as e:  # noqa: BLE001
        return (f'the ✨ Score environment could not be verified '
                f'({type(e).__name__}) — text search needs it')
    return ('text search needs the same environment as ✨ Score '
            '(torch + open_clip). Install the Quality tools step in Setup, or '
            'point ✨ Score at a Python that already has them.')


def weights_warning(engine='clip'):
    """A warning string when the FIRST search might have to download the ~1.6 GB
    ViT-L/14 checkpoint, else None.

    Normally it cannot: text search requires ✨ Score embeddings to exist, and
    producing those downloaded this very checkpoint. The exception is a
    models_root that has been re-pointed since — we can see the configured folder
    is empty of an open_clip cache and say so, instead of letting the user stare
    at a spinner for ten minutes. Best effort by design: an unreadable folder
    returns None (we do not cry wolf about a layout we could not inspect)."""
    if _semantic_engine(engine) == 'siglip2':
        # The SigLIP2 capability probe is local-only and reports missing weights
        # as unavailability.  It can therefore never surprise-start a download.
        return None
    root = (cfg.get('bank_scoring.models_root') or '').strip()
    if not root:
        return None            # default HF cache — the Score pass filled it
    try:
        d = os.path.join(root, 'open_clip')
        if os.path.isdir(d) and any(os.scandir(d)):
            return None
    except OSError:
        return None
    return ('the CLIP weights are not in the models folder currently configured '
            'for ✨ Score — the first search may download ~1.6 GB')


# --- warm worker --------------------------------------------------------------
def _reap_if_idle():
    """Reaper loop: exit the worker once it has been unused for the idle window."""
    global _reaper
    while True:
        with _lock:
            if _proc is None:
                _reaper = None
                return
            window = idle_minutes() * 60.0
            quiet = time.time() - _last_used
            if window <= 0 or quiet >= window:
                _stop_worker_locked()
                _reaper = None
                return
            sleep_for = max(1.0, min(30.0, window - quiet))
        time.sleep(sleep_for)


def _siglip2_reap_if_idle():
    """Independent idle reaper for the SigLIP2 text tower."""
    global _siglip2_reaper
    while True:
        with _lock:
            if _siglip2_proc is None:
                _siglip2_reaper = None
                return
            window = idle_minutes() * 60.0
            quiet = time.time() - _siglip2_last_used
            if window <= 0 or quiet >= window:
                _stop_siglip2_worker_locked()
                _siglip2_reaper = None
                return
            sleep_for = max(1.0, min(30.0, window - quiet))
        time.sleep(sleep_for)


def _stop_worker_locked():
    """Terminate the worker and give its ~2.4 GB back. Caller holds the lock."""
    global _proc
    proc, _proc = _proc, None
    if proc is None:
        return
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()           # EOF = the child's clean exit path
    except Exception:  # noqa: BLE001
        pass
    try:
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def _stop_siglip2_worker_locked():
    """Terminate only the SigLIP2 child. Caller holds ``_lock``."""
    global _siglip2_proc
    proc, _siglip2_proc = _siglip2_proc, None
    if proc is None:
        return
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    except Exception:  # noqa: BLE001
        pass
    try:
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def release(engine='clip') -> bool:
    """Reap the warm worker now (the search panel was closed). True when there
    was one to reap — the UI turns that into an honest "memory released"."""
    with _lock:
        selected = _semantic_engine(engine)
        if selected == 'clip':
            had = _proc is not None
            _stop_worker_locked()
        else:
            had = _siglip2_proc is not None
            _stop_siglip2_worker_locked()
    return had


def status(engine='clip') -> dict:
    """What the UI needs to set expectations BEFORE the click: is the model
    already warm (next search instant), how many phrases are cached, is the
    feature available at all, and would a download be needed."""
    selected = _semantic_engine(engine)
    with _lock:
        if selected == 'clip':
            warm = _proc is not None and _proc.poll() is None
        else:
            warm = _siglip2_proc is not None and _siglip2_proc.poll() is None
    # Exact no-argument calls on the legacy branch preserve tests/extensions
    # that monkeypatch these historical seams with zero-argument functions.
    if selected == 'clip':
        reason = unavailable_reason()
        cached = cached_queries()
        warning = weights_warning()
    else:
        reason = unavailable_reason(engine=selected)
        cached = cached_queries(engine=selected)
        warning = weights_warning(engine=selected)
    return {'available': reason is None, 'reason': reason, 'warm': warm,
            'idle_minutes': idle_minutes(), 'cached_queries': cached,
            'weights_warning': warning, 'engine': selected}


def _start_worker_locked():
    """Spawn the child and block until it says it is ready. Caller holds the lock."""
    global _proc, _reaper
    python = cfg.get('bank_scoring.python') or sys.executable
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = ''          # belt and braces with the child
    env['PYTHONUTF8'] = '1'
    try:
        proc = subprocess.Popen(
            infer_env.worker_argv(python, _SCRIPT),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding='utf-8',
            errors='replace', bufsize=1, env=env,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except Exception as e:  # noqa: BLE001
        raise TextEncodeError(f'could not start the text encoder: '
                              f'{type(e).__name__}: {e}') from None
    _attach_stderr_drain(proc)
    noise = []
    try:
        proc.stdin.write(json.dumps({
            'models_root': cfg.get('bank_scoring.models_root') or None}) + '\n')
        proc.stdin.flush()
        data = _read_json_with_timeout(proc, START_TIMEOUT, noise)
    except TextEncodeError as e:
        _kill(proc)
        raise TextEncodeError(f'{e}{_transcript(proc, noise)}') from None
    except Exception:  # noqa: BLE001
        _kill(proc)
        raise TextEncodeError('the text encoder did not answer in a form this '
                              'app could read'
                              + _transcript(proc, noise)) from None
    if not data.get('ok') or not data.get('ready'):
        _kill(proc)
        raise TextEncodeError(str(data.get('error') or 'unknown encoder error'))
    _proc = proc
    if _reaper is None:
        _reaper = threading.Thread(target=_reap_if_idle, daemon=True,
                                   name='clip-text-reaper')
        _reaper.start()
    return proc


def _start_siglip2_worker_locked():
    """Spawn the local-only CPU SigLIP2 child and validate its exact space."""
    global _siglip2_proc, _siglip2_reaper
    from . import bank_semantic_engine, bank_semantic_models
    python = bank_semantic_models.semantic_python()
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['PYTHONUTF8'] = '1'
    try:
        proc = subprocess.Popen(
            infer_env.worker_argv(python, _SIGLIP2_SCRIPT), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding='utf-8', errors='replace', bufsize=1, env=env,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except Exception as exc:  # noqa: BLE001
        raise TextEncodeError(
            f'could not start the SigLIP2 text encoder: '
            f'{type(exc).__name__}: {exc}') from None
    _attach_stderr_drain(proc)
    noise = []
    contract = bank_semantic_engine.text_worker_handshake('siglip2')
    try:
        proc.stdin.write(json.dumps(contract) + '\n')
        proc.stdin.flush()
        data = _read_json_with_timeout(proc, START_TIMEOUT, noise)
    except TextEncodeError as exc:
        _kill(proc)
        raise TextEncodeError(f'{exc}{_transcript(proc, noise)}') from None
    except Exception:  # noqa: BLE001
        _kill(proc)
        raise TextEncodeError(
            'the SigLIP2 text encoder did not answer in a form this app could read'
            + _transcript(proc, noise)) from None
    try:
        dimension = int(data.get('dimension'))
    except (TypeError, ValueError):
        dimension = -1
    if (not data.get('ok') or not data.get('ready')
            or data.get('engine') != contract['engine']
            or data.get('model_key') != contract['model_key']
            or dimension != contract['dimension']):
        _kill(proc)
        reason = data.get('error') or 'SigLIP2 text/image provenance mismatch'
        raise TextEncodeError(str(reason))
    _siglip2_proc = proc
    if _siglip2_reaper is None:
        _siglip2_reaper = threading.Thread(
            target=_siglip2_reap_if_idle, daemon=True,
            name='siglip2-text-reaper')
        _siglip2_reaper.start()
    return proc


def _kill(proc):
    try:
        proc.kill()
    except Exception:  # noqa: BLE001
        pass


def _readline_with_timeout(proc, timeout, budget=None):
    """One stdout line, or TextEncodeError. subprocess pipes have no timed read,
    so the wait happens on a helper thread — a wedged child must never hang a
    Flask worker forever.

    ``budget`` is the whole allowance this read is one slice of, so the message
    names the budget the user is actually up against and not whatever was left
    of it after a banner line."""
    box = {}

    def _read():
        try:
            box['line'] = proc.stdout.readline()
        except Exception as e:  # noqa: BLE001
            box['err'] = e

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive() or 'err' in box:
        raise TextEncodeError(
            f'the text encoder did not answer within '
            f'{int((budget or timeout) // 60)} minutes '
            '— loading CLIP on this machine is unusually slow')
    line = box.get('line') or ''
    if not line.strip():
        raise TextEncodeError('the text encoder exited before answering')
    return line


def _read_json_with_timeout(proc, timeout, noise=None):
    """The next JSON object the child prints, STEPPING OVER anything that is not
    one — the fix for a first-load banner making a healthy encoder look broken.

    Everything skipped is appended to ``noise`` so a genuine failure can quote
    what really arrived instead of asserting that nothing did. The timeout is a
    budget for the WHOLE read, never per line: a child printing a line a second
    must not be able to extend the wait forever."""
    deadline = time.monotonic() + timeout
    for _ in range(_MAX_NOISE_LINES):
        left = deadline - time.monotonic()
        line = _readline_with_timeout(proc, max(left, 1.0), budget=timeout)
        if line.lstrip().startswith('{'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass          # a truncated or interleaved line is noise too
        if noise is not None:
            noise.append(line.strip())
        logger.info('clip text encoder: ignored a non-JSON line from the worker: %s',
                    _one_line(line))
    raise TextEncodeError('the text encoder kept printing without ever answering')


def _attach_stderr_drain(proc):
    """Keep the child's last stderr lines, and keep its pipe empty.

    stderr used to be DEVNULL: the one place a ModuleNotFoundError or a download
    URL would have appeared was thrown away before anyone could read it, which
    is why the only bug report we got could not say WHAT the worker printed. It
    cannot be an unread pipe either — a child that fills the buffer would block
    for ever. So a daemon thread drains it into a small ring buffer, which is
    all an error message needs."""
    stream = proc.stderr
    if stream is None:
        return
    buf = deque(maxlen=_STDERR_KEEP)

    def _pump():
        try:
            for line in stream:
                s = str(line).strip()
                if s:
                    buf.append(s)
        except Exception:  # noqa: BLE001 — a closed pipe just ends the pump
            pass
        finally:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    t = threading.Thread(target=_pump, daemon=True, name='clip-text-stderr')
    t.start()
    try:
        proc.lds_stderr_tail = (buf, t)
    except Exception:  # noqa: BLE001 — diagnostics are never worth an exception
        pass


def _stderr_tail(proc):
    got = getattr(proc, 'lds_stderr_tail', None)
    if not got:
        return []
    buf, t = got
    t.join(0.5)     # the child is being killed; let the pump flush what it read
    return list(buf)


def _one_line(text, limit=_NOISE_CHARS):
    """A single, truncated, paste-safe line out of whatever the child printed.

    Paste-safe matters here: this text goes into an error the user reads on
    screen and pastes into a public help thread, and a worker's chatter is
    exactly where an absolute home-dir path or a token shows up."""
    s = ' '.join(str(text or '').split())
    if not s:
        return ''
    s = redact_user_paths(redact_tokens(s))
    return s if len(s) <= limit else s[:limit] + '…'


def _transcript(proc, noise=None):
    """What the worker actually said, appended to the message the user sees.

    The report this exists for carried a stack trace that named no cause at all:
    the offending line was dropped and stderr was silenced, so the message could
    only guess — and it guessed the wrong component, sending the reporter to
    check an interpreter that was provably fine."""
    chunks = []
    out = _one_line(' ⏎ '.join(x for x in (noise or [])[-_NOISE_KEEP:] if x))
    if out:
        chunks.append(f'it printed "{out}"')
    err = _one_line(' ⏎ '.join(_stderr_tail(proc)[-_NOISE_KEEP:]))
    if err:
        chunks.append(f'its error output ends with "{err}"')
    if not chunks:
        return ' — it printed nothing at all (see 🪵 Server log in Settings)'
    return ' — ' + '; '.join(chunks)


def _encode_uncached(texts, models_root=None):
    """[vector] for `texts`, through the warm worker (started on demand). Raises
    TextEncodeError with the child's message on any failure. Monkeypatched
    wholesale in tests — no test ever loads a real model."""
    global _last_used
    import numpy as np
    out = []
    with _lock:
        proc = _proc if (_proc is not None and _proc.poll() is None) else None
        if proc is None:
            proc = _start_worker_locked()
        for text in texts:
            noise = []
            try:
                proc.stdin.write(json.dumps({'text': text}) + '\n')
                proc.stdin.flush()
                data = _read_json_with_timeout(proc, QUERY_TIMEOUT, noise)
            except TextEncodeError as e:
                _stop_worker_locked()      # a wedged worker must not be reused
                raise TextEncodeError(f'{e}{_transcript(proc, noise)}') from None
            except Exception:  # noqa: BLE001
                _stop_worker_locked()
                raise TextEncodeError('the text encoder stopped responding'
                                      + _transcript(proc, noise)) from None
            if not data.get('ok'):
                raise TextEncodeError(str(data.get('error') or 'unknown encoder error'))
            out.append(np.asarray(data.get('vector') or [], dtype='float32'))
        _last_used = time.time()
        # idle_minutes == 0 means "never stay warm" — reap as soon as the answer
        # is in hand, so a memory-tight machine never carries the 2.4 GB.
        if idle_minutes() <= 0:
            _stop_worker_locked()
    return out


def _encode_siglip2_uncached(texts):
    """Encode through only the paired SigLIP2 worker, never the CLIP child."""
    global _siglip2_last_used
    import numpy as np
    from . import bank_semantic_engine
    contract = bank_semantic_engine.text_worker_handshake('siglip2')
    out = []
    with _lock:
        proc = (_siglip2_proc if (_siglip2_proc is not None
                                  and _siglip2_proc.poll() is None) else None)
        if proc is None:
            proc = _start_siglip2_worker_locked()
        for text in texts:
            noise = []
            try:
                proc.stdin.write(json.dumps({'text': text}) + '\n')
                proc.stdin.flush()
                data = _read_json_with_timeout(proc, QUERY_TIMEOUT, noise)
            except TextEncodeError as exc:
                _stop_siglip2_worker_locked()
                raise TextEncodeError(f'{exc}{_transcript(proc, noise)}') from None
            except Exception:  # noqa: BLE001
                _stop_siglip2_worker_locked()
                raise TextEncodeError(
                    'the SigLIP2 text encoder stopped responding'
                    + _transcript(proc, noise)) from None
            try:
                dimension = int(data.get('dimension'))
            except (TypeError, ValueError):
                dimension = -1
            if not data.get('ok'):
                raise TextEncodeError(
                    str(data.get('error') or 'unknown SigLIP2 encoder error'))
            vector = np.asarray(data.get('vector') or [], dtype='float32')
            if (data.get('engine') != contract['engine']
                    or data.get('model_key') != contract['model_key']
                    or dimension != contract['dimension']
                    or vector.shape != (contract['dimension'],)
                    or not np.isfinite(vector).all()):
                _stop_siglip2_worker_locked()
                raise TextEncodeError(
                    'the SigLIP2 text encoder returned an incompatible vector')
            out.append(vector)
        _siglip2_last_used = time.time()
        if idle_minutes() <= 0:
            _stop_siglip2_worker_locked()
    return out


def encode_query(text: str, engine='clip'):
    """(vector, from_cache) for ONE query, L2-normed float32.

    Raises TextEncodeError when the encoder is unavailable or failed — the caller
    turns that into an announced 503, never a 500."""
    import numpy as np
    selected = _semantic_engine(engine)
    limit_error = query_limit_error(text)
    if limit_error:
        raise TextEncodeError(limit_error)
    key = normalize_query(text)
    if not key:
        raise TextEncodeError('empty query')
    if selected == 'clip':
        with _lock:
            _load_disk_cache()
            hit = _memory.get(key)
            if hit is not None:
                _memory.move_to_end(key)
        if hit is not None:
            return np.asarray(hit, dtype='float32'), True
        # Preserve the historical zero-argument seams for tests/extensions.
        reason = unavailable_reason()
        if reason:
            raise TextEncodeError(reason)
        vec = _encode_uncached([key])[0]
        vec = np.asarray(vec, dtype='float32')
        if vec.shape != (_CLIP_TEXT_DIMENSION,) or not np.isfinite(vec).all():
            raise TextEncodeError('the CLIP text encoder returned an invalid vector')
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(norm) or norm <= 0:
            raise TextEncodeError('the CLIP text encoder returned an empty vector')
        vec /= norm
        with _lock:
            _remember_query(_memory, key, vec)
            _save_disk_cache()
        return vec, False

    with _lock:
        _load_siglip2_disk_cache()
        hit = _siglip2_memory.get(key)
        if hit is not None:
            _siglip2_memory.move_to_end(key)
    if hit is not None:
        return np.asarray(hit, dtype='float32'), True
    reason = unavailable_reason(engine=selected)
    if reason:
        raise TextEncodeError(reason)
    vec = np.asarray(_encode_siglip2_uncached([key])[0], dtype='float32')
    from . import bank_semantic_engine
    dimension = bank_semantic_engine.engine_dimension(selected)
    if vec.shape != (dimension,) or not np.isfinite(vec).all():
        raise TextEncodeError('the SigLIP2 text encoder returned an invalid vector')
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 0:
        raise TextEncodeError('the SigLIP2 text encoder returned an empty vector')
    vec /= norm
    with _lock:
        _remember_query(_siglip2_memory, key, vec)
        _save_siglip2_disk_cache()
    return vec, False
