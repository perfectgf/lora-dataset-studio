"""Quantize a checkpoint you ALREADY have, locally, to the fp8 file ComfyUI loads.

WHY THIS EXISTS ALONGSIDE THE POD EXPORT
----------------------------------------
The cloud lane produces the fp8 twin automatically at the end of a dense run. But
the models people want to shrink are not only the ones they trained today: a
25.6 GB bf16 full model already downloaded from Hugging Face, a dense checkpoint
from an earlier run, a big finetune someone shared. This is the same conversion,
started by hand, on a file on this machine.

It is deliberately the SAME implementation — ``fp8_export.export_scaled_fp8``.
There is exactly one definition of "the fp8 file LDS produces", it is unit tested
against real torch, and its output was fed to ComfyUI's own ``convert_old_quants``
to prove the loader accepts it. A second local-only quantizer would have been a
second format nobody would notice diverging.

NOT THE SAME THING AS ai-toolkit's `quantize`
---------------------------------------------
ai-toolkit's ``model.quantize`` (and the app's Advanced ▸ memory settings) quantize
the model IN MEMORY while it loads, so a 24 GB card can train something that would
not otherwise fit. It produces no file and changes nothing on disk — the saved
checkpoint is still full precision. THIS produces a file, and that file is the
artifact you load in ComfyUI. The help text says so, because the two are
constantly confused.

CPU, NOT GPU
------------
Quantization here is an elementwise cast plus one reduction per tensor. Measured
on this machine: ~1.2 GB/s of source through the streaming writer, i.e. under a
minute of compute for a 25.6 GB checkpoint — the run is bound by disk, not by
arithmetic. Putting it on the GPU would buy nothing and would fight ComfyUI and
any training run for VRAM. It stays on the CPU, and takes no GPU lock.

IN A SUBPROCESS, NOT IN THE SERVER
----------------------------------
The conversion needs ``torch`` and ``safetensors``. The app's own environment
does NOT have them, and must not: torch is gigabytes, and LDS installs and runs
without it. Importing them in-process shipped a feature that could not execute at
all on a real install — the job died on ``No module named 'safetensors'`` while
every test passed, because the tests ran under the one interpreter on the machine
that happened to have both.

So this delegates, exactly like ``bank_scoring`` / ``masks`` / ``watermark``
already do: a configured interpreter (``quantize.python``, empty = the same one
✨ Score uses, then ai-toolkit's, then this app's) runs ``fp8_export.py`` — the
same file the pod runs — as a CLI. And because "can it run at all" is part of
what the user must know BEFORE clicking, the interpreter is probed in ``plan``:
one that lacks the dependencies is a refusal with the pip command in it, not an
error thirty seconds after the button.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time

from .. import config as cfg
from . import infer_env
from ..job_queue import queue_manager
from . import fp8_export, model_integrity

logger = logging.getLogger(__name__)

# system_state key — one quantization at a time, app-wide. Two 25 GB reads and
# two multi-GB writes at once would be slower than doing them in sequence and
# could fill the drive between the two free-space checks.
_STATE_KEY = 'fp8_quantize'
_STATE_TTL = 6 * 3600
_lock = threading.Lock()

# Working headroom on top of the file we are about to write. It is NOT a second
# copy: the exporter streams into `<dst>.part` and renames, so the only bytes
# claimed are the output's. This covers the filesystem being unhappy near zero
# and whatever else the app writes while a long conversion runs.
#
# It replaces a flat `MIN_FREE_GB = 30`, which refused a real conversion that fit:
# a 25.6 GB master quantizes to 12.8 GB and the drive had 17.6 GB free — enough,
# twice over, for the file actually being written. A budget must be derived from
# what the operation costs, or it is just a number saying no.
WRITE_HEADROOM_BYTES = 2 * 1000 ** 3

_ACCEPTED_EXT = ('.safetensors', '.sft')

# What the worker interpreter must be able to import.
#
# torch, and ONLY torch. This used to also demand `safetensors`, which was true
# while the worker opened checkpoints with `safe_open` — it no longer does:
# fp8_export (and lora_merge, the other user of this probe) read and write the
# format by hand, precisely so that nothing memory-maps a 26 GB file. Leaving
# the old spelling here would refuse an environment that works, which is the
# expensive kind of wrong: a check that describes a world that stopped existing
# one commit ago, and that nobody thinks to re-read because it is "just a probe".
DEP_MODULES = ('torch',)
_PROBE_CODE = (
    'import importlib.util as u, json, sys\n'
    'print(json.dumps({m: u.find_spec(m) is not None for m in '
    + repr(list(DEP_MODULES)) + '}))\n'
)
_PROBE_TIMEOUT = 90          # a cold `import torch` behind an antivirus is slow
_PROBE_TTL = 300
_probe_cache = {}            # normalised interpreter path -> (ts, dict|None)


class QuantizeError(ValueError):
    """Refusal with a sentence for the user. Never a stack trace."""


# --- the interpreter that does the work -------------------------------------------

def candidates() -> list:
    """Interpreters to try, best first. Only ones the app already knows about.

    ``quantize.python`` is not first, it is EXCLUSIVE: someone who filled that
    field said which environment does this work, and quietly using another one
    because theirs turned out to be incomplete would hide the very problem they
    need to fix. With it empty: the environment ✨ Score uses, then ai-toolkit's
    — both have torch by construction — and finally this app's own, honest about
    being last, because on most installs it is the one WITHOUT the dependencies.
    """
    out = []

    def add(path):
        path = str(path or '').strip().strip('"')
        if path and path not in out:
            out.append(path)

    try:
        explicit = str(cfg.get('quantize.python') or '').strip().strip('"')
    except Exception:                            # noqa: BLE001 — config hiccup
        explicit = ''
    if explicit:
        return [explicit]

    try:
        add(cfg.get('bank_scoring.python'))
    except Exception:                            # noqa: BLE001
        pass
    try:
        add(cfg.aitoolkit_path('venv_python'))
    except Exception:                            # noqa: BLE001
        pass
    add(sys.executable)
    return out


def _probe(python: str):
    """``{module: bool}`` for one interpreter, or None when it cannot be asked.

    None is UNKNOWN, never "unusable": a probe that times out must not freeze a
    working venv into a refusal.
    """
    key = os.path.normcase(os.path.abspath(python))
    hit = _probe_cache.get(key)
    now = time.time()
    if hit and (now - hit[0]) < _PROBE_TTL:
        return hit[1]
    try:
        proc = subprocess.run(
            infer_env.worker_argv(python, '-c', _PROBE_CODE),
            capture_output=True, text=True,
            encoding='utf-8', errors='replace', timeout=_PROBE_TIMEOUT,
            env=infer_env.worker_env(python),
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        info = json.loads(((proc.stdout or '').strip().splitlines() or [''])[-1])
        info = info if isinstance(info, dict) else None
    except Exception:                            # noqa: BLE001 — missing, broken, slow
        info = None
    if info is not None:
        _probe_cache[key] = (now, info)
    return info


def clear_probe_cache() -> None:
    _probe_cache.clear()


def interpreter() -> dict:
    """Which interpreter will run the conversion, and whether it actually can.

    ``{'python', 'ready', 'missing', 'reason'}``. Never raises: an unanswerable
    probe reports ``ready`` (the run may still work) rather than inventing a
    refusal, but a probe that ANSWERED "no torch" is a hard, actionable no.
    """
    tried = []
    for python in candidates():
        info = _probe(python)
        if info is None:
            tried.append((python, None))
            continue
        missing = [m for m in DEP_MODULES if not info.get(m)]
        if not missing:
            return {'python': python, 'ready': True, 'missing': [], 'reason': None}
        tried.append((python, missing))
    # Nothing usable. Prefer naming an interpreter that ANSWERED, so the remedy
    # names a real environment instead of one we could not even start.
    answered = [(p, m) for p, m in tried if m is not None]
    if answered:
        python, missing = answered[0]
        return {
            'python': python, 'ready': False, 'missing': missing,
            # "them" reads as "the missing modules" whether that is one or
            # several, so the sentence survived DEP_MODULES shrinking to a single
            # entry. lora_merge_job rewrites the first clause to say "Merging"
            # instead of "Quantizing" — if this wording moves, move that too.
            'reason': (f'the Python that would do the conversion is missing '
                       f'{" and ".join(missing)}. Quantizing needs them, and this '
                       'app deliberately ships without torch (gigabytes). Pick an '
                       'environment that has them — the one ✨ Score uses (Bank ▸ '
                       'Scoring ▸ change the Python) or ai-toolkit\'s — or set '
                       '"quantize": {"python": "…"} in config.json. Installing them '
                       f'there also works: pip install {" ".join(missing)}'),
        }
    python = (candidates() or [sys.executable])[0]
    return {'python': python, 'ready': True, 'missing': [], 'reason': None}


def status() -> dict:
    return queue_manager._get_system_state(_STATE_KEY, {}) or {}


def _free_gb(path) -> float | None:
    """Free space on the volume that REALLY holds this path.

    ``realpath`` first, on purpose: a ComfyUI models folder is very often a
    junction onto another drive (``C:\\…\\models\\unet`` →  ``A:\\ComfyUI\\models\\unet``
    is exactly the layout on the machine this was measured on). Asking about the
    apparent path can answer for the wrong volume, which turns a disk guard into
    a coin toss.
    """
    try:
        import shutil
        return shutil.disk_usage(
            os.path.dirname(os.path.realpath(path))).free / (1000 ** 3)
    except Exception:
        return None


def space_error(free_gb, output_bytes) -> str | None:
    """The refusal sentence when the output does not fit, or None.

    States its own arithmetic. "~30 GB needed" next to a 12.8 GB output was a
    number the user could neither check nor act on.
    """
    if free_gb is None:
        return None                            # unmeasurable never blocks
    need = (int(output_bytes or 0) + WRITE_HEADROOM_BYTES) / (1000 ** 3)
    if free_gb >= need:
        return None
    return (f'not enough disk space where the output would go: {free_gb:.1f} GB free, '
            f'and this needs {need:.1f} GB — the {output_bytes / 1000 ** 3:.1f} GB fp8 '
            f'file plus {WRITE_HEADROOM_BYTES / 1000 ** 3:.0f} GB of working headroom. '
            'Free up space, or write it to another folder.')


def plan(source, *, overwrite=False, destination=None) -> dict:
    """Validate one source file and describe what quantizing it would produce.

    EVERY condition that would make ``quantize`` fail is evaluated HERE. It used
    not to be: ``plan`` reported ``ok: true`` and a free-space figure, then the
    start call refused on a threshold ``plan`` had never applied — so the button
    stayed enabled, the user clicked, and the refusal arrived after the decision.
    A refusal that only exists at run time is a refusal the UI cannot show.

    ``destination`` overrides where the output would go (another folder, another
    volume); the exists- and space-checks then answer about THAT folder.
    """
    path = str(source or '').strip().strip('"')
    if not path:
        raise QuantizeError('choose a .safetensors model file to quantize')
    if not os.path.isabs(path):
        raise QuantizeError('give the full path to the model file')
    if not os.path.isfile(path):
        raise QuantizeError(f'no file at {os.path.basename(path)} — check the path')
    if not path.lower().endswith(_ACCEPTED_EXT):
        raise QuantizeError('only .safetensors checkpoints can be quantized '
                            '(a .gguf file is already quantized)')

    # The volet-3 guard, used in reverse: there it refuses a quantized file as a
    # TRAINING base; here it refuses to quantize something already quantized,
    # which would double the error and produce a file nothing can load.
    report = model_integrity.quantization_report(path)
    if report.get('quantized'):
        raise QuantizeError(
            f'{os.path.basename(path)} is already a quantized export '
            f'({", ".join(report.get("signals") or []) or "quantized dtypes"}) — '
            'quantizing it again would only lose more precision. Use the '
            'full-precision (bf16/fp16) version.')

    integrity = model_integrity.validate_model_file(path)
    if integrity.get('blocking'):
        raise QuantizeError(integrity.get('reason') or 'this file is not a readable model')

    header = fp8_export.read_header(path)          # raises Fp8ExportError -> caught below
    layout = fp8_export.plan_quantization(header)
    if not layout['quantize']:
        raise QuantizeError(
            f'{os.path.basename(path)} has no large 2-D weight matrices to '
            'quantize — this is a LoRA or an adapter, not a full model. '
            'Quantizing it would save nothing.')

    # Written NEXT TO the source by default, never over it: the master is the
    # only file that can be trained again, and a user who chose the wrong file
    # must be able to just delete the output. A caller with somewhere better to
    # put it (ComfyUI's own models folder, or simply a drive with room) says so.
    destination = str(destination or os.path.join(
        os.path.dirname(path), fp8_export.fp8_name_for(os.path.basename(path))))
    exists = os.path.isfile(destination)
    if exists and not overwrite:
        raise QuantizeError(
            f'{os.path.basename(destination)} already exists next to the source — '
            'delete it first, or re-run with overwrite.')
    free_gb = _free_gb(destination)
    refusal = space_error(free_gb, layout['bytes_after'])
    if refusal:
        raise QuantizeError(refusal)
    # "Can this machine run the conversion at all" is the third thing that used
    # to be discovered only after the click. It is a plan question.
    worker = interpreter()
    if not worker['ready']:
        raise QuantizeError(worker['reason'])
    return {
        'python': worker['python'],
        'source': path,
        'source_name': os.path.basename(path),
        'source_bytes': os.path.getsize(path),
        'destination': destination,
        'destination_name': os.path.basename(destination),
        'destination_exists': exists,
        'quantized_tensors': len(layout['quantize']),
        'kept_tensors': len(layout['keep']),
        'estimated_bytes': layout['bytes_after'],
        'free_gb': free_gb,
    }


def describe(source, *, overwrite=False, destination=None) -> dict:
    """``plan`` as a payload the UI can render, refusal included. Never raises —
    a disabled button with a reason beats an error toast on click."""
    try:
        return {'ok': True, **plan(source, overwrite=overwrite, destination=destination)}
    except (QuantizeError, fp8_export.Fp8ExportError) as e:
        return {'ok': False, 'error': str(e), 'source': str(source or '')}


def quantize(source, *, overwrite=False, destination=None, progress=None,
             cancelled=None) -> dict:
    """Do it (BLOCKING, minutes on a 26 GB file). Returns the verified summary.

    Every refusal lives in ``plan``; the space check is repeated here only
    because a long download or another job can eat the drive between the two.
    """
    info = plan(source, overwrite=overwrite, destination=destination)
    refusal = space_error(_free_gb(info['destination']), info['estimated_bytes'])
    if refusal:
        raise QuantizeError(refusal)
    result = run_worker(info['python'], info['source'], info['destination'],
                        progress=progress, cancelled=cancelled)
    return {**info, **result}


def worker_command(python, source, destination) -> list:
    """The exact argv. Exposed so a test can assert it without running torch.

    No user site-packages: with `quantize.python` unset this falls back on the
    interpreter ✨ Score borrows, so it inherits that environment's exposure to
    whatever an unrelated `pip install --user` left on the machine. `_probe`
    asks the same question the same way — see services/infer_env."""
    return infer_env.worker_argv(
        python, os.path.abspath(fp8_export.__file__),
        '--src', str(source), '--dst', str(destination), '--progress')


def run_worker(python, source, destination, *, progress=None, cancelled=None) -> dict:
    """Run the conversion in ``python`` and stream its progress back.

    The child is ``fp8_export.py`` itself — the same file the pod runs, so there
    is exactly one conversion and one verification in the product. It prints one
    ``LDS_FP8_PROGRESS done total`` line per tensor and finishes with the
    ``LDS_FP8_RESULT`` JSON, which already carries the read-back verification.
    """
    command = worker_command(python, source, destination)
    try:
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1,
            env=infer_env.worker_env(python),
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(fp8_export.__file__))),
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except OSError as e:
        raise QuantizeError(
            f'the Python that would do the conversion could not be started: '
            f'{python} ({e}). Check "quantize": {{"python": "…"}} in config.json, '
            'or clear it to fall back on the environment ✨ Score uses.') from e

    result, tail = None, []
    try:
        for line in proc.stdout:
            line = line.rstrip('\n')
            if line.startswith(fp8_export.PROGRESS_PREFIX):
                parts = line.split()
                if progress and len(parts) == 3:
                    try:
                        progress(int(parts[1]), int(parts[2]))
                    except Exception:                   # noqa: BLE001 — never fatal
                        pass
            elif line.startswith(fp8_export.RESULT_PREFIX):
                try:
                    result = json.loads(line[len(fp8_export.RESULT_PREFIX):].strip())
                except ValueError:
                    result = None
            elif line.strip():
                tail = (tail + [line])[-12:]            # kept for the error message
            if cancelled is not None and cancelled():
                proc.kill()
                raise QuantizeError('the conversion was stopped')
    finally:
        try:
            proc.wait(timeout=30)
        except Exception:                               # noqa: BLE001
            proc.kill()

    if not isinstance(result, dict):
        raise QuantizeError(
            'the conversion produced no result. Its last output was: '
            + (' | '.join(tail) or '(nothing)')[:400])
    if not result.get('ok'):
        raise QuantizeError(str(result.get('error') or 'the conversion failed'))
    return result


def verify(path) -> dict:
    """Read-back proof. Delegated, because it needs torch just as the write does."""
    return fp8_export.verify_export(path)


def start_async(app, source, *, overwrite=False) -> dict:
    """Run it in a daemon thread; progress and outcome live in system_state.

    Refuses immediately (before the thread) when another quantization is running
    or the source is not usable — a rejection the user sees on click, not in a
    status poll thirty seconds later.
    """
    info = plan(source, overwrite=overwrite)
    with _lock:
        if status().get('status') == 'running':
            raise QuantizeError('a quantization is already running — wait for it to finish')
        _set('running', info, done=0, total=0)

    def _run():
        with app.app_context():
            def on_progress(done, total):
                # Cheap and throttled by the tensor count itself (hundreds, not
                # millions), so every update is a real step forward.
                _set('running', info, done=done, total=total)
            try:
                result = quantize(info['source'], overwrite=overwrite,
                                  progress=on_progress)
                _set('error' if result.get('verify_error') else 'done', info,
                     result=result, error=result.get('verify_error'))
                logger.info('fp8 quantization finished: %s', info['destination_name'])
            except Exception as e:
                _set('error', info, error=str(e)[:400])
                logger.warning('fp8 quantization failed (%s): %s',
                               info['source_name'], e)

    threading.Thread(target=_run, daemon=True).start()
    return info


def _set(state, info, **extra):
    queue_manager._set_system_state(_STATE_KEY, {
        'status': state,
        'source_name': info['source_name'],
        'destination_name': info['destination_name'],
        'source_bytes': info['source_bytes'],
        'estimated_bytes': info['estimated_bytes'],
        **extra,
    }, ttl_seconds=_STATE_TTL)
