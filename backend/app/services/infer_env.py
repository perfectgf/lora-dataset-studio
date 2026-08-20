"""One environment for every ``backend/infer/*`` worker — and for the probe that
vouches for it.

WHY THIS EXISTS. A per-user site-packages directory
(``%APPDATA%\\Python\\PythonXY\\site-packages`` on Windows,
``~/.local/lib/pythonX.Y/site-packages`` elsewhere) is shared by EVERY
interpreter of that Python version on the machine, and it sits AHEAD of the
interpreter's own ``site-packages`` on ``sys.path``. So one unrelated
``pip install --user`` — from a tutorial, a different project, an installer that
guessed — silently shadows a package our worker needs, in an environment neither
the app nor the user ever pointed at it.

That is not hypothetical. A bank of 861 shots was made permanently unsearchable
by an ``eventlet`` left in a user site-packages by another project: it is
imported transitively from ``open_clip``, the copy there predated Python 3.12
and died on ``ssl.wrap_socket``, and the CLIP image worker therefore failed to
start 855 times in a row. Nothing was missing and nothing the app installed was
wrong.

WHY BOTH LEVERS. ``-s`` and ``PYTHONNOUSERSITE=1`` have the same effect on the
process that receives them (measured: both flip ``site.ENABLE_USER_SITE`` to
False and drop the directory from ``sys.path``). They are applied together
because they fail differently:

* ``-s`` lives in argv, so it cannot be lost by a caller that rebuilds ``env=``
  from scratch, and it is what the readiness probes already assert. It does NOT
  reach a process the worker itself starts.
* ``PYTHONNOUSERSITE`` is inherited, so it covers those grandchildren — and it
  stops a ``pip`` running inside one from defaulting to a ``--user`` install,
  which is how the junk drawer gets refilled in the first place.

⚠️ IT APPLIES TO A BORROWED INTERPRETER, NEVER TO THE APP'S OWN — and that line
is measured, not cautious. On an install whose app runs on a system-wide Python
rather than a venv, ``pip install`` cannot write to that Python's own
site-packages and silently falls back to a ``--user`` install: measured on such
a machine, ``import torch`` resolves to
``…\\Roaming\\Python\\Python314\\site-packages\\torch`` and ``python -s -c
"import torch"`` raises ModuleNotFoundError. There, the user site-packages IS
the app's dependency store, and nothing can tell its packages apart from the
junk. Isolating it would not protect that install, it would disable it.

A BORROWED interpreter is the opposite case, and safely so:

* The environments the app builds itself are venvs, where ``ENABLE_USER_SITE``
  is already False — for those this is provably a no-op.
* One the user pointed us at (ComfyUI's, ai-toolkit's, a conda env) is only ever
  accepted by ``scoring_python.select()``, which refuses anything whose
  dependencies do not import under ``python -s``. An interpreter that reached
  the config has therefore already proved it holds them WITHOUT the user site.

So what the flag can remove from a borrowed interpreter is, by construction,
only what nobody asked for. Which is exactly what broke: the interpreter in the
incident was a borrowed one.

⚠️ THE PROBE MUST USE THIS TOO. A probe isolated while its worker is not is the
worst of both worlds: it reports ✓ against a sanitised environment the worker
never sees, and the failure lands an hour later, per item, as if the DATA were
at fault. That asymmetry is exactly what turned the incident above from a
five-minute environment fix into a dead bank. ``test_infer_env.py`` pins the
pairs.

WHAT THIS DELIBERATELY DOES NOT COVER. Interpreters the app merely LAUNCHES on
the user's behalf and does not shape the argv of — ai-toolkit's training venv
above all. Their environment is the user's to arrange, and their readiness
probes are unisolated to match.
"""
from __future__ import annotations

import os
import sys

#: The interpreter flag that drops the per-user site-packages directory.
NO_USER_SITE_FLAG = '-s'

#: The inherited form of the same instruction, for the worker's own children.
NO_USER_SITE_ENV = 'PYTHONNOUSERSITE'


def _norm(path) -> str:
    try:
        return os.path.normcase(os.path.abspath(str(path or '')))
    except (OSError, ValueError):
        return str(path or '')


def is_borrowed(python) -> bool:
    """True when `python` is somebody else's interpreter rather than ours.

    The whole rule turns on this. Ours may be a bare system Python whose ML
    extras live in the user site-packages (see the module docstring); anything
    else earned its place by importing them from its own."""
    return bool(python) and _norm(python) != _norm(sys.executable)


def worker_argv(python, *args) -> list:
    """How a ``backend/infer/*`` worker is run: ``[python, '-s', *args]`` for a
    borrowed interpreter, ``[python, *args]`` for our own.

    Takes the trailing arguments rather than a whole argv so a caller cannot
    accidentally insert the flag after the script, where the interpreter would
    hand it to the script instead of acting on it."""
    rest = [str(a) for a in args]
    if not is_borrowed(python):
        return [str(python), *rest]
    return [str(python), NO_USER_SITE_FLAG, *rest]


def worker_env(python=None, base=None, **extra) -> dict:
    """The environment for an infer worker: `base` (``os.environ`` by default),
    the no-user-site instruction when `python` is borrowed, and the caller's own
    keys on top.

    `python` defaults to None meaning "a borrowed one" — the common case, and
    the safe default for a caller that forgets, since the failure it produces
    (a capability that reads ✗) is visible where the one it prevents is not.

    A fresh dict every time — callers mutate what they get back, and handing out
    a view of ``os.environ`` would let one pass rewrite the whole process."""
    env = dict(os.environ if base is None else base)
    if python is None or is_borrowed(python):
        env[NO_USER_SITE_ENV] = '1'
    else:
        env.pop(NO_USER_SITE_ENV, None)
    for key, value in extra.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    return env
