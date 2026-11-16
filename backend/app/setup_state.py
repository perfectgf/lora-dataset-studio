"""Remembering that an install was already checked out fine, so coming back to
the app does not mean sitting through Setup again.

The Setup wizard is a FIRST-RUN experience, but nothing recorded that the first
run had happened: the onboarding redirect keyed off a per-tab sessionStorage
flag, so every new tab / browser session re-offered Setup on a machine that had
been working for weeks. That is the redundancy this module removes.

Three independent signals:

  completed — the user opened the core workspace and its setup state was saved.
              No plugin or image engine is required for this durable choice.

  verified  — this install was once observed working (configured + at least one
              image engine ready). Persisted in the data dir, so it survives a
              new tab, a new browser, a server restart, and a second machine on
              the LAN. Never expires: an install does not become un-set-up.

  checks    — a HIGH-WATER MARK of what this install has proven it can do
              (`TRACKED` below). A re-check compares the live probe against it
              and reports what STOPPED working — a regression is a real failure
              worth interrupting for, whereas "not everything is installed" is
              the normal state of almost every install and must never nag.

Only DURABLE signals are tracked. Whether ComfyUI or Ollama happens to be
running right now is not a setup failure — those are started on demand and go
dark a dozen times a day; treating them as regressions would turn this into the
exact nag it exists to remove. See TRACKED / the module docstring of
capabilities.py for the full published surface.

Nothing here is a user SETTING: it is app state the user never edits, so it
lives in its own small file rather than in config.json.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

from . import config
from .engines import registry as _engines

_LOCK = threading.RLock()

# Durable capability checks, keyed by their dotted path in the /api/capabilities
# payload. The KEYS are persisted (never rename one without an alias — CLAUDE.md
# rule 7); the labels are read at response time and are free to change.
# The non-engine checks. The engines' durable checks come from the engine
# registry AT CALL TIME (an engine that declares `tracked_capability` — the
# API engines register when their plugin loads); the keys are `engines.<id>`
# and ids never change, so the persisted keys are stable.
_STATIC_TRACKED = (
    ('comfyui.dir_valid', 'ComfyUI folder'),
    ('ollama.installed', 'Ollama'),
    ('aitoolkit.valid', 'ai-toolkit'),
    ('captioners.joycaption', 'JoyCaption captioning'),
    ('face_scoring', 'Face-similarity scoring'),
    ('masks', 'Person masks'),
    ('watermark_inpaint', 'Watermark inpainting'),
    ('training_visible', 'LoRA training'),
)


def tracked() -> tuple:
    """``((key, label), …)`` — the engines' checks first, then the rest."""
    return (*_engines.tracked(), *_STATIC_TRACKED)


def tracked_keys() -> tuple:
    return tuple(k for k, _ in tracked())


def _labels() -> dict:
    return dict(tracked())

# The legacy verified flag records a proven generator configuration. Core
# completion is independent: both suppress repeated first-run onboarding.


def _state_path():
    return config.data_dir() / 'setup_state.json'


def _dig(caps: dict, dotted: str):
    """Read a dotted path out of the capabilities payload; missing -> None, so a
    key this build does not publish is 'unknown', never a false regression."""
    cur = caps
    for part in dotted.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _daemon_bound_keys() -> frozenset:
    """The tracked checks only ComfyUI itself can answer: a LOCAL engine's
    readiness is read off the running server (its nodes, its accepted values)."""
    return frozenset(f'engines.{s.id}' for s in _engines.all_specs()
                     if s.tracked_capability and s.kind == _engines.LOCAL)


def snapshot(caps: dict) -> dict:
    """The tracked checks as they stand in `caps`. Unknown keys are dropped
    rather than recorded as False — an older/newer payload must not manufacture
    a regression out of a field it simply does not have.

    A local engine that is not ready while ComfyUI is stopped is unknown too:
    it could not be asked. Without this, a plugin engine that declares a tracked
    capability was reported as "stopped working" every time ComfyUI was off —
    the nag the module docstring rules out. Its "ready" still counts, and so
    does its "not ready" once ComfyUI answers."""
    caps = caps or {}
    comfyui_answers = bool(_dig(caps, 'comfyui.reachable'))
    daemon_bound = _daemon_bound_keys()
    out = {}
    for key in tracked_keys():
        val = _dig(caps, key)
        if val is None:
            continue
        if not val and not comfyui_answers and key in daemon_bound:
            continue
        out[key] = bool(val)
    return out


def install_works(caps: dict) -> bool:
    """Is this a set-up machine? Configured AND able to generate an image."""
    c = caps or {}
    if not c.get('configured'):
        return False
    engines = c.get('engines') or {}
    return any(bool(engines.get(name)) for name in _engines.recommended_ids())


def read() -> dict:
    """Stored state, or the empty shape. Never raises: a corrupt/unreadable file
    means 'we have not verified this install', which degrades to the classic
    first-run wizard rather than to an error screen."""
    with _LOCK:
        try:
            raw = json.loads(_state_path().read_text(encoding='utf-8'))
        except Exception:
            return {'verified': False, 'verified_at': None, 'checks': {}}
        if not isinstance(raw, dict):
            return {'verified': False, 'verified_at': None, 'checks': {}}
        checks = raw.get('checks')
        if not isinstance(checks, dict):
            checks = {}
        state = {
            'verified': bool(raw.get('verified')),
            'verified_at': raw.get('verified_at') or None,
            # Unknown keys are dropped on read, so a key retired in a later
            # version cannot resurface as a phantom regression.
            'checks': {k: bool(v) for k, v in checks.items() if k in tracked_keys()},
        }
        if raw.get('completed') is True:
            state.update(completed=True, completed_at=raw.get('completed_at'))
        return state


def _write(state: dict, *, strict: bool = False) -> dict:
    with _LOCK:
        path = _state_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(state, indent=2), encoding='utf-8')
            tmp.replace(path)
        except OSError:
            if strict:
                raise
            # A read-only data dir must not break the app: the feature simply
            # stops remembering and the user gets today's first-run behaviour.
            pass
        return state


def complete_core() -> dict:
    """Finish onboarding by proving the workspace can persist its own state.

    Generators, ML helpers and plugins are optional. Explicit completion must
    survive a new browser session without claiming any of those tools work.
    A failed write is reported instead of presenting a successful installation.
    """
    with _LOCK:
        state = read()
        return _write({**state, 'completed': True,
                       'completed_at': state.get('completed_at') or _now()}, strict=True)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def observe(caps: dict) -> dict:
    """Record what this install can currently do, and stamp it verified the first
    time it is seen working.

    The stored checks are a HIGH-WATER MARK: a capability that has ever worked
    stays True here even when it is currently failing, which is precisely what
    lets `compare` keep reporting a regression until it is fixed (or explicitly
    dismissed) instead of quietly re-baselining the broken state.

    Called on every read/re-check, so an install set up BEFORE this feature
    shipped is retrofitted on its first page load — no re-run of the wizard.
    """
    with _LOCK:
        state = read()
        checks = dict(state['checks'])
        for key, val in snapshot(caps).items():
            if val:
                checks[key] = True
            else:
                checks.setdefault(key, False)
        verified = bool(state['verified'] or install_works(caps))
        nxt = {
            **state,
            'verified': verified,
            'verified_at': state['verified_at'] or (_now() if verified else None),
            'checks': checks,
        }
        # This runs on every page load; a steady install would otherwise rewrite
        # a byte-identical file each time for nothing.
        return nxt if nxt == state else _write(nxt)


def compare(caps: dict, state: dict | None = None) -> list:
    """Capabilities this install has proven it can do and that are failing NOW.

    Returns [{'key', 'label'}] — empty means the background re-check found
    nothing to interrupt for. A key that is missing from the live payload is
    skipped (unknown != broken)."""
    stored = (state or read())['checks']
    live = snapshot(caps)
    labels = _labels()
    return [{'key': k, 'label': labels[k]}
            for k in tracked_keys()
            if stored.get(k) and k in live and not live[k]]


def dismiss(keys) -> dict:
    """Forget that a capability used to work — the user's "yes, I removed that
    on purpose". Without this, an intentional uninstall would be reported as a
    regression forever, and the only escape would be deleting a state file the
    user has no reason to know exists."""
    with _LOCK:
        state = read()
        checks = dict(state['checks'])
        for key in keys or ():
            if key in tracked_keys():
                checks[key] = False
        return _write({**state, 'checks': checks})
