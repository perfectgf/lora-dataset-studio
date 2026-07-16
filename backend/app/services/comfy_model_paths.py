"""Home-grown reader for ComfyUI's ``extra_model_paths.yaml``.

Portable / Stability-Matrix / A1111-shared ComfyUI installs keep their weights
OUTSIDE ``<base>/models`` and register the real locations in an
``extra_model_paths.yaml`` next to ``main.py``. This app hardcoded ``<base>/models``
everywhere, so on those installs it declared models "missing" and re-downloaded
them (PR #5 proved the need). This module resolves models EXACTLY where a running
ComfyUI would, by mirroring ComfyUI's own parser.

Reference semantics (verbatim from comfyanonymous/ComfyUI, read 2026-07):
  * ``utils/extra_config.py::load_extra_path_config`` (the ~34-line parser):
      - ``yaml.safe_load`` the file; ``yaml_dir = dirname(abspath(path))``.
      - each TOP-LEVEL key is a PROFILE with an arbitrary name (``comfyui``,
        ``a111``, ``stability_matrix`` …); iterate all; ``conf is None`` → skip.
      - ``base_path`` (popped, optional): ``expandvars(expanduser(...))``; if it is
        relative it resolves against ``yaml_dir`` (NOT the cwd).
      - ``is_default`` (popped, optional bool): its dirs are inserted at the FRONT
        of the search list (highest priority).
      - every remaining key is a folder TYPE; its value is a string that may be a
        multi-line block (``|``) → ``split("\\n")``, empty lines skipped. Each line:
        with ``base_path`` → ``join(base_path, line)``; else if relative →
        ``abspath(join(yaml_dir, line))``; then ``normpath``.
  * ``folder_paths.py``:
      - ``map_legacy`` = ``{"unet": "diffusion_models", "clip": "text_encoders"}``
        (canonical alias source — we reuse it, no home-grown alias list).
      - default roots: ``diffusion_models``=[models/unet, models/diffusion_models],
        ``text_encoders``=[models/text_encoders, models/clip], ``vae``=[models/vae],
        ``loras``=[models/loras], ``checkpoints``=[models/checkpoints].
      - ``add_model_folder_path(name, path, is_default)``: map_legacy the name,
        then insert(0) if is_default else append (dedup: move-to-front if default).
      - ``recursive_search`` / ``get_filename_list`` return each file's path
        RELATIVE to its root, subfolders included — which is EXACTLY the string a
        loader node (``unet_name`` / ``vae_name`` / ``lora_name`` …) expects. This
        is the real fix for the original ``klein/`` prefix bug: the name carries
        ``klein/`` because the file sits under ``loras/klein/``, not by a magic
        constant.

Extensions: ComfyUI's ``supported_pt_extensions`` = {.ckpt,.pt,.pt2,.bin,.pth,
.safetensors,.pkl,.sft}. We list ``{.safetensors, .sft, .gguf}``: the app's
pipelines only load these, and the app ALREADY lists ``.gguf`` across
klein_edit_helper + utils/comfyui (ComfyUI-GGUF quantised diffusion models are
common in this community). Dropping ``.gguf`` to match core ComfyUI would HIDE
models existing users already see, so we keep it and stay narrow on the rest so
``picker == probe == resolver``.

Degradation is total and silent-safe: no base_dir / no yaml / malformed yaml /
PyYAML not importable all resolve to "no extra roots" (logged once), never an
exception. With no yaml the search roots ARE the historical ``<base>/models``
folders, so resolution is byte-for-byte what it was before this module existed.
"""
from __future__ import annotations

import logging
import os
import threading

from .. import config as cfg

logger = logging.getLogger(__name__)

# PyYAML guard: many existing installs `git pull` (Update & restart) WITHOUT
# reinstalling requirements, and pyyaml is only a TRANSITIVE dep today (via
# huggingface-hub). A missing pyyaml must DISABLE the feature (extra paths
# ignored), never crash the app at import/probe/boot. requirements.txt now lists
# it directly so fresh installs get it.
try:
    import yaml as _yaml
except Exception:  # pragma: no cover - the _yaml is None branch is tested via monkeypatch
    _yaml = None
    logger.info(
        "PyYAML not installed - extra_model_paths.yaml support disabled "
        "(models resolve from <ComfyUI>/models only). Run `pip install pyyaml` to enable."
    )

YAML_FILENAME = 'extra_model_paths.yaml'

# folder_paths.supported_pt_extensions narrowed to what this app's loaders can
# actually consume, plus .gguf (see module docstring). endswith() takes a tuple.
_MODEL_EXTENSIONS = ('.safetensors', '.sft', '.gguf')

# folder_paths.map_legacy — the canonical alias source (unet↔diffusion_models,
# clip↔text_encoders). Query and yaml keys are both normalised through it.
_LEGACY = {'unet': 'diffusion_models', 'clip': 'text_encoders'}

# folder_paths default roots as subdirs of <base>/models, keyed by CANONICAL type.
# ('loras' is special-cased in _default_roots to honour the app's loras_dir override.)
_DEFAULT_SUBDIRS = {
    'checkpoints': ('checkpoints',),
    'loras': ('loras',),
    'vae': ('vae',),
    'text_encoders': ('text_encoders', 'clip'),
    'diffusion_models': ('unet', 'diffusion_models'),
}

_lock = threading.Lock()
_cache = {'key': None, 'data': {}}
_warned = set()   # (path, kind) already logged, so a bad file warns once not per probe


def _canon(folder_type: str) -> str:
    return _LEGACY.get(folder_type, folder_type)


def _warn_once(path: str, kind: str, message: str) -> None:
    marker = (path, kind)
    if marker in _warned:
        return
    _warned.add(marker)
    logger.warning('%s: %s', YAML_FILENAME, message)


def _models_dir():
    try:
        d = cfg.comfyui_dir('models')
    except Exception:
        return None
    return os.path.normpath(str(d)) if d else None


def _default_roots(canon: str) -> list[str]:
    """The <base>/models default roots for a canonical folder type. `loras` honours
    the app's dedicated ``comfyui.loras_dir`` override (the app allows one; other
    types don't), so its default matches exactly what the consumers use today."""
    if canon == 'loras':
        try:
            d = cfg.comfyui_dir('loras')
        except Exception:
            d = None
        return [os.path.normpath(str(d))] if d else []
    models = _models_dir()
    if not models:
        return []
    return [os.path.normpath(os.path.join(models, sub))
            for sub in _DEFAULT_SUBDIRS.get(canon, (canon,))]


def _yaml_path():
    """``<ComfyUI base>/extra_model_paths.yaml`` (next to main.py, where ComfyUI
    itself looks), or None when base_dir is unset. Uses the SAME base the models
    dir derives from (``comfyui.base_dir``) so the yaml location and the default
    model roots can never point at different trees."""
    base = (cfg.get('comfyui.base_dir') or '').strip()
    return os.path.join(base, YAML_FILENAME) if base else None


def _parse(path: str) -> dict:
    """Parse the yaml into ``{canonical_type: [(abs_root, is_default), ...]}`` in
    declaration order, mirroring load_extra_path_config. Never raises."""
    try:
        # utf-8-sig tolerates a BOM (more robust than ComfyUI's plain utf-8).
        with open(path, 'r', encoding='utf-8-sig') as fh:
            config = _yaml.safe_load(fh)
    except (OSError, _yaml.YAMLError) as e:
        _warn_once(path, 'parse', f'could not read/parse ({e}) - ignoring extra model paths')
        return {}
    if config is None:
        return {}
    if not isinstance(config, dict):
        _warn_once(path, 'shape', 'top level is not a mapping - ignoring extra model paths')
        return {}
    yaml_dir = os.path.dirname(os.path.abspath(path))
    out: dict[str, list] = {}
    for profile_name, conf in config.items():
        if conf is None:
            continue
        if not isinstance(conf, dict):
            _warn_once(path, f'profile:{profile_name}', f'profile {profile_name!r} is not a mapping - skipped')
            continue
        conf = dict(conf)   # copy so pop() doesn't touch safe_load's structure
        base_path = conf.pop('base_path', None)
        if base_path is not None:
            base_path = os.path.expandvars(os.path.expanduser(str(base_path)))
            if not os.path.isabs(base_path):
                base_path = os.path.abspath(os.path.join(yaml_dir, base_path))
        is_default = bool(conf.pop('is_default', False))
        for ftype, value in conf.items():
            if value is None:
                continue
            canon = _canon(str(ftype))
            for line in str(value).split('\n'):
                if len(line) == 0:      # faithful to ComfyUI: empty lines only
                    continue
                full = line
                if base_path:
                    full = os.path.join(base_path, full)
                elif not os.path.isabs(full):
                    full = os.path.abspath(os.path.join(yaml_dir, line))
                out.setdefault(canon, []).append((os.path.normpath(full), is_default))
    return out


def _extra_config() -> dict:
    """Cached parse keyed on (yaml path, mtime): re-parsed only when the file
    changes on disk (no restart needed, no re-parse per probe). ``{}`` when the
    file is absent, PyYAML is missing, or base_dir is unset."""
    path = _yaml_path()
    if not path or _yaml is None:
        return {}
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {}   # absent → not cached (so creating it later is picked up next call)
    key = (path, mtime)
    with _lock:
        if _cache['key'] == key:
            return _cache['data']
    data = _parse(path)
    with _lock:
        _cache['key'] = key
        _cache['data'] = data
    return data


def clear_cache() -> None:
    """Drop the parse cache (test hygiene; not needed in production — the mtime key
    self-invalidates when the file or base_dir changes)."""
    with _lock:
        _cache['key'] = None
        _cache['data'] = {}


def extra_roots(folder_type: str) -> list[str]:
    """The EXTRA roots (from the yaml) for a folder type, WITHOUT the default
    ``<base>/models`` roots. ``is_default`` entries first, then declaration order,
    de-duplicated. ``[]`` with no yaml — so callers that append these to their own
    base scan are byte-for-byte unchanged when no yaml exists."""
    entries = _extra_config().get(_canon(folder_type), [])
    ordered = [p for p, d in entries if d] + [p for p, d in entries if not d]
    seen, out = set(), []
    for p in ordered:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def search_roots(folder_type: str) -> list[str]:
    """All roots for a folder type — default ``<base>/models`` roots plus extras —
    in ComfyUI's own priority order, replicating folder_paths.add_model_folder_path
    (default roots first; each extra inserted at front when ``is_default`` else
    appended; duplicates moved to front when default). With no yaml this is exactly
    the historical default roots, so consumers that scan these behave identically."""
    canon = _canon(folder_type)
    roots = list(_default_roots(canon))
    for path, is_default in _extra_config().get(canon, []):
        if path in roots:
            if is_default and roots and roots[0] != path:
                roots.remove(path)
                roots.insert(0, path)
        elif is_default:
            roots.insert(0, path)
        else:
            roots.append(path)
    return roots


def _recursive_models(root: str):
    """Yield ``(rel_name, abs_path)`` for every model file under ``root`` (os.walk,
    followlinks), mirroring folder_paths.recursive_search: ``rel_name`` is the path
    relative to ``root`` with the OS separator and subfolders included."""
    if not os.path.isdir(root):
        return
    for dirpath, _subdirs, filenames in os.walk(root, followlinks=True):
        for fn in filenames:
            if fn.lower().endswith(_MODEL_EXTENSIONS):
                ab = os.path.join(dirpath, fn)
                yield os.path.relpath(ab, root), ab


def list_models(folder_type: str) -> list[tuple[str, str]]:
    """``[(rel_name, abs_path)]`` for every model file across the search roots of a
    folder type — the faithful mirror of folder_paths.get_filename_list. ``rel_name``
    is exactly the string a workflow loader node expects. De-duplicated by
    ``rel_name`` (highest-priority root wins), roots scanned in priority order."""
    seen, out = set(), []
    for root in search_roots(folder_type):
        for rel, ab in _recursive_models(root):
            if rel in seen:
                continue
            seen.add(rel)
            out.append((rel, ab))
    return out
