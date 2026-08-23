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
.safetensors,.pkl,.sft}. We list ``{.safetensors, .sft, .gguf}`` and stay narrow
on the rest so ``picker == probe == resolver``.

``.gguf`` is the exception that is listed but NOT loadable: core ComfyUI has no
``.gguf`` in ``supported_pt_extensions``, so it never scans such a file in ANY
model root, and the shipped graphs emit core ``UNETLoader``, which cannot read one
even when the ComfyUI-GGUF pack is installed (that pack adds a separate
``UnetLoaderGGUF`` node, which nothing here emits). Dropping it would hide a file
users can see on disk and turn a nameable problem into a silent absence; keeping
it is only honest because ``utils.comfyui.unavailable_model_files`` states the
extension as the cause before a job is queued (naniii2352, Discord).

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

# What a loader can actually OPEN — the same tuple minus .gguf. Listing and
# loading are two different questions and conflating them shipped a real bug:
# a base resolver picks a file by NAME (the first one containing 'turbo'), and
# `krea2_turbo-Q4_K_M.gguf` contains 'turbo'. So dropping that file into a krea
# folder was enough for the app to start choosing, on its own, a model ComfyUI
# cannot read — mid-batch, with no setting touched. Reported by naniii2352
# (Discord) as "it generated half the images and then started throwing a gguf
# error"; the batch straddled the moment he copied the file in.
# Any code that CHOOSES a file must filter on this; code that merely LISTS may
# use _MODEL_EXTENSIONS so the user still sees what is on disk.
LOADABLE_MODEL_EXTENSIONS = ('.safetensors', '.sft')


def is_loadable_model(name: str) -> bool:
    """True when a loader node could actually open this file.

    Deliberately a function and not an inline endswith(): every resolver that
    picks a base model must go through the same predicate, or the next format
    we list-but-cannot-load repeats this bug somewhere else.
    """
    return str(name or '').lower().endswith(LOADABLE_MODEL_EXTENSIONS)


# What a LISTER shows, as opposed to what a resolver may CHOOSE: `.gguf` is on
# disk, ComfyUI's GGUF pack loads it, and hiding it would make the user's own
# folder look wrong. `is_loadable_model` is the narrower predicate the choosers
# use — the two are deliberately different sets, and the comment above says why.
MODEL_FILE_SUFFIXES = ('.safetensors', '.gguf', '.sft')


# ComfyUI's ONE directory exclusion, from
# `folder_paths.get_filename_list_` → `recursive_search(x, excluded_dir_names=[".git"])`.
# Measured against a live 0.30.1 instance: a `.safetensors` planted under
# `models/unet/.git/` never appears in `GET /object_info`, while the same file in
# a dot-directory that is NOT `.git` does. Listing one would be a trap — ComfyUI
# validates a loader's combo value against this very list, so the job would be
# refused at queue time for a file the app had offered.
EXCLUDED_DIR_NAMES = frozenset({'.git'})


# --- family scans -------------------------------------------------------------
# Four functions used to walk these same folders looking for one family's
# weights: the two Studio listers (`comfyui.get_krea_models`,
# `comfyui.get_zimage_models`) and the two Generate resolvers'
# (`krea_edit_helper._krea_unet_folders`, `klein_edit_helper._klein_unet_folders`).
# Four hand-written walks over one folder layout, and their drift has already
# shipped a bug: two of them answered differently about the SAME folder, so a
# file one screen offered was a file the other refused.
#
# They are folded onto the two below. Not onto ONE, because the two SHAPES are
# genuinely different and pretending otherwise would cost more than it saves:
#
#   * a lister returns flat relative names, because the picker's value is a path
#     a loader accepts;
#   * a resolver returns (prefix, [names]) groups, because it walks roots in
#     ComfyUI's own priority order and has to say WHICH folder a candidate came
#     from.
#
# The SHAPE is the only difference left. DEPTH used to be a second one — the
# lister walked the tree, the resolver read one level — and that was a bug, not a
# design: a model two folders down was offered by the Test Studio and refused by
# Generate. Both now walk to any depth, because that is what ComfyUI does. See
# the note on `scan_family_folders` for the measurement.
#
# What they must not do is disagree about the rules — which extensions count,
# what makes a folder this family's, whether a file sitting at a root counts.
# Those are the arguments below, so a difference between two families is now
# something you can read in one call rather than diff across four loops.

def scan_family_tree(roots, dir_tokens, *, root_file_accept=None, accept=None,
                     suffixes=MODEL_FILE_SUFFIXES):
    """Model files under `roots` whose RELATIVE DIRECTORY carries one of
    `dir_tokens` (case-insensitive, at any depth), as names relative to their
    root — sorted, de-duplicated, joined with the separator of the tree actually
    walked (``os.sep``; the queue respells them for the target ComfyUI).

    `root_file_accept(filename) -> bool` decides files sitting at the ROOT of a
    search folder, where there is no directory to carry the claim. ``None`` means
    a root file is never listed: that is a real difference between families (a
    `diffusion_models` root also holds Z-Image, FLUX and Klein weights), not an
    oversight, so it is an argument rather than a hardcoded rule.

    `accept(filename) -> bool` is the family's exclusion list, and it applies
    EVERYWHERE — root and subfolder alike. It is separate from
    `root_file_accept` on purpose: one answers "does this file at a root belong
    to the family at all", the other "is this file, wherever it sits, one the
    family knows is unusable". Folding the second into the first is what made
    the same checkpoint refused at a root and accepted one folder down.

    `followlinks=True` mirrors `folder_paths.recursive_search`. It is inert on
    Windows — Python reports `os.path.islink() is False` for a junction, so the
    default would descend anyway — and load-bearing everywhere else: on Linux,
    and therefore on the rented cloud trainers, a symlinked model folder is a
    real symlink. A defect that can only appear on the rented machine is one
    nobody would ever find locally.
    """
    out = []
    for base_dir in roots or []:
        if not os.path.isdir(base_dir):
            continue
        for root, dirs, files in os.walk(base_dir, followlinks=True):
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_NAMES]
            rel_dir = os.path.relpath(root, base_dir)
            at_root = rel_dir == '.'
            if at_root:
                if root_file_accept is None:
                    continue
            elif not any(t in rel_dir.lower() for t in dir_tokens):
                continue
            for f in files:
                if not f.lower().endswith(suffixes):
                    continue
                if at_root and not root_file_accept(f):
                    continue
                if accept is not None and not accept(f):
                    continue
                out.append(f if at_root else os.path.join(rel_dir, f))
    return sorted(set(out))


def scan_family_folders(roots, dir_tokens, *, accept=None,
                        suffixes=MODEL_FILE_SUFFIXES):
    """``[(prefix, [filenames])]`` at ANY depth under each root, in the order the
    roots were given (ComfyUI's own priority order).

    `prefix` is a relative DIRECTORY whose path carries one of `dir_tokens`
    (case-insensitive, in any component — so ``Krea/nested`` qualifies through
    its parent, exactly as it does for `scan_family_tree`), or ``''`` for a file
    dropped straight into the root of a search folder — flat / Stability-Matrix
    installs have no subfolder, and ``os.path.join('', name) == name`` is exactly
    what a UNETLoader loads.

    DEPTH, and why this walks the tree
    ----------------------------------
    This read ONE level until 2026-08-04, and that was the bug. ComfyUI descends
    without limit: `folder_paths.recursive_search` is a plain `os.walk`, and the
    name it publishes is `os.path.relpath(file, root)` — subfolder included — so a
    model two folders down is listed AND loadable (`get_full_path` joins that same
    relative string back onto the root). Measured on a live 0.30.1 instance: fake
    weights planted at depth 2 and at depth 4 both appeared in
    `GET /object_info` under their full relative name.

    The Studio's lister already walked the tree, so the two halves disagreed about
    the same folder. What that cost, stated at the precision it was checked at:

      * REACHED TODAY — `krea.base_model` is a free-text field whose own help text
        invites "unless you own several Krea builds" to name one. Name a file more
        than one folder down and `resolve_krea_unet` matched its basename against
        a one-level scan, found nothing, logged "not found under any krea folder"
        and fell through to `elect_krea_base`. The job then ran on a DIFFERENT
        base than the one typed, and the image looked plausible, so nothing ever
        raised the question. A setting silently ignored is worse than one that
        errors.
      * LATENT — `klein_model_on_disk` returned None for the same shape, which its
        caller turns into "the Klein model chosen for this dataset is no longer on
        disk" about a file that IS on disk. Not reachable from the UI today: the
        Klein picker's choices come from `capabilities._scan_models`, a THIRD
        scanner that is still one-level (`routes/datasets.py`, klein choices), so
        a deep model is never offered there in the first place. Fixed here anyway
        because the invariant that function documents was false, and because the
        remaining gap is now in one place instead of two.

    ORDER, which is behaviour and not cosmetics
    -------------------------------------------
    Subfolder groups are ordered by the rule `resolve_model_file` already
    documents — shortest relative path first, then alphabetical — so a shallower
    folder always wins a basename collision against a deeper one. Deliberately the
    SAME rule and not a second one: two tie-breaks would only ever diverge on
    collisions, which is to say rarely, which is to say long after the change.
    For the depth-1 layouts every install has today it is character-for-character
    the previous `sorted()`, so no existing install changes its elected base.

    The root entry stays LAST within its root, which a strict "shortest path
    first" reading would put first. That position is load-bearing, not an
    oversight: `elect_krea_base` breaks ties on the candidate's INDEX and
    `resolve_klein_unet` falls back to the first loadable name, so promoting root
    files would silently re-elect a different default base on installs that hold
    both. Fixing depth is this change; re-ranking root against subfolder is not,
    and it would be invisible in exactly the same way.

    `accept(filename) -> bool` is the family's own exclusion list — the
    checkpoints that carry its token without being one of its bases. ``None`` =
    keep everything found, which is a family that has no such list, not a family
    that forgot one.
    """
    keep = accept or (lambda _name: True)
    out = []
    for base_dir in roots or []:
        if not os.path.isdir(base_dir):
            continue
        subs, root_names = [], []
        for dirpath, dirs, filenames in os.walk(base_dir, followlinks=True):
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_NAMES]
            rel_dir = os.path.relpath(dirpath, base_dir)
            names = sorted(n for n in filenames
                           if n.lower().endswith(suffixes) and keep(n))
            if rel_dir == '.':
                # At a root there is no directory to carry the claim, so the
                # FILENAME has to — a `diffusion_models` root holds every
                # family's weights at once.
                root_names = [n for n in names
                              if any(t in n.lower() for t in dir_tokens)]
            elif names and any(t in rel_dir.lower() for t in dir_tokens):
                subs.append((rel_dir, names))
        subs.sort(key=lambda group: (group[0].count(os.sep), group[0].lower()))
        out.extend(subs)
        if root_names:
            out.append(('', root_names))
    return out


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


# Folder types for which the app has a dedicated "put it exactly here" override.
# Only `loras` today: it is the only type the app WRITES into (deployed LoRAs).
_WRITE_OVERRIDE_KEYS = {'loras': 'comfyui.loras_dir'}


def write_root(folder_type: str) -> str | None:
    """The ONE root new files are WRITTEN to for a folder type. ``None`` when
    ComfyUI is unconfigured.

    Reading may span every root; writing has to pick one, and picking it silently
    is what shipped GitHub #25 (Geekswordsman): deploys and the "open LoRA folder"
    button used ``<base>/models/loras`` even when the yaml declared the real one.
    Priority, highest first:

      1. the explicit ``comfyui.loras_dir`` override — someone who filled that
         field said exactly where their files go, and no yaml may take it back;
      2. ``search_roots()[0]`` — the root ComfyUI ITSELF treats as primary, which
         is a yaml root only when that profile carries ``is_default: true``
         (that flag is precisely how ComfyUI is told "look here first");
      3. nothing else — with no yaml this IS ``<base>/models/loras``, so installs
         without an ``extra_model_paths.yaml`` are unchanged.

    Consequence worth stating out loud: a yaml root declared WITHOUT
    ``is_default`` is a secondary location for ComfyUI, so it stays secondary
    here too and deploys keep landing in ``<base>/models/loras``. Making a plain
    extra root capture the writes would hijack the very common "also read my
    A1111 folder" setup. Users who want the yaml root to receive files have two
    levers, both already theirs: ``is_default: true``, or the loras override.
    """
    canon = _canon(folder_type)
    key = _WRITE_OVERRIDE_KEYS.get(canon)
    if key:
        try:
            explicit = (cfg.get(key) or '').strip()
        except Exception:
            explicit = ''
        if explicit:
            return os.path.normpath(explicit)
    roots = search_roots(canon)
    return roots[0] if roots else None


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


def ci_resolve(root: str, rel: str):
    """The real absolute path of ``root/rel`` with each component matched
    case-INSENSITIVELY below ``root``, or None when no such entry exists.

    ComfyUI on Windows is case-insensitive and stored model values carry whatever
    casing the workflow template or the user's picker had at the time (``Z image\\``
    vs the on-disk ``z image``). On a case-SENSITIVE filesystem — Linux, and every
    cloud trainer — a plain join reads those as missing. Never escapes ``root``:
    only components of ``rel`` are followed, one directory listing at a time.

    (``lora_test_studio`` carries an identical private ``_ci_resolve``; this is the
    shared home for it. Folding the two is a follow-up, not part of this fix.)"""
    if not root or not os.path.isdir(root):
        return None
    cur = root
    for part in str(rel).split(os.sep):
        if not part or part == '.':
            continue
        nxt = os.path.join(cur, part)
        if os.path.exists(nxt):
            cur = nxt
            continue
        try:
            match = next((e for e in os.listdir(cur) if e.lower() == part.lower()), None)
        except OSError:
            return None
        if match is None:
            return None
        cur = os.path.join(cur, match)
    return cur if os.path.exists(cur) else None


def resolve_model_file(folder_type: str, ref: str):
    """Absolute path of a model ``ref`` across ALL search roots of a folder type, in
    ComfyUI's own priority order — or None. ``ref`` is a relative name (a loader
    value, possibly carrying its own subfolder) or a bare basename.

    Two phases, both walking ``search_roots`` in priority order so the answer is the
    file a running ComfyUI would load — the point of the ordering. Training on a
    twin of the file that generates, because two roots hold the same filename and we
    picked the wrong one, is invisible until the results are wrong.
      1. ``ref`` read as a path relative to each root (case-insensitively).
      2. ``ref``'s BASENAME searched recursively under each root. This phase exists
         because the checkpoint picker flattens names to a basename (the subfolder
         is lost before the value is ever stored), so a subfoldered file can only be
         found this way. Within one root the match is deterministic: the shortest
         relative path, then alphabetical.

    Absolute refs and ``..`` are refused outright: callers pass user-controlled
    values and a resolver is not the place to widen what a path may reach.
    ``None`` — never a bare-name fallback: a caller that cannot say WHICH file is
    missing pushes the failure into a subprocess that has no idea what it was asked
    for."""
    ref = str(ref or '')
    if not ref or os.path.isabs(ref) or '..' in ref.replace('\\', '/').split('/'):
        return None
    rel = ref.replace('\\', os.sep).replace('/', os.sep).strip(os.sep)
    if not rel:
        return None
    roots = search_roots(folder_type)
    for root in roots:
        hit = ci_resolve(root, rel)
        if hit and os.path.isfile(hit):
            return hit
    base = os.path.basename(rel).lower()
    for root in roots:
        found = sorted((r for r, _ab in _recursive_models(root)
                        if os.path.basename(r).lower() == base),
                       key=lambda r: (r.count(os.sep), r.lower()))
        if found:
            return os.path.join(root, found[0])
    return None


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
