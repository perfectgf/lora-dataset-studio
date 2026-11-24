"""Pure custom-node recipe contract shared by authoring tools and the runtime.

Validating a manifest must not import LDS configuration, probe ComfyUI, or load
the installation worker. Filesystem provenance is checked separately at install.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import urlsplit

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name, parse_wheel_filename

from .install import ArchiveError, _entry_relpath

_ID = re.compile(r'[a-z][a-z0-9_.-]{0,79}\Z')
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024


class NodeInstallError(ValueError):
    def __init__(self, message, *, dependencies_may_have_changed=False):
        super().__init__(message)
        self.dependencies_may_have_changed = dependencies_may_have_changed


class NodeInstallManual(NodeInstallError):
    """Nothing was installed: the recipe adds Python packages and LDS does not
    manage this ComfyUI's Python. `manual_steps` lists what the user can run."""

    def __init__(self, message, manual_steps):
        super().__init__(message)
        self.manual_steps = list(manual_steps)


@dataclass(frozen=True)
class BundledWheel:
    filename: str
    sha256: str


@dataclass(frozen=True)
class NodeRecipe:
    id: str
    version: str
    folder: str
    url: str
    sha256: str
    expected_classes: tuple[str, ...]
    requirements: tuple[str, ...] = ()
    archive_prefix: str = ''
    schema_version: int = 1
    wheels: tuple[BundledWheel, ...] = ()
    python: str = '>=3.10'


def _https(url):
    try:
        parsed = urlsplit(url)
        if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
                or parsed.fragment or parsed.port not in (None, 443)
                or any(ord(c) < 33 for c in url)):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise NodeInstallError('Node recipes and wheels require a plain HTTPS URL.') from exc
    return parsed


def _python_spec(value):
    try:
        if not isinstance(value, str) or not 0 < len(value) <= 120:
            raise ValueError
        spec = SpecifierSet(value)
        if not spec:
            raise ValueError
        return spec
    except ValueError as exc:
        raise NodeInstallError('The node recipe requires a nonempty Python version specifier (at most 120 characters).') from exc


def _validate(recipe, owner):
    if not isinstance(recipe, NodeRecipe) or recipe.schema_version != 1:
        raise NodeInstallError('Unsupported custom-node recipe schema.')
    if not isinstance(owner, str) or not _ID.fullmatch(owner):
        raise NodeInstallError('A registered plugin owner is required.')
    _python_spec(recipe.python)
    try:
        valid = (_ID.fullmatch(recipe.id) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', recipe.folder)
                 and recipe.folder not in ('__pycache__', 'site-packages')
                 and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.+-]{0,79}', recipe.version)
                 and re.fullmatch(r'[a-f0-9]{64}', recipe.sha256)
                 and isinstance(recipe.expected_classes, tuple) and recipe.expected_classes
                 and len(set(recipe.expected_classes)) == len(recipe.expected_classes)
                 and all(isinstance(n, str) and 0 < len(n) <= 160 and n == n.strip()
                         and not any(ord(c) < 32 or ord(c) == 127 for c in n)
                         for n in recipe.expected_classes)
                 and isinstance(recipe.requirements, tuple) and len(recipe.requirements) <= 100)
        if not valid:
            raise ValueError
        _entry_relpath(recipe.folder)
        if recipe.archive_prefix:
            if _entry_relpath(recipe.archive_prefix) != recipe.archive_prefix:
                raise ValueError
        seen = set()
        for text in recipe.requirements:
            text = str(text).split(' #', 1)[0].strip()
            req = Requirement(text)
            pins = list(req.specifier)
            if (req.url or req.extras or req.marker or len(pins) != 1 or pins[0].operator != '=='
                    or '*' in pins[0].version or canonicalize_name(req.name) in seen):
                raise ValueError
            seen.add(canonicalize_name(req.name))
        if not isinstance(recipe.wheels, tuple) or len(recipe.wheels) > 100:
            raise ValueError
        seen_wheels = set()
        for wheel in recipe.wheels:
            if (not isinstance(wheel, BundledWheel) or not isinstance(wheel.filename, str)
                    or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.+-]{0,235}\.whl', wheel.filename)
                    or _entry_relpath(wheel.filename) != wheel.filename
                    or not isinstance(wheel.sha256, str) or not re.fullmatch(r'[a-f0-9]{64}', wheel.sha256)):
                raise ValueError
            name, _, _, _ = parse_wheel_filename(wheel.filename)
            if name in seen_wheels or _gpu_package(name):
                raise ValueError
            seen_wheels.add(name)
    except (ValueError, TypeError, ArchiveError) as exc:
        raise NodeInstallError('The node recipe needs safe names, expected classes and exact dependency pins.') from exc
    _https(recipe.url)


def _gpu_package(name):
    return (name.startswith(('nvidia-', 'cuda-', 'cupy', 'jax-cuda', 'tensorrt', 'torch', 'pytorch-'))
            or name in {'triton', 'xformers', 'bitsandbytes', 'flash-attn', 'sageattention',
                        'onnxruntime-gpu', 'tensorflow', 'tensorflow-gpu', 'paddlepaddle-gpu'})
