"""Atomic, portable training-state bundles.

This module is deliberately byte-oriented.  It copies and hashes opaque
artifacts, but never imports torch and never deserializes pickle-backed state.
The training runtime remains the only component allowed to interpret those
files.

Published layout::

    <save_root>/.lds-state/<opaque uuid hex>/
        COMPLETE
        manifest.json
        artifacts/
            ...

Creation happens in a sibling ``.partial-<uuid>`` directory.  Every artifact is
flushed first, ``COMPLETE`` commits the hash of the canonical manifest, the
manifest is written last, and only then is the whole directory renamed.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import threading
import time
import uuid


SCHEMA = 'lds.training-state/v1'
STATE_DIRECTORY = '.lds-state'
MANIFEST_FILENAME = 'manifest.json'
COMPLETE_FILENAME = 'COMPLETE'
DEFAULT_RETENTION = 2
DEFAULT_FREE_SPACE_RESERVE_BYTES = 5 * (1 << 30)
DEFAULT_MAX_BUNDLE_BYTES = 64 * (1 << 30)
DEFAULT_MAX_STORE_BYTES = 128 * (1 << 30)
DEFAULT_PARTIAL_MAX_AGE_SECONDS = 24 * 60 * 60
STATE_LEVELS = ('weights', 'weights+optimizer', 'exact')

_BUNDLE_ID_RE = re.compile(r'^[0-9a-f]{32}$')
_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')
_SEGMENT_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')
_CAPABILITY_RE = re.compile(r'^[a-z][a-z0-9_.-]{0,63}$')
_MAX_MANIFEST_BYTES = 1 << 20
_COPY_CHUNK_BYTES = 1 << 20
_BUNDLE_OVERHEAD_BYTES = 1 << 20
_REPARSE_POINT = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x0400)
_PARTIAL_RE = re.compile(r'^\.partial-[0-9a-f]{32}$')
_COMPLETE_RE = re.compile(
    rb'^lds\.training-state/v1 sha256=([0-9a-f]{64})\n$')
_MANIFEST_KEYS = frozenset({
    'schema', 'bundle_id', 'created_at', 'created_at_ns',
    'completed_step', 'next_step', 'optimizer_updates_completed',
    'toolkit_revision', 'toolkit_runtime',
    'config_hash', 'dataset_hash', 'base_model_hash', 'network_hash',
    'capabilities', 'state_level', 'artifacts',
})
_STORE_LOCK = threading.RLock()


class TrainingStateBundleError(RuntimeError):
    """Base error carrying a stable, path-free reason code."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class InvalidBundleError(TrainingStateBundleError):
    """The bundle is missing, incomplete, malformed, or corrupt."""


class IncompatibleBundleError(TrainingStateBundleError):
    """The bundle is intact but cannot resume in the requested context."""


class RestoreTargetError(TrainingStateBundleError):
    """The restore destination is not a fresh path."""


class InsufficientSpaceError(TrainingStateBundleError):
    """A bundle would exceed its cap, store quota, or safe free-space reserve."""


@dataclass(frozen=True)
class BundleMetadata:
    """Identity and progress facts recorded alongside opaque state artifacts."""

    completed_step: int
    next_step: int
    optimizer_updates_completed: int
    toolkit_revision: str
    toolkit_runtime: Mapping[str, object]
    config_hash: str
    dataset_hash: str
    base_model_hash: str
    network_hash: str
    capabilities: Sequence[str]
    state_level: str


@dataclass(frozen=True)
class CompatibilitySpec:
    """Subset of facts a restore target requires to match.

    ``None`` means that a fact is not being checked.  Runtime mappings are
    subset-matched, which lets a caller require the material versions (for
    example torch and CUDA) without coupling to harmless extra diagnostics.
    """

    config_hash: str | None = None
    dataset_hash: str | None = None
    base_model_hash: str | None = None
    network_hash: str | None = None
    toolkit_revision: str | None = None
    toolkit_runtime: Mapping[str, object] | None = None
    required_capabilities: frozenset[str] = frozenset()
    minimum_state_level: str | None = None

    @classmethod
    def from_metadata(
        cls,
        metadata: BundleMetadata,
        *,
        required_capabilities: Sequence[str] = (),
        minimum_state_level: str | None = None,
    ) -> 'CompatibilitySpec':
        return cls(
            config_hash=metadata.config_hash,
            dataset_hash=metadata.dataset_hash,
            base_model_hash=metadata.base_model_hash,
            network_hash=metadata.network_hash,
            toolkit_revision=metadata.toolkit_revision,
            toolkit_runtime=metadata.toolkit_runtime,
            required_capabilities=frozenset(required_capabilities),
            minimum_state_level=minimum_state_level,
        )


@dataclass(frozen=True)
class ArtifactRecord:
    """One manifest artifact.  Paths are bundle-relative and never absolute."""

    name: str
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class BundleInspection:
    """Path-free inspection result suitable for service and route consumers."""

    bundle_id: str
    status: str
    integrity: str
    state_level: str | None = None
    reason: str | None = None
    size_bytes: int = 0
    capabilities: tuple[str, ...] = ()
    completed_step: int | None = None
    next_step: int | None = None
    optimizer_updates_completed: int | None = None
    created_at_ns: int = 0
    artifacts: tuple[ArtifactRecord, ...] = ()
    manifest: Mapping[str, object] | None = field(
        default=None, repr=False, compare=False)

    @property
    def restorable(self) -> bool:
        return self.status == 'complete' and self.integrity == 'verified'

    def to_ui_dict(self) -> dict:
        """Return only stable, path-free UI facts."""
        return {
            'bundle_id': self.bundle_id,
            'status': self.status,
            'integrity': self.integrity,
            'state_level': self.state_level,
            'reason': self.reason,
            'size_bytes': self.size_bytes,
            'capabilities': list(self.capabilities),
        }


@dataclass(frozen=True)
class RestoreResult:
    bundle_id: str
    restore_root: Path
    artifacts: Mapping[str, Path]
    inspection: BundleInspection


def new_bundle_id() -> str:
    """Return a non-semantic identifier safe as one directory component."""
    return uuid.uuid4().hex


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash a regular file as bytes, without interpreting its format."""
    digest, _size = _hash_regular_file(Path(path))
    return digest


def sha256_json(value: object) -> str:
    """Hash canonical JSON for config/dataset/network identity producers."""
    return hashlib.sha256(_canonical_json_bytes(value, newline=False)).hexdigest()


def resolve_bundle_path(
    save_root: str | os.PathLike[str],
    bundle_id: str,
    *,
    require_exists: bool = False,
) -> Path:
    """Resolve one strict opaque ID below ``.lds-state``.

    Separators, traversal, semantic ``step-N`` names, and symlinked bundle
    directories are rejected before any manifest is opened.
    """
    bundle_id = _validate_bundle_id(bundle_id)
    state_root = _state_root(save_root)
    candidate = state_root / bundle_id
    try:
        candidate_info = os.lstat(candidate)
    except FileNotFoundError:
        candidate_info = None
    except OSError as exc:
        raise InvalidBundleError('unsafe_bundle_path') from exc
    if (
        candidate_info is not None
        and _is_link_or_reparse_info(candidate_info)
    ):
        raise InvalidBundleError('unsafe_bundle_path')
    resolved = Path(os.path.realpath(os.path.abspath(os.fspath(candidate))))
    if resolved.parent != state_root:
        raise InvalidBundleError('unsafe_bundle_path')
    if require_exists and not resolved.is_dir():
        raise InvalidBundleError('bundle_not_found')
    return resolved


def preflight_bundle_space(
    save_root: str | os.PathLike[str],
    candidate_bytes: int,
    *,
    retention: int = DEFAULT_RETENTION,
    reserve_bytes: int = DEFAULT_FREE_SPACE_RESERVE_BYTES,
    max_bundle_bytes: int = DEFAULT_MAX_BUNDLE_BYTES,
    max_store_bytes: int | None = DEFAULT_MAX_STORE_BYTES,
    copy_multiplier: int = 1,
) -> dict[str, int]:
    """Fail before staging when a candidate cannot fit safely.

    ``copy_multiplier=1`` is the canonical writer's own peak.  A producer which
    first serialises a temporary source tree and then asks this writer to copy
    it can preflight with ``2`` before creating that source tree.
    """
    candidate_bytes = _positive_int(
        candidate_bytes, label='candidate_bytes', allow_zero=True)
    retention = _positive_int(retention, label='retention')
    reserve_bytes = _positive_int(reserve_bytes, label='reserve_bytes')
    max_bundle_bytes = _positive_int(
        max_bundle_bytes, label='max_bundle_bytes')
    copy_multiplier = _positive_int(
        copy_multiplier, label='copy_multiplier')
    if max_store_bytes is not None:
        max_store_bytes = _positive_int(
            max_store_bytes, label='max_store_bytes')
    estimated_bundle = candidate_bytes + _BUNDLE_OVERHEAD_BYTES
    if estimated_bundle > max_bundle_bytes:
        raise InsufficientSpaceError('bundle_size_limit_exceeded')

    existing = list_bundles(save_root, verify=False)
    retained = existing[:max(0, retention - 1)]
    retained_bytes = sum(
        item.size_bytes + _BUNDLE_OVERHEAD_BYTES for item in retained)
    projected_store = retained_bytes + estimated_bundle
    if max_store_bytes is not None and projected_store > max_store_bytes:
        raise InsufficientSpaceError('state_store_quota_exceeded')

    volume_path = _nearest_existing_path(_state_root(save_root))
    free_bytes = int(shutil.disk_usage(volume_path).free)
    required_free = estimated_bundle * copy_multiplier + reserve_bytes
    if free_bytes < required_free:
        raise InsufficientSpaceError('insufficient_free_space')
    return {
        'candidate_bytes': candidate_bytes,
        'estimated_bundle_bytes': estimated_bundle,
        'retained_bytes': retained_bytes,
        'projected_store_bytes': projected_store,
        'required_free_bytes': required_free,
        'free_bytes': free_bytes,
    }


def preflight_restore_space(
    save_root: str | os.PathLike[str],
    restore_bytes: int,
    *,
    destination: str | os.PathLike[str] | None = None,
    reserve_bytes: int = DEFAULT_FREE_SPACE_RESERVE_BYTES,
    copy_multiplier: int = 1,
) -> dict[str, int]:
    """Require restore peak space before any staging directory is created."""
    restore_bytes = _positive_int(
        restore_bytes, label='restore_bytes', allow_zero=True)
    reserve_bytes = _positive_int(reserve_bytes, label='reserve_bytes')
    copy_multiplier = _positive_int(
        copy_multiplier, label='copy_multiplier')
    state_root = _state_root(save_root)
    volume_target = (
        state_root
        if destination is None
        else _absolute_local_path(destination)
    )
    volume_path = _nearest_existing_path(volume_target)
    free_bytes = int(shutil.disk_usage(volume_path).free)
    required_free = restore_bytes * copy_multiplier + reserve_bytes
    if free_bytes < required_free:
        raise InsufficientSpaceError('insufficient_restore_space')
    return {
        'restore_bytes': restore_bytes,
        'copy_multiplier': copy_multiplier,
        'required_free_bytes': required_free,
        'free_bytes': free_bytes,
    }


def scavenge_stale_partials(
    save_root: str | os.PathLike[str],
    *,
    older_than_seconds: int = DEFAULT_PARTIAL_MAX_AGE_SECONDS,
    now: float | None = None,
) -> tuple[str, ...]:
    """Remove only stale, clearly uncommitted canonical staging directories.

    Any ``COMPLETE`` *or* ``manifest.json`` evidence is preserved fail-closed,
    even when malformed.  Published UUID directories and non-exact names are
    never candidates.
    """
    older_than_seconds = _positive_int(
        older_than_seconds, label='older_than_seconds', allow_zero=True)
    cutoff = (time.time() if now is None else float(now)) - older_than_seconds
    state_root = _state_root(save_root)
    try:
        entries = list(state_root.iterdir())
    except (FileNotFoundError, NotADirectoryError, OSError):
        return ()
    removed = []
    with _STORE_LOCK:
        for entry in entries:
            if not _PARTIAL_RE.fullmatch(entry.name):
                continue
            try:
                info = os.lstat(entry)
            except OSError:
                continue
            if (
                not stat.S_ISDIR(info.st_mode)
                or _is_link_or_reparse_info(info)
                or info.st_mtime > cutoff
            ):
                continue
            if (
                os.path.lexists(entry / COMPLETE_FILENAME)
                or os.path.lexists(entry / MANIFEST_FILENAME)
            ):
                continue
            try:
                _remove_tree_no_follow(entry, root=state_root)
            except OSError:
                continue
            removed.append(entry.name)
        if removed:
            _fsync_directory(state_root)
    return tuple(removed)


def create_bundle(
    save_root: str | os.PathLike[str],
    artifacts: Mapping[str, str | os.PathLike[str]],
    metadata: BundleMetadata,
    *,
    bundle_id: str | None = None,
    retention: int = DEFAULT_RETENTION,
    reserve_bytes: int = DEFAULT_FREE_SPACE_RESERVE_BYTES,
    max_bundle_bytes: int = DEFAULT_MAX_BUNDLE_BYTES,
    max_store_bytes: int | None = DEFAULT_MAX_STORE_BYTES,
) -> BundleInspection:
    """Copy, hash, fsync, and atomically publish one state bundle."""
    normalized_artifacts = _validate_artifact_sources(artifacts)
    normalized_metadata = _validate_metadata(metadata)
    if isinstance(retention, bool) or not isinstance(retention, int) or retention < 1:
        raise ValueError('retention must be a positive integer')
    bundle_id = _validate_bundle_id(bundle_id or new_bundle_id())

    with _STORE_LOCK:
        state_root = _state_root(save_root, create=True)
        scavenge_stale_partials(save_root)
        final = resolve_bundle_path(save_root, bundle_id)
        if final.exists():
            raise InvalidBundleError('bundle_id_exists')
        source_bytes = sum(
            source.stat().st_size for _name, source in normalized_artifacts)
        preflight_bundle_space(
            save_root,
            source_bytes,
            retention=retention,
            reserve_bytes=reserve_bytes,
            max_bundle_bytes=max_bundle_bytes,
            max_store_bytes=max_store_bytes,
        )
        staging = state_root / f'.partial-{new_bundle_id()}'
        try:
            staging.mkdir()
            records = []
            for name, source in normalized_artifacts:
                relative_path = f'artifacts/{name}'
                destination = _contained_artifact_path(staging, relative_path)
                digest, size = _copy_regular_file(source, destination)
                records.append({
                    'name': name,
                    'path': relative_path,
                    'size_bytes': size,
                    'sha256': digest,
                })
            _fsync_tree_directories(staging / 'artifacts')

            now_ns = time.time_ns()
            manifest = {
                'schema': SCHEMA,
                'bundle_id': bundle_id,
                'created_at': datetime.now(timezone.utc).isoformat(
                    timespec='microseconds').replace('+00:00', 'Z'),
                'created_at_ns': now_ns,
                **normalized_metadata,
                'artifacts': records,
            }
            manifest_bytes = _canonical_json_bytes(manifest, newline=True)
            marker = (
                f'{SCHEMA} sha256={hashlib.sha256(manifest_bytes).hexdigest()}\n'
            ).encode('ascii')

            # Both commit files appear only after every artifact is durable.
            _write_bytes_fsynced(staging / COMPLETE_FILENAME, marker)
            # The manifest is intentionally the final file written.
            _write_bytes_fsynced(staging / MANIFEST_FILENAME, manifest_bytes)
            _fsync_directory(staging)
            os.replace(staging, final)
            _fsync_directory(state_root)
        except BaseException:
            try:
                if os.path.lexists(staging):
                    _remove_tree_no_follow(staging, root=state_root)
            except OSError:
                pass
            raise

        _prune_bundles_locked(save_root, keep=retention)
        return verify_bundle(save_root, bundle_id)


def inspect_bundle(
    save_root: str | os.PathLike[str],
    bundle_id: str,
    *,
    expected: CompatibilitySpec | Mapping[str, object] | None = None,
    verify: bool = True,
) -> BundleInspection:
    """Inspect structure, manifest commitment, artifact hashes, and compatibility.

    Corrupt on-disk data is returned as a path-free result instead of raising.
    An unsafe caller-supplied ID still raises immediately.
    """
    bundle_id = _validate_bundle_id(bundle_id)
    expected = _coerce_expected(expected)
    inspection = None
    try:
        bundle = resolve_bundle_path(save_root, bundle_id)
    except InvalidBundleError:
        raise
    if not bundle.is_dir():
        return _failed_inspection(bundle_id, 'bundle_not_found', 'missing')
    if bundle.is_symlink():
        return _failed_inspection(bundle_id, 'unsafe_bundle_path')

    try:
        manifest, raw = _read_committed_manifest(bundle, bundle_id)
        artifacts = _parse_artifacts(manifest)
        inspection = BundleInspection(
            bundle_id=bundle_id,
            status='complete',
            integrity='unchecked',
            state_level=manifest['state_level'],
            size_bytes=sum(record.size_bytes for record in artifacts),
            capabilities=tuple(manifest['capabilities']),
            completed_step=manifest['completed_step'],
            next_step=manifest['next_step'],
            optimizer_updates_completed=manifest['optimizer_updates_completed'],
            created_at_ns=manifest['created_at_ns'],
            artifacts=artifacts,
            manifest=manifest,
        )
        del raw
        if verify:
            _verify_artifacts(bundle, artifacts)
            inspection = _replace_inspection(inspection, integrity='verified')
        reason = _compatibility_reason(manifest, expected)
        if reason:
            return _replace_inspection(
                inspection, status='incompatible', reason=reason)
        return inspection
    except InvalidBundleError as exc:
        # Once the committed manifest has been parsed, keep its path-free
        # checkpoint identity even when byte verification fails.  Callers can
        # then attach "corrupt state bundle" to the checkpoint it belongs to
        # instead of mislabelling it as a legacy weights-only save.
        if inspection is not None:
            return _replace_inspection(
                inspection,
                status='invalid',
                integrity='failed',
                reason=exc.reason,
            )
        return _failed_inspection(bundle_id, exc.reason)
    except (OSError, UnicodeError, ValueError, TypeError):
        return _failed_inspection(bundle_id, 'manifest_invalid')


def verify_bundle(
    save_root: str | os.PathLike[str],
    bundle_id: str,
    *,
    expected: CompatibilitySpec | Mapping[str, object] | None = None,
) -> BundleInspection:
    """Return a verified compatible bundle or raise a stable typed error."""
    inspection = inspect_bundle(
        save_root, bundle_id, expected=expected, verify=True)
    if inspection.status == 'incompatible':
        raise IncompatibleBundleError(inspection.reason or 'incompatible')
    if not inspection.restorable:
        raise InvalidBundleError(inspection.reason or 'bundle_invalid')
    return inspection


def list_bundles(
    save_root: str | os.PathLike[str],
    *,
    expected: CompatibilitySpec | Mapping[str, object] | None = None,
    verify: bool = False,
    include_invalid: bool = False,
) -> list[BundleInspection]:
    """List newest-first, ignoring partials and all non-opaque directory names."""
    state_root = _state_root(save_root)
    try:
        entries = list(state_root.iterdir())
    except (FileNotFoundError, NotADirectoryError, OSError):
        return []
    results = []
    for entry in entries:
        if not _BUNDLE_ID_RE.fullmatch(entry.name):
            continue
        try:
            info = os.lstat(entry)
        except OSError:
            continue
        if (
            not stat.S_ISDIR(info.st_mode)
            or _is_link_or_reparse_info(info)
        ):
            continue
        result = inspect_bundle(
            save_root, entry.name, expected=expected, verify=verify)
        if include_invalid or result.status not in ('invalid', 'missing'):
            results.append(result)
    return sorted(
        results,
        key=lambda item: (item.created_at_ns, item.completed_step or -1,
                          item.bundle_id),
        reverse=True,
    )


def checkpoint_resume_state(
    save_root: str | os.PathLike[str],
    step: int,
    *,
    expected: CompatibilitySpec | Mapping[str, object] | None = None,
    verify: bool = True,
) -> BundleInspection | None:
    """Return the newest bundle whose completed checkpoint step is ``step``."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError('step must be a non-negative integer')
    for inspection in list_bundles(
            save_root, expected=expected, verify=verify, include_invalid=True):
        if inspection.completed_step == step:
            return inspection
    return None


def stage_restore(
    save_root: str | os.PathLike[str],
    bundle_id: str,
    restore_root: str | os.PathLike[str],
    *,
    expected: CompatibilitySpec | Mapping[str, object] | None = None,
    reserve_bytes: int = DEFAULT_FREE_SPACE_RESERVE_BYTES,
) -> RestoreResult:
    """Verify and atomically materialize artifacts into a brand-new save root.

    Existing destinations, including empty directories, are refused.  Every
    copied byte is re-hashed before the staging directory is renamed.
    """
    with _STORE_LOCK:
        inspection = verify_bundle(save_root, bundle_id, expected=expected)
        preflight_restore_space(
            save_root,
            inspection.size_bytes,
            destination=restore_root,
            reserve_bytes=reserve_bytes,
        )
        bundle = resolve_bundle_path(save_root, bundle_id, require_exists=True)
        destination = _canonical_root(restore_root)
        if destination.exists() or destination.is_symlink():
            raise RestoreTargetError('restore_root_exists')
        state_root = _state_root(save_root)
        try:
            destination.relative_to(state_root)
        except ValueError:
            pass
        else:
            raise RestoreTargetError('restore_root_unsafe')
        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        staging = parent / f'.{destination.name}.partial-{new_bundle_id()}'
        if staging.exists():
            raise RestoreTargetError('restore_staging_exists')
        restored = {}
        try:
            staging.mkdir()
            for record in inspection.artifacts:
                source = _contained_artifact_path(bundle, record.path)
                target = _contained_artifact_path(staging, record.name)
                digest, size = _copy_regular_file(source, target)
                if size != record.size_bytes:
                    raise InvalidBundleError('artifact_size_mismatch')
                if digest != record.sha256:
                    raise InvalidBundleError('artifact_sha256_mismatch')
                restored[record.name] = destination / Path(
                    *PurePosixPath(record.name).parts)
            _fsync_tree_directories(staging)
            if destination.exists() or destination.is_symlink():
                raise RestoreTargetError('restore_root_exists')
            os.rename(staging, destination)
            _fsync_directory(parent)
        except BaseException:
            try:
                if os.path.lexists(staging):
                    _remove_tree_no_follow(staging, root=parent)
            except OSError:
                pass
            raise
        return RestoreResult(
            bundle_id=bundle_id,
            restore_root=destination,
            artifacts=restored,
            inspection=inspection,
        )


def _validate_bundle_id(bundle_id: object) -> str:
    if not isinstance(bundle_id, str) or not _BUNDLE_ID_RE.fullmatch(bundle_id):
        raise InvalidBundleError('invalid_bundle_id')
    return bundle_id


def _positive_int(
    value: object,
    *,
    label: str,
    allow_zero: bool = False,
) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')
    return value


def _nearest_existing_path(path: Path) -> Path:
    current = path
    while True:
        try:
            if current.exists():
                return current
        except OSError:
            pass
        if current.parent == current:
            return current
        current = current.parent


def _is_link_or_reparse_info(info: os.stat_result) -> bool:
    return bool(
        stat.S_ISLNK(info.st_mode)
        or (getattr(info, 'st_file_attributes', 0) & _REPARSE_POINT)
    )


def _remove_tree_no_follow(path: Path, *, root: Path) -> None:
    """Delete one claimed tree without traversing symlinks or junctions."""
    root = Path(os.path.abspath(os.fspath(root)))
    path = Path(os.path.abspath(os.fspath(path)))
    if path.parent != root:
        raise OSError('refusing to remove a tree outside its exact root')
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or _is_link_or_reparse_info(info):
        raise OSError('refusing to remove a linked staging tree')

    # Identical twin in training_bridge/lds_aitk_bridge_runtime.py, which runs
    # inside ai-toolkit's interpreter and cannot import app.* - the copy is the
    # coupling boundary. test_cross_interpreter_twins.py pins the two bodies.
    def remove_directory(directory: Path) -> None:
        with os.scandir(directory) as entries:
            children = list(entries)
        for entry in children:
            child = Path(entry.path)
            child_info = entry.stat(follow_symlinks=False)
            if (
                stat.S_ISDIR(child_info.st_mode)
                and not _is_link_or_reparse_info(child_info)
            ):
                remove_directory(child)
            elif stat.S_ISDIR(child_info.st_mode):
                os.rmdir(child)
            else:
                os.unlink(child)
        os.rmdir(directory)

    remove_directory(path)


def _canonical_root(path: str | os.PathLike[str]) -> Path:
    try:
        return Path(os.path.realpath(os.path.abspath(os.fspath(path))))
    except (TypeError, ValueError, OSError) as exc:
        raise ValueError('invalid save root') from exc


def _absolute_local_path(path: str | os.PathLike[str]) -> Path:
    try:
        value = Path(os.path.abspath(os.fspath(path)))
    except (TypeError, ValueError, OSError) as exc:
        raise ValueError('invalid save root') from exc
    if '\x00' in os.fspath(value):
        raise ValueError('invalid save root')
    return value


def _path_components(path: Path):
    parts = path.parts
    if not parts:
        raise InvalidBundleError('unsafe_state_root')
    current = Path(parts[0])
    yield current
    for part in parts[1:]:
        current = current / part
        yield current


def _assert_real_directory_components(path: Path) -> None:
    """Reject every existing link/junction or non-directory component."""
    for component in _path_components(path):
        try:
            info = os.lstat(component)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise InvalidBundleError('unsafe_state_root') from exc
        if (
            not stat.S_ISDIR(info.st_mode)
            or _is_link_or_reparse_info(info)
        ):
            raise InvalidBundleError('unsafe_state_root')


def _ensure_real_directory(path: Path) -> None:
    """Create missing components without knowingly traversing a reparse."""
    for component in _path_components(path):
        try:
            info = os.lstat(component)
        except FileNotFoundError:
            _assert_real_directory_components(component.parent)
            try:
                os.mkdir(component)
            except FileExistsError:
                pass
            except OSError as exc:
                raise InvalidBundleError('unsafe_state_root') from exc
            try:
                info = os.lstat(component)
            except OSError as exc:
                raise InvalidBundleError('unsafe_state_root') from exc
        except OSError as exc:
            raise InvalidBundleError('unsafe_state_root') from exc
        if (
            not stat.S_ISDIR(info.st_mode)
            or _is_link_or_reparse_info(info)
        ):
            raise InvalidBundleError('unsafe_state_root')
    _assert_real_directory_components(path)


def _state_root(
    save_root: str | os.PathLike[str],
    *,
    create: bool = False,
) -> Path:
    # The save root is canonicalized FIRST — knowingly. It comes from the
    # app's own configuration (the ai-toolkit output folder plus run names),
    # not from bundle content, and a junction in that prefix is a legitimate
    # layout: a real install moved C:\ai-toolkit\output to another drive to
    # free the system disk, and every checkpoint inspection under it died on
    # 'unsafe_state_root'. Resolving turns "unknowingly traversing a link"
    # (the thing this module must never do) into writing where the OWNER of
    # the machine pointed the folder. What stays fatal, unresolved, is any
    # link AT or UNDER the state directory itself — that is the surface a
    # bundle or a concurrent writer could swap — because the component walk
    # below runs on the RESOLVED chain, where such a link cannot hide.
    root = _canonical_root(_absolute_local_path(save_root))
    state_root = root / STATE_DIRECTORY
    if create:
        _ensure_real_directory(state_root)
    else:
        _assert_real_directory_components(state_root)
    return state_root


def _validate_artifact_name(name: object) -> str:
    if not isinstance(name, str) or not name or '\\' in name or '\x00' in name:
        raise ValueError('unsafe artifact name')
    pure = PurePosixPath(name)
    if pure.is_absolute() or not pure.parts:
        raise ValueError('unsafe artifact name')
    for segment in pure.parts:
        if segment in ('.', '..') or not _SEGMENT_RE.fullmatch(segment):
            raise ValueError('unsafe artifact name')
    normalized = pure.as_posix()
    if normalized != name or len(normalized) > 512:
        raise ValueError('unsafe artifact name')
    return normalized


def _validate_artifact_sources(
    artifacts: Mapping[str, str | os.PathLike[str]],
) -> list[tuple[str, Path]]:
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError('artifacts must be a non-empty mapping')
    normalized = []
    seen = set()
    for raw_name, raw_source in artifacts.items():
        name = _validate_artifact_name(raw_name)
        if name in seen:
            raise ValueError('duplicate artifact name')
        seen.add(name)
        source = _canonical_root(raw_source)
        try:
            mode = source.stat().st_mode
        except OSError as exc:
            raise ValueError('artifact source is not a regular file') from exc
        if not stat.S_ISREG(mode):
            raise ValueError('artifact source is not a regular file')
        normalized.append((name, source))
    return sorted(normalized, key=lambda item: item[0])


def _validate_metadata(metadata: BundleMetadata) -> dict:
    if not isinstance(metadata, BundleMetadata):
        raise TypeError('metadata must be BundleMetadata')
    for field_name in (
            'completed_step', 'next_step', 'optimizer_updates_completed'):
        value = getattr(metadata, field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'{field_name} must be a non-negative integer')
    if metadata.next_step < metadata.completed_step:
        raise ValueError('next_step cannot precede completed_step')
    # ai-toolkit labels a numbered checkpoint with the zero-based loop step K
    # after applying that step's optimizer update. Such a boundary has K+1
    # completed updates; a terminal save instead uses next step N and has N.
    if metadata.optimizer_updates_completed > metadata.completed_step + 1:
        raise ValueError(
            'optimizer_updates_completed cannot exceed completed_step + 1')
    if metadata.state_level not in STATE_LEVELS:
        raise ValueError('invalid state_level')
    for field_name in (
            'toolkit_revision', 'config_hash', 'dataset_hash',
            'base_model_hash', 'network_hash'):
        value = getattr(metadata, field_name)
        if (not isinstance(value, str) or not value or len(value) > 256
                or any(ord(char) < 32 for char in value)):
            raise ValueError(f'invalid {field_name}')
    if not isinstance(metadata.toolkit_runtime, Mapping):
        raise ValueError('toolkit_runtime must be a JSON object')
    runtime = _json_roundtrip(dict(metadata.toolkit_runtime))
    if not isinstance(runtime, dict):
        raise ValueError('toolkit_runtime must be a JSON object')
    if isinstance(metadata.capabilities, (str, bytes)):
        raise ValueError('invalid capabilities')
    try:
        capabilities = sorted(set(metadata.capabilities))
    except TypeError as exc:
        raise ValueError('invalid capabilities') from exc
    if not capabilities or any(
            not isinstance(item, str)
            or not _CAPABILITY_RE.fullmatch(item) for item in capabilities):
        raise ValueError('invalid capabilities')
    return {
        'completed_step': metadata.completed_step,
        'next_step': metadata.next_step,
        'optimizer_updates_completed': metadata.optimizer_updates_completed,
        'toolkit_revision': metadata.toolkit_revision,
        'toolkit_runtime': runtime,
        'config_hash': metadata.config_hash,
        'dataset_hash': metadata.dataset_hash,
        'base_model_hash': metadata.base_model_hash,
        'network_hash': metadata.network_hash,
        'capabilities': capabilities,
        'state_level': metadata.state_level,
    }


def _canonical_json_bytes(value: object, *, newline: bool) -> bytes:
    try:
        text = json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True,
            separators=(',', ':'))
    except (TypeError, ValueError) as exc:
        raise ValueError('value is not canonical JSON') from exc
    return (text + ('\n' if newline else '')).encode('utf-8')


def _json_roundtrip(value: object) -> object:
    return json.loads(_canonical_json_bytes(value, newline=False).decode('utf-8'))


def _contained_artifact_path(root: Path, relative_path: object) -> Path:
    if not isinstance(relative_path, str):
        raise InvalidBundleError('unsafe_artifact_path')
    try:
        name = relative_path
        if relative_path.startswith('artifacts/'):
            name = relative_path[len('artifacts/'):]
            prefix = ('artifacts',)
        else:
            prefix = ()
        safe_name = _validate_artifact_name(name)
    except ValueError as exc:
        raise InvalidBundleError('unsafe_artifact_path') from exc
    candidate = root.joinpath(*prefix, *PurePosixPath(safe_name).parts)
    root_resolved = Path(os.path.realpath(os.path.abspath(os.fspath(root))))
    candidate_resolved = Path(
        os.path.realpath(os.path.abspath(os.fspath(candidate))))
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise InvalidBundleError('unsafe_artifact_path') from exc
    return candidate_resolved


def _write_bytes_fsynced(path: Path, data: bytes) -> None:
    with open(path, 'xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _copy_regular_file(source: Path, destination: Path) -> tuple[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with open(source, 'rb') as src, open(destination, 'xb') as dst:
        before = os.fstat(src.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('artifact source is not a regular file')
        while True:
            chunk = src.read(_COPY_CHUNK_BYTES)
            if not chunk:
                break
            dst.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        dst.flush()
        os.fsync(dst.fileno())
        after = os.fstat(src.fileno())
    if (size != before.st_size or after.st_size != before.st_size
            or getattr(after, 'st_mtime_ns', None)
            != getattr(before, 'st_mtime_ns', None)):
        raise InvalidBundleError('artifact_changed_during_copy')
    return digest.hexdigest(), size


def _hash_regular_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    if path.is_symlink():
        raise InvalidBundleError('unsafe_artifact_path')
    with open(path, 'rb') as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise InvalidBundleError('artifact_not_regular')
        while True:
            chunk = handle.read(_COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _fsync_directory(path: Path) -> None:
    """Best effort: Windows does not consistently allow opening directories."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _fsync_tree_directories(root: Path) -> None:
    if not root.exists():
        return
    directories = [Path(path) for path, _dirs, _files in os.walk(root)]
    for directory in reversed(directories):
        _fsync_directory(directory)


def _reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise InvalidBundleError('manifest_duplicate_key')
        value[key] = item
    return value


def _reject_json_constant(_value):
    raise InvalidBundleError('manifest_invalid')


def _read_committed_manifest(
    bundle: Path,
    bundle_id: str,
) -> tuple[dict, bytes]:
    marker_path = bundle / COMPLETE_FILENAME
    manifest_path = bundle / MANIFEST_FILENAME
    if marker_path.is_symlink() or manifest_path.is_symlink():
        raise InvalidBundleError('unsafe_commit_file')
    if not marker_path.is_file():
        raise InvalidBundleError('complete_marker_missing')
    if not manifest_path.is_file():
        raise InvalidBundleError('manifest_missing')
    if marker_path.stat().st_size > 256:
        raise InvalidBundleError('complete_marker_invalid')
    marker = marker_path.read_bytes()
    match = _COMPLETE_RE.fullmatch(marker)
    if not match:
        raise InvalidBundleError('complete_marker_invalid')
    if manifest_path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise InvalidBundleError('manifest_too_large')
    raw = manifest_path.read_bytes()
    if not raw or hashlib.sha256(raw).hexdigest().encode('ascii') != match.group(1):
        raise InvalidBundleError('manifest_sha256_mismatch')
    try:
        manifest = json.loads(
            raw.decode('utf-8'),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except InvalidBundleError:
        raise
    except (UnicodeError, ValueError, TypeError) as exc:
        raise InvalidBundleError('manifest_invalid') from exc
    _validate_manifest(manifest, bundle_id)
    return manifest, raw


def _validate_manifest(manifest: object, bundle_id: str) -> None:
    if not isinstance(manifest, dict) or frozenset(manifest) != _MANIFEST_KEYS:
        raise InvalidBundleError('manifest_shape_invalid')
    if manifest.get('schema') != SCHEMA:
        raise InvalidBundleError('schema_unsupported')
    if manifest.get('bundle_id') != bundle_id:
        raise InvalidBundleError('bundle_id_mismatch')
    if (not isinstance(manifest.get('created_at'), str)
            or not manifest['created_at'].endswith('Z')):
        raise InvalidBundleError('manifest_created_at_invalid')
    if (isinstance(manifest.get('created_at_ns'), bool)
            or not isinstance(manifest.get('created_at_ns'), int)
            or manifest['created_at_ns'] <= 0):
        raise InvalidBundleError('manifest_created_at_invalid')
    for key in ('completed_step', 'next_step', 'optimizer_updates_completed'):
        value = manifest.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise InvalidBundleError('manifest_progress_invalid')
    if manifest['next_step'] < manifest['completed_step']:
        raise InvalidBundleError('manifest_progress_invalid')
    if manifest['optimizer_updates_completed'] > manifest['completed_step'] + 1:
        raise InvalidBundleError('manifest_progress_invalid')
    if manifest.get('state_level') not in STATE_LEVELS:
        raise InvalidBundleError('state_level_invalid')
    for key in (
            'toolkit_revision', 'config_hash', 'dataset_hash',
            'base_model_hash', 'network_hash'):
        value = manifest.get(key)
        if not isinstance(value, str) or not value or len(value) > 256:
            raise InvalidBundleError('manifest_identity_invalid')
    if not isinstance(manifest.get('toolkit_runtime'), dict):
        raise InvalidBundleError('manifest_runtime_invalid')
    capabilities = manifest.get('capabilities')
    if (not isinstance(capabilities, list) or not capabilities
            or capabilities != sorted(set(capabilities))
            or any(not isinstance(item, str)
                   or not _CAPABILITY_RE.fullmatch(item)
                   for item in capabilities)):
        raise InvalidBundleError('manifest_capabilities_invalid')
    if not isinstance(manifest.get('artifacts'), list) or not manifest['artifacts']:
        raise InvalidBundleError('manifest_artifacts_invalid')


def _parse_artifacts(manifest: Mapping[str, object]) -> tuple[ArtifactRecord, ...]:
    records = []
    seen_names = set()
    seen_paths = set()
    for raw in manifest['artifacts']:
        if not isinstance(raw, dict) or frozenset(raw) != frozenset({
                'name', 'path', 'size_bytes', 'sha256'}):
            raise InvalidBundleError('manifest_artifacts_invalid')
        try:
            name = _validate_artifact_name(raw['name'])
        except ValueError as exc:
            raise InvalidBundleError('unsafe_artifact_path') from exc
        expected_path = f'artifacts/{name}'
        if raw['path'] != expected_path:
            raise InvalidBundleError('unsafe_artifact_path')
        size = raw['size_bytes']
        digest = raw['sha256']
        if (isinstance(size, bool) or not isinstance(size, int) or size < 0
                or not isinstance(digest, str)
                or not _SHA256_RE.fullmatch(digest)):
            raise InvalidBundleError('manifest_artifacts_invalid')
        if name in seen_names or expected_path in seen_paths:
            raise InvalidBundleError('manifest_artifact_duplicate')
        seen_names.add(name)
        seen_paths.add(expected_path)
        records.append(ArtifactRecord(name, expected_path, size, digest))
    if [record.name for record in records] != sorted(
            record.name for record in records):
        raise InvalidBundleError('manifest_artifact_order_invalid')
    return tuple(records)


def _verify_artifacts(bundle: Path, records: tuple[ArtifactRecord, ...]) -> None:
    expected_paths = {record.path for record in records}
    actual_paths = set()
    artifacts_root = bundle / 'artifacts'
    if artifacts_root.is_symlink() or not artifacts_root.is_dir():
        raise InvalidBundleError('artifacts_directory_missing')
    for directory, dirs, files in os.walk(artifacts_root, followlinks=False):
        directory_path = Path(directory)
        for dirname in list(dirs):
            child = directory_path / dirname
            if child.is_symlink():
                raise InvalidBundleError('unsafe_artifact_path')
        for filename in files:
            path = directory_path / filename
            if path.is_symlink():
                raise InvalidBundleError('unsafe_artifact_path')
            relative = path.relative_to(bundle).as_posix()
            actual_paths.add(relative)
    if actual_paths != expected_paths:
        raise InvalidBundleError('artifact_set_mismatch')
    allowed_top_level = {
        COMPLETE_FILENAME, MANIFEST_FILENAME, 'artifacts',
    }
    if {entry.name for entry in bundle.iterdir()} != allowed_top_level:
        raise InvalidBundleError('bundle_entry_set_mismatch')
    for record in records:
        path = _contained_artifact_path(bundle, record.path)
        try:
            digest, size = _hash_regular_file(path)
        except FileNotFoundError as exc:
            raise InvalidBundleError('artifact_missing') from exc
        if size != record.size_bytes:
            raise InvalidBundleError('artifact_size_mismatch')
        if digest != record.sha256:
            raise InvalidBundleError('artifact_sha256_mismatch')


def _coerce_expected(
    expected: CompatibilitySpec | Mapping[str, object] | None,
) -> CompatibilitySpec | None:
    if expected is None:
        return None
    if isinstance(expected, CompatibilitySpec):
        return _validate_compatibility_spec(expected)
    if not isinstance(expected, Mapping):
        raise TypeError('expected must be CompatibilitySpec or a mapping')
    allowed = {
        'config_hash', 'dataset_hash', 'base_model_hash', 'network_hash',
        'toolkit_revision', 'toolkit_runtime', 'required_capabilities',
        'minimum_state_level',
    }
    if set(expected) - allowed:
        raise ValueError('unknown compatibility field')
    values = dict(expected)
    if 'required_capabilities' in values:
        values['required_capabilities'] = frozenset(
            values['required_capabilities'] or ())
    return _validate_compatibility_spec(CompatibilitySpec(**values))


def _validate_compatibility_spec(
    spec: CompatibilitySpec,
) -> CompatibilitySpec:
    if isinstance(spec.required_capabilities, (str, bytes)):
        raise ValueError('invalid required_capabilities')
    required = frozenset(spec.required_capabilities)
    if any(
            not isinstance(item, str)
            or not _CAPABILITY_RE.fullmatch(item) for item in required):
        raise ValueError('invalid required_capabilities')
    if (spec.minimum_state_level is not None
            and spec.minimum_state_level not in STATE_LEVELS):
        raise ValueError('invalid minimum_state_level')
    if (spec.toolkit_runtime is not None
            and not isinstance(spec.toolkit_runtime, Mapping)):
        raise ValueError('toolkit_runtime compatibility must be a mapping')
    return _normalized_compatibility_spec(spec, required)


def _normalized_compatibility_spec(
    spec: CompatibilitySpec,
    required: frozenset[str],
) -> CompatibilitySpec:
    """Normalize caller-owned capability sequences without mutating them."""
    return CompatibilitySpec(
        config_hash=spec.config_hash,
        dataset_hash=spec.dataset_hash,
        base_model_hash=spec.base_model_hash,
        network_hash=spec.network_hash,
        toolkit_revision=spec.toolkit_revision,
        toolkit_runtime=spec.toolkit_runtime,
        required_capabilities=required,
        minimum_state_level=spec.minimum_state_level,
    )


def _compatibility_reason(
    manifest: Mapping[str, object],
    expected: CompatibilitySpec | Mapping[str, object] | None,
) -> str | None:
    spec = _coerce_expected(expected)
    if spec is None:
        return None
    for key in (
            'config_hash', 'dataset_hash', 'base_model_hash', 'network_hash',
            'toolkit_revision'):
        wanted = getattr(spec, key)
        if wanted is not None and manifest[key] != wanted:
            return f'{key}_mismatch'
    if spec.toolkit_runtime is not None:
        actual_runtime = manifest['toolkit_runtime']
        for key, wanted in spec.toolkit_runtime.items():
            if key not in actual_runtime or actual_runtime[key] != wanted:
                return 'toolkit_runtime_mismatch'
    required = frozenset(spec.required_capabilities)
    if not required.issubset(manifest['capabilities']):
        return 'capability_missing'
    if spec.minimum_state_level is not None:
        if STATE_LEVELS.index(manifest['state_level']) < STATE_LEVELS.index(
                spec.minimum_state_level):
            return 'state_level_insufficient'
    return None


def _failed_inspection(
    bundle_id: str,
    reason: str,
    status: str = 'invalid',
) -> BundleInspection:
    return BundleInspection(
        bundle_id=bundle_id,
        status=status,
        integrity='failed',
        reason=reason,
    )


def _replace_inspection(
    inspection: BundleInspection,
    **changes,
) -> BundleInspection:
    values = {
        name: getattr(inspection, name)
        for name in BundleInspection.__dataclass_fields__
    }
    values.update(changes)
    return BundleInspection(**values)


def _prune_bundles_locked(
    save_root: str | os.PathLike[str],
    *,
    keep: int,
) -> tuple[str, ...]:
    complete = list_bundles(save_root, verify=False)
    removed = []
    for inspection in complete[keep:]:
        try:
            path = resolve_bundle_path(
                save_root, inspection.bundle_id, require_exists=True)
            _remove_tree_no_follow(path, root=_state_root(save_root))
            removed.append(inspection.bundle_id)
        except (OSError, InvalidBundleError):
            continue
    if removed:
        _fsync_directory(_state_root(save_root))
    return tuple(removed)
