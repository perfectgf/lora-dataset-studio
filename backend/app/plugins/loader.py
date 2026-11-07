"""Discover, enable, order and register plugins at boot.

Sources under ``bundled/`` are discovered only in explicit development mode.
External packages live under the configured data directory. The host must call
this loader after its access guards and before handling requests; importing this
package does not activate plugins or expose installation/restart endpoints.

A plugin that fails to import or to register is marked ``error`` with a
paste-safe one-line reason, logged with its traceback, and the app continues
without it. Nothing about a plugin can take a core surface down.

An enabled bundled plugin blocked only by a dependency's import/registration
failure may expose ``register_recovery(ctx)``. This restricted context lends
boot hooks, workers and a disable blocker for existing liabilities; the plugin stays disabled
and normal rental admission must reject it. Explicit disablement never invokes
this fallback. Its package initializer must be importable without dependencies.
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
from pathlib import Path

from .. import config as cfg
from ..utils.redact import redact_tokens, redact_user_paths
from .api import LDS_PLUGIN_API_MAJOR, PluginContext
from .manifest import MANIFEST_NAME, ManifestError, load_manifest
from .legacy import discover_legacy, legacy_dir, legacy_manifest, register_legacy
from .registry import PluginRecord, PluginRegistry, set_active
from . import storage
from . import official

log = logging.getLogger(__name__)

KILL_SWITCH = 'LDS_PLUGINS'


def bundled_dir() -> Path:
    override = os.environ.get('LDS_BUNDLED_DIR')
    if override:
        return Path(override)
    from .discovery import development_bundles
    return cfg.REPO_ROOT / ('bundled' if development_bundles() else '.no-bundled-plugins')


def external_dir() -> Path:
    override = os.environ.get('LDS_PLUGINS_DIR')
    return Path(override) if override else cfg.data_dir() / 'plugins'


def plugin_data_dir(record: PluginRecord) -> Path:
    """Persistent plugin state, independent of installed and recovery code."""
    return storage.data_dir(record.id)


def _paste_safe(text: str, limit: int = 300) -> str:
    return redact_user_paths(redact_tokens(str(text)))[:limit]


def _discover(registry: PluginRegistry, root: Path, *, bundled: bool) -> None:
    if not root.is_dir():
        return
    for name in sorted(os.listdir(root)):
        path = root / name
        if not path.is_dir() or name.startswith(('.', '_')):
            continue
        if bundled:
            try:
                if official.bundled_retired(name):
                    continue
            except (OSError, ValueError):
                registry.invalid.append({'dir': name, 'reason': 'The migration marker cannot be verified; the old bundled copy was not loaded.'})
                continue
        if not bundled and name.endswith('.data-kept'):
            continue  # Recovery data from the old installer, never a plugin.
        if not (path / MANIFEST_NAME).is_file():
            if bundled:
                registry.misplaced.append({'dir': name, 'reason': f'no {MANIFEST_NAME}'})
            else:
                registry.invalid.append({'dir': name, 'reason': f'no {MANIFEST_NAME}'})
            continue
        try:
            manifest = load_manifest(path, official=bool(
                not bundled and official.installed_provenance(name, path)))
        except ManifestError as exc:
            registry.invalid.append({'dir': name, 'reason': _paste_safe(exc)})
            continue
        if bundled and not manifest.bundled:
            registry.misplaced.append({
                'dir': name, 'id': manifest.id,
                'reason': 'this folder ships with the app and is replaced by every update — '
                          f'move the plugin to {registry.dirs["external"]}'})
            continue
        if not bundled and manifest.bundled:
            registry.invalid.append({'dir': name, 'id': manifest.id,
                                     'reason': 'a bundled manifest cannot be installed here'})
            continue
        if manifest.id != name:
            registry.invalid.append({'dir': name, 'id': manifest.id,
                                     'reason': f'the folder must be named after the id {manifest.id!r}'})
            continue
        if manifest.id in registry.records:
            registry.invalid.append({'dir': name, 'reason': f'plugin id {manifest.id!r} is already taken'})
            continue
        registry.records[manifest.id] = PluginRecord(
            id=manifest.id, manifest=manifest, dir=str(path), bundled=manifest.bundled,
            # The UI route serves files relative to the folder of the manifest's
            # `frontend` entry, so the entry's URL is its basename (measured: the
            # full relative path doubled the folder and answered 404).
            frontend_url=(f"/api/plugins/{manifest.id}/ui/boot/{registry.boot_id}/{manifest.frontend.rsplit('/', 1)[-1]}"
                          if (manifest.frontend and not manifest.bundled) else None),
            styles=[f"/api/plugins/{manifest.id}/ui/boot/{registry.boot_id}/"
                    + style[len(manifest.frontend.rsplit('/', 1)[0]) + 1:]
                    for style in manifest.contract.get('frontend_styles', [])]
            if manifest.frontend and not manifest.bundled else [])


def _enabled_map() -> dict:
    raw = cfg.get('plugins.enabled')
    return raw if isinstance(raw, dict) else {}


def _order(records: dict) -> tuple[list, list]:
    """Topological order of the loadable records by ``requires``; the second
    list is what sits on a cycle."""
    pending = {pid for pid, r in records.items() if r.state == 'pending'}
    order, remaining = [], set(pending)
    while remaining:
        ready = sorted(pid for pid in remaining
                       if all(req not in remaining for req in records[pid].manifest.requires))
        if not ready:
            return order, sorted(remaining)
        order.extend(ready)
        remaining -= set(ready)
    return order, []


class _RecoveryContext:
    """Bundled recovery stages workers and their disable blocker, never features.

    This is an API boundary, not a sandbox: bundled Python already runs with
    the app's trust. Staging makes a failed recovery registration atomic.
    """

    def __init__(self, app, plugin_id):
        self.app, self.id = app, plugin_id
        self._boot_hooks, self._workers, self._disable_blockers = [], [], []

    def register_boot_hook(self, fn):
        if not callable(fn):
            raise TypeError('a recovery boot hook must be callable')
        self._boot_hooks.append((self.id, fn))

    def register_worker(self, name, fn):
        if not callable(fn):
            raise TypeError('a recovery worker must be callable')
        self._workers.append((self.id, name, fn))


    def register_disable_blocker(self, fn):
        if not callable(fn):
            raise TypeError('a recovery disable blocker must be callable')
        self._disable_blockers.append(fn)


def _only_failed_requirements(record, records, failed_at_load, seen=frozenset()):
    """True only for dependency import/registration failures, including chains.

    Missing, incompatible, cyclic and explicitly disabled requirements never
    authorize recovery. An enabled lane needs at least one actual failure.
    """
    if record.id in seen:
        return False
    seen = seen | {record.id}
    failed = False
    for required in record.manifest.requires:
        dep = records.get(required)
        if dep is None or not dep.enabled:
            return False
        if dep.state == 'loaded':
            continue
        if dep.state == 'error' and required in failed_at_load:
            failed = True
        elif dep.state == 'disabled' and _only_failed_requirements(
                dep, records, failed_at_load, seen):
            failed = True
        else:
            return False
    return failed


def _register_recovery(app, registry, failed_at_load):
    """Opt-in bundled safety workers survive a failed dependency registration.

    ``register_recovery(ctx)`` cannot make the record loaded. Its workers must
    recover existing work only; normal admission still requires loaded plugins.
    """
    for record in registry.records.values():
        if (not (record.bundled or record.manifest.official) or not record.enabled or record.state != 'disabled'
                or not record.manifest.python_package
                or not _only_failed_requirements(record, registry.records, failed_at_load)):
            continue
        try:
            if record.dir not in sys.path:
                sys.path.append(record.dir)
            module = importlib.import_module(record.manifest.python_package)
            recover = getattr(module, 'register_recovery', None)
            if recover is None:
                continue
            ctx = _RecoveryContext(app, record.id)
            recover(ctx)
            boot_hooks = [*registry.boot_hooks, *ctx._boot_hooks]
            workers = [*registry.workers, *ctx._workers]
            hooks = {name: list(values) for name, values in registry.hooks.items()}
            for fn in ctx._disable_blockers:
                hooks.setdefault('plugin.disable_blockers', []).append((record.id, fn))
            registry.boot_hooks, registry.workers, registry.hooks = boot_hooks, workers, hooks
            record.recovery_active = True
            log.warning('plugin %r remains disabled; registered recovery workers only', record.id)
        except Exception as exc:  # noqa: BLE001 — expose a failed safety fallback without enabling features
            log.exception('plugin %r recovery registration failed', record.id)
            record.error = _paste_safe(f'recovery failed: {type(exc).__name__}: {exc}')


def load_plugins(app, csrf) -> PluginRegistry:
    existing = registry_of(app)
    if existing is not None:
        return existing  # The compatibility entry point must never register twice.
    registry = PluginRegistry()
    app.extensions['lds_plugins'] = registry
    app.config['EXTENSIONS_MANIFEST'] = []
    set_active(registry)   # reachable outside a request: installer threads, caption passes
    registry.dirs = {'bundled': str(bundled_dir()), 'external': str(external_dir()),
                     'legacy': str(legacy_dir())}
    if os.environ.get(KILL_SWITCH) == '0':
        return registry

    from .boot_transaction import boot_transaction
    with boot_transaction(app, external_dir()) as boot:
        registry.lifecycle_errors = storage.apply_pending(external_dir(), app=app)
        _register_discovered(app, csrf, registry)
        if boot is not None:
            boot.confirm_health(registry)
    _purge_committed_changes()
    return registry


def _purge_committed_changes():
    """Once a plugin change is committed, its recovery copies are dead weight."""
    from .transactions import purge_committed
    try:
        removed = purge_committed(external_dir())
    except Exception:  # Housekeeping never blocks startup; the next start retries.
        log.warning('Could not remove the recovery copies of committed plugin changes.', exc_info=True)
        return
    if removed:
        log.info('Removed the recovery copies of %d committed plugin change(s).', removed)


def _register_discovered(app, csrf, registry):
    from .discovery import development_bundles
    if development_bundles():
        _discover(registry, bundled_dir(), bundled=True)
    _discover(registry, external_dir(), bundled=False)
    discover_legacy(registry)

    failed_changes = {item['id']: item['error'] for item in registry.lifecycle_errors}
    for record in registry.records.values():
        if record.bundled or record.legacy is not None:
            continue
        try:
            if record.id in failed_changes:
                raise storage.StorageError(failed_changes[record.id])
            storage.migrate_data(external_dir(), record.id)
        except (OSError, ValueError) as exc:
            record.state, record.error = 'error', _paste_safe(str(exc))
            if record.id not in failed_changes:
                registry.lifecycle_errors.append({'id': record.id, 'error': record.error})

    # Ownership: every manifest's `owns` table, disjoint across all plugins.
    for record in registry.records.values():
        conflict = registry.claim_manifest(record.manifest)
        if conflict:
            record.state, record.error = 'error', conflict
    # API major.
    from .compatibility import host_issues
    dependency_versions = {pid: record.manifest.version for pid, record in registry.records.items()}
    for record in registry.records.values():
        if record.state == 'error':
            continue
        if record.manifest.api != LDS_PLUGIN_API_MAJOR:
            record.state = 'incompatible'
            record.error = (f'written for plugin API {record.manifest.api}, '
                            f'this app provides {LDS_PLUGIN_API_MAJOR}')
        else:
            issues = host_issues(record.manifest, dependencies=dependency_versions)
            if issues:
                record.state = 'incompatible'
                record.error = '; '.join(issue['message'] for issue in issues)
    # Enablement (a missing key means enabled; None means "no entry").
    enabled_map = _enabled_map()
    registry.stale = sorted(k for k, v in enabled_map.items() if v is not None and k not in registry.records)
    for pid, record in registry.records.items():
        flag = enabled_map.get(pid)
        record.enabled = True if flag is None else bool(flag)
        if record.state in ('error', 'incompatible'):
            continue
        record.state = 'pending' if record.enabled else 'disabled'
    # Dependency closure: a requirement that is missing, disabled, broken or
    # incompatible disables its dependents, and says which one.
    changed = True
    while changed:
        changed = False
        for pid, record in registry.records.items():
            if record.state != 'pending':
                continue
            blockers = [req for req in record.manifest.requires
                        if req not in registry.records or registry.records[req].state != 'pending']
            if blockers:
                record.state, record.disabled_by, changed = 'disabled', blockers, True
    order, cyclic = _order(registry.records)
    for pid in cyclic:
        registry.records[pid].state = 'error'
        registry.records[pid].error = 'requirement cycle: ' + ' <-> '.join(cyclic)

    # Registration, in dependency order.
    failed_at_load = set()
    for pid in order:
        record = registry.records[pid]
        manifest = record.manifest
        # A requirement may have failed during register(), after the initial
        # dependency closure was computed. Never run its dependents' code.
        blockers = [req for req in manifest.requires if registry.records[req].state != 'loaded']
        if blockers:
            record.state, record.disabled_by = 'disabled', blockers
            continue
        try:
            if record.legacy is not None:
                register_legacy(app, csrf, registry, record)  # Owns its registration transaction.
            elif manifest.python_package:
                from .registration import registration_transaction
                with registration_transaction(app, csrf, registry):
                    if record.dir not in sys.path:
                        sys.path.append(record.dir)
                    module = importlib.import_module(manifest.python_package)
                    register = getattr(module, 'register', None)
                    if register is None:
                        raise AttributeError(f'{manifest.python_package} defines no register(ctx)')
                    ctx = PluginContext(app, registry, manifest, plugin_data_dir(record))
                    register(ctx)
            record.state = 'loaded'
            log.info('plugin loaded: %s %s (%s)', pid, manifest.version,
                     'bundled' if record.bundled else 'external')
        except Exception as exc:  # noqa: BLE001 — a plugin must never take the app down
            log.exception('plugin %r failed to load; continuing without it', pid)
            record.state = 'error'
            record.error = _paste_safe(f'{type(exc).__name__}: {exc}')
            failed_at_load.add(pid)
    _register_recovery(app, registry, failed_at_load)
    app.config['EXTENSIONS_MANIFEST'] = legacy_manifest(registry)
    # Enablement was read before plugins registered their engines. Re-read the
    # saved preferences against the completed catalog so a newly discovered
    # engine joins an older selection on this first boot, without a Settings
    # save. load_config is read-only and preserves explicitly unchecked engines.
    cfg.load_config(force=True)
    return registry


def registry_of(app) -> PluginRegistry | None:
    return app.extensions.get('lds_plugins')


def run_boot_hooks(app) -> None:
    """Boot hooks and workers of the loaded plugins — called by
    ``_start_workers``, i.e. never under TESTING, like the core's own."""
    registry = registry_of(app)
    if registry is None:
        return
    for pid, fn in registry.boot_hooks:
        try:
            fn(app)
        except Exception:  # noqa: BLE001 — one plugin's hook must not stop the others
            log.exception('plugin %r boot hook failed', pid)
    for pid, name, fn in registry.workers:
        threading.Thread(target=fn, args=(app,), daemon=True, name=f'plugin-{pid}-{name}').start()
