"""``lds.api`` v1 — the only thing a plugin is promised.

A plugin's ``register(ctx)`` receives a :class:`PluginContext`. Its public
names are listed in :data:`PUBLIC_NAMES` and pinned by a contract test:
adding one is a minor change, removing or renaming one bumps
:data:`LDS_PLUGIN_API_MAJOR`. Everything else in ``app/`` is reachable but
unpromised.
"""
from __future__ import annotations
from ..timeout_settings import processing_timeout

import logging
import subprocess
from contextlib import ExitStack, contextmanager
from pathlib import Path

from sqlalchemy import text

from .. import config as cfg
from ..extensions import db
from .registry import OwnershipConflict, PluginRegistry

LDS_PLUGIN_API_MAJOR = 1
LDS_PLUGIN_API_MINOR = 23  # Local dataset engines through the shared image queue.


PUBLIC_NAMES = (
    'id', 'dir', 'data_dir', 'app', 'db', 'config', 'log',
    'register_blueprint', 'register_config_defaults', 'register_probe',
    'register_install_action', 'register_boot_hook', 'register_worker',
    'register_job_handler', 'ensure_columns', 'run_in_plugin_env', 'secret', 'netfetch',
    'register_model_download', 'register_install_group', 'register_hook', 'register_model_slot',
    'register_engine', 'register_restore_engine', 'register_request_limit',
    'register_data_migration', 'register_health_check',
    'plugin_environment', 'use_plugin_env', 'use_plugin_worker',
    'register_node_pack',
)


class PluginPermissionError(PermissionError):
    """A plugin asked for something its manifest does not declare."""


class ConfigView:
    """Read host configuration and update only this plugin's own namespace.

    ``get`` takes the core's dotted keys (``'server.port'``). A plugin's OWN
    settings live under ``plugins.<id>`` and an external id carries a dot
    (``example.hello``), which a dotted path cannot express — ``own`` reads
    them with the id taken whole: ``ctx.config.own('greeting')``.
    """

    def __init__(self, plugin_id: str):
        self._plugin_id = plugin_id

    def get(self, key, default=None):
        value = cfg.get(key)
        return default if value is None else value

    def own(self, key: str, default=None):
        section = cfg.get('plugins')
        node = section.get(self._plugin_id) if isinstance(section, dict) else None
        for part in key.split('.'):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return default if node is None else node

    def update_own(self, mapping: dict) -> dict:
        """Merge values only under ``plugins[plugin_id]`` (API 1.10).

        The plugin id is taken whole, including dots. Mapping keys never
        become host paths; even ``plugins`` or ``enabled`` stay inside this
        plugin's namespace. Returns the resulting own settings only.
        """
        if not isinstance(mapping, dict):
            raise TypeError('update_own expects a dict')
        updated = cfg.save_config({'plugins': {self._plugin_id: mapping}})
        return updated['plugins'][self._plugin_id]


class PluginContext:
    def __init__(self, app, registry: PluginRegistry, manifest, data_dir: Path):
        self.id = manifest.id
        self.dir = Path(manifest.dir)
        self.data_dir = Path(data_dir)
        self.app = app
        self.db = db
        self.config = ConfigView(manifest.id)
        self.log = logging.getLogger(f'plugin.{manifest.id}')
        self._registry = registry
        self._manifest = manifest

    # --- routes --------------------------------------------------------------
    def register_blueprint(self, blueprint, *, url_prefix: str | None = None) -> None:
        from flask import jsonify, request
        from functools import wraps

        prefix = url_prefix or f'/api/plugins/{self.id}'
        # `/api` itself is allowed: a bundled plugin extracted from the core
        # keeps the URLs its screens already call (`/api/scrape/scan`).
        if not (prefix == '/api' or prefix.startswith('/api/')):
            raise ValueError(f'plugin {self.id}: a blueprint must live under /api/ (got {prefix!r}) — '
                             'that is where the access-token gate stands')
        previous = set(self.app.view_functions)
        endpoints = frozenset()

        def available():
            record = self._registry.records.get(self.id)
            return (record is not None and record.enabled and record.state == 'loaded'
                    and (cfg.get('plugins.enabled') or {}).get(self.id) is not False)

        def admit_plugin_request():
            if request.endpoint not in endpoints:
                return None
            if not available():
                return jsonify(ok=False, code='plugin_unavailable',
                               error='This plugin is unavailable. Enable it in Plugins and restart LDS.'), 409
            from .membership import refusal
            access = self._registry.membership.check(self._registry.records.get(self.id))
            if not access['allowed']:
                return refusal(access)
            return None

        # Install before deferred before_app_request callbacks can short-circuit
        # an owned request. The endpoint set is complete before serving starts.
        self.app.before_request(admit_plugin_request)
        fields = ('before_request_funcs', 'url_value_preprocessors')
        lengths = {field: {scope: len(callbacks) for scope, callbacks in getattr(self.app, field).items()}
                   for field in fields}
        self.app.register_blueprint(blueprint, url_prefix=prefix)
        endpoints = frozenset(set(self.app.view_functions) - previous)

        def while_available(callback):
            @wraps(callback)
            def run(*args, **kwargs):
                if available() and self._registry.membership.check(
                        self._registry.records.get(self.id), refresh=request.endpoint in endpoints)['allowed']:
                    return self.app.ensure_sync(callback)(*args, **kwargs)
                return None
            return run

        # URL preprocessors execute even before app.before_request; global
        # blueprint callbacks also run on core routes. Disable only callbacks
        # added by this registration, on this app, without mutating the shared
        # Blueprint. Registration rollback restores all these lists on failure.
        for field in fields:
            for scope, callbacks in getattr(self.app, field).items():
                start = lengths[field].get(scope, 0)
                callbacks[start:] = [while_available(callback) for callback in callbacks[start:]]

    # --- config ----------------------------------------------------------------
    def register_config_defaults(self, mapping: dict) -> None:
        """Defaults for the plugin's own settings, under ``plugins.<id>``.
        Keys a plugin owns inside a shared core section stay in the core
        DEFAULTS and are declared in the manifest, never registered here."""
        if not isinstance(mapping, dict):
            raise TypeError('register_config_defaults expects a dict')
        cfg.register_plugin_defaults(self.id, mapping)

    # --- capabilities / setup --------------------------------------------------
    def register_probe(self, key: str, fn) -> None:
        self._must_own('probes', key)
        self._registry.add_probe(self.id, key, fn)

    def register_install_action(self, key: str, *, label: str, packages=None, requirements=None,
                                models=None, run=None, python: str = 'plugin', verify=None) -> None:
        """Register a pip or custom install and an optional no-argument readiness check.

        ``python='capability'`` preserves a scoped host capability's configured
        interpreter, constraints and post-install checks. The action must own
        that capability's name. ``verify()`` runs after a successful install;
        false means installation remains incomplete, and Setup reports failure.
        """
        self._must_own('install_actions', key)
        if self._registry.install_actions.get(key, {}).get('node_pack') is not None:
            raise ValueError('A registered node-pack recipe cannot be replaced by another install action.')
        from .environment import ACTION_PREFIX
        if key.startswith(ACTION_PREFIX):
            raise ValueError('The plugin_environment: action namespace belongs to Setup.')
        if python not in ('app', 'plugin', 'capability'):
            raise ValueError('An install action targets the app, plugin or configured capability interpreter.')
        if verify is not None and not callable(verify):
            raise TypeError('An install verification must be callable.')
        if python == 'capability':
            from ..setup_installer import _MANAGED_CAPABILITY_ACTIONS
            if key not in _MANAGED_CAPABILITY_ACTIONS or callable(run):
                raise ValueError('A capability install must name a scoped host capability and use its pip recipe.')
        if python in ('app', 'capability') and not ((self._manifest.bundled or self._manifest.official)
                                    and self._manifest.in_process_requirements):
            raise ValueError('Only a bundled plugin declaring in_process_requirements may install into the app interpreter.')
        self._registry.add_install_action(self.id, key, {
            'label': label, 'packages': list(packages or []), 'requirements': requirements,
            'models': list(models or []), 'run': run, 'python': python, 'verify': verify,
            **({'capability': key} if python == 'capability' else {}),
        })


    def register_node_pack(self, action: str) -> None:
        """Prepare exactly the pinned node_packs installation declared in the manifest (API 1.20)."""
        self._must_own('install_actions', action)
        from .node_packs import register
        register(self, action)


    def register_model_download(self, key: str, *, url: str, dest, min_free_gb, min_bytes,
                                license_url: str | None = None, gated: bool = False,
                                legacy_names=(), superseded_bytes=(), expected_bytes: int | None = None,
                                sha256: str | None = None, companions=(), complete=None, extra_roots=None) -> None:
        """A weight file Setup can fetch into ComfyUI's models folder, under the
        core's own downloader (streaming, disk precondition, integrity, the
        401/403 recovery). ``key`` is a Setup action name (``owns.install_actions``);
        ``dest`` is the ComfyUI-relative path as parts, e.g.
        ``('loras', 'qwen', 'file.safetensors')``. Since API 1.8, ``companions``
        names same-host files that complete a model directory; ``complete`` is
        an optional product-owned check for a complete stage under extra roots.
        ``extra_roots`` returns product-derived model roots for read-only presence
        checks; the downloader still writes only to the configured base folder.
        ``superseded_bytes`` identifies obsolete file sizes that Setup must
        replace with a verified download of the current model revision.
        """
        self._must_own('install_actions', key)
        if self._registry.install_actions.get(key, {}).get('node_pack') is not None:
            raise ValueError('A registered node-pack recipe cannot be replaced by a model download.')
        from urllib.parse import urlsplit
        if complete is not None and not callable(complete):
            raise ValueError('Model completeness must be a callable.')
        if extra_roots is not None and not callable(extra_roots):
            raise ValueError('Model extra roots must be a callable.')
        checked = []
        for companion in companions:
            if not isinstance(companion, dict) or set(companion) - {
                    'url', 'dest', 'kind', 'min_bytes', 'expected_bytes', 'sha256', 'license_url'}:
                raise ValueError('Invalid model companion fields.')
            target = companion.get('dest')
            source = urlsplit(companion.get('url', ''))
            if (not isinstance(target, (list, tuple)) or not target
                    or any(not isinstance(p, str) or not p or p in ('.', '..')
                           or any(c in p for c in '/\\:') for p in target)
                    or source.scheme != 'https' or source.netloc != urlsplit(url).netloc
                    or source.username or source.password
                    or companion.get('kind') not in (None, 'json')):
                raise ValueError('Model companion must stay in the declared host and model folders.')
            checked.append({**companion, 'dest': tuple(target)})
        self._registry.add_model_download(self.id, key, {
            'url': url, 'dest': tuple(dest), 'min_free_gb': min_free_gb, 'min_bytes': min_bytes,
            'gated': bool(gated), 'license_url': license_url,
            **({'legacy_names': tuple(legacy_names)} if legacy_names else {}),
            **({'superseded_bytes': tuple(int(size) for size in superseded_bytes)} if superseded_bytes else {}),
            **({'expected_bytes': int(expected_bytes)} if expected_bytes else {}),
            **({'sha256': sha256} if sha256 else {}),
            **({'companions': tuple(checked)} if checked else {}),
            **({'complete': complete} if complete is not None else {}),
            **({'extra_roots': extra_roots} if extra_roots is not None else {}),
        })

    def register_install_group(self, key: str, members, *, missing_key: str,
                               invalid_key: str | None = None, pack_action: str | None = None,
                               nodes_missing_key: str | None = None,
                               nodes_installed_key: str | None = None) -> None:
        """A one-click install of several actions (``owns.install_groups``). The
        ``*_key`` names are the ``comfyui.*`` capability keys the plan reads its
        gaps from — the plugin's own probes, usually."""
        self._must_own('install_groups', key)
        self._registry.add_install_group(self.id, key, members, {
            'missing': missing_key, 'invalid': invalid_key, 'pack_action': pack_action,
            'nodes_missing': nodes_missing_key, 'nodes_installed': nodes_installed_key,
        })

    def register_hook(self, name: str, fn) -> None:
        """Attach ``fn`` to a hook point the core names (``plugins/hooks.py``
        lists them). Not owned: several plugins may filter the same value."""
        from .hooks import HOOK_POINTS
        if name not in HOOK_POINTS:
            raise ValueError(f'plugin {self.id}: {name!r} is not a hook point the core exposes '
                             f'({", ".join(HOOK_POINTS)})')
        self._registry.add_hook(self.id, name, fn)

    def register_model_slot(self, slot: str, *, folder_type: str, hint: str, candidates=None) -> None:
        """A slot of the ComfyUI model picker (``owns.model_slots``): which models
        folder it lists and the words that say where to put a file; ``candidates``
        may replace the folder scan with the plugin's own resolver walk."""
        self._must_own('model_slots', slot)
        self._registry.add_model_slot(self.id, slot, {'folder_type': folder_type, 'hint': hint,
                                                       'candidates': candidates})

    def register_engine(self, **fields) -> None:
        """An image engine (``owns.engines``), into the core's engine registry
        (``app/engines``): the workspace cards, the batch builder, the Setup
        rows, the tracked capability keys and the key-test button all derive
        from it. ``fields`` are :class:`EngineSpec`'s — id, label, kind, order,
        rate, secret, key_test_target, file_tag, generate, probe… A plugin may
        re-register its own engine (a reload); it may not take another's."""
        from ..engines import registry as engine_registry
        engine_id = fields.get('id')
        if not engine_id:
            raise ValueError(f'plugin {self.id}: register_engine needs an id')
        self._must_own('engines', engine_id)
        existing = engine_registry.get(engine_id)
        if existing is not None and existing.plugin != self.id:
            raise ValueError(f'plugin {self.id}: engine {engine_id!r} is registered by '
                             f'{existing.plugin or "the core"!r}')
        engine_registry.register(engine_registry.EngineSpec(plugin=self.id, **fields), replace=True)

    def register_restore_engine(self, engine_id, *, preflight, enqueue, error_response,
                                profile=None, instruction=None, preset_rows=None, finishing=None):
        """API 1.9: a restoration provider, owned through ``owns.engines``.

        ``preflight()`` runs before any candidate is reserved. ``enqueue`` accepts
        source_filename, source_path, user_id and extra_metadata; it preserves
        the host candidate, queue and historical completion contract. Errors
        are mapped by ``error_response(error) -> (body, status) | None``.
        This does not register a prompt-generation engine.
        """
        self._must_own('engines', engine_id)
        if not all(callable(fn) for fn in (preflight, enqueue, error_response)):
            raise TypeError('A restoration needs preflight, enqueue and error callbacks.')
        if any(fn is not None and not callable(fn) for fn in (profile, instruction, preset_rows, finishing)):
            raise TypeError('Optional restoration recipe callbacks must be callable.')
        self._registry.claim('engines', engine_id, self.id)
        self._registry.restore_engines[engine_id] = {
            'plugin': self.id, 'preflight': preflight, 'enqueue': enqueue,
            'error_response': error_response, 'profile': profile,
            'instruction': instruction, 'preset_rows': preset_rows, 'finishing': finishing,
        }

    # --- lifecycle -------------------------------------------------------------------
    def register_boot_hook(self, fn) -> None:
        self._registry.add_boot_hook(self.id, fn)

    def register_worker(self, name: str, fn) -> None:
        self._registry.add_worker(self.id, name, fn)

    def register_job_handler(self, kind: str, fn, *, presentation=None) -> None:
        """Route a finished ComfyUI job to ``fn`` when its queue metadata flags
        ``kind`` (the flag your lane writes on the rows it enqueues — the
        Studio's ``is_lora_test`` is the core's own example). Consulted after
        the core Studio/reference edit and before Bank/dataset branches;
        ``fn(job_id, filename, failed=, reason=, metadata=)``
        owns its rows, failed marks included. ``kind`` must be in
        ``owns.job_kinds``: that declaration is what holds the boot sweep of
        ComfyUI's input folder while this plugin is not loaded.

        API 1.16: optional static ``presentation`` names ``title``, ``surface``,
        optional ``engine``, and ``cancel_scope`` (``job`` or ``owner``).
        ``owner`` requires ``stop_label`` and refuses cancellation from the
        global queue, without changing the owner's Stop or host recovery.
        No callback is accepted; see sdk/python/API-1.16.md for the contract.
        """
        self._must_own('job_kinds', kind)
        try:
            self._registry.add_job_handler(self.id, kind, fn, presentation=presentation)
        except OwnershipConflict as exc:
            raise ValueError(str(exc)) from exc

    def register_request_limit(self, endpoint: str, max_bytes: int) -> None:
        """A raised request-body ceiling for ONE endpoint of yours
        (``'<blueprint>.<view>'``), read by the app's request class before
        anything touches the body — the way the core's own archive uploads get
        theirs. The ordinary ``MAX_CONTENT_LENGTH`` governs everything else."""
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
            raise ValueError(f'plugin {self.id}: the request limit of {endpoint!r} must be a positive int')
        try:
            self._registry.add_request_limit(self.id, endpoint, max_bytes)
        except OwnershipConflict as exc:
            raise ValueError(str(exc)) from exc

    # --- storage ---------------------------------------------------------------------
    def register_data_migration(self, from_schema: int, to_schema: int, fn) -> None:
        """Register one ascending schema step, executed only in a trial boot.

        ``fn(ctx)`` may update owned SQL tables and ``ctx.data_dir``. Its normal
        session commits remain savepoints until every member passes health.
        Network requests, workers and writes outside these stores are forbidden
        migration side effects, not things a local Python sandbox can undo.
        """
        self._require_transaction_api()
        if (type(from_schema) is not int or type(to_schema) is not int or from_schema < 1
                or to_schema != from_schema + 1 or to_schema > self._manifest.contract.get('data_schema', 1)
                or not callable(fn)):
            raise ValueError('A data migration must be a callable advancing exactly one declared schema version.')
        steps = self.app.extensions.setdefault('lds_plugin_migrations', {}).setdefault(self.id, {})
        if from_schema in steps:
            raise ValueError('A data migration step was registered twice.')
        steps[from_schema] = lambda: fn(self)

    def register_health_check(self, name: str, fn) -> None:
        """A local trial-boot check: ``fn(ctx)`` must return exactly True."""
        self._require_transaction_api()
        if not isinstance(name, str) or not name.strip() or len(name) > 100 or not callable(fn):
            raise ValueError('A plugin health check needs a short name and a callable.')
        checks = self.app.extensions.setdefault('lds_plugin_health_checks', {}).setdefault(self.id, {})
        if name in checks:
            raise ValueError('A plugin health check name was registered twice.')
        checks[name] = lambda: fn(self)

    def _require_transaction_api(self):
        from packaging.specifiers import SpecifierSet
        contract = self._manifest.contract
        api_range = contract.get('compatibility', {}).get('api', '')
        if contract.get('schema_version', 1) < 2 or not api_range or '1.5' in SpecifierSet(api_range):
            raise ValueError('Trial health checks and data migrations require manifest v2 with plugin API >=1.6.')

    def ensure_columns(self, table: str, columns: dict) -> None:
        """Additive migration for a plugin-owned table: add the columns that are
        missing (``{name: 'SQL TYPE'}``). SQLite only, like the core's."""
        self._must_own('tables', table)
        existing = {row[1] for row in db.session.execute(text(f'PRAGMA table_info({table})'))}
        for name, col_type in columns.items():
            if name not in existing:
                db.session.execute(text(f'ALTER TABLE {table} ADD COLUMN {name} {col_type}'))
        db.session.commit()

    # --- isolation -------------------------------------------------------------------
    def _environment_record(self):
        from .environment import EnvironmentError
        record = self._registry.records.get(self.id)
        if record is None:
            raise EnvironmentError('This plugin is no longer installed.')
        if record.manifest is not self._manifest:
            raise EnvironmentError('Restart LDS before using the replaced plugin environment.')
        return record

    def plugin_environment(self):
        """Read this plugin's runnable environment state, without installing it (API 1.14).

        ``python`` is null unless the installed environment is ready and the
        plugin can run now. Dependency imports belong to the product's worker
        probe, never to the host process or this metadata query.
        """
        from . import environment
        from .lifecycle import state_change_lock
        result = {'ready': False, 'can_install': False, 'action': None,
                  'reason': '', 'python': None}
        with state_change_lock:
            try:
                host_installer = environment.installer()
                record = self._environment_record()
                state = environment.summary(record, self._registry)
                if state is None:
                    result['reason'] = 'This plugin does not declare a managed Python environment.'
                    return result
                result.update(state)
                environment.check_enabled(record, loaded=True)
                if host_installer.plugin_install_busy(self.id):
                    raise environment.EnvironmentError('The plugin environment is being installed. Wait for Setup to finish.')
                if result['ready']:
                    result['python'] = environment.interpreter(record)
                if environment.running(self.id):
                    result['can_install'] = False
            except environment.EnvironmentError as exc:
                result.update(ready=False, can_install=False, python=None, reason=str(exc))
        return result

    @contextmanager
    def use_plugin_worker(self):
        """Reserve this product's worker lifetime without requiring a venv (API 1.15).

        External interpreters still need current plugin admission and protection
        from disable/remove/replace/restart until the process has been reaped.
        This scope neither selects an interpreter nor manages its packages.
        """
        from . import environment
        from .lifecycle import state_change_lock
        with ExitStack() as stack:
            with state_change_lock:
                stack.enter_context(environment.worker(self._environment_record()))
            yield

    @contextmanager
    def use_plugin_env(self):
        """Lease this plugin's interpreter for the complete worker lifetime (API 1.14).

        Keep Popen, output reading, wait and any kill/reap cleanup inside the
        ``with``. Exiting releases the lease even when the worker raises.
        """
        from . import environment
        from .lifecycle import state_change_lock
        with ExitStack() as stack:
            with state_change_lock:
                python = stack.enter_context(environment.use(self._environment_record()))
            yield python

    def run_in_plugin_env(self, script: str, args=(), *, timeout: int = 600):
        """Run ``script`` (a path inside the plugin directory) with the plugin's
        own versioned interpreter, built by Setup ▸ Plugins —
        as a subprocess. Heavy dependencies live there, never in the app's
        process."""
        from . import environment
        record = self._registry.records.get(self.id)
        if record is None:
            raise RuntimeError('This plugin is no longer installed.')
        script_path = (self.dir / script).resolve()
        if self.dir.resolve() not in script_path.parents:
            raise ValueError(f'plugin {self.id}: {script!r} is not inside the plugin directory')
        with environment.use(record) as python:
            return subprocess.run([python, '-s', str(script_path), *map(str, args)], capture_output=True,
                                  text=True, timeout=processing_timeout(timeout), env=environment.subprocess_env())

    # --- secrets / fetch ---------------------------------------------------------------
    def secret(self, name: str):
        if f'secrets:{name}' not in self._manifest.permissions:
            raise PluginPermissionError(f'plugin {self.id} does not declare permission secrets:{name}')
        return cfg.secret(name)

    @property
    def netfetch(self):
        from ..scrape import netfetch
        return netfetch

    # --- internals -----------------------------------------------------------------------
    def _must_own(self, kind: str, name: str) -> None:
        owner = self._registry.owner_of(kind, name)
        if owner != self.id:
            raise ValueError(f'plugin {self.id}: {kind} {name!r} is not in its manifest "owns" table'
                             + (f' (owned by {owner!r})' if owner else ''))
