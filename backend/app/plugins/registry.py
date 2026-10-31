"""What the loader knows about every plugin, and what plugins registered.

One :class:`PluginRegistry` per Flask app (``app.extensions['lds_plugins']``).
It holds the discovered records (bundled, external, misplaced, broken), the
ownership table built from every manifest's ``owns`` (two plugins claiming the
same install action, probe, config section, help topic, table or What's-new id
is an error for the later one), and the runtime registries the façade fills:
capability probes, install actions, model downloads, install groups, hook
points, picker slots, boot hooks, workers, job handlers.

The registry of the app that booted last is also reachable without a
request (:func:`active`): the installer's worker threads and the caption
passes run outside any Flask context and still need what plugins
registered.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from .manifest import PluginManifest
from .ownership import claims

log = logging.getLogger(__name__)

STATES = ('loaded', 'disabled', 'error', 'incompatible', 'misplaced')

_ACTIVE: 'PluginRegistry | None' = None


def set_active(registry: 'PluginRegistry | None') -> None:
    global _ACTIVE
    _ACTIVE = registry


def active() -> 'PluginRegistry | None':
    """The registry of the app that loaded plugins last, or None before any
    did (a bare test, a tool script) — callers treat None as 'no plugins'."""
    return _ACTIVE


class OwnershipConflict(ValueError):
    pass


def _job_presentation(value):
    """Validate static display data before either registration table changes."""
    if value is None:
        return None
    allowed = {'title', 'surface', 'engine', 'cancel_scope', 'stop_label'}
    if not isinstance(value, dict) or value.keys() - allowed:
        raise ValueError('Job presentation must be a dict of title, surface, engine, cancel_scope and stop_label.')
    spec = dict(value)
    for key in ('title', 'surface', 'engine', 'stop_label'):
        text = spec.get(key)
        if key in ('engine', 'stop_label') and text is None and key not in spec:
            continue
        if key == 'engine' and text is None:
            continue
        if (not isinstance(text, str) or not text.strip() or len(text) > 120
                or any(not ch.isprintable() for ch in text)):
            raise ValueError(f'Job presentation {key} must be a nonempty single-line string of at most 120 characters.')
    scope = spec.setdefault('cancel_scope', 'job')
    if scope not in ('job', 'owner'):
        raise ValueError('Job presentation cancel_scope must be job or owner.')
    if scope == 'owner' and not spec.get('stop_label'):
        raise ValueError('Owner-only job cancellation requires a stop_label.')
    return spec


@dataclass
class PluginRecord:
    id: str
    manifest: PluginManifest | None
    dir: str
    bundled: bool
    enabled: bool = True
    state: str = 'disabled'
    disabled_by: list = field(default_factory=list)
    error: str | None = None
    frontend_url: str | None = None
    styles: list = field(default_factory=list)
    legacy: dict | None = None
    pending_restart: bool = False
    recovery_active: bool = False

    def payload(self) -> dict:
        m = self.manifest
        out = {
            'id': self.id, 'bundled': self.bundled, 'enabled': self.enabled,
            'state': self.state, 'disabled_by': list(self.disabled_by),
            'error': self.error, 'frontend': self.frontend_url,
            'name': m.name if m else self.id, 'version': m.version if m else None,
            'description': m.description if m else '', 'requires': list(m.requires) if m else [],
            'permissions': list(m.permissions) if m else [],
            'requirements': m.requirements if m else None,
            'models': [dict(x) for x in m.models] if m else [],
            'node_packs': [dict(x) for x in m.node_packs] if m else [],
            'license': m.license if m else '', 'author': m.author if m else '',
            'official': bool(m and m.official),
            'package_contract': m.contract if m else {},
            'guide_ownership': {'chapters': list(m.owned('guide_chapters')) if m else [],
                                'sections': list(m.owned('guide_sections')) if m else []},
            'schema_version': m.contract.get('schema_version', 1) if m else 1,
            'styles': list(self.styles),
        }
        if self.legacy is not None:
            out.update(legacy=True, deprecation=m.description)
        return out


class PluginRegistry:
    def __init__(self):
        from .membership import MemberAccess
        self.membership = MemberAccess()
        self.boot_id = uuid.uuid4().hex
        self.lifecycle_errors: list[dict] = []
        self.records: dict[str, PluginRecord] = {}
        self.misplaced: list[dict] = []      # directories under bundled/ that are not bundled plugins
        self.invalid: list[dict] = []        # directories whose manifest does not parse or is mis-shaped
        self.stale: list[str] = []           # plugins.enabled entries for ids no plugin carries
        self.ownership: dict[tuple, str] = {}  # (kind, name) -> plugin id
        self.probes: dict[str, tuple] = {}     # capability key -> (plugin id, fn)
        self.install_actions: dict[str, dict] = {}
        self.model_downloads: dict[str, dict] = {}   # Setup action -> {'plugin', url, dest, ...}
        self.install_groups: dict[str, dict] = {}    # group -> {'plugin', 'members', 'caps_keys'}
        self.restore_engines: dict[str, dict] = {}
        self.hooks: dict[str, list] = {}             # hook point -> [(plugin id, fn)]
        self.model_slots: dict[str, tuple] = {}      # picker slot -> (plugin id, spec)
        self.boot_hooks: list[tuple] = []
        self.workers: list[tuple] = []
        self.job_handlers: dict[str, tuple] = {}     # metadata flag -> (plugin id, fn)
        self.job_presentations: dict[str, dict] = {}  # same kind/owner as job_handlers; static data only
        self.request_limits: dict[str, tuple] = {}   # endpoint -> (plugin id, max bytes)
        self.dirs: dict = {}

    # --- ownership -----------------------------------------------------------
    def claim_manifest(self, manifest: PluginManifest) -> str | None:
        """Claim every name the manifest owns. Returns the conflict sentence
        when one of them already belongs to another plugin (nothing is claimed
        then), else None."""
        requested_claims = sorted(claims(manifest))
        for key in requested_claims:
            owner = self.ownership.get(key)
            if owner is not None and owner != manifest.id:
                return f'{key[0]} {key[1]!r} is already owned by plugin {owner!r}'
        for key in requested_claims:
            self.ownership[key] = manifest.id
        return None

    def owner_of(self, kind: str, name: str) -> str | None:
        return self.ownership.get((kind, name))

    def claim(self, kind: str, name: str, plugin_id: str) -> None:
        owner = self.ownership.get((kind, name))
        if owner is not None and owner != plugin_id:
            raise OwnershipConflict(f'{kind} {name!r} is owned by plugin {owner!r}')
        self.ownership[(kind, name)] = plugin_id

    # --- runtime registries -----------------------------------------------------
    def add_probe(self, plugin_id: str, key: str, fn) -> None:
        self.claim('probes', key, plugin_id)
        self.probes[key] = (plugin_id, fn)

    def add_install_action(self, plugin_id: str, key: str, spec: dict) -> None:
        self.claim('install_actions', key, plugin_id)
        self.install_actions[key] = {'plugin': plugin_id, **spec}

    def add_model_download(self, plugin_id: str, key: str, spec: dict) -> None:
        self.claim('install_actions', key, plugin_id)
        self.model_downloads[key] = {'plugin': plugin_id, **spec}

    def add_install_group(self, plugin_id: str, key: str, members, caps_keys: dict) -> None:
        self.claim('install_groups', key, plugin_id)
        self.install_groups[key] = {'plugin': plugin_id, 'members': tuple(members),
                                    'caps_keys': dict(caps_keys)}

    def add_hook(self, plugin_id: str, name: str, fn) -> None:
        self.hooks.setdefault(name, []).append((plugin_id, fn))

    def add_model_slot(self, plugin_id: str, slot: str, spec: dict) -> None:
        self.claim('model_slots', slot, plugin_id)
        self.model_slots[slot] = (plugin_id, dict(spec))

    def add_boot_hook(self, plugin_id: str, fn) -> None:
        self.boot_hooks.append((plugin_id, fn))

    def add_worker(self, plugin_id: str, name: str, fn) -> None:
        self.workers.append((plugin_id, name, fn))

    def add_job_handler(self, plugin_id: str, kind: str, fn, *, presentation=None) -> None:
        if kind in self.job_handlers and self.job_handlers[kind][0] != plugin_id:
            raise OwnershipConflict(f'job kind {kind!r} is handled by plugin {self.job_handlers[kind][0]!r}')
        spec = _job_presentation(presentation)
        self.job_handlers[kind] = (plugin_id, fn)
        if spec is None:
            self.job_presentations.pop(kind, None)
        else:
            self.job_presentations[kind] = spec

    def add_request_limit(self, plugin_id: str, endpoint: str, max_bytes: int) -> None:
        owner = self.request_limits.get(endpoint, (None, 0))[0]
        if owner is not None and owner != plugin_id:
            raise OwnershipConflict(f'the request limit of {endpoint!r} is set by plugin {owner!r}')
        self.request_limits[endpoint] = (plugin_id, int(max_bytes))

    # --- views -----------------------------------------------------------------------
    def loaded(self) -> list[PluginRecord]:
        return [r for r in self.records.values() if r.state == 'loaded']

    def payload(self) -> dict:
        return {
            'plugins': [r.payload() for r in self.records.values()],
            'misplaced': list(self.misplaced),
            'invalid': list(self.invalid),
            'stale': list(self.stale),
            'dirs': dict(self.dirs),
        }
