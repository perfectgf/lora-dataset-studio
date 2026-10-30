"""``plugin.json`` — the one file that makes a directory a plugin.

The manifest is JSON (not TOML: Python 3.10 has no ``tomllib`` and Pinokio
ships 3.10). Its full contract is ``docs/specs/2026-09-04-plugin-system-design.md``
§4; this module parses and validates it and nothing else — the loader decides
what to do with a valid manifest.

Two rules that came out of the design review and are enforced here:

* the plugin ``id`` prefixes exactly two things — its config key and its
  directory. Every install action, probe, config section, help topic, table
  or What's-new id a plugin carries is listed in its ``owns`` table, never
  derived from the id (``video`` is already an install action, ``video_text``
  is a CORE capability, and ``camera_angles`` owns ``camera_*``);
* an external plugin's id is namespaced ``publisher.name``, so it can never
  equal a bundled id — not at install, and not later through a stale entry in
  ``plugins.enabled``.
"""
from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_NAME = 'plugin.json'

BUNDLED_ID = re.compile(r'^[a-z][a-z0-9_]{1,31}$')
EXTERNAL_ID = re.compile(r'^[a-z][a-z0-9_]{1,31}\.[a-z][a-z0-9_]{1,31}$')
PACKAGE_NAME = re.compile(r'^[a-z_][a-z0-9_]{0,63}$')
VERSION = re.compile(r'^\d+(\.\d+){0,3}([-+][0-9A-Za-z.-]+)?$')
SHA256 = re.compile(r'^[0-9a-f]{64}$')
# Names that would shadow the app, its tests or the plugin trees on sys.path.
RESERVED_PACKAGES = frozenset({'app', 'backend', 'tests', 'frontend', 'bundled', 'plugins',
                               'scripts', 'packaging', 'docs', 'infer', 'workflows', 'lds_sdk'})
# `plugins.enabled` is the switch map; a plugin's own settings live beside it
# under `plugins.<id>`, so no plugin may be called that.
RESERVED_IDS = frozenset({'enabled'})
PERMISSIONS = frozenset({'network', 'datasets:read', 'datasets:write', 'banks:read',
                         'banks:write', 'comfyui', 'gpu', 'filesystem'})
SECRET_PERMISSION = re.compile(r'^secrets:[A-Z][A-Z0-9_]{1,63}$')
OWNS_LISTS = ('install_actions', 'install_groups', 'probes', 'config_sections',
              'help_topics', 'tables', 'whats_new_ids', 'model_slots', 'engines',
              # the queue-metadata flags a lane writes on the jobs it enqueues: a
              # plugin that stages inputs says so, and the boot sweep waits for it
              'job_kinds')
OWNS_KEYS = OWNS_LISTS + ('config_keys_in_shared_sections',)
GUIDE_OWNS = ('guide_chapters', 'guide_sections')


class ManifestError(ValueError):
    """The manifest is not usable. The message names the field and the rule."""


class ManifestNeedsNewerApp(ManifestError):
    """The manifest uses vocabulary added after this version of the app (a
    permission, an ownership kind): a newer LDS reads it, this one cannot."""


@dataclass(frozen=True)
class PluginManifest:
    id: str
    bundled: bool
    name: str
    version: str
    api: int
    dir: Path
    description: str = ''
    requires: tuple = ()
    python_package: str | None = None
    frontend: str | None = None
    requirements: str | None = None
    in_process_requirements: bool = False
    owns: dict = field(default_factory=dict)
    models: tuple = ()
    node_packs: tuple = ()
    permissions: tuple = ()
    license: str = ''
    author: str = ''
    official: bool = False
    contract: dict = field(default_factory=dict)
    guide_ownership: dict = field(default_factory=dict)

    @property
    def external(self) -> bool:
        return not self.bundled

    def owned(self, kind: str) -> tuple:
        """The names this plugin claims for one ``owns`` kind (lists only)."""
        if kind in GUIDE_OWNS:
            return tuple(self.guide_ownership.get(kind.removeprefix('guide_'), ()))
        return tuple(self.owns.get(kind, ()))

    def summary(self) -> dict:
        """What the consent screen and the plugins list show."""
        return {
            'id': self.id, 'bundled': self.bundled, 'name': self.name,
            'version': self.version, 'api': self.api, 'description': self.description,
            'requires': list(self.requires), 'frontend': self.frontend,
            'requirements': self.requirements,
            'in_process_requirements': self.in_process_requirements,
            'models': [dict(m) for m in self.models],
            'node_packs': deepcopy(list(self.node_packs)),
            'permissions': list(self.permissions), 'license': self.license,
            'author': self.author,
            'official': self.official,
            **self.contract,
            **({'guide_ownership': {key: list(value) for key, value in self.guide_ownership.items()}}
               if any(self.guide_ownership.values()) else {}),
        }


def _require(data: dict, key: str, kind, where: str):
    if key not in data:
        raise ManifestError(f'{where}: "{key}" is required')
    value = data[key]
    if not isinstance(value, kind) or (kind is str and not value.strip()):
        raise ManifestError(f'{where}: "{key}" must be a non-empty {kind.__name__}')
    return value


def _str_list(data: dict, key: str, where: str) -> tuple:
    value = data.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise ManifestError(f'{where}: "{key}" must be a list of non-empty strings')
    return tuple(value)


def _relative_file(value, key: str, where: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ManifestError(f'{where}: "{key}" must be a relative path')
    parts = value.replace('\\', '/').split('/')
    if value.startswith(('/', '\\')) or ':' in value or '..' in parts or '' in parts:
        raise ManifestError(f'{where}: "{key}" must stay inside the plugin directory')
    return '/'.join(parts)


def parse_manifest(data: dict, directory: Path, *, where: str = MANIFEST_NAME,
                   official: bool = False) -> PluginManifest:
    """Validate a decoded manifest. Raises :class:`ManifestError`."""
    if not isinstance(data, dict):
        raise ManifestError(f'{where}: the manifest must be a JSON object')
    plugin_id = _require(data, 'id', str, where)
    bundled = data.get('bundled', False)
    if type(bundled) is not bool:
        raise ManifestError(f'{where}: "bundled" must be a boolean')
    from .official import is_official_id
    official = bool(official and is_official_id(plugin_id))
    pattern = BUNDLED_ID if bundled or official else EXTERNAL_ID
    if plugin_id in RESERVED_IDS:
        raise ManifestError(f'{where}: "id" {plugin_id!r} is reserved')
    if not pattern.match(plugin_id):
        shape = 'letters, digits and _ (2-32 chars)' if bundled else '"publisher.name", each part letters, digits and _'
        raise ManifestError(f'{where}: "id" {plugin_id!r} must be {shape}')
    name = _require(data, 'name', str, where)
    version = _require(data, 'version', str, where)
    if len(version) > 128 or (data.get('schema_version', 1) == 1 and not VERSION.fullmatch(version)):
        raise ManifestError(f'{where}: "version" {version!r} is not a version number')
    api = data.get('api')
    if not isinstance(api, int) or isinstance(api, bool) or api < 1:
        raise ManifestError(f'{where}: "api" must be a positive integer (the plugin API major)')
    requires = _str_list(data, 'requires', where)
    if plugin_id in requires:
        raise ManifestError(f'{where}: a plugin cannot require itself')
    package = data.get('python_package')
    if package is not None:
        if not isinstance(package, str) or not PACKAGE_NAME.match(package):
            raise ManifestError(f'{where}: "python_package" must be a valid importable name')
        if package in RESERVED_PACKAGES:
            raise ManifestError(f'{where}: "python_package" {package!r} would shadow the app')
    frontend = _relative_file(data.get('frontend'), 'frontend', where)
    requirements = _relative_file(data.get('requirements'), 'requirements', where)
    in_process = data.get('in_process_requirements', False)
    if type(in_process) is not bool:
        raise ManifestError(f'{where}: "in_process_requirements" must be a boolean')
    if in_process and not (bundled or official):
        raise ManifestError(f'{where}: "in_process_requirements" is only accepted from a bundled plugin — '
                            'a third-party plugin installs into its own environment')
    owns_raw = data.get('owns', {})
    if not isinstance(owns_raw, dict):
        raise ManifestError(f'{where}: "owns" must be an object')
    for key in owns_raw:
        if key not in OWNS_KEYS:
            raise ManifestNeedsNewerApp(f'{where}: "owns.{key}" is not a known ownership kind')
    owns = {}
    for key in OWNS_LISTS:
        owns[key] = list(_str_list(owns_raw, key, f'{where} owns'))
    # Additive top-level metadata: a 1.6 Store can parse a feed containing 1.7
    # releases before its compatibility resolver selects an older release.
    guide = data.get('guide_ownership', {})
    if (not isinstance(guide, dict) or set(guide) - {'chapters', 'sections'}):
        raise ManifestError(f'{where}: "guide_ownership" must contain chapters and sections only')
    guide_ownership = {}
    for guide_kind in ('chapters', 'sections'):
        claims = list(_str_list(guide, guide_kind, f'{where} guide_ownership'))
        if len(claims) != len(set(claims)):
            raise ManifestError(f'{where}: "guide_ownership.{guide_kind}" contains duplicates')
        if guide_kind == 'chapters':
            valid = all(value.startswith(plugin_id + '.') and re.fullmatch(r'[a-z0-9][a-z0-9_.-]*', value) for value in claims)
        else:
            valid = all(re.fullmatch(r'[a-z0-9][a-z0-9_.-]*#[a-z0-9][a-z0-9_.-]*', value) for value in claims)
        if not valid or len(claims) > (12 if guide_kind == 'chapters' else 150):
            raise ManifestError(f'{where}: invalid "guide_ownership.{guide_kind}"')
        guide_ownership[guide_kind] = claims
    if any(guide_ownership.values()):
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
        from packaging.version import Version
        compatibility = data.get('compatibility', {})
        api_range = compatibility.get('api') if isinstance(compatibility, dict) else None
        try:
            specifiers = SpecifierSet(api_range) if isinstance(api_range, str) else ()
            explicit_floor = any(spec.operator in ('>=', '>', '~=', '==')
                                 and Version(spec.version.removesuffix('.*')) >= Version('1.7')
                                 for spec in specifiers)
        except (InvalidSpecifier, ValueError):
            explicit_floor = False
        if not explicit_floor:
            raise ManifestError(f'{where}: guide content requires an explicit compatibility.api floor >=1.7')
    shared = owns_raw.get('config_keys_in_shared_sections', {})
    if not isinstance(shared, dict) or not all(
            isinstance(k, str) and isinstance(v, list) and all(isinstance(x, str) and x for x in v)
            for k, v in shared.items()):
        raise ManifestError(f'{where}: "owns.config_keys_in_shared_sections" must map a section to a list of keys')
    owns['config_keys_in_shared_sections'] = {k: list(v) for k, v in shared.items()}
    models = []
    for i, m in enumerate(data.get('models', []) or []):
        if not isinstance(m, dict):
            raise ManifestError(f'{where}: models[{i}] must be an object')
        for k in ('key', 'url', 'sha256', 'dest'):
            _require(m, k, str, f'{where} models[{i}]')
        if not SHA256.match(m['sha256']):
            raise ManifestError(f'{where}: models[{i}].sha256 must be 64 hex characters — a weight without '
                                'a verifiable hash is not listed')
        models.append({k: m[k] for k in ('key', 'url', 'sha256', 'dest')}
                      | ({'size_bytes': int(m['size_bytes'])} if 'size_bytes' in m else {}))
    packs = []
    for i, p in enumerate(data.get('node_packs', []) or []):
        if not isinstance(p, dict):
            raise ManifestError(f'{where}: node_packs[{i}] must be an object')
        for k in ('pack', 'url'):
            _require(p, k, str, f'{where} node_packs[{i}]')
        packs.append({'pack': p['pack'], 'url': p['url'], 'search': p.get('search', ''),
                      'classes': deepcopy(p.get('classes', []) or []),
                      **({'installation': deepcopy(p['installation'])} if 'installation' in p else {})})
    permissions = _str_list(data, 'permissions', where)
    for perm in permissions:
        if perm not in PERMISSIONS and not SECRET_PERMISSION.match(perm):
            raise ManifestNeedsNewerApp(f'{where}: permission {perm!r} is not in the vocabulary')
    from .package_contract import PackageContractError, validate_contract
    try:
        contract = validate_contract(data)
    except PackageContractError as exc:
        raise ManifestError(f'{where}: {exc}') from exc
    from .node_packs import NodePackError, validate as validate_node_packs
    try:
        validate_node_packs(packs, plugin_id=plugin_id, owns=owns, permissions=permissions, contract=contract)
    except NodePackError as exc:
        raise ManifestError(f'{where}: {exc}') from exc
    if contract['schema_version'] >= 2:
        from packaging.version import InvalidVersion, Version
        try:
            Version(version)
        except InvalidVersion as exc:
            raise ManifestError(f'{where}: "version" must be a valid PEP 440 version') from exc
    if contract.get('frontend_styles'):
        if not frontend or '/' not in frontend:
            raise ManifestError(f'{where}: frontend styles require an entry in a UI folder')
        ui_folder = frontend.rsplit('/', 1)[0] + '/'
        if any(not path.startswith(ui_folder) for path in contract['frontend_styles']):
            raise ManifestError(f'{where}: frontend styles must be inside the UI entry folder')
    if official and (contract.get('publisher') or {}).get('id') != 'lds':
        raise ManifestError(f'{where}: an official package must identify the LDS publisher')
    return PluginManifest(
        id=plugin_id, bundled=bundled, name=name, version=version, api=api, dir=Path(directory),
        description=str(data.get('description', '') or ''), requires=requires,
        python_package=package, frontend=frontend, requirements=requirements,
        in_process_requirements=in_process, owns=owns, models=tuple(models),
        node_packs=tuple(packs), permissions=permissions,
        license=str(data.get('license', '') or ''), author=str(data.get('author', '') or ''),
        official=official, contract=contract, guide_ownership=guide_ownership,
    )


def load_manifest(directory: Path, *, official: bool = False) -> PluginManifest:
    """Read and validate ``<directory>/plugin.json``."""
    path = Path(directory) / MANIFEST_NAME
    try:
        raw = path.read_text(encoding='utf-8')
    except OSError as exc:
        raise ManifestError(f'{MANIFEST_NAME}: cannot read ({exc.__class__.__name__})') from exc
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ManifestError(f'{MANIFEST_NAME}: not valid JSON ({exc})') from exc
    return parse_manifest(data, Path(directory), official=official)
