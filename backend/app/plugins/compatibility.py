"""The same compatibility decision is used before installation and at boot."""
import platform
import struct
import sys
from importlib.metadata import PackageNotFoundError, version

from ..version import APP_VERSION
from .api import LDS_PLUGIN_API_MAJOR, LDS_PLUGIN_API_MINOR
from .package_contract import compatibility_issues
from .ownership import conflict

# Host libraries this core moved past the exact pin that existing packages
# recorded: the SDK writes host_dependencies from the backend/requirements.txt of
# the core a package was built against. Pillow 12.3.0 -> 13.0.0 and filelock
# 3.32.5 -> 4.2.10 close security advisories, and the official packages were
# checked on the new releases. A package recorded against the previous pin keeps
# loading when the core updates instead of turning incompatible; a package built
# against the new pin still refuses an older core through its own constraint.
# Canonical name -> {previous exact version: the release this core installs}.
HOST_DEPENDENCY_SUCCESSORS = {
    'pillow': {'12.3.0': '13.0.0'},
    'filelock': {'3.32.5': '4.2.10'},
}


def _recorded_host_version(name, specifier, installed):
    """The installed version as the package's recorded constraint sees it."""
    spec = str(specifier).strip()
    if spec.startswith('==') and not spec.startswith('==='):
        previous = spec[2:].strip()
        if HOST_DEPENDENCY_SUCCESSORS.get(name, {}).get(previous) == installed:
            return previous
    return installed


def host_issues(manifest, *, dependencies=None):
    installed = {}
    for name, specifier in manifest.contract.get('host_dependencies', {}).items():
        try:
            installed[name] = _recorded_host_version(name, specifier, version(name))
        except PackageNotFoundError:
            pass
    issues = compatibility_issues(
        manifest.contract, lds_version=APP_VERSION,
        api_version=f'{LDS_PLUGIN_API_MAJOR}.{LDS_PLUGIN_API_MINOR}',
        python_version='.'.join(map(str, sys.version_info[:3])),
        system=platform.system(), machine=platform.machine(), dependencies=dependencies, host_dependencies=installed,
        python_implementation=sys.implementation.name, python_cache_tag=sys.implementation.cache_tag,
        pointer_bits=struct.calcsize('P') * 8,
    )
    if manifest.api != LDS_PLUGIN_API_MAJOR:
        issues.insert(0, {'code': 'api_major_mismatch', 'field': 'api',
                         'message': f'This plugin requires plugin API major {manifest.api}; this app provides {LDS_PLUGIN_API_MAJOR}.',
                         'expected': manifest.api, 'actual': LDS_PLUGIN_API_MAJOR})
    return issues


class IncompatiblePackage(ValueError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__('This plugin cannot be installed in the current app configuration.')


def installation_issues(manifest, records, enabled):
    """Check one manual package against the requested next-boot composition.

    No product is imported and no dependency is silently enabled. Existing ON
    consumers still constrain an update; optional integrations do not enter this
    graph. The caller replays this check under transaction admission at staging.
    """
    available = {pid: record.manifest for pid, record in records.items()
                 if record.manifest and enabled.get(pid, record.enabled) is not False
                 and record.state not in {'error', 'misplaced'}}
    target_enabled = enabled.get(manifest.id, True) is not False
    if target_enabled:
        available[manifest.id] = manifest
    else:
        available.pop(manifest.id, None)
    versions = {pid: item.version for pid, item in available.items()}
    issues, visited = [], set()

    def visit(item, trail=()):
        if item.id in trail:
            issues.append({'code': 'dependency_cycle', 'field': 'requires',
                           'message': 'This package would create a required-plugin cycle.',
                           'expected': 'acyclic dependencies', 'actual': list(trail + (item.id,))})
            return
        if item.id in visited:
            return
        visited.add(item.id)
        for issue in host_issues(item, dependencies=versions):
            issues.append(issue if item.id == manifest.id else {
                **issue, 'message': f'{item.id}: {issue["message"]}'})
        for pid in item.requires:
            if pid not in available:
                if pid not in item.contract.get('dependency_versions', {}):
                    issues.append({'code': 'dependency_unavailable', 'field': f'requires.{pid}',
                                   'message': f'{item.id} requires {pid}. Install and enable that plugin first.',
                                   'expected': 'installed and enabled', 'actual': records[pid].state if pid in records else None})
            else:
                visit(available[pid], trail + (item.id,))

    if target_enabled:
        visit(manifest)
    else:
        # An OFF update never imports the product or silently enables its
        # dependencies. Host metadata remains checked for later activation.
        issues.extend(host_issues(manifest))
    for pid, record in records.items():
        if (pid != manifest.id and pid in available and record.state == 'loaded'
                and manifest.id in record.manifest.requires):
            visit(record.manifest)
    # Disabled manifests still reserve their names at boot. A manual archive
    # cannot silently replace a different product to make its own claims fit.
    composition = {pid: record.manifest for pid, record in records.items() if record.manifest}
    composition[manifest.id] = manifest
    if collision := conflict(composition.values()):
        issues.append({'code': 'ownership_conflict', 'field': f'owns.{collision["kind"]}',
                       'message': f'{collision["kind"]} {collision["name"]!r} is claimed by '
                                  f'{collision["owners"][0]} and {collision["owners"][1]}. '
                                  'Update or remove the previous owner before installing this archive.',
                       'expected': 'one owner', 'actual': list(collision['owners'])})
    return issues
