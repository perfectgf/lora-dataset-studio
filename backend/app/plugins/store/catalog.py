"""Validate a signed catalog and resolve compatible package sets without imports."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from ..compatibility import host_issues
from ..manifest import parse_manifest
from ..ownership import conflict
from .client import StoreError
from .media_contract import parse_presentation


@dataclass(frozen=True)
class Release:
    manifest: object
    target: str
    manifest_sha256: str
    changelog: tuple
    price: dict
    images: tuple = ()

    def payload(self):
        return {'manifest': self.manifest.summary(), 'target': self.target,
                'changelog': list(self.changelog), 'price': self.price,
                'compatibility_issues': host_issues(self.manifest),
                'presentation': {'schema_version': 1, 'images': [
                    {**item, 'url': f'/api/plugins/store/media/{self.manifest.id}/{self.manifest.version}/'
                     f'{item["id"]}/{item["target"].rsplit("/", 1)[-1]}'} for item in self.images]}}


def parse_catalog(data, config):
    """The root configuration, not a publisher claim, authorizes historic IDs."""
    if not isinstance(data, dict) or data.get('schema_version') != 1:
        raise StoreError('The catalog format is not supported by this app.')
    products = data.get('products')
    if not isinstance(products, list) or len(products) > 1000:
        raise StoreError('The catalog product list is invalid.')
    result = {}
    try:
        for product in products:
            plugin_id = product['id']
            if config.scoped and not config.serves(plugin_id):
                continue
            if plugin_id in result:
                raise ValueError('duplicate product')
            versions = {}
            releases = product['releases']
            if not isinstance(releases, list) or not releases or len(releases) > 100:
                raise ValueError('invalid releases')
            for release in releases:
                manifest = parse_manifest(release['manifest'], Path('.'), official=plugin_id in config.official_ids)
                if manifest.bundled or manifest.id != plugin_id or manifest.contract['schema_version'] != 2:
                    raise ValueError('invalid package identity')
                if not manifest.official and not config.serves(plugin_id):
                    raise ValueError('publisher is not approved for this store')
                version = Version(manifest.version)
                if version in versions:
                    raise ValueError('duplicate release')
                target, digest = release['target'], release['manifest_sha256']
                if (not isinstance(target, str) or not target.endswith('.ldsplugin')
                        or not isinstance(digest, str) or len(digest) != 64
                        or any(c not in '0123456789abcdef' for c in digest)):
                    raise ValueError('invalid target')
                price = release.get('price', {'kind': 'free'})
                if not isinstance(price, dict) or price.get('kind') not in ('free', 'paid'):
                    raise ValueError('invalid price')
                if price['kind'] == 'paid' and (not isinstance(price.get('product'), str) or not price['product']):
                    raise ValueError('invalid commerce product')
                # Only the operator-configured Patreon source may label a
                # release as a Patreon product; that source sells nothing else.
                patreon = getattr(config, 'provider', '') == 'patreon'
                if patreon != (price.get('provider') == 'patreon') or (patreon and price['kind'] != 'paid'):
                    raise ValueError('invalid Patreon release')
                changelog = release.get('changelog', [])
                if not isinstance(changelog, list) or any(not isinstance(s, str) for s in changelog):
                    raise ValueError('invalid changelog')
                images = parse_presentation(release.get('presentation'), manifest.id, manifest.version)
                versions[version] = Release(manifest, target, digest, tuple(changelog), price, images)
            result[plugin_id] = sorted(versions.values(), key=lambda r: Version(r.manifest.version), reverse=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise StoreError('The authenticated catalog contains an invalid product or release.') from exc
    return result


def _satisfies(version, ranges, *, installed=False):
    version = Version(version)
    if version.is_prerelease and not ranges:
        return installed
    return all(SpecifierSet(spec).contains(version, prereleases=SpecifierSet(spec).prereleases is True) for spec in ranges)


# Requirements that the members version of LDS changes. Python or platform
# issues would refuse a release on every core.
_CORE_REQUIREMENTS = ('compatibility.lds', 'compatibility.api', 'host_dependencies.')
# The core's own version requirements. A core dependency refused alone is this
# installation's environment, which repairing the public core fixes.
_CORE_VERSIONS = ('compatibility.lds', 'compatibility.api')


def members_releases(releases, members, *, current):
    """The releases of a public-catalog plugin that need the members version of LDS.

    ``releases`` are ``Release.payload()`` values, newest first. ``members``
    are the published members core versions (Patreon ``core-releases.json``),
    newest first; when they cannot be read, nothing is claimed.

    On the public core (``current``), a release qualifies while only core
    requirements refuse it (LDS range, plugin API, core dependencies), at least
    one of them its LDS range or plugin API, and its LDS range admits the
    latest members core. Assumption: the members core meets the plugin API and
    the core dependencies, since it always carries the newest API and pins its
    own dependencies; ``core-releases.json`` names only its version. The first
    release the public core accepts ends the list.

    A published members core cannot see what refuses the public one. It infers
    the public version from its own date (2026.10.18.3 builds on 2026.10.18)
    and lists the releases whose LDS range refuses that version, so a release
    the public core refuses for its plugin API alone is not recognized there.
    Members-only plugins (the Patreon catalog) are reserved as a whole and list
    nothing here."""
    if not members:
        return []
    public = None  # This core is the public one: its issues are the public refusals.
    if current in members:
        parts = Version(current).release
        if len(parts) < 4:
            return []
        public = '.'.join(map(str, parts[:3]))
    found = []
    for release in releases:
        if release['price'].get('provider') == 'patreon':
            return []
        lds = release['manifest']['compatibility']['lds']
        issues = release['compatibility_issues']
        refused = (any(issue['field'] in _CORE_VERSIONS for issue in issues) if public is None
                   else not _satisfies(public, [lds]))
        if not refused:
            break  # The public core accepts this release, and the older ones.
        if (_satisfies(members[0], [lds])
                and all(issue['field'].startswith(_CORE_REQUIREMENTS) for issue in issues)):
            found.append({'version': release['manifest']['version'], 'changelog': release['changelog'],
                          'available': not issues})
    return found


def resolve(catalog, requested, installed, *, version=None, active=None, enable_requested=True):
    """Backtrack the small reviewed catalog, preserving reverse dependencies.

    Enabled products constrain dependencies. All installed manifests, including
    disabled products, reserve ownership at boot. A split product can therefore
    require updating its old combined owner in this same consented transaction,
    without acquiring a permanent dependency on it or enabling it.
    """
    roots = [requested] if isinstance(requested, str) else requested
    if (not isinstance(roots, (list, tuple)) or not 1 <= len(roots) <= 50
            or any(not isinstance(pid, str) or pid not in catalog for pid in roots)):
        raise StoreError('This plugin is not available from this catalog.')
    roots = sorted(set(roots))
    constraints = {}
    if version is not None:
        if len(roots) != 1:
            raise StoreError('Review a specific version of one plugin at a time.')
        try:
            constraints.setdefault(roots[0], []).append('==' + str(Version(version)))
        except (ValueError, TypeError) as exc:
            raise StoreError('The requested version is invalid.') from exc

    attempts = 0
    active = set(installed) if active is None else set(active)

    def enabled_ids(chosen):
        result = active | (set(roots) if enable_requested else set())
        while True:
            expanded = result | {dep for pid in result if pid in chosen for dep in chosen[pid].manifest.requires}
            if expanded == result:
                return result
            result = expanded

    def visit(chosen, needed):
        nonlocal attempts
        attempts += 1
        if attempts > 10000:
            raise StoreError('The dependency plan is too complex. Select fewer version changes.')
        if not needed:
            return None if any(host_issues(r.manifest) for pid, r in chosen.items() if pid in enabled_ids(chosen)) else chosen
        pid = needed[0]
        if pid in chosen:
            return visit(chosen, needed[1:])
        candidates = list(catalog.get(pid, []))
        current = installed.get(pid)
        if current and pid not in roots:
            # An installed version need not remain offered for sale to satisfy a dependency.
            candidates.insert(0, Release(current, '', '', (), {'kind': 'installed'}))
        for candidate in candidates:
            m = candidate.manifest
            next_chosen = {**chosen, pid: candidate}
            enabled = enabled_ids(next_chosen)
            ranges = constraints.get(pid, []) + [spec for release in chosen.values()
                     if release.manifest.id in enabled
                     for dep, spec in release.manifest.contract.get('dependency_versions', {}).items() if dep == pid]
            if ((candidate.target or pid in enabled) and host_issues(m)) or not _satisfies(m.version, ranges, installed=not candidate.target):
                continue
            if pid in enabled and pid in m.requires:
                continue
            if conflict(r.manifest for r in next_chosen.values()):
                continue
            if _has_cycle({key: value for key, value in next_chosen.items() if key in enabled}):
                continue
            if any(dep in next_chosen and not _satisfies(next_chosen[dep].manifest.version, [spec])
                   for r in next_chosen.values() if r.manifest.id in enabled
                   for dep, spec in r.manifest.contract.get('dependency_versions', {}).items()):
                continue
            # A package selected earlier as an OFF owner may become a required
            # dependency later in the search. Traverse the whole enabled set.
            dependencies = {dep for key, r in next_chosen.items() if key in enabled for dep in r.manifest.requires}
            result = visit(next_chosen, sorted(dependencies - next_chosen.keys()) + needed[1:])
            if result is not None:
                return result
        return None

    solution = visit({}, roots + sorted((set(installed) | active) - set(roots)))
    if solution is None:
        raise StoreError('No compatible set of plugin versions satisfies the installed plugins, their ownership and this app. '
                         'Update or remove an older combined product if it reserves this feature, even when disabled.')
    # A real cycle is rejected separately, even if an already chosen candidate
    # made the search terminate. Sort dependencies before their dependents.
    enabled = enabled_ids(solution)
    order, remaining = [], {pid: r for pid, r in solution.items() if pid in enabled or r.target}
    while remaining:
        ready = sorted(pid for pid, r in remaining.items()
                       if pid not in enabled or not (set(r.manifest.requires) & remaining.keys()))
        if not ready:
            raise StoreError('The catalog contains a dependency cycle.')
        for pid in ready:
            order.append(remaining.pop(pid))
    return order


def _has_cycle(chosen):
    visiting, complete = set(), set()

    def visit(pid):
        if pid in visiting:
            return True
        if pid in complete or pid not in chosen:
            return False
        visiting.add(pid)
        if any(visit(dep) for dep in chosen[pid].manifest.requires):
            return True
        visiting.remove(pid)
        complete.add(pid)
        return False

    return any(visit(pid) for pid in chosen)


def plan_digest(releases, installed, catalog_identity):
    value = {'store': catalog_identity,
             'packages': [(r.manifest.id, r.manifest.version, r.target, r.manifest_sha256,
                           list(r.manifest.permissions), r.price) for r in releases],
             'installed': sorted((pid, m.version) for pid, m in installed.items())}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
