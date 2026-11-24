"""Stable first-party identifiers require provenance outside the plugin package.

An archive cannot grant itself first-party rights. Only a store whose trusted
root is explicitly authorized for LDS products, or the legacy migration, writes
these receipts. They are app-managed state, never files supplied by a ZIP.
This is an installation boundary, not protection against a local administrator
or an already trusted in-process plugin modifying the application itself.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

OFFICIAL_IDS = frozenset({
    'api_engines', 'camera_angles', 'civitai_publish', 'cloud_training',
    'hf_publish', 'model_tools', 'scrape', 'video', 'canvas',
    'image_upscale', 'resource_monitor', 'seedvr2', 'live',
    'manga', 'dlss5', 'creature_battle', 'qwen_dataset', 'remove_background',
})


def is_official_id(plugin_id):
    return isinstance(plugin_id, str) and plugin_id in OFFICIAL_IDS


def manifest_digest(directory):
    return hashlib.sha256((Path(directory) / 'plugin.json').read_bytes()).hexdigest()


def valid_provenance(provenance, plugin_id, directory):
    """Validate a previously authenticated receipt against the staged bytes."""
    if not is_official_id(plugin_id) or not isinstance(provenance, dict):
        return False
    if (provenance.get('id') != plugin_id or provenance.get('publisher') != 'lds'
            or provenance.get('source') not in ('verified_store', 'legacy_migration')):
        return False
    try:
        return provenance.get('manifest_sha256') == manifest_digest(directory)
    except OSError:
        return False


def receipt_path(plugin_id):
    from .. import config as cfg
    from .storage import managed_path
    if not is_official_id(plugin_id):
        raise ValueError('This is not an LDS product identifier.')
    return managed_path(cfg.data_dir(), 'plugin-store', 'installed', plugin_id + '.json')


def installed_provenance(plugin_id, directory):
    if not is_official_id(plugin_id):
        return None
    try:
        provenance = json.loads(receipt_path(plugin_id).read_text(encoding='utf-8'))
        return provenance if valid_provenance(provenance, plugin_id, directory) else None
    except (OSError, ValueError):
        return None


def record_install(plugin_id, directory, provenance):
    from .storage import _write_json
    if not valid_provenance(provenance, plugin_id, directory):
        raise ValueError('The official package does not match its verified origin.')
    _write_json(receipt_path(plugin_id), provenance)
    # Uninstall must not resurrect an old bundled copy after this migration.
    _write_json(retired_path(plugin_id), {'id': plugin_id, 'external': True})


def forget_install(plugin_id):
    if is_official_id(plugin_id):
        receipt_path(plugin_id).unlink(missing_ok=True)


def retired_path(plugin_id):
    from .. import config as cfg
    from .storage import managed_path
    if not is_official_id(plugin_id):
        raise ValueError('This is not an LDS product identifier.')
    return managed_path(cfg.data_dir(), 'plugin-store', 'retired-bundled', plugin_id + '.json')


def bundled_retired(plugin_id):
    if not is_official_id(plugin_id):
        return False
    # A damaged marker is not permission to silently load an older version.
    return retired_path(plugin_id).exists()
