"""Explicit public product activation for backend tests, without legacy aliases."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sys
from urllib.parse import urlsplit

import pytest


ROOT = Path(__file__).resolve().parents[2]
BUNDLED = ROOT / 'bundled'
PRODUCTS = tuple(sorted(path.parent.name for path in BUNDLED.glob('*/plugin.json')))


def bootstrap_public_packages():
    """Make package imports collectable; do not import a product or its ORM."""
    for plugin_id in reversed(PRODUCTS):
        path = str(BUNDLED / plugin_id)
        if path not in sys.path:
            sys.path.insert(0, path)


def validate_plugins(enabled):
    if isinstance(enabled, str):
        raise ValueError('Pass plugin ids as a tuple/list, not one string.')
    enabled = tuple(enabled)
    if any(not isinstance(pid, str) for pid in enabled):
        raise ValueError('Every requested plugin id must be a string.')
    unknown = sorted(set(enabled) - set(PRODUCTS))
    if unknown:
        raise ValueError('Unknown public test plugins: ' + ', '.join(unknown))
    return frozenset(enabled)


def marked_plugins(node):
    """The closest marker replaces the outer selection; plugins() means OFF."""
    marker = node.get_closest_marker('plugins')
    if marker is None:
        return frozenset()
    if marker.kwargs:
        raise ValueError('Use @pytest.mark.plugins("video", ...), without keyword arguments.')
    return validate_plugins(marker.args)


def reset_capability_caches():
    from app import capabilities
    capabilities.clear_import_cache()
    capabilities._cache_plugin_signature = None


@contextmanager
def isolated_plugin_runtime():
    """Reset host globals even for tests that never construct an application.

    Retain imported product modules and their mapped classes. Re-importing those
    classes would duplicate SQLAlchemy mappings in a process that runs many files.
    """
    from app import config
    from app.engines import registry as engines
    from app.plugins import registry

    original_path = list(sys.path)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(config, 'DEFAULTS', deepcopy(config.DEFAULTS))
        patch.setattr(config, '_cache', None)
        patch.setattr(engines, '_specs', {})
        patch.setattr(registry, '_ACTIVE', None)
        reset_capability_caches()
        try:
            yield
        finally:
            reset_capability_caches()
            sys.path[:] = original_path


def install_hf_gate_guard(monkeypatch):
    """Refuse only the shared HF model-listing transport, including late imports.

    Both historical and product cloud gates catch URLError as their ordinary
    offline/fail-open path. Keeping the seam below the modules avoids importing
    an unrequested product merely to patch it. Tests can replace urlopen later.
    """
    import urllib.error
    import urllib.request

    original = urllib.request.urlopen

    def offline_model_gate(request, *args, **kwargs):
        url = urlsplit(str(getattr(request, 'full_url', request)))
        if (url.hostname == 'huggingface.co' and url.path.startswith('/api/models/')
                and url.path.endswith('/tree/main')):
            raise urllib.error.URLError('The unit suite does not contact the HF model gate.')
        return original(request, *args, **kwargs)

    monkeypatch.setattr(urllib.request, 'urlopen', offline_model_gate)


def create_test_app(tmp_path, monkeypatch, enabled=(), config_object=None):
    """Boot the real loader with only the requested owners and temporary data."""
    from app import config, create_app
    from app.engines import registry as engines
    from app.plugins import registry

    enabled = validate_plugins(enabled)
    tmp_path.mkdir(parents=True, exist_ok=True)
    for key, name in (('LDS_DATA_DIR', 'data'), ('LDS_CONFIG', 'config.json'),
                      ('LDS_ENV', '.env'), ('LDS_EXTENSIONS_DIR', 'no-extensions'),
                      ('LDS_PLUGINS_DIR', 'no-external-plugins')):
        monkeypatch.setenv(key, str(tmp_path / name))
    monkeypatch.setenv('LDS_PLUGIN_DISTRIBUTION', 'store')
    monkeypatch.delenv('LDS_BUNDLED_DIR', raising=False)
    monkeypatch.delenv('LDS_PLUGINS', raising=False)
    monkeypatch.setattr(config, 'ENV_PATH', tmp_path / '.env')
    config._cache = None
    engines._specs = {}
    registry.set_active(None)
    reset_capability_caches()
    if enabled:
        monkeypatch.setenv('LDS_BUNDLED_DIR', str(BUNDLED))
        config.save_config({'plugins': {'enabled': {
            pid: pid in enabled for pid in PRODUCTS}}})
    options = {'TESTING': True, 'WTF_CSRF_ENABLED': False,
               'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:'}
    options.update(config_object or {})
    if not options['TESTING']:
        raise ValueError('The test factory must not start production workers.')
    application = create_app(options)
    records = application.extensions['lds_plugins'].records
    unexpected = {pid: record.state for pid, record in records.items()
                  if pid not in enabled and record.state != 'disabled'}
    failed = {pid: (records[pid].state, records[pid].error) if pid in records else 'missing'
              for pid in enabled if pid not in records or records[pid].state != 'loaded'}
    if failed or unexpected or (not enabled and records):
        dispose_test_app(application)
        raise AssertionError(f'Explicit plugin boot failed: requested={failed}, unexpected={unexpected}')
    if 'cloud_training' in enabled:
        # public_api_v1 copies its functions from cloud_video_training when it is first imported,
        # and product modules stay imported for the whole worker. Imported lazily by a test that had
        # replaced cvt.launch_cloud_video_training (the continue-route test), it kept that stand-in
        # for every later test of the worker: the v2026.10.19 release runs failed the preflight and
        # video rank tests that way. Bind the real functions here, before any test body patches them.
        import importlib
        importlib.import_module('lds_cloud_training.public_api_v1')
    return application


def dispose_test_app(application):
    from app.extensions import db
    with application.app_context():
        db.session.remove()
        db.engine.dispose()
