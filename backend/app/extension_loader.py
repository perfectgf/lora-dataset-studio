"""Generic extension loader.

Optional packages dropped into ``backend/extensions/`` (gitignored, excluded
from release bundles) are imported at boot and get ``register(app, csrf)``
called. With the directory absent — every normal install — this whole module
is a no-op. ``LDS_EXTENSIONS=0`` disables loading; ``LDS_EXTENSIONS_DIR``
overrides the directory (used by tests).
"""
import importlib
import logging
import os
import sys

log = logging.getLogger(__name__)


def _extensions_dir():
    override = os.environ.get('LDS_EXTENSIONS_DIR')
    if override:
        return override
    backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(backend_root, 'extensions')


def load_extensions(app, csrf):
    manifest = []
    app.config['EXTENSIONS_MANIFEST'] = manifest
    if os.environ.get('LDS_EXTENSIONS') == '0':
        return
    base = _extensions_dir()
    if not os.path.isdir(base):
        return
    if base not in sys.path:
        sys.path.insert(0, base)
    for name in sorted(os.listdir(base)):
        if not os.path.isfile(os.path.join(base, name, '__init__.py')):
            continue
        try:
            mod = importlib.import_module(name)
            mod.register(app, csrf)
        except Exception:
            # A broken extension must never take the app down with it.
            log.exception('extension %r failed to load; continuing without it', name)
            continue
        manifest.append({
            'name': name,
            'version': getattr(mod, '__version__', None),
            'frontend_entry': getattr(mod, 'FRONTEND_ENTRY', None),
        })
        log.info('extension loaded: %s', name)
