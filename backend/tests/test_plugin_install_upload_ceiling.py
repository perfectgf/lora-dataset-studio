"""Install from a ZIP takes an archive as large as the Store accepts.

An archive over 64 MiB installed from the Store but was refused with a 413 by Install from a ZIP,
which still had the generic request ceiling."""
from flask import request

from app.plugins.install import MAX_TOTAL_BYTES


def test_plugin_zip_install_accepts_a_store_sized_archive(app):
    with app.test_request_context('/api/plugins/install', method='POST'):
        assert request.endpoint == 'plugins.install_plugin'
        assert request.max_content_length >= MAX_TOTAL_BYTES


def test_other_routes_keep_the_generic_ceiling(app):
    with app.test_request_context('/api/health', method='GET'):
        assert request.max_content_length == app.config['MAX_CONTENT_LENGTH']
