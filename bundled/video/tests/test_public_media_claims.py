"""Media settings ownership and read-only probes against the real public SDK."""
import importlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT / 'backend' / 'tests'))
import test_public_sdk_integration as sdk_tests  # noqa: E402

activate = sdk_tests.activate
host = sdk_tests.host
no_external_runtime = sdk_tests.no_external_runtime


def manifest(pid):
    return json.loads((ROOT / 'bundled' / pid / 'plugin.json').read_text(encoding='utf-8'))


def test_video_claims_its_product_settings_and_leaves_shared_decoders_to_host():
    assert set(manifest('video')['owns']['config_sections']) == {
        'custom_shots', 'video_caption', 'video_bank'}
    assert manifest('live')['owns']['config_sections'] == []
    for pid in ('video', 'live'):
        assert 'video_text' not in manifest(pid)['owns']['config_sections']
        assert not {'live', 'neural_render', 'dlss'} & set(manifest(pid)['owns']['config_sections'])


@pytest.mark.parametrize(('pid', 'expected'), [
    ('video', {'video', 'video_detail', 'video_decode', 'video_detect', 'video_encode'}),
    ('scrape', {'scrape_deps', 'scrape_deps_detail'}),
    ('civitai_publish', {'civitai'}),
])
def test_public_probes_are_declared_and_registered_by_their_product(host, pid, expected):
    loaded = activate(host, {pid})
    assert loaded.records[pid].state == 'loaded', loaded.records[pid].error
    assert expected <= set(manifest(pid)['owns']['probes'])
    assert expected <= {key for key, (owner, _fn) in loaded.probes.items() if owner == pid}
    assert 'video_text' not in loaded.probes


def test_live_off_does_not_inspect_models_or_encoder(host, monkeypatch):
    loaded = activate(host, {'live'})
    from app import config
    setup = importlib.import_module('lds_live.setup')
    monkeypatch.setattr(setup, 'missing_weights', lambda: pytest.fail('OFF Live inspected H3 weights'))
    monkeypatch.setattr(setup, 'encoder_ready', lambda: pytest.fail('OFF Live inspected the encoder'))
    config.save_config({'plugins': {'enabled': {'live': False}}})
    with host[0].app_context():
        assert loaded.probes['live'][1]() == {}


@pytest.mark.parametrize(('decode', 'detect', 'encode'), [
    (True, True, True), (False, True, True), (True, False, True), (True, True, False),
])
def test_video_probe_preserves_main_piecewise_readiness_and_python_choices(host, monkeypatch, decode, detect, encode):
    loaded = activate(host, {'video'})
    from app import capabilities, config
    calls = []
    config.save_config({'video': {'python': 'fixture-decoder'}, 'shot_detect': {'python': 'fixture-detector'}})

    def imports(key, python, expression):
        calls.append((key, python, expression))
        return decode if key == 'video_decode' else detect

    monkeypatch.setattr(capabilities, '_cached_import', imports)
    monkeypatch.setattr(capabilities.ffmpeg_tools, 'ffmpeg_ready', lambda: {'ok': encode, 'reason': 'fixture unavailable'})
    keys = ('video', 'video_detail', 'video_decode', 'video_detect', 'video_encode')
    with host[0].app_context():
        payload = {key: loaded.probes[key][1]() for key in keys}
    assert payload['video'] is (decode and detect and encode)
    assert (payload['video_decode'], payload['video_detect'], payload['video_encode']) == (decode, detect, encode)
    assert payload['video_detail'] == ('video extra ready' if all((decode, detect, encode)) else
        'missing: ' + ', '.join(x for present, x in [
            (decode, 'av (video decoding)'), (detect, 'shot detection (transnetv2-pytorch)'),
            (encode, 'ffmpeg (clip encoding) — fixture unavailable')] if not present))
    assert set(calls) == {
        ('video_decode', 'fixture-decoder', capabilities.CAPABILITY_IMPORTS['video']),
        ('video_detect', 'fixture-detector', capabilities.CAPABILITY_IMPORTS['shot_detect'])}


def test_video_off_skips_all_previously_registered_probes(host, monkeypatch):
    loaded = activate(host, {'video'})
    from app import capabilities, config
    from lds_video import probes, video_test_studio

    def forbidden(*args, **kwargs):
        pytest.fail('OFF Video invoked a capability, model or runtime probe')

    monkeypatch.setattr(capabilities, 'probe_video', forbidden)
    monkeypatch.setattr(probes, '_comfy', forbidden)
    monkeypatch.setattr(video_test_studio, 'missing_weights', forbidden)
    config.save_config({'plugins': {'enabled': {'video': False}}})
    with host[0].app_context():
        payload = {key: fn() for key, (owner, fn) in loaded.probes.items() if owner == 'video'}
    assert payload == {'video': False, 'video_detail': '', 'video_decode': False,
                       'video_detect': False, 'video_encode': False, 'video_host_ready': False,
                       'comfyui.video_studio_missing': [], 'comfyui.video_studio_ready': False,
                       'comfyui.video_studio_options': {}, 'comfyui.video_studio_sage': {},
                       'comfyui.video_studio_reference': {},
                       'comfyui.h3_attention_nodes_installed': False,
                       'comfyui.h3_attention_nodes_missing': []}


@pytest.mark.parametrize('missing', [None, 'curl_cffi', 'gallery_dl', 'bs4', 'cloudscraper', 'instaloader', 'ddgs', 'yt_dlp'])
def test_scrape_package_presence_matches_public_main_without_import_or_network(host, monkeypatch, missing):
    loaded = activate(host, {'scrape'})
    seen = []

    def find(name):
        seen.append(name)
        return None if name == missing else object()

    # The product delegates presence checks to the real SDK; only its lookup
    # transport is replaced so no optional package is imported.
    monkeypatch.setattr(importlib.util, 'find_spec', find)
    with host[0].app_context():
        assert loaded.probes['scrape_deps'][1]() is (missing is None)
        assert loaded.probes['scrape_deps_detail'][1]() == ('scrape deps OK' if missing is None else 'missing: ' + missing)
    assert seen == ['curl_cffi', 'gallery_dl', 'bs4', 'cloudscraper', 'instaloader', 'ddgs', 'yt_dlp'] * 2


def test_scrape_off_never_inspects_installed_packages(host, monkeypatch):
    loaded = activate(host, {'scrape'})
    from app import config
    monkeypatch.setattr(importlib.util, 'find_spec', lambda *_a: pytest.fail('OFF Scrape inspected packages'))
    config.save_config({'plugins': {'enabled': {'scrape': False}}})
    with host[0].app_context():
        assert loaded.probes['scrape_deps'][1]() is False
        assert loaded.probes['scrape_deps_detail'][1]() == ''


@pytest.mark.parametrize(('credential', 'expected'), [(None, False), ('fixture-only-key', True)])
def test_civitai_probe_preserves_dict_shape_and_only_reports_key_presence(host, monkeypatch, credential, expected):
    loaded = activate(host, {'civitai_publish'})
    from lds_sdk import credentials
    monkeypatch.setattr(credentials, 'civitai_api_key', lambda: credential)
    with host[0].app_context():
        result = loaded.probes['civitai'][1]()
    assert result == {'ok': expected, 'detail': 'key set' if expected else 'key missing'}
    assert 'fixture-only-key' not in json.dumps(result)


def test_civitai_resolver_exception_uses_main_secret_fallback(host, monkeypatch):
    loaded = activate(host, {'civitai_publish'})
    from lds_sdk import config, credentials
    monkeypatch.setattr(credentials, 'civitai_api_key', lambda: (_ for _ in ()).throw(OSError('fixture')))
    monkeypatch.setattr(config, 'secret', lambda key: 'fixture-key' if key == 'CIVITAI_API_KEY' else None)
    with host[0].app_context():
        assert loaded.probes['civitai'][1]() == {'ok': True, 'detail': 'key set'}


def test_civitai_off_never_resolves_credentials(host, monkeypatch):
    loaded = activate(host, {'civitai_publish'})
    from app import config
    from lds_sdk import credentials
    monkeypatch.setattr(credentials, 'civitai_api_key', lambda: pytest.fail('OFF Civitai resolved a credential'))
    config.save_config({'plugins': {'enabled': {'civitai_publish': False}}})
    with host[0].app_context():
        assert loaded.probes['civitai'][1]() == {'ok': False, 'detail': 'plugin disabled'}


def test_live_readiness_is_independent_of_video_plugin(host, monkeypatch):
    loaded = activate(host, {'live'})
    setup = importlib.import_module('lds_live.setup')
    from lds_sdk import local_render
    monkeypatch.setattr(setup, 'missing_weights', lambda: [])
    monkeypatch.setattr(setup, 'encoder_ready', lambda: True)
    monkeypatch.setattr(local_render, 'comfyui_reachable', lambda: True)
    monkeypatch.setattr(setup.h3, 'studio_ready', lambda missing: not missing)
    monkeypatch.setattr(setup.h3, 'option_availability', lambda: {'turbo': {'available': True}})
    monkeypatch.setattr(setup.h3, 'sage_available', lambda: False)
    with host[0].app_context():
        result = loaded.probes['live'][1]()
    assert loaded.records['video'].state == 'disabled'
    assert result['ready'] and result['encoder'] and result['missing'] == []
    assert result['options'] == {'turbo': {'available': True}}


def test_shot_detection_install_is_owned_by_loaded_video_after_core_transfer(host, monkeypatch):
    from app import config, setup_installer as installer
    # This source-only product fix is also qualified on the pre-transfer host.
    # The integrated host owns the separate managed-capability allowlist and
    # dispatch to its dedicated environment worker; it is never a Flask pip job.
    if not hasattr(installer, '_MANAGED_CAPABILITY_ACTIONS'):
        monkeypatch.setattr(installer, '_CAPABILITY_ML_ACTIONS', (*installer._CAPABILITY_ML_ACTIONS, 'shot_detect'))
    else:
        assert 'shot_detect' in installer._MANAGED_CAPABILITY_ACTIONS
    monkeypatch.setattr(installer, 'INSTALL_ACTIONS', tuple(a for a in installer.INSTALL_ACTIONS if a != 'shot_detect'))
    monkeypatch.setattr(installer, '_WORKERS', {a: fn for a, fn in installer._WORKERS.items() if a != 'shot_detect'})
    loaded = activate(host, {'video'})
    assert loaded.records['video'].state == 'loaded', loaded.records['video'].error
    with host[0].app_context():
        assert installer.known_action('shot_detect')
        spec = installer.plugin_action_spec('shot_detect')
        assert spec['plugin'] == 'video'
        assert spec['python'] == 'capability' and spec['capability'] == 'shot_detect'
        assert spec['packages'] == ['transnetv2-pytorch', 'av']
        assert 'shot_detect' in manifest('video')['owns']['install_actions']
        config.save_config({'plugins': {'enabled': {'video': False}}})
        assert not installer.known_action('shot_detect')
