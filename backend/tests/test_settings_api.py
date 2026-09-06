import json
import pathlib
import pytest


UNSAFE_SECRET_CHARS = [
    pytest.param('\r', id='cr'),
    pytest.param('\n', id='lf'),
    pytest.param('\x00', id='nul'),
    pytest.param('\x0b', id='vertical-tab'),
    pytest.param('\x0c', id='form-feed'),
    pytest.param('\x1c', id='file-separator'),
    pytest.param('\x1d', id='group-separator'),
    pytest.param('\x1e', id='record-separator'),
    pytest.param('\x85', id='next-line'),
    pytest.param('\u2028', id='unicode-line-separator'),
    pytest.param('\u2029', id='unicode-paragraph-separator'),
    pytest.param('\t', id='other-control'),
    pytest.param('\u200b', id='unicode-format'),
]


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch):
    """GET /api/capabilities calls probe(), which hits every reachability
    probe. Stub the network/subprocess seams so this file never makes a
    real call, mirroring test_capabilities.py's isolation."""
    from app import capabilities
    capabilities._cache = None
    capabilities._cache_ts = 0.0
    capabilities._import_cache.clear()
    monkeypatch.setattr(capabilities, '_http_ok', lambda *a, **k: False)
    monkeypatch.setattr(capabilities, '_import_ok', lambda *a, **k: False)
    yield
    capabilities._cache = None
    capabilities._cache_ts = 0.0
    capabilities._import_cache.clear()


def test_get_settings_masks_secrets(client, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-secret')
    data = client.get('/api/settings').get_json()
    assert data['secrets']['OPENAI_API_KEY'] is True
    assert 'sk-secret' not in str(data)

def test_get_settings_exposes_identity_prompt_defaults(client):
    """The payload carries the shipped default prompts read-only, so the UI can
    show the real default text instead of a blank "leave blank" field — the four
    identity locks, and the five other prompt parts that became editable with
    them (markings lock, the two directives, the garment palette, the rendering
    tail, the per-framing detail)."""
    from app.services import face_variations as fv
    data = client.get('/api/settings').get_json()
    defaults = data['identity_prompt_defaults']
    assert defaults['face_single'] == fv.IDENTITY_GUARD
    assert defaults['face_multi'] == fv.IDENTITY_GUARD_MULTI
    assert defaults['klein_identity'] == fv.IDENTITY_GUARD_KLEIN
    assert defaults['klein_improve'] == fv.KLEIN_IMAGE_IMPROVE_PROMPT
    assert set(defaults) == set(fv.IDENTITY_PROMPT_KINDS) | set(fv.PROMPT_PART_KINDS)
    # Every one of them carries REAL text — an empty default would render as an
    # empty box the user cannot tell from "nothing is applied here".
    assert all(v.strip() for v in defaults.values())


def test_put_settings_persists_config_and_secret(client, tmp_path):
    r = client.put('/api/settings', json={
        'config': {'ollama': {'url': 'http://127.0.0.1:11500'}},
        'secrets': {'GEMINI_API_KEY': 'g-123'}})
    assert r.status_code == 200
    assert r.get_json()['config']['ollama']['url'] == 'http://127.0.0.1:11500'
    assert r.get_json()['secrets']['GEMINI_API_KEY'] is True


@pytest.mark.parametrize('separator', UNSAFE_SECRET_CHARS)
def test_put_settings_rejects_secret_environment_injection(
        client, monkeypatch, separator):
    """A secret occupies exactly one .env assignment; it cannot introduce the
    lifecycle variables that decide how this process updates or restarts."""
    import os
    before = client.get('/api/settings').get_json()['config']['ollama']['url']
    monkeypatch.delenv('LDS_RESTART_MODE', raising=False)
    monkeypatch.delenv('LDS_RUNTIME', raising=False)
    monkeypatch.delenv('LDS_BIND_MANAGED', raising=False)
    monkeypatch.delenv('FLASK_DEBUG', raising=False)

    response = client.put('/api/settings', json={
        'config': {'ollama': {'url': 'http://must-not-save.invalid'}},
        'secrets': {'OPENAI_API_KEY': f'key{separator}FLASK_DEBUG=1'},
    })

    assert response.status_code == 400
    assert 'single line' in response.get_json()['error']
    assert client.get('/api/settings').get_json()['config']['ollama']['url'] == before
    assert 'LDS_RESTART_MODE' not in os.environ
    assert 'LDS_RUNTIME' not in os.environ
    assert 'LDS_BIND_MANAGED' not in os.environ
    assert 'FLASK_DEBUG' not in os.environ


@pytest.mark.parametrize('separator', [
    pytest.param('\x0b', id='vertical-tab'),
    pytest.param('\u2028', id='unicode-line-separator'),
])
def test_put_settings_refuses_poisoned_existing_env_before_saving_config(
        client, monkeypatch, separator):
    import os
    from app import config
    before_url = client.get('/api/settings').get_json()['config']['ollama']['url']
    poisoned = f'OPENAI_API_KEY=old{separator}FLASK_DEBUG=1\n'.encode('utf-8')
    config.ENV_PATH.write_bytes(poisoned)
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('FLASK_DEBUG', raising=False)

    response = client.put('/api/settings', json={
        'config': {'ollama': {'url': 'http://must-not-save.invalid'}},
        'secrets': {'GEMINI_API_KEY': 'safe-value'},
    })

    assert response.status_code == 400
    assert 'existing .env' in response.get_json()['error']
    assert client.get('/api/settings').get_json()['config']['ollama']['url'] == before_url
    assert config.ENV_PATH.read_bytes() == poisoned
    assert 'GEMINI_API_KEY' not in os.environ
    assert 'FLASK_DEBUG' not in os.environ

def test_put_settings_clears_skip_when_dir_provided(client, tmp_path):
    """Entering a ComfyUI directory annuls a prior "continue without ComfyUI" skip —
    the stored flag is cleared so it can't resurface if base_dir is later emptied."""
    from app import config
    config.save_config({'comfyui': {'setup_skipped': True}})
    r = client.put('/api/settings', json={'config': {'comfyui': {'base_dir': str(tmp_path / 'Comfy')}}})
    assert r.status_code == 200
    assert r.get_json()['config']['comfyui']['setup_skipped'] is False


def test_put_settings_skip_persists_when_dir_empty(client):
    """Saving the skip with no directory keeps the flag set (the wizard's skip flow)."""
    r = client.put('/api/settings', json={'config': {'comfyui': {'setup_skipped': True}}})
    assert r.status_code == 200
    assert r.get_json()['config']['comfyui']['setup_skipped'] is True


def test_put_settings_saves_scrape_credentials(client, monkeypatch):
    """Scrape credentials are presence-only and effective without restart."""
    import os
    monkeypatch.setenv('PEXELS_API_KEY', '')
    r = client.put('/api/settings', json={'secrets': {'REDDIT_CLIENT_ID': 'my-cid',
                                                      'CIVITAI_API_KEY': 'civ-key',
                                                      'PEXELS_API_KEY': 'pexels-key'}})
    assert r.status_code == 200
    secrets = r.get_json()['secrets']
    assert secrets['REDDIT_CLIENT_ID'] is True and secrets['CIVITAI_API_KEY'] is True
    assert secrets['PEXELS_API_KEY'] is True
    payload = str(r.get_json())
    assert 'my-cid' not in payload and 'pexels-key' not in payload  # presence only
    assert os.environ['REDDIT_CLIENT_ID'] == 'my-cid'      # effective immediately
    from app.scrape.sources import reddit
    from app.scrape.sources.civitai import civitai_api_key
    from app.scrape.sources.pexels import pexels_api_key
    assert reddit._client_id() == 'my-cid'
    assert civitai_api_key() == 'civ-key'
    assert pexels_api_key() == 'pexels-key'


def test_delete_scrape_credential_falls_back_to_shared_id(client, monkeypatch):
    """Removing the saved Reddit client id must drop it from the env too, so the
    source falls back to the shared gallery-dl id instead of a stale value."""
    import os
    from app.scrape.sources import reddit
    monkeypatch.setattr(reddit, 'resolve_cookies', lambda key: None)  # ignore any local admin file
    client.put('/api/settings', json={'secrets': {'REDDIT_CLIENT_ID': 'my-cid'}})
    r = client.delete('/api/settings/secret/REDDIT_CLIENT_ID')
    assert r.status_code == 200
    assert r.get_json()['secrets']['REDDIT_CLIENT_ID'] is False
    assert 'REDDIT_CLIENT_ID' not in os.environ
    assert reddit._client_id() == reddit._GDL_CLIENT_ID


def test_delete_pexels_api_key_clears_runtime_secret_without_leak(client):
    import os
    from app.scrape.sources.pexels import pexels_api_key

    client.put('/api/settings', json={'secrets': {'PEXELS_API_KEY': 'delete-me'}})
    assert pexels_api_key() == 'delete-me'
    r = client.delete('/api/settings/secret/PEXELS_API_KEY')

    assert r.status_code == 200
    assert r.get_json()['secrets']['PEXELS_API_KEY'] is False
    assert 'delete-me' not in str(r.get_json())
    assert 'PEXELS_API_KEY' not in os.environ
    assert pexels_api_key() is None


def test_put_tolerates_unknown_section(client):
    """Behavior change: an unknown config section used to 400 the whole Save
    (see test_put_settings_preserves_unknown_config_section_on_disk for why that
    was a bug, not a guard). It is now dropped and reported, not rejected."""
    r = client.put('/api/settings', json={'config': {'nope': 1}})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()['preserved_unknown_sections'] == ['nope']

def test_put_rejects_non_object_section_value(client):
    """{"config": {"ollama": "x"}} would otherwise pass the top-level key-name
    check and let _deep_merge overwrite the whole 'ollama' section with a
    string, persistently corrupting config.json. Must be rejected AND leave
    the existing config untouched."""
    before = client.get('/api/settings').get_json()['config']['ollama']
    r = client.put('/api/settings', json={'config': {'ollama': 'x'}})
    assert r.status_code == 400
    after = client.get('/api/settings').get_json()['config']['ollama']
    assert after == before
    assert isinstance(after, dict) and 'url' in after

def test_put_settings_preserves_unknown_config_section_on_disk(client):
    """A payload can carry a config section this running version doesn't know
    about -- written by a dev branch that ran once, a downgrade, or an installer
    ahead of the app (the client echoes back the WHOLE config on every Save). That
    section must never take the Save hostage: this used to 400 the entire PUT,
    discarding every valid setting the user just entered along with it (issue
    class already hit for 'bank_scoring', then 'variations'). The unknown section
    is dropped from the incoming partial -- neither validated nor applied -- and
    since it already exists on disk, save_config's deep-merge leaves it there
    untouched, preserved for whichever version wrote it."""
    from app import config
    config._config_path().write_text(
        json.dumps({'variations_from_the_future': {'flag': True}}),
        encoding='utf-8')
    config._cache = None
    r = client.put('/api/settings', json={'config': {
        'ollama': {'url': 'http://example.test:11434'},
        'variations_from_the_future': {'flag': False, 'brand_new_field': 'x'},
    }})
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body['config']['ollama']['url'] == 'http://example.test:11434'
    assert body['preserved_unknown_sections'] == ['variations_from_the_future']
    on_disk = json.loads(config._config_path().read_text(encoding='utf-8'))
    # Untouched semantically, not byte-for-byte: save_config() rewrites the whole
    # file (indent=2), so formatting can differ even though the value doesn't.
    assert on_disk['variations_from_the_future'] == {'flag': True}


def test_put_settings_ignores_unknown_section_absent_from_disk(client):
    """An unknown section the client sends but that never existed on disk has
    nothing to preserve -- it's just ignored (still reported, still no 400)."""
    r = client.put('/api/settings', json={'config': {
        'ollama': {'url': 'http://example.test:11434'},
        'totally_new_future_section': {'x': 1},
    }})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()['preserved_unknown_sections'] == ['totally_new_future_section']
    from app import config
    assert 'totally_new_future_section' not in config.load_config()


def test_put_settings_ignores_non_dict_unknown_section(client):
    """A non-dict value under an unknown key is dropped the same as any other
    unknown section -- only KNOWN sections must stay an object."""
    r = client.put('/api/settings', json={'config': {
        'ollama': {'url': 'http://example.test:11434'},
        'weird_unknown': 'not-a-dict',
    }})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()['preserved_unknown_sections'] == ['weird_unknown']


def test_put_settings_no_unknown_sections_field_when_none(client):
    """The diagnostic field stays absent (or empty) on an ordinary Save -- it
    only ever lists sections that were actually dropped."""
    r = client.put('/api/settings', json={'config': {'ollama': {'url': 'http://x'}}})
    assert r.status_code == 200
    assert r.get_json().get('preserved_unknown_sections', []) == []


def test_put_rejects_non_object_config(client):
    assert client.put('/api/settings', json={'config': 'oops'}).status_code == 400

def test_put_rejects_non_object_secrets(client):
    assert client.put('/api/settings', json={'secrets': ['x']}).status_code == 400

def test_put_settings_autocorrects_portable_base_dir(client, tmp_path):
    """Saving a base_dir that points at the portable WRAPPER
    (...\\ComfyUI_windows_portable) must be rewritten to the nested ...\\ComfyUI
    that actually holds main.py + models/ -- otherwise every model lister scans an
    empty wrapper\\models and reports 'No checkpoint found' even though ComfyUI runs."""
    wrapper = tmp_path / 'ComfyUI_windows_portable'
    inner = wrapper / 'ComfyUI'
    inner.mkdir(parents=True)
    (inner / 'main.py').touch()
    (inner / 'models').mkdir()
    r = client.put('/api/settings', json={'config': {'comfyui': {'base_dir': str(wrapper)}}})
    assert r.status_code == 200
    saved = r.get_json()['config']['comfyui']['base_dir']
    assert pathlib.Path(saved) == inner   # auto-corrected down into the real install

def test_put_settings_keeps_valid_base_dir_unchanged(client, tmp_path):
    """A base_dir already pointing straight at a real ComfyUI install is left as-is."""
    base = tmp_path / 'Comfy'
    base.mkdir()
    (base / 'main.py').touch()
    (base / 'models').mkdir()
    r = client.put('/api/settings', json={'config': {'comfyui': {'base_dir': str(base)}}})
    assert pathlib.Path(r.get_json()['config']['comfyui']['base_dir']) == base

def test_put_settings_full_config_save_keeps_autoprovisioned_watermark_python(client):
    """A full-config Save (the frontend sends the WHOLE config) must not blank out an
    interpreter the installer auto-provisioned out-of-band. watermark.python has no UI
    input, so the browser always echoes it back as "" on a fresh install; if that blank
    reached save_config it would deep-merge over the dedicated venv path and the feature
    would read "NOT installed" forever despite a perfect install (the reported bug)."""
    from app import config
    config.save_config({'watermark': {'python': '/data/envs/watermark/py.exe'}})
    # The stale full-config Save the browser sends after the install finished.
    r = client.put('/api/settings', json={'config': {
        'watermark': {'allow_crop': True, 'device': 'auto', 'python': ''},
        'masks': {'python': ''},
    }})
    assert r.status_code == 200
    assert r.get_json()['config']['watermark']['python'] == '/data/envs/watermark/py.exe'
    # The user's other watermark edits still land.
    assert r.get_json()['config']['watermark']['allow_crop'] is True


def test_put_settings_accepts_and_protects_bank_scoring_section(client):
    """The bank_scoring installer records bank_scoring.python out-of-band. The section
    must (a) be a known DEFAULTS section — a full-config Save echoing it back used to
    fail 400 "unknown config section 'bank_scoring'" (the reported bug) — and (b) enjoy
    the same blank-python clobber guard as the other auto-provisioned interpreters."""
    from app import config
    config.save_config({'bank_scoring': {'python': '/data/envs/bank_scoring/py.exe'}})
    r = client.put('/api/settings', json={'config': {
        'bank_scoring': {'python': ''},
    }})
    assert r.status_code == 200, r.get_json()
    assert (r.get_json()['config']['bank_scoring']['python']
            == '/data/envs/bank_scoring/py.exe')


def test_put_settings_protects_the_shot_detection_interpreter(client):
    """shot_detect.python is written by its installer and has no Settings input, so
    the frontend echoes it back as "" on every full Save. Without the guard, saving
    any unrelated setting silently un-installs shot detection: the probe falls back
    to the app's own Python, which has no torch, and the capability reads "missing"
    forever despite a perfect install — the exact defect the other auto-provisioned
    interpreters already carry a guard for."""
    from app import config
    config.save_config({'shot_detect': {'python': '/envs/scoring/py.exe'}})
    r = client.put('/api/settings', json={'config': {
        'shot_detect': {'python': '', 'device': 'cuda'},
    }})
    assert r.status_code == 200, r.get_json()
    saved = r.get_json()['config']['shot_detect']
    assert saved['python'] == '/envs/scoring/py.exe'
    assert saved['device'] == 'cuda'      # the rest of the section still saves


def test_stale_full_config_save_keeps_installed_bank_semantic_python(client):
    """A Settings form opened before SigLIP2 finishes must not undo the install.

    The stale form still carries ``bank_semantic.python: ''``. Setup records the
    managed runtime out-of-band while that tab is open; saving another semantic
    setting afterwards must preserve the new runtime instead of falling back to
    Score's borrowed interpreter.
    """
    from app import config

    borrowed = '/borrowed/gpu/python'
    managed = '/data/envs/bank_scoring/python'
    config.save_config({
        'bank_scoring': {'python': borrowed},
        'bank_semantic': {'python': ''},
    })
    stale = client.get('/api/settings').get_json()['config']
    assert stale['bank_semantic']['python'] == ''

    # SigLIP2 Install completes after the Settings tab loaded.
    config.save_config({'bank_semantic': {'python': managed}})
    stale['bank_semantic']['device'] = 'cpu'
    r = client.put('/api/settings', json={'config': stale})

    assert r.status_code == 200, r.get_json()
    saved = r.get_json()['config']
    assert saved['bank_semantic']['python'] == managed
    assert saved['bank_semantic']['device'] == 'cpu'
    assert saved['bank_scoring']['python'] == borrowed


def test_capabilities_endpoint(client):
    caps = client.get('/api/capabilities').get_json()
    assert 'engines' in caps and 'studio_visible' in caps

def _cloud_status(*, ok=True, namespace='lds-deliveries', error=None,
                  code=None, severity=None, warning=None):
    return {
        'ok': ok,
        'configured': True,
        'code': code or ('ready' if ok else 'invalid'),
        'severity': severity or ('success' if ok else 'error'),
        'namespace': namespace if ok else None,
        'settings_focus': 'HF_CLOUD_TOKEN',
        'warning': warning,
        'error': error,
    }


def test_put_settings_validates_cloud_token_candidate_before_saving(
        client, monkeypatch):
    import os
    from app.services import cloud_training

    candidate = 'hf_candidate_SECRET_MUST_NOT_BE_RETURNED'
    seen = []

    def validate(token, _api=None):
        seen.append(token)
        return _cloud_status()

    monkeypatch.setattr(
        cloud_training, 'full_transformer_token_status', validate)
    response = client.put('/api/settings', json={
        'secrets': {'HF_CLOUD_TOKEN': candidate},
    })

    assert response.status_code == 200
    payload = response.get_json()
    check = payload['secret_checks']['HF_CLOUD_TOKEN']
    assert seen == [candidate]
    assert check == {
        'ok': True,
        'detail': (
            'HF_CLOUD_TOKEN verified: krea/Krea-2-Raw is readable; '
            'delivery namespace: lds-deliveries.'),
        'code': 'ready',
        'configured': True,
        'severity': 'success',
        'settings_focus': 'HF_CLOUD_TOKEN',
        'namespace': 'lds-deliveries',
    }
    assert payload['secrets']['HF_CLOUD_TOKEN'] is True
    assert os.environ['HF_CLOUD_TOKEN'] == candidate
    assert candidate not in response.get_data(as_text=True)


def test_put_settings_accepts_global_write_token_with_warning(
        client, monkeypatch):
    from app import config
    from app.services import cloud_training

    candidate = 'hf_global_write_SECRET_MUST_NOT_BE_RETURNED'
    warning = (
        'This global write token is accepted, but it can modify every '
        'Hugging Face repository available to this account.')
    monkeypatch.setattr(
        cloud_training,
        'full_transformer_token_status',
        lambda token, _api=None: _cloud_status(
            namespace='tester', code='broad_access', severity='warning',
            warning=warning),
    )

    response = client.put('/api/settings', json={
        'secrets': {'HF_CLOUD_TOKEN': candidate},
    })

    assert response.status_code == 200
    payload = response.get_json()
    check = payload['secret_checks']['HF_CLOUD_TOKEN']
    assert check['ok'] is True
    assert check['code'] == 'broad_access'
    assert check['severity'] == 'warning'
    assert check['detail'] == warning
    assert check['warning'] == warning
    assert check['namespace'] == 'tester'
    assert config.secret('HF_CLOUD_TOKEN') == candidate
    assert candidate not in response.get_data(as_text=True)


def test_put_settings_rejects_invalid_cloud_token_atomically(
        client, monkeypatch):
    import os
    from app import config
    from app.services import cloud_training

    previous = 'hf_previous_valid_token'
    candidate = 'hf_bad_candidate_MUST_NOT_BE_RETURNED'
    config.set_secrets({'HF_CLOUD_TOKEN': previous})
    before_env = config.ENV_PATH.read_bytes()
    before_url = client.get('/api/settings').get_json()['config']['ollama']['url']
    seen = []

    def reject(token, _api=None):
        seen.append(token)
        return _cloud_status(
            ok=False,
            error=(
                'HF_CLOUD_TOKEN needs exact repo.content.read access to '
                'krea/Krea-2-Raw.'))

    monkeypatch.setattr(
        cloud_training, 'full_transformer_token_status', reject)
    response = client.put('/api/settings', json={
        'config': {'ollama': {'url': 'http://must-not-save.invalid'}},
        'secrets': {'HF_CLOUD_TOKEN': candidate},
    })

    assert response.status_code == 400
    payload = response.get_json()
    check = payload['secret_checks']['HF_CLOUD_TOKEN']
    assert seen == [candidate]
    assert check['ok'] is False
    assert check['code'] == 'invalid'
    assert 'repo.content.read' in check['detail']
    assert candidate not in response.get_data(as_text=True)
    assert client.get('/api/settings').get_json()['config']['ollama']['url'] == before_url
    assert config.secret('HF_CLOUD_TOKEN') == previous
    assert os.environ['HF_CLOUD_TOKEN'] == previous
    assert config.ENV_PATH.read_bytes() == before_env


def test_put_settings_skips_dense_validation_without_new_cloud_token(
        client, monkeypatch):
    from app.services import cloud_training

    calls = []
    monkeypatch.setattr(
        cloud_training,
        'full_transformer_token_status',
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert client.put('/api/settings', json={
        'config': {'ollama': {'url': 'http://127.0.0.1:11501'}},
    }).status_code == 200
    assert client.put('/api/settings', json={
        'secrets': {'HF_TOKEN': 'hf_classic_read_token'},
    }).status_code == 200
    assert calls == []


def test_hf_cloud_connection_target_reports_ready_and_invalid(
        client, monkeypatch):
    from app.services import cloud_training

    monkeypatch.setattr(
        cloud_training,
        'full_transformer_token_preflight',
        lambda: _cloud_status(namespace='private-lds'),
    )
    ready = client.post('/api/settings/test/hf_cloud')
    assert ready.status_code == 200
    assert ready.get_json()['ok'] is True
    assert ready.get_json()['namespace'] == 'private-lds'
    assert 'krea/Krea-2-Raw is readable' in ready.get_json()['detail']

    warning = 'Global write access is accepted with a warning.'
    monkeypatch.setattr(
        cloud_training,
        'full_transformer_token_preflight',
        lambda: _cloud_status(
            namespace='tester', code='broad_access', severity='warning',
            warning=warning),
    )
    broad = client.post('/api/settings/test/hf_cloud')
    assert broad.status_code == 200
    assert broad.get_json()['ok'] is True
    assert broad.get_json()['code'] == 'broad_access'
    assert broad.get_json()['severity'] == 'warning'
    assert broad.get_json()['detail'] == warning

    monkeypatch.setattr(
        cloud_training,
        'full_transformer_token_preflight',
        lambda: _cloud_status(
            ok=False, error=('HF_CLOUD_TOKEN requires repository write access; '
                             'read-only tokens cannot be used.')),
    )
    invalid = client.post('/api/settings/test/hf_cloud')
    assert invalid.status_code == 200
    assert invalid.get_json()['ok'] is False
    assert invalid.get_json()['detail'] == (
        'HF_CLOUD_TOKEN requires repository write access; read-only tokens cannot be used.')


def test_test_connection_unknown_target(client):
    assert client.post('/api/settings/test/nope').status_code == 404


def test_test_connection_ollama_checks_model_not_just_reachability(client, monkeypatch):
    """POST /api/settings/test/ollama runs the unified probe (reachable AND vision
    model pulled), so the Test button can't be green while the diagnostic says the
    model isn't pulled (issue #7)."""
    from app import capabilities
    monkeypatch.setattr(capabilities, 'probe_ollama', lambda: {'ok': True, 'detail': 'http://o'})
    # reachable but model absent -> honest failure through the SAME model probe
    monkeypatch.setattr(capabilities, 'probe_ollama_model',
                        lambda **k: {'ok': False, 'detail': 'model not pulled'})
    r = client.post('/api/settings/test/ollama').get_json()
    assert r['ok'] is False and 'not pulled' in r['detail']
    # reachable + present -> green
    monkeypatch.setattr(capabilities, 'probe_ollama_model',
                        lambda **k: {'ok': True, 'detail': 'ready'})
    assert client.post('/api/settings/test/ollama').get_json()['ok'] is True


def test_test_connection_lmstudio_checks_a_usable_model_not_just_reachability(
        client, monkeypatch):
    """The LM Studio Test button owes exactly what Ollama's owes.

    A server that answers on 1234 with nothing loaded is the COMMONEST state after
    an LM Studio install -- JIT loading is off by default -- so a Test that went
    green on reachability alone would be green on the one machine state where
    captioning cannot run at all.
    """
    from app import capabilities
    monkeypatch.setattr(capabilities, 'probe_lmstudio',
                        lambda: {'ok': True, 'detail': 'http://lm'})
    monkeypatch.setattr(capabilities, 'probe_lmstudio_model',
                        lambda **k: {'ok': False, 'detail': 'no model loaded'})
    r = client.post('/api/settings/test/lmstudio').get_json()
    assert r['ok'] is False and 'no model loaded' in r['detail']

    monkeypatch.setattr(capabilities, 'probe_lmstudio_model',
                        lambda **k: {'ok': True, 'detail': 'qwen loaded'})
    assert client.post('/api/settings/test/lmstudio').get_json()['ok'] is True


def test_an_unreachable_lmstudio_never_reaches_the_model_probe(client, monkeypatch):
    """...and it says the server is down rather than "no model loaded".

    Two different remedies -- press Start Server, or load a model -- so collapsing
    them would send the user to the wrong one.
    """
    from app import capabilities
    monkeypatch.setattr(capabilities, 'probe_lmstudio',
                        lambda: {'ok': False, 'detail': 'not answering on 1234'})

    def never(**k):
        raise AssertionError('the model probe ran against a server that is not there')

    monkeypatch.setattr(capabilities, 'probe_lmstudio_model', never)
    r = client.post('/api/settings/test/lmstudio').get_json()
    assert r['ok'] is False and 'not answering' in r['detail']


# --- CSRF cookie freshness (long-lived SPA session) ---------------------------
# Flask-WTF time-limits the CSRF token (WTF_CSRF_TIME_LIMIT). The cookie used to
# be planted ONLY on GET /, so a tab left open past that limit kept echoing a
# stale token and every Save/Test POST failed with a cryptic HTML 400 until a hard
# refresh. An after_request hook now re-plants a fresh token on / and every /api
# response — including the CSRF-rejection 400 itself, so the client's one-shot
# retry can recover without a reload.

def _csrf_cookies(resp):
    """The Set-Cookie header(s) that (re)plant csrf_token, if any."""
    return [c for c in resp.headers.getlist('Set-Cookie') if c.startswith('csrf_token=')]


def test_after_request_plants_fresh_csrf_cookie_on_api(client):
    """Any /api response re-plants the csrf_token cookie, JS-readable (not
    HttpOnly, since the SPA must echo it back in the X-CSRFToken header)."""
    r = client.get('/api/health')
    assert r.status_code == 200
    planted = _csrf_cookies(r)
    assert planted, 'csrf_token cookie must be (re)planted on /api responses'
    assert 'HttpOnly' not in planted[0]
    assert 'SameSite=Lax' in planted[0]


def test_static_assets_do_not_replant_csrf_cookie(client):
    """The hook stays quiet on static assets (pure noise) — only / and /api."""
    r = client.get('/assets/does-not-exist.js')
    assert not _csrf_cookies(r)


@pytest.fixture()
def csrf_client(tmp_path, monkeypatch):
    """A client on an app with CSRF actually enforced (the default fixture turns
    it off). Mirrors conftest's app fixture env/cache isolation."""
    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    import app.config as _cfg
    monkeypatch.setattr(_cfg, 'ENV_PATH', tmp_path / '.env')
    monkeypatch.setattr(_cfg, '_cache', None)
    from app import create_app
    application = create_app({'TESTING': True, 'WTF_CSRF_ENABLED': True,
                              'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:'})
    return application.test_client()


def test_csrf_rejection_carries_fresh_cookie_and_allows_retry(csrf_client):
    """A mutating POST with no/stale token is still rejected (400) — but that very
    rejection response re-plants a fresh csrf_token cookie, so the client can read
    it and replay the request once and succeed, with no hard refresh."""
    r = csrf_client.put('/api/settings', json={'config': {'ollama': {'url': 'http://x'}}})
    assert r.status_code == 400                      # token missing -> Flask-WTF rejects
    planted = _csrf_cookies(r)
    assert planted, 'the CSRF-rejection response must still refresh the token cookie'
    token = planted[0].split('csrf_token=', 1)[1].split(';', 1)[0]
    # Replay WITH the fresh token in the header (the session cookie rode along on
    # the test client's jar) -> accepted, no longer a 400.
    r2 = csrf_client.put('/api/settings', json={'config': {'ollama': {'url': 'http://x'}}},
                         headers={'X-CSRFToken': token})
    assert r2.status_code == 200
    assert r2.get_json()['config']['ollama']['url'] == 'http://x'


@pytest.fixture()
def _reset_update_cache():
    from app.routes import settings as sroutes
    sroutes._update_cache.update(ts=0.0, data=None)
    sroutes._git_check_cache.update(ts=0.0, data=None)
    yield
    sroutes._update_cache.update(ts=0.0, data=None)
    sroutes._git_check_cache.update(ts=0.0, data=None)


class _FakeResp:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self._body = body or {}
    def json(self):
        return self._body


def test_update_check_detects_newer_release(client, monkeypatch, _reset_update_cache):
    import requests
    monkeypatch.setattr(requests, 'get', lambda *a, **k: _FakeResp(200, {
        'tag_name': 'v9999.12.31', 'html_url': 'https://github.com/x/releases/tag/v9999.12.31'}))
    d = client.get('/api/update/check').get_json()
    assert d['update_available'] is True and d['latest'] == '9999.12.31'
    assert d['url'].endswith('v9999.12.31')


def test_update_check_docker_reports_manual_rebuild_even_with_zip(
        client, monkeypatch, _reset_update_cache):
    import requests
    from app.services import updater
    monkeypatch.setenv('LDS_RUNTIME', 'docker-gpu')
    monkeypatch.setattr(updater, 'is_git_checkout', lambda root=None: True)
    monkeypatch.setattr(
        updater, 'git_update_status',
        lambda root=None: (_ for _ in ()).throw(
            AssertionError('Docker checks must not fetch through the git updater')),
    )
    monkeypatch.setattr(requests, 'get', lambda *a, **k: _FakeResp(200, {
        'tag_name': 'v9999.12.31',
        'assets': [{'name': 'LoRA-Dataset-Studio-windows.zip',
                    'browser_download_url': 'https://x/win'}],
    }))

    d = client.get('/api/update/check?force=1').get_json()

    assert d['update_available'] is True
    assert d['install_mode'] == 'docker' and d['can_apply'] is False
    assert d['manual'] is True
    assert d['instructions'] == [
        'git pull',
        'docker compose -f docker-compose.gpu.yml up -d --build',
    ]


def test_update_check_pinokio_keeps_the_git_answer_but_drops_the_button(
        client, monkeypatch, _reset_update_cache):
    """Unlike Docker, a Pinokio install IS a git checkout worth measuring: its
    Update tab pulls exactly those commits, so "3 commits behind" stays true and
    useful. Only the in-app apply goes away."""
    from app.services import updater
    monkeypatch.setenv('LDS_RUNTIME', 'pinokio')
    monkeypatch.setattr(updater, 'is_git_checkout', lambda root=None: True)
    monkeypatch.setattr(updater, 'git_update_status', lambda root=None: {
        'ok': True, 'is_git': True, 'update_available': True, 'behind': 3,
        'current_sha': 'aaaaaaa', 'remote_sha': 'bbbbbbb',
    })

    d = client.get('/api/update/check?force=1').get_json()

    assert d['behind'] == 3 and d['is_git'] is True
    assert d['install_mode'] == 'pinokio' and d['can_apply'] is False
    assert d['manual'] is True
    assert d['instructions'] == [
        'Stop the app in Pinokio',
        'Click the Update tab',
        'Click Start',
    ]


def test_update_check_pinokio_never_advertises_a_zip_apply(
        client, monkeypatch, _reset_update_cache):
    """The passive (non-git-aware) path must not hand back can_apply just
    because the latest release ships a ZIP asset."""
    import requests
    from app.services import updater
    monkeypatch.setenv('LDS_RUNTIME', 'pinokio')
    monkeypatch.setattr(updater, 'is_git_checkout', lambda root=None: False)
    monkeypatch.setattr(requests, 'get', lambda *a, **k: _FakeResp(200, {
        'tag_name': 'v9999.12.31',
        'assets': [{'name': 'LoRA-Dataset-Studio-windows.zip',
                    'browser_download_url': 'https://x/win'}],
    }))

    d = client.get('/api/update/check').get_json()

    assert d['update_available'] is True
    assert d['install_mode'] == 'pinokio' and d['can_apply'] is False


def test_update_apply_pinokio_refuses_before_touching_git(client, monkeypatch):
    from app.services import updater
    monkeypatch.setenv('LDS_RUNTIME', 'pinokio')
    forbidden = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('Pinokio apply must refuse before an updater is selected'))
    monkeypatch.setattr(updater, 'is_git_checkout', forbidden)
    monkeypatch.setattr(updater, 'apply_update', forbidden)
    monkeypatch.setattr(updater, 'start_zip_update', forbidden)
    monkeypatch.setattr(updater, 'schedule_restart', forbidden)

    response = client.post('/api/update/apply')
    body = response.get_json()

    assert response.status_code == 200
    assert body['ok'] is False and body['manual'] is True
    assert body['install_mode'] == 'pinokio' and body['can_apply'] is False
    assert body['instructions'] == [
        'Stop the app in Pinokio',
        'Click the Update tab',
        'Click Start',
    ]


def test_update_check_same_version_and_cache(client, monkeypatch, _reset_update_cache):
    import requests
    from app.version import APP_VERSION
    calls = []
    monkeypatch.setattr(requests, 'get',
                        lambda *a, **k: calls.append(1) or _FakeResp(200, {'tag_name': f'v{APP_VERSION}'}))
    d = client.get('/api/update/check').get_json()
    assert d['update_available'] is False and d['latest'] == APP_VERSION
    client.get('/api/update/check')          # second call served from the 6h cache
    assert len(calls) == 1


def test_update_check_auto_fetches_git_then_serves_cache(client, monkeypatch, _reset_update_cache):
    """auto=1 (nav badge): the git-aware check RUNS (unlike the bare passive
    path) but is served from a TTL cache — SPA loads cost one fetch per 6 h."""
    from app.services import updater
    calls = []
    monkeypatch.setattr(updater, 'is_git_checkout', lambda root=None: True)
    monkeypatch.setattr(updater, 'git_update_status',
                        lambda root=None: calls.append(1) or {
                            'ok': True, 'is_git': True, 'update_available': True,
                            'behind': 2, 'current': '1.0'})
    d = client.get('/api/update/check?auto=1').get_json()
    assert d['update_available'] is True and d['behind'] == 2
    client.get('/api/update/check?auto=1')       # second auto call -> cache
    assert len(calls) == 1
    # a manual force check is always fresh AND refreshes the cache
    client.get('/api/update/check?force=1')
    assert len(calls) == 2


def test_update_check_bare_passive_never_fetches_git(client, monkeypatch, _reset_update_cache):
    """The bare passive path (no force, no auto) must not run the git check."""
    import requests
    from app.services import updater
    monkeypatch.setattr(updater, 'is_git_checkout', lambda root=None: True)
    monkeypatch.setattr(updater, 'git_update_status',
                        lambda root=None: (_ for _ in ()).throw(
                            AssertionError('git check must not run')))
    monkeypatch.setattr(requests, 'get', lambda *a, **k: _FakeResp(404))
    d = client.get('/api/update/check').get_json()
    assert d['ok'] is True


def test_update_check_degrades_when_feed_unreachable(client, monkeypatch, _reset_update_cache):
    import requests
    def boom(*a, **k):
        raise requests.ConnectionError('offline')
    monkeypatch.setattr(requests, 'get', boom)
    d = client.get('/api/update/check').get_json()
    assert d['ok'] is True and d['update_available'] is False
    assert 'unreachable' in d['reason']


def test_update_check_private_repo_404(client, monkeypatch, _reset_update_cache):
    import requests
    monkeypatch.setattr(requests, 'get', lambda *a, **k: _FakeResp(404))
    d = client.get('/api/update/check').get_json()
    assert d['update_available'] is False and '404' in d['reason']


def test_logs_tail_reads_app_log(client, tmp_path, monkeypatch):
    import os
    data_dir = os.environ['LDS_DATA_DIR']    # tmp dir set by the app fixture
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, 'app.log'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(f'line {i}' for i in range(500)) + '\n')
    d = client.get('/api/logs/tail?n=100').get_json()
    assert d['ok'] is True and d['file'] == 'app.log'
    assert len(d['lines']) == 100 and d['lines'][-1] == 'line 499'


def test_logs_tail_empty_when_no_log(client):
    d = client.get('/api/logs/tail').get_json()
    assert d == {'ok': True, 'file': None, 'lines': []}


def test_chatgpt_oauth_routes(client, monkeypatch):
    from unittest.mock import patch
    from app.services import chatgpt_oauth
    with patch.object(chatgpt_oauth, 'login_start',
                      return_value={'ok': True, 'verification_url': 'https://x/device',
                                    'user_code': 'AB-12'}):
        r = client.post('/api/settings/chatgpt-oauth/start')
        assert r.status_code == 200 and r.get_json()['user_code'] == 'AB-12'
    with patch.object(chatgpt_oauth, 'login_start',
                      return_value={'ok': False, 'detail': 'network error'}):
        assert client.post('/api/settings/chatgpt-oauth/start').status_code == 502
    with patch.object(chatgpt_oauth, 'login_poll', return_value={'status': 'pending',
                                                                 'detail': None}):
        r = client.get('/api/settings/chatgpt-oauth/poll')
        assert r.status_code == 200 and r.get_json()['status'] == 'pending'
    with patch.object(chatgpt_oauth, 'import_codex_cli',
                      return_value={'ok': False, 'detail': 'no session'}):
        assert client.post('/api/settings/chatgpt-oauth/import-codex').status_code == 404
    with patch.object(chatgpt_oauth, 'import_codex_cli',
                      return_value={'ok': True, 'detail': 'imported'}):
        assert client.post('/api/settings/chatgpt-oauth/import-codex').status_code == 200
    r = client.post('/api/settings/chatgpt-oauth/logout')
    assert r.status_code == 200 and r.get_json()['ok'] is True


def test_put_settings_saves_chatgpt_auth_mode(client):
    r = client.put('/api/settings', json={'config': {'engines': {'chatgpt_auth': 'subscription'}}})
    assert r.status_code == 200
    assert r.get_json()['config']['engines']['chatgpt_auth'] == 'subscription'


# --- Server settings (host/port/LAN/access token) ----------------------------
def test_settings_payload_includes_server_defaults(client):
    cfg = client.get('/api/settings').get_json()['config']
    assert cfg['server'] == {'host': '127.0.0.1', 'port': 5050,
                             'require_token': False, 'access_token': ''}


def test_put_settings_saves_require_token(client):
    r = client.put('/api/settings', json={'config': {'server': {'require_token': True}}})
    assert r.status_code == 200
    assert r.get_json()['config']['server']['require_token'] is True


def test_settings_restart_pins_saved_host_port_to_env(client, monkeypatch):
    """The launcher exports LDS_PORT, which otherwise wins over config forever.
    The restart route must stamp the SAVED host/port into env so the relaunch
    actually binds where the user asked (else the port field looks broken)."""
    import os
    from app.services import updater
    seen = {}
    def fake_schedule(*args, **kwargs):
        seen.update(kwargs)
        os.environ.update(kwargs.get('environment_updates') or {})
        return True
    monkeypatch.setattr(updater, 'schedule_restart', fake_schedule)
    monkeypatch.delenv('LDS_PORT', raising=False)
    monkeypatch.delenv('LDS_HOST', raising=False)
    client.put('/api/settings', json={'config': {'server': {'host': '0.0.0.0', 'port': 5123}}})
    client.post('/api/settings/restart')
    assert os.environ['LDS_HOST'] == '0.0.0.0'
    assert os.environ['LDS_PORT'] == '5123'
    assert seen['block_during_update'] is True


def test_settings_restart_preserves_supervisor_managed_bind(client, monkeypatch):
    """Inside Docker the container bind is fixed by its environment.  Saving a
    desktop host/port must not turn the restarted backend into a loopback-only
    process that the published port cannot reach."""
    import os
    from app.services import updater
    monkeypatch.setenv('LDS_BIND_MANAGED', '1')
    monkeypatch.setenv('LDS_HOST', '0.0.0.0')
    monkeypatch.setenv('LDS_PORT', '5050')
    seen = {}
    monkeypatch.setattr(updater, 'schedule_restart',
                        lambda *a, **k: seen.update(k) or True)
    client.put('/api/settings', json={'config': {'server': {
        'host': '127.0.0.1', 'port': 5999,
    }}})

    response = client.post('/api/settings/restart')

    assert response.status_code == 200
    assert os.environ['LDS_HOST'] == '0.0.0.0'
    assert os.environ['LDS_PORT'] == '5050'
    assert seen['block_during_update'] is True
    assert seen['environment_updates'] is None


def test_put_settings_saves_server_lan_and_port(client):
    r = client.put('/api/settings', json={'config': {'server': {'host': '0.0.0.0', 'port': 5001}}})
    assert r.status_code == 200
    assert r.get_json()['config']['server']['host'] == '0.0.0.0'
    assert r.get_json()['config']['server']['port'] == 5001


def test_runtime_reflects_what_run_py_stamped_on_boot(client, app):
    """Before run.py's __main__ block runs (dev/test boots go through create_app()
    directly), nothing has been bound yet -- the card must show that as 'unknown',
    never fabricate a value that looks like a real running bind."""
    rt = client.get('/api/settings').get_json()['runtime']
    assert (rt['host'], rt['port']) == (None, None)   # lan_ip is orthogonal (see its own test)
    app.config['LDS_BOUND_HOST'] = '0.0.0.0'
    app.config['LDS_BOUND_PORT'] = 5000
    rt = client.get('/api/settings').get_json()['runtime']
    assert (rt['host'], rt['port']) == ('0.0.0.0', 5000)


def test_runtime_exposes_whether_the_bind_is_managed(client, monkeypatch):
    monkeypatch.delenv('LDS_BIND_MANAGED', raising=False)
    assert client.get('/api/settings').get_json()['runtime']['bind_managed'] is False
    monkeypatch.setenv('LDS_BIND_MANAGED', '1')
    assert client.get('/api/settings').get_json()['runtime']['bind_managed'] is True


def test_runtime_can_differ_from_saved_config_until_restart(client, app):
    """Saving a new port must NOT retroactively change what's reported as running --
    that would lie about a bind change that hasn't taken effect yet."""
    app.config['LDS_BOUND_HOST'] = '127.0.0.1'
    app.config['LDS_BOUND_PORT'] = 5000
    client.put('/api/settings', json={'config': {'server': {'host': '0.0.0.0', 'port': 5001}}})
    data = client.get('/api/settings').get_json()
    assert data['config']['server']['port'] == 5001
    assert (data['runtime']['host'], data['runtime']['port']) == ('127.0.0.1', 5000)


def test_settings_runtime_includes_lan_ip(client):
    """The Server card builds a real copyable http://<ip>:port/ URL from this;
    it's the machine's primary LAN IPv4, or None (UI falls back to a placeholder)
    when offline / loopback-only. It must never be a loopback address."""
    runtime = client.get('/api/settings').get_json()['runtime']
    assert 'lan_ip' in runtime
    ip = runtime['lan_ip']
    assert ip is None or (isinstance(ip, str) and not ip.startswith('127.'))


def test_settings_runtime_includes_tailscale_ip(client):
    """The Server card offers a Tailscale URL beside the LAN one as the phone's
    off-perimeter path. It's a tailnet address (100.64.0.0/10) or None when the
    tunnel is down — never a bare LAN IP masquerading as a tailnet address."""
    from app.routes import settings as sroutes
    runtime = client.get('/api/settings').get_json()['runtime']
    assert 'tailscale_ip' in runtime
    ip = runtime['tailscale_ip']
    assert ip is None or sroutes._is_cgnat(ip)


def test_is_cgnat_classifies_tailscale_range():
    """Only 100.64.0.0/10 (Tailscale's CGNAT block) counts — a real LAN IP or a
    100.x address outside the block must not be mistaken for a tailnet address."""
    from app.routes import settings as sroutes
    assert sroutes._is_cgnat('100.100.100.100') is True   # in-block (Tailscale's own)
    assert sroutes._is_cgnat('100.64.0.1') is True        # lower edge
    assert sroutes._is_cgnat('100.127.255.254') is True   # upper edge
    assert sroutes._is_cgnat('100.63.255.255') is False   # just below the block
    assert sroutes._is_cgnat('100.128.0.1') is False      # just above the block
    assert sroutes._is_cgnat('192.0.2.10') is False       # outside the block
    assert sroutes._is_cgnat('') is False
    assert sroutes._is_cgnat(None) is False


def test_settings_restart_triggers_schedule_restart(client, monkeypatch):
    from app.services import updater
    called = []
    monkeypatch.setattr(updater, 'schedule_restart', lambda *a, **k: called.append(k))
    r = client.post('/api/settings/restart')
    assert r.status_code == 200
    assert r.get_json() == {'ok': True, 'restarting': True}
    assert len(called) == 1
    assert called[0]['block_during_update'] is True
    assert called[0]['environment_updates']['LDS_HOST'] == '127.0.0.1'
    assert called[0]['environment_updates']['LDS_PORT'] == '5050'


def test_update_apply_defers_changed_requirements_to_restart(client, monkeypatch):
    from app.services import updater
    monkeypatch.setattr(
        updater, 'apply_update',
        lambda: {'ok': True, 'changed': True, 'deps_changed': True},
    )
    calls = []
    monkeypatch.setattr(updater, 'schedule_restart',
                        lambda *a, **k: calls.append((a, k)))

    response = client.post('/api/update/apply')

    assert response.status_code == 200
    assert response.get_json()['restarting'] is True
    assert calls == [((), {'install_requirements': True})]


def test_update_apply_docker_refuses_before_any_pull_or_download(client, monkeypatch):
    from app.services import updater
    monkeypatch.setenv('LDS_RUNTIME', 'docker-gpu')
    forbidden = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('Docker apply must refuse before an updater is selected'))
    monkeypatch.setattr(updater, 'is_git_checkout', forbidden)
    monkeypatch.setattr(updater, 'apply_update', forbidden)
    monkeypatch.setattr(updater, 'start_zip_update', forbidden)

    response = client.post('/api/update/apply')
    body = response.get_json()

    assert response.status_code == 200
    assert body['ok'] is False and body['manual'] is True
    assert body['install_mode'] == 'docker' and body['can_apply'] is False
    assert body['instructions'] == [
        'git pull',
        'docker compose -f docker-compose.gpu.yml up -d --build',
    ]


def test_settings_restart_is_refused_during_active_update(client, monkeypatch):
    import os
    from app.services import updater
    monkeypatch.setenv('LDS_HOST', 'keep-host')
    monkeypatch.setenv('LDS_PORT', '4321')
    def reject(*args, **kwargs):
        assert kwargs['block_during_update'] is True
        # The atomic scheduler rejects before applying these proposed values.
        raise updater.UpdateInProgressError('restarting')
    monkeypatch.setattr(updater, 'schedule_restart', reject)

    response = client.post('/api/settings/restart')

    assert response.status_code == 409
    assert response.get_json()['phase'] == 'restarting'
    assert os.environ['LDS_HOST'] == 'keep-host' and os.environ['LDS_PORT'] == '4321'


def test_update_check_reports_can_apply_for_zip_release(client, monkeypatch, _reset_update_cache):
    """A release with a ZIP asset -> can_apply True, so a packaged install can
    update in-app instead of only linking to the releases page."""
    import requests
    monkeypatch.setattr(requests, 'get', lambda *a, **k: _FakeResp(200, {
        'tag_name': 'v9999.12.31',
        'assets': [{'name': 'LoRA-Dataset-Studio-windows.zip',
                    'browser_download_url': 'https://x/win'}]}))
    d = client.get('/api/update/check').get_json()
    assert d['update_available'] is True and d['can_apply'] is True


def test_update_apply_zip_install_starts_async_release_update(client, monkeypatch):
    """A NON-git install routes /update/apply to the async release-ZIP updater
    (never the git pull path) and returns an async handle the client then polls."""
    from app.services import updater
    monkeypatch.setattr(updater, 'is_git_checkout', lambda root=None: False)
    called = {}
    def fake_start():
        called['zip'] = True
        return {'ok': True, 'async': True, 'from': '2026.07.16.1',
                'to': '9999.01.01', 'total': 42_000_000}
    monkeypatch.setattr(updater, 'start_zip_update', fake_start)
    monkeypatch.setattr(updater, 'apply_update',
                        lambda: (_ for _ in ()).throw(AssertionError('git path must not run')))

    r = client.post('/api/update/apply')

    body = r.get_json()
    assert r.status_code == 200 and called.get('zip') is True
    assert body['async'] is True and body['to'] == '9999.01.01' and body['total'] == 42_000_000


def test_update_progress_endpoint_returns_state(client, monkeypatch):
    from app.services import updater
    monkeypatch.setattr(updater, 'zip_update_progress',
                        lambda: {'phase': 'downloading', 'downloaded': 10, 'total': 100})
    d = client.get('/api/update/progress').get_json()
    assert d['phase'] == 'downloading' and d['downloaded'] == 10 and d['total'] == 100


def test_settings_offers_an_engine_added_by_an_update(client, tmp_path, monkeypatch):
    """End to end over HTTP: someone who saved their settings back when only
    three engines existed opens Settings after updating and is OFFERED the new
    one — the checkbox list is rendered from this payload."""
    import json
    import app.config as _cfg
    (tmp_path / 'config.json').write_text(
        json.dumps({'engines': {'enabled': ['nanobanana', 'chatgpt', 'klein']}}),
        encoding='utf-8')
    monkeypatch.setattr(_cfg, '_cache', None)
    enabled = client.get('/api/settings').get_json()['config']['engines']['enabled']
    assert 'openrouter' in enabled
    assert enabled[:3] == ['nanobanana', 'chatgpt', 'klein']


def test_unchecking_an_engine_over_the_api_sticks(client, monkeypatch):
    """The counter-test over HTTP: the SPA saves the full config it was shown,
    minus the engine the user just unchecked. It must not reappear on reload."""
    import app.config as _cfg
    monkeypatch.setattr(_cfg, '_cache', None)
    shown = client.get('/api/settings').get_json()['config']['engines']
    kept = [e for e in shown['enabled'] if e != 'openrouter']
    r = client.put('/api/settings', json={'config': {'engines': dict(shown, enabled=kept)}})
    assert r.status_code == 200
    monkeypatch.setattr(_cfg, '_cache', None)
    assert client.get('/api/settings').get_json()['config']['engines']['enabled'] == kept


# --- config_defaults: the scalar counterpart of identity_prompt_defaults ------
# The Settings UI offers a per-field "Reset to default" on numbers, paths and
# selects. The value it resets to MUST come from the server: a literal typed into
# the JSX would go stale the next time a default moves in config.DEFAULTS, and the
# button would then restore a number that is no longer the default without saying
# so. These tests pin that the payload is DERIVED, not a second copy.

def test_get_settings_exposes_the_shipped_config_defaults(client):
    import app.config as _cfg
    data = client.get('/api/settings').get_json()
    d = data['config_defaults']
    # every section of DEFAULTS is offered, with its shipped value
    assert set(d) == set(_cfg.DEFAULTS)
    assert d['klein']['improve_steps'] == _cfg.DEFAULTS['klein']['improve_steps']
    assert d['krea']['grounding_px'] == _cfg.DEFAULTS['krea']['grounding_px']
    # blank-means-auto keys keep their EMPTY default: resetting one must write ''
    # back, not a made-up value that would freeze the field (see the frontend's
    # resetToDefault.test.js for the UI half of this contract).
    assert d['engines']['nanobanana_model'] == ''
    assert d['krea']['base_model'] == ''


def test_config_defaults_follows_DEFAULTS_and_is_not_a_frozen_copy(client, monkeypatch):
    """The regression this whole payload exists to prevent: move a default, and
    what the UI would reset to moves with it. A hand-maintained copy anywhere
    between DEFAULTS and the button fails here."""
    import app.config as _cfg
    monkeypatch.setitem(_cfg.DEFAULTS['klein'], 'improve_steps', 43)
    assert client.get('/api/settings').get_json()['config_defaults']['klein']['improve_steps'] == 43


def test_config_defaults_is_a_copy_the_caller_cannot_corrupt(client):
    """Serving the live dict would let any mutation downstream rewrite the app's
    defaults for the rest of the process."""
    import app.config as _cfg
    d = _cfg.defaults()
    d['klein']['improve_steps'] = 999
    assert _cfg.DEFAULTS['klein']['improve_steps'] != 999


def test_config_defaults_carries_no_secret(client, monkeypatch):
    """It is a public payload: DEFAULTS holds knobs, never a key."""
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-secret')
    data = client.get('/api/settings').get_json()
    assert 'sk-secret' not in str(data['config_defaults'])
    # the generated access token lives in the user's config, never in the defaults
    assert data['config_defaults']['server']['access_token'] == ''


def test_runtime_reports_public_bind(client, monkeypatch):
    """The Server card needs to know the gate is forced, or a user toggles
    require_token off, sees it off, and the gate stays on."""
    monkeypatch.setenv('LDS_PUBLIC', '1')
    assert client.get('/api/settings').get_json()['runtime']['public'] is True


def test_runtime_public_defaults_false(client, monkeypatch):
    monkeypatch.delenv('LDS_PUBLIC', raising=False)
    assert client.get('/api/settings').get_json()['runtime']['public'] is False
