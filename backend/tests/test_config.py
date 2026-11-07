import json, os
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

def _fresh(monkeypatch, tmp_path):
    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    import app.config as config
    # Reset test state without replacing functions already imported by the SDK.
    monkeypatch.setattr(config, 'ENV_PATH', tmp_path / '.env')
    monkeypatch.setattr(config, '_cache', None)
    return config

def test_defaults_when_no_file(tmp_path, monkeypatch):
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('server.port') == 5050
    assert config.load_config()['engines']['default'] == 'klein'
    assert config.load_config()['engines']['enabled'] == ['klein', 'krea']
    assert config.is_configured() is False

def test_save_and_reload_deep_merge(tmp_path, monkeypatch):
    config = _fresh(monkeypatch, tmp_path)
    config.save_config({'comfyui': {'api_url': 'http://10.0.0.2:8188'}})
    assert config.get('comfyui.api_url') == 'http://10.0.0.2:8188'
    assert config.get('server.port') == 5050          # untouched default survives
    assert config.is_configured() is True
    on_disk = json.loads((tmp_path / 'config.json').read_text(encoding='utf-8'))
    assert on_disk['comfyui']['api_url'] == 'http://10.0.0.2:8188'

def test_comfyui_dir_derivation(tmp_path, monkeypatch):
    config = _fresh(monkeypatch, tmp_path)
    assert config.comfyui_dir('loras') is None        # unconfigured
    base = tmp_path / 'Comfy'
    (base / 'models' / 'loras').mkdir(parents=True)
    config.save_config({'comfyui': {'base_dir': str(base)}})
    assert config.comfyui_dir('loras') == base / 'models' / 'loras'
    assert config.comfyui_dir('output') == base / 'output'

def test_saved_secret_beats_a_stale_inherited_copy(tmp_path, monkeypatch):
    """start.bat's supervisor relaunches the server with the environment it
    started with. A secret replaced in Settings since then must come from .env,
    not from that stale copy; other variables keep the environment's precedence."""
    config = _fresh(monkeypatch, tmp_path)
    (tmp_path / '.env').write_text("HF_CLOUD_TOKEN='hf_new'\nLDS_PORT='6000'\n",
                                   encoding='utf-8')
    env = {'HF_CLOUD_TOKEN': 'hf_stale', 'LDS_PORT': '5050', 'VAST_API_KEY': 'kept'}
    assert config._prefer_env_file_secrets(tmp_path / '.env', env) == ['HF_CLOUD_TOKEN']
    assert env == {'HF_CLOUD_TOKEN': 'hf_new', 'LDS_PORT': '5050', 'VAST_API_KEY': 'kept'}
    assert config._prefer_env_file_secrets(tmp_path / 'missing.env', env) == []


def test_secrets_roundtrip(tmp_path, monkeypatch):
    config = _fresh(monkeypatch, tmp_path)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    assert config.secret('OPENAI_API_KEY') is None
    config.set_secrets({'OPENAI_API_KEY': 'sk-test-123'})
    assert config.secret('OPENAI_API_KEY') == 'sk-test-123'
    env_text = (config.ENV_PATH).read_text(encoding='utf-8')
    assert 'sk-test-123' in env_text


def test_secret_printable_characters_roundtrip_exactly(tmp_path, monkeypatch):
    config = _fresh(monkeypatch, tmp_path)
    value = "sk printable # = ' \\\\ ü"

    config.set_secrets({'OPENAI_API_KEY': value})

    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    from dotenv import load_dotenv
    load_dotenv(config.ENV_PATH, override=True)
    assert config.secret('OPENAI_API_KEY') == value


@pytest.mark.parametrize('separator', UNSAFE_SECRET_CHARS)
def test_set_secrets_rejects_controls_without_mutating_file_or_environment(
        tmp_path, monkeypatch, separator):
    config = _fresh(monkeypatch, tmp_path)
    config.ENV_PATH.write_bytes(b"GEMINI_API_KEY='old-value'\n")
    before = config.ENV_PATH.read_bytes()
    monkeypatch.setenv('OPENAI_API_KEY', 'runtime-before')
    monkeypatch.delenv('FLASK_DEBUG', raising=False)

    with pytest.raises(ValueError, match='single line'):
        config.set_secrets({
            'OPENAI_API_KEY': f'new-value{separator}FLASK_DEBUG=1',
        })

    assert config.ENV_PATH.read_bytes() == before
    assert config.secret('OPENAI_API_KEY') == 'runtime-before'
    assert 'FLASK_DEBUG' not in os.environ


@pytest.mark.parametrize('separator', [
    pytest.param('\x0b', id='vertical-tab'),
    pytest.param('\u2028', id='unicode-line-separator'),
])
def test_set_secrets_refuses_to_normalize_poisoned_existing_env(
        tmp_path, monkeypatch, separator):
    config = _fresh(monkeypatch, tmp_path)
    poisoned = f'OPENAI_API_KEY=old{separator}FLASK_DEBUG=1\n'.encode('utf-8')
    config.ENV_PATH.write_bytes(poisoned)
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('FLASK_DEBUG', raising=False)

    with pytest.raises(ValueError, match='existing .env'):
        config.set_secrets({'GEMINI_API_KEY': 'safe-value'})

    assert config.ENV_PATH.read_bytes() == poisoned
    assert 'GEMINI_API_KEY' not in os.environ
    assert 'FLASK_DEBUG' not in os.environ

def test_secret_strips_trailing_whitespace(tmp_path, monkeypatch):
    """A pasted key with a trailing newline/space must not corrupt the Bearer header."""
    config = _fresh(monkeypatch, tmp_path)
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test-123\n')
    assert config.secret('OPENAI_API_KEY') == 'sk-test-123'
    monkeypatch.setenv('OPENAI_API_KEY', '  sk-test-456  ')
    assert config.secret('OPENAI_API_KEY') == 'sk-test-456'

def test_local_user_constant(tmp_path, monkeypatch):
    config = _fresh(monkeypatch, tmp_path)
    assert config.LOCAL_USER == 'local'

def test_load_config_returns_defensive_copy(tmp_path, monkeypatch):
    config = _fresh(monkeypatch, tmp_path)
    cfg = config.load_config()
    cfg['server']['port'] = 9999          # caller mutation must not corrupt the cache
    assert config.get('server.port') == 5050


def test_legacy_krea_default_pair_migrates_on_read_and_persists_on_next_save(tmp_path, monkeypatch):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'krea': {'grounding_px': 1024, 'ref_boost': 4.0}}),
                    encoding='utf-8')
    config = _fresh(monkeypatch, tmp_path)

    # Reads do not rewrite user configuration, but the corrected profile applies
    # immediately to generation on an upgraded install.
    assert config.get('krea.grounding_px') == 1024
    assert config.get('krea.ref_boost') == 4.0
    assert config.get('krea.steps') == 12
    untouched = json.loads(path.read_text(encoding='utf-8'))
    assert untouched['krea'] == {'grounding_px': 1024, 'ref_boost': 4.0}

    # Its next ordinary Settings save records the one-time migration too.
    config.save_config({'server': {'port': 5051}})
    saved = json.loads(path.read_text(encoding='utf-8'))['krea']
    assert saved['grounding_px'] == 1024
    assert saved['ref_boost'] == 4.0
    assert saved['steps'] == 12
    assert saved['calibration_version'] == config.KREA_CALIBRATION_VERSION


def test_previous_krea_default_profile_migrates_but_a_v2_custom_profile_survives(
        tmp_path, monkeypatch):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'krea': {
        'calibration_version': 2, 'grounding_px': 512, 'ref_boost': 1.0, 'steps': 10,
    }}), encoding='utf-8')
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('krea.grounding_px') == 1024
    assert config.get('krea.ref_boost') == 4.0
    assert config.get('krea.steps') == 12

    path.write_text(json.dumps({'krea': {
        'calibration_version': 2, 'grounding_px': 512, 'ref_boost': 0.5, 'steps': 8,
    }}), encoding='utf-8')
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('krea.grounding_px') == 512
    assert config.get('krea.ref_boost') == 0.5
    assert config.get('krea.steps') == 8

    path.write_text(json.dumps({'krea': {
        'calibration_version': 2, 'grounding_px': 1024, 'ref_boost': 4.0, 'steps': 10,
    }}), encoding='utf-8')
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('krea.grounding_px') == 1024
    assert config.get('krea.ref_boost') == 4.0
    assert config.get('krea.steps') == 10


def test_a_custom_legacy_krea_calibration_is_not_rewritten(tmp_path, monkeypatch):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'krea': {'grounding_px': 768, 'ref_boost': 4.0}}),
                    encoding='utf-8')
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('krea.grounding_px') == 768
    assert config.get('krea.ref_boost') == 4.0

    # An explicit post-update choice, even the old 1024 value, receives a marker
    # and is never mistaken for the historical untouched pair on a later reload.
    config.save_config({'krea': {'grounding_px': 1024}})
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('krea.grounding_px') == 1024
    assert config.get('krea.ref_boost') == 4.0
    saved = json.loads(path.read_text(encoding='utf-8'))['krea']
    assert saved['calibration_version'] == config.KREA_CALIBRATION_VERSION


def test_a_legacy_krea_pair_with_custom_steps_is_not_rewritten(tmp_path, monkeypatch):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'krea': {
        'grounding_px': 1024, 'ref_boost': 4.0, 'steps': 20,
    }}), encoding='utf-8')
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('krea.grounding_px') == 1024
    assert config.get('krea.ref_boost') == 4.0
    assert config.get('krea.steps') == 20


def test_v3_krea_default_profile_migrates_to_v4_but_a_v3_custom_profile_survives(
        tmp_path, monkeypatch):
    """v4 = 1024 / 4.0 / 12 — the identity-first profile.

    v3 shipped 512 / 0.25, which is neither calibrated pair the engine ever had
    (v1 = 1024/4.0, v2 = 512/1.0): the reference pull had drifted to a quarter of
    v2's. A benchmark on one reference measured 1024/4.0 ahead by +0.17 face
    similarity on bust framing, with no overlap between the two runs, and at a
    LOWER pull — so more likeness bought with less recopying, not more.

    This is a deliberate product choice, not a proven optimum: the evidence is
    one face and four scored images, and it walks back a default the project
    moved away from twice on purpose (the v1 and v2 profiles were judged too
    reference-dominated for varied dataset poses). It is recorded here so
    whoever reconsiders it knows exactly what it rests on.

    An install that never touched the v3 dials follows. Anything else is a
    choice and is left alone — the same contract every earlier bump honoured.
    """
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'krea': {
        'calibration_version': 3, 'grounding_px': 512, 'ref_boost': 0.25, 'steps': 8,
    }}), encoding='utf-8')
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('krea.grounding_px') == 1024
    assert config.get('krea.ref_boost') == 4.0
    assert config.get('krea.steps') == 12

    # One dial moved by hand = a profile the user owns. Untouched, in full.
    path.write_text(json.dumps({'krea': {
        'calibration_version': 3, 'grounding_px': 512, 'ref_boost': 2.0, 'steps': 8,
    }}), encoding='utf-8')
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('krea.grounding_px') == 512
    assert config.get('krea.ref_boost') == 2.0
    assert config.get('krea.steps') == 8


def test_a_fresh_install_gets_the_v4_identity_profile(tmp_path, monkeypatch):
    """No config.json at all: what a brand-new install actually generates with."""
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('krea.grounding_px') == 1024
    assert config.get('krea.ref_boost') == 4.0
    assert config.get('krea.steps') == 12
