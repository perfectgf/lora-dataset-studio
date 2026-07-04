from unittest.mock import patch
import pytest


@pytest.fixture(autouse=True)
def _no_real_subprocess(monkeypatch):
    """Import-probes (face_scoring/masks) shell out to python -c 'import ...'.
    Stub the seam so the suite never spawns a real subprocess; individual
    tests that care about the ok/False split re-patch it locally."""
    from app import capabilities
    capabilities._import_cache.clear()
    capabilities._cache = None
    capabilities._cache_ts = 0.0
    monkeypatch.setattr(capabilities, '_import_ok', lambda *a, **k: False)
    yield
    capabilities._import_cache.clear()
    capabilities._cache = None
    capabilities._cache_ts = 0.0


# --- brief tests, verbatim ---------------------------------------------

def test_probe_all_off_when_unconfigured(app):
    with app.app_context():
        from app import capabilities
        with patch('app.capabilities._http_ok', return_value=False):
            caps = capabilities.probe(force=True)
    assert caps['engines'] == {'nanobanana': False, 'chatgpt': False, 'klein': False}
    assert caps['training_visible'] is False and caps['studio_visible'] is False

def test_chatgpt_on_with_key(app, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-x')
    with app.app_context():
        from app import capabilities
        with patch('app.capabilities._http_ok', return_value=False):
            caps = capabilities.probe(force=True)
    assert caps['engines']['chatgpt'] is True

def test_comfyui_reachable_lights_studio_and_klein(app, monkeypatch, tmp_path):
    monkeypatch.setenv('OPENAI_API_KEY', '')
    with app.app_context():
        from app import capabilities, config
        base = tmp_path / 'Comfy'
        (base / 'models' / 'unet' / 'klein').mkdir(parents=True)
        (base / 'models' / 'unet' / 'klein' / 'k.safetensors').touch()
        config.save_config({'comfyui': {'base_dir': str(base)}})
        with patch('app.capabilities._http_ok', return_value=True):
            caps = capabilities.probe(force=True)
    assert caps['comfyui']['reachable'] is True
    assert caps['studio_visible'] is True
    assert caps['engines']['klein'] is True


# --- extra coverage: individual probe_* ok/detail contract --------------

def test_probe_gemini_missing_key(app, monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    with app.app_context():
        from app import capabilities
        result = capabilities.probe_gemini()
    assert result == {'ok': False, 'detail': 'key missing'}

def test_probe_gemini_with_key(app, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'g-x')
    with app.app_context():
        from app import capabilities
        result = capabilities.probe_gemini()
    assert result == {'ok': True, 'detail': 'key set'}

def test_probe_openai_missing_key(app, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with app.app_context():
        from app import capabilities
        result = capabilities.probe_openai()
    assert result == {'ok': False, 'detail': 'key missing'}

def test_probe_aitoolkit_invalid_when_unconfigured(app):
    with app.app_context():
        from app import capabilities
        result = capabilities.probe_aitoolkit()
    assert result['ok'] is False

def test_probe_aitoolkit_invalid_when_dir_set_but_incomplete(app, tmp_path):
    with app.app_context():
        from app import capabilities, config
        root = tmp_path / 'aitoolkit'
        root.mkdir()  # exists, but no run.py, no venv
        config.save_config({'aitoolkit': {'dir': str(root)}})
        result = capabilities.probe_aitoolkit()
    assert result['ok'] is False

def test_probe_aitoolkit_valid(app, tmp_path):
    with app.app_context():
        from app import capabilities, config
        root = tmp_path / 'aitoolkit'
        (root / 'venv' / 'Scripts').mkdir(parents=True)
        (root / 'venv' / 'Scripts' / 'python.exe').touch()
        (root / 'run.py').touch()
        config.save_config({'aitoolkit': {'dir': str(root)}})
        result = capabilities.probe_aitoolkit()
    assert result['ok'] is True

def test_probe_comfyui_unreachable(app):
    with app.app_context():
        from app import capabilities
        with patch('app.capabilities._http_ok', return_value=False):
            result = capabilities.probe_comfyui()
    assert result['ok'] is False

def test_probe_ollama_reachable(app):
    with app.app_context():
        from app import capabilities
        with patch('app.capabilities._http_ok', return_value=True):
            result = capabilities.probe_ollama()
    assert result['ok'] is True

def test_probe_face_scoring_goes_through_import_seam(app, monkeypatch):
    with app.app_context():
        from app import capabilities
        monkeypatch.setattr(capabilities, '_import_ok', lambda *a, **k: True)
        capabilities._import_cache.clear()
        result = capabilities.probe_face_scoring()
    assert result == {'ok': True, 'detail': 'insightface + onnxruntime import OK'}

def test_probe_masks_goes_through_import_seam(app, monkeypatch):
    with app.app_context():
        from app import capabilities
        monkeypatch.setattr(capabilities, '_import_ok', lambda *a, **k: True)
        capabilities._import_cache.clear()
        result = capabilities.probe_masks()
    assert result == {'ok': True, 'detail': 'rembg import OK'}

def test_import_probe_result_is_cached(app, monkeypatch):
    """Second call within the 10 min TTL must not re-invoke the seam."""
    with app.app_context():
        from app import capabilities
        calls = []
        monkeypatch.setattr(capabilities, '_import_ok', lambda *a, **k: calls.append(1) or True)
        capabilities._import_cache.clear()
        capabilities.probe_face_scoring()
        capabilities.probe_face_scoring()
    assert len(calls) == 1


def test_import_probe_cache_key_includes_interpreter_path(app, monkeypatch):
    """Changing interpreter path should invalidate the import cache."""
    with app.app_context():
        from app import capabilities, config
        import sys

        # First call: interpreter A returns True
        monkeypatch.setattr(capabilities, '_import_ok', lambda *a, **k: True)
        capabilities._import_cache.clear()
        result1 = capabilities.probe_face_scoring()
        assert result1['ok'] is True

        # Second call: same interpreter, should use cache and return True
        monkeypatch.setattr(capabilities, '_import_ok', lambda *a, **k: False)
        result2 = capabilities.probe_face_scoring()
        assert result2['ok'] is True  # cached result

        # Third call: different interpreter path, should bypass cache and return False
        config.save_config({'face_scoring': {'python': '/different/python/path'}})
        result3 = capabilities.probe_face_scoring()
        assert result3['ok'] is False  # new interpreter, not cached


# --- model listing scan rules --------------------------------------------

def test_scan_models_empty_when_comfyui_unset(app):
    with app.app_context():
        from app import capabilities
        models = capabilities._scan_models()
    assert models == {'zimage': [], 'sdxl': [], 'krea': [], 'klein': []}

def test_scan_models_matches_rules(app, tmp_path):
    with app.app_context():
        from app import capabilities, config
        base = tmp_path / 'Comfy'
        (base / 'models' / 'unet' / 'Z-Image').mkdir(parents=True)
        (base / 'models' / 'unet' / 'Z-Image' / 'a.safetensors').touch()
        (base / 'models' / 'unet' / 'Z-Image' / 'notes.txt').touch()   # filtered out
        (base / 'models' / 'unet' / 'krea-turbo').mkdir(parents=True)
        (base / 'models' / 'unet' / 'krea-turbo' / 'k.gguf').touch()
        (base / 'models' / 'unet' / 'klein').mkdir(parents=True)
        (base / 'models' / 'unet' / 'klein' / 'k.safetensors').touch()
        (base / 'models' / 'checkpoints').mkdir(parents=True)
        (base / 'models' / 'checkpoints' / 'sdxl_base.safetensors').touch()
        config.save_config({'comfyui': {'base_dir': str(base)}})
        models = capabilities._scan_models()
    assert models['zimage'] == ['a.safetensors']
    assert models['krea'] == ['k.gguf']
    assert models['klein'] == ['k.safetensors']
    assert models['sdxl'] == ['sdxl_base.safetensors']

def test_scan_models_never_raises_on_absent_dir(app, tmp_path):
    with app.app_context():
        from app import capabilities, config
        config.save_config({'comfyui': {'base_dir': str(tmp_path / 'does_not_exist')}})
        models = capabilities._scan_models()
    assert models == {'zimage': [], 'sdxl': [], 'krea': [], 'klein': []}


# --- probe() caching ------------------------------------------------------

def test_probe_caches_for_30s_without_force(app, monkeypatch):
    with app.app_context():
        from app import capabilities
        capabilities._cache = None
        capabilities._cache_ts = 0.0
        with patch('app.capabilities._http_ok', return_value=False):
            first = capabilities.probe(force=True)
            monkeypatch.setenv('OPENAI_API_KEY', 'sk-new')
            second = capabilities.probe()  # stale cache, ignores the new key
    assert second == first
    assert second['engines']['chatgpt'] is False

def test_probe_force_bypasses_cache(app, monkeypatch):
    with app.app_context():
        from app import capabilities
        with patch('app.capabilities._http_ok', return_value=False):
            capabilities.probe(force=True)
            monkeypatch.setenv('OPENAI_API_KEY', 'sk-new')
            refreshed = capabilities.probe(force=True)
    assert refreshed['engines']['chatgpt'] is True
