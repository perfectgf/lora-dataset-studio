from unittest.mock import patch
import pathlib
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

def test_python_ml_status_reports_version_and_range(app):
    """The probe exposes the interpreter version + whether it's inside the ML-wheel
    range (3.10–3.12), so the setup can warn before a doomed pip install."""
    with app.app_context():
        from app import capabilities
        with patch('app.capabilities._http_ok', return_value=False):
            caps = capabilities.probe(force=True)
    py = caps['python']
    assert py['ml_range'] == '3.10–3.12'
    assert isinstance(py['ml_supported'], bool)
    # ml_supported must agree with the reported version's minor.
    major, minor = (int(x) for x in py['version'].split('.')[:2])
    assert py['ml_supported'] == (major == 3 and 10 <= minor <= 12)


@pytest.mark.parametrize('info,ok', [((3, 9, 1), False), ((3, 10, 0), True),
                                     ((3, 12, 9), True), ((3, 13, 0), False), ((3, 14, 0), False)])
def test_python_ml_status_boundaries(app, info, ok):
    import types
    with app.app_context():
        from app import capabilities
        vi = types.SimpleNamespace(major=info[0], minor=info[1], micro=info[2])
        with patch('app.capabilities.sys.version_info', vi):
            st = capabilities.python_ml_status()
    assert st['ml_supported'] is ok
    assert st['version'] == f'{info[0]}.{info[1]}.{info[2]}'


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


def test_import_probe_timeout_is_not_cached_as_failure(app, monkeypatch):
    """_import_ok → None (subprocess TIMEOUT, e.g. rembg's first cold import
    compiling numba caches) must report not-ready NOW but not poison the 10 min
    cache: the next probe re-tries (warm import ~1 s → ✓). A real import error
    (False) stays cached as before."""
    with app.app_context():
        from app import capabilities
        calls = []
        monkeypatch.setattr(capabilities, '_import_ok',
                            lambda *a, **k: calls.append(1) or None)   # timeout
        capabilities._import_cache.clear()
        assert capabilities.probe_masks()['ok'] is False
        assert capabilities.probe_masks()['ok'] is False
        assert len(calls) == 2                       # re-probed: nothing cached
        monkeypatch.setattr(capabilities, '_import_ok',
                            lambda *a, **k: calls.append(1) or False)  # real failure
        assert capabilities.probe_masks()['ok'] is False
        assert capabilities.probe_masks()['ok'] is False
        assert len(calls) == 3                       # cached after the real False


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


# --- resolve_comfyui_base: portable-wrapper nesting ----------------------

def _make_comfyui(root):
    """Minimal ComfyUI marker: main.py + models/ is what _is_comfyui_dir checks."""
    root.mkdir(parents=True, exist_ok=True)
    (root / 'main.py').touch()
    (root / 'models').mkdir()


def test_resolve_comfyui_base_direct(tmp_path):
    from app.capabilities import resolve_comfyui_base
    _make_comfyui(tmp_path)
    r = resolve_comfyui_base(str(tmp_path))
    assert r['valid'] is True and r['nested'] is False
    assert pathlib.Path(r['resolved']) == tmp_path

def test_resolve_comfyui_base_portable_nested(tmp_path):
    """User points at ...\\ComfyUI_windows_portable; the real install is one level
    down in .../ComfyUI. resolve descends and flags nested=True so the caller
    can auto-correct base_dir."""
    from app.capabilities import resolve_comfyui_base
    wrapper = tmp_path / 'ComfyUI_windows_portable'
    _make_comfyui(wrapper / 'ComfyUI')
    r = resolve_comfyui_base(str(wrapper))
    assert r['valid'] is True and r['nested'] is True
    assert pathlib.Path(r['resolved']) == wrapper / 'ComfyUI'

def test_resolve_comfyui_base_invalid(tmp_path):
    from app.capabilities import resolve_comfyui_base
    r = resolve_comfyui_base(str(tmp_path))   # empty dir, no main.py/models
    assert r['valid'] is False and r['nested'] is False
    assert pathlib.Path(r['resolved']) == tmp_path

def test_resolve_comfyui_base_empty():
    from app.capabilities import resolve_comfyui_base
    assert resolve_comfyui_base('') == {'valid': False, 'resolved': '', 'nested': False}

def test_probe_exposes_dir_valid(app, tmp_path):
    """probe() surfaces dir_configured/dir_valid/resolved_dir so the wizard can
    tell a wrong path from a right one without a second round-trip."""
    with app.app_context():
        from app import capabilities, config
        _make_comfyui(tmp_path / 'ComfyUI')
        config.save_config({'comfyui': {'base_dir': str(tmp_path)}})   # wrapper, nested install
        with patch('app.capabilities._http_ok', return_value=False):
            caps = capabilities.probe(force=True)
    c = caps['comfyui']
    assert c['dir_configured'] is True and c['dir_valid'] is True
    assert pathlib.Path(c['resolved_dir']) == tmp_path / 'ComfyUI'


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


# --- ollama vision-model presence + import-cache clear --------------------

def test_ollama_vision_model_ready_true(app, monkeypatch):
    with app.app_context():
        from app import capabilities, config
        config.save_config({'ollama': {'url': 'http://o', 'vision_model': 'qwen3-vl:8b'}})
        monkeypatch.setattr(capabilities, '_http_ok', lambda *a, **k: True)
        monkeypatch.setattr(capabilities, '_ollama_tags', lambda *a, **k: ['qwen3-vl:8b'])
        result = capabilities.probe_ollama_model()
    assert result['ok'] is True

def test_ollama_vision_model_ready_false_when_absent(app, monkeypatch):
    with app.app_context():
        from app import capabilities, config
        config.save_config({'ollama': {'url': 'http://o', 'vision_model': 'qwen3-vl:8b'}})
        monkeypatch.setattr(capabilities, '_http_ok', lambda *a, **k: True)
        monkeypatch.setattr(capabilities, '_ollama_tags', lambda *a, **k: ['llama3:8b'])
        result = capabilities.probe_ollama_model()
    assert result['ok'] is False

def test_ollama_vision_model_base_tag_match(app, monkeypatch):
    with app.app_context():
        from app import capabilities, config
        config.save_config({'ollama': {'url': 'http://o', 'vision_model': 'qwen3-vl'}})
        monkeypatch.setattr(capabilities, '_http_ok', lambda *a, **k: True)
        monkeypatch.setattr(capabilities, '_ollama_tags', lambda *a, **k: ['qwen3-vl:8b'])
        result = capabilities.probe_ollama_model()
    assert result['ok'] is True

def test_ollama_vision_model_false_when_unreachable(app, monkeypatch):
    with app.app_context():
        from app import capabilities, config
        config.save_config({'ollama': {'url': 'http://o', 'vision_model': 'qwen3-vl:8b'}})
        monkeypatch.setattr(capabilities, '_http_ok', lambda *a, **k: False)
        # _ollama_tags must not even be consulted when unreachable:
        monkeypatch.setattr(capabilities, '_ollama_tags',
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError('called')))
        result = capabilities.probe_ollama_model()
    assert result['ok'] is False

def test_probe_exposes_vision_model_fields(app, monkeypatch):
    with app.app_context():
        from app import capabilities, config
        config.save_config({'ollama': {'url': 'http://o', 'vision_model': 'qwen3-vl:8b'}})
        monkeypatch.setattr(capabilities, '_http_ok', lambda *a, **k: True)
        monkeypatch.setattr(capabilities, '_ollama_tags', lambda *a, **k: ['qwen3-vl:8b'])
        caps = capabilities.probe(force=True)
    assert caps['ollama']['vision_model'] == 'qwen3-vl:8b'
    assert caps['ollama']['vision_model_ready'] is True

def test_clear_import_cache_empties_caches(app, monkeypatch):
    with app.app_context():
        from app import capabilities
        monkeypatch.setattr(capabilities, '_import_ok', lambda *a, **k: True)
        capabilities.probe_face_scoring()          # populates _import_cache
        assert capabilities._import_cache
        capabilities._cache = {'x': 1}; capabilities._cache_ts = 123.0
        capabilities.clear_import_cache()
    assert capabilities._import_cache == {}
    assert capabilities._cache is None

def test_probe_ollama_model_uses_passed_reachability(app, monkeypatch):
    """probe() supplies the already-computed reachability so probe_ollama_model
    does not re-hit _http_ok — avoids the redundant/doubled /api/tags call."""
    with app.app_context():
        from app import capabilities, config
        config.save_config({'ollama': {'url': 'http://o', 'vision_model': 'qwen3-vl:8b'}})
        http_calls = []
        monkeypatch.setattr(capabilities, '_http_ok', lambda *a, **k: http_calls.append(1) or True)
        monkeypatch.setattr(capabilities, '_ollama_tags', lambda *a, **k: ['qwen3-vl:8b'])
        ready = capabilities.probe_ollama_model(reachable=True)
        monkeypatch.setattr(capabilities, '_ollama_tags',
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError('tags fetched')))
        down = capabilities.probe_ollama_model(reachable=False)
    assert ready['ok'] is True
    assert http_calls == []          # reachability supplied, not re-fetched
    assert down['ok'] is False       # short-circuited without fetching tags


# --- Task 5: probe_openai() matrix + chatgpt_subscription payload ---------

def _sub(connected, email=None):
    return {'connected': connected, 'email': email, 'plan': 'plus' if connected else None}


def test_probe_openai_matrix(app, monkeypatch):
    from unittest.mock import patch
    from app import capabilities
    from app.services import chatgpt_oauth
    # neither
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with patch.object(chatgpt_oauth, 'status', return_value=_sub(False)):
        r = capabilities.probe_openai()
        assert r['ok'] is False and r['detail'] == 'key missing'
    # key only
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-x')
    with patch.object(chatgpt_oauth, 'status', return_value=_sub(False)):
        r = capabilities.probe_openai()
        assert r['ok'] is True and r['detail'] == 'key set'
    # subscription only
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with patch.object(chatgpt_oauth, 'status', return_value=_sub(True, 'u@x.io')):
        r = capabilities.probe_openai()
        assert r['ok'] is True and r['detail'] == 'subscription connected'
    # both
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-x')
    with patch.object(chatgpt_oauth, 'status', return_value=_sub(True, 'u@x.io')):
        r = capabilities.probe_openai()
        assert r['ok'] is True and r['detail'] == 'key set + subscription connected'


def test_probe_exposes_chatgpt_subscription_block(app, monkeypatch):
    from unittest.mock import patch
    from app import capabilities
    from app.services import chatgpt_oauth
    with patch.object(chatgpt_oauth, 'status', return_value=_sub(True, 'u@x.io')):
        caps = capabilities.probe(force=True)
    sub = caps['chatgpt_subscription']
    assert sub['connected'] is True
    assert sub['email'] == 'u@x.io'
    assert isinstance(sub['codex_cli_detected'], bool)
    assert caps['engines']['chatgpt'] is True     # subscription alone enables the engine


def test_probe_aitoolkit_accepts_dot_venv_and_explicit_python(app, tmp_path, monkeypatch):
    """Installs without `venv/` exist in the wild (Reddit-reported): `.venv/`
    must be auto-detected, and an explicit aitoolkit.python must win over
    both. run.py present but no interpreter -> ACTIONABLE detail."""
    import os
    from app import capabilities, config as cfg
    root = tmp_path / 'aitk'
    (root / '.venv' / ('Scripts' if os.name == 'nt' else 'bin')).mkdir(parents=True)
    py = root / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    py.touch()
    (root / 'run.py').touch()
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(root)}})
        assert capabilities.probe_aitoolkit()['ok'] is True
        # explicit interpreter wins (even over an existing .venv)
        other = tmp_path / 'conda-python.exe'
        other.touch()
        cfg.save_config({'aitoolkit': {'dir': str(root), 'python': str(other)}})
        assert cfg.aitoolkit_path('venv_python') == other
        assert capabilities.probe_aitoolkit()['ok'] is True
        # run.py present, no interpreter anywhere -> actionable message
        bare = tmp_path / 'bare'
        bare.mkdir()
        (bare / 'run.py').touch()
        cfg.save_config({'aitoolkit': {'dir': str(bare), 'python': ''}})
        probe = capabilities.probe_aitoolkit()
        assert probe['ok'] is False
        assert 'Python interpreter' in probe['detail']


def test_is_comfyui_dir_accepts_desktop_layout(tmp_path):
    """The ComfyUI Desktop app's basedir has models/ + custom_nodes/ but NO
    main.py (a user had to symlink one to pass the old check)."""
    from app.capabilities import _is_comfyui_dir
    desktop = tmp_path / 'desktop'
    (desktop / 'models').mkdir(parents=True)
    (desktop / 'custom_nodes').mkdir()
    assert _is_comfyui_dir(desktop) is True
    classic = tmp_path / 'classic'
    (classic / 'models').mkdir(parents=True)
    (classic / 'main.py').touch()
    assert _is_comfyui_dir(classic) is True
    not_comfy = tmp_path / 'other'
    (not_comfy / 'custom_nodes').mkdir(parents=True)   # no models/
    assert _is_comfyui_dir(not_comfy) is False
