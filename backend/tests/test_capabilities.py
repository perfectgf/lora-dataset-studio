from unittest.mock import patch
import os
import pathlib
import struct
import pytest

# Smallest structurally-valid safetensors: 8-byte LE header length + a 2-byte empty
# JSON object ('{}'). model_integrity now reads a 0-byte touch()ed stub as a broken
# file, so a fixture that needs a PRESENT-and-VALID model writes these bytes. It is
# tiny (far below the type floor), so it also reads as the advisory `too_small` —
# which, being non-blocking, never gates the engine (only a hard-invalid file does).
_VALID_ST = struct.pack('<Q', 2) + b'{}'


@pytest.fixture(autouse=True)
def _no_real_subprocess(monkeypatch):
    """Import-probes (face_scoring/masks) shell out to python -c 'import ...'.
    Stub the seam so the suite never spawns a real subprocess; individual
    tests that care about the ok/False split re-patch it locally."""
    from app import capabilities
    real_import_ok = capabilities._import_ok
    capabilities._import_cache.clear()
    capabilities._cache = None
    capabilities._cache_ts = 0.0
    monkeypatch.setattr(capabilities, '_import_ok', lambda *a, **k: False)
    yield real_import_ok
    capabilities._import_cache.clear()
    capabilities._cache = None
    capabilities._cache_ts = 0.0


# --- brief tests, verbatim ---------------------------------------------

def test_probe_all_off_when_unconfigured(app):
    with app.app_context():
        from app import capabilities
        with patch('app.capabilities._http_ok', return_value=False):
            caps = capabilities.probe(force=True)
    assert caps['engines'] == {'nanobanana': False, 'chatgpt': False,
                               'openrouter': False, 'klein': False, 'krea': False}
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
        (base / 'models' / 'unet' / 'klein' / 'k.safetensors').write_bytes(_VALID_ST)
        # Klein readiness is now tri-component (unet + vae + text-encoder), and each
        # must be REAL weights, not just a file with the right name: the engine only
        # lights when a generate could actually run. Materialise the canonical vae +
        # text-encoder too, with valid headers.
        (base / 'models' / 'vae').mkdir(parents=True)
        (base / 'models' / 'vae' / 'flux2-vae.safetensors').write_bytes(_VALID_ST)
        (base / 'models' / 'text_encoders').mkdir(parents=True)
        (base / 'models' / 'text_encoders' / 'qwen_3_8b_fp8mixed.safetensors').write_bytes(_VALID_ST)
        config.save_config({'comfyui': {'base_dir': str(base)}})
        with patch('app.capabilities._http_ok', return_value=True):
            caps = capabilities.probe(force=True)
    assert caps['comfyui']['reachable'] is True
    assert caps['studio_visible'] is True
    assert caps['engines']['klein'] is True
    # Required trio on disk -> none of them appear in the per-asset gap list the
    # Setup UI reads (only the recommended consistency LoRA is still absent here).
    missing = caps['comfyui']['klein_missing']
    assert 'klein_model' not in missing
    assert 'klein_text_encoder' not in missing
    assert 'klein_vae' not in missing


def test_klein_engine_stays_dark_until_all_three_assets_present(app, monkeypatch, tmp_path):
    """Honest readiness: a reachable ComfyUI with ONLY the unet (no vae / no
    text-encoder) must NOT light the Klein engine — the generate would 409 for the
    missing assets (which the 409 then names + auto-downloads)."""
    monkeypatch.setenv('OPENAI_API_KEY', '')
    with app.app_context():
        from app import capabilities, config
        base = tmp_path / 'Comfy'
        (base / 'models' / 'unet' / 'klein').mkdir(parents=True)
        (base / 'models' / 'unet' / 'klein' / 'k.safetensors').write_bytes(_VALID_ST)
        config.save_config({'comfyui': {'base_dir': str(base)}})
        with patch('app.capabilities._http_ok', return_value=True):
            caps = capabilities.probe(force=True)
    assert caps['comfyui']['models']['klein'] == ['k.safetensors']   # picker still lists it
    assert caps['engines']['klein'] is False                          # but engine stays dark
    # The payload names the exact gap so the Setup step lists the missing weights
    # (and keeps their download buttons) instead of blaming the already-present unet.
    missing = caps['comfyui']['klein_missing']
    assert 'klein_model' not in missing          # unet IS on disk
    assert 'klein_text_encoder' in missing
    assert 'klein_vae' in missing


def test_klein_engine_lights_for_flat_root_layout_unet(app, monkeypatch, tmp_path):
    """waltm' flat / Stability-Matrix layout: the UNET sits straight in
    diffusion_models/ with NO klein/ subfolder. The engine gate now runs on
    resolve_klein_unet() (symmetric with the vae/te resolvers), so a root-level
    model that the picker lists and the resolver can build ALSO lights the engine —
    picker == probe == resolver at the readiness gate too."""
    monkeypatch.setenv('OPENAI_API_KEY', '')
    with app.app_context():
        from app import capabilities, config
        base = tmp_path / 'Comfy'
        # UNET dropped at the root of diffusion_models/ (no klein/ subfolder).
        (base / 'models' / 'diffusion_models').mkdir(parents=True)
        (base / 'models' / 'diffusion_models' / 'flux-2-klein-9b-fp8.safetensors').write_bytes(_VALID_ST)
        (base / 'models' / 'vae').mkdir(parents=True)
        (base / 'models' / 'vae' / 'flux2-vae.safetensors').write_bytes(_VALID_ST)
        (base / 'models' / 'text_encoders').mkdir(parents=True)
        (base / 'models' / 'text_encoders' / 'qwen_3_8b_fp8mixed.safetensors').write_bytes(_VALID_ST)
        config.save_config({'comfyui': {'base_dir': str(base)}})
        with patch('app.capabilities._http_ok', return_value=True):
            caps = capabilities.probe(force=True)
    # Picker lists the bare root name, and the engine lights (all three resolvable).
    assert caps['comfyui']['models']['klein'] == ['flux-2-klein-9b-fp8.safetensors']
    assert caps['engines']['klein'] is True
    assert caps['comfyui']['klein_missing'] == ['klein_lora', 'klein_enhancement_lora']


def test_klein_engine_dark_when_unet_is_html_gate_page(app, monkeypatch, tmp_path):
    """The #help incident: a licence-gated model downloaded from a browser WITHOUT
    accepting the licence saves the HTML gate PAGE to <name>.safetensors. The file
    EXISTS, so the old existence check went green — then UNETLoader crashed at
    generate time on 'Expecting value: line 1 column 1'. The engine must now stay
    dark, and the payload must report the file as present-but-INVALID (distinct from
    missing) so the Setup step can say 'delete it and re-download'."""
    monkeypatch.setenv('OPENAI_API_KEY', '')
    with app.app_context():
        from app import capabilities, config
        from app.services import model_integrity
        model_integrity.clear_cache()
        base = tmp_path / 'Comfy'
        (base / 'models' / 'unet' / 'klein').mkdir(parents=True)
        # A real-looking filename, but the bytes are the HTML licence-gate page.
        (base / 'models' / 'unet' / 'klein' / 'flux-2-klein-9b-fp8.safetensors').write_bytes(
            b'<!doctype html><html><head><title>Access gated</title></head></html>')
        (base / 'models' / 'vae').mkdir(parents=True)
        (base / 'models' / 'vae' / 'flux2-vae.safetensors').write_bytes(_VALID_ST)
        (base / 'models' / 'text_encoders').mkdir(parents=True)
        (base / 'models' / 'text_encoders' / 'qwen_3_8b_fp8mixed.safetensors').write_bytes(_VALID_ST)
        config.save_config({'comfyui': {'base_dir': str(base)}})
        with patch('app.capabilities._http_ok', return_value=True):
            caps = capabilities.probe(force=True)
    # NOT missing — the file is on disk ...
    assert 'klein_model' not in caps['comfyui']['klein_missing']
    # ... but present-and-INVALID, so the engine stays dark instead of luring the
    # user into a generate that crashes ComfyUI.
    assert caps['engines']['klein'] is False
    invalid = {i['asset']: i for i in caps['comfyui']['klein_invalid']}
    assert 'klein_model' in invalid
    assert invalid['klein_model']['blocking'] is True
    assert invalid['klein_model']['verdict'] == 'html_or_text'
    assert 'HTML' in invalid['klein_model']['reason']


def test_a_truncated_download_is_reported_as_broken_not_missing(app, monkeypatch, tmp_path):
    """zigzag4794's install (Discord), as a payload contract.

    ComfyUI up, every required file in its folder at a plausible size, Setup
    showing "✓ Installed" — and the Generate page refusing Klein with "model
    missing". The bytes tell the real story: a .safetensors declares its JSON
    header length in its first 8 bytes, and an interrupted download leaves a file
    shorter than it claims. `truncated_or_garbage`.

    This pins the wire format both screens read, because that is where the two
    used to be able to disagree: the asset is NOT in klein_missing (it is on
    disk), it IS in klein_invalid as blocking, and engines.klein is False. A front
    end given this payload has everything it needs to say "corrupted — delete and
    download again" instead of guessing "missing"."""
    monkeypatch.setenv('OPENAI_API_KEY', '')
    with app.app_context():
        from app import capabilities, config
        from app.services import model_integrity
        model_integrity.clear_cache()
        base = tmp_path / 'Comfy'
        (base / 'models' / 'unet' / 'klein').mkdir(parents=True)
        # Declares a 4 GB header it does not carry — the shape of a download that
        # stopped, or a file copied off a full disk.
        (base / 'models' / 'unet' / 'klein' / 'flux-2-klein-9b-kv-fp8.safetensors').write_bytes(
            struct.pack('<Q', 4 * 1024 ** 3) + b'\0' * 4096)
        (base / 'models' / 'vae').mkdir(parents=True)
        (base / 'models' / 'vae' / 'flux2-vae.safetensors').write_bytes(_VALID_ST)
        (base / 'models' / 'text_encoders').mkdir(parents=True)
        (base / 'models' / 'text_encoders' / 'qwen_3_8b_fp8mixed.safetensors').write_bytes(_VALID_ST)
        config.save_config({'comfyui': {'base_dir': str(base)}})
        with patch('app.capabilities._http_ok', return_value=True):
            caps = capabilities.probe(force=True)
    assert caps['engines']['klein'] is False
    assert 'klein_model' not in caps['comfyui']['klein_missing']   # it is NOT missing
    invalid = {i['asset']: i for i in caps['comfyui']['klein_invalid']}
    assert invalid['klein_model']['verdict'] == 'truncated_or_garbage'
    assert invalid['klein_model']['blocking'] is True
    assert invalid['klein_model']['filename'] == 'flux-2-klein-9b-kv-fp8.safetensors'
    # Nothing else was wrong on his machine — which is exactly why blaming the
    # weights read as nonsense: the enum probe found no gap either.
    assert caps['comfyui']['klein_unsupported_enums'] == []


def test_klein_invalid_too_small_is_advisory_and_does_not_gate(app, monkeypatch, tmp_path):
    """A tiny-but-structurally-valid stub (a download that stopped right after the
    header) is reported in klein_invalid as an advisory `too_small` — but being
    non-blocking it does NOT darken the engine; only a hard-invalid file does."""
    monkeypatch.setenv('OPENAI_API_KEY', '')
    with app.app_context():
        from app import capabilities, config
        from app.services import model_integrity
        model_integrity.clear_cache()
        base = tmp_path / 'Comfy'
        (base / 'models' / 'unet' / 'klein').mkdir(parents=True)
        (base / 'models' / 'unet' / 'klein' / 'flux-2-klein-9b-fp8.safetensors').write_bytes(_VALID_ST)
        (base / 'models' / 'vae').mkdir(parents=True)
        (base / 'models' / 'vae' / 'flux2-vae.safetensors').write_bytes(_VALID_ST)
        (base / 'models' / 'text_encoders').mkdir(parents=True)
        (base / 'models' / 'text_encoders' / 'qwen_3_8b_fp8mixed.safetensors').write_bytes(_VALID_ST)
        config.save_config({'comfyui': {'base_dir': str(base)}})
        with patch('app.capabilities._http_ok', return_value=True):
            caps = capabilities.probe(force=True)
    assert caps['engines']['klein'] is True                      # advisory never gates
    invalid = {i['asset']: i for i in caps['comfyui']['klein_invalid']}
    assert invalid['klein_model']['verdict'] == 'too_small'
    assert all(not i['blocking'] for i in caps['comfyui']['klein_invalid'])


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

# ── ai-toolkit without a venv (reported on Reddit by Psyko_2000) ─────────────
# A community easy-install script sets ai-toolkit up with a `python_embeded`
# folder and no venv at all. The wizard used to answer "set up its Python venv
# per the README" — a remedy that install can never follow. The probe must state
# what it FOUND, and hand back the interpreter sitting right there so the UI can
# offer it in one click.

def _portable_aitoolkit(tmp_path):
    root = tmp_path / 'aitoolkit'
    (root / 'python_embeded').mkdir(parents=True)
    (root / 'python_embeded' / 'python.exe').write_text('fake')
    (root / 'run.py').write_text('fake')
    return root

def test_probe_aitoolkit_finds_a_portable_interpreter_in_the_folder(app, tmp_path):
    with app.app_context():
        from app import capabilities, config
        root = _portable_aitoolkit(tmp_path)
        config.save_config({'aitoolkit': {'dir': str(root)}})
        result = capabilities.probe_aitoolkit()
    assert result['ok'] is False          # nothing is auto-applied
    assert result['has_run'] is True      # ...but the checkout IS recognised
    assert result['python_candidates'] == [str(root / 'python_embeded' / 'python.exe')]
    # The detail names the observation and BOTH ways out — never "a venv is missing".
    assert 'no Python interpreter found' in result['detail']
    assert 'Settings' in result['detail']
    assert 'no venv/.venv' not in result['detail']

def test_probe_aitoolkit_degrades_cleanly_when_nothing_is_there(app, tmp_path):
    with app.app_context():
        from app import capabilities, config
        root = tmp_path / 'aitoolkit'
        (root / 'toolkit').mkdir(parents=True)     # a checkout, no interpreter anywhere
        (root / 'run.py').write_text('fake')
        config.save_config({'aitoolkit': {'dir': str(root)}})
        result = capabilities.probe_aitoolkit()
    assert result['ok'] is False
    assert result['has_run'] is True
    assert result['python_candidates'] == []      # empty, not a guess
    assert 'no Python interpreter found' in result['detail']

def test_aitoolkit_candidates_survive_an_unreadable_folder(app, tmp_path):
    with app.app_context():
        from app import capabilities
        assert capabilities.aitoolkit_python_candidates(tmp_path / 'nope') == []

def test_explicit_aitoolkit_python_makes_a_venv_less_install_valid(app, tmp_path):
    """The one-click offer lands here: saving aitoolkit.python flips the step."""
    with app.app_context():
        from app import capabilities, config
        root = _portable_aitoolkit(tmp_path)
        config.save_config({'aitoolkit': {
            'dir': str(root),
            'python': str(root / 'python_embeded' / 'python.exe')}})
        result = capabilities.probe_aitoolkit()
    assert result['ok'] is True


# --- the interpreter can be there and still be useless (GitHub #19, strouder) ---

def _explicit_and_venv(tmp_path):
    """An ai-toolkit checkout with a working `venv/` AND an explicit
    `aitoolkit.python` pointing somewhere else — the exact #19 shape."""
    root = tmp_path / 'ai-toolkit'
    (root / 'venv' / 'Scripts').mkdir(parents=True)
    (root / 'venv' / 'bin').mkdir(parents=True)
    (root / 'venv' / 'Scripts' / 'python.exe').write_text('fake')
    (root / 'venv' / 'bin' / 'python').write_text('fake')
    (root / 'run.py').write_text('fake')
    stray = tmp_path / 'stray'
    stray.mkdir()
    exe = stray / ('python.exe' if os.name == 'nt' else 'python')
    exe.write_text('fake')
    return root, exe

def _probe(**by_python):
    """Fake `_cached_import_state`: decide per interpreter path. `torch` = has
    torch, `alive` = runs at all but has no torch, `dead` = cannot be executed."""
    def fake(key, python, expr):
        kind = next((v for k, v in by_python.items() if k in python), 'dead')
        if kind == 'dead':
            return False                    # the seam reports False for everything
        if expr == 'pass':
            return True
        return kind == 'torch'
    return fake

def test_a_torchless_interpreter_is_reported_with_the_venv_that_works(app, tmp_path):
    """The offer that turns a dead end into one click."""
    with app.app_context():
        from app import capabilities, config
        root, stray = _explicit_and_venv(tmp_path)
        config.save_config({'aitoolkit': {'dir': str(root), 'python': str(stray)}})
        with patch.object(capabilities, '_cached_import_state',
                          side_effect=_probe(venv='torch', stray='alive')):
            report = capabilities.aitoolkit_interpreter_report()
    assert report['torch'] is False
    assert 'venv' in report['alternative']

def test_an_interpreter_that_cannot_run_at_all_is_UNKNOWN_not_torchless(app, tmp_path):
    """"This file is not a working Python" is a different problem from "this
    Python has no torch" — and a launch must never be refused on a guess."""
    with app.app_context():
        from app import capabilities, config
        root, stray = _explicit_and_venv(tmp_path)
        config.save_config({'aitoolkit': {'dir': str(root), 'python': str(stray)}})
        with patch.object(capabilities, '_cached_import_state',
                          side_effect=_probe(venv='torch')):   # stray -> dead
            report = capabilities.aitoolkit_interpreter_report()
    assert report['torch'] is None
    assert report['alternative'] == ''

def test_no_alternative_is_claimed_when_the_user_never_set_one(app, tmp_path):
    """With no explicit override there is nothing to switch AWAY from — the
    resolved interpreter IS the venv, so offering it would be nonsense."""
    with app.app_context():
        from app import capabilities, config
        root, _stray = _explicit_and_venv(tmp_path)
        config.save_config({'aitoolkit': {'dir': str(root), 'python': ''}})
        with patch.object(capabilities, '_cached_import_state',
                          side_effect=_probe(venv='alive')):
            report = capabilities.aitoolkit_interpreter_report()
    assert report['torch'] is False
    assert report['alternative'] == ''

def test_the_aitoolkit_test_button_fails_on_a_python_without_torch(app, tmp_path):
    """RED before this wave: folder checks alone went green on the broken setup."""
    with app.app_context():
        from app import capabilities, config
        root, stray = _explicit_and_venv(tmp_path)
        config.save_config({'aitoolkit': {'dir': str(root), 'python': str(stray)}})
        assert capabilities.probe_aitoolkit()['ok'] is True     # folders: fine
        with patch.object(capabilities, '_cached_import_state',
                          side_effect=_probe(venv='torch', stray='alive')):
            result = capabilities.probe_aitoolkit_test()
    assert result['ok'] is False
    assert 'import torch' in result['detail']

def test_the_test_button_stays_green_on_an_unanswered_probe(app, tmp_path):
    """A cold-import timeout is not a failed test."""
    with app.app_context():
        from app import capabilities, config
        root, stray = _explicit_and_venv(tmp_path)
        config.save_config({'aitoolkit': {'dir': str(root), 'python': str(stray)}})
        with patch.object(capabilities, '_cached_import_state', return_value=None):
            assert capabilities.probe_aitoolkit_test()['ok'] is True

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

def _configure_aitoolkit_ok(cfg, tmp_path):
    """A valid ai-toolkit checkout (run.py + a venv python) so probe_aitoolkit()
    passes and probe_joycaption() advances to the dep import check."""
    root = tmp_path / 'aitoolkit'
    (root / 'venv' / 'Scripts').mkdir(parents=True)
    (root / 'venv' / 'Scripts' / 'python.exe').write_text('fake')
    (root / 'run.py').write_text('fake')
    cfg.save_config({'aitoolkit': {'dir': str(root)}})
    return root


def test_probe_joycaption_false_when_deps_missing(app, tmp_path, monkeypatch):
    """Issue #6: the ai-toolkit venv exists but can't import transformers etc. —
    probe must report NOT available and name the exact pip fix, not lie 'ready'."""
    with app.app_context():
        from app import capabilities, config
        _configure_aitoolkit_ok(config, tmp_path)
        monkeypatch.setattr(capabilities, '_import_ok', lambda *a, **k: False)  # deps absent
        capabilities._import_cache.clear()
        result = capabilities.probe_joycaption()
    assert result['ok'] is False
    assert 'pip install' in result['detail']
    assert 'transformers' in result['detail'] and 'bitsandbytes' in result['detail']


def test_probe_joycaption_true_when_deps_import(app, tmp_path, monkeypatch):
    with app.app_context():
        from app import capabilities, config
        _configure_aitoolkit_ok(config, tmp_path)
        monkeypatch.setattr(capabilities, '_import_ok', lambda *a, **k: True)  # deps present
        capabilities._import_cache.clear()
        result = capabilities.probe_joycaption()
    assert result == {'ok': True, 'detail': 'JoyCaption deps import OK'}


def test_probe_joycaption_false_when_aitoolkit_unconfigured(app):
    """No ai-toolkit at all: name THAT gap, never a misleading deps message, and
    never spawn the import subprocess (returns before the seam)."""
    with app.app_context():
        from app import capabilities
        result = capabilities.probe_joycaption()
    assert result['ok'] is False
    assert 'pip install' not in result['detail']


def test_probe_joycaption_import_result_is_cached(app, tmp_path, monkeypatch):
    """Uses the same cached seam as the other import-probes — no per-call subprocess."""
    with app.app_context():
        from app import capabilities, config
        _configure_aitoolkit_ok(config, tmp_path)
        calls = []
        monkeypatch.setattr(capabilities, '_import_ok', lambda *a, **k: calls.append(1) or True)
        capabilities._import_cache.clear()
        capabilities.probe_joycaption()
        capabilities.probe_joycaption()
    assert len(calls) == 1


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
    compiling numba caches) must report not-ready NOW and must not poison the
    10 min cache — but it is remembered BRIEFLY (_UNKNOWN_TTL), because the
    caller polls: an uncached unknown re-spawned a 90 s subprocess on every
    single call. A real import error (False) stays cached for the long TTL."""
    with app.app_context():
        from app import capabilities
        calls = []
        monkeypatch.setattr(capabilities, '_import_ok',
                            lambda *a, **k: calls.append(1) or None)   # timeout
        capabilities._import_cache.clear()
        assert capabilities.probe_masks()['ok'] is False
        assert capabilities.probe_masks()['ok'] is False
        assert len(calls) == 1                       # the unknown is remembered…
        # …but only briefly: past _UNKNOWN_TTL it re-tries, against what is by
        # then a warm import. Age the entry rather than sleep.
        for k, (ts, ok) in list(capabilities._import_cache.items()):
            capabilities._import_cache[k] = (ts - capabilities._UNKNOWN_TTL - 1, ok)
        assert capabilities.probe_masks()['ok'] is False
        assert len(calls) == 2                       # re-probed after the short TTL
        assert capabilities._UNKNOWN_TTL < capabilities._IMPORT_TTL
        monkeypatch.setattr(capabilities, '_import_ok',
                            lambda *a, **k: calls.append(1) or False)  # real failure
        for k, (ts, ok) in list(capabilities._import_cache.items()):
            capabilities._import_cache[k] = (ts - capabilities._UNKNOWN_TTL - 1, ok)
        assert capabilities.probe_masks()['ok'] is False
        assert capabilities.probe_masks()['ok'] is False
        assert len(calls) == 3                       # cached after the real False


def test_import_probe_budget_matches_the_scoring_probe(app):
    """60 s was not enough for a cold `import torch` behind an antivirus — the
    repo says so itself in scoring_python.PROBE_TIMEOUT (90 s), and the two
    probes disagreeing about the SAME interpreter is exactly the bug."""
    from app import capabilities
    from app.services import scoring_python
    assert capabilities._IMPORT_TIMEOUT >= scoring_python.PROBE_TIMEOUT


def test_siglip2_probe_checks_pinned_weights_before_heavy_import(app, monkeypatch):
    from app import capabilities
    from app.services import bank_semantic_models as assets
    calls = []
    monkeypatch.setattr(assets, 'weights_present', lambda: False)
    monkeypatch.setattr(
        capabilities, '_cached_import',
        lambda *args, **kwargs: calls.append(args) or True)
    with app.app_context():
        result = capabilities.probe_bank_siglip2()
    assert result['ok'] is False
    assert result['model'] == assets.MODEL_ID
    assert calls == []


def test_siglip2_probe_uses_its_own_import_contract(app, monkeypatch):
    from app import capabilities, config
    from app.services import bank_semantic_models as assets
    seen = []
    monkeypatch.setattr(assets, 'weights_present', lambda: True)
    monkeypatch.setattr(
        capabilities, '_cached_import',
        lambda name, python, expression: seen.append(
            (name, python, expression)) or True)
    with app.app_context():
        config.save_config({
            'bank_scoring': {'python': '/borrowed/score/python'},
            'bank_semantic': {'python': '/managed/semantic/python'},
        })
        result = capabilities.probe_bank_siglip2()
    assert result['ok'] is True
    assert seen == [(
        'bank_siglip2', '/managed/semantic/python',
        capabilities.CAPABILITY_IMPORTS['bank_siglip2'])]


def test_score_probe_remains_scoped_to_score_python(app, monkeypatch):
    from app import capabilities, config
    seen = []
    monkeypatch.setattr(
        capabilities, '_cached_import',
        lambda name, python, expression: seen.append((name, python)) or True)
    with app.app_context():
        config.save_config({
            'bank_scoring': {'python': '/borrowed/score/python'},
            'bank_semantic': {'python': '/managed/semantic/python'},
        })
        assert capabilities.probe_bank_scoring()['ok'] is True
    assert seen == [('bank_scoring', '/borrowed/score/python')]


def test_import_capabilities_disable_the_user_site(
        app, monkeypatch, _no_real_subprocess):
    """Every probe that vouches for a backend/infer worker asks the question in
    the environment that worker will run in: no user site-packages, by argv AND
    by inherited environment.

    A borrowed ComfyUI Python must be checked with its own packages only. The
    user site-packages directory is shared by every interpreter of that Python
    version on the machine and sits AHEAD of the interpreter's own — one
    unrelated `pip install --user` shadows a package our worker needs."""
    from types import SimpleNamespace
    from app import capabilities
    seen = []

    def run(command, **kwargs):
        seen.append((command, kwargs.get('env')))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(capabilities.subprocess, 'run', run)
    monkeypatch.setattr(capabilities, '_import_ok', _no_real_subprocess)
    python = r'C:\ComfyUI\python_embeded\python.exe'

    for key in ('bank_scoring', 'masks', 'face_scoring', 'watermark_detect',
                'shot_detect', 'video_text', 'bank_siglip2', 'joycaption'):
        capabilities._import_cache.clear()
        seen.clear()
        assert capabilities._cached_import_state(key, python, 'import x') is True
        command, env = seen[0]
        assert command[:3] == [python, '-s', '-c'], key
        assert (env or {}).get('PYTHONNOUSERSITE') == '1', key


def test_the_probes_that_vouch_for_someone_elses_environment_leave_it_alone(
        app, monkeypatch, _no_real_subprocess):
    """The exceptions, and why each is one.

    ai-toolkit's venv is the USER's — the training bridge launches it as it
    stands, so a probe that sanitised it would answer about an environment that
    never runs. And 'video_decode' asks the app's OWN interpreter for PyAV,
    which Flask imports in-process, with no `-s` of its own to match.

    A probe must ask the question the real launch will ask. That cuts both
    ways, which is the whole reason this test sits next to the one above."""
    from types import SimpleNamespace
    from app import capabilities
    seen = []

    def run(command, **kwargs):
        seen.append((command, kwargs.get('env')))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(capabilities.subprocess, 'run', run)
    monkeypatch.setattr(capabilities, '_import_ok', _no_real_subprocess)
    python = r'C:\ai-toolkit\venv\Scripts\python.exe'

    for key in ('aitoolkit_torch', 'aitoolkit_alive', 'video_decode'):
        capabilities._import_cache.clear()
        seen.clear()
        assert capabilities._cached_import_state(key, python, 'import x') is True
        command, env = seen[0]
        assert command[:2] == [python, '-c'], key
        assert env is None, key


def test_siglip2_cuda_probe_uses_semantic_python_and_unknown_means_cpu(
        app, monkeypatch):
    from app import capabilities, config
    seen = []

    def state(name, python, expression):
        seen.append((name, python, expression))
        return None

    monkeypatch.setattr(capabilities, '_cached_import_state', state)
    monkeypatch.setattr(capabilities, 'gpu_vram_gb', lambda: 24.0)
    with app.app_context():
        config.save_config({
            'bank_scoring': {'python': '/borrowed/gpu/python'},
            'bank_semantic': {'python': '/managed/cpu/python'},
        })
        assert capabilities.bank_siglip2_gpu_available() is False
    assert len(seen) == 1
    assert seen[0][0:2] == ('bank_siglip2_gpu', '/managed/cpu/python')


def test_unanswered_cuda_probe_does_not_read_as_no_cuda(app, monkeypatch):
    """The GPU-exclusive window is decided by bank_scoring_gpu_available(). A
    probe that never answered must not be reported as 'this interpreter has no
    CUDA': the scoring child picks cuda on its own, so on a machine that HAS a
    card the window has to be taken until we know better. A card-less machine
    still gets False — it can never use a window."""
    with app.app_context():
        from app import capabilities
        monkeypatch.setattr(capabilities, '_import_ok', lambda *a, **k: None)  # timeout
        capabilities._import_cache.clear()
        monkeypatch.setattr(capabilities, 'gpu_vram_gb', lambda: 24.0)
        assert capabilities.bank_scoring_gpu_available() is True
        capabilities._import_cache.clear()
        monkeypatch.setattr(capabilities, 'gpu_vram_gb', lambda: None)
        assert capabilities.bank_scoring_gpu_available() is False
        # …and a probe that DID answer still rules, card or no card
        capabilities._import_cache.clear()
        monkeypatch.setattr(capabilities, '_import_ok', lambda *a, **k: False)
        monkeypatch.setattr(capabilities, 'gpu_vram_gb', lambda: 24.0)
        assert capabilities.bank_scoring_gpu_available() is False


def test_unanswered_cuda_probe_is_not_re_spawned_on_every_poll(app, monkeypatch):
    """The Bank panel asks for the score device every ~2 s. Each unanswered
    probe used to cost a fresh 90 s `import torch` subprocess."""
    with app.app_context():
        from app import capabilities
        calls = []
        monkeypatch.setattr(capabilities, '_import_ok',
                            lambda *a, **k: calls.append(1) or None)
        monkeypatch.setattr(capabilities, 'gpu_vram_gb', lambda: None)
        capabilities._import_cache.clear()
        for _ in range(5):
            capabilities.bank_scoring_gpu_available()
        assert len(calls) == 1


def test_import_probe_cache_key_includes_interpreter_path(app, monkeypatch):
    """Changing interpreter path should invalidate the import cache."""
    with app.app_context():
        from app import capabilities, config

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


def test_ollama_requested_model_ready_when_global_is_absent(app, monkeypatch):
    """A per-dataset model installed on its own is ready even if the global model
    is not pulled; omitting the override still checks the global model."""
    with app.app_context():
        from app import capabilities, config
        config.save_config({
            'ollama': {'url': 'http://o', 'vision_model': 'global-vlm:latest'}})
        monkeypatch.setattr(capabilities, '_http_ok', lambda *a, **k: True)
        monkeypatch.setattr(
            capabilities, '_ollama_tags', lambda *a, **k: ['custom-vlm:latest'])

        custom = capabilities.probe_ollama_model(model='custom-vlm:latest')
        global_default = capabilities.probe_ollama_model()

    assert custom == {'ok': True, 'detail': 'custom-vlm:latest ready'}
    assert global_default == {'ok': False, 'detail': 'global-vlm:latest not pulled'}


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


# --- issue #7: /api/tags shape robustness + honest, unified probe ----------
# boorad64 had huihui_ai/qwen3-vl-abliterated:8b-instruct genuinely pulled (LDS
# pulled it in Setup step 3, `describe` worked) yet the diagnostic read
# vision_model=no. The model IS present in /api/tags; the probe just couldn't
# recognise it once the /api/tags entry differed from a byte-exact `name` match.

_ABLIT = 'huihui_ai/qwen3-vl-abliterated:8b-instruct'


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status
    def json(self):
        return self._payload


def test_ollama_tags_reads_name_and_model_fields(app, monkeypatch):
    """Some Ollama builds populate `model` but leave `name` blank. Reading a single
    field turned a pulled model into a blank -> 'not pulled'. Both fields are read
    now, de-duplicated."""
    with app.app_context():
        from app import capabilities
        payload = {'models': [
            {'name': '', 'model': 'gemma4:e2b-it-q4_K_M'},
            {'name': '', 'model': 'qwen3-vl:8b-instruct'},
            {'name': '', 'model': _ABLIT},
        ]}
        monkeypatch.setattr(capabilities.requests, 'get', lambda *a, **k: _FakeResp(payload))
        tags = capabilities._ollama_tags('http://o')
    assert _ABLIT in tags
    assert capabilities._model_present(_ABLIT, tags) is True


def test_model_present_namespaced_exact_boorad_list(app):
    """boorad's exact 3-model list, spec-compliant shape: the namespaced model is
    recognised (this is the shape that already worked — the regression guard)."""
    with app.app_context():
        from app import capabilities
        names = ['gemma4:e2b-it-q4_K_M', 'qwen3-vl:8b-instruct', _ABLIT]
        assert capabilities._model_present(_ABLIT, names) is True


def test_model_present_registry_host_prefixed(app):
    """/api/tags entries carrying a registry host prefix
    (registry.ollama.ai/…) must still match the namespaced config string."""
    with app.app_context():
        from app import capabilities
        names = [f'registry.ollama.ai/{_ABLIT}', 'registry.ollama.ai/library/qwen3-vl:8b-instruct']
        assert capabilities._model_present(_ABLIT, names) is True


def test_model_present_no_false_positive_vanilla_vs_abliterated(app):
    """Probes never lie the OTHER way: the vanilla qwen3-vl must NOT satisfy a config
    that asks for the abliterated build — the difference is in the model NAME, which
    normalization preserves."""
    with app.app_context():
        from app import capabilities
        assert capabilities._model_present(_ABLIT, ['qwen3-vl:8b-instruct']) is False
        # different publisher of a same-suffix name never collides either
        assert capabilities._model_present(_ABLIT, ['someoneelse/qwen3-vl:8b-instruct']) is False


def test_model_present_implicit_latest_tag(app):
    """':latest' is implicit both ways: config 'x' matches tag 'x:latest' and back."""
    with app.app_context():
        from app import capabilities
        assert capabilities._model_present('llava', ['llava:latest']) is True
        assert capabilities._model_present('llava:latest', ['llava']) is True


def test_probe_ollama_connection_unifies_test_button_with_model_probe(app, monkeypatch):
    """The Settings 'Test' target now shares probe_ollama_model, so it can no longer
    show green while the Setup/diagnostic say the model isn't pulled (issue #7)."""
    with app.app_context():
        from app import capabilities, config
        config.save_config({'ollama': {'url': 'http://o', 'vision_model': _ABLIT}})
        monkeypatch.setattr(capabilities, '_http_ok', lambda *a, **k: True)
        # reachable but model absent -> the Test button is HONEST (not green)
        monkeypatch.setattr(capabilities, '_ollama_tags', lambda *a, **k: ['llama3:8b'])
        assert capabilities.probe_ollama_connection()['ok'] is False
        # reachable + present -> green, and it's the model probe that said so
        monkeypatch.setattr(capabilities, '_ollama_tags', lambda *a, **k: [_ABLIT])
        assert capabilities.probe_ollama_connection()['ok'] is True
        # unreachable -> surfaces the reachability failure verbatim
        monkeypatch.setattr(capabilities, '_http_ok', lambda *a, **k: False)
        assert capabilities.probe_ollama_connection()['ok'] is False


def test_ollama_diagnostic_exposes_configured_and_seen_tags(app, monkeypatch):
    """The diagnostic snapshot carries the configured model string AND the tags the
    probe actually sees, so a 'vision_model=no' report is self-diagnosing."""
    with app.app_context():
        from app import capabilities, config
        config.save_config({'ollama': {'url': 'http://o', 'vision_model': _ABLIT}})
        monkeypatch.setattr(capabilities, '_ollama_tags',
                            lambda *a, **k: ['gemma4:e2b-it-q4_K_M', 'qwen3-vl:8b-instruct', _ABLIT])
        diag = capabilities.ollama_diagnostic()
    assert diag['vision_model'] == _ABLIT
    assert _ABLIT in diag['tags_seen']


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


# --- ollama install detection (execution-independent) ---------------------

def test_ollama_binary_found_on_path(app, monkeypatch):
    """shutil.which hit is the primary signal — works whether or not the server runs."""
    from app import capabilities
    monkeypatch.setattr(capabilities.shutil, 'which', lambda name: r'C:\bin\ollama.exe')
    assert capabilities._ollama_binary() == r'C:\bin\ollama.exe'


def test_ollama_binary_windows_localappdata_fallback(app, tmp_path, monkeypatch):
    """Not on PATH (stale shell) but present at the official per-user install
    location %LOCALAPPDATA%\\Programs\\Ollama\\ollama.exe -> still detected."""
    from app import capabilities
    monkeypatch.setattr(capabilities.shutil, 'which', lambda name: None)
    monkeypatch.setattr(capabilities.os, 'name', 'nt')
    exe = tmp_path / 'Programs' / 'Ollama' / 'ollama.exe'
    exe.parent.mkdir(parents=True)
    exe.touch()
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    assert capabilities._ollama_binary() == str(exe)


def test_ollama_binary_absent_returns_empty(app, monkeypatch):
    from app import capabilities
    monkeypatch.setattr(capabilities.shutil, 'which', lambda name: None)
    monkeypatch.setattr(capabilities.os, 'name', 'nt')
    monkeypatch.setenv('LOCALAPPDATA', r'C:\nope-nonexistent-xyz')
    assert capabilities._ollama_binary() == ''
    assert capabilities.probe_ollama_installed()['ok'] is False


def test_probe_exposes_ollama_installed_independent_of_reachable(app, monkeypatch):
    """installed can be True while reachable is False — the whole point: an
    installed-but-stopped Ollama must NOT read as absent."""
    with app.app_context():
        from app import capabilities, config
        config.save_config({'ollama': {'url': 'http://o', 'vision_model': 'qwen3-vl:8b'}})
        monkeypatch.setattr(capabilities, '_http_ok', lambda *a, **k: False)   # server down
        monkeypatch.setattr(capabilities, '_ollama_binary', lambda: r'C:\bin\ollama.exe')
        caps = capabilities.probe(force=True)
    o = caps['ollama']
    assert o['installed'] is True
    assert o['reachable'] is False
    assert o['binary_path'] == r'C:\bin\ollama.exe'


def test_probe_ollama_installed_false_when_binary_missing(app, monkeypatch):
    with app.app_context():
        from app import capabilities
        monkeypatch.setattr(capabilities, '_http_ok', lambda *a, **k: False)
        monkeypatch.setattr(capabilities, '_ollama_binary', lambda: '')
        caps = capabilities.probe(force=True)
    assert caps['ollama']['installed'] is False
    assert caps['ollama']['binary_path'] == ''


# --- classify_comfyui_dir: rich, actionable verdicts (Setup Volet 1) ------

def test_classify_comfyui_dir_valid(tmp_path):
    from app.capabilities import classify_comfyui_dir
    _make_comfyui(tmp_path)
    r = classify_comfyui_dir(str(tmp_path))
    assert r['status'] == 'valid'
    assert pathlib.Path(r['resolved']) == tmp_path
    assert r['suggestion'] == ''


def test_classify_comfyui_dir_nested_proposes_child(tmp_path):
    """The launcher/parent-folder mistake: <folder>/ComfyUI is the real install, so
    the verdict proposes that child for the UI to adopt with one click."""
    from app.capabilities import classify_comfyui_dir
    _make_comfyui(tmp_path / 'ComfyUI')
    r = classify_comfyui_dir(str(tmp_path))
    assert r['status'] == 'nested'
    assert pathlib.Path(r['resolved']) == tmp_path / 'ComfyUI'
    assert pathlib.Path(r['suggestion']) == tmp_path / 'ComfyUI'


def test_classify_comfyui_dir_missing(tmp_path):
    from app.capabilities import classify_comfyui_dir
    r = classify_comfyui_dir(str(tmp_path / 'nope'))
    assert r['status'] == 'missing' and r['suggestion'] == ''


def test_classify_comfyui_dir_empty_folder(tmp_path):
    from app.capabilities import classify_comfyui_dir
    empty = tmp_path / 'empty'
    empty.mkdir()
    r = classify_comfyui_dir(str(empty))
    assert r['status'] == 'empty_dir'


def test_classify_comfyui_dir_random_folder(tmp_path):
    """Exists, has content, but no main.py/models and no child ComfyUI -> not_comfyui."""
    from app.capabilities import classify_comfyui_dir
    rnd = tmp_path / 'downloads'
    (rnd / 'stuff').mkdir(parents=True)
    (rnd / 'note.txt').write_text('hi')
    r = classify_comfyui_dir(str(rnd))
    assert r['status'] == 'not_comfyui'


def test_classify_comfyui_dir_blank():
    from app.capabilities import classify_comfyui_dir
    assert classify_comfyui_dir('') == {'status': 'empty', 'resolved': '', 'suggestion': '',
                                        # nothing typed -> nothing to probe (see
                                        # test_comfy_input_folder_handoff.py)
                                        'input_check': {'path': '', 'ok': None, 'problem': ''}}
    assert classify_comfyui_dir('   ')['status'] == 'empty'


# --- comfyui.skipped: derived, self-annulling, never masks a real error ----

def test_probe_skipped_true_when_flag_set_and_no_dir(app, monkeypatch):
    with app.app_context():
        from app import capabilities, config
        config.save_config({'comfyui': {'setup_skipped': True, 'base_dir': ''}})
        with patch('app.capabilities._http_ok', return_value=False):
            caps = capabilities.probe(force=True)
    assert caps['comfyui']['skipped'] is True


def test_probe_skip_annulled_when_dir_configured(app, monkeypatch, tmp_path):
    """A configured directory ANNULS the skip on the spot (derived), so the flag can
    never hide a real error of a set-up ComfyUI."""
    with app.app_context():
        from app import capabilities, config
        _make_comfyui(tmp_path / 'ComfyUI')
        config.save_config({'comfyui': {'setup_skipped': True, 'base_dir': str(tmp_path / 'ComfyUI')}})
        with patch('app.capabilities._http_ok', return_value=False):
            caps = capabilities.probe(force=True)
    assert caps['comfyui']['skipped'] is False
    assert caps['comfyui']['dir_valid'] is True     # the real verdict still surfaces


def test_probe_skipped_false_by_default(app):
    with app.app_context():
        from app import capabilities
        with patch('app.capabilities._http_ok', return_value=False):
            caps = capabilities.probe(force=True)
    assert caps['comfyui']['skipped'] is False


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
