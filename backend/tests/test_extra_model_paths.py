"""Home-grown extra_model_paths.yaml support.

Pins the module to ComfyUI's OFFICIAL semantics (utils/extra_config.py +
folder_paths.py): arbitrary profile names, base_path (with ~/$VAR expansion and
yaml-dir-relative resolution), is_default ordering, multi-line block values,
map_legacy aliases, relative-name-from-root output. Plus the app wiring: Klein
resolution / _scan_models / setup skip-download become extra-paths-aware WITHOUT
changing anything when no yaml exists (the explicit no-break mandate), and the
degradation paths (absent / empty / malformed / PyYAML missing) never raise.
"""
import os
import textwrap

import pytest


@pytest.fixture(autouse=True)
def _clear_cmp_cache():
    from app.services import comfy_model_paths as cmp
    cmp.clear_cache()
    cmp._warned.clear()
    yield
    cmp.clear_cache()
    cmp._warned.clear()


def _comfy_base(tmp_path, cfg):
    """Minimal valid ComfyUI base with base_dir set. Returns the base Path."""
    base = tmp_path / 'ComfyUI'
    (base / 'models').mkdir(parents=True)
    (base / 'main.py').write_text('# fake', encoding='utf-8')
    cfg.save_config({'comfyui': {'base_dir': str(base)}})
    return base


def _write_yaml(base, text):
    (base / 'extra_model_paths.yaml').write_text(textwrap.dedent(text), encoding='utf-8')


def _touch(*parts):
    p = os.path.join(*parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8') as fh:
        fh.write('x')
    return p


# --- Parser: canonical profile + base_path + relatives -----------------------
def test_canonical_profile_with_base_path_and_relatives(app, tmp_path):
    from app import config as cfg
    from app.services import comfy_model_paths as cmp
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        root = tmp_path / 'external'
        (root / 'checkpoints').mkdir(parents=True)
        _write_yaml(base, f"""
            comfyui:
              base_path: {root}
              checkpoints: checkpoints
              loras: loras
        """)
        roots = cmp.search_roots('checkpoints')
        # default <base>/models/checkpoints first, then the extra (relative to base_path).
        assert roots[0] == os.path.normpath(str(base / 'models' / 'checkpoints'))
        assert os.path.normpath(str(root / 'checkpoints')) in roots
        assert cmp.extra_roots('loras') == [os.path.normpath(str(root / 'loras'))]


def test_base_path_relative_resolves_against_yaml_dir(app, tmp_path):
    from app import config as cfg
    from app.services import comfy_model_paths as cmp
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        _write_yaml(base, """
            comfyui:
              base_path: shared
              vae: vae
        """)
        # relative base_path resolves against the yaml's own directory (= base), NOT cwd.
        assert cmp.extra_roots('vae') == [os.path.normpath(str(base / 'shared' / 'vae'))]


def test_flat_absolute_profile_stability_matrix(app, tmp_path):
    """The Stability-Matrix shape: a profile with absolute per-type paths and NO
    base_path — a plain canonical profile, no special branch needed."""
    from app import config as cfg
    from app.services import comfy_model_paths as cmp
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        sm = tmp_path / 'SM' / 'Models'
        _write_yaml(base, f"""
            stability_matrix:
              checkpoints: {sm / 'StableDiffusion'}
              loras: {sm / 'Lora'}
              vae: {sm / 'VAE'}
        """)
        assert cmp.extra_roots('checkpoints') == [os.path.normpath(str(sm / 'StableDiffusion'))]
        assert cmp.extra_roots('loras') == [os.path.normpath(str(sm / 'Lora'))]
        assert cmp.extra_roots('vae') == [os.path.normpath(str(sm / 'VAE'))]


# --- Multi-profile + is_default + multi-line ---------------------------------
def test_multiple_profiles_merge(app, tmp_path):
    from app import config as cfg
    from app.services import comfy_model_paths as cmp
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        a, b = tmp_path / 'a', tmp_path / 'b'
        _write_yaml(base, f"""
            comfyui:
              loras: {a / 'loras'}
            a111:
              loras: {b / 'loras'}
        """)
        roots = cmp.extra_roots('loras')
        assert os.path.normpath(str(a / 'loras')) in roots
        assert os.path.normpath(str(b / 'loras')) in roots


def test_is_default_inserts_at_front(app, tmp_path):
    from app import config as cfg
    from app.services import comfy_model_paths as cmp
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        ea, eb = tmp_path / 'ea', tmp_path / 'eb'
        _write_yaml(base, f"""
            primary:
              is_default: true
              checkpoints: {ea}
            secondary:
              checkpoints: {eb}
        """)
        roots = cmp.search_roots('checkpoints')
        default_base = os.path.normpath(str(base / 'models' / 'checkpoints'))
        # is_default root is FIRST (highest priority), base next, non-default extra last.
        assert roots[0] == os.path.normpath(str(ea))
        assert roots == [os.path.normpath(str(ea)), default_base, os.path.normpath(str(eb))]


def test_multiline_block_value_is_split(app, tmp_path):
    from app import config as cfg
    from app.services import comfy_model_paths as cmp
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        l1, l2 = tmp_path / 'l1', tmp_path / 'l2'
        _write_yaml(base, f"""
            comfyui:
              loras: |
                {l1}
                {l2}
        """)
        roots = cmp.extra_roots('loras')
        assert os.path.normpath(str(l1)) in roots
        assert os.path.normpath(str(l2)) in roots


# --- Aliases (map_legacy) ----------------------------------------------------
def test_legacy_aliases_unet_and_clip(app, tmp_path):
    from app import config as cfg
    from app.services import comfy_model_paths as cmp
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        u, c = tmp_path / 'u', tmp_path / 'c'
        _write_yaml(base, f"""
            comfyui:
              unet: {u}
              clip: {c}
        """)
        # unet folds to diffusion_models, clip to text_encoders — queryable by either name.
        assert os.path.normpath(str(u)) in cmp.extra_roots('diffusion_models')
        assert os.path.normpath(str(u)) in cmp.extra_roots('unet')
        assert os.path.normpath(str(c)) in cmp.extra_roots('text_encoders')
        assert os.path.normpath(str(c)) in cmp.extra_roots('clip')


# --- Expansion: ~ and $VAR ---------------------------------------------------
def test_base_path_expands_env_var(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import comfy_model_paths as cmp
    monkeypatch.setenv('LDS_TEST_ROOT', str(tmp_path / 'envroot'))
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        _write_yaml(base, """
            comfyui:
              base_path: $LDS_TEST_ROOT/models
              checkpoints: ckpts
        """)
        assert cmp.extra_roots('checkpoints') == [
            os.path.normpath(str(tmp_path / 'envroot' / 'models' / 'ckpts'))]


def test_base_path_expands_home(app, tmp_path):
    from app import config as cfg
    from app.services import comfy_model_paths as cmp
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        _write_yaml(base, """
            comfyui:
              base_path: ~/lds_extra_paths_test
              vae: v
        """)
        (root,) = cmp.extra_roots('vae')
        assert root.startswith(os.path.normpath(os.path.expanduser('~')))
        assert root.endswith(os.path.normpath('lds_extra_paths_test/v'))


# --- list_models: relative name from a root, subfolders included -------------
def test_list_models_returns_relative_names_with_subfolders(app, tmp_path):
    from app import config as cfg
    from app.services import comfy_model_paths as cmp
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        ext = tmp_path / 'extdiff'
        _touch(str(ext), 'klein', 'x.safetensors')
        _touch(str(ext), 'root.safetensors')
        _write_yaml(base, f"""
            comfyui:
              diffusion_models: {ext}
        """)
        names = dict(cmp.list_models('diffusion_models'))
        # subfolder file -> 'klein\\x.safetensors'; root file -> 'root.safetensors'.
        assert os.path.join('klein', 'x.safetensors') in names
        assert 'root.safetensors' in names


def test_list_models_extension_policy_sft_yes_ckpt_no(app, tmp_path):
    """.safetensors/.sft/.gguf are listed; official-but-unused .ckpt is not (see
    module docstring: narrowed to what the app's loaders consume, .gguf kept)."""
    from app import config as cfg
    from app.services import comfy_model_paths as cmp
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        ext = tmp_path / 'extvae'
        for fn in ('a.safetensors', 'b.sft', 'c.gguf', 'd.ckpt', 'e.pt'):
            _touch(str(ext), fn)
        _write_yaml(base, f"""
            comfyui:
              vae: {ext}
        """)
        names = {rel for rel, _ in cmp.list_models('vae')}
        assert {'a.safetensors', 'b.sft', 'c.gguf'} <= names
        assert 'd.ckpt' not in names and 'e.pt' not in names


# --- Degradation: absent / empty / malformed / dangerous / no-pyyaml ---------
def test_no_yaml_means_defaults_only(app, tmp_path):
    from app import config as cfg
    from app.services import comfy_model_paths as cmp
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)   # no yaml written
        assert cmp.extra_roots('checkpoints') == []
        assert cmp.search_roots('checkpoints') == [
            os.path.normpath(str(base / 'models' / 'checkpoints'))]
        assert cmp.search_roots('diffusion_models') == [
            os.path.normpath(str(base / 'models' / 'unet')),
            os.path.normpath(str(base / 'models' / 'diffusion_models'))]


def test_empty_yaml_degrades(app, tmp_path):
    from app import config as cfg
    from app.services import comfy_model_paths as cmp
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        _write_yaml(base, '')
        assert cmp.extra_roots('loras') == []


def test_malformed_yaml_degrades_without_raising(app, tmp_path):
    from app import config as cfg
    from app.services import comfy_model_paths as cmp
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        (base / 'extra_model_paths.yaml').write_text('checkpoints: [unclosed\n', encoding='utf-8')
        assert cmp.extra_roots('checkpoints') == []   # no exception, empty


def test_dangerous_python_tag_is_not_executed(app, tmp_path):
    """safe_load refuses !!python tags: the file fails to parse (degrades to {})
    and NOTHING is constructed/executed — proof we never fall back to full load."""
    from app import config as cfg
    from app.services import comfy_model_paths as cmp
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        (base / 'extra_model_paths.yaml').write_text(
            'comfyui:\n  checkpoints: !!python/object/apply:os.system ["echo pwned"]\n',
            encoding='utf-8')
        assert cmp.extra_roots('checkpoints') == []


def test_pyyaml_missing_disables_feature(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import comfy_model_paths as cmp
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        _write_yaml(base, f"""
            comfyui:
              loras: {tmp_path / 'x'}
        """)
        monkeypatch.setattr(cmp, '_yaml', None)   # simulate pyyaml not installed
        cmp.clear_cache()
        assert cmp.extra_roots('loras') == []
        # search_roots still returns the base defaults (feature just off).
        assert cmp.search_roots('loras') == [os.path.normpath(str(base / 'models' / 'loras'))]


# --- Cache invalidates on mtime ----------------------------------------------
def test_cache_invalidates_on_mtime(app, tmp_path):
    from app import config as cfg
    from app.services import comfy_model_paths as cmp
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        first, second = tmp_path / 'first', tmp_path / 'second'
        _write_yaml(base, f"comfyui:\n  loras: {first}\n")
        assert cmp.extra_roots('loras') == [os.path.normpath(str(first))]
        # Rewrite with a bumped mtime -> the (path, mtime) key changes -> re-parsed.
        yaml_file = base / 'extra_model_paths.yaml'
        yaml_file.write_text(f"comfyui:\n  loras: {second}\n", encoding='utf-8')
        st = yaml_file.stat()
        os.utime(str(yaml_file), (st.st_atime + 5, st.st_mtime + 5))
        assert cmp.extra_roots('loras') == [os.path.normpath(str(second))]


# --- Klein wiring: resolve from an extra path, no false positives ------------
def test_klein_resolves_entirely_from_extra_paths(app, tmp_path):
    """All four Klein assets live OUTSIDE <base>/models, registered via the yaml.
    Resolution must find them (unet keeps its 'klein\\' subfolder prefix, the LoRA
    keeps its relative 'klein\\' name) and klein_missing_assets must be empty — so
    the generate route neither 409s nor re-downloads."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        ext = tmp_path / 'ext'
        _touch(str(ext), 'unet', 'klein', 'flux-2-klein-9b-fp8.safetensors')
        _touch(str(ext), 'vae', 'flux2-vae.safetensors')
        _touch(str(ext), 'text_encoders', 'qwen_3_8b_fp8mixed.safetensors')
        _touch(str(ext), 'loras', 'klein', 'Flux2-Klein-9B-consistency-V2.safetensors')
        _write_yaml(base, f"""
            comfyui:
              diffusion_models: {ext / 'unet'}
              vae: {ext / 'vae'}
              text_encoders: {ext / 'text_encoders'}
              loras: {ext / 'loras'}
        """)
        assert keh.resolve_klein_unet() == os.path.join('klein', 'flux-2-klein-9b-fp8.safetensors')
        assert keh.resolve_klein_vae() == 'flux2-vae.safetensors'
        assert keh.resolve_klein_text_encoder() == 'qwen_3_8b_fp8mixed.safetensors'
        name, path = keh._consistency_lora()
        assert name == os.path.join('klein', 'Flux2-Klein-9B-consistency-V2.safetensors')
        assert path and os.path.isfile(path)
        assert keh.klein_missing_assets() == []


def test_scan_models_includes_extra_klein_without_false_positive(app, tmp_path):
    """An extra diffusion_models root feeds the Klein picker; an extra LORAS root
    whose subfolder happens to be named 'klein' must NOT leak into the unet bucket
    (the loras root is not a diffusion root)."""
    from app import config as cfg, capabilities
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        extdiff, extloras = tmp_path / 'extdiff', tmp_path / 'extloras'
        _touch(str(extdiff), 'Flux2 klein', 'flux-2-klein-9b-kv-fp8.safetensors')
        _touch(str(extloras), 'klein', 'some_style_lora.safetensors')   # a LoRA, not a unet
        _write_yaml(base, f"""
            comfyui:
              diffusion_models: {extdiff}
              loras: {extloras}
        """)
        models = capabilities._scan_models()
        assert 'flux-2-klein-9b-kv-fp8.safetensors' in models['klein']
        assert 'some_style_lora.safetensors' not in models['klein']


def test_scan_models_extra_checkpoints_feed_sdxl(app, tmp_path):
    from app import config as cfg, capabilities
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        extck = tmp_path / 'extck'
        _touch(str(extck), 'bigLove_photo5.safetensors')
        _write_yaml(base, f"""
            comfyui:
              checkpoints: {extck}
        """)
        assert 'bigLove_photo5.safetensors' in capabilities._scan_models()['sdxl']


def test_setup_skip_download_when_asset_in_extra_path(app, tmp_path):
    """The Klein download worker skips (rc 0, no fetch) when the canonical file is
    already present under an extra_model_paths root — no forced re-download."""
    from app import config as cfg, setup_installer
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        ext = tmp_path / 'ext'
        _touch(str(ext), 'vae', 'flux2-vae.safetensors')
        _touch(str(ext), 'unet', 'klein', 'flux-2-klein-9b-fp8.safetensors')
        _write_yaml(base, f"""
            comfyui:
              vae: {ext / 'vae'}
              diffusion_models: {ext / 'unet'}
        """)
        assert setup_installer._klein_present_in_extra('klein_vae') is True
        assert setup_installer._klein_present_in_extra('klein_model') is True
        # text-encoder NOT provided -> still needs downloading.
        assert setup_installer._klein_present_in_extra('klein_text_encoder') is False


# --- The no-break mandate: full Klein resolution WITHOUT yaml == today --------
def test_regression_full_klein_resolution_without_yaml_unchanged(app, tmp_path):
    """No extra_model_paths.yaml: every Klein resolver returns EXACTLY the values it
    returned before this feature existed (standard <base>/models layout)."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        _touch(str(base), 'models', 'unet', 'klein', 'flux-2-klein-9b-fp8.safetensors')
        _touch(str(base), 'models', 'vae', 'flux2-vae.safetensors')
        _touch(str(base), 'models', 'text_encoders', 'qwen_3_8b_fp8mixed.safetensors')
        _touch(str(base), 'models', 'loras', 'klein', 'Flux2-Klein-9B-consistency-V2.safetensors')
        cfg.save_config({'klein': {'consistency_lora': 'klein/Flux2-Klein-9B-consistency-V2.safetensors'}})
        assert not (base / 'extra_model_paths.yaml').exists()
        assert keh.resolve_klein_unet() == os.path.join('klein', 'flux-2-klein-9b-fp8.safetensors')
        assert keh.resolve_klein_vae() == 'flux2-vae.safetensors'
        assert keh.resolve_klein_text_encoder() == 'qwen_3_8b_fp8mixed.safetensors'
        name, path = keh._consistency_lora()
        assert name == os.path.join('klein', 'Flux2-Klein-9B-consistency-V2.safetensors')
        assert os.path.isfile(path)
        assert keh.klein_missing_assets() == []
