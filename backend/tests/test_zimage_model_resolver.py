"""Z-Image text-encoder / VAE resolution + per-base sampler defaults (GitHub #18).

RED BEFORE GREEN — every layout test below fails on the pre-fix code, because the
pre-fix code did not resolve anything: the shipped workflow's `z ae.safetensors`
(with a SPACE) and `Z image\\qwen_3_4b.safetensors` (with that exact capitalisation)
went to ComfyUI verbatim, so `z_ae.safetensors` and `Z Image/` were "missing".
"""
import os
import pathlib
import struct

import pytest

# A minimal, structurally VALID safetensors file (8-byte little-endian header
# length + '{}'), so model_integrity never flags these fixtures as fake weights.
_ST = struct.pack('<Q', 2) + b'{}'


def _comfy(tmp_path):
    from app import config
    base = tmp_path / 'Comfy'
    (base / 'models').mkdir(parents=True)
    config.save_config({'comfyui': {'base_dir': str(base)}})
    return base


def _write(path, data=_ST):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture(autouse=True)
def _clear_path_cache():
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    yield
    comfy_model_paths.clear_cache()


# --- The reported layouts ------------------------------------------------------

@pytest.mark.parametrize('filename', [
    'z_ae.safetensors',     # bobba84's actual file — the one he had to RENAME
    'z ae.safetensors',     # the app's own documented name
    'z-ae.safetensors',
    'Z_AE.safetensors',
])
def test_vae_resolves_whatever_the_separator_and_case(app, tmp_path, filename):
    from app.services import zimage_model_resolver as zr
    with app.app_context():
        base = _comfy(tmp_path)
        _write(base / 'models' / 'vae' / filename)
        assert zr.resolve_zimage_vae() == filename


@pytest.mark.parametrize('folder', ['Z image', 'Z Image', 'z-image', 'ZIMAGE'])
def test_text_encoder_resolves_whatever_the_folder_case(app, tmp_path, folder):
    """bobba84's report verbatim: 'the folder name is case sensitive, meaning
    Z Image did not work as I had a capital I'. The value returned carries the REAL
    on-disk folder, which is what CLIPLoader lists and validates against."""
    from app.services import zimage_model_resolver as zr
    with app.app_context():
        base = _comfy(tmp_path)
        _write(base / 'models' / 'text_encoders' / folder / 'qwen_3_4b.safetensors')
        assert zr.resolve_zimage_text_encoder() == os.path.join(folder, 'qwen_3_4b.safetensors')


def test_text_encoder_resolves_at_the_root_comfyui_documents(app, tmp_path):
    """ComfyUI's own Z-Image docs say models/text_encoders/qwen_3_4b.safetensors —
    flat, no sub-folder. That layout must work too."""
    from app.services import zimage_model_resolver as zr
    with app.app_context():
        base = _comfy(tmp_path)
        _write(base / 'models' / 'text_encoders' / 'qwen_3_4b.safetensors')
        assert zr.resolve_zimage_text_encoder() == 'qwen_3_4b.safetensors'


def test_vae_resolves_bare_ae_from_comfyui_docs_layout(app, tmp_path):
    """models/vae/ae.safetensors is what ComfyUI's Z-Image page tells people to save."""
    from app.services import zimage_model_resolver as zr
    with app.app_context():
        base = _comfy(tmp_path)
        _write(base / 'models' / 'vae' / 'ae.safetensors')
        assert zr.resolve_zimage_vae() == 'ae.safetensors'


def test_vae_prefers_a_z_qualified_name_over_the_ambiguous_bare_ae(app, tmp_path):
    """`ae.safetensors` is ALSO the FLUX.1 VAE filename. Any z-qualified candidate
    outranks it, whatever os.walk happened to yield first."""
    from app.services import zimage_model_resolver as zr
    with app.app_context():
        base = _comfy(tmp_path)
        _write(base / 'models' / 'vae' / 'ae.safetensors')
        _write(base / 'models' / 'vae' / 'z_ae.safetensors')
        assert zr.resolve_zimage_vae() == 'z_ae.safetensors'


def test_vae_prefers_the_canonical_name_over_another_z_qualified_one(app, tmp_path):
    """Two tier-0 candidates -> the documented canonical name wins. Deterministic,
    never 'whichever the filesystem listed first'."""
    from app.services import zimage_model_resolver as zr
    with app.app_context():
        base = _comfy(tmp_path)
        _write(base / 'models' / 'vae' / 'zae.safetensors')
        _write(base / 'models' / 'vae' / 'z ae.safetensors')
        _write(base / 'models' / 'vae' / 'z-ae.safetensors')
        assert zr.resolve_zimage_vae() == 'z ae.safetensors'


def test_resolution_is_stable_across_repeated_calls(app, tmp_path):
    """The tie-break is an explicit sort, so the answer cannot drift between calls."""
    from app.services import zimage_model_resolver as zr
    with app.app_context():
        base = _comfy(tmp_path)
        for n in ('z_ae.safetensors', 'z-ae.safetensors', 'zae.safetensors'):
            _write(base / 'models' / 'vae' / n)
        picks = {zr.resolve_zimage_vae() for _ in range(5)}
        assert len(picks) == 1


# --- Narrowness: a WRONG model is worse than a missing one ---------------------

def test_text_encoder_never_picks_kreas_qwen3vl_or_kleins_qwen_3_8b(app, tmp_path):
    """These three encoders share models/text_encoders/ on any multi-family install
    and produce incompatible embeddings. A loose 'contains qwen' match has already
    shipped that bug once (see klein_edit_helper)."""
    from app.services import zimage_model_resolver as zr
    with app.app_context():
        base = _comfy(tmp_path)
        te = base / 'models' / 'text_encoders'
        _write(te / 'qwen3vl_4b_fp8_scaled.safetensors')   # Krea 2
        _write(te / 'qwen_3_8b_fp8mixed.safetensors')      # Klein
        _write(te / 'qwen_2.5_vl_7b.safetensors')          # Qwen-Image
        assert zr.resolve_zimage_text_encoder() is None


def test_vae_never_picks_an_unrelated_name_ending_in_ae(app, tmp_path):
    """Phrase matching, not substring: 'xyz_ae' normalises to 'xyz ae', which does
    NOT contain the whole word 'z'."""
    from app.services import zimage_model_resolver as zr
    with app.app_context():
        base = _comfy(tmp_path)
        _write(base / 'models' / 'vae' / 'xyz_ae.safetensors')
        _write(base / 'models' / 'vae' / 'qwen_image_vae.safetensors')
        assert zr.resolve_zimage_vae() is None


def test_gguf_is_never_auto_selected(app, tmp_path):
    """.gguf is LISTED by this app but core VAELoader/CLIPLoader cannot open one.
    A resolver picking by name alone shipped exactly this bug for Krea."""
    from app.services import zimage_model_resolver as zr
    with app.app_context():
        base = _comfy(tmp_path)
        _write(base / 'models' / 'vae' / 'z_ae-Q4_K_M.gguf')
        _write(base / 'models' / 'text_encoders' / 'qwen_3_4b-Q4_K_M.gguf')
        assert zr.resolve_zimage_vae() is None
        assert zr.resolve_zimage_text_encoder() is None


def test_gguf_loses_to_a_loadable_sibling(app, tmp_path):
    from app.services import zimage_model_resolver as zr
    with app.app_context():
        base = _comfy(tmp_path)
        _write(base / 'models' / 'vae' / 'z_ae-Q4_K_M.gguf')
        _write(base / 'models' / 'vae' / 'z_ae.safetensors')
        assert zr.resolve_zimage_vae() == 'z_ae.safetensors'


# --- Explicit choice always wins ----------------------------------------------

def test_explicit_setting_beats_the_scan(app, tmp_path):
    from app import config
    from app.services import zimage_model_resolver as zr
    with app.app_context():
        base = _comfy(tmp_path)
        _write(base / 'models' / 'vae' / 'z_ae.safetensors')
        _write(base / 'models' / 'vae' / 'my_pinned_ae.safetensors')
        config.save_config({'comfyui': {'base_dir': str(base)},
                            'zimage': {'vae': 'my_pinned_ae.safetensors'}})
        assert zr.resolve_zimage_vae() == 'my_pinned_ae.safetensors'


def test_explicit_argument_beats_the_setting_and_the_scan(app, tmp_path):
    from app import config
    from app.services import zimage_model_resolver as zr
    with app.app_context():
        base = _comfy(tmp_path)
        _write(base / 'models' / 'text_encoders' / 'qwen_3_4b.safetensors')
        _write(base / 'models' / 'text_encoders' / 'chosen.safetensors')
        config.save_config({'comfyui': {'base_dir': str(base)},
                            'zimage': {'text_encoder': 'qwen_3_4b.safetensors'}})
        assert zr.resolve_zimage_text_encoder('chosen.safetensors') == 'chosen.safetensors'


def test_explicit_choice_that_is_absent_is_returned_as_is(app, tmp_path):
    """If the user names a file, the failure must be about THEIR file — never a
    silent substitution of another one (klein_edit_helper's rule)."""
    from app import config
    from app.services import zimage_model_resolver as zr
    with app.app_context():
        base = _comfy(tmp_path)
        _write(base / 'models' / 'vae' / 'z_ae.safetensors')
        config.save_config({'comfyui': {'base_dir': str(base)},
                            'zimage': {'vae': 'nope.safetensors'}})
        assert zr.resolve_zimage_vae() == 'nope.safetensors'


# --- extra_model_paths.yaml roots ---------------------------------------------

def test_resolves_from_an_extra_model_paths_root(app, tmp_path):
    """Portable / Stability-Matrix installs keep weights outside <base>/models."""
    from app.services import zimage_model_resolver as zr
    with app.app_context():
        base = _comfy(tmp_path)
        shared = tmp_path / 'Shared'
        _write(shared / 'vae' / 'z_ae.safetensors')
        _write(shared / 'text_encoders' / 'Z Image' / 'qwen_3_4b.safetensors')
        (base / 'extra_model_paths.yaml').write_text(
            f"other:\n  base_path: {shared.as_posix()}\n  vae: vae\n"
            f"  text_encoders: text_encoders\n", encoding='utf-8')
        assert zr.resolve_zimage_vae() == 'z_ae.safetensors'
        assert zr.resolve_zimage_text_encoder() == os.path.join('Z Image', 'qwen_3_4b.safetensors')


# --- The workflow actually carries the resolved values ------------------------

def test_apply_zimage_settings_rewrites_the_loader_nodes(app, tmp_path):
    from app.utils.zimage_helper import apply_zimage_settings
    with app.app_context():
        base = _comfy(tmp_path)
        _write(base / 'models' / 'vae' / 'z_ae.safetensors')
        _write(base / 'models' / 'text_encoders' / 'Z Image' / 'qwen_3_4b.safetensors')
        wf = {'2': {'class_type': 'CLIPLoader',
                    'inputs': {'clip_name': 'Z image\\qwen_3_4b.safetensors', 'type': 'lumina2'}},
              '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'z ae.safetensors'}}}
        apply_zimage_settings(wf)
        assert wf['3']['inputs']['vae_name'] == 'z_ae.safetensors'
        assert wf['2']['inputs']['clip_name'] == os.path.join('Z Image', 'qwen_3_4b.safetensors')
        assert '_meta' not in wf['3']            # resolved -> no "missing" tag


def test_unresolved_workflow_keeps_the_template_ref_and_gains_a_hint(app, tmp_path):
    """The preflight must stay HONEST: nothing on disk -> we do not pretend, we tag
    the node with what was searched so the 409 can say it."""
    from app.utils.zimage_helper import apply_zimage_settings
    with app.app_context():
        _comfy(tmp_path)
        wf = {'2': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'Z image\\qwen_3_4b.safetensors'}},
              '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'z ae.safetensors'}}}
        apply_zimage_settings(wf)
        assert wf['3']['inputs']['vae_name'] == 'z ae.safetensors'
        assert 'z_ae' in wf['3']['_meta']['lds_missing_hint']
        assert 'qwen' in wf['2']['_meta']['lds_missing_hint'].lower()


def test_preflight_still_names_the_missing_files_and_carries_the_hint(app, tmp_path, monkeypatch):
    """End-to-end honesty check: an empty ComfyUI still gets a 409 naming BOTH refs,
    now with the accepted-spellings hint attached."""
    from app.services import lora_test_studio as lts
    from app.utils.zimage_helper import apply_zimage_settings
    with app.app_context():
        _comfy(tmp_path)
        monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                            lambda *a, **k: {'CLIPLoader', 'VAELoader'})
        wf = {'2': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'Z image\\qwen_3_4b.safetensors'}},
              '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'z ae.safetensors'}}}
        apply_zimage_settings(wf)
        with pytest.raises(lts.StudioAssetsMissing) as ei:
            lts.preflight_family('zimage', [wf])
        paths = ' '.join(f['path'] for f in ei.value.missing_files)
        assert 'z ae.safetensors' in paths and 'qwen_3_4b.safetensors' in paths
        assert all(f.get('hint') for f in ei.value.missing_files)


def test_preflight_is_silent_once_the_users_own_layout_is_on_disk(app, tmp_path, monkeypatch):
    """The regression this whole change exists for: bobba84's layout (underscore VAE,
    capital-I folder) must launch WITHOUT any rename."""
    from app.services import lora_test_studio as lts
    from app.utils.zimage_helper import apply_zimage_settings
    with app.app_context():
        base = _comfy(tmp_path)
        _write(base / 'models' / 'vae' / 'z_ae.safetensors')
        _write(base / 'models' / 'text_encoders' / 'Z Image' / 'qwen_3_4b.safetensors')
        monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                            lambda *a, **k: {'CLIPLoader', 'VAELoader'})
        wf = {'2': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'Z image\\qwen_3_4b.safetensors'}},
              '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'z ae.safetensors'}}}
        apply_zimage_settings(wf)
        lts.preflight_family('zimage', [wf])   # no raise


# --- LDS's own downloads: found with no setting, never over the user's choice ---

def _downloaded_comfy(tmp_path):
    """A valid fake ComfyUI holding the three Z-Image files at the exact
    destinations the installer writes them to."""
    from app import setup_installer
    from app.services import trained_image_models as models
    base = _comfy(tmp_path)
    (base / 'main.py').write_text('# fake ComfyUI entrypoint', encoding='utf-8')
    for slot in models.ZIMAGE_ASSET_SPECS:
        _write(pathlib.Path(setup_installer._download_dest_path(models.install_action('zimage', slot))))
    return base


def test_downloaded_files_are_what_the_studio_loads_with_no_setting(app, tmp_path, monkeypatch):
    from app.services import lora_test_studio as lts
    from app.services import trained_image_models as models
    from app.utils.comfyui import clear_model_caches
    from app.utils.zimage_helper import apply_zimage_settings
    with app.app_context():
        _downloaded_comfy(tmp_path)
        clear_model_caches()
        official = os.path.join('z image', 'z_image_turbo_bf16.safetensors')
        bases = lts.family_base_models('zimage')
        assert bases == [official]
        assert lts._select_base_models(bases, family='zimage') == [official]
        wf = lts.load_workflow_local(str(lts.WORKFLOW_ZTURBO_PATH))
        apply_zimage_settings(wf, z_model=official)
        assert wf['1']['inputs']['unet_name'] == official
        assert wf['2']['inputs']['clip_name'] == 'qwen_3_4b.safetensors'
        assert wf['3']['inputs']['vae_name'] == os.path.join('z image', 'ae.safetensors')
        monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes', lambda *a, **k: None)
        lts.preflight_family('zimage', [wf])   # every file the graph names is on disk
        status = models.generation_readiness('zimage', check_nodes=False)
        assert status['models_ready'] and status['downloads'] == []
        clear_model_caches()


def test_the_users_own_choices_still_win_over_the_downloads(app, tmp_path):
    from app import config
    from app.services import lora_test_studio as lts
    from app.services import trained_image_models as models
    from app.services import zimage_model_resolver as zr
    from app.utils.comfyui import clear_model_caches
    with app.app_context():
        base = _downloaded_comfy(tmp_path)
        _write(base / 'models' / 'unet' / 'z image' / 'bigLove_zt3.safetensors')
        _write(base / 'models' / 'vae' / 'mine' / 'my_vae.safetensors')
        _write(base / 'models' / 'text_encoders' / 'mine' / 'qwen3_4b_custom.safetensors')
        config.save_config({'zimage': {'vae': 'mine/my_vae.safetensors',
                                       'text_encoder': 'mine/qwen3_4b_custom.safetensors'}})
        clear_model_caches()
        mine = os.path.join('z image', 'bigLove_zt3.safetensors')
        bases = lts.family_base_models('zimage')
        assert mine in bases
        assert lts._select_base_models(bases, mine, family='zimage') == [mine]
        assert zr.resolve_zimage_vae() == os.path.join('mine', 'my_vae.safetensors')
        assert zr.resolve_zimage_text_encoder() == os.path.join('mine', 'qwen3_4b_custom.safetensors')
        status = models.generation_readiness('zimage', check_nodes=False)
        assert status['models_ready'] and status['downloads'] == [] and not status['pin_warnings']
        clear_model_caches()


# --- Point 2: Z-Image Base must not inherit Turbo's sampler defaults -----------

@pytest.mark.parametrize('name, build', [
    ('z image\\z_image_turbo_bf16.safetensors', 'turbo'),
    ('z image\\bigLove_zt3.safetensors', 'turbo'),
    ('z image\\Z-Image-Base-bf16.safetensors', 'base'),
    ('z image\\z_image_base.safetensors', 'base'),
    ('z image\\zimage_deturbo.safetensors', 'base'),
    ('z image\\mystery_finetune.safetensors', 'unknown'),
])
def test_zimage_build_detection(app, name, build):
    from app.services import lora_test_studio as lts
    assert lts.zimage_build_of(name) == build


def test_base_and_turbo_do_not_share_defaults(app):
    from app.services import lora_test_studio as lts
    turbo = lts.zimage_model_defaults('z image\\z_image_turbo_bf16.safetensors')
    base = lts.zimage_model_defaults('z image\\Z-Image-Base-bf16.safetensors')
    assert turbo != base
    assert turbo == {'cfg': 1.0, 'steps': 8}          # unchanged for everyone else
    assert base['cfg'] > 1.0 and base['steps'] >= 30  # non-distilled: guided + slow


def test_unknown_build_keeps_the_historical_turbo_defaults(app):
    from app.services import lora_test_studio as lts
    assert lts.zimage_model_defaults('z image\\mystery.safetensors') == {'cfg': 1.0, 'steps': 8}


def test_every_published_default_is_reachable_from_the_pickers(app):
    """A default the CFG/steps picker cannot show is a default the user cannot get
    back to after touching the axis."""
    from app.services import lora_test_studio as lts
    for d in (lts.ZIMAGE_TURBO_DEFAULTS, lts.ZIMAGE_BASE_DEFAULTS):
        assert d['cfg'] in lts.CFG_CHOICES
        assert d['steps'] in lts.STEPS_CHOICES


def test_studio_model_defaults_covers_the_two_split_families_and_is_keyed_by_picker_value(app):
    """Z-Image and Krea 2 both ship a distilled and an undistilled build that need
    OPPOSITE sampler settings; SDXL does not, and keeps the family-wide
    default_cfg/default_steps. Krea joined this list when full-model training
    started delivering Raw checkpoints, on which Turbo's cfg 1 / 8 steps render a
    blurry sketch that reads as a failed fine-tune."""
    from app.services import lora_test_studio as lts
    models = [{'value': 'z image\\z_image_turbo.safetensors', 'label': 'turbo'},
              {'value': 'z image\\z_image_base.safetensors', 'label': 'base'}]
    out = lts.studio_model_defaults('zimage', models)
    assert set(out) == {m['value'] for m in models}
    assert out['z image\\z_image_base.safetensors']['cfg'] > 1.0
    assert lts.studio_model_defaults('sdxl', models) == {}

    krea = lts.studio_model_defaults('krea', [
        {'value': '', 'label': 'Official - Krea 2 Turbo'},
        {'value': 'Krea_full_subject1_000002500_fp8.safetensors', 'label': 'full'}])
    # The official Turbo entry has an empty picker value and is deliberately not
    # listed: the frontend falls back to default_cfg/default_steps for it, which
    # ARE Turbo's numbers. Only the Raw/full/fp8 build gets its own row.
    assert '' not in krea
    assert krea['Krea_full_subject1_000002500_fp8.safetensors'] == {'cfg': 4.0, 'steps': 25}
