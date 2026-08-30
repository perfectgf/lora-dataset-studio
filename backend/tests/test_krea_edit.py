"""Krea 2 Identity Edit — the second LOCAL generation engine.

WHAT THESE TESTS ARE FOR
------------------------
The engine was validated by hand on ONE machine, against ONE ComfyUI. Everything
that made it work there is a fact about that install: a `Krea` folder, a
`krea2_identity_edit_v1_2.safetensors`, a node pack, a Qwen3-VL encoder sitting
next to two other Qwen encoders that would silently produce garbage. Nobody else
has that machine. So these tests pin the parts that have to survive a DIFFERENT
install: resolution (canonical first, narrow tokens after, never a blind guess,
never an incompatible base), preflight (say what is missing, do not crash and do
not invent a downloader), the graph wiring (both custom nodes, grounded negative,
cfg 1), the output geometry (source aspect for free edits, catalog aspect for
dataset cards, ≤ 2 MP) and the prompt rewriting (positive garments instead of
negations the model ignores).

NOTHING here renders anything: not one GPU second, not one paid call.
"""
import importlib
import os
import struct

import pytest

from app.services import face_variations as fv


# --- fixtures ---------------------------------------------------------------

def _fresh_config(monkeypatch, tmp_path):
    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    import app.config as config
    importlib.reload(config)
    return config


def _comfy_tree(tmp_path):
    """A minimal ComfyUI models/ tree, empty. Tests fill in only what they mean
    to test — an install that has nothing must be a supported state, not a crash."""
    base = tmp_path / 'Comfy'
    for sub in ('diffusion_models', 'unet', 'loras', 'text_encoders', 'vae'):
        (base / 'models' / sub).mkdir(parents=True, exist_ok=True)
    return base


# Smallest structurally-valid safetensors header (8-byte LE length + '{}'), so a
# fixture file reads as REAL (if tiny) weights. Krea now validates the header of
# every present asset (a licence-gate HTML page saved as .safetensors is the
# failure this catches), and the readiness gate keys off it — the default here
# keeps every "asset present" fixture honest without writing multi-GB test data.
_VALID_ST = struct.pack('<Q', 2) + b'{}'


def _write(path, data=_VALID_ST):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def krea(monkeypatch, tmp_path):
    """krea_edit_helper bound to a throwaway ComfyUI tree + throwaway config."""
    config = _fresh_config(monkeypatch, tmp_path)
    base = _comfy_tree(tmp_path)
    config.save_config({'comfyui': {'base_dir': str(base)}})
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    import app.services.krea_edit_helper as keh
    importlib.reload(keh)
    keh._nodes_ok_until = 0.0
    yield keh, base, config
    comfy_model_paths.clear_cache()


# --- base model resolution --------------------------------------------------

def test_no_krea_model_at_all_resolves_to_none_not_a_wrong_file(krea, tmp_path):
    """An install with other families' models must report 'missing', never hand a
    foreign checkpoint to the loader: missing produces an actionable message,
    wrong produces noise the user blames the app for."""
    keh, base, _ = krea
    _write(base / 'models' / 'diffusion_models' / 'flux-2-klein-9b-fp8.safetensors')
    assert keh.resolve_krea_unet() is None
    assert 'krea_model' in keh.krea_missing_assets()


def test_base_model_is_returned_with_its_subfolder_prefix(krea):
    keh, base, _ = krea
    _write(base / 'models' / 'diffusion_models' / 'Krea' / 'krea2_turbo_fp8.safetensors')
    assert keh.resolve_krea_unet() == os.path.join('Krea', 'krea2_turbo_fp8.safetensors')


def test_a_flat_install_with_no_krea_subfolder_still_resolves(krea):
    """Stability-Matrix / flat layouts drop the file straight into
    diffusion_models/. os.path.join('', name) == name is exactly what the loader
    wants for a root-level file."""
    keh, base, _ = krea
    _write(base / 'models' / 'diffusion_models' / 'krea2_raw_fp8.safetensors')
    assert keh.resolve_krea_unet() == 'krea2_raw_fp8.safetensors'


def test_turbo_wins_over_raw_and_the_choice_is_deterministic(krea):
    keh, base, _ = krea
    d = base / 'models' / 'diffusion_models' / 'Krea'
    _write(d / 'krea2_raw_fp8.safetensors')
    _write(d / 'krea2_turbo_fp8.safetensors')
    first = keh.resolve_krea_unet()
    assert first.endswith('krea2_turbo_fp8.safetensors')
    assert keh.resolve_krea_unet() == first, 'a regenerate must reproduce the render'


def test_the_incompatible_biglove_base_is_offered_but_never_preferred(krea):
    """MEASURED: the identity LoRA renders PURE NOISE on BigLoveKreaEdit1. That
    fact used to REMOVE the file from every list, which meant a build sitting in
    the user's own Krea folder was invisible, with nothing on screen saying it
    existed or why it was gone.

    Choosing is the user's call, so the file is listed. What survives is the rule
    about the app choosing FOR them: a flagged build never wins an election it
    shares with a real base. Alone on disk it IS elected -- refusing to run at
    all, on the only model present, would be the app deciding again."""
    keh, base, _ = krea
    d = base / 'models' / 'diffusion_models' / 'Krea'
    _write(d / 'BigLoveKreaEdit1_fp8mixed.safetensors')
    listed = [n for _sub, names in keh._krea_unet_folders() for n in names]
    assert 'BigLoveKreaEdit1_fp8mixed.safetensors' in listed, (
        'a file on the user own disk must not be hidden from them')
    assert keh.resolve_krea_unet().endswith('BigLoveKreaEdit1_fp8mixed.safetensors')
    # ...but a real base alongside it wins the election every time.
    _write(d / 'krea2_turbo_fp8.safetensors')
    assert keh.resolve_krea_unet().endswith('krea2_turbo_fp8.safetensors')


def test_an_explicit_base_model_setting_wins_and_a_stale_one_STOPS(krea):
    """This test used to assert the OPPOSITE of its second half, and the reversal
    is the point.

    It read: "a setting pointing at a file that is no longer there must NOT block
    the engine — it falls back to automatic resolution with a log line." That is
    the behaviour that let a whole training run on a third-party finetune nobody
    chose: Settings displayed one filename, the graph loaded another, and the log
    line was in a file nobody opens. A setting that silently resolves to
    something other than what it shows is invisible until after the expensive
    part.

    So a stale pin is now a STOP that names the file, and clearing the field is
    the explicit gesture that returns to auto-detection.
    """
    keh, base, config = krea
    d = base / 'models' / 'diffusion_models' / 'Krea'
    _write(d / 'krea2_turbo_fp8.safetensors')
    _write(d / 'krea2_raw_fp8.safetensors')
    config.save_config({'krea': {'base_model': 'krea2_raw_fp8.safetensors'}})
    assert keh.resolve_krea_unet().endswith('krea2_raw_fp8.safetensors')
    # The pinned file is gone: nothing is elected in its place...
    config.save_config({'krea': {'base_model': 'deleted_yesterday.safetensors'}})
    assert keh.resolve_krea_unet() is None
    # ...the gap is reported by name, so the UI can point at the field...
    gaps = keh.krea_pin_gaps()
    assert [g['key'] for g in gaps] == ['krea.base_model']
    assert gaps[0]['configured'] == 'deleted_yesterday.safetensors'
    # ...the run refuses rather than rendering on another base...
    with pytest.raises(keh.KreaPinnedModelMissing) as exc:
        keh.preflight()
    assert 'deleted_yesterday.safetensors' in str(exc.value)
    # ...and clearing the field is what goes back to automatic resolution.
    config.save_config({'krea': {'base_model': ''}})
    assert keh.resolve_krea_unet().endswith('krea2_turbo_fp8.safetensors')
    assert keh.krea_pin_gaps() == []


def test_the_shipped_identity_lora_name_is_not_a_user_pin(krea):
    """`krea.identity_lora` defaults to the canonical DOWNLOAD name, so every
    install has it "set" while nobody typed it. Treating that as a pin would turn
    the renamed-download recovery into a hard stop for everyone — the refusal is
    for values a user actually chose."""
    keh, base, config = krea
    loras = base / 'models' / 'loras' / 'krea'
    # The default name is absent, the renamed file is there: still found, no gap.
    _write(loras / 'krea2_identity_edit_v12_renamed.safetensors')
    name, path = keh.resolve_krea_identity_lora()
    assert name and name.endswith('krea2_identity_edit_v12_renamed.safetensors')
    assert keh.krea_pin_gaps() == []
    # A name the user typed themselves, absent: that one stops.
    config.save_config({'krea': {'identity_lora': 'krea/my_own_identity.safetensors'}})
    assert keh.resolve_krea_identity_lora() == (None, None)
    assert [g['slot'] for g in keh.krea_pin_gaps()] == ['identity_lora']


# --- encoder / VAE: the narrow-token discipline ------------------------------

def test_the_text_encoder_never_matches_a_bare_qwen(krea):
    """Three different Qwen encoders live in the same folder on a shared install.
    Klein's qwen_3_8b and Qwen-Image's qwen_2.5_vl produce incompatible
    embeddings — picking one would die at sampling on a shape mismatch."""
    keh, base, _ = krea
    te = base / 'models' / 'text_encoders'
    _write(te / 'qwen_3_8b_fp8mixed.safetensors')
    _write(te / 'qwen_2.5_vl_7b_fp8_scaled.safetensors')
    assert keh.resolve_krea_text_encoder() is None
    _write(te / 'qwen3vl_4b_fp8_scaled.safetensors')
    assert keh.resolve_krea_text_encoder() == 'qwen3vl_4b_fp8_scaled.safetensors'


def test_the_vae_never_matches_kleins(krea):
    keh, base, _ = krea
    _write(base / 'models' / 'vae' / 'flux2-vae.safetensors')
    assert keh.resolve_krea_vae() is None
    _write(base / 'models' / 'vae' / 'qwen_image_vae.safetensors')
    assert keh.resolve_krea_vae() == 'qwen_image_vae.safetensors'


def test_the_encoder_and_vae_are_found_in_subfolders(krea):
    """Same fix as the Klein slots (the reported bug): a user who files the
    family's encoder in a subfolder — mirroring the qwen/ and krea/ layouts the
    installer itself uses elsewhere — must not read as a missing model, and the
    returned name keeps the prefix the loader wants. The narrow-token discipline
    is untouched: the two tests above still hold at any depth."""
    keh, base, _ = krea
    _write(base / 'models' / 'text_encoders' / 'qwen' / 'qwen3vl_4b_fp8_scaled.safetensors')
    _write(base / 'models' / 'vae' / 'qwen' / 'qwen_image_vae.safetensors')
    assert keh.resolve_krea_text_encoder() == os.path.join(
        'qwen', 'qwen3vl_4b_fp8_scaled.safetensors')
    assert keh.resolve_krea_vae() == os.path.join('qwen', 'qwen_image_vae.safetensors')


def test_the_identity_lora_is_found_even_when_renamed_or_moved(krea):
    keh, base, _ = krea
    loras = base / 'models' / 'loras'
    assert keh.resolve_krea_identity_lora() == (None, None)
    # Not at the configured path, and renamed — the by-name scan still finds it.
    _write(loras / 'edits' / 'krea2_identity_edit_v1.3.safetensors')
    rel, path = keh.resolve_krea_identity_lora()
    assert rel == os.path.join('edits', 'krea2_identity_edit_v1.3.safetensors')
    assert os.path.isfile(path)
    # The canonical location wins once it exists.
    _write(loras / 'krea' / 'krea2_identity_edit_v1_2.safetensors')
    rel, _ = keh.resolve_krea_identity_lora()
    assert rel == os.path.join('krea', 'krea2_identity_edit_v1_2.safetensors')


def test_extra_model_paths_roots_are_searched_like_comfyui_does(krea, tmp_path):
    """The engine must work on an install whose models live outside the ComfyUI
    folder — that is the whole point of extra_model_paths.yaml."""
    keh, base, _ = krea
    external = tmp_path / 'A_drive' / 'models'
    _write(external / 'diffusion_models' / 'Krea' / 'krea2_turbo_fp8.safetensors')
    _write(external / 'loras' / 'krea2_identity_edit_v1_2.safetensors')
    (base / 'extra_model_paths.yaml').write_text(
        'mine:\n'
        f'  base_path: {external.parent.as_posix()}\n'
        '  diffusion_models: models/diffusion_models\n'
        '  loras: models/loras\n', encoding='utf-8')
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    assert keh.resolve_krea_unet().endswith('krea2_turbo_fp8.safetensors')
    assert keh.resolve_krea_identity_lora()[0] is not None


# --- preflight: say what's missing, never crash, never invent an installer ---

def _install_everything(base):
    _write(base / 'models' / 'diffusion_models' / 'Krea' / 'krea2_turbo_fp8.safetensors')
    _write(base / 'models' / 'loras' / 'krea' / 'krea2_identity_edit_v1_2.safetensors')
    _write(base / 'models' / 'text_encoders' / 'qwen3vl_4b_fp8_scaled.safetensors')
    _write(base / 'models' / 'vae' / 'qwen_image_vae.safetensors')


def test_an_empty_install_lists_every_gap_instead_of_raising_on_the_first(krea):
    keh, _base, _ = krea
    assert set(keh.krea_missing_assets()) == set(keh.KREA_REQUIRED)


def test_preflight_raises_with_both_halves_and_stays_actionable(krea, monkeypatch):
    keh, base, _ = krea
    monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes', lambda *a, **k: set())
    with pytest.raises(keh.KreaModelsMissing) as exc:
        keh.preflight()
    assert set(exc.value.missing) == set(keh.KREA_REQUIRED)
    assert set(exc.value.missing_nodes) == set(keh.KREA_NODE_CLASSES)
    # Every missing asset can be turned into a "place it HERE, get it THERE" line.
    for entry in keh.missing_file_entries(exc.value.missing):
        assert entry['path'] and entry['kind'] and entry['source'].startswith('http')
    for hint in keh.krea_node_hints(exc.value.missing_nodes):
        assert hint['pack'] and hint['url'].startswith('https://')


# The three Hugging Face weights all live in ONE public, non-gated Comfy-Org repo
# (measured 2026-07-27), under the exact canonical filenames the resolvers look
# for. Pinning the repo here is what stops the `source` links from drifting back
# to a page the user cannot act on: the previous values pointed at
# huggingface.co/krea/krea-2 (401 "Invalid username or password" — not a licence
# gate, a wall) and at Comfy-Org/Qwen-Image_ComfyUI, which holds qwen_2.5_vl_*
# and NOT the qwen3vl_4b encoder this engine needs — an install that followed
# that link downloaded a file `resolve_krea_text_encoder` deliberately refuses.
KREA_WEIGHTS_REPO = 'https://huggingface.co/Comfy-Org/Krea-2'
# Values that are known-bad: a source must never come back to one of these.
KREA_DEAD_SOURCES = ('huggingface.co/krea/krea-2', 'Qwen-Image_ComfyUI')


def test_every_asset_source_points_at_the_canonical_repo(krea):
    """A `source` is the ONLY thing a user has to go on when an asset is missing —
    a link that 401s or that lands in a repo without the file is worse than no
    link, because it costs a download before the app still says 'missing'."""
    keh, _base, _ = krea
    for key in ('krea_model', 'krea_text_encoder', 'krea_vae'):
        src = keh.KREA_ASSETS[key]['source']
        assert src.startswith(KREA_WEIGHTS_REPO), (
            f'{key} source {src!r} is not in the canonical weights repo')
    # The identity LoRA is the one piece that is NOT on Hugging Face.
    assert keh.KREA_ASSETS['krea_identity_lora']['source'].startswith(
        'https://civitai.com/')
    for meta in keh.KREA_ASSETS.values():
        for dead in KREA_DEAD_SOURCES:
            assert dead not in meta['source'], f'{dead} is a dead end for a user'


def test_the_node_probe_fails_OPEN_when_comfyui_cannot_be_reached(krea, monkeypatch):
    """A transient probe failure must never look like a missing node pack — the
    user would be sent to install something they already have."""
    keh, _base, _ = krea
    monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes', lambda *a, **k: None)
    assert keh.krea_missing_nodes() == []


def test_a_complete_install_preflights_clean(krea, monkeypatch):
    keh, base, _ = krea
    _install_everything(base)
    monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                        lambda *a, **k: set(keh.KREA_NODE_CLASSES) | {'KSampler'})
    assert keh.krea_missing_assets() == []
    # Valid (if tiny) headers: only the ADVISORY too_small may show up, never a
    # blocking verdict — an advisory must not keep a working engine dark.
    assert all(not i['blocking'] for i in keh.krea_invalid_assets())
    keh.preflight()   # must not raise


# --- Present-but-INVALID: the state between "missing" and "ready" ------------
def test_a_licence_gate_page_saved_as_safetensors_is_named_not_silently_run(krea):
    """The Krea base sits behind a Hugging Face licence gate and the identity
    LoRA behind a Civitai login: a browser download that skipped either saves the
    HTML gate PAGE to <name>.safetensors. It is ON DISK, so krea_missing_assets
    says nothing — and until now the only symptom was ComfyUI's raw
    `UNETLoader: Expecting value: line 1 column 1 (char 0)` at generate time."""
    from app.services import model_integrity
    keh, base, _ = krea
    _install_everything(base)
    _write(base / 'models' / 'diffusion_models' / 'Krea' / 'krea2_turbo_fp8.safetensors',
           data=b'<!doctype html><html>You need to accept the licence</html>')
    model_integrity.clear_cache()
    assert 'krea_model' not in keh.krea_missing_assets()      # it IS on disk
    inv = {i['asset']: i for i in keh.krea_invalid_assets()}
    assert 'krea_model' in inv
    assert inv['krea_model']['blocking'] is True
    assert inv['krea_model']['verdict'] == 'html_or_text'
    assert 'krea2_turbo_fp8.safetensors' in inv['krea_model']['reason']


def test_a_truncated_identity_lora_is_flagged_too(krea):
    """Not just the base: a half-downloaded LoRA renders SILENTLY distorted
    images, with no error anywhere to point at."""
    from app.services import model_integrity
    keh, base, _ = krea
    _install_everything(base)
    _write(base / 'models' / 'loras' / 'krea' / 'krea2_identity_edit_v1_2.safetensors',
           data=b'\0' * 16)
    model_integrity.clear_cache()
    inv = {i['asset']: i for i in keh.krea_invalid_assets()}
    assert 'krea_identity_lora' in inv and inv['krea_identity_lora']['blocking'] is True


def test_invalid_assets_is_empty_on_an_install_with_nothing(krea):
    """Nothing on disk = 'missing', which krea_missing_assets owns. This one must
    stay silent rather than double-report every gap."""
    keh, _base, _ = krea
    assert keh.krea_invalid_assets() == []


# --- output geometry --------------------------------------------------------

@pytest.mark.parametrize('w,h', [(4000, 3000), (1024, 1024), (832, 1216), (5000, 1000)])
def test_output_keeps_the_source_aspect_under_two_megapixels(w, h):
    from app.services import krea_edit_helper as keh
    ow, oh = keh.fit_output_size(w, h)
    assert ow * oh <= 2_000_000, 'the model drifts above ~2 MP'
    assert ow % 16 == 0 and oh % 16 == 0, 'the latent grid is 16-aligned'
    assert abs((ow / oh) - (w / h)) / (w / h) < 0.05, 'aspect must be preserved'


@pytest.mark.parametrize(('requested_aspect', 'expected_ratio'), [
    ('1:1', 1.0),
    ('3:4', 3 / 4),
])
def test_requested_catalog_aspect_uses_a_16_aligned_canvas_under_two_megapixels(
        requested_aspect, expected_ratio):
    """The v1.2 fit node can use a card ratio even from a mismatched reference."""
    from app.services import krea_edit_helper as keh
    ow, oh = keh.fit_output_size(4000, 1000, requested_aspect=requested_aspect)
    assert ow * oh <= 2_000_000
    assert ow % 16 == 0 and oh % 16 == 0
    assert abs((ow / oh) - expected_ratio) / expected_ratio < 0.02


def test_an_invalid_requested_aspect_keeps_the_reference_edit_source_geometry():
    from app.services import krea_edit_helper as keh
    assert (keh.fit_output_size(1536, 2048, requested_aspect='not-a-ratio') ==
            keh.fit_output_size(1536, 2048))


def test_requested_catalog_aspect_spends_the_dial_not_the_reference_size():
    """REPLACES the previous no-upscale rule on the card path (deliberate).

    That rule capped a card at the reference's own pixel count, so a 1024x832
    reference produced 0.84 MP Krea tiles beside 2 MP Klein ones in the same
    dataset — the dial the user set said 2. The output size is now the setting's
    answer on both engines; see tests/test_variation_output_size.py. The
    no-upscale rule still governs the free reference edit below.
    """
    from app.services import krea_edit_helper as keh
    ow, oh = keh.fit_output_size(1024, 1024, requested_aspect='3:4')
    assert ow * oh > 1024 * 1024, 'the card budget is no longer capped by the source'
    assert ow * oh <= 2_000_000
    assert abs((ow / oh) - (3 / 4)) / (3 / 4) < 0.02


def test_a_small_source_is_never_upscaled_into_invented_detail():
    from app.services import krea_edit_helper as keh
    assert keh.fit_output_size(512, 512) == (512, 512)


def test_an_unreadable_size_degrades_to_a_square_instead_of_crashing():
    from app.services import krea_edit_helper as keh
    assert keh.fit_output_size(0, 0) == (1024, 1024)
    assert keh.fit_output_size(None, 'x') == (1024, 1024)


# --- free-reference edit geometry -------------------------------------------
# Dataset cards now request their own canvas above. These tests deliberately
# cover the other Krea path: a free reference edit keeps the source frame, so a
# user who manually crops their source gets exactly that geometry back.

def test_a_free_reference_edit_keeps_a_square_source_frame():
    from app.services import krea_edit_helper as keh
    ow, oh = keh.fit_output_size(1024, 1024)
    assert ow == oh


@pytest.mark.parametrize('w,h', [
    (1536, 2048),   # a 3:4 crop of a 4 MP phone photo
    (768, 1024),    # a 3:4 crop of a small reference
    (600, 800),     # a 3:4 crop of a tiny one (must not be upscaled either)
    (1080, 1920),   # 9:16 — the taller preset the crop editor also offers
])
def test_a_free_reference_edit_keeps_a_manual_portrait_crop(w, h):
    """Manual source crops retain their shape in the free-edit path."""
    from app.services import krea_edit_helper as keh
    ow, oh = keh.fit_output_size(w, h)
    assert oh > ow, f'{w}x{h} must stay portrait through the sizing, got {ow}x{oh}'
    assert abs((ow / oh) - (w / h)) / (w / h) < 0.05, 'the crop ratio is preserved'
    assert ow % 16 == 0 and oh % 16 == 0
    assert ow * oh <= 2_000_000
    # And it stays big enough to be worth training on: a "portrait" that came
    # back at 300 px would be a technically-correct, useless answer.
    assert min(ow, oh) >= 512


# --- the graph --------------------------------------------------------------

def _graph():
    from app.services import krea_edit_helper as keh
    return keh.build_workflow('ref.png', 'a prompt', unet='Krea/base.safetensors',
                              clip='te.safetensors', vae='vae.safetensors',
                              lora_name='krea/id.safetensors', width=1024, height=1024,
                              seed=7)


def test_both_custom_nodes_are_present_and_the_negative_is_grounded_too():
    """MEASURED: with a plain CLIPTextEncode the reference has NO effect — the
    encoder never sees the image. The pack's own workflow grounds the negative
    branch as well, and so do we."""
    g = _graph()
    classes = [n['class_type'] for n in g.values()]
    assert classes.count('Krea2EditGroundedEncode') == 2
    assert 'Krea2EditModelPatch' in classes
    assert 'CLIPTextEncode' not in classes
    encodes = [n for n in g.values() if n['class_type'] == 'Krea2EditGroundedEncode']
    assert all('image' in n['inputs'] for n in encodes), 'both branches see the reference'
    assert sorted(n['inputs']['prompt'] for n in encodes) == ['', 'a prompt']


def test_the_identity_lora_sits_between_the_unet_and_the_patch():
    g = _graph()
    lora = next(k for k, n in g.items() if n['class_type'] == 'LoraLoaderModelOnly')
    unet = next(k for k, n in g.items() if n['class_type'] == 'UNETLoader')
    patch = next(k for k, n in g.items() if n['class_type'] == 'Krea2EditModelPatch')
    sampler = next(k for k, n in g.items() if n['class_type'] == 'KSampler')
    assert g[lora]['inputs']['model'] == [unet, 0]
    assert g[patch]['inputs']['model'] == [lora, 0]
    assert g[sampler]['inputs']['model'] == [patch, 0]


def test_the_sampler_settings_are_the_measured_ones():
    g = _graph()
    ks = next(n for n in g.values() if n['class_type'] == 'KSampler')['inputs']
    assert ks['cfg'] == 1.0, 'guidance-distilled: any other cfg is a mistake'
    assert (ks['sampler_name'], ks['scheduler']) == ('euler', 'simple')
    assert ks['steps'] == 8


def test_the_krea_graph_defaults_favor_prompt_adherence_without_weakening_identity():
    g = _graph()
    encodes = [n for n in g.values() if n['class_type'] == 'Krea2EditGroundedEncode']
    patch = next(n for n in g.values() if n['class_type'] == 'Krea2EditModelPatch')
    lora = next(n for n in g.values() if n['class_type'] == 'LoraLoaderModelOnly')
    assert {n['inputs']['grounding_px'] for n in encodes} == {512}
    assert patch['inputs']['ref_boost'] == 0.25
    assert lora['inputs']['strength_model'] == 1.0


def _graph_with_extra(extra='extra.png'):
    from app.services import krea_edit_helper as keh
    return keh.build_workflow('ref.png', 'a prompt', unet='Krea/base.safetensors',
                              clip='te.safetensors', vae='vae.safetensors',
                              lora_name='krea/id.safetensors', width=1024, height=1024,
                              seed=7, extra_source_image=extra)


def _patch_inputs(g):
    return next(n for n in g.values()
                if n['class_type'] == 'Krea2EditModelPatch')['inputs']


def test_a_second_reference_reaches_the_patch_and_BOTH_grounded_encodes():
    """The pack's own two-reference workflow grounds the NEGATIVE branch on both
    images as well — that is the trained unconditional, not an optimisation we
    could skip on the branch whose prompt is empty."""
    g = _graph_with_extra()
    extra_id = next(k for k, n in g.items()
                    if n['class_type'] == 'LoadImage' and n['inputs']['image'] == 'extra.png')
    encodes = [n for n in g.values() if n['class_type'] == 'Krea2EditGroundedEncode']
    assert _patch_inputs(g)['source_image_b'] == [extra_id, 0]
    assert len(encodes) == 2
    assert all(n['inputs']['image_b'] == [extra_id, 0] for n in encodes)


def test_the_second_reference_adds_no_vae_encode_the_node_would_never_read():
    """`Krea2EditModelPatch` rebuilds its sources from the PIXEL path whenever vae
    + source_image are connected, which this graph always does — `source_latent_b`
    is then never read. Wiring it 'for symmetry' would buy a VAE encode per render
    and change nothing."""
    g = _graph_with_extra()
    assert sum(1 for n in g.values() if n['class_type'] == 'VAEEncode') == 1
    assert 'source_latent_b' not in _patch_inputs(g)


def test_a_second_reference_does_not_move_the_boost_off_the_primary():
    """The node applies `ref_boost` to the LAST reference and `ref_boost_a` to the
    first. Adding an angle demotes the primary to first — without the handoff the
    user's tuned value would land on the extra and the primary would silently
    fall back to the hardcoded 1.0."""
    one, two = _patch_inputs(_graph()), _patch_inputs(_graph_with_extra())
    assert (one['ref_boost'], one['ref_boost_a']) == (0.25, 1.0)
    assert two['ref_boost'] == two['ref_boost_a'] == 0.25


def test_no_second_reference_leaves_the_graph_exactly_as_it_was():
    """The single-reference render is what every existing install gets: adding the
    optional slot must not perturb one node of it."""
    from app.services import krea_edit_helper as keh
    assert _graph() == keh.build_workflow(
        'ref.png', 'a prompt', unet='Krea/base.safetensors', clip='te.safetensors',
        vae='vae.safetensors', lora_name='krea/id.safetensors', width=1024,
        height=1024, seed=7, extra_source_image=None)


def test_the_clip_loader_asks_for_the_krea2_type():
    g = _graph()
    clip = next(n for n in g.values() if n['class_type'] == 'CLIPLoader')
    assert clip['inputs']['type'] == 'krea2'


def test_no_loader_value_is_ever_hardcoded_in_the_graph_builder():
    """Every model name in the built graph must be one the CALLER passed, i.e.
    one a resolver produced. This is the invariant that keeps the engine working
    on an install that looks nothing like the one it was written on."""
    g = _graph()
    names = {n['inputs'].get(k) for n in g.values()
             for k in ('unet_name', 'clip_name', 'vae_name', 'lora_name')
             if n['inputs'].get(k)}
    assert names == {'Krea/base.safetensors', 'te.safetensors', 'vae.safetensors',
                     'krea/id.safetensors'}


# --- settings ---------------------------------------------------------------

def test_grounding_is_clamped_and_snapped_and_junk_degrades_to_the_default(krea):
    keh, _base, config = krea
    # The v4 identity-first pair — grounding and pull always ship together.
    assert config.get('krea.grounding_px') == 1024
    assert config.get('krea.ref_boost') == 4.0
    assert keh.grounding_px() == 1024
    assert keh._ref_boost() == 4.0
    config.save_config({'krea': {'grounding_px': 700}})
    assert keh.grounding_px() == 704, 'snapped to the 64px patch grid'
    config.save_config({'krea': {'grounding_px': 99999}})
    assert keh.grounding_px() == keh.GROUNDING_PX_MAX
    config.save_config({'krea': {'grounding_px': 'a lot'}})
    assert keh.grounding_px() == 512


# --- prompt: negations the model ignores become positive directives ---------

def test_the_default_outfit_negation_becomes_a_concrete_garment():
    """MEASURED: the outfit NEVER changed. The catalog says "a different casual
    everyday outfit … (not the outfit from the reference image)" and this model
    preserves whatever it is not positively ordered to change."""
    prompt = 'upper body portrait, front view, ' + fv.OUTFIT_VARY
    out = fv.krea_outfit_directive(prompt, 'bust_front')
    assert fv.OUTFIT_VARY not in out
    assert 'not the outfit' not in out
    assert any(g in out for g in fv.KREA_OUTFIT_PALETTE)


def test_outfits_differ_across_shots_but_a_shot_always_gets_the_same_one():
    """The two properties a dataset needs at once: variety between images, and a
    regenerate that reproduces its own image."""
    labels = [e['label'] for e in fv.VARIATION_CATALOG]
    picks = [fv.krea_outfit_for(l) for l in labels]
    assert len(set(picks)) > 1, 'a single garment everywhere is the bug being fixed'
    assert picks == [fv.krea_outfit_for(l) for l in labels], 'must be stable'
    # Stable ACROSS PROCESSES too: hash() would be randomised by PYTHONHASHSEED.
    assert fv.krea_outfit_for('Bust, front') == fv.KREA_OUTFIT_PALETTE[
        __import__('zlib').crc32(b'Bust, front') % len(fv.KREA_OUTFIT_PALETTE)]


def test_a_named_garment_keeps_its_name_and_only_loses_the_dead_negation():
    out = fv.krea_outfit_directive(
        'upper body portrait, wearing a jacket different from the reference outfit, urban', 'x')
    assert 'wearing a jacket' in out
    assert 'different from the reference' not in out


def test_the_expression_negation_is_dropped_but_the_intent_is_kept():
    out = fv.krea_outfit_directive('close-up, ' + fv.EXPRESSION_NEUTRAL, 'x')
    assert 'a calm neutral facial expression' in out
    assert 'not copying the expression' not in out


def test_the_rewrite_is_idempotent_and_a_no_op_on_non_human_catalogs():
    once = fv.krea_outfit_directive('bust, ' + fv.OUTFIT_VARY, 'lbl')
    assert fv.krea_outfit_directive(once, 'lbl') == once
    animal = 'full body from head to tail, standing on grass, daylight'
    assert fv.krea_outfit_directive(animal, 'lbl') == animal


def test_the_wrapper_is_instruction_first_and_locks_permanent_markings():
    """MEASURED: the framing WAS honoured with this shape (the API-engine wrapper
    put preservation first and returned a close-up whatever the shot asked). And
    the tattoos were REDRAWN every render, which a LoRA would learn as an average
    tattoo — hence the explicit hold order.

    That order used to assert its own wording here ('same design, same placement
    and same size'), which is how it survived long enough to be shipped: the
    sentence enumerated tattoos, scars, moles and piercings, and the encoder
    painted them on subjects who had none. The assertion now checks what the
    order must DO, not the words it uses — the wording itself is pinned by
    test_the_markings_lock_names_no_feature_it_could_summon."""
    out = fv.wrap_variation_krea('full body shot, standing', framing='body',
                                 label='Body standing, front')
    assert out.startswith('Create a new photograph of the same person')
    assert 'ENTIRE body visible from head to toe' in out
    assert fv.KREA_MARKINGS_LOCK.strip() in out
    # The identity lock is the SHARED, user-editable klein_identity one — not a
    # second copy the user would have to keep in sync.
    assert fv.get_identity_prompt('klein_identity', 'human') in out


@pytest.mark.parametrize(
    ('entry_id', 'conflicting_detail', 'requested_card_text'),
    [
        ('face_profile_l', 'both eyes in crisp focus', 'left profile view'),
        ('body_sit', 'natural standing distance', 'sitting on a chair'),
    ],
)
def test_krea_pose_cards_override_conflicting_generic_framing_details(
        entry_id, conflicting_detail, requested_card_text):
    """The card, not a reusable front/standing hint, is Krea's final command.

    These are the two field failures: the default face detail asks for both eyes
    on a strict profile, while the body detail asks for a standing composition on
    a seated card.  The generic detail must disappear and the exact card intent
    must be the last imperative Krea reads.
    """
    entry = next(e for e in fv.VARIATION_CATALOG if e['id'] == entry_id)
    out = fv.wrap_variation_krea(entry['prompt'], framing=entry['framing'],
                                 label=entry['label'])
    final_card = fv.krea_card_pose_priority(
        fv._krea_builtin_card_prompt(entry['prompt'], entry['label'], 'human'),
        true_profile=entry_id == 'face_profile_l')

    assert requested_card_text in out
    assert conflicting_detail not in out
    assert out.endswith(final_card)
    # The concrete words must be repeated after the identity lock and rendering
    # tail, not merely referred to as an earlier instruction.
    assert out.rfind(requested_card_text) > out.rfind(
        fv.get_identity_prompt('render_tail_sfw', 'human'))
    assert (fv.KREA_TRUE_PROFILE_REQUIREMENT in out) == (entry_id == 'face_profile_l')


def test_krea_card_priority_is_scoped_to_krea_not_klein_or_api_wrappers():
    """Do not generalise a Krea-specific prompt correction to other engines."""
    entry = next(e for e in fv.VARIATION_CATALOG if e['id'] == 'face_profile_l')
    krea = fv.wrap_variation_krea(entry['prompt'], framing=entry['framing'],
                                  label=entry['label'])
    klein = fv.wrap_variation_klein(entry['prompt'], framing=entry['framing'],
                                    label=entry['label'])
    api = fv.wrap_variation(entry['prompt'])

    assert 'both eyes in crisp focus' not in krea
    assert fv.KREA_CARD_POSE_PRIORITY_PREFIX in krea
    assert 'both eyes in crisp focus' in klein
    assert fv.KREA_CARD_POSE_PRIORITY_PREFIX not in klein
    assert fv.KREA_CARD_POSE_PRIORITY_PREFIX not in api


def test_krea_card_priority_stays_human_and_uses_the_current_regenerate_prompt():
    """A historical label must not overrule an edited prompt or non-human detail."""
    front = fv.wrap_variation_krea('close-up portrait, front view', framing='face',
                                   label='Profile left')
    assert 'both eyes in crisp focus' in front
    assert fv.KREA_CARD_POSE_PRIORITY_PREFIX not in front

    sitting = next(e for e in fv.VARIATION_CATALOG if e['id'] == 'body_sit')
    animal = fv.wrap_variation_krea(sitting['prompt'], framing='body',
                                    subject_type='animal', label=sitting['label'])
    animal_detail = fv.get_identity_prompt('framing_body', 'animal')
    assert animal_detail in animal
    assert fv.KREA_CARD_POSE_PRIORITY_PREFIX not in animal


def test_krea_three_quarter_profile_is_not_rewritten_as_a_true_side_profile():
    out = fv.wrap_variation_krea('close-up portrait, three-quarter profile, smiling',
                                 framing='face', label='Face 3/4')
    assert 'both eyes in crisp focus' in out
    assert fv.KREA_CARD_POSE_PRIORITY_PREFIX not in out
    assert fv.KREA_TRUE_PROFILE_REQUIREMENT not in out


def test_krea_profile_priority_is_not_disabled_by_an_age():
    out = fv.wrap_variation_krea('close-up portrait, left profile view, 34-year-old',
                                 framing='face', label='Profile left')
    assert 'both eyes in crisp focus' not in out
    assert fv.KREA_TRUE_PROFILE_REQUIREMENT in out


def test_krea_card_priority_does_not_duplicate_a_dataset_suffix():
    entry = next(e for e in fv.VARIATION_CATALOG if e['id'] == 'face_profile_l')
    suffix = 'shot on 35mm film'
    out = fv.wrap_variation_krea(entry['prompt'], framing=entry['framing'],
                                 label=entry['label'], suffix=suffix)
    assert out.count(suffix) == 1


def test_krea_card_priority_does_not_repeat_edited_or_custom_prompt_text():
    """Free-form prompt text stays before the identity/SFW locks exactly once."""
    custom = 'left profile view, ' + ('deliberate detail ' * 800)
    out = fv.wrap_variation_krea(custom, framing='face', label='Custom profile')
    assert out.count(custom) == 1
    assert out.endswith(fv.krea_card_pose_priority(true_profile=True))
    assert not out.endswith(fv.krea_card_pose_priority(custom))


def test_krea_card_priority_tail_is_not_built_from_the_editable_palette(monkeypatch):
    """A palette choice may shape the description, never the post-SFW tail."""
    entry = next(e for e in fv.VARIATION_CATALOG if e['id'] == 'face_profile_l')
    palette_text = 'untrusted palette text that must remain before identity'
    monkeypatch.setattr(fv, 'outfit_palette', lambda *_args: (palette_text,))

    out = fv.wrap_variation_krea(entry['prompt'], framing=entry['framing'],
                                 label=entry['label'])
    assert out.count(palette_text) == 1
    assert out.rfind(palette_text) < out.rfind(
        fv.get_identity_prompt('klein_identity', 'human'))
    assert out.endswith(fv.krea_card_pose_priority(
        fv._krea_builtin_card_prompt(entry['prompt'], entry['label'], 'human'),
        true_profile=True))


def test_the_wrapper_respects_the_subject_type_and_the_nsfw_switch():
    anime = fv.wrap_variation_krea('close-up', framing='face', subject_type='anime',
                                   label='Face front')
    # The MEDIUM word of the opening command switches; the anime lock's own
    # "don't turn it into a photograph" clause legitimately says the word again.
    assert anime.startswith('Create a new illustration of the same character')
    assert 'Anime illustration, same art style as the reference' in anime
    nsfw = fv.wrap_variation_krea('full body', framing='body', nsfw=True, label='x')
    assert 'SFW' not in nsfw and 'Explicit nudity is allowed' in nsfw


def test_the_dataset_suffix_is_spliced_once_into_the_description():
    out = fv.wrap_variation_krea('close-up portrait', framing='face',
                                 suffix='shot on 35mm film', label='x')
    assert out.count('shot on 35mm film') == 1


# --- engine identity: ids and labels agree with the frontend -----------------

def test_the_local_engine_ids_and_labels_match_the_frontend():
    """Same seam as test_engine_lists_contract, for the LOCAL engines: the ids
    are persisted in FaceDatasetImage.klein_model and both sides word user-facing
    messages from the labels."""
    import re
    from pathlib import Path
    from app.services import face_dataset_service as svc
    js = (Path(__file__).resolve().parents[2] / 'frontend' / 'src' / 'components'
          / 'dataset' / 'engineSelection.js')
    if not js.exists():
        pytest.skip('frontend source not present')
    src = js.read_text(encoding='utf-8')
    m = re.search(r'export const LOCAL_ENGINES\s*=\s*\[(.*?)\];', src, re.S)
    assert m, 'LOCAL_ENGINES declaration not found in engineSelection.js'
    assert tuple(re.findall(r"'([^']+)'", m.group(1))) == svc.LOCAL_ENGINES
    labels = dict(re.findall(
        r"(\w+):\s*'([^']*)'",
        re.search(r'export const ENGINE_LABELS\s*=\s*\{(.*?)\};', src, re.S).group(1)))
    for engine in svc.LOCAL_ENGINES:
        assert labels.get(engine) == svc.LOCAL_ENGINE_LABELS[engine], engine


def test_krea_is_a_known_engine_the_routes_accept():
    from app.services import face_dataset_service as svc
    assert 'krea' in svc.KNOWN_ENGINES and 'krea' in svc.LOCAL_ENGINES
    assert 'krea' not in svc.API_ENGINES


def test_krea_dataset_batch_and_regenerate_forward_the_catalog_card_aspect(app, monkeypatch):
    """Krea alone turns the card ratio into its local target canvas.

    ``Body, wide urban shot`` deliberately overrides the body's normal 3:4 with
    16:9, so this catches both a missing label lookup and an accidental fallback
    to the source geometry. The API engine functions are never involved here.
    """
    from app.extensions import db
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import krea_edit_helper as keh

    calls = []
    monkeypatch.setattr(keh, 'preflight', lambda: None)

    def enqueue(**kwargs):
        calls.append(kwargs)
        return f'krea-aspect-{len(calls)}'

    monkeypatch.setattr(keh, 'enqueue_krea_edit', enqueue)
    shot = {'label': 'Body, wide urban shot', 'framing': 'body',
            'prompt': 'full body shot, wide environmental framing'}
    with app.app_context():
        ds = svc.create_dataset('local', 'Krea aspect', 'krea_aspect')
        ds.ref_filename = 'ref.png'
        db.session.commit()

        assert svc.generate_variations_krea('local', ds.id, [shot], 1)

        # A finished/failed row avoids cancelling a fake job while exercising the
        # same Krea regenerate branch a real completed tile takes.
        row = FaceDatasetImage(dataset_id=ds.id, source='generated', status='failed',
                               variation_label=shot['label'], framing=shot['framing'],
                               variation_prompt=shot['prompt'], klein_model='krea')
        db.session.add(row)
        db.session.commit()
        assert svc.regenerate_image('local', row.id) == 'krea-aspect-2'
        db.session.refresh(row)
        assert row.klein_model == 'krea'

    assert [call['aspect_ratio'] for call in calls] == ['16:9', '16:9']


def test_plain_krea_retry_uses_row_engine_without_klein_preflight(client, app, monkeypatch):
    """Retry sends no override, so the saved Krea tag decides the lane.

    In particular the route must not inspect the Klein graph before the service
    can resolve a failed Krea row: an unavailable Klein node is irrelevant to a
    Krea retry and must not reject it before the Krea lane is selected.
    """
    from app.extensions import db
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import krea_edit_helper as keh
    from app.services import klein_edit_helper as kleh

    seen = []
    monkeypatch.setattr(
        kleh, 'klein_missing_nodes',
        lambda: (_ for _ in ()).throw(AssertionError('Klein preflight must not run for Krea')))
    monkeypatch.setattr(
        svc, '_api_generate_fn',
        lambda _engine: (_ for _ in ()).throw(AssertionError('API must not run for Krea')))
    monkeypatch.setattr(
        keh, 'enqueue_krea_edit',
        lambda **kwargs: (seen.append(kwargs), 'krea-retry-job')[1])

    with app.app_context():
        ds = svc.create_dataset('local', 'Krea retry', 'krea_retry')
        ds.ref_filename = 'ref.png'
        row = FaceDatasetImage(
            dataset_id=ds.id, source='generated', status='failed',
            variation_label='Bust, front', framing='bust',
            variation_prompt='upper body portrait', klein_model='krea')
        db.session.add(row)
        db.session.commit()
        row_id = row.id

    response = client.post(f'/api/dataset/image/{row_id}/regenerate', json={})
    assert response.status_code == 200
    assert response.get_json() == {'ok': True, 'job_id': 'krea-retry-job'}
    assert len(seen) == 1

    with app.app_context():
        retried = db.session.get(FaceDatasetImage, row_id)
        assert (retried.status, retried.job_id, retried.klein_model) == (
            'pending', 'krea-retry-job', 'krea')


def test_plain_klein_retry_keeps_its_node_preflight_and_maps_the_409(
        client, app, monkeypatch):
    """Moving preflight behind origin resolution must not weaken Klein retries."""
    from app.extensions import db
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as kleh

    missing = [{'class_type': 'ExampleKleinNode', 'pack': 'Example-Pack',
                'url': 'https://example.invalid/pack'}]
    monkeypatch.setattr(kleh, 'klein_missing_nodes', lambda: missing)
    monkeypatch.setattr(kleh, 'klein_missing_assets', lambda: [])

    with app.app_context():
        ds = svc.create_dataset('local', 'Klein retry', 'klein_retry')
        ds.ref_filename = 'ref.png'
        row = FaceDatasetImage(
            dataset_id=ds.id, source='generated', status='failed',
            variation_label='Bust, front', framing='bust',
            variation_prompt='upper body portrait',
            klein_model='flux-2-klein.safetensors')
        db.session.add(row)
        db.session.commit()
        row_id = row.id

    response = client.post(f'/api/dataset/image/{row_id}/regenerate', json={})
    assert response.status_code == 409
    body = response.get_json()
    assert body['ok'] is False
    assert body['klein_nodes_missing'] == missing
    assert 'ExampleKleinNode' in body['error']


# --- route: the 409 is the ONLY thing standing between a fresh install and a
#     grid of silently-failing tiles, so it is pinned end to end -------------

def test_generating_on_krea_without_the_weights_answers_one_actionable_409(client):
    """Nothing installed: ONE 409 whose message says where each file goes and
    where it comes from — and NOT a single row created. Nothing is downloaded:
    unlike Klein's public direct links these have no verified URL, and a fake
    installer would be worse than an honest message."""
    ds = client.post('/api/dataset/create',
                     json={'name': 'K', 'trigger_word': 'ktrig'}).get_json()['id']
    r = client.post(f'/api/dataset/{ds}/generate', json={
        'engine_batches': [{'generator': 'krea',
                            'variations': [{'label': 'Bust, front', 'framing': 'bust',
                                            'prompt': 'upper body portrait'}]}],
        'multiplier': 1})
    assert r.status_code == 409
    body = r.get_json()
    assert body['ok'] is False
    msg = body['error']
    assert 'models/loras' in msg and 'models/vae' in msg
    assert 'https://' in msg and 'Then retry' in msg
    assert set(body['krea_missing']['assets']) >= {'krea_model', 'krea_identity_lora'}
    for f in body['krea_missing']['files']:
        assert f['path'] and f['kind'] and f['source']
    # Nothing was created: a preflight that leaves half a batch behind is worse
    # than no preflight at all.
    assert client.get(f'/api/dataset/{ds}').get_json()['images'] == []


def test_a_missing_node_pack_is_named_in_the_same_409(client, monkeypatch):
    """The other half of the same message. Separate test because an unreachable
    ComfyUI fails the node probe OPEN (which the default test env is) — here the
    probe answers, and answers that the pack is absent."""
    monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                        lambda *a, **k: {'KSampler'})
    from app.services import krea_edit_helper as keh
    monkeypatch.setattr(keh, '_nodes_ok_until', 0.0, raising=False)
    ds = client.post('/api/dataset/create',
                     json={'name': 'KN', 'trigger_word': 'kntrig'}).get_json()['id']
    r = client.post(f'/api/dataset/{ds}/generate', json={
        'engine_batches': [{'generator': 'krea',
                            'variations': [{'label': 'Bust, front', 'framing': 'bust',
                                            'prompt': 'upper body portrait'}]}]})
    assert r.status_code == 409
    body = r.get_json()
    assert 'comfyui-krea2edit' in body['error'] and 'github.com' in body['error']
    assert 'restart ComfyUI' in body['error']
    assert set(body['krea_missing']['nodes']) == set(keh.KREA_NODE_CLASSES)
    assert body['krea_missing']['node_packs'], 'name the pack, not just the class'


def test_picking_krea_and_pressing_generate_installs_it(client, tmp_path, monkeypatch):
    """The whole point of the wave: selecting the engine and hitting Generate is
    the request to install it — the same trigger Klein has had. The 409 must fire
    the installs itself and SAY so, including the restart nobody can guess."""
    from app import setup_installer, config
    base = tmp_path / 'ComfyUI'
    (base / 'models').mkdir(parents=True)
    (base / 'main.py').write_text('# fake', encoding='utf-8')
    config.save_config({'comfyui': {'base_dir': str(base)}})
    monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                        lambda *a, **k: {'KSampler'})
    from app.services import krea_edit_helper as keh
    monkeypatch.setattr(keh, '_nodes_ok_until', 0.0, raising=False)
    started = []
    monkeypatch.setattr(setup_installer, 'start',
                        lambda action: started.append(action))
    ds = client.post('/api/dataset/create',
                     json={'name': 'KI', 'trigger_word': 'kitrig'}).get_json()['id']
    r = client.post(f'/api/dataset/{ds}/generate', json={
        'engine_batches': [{'generator': 'krea',
                            'variations': [{'label': 'Bust, front', 'framing': 'bust',
                                            'prompt': 'upper body portrait'}]}]})
    assert r.status_code == 409
    body = r.get_json()
    assert 'krea_nodes' in started
    assert {'krea_model', 'krea_text_encoder', 'krea_vae',
            'krea_identity_lora'} <= set(started)
    assert set(body['downloading']) == set(started)
    assert 'started downloading' in body['error']
    assert 'RESTART ComfyUI' in body['error']


def test_without_a_comfyui_folder_the_krea_409_still_explains_by_hand(client, monkeypatch):
    """Nothing can be installed with nowhere to install it — the message then has
    to keep the manual paths AND say which setting unlocks the automatic path."""
    monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                        lambda *a, **k: {'KSampler'})
    from app.services import krea_edit_helper as keh
    monkeypatch.setattr(keh, '_nodes_ok_until', 0.0, raising=False)
    ds = client.post('/api/dataset/create',
                     json={'name': 'KJ', 'trigger_word': 'kjtrig'}).get_json()['id']
    r = client.post(f'/api/dataset/{ds}/generate', json={
        'engine_batches': [{'generator': 'krea',
                            'variations': [{'label': 'Bust, front', 'framing': 'bust',
                                            'prompt': 'upper body portrait'}]}]})
    body = r.get_json()
    assert r.status_code == 409
    assert body['downloading'] == []
    assert 'Setup ▸ ComfyUI' in body['error']
    assert 'models/vae' in body['error']       # the by-hand paths survive


def test_an_unknown_engine_is_still_refused(client):
    ds = client.post('/api/dataset/create',
                     json={'name': 'K2', 'trigger_word': 'k2trig'}).get_json()['id']
    r = client.post(f'/api/dataset/{ds}/generate', json={
        'engine_batches': [{'generator': 'kreaa', 'variations': [{'label': 'x',
                                                                  'prompt': 'p'}]}]})
    assert r.status_code == 400
    assert 'unknown engine' in r.get_json()['error']


def test_nsfw_shots_are_allowed_on_krea_and_still_refused_on_an_api_engine(client):
    """The NSFW rule is "local only", not "Klein only" — widening it must not
    open the API lane by accident."""
    ds = client.post('/api/dataset/create',
                     json={'name': 'K3', 'trigger_word': 'k3trig'}).get_json()['id']
    shot = [{'label': 'nsfw_x', 'framing': 'body', 'prompt': 'p', 'nsfw': True}]
    api = client.post(f'/api/dataset/{ds}/generate', json={
        'engine_batches': [{'generator': 'chatgpt', 'variations': shot}]})
    assert api.status_code == 400
    assert 'local engine only' in api.get_json()['error']
    # Krea gets past the NSFW gate and stops on its OWN preflight (409), which is
    # the proof it was never refused for being NSFW.
    local = client.post(f'/api/dataset/{ds}/generate', json={
        'engine_batches': [{'generator': 'krea', 'variations': shot}]})
    assert local.status_code == 409
    assert 'krea_missing' in local.get_json()


def test_capabilities_publishes_the_krea_engine_and_its_gaps(client):
    caps = client.get('/api/capabilities').get_json()
    assert 'krea' in caps['engines']
    assert caps['engines']['krea'] is False, 'nothing installed in the test env'
    assert 'krea_missing' in caps['comfyui'] and 'krea_nodes_missing' in caps['comfyui']


def test_a_krea_row_is_badged_krea_and_a_legacy_klein_row_still_reads_klein():
    """`klein_model` carries an engine TAG for API + Krea rows and a model FILE
    for Klein ones. A wrong badge is worse than none."""
    from app.services import face_dataset_service as svc

    class Row:
        def __init__(self, value):
            self.klein_model = value

    assert svc._image_engine(Row('krea')) == 'krea'
    assert svc._image_engine(Row('chatgpt')) == 'chatgpt'
    assert svc._image_engine(Row('Krea\\krea2_turbo_fp8.safetensors')) == 'klein'
    assert svc._image_engine(Row(None)) is None


# --- The markings hold order must not SUMMON what it protects -----------------
# Reported within hours of shipping: "why do my Krea 2 generations
# always add tattoos?". The first version of KREA_MARKINGS_LOCK enumerated the
# features to preserve ("tattoos with the same design..., scars, moles and
# piercings") and was assumed to "cost nothing when the subject has no
# markings". A text encoder does not bind "reproduce X as in the reference": it
# reads the word and paints it. These tests pin the rule that came out of it.

_SUMMONABLE = ('tattoo', 'scar', 'mole', 'piercing', 'freckle', 'birthmark')


def test_the_markings_lock_names_no_feature_it_could_summon():
    from app.services.face_variations import KREA_MARKINGS_LOCK
    low = KREA_MARKINGS_LOCK.lower()
    named = [w for w in _SUMMONABLE if w in low]
    assert not named, (
        f'KREA_MARKINGS_LOCK names {named} — on a subject who has none, the '
        f'encoder paints them. Hold the skin, do not enumerate its features.')


def test_the_markings_lock_still_forbids_adding_and_redrawing():
    """Dropping the enumeration must not drop the protection."""
    from app.services.face_variations import KREA_MARKINGS_LOCK
    low = KREA_MARKINGS_LOCK.lower()
    assert 'do not add' in low, 'nothing forbids inventing new marks'
    assert 'redraw' in low, 'nothing forbids redrawing the existing ones'


def test_no_krea_prompt_names_a_summonable_feature():
    """The whole composed prompt, not just the constant — a feature named
    anywhere in it is a feature the model may paint."""
    from app.services.face_variations import VARIATION_CATALOG, wrap_variation_krea
    for entry in VARIATION_CATALOG[:12]:
        prompt = wrap_variation_krea(entry['prompt'], nsfw=False,
                                     framing=entry.get('framing'),
                                     label=entry.get('label', '')).lower()
        named = [w for w in _SUMMONABLE if w in prompt]
        assert not named, f"shot {entry['id']} names {named} in its Krea prompt"


def test_no_klein_prompt_names_a_summonable_feature():
    """Klein carries the same hold order since 2026-07-27, so it inherits the
    same trap: whatever a garment or a shot is renamed to, no composed Klein
    prompt may name a feature the encoder would paint on a subject who has none.
    ('scarf' contains 'scar' — that is the kind of addition this catches.)"""
    from app.services.face_variations import VARIATION_CATALOG, wrap_variation_klein
    for entry in VARIATION_CATALOG:
        prompt = wrap_variation_klein(entry['prompt'], nsfw=False,
                                      framing=entry.get('framing'),
                                      label=entry.get('label', '')).lower()
        named = [w for w in _SUMMONABLE if w in prompt]
        assert not named, f"shot {entry['id']} names {named} in its Klein prompt"


# --- Klein inherits the two local-edit fixes (measured 2026-07-27) ------------

def test_klein_holds_the_skin_and_gets_a_concrete_garment():
    """MEASURED, same seed, one factor at a time. The hold order: on a subject
    with NO markings it invented none (3/3), and on the tattooed one it kept a
    forehead piece that vanished without it. The concrete garment: obeyed 5/5,
    and it broke the "every wide shot ends in blue jeans" collapse the bare
    negation produced in 3/3."""
    from app.services import face_variations as fv
    out = fv.wrap_variation_klein('upper body portrait, ' + fv.OUTFIT_VARY,
                                  framing='bust', label='Bust, studio')
    assert fv.KREA_MARKINGS_LOCK.strip() in out
    assert fv.OUTFIT_VARY not in out and 'not the outfit' not in out
    assert any(g in out for g in fv.KREA_OUTFIT_PALETTE)
    # Same shot, same garment, run after run and process after process.
    assert out == fv.wrap_variation_klein('upper body portrait, ' + fv.OUTFIT_VARY,
                                          framing='bust', label='Bust, studio')


def test_the_api_engines_are_left_alone():
    """Nothing was measured on Nano Banana / ChatGPT / OpenRouter, so the two
    local-edit fixes must not reach the shared API wrapper. Extending a Klein
    result to another model family is the mistake this pins shut."""
    from app.services import face_variations as fv
    out = fv.wrap_variation('upper body portrait, ' + fv.OUTFIT_VARY)
    assert fv.OUTFIT_VARY in out, 'the API wrapper must keep the raw catalog text'
    assert fv.KREA_MARKINGS_LOCK.strip() not in out


def test_the_outfit_palette_spreads_over_the_catalog():
    """A palette too short trades a uniform for a quasi-uniform: at 12 garments
    the worst one carried 6 of the 41 eligible shots and only 11 were ever used.
    crc32-modulo means the load histogram depends on the palette SIZE, so this
    pins the OUTCOME — a catalog or palette edit that re-concentrates the picks
    fails here instead of silently shipping."""
    import collections
    from app.services import face_variations as fv
    # Only the shots that actually RECEIVE a garment: krea_outfit_directive also
    # strips the dead expression negation, which is not an outfit decision.
    eligible = [e for e in fv.VARIATION_CATALOG
                if any(g in fv.krea_outfit_directive(e['prompt'], e['label'])
                       for g in fv.KREA_OUTFIT_PALETTE)]
    assert len(eligible) > 30, 'catalog shrank — re-measure before relaxing this'
    counts = collections.Counter(fv.krea_outfit_for(e['label']) for e in eligible)
    assert max(counts.values()) <= 4, (
        f'garment overload: {counts.most_common(3)} over {len(eligible)} shots')
    assert len(counts) >= 20, f'only {len(counts)} distinct garments used'
