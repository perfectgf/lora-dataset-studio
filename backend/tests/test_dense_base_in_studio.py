"""A delivered full model must be selectable as a Test Studio base.

THE ONE LINE THAT HID IT
------------------------
``get_krea_models`` accepted a checkpoint only if its RELATIVE DIRECTORY carried
'krea' (or if it was literally the wired default at the root). The fp8 twin of a
dense run is written to the ROOT of ComfyUI's ``diffusion_models`` — that is
where ``fp8_local_delivery`` puts it, deliberately, because that is where ComfyUI
looks. So the file the whole dense lane exists to produce was invisible to the
one screen meant to try it.

The asymmetry proves it was an oversight rather than a rule: the Generate surface
(``krea_edit_helper._krea_unet_folders``) has always matched 'krea' in the folder
OR in the filename, root included. This aligns the Studio with that rule, which
retro-fits every twin already delivered without moving a byte.

WHAT COMES FREE ONCE IT IS LISTED: ``krea_build_of`` already reads ``_fp8`` as a
RAW build, and ``krea_model_defaults`` already answers with the training recipe's
own sample settings (CFG 4 / 25 steps). The base picker seeds its axes from those.
"""
import pytest


@pytest.fixture()
def comfy(app, tmp_path, monkeypatch):
    """A ComfyUI tree with a diffusion_models folder, wired the way the app reads
    it (`_out_dir()/../models/<base>`), caches cleared."""
    from app.utils import comfyui
    root = tmp_path / 'ComfyUI'
    (root / 'models' / 'diffusion_models').mkdir(parents=True)
    (root / 'models' / 'unet').mkdir(parents=True)
    (root / 'output').mkdir(parents=True)
    monkeypatch.setattr(comfyui, '_out_dir', lambda: str(root / 'output'))
    comfyui.clear_model_caches()
    yield root / 'models' / 'diffusion_models'
    comfyui.clear_model_caches()


def _put(folder, name):
    (folder / name).write_bytes(b'W')


# --- the fix -----------------------------------------------------------------------

def test_a_delivered_fp8_twin_at_the_root_is_listed(comfy):
    """The exact filename fp8_local_delivery writes, in the exact place."""
    from app.utils.comfyui import get_krea_models
    _put(comfy, 'Krea_full_subject1_000002500_fp8.safetensors')
    assert 'Krea_full_subject1_000002500_fp8.safetensors' in get_krea_models()


def test_a_root_file_with_no_krea_in_its_name_is_still_ignored(comfy):
    """The rule is 'krea' in the folder OR the filename — not 'every root file'.
    A diffusion_models root holds Z-Image, FLUX and Klein weights too."""
    from app.utils.comfyui import get_krea_models
    _put(comfy, 'flux2-klein-9b.safetensors')
    _put(comfy, 'zimage_turbo.safetensors')
    assert get_krea_models() == []


def test_a_build_measured_to_render_noise_is_still_offered(comfy):
    """BigLove* carries 'krea' and renders pure noise under the Krea pipeline —
    measured. The Studio used to drop it, so a file the user had put there was
    absent from the list with nothing saying it existed or why.

    It is listed now. The measurement is a warning the user is entitled to
    overrule; what it still does is stop the app PREFERRING it when nobody picked
    (test_krea_default_base_election)."""
    from app.utils.comfyui import get_krea_models
    _put(comfy, 'BigLoveKreaEdit1_fp8mixed.safetensors')
    assert get_krea_models() == ['BigLoveKreaEdit1_fp8mixed.safetensors']


def test_a_krea_subfolder_still_wins_everything_it_used_to(comfy):
    """Regression: the old rule keeps every file it listed, including one whose
    name says nothing — the folder is the claim there."""
    from app.utils.comfyui import get_krea_models
    sub = comfy / 'Krea'
    sub.mkdir()
    _put(sub, 'myMerge.safetensors')
    models = get_krea_models()
    assert any(m.endswith('myMerge.safetensors') for m in models)


def test_the_wired_default_at_the_root_is_still_offered(comfy):
    from app.utils.comfyui import get_krea_models
    _put(comfy, 'krea2_turbo_fp8.safetensors')
    assert 'krea2_turbo_fp8.safetensors' in get_krea_models()


def test_the_alternatives_list_still_excludes_the_wired_default(comfy):
    """`krea_alt_base_models` drops the default so the picker does not show the
    same model twice — the validation whitelist keeps it."""
    from app.services.lora_test_studio import krea_alt_base_models
    _put(comfy, 'krea2_turbo_fp8.safetensors')
    _put(comfy, 'Krea_full_x_fp8.safetensors')
    assert krea_alt_base_models() == ['Krea_full_x_fp8.safetensors']


def test_a_yaml_declared_root_is_seen_too(app, tmp_path, monkeypatch):
    """A model kept outside <ComfyUI>/models through extra_model_paths.yaml was
    invisible here while the Z-Image lister has read those roots for a while.
    Same argument, same fix: the picker must see what ComfyUI sees."""
    from app.services import comfy_model_paths
    from app.utils import comfyui
    root = tmp_path / 'ComfyUI'
    (root / 'models' / 'diffusion_models').mkdir(parents=True)
    (root / 'output').mkdir(parents=True)
    extra = tmp_path / 'elsewhere'
    extra.mkdir()
    (extra / 'Krea_full_x_fp8.safetensors').write_bytes(b'W')
    monkeypatch.setattr(comfyui, '_out_dir', lambda: str(root / 'output'))
    monkeypatch.setattr(comfy_model_paths, 'extra_roots',
                        lambda folder_type: [str(extra)])
    comfyui.clear_model_caches()
    try:
        assert 'Krea_full_x_fp8.safetensors' in comfyui.get_krea_models()
    finally:
        comfyui.clear_model_caches()


# --- what it unlocks ----------------------------------------------------------------

def test_the_twin_is_read_as_an_undistilled_build(comfy):
    """This is what makes the settings right without a single new constant."""
    from app.services.lora_test_studio import krea_build_of, krea_model_defaults
    name = 'Krea_full_subject1_000002500_fp8.safetensors'
    assert krea_build_of(name) == 'raw'
    assert krea_model_defaults(name) == {'cfg': 4.0, 'steps': 25}


def test_the_comparison_branch_now_publishes_the_per_base_settings(client, comfy):
    """`/studio/base-models` fed the comparison / blend screen with no
    `model_defaults`, so a Raw base there silently kept Turbo's cfg 1 / 8 steps —
    the exact mush the solo payload had already been fixed for.

    The delivered twin is now the ELECTED default (it is the only Krea base on
    disk), so it is no longer repeated in the alternatives — the picker would show
    the same file twice. What must survive is the settings, and they must reach
    BOTH the '' key and the family fallback, because the screen has no selector to
    read them from."""
    _put(comfy, 'Krea_full_x_fp8.safetensors')
    body = client.get('/api/studio/base-models?type=krea').get_json()
    assert body['model_defaults'][''] == {'cfg': 4.0, 'steps': 25}
    assert body['axes']['default_cfg'] == 4.0 and body['axes']['default_steps'] == 25


def test_a_second_local_base_is_offered_next_to_the_elected_one(client, comfy):
    """With an alternative on disk the picker comes back, and the alternative
    keeps its own settings."""
    _put(comfy, 'Krea_full_x_fp8.safetensors')
    _put(comfy, 'krea_turbo_merge.safetensors')
    body = client.get('/api/studio/base-models?type=krea').get_json()
    # The turbo build wins the election (the graph is distilled), so the Raw twin
    # is the one offered as an alternative.
    assert [m['filename'] for m in body['models']] == ['', 'Krea_full_x_fp8.safetensors']
    assert body['model_defaults']['Krea_full_x_fp8.safetensors'] == {'cfg': 4.0, 'steps': 25}


def test_the_default_entry_carries_the_settings_of_the_file_it_elects(client, comfy):
    """It used to carry the family's Turbo numbers because the '' entry stood for
    the wired UNET, which was always a Turbo build. It now stands for whatever was
    elected — so pinning it to the family constants would put an undistilled base
    back on cfg 1 / 8 steps, which is the bug, not the guard."""
    _put(comfy, 'Krea_full_x_fp8.safetensors')     # reads as a RAW build
    body = client.get('/api/studio/base-models?type=krea').get_json()
    assert body['model_defaults'][''] == {'cfg': 4.0, 'steps': 25}


def test_the_other_families_are_unchanged(client, comfy):
    """Additive only: an sdxl/zimage caller reads exactly what it always did."""
    for kind in ('sdxl', 'zimage'):
        body = client.get(f'/api/studio/base-models?type={kind}').get_json()
        assert 'models' in body and 'axes' in body
        assert isinstance(body.get('model_defaults', {}), dict)


def test_a_base_outside_the_list_is_still_refused(app, comfy):
    """The widening must not widen the WHITELIST beyond what is on disk: the
    picker's value goes into a workflow node, and '..\\evil.safetensors' has
    always been refused there."""
    from app.services import lora_test_studio as lts
    _put(comfy, 'Krea_full_x_fp8.safetensors')
    allowed = set(lts.get_krea_models())
    assert 'Krea_full_x_fp8.safetensors' in allowed
    assert '..\\evil.safetensors' not in allowed
    with pytest.raises(ValueError):
        lts.apply_krea_lora_test_settings(
            {'20': {'inputs': {'unet_name': 'krea2_turbo_fp8.safetensors'}}},
            lora_name=None, strength=1.0, prompt='p', seed=1, width=8, height=8,
            base_model='..\\evil.safetensors', allowed_bases=allowed)


def test_a_raw_base_carries_the_negative_prompt_to_the_sampler(app, comfy):
    """A Raw base samples at CFG 4, where ComfyUI evaluates the negative branch.
    Krea was left out of the negative-prompt families: the panel hid the field,
    the knobs dropped the text and the cell builder never wrote node 24."""
    from app.services import lora_test_studio as lts
    caps = lts.generation_capabilities('krea')
    assert caps['negative_prompt'] and caps['negative_prompt_needs_cfg']
    assert lts._sanitize_gen_knobs('krea', negative=' blurry ')['negative'] == 'blurry'
    with app.app_context():
        wf = lts._build_cell_workflow(1, None, 1.0, 'a photo', 7, None, set(), cfg=4.0,
                                      train_type='krea', negative='blurry')
    assert wf['24']['inputs']['text'] == 'blurry'
    assert wf['26']['inputs']['cfg'] == 4.0
