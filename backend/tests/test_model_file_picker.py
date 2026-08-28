"""The list behind the model-file pickers, and the refusal that protects it.

Two questions, and the second is the expensive one:

  1. does the picker offer exactly what the RESOLVER can load? A second way of
     listing model files is how a dropdown starts offering a file the graph
     refuses (or hiding one it would happily load, the day an install has an
     extra_model_paths.yaml);
  2. what happens when a pinned file is NOT there? It used to be a log line and
     a silent election — the failure mode that ran a whole training on a
     third-party finetune nobody chose.
"""
import os

import pytest

from app.services import comfy_model_picker as picker


def _write(path, size=2048):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'\0' * size)


@pytest.fixture
def comfy(app, tmp_path, monkeypatch):
    """A ComfyUI tree with one file in each folder the pickers scan."""
    from app import config as cfg
    base = tmp_path / 'ComfyUI'
    (base / 'models').mkdir(parents=True)
    with app.app_context():
        cfg.save_config({'comfyui': {'base_dir': str(base)}})
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    picker.clear_cache()
    yield base
    comfy_model_paths.clear_cache()
    picker.clear_cache()


def test_each_slot_lists_its_own_folder_and_says_where_that_is(app, comfy):
    _write(comfy / 'models' / 'unet' / 'klein' / 'flux-2-klein-9b-fp8.safetensors')
    _write(comfy / 'models' / 'text_encoders' / 'qwen_3_8b_fp8mixed.safetensors')
    _write(comfy / 'models' / 'vae' / 'flux2-vae.safetensors')
    _write(comfy / 'models' / 'loras' / 'klein' / 'consistency.safetensors')
    with app.app_context():
        unet, hint = picker.list_slot_files('klein_unet')
        assert unet == [os.path.join('klein', 'flux-2-klein-9b-fp8.safetensors')], (
            'the name must be the loader-relative string the field already stored — '
            'anything else would need an alias for every install that typed one')
        assert 'unet' in hint, 'the empty state has to be able to say WHERE to put a file'
        assert picker.list_slot_files('klein_vae')[0] == ['flux2-vae.safetensors']
        assert picker.list_slot_files('klein_text_encoder')[0] == [
            'qwen_3_8b_fp8mixed.safetensors']
        assert picker.list_slot_files('klein_consistency_lora')[0] == [
            os.path.join('klein', 'consistency.safetensors')]


def test_the_krea_base_list_is_the_RESOLVER_s_candidates_not_the_whole_folder(app, comfy):
    """The picker lists what the RESOLVER can build, not everything in
    diffusion_models — offering a file the resolver will not load is a choice that
    silently does nothing.

    That is a question about the FOLDER, not about the build. BigLove is listed:
    it renders noise under the identity LoRA, which is a measured warning the user
    is entitled to overrule, not a reason to hide a file from their own disk. What
    the resolver still declines is to PREFER it when nobody picked (see
    test_krea_default_base_election)."""
    d = comfy / 'models' / 'diffusion_models' / 'Krea'
    _write(d / 'krea2_turbo_fp8_scaled.safetensors')
    _write(d / 'BigLoveKreaEdit1_fp8mixed.safetensors')
    _write(comfy / 'models' / 'unet' / 'some_other_model.safetensors')
    with app.app_context():
        files, _hint = picker.list_slot_files('krea_base_model')
    assert any('krea2_turbo' in f for f in files)
    assert any('BigLove' in f for f in files), (
        'the picker hides a file sitting in the user own Krea folder')
    assert not any('some_other_model' in f for f in files), (
        'a non-krea folder file is not a Krea base candidate')


def test_the_camera_unet_list_is_the_resolvers_candidates_too(app, comfy):
    """Same doctrine as the two other UNET slots: the 📷 Model row lists the
    lane's own candidate walk — qwen-named folders anywhere under a search
    root, plus root-level files whose NAME carries the claim — never the whole
    diffusion_models tree and never another family's folder. Inside a qwen/
    folder the directory carries the claim, so an NSFW finetune with no 'qwen'
    in its filename is still offered."""
    q = comfy / 'models' / 'diffusion_models' / 'qwen'
    _write(q / 'qwen_image_edit_2511_fp8mixed.safetensors')
    _write(q / 'rapid-aio-finetune.safetensors')
    _write(comfy / 'models' / 'diffusion_models' / 'my-qwen-edit-merge.safetensors')
    _write(comfy / 'models' / 'diffusion_models' / 'unrelated-family.safetensors')
    _write(comfy / 'models' / 'unet' / 'klein' / 'not-this-family.safetensors')
    with app.app_context():
        files, hint = picker.list_slot_files('camera_unet', force=True)
    assert os.path.join('qwen', 'qwen_image_edit_2511_fp8mixed.safetensors') in files
    assert os.path.join('qwen', 'rapid-aio-finetune.safetensors') in files
    assert 'my-qwen-edit-merge.safetensors' in files
    assert not any('unrelated-family' in f for f in files), (
        'a root-level file with no qwen in its name is not a camera candidate')
    assert not any('not-this-family' in f for f in files)
    assert 'qwen' in hint, 'the empty state has to say WHERE to put a file'


def test_an_unknown_slot_and_an_unconfigured_comfyui_degrade_to_free_text(app, tmp_path):
    """Never an error: the field falls back to plain text rather than blocking
    the whole Settings panel on an install that has no ComfyUI at all."""
    with app.app_context():
        assert picker.list_slot_files('not_a_slot') == ([], '')
        files, hint = picker.list_slot_files('klein_vae')
        assert files == []
        assert hint


def test_the_endpoint_answers_the_shape_the_picker_reads(client, app, comfy):
    _write(comfy / 'models' / 'vae' / 'flux2-vae.safetensors')
    with app.app_context():
        picker.clear_cache()
    r = client.get('/api/comfy/model-files?slot=klein_vae')
    assert r.status_code == 200
    body = r.get_json()
    assert body['files'] == ['flux2-vae.safetensors']
    assert body['folder']
    # A junk slot from a stale client is a 200 with nothing, not a 500.
    assert client.get('/api/comfy/model-files?slot=../../etc').status_code == 200


def test_a_broken_unet_pin_asks_for_a_model_instead_of_killing_the_engine(app, comfy):
    """A stale pin must not switch the engine off while usable models sit on disk.

    THIS REPLACES the previous decision, deliberately. That one made
    `klein_engine_ready` False on any pin gap, which darkened the whole engine
    card — including the model picker sitting right under it. The user could see
    four valid Klein builds and had no way to say "use that one": the only exit
    was a Settings field on another page. Blocking the surface that would have
    fixed the problem is a worse answer than the problem.

    The property the old decision protected is NOT relaxed: nothing may ever run
    on a file the screen did not show. It moves to where it belongs — the run
    itself (see the two tests below) — instead of being enforced by removing the
    user's ability to choose.
    """
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    _write(comfy / 'models' / 'unet' / 'klein' / 'flux-2-klein-9b-fp8.safetensors')
    with app.app_context():
        cfg.save_config({'klein': {'unet': 'klein/a-model-i-deleted.safetensors'}})
        gaps = keh.klein_pin_gaps()
        assert [g['slot'] for g in gaps] == ['unet']
        assert gaps[0]['configured'] == 'klein/a-model-i-deleted.safetensors'
        # The gap is still REPORTED — the card names the file and asks for a pick.
        # Ingredients passed in so this asserts the PIN rule and not the asset
        # scan: the fixture writes the UNET only, and a missing VAE would refuse
        # for a reason that has nothing to do with what is under test here.
        assert keh.klein_engine_ready(
            True, missing=[], invalid=[], unsupported_enums=[]) is True
        # Clearing the field is still the explicit way back to auto-detection.
        cfg.save_config({'klein': {'unet': ''}})
        assert keh.klein_pin_gaps() == []


def test_a_broken_unet_pin_with_no_model_on_disk_still_keeps_the_engine_dark(app, comfy):
    """Nothing to choose from = nothing to ask the user. The card stays off."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    with app.app_context():
        cfg.save_config({'klein': {'unet': 'klein/a-model-i-deleted.safetensors'}})
        assert keh.klein_engine_ready(
            True, missing=[], invalid=[], unsupported_enums=[]) is False


def test_a_run_that_names_no_model_refuses_rather_than_auto_detecting(app, comfy):
    """The incident, pinned. A broken pin + no explicit pick used to resolve to
    whatever the scan found, so the job ran on a file nobody chose and the result
    was indistinguishable from a correct one. It now REFUSES, by name."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    _write(comfy / 'models' / 'unet' / 'klein' / 'flux-2-klein-9b-fp8.safetensors')
    with app.app_context():
        cfg.save_config({'klein': {'unet': 'klein/a-model-i-deleted.safetensors'}})
        with pytest.raises(keh.KleinModelGone) as e:
            keh.unet_for_job()
        assert 'a-model-i-deleted.safetensors' in str(e.value)


def test_a_run_that_names_a_model_uses_exactly_that_one(app, comfy):
    """The way out the card now offers: an explicit pick overrides the stale pin,
    and loads the file the screen showed — never a neighbour of it."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    _write(comfy / 'models' / 'unet' / 'klein' / 'flux-2-klein-9b-fp8.safetensors')
    with app.app_context():
        cfg.save_config({'klein': {'unet': 'klein/a-model-i-deleted.safetensors'}})
        chosen = keh.unet_for_job('klein/flux-2-klein-9b-fp8.safetensors')
        assert chosen.replace('/', os.sep).endswith(
            os.path.join('klein', 'flux-2-klein-9b-fp8.safetensors'))


def test_the_shipped_consistency_lora_name_is_not_a_user_pin(app, comfy):
    """klein.consistency_lora ships a NON-EMPTY default, so 'set' is not
    'chosen'. Treating it as a pin would brick every install that never
    downloaded that optional LoRA — the whole point of its documented 'skipped'
    behaviour."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    with app.app_context():
        shipped = (cfg.DEFAULTS.get('klein') or {}).get('consistency_lora')
        assert shipped, 'this test is meaningless if the default becomes blank'
        cfg.save_config({'klein': {'consistency_lora': shipped}})
        assert keh.klein_pin_gaps() == []
        cfg.save_config({'klein': {'consistency_lora': 'klein/mine.safetensors'}})
        assert [g['slot'] for g in keh.klein_pin_gaps()] == ['consistency_lora']


def test_the_klein_unet_list_is_the_RESOLVER_s_candidates_too(app, comfy):
    """The Klein UNET slot had the same defect the Krea base was protected from,
    and one consequence more.

    A picker built on its own walk of diffusion_models offers every file in
    there, so it can hand over one the resolver will not build — a choice that
    silently does nothing. But Klein ALSO has a second, older picker: the
    per-dataset one in KleinModelSetting, whose list comes from the capabilities
    probe and has always been filtered to klein-named candidates. Two dropdowns
    driving the SAME UNETLoader with different contents is the drift the repo
    warned about in that component's own header, and it is invisible on a machine
    whose diffusion_models holds nothing else."""
    d = comfy / 'models' / 'diffusion_models' / 'klein'
    _write(d / 'flux-2-klein-9b-fp8.safetensors')
    _write(comfy / 'models' / 'diffusion_models' / 'some_sdxl_unet.safetensors')
    with app.app_context():
        files, hint = picker.list_slot_files('klein_unet')
    assert any('flux-2-klein' in f for f in files)
    assert not any('some_sdxl' in f for f in files), (
        'the picker offered a UNET the Klein resolver refuses to build')
    assert hint


def test_both_unet_pickers_agree_with_what_capabilities_publishes(app, comfy):
    """The two ends, side by side, on one tree. This is the assertion that would
    have caught the divergence at the commit that introduced it: what Settings
    offers globally and what the per-dataset control offers must be the same set,
    because they write to the same loader."""
    d = comfy / 'models' / 'diffusion_models' / 'klein'
    _write(d / 'flux-2-klein-9b-fp8.safetensors')
    _write(d / 'flux-2-klein-9b-bf16.safetensors')
    _write(comfy / 'models' / 'diffusion_models' / 'some_sdxl_unet.safetensors')
    from app import capabilities
    with app.app_context():
        globally, _hint = picker.list_slot_files('klein_unet', force=True)
        per_dataset = capabilities._scan_models().get('klein') or []
    assert sorted(os.path.basename(f) for f in globally) == \
        sorted(os.path.basename(f) for f in per_dataset), (
            'the Settings picker and the per-dataset picker disagree about which '
            'files the same UNETLoader can load')


def test_a_bare_name_the_picker_itself_writes_is_not_a_gap(app, comfy):
    """THE self-inflicted brick, pinned.

    `set_dataset_klein_model` — what the run panel's dropdown calls — writes the
    global `klein.unet` and REFUSES anything carrying a folder ("the loader
    prefix is resolve_klein_unet's job"). So the value it stores for a model in a
    klein/ sub-folder is the BARE file name.

    `klein_pin_gaps` judged that same key with `resolve_model_ref`, which only
    looks for a relative name at the ROOT of a search folder. Bare name + file in
    a sub-folder = 'missing' — so choosing a model from the app's own dropdown
    wrote a value the app's own gate then called broken, and the engine went dark
    with "no missing or broken file" as its explanation. Nothing the user typed
    was ever wrong.

    The gate now asks the loader (`klein_model_on_disk`, the same scan the picker
    lists from) before declaring a gap. A file that is genuinely gone still is
    one — see the test above.
    """
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    _write(comfy / 'models' / 'unet' / 'klein' / 'flux-2-klein-9b-fp8.safetensors')
    with app.app_context():
        # Exactly what the dropdown stores: no folder, because it may not store one.
        cfg.save_config({'klein': {'unet': 'flux-2-klein-9b-fp8.safetensors'}})
        assert keh.klein_pin_gaps() == []
        assert keh.klein_engine_ready(
            True, missing=[], invalid=[], unsupported_enums=[]) is True
        # …and the run loads THAT file, prefix restored by the loader.
        assert keh.unet_for_job().replace('/', os.sep) == os.path.join(
            'klein', 'flux-2-klein-9b-fp8.safetensors')
