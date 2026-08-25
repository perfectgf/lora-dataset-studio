"""The FLUX.2 Klein generation lane for the Test Studio (GitHub #53).

Klein is the app's EDIT engine everywhere else: variations, improve, inpaint,
all of which start from an image. The Test Studio starts from a prompt, so this
lane is the only place Klein is asked to generate from nothing, and it needed a
graph of its own rather than a branch on somebody else's.

What is pinned here is what makes the lane trustworthy: the graph is built from
the SAME asset resolvers the rest of the Klein lane uses (never the names frozen
into the shipped JSON), the tested LoRA and any user-supplied base go through a
whitelist, and cfg cannot escape 1.0 on a guidance-distilled model.
"""
import json

import pytest

from app.services import lora_test_studio as studio


def _wf():
    return json.loads(studio.WORKFLOW_FLUX2KLEIN_PATH.read_text(encoding='utf-8'))


def test_the_graph_generates_from_a_prompt_not_from_an_image():
    """An empty latent is the whole difference from the Klein edit graphs."""
    classes = {n['class_type'] for n in _wf().values()}
    assert 'EmptyFlux2LatentImage' in classes
    assert 'LoadImage' not in classes, 'a test cell has no source image to edit'
    assert 'LoraLoaderModelOnly' in classes, 'the tested LoRA must reach the model'


def test_the_lora_sits_between_the_unet_and_the_sampler():
    """Wiring, not just presence: a LoRA node nothing consumes changes nothing."""
    w = _wf()
    assert w['29']['inputs']['model'] == ['20', 0], 'the LoRA must read the UNET'
    assert w['31']['inputs']['model'] == ['29', 0], 'sampling must read the LoRA'
    assert w['26']['inputs']['model'] == ['31', 0], 'the sampler must read that'


def test_assets_come_from_the_resolvers_never_from_the_shipped_json(monkeypatch):
    """The names in a captured workflow are true on ONE machine. Every other
    Klein path resolves them; so does this one."""
    from app.services import klein_edit_helper as keh
    monkeypatch.setattr(keh, 'unet_for_job', lambda *a, **k: 'klein/mine.safetensors')
    monkeypatch.setattr(keh, '_unet_weight_dtype', lambda *a, **k: 'fp8_e4m3fn')
    monkeypatch.setattr(keh, 'resolve_klein_text_encoder', lambda: 'te_mine.safetensors')
    monkeypatch.setattr(keh, 'resolve_klein_vae', lambda: 'vae_mine.safetensors')
    w = _wf()
    studio.apply_klein_lora_test_settings(
        w, lora_name='flux2klein/lora_x.safetensors', strength=0.8, prompt='a cat',
        seed=7, width=768, height=1024, steps=6, allowed_loras=None)
    assert w['20']['inputs']['unet_name'] == 'klein/mine.safetensors'
    assert w['21']['inputs']['clip_name'] == 'te_mine.safetensors'
    assert w['22']['inputs']['vae_name'] == 'vae_mine.safetensors'


def test_the_cell_axes_all_land_where_they_belong():
    w = _wf()
    studio.apply_klein_lora_test_settings(
        w, lora_name='flux2klein/lora_x.safetensors', strength=0.65, prompt='a cat',
        seed=1234, width=768, height=1024, steps=6, filename_prefix='pfx',
        allowed_loras=None)
    assert w['29']['inputs']['lora_name'] == 'flux2klein/lora_x.safetensors'
    assert w['29']['inputs']['strength_model'] == pytest.approx(0.65)
    assert w['23']['inputs']['text'] == 'a cat'
    assert w['26']['inputs']['seed'] == 1234 and w['26']['inputs']['steps'] == 6
    assert w['28']['inputs']['filename_prefix'] == 'pfx'
    # The size must reach BOTH the latent and the shift, or the shift is
    # computed for a picture nobody asked for.
    for node in ('25', '31'):
        assert w[node]['inputs']['width'] == 768
        assert w[node]['inputs']['height'] == 1024


def test_cfg_cannot_escape_one_on_a_guidance_distilled_model():
    """The studio sweeps cfg as an axis. Klein 9B diverges above 1, and a burnt
    cell reads to the user as a bad checkpoint rather than a bad setting."""
    w = _wf()
    studio.apply_klein_lora_test_settings(
        w, lora_name='l.safetensors', strength=1.0, prompt='p', seed=1,
        width=512, height=512, cfg=7.5, allowed_loras=None)
    assert w['26']['inputs']['cfg'] == 1.0


@pytest.mark.parametrize('kwargs, wrong', [
    ({'lora_name': '../../etc/passwd', 'allowed_loras': {'ok.safetensors'}}, 'LoRA'),
    ({'lora_name': 'ok.safetensors', 'allowed_loras': {'ok.safetensors'},
      'base_model': '../evil.safetensors', 'allowed_bases': {'klein/a.safetensors'}}, 'base'),
])
def test_a_name_outside_its_whitelist_is_refused(kwargs, wrong):
    """Both user-supplied names are guarded, exactly like the Krea lane."""
    with pytest.raises(ValueError, match=wrong):
        studio.apply_klein_lora_test_settings(
            _wf(), strength=1.0, prompt='p', seed=1, width=512, height=512, **kwargs)


def test_an_elected_base_is_not_forced_through_the_user_whitelist():
    """No base_model given means the app elects one; that value is ours, not a
    request, so it must not be judged against a whitelist meant for user input."""
    w = _wf()
    studio.apply_klein_lora_test_settings(
        w, lora_name='ok.safetensors', strength=1.0, prompt='p', seed=1,
        width=512, height=512, allowed_loras={'ok.safetensors'},
        allowed_bases={'nothing-matches'})
    assert w['20']['inputs']['unet_name']       # elected, not refused
