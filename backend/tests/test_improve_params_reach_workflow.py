"""Proof that the "Upscale & improve" settings actually reach ComfyUI.

The knobs were exposed in Settings, but exposing a field proves nothing: the value
still has to survive the config read, the call site, and the workflow patch, and
land on the RIGHT node. This captures the exact workflow dict that would be sent
to ComfyUI (the queue_manager.add_job seam) and asserts each value on its node —
no GPU, and repeatable, unlike trying one image by hand.

Node map of `improve skin.json`, verified against the shipped file:
  139 LoraLoaderModelOnly  -> strength_model : the enhancement LoRA (realistic detail)
  77  KSampler             -> steps
  174 ImageScaleToTotalPixels -> megapixels : the OUTPUT resolution
and lora_strength drives the consistency LoRA (klein.consistency_strength), which
anchors composition — it is NOT an identity LoRA.
"""
import json
from unittest.mock import patch

import pytest


@pytest.fixture()
def captured(app, tmp_path):
    """Run an improve enqueue and hand back the workflow that would be submitted."""
    def run(enhancement_lora_installed=True, **config):
        from app import config as cfg
        from app.services import klein_edit_helper as keh
        seen = {}

        src = tmp_path / 'src.png'
        src.write_bytes(b'\x89PNG\r\n\x1a\n')
        comfy_in = tmp_path / 'comfy_input'
        comfy_in.mkdir(exist_ok=True)
        # Node 139 carries klein/realistic.safetensors, which ships with NEITHER the
        # app nor the Klein install — enqueue_klein_edit bypasses the node when the
        # file is absent. Whether it is there decides if the strength does anything
        # at all, so the fixture makes that explicit instead of incidental.
        comfy = tmp_path / 'comfy'
        loras = comfy / 'models' / 'loras' / 'klein'
        loras.mkdir(parents=True, exist_ok=True)
        target = loras / 'realistic.safetensors'
        if enhancement_lora_installed:
            target.write_bytes(b'0')
        elif target.exists():
            target.unlink()
        cfg.save_config({'comfyui': {'base_dir': str(comfy)}})

        if config:
            cfg.save_config({'klein': config})
        with patch.object(keh.queue_manager, 'add_job',
                          side_effect=lambda **kw: seen.update(kw)), \
             patch.object(keh, '_comfy_input_dir', return_value=str(comfy_in)), \
             patch.object(keh, 'resolve_klein_unet', return_value='unet.safetensors'), \
             patch.object(keh, 'resolve_klein_vae', return_value='vae.safetensors'), \
             patch.object(keh, 'resolve_klein_text_encoder', return_value='te.safetensors'), \
             patch.object(keh, 'klein_missing_assets', return_value=[]):
            from app.services import face_dataset_service as fds
            keh.enqueue_klein_edit(
                user_id='local', source_filename='src.png', source_path=str(src),
                edit_prompt='improve',
                lora_strength=fds._improve_float('improve_consistency_strength', 0.0, 1.5),
                sampler_steps=fds._improve_int('improve_steps', 4),
                base_lora_strength=fds._improve_float('improve_base_lora_strength', 0.0),
                output_megapixels=fds._improve_float('improve_megapixels', 2.0, 8.0),
            )
        return seen['workflow_data']
    with app.app_context():
        yield run


def test_defaults_reproduce_the_shipped_behaviour(captured):
    """An untouched install must submit exactly what it submitted before the
    settings existed — otherwise adding a knob silently changed everyone's output."""
    w = captured()
    assert w['77']['inputs']['steps'] == 4
    assert w['139']['inputs']['strength_model'] == 0.0     # enhancement LoRA off
    assert w['174']['inputs']['megapixels'] == 2.0         # the historical 2 MP


def test_every_knob_lands_on_its_own_node(captured):
    w = captured(improve_base_lora_strength=0.75, improve_steps=9, improve_megapixels=4)
    assert w['139']['inputs']['strength_model'] == 0.75
    assert w['77']['inputs']['steps'] == 9
    assert w['174']['inputs']['megapixels'] == 4.0
    # untouched knobs keep the workflow's own values — we patch, never rebuild
    assert w['77']['inputs']['sampler_name'] == 'euler'
    assert w['77']['inputs']['cfg'] == 1


def test_output_size_is_what_makes_it_an_upscale(captured):
    """Node 174 rescales the source to a pixel budget BEFORE sampling, so it is the
    result's resolution. Hardcoded at 2 MP until now, which made "Upscale" produce
    the same size whatever you asked for."""
    assert captured(improve_megapixels=1)['174']['inputs']['megapixels'] == 1.0
    assert captured(improve_megapixels=6)['174']['inputs']['megapixels'] == 6.0


def test_the_consistency_lora_is_clamped_where_the_helper_clamps_it(captured):
    """It drives klein.consistency_strength, which enqueue_klein_edit caps at 1.5.
    Offering more in Settings would be a value the engine silently pulls back."""
    from app.services import face_dataset_service as fds
    from app import config as cfg
    cfg.save_config({'klein': {'improve_consistency_strength': 99}})
    assert fds._improve_float('improve_consistency_strength', 0.0, 1.5) == 1.5


def test_the_enhancement_strength_is_INERT_without_its_lora_file(captured):
    """The trap this whole exercise uncovered: klein/realistic.safetensors ships with
    neither the app nor the Klein install, and node 139 is BYPASSED when it is absent.
    On such a machine the Enhancement LoRA setting does nothing at all, silently —
    raising it changes no pixel. Asserted so the behaviour is documented rather than
    discovered again, and so the UI can be held to warning about it."""
    w = captured(enhancement_lora_installed=False, improve_base_lora_strength=0.8)
    assert '139' not in w                       # node removed, strength irrelevant
    # everything else still applies — only that one knob goes dead
    assert w['77']['inputs']['steps'] == 4
    assert w['174']['inputs']['megapixels'] == 2.0


def test_a_broken_config_still_submits_a_valid_workflow(captured):
    """A hand-edited config must degrade the pass, never break the enqueue."""
    w = captured(improve_steps='nonsense', improve_megapixels=None,
                 improve_base_lora_strength=[])
    assert w['77']['inputs']['steps'] == 4
    assert w['174']['inputs']['megapixels'] == 2.0
    assert w['139']['inputs']['strength_model'] == 0.0
    json.dumps(w)          # still serialisable, i.e. still submittable
