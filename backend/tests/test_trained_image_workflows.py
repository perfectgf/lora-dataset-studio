"""Architecture, adapter chaining and model-selection contracts (no GPU)."""
import pytest

from app.utils.trained_image_workflows import (
    ASSET_SPECS, REQUIRED_NODES, build_trained_image_workflow, family_defaults,
)


def _build(family, **kwargs):
    assets = {key: spec['filename'] for key, spec in ASSET_SPECS[family].items()}
    options = {'assets': assets, 'allowed_bases': {assets['diffusion_model']},
               'allowed_loras': {f'{family}/subject.safetensors', f'{family}/style.safetensors'},
               'available_classes': set(REQUIRED_NODES[family])}
    options.update(kwargs)
    return build_trained_image_workflow(family, **options)


@pytest.mark.parametrize('family', ('flux', 'anima', 'qwenimage21'))
def test_lora_stack_reaches_sampler_and_graph_has_no_dangling_links(family):
    graph = _build(family, prompt='a painted portrait', negative='blur',
                   seed=2**63 + 17, width=768, height=1024,
                   loras=[{'filename': f'{family}/subject.safetensors', 'strength': 0.7},
                          {'filename': f'{family}/style.safetensors', 'strength': -0.2}])
    for node in graph.values():
        for value in node['inputs'].values():
            if isinstance(value, list):
                assert value[0] in graph
    assert graph['lora_0']['inputs']['model'] == ['1', 0]
    assert graph['lora_1']['inputs']['model'] == ['lora_0', 0]
    model_consumer = graph['sampling'] if family == 'flux' else graph['7']
    assert model_consumer['inputs']['model'] == ['lora_1', 0]
    assert graph['lora_1']['inputs']['strength_model'] == -0.2
    assert all(n['class_type'] != 'LoraLoader' for n in graph.values())
    assert graph['7']['inputs']['seed'] == 2**63 + 17
    assert graph['6']['inputs'] == {'width': 768, 'height': 1024, 'batch_size': 1}
    assert graph['9']['inputs']['images'] == ['8', 0]


def test_flux_uses_dual_encoder_distilled_guidance_and_16_channel_latent():
    graph = _build('flux', cfg=4.5)
    assert graph['2']['class_type'] == 'DualCLIPLoader'
    assert graph['2']['inputs']['type'] == 'flux'
    assert graph['2']['inputs']['clip_name1'] == 'clip_l.safetensors'
    assert graph['6']['class_type'] == 'EmptySD3LatentImage'
    assert graph['guidance']['inputs']['guidance'] == 4.5
    assert graph['7']['inputs']['cfg'] == 1.0
    assert graph['7']['inputs']['positive'] == ['guidance', 0]


def test_anima_uses_its_small_qwen_encoder_and_real_negative_conditioning():
    graph = _build('anima', negative='low quality', cfg=5.0)
    assert graph['2']['inputs']['type'] == 'stable_diffusion'
    assert graph['2']['inputs']['clip_name'] == 'qwen_3_06b_base.safetensors'
    assert graph['3']['inputs']['vae_name'] == 'qwen_image_vae.safetensors'
    assert graph['5']['inputs']['text'] == 'low quality'
    assert graph['7']['inputs']['negative'] == ['5', 0]
    assert graph['7']['inputs']['cfg'] == 5.0


def test_moss_never_renders_past_four_megapixels():
    from app.services.lora_test_studio import _aspect_dims, generation_capabilities
    from app.utils.trained_image_workflows import MOSS_MAX_PIXELS
    # Max tier x1.9 asks ~5.8 MP; OpenMOSS documents MOSS-Image up to 4 MP (2048²).
    w, h = _aspect_dims('3:4', 'moss', 'max', 1.9)
    assert w * h <= MOSS_MAX_PIXELS and w % 16 == h % 16 == 0 and abs(w / h - 3 / 4) < 0.01
    assert _aspect_dims('3:4', 'krea', 'max', 1.9) != (w, h)
    latent = _build('moss', width=2560, height=2048)['6']['inputs']
    assert latent['width'] * latent['height'] <= MOSS_MAX_PIXELS
    assert generation_capabilities('moss')['max_pixels'] == MOSS_MAX_PIXELS
    assert 'max_pixels' not in generation_capabilities('krea')


def test_moss_matches_the_validated_runtime_and_keeps_shift_before_skip_guidance():
    graph = _build('moss', prompt='a fox', negative='blur')
    assert graph['1']['class_type'] == 'LDSMossUNETLoader'
    assert graph['2']['class_type'] == 'LDSMossCLIPLoader'
    assert graph['2']['inputs']['clip_name'] == 'qwen3.5_9b_int8_convrot.safetensors'
    assert graph['6']['class_type'] == 'EmptyFlux2LatentImage'
    assert graph['sampling']['inputs'] == {'model': ['1', 0], 'shift': 5.0}
    assert graph['slg']['inputs']['model'] == ['sampling', 0]
    assert graph['slg']['inputs']['single_layers'] == '22'
    assert graph['7']['inputs']['model'] == ['slg', 0]
    assert graph['7']['inputs']['steps'] == 50
    assert graph['7']['inputs']['cfg'] == 3.0
    assert graph['7']['inputs']['scheduler'] == 'sgm_uniform'
    assert graph['5']['inputs']['text'] == 'blur'
    assert _build('moss', steps=30, cfg=4)['7']['inputs']['steps'] == 30
    with pytest.raises(ValueError, match='base model only'):
        _build('moss', loras=[{'filename': 'moss/style.safetensors'}])


def test_qwen21_uses_new_native_conditioner_and_rgba_vae_not_older_qwen():
    graph = _build('qwenimage21', prompt='portrait', negative='blur', batch_size=2,
                   weight_dtype='fp8_e4m3fn')
    assert graph['4']['class_type'] == 'TextEncodeQwenImage21'
    assert graph['4']['inputs']['negative_prompt'] == 'blur'
    assert graph['2']['inputs']['clip_name'].startswith('qwen3vl_8b_')
    assert graph['3']['inputs']['vae_name'] == 'qwen_image_2.1_vae_bf16.safetensors'
    assert graph['7']['inputs']['negative'] == ['4', 1]
    assert graph['7']['inputs']['cfg'] == 1.0
    assert graph['6']['inputs']['batch_size'] == 2
    assert graph['1']['inputs']['weight_dtype'] == 'default'  # prequantized convrot


@pytest.mark.parametrize('name', ('../bad.safetensors', '/bad.safetensors',
                                  r'\bad.safetensors', r'Z:\bad.safetensors',
                                  r'flux\..\bad.safetensors', 'flux/bad.safetensors:stream',
                                  'flux/./bad.safetensors', 'flux//bad.safetensors'))
def test_even_allowlisted_paths_cannot_escape_model_roots(name):
    with pytest.raises(ValueError, match='relative model filename'):
        _build('flux', base_model=name, allowed_bases={name})
    with pytest.raises(ValueError, match='relative model filename'):
        _build('flux', loras=[{'filename': name}], allowed_loras={name})


def test_allowlists_and_explicit_lora_family_are_enforced():
    with pytest.raises(ValueError, match='Unknown base model'):
        _build('flux', allowed_bases=set())
    with pytest.raises(ValueError, match='Unknown LoRA'):
        _build('anima', loras=[{'filename': 'qwenimage21/subject.safetensors'}])
    with pytest.raises(ValueError, match='architecture'):
        _build('anima', loras=[{'filename': 'anima/subject.safetensors', 'family': 'flux'}])
    with pytest.raises(ValueError, match='explicit family allowlist'):
        _build('flux', loras=[{'filename': 'flux/subject.safetensors'}], allowed_loras=None)


def test_missing_qwen21_node_is_an_actionable_update_error():
    classes = set(REQUIRED_NODES['qwenimage21']) - {'TextEncodeQwenImage21'}
    with pytest.raises(ValueError, match='Update ComfyUI.*TextEncodeQwenImage21'):
        _build('qwenimage21', available_classes=classes)


def test_schnell_keeps_native_sampling_and_explicit_step_count():
    kwargs = {'base_model': 'flux1-schnell.safetensors',
              'allowed_bases': {'flux1-schnell.safetensors'}}
    graph = _build('flux', **kwargs)
    assert 'sampling' not in graph
    assert graph['7']['inputs']['steps'] == 4
    assert _build('flux', **kwargs, steps=20)['7']['inputs']['steps'] == 20


def test_unknown_family_never_falls_back_and_defaults_are_copied():
    with pytest.raises(ValueError, match='Unsupported trained image family'):
        build_trained_image_workflow('typo', assets={}, allowed_bases=set(), allowed_loras=set())
    defaults = family_defaults('anima')
    defaults['cfg'] = 999
    assert family_defaults('anima')['cfg'] == 4.0


@pytest.mark.parametrize('kwargs', ({'cfg': float('nan')}, {'steps': float('inf')},
                                   {'seed': -1}, {'seed': 2**64}, {'width': 1000},
                                   {'loras': [{'filename': 'flux/subject.safetensors',
                                               'strength': float('nan')}]}))
def test_invalid_numeric_inputs_fail_before_enqueue(kwargs):
    with pytest.raises(ValueError):
        _build('flux', **kwargs)
