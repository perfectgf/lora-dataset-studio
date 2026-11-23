"""Preparation must resolve the same files Studio will load, without a GPU."""
import json
import struct

import pytest

from app.services import trained_image_models as models


def weights(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps({'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
    path.write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * 4)
    return path


@pytest.fixture
def model_tree(tmp_path, monkeypatch):
    pins = {}
    roots = {kind: [str(tmp_path / kind), str(tmp_path / 'shared' / kind)]
             for kind in ('diffusion_models', 'text_encoders', 'vae')}
    monkeypatch.setattr(models.comfy_model_paths, 'search_roots', lambda kind: roots[kind])
    monkeypatch.setattr(models.cfg, 'get', lambda key, default=None: pins.get(key, default))
    return tmp_path, pins


def test_qwen21_does_not_offer_old_qwen_or_edit(model_tree):
    root, _ = model_tree
    for name in ('qwen_image.safetensors', 'qwen_image_edit_2511.safetensors',
                 'qwen_image_2.1_int8_convrot.safetensors', 'flux2_klein.safetensors'):
        weights(root / 'diffusion_models' / 'qwen' / name)
    assert [m['filename'].replace('\\', '/') for m in models.list_family_models('qwenimage21')] == [
        'qwen/qwen_image_2.1_int8_convrot.safetensors']


def test_flux_does_not_offer_flux2_klein_or_krea(model_tree):
    root, _ = model_tree
    for name in ('flux1-dev.safetensors', 'flux2-dev.safetensors',
                 'flux2-klein.safetensors', 'flux-krea.safetensors'):
        weights(root / 'diffusion_models' / name)
    assert [m['filename'] for m in models.list_family_models('flux')] == ['flux1-dev.safetensors']


def test_moss_reuses_shared_files_and_reports_its_loader_preparation(model_tree, monkeypatch):
    root, _ = model_tree
    for spec in models.ASSET_SPECS['moss'].values():
        weights(root / 'shared' / spec['kind'] / spec['filename'])
    for filename in ('moss_image_base_bf16.safetensors', 'moss_image_sft_int8_convrot.safetensors',
                     'moss_image_fp8.safetensors', 'other.safetensors'):
        weights(root / 'diffusion_models' / filename)
    offered = [item['filename'] for item in models.list_family_models('moss')]
    assert offered == ['moss_image_sft_bf16.safetensors', 'moss_image_base_bf16.safetensors']
    from app.utils import comfyui
    required = set(models.REQUIRED_NODES['moss'])
    monkeypatch.setattr(comfyui, 'fetch_object_info_classes', lambda: required - {'LDSMossUNETLoader'})
    status = models.generation_readiness('moss')
    assert status['models_ready'] and status['downloads'] == []
    assert status['node_packs'][0]['install_action'] == 'moss_nodes'
    from app import setup_installer
    assert setup_installer.known_action('moss_nodes')
    assert setup_installer._krea_asset_already_installed('studio_moss_text_encoder')


def test_family_folder_does_not_reclassify_an_unrelated_model(model_tree):
    root, _ = model_tree
    weights(root / 'diffusion_models' / 'anima' / 'z_image_turbo.safetensors')
    assert models.list_family_models('anima') == []


def test_encoder_variants_in_deep_shared_folders_are_reused(model_tree):
    root, _ = model_tree
    path = weights(root / 'shared' / 'text_encoders' / 'my-encoders' / 'qwen3vl_8b_bf16.safetensors')
    result = models.resolve_family_assets('qwenimage21')
    assert result['text_encoder'].replace('\\', '/') == 'my-encoders/' + path.name
    assert 'studio_qwenimage21_text_encoder' not in models.missing_assets('qwenimage21')


def test_explicit_pin_is_prioritized_and_missing_pin_preserved(model_tree):
    root, pins = model_tree
    weights(root / 'diffusion_models' / 'flux1-dev.safetensors')
    weights(root / 'diffusion_models' / 'renamed.safetensors')
    pins['studio_models.flux.diffusion_model'] = 'renamed.safetensors'
    assert models.list_family_models('flux')[0]['filename'] == 'renamed.safetensors'
    assert models.resolve_family_assets('flux')['diffusion_model'] == 'renamed.safetensors'
    pins['studio_models.flux.diffusion_model'] = 'missing.safetensors'
    assert models.resolve_family_assets('flux')['diffusion_model'] == 'missing.safetensors'
    assert models.generation_readiness('flux', check_nodes=False)['missing_assets'][0]['name'] == 'missing.safetensors'
    status = models.generation_readiness('flux', check_nodes=False)
    assert 'studio_flux_diffusion_model' not in status['install_actions']
    assert status['pin_warnings'][0]['slot'] == 'diffusion_model'


@pytest.mark.parametrize('pin', ('../outside.safetensors', '/tmp/model.safetensors',
                                 'C:\\models\\x.safetensors', 'bad\nname.safetensors'))
def test_unsafe_pins_are_actionable_without_exposing_paths(model_tree, pin):
    _, pins = model_tree
    pins['studio_models.anima.vae'] = pin
    with pytest.raises(ValueError, match='inside ComfyUI model folders'):
        models.resolve_family_assets('anima')
    status = models.generation_readiness('anima')
    assert not status['ready'] and status['config_error']
    assert pin not in status['config_error']


def test_corrupt_existing_file_is_not_treated_as_installed(model_tree):
    root, _ = model_tree
    spec = models.ASSET_SPECS['anima']['text_encoder']
    path = root / spec['kind'] / spec['filename']
    path.parent.mkdir(parents=True)
    path.write_text('<!doctype html><title>Access denied</title>')
    assert models.invalid_assets('anima')[0]['asset'] == 'studio_anima_text_encoder'
    assert not models.generation_readiness('anima', check_nodes=False)['models_ready']


@pytest.mark.parametrize('family', ('flux', 'anima', 'qwenimage21', 'moss'))
def test_readiness_requires_assets_and_actual_native_nodes(model_tree, monkeypatch, family):
    root, _ = model_tree
    for spec in models.ASSET_SPECS[family].values():
        weights(root / spec['kind'] / spec['filename'])
    from app.utils import comfyui
    monkeypatch.setattr(comfyui, 'fetch_object_info_classes', lambda: None)
    status = models.generation_readiness(family)
    assert status['models_ready'] and not status['ready'] and not status['nodes_checked']
    required = set(models.REQUIRED_NODES[family])
    monkeypatch.setattr(comfyui, 'fetch_object_info_classes', lambda: required)
    assert models.generation_readiness(family)['ready']
    required.remove('KSampler')
    assert models.generation_readiness(family)['missing_nodes'] == ['KSampler']


def test_install_catalog_registered_and_reuses_compatible_existing_file(model_tree):
    root, _ = model_tree
    from app import setup_installer
    downloads = models.model_downloads()
    assert downloads.keys() <= set(setup_installer.INSTALL_ACTIONS)
    spec = models.ASSET_SPECS['qwenimage21']['text_encoder']
    weights(root / 'shared' / spec['kind'] / 'existing' / spec['alternatives'][-1])
    assert setup_installer._krea_asset_already_installed('studio_qwenimage21_text_encoder')
    assert downloads['studio_anima_diffusion_model']['min_free_gb'] < 10


def test_settings_catalog_preserves_controls_when_one_pin_is_invalid(model_tree, monkeypatch):
    _, pins = model_tree
    from app.utils import comfyui
    monkeypatch.setattr(comfyui, 'fetch_object_info_classes', lambda: None)
    pins['studio_models.anima.vae'] = '../bad.safetensors'
    catalog = models.settings_catalog()
    anima = next(f for f in catalog if f['family'] == 'anima')
    assert anima['config_error'] and len(anima['slots']) == 3
    assert [f['family'] for f in catalog] == ['zimage', 'moss', 'flux', 'anima', 'qwenimage21']
    # Z-Image's files are detected, never chosen in this card.
    assert [s['configurable'] for s in catalog[0]['slots']] == [False, False, False]


def test_download_details_offer_only_actions_that_repair_the_resolved_family(model_tree):
    root, pins = model_tree
    status = models.generation_readiness('qwenimage21', check_nodes=False)
    assert {item['action'] for item in status['downloads']} == set(status['install_actions'])
    assert len(status['downloads']) == 3
    assert status['downloads'][0] == {
        'action': 'studio_qwenimage21_diffusion_model', 'slot': 'diffusion_model',
        'label': 'Qwen-Image 2.1 diffusion model',
        'filename': 'qwen_image_2.1_int8_convrot.safetensors',
        'size_bytes': 7256783064, 'source_url': 'https://huggingface.co/Comfy-Org/Qwen-Image-2.1',
        'repair': False,
    }
    weights(root / 'text_encoders' / 'qwen3vl_8b_int8_convrot.safetensors')
    corrupt = root / 'vae' / 'qwen_image_2.1_vae_bf16.safetensors'
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text('<!doctype html><title>Access denied</title>')
    pins['studio_models.qwenimage21.diffusion_model'] = 'custom/missing.safetensors'
    status = models.generation_readiness('qwenimage21', check_nodes=False)
    # A download fixes the corrupt automatic VAE, not a custom pin or a valid encoder.
    assert status['install_actions'] == ['studio_qwenimage21_vae']
    assert [(item['slot'], item['repair']) for item in status['downloads']] == [('vae', True)]
    pins['studio_models.qwenimage21.diffusion_model'] = '../unsafe.safetensors'
    status = models.generation_readiness('qwenimage21', check_nodes=False)
    assert status['config_error'] and status['downloads'] == [] and status['install_actions'] == []


# --- Z-Image: LDS provides the Studio's three default files -------------------
_ZIMAGE_FILES = {
    # action: (repository file, published size, published SHA256, destination)
    'studio_zimage_diffusion_model': (
        'split_files/diffusion_models/z_image_turbo_bf16.safetensors', 12309866400,
        '2407613050b809ffdff18a4ac99af83ea6b95443ecebdf80e064a79c825574a6',
        ('diffusion_models', 'z image', 'z_image_turbo_bf16.safetensors')),
    'studio_zimage_text_encoder': (
        'split_files/text_encoders/qwen_3_4b.safetensors', 8044982048,
        '6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a',
        ('text_encoders', 'qwen_3_4b.safetensors')),
    'studio_zimage_vae': (
        'split_files/vae/ae.safetensors', 335304388,
        'afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38',
        ('vae', 'z image', 'ae.safetensors')),
}


@pytest.fixture
def zimage_comfy(app, tmp_path, monkeypatch):
    """A fake ComfyUI install the real resolvers, listers and installer all read."""
    from app import config
    from app.utils import comfyui
    base = tmp_path / 'ComfyUI'
    (base / 'models').mkdir(parents=True)
    (base / 'main.py').write_text('# fake ComfyUI entrypoint', encoding='utf-8')
    monkeypatch.setattr(comfyui, 'fetch_object_info_classes', lambda *a, **k: None)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        models.comfy_model_paths.clear_cache()
        comfyui.clear_model_caches()
        yield base
        comfyui.clear_model_caches()


def test_zimage_downloads_are_the_pinned_official_files(monkeypatch):
    from app import setup_installer
    revision = '6fc90a3b1b653e935a0d175e260736de25b84df5'
    monkeypatch.setenv('HF_TOKEN', 'hf_test_token')
    for action, (repo_file, size, sha256, dest) in _ZIMAGE_FILES.items():
        spec = setup_installer.model_download_spec(action)
        assert action in setup_installer.INSTALL_ACTIONS
        assert spec['url'] == f'https://huggingface.co/Comfy-Org/Z-Image-Turbo/resolve/{revision}/{repo_file}'
        assert (spec['expected_bytes'], spec['sha256'], spec['dest']) == (size, sha256, dest)
        assert spec['min_free_gb'] * 1024 ** 3 > size and not spec['gated']
        # Hugging Face's own token, and only towards Hugging Face.
        headers, provider = setup_installer._download_auth(spec)
        assert provider == 'hf' and headers == {'Authorization': 'Bearer hf_test_token'}
    # Not part of the unattended "Install everything" plan.
    assert not set(_ZIMAGE_FILES) & set(setup_installer._INSTALL_ALL_ORDER)


def test_no_two_downloads_write_one_file_with_different_expectations():
    from app import setup_installer
    writers = {}
    for action, spec in setup_installer._MODEL_DOWNLOADS.items():
        writers.setdefault('/'.join(spec['dest']).casefold(), []).append(
            (action, spec.get('expected_bytes'), spec.get('sha256')))
    for dest, entries in writers.items():
        assert len({entry[1:] for entry in entries}) == 1, (dest, entries)
    # Each Z-Image file has a writer of its own: FLUX.1 keeps vae/ae.safetensors.
    for action, (*_, dest) in _ZIMAGE_FILES.items():
        assert [entry[0] for entry in writers['/'.join(dest).casefold()]] == [action]


def test_missing_zimage_files_are_offered_with_their_sizes(zimage_comfy):
    status = models.generation_readiness('zimage', check_nodes=False)
    assert not status['models_ready'] and not status['pin_warnings']
    assert status['install_actions'] == list(_ZIMAGE_FILES)
    assert [(d['action'], d['size_bytes'], d['repair']) for d in status['downloads']] == [
        (action, size, False) for action, (_, size, _, _) in _ZIMAGE_FILES.items()]
    assert {d['source_url'] for d in status['downloads']} == {'https://huggingface.co/Comfy-Org/Z-Image-Turbo'}
    assert models.zimage_studio_readiness() == status


def test_existing_ae_vae_is_reused_in_both_directions(zimage_comfy):
    from app import setup_installer
    vae = weights(zimage_comfy / 'models' / 'vae' / 'ae.safetensors')   # FLUX.1's VAE
    status = models.generation_readiness('zimage', check_nodes=False)
    assert 'studio_zimage_vae' not in status['install_actions']
    assert status['assets']['vae'] == 'ae.safetensors'
    assert setup_installer._krea_asset_already_installed('studio_zimage_vae')
    # The Z-Image download is FLUX.1's VAE too: same bytes, found by its basename.
    vae.unlink()
    weights(zimage_comfy / 'models' / 'vae' / 'z image' / 'ae.safetensors')
    assert 'studio_flux_vae' not in models.missing_assets('flux')


def test_a_broken_official_download_is_repaired_but_a_pin_is_not_replaced(zimage_comfy):
    from app import config, setup_installer
    from app.utils.comfyui import clear_model_caches
    weights(zimage_comfy / 'models' / 'diffusion_models' / 'z image' / 'z_image_turbo_bf16.safetensors')
    weights(zimage_comfy / 'models' / 'text_encoders' / 'qwen_3_4b.safetensors')
    broken = zimage_comfy / 'models' / 'vae' / 'z image' / 'ae.safetensors'
    broken.parent.mkdir(parents=True)
    broken.write_text('<!doctype html><title>Access denied</title>')
    clear_model_caches()
    status = models.generation_readiness('zimage', check_nodes=False)
    assert status['install_actions'] == ['studio_zimage_vae']
    assert [(d['slot'], d['repair']) for d in status['downloads']] == [('vae', True)]
    assert not setup_installer._krea_asset_already_installed('studio_zimage_vae')
    # A pinned file is the user's: a warning names the setting, no download replaces it,
    # and the launch preflight keeps naming that file.
    config.save_config({'zimage': {'vae': 'mine/missing_vae.safetensors'}})
    status = models.generation_readiness('zimage', check_nodes=False)
    assert status['install_actions'] == [] and status['models_ready'] and not status['ready']
    assert [w['slot'] for w in status['pin_warnings']] == ['vae']
    assert 'zimage.vae' in status['pin_warnings'][0]['message']
    assert 'mine/missing_vae' not in status['pin_warnings'][0]['message']


def test_zimage_readiness_checks_the_graph_nodes(zimage_comfy, monkeypatch):
    from app.utils import comfyui
    for *_, dest in _ZIMAGE_FILES.values():
        weights(zimage_comfy.joinpath('models', *dest))
    comfyui.clear_model_caches()
    required = models._zimage_required_nodes()
    assert {'UNETLoader', 'CLIPLoader', 'VAELoader', 'SamplerCustomAdvanced',
            'LoraLoaderModelOnly'} <= required
    monkeypatch.setattr(comfyui, 'fetch_object_info_classes', lambda *a, **k: set(required))
    assert models.generation_readiness('zimage')['ready']
    monkeypatch.setattr(comfyui, 'fetch_object_info_classes',
                        lambda *a, **k: set(required) - {'SamplerCustomAdvanced'})
    assert models.generation_readiness('zimage')['missing_nodes'] == ['SamplerCustomAdvanced']


def test_zimage_studio_payloads_carry_the_downloads_without_a_default_model(zimage_comfy, client):
    body = client.get('/api/studio/base-models?type=zimage').get_json()
    assert body['models'] == [] and 'default_model' not in body
    assert [d['action'] for d in body['generation_readiness']['downloads']] == list(_ZIMAGE_FILES)
    families = client.get('/api/comfy/trained-image-models').get_json()['families']
    assert families[0]['family'] == 'zimage' and families[0]['install_actions'] == list(_ZIMAGE_FILES)


def test_a_failing_zimage_check_never_takes_the_studio_down(zimage_comfy, client, monkeypatch):
    def broken():
        raise OSError('disk went away')
    monkeypatch.setattr(models, '_zimage_status', broken)
    response = client.get('/api/studio/base-models?type=zimage')
    assert response.status_code == 200 and response.get_json()['generation_readiness'] is None
    families = client.get('/api/comfy/trained-image-models').get_json()['families']
    assert [f['family'] for f in families] == list(models.ASSET_SPECS)
