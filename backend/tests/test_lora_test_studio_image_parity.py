"""Training families keep their own graph, defaults and base through Studio replay."""
import json

import pytest

from app.services import lora_test_studio as studio
from test_studio_prompt_batch import _neutralise_preflights


def test_every_trainable_image_family_has_a_generation_lane():
    from app.services.face_dataset_service import TRAIN_TYPES
    assert set(studio.GENERATION_FAMILIES) == set(TRAIN_TYPES)
    assert all(studio.can_generate_with(family) for family in TRAIN_TYPES)


def _install_family(monkeypatch, family):
    from app.services import trained_image_models as models
    base = f'{family}/base.safetensors'
    assets = {'diffusion_model': base, 'text_encoder': f'{family}/encoder.safetensors',
              'text_encoder_2': 'clip_l.safetensors', 'vae': f'{family}/vae.safetensors'}
    monkeypatch.setattr(models, 'list_family_models',
                        lambda _family: [{'filename': base, 'displayName': 'Base'}])
    monkeypatch.setattr(models, 'resolve_family_assets', lambda *a, **k: dict(assets))
    monkeypatch.setattr(models, 'generation_readiness',
                        lambda *a, **k: {'ready': True, 'assets': assets})
    return base


@pytest.mark.parametrize('family', studio.TRAINED_IMAGE_FAMILIES)
@pytest.mark.parametrize('mode', ['single', 'compare', 'blend'])
def test_launch_and_resume_preserve_family_graph_defaults_and_loras(app, monkeypatch, family, mode):
    from app.config import LOCAL_USER
    from app.extensions import db
    from app.models import LoraTestImage
    from app.services import face_dataset_service as fds
    with app.app_context():
        base = _install_family(monkeypatch, family)
        _neutralise_preflights(monkeypatch, studio)
        monkeypatch.setattr(studio, 'preflight_family', lambda *_a, **_kw: None)
        first = fds.create_dataset(LOCAL_USER, 'First subject', 'subjecta', train_type=family)
        second = fds.create_dataset(LOCAL_USER, 'Second subject', 'subjectb', train_type=family)
        a, b = f'{family}/lora_subjecta.safetensors', f'{family}/lora_subjectb.safetensors'
        monkeypatch.setattr(studio, 'list_test_checkpoints',
                            lambda ds, _family=None: [{'filename': a if ds.id == first.id else b}])
        captured = []
        monkeypatch.setattr(studio.queue_manager, 'add_job',
                            lambda **kwargs: captured.append(kwargs))
        settings = studio.StudioGenSettings(prompt='portrait', seed=73, z_model=base,
                                            negative='blurred', aspects=['1:1'])
        if mode == 'single':
            run = studio.create_run(LOCAL_USER, first.id, [a], [0.6], settings)
        else:
            run = studio.create_comparison_run(LOCAL_USER, [
                {'dataset_id': first.id, 'checkpoint': a, 'weight': 0.6},
                {'dataset_id': second.id, 'checkpoint': b, 'weight': 0.8}],
                [0.6], settings, combine=mode == 'blend')
        expected_count = 2 if mode == 'compare' else 1
        assert run['created'] == expected_count
        rows = LoraTestImage.query.filter_by(run_id=run['run_id']).all()
        defaults = studio.studio_family_defaults(family)
        for row, job in zip(rows, captured):
            assert (row.z_model, row.cfg, row.steps) == (base, defaults['cfg'], defaults['steps'])
            assert row.negative == ('blurred' if family != 'flux' else None)
            assert job['metadata']['family'] == family
            assert job['metadata']['base_model'] == base
            workflow = job['workflow_data']
            loaders = [n['inputs'] for n in workflow.values()
                       if n['class_type'] == 'LoraLoaderModelOnly']
            assert len(loaders) == (2 if mode == 'blend' else 1)
            assert loaders[0]['lora_name'] == row.checkpoint
            assert loaders[0]['strength_model'] == 0.6
            assert [n['inputs']['unet_name'] for n in workflow.values()
                    if n['class_type'] == 'UNETLoader'] == [base]
            assert [n['inputs']['steps'] for n in workflow.values()
                    if 'steps' in n.get('inputs', {})] == [defaults['steps']]
            row.status = 'cancelled'
        db.session.commit()
        before = list(captured)
        assert studio.resume_run(LOCAL_USER, run_id=run['run_id']) == {'resumed': expected_count}
        for original, resumed in zip(before, captured[expected_count:]):
            def stable_graph(job):
                graph = json.loads(json.dumps(job['workflow_data']))
                for node in graph.values():
                    node.get('inputs', {}).pop('filename_prefix', None)
                return graph
            assert stable_graph(original) == stable_graph(resumed)
        # Removing the saved model cannot turn replay into a different experiment.
        for row in rows:
            row.status = 'cancelled'
        db.session.commit()
        monkeypatch.setattr(studio, 'family_base_models', lambda _family: ['other.safetensors'])
        assert studio.resume_run(LOCAL_USER, run_id=run['run_id']) == {'resumed': 0}
        assert len(captured) == expected_count * 2
        assert all(row.status == 'failed' and 'unavailable' in row.error for row in rows)


@pytest.mark.parametrize('family', studio.TRAINED_IMAGE_FAMILIES)
def test_dataset_and_comparison_publish_same_family_bases_and_defaults(app, client, monkeypatch, family):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as fds
    with app.app_context():
        base = _install_family(monkeypatch, family)
        ds = fds.create_dataset(LOCAL_USER, 'Subject', 'subject', train_type=family)
        monkeypatch.setattr(studio, 'list_test_checkpoints',
                            lambda *_a, **_kw: [{'filename': f'{family}/lora_subject.safetensors'}])
        monkeypatch.setattr(studio, 'permanent_lora_candidates', lambda *_a: [])
        payload = studio.studio_payload(LOCAL_USER, ds.id, family=family)
        reply = client.get(f'/api/studio/base-models?type={family}')
        assert reply.status_code == 200
        body = reply.get_json()
        assert [m['value'] for m in payload['z_models']] == [base]
        assert [m['filename'] for m in body['models']] == [base]
        for key in ('default_steps', 'default_cfg'):
            assert payload[key] == body['axes'][key]
        assert payload['model_defaults'] == body['model_defaults']
        assert payload['generation_capabilities'] == body['generation_capabilities']
        assert payload['default_model'] == body['default_model'] == base


def test_invalid_config_is_explained_without_selecting_another_base(app, client, monkeypatch):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as fds, trained_image_models as models
    with app.app_context():
        _install_family(monkeypatch, 'anima')
        monkeypatch.setattr(models, 'generation_readiness', lambda *_a, **_kw: {
            'assets': {}, 'ready': False, 'config_error': 'Choose a relative model filename'})
        monkeypatch.setattr(studio, 'list_test_checkpoints',
                            lambda *_a, **_kw: [{'filename': 'anima/lora_subject.safetensors'}])
        monkeypatch.setattr(studio, 'permanent_lora_candidates', lambda *_a: [])
        ds = fds.create_dataset(LOCAL_USER, 'Subject', 'subject', train_type='anima')
        payload = studio.studio_payload(LOCAL_USER, ds.id, family='anima')
        response = client.get('/api/studio/base-models?type=anima')
        assert response.status_code == 200
        for body in (payload, response.get_json()):
            assert body['default_model'] == ''
            assert body['base_note'] == 'Choose a relative model filename'


def test_unknown_family_and_explicit_wrong_base_do_not_fall_back(client):
    assert client.get('/api/studio/base-models?type=not-a-family').status_code == 400
    with pytest.raises(ValueError, match='unavailable'):
        studio._select_base_models(['anima/base.safetensors'], 'flux/base.safetensors')
    with pytest.raises(ValueError, match='not-a-family'):
        studio._build_cell_workflow('u', 'x.safetensors', 1, '', 1, None,
                                    {'x.safetensors'}, train_type='not-a-family')


@pytest.mark.parametrize('family', studio.TRAINED_IMAGE_FAMILIES)
def test_preflight_builder_failure_is_not_swallowed(monkeypatch, family):
    def fail(*_a, **_kw):
        raise ValueError('required model missing')
    monkeypatch.setattr(studio, '_build_cell_workflow', fail)
    with pytest.raises(ValueError, match='required model missing'):
        studio._preflight_run('u', family, 'lora.safetensors', ['base.safetensors'],
                              {'lora.safetensors'}, 'portrait', 1, 1, 'subject')


def test_missing_explicit_default_pin_does_not_elect_another_base(monkeypatch):
    from app.services import trained_image_models as models
    monkeypatch.setattr(models, 'resolve_family_assets',
                        lambda *_a, **_kw: {'diffusion_model': 'missing.safetensors'})
    with pytest.raises(ValueError, match='unavailable'):
        studio._select_base_models(['other.safetensors'], family='anima')
    assert studio._select_base_models(['other.safetensors'], 'other.safetensors',
                                     family='anima') == ['other.safetensors']


def test_klein_stacks_extra_loras_and_persists_the_cfg_it_actually_uses(app, monkeypatch):
    from app.services import klein_edit_helper as klein
    from app.utils import comfyui
    monkeypatch.setattr(comfyui, 'get_flux2_klein_models',
                        lambda: [{'filename': 'klein/base.safetensors'}])
    monkeypatch.setattr(klein, 'resolve_klein_text_encoder', lambda: 'klein/encoder.safetensors')
    monkeypatch.setattr(klein, 'resolve_klein_vae', lambda: 'klein/vae.safetensors')
    head, extra = 'flux2klein/head.safetensors', 'flux2klein/extra.safetensors'
    cell = studio.build_matrix([head], [0.7], cfgs=[5], family='flux2klein')[0]
    assert (cell[3], cell[4]) == (1.0, 4)
    with app.app_context():
        graph = studio._build_cell_workflow('u', head, 0.7, 'portrait', 3,
                                          'klein/base.safetensors', {head},
                                          train_type='flux2klein', cfg=cell[3], steps=cell[4],
                                          extra_loras=[{'filename': extra, 'strength': 0.4}])
    tail = graph[graph['31']['inputs']['model'][0]]
    assert tail['class_type'] == 'LoraLoaderModelOnly'
    assert tail['inputs'] == {'model': ['29', 0], 'lora_name': extra, 'strength_model': 0.4}


def test_missing_native_node_and_setup_node_have_actionable_409(app):
    from app.routes._common import _studio_missing_response
    with app.app_context():
        response, status = _studio_missing_response(studio.StudioAssetsMissing(
            'qwenimage21', [], ['TextEncodeQwenImage21', 'LDSKrea2PresetSampler']))
        body = response.get_json()
        assert status == 409
        assert 'Qwen-Image 2.1' in body['error'] and 'Update ComfyUI' in body['error']
        assert 'Setup' in body['error']
        assert {'class_type': 'TextEncodeQwenImage21', 'pack': 'ComfyUI', 'core': True} in (
            body['studio_missing']['node_packs'])


def test_a_node_comfyui_cannot_describe_asks_for_a_restart_not_an_update(app, monkeypatch, caplog):
    """A model folder deleted under a running ComfyUI drops UNETLoader from its node
    list until it restarts; sending the user to update ComfyUI fixes nothing."""
    from app.routes._common import _studio_missing_response
    from app.utils import comfyui
    monkeypatch.setattr(comfyui, 'node_description_fails', lambda name: name == 'UNETLoader')
    with app.app_context(), caplog.at_level('WARNING'):
        response, status = _studio_missing_response(studio.StudioAssetsMissing(
            'krea', [], ['UNETLoader', 'LDSKrea2PresetSampler']))
        body = response.get_json()
    assert status == 409
    assert 'ComfyUI has UNETLoader but cannot describe it' in body['error']
    assert 'restart ComfyUI' in body['error'] and 'Update ComfyUI' not in body['error']
    assert '1 required ComfyUI node(s) are missing' in body['error']
    packs = {pack['class_type']: pack for pack in body['studio_missing']['node_packs']}
    assert 'restart ComfyUI' in packs['UNETLoader']['unavailable'] and 'core' not in packs['UNETLoader']
    assert packs['LDSKrea2PresetSampler']['pack'] == 'Krea 2 preset sampler'
    assert "cannot describe ['UNETLoader']" in caplog.text


@pytest.mark.comfyui_http
@pytest.mark.parametrize('answer, verdict', [(500, True), (200, False), (502, None), (None, None)])
def test_only_comfyuis_own_error_says_a_node_cannot_be_described(monkeypatch, answer, verdict):
    from types import SimpleNamespace
    from app.utils import comfyui
    asked = []

    def get(url, **_kw):
        asked.append(url)
        if answer is None:
            raise comfyui.requests.exceptions.ConnectionError('refused')
        return SimpleNamespace(status_code=answer)

    monkeypatch.setattr(comfyui.requests, 'get', get)
    assert comfyui.node_description_fails('UNet loader/x') is verdict
    assert asked == [comfyui.urljoin(comfyui.api_address(), '/object_info/UNet%20loader%2Fx')]


def test_preflight_reads_nodes_from_the_built_qwen_graph(monkeypatch):
    from app.utils import comfyui
    base = _install_family(monkeypatch, 'qwenimage21')
    cp = 'qwenimage21/subject.safetensors'
    workflow = studio._build_cell_workflow('u', cp, 1, 'portrait', 1, base, {cp},
                                          train_type='qwenimage21')
    classes = {node['class_type'] for node in workflow.values()} - {'TextEncodeQwenImage21'}
    monkeypatch.setattr(studio, '_models_root', lambda: None)
    monkeypatch.setattr(comfyui, 'fetch_object_info_classes', lambda: classes)
    with pytest.raises(studio.StudioAssetsMissing) as exc:
        studio.preflight_family('qwenimage21', [workflow])
    assert exc.value.missing_nodes == ['TextEncodeQwenImage21']
