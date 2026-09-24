"""Import, queue and recover standalone views through the registered HTTP API."""
import io
from pathlib import Path
import sys

import pytest
from flask import Flask
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'backend'), str(ROOT / 'bundled/camera_angles')]
from app.plugins.api import PluginContext
from app.plugins.manifest import load_manifest
from app.plugins.registry import PluginRecord, PluginRegistry
from lds_camera_angles import register, qwen_camera_helper as qch, routes, studio


@pytest.fixture
def product(tmp_path, monkeypatch):
    app = Flask(__name__)
    app.config.update(TESTING=True, MAX_CONTENT_LENGTH=34 * 1024 * 1024)
    directory = ROOT / 'bundled/camera_angles'
    manifest = load_manifest(directory)
    registry = PluginRegistry()
    registry.records[manifest.id] = PluginRecord(manifest.id, manifest, str(directory), True, state='loaded')
    assert registry.claim_manifest(manifest) is None
    app.extensions['lds_plugins'] = registry
    ctx = PluginContext(app, registry, manifest, tmp_path / 'camera-data')
    register(ctx)
    monkeypatch.setattr(routes, '_require_no_stalled_comfyui', lambda: None)
    monkeypatch.setattr(studio.cfg, 'local_user', lambda: 'test-user')
    monkeypatch.setattr(studio.cfg, 'get', lambda *_: '')
    output, input_dir = tmp_path / 'output', tmp_path / 'input'
    output.mkdir()
    input_dir.mkdir()
    monkeypatch.setattr(studio.cfg, 'comfyui_dir', lambda kind: output if kind == 'output' else input_dir)
    monkeypatch.setattr(qch, 'camera_missing_assets', lambda: [])
    monkeypatch.setattr(qch, 'resolve_camera_unet', lambda: 'qwen.safetensors')
    monkeypatch.setattr(qch, 'resolve_camera_text_encoder', lambda: 'text.safetensors')
    monkeypatch.setattr(qch, 'resolve_camera_vae', lambda: 'vae.safetensors')
    monkeypatch.setattr(qch, 'resolve_camera_lora', lambda: ('angles.safetensors', 'angles'))
    monkeypatch.setattr(qch, 'resolve_camera_speed_lora', lambda: (None, None))
    monkeypatch.setattr(qch.comfy_fs, 'ensure_input_usable', str)
    monkeypatch.setattr(qch.comfy_fs, 'stage_input_image', lambda _path, name, _folder: name)
    jobs = {}
    def enqueue(plugin, kind, **kwargs):
        assert plugin == 'camera_angles' and kind == 'is_camera_studio'
        assert kwargs['workflow_data']['112']['inputs']['prompt'].startswith('<sks>')
        jobs[kwargs['job_id']] = {**kwargs, 'status': 'pending', 'result_filename': None, 'error': None}
        return kwargs['job_id']
    monkeypatch.setattr(qch.comfy_fs, 'add_plugin_job', enqueue)
    monkeypatch.setattr(studio.comfy, 'plugin_job', lambda _plugin, ident, _user: jobs.get(ident))
    return app, app.test_client(), jobs, output


def upload(client, color='blue'):
    source = io.BytesIO()
    Image.new('RGB', (24, 16), color).save(source, 'PNG')
    source.seek(0)
    response = client.post('/api/camera/studio/images', data={'image': (source, 'source.png')})
    assert response.status_code == 201, response.get_json()
    return response.get_json()['image']


def test_import_queue_complete_download_and_recovery(product):
    app, client, jobs, output = product
    image = upload(client)
    url = '/api/camera/studio/images/' + image['id']
    original = client.get(url + '/original').data
    response = client.post(url + '/shoot', json={'poses': ['right/eye/medium', 'back/eye/medium']})
    assert response.status_code == 200, response.get_json()
    assert response.json['queued'] == 2
    assert client.get(url).json['image']['views'][0]['status'] == 'queued'
    assert client.post(url + '/shoot', json={'poses': ['left/eye/medium']}).status_code == 400
    job_ids = list(jobs)
    jobs[job_ids[0]]['status'] = 'sent_to_comfy'
    assert client.get(url).json['image']['views'][0]['status'] == 'running'
    Image.new('RGB', (16, 24), 'red').save(output / 'result.png')
    callback = app.extensions['lds_plugins'].job_handlers['is_camera_studio'][1]
    with app.app_context():
        callback(job_ids[0], 'result.png', metadata=jobs[job_ids[0]]['metadata'])
    # The other completion is recovered from the durable queue after a missed callback.
    jobs[job_ids[1]].update(status='completed', result_filename='result.png')
    refreshed = client.get(url).json['image']
    assert [v['status'] for v in refreshed['views']] == ['done', 'done']
    assert client.get(url + '/original').data == original
    result = client.get(url + '/views/' + job_ids[1] + '?download=1')
    assert result.status_code == 200 and 'attachment' in result.headers['Content-Disposition']
    assert Image.open(io.BytesIO(result.data)).size == (16, 24)
    assert len(client.get('/api/camera/studio/images').json['images'][0]['views']) == 2
    assert set(app.extensions['lds_plugins'].records) == {'camera_angles'}


def test_failed_cancelled_and_interrupted_jobs_do_not_stay_busy(product):
    _app, client, jobs, _output = product
    image = upload(client)
    url = '/api/camera/studio/images/' + image['id']
    client.post(url + '/shoot', json={'poses': ['right/eye/medium', 'back/eye/medium', 'left/eye/medium']})
    ids = list(jobs)
    jobs[ids[0]].update(status='failed', error='Model could not load')
    jobs[ids[1]]['status'] = 'cancelled'
    del jobs[ids[2]]
    result = client.get(url).json['image']['views']
    assert all(view['status'] == 'failed' for view in result)
    assert result[0]['error'] == 'Model could not load'
    assert client.post(url + '/shoot', json={'poses': ['right/eye/medium']}).status_code == 200


def test_invalid_import_poses_and_missing_models_create_no_jobs(product, monkeypatch):
    _app, client, jobs, _output = product
    assert client.post('/api/camera/studio/images', data={'image': (io.BytesIO(b'bad'), 'image.png')}).status_code == 400
    assert client.get('/api/camera/studio/images').json == {'images': []}
    image = upload(client)
    url = '/api/camera/studio/images/' + image['id']
    assert client.post(url + '/shoot', json={'poses': ['unknown']}).status_code == 400
    assert client.post(url + '/shoot', json=['right/eye/medium']).status_code == 400
    monkeypatch.setattr(qch, 'camera_missing_assets', lambda: ['camera_model'])
    # Installation is stubbed: exercising the missing-model branch must not download.
    monkeypatch.setattr(routes, '_camera_missing_response', lambda _exc: ({'error': 'Install Camera model'}, 409))
    assert client.post(url + '/shoot', json={'poses': ['right/eye/medium']}).status_code == 409
    assert not jobs
    assert not client.get(url).json['image']['views']


def test_sources_are_confined_to_the_owner_and_results_cannot_escape(product, monkeypatch):
    app, client, jobs, output = product
    image = upload(client)
    url = '/api/camera/studio/images/' + image['id']
    client.post(url + '/shoot', json={'poses': ['right/eye/medium']})
    ident = next(iter(jobs))
    Image.new('RGB', (1, 1)).save(output.parent / 'outside.png')
    jobs[ident].update(status='completed', result_filename='../outside.png')
    assert client.get(url).json['image']['views'][0]['status'] == 'failed'
    assert client.get(url + '/views/' + ident).status_code == 404
    monkeypatch.setattr(studio.cfg, 'local_user', lambda: 'another-user')
    assert client.get(url).status_code == 404
    assert client.get(url + '/original').status_code == 404
    with app.app_context(), pytest.raises(LookupError):
        studio.folder('another-user', '../outside')


def test_partial_admission_keeps_queued_views(product, monkeypatch):
    _app, client, jobs, _output = product
    image = upload(client)
    real = qch.comfy_fs.add_plugin_job
    def fail_second(*args, **kwargs):
        if jobs:
            raise ValueError('Queue unavailable')
        return real(*args, **kwargs)
    monkeypatch.setattr(qch.comfy_fs, 'add_plugin_job', fail_second)
    result = client.post('/api/camera/studio/images/' + image['id'] + '/shoot',
                         json={'poses': ['right/eye/medium', 'back/eye/medium']}).json
    assert result['queued'] == 1 and result['requested'] == 2
    assert [v['status'] for v in result['image']['views']] == ['queued', 'failed']


def test_completion_without_an_output_is_reported_and_can_be_retried(product):
    _app, client, jobs, _output = product
    image = upload(client)
    url = '/api/camera/studio/images/' + image['id']
    client.post(url + '/shoot', json={'poses': ['right/eye/medium']})
    next(iter(jobs.values()))['status'] = 'completed'
    view = client.get(url).json['image']['views'][0]
    assert view['status'] == 'failed' and 'without an output' in view['error']
