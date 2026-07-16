"""Training blueprint: ai-toolkit gating + kwargs forwarding to the service.

Every test patches `app.capabilities.probe` so none of this ever touches a
real HTTP/subprocess probe, and patches the `lora_training`/`zimage_convert`
service functions it exercises so no test spawns a real subprocess.
"""
import pytest

def _create(client, name='Lola', trigger='lola'):
    return client.post('/api/dataset/create', json={'name': name, 'trigger_word': trigger}).get_json()['id']


def _valid(monkeypatch, ok=True):
    monkeypatch.setattr('app.capabilities.probe', lambda *a, **k: {'aitoolkit': {'valid': ok}})


# --- Gating -------------------------------------------------------------------

def test_train_unconfigured_returns_409_with_hint(client, monkeypatch):
    _valid(monkeypatch, False)
    resp = client.post('/api/dataset/1/train', json={})
    assert resp.status_code == 409
    body = resp.get_json()
    assert body['error'] == 'ai-toolkit is not configured'
    assert body['hint'] == 'Set its folder in Settings'


def test_status_available_false_when_unconfigured(client, monkeypatch):
    _valid(monkeypatch, False)
    resp = client.get('/api/dataset/train/status')
    assert resp.status_code == 200
    assert resp.get_json() == {'available': False}


def test_status_configured_polls_queue_then_status(client, monkeypatch):
    _valid(monkeypatch, True)
    calls = []
    monkeypatch.setattr('app.services.lora_training.process_training_queue', lambda: calls.append('polled'))
    monkeypatch.setattr('app.services.lora_training.training_status',
                        lambda user_id=None: {'in_progress': False, 'user': user_id})
    resp = client.get('/api/dataset/train/status')
    assert resp.status_code == 200
    assert calls == ['polled']
    assert resp.get_json() == {'in_progress': False, 'user': 'local'}


def test_stop_gated_when_unconfigured(client, monkeypatch):
    _valid(monkeypatch, False)
    resp = client.post('/api/dataset/train/stop')
    assert resp.status_code == 409


def test_train_unknown_dataset_404(client, monkeypatch):
    _valid(monkeypatch, True)
    resp = client.post('/api/dataset/999999/train', json={})
    assert resp.status_code == 404


# --- /train ---------------------------------------------------------------

def test_train_configured_forwards_kwargs(client, monkeypatch):
    _valid(monkeypatch, True)
    ds_id = _create(client)
    captured = {}

    def fake_launch(user_id, dataset_id, **kw):
        captured['user_id'] = user_id
        captured['dataset_id'] = dataset_id
        captured.update(kw)
        return {'started': True, 'pid': 123, 'config_path': 'x', 'steps': 1234,
                'dataset_folder': 'y', 'log_path': 'z'}

    monkeypatch.setattr('app.services.lora_training.launch_training', fake_launch)
    resp = client.post(f'/api/dataset/{ds_id}/train', json={
        'steps': 1234, 'masked': False, 'train_type': 'sdxl',
        'allow_caption_mismatch': True, 'allow_caption_quality': True})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] is True and body['pid'] == 123
    assert captured == {
        'user_id': 'local',
        'dataset_id': ds_id,
        'steps': 1234,
        'base_model': None,
        'variant': 'turbo',
        'train_type': 'sdxl',
        'allow_caption_mismatch': True,
        'allow_uncaptioned': False,   # absent du body → False (confirm non donné)
        'allow_caption_quality': True,
        'allow_unverified_weights': False,   # custom-weights confirm non donné
        'masked': False,
        'fresh': False,          # absent du body → False (resume historique)
    }
    # fresh=true (choix « Start fresh » du panneau) traverse jusqu'au service.
    client.post(f'/api/dataset/{ds_id}/train', json={'fresh': True})
    assert captured['fresh'] is True


def test_train_value_error_returns_400(client, monkeypatch):
    _valid(monkeypatch, True)
    ds_id = _create(client)
    monkeypatch.setattr('app.services.lora_training.launch_training',
                        lambda *a, **k: (_ for _ in ()).throw(ValueError('bad state')))
    resp = client.post(f'/api/dataset/{ds_id}/train', json={})
    assert resp.status_code == 400
    assert resp.get_json()['error'] == 'bad state'


def test_train_runtime_error_returns_409(client, monkeypatch):
    _valid(monkeypatch, True)
    ds_id = _create(client)
    monkeypatch.setattr('app.services.lora_training.launch_training',
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('not installed')))
    resp = client.post(f'/api/dataset/{ds_id}/train', json={})
    assert resp.status_code == 409


# --- /train/continue --------------------------------------------------------

def test_continue_forwards_kwargs(client, monkeypatch):
    _valid(monkeypatch, True)
    ds_id = _create(client)
    captured = {}

    def fake_continue(user_id, dataset_id, **kw):
        captured.update(kw)
        return {'started': True, 'resumed_from': 500, 'target_steps': 1500}

    monkeypatch.setattr('app.services.lora_training.continue_training', fake_continue)
    resp = client.post(f'/api/dataset/{ds_id}/train/continue', json={
        'extra_steps': 1000, 'base_model': 'merge.safetensors',
        'variant': 'base', 'train_type': 'zimage', 'masked': False,
        'allow_unverified_weights': True,
        'allow_caption_mismatch': True,
        'allow_uncaptioned': True,
        'allow_caption_quality': True,
    })
    assert resp.status_code == 200
    assert captured == {
        'extra_steps': 1000, 'base_model': 'merge.safetensors',
        'variant': 'base', 'train_type': 'zimage', 'masked': False,
        'allow_unverified_weights': True,
        'allow_caption_mismatch': True,
        'allow_uncaptioned': True,
        'allow_caption_quality': True,
    }


# --- /train/enqueue ----------------------------------------------------------

def test_enqueue_forwards_kwargs(client, monkeypatch):
    _valid(monkeypatch, True)
    ds_id = _create(client)
    captured = {}

    def fake_enqueue(user_id, dataset_id, **kw):
        captured.update(kw)
        return {'queued': True, 'position': 1, 'not_before': None}

    monkeypatch.setattr('app.services.lora_training.enqueue_training', fake_enqueue)
    resp = client.post(f'/api/dataset/{ds_id}/train/enqueue',
                       json={'extra_steps': 500, 'steps': 3000,
                             'allow_caption_mismatch': True,
                             'allow_caption_quality': True})
    assert resp.status_code == 200
    assert captured == {
        'extra_steps': 500,
        'masked': True,
        'steps': 3000,
        'allow_caption_mismatch': True,
        'allow_caption_quality': True,
    }


# --- /train/schedule ---------------------------------------------------------

def test_schedule_past_returns_400(client, monkeypatch):
    _valid(monkeypatch, True)
    ds_id = _create(client)
    resp = client.post(f'/api/dataset/{ds_id}/train/schedule', json={'at': '2000-01-01T00:00'})
    assert resp.status_code == 400
    assert resp.get_json()['error'] == 'scheduled time is in the past'


def test_schedule_invalid_datetime_returns_400(client, monkeypatch):
    _valid(monkeypatch, True)
    ds_id = _create(client)
    resp = client.post(f'/api/dataset/{ds_id}/train/schedule', json={'at': 'not-a-date'})
    assert resp.status_code == 400


def test_schedule_future_enqueues_with_not_before(client, monkeypatch):
    _valid(monkeypatch, True)
    ds_id = _create(client)
    captured = {}

    def fake_enqueue(user_id, dataset_id, **kw):
        captured.update(kw)
        return {'queued': True, 'position': 1, 'not_before': kw.get('not_before')}

    monkeypatch.setattr('app.services.lora_training.enqueue_training', fake_enqueue)
    resp = client.post(f'/api/dataset/{ds_id}/train/schedule', json={
        'at': '2999-01-01T00:00', 'allow_caption_quality': True,
    })
    assert resp.status_code == 200
    assert captured == {
        'extra_steps': None,
        'not_before': '2999-01-01T00:00',
        'masked': True,
        'allow_caption_quality': True,
    }


def test_schedule_tzaware_future_normalizes_and_enqueues(client, monkeypatch):
    _valid(monkeypatch, True)
    ds_id = _create(client)
    captured = {}

    def fake_enqueue(user_id, dataset_id, **kw):
        captured.update(kw)
        return {'queued': True, 'position': 1, 'not_before': kw.get('not_before')}

    monkeypatch.setattr('app.services.lora_training.enqueue_training', fake_enqueue)
    # Use UTC-05:00 offset so converting to local (UTC-based or positive) keeps date in 2999
    resp = client.post(f'/api/dataset/{ds_id}/train/schedule', json={'at': '2999-01-02T00:00:00-05:00'})
    assert resp.status_code == 200
    # tz-aware input is normalized; not_before should be naive local ISO format with year 2999+
    assert 'not_before' in captured
    # After normalization to local time, should be in year 2999 or later
    assert int(captured['not_before'][:4]) >= 2999


def test_schedule_tzaware_past_returns_400(client, monkeypatch):
    _valid(monkeypatch, True)
    ds_id = _create(client)
    resp = client.post(f'/api/dataset/{ds_id}/train/schedule', json={'at': '1999-01-01T00:00:00+02:00'})
    assert resp.status_code == 400
    assert resp.get_json()['error'] == 'scheduled time is in the past'


# --- /train/dequeue, /train/stop ---------------------------------------------

def test_dequeue_calls_service(client, monkeypatch):
    _valid(monkeypatch, True)
    ds_id = _create(client)
    monkeypatch.setattr('app.services.lora_training.dequeue_training', lambda dataset_id: 1)
    resp = client.post(f'/api/dataset/{ds_id}/train/dequeue')
    assert resp.status_code == 200
    assert resp.get_json() == {'ok': True, 'removed': 1}


def test_stop_calls_stop_training(client, monkeypatch):
    _valid(monkeypatch, True)
    calls = []
    monkeypatch.setattr('app.services.lora_training.stop_training', lambda: calls.append(True))
    resp = client.post('/api/dataset/train/stop')
    assert resp.status_code == 200
    assert calls == [True]


def test_stop_targets_the_local_run_shown_by_the_runs_hub(client, monkeypatch):
    _valid(monkeypatch, True)
    calls = []
    monkeypatch.setattr('app.services.lora_training.stop_training',
                        lambda expected_dataset_id=None, expected_run_token=None:
                        calls.append((expected_dataset_id, expected_run_token)) or True)
    resp = client.post('/api/dataset/train/stop',
                       json={'dataset_id': 42, 'run_token': 'run-abc'})
    assert resp.status_code == 200
    assert calls == [(42, 'run-abc')]


@pytest.mark.parametrize('body', [
    {'dataset_id': 42},
    {'run_token': 'run-abc'},
])
def test_stop_rejects_partial_run_identity(client, monkeypatch, body):
    _valid(monkeypatch, True)
    called = []
    monkeypatch.setattr(
        'app.services.lora_training.stop_training',
        lambda **_kw: called.append(True))

    resp = client.post('/api/dataset/train/stop', json=body)

    assert resp.status_code == 400
    assert resp.get_json()['error'] == (
        'dataset_id and run_token must be provided together')
    assert called == []


def test_stop_rejects_a_stale_local_run_card(client, monkeypatch):
    _valid(monkeypatch, True)
    monkeypatch.setattr('app.services.lora_training.stop_training',
                        lambda **_kw: False)
    resp = client.post('/api/dataset/train/stop',
                       json={'dataset_id': 42, 'run_token': 'stale-token'})
    assert resp.status_code == 409
    assert resp.get_json() == {
        'ok': False,
        'error': 'This local run is no longer active. The Runs page was refreshed.',
    }


# --- /train/checkpoints -------------------------------------------------------

def test_checkpoints_returns_family_variant_recommendations(client, monkeypatch):
    _valid(monkeypatch, True)
    ds_id = _create(client)
    monkeypatch.setattr('app.services.lora_training.list_checkpoints',
                        lambda *a, **k: [{'step': 500, 'filename': 'x.safetensors'}])
    step_calls = []
    info_calls = []
    monkeypatch.setattr(
        'app.services.lora_training.recommended_steps',
        lambda dataset_id, **kw: step_calls.append((dataset_id, kw)) or 2500)
    monkeypatch.setattr(
        'app.services.lora_training.recommended_steps_info',
        lambda dataset_id, **kw: info_calls.append((dataset_id, kw))
        or {'steps': 2500, 'family': kw.get('train_type'),
            'variant': kw.get('variant')})
    monkeypatch.setattr('app.services.lora_training.list_imported_checkpoints', lambda *a, **k: [])
    resp = client.get(
        f'/api/dataset/{ds_id}/train/checkpoints?train_type=zimage&variant=base')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['recommended_steps'] == 2500
    assert body['checkpoints'][0]['step'] == 500
    assert body['imported'] == []
    assert step_calls == [(ds_id, {'train_type': 'zimage', 'variant': 'base'})]
    assert info_calls == [(ds_id, {'train_type': 'zimage', 'variant': 'base'})]
    assert body['recommended_steps_info']['variant'] == 'base'


def test_checkpoints_query_forwards_variant_to_local_and_cloud(client, monkeypatch):
    _valid(monkeypatch, True)
    ds_id = _create(client, name='Variant', trigger='variant')
    local_calls = []
    cloud_calls = []
    monkeypatch.setattr(
        'app.services.lora_training.list_checkpoints',
        lambda *a, **kw: local_calls.append(kw) or [])
    monkeypatch.setattr(
        'app.services.lora_training.dataset_disk_usage',
        lambda *a, **kw: {'total_bytes': 0})
    monkeypatch.setattr(
        'app.services.lora_training.list_imported_checkpoints',
        lambda *a, **kw: [])
    monkeypatch.setattr(
        'app.services.cloud_training.cloud_checkpoints',
        lambda dataset_id, train_type=None, variant=None:
        cloud_calls.append((dataset_id, train_type, variant)) or [])
    resp = client.get(
        f'/api/dataset/{ds_id}/train/checkpoints'
        '?base_model=&train_type=zimage&variant=deturbo')
    assert resp.status_code == 200
    assert local_calls and all(
        call['variant'] == 'deturbo' for call in local_calls)
    assert cloud_calls == [(ds_id, 'zimage', 'deturbo')]


# --- /train/base-info ---------------------------------------------------------

def test_base_info_returns_bases_by_type(client, monkeypatch):
    _valid(monkeypatch, True)
    ds_id = _create(client)
    resp = client.get(f'/api/dataset/{ds_id}/train/base-info')
    assert resp.status_code == 200
    body = resp.get_json()
    assert set(body['bases_by_type']) == {'zimage', 'sdxl', 'krea', 'flux', 'flux2klein'}
    assert body['train_type'] == 'zimage'


def test_base_info_unknown_dataset_404(client, monkeypatch):
    _valid(monkeypatch, True)
    resp = client.get('/api/dataset/999999/train/base-info')
    assert resp.status_code == 404


def test_base_info_comfyui_unconfigured_flag(client, monkeypatch):
    """Fresh config: no comfyui.base_dir -> comfyui_configured False, so the UI can
    say 'point the app at ComfyUI' instead of a blind 'No checkpoint found'."""
    _valid(monkeypatch, True)
    ds_id = _create(client)
    body = client.get(f'/api/dataset/{ds_id}/train/base-info').get_json()
    assert body['comfyui_configured'] is False
    assert body['models_dir'] == ''


def test_base_info_comfyui_configured_flag(client, monkeypatch, tmp_path):
    from app import config as cfg
    _valid(monkeypatch, True)
    base = tmp_path / 'comfyui'
    (base / 'models').mkdir(parents=True)
    cfg.save_config({'comfyui': {'base_dir': str(base)}})
    ds_id = _create(client)
    body = client.get(f'/api/dataset/{ds_id}/train/base-info').get_json()
    assert body['comfyui_configured'] is True
    assert body['models_dir'].replace('/', '\\').endswith('models')


# --- /train/prepare-base -------------------------------------------------------

def test_prepare_base_rejects_unknown_base(client, monkeypatch):
    _valid(monkeypatch, True)
    ds_id = _create(client)
    monkeypatch.setattr('app.routes.training.get_zimage_models', lambda: ['z image\\known.safetensors'])
    resp = client.post(f'/api/dataset/{ds_id}/train/prepare-base', json={'base_model': 'unknown.safetensors'})
    assert resp.status_code == 400


def test_prepare_base_already_converted_returns_done(client, monkeypatch):
    _valid(monkeypatch, True)
    ds_id = _create(client)
    monkeypatch.setattr('app.routes.training.get_zimage_models', lambda: ['z image\\known.safetensors'])
    monkeypatch.setattr('app.services.zimage_convert.is_converted', lambda m: True)
    resp = client.post(f'/api/dataset/{ds_id}/train/prepare-base',
                       json={'base_model': 'z image\\known.safetensors'})
    assert resp.status_code == 200
    assert resp.get_json()['status'] == 'done'


def test_prepare_base_starts_conversion(client, monkeypatch):
    _valid(monkeypatch, True)
    ds_id = _create(client)
    monkeypatch.setattr('app.routes.training.get_zimage_models', lambda: ['z image\\known.safetensors'])
    monkeypatch.setattr('app.services.zimage_convert.is_converted', lambda m: False)
    calls = []
    monkeypatch.setattr('app.services.zimage_convert.start_convert_async',
                        lambda app, m: calls.append(m))
    resp = client.post(f'/api/dataset/{ds_id}/train/prepare-base',
                       json={'base_model': 'z image\\known.safetensors'})
    assert resp.status_code == 200
    assert resp.get_json()['status'] == 'running'
    assert calls == ['z image\\known.safetensors']


def test_prepare_base_requires_base_model(client, monkeypatch):
    _valid(monkeypatch, True)
    ds_id = _create(client)
    resp = client.post(f'/api/dataset/{ds_id}/train/prepare-base', json={})
    assert resp.status_code == 400


# --- /train/checkpoint/delete, /train/import -----------------------------------

def test_checkpoint_delete_calls_service(client, monkeypatch):
    _valid(monkeypatch, True)
    ds_id = _create(client)
    monkeypatch.setattr('app.services.lora_training.delete_imported_checkpoint',
                        lambda user_id, dataset_id, fn, family=None: fn)
    resp = client.post(f'/api/dataset/{ds_id}/train/checkpoint/delete', json={'filename': 'x.safetensors'})
    assert resp.status_code == 200
    assert resp.get_json() == {'ok': True, 'removed': 'x.safetensors'}


def test_checkpoint_delete_unknown_returns_400(client, monkeypatch):
    _valid(monkeypatch, True)
    ds_id = _create(client)
    monkeypatch.setattr('app.services.lora_training.delete_imported_checkpoint',
                        lambda *a, **k: (_ for _ in ()).throw(ValueError('checkpoint inconnu')))
    resp = client.post(f'/api/dataset/{ds_id}/train/checkpoint/delete', json={'filename': 'nope.safetensors'})
    assert resp.status_code == 400


def test_import_checkpoint_calls_service(client, monkeypatch):
    _valid(monkeypatch, True)
    ds_id = _create(client)
    captured = {}

    def fake_import(user_id, dataset_id, fn, **kw):
        captured.update(kw)
        return f'/some/dir/{fn}'

    monkeypatch.setattr('app.services.lora_training.import_checkpoint', fake_import)
    resp = client.post(f'/api/dataset/{ds_id}/train/import', json={
        'filename': 'x.safetensors', 'base_model': '',
        'train_type': 'zimage', 'variant': 'deturbo',
    })
    assert resp.status_code == 200
    assert resp.get_json() == {'ok': True, 'dest': 'x.safetensors'}
    assert captured == {
        'base_model': '', 'family': 'zimage', 'variant': 'deturbo'}


def test_variant_forwarded_to_open_delete_and_cleanup(client, monkeypatch):
    _valid(monkeypatch, True)
    ds_id = _create(client, name='Operations', trigger='operations')
    captured = {}

    monkeypatch.setattr(
        'app.services.lora_training.open_training_folder',
        lambda *a, **kw: captured.setdefault('open', kw) or 'C:/run')
    monkeypatch.setattr(
        'app.services.lora_training.delete_checkpoint',
        lambda *a, **kw: captured.setdefault('delete', kw) or 'x.safetensors')
    monkeypatch.setattr(
        'app.services.lora_training.cleanup_checkpoints',
        lambda *a, **kw: captured.setdefault('cleanup', kw)
        or {'removed': 0, 'kept': []})

    common = {'base_model': '', 'train_type': 'zimage', 'variant': 'base'}
    assert client.post(
        f'/api/dataset/{ds_id}/train/open-folder',
        json={**common, 'target': 'run'}).status_code == 200
    assert client.post(
        f'/api/dataset/{ds_id}/train/run-checkpoint/delete',
        json={**common, 'filename': 'x.safetensors'}).status_code == 200
    assert client.post(
        f'/api/dataset/{ds_id}/train/checkpoints/cleanup',
        json={**common, 'keep_filenames': []}).status_code == 200
    assert captured['open']['variant'] == 'base'
    assert captured['delete']['variant'] == 'base'
    assert captured['cleanup']['variant'] == 'base'
