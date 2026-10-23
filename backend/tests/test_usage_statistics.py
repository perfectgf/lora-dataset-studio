"""Consent and outbound schema are the privacy boundary, not SDK defaults."""
import json
import threading

import pytest

from app.usage_statistics import MAX_AGE, MAX_EVENTS, POLICY_VERSION, UsageStatistics

TOKEN = 'phc_' + 'test' * 12


@pytest.fixture
def recorder(tmp_path):
    sent = []
    now = [1800000000.0]
    service = UsageStatistics(tmp_path, project_token=TOKEN, start_worker=False,
                              clock=lambda: now[0], sender=lambda payload: sent.append(payload) or True)
    return service, sent, now


def test_default_and_decline_never_create_identity_or_contact_collector(recorder):
    service, sent, _ = recorder
    assert service.status()['choice'] is None
    service.activity('datasets')
    service.flush_once()
    assert not service.path.exists()
    assert sent == []
    service.set_enabled(False)
    assert service._state['installation_id'] is None
    service.activity('bank')
    service.flush_once()
    assert sent == []
    again = UsageStatistics(service.path.parent, project_token=TOKEN, start_worker=False)
    assert again.status()['choice'] == 'disabled'


def test_missing_or_admin_token_cannot_enable(tmp_path):
    for token in ('', 'phx_' + 'test' * 12):
        service = UsageStatistics(tmp_path, project_token=token, start_worker=False)
        with pytest.raises(ValueError):
            service.set_enabled(True)
        assert not service.enabled and not service.path.exists()


def test_active_days_and_features_deduplicate_across_restart(recorder):
    service, sent, now = recorder
    service.set_enabled(True)
    installation = service._state['installation_id']
    for _ in range(3):
        service.activity('datasets')
    service.activity('bank')
    assert [row['event'] for row in service._state['events']].count('lds_active_day') == 1
    again = UsageStatistics(service.path.parent, project_token=TOKEN, start_worker=False, clock=lambda: now[0])
    assert again._state['installation_id'] == installation
    again.activity('datasets')
    assert len(again._state['events']) == 3
    now[0] += 86400
    again.activity('datasets')
    assert len(again._state['events']) == 5
    assert sent == []  # startup/activity never performs HTTPS


def test_outbound_closed_schema_and_original_version(recorder, monkeypatch):
    service, sent, _ = recorder
    service.set_enabled(True)
    service.activity('datasets')
    generation = service.consent_generation()
    service.operation(generation, 'datasets', 'caption', 'failed', seconds=12,
                      error_code='out_of_memory')
    service.activity('private-route-content')
    service.operation(generation, 'datasets', 'private-action', 'failed')
    original_version = service._state['events'][0]['version']
    monkeypatch.setattr('app.usage_statistics.APP_VERSION', '2099.01.01')
    assert service.flush_once()
    assert len(sent[0]['batch']) == 3
    for event in sent[0]['batch']:
        assert set(event) == {'uuid', 'event', 'distinct_id', 'timestamp', 'properties'}
        properties = event['properties']
        assert properties['$process_person_profile'] is False
        assert properties['$geoip_disable'] is True
        assert properties['app_version'] == original_version
        assert properties['environment'] == 'production'
        assert not set(properties) - {
            'schema_version', 'app_version', 'os', 'environment', '$process_person_profile', '$geoip_disable',
            'first_active_day', 'feature', 'action', 'result', 'error_code', 'error_reason', 'duration_bucket',
            'failure_stage', 'execution', 'model_family', 'caption_backend'}
    assert 'private-' not in json.dumps(sent)
    assert sent[0]['batch'][-1]['properties']['duration_bucket'] == '10_to_60s'


def test_withdrawal_clears_outbox_identity_and_old_operation_tickets(recorder):
    service, sent, _ = recorder
    service.set_enabled(True)
    generation = service.consent_generation()
    old_identity = service._state['installation_id']
    service.activity('datasets')
    service.set_enabled(False)
    assert service.status()['queued_events'] == 0
    assert service._state['installation_id'] is None
    service.flush_once()
    service.set_enabled(True)
    assert service._state['installation_id'] != old_identity
    service.operation(generation, 'datasets', 'caption', 'completed')
    service.operation(None, 'datasets', 'caption', 'completed')
    assert service.status()['queued_events'] == 0
    assert sent == []


def test_recipient_change_requires_new_consent(tmp_path, recorder):
    service, _, _ = recorder
    service.set_enabled(True)
    service.activity('datasets')
    changed = UsageStatistics(tmp_path, project_token=TOKEN + 'changed', start_worker=False)
    assert changed.status()['choice'] is None
    assert not changed.enabled and changed.status()['queued_events'] == 0


def test_test_environment_is_labelled_and_cannot_reuse_production_consent(recorder):
    service, sent, _ = recorder
    service.set_enabled(True)
    service.activity('datasets')
    trial = UsageStatistics(service.path.parent, project_token=TOKEN, environment='test',
                            start_worker=False, sender=lambda payload: sent.append(payload) or True)
    assert not trial.enabled and trial.status()['queued_events'] == 0
    trial.set_enabled(True)
    trial.activity('datasets')
    assert trial.flush_once()
    assert all(row['properties']['environment'] == 'test' for row in sent[0]['batch'])


def test_corrupted_preferences_fail_closed(tmp_path):
    path = tmp_path / 'usage-statistics.json'
    path.write_text('{corrupt', encoding='utf-8')
    service = UsageStatistics(tmp_path, project_token=TOKEN, start_worker=False)
    assert not service.enabled and service.status()['choice'] is None


def test_disk_and_queue_fields_cannot_be_used_to_send_arbitrary_content(recorder):
    service, sent, _ = recorder
    service.set_enabled(True)
    service.activity('datasets')
    service._state['events'][0]['properties']['prompt'] = 'private-prompt-content'
    service._state['events'][1]['properties']['feature'] = ['private-caption-content']
    assert not service.flush_once()
    assert sent == []


def test_offline_retries_are_bounded_backoff_and_keep_same_event_ids(recorder):
    service, _, now = recorder
    attempts = []
    service._sender = lambda payload: attempts.append(payload) or False
    service.set_enabled(True)
    service.activity('bank')
    for _ in range(5):
        assert not service.flush_once()
        count = len(attempts)
        service.flush_once()
        assert len(attempts) == count
        now[0] += 4000
    assert len(attempts) == 5
    assert len({batch['batch'][0]['uuid'] for batch in attempts}) == 1
    assert service.status()['queued_events'] == 0


def test_outbox_is_bounded_and_expires_offline_work(recorder):
    service, sent, now = recorder
    service.set_enabled(True)
    for _ in range(MAX_EVENTS + 3):
        service._queue('lds_active_day', {}, now[0])
    assert service.status()['queued_events'] == MAX_EVENTS
    now[0] += MAX_AGE + 1
    assert not service.flush_once() and sent == []


def test_opt_out_serializes_with_inflight_send(recorder):
    service, _, _ = recorder
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    def send(_payload):
        entered.set()
        assert release.wait(3)
        return True
    service._sender = send
    service.set_enabled(True)
    service.activity('bank')
    sender = threading.Thread(target=service.flush_once)
    sender.start()
    assert entered.wait(3)
    def disable():
        service.set_enabled(False)
        done.set()
    disabler = threading.Thread(target=disable)
    disabler.start()
    assert not done.wait(0.02)
    release.set()
    sender.join(3)
    disabler.join(3)
    assert done.is_set() and not service.enabled
    assert service.status()['queued_events'] == 0


@pytest.fixture
def configured_app(plugin_app_factory, monkeypatch):
    monkeypatch.setenv('LDS_USAGE_PROJECT_TOKEN', TOKEN)
    monkeypatch.delenv('LDS_USAGE_DISABLED', raising=False)
    return plugin_app_factory()


def _enable(client):
    return client.put('/api/usage-statistics', json={'enabled': True, 'policy_version': POLICY_VERSION})


def test_consent_api_validates_choice_and_activity_schema(configured_app):
    client = configured_app.test_client()
    assert client.get('/api/usage-statistics').json['choice'] is None
    assert client.put('/api/usage-statistics', json={'enabled': 'true', 'policy_version': 1}).status_code == 400
    assert client.put('/api/usage-statistics', json={'enabled': True, 'policy_version': True}).status_code == 400
    assert client.put('/api/usage-statistics', json={'enabled': True, 'policy_version': POLICY_VERSION - 1}).status_code == 400
    assert _enable(client).json['enabled'] is True
    assert client.post('/api/usage-statistics/activity', json={'feature': 'datasets', 'prompt': 'private-text'}).status_code == 400
    assert client.post('/api/usage-statistics/activity', json={'feature': 'datasets'}).status_code == 204
    state = client.get('/api/usage-statistics').json
    assert state['queued_events'] == 2
    assert 'installation_id' not in state and 'project_token' not in state


def test_existing_request_guards_still_protect_consent(configured_app):
    client = configured_app.test_client()
    configured_app.config['WTF_CSRF_ENABLED'] = True
    assert _enable(client).status_code == 400
    assert client.get('/api/usage-statistics').json['enabled'] is False


def test_real_operation_hook_counts_result_without_any_request_content(configured_app):
    client = configured_app.test_client()
    _enable(client)
    response = client.post('/api/dataset/create', json={'name': 'private-name-content', 'trigger_word': 'private-trigger-content'})
    assert response.status_code in {200, 201}
    service = configured_app.extensions['lds_usage_statistics']
    rows = service._state['events']
    result = [row for row in rows if row['event'] == 'lds_action_result']
    assert len(result) == 1 and result[0]['properties']['result'] == 'completed'
    assert 'private-' not in service.path.read_text(encoding='utf-8')
    before = len(rows)
    client.get('/api/health')
    assert len(service._state['events']) == before


def test_failed_operation_hook_reports_fixed_code_only(configured_app):
    client = configured_app.test_client()
    _enable(client)
    response = client.post('/api/dataset/create', json={})
    assert response.status_code == 400
    rows = configured_app.extensions['lds_usage_statistics']._state['events']
    result = next(row['properties'] for row in rows if row['event'] == 'lds_action_result')
    assert result['result'] == 'failed' and result['error_code'] == 'invalid_input'


def test_bank_background_result_is_actual_completion_and_respects_revocation(configured_app):
    from app.services import bank_jobs
    service = configured_app.extensions['lds_usage_statistics']
    service.set_enabled(True)
    def failed(_job):
        raise RuntimeError('private-error-content')
    bank_jobs.start(configured_app, 81001, 'scan', failed)
    results = [row['properties'] for row in service._state['events'] if row['event'] == 'lds_action_result']
    assert results[-1]['result'] == 'failed'
    assert 'private-error-content' not in service.path.read_text(encoding='utf-8')
    bank_jobs.start(configured_app, 81002, 'scan', lambda _job: service.set_enabled(False))
    assert service.status()['queued_events'] == 0


@pytest.mark.parametrize('outcome,expected', [
    ({'ok': True, 'captioned': 3, 'skipped': 2}, 'partial'),
    ({'ok': True, 'captioned': 0, 'skipped': 2}, 'failed'),
    ({'ok': True, 'captioned': 3, 'skipped': 0}, 'completed'),
    ({'ok': True, 'captioned': 3, 'skipped': 2, 'stopped': True}, 'cancelled'),
])
def test_caption_hook_distinguishes_partial_and_cancelled_outcomes(configured_app, outcome, expected):
    from flask import jsonify
    configured_app.view_functions['datasets.dataset_caption'] = lambda **_kw: jsonify(outcome)
    client = configured_app.test_client()
    _enable(client)
    assert client.post('/api/dataset/1/caption', json={}).status_code == 200
    results = [row['properties'] for row in configured_app.extensions['lds_usage_statistics']._state['events']
               if row['event'] == 'lds_action_result']
    assert results[-1]['result'] == expected


def test_bank_partial_result_is_not_a_success(configured_app):
    from app.services import bank_jobs
    service = configured_app.extensions['lds_usage_statistics']
    service.set_enabled(True)
    bank_jobs.start(configured_app, 81003, 'caption', lambda job: job.update(_usage_result='partial'))
    results = [row['properties'] for row in service._state['events'] if row['event'] == 'lds_action_result']
    assert results[-1]['result'] == 'partial'


@pytest.mark.parametrize('kind,status,code,reason', [
    ('training', 400, 'busy', 'training_already_running'),
    ('caption', 409, 'unavailable', 'caption_backend_unavailable'),
    ('gpu', 503, 'busy', 'gpu_busy'),
    ('dataset', 409, 'busy', 'dataset_busy'),
    ('fence', 409, 'busy', 'ollama_fence_blocked'),
    ('generic', 409, 'operation_failed', 'unknown'),
])
def test_typed_refusals_preserve_api_and_classify_without_messages(configured_app, kind, status, code, reason):
    from app.operation_errors import TrainingAlreadyRunning, CaptionBackendUnavailable
    from app.gpu_window import GpuBusyError
    from app.services.dataset_activity import DatasetActivityBusy
    from app.services.vision_ollama import LocalOllamaFenceError
    from app.routes._common import _map_error
    classes = {'training': TrainingAlreadyRunning, 'caption': CaptionBackendUnavailable,
               'gpu': GpuBusyError, 'dataset': DatasetActivityBusy,
               'fence': LocalOllamaFenceError, 'generic': RuntimeError}
    configured_app.view_functions['datasets.dataset_caption'] = lambda **kw: _map_error(classes[kind]('private-error-content'))
    client = configured_app.test_client()
    _enable(client)
    assert client.post('/api/dataset/1/caption', json={}).status_code == status
    service = configured_app.extensions['lds_usage_statistics']
    props = [row['properties'] for row in service._state['events'] if row['event'] == 'lds_action_result'][-1]
    assert (props['result'], props['error_code'], props['error_reason']) == ('failed', code, reason)
    assert 'private-error-content' not in service.path.read_text(encoding='utf-8')


@pytest.mark.parametrize('payload,reason', [({}, 'bank_name_required'),
    ({'name': 'private-name', 'folder': ''}, 'bank_folder_unavailable')])
def test_bank_validation_reports_fixed_reason(configured_app, payload, reason):
    client = configured_app.test_client()
    _enable(client)
    assert client.post('/api/bank/create', json=payload).status_code == 400
    props = [row['properties'] for row in configured_app.extensions['lds_usage_statistics']._state['events']
             if row['event'] == 'lds_action_result'][-1]
    assert props['error_reason'] == reason


def test_error_reasons_reject_untrusted_values_and_accept_older_queued_events(recorder):
    from app.usage_statistics import _valid_row, error_categories
    service, sent, _ = recorder
    service.set_enabled(True)
    generation = service.consent_generation()
    service.operation(generation, 'datasets', 'caption', 'failed', error_reason='private-content')
    assert service.status()['queued_events'] == 0
    service.operation(generation, 'datasets', 'caption', 'completed')
    row = service._state['events'][-1]
    del row['properties']['error_reason']
    assert _valid_row(row)
    row['properties']['error_reason'] = 'private-content'
    assert not _valid_row(row)
    class Untrusted(RuntimeError):
        usage_error_code = ['private-content']
        usage_error_reason = {'private-content': True}
    assert error_categories(Untrusted('private-content')) == ('operation_failed', 'unknown')
    service.flush_once()
    assert not sent


@pytest.mark.parametrize('endpoint,suffix,action', [
    ('dataset_train', '', 'train_start'),
    ('dataset_train_continue', '/continue', 'train_resume'),
    ('dataset_train_enqueue', '/enqueue', 'train_enqueue'),
])
def test_training_requests_distinguish_launch_resume_and_queue(configured_app, endpoint, suffix, action):
    from flask import jsonify
    configured_app.view_functions[f'training.{endpoint}'] = lambda **kw: jsonify(ok=True)
    client = configured_app.test_client()
    _enable(client)
    assert client.post('/api/dataset/1/train' + suffix, json={}).status_code == 200
    props = [row['properties'] for row in configured_app.extensions['lds_usage_statistics']._state['events']
             if row['event'] == 'lds_action_result'][-1]
    assert (props['action'], props['result']) == (action, 'accepted')


def test_bank_worker_reports_caption_availability_like_the_dataset_route(configured_app, monkeypatch):
    from app.services import bank_jobs
    from app.services import vision_llm
    from app.operation_errors import CaptionBackendUnavailable
    monkeypatch.setattr(vision_llm, 'provider', lambda: 'lmstudio')
    service = configured_app.extensions['lds_usage_statistics']
    service.set_enabled(True)
    def fail(job):
        raise CaptionBackendUnavailable('private-engine-details')
    bank_jobs.start(configured_app, 81004, 'caption', fail)
    props = [row['properties'] for row in service._state['events'] if row['event'] == 'lds_action_result'][-1]
    assert (props['error_code'], props['error_reason']) == ('unavailable', 'caption_backend_unavailable')
    assert (props['failure_stage'], props['execution'], props['caption_backend']) == ('preparation', 'local', 'lmstudio')
    assert 'private-engine-details' not in service.path.read_text(encoding='utf-8')
