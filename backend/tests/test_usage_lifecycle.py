"""Committed outcomes and consent boundaries, using real mapped transactions."""
import json
import uuid

import pytest

from app.extensions import db
from app.models import CloudTrainingRun, FaceDataset, ImageGenerationQueue, LoraTestImage
from app.usage_statistics import UsageStatistics, operation_context


@pytest.fixture
def observed(plugin_app_factory, monkeypatch):
    monkeypatch.setenv('LDS_USAGE_PROJECT_TOKEN', 'phc_' + 'test' * 12)
    monkeypatch.delenv('LDS_USAGE_DISABLED', raising=False)
    app = plugin_app_factory()
    with app.app_context():
        service = app.extensions['lds_usage_statistics']
        service.set_enabled(True)
        yield app, service


def outcomes(service, result=None):
    return [row['properties'] for row in service._state['events']
            if row['event'] == 'lds_action_result'
            and (result is None or row['properties']['result'] == result)]


def cloud(status='preparing', family='flux'):
    row = CloudTrainingRun(dataset_id=1, status=status,
                           train_params=json.dumps({'train_type': family, 'base_model': 'private-model-path'}),
                           error='private-error-content', auth_token='private-auth-content')
    db.session.add(row)
    db.session.commit()
    return row


def studio():
    dataset = FaceDataset(user_id='local', name='test', trigger_word='test')
    db.session.add(dataset)
    db.session.flush()
    job_id = str(uuid.uuid4())
    job = ImageGenerationQueue(job_id=job_id, user_id='local', status='pending', workflow_data='{}',
                               prompt='private-prompt', job_metadata=json.dumps({
                                   'is_lora_test': True, 'family': 'krea', 'base_model': 'private-file'}))
    image = LoraTestImage(dataset_id=dataset.id, checkpoint='private-checkpoint', strength=1,
                          job_id=job_id, status='pending')
    db.session.add_all([job, image])
    db.session.commit()
    return job, image


def test_cloud_sdk_rows_report_the_actual_failed_phase_once(observed):
    _, service = observed
    from lds_sdk.cloud_host.models import CloudTrainingRun as PluginRun
    assert PluginRun is CloudTrainingRun
    run = cloud()
    run.status = 'uploading'
    db.session.commit()
    run.status = 'error'
    db.session.commit()
    run.phase_detail = 'private-progress-detail'
    db.session.commit()
    assert len(outcomes(service, 'accepted')) == 1
    failed = outcomes(service, 'failed')
    assert len(failed) == 1
    assert failed[0]['failure_stage'] == 'uploading'
    assert failed[0]['execution'] == 'cloud' and failed[0]['model_family'] == 'flux'
    assert 'private-' not in service.path.read_text(encoding='utf-8')


@pytest.mark.parametrize('terminal,result', [('done', 'completed'), ('stopped', 'cancelled'), ('error_pod_kept', 'failed')])
def test_cloud_terminal_states(observed, terminal, result):
    _, service = observed
    run = cloud()
    run.status = 'training'
    db.session.commit()
    run.status = terminal
    db.session.commit()
    assert len(outcomes(service, result)) == 1


def test_studio_waits_until_the_finished_image_has_been_saved(observed):
    _, service = observed
    job, image = studio()
    job.status = 'sent_to_comfy'
    db.session.commit()
    job.status = 'completed'
    db.session.commit()
    assert not outcomes(service, 'completed')
    image.status = 'failed'
    image.error = 'private-storage-path'
    db.session.commit()
    failed = outcomes(service, 'failed')
    assert len(failed) == 1
    assert failed[0]['failure_stage'] == 'saving'
    assert failed[0]['error_reason'] == 'studio_result_save_failed'
    assert failed[0]['model_family'] == 'krea'


@pytest.mark.parametrize('terminal,result', [('done', 'completed'), ('cancelled', 'cancelled')])
def test_studio_completion_and_cancellation_are_counted_once(observed, terminal, result):
    _, service = observed
    job, image = studio()
    image.status = terminal
    job.status = 'completed' if terminal == 'done' else 'cancelled'
    db.session.commit()
    image.rating = 1
    db.session.commit()
    assert len(outcomes(service, result)) == 1


def test_rolled_back_reservation_or_terminal_transition_is_not_reported(observed):
    _, service = observed
    run = CloudTrainingRun(dataset_id=1, status='preparing')
    db.session.add(run)
    db.session.flush()
    db.session.rollback()
    assert outcomes(service) == []
    run = cloud()
    run.status = 'error'
    db.session.flush()
    db.session.rollback()
    assert not outcomes(service, 'failed')
    run.status = 'done'
    db.session.commit()
    assert len(outcomes(service, 'completed')) == 1


def test_released_savepoint_does_not_report_before_outer_commit(observed):
    _, service = observed
    with db.session.begin_nested():
        run = CloudTrainingRun(dataset_id=1, status='preparing')
        db.session.add(run)
        db.session.flush()
    assert outcomes(service) == []
    db.session.rollback()
    assert outcomes(service) == []


def test_rolled_back_savepoint_keeps_the_outer_operation(observed):
    _, service = observed
    run = CloudTrainingRun(dataset_id=1, status='preparing')
    db.session.add(run)
    db.session.flush()
    nested = db.session.begin_nested()
    run.status = 'error'
    db.session.flush()
    nested.rollback()
    db.session.commit()
    assert len(outcomes(service, 'accepted')) == 1
    assert not outcomes(service, 'failed')
    run.status = 'done'
    db.session.commit()
    assert len(outcomes(service, 'completed')) == 1


def test_choice_changed_before_commit_and_preexisting_runs_are_not_adopted(observed):
    _, service = observed
    db.session.add(CloudTrainingRun(dataset_id=1, status='preparing'))
    db.session.flush()
    service.set_enabled(False)
    service.set_enabled(True)
    db.session.commit()
    assert outcomes(service) == []
    run = cloud(status='training')
    run.status = 'done'
    db.session.commit()
    assert outcomes(service) == []


def test_old_rows_and_new_consent_do_not_report_old_runs(observed):
    app, service = observed
    run = cloud()
    service.set_enabled(False)
    service.set_enabled(True)
    run.status = 'done'
    db.session.commit()
    assert outcomes(service) == []
    run = cloud()
    app.extensions['lds_usage_lifecycle']._runs.clear()  # process restart loses old tickets
    run.status = 'error'
    db.session.commit()
    assert not outcomes(service, 'failed')


def test_unknown_context_is_coarsened_and_tampered_context_is_never_sent(observed):
    _, service = observed
    run = cloud(family='private-model-path')
    run.status = 'done'
    db.session.commit()
    assert outcomes(service, 'completed')[0]['model_family'] == 'unknown'
    generation = service.consent_generation()
    service.operation(generation, 'training', 'train_run', 'completed',
                      context={**operation_context(), 'filename': 'private-file'})
    assert len(outcomes(service, 'completed')) == 1
    row = service._state['events'][-1]
    row['properties']['model_family'] = 'private-model-path'
    sent = []
    service._sender = lambda payload: sent.append(payload) or True
    service.flush_once()
    assert 'private-' not in json.dumps(sent)


def test_new_policy_requires_reconsent_but_remembers_a_refusal(tmp_path, monkeypatch):
    import app.usage_statistics as usage
    token = 'phc_' + 'test' * 12
    monkeypatch.setattr(usage, 'POLICY_VERSION', 1)
    old = UsageStatistics(tmp_path, project_token=token, start_worker=False)
    old.set_enabled(True)
    old.activity('studio')
    monkeypatch.setattr(usage, 'POLICY_VERSION', 2)
    fresh = UsageStatistics(tmp_path, project_token=token, start_worker=False)
    assert fresh.status()['choice'] is None and not fresh.enabled
    assert fresh.status()['queued_events'] == 0
    old.set_enabled(False)
    refused = UsageStatistics(tmp_path, project_token=token, start_worker=False)
    assert refused.status()['choice'] == 'disabled'


def test_recorder_failure_cannot_break_the_business_commit(observed, monkeypatch):
    app, _ = observed
    def broken(*args):
        raise OSError('test storage failure')
    monkeypatch.setattr(app.extensions['lds_usage_lifecycle'], 'accept', broken)
    run = cloud()
    assert db.session.get(CloudTrainingRun, run.id).status == 'preparing'
