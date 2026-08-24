"""The minutes between the click and step 1, made observable.

A cloud launch rents a machine and waits for it to boot: several minutes is
normal, and run #134 (production) died on 'pod did not become ready in time —
no boot progress for 25 min'. Both looked identical from the UI — one frozen
'Launching…' button and, later, one motionless phase sentence. The monitor
always knew which step it was on; ``launch_view`` is that knowledge, ordered
and clocked.

Everything here is offline: no pod, no vast API, no thread.
"""
from datetime import timedelta
from app.utils.timestamps import naive_utcnow

import pytest


@pytest.fixture()
def ct(app, monkeypatch):
    from app.services import cloud_training
    monkeypatch.setattr(cloud_training, '_start_monitor', lambda *a, **k: None)
    with app.app_context():
        yield cloud_training


def _mkrun(ct, **kw):
    fields = dict(dataset_id=1, status='preparing', job_name='j1',
                  created_at=naive_utcnow())
    fields.update(kw)
    run = ct.CloudTrainingRun(**fields)
    ct.db.session.add(run)
    ct.db.session.commit()
    return run


def _keys(view, state):
    return [s['key'] for s in view['steps'] if s['state'] == state]


# ---------------------------------------------- the five ordered steps --

def test_the_launch_is_five_named_steps_in_order(ct):
    run = _mkrun(ct, phase_detail='Preparing dataset (masks)…')
    view = ct.launch_view(run)
    assert [s['key'] for s in view['steps']] == [
        'staging', 'offer', 'boot', 'upload', 'start']
    assert view['active_step'] == 'staging'
    assert _keys(view, 'done') == []
    assert _keys(view, 'pending') == ['offer', 'boot', 'upload', 'start']


@pytest.mark.parametrize('status, detail, expected', [
    ('preparing', 'Preparing dataset (masks)…', 'staging'),
    # The offer search runs under 'preparing' too — only its phase line tells
    # them apart, which is exactly why _provision now writes one.
    ('preparing', 'Searching for a GPU offer…', 'offer'),
    ('provisioning', 'Instance created — booting', 'boot'),
    ('provisioning', 'Waiting for the pod to boot — pod up — waiting for the UI to answer', 'boot'),
    ('uploading', 'Uploading dataset', 'upload'),
    # The job is created and queued while the row still reads 'uploading':
    # only start_job earns 'training'.
    ('uploading', 'Job created on the pod', 'start'),
])
def test_every_real_phase_line_places_the_run_on_a_step(ct, status, detail, expected):
    view = ct.launch_view(_mkrun(ct, status=status, phase_detail=detail))
    assert view['active_step'] == expected
    assert _keys(view, 'active') == [expected]


def test_an_unknown_phase_line_still_places_the_run_instead_of_vanishing(ct):
    # A phase sentence this function has never seen must degrade to the first
    # step of its status, never to a blank card.
    view = ct.launch_view(_mkrun(ct, status='preparing', phase_detail='something new'))
    assert view['active_step'] == 'staging'


def test_earlier_steps_are_marked_done_so_the_list_reads_as_progress(ct):
    view = ct.launch_view(_mkrun(ct, status='uploading', phase_detail='Uploading dataset'))
    assert _keys(view, 'done') == ['staging', 'offer', 'boot']
    assert _keys(view, 'pending') == ['start']


# ------------------------------------------------------- the two clocks --

def test_elapsed_is_counted_from_the_click_not_from_the_last_write(ct):
    # updated_at is re-stamped on every monitor poll; only created_at answers
    # "how long have I been waiting since I pressed the button?".
    run = _mkrun(ct, status='provisioning',
                 created_at=naive_utcnow() - timedelta(minutes=7, seconds=30))
    assert 440 <= ct.launch_view(run)['elapsed_seconds'] <= 460


def test_the_boot_deadlines_come_from_the_config_the_boot_wait_enforces(ct):
    # run #134's 25 min was a configured ready_timeout, not the code default.
    run = _mkrun(ct, status='provisioning', phase_detail='Waiting for the pod to boot')
    view = ct.launch_view(run, cloud_cfg={'ready_timeout_minutes': 25,
                                          'boot_budget_minutes': 40})
    assert view['boot_idle_limit_seconds'] == 25 * 60
    assert view['boot_budget_seconds'] == 40 * 60


def test_absent_config_reports_the_defaults_the_code_actually_uses(ct):
    view = ct.launch_view(_mkrun(ct, status='provisioning'), cloud_cfg={})
    assert view['boot_idle_limit_seconds'] == ct.READY_TIMEOUT_SECONDS
    assert view['boot_budget_seconds'] == 90 * 60


# ------------------------------------------- it stops when launching stops --

@pytest.mark.parametrize('status', ['training', 'downloading', 'done', 'error',
                                    'stopped', 'error_pod_kept'])
def test_no_launch_view_once_the_job_is_running_or_the_run_is_over(ct, status):
    # The step counter (and, for a failure, the error) takes over — a launch
    # checklist still on screen next to 'training' would be a lie.
    assert ct.launch_view(_mkrun(ct, status=status)) is None


def test_the_offer_search_stops_wearing_the_staging_phase_line(ct, monkeypatch):
    # Searching vast.ai runs under the 'preparing' status; while it kept the
    # 'Preparing dataset (masks)…' sentence, a search that found nothing was
    # indistinguishable from a dataset export that had hung.
    run = _mkrun(ct, phase_detail='Preparing dataset (masks)…', staging_dir='s',
                 train_params='{"train_type": "zimage"}')
    monkeypatch.setattr(ct.vast_client, 'search_offers', lambda **kw: [])
    with pytest.raises(RuntimeError, match='offer'):
        ct._provision(run)
    assert run.phase_detail.startswith('Searching for a GPU offer')
    assert ct._active_launch_step(run.status, run.phase_detail) == 'offer'


def test_the_run_payload_carries_the_launch_view_for_the_cards(ct):
    run = _mkrun(ct, status='provisioning', phase_detail='Instance created — booting')
    payload = ct._run_payload(run)
    assert payload['launch']['active_step'] == 'boot'
    assert ct._run_payload(_mkrun(ct, status='done'))['launch'] is None
