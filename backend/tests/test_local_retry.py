"""↻ Retry of a FAILED LOCAL training run (Runs page).

Local runs carry no status column: their launch is recorded once in the
provenance registry (TrainingRunRecord, source='local'), and the only signal
that one crashed is the transient global `training_error` the watcher writes on
rc≠0 (cleared on the next launch, TTL-capped). Local training is single-flight,
so at most one local run is "failed" at a time.

`retry_local_run` mirrors the cloud ↻ Retry: a REAL launch_training replaying the
identity params stamped for that launch (family/variant/base/masked/steps) with
every normal guardrail — GPU-collision refusal, normal preflight, no bypass —
and the live dataset (slider settings included) as the source of truth.
"""
import json

import pytest

from app.config import LOCAL_USER


def _configure_aitoolkit(tmp_path, monkeypatch, app):
    """Fake ai-toolkit install: venv python + run.py present, dir configured."""
    from app import config as cfg
    root = tmp_path / 'aitoolkit'
    (root / 'venv' / 'Scripts').mkdir(parents=True)
    (root / 'venv' / 'Scripts' / 'python.exe').write_text('fake')
    (root / 'run.py').write_text('fake')
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(root)}})
    return root


class _FakeProc:
    def __init__(self, pid=4242):
        self.pid = pid

    def wait(self):
        return None


def _mk_ds(app, n_keep=6, trigger='rt_trig', name='Retry', slider=False):
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.services import lora_training as lt
    ds = svc.create_dataset(LOCAL_USER, name, trigger, train_type='zimage')
    for i in range(n_keep):
        svc.db.session.add(FaceDatasetImage(
            dataset_id=ds.id, filename=f'k{i}.webp', status='keep',
            framing='face', caption=f'a nice varied caption number {i}'))
    svc.db.session.commit()
    if slider:
        lt.update_slider_settings(LOCAL_USER, ds.id, {
            'enabled': True, 'positive': 'very muscular body',
            'negative': 'skinny frail body', 'target_class': 'person'})
    return ds


def _register(app, ds, **kw):
    from app.services import checkpoint_registry as cr
    defaults = dict(family='zimage', source='local', base_model='', variant='turbo',
                    masked=True, steps=1000)
    defaults.update(kw)
    return cr.register_launch(LOCAL_USER, ds.id, **defaults)


def _mark_failed(ds, rc=1, log_tail='boom\nRuntimeError: bad allocation'):
    from app.job_queue import queue_manager
    queue_manager._set_system_state('training_in_progress', False, ttl_seconds=60)
    queue_manager._set_system_state(
        'training_error', {'dataset_id': ds.id, 'rc': rc, 'log_tail': log_tail},
        ttl_seconds=60)


# --- failed-run detection --------------------------------------------------------

def test_failed_local_run_detection(app):
    from app.services import lora_training as lt
    from app.job_queue import queue_manager
    with app.app_context():
        ds = _mk_ds(app)
        rec = _register(app, ds)
        queue_manager._set_system_state('training_in_progress', False, ttl_seconds=60)
        assert lt.failed_local_run() is None          # no crash recorded yet
        _mark_failed(ds)
        fid, msg = lt.failed_local_run()
        assert fid == rec.id
        assert 'bad allocation' in msg and 'exit code 1' in msg
        # a live run hides the affordance (its own launch cleared the error state)
        queue_manager._set_system_state('training_in_progress', True, ttl_seconds=60)
        assert lt.failed_local_run() is None


# --- Runs hub wiring -------------------------------------------------------------

def test_all_runs_marks_failed_local_run_only(app):
    """The Runs hub tags the failed local run with status='error' + its record_id
    (the ↻ Retry target) + a crash message; an earlier completed local launch of
    the same dataset stays unmarked, so no Retry button shows on it."""
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _mk_ds(app, trigger='ar_trig', name='AllRuns')
        older = _register(app, ds, steps=1000)     # earlier launch (completed)
        newest = _register(app, ds, steps=1500)    # the launch that crashed
        _mark_failed(ds, rc=1, log_tail='RuntimeError: bad allocation')
        out = ct.all_runs(limit=10)
        local_rows = {r['record_id']: r for r in out['recent'] if r['source'] == 'local'}
        assert local_rows[newest.id]['status'] == 'error'
        assert 'bad allocation' in local_rows[newest.id]['error']
        assert local_rows[older.id].get('status') is None


# --- param replay ----------------------------------------------------------------

def test_retry_local_run_replays_stamped_params(app, monkeypatch):
    """The retry hands launch_training the EXACT identity params stamped for the
    failed launch — family/variant/base/masked/steps."""
    from app.services import lora_training as lt
    with app.app_context():
        ds = _mk_ds(app)
        rec = _register(app, ds, family='krea', base_model='merged/base',
                        variant='turbo', masked=False, steps=1500)
        _mark_failed(ds, rc=3221225477, log_tail='std::bad_alloc')
        captured = {}
        monkeypatch.setattr(
            lt, 'launch_training',
            lambda user_id, dataset_id, **kw: captured.update(
                user_id=user_id, dataset_id=dataset_id, **kw) or {'started': True})
        out = lt.retry_local_run(LOCAL_USER, rec.id)
        assert out == {'started': True}
        assert captured['dataset_id'] == ds.id
        assert captured['train_type'] == 'krea'
        assert captured['variant'] == 'turbo'
        assert captured['base_model'] == 'merged/base'
        assert captured['masked'] is False
        assert captured['steps'] == 1500


def test_retry_local_run_maps_official_base_to_none(app, monkeypatch):
    """A stamped empty base_model is the OFFICIAL base — replayed as None, never
    an empty string that a downstream would treat as a custom path."""
    from app.services import lora_training as lt
    with app.app_context():
        ds = _mk_ds(app)
        rec = _register(app, ds, base_model='')
        _mark_failed(ds)
        captured = {}
        monkeypatch.setattr(
            lt, 'launch_training',
            lambda user_id, dataset_id, **kw: captured.update(**kw) or {'started': True})
        lt.retry_local_run(LOCAL_USER, rec.id)
        assert captured['base_model'] is None


# --- guards ----------------------------------------------------------------------

def test_retry_local_run_refused_when_run_in_progress(app, monkeypatch):
    """A live run yields the exact GPU-collision message (before any preflight),
    and launch_training is never reached."""
    from app.services import lora_training as lt
    from app.job_queue import queue_manager
    with app.app_context():
        ds = _mk_ds(app)
        rec = _register(app, ds)
        _mark_failed(ds)
        queue_manager._set_system_state('training_in_progress', True, ttl_seconds=60)
        queue_manager._set_system_state('training_pid', 4242, ttl_seconds=60)
        monkeypatch.setattr(lt, '_pid_alive', lambda pid: True)
        monkeypatch.setattr(lt, 'launch_training',
                            lambda *a, **k: pytest.fail('must not launch during a live run'))
        with pytest.raises(ValueError, match='already in progress'):
            lt.retry_local_run(LOCAL_USER, rec.id)


def test_retry_local_run_refused_when_not_failed(app, monkeypatch):
    """A completed/never-failed run has no recorded failure — refused cleanly
    (this is the backend mirror of the button only showing on error rows)."""
    from app.services import lora_training as lt
    from app.job_queue import queue_manager
    with app.app_context():
        ds = _mk_ds(app)
        rec = _register(app, ds)
        queue_manager._set_system_state('training_in_progress', False, ttl_seconds=60)
        monkeypatch.setattr(lt, 'launch_training',
                            lambda *a, **k: pytest.fail('must not launch a non-failed run'))
        with pytest.raises(ValueError, match='no recorded failure'):
            lt.retry_local_run(LOCAL_USER, rec.id)


def test_retry_local_run_rejects_unknown_and_cloud_records(app):
    from app.services import lora_training as lt
    with app.app_context():
        with pytest.raises(ValueError, match='unknown training run'):
            lt.retry_local_run(LOCAL_USER, 999999)
        ds = _mk_ds(app)
        crec = _register(app, ds, source='cloud', cloud_run_id=1)
        with pytest.raises(ValueError, match='local run can be retried'):
            lt.retry_local_run(LOCAL_USER, crec.id)


# --- slider replay: end to end (config actually written) -------------------------

def test_retry_local_run_replays_slider_config_with_768(app, tmp_path, monkeypatch):
    """Retrying a failed SLIDER run re-emits the slider recipe faithfully — the
    concept_slider process — and, thanks to the new default, at 768 only (the
    resolution that keeps the slider loss's VRAM peak under 24 GB). This is the
    exact regression Jeremy hit: the first slider run OOM'd in 768+1024."""
    from app.services import lora_training as lt
    _configure_aitoolkit(tmp_path, monkeypatch, app)
    monkeypatch.setattr(lt.shutil, 'disk_usage',
                        lambda p: type('u', (), {'free': 500e9})())
    monkeypatch.setattr(lt, '_aitoolkit_supports_concept_slider', lambda: True)
    monkeypatch.setattr(lt, '_watch_training', lambda *a, **k: None)
    monkeypatch.setattr(lt.subprocess, 'Popen', lambda args, **kw: _FakeProc())

    def fake_export(user_id, dataset_id, masked=True):
        folder = tmp_path / 'exp'
        folder.mkdir(exist_ok=True)
        return str(folder)

    monkeypatch.setattr(lt, 'export_dataset_to_aitoolkit', fake_export)
    with app.app_context():
        ds = _mk_ds(app, slider=True)
        rec = _register(app, ds, family='zimage', variant='turbo', steps=1000)
        _mark_failed(ds, log_tail='RuntimeError: bad allocation')
        out = lt.retry_local_run(LOCAL_USER, rec.id)
        with open(out['config_path'], encoding='utf-8') as fh:
            cfg = json.load(fh)
        proc = cfg['config']['process'][0]
        assert proc['type'] == 'concept_slider'
        assert proc['datasets'][0]['resolution'] == [768]


# --- route surface ---------------------------------------------------------------

def test_retry_route_posts_record_id(app, client, monkeypatch):
    from app import capabilities
    from app.services import lora_training as lt
    monkeypatch.setattr(capabilities, 'probe', lambda: {'aitoolkit': {'valid': True}})
    captured = {}
    monkeypatch.setattr(
        lt, 'retry_local_run',
        lambda user_id, record_id: captured.update(record_id=record_id)
        or {'started': True, 'pid': 7})
    r = client.post('/api/dataset/train/retry', json={'record_id': 55})
    assert r.status_code == 200
    d = r.get_json()
    assert d['ok'] is True and d['pid'] == 7
    assert captured['record_id'] == 55


def test_retry_route_gated_when_aitoolkit_unconfigured(app, client, monkeypatch):
    from app import capabilities
    monkeypatch.setattr(capabilities, 'probe', lambda: {'aitoolkit': {'valid': False}})
    r = client.post('/api/dataset/train/retry', json={'record_id': 1})
    assert r.status_code == 409
