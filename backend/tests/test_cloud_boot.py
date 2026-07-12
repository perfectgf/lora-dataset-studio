"""Boot recovery: orphan reconciliation + resuming the monitor of a run that
was active when the app was closed."""


def test_boot_recover_resumes_active_run(app, monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    from app.services import cloud_training as ct
    from app.extensions import db
    from app.models import CloudTrainingRun
    resumed = []
    monkeypatch.setattr(ct, 'reconcile_orphans', lambda a: 0)
    monkeypatch.setattr(ct, '_start_monitor_for_app',
                        lambda app_, run_id: resumed.append(run_id))
    with app.app_context():
        run = CloudTrainingRun(dataset_id=1, status='training',
                               vast_instance_id='777', vast_label='lds-1',
                               job_name='j', staging_dir='/tmp/x')
        db.session.add(run)
        db.session.commit()
        ct.boot_recover(app)
        assert resumed == [run.id]


def test_boot_recover_fails_instanceless_active_run(app, monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    from app.services import cloud_training as ct
    from app.extensions import db
    from app.models import CloudTrainingRun
    monkeypatch.setattr(ct, 'reconcile_orphans', lambda a: 0)
    with app.app_context():
        run = CloudTrainingRun(dataset_id=1, status='preparing',
                               vast_label='lds-1', job_name='j')
        db.session.add(run)
        db.session.commit()
        ct.boot_recover(app)
        assert CloudTrainingRun.query.get(run.id).status == 'error'


def test_boot_recover_without_key_is_silent(app, monkeypatch):
    monkeypatch.delenv('VAST_API_KEY', raising=False)
    from app.services import cloud_training as ct
    ct.boot_recover(app)          # must not raise
