"""Full monitor loop against a scripted fake pod + fake vast. Covers the
happy path, the stop path, max-runtime kill, unreachable-pod failure, the
download-failure pod-kept state, and the resume contract (Task 7 needs
_monitor to skip re-provisioning/re-submitting an already-running job).
No sleeping: _sleep is a no-op seam."""
import json
import os

import pytest


@pytest.fixture()
def ct(app, monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    from app.services import cloud_training
    monkeypatch.setattr(cloud_training, '_sleep', lambda s: None)
    monkeypatch.setattr(cloud_training, '_start_monitor', lambda *a, **k: None)
    # launch_cloud_training now reconciles orphans on every call (Task 7) --
    # no-op that seam so these monitor tests, which never mock vast_client
    # .list_instances, stay offline.
    monkeypatch.setattr(cloud_training, '_reconcile_before_launch', lambda a: None)
    return cloud_training


class FakeRemote:
    """Scripted ai-toolkit pod: job runs for N polls then completes."""

    def __init__(self, polls_to_complete=3, fail_downloads=False):
        self.polls = 0
        self.n = polls_to_complete
        self.fail_downloads = fail_downloads
        self.stopped = False
        self.uploaded = {}
        self.job_config = None

    def is_ready(self):
        return True

    def get_settings(self):
        return {'TRAINING_FOLDER': '/pod/out', 'DATASETS_FOLDER': '/pod/ds'}

    def ensure_settings(self, hf_token=None):
        return self.get_settings()

    def upload_dataset(self, name, folder):
        self.uploaded[name] = len(os.listdir(folder))
        return self.uploaded[name]

    def create_job(self, name, job_config, gpu_ids='0'):
        self.job_config = job_config
        return 'j-1'

    def start_job(self, job_id, gpu_ids='0'):
        pass

    def stop_job(self, job_id):
        self.stopped = True

    def get_job(self, job_id):
        self.polls += 1
        if self.stopped:
            return {'status': 'stopped', 'step': self.polls * 10, 'total_steps': 100}
        if self.polls >= self.n:
            return {'status': 'completed', 'step': 100, 'total_steps': 100}
        return {'status': 'running', 'step': self.polls * 10, 'total_steps': 100,
                'info': 'Training', 'speed_string': '1 it/s'}

    def get_log(self, job_id):
        return f'step {self.polls * 10}/100'

    def get_samples(self, job_id):
        return ['/pod/out/lds1_run/samples/168__50_0.jpg']

    def download_sample(self, remote_path, dest):
        with open(dest, 'wb') as f:
            f.write(b'IMG')

    def list_files(self, job_id):
        return [{'path': '/pod/out/lds1_run/lds1_run_000000100.safetensors', 'size': 4}]

    def download_public_file(self, remote_path, dest):
        if self.fail_downloads:
            raise RuntimeError('download failed')
        with open(dest, 'wb') as f:
            f.write(b'CKPT')


def _launch(ct, app, client, monkeypatch, remote, destroy_log):
    monkeypatch.setattr(ct.lt, 'export_dataset_to_aitoolkit',
                        lambda uid, did, masked=True, dest_dir=None:
                        (os.makedirs(dest_dir, exist_ok=True),
                         open(os.path.join(dest_dir, '0001.png'), 'wb').close(),
                         dest_dir)[-1])
    monkeypatch.setattr(ct.lt, 'default_steps', lambda ds: 100)
    monkeypatch.setattr(ct.lt, 'assert_trainable', lambda *a, **k: None)
    monkeypatch.setattr(ct.lt, 'build_job_config',
                        lambda ds, folder, steps=3000, training_folder=None: {
                            'job': 'extension',
                            'config': {'name': 'lora_lola', 'process': [{
                                'type': 'sd_trainer',
                                'training_folder': training_folder,
                                'device': 'cuda:0',
                                'datasets': [{'folder_path': folder}],
                            }]}})
    monkeypatch.setattr(ct.lt, 'import_checkpoint',
                        lambda *a, **kw: str(kw.get('src_dir')) + '/imported')
    monkeypatch.setattr(ct.vast_client, 'search_offers',
                        lambda **kw: [{'offer_id': 9, 'gpu_name': 'RTX 4090',
                                       'dph_total': 0.4, 'gpu_ram_gb': 24.0}])
    monkeypatch.setattr(ct.vast_client, 'create_instance', lambda *a, **kw: '777')
    monkeypatch.setattr(ct.vast_client, 'get_instance', lambda iid: {
        'instance_id': iid, 'actual_status': 'running', 'public_ipaddr': '1.2.3.4',
        'ports': {'18675/tcp': [{'HostPort': '40123'}]}, 'label': 'lds-x',
        'jupyter_token': 'jtok-vast'})
    monkeypatch.setattr(ct.vast_client, 'destroy_instance',
                        lambda iid: destroy_log.append(iid) or True)
    monkeypatch.setattr(ct, '_make_remote', lambda run: remote)
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'Lola', 'trigger_word': 'lola'}).get_json()['id']
    with app.app_context():
        res = ct.launch_cloud_training('local', ds_id)
    return ds_id, res['run_id']


def test_happy_path_completes_and_terminates(ct, app, client, monkeypatch):
    destroyed = []
    remote = FakeRemote(polls_to_complete=3)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'done'
        assert destroyed == ['777']
        # dataset + masks names, job config cloudified
        assert remote.job_config['config']['name'] == run.job_name
        proc = remote.job_config['config']['process'][0]
        assert proc['type'] == 'diffusion_trainer'
        assert proc['training_folder'] == '/pod/out'
        assert proc['datasets'][0]['folder_path'] == f'/pod/ds/{run.job_name}'
        # checkpoint downloaded into staging then imported
        assert run.checkpoint_local_path and run.checkpoint_local_path.endswith('.safetensors')
        # template mode: the vast-generated per-instance token was picked up
        # from the instance record during boot-wait
        assert run.auth_token == 'jtok-vast'
        assert os.path.exists(run.checkpoint_local_path)


def test_stop_requested_stops_job_and_terminates(ct, app, client, monkeypatch):
    destroyed = []
    remote = FakeRemote(polls_to_complete=50)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    ct._stop_event.set()
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert remote.stopped is True
        assert run.status == 'stopped'
        assert destroyed == ['777']


def test_max_runtime_cap_kills_pod(ct, app, client, monkeypatch):
    destroyed = []
    remote = FakeRemote(polls_to_complete=10_000)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    clock = {'t': 0.0}

    def fake_time():
        clock['t'] += 3600.0          # each check jumps one hour
        return clock['t']

    monkeypatch.setattr(ct, '_now', fake_time)
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status in ('stopped', 'error')
        assert 'runtime' in (run.error or run.phase_detail or '').lower()
        assert destroyed == ['777']


def test_max_runtime_cap_counts_pre_restart_time(ct, app, client, monkeypatch):
    """A resumed run whose total age already exceeds the cap is stopped
    immediately — restarting the app must not grant a fresh window."""
    from datetime import datetime, timedelta
    destroyed = []
    remote = FakeRemote(polls_to_complete=10_000)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    with app.app_context():
        run = ct.CloudTrainingRun.query.get(run_id)
        ct._set(run, vast_instance_id='777', remote_job_id='j-1', status='training',
                base_url='http://1.2.3.4:40123', auth_token='tok',
                created_at=datetime.utcnow() - timedelta(minutes=500))  # > 240 min cap
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status in ('stopped', 'error')
        assert 'runtime' in ((run.error or '') + (run.phase_detail or '')).lower()
        assert destroyed == ['777']


def test_download_failure_keeps_pod(ct, app, client, monkeypatch):
    destroyed = []
    remote = FakeRemote(polls_to_complete=2, fail_downloads=True)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'error_pod_kept'
        assert destroyed == []                      # pod intentionally kept
        assert run.base_url                          # surfaced for manual recovery


def test_pod_never_ready_fails_and_destroys(ct, app, client, monkeypatch):
    destroyed = []
    remote = FakeRemote()
    remote.is_ready = lambda: False
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    clock = {'t': 0.0}
    monkeypatch.setattr(ct, '_now', lambda: clock.__setitem__('t', clock['t'] + 120) or clock['t'])
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'error'
        assert destroyed == ['777']


def test_cloud_progress_shape_matches_local(ct, app, client, monkeypatch):
    destroyed = []
    remote = FakeRemote(polls_to_complete=3)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    with app.app_context():
        ct._monitor(app, run_id)
        prog = ct.cloud_progress('local', ds_id)
        for key in ('active', 'log_exists', 'step', 'total', 'loss', 'speed',
                    'eta', 'loss_curve', 'samples', 'phase', 'cost_estimate'):
            assert key in prog
        assert prog['active'] is False               # run finished
        assert prog['phase'] == 'done'
        assert isinstance(prog['samples'], list)


def test_stale_ui_port_8675_coerced_in_template_mode(ct, app, client, monkeypatch):
    """A pre-template config.json baked ui_port=8675; the template only
    publishes 18675 — the monitor must coerce or the boot-wait always
    times out (run #3, 2026-07-12)."""
    destroyed = []
    remote = FakeRemote(polls_to_complete=2)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    monkeypatch.setattr(ct.cfg, 'get', (lambda orig: (lambda k, d=None:
        8675 if k == 'cloud.ui_port'
        else ({**(orig('cloud') or {}), 'ui_port': 8675} if k == 'cloud' else orig(k, d))))(ct.cfg.get))
    with app.app_context():
        ct._monitor(app, run_id)
        assert ct.CloudTrainingRun.query.get(run_id).status == 'done'


def test_boot_wait_tolerates_transient_vast_errors(ct, app, client, monkeypatch):
    """A VastError during boot-wait must NOT kill the run — retry until ready.
    READY_TIMEOUT_SECONDS already bounds the wait, so a transient vast.ai API
    hiccup is just 'not ready yet', never a destroyed pod."""
    destroyed = []
    remote = FakeRemote(polls_to_complete=2)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    calls = {'n': 0}
    good = {'instance_id': '777', 'actual_status': 'running',
            'public_ipaddr': '1.2.3.4',
            'ports': {'18675/tcp': [{'HostPort': '40123'}]}, 'label': 'lds-x',
            'jupyter_token': 'jtok-vast'}

    def flaky(iid):
        calls['n'] += 1
        if calls['n'] <= 2:
            raise ct.vast_client.VastError('bundles endpoint 502')
        return good

    monkeypatch.setattr(ct.vast_client, 'get_instance', flaky)
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'done'          # survived the hiccup
        assert destroyed == ['777']          # normal terminate at completion


def test_monitor_resume_skips_upload_and_submit(ct, app, client, monkeypatch):
    """Resume contract (needed by Task 7): if the run already has a
    vast_instance_id + remote_job_id (e.g. the app restarted mid-run), the
    monitor must reattach without re-provisioning a pod or re-submitting the
    job -- it goes straight to polling the existing remote job."""
    destroyed = []
    remote = FakeRemote(polls_to_complete=3)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    with app.app_context():
        run = ct.CloudTrainingRun.query.get(run_id)
        run.vast_instance_id = '777'
        run.remote_job_id = 'j-resumed'
        run.base_url = 'http://1.2.3.4:40123'
        ct.db.session.commit()
    # Any attempt to re-provision proves the resume guard failed.
    monkeypatch.setattr(ct.vast_client, 'search_offers',
                        lambda **kw: (_ for _ in ()).throw(AssertionError('should not re-provision')))
    monkeypatch.setattr(ct.vast_client, 'create_instance',
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError('should not re-provision')))
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'done'
        assert remote.uploaded == {}                 # upload_dataset never called
        assert remote.job_config is None              # create_job never called
        assert destroyed == ['777']
