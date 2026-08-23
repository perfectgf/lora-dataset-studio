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

    def upload_dataset(self, name, folder, on_progress=None):
        # The real driver reports its progress as it goes (a 24 GB dataset is
        # 1 553 POSTs and used to report nothing — run #138). The double calls
        # the callback once so the monitor's wiring is exercised here too.
        self.uploaded[name] = len(os.listdir(folder))
        if on_progress:
            on_progress(self.uploaded[name], self.uploaded[name], 1, 1)
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

    def download_public_file(self, remote_path, dest, timeout=None, **kw):
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
    # Respect the production launch invariant: cloud runs never target fewer
    # than one usable 500-step save.  Keeping the fixture valid also lets the
    # automatic-retry assertion require byte-for-byte-equivalent parameters.
    monkeypatch.setattr(ct.lt, 'default_steps', lambda ds, **kw: 500)
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
                                       'dph_total': 0.4, 'gpu_ram_gb': 24.0,
                                       'machine_id': 43503, 'reliability': 0.99}])
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


def test_done_run_mirrors_checkpoint_into_local_run_dir(ct, app, client, monkeypatch, tmp_path):
    """The cloud result must land in the LOCAL ai-toolkit run dir too, renamed
    to the local convention — the user looked in ai-toolkit/output and found
    it empty (2026-07-13). Local naming makes the panel's checkpoint list,
    Resume-or-Fresh and Continue treat cloud results like local ones."""
    destroyed = []
    remote = FakeRemote(polls_to_complete=3)
    local_run = tmp_path / 'ulocal_lola' / 'lora_lola'
    run_dir_calls = []
    monkeypatch.setattr(
        ct.lt, '_run_dir',
        lambda *a, **k: run_dir_calls.append(k) or str(local_run))
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'done'
        # pod file lds1_run_000000100.safetensors -> local lora_lola_000000100
        assert (local_run / 'lora_lola_000000100.safetensors').is_file()
        assert run_dir_calls[-1]['base_model'] == ''
        assert run_dir_calls[-1]['family'] == 'zimage'
        assert run_dir_calls[-1]['variant'] == 'turbo'


def test_mirror_never_clobbers_an_existing_local_checkpoint(ct, app, client, monkeypatch, tmp_path):
    """Cloud training on a dataset that ALSO trained locally: the mirror's
    dest name can collide with the local run's files — local work must never
    be overwritten (the cloud result stays in staging/ComfyUI/hub)."""
    destroyed = []
    remote = FakeRemote(polls_to_complete=3)
    local_run = tmp_path / 'ulocal_lola' / 'lora_lola'
    local_run.mkdir(parents=True)
    prior = local_run / 'lora_lola_000000100.safetensors'    # same step as pod's
    prior.write_bytes(b'LOCAL-WEIGHTS')
    monkeypatch.setattr(ct.lt, '_run_dir', lambda *a, **k: str(local_run))
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    with app.app_context():
        ct._monitor(app, run_id)
        assert ct.CloudTrainingRun.query.get(run_id).status == 'done'
        assert prior.read_bytes() == b'LOCAL-WEIGHTS'        # untouched


def test_stop_requested_stops_job_and_terminates(ct, app, client, monkeypatch):
    destroyed = []
    remote = FakeRemote(polls_to_complete=50)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    ct._stop_event_for(run_id).set()
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
                created_at=datetime.utcnow() - timedelta(minutes=500))  # > 480 min cap
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
        # a host that cannot DELIVER its result is a bad host — blacklisted
        assert '43503' in ct._load_bad_hosts()


def test_pod_unreachable_mid_run_blacklists_host(ct, app, client, monkeypatch):
    """A pod dying mid-run (live 2026-07-13: a krea run lost at ~$0.93) is a
    host-quality signal: the machine is blacklisted for the next launches."""
    destroyed = []
    remote = FakeRemote(polls_to_complete=10_000)

    def dead_get_job(job_id):
        raise RuntimeError('connection refused')

    remote.get_job = dead_get_job
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    clock = {'t': 0.0}
    monkeypatch.setattr(ct, '_now', lambda: clock.__setitem__('t', clock['t'] + 120) or clock['t'])
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'error'
        assert 'unreachable' in (run.error or '')
        assert destroyed == ['777']                 # leak-safety unchanged
        assert '43503' in ct._load_bad_hosts()
        # A transient pod loss gets exactly one fresh launch.  The retry locks
        # to the GPU that was ACTUALLY rented (not merely the original picker
        # preference), and preserves every persisted training parameter.
        parent_params = json.loads(run.train_params)
        child_id = parent_params['auto_retry_run_id']
        child = ct.CloudTrainingRun.query.get(child_id)
        child_params = json.loads(child.train_params)
        assert child_params['auto_retry_of'] == run.id
        assert child_params['auto_retry_count'] == 1
        assert child_params['requested_gpu'] == 'RTX 4090'
        assert child_params['strict_gpu'] is True
        for key in ('steps', 'variant', 'train_type', 'masked'):
            assert child_params[key] == parent_params[key]

        # Calling the recovery seam again (e.g. after an app restart in the
        # claim-to-child window) finds the same child instead of paying for a
        # second pod.
        before = ct.CloudTrainingRun.query.count()
        again = ct._maybe_auto_retry(run, run.error)
        assert again['run_id'] == child_id
        assert ct.CloudTrainingRun.query.count() == before


def test_slow_mirror_between_polls_not_counted_as_unreachable(ct, app, client, monkeypatch):
    """Root-cause fix for the ~54-min 'pod unreachable' deaths: the grace must
    measure CONSECUTIVE get_job failure time, not time since the last success.
    A per-poll log/sample mirror that blocks for minutes on a degrading proxy
    used to push the last-success anchor far into the past, so the pod was
    declared unreachable on its FIRST failed probe. Here get_job succeeds, a
    slow mirror burns 500 s (> the 360 s default grace), then a SINGLE get_job
    failure occurs before the pod recovers — the run must complete, not die."""
    clock = {'t': 0.0}
    monkeypatch.setattr(ct, '_now', lambda: clock['t'])   # advances only via the remote

    class SlowMirrorRemote(FakeRemote):
        def get_job(self, job_id):
            self.polls += 1
            if self.polls == 2:                           # one transient blip
                raise RuntimeError('read timed out')
            if self.polls >= 4:
                return {'status': 'completed', 'step': 100, 'total_steps': 100}
            return {'status': 'running', 'step': self.polls * 10,
                    'total_steps': 100, 'info': 'Training'}

        def get_log(self, job_id):
            if self.polls == 1:                           # slow mirror after 1st success
                clock['t'] += 500
            return f'step {self.polls * 10}/100'

    destroyed = []
    remote = SlowMirrorRemote(polls_to_complete=4)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'done'                       # blip tolerated
        assert destroyed == ['777']


def test_default_grace_kills_prolonged_blackout(ct, app, client, monkeypatch):
    """The default grace still ends a genuinely unreachable pod: a blackout that
    stays down past the 360 s default is declared unreachable and the pod reaped."""
    clock = {'t': 0.0}
    monkeypatch.setattr(ct, '_now', lambda: clock['t'])

    class BlipRemote(FakeRemote):
        def get_job(self, job_id):
            self.polls += 1
            clock['t'] += 200                             # 200 s of wall time per poll
            if 2 <= self.polls <= 4:                      # 3 consecutive failures (>360 s)
                raise RuntimeError('connection reset')
            if self.polls >= 6:
                return {'status': 'completed', 'step': 100, 'total_steps': 100}
            return {'status': 'running', 'step': self.polls * 10,
                    'total_steps': 100, 'info': 'Training'}

    destroyed = []
    remote = BlipRemote(polls_to_complete=6)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'error'
        assert 'unreachable' in (run.error or '')
        assert destroyed == ['777']


def test_unreachable_grace_minutes_config_extends_tolerance(ct, app, client, monkeypatch):
    """cloud.unreachable_grace_minutes lengthens the tolerated blackout: the SAME
    3-poll blackout that the default (360 s) kills is ridden out when the grace
    is raised to 30 min, and the pod recovers to completion."""
    ct.cfg.save_config({'cloud': {'unreachable_grace_minutes': 30}})
    clock = {'t': 0.0}
    monkeypatch.setattr(ct, '_now', lambda: clock['t'])

    class BlipRemote(FakeRemote):
        def get_job(self, job_id):
            self.polls += 1
            clock['t'] += 200
            if 2 <= self.polls <= 4:
                raise RuntimeError('connection reset')
            if self.polls >= 6:
                return {'status': 'completed', 'step': 100, 'total_steps': 100}
            return {'status': 'running', 'step': self.polls * 10,
                    'total_steps': 100, 'info': 'Training'}

    destroyed = []
    remote = BlipRemote(polls_to_complete=6)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'done'                       # blackout ridden out
        assert destroyed == ['777']


def test_auto_retry_classifier_only_accepts_transient_pod_failures(ct):
    assert ct._is_retryable_pod_failure(
        "('Connection aborted.', ConnectionResetError(10054, 'closed'))")
    assert ct._is_retryable_pod_failure('pod did not become ready in time')
    assert ct._is_retryable_pod_failure('pod unreachable: connection refused')
    assert not ct._is_retryable_pod_failure('CUDA out of memory')
    assert not ct._is_retryable_pod_failure('stall watchdog')


def test_transient_create_error_classifier(ct):
    assert ct._is_transient_create_error('create_instance failed: HTTP 400 {}')
    assert ct._is_transient_create_error('create_instance failed: HTTP 429 {}')
    assert ct._is_transient_create_error('create_instance failed: HTTP 503 {}')
    assert not ct._is_transient_create_error('create_instance failed: HTTP 403 {}')
    assert not ct._is_transient_create_error('create_instance failed: HTTP 404 {}')
    assert not ct._is_transient_create_error('boom')


_TWO_OFFERS = [
    {'offer_id': 9, 'gpu_name': 'RTX 4090', 'dph_total': 0.40,
     'gpu_ram_gb': 24.0, 'machine_id': 43503, 'reliability': 0.99},
    {'offer_id': 11, 'gpu_name': 'RTX 4090', 'dph_total': 0.42,
     'gpu_ram_gb': 24.0, 'machine_id': 43600, 'reliability': 0.99},
]


def test_provision_retries_transient_create_refusal(ct, app, client, monkeypatch):
    """A transient create refusal (offer just taken -> HTTP 400 {}, run #80) no
    longer fails the run: _provision re-searches, excludes the tried offer, and
    rents a FRESH one."""
    destroyed = []
    remote = FakeRemote(polls_to_complete=3)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    monkeypatch.setattr(ct.vast_client, 'search_offers', lambda **kw: list(_TWO_OFFERS))
    tried = []

    def flaky_create(offer_id, **kw):
        tried.append(offer_id)
        if len(tried) == 1:
            raise ct.vast_client.VastError('create_instance failed: HTTP 400 {}')
        return '777'

    monkeypatch.setattr(ct.vast_client, 'create_instance', flaky_create)
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'done'
        assert len(tried) == 2                # first refused, second succeeded
        assert tried[0] != tried[1]           # a fresh offer on retry
        assert destroyed == ['777']


def test_provision_only_looks_at_machines_with_the_disk_it_will_claim(
        ct, app, client, monkeypatch):
    """vast refuses an ask whose `disk` exceeds the offer's free space, and the
    refusal is a bare HTTP 400 — the failure that cost the cloud quantization
    lane its first run. Filtering for LESS than what will be asked for is the
    bug; the two numbers are pinned equal here."""
    destroyed = []
    remote = FakeRemote(polls_to_complete=3)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    seen, created = {}, {}

    def search(**kw):
        seen.update(kw)
        return list(_TWO_OFFERS)

    def create(offer_id, **kw):
        created.update(kw)
        return '777'

    monkeypatch.setattr(ct.vast_client, 'search_offers', search)
    monkeypatch.setattr(ct.vast_client, 'create_instance', create)
    with app.app_context():
        ct._monitor(app, run_id)
    assert created['disk_gb'] > 0
    assert seen['min_disk_gb'] == created['disk_gb']


def test_provision_does_not_retry_non_transient_create_error(ct, app, client, monkeypatch):
    """An auth-class refusal (HTTP 403) is not worth retrying — fail fast, once."""
    destroyed = []
    remote = FakeRemote(polls_to_complete=3)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    monkeypatch.setattr(ct.vast_client, 'search_offers', lambda **kw: list(_TWO_OFFERS))
    calls = {'n': 0}

    def create(offer_id, **kw):
        calls['n'] += 1
        raise ct.vast_client.VastError('create_instance failed: HTTP 403 {}')

    monkeypatch.setattr(ct.vast_client, 'create_instance', create)
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'error'
        assert '403' in (run.error or '')
        assert calls['n'] == 1                # no retry on a non-transient refusal
        assert destroyed == []                # no pod was ever created


def test_provision_transient_create_gives_up_after_bounded_attempts(ct, app, client, monkeypatch):
    """A persistently-refusing marketplace is bounded: exactly
    _CREATE_INSTANCE_ATTEMPTS create tries, then the run errors (not an
    unbounded rent loop)."""
    destroyed = []
    remote = FakeRemote(polls_to_complete=3)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    three_offers = _TWO_OFFERS + [
        {'offer_id': 13, 'gpu_name': 'RTX 4090', 'dph_total': 0.44,
         'gpu_ram_gb': 24.0, 'machine_id': 43700, 'reliability': 0.99}]
    monkeypatch.setattr(ct.vast_client, 'search_offers', lambda **kw: list(three_offers))
    calls = {'n': 0}

    def create(offer_id, **kw):
        calls['n'] += 1
        raise ct.vast_client.VastError('create_instance failed: HTTP 429 {}')

    monkeypatch.setattr(ct.vast_client, 'create_instance', create)
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'error'
        assert '429' in (run.error or '')
        assert calls['n'] == ct._CREATE_INSTANCE_ATTEMPTS
        assert destroyed == []


def test_auto_retry_preserves_resume_and_uses_effective_gpu(ct, app, client,
                                                            monkeypatch, tmp_path):
    """The paid retry is the same run recipe, including a cloud continuation
    seed, and strict-locks to the effective GPU recorded on the failed pod."""
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'Retry recipe', 'trigger_word': 'retry'}).get_json()['id']
    seed = tmp_path / 'resume.safetensors'
    seed.write_bytes(b'weights')
    captured = {}
    with app.app_context():
        run = ct.CloudTrainingRun(
            dataset_id=ds_id, status='error', run_name='retry-source',
            vast_instance_id='old-pod', gpu_name='RTX 5090',
            train_params=json.dumps({
                'steps': 4100, 'variant': 'deturbo', 'train_type': 'zimage',
                'recipe_version': ct.lt.ZIMAGE_RECIPE_VERSION,
                'effective_base': ct.lt.ZIMAGE_DETURBO_BASE,
                'training_adapter': None,
                'masked': False, 'requested_gpu': 'RTX 4090',
                'resume_ckpt_path': str(seed), 'resume_step': 2500,
            }))
        ct.db.session.add(run)
        ct.db.session.commit()
        monkeypatch.setattr(
            ct, 'launch_cloud_training',
            lambda user_id, dataset_id, **kw:
            (captured.update(user_id=user_id, dataset_id=dataset_id, **kw),
             {'run_id': 99, 'status': 'preparing'})[1])

        result = ct._maybe_auto_retry(run, 'connection reset by peer')

    assert result['run_id'] == 99
    assert captured['dataset_id'] == ds_id
    assert captured['steps'] == 4100
    assert captured['variant'] == 'deturbo'
    assert captured['train_type'] == 'zimage'
    assert captured['masked'] is False
    assert captured['resume_ckpt_path'] == str(seed)
    assert captured['resume_step'] == 2500
    assert captured['gpu_name'] == 'RTX 5090'       # effective, not requested
    assert captured['strict_gpu'] is True
    assert captured['auto_retry_count'] == 1
    assert captured['auto_retry_of'] == run.id


def test_auto_retry_resume_blocks_legacy_lokr_without_full_rank(
        ct, app, client, monkeypatch):
    """The automatic retry must not rent a pod for an ambiguous LoKr seed."""
    from app.models import TrainingRunRecord
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'Legacy auto retry', 'trigger_word': 'legacyauto'}).get_json()['id']
    legacy = {'rank': 64, 'alpha': 32, 'network_type': 'lokr', 'lokr_factor': 16}
    launched = []
    monkeypatch.setattr(ct, 'launch_cloud_training',
                        lambda *a, **k: launched.append(k))
    with app.app_context():
        run = ct.CloudTrainingRun(
            dataset_id=ds_id, status='error', run_name='legacy-auto',
            vast_instance_id='old-pod', train_params=json.dumps({
                'steps': 1500, 'variant': 'base', 'train_type': 'krea',
                'resume_ckpt_path': 'C:/staging/legacy.safetensors',
                'resume_step': 1000,
                'train_settings_snapshot': json.dumps(legacy),
            }))
        ct.db.session.add(run)
        ct.db.session.commit()
        parent = TrainingRunRecord(
            dataset_id=ds_id, family='krea', source='cloud', base_model='',
            variant='base', steps=1000, version=1, fingerprint='parent', manifest='[]',
            settings=json.dumps(legacy))
        ct.db.session.add(parent)
        ct.db.session.commit()
        child = TrainingRunRecord(
            dataset_id=ds_id, family='krea', source='cloud', base_model='',
            variant='base', steps=1500, version=1, fingerprint='child', manifest='[]',
            cloud_run_id=run.id, parent_record_id=parent.id,
            settings=json.dumps({'network_type': 'lokr', 'lokr_full_rank': False}))
        ct.db.session.add(child)
        ct.db.session.commit()
        before = run.train_params
        assert ct._maybe_auto_retry(run, 'connection reset by peer') is None
        assert run.train_params == before
    assert launched == []


def test_auto_retry_resume_uses_recorded_lokr_parent_topology(
        ct, app, client, monkeypatch):
    """A modern parent record supplies the frozen topology for an auto retry."""
    from app.models import TrainingRunRecord
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'Known auto retry', 'trigger_word': 'knownauto'}).get_json()['id']
    topology = {
        'rank': 64, 'alpha': 32, 'network_type': 'lokr',
        'lokr_factor': 'auto', 'lokr_full_rank': False,
    }
    captured = {}
    monkeypatch.setattr(ct, 'launch_cloud_training',
                        lambda user_id, dataset_id, **kw:
                        (captured.update(**kw), {'run_id': 77, 'status': 'preparing'})[1])
    with app.app_context():
        run = ct.CloudTrainingRun(
            dataset_id=ds_id, status='error', run_name='known-auto',
            vast_instance_id='old-pod', train_params=json.dumps({
                'steps': 1500, 'variant': 'base', 'train_type': 'krea',
                'resume_ckpt_path': 'C:/staging/known.safetensors',
                'resume_step': 1000,
                'train_settings_snapshot': json.dumps({
                    'rank': 64, 'alpha': 32, 'network_type': 'lokr',
                }),
            }))
        ct.db.session.add(run)
        ct.db.session.commit()
        parent = TrainingRunRecord(
            dataset_id=ds_id, family='krea', source='cloud', base_model='',
            variant='base', steps=1000, version=1, fingerprint='parent', manifest='[]',
            settings=json.dumps(topology))
        ct.db.session.add(parent)
        ct.db.session.commit()
        child = TrainingRunRecord(
            dataset_id=ds_id, family='krea', source='cloud', base_model='',
            variant='base', steps=1500, version=1, fingerprint='child', manifest='[]',
            cloud_run_id=run.id, parent_record_id=parent.id,
            settings=json.dumps(topology))
        ct.db.session.add(child)
        ct.db.session.commit()
        result = ct._maybe_auto_retry(run, 'connection reset by peer')
    assert result['run_id'] == 77
    snapshot = json.loads(captured['train_settings_snapshot'])
    assert {key: snapshot[key] for key in topology} == topology


def test_auto_retry_requires_confirmed_old_pod_termination(ct, app, client,
                                                            monkeypatch):
    """Never rent the replacement while Vast has not confirmed that the old
    billed instance is gone: two overlapping pods would double-charge."""
    destroyed = []
    remote = FakeRemote()
    remote.is_ready = lambda: False
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    monkeypatch.setattr(ct.vast_client, 'destroy_instance',
                        lambda iid: destroyed.append(iid) or False)
    clock = {'t': 0.0}
    monkeypatch.setattr(ct, '_now',
                        lambda: clock.__setitem__('t', clock['t'] + 120) or clock['t'])
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'error'
        assert destroyed == ['777']
        assert 'termination was not confirmed' in run.phase_detail
        assert ct.CloudTrainingRun.query.count() == 1
        assert 'auto_retry_run_id' not in json.loads(run.train_params)


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
        # the host that burned the whole boot budget is blacklisted for the
        # next launches (machine_id stamped by _provision from the offer)
        assert '43503' in ct._load_bad_hosts()


def test_stop_during_boot_wait_terminates_immediately(ct, app, client, monkeypatch):
    """"Stop run" while the pod is still booting (dead host stuck in 'loading',
    observed live 2026-07-13) must terminate the pod on the next poll — NOT spin
    silently until the boot timeout. The clock never advances here, so only the
    stop check (never the timeout) can end the loop."""
    destroyed = []
    remote = FakeRemote()
    remote.is_ready = lambda: False               # pod never becomes ready
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    monkeypatch.setattr(ct, '_now', lambda: 0.0)  # frozen: boot timeout can't fire
    ct._stop_event_for(run_id).set()
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'stopped'
        assert 'boot' in (run.phase_detail or '')
        assert destroyed == ['777']
        # an EARLY stop (frozen clock -> 0 min elapsed) says nothing about the
        # host: it must NOT be blacklisted (only stops past 8 min are)
        assert ct._load_bad_hosts() == {}


def test_stop_during_dataset_wait_lands_as_stopped_not_error(ct, app, client, monkeypatch):
    """A user Stop while _prepare_staging is retrying a busy export lease (a
    sibling run's export can take minutes) must land the run as 'stopped' —
    the same treatment the boot-wait's Stop check gets — not fall through to
    the monitor's generic error handler as a red 'Run failed'."""
    destroyed = []
    remote = FakeRemote()
    # The launch above must go through normally (real begin_exclusive) so the
    # run row exists; only from _monitor's _prepare_staging onward do we
    # simulate a sibling run holding the export lease.
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)

    from app.services import dataset_activity
    monkeypatch.setattr(dataset_activity, 'begin_exclusive', lambda *a, **kw: None)

    def sleep_then_stop(s):
        ct._stop_event_for(run_id).set()

    monkeypatch.setattr(ct, '_wait_sleep', sleep_then_stop)

    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'stopped'
        assert 'dataset' in (run.phase_detail or '').lower()
        # the stop event is cleared, exactly like the boot-wait's Stop path
        assert not ct._stop_event_for(run_id).is_set()
        # no pod was rented yet — nothing to destroy
        assert destroyed == []


def test_stop_survives_a_wait_past_the_stop_handoff_window(ct, app, client, monkeypatch):
    """Regression: on_wait used to fire ONCE, before the first sleep. With
    _prepare_staging's wait bounded at wait_seconds=3600, a single write left
    run.updated_at stale past STOP_HANDOFF_SECONDS (120 s) long before an
    hour-long sibling export finished — so _monitor_is_responsive read False
    and a Stop pressed then would have force-terminated a pod that was never
    even rented, with a message blaming a monitor that was alive and simply
    waiting (see _stop_one).

    _prepare_staging stamps phase_detail (hence updated_at) once BEFORE the
    wait even starts, so backdating updated_at only ONCE up front (and
    checking only the final status) would pass even without the heartbeat —
    that write alone keeps the run 'responsive' through a short test. Instead
    this backdates updated_at EVERY time the wait goes on to sleep — undoing
    whatever the loop's own on_wait tick just wrote — and checks
    _monitor_is_responsive at the START of the NEXT call, i.e. right after
    that iteration's tick ran and before this callback stales it again: with
    the fix, every check after the first sees a monitor the Stop button would
    trust; with the old fire-once on_wait, only the first would.

    The 30 s heartbeat throttle is real, so the wait clock is faked to jump
    well past it (and still nowhere near the 3600 s deadline) on every
    read — otherwise this near-instant test would never cross 30 s of
    simulated wait time and the tick would always skip."""
    import threading
    from datetime import datetime, timedelta
    destroyed = []
    remote = FakeRemote()
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)

    from app.services import dataset_activity
    monkeypatch.setattr(dataset_activity, 'begin_exclusive', lambda *a, **kw: None)

    fake_clock = [0.0]

    def fake_wait_clock():
        fake_clock[0] += 40.0
        return fake_clock[0]

    monkeypatch.setattr(ct, '_wait_clock', fake_wait_clock)

    # _monitor_is_responsive also requires a registered, live monitor thread
    # (see test_monitor_responsiveness_still_reads_updated_at) — this test
    # runs _monitor synchronously in-process, so stand one in for real.
    class _Alive(threading.Thread):
        def is_alive(self):
            return True

    ct._monitor_threads[int(run_id)] = _Alive()
    try:
        calls = {'n': 0}
        responsive_checks = []

        def sleep_then_stop(s):
            calls['n'] += 1
            with app.app_context():
                run = ct.CloudTrainingRun.query.get(run_id)
                # Recorded BEFORE backdating: this reflects whether the
                # on_wait tick that ran just before THIS sleep (inside the
                # real wait loop) refreshed updated_at since the LAST time
                # this callback staled it, below.
                responsive_checks.append(ct._monitor_is_responsive(run))
                run.updated_at = (datetime.utcnow()
                                  - timedelta(seconds=ct.STOP_HANDOFF_SECONDS + 80))
                ct.db.session.commit()
            if calls['n'] >= 3:
                ct._stop_event_for(run_id).set()

        monkeypatch.setattr(ct, '_wait_sleep', sleep_then_stop)

        with app.app_context():
            ct._monitor(app, run_id)
            run = ct.CloudTrainingRun.query.get(run_id)
            assert run.status == 'stopped'
            assert 'dataset' in (run.phase_detail or '').lower()
            assert not ct._stop_event_for(run_id).is_set()
            assert destroyed == []
        assert calls['n'] == 3
        # call 1 is not meaningful (nothing was backdated yet); calls 2 and 3
        # each ran after a PRIOR backdate, and prove the heartbeat refreshed
        # updated_at since then — the actual regression this test guards.
        assert responsive_checks[1:] == [True, True]
    finally:
        ct._monitor_threads.pop(int(run_id), None)


def test_midrun_checkpoint_sync_harvests_every_save(ct, app, client, monkeypatch):
    """Saves are mirrored locally DURING the run (user-observed gap 2026-07-13:
    step 1400 reached, step-1250 save existed on the pod, nothing local — a
    dead host would have lost everything) and EVERY synced epoch is kept (the
    pod prunes its own saves; harvesting each as it appears is the only way to
    collect the full history — user ask)."""
    monkeypatch.setattr(ct, '_CKPT_SYNC_EVERY_POLLS', 1)
    downloads = []

    class EvolvingRemote(FakeRemote):
        def list_files(self, job_id):
            step = min(self.polls, 3) * 100
            if step == 0:
                return []
            return [{'path': f'/pod/out/j/j_{step:09d}.safetensors', 'size': 4}]

        def download_public_file(self, remote_path, dest, timeout=None, **kw):
            downloads.append(os.path.basename(remote_path))
            super().download_public_file(remote_path, dest)

    destroyed = []
    remote = EvolvingRemote(polls_to_complete=3)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'done'
        # intermediate saves were pulled during the run, not only at the end
        assert downloads == ['j_000000100.safetensors', 'j_000000200.safetensors',
                             'j_000000300.safetensors']
        # checkpoint_local_path tracks the NEWEST; every epoch stays on disk
        assert run.checkpoint_local_path.endswith('j_000000300.safetensors')
        # Saves land in the DURABLE store, not in the disposable staging dir:
        # the cleanup used to be the only copy of a never-deployed checkpoint.
        files = os.listdir(ct.checkpoint_store_dir(run))
        assert sorted(f for f in files if f.endswith('.safetensors')) == \
            ['j_000000100.safetensors', 'j_000000200.safetensors',
             'j_000000300.safetensors']
        assert not any(f.endswith('.part') for f in files)
        assert not any(f.endswith('.safetensors')
                       for f in os.listdir(run.staging_dir))
        # ...and the panel lists all of them
        assert [c['step'] for c in ct.cloud_checkpoints(ds_id)] == [100, 200, 300]


def test_completion_retrieves_intermediates_still_on_pod(ct, app, client, monkeypatch, tmp_path):
    """A pod whose mid-run sync failed (unservable host) still delivers its
    remaining saves at COMPLETION: after the strict final download, every
    other listed .safetensors is fetched best-effort, and the local mirror
    carries them all (local parity — pick any epoch)."""
    class MultiSaveRemote(FakeRemote):
        def list_files(self, job_id):
            return [{'path': '/pod/out/j/j_000000050.safetensors', 'size': 4},
                    {'path': '/pod/out/j/j_000000100.safetensors', 'size': 4}]

    destroyed = []
    remote = MultiSaveRemote(polls_to_complete=2)
    local_run = tmp_path / 'ulocal_lola' / 'lora_lola'
    monkeypatch.setattr(ct.lt, '_run_dir', lambda *a, **k: str(local_run))
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'done'
        staged = sorted(f for f in os.listdir(ct.checkpoint_store_dir(run))
                        if f.endswith('.safetensors'))
        assert staged == ['j_000000050.safetensors', 'j_000000100.safetensors']
        # every epoch mirrored into the local run dir under local naming
        assert (local_run / 'lora_lola_000000050.safetensors').is_file()
        assert (local_run / 'lora_lola_000000100.safetensors').is_file()


class _BoomRemote:
    def list_files(self, job_id):
        raise RuntimeError('pod gone')


def test_truncated_download_is_rejected_not_registered(ct, app, tmp_path):
    """A pod closing the stream early with a clean EOF (observed live
    2026-07-13) yields a short file that LOOKS complete — the size check
    against list_files must delete it and fail the fetch, never register
    4 bytes of a 100-byte LoRA as the official checkpoint."""
    class TruncatingRemote:
        def list_files(self, job_id):
            return [{'path': '/pod/out/j/j_000000100.safetensors', 'size': 100}]

        def download_public_file(self, remote_path, dest, timeout=None, **kw):
            with open(dest, 'wb') as f:
                f.write(b'CKPT')                      # 4 of 100 bytes

    with app.app_context():
        run = ct.CloudTrainingRun(dataset_id=1, status='training', job_name='j',
                                  vast_label='lds-1', staging_dir=str(tmp_path),
                                  remote_job_id='j-1')
        ct.db.session.add(run)
        ct.db.session.commit()
        remote = TruncatingRemote()
        ct._sync_latest_checkpoint(run, remote)
        assert run.checkpoint_local_path is None      # nothing registered
        assert not any(f.endswith('.safetensors')
                       for f in os.listdir(tmp_path))  # garbage deleted
        # the strict completion path refuses it too
        assert ct._try_download_checkpoint(run, remote) is False
        ct._sync_state.pop(run.id, None)


def test_midrun_sync_gives_up_after_repeated_failures_resets_on_new_save(ct, app, tmp_path):
    """Some pods cannot serve big files WHILE training (live 2026-07-13:
    streams died after a few chunks on 2 of 3 pods). The sync must stop
    retrying a save after 3 failures — not hammer it every 2 min for hours —
    and try again when a NEWER save appears."""
    calls = {'dl': 0}

    class FlakyRemote:
        path = '/pod/out/j/j_000000100.safetensors'

        def list_files(self, job_id):
            return [{'path': self.path, 'size': 4}]

        def download_public_file(self, remote_path, dest, timeout=None, **kw):
            calls['dl'] += 1
            raise RuntimeError('stream died')

    with app.app_context():
        run = ct.CloudTrainingRun(dataset_id=1, status='training', job_name='j',
                                  vast_label='lds-1', staging_dir=str(tmp_path),
                                  remote_job_id='j-1')
        ct.db.session.add(run)
        ct.db.session.commit()
        remote = FlakyRemote()
        for _ in range(6):
            ct._sync_latest_checkpoint(run, remote)
        assert calls['dl'] == 3                     # gave up after 3 attempts
        remote.path = '/pod/out/j/j_000000200.safetensors'
        ct._sync_latest_checkpoint(run, remote)
        assert calls['dl'] == 4                     # newer save -> retried
        assert run.checkpoint_local_path is None    # still nothing usable
        ct._sync_state.pop(run.id, None)            # don't leak into other tests


def test_rescue_download_accepts_synced_save_completion_stays_strict(ct, app, tmp_path):
    """When the pod can't serve files anymore, a mid-run synced save counts as
    success on RESCUE paths (stop/stall/cap) — but the COMPLETION path stays
    strict: an older save must not silently replace the final steps."""
    with app.app_context():
        ckpt = tmp_path / 'j_000000100.safetensors'
        ckpt.write_bytes(b'CKPT')
        run = ct.CloudTrainingRun(dataset_id=1, status='training', job_name='j',
                                  vast_label='lds-1', staging_dir=str(tmp_path),
                                  remote_job_id='j-1',
                                  checkpoint_local_path=str(ckpt))
        ct.db.session.add(run)
        ct.db.session.commit()
        assert ct._try_download_checkpoint(run, _BoomRemote(), allow_stale=True) is True
        assert ct._try_download_checkpoint(run, _BoomRemote()) is False


def test_set_survives_a_transient_sqlite_write_lock(ct, app, monkeypatch):
    """A monitor state write that loses the SQLite write lock retries.

    Regression: two runs created their job on the pod, then died committing
    status='training' with 'database is locked' three minutes in, and sat
    abandoned for an hour of paid GPU time before an app restart resubmitted
    the job and hit the pod's 409 on the duplicate name."""
    from sqlalchemy.exc import OperationalError

    monkeypatch.setattr(ct, '_sleep', lambda s: None)
    with app.app_context():
        run = ct.CloudTrainingRun(dataset_id=1, status='provisioning',
                                  job_name='j', vast_label='lds-1')
        ct.db.session.add(run)
        ct.db.session.commit()

        real_commit = ct.db.session.commit
        calls = {'n': 0}

        def flaky_commit():
            calls['n'] += 1
            if calls['n'] <= 2:
                raise OperationalError('UPDATE cloud_training_run ...', {},
                                       Exception('database is locked'))
            return real_commit()

        monkeypatch.setattr(ct.db.session, 'commit', flaky_commit)
        ct._set(run, status='training', phase_detail='Job queued on the pod',
                remote_job_id='j-1')
        monkeypatch.undo()

        fresh = ct.CloudTrainingRun.query.get(run.id)
        assert calls['n'] == 3                    # two refusals, then the write
        assert fresh.status == 'training'         # fields survived the rollback
        assert fresh.remote_job_id == 'j-1'
        assert fresh.phase_detail == 'Job queued on the pod'


def test_set_survives_a_write_lock_that_really_poisons_the_session(ct, app, monkeypatch):
    """The retry must survive a lock raised by the REAL flush, not a stubbed commit.

    Regression 2026-07-28, run #121: stubbing `db.session.commit` (as the test
    above does) raises without ever putting the session into pending-rollback,
    so the failure path could read `run.id` and the retry looked healthy. In
    production the lock comes from the flush: the session is poisoned, the
    instance's attributes were expired by the previous commit, and formatting
    the retry's own log line fired a lazy load that raised PendingRollbackError
    out of _set. The monitor thread died, the run stayed 'training' with no
    error, and a rented RTX 5090 kept billing for ~36 min until a restart.

    Refusing the UPDATE at the cursor reproduces that faithfully: the failure
    goes through SQLAlchemy's real flush machinery.
    """
    import sqlite3

    from sqlalchemy import event

    monkeypatch.setattr(ct, '_sleep', lambda s: None)
    with app.app_context():
        run = ct.CloudTrainingRun(dataset_id=1, status='provisioning',
                                  job_name='j', vast_label='lds-1')
        ct.db.session.add(run)
        ct.db.session.commit()        # expires every attribute, as in production
        run_id = run.id
        ct.db.session.expire(run)     # nothing left in __dict__ but the identity

        engine = ct.db.session.get_bind()
        refused = {'n': 0}

        @event.listens_for(engine, 'before_cursor_execute')
        def _refuse_the_first_updates(conn, cursor, statement, parameters,
                                      context, executemany):
            if statement.lstrip().upper().startswith('UPDATE') and refused['n'] < 2:
                refused['n'] += 1
                raise sqlite3.OperationalError('database is locked')

        try:
            ct._set(run, status='training', phase_detail='Job queued on the pod',
                    remote_job_id='j-1')
        finally:
            event.remove(engine, 'before_cursor_execute', _refuse_the_first_updates)

        assert refused['n'] == 2                  # two real refusals, then the write
        fresh = ct.db.session.get(ct.CloudTrainingRun, run_id)
        assert fresh.status == 'training'         # fields survived the rollbacks
        assert fresh.remote_job_id == 'j-1'
        assert fresh.phase_detail == 'Job queued on the pod'


def test_set_leaves_a_usable_session_when_it_gives_up(ct, app, monkeypatch):
    """A _set that raises must hand back a session its caller can still write on.

    _monitor's failure path answers a raising _set by calling _finish() to mark
    the run 'error' and destroy the pod. On 2026-07-28 that recovery inherited a
    session in pending-rollback and failed too, which is why run #121 was left
    'training' with error=None and a live pod instead of being closed.
    """
    import sqlite3

    from sqlalchemy import event

    monkeypatch.setattr(ct, '_sleep', lambda s: None)
    with app.app_context():
        run = ct.CloudTrainingRun(dataset_id=1, status='training',
                                  job_name='j', vast_label='lds-1')
        ct.db.session.add(run)
        ct.db.session.commit()
        run_id = run.id

        engine = ct.db.session.get_bind()
        refused = {'n': 0}

        @event.listens_for(engine, 'before_cursor_execute')
        def _refuse_every_update(conn, cursor, statement, parameters,
                                 context, executemany):
            if statement.lstrip().upper().startswith('UPDATE'):
                refused['n'] += 1
                raise sqlite3.OperationalError('database is locked')

        try:
            with pytest.raises(Exception):
                ct._set(run, phase_detail='running: fetching transformer weights')
        finally:
            event.remove(engine, 'before_cursor_execute', _refuse_every_update)

        assert refused['n'] == ct._COMMIT_RETRIES     # it really exhausted the budget
        # The lock is gone; the caller's recovery write must now go through.
        ct._set(run, status='error', error='pod unreachable')
        fresh = ct.db.session.get(ct.CloudTrainingRun, run_id)
        assert fresh.status == 'error'
        assert fresh.error == 'pod unreachable'


def test_progress_heartbeat_survives_a_lock_that_outlives_the_retries(
        ct, app, monkeypatch):
    """A cosmetic progress write must never end a healthy paid run.

    Regression 2026-08-01, run #137: an unrelated long write transaction held
    the SQLite write lock for over an hour. The monitor's per-poll heartbeat --
    which only refreshes phase_detail text -- exhausted its retry budget, raised
    out of _monitor, and a run that was training normally on a rented 5090 was
    recorded status='error' with error='database is locked'.
    """
    import sqlite3

    from sqlalchemy import event

    monkeypatch.setattr(ct, '_sleep', lambda s: None)
    with app.app_context():
        run = ct.CloudTrainingRun(dataset_id=1, status='training',
                                  job_name='j', vast_label='lds-1',
                                  phase_detail='running: Training')
        ct.db.session.add(run)
        ct.db.session.commit()
        run_id = run.id

        engine = ct.db.session.get_bind()
        refused = {'n': 0}

        @event.listens_for(engine, 'before_cursor_execute')
        def _refuse_every_update(conn, cursor, statement, parameters,
                                 context, executemany):
            if statement.lstrip().upper().startswith('UPDATE'):
                refused['n'] += 1
                raise sqlite3.OperationalError('database is locked')

        try:
            landed = ct._set_soft(run, phase_detail='running: step 1164')
        finally:
            event.remove(engine, 'before_cursor_execute', _refuse_every_update)

        assert landed is False                       # the miss is reported...
        assert refused['n'] == ct._COMMIT_RETRIES    # ...after a real budget
        # The run is untouched: still training, never marked failed.
        fresh = ct.db.session.get(ct.CloudTrainingRun, run_id)
        assert fresh.status == 'training'
        assert fresh.error is None
        # And the session is still usable for the next poll's write.
        ct._set(run, phase_detail='running: step 1200')
        assert ct.db.session.get(
            ct.CloudTrainingRun, run_id).phase_detail == 'running: step 1200'


def test_progress_heartbeat_still_raises_on_a_real_database_error(ct, app,
                                                                  monkeypatch):
    """Only the lock is tolerated — a genuine schema/IO fault stays loud."""
    from sqlalchemy.exc import OperationalError

    monkeypatch.setattr(ct, '_sleep', lambda s: None)
    with app.app_context():
        run = ct.CloudTrainingRun(dataset_id=1, status='training',
                                  job_name='j', vast_label='lds-1')
        ct.db.session.add(run)
        ct.db.session.commit()

        def broken_commit():
            raise OperationalError('UPDATE cloud_training_run ...', {},
                                   Exception('no such column: nope'))

        monkeypatch.setattr(ct.db.session, 'commit', broken_commit)
        with pytest.raises(OperationalError):
            ct._set_soft(run, phase_detail='running: step 1164')


def test_monitor_uses_the_soft_write_for_its_progress_heartbeat(ct):
    """The heartbeat seam must stay wired, or the regression returns silently.

    The polling loop moved out of _monitor into _poll_job_until_terminal
    (2026-08-23, verbatim extraction) — the seam moved with it."""
    import inspect

    source = inspect.getsource(ct._poll_job_until_terminal)
    assert '_set_soft(run, phase_detail=f"{status}: {info}"[:500])' in source


def test_set_still_raises_on_a_non_lock_database_error(ct, app, monkeypatch):
    """Only the transient lock is retried — a real failure must stay loud."""
    from sqlalchemy.exc import OperationalError

    monkeypatch.setattr(ct, '_sleep', lambda s: None)
    with app.app_context():
        run = ct.CloudTrainingRun(dataset_id=1, status='provisioning',
                                  job_name='j', vast_label='lds-1')
        ct.db.session.add(run)
        ct.db.session.commit()

        calls = {'n': 0}

        def broken_commit():
            calls['n'] += 1
            raise OperationalError('UPDATE cloud_training_run ...', {},
                                   Exception('no such column: nope'))

        monkeypatch.setattr(ct.db.session, 'commit', broken_commit)
        with pytest.raises(OperationalError):
            ct._set(run, status='training')
        assert calls['n'] == 1                    # no retry storm


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


# ── The create/record window: the pod holds the job, our row does not ────────
# create_job succeeded, then the app died before remote_job_id reached the DB.
# The resume then walked back into the creation branch and the pod refused the
# duplicate name (409) — one hour of 5090 time destroyed, twice over, for a
# run that was training fine (run #107).

class ConflictingRemote(FakeRemote):
    """Pod that already holds this run's job: a second create_job with the same
    name is refused with the ai-toolkit 409, and the job list can name it."""

    def __init__(self, existing_name, existing_id='j-existing', **kw):
        super().__init__(**kw)
        self.existing_name = existing_name
        self.existing_id = existing_id
        self.started = []
        self.create_attempts = []
        self.listed = 0

    def create_job(self, name, job_config, gpu_ids='0'):
        self.create_attempts.append(name)
        if name == self.existing_name:
            raise RuntimeError(
                'create_job -> HTTP 409: {"error":"Job name already exists"}')
        return super().create_job(name, job_config)

    def find_job_by_name(self, name):
        self.listed += 1
        if name == self.existing_name:
            return {'id': self.existing_id, 'name': name,
                    'status': 'running', 'step': 120}
        return None

    def start_job(self, job_id, gpu_ids='0'):
        self.started.append(job_id)

    def get_job(self, job_id):
        return super().get_job(job_id)


def test_resume_adopts_the_job_already_created_on_the_pod(ct, app, client, monkeypatch):
    """THE incident: create_job succeeded but the app restarted before
    remote_job_id was written, so the resume re-enters the creation branch and
    the pod answers 409. The run must ADOPT the existing job (the name proves
    it is ours) and keep polling it — not die with the hour already paid."""
    destroyed = []
    remote = FakeRemote(polls_to_complete=3)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    with app.app_context():
        run = ct.CloudTrainingRun.query.get(run_id)
        job_name = run.job_name
        # The crash window: the pod holds the job, our row still says NULL.
        ct._set(run, vast_instance_id='777', remote_job_id=None,
                status='uploading', base_url='http://pod.invalid:40123',
                auth_token='tok')
    conflicting = ConflictingRemote(job_name, polls_to_complete=3)
    monkeypatch.setattr(ct, '_make_remote', lambda run: conflicting)
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'done'                      # the run SURVIVES
        assert run.remote_job_id == 'j-existing'         # adopted, not recreated
        assert conflicting.listed == 1                   # resolved by name
        # A job already at step 120 must never be re-queued under a live trainer
        assert conflicting.started == []
        assert destroyed == ['777']


def test_remote_job_id_is_recorded_before_seeding_and_start(ct, app, client,
                                                            monkeypatch, tmp_path):
    """Remedy 1: the window itself. The id must be in the DB the instant the pod
    accepts the job — before the (slow) checkpoint seeding and before start_job.
    Simulated crash: seeding raises, and the row must still carry the id."""
    destroyed = []
    remote = FakeRemote(polls_to_complete=3)
    seed = tmp_path / 'source.safetensors'
    seed.write_bytes(b'CKPT')
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    with app.app_context():
        run = ct.CloudTrainingRun.query.get(run_id)
        params = json.loads(run.train_params)
        params.update({'resume_ckpt_path': str(seed), 'resume_step': 500})
        ct._set(run, train_params=json.dumps(params))

    def boom(datasets_folder, dest_dir, remote_name, local_path):
        raise RuntimeError('app died while seeding')

    remote.seed_checkpoint = boom
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'error'
        assert run.remote_job_id == 'j-1'    # recorded BEFORE the failure point


class NeverStartedRemote(FakeRemote):
    """Pod holding a job that was CREATED but never STARTED: ai-toolkit's
    default row status is 'stopped' at step 0 until `start` queues it."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.started = []

    def start_job(self, job_id, gpu_ids='0'):
        self.started.append(job_id)

    def get_job(self, job_id):
        if not self.started:
            return {'status': 'stopped', 'step': 0, 'info': 'created'}
        return super().get_job(job_id)


def test_resume_starts_a_job_that_was_created_but_never_launched(ct, app, client,
                                                                 monkeypatch):
    """The bug remedy 1 could trade in for: with the id recorded early, a crash
    between create_job and start_job leaves a job sitting at 'stopped' — which
    the poll loop reads as terminal. The resume must recognise 'created but not
    started', start it for real, and let it train."""
    destroyed = []
    remote = NeverStartedRemote(polls_to_complete=3)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    with app.app_context():
        run = ct.CloudTrainingRun.query.get(run_id)
        # Exactly what the DB holds after the create/start crash under remedy 1.
        ct._set(run, vast_instance_id='777', remote_job_id='j-1',
                status='uploading', base_url='http://pod.invalid:40123',
                auth_token='tok')
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert remote.started == ['j-1']      # started for real, not declared dead
        assert run.status == 'done'
        assert destroyed == ['777']


def test_unadoptable_conflict_says_what_to_do_and_what_happens_to_the_pod(
        ct, app, client, monkeypatch):
    """When the pod refuses the name AND will not say which job owns it, the run
    still fails — but the message must name the next move and the fate of the
    pod, because this failure costs real money."""
    destroyed = []
    remote = FakeRemote(polls_to_complete=3)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    with app.app_context():
        job_name = ct.CloudTrainingRun.query.get(run_id).job_name
    conflicting = ConflictingRemote(job_name, polls_to_complete=3)
    conflicting.find_job_by_name = lambda name: None      # listing resolves nothing
    monkeypatch.setattr(ct, '_make_remote', lambda run: conflicting)
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'error'
        msg = (run.error or '').lower()
        assert 'retry' in msg                       # what to do
        assert 'terminated' in msg                  # what happens to the pod
        assert destroyed == ['777']                 # and it really is


class FrozenRemote(FakeRemote):
    """Pod whose step counter progresses a few polls then freezes at 50 while
    the job status stays 'running' forever (wedged trainer, silent OOM loop)."""

    def get_job(self, job_id):
        self.polls += 1
        return {'status': 'running', 'step': min(self.polls * 10, 50),
                'total_steps': 100, 'info': 'Training', 'speed_string': '1 it/s'}


def test_stall_watchdog_kills_frozen_run(ct, app, client, monkeypatch):
    """A run whose step counter stops moving past stall_timeout_minutes is
    rescued (checkpoint attempted) and killed: status 'error', pod destroyed.
    Fake clock: 600 s per _now() call — several no-progress poll iterations
    blow through the 30-min stall budget long before the 480-min runtime cap."""
    destroyed = []
    remote = FrozenRemote(polls_to_complete=10_000)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    clock = {'t': 0.0}
    monkeypatch.setattr(ct, '_now',
                        lambda: clock.__setitem__('t', clock['t'] + 600.0) or clock['t'])
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'error'
        assert 'stall' in ((run.error or '') + (run.phase_detail or '')).lower()
        assert destroyed == ['777']
        # rescue-before-kill: the checkpoint download was attempted (and, with
        # this FakeRemote, succeeded into staging)
        assert run.checkpoint_local_path
        assert run.checkpoint_local_path.endswith('.safetensors')


class NeverStartsRemote(FakeRemote):
    """Pod that boots and accepts the job but never produces a training step:
    the job status stays 'running' with step 0 forever (base-model download
    collapsed to a crawl — the run #75 failure mode, 2026-07-19)."""

    def get_job(self, job_id):
        self.polls += 1
        if self.stopped:
            return {'status': 'stopped', 'step': 0, 'total_steps': 2100}
        return {'status': 'running', 'step': 0, 'total_steps': 2100,
                'info': 'fetching transformer weights', 'speed_string': ''}


def test_first_step_watchdog_kills_run_stuck_before_step_one(ct, app, client, monkeypatch):
    """A pod that never reaches step 1 (wedged base-model download) must be
    killed by the first-step watchdog — NOT left to burn the whole runtime cap.
    Without the watchdog this run runs until max_runtime (run #75: 10h45 / 7 €
    for zero saves). Coarse 600 s/_now() clock blows the 45-min first-step
    budget within a few polls, long before the 480-min runtime cap."""
    destroyed = []
    remote = NeverStartsRemote(polls_to_complete=10_000)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    clock = {'t': 0.0}
    monkeypatch.setattr(ct, '_now',
                        lambda: clock.__setitem__('t', clock['t'] + 600.0) or clock['t'])
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'error'
        assert 'first-step' in ((run.error or '') + (run.phase_detail or '')).lower()
        assert destroyed == ['777']            # pod terminated, not leaked
        assert not run.checkpoint_local_path   # nothing was ever produced to rescue
        assert remote.stopped                  # job stop was requested


# --- The first-step watchdog must read BYTES, not just steps -------------------
# j_o_e_l. (Discord, 2026-07-27) launched a KREA-2 RAW run on a 5090 and was
# told to sit tight because "we have some protection for buggy pods". 59 min
# later that protection killed the run: FAILED, error 'first-step watchdog',
# never past step 0 — while the pod was downloading its 26.3 GB base model
# perfectly normally. Measured on another pod's log (run #121): 2.58 MB/s puts
# that download at ~2 h 50, so the 45-min budget GUARANTEED the failure.


class DownloadingRemote(FakeRemote):
    """Pod stuck at step 0 forever — but whose log shows the base-model download
    advancing by 250 MB at every poll. j_o_e_l.'s pod."""

    def get_job(self, job_id):
        self.polls += 1
        if self.stopped:
            return {'status': 'stopped', 'step': 0, 'total_steps': 3500}
        return {'status': 'running', 'step': 0, 'total_steps': 3500,
                'info': 'downloading base model', 'speed_string': ''}

    def get_log(self, job_id):
        mb = 250 * self.polls
        return (f'raw.safetensors:   {min(99, mb // 270)}%|  | {mb}M/26.3G '
                f'[{self.polls}:00<2:37:06, 2.58MB/s]')


def test_first_step_watchdog_spares_a_pod_whose_download_advances(
        ct, app, client, monkeypatch):
    """j_o_e_l.'s run. No step for far longer than first_step_timeout_minutes, but
    the pod's byte counter moves at every poll — a pod whose bytes advance is a
    pod that progresses, and the guiding rule is NEVER kill a run that
    progresses. The coarse 600 s/_now() clock blows through the 45-min budget
    within two polls, so before the fix this run dies; after it, it trains."""
    destroyed = []
    remote = DownloadingRemote(polls_to_complete=10_000)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    clock = {'t': 0.0}
    monkeypatch.setattr(ct, '_now',
                        lambda: clock.__setitem__('t', clock['t'] + 600.0) or clock['t'])
    with app.app_context():
        # Download budget off: this test is about the idle clock being rearmed,
        # the absolute ceiling has its own test below.
        ct.cfg.save_config({'cloud': {'first_step_download_budget_minutes': 0}})
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        # It dies of the RUNTIME CAP (the honest backstop), never of the
        # first-step watchdog.
        assert 'first-step' not in ((run.error or '') + (run.phase_detail or '')).lower()
        assert run.status == 'stopped'
        assert 'max runtime' in (run.error or '').lower()


def test_first_step_watchdog_still_kills_a_genuinely_wedged_pod(
        ct, app, client, monkeypatch):
    """The anti-masking guard. No step AND no byte ever reported = a wedged pod
    burning money, and it must still die on the 45-min budget, not on the 8-hour
    runtime cap. If this ever regresses, the fix above became a cover-up."""
    destroyed = []
    remote = NeverStartsRemote(polls_to_complete=10_000)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    clock = {'t': 0.0}
    monkeypatch.setattr(ct, '_now',
                        lambda: clock.__setitem__('t', clock['t'] + 600.0) or clock['t'])
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'error'
        assert 'first-step' in (run.error or '').lower()
        assert destroyed == ['777']
        # And the message says what was MEASURED, not what was guessed. The old
        # wording ("pod likely stuck downloading the base model") is exactly
        # what j_o_e_l. read while his pod was downloading fine.
        detail = (run.phase_detail or '')
        assert 'likely' not in detail.lower()
        assert 'never reported' in detail.lower()


class FrozenDownloadRemote(FakeRemote):
    """MEASURED failure mode (run #121): the byte counter sticks at 1.95G while
    tqdm keeps reprinting the bar with an advancing `elapsed` and a decaying
    rate. The LINE moves at every poll; the DOWNLOAD is dead."""

    def get_job(self, job_id):
        self.polls += 1
        if self.stopped:
            return {'status': 'stopped', 'step': 0, 'total_steps': 3500}
        return {'status': 'running', 'step': 0, 'total_steps': 3500,
                'info': 'downloading base model', 'speed_string': ''}

    def get_log(self, job_id):
        return (f'raw.safetensors:   7%|  | 1.95G/26.3G '
                f'[{self.polls}:30<9:12:44, {max(0.01, 3.0 - self.polls * 0.1):.2f}MB/s]')


def test_frozen_byte_counter_is_not_progress(ct, app, client, monkeypatch):
    """A moving line is not a moving download. Only the bytes count — so this
    pod must die on the first-step budget like any other wedged pod."""
    destroyed = []
    remote = FrozenDownloadRemote(polls_to_complete=10_000)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    clock = {'t': 0.0}
    monkeypatch.setattr(ct, '_now',
                        lambda: clock.__setitem__('t', clock['t'] + 600.0) or clock['t'])
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'error'
        assert 'first-step' in (run.error or '').lower()
        assert destroyed == ['777']
        # ... and the message reports where it actually got to.
        assert '1.9' in (run.phase_detail or '') or '2.0 GB' in (run.phase_detail or '')


class DownloadThenSilentRemote(FakeRemote):
    """Pod that downloads for a few polls, finishes, and then never produces a
    step. Once the bytes stop the clock must RESTART — this is where the 45-min
    budget keeps all its meaning."""

    def get_job(self, job_id):
        self.polls += 1
        if self.stopped:
            return {'status': 'stopped', 'step': 0, 'total_steps': 3500}
        return {'status': 'running', 'step': 0, 'total_steps': 3500,
                'info': 'preparing', 'speed_string': ''}

    def get_log(self, job_id):
        done = min(self.polls, 3)
        return (f'raw.safetensors: 100%|  | {done * 8.7:.1f}G/26.1G '
                f'[{self.polls}:00<00:00, 40MB/s]')


def test_clock_restarts_when_the_download_ends_without_a_step(
        ct, app, client, monkeypatch):
    """The download completes at poll 3 and no step ever comes. The run must
    still be killed by the first-step watchdog — the rearm buys time for a
    download, it does not disarm the watchdog."""
    destroyed = []
    remote = DownloadThenSilentRemote(polls_to_complete=10_000)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    clock = {'t': 0.0}
    monkeypatch.setattr(ct, '_now',
                        lambda: clock.__setitem__('t', clock['t'] + 600.0) or clock['t'])
    with app.app_context():
        ct.cfg.save_config({'cloud': {'first_step_download_budget_minutes': 0}})
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'error'
        assert 'first-step' in (run.error or '').lower()
        assert destroyed == ['777']
        # It got to 26.1 GB, and the message says so instead of blaming the
        # download it just watched succeed.
        assert 'GB' in (run.phase_detail or '')


def test_download_budget_caps_an_endlessly_slow_pod(ct, app, client, monkeypatch):
    """The ceiling that keeps the rearm honest. A host at 200 kB/s advances its
    bytes at every poll for 36 h — rearming alone would let it burn the whole
    runtime cap for zero steps, which is the very failure (run #75) the
    first-step watchdog was built to stop. Past the budget it dies anyway."""
    destroyed = []
    remote = DownloadingRemote(polls_to_complete=10_000)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    clock = {'t': 0.0}
    monkeypatch.setattr(ct, '_now',
                        lambda: clock.__setitem__('t', clock['t'] + 600.0) or clock['t'])
    with app.app_context():
        ct.cfg.save_config({'cloud': {'first_step_download_budget_minutes': 60}})
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'error'
        assert 'download budget' in (run.error or '').lower()
        assert destroyed == ['777']
        # Killed WELL before the 480-min runtime cap.
        assert clock['t'] < 480 * 60


def test_progressing_run_never_trips_stall_watchdog(ct, app, client, monkeypatch):
    """Guiding principle: NEVER kill a run that makes progress. Same coarse
    fake clock as the stall test — a run whose step advances at every poll
    must ride through to completion untouched by the watchdog."""
    destroyed = []
    remote = FakeRemote(polls_to_complete=8)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    clock = {'t': 0.0}
    monkeypatch.setattr(ct, '_now',
                        lambda: clock.__setitem__('t', clock['t'] + 600.0) or clock['t'])
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'done'                  # not 'error'/'stall'
        assert destroyed == ['777']                  # normal terminate at completion


# --- Multi-family parallelism: each run builds from its OWN stamped params ------
# Root fix for the 2026-07-14 incident (a Krea run's pod would have been rebuilt
# as Z-Image after a later Z-Image launch overwrote ds.train_type). The monitor
# now builds via _run_config_dataset(run params), never the current dataset row.

def _echo_family_build(ct, monkeypatch):
    """build_job_config replacement that ECHOES the family/variant of whatever
    ds-view it is handed into the config, so the captured job_config proves which
    family/variant the monitor actually built from."""
    monkeypatch.setattr(ct.lt, 'build_job_config',
        lambda ds, folder, steps=3000, training_folder=None: {
            'job': 'extension', 'config': {'name': 'x', 'process': [{
                'type': 'sd_trainer', 'training_folder': training_folder,
                'device': 'cuda:0', 'datasets': [{'folder_path': folder}],
                            'model': {'arch': ds.train_type,
                                      'name_or_path': ds.train_base_model},
                            'variant': ds.train_variant}]}})


def test_two_families_each_build_own_config_despite_ds_change(ct, app, client, monkeypatch):
    """Two cloud runs of DIFFERENT families on the SAME dataset. Each launch
    persists ds.train_type/ds.train_variant (last writer wins), so by boot time
    the row no longer matches the FIRST run — yet each monitor must still build
    ITS OWN family/variant, or the first run's pod trains the wrong architecture
    on a rented GPU (incident 2026-07-14)."""
    r_zimage = FakeRemote(polls_to_complete=3)
    ds_id, run1 = _launch(ct, app, client, monkeypatch, r_zimage, [])   # zimage default
    r_krea = FakeRemote(polls_to_complete=3)
    ct.cfg.save_config({'cloud': {'max_concurrent_runs': 2}})
    with app.app_context():
        run2 = ct.launch_cloud_training('local', ds_id,
                                        train_type='krea', variant='base')['run_id']
    _echo_family_build(ct, monkeypatch)
    remotes = {run1: r_zimage, run2: r_krea}
    monkeypatch.setattr(ct, '_make_remote', lambda run: remotes[run.id])
    with app.app_context():
        # Adversary: the dataset row currently says krea (run2 was the last
        # writer) — mutate it AGAIN to prove NEITHER run reads it.
        ds = ct.fds.get_dataset('local', ds_id)
        ds.train_type = 'zimage'
        ds.train_variant = 'deturbo'
        ds.train_base_model = r'C:\later\custom-zimage.safetensors'
        ct.db.session.commit()
        ct._monitor(app, run1)
        ct._monitor(app, run2)
        assert ct.CloudTrainingRun.query.get(run1).status == 'done'
        assert ct.CloudTrainingRun.query.get(run2).status == 'done'
    p1 = r_zimage.job_config['config']['process'][0]
    p2 = r_krea.job_config['config']['process'][0]
    assert (p1['model']['arch'], p1['variant']) == ('zimage', 'turbo')
    assert (p2['model']['arch'], p2['variant']) == ('krea', 'base')
    assert p1['model']['name_or_path'] == ''
    assert p2['model']['name_or_path'] == ''


def test_run_config_immune_to_ds_change_while_booting(ct, app, client, monkeypatch):
    """A /train-type change (or a second launch) on the dataset WHILE a run boots
    must not alter that run's architecture. Launch a Krea run, flip the dataset
    to Z-Image, then run its monitor: the built config stays Krea."""
    ct.cfg.save_config({'cloud': {'max_concurrent_runs': 2}})
    remote = FakeRemote(polls_to_complete=3)
    ds_id, _ = _launch(ct, app, client, monkeypatch, remote, [])       # zimage run (inert)
    with app.app_context():
        run_id = ct.launch_cloud_training('local', ds_id,
                                          train_type='krea', variant='base')['run_id']
    _echo_family_build(ct, monkeypatch)
    with app.app_context():
        ds = ct.fds.get_dataset('local', ds_id)
        ds.train_type = 'zimage'
        ds.train_variant = 'turbo'
        ct.db.session.commit()
        ct._monitor(app, run_id)
        assert ct.CloudTrainingRun.query.get(run_id).status == 'done'
    proc = remote.job_config['config']['process'][0]
    assert proc['model']['arch'] == 'krea'
    assert proc['variant'] == 'base'


def test_harvest_imports_with_run_family_not_current_ds(ct, app, client, monkeypatch, tmp_path):
    """Deploy/import at harvest routes by the RUN's family, never the dataset's
    current row (which a later launch / train-type change may have moved)."""
    ct.cfg.save_config({'comfyui': {'loras_dir': str(tmp_path)}})
    remote = FakeRemote(polls_to_complete=3)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, [])  # a zimage run
    captured = {}

    def fake_import(*a, **kw):
        captured.update(kw)
        return 'x'

    monkeypatch.setattr(ct.lt, 'import_checkpoint', fake_import)
    with app.app_context():
        # Flip the dataset to krea AFTER launch — this run's harvest must still
        # deploy as zimage (its stamped family).
        ds = ct.fds.get_dataset('local', ds_id)
        ds.train_type = 'krea'
        ds.train_variant = 'base'
        ct.db.session.commit()
        ct._monitor(app, run_id)
        assert ct.CloudTrainingRun.query.get(run_id).status == 'done'
    assert captured['family'] == 'zimage'
    assert captured['base_model'] == ''
    assert captured['variant'] == 'turbo'


def test_boot_timeout_anchored_to_created_at_on_resume(ct, app, client, monkeypatch):
    """A resumed run (app restarted while the pod was still booting) must NOT get
    a brand-new readiness window: the boot timeout anchors to the durable
    created_at, so a pod whose UI never answers is given up at the FIRST poll
    once total time since launch exceeds READY_TIMEOUT. Before the fix each
    restart reset the window (incident 2026-07-14: 37 min instead of 15)."""
    from datetime import datetime, timedelta
    destroyed = []
    remote = FakeRemote()
    remote.is_ready = lambda: False                   # UI never answers
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    # Constant clock: only the durable anchor (run age), never elapsed monitor
    # time, can trip the boot timeout here.
    monkeypatch.setattr(ct, '_now', lambda: 0.0)
    with app.app_context():
        run = ct.CloudTrainingRun.query.get(run_id)
        ct._set(run, vast_instance_id='777', staging_dir='/tmp/x',
                base_url='http://1.2.3.4:40123',
                created_at=datetime.utcnow() - timedelta(minutes=30))   # > 15 min
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'error'
        assert 'ready' in (run.error or '').lower()   # 'did not become ready in time'
        assert destroyed == ['777']


def test_boot_timeout_fresh_run_not_charged_for_stale_created_at(ct, app, client, monkeypatch):
    """A FRESH launch (no pod yet at monitor entry) must NOT be charged the
    staging/provision time — nor a stale created_at — against its boot budget:
    its readiness window starts post-provision. With an old created_at and a UI
    that answers only after a couple of polls, the run must still reach the pod
    and complete. A durable-always anchor (the resume behaviour applied to a
    fresh run) would instead kill it on the FIRST poll, its age already past
    READY_TIMEOUT — so this test fails if the fresh/resume distinction is lost."""
    from datetime import datetime, timedelta
    destroyed = []
    remote = FakeRemote(polls_to_complete=3)
    ready_calls = {'n': 0}

    def slow_ready():
        ready_calls['n'] += 1
        return ready_calls['n'] >= 3                   # not ready for the first two polls

    remote.is_ready = slow_ready
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    clock = {'t': 0.0}
    monkeypatch.setattr(ct, '_now',
                        lambda: clock.__setitem__('t', clock['t'] + 60.0) or clock['t'])
    with app.app_context():
        run = ct.CloudTrainingRun.query.get(run_id)    # fresh, no pod yet
        ct._set(run, created_at=datetime.utcnow() - timedelta(minutes=30))
        ct._monitor(app, run_id)
        assert ct.CloudTrainingRun.query.get(run_id).status == 'done'
        assert destroyed == ['777']


# ── A close decided elsewhere wins over the monitor's terminal write ─────────
# _assert_run_open at the top of the poll loop only protects the loop. Every
# terminal branch then works for minutes (stop the job, pull the checkpoint,
# import, mirror) before it writes — and the supervisor runs in another thread,
# on another session. These two tests close the run INSIDE that window and
# require the monitor to stand down rather than rewrite the row.

def test_completed_branch_does_not_resurrect_a_run_closed_mid_download(
        ct, app, client, monkeypatch):
    """The pod completed, and while we were importing the result the supervisor
    force-stopped the run and destroyed the instance. The monitor must NOT flip
    that row back to 'done' — the pod it would be reporting on is gone."""
    destroyed = []
    remote = FakeRemote(polls_to_complete=2)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    real_mirror = ct._mirror_into_local_run

    def mirror_then_supervisor_closes(run, *a, **kw):
        out = real_mirror(run, *a, **kw)
        # what _force_stop writes once the vast API confirms the kill
        ct._set(run, status='stopped', phase_detail='Stopped by the supervisor',
                error='stop request not honoured in time',
                finished_at=ct.datetime.utcnow())
        return out

    monkeypatch.setattr(ct, '_mirror_into_local_run', mirror_then_supervisor_closes)
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'stopped'
        assert 'supervisor' in (run.phase_detail or '')
        # the checkpoint pulled before the close is still on disk — standing
        # down throws away the WRITE, never the work
        assert run.checkpoint_local_path and os.path.exists(run.checkpoint_local_path)


def test_pod_kept_is_not_announced_for_a_run_closed_mid_download(
        ct, app, client, monkeypatch):
    """The download failed, so the monitor wants to park the run in
    'error_pod_kept' and point the user at the pod. If the supervisor already
    destroyed that pod, that message sends them to an instance that no longer
    exists — the close must win."""
    destroyed = []
    remote = FakeRemote(polls_to_complete=2, fail_downloads=True)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    real_download = ct._try_download_checkpoint

    def download_then_supervisor_closes(run, *a, **kw):
        out = real_download(run, *a, **kw)
        ct._set(run, status='stopped', phase_detail='Stopped by the supervisor',
                finished_at=ct.datetime.utcnow())
        return out

    monkeypatch.setattr(ct, '_try_download_checkpoint', download_then_supervisor_closes)
    with app.app_context():
        ct._monitor(app, run_id)
        run = ct.CloudTrainingRun.query.get(run_id)
        assert run.status == 'stopped'
        assert 'recover manually' not in (run.error or '')
