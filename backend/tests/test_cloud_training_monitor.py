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


def test_auto_retry_classifier_only_accepts_transient_pod_failures(ct):
    assert ct._is_retryable_pod_failure(
        "('Connection aborted.', ConnectionResetError(10054, 'closed'))")
    assert ct._is_retryable_pod_failure('pod did not become ready in time')
    assert ct._is_retryable_pod_failure('pod unreachable: connection refused')
    assert not ct._is_retryable_pod_failure('CUDA out of memory')
    assert not ct._is_retryable_pod_failure('stall watchdog')


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
        files = os.listdir(run.staging_dir)
        assert sorted(f for f in files if f.endswith('.safetensors')) == \
            ['j_000000100.safetensors', 'j_000000200.safetensors',
             'j_000000300.safetensors']
        assert not any(f.endswith('.part') for f in files)
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
        staged = sorted(f for f in os.listdir(run.staging_dir)
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
