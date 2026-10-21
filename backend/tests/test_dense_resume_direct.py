"""Continuing a FULL MODEL from the copy on THIS COMPUTER, and choosing between
the two roads with the price in front of you.

The Hub road was for a long time the only one, and the refusal that enforced it
was honest about its cause: the pod's upload seam built its whole multipart body
in memory, so 26 GB was an OOM. That is fixed, and what is left is not a wall
but a COST — the user's uplink, billed at the pod's hourly rate for the whole
climb, because the pod is rented and idle while it is being handed its file.

So the tests here are about two things in equal measure: that the file really
arrives, and that nobody is asked to choose a road without being told what it
costs.
"""

from public_dense_test_io import no_dense_provider_io  # noqa: F401
import json

import pytest

from lds_cloud_training import pod_checkpoint_push, pod_transfer_plan

pytestmark = pytest.mark.plugins('cloud_training')


@pytest.fixture()
def ct(app, monkeypatch):
    from lds_cloud_training import cloud_training
    from app.services import storage_locations
    monkeypatch.setattr(storage_locations, 'free_space',
                        lambda path: {'free_bytes': 4 * 1000 ** 4,
                                      'total_bytes': 8 * 1000 ** 4})
    return cloud_training


@pytest.fixture()
def dataset_id(client):
    return client.post('/api/dataset/create', json={
        'name': 'Dense direct', 'trigger_word': 'person',
    }).get_json()['id']


# The real article is ~26 GB. NOTHING here writes that: a sparse `truncate` of
# it costs a minute and the disk anyway (measured), and a test that fills the
# machine it runs on is a worse bug than the one it guards. Everything that
# depends on the SIZE is asserted through the pure estimators with plain
# integers; the files on disk exist only to be found, opened and routed.
_MASTER_BYTES = 2 * 10 ** 6
REAL_MASTER_BYTES = 26 * 10 ** 9


def _run_with_local_master(ct, dataset_id, tmp_path, size=_MASTER_BYTES, **params):
    """A finished dense run whose master is still on this computer."""
    store = tmp_path / 'store'
    store.mkdir(exist_ok=True)
    name = 'Krea_lds7_dense_000003000.safetensors'
    (store / name).write_bytes(b'w' * size)
    run = ct.CloudTrainingRun(
        dataset_id=dataset_id, status='done', run_name='dense',
        job_name='Krea_lds7_dense', staging_dir=str(store),
        price_per_hour=1.40,
        train_params=json.dumps({
            'training_mode': 'full_transformer', 'train_type': 'krea',
            'variant': 'base', 'dense_delivery': 'local', 'steps': 3000,
            **params}))
    ct.db.session.add(run)
    ct.db.session.commit()
    return run, str(store / name)


def _with_hub(ct, run, status='available', size=26 * 10 ** 9):
    params = json.loads(run.train_params)
    params.update({'dense_delivery': 'both', 'artifact_status': status,
                   'hf_repo_id': 'tester/Krea-2-full-7-dense',
                   'hf_weight_filename': 'Krea_lds7_dense_000003000.safetensors',
                   'hf_artifact_proof': {'size_bytes': size}})
    run.train_params = json.dumps(params)
    ct.db.session.commit()
    return run


# --- the choice exists, and defaults the way it always behaved ------------------

def test_the_local_copy_is_now_an_offered_road(ct, app, dataset_id, tmp_path):
    with app.app_context():
        run, path = _run_with_local_master(ct, dataset_id, tmp_path)
        [candidate] = ct._dense_resume_candidates(run)
        assert candidate['source'] == 'local'
        assert candidate['path'] == path
        assert candidate['size_bytes'] == _MASTER_BYTES
        assert candidate['step'] == 3000


def test_a_quantized_twin_is_not_a_road(ct, app, dataset_id, tmp_path):
    """fp8 weights carry per-tensor scales, not the bf16 the trainer loads."""
    with app.app_context():
        run, _ = _run_with_local_master(ct, dataset_id, tmp_path)
        (tmp_path / 'store' / 'Krea_lds7_dense_000003000_fp8.safetensors'
         ).write_bytes(b'q' * 1000)
        assert [c['filename'] for c in ct._dense_resume_candidates(run)] == [
            'Krea_lds7_dense_000003000.safetensors']


def test_with_both_roads_open_the_default_is_still_the_hub(ct, app, dataset_id,
                                                           tmp_path):
    """The historical behaviour, preserved deliberately: the road that takes
    hours and bills a GPU the whole time is a decision, never a default."""
    with app.app_context():
        run, _ = _run_with_local_master(ct, dataset_id, tmp_path)
        _with_hub(ct, run)
        candidates = ct._dense_resume_candidates(run)
        assert {c['source'] for c in candidates} == {'local', 'hub'}
        assert ct.dense_resume_transport(candidates) == 'hub'
        assert candidates[-1]['source'] == 'hub', (
            'the no-argument pick (cks[-1]) must keep landing on the Hub copy')


def test_the_default_stays_the_hub_even_when_a_LATER_local_save_exists(
        ct, app, dataset_id, tmp_path):
    """"Newest checkpoint" and "cheapest road" are different rules, and silently
    following the first would put a user on a three-hour upload they never
    chose."""
    with app.app_context():
        run, _ = _run_with_local_master(ct, dataset_id, tmp_path)
        _with_hub(ct, run)
        (tmp_path / 'store' / 'Krea_lds7_dense_000004500.safetensors'
         ).write_bytes(b'w' * 1000)
        candidates = ct._dense_resume_candidates(run)
        assert max(c['step'] for c in candidates) == 4500
        assert ct.dense_resume_transport(candidates) == 'hub'


def test_with_no_hub_copy_the_local_road_is_the_default(ct, app, dataset_id,
                                                        tmp_path):
    with app.app_context():
        run, _ = _run_with_local_master(ct, dataset_id, tmp_path)
        assert ct.dense_resume_transport(ct._dense_resume_candidates(run)) == 'local'


def test_an_explicit_choice_wins_over_the_default(ct, app, dataset_id, tmp_path):
    with app.app_context():
        run, _ = _run_with_local_master(ct, dataset_id, tmp_path)
        candidates = _with_hub(ct, run) and ct._dense_resume_candidates(run)
        assert ct.dense_resume_transport(candidates, 'direct') == 'local'
        assert ct.dense_resume_transport(candidates, 'hub') == 'hub'


# --- what continue_cloud_run does with the choice -------------------------------

def test_continue_with_direct_transport_stamps_the_local_path(
        ct, app, dataset_id, tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(ct, 'launch_cloud_training',
                        lambda u, d, **kw: (seen.update(kw), {'run_id': 9})[1])
    with app.app_context():
        run, path = _run_with_local_master(ct, dataset_id, tmp_path)
        _with_hub(ct, run)
        res = ct.continue_cloud_run('local', run.id, transport='direct')
    assert res['resume_transport'] == 'direct'
    assert seen['resume_ckpt_path'] == path
    assert seen['resume_hf'] is None
    assert seen['training_mode'] == 'full_transformer'


def test_continue_with_hub_transport_still_stamps_the_repository(
        ct, app, dataset_id, tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(ct, 'launch_cloud_training',
                        lambda u, d, **kw: (seen.update(kw), {'run_id': 9})[1])
    with app.app_context():
        run, _ = _run_with_local_master(ct, dataset_id, tmp_path)
        _with_hub(ct, run)
        res = ct.continue_cloud_run('local', run.id, transport='hub')
    assert res['resume_transport'] == 'hub'
    assert seen['resume_ckpt_path'] is None
    assert seen['resume_hf']['repo_id'] == 'tester/Krea-2-full-7-dense'


def test_asking_for_a_road_that_is_closed_says_which_one_and_why(
        ct, app, dataset_id, tmp_path):
    with app.app_context():
        run, _ = _run_with_local_master(ct, dataset_id, tmp_path)
        with pytest.raises(ValueError, match='no Hugging Face copy'):
            ct.continue_cloud_run('local', run.id, transport='hub')


def test_a_run_with_nothing_left_says_so_without_blaming_transport(
        ct, app, dataset_id, tmp_path):
    with app.app_context():
        run, path = _run_with_local_master(ct, dataset_id, tmp_path)
        __import__('os').remove(path)
        with pytest.raises(ValueError) as err:
            ct.continue_cloud_run('local', run.id)
        assert 'nothing left to continue from' in str(err.value)
        assert 'memory' not in str(err.value), (
            'the in-memory upload limit is gone; a message that still blames it '
            'sends the user to fix something that is not broken')


# --- the launch no longer refuses a local dense seed ----------------------------

def _launch_ready(ct, monkeypatch):
    monkeypatch.setattr(ct, '_start_monitor', lambda *a, **k: None)
    monkeypatch.setattr(ct, '_reconcile_before_launch', lambda *a, **k: None)
    monkeypatch.setattr(ct.lt, 'assert_trainable', lambda *a, **k: None)
    monkeypatch.setattr(ct, '_assert_official_base_reachable', lambda *a, **k: None)
    monkeypatch.setattr(ct, '_validate_full_transformer_token',
                        lambda token, _api=None, **_kw: (object(), 'tester', False))
    monkeypatch.setattr(ct, '_assert_dense_storage_headroom',
                        lambda *a, **k: {'fits': True})
    monkeypatch.setattr(ct, '_create_full_transformer_repo', lambda run, token, **_kw: {
        'hf_repo_id': 'tester/Krea-2-full-9-dense', 'hf_url': 'https://hf/x'})
    monkeypatch.setenv('VAST_API_KEY', 'vast-test')
    monkeypatch.setenv('HF_CLOUD_TOKEN', 'hf-cloud-secret-x')


def test_a_dense_launch_accepts_a_local_seed_and_measures_it(
        ct, app, dataset_id, tmp_path, monkeypatch):
    _launch_ready(ct, monkeypatch)
    with app.app_context():
        _run, path = _run_with_local_master(ct, dataset_id, tmp_path)
        result = ct.launch_cloud_training(
            'local', dataset_id, train_type='krea', variant='base',
            training_mode='full_transformer', steps=4000,
            resume_ckpt_path=path, resume_step=3000, resumed_from=3000)
        params = json.loads(ct.db.session.get(
            ct.CloudTrainingRun, result['run_id']).train_params)
    assert params['resume_source'] == 'local'
    # Frozen at launch because the pod is RENTED from it, minutes later.
    assert params['resume_ckpt_bytes'] == _MASTER_BYTES


def test_a_pod_is_rented_big_enough_for_the_checkpoint_being_pushed_to_it(ct):
    """At RENTAL time on purpose: finding the shortfall when the file is half
    sent means the money is spent and the pod is scrap."""
    dense = {'full_transformer': {}}
    # A Hub seed asks for nothing extra — the pod writes straight to destination.
    assert ct._disk_gb_for(dense, {'training_mode': 'full_transformer'}) == 200
    # A pushed one asks for itself plus the single slice it is assembled through,
    # NOT for twice itself: the assembly deletes each slice as it consumes it.
    assert ct._disk_gb_for(dense, {'training_mode': 'full_transformer',
                                   'resume_ckpt_bytes': REAL_MASTER_BYTES}) == 229


def test_a_dense_launch_still_refuses_a_seed_that_is_not_there(
        ct, app, dataset_id, tmp_path, monkeypatch):
    _launch_ready(ct, monkeypatch)
    with app.app_context():
        with pytest.raises(ValueError, match='no longer on this computer'):
            ct.launch_cloud_training(
                'local', dataset_id, train_type='krea', variant='base',
                training_mode='full_transformer', steps=4000,
                resume_ckpt_path=str(tmp_path / 'gone.safetensors'),
                resume_step=3000)


def test_full_models_are_still_cloud_only_and_say_the_right_reason(
        ct, app, dataset_id):
    """The refusal survived, its wording did not: it used to blame the transport
    seam, which is now fixed. The real reason is upstream — there is no local
    full-model run to continue, because they are never trained here."""
    with app.app_context():
        with pytest.raises(ValueError) as err:
            ct.continue_local_run_in_cloud('local', dataset_id,
                                           training_mode='full_transformer')
    assert 'only trained in the cloud' in str(err.value)


# --- the seeding routes to the sliced push --------------------------------------

def test_a_big_local_seed_goes_through_the_resumable_push(
        ct, app, dataset_id, tmp_path, monkeypatch):
    calls = {}

    def _push(remote, **kwargs):
        calls.update(kwargs)
        return {'bytes': 26 * 10 ** 9, 'sent_bytes': 26 * 10 ** 9, 'slices': 13,
                'sent_slices': 13, 'skipped_slices': 0, 'reused': False}

    monkeypatch.setattr(pod_checkpoint_push, 'push_checkpoint', _push)

    class _Remote:
        def seed_checkpoint(self, *a, **k):
            raise AssertionError('a 26 GB file must not go in one request — an '
                                 'interruption at 80% would restart from zero')

    with app.app_context():
        src = tmp_path / 'master.safetensors'
        src.write_bytes(b'w' * 5000)
        run = ct.CloudTrainingRun(
            dataset_id=dataset_id, status='provisioning', run_name='dense',
            job_name='Krea_lds9_dense', vast_instance_id='pod-9',
            staging_dir=str(tmp_path),
            train_params=json.dumps({
                'training_mode': 'full_transformer', 'resume_step': 3000,
                'resume_ckpt_path': str(src)}))
        ct.db.session.add(run)
        ct.db.session.commit()
        ct._seed_resume_checkpoint(run, _Remote(), {
            'DATASETS_FOLDER': '/datasets', 'TRAINING_FOLDER': '/output'})

    assert calls['dest_dir'] == '/output/Krea_lds9_dense'
    assert calls['remote_name'] == 'Krea_lds9_dense_000003000.safetensors'
    assert calls['instance_id'] == 'pod-9'
    assert calls['job_name'] == 'Krea_lds9_dense'


def test_a_small_lora_uses_the_staged_push(ct, app, dataset_id, tmp_path,
                                        monkeypatch):
    """Small LoRAs must also respect the upload route's dataset boundary."""
    seen = {}

    def push(remote, **kwargs):
        seen.update(kwargs)
        return {'bytes': 850_000, 'sent_bytes': 850_000}

    monkeypatch.setattr(pod_checkpoint_push, 'push_checkpoint', push)

    class _Remote:
        def seed_checkpoint(self, *args, **kwargs):
            pytest.fail('a resume must not upload directly to the output folder')

    with app.app_context():
        src = tmp_path / 'lora.safetensors'
        src.write_bytes(b'l' * 85 * 10 ** 4)
        run = ct.CloudTrainingRun(
            dataset_id=dataset_id, status='provisioning', run_name='lora',
            job_name='lds9_lora', vast_instance_id='pod-9', staging_dir=str(tmp_path),
            train_params=json.dumps({'resume_step': 500,
                                     'resume_ckpt_path': str(src)}))
        ct.db.session.add(run)
        ct.db.session.commit()
        ct._seed_resume_checkpoint(run, _Remote(), {
            'DATASETS_FOLDER': '/datasets', 'TRAINING_FOLDER': '/output'})
    assert seen['remote_name'] == 'lds9_lora_000000500.safetensors'
    assert seen['dest_dir'] == '/output/lds9_lora'
    assert seen['datasets_folder'] == '/datasets'
    assert seen['instance_id'] == 'pod-9'
    assert seen['local_path'] == str(src)


def test_a_resumed_push_does_not_record_a_fake_uplink_speed(
        ct, app, dataset_id, tmp_path, monkeypatch):
    """A push that skipped 20 GB already on the pod took seconds. Timing the
    WHOLE file against that would report an uplink several times faster than the
    line has ever been — and that number becomes a price."""
    monkeypatch.setattr(pod_checkpoint_push, 'push_checkpoint',
                        lambda *a, **k: {'bytes': 26 * 10 ** 9,
                                         'sent_bytes': 0, 'reused': True})
    recorded = []
    monkeypatch.setattr(pod_transfer_plan, 'record_uplink_sample',
                        lambda b, s, kind=None: recorded.append((b, s, kind)))

    class _Remote:
        pass

    with app.app_context():
        src = tmp_path / 'master.safetensors'
        src.write_bytes(b'w' * 5000)
        run = ct.CloudTrainingRun(
            dataset_id=dataset_id, status='provisioning', run_name='dense',
            job_name='Krea_j', vast_instance_id='pod-9', staging_dir=str(tmp_path),
            train_params=json.dumps({'resume_step': 3000,
                                     'resume_ckpt_path': str(src)}))
        ct.db.session.add(run)
        ct.db.session.commit()
        ct._seed_resume_checkpoint(run, _Remote(), {
            'DATASETS_FOLDER': '/datasets', 'TRAINING_FOLDER': '/output'})
    assert recorded == [(0, pytest.approx(0, abs=5), pod_transfer_plan.KIND_STREAM)]


def test_a_dataset_upload_is_filed_as_bulk_not_as_line_throughput(ct, tmp_path,
                                                                  monkeypatch):
    """A dataset is thousands of small files at eight per POST: what that timed
    is per-request latency, not the throughput a 26 GB push would see."""
    recorded = []
    monkeypatch.setattr(pod_transfer_plan, 'record_uplink_sample',
                        lambda b, s, kind=None: recorded.append((b, s, kind)))
    folder = tmp_path / 'dataset'
    folder.mkdir()
    for i in range(3):
        (folder / f'{i}.png').write_bytes(b'x' * 1000)
    ct._record_uplink(None, str(folder), 12.0)
    assert recorded == [(3000, 12.0, pod_transfer_plan.KIND_BULK)]


# --- the plan: the numbers shown before the click -------------------------------

def test_the_plan_quotes_each_road_the_size_of_ITS_OWN_file(ct, app, dataset_id,
                                                             tmp_path):
    """The Hub size comes from the verification proof (the file measured ON the
    Hub); the local size from the file on disk. Quoting one for the other is
    how a forecast becomes confidently wrong."""
    with app.app_context():
        run, _ = _run_with_local_master(ct, dataset_id, tmp_path)
        _with_hub(ct, run, size=REAL_MASTER_BYTES)
        plan = ct.dense_resume_plan('local', run.id)
    by_road = {o['transport']: o for o in plan['options']}
    assert set(by_road) == {'hub', 'direct'}
    assert all(o['available'] for o in by_road.values())
    assert by_road['hub']['bytes'] == REAL_MASTER_BYTES
    assert by_road['direct']['bytes'] == _MASTER_BYTES
    assert plan['price_per_hour'] == 1.40
    assert plan['price_source'] == 'this run'
    assert plan['default_transport'] == 'hub'


def test_at_the_same_size_the_direct_road_costs_more_gpu_than_the_hub(app):
    """The number this whole panel exists for: a pod is rented and idle while
    it is being handed its file, so the slow road is the expensive one."""
    with app.app_context():
        direct = pod_transfer_plan.estimate_direct(REAL_MASTER_BYTES, 1.40)
        hub = pod_transfer_plan.estimate_hub(REAL_MASTER_BYTES, 1.40)
    assert direct['seconds'] > hub['seconds']
    assert direct['gpu_cost'] > hub['gpu_cost'] > 0
    # 26 GB over an assumed 50 Mbit/s uplink is hours, and at $1.40/h that is
    # dollars of graphics card computing nothing. If this ever reads as cents,
    # the arithmetic broke, not the world.
    assert direct['gpu_cost'] > 1.0


def test_the_plan_says_whether_the_speed_was_measured_or_assumed(
        ct, app, dataset_id, tmp_path):
    with app.app_context():
        run, _ = _run_with_local_master(ct, dataset_id, tmp_path)
        direct = [o for o in ct.dense_resume_plan('local', run.id)['options']
                  if o['transport'] == 'direct'][0]
        assert direct['rate_source'] == 'assumed' and direct['rate_samples'] == 0

        pod_transfer_plan.record_uplink_sample(2 * 10 ** 9, 100.0)
        direct = [o for o in ct.dense_resume_plan('local', run.id)['options']
                  if o['transport'] == 'direct'][0]
        assert direct['rate_source'] == 'measured' and direct['rate_samples'] == 1
        assert direct['rate_mbps'] == pytest.approx(160.0, rel=0.01)


@pytest.mark.parametrize('params,fragment', [
    # Never delivered to the Hub at all.
    ({'dense_delivery': 'local'}, 'this computer only'),
    # Delivered there, but the copy is not trustworthy yet.
    ({'dense_delivery': 'both', 'artifact_status': 'verification_pending',
      'hf_repo_id': 'tester/x', 'hf_weight_filename': 'w.safetensors'},
     'not verified'),
    # Delivered there on paper, but no copy is on record. Reporting this as
    # "not verified" would send the user to re-verify something that is not
    # there; reporting it as a delivery setting would send them to change a
    # setting that is already right.
    ({'dense_delivery': 'both'}, 'no Hugging Face copy on record'),
])
def test_a_closed_hub_road_names_the_reason_it_is_closed(
        ct, app, dataset_id, tmp_path, params, fragment):
    """A greyed-out option with no explanation is how a user ends up reading
    source code to find out that a trade-off exists at all."""
    with app.app_context():
        run, _ = _run_with_local_master(ct, dataset_id, tmp_path, **params)
        hub = [o for o in ct.dense_resume_plan('local', run.id)['options']
               if o['transport'] == 'hub'][0]
    assert hub['available'] is False
    assert fragment in hub['reason']


def test_a_deleted_repository_closes_the_hub_road_before_a_pod_is_rented(
        ct, app, dataset_id, tmp_path, monkeypatch):
    """The registry cannot answer this. `artifact_status` is stamped once at
    delivery and never revisited, so a repository deleted last night still reads
    'available' — the road was offered, PRICED, given an ETA, and choosing it
    rented a pod that took a 404. The plan and the launch both measure."""
    from app.services import hub_presence
    monkeypatch.setattr(hub_presence, 'check', lambda repo_id, **k: {
        'repo_id': repo_id, 'state': hub_presence.GONE, 'detail': 'deleted',
        'checked_at': 'now', 'cached': False})
    with app.app_context():
        run, _ = _run_with_local_master(ct, dataset_id, tmp_path)
        _with_hub(ct, run)                      # registry still says 'available'
        assert any(c['source'] == 'hub' for c in ct._dense_resume_candidates(run)), (
            'the registry still believes in the copy — that is the whole trap')

        hub = [o for o in ct.dense_resume_plan('local', run.id)['options']
               if o['transport'] == 'hub'][0]
        assert hub['available'] is False
        assert hub['gpu_cost'] == 0, 'a dead road must never carry a price'
        assert 'gone' in hub['reason'] and 'checked just now' in hub['reason']

        # ...and the launch refuses too. The plan is advice; this is the spend.
        with pytest.raises(ValueError, match='does not answer'):
            ct.continue_cloud_run('local', run.id, transport='hub')


def test_a_repository_that_could_not_be_checked_keeps_its_road_open(
        ct, app, dataset_id, tmp_path, monkeypatch):
    """Offline, no token, a 5xx. Closing the fast road because someone's Wi-Fi
    dropped would be a worse failure than the one being prevented."""
    from app.services import hub_presence
    monkeypatch.setattr(hub_presence, 'check', lambda repo_id, **k: {
        'repo_id': repo_id, 'state': hub_presence.UNKNOWN,
        'detail': 'could not check', 'checked_at': 'now', 'cached': False})
    seen = {}
    monkeypatch.setattr(ct, 'launch_cloud_training',
                        lambda u, d, **kw: (seen.update(kw), {'run_id': 9})[1])
    with app.app_context():
        run, _ = _run_with_local_master(ct, dataset_id, tmp_path)
        _with_hub(ct, run)
        hub = [o for o in ct.dense_resume_plan('local', run.id)['options']
               if o['transport'] == 'hub'][0]
        assert hub['available'] is True
        ct.continue_cloud_run('local', run.id, transport='hub')
    assert seen['resume_hf']['repo_id'] == 'tester/Krea-2-full-7-dense'


def test_an_fp8_only_hub_copy_is_not_reported_as_deleted(ct, app, dataset_id,
                                                          tmp_path):
    """The branch that said "gone" used to be reachable ONLY through this case,
    which is not a deletion at all — the file is right there and simply cannot
    be trained. Blaming a deletion that never happened would send someone
    looking for a repository that is fine."""
    with app.app_context():
        run, _ = _run_with_local_master(ct, dataset_id, tmp_path)
        _with_hub(ct, run)
        params = json.loads(run.train_params)
        params['hf_weight_filename'] = 'Krea_lds7_dense_000003000_fp8.safetensors'
        run.train_params = json.dumps(params)
        ct.db.session.commit()
        hub = [o for o in ct.dense_resume_plan('local', run.id)['options']
               if o['transport'] == 'hub'][0]
    assert hub['available'] is False
    assert 'fp8' in hub['reason']
    assert 'gone' not in hub['reason'] and 'deleted' not in hub['reason']


def test_a_closed_local_road_explains_the_fp8_twin(ct, app, dataset_id, tmp_path):
    with app.app_context():
        run, path = _run_with_local_master(ct, dataset_id, tmp_path)
        _with_hub(ct, run)
        __import__('os').remove(path)
        direct = [o for o in ct.dense_resume_plan('local', run.id)['options']
                  if o['transport'] == 'direct'][0]
    assert direct['available'] is False
    assert 'fp8' in direct['reason']


def test_the_plan_falls_back_to_the_price_cap_and_says_so(ct, app, dataset_id,
                                                          tmp_path):
    with app.app_context():
        run, _ = _run_with_local_master(ct, dataset_id, tmp_path)
        run.price_per_hour = None
        ct.db.session.commit()
        plan = ct.dense_resume_plan('local', run.id)
    assert plan['price_source'] == 'the price cap in Settings'
    assert plan['price_per_hour'] > 0


def test_the_plan_endpoint_answers_200_even_for_a_closed_road(client, ct, app,
                                                              dataset_id, tmp_path,
                                                              monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'vast-test')
    with app.app_context():
        run, _ = _run_with_local_master(ct, dataset_id, tmp_path)
        run_id = run.id
    r = client.post('/api/dataset/train/cloud/resume-plan', json={'run_id': run_id})
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is True
    assert {o['transport'] for o in body['options']} == {'hub', 'direct'}
