"""Dense Krea 2 cloud-lifecycle contract (all external services mocked)."""
import json
from datetime import timedelta
from types import SimpleNamespace

import pytest


class _FakeHfApi:
    def __init__(self, root, files=None, failure=None, role='fineGrained',
                 permissions=None, scopes=None, global_permissions=None,
                 can_read_gated=False, orgs=None, file_sizes=None,
                 file_sha256=None, file_blob_ids=None):
        self.root = root
        self.files = list(files or [])
        self.failure = failure
        self.role = role
        delivery_permissions = list(permissions or [
            'repo.content.read', 'repo.write'])
        self.scopes = list(scopes or [
            {
                'entity': {'type': 'model', 'name': 'krea/Krea-2-Raw'},
                'permissions': ['repo.content.read'],
            },
            {
                'entity': {'type': 'user', 'name': 'tester'},
                'permissions': delivery_permissions,
            },
        ])
        self.global_permissions = list(global_permissions or [])
        self.can_read_gated = can_read_gated
        self.orgs = list(orgs or [])
        self.file_sizes = dict(file_sizes or {})
        self.file_sha256 = dict(file_sha256 or {})
        self.file_blob_ids = dict(file_blob_ids or {})
        self.created = []
        self.deleted = []
        self.uploaded = {}
        self.list_calls = []

    def whoami(self):
        return {
            'name': 'tester',
            'orgs': self.orgs,
            'auth': {'accessToken': {
                'role': self.role,
                'fineGrained': {
                    'global': self.global_permissions,
                    'canReadGatedRepos': self.can_read_gated,
                    'scoped': self.scopes,
                },
            }},
        }

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def list_repo_files(self, **kwargs):
        self.list_calls.append(kwargs)
        if self.failure:
            raise self.failure
        if str(kwargs.get('repo_id') or '').startswith('krea/Krea-2-'):
            return ['LICENSE.pdf', 'README.md']
        return self.files

    def repo_info(self, **kwargs):
        self.list_calls.append(kwargs)
        if self.failure:
            raise self.failure
        siblings = []
        for filename in self.files:
            is_weight = str(filename).lower().endswith('.safetensors')
            size = self.file_sizes.get(
                filename,
                16 * 1024 ** 3 if is_weight else len(
                    self.uploaded.get((kwargs.get('repo_id'), filename), b'x')))
            sha256 = self.file_sha256.get(filename, 'b' * 64)
            blob_id = self.file_blob_ids.get(filename, 'a' * 40)
            siblings.append(SimpleNamespace(
                rfilename=filename, size=size, blob_id=blob_id,
                lfs=({'size': size, 'sha256': sha256} if is_weight else None)))
        return SimpleNamespace(siblings=siblings)

    def upload_file(self, **kwargs):
        payload = kwargs['path_or_fileobj']
        if not isinstance(payload, bytes):
            payload = bytes(payload)
        key = (kwargs['repo_id'], kwargs['path_in_repo'])
        self.uploaded[key] = payload
        if kwargs['path_in_repo'] not in self.files:
            self.files.append(kwargs['path_in_repo'])

    def hf_hub_download(self, **kwargs):
        repo_id = kwargs['repo_id']
        filename = kwargs['filename']
        if repo_id.startswith('krea/Krea-2-') and filename == 'LICENSE.pdf':
            payload = b'%PDF-1.7 official Krea 2 licence fixture'
        else:
            payload = self.uploaded[(repo_id, filename)]
        path = self.root / f'{len(list(self.root.iterdir()))}-{filename}'
        path.write_bytes(payload)
        return str(path)

    def delete_repo(self, **kwargs):
        self.deleted.append(kwargs)


@pytest.fixture()
def ct(app, monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'vast-test')
    monkeypatch.setenv('HF_TOKEN', 'hf-general-must-not-reach-dense')
    monkeypatch.setenv('HF_CLOUD_TOKEN', 'hf-cloud-secret-x')
    # A dense launch now also measures THIS machine's disk (the delivery lands
    # here first). The measurement is the only part faked: the forecast, the
    # refusal and its override all stay live in every test below — a suite whose
    # verdict depends on how full the developer's C: drive is would be worse
    # than no test at all.
    from app.services import storage_locations
    monkeypatch.setattr(storage_locations, 'free_space',
                        lambda path: {'free_bytes': 4 * 1000 ** 4,
                                      'total_bytes': 8 * 1000 ** 4})
    from app.services import cloud_training
    monkeypatch.setattr(cloud_training, '_start_monitor', lambda *a, **k: None)
    monkeypatch.setattr(cloud_training, '_reconcile_before_launch', lambda *a, **k: None)
    monkeypatch.setattr(cloud_training.lt, 'assert_trainable', lambda *a, **k: None)
    monkeypatch.setattr(cloud_training.lt, 'default_steps', lambda *a, **k: 800)
    monkeypatch.setattr(cloud_training, '_assert_official_base_reachable',
                        lambda *a, **k: None)
    return cloud_training


@pytest.fixture()
def dataset_id(client):
    return client.post('/api/dataset/create', json={
        'name': 'Dense portrait', 'trigger_word': 'person',
    }).get_json()['id']


def test_full_launch_creates_private_per_run_repo_and_freezes_mode(
        ct, app, dataset_id, monkeypatch, tmp_path):
    api = _FakeHfApi(tmp_path)
    seen_tokens = []
    monkeypatch.setattr(
        ct, '_make_hf_api',
        lambda token: seen_tokens.append(token) or api)

    with app.app_context():
        result = ct.launch_cloud_training(
            'local', dataset_id, train_type='krea', variant='base',
            training_mode='full_transformer')
        run = ct.db.session.get(ct.CloudTrainingRun, result['run_id'])
        params = json.loads(run.train_params)
        from app.services import face_dataset_service as fds
        ds = fds.get_dataset('local', dataset_id)

        assert params['training_mode'] == 'full_transformer'
        assert params['artifact_kind'] == 'full_transformer'
        assert params['artifact_status'] == 'pending'
        assert params['hf_repo_id'].split('/', 1)[1].startswith('Krea')
        assert run.job_name.startswith('Krea')
        assert params['hf_url'] == f"https://huggingface.co/{params['hf_repo_id']}"
        assert result['hf_repo_id'] == params['hf_repo_id']
        assert ds.training_mode == 'full_transformer'
        assert api.created == [{
            'repo_id': params['hf_repo_id'], 'repo_type': 'model',
            'private': True, 'exist_ok': False,
        }]
        assert seen_tokens and set(seen_tokens) == {'hf-cloud-secret-x'}
        assert {name for repo, name in api.uploaded
                if repo == params['hf_repo_id']} == {
                    'LICENSE.pdf', 'NOTICE', 'README.md'}
        notice = api.uploaded[(params['hf_repo_id'], 'NOTICE')].decode()
        readme = api.uploaded[(params['hf_repo_id'], 'README.md')].decode()
        assert ct._KREA_REQUIRED_ATTRIBUTION in notice
        assert 'modified derivative' in notice
        assert 'not endorsed by Krea' in notice
        assert 'base_model: krea/Krea-2-Raw' in readme
        assert 'license_name: krea-2-community-license' in readme

        # The monitor view replays the run snapshot, never today's preference.
        view = ct._run_config_dataset(ds, params)
        ds.training_mode = 'lora'
        ct.db.session.commit()
        assert view.training_mode == 'full_transformer'
        assert 'hf-cloud-secret-x' not in run.train_params
        assert 'hf-general-must-not-reach-dense' not in run.train_params
        assert 'hf-cloud-secret-x' not in json.dumps(ct._run_payload(run))


@pytest.mark.parametrize('kwargs,message', [
    ({'training_mode': 'FULL_TRANSFORMER', 'train_type': 'krea'}, 'training_mode'),
    ({'training_mode': 'full_transformer', 'train_type': 'zimage'}, 'only for Krea'),
    # Turbo is ALLOWED now (the config carries krea/Krea-2-Turbo); what is still
    # refused before any reservation is a base that is not an absolute path,
    # because the krea2 loader would ignore it and train the official base under
    # a name that says otherwise.
    ({'training_mode': 'full_transformer', 'train_type': 'krea',
      'base_model': 'custom.safetensors'}, 'full path'),
    # A local seed is no longer refused for BEING one — it is sent to the pod in
    # resumable slices now. What is still refused, and still before any
    # reservation, is a seed that is not on this computer at all.
    ({'training_mode': 'full_transformer', 'train_type': 'krea',
      'resume_ckpt_path': 'seed.safetensors'}, 'no longer on this computer'),
])
def test_full_launch_validation_happens_before_reservation(
        ct, app, dataset_id, kwargs, message):
    with app.app_context(), pytest.raises(ValueError, match=message):
        ct.launch_cloud_training('local', dataset_id, **kwargs)
    with app.app_context():
        assert ct.CloudTrainingRun.query.count() == 0


def test_full_launch_rejects_slider_and_missing_hf_token(
        ct, app, dataset_id, monkeypatch):
    from app.services import face_dataset_service as fds
    with app.app_context():
        ds = fds.get_dataset('local', dataset_id)
        ds.train_slider = json.dumps({'enabled': True})
        ct.db.session.commit()
        with pytest.raises(ValueError, match='Slider'):
            ct.launch_cloud_training(
                'local', dataset_id, train_type='krea', variant='base',
                training_mode='full_transformer')
        ds.train_slider = None
        ct.db.session.commit()
        monkeypatch.delenv('HF_CLOUD_TOKEN', raising=False)
        with pytest.raises(ValueError, match='HF_CLOUD_TOKEN'):
            ct.launch_cloud_training(
                'local', dataset_id, train_type='krea', variant='base',
                training_mode='full_transformer')
        assert ct.CloudTrainingRun.query.count() == 0


class _FullAccountHfApi(_FakeHfApi):
    """The same dense token, on an account whose PRIVATE storage is already
    full — the state that killed run #146 at step 2750/3000."""

    def list_models(self, author=None, expand=None):
        repo = SimpleNamespace(id=f'{author}/lds-base-h1111', private=True,
                               last_modified=None)
        repo.usedStorage = 95 * 1000 ** 3
        return [repo]

    def list_datasets(self, author=None, expand=None):
        return []


def _hub_only(monkeypatch):
    """Force the historical Hugging-Face-only delivery for one test."""
    from app import config as cfg
    cloud = dict(cfg.get('cloud') or {})
    dense = dict(cloud.get('full_transformer') or {})
    dense['delivery'] = 'hub'
    cloud['full_transformer'] = dense
    monkeypatch.setattr(
        cfg, 'get',
        lambda key, default=None, _real=cfg.get: (
            cloud if key == 'cloud' else _real(key, default)))


def test_full_launch_refuses_before_renting_when_hf_storage_is_full(
        ct, app, dataset_id, monkeypatch, tmp_path):
    """The pre-check has to fire BEFORE the reservation row exists: the point is
    to lose nothing, and every later refusal has already cost something.

    Scoped to the hub-ONLY delivery, which is the case the refusal is still
    about: there the repository is the artifact's only address, so a forecast
    that does not fit means the run has nowhere to land. A delivery that also
    brings the model home no longer refuses — see the warning test below."""
    _hub_only(monkeypatch)
    api = _FullAccountHfApi(tmp_path)
    monkeypatch.setattr(ct, '_make_hf_api', lambda token: api)
    with app.app_context():
        with pytest.raises(ValueError) as excinfo:
            ct.launch_cloud_training(
                'local', dataset_id, train_type='krea', variant='base',
                training_mode='full_transformer')
        message = str(excinfo.value)
        assert message.startswith('HF_STORAGE_FULL: ')
        assert 'lds-base-h1111' in message
        assert ct.CloudTrainingRun.query.count() == 0
        assert api.created == []


def test_full_launch_warns_instead_of_refusing_when_the_model_comes_home(
        ct, app, dataset_id, monkeypatch, tmp_path):
    """A full private quota can no longer end a dense run — only its backup.

    The default delivery downloads the model to this computer FIRST and pushes
    the master afterwards, so a Hub that will not fit costs resumability, not
    the run. It is still said BEFORE the GPU is rented: learning that a model
    cannot be continued is worth eight hours of notice."""
    api = _FullAccountHfApi(tmp_path)
    monkeypatch.setattr(ct, '_make_hf_api', lambda token: api)
    with app.app_context():
        result = ct.launch_cloud_training(
            'local', dataset_id, train_type='krea', variant='base',
            training_mode='full_transformer')
        run = ct.db.session.get(ct.CloudTrainingRun, result['run_id'])
        params = json.loads(run.train_params)
        assert params['dense_delivery'] == 'both'
        assert 'not be able to continue' in result['warning'] \
            or 'cannot be continued' in result['warning']
        assert params['hub_backup_warning'] == result['warning']
        # The run exists, the repository was created, nothing was refused.
        assert api.created and params['artifact_status'] == 'pending'


def test_full_launch_honours_the_storage_override_and_stamps_it(
        ct, app, dataset_id, monkeypatch, tmp_path):
    """The ceiling is an estimate, so 'Train anyway' must exist — and be
    replayed by an automatic retry rather than re-litigated."""
    _hub_only(monkeypatch)
    api = _FullAccountHfApi(tmp_path)
    monkeypatch.setattr(ct, '_make_hf_api', lambda token: api)
    with app.app_context():
        result = ct.launch_cloud_training(
            'local', dataset_id, train_type='krea', variant='base',
            training_mode='full_transformer', allow_hf_storage=True)
        run = ct.db.session.get(ct.CloudTrainingRun, result['run_id'])
        params = json.loads(run.train_params)
        assert params['allow_hf_storage'] is True
        assert ct._confirmation_flags(params)['allow_hf_storage'] is True


def test_the_storage_precheck_never_reaches_for_the_general_token(
        ct, app, dataset_id, monkeypatch, tmp_path):
    """The pre-check measures with HF_CLOUD_TOKEN and stays blind otherwise.

    Reading the general HF_TOKEN would give a better forecast on a dense token
    too narrow to list its namespace — and it is deliberately NOT done: dense
    runs are cut off from that credential, and widening a least-privilege
    boundary to sharpen an estimate is the wrong trade. Pinned here so the
    blind spot stays a decision rather than becoming a regression."""
    dense = _FakeHfApi(tmp_path)                     # no list_models at all
    general = _FullAccountHfApi(tmp_path)            # would have said "full"
    monkeypatch.setattr(
        ct, '_make_hf_api',
        lambda token: general if token == 'hf-general-must-not-reach-dense' else dense)
    with app.app_context():
        result = ct.launch_cloud_training(
            'local', dataset_id, train_type='krea', variant='base',
            training_mode='full_transformer')
        assert result['run_id']


def test_full_launch_is_not_blocked_by_an_unmeasurable_account(
        ct, app, dataset_id, monkeypatch, tmp_path):
    """_FakeHfApi has no list_models at all — the Hub could not be measured, and
    an unmeasurable number must never lock a user out of their own GPU."""
    api = _FakeHfApi(tmp_path)
    monkeypatch.setattr(ct, '_make_hf_api', lambda token: api)
    with app.app_context():
        result = ct.launch_cloud_training(
            'local', dataset_id, train_type='krea', variant='base',
            training_mode='full_transformer')
        assert result['run_id']


def test_dense_continue_refuses_only_what_is_genuinely_impossible(
        ct, app, dataset_id):
    """Three refusals that SURVIVED the arrival of the direct road, each for its
    own reason — and none of them for the old one.

    This test used to be called "needs a hub copy and never a local seed", and
    its docstring explained all three by the same dead fact: that the only seam
    putting a file on a pod built its whole request in memory. That is no longer
    true (pod_checkpoint_push), so a test that still asserted it would have been
    guarding the absence of a feature that now exists. What is left is narrower
    and still worth holding:

    * a run with NO source at all — nothing on disk, no verified Hub copy —
      cannot be continued, and says so instead of pretending the option exists;
    * a retry replays its seed verbatim, so a seed that has since been deleted
      is a refusal rather than a silent train-from-scratch;
    * a full model still cannot be continued from the LOCAL lane, for a reason
      that predates all of this: full-model training is cloud-only, so there is
      no local full-model run to continue in the first place."""
    with app.app_context():
        done = ct.CloudTrainingRun(
            dataset_id=dataset_id, status='done', run_name='dense',
            train_params=json.dumps({'training_mode': 'full_transformer',
                                     'train_type': 'krea', 'variant': 'base'}))
        retry = ct.CloudTrainingRun(
            dataset_id=dataset_id, status='error', run_name='dense-retry',
            train_params=json.dumps({
                'training_mode': 'full_transformer', 'train_type': 'krea',
                'variant': 'base', 'resume_ckpt_path': 'seed.safetensors'}))
        ct.db.session.add_all([done, retry])
        ct.db.session.commit()
        with pytest.raises(ValueError, match='nothing left to continue from'):
            ct.continue_cloud_run('local', done.id)
        with pytest.raises(ValueError, match='no longer on this computer'):
            ct.retry_cloud_run('local', retry.id)
        with pytest.raises(ValueError, match='only trained in the cloud'):
            ct.continue_local_run_in_cloud(
                'local', dataset_id, training_mode='full_transformer')


def test_full_profile_is_applied_to_offers_and_provisioning(
        ct, app, dataset_id, monkeypatch):
    searches = []
    creates = []
    offers = [{'offer_id': 17, 'gpu_name': 'H100 SXM', 'dph_total': 2.5,
               'gpu_ram_gb': 80.0}]
    monkeypatch.setattr(ct, '_filter_offers', lambda rows: rows)
    monkeypatch.setattr(
        ct.vast_client, 'search_offers',
        lambda **kwargs: searches.append(kwargs) or offers)
    monkeypatch.setattr(
        ct.vast_client, 'create_instance',
        lambda offer_id, **kwargs: creates.append((offer_id, kwargs)) or 'pod-1')
    monkeypatch.setattr(ct, '_register_instance', lambda *a, **k: None)

    with app.app_context():
        tiers = ct.gpu_tiers(
            'local', dataset_id, train_type='krea', variant='base', steps=500,
            training_mode='full_transformer')
        run = SimpleNamespace(
            train_params=json.dumps({'training_mode': 'full_transformer',
                                     'train_type': 'krea'}),
            vast_label='lds-dense')
        ct._provision(run)

    assert tiers['training_mode'] == 'full_transformer'
    assert tiers['disk_gb'] == 200
    assert tiers['tiers'][0]['estimate_status'] == 'unavailable'
    assert tiers['tiers'][0]['est_minutes'] is None
    assert tiers['tiers'][0]['est_cost'] is None
    assert tiers['tiers'][0]['exceeds_cap'] is None
    assert searches[0]['min_vram_gb'] == 80
    assert searches[1]['min_vram_gb'] == 80
    assert creates[0][1]['disk_gb'] == 200


def _job_config(network=False):
    proc = {
        'type': 'sd_trainer', 'device': 'cpu', 'training_folder': 'local',
        'save': {'dtype': 'bf16'}, 'datasets': [{'folder_path': 'C:/staging'}],
        'train': {},
    }
    if network:
        proc['network'] = {'type': 'lora'}
    return {'config': {'name': 'local-name', 'process': [proc]}}


def test_cloudify_injects_private_hf_push_without_secret(ct):
    out = ct._cloudify_job_config(
        _job_config(), 'Krea_job', 'C:/staging',
        {'DATASETS_FOLDER': '/datasets', 'TRAINING_FOLDER': '/output'},
        run_params={'training_mode': 'full_transformer',
                    'hf_repo_id': 'tester/Krea-dense'})
    proc = out['config']['process'][0]
    assert proc['type'] == 'diffusion_trainer'
    assert proc['save'] == {
        'dtype': 'bf16', 'push_to_hub': True,
        'hf_repo_id': 'tester/Krea-dense', 'hf_private': True,
    }
    assert 'token' not in json.dumps(out).lower()
    with pytest.raises(RuntimeError, match='LoRA network'):
        ct._cloudify_job_config(
            _job_config(network=True), 'Krea_job', 'C:/staging',
            {'DATASETS_FOLDER': '/datasets', 'TRAINING_FOLDER': '/output'},
            run_params={'training_mode': 'full_transformer',
                        'hf_repo_id': 'tester/Krea-dense'})


def test_full_short_circuits_every_lora_artifact_path(
        ct, tmp_path, monkeypatch):
    staging = tmp_path / 'run'
    staging.mkdir()
    (staging / 'stray.safetensors').write_bytes(b'not-a-lora')
    run = SimpleNamespace(
        id=91, remote_job_id='job', staging_dir=str(staging),
        checkpoint_local_path=str(staging / 'stray.safetensors'),
        train_params=json.dumps({'training_mode': 'full_transformer'}))

    class BombRemote:
        def __getattr__(self, name):
            raise AssertionError(f'LoRA remote path called: {name}')

    monkeypatch.setattr(ct.lt, 'import_checkpoint',
                        lambda *a, **k: pytest.fail('ComfyUI import called'))
    remote = BombRemote()
    assert ct._run_staging_checkpoints(run) == []
    assert ct._sync_latest_checkpoint(run, remote) is None
    assert ct._try_download_checkpoint(run, remote) is False
    assert ct._download_intermediates(run, remote) is None
    assert ct._import_result(run) is None
    assert ct._mirror_into_local_run(run) is None


def test_dense_routes_never_import_or_serve_staging_weights_as_lora(
        ct, app, client, dataset_id, tmp_path, monkeypatch):
    staging = tmp_path / 'dense-route-run'
    staging.mkdir()
    stray = staging / 'stray.safetensors'
    stray.write_bytes(b'not-a-lora')
    with app.app_context():
        run = ct.CloudTrainingRun(
            dataset_id=dataset_id, status='done', run_name='dense-route',
            staging_dir=str(staging), checkpoint_local_path=str(stray),
            train_params=json.dumps({
                'training_mode': 'full_transformer',
                'artifact_kind': 'full_transformer',
                'artifact_status': 'available',
                'hf_url': 'https://huggingface.co/tester/Krea-dense-route',
            }))
        ct.db.session.add(run)
        ct.db.session.commit()
        run_id = run.id

    monkeypatch.setattr(
        'app.routes.training.capabilities.probe',
        lambda: {'aitoolkit': {'valid': True}, 'cloud_training': True})
    monkeypatch.setattr(
        ct.lt, 'import_checkpoint',
        lambda *a, **k: pytest.fail('dense staging file reached LoRA import'))

    imported = client.post(
        f'/api/dataset/{dataset_id}/train/import',
        json={'cloud_run_id': run_id, 'filename': stray.name})
    downloaded = client.get(
        f'/api/dataset/{dataset_id}/train/cloud/checkpoint'
        f'?run_id={run_id}&filename={stray.name}')

    # Delivery metadata remains the answer even after local staging was purged.
    with app.app_context():
        run = ct.db.session.get(ct.CloudTrainingRun, run_id)
        run.staging_dir = None
        ct.db.session.commit()
    imported_after_purge = client.post(
        f'/api/dataset/{dataset_id}/train/import',
        json={'cloud_run_id': run_id, 'filename': stray.name})

    for response in (imported, downloaded, imported_after_purge):
        assert response.status_code == 409
        payload = response.get_json()
        assert payload['training_mode'] == 'full_transformer'
        assert payload['artifact_status'] == 'available'
        assert payload['hf_url'].endswith('/tester/Krea-dense-route')
        assert 'cannot be imported or downloaded as a LoRA' in payload['error']


def test_hf_verification_controls_availability_and_serialization(
        ct, app, dataset_id, caplog, tmp_path):
    with app.app_context():
        run = ct.CloudTrainingRun(
            dataset_id=dataset_id, status='training', run_name='dense',
            job_name='Krea_dense',
            checkpoint_local_path='C:/must/not/be/exposed.safetensors',
            train_params=json.dumps({
                'training_mode': 'full_transformer',
                'artifact_kind': 'full_transformer',
                'artifact_status': 'pending',
                'hf_repo_id': 'tester/Krea-dense',
                'hf_url': 'https://huggingface.co/tester/Krea-dense',
            }))
        ct.db.session.add(run)
        ct.db.session.commit()

        assert ct._verify_full_transformer_artifact(
            run, _api=_FakeHfApi(
                tmp_path, ['README.md', 'model/Krea_dense_final.safetensors'])) == 'available'
        payload = ct._run_payload(run)
        assert payload['artifact_status'] == 'available'
        assert payload['artifact_status_detail']
        assert payload['hf_weight_filename'] == 'model/Krea_dense_final.safetensors'
        assert payload['hf_artifact_proof'] == {
            'size_bytes': 16 * 1024 ** 3,
            'sha256': 'b' * 64,
            'blob_id': 'a' * 40,
            'metadata_source': 'huggingface_repo_info_files_metadata',
        }
        assert payload['verified_at']
        assert payload['delivery_last_checked_at']
        assert payload['checkpoint_ready'] is False
        assert payload['checkpoint_local_path'] is None
        assert payload['hf_url'].endswith('/tester/Krea-dense')
        assert 'no checkpoint_local_path' in payload['artifact_delivery']

        # A random safetensors must not masquerade as this dense job's result.
        assert ct._verify_full_transformer_artifact(
            run, _api=_FakeHfApi(
                tmp_path, ['README.md', 'model/stray.safetensors'])) == 'missing'
        assert ct._run_param(run, 'artifact_status') == 'missing'

        malicious = RuntimeError('Bearer hf-cloud-secret-x')
        assert ct._verify_full_transformer_artifact(
            run, _api=_FakeHfApi(tmp_path, failure=malicious)) == 'verification_pending'
        assert 'hf-cloud-secret-x' not in caplog.text
        assert 'hf-cloud-secret-x' not in run.train_params


def test_dense_completion_is_fail_closed_and_keeps_unverified_pods(
        ct, app, dataset_id, tmp_path, monkeypatch):
    destroyed = []
    monkeypatch.setattr(
        ct.vast_client, 'destroy_instance',
        lambda instance_id: destroyed.append(instance_id) or True)
    monkeypatch.setattr(ct, '_sleep', lambda *_: None)

    def make_run(suffix):
        run = ct.CloudTrainingRun(
            dataset_id=dataset_id, status='training', run_name=suffix,
            job_name=f'Krea_{suffix}', vast_instance_id=f'pod-{suffix}',
            train_params=json.dumps({
                'training_mode': 'full_transformer',
                'artifact_kind': 'full_transformer',
                'artifact_status': 'pending',
                'hf_repo_id': f'tester/Krea-{suffix}',
                'hf_url': f'https://huggingface.co/tester/Krea-{suffix}',
            }))
        ct.db.session.add(run)
        ct.db.session.commit()
        return run

    with app.app_context():
        available = make_run('available')
        ct._complete_full_transformer_delivery(
            available, _api=_FakeHfApi(
                tmp_path, ['Krea_available_final.safetensors']))
        assert available.status == 'done'
        assert destroyed == ['pod-available']

        pending = make_run('pending')
        pending_api = _FakeHfApi(
            tmp_path, failure=RuntimeError('Bearer hf-cloud-secret-x'))
        ct._complete_full_transformer_delivery(pending, _api=pending_api)
        assert pending.status == 'error_pod_kept'
        assert ct._run_param(pending, 'artifact_status') == 'verification_pending'
        assert len(pending_api.list_calls) == 3
        assert destroyed == ['pod-available']

        stray = make_run('stray')
        ct._complete_full_transformer_delivery(
            stray, _api=_FakeHfApi(
                tmp_path, ['model/not-this-job.safetensors']))
        assert stray.status == 'error_pod_kept'
        assert ct._run_param(stray, 'artifact_status') == 'missing'
        assert destroyed == ['pod-available']


@pytest.mark.parametrize('size', [0, 1024 * 1024])
def test_empty_or_truncated_dense_weight_never_releases_pod(
        ct, app, dataset_id, tmp_path, monkeypatch, size):
    destroyed = []
    monkeypatch.setattr(
        ct.vast_client, 'destroy_instance',
        lambda instance_id: destroyed.append(instance_id) or True)
    filename = 'model/Krea_integrity_final.safetensors'
    api = _FakeHfApi(
        tmp_path, [filename], file_sizes={filename: size})

    with app.app_context():
        run = ct.CloudTrainingRun(
            dataset_id=dataset_id, status='training', run_name='integrity',
            job_name='Krea_integrity', vast_instance_id='pod-integrity',
            train_params=json.dumps({
                'training_mode': 'full_transformer',
                'artifact_status': 'pending',
                'hf_repo_id': 'tester/Krea-integrity',
            }))
        ct.db.session.add(run)
        ct.db.session.commit()

        ct._complete_full_transformer_delivery(run, _api=api)
        assert run.status == 'error_pod_kept'
        assert ct._run_param(run, 'artifact_status') == 'missing'
        assert ct._run_param(run, 'hf_artifact_proof') is None
        assert 'truncated' in ct._run_param(run, 'artifact_status_detail')
        assert destroyed == []


def test_late_hf_propagation_is_reconciled_and_releases_pod(
        ct, app, dataset_id, tmp_path, monkeypatch):
    destroyed = []
    monkeypatch.setattr(
        ct.vast_client, 'destroy_instance',
        lambda instance_id: destroyed.append(instance_id) or True)
    api = _FakeHfApi(tmp_path)

    with app.app_context():
        run = ct.CloudTrainingRun(
            dataset_id=dataset_id, status='error_pod_kept',
            run_name='late', job_name='Krea_late',
            vast_instance_id='pod-late', finished_at=ct.naive_utcnow(),
            train_params=json.dumps({
                'training_mode': 'full_transformer',
                'artifact_status': 'verification_pending',
                'hf_repo_id': 'tester/Krea-late',
                'hf_url': 'https://huggingface.co/tester/Krea-late',
            }))
        ct.db.session.add(run)
        ct.db.session.commit()
        run_id = run.id

        first = ct.reconcile_full_transformer_deliveries(_api=api)
        assert first == [{
            'run_id': run_id, 'delivery': 'missing', 'completed': False}]
        assert run.status == 'error_pod_kept'
        assert destroyed == []

        api.files.append('model/Krea_late_final.safetensors')
        second = ct.reconcile_full_transformer_deliveries(_api=api)
        assert second == [{
            'run_id': run_id, 'delivery': 'available', 'completed': True}]
        ct.db.session.expire_all()
        run = ct.db.session.get(ct.CloudTrainingRun, run_id)
        assert run.status == 'done'
        assert ct._run_param(run, 'artifact_status') == 'available'
        assert ct._run_param(run, 'hf_artifact_proof')['size_bytes'] >= 8 * 1024 ** 3
        assert destroyed == ['pod-late']


def test_verified_delivery_stays_visible_and_retries_cleanup_until_confirmed(
        ct, app, dataset_id, monkeypatch, caplog):
    """False/exception are ambiguous: neither may publish a completed run."""
    outcomes = iter([False, RuntimeError('vast-secret-diagnostic'), True])
    destroyed = []

    def destroy(instance_id):
        destroyed.append(instance_id)
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    class NoHfCalls:
        def __getattr__(self, name):
            raise AssertionError(
                f'already-verified cleanup must not call Hugging Face: {name}')

    monkeypatch.setattr(ct.vast_client, 'destroy_instance', destroy)
    with app.app_context():
        run = ct.CloudTrainingRun(
            dataset_id=dataset_id, status='error_pod_kept',
            run_name='cleanup-retry', job_name='Krea_cleanup_retry',
            vast_instance_id='pod-cleanup-retry',
            finished_at=ct.naive_utcnow(),
            train_params=json.dumps({
                'training_mode': 'full_transformer',
                'artifact_kind': 'full_transformer',
                'artifact_status': 'available',
                'artifact_cleanup_status': 'pending',
                'hf_repo_id': 'tester/Krea-cleanup-retry',
                'hf_url': 'https://huggingface.co/tester/Krea-cleanup-retry',
                'hf_weight_filename': (
                    'model/Krea_cleanup_retry_final.safetensors'),
                'hf_artifact_proof': {
                    'size_bytes': 16 * 1024 ** 3,
                    'sha256': 'b' * 64,
                    'blob_id': 'a' * 40,
                    'metadata_source': (
                        'huggingface_repo_info_files_metadata'),
                },
            }))
        ct.db.session.add(run)
        ct.db.session.commit()
        run_id = run.id

        first = ct.recheck_full_transformer_delivery(
            run_id, _api=NoHfCalls())
        assert first['delivery'] == 'available'
        assert first['cleanup_pending'] is True
        assert first['run']['status'] == 'error_pod_kept'
        assert first['run']['artifact_status'] == 'available'
        assert first['run']['artifact_cleanup_status'] == 'pending'
        assert first['run']['hf_url'].endswith('/tester/Krea-cleanup-retry')

        second = ct.reconcile_full_transformer_deliveries(_api=NoHfCalls())
        assert second == [{
            'run_id': run_id, 'delivery': 'available', 'completed': False}]
        ct.db.session.expire_all()
        run = ct.db.session.get(ct.CloudTrainingRun, run_id)
        assert run.status == 'error_pod_kept'
        assert ct._run_param(run, 'artifact_status') == 'available'
        assert ct._run_param(run, 'artifact_cleanup_status') == 'pending'
        assert 'vast-secret-diagnostic' not in caplog.text
        assert 'vast-secret-diagnostic' not in (run.error or '')

        third = ct.reconcile_full_transformer_deliveries(_api=NoHfCalls())
        assert third == [{
            'run_id': run_id, 'delivery': 'available', 'completed': True}]
        ct.db.session.expire_all()
        run = ct.db.session.get(ct.CloudTrainingRun, run_id)
        assert run.status == 'done'
        assert run.error is None
        assert ct._run_param(run, 'artifact_status') == 'available'
        assert ct._run_param(run, 'artifact_cleanup_status') == 'complete'
        assert destroyed == ['pod-cleanup-retry'] * 3


def test_supervisor_periodically_reaps_expired_verified_cleanup_and_retries(
        ct, app, dataset_id, monkeypatch):
    """An open app reaps after the deadline and retries an ambiguous delete."""
    destroyed = []
    outcomes = iter([False, True])
    monkeypatch.setattr(ct, 'supervise_active_runs', lambda: None)
    # Isolate the account-reaper policy. Delivery cleanup has its own retry test
    # above and normally runs before this throttled pass.
    monkeypatch.setattr(
        ct, 'reconcile_full_transformer_deliveries', lambda: [])
    monkeypatch.setattr(
        ct.vast_client, 'list_instances',
        lambda: [{'instance_id': 'pod-expired', 'label': 'lds-expired'}])
    monkeypatch.setattr(
        ct.vast_client, 'destroy_instance',
        lambda instance_id: destroyed.append(instance_id) or next(outcomes))

    with app.app_context():
        run = ct.CloudTrainingRun(
            dataset_id=dataset_id, status='error_pod_kept',
            run_name='expired', job_name='Krea_expired',
            vast_instance_id='pod-expired', finished_at=ct.naive_utcnow(),
            train_params=json.dumps({
                'training_mode': 'full_transformer',
                'artifact_kind': 'full_transformer',
                'artifact_status': 'available',
                'artifact_cleanup_status': 'pending',
                'hf_repo_id': 'tester/Krea-expired',
                'hf_url': 'https://huggingface.co/tester/Krea-expired',
            }))
        ct.db.session.add(run)
        ct.db.session.commit()
        run_id = run.id

    # Still in the recovery window: the periodic account scan must spare it.
    ct._supervisor_tick(app, reap_orphans=True)
    assert destroyed == []

    with app.app_context():
        run = ct.db.session.get(ct.CloudTrainingRun, run_id)
        run.finished_at = ct.naive_utcnow() - timedelta(minutes=481)
        ct.db.session.commit()

    # First expired pass cannot confirm destruction, so the row stays
    # retryable and the already-verified model remains visible.
    ct._supervisor_tick(app, reap_orphans=True)
    assert destroyed == ['pod-expired']
    with app.app_context():
        run = ct.db.session.get(ct.CloudTrainingRun, run_id)
        assert run.status == 'error_pod_kept'
        assert ct._run_param(run, 'artifact_status') == 'available'
        assert ct._run_param(run, 'artifact_cleanup_status') == 'pending'

    # A later tick retries, confirms cleanup, and only then publishes done.
    ct._supervisor_tick(app, reap_orphans=True)
    assert destroyed == ['pod-expired', 'pod-expired']
    with app.app_context():
        run = ct.db.session.get(ct.CloudTrainingRun, run_id)
        assert run.status == 'done'
        assert run.error is None
        assert ct._run_param(run, 'artifact_status') == 'available'
        assert ct._run_param(run, 'artifact_cleanup_status') == 'complete'


def test_reconcile_marks_legacy_verified_cleanup_done_when_listing_proves_absence(
        ct, app, dataset_id, monkeypatch):
    monkeypatch.setattr(ct.vast_client, 'list_instances', lambda: [])
    monkeypatch.setattr(
        ct.vast_client, 'destroy_instance',
        lambda *_: (_ for _ in ()).throw(
            AssertionError('an absent pod must not be destroyed again')))

    with app.app_context():
        run = ct.CloudTrainingRun(
            dataset_id=dataset_id, status='error_pod_kept',
            run_name='already-absent', job_name='Krea_already_absent',
            vast_instance_id='pod-already-absent',
            finished_at=ct.naive_utcnow(),
            train_params=json.dumps({
                'training_mode': 'full_transformer',
                'artifact_kind': 'full_transformer',
                'artifact_status': 'available',
                'hf_repo_id': 'tester/Krea-already-absent',
                'hf_url': 'https://huggingface.co/tester/Krea-already-absent',
            }))
        ct.db.session.add(run)
        ct.db.session.commit()
        run_id = run.id

    assert ct.reconcile_orphans(app) == 0
    with app.app_context():
        run = ct.db.session.get(ct.CloudTrainingRun, run_id)
        assert run.status == 'done'
        assert run.error is None
        assert ct._run_param(run, 'artifact_status') == 'available'
        assert ct._run_param(run, 'artifact_cleanup_status') == 'complete'
        assert 'listing confirmed' in ct._run_param(
            run, 'artifact_cleanup_detail')


def test_reconcile_listing_failure_never_claims_verified_cleanup_complete(
        ct, app, dataset_id, monkeypatch):
    monkeypatch.setattr(
        ct.vast_client, 'list_instances',
        lambda: (_ for _ in ()).throw(RuntimeError('listing unavailable')))

    with app.app_context():
        run = ct.CloudTrainingRun(
            dataset_id=dataset_id, status='error_pod_kept',
            run_name='unknown-presence', job_name='Krea_unknown_presence',
            vast_instance_id='pod-unknown-presence',
            finished_at=ct.naive_utcnow(),
            train_params=json.dumps({
                'training_mode': 'full_transformer',
                'artifact_kind': 'full_transformer',
                'artifact_status': 'available',
                'artifact_cleanup_status': 'pending',
                'hf_repo_id': 'tester/Krea-unknown-presence',
                'hf_url': 'https://huggingface.co/tester/Krea-unknown-presence',
            }))
        ct.db.session.add(run)
        ct.db.session.commit()
        run_id = run.id

    assert ct.reconcile_orphans(app) == 0
    with app.app_context():
        run = ct.db.session.get(ct.CloudTrainingRun, run_id)
        assert run.status == 'error_pod_kept'
        assert ct._run_param(run, 'artifact_status') == 'available'
        assert ct._run_param(run, 'artifact_cleanup_status') == 'pending'


def test_supervisor_loop_throttles_but_repeats_account_reconciliation(
        ct, app, monkeypatch):
    reaping = []
    monotonic = iter([0.0, 60.0, 301.0])

    class StopLoop(Exception):
        pass

    def sleep(_seconds):
        if len(reaping) == 3:
            raise StopLoop

    monkeypatch.setattr(ct.time, 'monotonic', lambda: next(monotonic))
    monkeypatch.setattr(
        ct, '_supervisor_tick',
        lambda _app, *, reap_orphans=False: reaping.append(reap_orphans))
    monkeypatch.setattr(ct, '_sleep', sleep)

    with pytest.raises(StopLoop):
        ct._supervisor_loop(app)
    assert reaping == [True, False, True]


def test_dense_repo_preparation_failure_is_cleaned_and_repo_id_persisted(
        ct, app, dataset_id, tmp_path, monkeypatch):
    api = _FakeHfApi(tmp_path)
    monkeypatch.setattr(
        ct, '_apply_full_transformer_compliance',
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('upload failed')))
    with app.app_context():
        run = ct.CloudTrainingRun(
            dataset_id=dataset_id, status='preparing', run_name='cleanup',
            train_params=json.dumps({
                'training_mode': 'full_transformer',
                'artifact_status': 'creating_repository',
            }))
        ct.db.session.add(run)
        ct.db.session.commit()
        with pytest.raises(RuntimeError, match='no GPU was rented'):
            ct._create_full_transformer_repo(
                run, 'hf-cloud-secret-x', _api=api)
        repo_id = ct._run_param(run, 'hf_repo_id')
        assert repo_id == f'tester/{ct._full_transformer_repo_name(run)}'
        assert ct._run_param(run, 'artifact_status') == 'repository_preparation_failed'
        assert api.deleted == [{'repo_id': repo_id, 'repo_type': 'model'}]


def test_cloud_token_accepts_global_write_and_requires_real_read_write_rights(
        ct, tmp_path):
    api, namespace, broad_access = ct._validate_full_transformer_token(
        'hf-classic-write', _api=_FakeHfApi(tmp_path, role='write'))
    assert namespace == 'tester'
    assert broad_access is True
    assert api.list_calls[-1]['repo_id'] == 'krea/Krea-2-Raw'

    with pytest.raises(ValueError, match='requires write access'):
        ct._validate_full_transformer_token(
            'hf-classic-read', _api=_FakeHfApi(tmp_path, role='read'))
    with pytest.raises(ValueError, match='delivery namespace'):
        ct._validate_full_transformer_token(
            'hf-fine-grained-without-write', _api=_FakeHfApi(
                tmp_path, permissions=['repo.content.read']))
    with pytest.raises(ValueError, match='cannot read'):
        ct._validate_full_transformer_token(
            'hf-no-base', _api=_FakeHfApi(
                tmp_path, failure=PermissionError('denied')))


def test_cloud_token_rejects_global_or_multiple_write_scopes_and_accepts_exact(
        ct, tmp_path):
    api, namespace, broad_access = ct._validate_full_transformer_token(
        'hf-exact', _api=_FakeHfApi(tmp_path))
    assert namespace == 'tester'
    assert broad_access is False
    assert api.list_calls[-1]['repo_id'] == 'krea/Krea-2-Raw'

    with pytest.raises(ValueError, match='global or broad'):
        ct._validate_full_transformer_token(
            'hf-global-write', _api=_FakeHfApi(
                tmp_path, global_permissions=['repo.write']))

    broad_scopes = [
        {
            'entity': {'type': 'model', 'name': 'krea/Krea-2-Raw'},
            'permissions': ['repo.content.read'],
        },
        {
            'entity': {'type': 'user', 'name': 'tester'},
            'permissions': ['repo.content.read', 'repo.write'],
        },
        {
            'entity': {'type': 'org', 'name': 'unrelated-org'},
            'permissions': ['repo.content.read', 'repo.write'],
        },
    ]
    with pytest.raises(ValueError, match='exactly one dedicated'):
        ct._validate_full_transformer_token(
            'hf-multiple-write', _api=_FakeHfApi(
                tmp_path, scopes=broad_scopes,
                orgs=[{'name': 'unrelated-org'}]))


def test_cloud_token_status_distinguishes_scoped_global_and_read_only(
        ct, tmp_path):
    scoped = ct.full_transformer_token_status(
        'hf-scoped', _api=_FakeHfApi(tmp_path))
    assert scoped['code'] == 'ready'
    assert scoped['severity'] == 'success'
    assert scoped['warning'] is None

    global_token = 'hf_global_NEVER_ECHO_THIS_VALUE'
    broad = ct.full_transformer_token_status(
        global_token, _api=_FakeHfApi(tmp_path, role='write'))
    assert broad['ok'] is True
    assert broad['code'] == 'broad_access'
    assert broad['severity'] == 'warning'
    assert broad['namespace'] == 'tester'
    assert 'global write access' in broad['warning'].lower()
    assert global_token not in str(broad)

    read_only = ct.full_transformer_token_status(
        'hf-read', _api=_FakeHfApi(tmp_path, role='read'))
    assert read_only['ok'] is False
    assert read_only['code'] == 'invalid'
    assert read_only['severity'] == 'error'
    assert 'write access' in read_only['error']


def test_candidate_token_status_uses_the_candidate_and_scrubs_it(
        ct, monkeypatch):
    candidate = 'hf_candidate_NEVER_ECHO_THIS_VALUE'
    saved = 'hf_saved_NEVER_ECHO_THIS_VALUE'
    seen = []

    def reject(token, _api=None, required_base_repo=None):
        seen.append(token)
        raise ValueError(f'rejected credential {token}')

    monkeypatch.setattr(ct, '_validate_full_transformer_token', reject)
    candidate_status = ct.full_transformer_token_status(candidate)
    assert seen == [candidate]
    assert candidate_status['code'] == 'invalid'
    assert candidate_status['configured'] is True
    assert candidate not in str(candidate_status)

    monkeypatch.setenv('HF_CLOUD_TOKEN', saved)
    saved_status = ct.full_transformer_token_preflight()
    assert seen[-1] == saved
    assert saved_status['code'] == 'invalid'
    assert saved not in str(saved_status)


def test_echoed_remote_hf_token_is_discarded_immediately(ct):
    received = []

    class Remote:
        def ensure_settings(self, hf_token):
            received.append(hf_token)
            return {
                'HF_TOKEN': hf_token,
                'DATASETS_FOLDER': '/datasets',
                'TRAINING_FOLDER': '/output',
            }

    run = SimpleNamespace(train_params=json.dumps({
        'training_mode': 'full_transformer'}))
    settings = ct._ensure_remote_settings_without_secret(run, Remote())
    assert received == ['hf-cloud-secret-x']
    assert settings == {
        'DATASETS_FOLDER': '/datasets', 'TRAINING_FOLDER': '/output'}
    assert 'hf-cloud-secret-x' not in json.dumps(settings)
    assert 'hf-general-must-not-reach-dense' not in json.dumps(settings)


def test_dense_start_timeout_after_remote_side_effect_keeps_pod(
        ct, app, dataset_id, tmp_path, monkeypatch):
    started = []
    stopped = []
    destroyed = []

    class Remote:
        def is_ready(self):
            return True

        def ensure_settings(self, hf_token):
            return {
                'HF_TOKEN': hf_token,
                'DATASETS_FOLDER': '/datasets',
                'TRAINING_FOLDER': '/output',
            }

        def upload_dataset(self, *args, **kwargs):
            return None

        def create_job(self, name, config):
            return 'remote-job'

        def start_job(self, job_id):
            # The remote accepted/started it, but the client timed out before
            # receiving the response.
            started.append(job_id)
            raise TimeoutError('start POST timed out after remote side effect')

        def stop_job(self, job_id):
            stopped.append(job_id)

    remote = Remote()
    staging = tmp_path / 'start-timeout'
    (staging / 'dataset').mkdir(parents=True)
    monkeypatch.setattr(ct, '_prepare_staging', lambda run: None)
    monkeypatch.setattr(ct, '_make_remote', lambda run: remote)
    monkeypatch.setattr(
        ct.vast_client, 'get_instance',
        lambda instance_id: {'actual_status': 'running'})
    monkeypatch.setattr(
        ct.vast_client, 'derive_base_url',
        lambda instance, port: 'https://dense-pod.invalid')
    monkeypatch.setattr(
        ct.vast_client, 'destroy_instance',
        lambda instance_id: destroyed.append(instance_id) or True)
    monkeypatch.setattr(ct.lt, 'build_job_config', lambda *a, **k: {})
    monkeypatch.setattr(ct, '_cloudify_job_config', lambda config, *a, **k: config)

    with app.app_context():
        run = ct.CloudTrainingRun(
            dataset_id=dataset_id, status='preparing', run_name='timeout',
            job_name='Krea_timeout', vast_label='lds-timeout',
            vast_instance_id='pod-timeout', staging_dir=str(staging),
            train_params=json.dumps({
                'training_mode': 'full_transformer',
                'train_type': 'krea', 'variant': 'base', 'steps': 500,
                'artifact_status': 'pending',
                'hf_repo_id': 'tester/Krea-timeout',
            }))
        ct.db.session.add(run)
        ct.db.session.commit()
        run_id = run.id

        ct._monitor(app, run_id)
        ct.db.session.expire_all()
        run = ct.db.session.get(ct.CloudTrainingRun, run_id)
        assert started == ['remote-job']
        assert stopped == ['remote-job']
        assert destroyed == []
        assert run.status == 'error_pod_kept'
        assert run.vast_instance_id == 'pod-timeout'
        assert 'timed out after remote side effect' in run.error


def test_already_live_job_db_failure_is_classified_post_start_and_keeps_pod(
        ct, app, dataset_id, tmp_path, monkeypatch):
    stopped = []
    destroyed = []

    class Remote:
        def is_ready(self):
            return True

        def get_job(self, job_id):
            return {'id': job_id, 'status': 'running', 'step': 3}

        def stop_job(self, job_id):
            stopped.append(job_id)

    staging = tmp_path / 'already-live-db-failure'
    staging.mkdir()
    monkeypatch.setattr(ct, '_prepare_staging', lambda run: None)
    monkeypatch.setattr(ct, '_make_remote', lambda run: Remote())
    monkeypatch.setattr(
        ct.vast_client, 'get_instance',
        lambda instance_id: {'actual_status': 'running'})
    monkeypatch.setattr(
        ct.vast_client, 'derive_base_url',
        lambda instance, port: 'https://dense-live.invalid')
    monkeypatch.setattr(
        ct.vast_client, 'destroy_instance',
        lambda instance_id: destroyed.append(instance_id) or True)
    original_set = ct._set
    failed = {'value': False}

    def fail_training_write(run, **fields):
        if fields.get('status') == 'training' and not failed['value']:
            failed['value'] = True
            raise RuntimeError('database write failed after observing live job')
        return original_set(run, **fields)

    monkeypatch.setattr(ct, '_set', fail_training_write)

    with app.app_context():
        run = ct.CloudTrainingRun(
            dataset_id=dataset_id, status='uploading', run_name='already-live',
            job_name='Krea_already_live', vast_instance_id='pod-already-live',
            remote_job_id='remote-already-live', staging_dir=str(staging),
            train_params=json.dumps({
                'training_mode': 'full_transformer',
                'artifact_status': 'pending',
                'hf_repo_id': 'tester/Krea-already-live',
            }))
        ct.db.session.add(run)
        ct.db.session.commit()
        run_id = run.id

        ct._monitor(app, run_id)
        ct.db.session.expire_all()
        run = ct.db.session.get(ct.CloudTrainingRun, run_id)
        assert failed['value'] is True
        assert stopped == ['remote-already-live']
        assert destroyed == []
        assert run.status == 'error_pod_kept'
        assert 'database write failed' in run.error


def test_supervisor_freeze_keeps_started_dense_pod(
        ct, app, dataset_id, monkeypatch):
    stopped = []
    destroyed = []

    class Remote:
        def stop_job(self, job_id):
            stopped.append(job_id)

    monkeypatch.setattr(ct, '_make_remote', lambda run: Remote())
    monkeypatch.setattr(ct, 'note_progress', lambda *a, **k: None)
    monkeypatch.setattr(ct, '_freeze_limit_seconds', lambda *a, **k: 1)
    monkeypatch.setattr(ct, '_silent_seconds', lambda *a, **k: 2)
    monkeypatch.setattr(
        ct.vast_client, 'destroy_instance',
        lambda instance_id: destroyed.append(instance_id) or True)

    with app.app_context():
        run = ct.CloudTrainingRun(
            dataset_id=dataset_id, status='training', run_name='freeze',
            job_name='Krea_freeze', vast_instance_id='dense-pod',
            remote_job_id='dense-job', base_url='https://pod.invalid',
            train_params=json.dumps({
                'training_mode': 'full_transformer',
                'artifact_status': 'pending',
            }))
        ct.db.session.add(run)
        ct.db.session.commit()
        run_id = run.id

        acted = ct.supervise_active_runs()
        run = ct.db.session.get(ct.CloudTrainingRun, run_id)
        assert acted == [{'run_id': run_id, 'reason': 'freeze',
                          'ok': True, 'pod_kept': True}]
        assert stopped == ['dense-job']
        assert destroyed == []
        assert run.status == 'error_pod_kept'
        assert run.vast_instance_id == 'dense-pod'
        ct._stop_events.pop(run_id, None)


def test_cloud_token_presence_is_exposed_but_value_never_is(
        client, monkeypatch):
    from app.services import cloud_training

    secret = 'hf_cloud_SUPERSECRET_NEVER_SERIALIZE'
    monkeypatch.delenv('HF_CLOUD_TOKEN', raising=False)
    monkeypatch.setattr(
        cloud_training,
        'full_transformer_token_status',
        lambda token, _api=None: {
            'ok': True, 'configured': True, 'code': 'ready',
            'namespace': 'tester', 'settings_focus': 'HF_CLOUD_TOKEN',
            'error': None,
        },
    )
    saved = client.put(
        '/api/settings', json={'secrets': {'HF_CLOUD_TOKEN': secret}})
    assert saved.status_code == 200
    assert saved.get_json()['secrets']['HF_CLOUD_TOKEN'] is True
    settings = client.get('/api/settings')
    diagnostic = client.get('/api/diagnostic')
    assert settings.get_json()['secrets']['HF_CLOUD_TOKEN'] is True
    assert diagnostic.get_json()['secrets_present']['HF_CLOUD_TOKEN'] is True
    assert secret not in saved.get_data(as_text=True)
    assert secret not in settings.get_data(as_text=True)
    assert secret not in diagnostic.get_data(as_text=True)


def test_lora_lane_remains_unchanged(ct):
    cloud_cfg = {'disk_gb': 60, 'full_transformer': {
        'disk_gb': 200, 'min_vram_gb': 80,
    }}
    assert ct._disk_gb_for(cloud_cfg, {'training_mode': 'lora'}) == 60
    out = ct._cloudify_job_config(
        _job_config(network=True), 'lora_job', 'C:/staging',
        {'DATASETS_FOLDER': '/datasets', 'TRAINING_FOLDER': '/output'},
        run_params={'training_mode': 'lora'})
    save = out['config']['process'][0]['save']
    assert save == {'dtype': 'bf16'}
    assert 'hf_repo_id' not in save


def test_dense_profile_defaults_are_explicit(app):
    from app import config as cfg
    assert 'HF_CLOUD_TOKEN' in cfg.SECRET_KEYS
    assert cfg.get('cloud.full_transformer.min_vram_gb') == 80
    assert cfg.get('cloud.full_transformer.disk_gb') == 200
    assert cfg.get('cloud.full_transformer.verification_attempts') == 3


# --- the base that reaches the POD ---------------------------------------------
#
# The recipe tests assert what build_job_config emits. These assert the rest of
# the chain — the stamped run params, the rebuild the monitor does minutes later
# at pod boot, and the cloud rewrite — because that is where a lifted refusal
# could still have produced a run named Turbo and trained on Raw.

_TURBO_SCOPES = [
    {'entity': {'type': 'model', 'name': 'krea/Krea-2-Turbo'},
     'permissions': ['repo.content.read']},
    {'entity': {'type': 'user', 'name': 'tester'},
     'permissions': ['repo.content.read', 'repo.write']},
]


def _pod_job(ct, run, ds, staging='/pod/ds'):
    """The exact chain the monitor runs at pod boot."""
    params = json.loads(run.train_params)
    job = ct.lt.build_job_config(
        ct._run_config_dataset(ds, params), staging,
        steps=params.get('steps') or 800, training_folder='__POD__')
    job = ct._cloudify_job_config(
        job, run.job_name, staging,
        {'DATASETS_FOLDER': '/datasets', 'TRAINING_FOLDER': '/output'},
        run_params=params)
    return job['config']['process'][0], params


def test_dense_turbo_launch_sends_turbo_to_the_pod(
        ct, app, dataset_id, monkeypatch, tmp_path):
    api = _FakeHfApi(tmp_path, scopes=_TURBO_SCOPES)
    monkeypatch.setattr(ct, '_make_hf_api', lambda token: api)
    from app.services import face_dataset_service as fds

    with app.app_context():
        result = ct.launch_cloud_training(
            'local', dataset_id, train_type='krea', variant='turbo',
            training_mode='full_transformer')
        run = ct.db.session.get(ct.CloudTrainingRun, result['run_id'])
        proc, params = _pod_job(ct, run, fds.get_dataset('local', dataset_id))

    # The variant is no longer overwritten with 'base' at launch...
    assert params['variant'] == 'turbo'
    # ...and the config the pod receives names Turbo.
    assert proc['model']['name_or_path'] == 'krea/Krea-2-Turbo'
    assert 'network' not in proc                      # still a dense run
    assert 'assistant_lora_path' not in proc['model']
    # The token was asked for the repository this run needs, not for Raw.
    assert any(call.get('repo_id') == 'krea/Krea-2-Turbo'
               for call in api.list_calls)


def test_a_raw_only_token_cannot_launch_a_turbo_dense_run(
        ct, app, dataset_id, monkeypatch, tmp_path):
    """The gate that keeps a 403 from landing on a rented GPU."""
    api = _FakeHfApi(tmp_path)          # default scopes: Krea-2-Raw only
    monkeypatch.setattr(ct, '_make_hf_api', lambda token: api)
    with app.app_context():
        with pytest.raises(ValueError, match='krea/Krea-2-Turbo'):
            ct.launch_cloud_training(
                'local', dataset_id, train_type='krea', variant='turbo',
                training_mode='full_transformer')
        assert ct.CloudTrainingRun.query.count() == 0


def test_dense_custom_base_rides_the_private_repo_to_the_pod(
        ct, app, dataset_id, monkeypatch, tmp_path):
    """The custom-base transport was already wired and mode-agnostic; what the
    dense lane could never obtain was the base_repo_id stamp. It can now."""
    from app.services import face_dataset_service as fds
    from app.services import hf_base_push as hbp

    api = _FakeHfApi(tmp_path)
    monkeypatch.setattr(ct, '_make_hf_api', lambda token: api)
    monkeypatch.setattr(hbp, 'require_base_repo', lambda *a, **k: {
        'repo_id': 'tester/lds-base-hdeadbeef', 'size_bytes': 20_000_000_000})
    readable = []
    monkeypatch.setattr(ct, '_assert_dense_custom_base_readable',
                        lambda repo, token, **k: readable.append((repo, token)))
    weights = tmp_path / 'my-krea.safetensors'
    weights.write_bytes(bytes(16))
    monkeypatch.setattr(ct.lt, 'preflight_custom_paths', lambda *a, **k: None)
    monkeypatch.setattr(ct.lt, 'assert_trainable_base_file', lambda p: {})

    with app.app_context():
        result = ct.launch_cloud_training(
            'local', dataset_id, train_type='krea', variant='base',
            base_model=str(weights), training_mode='full_transformer')
        run = ct.db.session.get(ct.CloudTrainingRun, result['run_id'])
        proc, params = _pod_job(ct, run, fds.get_dataset('local', dataset_id))

    assert params['base_repo_id'] == 'tester/lds-base-hdeadbeef'
    assert params['base_size_bytes'] == 20_000_000_000
    # The pod pulls the private repo, by the exact filename it will find there…
    assert proc['model']['name_or_path'] == 'tester/lds-base-hdeadbeef'
    assert proc['model']['model_kwargs']['checkpoint_filename'] == (
        'hdeadbeef.safetensors')
    # …and the dense VAE survives the rewrite.
    assert proc['model']['model_kwargs']['vae_path'] == 'Qwen/Qwen-Image-2512'
    # The pod's own credential was proven able to read that repository.
    assert readable == [('tester/lds-base-hdeadbeef', 'hf-cloud-secret-x')]


def test_the_pod_credential_must_be_able_to_read_the_custom_base(ct, monkeypatch):
    """A 401/403 refuses the launch with its reason; anything else fails OPEN —
    an HF outage must never block a launch that would have worked."""
    import urllib.error
    import urllib.request

    codes = []

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, codes.pop(0), 'no', {}, None)

    monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
    codes.append(403)
    with pytest.raises(ValueError, match='HF_CLOUD_TOKEN cannot read'):
        ct._assert_dense_custom_base_readable('tester/lds-base-h1', 'tok')
    codes.append(404)
    ct._assert_dense_custom_base_readable('tester/lds-base-h1', 'tok')
    # No repository to check at all is never an error.
    ct._assert_dense_custom_base_readable(None, 'tok')


def test_the_model_card_names_the_base_the_run_actually_used(ct):
    assert 'base_model: krea/Krea-2-Turbo' in ct._full_transformer_readme(
        'tester/Krea-2-full-7-x', 'krea/Krea-2-Turbo')
    assert 'krea/Krea-2-Turbo`' in ct._full_transformer_readme(
        'tester/Krea-2-full-7-x', 'krea/Krea-2-Turbo')
    # …resolved from the run's own stamped params, never from a constant.
    assert ct._dense_base_repo_for({'variant': 'turbo'}) == 'krea/Krea-2-Turbo'
    assert ct._dense_base_repo_for({'variant': 'base'}) == 'krea/Krea-2-Raw'
    assert ct._dense_base_repo_for({}) == 'krea/Krea-2-Raw'
    assert ct._dense_base_repo_for(
        {'variant': 'turbo', 'base_repo_id': 'tester/lds-base-h9'}
    ) == 'tester/lds-base-h9'
    # The licence PDF only ever comes from an OFFICIAL Krea repository.
    assert ct._krea_license_source('tester/lds-base-h9') == 'krea/Krea-2-Raw'
    assert ct._krea_license_source('krea/Krea-2-Turbo') == 'krea/Krea-2-Turbo'


def test_the_licence_is_fetched_from_whichever_krea_repo_the_token_can_read(ct):
    """A custom-base run needs no official read scope at all, so the token may
    legitimately be scoped to Turbo, to Raw, or to neither. The licence still
    has to travel with the derivative — so try both, and fail before renting
    only if neither answers."""
    class _OneRepoOnly:
        def __init__(self, readable):
            self.readable = readable
            self.asked = []

        def hf_hub_download(self, **kwargs):
            self.asked.append(kwargs['repo_id'])
            if kwargs['repo_id'] != self.readable:
                raise RuntimeError('403')
            return __import__('tempfile').mkstemp()[1]

    turbo_only = _OneRepoOnly('krea/Krea-2-Turbo')
    ct._krea_license_bytes(turbo_only, 'tester/lds-base-h9')
    assert turbo_only.asked == ['krea/Krea-2-Raw', 'krea/Krea-2-Turbo']

    raw_only = _OneRepoOnly('krea/Krea-2-Raw')
    ct._krea_license_bytes(raw_only, 'krea/Krea-2-Turbo')
    assert raw_only.asked == ['krea/Krea-2-Turbo', 'krea/Krea-2-Raw']

    with pytest.raises(Exception):
        ct._krea_license_bytes(_OneRepoOnly('nobody/nothing'), 'krea/Krea-2-Raw')
