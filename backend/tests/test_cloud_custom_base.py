"""Custom weights on the CLOUD lane via a one-time push to a PRIVATE HF repo.

Contract under test (hf_base_push + cloud_training seams):
  a) deterministic repo name = lds-base-<combo hash> (the existing custom-run
     cache key), and per-family weight filenames matching what the POD's
     ai-toolkit loaders actually fetch (Klein's hardcoded per-size names,
     Krea's derived-filename fallback);
  b) push: private=True FORCED (this file fails if anyone ever makes it
     public/togglable), read-only token refused before upload, cache-hit skips
     the upload entirely, zimage pushes the CONVERTED diffusers folder;
  c) launch: the refusal is lifted ONLY for zimage/krea/flux2klein — the launch
     stamps base_model + base_repo_id + base_size_bytes into train_params (so
     the monitor rebuild and retry/continue replay it), names the run with the
     combo-hash suffix, and fails actionably without HF_TOKEN / without a
     pushed repo;
  d) _cloudify_job_config routes model.name_or_path to the private repo (the
     symmetric seam to the dataset path swap); official runs stay bit-for-bit;
  e) _provision bumps disk_gb from the stamped base size (official unchanged).

HfApi is always faked — no network, ever."""
import json
import os
import struct
import types

import pytest


# --- fake .safetensors (same shape as test_custom_base_paths) -------------------

def _write_safetensors(path, keys):
    meta, off = {}, 0
    for k in keys:
        meta[k] = {'dtype': 'F32', 'shape': [1], 'data_offsets': [off, off + 4]}
        off += 4
    header = json.dumps(meta).encode('utf-8')
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<Q', len(header)))
        fh.write(header)
        fh.write(b'\x00' * off)
    return str(path)


_KREA_KEYS = ['first.weight', 'blocks.0.attn.qkv.weight', 'txtfusion.0.attn.qkv.weight',
              'tmlp.0.weight', 'last.linear.weight']
_FLUX_KEYS = ['double_blocks.0.img_attn.qkv.weight', 'single_blocks.0.linear1.weight',
              'img_in.weight', 'txt_in.weight', 'final_layer.linear.weight']


# --- fake HfApi ------------------------------------------------------------------

class _HttpErr(Exception):
    def __init__(self, status):
        super().__init__(f'HTTP {status}')
        self.response = types.SimpleNamespace(status_code=status)


class _FakeApi:
    """Stateful HfApi stand-in: repo existence flips on create_repo, uploads
    register the paths/sizes readiness checks read back."""

    def __init__(self, who=None, exists=False, private=True, paths=None):
        self.calls = []
        self.who = who or {'name': 'tester',
                           'auth': {'accessToken': {'role': 'write'}}}
        self.exists = exists
        self.private = private
        self.paths = dict(paths or {})            # path_in_repo -> size
        self.update_settings_raises = False

    def whoami(self):
        return self.who

    def repo_info(self, repo_id, repo_type=None):
        self.calls.append(('repo_info', repo_id))
        if not self.exists:
            raise _HttpErr(404)
        return types.SimpleNamespace(private=self.private)

    def get_paths_info(self, repo_id, paths, repo_type=None):
        self.calls.append(('get_paths_info', tuple(paths)))
        if not self.exists:
            raise _HttpErr(404)
        return [types.SimpleNamespace(path=p, size=self.paths[p], lfs=None)
                for p in paths if p in self.paths]

    def create_repo(self, **kw):
        self.calls.append(('create_repo', kw))
        self.exists = True

    def update_repo_settings(self, **kw):
        self.calls.append(('update_repo_settings', kw))
        if self.update_settings_raises:
            raise _HttpErr(403)
        self.private = bool(kw.get('private'))

    def upload_file(self, **kw):
        self.calls.append(('upload_file', kw))
        import os
        self.paths[kw['path_in_repo']] = os.path.getsize(kw['path_or_fileobj'])

    def upload_folder(self, **kw):
        self.calls.append(('upload_folder', kw))
        import os
        root = kw['folder_path']
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                full = os.path.join(dirpath, f)
                rel = os.path.relpath(full, root).replace('\\', '/')
                self.paths[rel] = os.path.getsize(full)

    def names(self):
        return [c[0] for c in self.calls]


def _mkds(app, name='CB', trigger='zc_cb', train_type='krea'):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, name, trigger, train_type=train_type)
        return ds.id


# --- a) naming contract ----------------------------------------------------------

def test_base_repo_name_matches_run_tag_hash(app, tmp_path):
    from app.services import hf_base_push as hbp
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'N', 'zc_n', train_type='krea')
        base = str(tmp_path / 'merge.safetensors')
        tag = lt._custom_combo_hash(ds, base, 'krea')
        assert tag.startswith('_h')
        assert hbp.base_repo_name(ds, 'krea', base) == 'lds-base-' + tag.lstrip('_')
        # official base -> nothing to push
        with pytest.raises(hbp.HfPublishError):
            hbp.base_repo_name(ds, 'krea', '')


def test_weight_filenames_match_pod_loaders():
    """Klein: ai-toolkit HARDCODES the per-size filename; Krea: the name must
    ALSO satisfy ai-toolkit's derived-filename fallback (repo tail after the
    last dash + .safetensors) so even a pod ignoring checkpoint_filename
    resolves it; zimage is a folder (no single filename)."""
    from app.services import hf_base_push as hbp
    repo = 'lds-base-h1234abcd'
    assert hbp.weight_filename('flux2klein', '4b', repo) == 'flux-2-klein-base-4b.safetensors'
    assert hbp.weight_filename('flux2klein', '9b', repo) == 'flux-2-klein-base-9b.safetensors'
    krea = hbp.weight_filename('krea', 'base', repo)
    assert krea == 'h1234abcd.safetensors'
    derived = repo.split('/')[-1].split('-')[-1].lower() + '.safetensors'
    assert krea == derived
    assert hbp.weight_filename('zimage', 'turbo', repo) is None
    assert hbp.expected_repo_files('zimage', 'turbo', repo) == [
        'transformer/config.json', 'transformer/diffusion_pytorch_model.safetensors']


# --- b) push ----------------------------------------------------------------------

def test_push_forces_private_repo_and_expected_filename(app, tmp_path):
    """THE privacy invariant: create_repo is called with private=True — this
    test fails the day anyone makes it public or togglable."""
    from app.services import hf_base_push as hbp
    base = _write_safetensors(tmp_path / 'k.safetensors', _KREA_KEYS)
    ds_id = _mkds(app)
    api = _FakeApi()
    with app.app_context():
        res = hbp.push_base_to_hf(ds_id, 'krea', 'base', base, 'hf_tok', _api=api)
    create = next(kw for name, kw in api.calls if name == 'create_repo')
    assert create['private'] is True
    assert create['repo_type'] == 'model'
    upload = next(kw for name, kw in api.calls if name == 'upload_file')
    assert upload['path_in_repo'].endswith('.safetensors')
    assert upload['path_in_repo'] == res['repo_id'].split('/')[-1].split('-')[-1] + '.safetensors'
    assert res['ok'] is True and res['cached'] is False
    assert res['repo_id'].startswith('tester/lds-base-h')


def test_push_cache_hit_skips_upload(app, tmp_path):
    from app.services import hf_base_push as hbp
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    base = _write_safetensors(tmp_path / 'k.safetensors', _KREA_KEYS)
    ds_id = _mkds(app)
    with app.app_context():
        ds = svc.get_dataset(LOCAL_USER, ds_id)
        repo_name = hbp.base_repo_name(ds, 'krea', base)
    fname = hbp.weight_filename('krea', 'base', repo_name)
    import os
    api = _FakeApi(exists=True, paths={fname: os.path.getsize(base)})
    with app.app_context():
        res = hbp.push_base_to_hf(ds_id, 'krea', 'base', base, 'hf_tok', _api=api)
    assert res['cached'] is True
    assert 'upload_file' not in api.names()
    assert 'create_repo' not in api.names()


def test_push_size_drift_reuploads(app, tmp_path):
    """Same repo, but the local file changed since the last push -> NOT a
    cache-hit: the file is uploaded again over the stale copy."""
    from app.services import hf_base_push as hbp
    base = _write_safetensors(tmp_path / 'k.safetensors', _KREA_KEYS)
    ds_id = _mkds(app)
    api = _FakeApi(exists=True, paths={'placeholder': 1})
    # register the expected filename with a WRONG size
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.get_dataset(LOCAL_USER, ds_id)
        fname = hbp.weight_filename('krea', 'base',
                                    hbp.base_repo_name(ds, 'krea', base))
    api.paths = {fname: 1}
    with app.app_context():
        res = hbp.push_base_to_hf(ds_id, 'krea', 'base', base, 'hf_tok', _api=api)
    assert res['cached'] is False
    assert 'upload_file' in api.names()


def test_push_refuses_read_only_token_before_upload(app, tmp_path):
    from app.services import hf_base_push as hbp
    base = _write_safetensors(tmp_path / 'k.safetensors', _KREA_KEYS)
    ds_id = _mkds(app)
    api = _FakeApi(who={'name': 'tester', 'auth': {'accessToken': {'role': 'read'}}})
    with app.app_context():
        with pytest.raises(hbp.HfPublishError) as e:
            hbp.push_base_to_hf(ds_id, 'krea', 'base', base, 'hf_tok', _api=api)
    assert e.value.code == 'read_only_token'
    assert 'write token' in e.value.message
    assert 'upload_file' not in api.names()
    assert 'create_repo' not in api.names()


def test_push_repairs_public_repo_or_refuses(app, tmp_path):
    from app.services import hf_base_push as hbp
    import os
    base = _write_safetensors(tmp_path / 'k.safetensors', _KREA_KEYS)
    ds_id = _mkds(app)
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.get_dataset(LOCAL_USER, ds_id)
        fname = hbp.weight_filename('krea', 'base',
                                    hbp.base_repo_name(ds, 'krea', base))
    # cache-hit on a repo that drifted PUBLIC -> flipped back to private
    api = _FakeApi(exists=True, private=False, paths={fname: os.path.getsize(base)})
    with app.app_context():
        hbp.push_base_to_hf(ds_id, 'krea', 'base', base, 'hf_tok', _api=api)
    fix = next(kw for name, kw in api.calls if name == 'update_repo_settings')
    assert fix['private'] is True
    # …and when the flip FAILS, the push refuses instead of using a public repo
    api2 = _FakeApi(exists=True, private=False, paths={fname: os.path.getsize(base)})
    api2.update_settings_raises = True
    with app.app_context():
        with pytest.raises(hbp.HfPublishError) as e:
            hbp.push_base_to_hf(ds_id, 'krea', 'base', base, 'hf_tok', _api=api2)
    assert e.value.code == 'repo_public'


def test_push_zimage_requires_conversion_then_uploads_folder(app, tmp_path, monkeypatch):
    from app.services import hf_base_push as hbp
    from app.services import zimage_convert as zc
    ds_id = _mkds(app, train_type='zimage', trigger='zc_zi')
    base = 'z image\\merge.safetensors'
    monkeypatch.setattr(zc, 'is_converted', lambda z: False)
    with app.app_context():
        with pytest.raises(hbp.HfPublishError) as e:
            hbp.push_base_to_hf(ds_id, 'zimage', 'turbo', base, 'hf_tok',
                                _api=_FakeApi())
    assert e.value.code == 'not_converted'
    # converted -> the DIFFUSERS FOLDER is the payload
    conv = tmp_path / 'converted' / 'merge'
    (conv / 'transformer').mkdir(parents=True)
    (conv / 'transformer' / 'config.json').write_text('{}', encoding='utf-8')
    (conv / 'transformer' / 'diffusion_pytorch_model.safetensors').write_bytes(b'w' * 64)
    monkeypatch.setattr(zc, 'is_converted', lambda z: True)
    monkeypatch.setattr(zc, 'converted_dir', lambda z: str(conv))
    api = _FakeApi()
    with app.app_context():
        res = hbp.push_base_to_hf(ds_id, 'zimage', 'turbo', base, 'hf_tok', _api=api)
    upload = next(kw for name, kw in api.calls if name == 'upload_folder')
    assert upload['folder_path'] == str(conv)
    assert res['ok'] is True
    assert 'transformer/diffusion_pytorch_model.safetensors' in api.paths


def test_push_missing_local_file_is_actionable(app, tmp_path):
    from app.services import hf_base_push as hbp
    ds_id = _mkds(app)
    with app.app_context():
        with pytest.raises(hbp.HfPublishError) as e:
            hbp.push_base_to_hf(ds_id, 'krea', 'base',
                                str(tmp_path / 'gone.safetensors'), 'hf_tok',
                                _api=_FakeApi())
    assert e.value.code == 'weights_missing'


def test_push_arch_sniff_is_confirmable(app, tmp_path):
    """A file whose header does not look like the family's arch raises the
    CUSTOM_WEIGHTS_UNVERIFIED marker (confirm-and-retry contract) unless
    allow_unverified_weights."""
    from app.services import hf_base_push as hbp
    base = _write_safetensors(tmp_path / 'x.safetensors', _FLUX_KEYS)  # flux keys on krea
    ds_id = _mkds(app)
    with app.app_context():
        with pytest.raises(ValueError, match='CUSTOM_WEIGHTS_UNVERIFIED'):
            hbp.push_base_to_hf(ds_id, 'krea', 'base', base, 'hf_tok',
                                _api=_FakeApi())
        res = hbp.push_base_to_hf(ds_id, 'krea', 'base', base, 'hf_tok',
                                  _api=_FakeApi(), allow_unverified_weights=True)
    assert res['ok'] is True


# --- launch-time guard -------------------------------------------------------------

def test_require_base_repo_actionable_errors(app, tmp_path, monkeypatch):
    from app.services import hf_base_push as hbp
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    import os
    base = _write_safetensors(tmp_path / 'k.safetensors', _KREA_KEYS)
    ds_id = _mkds(app)
    with app.app_context():
        ds = svc.get_dataset(LOCAL_USER, ds_id)
        repo_name = hbp.base_repo_name(ds, 'krea', base)
        fname = hbp.weight_filename('krea', 'base', repo_name)
        # no token
        with pytest.raises(ValueError, match='HF_TOKEN'):
            hbp.require_base_repo(ds, 'krea', 'base', base, None)
        # not pushed yet
        monkeypatch.setattr(hbp, '_make_api', lambda tok: _FakeApi())
        with pytest.raises(ValueError, match='not on your Hugging Face account yet'):
            hbp.require_base_repo(ds, 'krea', 'base', base, 'hf_tok')
        # repo exists but the variant's file is missing
        monkeypatch.setattr(hbp, '_make_api',
                            lambda tok: _FakeApi(exists=True, paths={'other': 5}))
        with pytest.raises(ValueError, match='missing'):
            hbp.require_base_repo(ds, 'krea', 'base', base, 'hf_tok')
        # size drift between the local file and the pushed copy
        monkeypatch.setattr(hbp, '_make_api',
                            lambda tok: _FakeApi(exists=True, paths={fname: 1}))
        with pytest.raises(ValueError, match='differs'):
            hbp.require_base_repo(ds, 'krea', 'base', base, 'hf_tok')
        # ready -> repo id + the REMOTE size (what the pod downloads)
        size = os.path.getsize(base)
        monkeypatch.setattr(hbp, '_make_api',
                            lambda tok: _FakeApi(exists=True, paths={fname: size}))
        out = hbp.require_base_repo(ds, 'krea', 'base', base, 'hf_tok')
        assert out == {'repo_id': f'tester/{repo_name}', 'size_bytes': size}
        # sdxl stays refused with the historical message
        with pytest.raises(ValueError, match='local-only'):
            hbp.require_base_repo(ds, 'sdxl', None, base, 'hf_tok')


# --- c) cloud launch ---------------------------------------------------------------

@pytest.fixture()
def ct(app, monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    from app.services import cloud_training
    monkeypatch.setattr(cloud_training, '_start_monitor', lambda *a, **k: None)
    monkeypatch.setattr(cloud_training, '_reconcile_before_launch', lambda a: None)
    return cloud_training


def _fake_export(monkeypatch, ct):
    monkeypatch.setattr(ct.lt, 'export_dataset_to_aitoolkit',
                        lambda uid, did, masked=True, dest_dir=None: dest_dir)
    monkeypatch.setattr(ct.lt, 'default_steps', lambda ds, **kw: 1200)
    monkeypatch.setattr(ct.lt, 'assert_trainable', lambda *a, **kw: None)


REPO = {'repo_id': 'tester/lds-base-hdeadbeef', 'size_bytes': 18_000_000_000}


@pytest.mark.parametrize('fam,variant,base_kind', [
    ('krea', 'base', 'abs'),
    ('flux2klein', '9b', 'abs'),
    ('zimage', 'turbo', 'merge'),
])
def test_launch_custom_base_stamps_repo_per_family(ct, app, tmp_path, monkeypatch,
                                                   fam, variant, base_kind):
    from app.services import hf_base_push as hbp
    _fake_export(monkeypatch, ct)
    seen = {}

    def fake_require(ds, family, var, base_model, token):
        seen.update(family=family, variant=var, base=base_model, token=token)
        return dict(REPO)

    monkeypatch.setattr(hbp, 'require_base_repo', fake_require)
    monkeypatch.setenv('HF_TOKEN', 'hf_tok')
    base = (str(tmp_path / f'{fam}.safetensors') if base_kind == 'abs'
            else 'z image\\merge.safetensors')
    ds_id = _mkds(app, name=f'L{fam}', trigger=f'zc_l{fam}', train_type=fam)
    with app.app_context():
        res = ct.launch_cloud_training('local', ds_id, train_type=fam,
                                       variant=variant, base_model=base)
        run = ct.db.session.get(ct.CloudTrainingRun, res['run_id'])
        params = json.loads(run.train_params)
    assert seen['family'] == fam and seen['base'] == base
    assert seen['token'] == 'hf_tok'
    assert params['base_model'] == base
    assert params['base_repo_id'] == REPO['repo_id']
    assert params['base_size_bytes'] == REPO['size_bytes']
    # the run folder carries the combo-hash isolation suffix, like local runs
    assert '_h' in run.run_name
    if fam == 'zimage':
        assert params['effective_base'] == base       # custom recipe stamped


def test_launch_zimage_custom_deturbo_needs_confirm(ct, app, monkeypatch):
    from app.services import hf_base_push as hbp
    _fake_export(monkeypatch, ct)
    monkeypatch.setattr(hbp, 'require_base_repo',
                        lambda *a, **kw: dict(REPO))
    monkeypatch.setenv('HF_TOKEN', 'hf_tok')
    ds_id = _mkds(app, name='ZC', trigger='zc_zc', train_type='zimage')
    with app.app_context():
        with pytest.raises(ValueError, match='CUSTOM_WEIGHTS_UNVERIFIED'):
            ct.launch_cloud_training('local', ds_id, train_type='zimage',
                                     variant='deturbo',
                                     base_model='z image\\m.safetensors')
        res = ct.launch_cloud_training('local', ds_id, train_type='zimage',
                                       variant='deturbo',
                                       base_model='z image\\m.safetensors',
                                       allow_unverified_weights=True)
        params = json.loads(ct.db.session.get(
            ct.CloudTrainingRun, res['run_id']).train_params)
    assert params['allow_unverified_weights'] is True


def test_launch_official_base_params_unchanged(ct, app, monkeypatch):
    """Zero-regression guard: an official launch stamps base_model '' and no
    repo fields, and never consults hf_base_push."""
    from app.services import hf_base_push as hbp
    _fake_export(monkeypatch, ct)
    monkeypatch.setattr(hbp, 'require_base_repo',
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError('called')))
    ds_id = _mkds(app, name='OF', trigger='zc_of', train_type='krea')
    with app.app_context():
        res = ct.launch_cloud_training('local', ds_id, train_type='krea',
                                       base_model='')
        params = json.loads(ct.db.session.get(
            ct.CloudTrainingRun, res['run_id']).train_params)
    assert params['base_model'] == ''
    assert 'base_repo_id' not in params
    assert 'base_size_bytes' not in params


def test_retry_replays_custom_base_and_confirm(ct, app, monkeypatch):
    """Retry/continue must carry the custom base to the fresh launch — repo id
    included via re-verification, confirm flag replayed like the caption ones."""
    ds_id = _mkds(app, name='RT', trigger='zc_rt', train_type='krea')
    captured = {}

    def fake_launch(user_id, dataset_id, **kw):
        captured.update(kw)
        return {'run_id': 999, 'status': 'preparing'}

    monkeypatch.setattr(ct, 'launch_cloud_training', fake_launch)
    with app.app_context():
        run = ct.CloudTrainingRun(
            dataset_id=ds_id, status='error', run_name='r', job_name='j',
            train_params=json.dumps({
                'train_type': 'krea', 'variant': 'base', 'steps': 1500,
                'base_model': 'C:\\models\\k.safetensors',
                'base_repo_id': REPO['repo_id'], 'masked': True,
                'allow_unverified_weights': True}))
        ct.db.session.add(run)
        ct.db.session.commit()
        ct.retry_cloud_run('local', run.id)
    assert captured['base_model'] == 'C:\\models\\k.safetensors'
    assert captured['allow_unverified_weights'] is True


# --- d) pod config routing ----------------------------------------------------------

_POD = {'DATASETS_FOLDER': '/root/ai-toolkit/datasets',
        'TRAINING_FOLDER': '/root/ai-toolkit/output'}


def _job(model):
    return {'config': {'name': 'lora_x', 'process': [{
        'type': 'sd_trainer', 'training_folder': '__POD__',
        'datasets': [{'folder_path': 'C:\\staging\\dataset'}],
        'model': model}]}}


def test_cloudify_routes_custom_base_to_private_repo(ct):
    params = {'train_type': 'krea', 'variant': 'base',
              'base_repo_id': 'tester/lds-base-hdeadbeef'}
    out = ct._cloudify_job_config(
        _job({'arch': 'krea2', 'name_or_path': 'C:\\models\\k.safetensors'}),
        'job1', 'C:\\staging\\dataset', _POD, run_params=params)
    model = out['config']['process'][0]['model']
    assert model['name_or_path'] == 'tester/lds-base-hdeadbeef'
    # krea's loader pulls ONE file from the repo — pinned explicitly
    assert model['model_kwargs']['checkpoint_filename'] == 'hdeadbeef.safetensors'


def test_cloudify_klein_keeps_model_kwargs_and_routes_repo(ct):
    params = {'train_type': 'flux2klein', 'variant': '9b',
              'base_repo_id': 'tester/lds-base-hdeadbeef'}
    out = ct._cloudify_job_config(
        _job({'arch': 'flux2_klein_9b', 'name_or_path': 'C:\\models\\f2.safetensors',
              'model_kwargs': {'match_target_res': False}}),
        'job1', 'C:\\staging\\dataset', _POD, run_params=params)
    model = out['config']['process'][0]['model']
    assert model['name_or_path'] == 'tester/lds-base-hdeadbeef'
    # the Klein loader hardcodes its per-size filename — no checkpoint_filename,
    # and the arch's own kwargs survive untouched
    assert model['model_kwargs'] == {'match_target_res': False}


def test_cloudify_zimage_routes_repo_and_keeps_extras(ct):
    params = {'train_type': 'zimage', 'variant': 'turbo',
              'base_repo_id': 'tester/lds-base-hdeadbeef'}
    out = ct._cloudify_job_config(
        _job({'arch': 'zimage', 'name_or_path': 'F:\\aitoolkit\\converted\\merge',
              'extras_name_or_path': 'Tongyi-MAI/Z-Image-Turbo',
              'assistant_lora_path': 'ostris/adapter.safetensors'}),
        'job1', 'C:\\staging\\dataset', _POD, run_params=params)
    model = out['config']['process'][0]['model']
    assert model['name_or_path'] == 'tester/lds-base-hdeadbeef'
    assert model['extras_name_or_path'] == 'Tongyi-MAI/Z-Image-Turbo'
    assert model['assistant_lora_path'] == 'ostris/adapter.safetensors'
    assert 'model_kwargs' not in model


def test_cloudify_official_untouched(ct):
    """Bit-for-bit regression guard for official runs (no base_repo_id)."""
    params = {'train_type': 'krea', 'variant': 'base', 'base_model': ''}
    out = ct._cloudify_job_config(
        _job({'arch': 'krea2', 'name_or_path': 'krea/Krea-2-Raw'}),
        'job1', 'C:\\staging\\dataset', _POD, run_params=params)
    model = out['config']['process'][0]['model']
    assert model == {'arch': 'krea2', 'name_or_path': 'krea/Krea-2-Raw'}


# --- e) pod disk sizing --------------------------------------------------------------

def test_disk_gb_bumped_for_large_custom_base(ct):
    assert ct._disk_gb_for({'disk_gb': 60}, {}) == 60
    assert ct._disk_gb_for({'disk_gb': 60}, {'base_size_bytes': 0}) == 60
    # 18 GB Klein 9B custom -> 2x + 30 headroom = 66
    assert ct._disk_gb_for({'disk_gb': 60},
                           {'base_size_bytes': 18_000_000_000}) == 66
    # small base never SHRINKS the configured disk
    assert ct._disk_gb_for({'disk_gb': 60},
                           {'base_size_bytes': 5_000_000_000}) == 60
    # a larger configured value always wins
    assert ct._disk_gb_for({'disk_gb': 100},
                           {'base_size_bytes': 18_000_000_000}) == 100
    # corrupt stamp degrades to the configured value, never a crash
    assert ct._disk_gb_for({'disk_gb': 60}, {'base_size_bytes': 'x'}) == 60


# --- routes ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_capabilities_cache():
    from app import capabilities
    capabilities._cache = None
    capabilities._cache_ts = 0.0
    yield


def _mkds_client(client):
    return client.post('/api/dataset/create',
                       json={'name': 'Lola', 'trigger_word': 'lola'}).get_json()['id']


def test_custom_base_status_route(client, monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    ds = _mkds_client(client)
    seen = {}

    def fake_state(user_id, dataset_id, family, variant, base_model, token, _api=None):
        seen.update(family=family, variant=variant, base_model=base_model)
        return {'supported': True, 'ready': True, 'repo_id': 'tester/lds-base-h1'}

    monkeypatch.setattr('app.services.hf_base_push.base_push_state', fake_state)
    r = client.get(f'/api/dataset/{ds}/train/cloud/custom-base'
                   '?train_type=krea&variant=base&base_model=C%3A%5Cm%5Ck.safetensors')
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is True and body['ready'] is True
    assert seen['family'] == 'krea' and seen['base_model'] == 'C:\\m\\k.safetensors'


def test_custom_base_push_route_requires_token_and_forwards(client, monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    ds = _mkds_client(client)
    r = client.post(f'/api/dataset/{ds}/train/cloud/custom-base/push',
                    json={'train_type': 'krea', 'base_model': 'C:\\m\\k.safetensors'})
    assert r.status_code == 400
    assert 'HF_TOKEN' in r.get_json()['error']

    monkeypatch.setenv('HF_TOKEN', 'hf_tok')
    seen = {}

    def fake_start(app, dataset_id, family, variant, base_model, token,
                   user_id='local', allow_unverified_weights=False):
        seen.update(dataset_id=dataset_id, family=family, base_model=base_model,
                    allow=allow_unverified_weights, token=token)
        return {'state': 'running', 'repo_name': 'lds-base-h1'}

    monkeypatch.setattr('app.services.hf_base_push.start_push', fake_start)
    r = client.post(f'/api/dataset/{ds}/train/cloud/custom-base/push',
                    json={'train_type': 'krea', 'variant': 'base',
                          'base_model': 'C:\\m\\k.safetensors',
                          'allow_unverified_weights': True})
    assert r.status_code == 200
    assert r.get_json()['state'] == 'running'
    assert seen['allow'] is True and seen['token'] == 'hf_tok'


def test_cloud_train_route_forwards_unverified_flag(client, monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    ds = _mkds_client(client)
    seen = {}

    def fake_launch(user_id, dataset_id, **kw):
        seen.update(kw)
        return {'run_id': 1, 'status': 'preparing', 'job_name': 'j', 'steps': 1200}

    monkeypatch.setattr('app.services.cloud_training.launch_cloud_training', fake_launch)
    client.post(f'/api/dataset/{ds}/train/cloud',
                json={'train_type': 'krea', 'base_model': 'C:\\m\\k.safetensors',
                      'allow_unverified_weights': True})
    assert seen['base_model'] == 'C:\\m\\k.safetensors'
    assert seen['allow_unverified_weights'] is True


# --- _resolve_merge: cross-drive junction (regression) --------------------------
#
# ComfyUI's models/ often lives on a second drive via a Windows junction (big
# weights don't fit the system disk). realpath() follows that junction, so the
# resolved target legitimately lands on another drive than models/'s own path.
# The old confinement check ran commonpath() on that resolved path, which raises
# `ValueError: Paths don't have the same drive` -> red toast, conversion dead.

def _patch_models_root(monkeypatch, root):
    """cfg.comfyui_dir('models') -> `root`, without touching real config."""
    monkeypatch.setattr('app.config.comfyui_dir',
                        lambda kind: root if kind == 'models' else None)


def test_resolve_merge_follows_cross_drive_junction(monkeypatch):
    from app.services import zimage_convert as zc
    _patch_models_root(monkeypatch, r'C:\models')
    # models/ is on C:, but the file resolves onto D: (junction to a 2nd drive).
    real_target = r'D:\weights\unet\z image\merge.safetensors'

    def fake_realpath(p):
        p = os.path.normpath(p)
        return real_target if p.lower().endswith('merge.safetensors') else p

    monkeypatch.setattr(os.path, 'realpath', fake_realpath)
    monkeypatch.setattr(os.path, 'isfile', lambda p: p == real_target)

    # Old code raised ValueError here; the fix returns the real cross-drive path.
    assert zc._resolve_merge(r'z image\merge.safetensors') == real_target


def test_resolve_merge_same_drive_still_works(monkeypatch):
    from app.services import zimage_convert as zc
    _patch_models_root(monkeypatch, r'C:\models')
    target = r'C:\models\unet\z image\merge.safetensors'
    monkeypatch.setattr(os.path, 'realpath', lambda p: os.path.normpath(p))
    monkeypatch.setattr(os.path, 'isfile', lambda p: os.path.normpath(p) == target)
    assert zc._resolve_merge(r'z image\merge.safetensors') == target


def test_resolve_merge_confinement_rejects_traversal(monkeypatch):
    from app.services import zimage_convert as zc
    _patch_models_root(monkeypatch, r'C:\models')
    # A forged z_model must never escape models/ — every isfile() is True to
    # prove the guard, not the filesystem, is what rejects these.
    monkeypatch.setattr(os.path, 'realpath', lambda p: os.path.normpath(p))
    monkeypatch.setattr(os.path, 'isfile', lambda p: True)
    assert zc._resolve_merge(r'..\..\Windows\evil.safetensors') is None
    assert zc._resolve_merge(r'D:\weights\evil.safetensors') is None
    assert zc._resolve_merge('') is None
