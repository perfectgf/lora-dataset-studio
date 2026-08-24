"""Hugging Face PRIVATE storage: the pre-check, the cache inventory, the 403.

The incident this file pins down (run #146, 2026-08-03): a dense Krea run died
at step 2750/3000 on ``403 … Private repository storage limit reached`` — hours
of paid GPU lost because the account's private allowance was full of
``lds-base-*`` custom-base caches nothing in the app listed or cleaned.

Contract under test:
  a) MEASUREMENT — private storage is summed from the Hub's own ``usedStorage``
     expansion over the namespace's PRIVATE repos (models + datasets). There is
     no quota endpoint, so an account that cannot be listed reads as UNKNOWN,
     never as full;
  b) FORECAST — one dense checkpoint is sized from what past runs actually
     delivered (their persisted Hub integrity proof), × saves kept, + margin;
  c) LAUNCH — a run that plainly will not fit is refused BEFORE a pod exists,
     with the confirmable HF_STORAGE_FULL marker; ``allow_hf_storage`` lets it
     through; an unmeasurable account never blocks;
  d) INVENTORY — every lds-base-* repo is joined to its local source file and
     the run that last used it, and deletion is only "serene" when that local
     file still exists;
  e) DELETION — the name is validated, so nothing but a cache can be addressed;
  f) THE WALL — a storage 403 seen mid-run classifies as a RECOVERABLE, named
     failure that keeps the paid pod.

No HfApi is ever real here.
"""
import json
from app.utils.timestamps import naive_utcnow
import struct
import types

import pytest

GB = 1000 ** 3


# --- a fake Hub that knows about usedStorage ----------------------------------

class _Repo:
    """Mirrors what huggingface_hub hands back: `usedStorage` is NOT a modelled
    field, it lands on the object through ModelInfo's __dict__.update(**kwargs)."""

    def __init__(self, repo_id, private=True, used=None):
        self.id = repo_id
        self.private = private
        self.last_modified = None
        if used is not None:
            self.usedStorage = used


class _StorageApi:
    def __init__(self, models=(), datasets=(), who=None):
        self.models = list(models)
        self.datasets = list(datasets)
        self.who = who if who is not None else {'name': 'tester'}
        self.expands = []
        self.deleted = []

    def whoami(self):
        return self.who

    def list_models(self, author=None, expand=None):
        self.expands.append(('model', author, tuple(expand or ())))
        return list(self.models)

    def list_datasets(self, author=None, expand=None):
        self.expands.append(('dataset', author, tuple(expand or ())))
        return list(self.datasets)

    def delete_repo(self, **kw):
        self.deleted.append(kw)


class _BlindApi:
    """A Hub that answers whoami but refuses to list anything — the honest
    "cannot measure" case."""

    def whoami(self):
        return {'name': 'tester'}

    def list_models(self, **kw):
        raise RuntimeError('listing unavailable')

    def list_datasets(self, **kw):
        raise RuntimeError('listing unavailable')


# --- a) measurement -------------------------------------------------------------

def test_usage_sums_private_repos_and_asks_for_used_storage(app):
    from app.services import hf_storage
    api = _StorageApi(
        models=[_Repo('tester/lds-base-h1111', used=24 * GB),
                _Repo('tester/public-lora', private=False, used=9 * GB),
                _Repo('tester/lds-base-h2222', used=20 * GB)],
        datasets=[_Repo('tester/private-set', used=6 * GB)])
    with app.app_context():
        usage = hf_storage.private_storage_usage('tester', 'tok', _api=api)
    assert usage['ok'] is True
    # public repo excluded: the private allowance is a different budget
    assert usage['used_bytes'] == 50 * GB
    assert usage['private_repo_count'] == 3
    assert usage['partial'] is False
    # The measurement only works because of the expansion — if this assert ever
    # breaks, every size above silently becomes None and the sum becomes 0.
    assert all('usedStorage' in expand for _kind, _a, expand in api.expands)


def test_usage_reports_unknown_instead_of_zero_when_it_cannot_list(app):
    from app.services import hf_storage
    with app.app_context():
        usage = hf_storage.private_storage_usage('tester', 'tok', _api=_BlindApi())
    assert usage['ok'] is False
    assert usage['reason'] == 'listing_unavailable'
    assert usage['used_bytes'] is None


def test_unsized_private_repos_make_the_total_a_floor(app):
    from app.services import hf_storage
    api = _StorageApi(models=[_Repo('tester/a', used=10 * GB),
                              _Repo('tester/b')])       # no usedStorage at all
    with app.app_context():
        usage = hf_storage.private_storage_usage('tester', 'tok', _api=api)
    assert usage['ok'] is True
    assert usage['used_bytes'] == 10 * GB
    assert usage['unsized_repo_count'] == 1


# --- b) forecast ----------------------------------------------------------------

def test_checkpoint_size_is_measured_from_what_past_runs_delivered(app):
    from app.services import hf_storage
    rows = [
        types.SimpleNamespace(train_params=json.dumps({'training_mode': 'lora'})),
        types.SimpleNamespace(train_params=json.dumps({
            'training_mode': 'full_transformer',
            'hf_artifact_proof': {'size_bytes': 25 * GB}})),
        types.SimpleNamespace(train_params='not json'),
    ]
    with app.app_context():
        size, source = hf_storage.dense_checkpoint_bytes(_runs=rows)
        assert (size, source) == (25 * GB, 'measured')
        # No dense history: the documented ~26 GB, flagged as an estimate.
        size, source = hf_storage.dense_checkpoint_bytes(_runs=[])
        assert source == 'estimated'
        assert size == hf_storage.DENSE_CHECKPOINT_FALLBACK_BYTES


def test_forecast_blocks_and_passes_around_the_assumed_ceiling(app):
    from app.services import hf_storage
    full = {'ok': True, 'namespace': 'tester', 'used_bytes': 90 * GB,
            'repos': [{'name': 'lds-base-h1', 'private': True,
                       'used_bytes': 40 * GB}]}
    empty = {'ok': True, 'namespace': 'tester', 'used_bytes': 0, 'repos': []}
    with app.app_context():
        blocked = hf_storage.dense_storage_forecast('tester', 'tok', _usage=full)
        roomy = hf_storage.dense_storage_forecast('tester', 'tok', _usage=empty)
    assert blocked['fits'] is False and blocked['shortfall_bytes'] > 0
    assert roomy['fits'] is True and roomy['free_bytes'] > 0
    # The ceiling is never presented as fact.
    assert blocked['limit_is_estimate'] is True
    assert blocked['limit_source'] == 'plan_free_documented'


def test_forecast_is_unknown_not_full_when_usage_is_unknown(app):
    from app.services import hf_storage
    with app.app_context():
        f = hf_storage.dense_storage_forecast(
            'tester', 'tok', _usage={'ok': False, 'reason': 'listing_unavailable'})
    assert f['fits'] is None
    assert f['shortfall_bytes'] is None


def test_configured_ceiling_wins_over_the_documented_plan(app, monkeypatch):
    from app import config as cfg
    from app.services import hf_storage
    with app.app_context():
        cfg.save_config({'cloud': {'full_transformer': {'private_storage_limit_gb': 50}}})
        limit, source = hf_storage.private_limit_bytes({'isPro': True})
        assert (limit, source) == (50 * GB, 'configured')


def test_refusal_message_names_the_gap_the_caches_and_the_escape(app):
    from app.services import hf_storage
    usage = {'ok': True, 'namespace': 'tester', 'used_bytes': 90 * GB,
             'repos': [{'name': 'lds-base-h1111', 'private': True,
                        'used_bytes': 24 * GB},
                       {'name': 'lds-base-h2222', 'private': True,
                        'used_bytes': 20 * GB}]}
    with app.app_context():
        msg = hf_storage.storage_refusal_message(
            hf_storage.dense_storage_forecast('tester', 'tok', _usage=usage))
    assert msg.startswith(hf_storage.STORAGE_REFUSAL_MARKER)
    assert 'short' in msg
    assert 'lds-base-h1111' in msg                     # what takes the room
    assert 'Settings' in msg                           # where to click
    assert 'ESTIMATE' in msg                           # never sold as fact


# --- c) launch guard ------------------------------------------------------------

def test_assert_blocks_then_lets_an_explicit_override_through(app):
    from app.services import hf_storage
    api = _StorageApi(models=[_Repo('tester/lds-base-h1', used=95 * GB)])
    with app.app_context():
        with pytest.raises(ValueError) as excinfo:
            hf_storage.assert_dense_storage_headroom('tester', 'tok', _api=api)
        assert hf_storage.STORAGE_REFUSAL_MARKER in str(excinfo.value)
        # The user's own account, an estimated ceiling: they keep the last word.
        out = hf_storage.assert_dense_storage_headroom(
            'tester', 'tok', allow_override=True, _api=api)
        assert out['fits'] is False


def test_assert_never_blocks_an_account_it_could_not_measure(app):
    from app.services import hf_storage
    with app.app_context():
        out = hf_storage.assert_dense_storage_headroom(
            'tester', 'tok', _api=_BlindApi())
    assert out['fits'] is None


# --- d/e) lds-base-* inventory and deletion -------------------------------------

def _write_safetensors(path):
    meta = {'k': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}
    header = json.dumps(meta).encode('utf-8')
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<Q', len(header)))
        fh.write(header)
        fh.write(b'\x00' * 4)
    return str(path)


def _dense_dataset_with_pushed_base(app, tmp_path, base_name='k.safetensors'):
    """A dataset whose custom base was pushed, plus the cloud run that used it."""
    from app.config import LOCAL_USER
    from app.models import CloudTrainingRun, db
    from app.services import face_dataset_service as svc
    from app.services import hf_base_push
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'CB', 'zc_cb', train_type='krea')
        base = _write_safetensors(tmp_path / base_name)
        repo_name = hf_base_push.base_repo_name(ds, 'krea', base)
        run = CloudTrainingRun(
            dataset_id=ds.id, status='done', run_name='zc_cb_run',
            train_params=json.dumps({'train_type': 'krea', 'base_model': base,
                                     'variant': 'base',
                                     'base_repo_id': f'tester/{repo_name}'}))
        db.session.add(run)
        db.session.commit()
        return repo_name, base, run.id


def test_inventory_joins_size_local_source_and_last_run(app, tmp_path):
    from app.services import hf_storage
    repo_name, base, run_id = _dense_dataset_with_pushed_base(app, tmp_path)
    api = _StorageApi(models=[_Repo(f'tester/{repo_name}', used=24 * GB),
                              _Repo('tester/unrelated', used=1 * GB)])
    with app.app_context():
        inv = hf_storage.base_cache_inventory('tester', 'tok', 'local', _api=api)
    assert [c['name'] for c in inv['caches']] == [repo_name]
    cache = inv['caches'][0]
    assert cache['used_bytes'] == 24 * GB
    assert cache['family'] == 'krea'
    assert cache['local_available'] is True
    assert cache['local_path'] == base
    assert cache['last_run']['id'] == run_id
    assert inv['cache_bytes'] == 24 * GB


def test_inventory_warns_when_the_local_source_is_gone(app, tmp_path):
    """A cache whose local file vanished is the LAST copy of those weights —
    deleting it is not an undo away, and the payload must say so."""
    import os
    from app.services import hf_storage
    repo_name, base, _run = _dense_dataset_with_pushed_base(app, tmp_path)
    os.remove(base)
    api = _StorageApi(models=[_Repo(f'tester/{repo_name}', used=24 * GB)])
    with app.app_context():
        inv = hf_storage.base_cache_inventory('tester', 'tok', 'local', _api=api)
    cache = inv['caches'][0]
    assert cache['local_available'] is False
    assert cache['local_reason'] == 'weights_missing'


def test_delete_refuses_any_repo_that_is_not_a_cache(app):
    from app.services import hf_storage
    api = _StorageApi()
    with app.app_context():
        for name in ('Krea_full_person_run12', '../lds-base-h1', 'lds-base',
                     'lds-base-h1/../../other', ''):
            with pytest.raises(ValueError):
                hf_storage.delete_base_cache('tester', name, 'tok', _api=api)
    assert api.deleted == []


def test_delete_all_sweeps_caches_and_reports_freed_space(app, tmp_path):
    from app.services import hf_storage
    repo_name, _base, _run = _dense_dataset_with_pushed_base(app, tmp_path)
    api = _StorageApi(models=[_Repo(f'tester/{repo_name}', used=24 * GB),
                              _Repo('tester/lds-base-hdead00', used=20 * GB),
                              _Repo('tester/keep-me', used=5 * GB)])
    with app.app_context():
        res = hf_storage.delete_all_base_caches('tester', 'tok', 'local', _api=api)
    assert res['ok'] is True
    assert sorted(res['deleted']) == sorted([repo_name, 'lds-base-hdead00'])
    assert res['freed_bytes'] == 44 * GB
    assert [kw['repo_id'] for kw in api.deleted] == [
        f'tester/{repo_name}', 'tester/lds-base-hdead00']


# --- the routes (wiring only — no Hub is reachable from a test) -----------------

@pytest.fixture()
def cloud_client(client, monkeypatch):
    from app import capabilities
    monkeypatch.setattr(capabilities, 'probe',
                        lambda *a, **k: {'cloud_training': True})
    return client


def test_storage_route_says_it_does_not_know_instead_of_failing(cloud_client):
    """No token: the card must be able to render an honest "unknown" — a 500
    here would make the whole section look broken."""
    res = cloud_client.get('/api/cloud/hf-storage')
    assert res.status_code == 200
    body = res.get_json()
    assert body['ok'] is False
    assert body['reason'] == 'no_token'


def test_delete_route_refuses_a_repo_that_is_not_a_cache(cloud_client, monkeypatch):
    from app.routes import training as routes
    api = _StorageApi()
    monkeypatch.setattr(routes, '_hf_storage_namespace',
                        lambda: ('tester', 'tok', {}))
    monkeypatch.setattr('app.services.hf_storage._make_api', lambda token: api)
    res = cloud_client.delete('/api/cloud/hf-storage/base/Krea_full_person_run12')
    assert res.status_code == 400
    assert api.deleted == []


# --- f) the wall, mid-run -------------------------------------------------------

@pytest.mark.parametrize('text', [
    '403 Forbidden: Private repository storage limit reached',
    'HfHubHTTPError: You have exceeded your storage quota',
    'Error: private storage limit for this account',
])
def test_storage_403_is_named_and_kept_recoverable(text):
    from app.services import cloud_training as ct
    assert ct._hf_storage_full(text) is True
    detail, error = ct._dense_remote_failure('error', text, '')
    assert 'HF private storage full' in detail
    assert 'kept pod' in detail
    assert 'Settings' in error


def test_storage_verdict_also_reads_the_pod_log_not_only_the_job_info():
    from app.services import cloud_training as ct
    detail, _error = ct._dense_remote_failure(
        'error', 'Job failed',
        'Traceback...\n403 Client Error: Private repository storage limit reached\n')
    assert 'HF private storage full' in detail


def test_an_unrelated_dense_failure_keeps_its_generic_wording():
    from app.services import cloud_training as ct
    assert ct._hf_storage_full('CUDA out of memory') is False
    detail, error = ct._dense_remote_failure('error', 'CUDA out of memory', '')
    assert detail == 'Remote dense job unexpectedly error; pod kept'
    assert error == 'remote job error; pod kept for recovery'


def test_a_kept_dense_pod_stays_recoverable_and_is_never_destroyed(app, monkeypatch):
    """The 403 path must land on error_pod_kept WITHOUT terminating the
    instance: the ~26 GB checkpoint exists nowhere else."""
    from app.models import CloudTrainingRun, db
    from app.services import cloud_training as ct
    destroyed = []
    monkeypatch.setattr(ct.vast_client, 'destroy_instance',
                        lambda i: destroyed.append(i) or True)
    with app.app_context():
        run = CloudTrainingRun(
            dataset_id=1, status='training', run_name='r', vast_instance_id='42',
            train_params=json.dumps({'training_mode': 'full_transformer',
                                     'train_type': 'krea'}))
        db.session.add(run)
        db.session.commit()
        detail, error = ct._dense_remote_failure(
            'error', '403 Private repository storage limit reached', '')
        ct._keep_full_transformer_pod(run, detail, error)
        db.session.refresh(run)
        assert run.status == 'error_pod_kept'
        assert run.vast_instance_id == '42'
        assert destroyed == []
        assert 'HF private storage full' in run.phase_detail
        run.finished_at = naive_utcnow()
        assert ct._full_transformer_recovery_open(run) is True
