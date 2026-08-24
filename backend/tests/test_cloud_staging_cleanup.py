"""Per-run staging cleanup on the Runs hub: sizes, the targeted 🧹, the sparing
rule both purges must share — and, above all, WHAT the cleanup is allowed to
throw away.

The hub used to offer ONE all-or-nothing "Clean finished runs" and no way to see
what a run weighed, so a 45-run / 60 GB history could only be cleaned blind.

Then the cleanup itself turned out to be dangerous: it trashed the whole staging
directory while advertising "checkpoint duplicates already imported". A
checkpoint that had never been deployed to ComfyUI had no duplicate — staging
was its only copy — and emptying the trash destroyed it. So the load-bearing
tests here are the ones that pin what SURVIVES a purge, not what it frees.
"""
import os
from app.extensions import db
from app.utils.timestamps import naive_utcnow
from datetime import timedelta

import pytest


@pytest.fixture()
def runs(app, tmp_path):
    """Five cloud runs with real staging dirs of known sizes: one done, one
    stopped, one still training, one whose pod was kept for recovery a minute
    ago, and one whose kept pod expired long ago."""
    from app.extensions import db
    from app.models import CloudTrainingRun
    made = {}
    now = naive_utcnow()
    finished = {'kept_open': now - timedelta(minutes=5),
                'kept_expired': now - timedelta(days=7)}
    with app.app_context():
        for key, status, kb in (('done', 'done', 4), ('stopped', 'stopped', 2),
                                ('training', 'training', 8),
                                ('kept_open', 'error_pod_kept', 6),
                                ('kept_expired', 'error_pod_kept', 6)):
            sd = tmp_path / f'staging_{key}'
            (sd / 'samples').mkdir(parents=True)
            (sd / 'lora.safetensors').write_bytes(b'x' * (kb * 1024))
            (sd / 'samples' / 's.png').write_bytes(b'y' * 1024)
            run = CloudTrainingRun(dataset_id=1, status=status,
                                   staging_dir=str(sd),
                                   finished_at=finished.get(key),
                                   checkpoint_local_path=str(sd / 'lora.safetensors'))
            db.session.add(run)
            db.session.flush()
            made[key] = run.id
        db.session.commit()
    return made


def _clear_size_cache():
    from app.services import cloud_training as ct
    ct._staging_size_cache.clear()


def test_spare_reason_is_the_single_rule_both_purges_use(app, runs):
    from app.models import CloudTrainingRun
    from app.services import cloud_training as ct
    with app.app_context():
        get = lambda s: db.session.get(CloudTrainingRun, runs[s])   # noqa: E731
        assert ct.staging_spare_reason(get('done')) is None
        assert ct.staging_spare_reason(get('stopped')) is None
        assert 'still active' in ct.staging_spare_reason(get('training'))
        assert 'manual recovery' in ct.staging_spare_reason(get('kept_open'))
        # A kept pod is billed for at most cloud.max_runtime_minutes past the
        # run's end. Once that window has closed the pod is gone, and sparing
        # its staging forever only froze tens of GB on a full disk.
        assert ct.staging_spare_reason(get('kept_expired')) is None


def test_staging_sizes_reports_only_what_is_on_disk(app, runs):
    from app.services import cloud_training as ct
    _clear_size_cache()
    with app.app_context():
        sizes = ct.staging_sizes()
        # every run has a staging dir here, sizes follow the bytes written
        assert sizes[runs['done']] == (4 * 1024) + 1024
        assert sizes[runs['training']] == (8 * 1024) + 1024
        # a run whose staging is gone simply drops out (no misleading 0)
        _clear_size_cache()
        import shutil
        shutil.rmtree(db.session.get(ct.CloudTrainingRun, runs['stopped']).staging_dir)
        sizes = ct.staging_sizes()
        assert runs['stopped'] not in sizes
        # narrowing to the shown cards only walks those
        assert set(ct.staging_sizes([runs['done']])) == {runs['done']}


def test_purge_keeps_the_checkpoint_and_only_frees_the_working_files(app, runs):
    """The regression test for the incident: a purge must move the samples and
    the dataset copy, and leave every .safetensors readable — in the store."""
    from app.models import CloudTrainingRun
    from app.services import cloud_training as ct
    _clear_size_cache()
    with app.app_context():
        target = db.session.get(CloudTrainingRun, runs['done'])
        target_dir = target.staging_dir
        other_dir = db.session.get(CloudTrainingRun, runs['stopped']).staging_dir
        res = ct.purge_run_staging(runs['done'])
        assert res['purged'] is True
        # ONLY the samples — the 4 KB checkpoint was rescued, not freed
        assert res['freed_bytes'] == 1024
        assert not os.path.isdir(os.path.join(target_dir, 'samples'))
        assert os.path.isdir(other_dir)           # the neighbour is untouched

        after = db.session.get(CloudTrainingRun, runs['done'])
        saves = ct.run_checkpoint_files(after)
        assert set(saves) == {'lora.safetensors'}
        assert os.path.isfile(saves['lora.safetensors'])
        # it lives in the store now, not under the purged staging dir
        assert os.path.dirname(saves['lora.safetensors']) \
            == ct.checkpoint_store_dir(after)
        # and the "checkpoint on disk" claim stays TRUE instead of being wiped
        assert after.checkpoint_local_path == saves['lora.safetensors']
        assert db.session.get(CloudTrainingRun, runs['done']) is not None


def test_purge_one_run_refuses_the_runs_the_global_purge_spares(app, runs):
    from app.models import CloudTrainingRun
    from app.services import cloud_training as ct
    with app.app_context():
        with pytest.raises(ValueError):
            ct.purge_run_staging(runs['training'])
        assert os.path.isdir(db.session.get(CloudTrainingRun, runs['training']).staging_dir)
        with pytest.raises(ValueError):
            ct.purge_run_staging(runs['kept_open'])
        # …but an EXPIRED kept pod is purgeable like any other finished run
        assert ct.purge_run_staging(runs['kept_expired'])['purged'] is True
        with pytest.raises(ValueError):
            ct.purge_run_staging(999999)


def test_purge_one_run_twice_is_an_honest_no_op(app, runs):
    from app.services import cloud_training as ct
    with app.app_context():
        ct.purge_run_staging(runs['done'])
        again = ct.purge_run_staging(runs['done'])
        assert again == {'purged': False, 'freed_bytes': 0, 'already_clean': True}


def test_global_purge_reports_already_clean_instead_of_a_bare_zero(app, runs):
    from app.services import cloud_training as ct
    with app.app_context():
        first = ct.purge_finished_runs()
        # done + stopped + the expired kept pod; the active run and the pod
        # still inside its recovery window are spared
        assert first['purged_runs'] == 3
        assert first['freed_bytes'] == 3 * 1024          # samples only
        assert first['already_clean'] is False
        # nothing left to purge: the caller can say so instead of "0 run(s), 0.0 GB"
        second = ct.purge_finished_runs()
        assert second['purged_runs'] == 0
        assert second['already_clean'] is True


def test_orphan_run_folders_are_named_instead_of_ignored(app, runs, monkeypatch,
                                                         tmp_path):
    """A run_<id> folder no row points at used to be answered 'already clean'
    while it held 25 GB. It is now listed, sized, and purgeable on request —
    with any loose checkpoint rescued into the store first."""
    from app.services import cloud_training as ct
    root = tmp_path / 'cloud_runs_root'
    (root / 'run_9001' / 'dataset').mkdir(parents=True)
    (root / 'run_9001' / 'dataset' / 'a.png').write_bytes(b'z' * 2048)
    (root / 'run_9001' / 'old.safetensors').write_bytes(b'w' * 512)
    (root / 'not_a_run').mkdir()
    monkeypatch.setattr(ct, '_staging_root', lambda: root)
    with app.app_context():
        found = ct.orphan_staging_dirs()
        assert [o['name'] for o in found] == ['run_9001']
        assert found[0]['size_bytes'] == 2048 + 512
        assert found[0]['checkpoints'] == 1

        res = ct.purge_orphan_staging_dirs(['run_9001'])
        assert res['purged_dirs'] == 1
        assert res['rescued_checkpoints'] == 1
        assert not (root / 'run_9001').exists()
        # the weight nobody could vouch for was kept, not destroyed
        from app import config as cfg
        assert (cfg.checkpoints_root() / 'run_9001' / 'old.safetensors').is_file()
        # a name that is not a reported orphan is refused, never resolved
        assert ct.purge_orphan_staging_dirs(['../etc'])['skipped'] == ['../etc']


def test_retrofit_moves_legacy_staging_checkpoints_into_the_store(app, runs):
    """An install that trained before the store existed still keeps its only
    copies in staging; the boot pass moves them without touching anything else."""
    from app.models import CloudTrainingRun
    from app.services import cloud_training as ct
    with app.app_context():
        res = ct.migrate_checkpoints_into_store(force=True)
        assert res['ran'] is True
        assert res['moved'] == 5 and res['runs'] == 5
        run = db.session.get(CloudTrainingRun, runs['done'])
        assert not any(f.endswith('.safetensors')
                       for f in os.listdir(run.staging_dir))
        assert os.path.isfile(os.path.join(ct.checkpoint_store_dir(run),
                                           'lora.safetensors'))
        # the samples are none of its business
        assert os.path.isdir(os.path.join(run.staging_dir, 'samples'))
        # idempotent: a second pass finds nothing left to move
        assert ct.migrate_checkpoints_into_store(force=True)['moved'] == 0


def test_routes_expose_sizes_and_the_targeted_purge(client, app, runs):
    _clear_size_cache()
    r = client.get('/api/dataset/train/cloud/staging-sizes')
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is True
    assert body['sizes'][str(runs['done'])] == (4 * 1024) + 1024
    assert body['total_bytes'] == sum(body['sizes'].values())
    # narrowed
    r = client.get(f"/api/dataset/train/cloud/staging-sizes?run_ids={runs['done']}")
    assert set(r.get_json()['sizes']) == {str(runs['done'])}
    r = client.get('/api/dataset/train/cloud/staging-sizes?run_ids=abc')
    assert r.status_code == 400

    r = client.post('/api/dataset/train/cloud/purge-run', json={'run_id': runs['done']})
    assert r.status_code == 200 and r.get_json()['purged'] is True
    # a spared run is REFUSED with a reason, never a silent {'ok': True}
    r = client.post('/api/dataset/train/cloud/purge-run',
                    json={'run_id': runs['kept_open']})
    assert r.status_code == 400
    assert 'recovery' in r.get_json()['error']
    r = client.post('/api/dataset/train/cloud/purge-run', json={})
    assert r.status_code == 400


def test_orphan_routes_report_and_purge(client, app, runs, monkeypatch, tmp_path):
    from app.services import cloud_training as ct
    root = tmp_path / 'cloud_runs_root2'
    (root / 'run_9002').mkdir(parents=True)
    (root / 'run_9002' / 'training.log').write_bytes(b'l' * 100)
    monkeypatch.setattr(ct, '_staging_root', lambda: root)
    body = client.get('/api/dataset/train/cloud/orphans').get_json()
    assert [o['name'] for o in body['orphans']] == ['run_9002']
    assert body['total_bytes'] == 100
    res = client.post('/api/dataset/train/cloud/purge-orphans', json={})
    assert res.status_code == 200 and res.get_json()['purged_dirs'] == 1
    assert client.post('/api/dataset/train/cloud/purge-orphans',
                       json={'names': 'run_9002'}).status_code == 400


def test_global_purge_route_carries_the_already_clean_flag(client, app, runs):
    assert client.post('/api/dataset/train/cloud/purge').get_json()['already_clean'] is False
    assert client.post('/api/dataset/train/cloud/purge').get_json()['already_clean'] is True
