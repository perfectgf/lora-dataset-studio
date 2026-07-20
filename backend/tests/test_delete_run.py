"""Remove a GONE run from the lineage graph (DELETE /dataset/train/runs/<id>).
Metadata only: the record + its checkpoint notes go, the disk is untouched.

Covers the four contracts the graph relies on: a gone run is removed with its
notes; a run whose checkpoints are still on disk is refused (409, never a silent
erase); deleting a parent detaches — never orphans a 500 on — a living child;
an unknown id is 404."""
import os


def _rec(dataset_id=1, family='zimage', source='local', steps=1000, version=1,
         parent=None, resumed_from=None, cloud_run_id=None):
    from app.models import TrainingRunRecord
    from app.extensions import db
    r = TrainingRunRecord(
        dataset_id=dataset_id, family=family, source=source, base_model='',
        variant='turbo', steps=steps, version=version, fingerprint='fp',
        manifest='[]', parent_record_id=parent, resumed_from=resumed_from,
        cloud_run_id=cloud_run_id)
    db.session.add(r)
    db.session.commit()
    return r


def test_gone_run_deleted_with_its_notes(client, app):
    """A local run with no checkpoints on disk is removed — record, run note and
    every checkpoint note gone (the ai-toolkit scan finds nothing in tests)."""
    from app.models import TrainingRunRecord, CheckpointNote
    from app.extensions import db
    with app.app_context():
        rec = _rec(steps=2000)
        rec.note = 'overcooked past 1500'
        db.session.add(CheckpointNote(record_id=rec.id, step=1000, note='a'))
        db.session.add(CheckpointNote(record_id=rec.id, step=1500, note='b'))
        db.session.commit()
        rid = rec.id

    assert client.delete(f'/api/dataset/train/runs/{rid}').status_code == 200

    with app.app_context():
        assert db.session.get(TrainingRunRecord, rid) is None
        assert CheckpointNote.query.filter_by(record_id=rid).count() == 0


def test_run_with_saves_on_disk_is_refused_409(client, app, tmp_path):
    """A cloud run whose staging still holds a .safetensors is a live run — the
    delete is refused with 409 and the record stays put."""
    from app.models import TrainingRunRecord, CloudTrainingRun
    from app.extensions import db
    staging = tmp_path / 'staging'
    staging.mkdir()
    (staging / 'lora_000001000.safetensors').write_bytes(b'W')
    with app.app_context():
        crun = CloudTrainingRun(dataset_id=1, status='done',
                                staging_dir=str(staging))
        db.session.add(crun)
        db.session.commit()
        rec = _rec(source='cloud', cloud_run_id=crun.id)
        rid = rec.id

    r = client.delete(f'/api/dataset/train/runs/{rid}')
    assert r.status_code == 409
    assert 'checkpoints on disk' in r.get_json()['error']

    with app.app_context():
        assert db.session.get(TrainingRunRecord, rid) is not None


def test_deleting_parent_detaches_living_child_no_500(client, app):
    """Removing a gone parent that a still-present child resumed from must not
    500 (the FK 'delete 500' trap) — the child survives, re-rooted."""
    from app.models import TrainingRunRecord
    from app.extensions import db
    with app.app_context():
        parent = _rec(steps=1000)
        child = _rec(steps=1500, parent=parent.id, resumed_from=1000)
        pid, cid = parent.id, child.id

    assert client.delete(f'/api/dataset/train/runs/{pid}').status_code == 200

    with app.app_context():
        assert db.session.get(TrainingRunRecord, pid) is None
        surviving = db.session.get(TrainingRunRecord, cid)
        assert surviving is not None
        assert surviving.parent_record_id is None      # detached, not orphaned


def test_unknown_record_is_404(client, app):
    assert client.delete('/api/dataset/train/runs/999999').status_code == 404
