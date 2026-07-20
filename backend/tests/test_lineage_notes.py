"""Lab notes: freeform text per run (a `note` column on TrainingRunRecord) and
per checkpoint (a CheckpointNote row keyed by (record_id, step)). PUT endpoints
round-trip; the node payload gains has_note; missing record → False / 404."""


def test_run_and_checkpoint_notes_roundtrip(client, app):
    from app.services import cloud_training as ct
    from app.models import TrainingRunRecord
    from app.extensions import db
    with app.app_context():
        rec = TrainingRunRecord(dataset_id=1, family='zimage', source='local',
                                version=1, fingerprint='fp', steps=2000)
        db.session.add(rec); db.session.commit(); rid = rec.id
    assert client.put(f'/api/dataset/train/runs/{rid}/note',
                      json={'note': 'overcooks past 1500'}).status_code == 200
    assert client.put(f'/api/dataset/train/runs/{rid}/checkpoints/1500/note',
                      json={'note': 'best face'}).status_code == 200
    with app.app_context():
        assert ct.set_run_note(9_999_999, 'x') is False           # missing record
        node = ct._lineage_node(TrainingRunRecord.query.get(rid), None, rid, None)
        assert node['has_note'] is True
        assert ct.checkpoint_notes_for(rid) == {1500: 'best face'}
