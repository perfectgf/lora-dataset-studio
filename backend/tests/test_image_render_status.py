"""The ✨ modal's heartbeat — two tiny polls, one per id space."""


def _mk_dataset(app):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    return svc.create_dataset(LOCAL_USER, 'Poll', 'polltrig')


def test_canvas_status_reports_the_three_states(client, app):
    from app.extensions import db
    from app.models import LoraTestImage
    with app.app_context():
        ds = _mk_dataset(app)
        done = LoraTestImage(dataset_id=ds.id, checkpoint='c.safetensors', strength=1,
                             status='done', filename='x.png')
        pending = LoraTestImage(dataset_id=ds.id, checkpoint='c.safetensors', strength=1,
                                status='pending')
        failed = LoraTestImage(dataset_id=ds.id, checkpoint='c.safetensors', strength=1,
                               status='failed', error='boom')
        db.session.add_all([done, pending, failed])
        db.session.commit()
        ids = (done.id, pending.id, failed.id, ds.id)
    d = client.get(f'/api/canvas/image/{ids[0]}/status').get_json()
    assert d['status'] == 'done' and d['url'].endswith('/x.png')
    p = client.get(f'/api/canvas/image/{ids[1]}/status').get_json()
    assert p['status'] == 'pending' and p['url'] is None and p['error'] is None
    f = client.get(f'/api/canvas/image/{ids[2]}/status').get_json()
    assert f['status'] == 'failed' and f['error'] == 'boom'
    assert client.get('/api/canvas/image/999999/status').status_code == 404


def test_dataset_status_is_the_twin_on_its_own_table(client, app):
    from app.extensions import db
    from app.models import FaceDatasetImage
    with app.app_context():
        ds = _mk_dataset(app)
        row = FaceDatasetImage(dataset_id=ds.id, source='generated', status='failed',
                               fail_reason='no engine')
        db.session.add(row)
        db.session.commit()
        rid = row.id
    d = client.get(f'/api/dataset/image/{rid}/status').get_json()
    assert d['status'] == 'failed' and d['error'] == 'no engine' and d['url'] is None
    assert client.get('/api/dataset/image/999999/status').status_code == 404
