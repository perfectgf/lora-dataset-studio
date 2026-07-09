"""train_type is chosen at CREATION (drives caption style + menu grouping from the
start), normalized, exposed in the list, and stays settable later so the grouped menu
re-sorts and the SDXL->booru / else->prose caption default follows."""
from app.services import face_dataset_service as svc


def test_create_dataset_persists_train_type(app):
    with app.app_context():
        ds = svc.create_dataset('local', 'Emma', 'zchar_emma', train_type='sdxl')
        assert ds.train_type == 'sdxl'


def test_create_dataset_defaults_and_normalizes(app):
    with app.app_context():
        assert svc.create_dataset('local', 'A', 'a').train_type == 'zimage'                    # default
        assert svc.create_dataset('local', 'B', 'b', train_type='SDXL').train_type == 'sdxl'   # case-fold
        assert svc.create_dataset('local', 'C', 'c', train_type='bogus').train_type == 'zimage'  # unknown


def test_set_train_type_updates_and_normalizes(app):
    with app.app_context():
        ds = svc.create_dataset('local', 'Emma', 'zchar_emma')
        assert svc.set_train_type('local', ds.id, 'krea') is True
        assert svc.get_dataset('local', ds.id).train_type == 'krea'
        assert svc.set_train_type('local', ds.id, 'nope') is True
        assert svc.get_dataset('local', ds.id).train_type == 'zimage'          # unknown -> zimage
        assert svc.set_train_type('local', 999999, 'sdxl') is False            # absent dataset


def test_create_route_forwards_train_type_and_list_exposes_it(client):
    did = client.post('/api/dataset/create',
                      json={'name': 'Zoe', 'trigger_word': 'zchar_zoe', 'train_type': 'krea'}).get_json()['id']
    rows = client.get('/api/dataset/list').get_json()['datasets']
    assert next(r for r in rows if r['id'] == did)['train_type'] == 'krea'


def test_train_type_route_updates(client):
    did = client.post('/api/dataset/create',
                      json={'name': 'Ivy', 'trigger_word': 'zchar_ivy'}).get_json()['id']
    assert client.post(f'/api/dataset/{did}/train-type', json={'train_type': 'sdxl'}).status_code == 200
    rows = client.get('/api/dataset/list').get_json()['datasets']
    assert next(r for r in rows if r['id'] == did)['train_type'] == 'sdxl'


def test_train_type_route_unknown_dataset_404(client):
    assert client.post('/api/dataset/999999/train-type', json={'train_type': 'sdxl'}).status_code == 404
