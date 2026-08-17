"""⏏ The whole-install list of deployed LoRAs, and undeploying several at once.

Asked for by the maintainer: undeploying was one pill at a time, buried in a
node's popover, with no way to even see how many were deployed.

THE PROPERTY THIS FILE EXISTS FOR is the one that makes the screen safe: the
list feeds a DELETE button, and `loras/<family>/` also holds the LoRAs the user
downloaded themselves. A plain directory scan would offer those for removal.
`list_all_deployed_checkpoints` delegates attribution to
`list_imported_checkpoints` precisely so it cannot, and the first test below is
what keeps that true.
"""
import os

from app.config import LOCAL_USER


def _create(client, name='Lola', trigger='lola'):
    return client.post('/api/dataset/create',
                       json={'name': name, 'trigger_word': trigger}).get_json()['id']


def _valid(monkeypatch, ok=True):
    monkeypatch.setattr('app.capabilities.probe',
                        lambda *a, **k: {'aitoolkit': {'valid': ok}})


def _deploy(app, tmp_path, ds_id, trigger, monkeypatch, step='000001000'):
    """Put one checkpoint through the REAL import path, so the deployed name and
    folder are whatever the app actually produces."""
    from app import config as cfg
    from app.services import lora_training as lt
    with app.app_context():
        cfg.save_config({'comfyui': {'base_dir': str(tmp_path / 'comfy')}})
        run_dir = tmp_path / f'run_{ds_id}_{step}'
        run_dir.mkdir(exist_ok=True)
        ck = run_dir / f'lora_{trigger}_{step}.safetensors'
        ck.write_bytes(b'W')
        monkeypatch.setattr(lt, '_run_dir', lambda *a, **k: str(run_dir))
        return lt.import_checkpoint(LOCAL_USER, ds_id, ck.name)


def test_the_list_never_offers_a_lora_the_user_brought_themselves(
        client, app, tmp_path, monkeypatch):
    """The safety rail. A Civitai download dropped in the same folder is NOT the
    app's to remove, and this screen must not present it as such."""
    from app.services import lora_training as lt
    ds_id = _create(client)
    dest = _deploy(app, tmp_path, ds_id, 'lola', monkeypatch)

    # A file the app never deployed, sitting in the very same folder.
    foreign = os.path.join(os.path.dirname(dest), 'someones_civitai_download.safetensors')
    with open(foreign, 'wb') as f:
        f.write(b'NOT OURS')

    with app.app_context():
        rows = lt.list_all_deployed_checkpoints(LOCAL_USER)
    names = [os.path.basename(r['filename']) for r in rows]
    assert os.path.basename(dest) in names
    assert 'someones_civitai_download.safetensors' not in names
    # …and it is still on disk, untouched.
    assert os.path.isfile(foreign)


def test_the_list_carries_what_the_undeploy_route_needs(client, app, tmp_path, monkeypatch):
    """Each row must be addressable on its own: the undeploy route is
    dataset-scoped, so a row without its dataset is a row nobody can act on."""
    ds_id = _create(client, name='Lola', trigger='lola')
    _deploy(app, tmp_path, ds_id, 'lola', monkeypatch)

    body = client.get('/api/deployed-loras').get_json()
    assert body['deployed'], 'nothing listed after a real import'
    row = body['deployed'][0]
    for key in ('dataset_id', 'dataset_name', 'family', 'filename', 'label'):
        assert key in row, key
    assert row['dataset_id'] == ds_id
    assert row['dataset_name'] == 'Lola'


def test_the_list_spans_several_datasets(client, app, tmp_path, monkeypatch):
    """One dataset at a time was the whole complaint — the point of the screen is
    that it shows the install, not the dataset you happen to be looking at."""
    from app.services import lora_training as lt
    a = _create(client, name='Lola', trigger='lola')
    b = _create(client, name='Elsa', trigger='elsa')
    _deploy(app, tmp_path, a, 'lola', monkeypatch)
    _deploy(app, tmp_path, b, 'elsa', monkeypatch)

    with app.app_context():
        rows = lt.list_all_deployed_checkpoints(LOCAL_USER)
    assert {r['dataset_name'] for r in rows} == {'Lola', 'Elsa'}
    # One row per file on disk, never one per (dataset, file) pairing.
    keys = [(r['family'], r['filename']) for r in rows]
    assert len(keys) == len(set(keys))


def test_bulk_undeploy_removes_the_ticked_files_and_reports_a_ledger(
        client, app, tmp_path, monkeypatch):
    _valid(monkeypatch, True)
    ds_id = _create(client)
    dest = _deploy(app, tmp_path, ds_id, 'lola', monkeypatch)
    listed = client.get('/api/deployed-loras').get_json()['deployed']
    assert listed

    resp = client.post('/api/deployed-loras/undeploy', json={'items': [
        {'dataset_id': r['dataset_id'], 'filename': r['filename'],
         'train_type': r['family']} for r in listed]})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] is True
    assert len(body['removed']) == len(listed)
    assert body['failed'] == [] and body['missing'] == []
    # Trashed, not destroyed — and gone from ComfyUI's folder, which is the point.
    assert not os.path.isfile(dest)
    assert client.get('/api/deployed-loras').get_json()['deployed'] == []


def test_a_file_deleted_by_hand_is_MISSING_not_a_failure(
        client, app, tmp_path, monkeypatch):
    """The user wanted it out of ComfyUI. Somebody already took it out. Reporting
    that as a failure would send them looking for a problem they do not have."""
    _valid(monkeypatch, True)
    ds_id = _create(client)
    dest = _deploy(app, tmp_path, ds_id, 'lola', monkeypatch)
    listed = client.get('/api/deployed-loras').get_json()['deployed']
    os.remove(dest)

    resp = client.post('/api/deployed-loras/undeploy', json={'items': [
        {'dataset_id': r['dataset_id'], 'filename': r['filename'],
         'train_type': r['family']} for r in listed]})
    body = resp.get_json()
    assert body['missing'] and body['removed'] == [] and body['failed'] == []


def test_a_name_outside_the_whitelist_FAILS_and_moves_nothing(
        client, app, tmp_path, monkeypatch):
    """The per-file guard rail of the single undeploy still applies to a bulk
    one: a name the app did not deploy is refused, not removed."""
    _valid(monkeypatch, True)
    ds_id = _create(client)
    dest = _deploy(app, tmp_path, ds_id, 'lola', monkeypatch)
    foreign = os.path.join(os.path.dirname(dest), 'someones_civitai_download.safetensors')
    with open(foreign, 'wb') as f:
        f.write(b'NOT OURS')

    resp = client.post('/api/deployed-loras/undeploy', json={'items': [
        {'dataset_id': ds_id, 'filename': os.path.join('z image', os.path.basename(foreign)),
         'train_type': 'zimage'}]})
    body = resp.get_json()
    assert body['removed'] == [] and body['missing'] == []
    assert len(body['failed']) == 1
    assert os.path.isfile(foreign), 'a refused name must not be touched'


def test_bulk_undeploy_refuses_an_empty_selection(client, monkeypatch):
    _valid(monkeypatch, True)
    for payload in ({}, {'items': []}, {'items': 'all'}):
        resp = client.post('/api/deployed-loras/undeploy', json=payload)
        assert resp.status_code == 400


def test_one_bad_row_does_not_abandon_the_rest_of_the_selection(
        client, app, tmp_path, monkeypatch):
    """Doing many at once is the whole feature: a single unusable row must not
    cost the user the other nineteen."""
    _valid(monkeypatch, True)
    ds_id = _create(client)
    _deploy(app, tmp_path, ds_id, 'lola', monkeypatch)
    listed = client.get('/api/deployed-loras').get_json()['deployed']

    items = [{'dataset_id': 999999, 'filename': 'ghost.safetensors', 'train_type': 'zimage'}]
    items += [{'dataset_id': r['dataset_id'], 'filename': r['filename'],
               'train_type': r['family']} for r in listed]
    body = client.post('/api/deployed-loras/undeploy', json={'items': items}).get_json()
    assert len(body['failed']) == 1
    assert len(body['removed']) == len(listed)
