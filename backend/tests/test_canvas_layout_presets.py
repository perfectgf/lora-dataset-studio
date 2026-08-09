"""💾 LoRA Canvas — named layout presets (canvas_layout_preset).

The board held ONE arrangement. Keeping a second one is only useful if putting
it back is TRUSTWORTHY, so these tests are mostly about the restore:

  * it goes through the live writers, so nothing a drag could not have written
    can enter the board through a preset (another user's dataset, another
    lane's image, unusable geometry);
  * what no longer exists is simply not put back, and the counts SAY so — a
    preset kept for three weeks routinely names a run that has been deleted
    since, and a silent partial restore is ten minutes spent hunting for the
    card that is missing;
  * a closed picture stays closed. Restoring a board that re-opened everything
    you had closed would be putting a different board back.
"""


def _dataset(name='Ada', trigger='ada'):
    from app.services import face_dataset_service as svc
    return svc.create_dataset('local', name, trigger)


def _image(dataset_id, record_id=7, step=2500, filename='a.png'):
    from app.extensions import db
    from app.models import LoraTestImage
    row = LoraTestImage(dataset_id=dataset_id, checkpoint='z image\\Ada-2500.safetensors',
                        strength=1.0, filename=filename, status='done',
                        seed=1, prompt='a portrait', step=step, record_id=record_id,
                        cfg=3.5, steps=20, sampler='euler', scheduler='simple',
                        z_model='zturbo.safetensors')
    db.session.add(row)
    db.session.commit()
    return row.id


def _save(client, name, positions=None, images=None):
    return client.post('/api/train/canvas/layouts', json={
        'name': name, 'positions': positions or {}, 'images': images or {}})


# ---- keeping one -----------------------------------------------------------

def test_a_saved_layout_reads_back_with_its_size(client, app):
    with app.app_context():
        ds_id = _dataset().id
        img_id = _image(ds_id)

    r = _save(client, '  likeness review  ',
              positions={str(ds_id): [{'record_id': 7, 'x': 10, 'y': 20},
                                      {'record_id': 8, 'x': 30, 'y': 40}]},
              images={str(ds_id): [{'image_id': img_id, 'x': 5, 'y': 6,
                                    'w': 300, 'h': 300, 'visible': True}]})
    assert r.status_code == 200
    preset = r.get_json()['preset']
    assert preset['name'] == 'likeness review'      # trimmed
    assert (preset['lanes'], preset['cards'], preset['images']) == (1, 2, 1)

    listed = client.get('/api/train/canvas/layouts').get_json()['presets']
    assert [p['name'] for p in listed] == ['likeness review']


def test_saving_under_the_same_name_replaces_it(client, app):
    with app.app_context():
        ds_id = _dataset().id
    _save(client, 'board', positions={str(ds_id): [{'record_id': 1, 'x': 1, 'y': 1}]})
    _save(client, 'board', positions={str(ds_id): [{'record_id': 1, 'x': 9, 'y': 9},
                                                   {'record_id': 2, 'x': 2, 'y': 2}]})
    listed = client.get('/api/train/canvas/layouts').get_json()['presets']
    assert len(listed) == 1, 'one name, one preset'
    assert listed[0]['cards'] == 2


def test_a_nameless_or_empty_layout_is_refused_with_a_sentence(client, app):
    with app.app_context():
        ds_id = _dataset().id
    r = _save(client, '   ', positions={str(ds_id): [{'record_id': 1, 'x': 1, 'y': 1}]})
    assert r.status_code == 400 and 'name' in r.get_json()['error']
    r = _save(client, 'empty')
    assert r.status_code == 400 and 'nothing' in r.get_json()['error']


# ---- putting one back ------------------------------------------------------

def test_restoring_puts_the_cards_and_the_pictures_back(client, app):
    with app.app_context():
        ds_id = _dataset().id
        img_id = _image(ds_id)
    pid = _save(client, 'keep',
                positions={str(ds_id): [{'record_id': 7, 'x': 111.0, 'y': 222.0}]},
                images={str(ds_id): [{'image_id': img_id, 'x': 12.0, 'y': 34.0,
                                      'w': 400.0, 'h': 300.0, 'visible': True}]}
                ).get_json()['preset']['id']

    # The board is then tidied away completely.
    client.delete(f'/api/dataset/{ds_id}/canvas/positions')
    client.delete(f'/api/dataset/{ds_id}/canvas/images')
    assert client.get('/api/train/canvas/positions').get_json()['positions'] == {}

    r = client.post(f'/api/train/canvas/layouts/{pid}/apply')
    assert r.status_code == 200
    assert r.get_json()['applied'] == {'cards': 1, 'images': 1}

    lane = client.get('/api/train/canvas/positions').get_json()['positions'][str(ds_id)]
    assert lane == [{'record_id': 7, 'x': 111.0, 'y': 222.0}]
    pics = client.get('/api/train/canvas/images').get_json()['nodes'][str(ds_id)]
    assert (pics[0]['x'], pics[0]['y'], pics[0]['w']) == (12.0, 34.0, 400.0)


def test_a_closed_picture_comes_back_closed(client, app):
    """The whole promise of ✕ is "it remembers". A restore that re-opened every
    closed picture would be putting a DIFFERENT board back."""
    with app.app_context():
        ds_id = _dataset().id
        img_id = _image(ds_id)
    pid = _save(client, 'closed',
                images={str(ds_id): [{'image_id': img_id, 'x': 1.0, 'y': 2.0,
                                      'w': 260.0, 'h': 260.0, 'visible': False}]}
                ).get_json()['preset']['id']
    client.delete(f'/api/dataset/{ds_id}/canvas/images')
    client.post(f'/api/train/canvas/layouts/{pid}/apply')
    pics = client.get('/api/train/canvas/images').get_json()['nodes'][str(ds_id)]
    assert pics[0]['visible'] is False


def test_a_picture_deleted_since_is_simply_not_put_back(client, app):
    """The preset is a MEMORY, not a foreign key: a row pointing at something
    that is gone must degrade, not raise."""
    from app.extensions import db
    from app.models import LoraTestImage
    with app.app_context():
        ds_id = _dataset().id
        img_id = _image(ds_id)
    pid = _save(client, 'stale',
                positions={str(ds_id): [{'record_id': 7, 'x': 1, 'y': 1}]},
                images={str(ds_id): [{'image_id': img_id, 'x': 1.0, 'y': 2.0,
                                      'w': 260.0, 'h': 260.0, 'visible': True}]}
                ).get_json()['preset']['id']
    with app.app_context():
        db.session.delete(LoraTestImage.query.get(img_id))
        db.session.commit()

    r = client.post(f'/api/train/canvas/layouts/{pid}/apply')
    assert r.status_code == 200
    applied = r.get_json()['applied']
    assert applied['cards'] == 1
    assert applied['images'] == 0, 'the missing picture is dropped, not restored'


def test_a_preset_cannot_smuggle_an_image_into_another_lane(client, app):
    """The restore goes through save_canvas_image_nodes, which refuses an image
    that does not belong to the dataset — so a hand-written preset cannot put
    one lane's render on another lane."""
    with app.app_context():
        a_id = _dataset('Ada', 'ada').id
        b_id = _dataset('Bea', 'bea').id
        img_id = _image(a_id)
    pid = _save(client, 'crossed',
                images={str(b_id): [{'image_id': img_id, 'x': 1.0, 'y': 1.0,
                                     'w': 260.0, 'h': 260.0, 'visible': True}]}
                ).get_json()['preset']['id']
    r = client.post(f'/api/train/canvas/layouts/{pid}/apply')
    assert r.get_json()['applied']['images'] == 0
    assert client.get('/api/train/canvas/images').get_json()['nodes'] == {}


def test_unusable_geometry_never_reaches_the_stored_preset(client, app):
    """A NaN stored here would come back on every restore, and there is no UI
    to fix it."""
    with app.app_context():
        ds_id = _dataset().id
        img_id = _image(ds_id)
    r = _save(client, 'junk',
              positions={str(ds_id): [{'record_id': 'x', 'x': 1, 'y': 1}]},
              images={str(ds_id): [{'image_id': img_id, 'x': float('nan'), 'y': 1,
                                    'w': 260, 'h': 260}]})
    assert r.status_code == 400, 'nothing usable was sent, so nothing is saved'


# ---- housekeeping ----------------------------------------------------------

def test_a_layout_can_be_forgotten_without_touching_the_board(client, app):
    with app.app_context():
        ds_id = _dataset().id
    pid = _save(client, 'gone',
                positions={str(ds_id): [{'record_id': 3, 'x': 4, 'y': 5}]}
                ).get_json()['preset']['id']
    client.post(f'/api/train/canvas/layouts/{pid}/apply')

    assert client.delete(f'/api/train/canvas/layouts/{pid}').status_code == 200
    assert client.delete(f'/api/train/canvas/layouts/{pid}').status_code == 404
    assert client.get('/api/train/canvas/layouts').get_json()['presets'] == []
    # The board itself is untouched — forgetting the memory is not tidying up.
    assert client.get('/api/train/canvas/positions').get_json()['positions'][str(ds_id)]


def test_applying_a_preset_that_does_not_exist_is_a_404(client):
    assert client.post('/api/train/canvas/layouts/9999/apply').status_code == 404


def test_the_number_of_presets_is_capped_with_a_sentence(client, app):
    from app.services.cloud_training import CANVAS_PRESET_MAX
    with app.app_context():
        ds_id = _dataset().id
    rows = {str(ds_id): [{'record_id': 1, 'x': 1, 'y': 1}]}
    for i in range(CANVAS_PRESET_MAX):
        assert _save(client, f'preset {i}', positions=rows).status_code == 200
    r = _save(client, 'one too many', positions=rows)
    assert r.status_code == 400 and 'limit' in r.get_json()['error']
    # …and overwriting an EXISTING name still works at the cap.
    assert _save(client, 'preset 0', positions=rows).status_code == 200
