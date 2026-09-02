"""◉ LoRA Canvas — where a whole LANE sits and how much room it keeps.

The third arrangeable object on this board, after the run cards and the pinned
pictures, and the one that was missing. A lane used to be pinned to x = 0 and
stacked by its TREE's height alone, while ``📌 Pin all`` lays a contact sheet
BELOW that tree: the sheet landed on the next dataset. Measured on a two-lane
board, 894 world units of the lane below were covered.

Three properties are held here:

  * a placement survives a reload — that is the whole point of storing one;
  * each gesture speaks only for its own half. A move sends {x, y}, a resize
    sends {h}, and the row keeps both. A replace instead of a merge would make
    moving a lane silently forget the room it was given, which is the failure
    the user would report as "it keeps resetting";
  * it can never stop a dataset from being deleted. Not hypothetical — a child
    table wired to face_dataset without a mapper-level relationship has no
    ordering dependency in SQLAlchemy's unit of work, the parent DELETE goes
    first, and a legacy database whose foreign key predates ON DELETE CASCADE
    answers HTTP 500. That has already shipped once in this project, so the
    deletion tests run with PRAGMA foreign_keys=OFF — the old-database shape
    where a missing flush order is fatal.
"""
from sqlalchemy import text


def _dataset(name='Ada', trigger='ada'):
    from app.services import face_dataset_service as svc
    return svc.create_dataset('local', name, trigger)


def _rows(dataset_id):
    from app.models import CanvasLanePlacement
    return CanvasLanePlacement.query.filter_by(dataset_id=dataset_id).count()


def _lanes(client):
    return {str(row['dataset_id']): row
            for row in client.get('/api/train/canvas/lanes').get_json()['lanes']}


# ---- persistence -----------------------------------------------------------

def test_an_arranged_lane_survives_a_reload(client, app):
    with app.app_context():
        ds_id = _dataset().id
    r = client.put(f'/api/dataset/{ds_id}/canvas/lane',
                   json={'x': 120.5, 'y': -340.0, 'h': 900.0})
    assert r.status_code == 200
    assert r.get_json()['saved'] == 1
    assert _lanes(client)[str(ds_id)] == {
        'dataset_id': ds_id, 'x': 120.5, 'y': -340.0, 'h': 900.0}


def test_a_lane_is_arranged_once_not_once_per_gesture(client, app):
    """Dragging the same lane again updates its row instead of adding one."""
    with app.app_context():
        ds_id = _dataset().id
    for x in (10, 20, 30):
        client.put(f'/api/dataset/{ds_id}/canvas/lane', json={'x': x, 'y': 1})
    with app.app_context():
        assert _rows(ds_id) == 1
    assert _lanes(client)[str(ds_id)]['x'] == 30.0


def test_moving_a_lane_keeps_the_room_it_was_given_and_the_other_way_round(client, app):
    """THE reason the write is a merge. Each gesture sends only its own half:
    a move that replaced the row would drop the height its owner had set, and
    the lane would silently snap back to its content — reported, rightly, as
    "the board keeps resetting what I do"."""
    with app.app_context():
        ds_id = _dataset().id
    client.put(f'/api/dataset/{ds_id}/canvas/lane', json={'h': 800})
    client.put(f'/api/dataset/{ds_id}/canvas/lane', json={'x': 40, 'y': 50})
    assert _lanes(client)[str(ds_id)] == {
        'dataset_id': ds_id, 'x': 40.0, 'y': 50.0, 'h': 800.0}
    client.put(f'/api/dataset/{ds_id}/canvas/lane', json={'h': 1200})
    assert _lanes(client)[str(ds_id)] == {
        'dataset_id': ds_id, 'x': 40.0, 'y': 50.0, 'h': 1200.0}


def test_a_reserved_height_is_clamped_into_a_usable_range(client, app):
    """The floor keeps the lane grabbable — a block shorter than its own title
    strip could never be dragged bigger again. The ceiling protects ✦ Fit: one
    lane reserving a hundred thousand units would collapse every other lane to
    a scale where nothing is readable."""
    from app.services.cloud_training import CANVAS_LANE_MAX_H, CANVAS_LANE_MIN_H
    with app.app_context():
        ds_id = _dataset().id
    client.put(f'/api/dataset/{ds_id}/canvas/lane', json={'h': 1})
    assert _lanes(client)[str(ds_id)]['h'] == CANVAS_LANE_MIN_H
    client.put(f'/api/dataset/{ds_id}/canvas/lane', json={'h': 10 ** 9})
    assert _lanes(client)[str(ds_id)]['h'] == CANVAS_LANE_MAX_H


def test_a_position_is_railed_on_both_sides_of_zero(client, app):
    """Negative is LEGAL — a lane may be parked above and left of the board's
    origin, like a pinned picture. Only the safety rail is enforced, so one
    corrupt row cannot blow the board's box up."""
    from app.services.cloud_training import CANVAS_LANE_REACH
    with app.app_context():
        ds_id = _dataset().id
    client.put(f'/api/dataset/{ds_id}/canvas/lane', json={'x': -10 ** 9, 'y': 10 ** 9})
    lane = _lanes(client)[str(ds_id)]
    assert (lane['x'], lane['y']) == (-CANVAS_LANE_REACH, CANVAS_LANE_REACH)


def test_unusable_numbers_are_refused_not_stored(client, app):
    """A lane parked at NaN would be unreachable on every future load and there
    is no UI to repair it. Absent, NaN and infinite all mean the same thing
    here, and it is not zero: let the board decide."""
    with app.app_context():
        ds_id = _dataset().id
    r = client.put(f'/api/dataset/{ds_id}/canvas/lane',
                   json={'x': float('nan'), 'y': 3, 'h': float('inf')})
    assert r.status_code == 200
    assert r.get_json()['placement'] is None
    with app.app_context():
        assert _rows(ds_id) == 0


def test_half_a_position_is_no_position(client, app):
    """A lane with a y and no x would sit at the board's left edge for reasons
    nobody could read off the row."""
    with app.app_context():
        ds_id = _dataset().id
    client.put(f'/api/dataset/{ds_id}/canvas/lane', json={'y': 300, 'h': 400})
    assert _lanes(client)[str(ds_id)] == {'dataset_id': ds_id, 'h': 400.0}


def test_writing_nothing_usable_takes_the_lane_back_to_automatic(client, app):
    """"No placement" stays ONE state. A row of three NULLs would be a second
    way of saying the same thing, free to disagree with the first."""
    with app.app_context():
        ds_id = _dataset().id
    client.put(f'/api/dataset/{ds_id}/canvas/lane', json={'x': 5, 'y': 6, 'h': 700})
    r = client.put(f'/api/dataset/{ds_id}/canvas/lane',
                   json={'x': None, 'y': None, 'h': None})
    assert r.status_code == 200
    assert r.get_json() == {'saved': 0, 'placement': None}
    with app.app_context():
        assert _rows(ds_id) == 0


def test_tidy_up_hands_the_lane_back_to_the_stack(client, app):
    with app.app_context():
        ds_id = _dataset().id
    client.put(f'/api/dataset/{ds_id}/canvas/lane', json={'x': 1, 'y': 2, 'h': 300})
    r = client.delete(f'/api/dataset/{ds_id}/canvas/lane')
    assert r.status_code == 200
    assert r.get_json()['cleared'] == 1
    assert client.get('/api/train/canvas/lanes').get_json()['lanes'] == []


def test_tidy_up_on_an_untouched_lane_is_a_no_op_not_an_error(client, app):
    with app.app_context():
        ds_id = _dataset().id
    r = client.delete(f'/api/dataset/{ds_id}/canvas/lane')
    assert r.status_code == 200
    assert r.get_json()['cleared'] == 0


def test_an_unknown_dataset_is_404_on_both_write_paths(client, app):
    assert client.put('/api/dataset/9999/canvas/lane', json={'h': 300}).status_code == 404
    assert client.delete('/api/dataset/9999/canvas/lane').status_code == 404


def test_another_users_lane_is_not_on_my_board(client, app):
    from app.extensions import db
    from app.models import CanvasLanePlacement
    from app.services import face_dataset_service as svc
    with app.app_context():
        mine = _dataset('Mine', 'mine').id
        theirs = svc.create_dataset('someone-else', 'Theirs', 'theirs').id
        db.session.add(CanvasLanePlacement(dataset_id=mine, h=400))
        db.session.add(CanvasLanePlacement(dataset_id=theirs, h=400))
        db.session.commit()
    assert set(_lanes(client)) == {str(mine)}


# ---- layout presets --------------------------------------------------------

def test_a_saved_layout_puts_the_lanes_back_too(client, app):
    """A preset that restored the cards and the pictures onto a board whose
    lanes had since been rearranged would put an arrangement back into the
    wrong room. The geometry is only half the memory."""
    with app.app_context():
        ds_id = _dataset().id
    saved = client.post('/api/train/canvas/layouts', json={
        'name': 'wide', 'lanes': {str(ds_id): {'x': 300, 'y': 40, 'h': 1100}}})
    assert saved.status_code == 200
    assert saved.get_json()['preset']['lanes'] == 1

    # The live board is rearranged since…
    client.put(f'/api/dataset/{ds_id}/canvas/lane', json={'x': 0, 'y': 0, 'h': 200})
    preset_id = client.get('/api/train/canvas/layouts').get_json()['presets'][0]['id']
    applied = client.post(f'/api/train/canvas/layouts/{preset_id}/apply', json={})
    assert applied.status_code == 200
    assert applied.get_json()['applied']['lanes'] == 1
    assert _lanes(client)[str(ds_id)] == {
        'dataset_id': ds_id, 'x': 300.0, 'y': 40.0, 'h': 1100.0}


# ---- the delete-500 guard --------------------------------------------------

def test_deleting_a_dataset_that_has_a_lane_placement_does_not_500(client, app):
    """foreign_keys=OFF reproduces a database created before ON DELETE CASCADE:
    if the parent DELETE were emitted first the child would be left dangling and
    an enforcing database would raise."""
    from app.extensions import db
    from app.models import CanvasLanePlacement
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds_id = _dataset().id
        db.session.add(CanvasLanePlacement(dataset_id=ds_id, x=1, y=2, h=300))
        db.session.commit()
        assert _rows(ds_id) == 1

        db.session.execute(text('PRAGMA foreign_keys=OFF'))
        assert db.session.execute(text('PRAGMA foreign_keys')).scalar_one() == 0
        db.session.commit()

        assert svc.delete_dataset('local', ds_id) is True
        assert _rows(ds_id) == 0
        assert svc.get_dataset('local', ds_id) is None


def test_deleting_a_dataset_through_the_api_still_answers_200(client, app):
    from app.extensions import db
    from app.models import CanvasLanePlacement
    with app.app_context():
        ds_id = _dataset().id
        db.session.add(CanvasLanePlacement(dataset_id=ds_id, x=9, y=9, h=900))
        db.session.commit()
    assert client.post(f'/api/dataset/{ds_id}/delete').status_code == 200
    with app.app_context():
        assert _rows(ds_id) == 0


def test_the_lane_model_declares_its_relationship_to_face_dataset():
    """A contract, not a style check: without the mapper-level relationship the
    explicit cleanup in delete_dataset becomes the ONLY thing between a user and
    an HTTP 500 — one refactor away from the bug coming back."""
    from sqlalchemy import inspect
    from app.models import CanvasLanePlacement, FaceDataset
    rels = inspect(CanvasLanePlacement).relationships
    assert 'dataset' in rels, 'CanvasLanePlacement must declare relationship() to face_dataset'
    assert rels['dataset'].mapper.class_ is FaceDataset


def test_the_clamps_agree_with_the_browsers(client, app):
    """The board clamps in JavaScript and the server clamps again, and the two
    numbers have to be the same one — a value the browser accepts and the server
    silently rewrites is a lane that jumps on the next reload."""
    import re
    from pathlib import Path
    from app.services.cloud_training import (
        CANVAS_LANE_MAX_H, CANVAS_LANE_MIN_H, CANVAS_LANE_REACH)
    src = (Path(__file__).resolve().parents[2]
           / 'frontend' / 'src' / 'utils' / 'canvasLanePlacement.js').read_text(encoding='utf-8')
    found = dict(re.findall(r'export const (LANE_\w+) = (\d+);', src))
    assert float(found['LANE_MIN_H']) == CANVAS_LANE_MIN_H
    assert float(found['LANE_MAX_H']) == CANVAS_LANE_MAX_H
    assert float(found['LANE_REACH']) == CANVAS_LANE_REACH
