"""🖼 LoRA Canvas — images pinned onto the board (canvas_image_node).

The feature's promise is narrow and testable: close a pinned image, re-open it,
and it comes back at the SAME place and the SAME size. That is the assertion
this file exists for; everything else here guards the ways a stored pointer can
rot — the image gets deleted, the dataset gets deleted, the geometry is
nonsense, or the row belongs to somebody else's lane.
"""
from sqlalchemy import text


def _dataset(name='Ada', trigger='ada'):
    from app.services import face_dataset_service as svc
    return svc.create_dataset('local', name, trigger)


def _image(dataset_id, record_id=7, step=2500, filename='a.png', status='done'):
    """One finished Test-Studio cell — what a pinned node points at."""
    from app.extensions import db
    from app.models import LoraTestImage
    row = LoraTestImage(dataset_id=dataset_id, checkpoint='z image\\Ada-2500.safetensors',
                        strength=1.0, filename=filename, status=status,
                        seed=208607443, prompt='a portrait', step=step,
                        record_id=record_id, cfg=3.5, steps=20, sampler='euler',
                        scheduler='simple', z_model='zturbo.safetensors')
    db.session.add(row)
    db.session.commit()
    return row.id


def _lane(client, dataset_id):
    return client.get('/api/train/canvas/images').get_json()['nodes'].get(str(dataset_id), [])


# ---- THE assertion: close, re-open, same place and same size ---------------

def test_a_closed_pinned_image_reopens_at_the_position_and_size_it_was_closed_at(client, app):
    with app.app_context():
        ds_id = _dataset().id
        img_id = _image(ds_id)

    # Pinned, then dragged and resized somewhere deliberate.
    client.put(f'/api/dataset/{ds_id}/canvas/images', json={'nodes': [
        {'image_id': img_id, 'x': 640.0, 'y': 275.5, 'w': 420.0, 'h': 310.0,
         'visible': True}]})

    # Closed — the geometry must NOT be forgotten.
    r = client.put(f'/api/dataset/{ds_id}/canvas/images', json={'nodes': [
        {'image_id': img_id, 'x': 640.0, 'y': 275.5, 'w': 420.0, 'h': 310.0,
         'visible': False}]})
    assert r.status_code == 200
    closed = _lane(client, ds_id)
    assert len(closed) == 1, 'closing must keep the row, not delete it'
    assert closed[0]['visible'] is False
    assert (closed[0]['x'], closed[0]['y']) == (640.0, 275.5)
    assert (closed[0]['w'], closed[0]['h']) == (420.0, 310.0)

    # Re-opened from the gallery: the client sends back what it read.
    node = closed[0]
    client.put(f'/api/dataset/{ds_id}/canvas/images', json={'nodes': [
        {'image_id': img_id, 'x': node['x'], 'y': node['y'],
         'w': node['w'], 'h': node['h'], 'visible': True}]})
    again = _lane(client, ds_id)[0]
    assert again['visible'] is True
    assert (again['x'], again['y'], again['w'], again['h']) == (640.0, 275.5, 420.0, 310.0)


def test_a_pinned_node_carries_the_image_row_so_a_lane_needs_no_second_fetch(client, app):
    with app.app_context():
        ds_id = _dataset().id
        img_id = _image(ds_id, record_id=7, step=2500)
    client.put(f'/api/dataset/{ds_id}/canvas/images', json={'nodes': [
        {'image_id': img_id, 'x': 10, 'y': 10, 'w': 200, 'h': 200}]})
    img = _lane(client, ds_id)[0]['image']
    # The link to the source checkpoint is READ from the image, never stored on
    # the node — the two can therefore not disagree.
    assert img['record_id'] == 7 and img['step'] == 2500
    assert img['url'].endswith('/img/a.png')
    # And the settings the lightbox now shows are actually published.
    assert img['sampler'] == 'euler' and img['cfg'] == 3.5
    assert img['base_model'] == 'zturbo.safetensors'


# ---- rot: the pointer outlives what it points at ---------------------------

def test_a_pinned_image_that_was_deleted_leaves_the_board_instead_of_a_ghost(client, app):
    """The bug that only shows up in week 3: a node rendering a picture that no
    longer exists. The read prunes it — the board loses the node, nothing else."""
    from app.extensions import db
    from app.models import CanvasImageNode, LoraTestImage
    with app.app_context():
        ds_id = _dataset().id
        img_id = _image(ds_id)
    client.put(f'/api/dataset/{ds_id}/canvas/images', json={'nodes': [
        {'image_id': img_id, 'x': 10, 'y': 10, 'w': 200, 'h': 200}]})

    with app.app_context():
        db.session.delete(db.session.get(LoraTestImage, img_id))
        db.session.commit()

    body = client.get('/api/train/canvas/images').get_json()
    assert body['nodes'] == {}
    assert body['pruned'] == 1
    with app.app_context():
        assert CanvasImageNode.query.filter_by(image_id=img_id).count() == 0


def test_an_image_whose_render_never_finished_is_not_a_node(client, app):
    with app.app_context():
        ds_id = _dataset().id
        img_id = _image(ds_id, filename=None, status='pending')
    client.put(f'/api/dataset/{ds_id}/canvas/images', json={'nodes': [
        {'image_id': img_id, 'x': 10, 'y': 10, 'w': 200, 'h': 200}]})
    assert _lane(client, ds_id) == []


# ---- refusals --------------------------------------------------------------

def test_an_image_from_another_dataset_cannot_be_pinned_into_this_lane(client, app):
    with app.app_context():
        mine = _dataset('Mine', 'mine').id
        other = _dataset('Other', 'other').id
        foreign = _image(other)
    r = client.put(f'/api/dataset/{mine}/canvas/images', json={'nodes': [
        {'image_id': foreign, 'x': 5, 'y': 5, 'w': 200, 'h': 200}]})
    assert r.status_code == 200 and r.get_json()['saved'] == 0
    assert _lane(client, mine) == []


def test_a_giant_or_nonsense_size_is_clamped_not_stored(client, app):
    """An 8 000-px node would make ✦ Fit collapse the board to an unreadable
    scale; a NaN would make it unreachable forever."""
    from app.services.cloud_training import CANVAS_IMAGE_MAX, CANVAS_IMAGE_MIN
    with app.app_context():
        ds_id = _dataset().id
        big = _image(ds_id, filename='b.png')
        tiny = _image(ds_id, filename='c.png')
        bad = _image(ds_id, filename='d.png')
    client.put(f'/api/dataset/{ds_id}/canvas/images', json={'nodes': [
        {'image_id': big, 'x': 0, 'y': 0, 'w': 8000, 'h': 9000},
        {'image_id': tiny, 'x': -50, 'y': -50, 'w': 2, 'h': 2},
        {'image_id': bad, 'x': 'nope', 'y': 0, 'w': 200, 'h': 200}]})
    lane = {n['image_id']: n for n in _lane(client, ds_id)}
    assert (lane[big]['w'], lane[big]['h']) == (CANVAS_IMAGE_MAX, CANVAS_IMAGE_MAX)
    assert (lane[tiny]['w'], lane[tiny]['h']) == (CANVAS_IMAGE_MIN, CANVAS_IMAGE_MIN)
    assert bad not in lane


# ---- free placement: the lane anchors a picture, it does not cage it -------

def test_a_picture_parked_above_and_left_of_its_lane_survives_the_round_trip(client, app):
    """THE assertion of free placement. Coordinates used to be floored at zero,
    so a render could be dragged down and right but never up or left: its own
    lane's corner was a wall. A picture is not a step of the lineage — the link
    to the checkpoint that made it is read off the image row, never off its
    position — so no coordinate can make it lie about where it came from."""
    with app.app_context():
        ds_id = _dataset().id
        img_id = _image(ds_id)
    client.put(f'/api/dataset/{ds_id}/canvas/images', json={'nodes': [
        {'image_id': img_id, 'x': -1840.5, 'y': -920.25, 'w': 320, 'h': 320}]})
    node = _lane(client, ds_id)[0]
    assert (node['x'], node['y']) == (-1840.5, -920.25)
    # …and it is still joined to the checkpoint that produced it.
    assert (node['image']['record_id'], node['image']['step']) == (7, 2500)


def test_the_reach_is_a_rail_on_both_sides_so_one_row_cannot_collapse_fit(client, app):
    """The position axes had no ceiling at all before: only the SIZE was
    bounded, so one corrupt row could blow a lane's extent up and collapse
    ✦ Fit to a scale where nothing on the board is readable."""
    from app.services.cloud_training import CANVAS_IMAGE_REACH
    with app.app_context():
        ds_id = _dataset().id
        far = _image(ds_id, filename='far.png')
        back = _image(ds_id, filename='back.png')
    client.put(f'/api/dataset/{ds_id}/canvas/images', json={'nodes': [
        {'image_id': far, 'x': 1e9, 'y': 5e12, 'w': 320, 'h': 320},
        {'image_id': back, 'x': -1e9, 'y': -5e12, 'w': 320, 'h': 320}]})
    lane = {n['image_id']: n for n in _lane(client, ds_id)}
    assert (lane[far]['x'], lane[far]['y']) == (CANVAS_IMAGE_REACH, CANVAS_IMAGE_REACH)
    assert (lane[back]['x'], lane[back]['y']) == (-CANVAS_IMAGE_REACH, -CANVAS_IMAGE_REACH)


def test_a_board_written_before_free_placement_reads_back_byte_for_byte(client, app):
    """NO MIGRATION — and this is what says so.

    Lane-local is the ANCHOR, not the cage, so relaxing the floor changed what a
    picture may DO, never what its stored numbers MEAN. Every row ever written
    is still a lane-local offset and still inside the rail, so a database that
    predates this change draws exactly the board it always drew. (Board-absolute
    coordinates would have needed the opposite: a migration adding each lane's
    world origin — a number the server cannot even compute, because lane heights
    are laid out in the browser and depend on which datasets are ticked.)"""
    from app.extensions import db
    from app.models import CanvasImageNode
    with app.app_context():
        ds_id = _dataset().id
        img_id = _image(ds_id)
        # A row exactly as an older build would have left it: no group columns,
        # geometry in the old non-negative half.
        db.session.add(CanvasImageNode(dataset_id=ds_id, image_id=img_id,
                                       x=312.0, y=40.0, w=320.0, h=320.0,
                                       visible=True))
        db.session.commit()
    before = _lane(client, ds_id)[0]
    assert (before['x'], before['y'], before['w'], before['h']) == (312.0, 40.0, 320.0, 320.0)
    assert (before['group_id'], before['group_pos']) == (None, None)
    # Re-reading is idempotent: nothing is rewritten, nudged or defaulted on the
    # way through the new clamp.
    after = _lane(client, ds_id)[0]
    assert after == before


def test_an_unknown_dataset_is_404_on_both_write_paths(client, app):
    assert client.put('/api/dataset/9999/canvas/images',
                      json={'nodes': []}).status_code == 404
    assert client.delete('/api/dataset/9999/canvas/images').status_code == 404


def test_pinned_images_of_another_user_are_not_on_my_board(client, app):
    from app.extensions import db
    from app.models import CanvasImageNode
    from app.services import face_dataset_service as svc
    with app.app_context():
        mine = _dataset('Mine', 'mine').id
        theirs = svc.create_dataset('someone-else', 'Theirs', 'theirs').id
        mine_img = _image(mine, filename='m.png')
        their_img = _image(theirs, filename='t.png')
        db.session.add(CanvasImageNode(dataset_id=mine, image_id=mine_img,
                                       x=1, y=1, w=200, h=200))
        db.session.add(CanvasImageNode(dataset_id=theirs, image_id=their_img,
                                       x=2, y=2, w=200, h=200))
        db.session.commit()
    assert set(client.get('/api/train/canvas/images')
               .get_json()['nodes']) == {str(mine)}


def test_forgetting_a_lane_clears_it_and_is_a_no_op_when_empty(client, app):
    with app.app_context():
        ds_id = _dataset().id
        img_id = _image(ds_id)
    client.put(f'/api/dataset/{ds_id}/canvas/images', json={'nodes': [
        {'image_id': img_id, 'x': 1, 'y': 1, 'w': 200, 'h': 200}]})
    assert client.delete(f'/api/dataset/{ds_id}/canvas/images').get_json()['cleared'] == 1
    assert client.delete(f'/api/dataset/{ds_id}/canvas/images').get_json()['cleared'] == 0


# ---- the delete-500 guard --------------------------------------------------

def test_deleting_a_dataset_that_has_pinned_images_does_not_500(client, app):
    """Same trap as canvas_node_position: foreign_keys=OFF reproduces a database
    created before ON DELETE CASCADE, where a parent-first DELETE is fatal."""
    from app.extensions import db
    from app.models import CanvasImageNode
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds_id = _dataset().id
        img_id = _image(ds_id)
    client.put(f'/api/dataset/{ds_id}/canvas/images', json={'nodes': [
        {'image_id': img_id, 'x': 3, 'y': 4, 'w': 200, 'h': 200}]})
    with app.app_context():
        db.session.execute(text('PRAGMA foreign_keys=OFF'))
        svc.delete_dataset('local', ds_id)
        assert CanvasImageNode.query.filter_by(dataset_id=ds_id).count() == 0


# ---- 🖼🖼 fused into one node, side by side --------------------------------

def test_a_group_survives_a_reload_with_its_order_intact(client, app):
    """Drop one pinned image onto another and they become ONE node. What the
    board needs back on the next load is exactly two extra fields per row."""
    with app.app_context():
        ds_id = _dataset().id
        a = _image(ds_id, filename='a.png')
        b = _image(ds_id, filename='b.png')
        c = _image(ds_id, filename='c.png')
    client.put(f'/api/dataset/{ds_id}/canvas/images', json={'nodes': [
        {'image_id': a, 'x': 100, 'y': 100, 'w': 320, 'h': 320,
         'group_id': 'g1', 'group_pos': 0},
        {'image_id': b, 'x': 700, 'y': 100, 'w': 480, 'h': 320,
         'group_id': 'g1', 'group_pos': 1},
        {'image_id': c, 'x': 900, 'y': 500, 'w': 320, 'h': 640,
         'group_id': 'g1', 'group_pos': 2}]})
    lane = {n['image_id']: n for n in _lane(client, ds_id)}
    assert [lane[i]['group_id'] for i in (a, b, c)] == ['g1', 'g1', 'g1']
    assert [lane[i]['group_pos'] for i in (a, b, c)] == [0, 1, 2]
    # And each member still remembers its OWN box — that is what it gets back
    # the day it is dragged out again.
    assert (lane[c]['w'], lane[c]['h']) == (320.0, 640.0)


def test_a_board_that_never_grouped_anything_reads_null_not_a_group(client, app):
    with app.app_context():
        ds_id = _dataset().id
        img_id = _image(ds_id)
    client.put(f'/api/dataset/{ds_id}/canvas/images', json={'nodes': [
        {'image_id': img_id, 'x': 10, 'y': 20, 'w': 200, 'h': 200}]})
    node = _lane(client, ds_id)[0]
    assert node['group_id'] is None
    assert node['group_pos'] is None


def test_a_plain_move_never_dissolves_a_group(client, app):
    """A row that does not MENTION the group fields keeps them. Otherwise every
    code path that only knows about geometry — an older client, a re-flow —
    would silently take the board's groups apart."""
    with app.app_context():
        ds_id = _dataset().id
        a = _image(ds_id, filename='a.png')
        b = _image(ds_id, filename='b.png')
    client.put(f'/api/dataset/{ds_id}/canvas/images', json={'nodes': [
        {'image_id': a, 'x': 0, 'y': 0, 'w': 300, 'h': 300, 'group_id': 'g1', 'group_pos': 0},
        {'image_id': b, 'x': 0, 'y': 0, 'w': 300, 'h': 300, 'group_id': 'g1', 'group_pos': 1}]})
    client.put(f'/api/dataset/{ds_id}/canvas/images', json={'nodes': [
        {'image_id': a, 'x': 55, 'y': 66, 'w': 300, 'h': 300, 'visible': True}]})
    lane = {n['image_id']: n for n in _lane(client, ds_id)}
    assert lane[a]['group_id'] == 'g1', 'the move must not have dropped the membership'
    assert (lane[a]['x'], lane[a]['y']) == (55.0, 66.0)


def test_taking_an_image_out_of_a_group_is_a_null_group_id(client, app):
    with app.app_context():
        ds_id = _dataset().id
        a = _image(ds_id, filename='a.png')
        b = _image(ds_id, filename='b.png')
    client.put(f'/api/dataset/{ds_id}/canvas/images', json={'nodes': [
        {'image_id': a, 'x': 0, 'y': 0, 'w': 300, 'h': 300, 'group_id': 'g1', 'group_pos': 0},
        {'image_id': b, 'x': 0, 'y': 0, 'w': 300, 'h': 300, 'group_id': 'g1', 'group_pos': 1}]})
    # b dragged out; a is left alone, so the group dissolves for both.
    client.put(f'/api/dataset/{ds_id}/canvas/images', json={'nodes': [
        {'image_id': b, 'x': 900, 'y': 900, 'w': 300, 'h': 300,
         'group_id': None, 'group_pos': None},
        {'image_id': a, 'x': 0, 'y': 0, 'w': 300, 'h': 300,
         'group_id': None, 'group_pos': None}]})
    lane = {n['image_id']: n for n in _lane(client, ds_id)}
    assert lane[a]['group_id'] is None and lane[b]['group_id'] is None
    assert lane[b]['group_pos'] is None, 'a position with no group is not a state'


def test_a_hostile_group_id_cannot_poison_the_column(client, app):
    with app.app_context():
        ds_id = _dataset().id
        img_id = _image(ds_id)
    client.put(f'/api/dataset/{ds_id}/canvas/images', json={'nodes': [
        {'image_id': img_id, 'x': 0, 'y': 0, 'w': 300, 'h': 300,
         'group_id': 'g' * 500, 'group_pos': 'not a number'}]})
    node = _lane(client, ds_id)[0]
    assert len(node['group_id']) <= 40
    assert node['group_pos'] == 0
    # An empty id is "in no group", position included.
    client.put(f'/api/dataset/{ds_id}/canvas/images', json={'nodes': [
        {'image_id': img_id, 'x': 0, 'y': 0, 'w': 300, 'h': 300,
         'group_id': '   ', 'group_pos': 4}]})
    node = _lane(client, ds_id)[0]
    assert node['group_id'] is None and node['group_pos'] is None


def test_the_group_columns_are_added_to_a_database_that_predates_them(app):
    """The additive-migration path, on the real table: an install that has been
    pinning images since before groups existed must gain the columns on boot,
    not be told its board is corrupt."""
    from app.extensions import db
    with app.app_context():
        cols = {r[1] for r in db.session.execute(
            text('PRAGMA table_info(canvas_image_node)'))}
        assert {'group_id', 'group_pos'} <= cols
