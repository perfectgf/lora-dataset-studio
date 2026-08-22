"""🗑 Deleting images from a checkpoint's gallery.

A checkpoint can accumulate dozens of test renders and most of them are misses;
until now the gallery could only show them. Deleting one is a REAL delete — the
row is the Test Studio's own cell, so it leaves both surfaces at once — which is
why every guarantee below is asserted rather than assumed:

  * the file leaves the dataset folder the RECOVERABLE way (OS trash → the app's
    own trash → a permanent unlink only when both refuse), and the mode is
    announced by the gallery BEFORE the click;
  * no orphan is left behind: neither a row whose file is gone, nor a
    ``checkpoint_preview`` pointer aimed at a deleted row;
  * an install that is not tidy — a file already missing, a cell still
    generating, two rows over one file, an id from another checkpoint — degrades
    with a reported reason instead of destroying the wrong thing.

Both disposal modes are exercised deterministically by injecting a fake / absent
``send2trash`` module (the real one would move test files to the OS recycle bin).
"""
import os
import sys
import types


def _create(client, name='Nova', trigger='nova'):
    return client.post('/api/dataset/create',
                       json={'name': name, 'trigger_word': trigger}).get_json()['id']


def _record(db, dataset_id):
    from app.models import TrainingRunRecord
    rec = TrainingRunRecord(dataset_id=dataset_id, family='krea', source='local',
                            fingerprint='f', version=1)
    db.session.add(rec)
    db.session.commit()
    return rec


def _image(db, dataset_id, rec, step, filename, status='done', write=True):
    """A gallery row plus, unless told otherwise, its file on disk."""
    from app.models import LoraTestImage
    from app.services import face_dataset_service as fds
    if write and filename:
        folder = fds._dataset_path(dataset_id)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, filename), 'wb') as fh:
            fh.write(b'PNG-ish')
    img = LoraTestImage(dataset_id=dataset_id, checkpoint='a.safetensors',
                        strength=1.0, status=status, filename=filename,
                        record_id=rec.id, step=step)
    db.session.add(img)
    db.session.commit()
    return img


def _gone(db, image_id) -> bool:
    """A bulk delete leaves the stale instance in the session's identity map, so
    reading ``img.id`` back would raise ObjectDeletedError — ask the table, with
    the id captured BEFORE the delete."""
    from app.models import LoraTestImage
    return LoraTestImage.query.filter_by(id=image_id).count() == 0


def _path(dataset_id, filename):
    from app.services import face_dataset_service as fds
    return os.path.join(fds._dataset_path(dataset_id), filename)


def _force_trash(monkeypatch):
    """Take the send2trash branch, routed to os.remove so nothing reaches the
    real OS recycle bin during a test run."""
    monkeypatch.setitem(sys.modules, 'send2trash',
                        types.SimpleNamespace(send2trash=lambda p: os.remove(p)))


def _force_app_trash(monkeypatch):
    """No send2trash — what a DEFAULT install actually gets."""
    monkeypatch.setitem(sys.modules, 'send2trash', None)


# --- the core contract -------------------------------------------------------

def test_deleting_removes_the_row_AND_the_file_and_leaves_no_orphan(
        client, app, monkeypatch):
    """THE guard: after a delete there is no row without a file, no file without
    a row, and no preview pointer aimed at nothing."""
    _force_trash(monkeypatch)
    from app.extensions import db
    from app.models import CheckpointPreview
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        doomed = _image(db, ds, rec, 1000, 'bad.png')
        kept = _image(db, ds, rec, 1000, 'good.png')
        doomed_id, kept_id = doomed.id, kept.id
        db.session.add(CheckpointPreview(record_id=rec.id, step=1000, dataset_id=ds,
                                         lora_test_image_id=doomed_id, prompt=''))
        db.session.add(CheckpointPreview(record_id=rec.id, step=1000, dataset_id=ds,
                                         lora_test_image_id=kept_id, prompt=''))
        db.session.commit()

        out = ct.delete_checkpoint_images(rec.id, 1000, [doomed_id])
        assert out['rows_removed'] == 1
        assert out['trashed'] == 1 and out['deleted'] == 0
        assert out['mode'] == 'trash'
        assert out['dataset_ids'] == [ds]
        assert out['skipped'] == []

        # The row is gone, the file is gone, the neighbour is untouched.
        assert _gone(db, doomed_id)
        assert not os.path.exists(_path(ds, 'bad.png'))
        assert not _gone(db, kept_id)
        assert os.path.exists(_path(ds, 'good.png'))

        # No dangling pointer: the preview aimed at the deleted row is gone too,
        # the one aimed at the survivor is not.
        assert out['previews_removed'] == 1
        left = CheckpointPreview.query.all()
        assert [p.lora_test_image_id for p in left] == [kept_id]

        # And the gallery agrees.
        gal = ct.checkpoint_gallery(rec.id, 1000)
        assert gal['count'] == 1
        assert [i['id'] for i in gal['images']] == [kept_id]


def test_a_deleted_image_leaves_the_test_studio_too(client, app, monkeypatch):
    """The consequence the confirmation has to state: one row, two surfaces."""
    _force_trash(monkeypatch)
    from app import config as cfg
    from app.extensions import db
    from app.services import cloud_training as ct
    from app.services import lora_test_studio as lts
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        img_id = _image(db, ds, rec, 1000, 'cell.png').id
        before = lts.studio_payload(cfg.LOCAL_USER, ds)
        assert any(c['id'] == img_id for c in before['cells'])

        ct.delete_checkpoint_images(rec.id, 1000, [img_id])
        after = lts.studio_payload(cfg.LOCAL_USER, ds)
        assert not any(c['id'] == img_id for c in after['cells'])


def test_without_send2trash_the_file_stays_recoverable_in_the_app_trash(
        client, app, monkeypatch):
    """The branch most installs take: the file is MOVED into data/trash, not
    destroyed — recoverable until the user empties it from Settings."""
    _force_app_trash(monkeypatch)
    from app.extensions import db
    from app.services import cloud_training as ct
    from app.services import trash
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        img_id = _image(db, ds, rec, 1000, 'oops.png').id
        out = ct.delete_checkpoint_images(rec.id, 1000, [img_id])
        assert out['mode'] == 'app_trash'
        assert out['trashed'] == 1
        assert not os.path.exists(_path(ds, 'oops.png'))
        recovered = [p for p in trash.trash_root().rglob('oops.png')]
        assert len(recovered) == 1
        assert recovered[0].read_bytes() == b'PNG-ish'


def test_the_gallery_announces_where_a_deleted_image_would_go(client, app, monkeypatch):
    """The confirmation must name the outcome BEFORE the button is armed, so the
    gallery payload carries the mode the deletion will actually resolve."""
    from app.services import cloud_training as ct
    with app.app_context():
        _force_app_trash(monkeypatch)
        assert ct.checkpoint_gallery(1, 1000)['delete_mode'] == 'app_trash'
        _force_trash(monkeypatch)
        assert ct.checkpoint_gallery(1, 1000)['delete_mode'] == 'trash'


def test_several_images_go_in_one_call(client, app, monkeypatch):
    """32 renders on a checkpoint: one confirmation, not thirty-two."""
    _force_trash(monkeypatch)
    from app.extensions import db
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        ids = [_image(db, ds, rec, 1000, f'n{i}.png').id for i in range(5)]
        out = ct.delete_checkpoint_images(rec.id, 1000, ids[:3])
        assert out['rows_removed'] == 3 and out['trashed'] == 3
        assert ct.checkpoint_gallery(rec.id, 1000)['count'] == 2


# --- an install that is not tidy ---------------------------------------------

def test_a_file_already_missing_still_loses_its_row(client, app, monkeypatch):
    """Deleted from the file explorer behind the app's back: the click must still
    clean the database rather than fail on a missing path."""
    _force_trash(monkeypatch)
    from app.extensions import db
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        img_id = _image(db, ds, rec, 1000, 'ghost.png').id
        os.remove(_path(ds, 'ghost.png'))
        out = ct.delete_checkpoint_images(rec.id, 1000, [img_id])
        assert out['already_absent'] == 1
        assert out['rows_removed'] == 1
        assert _gone(db, img_id)


def test_a_row_with_no_file_at_all_is_handled(client, app, monkeypatch):
    """A cancelled job never wrote a file (`filename` NULL). Only the row goes,
    and nothing tries to trash None."""
    _force_trash(monkeypatch)
    from app.extensions import db
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        img_id = _image(db, ds, rec, 1000, None, status='cancelled').id
        out = ct.delete_checkpoint_images(rec.id, 1000, [img_id])
        assert out['rows_removed'] == 1
        assert out['trashed'] == 0 and out['deleted'] == 0
        assert _gone(db, img_id)


def test_an_image_still_generating_is_refused_not_cancelled(client, app, monkeypatch):
    """A delete click never kills someone's running job — it says why it passed."""
    _force_trash(monkeypatch)
    from app.extensions import db
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        img = _image(db, ds, rec, 1000, None, status='pending', write=False)
        out = ct.delete_checkpoint_images(rec.id, 1000, [img.id])
        assert out['rows_removed'] == 0
        assert out['skipped'] == [{'id': img.id, 'reason': 'generating'}]
        assert not _gone(db, img.id)


def test_an_id_from_another_checkpoint_is_refused(client, app, monkeypatch):
    """Scoping, not trust: the route cannot be turned into 'delete any image'."""
    _force_trash(monkeypatch)
    from app.extensions import db
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        mine_id = _image(db, ds, rec, 1000, 'mine.png').id
        other_id = _image(db, ds, rec, 2000, 'other.png').id
        out = ct.delete_checkpoint_images(rec.id, 1000, [mine_id, other_id])
        assert out['rows_removed'] == 1
        assert out['skipped'] == [{'id': other_id, 'reason': 'not_in_gallery'}]
        assert not _gone(db, other_id)
        assert os.path.exists(_path(ds, 'other.png'))


def test_a_file_another_row_still_shows_is_kept_on_disk(client, app, monkeypatch):
    """Two rows over one picture (a preview reusing an existing cell): deleting
    one must not amputate the other's thumbnail."""
    _force_trash(monkeypatch)
    from app.extensions import db
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        a_id = _image(db, ds, rec, 1000, 'shared.png').id
        b_id = _image(db, ds, rec, 2000, 'shared.png', write=False).id
        out = ct.delete_checkpoint_images(rec.id, 1000, [a_id])
        assert out['rows_removed'] == 1
        assert out['trashed'] == 0            # nothing left the disk
        assert os.path.exists(_path(ds, 'shared.png'))
        assert not _gone(db, b_id)


def test_a_locked_file_keeps_its_row_so_it_stays_retryable(client, app, monkeypatch):
    """A file the OS refuses to move must not vanish from the UI while it sits on
    disk — the row survives, the failure is reported."""
    from app.extensions import db
    from app.services import cloud_training as ct
    from app.services import trash
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        img = _image(db, ds, rec, 1000, 'locked.png')
        monkeypatch.setattr(trash, 'dispose',
                            lambda *a, **k: (_ for _ in ()).throw(OSError('locked')))
        out = ct.delete_checkpoint_images(rec.id, 1000, [img.id])
        assert out['rows_removed'] == 0
        assert out['skipped'][0]['id'] == img.id
        assert not _gone(db, img.id)
        assert os.path.exists(_path(ds, 'locked.png'))


def test_an_empty_selection_is_a_no_op_not_an_error(client, app):
    from app.services import cloud_training as ct
    with app.app_context():
        out = ct.delete_checkpoint_images(1, 1000, [])
        assert out['rows_removed'] == 0 and out['skipped'] == []


# --- the route ---------------------------------------------------------------

def test_the_route_deletes_and_reports_what_happened(client, app, monkeypatch):
    _force_trash(monkeypatch)
    from app.extensions import db
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        img = _image(db, ds, rec, 1000, 'route.png')
        rid, iid = rec.id, img.id

    r = client.post(f'/api/train/checkpoint/{rid}/1000/images/delete',
                    json={'image_ids': [iid]})
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is True and body['rows_removed'] == 1
    assert body['mode'] == 'trash'
    with app.app_context():
        assert _gone(db, iid)


def test_the_route_rejects_a_malformed_body(client, app):
    r = client.post('/api/train/checkpoint/1/1000/images/delete',
                    json={'image_ids': 'all'})
    assert r.status_code == 400
