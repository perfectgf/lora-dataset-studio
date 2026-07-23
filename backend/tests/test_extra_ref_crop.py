"""Cropping an EXTRA reference (the additional angles beside the main photo).

Extras used to be import-or-delete only. They can now be re-framed with the same
editor as the primary reference, which raises the same question the primary already
answered: does a crop widen back out, or does each pass eat into the last one?

Design under test:
  - a full-frame ORIGINAL is kept beside every extra, named by convention
    (`..._datasetrefx_<id>` -> `..._datasetrefxorig_<id>`) — no schema change, since
    extras live in a JSON list of names inside a long-frozen table;
  - extras imported BEFORE that (no original on disk) are snapshotted at their first
    crop — what's on disk is still the uncropped frame, so they get the same
    widen-back-out behaviour rather than "new imports only";
  - the client-supplied filename is validated by membership in the dataset's stored
    extras, exactly like the delete route (path-traversal guard).
"""
import io
import json
import os

from PIL import Image

from app.extensions import db
from app.services import face_dataset_service as svc


def _png(w, h, color=(120, 40, 40)):
    b = io.BytesIO()
    Image.new('RGB', (w, h), color).save(b, 'PNG')
    return b.getvalue()


def _dataset_with_extra(width=2400, height=1350):
    ds = svc.create_dataset('local', 'ds', 'trg')
    folder = svc.dataset_path(ds.id)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'ref.webp'), 'wb') as fh:
        fh.write(svc.normalize_to_webp(_png(512, 512)))
    ds.ref_filename = 'ref.webp'
    db.session.commit()
    fn = svc.add_extra_ref('local', ds.id, _png(width, height))
    return ds, folder, fn


def test_add_extra_ref_keeps_a_full_frame_original(app):
    with app.app_context():
        ds, folder, fn = _dataset_with_extra()
        orig = svc.extra_ref_original_name(fn)
        assert orig and orig != fn
        assert os.path.isfile(os.path.join(folder, orig))
        # The extra itself keeps its aspect ratio (no square pad) and the original is
        # the one stored at the higher cap, so a crop has pixels to work with.
        with Image.open(os.path.join(folder, orig)) as im:
            assert im.size == (2048, 1152)
        with Image.open(os.path.join(folder, fn)) as im:
            assert im.size == (1024, 576)
        # The payload tells the UI which file its ✂ editor must open.
        payload = svc.dataset_payload('local', ds.id)
        assert payload['ref_extra_filenames'] == [fn]
        assert payload['ref_extra_crop_sources'] == [orig]


def test_crop_extra_ref_can_widen_back_out(app):
    with app.app_context():
        ds, folder, fn = _dataset_with_extra()
        path = os.path.join(folder, fn)

        # Tight crop first...
        assert svc.crop_extra_ref('local', ds.id, fn, 100, 100, 400, 400) is True
        with Image.open(path) as im:
            assert im.size == (1024, 1024)
        # ...then a WIDER box in the original's pixel space. It only works because the
        # original was never overwritten.
        assert svc.crop_extra_ref('local', ds.id, fn, 0, 0, 2048, 1152) is True
        with Image.open(path) as im:
            assert im.size == (1024, 576)
        with Image.open(os.path.join(folder, svc.extra_ref_original_name(fn))) as im:
            assert im.size == (2048, 1152)   # untouched by either crop


def test_crop_extra_ref_snapshots_legacy_extras_before_the_first_crop(app):
    """Extras imported before originals were kept: the file on disk IS the full frame,
    so the first crop snapshots it and they widen back out like any other."""
    with app.app_context():
        ds, folder, fn = _dataset_with_extra()
        orig_path = os.path.join(folder, svc.extra_ref_original_name(fn))
        os.remove(orig_path)                      # emulate a pre-feature dataset

        assert svc.crop_extra_ref('local', ds.id, fn, 300, 100, 200, 200) is True
        assert os.path.isfile(orig_path)
        with Image.open(orig_path) as im:
            assert im.size == (1024, 576)          # the frame as it stood, snapshotted
        # Widening now works, bounded by what that snapshot holds.
        assert svc.crop_extra_ref('local', ds.id, fn, 0, 0, 1024, 576) is True
        with Image.open(os.path.join(folder, fn)) as im:
            assert im.size == (1024, 576)


def test_crop_extra_ref_rejects_filenames_outside_the_dataset_extras(app, tmp_path):
    """The filename is client input: anything not in the stored extras is refused,
    and nothing derived from it is written (same guard as remove_extra_ref)."""
    with app.app_context():
        ds, folder, fn = _dataset_with_extra()
        victim = os.path.join(folder, 'ref.webp')
        before = open(victim, 'rb').read()
        for hostile in ('../ref.webp', '..\\ref.webp', 'ref.webp',
                        os.path.join(str(tmp_path), 'anything.webp'), ''):
            assert svc.crop_extra_ref('local', ds.id, hostile, 0, 0, 10, 10) is False
        assert open(victim, 'rb').read() == before
        assert not os.path.isfile(os.path.join(folder, 'ref_datasetrefxorig_.webp'))


def test_crop_extra_ref_route(client, app):
    with app.app_context():
        ds, folder, fn = _dataset_with_extra()
        dsid = ds.id

    r = client.post(f'/api/dataset/{dsid}/ref/extra/crop',
                    json={'filename': fn, 'x': 10, 'y': 10, 'w': 300, 'h': 300})
    assert r.status_code == 200 and r.get_json()['ok'] is True

    r = client.post(f'/api/dataset/{dsid}/ref/extra/crop',
                    json={'filename': fn, 'x': 'nope', 'y': 0, 'w': 10, 'h': 10})
    assert r.status_code == 400

    r = client.post(f'/api/dataset/{dsid}/ref/extra/crop',
                    json={'filename': '../ref.webp', 'x': 0, 'y': 0, 'w': 10, 'h': 10})
    assert r.status_code == 404


def test_removing_an_extra_ref_takes_its_original_along(app):
    with app.app_context():
        ds, folder, fn = _dataset_with_extra()
        orig_path = os.path.join(folder, svc.extra_ref_original_name(fn))
        assert svc.remove_extra_ref('local', ds.id, fn) is True
        assert not os.path.exists(os.path.join(folder, fn))
        assert not os.path.exists(orig_path)      # never left orphaned in the folder
        assert json.loads(ds.ref_extra_filenames) == []
