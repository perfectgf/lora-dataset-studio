"""Caption the images in ANOTHER tool, then bring the captions back.

Reported by Qeeyana on Reddit: "I am unable to export zip unless I caption,
which is annoying if I want to use a different captioning tool than the ones
available." The export gate lives in the UI; the trip itself lives here, and
it had a hole nobody had walked into: re-importing the SAME images with their
freshly written .txt files hit the perceptual-duplicate filter, so every image
was skipped and every caption went with it — silently, reported as
"0 imported · N duplicates skipped".
"""
import io
from app.extensions import db
import zipfile

from PIL import Image


def _photo(seed):
    im = Image.new('RGB', (96, 96), (255, 255, 255))
    for i in range(8):
        x = (seed * 13 + i * 7) % 80
        im.paste(((seed * 37) % 255, (i * 61) % 255, (seed * 7 + i * 29) % 255),
                 (x, i * 12, x + 12, i * 12 + 12))
    buf = io.BytesIO()
    im.save(buf, 'PNG')
    return buf.getvalue()


def _zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        for name, data in entries:
            z.writestr(name, data)
    return buf.getvalue()


def test_foreign_tool_sidecars_are_read(app):
    """The convention every external captioner writes: `<image>.txt` next to
    `<image>.<ext>`, flat, arbitrary names, CRLF and a trailing newline."""
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Foreign', 'foreign')
        buf = io.BytesIO()
        Image.open(io.BytesIO(_photo(4))).save(buf, 'JPEG')
        stats = {}
        ids, failed = svc.import_dataset_zip(LOCAL_USER, ds.id, _zip([
            ('IMG_2201.jpg', buf.getvalue()),
            ('IMG_2201.txt', b'a woman in a red coat, city street\r\n'),
        ]), stats=stats)
        assert len(ids) == 1 and failed == 0 and stats.get('captions') == 1
        row = FaceDatasetImage.query.filter_by(dataset_id=ds.id).one()
        assert row.caption == 'a woman in a red coat, city street'


def test_captions_come_back_onto_the_images_already_here(app):
    """THE round trip: export uncaptioned → caption elsewhere → re-import.
    The images are already in the dataset, so they are duplicates by design —
    their captions must land on the rows that hold them, not be dropped."""
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Trip', 'trip')
        originals = [_photo(1), _photo(2)]
        ids, _ = svc.import_images(LOCAL_USER, ds.id, originals, crop=False)
        assert len(ids) == 2
        stats = {}
        back, failed = svc.import_dataset_zip(LOCAL_USER, ds.id, _zip([
            ('trip_001.png', originals[0]), ('trip_001.txt', b'first, outdoors'),
            ('trip_002.png', originals[1]), ('trip_002.txt', b'second, indoors'),
        ]), stats=stats)
        assert back == [] and failed == 0
        assert stats.get('duplicates') == 2
        assert stats.get('captions_applied') == 2
        rows = FaceDatasetImage.query.filter_by(dataset_id=ds.id).order_by(
            FaceDatasetImage.id.asc()).all()
        assert [r.caption for r in rows] == ['first, outdoors', 'second, indoors']


def test_a_caption_already_written_here_is_never_overwritten(app):
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Keep', 'keep')
        raw = _photo(9)
        ids, _ = svc.import_images(LOCAL_USER, ds.id, [raw], crop=False)
        row = db.session.get(FaceDatasetImage, ids[0])
        row.caption = 'mine, written here'
        svc.db.session.commit()
        stats = {}
        svc.import_dataset_zip(LOCAL_USER, ds.id,
                               _zip([('x.png', raw), ('x.txt', b'theirs')]),
                               stats=stats)
        assert db.session.get(FaceDatasetImage, ids[0]).caption == 'mine, written here'
        assert stats.get('captions_applied', 0) == 0
        assert stats.get('captions_kept') == 1


def test_route_reports_what_came_back(client, app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'RouteTrip', 'routetrip')
        did = ds.id
        raw = _photo(6)
        svc.import_images(LOCAL_USER, did, [raw], crop=False)
    resp = client.post(f'/api/dataset/{did}/import-zip',
                       data={'file': (io.BytesIO(_zip([
                           ('a.png', raw), ('a.txt', b'brought back')])), 'caps.zip')},
                       content_type='multipart/form-data')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['imported'] == 0 and body['duplicates'] == 1
    assert body['captions_applied'] == 1


def test_style_export_writes_the_images_even_uncaptioned(app):
    """The backend never demanded captions — pinning it, because the UI refusal
    is now bypassable and must land on a real, usable archive."""
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'StyleOut', 'styleout')
        ds.kind = 'style'
        svc.db.session.commit()
        assert svc.is_style(ds)
        svc.import_images(LOCAL_USER, ds.id, [_photo(3)], crop=False)
        for row in FaceDatasetImage.query.filter_by(dataset_id=ds.id).all():
            row.status = 'keep'
        svc.db.session.commit()
        out = io.BytesIO()
        svc.write_export_zip(LOCAL_USER, ds.id, out)
        names = zipfile.ZipFile(io.BytesIO(out.getvalue())).namelist()
        assert any(n.endswith('.png') for n in names)
        assert any(n.endswith('.txt') for n in names)
