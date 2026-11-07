"""HEIC/HEIF photos import on every dataset ingress lane, and never stay HEIF.

No browser but Safari, neither trainer and none of the OpenCV workers read HEIF,
so an import decodes it once (pi-heif, registered when the app package loads) and
stores what the rest of the pipeline reads: a lossless PNG of the same picture in
the default preserve mode, the mode's WebP otherwise, a WebP head crop with
auto-crop on.

The fixture was made once with pillow-heif's x265 encoder: a 96x64 synthetic
gradient with a red top-left and a blue bottom-right block, an sRGB ICC profile,
no EXIF and no encoder banner. 1208 bytes, no personal data.
"""
import base64
import io
import os
import zipfile

import pytest
from PIL import Image

TINY_HEIC = base64.b64decode(''.join("""
AAAAHGZ0eXBoZWljAAAAAG1pZjFoZWljbWlhZgAAA5ttZXRhAAAAAAAAACFoZGxyAAAAAAAAAABw
aWN0AAAAAAAAAAAAAAAAAAAAACJpbG9jAAAAAERAAAEAAQAAAAADvwABAAAAAAAAAPkAAAAjaWlu
ZgAAAAAAAQAAABVpbmZlAgAAAAABAABodmMxAAAAAA5waXRtAAAAAAABAAADG2lwcnAAAAL8aXBj
bwAAAHhodmNDAQNwAAAAkAAAAAAAHvAA/P34+AAADwNgAAEAGEABDAH//wNwAAADAJAAAAMAAAMA
HroCQGEAAQArQgEBA3AAAAMAkAAAAwAAAwAeoDCBBZbqSSmubgIaDAgAAAMAyAAAAwAIQGIAAQAH
RAHBcrAiQAAAAlhjb2xycHJvZgAAAkxsY21zBEAAAG1udHJSR0IgWFlaIAfqAAsABwAOAC0AAmFj
c3BNU0ZUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD21gABAAAAANMtbGNtcwAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAC2Rlc2MAAAEIAAAANmNwcnQAAAFA
AAAATHd0cHQAAAGMAAAAFGNoYWQAAAGgAAAALHJYWVoAAAHMAAAAFGJYWVoAAAHgAAAAFGdYWVoA
AAH0AAAAFHJUUkMAAAIIAAAAIGdUUkMAAAIIAAAAIGJUUkMAAAIIAAAAIGNocm0AAAIoAAAAJG1s
dWMAAAAAAAAAAQAAAAxlblVTAAAAGgAAABwAcwBSAEcAQgAgAGIAdQBpAGwAdAAtAGkAbgAAbWx1
YwAAAAAAAAABAAAADGVuVVMAAAAwAAAAHABOAG8AIABjAG8AcAB5AHIAaQBnAGgAdAAsACAAdQBz
AGUAIABmAHIAZQBlAGwAeVhZWiAAAAAAAAD21gABAAAAANMtc2YzMgAAAAAAAQxCAAAF3v//8yUA
AAeTAAD9kP//+6H///2iAAAD3AAAwG5YWVogAAAAAAAAb6AAADj1AAADkFhZWiAAAAAAAAAknwAA
D4QAALbDWFlaIAAAAAAAAGKXAAC3hwAAGNlwYXJhAAAAAAADAAAAAmZmAADypwAADVkAABPQAAAK
W2Nocm0AAAAAAAMAAAAAo9cAAFR7AABMzQAAmZoAACZmAAAPXAAAABRpc3BlAAAAAAAAAGAAAABA
AAAAEHBpeGkAAAAAAwgICAAAABdpcG1hAAAAAAAAAAEAAQSBAgMEAAABAW1kYXQAAAD1KAGvCeDW
g5rPgr//Spf/zB2h8P5b3Ewn5QY38H3Byv/8nWZ9cdy8BgWToQLffBZ8AGnLvsWy3tUQXl5WzV6y
hwaG/h8/+MefWQuNXf4wbEW/VtWh5ImM+JQ+JFIRsyWiJw1pONyylv+CE0tzKtHIckUYtYV3sn2E
+4Ia3O+eE4K0hVwKNNOAPQ5rGhQ+8hJSLUVioBM8vw//N+KhDn189dPV38JZqCQ0Vm5Ya3iwKW9R
I6nP1UJQab+IID1vOJXj7iZnzTJi/lxA8ntQbuUV4FTu44DGKlNVVFjdwRMGQWN/DmZHaxJZqyBU
3aY85QgOPIW8K/w=
""".split()))


def _decoded(raw):
    """(format, size, RGB pixels, ICC) — through the app's registered decoders."""
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        return im.format, im.size, im.convert('RGB').tobytes(), im.info.get('icc_profile')


def _stored(svc, dataset_id, image_id):
    from app.models import FaceDatasetImage
    row = svc.db.session.get(FaceDatasetImage, image_id)
    with open(os.path.join(svc._dataset_dir(dataset_id), row.filename), 'rb') as fh:
        return row.filename, fh.read()


def test_preserve_mode_stores_heic_as_a_lossless_png_of_the_same_picture(app):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    heic_format, heic_size, heic_pixels, heic_icc = _decoded(TINY_HEIC)
    assert (heic_format, heic_size) == ('HEIF', (96, 64)) and heic_icc
    with app.app_context():
        assert svc.import_encode_policy()['preserve'] is True
        dataset = svc.create_dataset(LOCAL_USER, 'HEIC preserve', 'heic_preserve')
        stats = {}
        ids, failed = svc.import_images(LOCAL_USER, dataset.id, [TINY_HEIC], crop=False, stats=stats)
        assert failed == 0 and len(ids) == 1 and 'failed_reasons' not in stats
        filename, stored = _stored(svc, dataset.id, ids[0])
        # The byte-preserving lanes (Bank promotion, backup restore) still refuse it:
        # a HEIF must never sit in a dataset folder under its own bytes.
        with pytest.raises(ValueError, match='supports only static'):
            svc._preserved_import_extension(TINY_HEIC)

    assert filename.endswith('.png')
    assert _decoded(stored) == ('PNG', heic_size, heic_pixels, heic_icc)
    with Image.open(io.BytesIO(stored)) as png:
        assert 'exif' not in png.info


@pytest.mark.parametrize('lane', ['webp_mode', 'head_crop'])
def test_heic_decodes_on_the_derived_lanes(app, monkeypatch, lane):
    from app.config import LOCAL_USER, save_config
    from app.services import face_dataset_service as svc

    # No vision server in a test: the head crop falls back to its centred square,
    # which still has to decode the HEIC first.
    monkeypatch.setattr(svc, 'detect_head_bbox', lambda _raw: None)
    with app.app_context():
        if lane == 'webp_mode':
            save_config({'dataset_import': {'max_side': 1024, 'encoding': 'standard'}})
        dataset = svc.create_dataset(LOCAL_USER, f'HEIC {lane}', f'heic_{lane}')
        ids, failed = svc.import_images(LOCAL_USER, dataset.id, [TINY_HEIC],
                                        crop=lane == 'head_crop')
        assert failed == 0 and len(ids) == 1
        filename, stored = _stored(svc, dataset.id, ids[0])
    fmt, size, _pixels, _icc = _decoded(stored)
    assert filename.endswith('.webp') and fmt == 'WEBP'
    assert size == ((1024, 1024) if lane == 'head_crop' else (96, 64))


def test_zip_and_folder_merges_pick_up_heic_files(app, tmp_path):
    """Both extensions, any case: phones write IMG_0001.HEIC."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('phone/IMG_0001.heic', TINY_HEIC)
        z.writestr('phone/IMG_0001.txt', 'a caption from another tool')
    folder = tmp_path / 'phone-export'
    folder.mkdir()
    (folder / 'IMG_0002.HEIF').write_bytes(TINY_HEIC)

    with app.app_context():
        zipped = svc.create_dataset(LOCAL_USER, 'HEIC zip', 'heic_zip')
        ids, failed = svc.import_dataset_zip(LOCAL_USER, zipped.id, archive.getvalue())
        assert failed == 0 and len(ids) == 1
        assert _stored(svc, zipped.id, ids[0])[0].endswith('.png')

        walked = svc.create_dataset(LOCAL_USER, 'HEIC folder', 'heic_folder')
        ids, failed = svc.import_dataset_folder(LOCAL_USER, walked.id, str(folder))
        assert failed == 0 and len(ids) == 1
        assert _stored(svc, walked.id, ids[0])[0].endswith('.png')


def test_dropzone_and_reference_routes_accept_a_heic_photo(app, client):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    with app.app_context():
        dataset_id = svc.create_dataset(LOCAL_USER, 'HEIC routes', 'heic_routes').id
    response = client.post(f'/api/dataset/{dataset_id}/import', content_type='multipart/form-data',
                           data={'crop': '0', 'files': [(io.BytesIO(TINY_HEIC), 'IMG_0003.HEIC')]})
    assert response.status_code == 200
    assert (response.json['imported'], response.json['failed']) == (1, 0)
    assert response.json['failed_reasons'] == {}

    response = client.post(f'/api/dataset/{dataset_id}/ref', content_type='multipart/form-data',
                           data={'file': (io.BytesIO(TINY_HEIC), 'IMG_0004.HEIC')})
    assert response.status_code == 200 and response.json['ok'] is True


def test_an_edit_reference_heic_is_sanitized_before_any_engine_sees_it(app):
    """The edit dialog's extra references go to API engines: fresh WebP pixels only."""
    from app.services import face_dataset_service as svc

    with app.app_context():
        payload = svc.sanitize_external_reference(TINY_HEIC, label='extra edit reference 1')
    with Image.open(io.BytesIO(payload)) as im:
        assert (im.format, im.size) == ('WEBP', (96, 64))
        assert 'icc_profile' not in im.info and 'exif' not in im.info
