"""HEIC/HEIF photos and AVIF images import on every dataset ingress lane, and
never stay HEIF or AVIF.

No browser but Safari, neither trainer and none of the OpenCV workers read HEIF,
and the trainers do not read AVIF either, so an import decodes them once (pi-heif,
registered when the app package loads; AVIF by Pillow) and stores what the rest of
the pipeline reads: a lossless PNG of the same picture in the default preserve
mode, the mode's WebP otherwise, a WebP head crop with auto-crop on.

The HEIC fixture was made once with pillow-heif's x265 encoder: a 96x64 synthetic
gradient with a red top-left and a blue bottom-right block, an sRGB ICC profile,
no EXIF and no encoder banner, 1208 bytes. The two AVIF fixtures were made once
with Pillow's own encoder: another 96x64 gradient with a disc, still (692 bytes)
and as a two-frame animation (1755 bytes). No personal data in any of them.
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

STILL_AVIF = base64.b64decode(''.join("""
AAAAIGZ0eXBhdmlmAAAAAGF2aWZtaWYxbWlhZk1BMUIAAADrbWV0YQAAAAAAAAAhaGRscgAAAAAA
AAAAcGljdAAAAAAAAAAAAAAAAAAAAAAOcGl0bQAAAAAAAQAAAB5pbG9jAAAAAEQAAAEAAQAAAAEA
AAETAAABoQAAAChpaW5mAAAAAAABAAAAGmluZmUCAAAAAAEAAGF2MDFDb2xvcgAAAABqaXBycAAA
AEtpcGNvAAAAFGlzcGUAAAAAAAAAYAAAAEAAAAAQcGl4aQAAAAADCAgIAAAADGF2MUOBAAwAAAAA
E2NvbHJuY2x4AAEADQAGgAAAABdpcG1hAAAAAAAAAAEAAQQBAoMEAAABqW1kYXQSAAoJGBlv/tEB
DQaEMpEDRPDw8MQwkkkkyqgAAQDZlNhAjW4croN6lypJcG8Tn8hISBA7UXTbapInNFr8Ho+tar/3
arZ2GzfjVbDy63FqZgmRZjRfqvXWZKNR2mH2qHxYmeohuEleWin3Z7zO1YRKau/l1brTyYVUTmrq
WPxxNaB/j6ngYwl6gh+BZIL0Fr+UY5O7CQEXHjXan4/4/uja3iraQxGEaP2vSj644WGuCkuKGBCe
ZwPbkAsUSkRVC5za3Zo86T3FSYQCtV9NnqlbvW4AmaOvz8TZl5t/0KsHZgcIJD84u+7PPTjPeHZx
wrxZaeg7auxaC+iCoEZr22drpJFUsLY+SUKLfiPn5Ifvxwg0hnvVgGoF8cz41VxKqfmpIE23ku4D
UHCG9qRP3NC6vphO1e/lOMlYFmQgRL+B3BYgI/BTaNMyyjmLSERN60PgCavNltGNqLZ6mwRipoiu
W7g3fXfPoi3GaFkglk9DoFldV5slPNPf1ScrVHQjqaVT4RfaIXTtPZjwD1qCzn76aWLctmPbv/kz
d8JwXBfaRYA=
""".split()))

ANIMATED_AVIF = base64.b64decode(''.join("""
AAAALGZ0eXBhdmlzAAAAAGF2aWZhdmlzbXNmMWlzbzhtaWYxbWlhZk1BMUIAAADrbWV0YQAAAAAA
AAAhaGRscgAAAAAAAAAAcGljdAAAAAAAAAAAAAAAAAAAAAAOcGl0bQAAAAAAAQAAAB5pbG9jAAAA
AEQAAAEAAQAAAAEAAAPlAAABtQAAAChpaW5mAAAAAAABAAAAGmluZmUCAAAAAAEAAGF2MDFDb2xv
cgAAAABqaXBycAAAAEtpcGNvAAAAFGlzcGUAAAAAAAAAYAAAAEAAAAAQcGl4aQAAAAADCAgIAAAA
DGF2MUOBAAwAAAAAE2NvbHJuY2x4AAEADQAGgAAAABdpcG1hAAAAAAAAAAEAAQQBAoMEAAACxm1v
b3YAAAB4bXZoZAEAAAAAAAAA5xUIIwAAAADnFQgjAAAD6P//////////AAEAAAEAAAAAAAAAAAAA
AAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAEAAAJGdHJhawAAAGh0a2hkAQAAAQAAAADnFQgjAAAAAOcVCCMAAAABAAAAAP//////
////AAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAABg
AAAAQAAAAAAALGVkdHMAAAAkZWxzdAEAAAEAAAABAAAAAAAAAZAAAAAAAAAAAAABAAAAAAGqbWRp
YQAAACxtZGhkAQAAAAAAAADnFQgjAAAAAOcVCCMAAAPoAAAAAAAAAZBVxAAAAAAAIWhkbHIAAAAA
AAAAAHBpY3QAAAAAAAAAAAAAAAAAAAABVW1pbmYAAAAUdm1oZAAAAAEAAAAAAAAAAAAAACRkaW5m
AAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAARVzdGJsAAAAlXN0c2QAAAAAAAAAAQAAAIVh
djAxAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAGAAQABIAAAASAAAAAAAAAABCkFPTSBDb2RpbmcA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAGP//AAAADGF2MUOBAAwAAAAAE2NvbHJuY2x4AAEADQAGgAAA
ABBjY3N0AAAAAHwAAAAAAAAYc3R0cwAAAAAAAAABAAAAAgAAAMgAAAAcc3RzYwAAAAAAAAABAAAA
AQAAAAIAAAABAAAAHHN0c3oAAAAAAAAAAAAAAAIAAAG1AAABQQAAABRzdGNvAAAAAAAAAAEAAAPl
AAAAFHN0c3MAAAAAAAAAAQAAAAEAAAL+bWRhdBIACg0AAAADLf/Nr5EBDQaEMqEDEACYQEEDiRJc
FNlROyJpq7c98REQMJ0u3gnfdm4Qj9jGe//IEFyh7HMXHrKejlwSFJCR8NWCZ+hteB3cPJ9BqOnn
njpn9BaBq2JUp6BMWbxiPXFmKwLyKmKdz67E75vv6rr/D6n19XRDGu1NZTKxD2yvKG8+ZEnT7TEe
mr/0Pt0RYp6pRzmr8JhlQmUbbjhW+zbKC/6I5KgshB5XQS/KI7cAqZPgz6aYQi4xi7lYZx/QMJy+
KDa9h6jYcPa5213fGaQ1afzsXOW4QZnKWgjcouNSloNAcI8WahRG98R6aYWjYlQ9jReEJG89qAA5
RRema8kRkC76/7l3N04fV+figcfSye+aMIuf7b6sSvJIK/MANsBZmZzYTEPd9bpn/GU23rPbLLkV
WkJ+6PQvrtvOvwizI8RWGhdorM6pOi++fKgMxpQ7Aiw0W9K3FwlZqvCTzc5HzslzTbl4bNXZxm1S
PbHLALfvhcKXG4QTSXcUIWJbBxbz557rVlGcKHaqrBnHo/O+QEz/jvlmKDCUzaBFbaOvYSwHG9n8
H7mVKd9L2RkIEgAyvAIwA8CAAABGmQAww5USXWUA2IGt6kzyF5zvQRXc1PVkBfRVJvadMrgQsI06
BwhM6uvkqP+0ikz48mtd1ixWNpP0Ur/BFxS0VBqBIyQUzcMB8KkNqG937yi/ZVBHRXWZjdZ/eCf0
xUC0ekBoNoCnklk6VoPJ5Dts5+PCO4gk2WyQszR09xpgnkRGFPGZRUU3XftVaz7Y2x9wJnLsQxzb
ZeOciBsXJlEBg8twDYk8MmlP6Jiy+HApBGqXcjH54+VrJbyzCdQNPRaIEDSOD+wUUukc7yalm3qC
25cQkdPfczvABnVlgpXKvW94sVRNeW3ByEqfkHN9RzZZpFJ9z3XVjsRwmKU30a0f2gR5wz+VEG3L
QgMV5LZnLuI9FhpYj7mGGH+vLOAgs+XM3LI1bVz8T5adGUKrbZxMyDXK687I
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


@pytest.mark.parametrize('raw, source_format', [(TINY_HEIC, 'HEIF'), (STILL_AVIF, 'AVIF')],
                         ids=['heic', 'avif'])
def test_preserve_mode_stores_a_lossless_png_of_the_same_picture(app, raw, source_format):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    fmt, size, pixels, icc = _decoded(raw)
    assert (fmt, size) == (source_format, (96, 64))
    with app.app_context():
        assert svc.import_encode_policy()['preserve'] is True
        dataset = svc.create_dataset(LOCAL_USER, f'{fmt} preserve', f'{fmt.lower()}_preserve')
        stats = {}
        ids, failed = svc.import_images(LOCAL_USER, dataset.id, [raw], crop=False, stats=stats)
        assert failed == 0 and len(ids) == 1 and 'failed_reasons' not in stats
        filename, stored = _stored(svc, dataset.id, ids[0])
        # The byte-preserving lanes (Bank promotion, backup restore) still refuse it:
        # neither format may sit in a dataset folder under its own bytes.
        with pytest.raises(ValueError, match='supports only static'):
            svc._preserved_import_extension(raw)

    assert filename.endswith('.png')
    # Same pixels, same colour profile (the HEIC carries an sRGB ICC, the AVIF none).
    assert _decoded(stored) == ('PNG', size, pixels, icc)
    with Image.open(io.BytesIO(stored)) as png:
        assert 'exif' not in png.info


def test_an_animated_avif_is_refused_like_any_animation(app):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    with app.app_context():
        with pytest.raises(ValueError, match='animated images are not supported'):
            svc.import_store_image(ANIMATED_AVIF)
        dataset = svc.create_dataset(LOCAL_USER, 'AVIF animated', 'avif_animated')
        stats = {}
        ids, failed = svc.import_images(LOCAL_USER, dataset.id, [ANIMATED_AVIF], crop=False, stats=stats)
    assert (ids, failed, stats['failed_reasons']) == ([], 1, {'unreadable': 1})


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


def test_zip_and_folder_merges_pick_up_heic_and_avif_files(app, tmp_path):
    """Every extension, any case: phones write IMG_0001.HEIC."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('phone/IMG_0001.heic', TINY_HEIC)
        z.writestr('phone/IMG_0001.txt', 'a caption from another tool')
        z.writestr('web/still.avif', STILL_AVIF)
    folder = tmp_path / 'phone-export'
    folder.mkdir()
    (folder / 'IMG_0002.HEIF').write_bytes(TINY_HEIC)
    (folder / 'still.AVIF').write_bytes(STILL_AVIF)

    with app.app_context():
        zipped = svc.create_dataset(LOCAL_USER, 'HEIC zip', 'heic_zip')
        ids, failed = svc.import_dataset_zip(LOCAL_USER, zipped.id, archive.getvalue())
        assert failed == 0 and len(ids) == 2
        assert all(_stored(svc, zipped.id, i)[0].endswith('.png') for i in ids)

        walked = svc.create_dataset(LOCAL_USER, 'HEIC folder', 'heic_folder')
        ids, failed = svc.import_dataset_folder(LOCAL_USER, walked.id, str(folder))
        assert failed == 0 and len(ids) == 2
        assert all(_stored(svc, walked.id, i)[0].endswith('.png') for i in ids)


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
