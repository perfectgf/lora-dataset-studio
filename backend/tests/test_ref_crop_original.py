"""Reference crop keeps the ORIGINAL so a re-crop can widen back out.

The bug (shared with the source app): /ref cropped once, discarded the upload, and
crop_reference re-cropped the already-cropped square in place — so a manual re-crop
could only tighten, never recover cropped-away pixels. Now /ref stores the full-frame
original, crop_reference reads THAT (writing the derived square without touching the
original), and recrop_reference_auto re-runs the auto head-crop on it.
"""
import io
import os
import contextlib

from PIL import Image

from app.services import face_dataset_service as svc


def _webp(color, size=(600, 400)):
    b = io.BytesIO()
    Image.new('RGB', size, color).save(b, 'WEBP')
    return b.getvalue()


def _png():
    b = io.BytesIO()
    Image.new('RGB', (256, 256), (120, 40, 40)).save(b, 'PNG')
    return b.getvalue()


def _seed_ref(ds, *, original, cropped):
    """Write an original (red, big) + a tight current ref (green, small) to disk and
    point the dataset at them. Distinct colors let a test prove WHICH file was read."""
    d = svc._dataset_dir(ds.id)
    orig_fn, ref_fn = 'local_datasetreforig_t.webp', 'local_datasetref_t.webp'
    if original is not None:
        with open(os.path.join(d, orig_fn), 'wb') as f:
            f.write(original)
        ds.ref_original_filename = orig_fn
    with open(os.path.join(d, ref_fn), 'wb') as f:
        f.write(cropped)
    ds.ref_filename = ref_fn
    svc.db.session.commit()
    return d, orig_fn, ref_fn


def test_crop_reference_reads_original_not_the_cropped_ref(app):
    with app.app_context():
        ds = svc.create_dataset('local', 'Emma', 'zchar_emma')
        d, orig_fn, ref_fn = _seed_ref(
            ds, original=_webp((255, 0, 0), (600, 400)), cropped=_webp((0, 255, 0), (64, 64)))
        # A 400px-wide box is IMPOSSIBLE from the 64px green ref (would clamp to 64) —
        # so a red 1024² result proves the crop was taken from the original.
        assert svc.crop_reference('local', ds.id, 0, 0, 400, 400) is True
        im = Image.open(os.path.join(d, ref_fn)).convert('RGB')
        assert im.size == (1024, 1024)
        r, g, _b = im.getpixel((512, 512))
        assert r > 200 and g < 60                       # red = from the original
        assert Image.open(os.path.join(d, orig_fn)).size == (600, 400)  # original untouched


def test_crop_reference_legacy_without_original_crops_ref_in_place(app):
    with app.app_context():
        ds = svc.create_dataset('local', 'Leo', 'zchar_leo')
        d, _o, ref_fn = _seed_ref(ds, original=None, cropped=_webp((0, 0, 255), (128, 128)))
        assert ds.ref_original_filename is None
        assert svc.crop_reference('local', ds.id, 0, 0, 128, 128) is True
        im = Image.open(os.path.join(d, ref_fn)).convert('RGB')
        assert im.size == (1024, 1024)
        _r, _g, b = im.getpixel((512, 512))
        assert b > 200                                  # blue = the legacy ref itself


def test_recrop_reference_auto_reruns_head_crop_on_original(app, monkeypatch):
    with app.app_context():
        ds = svc.create_dataset('local', 'Mia', 'zchar_mia')
        d, _o, ref_fn = _seed_ref(
            ds, original=_webp((255, 0, 0), (400, 400)), cropped=_webp((0, 255, 0), (64, 64)))
        monkeypatch.setattr(svc, 'detect_head_bbox', lambda *a, **k: (0.25, 0.25, 0.75, 0.75))
        ok, detected = svc.recrop_reference_auto('local', ds.id)
        assert ok is True and detected is True
        im = Image.open(os.path.join(d, ref_fn)).convert('RGB')
        assert im.size == (1024, 1024)
        r, g, _b = im.getpixel((512, 512))
        assert r > 200 and g < 60                       # regenerated from the red original


def test_recrop_reference_auto_no_reference_returns_false(app):
    with app.app_context():
        ds = svc.create_dataset('local', 'NoRef', 'zchar_noref')
        ok, detected = svc.recrop_reference_auto('local', ds.id)
        assert ok is False and detected is False


def test_ref_route_stores_original_and_exposes_it_in_payload(client, monkeypatch):
    import app.routes.datasets as dr
    monkeypatch.setattr(dr, 'gpu_exclusive_vision_window', lambda: contextlib.nullcontext())
    monkeypatch.setattr(dr.svc, 'face_crop_to_square_webp', lambda raw, **k: (b'RIFFcrop', True))
    did = client.post('/api/dataset/create',
                      json={'name': 'Zoe', 'trigger_word': 'zchar_zoe'}).get_json()['id']
    resp = client.post(f'/api/dataset/{did}/ref',
                       data={'file': (io.BytesIO(_png()), 'r.png')},
                       content_type='multipart/form-data')
    assert resp.status_code == 200 and resp.get_json()['ok'] is True
    payload = client.get(f'/api/dataset/{did}').get_json()
    assert payload['ref_original_filename']            # a real original was stored
    assert payload['ref_original_filename'] != payload['ref_filename']
    # the stored original is a decodable webp (the full-frame, not the crop stub)
    orig_path = os.path.join(svc._dataset_dir(did), payload['ref_original_filename'])
    assert Image.open(orig_path).size == (256, 256)
