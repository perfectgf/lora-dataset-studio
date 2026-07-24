"""Edit the reference photo: stateless edit (candidate held by the caller) + an
atomic, fail-safe Keep that never strands the dataset without a reference.

The engine is ALWAYS stubbed here — no network, no dollars. These tests pin the
two invariants the design rests on:
  - /ref/edit writes NOTHING to the dataset (the candidate is leak-proof);
  - commit writes+verifies the new files BEFORE deleting the old ones, so a failed
    Keep leaves the previous reference intact.
"""
import contextlib
import io
import json
import os

import pytest
from PIL import Image

from app.services import face_dataset_service as svc


def _webp(color, size=(300, 300)):
    b = io.BytesIO()
    Image.new('RGB', size, color).save(b, 'WEBP')
    return b.getvalue()


def _png():
    b = io.BytesIO()
    Image.new('RGB', (256, 256), (120, 40, 40)).save(b, 'PNG')
    return b.getvalue()


def _seed_ref(ds, *, original, cropped):
    d = svc._dataset_dir(ds.id)
    orig_fn, ref_fn = 'local_datasetreforig_e.webp', 'local_datasetref_e.webp'
    if original is not None:
        with open(os.path.join(d, orig_fn), 'wb') as f:
            f.write(original)
        ds.ref_original_filename = orig_fn
    with open(os.path.join(d, ref_fn), 'wb') as f:
        f.write(cropped)
    ds.ref_filename = ref_fn
    svc.db.session.commit()
    return d, orig_fn, ref_fn


# --- edit_reference (dispatch, ref list, guards) -----------------------------

def test_edit_reference_dispatches_and_passes_full_ref_list(app, monkeypatch):
    with app.app_context():
        ds = svc.create_dataset('local', 'Ann', 'zchar_ann')
        d = svc._dataset_dir(ds.id)
        ref_fn, extra_fn = 'local_datasetref_e.webp', 'local_datasetrefx_e.webp'
        with open(os.path.join(d, ref_fn), 'wb') as f:
            f.write(_webp((10, 20, 30)))
        with open(os.path.join(d, extra_fn), 'wb') as f:
            f.write(_webp((40, 50, 60)))
        ds.ref_filename = ref_fn
        ds.ref_extra_filenames = json.dumps([extra_fn])
        svc.db.session.commit()

        seen, picked = {}, {}

        def stub(refs, prompt, aspect_ratio='1:1'):
            seen.update(refs=refs, prompt=prompt, aspect=aspect_ratio)
            return b'EDITED'

        def fake_fn(engine):
            picked['engine'] = engine
            return stub
        monkeypatch.setattr(svc, '_api_generate_fn', fake_fn)

        out = svc.edit_reference(ds, '  add glasses ', 'chatgpt', extra_edit_ref_bytes=[b'MODALREF'])
        assert out == b'EDITED'
        assert picked['engine'] == 'chatgpt'
        assert seen['prompt'] == 'add glasses'            # trimmed
        assert seen['aspect'] == '1:1'
        # primary + dataset extra + modal edit-ref, primary first, modal ref last
        assert len(seen['refs']) == 3
        assert seen['refs'][-1] == b'MODALREF'


def test_edit_reference_rejects_non_api_engine(app):
    with app.app_context():
        ds = svc.create_dataset('local', 'Bo', 'zchar_bo')
        with pytest.raises(ValueError):
            svc.edit_reference(ds, 'x', 'klein')


def test_edit_reference_empty_prompt_raises(app):
    with app.app_context():
        ds = svc.create_dataset('local', 'Cy', 'zchar_cy')
        ds.ref_filename = 'local_datasetref_e.webp'
        svc.db.session.commit()
        with pytest.raises(ValueError):
            svc.edit_reference(ds, '   ', 'chatgpt')


def test_edit_reference_passes_through_none(app, monkeypatch):
    with app.app_context():
        ds = svc.create_dataset('local', 'Di', 'zchar_di')
        _seed_ref(ds, original=None, cropped=_webp((5, 5, 5)))
        monkeypatch.setattr(svc, '_api_generate_fn',
                            lambda e: (lambda refs, prompt, aspect_ratio='1:1': None))
        assert svc.edit_reference(ds, 'x', 'nanobanana') is None


# --- commit_edited_reference (atomic swap) -----------------------------------

def test_commit_edited_reference_swaps_and_deletes_old(app):
    with app.app_context():
        ds = svc.create_dataset('local', 'Ev', 'zchar_ev')
        d, old_orig, old_ref = _seed_ref(
            ds, original=_webp((255, 0, 0), (400, 400)), cropped=_webp((0, 255, 0), (64, 64)))
        new = svc.commit_edited_reference('local', ds.id, _webp((0, 0, 255), (300, 300)))
        assert new != old_ref
        assert ds.ref_filename == new
        assert ds.ref_original_filename and ds.ref_original_filename != old_orig
        # both new files are the blue edit (ref AND original point at the edited image)
        for fn in (ds.ref_filename, ds.ref_original_filename):
            _r, _g, b = Image.open(os.path.join(d, fn)).convert('RGB').getpixel((10, 10))
            assert b > 200
        # superseded files removed
        assert not os.path.exists(os.path.join(d, old_ref))
        assert not os.path.exists(os.path.join(d, old_orig))


def test_commit_failure_preserves_old_reference(app):
    """The invariant the lead asked to prove: an unusable candidate must NOT strand
    the dataset — the previous reference stays intact and the DB is untouched."""
    with app.app_context():
        ds = svc.create_dataset('local', 'Fa', 'zchar_fa')
        d, old_orig, old_ref = _seed_ref(
            ds, original=_webp((255, 0, 0)), cropped=_webp((0, 255, 0), (64, 64)))
        with pytest.raises(Exception):
            svc.commit_edited_reference('local', ds.id, b'not an image at all')
        assert ds.ref_filename == old_ref
        assert ds.ref_original_filename == old_orig
        assert os.path.exists(os.path.join(d, old_ref))
        assert os.path.exists(os.path.join(d, old_orig))


def test_commit_missing_reference_raises(app):
    with app.app_context():
        ds = svc.create_dataset('local', 'Gu', 'zchar_gu')
        with pytest.raises(ValueError):
            svc.commit_edited_reference('local', ds.id, _webp((1, 1, 1)))


# --- routes ------------------------------------------------------------------

def _create_with_ref(client, monkeypatch, name, trig):
    import app.routes.datasets as dr
    monkeypatch.setattr(dr, 'gpu_exclusive_vision_window', lambda: contextlib.nullcontext())
    monkeypatch.setattr(dr.svc, 'face_crop_to_square_webp', lambda raw, **k: (_webp((1, 2, 3)), True))
    did = client.post('/api/dataset/create',
                      json={'name': name, 'trigger_word': trig}).get_json()['id']
    client.post(f'/api/dataset/{did}/ref',
                data={'file': (io.BytesIO(_png()), 'r.png')},
                content_type='multipart/form-data')
    return did


def test_ref_edit_route_returns_bytes_without_mutating(client, monkeypatch):
    import app.routes.datasets as dr
    did = _create_with_ref(client, monkeypatch, 'Zoe', 'zchar_zoe')
    ref_before = client.get(f'/api/dataset/{did}').get_json()['ref_filename']
    monkeypatch.setattr(dr.svc, '_api_generate_fn',
                        lambda e: (lambda refs, prompt, aspect_ratio='1:1': b'EDITEDBYTES'))
    resp = client.post(f'/api/dataset/{did}/ref/edit',
                       data={'prompt': 'add glasses', 'engine': 'chatgpt'},
                       content_type='multipart/form-data')
    assert resp.status_code == 200
    assert resp.data == b'EDITEDBYTES'
    assert resp.mimetype == 'image/webp'
    # the stateless edit must not have touched the dataset reference
    assert client.get(f'/api/dataset/{did}').get_json()['ref_filename'] == ref_before


def test_ref_edit_commit_route_swaps_ref(client, monkeypatch):
    did = _create_with_ref(client, monkeypatch, 'Ivy', 'zchar_ivy')
    before = client.get(f'/api/dataset/{did}').get_json()['ref_filename']
    resp = client.post(f'/api/dataset/{did}/ref/edit/commit',
                       data={'file': (io.BytesIO(_webp((9, 9, 200), (200, 200))), 'e.webp')},
                       content_type='multipart/form-data')
    assert resp.status_code == 200
    j = resp.get_json()
    assert j['ok'] is True and j['ref_filename'] != before
    assert client.get(f'/api/dataset/{did}').get_json()['ref_filename'] == j['ref_filename']


def test_ref_edit_route_bad_engine_400(client, monkeypatch):
    did = _create_with_ref(client, monkeypatch, 'Jay', 'zchar_jay')
    resp = client.post(f'/api/dataset/{did}/ref/edit',
                       data={'prompt': 'x', 'engine': 'klein'},
                       content_type='multipart/form-data')
    assert resp.status_code == 400


def test_ref_edit_route_missing_dataset_404(client):
    resp = client.post('/api/dataset/999999/ref/edit',
                       data={'prompt': 'x', 'engine': 'chatgpt'},
                       content_type='multipart/form-data')
    assert resp.status_code == 404
