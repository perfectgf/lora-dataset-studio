"""Every lane that starts Klein work says WHICH model it starts it on.

The dataset-scoped Klein model choice shipped for ✨ Upscale & improve and for
Klein generation. The inventory of that delivery found three more lanes still
calling into Klein with `klein_model=None` — i.e. running on a model nobody
named, and silently swapping in another one when a named model went missing:

  1. the LOCAL reference edit (`_start_local_reference_edit`);
  2. the rescue of scraped images under 768 px (`_save_small_scrape_pair`);
  3. the watermark inpaint (`watermark_klein._run_klein_job`), which called
     `resolve_klein_unet()` with no argument at all.

1 and 2 hold a dataset, so they inherit its pick — one resolution, reused, never
a second one. 3 runs on datasets AND on banks; a bank has no dataset to inherit
from, so it keeps the auto resolution and the SCREEN names the model instead
(there is deliberately no second place to choose one — see the comment at the
bank call site).

Pinned here, for each lane:
  * the dataset's stored choice ARRIVES at the enqueue / the inpaint;
  * a dataset that never chose sends EXACTLY what it sent before (`None`), which
    is the whole anti-regression contract of the migration;
  * a chosen model that has left the disk is refused BY NAME, never replaced;
  * the bank lane sends no model at all, on purpose.
"""
import io
import json
import os
import struct

import pytest
from PIL import Image

_VALID_ST = struct.pack('<Q', 2) + b'{}'
KLEIN_FILE = 'flux-2-klein-9b-fp8.safetensors'
OTHER_FILE = 'flux-2-klein-32b-heavy.safetensors'


def _png(size=(96, 64)):
    buf = io.BytesIO()
    Image.new('RGB', size, (25, 50, 75)).save(buf, 'PNG')
    return buf.getvalue()


def _comfy_with(tmp_path, *names):
    """A minimal ComfyUI holding exactly `names` under models/unet/klein."""
    from app import config as cfg
    base = tmp_path / 'ComfyUI'
    (base / 'models' / 'unet' / 'klein').mkdir(parents=True, exist_ok=True)
    (base / 'input').mkdir(parents=True, exist_ok=True)
    (base / 'output').mkdir(parents=True, exist_ok=True)
    (base / 'main.py').write_text('# fake', encoding='utf-8')
    for name in names:
        (base / 'models' / 'unet' / 'klein' / name).write_bytes(_VALID_ST)
    cfg.save_config({'comfyui': {'base_dir': str(base)}})
    return base


# --------------------------------------------------------------------------
def _start_reference_edit(svc, ds, prompt):
    """Drive ONE local (Klein) reference edit through the lane's real entry point.

    The multi-engine compare batch (f55ae7a) renamed `_start_local_reference_edit`
    to `_enqueue_local_reference_edit` and gave it the batch's per-engine token, so
    this helper registers the batch the enqueue attaches itself to — the same two
    steps `start_reference_edit` performs. Calling the private enqueue directly
    keeps the assertions on the KLEIN arguments instead of on the batch plumbing,
    which has its own file (test_ref_edit_local_engines.py)."""
    from app.config import LOCAL_USER
    from app.services import reference_edit_jobs as rej
    dsdir = svc._dataset_dir(ds.id)
    started = rej.start_batch(ds.id, dsdir, ('klein',), prompt)
    return svc._enqueue_local_reference_edit(
        LOCAL_USER, ds.id, ds, 'klein', prompt, started['tokens']['klein'],
        os.path.join(dsdir, ds.ref_filename), [])


# Lanes 1 & 2 — a dataset in hand, so the dataset's pick
# --------------------------------------------------------------------------
@pytest.fixture()
def lanes(app, monkeypatch):
    """The reference-edit and small-rescue lanes, returning the kwargs the
    enqueue saw. Nothing reaches ComfyUI: the enqueue itself is the seam."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    queued = []
    counter = {'n': 0}

    def _enqueue(**kwargs):
        queued.append(kwargs)
        counter['n'] += 1
        return f'job-{counter["n"]}'

    monkeypatch.setattr(keh, 'enqueue_klein_edit', _enqueue)

    class Lanes:
        calls = queued

        @staticmethod
        def _dataset(stored=None):
            ds = svc.create_dataset(LOCAL_USER, 'Klein lanes', 'kleinlanes')
            os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
            if stored is not None:
                svc.set_dataset_klein_model(LOCAL_USER, ds.id, stored)
            return ds

        @staticmethod
        def reference_edit(stored=None):
            ds = Lanes._dataset(stored)
            ds.ref_filename = 'ref.png'
            with open(os.path.join(svc._dataset_dir(ds.id), 'ref.png'), 'wb') as fh:
                fh.write(_png())
            svc.db.session.commit()
            _start_reference_edit(svc, ds, 'make it sharper')
            return ds

        @staticmethod
        def small_rescue(stored=None):
            ds = Lanes._dataset(stored)
            svc._save_small_scrape_pair(LOCAL_USER, ds.id, _png((400, 500)),
                                        'upscale this')
            return ds

    with app.app_context():
        yield Lanes


@pytest.mark.parametrize('lane', ['reference_edit', 'small_rescue'])
def test_the_dataset_choice_reaches_the_queue(lanes, lane):
    """A choice honoured by some lanes and ignored by others is worse than no
    choice: the dataset ends up made of images from two different models."""
    getattr(lanes, lane)(stored=OTHER_FILE)
    assert lanes.calls, f'{lane} enqueued nothing'
    assert lanes.calls[-1].get('klein_model') == OTHER_FILE, lane


@pytest.mark.parametrize('lane', ['reference_edit', 'small_rescue'])
def test_a_dataset_that_never_chose_sends_exactly_what_it_sent_before(lanes, lane):
    """`klein_model=None` is the value both lanes have always sent, and what
    makes resolve_klein_unet pick the canonical download. Anything else here
    means an install that touched nothing changed model."""
    getattr(lanes, lane)()
    assert lanes.calls[-1].get('klein_model') is None, lane


def test_reference_edit_refuses_a_vanished_model_by_name(app, tmp_path):
    """The reference image is what the whole dataset is anchored on. Repainting
    it on a neighbour model because the chosen file moved is exactly the swap
    nobody can detect afterwards."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    with app.app_context():
        _comfy_with(tmp_path, KLEIN_FILE)              # OTHER_FILE is NOT there
        ds = svc.create_dataset(LOCAL_USER, 'Gone ref', 'goneref')
        os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
        with open(os.path.join(svc._dataset_dir(ds.id), 'ref.png'), 'wb') as fh:
            fh.write(_png())
        ds.ref_filename = 'ref.png'
        svc.db.session.commit()
        svc.set_dataset_klein_model(LOCAL_USER, ds.id, OTHER_FILE)

        with pytest.raises(ValueError) as exc:
            _start_reference_edit(svc, ds, 'sharper')
        assert OTHER_FILE in str(exc.value)


def test_small_rescue_refuses_a_vanished_model_by_name(app, tmp_path):
    """The rescue never raises (it is one item of a scrape import); it records
    why. That reason has to NAME the file, or the row just says "failed"."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    with app.app_context():
        _comfy_with(tmp_path, KLEIN_FILE)
        ds = svc.create_dataset(LOCAL_USER, 'Gone rescue', 'gonerescue')
        os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
        svc.set_dataset_klein_model(LOCAL_USER, ds.id, OTHER_FILE)

        assert svc._save_small_scrape_pair(LOCAL_USER, ds.id, _png((400, 500)),
                                           'upscale this') is False
        row = (FaceDatasetImage.query
               .filter_by(dataset_id=ds.id, derivation_kind=svc.KLEIN_SMALL_IMAGE)
               .first())
        assert row.status == 'failed'
        assert OTHER_FILE in (row.fail_reason or '')


# --------------------------------------------------------------------------
# Lane 3 — watermark inpaint: dataset side inherits, bank side does not
# --------------------------------------------------------------------------
def _flagged_dataset_image(svc, image_cls, user_id, tmp_path, stored=None):
    ds = svc.create_dataset(user_id, 'WM klein', 'wmklein')
    os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
    with open(os.path.join(svc._dataset_dir(ds.id), 'a.png'), 'wb') as fh:
        fh.write(_png((1000, 1000)))
    img = image_cls(dataset_id=ds.id, filename='a.png', source='import',
                    status='keep', watermark_state='detected',
                    watermark_bbox=json.dumps([0.30, 0.30, 0.36, 0.36]))
    svc.db.session.add(img)
    svc.db.session.commit()
    if stored is not None:
        svc.set_dataset_klein_model(user_id, ds.id, stored)
    return ds, img


@pytest.fixture()
def wm_dataset(app, monkeypatch, tmp_path):
    """clean_watermarks(method='klein') with the GPU round-trip replaced by a
    recorder — what matters is which model the pass asks for."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import watermark_klein, watermark_lama

    seen = []
    monkeypatch.setattr(watermark_klein, 'is_available', lambda: True)
    monkeypatch.setattr(watermark_lama, 'is_available', lambda: False)

    def _fake(user_id, path, boxes, **kwargs):
        seen.append(kwargs)
        return True, None

    monkeypatch.setattr(watermark_klein, 'inpaint_watermark_klein', _fake)

    class WM:
        calls = seen

        @staticmethod
        def clean(stored=None):
            ds, img = _flagged_dataset_image(svc, FaceDatasetImage, LOCAL_USER,
                                             tmp_path, stored)
            out, _err = svc.clean_watermarks(LOCAL_USER, ds.id, method='klein')
            return ds, img, out

    with app.app_context():
        # Both models present: this fixture is about which one is ASKED for, not
        # about a missing file (that is the refusal test below).
        _comfy_with(tmp_path, KLEIN_FILE, OTHER_FILE)
        yield WM


def test_watermark_clean_inherits_the_dataset_model(wm_dataset):
    """A watermark clean OVERWRITES the image in place. Running it on a model
    the dataset did not choose is the one lane whose swap cannot be spotted
    afterwards by comparing with a source — there is no source left."""
    _ds, _img, out = wm_dataset.clean(stored=OTHER_FILE)
    assert out['inpainted_klein'] == 1
    assert wm_dataset.calls[-1].get('klein_model') == OTHER_FILE


def test_watermark_clean_on_a_dataset_that_never_chose_is_unchanged(wm_dataset):
    _ds, _img, out = wm_dataset.clean()
    assert out['inpainted_klein'] == 1
    assert wm_dataset.calls[-1].get('klein_model') is None


def test_watermark_clean_refuses_a_vanished_model_before_touching_a_file(
        app, tmp_path, monkeypatch):
    """Refused as a WHOLE pass, by name: every image would fail identically, and
    a half-cleaned dataset is worse than an untouched one."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh
    from app.services import watermark_klein, watermark_lama
    with app.app_context():
        _comfy_with(tmp_path, KLEIN_FILE)
        monkeypatch.setattr(watermark_klein, 'is_available', lambda: True)
        monkeypatch.setattr(watermark_lama, 'is_available', lambda: False)
        ran = []
        monkeypatch.setattr(watermark_klein, 'inpaint_watermark_klein',
                            lambda *a, **k: ran.append(a) or (True, None))
        ds, img = _flagged_dataset_image(svc, FaceDatasetImage, LOCAL_USER,
                                         tmp_path, stored=OTHER_FILE)

        with pytest.raises(keh.KleinModelGone) as exc:
            svc.clean_watermarks(LOCAL_USER, ds.id, method='klein')
        assert OTHER_FILE in str(exc.value)
        assert not ran                                   # nothing was repainted
        svc.db.session.rollback()
        assert svc.db.session.get(FaceDatasetImage, img.id).watermark_state == 'detected'


def test_region_inpaint_inherits_the_dataset_model(app, tmp_path, monkeypatch):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import watermark_klein
    seen = []
    monkeypatch.setattr(watermark_klein, 'is_available', lambda: True)

    def _fake(user_id, path, boxes, **kwargs):
        seen.append(kwargs)
        return True, None

    monkeypatch.setattr(watermark_klein, 'inpaint_mask_klein', _fake)
    with app.app_context():
        _comfy_with(tmp_path, KLEIN_FILE, OTHER_FILE)
        ds, img = _flagged_dataset_image(svc, FaceDatasetImage, LOCAL_USER,
                                         tmp_path, stored=OTHER_FILE)
        out = svc.inpaint_region(
            LOCAL_USER, ds.id, img.id, [[0.2, 0.2, 0.4, 0.4]], 'remove necklace')
        assert out['ok'] is True
        assert seen[-1].get('klein_model') == OTHER_FILE


def test_region_inpaint_on_a_dataset_that_never_chose_sends_none(
        app, tmp_path, monkeypatch):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import watermark_klein
    seen = []
    monkeypatch.setattr(watermark_klein, 'is_available', lambda: True)
    monkeypatch.setattr(watermark_klein, 'inpaint_mask_klein',
                        lambda *a, **k: seen.append(k) or (True, None))
    with app.app_context():
        _comfy_with(tmp_path, KLEIN_FILE, OTHER_FILE)
        ds, img = _flagged_dataset_image(svc, FaceDatasetImage, LOCAL_USER, tmp_path)
        svc.inpaint_region(
            LOCAL_USER, ds.id, img.id, [[0.2, 0.2, 0.4, 0.4]], 'remove necklace')
        assert seen[-1].get('klein_model') is None


def test_region_inpaint_refuses_a_vanished_model_before_touching_a_file(
        app, tmp_path, monkeypatch):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh
    from app.services import watermark_klein
    with app.app_context():
        _comfy_with(tmp_path, KLEIN_FILE)
        monkeypatch.setattr(watermark_klein, 'is_available', lambda: True)
        ran = []
        monkeypatch.setattr(watermark_klein, 'inpaint_mask_klein',
                            lambda *a, **k: ran.append(a) or (True, None))
        ds, img = _flagged_dataset_image(svc, FaceDatasetImage, LOCAL_USER,
                                         tmp_path, stored=OTHER_FILE)
        path = svc._img_path(img)
        before = open(path, 'rb').read()
        with pytest.raises(keh.KleinModelGone) as exc:
            svc.inpaint_region(
                LOCAL_USER, ds.id, img.id, [[0.2, 0.2, 0.4, 0.4]], 'remove necklace')
        assert OTHER_FILE in str(exc.value)
        assert not ran
        assert open(path, 'rb').read() == before


def test_the_bank_lane_names_no_model_on_purpose(client, tmp_path, app, monkeypatch):
    """A bank has no dataset to inherit a pick from, and the Klein model choice
    lives on the dataset. So this lane keeps the auto resolution it always had —
    the alternative (a global setting, or a second picker on the bank) would be a
    second authority for the same UNETLoader. The bank SCREEN names the resolved
    model instead; see /api/klein-model."""
    from app.services import watermark_klein, watermark_lama
    seen = []
    monkeypatch.setattr(watermark_klein, 'is_available', lambda: True)
    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
    monkeypatch.setattr(watermark_lama, 'resolve_device', lambda: 'cpu')

    def _fake(user_id, path, boxes, **kwargs):
        seen.append(kwargs)
        return True, None

    monkeypatch.setattr(watermark_klein, 'inpaint_watermark_klein', _fake)

    from test_bank_watermark_clean import _flag, _mkbank, _photo
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    _flag(app, bank_id, [0.40, 0.40, 0.60, 0.60])       # on-subject → Klein's lane

    r = client.post(f'/api/bank/{bank_id}/watermark/inpaint', json={'method': 'klein'})
    assert r.status_code == 202, r.get_json()
    assert seen, 'the bank never reached the Klein inpaint'
    assert seen[-1].get('klein_model') is None


# --------------------------------------------------------------------------
# The shared resolution — one helper, three lanes
# --------------------------------------------------------------------------
def test_a_named_model_and_no_name_resolve_through_the_same_helper(app, tmp_path):
    from app.services import klein_edit_helper as keh
    with app.app_context():
        _comfy_with(tmp_path, KLEIN_FILE)
        # No name: byte-for-byte what every lane resolved before.
        assert keh.unet_for_job() == keh.resolve_klein_unet()
        assert keh.unet_for_job(None) == os.path.join('klein', KLEIN_FILE)
        # A name that is there: exactly it, with its loader prefix.
        assert keh.unet_for_job(KLEIN_FILE) == os.path.join('klein', KLEIN_FILE)
        # A name that is gone: refused, never the neighbour.
        with pytest.raises(keh.KleinModelGone) as exc:
            keh.unet_for_job(OTHER_FILE)
        assert exc.value.name == OTHER_FILE


def test_the_watermark_workflow_loads_the_named_model(app, tmp_path, monkeypatch):
    """The seam that used to call resolve_klein_unet() with no argument at all."""
    from app.services import watermark_klein as wk
    with app.app_context():
        _comfy_with(tmp_path, KLEIN_FILE, OTHER_FILE)
        loaded = {}

        def _add_job(**kwargs):
            loaded['unet'] = kwargs['workflow_data']['114']['inputs']['unet_name']

        monkeypatch.setattr(wk.queue_manager, 'add_job', _add_job)
        monkeypatch.setattr(wk, '_wait_for_job', lambda *a, **k: ('failed', None, 'stop'))
        monkeypatch.setattr(wk.keh, 'klein_missing_assets', lambda: [])
        crop = Image.new('RGB', (256, 256), (10, 20, 30))

        wk._run_klein_job('local', crop, seed=1, klein_model=OTHER_FILE)
        assert loaded['unet'] == os.path.join('klein', OTHER_FILE)

        wk._run_klein_job('local', crop, seed=1)
        assert loaded['unet'] == wk.keh.resolve_klein_unet()

        with pytest.raises(wk.keh.KleinModelGone):
            wk._run_klein_job('local', crop, seed=1, klein_model='not-here.safetensors')


def test_the_global_endpoint_names_a_model_without_offering_a_choice(client, tmp_path):
    """What a bank screen reads: the model that will run, and no way to set one —
    choosing stays a dataset gesture."""
    _comfy_with(tmp_path, KLEIN_FILE)
    r = client.get('/api/klein-model')
    assert r.status_code == 200
    body = r.get_json()
    assert body['stored'] is None
    assert body['effective'] == KLEIN_FILE
    assert client.post('/api/klein-model', json={}).status_code == 405
