"""Persistent per-dataset batch-activity registry (dataset_activity) + its wiring
into dataset_payload and the batch services.

The UI's "in progress" indicator used to be React-local state, so a page reload
dropped it while the server kept working. These tests cover the registry lifecycle
(begin/progress/end), the try/finally crash-safety net (an exception must clear the
indicator), the TTL purge, thread-safety across datasets, and the payload exposing
`activity` while a batch runs and `None` after it ends.
"""
import io
import os
import threading
import time

import pytest
from PIL import Image


def _img_bytes(color=(200, 30, 30), size=(64, 64), fmt='WEBP'):
    buf = io.BytesIO()
    Image.new('RGB', size, color).save(buf, fmt)
    return buf.getvalue()


def _kept_image(svc, ds_id, filename, *, state=None, bbox=None):
    """A kept FaceDatasetImage backed by a real file on disk."""
    import json
    from app.models import FaceDatasetImage
    d = svc._dataset_dir(ds_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, filename), 'wb') as fh:
        fh.write(_img_bytes(size=(1024, 1024)))
    img = FaceDatasetImage(dataset_id=ds_id, source='import', status='keep',
                           filename=filename, framing='body',
                           watermark_state=state,
                           watermark_bbox=json.dumps(bbox) if bbox is not None else None)
    svc.db.session.add(img)
    svc.db.session.commit()
    return img


# --- registry lifecycle (no app needed) ------------------------------------

def test_begin_progress_end_lifecycle():
    from app.services import dataset_activity as da
    da.reset()
    t = da.begin(5, 'watermark_detect', total=64)
    a = da.get(5)
    assert a['kind'] == 'watermark_detect' and a['done'] == 0 and a['total'] == 64
    assert isinstance(a['started_at'], float)
    da.progress(t, done=12)
    assert da.get(5)['done'] == 12
    da.bump(t, 3)
    assert da.get(5)['done'] == 15
    da.progress(t, total=70)
    assert da.get(5)['total'] == 70
    da.end(t)
    assert da.get(5) is None


def test_activity_detail_can_explain_a_long_running_stage():
    from app.services import dataset_activity as da
    da.reset()
    t = da.begin(5, 'caption', total=12, detail='Preparing captioning…')
    assert da.get(5)['detail'] == 'Preparing captioning…'
    da.progress(t, detail='Loading JoyCaption model and captioning 12 images…')
    a = da.get(5)
    assert a['done'] == 0
    assert a['detail'] == 'Loading JoyCaption model and captioning 12 images…'
    da.end(t)


def test_generate_activity_exposes_its_engine():
    from app.services import dataset_activity as da
    da.reset()
    t = da.begin(5, 'generate', total=2, engine='chatgpt')
    assert da.get(5)['engine'] == 'chatgpt'
    da.end(t)

    da.sync_pending(5, 'generate', 2, engine='klein')
    assert da.get(5)['engine'] == 'klein'
    da.sync_pending(5, 'generate', 0, engine='klein')


def test_end_idempotent_and_unknown_token_is_noop():
    from app.services import dataset_activity as da
    da.reset()
    t = da.begin(1, 'classify', total=2)
    da.end(t)
    da.end(t)               # idempotent — no raise
    da.end('nope')          # unknown token — no raise
    da.progress('nope', done=1)
    da.bump(None)           # None token — no raise
    assert da.get(1) is None


def test_two_datasets_are_independent():
    from app.services import dataset_activity as da
    da.reset()
    t1 = da.begin(1, 'caption', total=10)
    t2 = da.begin(2, 'analyze_faces', total=3)
    assert da.get(1)['kind'] == 'caption' and da.get(1)['total'] == 10
    assert da.get(2)['kind'] == 'analyze_faces' and da.get(2)['total'] == 3
    da.end(t1)
    assert da.get(1) is None
    assert da.get(2)['kind'] == 'analyze_faces'   # unaffected by the other's end
    da.end(t2)


def test_get_returns_latest_started_and_end_targets_only_its_token():
    """When two batches overlap on one dataset (CPU pass beside a GPU pass), get()
    returns the most recently started one; ending it falls back to the earlier one —
    a stale end() from a crashed batch never wipes a newer indicator."""
    from app.services import dataset_activity as da
    da.reset()
    t1 = da.begin(7, 'watermark_clean', total=4)
    time.sleep(0.002)
    t2 = da.begin(7, 'caption', total=9)
    assert da.get(7)['kind'] == 'caption'          # latest started wins
    da.end(t2)
    assert da.get(7)['kind'] == 'watermark_clean'  # earlier one still there
    da.end(t1)
    assert da.get(7) is None


def test_sync_pending_tracks_count_and_clears():
    """The Klein 'generate' reconcile: pending>0 keeps an entry whose total is the
    high-water mark and done = total - pending; pending==0 clears it."""
    from app.services import dataset_activity as da
    da.reset()
    da.sync_pending(9, 'generate', 3)
    a = da.get(9)
    assert a['kind'] == 'generate' and a['total'] == 3 and a['done'] == 0
    da.sync_pending(9, 'generate', 1)          # two of the three finished
    a = da.get(9)
    assert a['total'] == 3 and a['done'] == 2
    da.sync_pending(9, 'generate', 0)          # last one done -> indicator clears
    assert da.get(9) is None


def test_sync_pending_total_is_high_water_mark():
    from app.services import dataset_activity as da
    da.reset()
    da.sync_pending(4, 'generate', 2)
    da.sync_pending(4, 'generate', 5)          # a second wave piled on before draining
    assert da.get(4)['total'] == 5
    da.sync_pending(4, 'generate', 0)
    assert da.get(4) is None


def test_sync_pending_does_not_corrupt_a_begin_entry():
    """A worker-owned begin() 'generate' entry (an API batch) and a sync_pending
    'generate' entry (Klein) can momentarily coexist on one dataset: sync must only
    ever touch its OWN (_synced) entry, and ending the worker's token must leave the
    sync entry intact (distinct tokens)."""
    from app.services import dataset_activity as da
    da.reset()
    t = da.begin(6, 'generate', total=10)
    da.progress(t, done=4)
    da.sync_pending(6, 'generate', 3)          # separate Klein-style entry
    da.end(t)                                   # end the worker's entry only
    a = da.get(6)
    assert a is not None and a['kind'] == 'generate' and a['total'] == 3
    da.sync_pending(6, 'generate', 0)
    assert da.get(6) is None


def test_ttl_purge_on_read():
    from app.services import dataset_activity as da
    da.reset()
    # A negative TTL makes any entry instantly stale, so the read must purge it —
    # proving a leaked entry can never strand a phantom indicator past the window.
    orig = da._TTL_SECONDS
    da._TTL_SECONDS = -1
    try:
        da.begin(3, 'classify', total=2)
        assert da.get(3) is None
    finally:
        da._TTL_SECONDS = orig


def test_thread_safety_concurrent_datasets():
    """Many begin/progress/bump/end cycles across distinct datasets in parallel:
    no lost updates, no crash, and every entry cleaned up at the end."""
    from app.services import dataset_activity as da
    da.reset()
    errors = []

    def worker(dsid):
        try:
            for _ in range(300):
                t = da.begin(dsid, 'classify', total=5)
                da.bump(t)
                da.progress(t, done=3)
                snap = da.get(dsid)
                assert snap is not None and snap['kind'] == 'classify'
                da.end(t)
        except Exception as e:   # noqa: BLE001 - surface any thread error to the assert
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(1, 6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for i in range(1, 6):
        assert da.get(i) is None   # all ended -> registry empty


# --- payload exposure (direct) ---------------------------------------------

def test_payload_exposes_activity_while_running_and_null_after(app):
    from app.services import face_dataset_service as svc
    from app.services import dataset_activity as da
    from app.config import LOCAL_USER
    da.reset()
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'P', 'p')
        assert svc.dataset_payload(LOCAL_USER, ds.id)['activity'] is None
        t = da.begin(ds.id, 'watermark_detect', total=8)
        da.progress(t, done=3)
        act = svc.dataset_payload(LOCAL_USER, ds.id)['activity']
        assert act == {'kind': 'watermark_detect', 'done': 3, 'total': 8,
                       'started_at': act['started_at']}
        da.end(t)
        assert svc.dataset_payload(LOCAL_USER, ds.id)['activity'] is None


# --- integration through the real batch services ---------------------------

def test_detect_watermarks_advertises_activity_then_clears(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import dataset_activity as da
    import app.services.vision_ollama as vo
    from app.config import LOCAL_USER
    da.reset()
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'W', 'w')
        _kept_image(svc, ds.id, 'a.webp')
        _kept_image(svc, ds.id, 'b.webp')
        seen = []

        def _mock(*a, **k):
            # Captured MID-BATCH: the payload must show the live indicator.
            seen.append(svc.dataset_payload(LOCAL_USER, ds.id)['activity'])
            return '{"present":true,"x1":0,"y1":0,"x2":100,"y2":50}'

        monkeypatch.setattr(vo, 'describe_image_ollama', _mock)
        monkeypatch.setattr(vo, 'unload_vision_model', lambda *a, **k: True)
        svc.detect_watermarks(LOCAL_USER, ds.id)
        assert seen and seen[0]['kind'] == 'watermark_detect' and seen[0]['total'] == 2
        assert seen[0]['done'] >= 1
        # After the batch, the indicator is gone (finally ran end()).
        assert svc.dataset_payload(LOCAL_USER, ds.id)['activity'] is None
        assert da.get(ds.id) is None


def test_detect_watermarks_clears_activity_on_exception(app, monkeypatch):
    """The try/finally net: a batch that raises mid-pass must NOT leave a phantom
    indicator behind."""
    from app.services import face_dataset_service as svc
    from app.services import dataset_activity as da
    import app.services.vision_ollama as vo
    from app.config import LOCAL_USER
    da.reset()
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'W', 'w')
        _kept_image(svc, ds.id, 'a.webp')

        def _boom(*a, **k):
            raise RuntimeError('vision crashed')

        monkeypatch.setattr(vo, 'describe_image_ollama', _boom)
        monkeypatch.setattr(vo, 'unload_vision_model', lambda *a, **k: True)
        with pytest.raises(RuntimeError):
            svc.detect_watermarks(LOCAL_USER, ds.id)
        assert da.get(ds.id) is None
        assert svc.dataset_payload(LOCAL_USER, ds.id)['activity'] is None


def test_analyze_faces_clears_activity_on_scoring_exception(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import dataset_activity as da
    from app.config import LOCAL_USER
    da.reset()
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'F', 'f')
        # a reference photo (analyze_faces requires it)
        d = svc._dataset_dir(ds.id)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'ref.webp'), 'wb') as fh:
            fh.write(_img_bytes())
        ds.ref_filename = 'ref.webp'
        svc.db.session.commit()
        _kept_image(svc, ds.id, 'a.webp')

        import app.services.face_similarity as fs
        monkeypatch.setattr(fs, 'score_dataset_faces',
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError('scorer boom')))
        with pytest.raises(RuntimeError):
            svc.analyze_faces(LOCAL_USER, ds.id)
        assert da.get(ds.id) is None


def test_caption_kind_is_recaption_when_forced(app, monkeypatch):
    """force=True (re-caption) advertises kind 'recaption'; force=False → 'caption'.
    Backend pinned to 'ollama' to bypass JoyCaption."""
    from app.services import face_dataset_service as svc
    from app.services import dataset_activity as da
    import app.services.vision_ollama as vo
    from app.config import LOCAL_USER
    da.reset()

    orig_get = svc.cfg.get
    monkeypatch.setattr(svc.cfg, 'get',
                        lambda k, *a, **kw: 'ollama' if k == 'captioning.backend' else orig_get(k, *a, **kw))

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'C', 'c')
        img = _kept_image(svc, ds.id, 'a.webp')
        img.caption = 'existing caption'          # so force is meaningful
        svc.db.session.commit()
        kinds = []

        def _mock(*a, **k):
            snap = da.get(ds.id)
            kinds.append(snap['kind'] if snap else None)
            return 'a fresh caption'

        monkeypatch.setattr(vo, 'describe_image_ollama', _mock)
        monkeypatch.setattr(vo, 'unload_vision_model', lambda *a, **k: True)

        svc.caption_images(LOCAL_USER, ds.id, force=True)
        assert kinds and kinds[0] == 'recaption'
        assert da.get(ds.id) is None

        kinds.clear()
        # force=False re-captions only the uncaptioned; clear the caption first.
        img2 = svc.db.session.get(type(img), img.id)
        img2.caption = ''
        svc.db.session.commit()
        svc.caption_images(LOCAL_USER, ds.id, force=False)
        assert kinds and kinds[0] == 'caption'
        assert da.get(ds.id) is None
