"""The dataset watermark surface: bulk reject, the engine choice, and Stop.

Three user-visible promises are pinned here, each measured on the PROPERTY and
not on a proxy for it:

* **the number next to ✕ Reject all flagged is the number it rejects** — counted
  on a set that deliberately mixes flagged, clean, already-rejected, failed and
  small-image-rescue rows, because two of those are exactly what turns an
  announced count into a lie;
* **one setting, two surfaces** — `watermark_detect.backend` is read the same way
  by the bank and by the dataset, all three values, including the one case that
  must never be silent: a pinned detector with no extra installed;
* **Stop keeps what it paid for** — a scan stopped mid-way keeps every verdict
  already written, and the next run finishes the rest.
"""
import io
import json
import os

import pytest
from PIL import Image


def _img_bytes(size=(64, 64)):
    buf = io.BytesIO()
    Image.new('RGB', size, (180, 60, 60)).save(buf, 'WEBP')
    return buf.getvalue()


def _image(svc, ds_id, filename, *, status='keep', state=None, bbox=None,
           derivation_kind=None):
    """A FaceDatasetImage backed by a REAL file (the detector resolves paths)."""
    from app.models import FaceDatasetImage
    directory = svc._dataset_dir(ds_id)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), 'wb') as fh:
        fh.write(_img_bytes())
    img = FaceDatasetImage(dataset_id=ds_id, source='import', status=status,
                           filename=filename, framing='body',
                           watermark_state=state,
                           derivation_kind=derivation_kind,
                           watermark_bbox=json.dumps(bbox) if bbox else None)
    svc.db.session.add(img)
    svc.db.session.commit()
    return img


def _pin_backend(monkeypatch, value):
    """Set watermark_detect.backend the way a user's config.json would, without
    touching the rest of the configuration (every other key must keep answering,
    or the thing under test stops being the setting)."""
    from app import config as cfg
    real = cfg.get

    def fake(dotted, default=None):
        if dotted == 'watermark_detect.backend':
            return value
        return real(dotted, default)
    monkeypatch.setattr(cfg, 'get', fake)


def _extra(monkeypatch, ok, detail='the detector weights are not downloaded yet '
                                   '(Setup ▸ Quality tools ▸ Watermark detector)'):
    import app.capabilities as caps
    monkeypatch.setattr(caps, 'probe_watermark_detect',
                        lambda *a, **k: {'ok': ok, 'detail': detail if not ok else 'ready'})
    monkeypatch.setattr(caps, 'watermark_detect_gpu_available', lambda *a, **k: False)


def _fake_scan(verdicts):
    """Stand in for watermark_detector.scan: yields (path, state, score, regions,
    fingerprint, error) per input path, in order, exactly like the real
    generator. The arity here is load-bearing: this stub once yielded five
    fields while the real generator had grown a sixth, so the suite stayed
    green on a dataset pass that crashed on the first real image —
    test_watermark_scan_tuple_contract.py now pins the two together."""
    def scan(paths, **kwargs):
        should_cancel = kwargs.get('should_cancel')
        for i, path in enumerate(paths):
            state, score, regions = verdicts[i % len(verdicts)]
            yield (path, state, score, regions, None, None)
            if should_cancel and should_cancel():
                return
    return scan


# --- A. ✕ Reject all flagged: the announced number is the treated number ------

def test_bulk_reject_moves_exactly_the_rows_the_button_counts(client, app):
    """The set is mixed on purpose. The button's own count (built client-side by
    rejectableFlagged) excludes the failed row and the rescue row; the server
    then rejects exactly that many, and says so."""
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'Bulk', 'trigger_word': 'bulk'}).get_json()['id']
    with app.app_context():
        flagged_a = _image(svc, ds_id, 'a.webp', state='detected', bbox=[0, 0, .1, .1])
        flagged_b = _image(svc, ds_id, 'b.webp', state='detected', bbox=[0, 0, .1, .1])
        clean = _image(svc, ds_id, 'c.webp', state='none')
        already = _image(svc, ds_id, 'd.webp', status='reject', state=None)
        failed = _image(svc, ds_id, 'e.webp', status='failed', state='detected')
        ids = [flagged_a.id, flagged_b.id, failed.id]
        clean_id, already_id, failed_id = clean.id, already.id, failed.id

    resp = client.post(f'/api/dataset/{ds_id}/images/batch',
                       json={'ids': ids, 'action': 'reject'})
    assert resp.status_code == 200, resp.get_json()
    # The server SKIPS the failed row (its loop `continue`s before counting), so
    # a UI that had announced 3 would have been lying by one.
    assert resp.get_json()['affected'] == 2

    with app.app_context():
        rows = {r.id: r for r in FaceDatasetImage.query.filter_by(dataset_id=ds_id).all()}
        assert [rows[i].status for i in ids[:2]] == ['reject', 'reject']
        # …and the flags are GONE — the one destructive part of a reject, which
        # is why the confirmation has to name it.
        assert all(rows[i].watermark_state is None for i in ids[:2])
        assert rows[failed_id].status == 'failed'      # untouched
        assert rows[clean_id].status == 'keep'         # never in the batch
        assert rows[already_id].status == 'reject'     # idempotent


def test_one_rescue_row_would_sink_the_whole_batch(client, app):
    """WHY the button filters rescue rows rather than trusting the server: a
    single one refuses the ENTIRE request before a byte is written. Announcing a
    count that includes it would promise N and deliver zero."""
    from app.services import face_dataset_service as svc
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'Rescue', 'trigger_word': 'r'}).get_json()['id']
    with app.app_context():
        flagged = _image(svc, ds_id, 'a.webp', state='detected', bbox=[0, 0, .1, .1])
        rescue = _image(svc, ds_id, 'b.webp', state='detected', bbox=[0, 0, .1, .1],
                        derivation_kind='small_image_source')
        ids, flagged_id = [flagged.id, rescue.id], flagged.id
    resp = client.post(f'/api/dataset/{ds_id}/images/batch',
                       json={'ids': ids, 'action': 'reject'})
    assert resp.status_code == 400
    with app.app_context():
        from app.models import FaceDatasetImage
        assert svc.db.session.get(FaceDatasetImage, flagged_id).status == 'keep'


# --- B. one setting, three values, two surfaces -------------------------------

@pytest.mark.parametrize('pinned,extra_ok,expected', [
    ('auto', False, 'vision'),
    ('auto', True, 'detector'),
    ('detector', True, 'detector'),
    ('detector', False, 'vision'),      # falls back rather than refusing
    ('vision', True, 'vision'),         # pinned: the extra is ignored
])
def test_resolve_backend_covers_every_value(monkeypatch, pinned, extra_ok, expected):
    from app.services import watermark_detector as wd
    _pin_backend(monkeypatch, pinned)
    _extra(monkeypatch, extra_ok)
    resolution = wd.resolve_backend()
    assert resolution['backend'] == expected
    assert resolution['requested'] == pinned
    # Only the pinned-but-missing case is a FALLBACK: 'auto' choosing the vision
    # model is the answer to the question that was asked, not a downgrade.
    assert resolution['fell_back'] is (pinned == 'detector' and not extra_ok)
    if resolution['fell_back']:
        assert 'Setup' in resolution['detail'] and 'vision model' in resolution['detail']
    else:
        assert resolution['detail'] == ''


def test_unknown_backend_value_reads_as_auto(monkeypatch):
    from app.services import watermark_detector as wd
    _pin_backend(monkeypatch, 'nonsense')
    _extra(monkeypatch, True)
    assert wd.resolve_backend()['requested'] == 'auto'


@pytest.mark.parametrize('pinned,extra_ok,detector_expected', [
    ('auto', False, False),
    ('auto', True, True),
    ('detector', True, True),
    ('detector', False, False),
    ('vision', True, False),
])
def test_bank_takes_the_route_the_setting_names(client, app, tmp_path, monkeypatch,
                                                pinned, extra_ok, detector_expected):
    """The BANK surface. Which job gets built is the whole verdict, so both job
    factories are replaced by recorders."""
    import app.capabilities as caps
    import app.services.image_bank_service as svc
    src = tmp_path / 'src'
    os.makedirs(src, exist_ok=True)
    Image.new('RGB', (64, 64), (90, 90, 90)).save(str(src / 'a.jpg'), 'JPEG')
    bank_id = client.post('/api/bank/create',
                          json={'name': 'B', 'folder': str(src)}).get_json()['id']
    _pin_backend(monkeypatch, pinned)
    _extra(monkeypatch, extra_ok)
    monkeypatch.setattr(caps, 'probe_ollama_model', lambda *a, **k: {'ok': True})
    monkeypatch.setattr(svc, '_gpu_busy_reason', lambda *a, **k: None)
    taken = {}
    monkeypatch.setattr(svc, '_watermark_detector_job',
                        lambda *a, **k: (taken.setdefault('route', 'detector'),
                                         (lambda job: None))[1])
    real_job = svc._watermark_job

    # `statuses`/`ids` are named rather than swallowed by a `**kwargs`: this test
    # is about WHICH ROUTE the setting picks, and a scoped run must pick the same
    # one as an unscoped run. Absorbing them would let a future change route a
    # scoped scan somewhere else without a single test going red.
    def spy(bank_id_, rescan, use_detector=False, statuses=None, ids=None, note=''):
        taken['note'] = note
        taken['scope'] = (statuses, ids)
        if not use_detector:
            taken['route'] = 'vision'
            return lambda job: None
        return real_job(bank_id_, rescan, use_detector=True,
                        statuses=statuses, ids=ids, note=note)
    monkeypatch.setattr(svc, '_watermark_job', spy)
    with app.app_context():
        svc.start_watermark(app, 'local', bank_id, rescan=True)
    assert taken['route'] == ('detector' if detector_expected else 'vision')
    # The fallback is never silent: the job carries the sentence that names both
    # what ran and how to install what was asked for.
    if pinned == 'detector' and not extra_ok:
        assert 'Setup' in taken['note']
    else:
        assert taken['note'] == ''


@pytest.mark.parametrize('pinned,extra_ok,detector_expected', [
    ('auto', False, False),
    ('auto', True, True),
    ('detector', True, True),
    ('detector', False, False),
    ('vision', True, False),
])
def test_dataset_takes_the_same_route_as_the_bank(client, app, monkeypatch,
                                                  pinned, extra_ok, detector_expected):
    """The DATASET surface, same table of cases — this is the drift the setting
    was introduced to end (the dataset used to ignore the extra entirely)."""
    import app.services.vision_ollama as vo
    from app.services import face_dataset_service as svc
    from app.services import watermark_detector as wd
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'D', 'trigger_word': 'd'}).get_json()['id']
    with app.app_context():
        _image(svc, ds_id, 'a.webp')
    _pin_backend(monkeypatch, pinned)
    _extra(monkeypatch, extra_ok)
    used = []
    monkeypatch.setattr(vo, 'describe_image_ollama',
                        lambda *a, **k: used.append('vision') or '{"present":false}')
    monkeypatch.setattr(vo, 'unload_vision_model', lambda *a, **k: True)

    def scan(paths, **kwargs):
        used.append('detector')
        yield (paths[0], 'detected', 0.97, [[0.0, 0.0, 0.2, 0.1]], None, None)
    monkeypatch.setattr(wd, 'scan', scan)

    body = client.post(f'/api/dataset/{ds_id}/watermarks/detect').get_json()
    assert used == ['detector' if detector_expected else 'vision']
    assert body['backend'] == ('detector' if detector_expected else 'vision')
    assert body['backend_requested'] == pinned
    if pinned == 'detector' and not extra_ok:
        assert 'Setup' in body['backend_note']
    else:
        assert body['backend_note'] == ''


def test_default_auto_without_the_extra_is_the_pass_that_always_shipped(client, app,
                                                                       monkeypatch):
    """BYTE-IDENTICAL default: setting untouched, no extra, no rescan — the same
    engine, the same three counters, the same persisted state as before."""
    import app.services.vision_ollama as vo
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app import config as cfg
    # No _pin_backend here on purpose: this reads the SHIPPED default.
    assert cfg.DEFAULTS['watermark_detect']['backend'] == 'auto'
    _extra(monkeypatch, False)
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'Auto', 'trigger_word': 'a'}).get_json()['id']
    with app.app_context():
        a = _image(svc, ds_id, 'a.webp')
        b = _image(svc, ds_id, 'b.webp')
        a_id, b_id = a.id, b.id
    raws = iter(['{"present":true,"x1":0,"y1":0,"x2":100,"y2":50}', '{"present":false}'])
    monkeypatch.setattr(vo, 'describe_image_ollama', lambda *a, **k: next(raws))
    monkeypatch.setattr(vo, 'unload_vision_model', lambda *a, **k: True)
    body = client.post(f'/api/dataset/{ds_id}/watermarks/detect').get_json()
    assert (body['detected'], body['none'], body['checked']) == (1, 1, 2)
    assert body['backend'] == 'vision' and body['stopped'] is False
    with app.app_context():
        assert svc.db.session.get(FaceDatasetImage, a_id).watermark_state == 'detected'
        assert svc.db.session.get(FaceDatasetImage, b_id).watermark_state == 'none'


def test_detector_route_records_who_ruled_and_flags_without_a_position(client, app,
                                                                       monkeypatch):
    """The two engines do NOT persist the same thing, and the difference is not
    hidden: the cascade stamps a source and a score, and it can legitimately flag
    an image WITHOUT a box — reported apart so the screen can say so."""
    from app.services import face_dataset_service as svc
    from app.services import watermark_detector as wd
    from app.models import FaceDatasetImage
    _pin_backend(monkeypatch, 'detector')
    _extra(monkeypatch, True)
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'Src', 'trigger_word': 's'}).get_json()['id']
    with app.app_context():
        located = _image(svc, ds_id, 'a.webp')
        blind = _image(svc, ds_id, 'b.webp')
        located_id, blind_id = located.id, blind.id
    monkeypatch.setattr(wd, 'scan', _fake_scan([
        ('detected', 0.981234, [[0.0, 0.0, 0.2, 0.1]]),
        ('detected', 0.9702, []),          # flagged, position unknown
    ]))
    body = client.post(f'/api/dataset/{ds_id}/watermarks/detect').get_json()
    assert (body['detected'], body['checked']) == (2, 2)
    assert body['located'] == 1 and body['unlocated'] == 1
    with app.app_context():
        a = svc.db.session.get(FaceDatasetImage, located_id)
        b = svc.db.session.get(FaceDatasetImage, blind_id)
        assert a.watermark_source == 'detector' and a.watermark_score == 0.9812
        assert json.loads(a.watermark_bbox) == [0.0, 0.0, 0.2, 0.1]
        assert b.watermark_state == 'detected' and b.watermark_bbox is None


def test_clean_leaves_a_position_less_flag_for_review_instead_of_failing_it(app):
    """A flag with no box is not a failure — it is a flag waiting for a zone.
    Stamping 'failed' would destroy a correct verdict over a missing coordinate."""
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Blind', 'blind')
        img = _image(svc, ds.id, 'a.webp', state='detected')
        img_id = img.id
        counts, _err = svc.clean_watermarks(LOCAL_USER, ds.id)
        assert counts['needs_review'] == 1 and counts['failed'] == 0
        assert svc.db.session.get(FaceDatasetImage, img_id).watermark_state == 'detected'


# --- C. Stop keeps what it already found --------------------------------------

def test_stop_keeps_every_verdict_and_the_rerun_finishes_the_rest(client, app,
                                                                  monkeypatch):
    """Stop after the FIRST image: image 1 keeps its verdict, images 2 and 3 stay
    unscanned (never falsely marked clean), and a second run completes them."""
    import app.services.vision_ollama as vo
    from app.services import dataset_activity
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    _extra(monkeypatch, False)
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'Stop', 'trigger_word': 's'}).get_json()['id']
    with app.app_context():
        ids = [_image(svc, ds_id, f'{i}.webp').id for i in range(3)]

    calls = {'n': 0}

    def answer(*a, **k):
        calls['n'] += 1
        # Ask for the stop from INSIDE the first call, the way a user clicking ⏹
        # while the pass runs does (the route is served on another thread).
        if calls['n'] == 1:
            assert dataset_activity.request_cancel(
                ds_id, dataset_activity.WATERMARK_KINDS) is True
        return '{"present":true,"x1":0,"y1":0,"x2":100,"y2":50}'
    monkeypatch.setattr(vo, 'describe_image_ollama', answer)
    monkeypatch.setattr(vo, 'unload_vision_model', lambda *a, **k: True)

    body = client.post(f'/api/dataset/{ds_id}/watermarks/detect').get_json()
    assert body['stopped'] is True
    assert body['detected'] == 1 and body['checked'] == 1
    assert calls['n'] == 1, 'the pass kept asking after the stop'
    with app.app_context():
        states = [svc.db.session.get(FaceDatasetImage, i).watermark_state for i in ids]
    assert states == ['detected', None, None]

    # The flag was consumed, so a re-run is not poisoned by it and finishes.
    monkeypatch.setattr(vo, 'describe_image_ollama', lambda *a, **k: '{"present":false}')
    body = client.post(f'/api/dataset/{ds_id}/watermarks/detect').get_json()
    assert body['stopped'] is False and body['checked'] == 3
    with app.app_context():
        states = [svc.db.session.get(FaceDatasetImage, i).watermark_state for i in ids]
    assert states == ['none', 'none', 'none']


def test_stop_route_answers_409_when_nothing_is_running(client, app):
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'S', 'trigger_word': 's'}).get_json()['id']
    resp = client.post(f'/api/dataset/{ds_id}/watermarks/detect/cancel')
    assert resp.status_code == 409
    assert client.post('/api/dataset/999999/watermarks/detect/cancel').status_code == 404


def test_watermark_stop_does_not_arm_or_disarm_the_captioning_stop(app):
    """Two Stop buttons, two scopes. Widening CANNCELLABLE_KINDS would have made
    one screen's Stop reach into another screen's pass."""
    from app.services import dataset_activity as da
    da.reset()
    token = da.begin(7, 'watermark_detect', total=3)
    try:
        assert da.request_cancel(7) is False          # the caption scope: nothing live
        assert da.request_cancel(7, da.WATERMARK_KINDS) is True
        assert da.cancel_requested(7) is False        # caption workers unaffected
        assert da.cancel_requested(7, da.WATERMARK_KINDS) is True
        da.clear_cancel(7)                            # a caption route unwinding
        assert da.cancel_requested(7, da.WATERMARK_KINDS) is True
    finally:
        da.end(token)
        da.reset()


def test_rescan_including_dismissed_is_reachable_from_the_route(client, app, monkeypatch):
    """Changing engine is pointless if the verdicts of the old one can never be
    re-judged: `include_dismissed` is the only door to them, and it is open."""
    import app.services.vision_ollama as vo
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    _extra(monkeypatch, False)
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'Dis', 'trigger_word': 'd'}).get_json()['id']
    with app.app_context():
        img_id = _image(svc, ds_id, 'a.webp', state='dismissed').id
    monkeypatch.setattr(vo, 'describe_image_ollama',
                        lambda *a, **k: '{"present":true,"x1":0,"y1":0,"x2":100,"y2":50}')
    monkeypatch.setattr(vo, 'unload_vision_model', lambda *a, **k: True)
    assert client.post(f'/api/dataset/{ds_id}/watermarks/detect').get_json()['checked'] == 0
    body = client.post(f'/api/dataset/{ds_id}/watermarks/detect',
                       json={'include_dismissed': True}).get_json()
    assert body['checked'] == 1 and body['detected'] == 1
    with app.app_context():
        assert svc.db.session.get(FaceDatasetImage, img_id).watermark_state == 'detected'
