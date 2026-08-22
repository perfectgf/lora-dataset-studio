"""The face-mask preview must SAY what it is doing, and be rejoinable.

Reported live while testing the shipped feature: "Looking for face…" was the only
thing on screen for the whole InsightFace pass, and leaving the page reset the
button to "Preview the mask" — so a pass still running was abandoned and a second
full one offered in its place.

Three things are asserted here, and all three used to be impossible:
  1. progress is EXPOSED — named phases then a per-image count;
  2. a failure ends as a readable message, never as an endless wait;
  3. zero faces found is a RESULT, not a failure.
Plus the rejoin contract: one live pass per dataset, and the last result survives
the page.

No real detection anywhere: InsightFace is not installed on the dev machine, so a
tiny stand-in script speaks the same stdout/stderr protocol as
infer/face_mask_infer.py. That protocol is the contract under test.
"""
import os
import sys

from PIL import Image

from app.extensions import db
from app.models import FaceDatasetImage
from app.services import face_dataset_service as svc
from app.services import face_mask, face_mask_preview as fmp
from app.config import LOCAL_USER, save_config

PREVIEW = '/api/dataset/{}/train/face-mask-preview'


def _dataset(app, tmp_path, n=3, kind='concept'):
    """Build a kept-set dataset and return its ID.

    The `app` fixture yields the application WITHOUT pushing a context, so every
    `db.session` call here needs its own. Leaning on an ambient one is what broke
    this file in CI and nowhere else: the dev machine happens to have the optional
    `pytest-flask` plugin, whose autouse `_push_request_context` silently wraps
    every test that takes an `app` fixture — and that plugin is not in
    backend/requirements.txt, which is all CI installs. Same code, no plugin, no
    context, `RuntimeError: Working outside of application context`.

    Returns the ID and not the instance on purpose: Flask-SQLAlchemy removes the
    session when the context pops, which would leave the ORM object detached.
    """
    save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
    kw = {'kind': kind}
    if kind == 'concept':
        kw['concept_desc'] = 'balancing a spoon'
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'FMP', 'fmp_act', **kw)
        img_dir = svc._dataset_dir(ds.id)
        for i in range(n):
            fn = f'k{i}.png'
            Image.new('RGB', (64, 64)).save(os.path.join(img_dir, fn))
            db.session.add(FaceDatasetImage(dataset_id=ds.id, status='keep', filename=fn))
        db.session.commit()
        dataset_id = ds.id
    fmp.reset()
    return dataset_id


def _stub_script(tmp_path, name, body):
    """A stand-in for infer/face_mask_infer.py — same protocol, no InsightFace."""
    p = tmp_path / name
    p.write_text(body, encoding='utf-8')
    return str(p)


# A well-behaved run: the phases, then one line per image, then the JSON.
_GOOD = (
    "import json, sys\n"
    "req = json.loads(sys.stdin.read())\n"
    "imgs = req['images']\n"
    "for ph in ('starting', 'loading'):\n"
    "    print('[facemask] phase=%s' % ph, file=sys.stderr, flush=True)\n"
    "print('[facemask] phase=detecting', file=sys.stderr, flush=True)\n"
    "for i, p in enumerate(imgs, 1):\n"
    "    print('[facemask] %d/%d masked faces=1' % (i, len(imgs)), file=sys.stderr, flush=True)\n"
    "print(json.dumps({'ok': True, 'written': 0, 'results': "
    "{p: {'state': 'masked', 'boxes': [[.4, .3, .6, .5]]} for p in imgs}}))\n"
)

# Nobody home: the child dies before printing any JSON — the shape of "InsightFace
# blew up on import" or "the model would not load".
_DEAD = (
    "import sys\n"
    "sys.stdin.read()\n"
    "print('[facemask] phase=starting', file=sys.stderr, flush=True)\n"
    "print('ModuleNotFoundError: No module named \\'insightface\\'', file=sys.stderr, flush=True)\n"
    "sys.exit(3)\n"
)

# A legitimate no-people dataset.
_NO_FACE = (
    "import json, sys\n"
    "imgs = json.loads(sys.stdin.read())['images']\n"
    "print('[facemask] phase=detecting', file=sys.stderr, flush=True)\n"
    "for i, p in enumerate(imgs, 1):\n"
    "    print('[facemask] %d/%d no_face' % (i, len(imgs)), file=sys.stderr, flush=True)\n"
    "print(json.dumps({'ok': True, 'written': 0, 'results': "
    "{p: {'state': 'no_face', 'boxes': []} for p in imgs}}))\n"
)


def _use(monkeypatch, tmp_path, name, body):
    monkeypatch.setattr(face_mask, '_SCRIPT', _stub_script(tmp_path, name, body))
    monkeypatch.setattr(face_mask, '_face_python', lambda: sys.executable)
    monkeypatch.setattr(face_mask, 'is_available', lambda: True)


# --- 1. progress is exposed -------------------------------------------------
def test_the_pass_reports_named_phases_then_a_per_image_count(app, tmp_path, monkeypatch):
    """RED before this wave: detect_faces took no callback and the script printed
    no phase, so nothing at all could be shown between click and result."""
    _use(monkeypatch, tmp_path, 'good.py', _GOOD)
    ds_id = _dataset(app, tmp_path, n=3)
    paths = [os.path.join(svc._dataset_dir(ds_id), f'k{i}.png') for i in range(3)]

    seen = []
    data = face_mask.detect_faces(paths, timeout=60, on_progress=seen.append)

    assert data['ok'] is True
    phases = [r['phase'] for r in seen]
    # The model load is named BEFORE any image is touched — that is the whole
    # point: a bar stuck at 0/3 through it reads as a crash.
    assert phases[:2] == ['starting', 'loading']
    counted = [(r['done'], r['total']) for r in seen if 'done' in r]
    assert counted == [(1, 3), (2, 3), (3, 3)]


def test_a_count_line_alone_still_means_detecting(app):
    """A missed `phase=detecting` must not leave the UI stuck on "Loading…"."""
    assert face_mask.parse_progress_line('[facemask] 2/7 masked faces=1') == {
        'phase': 'detecting', 'done': 2, 'total': 7}
    assert face_mask.parse_progress_line('[facemask] phase=downloading') == {
        'phase': 'downloading'}
    assert face_mask.parse_progress_line('some unrelated chatter') is None


def test_the_route_streams_the_count_into_a_joinable_job(app, client, tmp_path, monkeypatch):
    _use(monkeypatch, tmp_path, 'good.py', _GOOD)
    ds_id = _dataset(app, tmp_path, n=3)
    r = client.post(PREVIEW.format(ds_id), json={'limit': 6})
    assert r.status_code == 202
    # TESTING runs the job inline, so by now it is finished and published.
    body = client.get(PREVIEW.format(ds_id)).get_json()
    assert body['job']['total'] == 3 and body['job']['done'] == 3
    assert body['job']['finished'] is True and body['job']['error'] is None
    assert body['result']['coverage']['masked'] == 3
    assert body['result']['stale'] is False


# --- 2. a failure looks like a failure --------------------------------------
def test_a_dead_subprocess_ends_as_a_message_not_an_endless_wait(app, client, tmp_path,
                                                                 monkeypatch):
    _use(monkeypatch, tmp_path, 'dead.py', _DEAD)
    ds_id = _dataset(app, tmp_path, n=2)
    client.post(PREVIEW.format(ds_id), json={})
    job = client.get(PREVIEW.format(ds_id)).get_json()['job']
    assert job['finished'] is True
    assert job['error'] and 'insightface' in job['error'].lower()


def test_the_tool_being_absent_is_answered_immediately(app, client, tmp_path, monkeypatch):
    """No job, no timeout, no spinner: an install without face detection hears it
    on the spot."""
    monkeypatch.setattr(face_mask, 'is_available', lambda: False)
    ds_id = _dataset(app, tmp_path, n=1)
    r = client.post(PREVIEW.format(ds_id), json={})
    assert r.status_code == 409
    assert r.get_json()['reason'] == 'face_scoring'
    assert fmp.get(ds_id) is None


# --- 3. zero faces is a result, not an error --------------------------------
def test_no_face_anywhere_is_a_valid_result(app, client, tmp_path, monkeypatch):
    _use(monkeypatch, tmp_path, 'noface.py', _NO_FACE)
    ds_id = _dataset(app, tmp_path, n=2)
    client.post(PREVIEW.format(ds_id), json={})
    body = client.get(PREVIEW.format(ds_id)).get_json()
    assert body['job']['error'] is None
    assert body['result']['coverage'] == {
        'total': 2, 'masked': 0, 'no_face': 2, 'too_large': 0, 'failed': 0}
    assert len(body['result']['samples']) == 2


# --- rejoin: the pass and its result survive the page -----------------------
def test_the_result_is_still_there_after_leaving_the_page(app, client, tmp_path, monkeypatch):
    """A fresh GET — what a remounted panel does — returns the computed preview,
    so coming back shows the mask instead of a virgin button."""
    _use(monkeypatch, tmp_path, 'good.py', _GOOD)
    ds_id = _dataset(app, tmp_path, n=3)
    client.post(PREVIEW.format(ds_id), json={})
    body = client.get(PREVIEW.format(ds_id)).get_json()
    assert body['ok'] is True and body['result']['samples']


def test_a_second_click_joins_the_running_pass_instead_of_starting_another(app, tmp_path):
    """One live pass per dataset. Two InsightFace passes over the same images at
    once would be pure burnt CPU."""
    ds_id = _dataset(app, tmp_path, n=1)
    calls = []
    job, started = fmp.start(app, ds_id, lambda j: calls.append(j), total=1)
    assert started is True and len(calls) == 1
    # The inline (TESTING) job above already finished, so simulate a live one.
    fmp.reset(ds_id)
    live = {'phase': 'loading', 'done': 0, 'total': 1, 'error': None,
            'finished': False, 'started_at': 0, '_touched': __import__('time').time(),
            '_fp': None}
    fmp._state[ds_id] = {'job': live, 'result': None}
    _, started2 = fmp.start(app, ds_id, lambda j: calls.append(j), total=1)
    assert started2 is False
    assert len(calls) == 1          # nothing new was run


def test_a_preview_of_images_that_changed_is_flagged_stale(app, client, tmp_path, monkeypatch):
    """Showing a stale preview as fresh would be worse than showing none: the
    boxes would describe photos that are no longer in the run."""
    _use(monkeypatch, tmp_path, 'good.py', _GOOD)
    ds_id = _dataset(app, tmp_path, n=2)
    client.post(PREVIEW.format(ds_id), json={})
    assert client.get(PREVIEW.format(ds_id)).get_json()['result']['stale'] is False

    fn = 'added.png'
    Image.new('RGB', (64, 64)).save(os.path.join(svc._dataset_dir(ds_id), fn))
    with app.app_context():     # see _dataset(): the fixture pushes none
        db.session.add(FaceDatasetImage(dataset_id=ds_id, status='keep', filename=fn))
        db.session.commit()
    assert client.get(PREVIEW.format(ds_id)).get_json()['result']['stale'] is True


def test_the_status_route_never_starts_anything(app, client, tmp_path, monkeypatch):
    monkeypatch.setattr(face_mask, 'is_available', lambda: True)
    monkeypatch.setattr(face_mask, '_SCRIPT', _stub_script(tmp_path, 'boom.py', _DEAD))
    ds_id = _dataset(app, tmp_path, n=1)
    body = client.get(PREVIEW.format(ds_id)).get_json()
    assert body['ok'] is True and body['job'] is None and body['result'] is None
