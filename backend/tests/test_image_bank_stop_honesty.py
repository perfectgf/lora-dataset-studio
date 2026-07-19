"""🗃️ Image bank — an honest, lossless Stop for the face + score passes.

Stopping 👥 Group by person or ✨ Score used to SIGKILL the child (losing the
in-flight slice) and return mutely. These tests pin the new contract:

* the child stops COOPERATIVELY on a sentinel file, flushing its cache (0 loss)
  and printing how much it kept — with a watchdog kill only as a fallback;
* the pass then reports an HONEST end line ("Stopped — N … (M remaining) …"),
  preferring the child's own counts and falling back to a flushed sidecar;
* relaunching a partly-cached bank shows a "resuming — K of T" hint;
* both passes share one driver, and a stopped step inside 🚀 Launch all inherits
  the same honest report.

The heavy ML runs in a subprocess; here the driver is exercised against a tiny
stdlib-only fake script, and the job branching against a monkeypatched driver —
no torch / insightface needed."""
import importlib.util
import json
import os
import pathlib
import threading
import time
from collections import deque

import pytest
from PIL import Image


def _flat(size=64, value=128):
    return Image.new('RGB', (size, size), (value, value, value))


def _mkbank(client, tmp_path, files, name='B'):
    src = tmp_path / 'src'
    for rel, im in files.items():
        p = src / rel
        os.makedirs(p.parent, exist_ok=True)
        im.save(str(p))
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


def _fresh_job(kind):
    """A job dict shaped like bank_jobs.start()'s, usable with its helpers."""
    now = time.time()
    return {'kind': kind, 'done': 0, 'total': 0, 'error': None,
            'cancelled': False, 'finished': False, 'detail': None,
            'started_at': now, '_touched': now, '_cancel_hook': None,
            'pipeline': None}


# --- infer-script helpers (no ML) --------------------------------------------
def _load_infer(name):
    spec = importlib.util.spec_from_file_location(
        name, pathlib.Path(__file__).resolve().parents[1] / 'infer' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('script', ['face_embed_infer', 'bank_score_infer'])
def test_infer_cancel_and_count_helpers(script, tmp_path):
    mod = _load_infer(script)
    cache = str(tmp_path / 'c.npz')
    cancel = cache + '.cancel'
    assert mod._cancel_requested(cancel) is False
    assert mod._cancel_requested(None) is False
    open(cancel, 'w').close()
    assert mod._cancel_requested(cancel) is True
    mod._write_count(cache, 42)
    with open(cache + '.count') as f:
        assert f.read().strip() == '42'
    mod._write_count(None, 7)   # no path → silent no-op, never raises


# --- service count sidecar + honest message ----------------------------------
def test_read_cache_count_roundtrip_and_missing(app, tmp_path):
    from app.services import image_bank_service as banks
    cache = tmp_path / 'score_cache.npz'
    assert banks._read_cache_count(cache) is None            # nothing written yet
    (tmp_path / 'score_cache.npz.count').write_text('1240', encoding='utf-8')
    assert banks._read_cache_count(cache) == 1240
    (tmp_path / 'score_cache.npz.count').write_text('junk', encoding='utf-8')
    assert banks._read_cache_count(cache) is None            # unparsable → None


def test_stopped_detail_prefers_child_counts(app):
    from app.services import image_bank_service as banks
    d = banks._stopped_detail('face embeddings cached',
                              {'cancelled': True, 'cached': 1240, 'remaining': 760},
                              '/nope', total=2000)
    assert d == ('Stopped — 1240 face embeddings cached (760 remaining); '
                 'relaunch to finish and cluster')


def test_stopped_detail_derives_remaining_and_falls_back_to_sidecar(app, tmp_path):
    from app.services import image_bank_service as banks
    # remaining absent → derived from total
    d = banks._stopped_detail('images scored', {'cached': 30}, '/nope', total=100)
    assert '30 images scored (70 remaining)' in d
    # no child count at all → read the flushed sidecar
    cache = tmp_path / 'c.npz'
    (tmp_path / 'c.npz.count').write_text('12', encoding='utf-8')
    d = banks._stopped_detail('images scored', {}, cache, total=50)
    assert '12 images scored (38 remaining)' in d
    # nothing to go on → honest, numberless line (never a fabricated count)
    d = banks._stopped_detail('images scored', {}, tmp_path / 'gone.npz', total=50)
    assert d == 'Stopped — progress saved to cache; relaunch to finish and cluster'


# --- the cooperative driver against a stdlib-only fake child ------------------
_FAKE_COOPERATIVE = '''\
import sys, os, json, time
req = json.loads(sys.stdin.read())
cf, cache = req["cancel_file"], req["cache"]
sys.stderr.write("[embed] 100 image(s), 40 cached\\n"); sys.stderr.flush()
for i in range(1, 100000):
    if os.path.exists(cf):
        with open(cache + ".count", "w") as f: f.write(str(40 + i))
        print(json.dumps({"ok": True, "cancelled": True,
                          "cached": 40 + i, "remaining": max(0, 60 - i)}))
        sys.exit(0)
    time.sleep(0.01)
'''

_FAKE_STUBBORN = '''\
import sys, json, time
json.loads(sys.stdin.read())
time.sleep(30)   # ignores the sentinel — must be hard-killed by the watchdog
'''


def _run_driver_in_thread(banks, job, script_path, payload, cache_path):
    import re
    from contextlib import nullcontext
    out = {}

    def worker():
        out['res'] = banks._drive_infer_subprocess(
            job, __import__('sys').executable, script_path, payload,
            cache_path, re.compile(r'\[embed\] (\d+)/(\d+)'), nullcontext())

    t = threading.Thread(target=worker)
    t.start()
    return t, out


def test_driver_cooperative_cancel_flushes_and_reports(app, tmp_path):
    from app.services import image_bank_service as banks
    script = tmp_path / 'fake_coop.py'
    script.write_text(_FAKE_COOPERATIVE, encoding='utf-8')
    cache_path = str(tmp_path / 'face_cache.npz')
    payload = json.dumps({'images': ['a'] * 100, 'cache': cache_path,
                          'cancel_file': cache_path + '.cancel'})
    job = _fresh_job('faces')
    t, out = _run_driver_in_thread(banks, job, str(script), payload, cache_path)
    # Wait until the child armed the cancel hook, then Stop it.
    for _ in range(500):
        if job['_cancel_hook']:
            break
        time.sleep(0.01)
    assert job['_cancel_hook'], 'cancel hook was never registered'
    job['cancelled'] = True
    job['_cancel_hook']()
    t.join(timeout=10)
    assert not t.is_alive()
    data, _tail, rc = out['res']
    assert data.get('cancelled') is True
    assert data['cached'] > 40 and data['remaining'] <= 60
    assert rc == 0                                          # clean exit, not a kill
    # The child flushed its cache AND its count sidecar (0-loss guarantee).
    assert banks._read_cache_count(cache_path) == data['cached']
    # The resume hint fired off the "40 cached" stderr line.
    assert job['detail'] == 'resuming — 40 of 100 already cached'
    # No sentinel is left behind to poison the next run.
    assert not os.path.exists(cache_path + '.cancel')


def test_driver_hard_kills_a_stubborn_child(app, tmp_path, monkeypatch):
    from app.services import image_bank_service as banks
    monkeypatch.setattr(banks, '_INFER_CANCEL_GRACE', 0.3)
    script = tmp_path / 'fake_stubborn.py'
    script.write_text(_FAKE_STUBBORN, encoding='utf-8')
    cache_path = str(tmp_path / 'face_cache.npz')
    payload = json.dumps({'images': ['a'], 'cache': cache_path,
                          'cancel_file': cache_path + '.cancel'})
    job = _fresh_job('faces')
    t, out = _run_driver_in_thread(banks, job, str(script), payload, cache_path)
    for _ in range(500):
        if job['_cancel_hook']:
            break
        time.sleep(0.01)
    job['cancelled'] = True
    started = time.time()
    job['_cancel_hook']()
    t.join(timeout=10)
    assert not t.is_alive()
    # Killed within a small multiple of the grace window — nowhere near the 30 s sleep.
    assert time.time() - started < 5
    data, _tail, rc = out['res']
    assert not data.get('cancelled')     # no clean cancel line — it was killed
    assert rc is not None                # the process really terminated


# --- job branching (driver monkeypatched — no subprocess) --------------------
def _paths_of(payload):
    return json.loads(payload)['images']


def test_faces_job_cooperative_stop_reports_and_leaves_db_untouched(
        client, tmp_path, app, monkeypatch):
    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _flat(), 'b.jpg': _flat(9)})
    from app.services import image_bank_service as banks
    monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))

    def fake_driver(job, python, script, payload, cache_path, rx, window):
        return {'ok': True, 'cancelled': True, 'cached': 1, 'remaining': 1}, deque(), 0

    monkeypatch.setattr(banks, '_drive_infer_subprocess', fake_driver)
    job = _fresh_job('faces')
    with app.app_context():
        banks._faces_job(bank_id)(job)
    assert job['detail'] == ('Stopped — 1 face embeddings cached (1 remaining); '
                             'relaunch to finish and cluster')
    # Cancelled → nothing persisted (the cache holds the truth, DB stays clean).
    with app.app_context():
        from app.models import BankImage
        assert all(r.face_state is None
                   for r in BankImage.query.filter_by(bank_id=bank_id).all())


def test_faces_job_hard_kill_uses_sidecar_count(client, tmp_path, app, monkeypatch):
    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _flat(), 'b.jpg': _flat(9),
                                            'c.jpg': _flat(30)})
    from app.services import image_bank_service as banks
    monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))

    def fake_driver(job, python, script, payload, cache_path, rx, window):
        # Simulate a watchdog kill: no cancel line, but a flushed sidecar remains.
        os.makedirs(os.path.dirname(str(cache_path)), exist_ok=True)
        with open(str(cache_path) + '.count', 'w', encoding='utf-8') as f:
            f.write('2')
        return {}, deque(), 137

    monkeypatch.setattr(banks, '_drive_infer_subprocess', fake_driver)
    job = _fresh_job('faces')
    job['cancelled'] = True                # the user did press Stop
    with app.app_context():
        banks._faces_job(bank_id)(job)
    assert job['detail'] == ('Stopped — 2 face embeddings cached (1 remaining); '
                             'relaunch to finish and cluster')


def test_score_job_cooperative_stop_reports_honestly(client, tmp_path, app, monkeypatch):
    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _flat(), 'b.jpg': _flat(9)})
    from app import capabilities
    from app.services import image_bank_service as banks
    monkeypatch.setattr(capabilities, 'probe_bank_scoring', lambda: {'ok': True})

    def fake_driver(job, python, script, payload, cache_path, rx, window):
        return {'ok': True, 'cancelled': True, 'cached': 1, 'remaining': 1}, deque(), 0

    monkeypatch.setattr(banks, '_drive_infer_subprocess', fake_driver)
    job = _fresh_job('score')
    with app.app_context():
        banks._score_job(bank_id)(job)
    assert job['detail'] == ('Stopped — 1 images scored (1 remaining); '
                             'relaunch to finish and cluster')


def test_faces_job_success_still_persists(client, tmp_path, app, monkeypatch):
    """Regression: the refactor must not break the happy path — a full result
    still writes face_state / cluster to every row."""
    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _flat(), 'b.jpg': _flat(9)})
    from app.services import image_bank_service as banks
    monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))

    def fake_driver(job, python, script, payload, cache_path, rx, window):
        imgs = _paths_of(payload)
        results = {p: {'state': 'scorable', 'det': 0.9} for p in imgs}
        clusters = {p: 1 for p in imgs}
        return ({'ok': True, 'results': results, 'clusters': clusters},
                deque(), 0)

    monkeypatch.setattr(banks, '_drive_infer_subprocess', fake_driver)
    job = _fresh_job('faces')
    with app.app_context():
        banks._faces_job(bank_id)(job)
        from app.models import BankImage
        rows = BankImage.query.filter_by(bank_id=bank_id).all()
        assert rows and all(r.face_state == 'scorable' for r in rows)
        assert all(r.face_cluster == 1 for r in rows)
    assert 'person cluster' in (job['detail'] or '')


def test_launch_all_step_inherits_honest_stop(client, tmp_path, app, monkeypatch):
    """A stopped faces step inside 🚀 Launch all reports the same honest line."""
    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _flat(), 'b.jpg': _flat(9)})
    from app.services import image_bank_service as banks
    monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))
    monkeypatch.setattr(banks, '_faces_prereq', lambda: None)

    def fake_driver(job, python, script, payload, cache_path, rx, window):
        return {'ok': True, 'cancelled': True, 'cached': 5, 'remaining': 3}, deque(), 0

    monkeypatch.setattr(banks, '_drive_infer_subprocess', fake_driver)
    job = _fresh_job('pipeline')
    entry = {'step': 'faces', 'status': 'ran'}
    with app.app_context():
        banks._run_pipeline_step(job, 1, bank_id, 'faces', {}, False, entry)
    assert entry['detail'] == ('Stopped — 5 face embeddings cached (3 remaining); '
                               'relaunch to finish and cluster')
