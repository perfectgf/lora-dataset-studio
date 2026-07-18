"""🗃️ Image bank V2 — aesthetic/NSFW/style scoring, overlaid-watermark scan and
subfolder scoping. The heavy ML runs in a dedicated subprocess/venv; here we
exercise the parts that don't need torch or Ollama: the read-time score flags
(persisted scores → verdicts against the live thresholds), the facets, the
aesthetic-aware keep-best, the gates, and the graceful degradation when an extra
is absent. Background jobs run inline under TESTING (see bank_jobs.start)."""
import importlib.util
import os
import pathlib
import random

import pytest
from PIL import Image


# --- factories (mirror test_image_bank) --------------------------------------
def _save(path, im):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.lower().endswith(('.jpg', '.jpeg')):
        im.save(path, 'JPEG', quality=92)
    else:
        im.save(path)


def _flat(size=128, value=128):
    return Image.new('RGB', (size, size), (value, value, value))


def _photo(size=256):
    """Smooth gradient + a disc — stable dHash across resizes (for dup groups)."""
    im = Image.new('L', (size, size))
    c, r2 = size / 2, (size / 3) ** 2
    im.putdata([min(255, int(150 * x / size + 50 * y / size)
                    + (80 if (x - c) ** 2 + (y - c) ** 2 < r2 else 0))
                for y in range(size) for x in range(size)])
    return im.convert('RGB')


def _mkbank(client, tmp_path, files, name='B'):
    src = tmp_path / 'src'
    for rel, im in files.items():
        _save(str(src / rel), im)
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


def _set_scores(app, bank_id, **by_name):
    """Write raw scores straight onto rows, keyed by file basename."""
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        rows = {os.path.basename(r.relpath): r
                for r in BankImage.query.filter_by(bank_id=bank_id).all()}
        for name, vals in by_name.items():
            for k, v in vals.items():
                setattr(rows[name], k, v)
        db.session.commit()


def _by_name(client, bank_id, **params):
    q = '&'.join(f'{k}={v}' for k, v in params.items())
    url = f'/api/bank/{bank_id}/images' + (f'?{q}' if q else '')
    return {i['name']: i for i in client.get(url).get_json()['images']}


# --- read-time score flags ---------------------------------------------------
def test_score_flags_are_read_time_and_retunable(client, tmp_path, app):
    bank_id, _ = _mkbank(client, tmp_path, {
        'nice.jpg': _flat(), 'ugly.jpg': _flat(200),
        'porn.jpg': _flat(50), 'marked.jpg': _flat(90)})
    _set_scores(app, bank_id,
                **{'nice.jpg': {'aesthetic_score': 7.5, 'nsfw_score': 0.01},
                   'ugly.jpg': {'aesthetic_score': 3.0, 'nsfw_score': 0.02},
                   'porn.jpg': {'aesthetic_score': 6.0, 'nsfw_score': 0.95},
                   'marked.jpg': {'watermark_state': 'detected'}})
    by = _by_name(client, bank_id)
    assert 'low_aesthetic' in by['ugly.jpg']['flags']       # 3.0 < default 5.0
    assert 'low_aesthetic' not in by['nice.jpg']['flags']
    assert 'nsfw' in by['porn.jpg']['flags']                # 0.95 > default 0.5
    assert 'nsfw' not in by['nice.jpg']['flags']
    assert 'watermark' in by['marked.jpg']['flags']
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['flags']['low_aesthetic'] == 1
    assert payload['flags']['nsfw'] == 1
    assert payload['flags']['watermark'] == 1
    assert payload['counts']['scored'] == 3                 # nice/ugly/porn scored
    assert payload['counts']['watermark_scanned'] == 1
    # Retuning a threshold re-sorts the bank with NO rescan.
    with app.app_context():
        import app.config as cfg
        cfg.save_config({'bank': {'aesthetic_min': 6.5}})
    by = _by_name(client, bank_id)
    assert 'low_aesthetic' in by['porn.jpg']['flags']       # 6.0 < 6.5 now
    assert 'low_aesthetic' not in by['nice.jpg']['flags']   # 7.5 still above


def test_null_score_is_not_below_threshold(client, tmp_path, app):
    """An unscored image (NULL) must never count as low_aesthetic/nsfw."""
    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _flat(), 'b.jpg': _flat(9)})
    _set_scores(app, bank_id, **{'a.jpg': {'aesthetic_score': 2.0}})
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['flags']['low_aesthetic'] == 1           # only the scored one
    by = _by_name(client, bank_id)
    assert by['b.jpg']['flags'] == [] or 'low_aesthetic' not in by['b.jpg']['flags']


def test_filter_by_score_flag_orders_worst_first(client, tmp_path, app):
    bank_id, _ = _mkbank(client, tmp_path, {
        'a.jpg': _flat(), 'b.jpg': _flat(60), 'c.jpg': _flat(200)})
    _set_scores(app, bank_id,
                **{'a.jpg': {'aesthetic_score': 2.0}, 'b.jpg': {'aesthetic_score': 4.0},
                   'c.jpg': {'aesthetic_score': 8.0}})
    r = client.get(f'/api/bank/{bank_id}/images?flag=low_aesthetic').get_json()
    assert [i['name'] for i in r['images']] == ['a.jpg', 'b.jpg']   # worst first, c out


def test_apply_flags_rejects_nsfw_pending_only(client, tmp_path, app):
    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _flat(), 'b.jpg': _flat(50)})
    _set_scores(app, bank_id, **{'b.jpg': {'nsfw_score': 0.9}})
    r = client.post(f'/api/bank/{bank_id}/apply-flags', json={'flags': ['nsfw']})
    assert r.get_json()['rejected'] == {'nsfw': 1}
    by = _by_name(client, bank_id)
    assert by['b.jpg']['status'] == 'reject'
    assert by['b.jpg']['reject_reason'] == 'nsfw'
    assert by['a.jpg']['status'] == 'pending'


# --- style clusters ----------------------------------------------------------
def test_style_cluster_orders_by_size():
    np = pytest.importorskip('numpy')
    spec = importlib.util.spec_from_file_location(
        'bank_score_infer',
        pathlib.Path(__file__).resolve().parents[1] / 'infer' / 'bank_score_infer.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    v1 = np.array([1.0] + [0.0] * 767, dtype='float32')
    v2 = np.array([0.0, 1.0] + [0.0] * 766, dtype='float32')
    cache = {'p1': ('ok', 6.0, 0.1, v1), 'p2': ('ok', 6.0, 0.1, v1),
             'p3': ('ok', 6.0, 0.1, v1), 'p4': ('ok', 6.0, 0.1, v2),
             'p5': ('ok', 6.0, 0.1, v2), 'p6': ('error', None, None, v2 * 0)}
    out = mod._cluster_style(list(cache), cache, 0.6)
    assert out['p1'] == out['p2'] == out['p3'] == 1         # biggest cluster = id 1
    assert out['p4'] == out['p5'] == 2
    assert 'p6' not in out                                  # errored/zero emb → no cluster


def test_style_clusters_in_payload_and_filter(client, tmp_path, app):
    bank_id, _ = _mkbank(client, tmp_path, {
        'a.jpg': _flat(), 'b.jpg': _flat(60), 'c.jpg': _flat(200)})
    _set_scores(app, bank_id, **{'a.jpg': {'style_cluster': 1}, 'b.jpg': {'style_cluster': 1},
                                 'c.jpg': {'style_cluster': 2}})
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert [(c['id'], c['size']) for c in payload['style_clusters']] == [(1, 2), (2, 1)]
    r = client.get(f'/api/bank/{bank_id}/images?style=1').get_json()
    assert {i['name'] for i in r['images']} == {'a.jpg', 'b.jpg'}


# --- subfolder scoping -------------------------------------------------------
def test_subfolders_facet_and_filter(client, tmp_path):
    bank_id, _ = _mkbank(client, tmp_path, {
        'chatA/1.jpg': _flat(), 'chatA/2.jpg': _flat(),
        'chatB/1.jpg': _flat(), 'root.jpg': _flat()})
    facet = client.get(f'/api/bank/{bank_id}/subfolders').get_json()
    assert {s['name']: s['count'] for s in facet['subfolders']} == \
        {'chatA': 2, 'chatB': 1, '': 1}
    assert facet['total'] == 4
    scoped = client.get(f'/api/bank/{bank_id}/images?subfolder=chatA').get_json()
    assert scoped['total'] == 2
    assert all(i['subfolder'] == 'chatA' for i in scoped['images'])
    root = client.get(f'/api/bank/{bank_id}/images?subfolder=').get_json()
    assert [i['name'] for i in root['images']] == ['root.jpg']


# --- aesthetic-aware keep-best ----------------------------------------------
def test_keep_best_prefers_aesthetic_when_scored(client, tmp_path, app):
    big = _photo(256)
    small = big.resize((96, 96), Image.LANCZOS)             # same content, downscaled
    bank_id, _ = _mkbank(client, tmp_path, {'orig.jpg': big, 'copy.jpg': small})
    client.post(f'/api/bank/{bank_id}/scan', json={})
    # Give the SMALLER copy the higher aesthetic — it must now win keep-best,
    # overriding the resolution heuristic.
    _set_scores(app, bank_id, **{'orig.jpg': {'aesthetic_score': 2.0},
                                 'copy.jpg': {'aesthetic_score': 8.0}})
    groups = client.get(f'/api/bank/{bank_id}/dup-groups').get_json()['groups']
    assert len(groups) == 1
    best = next(i for i in groups[0]['images'] if i['id'] == groups[0]['best_id'])
    assert best['name'] == 'copy.jpg'


# --- gates -------------------------------------------------------------------
def test_score_gate_503_when_extra_absent(client, tmp_path, monkeypatch):
    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _flat()})
    from app import capabilities
    monkeypatch.setattr(capabilities, 'probe_bank_scoring',
                        lambda: {'ok': False, 'detail': 'import failed'})
    r = client.post(f'/api/bank/{bank_id}/score', json={})
    assert r.status_code == 503
    assert 'bank scoring' in r.get_json()['error']


def test_watermark_gate_503_when_model_absent(client, tmp_path, monkeypatch):
    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _flat()})
    from app import capabilities
    monkeypatch.setattr(capabilities, 'probe_ollama_model',
                        lambda *a, **k: {'ok': False, 'detail': 'not pulled'})
    r = client.post(f'/api/bank/{bank_id}/watermark', json={})
    assert r.status_code == 503
    assert 'vision model' in r.get_json()['error']


def test_score_refuses_when_gpu_busy(client, tmp_path, monkeypatch):
    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _flat()})
    from app import capabilities
    from app.services import image_bank_service as banks
    monkeypatch.setattr(capabilities, 'probe_bank_scoring', lambda: {'ok': True, 'detail': ''})
    monkeypatch.setattr(banks, '_gpu_busy_reason', lambda: 'training is running on the GPU')
    r = client.post(f'/api/bank/{bank_id}/score', json={})
    assert r.status_code == 503
    assert 'training' in r.get_json()['error']


# --- watermark pass (mocked detector — hermetic) -----------------------------
def test_watermark_job_marks_and_leaves_untouched(client, tmp_path, app, monkeypatch):
    # PNG keeps the flat pixel values exact so the fake detector can key off the
    # image content (row scan order is os.walk's, not our dict's — don't assume it).
    bank_id, _ = _mkbank(client, tmp_path, {
        'm.png': _flat(value=128), 'clean.png': _flat(value=60),
        'down.png': _flat(value=90)})
    from app import capabilities
    from app.services import vision_ollama
    monkeypatch.setattr(capabilities, 'probe_ollama_model', lambda *a, **k: {'ok': True})

    def fake_describe(image_bytes, *a, **k):
        import io
        v = Image.open(io.BytesIO(image_bytes)).convert('L').getpixel((0, 0))
        if v == 128:      # m → a present box
            return '{"present": true, "y1": 100, "x1": 100, "y2": 300, "x2": 300}'
        if v == 60:       # clean → a clean answer
            return '{"present": false}'
        return ''         # down → EMPTY (Ollama unreachable): must be left for a retry

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)
    r = client.post(f'/api/bank/{bank_id}/watermark', json={})
    assert r.status_code == 202
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['activity']['finished'] is True
    assert payload['activity']['error'] is None
    by = _by_name(client, bank_id)
    assert by['m.png']['watermark_state'] == 'detected'
    assert 'watermark' in by['m.png']['flags']
    assert by['clean.png']['watermark_state'] == 'none'
    assert by['down.png']['watermark_state'] is None        # left for a retry
    assert payload['counts']['watermark_scanned'] == 2
    assert payload['flags']['watermark'] == 1


# --- score pass degradation (real subprocess, no torch) ----------------------
def test_score_pass_surfaces_missing_ml_deps(client, tmp_path, monkeypatch):
    """With the capability faked available but the ML stack genuinely absent, the
    score subprocess must fail LOUDLY (a job error the UI shows), never silently."""
    if importlib.util.find_spec('open_clip') is not None:
        pytest.skip('open_clip present — the subprocess would attempt a real load')
    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _flat()})
    from app import capabilities
    monkeypatch.setattr(capabilities, 'probe_bank_scoring', lambda: {'ok': True})
    r = client.post(f'/api/bank/{bank_id}/score', json={})
    assert r.status_code == 202
    act = client.get(f'/api/bank/{bank_id}').get_json()['activity']
    assert act['finished'] is True
    assert act['error']
    assert any(s in act['error']
               for s in ('ML deps', 'No module', 'open_clip', 'torch'))
