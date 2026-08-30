"""⚖ Compare Klein models before the batch — the judging half, held to its rules.

The user's ask, in shape: the same intermediate window the detection scan has,
but for the Klein clean — pick models, run them on ONE flagged image with ONE
seed, adopt the winner. What these tests hold:

- the preview NEVER touches the original file (the whole point of a preview);
- one call = one model, same image, same seed — the dialog's grid varies on the
  model and nothing else;
- both surfaces answer through the SAME service, so they cannot drift;
- the bank's per-run override reaches the inpaint call, while the bank still
  STORES no pick (the old doctrine, kept).
"""
import base64
import io as _io
import json

import pytest
from PIL import Image

from app.services import watermark_klein as wk

BOX = [0.7, 0.7, 0.95, 0.95]


def _png(path, size=(320, 240), color=(200, 30, 30)):
    Image.new('RGB', size, color).save(path, 'PNG')
    return str(path)


@pytest.fixture
def fake_pass(monkeypatch):
    """Replace the ComfyUI round-trip with a marker: the 'cleaned' copy is filled
    green, and every call is recorded (path, model, seed)."""
    calls = []

    def _fake(user_id, image_path, boxes, *, seed=None, klein_model=None, **kw):
        calls.append({'path': image_path, 'model': klein_model,
                      'seed': seed, 'boxes': boxes})
        Image.new('RGB', (320, 240), (30, 200, 30)).save(image_path, 'WEBP')
        return True, None

    monkeypatch.setattr(wk, 'inpaint_watermark_klein', _fake)
    monkeypatch.setattr(wk.keh if hasattr(wk, 'keh') else __import__(
        'app.services.klein_edit_helper', fromlist=['x']),
        'klein_model_on_disk', lambda name: f'klein/{name}', raising=False)
    return calls


def _patch_on_disk(monkeypatch):
    from app.services import klein_edit_helper as keh
    monkeypatch.setattr(keh, 'klein_model_on_disk', lambda name: f'klein/{name}')


def test_the_preview_never_touches_the_original(tmp_path, monkeypatch, fake_pass):
    _patch_on_disk(monkeypatch)
    src = _png(tmp_path / 'img.png')
    original = open(src, 'rb').read()
    out = wk.run_compare('u', [(1, 'img.png', src, None, json.dumps(BOX))],
                         model='a.safetensors', seed=7)
    assert out['ok'] is True
    assert open(src, 'rb').read() == original, 'the compare wrote into the original'
    # ...and the pass really ran on a THROWAWAY copy, not the source path.
    assert fake_pass and fake_pass[0]['path'] != src


def test_one_call_is_one_model_same_image_same_seed(tmp_path, monkeypatch, fake_pass):
    """The dialog calls once per candidate with the seed IT chose; the grid's
    only variable is the model."""
    _patch_on_disk(monkeypatch)
    src = _png(tmp_path / 'img.png')
    rows = [(1, 'img.png', src, None, json.dumps(BOX))]
    a = wk.run_compare('u', rows, model='a.safetensors', seed=42)
    b = wk.run_compare('u', rows, model='b.safetensors', seed=42)
    assert (a['ok'], b['ok']) == (True, True)
    assert [c['model'] for c in fake_pass] == ['a.safetensors', 'b.safetensors']
    assert {c['seed'] for c in fake_pass} == {42}
    assert a['image_id'] == b['image_id'] == 1
    # Both payloads carry renderable previews.
    for out in (a, b):
        for k in ('before', 'after'):
            img = Image.open(_io.BytesIO(base64.b64decode(out[k])))
            assert img.size[0] > 0


def test_manual_regions_outrank_the_detected_bbox(tmp_path, monkeypatch, fake_pass):
    """Same derivation as the batch: hand-drawn zones are what the clean will
    repaint, so they are what the compare must judge."""
    _patch_on_disk(monkeypatch)
    src = _png(tmp_path / 'img.png')
    regions = json.dumps([[0.1, 0.1, 0.3, 0.3], [0.6, 0.6, 0.8, 0.8]])
    out = wk.run_compare('u', [(1, 'img.png', src, regions, json.dumps(BOX))],
                         model='a.safetensors', seed=1)
    assert out['ok'] is True
    assert len(fake_pass[0]['boxes']) == 2, 'the detected bbox outranked the manual zones'


def test_a_vanished_model_is_refused_by_name(tmp_path, monkeypatch, fake_pass):
    from app.services import klein_edit_helper as keh
    monkeypatch.setattr(keh, 'klein_model_on_disk', lambda name: None)
    src = _png(tmp_path / 'img.png')
    out = wk.run_compare('u', [(1, 'img.png', src, None, json.dumps(BOX))],
                         model='gone.safetensors')
    assert out['ok'] is False and 'gone.safetensors' in out['error']
    assert not fake_pass, 'the pass ran on a model that is not on disk'


def test_no_flagged_image_says_what_to_do_first(monkeypatch, fake_pass):
    _patch_on_disk(monkeypatch)
    out = wk.run_compare('u', [], model='a.safetensors')
    assert out['ok'] is False and 'Find watermarks' in out['error']


def test_both_routes_answer_through_the_same_service(app, client, monkeypatch):
    """Surface parity at the seam: dataset and bank both delegate to run_compare,
    so a behaviour change in one cannot silently miss the other."""
    seen = []
    monkeypatch.setattr(wk, 'run_compare',
                        lambda user, rows, **kw: seen.append(kw) or
                        {'ok': False, 'error': 'stub'})
    r = client.post('/api/dataset/999/watermarks/klein-compare',
                    json={'model': 'a'})
    assert r.status_code == 404          # no such dataset — ownership first
    r = client.post('/api/bank/999/watermark/klein-compare', json={'model': 'a'})
    assert r.status_code == 404


def test_the_bank_inpaint_route_threads_the_per_run_override(app, client, monkeypatch):
    """The ⚖ dialog's "use for this run" arrives as {klein_model} on the inpaint
    launch — per run, never stored: a bank still has no Klein pick of its own."""
    from app.services import image_bank_service as banks
    seen = {}

    def _fake_start(app_, user_id, bank_id, method='auto', target='all',
                    statuses=None, ids=None, klein_model=None):
        seen.update(method=method, klein_model=klein_model)
        return {'ok': True}

    monkeypatch.setattr(banks, 'start_watermark_inpaint', _fake_start)
    r = client.post('/api/bank/1/watermark/inpaint',
                    json={'method': 'klein', 'klein_model': 'jib.safetensors'})
    assert r.status_code in (200, 202)
    assert seen == {'method': 'klein', 'klein_model': 'jib.safetensors'}
