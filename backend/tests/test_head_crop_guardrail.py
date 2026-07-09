"""Reference head-crop: robustness + the silent-fallback guard-rail.

Root cause we hardened against: the configured Ollama vision model wasn't pulled, so
detect_head_bbox got a 404 -> None -> face_crop silently produced a body-centered crop
and the user had no idea. Fixes:
  - detect_head_bbox normalizes SWAPPED corners (Qwen3-VL emits y1>y2/x1>x2) instead of
    rejecting them to None;
  - face_crop_to_square_webp reports whether it actually head-cropped;
  - the /ref route returns a WARNING (naming the missing vision model) when it fell back.

Vision is imported locally by detect_head_bbox, so we patch app.services.vision_ollama.*.
"""
import io
import contextlib

from PIL import Image

from app.services import face_dataset_service as svc


def _png(w=256, h=256):
    b = io.BytesIO()
    Image.new('RGB', (w, h), (120, 40, 40)).save(b, 'PNG')
    return b.getvalue()


# --- detect_head_bbox robustness --------------------------------------------
def test_detect_head_bbox_normalizes_swapped_corners(app, monkeypatch):
    """Qwen3-VL often returns y1>y2 / x1>x2. We must normalize to min/max, not reject
    (rejecting was the silent-None that fell back to a body-centered crop)."""
    from app.services import vision_ollama
    with app.app_context():
        # corners deliberately swapped: y1>y2 and x1>x2
        monkeypatch.setattr(vision_ollama, 'describe_image_ollama',
                            lambda *a, **k: '{"y1":800,"x1":700,"y2":200,"x2":100}')
        box = svc.detect_head_bbox(_png())
        assert box == (0.1, 0.2, 0.7, 0.8)  # normalized + scaled to 0-1


def test_detect_head_bbox_forces_json_format(app, monkeypatch):
    """The bbox call must pass fmt='json' so a reasoning model can't ramble past the
    token budget and never emit coords."""
    from app.services import vision_ollama
    seen = {}

    def fake(image_bytes, prompt, **kw):
        seen.update(kw)
        return '{"y1":100,"x1":100,"y2":400,"x2":400}'

    with app.app_context():
        monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake)
        svc.detect_head_bbox(_png())
        assert seen.get('fmt') == 'json'


def test_detect_head_bbox_empty_response_is_none(app, monkeypatch):
    """Model unavailable / 404 -> describe returns '' -> None (caller centers + warns)."""
    from app.services import vision_ollama
    with app.app_context():
        monkeypatch.setattr(vision_ollama, 'describe_image_ollama', lambda *a, **k: '')
        assert svc.detect_head_bbox(_png()) is None


# --- face_crop reports whether it head-cropped ------------------------------
def test_face_crop_reports_detection(app, monkeypatch):
    with app.app_context():
        monkeypatch.setattr(svc, 'detect_head_bbox', lambda *a, **k: (0.25, 0.15, 0.75, 0.85))
        webp, detected = svc.face_crop_to_square_webp(_png(), return_detected=True)
        assert detected is True and webp[:4] == b'RIFF'  # WEBP container


def test_face_crop_reports_fallback_when_no_head(app, monkeypatch):
    with app.app_context():
        monkeypatch.setattr(svc, 'detect_head_bbox', lambda *a, **k: None)
        webp, detected = svc.face_crop_to_square_webp(_png(), return_detected=True)
        assert detected is False and webp  # still a valid centered crop


# --- /ref route surfaces the guard-rail warning -----------------------------
def _create_concept_free_dataset(client):
    return client.post('/api/dataset/create',
                       json={'name': 'Emma', 'trigger_word': 'zchar_emma'}).get_json()['id']


def test_ref_route_warns_when_model_not_ready(client, monkeypatch):
    import app.routes.datasets as dr
    import app.capabilities as caps
    did = _create_concept_free_dataset(client)
    # head-crop falls back (no detection) AND the vision model probes as not-ready
    monkeypatch.setattr(dr.svc, 'face_crop_to_square_webp', lambda raw, **k: (b'RIFFwebp', False))
    monkeypatch.setattr(dr, 'gpu_exclusive_vision_window', lambda: contextlib.nullcontext())
    monkeypatch.setattr(caps, 'probe_ollama_model', lambda *a, **k: {'ok': False, 'detail': 'not pulled'})

    resp = client.post(f'/api/dataset/{did}/ref',
                       data={'file': (io.BytesIO(_png()), 'ref.png')},
                       content_type='multipart/form-data')
    body = resp.get_json()
    assert resp.status_code == 200 and body['ok'] is True
    assert body['head_crop'] is False
    assert 'Setup' in body['warning'] and 'vision model' in body['warning']


def test_ref_route_warns_face_not_found_when_model_ready(client, monkeypatch):
    import app.routes.datasets as dr
    import app.capabilities as caps
    did = _create_concept_free_dataset(client)
    monkeypatch.setattr(dr.svc, 'face_crop_to_square_webp', lambda raw, **k: (b'RIFFwebp', False))
    monkeypatch.setattr(dr, 'gpu_exclusive_vision_window', lambda: contextlib.nullcontext())
    monkeypatch.setattr(caps, 'probe_ollama_model', lambda *a, **k: {'ok': True, 'detail': 'ready'})

    resp = client.post(f'/api/dataset/{did}/ref',
                       data={'file': (io.BytesIO(_png()), 'ref.png')},
                       content_type='multipart/form-data')
    body = resp.get_json()
    assert body['head_crop'] is False
    assert "Couldn't detect a face" in body['warning']


def test_ref_route_no_warning_when_head_detected(client, monkeypatch):
    import app.routes.datasets as dr
    did = _create_concept_free_dataset(client)
    monkeypatch.setattr(dr.svc, 'face_crop_to_square_webp', lambda raw, **k: (b'RIFFwebp', True))
    monkeypatch.setattr(dr, 'gpu_exclusive_vision_window', lambda: contextlib.nullcontext())

    resp = client.post(f'/api/dataset/{did}/ref',
                       data={'file': (io.BytesIO(_png()), 'ref.png')},
                       content_type='multipart/form-data')
    body = resp.get_json()
    assert body['head_crop'] is True and 'warning' not in body
