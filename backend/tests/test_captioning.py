import base64
import io
import json
import os
import subprocess

import pytest
from PIL import Image


def _png(color=(255, 0, 0)):
    buf = io.BytesIO()
    Image.new('RGB', (64, 64), color).save(buf, 'PNG')
    return buf.getvalue()


def _jpeg(color=(0, 128, 255)):
    buf = io.BytesIO()
    Image.new('RGB', (64, 64), color).save(buf, 'JPEG', quality=88)
    return buf.getvalue()


def _webp(color=(0, 200, 0), size=(64, 64), mode='RGB'):
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, 'WEBP', quality=92)
    return buf.getvalue()


class _CapturePost:
    """requests.post stand-in that records the JSON payload so a test can inspect the
    exact image bytes describe_image_ollama sent (after any re-encode)."""
    def __init__(self, response=None):
        self.payload = None
        self._response = response or {'response': 'a caption'}

    def __call__(self, *args, **kwargs):
        self.payload = kwargs.get('json')
        return _Resp(self._response)

    def sent_image_bytes(self):
        return base64.b64decode(self.payload['images'][0])


class _Resp:
    """Minimal stand-in for requests.Response, only what describe_image_ollama reads."""
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _ErrResp:
    """Response stand-in that raises HTTPError like a real Ollama 4xx would — the
    error body ({"error": ...}) rides on the exception's .response, exactly where
    describe_image_ollama now reads it. `json_error` simulates a non-JSON body."""
    def __init__(self, status, body, *, text=None, json_error=False):
        self.status_code = status
        self._body = body
        self._text = text
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise ValueError('response is not JSON')
        return self._body

    @property
    def text(self):
        if self._text is not None:
            return self._text
        return '' if self._body is None else str(self._body)

    def raise_for_status(self):
        import requests
        raise requests.HTTPError(f'{self.status_code} Client Error', response=self)


def _dataset_with_kept_image(svc, LOCAL_USER):
    """Real FaceDataset + one kept FaceDatasetImage backed by a real file on disk,
    matching the pattern used in test_dataset_service.py."""
    from app.models import FaceDatasetImage
    ds = svc.create_dataset(LOCAL_USER, 'CapTest', 'captest')
    d = svc._dataset_dir(ds.id)
    os.makedirs(d, exist_ok=True)
    fn = 'kept.webp'
    with open(os.path.join(d, fn), 'wb') as fh:
        fh.write(_png())
    img = FaceDatasetImage(dataset_id=ds.id, source='import', status='keep', filename=fn, framing='face')
    svc.db.session.add(img)
    svc.db.session.commit()
    return ds, img


# --- vision_ollama.describe_image_ollama ------------------------------------

def test_describe_image_ollama_returns_text_on_success(app, monkeypatch):
    from app.services import vision_ollama
    monkeypatch.setattr(vision_ollama.requests, 'post',
                        lambda *a, **k: _Resp({'response': 'a caption'}))
    with app.app_context():
        out = vision_ollama.describe_image_ollama(_png(), 'describe this')
    assert out == 'a caption'


def test_describe_image_ollama_returns_empty_on_connection_error(app, monkeypatch):
    from app.services import vision_ollama

    def _raise(*a, **k):
        raise ConnectionError('ollama unreachable')

    monkeypatch.setattr(vision_ollama.requests, 'post', _raise)
    with app.app_context():
        out = vision_ollama.describe_image_ollama(_png(), 'describe this')
    assert out == ''


def test_describe_image_ollama_starts_local_server_and_retries_once(app, monkeypatch):
    from app.services import vision_ollama, ollama_control
    calls = []

    def _post(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 1:
            raise ConnectionError('server stopped')
        return _Resp({'response': 'caption after start'})

    monkeypatch.setattr(vision_ollama.requests, 'post', _post)
    monkeypatch.setattr(ollama_control, 'ensure_captioning_ready',
                        lambda: {'ok': True, 'reachable': True})
    with app.app_context():
        out = vision_ollama.describe_image_ollama(
            _png(), 'describe this', auto_start_local=True)
    assert out == 'caption after start' and len(calls) == 2


def test_describe_image_ollama_surfaces_failure_after_successful_start(app, monkeypatch):
    from app.services import vision_ollama, ollama_control

    monkeypatch.setattr(vision_ollama.requests, 'post',
                        lambda *a, **k: (_ for _ in ()).throw(ConnectionError('still down')))
    monkeypatch.setattr(ollama_control, 'ensure_captioning_ready',
                        lambda: {'ok': True, 'reachable': True})
    with app.app_context(), pytest.raises(RuntimeError, match='did not return a caption'):
        vision_ollama.describe_image_ollama(
            _png(), 'describe this', auto_start_local=True)


def test_caption_images_surfaces_ollama_start_failure(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import vision_ollama, ollama_control
    from app.config import LOCAL_USER, save_config

    monkeypatch.setattr(vision_ollama.requests, 'post',
                        lambda *a, **k: (_ for _ in ()).throw(ConnectionError('down')))
    monkeypatch.setattr(ollama_control, 'ensure_captioning_ready',
                        lambda: {'ok': False, 'error': 'Ollama could not start'})
    with app.app_context():
        save_config({'captioning': {'backend': 'ollama'}})
        ds, _ = _dataset_with_kept_image(svc, LOCAL_USER)
        with pytest.raises(RuntimeError, match='Ollama could not start'):
            svc.caption_images(LOCAL_USER, ds.id)


def test_describe_image_ollama_surfaces_http_error_body(app, monkeypatch):
    """Ollama explains a 400 in its JSON body; the user-facing RuntimeError must
    carry that exact reason (not just the status), so no log-diving is needed."""
    from app.services import vision_ollama
    monkeypatch.setattr(vision_ollama.requests, 'post',
                        lambda *a, **k: _ErrResp(400, {'error': 'model requires more system memory'}))
    with app.app_context(), pytest.raises(RuntimeError) as ei:
        vision_ollama.describe_image_ollama(_png(), 'p', auto_start_local=True)
    msg = str(ei.value)
    assert 'HTTP 400' in msg
    assert 'model requires more system memory' in msg


def test_describe_image_ollama_surfaces_non_json_error_body(app, monkeypatch):
    """A non-JSON error body (proxy/HTML) must still reach the user as text,
    not collapse to a bare status code."""
    from app.services import vision_ollama
    monkeypatch.setattr(vision_ollama.requests, 'post',
                        lambda *a, **k: _ErrResp(400, None, text='Bad Request: unknown field images',
                                                 json_error=True))
    with app.app_context(), pytest.raises(RuntimeError) as ei:
        vision_ollama.describe_image_ollama(_png(), 'p', auto_start_local=True)
    assert 'Bad Request: unknown field images' in str(ei.value)


def test_describe_image_ollama_reports_status_when_body_empty(app, monkeypatch):
    """Empty body: no dangling 'reason' — just the clean status message."""
    from app.services import vision_ollama
    monkeypatch.setattr(vision_ollama.requests, 'post',
                        lambda *a, **k: _ErrResp(500, None, text='', json_error=True))
    with app.app_context(), pytest.raises(RuntimeError) as ei:
        vision_ollama.describe_image_ollama(_png(), 'p', auto_start_local=True)
    msg = str(ei.value)
    assert msg.strip().endswith('rejected the request (HTTP 500)')


def test_describe_image_ollama_best_effort_logs_reason_and_returns_empty(app, monkeypatch, caplog):
    """Ordinary best-effort call keeps the "" contract on a 400, but the concrete
    reason now reaches the log (previously only the opaque status code did)."""
    from app.services import vision_ollama
    monkeypatch.setattr(vision_ollama.requests, 'post',
                        lambda *a, **k: _ErrResp(400, {'error': 'this model does not support images'}))
    with app.app_context():
        with caplog.at_level('WARNING'):
            out = vision_ollama.describe_image_ollama(_png(), 'p')  # auto_start_local=False
    assert out == ''
    assert 'this model does not support images' in caplog.text


def test_caption_images_auto_reports_both_backend_failures(app, monkeypatch):
    """Regression for issue #6: in 'auto', JoyCaption unavailable + Ollama rejecting
    the request must surface BOTH reasons — the user otherwise sees only the Ollama
    error and never learns JoyCaption's deps are missing."""
    from app.services import face_dataset_service as svc
    from app.services import vision_ollama
    import app.services.joycaption as jc_mod
    from app.config import LOCAL_USER, save_config

    monkeypatch.setattr(jc_mod, 'is_available', lambda: False)
    monkeypatch.setattr(jc_mod, 'availability',
                        lambda: {'ok': False,
                                 'detail': 'transformers not importable in the ai-toolkit venv'})
    monkeypatch.setattr(vision_ollama.requests, 'post',
                        lambda *a, **k: _ErrResp(400, {'error': 'model does not support images'}))
    with app.app_context():
        save_config({'captioning': {'backend': 'auto'}})
        ds, _ = _dataset_with_kept_image(svc, LOCAL_USER)
        with pytest.raises(RuntimeError) as ei:
            svc.caption_images(LOCAL_USER, ds.id)
    msg = str(ei.value)
    assert 'JoyCaption unavailable' in msg and 'transformers' in msg
    assert 'model does not support images' in msg and 'HTTP 400' in msg


def test_vision_ollama_never_raises_when_config_returns_none(app, monkeypatch):
    """cfg.get() returning None for every key (missing/corrupted config section)
    must never surface as an AttributeError from the url.rstrip('/') call --
    the never-raise contract is structural, not dependent on config health.
    requests.post is also stubbed to fail so the test doesn't depend on whether
    a real Ollama happens to be reachable on this machine."""
    from app.services import vision_ollama

    def _raise(*a, **k):
        raise ConnectionError('ollama unreachable')

    monkeypatch.setattr(vision_ollama.cfg, 'get', lambda *a, **k: None)
    monkeypatch.setattr(vision_ollama.requests, 'post', _raise)
    with app.app_context():
        assert vision_ollama.describe_image_ollama(b'x', 'p') == ''
        assert vision_ollama.unload_vision_model() is False


# --- caption_images backend selection ---------------------------------------

def test_caption_images_backend_none_raises(app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER, save_config
    with app.app_context():
        save_config({'captioning': {'backend': 'none'}})
        ds, _ = _dataset_with_kept_image(svc, LOCAL_USER)
        with pytest.raises(RuntimeError):
            svc.caption_images(LOCAL_USER, ds.id)


def test_caption_images_stores_long_caption_in_full(app, monkeypatch):
    """A long descriptive caption is stored WHOLE — the legacy 800-char slice that cut
    captions mid-sentence (« …a pale, neutral tone, and a ») is gone. Descriptive
    captions are wanted for FLUX/Klein and the backend no longer truncates at 800."""
    from app.services import face_dataset_service as svc
    from app.services import vision_ollama
    from app.config import LOCAL_USER, save_config
    from app.models import FaceDatasetImage

    long_caption = 'a caption word ' * 100  # 1499 chars once stripped, well past the old 800
    assert len(long_caption.strip()) > 800
    monkeypatch.setattr(vision_ollama.requests, 'post',
                        lambda *a, **k: _Resp({'response': long_caption}))
    with app.app_context():
        save_config({'captioning': {'backend': 'ollama'}})
        ds, img = _dataset_with_kept_image(svc, LOCAL_USER)
        n = svc.caption_images(LOCAL_USER, ds.id)
        assert n == 1
        refreshed = svc.db.session.get(FaceDatasetImage, img.id)
        # Full text survives (this leak-free prose passes the cleaner unchanged): no 800 cut.
        assert len(refreshed.caption) > 800
        assert refreshed.caption.endswith('word')


def test_cap_caption_leaves_normal_captions_untouched():
    from app.services import face_dataset_service as svc
    text = 'A calm portrait. Soft light. A pale, neutral tone, and a quiet mood.'
    assert svc._cap_caption(text) == text
    assert svc._cap_caption('  trimmed  ') == 'trimmed'
    assert svc._cap_caption('') == ''
    assert svc._cap_caption(None) == ''


def test_cap_caption_guardrail_cuts_on_sentence_boundary():
    """When the (very high) safety ceiling is actually hit, the cut lands on a sentence
    end — never mid-word — so a stored caption never ends like « …and a »."""
    from app.services import face_dataset_service as svc
    ceiling = svc.CAPTION_MAX_CHARS
    body = 'The subject stands in a bright room. ' * ((ceiling // 37) + 5)
    assert len(body) > ceiling
    out = svc._cap_caption(body)
    assert len(out) <= ceiling
    assert out.endswith('room.')     # whole sentence kept, not a mid-word slice
    assert not out.endswith('roo')


def test_cap_caption_guardrail_falls_back_to_word_boundary():
    """A run-on past the ceiling with no sentence break still never cuts mid-word."""
    from app.services import face_dataset_service as svc
    ceiling = svc.CAPTION_MAX_CHARS
    body = 'word ' * ((ceiling // 5) + 20)  # over the ceiling, no . ! ?
    out = svc._cap_caption(body)
    assert len(out) <= ceiling
    assert out.endswith('word')      # last WHOLE word, never a partial token


def test_caption_images_allows_slow_local_inference(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import vision_ollama
    from app.config import LOCAL_USER, save_config

    seen = []

    def _describe(*args, **kwargs):
        seen.append(kwargs)
        return 'a caption'

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', _describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda **kwargs: True)
    with app.app_context():
        save_config({'captioning': {'backend': 'ollama'}})
        ds, _ = _dataset_with_kept_image(svc, LOCAL_USER)
        assert svc.caption_images(LOCAL_USER, ds.id) == 1
    assert seen[0]['timeout'] == (10, 300)
    assert seen[0]['auto_start_local'] is True


def test_caption_images_backend_ollama_never_touches_joycaption(app, monkeypatch):
    """backend='ollama' skips JoyCaption entirely -- the lazy `from .joycaption
    import ...` inside caption_images must never even execute in this mode."""
    from app.services import face_dataset_service as svc
    from app.services import vision_ollama
    import app.services.joycaption as jc_mod
    from app.config import LOCAL_USER, save_config

    def _boom(*a, **k):
        raise AssertionError('joycaption must not be called in ollama-only mode')

    monkeypatch.setattr(jc_mod, 'is_available', _boom)
    monkeypatch.setattr(jc_mod, 'caption_images_joycaption', _boom)
    monkeypatch.setattr(vision_ollama.requests, 'post',
                        lambda *a, **k: _Resp({'response': 'a caption'}))
    with app.app_context():
        save_config({'captioning': {'backend': 'ollama'}})
        ds, _ = _dataset_with_kept_image(svc, LOCAL_USER)
        n = svc.caption_images(LOCAL_USER, ds.id)
        assert n == 1


def test_caption_images_backend_joycaption_skips_ollama_fallback(app, monkeypatch):
    """backend='joycaption' must never fall back to Ollama, even for images
    JoyCaption didn't caption."""
    from app.services import face_dataset_service as svc
    from app.services import vision_ollama
    import app.services.joycaption as jc_mod
    from app.config import LOCAL_USER, save_config
    from app.models import FaceDatasetImage

    def _boom(*a, **k):
        raise AssertionError('ollama must not be called in joycaption-only mode')

    monkeypatch.setattr(vision_ollama.requests, 'post', _boom)
    with app.app_context():
        save_config({'captioning': {'backend': 'joycaption'}})
        ds, img = _dataset_with_kept_image(svc, LOCAL_USER)
        path = svc._img_path(img)
        monkeypatch.setattr(jc_mod, 'is_available', lambda: True)
        monkeypatch.setattr(jc_mod, 'caption_images_joycaption',
                            lambda paths, prompt=None, max_tokens=300, timeout=1800, **kw: {path: 'a joycaption result'})
        n = svc.caption_images(LOCAL_USER, ds.id)
        assert n == 1
        refreshed = svc.db.session.get(FaceDatasetImage, img.id)
        assert refreshed.caption == 'a joycaption result'


def test_caption_images_exposes_joycaption_stage_while_batch_runs(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import dataset_activity as da
    import app.services.joycaption as jc_mod
    from app.config import LOCAL_USER, save_config

    seen = []
    with app.app_context():
        save_config({'captioning': {'backend': 'joycaption'}})
        ds, img = _dataset_with_kept_image(svc, LOCAL_USER)
        path = svc._img_path(img)
        monkeypatch.setattr(jc_mod, 'is_available', lambda: True)

        def _caption(paths, **kwargs):
            seen.append(da.get(ds.id))
            return {path: 'a joycaption result'}

        monkeypatch.setattr(jc_mod, 'caption_images_joycaption', _caption)
        assert svc.caption_images(LOCAL_USER, ds.id) == 1

    assert seen and seen[0]['kind'] == 'caption'
    assert seen[0]['done'] == 0 and seen[0]['total'] == 1
    assert 'JoyCaption' in seen[0]['detail']


def test_caption_images_backend_joycaption_unavailable_raises(app, monkeypatch):
    """backend='joycaption' is an explicit user choice in Settings -- if the
    ai-toolkit venv isn't there (is_available() False), the caller must get a
    clear error, not a silent 0 (only 'auto' is allowed to fall back quietly)."""
    from app.services import face_dataset_service as svc
    import app.services.joycaption as jc_mod
    from app.config import LOCAL_USER, save_config

    monkeypatch.setattr(jc_mod, 'is_available', lambda: False)
    with app.app_context():
        save_config({'captioning': {'backend': 'joycaption'}})
        ds, _ = _dataset_with_kept_image(svc, LOCAL_USER)
        with pytest.raises(RuntimeError):
            svc.caption_images(LOCAL_USER, ds.id)


def _dataset_with_two_kept_images(svc, LOCAL_USER):
    """A FaceDataset with two kept images, each already carrying a (leaking) caption —
    the state the Identity-leak panel targets when it re-captions a single row."""
    from app.models import FaceDatasetImage
    ds = svc.create_dataset(LOCAL_USER, 'SubsetTest', 'subset')
    d = svc._dataset_dir(ds.id)
    os.makedirs(d, exist_ok=True)
    imgs = []
    for i, color in enumerate([(255, 0, 0), (0, 0, 255)]):
        fn = f'kept{i}.webp'
        with open(os.path.join(d, fn), 'wb') as fh:
            fh.write(_png(color))
        img = FaceDatasetImage(dataset_id=ds.id, source='import', status='keep',
                               filename=fn, framing='face', caption='old leaking caption')
        svc.db.session.add(img)
        imgs.append(img)
    svc.db.session.commit()
    return ds, imgs


def test_caption_images_scopes_to_image_ids_subset(app, monkeypatch):
    """The Identity-leak panel's targeted 🔄 Re-caption re-writes ONE image, leaving the
    rest of the dataset untouched — even though both already have captions."""
    from app.services import face_dataset_service as svc
    from app.services import vision_ollama
    from app.config import LOCAL_USER, save_config
    from app.models import FaceDatasetImage

    monkeypatch.setattr(vision_ollama.requests, 'post',
                        lambda *a, **k: _Resp({'response': 'fresh caption'}))
    with app.app_context():
        save_config({'captioning': {'backend': 'ollama'}})
        ds, (img_a, img_b) = _dataset_with_two_kept_images(svc, LOCAL_USER)
        n = svc.caption_images(LOCAL_USER, ds.id, force=True, image_ids=[img_a.id])
        assert n == 1
        assert svc.db.session.get(FaceDatasetImage, img_a.id).caption == 'fresh caption'
        # The image OUTSIDE the subset keeps its original caption verbatim.
        assert svc.db.session.get(FaceDatasetImage, img_b.id).caption == 'old leaking caption'


def test_caption_images_empty_subset_captions_nothing(app, monkeypatch):
    """An empty subset must short-circuit to 0 — never fall through to captioning the
    whole dataset. The vision backend is stubbed to blow up if it is ever reached."""
    from app.services import face_dataset_service as svc
    from app.services import vision_ollama
    from app.config import LOCAL_USER, save_config
    from app.models import FaceDatasetImage

    def _boom(*a, **k):
        raise AssertionError('an empty subset must not caption any image')

    monkeypatch.setattr(vision_ollama.requests, 'post', _boom)
    with app.app_context():
        save_config({'captioning': {'backend': 'ollama'}})
        ds, (img_a, img_b) = _dataset_with_two_kept_images(svc, LOCAL_USER)
        assert svc.caption_images(LOCAL_USER, ds.id, force=True, image_ids=[]) == 0
        assert svc.db.session.get(FaceDatasetImage, img_a.id).caption == 'old leaking caption'
        assert svc.db.session.get(FaceDatasetImage, img_b.id).caption == 'old leaking caption'


def test_caption_route_scopes_and_implies_force(client, app, monkeypatch):
    """{image_ids:[...]} reaches the service scoped, and a targeted call OVERWRITES —
    the route forces even when the body omits force (the leaking captions exist already)."""
    from app.services import face_dataset_service as svc
    seen = {}

    def _fake(user, dataset_id, force=False, mode=None, image_ids=None):
        seen.update(force=force, mode=mode, image_ids=image_ids)
        return len(image_ids or [])

    ds_id = client.post('/api/dataset/create',
                        json={'name': 'R', 'trigger_word': 'r'}).get_json()['id']
    monkeypatch.setattr(svc, 'caption_images', _fake)
    resp = client.post(f'/api/dataset/{ds_id}/caption', json={'image_ids': [4, 7]})
    assert resp.status_code == 200 and resp.get_json()['captioned'] == 2
    assert seen == {'force': True, 'mode': None, 'image_ids': [4, 7]}


def test_caption_route_rejects_non_list_image_ids(client, app):
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'R', 'trigger_word': 'r'}).get_json()['id']
    resp = client.post(f'/api/dataset/{ds_id}/caption', json={'image_ids': '4'})
    assert resp.status_code == 400 and 'image_ids' in resp.get_json()['error']


def test_caption_route_subset_503_while_vision_running(client, app):
    """A targeted re-caption is serialized against any other vision/training pass by the
    same GPU window as the batch — it returns 503 GPU busy rather than running concurrently."""
    from app.job_queue import queue_manager
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'R', 'trigger_word': 'r'}).get_json()['id']
    with app.app_context():
        queue_manager._set_system_state('vision_in_progress', True, ttl_seconds=300)
    resp = client.post(f'/api/dataset/{ds_id}/caption', json={'image_ids': [1]})
    assert resp.status_code == 503 and 'GPU busy' in resp.get_json()['error']


def test_caption_images_backend_auto_falls_back_to_ollama(app, monkeypatch):
    """backend='auto' (SRC/default behavior): JoyCaption tried first, Ollama
    fills in whatever it missed."""
    from app.services import face_dataset_service as svc
    from app.services import vision_ollama
    import app.services.joycaption as jc_mod
    from app.config import LOCAL_USER, save_config
    from app.models import FaceDatasetImage

    monkeypatch.setattr(jc_mod, 'is_available', lambda: True)
    monkeypatch.setattr(jc_mod, 'caption_images_joycaption',
                        lambda paths, prompt=None, max_tokens=300, timeout=1800, **kw: {})  # misses everything
    monkeypatch.setattr(vision_ollama.requests, 'post',
                        lambda *a, **k: _Resp({'response': 'ollama caption'}))
    with app.app_context():
        save_config({'captioning': {'backend': 'auto'}})
        ds, img = _dataset_with_kept_image(svc, LOCAL_USER)
        n = svc.caption_images(LOCAL_USER, ds.id)
        assert n == 1
        refreshed = svc.db.session.get(FaceDatasetImage, img.id)
        assert refreshed.caption == 'ollama caption'


# --- Problem A (issue #6): WebP -> JPEG re-encode at the Ollama seam ---------
# Ollama's server-side image decoder (stb_image on llama.cpp GGUF runners; Go's
# image.Decode on the native engine) can't read WebP on many builds, so a WebP payload
# fails with HTTP 400 "Failed to load image or audio file". describe_image_ollama now
# re-encodes anything that isn't already JPEG/PNG to JPEG before base64.

def test_describe_image_ollama_reencodes_webp_payload_to_jpeg(app, monkeypatch):
    """The bytes that actually leave for Ollama must be JPEG when the source is WebP —
    proving the fix works regardless of the Ollama/runner version."""
    from app.services import vision_ollama
    cap = _CapturePost()
    monkeypatch.setattr(vision_ollama.requests, 'post', cap)
    with app.app_context():
        vision_ollama.describe_image_ollama(_webp(), 'describe this')
    sent = cap.sent_image_bytes()
    assert sent[:3] == b'\xff\xd8\xff', 'payload image should be JPEG, not WebP'
    # And it must be a valid, decodable JPEG (not a corrupted re-wrap).
    with Image.open(io.BytesIO(sent)) as im:
        assert im.format == 'JPEG'


def test_describe_image_ollama_reencodes_webp_with_alpha_flattens_on_white(app, monkeypatch):
    """A WebP with alpha must flatten to opaque RGB JPEG (no alpha channel that a JPEG
    can't hold, no crash)."""
    from app.services import vision_ollama
    cap = _CapturePost()
    monkeypatch.setattr(vision_ollama.requests, 'post', cap)
    with app.app_context():
        vision_ollama.describe_image_ollama(_webp(mode='RGBA', color=(0, 200, 0, 128)),
                                            'describe this')
    sent = cap.sent_image_bytes()
    assert sent[:3] == b'\xff\xd8\xff'
    with Image.open(io.BytesIO(sent)) as im:
        assert im.mode == 'RGB'


def test_describe_image_ollama_passes_jpeg_through_unchanged(app, monkeypatch):
    """A source already JPEG is forwarded byte-for-byte — a batch of hundreds must not
    pay for a needless re-encode."""
    from app.services import vision_ollama
    src = _jpeg()
    cap = _CapturePost()
    monkeypatch.setattr(vision_ollama.requests, 'post', cap)
    with app.app_context():
        vision_ollama.describe_image_ollama(src, 'describe this')
    assert cap.sent_image_bytes() == src


def test_describe_image_ollama_passes_png_through_unchanged(app, monkeypatch):
    """PNG is decodable everywhere too, so it is also forwarded untouched."""
    from app.services import vision_ollama
    src = _png()
    cap = _CapturePost()
    monkeypatch.setattr(vision_ollama.requests, 'post', cap)
    with app.app_context():
        vision_ollama.describe_image_ollama(src, 'describe this')
    assert cap.sent_image_bytes() == src


def test_ensure_ollama_decodable_returns_original_on_non_image(app):
    """A non-image blob can't be re-encoded — it is returned unchanged so the caller
    still surfaces Ollama's own error instead of crashing the never-raise contract."""
    from app.services import vision_ollama
    with app.app_context():
        assert vision_ollama._ensure_ollama_decodable(b'not an image') == b'not an image'


# --- Problem B (issue #6): JoyCaption live stderr streaming ------------------
# The first run silently downloads the ~7 GB model; capture_output=True hid all stderr
# until the process ended, so the app log looked frozen. It is now streamed line-by-line.

class _FakeStdin:
    def write(self, data):
        return len(data)

    def close(self):
        pass


class _FakeStdout:
    def __init__(self, text):
        self._text = text

    def read(self):
        return self._text


class _FakePopen:
    """Minimal Popen stand-in: hands the drain threads pre-buffered stdout/stderr and a
    wait() that can raise TimeoutExpired exactly once (a killed process reaps cleanly)."""
    def __init__(self, stderr_lines, stdout_text, returncode=0, wait_raises=None):
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(stdout_text)
        self.stderr = iter(stderr_lines)
        self.returncode = returncode
        self._wait_raises = wait_raises
        self._waited = False
        self.killed = False

    def wait(self, timeout=None):
        if self._wait_raises is not None and not self._waited:
            self._waited = True
            raise self._wait_raises
        return self.returncode

    def kill(self):
        self.killed = True


def _prep_joycaption(monkeypatch, jc, tmp_path):
    img = tmp_path / 'a.webp'
    img.write_bytes(_webp())
    monkeypatch.setattr(jc, 'is_available', lambda: True)
    monkeypatch.setattr(jc.cfg, 'aitoolkit_path', lambda k: str(tmp_path / str(k)))
    return str(img)


def test_joycaption_streams_stderr_to_log_live(app, monkeypatch, caplog, tmp_path):
    import app.services.joycaption as jc
    img = _prep_joycaption(monkeypatch, jc, tmp_path)
    stderr_lines = [
        '[joycaption] first run: downloading model (~7 GB) from Hugging Face …\n',
        '[joycaption] model loaded\n',
        '[joycaption] 1/1 ok (42 chars)\n',
    ]
    stdout_text = json.dumps({'captions': {img: 'a caption'}, 'errors': {}}) + '\n'
    monkeypatch.setattr(jc.subprocess, 'Popen',
                        lambda *a, **k: _FakePopen(stderr_lines, stdout_text))
    with app.app_context():
        with caplog.at_level('INFO'):
            out = jc.caption_images_joycaption([img])
    assert out == {img: 'a caption'}
    # Every subprocess stderr line reached the app log live (not only at the end).
    assert 'first run: downloading model' in caplog.text
    assert 'model loaded' in caplog.text


def test_joycaption_reflects_stage_markers_into_activity(app, monkeypatch, tmp_path):
    import app.services.joycaption as jc
    from app.services import dataset_activity as da
    img = _prep_joycaption(monkeypatch, jc, tmp_path)
    seen = []
    real_progress = da.progress

    def _spy(token, **kw):
        if kw.get('detail'):
            seen.append(kw['detail'])
        return real_progress(token, **kw)

    monkeypatch.setattr(da, 'progress', _spy)
    stderr_lines = [
        '[joycaption] first run: downloading model (~7 GB) …\n',
        'Downloading shards:  40%|####      | 2/5\n',   # raw HF tqdm -> log only
        '[joycaption] model loaded\n',
    ]
    stdout_text = json.dumps({'captions': {img: 'cap'}, 'errors': {}})
    monkeypatch.setattr(jc.subprocess, 'Popen',
                        lambda *a, **k: _FakePopen(stderr_lines, stdout_text))
    with app.app_context():
        da.reset()
        token = da.begin(1, 'caption', total=1)
        out = jc.caption_images_joycaption([img], activity_token=token)
    assert out == {img: 'cap'}
    # The controlled [joycaption] markers were reflected into the UI stage…
    assert any('first run: downloading' in d for d in seen)
    assert any('model loaded' in d for d in seen)
    # …but the noisy raw tqdm line was NOT (log only), keeping the UI detail readable.
    assert not any('Downloading shards' in d for d in seen)


def test_joycaption_timeout_returns_empty_and_logs_first_run_hint(app, monkeypatch, caplog, tmp_path):
    import app.services.joycaption as jc
    img = _prep_joycaption(monkeypatch, jc, tmp_path)
    stderr_lines = ['[joycaption] first run: downloading model (~7 GB) …\n']
    fake = _FakePopen(stderr_lines, '',
                      wait_raises=subprocess.TimeoutExpired(cmd='joycaption', timeout=1))
    monkeypatch.setattr(jc.subprocess, 'Popen', lambda *a, **k: fake)
    with app.app_context():
        with caplog.at_level('ERROR'):
            out = jc.caption_images_joycaption([img], timeout=1)
    assert out == {}
    assert fake.killed, 'a timed-out subprocess must be killed'
    # The timeout message explains the first-run download so the user knows to re-run.
    assert '7 GB' in caplog.text and 'resume' in caplog.text.lower()
