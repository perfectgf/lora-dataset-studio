import io
import os

import pytest
from PIL import Image


def _png(color=(255, 0, 0)):
    buf = io.BytesIO()
    Image.new('RGB', (64, 64), color).save(buf, 'PNG')
    return buf.getvalue()


class _Resp:
    """Minimal stand-in for requests.Response, only what describe_image_ollama reads."""
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


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


def test_caption_images_backend_ollama_writes_and_truncates(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import vision_ollama
    from app.config import LOCAL_USER, save_config
    from app.models import FaceDatasetImage

    long_caption = 'a caption word ' * 100  # well over CAPTION_MAX_CHARS (800)
    assert len(long_caption) > svc.CAPTION_MAX_CHARS
    monkeypatch.setattr(vision_ollama.requests, 'post',
                        lambda *a, **k: _Resp({'response': long_caption}))
    with app.app_context():
        save_config({'captioning': {'backend': 'ollama'}})
        ds, img = _dataset_with_kept_image(svc, LOCAL_USER)
        n = svc.caption_images(LOCAL_USER, ds.id)
        assert n == 1
        refreshed = svc.db.session.get(FaceDatasetImage, img.id)
        assert refreshed.caption
        assert len(refreshed.caption) <= svc.CAPTION_MAX_CHARS


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
                            lambda paths, prompt=None, max_tokens=300, timeout=1800: {path: 'a joycaption result'})
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
                        lambda paths, prompt=None, max_tokens=300, timeout=1800: {})  # misses everything
    monkeypatch.setattr(vision_ollama.requests, 'post',
                        lambda *a, **k: _Resp({'response': 'ollama caption'}))
    with app.app_context():
        save_config({'captioning': {'backend': 'auto'}})
        ds, img = _dataset_with_kept_image(svc, LOCAL_USER)
        n = svc.caption_images(LOCAL_USER, ds.id)
        assert n == 1
        refreshed = svc.db.session.get(FaceDatasetImage, img.id)
        assert refreshed.caption == 'ollama caption'
