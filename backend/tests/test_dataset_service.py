import io, zipfile
from PIL import Image


def _png(color=(255, 0, 0)):
    buf = io.BytesIO(); Image.new('RGB', (64, 64), color).save(buf, 'PNG')
    return buf.getvalue()


def test_create_and_payload(app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Lola', 'lola')
        p = svc.dataset_payload(LOCAL_USER, ds.id)
        # NB: the brief's snippet checked `p['comp']`, but dataset_payload's actual
        # key (SRC-identical) is 'composition' -- 'comp' does not exist and would
        # KeyError. Corrected here; see task-8-report.md.
        assert p['name'] == 'Lola' and p['composition'] == {'face': 0, 'bust': 0, 'body': 0, 'back': 0}


def test_api_fanout_creates_pending_rows(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    from app.services.face_variations import select_preset
    calls = []
    monkeypatch.setattr('app.services.face_dataset_service.threading.Thread',
                        lambda target, args=(), daemon=True: type('T', (), {'start': lambda s: calls.append(args)})())
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'A', 'a')
        # give it a reference so _all_ref_bytes works
        import os
        os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
        open(os.path.join(svc._dataset_dir(ds.id), 'ref.webp'), 'wb').write(_png())
        ds.ref_filename = 'ref.webp'
        svc.db.session.commit()
        svc.generate_variations_nanobanana(app, LOCAL_USER, ds.id,
                                           select_preset('zimage_12')[:2], 1, engine='chatgpt')
        rows = FaceDatasetImage.query.filter_by(dataset_id=ds.id).all()
        assert len(rows) == 2 and all(r.status == 'pending' and r.klein_model == 'chatgpt' for r in rows)
        assert calls  # background batch was dispatched


def test_export_zip_layout(app):
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    import os
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Zoe', 'zoe')
        d = svc._dataset_dir(ds.id); os.makedirs(d, exist_ok=True)
        open(os.path.join(d, 'ref.webp'), 'wb').write(_png()); ds.ref_filename = 'ref.webp'
        open(os.path.join(d, 'img1.webp'), 'wb').write(_png((0, 255, 0)))
        svc.db.session.add(FaceDatasetImage(dataset_id=ds.id, filename='img1.webp',
                                            status='keep', framing='face', caption='a smile'))
        svc.db.session.commit()
        z = zipfile.ZipFile(io.BytesIO(svc.build_export_zip(LOCAL_USER, ds.id)))
        names = z.namelist()
        assert any(n.endswith('_000_ref.png') for n in names)
        txt = [n for n in names if n.endswith('_001.txt')][0]
        assert z.read(txt).decode('utf-8').startswith('zoe, ')


def test_status_validation(app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'B', 'b')
        try:
            svc.set_image_status(LOCAL_USER, 99999, 'nonsense'); raised = False
        except Exception:
            raised = True
        assert raised


def test_import_images_normalizes_and_persists(app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'C', 'c')
        ids, failed = svc.import_images(LOCAL_USER, ds.id, [_png()], crop=False)
        assert len(ids) == 1 and failed == 0
        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        assert len(payload['images']) == 1
        assert payload['images'][0]['status'] == 'keep'


def test_delete_dataset_without_lora_training_module(app):
    """lora_training (Task 19) doesn't exist yet in phase 1 -> delete_dataset must
    still succeed (purge step is best-effort and silently skipped)."""
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'D', 'd')
        assert svc.delete_dataset(LOCAL_USER, ds.id) is True
        assert svc.get_dataset(LOCAL_USER, ds.id) is None


def test_classify_returns_zero_when_ollama_unreachable(app, monkeypatch):
    """vision_ollama (Task 9) now exists, so classify_images no longer hits the
    ImportError->RuntimeError path; describe_image_ollama's never-raise contract
    means an unreachable Ollama server just yields 0 classified (no exception).
    requests.post is stubbed so this test never touches a real Ollama server."""
    from app.services import face_dataset_service as svc
    from app.services import vision_ollama
    from app.config import LOCAL_USER

    def _raise(*a, **k):
        raise ConnectionError('ollama unreachable')

    monkeypatch.setattr(vision_ollama.requests, 'post', _raise)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'E', 'e')
        assert svc.classify_images(LOCAL_USER, ds.id) == 0


def test_detect_head_bbox_falls_back_to_none_when_ollama_unreachable(app, monkeypatch):
    """detect_head_bbox has an existing graceful fallback for 'no detection'
    (face_crop_to_square_webp centers the crop instead) -- an unreachable Ollama
    server must hit THAT path (return None), not raise. requests.post is stubbed
    so this test never touches a real Ollama server."""
    from app.services import face_dataset_service as svc
    from app.services import vision_ollama

    def _raise(*a, **k):
        raise ConnectionError('ollama unreachable')

    monkeypatch.setattr(vision_ollama.requests, 'post', _raise)
    with app.app_context():
        assert svc.detect_head_bbox(_png()) is None
        # face_crop_to_square_webp must still produce a valid centered-crop webp.
        out = svc.face_crop_to_square_webp(_png())
        assert isinstance(out, (bytes, bytearray)) and len(out) > 0


def test_generate_variations_klein_raises_runtime_error_when_unconfigured(app):
    """klein_edit_helper (Task 14) exists, so the fan-out actually reaches
    enqueue_klein_edit -- with no comfyui.base_dir configured, that raises
    RuntimeError('ComfyUI is not configured'). Needs a non-empty variations
    list (an empty one short-circuits the fan-out loop before ComfyUI is ever
    touched) and a reference image (checked before the fan-out starts)."""
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    import os
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'F', 'f')
        d = svc._dataset_dir(ds.id)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'ref.webp'), 'wb') as fh:
            fh.write(_png())
        ds.ref_filename = 'ref.webp'
        svc.db.session.commit()
        try:
            svc.generate_variations(LOCAL_USER, ds.id,
                                    [{'label': 'x', 'framing': 'face', 'prompt': 'p'}],
                                    1, 'some_klein_model')
            raised = False
        except RuntimeError as e:
            raised = 'ComfyUI is not configured' in str(e)
        assert raised


def test_link_completed_dataset_image_without_comfyui_configured(app):
    """comfyui.base_dir/output_dir are unset in phase-1 test config -> the
    completion link must mark the row failed instead of crashing (checklist item 3)."""
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'G', 'g')
        img = FaceDatasetImage(dataset_id=ds.id, source='generated', status='pending',
                               job_id='job-123', klein_model='some_klein_model')
        svc.db.session.add(img)
        svc.db.session.commit()
        svc.link_completed_dataset_image('job-123', 'result.webp', failed=False)
        refreshed = svc.db.session.get(FaceDatasetImage, img.id)
        assert refreshed.status == 'failed'
