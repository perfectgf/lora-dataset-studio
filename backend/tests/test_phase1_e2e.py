"""Phase-1 end-to-end acceptance path: create -> upload ref -> generate (API
engine, mocked) -> curate (keep + caption) -> export ZIP. Exercises the same
HTTP surface the lifted frontend (Task 11) drives, with the fan-out's
background thread stubbed the same way test_dataset_service.py does (never
race a real thread against test teardown)."""
import io
import zipfile
from unittest.mock import patch

from PIL import Image


def _png():
    buf = io.BytesIO(); Image.new('RGB', (64, 64), (1, 2, 3)).save(buf, 'PNG')
    return buf.getvalue()


class _SyncExecutor:
    """Drop-in for ThreadPoolExecutor that runs `map` inline, same thread.

    `_run_nanobanana_batch` fans its per-image work out over a real
    ThreadPoolExecutor; each worker thread would open its own connection to
    the test's sqlite `:memory:` database, which (absent a StaticPool, not
    configured here) is a SEPARATE, empty database per connection -- rows
    committed by the test's thread would be invisible to it. Running inline
    keeps everything on the one connection the test set up.
    """
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, iterable):
        return [fn(x) for x in iterable]


def test_api_only_end_to_end(client, app, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-x')

    ds = client.post('/api/dataset/create',
                     json={'name': 'E2E', 'trigger_word': 'e2e'}).get_json()
    did = ds['id']

    r = client.post(f'/api/dataset/{did}/ref',
                    data={'file': (io.BytesIO(_png()), 'ref.png')},
                    content_type='multipart/form-data')
    assert r.status_code == 200

    from app.services import face_dataset_service as svc
    from app.services.face_variations import select_preset
    from app.config import LOCAL_USER

    # Stub the background Thread (same technique as test_dataset_service.py's
    # test_api_fanout_creates_pending_rows) so the batch dispatch is captured
    # instead of actually starting a background thread. This monkeypatch is
    # NOT scoped to the dispatch call -- monkeypatch.setattr stays active for
    # the rest of the test, including the manual `_run_nanobanana_batch` call
    # below. What actually prevents a TypeError there is the `_SyncExecutor`
    # patch over `concurrent.futures.ThreadPoolExecutor`: without it,
    # `_run_nanobanana_batch`'s real ThreadPoolExecutor would call
    # `threading.Thread(..., name=...)` internally, which collides with this
    # stub's lambda signature. Removing the `_SyncExecutor` patch would
    # reintroduce that Thread-stub collision.
    calls = []
    monkeypatch.setattr(
        'app.services.face_dataset_service.threading.Thread',
        lambda target, args=(), daemon=True: type('T', (), {'start': lambda s: calls.append(args)})())

    with patch('app.services.chatgpt_image.generate_variation', return_value=_png()), \
         patch('concurrent.futures.ThreadPoolExecutor', _SyncExecutor):
        with app.app_context():
            ids = svc.generate_variations_nanobanana(app, LOCAL_USER, did,
                                                      select_preset('zimage_12')[:2], 1,
                                                      engine='chatgpt')
            assert len(ids) == 2
            assert calls  # background batch was dispatched (Thread stubbed)
            # Emulate thread completion synchronously: run the captured worker
            # body (its internal ThreadPoolExecutor is stubbed inline above).
            svc._run_nanobanana_batch(*calls[0])
            rows = svc.FaceDatasetImage.query.filter_by(dataset_id=did).all()
            assert len(rows) == 2
            assert all(row.filename for row in rows)  # generation actually "completed"

    payload = client.get(f'/api/dataset/{did}').get_json()
    assert len(payload['images']) == 2

    # Keep one, caption it manually, export.
    iid = payload['images'][0]['id']
    r = client.post(f'/api/dataset/image/{iid}/status', json={'status': 'keep'})
    assert r.status_code == 200
    r = client.post(f'/api/dataset/image/{iid}/caption', json={'caption': 'a portrait'})
    assert r.status_code == 200

    z = client.get(f'/api/dataset/{did}/export')
    assert z.status_code == 200 and z.mimetype == 'application/zip'
    zf = zipfile.ZipFile(io.BytesIO(z.data))
    names = zf.namelist()
    assert any(n.endswith('_000_ref.png') for n in names)  # reference kept as the real anchor
    caption_file = next(n for n in names if n.endswith('_001.txt'))
    assert zf.read(caption_file).decode('utf-8') == 'e2e, a portrait'
