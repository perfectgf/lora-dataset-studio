"""A Bank caption pass says how many images never reached the model.

Reported on Discord #help (2026-09-03): a Bank caption pass over a selection of
~30 images ended with "done — 1 captioned" and nothing else, while the log said
every image after the first had been refused by the local GPU fence.
`caption_paths` walks the batch through a pool that turns a per-image exception
into a log line and one step of the progress bar, so the pass's own report had
no way to know — and "done — 1 captioned" over a 30-image selection tells the
user the pass looked at them all and found 29 not worth a caption.

The refusal itself was the fence comparing model names byte-for-byte (see
test_ollama_gpu_fence_names.py). This file pins the other half: whatever refuses
an image, the finished line counts it and quotes the refusal's own sentence.
"""
import os

from PIL import Image

from app.services.ollama_gpu_fence import FENCE_BLOCKED_MESSAGE


def _images(folder, n):
    os.makedirs(folder, exist_ok=True)
    paths = []
    for i in range(n):
        p = os.path.join(folder, f'img{i:02d}.png')
        Image.new('RGB', (64, 64), (10 * i + 20, 90, 160)).save(p)
        paths.append(p)
    return paths


def _one_caption_then(monkeypatch, then):
    """Fake the Ollama vision seam: the first image gets a caption, every later
    one gets whatever ``then`` does (an exception to raise, or a string)."""
    from app.services import vision_ollama, vision_pool
    calls = []

    def describe(image_bytes, *a, **k):
        calls.append(1)
        if len(calls) == 1:
            return 'a plain wall'
        if isinstance(then, BaseException):
            raise then
        return then

    monkeypatch.setattr(vision_pool, 'vision_concurrency', lambda *a, **k: 1)
    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)
    return calls


# --- the shared brick -----------------------------------------------------------
def test_caption_paths_reports_the_images_the_fence_refused(app, tmp_path, monkeypatch):
    from app.services import face_dataset_service as fds
    from app.services.vision_ollama import LocalOllamaFenceError
    paths = _images(str(tmp_path / 'src'), 4)
    _one_caption_then(monkeypatch, LocalOllamaFenceError(FENCE_BLOCKED_MESSAGE))
    outcome = {}
    with app.app_context():
        out = fds.caption_paths(paths, backend='ollama', outcome=outcome)
    assert list(out.values()) == ['a plain wall']
    assert outcome['fenced'] == 3
    assert outcome['fence_reason'] == FENCE_BLOCKED_MESSAGE
    assert outcome['unanswered'] == 0 and outcome['failed'] == 0


def test_caption_paths_counts_the_images_the_model_answered_with_nothing(app, tmp_path,
                                                                            monkeypatch):
    """An empty answer is how a connection error or a 4xx reaches this brick
    (describe is best-effort past the first image), so it is counted apart."""
    from app.services import face_dataset_service as fds
    paths = _images(str(tmp_path / 'src'), 3)
    _one_caption_then(monkeypatch, '')
    outcome = {}
    with app.app_context():
        fds.caption_paths(paths, backend='ollama', outcome=outcome)
    assert outcome == {'fenced': 0, 'unanswered': 2, 'failed': 0, 'fence_reason': ''}


def test_caption_paths_costs_nothing_to_a_caller_that_did_not_ask(app, tmp_path, monkeypatch):
    from app.services import face_dataset_service as fds
    from app.services.vision_ollama import LocalOllamaFenceError
    paths = _images(str(tmp_path / 'src'), 2)
    _one_caption_then(monkeypatch, LocalOllamaFenceError(FENCE_BLOCKED_MESSAGE))
    with app.app_context():
        out = fds.caption_paths(paths, backend='ollama')
    assert list(out.values()) == ['a plain wall']


# --- the Bank pass, as the user reads it --------------------------------------------
def _use_ollama_backend(app):
    with app.app_context():
        import app.config as cfg
        cfg.save_config({'captioning': {'backend': 'ollama'}})


def _mkbank(client, tmp_path, n):
    src = tmp_path / 'src'
    _images(str(src), n)
    r = client.post('/api/bank/create', json={'name': 'B', 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id']


def test_bank_caption_pass_names_the_refused_images_in_its_last_line(client, tmp_path, app,
                                                                        monkeypatch):
    from app.services.vision_ollama import LocalOllamaFenceError
    _use_ollama_backend(app)
    bank_id = _mkbank(client, tmp_path, 4)
    _one_caption_then(monkeypatch, LocalOllamaFenceError(FENCE_BLOCKED_MESSAGE))

    r = client.post(f'/api/bank/{bank_id}/caption', json={})
    assert r.status_code == 202, r.get_json()
    activity = client.get(f'/api/bank/{bank_id}').get_json()['activity']
    assert activity['finished'] is True and activity['error'] is None
    detail = activity['detail']
    assert detail.startswith('done — 1 captioned'), detail
    # The refusal's own sentence, its closing period folded into the clause.
    reason = FENCE_BLOCKED_MESSAGE.rstrip('.')
    assert f'3 not captioned ({reason} — run the pass again to finish them)' in detail, detail


def test_bank_caption_pass_names_the_images_the_model_left_blank(client, tmp_path, app,
                                                                    monkeypatch):
    _use_ollama_backend(app)
    bank_id = _mkbank(client, tmp_path, 3)
    _one_caption_then(monkeypatch, '')

    r = client.post(f'/api/bank/{bank_id}/caption', json={})
    assert r.status_code == 202, r.get_json()
    detail = client.get(f'/api/bank/{bank_id}').get_json()['activity']['detail']
    assert detail.startswith('done — 1 captioned'), detail
    assert '2 not captioned (the vision model returned nothing' in detail, detail
