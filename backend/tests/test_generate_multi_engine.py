"""Multi-engine generation: /dataset/<id>/generate with `engine_batches`.

The workspace can now run one batch per selected engine (shots shared between
them, or every shot on every engine). The split itself is computed client side —
so what MUST be verified here is that the route re-checks every entry rather
than trusting the payload, and that a refusal never leaves a half-dispatched run.

Every API batch's background Thread is stubbed out (same technique as
test_dataset_routes.test_generate_chatgpt_no_key_accepts_and_creates_pending_rows)
so no test ever fires a real, billed request.
"""
import io

import pytest
from PIL import Image

from app.services import face_dataset_service as svc


def _png_bytes(color=(255, 0, 0)):
    buf = io.BytesIO(); Image.new('RGB', (64, 64), color).save(buf, 'PNG')
    return buf.getvalue()


def _create(client, name='Lola', trigger='lola'):
    return client.post('/api/dataset/create', json={'name': name, 'trigger_word': trigger})


@pytest.fixture
def no_threads(monkeypatch):
    """Capture the API batch dispatches instead of running them."""
    calls = []
    monkeypatch.setattr(
        'app.services.face_dataset_service.threading.Thread',
        lambda target, args=(), daemon=True: type('T', (), {'start': lambda s: calls.append(args)})())
    return calls


def _dataset_with_ref(client, name='Iris'):
    ds_id = _create(client, name, name.lower()).get_json()['id']
    client.post(f'/api/dataset/{ds_id}/ref',
                data={'file': (io.BytesIO(_png_bytes()), 'ref.png')},
                content_type='multipart/form-data')
    return ds_id


def _shots(n, prefix='Shot'):
    return [{'label': f'{prefix} {i}', 'framing': 'face', 'prompt': f'prompt {i}'}
            for i in range(n)]


def test_two_api_engines_split_the_shots(client, no_threads):
    """The headline case: half the shots on Nano Banana, half on ChatGPT, one
    request, and each row carries the engine that made it."""
    ds_id = _dataset_with_ref(client)
    resp = client.post(f'/api/dataset/{ds_id}/generate', json={
        'engine_batches': [
            {'generator': 'nanobanana', 'variations': _shots(2, 'NB')},
            {'generator': 'chatgpt', 'variations': _shots(3, 'GPT')},
        ],
        'multiplier': 1,
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] is True
    assert body['created'] == 5
    assert body['per_engine'] == {'nanobanana': 2, 'chatgpt': 3}
    assert len(no_threads) == 2          # one background batch per engine

    payload = client.get(f'/api/dataset/{ds_id}').get_json()
    engines = sorted(i['engine'] for i in payload['images'])
    assert engines == ['chatgpt'] * 3 + ['nanobanana'] * 2


def test_all_engines_mode_sends_every_shot_to_every_engine(client, no_threads):
    ds_id = _dataset_with_ref(client)
    shots = _shots(4)
    resp = client.post(f'/api/dataset/{ds_id}/generate', json={
        'engine_batches': [
            {'generator': 'nanobanana', 'variations': shots},
            {'generator': 'chatgpt', 'variations': shots},
        ],
        'multiplier': 1,
    })
    assert resp.status_code == 200
    assert resp.get_json()['per_engine'] == {'nanobanana': 4, 'chatgpt': 4}


def test_multiplier_applies_to_every_batch(client, no_threads):
    ds_id = _dataset_with_ref(client)
    resp = client.post(f'/api/dataset/{ds_id}/generate', json={
        'engine_batches': [
            {'generator': 'nanobanana', 'variations': _shots(2)},
            {'generator': 'chatgpt', 'variations': _shots(2)},
        ],
        'multiplier': 3,
    })
    assert resp.status_code == 200
    assert resp.get_json()['created'] == 12


def test_unknown_engine_in_any_entry_is_refused_and_nothing_is_created(client, no_threads):
    """The guard applies to EVERY entry, not just the first one — an unknown
    engine hiding behind a valid one must not ride along."""
    ds_id = _dataset_with_ref(client)
    resp = client.post(f'/api/dataset/{ds_id}/generate', json={
        'engine_batches': [
            {'generator': 'nanobanana', 'variations': _shots(2)},
            {'generator': 'midjourney', 'variations': _shots(2)},
        ],
        'multiplier': 1,
    })
    assert resp.status_code == 400
    assert 'midjourney' in resp.get_json()['error']
    assert not no_threads                                    # nothing dispatched
    assert client.get(f'/api/dataset/{ds_id}').get_json()['images'] == []


def test_nsfw_shot_on_a_later_api_entry_refuses_the_whole_run(client, no_threads):
    """NSFW rides the local Klein path only. A run whose SECOND entry smuggles an
    NSFW shot onto an API engine is refused whole — the first, valid entry must
    not already be in flight."""
    ds_id = _dataset_with_ref(client)
    resp = client.post(f'/api/dataset/{ds_id}/generate', json={
        'engine_batches': [
            {'generator': 'nanobanana', 'variations': _shots(2)},
            {'generator': 'chatgpt',
             'variations': [{'label': 'hot', 'framing': 'body', 'prompt': 'p', 'nsfw': True}]},
        ],
        'multiplier': 1,
    })
    assert resp.status_code == 400
    assert 'Klein' in resp.get_json()['error']
    assert not no_threads
    assert client.get(f'/api/dataset/{ds_id}').get_json()['images'] == []


def test_nsfw_detected_by_label_too(client, no_threads):
    """A 🔞 custom card carries no `nsfw` flag of its own — the label prefix is
    what marks it, and the per-entry guard must read it."""
    ds_id = _dataset_with_ref(client)
    resp = client.post(f'/api/dataset/{ds_id}/generate', json={
        'engine_batches': [
            {'generator': 'chatgpt',
             'variations': [{'label': '🔞 topless on the couch', 'framing': 'body', 'prompt': 'p'}]},
        ],
    })
    assert resp.status_code == 400
    assert not no_threads


def test_klein_preflight_refuses_before_the_api_batches_are_dispatched(client, no_threads):
    """Mixed run on a machine without ComfyUI: the Klein half is impossible, so
    the whole request 409s. The API half must NOT have been queued already —
    that would bill for images belonging to a batch the user was told failed."""
    ds_id = _dataset_with_ref(client)
    resp = client.post(f'/api/dataset/{ds_id}/generate', json={
        'engine_batches': [
            {'generator': 'nanobanana', 'variations': _shots(2)},
            {'generator': 'klein', 'variations': _shots(2)},
        ],
        'multiplier': 1,
        'klein_model': 'some_model',
    })
    assert resp.status_code == 409
    assert not no_threads
    assert client.get(f'/api/dataset/{ds_id}').get_json()['images'] == []


def test_aggregate_fanout_cap_refuses_the_whole_run(client, no_threads):
    """MAX_FANOUT is a per-batch cap: three 25-image entries each pass on their
    own while the run totals 75. The aggregate check refuses up front instead of
    creating rows for the first entries and failing on the last."""
    ds_id = _dataset_with_ref(client)
    over = svc.MAX_FANOUT // 2 + 1
    resp = client.post(f'/api/dataset/{ds_id}/generate', json={
        'engine_batches': [
            {'generator': 'nanobanana', 'variations': _shots(over)},
            {'generator': 'chatgpt', 'variations': _shots(over)},
        ],
        'multiplier': 1,
    })
    assert resp.status_code == 400
    assert 'fan-out too large' in resp.get_json()['error']
    assert not no_threads
    assert client.get(f'/api/dataset/{ds_id}').get_json()['images'] == []


def test_in_flight_generations_count_against_the_budget(client, no_threads):
    """A second run must see the first one's still-pending rows."""
    ds_id = _dataset_with_ref(client)
    half = svc.MAX_FANOUT // 2
    first = client.post(f'/api/dataset/{ds_id}/generate', json={
        'engine_batches': [{'generator': 'chatgpt', 'variations': _shots(half)}]})
    assert first.status_code == 200
    second = client.post(f'/api/dataset/{ds_id}/generate', json={
        'engine_batches': [{'generator': 'nanobanana', 'variations': _shots(half + 1)}]})
    assert second.status_code == 400
    assert 'in flight' in second.get_json()['error']


def test_empty_engine_batches_is_a_clean_400(client, no_threads):
    """Zero engines selected must never queue a silent empty batch."""
    ds_id = _dataset_with_ref(client)
    for payload in ([], [{'generator': 'chatgpt', 'variations': []}]):
        resp = client.post(f'/api/dataset/{ds_id}/generate',
                           json={'engine_batches': payload})
        assert resp.status_code == 400
    assert not no_threads


def test_malformed_engine_batches_is_a_400_not_a_500(client, no_threads):
    ds_id = _dataset_with_ref(client)
    for payload in ('chatgpt', ['chatgpt'], [{'generator': 'chatgpt', 'variations': 'nope'}]):
        resp = client.post(f'/api/dataset/{ds_id}/generate',
                           json={'engine_batches': payload})
        assert resp.status_code == 400, payload
    assert not no_threads


def test_legacy_single_generator_payload_is_unchanged(client, no_threads):
    """An old tab that was never reloaded keeps posting `generator` +
    `variations`. That shape must behave exactly as before."""
    ds_id = _dataset_with_ref(client)
    resp = client.post(f'/api/dataset/{ds_id}/generate', json={
        'generator': 'chatgpt',
        'variations': _shots(2),
        'multiplier': 1,
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['created'] == 2
    assert body['per_engine'] == {'chatgpt': 2}
    assert len(no_threads) == 1


def test_image_engine_is_absent_rather_than_wrong(client, no_threads):
    """Rows that cannot name their engine (imports, pre-feature generations)
    report None — the tile then shows no badge instead of a false one."""
    ds_id = _dataset_with_ref(client)
    client.post(f'/api/dataset/{ds_id}/generate', json={
        'engine_batches': [{'generator': 'chatgpt', 'variations': _shots(1)}]})
    # The DB query needs an app context of its own — relying on one leaking from
    # a prior test made this pass locally but fail under CI's ordering.
    with client.application.app_context():
        img = svc.FaceDatasetImage.query.filter_by(dataset_id=ds_id).first()
        assert svc._image_engine(img) == 'chatgpt'

        img.klein_model = 'flux2_klein_fp8.safetensors'      # a local Klein row
        assert svc._image_engine(img) == 'klein'
        img.klein_model = None                               # legacy / imported
        assert svc._image_engine(img) is None
        img.klein_model = '   '
        assert svc._image_engine(img) is None


def test_capabilities_publishes_the_fanout_cap(client):
    """The workspace mirrors this number to warn BEFORE the click; it must not
    hardcode its own copy."""
    caps = client.get('/api/capabilities').get_json()
    assert caps['max_fanout'] == svc.MAX_FANOUT
