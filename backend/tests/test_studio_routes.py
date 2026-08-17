"""Studio blueprint + per-dataset lora-test routes: ComfyUI gating (only on
routes that actually enqueue a job) + service wiring.

Every test that touches gating patches `app.capabilities.probe`; tests that
exercise `create_run`/`create_comparison_run` patch the service layer instead
of the gate, since those are covered end-to-end by test_studio_service.py.
"""
import json
import os


def _create(client, name='Nova', trigger='nova'):
    return client.post('/api/dataset/create', json={'name': name, 'trigger_word': trigger}).get_json()['id']


def _comfy(monkeypatch, reachable):
    monkeypatch.setattr('app.capabilities.probe', lambda *a, **k: {'comfyui': {'reachable': reachable}})


# --- /api/studio/run gating ---------------------------------------------------

def test_studio_run_unreachable_comfyui_returns_409_with_hint(client, monkeypatch):
    _comfy(monkeypatch, False)
    resp = client.post('/api/studio/run', json={})
    assert resp.status_code == 409
    body = resp.get_json()
    assert body['error'] == 'ComfyUI is not reachable'
    assert body['hint'] == 'Check the URL in Settings'


def test_studio_run_resume_unreachable_comfyui_returns_409(client, monkeypatch):
    _comfy(monkeypatch, False)
    resp = client.post('/api/studio/run/some-run-id/resume')
    assert resp.status_code == 409


def test_studio_run_gpu_busy_returns_503(client, monkeypatch):
    """GPU busy (training/vision) must map to 503 like the vision routes'
    GpuBusyError, not the 400 a plain ValueError would give."""
    _comfy(monkeypatch, True)
    from app.job_queue import queue_manager
    with client.application.app_context():
        queue_manager._set_system_state('training_in_progress', True, ttl_seconds=60)
    resp = client.post('/api/studio/run', json={'selections': [{'dataset_id': 1, 'checkpoint': 'x'}]})
    assert resp.status_code == 503
    assert 'GPU busy' in resp.get_json()['error']


def test_studio_run_invalid_params_still_400(client, monkeypatch):
    _comfy(monkeypatch, True)
    resp = client.post('/api/studio/run', json={'selections': []})
    assert resp.status_code == 400
    assert resp.get_json()['error'] == 'no LoRA selected'


def test_studio_run_reachable_forwards_to_service(client, monkeypatch):
    _comfy(monkeypatch, True)
    monkeypatch.setattr('app.services.lora_test_studio.create_comparison_run',
                        lambda *a, **k: {'created': 2, 'seed': 42, 'count': 1, 'run_id': 'r1'})
    resp = client.post('/api/studio/run', json={'selections': [{'dataset_id': 1, 'checkpoint': 'x'}]})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body == {'ok': True, 'created': 2, 'seed': 42, 'count': 1, 'run_id': 'r1'}


def test_recovery_barrier_blocks_run_and_resume_but_not_recovery_routes(
        client, monkeypatch):
    """Only enqueue paths are gated; cancel and explicit restart confirmation
    remain available so the user can clear the condition."""
    _comfy(monkeypatch, True)
    ds_id = _create(client)
    from app.job_queue import COMFYUI_STALLED_BARRIER_KEY, queue_manager
    from app.services import lora_test_studio as lts

    with client.application.app_context():
        queue_manager._set_system_state(
            COMFYUI_STALLED_BARRIER_KEY, {'job_id': 'unresolved'})

    def must_not_enqueue(*_args, **_kwargs):
        raise AssertionError('enqueue service must not run')

    monkeypatch.setattr(lts, 'create_comparison_run', must_not_enqueue)
    monkeypatch.setattr(lts, 'resume_run', must_not_enqueue)
    monkeypatch.setattr(lts, 'create_run', must_not_enqueue)

    blocked = [
        client.post('/api/studio/run', json={}),
        client.post('/api/studio/run/run-1/resume'),
        client.post(f'/api/dataset/{ds_id}/lora-test/run', json={}),
        client.post(f'/api/dataset/{ds_id}/lora-test/resume'),
    ]
    assert all(response.status_code == 409 for response in blocked)
    assert all(response.get_json()['code'] == 'comfyui_recovery_required'
               for response in blocked)

    monkeypatch.setattr(lts, 'cancel_run', lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(
        lts, 'confirm_unknown_comfyui_restart',
        lambda *_args, **_kwargs: 1)
    assert client.post('/api/studio/run/run-1/cancel').status_code == 200
    assert client.post(
        '/api/studio/run/run-1/confirm-comfyui-restart',
        json={'confirmed_comfyui_restart': True}).status_code == 200
    assert client.post(
        f'/api/dataset/{ds_id}/lora-test/cancel').status_code == 200
    assert client.post(
        f'/api/dataset/{ds_id}/lora-test/confirm-comfyui-restart',
        json={'confirmed_comfyui_restart': True}).status_code == 200


# --- /api/studio/describe-image (image -> test prompt via Ollama vision) ------

def _png_bytes(color=(120, 60, 30), size=(48, 64)):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new('RGB', size, color).save(buf, 'PNG')
    return buf.getvalue()


def _mock_vram(monkeypatch):
    # The GPU-exclusive vision window frees ComfyUI VRAM on entry; stub it so
    # this route test never reaches a real ComfyUI.
    from app.utils.comfyui import ComfyVramFreeVerdict
    monkeypatch.setattr('app.utils.comfyui.free_comfyui_vram',
                        lambda *a, **k: ComfyVramFreeVerdict.FREED)


def test_describe_image_returns_prompt(client, monkeypatch):
    _mock_vram(monkeypatch)
    monkeypatch.setattr('app.services.vision_ollama.describe_image_ollama',
                        lambda *a, **k: 'Three-quarter shot, the subject standing by a window, soft light.')
    import io
    resp = client.post('/api/studio/describe-image',
                       data={'image': (io.BytesIO(_png_bytes()), 'ref.png')},
                       content_type='multipart/form-data')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] is True
    assert body['prompt'].startswith('Three-quarter shot')


def test_describe_image_ollama_error_surfaces_exact_message(client, monkeypatch):
    """Ollama unavailable/rejected -> the service raises RuntimeError with Ollama's
    own reason; the route must carry that exact message through as a 409."""
    _mock_vram(monkeypatch)
    msg = 'Ollama rejected the request (HTTP 400): this model has no vision support'

    def boom(*a, **k):
        raise RuntimeError(msg)

    monkeypatch.setattr('app.services.vision_ollama.describe_image_ollama', boom)
    import io
    resp = client.post('/api/studio/describe-image',
                       data={'image': (io.BytesIO(_png_bytes()), 'ref.png')},
                       content_type='multipart/form-data')
    assert resp.status_code == 409
    assert resp.get_json()['error'] == msg


def test_describe_image_rejects_non_image(client, monkeypatch):
    _mock_vram(monkeypatch)
    import io
    resp = client.post('/api/studio/describe-image',
                       data={'image': (io.BytesIO(b'this is not an image'), 'note.txt')},
                       content_type='multipart/form-data')
    assert resp.status_code == 400
    assert 'unreadable image' in resp.get_json()['error']


def test_describe_image_rejects_oversized(client, monkeypatch):
    _mock_vram(monkeypatch)
    from app.services import lora_test_studio as lts
    monkeypatch.setattr(lts, 'STUDIO_DESCRIBE_MAX_BYTES', 8)   # any real PNG exceeds this
    import io
    resp = client.post('/api/studio/describe-image',
                       data={'image': (io.BytesIO(_png_bytes()), 'ref.png')},
                       content_type='multipart/form-data')
    assert resp.status_code == 400
    assert 'too large' in resp.get_json()['error']


def test_describe_image_no_image_returns_400(client, monkeypatch):
    _mock_vram(monkeypatch)
    resp = client.post('/api/studio/describe-image', json={})
    assert resp.status_code == 400
    assert resp.get_json()['error'] == 'no image provided'


def test_describe_image_gpu_busy_returns_503(client, monkeypatch):
    """A held vision window / training in progress must map to 503, not 409/400."""
    _mock_vram(monkeypatch)
    from app.job_queue import queue_manager
    with client.application.app_context():
        queue_manager._set_system_state('training_in_progress', True, ttl_seconds=60)
    import io
    resp = client.post('/api/studio/describe-image',
                       data={'image': (io.BytesIO(_png_bytes()), 'ref.png')},
                       content_type='multipart/form-data')
    assert resp.status_code == 503
    assert 'GPU busy' in resp.get_json()['error']


# --- /api/studio/run/<id>/status + /cancel work regardless of ComfyUI --------

def test_studio_run_status_unknown_returns_404_even_when_comfyui_down(client, monkeypatch):
    _comfy(monkeypatch, False)
    resp = client.get('/api/studio/run/does-not-exist/status')
    assert resp.status_code == 404


def test_studio_run_cancel_ungated_when_comfyui_down(client, monkeypatch):
    _comfy(monkeypatch, False)
    resp = client.post('/api/studio/run/does-not-exist/cancel')
    assert resp.status_code == 200
    assert resp.get_json() == {'ok': True, 'cancelled': 0}


# --- /api/studio listings (ungated, no ComfyUI dependency) -------------------

def test_studio_checkpoints_and_recent_prompts_smoke(client):
    assert client.get('/api/studio/checkpoints').get_json() == {'loras': []}
    assert client.get('/api/studio/recent-prompts').get_json() == {'ok': True, 'prompts': []}


def test_studio_random_caption_returns_a_trimmed_kept_caption(client, app, monkeypatch):
    """Only kept, non-blank captions are eligible, and the response is clean."""
    ds_id = _create(client)
    with app.app_context():
        from app.extensions import db
        from app.models import FaceDatasetImage
        db.session.add_all([
            # Python strips U+2003 too; forcing offset 0 below proves SQL excludes it.
            FaceDatasetImage(dataset_id=ds_id, status='keep', caption='\u2003'),
            FaceDatasetImage(dataset_id=ds_id, status='keep', caption='  usable caption  '),
            FaceDatasetImage(dataset_id=ds_id, status='keep', caption=' \t\r\n '),
            FaceDatasetImage(dataset_id=ds_id, status='reject', caption='rejected caption'),
            FaceDatasetImage(dataset_id=ds_id, status='pending', caption='pending caption'),
        ])
        db.session.commit()

    monkeypatch.setattr('app.services.face_dataset_service.random.randrange', lambda count: 0)
    response = client.post('/api/studio/random-caption', json={'dataset_id': ds_id})
    assert response.status_code == 200
    assert response.get_json() == {'ok': True, 'caption': 'usable caption'}


def test_studio_random_caption_rejects_missing_or_invalid_dataset_id(client):
    # A source WAS named and is unusable -> the refusal names that source.
    for payload in ({'dataset_id': 0}, {'dataset_id': 2 ** 63},
                    {'dataset_id': '1'}, {'dataset_id': True}):
        response = client.post('/api/studio/random-caption', json=payload)
        assert response.status_code == 400
        assert response.get_json() == {'error': 'dataset_id must be a positive integer'}
    # NO source named -> the refusal names BOTH keys. The wording moved when a
    # bank became a legal source: saying "dataset_id" alone would send the caller
    # looking for the one key it happened to mention.
    for payload in ({}, [], {'bank_id': None}):
        response = client.post('/api/studio/random-caption', json=payload)
        assert response.status_code == 400
        assert response.get_json() == {
            'error': 'dataset_id or bank_id must be a positive integer'}
    # And a bank named badly is refused in the bank's own words.
    for payload in ({'bank_id': 0}, {'bank_id': '1'}, {'bank_id': True}):
        response = client.post('/api/studio/random-caption', json=payload)
        assert response.status_code == 400
        assert response.get_json() == {'error': 'bank_id must be a positive integer'}


def test_studio_random_caption_refuses_two_sources_at_once(client):
    """Guessing which pile the caller meant would draw from the wrong one in
    silence — the exact failure mode a prompt shortcut cannot afford."""
    response = client.post('/api/studio/random-caption',
                           json={'dataset_id': 1, 'bank_id': 1})
    assert response.status_code == 400
    assert response.get_json() == {'error': 'send either dataset_id or bank_id, not both'}


def test_studio_random_caption_hides_missing_or_inaccessible_dataset(client, app):
    with app.app_context():
        from app.extensions import db
        from app.models import FaceDataset
        foreign = FaceDataset(user_id='another-user', name='Private', trigger_word='private')
        db.session.add(foreign)
        db.session.commit()
        foreign_id = foreign.id

    for dataset_id in (999999, foreign_id):
        response = client.post('/api/studio/random-caption', json={'dataset_id': dataset_id})
        assert response.status_code == 404
        assert response.get_json() == {
            'error': ('The selected dataset was not found or is inaccessible. '
                      'Choose a dataset from your library and try again.')
        }


def test_random_caption_query_rechecks_ownership_after_precheck(app, monkeypatch):
    """The caption query itself must not trust an earlier ownership lookup.

    Simulates ownership changing after the preliminary check: no private caption
    may be returned even when that check was true a moment ago.
    """
    with app.app_context():
        from app.config import LOCAL_USER
        from app.extensions import db
        from app.models import FaceDataset, FaceDatasetImage
        from app.services import face_dataset_service as svc

        foreign = FaceDataset(user_id='another-user', name='Private', trigger_word='private')
        db.session.add(foreign)
        db.session.flush()
        db.session.add(FaceDatasetImage(dataset_id=foreign.id, status='keep',
                                        caption='private training caption'))
        db.session.commit()

        # Model the time-of-check result before another request transferred access.
        monkeypatch.setattr(svc, 'get_dataset', lambda *args: object())
        assert svc.random_kept_caption(LOCAL_USER, foreign.id) is None


def test_studio_random_caption_explains_when_no_usable_caption_exists(client, app):
    ds_id = _create(client)
    with app.app_context():
        from app.extensions import db
        from app.models import FaceDatasetImage
        db.session.add_all([
            FaceDatasetImage(dataset_id=ds_id, status='keep', caption=' \t\r\n '),
            FaceDatasetImage(dataset_id=ds_id, status='reject', caption='not eligible'),
            FaceDatasetImage(dataset_id=ds_id, status='pending', caption='not eligible either'),
        ])
        db.session.commit()

    response = client.post('/api/studio/random-caption', json={'dataset_id': ds_id})
    assert response.status_code == 422
    assert response.get_json() == {
        'error': ('This dataset has no usable kept captions. Caption at least one '
                  'kept image and try again.')
    }


def _make_bank(app, name='Dump', captions=(), user='local'):
    """A bank with rows straight in the DB — the picker only ever reads counts
    and captions, so no folder on disk is needed to exercise the draw."""
    from app.extensions import db
    from app.models import BankImage, ImageBank
    with app.app_context():
        bank = ImageBank(user_id=user, name=name, source_path=r'\\nowhere\unused')
        db.session.add(bank)
        db.session.flush()
        for i, (status, caption) in enumerate(captions):
            db.session.add(BankImage(bank_id=bank.id, relpath=f'{i}.jpg',
                                     status=status, caption=caption))
        db.session.commit()
        return bank.id


def test_studio_random_caption_draws_from_a_BANK_too(client, app, monkeypatch):
    """The half that was missing: a bank is captioned by the 🏷️ Caption pass long
    before anything is promoted, so it is usually the biggest pile of real
    captions on the machine. Same eligibility rule as a dataset — kept only,
    non-blank only — and the same trimming."""
    bank_id = _make_bank(app, captions=[
        ('keep', '\u2003'),                  # whitespace Python strips, SQL must too
        ('keep', '  usable bank caption  '),
        ('keep', ' \t\r\n '),
        ('reject', 'rejected caption'),
        ('pending', 'pending caption'),
    ])
    monkeypatch.setattr('app.services.image_bank_service.random.randrange', lambda count: 0)
    response = client.post('/api/studio/random-caption', json={'bank_id': bank_id})
    assert response.status_code == 200
    assert response.get_json() == {'ok': True, 'caption': 'usable bank caption'}


def test_studio_random_caption_hides_missing_or_foreign_bank(client, app):
    """Same ownership silence as the dataset lane: a bank belonging to somebody
    else is answered exactly like one that does not exist."""
    foreign = _make_bank(app, name='Private', user='another-user',
                         captions=[('keep', 'secret caption')])
    for bank_id in (999999, foreign):
        response = client.post('/api/studio/random-caption', json={'bank_id': bank_id})
        assert response.status_code == 404
        assert response.get_json() == {
            'error': ('The selected bank was not found or is inaccessible. '
                      'Choose a bank from your library and try again.')
        }


def test_studio_random_caption_explains_an_uncaptioned_bank_in_its_own_noun(client, app):
    bank_id = _make_bank(app, captions=[('keep', ' \t '), ('reject', 'not eligible')])
    response = client.post('/api/studio/random-caption', json={'bank_id': bank_id})
    assert response.status_code == 422
    assert response.get_json() == {
        'error': ('This bank has no usable kept captions. Caption at least one '
                  'kept image and try again.')
    }


def test_recent_prompts_has_no_ten_prompt_cap():
    """The old 10-prompt ceiling is gone (user request): every DISTINCT prompt in
    the scanned rows comes back, not just the ten most recent. The 1500-cell scan
    bound in user_recent_prompts stays — that's a perf guard, not this cap."""
    from types import SimpleNamespace
    from app.services.lora_test_studio import _recent_prompts
    # 12 distinct prompts, newest id first once sorted — all must survive.
    rows = [SimpleNamespace(id=i, prompt=f'prompt {i}', filename=None,
                            rating=0, dataset_id=1) for i in range(1, 13)]
    out = _recent_prompts(rows)
    assert len(out) == 12
    assert [e['prompt'] for e in out] == [f'prompt {i}' for i in range(12, 0, -1)]
    # An explicit limit still caps, so callers that want a cap keep it.
    assert len(_recent_prompts(rows, limit=5)) == 5


def test_studio_base_models_krea_type_returns_empty_list(client):
    # Aucun UNET Krea ALTERNATIF sur disque (env de test nu) → liste vide, le
    # front cache le sélecteur (le UNET câblé du workflow reste le seul choix).
    resp = client.get('/api/studio/base-models?type=krea')
    assert resp.status_code == 200
    # `models` is asserted on its own: the response also carries the family's
    # `axes` ladders (CFG/steps), which the multi-LoRA comparison has no dataset
    # to read them from — see test_studio_multilora_steps.py.
    assert resp.get_json()['models'] == []


def test_studio_base_models_krea_lists_official_then_alternatives(client, monkeypatch):
    """Des UNET Krea locaux existent → « Official » (filename vide = défaut câblé)
    en tête, puis les alternatives, labels sans extension."""
    from app.services import lora_test_studio as lts
    monkeypatch.setattr(lts, 'krea_alt_base_models',
                        lambda: ['krea\\my_custom_krea.safetensors'])
    resp = client.get('/api/studio/base-models?type=krea')
    assert resp.status_code == 200
    assert resp.get_json()['models'] == [
        {'filename': '', 'label': 'Official – Krea 2 Turbo'},
        {'filename': 'krea\\my_custom_krea.safetensors', 'label': 'my_custom_krea'},
    ]


# --- per-dataset lora-test/status --------------------------------------------

def test_lora_test_status_fresh_dataset_is_well_formed(client):
    ds_id = _create(client)
    resp = client.get(f'/api/dataset/{ds_id}/lora-test/status')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['checkpoints'] == [] and body['cells'] == [] and body['pending'] == 0


def test_lora_test_status_unknown_dataset_404(client):
    resp = client.get('/api/dataset/999999/lora-test/status')
    assert resp.status_code == 404


def test_lora_test_status_reachable_regardless_of_comfyui(client, monkeypatch):
    """History/status routes must stay viewable even with ComfyUI offline."""
    _comfy(monkeypatch, False)
    ds_id = _create(client)
    resp = client.get(f'/api/dataset/{ds_id}/lora-test/status')
    assert resp.status_code == 200


# --- per-dataset lora-test/run gating -----------------------------------------

def test_dataset_lora_test_run_unreachable_comfyui_returns_409(client, monkeypatch):
    _comfy(monkeypatch, False)
    ds_id = _create(client)
    resp = client.post(f'/api/dataset/{ds_id}/lora-test/run', json={})
    assert resp.status_code == 409
    assert resp.get_json()['hint'] == 'Check the URL in Settings'


def test_dataset_lora_test_resume_unreachable_comfyui_returns_409(client, monkeypatch):
    _comfy(monkeypatch, False)
    ds_id = _create(client)
    resp = client.post(f'/api/dataset/{ds_id}/lora-test/resume')
    assert resp.status_code == 409


def test_dataset_lora_test_run_gpu_busy_returns_503(client, monkeypatch):
    _comfy(monkeypatch, True)
    ds_id = _create(client)  # default trigger_word='nova'
    from app.job_queue import queue_manager
    with client.application.app_context():
        queue_manager._set_system_state('training_in_progress', True, ttl_seconds=60)
    resp = client.post(f'/api/dataset/{ds_id}/lora-test/run',
                       json={'checkpoints': ['x'], 'strengths': [1.0]})
    assert resp.status_code == 503
    assert 'GPU busy' in resp.get_json()['error']


# --- missing Studio assets -> structured 409 (P0-a) --------------------------

def test_dataset_lora_test_run_missing_assets_returns_structured_409(client, monkeypatch):
    """A StudioAssetsMissing from the service maps to a 409 carrying the itemized
    file/node lists (same spirit as Klein's missing-models 409) so the front lists
    the manques instead of the current silent grid."""
    _comfy(monkeypatch, True)
    ds_id = _create(client)
    from app.services import lora_test_studio as lts

    def boom(*a, **k):
        raise lts.StudioAssetsMissing(
            'sdxl', [{'path': 'models/loras/DMD2/dmd2_sdxl_4step_lora_fp16.safetensors',
                      'kind': 'LoRA'}], ['DetailDaemonSamplerNode'])
    monkeypatch.setattr('app.services.lora_test_studio.create_run', boom)
    resp = client.post(f'/api/dataset/{ds_id}/lora-test/run',
                       json={'checkpoints': ['x'], 'strengths': [1.0]})
    assert resp.status_code == 409
    body = resp.get_json()
    assert body['ok'] is False
    sm = body['studio_missing']
    assert sm['family'] == 'sdxl'
    assert sm['nodes'] == ['DetailDaemonSamplerNode']
    assert sm['files'][0]['path'] == 'models/loras/DMD2/dmd2_sdxl_4step_lora_fp16.safetensors'
    assert 'SDXL' in body['error']  # human-readable, actionable summary


def test_dataset_lora_test_run_present_but_invalid_returns_structured_409(client, monkeypatch):
    """A present-but-INVALID model file (an HTML page saved as .safetensors, a
    truncated download) surfaces in the same structured 409 under a DISTINCT `invalid`
    list, with an actionable 'delete + re-download' message — not lumped in with the
    'missing' files."""
    _comfy(monkeypatch, True)
    ds_id = _create(client)
    from app.services import lora_test_studio as lts

    def boom(*a, **k):
        raise lts.StudioAssetsMissing(
            'sdxl', [], [],
            invalid_files=[{'path': 'models/checkpoints/base.safetensors', 'kind': 'checkpoint',
                            'reason': 'base.safetensors is not a real model (looks like an HTML page).'}])
    monkeypatch.setattr('app.services.lora_test_studio.create_run', boom)
    resp = client.post(f'/api/dataset/{ds_id}/lora-test/run',
                       json={'checkpoints': ['x'], 'strengths': [1.0]})
    assert resp.status_code == 409
    body = resp.get_json()
    sm = body['studio_missing']
    assert sm['files'] == [] and sm['nodes'] == []
    assert sm['invalid'][0]['path'] == 'models/checkpoints/base.safetensors'
    assert 'not real weights' in body['error'] and 'SDXL' in body['error']


def test_studio_comparison_run_missing_assets_returns_structured_409(client, monkeypatch):
    _comfy(monkeypatch, True)
    from app.services import lora_test_studio as lts

    def boom(*a, **k):
        raise lts.StudioAssetsMissing('krea', [], ['ConditioningKrea2Rebalance'])
    monkeypatch.setattr('app.services.lora_test_studio.create_comparison_run', boom)
    resp = client.post('/api/studio/run',
                       json={'selections': [{'dataset_id': 1, 'checkpoint': 'x'}]})
    assert resp.status_code == 409
    body = resp.get_json()
    sm = body['studio_missing']
    assert sm['family'] == 'krea'
    assert sm['nodes'] == ['ConditioningKrea2Rebalance'] and sm['files'] == []
    # Discoverability: the 409 names the pack + ComfyUI-Manager search term to install.
    assert any(h['class_type'] == 'ConditioningKrea2Rebalance'
               and h['pack'] == 'ComfyUI-Conditioning-Rebalance'
               for h in sm.get('node_packs', []))
    assert 'ComfyUI-Conditioning-Rebalance' in body['error']


# --- rate: valid ratings 1/-1/0 ok, invalid -> 400 ---------------------------

def test_rate_valid_ratings_accepted(client):
    ds_id = _create(client)
    with client.application.app_context():
        from app.services import face_dataset_service as svc
        from app.models import LoraTestImage
        from app.config import LOCAL_USER
        img = LoraTestImage(dataset_id=ds_id, checkpoint='z image\\lora_nova_000001000.safetensors',
                            strength=1.0, status='done')
        svc.db.session.add(img)
        svc.db.session.commit()
        image_id = img.id
    for rating in (1, -1, 0):
        resp = client.post(f'/api/dataset/lora-test/image/{image_id}/rate', json={'rating': rating})
        assert resp.status_code == 200, (rating, resp.get_json())
        assert resp.get_json() == {'ok': True}


def test_rate_invalid_rating_returns_400(client):
    ds_id = _create(client)
    with client.application.app_context():
        from app.services import face_dataset_service as svc
        from app.models import LoraTestImage
        img = LoraTestImage(dataset_id=ds_id, checkpoint='z image\\lora_nova_000001000.safetensors',
                            strength=1.0, status='done')
        svc.db.session.add(img)
        svc.db.session.commit()
        image_id = img.id
    resp = client.post(f'/api/dataset/lora-test/image/{image_id}/rate', json={'rating': 5})
    assert resp.status_code == 400
    assert resp.get_json() == {'ok': False, 'error': 'invalid'}


# --- best set/clear roundtrip persists into FaceDataset.best_settings -------

def test_best_set_then_clear_roundtrips_through_facedataset_best_settings(client, monkeypatch, tmp_path):
    from app import config
    base = tmp_path / 'Comfy'
    lora_dir = base / 'models' / 'loras' / 'z image'
    lora_dir.mkdir(parents=True)
    ck = 'z image\\lora_nova_000002000.safetensors'
    (lora_dir / 'lora_nova_000002000.safetensors').touch()
    config.save_config({'comfyui': {'base_dir': str(base)}})
    import app.utils.comfyui as comfyui_utils
    monkeypatch.setattr(comfyui_utils, '_zimage_models_cache', {'data': None, 'timestamp': 0})

    ds_id = _create(client)
    resp = client.post(f'/api/dataset/{ds_id}/lora-test/best',
                       json={'checkpoint': ck, 'strength': 0.9})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] is True
    assert body['best_settings']['lora_filename'] == ck

    with client.application.app_context():
        from app.services import face_dataset_service as svc
        from app.models import FaceDataset
        ds = svc.db.session.get(FaceDataset, ds_id)
        stored = json.loads(ds.best_settings)
        assert stored['zimage']['lora_filename'] == ck
        assert stored['zimage']['strength'] == 0.9

    resp = client.delete(f'/api/dataset/{ds_id}/lora-test/best')
    assert resp.status_code == 200
    with client.application.app_context():
        from app.services import face_dataset_service as svc
        from app.models import FaceDataset
        ds = svc.db.session.get(FaceDataset, ds_id)
        assert ds.best_settings is None


# --- score-faces: reaches the service, no ComfyUI gate ----------------------

def test_score_faces_reaches_service_without_comfyui_gate(client, monkeypatch):
    _comfy(monkeypatch, False)  # ComfyUI down -> must NOT be gated (CPU subprocess only)
    ds_id = _create(client)
    captured = {}

    def fake_score_faces(user_id, dataset_id, family=None):
        captured['user_id'] = user_id
        captured['dataset_id'] = dataset_id
        captured['family'] = family
        return {'ranking': []}
    monkeypatch.setattr('app.services.lora_test_studio.score_faces', fake_score_faces)
    resp = client.post(f'/api/dataset/{ds_id}/lora-test/score-faces', json={'family': 'zimage'})
    assert resp.status_code == 200
    assert resp.get_json() == {'ok': True, 'ranking': []}
    assert captured == {'user_id': 'local', 'dataset_id': ds_id, 'family': 'zimage'}


# --- prompt delete: DB-only, no gate -----------------------------------------

def test_lora_test_prompt_delete_ungated_when_comfyui_down(client, monkeypatch):
    _comfy(monkeypatch, False)
    ds_id = _create(client)
    resp = client.delete(f'/api/dataset/{ds_id}/lora-test/prompt', json={'prompt': 'anything'})
    assert resp.status_code == 200
    assert resp.get_json() == {'ok': True, 'deleted': 0}


# --- export-grid: compose one run into a shareable image ---------------------

def _seed_done_cell(client, ds_id, *, filename='cell.png', aspect='16:9', run_seed=77):
    """Create a done LoraTestImage row + its on-disk fixture tile for `ds_id`."""
    import os
    from PIL import Image
    with client.application.app_context():
        from app.services import face_dataset_service as svc
        from app.models import LoraTestImage
        ds_dir = svc._dataset_dir(ds_id)
        os.makedirs(ds_dir, exist_ok=True)
        Image.new('RGB', (64, 36), (80, 90, 120)).save(os.path.join(ds_dir, filename))
        svc.db.session.add(LoraTestImage(
            dataset_id=ds_id, checkpoint='z image\\lora_nova_000002000.safetensors',
            strength=1.0, aspect=aspect, filename=filename, status='done',
            run_seed=run_seed, seed=run_seed, prompt='p', cfg=1.0, steps=12))
        svc.db.session.commit()
        return LoraTestImage.query.order_by(LoraTestImage.id.desc()).first().id


def test_export_grid_unknown_dataset_404(client):
    resp = client.post('/api/dataset/999999/lora-test/export-grid', json={})
    assert resp.status_code == 404


def test_export_grid_empty_run_returns_409(client):
    ds_id = _create(client)
    resp = client.post(f'/api/dataset/{ds_id}/lora-test/export-grid', json={})
    assert resp.status_code == 409
    assert 'export' in resp.get_json()['error'].lower()


def test_export_grid_ungated_when_comfyui_down(client, monkeypatch):
    """Composition is DB + PIL only — it must work with ComfyUI offline."""
    _comfy(monkeypatch, False)
    ds_id = _create(client)
    _seed_done_cell(client, ds_id)
    resp = client.post(f'/api/dataset/{ds_id}/lora-test/export-grid', json={})
    assert resp.status_code == 200
    assert resp.mimetype == 'image/jpeg'
    assert resp.data[:3] == b'\xff\xd8\xff'
    assert resp.headers['Content-Disposition'].startswith('attachment;')
    assert resp.headers['X-Grid-Downscaled'] in ('0', '1')


def test_export_grid_png_format_option(client):
    ds_id = _create(client)
    _seed_done_cell(client, ds_id)
    resp = client.post(f'/api/dataset/{ds_id}/lora-test/export-grid',
                       json={'format': 'png'})
    assert resp.status_code == 200
    assert resp.mimetype == 'image/png' and resp.data[:4] == b'\x89PNG'


def test_export_grid_accepts_ordered_canvas_image_ids(client, monkeypatch):
    _comfy(monkeypatch, False)
    ds_id = _create(client)
    first = _seed_done_cell(client, ds_id, filename='first.png', run_seed=11)
    second = _seed_done_cell(client, ds_id, filename='second.png', run_seed=22)
    resp = client.post(f'/api/dataset/{ds_id}/lora-test/export-grid',
                       json={'image_ids': [second, first], 'format': 'png'})
    assert resp.status_code == 200
    assert resp.mimetype == 'image/png'
    assert '_canvas.png' in resp.headers['Content-Disposition']


def test_export_grid_rejects_foreign_canvas_image(client):
    first_ds = _create(client)
    second_ds = _create(client)
    foreign = _seed_done_cell(client, second_ds, filename='foreign.png')
    resp = client.post(f'/api/dataset/{first_ds}/lora-test/export-grid',
                       json={'image_ids': [foreign]})
    assert resp.status_code == 400
    assert 'another dataset' in resp.get_json()['error']


# --- /api/index_config: deterministic field inventory ------------------------

def test_index_config_returns_documented_fields(client):
    resp = client.get('/api/index_config')
    assert resp.status_code == 200
    body = resp.get_json()
    # Exact field set: the only fields StudioGenerationSettings.jsx reads off
    # config (config.krea_loras / config.krea_samplers / config.krea_schedulers).
    assert set(body.keys()) == {'krea_loras', 'krea_samplers', 'krea_schedulers'}
    assert body['krea_loras'] == []
    assert 'er_sde' in body['krea_samplers']
    assert 'simple' in body['krea_schedulers']


# --- /api/studio/run: combine flag + /api/studio/enhance-prompt ---------------

def test_studio_run_forwards_the_combine_flag_and_per_lora_weights(client, monkeypatch):
    """The route is a pass-through: `combine` and the per-selection `weight` must reach
    the service untouched, or the stack silently degrades into a comparison grid."""
    _comfy(monkeypatch, True)
    seen = {}

    def fake(user_id, selections, strengths, **kwargs):
        seen['selections'] = selections
        seen['strengths'] = strengths
        seen['combine'] = kwargs.get('combine')
        return {'created': 1, 'seed': 7, 'count': 1, 'run_id': 'r1', 'ids': [1]}
    monkeypatch.setattr('app.services.lora_test_studio.create_comparison_run', fake)
    resp = client.post('/api/studio/run', json={
        'selections': [{'dataset_id': 1, 'checkpoint': 'a.safetensors', 'weight': 0.9},
                       {'dataset_id': 2, 'checkpoint': 'b.safetensors', 'weight': 0.55}],
        'combine': True, 'count': 1})
    assert resp.status_code == 200
    assert seen['combine'] is True
    assert [s.get('weight') for s in seen['selections']] == [0.9, 0.55]


def test_studio_enhance_prompt_returns_the_enriched_prompt(client, monkeypatch):
    monkeypatch.setattr('app.services.lora_test_studio.enhance_test_prompt',
                        lambda p: f'{p}, cinematic lighting')
    resp = client.post('/api/studio/enhance-prompt', json={'prompt': 'a girl'})
    assert resp.status_code == 200
    assert resp.get_json() == {'ok': True, 'prompt': 'a girl, cinematic lighting'}


def test_studio_enhance_prompt_maps_a_missing_ollama_to_409(client, monkeypatch):
    def boom(_p):
        raise RuntimeError('Ollama could not start')
    monkeypatch.setattr('app.services.lora_test_studio.enhance_test_prompt', boom)
    resp = client.post('/api/studio/enhance-prompt', json={'prompt': 'a girl'})
    assert resp.status_code == 409
    assert 'Ollama' in resp.get_json()['error']


def test_studio_enhance_prompt_maps_an_empty_prompt_to_400(client, monkeypatch):
    resp = client.post('/api/studio/enhance-prompt', json={'prompt': '  '})
    assert resp.status_code == 400
    assert 'nothing to enhance' in resp.get_json()['error']


def test_studio_enhance_prompt_is_not_gated_on_comfyui(client, monkeypatch):
    """Enhance touches Ollama, not ComfyUI: a stopped ComfyUI must not block it
    (the Describe route is ungated for the same reason)."""
    _comfy(monkeypatch, False)
    monkeypatch.setattr('app.services.lora_test_studio.enhance_test_prompt',
                        lambda p: 'enriched')
    resp = client.post('/api/studio/enhance-prompt', json={'prompt': 'a girl'})
    assert resp.status_code == 200


# --- ✦ Repair a GENERATED image (.samexit, Discord) ---------------------------

def _generated(app, ds_id, filename='gen.webp', status='done'):
    """A finished generated image with a real file in the dataset folder."""
    from app.extensions import db
    from app.models import LoraTestImage
    from app.services import face_dataset_service as fds
    from PIL import Image
    with app.app_context():
        d = fds._dataset_dir(ds_id)
        os.makedirs(d, exist_ok=True)
        Image.new('RGB', (256, 256), (90, 90, 90)).save(os.path.join(d, filename), 'WEBP')
        row = LoraTestImage(dataset_id=ds_id, checkpoint='z image\\x.safetensors',
                            strength=1.0, filename=filename, status=status)
        db.session.add(row)
        db.session.commit()
        return row.id


def _stub_klein_ok(monkeypatch, recorder=None, result=(True, None)):
    from app.services import watermark_klein as wk
    monkeypatch.setattr(wk, 'is_available', lambda: True)

    def _fake(user_id, path, boxes, **kw):
        if recorder is not None:
            recorder.append({'path': path, 'boxes': boxes, **kw})
        return result

    monkeypatch.setattr(wk, 'inpaint_watermark_klein', _fake)


def test_repair_of_a_generated_image_sends_the_prompt(client, app, monkeypatch):
    """Fix one detail instead of regenerating the whole picture (.samexit).

    The dataset lane already did this; a generated image simply has no
    FaceDatasetImage row to hang it on, so it is addressed by its LoraTestImage
    id and the file is resolved server-side.
    """
    calls = []
    _stub_klein_ok(monkeypatch, recorder=calls)
    ds_id = _create(client)
    img_id = _generated(app, ds_id)
    r = client.post(f'/api/studio/image/{img_id}/repair',
                    json={'boxes': [[0.3, 0.3, 0.2, 0.2]], 'prompt': 'remove the extra finger'})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()['ok'] is True
    assert len(calls) == 1 and calls[0]['prompt'] == 'remove the extra finger'


def test_repair_of_a_generated_image_refuses_a_blank_prompt(client, app, monkeypatch):
    calls = []
    _stub_klein_ok(monkeypatch, recorder=calls)
    ds_id = _create(client)
    img_id = _generated(app, ds_id)
    for payload in ({'boxes': [[0.3, 0.3, 0.2, 0.2]]},
                    {'boxes': [[0.3, 0.3, 0.2, 0.2]], 'prompt': '   '}):
        r = client.post(f'/api/studio/image/{img_id}/repair', json=payload)
        assert r.status_code == 400
    assert calls == [], 'nothing may reach the GPU without a prompt'


def test_repair_of_a_generated_image_needs_a_zone(client, app, monkeypatch):
    _stub_klein_ok(monkeypatch)
    ds_id = _create(client)
    img_id = _generated(app, ds_id)
    r = client.post(f'/api/studio/image/{img_id}/repair', json={'prompt': 'x'})
    assert r.status_code == 400
    assert 'draw the area' in r.get_json()['error']


def test_repair_hides_an_unknown_generated_image(client, app, monkeypatch):
    _stub_klein_ok(monkeypatch)
    r = client.post('/api/studio/image/999999/repair',
                    json={'boxes': [[0.1, 0.1, 0.2, 0.2]], 'prompt': 'x'})
    assert r.status_code == 404


def test_repair_refuses_an_image_still_rendering(client, app, monkeypatch):
    """A pending cell has no pixels to repair yet — say so instead of failing
    somewhere deeper with a file-not-found."""
    _stub_klein_ok(monkeypatch)
    ds_id = _create(client)
    img_id = _generated(app, ds_id, filename='pending.webp', status='pending')
    r = client.post(f'/api/studio/image/{img_id}/repair',
                    json={'boxes': [[0.1, 0.1, 0.2, 0.2]], 'prompt': 'x'})
    assert r.status_code == 400
    assert 'still rendering' in r.get_json()['error']


def test_a_failed_repair_leaves_the_generated_file_untouched(client, app, monkeypatch):
    """The picture is preserved BEFORE anything is written, so a repair that
    fails costs nothing — the same guarantee the dataset lane gives."""
    from app.services import face_dataset_service as fds
    _stub_klein_ok(monkeypatch, result=(False, {'kind': 'failed', 'detail': 'boom'}))
    ds_id = _create(client)
    img_id = _generated(app, ds_id)
    with app.app_context():
        path = os.path.join(fds._dataset_dir(ds_id), 'gen.webp')
        before = open(path, 'rb').read()
    r = client.post(f'/api/studio/image/{img_id}/repair',
                    json={'boxes': [[0.3, 0.3, 0.2, 0.2]], 'prompt': 'remove it'})
    assert r.status_code == 400
    assert open(path, 'rb').read() == before
