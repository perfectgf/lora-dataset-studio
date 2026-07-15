import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image


def _png(color=(25, 50, 75)):
    buf = io.BytesIO()
    Image.new('RGB', (96, 64), color).save(buf, 'PNG')
    return buf.getvalue()


def _source(svc, image_cls, user_id, *, filename='source.png', derivation_kind=None):
    ds = svc.create_dataset(user_id, 'Improve', 'improve')
    os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
    raw = _png()
    if filename:
        with open(os.path.join(svc._dataset_dir(ds.id), filename), 'wb') as fh:
            fh.write(raw)
    image = image_cls(
        dataset_id=ds.id,
        filename=filename,
        source='import',
        status='keep',
        framing='body',
        caption='full body, outdoor light',
        variation_label='Imported low-resolution image',
        variation_prompt='original prompt',
        derivation_kind=derivation_kind,
    )
    svc.db.session.add(image)
    svc.db.session.commit()
    return ds, image, raw


@pytest.mark.parametrize('configured_prompt', [
    '',
    'Restore natural detail while preserving the person and composition.',
])
def test_improve_existing_image_is_non_destructive_and_uses_config_prompt(
        app, monkeypatch, configured_prompt):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    queued = []
    syncs = []
    monkeypatch.setattr(keh, 'klein_missing_assets', lambda: [])
    monkeypatch.setattr(keh, 'klein_missing_nodes', lambda: [])
    monkeypatch.setattr(
        keh, 'enqueue_klein_edit',
        lambda **kwargs: (queued.append(kwargs) or 'improve-job-1'))
    monkeypatch.setattr(
        svc.cfg, 'get',
        lambda key, default=None: configured_prompt
        if key == 'klein.small_image_prompt' else default)
    monkeypatch.setattr(svc, '_sync_generate_activity', syncs.append)

    with app.app_context():
        ds, source, raw = _source(svc, FaceDatasetImage, LOCAL_USER)
        source_id = source.id
        original_values = {
            field: getattr(source, field)
            for field in ('filename', 'source', 'status', 'framing', 'caption',
                          'variation_label', 'variation_prompt', 'derivation_kind',
                          'job_id', 'parent_image_id')
        }

        result = svc.improve_existing_image(LOCAL_USER, source_id)

        svc.db.session.expire_all()
        source = svc.db.session.get(FaceDatasetImage, source_id)
        candidate = svc.db.session.get(FaceDatasetImage, result['candidate_id'])
        assert {field: getattr(source, field) for field in original_values} == original_values
        with open(svc._img_path(source), 'rb') as fh:
            assert fh.read() == raw
        assert result == {'candidate_id': candidate.id, 'job_id': 'improve-job-1'}
        assert candidate.dataset_id == ds.id
        assert candidate.source == 'generated'
        assert candidate.status == 'pending'
        assert candidate.filename is None
        assert candidate.parent_image_id == source_id
        assert candidate.derivation_kind == svc.KLEIN_IMAGE_IMPROVE
        assert candidate.derivation_kind not in svc._SMALL_IMAGE_DERIVATIONS
        assert candidate.framing == source.framing
        assert candidate.caption == source.caption
        assert candidate.variation_prompt == configured_prompt
        assert candidate.variation_label.startswith('Klein upscale & improve')
        assert candidate.job_id == 'improve-job-1'
        assert queued[0]['source_filename'] == source.filename
        assert queued[0]['source_path'] == svc._img_path(source)
        assert queued[0]['edit_prompt'] == configured_prompt
        assert queued[0]['extra_metadata']['source_image_id'] == source_id
        assert queued[0]['extra_metadata']['derivation_kind'] == svc.KLEIN_IMAGE_IMPROVE
        assert syncs == [ds.id]


def test_improve_existing_image_returns_active_candidate_idempotently(app, monkeypatch):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    with app.app_context():
        ds, source, _raw = _source(svc, FaceDatasetImage, LOCAL_USER)
        active = FaceDatasetImage(
            dataset_id=ds.id, source='generated', status='pending',
            parent_image_id=source.id, derivation_kind=svc.KLEIN_IMAGE_IMPROVE,
            variation_label='Klein upscale & improve', job_id='already-running')
        svc.db.session.add(active)
        svc.db.session.commit()
        active_id = active.id

        monkeypatch.setattr(
            keh, 'klein_missing_assets',
            lambda: (_ for _ in ()).throw(AssertionError('idempotent path must not preflight')))
        monkeypatch.setattr(
            keh, 'klein_missing_nodes',
            lambda: (_ for _ in ()).throw(AssertionError('idempotent path must not preflight')))
        monkeypatch.setattr(
            keh, 'enqueue_klein_edit',
            lambda **_kwargs: (_ for _ in ()).throw(AssertionError('must not enqueue twice')))

        first = svc.improve_existing_image(LOCAL_USER, source.id)
        second = svc.improve_existing_image(LOCAL_USER, source.id)
        assert first == second == {
            'candidate_id': active_id, 'job_id': 'already-running'}
        assert FaceDatasetImage.query.filter_by(
            parent_image_id=source.id,
            derivation_kind=svc.KLEIN_IMAGE_IMPROVE).count() == 1


def test_improve_existing_image_rejects_missing_and_review_sources(app, monkeypatch):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    monkeypatch.setattr(keh, 'klein_missing_assets', lambda: [])
    monkeypatch.setattr(keh, 'klein_missing_nodes', lambda: [])
    with app.app_context():
        assert svc.improve_existing_image(LOCAL_USER, 999999) is None

        _ds, missing_name, _ = _source(
            svc, FaceDatasetImage, LOCAL_USER, filename=None)
        with pytest.raises(ValueError, match='image file required'):
            svc.improve_existing_image(LOCAL_USER, missing_name.id)

        _ds, missing_file, _ = _source(svc, FaceDatasetImage, LOCAL_USER)
        os.remove(svc._img_path(missing_file))
        with pytest.raises(ValueError, match='image file missing'):
            svc.improve_existing_image(LOCAL_USER, missing_file.id)

        _ds, review_source, _ = _source(
            svc, FaceDatasetImage, LOCAL_USER,
            derivation_kind=svc.SMALL_IMAGE_SOURCE)
        with pytest.raises(ValueError, match='resolve the small-image rescue pair'):
            svc.improve_existing_image(LOCAL_USER, review_source.id)

        _ds, improve_candidate, _ = _source(
            svc, FaceDatasetImage, LOCAL_USER,
            derivation_kind=svc.KLEIN_IMAGE_IMPROVE)
        improve_candidate.source = 'generated'
        svc.db.session.commit()
        with pytest.raises(ValueError, match='cannot be improved again'):
            svc.improve_existing_image(LOCAL_USER, improve_candidate.id)
        with pytest.raises(ValueError, match='cannot be regenerated'):
            svc.regenerate_image(LOCAL_USER, improve_candidate.id)


def test_improve_existing_image_preflights_models_and_fanout(app, monkeypatch):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh
    from app.services.klein_edit_helper import KleinModelsMissing

    with app.app_context():
        ds, source, _raw = _source(svc, FaceDatasetImage, LOCAL_USER)
        monkeypatch.setattr(keh, 'klein_missing_assets', lambda: ['klein_model'])
        monkeypatch.setattr(keh, 'klein_missing_nodes', lambda: [])
        with pytest.raises(KleinModelsMissing):
            svc.improve_existing_image(LOCAL_USER, source.id)
        assert FaceDatasetImage.query.filter_by(
            derivation_kind=svc.KLEIN_IMAGE_IMPROVE).count() == 0

        monkeypatch.setattr(keh, 'klein_missing_assets', lambda: [])
        for _ in range(svc.MAX_FANOUT):
            svc.db.session.add(FaceDatasetImage(
                dataset_id=ds.id, source='generated', status='pending'))
        svc.db.session.commit()
        with pytest.raises(ValueError, match='too many generations in flight'):
            svc.improve_existing_image(LOCAL_USER, source.id)
        assert FaceDatasetImage.query.filter_by(
            derivation_kind=svc.KLEIN_IMAGE_IMPROVE).count() == 0


def test_improve_existing_image_removes_candidate_when_enqueue_fails(app, monkeypatch):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    monkeypatch.setattr(keh, 'klein_missing_assets', lambda: [])
    monkeypatch.setattr(keh, 'klein_missing_nodes', lambda: [])
    monkeypatch.setattr(
        keh, 'enqueue_klein_edit',
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError('ComfyUI offline')))
    with app.app_context():
        _ds, source, raw = _source(svc, FaceDatasetImage, LOCAL_USER)
        source_id = source.id
        with pytest.raises(RuntimeError, match='ComfyUI offline'):
            svc.improve_existing_image(LOCAL_USER, source_id)
        assert FaceDatasetImage.query.filter_by(
            derivation_kind=svc.KLEIN_IMAGE_IMPROVE).count() == 0
        source = svc.db.session.get(FaceDatasetImage, source_id)
        assert source.status == 'keep' and source.caption == 'full body, outdoor light'
        with open(svc._img_path(source), 'rb') as fh:
            assert fh.read() == raw


def test_concurrent_improve_requests_enqueue_only_once(app, monkeypatch):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    entered = threading.Event()
    release = threading.Event()
    calls = []

    def enqueue(**kwargs):
        calls.append(kwargs)
        entered.set()
        assert release.wait(3), 'test did not release the fake enqueue'
        return 'one-concurrent-job'

    monkeypatch.setattr(keh, 'klein_missing_assets', lambda: [])
    monkeypatch.setattr(keh, 'klein_missing_nodes', lambda: [])
    monkeypatch.setattr(keh, 'enqueue_klein_edit', enqueue)
    monkeypatch.setattr(svc, '_sync_generate_activity', lambda _dataset_id: None)
    with app.app_context():
        _ds, source, _raw = _source(svc, FaceDatasetImage, LOCAL_USER)
        source_id = source.id

    def run():
        with app.app_context():
            return svc.improve_existing_image(LOCAL_USER, source_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run)
        assert entered.wait(3), 'first request never reached enqueue'
        second = pool.submit(run)
        time.sleep(0.1)
        assert not second.done(), 'second request bypassed the per-image lock'
        release.set()
        first_result = first.result(timeout=3)
        second_result = second.result(timeout=3)

    assert first_result == second_result
    assert first_result['job_id'] == 'one-concurrent-job'
    assert len(calls) == 1
    with app.app_context():
        assert FaceDatasetImage.query.filter_by(
            parent_image_id=source_id,
            derivation_kind=svc.KLEIN_IMAGE_IMPROVE).count() == 1


def test_improve_route_accepts_empty_json_and_returns_contract(client, monkeypatch):
    from app.services import face_dataset_service as svc

    monkeypatch.setattr(
        svc, 'improve_existing_image',
        lambda user_id, image_id: {'candidate_id': 41, 'job_id': 'route-job'})
    response = client.post('/api/dataset/image/7/improve', json={})
    assert response.status_code == 200
    assert response.get_json() == {
        'ok': True, 'candidate_id': 41, 'job_id': 'route-job'}


def test_improve_route_maps_not_found_and_klein_missing(client, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh
    from app.services.klein_edit_helper import KleinModelsMissing

    monkeypatch.setattr(
        keh, 'klein_missing_nodes',
        lambda: (_ for _ in ()).throw(AssertionError('route must not preflight before ownership')))
    monkeypatch.setattr(svc, 'improve_existing_image', lambda *_args: None)
    assert client.post('/api/dataset/image/404/improve').status_code == 404

    monkeypatch.setattr(
        svc, 'improve_existing_image',
        lambda *_args: (_ for _ in ()).throw(KleinModelsMissing(['klein_model'])))
    response = client.post('/api/dataset/image/8/improve', json={})
    assert response.status_code == 409
    assert response.get_json()['ok'] is False


def test_improve_route_preflights_missing_nodes(client, monkeypatch):
    from app.services import face_dataset_service as svc

    missing = [{'class_type': 'ExampleNode', 'pack': None, 'url': None}]
    monkeypatch.setattr(
        svc, 'improve_existing_image',
        lambda *_args: (_ for _ in ()).throw(svc.KleinNodesMissing([], missing)))
    response = client.post('/api/dataset/image/8/improve', json={})
    assert response.status_code == 409
    assert response.get_json()['klein_nodes_missing'] == missing
