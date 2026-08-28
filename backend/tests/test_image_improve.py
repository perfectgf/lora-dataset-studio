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


def _improve_candidate(svc, image_cls, dataset_id, parent_image_id, *,
                       status='pending', filename='improved.png', write_file=True,
                       job_id=None):
    candidate = image_cls(
        dataset_id=dataset_id,
        source='generated',
        status=status,
        filename=filename,
        job_id=job_id,
        parent_image_id=parent_image_id,
        derivation_kind=svc.KLEIN_IMAGE_IMPROVE,
        variation_label='Klein upscale & improve',
    )
    svc.db.session.add(candidate)
    svc.db.session.flush()
    if filename and write_file:
        os.makedirs(svc._dataset_dir(dataset_id), exist_ok=True)
        with open(os.path.join(svc._dataset_dir(dataset_id), filename), 'wb') as fh:
            fh.write(_png((200, 180, 160)))
    svc.db.session.commit()
    return candidate


def test_keeping_an_improvement_unkeeps_its_kept_parent_without_deleting_files(app):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    with app.app_context():
        ds, parent, raw = _source(svc, FaceDatasetImage, LOCAL_USER)
        parent_id = parent.id
        candidate = _improve_candidate(svc, FaceDatasetImage, ds.id, parent_id)
        candidate_id = candidate.id

        assert svc.set_image_status(LOCAL_USER, candidate_id, 'keep') is True

        svc.db.session.expire_all()
        parent = svc.db.session.get(FaceDatasetImage, parent_id)
        candidate = svc.db.session.get(FaceDatasetImage, candidate_id)
        assert candidate.status == 'keep'
        assert parent.status == 'pending'
        with open(svc._img_path(parent), 'rb') as fh:
            assert fh.read() == raw


def test_keeping_an_inflight_or_missing_improvement_leaves_parent_kept(app):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    with app.app_context():
        ds, parent, _raw = _source(svc, FaceDatasetImage, LOCAL_USER)
        in_flight = _improve_candidate(
            svc, FaceDatasetImage, ds.id, parent.id, filename=None)
        missing_file = _improve_candidate(
            svc, FaceDatasetImage, ds.id, parent.id,
            filename='result-not-written.png', write_file=False)
        non_leaf_filename = _improve_candidate(
            svc, FaceDatasetImage, ds.id, parent.id,
            # This resolves to the real source file unless the helper rejects
            # path-like legacy/corrupt names before looking on disk.
            filename='./source.png', write_file=False)

        assert svc.set_image_status(LOCAL_USER, in_flight.id, 'keep') is True
        assert svc.set_image_status(LOCAL_USER, missing_file.id, 'keep') is True
        assert svc.set_image_status(LOCAL_USER, non_leaf_filename.id, 'keep') is True

        svc.db.session.expire_all()
        assert svc.db.session.get(FaceDatasetImage, in_flight.id).status == 'keep'
        assert svc.db.session.get(FaceDatasetImage, missing_file.id).status == 'keep'
        assert svc.db.session.get(FaceDatasetImage, non_leaf_filename.id).status == 'keep'
        assert svc.db.session.get(FaceDatasetImage, parent.id).status == 'keep'


@pytest.mark.parametrize('current_status', ('pending', 'reject'))
def test_keeping_improvement_uses_current_database_candidate_status(app, current_status):
    """Completion may hold a stale ORM candidate while the user changes it."""
    from sqlalchemy import update

    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    with app.app_context():
        ds, parent, _raw = _source(svc, FaceDatasetImage, LOCAL_USER)
        candidate = _improve_candidate(
            svc, FaceDatasetImage, ds.id, parent.id, status='keep')
        # Deliberately leave `candidate.status` stale at Keep while changing its
        # database row, which simulates a user decision racing a completion.
        svc.db.session.execute(
            update(FaceDatasetImage)
            .where(FaceDatasetImage.id == candidate.id)
            .values(status=current_status)
            .execution_options(synchronize_session=False))

        assert svc._unkeep_parent_for_kept_improvement(candidate) is False
        svc.db.session.commit()
        svc.db.session.expire_all()
        assert svc.db.session.get(FaceDatasetImage, parent.id).status == 'keep'


def test_late_improvement_success_unkeeps_a_parent_only_if_candidate_stays_kept(
        app, monkeypatch, tmp_path):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    output_dir = tmp_path / 'comfy-output'
    output_dir.mkdir()
    monkeypatch.setattr(svc, '_comfy_output_dir', lambda: str(output_dir))
    with app.app_context():
        ds, parent, _raw = _source(svc, FaceDatasetImage, LOCAL_USER)
        candidate = _improve_candidate(
            svc, FaceDatasetImage, ds.id, parent.id,
            filename=None, job_id='late-success')

        assert svc.set_image_status(LOCAL_USER, candidate.id, 'keep') is True
        assert svc.db.session.get(FaceDatasetImage, parent.id).status == 'keep'

        (output_dir / 'late-success.png').write_bytes(_png((120, 160, 200)))
        svc.link_completed_dataset_image('late-success', 'late-success.png', failed=False)

        svc.db.session.expire_all()
        candidate = svc.db.session.get(FaceDatasetImage, candidate.id)
        assert candidate.status == 'keep'
        assert candidate.filename == 'late-success.png'
        assert os.path.isfile(svc._img_path(candidate))
        assert svc.db.session.get(FaceDatasetImage, parent.id).status == 'pending'


def test_late_improvement_success_respects_a_later_return_to_pending(
        app, monkeypatch, tmp_path):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    output_dir = tmp_path / 'comfy-output'
    output_dir.mkdir()
    monkeypatch.setattr(svc, '_comfy_output_dir', lambda: str(output_dir))
    with app.app_context():
        ds, parent, _raw = _source(svc, FaceDatasetImage, LOCAL_USER)
        candidate = _improve_candidate(
            svc, FaceDatasetImage, ds.id, parent.id,
            filename=None, job_id='late-reset')

        assert svc.set_image_status(LOCAL_USER, candidate.id, 'keep') is True
        assert svc.set_image_status(LOCAL_USER, candidate.id, 'pending') is True
        (output_dir / 'late-reset.png').write_bytes(_png((120, 160, 200)))
        svc.link_completed_dataset_image('late-reset', 'late-reset.png', failed=False)

        svc.db.session.expire_all()
        assert svc.db.session.get(FaceDatasetImage, candidate.id).status == 'pending'
        assert svc.db.session.get(FaceDatasetImage, parent.id).status == 'keep'


def test_late_improvement_failure_leaves_a_kept_parent_as_fallback(app):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    with app.app_context():
        ds, parent, _raw = _source(svc, FaceDatasetImage, LOCAL_USER)
        candidate = _improve_candidate(
            svc, FaceDatasetImage, ds.id, parent.id,
            filename=None, job_id='late-failure')

        assert svc.set_image_status(LOCAL_USER, candidate.id, 'keep') is True
        svc.link_completed_dataset_image('late-failure', 'never-arrived.png', failed=True)

        svc.db.session.expire_all()
        assert svc.db.session.get(FaceDatasetImage, candidate.id).status == 'failed'
        assert svc.db.session.get(FaceDatasetImage, parent.id).status == 'keep'


@pytest.mark.parametrize('parent_status', ('pending', 'reject', 'failed'))
def test_keeping_an_improvement_leaves_a_nonkept_parent_alone(app, parent_status):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    with app.app_context():
        ds, parent, _raw = _source(svc, FaceDatasetImage, LOCAL_USER)
        parent.status = parent_status
        svc.db.session.commit()
        candidate = _improve_candidate(svc, FaceDatasetImage, ds.id, parent.id)

        assert svc.set_image_status(LOCAL_USER, candidate.id, 'keep') is True
        svc.db.session.expire_all()
        assert svc.db.session.get(FaceDatasetImage, parent.id).status == parent_status


def test_keeping_an_improvement_ignores_missing_or_cross_dataset_parent(app):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    with app.app_context():
        local, _local_parent, _raw = _source(svc, FaceDatasetImage, LOCAL_USER)
        foreign, foreign_parent, _raw = _source(svc, FaceDatasetImage, LOCAL_USER)
        cross_dataset = _improve_candidate(
            svc, FaceDatasetImage, local.id, foreign_parent.id,
            filename='cross-dataset.png')
        missing_parent = _improve_candidate(
            svc, FaceDatasetImage, local.id, foreign_parent.id + 1_000_000,
            filename='missing-parent.png')

        assert svc.set_image_status(LOCAL_USER, cross_dataset.id, 'keep') is True
        assert svc.set_image_status(LOCAL_USER, missing_parent.id, 'keep') is True
        svc.db.session.expire_all()
        assert svc.db.session.get(FaceDatasetImage, foreign_parent.id).status == 'keep'


def test_batch_keep_makes_improvement_win_after_all_selected_rows_are_kept(app):
    """A restored/legacy lineage may list the child before its parent by id.

    This deliberately creates that order: an inline implementation would see a
    pending parent while processing the candidate, then put the parent back to
    keep later in the loop.  The second status phase must leave it pending.
    """
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Batch improve', 'batch-improve')
        candidate = FaceDatasetImage(
            dataset_id=ds.id, source='generated', status='pending',
            filename='improved.png',
            derivation_kind=svc.KLEIN_IMAGE_IMPROVE,
            variation_label='Klein upscale & improve')
        svc.db.session.add(candidate)
        svc.db.session.flush()
        os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
        with open(os.path.join(svc._dataset_dir(ds.id), candidate.filename), 'wb') as fh:
            fh.write(_png((200, 180, 160)))
        parent = FaceDatasetImage(
            dataset_id=ds.id, source='import', status='pending',
            filename='original.png')
        svc.db.session.add(parent)
        svc.db.session.flush()
        candidate.parent_image_id = parent.id
        svc.db.session.commit()
        assert candidate.id < parent.id

        assert svc.batch_image_action(
            LOCAL_USER, ds.id, [candidate.id, parent.id], 'keep') == 2

        svc.db.session.expire_all()
        assert svc.db.session.get(FaceDatasetImage, candidate.id).status == 'keep'
        assert svc.db.session.get(FaceDatasetImage, parent.id).status == 'pending'


@pytest.mark.parametrize('configured_prompt', [
    '',
    'Restore natural detail while preserving the person and composition.',
])
def test_improve_existing_image_is_non_destructive_and_uses_metadata_profile(
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
        assert candidate.variation_prompt == svc.KLEIN_IMAGE_IMPROVE_PROMPT
        assert candidate.variation_label.startswith('Klein upscale & improve')
        assert candidate.job_id == 'improve-job-1'
        assert queued[0]['source_filename'] == source.filename
        assert queued[0]['source_path'] == svc._img_path(source)
        assert queued[0]['edit_prompt'] == svc.KLEIN_IMAGE_IMPROVE_PROMPT
        # The shipped improve profile, now settings-driven (klein.improve_*): a HIGH
        # consistency because this pass must add detail without redrawing the
        # composition, the enhancement LoRA off until the user raises it, 4 steps and
        # the historical 2 MP output budget.
        assert queued[0]['lora_strength'] == 1.0
        assert queued[0]['sampler_steps'] == 4
        assert queued[0]['base_lora_strength'] == 0.0
        assert queued[0]['output_megapixels'] == 2.0
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
        # Improving an improve result CHAINS now (dedicated test below);
        # regenerate still refuses — an improve pass has no generation
        # prompt of its own to re-run.
        with pytest.raises(ValueError, match='cannot be regenerated'):
            svc.regenerate_image(LOCAL_USER, improve_candidate.id)


def test_an_improve_result_can_be_improved_again(app, monkeypatch):
    """Improves chain on the dataset lane too — parity with the Canvas lane,
    reversed on the same phone report: Klein detail THEN SeedVR2 resolution on
    one picture, each pass its own candidate. The new candidate hangs off the
    improve result it was clicked on, not off the original image, and the file
    the pass reads is the improved one."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    queued = []
    monkeypatch.setattr(keh, 'klein_missing_assets', lambda: [])
    monkeypatch.setattr(keh, 'klein_missing_nodes', lambda: [])
    monkeypatch.setattr(
        keh, 'enqueue_klein_edit',
        lambda **kwargs: (queued.append(kwargs) or f'improve-job-{len(queued)}'))
    monkeypatch.setattr(svc, '_sync_generate_activity', lambda *a, **k: None)

    with app.app_context():
        _ds, improve_result, _raw = _source(
            svc, FaceDatasetImage, LOCAL_USER, filename='improved.png',
            derivation_kind=svc.KLEIN_IMAGE_IMPROVE)
        improve_result.source = 'generated'
        improve_result.parent_image_id = 424242   # the original, long curated
        svc.db.session.commit()

        chained = svc.improve_existing_image(LOCAL_USER, improve_result.id)

        assert chained['job_id'] == 'improve-job-1'
        second = svc.db.session.get(FaceDatasetImage, chained['candidate_id'])
        assert second.id != improve_result.id
        assert second.parent_image_id == improve_result.id   # honest lineage
        assert second.derivation_kind == svc.KLEIN_IMAGE_IMPROVE
        # The file the second pass actually reads is the IMPROVED one.
        assert queued[0]['source_path'] == svc._img_path(improve_result)


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
        lambda user_id, image_id, engine=None: {'candidate_id': 41, 'job_id': 'route-job'})
    response = client.post('/api/dataset/image/7/improve', json={})
    assert response.status_code == 200
    assert response.get_json() == {
        'ok': True, 'candidate_id': 41, 'job_id': 'route-job'}


def test_improve_route_recovery_barrier_has_no_service_side_effect(client, monkeypatch):
    from app.job_queue import COMFYUI_STALLED_BARRIER_KEY, queue_manager
    from app.services import face_dataset_service as svc

    with client.application.app_context():
        queue_manager._set_system_state(
            COMFYUI_STALLED_BARRIER_KEY, {'job_id': 'unresolved'})
    monkeypatch.setattr(
        svc, 'improve_existing_image',
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('service must not run')))

    response = client.post('/api/dataset/image/7/improve', json={})
    assert response.status_code == 409
    assert response.get_json()['code'] == 'comfyui_recovery_required'


def test_improve_route_maps_not_found_and_klein_missing(client, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh
    from app.services.klein_edit_helper import KleinModelsMissing

    monkeypatch.setattr(
        keh, 'klein_missing_nodes',
        lambda: (_ for _ in ()).throw(AssertionError('route must not preflight before ownership')))
    monkeypatch.setattr(svc, 'improve_existing_image', lambda *_a, **_k: None)
    assert client.post('/api/dataset/image/404/improve').status_code == 404

    monkeypatch.setattr(
        svc, 'improve_existing_image',
        lambda *_a, **_k: (_ for _ in ()).throw(KleinModelsMissing(['klein_model'])))
    response = client.post('/api/dataset/image/8/improve', json={})
    assert response.status_code == 409
    assert response.get_json()['ok'] is False


def test_improve_route_preflights_missing_nodes(client, monkeypatch):
    from app.services import face_dataset_service as svc

    missing = [{'class_type': 'ExampleNode', 'pack': None, 'url': None}]
    monkeypatch.setattr(
        svc, 'improve_existing_image',
        lambda *_a, **_k: (_ for _ in ()).throw(svc.KleinNodesMissing([], missing)))
    response = client.post('/api/dataset/image/8/improve', json={})
    assert response.status_code == 409
    assert response.get_json()['klein_nodes_missing'] == missing


def test_dataset_payload_publishes_the_parent_link_of_every_derived_image(app):
    """The lightbox can only put an improvement NEXT TO its original if the
    payload names that original. Both columns exist in the database, but the UI
    is fed `dataset_payload` — so this is where the link has to surface, for the
    two derivations that produce a candidate (manual improve, small-image
    rescue). A pending candidate has no file yet and must still carry the link."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    with app.app_context():
        ds, source, _raw = _source(svc, FaceDatasetImage, LOCAL_USER)
        candidate = FaceDatasetImage(
            dataset_id=ds.id, source='generated', status='pending',
            parent_image_id=source.id, derivation_kind=svc.KLEIN_IMAGE_IMPROVE,
            variation_label='Klein upscale & improve')
        rescue_source = FaceDatasetImage(
            dataset_id=ds.id, filename='small.png', source='import', status='pending',
            derivation_kind=svc.SMALL_IMAGE_SOURCE)
        svc.db.session.add_all([candidate, rescue_source])
        svc.db.session.commit()
        rescue = FaceDatasetImage(
            dataset_id=ds.id, filename='rescued.png', source='generated', status='pending',
            parent_image_id=rescue_source.id, derivation_kind=svc.KLEIN_SMALL_IMAGE)
        svc.db.session.add(rescue)
        svc.db.session.commit()

        rows = {row['id']: row
                for row in svc.dataset_payload(LOCAL_USER, ds.id)['images']}

        assert rows[candidate.id]['derivation_kind'] == svc.KLEIN_IMAGE_IMPROVE
        assert rows[candidate.id]['parent_image_id'] == source.id
        assert rows[candidate.id]['filename'] is None
        assert rows[rescue.id]['derivation_kind'] == svc.KLEIN_SMALL_IMAGE
        assert rows[rescue.id]['parent_image_id'] == rescue_source.id
        # A plain image carries no link — the UI must keep falling back to
        # today's single-image lightbox for it.
        assert rows[source.id]['derivation_kind'] is None
        assert rows[source.id]['parent_image_id'] is None
