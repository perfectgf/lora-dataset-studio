"""Effective-byte analysis contract across Bank transforms and transfers.

These regressions deliberately use tiny local images and mocked model subprocesses.
They protect the boundary that matters to users: a rotation/clean changes the bytes
the Bank displays, so pre-transform results can remain historical evidence but may
never become active for the new pixels without a pass explicitly rebinding them.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
from PIL import Image, ImageDraw


RAW_WATERMARK = {
    'watermark_state': 'detected',
    'watermark_bbox': '[0.1,0.01,0.4,0.05]',
    'watermark_regions': None,
    'watermark_source': 'detector',
    'watermark_score': 0.97,
}


def _write_photo(path: Path, size=(1000, 1000), *, quality=91) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new('RGB', size, (74, 116, 173))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0] // 3, size[1] // 5), fill=(231, 184, 51))
    draw.ellipse((size[0] // 2, size[1] // 3,
                  size[0] - 17, size[1] - 29), fill=(38, 211, 126))
    image.save(path, 'JPEG', quality=quality)


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _make_bank(app, tmp_path, *, asserted=False, size=(1000, 1000)):
    from app.extensions import db
    from app.models import BankImage
    from app.services import image_bank_service as banks

    folder = tmp_path / 'effective-bank-source'
    source = folder / 'portrait.jpg'
    _write_photo(source, size=size)
    bank, added = banks.create_bank('local', 'Effective source', str(folder))
    assert added == 1
    row = BankImage.query.filter_by(bank_id=bank.id).one()
    row.status = 'keep'
    row.caption = 'a hand-authored caption that must survive'
    row.caption_origin = 'asserted'
    _seed_effective_lanes(row, source, asserted=asserted)
    for name, value in RAW_WATERMARK.items():
        setattr(row, name, value)
    row.watermark_fingerprint = _sha(source)
    db.session.commit()
    return bank, row, source


def _seed_effective_lanes(row, path, *, asserted=False) -> str:
    """Publish one complete, internally valid analysis generation."""
    from app.services import bank_transfer_metadata as transfer
    from app.services import face_dataset_service as datasets

    deterministic = datasets.bank_deterministic_analysis(str(path))
    assert deterministic is not None
    for name in transfer.DETERMINISTIC_ANALYSIS_FIELDS:
        setattr(row, name, deterministic[name])
    with Image.open(path) as image:
        row.width, row.height = image.size
    row.analysis_fingerprint = _sha(path)
    row.face_state = 'scorable'
    row.face_det = 0.91
    row.face_yaw = -12.5
    row.aesthetic_score = 7.2
    row.nsfw_score = 0.03
    row.medium = 'photo'
    row.medium_margin = 0.42
    row.framing = 'body'
    row.dup_group = 7
    row.semantic_dup_group = 8
    row.face_cluster = 4
    row.face_cluster_origin = 'asserted' if asserted else None
    row.style_cluster = 3
    return row.analysis_fingerprint


def _seed_runtime_caches(bank, row, path) -> None:
    """Model-child output shape, written through the production safe NPZ codec."""
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    score_unit = 1.0 / math.sqrt(768)
    face_unit = 1.0 / math.sqrt(512)
    key = str(path)
    entries = {
        key: {
            'score': {
                'state': 'ok', 'aesthetic': row.aesthetic_score,
                'nsfw': row.nsfw_score, 'embedding': (score_unit,) * 768,
            },
            'face': {
                'state': row.face_state, 'det': row.face_det,
                'bbox_frac': 0.2, 'yaw': row.face_yaw,
                'embedding': (face_unit,) * 512,
            },
        },
    }
    assert transfer.write_runtime_caches(
        banks._score_cache_path(bank.id), banks._face_cache_path(bank.id),
        entries, expected_fingerprints={key: _sha(path)}) == {
            'score': 1, 'face': 1,
        }


def _assert_only_user_asserted_effective_state_survives(row) -> None:
    from app.services import bank_transfer_metadata as transfer

    for name in transfer.BANK_EFFECTIVE_ANALYSIS_FIELDS:
        if name == 'face_cluster':
            assert getattr(row, name) == 4
        elif name == 'face_cluster_origin':
            assert getattr(row, name) == 'asserted'
        else:
            assert getattr(row, name) is None, name
    assert row.analysis_fingerprint is None
    assert row.width is None and row.height is None
    assert row.caption == 'a hand-authored caption that must survive'
    assert row.caption_origin == 'asserted'
    assert row.status == 'keep'
    assert row.watermark_bbox == RAW_WATERMARK['watermark_bbox']
    assert row.watermark_regions is None
    assert row.watermark_source == 'detector'
    assert row.watermark_score == pytest.approx(0.97)


def _mutate_then_isolate_assertion(app, bank, row, mutation):
    """Change effective bytes, then remove the independent watermark history."""
    from app.extensions import db
    from app.models import BankImage
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    if mutation == 'rotate':
        result = banks.rotate_images('local', bank.id, [row.id], 90)
        assert result['rotated'] == 1
    else:
        job = banks.start_watermark_crop(app, 'local', bank.id)
        assert job['error'] is None, job
        assert banks.clean_image_path(bank.id, row.id).is_file()
    row = db.session.get(BankImage, row.id, populate_existing=True)
    if mutation == 'rotate':
        # Rotation does not depend on raw watermark history.  Removing it makes
        # this case exercise the assertion-only overlay with assurance=None.
        for name in transfer.BANK_WATERMARK_ANALYSIS_FIELDS:
            setattr(row, name, None)
        row.watermark_fingerprint = None
        db.session.commit()
    return row


@pytest.mark.parametrize(
    ('mutation', 'expected_state', 'expected_clean_method'),
    (
        ('rotate', 'detected', None),
        ('clean', 'cleaned', 'crop'),
        ('undo_clean', 'detected', None),
    ),
)
def test_pixel_mutations_invalidate_effective_lanes_but_keep_raw_and_user_history(
        app, tmp_path, mutation, expected_state, expected_clean_method):
    from app.extensions import db
    from app.models import BankImage
    from app.services import image_bank_service as banks

    with app.app_context():
        bank, row, raw_path = _make_bank(app, tmp_path, asserted=True)
        raw_fingerprint = row.watermark_fingerprint

        if mutation == 'rotate':
            assert banks.rotate_images('local', bank.id, [row.id], 90)['rotated'] == 1
        else:
            job = banks.start_watermark_crop(app, 'local', bank.id)
            assert job['error'] is None, job
            row = db.session.get(BankImage, row.id, populate_existing=True)
            assert banks.clean_image_path(bank.id, row.id).is_file()
            if mutation == 'undo_clean':
                # Undo is itself a byte-generation change. Seed measurements for
                # the cleaned generation first, then prove they cannot leak back
                # onto the raw image restored by Undo.
                clean_path = banks.analysis_image_path(bank, row)
                _seed_effective_lanes(row, clean_path, asserted=True)
                db.session.commit()
                assert banks.undo_watermark_clean(
                    'local', bank.id, [row.id]) == 1

        row = db.session.get(BankImage, row.id, populate_existing=True)
        _assert_only_user_asserted_effective_state_survives(row)
        assert raw_path.is_file()
        assert row.watermark_state == expected_state
        assert row.watermark_clean_method == expected_clean_method
        assert row.watermark_fingerprint == raw_fingerprint == _sha(raw_path)


@pytest.mark.parametrize('mutation', ('rotate', 'clean'))
def test_asserted_face_membership_survives_mutation_and_bank_copy_without_stale_lanes(
        app, tmp_path, mutation):
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import bank_jobs, bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source, _raw = _make_bank(
            app, tmp_path, asserted=True)
        source = _mutate_then_isolate_assertion(
            app, source_bank, source, mutation)
        effective = banks.analysis_image_path(
            source_bank, source, refresh_rotation=True)
        payload = Path(effective).read_bytes()

        assert (source.face_cluster, source.face_cluster_origin) == (4, 'asserted')
        assert not banks._has_bank_pixel_analysis(source)
        assurance = banks._analysis_transfer_assurance(
            source, effective, payload)
        assert assurance is None

        destination_id = banks.start_bank_promote(
            app, 'local', source_bank.id, [source.id],
            f'Asserted {mutation} copy')
        destination = db.session.get(ImageBank, destination_id)
        assert destination is not None, bank_jobs.get(source_bank.id)
        copied = BankImage.query.filter_by(bank_id=destination_id).one()
        assert (copied.face_cluster,
                copied.face_cluster_origin) == (4, 'asserted')
        for name in transfer.BANK_TRANSFORM_STALE_ANALYSIS_FIELDS:
            if (name not in ('face_cluster', 'face_cluster_origin')
                    and name not in transfer.BANK_WATERMARK_ANALYSIS_FIELDS):
                assert getattr(copied, name) is None, name


@pytest.mark.parametrize('mutation', ('rotate', 'clean'))
def test_asserted_face_membership_survives_bank_dataset_bank_roundtrip(
        app, tmp_path, mutation):
    from app.models import BankImage, FaceDatasetImage
    from app.services import bank_transfer_metadata as transfer
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source, _raw = _make_bank(
            app, tmp_path, asserted=True)
        source = _mutate_then_isolate_assertion(
            app, source_bank, source, mutation)
        dataset = datasets.create_dataset(
            'local', f'Asserted {mutation}', f'asserted_{mutation}')

        banks.start_promote(
            app, 'local', source_bank.id, [source.id], dataset.id)
        dataset_row = FaceDatasetImage.query.filter_by(
            dataset_id=dataset.id).one()
        snapshot = transfer.parse_snapshot(dataset_row.bank_analysis_snapshot)
        assert snapshot is not None
        assert (snapshot['analysis']['face_cluster'],
                snapshot['analysis']['face_cluster_origin']) == (4, 'asserted')
        for name in transfer.BANK_TRANSFORM_STALE_ANALYSIS_FIELDS:
            if (name not in ('face_cluster', 'face_cluster_origin')
                    and name not in transfer.BANK_WATERMARK_ANALYSIS_FIELDS):
                assert snapshot['analysis'][name] is None, name
        assert snapshot['cache_ref'] is None

        returned_bank_id = banks.start_dataset_import(
            app, 'local', dataset.id, f'Asserted {mutation} returned')
        returned = BankImage.query.filter_by(bank_id=returned_bank_id).one()
        assert returned.face_cluster is not None
        assert returned.face_cluster_origin == 'asserted'
        for name in transfer.BANK_TRANSFORM_STALE_ANALYSIS_FIELDS:
            if (name not in ('face_cluster', 'face_cluster_origin')
                    and name not in transfer.BANK_WATERMARK_ANALYSIS_FIELDS):
                assert getattr(returned, name) is None, name


@pytest.mark.parametrize('destination', ('bank', 'dataset'))
def test_asserted_face_membership_does_not_hide_a_true_stale_pixel_lane(
        app, tmp_path, destination):
    from app.extensions import db
    from app.models import FaceDatasetImage, ImageBank
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source, _raw = _make_bank(
            app, tmp_path, asserted=True)
        source = _mutate_then_isolate_assertion(
            app, source_bank, source, 'rotate')
        # Simulate a legacy/corrupt row that retained a real pixel-derived fact
        # after its shared fingerprint was cleared.
        source.framing = 'body'
        db.session.commit()
        effective = banks.analysis_image_path(
            source_bank, source, refresh_rotation=True)
        assert banks._has_bank_pixel_analysis(source)
        assert banks._analysis_transfer_assurance(
            source, effective, Path(effective).read_bytes()) is None

        if destination == 'bank':
            destination_id = banks.start_bank_promote(
                app, 'local', source_bank.id, [source.id], 'Must be refused')
            assert db.session.get(ImageBank, destination_id) is None
        else:
            dataset = datasets.create_dataset(
                'local', 'Must be refused', 'must_be_refused')
            banks.start_promote(
                app, 'local', source_bank.id, [source.id], dataset.id)
            assert FaceDatasetImage.query.filter_by(
                dataset_id=dataset.id).count() == 0


def test_all_effective_analysis_passes_resolve_the_transformed_image(
        app, tmp_path, monkeypatch):
    """Quality, Score, Face and Framing must all consume the turned derivative."""
    from app import capabilities
    from app.extensions import db
    from app.models import BankImage
    from app.services import bank_transfer_metadata as transfer
    from app.services import face_similarity, image_bank_service as banks
    from app.services import vision_ollama

    with app.app_context():
        bank, row, raw_path = _make_bank(app, tmp_path, asserted=False,
                                         size=(960, 800))
        banks.rotate_images('local', bank.id, [row.id], 90)
        row = db.session.get(BankImage, row.id, populate_existing=True)
        expected_path = banks.analysis_image_path(
            bank, row, refresh_rotation=True)
        assert expected_path and Path(expected_path).is_file()
        assert Path(expected_path).resolve() != raw_path.resolve()

        phase = {'name': None}
        resolved = {'quality': [], 'score': [], 'face': [], 'framing': []}
        real_resolver = banks.analysis_image_path

        def resolver_spy(*args, **kwargs):
            path = real_resolver(*args, **kwargs)
            if phase['name']:
                resolved[phase['name']].append(path)
            return path

        monkeypatch.setattr(banks, 'analysis_image_path', resolver_spy)

        # Quality is pure PIL; retain the real worker and only record its input.
        quality_inputs = []
        real_scan_one = banks._scan_one

        def scan_spy(src_root, thumbs, item):
            quality_inputs.append(item[1])
            return real_scan_one(src_root, thumbs, item)

        monkeypatch.setattr(banks, '_scan_one', scan_spy)
        # A one-row quality pass has no duplicate relation to rebuild. The real
        # regroup implementation imports NumPy, intentionally absent from the
        # lightweight Flask environment used by this test file.
        monkeypatch.setattr(banks, 'rebuild_dup_groups', lambda *_a, **_k: 0)
        phase['name'] = 'quality'
        quality_job = banks.start_scan(
            app, 'local', bank.id, rescan=True, ids=[row.id])
        assert quality_job['error'] is None, quality_job
        assert quality_inputs == [expected_path]

        monkeypatch.setattr(
            capabilities, 'probe_bank_scoring', lambda: {'ok': True})
        monkeypatch.setattr(face_similarity, 'is_available', lambda: True)
        monkeypatch.setattr(banks, '_resolve_score_device', lambda: ('cpu', False))
        monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))
        monkeypatch.setattr(banks, '_chain_medium_after_score', lambda *_args: '')

        subprocess_inputs = {}

        def fake_infer(_job, _python, script, payload, _cache, _regex, _window):
            paths = json.loads(payload)['images']
            subprocess_inputs[Path(script).name] = paths
            assert paths == [expected_path]
            fingerprint = _sha(expected_path)
            if Path(script) == Path(banks._SCORE_SCRIPT):
                result = {
                    'state': 'ok', 'aesthetic': 8.1, 'nsfw': 0.02,
                    'fingerprint': fingerprint,
                }
            else:
                result = {
                    'state': 'scorable', 'det': 0.94, 'bbox_frac': 0.2,
                    'yaw': 7.0, 'fingerprint': fingerprint,
                }
            return ({'ok': True, 'results': {expected_path: result},
                     'clusters': {expected_path: 1}, 'computed': 1, 'reused': 0},
                    [], 0)

        monkeypatch.setattr(banks, '_drive_infer_subprocess', fake_infer)

        phase['name'] = 'score'
        score_job = banks.start_score(app, 'local', bank.id, rescore=True)
        assert score_job['error'] is None, score_job
        phase['name'] = 'face'
        face_job = banks.start_faces(app, 'local', bank.id)
        assert face_job['error'] is None, face_job

        monkeypatch.setattr(
            capabilities, 'probe_ollama_model', lambda: {'ok': True})
        monkeypatch.setattr(banks, '_gpu_busy_reason', lambda: None)
        framing_payloads = []

        def fake_describe(payload, *_args, **_kwargs):
            framing_payloads.append(hashlib.sha256(payload).hexdigest())
            return '{"framing":"body","angle":"front"}'

        monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
        monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda: True)
        phase['name'] = 'framing'
        framing_job = banks.start_framing(
            app, 'local', bank.id, rescan=True, ids=[row.id])
        assert framing_job['error'] is None, framing_job
        phase['name'] = None

        for name, calls in resolved.items():
            assert calls, f'{name} never resolved an analysis path'
            assert set(calls) == {expected_path}, (name, calls)
        assert set(subprocess_inputs) == {
            Path(banks._SCORE_SCRIPT).name, Path(banks._EMBED_SCRIPT).name,
        }
        assert framing_payloads == [_sha(expected_path)]
        row = db.session.get(BankImage, row.id, populate_existing=True)
        assert row.analysis_fingerprint == _sha(expected_path)
        assert row.framing == 'body'
        assert row.aesthetic_score == pytest.approx(8.1)
        assert row.face_state == 'scorable'
        assert row.face_det == pytest.approx(0.94)
        assert row.face_yaw == pytest.approx(7.0)
        assert all(getattr(row, name) is not None
                   for name in transfer.DETERMINISTIC_ANALYSIS_FIELDS
                   if name not in ('detail_ratio', 'bars_ratio', 'jpeg_quality',
                                   'origin_evidence'))


def test_first_watermark_scan_uses_raw_coordinates_without_erasing_score_lanes(
        app, tmp_path, monkeypatch):
    """The pipeline runs Score before Watermark, including on rotated rows.

    Watermark owns raw-source geometry; merely attesting that raw source for the
    first time must neither analyse the rotated derivative nor erase the current
    Score/Face/quality generation.
    """
    from app import capabilities
    from app.extensions import db
    from app.models import BankImage
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks
    from app.services import vision_ollama, watermark_detector

    with app.app_context():
        bank, row, raw_path = _make_bank(app, tmp_path, asserted=False,
                                         size=(960, 800))
        banks.rotate_images('local', bank.id, [row.id], 90)
        row = db.session.get(BankImage, row.id, populate_existing=True)
        effective_path = banks.analysis_image_path(
            bank, row, refresh_rotation=True)
        assert effective_path and Path(effective_path).resolve() != raw_path.resolve()
        _seed_effective_lanes(row, effective_path, asserted=False)
        for name in transfer.BANK_WATERMARK_ANALYSIS_FIELDS:
            setattr(row, name, None)
        row.watermark_fingerprint = None
        expected = {
            name: getattr(row, name)
            for name in transfer.BANK_EFFECTIVE_ANALYSIS_FIELDS
        }
        expected.update({
            'analysis_fingerprint': row.analysis_fingerprint,
            'width': row.width,
            'height': row.height,
        })
        db.session.commit()

        monkeypatch.setattr(
            capabilities, 'probe_ollama_model', lambda: {'ok': True})
        monkeypatch.setattr(
            watermark_detector, 'resolve_backend',
            lambda: {'backend': 'vision', 'detail': '', 'fell_back': False})
        monkeypatch.setattr(banks, '_gpu_busy_reason', lambda: None)
        seen = []

        def fake_describe(payload, *_args, **_kwargs):
            seen.append(hashlib.sha256(payload).hexdigest())
            return '{"present": false}'

        monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
        monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda: True)

        job = banks.start_watermark(
            app, 'local', bank.id, rescan=True, ids=[row.id])
        assert job['error'] is None, job

        row = db.session.get(BankImage, row.id, populate_existing=True)
        assert seen == [_sha(raw_path)]
        assert seen != [_sha(effective_path)]
        assert row.watermark_state == 'none'
        assert row.watermark_fingerprint == _sha(raw_path)
        for name, value in expected.items():
            assert getattr(row, name) == value, name
        source, fingerprint = banks._current_watermark_source(bank, row)
        assert Path(source).resolve() == raw_path.resolve()
        assert fingerprint == _sha(raw_path)


def test_first_watermark_attestation_invalidates_a_replaced_raw_generation(
        app, tmp_path):
    """A NULL watermark SHA is not permission to bless stale Score results."""
    from app.extensions import db
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    with app.app_context():
        bank, row, raw_path = _make_bank(app, tmp_path, asserted=False)
        old_analysis_fingerprint = row.analysis_fingerprint
        for name in transfer.BANK_WATERMARK_ANALYSIS_FIELDS:
            setattr(row, name, None)
        row.watermark_fingerprint = None
        db.session.commit()

        _write_photo(raw_path, quality=73)
        new_raw_fingerprint = _sha(raw_path)
        assert new_raw_fingerprint != old_analysis_fingerprint
        assert banks._prepare_watermark_write(
            row, str(raw_path), new_raw_fingerprint)

        assert row.watermark_fingerprint == new_raw_fingerprint
        assert row.analysis_fingerprint is None
        assert row.aesthetic_score is None
        assert row.face_state is None
        assert row.framing is None


def _cache_index(bank, rows):
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    paths = [str(Path(bank.source_path) / row.relpath) for row in rows]
    return transfer.load_runtime_cache_index(
        banks._score_cache_path(bank.id), banks._face_cache_path(bank.id),
        wanted_paths=paths)


def test_old_raw_generation_never_reactivates_but_reanalysed_transform_transfers(
        app, tmp_path):
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage, ImageBank
    from app.services import bank_jobs, bank_transfer_metadata as transfer
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks
    from app.services.dataset_storage import dataset_path

    with app.app_context():
        bank, row, raw_path = _make_bank(app, tmp_path, asserted=False,
                                         size=(960, 800))
        old_scores = (row.aesthetic_score, row.nsfw_score, row.face_det,
                      row.framing, row.style_cluster)
        _seed_runtime_caches(bank, row, raw_path)
        raw_cache = _cache_index(bank, (row,))
        assert set(raw_cache[str(raw_path)]) == {'score', 'face'}

        banks.rotate_images('local', bank.id, [row.id], 90)
        row = db.session.get(BankImage, row.id)
        effective_path = Path(banks.analysis_image_path(
            bank, row, refresh_rotation=True))
        assert effective_path != raw_path and _sha(effective_path) != _sha(raw_path)

        # The raw cache still exists as history, but its path+SHA cannot authorize
        # a single scalar for the turned generation.
        stale_bank_id = banks.start_bank_promote(
            app, 'local', bank.id, [row.id], 'Before re-analysis')
        stale_bank = db.session.get(ImageBank, stale_bank_id)
        assert stale_bank is not None, bank_jobs.get(bank.id)
        stale_copy = BankImage.query.filter_by(bank_id=stale_bank_id).one()
        assert (stale_copy.aesthetic_score, stale_copy.nsfw_score,
                stale_copy.face_det, stale_copy.framing,
                stale_copy.style_cluster) == (None, None, None, None, None)
        assert _cache_index(stale_bank, (stale_copy,)) == {}

        stale_dataset = datasets.create_dataset(
            'local', 'Before re-analysis', 'before_reanalysis')
        banks.start_promote(
            app, 'local', bank.id, [row.id], stale_dataset.id)
        stale_dataset_row = FaceDatasetImage.query.filter_by(
            dataset_id=stale_dataset.id).one()
        stale_snapshot = transfer.parse_snapshot(
            stale_dataset_row.bank_analysis_snapshot)
        assert stale_snapshot is not None
        assert stale_snapshot['cache_ref'] is None
        assert tuple(stale_snapshot['analysis'][name] for name in (
            'aesthetic_score', 'nsfw_score', 'face_det', 'framing',
            'style_cluster')) == (None, None, None, None, None)

        # Simulate the model workers' write-back for the bytes they actually saw.
        # The central guard is the same one used by Quality/Score/Face/Framing.
        fingerprint = _sha(effective_path)
        assert banks._prepare_analysis_write(row, str(effective_path), fingerprint)
        _seed_effective_lanes(row, effective_path, asserted=False)
        db.session.commit()
        _seed_runtime_caches(bank, row, effective_path)
        assert (row.aesthetic_score, row.nsfw_score, row.face_det,
                row.framing, row.style_cluster) == old_scores

        fresh_bank_id = banks.start_bank_promote(
            app, 'local', bank.id, [row.id], 'After re-analysis')
        fresh_bank = db.session.get(ImageBank, fresh_bank_id)
        assert fresh_bank is not None, bank_jobs.get(bank.id)
        fresh_copy = BankImage.query.filter_by(bank_id=fresh_bank_id).one()
        fresh_path = Path(fresh_bank.source_path) / fresh_copy.relpath
        assert fresh_copy.analysis_fingerprint == _sha(fresh_path) == fingerprint
        assert (fresh_copy.aesthetic_score, fresh_copy.nsfw_score,
                fresh_copy.face_det, fresh_copy.framing,
                fresh_copy.style_cluster) == old_scores
        fresh_cache = _cache_index(fresh_bank, (fresh_copy,))
        assert set(fresh_cache[str(fresh_path)]) == {'score', 'face'}
        for _kind, (_payload, _signature, digest) in fresh_cache[
                str(fresh_path)].items():
            assert digest == bytes.fromhex(fingerprint)

        fresh_dataset = datasets.create_dataset(
            'local', 'After re-analysis', 'after_reanalysis')
        banks.start_promote(
            app, 'local', bank.id, [row.id], fresh_dataset.id)
        fresh_dataset_row = FaceDatasetImage.query.filter_by(
            dataset_id=fresh_dataset.id).one()
        final_dataset_path = (Path(dataset_path(fresh_dataset.id))
                              / fresh_dataset_row.filename)
        snapshot = transfer.parse_snapshot(
            fresh_dataset_row.bank_analysis_snapshot)
        assert snapshot is not None and snapshot['cache_ref']
        assert snapshot['fingerprint'] == _sha(final_dataset_path) == fingerprint
        assert tuple(snapshot['analysis'][name] for name in (
            'aesthetic_score', 'nsfw_score', 'face_det', 'framing',
            'style_cluster')) == old_scores
        sidecar = (Path(dataset_path(fresh_dataset.id)) / '.bank-analysis-cache'
                   / f"{snapshot['cache_ref']}.npz")
        assert sidecar.is_file()
        assert set(transfer.read_cache_sidecar(
            sidecar.parent, snapshot['cache_ref'])) == {'score', 'face'}


def test_legacy_unbound_watermark_history_stays_inactive_after_baked_copy(
        app, tmp_path):
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source, _raw = _make_bank(app, tmp_path, asserted=False,
                                               size=(960, 800))
        banks.rotate_images('local', source_bank.id, [source.id], 90)
        source = db.session.get(BankImage, source.id, populate_existing=True)
        effective = banks.analysis_image_path(
            source_bank, source, refresh_rotation=True)
        _seed_effective_lanes(source, effective, asserted=False)
        _seed_runtime_caches(source_bank, source, effective)
        source.watermark_state = 'detected'
        source.watermark_bbox = '[0.1,0.1,0.3,0.3]'
        source.watermark_source = 'vision'
        source.watermark_fingerprint = None
        db.session.commit()

        destination_id = banks.start_bank_promote(
            app, 'local', source_bank.id, [source.id], 'Legacy watermark copy')
        destination_bank = db.session.get(ImageBank, destination_id)
        copied = BankImage.query.filter_by(bank_id=destination_id).one()
        assert copied.rotation is None
        assert copied.watermark_state == 'detected'  # history is still conserved
        assert copied.watermark_fingerprint is None
        assert banks._watermark_history_inactive(copied)
        assert 'watermark' not in banks.image_flags(copied, banks.thresholds())
        assert banks._needs_rescan_count(destination_bank.id) == 1
