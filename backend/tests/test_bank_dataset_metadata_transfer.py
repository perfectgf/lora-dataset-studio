"""Strict v3 analysis/cache transport between image banks and datasets."""
import hashlib
import io
import json
import math
import os
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image


SOURCE_METADATA = {
    'platform': 'pexels',
    'source_url': 'https://www.pexels.com/photo/12345/',
    'photographer': 'Jane Doe',
    'photographer_url': 'https://www.pexels.com/@jane-doe/',
}
WATERMARK_BBOX = '[0.1,0.1,0.3,0.3]'
WATERMARK_REGIONS = '[[0.1,0.1,0.3,0.3]]'

# Deliberately implausible seeded deterministic values make recalculation easy
# to detect. Direct-copy tests first synchronize them to the real source bytes,
# while their complete ML/review/group history remains the preservation oracle.
HISTORICAL_ANALYSIS = {
    'quality_state': 'ok',
    'blur_score': 13.5,
    'noise_score': 2.25,
    'uniformity_score': 41.0,
    'dhash': '0123456789abcdef',
    'face_state': 'scorable',
    'face_det': 0.91,
    'aesthetic_score': 7.2,
    'nsfw_score': 0.03,
    'detail_ratio': 0.88,
    'bars_ratio': 0.01,
    'jpeg_quality': 92.0,
    'origin': 'camera',
    'origin_evidence': 'exif-camera',
    'framing': 'body',
    'watermark_state': 'detected',
    'watermark_bbox': WATERMARK_BBOX,
    'watermark_regions': WATERMARK_REGIONS,
    'watermark_source': 'detector',
    'watermark_score': 0.97,
    'medium': 'photo',
    'medium_margin': 0.42,
    'face_yaw': -17.25,
}

EXPECTED_DIRECT_BANK_ANALYSIS_FIELDS = (
    'quality_state',
    'blur_score',
    'noise_score',
    'uniformity_score',
    'dhash',
    'detail_ratio',
    'bars_ratio',
    'jpeg_quality',
    'origin',
    'origin_evidence',
    'face_state',
    'face_det',
    'aesthetic_score',
    'nsfw_score',
    'framing',
    'watermark_state',
    'watermark_bbox',
    'watermark_regions',
    'watermark_source',
    'watermark_score',
    'medium',
    'medium_margin',
    'face_yaw',
    'dup_group',
    'semantic_dup_group',
    'clip_semantic_dup_group',
    'siglip2_semantic_dup_group',
    'face_cluster',
    'face_cluster_origin',
    'style_cluster',
)


def _write_photo(path: Path, size=(160, 96), *, quality=35):
    """Write a small, asymmetric patterned image in the format from ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new('RGB', size)
    pixels = image.load()
    for y in range(size[1]):
        for x in range(size[0]):
            pixels[x, y] = ((17 * x + 3 * y) % 256,
                            (5 * x + 19 * y) % 256,
                            (11 * x + 7 * y) % 256)
    suffix = path.suffix.lower()
    if suffix in ('.jpg', '.jpeg'):
        image.save(path, 'JPEG', quality=quality)
    elif suffix == '.webp':
        image.save(path, 'WEBP', quality=80)
    else:
        image.save(path, 'PNG')


def _dimensions(path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _row_values(row, fields) -> dict:
    return {name: getattr(row, name) for name in fields}


def _final_deterministic_analysis(path) -> dict:
    """The oracle shared by every strict transfer assertion in this module."""
    from app.services import bank_transfer_metadata as transfer
    from app.services import face_dataset_service as datasets

    analysis = datasets.bank_deterministic_analysis(path)
    return {name: analysis[name] for name in transfer.DETERMINISTIC_ANALYSIS_FIELDS}


def _assert_v3_snapshot_matches_final(snapshot_text, final_path):
    from app.services import bank_transfer_metadata as transfer

    snapshot = transfer.parse_snapshot(snapshot_text)
    assert snapshot is not None
    assert snapshot['v'] == 3
    assert set(snapshot['analysis']) == set(transfer.BANK_DIRECT_COPY_ANALYSIS_FIELDS)
    assert snapshot['fingerprint'] == hashlib.sha256(Path(final_path).read_bytes()).hexdigest()
    assert {name: snapshot['analysis'][name]
            for name in transfer.DETERMINISTIC_ANALYSIS_FIELDS} == (
                _final_deterministic_analysis(final_path))
    return snapshot


def _assert_model_analysis_is_empty(row):
    from app.services import bank_transfer_metadata as transfer

    assert _row_values(row, transfer.MODEL_ANALYSIS_FIELDS) == {
        name: None for name in transfer.MODEL_ANALYSIS_FIELDS
    }


def _valid_v2_snapshot_payload():
    """One independently valid payload to mutate for rejection cases."""
    return {
        'v': 2,
        'fingerprint': 'a' * 64,
        'analysis': {
            'quality_state': 'ok',
            'blur_score': 13.5,
            'noise_score': 2.25,
            'uniformity_score': 41.0,
            'dhash': '0123456789abcdef',
            'detail_ratio': 0.88,
            'bars_ratio': 0.01,
            'jpeg_quality': 92.0,
            'origin': 'camera',
            'origin_evidence': 'exif-camera',
        },
    }


def _bank_with_analysed_image(app, tmp_path, *, filename='one.jpg'):
    """Seed a Bank row with complete history for transfer-policy tests."""
    from app.extensions import db
    from app.models import BankImage, ImageBank

    folder = tmp_path / 'bank-source'
    source = folder / filename
    _write_photo(source)
    bank = ImageBank(user_id='local', name='Source bank', source_path=str(folder))
    db.session.add(bank)
    db.session.flush()
    image = BankImage(
        bank_id=bank.id, relpath=filename, file_size=source.stat().st_size,
        width=160, height=96, **HISTORICAL_ANALYSIS,
        # A direct Bank copy retains these relationships inside its selection;
        # Dataset round-trips and transformed copies must clear them.
        dup_group=7, semantic_dup_group=8, clip_semantic_dup_group=8,
        face_cluster=4,
        face_cluster_origin='asserted', style_cluster=3,
        caption='caption from the bank', caption_origin='asserted',
        source_metadata=json.dumps(SOURCE_METADATA), status='keep',
    )
    db.session.add(image)
    db.session.commit()
    _sync_bank_row_to_current_file(bank, image)
    image.analysis_fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    db.session.commit()
    _seed_analysis_caches(bank, (image,))
    return bank, image


def _sync_bank_row_to_current_file(bank, image) -> Path:
    """Make a seeded row describe its current source bytes exactly."""
    from app.services import bank_transfer_metadata as transfer

    path = Path(bank.source_path) / image.relpath
    analysis = _final_deterministic_analysis(path)
    for name in transfer.DETERMINISTIC_ANALYSIS_FIELDS:
        setattr(image, name, analysis[name])
    image.width, image.height = _dimensions(path)
    image.file_size = path.stat().st_size
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    image.analysis_fingerprint = fingerprint
    # Watermark geometry is attached to the pristine, pre-rotation source.  The
    # fixture seeds a real detector result, so it must carry that lane's own raw
    # authority as well as the effective-analysis authority above.
    image.watermark_fingerprint = fingerprint
    return path


def _seed_analysis_caches(bank, images):
    """Write real Score/Face NPZ files without requiring NumPy in Flask."""
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    score_unit = 1.0 / math.sqrt(768)
    face_unit = 1.0 / math.sqrt(512)
    entries = {}
    for image in images:
        path = (banks.analysis_image_path(bank, image)
                or str(Path(bank.source_path) / image.relpath))
        entries[path] = {
            'score': {
                'state': 'ok', 'aesthetic': image.aesthetic_score,
                'nsfw': image.nsfw_score,
                'embedding': (score_unit,) * 768,
            },
            'face': {
                'state': image.face_state, 'det': image.face_det,
                'bbox_frac': 0.2, 'yaw': image.face_yaw,
                'embedding': (face_unit,) * 512,
            },
        }
    counts = transfer.write_runtime_caches(
        banks._score_cache_path(bank.id), banks._face_cache_path(bank.id), entries)
    assert counts == {'score': len(entries), 'face': len(entries)}
    return entries


def _seed_analysis_caches_with_semantic(bank, images):
    """Semantic-specific fixture; historical Score/Face helpers stay unchanged."""
    from app.services import bank_semantic_models as assets
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    entries = _seed_analysis_caches(bank, images)
    semantic_unit = 1.0 / math.sqrt(assets.DIMENSION)
    for bundle in entries.values():
        bundle['semantic'] = {
            'state': 'ok', 'engine': 'siglip2',
            'model_id': assets.MODEL_ID, 'revision': assets.REVISION,
            'model_key': assets.MODEL_KEY, 'dimension': assets.DIMENSION,
            'embedding': (semantic_unit,) * assets.DIMENSION,
        }
    fingerprints = {
        path: transfer.content_fingerprint_path(path) for path in entries
    }
    counts = transfer.write_runtime_caches(
        banks._score_cache_path(bank.id), banks._face_cache_path(bank.id), entries,
        semantic_path=banks._semantic_cache_path(bank.id),
        expected_fingerprints=fingerprints)
    assert counts == {
        'score': len(entries), 'semantic': len(entries), 'face': len(entries)}
    return entries


def test_legacy_v3_active_only_semantic_group_remains_readable(app):
    from app.services import bank_transfer_metadata as transfer

    payload = b'legacy-v3-image'
    analysis = {name: None for name in transfer.BANK_DIRECT_COPY_ANALYSIS_FIELDS}
    analysis.update(HISTORICAL_ANALYSIS)
    analysis['semantic_dup_group'] = 8
    captured = transfer.captured_bank_analysis(
        analysis, payload, group_scope='a' * 32)
    stored = transfer.snapshot_storage(
        {name: HISTORICAL_ANALYSIS[name]
         for name in transfer.DETERMINISTIC_ANALYSIS_FIELDS},
        payload, captured=captured)
    legacy = json.loads(stored)
    legacy['analysis'].pop('clip_semantic_dup_group')
    legacy['analysis'].pop('siglip2_semantic_dup_group')

    parsed = transfer.parse_snapshot(json.dumps(legacy))
    assert parsed is not None and parsed['v'] == 3
    assert parsed['analysis']['semantic_dup_group'] == 8
    assert parsed['analysis']['clip_semantic_dup_group'] is None
    assert parsed['analysis']['siglip2_semantic_dup_group'] is None


def _runtime_cache_payloads(bank, rows):
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    paths = [str(Path(bank.source_path) / row.relpath) for row in rows]
    return transfer.load_runtime_cache_index(
        banks._score_cache_path(bank.id), banks._face_cache_path(bank.id),
        wanted_paths=paths)


def _runtime_cache_payloads_with_semantic(bank, rows):
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    paths = [str(Path(bank.source_path) / row.relpath) for row in rows]
    return transfer.load_runtime_cache_index(
        banks._score_cache_path(bank.id), banks._face_cache_path(bank.id),
        semantic_path=banks._semantic_cache_path(bank.id), wanted_paths=paths)


def _drop_runtime_cache_hashes(path):
    """Turn a current cache into the exact pre-hash legacy schema."""
    from app.services import npz_transport

    arrays = npz_transport.read(
        path, max_file_bytes=1024 * 1024 * 1024,
        max_uncompressed_bytes=1024 * 1024 * 1024,
        max_elements=200_000 * 768)
    assert arrays is not None and arrays.pop('hashes', None) is not None
    assert npz_transport.write_atomic(
        path, arrays, max_file_bytes=1024 * 1024 * 1024)


def _add_group_peer(bank, *, filename='two.jpg'):
    """Add a second fully-analysed member of the seeded duplicate clusters."""
    from app.extensions import db
    from app.models import BankImage

    path = Path(bank.source_path) / filename
    _write_photo(path)
    image = BankImage(
        bank_id=bank.id, relpath=filename, file_size=path.stat().st_size,
        width=160, height=96, **HISTORICAL_ANALYSIS,
        dup_group=7, semantic_dup_group=8, clip_semantic_dup_group=8,
        face_cluster=4,
        face_cluster_origin='asserted', style_cluster=3,
        caption='peer caption', caption_origin='asserted',
        source_metadata=json.dumps(SOURCE_METADATA), status='keep',
    )
    db.session.add(image)
    _sync_bank_row_to_current_file(bank, image)
    db.session.flush()
    _seed_analysis_caches(
        bank, BankImage.query.filter_by(bank_id=bank.id).all())
    return image


def _promote_to_dataset(app, bank, image, *, name='Transfer target'):
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks

    dataset = datasets.create_dataset('local', name, 'transfer')
    banks.start_promote(app, 'local', bank.id, [image.id], dataset.id)
    return dataset


def _make_source_transform(bank, image, kind):
    """Materialize a transform, returning the dimensions its destination owns."""
    from app.extensions import db
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    if kind == 'direct':
        return (160, 96)
    if kind == 'rotation':
        image.rotation = 90
        expected = (96, 160)
    elif kind == 'clean':
        clean = banks.clean_image_path(bank.id, image.id)
        _write_photo(clean, size=(83, 57))
        image.watermark_clean_method = 'crop'
        expected = (83, 57)
    else:
        raise AssertionError(f'unknown transform {kind!r}')
    # Effective pixels changed; raw-file analyses are stale until their passes
    # explicitly rebind to the resolved payload.
    for field in transfer.BANK_PIXEL_DERIVED_FIELDS:
        setattr(image, field, None)
    image.analysis_fingerprint = None
    db.session.commit()
    return expected


def _bind_complete_analysis_to_effective_bytes(bank, image) -> Path:
    """Simulate every pass rerun after a clean/rotation and seed exact caches."""
    from app.extensions import db
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    path = Path(banks.analysis_image_path(bank, image, refresh_rotation=True))
    for name, value in _final_deterministic_analysis(path).items():
        setattr(image, name, value)
    for name in transfer.MODEL_ANALYSIS_FIELDS:
        setattr(image, name, HISTORICAL_ANALYSIS[name])
    for name in ('framing', 'medium', 'medium_margin', 'face_yaw'):
        setattr(image, name, HISTORICAL_ANALYSIS[name])
    image.dup_group = 7
    image.semantic_dup_group = 8
    image.face_cluster = 4
    image.face_cluster_origin = 'asserted'
    image.style_cluster = 3
    image.width, image.height = _dimensions(path)
    image.analysis_fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    db.session.commit()
    _seed_analysis_caches(bank, (image,))
    return path


@pytest.mark.parametrize(
    ('transform', 'expected_size', 'expected_format', 'expected_suffix'),
    (('direct', (160, 96), 'JPEG', '.jpg'),
     ('rotation', (96, 160), 'JPEG', '.jpg'),
     ('clean', (83, 57), 'WEBP', '.webp')),
)
def test_bank_to_dataset_snapshot_describes_the_final_preserved_file(
        app, tmp_path, transform, expected_size, expected_format, expected_suffix):
    """Direct/rotated Bank masters retain JPEG; a cleaned Bank derivative remains
    WebP. In every case the snapshot describes the final Dataset file, never the
    old Bank row."""
    from app.models import FaceDatasetImage
    from app.services.dataset_storage import dataset_path

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        assert _make_source_transform(source_bank, source_image, transform) == expected_size
        source_jpeg_quality = source_image.jpeg_quality
        dataset = _promote_to_dataset(app, source_bank, source_image)
        dataset_image = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).one()
        final_path = Path(dataset_path(dataset.id)) / dataset_image.filename

        with Image.open(final_path) as stored:
            assert stored.format == expected_format
            assert stored.size == expected_size
        assert dataset_image.filename.endswith(expected_suffix)
        snapshot = _assert_v3_snapshot_matches_final(
            dataset_image.bank_analysis_snapshot, final_path)
        # A clean derivative is WebP and carries no JPEG estimate. The helper
        # above already proves that a preserved JPEG's value is measured from its
        # final bytes, not copied from Bank history.
        if expected_format == 'WEBP':
            assert snapshot['analysis']['jpeg_quality'] is None
        assert source_image.jpeg_quality == source_jpeg_quality


def test_dataset_to_bank_restores_v3_analysis_and_current_user_metadata(
        app, tmp_path):
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks
    from app.services.dataset_storage import dataset_path

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        _sync_bank_row_to_current_file(source_bank, source_image)
        db.session.commit()
        dataset = _promote_to_dataset(app, source_bank, source_image)
        dataset_image = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).one()
        dataset_file = Path(dataset_path(dataset.id)) / dataset_image.filename
        snapshot = _assert_v3_snapshot_matches_final(
            dataset_image.bank_analysis_snapshot, dataset_file)

        # The Dataset is now the owner of user-facing choices.
        dataset_image.caption = 'caption edited in the dataset'
        dataset_image.framing = 'face'
        dataset_image.watermark_state = 'dismissed'
        dataset_image.watermark_regions = '[]'
        db.session.commit()

        returned_bank_id = banks.start_dataset_import(app, 'local', dataset.id, 'Returned')
        returned = BankImage.query.filter_by(bank_id=returned_bank_id).one()
        returned_bank = db.session.get(ImageBank, returned_bank_id)
        returned_file = Path(returned_bank.source_path) / returned.relpath
        assert returned.caption == 'caption edited in the dataset'
        assert returned.framing == 'face'
        assert returned.watermark_state == 'dismissed'
        assert returned.watermark_regions == '[]'
        assert json.loads(returned.source_metadata) == SOURCE_METADATA
        assert returned.status == 'keep'
        assert _row_values(returned, transfer.DETERMINISTIC_ANALYSIS_FIELDS) == {
            name: snapshot['analysis'][name]
            for name in transfer.DETERMINISTIC_ANALYSIS_FIELDS
        }
        assert _row_values(returned, transfer.MODEL_ANALYSIS_FIELDS) == {
            name: snapshot['analysis'][name]
            for name in transfer.MODEL_ANALYSIS_FIELDS
        }
        assert returned.watermark_source == 'detector'
        assert returned.watermark_score == pytest.approx(0.97)
        assert (returned.width, returned.height) == _dimensions(returned_file)
        assert (returned.dup_group, returned.semantic_dup_group,
                returned.face_cluster, returned.style_cluster) == (None, None, 1, 1)


def test_dataset_roundtrip_preserves_score_face_caches_without_quality_scan(
        app, tmp_path):
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks
    from app.services.dataset_storage import dataset_path

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        source_path = Path(source_bank.source_path) / source_image.relpath
        for name in transfer.DETERMINISTIC_ANALYSIS_FIELDS:
            setattr(source_image, name, None)
        source_image.file_size = source_path.stat().st_size
        source_image.width, source_image.height = _dimensions(source_path)
        db.session.commit()
        _seed_analysis_caches(source_bank, (source_image,))

        dataset = _promote_to_dataset(app, source_bank, source_image)
        dataset_image = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).one()
        snapshot = transfer.parse_snapshot(dataset_image.bank_analysis_snapshot)
        assert snapshot['v'] == 3 and snapshot['cache_ref']
        sidecar = (Path(dataset_path(dataset.id)) / '.bank-analysis-cache'
                   / f"{snapshot['cache_ref']}.npz")
        assert sidecar.is_file()

        returned_id = banks.start_dataset_import(
            app, 'local', dataset.id, 'Roundtrip cached')
        returned_bank = db.session.get(ImageBank, returned_id)
        returned = BankImage.query.filter_by(bank_id=returned_id).one()
        assert returned.face_state == 'scorable'
        assert returned.face_det == pytest.approx(0.91)
        assert returned.aesthetic_score == pytest.approx(7.2)
        assert returned.nsfw_score == pytest.approx(0.03)
        cache = _runtime_cache_payloads(returned_bank, (returned,))
        returned_path = str(Path(returned_bank.source_path) / returned.relpath)
        assert set(cache[returned_path]) == {'score', 'face'}
        assert len(cache[returned_path]['score'][0]['embedding']) == 768
        assert len(cache[returned_path]['face'][0]['embedding']) == 512
        expected_hash = hashlib.sha256(Path(returned_path).read_bytes()).digest()
        assert cache[returned_path]['score'][2] == expected_hash
        assert cache[returned_path]['face'][2] == expected_hash


def test_siglip2_bank_dataset_roundtrip_preserves_all_cache_lanes_and_engine(
        app, tmp_path):
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks
    from app.services.dataset_storage import dataset_path

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        source_bank.semantic_engine = 'siglip2'
        db.session.commit()
        _seed_analysis_caches_with_semantic(source_bank, (source_image,))

        dataset = _promote_to_dataset(app, source_bank, source_image)
        dataset_image = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).one()
        metadata = transfer.parse_transfer_metadata(dataset_image.transfer_metadata)
        assert metadata['bank']['semantic_engine'] == 'siglip2'
        snapshot = transfer.parse_snapshot(dataset_image.bank_analysis_snapshot)
        sidecar_root = Path(dataset_path(dataset.id)) / '.bank-analysis-cache'
        sidecar = transfer.read_cache_sidecar(sidecar_root, snapshot['cache_ref'])
        assert set(sidecar) == {'score', 'semantic', 'face'}

        returned_id = banks.start_dataset_import(
            app, 'local', dataset.id, 'SigLIP2 roundtrip')
        returned_bank = db.session.get(ImageBank, returned_id)
        returned = BankImage.query.filter_by(bank_id=returned_id).one()
        assert returned_bank.semantic_engine == 'siglip2'
        runtime = _runtime_cache_payloads_with_semantic(returned_bank, (returned,))
        returned_path = str(Path(returned_bank.source_path) / returned.relpath)
        assert set(runtime[returned_path]) == {'score', 'semantic', 'face'}


@pytest.mark.parametrize('metadata_mode', ('missing', 'incompatible'))
def test_dataset_to_bank_defaults_clip_without_exact_engine_metadata(
        app, tmp_path, metadata_mode):
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        peer = _add_group_peer(source_bank)
        source_bank.semantic_engine = 'siglip2'
        db.session.commit()
        _seed_analysis_caches_with_semantic(source_bank, (source_image, peer))
        dataset = datasets.create_dataset('local', 'Ambiguous', 'ambiguous')
        banks.start_promote(
            app, 'local', source_bank.id, [source_image.id, peer.id], dataset.id)
        dataset_images = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).all()
        assert len(dataset_images) == 2
        for dataset_image in dataset_images:
            if metadata_mode == 'missing':
                dataset_image.transfer_metadata = None
            else:
                dataset_image.transfer_metadata = transfer.capture_transfer_metadata(
                    bank={'semantic_engine': 'siglip2'},
                    bank_fingerprint='0' * 64)
        db.session.commit()

        returned_id = banks.start_dataset_import(
            app, 'local', dataset.id, f'Default CLIP {metadata_mode}')
        assert db.session.get(ImageBank, returned_id).semantic_engine == 'clip'
        returned = BankImage.query.filter_by(bank_id=returned_id).all()
        assert len(returned) == 2
        assert all(row.semantic_dup_group == row.clip_semantic_dup_group
                   for row in returned)
        runtime = _runtime_cache_payloads_with_semantic(
            db.session.get(ImageBank, returned_id), returned)
        assert all(set(bundle) == {'score', 'semantic', 'face'}
                   for bundle in runtime.values())


@pytest.mark.parametrize('selected_engine', ('clip', 'siglip2'))
def test_bank_to_bank_copies_selected_engine_and_all_runtime_caches(
        app, tmp_path, selected_engine):
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        source_bank.semantic_engine = selected_engine
        db.session.commit()
        _seed_analysis_caches_with_semantic(source_bank, (source_image,))

        destination_id = banks.start_bank_promote(
            app, 'local', source_bank.id, [source_image.id], 'SigLIP2 copy')
        destination = db.session.get(ImageBank, destination_id)
        copied = BankImage.query.filter_by(bank_id=destination_id).one()
        assert destination.semantic_engine == selected_engine
        runtime = _runtime_cache_payloads_with_semantic(destination, (copied,))
        copied_path = str(Path(destination.source_path) / copied.relpath)
        assert set(runtime[copied_path]) == {'score', 'semantic', 'face'}


def test_legacy_score_only_dataset_history_restores_clip_semantic_groups(
        app, tmp_path):
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source = _bank_with_analysed_image(app, tmp_path)
        peer = _add_group_peer(source_bank)
        dataset = datasets.create_dataset('local', 'Legacy CLIP', 'legacy_clip')
        banks.start_promote(
            app, 'local', source_bank.id, [source.id, peer.id], dataset.id)
        dataset_rows = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).all()
        for row in dataset_rows:
            metadata = transfer.parse_transfer_metadata(row.transfer_metadata)
            metadata['bank']['semantic_engine'] = None
            row.transfer_metadata = json.dumps(
                metadata, ensure_ascii=False, separators=(',', ':'), sort_keys=True)
        db.session.commit()

        returned_id = banks.start_dataset_import(
            app, 'local', dataset.id, 'Legacy score-only return')
        returned_bank = db.session.get(ImageBank, returned_id)
        returned = BankImage.query.filter_by(bank_id=returned_id).all()
        assert returned_bank.semantic_engine == 'clip'
        assert len(returned) == 2
        assert {row.semantic_dup_group for row in returned} == {1}
        runtime = _runtime_cache_payloads_with_semantic(returned_bank, returned)
        assert all(set(bundle) == {'score', 'face'} for bundle in runtime.values())


def test_mixed_engine_dataset_history_projects_default_lane_without_losing_either(
        app, tmp_path):
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks

    with app.app_context():
        clip_bank, clip_image = _bank_with_analysed_image(
            app, tmp_path / 'clip-source', filename='clip-a.jpg')
        clip_peer = _add_group_peer(clip_bank, filename='clip-b.jpg')
        siglip_bank, siglip_image = _bank_with_analysed_image(
            app, tmp_path / 'siglip-source', filename='siglip-a.jpg')
        siglip_peer = _add_group_peer(siglip_bank, filename='siglip-b.jpg')
        siglip_bank.semantic_engine = 'siglip2'
        db.session.commit()
        _seed_analysis_caches_with_semantic(
            siglip_bank, (siglip_image, siglip_peer))

        dataset = datasets.create_dataset('local', 'Mixed spaces', 'mixed_spaces')
        banks.start_promote(
            app, 'local', clip_bank.id, [clip_image.id, clip_peer.id], dataset.id)
        banks.start_promote(
            app, 'local', siglip_bank.id, [siglip_image.id, siglip_peer.id], dataset.id)
        returned_id = banks.start_dataset_import(
            app, 'local', dataset.id, 'Mixed return')
        returned_bank = db.session.get(ImageBank, returned_id)
        returned = BankImage.query.filter_by(bank_id=returned_id).all()
        assert returned_bank.semantic_engine == 'clip'
        assert len(returned) == 4
        clip_before = {row.id: row.clip_semantic_dup_group for row in returned}
        siglip_before = {row.id: row.siglip2_semantic_dup_group for row in returned}
        assert all(row.semantic_dup_group == row.clip_semantic_dup_group
                   for row in returned)
        banks.set_semantic_engine('local', returned_id, 'siglip2')
        db.session.expire_all()
        switched = BankImage.query.filter_by(bank_id=returned_id).all()
        assert {row.id: row.clip_semantic_dup_group for row in switched} == clip_before
        assert {row.id: row.siglip2_semantic_dup_group
                for row in switched} == siglip_before
        assert all(row.semantic_dup_group == row.siglip2_semantic_dup_group
                   for row in switched)
        runtime = _runtime_cache_payloads_with_semantic(returned_bank, returned)
        assert len(runtime) == 4
        assert all({'score', 'face'}.issubset(bundle)
                   for bundle in runtime.values())
        assert sum('semantic' in bundle for bundle in runtime.values()) == 2


def test_dataset_pixel_edit_keeps_historical_vault_but_cannot_activate_it(
        app, tmp_path):
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks
    from app.services.dataset_storage import dataset_path

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        source_path = Path(source_bank.source_path) / source_image.relpath
        for name in transfer.DETERMINISTIC_ANALYSIS_FIELDS:
            setattr(source_image, name, None)
        source_image.file_size = source_path.stat().st_size
        source_image.width, source_image.height = _dimensions(source_path)
        db.session.commit()
        _seed_analysis_caches(source_bank, (source_image,))

        dataset = _promote_to_dataset(app, source_bank, source_image)
        dataset_image = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).one()
        historical_text = dataset_image.bank_analysis_snapshot
        historical = transfer.parse_snapshot(historical_text)
        cache_file = (Path(dataset_path(dataset.id)) / '.bank-analysis-cache'
                      / f"{historical['cache_ref']}.npz")
        historical_cache = cache_file.read_bytes()

        assert datasets.rotate_image('local', dataset_image.id, 90)
        db.session.expire_all()
        edited = db.session.get(FaceDatasetImage, dataset_image.id)
        edited_path = Path(dataset_path(dataset.id)) / edited.filename
        assert edited.bank_analysis_snapshot == historical_text
        assert cache_file.read_bytes() == historical_cache
        assert transfer.compatible_snapshot(historical_text, edited_path) is None
        assert edited.watermark_source is None and edited.watermark_score is None

        returned_id = banks.start_dataset_import(
            app, 'local', dataset.id, 'Edited Dataset Bank')
        returned_bank = db.session.get(ImageBank, returned_id)
        returned = BankImage.query.filter_by(bank_id=returned_id).one()
        assert _row_values(returned, transfer.DETERMINISTIC_ANALYSIS_FIELDS) == {
            name: None for name in transfer.DETERMINISTIC_ANALYSIS_FIELDS
        }
        _assert_model_analysis_is_empty(returned)
        assert _runtime_cache_payloads(returned_bank, (returned,)) == {}


def test_dataset_to_bank_discards_destination_on_incomplete_cache_write(
        app, tmp_path):
    from app.extensions import db
    from app.models import FaceDatasetImage, ImageBank
    from app.services import bank_jobs
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks
    from app.services.dataset_storage import dataset_path

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        source_path = Path(source_bank.source_path) / source_image.relpath
        for name in transfer.DETERMINISTIC_ANALYSIS_FIELDS:
            setattr(source_image, name, None)
        source_image.file_size = source_path.stat().st_size
        source_image.width, source_image.height = _dimensions(source_path)
        db.session.commit()
        _seed_analysis_caches(source_bank, (source_image,))
        dataset = _promote_to_dataset(app, source_bank, source_image)
        dataset_image = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).one()
        historical = dataset_image.bank_analysis_snapshot
        snapshot = transfer.parse_snapshot(historical)
        sidecar = (Path(dataset_path(dataset.id)) / '.bank-analysis-cache'
                   / f"{snapshot['cache_ref']}.npz")
        sidecar_bytes = sidecar.read_bytes()

        with patch.object(
                transfer, 'write_runtime_caches',
                return_value={'score': 0, 'face': 0}):
            destination_id = banks.start_dataset_import(
                app, 'local', dataset.id, 'Must be discarded')

        assert db.session.get(ImageBank, destination_id) is None
        db.session.refresh(dataset_image)
        assert dataset_image.bank_analysis_snapshot == historical
        assert sidecar.read_bytes() == sidecar_bytes
        job = bank_jobs.get(destination_id)
        assert job['finished'] and 'every Score/Face cache' in job['error']


@pytest.mark.parametrize('failure_seam', ('sidecar', 'snapshot'))
def test_bank_to_dataset_refuses_silent_analysis_cache_loss(
        app, tmp_path, failure_seam):
    from app.extensions import db
    from app.models import FaceDatasetImage
    from app.services import bank_jobs
    from app.services import bank_transfer_metadata as transfer
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks
    from app.services.dataset_storage import dataset_path

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        source_path = Path(source_bank.source_path) / source_image.relpath
        for name in transfer.DETERMINISTIC_ANALYSIS_FIELDS:
            setattr(source_image, name, None)
        source_image.file_size = source_path.stat().st_size
        source_image.width, source_image.height = _dimensions(source_path)
        db.session.commit()
        _seed_analysis_caches(source_bank, (source_image,))
        dataset = datasets.create_dataset('local', 'Failure target', 'failure')

        target = ('write_cache_sidecar' if failure_seam == 'sidecar'
                  else 'snapshot_storage')
        with patch.object(transfer, target, return_value=None):
            banks.start_promote(
                app, 'local', source_bank.id, [source_image.id], dataset.id)

        assert FaceDatasetImage.query.filter_by(dataset_id=dataset.id).count() == 0
        cache_dir = Path(dataset_path(dataset.id)) / '.bank-analysis-cache'
        assert not cache_dir.exists() or not list(cache_dir.glob('*.npz'))
        job = bank_jobs.get(source_bank.id)
        assert job['finished'] and 'rolled back' in job['error']


def test_dataset_to_bank_without_preservation_never_uses_a_snapshot_and_sets_dimensions(
        app, tmp_path):
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        dataset = _promote_to_dataset(app, source_bank, source_image)
        dataset_image = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).one()
        dataset_image.caption = 'dataset-owned caption'
        db.session.commit()

        # This is stronger than merely seeing nulls: the compatibility parser
        # is not even consulted when the caller explicitly asks for a fresh Bank.
        with patch.object(banks.bank_transfer_metadata, 'compatible_snapshot',
                          side_effect=AssertionError('snapshot must not be read')):
            returned_bank_id = banks.start_dataset_import(
                app, 'local', dataset.id, 'Fresh', preserve_analysis=False)
        returned = BankImage.query.filter_by(bank_id=returned_bank_id).one()
        returned_bank = db.session.get(ImageBank, returned_bank_id)
        returned_file = Path(returned_bank.source_path) / returned.relpath
        assert returned.caption == 'dataset-owned caption'
        assert returned.framing == 'body'
        assert returned.watermark_state == 'detected'
        assert _row_values(returned, transfer.DETERMINISTIC_ANALYSIS_FIELDS) == {
            name: None for name in transfer.DETERMINISTIC_ANALYSIS_FIELDS
        }
        _assert_model_analysis_is_empty(returned)
        assert (returned.width, returned.height) == _dimensions(returned_file)


def test_dataset_to_bank_rejects_a_same_size_middle_byte_mutation(app, tmp_path):
    """The transfer signature is a full SHA-256, not sampled first/last bytes."""
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks
    from app.services.dataset_storage import dataset_path

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        dataset = _promote_to_dataset(app, source_bank, source_image)
        dataset_image = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).one()
        dataset_file = Path(dataset_path(dataset.id)) / dataset_image.filename
        snapshot = _assert_v3_snapshot_matches_final(
            dataset_image.bank_analysis_snapshot, dataset_file)

        # Keep the image readable by changing an ignored trailing byte, but make
        # it a byte in the MIDDLE of a >8 MiB fingerprinted file and preserve
        # its size.  The old sampled implementation skipped this middle range.
        original = dataset_file.read_bytes()
        padding = b'P' * (8 * 1024 * 1024 + 17)
        signed = original + padding
        dataset_file.write_bytes(signed)
        dataset_image.bank_analysis_snapshot = json.dumps({
            'v': 2,
            'fingerprint': hashlib.sha256(signed).hexdigest(),
            'analysis': snapshot['analysis'],
        })
        db.session.commit()
        with dataset_file.open('r+b') as fh:
            fh.seek(len(original) + len(padding) // 2)
            fh.write(b'X')
        assert _dimensions(dataset_file) == (160, 96)

        returned_bank_id = banks.start_dataset_import(app, 'local', dataset.id, 'Changed')
        returned = BankImage.query.filter_by(bank_id=returned_bank_id).one()
        assert _row_values(returned, transfer.DETERMINISTIC_ANALYSIS_FIELDS) == {
            name: None for name in transfer.DETERMINISTIC_ANALYSIS_FIELDS
        }
        _assert_model_analysis_is_empty(returned)


def test_legacy_v2_snapshot_still_restores_deterministic_analysis(app, tmp_path):
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks
    from app.services.dataset_storage import dataset_path

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        dataset = _promote_to_dataset(app, source_bank, source_image)
        dataset_image = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).one()
        dataset_file = Path(dataset_path(dataset.id)) / dataset_image.filename
        deterministic = _final_deterministic_analysis(dataset_file)
        dataset_image.bank_analysis_snapshot = json.dumps({
            'v': 2,
            'fingerprint': hashlib.sha256(dataset_file.read_bytes()).hexdigest(),
            'analysis': deterministic,
        })
        db.session.commit()

        returned_id = banks.start_dataset_import(
            app, 'local', dataset.id, 'Legacy v2 restored')
        returned = BankImage.query.filter_by(bank_id=returned_id).one()
        assert _row_values(returned, transfer.DETERMINISTIC_ANALYSIS_FIELDS) == (
            deterministic)
        _assert_model_analysis_is_empty(returned)
        assert (returned.dup_group, returned.semantic_dup_group,
                returned.face_cluster, returned.style_cluster) == (
                    None, None, None, None)


def test_v1_snapshot_is_rejected_even_when_its_fingerprint_matches(app, tmp_path):
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage
    from app.services import bank_transfer_metadata as transfer
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks
    from app.services.dataset_storage import dataset_path

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        dataset = _promote_to_dataset(app, source_bank, source_image)
        dataset_image = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).one()
        dataset_file = Path(dataset_path(dataset.id)) / dataset_image.filename
        v3 = _assert_v3_snapshot_matches_final(
            dataset_image.bank_analysis_snapshot, dataset_file)
        legacy = json.dumps({
            'v': 1,
            'fingerprint': hashlib.sha256(dataset_file.read_bytes()).hexdigest(),
            'analysis': {name: v3['analysis'][name]
                         for name in transfer.DETERMINISTIC_ANALYSIS_FIELDS},
        })
        assert transfer.parse_snapshot(legacy) is None
        dataset_image.bank_analysis_snapshot = legacy
        db.session.commit()

        # Backup canonicalization must omit the rejected v1 payload too, rather
        # than shipping an opaque historical object to another installation.
        archive = datasets.build_backup_zip('local', dataset.id)
        restored = datasets.import_backup_zip('local', archive)
        restored_image = FaceDatasetImage.query.filter_by(dataset_id=restored.id).one()
        assert restored_image.bank_analysis_snapshot is None

        returned_bank_id = banks.start_dataset_import(app, 'local', dataset.id, 'Legacy')
        returned = BankImage.query.filter_by(bank_id=returned_bank_id).one()
        assert _row_values(returned, transfer.DETERMINISTIC_ANALYSIS_FIELDS) == {
            name: None for name in transfer.DETERMINISTIC_ANALYSIS_FIELDS
        }
        _assert_model_analysis_is_empty(returned)


@pytest.mark.parametrize(
    'overrides',
    (
        {'detail_ratio': -0.001},
        {'detail_ratio': 1.001},
        {'bars_ratio': -0.001},
        {'bars_ratio': 1.001},
        {'jpeg_quality': 0},
        {'jpeg_quality': 101},
        {'blur_score': -0.001},
        # The Laplacian has range [-4*255, +4*255], so its variance cannot
        # exceed (4*255)^2; see image_quality.quality_metrics.
        {'blur_score': float((4 * 255) ** 2) + 0.001},
        {'noise_score': -0.001},
        # Noise is RMS over an 8-bit difference image, bounded by 255.
        {'noise_score': 255.001},
        {'uniformity_score': -0.001},
        {'uniformity_score': 128.001},
        {'quality_state': 'unreadable'},
        {'origin': 'unknown', 'origin_evidence': 'claimed-source'},
        {'origin': 'ai', 'origin_evidence': None},
        {'origin': 'camera', 'origin_evidence': None},
    ),
    ids=(
        'detail_ratio-below-zero',
        'detail_ratio-above-one',
        'bars_ratio-below-zero',
        'bars_ratio-above-one',
        'jpeg_quality-below-one',
        'jpeg_quality-above-100',
        'blur-negative',
        'blur-above-physical-laplacian-bound',
        'noise-negative',
        'noise-above-physical-rms-bound',
        'uniformity-negative',
        'uniformity-above-128',
        'quality-state-not-ok',
        'unknown-origin-with-evidence',
        'ai-origin-without-evidence',
        'camera-origin-without-evidence',
    ),
)
def test_strict_v2_snapshot_validation_rejects_invalid_analysis_in_parser_and_backup_path(
        overrides):
    """Both direct reads and backup canonicalization reject invalid v2 claims."""
    from app.services import bank_transfer_metadata as transfer

    payload = _valid_v2_snapshot_payload()
    payload['analysis'].update(overrides)
    serialized = json.dumps(payload)
    assert transfer.parse_snapshot(serialized) is None
    # Backup import passes persisted JSON through this canonicalizer, so it must
    # have the same fail-closed policy as the public parser.
    assert transfer.normalized_snapshot_storage(serialized) is None


def test_snapshot_parser_rejects_deeply_nested_json_without_raising():
    from app.services import bank_transfer_metadata as transfer

    nested = '[' * 1500 + '0' + ']' * 1500
    payload = ('{"v":2,"fingerprint":"' + '0' * 64 +
               '","analysis":' + nested + '}')
    assert len(payload.encode('utf-8')) < 8 * 1024
    assert transfer.parse_snapshot(payload) is None


def test_bank_deterministic_analysis_skips_oversized_header_before_load(monkeypatch):
    """The local Bank-analysis guard must reject pixels before decoding them."""
    from app.services import face_dataset_service as datasets

    class OversizedImage:
        size = (datasets.BANK_ANALYSIS_MAX_SIDE,
                datasets.BANK_ANALYSIS_MAX_SIDE)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def draft(self, *_args):
            raise AssertionError('oversized header reached draft before guard')

        def load(self):
            raise AssertionError('oversized header reached load before guard')

    monkeypatch.setattr(datasets.Image, 'open', lambda *_args, **_kwargs: OversizedImage())
    assert datasets.bank_deterministic_analysis('oversized-image') is None


def test_bank_deterministic_analysis_skips_pillow_bombs_without_filter_mutation(
        monkeypatch):
    """Bomb errors/warnings degrade safely without changing global filters."""
    from app.services import face_dataset_service as datasets

    filters_before = list(warnings.filters)

    def bomb_error(*_args, **_kwargs):
        raise datasets.Image.DecompressionBombError('too many pixels')

    monkeypatch.setattr(datasets.Image, 'open', bomb_error)
    assert datasets.bank_deterministic_analysis('bomb-error') is None

    def bomb_warning(*_args, **_kwargs):
        raise datasets.Image.DecompressionBombWarning('too many pixels')

    monkeypatch.setattr(datasets.Image, 'open', bomb_warning)
    assert datasets.bank_deterministic_analysis('bomb-warning') is None
    assert warnings.filters == filters_before


def test_bank_to_bank_promotion_rejects_pillow_bomb_without_creating_a_row(
        app, tmp_path, monkeypatch):
    """A bomb is rejected safely before it becomes a destination row."""
    from app.models import BankImage
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        filters_before = list(warnings.filters)

        def bomb(*_args, **_kwargs):
            raise banks.Image.DecompressionBombError('too many pixels')

        # ``banks.Image`` and the Dataset analysis share Pillow's module, so
        # the input validation rejects the copy before it can be registered.
        monkeypatch.setattr(banks.Image, 'open', bomb)
        destination_id = banks.start_bank_promote(
            app, 'local', source_bank.id, [source_image.id], 'Bomb-safe copy')
        assert BankImage.query.filter_by(bank_id=destination_id).first() is None
        assert warnings.filters == filters_before


def test_direct_bank_to_bank_copy_preserves_all_analysis_and_curation_state(
        app, tmp_path, monkeypatch):
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        source_path = _sync_bank_row_to_current_file(source_bank, source_image)
        peer = _add_group_peer(source_bank)
        source_image.siglip2_semantic_dup_group = 18
        peer.siglip2_semantic_dup_group = 18
        db.session.commit()
        _seed_analysis_caches_with_semantic(source_bank, (source_image, peer))
        assert transfer.BANK_DIRECT_COPY_ANALYSIS_FIELDS == (
            EXPECTED_DIRECT_BANK_ANALYSIS_FIELDS)
        source_analysis = _row_values(
            source_image, EXPECTED_DIRECT_BANK_ANALYSIS_FIELDS)
        # Some valid deterministic results are optional (for example a JPEG
        # without provenance evidence); every preserved ML/group result is set.
        assert all(
            source_analysis[name] is not None
            for name in transfer.BANK_TRANSFORM_STALE_ANALYSIS_FIELDS)
        # None of this destination/curation state may leak into the new row.
        source_image.status = 'reject'
        source_image.reject_reason = 'manual'
        source_image.promoted_dataset_id = 987
        source_image.promoted_bank_id = 654
        db.session.commit()
        # Exercise the canonical-path comparison, not just equal strings: a
        # harmless alternate spelling of the raw source is still a direct copy.
        alias_dir = Path(source_bank.source_path) / 'alias'
        alias_dir.mkdir()
        monkeypatch.setattr(
            banks, 'resolved_image_path',
            lambda _bank, row: str(alias_dir / '..' / row.relpath))
        destination_id = banks.start_bank_promote(
            app, 'local', source_bank.id, [source_image.id, peer.id], 'Bank copy')
        copied = BankImage.query.filter_by(
            bank_id=destination_id, relpath=source_image.relpath).one_or_none()
        if copied is None:
            from app.services import bank_jobs
            pytest.fail(str(bank_jobs.get(source_bank.id)))
        copies = BankImage.query.filter_by(bank_id=destination_id).all()
        destination = db.session.get(ImageBank, destination_id)
        copied_file = Path(destination.source_path) / copied.relpath
        assert copied_file.read_bytes() == source_path.read_bytes()
        assert copied.caption == 'caption from the bank'
        assert copied.caption_origin == 'asserted'
        assert json.loads(copied.source_metadata) == SOURCE_METADATA
        assert _row_values(copied, EXPECTED_DIRECT_BANK_ANALYSIS_FIELDS) == (
            source_analysis)
        assert _row_values(copied, transfer.DETERMINISTIC_ANALYSIS_FIELDS) == (
            _final_deterministic_analysis(copied_file))
        assert _row_values(copied, transfer.DETERMINISTIC_ANALYSIS_FIELDS) == (
            _row_values(source_image, transfer.DETERMINISTIC_ANALYSIS_FIELDS))
        assert {(row.dup_group, row.semantic_dup_group) for row in copies} == {
            (7, 8)}
        assert {(row.clip_semantic_dup_group,
                 row.siglip2_semantic_dup_group) for row in copies} == {(8, 18)}
        assert (copied.width, copied.height) == _dimensions(copied_file)
        assert copied.status == 'reject'
        assert copied.reject_reason == 'manual'
        assert copied.promoted_dataset_id is None
        assert copied.promoted_bank_id is None
        assert copied.watermark_clean_method is None
        assert copied.rotation is None


def test_direct_bank_copy_transfers_cache_subset_and_rewrites_paths_and_signatures(
        app, tmp_path):
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, first = _bank_with_analysed_image(app, tmp_path)
        _sync_bank_row_to_current_file(source_bank, first)
        second = _add_group_peer(source_bank)
        third = _add_group_peer(source_bank, filename='three.jpg')
        db.session.commit()
        _seed_analysis_caches(source_bank, (first, second, third))

        destination_id = banks.start_bank_promote(
            app, 'local', source_bank.id, [first.id, third.id], 'Cached subset')
        destination = db.session.get(ImageBank, destination_id)
        copied = (BankImage.query.filter_by(bank_id=destination_id)
                  .order_by(BankImage.relpath).all())
        assert [row.relpath for row in copied] == ['one.jpg', 'three.jpg']
        cache = _runtime_cache_payloads(destination, copied)
        expected_paths = {
            str(Path(destination.source_path) / row.relpath) for row in copied
        }
        assert set(cache) == expected_paths
        for path in expected_paths:
            assert set(cache[path]) == {'score', 'face'}
            for kind in ('score', 'face'):
                payload, signature, digest = cache[path][kind]
                assert signature == transfer.runtime_stat_signature(path)
                assert digest == hashlib.sha256(Path(path).read_bytes()).digest()
                assert payload['state'] in ('ok', 'scorable')


def test_legacy_runtime_caches_without_hashes_are_never_transferred(app, tmp_path):
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks
    from app.services.dataset_storage import dataset_path

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        peer = _add_group_peer(source_bank)
        source_path = Path(source_bank.source_path) / source_image.relpath
        for name in transfer.DETERMINISTIC_ANALYSIS_FIELDS:
            setattr(source_image, name, None)
        source_image.file_size = source_path.stat().st_size
        source_image.width, source_image.height = _dimensions(source_path)
        db.session.commit()
        _seed_analysis_caches(source_bank, (source_image,))
        _drop_runtime_cache_hashes(banks._score_cache_path(source_bank.id))
        _drop_runtime_cache_hashes(banks._face_cache_path(source_bank.id))

        source_cache = _runtime_cache_payloads(source_bank, (source_image,))
        source_key = str(source_path)
        assert source_cache[source_key]['score'][2] == b''
        assert source_cache[source_key]['face'][2] == b''

        destination_id = banks.start_bank_promote(
            app, 'local', source_bank.id, [source_image.id, peer.id],
            'Legacy cache copy')
        destination = db.session.get(ImageBank, destination_id)
        assert destination is not None
        copied = BankImage.query.filter_by(bank_id=destination_id).all()
        sources = {row.relpath: row for row in (source_image, peer)}
        assert len(copied) == 2
        for row in copied:
            assert _row_values(row, transfer.BANK_DIRECT_COPY_ANALYSIS_FIELDS) == (
                _row_values(sources[row.relpath],
                            transfer.BANK_DIRECT_COPY_ANALYSIS_FIELDS))
            assert row.analysis_fingerprint == hashlib.sha256(
                (Path(destination.source_path) / row.relpath).read_bytes()).hexdigest()
        assert _runtime_cache_payloads(destination, copied) == {}

        dataset = datasets.create_dataset('local', 'Legacy cache Dataset', 'legacy_cache')
        banks.start_promote(
            app, 'local', source_bank.id, [source_image.id], dataset.id)
        dataset_row = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).one()
        snapshot = transfer.parse_snapshot(dataset_row.bank_analysis_snapshot)
        assert snapshot['analysis']['aesthetic_score'] == source_image.aesthetic_score
        assert snapshot['analysis']['face_cluster'] == source_image.face_cluster
        assert snapshot['cache_ref'] is None
        cache_dir = Path(dataset_path(dataset.id)) / '.bank-analysis-cache'
        assert not cache_dir.exists() or not list(cache_dir.glob('*.npz'))


def test_complete_legacy_quality_carries_face_scalars_without_an_old_face_hash(
        app, tmp_path):
    """An old unhashed Face cache must not discard a proven promotion.

    The scalar face measurements remain portable through the narrow legacy
    TOFU gate; the unverifiable Face embedding itself deliberately does not.
    """
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source = _bank_with_analysed_image(app, tmp_path)
        source.analysis_fingerprint = None
        db.session.commit()
        _drop_runtime_cache_hashes(banks._face_cache_path(source_bank.id))

        destination_id = banks.start_bank_promote(
            app, 'local', source_bank.id, [source.id], 'Legacy face copy')
        destination = db.session.get(ImageBank, destination_id)
        assert destination is not None
        copied = BankImage.query.filter_by(bank_id=destination_id).one()
        carried = (
            'face_state', 'face_det', 'face_yaw',
            'face_cluster', 'face_cluster_origin',
            'aesthetic_score', 'nsfw_score', 'style_cluster',
        )
        for name in carried:
            assert getattr(copied, name) == getattr(source, name), name
        assert copied.analysis_fingerprint is None
        copied_cache = _runtime_cache_payloads(destination, (copied,))
        copied_path = str(Path(destination.source_path) / copied.relpath)
        assert set(copied_cache[copied_path]) == {'score'}

        dataset = datasets.create_dataset(
            'local', 'Legacy face Dataset', 'legacy_face_dataset')
        banks.start_promote(
            app, 'local', source_bank.id, [source.id], dataset.id)
        dataset_row = FaceDatasetImage.query.filter_by(
            dataset_id=dataset.id).one()
        snapshot = transfer.parse_snapshot(dataset_row.bank_analysis_snapshot)
        assert snapshot is not None
        assert snapshot['assurance'] == 'legacy_tofu'
        for name in carried:
            assert snapshot['analysis'][name] == getattr(source, name), name

        returned_id = banks.start_dataset_import(
            app, 'local', dataset.id, 'Legacy face returned')
        returned_bank = db.session.get(ImageBank, returned_id)
        assert returned_bank is not None
        returned = BankImage.query.filter_by(bank_id=returned_id).one()
        for name in ('face_state', 'face_det', 'face_yaw',
                     'aesthetic_score', 'nsfw_score'):
            assert getattr(returned, name) == getattr(source, name), name
        assert returned.face_cluster is not None
        assert returned.face_cluster_origin == 'asserted'
        assert returned.style_cluster is not None
        assert returned.analysis_fingerprint is None
        returned_cache = _runtime_cache_payloads(returned_bank, (returned,))
        returned_path = str(Path(returned_bank.source_path) / returned.relpath)
        assert set(returned_cache[returned_path]) == {'score'}


def test_same_size_mtime_replacement_cannot_reuse_hashed_runtime_cache(
        app, tmp_path):
    from app.extensions import db
    from app.models import ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        source_path = Path(source_bank.source_path) / source_image.relpath
        source_path.write_bytes(source_path.read_bytes() + b'A' * 64)
        for name in transfer.DETERMINISTIC_ANALYSIS_FIELDS:
            setattr(source_image, name, None)
        source_image.file_size = source_path.stat().st_size
        source_image.width, source_image.height = _dimensions(source_path)
        db.session.commit()
        _seed_analysis_caches(source_bank, (source_image,))
        old_signature = transfer.runtime_stat_signature(source_path)
        old_stat = source_path.stat()

        replacement = bytearray(source_path.read_bytes())
        replacement[-1] = ord('B')
        source_path.write_bytes(replacement)
        os.utime(source_path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
        assert transfer.runtime_stat_signature(source_path) == old_signature

        destination_id = banks.start_bank_promote(
            app, 'local', source_bank.id, [source_image.id], 'Same-stat copy')
        assert db.session.get(ImageBank, destination_id) is None


def test_direct_bank_copy_discards_destination_on_incomplete_cache_write(
        app, tmp_path):
    from app.extensions import db
    from app.models import ImageBank
    from app.services import bank_jobs
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        _sync_bank_row_to_current_file(source_bank, source_image)
        db.session.commit()
        _seed_analysis_caches(source_bank, (source_image,))

        with patch.object(
                transfer, 'write_runtime_caches',
                return_value={'score': 0, 'face': 0}):
            destination_id = banks.start_bank_promote(
                app, 'local', source_bank.id, [source_image.id],
                'Must be discarded')

        assert db.session.get(ImageBank, destination_id) is None
        db.session.refresh(source_image)
        assert source_image.promoted_bank_id is None
        job = bank_jobs.get(source_bank.id)
        assert job['finished'] and 'every Score/Face cache' in job['error']


def test_same_path_replacement_invalidates_source_analysis(app, tmp_path):
    from app.extensions import db
    from app.models import ImageBank
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        source_path = _sync_bank_row_to_current_file(source_bank, source_image)
        db.session.commit()
        # Same canonical path and dimensions, different bytes after the scan.
        _write_photo(source_path, quality=92)

        destination_id = banks.start_bank_promote(
            app, 'local', source_bank.id, [source_image.id], 'Replaced copy')
        assert db.session.get(ImageBank, destination_id) is None


@pytest.mark.parametrize('marker', ('missing_clean', 'rotation_fallback'))
def test_unmaterialized_transform_marker_invalidates_source_analysis(
        app, tmp_path, monkeypatch, marker):
    from app.extensions import db
    from app.models import ImageBank
    from app.services import bank_jobs
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        source_path = _sync_bank_row_to_current_file(source_bank, source_image)
        if marker == 'missing_clean':
            source_image.watermark_clean_method = 'crop'
            assert not banks.clean_image_path(
                source_bank.id, source_image.id).exists()
        else:
            source_image.rotation = 90
            monkeypatch.setattr(
                banks, '_ensure_rotated',
                lambda _bank_id, _row, source: source)
        db.session.commit()
        assert banks._same_resolved_path(
            banks.resolved_image_path(source_bank, source_image), source_path)

        destination_id = banks.start_bank_promote(
            app, 'local', source_bank.id, [source_image.id], f'{marker} copy')
        # Display may fail open to the pristine file, but analysis transport may
        # never bless that fallback as the requested clean/rotation.  Promotion
        # is all-or-nothing, so the empty destination is unmade as well.
        assert db.session.get(ImageBank, destination_id) is None
        job = bank_jobs.get(source_bank.id)
        assert job['finished'] and 'discarded' in job['error']


@pytest.mark.parametrize(
    ('scenario', 'expected_count', 'expected_duplicate_groups'),
    (('single_selection', 1, {(None, None)}),
     ('unreadable_peer', 0, set()),
     ('two_members', 2, {(7, 8)})),
)
def test_bank_copy_removes_only_singleton_duplicate_groups(
        app, tmp_path, scenario, expected_count, expected_duplicate_groups):
    from app.extensions import db
    from app.models import BankImage
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, first = _bank_with_analysed_image(app, tmp_path)
        _sync_bank_row_to_current_file(source_bank, first)
        peer = _add_group_peer(source_bank)
        db.session.commit()
        selected = ([first.id] if scenario == 'single_selection'
                    else [first.id, peer.id])
        if scenario == 'unreadable_peer':
            (Path(source_bank.source_path) / peer.relpath).unlink()

        destination_id = banks.start_bank_promote(
            app, 'local', source_bank.id, selected, f'{scenario} copy')
        copied = BankImage.query.filter_by(bank_id=destination_id).all()

        assert len(copied) == expected_count
        assert {(row.dup_group, row.semantic_dup_group) for row in copied} == (
            expected_duplicate_groups)
        # These clusters remain useful row classifications even with one member.
        expected_clusters = {(4, 'asserted', 3)} if copied else set()
        assert {(row.face_cluster, row.face_cluster_origin, row.style_cluster)
                for row in copied} == expected_clusters


def test_dataset_roundtrip_scopes_identical_group_ids_from_separate_promotions(
        app, tmp_path):
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage
    from app.services import bank_transfer_metadata as transfer
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks

    with app.app_context():
        first_bank, first = _bank_with_analysed_image(
            app, tmp_path, filename='first.jpg')
        second_bank, second = _bank_with_analysed_image(
            app, tmp_path, filename='second.jpg')
        second_path = Path(second_bank.source_path) / second.relpath
        distinct = Image.new('RGB', (173, 109))
        pixels = distinct.load()
        for y in range(109):
            for x in range(173):
                tile = ((x // 9) + (y // 7)) % 2
                pixels[x, y] = ((240 if tile else 8),
                                (20 if tile else 220), (x * y) % 251)
        distinct.save(second_path, 'JPEG', quality=83)
        _sync_bank_row_to_current_file(first_bank, first)
        _sync_bank_row_to_current_file(second_bank, second)
        db.session.commit()
        _seed_analysis_caches(first_bank, (first,))
        _seed_analysis_caches(second_bank, (second,))

        dataset = datasets.create_dataset('local', 'Two scopes', 'two_scopes')
        banks.start_promote(app, 'local', first_bank.id, [first.id], dataset.id)
        banks.start_promote(app, 'local', second_bank.id, [second.id], dataset.id)
        dataset_rows = (FaceDatasetImage.query.filter_by(dataset_id=dataset.id)
                        .order_by(FaceDatasetImage.id).all())
        assert len(dataset_rows) == 2
        snapshots = [transfer.parse_snapshot(row.bank_analysis_snapshot)
                     for row in dataset_rows]
        assert len({snapshot['group_scope'] for snapshot in snapshots}) == 2
        assert {snapshot['analysis']['face_cluster'] for snapshot in snapshots} == {4}

        returned_id = banks.start_dataset_import(
            app, 'local', dataset.id, 'Two scoped groups')
        returned = BankImage.query.filter_by(bank_id=returned_id).all()
        # Same old numeric ids from unrelated Banks never merge.  Duplicate
        # relations become singletons and are removed; classifications remain.
        assert {(row.dup_group, row.semantic_dup_group) for row in returned} == {
            (None, None)}
        assert {(row.face_cluster, row.face_cluster_origin, row.style_cluster)
                for row in returned} == {
                    (1, 'asserted', 1), (2, 'asserted', 2)}


@pytest.mark.parametrize(
    ('transform', 'expected_size'),
    (('rotation', (96, 160)), ('clean', (83, 57))),
)
def test_transformed_bank_to_bank_copy_recalculates_deterministic_analysis_only(
        app, tmp_path, transform, expected_size):
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        source_image.status = 'reject'
        source_image.reject_reason = 'manual'
        source_image.promoted_dataset_id = 987
        source_image.promoted_bank_id = 654
        db.session.commit()
        assert _make_source_transform(source_bank, source_image, transform) == expected_size
        destination_id = banks.start_bank_promote(
            app, 'local', source_bank.id, [source_image.id], f'{transform} copy')
        copied = BankImage.query.filter_by(bank_id=destination_id).one()
        destination = db.session.get(ImageBank, destination_id)
        copied_file = Path(destination.source_path) / copied.relpath
        assert _dimensions(copied_file) == expected_size
        assert (copied.width, copied.height) == expected_size
        assert _row_values(copied, transfer.DETERMINISTIC_ANALYSIS_FIELDS) == (
            _final_deterministic_analysis(copied_file))
        assert _row_values(copied, transfer.DETERMINISTIC_ANALYSIS_FIELDS) != (
            _row_values(source_image, transfer.DETERMINISTIC_ANALYSIS_FIELDS))
        assert transfer.BANK_TRANSFORM_STALE_ANALYSIS_FIELDS == tuple(
            name for name in EXPECTED_DIRECT_BANK_ANALYSIS_FIELDS
            if name not in transfer.DETERMINISTIC_ANALYSIS_FIELDS)
        assert _row_values(
            copied, transfer.BANK_TRANSFORM_STALE_ANALYSIS_FIELDS) == {
                name: None
                for name in transfer.BANK_TRANSFORM_STALE_ANALYSIS_FIELDS
            }
        _assert_model_analysis_is_empty(copied)
        assert copied.status == 'reject'
        assert copied.reject_reason == 'manual'
        assert copied.promoted_dataset_id is None
        assert copied.promoted_bank_id is None
        assert copied.caption == 'caption from the bank'
        assert copied.caption_origin == 'asserted'
        assert json.loads(copied.source_metadata) == SOURCE_METADATA
        assert copied.watermark_clean_method is None
        assert copied.rotation is None


@pytest.mark.parametrize('transform', ('rotation', 'clean'))
def test_reanalysed_transform_preserves_effective_lanes_and_raw_watermark_history(
        app, tmp_path, transform):
    """Once passes rerun, their shared SHA describes the baked payload and every
    effective lane/cache can cross both destinations. Watermark geometry remains
    separately bound to the pristine raw source and is historical, not actionable,
    on that baked payload."""
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage, ImageBank
    from app.services import bank_jobs
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks
    from app.services.dataset_storage import dataset_path

    with app.app_context():
        source_bank, source = _bank_with_analysed_image(app, tmp_path)
        raw_fingerprint = source.watermark_fingerprint
        _make_source_transform(source_bank, source, transform)
        effective = _bind_complete_analysis_to_effective_bytes(source_bank, source)
        effective_fingerprint = hashlib.sha256(effective.read_bytes()).hexdigest()
        assert source.analysis_fingerprint == effective_fingerprint
        assert raw_fingerprint != effective_fingerprint
        source_view = banks._image_dict(source, banks.thresholds())
        assert source_view['watermark_state'] == 'detected'
        assert 'watermark' in source_view['flags']

        copied_bank_id = banks.start_bank_promote(
            app, 'local', source_bank.id, [source.id], f'Reanalysed {transform}')
        copied_bank = db.session.get(ImageBank, copied_bank_id)
        assert copied_bank is not None, bank_jobs.get(source_bank.id)
        copied = BankImage.query.filter_by(bank_id=copied_bank_id).one()
        copied_path = Path(copied_bank.source_path) / copied.relpath
        assert copied_path.read_bytes() == effective.read_bytes()
        preserved = tuple(
            name for name in transfer.BANK_DIRECT_COPY_ANALYSIS_FIELDS
            if name not in transfer.BANK_COPY_DUPLICATE_GROUP_FIELDS)
        assert _row_values(copied, preserved) == _row_values(source, preserved)
        assert (copied.dup_group, copied.semantic_dup_group) == (None, None)
        assert copied.analysis_fingerprint == effective_fingerprint
        assert copied.watermark_fingerprint == raw_fingerprint
        copied_view = banks._image_dict(copied, banks.thresholds())
        assert copied_view['watermark_state'] is None
        assert copied_view['watermark_source'] is None
        assert 'watermark' not in copied_view['flags']
        assert banks._clean_pool_query(copied_bank_id).count() == 0
        assert banks._watermark_scan_query(
            copied_bank_id, rescan=False).count() == 1
        copied_cache = _runtime_cache_payloads(copied_bank, (copied,))
        assert set(copied_cache[str(copied_path)]) == {'score', 'face'}

        dataset = _promote_to_dataset(
            app, source_bank, source, name=f'Dataset {transform}')
        dataset_row = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).one()
        dataset_file = Path(dataset_path(dataset.id)) / dataset_row.filename
        snapshot = transfer.parse_snapshot(dataset_row.bank_analysis_snapshot)
        assert snapshot['fingerprint'] == effective_fingerprint
        assert snapshot['watermark_fingerprint'] == raw_fingerprint
        assert snapshot['analysis']['watermark_state'] == 'detected'
        # The Dataset's visible watermark lane is tied to its own baked pixels;
        # raw-source geometry stays only in the sealed historical snapshot.
        assert dataset_row.watermark_state is None
        assert dataset_row.watermark_bbox is None
        assert dataset_file.read_bytes() == effective.read_bytes()

        returned_id = banks.start_dataset_import(
            app, 'local', dataset.id, f'Returned {transform}')
        returned_bank = db.session.get(ImageBank, returned_id)
        returned = BankImage.query.filter_by(bank_id=returned_id).one()
        returned_path = Path(returned_bank.source_path) / returned.relpath
        returned_preserved = tuple(
            name for name in preserved
            if name not in transfer.BANK_LOCAL_GROUP_FIELDS)
        assert _row_values(returned, returned_preserved) == _row_values(
            source, returned_preserved)
        assert (returned.face_cluster, returned.face_cluster_origin,
                returned.style_cluster) == (1, 'asserted', 1)
        assert returned.analysis_fingerprint == effective_fingerprint
        assert returned.watermark_fingerprint == raw_fingerprint
        returned_view = banks._image_dict(returned, banks.thresholds())
        assert returned_view['watermark_state'] is None
        assert 'watermark' not in returned_view['flags']
        assert banks._clean_pool_query(returned_id).count() == 0
        assert banks._watermark_scan_query(
            returned_id, rescan=False).count() == 1
        returned_cache = _runtime_cache_payloads(returned_bank, (returned,))
        assert set(returned_cache[str(returned_path)]) == {'score', 'face'}


def test_from_dataset_route_defaults_analysis_to_true_and_accepts_false(client, app):
    from app.services import image_bank_service as banks

    with patch.object(banks, 'start_dataset_import', return_value=42) as start:
        response = client.post('/api/bank/from-dataset', json={
            'dataset_id': 3, 'name': 'Copy', 'preserve_analysis': False,
        })
    assert response.status_code == 202
    assert start.call_args.kwargs['preserve_analysis'] is False
    with patch.object(banks, 'start_dataset_import', return_value=43) as start:
        response = client.post('/api/bank/from-dataset', json={
            'dataset_id': 3, 'name': 'Default copy',
        })
    assert response.status_code == 202
    assert start.call_args.kwargs['preserve_analysis'] is True
    assert client.post('/api/bank/from-dataset', json={
        'dataset_id': 3, 'name': 'Copy', 'preserve_analysis': 'false',
    }).status_code == 400


def test_dataset_backup_v2_roundtrip_keeps_v3_snapshot_and_cache_sidecar(
        app, tmp_path):
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks
    from app.services.dataset_storage import dataset_path

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        source_path = Path(source_bank.source_path) / source_image.relpath
        for name in transfer.DETERMINISTIC_ANALYSIS_FIELDS:
            setattr(source_image, name, None)
        source_image.file_size = source_path.stat().st_size
        source_image.width, source_image.height = _dimensions(source_path)
        db.session.commit()
        _seed_analysis_caches(source_bank, (source_image,))
        dataset = _promote_to_dataset(app, source_bank, source_image)
        original = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).one()
        original_snapshot = transfer.parse_snapshot(original.bank_analysis_snapshot)
        assert original_snapshot['cache_ref']
        archive = datasets.build_backup_zip('local', dataset.id)
        with zipfile.ZipFile(io.BytesIO(archive)) as backup:
            manifest = json.loads(backup.read('manifest.json'))
            assert manifest['version'] == 2
            assert (f"analysis-cache/{original_snapshot['cache_ref']}.npz"
                    in backup.namelist())
        restored = datasets.import_backup_zip('local', archive)
        restored_image = FaceDatasetImage.query.filter_by(dataset_id=restored.id).one()
        restored_snapshot = transfer.parse_snapshot(restored_image.bank_analysis_snapshot)
        assert restored_snapshot == original_snapshot
        restored_cache_dir = Path(dataset_path(restored.id)) / '.bank-analysis-cache'
        assert set(transfer.read_cache_sidecar(
            restored_cache_dir, restored_snapshot['cache_ref'])) == {'score', 'face'}

        returned_id = banks.start_dataset_import(
            app, 'local', restored.id, 'Restored backup Bank')
        returned_bank = db.session.get(ImageBank, returned_id)
        returned = BankImage.query.filter_by(bank_id=returned_id).one()
        cache = _runtime_cache_payloads(returned_bank, (returned,))
        returned_path = str(Path(returned_bank.source_path) / returned.relpath)
        assert set(cache[returned_path]) == {'score', 'face'}


def _cached_dataset_backup_fixture(app, tmp_path, name='Backup strict'):
    """Return (dataset, row, cache_dir, cache_ref, valid_archive)."""
    from app.extensions import db
    from app.models import FaceDatasetImage
    from app.services import bank_transfer_metadata as transfer
    from app.services import face_dataset_service as datasets
    from app.services.dataset_storage import dataset_path

    source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
    source_path = Path(source_bank.source_path) / source_image.relpath
    for field in transfer.DETERMINISTIC_ANALYSIS_FIELDS:
        setattr(source_image, field, None)
    source_image.file_size = source_path.stat().st_size
    source_image.width, source_image.height = _dimensions(source_path)
    db.session.commit()
    _seed_analysis_caches(source_bank, (source_image,))
    dataset = _promote_to_dataset(app, source_bank, source_image, name=name)
    row = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).one()
    snapshot = transfer.parse_snapshot(row.bank_analysis_snapshot)
    cache_ref = snapshot['cache_ref']
    assert transfer.is_content_addressed_cache_ref(cache_ref)
    cache_dir = Path(dataset_path(dataset.id)) / '.bank-analysis-cache'
    archive = datasets.build_backup_zip('local', dataset.id)
    return dataset, row, cache_dir, cache_ref, archive


@pytest.mark.parametrize(
    'corruption', ('missing', 'oversize', 'malformed', 'digest', 'legacy_ref'))
def test_backup_v2_export_refuses_an_unbound_or_invalid_declared_sidecar(
        app, tmp_path, corruption):
    from app.extensions import db
    from app.services import bank_transfer_metadata as transfer
    from app.services import face_dataset_service as datasets

    with app.app_context():
        dataset, row, cache_dir, cache_ref, _archive = (
            _cached_dataset_backup_fixture(app, tmp_path))
        sidecar = cache_dir / f'{cache_ref}.npz'
        if corruption == 'missing':
            sidecar.unlink()
        elif corruption == 'oversize':
            sidecar.write_bytes(
                b'X' * (transfer.CACHE_SIDECAR_MAX_BYTES + 1))
        elif corruption == 'malformed':
            sidecar.write_bytes(b'not-an-npz')
        elif corruption == 'digest':
            bundle = transfer.read_cache_sidecar(cache_dir, cache_ref)
            bundle['score']['aesthetic'] += 0.5
            other_ref = transfer.write_cache_sidecar(cache_dir, bundle)
            sidecar.write_bytes((cache_dir / f'{other_ref}.npz').read_bytes())
        else:
            snapshot = transfer.parse_snapshot(row.bank_analysis_snapshot)
            legacy_ref = 'a' * 32
            snapshot['cache_ref'] = legacy_ref
            row.bank_analysis_snapshot = transfer.normalized_snapshot_storage(
                snapshot)
            (cache_dir / f'{legacy_ref}.npz').write_bytes(sidecar.read_bytes())
            db.session.commit()

        output = io.BytesIO()
        with pytest.raises(ValueError, match='analysis cache|Bank analysis cache'):
            datasets.write_backup_zip('local', dataset.id, output)
        # Validation happens before ZipFile writes even its central directory.
        assert output.getvalue() == b''


@pytest.mark.parametrize(
    'corruption',
    ('missing', 'duplicate', 'invalid_name', 'oversize', 'malformed',
     'digest', 'legacy_ref'),
)
def test_backup_v2_import_rejects_every_invalid_declared_sidecar_atomically(
        app, tmp_path, corruption):
    from app.extensions import db
    from app.models import FaceDataset
    from app.services import bank_transfer_metadata as transfer
    from app.services import face_dataset_service as datasets
    from app import config as cfg

    with app.app_context():
        _dataset, _row, cache_dir, cache_ref, archive = (
            _cached_dataset_backup_fixture(app, tmp_path))
        cache_name = f'analysis-cache/{cache_ref}.npz'
        with zipfile.ZipFile(io.BytesIO(archive)) as source:
            entries = [(info.filename, source.read(info))
                       for info in source.infolist() if not info.is_dir()]

        if corruption == 'missing':
            entries = [(name, raw) for name, raw in entries
                       if name != cache_name]
        elif corruption == 'duplicate':
            payload = next(raw for name, raw in entries if name == cache_name)
            entries.append((cache_name, payload))
        elif corruption == 'invalid_name':
            entries = [
                ('analysis-cache/../bad.npz' if name == cache_name else name, raw)
                for name, raw in entries
            ]
        elif corruption == 'oversize':
            entries = [
                (name, b'X' * (transfer.CACHE_SIDECAR_MAX_BYTES + 1)
                 if name == cache_name else raw)
                for name, raw in entries
            ]
        elif corruption == 'malformed':
            entries = [(name, b'not-an-npz' if name == cache_name else raw)
                       for name, raw in entries]
        elif corruption == 'digest':
            bundle = transfer.read_cache_sidecar(cache_dir, cache_ref)
            bundle['score']['aesthetic'] += 0.75
            other_ref = transfer.write_cache_sidecar(cache_dir, bundle)
            other_payload = (cache_dir / f'{other_ref}.npz').read_bytes()
            entries = [(name, other_payload if name == cache_name else raw)
                       for name, raw in entries]
        else:
            legacy_ref = 'b' * 32
            rewritten = []
            for name, raw in entries:
                if name == 'images.json':
                    metadata = json.loads(raw)
                    snapshot = json.loads(metadata[0]['bank_analysis_snapshot'])
                    snapshot['cache_ref'] = legacy_ref
                    metadata[0]['bank_analysis_snapshot'] = json.dumps(snapshot)
                    raw = json.dumps(metadata).encode()
                elif name == cache_name:
                    name = f'analysis-cache/{legacy_ref}.npz'
                rewritten.append((name, raw))
            entries = rewritten

        poisoned = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(poisoned, 'w', zipfile.ZIP_STORED) as target:
                for name, raw in entries:
                    target.writestr(name, raw)

        before_count = FaceDataset.query.count()
        root = Path(cfg.dataset_images_root())
        before_staging = set(root.glob('.restore-*.tmp'))
        with pytest.raises(ValueError, match='analysis cache|Bank analysis'):
            datasets.import_backup_zip('local', poisoned.getvalue())
        db.session.rollback()
        assert FaceDataset.query.count() == before_count
        assert set(root.glob('.restore-*.tmp')) == before_staging


def test_duplicate_attach_commit_failure_leaves_no_orphan_analysis_sidecar(
        app, tmp_path, monkeypatch):
    from app.extensions import db
    from app.models import FaceDatasetImage
    from app.services import bank_transfer_metadata as transfer
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks
    from app.services.dataset_storage import dataset_path

    with app.app_context():
        source_bank, source = _bank_with_analysed_image(app, tmp_path)
        source_path = Path(source_bank.source_path) / source.relpath
        raw = source_path.read_bytes()
        dataset = datasets.create_dataset('local', 'Duplicate fault', 'dup')
        created, failed = datasets.import_images(
            'local', dataset.id, [raw], dedupe=True,
            preserve_exact_bytes=True)
        assert len(created) == 1 and failed == 0
        existing = db.session.get(FaceDatasetImage, created[0])
        assert existing.bank_analysis_snapshot is None

        cache_bundle = {
            'score': {
                'state': 'ok', 'aesthetic': 7.0, 'nsfw': 0.05,
                'embedding': (0.001,) * 768,
            },
        }
        captured = transfer.captured_bank_analysis(
            banks._bank_row_analysis(source), raw,
            group_scope='c' * 32, cache_bundle=cache_bundle,
            watermark_fingerprint=source.watermark_fingerprint)
        assert captured is not None
        real_commit = db.session.commit
        calls = {'count': 0}

        def fail_attach_commit():
            calls['count'] += 1
            raise RuntimeError('fault injected before duplicate attach commit')

        monkeypatch.setattr(db.session, 'commit', fail_attach_commit)
        with pytest.raises(RuntimeError, match='fault injected'):
            datasets.import_images(
                'local', dataset.id, [raw], dedupe=True,
                preserve_exact_bytes=True,
                bank_image_ids=[source.id],
                bank_analysis_snapshots=[captured])
        assert calls['count'] == 1
        monkeypatch.setattr(db.session, 'commit', real_commit)
        db.session.rollback()

        existing = db.session.get(FaceDatasetImage, created[0])
        assert existing.bank_image_id is None
        assert existing.bank_analysis_snapshot is None
        cache_dir = Path(dataset_path(dataset.id)) / '.bank-analysis-cache'
        assert not cache_dir.exists() or not list(cache_dir.glob('*.npz'))


def test_bank_reject_history_survives_dataset_roundtrip_without_overriding_user(
        app, tmp_path):
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source = _bank_with_analysed_image(app, tmp_path)
        source.status = 'reject'
        source.reject_reason = 'low_aesthetic'
        db.session.commit()

        dataset = _promote_to_dataset(app, source_bank, source)
        dataset_row = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).one()
        assert dataset_row.status == 'reject'
        captured = transfer.parse_transfer_metadata(dataset_row.transfer_metadata)
        assert captured['bank']['status'] == 'reject'
        assert captured['bank']['reject_reason'] == 'low_aesthetic'

        # The Dataset is now the active owner. Its newer keep decision wins on
        # return, while the original Bank decision remains in the inert capsule.
        dataset_row.status = 'keep'
        db.session.commit()
        returned_id = banks.start_dataset_import(
            app, 'local', dataset.id, 'Reject history roundtrip')
        returned = BankImage.query.filter_by(bank_id=returned_id).one()
        assert returned.status == 'keep'
        assert returned.reject_reason is None
        captured = transfer.parse_transfer_metadata(returned.transfer_metadata)
        assert captured['bank']['status'] == 'reject'
        assert captured['bank']['reject_reason'] == 'low_aesthetic'


def test_dataset_only_metadata_survives_dataset_bank_dataset_roundtrip(
        app, tmp_path):
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage
    from app.services import bank_transfer_metadata as transfer
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source = _bank_with_analysed_image(app, tmp_path)
        dataset = _promote_to_dataset(app, source_bank, source)
        row = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).one()
        row.source = 'generated'
        row.variation_label = 'three-quarter studio portrait'
        row.caption_short = 'short human caption'
        row.caption_short_origin = 'asserted'
        row.job_id = 'historical-job-id'
        row.variation_prompt = 'soft rim light, neutral background'
        row.klein_model = 'flux-klein-test.safetensors'
        row.parent_image_id = 4242
        row.derivation_kind = 'historical-derivation'
        row.face_score = 0.876
        row.face_state = 'scorable'
        row.fail_reason = 'historical provider detail'
        row.fail_kind = 'error'
        row.upscale_ratio = 1.37
        row.content_sig = 'legacy-content-sig'
        row.content_sig_stat = '123:456'
        db.session.commit()

        returned_bank_id = banks.start_dataset_import(
            app, 'local', dataset.id, 'Dataset metadata carrier')
        bank_row = BankImage.query.filter_by(bank_id=returned_bank_id).one()
        carried = transfer.parse_transfer_metadata(bank_row.transfer_metadata)
        assert carried is not None
        for field in transfer.DATASET_PORTABLE_FIELDS:
            expected = getattr(row, field)
            if field == 'created_at' and expected is not None:
                expected = expected.isoformat()
            assert carried['dataset'][field] == expected, field

        target = datasets.create_dataset(
            'local', 'Dataset metadata restored', 'metadata_restored')
        banks.start_promote(
            app, 'local', returned_bank_id, [bank_row.id], target.id)
        restored = FaceDatasetImage.query.filter_by(dataset_id=target.id).one()
        assert restored.source == 'generated'
        assert restored.variation_label == row.variation_label
        assert restored.caption_short == row.caption_short
        assert restored.caption_short_origin == 'asserted'
        assert restored.variation_prompt == row.variation_prompt
        assert restored.klein_model == row.klein_model
        assert restored.face_score == pytest.approx(row.face_score)
        assert restored.face_state == row.face_state
        assert restored.fail_reason == row.fail_reason
        assert restored.fail_kind == row.fail_kind
        assert restored.upscale_ratio == pytest.approx(row.upscale_ratio)
        # Container-local graph/job/id fields are preserved as inert history,
        # never reactivated against unrelated destination ids.
        assert restored.job_id is None
        assert restored.parent_image_id is None
        assert restored.derivation_kind is None
        final_carried = transfer.parse_transfer_metadata(
            restored.transfer_metadata)
        assert final_carried['dataset'] == carried['dataset']


def test_bank_promotion_copies_exact_duplicates_as_independent_rows(
        app, tmp_path):
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source = _bank_with_analysed_image(app, tmp_path)
        raw = (Path(source_bank.source_path) / source.relpath).read_bytes()
        target = datasets.create_dataset('local', 'No merge', 'no_merge')
        existing_ids, failed = datasets.import_images(
            'local', target.id, [raw], dedupe=True,
            captions=['existing dataset caption'], preserve_exact_bytes=True)
        assert len(existing_ids) == 1 and failed == 0

        banks.start_promote(app, 'local', source_bank.id, [source.id], target.id)
        rows = (FaceDatasetImage.query.filter_by(dataset_id=target.id)
                .order_by(FaceDatasetImage.id.asc()).all())
        assert len(rows) == 2
        assert rows[0].caption == 'existing dataset caption'
        assert rows[1].caption == 'caption from the bank'
        assert rows[1].bank_image_id == source.id
        assert rows[1].transfer_metadata is not None


def test_retrying_the_same_bank_source_is_idempotent(app, tmp_path):
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source = _bank_with_analysed_image(app, tmp_path)
        target = datasets.create_dataset('local', 'Idempotent', 'idempotent')
        for _ in range(2):
            banks.start_promote(
                app, 'local', source_bank.id, [source.id], target.id)
        rows = FaceDatasetImage.query.filter_by(dataset_id=target.id).all()
        assert len(rows) == 1
        assert rows[0].bank_image_id == source.id


def test_one_request_deduplicates_ids_before_sql_chunking(app, tmp_path):
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source = _bank_with_analysed_image(app, tmp_path)
        target = datasets.create_dataset(
            'local', 'Duplicate request ids', 'duplicate_request_ids')
        banks.start_promote(
            app, 'local', source_bank.id,
            [source.id] * (banks._SQL_IN_CHUNK + 1), target.id)

        rows = FaceDatasetImage.query.filter_by(dataset_id=target.id).all()
        assert len(rows) == 1
        assert rows[0].bank_image_id == source.id


def test_bank_to_dataset_copy_holds_an_exclusive_dataset_activity(
        app, tmp_path, monkeypatch):
    from app.models import FaceDatasetImage
    from app.services import dataset_activity
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source = _bank_with_analysed_image(app, tmp_path)
        target = datasets.create_dataset(
            'local', 'Reserved promotion', 'reserved_promotion')
        observed = []
        original_import = banks.import_images

        def guarded_import(*args, **kwargs):
            observed.append(dataset_activity.get(target.id))
            return original_import(*args, **kwargs)

        monkeypatch.setattr(banks, 'import_images', guarded_import)
        banks.start_promote(
            app, 'local', source_bank.id, [source.id], target.id)

        assert observed and observed[0]['kind'] == 'bank_import'
        assert observed[0]['total'] == 1
        assert FaceDatasetImage.query.filter_by(dataset_id=target.id).count() == 1
        assert dataset_activity.get(target.id) is None


def test_bank_to_dataset_copy_refuses_a_dataset_being_frozen_for_training(
        app, tmp_path):
    from app.models import FaceDatasetImage
    from app.services import dataset_activity
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source = _bank_with_analysed_image(app, tmp_path)
        target = datasets.create_dataset(
            'local', 'Busy promotion', 'busy_promotion')
        token = dataset_activity.begin_exclusive(
            target.id, 'training_export', detail='freezing for training')
        try:
            with pytest.raises(
                    dataset_activity.DatasetActivityBusy,
                    match='already has work in progress'):
                banks.start_promote(
                    app, 'local', source_bank.id, [source.id], target.id)
            assert FaceDatasetImage.query.filter_by(
                dataset_id=target.id).count() == 0
        finally:
            dataset_activity.end(token)


def test_dataset_only_metadata_is_inert_after_external_bank_file_replacement(
        app, tmp_path, monkeypatch):
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source = _bank_with_analysed_image(app, tmp_path)
        dataset = _promote_to_dataset(app, source_bank, source)
        dataset_row = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).one()
        dataset_row.source = 'generated'
        dataset_row.caption_short = 'belongs to image A'
        dataset_row.caption_short_origin = 'asserted'
        dataset_row.variation_prompt = 'generation prompt for image A'
        dataset_row.klein_model = 'model-a.safetensors'
        db.session.commit()

        carrier_id = banks.start_dataset_import(
            app, 'local', dataset.id, 'External replacement carrier')
        carrier_bank = db.session.get(ImageBank, carrier_id)
        carrier = BankImage.query.filter_by(bank_id=carrier_id).one()
        carrier_path = Path(carrier_bank.source_path) / carrier.relpath
        before = transfer.parse_transfer_metadata(carrier.transfer_metadata)
        assert before['dataset_fingerprint'] == hashlib.sha256(
            carrier_path.read_bytes()).hexdigest()

        _write_photo(carrier_path, size=(111, 173), quality=72)
        monkeypatch.setattr(banks, 'rebuild_dup_groups', lambda *_a, **_k: 0)
        scan = banks.start_scan(
            app, 'local', carrier_id, rescan=True, ids=[carrier.id])
        assert scan['error'] is None, scan
        carrier = db.session.get(BankImage, carrier.id, populate_existing=True)
        assert carrier.analysis_fingerprint == hashlib.sha256(
            carrier_path.read_bytes()).hexdigest()

        target = datasets.create_dataset(
            'local', 'External replacement target', 'replacement_target')
        banks.start_promote(
            app, 'local', carrier_id, [carrier.id], target.id)
        restored = FaceDatasetImage.query.filter_by(dataset_id=target.id).one()
        assert restored.source == 'import'
        assert restored.caption_short is None
        assert restored.variation_prompt is None
        assert restored.klein_model is None
        after = transfer.parse_transfer_metadata(restored.transfer_metadata)
        assert after['dataset']['caption_short'] == 'belongs to image A'
        assert after['dataset_fingerprint'] == before['dataset_fingerprint']
        assert after['dataset_fingerprint'] != hashlib.sha256(
            carrier_path.read_bytes()).hexdigest()


def test_tracked_bank_rotation_restores_invariant_metadata_not_pixel_scores(
        app, tmp_path, monkeypatch):
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage
    from app.services import bank_transfer_metadata as transfer
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source = _bank_with_analysed_image(app, tmp_path)
        dataset = _promote_to_dataset(app, source_bank, source)
        row = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).one()
        row.source = 'generated'
        row.caption_short = 'safe through a tracked turn'
        row.caption_short_origin = 'asserted'
        row.variation_prompt = 'original generation provenance'
        row.face_score = 0.876
        row.face_state = 'scorable'
        row.upscale_ratio = 1.42
        db.session.commit()

        carrier_id = banks.start_dataset_import(
            app, 'local', dataset.id, 'Tracked transform carrier')
        carrier = BankImage.query.filter_by(bank_id=carrier_id).one()
        banks.rotate_images('local', carrier_id, [carrier.id], 90)
        monkeypatch.setattr(banks, 'rebuild_dup_groups', lambda *_a, **_k: 0)
        scan = banks.start_scan(
            app, 'local', carrier_id, rescan=True, ids=[carrier.id])
        assert scan['error'] is None, scan

        target = datasets.create_dataset(
            'local', 'Tracked transform target', 'tracked_transform')
        banks.start_promote(app, 'local', carrier_id, [carrier.id], target.id)
        restored = FaceDatasetImage.query.filter_by(dataset_id=target.id).one()
        assert restored.source == 'generated'
        assert restored.caption_short == 'safe through a tracked turn'
        assert restored.caption_short_origin == 'asserted'
        assert restored.variation_prompt == 'original generation provenance'
        assert restored.face_score is None
        assert restored.face_state is None
        assert restored.upscale_ratio is None
        history = transfer.parse_transfer_metadata(restored.transfer_metadata)
        assert history['dataset']['face_score'] == pytest.approx(0.876)
        assert history['dataset_analysis_fingerprint'] != history['dataset_fingerprint']


def test_transfer_metadata_migrations_upgrade_legacy_tables_idempotently(app, tmp_path):
    """Both columns must appear on a DB that already has the two image tables."""
    from sqlalchemy import text
    from app import _SCHEMA_ADDITIONS, _apply_additive_migrations
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage, ImageBank
    from app.services import face_dataset_service as datasets

    assert ('face_dataset_image', 'bank_analysis_snapshot', 'TEXT') in _SCHEMA_ADDITIONS
    assert ('bank_image', 'source_metadata', 'TEXT') in _SCHEMA_ADDITIONS
    with app.app_context():
        dataset = datasets.create_dataset('local', 'Legacy', 'legacy-transfer')
        dataset_image = FaceDatasetImage(
            dataset_id=dataset.id, filename='legacy.webp', status='keep',
            caption='keep this caption')
        legacy_source = tmp_path / 'legacy-source'
        legacy_source.mkdir()
        bank = ImageBank(user_id='local', name='Legacy bank', source_path=str(legacy_source))
        db.session.add_all((dataset_image, bank))
        db.session.flush()
        bank_image = BankImage(bank_id=bank.id, relpath='legacy.png', caption='keep bank caption')
        db.session.add(bank_image)
        db.session.commit()
        dataset_image_id, bank_image_id = dataset_image.id, bank_image.id

        db.session.execute(text(
            'ALTER TABLE face_dataset_image DROP COLUMN bank_analysis_snapshot'))
        db.session.execute(text('ALTER TABLE bank_image DROP COLUMN source_metadata'))
        db.session.commit()

        def columns(table):
            return {r[1] for r in db.session.execute(text(f'PRAGMA table_info({table})'))}

        assert 'bank_analysis_snapshot' not in columns('face_dataset_image')
        assert 'source_metadata' not in columns('bank_image')
        _apply_additive_migrations()
        _apply_additive_migrations()       # runs on every boot: must remain a no-op
        assert 'bank_analysis_snapshot' in columns('face_dataset_image')
        assert 'source_metadata' in columns('bank_image')

        legacy_dataset = db.session.execute(text(
            'SELECT caption, bank_analysis_snapshot FROM face_dataset_image WHERE id=:id'),
            {'id': dataset_image_id}).one()
        legacy_bank = db.session.execute(text(
            'SELECT caption, source_metadata FROM bank_image WHERE id=:id'),
            {'id': bank_image_id}).one()
        assert legacy_dataset == ('keep this caption', None)
        assert legacy_bank == ('keep bank caption', None)
