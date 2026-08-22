"""Durable CLIP/SigLIP2 semantic-duplicate partitions."""

import pytest

np = pytest.importorskip('numpy')


def _seed_bank(app, tmp_path, *, engine='clip', row_count=2):
    from app.extensions import db
    from app.models import BankImage, ImageBank

    source = tmp_path / f'source-{engine}'
    source.mkdir(parents=True, exist_ok=True)
    with app.app_context():
        bank = ImageBank(
            user_id='local', name=f'{engine} bank', source_path=str(source),
            semantic_engine=engine)
        db.session.add(bank)
        db.session.flush()
        for index in range(row_count):
            relpath = f'{index}.jpg'
            (source / relpath).write_bytes(f'image-{index}'.encode())
            db.session.add(BankImage(bank_id=bank.id, relpath=relpath))
        db.session.commit()
        return bank.id, source


def _lane_values(bank_id):
    from app.models import BankImage

    rows = (BankImage.query.filter_by(bank_id=bank_id)
            .order_by(BankImage.id.asc()).all())
    return [
        (row.semantic_dup_group, row.clip_semantic_dup_group,
         row.siglip2_semantic_dup_group)
        for row in rows
    ]


def test_additive_migration_backfills_active_partition_without_overwriting_other_lane(
        app, tmp_path):
    from app import _SCHEMA_ADDITIONS, _apply_additive_migrations
    from app.extensions import db
    from app.models import BankImage

    assert ('bank_image', 'clip_semantic_dup_group', 'INTEGER') in _SCHEMA_ADDITIONS
    assert ('bank_image', 'siglip2_semantic_dup_group', 'INTEGER') in _SCHEMA_ADDITIONS
    clip_id, _ = _seed_bank(app, tmp_path / 'clip', engine='clip')
    siglip_id, _ = _seed_bank(app, tmp_path / 'siglip', engine='siglip2')

    with app.app_context():
        clip_rows = BankImage.query.filter_by(bank_id=clip_id).all()
        siglip_rows = BankImage.query.filter_by(bank_id=siglip_id).all()
        for row in clip_rows:
            row.semantic_dup_group = 4
            row.clip_semantic_dup_group = None
            row.siglip2_semantic_dup_group = 40
        for row in siglip_rows:
            row.semantic_dup_group = 9
            row.clip_semantic_dup_group = 90
            row.siglip2_semantic_dup_group = None
        db.session.commit()

        _apply_additive_migrations()
        _apply_additive_migrations()  # every boot: backfill stays idempotent

        assert _lane_values(clip_id) == [(4, 4, 40), (4, 4, 40)]
        assert _lane_values(siglip_id) == [(9, 90, 9), (9, 90, 9)]


def test_engine_switch_saves_outgoing_and_restores_target_without_touching_caches(
        app, tmp_path, monkeypatch):
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import image_bank_service as banks

    bank_id, _ = _seed_bank(app, tmp_path, engine='clip')
    with app.app_context():
        rows = BankImage.query.filter_by(bank_id=bank_id).all()
        for row in rows:
            # Legacy CLIP state has not yet been copied to its private lane.
            row.semantic_dup_group = 3
            row.clip_semantic_dup_group = None
            row.siglip2_semantic_dup_group = 8
        db.session.commit()

        score_cache = banks._score_cache_path(bank_id)
        siglip_cache = banks._semantic_cache_path(bank_id)
        score_cache.parent.mkdir(parents=True, exist_ok=True)
        siglip_cache.parent.mkdir(parents=True, exist_ok=True)
        score_cache.write_bytes(b'clip-cache-must-survive')
        siglip_cache.write_bytes(b'siglip-cache-must-survive')
        cache_before = {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in (score_cache, siglip_cache)
        }
        monkeypatch.setattr(
            banks, 'semantic_engine_info',
            lambda *_args, **_kwargs: {'engine': 'test'})

        banks.set_semantic_engine('local', bank_id, 'siglip2')
        assert db.session.get(ImageBank, bank_id).semantic_engine == 'siglip2'
        assert _lane_values(bank_id) == [(8, 3, 8), (8, 3, 8)]

        banks.set_semantic_engine('local', bank_id, 'clip')
        assert db.session.get(ImageBank, bank_id).semantic_engine == 'clip'
        assert _lane_values(bank_id) == [(3, 3, 8), (3, 3, 8)]
        assert {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in (score_cache, siglip_cache)
        } == cache_before


def test_engine_switch_rolls_back_bank_and_lanes_together(
        app, tmp_path, monkeypatch):
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import image_bank_service as banks

    bank_id, _ = _seed_bank(app, tmp_path, engine='clip')
    with app.app_context():
        rows = BankImage.query.filter_by(bank_id=bank_id).all()
        for row in rows:
            row.semantic_dup_group = 2
            row.clip_semantic_dup_group = 20
            row.siglip2_semantic_dup_group = 7
        db.session.commit()

        session = db.session()
        original_commit = session.commit

        def fail_commit():
            raise RuntimeError('simulated commit failure')

        monkeypatch.setattr(session, 'commit', fail_commit)
        with pytest.raises(RuntimeError, match='simulated commit failure'):
            banks.set_semantic_engine('local', bank_id, 'siglip2')
        session.rollback()
        monkeypatch.setattr(session, 'commit', original_commit)
        db.session.expire_all()

        assert db.session.get(ImageBank, bank_id).semantic_engine == 'clip'
        assert _lane_values(bank_id) == [(2, 20, 7), (2, 20, 7)]


@pytest.mark.parametrize(
    ('engine', 'inactive_values'),
    [('clip', (71, 72)), ('siglip2', (81, 82))],
)
def test_rebuild_mirrors_active_partition_to_selected_lane_only(
        app, tmp_path, monkeypatch, engine, inactive_values):
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import bank_transfer_metadata
    from app.services import image_bank_service as banks

    bank_id, _ = _seed_bank(app, tmp_path, engine=engine)
    cache_path = tmp_path / f'{engine}-cache.npz'
    cache_path.write_bytes(b'stable-cache-generation')

    with app.app_context():
        bank = db.session.get(ImageBank, bank_id)
        rows = (BankImage.query.filter_by(bank_id=bank_id)
                .order_by(BankImage.id.asc()).all())
        paths = [banks.analysis_image_path(bank, row) for row in rows]
        fingerprints = {
            path: bank_transfer_metadata.content_fingerprint_path(path)
            for path in paths
        }
        for index, row in enumerate(rows):
            row.semantic_dup_group = 6
            if engine == 'clip':
                row.clip_semantic_dup_group = 6
                row.siglip2_semantic_dup_group = inactive_values[index]
            else:
                row.clip_semantic_dup_group = inactive_values[index]
                row.siglip2_semantic_dup_group = 6
            row.analysis_fingerprint = fingerprints[paths[index]]
        db.session.commit()

        vector = np.zeros(768, dtype='float32')
        vector[0] = 1.0
        embeddings = {path: vector.copy() for path in paths}
        monkeypatch.setattr(banks, '_score_cache_path', lambda *_args: cache_path)
        monkeypatch.setattr(banks, '_semantic_cache_path', lambda *_args: cache_path)
        monkeypatch.setattr(
            banks, '_load_semantic_embeddings', lambda _bank: embeddings)
        monkeypatch.setattr(
            banks, '_semantic_embedding_fingerprint',
            lambda _bank, path: fingerprints[path])

        assert banks.rebuild_semantic_dup_groups(bank_id, threshold=0.95) == 1
        expected = ([(1, 1, 71), (1, 1, 72)] if engine == 'clip'
                    else [(1, 81, 1), (1, 82, 1)])
        assert _lane_values(bank_id) == expected


def test_rebuild_clears_only_active_and_selected_lane_when_no_group(
        app, tmp_path, monkeypatch):
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import bank_transfer_metadata
    from app.services import image_bank_service as banks

    bank_id, _ = _seed_bank(app, tmp_path, engine='siglip2')
    cache_path = tmp_path / 'siglip2-no-groups.npz'
    cache_path.write_bytes(b'stable-cache-generation')
    with app.app_context():
        bank = db.session.get(ImageBank, bank_id)
        rows = (BankImage.query.filter_by(bank_id=bank_id)
                .order_by(BankImage.id.asc()).all())
        paths = [banks.analysis_image_path(bank, row) for row in rows]
        fingerprints = {
            path: bank_transfer_metadata.content_fingerprint_path(path)
            for path in paths
        }
        for index, row in enumerate(rows):
            row.semantic_dup_group = 5
            row.clip_semantic_dup_group = 30 + index
            row.siglip2_semantic_dup_group = 5
            row.analysis_fingerprint = fingerprints[paths[index]]
        db.session.commit()

        first = np.zeros(768, dtype='float32')
        second = np.zeros(768, dtype='float32')
        first[0], second[1] = 1.0, 1.0
        embeddings = {paths[0]: first, paths[1]: second}
        monkeypatch.setattr(banks, '_semantic_cache_path', lambda *_args: cache_path)
        monkeypatch.setattr(
            banks, '_load_semantic_embeddings', lambda _bank: embeddings)
        monkeypatch.setattr(
            banks, '_semantic_embedding_fingerprint',
            lambda _bank, path: fingerprints[path])

        assert banks.rebuild_semantic_dup_groups(bank_id, threshold=0.95) == 0
        assert _lane_values(bank_id) == [(None, 30, None), (None, 31, None)]


def test_first_score_fingerprint_bind_preserves_exact_inactive_siglip2_lane(
        app, tmp_path, monkeypatch):
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import bank_semantic_engine, bank_semantic_models
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    bank_id, _ = _seed_bank(app, tmp_path, engine='clip', row_count=1)
    with app.app_context():
        bank = db.session.get(ImageBank, bank_id)
        row = BankImage.query.filter_by(bank_id=bank_id).one()
        path = banks.analysis_image_path(bank, row)
        fingerprint = transfer.content_fingerprint_path(path)
        row.semantic_dup_group = None
        row.clip_semantic_dup_group = None
        row.siglip2_semantic_dup_group = 9
        row.analysis_fingerprint = None
        db.session.commit()

        vector = (1.0,) + (0.0,) * (bank_semantic_models.DIMENSION - 1)
        semantic = {
            'state': 'ok', 'engine': 'siglip2',
            'model_id': bank_semantic_models.MODEL_ID,
            'revision': bank_semantic_models.REVISION,
            'model_key': bank_semantic_models.MODEL_KEY,
            'dimension': bank_semantic_models.DIMENSION,
            'embedding': vector,
        }
        counts = transfer.write_runtime_caches(
            None, None, {path: {'semantic': semantic}},
            semantic_path=banks._semantic_cache_path(bank_id),
            expected_fingerprints={path: fingerprint})
        assert counts == {'score': 0, 'face': 0, 'semantic': 1}
        bank_semantic_engine.reset_memo()

        by_path = {path: row.id}
        preserved = banks._preserved_siglip2_groups(bank_id, by_path)
        assert preserved == {row.id: (9, fingerprint)}
        # The generic binder is shared by Quality, Face, Medium, Framing and
        # folder-person as well as Score; the first writer of the shared marker
        # must retain the independently proven SigLIP2 lane.
        assert banks._prepare_analysis_write(row, path, fingerprint) is True
        assert row.siglip2_semantic_dup_group == 9
        row.analysis_fingerprint = None
        db.session.commit()
        result = {
            path: {'state': 'ok', 'fingerprint': fingerprint,
                   'aesthetic': 7.0, 'nsfw': 0.02},
        }
        banks._apply_score_results(
            {'done': 0, 'cancelled': False}, by_path, result, True,
            preserved_siglip2_groups=preserved, selected_engine='clip')
        db.session.expire_all()

        restored = BankImage.query.filter_by(bank_id=bank_id).one()
        assert restored.analysis_fingerprint == fingerprint
        assert (restored.semantic_dup_group,
                restored.clip_semantic_dup_group,
                restored.siglip2_semantic_dup_group) == (None, None, 9)
        monkeypatch.setattr(
            banks, 'semantic_engine_info',
            lambda *_args, **_kwargs: {'engine': 'test'})
        banks.set_semantic_engine('local', bank_id, 'siglip2')
        assert _lane_values(bank_id) == [(9, None, 9)]
        banks.set_semantic_engine('local', bank_id, 'clip')
        assert _lane_values(bank_id) == [(None, None, 9)]


def test_real_siglip2_rebuild_binds_generation_then_survives_quality_and_roundtrip(
        app, tmp_path, monkeypatch):
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import bank_semantic_engine, bank_semantic_models
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    bank_id, _ = _seed_bank(app, tmp_path, engine='siglip2', row_count=2)
    with app.app_context():
        bank = db.session.get(ImageBank, bank_id)
        rows = (BankImage.query.filter_by(bank_id=bank_id)
                .order_by(BankImage.id.asc()).all())
        paths = [banks.analysis_image_path(bank, row) for row in rows]
        fingerprints = {
            path: transfer.content_fingerprint_path(path) for path in paths}
        vector = (1.0,) + (0.0,) * (bank_semantic_models.DIMENSION - 1)
        entries = {
            path: {'semantic': {
                'state': 'ok', 'engine': 'siglip2',
                'model_id': bank_semantic_models.MODEL_ID,
                'revision': bank_semantic_models.REVISION,
                'model_key': bank_semantic_models.MODEL_KEY,
                'dimension': bank_semantic_models.DIMENSION,
                'embedding': vector,
            }} for path in paths
        }
        assert transfer.write_runtime_caches(
            None, None, entries,
            semantic_path=banks._semantic_cache_path(bank_id),
            expected_fingerprints=fingerprints)['semantic'] == 2
        bank_semantic_engine.reset_memo()

        assert banks.rebuild_semantic_dup_groups(
            bank_id, threshold=0.95) == 1
        db.session.expire_all()
        assert _lane_values(bank_id) == [(1, None, 1), (1, None, 1)]
        rebound = (BankImage.query.filter_by(bank_id=bank_id)
                   .order_by(BankImage.id.asc()).all())
        assert [row.analysis_fingerprint for row in rebound] == [
            fingerprints[path] for path in paths]

        monkeypatch.setattr(
            banks, 'semantic_engine_info',
            lambda *_args, **_kwargs: {'engine': 'test'})
        banks.set_semantic_engine('local', bank_id, 'clip')
        quality_row = BankImage.query.filter_by(bank_id=bank_id).first()
        assert banks._prepare_analysis_write(
            quality_row, paths[0], fingerprints[paths[0]]) is True
        db.session.commit()
        assert _lane_values(bank_id) == [(None, None, 1), (None, None, 1)]
        banks.set_semantic_engine('local', bank_id, 'siglip2')
        assert _lane_values(bank_id) == [(1, None, 1), (1, None, 1)]


def test_generic_lane_proof_loads_each_cache_only_once_per_generation(
        app, tmp_path, monkeypatch):
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import bank_semantic_engine
    from app.services import image_bank_service as banks

    bank_id, _ = _seed_bank(app, tmp_path, engine='clip', row_count=2)
    with app.app_context():
        bank = db.session.get(ImageBank, bank_id)
        rows = (BankImage.query.filter_by(bank_id=bank_id)
                .order_by(BankImage.id.asc()).all())
        paths = [banks.analysis_image_path(bank, row) for row in rows]
        fingerprints = {path: f'{index + 1:064x}'
                        for index, path in enumerate(paths)}
        for row in rows:
            row.siglip2_semantic_dup_group = 7
        db.session.commit()
        calls = []
        monkeypatch.setattr(
            bank_semantic_engine, 'cache_generation',
            lambda *_args, **_kwargs: (123, 456))
        monkeypatch.setattr(
            bank_semantic_engine, 'load_semantic_embeddings',
            lambda *_args, **_kwargs: calls.append('load') or {
                path: object() for path in paths})
        monkeypatch.setattr(
            bank_semantic_engine, 'embedding_fingerprint',
            lambda path, **_kwargs: fingerprints[path])
        with banks._SEMANTIC_GROUP_PROOF_LOCK:
            banks._semantic_group_proof_memo.clear()

        assert [banks._strict_semantic_group_for_generation(
            row, path, fingerprints[path], 'siglip2')
                for row, path in zip(rows, paths)] == [7, 7]
        assert calls == ['load']
