"""Issue #72: protect original files, while keeping Bank curation usable."""
import os
import sys
import types

import pytest
from PIL import Image


def _create(client, folder, read_only=False):
    folder.mkdir(parents=True, exist_ok=True)
    image = folder / 'original.png'
    if not image.exists():
        Image.new('RGB', (128, 128), (20, 50, 80)).save(image)
    response = client.post('/api/bank/create', json={
        'name': 'Original photos', 'folder': str(folder), 'read_only': read_only,
    })
    assert response.status_code == 200, response.get_json()
    bank_id = response.get_json()['id']
    rows = client.get(f'/api/bank/{bank_id}/images').get_json()['images']
    return bank_id, rows[0]['id'], image


def _protect(client, bank_id, value=True):
    return client.patch(f'/api/bank/{bank_id}/read-only', json={'read_only': value})


def _fingerprint(folder):
    return {str(p.relative_to(folder)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in folder.rglob('*') if p.is_file()}


def test_read_only_persists_and_refuses_api_and_direct_source_deletion(
        client, app, tmp_path, monkeypatch):
    from app.services import image_bank_service as banks

    bank_id, image_id, image = _create(client, tmp_path / 'nas', True)
    image.with_suffix('.txt').write_text('original caption', encoding='utf-8')
    before = _fingerprint(image.parent)
    monkeypatch.setattr(banks, '_trash_or_remove', lambda *_: pytest.fail('source deletion'))
    assert client.post(f'/api/bank/{bank_id}/images/status', json={
        'ids': [image_id], 'status': 'reject',
    }).status_code == 200
    assert client.get('/api/banks').get_json()['banks'][0]['read_only'] is True
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['read_only'] is True and payload['source_write_block']
    preview = client.get(f'/api/bank/{bank_id}/delete-rejected/preview').get_json()
    assert preview['read_only'] is True and preview['source_write_block']
    response = client.post(f'/api/bank/{bank_id}/delete-rejected', json={})
    assert response.status_code == 403
    assert response.get_json()['code'] == 'bank_source_read_only'
    with app.app_context(), pytest.raises(banks.BankSourceReadOnly):
        banks.delete_rejected('local', bank_id)
    assert _fingerprint(image.parent) == before
    assert client.get(f'/api/bank/{bank_id}/images').get_json()['total'] == 1


@pytest.mark.parametrize('value', [None, 'false', 'true', 0, 1, [], {}])
def test_protection_requires_real_booleans(client, tmp_path, value):
    bank_id, _, image = _create(client, tmp_path / 'nas', True)
    assert _protect(client, bank_id, value).status_code == 400
    response = client.post('/api/bank/create', json={
        'name': 'Invalid', 'folder': str(image.parent), 'read_only': value,
    })
    assert response.status_code == 400
    assert client.get(f'/api/bank/{bank_id}').get_json()['read_only'] is True


def test_unlock_restores_explicit_delete_behavior(client, tmp_path, monkeypatch):
    bank_id, image_id, image = _create(client, tmp_path / 'nas', True)
    client.post(f'/api/bank/{bank_id}/images/status', json={'ids': [image_id], 'status': 'reject'})
    assert _protect(client, bank_id, False).get_json()['read_only'] is False
    monkeypatch.setitem(sys.modules, 'send2trash', types.SimpleNamespace(send2trash=os.remove))
    response = client.post(f'/api/bank/{bank_id}/delete-rejected', json={})
    assert response.status_code == 202
    assert response.get_json()['rows_removed'] == 1
    assert not image.exists()


def test_read_only_blocks_import_before_download(client, app, tmp_path, monkeypatch):
    from app.services import image_bank_service as banks

    bank_id, _, image = _create(client, tmp_path / 'nas', True)
    before = _fingerprint(image.parent)
    monkeypatch.setattr(banks, '_download_scrape_item', lambda *_: pytest.fail('download'))
    response = client.post('/api/bank/scrape-import', json={
        'bank_id': bank_id, 'items': [{'url': 'https://example.org/photo.png'}],
    })
    assert response.status_code == 403, response.get_json()
    with app.app_context(), pytest.raises(banks.BankSourceReadOnly):
        banks.scrape_import_to_bank('local', [{'url': 'https://example.org/photo.png'}], bank_id)
    assert _fingerprint(image.parent) == before


def test_removing_protected_imported_bank_preserves_source_folder(client, app):
    from app import config

    with app.app_context():
        folder = config.bank_sources_root() / 'protected-copy'
    bank_id, _, image = _create(client, folder, True)
    before = _fingerprint(folder)
    assert client.delete(f'/api/bank/{bank_id}').status_code == 200
    assert _fingerprint(folder) == before and image.exists()
    assert client.get(f'/api/bank/{bank_id}').status_code == 404


@pytest.mark.parametrize('relation', ['same', 'parent', 'child'])
def test_other_banks_cannot_write_into_a_protected_tree(client, tmp_path, relation):
    parent = tmp_path / 'nas'
    child = parent / 'album'
    writable, image_id, image = _create(client, child if relation == 'parent' else parent)
    protected = parent if relation in ('same', 'parent') else child
    _create(client, protected, True)
    before = _fingerprint(parent)
    client.post(f'/api/bank/{writable}/images/status', json={'ids': [image_id], 'status': 'reject'})
    response = client.post(f'/api/bank/{writable}/delete-rejected', json={})
    assert response.status_code == 403
    assert 'overlaps' in response.get_json()['error']
    assert image.exists() and _fingerprint(parent) == before


def test_protection_waits_for_own_and_overlapping_writers(client, tmp_path):
    from app.services import bank_jobs

    bank_id, _, image = _create(client, tmp_path / 'nas')
    other, _, _ = _create(client, image.parent)
    for busy_id in (bank_id, other):
        with bank_jobs.mutation_lease(busy_id, 'delete_rejected'):
            assert _protect(client, bank_id).status_code == 409
        assert client.get(f'/api/bank/{bank_id}').get_json()['read_only'] is False
    assert _protect(client, bank_id).status_code == 200


def test_read_only_creation_refuses_a_busy_overlapping_bank(client, tmp_path):
    from app.services import bank_jobs

    bank_id, _, image = _create(client, tmp_path / 'nas')
    with bank_jobs.mutation_lease(bank_id, 'delete_rejected'):
        response = client.post('/api/bank/create', json={
            'name': 'Protected', 'folder': str(image.parent), 'read_only': True,
        })
        assert response.status_code == 409
    assert len(client.get('/api/banks').get_json()['banks']) == 1


def test_worker_rechecks_protection_after_preflight(client, app, tmp_path):
    from app.services import image_bank_service as banks

    bank_id, _, _ = _create(client, tmp_path / 'nas')
    with app.app_context():
        banks._delete_rejected_guard('local', bank_id)
        banks.set_read_only('local', bank_id, True)
        with pytest.raises(banks.BankSourceReadOnly):
            banks.delete_rejected('local', bank_id, job={})


def test_curation_scan_and_copies_still_work_without_touching_originals(client, app, tmp_path):
    from app.services import image_bank_service as banks

    bank_id, image_id, image = _create(client, tmp_path / 'nas', True)
    before = _fingerprint(image.parent)
    assert client.post(f'/api/bank/{bank_id}/scan', json={}).status_code == 202
    assert client.post(f'/api/bank/{bank_id}/images/status', json={
        'ids': [image_id], 'status': 'keep',
    }).status_code == 200
    with app.app_context():
        banks.crop_image('local', bank_id, image_id, 0, 0, 64, 64)
        bank = banks.get_bank('local', bank_id)
        from app.models import BankImage
        row = BankImage.query.filter_by(bank_id=bank_id).first()
        assert banks.resolved_image_path(bank, row) != str(image)
        from app.services import face_dataset_service
        dataset_id = face_dataset_service.create_dataset('local', 'Selected copy', 'photo').id
    response = client.post(f'/api/bank/{bank_id}/promote', json={'dataset_id': dataset_id})
    assert response.status_code == 202, response.get_json()
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['activity']['error'] is None
    assert payload['counts']['promoted'] == 1
    assert _fingerprint(image.parent) == before


def test_a_protected_drive_root_covers_child_banks(client, app, tmp_path):
    from app.extensions import db
    from app.models import ImageBank
    from app.services import image_bank_service as banks

    bank_id, _, _ = _create(client, tmp_path / 'nas')
    with app.app_context():
        db.session.add(ImageBank(user_id='local', name='Protected root',
                                 source_path=tmp_path.anchor, read_only=True))
        db.session.commit()
        assert banks.source_write_block(banks.get_bank('local', bank_id))


def test_unknown_or_unowned_bank_cannot_change_protection(client, app, tmp_path):
    from app.extensions import db
    from app.models import ImageBank

    bank_id, _, _ = _create(client, tmp_path / 'nas')
    with app.app_context():
        db.session.get(ImageBank, bank_id).user_id = 'another-user'
        db.session.commit()
    assert _protect(client, bank_id).status_code == 404
    assert _protect(client, 999999).status_code == 404


def test_existing_database_migrates_without_enabling_protection(client, app, tmp_path):
    from sqlalchemy import text
    from app import _apply_additive_migrations
    from app.extensions import db

    bank_id, _, image = _create(client, tmp_path / 'nas')
    with app.app_context():
        db.session.execute(text('ALTER TABLE image_bank DROP COLUMN read_only'))
        db.session.commit()
        _apply_additive_migrations()
        _apply_additive_migrations()
        rows = db.session.execute(text('SELECT id, read_only FROM image_bank')).all()
        assert rows == [(bank_id, 0)]
    assert image.exists()
    assert client.get(f'/api/bank/{bank_id}').get_json()['read_only'] is False
