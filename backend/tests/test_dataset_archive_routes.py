import io
import json
import zipfile
from contextlib import nullcontext
from types import SimpleNamespace

import pytest


def _empty_backup(service, *, manifest_extra=None, images_meta=None):
    manifest = {
        'format': service.BACKUP_FORMAT,
        'version': service.BACKUP_VERSION,
        'name': 'Imported backup',
        'trigger_word': 'imported_backup',
    }
    manifest.update(manifest_extra or {})
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('manifest.json', json.dumps(manifest))
        archive.writestr('images.json', json.dumps(images_meta or []))
    return output.getvalue()


def test_archive_limit_override_runs_before_csrf_form_parsing(app, client):
    """The endpoint override must beat Flask-WTF's eager request.form read."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as service

    with app.app_context():
        dataset = service.create_dataset(LOCAL_USER, 'Source', 'source')
        backup = service.build_backup_zip(LOCAL_USER, dataset.id)

    assert len(backup) > 128
    app.config.update(
        WTF_CSRF_ENABLED=True,
        MAX_CONTENT_LENGTH=128,
        DATASET_ARCHIVE_MAX_UPLOAD_BYTES=64 * 1024,
        DATASET_ARCHIVE_MULTIPART_OVERHEAD_BYTES=16 * 1024,
    )
    token = client.get('/api/csrf-token').get_json()['csrf_token']
    response = client.post(
        '/api/dataset/backup/import',
        data={'file': (io.BytesIO(backup), 'backup.zip')},
        headers={'X-CSRFToken': token},
        content_type='multipart/form-data',
    )

    assert response.status_code == 200
    assert response.get_json()['ok'] is True


@pytest.mark.parametrize('preparse', [False, True])
def test_archive_file_and_request_limits_return_json_413(app, client, preparse):
    app.config.update(
        DATASET_ARCHIVE_MAX_UPLOAD_BYTES=512,
        # Large overhead reaches the exact file-size check; tiny overhead makes
        # Werkzeug reject while parsing, before the view is entered.
        DATASET_ARCHIVE_MULTIPART_OVERHEAD_BYTES=64 if preparse else 4096,
    )
    payload = b'x' * (4096 if preparse else 1024)
    response = client.post(
        '/api/dataset/backup/import',
        data={'file': (io.BytesIO(payload), 'too-large.zip')},
        content_type='multipart/form-data',
    )

    assert response.status_code == 413
    assert response.is_json
    assert response.get_json()['ok'] is False
    assert 'archive too large' in response.get_json()['error']


def test_non_api_request_too_large_keeps_standard_413(app, client):
    from flask import request

    @app.post('/outside-api')
    def outside_api():
        request.get_data()
        return 'ok'

    app.config['MAX_CONTENT_LENGTH'] = 8
    response = client.post('/outside-api', data=b'x' * 32)

    assert response.status_code == 413
    assert not response.is_json


def test_both_archive_import_routes_pass_seekable_streams(app, client, monkeypatch):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as service

    with app.app_context():
        dataset = service.create_dataset(LOCAL_USER, 'Merge target', 'merge_target')
        dataset_id = dataset.id

    seen = []

    def fake_restore(_user_id, archive):
        seen.append(('backup', isinstance(archive, (bytes, bytearray)), archive.tell()))
        return SimpleNamespace(id=123, name='Restored')

    def fake_merge(_user_id, _dataset_id, archive, stats=None):
        seen.append(('training', isinstance(archive, (bytes, bytearray)), archive.tell()))
        return [], 0

    monkeypatch.setattr(service, 'import_backup_zip', fake_restore)
    monkeypatch.setattr(service, 'import_dataset_zip', fake_merge)

    backup_response = client.post(
        '/api/dataset/backup/import',
        data={'file': (io.BytesIO(b'backup'), 'backup.zip')},
        content_type='multipart/form-data',
    )
    training_response = client.post(
        f'/api/dataset/{dataset_id}/import-zip',
        data={'file': (io.BytesIO(b'training'), 'training.zip')},
        content_type='multipart/form-data',
    )

    assert backup_response.status_code == 200
    assert training_response.status_code == 200
    assert seen == [('backup', False, 0), ('training', False, 0)]


@pytest.mark.parametrize(
    ('endpoint_suffix', 'writer_name'),
    [('backup', 'write_backup_zip'), ('export', 'write_export_zip')],
)
def test_archive_exports_roll_to_disk_and_close_with_response(
        app, client, monkeypatch, endpoint_suffix, writer_name):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as service

    with app.app_context():
        dataset = service.create_dataset(LOCAL_USER, 'Export source', 'export_source')
        dataset_id = dataset.id

    captured = {}

    def fake_writer(_user_id, _dataset_id, output):
        output.write(b'z' * 256)
        captured['output'] = output
        captured['rolled'] = output._rolled

    monkeypatch.setattr(service, writer_name, fake_writer)
    app.config['DATASET_ARCHIVE_SPOOL_MEMORY_BYTES'] = 64

    response = client.get(f'/api/dataset/{dataset_id}/{endpoint_suffix}')

    assert response.status_code == 200
    assert response.data == b'z' * 256
    assert response.content_length == 256
    assert captured['rolled'] is True
    response.close()
    assert captured['output'].closed


def test_archive_export_writer_failure_closes_spool(app, client, monkeypatch):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as service

    with app.app_context():
        dataset = service.create_dataset(LOCAL_USER, 'Broken export', 'broken_export')
        dataset_id = dataset.id

    captured = {}

    def broken_writer(_user_id, _dataset_id, output):
        captured['output'] = output
        output.write(b'partial')
        raise RuntimeError('injected writer failure')

    monkeypatch.setattr(service, 'write_backup_zip', broken_writer)
    with pytest.raises(RuntimeError, match='injected writer failure'):
        client.get(f'/api/dataset/{dataset_id}/backup')
    assert captured['output'].closed


def test_archive_services_accept_streams_without_closing_callers(app):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as service

    with app.app_context():
        source = service.create_dataset(LOCAL_USER, 'Stream source', 'stream_source')
        backup_output = io.BytesIO()
        service.write_backup_zip(LOCAL_USER, source.id, backup_output)
        backup_stream = io.BytesIO(backup_output.getvalue())
        restored = service.import_backup_zip(LOCAL_USER, backup_stream)
        assert restored.name == 'Stream source'
        assert not backup_stream.closed

        target = service.create_dataset(LOCAL_USER, 'Training target', 'training_target')
        training_stream = io.BytesIO()
        with zipfile.ZipFile(training_stream, 'w') as archive:
            archive.writestr('notes.md', 'no images')
        service.import_dataset_zip(LOCAL_USER, target.id, training_stream)
        assert not training_stream.closed


def _make_max_content_length_read_only(monkeypatch):
    """Reproduce a Flask whose per-request upload ceiling cannot be assigned.

    ``Request.max_content_length`` only became settable per request in Flask
    3.1; before that it was a plain read-only property, and any environment
    that resolves an older Flask (a stale user site-packages, a distro build)
    turned every archive upload into a 500. Stripping the setter off the
    property reproduces exactly that class on the pinned Flask.
    """
    import flask

    getter = flask.Request.__dict__['max_content_length'].fget
    monkeypatch.setattr(flask.Request, 'max_content_length', property(getter))


def test_archive_ceiling_holds_when_the_request_property_is_read_only(
        app, client, monkeypatch):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as service

    with app.app_context():
        dataset = service.create_dataset(LOCAL_USER, 'Read only', 'read_only')
        backup = service.build_backup_zip(LOCAL_USER, dataset.id)

    assert len(backup) > 128
    app.config.update(
        MAX_CONTENT_LENGTH=128,
        DATASET_ARCHIVE_MAX_UPLOAD_BYTES=64 * 1024,
        DATASET_ARCHIVE_MULTIPART_OVERHEAD_BYTES=16 * 1024,
    )
    _make_max_content_length_read_only(monkeypatch)

    response = client.post(
        '/api/dataset/backup/import',
        data={'file': (io.BytesIO(backup), 'backup.zip')},
        content_type='multipart/form-data',
    )

    assert response.status_code == 200
    assert response.get_json()['ok'] is True


def test_ordinary_endpoints_keep_the_plain_ceiling(app, client, monkeypatch):
    """The raised ceiling stays the archive endpoints' privilege, not a global."""
    from flask import request

    @app.post('/api/ordinary-upload')
    def ordinary_upload():
        request.get_data()
        return {'ok': True}

    app.config.update(
        MAX_CONTENT_LENGTH=128,
        DATASET_ARCHIVE_MAX_UPLOAD_BYTES=64 * 1024,
        DATASET_ARCHIVE_MULTIPART_OVERHEAD_BYTES=16 * 1024,
    )
    _make_max_content_length_read_only(monkeypatch)

    response = client.post('/api/ordinary-upload', data=b'x' * 1024)

    assert response.status_code == 413
    error = response.get_json()['error']
    assert error.startswith('upload too large')
    # …and it names the numbers: a bare "too large" sent people resizing single
    # photos when the whole drop was over the ceiling (_nofaceman, Discord).
    assert 'this request is 0.0 MiB' in error and 'the app accepts 0.0 MiB at a time' in error
    assert 'drop fewer files at once' in error


def test_backup_metadata_cap_is_checked_before_inflation(app, monkeypatch):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as service

    data = _empty_backup(service, manifest_extra={'padding': 'x' * 512})
    monkeypatch.setattr(service, '_BACKUP_MAX_METADATA_BYTES', 128)

    def forbidden_read(*_args, **_kwargs):
        raise AssertionError('oversized metadata must not be inflated')

    monkeypatch.setattr(zipfile.ZipFile, 'read', forbidden_read)
    with app.app_context(), pytest.raises(ValueError, match='manifest.json is too large'):
        service.import_backup_zip(LOCAL_USER, io.BytesIO(data))


def test_backup_uncompressed_cap_counts_unknown_entries(app, monkeypatch):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as service

    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('manifest.json', json.dumps({
            'format': service.BACKUP_FORMAT,
            'version': service.BACKUP_VERSION,
        }))
        archive.writestr('images.json', '[]')
        archive.writestr('ignored.bin', b'x' * 256)
    monkeypatch.setattr(service, '_BACKUP_MAX_BYTES', 128)

    with app.app_context(), pytest.raises(ValueError, match='backup too large'):
        service.import_backup_zip(LOCAL_USER, io.BytesIO(output.getvalue()))


@pytest.mark.parametrize(
    ('entries', 'central_size', 'message'),
    [
        ('too-many', 0, 'too many files in backup'),
        (0, 'too-large', 'backup central directory is too large'),
    ],
    ids=('entry-count', 'central-directory-bytes'),
)
def test_backup_eocd_caps_run_before_zipfile_parses_the_directory(
        app, monkeypatch, entries, central_size, message):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as service

    end_record = [0] * 10
    end_record[getattr(zipfile, '_ECD_ENTRIES_TOTAL', 4)] = (
        service._BACKUP_MAX_FILES + 3 if entries == 'too-many' else entries)
    end_record[getattr(zipfile, '_ECD_SIZE', 5)] = (
        service._BACKUP_MAX_CENTRAL_DIRECTORY_BYTES + 1
        if central_size == 'too-large' else central_size)
    stream = io.BytesIO(b'EOCD preflight owns this rejection')
    monkeypatch.setattr(service.zipfile, '_EndRecData', lambda source: end_record)

    def forbidden_zipfile(*_args, **_kwargs):
        raise AssertionError('ZipFile must not parse an over-budget central directory')

    monkeypatch.setattr(service.zipfile, 'ZipFile', forbidden_zipfile)
    with app.app_context(), pytest.raises(ValueError, match=message):
        service.import_backup_zip(LOCAL_USER, stream)
    assert stream.tell() == 0


def test_training_zip_eocd_cap_runs_before_zipfile_parses_the_directory(
        app, monkeypatch):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as service

    with app.app_context():
        dataset = service.create_dataset(
            LOCAL_USER, 'Training EOCD cap', 'training_eocd_cap')
        end_record = [0] * 10
        end_record[getattr(zipfile, '_ECD_ENTRIES_TOTAL', 4)] = \
            service.DATASET_ZIP_MAX_FILES + 1
        end_record[getattr(zipfile, '_ECD_SIZE', 5)] = 0
        stream = io.BytesIO(b'EOCD preflight owns this rejection')
        monkeypatch.setattr(
            service.zipfile, '_EndRecData', lambda _source: end_record)

        def forbidden_zipfile(*_args, **_kwargs):
            raise AssertionError(
                'ZipFile must not parse an over-budget central directory')

        monkeypatch.setattr(service.zipfile, 'ZipFile', forbidden_zipfile)
        with pytest.raises(ValueError, match='too many files in zip'):
            service.import_dataset_zip(LOCAL_USER, dataset.id, stream)
        assert stream.tell() == 0


def test_backup_eocd_preflight_rewinds_before_zipfile(app, monkeypatch):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as service

    data = _empty_backup(service)
    stream = io.BytesIO(data)
    native_end_reader = zipfile._EndRecData
    native_zipfile = zipfile.ZipFile
    constructor_offsets = []

    def end_reader(source):
        result = native_end_reader(source)
        assert source.tell() != 0
        return result

    class RewoundZipFile(native_zipfile):
        def __init__(self, source, *args, **kwargs):
            constructor_offsets.append(source.tell())
            super().__init__(source, *args, **kwargs)

    sentinel = object()
    monkeypatch.setattr(service.zipfile, '_EndRecData', end_reader)
    monkeypatch.setattr(service.zipfile, 'ZipFile', RewoundZipFile)
    monkeypatch.setattr(service, '_import_backup_zipfile', lambda *_args: sentinel)

    with app.app_context():
        assert service.import_backup_zip(LOCAL_USER, stream) is sentinel
    assert constructor_offsets == [0]


def test_backup_eocd_preflight_has_a_compatibility_fallback(app, monkeypatch):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as service

    stream = io.BytesIO(_empty_backup(service))
    stream.seek(7)
    constructor_offsets = []
    native_zipfile = zipfile.ZipFile

    def recording_zipfile(source, *args, **kwargs):
        constructor_offsets.append(source.tell())
        return native_zipfile(source, *args, **kwargs)

    # Model a runtime whose public ZipFile API exists but CPython's private EOCD
    # helper does not.  The fallback must preserve compatibility and rewind.
    compat_zipfile = SimpleNamespace(
        BadZipFile=zipfile.BadZipFile,
        ZipFile=recording_zipfile,
    )
    sentinel = object()
    monkeypatch.setattr(service, 'zipfile', compat_zipfile)
    monkeypatch.setattr(service, '_import_backup_zipfile', lambda *_args: sentinel)

    with app.app_context():
        assert service.import_backup_zip(LOCAL_USER, stream) is sentinel
    assert constructor_offsets == [0]


def test_corrupt_deflate_backup_member_is_a_clean_400(app, client):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as service

    data = bytearray(_empty_backup(
        service, manifest_extra={'padding': 'compress-me-' * 200}))
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        info = archive.getinfo('manifest.json')
        assert info.compress_type == zipfile.ZIP_DEFLATED
        header = info.header_offset
        name_size = int.from_bytes(data[header + 26:header + 28], 'little')
        extra_size = int.from_bytes(data[header + 28:header + 30], 'little')
        payload_start = header + 30 + name_size + extra_size
        payload_end = payload_start + info.compress_size
    data[payload_start:payload_end] = b'\xff' * info.compress_size
    corrupted = bytes(data)

    with app.app_context(), pytest.raises(ValueError, match='not a dataset backup'):
        service.import_backup_zip(LOCAL_USER, io.BytesIO(corrupted))
    response = client.post(
        '/api/dataset/backup/import',
        data={'file': (io.BytesIO(corrupted), 'corrupt.zip')},
        content_type='multipart/form-data',
    )
    assert response.status_code == 400
    assert 'not a dataset backup' in response.get_json()['error']


def test_training_zip_rejects_oversized_image_before_read(app, monkeypatch):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as service

    with app.app_context():
        dataset = service.create_dataset(LOCAL_USER, 'Oversized image', 'oversized_image')
        assert service.DATASET_ZIP_MAX_IMAGE_BYTES == 128 * 1024 * 1024
        archive_stream = io.BytesIO()
        with zipfile.ZipFile(archive_stream, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('image.png', b'x' * 256)
        monkeypatch.setattr(service, 'DATASET_ZIP_MAX_IMAGE_BYTES', 128)

        def forbidden_read(*_args, **_kwargs):
            raise AssertionError('oversized image must be rejected before ZipFile.read')

        monkeypatch.setattr(zipfile.ZipFile, 'read', forbidden_read)
        with pytest.raises(ValueError, match='128 MiB per image'):
            service.import_dataset_zip(LOCAL_USER, dataset.id, archive_stream)


@pytest.mark.parametrize(
    ('entries', 'prefix'),
    [
        (('images/duplicate.webp', 'images/duplicate.webp'), 'images'),
        (('ref/Portrait.webp', 'ref/portrait.WEBP'), 'ref'),
    ],
)
def test_backup_rejects_exact_and_casefold_duplicates_before_extraction(
        app, entries, prefix):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as service

    output = io.BytesIO()
    with pytest.warns(UserWarning) if entries[0] == entries[1] else nullcontext():
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('manifest.json', json.dumps({
                'format': service.BACKUP_FORMAT,
                'version': service.BACKUP_VERSION,
                'name': 'Duplicate backup',
                'trigger_word': 'duplicate_backup',
            }))
            archive.writestr('images.json', '[]')
            archive.writestr(entries[0], b'first')
            archive.writestr(entries[1], b'second')

    with app.app_context(), pytest.raises(
            ValueError, match=rf'duplicate {prefix} filename'):
        service.import_backup_zip(LOCAL_USER, io.BytesIO(output.getvalue()))
    assert list(service.cfg.dataset_images_root().iterdir()) == []


@pytest.mark.parametrize(
    ('manifest_extra', 'images_meta', 'message'),
    [
        ({'version': '1'}, None, 'invalid backup version'),
        ({'name': ['not', 'text']}, None, 'invalid backup name'),
        ({'trigger_word': {'not': 'text'}}, None, 'invalid backup trigger_word'),
        ({'kind': {'not': 'text'}}, None, 'invalid backup kind'),
        ({'train_settings': '{"rank": NaN}'}, None,
         'invalid backup train_settings'),
        (None, [{'filename': ['not', 'text'], 'status': 'keep'}],
         'invalid backup image filename'),
        (None, [{'filename': 'bad.webp', 'status': {'not': 'text'}}],
         'invalid backup image status'),
        (None, [{'filename': 'bad.webp', 'status': 'keep',
                 'face_score': float('nan')}],
         'invalid backup image face_score'),
    ],
)
def test_malformed_backup_types_are_value_errors_and_route_400(
        app, client, manifest_extra, images_meta, message):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as service

    data = _empty_backup(
        service, manifest_extra=manifest_extra, images_meta=images_meta)
    with app.app_context(), pytest.raises(ValueError, match=message):
        service.import_backup_zip(LOCAL_USER, io.BytesIO(data))

    response = client.post(
        '/api/dataset/backup/import',
        data={'file': (io.BytesIO(data), 'malformed.zip')},
        content_type='multipart/form-data',
    )
    assert response.status_code == 400
    assert response.get_json()['error'] == message


@pytest.mark.parametrize(
    ('entry', 'images_meta', 'manifest_extra', 'message'),
    [
        ('images/orphan.webp', [], {}, 'unreferenced image in backup'),
        ('ref/orphan.webp', [], {}, 'unreferenced reference image in backup'),
        ('images/NUL.jpg', [{'filename': 'NUL.jpg', 'status': 'keep'}], {},
         'invalid backup image filename'),
    ],
)
def test_backup_v2_rejects_unowned_or_windows_device_payloads_before_extraction(
        app, entry, images_meta, manifest_extra, message):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as service

    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        manifest = {
            'format': service.BACKUP_FORMAT,
            'version': service.BACKUP_VERSION,
            'name': 'Owned payloads',
            'trigger_word': 'owned_payloads',
            **manifest_extra,
        }
        archive.writestr('manifest.json', json.dumps(manifest))
        archive.writestr('images.json', json.dumps(images_meta))
        archive.writestr(entry, b'not decoded because ownership fails first')

    with app.app_context(), pytest.raises(ValueError, match=message):
        service.import_backup_zip(LOCAL_USER, output.getvalue())
    assert list(service.cfg.dataset_images_root().iterdir()) == []


def test_backup_v2_rejects_unowned_analysis_cache(app):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as service

    cache_ref = 'a' * 64
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('manifest.json', json.dumps({
            'format': service.BACKUP_FORMAT,
            'version': service.BACKUP_VERSION,
            'name': 'No cache owner',
            'trigger_word': 'no_cache_owner',
        }))
        archive.writestr('images.json', '[]')
        archive.writestr(f'analysis-cache/{cache_ref}.npz', b'unowned')

    with app.app_context(), pytest.raises(
            ValueError, match='unreferenced analysis cache'):
        service.import_backup_zip(LOCAL_USER, output.getvalue())


def test_photo_import_gets_its_own_raised_ceiling(app, client, monkeypatch):
    """A drop of photographs is a local file copy, not a stranger's upload: the
    generic 64 MiB web default refused five to eight body shots (_nofaceman,
    Discord). The route that already knew how to raise a ceiling for archives
    raises one for photos too — bounded, because 20 files is the real cap."""
    app.config.update(
        MAX_CONTENT_LENGTH=128,
        DATASET_IMPORT_MAX_UPLOAD_BYTES=64 * 1024,
    )
    _make_max_content_length_read_only(monkeypatch)

    with app.test_request_context('/api/dataset/1/import', method='POST'):
        from flask import request
        request.url_rule = type('R', (), {'endpoint': 'datasets.dataset_import'})()
        assert request.max_content_length == 64 * 1024

    # …and the raise is that endpoint's privilege, not a global one.
    with app.test_request_context('/api/dataset/1/generate', method='POST'):
        from flask import request
        request.url_rule = type('R', (), {'endpoint': 'datasets.dataset_generate'})()
        assert request.max_content_length == 128


def test_the_import_ceiling_is_env_tunable_and_never_zero(monkeypatch):
    """Someone who shoots 60 MP masters can raise it; a broken value cannot
    silently disable the guard."""
    from app import (_DEFAULT_DATASET_IMPORT_MAX_UPLOAD_BYTES, _positive_env_int)
    monkeypatch.setenv('LDS_DATASET_IMPORT_MAX_UPLOAD_BYTES', str(3 * 1024 * 1024 * 1024))
    assert _positive_env_int('LDS_DATASET_IMPORT_MAX_UPLOAD_BYTES',
                             _DEFAULT_DATASET_IMPORT_MAX_UPLOAD_BYTES) == 3 * 1024 * 1024 * 1024
    for bad in ('0', '-1', 'lots', ''):
        monkeypatch.setenv('LDS_DATASET_IMPORT_MAX_UPLOAD_BYTES', bad)
        assert _positive_env_int('LDS_DATASET_IMPORT_MAX_UPLOAD_BYTES',
                                 _DEFAULT_DATASET_IMPORT_MAX_UPLOAD_BYTES) == 512 * 1024 * 1024
