"""'Back up everything' — the master archive that bundles every dataset's
portable backup plus a secrets-free config, and the restore that rebuilds them."""
import io
import os
import zipfile

import pytest
from PIL import Image


def _png(color=(200, 40, 40)):
    buf = io.BytesIO()
    Image.new('RGB', (48, 48), color).save(buf, 'PNG')
    return buf.getvalue()


def _dataset_with_image(svc, user, name, trigger, *, caption='a portrait',
                        filename='img1.webp'):
    """A dataset with one real on-disk image row — so a restore can be proven to
    carry pixels, not just metadata."""
    from app.models import FaceDatasetImage
    ds = svc.create_dataset(user, name, trigger)
    d = svc._dataset_dir(ds.id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, filename), 'wb') as fh:
        fh.write(_png())
    svc.db.session.add(FaceDatasetImage(dataset_id=ds.id, filename=filename,
                                        status='keep', framing='face', caption=caption))
    svc.db.session.commit()
    return ds


def _all_plaintext(archive_bytes):
    """Every entry's DECOMPRESSED bytes, descending into nested .zip entries —
    so a secret-scan can't be fooled by deflate compression."""
    out = []
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            data = z.read(info)
            out.append(data)
            if info.filename.endswith('.zip'):
                out.extend(_all_plaintext(data))
    return out


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def test_full_backup_bundles_every_dataset_and_config(app, tmp_path):
    import json
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import full_backup as fb

    with app.app_context():
        _dataset_with_image(svc, LOCAL_USER, 'Alice', 'alice')
        _dataset_with_image(svc, LOCAL_USER, 'Bob', 'bob')
        _dataset_with_image(svc, LOCAL_USER, 'Carol', 'carol')

        out = str(tmp_path / 'master.zip')
        result = fb.build_full_backup(LOCAL_USER, out)

    assert result['ok'] and result['datasets_total'] == 3
    assert result['datasets_backed_up'] == 3 and result['skipped'] == []
    assert os.path.getsize(out) == result['size_bytes']

    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        assert 'manifest.json' in names and 'config.json' in names
        entries = [n for n in names if n.startswith('datasets/') and n.endswith('.zip')]
        assert len(entries) == 3
        manifest = json.loads(z.read('manifest.json'))
        assert manifest['format'] == fb.FULL_BACKUP_FORMAT
        assert manifest['version'] == fb.FULL_BACKUP_VERSION
        assert {d['name'] for d in manifest['datasets']} == {'Alice', 'Bob', 'Carol'}
        assert all(d['images'] == 1 for d in manifest['datasets'])
        # Each embedded entry is itself a valid single-dataset backup.
        for entry in entries:
            inner = zipfile.ZipFile(io.BytesIO(z.read(entry)))
            assert 'manifest.json' in inner.namelist()


def test_full_backup_never_contains_secrets(app, tmp_path, monkeypatch):
    """The archive must not carry any API key / token — proven by scanning every
    decompressed byte, including nested per-dataset archives."""
    import json
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import full_backup as fb
    from app import config as cfg

    api_secret = 'sk-LEAKCANARY-0123456789'
    lan_secret = 'LANTOKEN-CANARY-abcdef'
    monkeypatch.setenv('OPENAI_API_KEY', api_secret)
    cfg.set_secrets({'HF_TOKEN': 'hf_CANARY_zzz'})

    with app.app_context():
        cfg.save_config({'server': {'access_token': lan_secret, 'require_token': True}})
        _dataset_with_image(svc, LOCAL_USER, 'Alice', 'alice')
        out = str(tmp_path / 'master.zip')
        fb.build_full_backup(LOCAL_USER, out)

    raw = open(out, 'rb').read()
    for chunk in _all_plaintext(raw):
        assert api_secret.encode() not in chunk
        assert lan_secret.encode() not in chunk
        assert b'hf_CANARY_zzz' not in chunk
        assert b'OPENAI_API_KEY' not in chunk

    with zipfile.ZipFile(out) as z:
        conf = json.loads(z.read('config.json'))
    # Config IS included (engines/training/etc.) but the LAN token is blanked.
    assert conf['server']['access_token'] == ''
    assert conf['server']['require_token'] is True
    assert 'engines' in conf and 'training' in conf


def test_build_skips_unreadable_dataset_without_aborting(app, tmp_path, monkeypatch):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import full_backup as fb

    with app.app_context():
        good = _dataset_with_image(svc, LOCAL_USER, 'Good', 'good')
        bad = _dataset_with_image(svc, LOCAL_USER, 'Bad', 'bad')

        real = svc.write_backup_zip

        def flaky(user, dataset_id, output):
            if dataset_id == bad.id:
                raise RuntimeError('C:/Users/somebody/locked.webp is busy')
            return real(user, dataset_id, output)

        monkeypatch.setattr(svc, 'write_backup_zip', flaky)
        out = str(tmp_path / 'master.zip')
        result = fb.build_full_backup(LOCAL_USER, out)

    assert result['datasets_total'] == 2 and result['datasets_backed_up'] == 1
    assert len(result['skipped']) == 1
    skipped = result['skipped'][0]
    assert skipped['name'] == 'Bad'
    # The surfaced reason stays paste-safe (no home path leaks through).
    assert 'C:/Users/somebody' not in skipped['reason']


def test_disk_space_precheck_blocks_build(app, tmp_path, monkeypatch):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import full_backup as fb

    with app.app_context():
        _dataset_with_image(svc, LOCAL_USER, 'Alice', 'alice')
        monkeypatch.setattr(fb, '_DISK_SAFETY_BYTES', 10 ** 18)  # more than any real disk
        with pytest.raises(fb.DiskSpaceError):
            fb.build_full_backup(LOCAL_USER, str(tmp_path / 'master.zip'))
        # With the check waived the same build succeeds.
        result = fb.build_full_backup(LOCAL_USER, str(tmp_path / 'ok.zip'),
                                      check_disk=False)
        assert result['ok']


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------

def test_restore_rebuilds_datasets_on_clean_db(app, tmp_path):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import full_backup as fb
    from app.models import FaceDatasetImage

    with app.app_context():
        _dataset_with_image(svc, LOCAL_USER, 'Alice', 'alice', caption='alice smiling')
        _dataset_with_image(svc, LOCAL_USER, 'Bob', 'bob', caption='bob waving')
        out = str(tmp_path / 'master.zip')
        fb.build_full_backup(LOCAL_USER, out)

        # Simulate a fresh install: remove every dataset first.
        for ds in list(svc.list_datasets(LOCAL_USER)):
            svc.delete_dataset(LOCAL_USER, ds.id)
        assert svc.list_datasets(LOCAL_USER) == []

        report = fb.restore_full_backup(LOCAL_USER, out)
        assert report['ok'] and report['restored'] == 2
        assert report['skipped'] == [] and report['renamed'] == []

        by_name = {d.name: d for d in svc.list_datasets(LOCAL_USER)}
        assert set(by_name) == {'Alice', 'Bob'}
        alice_imgs = FaceDatasetImage.query.filter_by(dataset_id=by_name['Alice'].id).all()
        assert len(alice_imgs) == 1 and alice_imgs[0].caption == 'alice smiling'
        # The pixels came back too, not just the row.
        d = svc._dataset_dir(by_name['Alice'].id)
        assert os.path.isfile(os.path.join(d, alice_imgs[0].filename))


def test_restore_name_collision_gets_a_suffix(app, tmp_path):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import full_backup as fb

    with app.app_context():
        _dataset_with_image(svc, LOCAL_USER, 'Alice', 'alice')
        out = str(tmp_path / 'master.zip')
        fb.build_full_backup(LOCAL_USER, out)

        # 'Alice' still present when the archive is restored back in.
        report = fb.restore_full_backup(LOCAL_USER, out)
        assert report['restored'] == 1
        assert report['renamed'] == [{'from': 'Alice', 'to': 'Alice (restored)'}]
        names = sorted(d.name for d in svc.list_datasets(LOCAL_USER))
        assert names == ['Alice', 'Alice (restored)']

        # A second restore stacks another suffix, never overwrites.
        report2 = fb.restore_full_backup(LOCAL_USER, out)
        assert report2['renamed'] == [{'from': 'Alice', 'to': 'Alice (restored 2)'}]
        assert len(svc.list_datasets(LOCAL_USER)) == 3


def test_restore_config_merge_is_non_destructive(app, tmp_path, monkeypatch):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import full_backup as fb
    from app import config as cfg

    with app.app_context():
        # Source machine: a distinctive, non-default engine choice + a dataset.
        cfg.save_config({'engines': {'default': 'nanobanana'}})
        _dataset_with_image(svc, LOCAL_USER, 'Alice', 'alice')
        out = str(tmp_path / 'master.zip')
        fb.build_full_backup(LOCAL_USER, out)

        # New machine: different engine default, an access token already set, and a
        # secret already entered — none of which the restore may clobber.
        cfg.save_config({'engines': {'default': 'chatgpt'},
                         'server': {'access_token': 'KEEPME', 'require_token': True}})
        monkeypatch.setenv('OPENAI_API_KEY', 'sk-already-here')

        report = fb.restore_full_backup(LOCAL_USER, out)
        assert report['config_restored'] is True

        conf = cfg.load_config()
        assert conf['engines']['default'] == 'nanobanana'       # merged from archive
        assert conf['server']['access_token'] == 'KEEPME'       # NOT overwritten
        assert cfg.secret('OPENAI_API_KEY') == 'sk-already-here'  # .env untouched


def test_restore_rejects_non_full_backup(app, tmp_path):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import full_backup as fb

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Alice', 'alice')
        single = svc.build_backup_zip(LOCAL_USER, ds.id)   # a per-dataset backup
        p = str(tmp_path / 'single.zip')
        open(p, 'wb').write(single)
        with pytest.raises(ValueError, match='not a full backup'):
            fb.restore_full_backup(LOCAL_USER, p)


# ---------------------------------------------------------------------------
# Config export helpers
# ---------------------------------------------------------------------------

def test_export_safe_config_blanks_token_and_sanitize_drops_it(app):
    from app.services import full_backup as fb
    from app import config as cfg

    with app.app_context():
        cfg.save_config({'server': {'access_token': 'TOP-SECRET'}})
        exported = fb.export_safe_config()
        assert exported['server']['access_token'] == ''

        # On restore the sensitive path is DROPPED (not blanked to ''), so a merge
        # can't wipe a token the user already set on the new machine.
        incoming = {'server': {'access_token': 'FROM-ARCHIVE', 'require_token': True},
                    'unknown_section': {'x': 1}}
        sanitized = fb._sanitize_incoming_config(incoming)
        assert 'access_token' not in sanitized['server']
        assert sanitized['server']['require_token'] is True
        assert 'unknown_section' not in sanitized   # only known DEFAULTS sections survive


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def test_restore_route_detects_single_vs_master(app, client, tmp_path):
    import time
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import full_backup as fb

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Solo', 'solo')
        single = svc.build_backup_zip(LOCAL_USER, ds.id)
        out = str(tmp_path / 'master.zip')
        fb.build_full_backup(LOCAL_USER, out, check_disk=False)
    master_bytes = open(out, 'rb').read()

    # A single-dataset backup is imported inline.
    r1 = client.post('/api/backup/full/restore',
                     data={'file': (io.BytesIO(single), 'backup.zip')},
                     content_type='multipart/form-data')
    assert r1.status_code == 200
    body1 = r1.get_json()
    assert body1['ok'] and body1['kind'] == 'single' and body1['name'] == 'Solo'

    # A master archive starts a background restore job.
    r2 = client.post('/api/backup/full/restore',
                     data={'file': (io.BytesIO(master_bytes), 'master.zip')},
                     content_type='multipart/form-data')
    assert r2.status_code == 200 and r2.get_json()['kind'] == 'full'

    # Poll until the job settles.
    for _ in range(200):
        st = client.get('/api/backup/full/restore/status').get_json()
        if st['state'] in ('done', 'error'):
            break
        time.sleep(0.02)
    assert st['state'] == 'done', st
    assert st['result']['restored'] == 1


def test_restore_route_rejects_garbage(app, client):
    r = client.post('/api/backup/full/restore',
                    data={'file': (io.BytesIO(b'not a zip'), 'x.zip')},
                    content_type='multipart/form-data')
    assert r.status_code == 400
    assert 'not a LoRA Dataset Studio backup' in r.get_json()['error']


def test_download_route_validates_basename(app, client, tmp_path):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import full_backup as fb
    from app import config as cfg

    with app.app_context():
        _dataset_with_image(svc, LOCAL_USER, 'Alice', 'alice')
        name = 'lds-full-backup-download-test.zip'
        out = str(cfg.backups_dir() / name)
        fb.build_full_backup(LOCAL_USER, out, check_disk=False)

    ok = client.get(f'/api/backup/full/download?name={name}')
    assert ok.status_code == 200 and ok.mimetype == 'application/zip'

    for bad in ('../config.json', 'nope.zip', 'sub/dir.zip'):
        assert client.get(f'/api/backup/full/download?name={bad}').status_code == 404
