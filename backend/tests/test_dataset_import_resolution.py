"""Dataset imports preserve masters; conversion is a training-time derivative.

The previous default was an implicit WebP q92 conversion at import. That meant a
dataset could never recover the file it was built from, even though the training
export already makes a disposable PNG copy. These tests pin the new split:

* default non-cropped imports retain approved source bytes and a content-derived
  extension (manual, ZIP and folder lanes);
* head crop remains an intentional WebP derivative;
* standard/high/lossless stay available as explicit legacy normalisation modes;
* the trainer, not the master dataset, is where a PNG conversion happens.
"""
import io
import os
import stat
import warnings
import zipfile

import pytest
from PIL import Image


def _image_bytes(fmt='PNG', size=(160, 100), seed=3):
    """A non-flat static image encoded under ``fmt``."""
    image = Image.new('RGB', size)
    pixels = image.load()
    for y in range(size[1]):
        for x in range(size[0]):
            pixels[x, y] = ((x * 5 + y * 7 + seed) % 256,
                            (x * 11 + y * 3 + seed * 5) % 256,
                            (x * 2 + y * 13 + seed * 17) % 256)
    out = io.BytesIO()
    if fmt == 'JPEG':
        image.save(out, fmt, quality=91, subsampling=0)
    elif fmt == 'WEBP':
        image.save(out, fmt, quality=91)
    else:
        image.save(out, fmt)
    return out.getvalue()


def _row(svc, dataset_id, image_id):
    from app.models import FaceDatasetImage
    row = svc.db.session.get(FaceDatasetImage, image_id)
    assert row is not None
    return row


def _stored_bytes(svc, dataset_id, row):
    return open(os.path.join(svc._dataset_dir(dataset_id), row.filename), 'rb').read()


@pytest.mark.parametrize('fmt, extension', [
    ('JPEG', '.jpg'),
    ('PNG', '.png'),
    ('WEBP', '.webp'),
    ('BMP', '.bmp'),
])
def test_default_non_cropped_import_preserves_bytes_and_content_extension(
        app, fmt, extension):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    raw = _image_bytes(fmt, seed=len(fmt))
    with app.app_context():
        policy = svc.import_encode_policy()
        assert policy['encoding'] == 'preserve' and policy['preserve'] is True
        dataset = svc.create_dataset(LOCAL_USER, f'Preserve {fmt}', f'preserve_{fmt}')
        ids, failed = svc.import_images(LOCAL_USER, dataset.id, [raw], crop=False)
        assert failed == 0 and len(ids) == 1
        row = _row(svc, dataset.id, ids[0])
        filename = row.filename
        stored = _stored_bytes(svc, dataset.id, row)

    assert filename.endswith(extension)
    assert stored == raw
    with Image.open(io.BytesIO(stored)) as image:
        assert image.format == fmt


def test_default_preserve_ignores_the_legacy_resolution_limit(app):
    """1024 remains the saved normalisation choice, but must not downscale a
    master when the default policy is `preserve`."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    raw = _image_bytes('PNG', size=(1500, 900))
    with app.app_context():
        dataset = svc.create_dataset(LOCAL_USER, 'No implicit resize', 'no_resize')
        ids, failed = svc.import_images(LOCAL_USER, dataset.id, [raw], crop=False)
        assert failed == 0
        row = _row(svc, dataset.id, ids[0])
        stored = _stored_bytes(svc, dataset.id, row)
    with Image.open(io.BytesIO(stored)) as image:
        assert image.size == (1500, 900)


def test_preserve_rejects_unsupported_or_animated_format_clearly(app):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    gif = io.BytesIO()
    Image.new('RGB', (20, 20), 'red').save(gif, 'GIF')
    with app.app_context():
        with pytest.raises(ValueError, match='preserve mode supports only static'):
            svc.import_store_image(gif.getvalue())
        dataset = svc.create_dataset(LOCAL_USER, 'Unsupported preserve', 'unsupported')
        ids, failed = svc.import_images(LOCAL_USER, dataset.id, [gif.getvalue()], crop=False)
    assert ids == [] and failed == 1


def test_head_crop_remains_an_intentional_webp_derivative(app, monkeypatch):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    raw = _image_bytes('PNG')
    derived = _image_bytes('WEBP', size=(64, 64))
    monkeypatch.setattr(svc, 'face_crop_to_square_webp',
                        lambda _raw, return_scale=False: (derived, 1.0))
    with app.app_context():
        dataset = svc.create_dataset(LOCAL_USER, 'Crop derives', 'crop_derives')
        ids, failed = svc.import_images(LOCAL_USER, dataset.id, [raw], crop=True)
        assert failed == 0 and len(ids) == 1
        row = _row(svc, dataset.id, ids[0])
        filename = row.filename
        stored = _stored_bytes(svc, dataset.id, row)
    assert filename.endswith('.webp') and stored == derived


def test_opt_in_standard_mode_keeps_the_old_normalised_webp_behaviour(app):
    from app.config import LOCAL_USER, save_config
    from app.services import face_dataset_service as svc

    raw = _image_bytes('PNG', size=(2000, 1200))
    with app.app_context():
        save_config({'dataset_import': {'max_side': 1024, 'encoding': 'standard'}})
        policy = svc.import_encode_policy()
        assert policy['preserve'] is False and policy['quality'] == 92
        dataset = svc.create_dataset(LOCAL_USER, 'Explicit standard', 'explicit_standard')
        ids, failed = svc.import_images(LOCAL_USER, dataset.id, [raw], crop=False)
        assert failed == 0 and len(ids) == 1
        row = _row(svc, dataset.id, ids[0])
        filename = row.filename
        stored = _stored_bytes(svc, dataset.id, row)
    assert filename.endswith('.webp') and stored != raw
    with Image.open(io.BytesIO(stored)) as image:
        assert image.format == 'WEBP' and image.size == (1024, 614)


def test_explicit_encoding_tiers_and_ceiling_remain_available(app):
    from app.config import save_config
    from app.services import face_dataset_service as svc

    raw = _image_bytes('PNG', size=(400, 300))
    with app.app_context():
        save_config({'dataset_import': {'max_side': 0, 'encoding': 'lossless'}})
        lossless, ext = svc.import_store_image(raw)
        assert ext == '.webp' and svc.import_encode_policy()['lossless'] is True
        save_config({'dataset_import': {'max_side': 0, 'encoding': 'standard'}})
        standard, ext = svc.import_store_image(raw)
        assert ext == '.webp' and svc.import_encode_policy()['quality'] == 92
        save_config({'dataset_import': {'max_side': 999999, 'encoding': 'standard'}})
        policy = svc.import_encode_policy()

    source = Image.open(io.BytesIO(raw)).convert('RGB')
    # tobytes() both sides: byte equality is pixel equality for one mode,
    # and getdata() is deprecated for removal in Pillow 14.
    assert Image.open(io.BytesIO(lossless)).convert('RGB').tobytes() == source.tobytes()
    assert Image.open(io.BytesIO(standard)).convert('RGB').tobytes() != source.tobytes()
    assert policy['max_side'] == svc.IMPORT_MAX_SIDE_CEILING and policy['capped'] is True


def test_invalid_policy_falls_back_to_preserve_default(app):
    from app.config import save_config
    from app.services import face_dataset_service as svc

    with app.app_context():
        save_config({'dataset_import': {'max_side': 'wide', 'encoding': 'ultra'}})
        policy = svc.import_encode_policy()
    assert policy['max_side'] == 1024
    assert policy['encoding'] == 'preserve' and policy['preserve'] is True
    assert policy['capped'] is False


class _OversizedImageHeader:
    """Pillow-like header object whose `load` must never be reached in a test."""
    format = 'JPEG'
    size = (12000, 9000)  # sides within budget, 103 Mi-pixel raster: over it
    n_frames = 1

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def load(self):
        raise AssertionError('unsafe header was decoded before it was rejected')


def test_import_header_budget_rejects_before_decode_for_every_ingress_lane(app, monkeypatch):
    """Manual, ZIP, scrape, crop and normalisation share ONE pixel gate."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    monkeypatch.setattr(svc.Image, 'open', lambda *_args, **_kwargs: _OversizedImageHeader())
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as zf:
        zf.writestr('train/unsafe.jpg', b'header-only')

    with app.app_context():
        manual = svc.create_dataset(LOCAL_USER, 'Manual guarded', 'manual_guarded')
        ids, failed = svc.import_images(LOCAL_USER, manual.id, [b'header-only'], crop=False)
        assert ids == [] and failed == 1

        zipped = svc.create_dataset(LOCAL_USER, 'Zip guarded', 'zip_guarded')
        ids, failed = svc.import_dataset_zip(LOCAL_USER, zipped.id, archive.getvalue())
        assert ids == [] and failed == 1

        scraped = svc.create_dataset(LOCAL_USER, 'Scrape guarded', 'scrape_guarded',
                                     kind='concept', concept_desc='a guarded concept')
        monkeypatch.setattr(svc, '_download_scrape_item',
                            lambda _item: ('ok', b'header-only'))
        result = svc.scrape_import_urls(
            LOCAL_USER, scraped.id, [{'url': 'https://example.invalid/unsafe.jpg'}])
        assert result['imported'] == 0 and result['skipped']['errors'] == 1

        with pytest.raises(ValueError, match='reduce the image before import'):
            svc.normalize_to_webp(b'header-only')
        with pytest.raises(ValueError, match='reduce the image before import'):
            svc.face_crop_to_square_webp(b'header-only', use_vision=False)


def test_preserve_bomb_warning_is_local_and_never_changes_global_filters(app, monkeypatch):
    """A Pillow bomb warning becomes a clear rejection without global state leaks."""
    from app.services import face_dataset_service as svc

    before = list(warnings.filters)

    def _bomb(*_args, **_kwargs):
        raise Image.DecompressionBombWarning('crafted header')

    monkeypatch.setattr(svc.Image, 'open', _bomb)
    with app.app_context(), pytest.raises(ValueError, match='unsafe image header'):
        svc.import_store_image(b'crafted')
    assert warnings.filters == before


def test_zip_and_folder_merge_preserve_raw_sources(app, tmp_path):
    """ZIP and folder routes share `_merge_training_images`; cover both public
    entries so a future shortcut cannot reintroduce a WebP rewrite in one lane."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    zip_raw = _image_bytes('PNG', seed=12)
    folder_raw = _image_bytes('BMP', seed=18)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('kohya/zip-master.png', zip_raw)
    folder = tmp_path / 'training-folder'
    folder.mkdir()
    (folder / 'folder-master.bmp').write_bytes(folder_raw)

    with app.app_context():
        zip_dataset = svc.create_dataset(LOCAL_USER, 'ZIP source', 'zip_source')
        ids, failed = svc.import_dataset_zip(LOCAL_USER, zip_dataset.id, archive.getvalue())
        assert failed == 0 and len(ids) == 1
        zip_row = _row(svc, zip_dataset.id, ids[0])
        assert zip_row.filename.endswith('.png')
        assert _stored_bytes(svc, zip_dataset.id, zip_row) == zip_raw

        folder_dataset = svc.create_dataset(LOCAL_USER, 'Folder source', 'folder_source')
        ids, failed = svc.import_dataset_folder(LOCAL_USER, folder_dataset.id, str(folder))
        assert failed == 0 and len(ids) == 1
        folder_row = _row(svc, folder_dataset.id, ids[0])
        assert folder_row.filename.endswith('.bmp')
        assert _stored_bytes(svc, folder_dataset.id, folder_row) == folder_raw


def test_folder_import_rejects_oversized_regular_file_before_reading(
        app, tmp_path, monkeypatch):
    """A sparse/live folder file over ZIP's per-image cap never reaches ``open``."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    folder = tmp_path / 'oversized-folder'
    folder.mkdir()
    source = folder / 'sparse.png'
    source.write_bytes(_image_bytes('PNG'))
    real_lstat = svc.os.lstat
    source_stat = real_lstat(source)
    real_open = open
    reads = []

    class _LargeRegular:
        st_mode = source_stat.st_mode
        st_size = svc.DATASET_ZIP_MAX_IMAGE_BYTES + 1

    def _reported_lstat(path):
        if os.path.normcase(os.fspath(path)) == os.path.normcase(str(source)):
            return _LargeRegular()
        return real_lstat(path)

    def _guarded_open(path, *args, **kwargs):
        if os.path.normcase(os.fspath(path)) == os.path.normcase(str(source)):
            reads.append(path)
            raise AssertionError('oversized folder image was opened')
        return real_open(path, *args, **kwargs)

    with app.app_context():
        dataset = svc.create_dataset(LOCAL_USER, 'Folder cap', 'folder_cap')
        monkeypatch.setattr(svc.os, 'lstat', _reported_lstat)
        monkeypatch.setattr(svc, 'open', _guarded_open, raising=False)
        with pytest.raises(ValueError, match='image too large in folder'):
            svc.import_dataset_folder(LOCAL_USER, dataset.id, str(folder))
    assert reads == []


def test_folder_import_skips_nonregular_image_without_opening(
        app, tmp_path, monkeypatch):
    """A named-pipe/symlink-shaped entry counts as failed but never blocks on open."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    folder = tmp_path / 'special-folder'
    folder.mkdir()
    source = folder / 'pipe.png'
    source.write_bytes(_image_bytes('PNG'))
    real_lstat = svc.os.lstat
    real_open = open
    reads = []

    class _Pipe:
        st_mode = stat.S_IFIFO
        st_size = 0

    def _reported_lstat(path):
        if os.path.normcase(os.fspath(path)) == os.path.normcase(str(source)):
            return _Pipe()
        return real_lstat(path)

    def _guarded_open(path, *args, **kwargs):
        if os.path.normcase(os.fspath(path)) == os.path.normcase(str(source)):
            reads.append(path)
            raise AssertionError('non-regular folder image was opened')
        return real_open(path, *args, **kwargs)

    with app.app_context():
        dataset = svc.create_dataset(LOCAL_USER, 'Special folder', 'special_folder')
        monkeypatch.setattr(svc.os, 'lstat', _reported_lstat)
        monkeypatch.setattr(svc, 'open', _guarded_open, raising=False)
        ids, failed = svc.import_dataset_folder(LOCAL_USER, dataset.id, str(folder))
    assert ids == [] and failed == 1
    assert reads == []


def test_training_staging_is_png_and_never_mutates_the_preserved_master(app, tmp_path):
    """Orientation is baked only into the disposable PNG: the JPEG master stays
    byte-identical in the dataset while AI Toolkit gets upright pixels."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_training as training

    image = Image.new('RGB', (40, 20), (40, 130, 220))
    exif = image.getexif()
    exif[274] = 6  # display is clockwise-rotated relative to stored pixels
    raw = io.BytesIO()
    image.save(raw, 'JPEG', quality=95, subsampling=0, exif=exif)
    raw = raw.getvalue()

    with app.app_context():
        dataset = svc.create_dataset(LOCAL_USER, 'Stage only', 'stage_only')
        ids, failed = svc.import_images(LOCAL_USER, dataset.id, [raw], crop=False)
        assert failed == 0 and len(ids) == 1
        row = _row(svc, dataset.id, ids[0])
        master = os.path.join(svc._dataset_dir(dataset.id), row.filename)
        before = open(master, 'rb').read()
        output = training.export_dataset_to_aitoolkit(
            LOCAL_USER, dataset.id, masked=False, dest_dir=tmp_path / 'staging')
        staged = next((tmp_path / 'staging').glob('*.png'))

    assert before == raw == open(master, 'rb').read()
    assert output == str(tmp_path / 'staging')
    with Image.open(staged) as staged_image:
        assert staged_image.format == 'PNG' and staged_image.size == (20, 40)


def test_settings_and_capabilities_advertise_preserve_as_the_default(client):
    settings = client.get('/api/settings').get_json()
    assert settings['config']['dataset_import']['max_side'] == 1024
    assert settings['config_defaults']['dataset_import']['encoding'] == 'preserve'
    assert client.put('/api/settings', json={
        'config': {'dataset_import': {'max_side': 2048, 'encoding': 'high'}}}).status_code == 200
    assert client.get('/api/settings').get_json()['config']['dataset_import'] == {
        'max_side': 2048, 'encoding': 'high'}

    capability = client.get('/api/capabilities').get_json()['dataset_import']
    assert capability['encoding'] == 'high'
    assert capability['max_side'] == 2048 and capability['ceiling'] >= 2048
    # The shipped budget, which is now a setting (see test_image_input_budget.py).
    assert capability['input_max_side'] == 16384
    assert capability['input_max_pixels'] == 64 * 1024 * 1024
