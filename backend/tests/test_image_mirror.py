import json
from pathlib import Path

import pytest
from PIL import Image


def _asymmetric(path: Path, fmt='PNG'):
    image = Image.new('RGB', (96, 40), (240, 20, 20))
    image.paste((20, 20, 240), (48, 0, 96, 40))
    kwargs = {}
    if fmt == 'WEBP':
        kwargs = {'lossless': True, 'quality': 100}
    elif fmt == 'JPEG':
        kwargs = {'quality': 100, 'subsampling': 0}
    image.save(path, fmt, **kwargs)


def _seed(app, *, filename='asym.png', fmt='PNG', create_file=True,
          watermark=True):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, f'Mirror {filename}', 'mirror')
        path = Path(svc._dataset_dir(ds.id)) / filename
        if create_file:
            _asymmetric(path, fmt)
        metadata = json.dumps({
            'platform': 'pexels',
            'page_url': 'https://www.pexels.com/photo/example-123/',
            'photographer': 'Jane Example',
        })
        image = FaceDatasetImage(
            dataset_id=ds.id, filename=filename, source='import', status='keep',
            framing='body', caption='a person walking from left to right',
            variation_label='asymmetric', variation_prompt='a candid photo',
            klein_model='test-model', derivation_kind='manual',
            upscale_ratio=1.25, face_score=0.88, face_state='scorable',
            source_metadata=metadata,
            watermark_state='detected' if watermark else None,
            watermark_bbox='[0.1, 0.2, 0.3, 0.4]' if watermark else None,
            watermark_regions='[[0.1, 0.2, 0.3, 0.4]]' if watermark else None,
        )
        svc.db.session.add(image)
        svc.db.session.commit()
        return ds.id, image.id, path


def _pixels(path):
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        return rgb.size, rgb.getpixel((8, 20)), rgb.getpixel((87, 20)), image.format


@pytest.mark.parametrize('fmt,filename', [
    ('PNG', 'image.png'),
    ('WEBP', 'image.webp'),
    ('JPEG', 'image.jpg'),
])
def test_mirror_flips_pixels_and_preserves_real_format(app, fmt, filename):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    _ds_id, image_id, path = _seed(app, filename=filename, fmt=fmt)
    with app.app_context():
        result = svc.mirror_image(LOCAL_USER, image_id)
    size, left, right, detected = _pixels(path)
    assert result['image_id'] == image_id
    assert isinstance(result['cache_bust'], int) and result['cache_bust'] > 0
    assert size == (96, 40) and detected == fmt
    assert left[2] > left[0] + 100       # former blue right half is now left
    assert right[0] > right[2] + 100     # former red left half is now right


def test_double_mirror_restores_pixels_and_preserves_metadata(app):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    _ds_id, image_id, path = _seed(app)
    with Image.open(path) as image:
        original_pixels = image.convert('RGB').tobytes()
    with app.app_context():
        before = svc.db.session.get(FaceDatasetImage, image_id)
        stable = {
            name: getattr(before, name) for name in (
                'dataset_id', 'filename', 'source', 'status', 'framing', 'caption',
                'variation_label', 'variation_prompt', 'klein_model',
                'derivation_kind', 'upscale_ratio', 'face_score', 'face_state',
                'source_metadata', 'created_at',
            )
        }
        svc.mirror_image(LOCAL_USER, image_id)
        row = svc.db.session.get(FaceDatasetImage, image_id)
        assert all(getattr(row, name) == value for name, value in stable.items())
        assert (row.watermark_state, row.watermark_bbox, row.watermark_regions) == (
            None, None, None)
        svc.mirror_image(LOCAL_USER, image_id)
    with Image.open(path) as image:
        assert image.convert('RGB').tobytes() == original_pixels


def test_mirror_rejects_foreign_corrupt_and_unsupported_without_changes(app):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    _ds_id, image_id, path = _seed(app)
    original = path.read_bytes()
    with app.app_context():
        assert svc.mirror_image('another-user', image_id) is None
        assert svc.mirror_image(LOCAL_USER, 999999) is None
    assert path.read_bytes() == original

    path.write_bytes(b'not an image')
    with app.app_context(), pytest.raises(ValueError, match='invalid image file'):
        svc.mirror_image(LOCAL_USER, image_id)
    with app.app_context():
        row = svc.db.session.get(FaceDatasetImage, image_id)
        assert row.watermark_state == 'detected'
    assert path.read_bytes() == b'not an image'
    assert not list(path.parent.glob(f'.{path.name}.mirror-*.tmp'))

    Image.new('RGB', (10, 10)).save(path, 'GIF')
    with app.app_context(), pytest.raises(ValueError, match='unsupported image format'):
        svc.mirror_image(LOCAL_USER, image_id)


def test_replace_failure_restores_watermark_and_cleans_temp(app, monkeypatch):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    _ds_id, image_id, path = _seed(app)
    original = path.read_bytes()

    def fail_replace(_src, _dst):
        raise OSError('locked')

    monkeypatch.setattr(svc.os, 'replace', fail_replace)
    with app.app_context(), pytest.raises(RuntimeError, match='could not update image file'):
        svc.mirror_image(LOCAL_USER, image_id)
    assert path.read_bytes() == original
    assert not list(path.parent.glob(f'.{path.name}.mirror-*.tmp'))
    with app.app_context():
        row = svc.db.session.get(FaceDatasetImage, image_id)
        assert row.watermark_state == 'detected'
        assert row.watermark_bbox == '[0.1, 0.2, 0.3, 0.4]'
        assert row.watermark_regions == '[[0.1, 0.2, 0.3, 0.4]]'


def test_mirror_route_success_and_error_contracts(app, client):
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    _ds_id, image_id, _path = _seed(app, watermark=False)
    response = client.post(f'/api/dataset/image/{image_id}/mirror')
    assert response.status_code == 200
    body = response.get_json()
    assert set(body) == {'ok', 'image_id', 'cache_bust'}
    assert body['ok'] is True and body['image_id'] == image_id
    assert isinstance(body['cache_bust'], int) and body['cache_bust'] > 0
    assert client.post('/api/dataset/image/999999/mirror').status_code == 404

    with app.app_context():
        ds = svc.create_dataset('local', 'Pending mirror', 'pending-mirror')
        pending = FaceDatasetImage(dataset_id=ds.id, filename=None, status='pending')
        missing = FaceDatasetImage(dataset_id=ds.id, filename='missing.png', status='keep')
        svc.db.session.add_all([pending, missing])
        svc.db.session.commit()
        pending_id, missing_id = pending.id, missing.id
    assert client.post(f'/api/dataset/image/{pending_id}/mirror').status_code == 400
    assert client.post(f'/api/dataset/image/{missing_id}/mirror').status_code == 409

    _ds_id, corrupt_id, corrupt_path = _seed(app, filename='corrupt.png')
    corrupt_path.write_bytes(b'broken')
    assert client.post(f'/api/dataset/image/{corrupt_id}/mirror').status_code == 400
