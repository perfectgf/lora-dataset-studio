"""Video reads a gallery file through the owned gallery SDK, without rendering."""
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.plugins('video')


def source_request(image_id):
    return SimpleNamespace(files={}, get_json=lambda **_: {'gallery_image_id': image_id})


@pytest.mark.parametrize('user_id,filename,write_file,expected', [
    ('local', 'source.png', True, None),
    ('different-fixture-user', 'source.png', True, 'gallery'),
    ('local', 'source.png', False, 'disk'),
    ('local', '../source.png', True, 'disk'),
])
def test_gallery_source_is_owned_present_and_confined(app, user_id, filename, write_file, expected):
    from app.extensions import db
    from app.models import FaceDataset, LoraTestImage
    from app.services.dataset_storage import dataset_path
    from lds_video.routes.video_studio import _resolve_source
    with app.app_context():
        dataset = FaceDataset(user_id=user_id, name='Gallery fixture', trigger_word='fixture')
        db.session.add(dataset)
        db.session.commit()
        folder = Path(dataset_path(dataset.id))
        folder.mkdir(parents=True, exist_ok=True)
        file = folder / filename
        if write_file:
            file.write_bytes(b'fixture content: resolver does not decode images')
        row = LoraTestImage(dataset_id=dataset.id, filename=filename,
                            prompt='fixture', checkpoint='fixture.safetensors', strength=1.0, status='done')
        db.session.add(row)
        db.session.commit()
        if expected:
            with pytest.raises(ValueError, match=expected):
                _resolve_source(source_request(row.id))
        else:
            assert _resolve_source(source_request(row.id)) == (str(file), False)
            assert file.read_bytes().startswith(b'fixture content')
