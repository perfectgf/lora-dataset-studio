""""Import to bank" is the reverse of promote: a dataset's kept images become a
new bank, under a name the user picks, so its material can be re-triaged with the
bank tools. It COPIES rather than pointing the bank at the dataset's live folder —
sharing files would mean curating one mutates the other — which mirrors promote,
copying in the other direction."""
import os

import pytest


def _seed(dataset_id, names, statuses=None):
    """A dataset row per name, with the matching file on disk."""
    from app.extensions import db
    from app.models import FaceDatasetImage
    from app.services.dataset_storage import ensure_dataset_dir
    folder = ensure_dataset_dir(dataset_id)
    for i, fn in enumerate(names):
        with open(os.path.join(folder, fn), 'wb') as fh:
            fh.write(b'\x89PNG' + bytes([i]))
        db.session.add(FaceDatasetImage(
            dataset_id=dataset_id, filename=fn,
            status=(statuses[i] if statuses else 'keep')))
    db.session.commit()
    return str(folder)


@pytest.fixture()
def ds(app):
    with app.app_context():
        from app.extensions import db
        from app.models import FaceDataset
        d = FaceDataset(user_id='local', name='Lola', trigger_word='Lola69382')
        db.session.add(d)
        db.session.commit()
        yield d


def _run_import(app, dataset_id, name):
    """Start the import and run its job body inline, so the copy is finished when
    the assertions run (bank_jobs.start spawns a thread in production)."""
    from unittest.mock import patch
    from app.services import image_bank_service as banks
    captured = {}

    def fake_start(_app, bank_id, kind, fn, total=0):
        captured['bank_id'] = bank_id
        captured['kind'] = kind
        fn(object())          # the job body ignores the handle except via bank_jobs
        return None

    with patch.object(banks.bank_jobs, 'start', fake_start), \
         patch.object(banks.bank_jobs, 'cancelled', lambda job: False), \
         patch.object(banks.bank_jobs, 'bump', lambda job, n=1: None), \
         patch.object(banks.bank_jobs, 'progress', lambda job, **kw: None):
        bank_id = banks.start_dataset_import(app, 'local', dataset_id, name)
    return bank_id, captured


def test_import_copies_kept_images_into_a_bank_of_its_own(app, ds):
    with app.app_context():
        from app.models import BankImage, ImageBank
        from app.extensions import db
        src = _seed(ds.id, ['a.png', 'b.png', 'c.png'],
                    statuses=['keep', 'reject', 'keep'])

        bank_id, cap = _run_import(app, ds.id, 'Lola archive')
        bank = db.session.get(ImageBank, bank_id)
        assert bank.name == 'Lola archive'
        # a folder of its OWN — never the dataset's live folder
        assert os.path.realpath(bank.source_path) != os.path.realpath(src)
        rows = BankImage.query.filter_by(bank_id=bank_id).all()
        assert sorted(r.relpath for r in rows) == ['a.png', 'c.png']   # rejected left out
        for r in rows:
            copied = os.path.join(bank.source_path, r.relpath)
            assert os.path.isfile(copied) and r.file_size > 0
            # the dataset keeps its own copy: this is a copy, not a move
            assert os.path.isfile(os.path.join(src, r.relpath))


def test_two_imports_of_the_same_name_never_share_a_folder(app, ds):
    """Reusing a folder would silently MERGE two imports into one set of files."""
    with app.app_context():
        from app.models import ImageBank
        from app.extensions import db
        _seed(ds.id, ['a.png'])
        first, _ = _run_import(app, ds.id, 'Same name')
        second, _ = _run_import(app, ds.id, 'Same name')
        a = db.session.get(ImageBank, first).source_path
        b = db.session.get(ImageBank, second).source_path
        assert os.path.realpath(a) != os.path.realpath(b)


def test_import_refuses_without_a_name_or_kept_images(app, ds):
    with app.app_context():
        from app.services import image_bank_service as banks
        _seed(ds.id, ['a.png'], statuses=['reject'])
        with pytest.raises(ValueError):                      # nothing kept
            banks.start_dataset_import(app, 'local', ds.id, 'Whatever')
        with pytest.raises(ValueError):                      # blank name
            banks.start_dataset_import(app, 'local', ds.id, '   ')
        with pytest.raises(ValueError):                      # unknown dataset
            banks.start_dataset_import(app, 'local', 99999, 'Whatever')


def test_import_survives_a_row_whose_file_is_gone(app, ds):
    """A dataset row can outlive its file; that must cost one image, not the import."""
    with app.app_context():
        from app.models import BankImage
        src = _seed(ds.id, ['a.png', 'b.png'])
        os.remove(os.path.join(src, 'a.png'))
        bank_id, _ = _run_import(app, ds.id, 'Partial')
        rows = BankImage.query.filter_by(bank_id=bank_id).all()
        assert [r.relpath for r in rows] == ['b.png']


def test_import_route_returns_the_new_bank_id(client, app, ds):
    with app.app_context():
        _seed(ds.id, ['a.png'])
    from unittest.mock import patch
    from app.services import image_bank_service as banks
    with patch.object(banks, 'start_dataset_import', return_value=42):
        r = client.post('/api/bank/from-dataset',
                        json={'dataset_id': ds.id, 'name': 'Lola archive'})
    assert r.status_code == 202 and r.get_json()['id'] == 42


def test_import_route_maps_a_bad_request_to_400(client, app, ds):
    r = client.post('/api/bank/from-dataset', json={'dataset_id': ds.id, 'name': ''})
    assert r.status_code == 400
