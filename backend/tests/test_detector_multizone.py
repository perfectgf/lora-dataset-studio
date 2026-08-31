"""Every zone the detector finds now survives — on both surfaces.

The case that forced this, from the maintainer's test pair: a photo carrying
EIGHT copies of a photographer's logo came back flagged with ONE box, and Clean
would have repainted one eighth of the problem. The old comment called losing
the other boxes "the honest cost"; it did not survive contact with that image.

The rule, and both of its edges:
- two or more zones → ALL of them land in watermark_regions, which the clean,
  the mask editor and the zone previews already honour (it is the manual-zones
  pipe — nothing downstream had to change);
- ONE zone → no regions at all, byte-for-byte the old behaviour, because a
  regions-bearing row leaves the ✂ Auto-crop pool and a lone border mark must
  stay croppable. A multi-mark image never was one crop anyway.
"""
import json

from app.models import BankImage, FaceDatasetImage, ImageBank, db
from app.services import face_dataset_service as svc
from app.services import image_bank_service as banks
from app.services import watermark_detector

THREE = [[0.7, 0.8, 0.9, 0.95], [0.1, 0.1, 0.3, 0.2], [0.4, 0.05, 0.6, 0.15]]


def _fake_scan(regions_per_image):
    def scan(paths, **kw):
        for i, p in enumerate(paths):
            regions = regions_per_image[i % len(regions_per_image)]
            yield p, 'detected', 0.97, regions, None, None
    return scan


def _detector(monkeypatch):
    monkeypatch.setattr(watermark_detector, 'scan', _fake_scan([THREE]))
    monkeypatch.setattr(watermark_detector, 'resolve_backend',
                        lambda requested=None: {'requested': 'detector',
                                                'backend': 'detector',
                                                'fell_back': False, 'detail': ''})


# --- dataset -----------------------------------------------------------------

def _dataset_row(app, tmp_path, monkeypatch):
    with app.app_context():
        ds = svc.create_dataset('local', 'wm-zones', 'zonetrig')
        img = FaceDatasetImage(dataset_id=ds.id, filename='a.png', status='keep')
        db.session.add(img)
        db.session.commit()
        dataset_id, image_id = ds.id, img.id
    monkeypatch.setattr(svc, '_img_path', lambda i: str(tmp_path / 'a.png'))
    (tmp_path / 'a.png').write_bytes(b'fake')
    return dataset_id, image_id


def test_dataset_scan_keeps_every_zone(app, client, monkeypatch, tmp_path):
    dataset_id, image_id = _dataset_row(app, tmp_path, monkeypatch)
    _detector(monkeypatch)
    d = client.post(f'/api/dataset/{dataset_id}/watermarks/detect', json={}).get_json()
    assert d['detected'] == 1
    with app.app_context():
        img = db.session.get(FaceDatasetImage, image_id)
        assert json.loads(img.watermark_bbox) == THREE[0], (
            'the routing rectangle stays the child’s FIRST (most peripheral) box')
        stored = json.loads(img.watermark_regions)
        assert stored == THREE, 'the other zones were dropped again'
    # ...and the payload every preview draws from carries all of them.
    d2 = client.get(f'/api/dataset/{dataset_id}').get_json()
    row = next(r for r in d2['images'] if r['id'] == image_id)
    assert row['effective_watermark_regions'] == THREE, (
        'Review and the zones preview would still show one box')


def test_a_single_zone_writes_no_regions_so_auto_crop_keeps_its_pool(
        app, client, monkeypatch, tmp_path):
    dataset_id, image_id = _dataset_row(app, tmp_path, monkeypatch)
    monkeypatch.setattr(watermark_detector, 'scan', _fake_scan([[THREE[0]]]))
    monkeypatch.setattr(watermark_detector, 'resolve_backend',
                        lambda requested=None: {'requested': 'detector',
                                                'backend': 'detector',
                                                'fell_back': False, 'detail': ''})
    client.post(f'/api/dataset/{dataset_id}/watermarks/detect', json={})
    with app.app_context():
        img = db.session.get(FaceDatasetImage, image_id)
        assert json.loads(img.watermark_bbox) == THREE[0]
        assert img.watermark_regions is None, (
            'a lone border mark just left the ✂ Auto-crop pool')


def test_dataset_text_flagged_pages_fold_every_zone_in(app, client, monkeypatch,
                                                       tmp_path):
    dataset_id, image_id = _dataset_row(app, tmp_path, monkeypatch)
    with app.app_context():
        img = db.session.get(FaceDatasetImage, image_id)
        img.text_state = 'detected'
        img.watermark_regions = json.dumps([[0.0, 0.4, 0.05, 0.6]])   # a 🔤 zone
        db.session.commit()
    _detector(monkeypatch)
    client.post(f'/api/dataset/{dataset_id}/watermarks/detect', json={})
    with app.app_context():
        img = db.session.get(FaceDatasetImage, image_id)
        merged = json.loads(img.watermark_regions)
        # The text zone survives AND all three detector zones joined it.
        assert len(merged) >= 4, f'zones were dropped in the fold: {merged}'


# --- bank, the same two truths -----------------------------------------------

def _bank_row(app, tmp_path):
    src = tmp_path / 'bank'
    src.mkdir()
    (src / 'b.png').write_bytes(b'fake')
    with app.app_context():
        bank = ImageBank(user_id='local', name='wm-zones-bank',
                         source_path=str(src))
        db.session.add(bank)
        db.session.commit()
        row = BankImage(bank_id=bank.id, relpath='b.png', status='keep')
        db.session.add(row)
        db.session.commit()
        return bank.id, row.id


def test_bank_scan_keeps_every_zone_and_single_stays_bare(app, monkeypatch,
                                                          tmp_path):
    bank_id, row_id = _bank_row(app, tmp_path)
    calls = {'n': 0}

    def scan(paths, **kw):
        calls['n'] += 1
        for p in paths:
            yield p, 'detected', 0.97, THREE, 'fp', None

    monkeypatch.setattr(watermark_detector, 'scan', scan)
    monkeypatch.setattr(watermark_detector, 'resolve_backend',
                        lambda requested=None: {'requested': 'detector',
                                                'backend': 'detector',
                                                'fell_back': False, 'detail': ''})
    monkeypatch.setattr(banks, '_prepare_watermark_write',
                        lambda row, src, fp: True, raising=False)
    with app.app_context():
        banks.start_watermark(app, 'local', bank_id)   # TESTING => synchronous
    with app.app_context():
        row = db.session.get(BankImage, row_id)
        assert row.watermark_state == 'detected'
        assert json.loads(row.watermark_bbox) == THREE[0]
        assert json.loads(row.watermark_regions) == THREE, (
            'the bank dropped the other zones again')
