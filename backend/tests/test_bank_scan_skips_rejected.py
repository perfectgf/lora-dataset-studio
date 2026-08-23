"""🗃️ Image bank — the quality scan leaves rejected images alone.

Every other pass (faces, score, framing, watermark, captions) already filtered
`status != 'reject'`; the quality scan did not. On a real 30 000-image bank with
two thirds rejected, a rescan spent two thirds of its time decoding shots the
user had already thrown away.

The subtlety these tests pin down: the scan is ALSO what rejects unreadable
files, so skipping rejected rows must not stop a FIRST scan from doing that.
"""
from unittest.mock import patch

from PIL import Image


def _bank(app, tmp_path, files):
    """files = {name: bytes|'image'} — 'image' writes a real decodable JPEG."""
    from app.services import image_bank_service as banks
    src = tmp_path / 'src'
    src.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        if content == 'image':
            Image.new('RGB', (900, 900), (40, 90, 160)).save(str(src / name))
        else:
            (src / name).write_bytes(content)
    bank, _added = banks.create_bank('local', 'Dump', str(src))
    return bank.id


def _run_scan(app, bank_id, rescan=False):
    """Start the scan and run its body inline; returns the announced total."""
    from app.services import image_bank_service as banks
    seen = {}

    def fake_start(_app, _bid, kind, fn, total=0):
        seen['total'] = total
        fn(object())
        return None

    with patch.object(banks.bank_jobs, 'start', fake_start), \
         patch.object(banks.bank_jobs, 'cancelled', lambda job: False), \
         patch.object(banks.bank_jobs, 'bump', lambda job, n=1: None), \
         patch.object(banks.bank_jobs, 'progress', lambda job, **kw: None):
        banks.start_scan(app, 'local', bank_id, rescan=rescan)
    return seen['total']


def test_a_rescan_skips_rejected_images_and_says_so_in_its_total(app, tmp_path):
    from app.extensions import db
    from app.models import BankImage
    from app.services import image_bank_service as banks

    with app.app_context():
        bank_id = _bank(app, tmp_path, {'good.jpg': 'image', 'bad.jpg': 'image'})
        assert _run_scan(app, bank_id) == 2          # first pass: both are pending

        rows = {r.relpath: r for r in BankImage.query.filter_by(bank_id=bank_id)}
        banks.set_status('local', bank_id, [rows['bad.jpg'].id], 'reject')
        rows['good.jpg'].blur_score = None           # so we can see it re-scanned
        rows['bad.jpg'].blur_score = None
        db.session.commit()

        # The total is what the progress bar promises — it must match the work.
        assert _run_scan(app, bank_id, rescan=True) == 1

        rows = {r.relpath: r for r in BankImage.query.filter_by(bank_id=bank_id)}
        assert rows['good.jpg'].blur_score is not None, 'the kept image was rescanned'
        assert rows['bad.jpg'].blur_score is None, 'the rejected image was left alone'


def test_a_first_scan_still_reaches_an_image_rejected_before_it(app, tmp_path):
    """Rejecting by hand before ever scanning is the one case the filter drops.
    It is deliberate — no CPU for a discarded image — and reversible."""
    from app.models import BankImage
    from app.services import image_bank_service as banks

    with app.app_context():
        bank_id = _bank(app, tmp_path, {'a.jpg': 'image', 'b.jpg': 'image'})
        rows = {r.relpath: r for r in BankImage.query.filter_by(bank_id=bank_id)}
        banks.set_status('local', bank_id, [rows['b.jpg'].id], 'reject')

        assert _run_scan(app, bank_id) == 1
        rows = {r.relpath: r for r in BankImage.query.filter_by(bank_id=bank_id)}
        assert rows['a.jpg'].quality_state == 'ok'
        assert rows['b.jpg'].quality_state is None

        # Un-reject it and it is back in the pool, with no rescan needed.
        banks.set_status('local', bank_id, [rows['b.jpg'].id], 'pending')
        assert _run_scan(app, bank_id) == 1
        rows = {r.relpath: r for r in BankImage.query.filter_by(bank_id=bank_id)}
        assert rows['b.jpg'].quality_state == 'ok'


def test_the_scan_still_auto_rejects_an_unreadable_file(app, tmp_path):
    """NON-REGRESSION: the filter must not disarm the pass's own verdict. A
    pending row that turns out unreadable is still rejected by the first scan."""
    from app.models import BankImage

    with app.app_context():
        bank_id = _bank(app, tmp_path, {'ok.jpg': 'image',
                                        'broken.jpg': b'not an image at all'})
        assert _run_scan(app, bank_id) == 2

        rows = {r.relpath: r for r in BankImage.query.filter_by(bank_id=bank_id)}
        assert rows['broken.jpg'].quality_state == 'unreadable'
        assert rows['broken.jpg'].status == 'reject'
        assert rows['broken.jpg'].reject_reason == 'unreadable'
        assert rows['ok.jpg'].status == 'pending'


def test_a_manual_reject_is_never_flipped_by_the_scan(app, tmp_path):
    from app.models import BankImage
    from app.services import image_bank_service as banks

    with app.app_context():
        bank_id = _bank(app, tmp_path, {'a.jpg': 'image'})
        rows = BankImage.query.filter_by(bank_id=bank_id).all()
        banks.set_status('local', bank_id, [rows[0].id], 'reject')
        _run_scan(app, bank_id, rescan=True)
        row = BankImage.query.filter_by(bank_id=bank_id).first()
        assert row.status == 'reject' and row.reject_reason != 'unreadable'
