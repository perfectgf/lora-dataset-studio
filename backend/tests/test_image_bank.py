"""🗃️ Image bank — inventory, quality scan, duplicate groups, triage statuses,
promotion. The bank references the source folder IN PLACE and must never write
to it; background jobs run inline under TESTING (see bank_jobs.start)."""
import io
import os
import random
import struct
import zlib

import pytest
from PIL import Image, ImageFilter


# --- image factories ---------------------------------------------------------
def _save(path, im):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.lower().endswith(('.jpg', '.jpeg')):
        im.save(path, 'JPEG', quality=92)
    else:
        im.save(path)


def checkerboard(size=128, cell=8):
    im = Image.new('L', (size, size))
    im.putdata([255 if ((x // cell + y // cell) % 2) else 0
                for y in range(size) for x in range(size)])
    return im.convert('RGB')


def photo_like(size=256):
    """Smooth gradient + a bright disc: enough large-scale structure for a
    STABLE dHash across resizes (a checkerboard aliases at 9×8)."""
    im = Image.new('L', (size, size))
    c, r2 = size / 2, (size / 3) ** 2
    im.putdata([min(255, int(150 * x / size + 50 * y / size)
                    + (80 if (x - c) ** 2 + (y - c) ** 2 < r2 else 0))
                for y in range(size) for x in range(size)])
    return im.convert('RGB')


def bokeh_like(size=512, subject=208, cell=2, amp=5, base=120):
    """A bokeh portrait in miniature: an in-focus subject (fine ±``amp``
    texture) covering ~17% of the frame, on a large creamy out-of-focus
    background (smooth gradient). The defaults sit in the exact regime the bug
    report describes — whole-frame Laplacian variance ≈78, i.e. under the
    default sharpness_min of 100, while the subject's own tiles score ≈320.
    ``amp=120`` with a tiny ``subject`` gives the opposite pathology: a single
    blazing block in an otherwise soft frame."""
    im = Image.new('L', (size, size))
    lo, hi = (size - subject) // 2, (size + subject) // 2

    def value(x, y):
        if lo <= x < hi and lo <= y < hi:
            return base + (amp if (x // cell + y // cell) % 2 else -amp)
        return min(255, int(60 + 140 * x / size + 40 * y / size))
    im.putdata([value(x, y) for y in range(size) for x in range(size)])
    return im.convert('RGB')


def noisy(size=128, seed=7):
    rng = random.Random(seed)
    im = Image.new('L', (size, size))
    im.putdata([rng.randrange(256) for _ in range(size * size)])
    return im.convert('RGB')


def flat(size=128, value=128):
    return Image.new('RGB', (size, size), (value, value, value))


def _mkbank(client, tmp_path, files, name='B'):
    """Write ``files`` = {relpath: PIL image or raw bytes} under a source dir,
    create the bank over it, return (bank_id, src_dir)."""
    src = tmp_path / 'src'
    for rel, im in files.items():
        p = src / rel
        if isinstance(im, (bytes, bytearray)):
            os.makedirs(p.parent, exist_ok=True)
            p.write_bytes(im)
        else:
            _save(str(p), im)
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


def _compact_png_header(width, height):
    def chunk(kind, data):
        return (struct.pack('>I', len(data)) + kind + data
                + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff))
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IEND', b''))


# --- quality metrics (pure PIL) ---------------------------------------------
def test_live_bank_pixel_bomb_is_rejected_before_thumbnail_or_scan_decode(
        app, client, tmp_path):
    """A file added after Bank inventory cannot become a late decompression path."""
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import image_bank_service as banks

    bank_id, src = _mkbank(
        client, tmp_path, {'compact-bomb.png': _compact_png_header(9000, 9000)})
    with app.app_context():
        bank = db.session.get(ImageBank, bank_id)
        row = BankImage.query.filter_by(bank_id=bank_id).one()
        assert banks.ensure_thumb(bank, row) is None
        result = banks._scan_one(str(src), tmp_path / 'thumbs', (row.id, row.relpath))
        assert result['quality_state'] == 'unreadable'
        with pytest.raises(ValueError, match='rejects images above'):
            banks._read_safe_bank_source_bytes(
                str(src / row.relpath), label='bank test')
    assert not (tmp_path / 'thumbs' / f'{row.id}.webp').exists()


def test_quality_metrics_discriminate():
    from app.services.image_quality import quality_metrics
    sharp = quality_metrics(checkerboard())
    blurry = quality_metrics(checkerboard().filter(ImageFilter.GaussianBlur(6)))
    gray = quality_metrics(flat())
    grain = quality_metrics(noisy())
    # Sharpness: checkerboard ≫ its blurred copy ≫ flat gray.
    assert sharp['blur_score'] > 100 > blurry['blur_score'] >= gray['blur_score']
    # Noise: per-pixel random ≫ smooth surfaces.
    assert grain['noise_score'] > 15 > gray['noise_score']
    # Uniformity: flat gray is near zero, textured images are not.
    assert gray['uniformity_score'] < 12 < sharp['uniformity_score']


def test_sharpness_reads_the_sharpest_region_not_the_whole_frame():
    """Reported by the community: bokeh portraits — a small razor-sharp subject
    on a large creamy background — were flagged 🌫 blurry, because whole-frame
    Laplacian variance averages the sharp subject away. Per-tile scoring must
    clear the default sharpness_min for the bokeh shot while a genuinely soft
    image (everything blurred) stays below it."""
    from app.services.image_quality import (
        _laplacian_variance, analysis_copy, quality_metrics, _LAPLACIAN,
        _LAPLACIAN_NEG, _LAP_SCALE)
    from app.config import DEFAULTS
    threshold = DEFAULTS['bank']['sharpness_min']

    bokeh = bokeh_like()
    soft = bokeh.filter(ImageFilter.GaussianBlur(6))

    # The regression this fixes: whole-frame variance sinks the bokeh shot.
    g = analysis_copy(bokeh)
    w, h = g.size
    whole = _laplacian_variance(
        g.filter(ImageFilter.Kernel((3, 3), _LAPLACIAN, scale=_LAP_SCALE)),
        g.filter(ImageFilter.Kernel((3, 3), _LAPLACIAN_NEG, scale=_LAP_SCALE)),
        (1, 1, w - 1, h - 1))
    assert whole < threshold, 'fixture no longer reproduces the reported bug'

    assert quality_metrics(bokeh)['blur_score'] > threshold
    assert quality_metrics(soft)['blur_score'] < threshold


def test_sharpness_tiles_ignore_a_single_hot_tile():
    """p90, not max: one blazing block in an otherwise soft frame (a JPEG
    artefact, a burnt highlight, a watermark) must NOT certify the image as
    sharp — and here the whole-frame score would have."""
    from app.services.image_quality import (
        _laplacian_variance, analysis_copy, quality_metrics, _LAPLACIAN,
        _LAPLACIAN_NEG, _LAP_SCALE)
    from app.config import DEFAULTS
    threshold = DEFAULTS['bank']['sharpness_min']
    speck = bokeh_like(subject=48, amp=120)  # <1% of the frame, full contrast
    g = analysis_copy(speck)
    w, h = g.size
    whole = _laplacian_variance(
        g.filter(ImageFilter.Kernel((3, 3), _LAPLACIAN, scale=_LAP_SCALE)),
        g.filter(ImageFilter.Kernel((3, 3), _LAPLACIAN_NEG, scale=_LAP_SCALE)),
        (1, 1, w - 1, h - 1))
    assert whole > threshold, 'whole-frame variance used to call this sharp'
    assert quality_metrics(speck)['blur_score'] < threshold


def test_quality_metrics_match_reference_laplacian():
    """The two-pass histogram trick must equal a direct signed Laplacian
    variance over the interior (PIL leaves the 1-px border unfiltered, so the
    implementation crops it — the reference does the same).
    48 px is below one tile's minimum side, so the grid degenerates to a single
    tile and blur_score is exactly the whole-interior variance."""
    from app.services.image_quality import quality_metrics
    im = photo_like(size=48).convert('L')
    px = list(im.getdata())
    w, h = im.size

    def at(x, y):
        return px[y * w + x]
    lap = [at(x, y - 1) + at(x - 1, y) + at(x + 1, y) + at(x, y + 1) - 4 * at(x, y)
           for y in range(1, h - 1) for x in range(1, w - 1)]
    mean = sum(lap) / len(lap)
    ref = sum((v - mean) ** 2 for v in lap) / len(lap)
    got = quality_metrics(im.convert('RGB'))['blur_score']
    assert got == pytest.approx(ref, rel=0.05, abs=2.0)


# --- inventory ---------------------------------------------------------------
def test_create_bank_walks_recursively(client, tmp_path, app):
    bank_id, src = _mkbank(client, tmp_path, {
        'a.jpg': checkerboard(), 'sub/b.png': flat(), 'sub/deep/c.webp': noisy(),
        'notes.txt': b'not an image',
    })
    with app.app_context():
        from app.models import BankImage
        rels = {r.relpath.replace('\\', '/') for r in
                BankImage.query.filter_by(bank_id=bank_id).all()}
    assert rels == {'a.jpg', 'sub/b.png', 'sub/deep/c.webp'}


def test_create_bank_validates_folder(client, tmp_path):
    r = client.post('/api/bank/create',
                    json={'name': 'X', 'folder': str(tmp_path / 'absent')})
    assert r.status_code == 400
    r = client.post('/api/bank/create', json={'name': '', 'folder': str(tmp_path)})
    assert r.status_code == 400


def test_banks_list(client, tmp_path):
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': flat()})
    data = client.get('/api/banks').get_json()
    assert [b['id'] for b in data['banks']] == [bank_id]
    assert data['banks'][0]['total'] == 1


def test_banks_list_previews_are_capped_stable_and_skip_rejects(client, tmp_path):
    """The card's thumbnail strip: at most 5 ids, in inventory (id) order so the
    strip doesn't reshuffle between reloads, rejected shots left out."""
    files = {f'{i:02d}.jpg': flat(value=10 * i) for i in range(1, 8)}
    bank_id, _src = _mkbank(client, tmp_path, files)

    def previews():
        banks = client.get('/api/banks').get_json()['banks']
        return next(b for b in banks if b['id'] == bank_id)['preview_ids']

    first = previews()
    assert len(first) == 5
    assert first == sorted(first)
    assert previews() == first          # stable across reloads

    # A rejected image drops out of the strip and the next one slides in.
    r = client.post(f'/api/bank/{bank_id}/images/status',
                    json={'ids': [first[0]], 'status': 'reject'})
    assert r.status_code == 200
    after = previews()
    assert first[0] not in after
    assert after[:4] == first[1:]


def test_bank_preview_thumb_served_without_a_scan(client, tmp_path):
    """A freshly created (never scanned) bank must still show thumbnails — the
    thumb route generates them on demand."""
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': photo_like()})
    b = client.get('/api/banks').get_json()['banks'][0]
    assert b['scanned'] == 0 and len(b['preview_ids']) == 1
    r = client.get(f"/api/bank/{bank_id}/thumb/{b['preview_ids'][0]}")
    assert r.status_code == 200
    assert r.mimetype == 'image/webp'


def test_banks_list_preview_empty_for_imageless_bank(client, tmp_path):
    _mkbank(client, tmp_path, {'notes.txt': b'not an image'})
    assert client.get('/api/banks').get_json()['banks'][0]['preview_ids'] == []


def test_banks_list_batches_the_promotable_counts(client, tmp_path, app):
    """?dataset_id= embeds every bank's per-target promotable count in the LIST,
    so the dataset-side bank chooser opens on one request instead of one per
    bank. Same numbers as /bank/<id>/promotable, which stays for single asks."""
    b1, _ = _mkbank(client, tmp_path / 'one', {'a.jpg': flat(), 'b.jpg': flat(80)}, name='B1')
    b2, _ = _mkbank(client, tmp_path / 'two', {'c.jpg': flat(160)}, name='B2')
    with app.app_context():
        from app.services import face_dataset_service as svc
        ds = svc.create_dataset('local', 'Target', 'dstgt').id
    ids1 = [i['id'] for i in client.get(f'/api/bank/{b1}/images').get_json()['images']]
    client.post(f'/api/bank/{b1}/images/status', json={'ids': ids1, 'status': 'keep'})

    rows = {b['id']: b for b in
            client.get(f'/api/banks?dataset_id={ds}').get_json()['banks']}
    assert rows[b1]['promotable'] == 2
    assert rows[b2]['promotable'] == 0          # nothing kept -> explicit zero
    for bank_id in (b1, b2):
        single = client.get(
            f'/api/bank/{bank_id}/promotable?dataset_id={ds}').get_json()
        assert single['count'] == rows[bank_id]['promotable']

    # No dataset_id, or one that doesn't exist: the field is OMITTED rather than
    # zeroed — "unknown" and "nothing to import" must not look the same.
    plain = client.get('/api/banks').get_json()['banks']
    assert all('promotable' not in b for b in plain)
    for bad in (f'{ds + 999}', 'abc', ''):
        got = client.get(f'/api/banks?dataset_id={bad}').get_json()['banks']
        assert all('promotable' not in b for b in got), bad


# --- quality scan ------------------------------------------------------------
def test_scan_scores_flags_and_unreadable(client, tmp_path):
    bank_id, src = _mkbank(client, tmp_path, {
        'sharp.jpg': checkerboard(), 'gray.jpg': flat(),
        'broken.jpg': b'\xff\xd8 definitely not a jpeg',
    })
    r = client.post(f'/api/bank/{bank_id}/scan', json={})
    assert r.status_code == 202
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['counts']['scanned'] == 3
    assert payload['activity']['finished'] is True
    assert payload['flags']['unreadable'] == 1
    # Source folder untouched: exactly the three files we wrote.
    assert sorted(p.name for p in src.rglob('*') if p.is_file()) == \
        ['broken.jpg', 'gray.jpg', 'sharp.jpg']
    imgs = client.get(f'/api/bank/{bank_id}/images').get_json()['images']
    by_name = {i['name']: i for i in imgs}
    assert 'blur' in by_name['gray.jpg']['flags']
    assert 'uniform' in by_name['gray.jpg']['flags']
    assert 'small' in by_name['sharp.jpg']['flags']      # 128 px < min_side 768
    assert 'blur' not in by_name['sharp.jpg']['flags']
    # The unreadable file is auto-rejected (it can never be promoted).
    assert by_name['broken.jpg']['status'] == 'reject'
    assert by_name['broken.jpg']['reject_reason'] == 'unreadable'


def test_flags_follow_threshold_changes_without_rescan(client, tmp_path, app):
    bank_id, _src = _mkbank(client, tmp_path, {'sharp.jpg': checkerboard()})
    client.post(f'/api/bank/{bank_id}/scan', json={})
    by = {i['name']: i for i in
          client.get(f'/api/bank/{bank_id}/images').get_json()['images']}
    assert 'blur' not in by['sharp.jpg']['flags']
    with app.app_context():
        import app.config as cfg
        cfg.save_config({'bank': {'sharpness_min': 10 ** 9}})
    by = {i['name']: i for i in
          client.get(f'/api/bank/{bank_id}/images').get_json()['images']}
    assert 'blur' in by['sharp.jpg']['flags']            # no rescan needed


# --- duplicates --------------------------------------------------------------
def test_duplicate_groups_and_keep_best(client, tmp_path):
    big = photo_like(size=256)
    small = big.resize((96, 96), Image.LANCZOS)          # same content, downscaled
    bank_id, _src = _mkbank(client, tmp_path, {
        'orig.jpg': big, 'copy.jpg': small, 'other.jpg': noisy(),
    })
    client.post(f'/api/bank/{bank_id}/scan', json={})
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['dup'] == {'groups': 1, 'images': 2, 'unresolved': 1}
    groups = client.get(f'/api/bank/{bank_id}/dup-groups').get_json()['groups']
    assert len(groups) == 1
    names = {i['name'] for i in groups[0]['images']}
    assert names == {'orig.jpg', 'copy.jpg'}
    # keep best = the higher-resolution member.
    best = next(i for i in groups[0]['images'] if i['id'] == groups[0]['best_id'])
    assert best['name'] == 'orig.jpg'
    r = client.post(f'/api/bank/{bank_id}/dups/resolve', json={'strategy': 'best'})
    assert r.get_json() == {'ok': True, 'resolved': 1, 'rejected': 1}
    by = {i['name']: i for i in
          client.get(f'/api/bank/{bank_id}/images').get_json()['images']}
    assert by['copy.jpg']['status'] == 'reject'
    assert by['copy.jpg']['reject_reason'] == 'duplicate'
    assert by['orig.jpg']['status'] == 'pending'         # keeper untouched
    assert client.get(f'/api/bank/{bank_id}').get_json()['dup']['unresolved'] == 0


def test_resolve_keep_first_and_manual_pick(client, tmp_path, app):
    im = checkerboard(size=256, cell=16)
    bank_id, _src = _mkbank(client, tmp_path, {
        'a_first.jpg': im, 'b_copy.jpg': im, 'c_copy.jpg': im,
    })
    client.post(f'/api/bank/{bank_id}/scan', json={})
    groups = client.get(f'/api/bank/{bank_id}/dup-groups').get_json()['groups']
    ids = {i['name']: i['id'] for i in groups[0]['images']}
    # Manual pick: keep c explicitly.
    r = client.post(f'/api/bank/{bank_id}/dups/resolve',
                    json={'keep_ids': [ids['c_copy.jpg']]})
    assert r.get_json()['rejected'] == 2
    by = {i['name']: i for i in
          client.get(f'/api/bank/{bank_id}/images').get_json()['images']}
    assert by['c_copy.jpg']['status'] == 'pending'
    assert by['a_first.jpg']['status'] == 'reject'
    # Reset then keep-first: lowest id (import order) wins.
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': list(ids.values()), 'status': 'pending'})
    r = client.post(f'/api/bank/{bank_id}/dups/resolve', json={'strategy': 'first'})
    assert r.get_json()['rejected'] == 2
    by = {i['name']: i for i in
          client.get(f'/api/bank/{bank_id}/images').get_json()['images']}
    assert by['a_first.jpg']['status'] == 'pending'
    assert by['b_copy.jpg']['status'] == 'reject'


def test_auto_resolve_never_flips_a_manual_keep(client, tmp_path, app):
    """The AUTOMATIC resolver (pipeline auto-reject) must never un-keep a manual
    pick: resolve_dups_keep_best keeps respect_existing_keep=True."""
    im = checkerboard(size=256, cell=16)
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': im, 'b.jpg': im})
    client.post(f'/api/bank/{bank_id}/scan', json={})
    groups = client.get(f'/api/bank/{bank_id}/dup-groups').get_json()['groups']
    ids = {i['name']: i['id'] for i in groups[0]['images']}
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [ids['b.jpg']], 'status': 'keep'})
    with app.app_context():
        from app.services import image_bank_service as banks
        assert banks.resolve_dups_keep_best('local', bank_id) == 0   # keep protected
    by = {i['name']: i for i in
          client.get(f'/api/bank/{bank_id}/images').get_json()['images']}
    assert by['b.jpg']['status'] == 'keep'               # manual keep survives auto


def test_explicit_resolve_rejects_kept_losers(client, tmp_path, app):
    """An EXPLICIT resolve (user clicks Keep best / Keep first / Resolve ALL on a
    same-shot group) must collapse the group to ONE — even when every member is
    already 'keep' (the usual same-shot case). Before the fix the guard skipped
    'keep' members, so the losers stayed kept and the toast read '0 rejected'."""
    im = checkerboard(size=256, cell=16)
    bank_id, _src = _mkbank(client, tmp_path, {
        'a.jpg': im, 'b.jpg': im, 'c.jpg': im,
    })
    client.post(f'/api/bank/{bank_id}/scan', json={})
    groups = client.get(f'/api/bank/{bank_id}/dup-groups').get_json()['groups']
    ids = {i['name']: i['id'] for i in groups[0]['images']}
    # ALL three members kept — the shape that produced 'Resolved N — 0 rejected'.
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': list(ids.values()), 'status': 'keep'})

    # AUTO path first: with every member kept it rejects nobody (unchanged).
    with app.app_context():
        from app.services import image_bank_service as banks
        assert banks.resolve_dups_keep_best('local', bank_id) == 0

    # EXPLICIT path: keep first → a.jpg elected, b + c fall to reject.
    r = client.post(f'/api/bank/{bank_id}/dups/resolve', json={'strategy': 'first'})
    assert r.get_json() == {'ok': True, 'resolved': 1, 'rejected': 2}
    by = {i['name']: i for i in
          client.get(f'/api/bank/{bank_id}/images').get_json()['images']}
    assert by['a.jpg']['status'] == 'keep'               # elected keeper untouched
    assert by['b.jpg']['status'] == 'reject'
    assert by['c.jpg']['status'] == 'reject'
    assert by['b.jpg']['reject_reason'] == 'duplicate'


def test_explicit_semantic_resolve_rejects_kept_losers(client, tmp_path, app):
    """Same per-group collapse for stage-2 semantic groups: an explicit resolve
    over an all-kept semantic_dup_group rejects the losers (reason semantic_dup),
    while the elected keeper survives — set up at the service layer so no CLIP
    extra is needed."""
    im = checkerboard(size=256, cell=16)
    bank_id, _src = _mkbank(client, tmp_path, {
        'a.jpg': im, 'b.jpg': im, 'c.jpg': im,
    })
    client.post(f'/api/bank/{bank_id}/scan', json={})
    with app.app_context():
        from app.models import BankImage
        from app.extensions import db
        rows = (BankImage.query.filter_by(bank_id=bank_id)
                .order_by(BankImage.id.asc()).all())
        for r in rows:
            r.semantic_dup_group, r.status = 1, 'keep'
        db.session.commit()
        keeper_id = rows[0].id
    # Explicit manual pick keeps the first row; the two kept losers get rejected.
    r = client.post(f'/api/bank/{bank_id}/semantic-dups/resolve',
                    json={'keep_ids': [keeper_id]})
    assert r.get_json()['rejected'] == 2
    with app.app_context():
        from app.models import BankImage
        rows = BankImage.query.filter_by(bank_id=bank_id).all()
        kept = [x for x in rows if x.status == 'keep']
        rej = [x for x in rows if x.status == 'reject']
        assert len(kept) == 1 and kept[0].id == keeper_id
        assert len(rej) == 2 and all(x.reject_reason == 'semantic_dup' for x in rej)


# --- flag application + statuses --------------------------------------------
def test_apply_flags_rejects_pending_only(client, tmp_path):
    bank_id, _src = _mkbank(client, tmp_path, {
        'gray.jpg': flat(), 'gray2.jpg': flat(value=90), 'sharp.jpg': checkerboard(),
    })
    client.post(f'/api/bank/{bank_id}/scan', json={})
    imgs = client.get(f'/api/bank/{bank_id}/images').get_json()['images']
    keep_id = next(i['id'] for i in imgs if i['name'] == 'gray.jpg')
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [keep_id], 'status': 'keep'})
    r = client.post(f'/api/bank/{bank_id}/apply-flags', json={'flags': ['uniform']})
    assert r.get_json()['rejected'] == {'uniform': 1}    # gray2 only — keep survives
    by = {i['name']: i for i in
          client.get(f'/api/bank/{bank_id}/images').get_json()['images']}
    assert by['gray.jpg']['status'] == 'keep'
    assert by['gray2.jpg']['status'] == 'reject'
    assert by['gray2.jpg']['reject_reason'] == 'uniform'
    assert by['sharp.jpg']['status'] == 'pending'


# --- listing filters + pagination -------------------------------------------
def test_images_filters_and_pagination(client, tmp_path):
    files = {f'n{i:02d}.jpg': checkerboard(size=128 + 2 * i) for i in range(5)}
    files['gray.jpg'] = flat()
    bank_id, _src = _mkbank(client, tmp_path, files)
    client.post(f'/api/bank/{bank_id}/scan', json={})
    page = client.get(f'/api/bank/{bank_id}/images?limit=2&offset=2').get_json()
    assert page['total'] == 6 and len(page['images']) == 2
    flagged = client.get(f'/api/bank/{bank_id}/images?flag=uniform').get_json()
    assert [i['name'] for i in flagged['images']] == ['gray.jpg']
    status = client.get(f'/api/bank/{bank_id}/images?status=reject').get_json()
    assert status['total'] == 0


def test_images_sort_by_resolution(client, tmp_path, app):
    """Sort=res_desc/res_asc orders by MEGAPIXELS (width×height), not width — a
    900×900 (810k px) outranks a wider 1200×300 (360k). Unscanned rows (width or
    height NULL) sink to the end in BOTH directions, and the sort composes with
    filters + pagination."""
    files = {f'{n}.jpg': checkerboard(size=64) for n in ('a', 'b', 'c', 'd', 'e')}
    bank_id, _src = _mkbank(client, tmp_path, files)
    # Set dimensions by hand (bypass the scan) so the areas are unambiguous.
    dims = {'a': (900, 900), 'b': (1200, 300), 'c': (1000, 1000),
            'd': (400, 400), 'e': (None, None)}   # e = unscanned → NULL area
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        for row in BankImage.query.filter_by(bank_id=bank_id).all():
            w, h = dims[row.relpath.split('.')[0]]
            row.width, row.height = w, h
        db.session.commit()

    def names(sort, **qs):
        params = '&'.join(f'{k}={v}' for k, v in {'sort': sort, **qs}.items())
        got = client.get(f'/api/bank/{bank_id}/images?{params}').get_json()
        return [i['name'].split('.')[0] for i in got['images']]

    # Descending megapixels: c(1M) > a(810k) > b(360k) > d(160k); NULL 'e' last.
    assert names('res_desc') == ['c', 'a', 'b', 'd', 'e']
    # Ascending: d < b < a < c; NULL 'e' still last (never first).
    assert names('res_asc') == ['d', 'b', 'a', 'c', 'e']
    # Composes with pagination — the top-2 of the descending order.
    page = client.get(f'/api/bank/{bank_id}/images?sort=res_desc&limit=2').get_json()
    assert [i['name'].split('.')[0] for i in page['images']] == ['c', 'a']
    assert page['total'] == 5
    # Composes with a status filter: reject the two largest, keep the sort.
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [i['id'] for i in
                              client.get(f'/api/bank/{bank_id}/images?sort=res_desc&limit=2')
                              .get_json()['images']], 'status': 'reject'})
    assert names('res_desc', status='pending') == ['b', 'd', 'e']
    # An unknown sort value is ignored (falls back to the default id order).
    assert names('bogus') == ['a', 'b', 'c', 'd', 'e']


def test_images_sort_by_score_and_sharpness(client, tmp_path, app):
    """Requested by nofaceman (Discord): the bank already MEASURES aesthetics and
    sharpness — the grid must be able to ORDER on them, not only filter. Same
    contract as the resolution sort: unscored rows sink to the END in BOTH
    directions (a NULL-first order would bury the very images the sort is for),
    the sort composes with every filter, it survives pagination, and the ids
    endpoint "Select all in filter" walks (same URL, bigger pages) sees the same
    order — so ▶ Review opens on what the user is looking at."""
    files = {f'{n}.jpg': checkerboard(size=64) for n in ('a', 'b', 'c', 'd', 'e')}
    bank_id, _src = _mkbank(client, tmp_path, files)
    # Hand-set the raw scores (bypass the passes, which need the ML extras).
    # 'e' is the never-analysed row: NULL everywhere.
    scores = {'a': (6.5, 120.0), 'b': (8.2, 40.0), 'c': (3.1, 900.0),
              'd': (7.0, 300.0), 'e': (None, None)}
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        for row in BankImage.query.filter_by(bank_id=bank_id).all():
            aes, blur = scores[row.relpath.split('.')[0]]
            row.aesthetic_score, row.blur_score = aes, blur
        db.session.commit()

    def names(sort, **qs):
        params = '&'.join(f'{k}={v}' for k, v in {'sort': sort, **qs}.items())
        got = client.get(f'/api/bank/{bank_id}/images?{params}').get_json()
        return [i['name'].split('.')[0] for i in got['images']]

    # Best first, then the unscored 'e' — never at the top.
    assert names('aesthetic_desc') == ['b', 'd', 'a', 'c', 'e']
    # Worst first (find the rejects) — 'e' STILL last, not first.
    assert names('aesthetic_asc') == ['c', 'a', 'd', 'b', 'e']
    # Sharpness = Laplacian variance: sharpest first / blurriest first.
    assert names('sharp_desc') == ['c', 'd', 'a', 'b', 'e']
    assert names('sharp_asc') == ['b', 'a', 'd', 'c', 'e']

    # Composes with pagination — the page boundary does not reshuffle the order,
    # and the whole set is still there once the pages are concatenated.
    first = client.get(f'/api/bank/{bank_id}/images?sort=aesthetic_desc&limit=2').get_json()
    second = client.get(f'/api/bank/{bank_id}/images'
                        '?sort=aesthetic_desc&limit=2&offset=2').get_json()
    assert [i['name'].split('.')[0] for i in first['images']] == ['b', 'd']
    assert [i['name'].split('.')[0] for i in second['images']] == ['a', 'c']
    assert first['total'] == second['total'] == 5

    # Composes with a filter: nothing lost, nothing invented. Reject 'c' and the
    # remaining set is exactly the other four, in the sorted order.
    cid = next(i['id'] for i in
               client.get(f'/api/bank/{bank_id}/images').get_json()['images']
               if i['name'] == 'c.jpg')
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [cid], 'status': 'reject'})
    assert names('aesthetic_desc', status='pending') == ['b', 'd', 'a', 'e']
    assert names('sharp_asc', status='pending') == ['b', 'a', 'd', 'e']

    # "Select all in filter" / ▶ Review walk the SAME endpoint with limit=500 —
    # the ids come back in the active sort AND under the active filter, so the
    # selection is exactly what the user is looking at, in that order.
    walked = client.get(f'/api/bank/{bank_id}/images'
                        '?sort=aesthetic_desc&status=pending&limit=500').get_json()
    assert [i['name'].split('.')[0] for i in walked['images']] == ['b', 'd', 'a', 'e']
    assert walked['total'] == 4


def test_resolution_buckets_counts_and_filter(client, tmp_path, app):
    """Resolution tiers bucket on MEGAPIXELS with HALF-OPEN [lo, hi) bounds. The
    exact boundaries matter: 1000×1000 (1.00 MP) and 1024×1024 (1.05 MP) both land
    in '1–2 MP' (lower-inclusive), 512×512 (0.26 MP) in '0.25–1 MP' (never the junk
    tier), 2000×2000 (4.00 MP) in '> 4 MP'. The payload reports one count per tier
    (unscanned excluded), and res_bucket narrows a page, composing with status +
    sort + pagination."""
    names = ('thumb', 'small', 'std', 'std2', 'big', 'huge', 'unscanned')
    files = {f'{n}.jpg': checkerboard(size=64) for n in names}
    bank_id, _src = _mkbank(client, tmp_path, files)
    # Hand-set dimensions on the exact tier boundaries (bypass the scan).
    dims = {'thumb': (400, 400),      # 0.16 MP  → res_lt_025
            'small': (512, 512),      # 0.26 MP  → res_025_1 (NOT junk)
            'std':   (1000, 1000),    # 1.00 MP  → res_1_2 (lower-inclusive)
            'std2':  (1024, 1024),    # 1.05 MP  → res_1_2
            'big':   (1920, 1080),    # 2.07 MP  → res_2_4
            'huge':  (2000, 2000),    # 4.00 MP  → res_gt_4 (lower-inclusive)
            'unscanned': (None, None)}  # NULL area → excluded from every tier
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        for row in BankImage.query.filter_by(bank_id=bank_id).all():
            w, h = dims[row.relpath.split('.')[0]]
            row.width, row.height = w, h
        db.session.commit()

    # One count per tier, every id present, unscanned counted nowhere.
    buckets = client.get(f'/api/bank/{bank_id}').get_json()['res_buckets']
    assert buckets == {'res_lt_025': 1, 'res_025_1': 1, 'res_1_2': 2,
                       'res_2_4': 1, 'res_gt_4': 1}

    def names_in(bucket, **qs):
        params = '&'.join(f'{k}={v}' for k, v in {'res_bucket': bucket, **qs}.items())
        got = client.get(f'/api/bank/{bank_id}/images?{params}').get_json()
        return sorted(i['name'].split('.')[0] for i in got['images'])

    # Each tier returns exactly its members — the boundary cases sit where claimed.
    assert names_in('res_lt_025') == ['thumb']
    assert names_in('res_025_1') == ['small']
    assert names_in('res_1_2') == ['std', 'std2']     # 1.00 and 1.05 MP together
    assert names_in('res_2_4') == ['big']
    assert names_in('res_gt_4') == ['huge']
    # Composes with the resolution sort: the two '1–2 MP' rows, largest first.
    got = client.get(f'/api/bank/{bank_id}/images'
                     '?res_bucket=res_1_2&sort=res_desc').get_json()
    assert [i['name'].split('.')[0] for i in got['images']] == ['std2', 'std']
    # Composes with a status filter + pagination: reject 'std', page the rest.
    std_id = next(i['id'] for i in
                  client.get(f'/api/bank/{bank_id}/images?res_bucket=res_1_2')
                  .get_json()['images'] if i['name'].startswith('std.'))
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [std_id], 'status': 'reject'})
    assert names_in('res_1_2', status='pending') == ['std2']
    # An unknown tier id is ignored (no filter → the whole scanned+unscanned set).
    assert names_in('bogus') == sorted(names)


def test_no_face_filter_only_matches_no_face_state(client, tmp_path, app):
    """The "No face" chip must show ONLY images where NO face was detected. The
    other non-scorable states (low_det / too_small / extreme_pose) DID find a
    face and must NOT appear — that regression surfaced photos with visible
    faces under a "No face" label. Unscanned rows (face_state NULL) stay out."""
    files = {f'n{i}.jpg': checkerboard(size=256) for i in range(6)}
    bank_id, _src = _mkbank(client, tmp_path, files)
    client.post(f'/api/bank/{bank_id}/scan', json={})
    states = ['no_face', 'low_det', 'too_small', 'extreme_pose', 'scorable', None]
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        rows = (BankImage.query.filter_by(bank_id=bank_id)
                .order_by(BankImage.id.asc()).all())
        for row, state in zip(rows, states):
            row.face_state = state
        db.session.commit()
    got = client.get(f'/api/bank/{bank_id}/images?flag=no_face').get_json()
    assert [i['face_state'] for i in got['images']] == ['no_face']


def test_face_device_resolves_cpu_without_cuda(app, monkeypatch):
    """Default device is CPU and the GPU window is never opened unless the face
    interpreter truly exposes CUDA. 'cuda' requested without CUDA degrades to
    CPU (so the parent won't serialize a CPU pass behind training)."""
    from app.services import image_bank_service as svc
    from app import capabilities
    with app.app_context():
        monkeypatch.setattr(capabilities, 'face_gpu_available', lambda: False)
        assert svc._resolve_face_device() == ('cpu', False)          # auto, no CUDA
        monkeypatch.setattr(svc.cfg, 'get',
                            lambda k, d=None: 'cuda' if k == 'face_scoring.device' else d)
        assert svc._resolve_face_device() == ('cpu', False)          # cuda asked, none
        monkeypatch.setattr(capabilities, 'face_gpu_available', lambda: True)
        assert svc._resolve_face_device() == ('cuda', True)          # cuda asked + CUDA


def test_face_embed_infer_cpu_providers_no_device():
    """The infer script must default to CPU-only providers — a bare CUDA-first
    list would grab the GPU the instant onnxruntime-gpu is present, outside the
    parent's GPU-exclusive window."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / 'infer' / 'face_embed_infer.py').read_text(encoding='utf-8')
    # No unconditional CUDA-first provider list survives.
    assert "['CUDAExecutionProvider', 'CPUExecutionProvider']}" not in src
    assert "device = str(req.get('device') or 'cpu').lower()" in src


# --- promotion ---------------------------------------------------------------
def test_promote_keeps_into_dataset(client, tmp_path, app):
    bank_id, _src = _mkbank(client, tmp_path, {
        'a.jpg': checkerboard(size=256, cell=16), 'b.jpg': noisy(size=256),
        'c.jpg': flat(size=256),
    })
    client.post(f'/api/bank/{bank_id}/scan', json={})
    imgs = client.get(f'/api/bank/{bank_id}/images').get_json()['images']
    keep_ids = [i['id'] for i in imgs if i['name'] in ('a.jpg', 'b.jpg')]
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': keep_ids, 'status': 'keep'})
    with app.app_context():
        from app.services import face_dataset_service as svc
        ds = svc.create_dataset('local', 'From bank', 'bnk')
        ds_id = ds.id
    r = client.post(f'/api/bank/{bank_id}/promote', json={'dataset_id': ds_id})
    assert r.status_code == 202
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['activity']['finished'] is True
    assert payload['activity']['error'] is None
    assert payload['counts']['promoted'] == 2
    with app.app_context():
        from app.models import FaceDatasetImage
        rows = FaceDatasetImage.query.filter_by(dataset_id=ds_id).all()
        assert len(rows) == 2
        assert all(r2.source == 'import' and r2.status == 'keep' for r2 in rows)
    # Second promotion with nothing left to promote → 400.
    r = client.post(f'/api/bank/{bank_id}/promote', json={'dataset_id': ds_id})
    assert r.status_code == 400


def test_promote_same_image_to_a_second_dataset(client, tmp_path, app):
    """A kept image already promoted to dataset A must still be promotable to a
    DIFFERENT dataset B. The scalar promoted_dataset_id only remembers the LAST
    target, so the eligible set is per-target (promoted-elsewhere ≠ promoted-here),
    not a global 'promoted anywhere' lock. Regression for the Bank 'nothing to
    promote' toast when the modal still counted the kept images."""
    bank_id, _src = _mkbank(client, tmp_path, {
        'a.jpg': checkerboard(size=256, cell=16), 'b.jpg': noisy(size=256),
    })
    client.post(f'/api/bank/{bank_id}/scan', json={})
    imgs = client.get(f'/api/bank/{bank_id}/images').get_json()['images']
    keep_ids = [i['id'] for i in imgs]
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': keep_ids, 'status': 'keep'})
    with app.app_context():
        from app.services import face_dataset_service as svc
        ds_a = svc.create_dataset('local', 'Dataset A', 'dsa').id
        ds_b = svc.create_dataset('local', 'Dataset B', 'dsb').id

    # Promote everything kept to A.
    r = client.post(f'/api/bank/{bank_id}/promote', json={'dataset_id': ds_a})
    assert r.status_code == 202

    # The honest promotable count must be per-target: 0 left for A, but all
    # kept images are still promotable to B.
    ca = client.get(f'/api/bank/{bank_id}/promotable?dataset_id={ds_a}').get_json()
    cb = client.get(f'/api/bank/{bank_id}/promotable?dataset_id={ds_b}').get_json()
    assert ca['count'] == 0
    assert cb['count'] == 2

    # Promoting to a DIFFERENT dataset succeeds (this raised 400 before the fix).
    r = client.post(f'/api/bank/{bank_id}/promote', json={'dataset_id': ds_b})
    assert r.status_code == 202
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['activity']['finished'] is True
    assert payload['activity']['error'] is None
    with app.app_context():
        from app.models import FaceDatasetImage
        assert FaceDatasetImage.query.filter_by(dataset_id=ds_b).count() == 2

    # Re-promoting to B (all now sit on B) is a no-op → 400, same as same-target.
    r = client.post(f'/api/bank/{bank_id}/promote', json={'dataset_id': ds_b})
    assert r.status_code == 400


def test_promote_requires_dataset(client, tmp_path):
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': checkerboard()})
    r = client.post(f'/api/bank/{bank_id}/promote', json={})
    assert r.status_code == 400
    r = client.post(f'/api/bank/{bank_id}/promote', json={'dataset_id': 999})
    assert r.status_code == 400


# --- faces gate --------------------------------------------------------------
def test_faces_unavailable_is_503(client, tmp_path, monkeypatch):
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': checkerboard()})
    from app.services import face_similarity
    monkeypatch.setattr(face_similarity, 'is_available', lambda: False)
    r = client.post(f'/api/bank/{bank_id}/faces', json={})
    assert r.status_code == 503
    assert 'face scoring' in r.get_json()['error']


def test_cluster_assignment_orders_by_size():
    np = pytest.importorskip('numpy')
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        'face_embed_infer',
        pathlib.Path(__file__).resolve().parents[1] / 'infer' / 'face_embed_infer.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    v1 = np.array([1.0] + [0.0] * 511, dtype='float32')
    v2 = np.array([0.0, 1.0] + [0.0] * 510, dtype='float32')
    cache = {
        'p1': ('scorable', 0.9, 0.5, v1), 'p2': ('scorable', 0.9, 0.5, v1),
        'p3': ('scorable', 0.9, 0.5, v1), 'p4': ('scorable', 0.9, 0.5, v2),
        'p5': ('scorable', 0.9, 0.5, v2), 'p6': ('no_face', 0.0, 0.0, v2 * 0),
    }
    out = mod._cluster(list(cache), cache, 0.45)
    assert out['p1'] == out['p2'] == out['p3'] == 1       # biggest cluster = id 1
    assert out['p4'] == out['p5'] == 2
    assert 'p6' not in out                                # no face → no cluster


# --- serving + deletion ------------------------------------------------------
def test_thumb_and_file_endpoints(client, tmp_path):
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': checkerboard(size=600)})
    img_id = client.get(f'/api/bank/{bank_id}/images').get_json()['images'][0]['id']
    r = client.get(f'/api/bank/{bank_id}/thumb/{img_id}')   # lazy — no scan yet
    assert r.status_code == 200
    with Image.open(io.BytesIO(r.data)) as t:
        assert max(t.size) <= 320
    r = client.get(f'/api/bank/{bank_id}/file/{img_id}')
    assert r.status_code == 200
    with Image.open(io.BytesIO(r.data)) as f:
        assert f.size == (600, 600)
    assert client.get(f'/api/bank/{bank_id}/thumb/99999').status_code == 404


def test_delete_bank_never_touches_source(client, tmp_path, app):
    bank_id, src = _mkbank(client, tmp_path, {'a.jpg': checkerboard()})
    client.post(f'/api/bank/{bank_id}/scan', json={})
    assert client.delete(f'/api/bank/{bank_id}').status_code == 200
    assert (src / 'a.jpg').is_file()                     # source intact
    assert client.get(f'/api/bank/{bank_id}').status_code == 404
    with app.app_context():
        from app.models import BankImage
        assert BankImage.query.filter_by(bank_id=bank_id).count() == 0
        import app.config as cfg
        assert not (cfg.banks_root() / str(bank_id)).exists()


def test_busy_bank_answers_409(client, tmp_path, app):
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': checkerboard()})
    from app.services import bank_jobs
    bank_jobs._jobs[bank_id] = {'kind': 'scan', 'done': 0, 'total': 1,
                                'error': None, 'cancelled': False,
                                'finished': False, 'detail': None,
                                'started_at': 0, '_touched': __import__('time').time(),
                                '_cancel_hook': None}
    r = client.post(f'/api/bank/{bank_id}/scan', json={})
    assert r.status_code == 409
    # cancel works, then the job is startable again
    assert client.post(f'/api/bank/{bank_id}/cancel', json={}).get_json()['ok'] is True
