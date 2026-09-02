"""🚩 Bank watermark cleaning — the two manual levels (crop, then inpaint).

The invariant that governs the whole feature: a bank is a READ-ONLY view over a
folder that belongs to the user, so cleaning a watermark may never touch the
source file. Cleaned pixels live in the bank's own working directory, and every
reader that DISPLAYS or COPIES an image must prefer that blob — otherwise the
passes run for nothing and the dataset still gets the watermarked original.

Both of those are asserted here: the source is compared byte-for-byte (and by
mtime) after a full clean, and the three known readers are pinned to the single
resolver, so rebinding one of them back to abs_image_path fails a test instead
of silently un-shipping the feature.
"""
import ast
import io
import json
import os

import pytest
from PIL import Image


def _photo(size=1000, value=90):
    """A plain, readable image — the tests care about pixels moving, not content."""
    im = Image.new('RGB', (size, size), (value, value, value))
    for y in range(0, size, 40):        # a bit of structure so a crop is visible
        for x in range(size):
            im.putpixel((x, y), (250, 250, 250))
    return im


def _mkbank(client, tmp_path, files, name='WM'):
    src = tmp_path / 'src'
    for rel, im in files.items():
        p = src / rel
        os.makedirs(p.parent, exist_ok=True)
        im.save(str(p), 'JPEG', quality=92)
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


def _flag(app, bank_id, bbox, *, index=0, state='detected'):
    """Put a row in the state the detection pass leaves behind. Returns its id."""
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks
    with app.app_context():
        bank = db.session.get(ImageBank, bank_id)
        rows = (BankImage.query.filter_by(bank_id=bank_id)
                .order_by(BankImage.id.asc()).all())
        row = rows[index]
        row.watermark_state = state
        row.watermark_bbox = json.dumps(bbox) if bbox is not None else None
        row.watermark_fingerprint = transfer.content_fingerprint_path(
            banks.abs_image_path(bank, row))
        db.session.commit()
        return row.id


def _row(app, image_id):
    from app.models import BankImage
    with app.app_context():
        row = db_get(BankImage, image_id)
        return (row.watermark_state, row.watermark_clean_method)


def db_get(model, pk):
    from app.extensions import db
    return db.session.get(model, pk)


def _fingerprint(path):
    """Bytes + mtime — what "the source was not touched" actually means."""
    return open(path, 'rb').read(), os.stat(path).st_mtime_ns


def _seed_effective_lanes(app, bank_id, image_id):
    """Give a row representatives from every effective-byte lane."""
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    with app.app_context():
        bank = db.session.get(ImageBank, bank_id)
        row = db.session.get(BankImage, image_id)
        row.analysis_fingerprint = transfer.content_fingerprint_path(
            banks.analysis_image_path(bank, row))
        row.quality_state = 'ok'
        row.blur_score = 123.0
        row.aesthetic_score = 7.5
        row.nsfw_score = 0.1
        row.face_state = 'scorable'
        row.face_det = 0.9
        row.face_yaw = 3.0
        row.framing = 'bust'
        row.medium = 'photo'
        row.dup_group = 9
        row.semantic_dup_group = 10
        row.style_cluster = 11
        # User-owned person assertions are identity metadata, not a computed
        # result, and must survive every pixel-generation invalidation.
        row.face_cluster = 12
        row.face_cluster_origin = 'asserted'
        db.session.commit()


def _assert_raw_generation_invalidated(app, bank_id, image_id, source):
    from app.extensions import db
    from app.models import BankImage
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    with app.app_context():
        row = db.session.get(BankImage, image_id)
        assert row.watermark_fingerprint == transfer.content_fingerprint_path(source)
        assert row.watermark_state is None
        assert row.watermark_bbox is None
        assert row.watermark_regions is None
        assert row.watermark_clean_method is None
        assert row.analysis_fingerprint is None
        assert row.width is None and row.height is None
        assert row.face_cluster == 12
        assert row.face_cluster_origin == 'asserted'
        for field in ('quality_state', 'blur_score', 'aesthetic_score',
                      'nsfw_score', 'face_state', 'face_det', 'face_yaw',
                      'framing', 'medium', 'dup_group', 'semantic_dup_group',
                      'style_cluster'):
            assert getattr(row, field) is None, field
        assert not banks.clean_image_path(bank_id, image_id).exists()


# A mark inside the top band of a 1000×1000 image: croppable (the crop keeps
# 950 px, above the 768 min side). Level 1's lane.
TOP_MARK = [0.1, 0.01, 0.4, 0.05]
# Small, off-centre: never croppable, LaMa's lane. Level 2's.
OFFCENTER_MARK = [0.30, 0.30, 0.36, 0.36]
# Straddles the centre: only Klein can repaint it, otherwise manual review.
CENTER_MARK = [0.40, 0.40, 0.60, 0.60]


# --- level 1: crop -----------------------------------------------------------
def test_crop_level_cleans_without_touching_the_source(client, tmp_path, app):
    """THE invariant: the crop lands in the bank's working copy, and the user's
    own file is byte-identical (and untouched by mtime) afterwards."""
    bank_id, src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, TOP_MARK)
    before = _fingerprint(src / 'a.jpg')

    r = client.post(f'/api/bank/{bank_id}/watermark/crop')
    assert r.status_code == 202, r.get_json()

    assert _fingerprint(src / 'a.jpg') == before      # source never written to
    assert _row(app, image_id) == ('cleaned', 'crop')
    with app.app_context():
        from app.services import image_bank_service as banks
        blob = banks.clean_image_path(bank_id, image_id)
    assert blob.is_file()
    with Image.open(blob) as im:
        # The band up to the mark's inner edge is gone: 1000 → 950 px tall.
        assert im.height == 950 and im.width == 1000


def test_crop_discards_result_when_raw_source_changes_during_engine_work(
        client, tmp_path, app, monkeypatch):
    """A crop result belongs to the raw bytes measured before the crop began."""
    from app.services import image_bank_service as banks

    bank_id, src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, TOP_MARK)
    _seed_effective_lanes(app, bank_id, image_id)
    source = src / 'a.jpg'
    real_cut = banks._cut_clean_copy

    def cut_then_replace(bank_id, row, src_path, box):
        out = real_cut(bank_id, row, src_path, box)
        _photo(value=210).save(source, 'JPEG', quality=92)
        return out

    monkeypatch.setattr(banks, '_cut_clean_copy', cut_then_replace)

    response = client.post(f'/api/bank/{bank_id}/watermark/crop')
    assert response.status_code == 202, response.get_json()
    _assert_raw_generation_invalidated(app, bank_id, image_id, source)


def test_crop_level_encodes_once_and_only_the_kept_band(client, tmp_path, app, monkeypatch):
    """The auto-crop level paid the same two lossless WebP encodes per image as
    the manual ✂ crop did — one of them of the WHOLE frame, staged only to be
    reopened, cut and thrown away — and it is a batch pass, so it paid it once
    per image. Pin the shape that makes it fast: one encode, of the kept band."""
    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _photo()})     # 1000×1000
    image_id = _flag(app, bank_id, TOP_MARK)                         # keeps 1000×950

    # Counted at Pillow's own save, UNDER image_encoding.save_edit: a staging
    # copy re-introduced through any write path shows up here.
    written = []
    real_save = Image.Image.save

    def counting_save(self, fp, format=None, **params):
        written.append((self.size, (format or '').upper()))
        return real_save(self, fp, format, **params)
    monkeypatch.setattr(Image.Image, 'save', counting_save)

    assert client.post(f'/api/bank/{bank_id}/watermark/crop').status_code == 202
    assert _row(app, image_id) == ('cleaned', 'crop')
    assert written == [((1000, 950), 'WEBP')], (
        'the level must encode the kept band once — a full-frame (1000, 1000) '
        'entry here is the staging copy coming back')


def _run_crop_job(app, bank_id):
    """Run the auto-crop level synchronously with a job dict shaped like
    bank_jobs.start()'s, so the line the user would read is in job['detail']
    and a crash lands in job['error'] instead of propagating."""
    from app.services import image_bank_service as banks
    job = {'kind': 'watermark_crop', 'done': 0, 'total': 0, 'error': None,
           'cancelled': False, 'finished': False, 'detail': None,
           'started_at': 0.0, '_touched': 0.0, '_cancel_hook': None, 'pipeline': None}
    with app.app_context():
        try:
            banks._watermark_crop_job(bank_id)(job)
        except Exception as e:  # noqa: BLE001 — mirrors the real runner
            job['error'] = f'{type(e).__name__}: {e}'
    return job


def test_crop_level_counts_an_empty_box_as_unreadable_and_writes_nothing(
        client, tmp_path, app, monkeypatch):
    """A box with no pixel left once clamped is the result the in-place crop
    used to answer with False: the image is not marked cleaned, no blob is
    written, and the pass SAYS it in the line the user reads — 'unreadable',
    not a silent skip."""
    from app.services import face_dataset_service as fds
    from app.services import image_bank_service as banks

    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, TOP_MARK)
    monkeypatch.setattr(fds, '_route_watermark',
                        lambda bbox, w, h, allow_crop=True: ('crop', (0, 0, 0, 10)))

    job = _run_crop_job(app, bank_id)
    assert job['error'] is None, job['error']
    assert '1 unreadable' in job['detail'], job['detail']
    assert _row(app, image_id) == ('detected', None)
    with app.app_context():
        clean = banks.clean_image_path(bank_id, image_id)
    assert not clean.exists()
    assert not clean.parent.exists() or not any(clean.parent.iterdir())   # no .part- either


def test_crop_level_survives_an_unreadable_source_and_says_so(
        client, tmp_path, app, monkeypatch):
    """A source the cut cannot read is one image lost, not the whole pass: the
    level moves on, leaves that image flagged with no blob, and counts it in
    the line the user reads."""
    from app.services import image_bank_service as banks

    bank_id, _ = _mkbank(client, tmp_path,
                         {'a.jpg': _photo(), 'b.jpg': _photo(value=120)})
    first = _flag(app, bank_id, TOP_MARK, index=0)
    second = _flag(app, bank_id, TOP_MARK, index=1)
    real_cut = banks._cut_clean_copy
    calls = {'n': 0}

    def cut_or_choke(bank_id_, row, src_path, box):
        calls['n'] += 1
        if calls['n'] == 1:
            raise OSError('cannot read the source')
        return real_cut(bank_id_, row, src_path, box)
    monkeypatch.setattr(banks, '_cut_clean_copy', cut_or_choke)

    job = _run_crop_job(app, bank_id)
    assert job['error'] is None, job['error']
    assert '1 cropped' in job['detail'] and '1 unreadable' in job['detail'], job['detail']
    assert _row(app, first) == ('detected', None)
    assert _row(app, second) == ('cleaned', 'crop')
    with app.app_context():
        assert not banks.clean_image_path(bank_id, first).exists()
        assert banks.clean_image_path(bank_id, second).is_file()


def test_bank_crop_uses_exif_visual_coordinates_and_keeps_camera_master(client, tmp_path, app):
    """Bank clean works on a neutral upright derivative, never the source JPEG.

    Vision's top-band bbox is expressed in visual space. A 1600×1000 raw JPEG
    with EXIF=6 displays at 1000×1600, so a 10% top crop must yield 1000×1440.
    """
    bank_id, src = _mkbank(client, tmp_path, {'camera.jpg': _photo()})
    camera = src / 'camera.jpg'
    image = Image.new('RGB', (1600, 1000), (70, 130, 220))
    exif = image.getexif()
    exif[274] = 6
    exif[271] = 'Private Camera'
    image.save(camera, 'JPEG', quality=95, subsampling=0, exif=exif)
    image_id = _flag(app, bank_id, [0.0, 0.0, 1.0, 0.10])
    before = _fingerprint(camera)

    assert client.post(f'/api/bank/{bank_id}/watermark/crop').status_code == 202

    assert _fingerprint(camera) == before
    with app.app_context():
        from app.services import image_bank_service as banks
        blob = banks.clean_image_path(bank_id, image_id)
    payload = blob.read_bytes()
    assert b'Private Camera' not in payload and b'Exif' not in payload
    with Image.open(blob) as cleaned:
        cleaned.load()
        assert cleaned.format == 'WEBP' and cleaned.size == (1000, 1440)
        assert not cleaned.getexif()


def test_bank_levels_recompute_exif_visual_dimensions_not_raw_scan_dimensions(
        client, tmp_path, app):
    """The level badge must offer the same crop route the worker will take."""
    bank_id, src = _mkbank(client, tmp_path, {'camera.jpg': _photo()})
    camera = src / 'camera.jpg'
    image = Image.new('RGB', (1600, 900), (70, 130, 220))
    exif = image.getexif()
    exif[274] = 6
    image.save(camera, 'JPEG', quality=95, subsampling=0, exif=exif)
    _flag(app, bank_id, [0.0, 0.0, 1.0, 0.20])

    # Raw height 900 would retain only 720 px and fail the 768px crop guard;
    # visual height 1600 retains 1200px and is correctly offered as croppable.
    levels = client.get(f'/api/bank/{bank_id}/watermark/levels').get_json()
    assert levels['croppable'] == 1


def test_crop_level_leaves_everything_else_flagged(client, tmp_path, app):
    """Level 1 only touches what the router calls croppable — an off-centre and
    an on-subject mark stay flagged, which is exactly level 2's pool."""
    bank_id, _src = _mkbank(client, tmp_path,
                            {'a.jpg': _photo(), 'b.jpg': _photo(value=120)})
    off = _flag(app, bank_id, OFFCENTER_MARK, index=0)
    mid = _flag(app, bank_id, CENTER_MARK, index=1)

    assert client.post(f'/api/bank/{bank_id}/watermark/crop').status_code == 202

    assert _row(app, off) == ('detected', None)
    assert _row(app, mid) == ('detected', None)
    with app.app_context():
        from app.services import image_bank_service as banks
        assert not banks.clean_image_path(bank_id, off).exists()


def test_crop_level_refuses_when_there_is_nothing_flagged(client, tmp_path):
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    r = client.post(f'/api/bank/{bank_id}/watermark/crop')
    assert r.status_code == 400
    assert 'watermark scan' in r.get_json()['error']


# --- the readers: promote / thumbnail / file --------------------------------
def test_promote_hands_the_cleaned_blob_to_the_dataset(client, tmp_path, app,
                                                       monkeypatch):
    """The point of the whole feature: after a clean, the dataset must receive
    the CLEANED pixels. Captured at the import_images boundary."""
    bank_id, _src = _mkbank(client, tmp_path,
                            {'a.jpg': _photo(), 'b.jpg': _photo(value=120)})
    cleaned_id = _flag(app, bank_id, TOP_MARK, index=0)
    assert client.post(f'/api/bank/{bank_id}/watermark/crop').status_code == 202
    assert _row(app, cleaned_id)[0] == 'cleaned'

    seen = []
    from app.services import image_bank_service as banks

    def fake_import(user_id, dataset_id, blobs, **kw):
        seen.extend(blobs)
        return list(range(len(blobs))), 0
    monkeypatch.setattr(banks, 'import_images', fake_import)

    with app.app_context():
        from app.services import face_dataset_service as svc
        dataset_id = svc.create_dataset('local', 'From bank', 'bnk').id
    ids = [i['id'] for i in
           client.get(f'/api/bank/{bank_id}/images').get_json()['images']]
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': ids, 'status': 'keep'})
    r = client.post(f'/api/bank/{bank_id}/promote', json={'dataset_id': dataset_id})
    assert r.status_code == 202, r.get_json()

    heights = sorted(Image.open(io.BytesIO(b)).height for b in seen)
    # The cleaned image arrives cropped (950), the untouched one at full height.
    assert heights == [950, 1000]


def test_file_route_serves_cleaned_and_original_side_by_side(client, tmp_path, app):
    """?original=1 is the before/after pair — no third lightbox needed."""
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, TOP_MARK)
    assert client.post(f'/api/bank/{bank_id}/watermark/crop').status_code == 202

    cleaned = client.get(f'/api/bank/{bank_id}/file/{image_id}')
    original = client.get(f'/api/bank/{bank_id}/file/{image_id}?original=1')
    assert Image.open(io.BytesIO(cleaned.data)).height == 950
    assert Image.open(io.BytesIO(original.data)).height == 1000


def test_thumbnail_follows_the_cleaned_version(client, tmp_path, app):
    """A stale thumbnail would show the watermark long after it was removed."""
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, TOP_MARK)
    first = client.get(f'/api/bank/{bank_id}/thumb/{image_id}')   # cache it
    assert Image.open(io.BytesIO(first.data)).height == Image.open(
        io.BytesIO(first.data)).width                            # square source

    assert client.post(f'/api/bank/{bank_id}/watermark/crop').status_code == 202
    after = client.get(f'/api/bank/{bank_id}/thumb/{image_id}')
    with Image.open(io.BytesIO(after.data)) as im:
        assert im.height < im.width          # regenerated from the cropped blob


def test_display_and_copy_readers_are_pinned_to_the_resolver():
    """CONTRACT — the cleaned blob only reaches the user if the readers ask for
    it. Each piece of this feature can be green while the JOIN is broken, so the
    three known readers are pinned here: promotion and the thumbnail must go
    through the strict analysis_image_path and must NOT call abs_image_path
    directly, and
    the file route must offer the resolver (its ?original=1 lane is the one
    place abs_image_path stays legitimate). Rebinding any of them back to the
    raw source path fails THIS test."""
    import app.routes.bank as bank_routes
    from app.services import image_bank_service as banks

    def calls(module, func_name, inner=None):
        tree = ast.parse(open(module.__file__, encoding='utf-8').read())
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == func_name)
        if inner:      # the job closure inside a start_* factory
            node = next(n for n in ast.walk(node)
                        if isinstance(n, ast.FunctionDef) and n.name == inner)
        return {n.func.id for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)} | \
               {n.func.attr for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}

    promote = calls(banks, '_promote_job', inner='run')
    assert 'analysis_image_path' in promote
    assert 'abs_image_path' not in promote

    thumb = calls(banks, 'ensure_thumb')
    assert 'resolved_image_path' in thumb
    assert 'abs_image_path' not in thumb

    assert 'resolved_image_path' in calls(bank_routes, 'bank_file')


# --- level 2: inpaint --------------------------------------------------------
def _fake_lama(monkeypatch, *, available=True, ok=True):
    """Stand in for the LaMa subprocess: paints the file it is handed (in the
    bank's working copy — never the source) so the effect is observable."""
    from app.services import watermark_lama
    monkeypatch.setattr(watermark_lama, 'is_available', lambda: available)
    monkeypatch.setattr(watermark_lama, 'resolve_device', lambda: 'cpu')
    calls = []

    def fake_batch(jobs, *, device, timeout=900):
        out = {}
        for job in jobs:
            calls.append(job['image_path'])
            if ok:
                with Image.open(job['image_path']) as im:
                    im.convert('RGB').resize((im.width, im.height // 2)).save(
                        job['image_path'], 'WEBP', quality=92)
                out[job['image_path']] = (True, None)
            else:
                out[job['image_path']] = (False, {'kind': 'failed',
                                                  'detail': 'inpainter blew up'})
        return out
    monkeypatch.setattr(watermark_lama, 'inpaint_batch', fake_batch)
    return calls


def test_inpaint_level_repaints_what_the_crop_left(client, tmp_path, app,
                                                   monkeypatch):
    bank_id, src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, OFFCENTER_MARK)
    before = _fingerprint(src / 'a.jpg')
    painted = _fake_lama(monkeypatch)

    r = client.post(f'/api/bank/{bank_id}/watermark/inpaint', json={'method': 'lama'})
    assert r.status_code == 202, r.get_json()

    assert _fingerprint(src / 'a.jpg') == before          # source still untouched
    assert _row(app, image_id) == ('cleaned', 'lama')
    # The engine was handed OUR working copy, not the user's file.
    assert len(painted) == 1
    assert str(src) not in painted[0]


@pytest.mark.parametrize('method', ['lama', 'klein'])
def test_inpaint_discards_result_when_raw_source_changes_during_engine_work(
        client, tmp_path, app, monkeypatch, method):
    """Both slow engines must revalidate the pristine source after returning."""
    from app.services import watermark_klein, watermark_lama

    bank_id, src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, OFFCENTER_MARK)
    _seed_effective_lanes(app, bank_id, image_id)
    source = src / 'a.jpg'

    def replace_source():
        _photo(value=220).save(source, 'JPEG', quality=92)

    if method == 'lama':
        monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
        monkeypatch.setattr(watermark_lama, 'resolve_device', lambda: 'cpu')

        def batch(jobs, *, device, timeout=900):
            replace_source()
            return {job['image_path']: (True, None) for job in jobs}

        monkeypatch.setattr(watermark_lama, 'inpaint_batch', batch)
    else:
        monkeypatch.setattr(watermark_klein, 'is_available', lambda: True)

        def klein(_user_id, _path, _boxes, **_kwargs):
            replace_source()
            return True, None

        monkeypatch.setattr(
            watermark_klein, 'inpaint_watermark_klein', klein)

    response = client.post(
        f'/api/bank/{bank_id}/watermark/inpaint', json={'method': method})
    assert response.status_code == 202, response.get_json()
    _assert_raw_generation_invalidated(app, bank_id, image_id, source)


@pytest.mark.parametrize('method, clean_method', [('lama', 'lama'), ('klein', 'klein')])
def test_bank_inpaint_stages_exif_upright_metadata_free_copy_for_each_engine(
        client, tmp_path, app, monkeypatch, method, clean_method):
    """Both external engines receive the same visual, neutral Bank work file."""
    from app.services import watermark_klein, watermark_lama

    bank_id, src = _mkbank(client, tmp_path, {'camera.jpg': _photo()})
    camera = src / 'camera.jpg'
    image = Image.new('RGB', (1600, 1000), (40, 150, 70))
    exif = image.getexif()
    exif[274] = 6
    exif[271] = 'Private Camera'
    image.save(camera, 'JPEG', quality=95, subsampling=0, exif=exif)
    # Manual regions make both engines actionable without depending on the
    # automatic router's review branch.
    image_id = _flag(app, bank_id, OFFCENTER_MARK)
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        row = db.session.get(BankImage, image_id)
        row.watermark_regions = json.dumps([[0.2, 0.2, 0.3, 0.3]])
        db.session.commit()
    before = _fingerprint(camera)
    seen = []

    def _assert_staged(path):
        seen.append(path)
        assert str(camera) != str(path)
        assert _fingerprint(camera) == before
        payload = open(path, 'rb').read()
        assert b'Private Camera' not in payload and b'Exif' not in payload
        with Image.open(path) as staged:
            staged.load()
            assert staged.format == 'WEBP' and staged.size == (1000, 1600)
            assert not staged.getexif()

    if method == 'lama':
        monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
        monkeypatch.setattr(watermark_lama, 'resolve_device', lambda: 'cpu')

        def _batch(jobs, *, device, timeout=900):
            for job in jobs:
                _assert_staged(job['image_path'])
            return {job['image_path']: (True, None) for job in jobs}

        monkeypatch.setattr(watermark_lama, 'inpaint_batch', _batch)
    else:
        monkeypatch.setattr(watermark_klein, 'is_available', lambda: True)

        def _klein(_user_id, path, _boxes, **_kwargs):
            _assert_staged(path)
            return True, None

        monkeypatch.setattr(watermark_klein, 'inpaint_watermark_klein', _klein)

    response = client.post(f'/api/bank/{bank_id}/watermark/inpaint', json={'method': method})
    assert response.status_code == 202, response.get_json()
    assert len(seen) == 1 and _fingerprint(camera) == before
    assert _row(app, image_id) == ('cleaned', clean_method)


def test_inpaint_level_leaves_on_subject_marks_for_klein(client, tmp_path, app,
                                                         monkeypatch):
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, CENTER_MARK)
    _fake_lama(monkeypatch)

    client.post(f'/api/bank/{bank_id}/watermark/inpaint', json={'method': 'lama'})

    assert _row(app, image_id) == ('detected', None)      # still flagged, honestly


def test_inpaint_level_says_what_to_install_instead_of_failing_silently(
        client, tmp_path, app, monkeypatch):
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, OFFCENTER_MARK)
    _fake_lama(monkeypatch, available=False)

    r = client.post(f'/api/bank/{bank_id}/watermark/inpaint', json={'method': 'lama'})
    assert r.status_code == 503
    assert 'not installed' in r.get_json()['error']
    assert _row(app, image_id) == ('detected', None)       # nothing half-done


def test_failed_inpaint_keeps_no_half_written_blob(client, tmp_path, app,
                                                   monkeypatch):
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, OFFCENTER_MARK)
    _fake_lama(monkeypatch, ok=False)

    client.post(f'/api/bank/{bank_id}/watermark/inpaint', json={'method': 'lama'})

    assert _row(app, image_id) == ('detected', None)
    with app.app_context():
        from app.services import image_bank_service as banks
        assert not banks.clean_image_path(bank_id, image_id).exists()


def test_inpaint_level_rejects_an_unknown_engine(client, tmp_path, app):
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    _flag(app, bank_id, OFFCENTER_MARK)
    r = client.post(f'/api/bank/{bank_id}/watermark/inpaint', json={'method': 'nope'})
    assert r.status_code == 400


# --- undo / dismiss ----------------------------------------------------------
def test_undo_drops_the_cleaned_version_and_re_flags(client, tmp_path, app):
    """Undo is only ever deleting our OWN blob — that's what makes both levels
    risk-free, and it must put the image back in the cleanable pool."""
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, TOP_MARK)
    client.post(f'/api/bank/{bank_id}/watermark/crop')

    r = client.post(f'/api/bank/{bank_id}/watermark/undo', json={})
    assert r.status_code == 200 and r.get_json()['restored'] == 1

    assert _row(app, image_id) == ('detected', None)
    with app.app_context():
        from app.services import image_bank_service as banks
        assert not banks.clean_image_path(bank_id, image_id).exists()
    # Full size and thumbnail are back to the original pixels.
    served = client.get(f'/api/bank/{bank_id}/file/{image_id}')
    assert Image.open(io.BytesIO(served.data)).height == 1000


def test_dismissed_images_leave_both_levels(client, tmp_path, app):
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, TOP_MARK)

    r = client.post(f'/api/bank/{bank_id}/watermark/dismiss',
                    json={'image_ids': [image_id]})
    assert r.status_code == 200 and r.get_json()['dismissed'] == 1

    assert client.post(f'/api/bank/{bank_id}/watermark/crop').status_code == 400
    levels = client.get(f'/api/bank/{bank_id}/watermark/levels').get_json()
    assert levels['dismissed'] == 1 and levels['flagged'] == 0


@pytest.mark.parametrize('action', ['mask', 'dismiss'])
def test_manual_watermark_actions_refuse_a_replaced_raw_generation(
        client, tmp_path, app, action):
    bank_id, src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, OFFCENTER_MARK)
    _seed_effective_lanes(app, bank_id, image_id)
    source = src / 'a.jpg'
    _photo(value=230).save(source, 'JPEG', quality=92)

    if action == 'mask':
        response = client.put(
            f'/api/bank/{bank_id}/image/{image_id}/watermark-regions',
            json={'regions': [[0.2, 0.2, 0.3, 0.3]]})
        assert response.status_code == 409, response.get_json()
        assert 'source image changed' in response.get_json()['error']
    else:
        response = client.post(
            f'/api/bank/{bank_id}/watermark/dismiss',
            json={'image_ids': [image_id]})
        assert response.status_code == 200, response.get_json()
        assert response.get_json()['dismissed'] == 0

    _assert_raw_generation_invalidated(app, bank_id, image_id, source)


def test_undo_refuses_a_replaced_raw_generation_and_drops_stale_clean_blob(
        client, tmp_path, app):
    bank_id, src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, TOP_MARK)
    assert client.post(
        f'/api/bank/{bank_id}/watermark/crop').status_code == 202
    _seed_effective_lanes(app, bank_id, image_id)
    source = src / 'a.jpg'
    _photo(value=240).save(source, 'JPEG', quality=92)

    response = client.post(
        f'/api/bank/{bank_id}/watermark/undo',
        json={'image_ids': [image_id]})
    assert response.status_code == 200, response.get_json()
    assert response.get_json()['restored'] == 0
    _assert_raw_generation_invalidated(app, bank_id, image_id, source)


# --- retrofit: rows flagged before the bbox was persisted --------------------
def test_legacy_flag_without_a_bbox_is_readopted_by_a_plain_scan(client, tmp_path,
                                                                 app):
    """Banks scanned by an earlier build carry 'detected' with no box. They must
    be re-picked by an ordinary scan (not only by an explicit rescan) and be
    counted out loud meanwhile — never silently invisible to both levels."""
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    _flag(app, bank_id, None)

    levels = client.get(f'/api/bank/{bank_id}/watermark/levels').get_json()
    assert levels['needs_rescan'] == 1 and levels['flagged'] == 0

    with app.app_context():
        from app.services import image_bank_service as banks
        rows = banks._watermark_scan_query(bank_id, rescan=False).all()
    assert len(rows) == 1        # a plain "Find watermarks" adopts it


def test_rescan_never_re_examines_a_dismissed_image(client, tmp_path, app):
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    _flag(app, bank_id, TOP_MARK, state='dismissed')
    with app.app_context():
        from app.services import image_bank_service as banks
        assert banks._watermark_scan_query(bank_id, rescan=True).count() == 0


# --- the per-level tallies the UI shows -------------------------------------
def test_levels_report_progress_of_each_stage(client, tmp_path, app, monkeypatch):
    bank_id, _src = _mkbank(client, tmp_path, {
        'a.jpg': _photo(), 'b.jpg': _photo(value=120), 'c.jpg': _photo(value=160)})
    _flag(app, bank_id, TOP_MARK, index=0)
    _flag(app, bank_id, OFFCENTER_MARK, index=1)
    _flag(app, bank_id, CENTER_MARK, index=2)

    levels = client.get(f'/api/bank/{bank_id}/watermark/levels').get_json()
    assert (levels['flagged'], levels['croppable'], levels['inpaintable']) == (3, 1, 2)

    client.post(f'/api/bank/{bank_id}/watermark/crop')
    levels = client.get(f'/api/bank/{bank_id}/watermark/levels').get_json()
    assert (levels['cropped'], levels['flagged'], levels['croppable']) == (1, 2, 0)

    _fake_lama(monkeypatch)
    client.post(f'/api/bank/{bank_id}/watermark/inpaint', json={'method': 'lama'})
    levels = client.get(f'/api/bank/{bank_id}/watermark/levels').get_json()
    # The on-subject mark is still flagged: LaMa doesn't repaint the subject.
    assert (levels['cropped'], levels['inpainted'], levels['flagged']) == (1, 1, 1)


def test_levels_404_on_an_unknown_bank(client):
    assert client.get('/api/bank/9999/watermark/levels').status_code == 404
