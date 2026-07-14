import io, zipfile
from PIL import Image


def _png(color=(255, 0, 0)):
    buf = io.BytesIO(); Image.new('RGB', (64, 64), color).save(buf, 'PNG')
    return buf.getvalue()


def test_create_and_payload(app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Lola', 'lola')
        p = svc.dataset_payload(LOCAL_USER, ds.id)
        # NB: the brief's snippet checked `p['comp']`, but dataset_payload's actual
        # key (SRC-identical) is 'composition' -- 'comp' does not exist and would
        # KeyError. Corrected here; see task-8-report.md.
        assert p['name'] == 'Lola' and p['composition'] == {'face': 0, 'bust': 0, 'body': 0, 'back': 0}


def test_api_fanout_creates_pending_rows(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    from app.services.face_variations import select_preset
    calls = []
    monkeypatch.setattr('app.services.face_dataset_service.threading.Thread',
                        lambda target, args=(), daemon=True: type('T', (), {'start': lambda s: calls.append(args)})())
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'A', 'a')
        # give it a reference so _all_ref_bytes works
        import os
        os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
        open(os.path.join(svc._dataset_dir(ds.id), 'ref.webp'), 'wb').write(_png())
        ds.ref_filename = 'ref.webp'
        svc.db.session.commit()
        svc.generate_variations_nanobanana(app, LOCAL_USER, ds.id,
                                           select_preset('zimage_12')[:2], 1, engine='chatgpt')
        rows = FaceDatasetImage.query.filter_by(dataset_id=ds.id).all()
        assert len(rows) == 2 and all(r.status == 'pending' and r.klein_model == 'chatgpt' for r in rows)
        assert calls  # background batch was dispatched


def test_export_zip_layout(app):
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    import os
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Zoe', 'zoe')
        d = svc._dataset_dir(ds.id); os.makedirs(d, exist_ok=True)
        open(os.path.join(d, 'ref.webp'), 'wb').write(_png()); ds.ref_filename = 'ref.webp'
        open(os.path.join(d, 'img1.webp'), 'wb').write(_png((0, 255, 0)))
        svc.db.session.add(FaceDatasetImage(dataset_id=ds.id, filename='img1.webp',
                                            status='keep', framing='face', caption='a smile'))
        svc.db.session.commit()
        z = zipfile.ZipFile(io.BytesIO(svc.build_export_zip(LOCAL_USER, ds.id)))
        names = z.namelist()
        assert any(n.endswith('_000_ref.png') for n in names)
        txt = [n for n in names if n.endswith('_001.txt')][0]
        assert z.read(txt).decode('utf-8').startswith('zoe, ')


def test_status_validation(app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'B', 'b')
        try:
            svc.set_image_status(LOCAL_USER, 99999, 'nonsense'); raised = False
        except Exception:
            raised = True
        assert raised


def test_import_images_normalizes_and_persists(app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'C', 'c')
        ids, failed = svc.import_images(LOCAL_USER, ds.id, [_png()], crop=False)
        assert len(ids) == 1 and failed == 0
        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        assert len(payload['images']) == 1
        assert payload['images'][0]['status'] == 'keep'


def _seed_images(svc, ds_id, n=3, status='pending'):
    """N committed image rows with real files, returns their ids."""
    import os
    from app.models import FaceDatasetImage
    d = svc._dataset_dir(ds_id); os.makedirs(d, exist_ok=True)
    ids = []
    for i in range(n):
        fn = f'img{i}.webp'
        open(os.path.join(d, fn), 'wb').write(_png((i * 40, 0, 0)))
        img = FaceDatasetImage(dataset_id=ds_id, filename=fn, status=status, framing='face')
        svc.db.session.add(img); svc.db.session.flush(); ids.append(img.id)
    svc.db.session.commit()
    return ids


def test_batch_keep_and_clear_caption(app):
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Bk', 'bk')
        ids = _seed_images(svc, ds.id)
        assert svc.batch_image_action(LOCAL_USER, ds.id, ids, 'keep') == 3
        rows = FaceDatasetImage.query.filter(FaceDatasetImage.id.in_(ids)).all()
        assert all(r.status == 'keep' for r in rows)
        rows[0].caption = 'a caption'; svc.db.session.commit()
        assert svc.batch_image_action(LOCAL_USER, ds.id, [ids[0]], 'clear_caption') == 1
        assert svc.db.session.get(FaceDatasetImage, ids[0]).caption is None


def test_batch_delete_removes_rows_and_files(app):
    import os
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Bd', 'bd')
        ids = _seed_images(svc, ds.id)
        assert svc.batch_image_action(LOCAL_USER, ds.id, ids, 'delete') == 3
        assert FaceDatasetImage.query.filter_by(dataset_id=ds.id).count() == 0
        assert not any(f.startswith('img') for f in os.listdir(svc._dataset_dir(ds.id)))


def test_batch_skips_foreign_and_failed(app):
    """Ids from ANOTHER dataset are silently skipped (stale selection can't cross
    datasets), and a 'failed' tile is never resurrected into keep."""
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds1 = svc.create_dataset(LOCAL_USER, 'B1', 'b1')
        ds2 = svc.create_dataset(LOCAL_USER, 'B2', 'b2')
        own = _seed_images(svc, ds1.id, n=1)
        foreign = _seed_images(svc, ds2.id, n=1)
        failed = _seed_images(svc, ds1.id, n=1, status='failed')
        n = svc.batch_image_action(LOCAL_USER, ds1.id, own + foreign + failed, 'keep')
        assert n == 1   # own only; failed skipped, foreign filtered out
        assert svc.db.session.get(FaceDatasetImage, foreign[0]).status == 'pending'
        assert svc.db.session.get(FaceDatasetImage, failed[0]).status == 'failed'


def _seed_captioned(svc, ds_id, captions):
    from app.models import FaceDatasetImage
    ids = []
    for i, cap in enumerate(captions):
        img = FaceDatasetImage(dataset_id=ds_id, filename=f'c{i}.webp',
                               status='keep', framing='face', caption=cap)
        svc.db.session.add(img); svc.db.session.flush(); ids.append(img.id)
    svc.db.session.commit()
    return ids


def test_replace_captions_text_mode(app):
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Rc', 'rc')
        ids = _seed_captioned(svc, ds.id, ['a woman in a red dress', 'a red car', 'no match'])
        n = svc.replace_in_captions(LOCAL_USER, ds.id, 'red', 'blue', mode='text')
        assert n == 2
        caps = [svc.db.session.get(FaceDatasetImage, i).caption for i in ids]
        assert caps == ['a woman in a blue dress', 'a blue car', 'no match']


def test_replace_captions_tag_mode_removes_cleanly(app):
    """Tag removal must not leave dangling commas, matches the WHOLE tag only
    (no substring bleed into 'blue eyeshadow'), and dedupes the result."""
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Rt', 'rt')
        ids = _seed_captioned(svc, ds.id, [
            '1girl, Blue Eyes, smile, blue eyeshadow',
            'blue eyes, standing',
            'sitting, smile'])
        n = svc.replace_in_captions(LOCAL_USER, ds.id, 'blue eyes', '', mode='tag')
        assert n == 2
        caps = [svc.db.session.get(FaceDatasetImage, i).caption for i in ids]
        assert caps == ['1girl, smile, blue eyeshadow', 'standing', 'sitting, smile']
        # replace variant + dedup: smile -> grin while a grin already exists
        svc.replace_in_captions(LOCAL_USER, ds.id, 'sitting', 'smile', mode='tag')
        assert svc.db.session.get(FaceDatasetImage, ids[2]).caption == 'smile'


def test_replace_captions_ignores_non_kept_and_validates(app):
    import pytest
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Rv', 'rv')
        img = FaceDatasetImage(dataset_id=ds.id, filename='r.webp',
                               status='reject', caption='a red car')
        svc.db.session.add(img); svc.db.session.commit()
        assert svc.replace_in_captions(LOCAL_USER, ds.id, 'red', 'blue') == 0
        assert svc.db.session.get(FaceDatasetImage, img.id).caption == 'a red car'
        with pytest.raises(ValueError):
            svc.replace_in_captions(LOCAL_USER, ds.id, '', 'x')
        with pytest.raises(ValueError):
            svc.replace_in_captions(LOCAL_USER, ds.id, 'a', 'b', mode='regex')


# --- Non-square manual crop ------------------------------------------------------

def test_crop_image_preserves_box_aspect(app):
    """A 2:1 crop box must yield a 2:1 file (1024x512), not a distorted square;
    a square box keeps the historical 1024x1024."""
    import os
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Cr', 'cr')
        d = svc._dataset_dir(ds.id); os.makedirs(d, exist_ok=True)
        buf = io.BytesIO(); Image.new('RGB', (1600, 1200), (90, 30, 30)).save(buf, 'PNG')
        open(os.path.join(d, 'w.webp'), 'wb').write(buf.getvalue())
        img = FaceDatasetImage(dataset_id=ds.id, filename='w.webp', status='keep')
        svc.db.session.add(img); svc.db.session.commit()
        assert svc.crop_image(LOCAL_USER, img.id, 0, 0, 1000, 500) is True
        with Image.open(os.path.join(d, 'w.webp')) as im:
            assert im.size == (1024, 512)
        # square box -> historical square output
        assert svc.crop_image(LOCAL_USER, img.id, 0, 0, 400, 400) is True
        with Image.open(os.path.join(d, 'w.webp')) as im:
            assert im.size == (1024, 1024)


# --- Full backup / restore -----------------------------------------------------

def test_backup_roundtrip_restores_everything(app):
    import os
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Bak', 'bak', train_type='sdxl')
        d = svc._dataset_dir(ds.id); os.makedirs(d, exist_ok=True)
        open(os.path.join(d, 'ref.webp'), 'wb').write(_png())
        ds.ref_filename = 'ref.webp'
        ds.best_settings = '{"strength": 0.8}'
        open(os.path.join(d, 'a.webp'), 'wb').write(_png((0, 255, 0)))
        svc.db.session.add(FaceDatasetImage(dataset_id=ds.id, filename='a.webp', status='keep',
                                            framing='bust', caption='a green coat',
                                            face_score=0.61, face_state='scorable'))
        svc.db.session.commit()
        data = svc.build_backup_zip(LOCAL_USER, ds.id)
        restored = svc.import_backup_zip(LOCAL_USER, data)
        assert restored.id != ds.id
        assert restored.name == 'Bak' and restored.trigger_word == 'bak'
        assert restored.train_type == 'sdxl' and restored.best_settings == '{"strength": 0.8}'
        assert restored.ref_filename == 'ref.webp'
        assert os.path.isfile(os.path.join(svc._dataset_dir(restored.id), 'ref.webp'))
        rows = FaceDatasetImage.query.filter_by(dataset_id=restored.id).all()
        assert len(rows) == 1
        r = rows[0]
        assert (r.filename, r.status, r.framing, r.caption) == ('a.webp', 'keep', 'bust', 'a green coat')
        assert r.face_score == 0.61 and r.face_state == 'scorable'
        assert os.path.isfile(os.path.join(svc._dataset_dir(restored.id), 'a.webp'))


def test_backup_import_rejects_garbage_and_traversal(app):
    import io as _io
    import zipfile as _zip
    import pytest
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        with pytest.raises(ValueError, match='not a zip'):
            svc.import_backup_zip(LOCAL_USER, b'garbage')
        # a zip without our manifest is refused
        buf = _io.BytesIO()
        with _zip.ZipFile(buf, 'w') as z:
            z.writestr('foo.txt', 'x')
        with pytest.raises(ValueError, match='not a dataset backup'):
            svc.import_backup_zip(LOCAL_USER, buf.getvalue())
        # traversal / nested entries are silently skipped, rows without files dropped
        buf = _io.BytesIO()
        with _zip.ZipFile(buf, 'w') as z:
            z.writestr('manifest.json', '{"format": "lds-dataset-backup", "version": 1, '
                                        '"name": "Evil", "trigger_word": "evil"}')
            z.writestr('images.json', '[{"filename": "../../evil.webp", "status": "keep"}]')
            z.writestr('images/../../evil.webp', 'x')
        restored = svc.import_backup_zip(LOCAL_USER, buf.getvalue())
        from app.models import FaceDatasetImage
        assert FaceDatasetImage.query.filter_by(dataset_id=restored.id).count() == 0
        import os
        assert not os.path.exists(os.path.join(svc._dataset_dir(restored.id), '..', '..', 'evil.webp'))


def test_batch_invalid_action_raises(app):
    import pytest
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Bx', 'bx')
        with pytest.raises(ValueError):
            svc.batch_image_action(LOCAL_USER, ds.id, [1], 'rm_rf')


def _grad_png(direction='ltr', w=800, h=800):
    """Low-frequency horizontal gradient — solid colors all dHash to 0, so dedup
    tests need a pattern that survives the 9x8 downscale (see the scrape tests)."""
    ramp = list(range(0, 256, 32))
    if direction == 'rtl':
        ramp = ramp[::-1]
    small = Image.new('L', (8, 8)); small.putdata([ramp[x] for _ in range(8) for x in range(8)])
    buf = io.BytesIO(); small.resize((w, h), Image.BILINEAR).convert('RGB').save(buf, 'PNG')
    return buf.getvalue()


def test_import_without_crop_keeps_aspect_ratio(app):
    """crop=False must PRESERVE the framing: an 800x400 photo stays 2:1 (no square
    pad, no black bands a LoRA would learn) — the old path padded to 1024x1024."""
    import os
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    buf = io.BytesIO(); Image.new('RGB', (800, 400), (10, 120, 40)).save(buf, 'PNG')
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Ar', 'ar')
        ids, failed = svc.import_images(LOCAL_USER, ds.id, [buf.getvalue()], crop=False)
        assert len(ids) == 1 and failed == 0
        img = svc.db.session.get(FaceDatasetImage, ids[0])
        with Image.open(os.path.join(svc._dataset_dir(ds.id), img.filename)) as im:
            w, h = im.size
    assert (w, h) == (800, 400)   # unchanged (<=1024), NOT padded to a square


def test_import_dedupe_skips_intra_batch_duplicate(app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Dd', 'dd')
        stats = {}
        ids, failed = svc.import_images(LOCAL_USER, ds.id,
                                        [_grad_png('ltr'), _grad_png('ltr'), _grad_png('rtl')],
                                        crop=False, dedupe=True, stats=stats)
        assert len(ids) == 2 and failed == 0          # ltr kept once, rtl distinct
        assert stats == {'duplicates': 1}


def test_import_dedupe_skips_vs_existing_images(app):
    """Re-importing a photo already in the dataset (earlier call) is dropped —
    the hash is computed on the NORMALIZED file, so it matches what's stored."""
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'De', 'de')
        ids1, _ = svc.import_images(LOCAL_USER, ds.id, [_grad_png('ltr')], crop=False, dedupe=True)
        assert len(ids1) == 1
        stats = {}
        ids2, _ = svc.import_images(LOCAL_USER, ds.id, [_grad_png('ltr')],
                                    crop=False, dedupe=True, stats=stats)
        assert ids2 == [] and stats == {'duplicates': 1}


def test_import_dedupe_off_by_default(app):
    """Historical behavior preserved: without dedupe=True the same bytes import twice
    (scrape flow dedupes upstream on the originals and must not pay a second pass)."""
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Df', 'df')
        ids1, _ = svc.import_images(LOCAL_USER, ds.id, [_grad_png('ltr')], crop=False)
        ids2, _ = svc.import_images(LOCAL_USER, ds.id, [_grad_png('ltr')], crop=False)
        assert len(ids1) == 1 and len(ids2) == 1


class _SerialPool:
    """Deterministic stand-in for ThreadPoolExecutor: the real 3-worker pool on the
    test's shared in-memory sqlite is flaky (thread-scoped sessions racing on one
    connection). Prod runs a WAL file DB — the concurrency isn't what's under test."""
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def map(self, fn, items): return [fn(i) for i in items]


def test_api_batch_skips_cancelled_rows(app, monkeypatch):
    """Stop during a Nano Banana batch: cancel_pending deletes the pending rows —
    the worker must then SKIP the API call for those items (each call is billed),
    not generate-then-discard."""
    import concurrent.futures
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor', _SerialPool)
    calls = []
    monkeypatch.setattr(svc, '_api_generate_fn',
                        lambda engine: (lambda *a, **k: calls.append(1) or _png()))
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Stop', 'stop')
        import os
        os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
        live = FaceDatasetImage(dataset_id=ds.id, status='pending', klein_model='nanobanana')
        gone = FaceDatasetImage(dataset_id=ds.id, status='pending', klein_model='nanobanana')
        svc.db.session.add_all([live, gone]); svc.db.session.commit()
        live_id, gone_id = live.id, gone.id
        svc.db.session.delete(gone); svc.db.session.commit()   # = cancel_pending
        svc._run_nanobanana_batch(app, [(live_id, 'p', '1:1'), (gone_id, 'p', '1:1')],
                                  [_png()], engine='nanobanana')
        assert len(calls) == 1                                  # only the live row hit the API
        # The worker committed in ITS OWN app context — drop this session's stale
        # snapshot before re-reading (same phenomenon link_completed_dataset_image
        # documents for the queue monitor thread).
        svc.db.session.expire_all()
        assert svc.db.session.get(FaceDatasetImage, live_id).filename


def test_api_batch_failure_stores_reason(app, monkeypatch):
    """A failed API generation must persist WHY (fail_reason) — the tile shows it
    instead of a mute 'failed'. Exposed in the payload; cleared on regenerate."""
    import concurrent.futures
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor', _SerialPool)
    def boom(*a, **k):
        raise RuntimeError('quota exceeded (429)')
    monkeypatch.setattr(svc, '_api_generate_fn', lambda engine: boom)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Fr', 'fr')
        import os
        os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
        img = FaceDatasetImage(dataset_id=ds.id, status='pending', klein_model='nanobanana')
        svc.db.session.add(img); svc.db.session.commit()
        svc._run_nanobanana_batch(app, [(img.id, 'p', '1:1')], [_png()], engine='nanobanana')
        svc.db.session.expire_all()
        row = svc.db.session.get(FaceDatasetImage, img.id)
        assert row.status == 'failed'
        assert 'nanobanana' in row.fail_reason and 'quota exceeded' in row.fail_reason
        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        assert payload['images'][0]['fail_reason'] == row.fail_reason


def _ds_with_ref_and_generated(svc, FaceDatasetImage, LOCAL_USER, engine='nanobanana'):
    """A dataset with a reference file on disk + one finished generated tile
    (engine-tagged so regenerate_image re-dispatches through the API path)."""
    import os
    ds = svc.create_dataset(LOCAL_USER, 'R', 'r')
    d = svc._dataset_dir(ds.id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'ref.webp'), 'wb') as fh:
        fh.write(_png())
    ds.ref_filename = 'ref.webp'
    img = FaceDatasetImage(dataset_id=ds.id, status='keep', source='generated',
                           filename=None, klein_model=engine,
                           variation_label='face_front_neutral',
                           variation_prompt='old prompt')
    svc.db.session.add(img)
    svc.db.session.commit()
    return ds, img


def test_regenerate_with_edited_prompt_persists_and_reaches_engine(app, monkeypatch):
    """✏️ edit-prompt regenerate: the edited core prompt is persisted into
    variation_prompt AND reaches the API engine wrapped by the identity guard
    (the face lock stays applied on top of the user's creative edit)."""
    from app.services import face_dataset_service as svc
    from app.services.face_variations import IDENTITY_GUARD
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    seen = {}
    def fake_generate(refs, prompt, aspect_ratio=None):
        seen['prompt'] = prompt
        return _png()
    monkeypatch.setattr(svc, '_api_generate_fn', lambda engine: fake_generate)
    with app.app_context():
        ds, img = _ds_with_ref_and_generated(svc, FaceDatasetImage, LOCAL_USER)
        svc.regenerate_image(LOCAL_USER, img.id, prompt='a candid mirror selfie')  # app=None -> sync
        svc.db.session.expire_all()
        row = svc.db.session.get(FaceDatasetImage, img.id)
        assert row.variation_prompt == 'a candid mirror selfie'   # edit persisted
        assert 'a candid mirror selfie' in seen['prompt']         # reached the engine
        assert IDENTITY_GUARD in seen['prompt']                   # face lock still applied
        assert row.filename                                        # a new file was written


def test_regenerate_without_prompt_keeps_existing(app, monkeypatch):
    """Empty/omitted prompt = current behaviour: variation_prompt is unchanged
    and the stored prompt is what feeds the engine (plain 🔄 / reject path)."""
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    seen = {}
    monkeypatch.setattr(svc, '_api_generate_fn',
                        lambda engine: (lambda refs, prompt, aspect_ratio=None: (seen.update(prompt=prompt) or _png())))
    with app.app_context():
        ds, img = _ds_with_ref_and_generated(svc, FaceDatasetImage, LOCAL_USER)
        svc.regenerate_image(LOCAL_USER, img.id)              # no prompt
        svc.db.session.expire_all()
        row = svc.db.session.get(FaceDatasetImage, img.id)
        assert row.variation_prompt == 'old prompt'           # unchanged
        assert 'old prompt' in seen['prompt']
        svc.regenerate_image(LOCAL_USER, img.id, prompt='   ')  # whitespace-only = no edit
        svc.db.session.expire_all()
        assert svc.db.session.get(FaceDatasetImage, img.id).variation_prompt == 'old prompt'


def test_regenerate_honors_currently_selected_api_engine(app, monkeypatch):
    """A tile born on Klein regenerates through the CURRENTLY selected engine
    when the workspace sends one — the row's origin no longer pins it."""
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    seen = {}
    def make(engine):
        seen['engine'] = engine
        return lambda refs, prompt, aspect_ratio=None: _png()
    monkeypatch.setattr(svc, '_api_generate_fn', make)
    with app.app_context():
        ds, img = _ds_with_ref_and_generated(svc, FaceDatasetImage, LOCAL_USER,
                                             engine='flux-2-klein.safetensors')  # Klein-born
        svc.regenerate_image(LOCAL_USER, img.id, engine='nanobanana')  # app=None -> sync
        svc.db.session.expire_all()
        row = svc.db.session.get(FaceDatasetImage, img.id)
        assert seen['engine'] == 'nanobanana'
        assert row.klein_model == 'nanobanana'    # the row's engine tag follows the switch
        assert row.filename                        # generated through the API path


def test_regenerate_switch_to_klein_uses_picker_model(app, monkeypatch):
    """A tile born on an API engine regenerates through Klein when requested:
    the enqueue receives the workspace's Klein model pick — never the API tag."""
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    seen = {}
    def fake_enqueue(**kwargs):
        seen.update(kwargs)
        return 'job-123'
    monkeypatch.setattr(klein_edit_helper, 'enqueue_klein_edit', fake_enqueue)
    with app.app_context():
        ds, img = _ds_with_ref_and_generated(svc, FaceDatasetImage, LOCAL_USER)  # nanobanana-born
        job = svc.regenerate_image(LOCAL_USER, img.id, engine='klein',
                                   klein_model='flux-2-klein.safetensors')
        assert job == 'job-123'
        assert seen['klein_model'] == 'flux-2-klein.safetensors'
        svc.db.session.expire_all()
        assert (svc.db.session.get(FaceDatasetImage, img.id).klein_model
                == 'flux-2-klein.safetensors')


def test_regenerate_nsfw_stays_local_despite_api_engine(app, monkeypatch):
    """Fail-closed: an NSFW-labelled tile regenerates on the LOCAL Klein path
    even when the workspace's selected engine is an API one (mirrors the batch
    rule — NSFW never reaches a third-party API)."""
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    seen = {}
    def fake_enqueue(**kwargs):
        seen.update(kwargs)
        return 'job-nsfw'
    monkeypatch.setattr(klein_edit_helper, 'enqueue_klein_edit', fake_enqueue)
    monkeypatch.setattr(svc, '_api_generate_fn',
                        lambda engine: (_ for _ in ()).throw(AssertionError('API engine must not be called')))
    with app.app_context():
        ds, img = _ds_with_ref_and_generated(svc, FaceDatasetImage, LOCAL_USER,
                                             engine='flux-2-klein.safetensors')
        img.variation_label = '🔞 custom shot'    # is_nsfw_label() -> True
        svc.db.session.commit()
        job = svc.regenerate_image(LOCAL_USER, img.id, engine='nanobanana')
        assert job == 'job-nsfw'                   # Klein path, not the API one
        assert seen['klein_model'] == 'flux-2-klein.safetensors'


def test_regenerate_skips_engines_disabled_in_settings(app, monkeypatch):
    """A Klein-born tile must not regenerate through Klein once Klein is
    disabled in Settings (engines.enabled) — it falls back to the default
    enabled engine, even when the client sends no engine at all."""
    from app.services import face_dataset_service as svc
    from app import config as cfg
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    seen = {}
    def make(engine):
        seen['engine'] = engine
        return lambda refs, prompt, aspect_ratio=None: _png()
    monkeypatch.setattr(svc, '_api_generate_fn', make)
    with app.app_context():
        cfg.save_config({'engines': {'enabled': ['nanobanana', 'chatgpt'],
                                     'default': 'nanobanana'}})
        ds, img = _ds_with_ref_and_generated(svc, FaceDatasetImage, LOCAL_USER,
                                             engine='flux-2-klein.safetensors')  # Klein-born
        svc.regenerate_image(LOCAL_USER, img.id)          # legacy client: no engine sent
        assert seen['engine'] == 'nanobanana'
        svc.db.session.expire_all()
        assert svc.db.session.get(FaceDatasetImage, img.id).klein_model == 'nanobanana'


def test_regenerate_rejects_unknown_engine(app):
    import pytest
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds, img = _ds_with_ref_and_generated(svc, FaceDatasetImage, LOCAL_USER)
        with pytest.raises(ValueError, match='unknown engine'):
            svc.regenerate_image(LOCAL_USER, img.id, engine='dalle3')


def test_regenerate_prompt_truncated_to_column_limit(app, monkeypatch):
    """A very long edited prompt is truncated to the variation_prompt column (500)."""
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    monkeypatch.setattr(svc, '_api_generate_fn',
                        lambda engine: (lambda refs, prompt, aspect_ratio=None: _png()))
    with app.app_context():
        ds, img = _ds_with_ref_and_generated(svc, FaceDatasetImage, LOCAL_USER)
        svc.regenerate_image(LOCAL_USER, img.id, prompt='x' * 800)
        svc.db.session.expire_all()
        assert len(svc.db.session.get(FaceDatasetImage, img.id).variation_prompt) == 500


def test_regenerate_edited_prompt_exposed_in_payload(app, monkeypatch):
    """After an edit, dataset_payload carries variation_prompt so the ✏️ bubble
    reopens seeded with the current prompt (not blank)."""
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    monkeypatch.setattr(svc, '_api_generate_fn',
                        lambda engine: (lambda refs, prompt, aspect_ratio=None: _png()))
    with app.app_context():
        ds, img = _ds_with_ref_and_generated(svc, FaceDatasetImage, LOCAL_USER)
        svc.regenerate_image(LOCAL_USER, img.id, prompt='new scene, golden hour')
        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        assert payload['images'][0]['variation_prompt'] == 'new scene, golden hour'


def test_delete_dataset_without_lora_training_module(app):
    """lora_training (Task 19) doesn't exist yet in phase 1 -> delete_dataset must
    still succeed (purge step is best-effort and silently skipped)."""
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'D', 'd')
        assert svc.delete_dataset(LOCAL_USER, ds.id) is True
        assert svc.get_dataset(LOCAL_USER, ds.id) is None


def test_detect_head_bbox_falls_back_to_none_when_ollama_unreachable(app, monkeypatch):
    """detect_head_bbox has an existing graceful fallback for 'no detection'
    (face_crop_to_square_webp centers the crop instead) -- an unreachable Ollama
    server must hit THAT path (return None), not raise. requests.post is stubbed
    so this test never touches a real Ollama server."""
    from app.services import face_dataset_service as svc
    from app.services import vision_ollama

    def _raise(*a, **k):
        raise ConnectionError('ollama unreachable')

    monkeypatch.setattr(vision_ollama.requests, 'post', _raise)
    with app.app_context():
        assert svc.detect_head_bbox(_png()) is None
        # face_crop_to_square_webp must still produce a valid centered-crop webp.
        out = svc.face_crop_to_square_webp(_png())
        assert isinstance(out, (bytes, bytearray)) and len(out) > 0


def test_generate_variations_klein_raises_models_missing_when_unconfigured(app):
    """With no comfyui.base_dir configured, the model preflight finds none of the
    Klein files on disk and raises KleinModelsMissing BEFORE creating any rows (the
    route turns that into an actionable 'configure ComfyUI / downloading' 409).
    Needs a non-empty variations list and a reference image (checked first)."""
    import pytest
    from app.services import face_dataset_service as svc
    from app.services.klein_edit_helper import KleinModelsMissing
    from app.config import LOCAL_USER
    import os
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'F', 'f')
        d = svc._dataset_dir(ds.id)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'ref.webp'), 'wb') as fh:
            fh.write(_png())
        ds.ref_filename = 'ref.webp'
        svc.db.session.commit()
        with pytest.raises(KleinModelsMissing):
            svc.generate_variations(LOCAL_USER, ds.id,
                                    [{'label': 'x', 'framing': 'face', 'prompt': 'p'}],
                                    1, 'some_klein_model')


def test_link_completed_dataset_image_without_comfyui_configured(app, monkeypatch):
    """comfyui.base_dir/output_dir are unset in phase-1 test config -> with no
    file on disk AND the /view API unreachable, the completion link must mark the
    row failed instead of crashing (checklist item 3). The API fetch is stubbed to
    None here to keep the test hermetic (no real localhost request)."""
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    monkeypatch.setattr('app.utils.comfyui.fetch_output_image_bytes', lambda *a, **k: None)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'G', 'g')
        img = FaceDatasetImage(dataset_id=ds.id, source='generated', status='pending',
                               job_id='job-123', klein_model='some_klein_model')
        svc.db.session.add(img)
        svc.db.session.commit()
        svc.link_completed_dataset_image('job-123', 'result.webp', failed=False)
        refreshed = svc.db.session.get(FaceDatasetImage, img.id)
        assert refreshed.status == 'failed'


# --- 'generate' activity indicator (blocks ⚡ Generate for the whole batch) ----

def test_api_batch_advertises_generate_activity_then_clears(app, monkeypatch):
    """The API fan-out (Nano Banana / ChatGPT) advertises a 'generate' activity for
    the WHOLE batch — kind + total up front, done growing per image — and clears it
    when the pool drains (finally end()). This is what keeps ⚡ Generate disabled
    past the launch request (busyLive = busy OR activity)."""
    import concurrent.futures, os
    from app.services import face_dataset_service as svc
    from app.services import dataset_activity as da
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor', _SerialPool)
    da.reset()
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Act', 'act')
        os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
        rows = [FaceDatasetImage(dataset_id=ds.id, status='pending', klein_model='nanobanana')
                for _ in range(3)]
        svc.db.session.add_all(rows); svc.db.session.commit()
        items = [(r.id, 'p', '1:1') for r in rows]
        seen = []
        def gen(*a, **k):
            seen.append(da.get(ds.id))   # live indicator captured MID-batch
            return _png()
        monkeypatch.setattr(svc, '_api_generate_fn', lambda engine: gen)
        svc._run_nanobanana_batch(app, items, [_png()], engine='nanobanana', dataset_id=ds.id)
        assert seen and seen[0]['kind'] == 'generate' and seen[0]['total'] == 3
        assert seen[-1]['done'] >= 1                       # bumped per handled item
        # After the batch: cleared (finally end()) — both directly and in the payload.
        assert da.get(ds.id) is None
        assert svc.dataset_payload(LOCAL_USER, ds.id)['activity'] is None


def test_api_batch_generate_activity_cleared_on_pool_exception(app, monkeypatch):
    """end() is guaranteed even if the pool itself raises — the indicator must never
    strand a phantom 'in progress' that would keep Generate disabled forever."""
    import concurrent.futures, os, pytest
    from app.services import face_dataset_service as svc
    from app.services import dataset_activity as da
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER

    class _BoomPool:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def map(self, fn, items): raise RuntimeError('pool crashed')

    monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor', _BoomPool)
    da.reset()
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Boom', 'boom')
        os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
        img = FaceDatasetImage(dataset_id=ds.id, status='pending', klein_model='nanobanana')
        svc.db.session.add(img); svc.db.session.commit()
        with pytest.raises(RuntimeError):
            svc._run_nanobanana_batch(app, [(img.id, 'p', '1:1')], [_png()],
                                      engine='nanobanana', dataset_id=ds.id)
        assert da.get(ds.id) is None                       # finally end() ran anyway


def test_klein_generate_activity_from_enqueue_to_last_completion(app, monkeypatch):
    """Klein: enqueue advertises 'generate' with the batch total (pending-count
    approximation); each job completion reconciles done; the LAST completion clears
    the indicator. The payload exposes it throughout the batch."""
    import os, itertools
    from app.services import face_dataset_service as svc
    from app.services import dataset_activity as da
    from app.services import klein_edit_helper as keh
    from app.config import LOCAL_USER
    da.reset()
    # Bypass the model preflight and stub the enqueue with deterministic job ids.
    monkeypatch.setattr(keh, 'klein_missing_assets', lambda *a, **k: set())
    counter = itertools.count(1)
    monkeypatch.setattr(keh, 'enqueue_klein_edit', lambda **k: f'job-{next(counter)}')
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'K', 'k')
        d = svc._dataset_dir(ds.id); os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'ref.webp'), 'wb') as fh:
            fh.write(_png())
        ds.ref_filename = 'ref.webp'; svc.db.session.commit()
        vs = [{'label': 'a', 'framing': 'face', 'prompt': 'p1'},
              {'label': 'b', 'framing': 'bust', 'prompt': 'p2'}]
        svc.generate_variations(LOCAL_USER, ds.id, vs, 1, 'some_model')
        act = svc.dataset_payload(LOCAL_USER, ds.id)['activity']
        assert act and act['kind'] == 'generate' and act['total'] == 2 and act['done'] == 0
        # One job finishes (failed path is hermetic — no output file needed).
        svc.link_completed_dataset_image('job-1', 'x.webp', failed=True)
        act = da.get(ds.id)
        assert act and act['kind'] == 'generate' and act['total'] == 2 and act['done'] == 1
        # Last job finishes -> indicator clears (Generate re-enables).
        svc.link_completed_dataset_image('job-2', 'y.webp', failed=True)
        assert da.get(ds.id) is None
        assert svc.dataset_payload(LOCAL_USER, ds.id)['activity'] is None


def test_klein_generate_activity_cleared_on_cancel(app, monkeypatch):
    """Stop deletes the in-flight rows (their completion callbacks never fire), so
    cancel_pending must clear the 'generate' indicator itself."""
    import os, itertools
    from app.services import face_dataset_service as svc
    from app.services import dataset_activity as da
    from app.services import klein_edit_helper as keh
    from app.config import LOCAL_USER
    da.reset()
    monkeypatch.setattr(keh, 'klein_missing_assets', lambda *a, **k: set())
    counter = itertools.count(1)
    monkeypatch.setattr(keh, 'enqueue_klein_edit', lambda **k: f'jc-{next(counter)}')
    # cancel_pending tries to cancel the queued job — stub the queue away.
    with app.app_context():
        import app.job_queue as jq
        monkeypatch.setattr(jq.queue_manager, 'cancel_job', lambda *a, **k: None)
        ds = svc.create_dataset(LOCAL_USER, 'KC', 'kc')
        d = svc._dataset_dir(ds.id); os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'ref.webp'), 'wb') as fh:
            fh.write(_png())
        ds.ref_filename = 'ref.webp'; svc.db.session.commit()
        svc.generate_variations(LOCAL_USER, ds.id,
                                [{'label': 'a', 'framing': 'face', 'prompt': 'p'}], 2, 'm')
        assert da.get(ds.id)['kind'] == 'generate' and da.get(ds.id)['total'] == 2
        svc.cancel_pending(LOCAL_USER, ds.id)
        assert da.get(ds.id) is None


# --- Import d'un dataset existant (ZIP kohya) --------------------------------
def _training_zip(entries):
    """entries: list of (arcname, bytes) — builds an in-memory zip."""
    import io as _io, zipfile as _zip
    buf = _io.BytesIO()
    with _zip.ZipFile(buf, 'w') as z:
        for name, data in entries:
            z.writestr(name, data)
    return buf.getvalue()


def _patterned_png(seed):
    """Distinct NON-uniform image: solid colors all share the same (zero) dHash
    and would read as perceptual duplicates of each other."""
    im = Image.new('RGB', (64, 64), (255, 255, 255))
    for i in range(8):
        x = (seed * 13 + i * 7) % 56
        im.paste(((seed * 37) % 255, (i * 61) % 255, (seed * 7 + i * 29) % 255),
                 (x, i * 8, x + 8, i * 8 + 8))
    buf = io.BytesIO(); im.save(buf, 'PNG')
    return buf.getvalue()


def test_import_dataset_zip_images_and_captions(app):
    """Kohya layout: images at any depth + same-stem .txt sidecars become rows
    with captions; non-image files are ignored; aspect is preserved."""
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'ZipIn', 'zipin')
        zb = _training_zip([
            ('10_woman/a.png', _patterned_png(1)),
            ('10_woman/a.txt', b'a woman standing on a beach, bikini'),
            ('10_woman/b.png', _patterned_png(2)),
            ('notes.md', b'ignore me'),
        ])
        stats = {}
        ids, failed = svc.import_dataset_zip(LOCAL_USER, ds.id, zb, stats=stats)
        assert len(ids) == 2 and failed == 0
        assert stats.get('captions') == 1
        rows = FaceDatasetImage.query.filter_by(dataset_id=ds.id).all()
        caps = {r.caption for r in rows}
        assert 'a woman standing on a beach, bikini' in caps
        assert all(r.status == 'keep' and r.source == 'import' for r in rows)


def test_import_dataset_zip_dedupes_and_rejects_bad_zip(app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'ZipDup', 'zipdup')
        same = _png((7, 7, 7))
        zb = _training_zip([('a.png', same), ('b.png', same)])   # perceptual dupe
        stats = {}
        ids, _ = svc.import_dataset_zip(LOCAL_USER, ds.id, zb, stats=stats)
        assert len(ids) == 1 and stats.get('duplicates') == 1
        try:
            svc.import_dataset_zip(LOCAL_USER, ds.id, b'not a zip at all')
            assert False, 'expected ValueError'
        except ValueError as e:
            assert 'zip' in str(e)


def test_import_zip_route(client, app):
    import io as _io
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'ZipRoute', 'ziproute')
        did = ds.id
    zb = _training_zip([('img.png', _png((9, 90, 200))), ('img.txt', b'caption here')])
    resp = client.post(f'/api/dataset/{did}/import-zip',
                       data={'file': (_io.BytesIO(zb), 'train.zip')},
                       content_type='multipart/form-data')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['imported'] == 1 and body['captions'] == 1
    assert client.post(f'/api/dataset/{did}/import-zip').status_code == 400  # no file


def test_subscription_quota_fails_remaining_rows_fast(app, monkeypatch):
    """Quota-429 mid-batch: the current row AND all remaining rows fail with a
    clear quota message, without burning more API calls. Never a silent switch
    to the paid API key."""
    import concurrent.futures
    from app.services import face_dataset_service as svc
    from app.services.chatgpt_image import SubscriptionQuotaExceeded
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor', _SerialPool)
    calls = []
    def boom(*a, **k):
        calls.append(1)
        raise SubscriptionQuotaExceeded('quota reached')
    monkeypatch.setattr(svc, '_api_generate_fn', lambda engine: boom)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Q', 'q')
        import os
        os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
        rows = [FaceDatasetImage(dataset_id=ds.id, status='pending', klein_model='chatgpt')
                for _ in range(3)]
        svc.db.session.add_all(rows); svc.db.session.commit()
        items = [(r.id, 'p', '1:1') for r in rows]
        svc._run_nanobanana_batch(app, items, [_png()], engine='chatgpt')
        assert len(calls) == 1                      # rows 2-3 never hit the API
        svc.db.session.expire_all()
        for r in rows:
            row = svc.db.session.get(FaceDatasetImage, r.id)
            assert row.status == 'failed'
            assert 'quota' in row.fail_reason


def test_subscription_disconnect_never_falls_back_to_api_key(app, monkeypatch):
    """INVARIANT: a mid-batch disconnect on the pinned subscription lane must
    stop the batch, never reroute the remaining rows onto the paid API key.
    The lane is pinned ONCE before the loop (force_lane='subscription'), so
    even though _use_subscription() would report False after a disconnect
    (token gone), rows 2-3 must still fail with the 'connection lost' message
    instead of silently calling the API-key path."""
    import concurrent.futures
    from app.services import face_dataset_service as svc
    from app.services import chatgpt_image
    from app.services.chatgpt_image import SubscriptionUnavailable
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor', _SerialPool)
    # Pin decides 'subscription' at batch start.
    monkeypatch.setattr(chatgpt_image, '_use_subscription', lambda: True)
    calls = []
    def boom(*a, **k):
        calls.append(1)
        raise SubscriptionUnavailable('ChatGPT connection lost — reconnect in Settings')
    monkeypatch.setattr(svc, '_api_generate_fn', lambda engine: boom)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'L', 'l')
        import os
        os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
        rows = [FaceDatasetImage(dataset_id=ds.id, status='pending', klein_model='chatgpt')
                for _ in range(3)]
        svc.db.session.add_all(rows); svc.db.session.commit()
        items = [(r.id, 'p', '1:1') for r in rows]
        svc._run_nanobanana_batch(app, items, [_png()], engine='chatgpt')
        # Exactly 1 call: the FIRST row hits the disconnected subscription lane,
        # rows 2-3 are stopped BEFORE any call — never routed to the API key.
        assert len(calls) == 1
        svc.db.session.expire_all()
        for r in rows:
            row = svc.db.session.get(FaceDatasetImage, r.id)
            assert row.status == 'failed'
            assert 'connection lost' in row.fail_reason
