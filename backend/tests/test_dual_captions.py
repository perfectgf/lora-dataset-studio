"""Dual long+short captioning (ai-toolkit short_and_long_captions).

Covers the whole seam:
  - additive caption_short migration on a LEGACY table (no column yet);
  - the dual_captions train-setting round-trip (update/effective + preset whitelist);
  - short-caption derivation from the long one (text-only stub) preserving the kind
    omission (concept ban-list scrub, style aesthetic strip), and no-op when OFF;
  - export writing the JSON caption file + the recipe pointing folder_path at it with
    short_and_long_captions on — and OFF being byte-identical to the historical export;
  - the cloud path stripping dual back to the historical folder + .txt sidecars.

Single-user extraction (LOCAL_USER). The vision seam is imported locally by the pipeline,
so it is patched at app.services.vision_ollama.*.
"""
import io
import json
import os

from PIL import Image
from sqlalchemy import text

from app.extensions import db
from app.models import FaceDataset, FaceDatasetImage
from app.services import face_dataset_service as svc
from app.services import lora_training as lt
from app.config import LOCAL_USER, save_config


def _png(w=64, h=64):
    b = io.BytesIO()
    Image.new('RGB', (w, h), (120, 40, 40)).save(b, 'PNG')
    return b.getvalue()


def _enable_dual(ds):
    ds.train_settings = json.dumps({'dual_captions': True})
    db.session.commit()


def _kept_image(ds, fn, caption=None, caption_short=None):
    img_dir = svc._dataset_dir(ds.id)
    os.makedirs(img_dir, exist_ok=True)
    Image.new('RGB', (32, 32)).save(os.path.join(img_dir, fn))
    row = FaceDatasetImage(dataset_id=ds.id, status='keep', filename=fn,
                           caption=caption, caption_short=caption_short)
    db.session.add(row)
    db.session.commit()
    return row


# --- additive migration on a legacy table ------------------------------------
def test_caption_short_migration_on_legacy_table(app):
    with app.app_context():
        from app import _apply_additive_migrations
        # Rebuild the table WITHOUT caption_short to mimic a database created before the
        # column existed, with a legacy row present.
        db.session.execute(text('DROP TABLE face_dataset_image'))
        db.session.execute(text(
            'CREATE TABLE face_dataset_image ('
            'id INTEGER PRIMARY KEY, dataset_id INTEGER, status TEXT, caption TEXT)'))
        db.session.execute(text(
            "INSERT INTO face_dataset_image (id, dataset_id, status, caption) "
            "VALUES (1, 1, 'keep', 'a long caption')"))
        db.session.commit()
        cols = {r[1] for r in db.session.execute(text('PRAGMA table_info(face_dataset_image)'))}
        assert 'caption_short' not in cols

        _apply_additive_migrations()

        cols = {r[1] for r in db.session.execute(text('PRAGMA table_info(face_dataset_image)'))}
        assert 'caption_short' in cols
        # Legacy row survived and the new column is NULL for it.
        row = db.session.execute(text(
            'SELECT caption, caption_short FROM face_dataset_image WHERE id=1')).first()
        assert row[0] == 'a long caption' and row[1] is None
        # Idempotent: a second run must not raise.
        _apply_additive_migrations()


# --- train-setting round-trip -------------------------------------------------
def test_dual_captions_setting_roundtrip(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        assert lt.effective_train_settings(ds)['dual_captions'] is False
        assert svc.dual_captions_enabled(ds) is False

        eff = lt.update_train_settings(LOCAL_USER, ds.id, {'dual_captions': True})
        assert eff['dual_captions'] is True
        assert svc.dual_captions_enabled(db.session.get(FaceDataset, ds.id)) is True

        # Falsy drops the key so OFF is byte-identical to a dataset that never set it.
        eff = lt.update_train_settings(LOCAL_USER, ds.id, {'dual_captions': False})
        assert eff['dual_captions'] is False
        stored = json.loads(db.session.get(FaceDataset, ds.id).train_settings or '{}')
        assert 'dual_captions' not in stored

        assert 'dual_captions' in lt.TRAIN_SETTING_KEYS


def test_preset_apply_roundtrips_dual_captions(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        eff, ignored, rejected = lt.apply_train_settings_dict(
            LOCAL_USER, ds.id, {'dual_captions': True})
        assert eff['dual_captions'] is True
        assert 'dual_captions' not in ignored and not rejected


# --- derivation preserves the kind omission ----------------------------------
def test_derive_short_scrubs_concept_ban_list(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'CIM', 'cim_act', kind='concept',
                                concept_desc='licking ice cream')
        _enable_dual(ds)
        img = _kept_image(ds, 'k0.png', caption='A woman on a bench with a dessert')

        # The shortener tries to reintroduce the banned concept; the mechanical scrub
        # (describe=None) must drop it, exactly like the long-caption guarantee.
        leaky = 'A woman on a bench, licking ice cream'
        n = svc.derive_short_captions(LOCAL_USER, ds.id, generate=lambda p: leaky)
        assert n == 1
        short = db.session.get(FaceDatasetImage, img.id).caption_short
        assert short
        low = short.lower()
        assert 'ice' not in low and 'cream' not in low and 'licking' not in low


def test_derive_short_strips_style_trigger(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Look', 'zstyle_look', kind='style')
        _enable_dual(ds)
        img = _kept_image(ds, 'k0.png', caption='a woman standing in a field')

        # A style short must stay content-only: a stray leading trigger is stripped.
        n = svc.derive_short_captions(LOCAL_USER, ds.id,
                                      generate=lambda p: 'zstyle_look, a woman in a field')
        assert n == 1
        short = db.session.get(FaceDatasetImage, img.id).caption_short
        assert short and not short.lower().startswith('zstyle_look')


def test_derive_short_noop_when_disabled(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')   # dual OFF
        img = _kept_image(ds, 'k0.png', caption='a long caption')
        called = {'n': 0}

        def gen(p):
            called['n'] += 1
            return 'short'

        assert svc.derive_short_captions(LOCAL_USER, ds.id, generate=gen) == 0
        assert called['n'] == 0
        assert db.session.get(FaceDatasetImage, img.id).caption_short is None


def test_derive_short_fills_only_missing_unless_forced(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        _enable_dual(ds)
        a = _kept_image(ds, 'a.png', caption='long a')
        b = _kept_image(ds, 'b.png', caption='long b', caption_short='kept short b')

        # force=False fills only the missing one, leaving an existing short untouched.
        n = svc.derive_short_captions(LOCAL_USER, ds.id, generate=lambda p: 'derived')
        assert n == 1
        assert db.session.get(FaceDatasetImage, a.id).caption_short == 'derived'
        assert db.session.get(FaceDatasetImage, b.id).caption_short == 'kept short b'

        # force=True overwrites all.
        n = svc.derive_short_captions(LOCAL_USER, ds.id, force=True, generate=lambda p: 'fresh')
        assert n == 2
        assert db.session.get(FaceDatasetImage, b.id).caption_short == 'fresh'


# --- set_image_caption short param -------------------------------------------
def test_set_image_caption_short_is_opt_in(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        img = _kept_image(ds, 'k0.png', caption='old', caption_short='old short')

        # Omitting short leaves the existing short intact.
        svc.set_image_caption(LOCAL_USER, img.id, 'new long')
        row = db.session.get(FaceDatasetImage, img.id)
        assert row.caption == 'new long' and row.caption_short == 'old short'

        # Passing short updates it.
        svc.set_image_caption(LOCAL_USER, img.id, 'new long', short='new short')
        row = db.session.get(FaceDatasetImage, img.id)
        assert row.caption_short == 'new short'


# --- export writes the JSON caption file, recipe points at it -----------------
def test_export_writes_dual_json_and_recipe(app, tmp_path):
    with app.app_context():
        save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        _enable_dual(ds)
        _kept_image(ds, 'a.png', caption='a full caption', caption_short='a short')
        _kept_image(ds, 'b.png', caption='b full caption')   # no short → fallback to long

        out = lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=False,
                                             dest_dir=str(tmp_path / 'export'))
        json_path = lt._dual_caption_json_path(out)
        assert os.path.isfile(json_path)
        data = json.loads(open(json_path, encoding='utf-8').read())
        assert len(data) == 2
        for entry in data.values():
            assert set(entry) == {'caption', 'caption_short'}
            # Character trigger prepended to BOTH the long and the short.
            assert entry['caption'].startswith('zchar_emma,')
            assert entry['caption_short'].startswith('zchar_emma,')
        # Missing short degrades to the long caption (short == long). Exported files are
        # renamed <trigger>_NNN.png, so match on the caption content, not the source name.
        b_entry = next(v for v in data.values() if 'b full caption' in v['caption'])
        assert b_entry['caption_short'] == b_entry['caption']

        # Recipe points folder_path at the JSON and turns on short_and_long_captions.
        cfg = lt.build_job_config(ds, out, steps=100)
        proc = cfg['config']['process'][0]
        assert proc['datasets'][0]['folder_path'] == json_path
        assert proc['train']['short_and_long_captions'] is True


def test_export_off_is_historical(app, tmp_path):
    with app.app_context():
        save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')   # dual OFF
        _kept_image(ds, 'a.png', caption='a full caption', caption_short='ignored short')

        out = lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=False,
                                             dest_dir=str(tmp_path / 'export'))
        # No JSON file; the .txt sidecar is the only caption source.
        assert not os.path.isfile(lt._dual_caption_json_path(out))
        assert os.path.isfile(os.path.join(out, 'zchar_emma_000.txt'))

        cfg = lt.build_job_config(ds, out, steps=100)
        proc = cfg['config']['process'][0]
        assert proc['datasets'][0]['folder_path'] == out
        assert 'short_and_long_captions' not in proc.get('train', {})


# --- the re-caption route regenerates BOTH captions --------------------------
def test_caption_route_regenerates_both_when_dual_on(app, client, monkeypatch):
    from app.services import vision_ollama
    with app.app_context():
        save_config({'captioning': {'backend': 'ollama'}})   # skip JoyCaption
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        _enable_dual(ds)
        img = _kept_image(ds, 'k0.png')   # no caption yet
        ds_id = ds.id
        img_id = img.id

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama',
                        lambda *a, **k: 'a woman standing in a park')
    monkeypatch.setattr(vision_ollama, 'generate_text_ollama',
                        lambda *a, **k: 'a woman in a park')
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)

    r = client.post(f'/api/dataset/{ds_id}/caption', json={'force': True})
    assert r.status_code == 200

    with app.app_context():
        row = db.session.get(FaceDatasetImage, img_id)
        assert (row.caption or '').strip()          # long written
        assert (row.caption_short or '').strip()     # short derived in the same pass


# --- cloud strips dual back to the historical shape --------------------------
def test_cloudify_strips_dual_captions(app):
    from app.services import cloud_training as ct
    with app.app_context():
        staging = 'C:/stage/dataset'
        job_config = {'job': 'extension', 'config': {'name': 'x', 'process': [{
            'type': 'sd_trainer',
            'datasets': [{'folder_path': staging + '/_captions.json', 'caption_ext': 'txt'}],
            'train': {'steps': 100, 'short_and_long_captions': True},
            'model': {},
        }]}}
        pod_settings = {'DATASETS_FOLDER': '/workspace/datasets',
                        'TRAINING_FOLDER': '/workspace/out'}
        out = ct._cloudify_job_config(job_config, 'myjob', staging, pod_settings)
        proc = out['config']['process'][0]
        # Reverted to the historical folder + .txt sidecars, dual flag dropped.
        assert proc['datasets'][0]['folder_path'] == '/workspace/datasets/myjob'
        assert proc['datasets'][0]['caption_ext'] == 'txt'
        assert 'short_and_long_captions' not in proc['train']
