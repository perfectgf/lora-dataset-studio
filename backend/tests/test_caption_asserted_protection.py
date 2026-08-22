"""A forced caption pass never rewrites a human's words.

The caption editor is where 'asserted' is born, and its tooltip has promised
since the origin column shipped that "a forced 🔄 Re-caption then skips it".
The bank pass kept that promise from day one (``start_caption`` prices and
walks a query filtered by ``caption_origin.unprotected_clause``); the dataset
pass did not — ``force=True`` overwrote every kept caption, hand-written ones
included. Classic two-surfaces drift (CLAUDE.md): the protection existed once,
on one of two passes that make the same promise to the user.

Pinned here, on the dataset side:

* a forced BATCH spares 'asserted' rows and rewrites everything else — a NULL
  origin stays re-captionable on purpose (``models.py`` says why);
* naming images (``image_ids``) is the explicit opt-out — the leak panel
  re-captions a leaking caption no matter who wrote it;
* a caption the user types WHILE an image sits in inference wins over the
  answer that inference returns (the bank's mid-pass guard, ported);
* the dual-caption derive pass extends the same promise to hand-written shorts.
"""
import json
import os

from PIL import Image

from app.extensions import db
from app.models import FaceDatasetImage
from app.services import caption_origin
from app.services import face_dataset_service as svc
from app.config import LOCAL_USER, save_config


def _kept_image(ds, fn, caption=None, origin=None):
    img_dir = svc._dataset_dir(ds.id)
    os.makedirs(img_dir, exist_ok=True)
    Image.new('RGB', (32, 32)).save(os.path.join(img_dir, fn))
    row = FaceDatasetImage(dataset_id=ds.id, status='keep', filename=fn,
                           caption=caption, caption_origin=origin)
    db.session.add(row)
    db.session.commit()
    return row.id


def _ollama(monkeypatch, text='an ollama caption of a person in a room'):
    from app.services import vision_ollama
    monkeypatch.setattr(vision_ollama, 'describe_image_ollama',
                        lambda *a, **k: text)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)


def test_forced_batch_spares_hand_written_captions(app, monkeypatch):
    _ollama(monkeypatch)
    with app.app_context():
        save_config({'captioning': {'backend': 'ollama'}})
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        asserted = _kept_image(ds, 'a.png', caption='my own words',
                               origin=caption_origin.ASSERTED)
        machine = _kept_image(ds, 'b.png', caption='machine words',
                              origin=caption_origin.OLLAMA)
        legacy = _kept_image(ds, 'c.png', caption='pre-origin words', origin=None)
        n = svc.caption_images(LOCAL_USER, ds.id, force=True)
        assert n == 2, 'the forced batch must rewrite exactly the non-asserted rows'
        assert db.session.get(FaceDatasetImage, asserted).caption == 'my own words', (
            "the editor's tooltip promises a forced Re-caption skips hand-written "
            'captions — the dataset pass just overwrote one')
        assert db.session.get(FaceDatasetImage, machine).caption != 'machine words'
        assert db.session.get(FaceDatasetImage, legacy).caption != 'pre-origin words', (
            'a NULL origin is re-captionable on purpose — sparing it would make '
            'Re-caption inert on every dataset that predates the origin column')


def test_targeted_recaption_rewrites_an_asserted_caption(app, monkeypatch):
    """Pointing at the image IS the ask: the Identity-leak panel re-captions a
    leaking caption regardless of who wrote it."""
    _ollama(monkeypatch)
    with app.app_context():
        save_config({'captioning': {'backend': 'ollama'}})
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        asserted = _kept_image(ds, 'a.png', caption='my leaking words',
                               origin=caption_origin.ASSERTED)
        n = svc.caption_images(LOCAL_USER, ds.id, force=True, image_ids=[asserted])
        assert n == 1
        assert db.session.get(FaceDatasetImage, asserted).caption != 'my leaking words'


def test_a_caption_typed_mid_pass_survives_the_pass(app, monkeypatch):
    """The pass plans its work before inference starts; a human can type into
    the editor while the model is still thinking. Their words win — on both
    surfaces (this is the bank's mid-pass guard, ported)."""
    from app.services import vision_ollama

    with app.app_context():
        save_config({'captioning': {'backend': 'ollama'}})
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        image_id = _kept_image(ds, 'a.png', caption='machine words',
                               origin=caption_origin.OLLAMA)

        def type_during_inference(*a, **k):
            row = db.session.get(FaceDatasetImage, image_id)
            caption_origin.stamp(row, 'typed while the model was thinking',
                                 caption_origin.ASSERTED)
            db.session.commit()
            return 'the answer inference returns'

        monkeypatch.setattr(vision_ollama, 'describe_image_ollama',
                            type_during_inference)
        monkeypatch.setattr(vision_ollama, 'unload_vision_model',
                            lambda *a, **k: True)
        svc.caption_images(LOCAL_USER, ds.id, force=True)
        assert (db.session.get(FaceDatasetImage, image_id).caption
                == 'typed while the model was thinking'), (
            'a caption written during inference must not be replaced by the '
            'answer that inference returns')


def test_derive_short_spares_a_hand_written_short(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        ds.train_settings = json.dumps({'dual_captions': True})
        db.session.commit()
        yours = _kept_image(ds, 'a.png', caption='a long machine caption',
                            origin=caption_origin.OLLAMA)
        row = db.session.get(FaceDatasetImage, yours)
        caption_origin.stamp(row, 'my own short', caption_origin.ASSERTED,
                             field='caption_short')
        machine = _kept_image(ds, 'b.png', caption='another long caption',
                              origin=caption_origin.OLLAMA)
        row = db.session.get(FaceDatasetImage, machine)
        caption_origin.stamp(row, 'machine short', caption_origin.OLLAMA,
                             field='caption_short')
        db.session.commit()
        n = svc.derive_short_captions(LOCAL_USER, ds.id, force=True,
                                      generate=lambda p: 'a derived short')
        assert n == 1, 'the forced derive must rewrite exactly the machine short'
        assert (db.session.get(FaceDatasetImage, yours).caption_short
                == 'my own short'), (
            'the expanded editor stamps shorts asserted too — the derive pass '
            'must spare them like the long pass spares longs')
        assert (db.session.get(FaceDatasetImage, machine).caption_short
                == 'a derived short')
