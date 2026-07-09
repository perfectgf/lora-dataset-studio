"""Concept LoRA — the caption-quality trio + the masked-training guard.

Covers the pieces ported from the source app on top of the base concept feature:
  A5  export forces masked training OFF for a concept dataset (a person-mask would
      erase the very concept we want to learn).
  A2  concept-omission guarantee: ban-list (LLM expansion cached in ds.concept_terms)
      -> leak-detection -> corrective LLM rewrite -> mechanical clause-scrub.
  A3  Joy->Qwen refinement: JoyCaption is literal (names the act) so its drafts are
      rewritten by Qwen; an unusable refine (reasoning trace) falls back to a direct
      Qwen caption.
  Route: creating a concept dataset without a description is a 400, not a 500.

Single-user extraction: LOCAL_USER (no admin fixture). The vision service is imported
LOCALLY by the caption pipeline, so we patch it at the source module
(app.services.vision_ollama.*) and the JoyCaption seam at app.services.joycaption.*.
"""
import io
import os

import pytest
from PIL import Image

from app.extensions import db
from app.models import FaceDataset, FaceDatasetImage
from app.services import face_dataset_service as svc
from app.config import LOCAL_USER, save_config

CONCEPT_DESC = 'licking ice cream'  # fallback ban-list: {licking, ice, cream}


def _png(w=512, h=512):
    b = io.BytesIO()
    Image.new('RGB', (w, h), (120, 40, 40)).save(b, 'PNG')
    return b.getvalue()


def _concept_with_image(app_ctx_desc=CONCEPT_DESC):
    """A concept dataset holding one kept image with a real file on disk, no caption."""
    ds = svc.create_dataset(LOCAL_USER, 'CIM', 'cim_act', kind='concept', concept_desc=app_ctx_desc)
    ids, _ = svc.import_images(LOCAL_USER, ds.id, [_png()], crop=False)
    return ds, db.session.get(FaceDatasetImage, ids[0])


# --- Route guard -------------------------------------------------------------
def test_route_create_concept_without_desc_is_400(client):
    r = client.post('/api/dataset/create',
                    json={'name': 'CIM', 'trigger_word': 'cim_act', 'kind': 'concept'})
    assert r.status_code == 400
    assert 'concept_desc' in (r.get_json().get('error') or '')
    # a character never needs it
    r2 = client.post('/api/dataset/create',
                     json={'name': 'Emma', 'trigger_word': 'zchar_emma'})
    assert r2.status_code == 200 and r2.get_json().get('ok') is True


# --- A5: masked training forced OFF for concept ------------------------------
def test_export_forces_masked_off_for_concept(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    with app.app_context():
        save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, 'CIM', 'cim_act', kind='concept', concept_desc=CONCEPT_DESC)
        img_dir = svc._dataset_dir(ds.id)
        for i in range(3):
            fn = f'k{i}.png'
            Image.new('RGB', (32, 32)).save(os.path.join(img_dir, fn))
            db.session.add(FaceDatasetImage(dataset_id=ds.id, status='keep', filename=fn))
        db.session.commit()

        def boom(*a, **k):
            raise AssertionError('generate_person_masks must NOT run for a concept dataset')

        monkeypatch.setattr(lt, 'generate_person_masks', boom)
        # masked=True requested, but the concept guard must flip it OFF -> no masks call,
        # no masks dir written.
        out = lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=True)
        assert not os.path.isdir(lt._masks_dir(out))


def test_export_keeps_masked_on_for_character(app, tmp_path, monkeypatch):
    """The guard is concept-only: a character export still masks when asked."""
    from app.services import lora_training as lt
    with app.app_context():
        save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')  # character
        img_dir = svc._dataset_dir(ds.id)
        Image.new('RGB', (32, 32)).save(os.path.join(img_dir, 'k0.png'))
        db.session.add(FaceDatasetImage(dataset_id=ds.id, status='keep', filename='k0.png'))
        db.session.commit()

        called = {}

        def fake_masks(paths, out_dir):
            called['yes'] = True
            os.makedirs(out_dir, exist_ok=True)
            for p in paths:
                open(os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + '.png'), 'wb').close()
            return {'ok': True, 'written': len(paths), 'results': {p: 'ok' for p in paths}}

        monkeypatch.setattr(lt, 'generate_person_masks', fake_masks)
        out = lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=True)
        assert called.get('yes') is True
        assert os.path.isdir(lt._masks_dir(out))


# --- A2: ban-list expansion is cached ----------------------------------------
def test_get_concept_terms_caches_llm_expansion(app, tmp_path):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'CIM', 'cim_act', kind='concept', concept_desc=CONCEPT_DESC)
        img = os.path.join(str(tmp_path), 'x.png')
        Image.new('RGB', (32, 32)).save(img)

        calls = {'n': 0}

        def fake_describe(image_bytes, prompt, **kw):
            calls['n'] += 1
            return '{"terms": ["fellatio", "blowjob", "the"]}'  # 'the' is a stopword -> dropped

        terms = svc._get_concept_terms(ds, image_path=img, describe=fake_describe)
        # LLM terms (minus stopword) UNION the desc words
        assert 'fellatio' in terms and 'blowjob' in terms
        assert 'licking' in terms and 'ice' in terms  # from concept_desc
        assert 'the' not in terms
        assert calls['n'] == 1
        # cached on the row
        assert ds.concept_terms and 'fellatio' in ds.concept_terms

        # second call must NOT hit the LLM again (would raise if it did)
        def boom(*a, **k):
            raise AssertionError('expansion must be cached, not recomputed')
        terms2 = svc._get_concept_terms(ds, image_path=img, describe=boom)
        assert 'fellatio' in terms2 and 'licking' in terms2


# --- A2: leak detected in a caption is rewritten then scrubbed ----------------
def test_caption_concept_enforces_omission_via_llm_fix(app, monkeypatch):
    from app.services import vision_ollama
    with app.app_context():
        save_config({'captioning': {'backend': 'ollama'}})  # no Joy; every image direct-Qwen
        ds, img = _concept_with_image()

        def fake_describe(image_bytes, prompt, **kw):
            if 'BLOCKLIST' in prompt:
                return 'not json -> forces desc-word fallback ban-list'
            if 'forbidden words' in prompt:                 # corrective rewrite -> clean
                return 'A woman with red hair stands in a sunny park.'
            return 'A woman licking ice cream in a sunny park.'  # caption -> LEAKS

        monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
        monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda: None)
        n = svc.caption_images(LOCAL_USER, ds.id)

        assert n == 1
        db.session.refresh(img)
        cap = (img.caption or '').lower()
        # the concept words are gone, the context (park, red hair) survives
        for banned in ('ice', 'cream', 'licking'):
            assert banned not in cap
        assert 'park' in cap and 'red hair' in cap


def test_caption_concept_mechanical_scrub_when_no_llm_fix(app, monkeypatch):
    """If the corrective LLM keeps leaking, the mechanical clause-scrub is the net."""
    from app.services import vision_ollama
    with app.app_context():
        save_config({'captioning': {'backend': 'ollama'}})
        ds, img = _concept_with_image()

        def fake_describe(image_bytes, prompt, **kw):
            if 'BLOCKLIST' in prompt:
                return 'no json'
            if 'forbidden words' in prompt:      # LLM fix STILL leaks -> scrub must save it
                return 'A woman still licking ice cream, red hair, in a park.'
            return 'A woman with red hair, licking ice cream, in a sunny park.'

        monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
        monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda: None)
        svc.caption_images(LOCAL_USER, ds.id)

        db.session.refresh(img)
        cap = (img.caption or '').lower()
        for banned in ('ice', 'cream', 'licking'):
            assert banned not in cap
        assert 'red hair' in cap  # a clean clause survived the scrub


# --- A3: Joy -> Qwen refinement ---------------------------------------------
def _patch_joy(monkeypatch, draft):
    import app.services.joycaption as jc
    monkeypatch.setattr(jc, 'is_available', lambda: True)
    monkeypatch.setattr(jc, 'caption_images_joycaption',
                        lambda paths, prompt=None: {p: draft for p in paths})


def test_caption_concept_uses_clean_qwen_refine(app, monkeypatch):
    from app.services import vision_ollama
    with app.app_context():
        save_config({'captioning': {'backend': 'auto'}})  # Joy then Qwen refine
        ds, img = _concept_with_image()
        _patch_joy(monkeypatch, 'A woman licking ice cream, watermark xxx.com, red hair.')

        def fake_describe(image_bytes, prompt, **kw):
            if 'BLOCKLIST' in prompt:
                return 'no json'
            if 'Rewrite it as ONE clean caption' in prompt:   # the refine pass -> clean
                return 'A woman with red hair in a sunlit kitchen.'
            return 'DIRECT-QWEN should not be used here'

        monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
        monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda: None)
        n = svc.caption_images(LOCAL_USER, ds.id)

        assert n == 1
        db.session.refresh(img)
        assert img.caption == 'A woman with red hair in a sunlit kitchen.'
        assert 'ice' not in img.caption.lower()


def test_caption_concept_falls_back_to_direct_qwen_on_reasoning_trace(app, monkeypatch):
    from app.services import vision_ollama
    with app.app_context():
        save_config({'captioning': {'backend': 'auto'}})
        ds, img = _concept_with_image()
        _patch_joy(monkeypatch, 'A woman licking ice cream, red hair.')

        def fake_describe(image_bytes, prompt, **kw):
            if 'BLOCKLIST' in prompt:
                return 'no json'
            if 'Rewrite it as ONE clean caption' in prompt:   # refine emits its reasoning
                return 'The task says we need to remove the licking and rephrase...'
            if 'forbidden words' in prompt:
                return 'A woman with red hair in a park.'
            # the direct-Qwen concept caption (fallback after the bad refine)
            return 'A woman with red hair, licking ice cream, in a park.'

        monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
        monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda: None)
        n = svc.caption_images(LOCAL_USER, ds.id)

        assert n == 1
        db.session.refresh(img)
        cap = (img.caption or '').lower()
        # reasoning trace rejected -> a real caption stored, concept still omitted
        assert 'the task says' not in cap
        for banned in ('ice', 'cream', 'licking'):
            assert banned not in cap
        assert 'red hair' in cap


# --- Backend 'joycaption' concept: Joy draft + mechanical scrub, NO Qwen -----
def test_caption_concept_joycaption_backend_scrubs_without_ollama(app, monkeypatch):
    from app.services import vision_ollama
    with app.app_context():
        save_config({'captioning': {'backend': 'joycaption'}})  # explicit: no Ollama fallback
        ds, img = _concept_with_image()
        _patch_joy(monkeypatch, 'A woman with red hair, licking ice cream, in a sunny park.')

        def boom(*a, **k):
            raise AssertionError("backend='joycaption' must not call Ollama")

        monkeypatch.setattr(vision_ollama, 'describe_image_ollama', boom)
        monkeypatch.setattr(vision_ollama, 'unload_vision_model', boom)
        n = svc.caption_images(LOCAL_USER, ds.id)

        assert n == 1
        db.session.refresh(img)
        cap = (img.caption or '').lower()
        for banned in ('ice', 'cream', 'licking'):
            assert banned not in cap
        assert 'red hair' in cap and 'park' in cap
