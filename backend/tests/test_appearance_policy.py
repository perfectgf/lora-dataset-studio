"""Character appearance policy: omit vs describe.

The historical identity lock is a hardcoded omit list (hair always banned) with a
silent gap (makeup neither asked nor forbidden). Extra instructions cannot move a
family between columns. A per-dataset policy does — and Extra instructions then
survive the cleaner iff that family is Describe.
"""
import json
import os

from PIL import Image

from app.config import LOCAL_USER, save_config
from app.extensions import db
from app.models import FaceDatasetImage
from app.services import face_dataset_service as svc
from app.services import face_variations as fv
from app.services import lora_training as lt


DEFAULTS = fv.normalize_appearance({'hair': 'omit'})  # fills makeup/facial_hair/glasses


def _kept_image(ds, fn, caption=None):
    img_dir = svc._dataset_dir(ds.id)
    os.makedirs(img_dir, exist_ok=True)
    Image.new('RGB', (32, 32)).save(os.path.join(img_dir, fn))
    row = FaceDatasetImage(dataset_id=ds.id, status='keep', filename=fn, caption=caption)
    db.session.add(row)
    db.session.commit()
    return row


# --- normalize / prompt -------------------------------------------------------

def test_normalize_appearance_empty_and_partial():
    assert fv.normalize_appearance(None) == {}
    assert fv.normalize_appearance({}) == {}
    assert fv.normalize_appearance({'nope': 'omit'}) == {}
    pol = fv.normalize_appearance({'hair': 'describe'})
    assert pol['hair'] == 'describe'
    assert pol['makeup'] == 'describe'          # default once a policy exists
    assert pol['facial_hair'] == 'omit'
    assert pol['glasses'] == 'describe'


def test_legacy_prompt_and_cleaners_are_byte_identical_without_a_policy():
    assert fv.caption_prompt_for('prose') == fv.JOYCAPTION_PROMPT
    assert fv.caption_prompt_for('booru') == fv.CAPTION_PROMPT_BOORU
    assert fv.caption_prompt_for('prose', appearance=None) == fv.JOYCAPTION_PROMPT
    assert fv.caption_prompt_for('prose', appearance={}) == fv.JOYCAPTION_PROMPT
    # Makeup is NOT a historical leak — the silent lottery.
    assert fv.caption_has_identity_leak('wearing mascara and red lipstick') is False
    assert fv.caption_has_identity_leak('a woman with long blonde hair') is True


def test_policy_prompt_asks_for_describe_families_and_forbids_omit_ones():
    # Default saved policy: hair omit, makeup describe.
    p = fv.caption_prompt_for('prose', appearance=DEFAULTS)
    assert p != fv.JOYCAPTION_PROMPT
    assert 'ponytail' in p.lower()                 # still forbids hair
    assert 'mascara' in p.lower()                  # asks for makeup
    assert 'no makeup' in p.lower()
    # Flip hair to describe: the forbid examples leave, the MUST-describe lands.
    described = fv.normalize_appearance({'hair': 'describe'})
    d = fv.caption_prompt_for('prose', appearance=described)
    assert 'visible hair' in d.lower()
    assert 'do NOT write "long hair"' not in d
    assert fv.caption_has_identity_leak('long blonde hair', appearance=described) is False
    assert fv.caption_has_identity_leak('pale skin', appearance=described) is True  # core still omits


def test_hair_clip_is_wardrobe_not_identity_under_policy():
    """Overlap: Omit hair must not strip a hair accessory."""
    cap = 'The subject wears a hair clip. Soft window light.'
    # Legacy `\bhair\b` still drops the sentence (byte-identical).
    assert 'hair clip' not in fv.drop_identity_sentences(cap)
    kept = fv.drop_identity_sentences(cap, appearance=DEFAULTS)
    assert 'hair clip' in kept
    assert fv.caption_has_identity_leak(cap, appearance=DEFAULTS) is False
    tags = 'standing, hair_bow, long_hair, red_dress'
    assert 'long_hair' not in fv.drop_identity_tags(tags, appearance=DEFAULTS)
    assert 'hair_bow' in fv.drop_identity_tags(tags, appearance=DEFAULTS)


def test_lipstick_is_makeup_not_lips_anatomy():
    described = DEFAULTS  # makeup describe
    omit_makeup = fv.normalize_appearance({'makeup': 'omit'})
    assert fv.caption_has_identity_leak('wearing red lipstick', appearance=described) is False
    assert fv.caption_has_identity_leak('wearing red lipstick', appearance=omit_makeup) is True
    # Lip *shape* stays locked omit either way.
    assert fv.caption_has_identity_leak('thick dark eyebrows', appearance=described) is True
    tags = 'smile, lipstick, red_dress, lips'
    kept_d = [t.strip() for t in fv.drop_identity_tags(tags, appearance=described).split(',') if t.strip()]
    assert 'lipstick' in kept_d and 'lips' not in kept_d
    kept_o = [t.strip() for t in fv.drop_identity_tags(tags, appearance=omit_makeup).split(',') if t.strip()]
    assert 'lipstick' not in kept_o and 'lips' not in kept_o


# --- persist: Extra instructions vs cleaners ---------------------------------

def test_saving_other_options_does_not_invent_a_policy(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'NoApp', 'zchar_noapp')
        svc.set_caption_options(LOCAL_USER, ds.id, {'instructions': 'name the season'})
        opts = svc.caption_options(ds)
        assert opts['appearance'] == {}
        assert opts['instructions'] == 'name the season'
        stored = json.loads(ds.caption_options)
        assert 'appearance' not in stored


def test_first_toggle_materializes_defaults_then_applies_the_flip(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Flip', 'zchar_flip')
        # Client sends only the flipped family; the server fills the rest.
        opts = svc.set_caption_options(LOCAL_USER, ds.id, {'appearance': {'hair': 'describe'}})
        assert opts['appearance'] == {
            'hair': 'describe', 'makeup': 'describe',
            'facial_hair': 'omit', 'glasses': 'describe'}


def test_empty_appearance_clears_the_policy(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Clr', 'zchar_clr')
        svc.set_caption_options(LOCAL_USER, ds.id, {'appearance': {'hair': 'describe'}})
        svc.set_caption_options(LOCAL_USER, ds.id, {'appearance': {}})
        assert svc.caption_options(ds)['appearance'] == {}
        assert 'appearance' not in json.loads(ds.caption_options or '{}')


def test_extra_describe_the_hair_survives_iff_hair_is_describe(app, client, monkeypatch):
    from app.services import vision_ollama
    with app.app_context():
        save_config({'captioning': {'backend': 'auto'}})
        ds = svc.create_dataset(LOCAL_USER, 'HairOn', 'zchar_hairon')
        svc.set_caption_options(LOCAL_USER, ds.id, {
            'backend': 'ollama',
            'appearance': {'hair': 'describe'},
            'instructions': 'Always describe the hair.',
        })
        img = _kept_image(ds, 'k0.png')
        ds_id, img_id = ds.id, img.id

    captured = {}

    def fake_describe(image_bytes, prompt, **kwargs):
        captured['prompt'] = prompt
        return ('Three-quarter shot of the subject standing. She has long hair in a high ponytail. '
                'Soft daylight, denim jacket.')

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)
    r = client.post(f'/api/dataset/{ds_id}/caption', json={'force': True})
    assert r.status_code == 200 and r.get_json()['captioned'] == 1
    assert 'visible hair' in captured['prompt'].lower()
    assert 'Always describe the hair.' in captured['prompt']
    with app.app_context():
        stored = (db.session.get(FaceDatasetImage, img_id).caption or '')
    assert 'ponytail' in stored.lower()


def test_extra_describe_the_hair_is_stripped_when_hair_is_omit(app, client, monkeypatch):
    from app.services import vision_ollama
    with app.app_context():
        save_config({'captioning': {'backend': 'auto'}})
        ds = svc.create_dataset(LOCAL_USER, 'HairOff', 'zchar_hairoff')
        svc.set_caption_options(LOCAL_USER, ds.id, {
            'backend': 'ollama',
            'appearance': {'hair': 'omit'},
            'instructions': 'Always describe the hair.',
        })
        img = _kept_image(ds, 'k0.png')
        ds_id, img_id = ds.id, img.id

    def fake_describe(image_bytes, prompt, **kwargs):
        return ('Three-quarter shot of the subject standing. She has long hair in a high ponytail. '
                'Soft daylight, denim jacket.')

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)
    client.post(f'/api/dataset/{ds_id}/caption', json={'force': True})
    with app.app_context():
        stored = (db.session.get(FaceDatasetImage, img_id).caption or '')
    assert 'ponytail' not in stored.lower()
    assert 'denim jacket' in stored.lower()
    # Eyes/skin still drop even when hair is the only family they flipped.
    assert 'pale skin' not in fv.drop_identity_sentences(
        'The subject has pale skin. A red coat.', appearance=DEFAULTS)


def test_mascara_survives_default_policy_and_strips_when_makeup_omitted(
        app, client, monkeypatch):
    from app.services import vision_ollama
    with app.app_context():
        save_config({'captioning': {'backend': 'auto'}})
        ds = svc.create_dataset(LOCAL_USER, 'Make', 'zchar_make')
        # First toggle (hair left at default omit) materializes makeup=describe.
        svc.set_caption_options(LOCAL_USER, ds.id, {
            'backend': 'ollama', 'appearance': {'hair': 'omit'},
            'instructions': 'Name the mascara.',
        })
        img = _kept_image(ds, 'k0.png')
        ds_id, img_id = ds.id, img.id

    def fake_describe(image_bytes, prompt, **kwargs):
        return ('Close-up of the subject looking at the viewer. She is wearing mascara. '
                'Soft window light.')

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)
    client.post(f'/api/dataset/{ds_id}/caption', json={'force': True})
    with app.app_context():
        stored = (db.session.get(FaceDatasetImage, img_id).caption or '').lower()
    assert 'mascara' in stored

    with app.app_context():
        svc.set_caption_options(LOCAL_USER, ds_id, {'appearance': {'makeup': 'omit'}})
    client.post(f'/api/dataset/{ds_id}/caption', json={'force': True})
    with app.app_context():
        stored = (db.session.get(FaceDatasetImage, img_id).caption or '').lower()
    assert 'mascara' not in stored
    assert 'window light' in stored


def test_virgin_dataset_prompt_is_the_historical_one(app, client, monkeypatch):
    from app.services import vision_ollama
    with app.app_context():
        save_config({'captioning': {'backend': 'auto'}})
        ds = svc.create_dataset(LOCAL_USER, 'Virg', 'zchar_virg')
        svc.set_caption_options(LOCAL_USER, ds.id, {'backend': 'ollama'})
        _kept_image(ds, 'k0.png')
        ds_id = ds.id

    captured = {}

    def fake_describe(image_bytes, prompt, **kwargs):
        captured['prompt'] = prompt
        return 'a woman standing in a park wearing a red coat'

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)
    client.post(f'/api/dataset/{ds_id}/caption', json={'force': True})
    assert captured['prompt'] == fv.JOYCAPTION_PROMPT


def test_leak_badge_and_preflight_honour_the_policy(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Badge', 'zchar_badge')
        _kept_image(ds, 'a.png', caption='wearing mascara, red dress')
        _kept_image(ds, 'b.png', caption='a woman with long blonde hair, denim jacket')
        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        # Virgin: mascara is NOT a leak, hair is.
        assert payload['caption_leak']['leaking'] == 1
        assert 'hair' in payload['caption_leak']['watched']
        assert 'makeup' not in ' '.join(payload['caption_leak']['watched'])

        svc.set_caption_options(LOCAL_USER, ds.id, {'appearance': {'hair': 'omit'}})
        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        # Default saved policy: makeup describe → mascara not a leak; hair still is.
        assert payload['caption_leak']['leaking'] == 1
        watched = ' '.join(payload['caption_leak']['watched'])
        assert 'hair' in watched and 'makeup' not in watched

        svc.set_caption_options(LOCAL_USER, ds.id, {'appearance': {'makeup': 'omit'}})
        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        assert payload['caption_leak']['leaking'] == 2  # hair + mascara
        assert 'makeup' in ' '.join(payload['caption_leak']['watched'])

        report = lt.training_preflight(LOCAL_USER, ds.id)
        assert len(report['leak_images']) == 2


def test_concise_length_names_describe_families(app, client, monkeypatch):
    from app.services import vision_ollama
    with app.app_context():
        save_config({'captioning': {'backend': 'auto'}})
        ds = svc.create_dataset(LOCAL_USER, 'LenApp', 'zchar_lenapp')
        svc.set_caption_options(LOCAL_USER, ds.id, {
            'backend': 'ollama', 'length': 'concise',
            'appearance': {'hair': 'describe'},
        })
        _kept_image(ds, 'k0.png')
        ds_id = ds.id

    captured = {}

    def fake_describe(image_bytes, prompt, **kwargs):
        captured['prompt'] = prompt
        return 'A woman in a red coat stands on a rainy street.'

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)
    client.post(f'/api/dataset/{ds_id}/caption', json={'force': True})
    assert 'the hairstyle' in captured['prompt']
    assert 'the makeup or lack of it' in captured['prompt']
    # The historical concise sentence is gone so it cannot fight the policy.
    assert 'the clothing and the setting' not in captured['prompt']


def test_invalid_appearance_is_rejected(app, client):
    with app.app_context():
        ds_id = svc.create_dataset(LOCAL_USER, 'BadApp', 'zchar_badapp').id
    r = client.post(f'/api/dataset/{ds_id}/caption/options',
                    json={'appearance': 'hair'})
    assert r.status_code == 400
    r = client.post(f'/api/dataset/{ds_id}/caption/options',
                    json={'appearance': {'nope': 'omit'}})
    assert r.status_code == 400
