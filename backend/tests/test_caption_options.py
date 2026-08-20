"""Per-dataset caption method options (Captions ⚙️ Options popover).

Covers the whole seam:
  - additive caption_options migration on a LEGACY table (no column yet);
  - caption_options() normalization + set_caption_options() round-trip/validation;
  - caption_images honoring the per-dataset engine override, appending the extra
    instructions to the prompt, and passing the chosen Ollama model — all captured
    at the describe seam;
  - the GET/POST /caption/options routes incl. the invalid-engine 400.

Single-user extraction (LOCAL_USER). The vision seam is imported locally by the
pipeline, so it is patched at app.services.vision_ollama.*.
"""
import json
import os

import pytest
from PIL import Image
from sqlalchemy import text

from app.extensions import db
from app.models import FaceDataset, FaceDatasetImage
from app.services import face_dataset_service as svc
from app.config import LOCAL_USER, save_config


def _kept_image(ds, fn, caption=None):
    img_dir = svc._dataset_dir(ds.id)
    os.makedirs(img_dir, exist_ok=True)
    Image.new('RGB', (32, 32)).save(os.path.join(img_dir, fn))
    row = FaceDatasetImage(dataset_id=ds.id, status='keep', filename=fn, caption=caption)
    db.session.add(row)
    db.session.commit()
    return row


# --- normalization -----------------------------------------------------------
def test_caption_options_defaults_and_normalization(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        # Never set → all-empty (follow the global defaults).
        assert svc.caption_options(ds) == {
            'backend': '', 'ollama_model': '', 'instructions': '', 'vocabulary': '',
            'length': '', 'appearance': {}}

        # A corrupt blob degrades to defaults, never raises.
        ds.caption_options = '{not json'
        db.session.commit()
        assert svc.caption_options(ds)['backend'] == ''

        # An unknown engine/vocab is dropped; a good one + model + instructions survive.
        ds.caption_options = json.dumps(
            {'backend': 'nope', 'ollama_model': '  m:latest ', 'instructions': '  hi  ',
             'vocabulary': 'spicy'})
        db.session.commit()
        opts = svc.caption_options(ds)
        assert opts['backend'] == '' and opts['ollama_model'] == 'm:latest'
        assert opts['instructions'] == 'hi' and opts['vocabulary'] == ''

        # A known vocabulary is normalized (case/space-insensitive).
        ds.caption_options = json.dumps({'vocabulary': '  Explicit '})
        db.session.commit()
        assert svc.caption_options(ds)['vocabulary'] == 'explicit'

        # Instructions are length-capped.
        ds.caption_options = json.dumps({'instructions': 'x' * 5000})
        db.session.commit()
        assert len(svc.caption_options(ds)['instructions']) == svc._CAPTION_INSTRUCTIONS_MAX


def test_caption_options_defensively_drop_invalid_stored_models(app):
    """Legacy/manual JSON blobs cannot bypass the current model-ref validator."""
    invalid = (None, 123, False, [], {}, 'bad model!', '/leading',
               'valid:tag\n', 'valid:tag\r\nsecond:tag', 'valid:tag\u2028',
               'a' * 201)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'LegacyModels', 'zchar_legacy_models')
        for model in invalid:
            ds.caption_options = json.dumps({
                'ollama_model': model,
                'instructions': 'keep this valid option',
            })
            db.session.commit()
            opts = svc.caption_options(ds)
            assert opts['ollama_model'] == ''
            assert opts['instructions'] == 'keep this valid option'


def test_set_caption_options_roundtrip_and_validation(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        eff = svc.set_caption_options(LOCAL_USER, ds.id,
                                      {'backend': 'ollama', 'ollama_model': 'q:8b',
                                       'instructions': 'name the season', 'vocabulary': 'explicit'})
        assert eff == {'backend': 'ollama', 'ollama_model': 'q:8b',
                       'instructions': 'name the season', 'vocabulary': 'explicit',
                       'length': '', 'appearance': {}}
        stored = json.loads(db.session.get(FaceDataset, ds.id).caption_options)
        assert stored == {'backend': 'ollama', 'ollama_model': 'q:8b',
                          'instructions': 'name the season', 'vocabulary': 'explicit'}

        # A partial patch changes only the named key.
        svc.set_caption_options(LOCAL_USER, ds.id, {'instructions': 'name the outfit'})
        assert svc.caption_options(db.session.get(FaceDataset, ds.id)) == {
            'backend': 'ollama', 'ollama_model': 'q:8b', 'instructions': 'name the outfit',
            'vocabulary': 'explicit', 'length': '', 'appearance': {}}

        # Clearing every field stores NULL (identical to never-touched).
        svc.set_caption_options(LOCAL_USER, ds.id,
                                {'backend': '', 'ollama_model': '', 'instructions': '',
                                 'vocabulary': '', 'length': ''})
        assert db.session.get(FaceDataset, ds.id).caption_options is None

        # Invalid engine → ValueError (mapped 400 by the route).
        try:
            svc.set_caption_options(LOCAL_USER, ds.id, {'backend': 'gpt5'})
            assert False, 'expected ValueError'
        except ValueError as e:
            assert 'invalid captioning backend' in str(e)

        # Invalid vocabulary → ValueError too.
        try:
            svc.set_caption_options(LOCAL_USER, ds.id, {'vocabulary': 'filthy'})
            assert False, 'expected ValueError'
        except ValueError as e:
            assert 'invalid caption vocabulary' in str(e)


def test_set_caption_options_validates_ollama_model_type_format_and_length(app):
    valid = (
        ('qwen3-vl:8b', 'qwen3-vl:8b'),
        (' registry.example:5000/team/model_name-1.2:latest ',
         'registry.example:5000/team/model_name-1.2:latest'),
        ('a' * 200, 'a' * 200),
        ('', ''),
    )
    invalid = (None, 123, False, [], {}, 'bad model!', '/leading',
               'valid:tag\n', 'valid:tag\r\nsecond:tag', 'valid:tag\u2028',
               'a' * 201)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'ValidatedModels', 'zchar_validated_models')
        for raw, expected in valid:
            opts = svc.set_caption_options(
                LOCAL_USER, ds.id, {'ollama_model': raw})
            assert opts['ollama_model'] == expected
        for raw in invalid:
            with pytest.raises(ValueError, match='ollama_model'):
                svc.set_caption_options(
                    LOCAL_USER, ds.id, {'ollama_model': raw})


# --- additive migration on a legacy table ------------------------------------
def test_caption_options_migration_on_legacy_table(app):
    with app.app_context():
        from app import _apply_additive_migrations
        db.session.execute(text('DROP TABLE face_dataset'))
        db.session.execute(text(
            'CREATE TABLE face_dataset ('
            'id INTEGER PRIMARY KEY, user_id TEXT, name TEXT, base_tag TEXT)'))
        db.session.execute(text(
            "INSERT INTO face_dataset (id, user_id, name, base_tag) "
            "VALUES (1, 'local', 'Legacy', 'zchar_x')"))
        db.session.commit()
        cols = {r[1] for r in db.session.execute(text('PRAGMA table_info(face_dataset)'))}
        assert 'caption_options' not in cols

        _apply_additive_migrations()

        cols = {r[1] for r in db.session.execute(text('PRAGMA table_info(face_dataset)'))}
        assert 'caption_options' in cols
        _apply_additive_migrations()  # idempotent


# --- caption_images honors the overrides at the describe seam -----------------
def test_caption_run_uses_engine_model_and_instructions(app, client, monkeypatch):
    from app.services import vision_ollama
    with app.app_context():
        save_config({'captioning': {'backend': 'auto'}})   # global default differs from override
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        svc.set_caption_options(LOCAL_USER, ds.id,
                                {'backend': 'ollama',            # override the global 'auto'
                                 'ollama_model': 'custom-vlm:latest',
                                 'instructions': 'Always name the visible clothing colors.'})
        img = _kept_image(ds, 'k0.png')
        ds_id, img_id = ds.id, img.id

    captured = {}

    def fake_describe(image_bytes, prompt, **kwargs):
        captured['prompt'] = prompt
        captured['model'] = kwargs.get('model')
        return 'a woman standing in a park wearing a red coat'

    # backend='ollama' → JoyCaption is never tried, so only this seam runs.
    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)

    r = client.post(f'/api/dataset/{ds_id}/caption', json={'force': True})
    assert r.status_code == 200 and r.get_json()['captioned'] == 1

    # The chosen model was passed and the extra instruction was appended to the prompt
    # (the base kind-omission prompt still fronts it).
    assert captured['model'] == 'custom-vlm:latest'
    assert captured['prompt'].endswith('Always name the visible clothing colors.')
    assert 'Additional instructions from the user:' in captured['prompt']
    assert captured['prompt'].index('Additional instructions') > 50  # base prompt came first

    with app.app_context():
        assert (db.session.get(FaceDatasetImage, img_id).caption or '').strip()


# --- NSFW vocabulary preset: explicit terms reach the prompt AND survive cleaners ------
def test_explicit_vocabulary_reaches_prompt_and_survives_cleaners(app, client, monkeypatch):
    """Proof for the NSFW lane: the 'explicit' preset instruction is appended to the caption
    prompt, and a crude caption the (abliterated) model returns keeps its explicit vocabulary
    — the identity/concept leak cleaners scrub identity, never anatomical/sexual words."""
    from app.services import vision_ollama
    with app.app_context():
        save_config({'captioning': {'backend': 'auto'}})
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        svc.set_caption_options(LOCAL_USER, ds.id,
                                {'backend': 'ollama', 'vocabulary': 'explicit'})
        img = _kept_image(ds, 'k0.png')
        ds_id, img_id = ds.id, img.id

    captured = {}
    # A deliberately crude caption — the words we assert survive are anatomical/sexual,
    # NOT identity (hair/skin/eyes) or a per-dataset concept term.
    explicit_caption = 'A nude woman on her knees with an erect penis near her open mouth.'

    def fake_describe(image_bytes, prompt, **kwargs):
        captured['prompt'] = prompt
        return explicit_caption

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)

    r = client.post(f'/api/dataset/{ds_id}/caption', json={'force': True})
    assert r.status_code == 200 and r.get_json()['captioned'] == 1

    # The explicit preset instruction rode into the prompt, after the base kind prompt.
    assert 'do not censor' in captured['prompt'].lower()
    assert 'crude' in captured['prompt'].lower()
    assert 'Additional instructions from the user:' in captured['prompt']

    # The crude vocabulary is intact in the stored caption — nothing filtered it.
    with app.app_context():
        stored = (db.session.get(FaceDatasetImage, img_id).caption or '').lower()
    assert 'penis' in stored and 'nude' in stored and 'erect' in stored


# --- routes ------------------------------------------------------------------
def test_caption_options_routes(app, client):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        ds_id = ds.id

    r = client.get(f'/api/dataset/{ds_id}/caption/options')
    assert r.status_code == 200
    assert r.get_json()['options'] == {
        'backend': '', 'ollama_model': '', 'instructions': '', 'vocabulary': '',
        'length': '', 'appearance': {}}

    r = client.post(f'/api/dataset/{ds_id}/caption/options',
                    json={'backend': 'joycaption', 'instructions': 'be terse'})
    assert r.status_code == 200
    assert r.get_json()['options']['backend'] == 'joycaption'

    r = client.post(f'/api/dataset/{ds_id}/caption/options', json={'backend': 'bogus'})
    assert r.status_code == 400

    r = client.get('/api/dataset/999999/caption/options')
    assert r.status_code == 404


def test_caption_options_route_rejects_invalid_ollama_models(app, client):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'RouteModels', 'zchar_route_models')
        ds_id = ds.id

    invalid = (None, 123, False, [], {}, 'bad model!', '/leading',
               'valid:tag\n', 'valid:tag\r\nsecond:tag', 'valid:tag\u2028',
               'a' * 201)
    for model in invalid:
        r = client.post(
            f'/api/dataset/{ds_id}/caption/options',
            json={'ollama_model': model})
        assert r.status_code == 400, (model, r.get_json())
        assert 'ollama_model' in r.get_json()['error']

    r = client.post(
        f'/api/dataset/{ds_id}/caption/options',
        json={'ollama_model': 'a' * 200})
    assert r.status_code == 200
    assert r.get_json()['options']['ollama_model'] == 'a' * 200
    r = client.post(
        f'/api/dataset/{ds_id}/caption/options',
        json={'ollama_model': ''})
    assert r.status_code == 200
    assert r.get_json()['options']['ollama_model'] == ''


# --- caption LENGTH preset (idea by djpraxis, Reddit) -------------------------
def test_caption_length_normalization_and_validation(app):
    """Same contract as the vocabulary register: unknown values are dropped on read,
    refused on write, and '' means "standard" (nothing appended)."""
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Len', 'zchar_len')
        ds.caption_options = json.dumps({'length': 'tiny'})
        db.session.commit()
        assert svc.caption_options(ds)['length'] == ''

        ds.caption_options = json.dumps({'length': '  Concise '})
        db.session.commit()
        assert svc.caption_options(ds)['length'] == 'concise'

        assert svc.set_caption_options(
            LOCAL_USER, ds.id, {'length': 'detailed'})['length'] == 'detailed'
        # Only the named key changed - the round-trip stays additive.
        assert svc.caption_options(db.session.get(FaceDataset, ds.id))['vocabulary'] == ''
        with pytest.raises(ValueError, match='invalid caption length'):
            svc.set_caption_options(LOCAL_USER, ds.id, {'length': 'epic'})

        # Cleared back to standard -> the key is not stored at all.
        svc.set_caption_options(LOCAL_USER, ds.id, {'length': ''})
        assert db.session.get(FaceDataset, ds.id).caption_options is None


def test_caption_length_route_rejects_unknown_value(app, client):
    with app.app_context():
        ds_id = svc.create_dataset(LOCAL_USER, 'LenRoute', 'zchar_len_route').id
    r = client.post(f'/api/dataset/{ds_id}/caption/options', json={'length': 'epic'})
    assert r.status_code == 400 and 'length' in r.get_json()['error']
    r = client.post(f'/api/dataset/{ds_id}/caption/options', json={'length': 'concise'})
    assert r.status_code == 200 and r.get_json()['options']['length'] == 'concise'


def test_length_preset_rides_after_vocabulary_and_before_free_instructions(
        app, client, monkeypatch):
    """The documented composition order - base prompt, then vocabulary register, then
    length preset, then the user's free text LAST so it can override a preset."""
    from app.services import vision_ollama
    with app.app_context():
        save_config({'captioning': {'backend': 'auto'}})
        ds = svc.create_dataset(LOCAL_USER, 'Order', 'zchar_order')
        svc.set_caption_options(LOCAL_USER, ds.id,
                                {'backend': 'ollama', 'vocabulary': 'clinical',
                                 'length': 'concise',
                                 'instructions': 'MY OWN STEER'})
        _kept_image(ds, 'k0.png')
        ds_id = ds.id

    captured = {}

    def fake_describe(image_bytes, prompt, **kwargs):
        captured['prompt'] = prompt
        return 'a woman standing in a park wearing a red coat'

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)

    r = client.post(f'/api/dataset/{ds_id}/caption', json={'force': True})
    assert r.status_code == 200 and r.get_json()['captioned'] == 1

    prompt = captured['prompt']
    i_vocab = prompt.index('clinical, anatomical terms')
    i_len = prompt.index('Keep the caption SHORT')
    i_free = prompt.index('MY OWN STEER')
    assert prompt.index('Additional instructions from the user:') < i_vocab < i_len < i_free
    assert prompt.rstrip().endswith('MY OWN STEER')


def test_standard_length_leaves_the_prompt_byte_identical(app, client, monkeypatch):
    """Retro-compat: length='' appends nothing at all - a dataset that never touched
    the dial builds exactly the prompt it built before this option existed."""
    from app.services import vision_ollama
    with app.app_context():
        save_config({'captioning': {'backend': 'auto'}})
        ds = svc.create_dataset(LOCAL_USER, 'Std', 'zchar_std')
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
    assert 'Additional instructions from the user:' not in captured['prompt']


def test_concise_preset_keeps_captions_on_the_prose_side_of_the_mismatch_guard():
    """A Concise caption must stay PROSE. caption_style() votes 'booru' on three-plus
    short comma segments with no sentence punctuation, and assert_trainable refuses a
    prose family on booru captions (MISMATCH_CAPTION) - so a length preset phrased as
    "a few key comma-separated phrases" would have made every Concise dataset need a
    forced launch. The shipped instruction asks for one COMPLETE SENTENCE and forbids
    a tag list; this pins both halves."""
    from app.services.face_variations import caption_style
    text = svc._LENGTH_INSTRUCTION['concise']
    assert 'complete sentence' in text
    assert 'never a comma-separated list of tags' in text
    # A caption written to that instruction reads as prose, even with commas in it.
    concise = ('A woman in a red coat stands on a rainy street at night, '
               'looking away from the camera.')
    assert caption_style(concise) == 'prose'
    # The Detailed preset is prose by construction (several sentences).
    assert caption_style(
        'A woman sits at a cafe table. She wears a denim jacket. '
        'Warm afternoon light comes from the left. The shot is a waist-up view.') == 'prose'
