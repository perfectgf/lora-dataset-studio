"""STYLE datasets: the medium must never reach the caption (it has to be absorbed
by the LoRA). The prompt already forbids it, but JoyCaption is not an instruction
follower, so the openers survive ("A digital illustration of a young woman…").
This is the post-filter net — the style twin of drop_identity_sentences."""
import io
import json
import os

import pytest
from PIL import Image

from app.config import LOCAL_USER, save_config
from app.extensions import db
from app.models import FaceDatasetImage
from app.services import face_dataset_service as svc
from app.services import face_variations as fv
from app.services import image_bank_service as bank


# --- the reported shapes -------------------------------------------------------

@pytest.mark.parametrize('caption, expected', [
    # "<medium> of <content>" — the shape actually observed on dataset captions
    ('A digital illustration of a young woman with red hair standing in a field.',
     'A young woman with red hair standing in a field.'),
    ('An illustration of a castle on a hill.', 'A castle on a hill.'),
    # "<medium> <verb> <content>" — the verb goes with the medium, not the content
    ('A digital illustration shows five colorful birds perched on a wire.',
     'Five colorful birds perched on a wire.'),
    ('An artwork depicting two men fighting.', 'Two men fighting.'),
    # a lead-in sentence of its own
    ('This is a pencil drawing of a castle beside a lake.',
     'A castle beside a lake.'),
    ('The image is a watercolor painting of a harbour at dawn.',
     'A harbour at dawn.'),
    ('This image shows a 3D render of a red sports car.', 'A red sports car.'),
    ('This is a digital illustration. A young woman sits on a bench.',
     'A young woman sits on a bench.'),
    # article-less and hyphenated variants
    ('Digital art of a wolf howling at the moon.', 'A wolf howling at the moon.'),
    ('An anime-style image of a schoolgirl running down a street.',
     'A schoolgirl running down a street.'),
    ('A comic book drawing of a masked hero on a rooftop.',
     'A masked hero on a rooftop.'),
    ('A sketch of two hands holding a cup.', 'Two hands holding a cup.'),
    ('A photograph of a busy market street.', 'A busy market street.'),
])
def test_lead_in_is_removed_and_grammar_repaired(caption, expected):
    assert fv.drop_style_lead_in(caption) == expected


def test_first_letter_is_capitalised_after_repair():
    out = fv.drop_style_lead_in('A painting of the old lighthouse at night.')
    assert out.startswith('The old lighthouse')


# --- what must SURVIVE ---------------------------------------------------------

@pytest.mark.parametrize('caption', [
    # a medium named as an OBJECT inside the scene is content, not a lead-in
    'A painting hangs on the wall above a leather sofa.',
    'An oil painting of a stormy sea hangs on the wall behind the desk.',
    'Two women stand in front of a mural, one holding a sketchbook.',
    'A young woman with red hair stands in a field at sunset.',
    # mid-text mentions are untouched: only the opener is a lead-in
    'A man reads by the window while a framed illustration of a bird sits nearby.',
    # a DEFINITE article points at something already in the scene, not at the medium
    'The artwork of the poster behind her reads FESTIVAL.',
    'The illustration of a bird on the cover is faded.',
    # nothing left to keep -> the caption is returned as-is rather than emptied
    'A digital illustration.',
])
def test_content_is_left_alone(caption):
    assert fv.drop_style_lead_in(caption) == caption


def test_booru_tag_lists_are_a_no_op():
    # Style datasets on SDXL caption in booru mode: an underscored tag list has no
    # prose lead-in to cut, and this filter must not invent one.
    tags = 'digital_art, 1girl, red_hair, standing, outdoors, sunset'
    assert fv.drop_style_lead_in(tags) == tags


def test_empty_and_none_are_safe():
    assert fv.drop_style_lead_in(None) == ''
    assert fv.drop_style_lead_in('   ') == ''


def test_filter_is_idempotent():
    once = fv.drop_style_lead_in('A digital illustration of a young woman in a field.')
    assert fv.drop_style_lead_in(once) == once


# --- the lexicon is TIED to the bank's medium prototypes -----------------------
# The bank already owns a medium vocabulary (MEDIUM_PROTOTYPES, a CLIP classifier
# tuned against measured margins — it must NOT be edited to serve a text filter).
# This test is the tie instead: every medium head noun the bank names has to be a
# head this filter recognises, so the two never drift apart.


# --- the style captioning lane actually USES it --------------------------------
# The filter existing is worth nothing if the lane still cleans with `lambda t: t`,
# which is exactly the bug: these two drive the real pipeline with the vision engine
# stubbed, and go red if the wiring is reverted.

def _style_dataset_with_image():
    ds = svc.create_dataset(LOCAL_USER, 'Look', 'zstyle_look', kind='style')
    img_dir = svc._dataset_dir(ds.id)
    os.makedirs(img_dir, exist_ok=True)
    Image.new('RGB', (32, 32), (120, 40, 40)).save(os.path.join(img_dir, 'k0.png'))
    row = FaceDatasetImage(dataset_id=ds.id, status='keep', filename='k0.png')
    db.session.add(row)
    db.session.commit()
    return ds, row


def test_long_caption_lane_strips_the_lead_in(app, monkeypatch):
    from app.services import vision_ollama
    with app.app_context():
        save_config({'captioning': {'backend': 'ollama'}})
        ds, img = _style_dataset_with_image()
        monkeypatch.setattr(
            vision_ollama, 'describe_image_ollama',
            lambda image_bytes, prompt, **kw:
                'A digital illustration of a young woman standing in a field.')
        monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: None)

        assert svc.caption_images(LOCAL_USER, ds.id) == 1
        db.session.refresh(img)
        assert img.caption == 'A young woman standing in a field.'


def test_short_caption_lane_strips_the_lead_in(app):
    with app.app_context():
        ds, img = _style_dataset_with_image()
        ds.train_settings = json.dumps({'dual_captions': True})
        img.caption = 'A young woman standing in a field.'
        db.session.commit()

        n = svc.derive_short_captions(
            LOCAL_USER, ds.id,
            generate=lambda p: 'A watercolor painting of a woman in a field')
        assert n == 1
        assert db.session.get(FaceDatasetImage, img.id).caption_short == \
            'A woman in a field'


# A screenshot is deliberately NOT a medium here: "a screenshot of a chat window" is
# what the picture IS OF, i.e. content — and the bank agrees, since its screenshot
# phrases live in the '_screen' DISTRACTOR bucket, not in a medium bucket.
_NOT_A_MEDIUM = {'a videogame screenshot'}


def test_every_bank_medium_word_is_known_to_the_filter():
    vocab = set(fv.STYLE_MEDIUM_HEADS) | set(fv.STYLE_MEDIUM_MODS)
    unknown = []
    for bucket, phrases in bank.MEDIUM_PROTOTYPES.items():
        if bucket.startswith('_'):      # distractors: text banners, screenshots
            continue
        for phrase in phrases:
            if phrase in _NOT_A_MEDIUM:
                continue
            words = {w for w in phrase.lower().replace('-', ' ').split() if w}
            if not (words & vocab):
                unknown.append(phrase)
    assert unknown == []
