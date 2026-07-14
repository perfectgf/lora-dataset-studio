"""Concept-leak detector + honest badge (kind=concept) — the leg_behind incident.

Real incident: a concept dataset (concept_desc "leg behind head position", trigger
`leg_behind`) whose captions ALL described the pose ("legs extended vertically upward",
"knees lifted high", "feet resting on her thighs") -> the concept bound to the WORDS, not
the trigger -> the LoRA had no effect. Two silent failures fixed here:

  1. the ban-list was the 4 words of concept_desc; the pose PERIPHRASES the captioner
     reaches for ("knees/feet/thighs/lifted") were never listed, so the omission net had
     nothing to catch. `concept_lexical_field` derives them from the description.
  2. the aggregate badge FORCED leaking=0 for concept datasets -> a false "0 leak". The
     payload now runs a real concept-leak count.

Pure detector lives beside caption_has_identity_leak in face_variations; the pipeline
retry/drop is exercised with the vision seam mocked (app.services.vision_ollama.*).
"""
import io

import pytest
from PIL import Image

from app.extensions import db
from app.models import FaceDatasetImage
from app.services import face_variations as fv
from app.services import face_dataset_service as svc
from app.config import LOCAL_USER, save_config

LEG = 'leg behind head position'

# The RAW captions from the incident (must all be flagged for the leg concept).
INCIDENT_LEAKS = [
    'her legs extended vertically upward, feet pointing toward the ceiling',
    'knees lifted high toward her chest',
    'feet resting on her thighs',
    'she is contorted with one foot tucked behind her neck',
]
# Clean captions for the SAME set (describe person/scene, not the pose) -> 0.
CLEAN = [
    'a woman with red hair sitting on a bed in a bright room, smiling softly',
    'wearing a sheer top, looking at the viewer, warm window light, indoor bedroom',
]


# --- 1) pure detector: incident captions leak, clean captions do not ----------

@pytest.mark.parametrize('cap', INCIDENT_LEAKS)
def test_incident_captions_leak_for_leg_concept(cap):
    assert fv.caption_has_concept_leak(cap, LEG) is True, cap


@pytest.mark.parametrize('cap', CLEAN)
def test_clean_captions_do_not_leak_for_leg_concept(cap):
    assert fv.caption_has_concept_leak(cap, LEG) is False, cap


def test_detector_returns_the_offending_terms():
    leaks = fv.caption_concept_leaks('knees lifted high toward her chest', LEG)
    assert 'knees' in leaks and 'lifted' in leaks


def test_empty_caption_is_not_a_leak():
    assert fv.caption_has_concept_leak('', LEG) is False
    assert fv.caption_has_concept_leak(None, LEG) is False


# --- 2) lexicon derivation is SCOPED, not sprayed onto every concept ----------

def test_lexical_field_derived_from_leg_concept():
    field = set(fv.concept_lexical_field(LEG))
    # nouns of the anchored family + posture verbs
    assert {'legs', 'knees', 'feet', 'thighs'} <= field
    assert {'lifted', 'raised', 'extended', 'bent'} <= field


def test_photographic_concept_gets_no_leg_words():
    # "a candid mirror selfie" anchors NO body family -> empty field, and a pose caption
    # does NOT leak for it (the leg lexicon is not hard-coded for all concepts).
    assert fv.concept_lexical_field('a candid mirror selfie') == []
    assert fv.caption_has_concept_leak('her knees are lifted and feet raised',
                                       'a candid mirror selfie') is False


def test_singular_plural_and_cached_terms_are_covered():
    # concept_terms (cached LLM ban-list, here a JSON string as stored on the row) unions in.
    terms = fv.concept_leak_terms(LEG, '["foot behind head", "backbend"]')
    assert 'backbend' in terms
    # singular/plural tolerated by the regex suffix
    assert fv.caption_has_concept_leak('a single knee raised', LEG) is True
    assert fv.caption_has_concept_leak('both knees raised', LEG) is True


# --- 3) generation prompt: dynamic omission clause ----------------------------

def test_concept_prompt_gains_pose_omission_clause():
    p = fv.caption_prompt_for_concept(LEG)
    assert 'do NOT describe the position or arrangement of the legs' in p
    assert 'captured by the trigger word' in p


def test_concept_prompt_unchanged_for_non_body_concept():
    # byte-identical to the historical prompt -> no behaviour change for e.g. "ice cream"
    base = fv.CAPTION_PROMPT_CONCEPT.format(concept='licking ice cream')
    assert fv.caption_prompt_for_concept('licking ice cream') == base


# --- 4) drop_concept_sentences: the concept twin of drop_identity_sentences ----

def test_drop_concept_sentences_removes_pose_sentence_keeps_rest():
    cap = ('A woman with red hair sits on a bed. Her knees are lifted high and her feet '
           'rest on her thighs. Warm window light fills the room.')
    out = fv.drop_concept_sentences(cap, LEG)
    assert 'red hair' in out and 'window light' in out
    for w in ('knees', 'feet', 'thighs', 'lifted'):
        assert w not in out.lower()


# --- 5) aggregation: concept NO LONGER forced to 0; style N/A; character kept --

def _add(dataset_id, caption, status='keep'):
    db.session.add(FaceDatasetImage(
        dataset_id=dataset_id, filename=f'{status}_{abs(hash(caption)) % 99999}.webp',
        status=status, caption=caption))
    db.session.commit()


def test_concept_badge_counts_real_leaks(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Leg', 'leg_behind', kind='concept',
                                concept_desc=LEG)
        _add(ds.id, INCIDENT_LEAKS[0])   # leaks (legs/feet/extended)
        _add(ds.id, INCIDENT_LEAKS[1])   # leaks (knees/lifted)
        _add(ds.id, CLEAN[0])            # clean
        _add(ds.id, '', status='keep')   # no caption -> not counted
        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        # the false "0 leak" is gone: 2 of 3 captions name the pose
        assert payload['caption_leak'] == {'leaking': 2, 'captioned': 3}
        flagged = [i for i in payload['images'] if i['leak']]
        assert len(flagged) == 2


def test_concept_badge_zero_when_captions_are_clean(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Leg2', 'leg_behind2', kind='concept',
                                concept_desc=LEG)
        for c in CLEAN:
            _add(ds.id, c)
        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        # a REAL 0 on 2 checked captions (not a forced 0)
        assert payload['caption_leak'] == {'leaking': 0, 'captioned': 2}
        assert all(i['leak'] is False for i in payload['images'])


def test_style_dataset_leak_not_applicable(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Sty', 'sty', kind='style')
        _add(ds.id, 'a woman with long blonde hair and blue eyes, pale skin')  # identity is CONTENT
        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        assert payload['caption_leak']['leaking'] == 0
        assert all(i['leak'] is False for i in payload['images'])


def test_character_dataset_still_flags_identity(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Char', 'char_cv')  # character (default)
        _add(ds.id, 'a woman with long blonde hair and blue eyes')  # identity leak
        _add(ds.id, 'standing in a field, green jacket, neutral gaze')  # clean
        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        assert payload['caption_leak'] == {'leaking': 1, 'captioned': 2}


# --- 6) pipeline post-processing: enriched ban-list makes retry/drop catch pose -

def _png(w=512, h=512):
    b = io.BytesIO()
    Image.new('RGB', (w, h), (120, 40, 40)).save(b, 'PNG')
    return b.getvalue()


def _leg_concept_with_image():
    ds = svc.create_dataset(LOCAL_USER, 'LegC', 'leg_behind_c', kind='concept',
                            concept_desc=LEG)
    ids, _ = svc.import_images(LOCAL_USER, ds.id, [_png()], crop=False)
    return ds, db.session.get(FaceDatasetImage, ids[0])


def test_pipeline_llm_fix_scrubs_pose_periphrases(app, monkeypatch):
    """A direct caption that DESCRIBES the pose is detected via the enriched ban-list and
    rewritten by the corrective LLM to a clean caption."""
    from app.services import vision_ollama
    with app.app_context():
        save_config({'captioning': {'backend': 'ollama'}})   # no Joy; direct-Qwen + enforce
        ds, img = _leg_concept_with_image()

        def fake_describe(image_bytes, prompt, **kw):
            if 'BLOCKLIST' in prompt:
                return 'not json -> desc + lexical-field fallback ban-list'
            if 'forbidden words' in prompt:      # corrective rewrite -> clean
                return 'A woman with red hair reclines on a bed in a sunlit room.'
            # the direct concept caption -> DESCRIBES the pose (the incident failure)
            return ('A woman with red hair on a bed, her knees lifted high and her feet '
                    'resting on her thighs.')

        monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
        monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda: None)
        n = svc.caption_images(LOCAL_USER, ds.id)

        assert n == 1
        db.session.refresh(img)
        cap = (img.caption or '').lower()
        for banned in ('knees', 'feet', 'thighs', 'lifted'):
            assert banned not in cap
        assert 'red hair' in cap


def test_pipeline_mechanical_scrub_when_llm_fix_still_leaks(app, monkeypatch):
    """If the corrective LLM keeps naming the pose, the mechanical clause-scrub is the net
    (now armed with the pose periphrases)."""
    from app.services import vision_ollama
    with app.app_context():
        save_config({'captioning': {'backend': 'ollama'}})
        ds, img = _leg_concept_with_image()

        def fake_describe(image_bytes, prompt, **kw):
            if 'BLOCKLIST' in prompt:
                return 'no json'
            if 'forbidden words' in prompt:      # LLM fix STILL leaks -> scrub must save it
                return 'A woman with red hair on a bed, knees still lifted high.'
            return 'A woman with red hair on a bed, her feet resting on her thighs.'

        monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
        monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda: None)
        svc.caption_images(LOCAL_USER, ds.id)

        db.session.refresh(img)
        cap = (img.caption or '').lower()
        for banned in ('knees', 'feet', 'thighs', 'lifted'):
            assert banned not in cap
        assert 'red hair' in cap  # a clean clause survived


def test_pipeline_joycaption_backend_scrubs_pose_without_ollama(app, monkeypatch):
    """backend='joycaption' (no Ollama): the Joy draft's pose words are still scrubbed by
    the mechanical net built from the derived lexical field."""
    from app.services import vision_ollama
    import app.services.joycaption as jc
    with app.app_context():
        save_config({'captioning': {'backend': 'joycaption'}})
        ds, img = _leg_concept_with_image()
        monkeypatch.setattr(jc, 'is_available', lambda: True)
        monkeypatch.setattr(jc, 'caption_images_joycaption',
                            lambda paths, prompt=None: {
                                p: 'A woman with red hair on a bed, knees lifted, feet on her thighs.'
                                for p in paths})

        def boom(*a, **k):
            raise AssertionError("backend='joycaption' must not call Ollama")

        monkeypatch.setattr(vision_ollama, 'describe_image_ollama', boom)
        monkeypatch.setattr(vision_ollama, 'unload_vision_model', boom)
        n = svc.caption_images(LOCAL_USER, ds.id)

        assert n == 1
        db.session.refresh(img)
        cap = (img.caption or '').lower()
        for banned in ('knees', 'feet', 'thighs', 'lifted'):
            assert banned not in cap
        assert 'red hair' in cap
