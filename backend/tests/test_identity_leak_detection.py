"""Guard tests for the identity-leak detector and the caption_leak badge.

Owner report: "no matter what, I get 0 leak" on a character dataset. These tests
pin the detector against BAIT captions (one per category: hair / eyes / skin /
face shape / facial features) so a regression that silently returns 0 is caught,
and pin the payload badge wiring (counts only kept+captioned, character-only,
per-image flag) so the badge can never read a false 0.
"""
import pytest

from app.services import face_variations as fv
from app.services import face_dataset_service as svc
from app.config import LOCAL_USER


# --- detector: every watched category must fire on an obvious leak -------------

@pytest.mark.parametrize('caption', [
    'a woman with long blonde hair',                       # hair
    'close-up, blue eyes looking at the viewer',           # eye colour
    'the subject has pale skin',                           # skin
    'olive complexion, soft studio light',                 # complexion
    'a light dusting of freckles across the nose',         # freckles
    'a strong, defined jawline',                           # jawline
    'thick dark eyebrows',                                 # eyebrows
    'delicate facial features',                            # facial features
    'a round face turned to the side',                     # face shape
])
def test_each_identity_category_is_flagged(caption):
    assert fv.caption_has_identity_leak(caption) is True, caption


def test_task_bait_caption_flags():
    # The exact bait from the investigation brief — every clause is a leak.
    cap = 'a woman with long blonde hair and blue eyes, pale skin, freckles'
    assert fv.caption_has_identity_leak(cap) is True


# --- detector: clean captions and expression/light NON-leaks stay 0 -----------

@pytest.mark.parametrize('caption', [
    'Three-quarter shot of the subject standing with hands on hips, white crop top, '
    'looking at the viewer with a neutral expression, outdoor cafe setting.',
    'Full-body shot walking across a crosswalk, mid-stride, denim jacket, soft daylight.',
    'the subject sits with eyes closed, warm shadow on the face',   # expression + light, NOT identity
    'looking away, dark room, dramatic lighting',
])
def test_clean_and_expression_captions_do_not_flag(caption):
    assert fv.caption_has_identity_leak(caption) is False, caption


def test_empty_caption_is_not_a_leak():
    assert fv.caption_has_identity_leak('') is False
    assert fv.caption_has_identity_leak(None) is False


# --- payload badge: counts, character-only gating, per-image flag -------------

def _add(dataset_id, caption, status='keep'):
    from app.models import FaceDatasetImage
    svc.db.session.add(FaceDatasetImage(
        dataset_id=dataset_id, filename=f'{status}_{abs(hash(caption)) % 99999}.webp',
        status=status, caption=caption))
    svc.db.session.commit()


def test_badge_counts_only_kept_captioned_and_flags_offenders(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Char', 'char_cv')          # character (default)
        _add(ds.id, 'standing in a field, green jacket, neutral gaze')  # clean, keep
        _add(ds.id, 'a woman with long blonde hair and blue eyes')      # LEAK, keep
        _add(ds.id, 'freckles and pale skin', status='reject')          # LEAK but rejected -> ignored
        _add(ds.id, '', status='keep')                                  # kept, no caption -> not counted

        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        assert payload['caption_leak'] == {'leaking': 1, 'captioned': 2}
        # per-image flag: exactly the kept leaking caption carries leak=True
        flagged = [i for i in payload['images'] if i['leak']]
        assert len(flagged) == 1
        assert 'hair' in flagged[0]['caption']


def test_badge_is_zero_when_captions_are_clean(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Clean', 'clean_cv')
        _add(ds.id, 'three-quarter shot, red dress, sitting, soft window light')
        _add(ds.id, 'full body, walking, denim jacket, city street')
        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        # A REAL 0 on 2 checked captions — the state the owner sees on a clean set.
        assert payload['caption_leak'] == {'leaking': 0, 'captioned': 2}
        assert all(i['leak'] is False for i in payload['images'])


def test_concept_dataset_never_flags_identity(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Cpt', 'cpt', kind='concept',
                                concept_desc='a specific pose')
        _add(ds.id, 'a woman with long blonde hair and blue eyes')      # identity is WANTED here
        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        assert payload['caption_leak']['leaking'] == 0
        assert all(i['leak'] is False for i in payload['images'])
