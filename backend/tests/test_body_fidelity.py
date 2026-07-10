"""Body-fidelity mode: permanent body marks are identity — banned from captions
(prompt suffix + post-filters + leak badge) so they bind to the trigger."""
import pytest

from app.services import face_variations as fv


# --- prompt selection ----------------------------------------------------------

def test_caption_prompt_for_appends_body_ban():
    assert fv.caption_prompt_for('prose') == fv.JOYCAPTION_PROMPT
    assert fv.caption_prompt_for('booru') == fv.CAPTION_PROMPT_BOORU
    p = fv.caption_prompt_for('prose', body=True)
    assert p.startswith(fv.JOYCAPTION_PROMPT) and 'tattoo' in p and 'BODY-FIDELITY' in p
    b = fv.caption_prompt_for('booru', body=True)
    assert b.startswith(fv.CAPTION_PROMPT_BOORU) and 'tattoo' in b


# --- leak detection ------------------------------------------------------------

def test_body_marks_leak_only_in_body_mode():
    cap = 'a full-body shot, a small tattoo on the left shoulder, red dress'
    assert fv.caption_has_identity_leak(cap) is False           # face mode: fine
    assert fv.caption_has_identity_leak(cap, body=True) is True # body mode: leak
    # face traits still leak in both modes
    assert fv.caption_has_identity_leak('long hair falls loose', body=True) is True


def test_drop_identity_sentences_body_mode():
    cap = ('Full-body shot of the subject standing. A dragon tattoo covers the arm. '
           'Warm evening light.')
    assert 'tattoo' in fv.drop_identity_sentences(cap)                 # face mode keeps it
    cleaned = fv.drop_identity_sentences(cap, body=True)
    assert 'tattoo' not in cleaned and 'Warm evening light.' in cleaned


def test_drop_identity_tags_body_mode():
    cap = 'standing, full_body, arm_tattoo, scar_on_cheek, red_dress, ear_piercing'
    assert fv.drop_identity_tags(cap) == cap                            # face mode keeps them
    assert fv.drop_identity_tags(cap, body=True) == 'standing, full_body, red_dress'


# --- dataset wiring --------------------------------------------------------------

def test_create_and_toggle_fidelity(app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Fb', 'fb', fidelity='body')
        assert ds.fidelity == 'body' and svc.is_body_fidelity(ds)
        assert svc.dataset_payload(LOCAL_USER, ds.id)['fidelity'] == 'body'
        assert svc.set_fidelity(LOCAL_USER, ds.id, 'face') is True
        assert svc.dataset_payload(LOCAL_USER, ds.id)['fidelity'] == 'face'
        # unknown value normalizes to face; concept datasets never carry a fidelity
        ds2 = svc.create_dataset(LOCAL_USER, 'Fx', 'fx', fidelity='nonsense')
        assert ds2.fidelity == 'face'
        c = svc.create_dataset(LOCAL_USER, 'Fc', 'fc', kind='concept',
                               concept_desc='an act', fidelity='body')
        assert c.fidelity is None


def test_fidelity_route_and_leak_badge(client, app):
    r = client.post('/api/dataset/create', json={
        'name': 'Body', 'trigger_word': 'body', 'fidelity': 'body'})
    ds_id = r.get_json()['id']
    assert client.get(f'/api/dataset/{ds_id}').get_json()['fidelity'] == 'body'
    # a caption with a tattoo counts as a leak for THIS dataset
    with app.app_context():
        from app.services import face_dataset_service as svc
        from app.models import FaceDatasetImage
        svc.db.session.add(FaceDatasetImage(dataset_id=ds_id, filename='x.webp', status='keep',
                                            caption='a tattoo on the arm, red dress'))
        svc.db.session.commit()
    payload = client.get(f'/api/dataset/{ds_id}').get_json()
    assert payload['caption_leak']['leaking'] == 1
    # switch back to face-only -> same caption no longer leaks
    assert client.post(f'/api/dataset/{ds_id}/fidelity',
                       json={'fidelity': 'face'}).status_code == 200
    payload = client.get(f'/api/dataset/{ds_id}').get_json()
    assert payload['caption_leak']['leaking'] == 0
    assert client.post('/api/dataset/999999/fidelity',
                       json={'fidelity': 'body'}).status_code == 404


def test_backup_preserves_fidelity(app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Bf', 'bf', fidelity='body')
        restored = svc.import_backup_zip(LOCAL_USER, svc.build_backup_zip(LOCAL_USER, ds.id))
        assert restored.fidelity == 'body'
