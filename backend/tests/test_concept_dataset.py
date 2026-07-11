"""Dataset de type CONCEPT (LoRA de concept, ≠ personnage).

Vérifie l'inversion de logique : `kind='concept'` persisté, import SANS head-crop
et aspect PRÉSERVÉ (pas de bandes noires), captioner qui GARDE l'identité (prompt
dédié + cleaner no-op), badge caption_leak neutralisé, et route d'import qui
n'ouvre PAS la fenêtre GPU exclusive (aucune passe vision pour un import brut).

Porté de l'app source, adapté à notre extraction mono-utilisateur : LOCAL_USER
(pas de fixture admin_user), racine d'images via la fixture `app` (LDS_DATA_DIR),
et vision_ollama importé LOCALEMENT par caption_images → on patche à la source.
"""
import io

from PIL import Image

from app.extensions import db
from app.models import FaceDataset, FaceDatasetImage
from app.services import face_dataset_service as svc
from app.services.face_variations import CAPTION_PROMPT_CONCEPT
from app.config import LOCAL_USER, save_config


# A concept dataset now REQUIRES a concept description (what every caption must omit).
# Chosen so none of its words appear in the identity caption used below → the omission
# guarantee is a no-op there and can't perturb the prompt-capture assertion.
CONCEPT_DESC = 'an ice cream cone being licked'


def _png(w=800, h=400):
    b = io.BytesIO()
    Image.new('RGB', (w, h), (120, 40, 40)).save(b, 'PNG')
    return b.getvalue()


# --- 1) Modèle / CRUD --------------------------------------------------------
def test_normalize_kind_and_is_concept():
    assert svc.normalize_kind('concept') == 'concept'
    assert svc.normalize_kind('CONCEPT') == 'concept'
    # tout le reste -> None (character, stocké NULL)
    assert svc.normalize_kind('character') is None
    assert svc.normalize_kind(None) is None
    assert svc.normalize_kind('') is None
    assert svc.normalize_kind('personnage') is None


def test_create_dataset_persists_concept(app):
    with app.app_context():
        c = svc.create_dataset(LOCAL_USER, 'CIM', 'cim_act', kind='concept', concept_desc=CONCEPT_DESC)
        p = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')  # défaut
        assert db.session.get(FaceDataset, c.id).kind == 'concept'
        assert db.session.get(FaceDataset, c.id).concept_desc == CONCEPT_DESC
        assert db.session.get(FaceDataset, p.id).kind is None
        assert db.session.get(FaceDataset, p.id).concept_desc is None
        assert svc.is_concept(db.session.get(FaceDataset, c.id)) is True
        assert svc.is_concept(db.session.get(FaceDataset, p.id)) is False


def test_create_concept_requires_desc(app):
    """Le concept_desc est ce que la caption OMET → sans lui, la logique inversée n'a rien
    à lier au trigger. create_dataset refuse (ValueError → 400 côté route)."""
    import pytest
    with app.app_context():
        with pytest.raises(ValueError):
            svc.create_dataset(LOCAL_USER, 'CIM', 'cim_act', kind='concept')
        with pytest.raises(ValueError):
            svc.create_dataset(LOCAL_USER, 'CIM', 'cim_act', kind='concept', concept_desc='   ')
        # un character sans desc reste parfaitement valide
        p = svc.create_dataset(LOCAL_USER, 'Emma', 'z')
        assert p.concept_desc is None


# --- 2) Payload : kind exposé + badge caption_leak neutralisé ----------------
def test_payload_exposes_kind_and_zeros_leak_badge(app):
    with app.app_context():
        c = svc.create_dataset(LOCAL_USER, 'CIM', 'cim_act', kind='concept', concept_desc=CONCEPT_DESC)
        # une caption GARDÉE qui mentionne l'identité (hair) → « fuite » côté perso,
        # mais VOULUE côté concept : le badge doit rester à 0.
        db.session.add(FaceDatasetImage(dataset_id=c.id, source='import', status='keep',
                                        filename='x.webp', caption='long brown hair, blue eyes'))
        db.session.commit()
        payload = svc.dataset_payload(LOCAL_USER, c.id)
        assert payload['kind'] == 'concept'
        assert payload['concept_desc'] == CONCEPT_DESC
        assert payload['caption_leak']['leaking'] == 0
        assert payload['caption_leak']['captioned'] == 1

        p = svc.create_dataset(LOCAL_USER, 'Emma', 'z')
        payload_p = svc.dataset_payload(LOCAL_USER, p.id)
        assert payload_p['kind'] == 'character'
        assert payload_p['concept_desc'] == ''  # character → jamais de concept_desc


# --- 3) Import service : concept garde le ratio, character pade en carré ------
def test_import_concept_keeps_aspect(app):
    with app.app_context():
        c = svc.create_dataset(LOCAL_USER, 'CIM', 'cim_act', kind='concept', concept_desc=CONCEPT_DESC)
        ids, failed = svc.import_images(LOCAL_USER, c.id, [_png(800, 400)], crop=False)
        assert failed == 0 and len(ids) == 1
        img = db.session.get(FaceDatasetImage, ids[0])
        w, h = Image.open(svc._img_path(img)).size
        # ratio PRÉSERVÉ (paysage 2:1) → jamais carré, pas de letterbox.
        assert w > h and w <= 1024


def test_import_character_no_crop_keeps_aspect(app):
    """Un import personnage SANS head-crop préserve le ratio (plus de carré padé :
    les bandes noires étaient apprises par le LoRA et tout import devenait carré)."""
    with app.app_context():
        p = svc.create_dataset(LOCAL_USER, 'Emma', 'z')  # character
        ids, _ = svc.import_images(LOCAL_USER, p.id, [_png(800, 400)], crop=False)
        img = db.session.get(FaceDatasetImage, ids[0])
        w, h = Image.open(svc._img_path(img)).size
        assert (w, h) == (800, 400)  # ratio d'origine, pas de padding


# --- 4) Route d'import concept : PAS de fenêtre GPU exclusive -----------------
def test_import_route_concept_skips_gpu_window(client, monkeypatch):
    import app.routes.datasets as dr
    did = client.post('/api/dataset/create',
                      json={'name': 'CIM', 'trigger_word': 'cim_act', 'kind': 'concept',
                            'concept_desc': CONCEPT_DESC}
                      ).get_json()['id']

    captured = {}

    def fake_import(user_id, dataset_id, files, crop=False, **kw):
        captured['crop'] = crop
        return [1], 0

    def boom(*a, **k):
        raise AssertionError('la fenêtre GPU exclusive ne doit PAS être ouverte pour un import concept')

    monkeypatch.setattr(dr.svc, 'import_images', fake_import)
    monkeypatch.setattr(dr, 'gpu_exclusive_vision_window', boom)
    resp = client.post(f'/api/dataset/{did}/import',
                       data={'files': (io.BytesIO(_png()), 'a.png')},
                       content_type='multipart/form-data')
    assert resp.status_code == 200, resp.get_data(as_text=True)[:300]
    assert captured['crop'] is False


# --- 5) Captioner concept : prompt dédié ({concept} rempli) + identité CONSERVÉE ------
def test_caption_concept_uses_concept_prompt_and_keeps_identity(app, monkeypatch):
    from app.services import vision_ollama
    with app.app_context():
        save_config({'captioning': {'backend': 'ollama'}})  # force Ollama, saute JoyCaption
        c = svc.create_dataset(LOCAL_USER, 'CIM', 'cim_act', kind='concept', concept_desc=CONCEPT_DESC)
        # image gardée sans caption, fichier réel sur disque
        ids, _ = svc.import_images(LOCAL_USER, c.id, [_png(512, 512)], crop=False)
        img = db.session.get(FaceDatasetImage, ids[0])

        prompts = []

        def fake_describe(image_bytes, prompt, **kwargs):
            prompts.append(prompt)
            # une caption qui DÉCRIT l'identité (hair) — doit survivre au no-op cleaner.
            # Aucun mot de CONCEPT_DESC (ice/cream/cone/licked) n'y figure → la garantie
            # d'omission est un no-op, elle ne relance donc pas describe.
            return 'close-up, a woman with long brown hair and blue eyes, soft light'

        monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
        monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda: None)
        n = svc.caption_images(LOCAL_USER, c.id)

        assert n == 1
        # Le prompt de caption est le prompt CONCEPT avec {concept} RÉELLEMENT injecté.
        expected = CAPTION_PROMPT_CONCEPT.format(concept=CONCEPT_DESC)
        assert expected in prompts
        assert CONCEPT_DESC in expected  # placeholder bien substitué
        db.session.refresh(img)
        # l'identité est CONSERVÉE (cleaner no-op) — pas de suppression de « hair ».
        assert 'hair' in (img.caption or '')


# --- update_dataset_settings (édition post-création) -------------------------
def test_update_settings_name_and_trigger(app):
    with app.app_context():
        d = svc.create_dataset(LOCAL_USER, 'Old', 'oldtrig')
        res = svc.update_dataset_settings(LOCAL_USER, d.id, name='New', trigger_word='newtrig')
        assert res == {'ok': True, 'concept_desc_changed': False}
        db.session.refresh(d)
        assert d.name == 'New' and d.trigger_word == 'newtrig'


def test_update_settings_empty_trigger_rejected(app):
    import pytest
    with app.app_context():
        d = svc.create_dataset(LOCAL_USER, 'X', 'trig')
        with pytest.raises(ValueError):
            svc.update_dataset_settings(LOCAL_USER, d.id, trigger_word='   ')
        db.session.refresh(d)
        assert d.trigger_word == 'trig'   # inchangé


def test_update_settings_concept_desc_resets_avoidlist_cache(app):
    with app.app_context():
        d = svc.create_dataset(LOCAL_USER, 'C', 'cact', kind='concept', concept_desc=CONCEPT_DESC)
        d.concept_terms = '["ice", "cream", "cone"]'   # cache LLM simulé
        db.session.commit()
        res = svc.update_dataset_settings(LOCAL_USER, d.id, concept_desc='a mirror selfie')
        assert res['concept_desc_changed'] is True
        db.session.refresh(d)
        assert d.concept_desc == 'a mirror selfie'
        assert d.concept_terms is None   # cache invalidé → régénéré au prochain caption


def test_update_settings_same_concept_desc_keeps_cache(app):
    with app.app_context():
        d = svc.create_dataset(LOCAL_USER, 'C', 'cact', kind='concept', concept_desc=CONCEPT_DESC)
        d.concept_terms = '["ice"]'
        db.session.commit()
        res = svc.update_dataset_settings(LOCAL_USER, d.id, concept_desc=CONCEPT_DESC)
        assert res['concept_desc_changed'] is False
        db.session.refresh(d)
        assert d.concept_terms == '["ice"]'   # inchangé → pas de re-génération inutile


def test_update_settings_concept_desc_ignored_on_character(app):
    with app.app_context():
        d = svc.create_dataset(LOCAL_USER, 'P', 'ptrig')   # character
        res = svc.update_dataset_settings(LOCAL_USER, d.id, concept_desc='whatever')
        assert res['concept_desc_changed'] is False
        db.session.refresh(d)
        assert d.concept_desc is None   # un personnage n'a pas de concept_desc


def test_update_settings_missing_dataset_returns_none(app):
    with app.app_context():
        assert svc.update_dataset_settings(LOCAL_USER, 999999, name='x') is None


# --- Ban-list parser: salvage the looping/unclosed JSON the abliterated Qwen emits ----
# Real failure mode (2026-07): the model lists the good concept terms first, then loops
# into combinatorial padding and never closes the array -> json.loads fails. The OLD
# parser returned [] -> empty ban-list -> the concept leaked into EVERY caption. The new
# parser must salvage the leading terms, keep order, de-dup, and cap.
def test_parse_terms_json_clean_object():
    out = svc._parse_terms_json('noise {"terms": ["Mirror", "phone", "selfie"]} trailing')
    assert out == ['mirror', 'phone', 'selfie']            # order kept, lowercased


def test_parse_terms_json_salvages_unclosed_loop():
    raw = ('{ "terms": ["mirror selfie", "self-portrait", "reflection", "camera", '
           '"phone", "selfie", "mirror", "mirror selfie shot", "self-portrait photo", '
           '"mirror image capture", "selfie')  # LOOPING + never closed (no final ] })
    out = svc._parse_terms_json(raw)
    # the concept-specific leaders survive even though json.loads can't parse this
    for t in ('mirror selfie', 'self-portrait', 'reflection', 'camera', 'phone', 'selfie', 'mirror'):
        assert t in out
    assert out == list(dict.fromkeys(out))                 # de-duped, order preserved
    assert len(out) <= 25                                  # padding can't dominate


def test_parse_terms_json_dedupes_and_drops_stopwords():
    raw = '{"terms": ["mirror", "mirror", "the", "a", "in", "phone", "bare"]}'
    # 'the'/'in'/'bare' are stopwords, 'a' is too short -> only the real terms, once each
    assert svc._parse_terms_json(raw) == ['mirror', 'phone']


def test_parse_terms_json_no_terms_returns_empty():
    assert svc._parse_terms_json('sorry, I cannot help with that') == []
    assert svc._parse_terms_json('') == []


# --- Reasoning-trace / degenerate caption rejection (the ~5 broken captions) -----------
def test_refine_output_rejects_reasoning_traces():
    prior = 'a detailed joycaption draft describing the whole scene at length' * 2
    # Reasoning/meta phrasings are rejected by BOTH gates (never committed, never a refine).
    for bad in (
        'Yes, this describes the mirror frame, the floor and the bed, but not the concept.',
        'The original caption says the lighting is soft and even, which is correct.',
        'Now, check for leaked words - none. The caption looks clean enough.',
        'We need to remove the mirror and rephrase around it so the prose stays natural.',
    ):
        assert svc._refine_output_ok(bad, prior) is False
        assert svc._usable_caption(bad) is False


def test_refine_output_rejects_degenerate_but_usable_allows_terse():
    prior = 'a long joycaption draft describing the whole scene in rich detail here'
    # A degenerate one-liner is bounced from the REFINE (too short -> fall back to direct),
    # but _usable_caption does NOT gate on length (a terse post-scrub caption still commits;
    # the truly degenerate case is scrubbed to empty upstream and rejected by the blank check).
    assert svc._refine_output_ok('taking a picture', prior) is False
    assert svc._usable_caption('taking a picture') is True
    assert svc._usable_caption('') is False
    assert svc._usable_caption('   ') is False


def test_refine_output_accepts_clean_caption():
    prior = 'joy draft'
    good = ('Full-body shot framed by a wooden doorway, a woman with dark hair tied back '
            'stands in a sunlit bedroom, her expression calm, soft warm light across the floor.')
    assert svc._refine_output_ok(good, good) is True
    assert svc._usable_caption(good) is True


def test_expand_prompt_is_loop_resistant():
    """Guard against reintroducing the loop-seeding examples: the prompt must NOT list
    residue examples and MUST tell the model to emit each term once and stop."""
    from app.services.face_variations import EXPAND_CONCEPT_TERMS_PROMPT as P
    low = P.lower()
    assert 'glistening' not in low and 'dripping' not in low and 'sticky' not in low
    assert 'once' in low and 'stop' in low
    assert '{concept}' in P or '{{' in P                   # placeholder survives .format


# --- Deterministic capture lexicon (the phone/camera/reflection leak fix) ---------------
def test_fallback_terms_inject_capture_lexicon_for_selfie():
    """A photographic concept ALWAYS bans the full capture vocabulary, even though the LLM
    expansion for 'a candid mirror selfie' only ever returned mirror/self-* variants."""
    terms = set(svc._fallback_concept_terms('a candid mirror selfie'))
    for must in ('phone', 'smartphone', 'camera', 'reflection', 'selfie', 'mirror',
                 'pov', 'point of view', 'webcam'):
        assert must in terms, f'{must!r} must be banned for a selfie concept'
    assert 'candid' in terms                                # the desc's own words too


def test_fallback_terms_trigger_variants():
    """Any photographic keyword in the desc pulls in the lexicon (not just 'selfie')."""
    for desc in ('a phone selfie', 'a bathroom mirror shot', 'a webcam photo',
                 'a candid self-portrait', 'a POV picture'):
        terms = set(svc._fallback_concept_terms(desc))
        assert {'phone', 'camera', 'reflection'} <= terms, desc


def test_fallback_terms_no_lexicon_for_nonphotographic_concept():
    """A non-photographic concept must NOT drag in phone/camera (those would wrongly scrub
    legitimate description in an unrelated dataset)."""
    terms = set(svc._fallback_concept_terms('a red balloon'))
    assert 'phone' not in terms and 'camera' not in terms and 'selfie' not in terms
    assert 'red' in terms and 'balloon' in terms


def test_leak_re_from_lexicon_catches_real_leaks():
    """End-to-end: the regex built from the selfie ban-list matches the exact phrasings
    that leaked in the live run (smartphone, standalone camera/mirror/reflection)."""
    leak_re = svc._concept_terms_re(svc._fallback_concept_terms('a candid mirror selfie'))
    for leaky in (
        'She holds a black smartphone in her right hand to frame the image.',
        'a woman holding a phone to capture her reflection in the mirror',
        'Close-up framing captures a woman angled toward the camera.',
        'A point of view reflection shot in the bathroom.',
    ):
        assert leak_re.search(leaky), leaky
    # a clean caption (no capture-language) must NOT trip the detector
    assert not leak_re.search(
        'Full-body shot of a woman with dark hair in a sunlit bedroom, soft warm light.')
