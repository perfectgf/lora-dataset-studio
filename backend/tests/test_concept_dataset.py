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


def test_import_character_no_crop_pads_square(app):
    with app.app_context():
        p = svc.create_dataset(LOCAL_USER, 'Emma', 'z')  # character
        ids, _ = svc.import_images(LOCAL_USER, p.id, [_png(800, 400)], crop=False)
        img = db.session.get(FaceDatasetImage, ids[0])
        w, h = Image.open(svc._img_path(img)).size
        assert w == h  # carré padé (comportement historique personnage)


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
