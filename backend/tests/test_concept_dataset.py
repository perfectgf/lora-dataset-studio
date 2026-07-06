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
        c = svc.create_dataset(LOCAL_USER, 'CIM', 'cim_act', kind='concept')
        p = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')  # défaut
        assert db.session.get(FaceDataset, c.id).kind == 'concept'
        assert db.session.get(FaceDataset, p.id).kind is None
        assert svc.is_concept(db.session.get(FaceDataset, c.id)) is True
        assert svc.is_concept(db.session.get(FaceDataset, p.id)) is False


# --- 2) Payload : kind exposé + badge caption_leak neutralisé ----------------
def test_payload_exposes_kind_and_zeros_leak_badge(app):
    with app.app_context():
        c = svc.create_dataset(LOCAL_USER, 'CIM', 'cim_act', kind='concept')
        # une caption GARDÉE qui mentionne l'identité (hair) → « fuite » côté perso,
        # mais VOULUE côté concept : le badge doit rester à 0.
        db.session.add(FaceDatasetImage(dataset_id=c.id, source='import', status='keep',
                                        filename='x.webp', caption='long brown hair, blue eyes'))
        db.session.commit()
        payload = svc.dataset_payload(LOCAL_USER, c.id)
        assert payload['kind'] == 'concept'
        assert payload['caption_leak']['leaking'] == 0
        assert payload['caption_leak']['captioned'] == 1

        p = svc.create_dataset(LOCAL_USER, 'Emma', 'z')
        assert svc.dataset_payload(LOCAL_USER, p.id)['kind'] == 'character'


# --- 3) Import service : concept garde le ratio, character pade en carré ------
def test_import_concept_keeps_aspect(app):
    with app.app_context():
        c = svc.create_dataset(LOCAL_USER, 'CIM', 'cim_act', kind='concept')
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
                      json={'name': 'CIM', 'trigger_word': 'cim_act', 'kind': 'concept'}
                      ).get_json()['id']

    captured = {}

    def fake_import(user_id, dataset_id, files, crop=False):
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


# --- 5) Captioner concept : prompt dédié + identité CONSERVÉE -----------------
def test_caption_concept_uses_concept_prompt_and_keeps_identity(app, monkeypatch):
    from app.services import vision_ollama
    with app.app_context():
        save_config({'captioning': {'backend': 'ollama'}})  # force Ollama, saute JoyCaption
        c = svc.create_dataset(LOCAL_USER, 'CIM', 'cim_act', kind='concept')
        # image gardée sans caption, fichier réel sur disque
        ids, _ = svc.import_images(LOCAL_USER, c.id, [_png(512, 512)], crop=False)
        img = db.session.get(FaceDatasetImage, ids[0])

        seen = {}

        def fake_describe(image_bytes, prompt, **kwargs):
            seen['prompt'] = prompt
            # une caption qui DÉCRIT l'identité (hair) — doit survivre au no-op cleaner
            return 'close-up, a woman with long brown hair and blue eyes, soft light'

        monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
        monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda: None)
        n = svc.caption_images(LOCAL_USER, c.id)

        assert n == 1
        assert seen['prompt'] == CAPTION_PROMPT_CONCEPT
        db.session.refresh(img)
        # l'identité est CONSERVÉE (cleaner no-op) — pas de suppression de « hair ».
        assert 'hair' in (img.caption or '')
