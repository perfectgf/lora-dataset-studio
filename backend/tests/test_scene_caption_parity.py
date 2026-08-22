"""🎬 Scenes — Bank and Dataset must answer IDENTICALLY for the same caption.

Two surfaces offer scene cards: a Bank's captions and a Dataset's. They are two
different queries over two different tables, and CLAUDE.md is explicit about
what that costs when the SHAPE is duplicated instead of shared — the face pass's
size gate diverged exactly this way and shipped divergent.

So the ceiling, the word-boundary cut, the framing fallback and the label live
in ``services/scene_captions.py``, and this file reads BOTH services against it.
It is a source-level pin: if someone re-inlines a 500 or a `'body'` on one side,
these fail before the two panels can start disagreeing on screen.
"""
import re

from app.config import LOCAL_USER

# One caption per case, chosen for the edges the two sides could drift on.
CAPTIONS = [
    'street at dawn',                       # ordinary
    ('word ' * 200).strip(),                # over the ceiling → cut on a word
    '   spaced    out   caption   ',        # whitespace collapsing
]


def _bank_scenes(tmp_path, captions, framings):
    from PIL import Image
    from app.extensions import db
    from app.models import BankImage
    from app.services import image_bank_service as banks
    src = tmp_path / 'parity-bank'
    src.mkdir()
    for i in range(len(captions)):
        Image.new('RGB', (40, 60), (10, 20, 30)).save(str(src / f'p{i:03d}.jpg'))
    bank, _ = banks.create_bank(LOCAL_USER, 'Parity', str(src))
    db.session.commit()
    rows = (BankImage.query.filter_by(bank_id=bank.id)
            .order_by(BankImage.id.asc()).all())
    for row, caption, framing in zip(rows, captions, framings):
        row.caption, row.framing = caption, framing
    db.session.commit()
    return banks.export_scene_captions(LOCAL_USER, bank.id)['scenes']


def _dataset_scenes(captions, framings):
    from app.extensions import db
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    ds = svc.create_dataset(LOCAL_USER, 'Parity', 'zchar_parity')
    for i, (caption, framing) in enumerate(zip(captions, framings)):
        db.session.add(FaceDatasetImage(
            dataset_id=ds.id, filename=f'p{i:03d}.jpg', source='import',
            status='keep', framing=framing, caption=caption))
    db.session.commit()
    return svc.export_scene_captions(LOCAL_USER, ds.id)['scenes']


def test_the_same_caption_becomes_the_same_prompt_on_both_surfaces(app, tmp_path):
    framings = ['face', None, 'unknown']
    with app.app_context():
        bank = _bank_scenes(tmp_path, CAPTIONS, framings)
        dataset = _dataset_scenes(CAPTIONS, framings)
    assert [s['prompt'] for s in bank] == [s['prompt'] for s in dataset]
    assert [s['framing'] for s in bank] == [s['framing'] for s in dataset]
    # The file names are identical here on purpose: the label is the same
    # sentence on both sides, so a card reads the same wherever it came from.
    assert [s['label'] for s in bank] == [s['label'] for s in dataset]


def test_neither_service_carries_its_own_copy_of_the_scene_shape(app):
    """A source pin, because a re-inlined constant passes every behaviour test
    on the day it is written and only diverges months later."""
    import inspect
    from app.services import face_dataset_service, image_bank_service, scene_captions

    for module in (image_bank_service, face_dataset_service):
        src = inspect.getsource(module)
        section = src[src.index('# --- Scene captions'):]
        assert 'SCENE_MAX_PROMPT = 500' not in section, (
            f'{module.__name__} re-inlined the prompt ceiling — it belongs to '
            'services/scene_captions.py, which the other surface reads too')
        assert not re.search(r"\('face',\s*'bust',\s*'body',\s*'back'\)", section), (
            f'{module.__name__} re-inlined the framing vocabulary')
        assert "f'Scene {" not in section, (
            f'{module.__name__} re-inlined the scene label')
    assert scene_captions.SCENE_MAX_PROMPT == 500
