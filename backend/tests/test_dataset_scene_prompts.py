"""🎬 Scenes from a DATASET — the second source of the ordered scene cards the
generation panels (Test Studio and the board's 🎨 Generate) offer as extra
passes of the 📝 prompt axis.

A Bank got this first. A dataset is where the captions the user actually
curated live, so the same read had to exist there or the two surfaces would
answer differently to the same question — the divergence CLAUDE.md's "Bank and
Dataset are two surfaces of one product" section exists to prevent.

The contract these tests hold:
  - dataset order IS the sequence (row id ascending);
  - a missing FRAMING is not a gate — it rides as 'body';
  - a missing CAPTION skips, and is COUNTED, never guessed;
  - REJECTED images stay out by default (a dataset is curated: the user already
    said no), and ?statuses= can widen the read deliberately;
  - each card carries the FILENAME its thumbnail is addressed by, and a row with
    no file on disk yet still yields its card;
  - the route is read-only and 404s on a dataset that does not exist.
"""
from app.config import LOCAL_USER


def _dataset_with_captions(captions, *, framings=None, statuses=None, name='Chapter'):
    """A dataset of len(captions) image rows, captioned in order. Returns its id.

    No pixels are written: every assertion here is about the ROWS and their
    order, and a scene card never opens the file it names.
    """
    from app.extensions import db
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    ds = svc.create_dataset(LOCAL_USER, name, 'zchar_scene')
    for i, caption in enumerate(captions):
        db.session.add(FaceDatasetImage(
            dataset_id=ds.id, filename=f'img_{i:03d}.png', source='import',
            status=(statuses[i] if statuses else 'keep'),
            framing=(framings[i] if framings else 'body'),
            caption=caption))
    db.session.commit()
    return ds.id


def test_scenes_come_out_in_dataset_order(app):
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds_id = _dataset_with_captions(['street at dawn',
                                        'close on her face',
                                        'rooftop chase'])
        out = svc.export_scene_captions(LOCAL_USER, ds_id)
    assert out['dataset_name'] == 'Chapter'
    assert [s['prompt'] for s in out['scenes']] == [
        'street at dawn', 'close on her face', 'rooftop chase']
    assert out['scenes'][2]['label'].startswith('Scene 3 — img_002.png')
    assert out['skipped'] == {'no_caption': 0}


def test_an_image_without_a_caption_is_skipped_and_counted(app):
    """Never guessed: an uncaptioned image cannot become a prompt, but the user
    must be told how many stayed behind rather than silently losing beats."""
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds_id = _dataset_with_captions(['one', '', '   ', 'four'])
        out = svc.export_scene_captions(LOCAL_USER, ds_id)
    assert [s['prompt'] for s in out['scenes']] == ['one', 'four']
    assert out['skipped']['no_caption'] == 2
    # Numbering follows the KEPT scenes, not the dataset rows.
    assert [s['label'].split(' — ')[0] for s in out['scenes']] == ['Scene 1', 'Scene 2']


def test_a_missing_framing_rides_as_body_instead_of_dropping_the_scene(app):
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds_id = _dataset_with_captions(['a', 'b', 'c'],
                                       framings=['face', 'unknown', None])
        out = svc.export_scene_captions(LOCAL_USER, ds_id)
    assert [s['framing'] for s in out['scenes']] == ['face', 'body', 'body']
    assert out['skipped']['no_caption'] == 0


def test_rejected_images_stay_out_unless_the_scope_asks_for_them(app):
    """A dataset is CURATED — an image the user rejected is an answer, not a
    gap. It never rides by default, and asking for it is explicit."""
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds_id = _dataset_with_captions(
            ['kept one', 'thrown away', 'still pending'],
            statuses=['keep', 'reject', 'pending'])
        default = svc.export_scene_captions(LOCAL_USER, ds_id)
        widened = svc.export_scene_captions(
            LOCAL_USER, ds_id, statuses=['keep', 'pending', 'reject'])
        only_kept = svc.export_scene_captions(LOCAL_USER, ds_id, statuses=['keep'])
    assert [s['prompt'] for s in default['scenes']] == ['kept one', 'still pending']
    assert [s['prompt'] for s in widened['scenes']] == [
        'kept one', 'thrown away', 'still pending']
    assert [s['prompt'] for s in only_kept['scenes']] == ['kept one']
    # A skipped row is one that had no caption — never one the scope excluded.
    assert default['skipped']['no_caption'] == 0


def test_a_long_caption_is_cut_on_a_word_under_the_importer_ceiling(app):
    from app.services import face_dataset_service as svc
    from app.services import scene_captions
    with app.app_context():
        ds_id = _dataset_with_captions([('word ' * 200).strip()])
        out = svc.export_scene_captions(LOCAL_USER, ds_id)
    prompt = out['scenes'][0]['prompt']
    assert len(prompt) <= scene_captions.SCENE_MAX_PROMPT
    assert prompt.endswith('…')
    assert 'wor…' not in prompt          # cut on a word boundary, never mid-word


def test_each_scene_carries_the_filename_its_thumbnail_is_addressed_by(app):
    """Display-only: the panel renders /api/dataset/<id>/thumb/<filename>. A row
    whose file is not written yet (a generation still in flight) still yields its
    card — the caption IS the scene — with no filename to point an <img> at."""
    from app.extensions import db
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds_id = _dataset_with_captions(['a', 'b'])
        db.session.add(FaceDatasetImage(
            dataset_id=ds_id, filename=None, source='generated', status='pending',
            framing='body', caption='still rendering'))
        db.session.commit()
        out = svc.export_scene_captions(LOCAL_USER, ds_id)
    assert [s['filename'] for s in out['scenes']] == [
        'img_000.png', 'img_001.png', None]
    assert out['scenes'][2]['label'].startswith('Scene 3 — image ')


def test_scenes_route_serves_the_cards_and_404s_on_a_missing_dataset(app, client):
    with app.app_context():
        ds_id = _dataset_with_captions(['panel one', 'panel two'])
    r = client.get(f'/api/dataset/{ds_id}/scenes')
    assert r.status_code == 200
    body = r.get_json()
    assert body['dataset_id'] == ds_id
    assert [s['prompt'] for s in body['scenes']] == ['panel one', 'panel two']
    assert client.get('/api/dataset/999999/scenes').status_code == 404


def test_the_route_rejects_an_unknown_scope_instead_of_ignoring_it(app, client):
    """A scope the server silently drops is a run over images the user believed
    excluded — 400, like every other pass scope."""
    with app.app_context():
        ds_id = _dataset_with_captions(['one'])
    assert client.get(f'/api/dataset/{ds_id}/scenes?statuses=deleted').status_code == 400
