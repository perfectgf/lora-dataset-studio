"""🖼 The app-wide Gallery feed — every generated image, one page at a time.

What is asserted here is what a second reader would plausibly get wrong:

  * the feed is id-DESCENDING and excludes rows that are not images yet
    (pending, failed, or missing their filename);
  * pagination is a CURSOR, so a render landing at the head between two pages
    can never duplicate or skip a row on the next one;
  * each filter narrows `count` along with the page — the header number must
    name what the grid shows — while `datasets` stays unfiltered, because it
    feeds the picker that CHANGES the filter;
  * the route mirrors the service and refuses an unknown `kind` before the
    query rather than answering something else.
"""


def _create(client, name='Nova', trigger='nova'):
    return client.post('/api/dataset/create',
                       json={'name': name, 'trigger_word': trigger}).get_json()['id']


def _image(db, dataset_id, checkpoint='a.safetensors', **kw):
    from app.models import LoraTestImage
    img = LoraTestImage(dataset_id=dataset_id, checkpoint=checkpoint, strength=1.0,
                        status=kw.pop('status', 'done'),
                        filename=kw.pop('filename', 'x.png'), **kw)
    db.session.add(img)
    db.session.commit()
    return img


# --- the feed ----------------------------------------------------------------

def test_the_feed_is_newest_first_across_datasets_and_skips_non_images(client, app):
    from app.extensions import db
    from app.services import cloud_training as ct
    with app.app_context():
        a = _create(client, 'Nova', 'nova')
        b = _create(client, 'Vega', 'vega')
        first = _image(db, a, filename='a.png')
        second = _image(db, b, filename='b.png')
        # None of these is a picture anyone can look at.
        _image(db, a, status='pending', filename=None)
        _image(db, a, status='failed', filename=None)
        _image(db, b, status='done', filename=None)
        third = _image(db, a, filename='c.png')

        out = ct.app_gallery()
        assert [i['id'] for i in out['images']] == [third.id, second.id, first.id]
        assert out['count'] == 3
        assert out['has_more'] is False
        assert out['next_before_id'] is None


def test_an_install_that_never_generated_answers_an_empty_page(client, app):
    from app.services import cloud_training as ct
    with app.app_context():
        out = ct.app_gallery()
        assert (out['count'], out['images'], out['datasets']) == (0, [], [])
        assert out['has_more'] is False


def test_cursor_pages_never_overlap_even_when_the_head_grows(client, app):
    from app.extensions import db
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _create(client)
        rows = [_image(db, ds, filename=f'{i}.png') for i in range(5)]

        page1 = ct.app_gallery(limit=2)
        assert [i['id'] for i in page1['images']] == [rows[4].id, rows[3].id]
        assert page1['has_more'] is True
        assert page1['next_before_id'] == rows[3].id

        # A render landing between two requests grows the HEAD of the feed —
        # the cursor keeps page 2 exactly where page 1 left off.
        _image(db, ds, filename='new.png')
        page2 = ct.app_gallery(limit=2, before_id=page1['next_before_id'])
        assert [i['id'] for i in page2['images']] == [rows[2].id, rows[1].id]
        assert page2['has_more'] is True
        page3 = ct.app_gallery(limit=2, before_id=page2['next_before_id'])
        assert [i['id'] for i in page3['images']] == [rows[0].id]
        assert page3['has_more'] is False
        assert page3['next_before_id'] is None


# --- filters -----------------------------------------------------------------

def test_filters_narrow_the_count_with_the_page(client, app):
    from app.extensions import db
    from app.services import cloud_training as ct
    with app.app_context():
        a = _create(client, 'Nova', 'nova')
        b = _create(client, 'Vega', 'vega')
        render = _image(db, a, filename='a.png', rating=1)
        improved = _image(db, a, filename='b.png',
                          derivation_kind='canvas_image_improve',
                          parent_image_id=render.id)
        other = _image(db, b, filename='c.png')

        by_ds = ct.app_gallery(dataset_id=b)
        assert ([i['id'] for i in by_ds['images']], by_ds['count']) == ([other.id], 1)

        renders = ct.app_gallery(kind='renders')
        assert {i['id'] for i in renders['images']} == {render.id, other.id}
        assert renders['count'] == 2

        just_improved = ct.app_gallery(kind='improved')
        assert ([i['id'] for i in just_improved['images']],
                just_improved['count']) == ([improved.id], 1)

        liked = ct.app_gallery(liked=True)
        assert ([i['id'] for i in liked['images']], liked['count']) == ([render.id], 1)


def test_the_dataset_list_feeds_the_picker_so_it_is_never_filtered(client, app):
    from app.extensions import db
    from app.services import cloud_training as ct
    with app.app_context():
        a = _create(client, 'Nova', 'nova')
        b = _create(client, 'Vega', 'vega')
        empty = _create(client, 'Mira', 'mira')
        _image(db, a, filename='a.png')
        _image(db, a, filename='b.png')
        _image(db, b, filename='c.png')

        out = ct.app_gallery(dataset_id=b)
        # Sorted by name; a dataset with nothing generated has no entry to
        # offer, and the current pick does not shrink the list to itself.
        assert out['datasets'] == [
            {'id': a, 'name': 'Nova', 'count': 2},
            {'id': b, 'name': 'Vega', 'count': 1},
        ]
        assert empty not in [d['id'] for d in out['datasets']]


# --- gestion: delete ---------------------------------------------------------

def test_gallery_delete_reaches_rows_from_any_run_but_never_a_generating_one(client, app):
    from app.extensions import db
    from app.models import LoraTestImage
    from app.services import cloud_training as ct
    with app.app_context():
        a = _create(client, 'Nova', 'nova')
        b = _create(client, 'Vega', 'vega')
        one = _image(db, a, record_id=1, step=500, filename='a.png')
        two = _image(db, b, record_id=2, step=1000, filename='b.png')
        cooking = _image(db, a, status='pending', filename=None)

        out = ct.delete_gallery_images([one.id, two.id, cooking.id])
        # No file on disk in this fixture: the rows still go, honestly counted.
        assert out['rows_removed'] == 2
        assert out['already_absent'] == 2
        assert out['skipped'] == [{'id': cooking.id, 'reason': 'generating'}]
        assert sorted(out['dataset_ids']) == sorted([a, b])
        assert LoraTestImage.query.count() == 1   # the generating cell survives


def test_the_checkpoint_scoped_delete_still_refuses_ids_outside_its_scope(client, app):
    """Widening the gallery must not have loosened the narrow routes."""
    from app.extensions import db
    from app.models import LoraTestImage
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _create(client)
        mine = _image(db, ds, record_id=7, step=500, filename='a.png')
        other = _image(db, ds, record_id=8, step=500, filename='b.png')
        out = ct.delete_checkpoint_images(7, 500, [mine.id, other.id])
        assert out['rows_removed'] == 1
        assert out['skipped'] == [{'id': other.id, 'reason': 'not_in_gallery'}]
        assert db.session.get(LoraTestImage, other.id) is not None


def test_delete_route_and_the_feeds_delete_mode_agree(client, app):
    from app.extensions import db
    with app.app_context():
        ds = _create(client)
        img = _image(db, ds, filename='a.png')
        image_id = img.id
    feed = client.get('/api/gallery/images').get_json()
    assert feed['delete_mode'] in ('trash', 'app_trash', 'delete')
    r = client.post('/api/gallery/images/delete', json={'image_ids': [image_id]})
    assert r.status_code == 200
    assert r.get_json()['rows_removed'] == 1
    assert client.get('/api/gallery/images').get_json()['count'] == 0


# --- gestion: ZIP ------------------------------------------------------------

def test_zip_plan_is_selection_only_and_names_the_lineage(client, app):
    import os
    from app.extensions import db
    from app.services import gallery_download as gdl
    from app.services.dataset_storage import dataset_path
    with app.app_context():
        ds = _create(client, 'Nova Style', 'nova')
        real = _image(db, ds, record_id=42, step=2500, seed=99, filename='real.png')
        gone = _image(db, ds, record_id=42, step=2500, filename='gone.png')
        folder = dataset_path(ds)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, 'real.png'), 'wb') as f:
            f.write(b'\x89PNG')

        plan = gdl.app_gallery_download_plan([real.id, gone.id])
        assert (plan['ok'], plan['total'], plan['included'], plan['missing']) == (
            True, 2, 1, 1)
        assert plan['entries'][0]['name'] == (
            f'Nova-Style_run42_step002500_seed99_{real.id}.png')
        assert plan['filename'] == 'Nova-Style_gallery_selection.zip'

        # Two datasets in one pick: no single name can carry the identity.
        other = _create(client, 'Vega', 'vega')
        v = _image(db, other, filename='v.png')
        folder2 = dataset_path(other)
        os.makedirs(folder2, exist_ok=True)
        with open(os.path.join(folder2, 'v.png'), 'wb') as f:
            f.write(b'\x89PNG')
        mixed = gdl.app_gallery_download_plan([real.id, v.id])
        assert mixed['filename'] == 'gallery_selection.zip'


def test_zip_routes_require_an_explicit_selection(client, app):
    assert client.get('/api/gallery/images/zip').status_code == 400
    assert client.get('/api/gallery/images/zip/plan').status_code == 400
    # An unparseable selection is refused out loud, never widened to everything.
    r = client.get('/api/gallery/images/zip/plan?ids=9999')
    assert r.status_code == 200
    assert r.get_json()['ok'] is False


# --- the route ---------------------------------------------------------------

def test_the_route_serves_the_feed_and_refuses_an_unknown_kind(client, app):
    from app.extensions import db
    with app.app_context():
        ds = _create(client)
        img = _image(db, ds, filename='a.png')
        image_id = img.id

    r = client.get('/api/gallery/images')
    assert r.status_code == 200
    d = r.get_json()
    assert [i['id'] for i in d['images']] == [image_id]
    # The page publishes the SAME shape the checkpoint gallery does — the
    # lightbox reads these keys (services.cloud_training._gallery_image).
    assert d['images'][0]['url'].endswith('/img/a.png')
    assert 'prompt' in d['images'][0] and 'seed' in d['images'][0]

    assert client.get('/api/gallery/images?kind=all').status_code == 400
    ok = client.get('/api/gallery/images?kind=improved&liked=1&limit=5')
    assert ok.status_code == 200
    assert ok.get_json()['count'] == 0
