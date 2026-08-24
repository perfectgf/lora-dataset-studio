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
