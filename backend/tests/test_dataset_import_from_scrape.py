"""Scrape DIRECT → dataset CONCEPT (scrape_import_urls).

Flux AUTONOME : les images scannées sélectionnées sont téléchargées directement
dans le dataset. Vérifie : filtres qualité (côté court ≥768, ratio ≤3:1), dedup
dHash intra-batch ET vs images existantes, mapping des raisons de download, et la
route (concept-only, cap, compteurs skipped). Le download réseau est monkeypatché.

Porté de l'app source, adapté à notre extraction mono-utilisateur : LOCAL_USER,
racine d'images via la fixture `app`, route sous app.routes.datasets.
"""
import io
from unittest.mock import patch

from PIL import Image

from app.models import FaceDatasetImage
from app.services import face_dataset_service as svc
from app.config import LOCAL_USER


def _img_bytes(w=1280, h=960, fmt='JPEG', grad=None):
    """grad='ltr'|'rtl' → gradient horizontal BASSE FRÉQUENCE. Indispensable pour
    les tests dHash : un motif fin se fait moyenner par le resize 9×8 en aplat
    uniforme → collisions. Un gradient survit au downscale (ltr→0, rtl→1)."""
    if grad:
        ramp = list(range(0, 256, 32))
        if grad == 'rtl':
            ramp = ramp[::-1]
        small = Image.new('L', (8, 8))
        small.putdata([ramp[x] for _ in range(8) for x in range(8)])
        im = small.resize((w, h), Image.BILINEAR).convert('RGB')
    else:
        im = Image.new('RGB', (w, h), (120, 40, 40))
    b = io.BytesIO()
    im.save(b, fmt)
    return b.getvalue()


def _concept(user_id=LOCAL_USER):
    return svc.create_dataset(user_id, 'CIM', 'cim_act', kind='concept')


def _item(url):
    return {'url': url, 'title': ''}


def _fake_downloader(by_url):
    """Retourne un _download_scrape_item bouchonné depuis {url: bytes|None}.
    None → échec réseau ('errors')."""
    def _dl(item):
        data = by_url.get(item['url'])
        return ('ok', data) if data is not None else ('errors', None)
    return _dl


# --- dHash -------------------------------------------------------------------
def test_dhash_dupe_and_distinct():
    as_jpg = Image.open(io.BytesIO(_img_bytes(grad='ltr')))
    as_png = Image.open(io.BytesIO(_img_bytes(grad='ltr', fmt='PNG')))
    other = Image.open(io.BytesIO(_img_bytes(grad='rtl')))
    assert svc._hamming(svc._dhash(as_jpg), svc._dhash(as_png)) <= svc.SCRAPE_DHASH_MAX_DISTANCE
    assert svc._hamming(svc._dhash(as_jpg), svc._dhash(other)) > svc.SCRAPE_DHASH_MAX_DISTANCE


# --- Service : happy path + filtres ------------------------------------------
def test_scrape_import_happy_path(app):
    with app.app_context():
        c = _concept()
        by_url = {'http://x/a.jpg': _img_bytes(grad='ltr'),
                  'http://x/b.jpg': _img_bytes(grad='rtl')}
        with patch.object(svc, '_download_scrape_item', _fake_downloader(by_url)):
            res = svc.scrape_import_urls(LOCAL_USER, c.id, [_item('http://x/a.jpg'), _item('http://x/b.jpg')])
        assert res['imported'] == 2
        assert all(v == 0 for v in res['skipped'].values()), res['skipped']
        rows = FaceDatasetImage.query.filter_by(dataset_id=c.id).all()
        assert len(rows) == 2 and all(r.source == 'import' and r.status == 'keep' for r in rows)


def test_scrape_import_filters(app):
    with app.app_context():
        c = _concept()
        by_url = {
            'http://x/low.jpg': _img_bytes(700, 500),          # côté court < 768
            'http://x/wide.jpg': _img_bytes(3000, 800),        # ratio 3.75
            'http://x/dup1.jpg': _img_bytes(grad='ltr'),
            'http://x/dup2.jpg': _img_bytes(grad='ltr', fmt='PNG'),
            'http://x/dead.jpg': None,                          # échec réseau
        }
        urls = ['low', 'wide', 'dup1', 'dup2', 'dead']
        items = [_item(f'http://x/{u}.jpg') for u in urls]
        with patch.object(svc, '_download_scrape_item', _fake_downloader(by_url)):
            res = svc.scrape_import_urls(LOCAL_USER, c.id, items)
        assert res['imported'] == 1  # dup1 seul survit
        assert res['skipped'] == {'duplicates': 1, 'low_res': 1, 'extreme_ratio': 1,
                                  'not_image': 0, 'errors': 1}


def test_scrape_import_dedup_vs_existing(app):
    with app.app_context():
        c = _concept()
        data = _img_bytes(grad='ltr')
        ids, _ = svc.import_images(LOCAL_USER, c.id, [data], crop=False)  # déjà présente (webp normalisé)
        assert len(ids) == 1
        with patch.object(svc, '_download_scrape_item', _fake_downloader({'http://x/again.jpg': data})):
            res = svc.scrape_import_urls(LOCAL_USER, c.id, [_item('http://x/again.jpg')])
        assert res['imported'] == 0 and res['skipped']['duplicates'] == 1


def test_scrape_import_maps_not_image(app):
    with app.app_context():
        c = _concept()

        def _dl(item):
            return ('not_image', None)
        with patch.object(svc, '_download_scrape_item', _dl):
            res = svc.scrape_import_urls(LOCAL_USER, c.id, [_item('http://x/a.gif')])
        assert res['imported'] == 0 and res['skipped']['not_image'] == 1


def test_scrape_import_ignores_itemless(app):
    with app.app_context():
        c = _concept()
        res = svc.scrape_import_urls(LOCAL_USER, c.id, [{}, {'title': 'x'}, None])
        assert res['imported'] == 0


# --- Downloader (SSRF + type) ------------------------------------------------
def test_download_scrape_item_rejects_private_host(app):
    with app.app_context():
        # _validate_public_http_url doit refuser un host qui résout en IP privée.
        reason, data = svc._download_scrape_item({'url': 'http://127.0.0.1/a.jpg'})
        assert reason == 'errors' and data is None


# --- Route -------------------------------------------------------------------
def test_route_rejects_non_concept(client, app):
    with app.app_context():
        p = svc.create_dataset(LOCAL_USER, 'Emma', 'z')  # character
        did = p.id
    resp = client.post(f'/api/dataset/{did}/scrape-import',
                       json={'items': [{'url': 'http://x/a.jpg'}]})
    assert resp.status_code == 400 and 'concept' in resp.get_json()['error']


def test_route_validates_payload_and_cap(client, app):
    with app.app_context():
        c = _concept()
        did = c.id
    assert client.post(f'/api/dataset/{did}/scrape-import', json={}).status_code == 400
    too_many = [{'url': f'http://x/{i}.jpg'} for i in range(svc.SCRAPE_IMPORT_MAX + 1)]
    assert client.post(f'/api/dataset/{did}/scrape-import',
                       json={'items': too_many}).status_code == 400


def test_route_happy_path_reports_counters(client, app):
    with app.app_context():
        c = _concept()
        did = c.id
    fake = {'imported': 2, 'skipped': {'duplicates': 1, 'low_res': 0, 'extreme_ratio': 0,
                                       'not_image': 0, 'errors': 0}}
    with patch('app.routes.datasets.svc.scrape_import_urls', return_value=fake) as m:
        resp = client.post(f'/api/dataset/{did}/scrape-import',
                           json={'items': [{'url': 'http://x/a.jpg'}, {'url': 'http://x/b.jpg'}]})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] is True and body['imported'] == 2 and body['skipped']['duplicates'] == 1
    m.assert_called_once()
