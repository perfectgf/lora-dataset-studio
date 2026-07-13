"""Scans de listings à galeries (PornPics catégorie/tag/recherche) : covers par
défaut, albums complets sur option.

Un scan mot-clé/catégorie remonte par défaut LA VIGNETTE que la page affiche pour
chaque galerie — l'image choisie par le site comme représentative du mot-clé
(jamais la 1re de l'album : _009_, _113_… observé en réel) — via le parse
HTML/AJAX du listing, sans gallery-dl. Le flag `include_albums` (case « Scan
full albums » de l'UI, transmis par /scan) rétablit la plongée intégrale
gallery-dl. Si le parse covers échoue (layout changé), repli gallery-dl borné à
1 image/album. L'URL directe d'une galerie (/galleries/...) n'est pas concernée.

Tout est mocké — aucun appel réseau ni process gallery-dl."""
import pytest

from app.scrape.sources import gdl, image_sites
from app.scrape.sources.base import Match
from app.scrape.sources.image_sites import PornpicsSource, _covers_scan, _full_size


def _mock_gdl_runs(monkeypatch):
    """_run_simulate factice : une « catégorie » de 2 albums, 5 images chacun.
    Retourne la liste des appels (url + image_range) pour inspection."""
    calls = []

    def fake(url, max_items, cookies, extra_opts, image_range=None):
        calls.append({'url': url, 'image_range': image_range})
        if 'category' in url:
            return [[6, 'https://x/album1/', {}], [6, 'https://x/album2/', {}]], None
        n = int((image_range or '1-99').split('-')[1])
        return [[3, f'{url}img{i}.jpg', {'extension': 'jpg'}]
                for i in range(1, 6)][:n], None
    monkeypatch.setattr(gdl, '_run_simulate', fake)
    return calls


def test_enumerate_per_album_1_returns_one_cover_per_album(monkeypatch):
    calls = _mock_gdl_runs(monkeypatch)
    items, err = gdl.enumerate('https://x/category/', per_album=1)
    assert err is None
    assert [it['url'] for it in items] == ['https://x/album1/img1.jpg',
                                           'https://x/album2/img1.jpg']
    # La simulation de chaque album est elle-même bornée (--range 1-1) : gallery-dl
    # ne doit pas énumérer tout l'album pour n'en garder qu'une image.
    assert [c['image_range'] for c in calls[1:]] == ['1-1', '1-1']


def test_enumerate_without_per_album_dives_full_albums(monkeypatch):
    _mock_gdl_runs(monkeypatch)
    items, err = gdl.enumerate('https://x/category/')
    assert err is None
    assert len(items) == 10          # 2 albums × 5 images : comportement historique


# --- Mode covers : parse des vignettes du listing -----------------------------
_TILE_HTML = '''
<li><a class="rel-link" href="/galleries/flexible-girl-123/">
  <img src="1px.png" data-src="https://cdni.pornpics.com/460/1/2/123/123_009_ab.jpg" alt="Flexible girl">
</a></li>
<li><a class="rel-link" href="https://www.pornpics.com/galleries/splits-babe-456/">
  <img src='1px.png' data-src='https://cdni.pornpics.com/300/3/4/456/456_113_cd.jpg' alt='Splits babe'>
</a></li>
<li><a class="rel-link" href="/channels/whatever/">nav link sans data-src de tuile</a></li>
'''


def test_full_size_swaps_cdn_size_segment():
    assert _full_size('https://cdni.pornpics.com/460/1/2/123/123_009_ab.jpg') \
        == 'https://cdni.pornpics.com/1280/1/2/123/123_009_ab.jpg'
    assert _full_size('https://cdni.pornpics.com/300/3/4/456/456_113_cd.jpg') \
        == 'https://cdni.pornpics.com/1280/3/4/456/456_113_cd.jpg'


def test_covers_page0_returns_the_listing_thumbnails(monkeypatch):
    monkeypatch.setattr(image_sites, '_listing_html', lambda url: _TILE_HTML)
    items, err = _covers_scan('https://www.pornpics.com/flexible/', 0)
    assert err is None
    # la vignette VISIBLE (image _009_/_113_), pas la 1re image de l'album —
    # et seulement les tuiles galerie (le lien /channels/ est ignoré)
    assert [it['url'] for it in items] == [
        'https://cdni.pornpics.com/1280/1/2/123/123_009_ab.jpg',
        'https://cdni.pornpics.com/1280/3/4/456/456_113_cd.jpg']
    assert items[0]['thumbnail'].startswith('https://cdni.pornpics.com/460/')
    assert items[0]['title'] == 'Flexible girl'


def test_covers_next_pages_use_the_ajax_endpoint(monkeypatch):
    seen = {}

    def fake_json(url, offset):
        seen['offset'] = offset
        return [{'g_url': 'https://www.pornpics.com/galleries/x-789/', 'desc': 'X',
                 't_url_460': 'https://cdni.pornpics.com/460/5/6/789/789_042_ef.jpg'}]
    monkeypatch.setattr(image_sites, '_listing_json', fake_json)
    items, err = _covers_scan('https://www.pornpics.com/flexible/', 2)
    assert err is None and seen['offset'] == 40
    assert items[0]['url'] == 'https://cdni.pornpics.com/1280/5/6/789/789_042_ef.jpg'
    assert items[0]['title'] == 'X'


def test_covers_scan_signals_fallback_never_raises(monkeypatch):
    def boom(url):
        raise RuntimeError('site down')
    monkeypatch.setattr(image_sites, '_listing_html', boom)
    assert _covers_scan('https://www.pornpics.com/flexible/', 0) == (None, None)
    monkeypatch.setattr(image_sites, '_listing_html', lambda url: '<html>layout changé</html>')
    assert _covers_scan('https://www.pornpics.com/flexible/', 0) == (None, None)


# --- Routage PornpicsSource.scan : covers / albums / galerie directe / repli ---
@pytest.fixture()
def _spies(monkeypatch):
    seen = {'enum': None, 'covers': 0}

    def fake_enum(url, **kw):
        seen['enum'] = kw
        return [], None
    monkeypatch.setattr(gdl, 'enumerate', fake_enum)
    monkeypatch.setattr(image_sites, '_listing_html', lambda url: seen.update(covers=seen['covers'] + 1) or _TILE_HTML)
    return seen


def test_pornpics_default_scan_serves_covers_without_gdl(_spies):
    m = Match(url='https://www.pornpics.com/flexible/')
    m.page = 0
    items, err = PornpicsSource().scan(m)
    assert err is None and len(items) == 2
    assert _spies['covers'] == 1 and _spies['enum'] is None   # gallery-dl jamais lancé


def test_pornpics_include_albums_dives_via_gdl(_spies):
    m = Match(url='https://www.pornpics.com/flexible/')
    m.page = 0
    m.include_albums = True
    PornpicsSource().scan(m)
    assert _spies['covers'] == 0                    # pas de parse covers
    assert _spies['enum']['per_album'] is None      # plongée intégrale


def test_pornpics_direct_gallery_url_bypasses_covers(_spies):
    m = Match(url='https://www.pornpics.com/galleries/flexible-girl-123/')
    m.page = 0
    PornpicsSource().scan(m)
    assert _spies['covers'] == 0                    # URL d'album → gallery-dl direct
    assert _spies['enum'] is not None


def test_pornpics_covers_failure_falls_back_to_bounded_gdl(monkeypatch, _spies):
    def boom(url):
        raise RuntimeError('site down')
    monkeypatch.setattr(image_sites, '_listing_html', boom)
    m = Match(url='https://www.pornpics.com/flexible/')
    m.page = 0
    items, err = PornpicsSource().scan(m)
    assert err is None
    assert _spies['enum']['per_album'] == 1         # repli borné : 1 image/album


def test_scan_route_passes_include_albums_to_match(client, monkeypatch):
    seen = {}

    def fake_scan(self, match):
        seen['include_albums'] = getattr(match, 'include_albums', None)
        return [{'url': 'https://cdni.pornpics.com/x.jpg', 'title': '',
                 'thumbnail': None, 'type': 'image', 'platform': 'pornpics'}], None
    monkeypatch.setattr(PornpicsSource, 'scan', fake_scan)
    r = client.post('/api/scrape/scan',
                    json={'url': 'https://www.pornpics.com/flexible/',
                          'include_albums': True})
    assert r.status_code == 200
    assert seen['include_albums'] is True
    r = client.post('/api/scrape/scan',
                    json={'url': 'https://www.pornpics.com/flexible/'})
    assert r.status_code == 200
    assert seen['include_albums'] is False       # défaut : covers seulement
