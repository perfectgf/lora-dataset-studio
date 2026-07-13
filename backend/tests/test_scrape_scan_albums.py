"""Scans de listings à galeries (PornPics catégorie/tag/recherche) : covers par
défaut, albums complets sur option.

Un scan mot-clé/catégorie remonte par défaut UNE image par galerie — la cover,
celle qui matche le mot-clé ; le reste de l'album est souvent hors-sujet. Le flag
`include_albums` (case « Scan full albums » de l'UI, transmis par /scan) rétablit
la plongée intégrale. L'URL directe d'une galerie (/galleries/...) n'est pas
concernée : ses médias sont top-level, `per_album` ne borne que la récursion.

Tout est mocké — aucun appel réseau ni process gallery-dl."""
from app.scrape.sources import gdl
from app.scrape.sources.base import Match
from app.scrape.sources.image_sites import PornpicsSource


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


def test_pornpics_scan_maps_include_albums_to_per_album(monkeypatch):
    seen = {}

    def fake_enum(url, **kw):
        seen.clear()
        seen.update(kw)
        return [], None
    monkeypatch.setattr(gdl, 'enumerate', fake_enum)
    src = PornpicsSource()
    m = Match(url='https://www.pornpics.com/flexible/')
    m.page = 0
    src.scan(m)                                  # défaut → covers seulement
    assert seen['per_album'] == 1
    m.include_albums = True
    src.scan(m)                                  # option cochée → plongée intégrale
    assert seen['per_album'] is None


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
