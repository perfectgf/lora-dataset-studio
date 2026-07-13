"""Source Sex.com (pinboard : 1 pin = 1 image, pas d'albums).

La recherche mot-clé (/pics?search=…, /en/pics?search=…, /search/pics?query=…,
/pics/<tag>) passe par l'API JSON du site — même principe que les covers
PornPics : l'image remontée EST celle qui matche le mot-clé, 1 requête par page.
Pins/boards/users sont délégués à gallery-dl avec une fenêtre --range courte.

Tout est mocké — aucun appel réseau ni process gallery-dl."""
from app.scrape.sources import gdl, sexcom
from app.scrape.sources.base import Match
from app.scrape.sources.sexcom import SexcomSource, _search_params_for
from app.scrape.validators import Platform, url_validator


# --- _search_params_for : routage d'URL (pur) ---------------------------------
def test_search_url_forms_map_to_api_params():
    for url in ('https://www.sex.com/en/pics?search=flexible',       # exemple utilisateur
                'https://www.sex.com/pics?search=flexible',
                'https://www.sex.com/search/pics?query=flexible',
                'https://sex.com/fr/search/pics?query=flexible'):
        p = _search_params_for(url)
        assert p == {'search': 'flexible', 'order': 'likeCount',
                     'sexual-orientation': 'straight'}, url


def test_tag_path_becomes_search_and_params_pass_through():
    p = _search_params_for('https://www.sex.com/pics/flexible-girls?order=latest')
    assert p['search'] == 'flexible girls' and p['order'] == 'latest'


def test_gif_and_video_searches_are_refused_clearly():
    assert _search_params_for('https://www.sex.com/gifs?search=x') == 'wrong-kind'
    assert _search_params_for('https://www.sex.com/en/search/videos?query=x') == 'wrong-kind'


def test_pin_board_and_user_urls_go_to_gallery_dl():
    assert _search_params_for('https://www.sex.com/en/pics/12345') is None   # pin (id numérique)
    assert _search_params_for('https://www.sex.com/pin/12345/') is None
    assert _search_params_for('https://www.sex.com/user/someone/board-name/') is None
    assert _search_params_for('https://www.sex.com/pics') is None            # listing sans mot-clé


# --- scan : API mockée ---------------------------------------------------------
def _api_payload(n, pages=3):
    return {'data': [{'id': i, 'uri': f'/images/pinporn/2024/01/0{i}/{i}.jpg',
                      'title': f'pin {i}'} for i in range(1, n + 1)],
            'paging': {'numberOfPages': pages, 'page': 1, 'limit': 40}}


def test_scan_search_maps_pins_to_cdn_items(monkeypatch):
    seen = {}

    def fake_json(params, page):
        seen.update(params=params, page=page)
        return _api_payload(2)
    monkeypatch.setattr(sexcom, '_search_json', fake_json)
    m = Match(url='https://www.sex.com/en/pics?search=flexible')
    items, err = SexcomSource().scan(m)
    assert err is None
    assert seen['page'] == 1                      # match.page 0-based → API 1-based
    assert seen['params']['search'] == 'flexible'
    assert items[0]['url'] == 'https://imagex1.sx.cdn.live/images/pinporn/2024/01/01/1.jpg'
    assert items[0]['type'] == 'image' and items[0]['platform'] == 'sexcom'
    assert items[0]['title'] == 'pin 1'


def test_scan_past_last_page_returns_empty(monkeypatch):
    monkeypatch.setattr(sexcom, '_search_json', lambda params, page: _api_payload(2, pages=3))
    m = Match(url='https://www.sex.com/pics?search=x')
    m.page = 3                                    # API page 4 > numberOfPages 3
    assert SexcomSource().scan(m) == ([], None)


def test_scan_gif_url_returns_actionable_error(monkeypatch):
    items, err = SexcomSource().scan(Match(url='https://www.sex.com/gifs?search=x'))
    assert items is None and '/pics' in err


def test_scan_api_failure_never_raises(monkeypatch):
    def boom(params, page):
        raise RuntimeError('api down')
    monkeypatch.setattr(sexcom, '_search_json', boom)
    items, err = SexcomSource().scan(Match(url='https://www.sex.com/pics?search=x'))
    assert items is None and 'api down' in err


def test_scan_pin_url_delegates_to_gdl_with_window(monkeypatch):
    seen = {}

    def fake_enum(url, **kw):
        seen.update(kw, url=url)
        return [{'url': 'https://imagex1.sx.cdn.live/x.jpg', 'title': '', 'thumbnail': None,
                 'type': 'image', 'platform': 'sexcom'}], None
    monkeypatch.setattr(sexcom.gdl, 'enumerate', fake_enum)
    m = Match(url='https://www.sex.com/pin/12345/')
    m.page = 1
    items, err = SexcomSource().scan(m)
    assert err is None and len(items) == 1
    assert seen['image_range'] == '13-24'         # fenêtre de 12 par page

# NB : sexcom.gdl est le MÊME module que app.scrape.sources.gdl — le monkeypatch
# ci-dessus suffit ; l'import en tête ne sert qu'à documenter la dépendance.
_ = gdl


# --- plateforme & registry -----------------------------------------------------
def test_detect_platform_and_registry_resolution():
    assert url_validator.detect_platform('https://www.sex.com/en/pics?search=x') == Platform.SEXCOM
    assert url_validator.detect_platform('https://sex.com/pin/1/') == Platform.SEXCOM
    assert url_validator.detect_platform('https://notsex.com/pics') != Platform.SEXCOM
    from app.scrape.sources import registry
    match = registry.resolve('https://www.sex.com/en/pics?search=flexible')
    assert match is not None and match.source.name == 'sexcom'
    assert match.source.category == 'image' and match.source.paginated is True
