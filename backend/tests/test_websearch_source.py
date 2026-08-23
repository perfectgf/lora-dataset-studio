"""Recherche d'images par mot-clé (source websearch).

La bibliothèque `ddgs` n'est JAMAIS appelée : les tests remplacent la seule
indirection `_images`."""
import pytest

from app.scrape.sources import websearch
from app.scrape.sources.websearch import WebSearchSource


_RESULT = {
    'title': 'Curly hair portrait',
    'image': 'https://cdn.example.test/photo.jpg',
    'thumbnail': 'https://cdn.example.test/thumb.jpg',
    'url': 'https://blog.example.test/post/42',
    'height': 1200, 'width': 800, 'source': 'example',
}


def _spy_images(monkeypatch, results=None, raises=None):
    seen = {}

    def fake(**kw):
        seen.update(kw)
        if raises is not None:
            raise raises
        return results if results is not None else []
    monkeypatch.setattr(websearch, '_images', fake)
    return seen


# --- match() -------------------------------------------------------------------
def test_match_reads_the_keyword_and_safesearch_from_the_url():
    m = WebSearchSource().match(
        'https://duckduckgo.com/?q=curly+hair&iax=images&ia=images&kp=-2')
    assert m is not None
    assert m.query == 'curly hair'
    assert m.safesearch == 'off'


def test_match_honours_the_strict_safesearch_flag():
    m = WebSearchSource().match('https://duckduckgo.com/?q=portrait&kp=1')
    assert m.safesearch == 'on'


@pytest.mark.parametrize('url', [
    'https://duckduckgo.com/?q=&iax=images',      # mot-clé vide
    'https://duckduckgo.com/',                     # pas de q
    'https://duckduckgo.com.evil.test/?q=x',       # suffixe d'hôte
    'https://example.test/?q=x',                   # autre site
])
def test_match_refuses_anything_else(url):
    assert WebSearchSource().match(url) is None


# --- scan() --------------------------------------------------------------------
def test_scan_maps_results_to_the_common_schema(monkeypatch):
    seen = _spy_images(monkeypatch, results=[_RESULT])
    m = WebSearchSource().match('https://duckduckgo.com/?q=curly+hair&kp=-2')
    m.page = 0

    items, err = WebSearchSource().scan(m)

    assert err is None
    assert items == [{
        'url': 'https://cdn.example.test/photo.jpg',      # média DIRECT
        'title': 'Curly hair portrait',
        'thumbnail': 'https://cdn.example.test/thumb.jpg',
        'type': 'image',
        'platform': 'websearch',
        'source_url': 'https://blog.example.test/post/42',  # provenance
    }]
    assert seen['query'] == 'curly hair'
    assert seen['safesearch'] == 'off'
    assert seen['page'] == 1        # ddgs compte à partir de 1, match.page de 0


def test_scan_asks_for_the_next_page(monkeypatch):
    seen = _spy_images(monkeypatch, results=[_RESULT])
    m = WebSearchSource().match('https://duckduckgo.com/?q=portrait')
    m.page = 3

    WebSearchSource().scan(m)

    assert seen['page'] == 4


def test_scan_drops_entries_without_a_usable_https_image(monkeypatch):
    _spy_images(monkeypatch, results=[
        {'image': 'http://cdn.example.test/insecure.jpg'},        # pas https
        {'image': 'https://user:pw@cdn.example.test/creds.jpg'},  # credentials
        {'title': 'no image at all'},
        _RESULT,
    ])
    m = WebSearchSource().match('https://duckduckgo.com/?q=portrait')
    m.page = 0

    items, err = WebSearchSource().scan(m)

    assert err is None
    assert [it['url'] for it in items] == ['https://cdn.example.test/photo.jpg']


def test_an_empty_search_is_empty_not_an_error(monkeypatch):
    """Zéro résultat pour un mot-clé est un fait, pas une panne."""
    _spy_images(monkeypatch, results=[])
    m = WebSearchSource().match('https://duckduckgo.com/?q=zzzznotathing')
    m.page = 0

    items, err = WebSearchSource().scan(m)

    assert err is None and items == []


def test_a_failing_library_is_reported_and_never_raises(monkeypatch):
    _spy_images(monkeypatch, raises=RuntimeError('ratelimit'))
    m = WebSearchSource().match('https://duckduckgo.com/?q=portrait')
    m.page = 0

    items, err = WebSearchSource().scan(m)

    assert items is None
    assert 'failed' in err.lower() and 'ratelimit' in err


def test_none_from_the_library_is_reported_not_treated_as_empty(monkeypatch):
    """Un blocage doux (ratelimit, filtre) peut renvoyer None sans lever.
    Ça reste une panne, pas une liste vide."""
    def fake(**kw):
        return None
    monkeypatch.setattr(websearch, '_images', fake)
    m = WebSearchSource().match('https://duckduckgo.com/?q=portrait')
    m.page = 0

    items, err = WebSearchSource().scan(m)

    assert items is None
    assert err is not None and 'no data' in err.lower()


def test_a_lazy_iterator_raising_mid_iteration_is_reported_not_raised(monkeypatch):
    """`ddgs` peut renvoyer un itérateur paresseux qui fait son I/O réseau à
    l'itération : si la compréhension de scan() était hors du try, l'exception
    s'échapperait et casserait le contrat « scan() ne lève jamais »."""
    def fake(**kw):
        def generator():
            yield _RESULT
            raise RuntimeError('connection reset mid-page')
        return generator()
    monkeypatch.setattr(websearch, '_images', fake)
    m = WebSearchSource().match('https://duckduckgo.com/?q=portrait')
    m.page = 0

    items, err = WebSearchSource().scan(m)

    assert items is None
    assert 'failed' in err.lower() and 'connection reset' in err


def test_a_missing_dependency_says_how_to_install_it(monkeypatch):
    """Le registry importe toutes les sources au démarrage : une dépendance
    optionnelle absente doit donner une consigne, jamais empêcher le boot."""
    _spy_images(monkeypatch, raises=ImportError('No module named ddgs'))
    m = WebSearchSource().match('https://duckduckgo.com/?q=portrait')
    m.page = 0

    items, err = WebSearchSource().scan(m)

    assert items is None
    assert 'requirements-scrape.txt' in err


# --- thumbnail fallback ---------------------------------------------------------
# `/api/scrape/thumb` fetches server-side and re-serves from our own https origin
# (`_validate_public_http_url` already accepts http there — no browser mixed-content
# concern), and other sources (gallery-dl) pass an unvalidated http thumbnail
# through unchanged. Requiring https here bought nothing but MORE frequent falls
# back to the full-size image on a picker page of 100+ results — bandwidth paid
# for no security this source's own sibling doesn't already skip.
def test_an_http_thumbnail_is_kept_rather_than_falling_back_to_the_full_image(monkeypatch):
    _spy_images(monkeypatch, results=[{**_RESULT, 'thumbnail': 'http://cdn.example.test/thumb.jpg'}])
    m = WebSearchSource().match('https://duckduckgo.com/?q=portrait')
    m.page = 0

    items, err = WebSearchSource().scan(m)

    assert err is None
    assert items[0]['thumbnail'] == 'http://cdn.example.test/thumb.jpg'


def test_a_missing_thumbnail_still_falls_back_to_the_full_image(monkeypatch):
    _spy_images(monkeypatch, results=[{**_RESULT, 'thumbnail': None}])
    m = WebSearchSource().match('https://duckduckgo.com/?q=portrait')
    m.page = 0

    items, err = WebSearchSource().scan(m)

    assert err is None
    assert items[0]['thumbnail'] == items[0]['url'] == 'https://cdn.example.test/photo.jpg'


def test_a_thumbnail_with_credentials_or_a_bad_scheme_still_falls_back(monkeypatch):
    _spy_images(monkeypatch, results=[
        {**_RESULT, 'thumbnail': 'https://user:pw@cdn.example.test/thumb.jpg'},
        {**_RESULT, 'image': 'https://cdn.example.test/photo2.jpg',
         'thumbnail': 'javascript:alert(1)'},
        {**_RESULT, 'image': 'https://cdn.example.test/photo3.jpg',
         'thumbnail': 'ftp://cdn.example.test/thumb.jpg'},
    ])
    m = WebSearchSource().match('https://duckduckgo.com/?q=portrait')
    m.page = 0

    items, err = WebSearchSource().scan(m)

    assert err is None
    assert [it['thumbnail'] for it in items] == [it['url'] for it in items]


def test_the_source_is_registered_ahead_of_the_universal_fallback():
    from app.scrape.sources import registry
    match = registry.resolve('https://duckduckgo.com/?q=portrait&iax=images')
    assert match is not None and match.source.name == 'websearch'
