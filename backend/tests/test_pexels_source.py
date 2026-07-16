"""Source Pexels via l'API officielle — aucun appel réseau réel."""
import requests
import pytest

from app.scrape.sources import pexels, registry
from app.scrape.sources.base import Match
from app.scrape.sources.pexels import PexelsSource
from app.scrape.validators import Platform, URLType, url_validator
from app.routes.scrape import MAX_SCAN_PAGE


_KEY = 'pexels-secret-test-value'


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv('PEXELS_API_KEY', _KEY)


class _Response:
    def __init__(self, status=200, payload=None, headers=None, json_error=False):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise ValueError('not json')
        return self._payload


def _photo(photo_id=123, **overrides):
    photo = {
        'id': photo_id,
        'url': f'https://www.pexels.com/photo/portrait-{photo_id}/',
        'photographer': 'Ada Lens',
        'photographer_url': 'https://www.pexels.com/@ada-lens/',
        'alt': 'Studio portrait',
        'src': {
            'original': f'https://images.pexels.com/photos/{photo_id}/original.jpeg',
            'medium': f'https://images.pexels.com/photos/{photo_id}/medium.jpeg',
        },
    }
    photo.update(overrides)
    return photo


def _mock_get(monkeypatch, response):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return response

    monkeypatch.setattr(pexels.requests, 'get', fake_get)
    return calls


def _scan(url, page=0):
    match = Match(url=url, page=page)
    items, err = PexelsSource().scan(match)
    return match, items, err


@pytest.mark.parametrize(('url', 'url_type'), [
    ('https://www.pexels.com/search/film%20portrait/', URLType.LISTING),
    ('https://www.pexels.com/en-us/search/film%20portrait/', URLType.LISTING),
    ('https://www.pexels.com/fr-fr/chercher/portrait-femme/', URLType.LISTING),
    ('https://pexels.com/collections/editorial-abc123/', URLType.LISTING),
    ('https://pexels.com/en-us/collections/editorial-abc123/', URLType.LISTING),
    ('https://pexels.com/fr-fr/collections/editorial-abc123/', URLType.LISTING),
    ('https://www.pexels.com/collections/ABC123', URLType.LISTING),
    ('https://www.pexels.com/photo/studio-portrait-123456/', URLType.POST),
    ('https://www.pexels.com/en-us/photo/studio-portrait-123456/', URLType.POST),
    ('https://www.pexels.com/fr-fr/photo/portrait-studio-123456/', URLType.POST),
    ('https://pexels.com/photo/123456', URLType.POST),
])
def test_validates_routes_supported_by_official_api(url, url_type):
    assert url_validator.detect_platform(url) == Platform.PEXELS
    result = url_validator.validate_url(url)
    assert result.is_valid is True
    assert result.platform == Platform.PEXELS
    assert result.url_type == url_type


@pytest.mark.parametrize('path', [
    '/', '/search/', '/search', '/search/portrait/extra/', '/collections/',
    '/collections/foo-%2Fsearch/', '/collections/foo-%3Fbar/',
    '/collections/foo-bad.id/', '/collections/bad_id/',
    '/@photographer-12345/', '/photo/',
    '/photo/portrait/', '/videos/',
    '/en-us/chercher/portrait/', '/fr-fr/search/portrait/',
    '/de-de/search/portrait/', '/es-es/chercher/portrait/',
    '/de-de/photo/studio-123/', '/it-it/collections/editorial-abc123/',
    '/fr-fr/videos/portrait/', '/fr-fr/inconnue/portrait/',
    '/en-us/search/portrait/extra/', '/fr-fr/collections/',
    '/en-us/photo/portrait/', '/fr-fr/chercher/portrait%ZZ/',
])
def test_rejects_routes_not_supported_by_official_api(path):
    result = url_validator.validate_url('https://www.pexels.com' + path)
    assert result.is_valid is False
    assert result.platform == Platform.PEXELS
    assert result.url_type == URLType.UNKNOWN
    assert 'API officielle' in result.error
    assert any('/search/' in suggestion for suggestion in result.suggestions)
    if path.startswith('/@'):
        assert any('profils publics' in suggestion for suggestion in result.suggestions)


@pytest.mark.parametrize('path', [
    '/en-us/chercher/portrait/',
    '/fr-fr/search/portrait/',
    '/de-de/search/portrait/',
    '/fr-fr/inconnue/portrait/',
    '/fr-fr/chercher/portrait/extra/',
])
def test_api_target_never_reinterprets_unsupported_localized_routes(path):
    assert pexels._target_for('https://www.pexels.com' + path, 0) is None


def test_domain_and_userinfo_guards_reject_lookalikes():
    assert url_validator.detect_platform(
        'https://www.pexels.com.evil.example/search/x/') != Platform.PEXELS
    assert url_validator.detect_platform(
        'https://pexels.com@evil.example/search/x/') != Platform.PEXELS

    result = url_validator.validate_url('https://evil.pexels.com/search/x/')
    assert result.is_valid is False and 'Domaine invalide' in result.error
    result = url_validator.validate_url(
        'https://evil.example@www.pexels.com/search/x/')
    assert result.is_valid is False and 'informations utilisateur' in result.error


def test_registry_exposes_authenticated_sfw_image_source():
    match = registry.resolve('https://www.pexels.com/search/portrait/')
    assert match is not None and match.source.name == 'pexels'
    assert match.source.category == 'image'
    assert match.source.paginated is True
    assert match.source.capabilities.needs_auth is True
    assert match.source.capabilities.can_enumerate_profile is False
    assert match.source.capabilities.media_kinds == frozenset({'image'})


@pytest.mark.parametrize(('url', 'query', 'locale'), [
    ('pexels.com/search/portrait/', 'portrait', None),
    ('pexels.com/en-us/search/film%20portrait/', 'film portrait', 'en-US'),
    ('pexels.com/fr-fr/chercher/portrait%20femme/', 'portrait femme', 'fr-FR'),
])
def test_schemeless_search_is_normalized_and_localized_before_scan(
        monkeypatch, url, query, locale):
    calls = _mock_get(monkeypatch, _Response(payload={
        'photos': [], 'next_page': None,
    }))

    match = registry.resolve(url)
    assert match is not None
    assert match.url == 'https://' + url

    items, err = match.source.scan(match)
    assert (items, err) == ([], None)
    assert calls[0][0] == 'https://api.pexels.com/v1/search'
    params = calls[0][1]['params']
    assert params['query'] == query
    if locale:
        assert params['locale'] == locale
    else:
        assert 'locale' not in params


@pytest.mark.parametrize(('url', 'endpoint'), [
    ('https://www.pexels.com/en-us/collections/editorial-abc123/',
     'https://api.pexels.com/v1/collections/abc123'),
    ('https://www.pexels.com/fr-fr/photo/studio-portrait-789/',
     'https://api.pexels.com/v1/photos/789'),
])
def test_localized_resource_routes_map_to_fixed_api_endpoints(
        monkeypatch, url, endpoint):
    payload = {'media': [], 'next_page': None} if '/collections/' in url else _photo()
    calls = _mock_get(monkeypatch, _Response(payload=payload))

    _match, items, err = _scan(url)

    assert err is None and items is not None
    assert calls[0][0] == endpoint


def test_search_decodes_query_maps_attribution_and_uses_raw_auth(monkeypatch):
    response = _Response(payload={
        'photos': [_photo(), {'id': 999, 'src': {}}],
        # État uniquement : le client ne doit jamais suivre cette URL.
        'next_page': 'https://evil.example/steal-authorization',
    })
    calls = _mock_get(monkeypatch, response)

    match, items, err = _scan(
        'https://www.pexels.com/search/film%20portrait/', page=1)

    assert err is None and match.paginated is True
    assert len(calls) == 1
    endpoint, kwargs = calls[0]
    assert endpoint == 'https://api.pexels.com/v1/search'
    assert kwargs['params'] == {'query': 'film portrait', 'page': 2, 'per_page': 80}
    assert kwargs['headers'] == {'Authorization': _KEY}
    assert not kwargs['headers']['Authorization'].startswith('Bearer ')
    assert kwargs['timeout'] == 20
    assert kwargs['allow_redirects'] is False
    assert items == [{
        'url': 'https://images.pexels.com/photos/123/original.jpeg',
        'thumbnail': 'https://images.pexels.com/photos/123/medium.jpeg',
        'title': 'Studio portrait',
        'type': 'image',
        'platform': 'pexels',
        'source_url': 'https://www.pexels.com/photo/portrait-123/',
        'photographer': 'Ada Lens',
        'photographer_url': 'https://www.pexels.com/@ada-lens/',
    }]


def test_search_orientation_is_allowlisted_with_locale_and_pagination(monkeypatch):
    calls = _mock_get(monkeypatch, _Response(payload={
        'photos': [], 'next_page': 'https://api.pexels.com/v1/search?page=4',
    }))
    match, items, err = _scan(
        'https://www.pexels.com/fr-fr/chercher/portrait%20studio/'
        '?orientation=portrait&color=red&page=999&per_page=1&locale=xx-XX',
        page=2,
    )

    assert err is None and items == [] and match.paginated is True
    endpoint, kwargs = calls[0]
    assert endpoint == 'https://api.pexels.com/v1/search'
    assert kwargs['params'] == {
        'query': 'portrait studio',
        'page': 3,
        'per_page': 80,
        'locale': 'fr-FR',
        'orientation': 'portrait',
    }


@pytest.mark.parametrize('query', [
    'orientation=wide',
    'orientation=',
    'orientation=PORTRAIT',
    'orientation=portrait&orientation=landscape',
    'orientation=portrait%0D%0Apage%3D999',
])
def test_invalid_or_repeated_orientation_never_reaches_api(monkeypatch, query):
    monkeypatch.setattr(
        pexels.requests, 'get',
        lambda *args, **kwargs: pytest.fail('invalid orientation reached requests'),
    )
    target = pexels._target_for(
        f'https://www.pexels.com/search/portrait/?{query}', 0)
    assert target is None

    _match, items, err = _scan(
        f'https://www.pexels.com/search/portrait/?{query}')
    assert items is None and 'format' in err


def test_search_last_page_disables_pagination(monkeypatch):
    _mock_get(monkeypatch, _Response(payload={'photos': [], 'next_page': None}))
    match, items, err = _scan('https://www.pexels.com/search/portrait/')
    assert (items, err, match.paginated) == ([], None, False)


@pytest.mark.parametrize(('public_path', 'api_id'), [
    ('editorial-abc123', 'abc123'),
    ('ABC123', 'ABC123'),
])
def test_collection_uses_id_suffix_photo_filter_and_page(monkeypatch, public_path, api_id):
    calls = _mock_get(monkeypatch, _Response(payload={
        'media': [_photo(456)], 'next_page': '',
    }))
    match, items, err = _scan(
        f'https://www.pexels.com/collections/{public_path}/', page=2)
    assert err is None and len(items) == 1 and match.paginated is False
    endpoint, kwargs = calls[0]
    assert endpoint == f'https://api.pexels.com/v1/collections/{api_id}'
    assert kwargs['params'] == {'type': 'photos', 'page': 3, 'per_page': 80}


def test_encoded_collection_path_never_reaches_requests(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('invalid collection must not reach requests')

    monkeypatch.setattr(pexels.requests, 'get', fail)
    result = url_validator.validate_url(
        'https://www.pexels.com/collections/foo-%2Fsearch/')
    assert result.is_valid is False
    assert registry.resolve(result.original_url) is None


def test_single_photo_endpoint_and_no_pagination(monkeypatch):
    calls = _mock_get(monkeypatch, _Response(payload=_photo(789)))
    match, items, err = _scan(
        'https://www.pexels.com/photo/studio-portrait-789/')
    assert err is None and len(items) == 1 and match.paginated is False
    assert calls[0][0] == 'https://api.pexels.com/v1/photos/789'
    assert calls[0][1]['params'] is None


def test_single_photo_later_page_is_empty_without_api_call(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('single photo must not repeat on page > 0')

    monkeypatch.setattr(pexels.requests, 'get', fail)
    match, items, err = _scan('https://www.pexels.com/photo/portrait-123/', page=1)
    assert (items, err, match.paginated) == ([], None, False)


def test_missing_key_is_actionable_and_never_calls_api(monkeypatch):
    monkeypatch.delenv('PEXELS_API_KEY')
    monkeypatch.setattr(pexels.requests, 'get', lambda *a, **k: pytest.fail('network'))
    _match, items, err = _scan('https://www.pexels.com/search/portrait/')
    assert items is None and 'PEXELS_API_KEY' in err and 'requise' in err


@pytest.mark.parametrize('status', [401, 403])
def test_rejected_key_is_actionable_without_leaking_it(monkeypatch, status):
    _mock_get(monkeypatch, _Response(status=status))
    _match, items, err = _scan('https://www.pexels.com/search/portrait/')
    assert items is None and f'HTTP {status}' in err and 'clé API refusée' in err
    assert _KEY not in err


def test_collection_404_explains_api_visibility_limit(monkeypatch):
    _mock_get(monkeypatch, _Response(status=404))
    _match, items, err = _scan(
        'https://www.pexels.com/collections/public-looking-abc123/')
    assert items is None
    assert 'collection introuvable ou non accessible' in err
    assert 'publiques arbitraires' in err
    assert _KEY not in err


def test_rate_limit_includes_reset_without_key(monkeypatch):
    _mock_get(monkeypatch, _Response(
        status=429, headers={'X-Ratelimit-Reset': '1760000000'}))
    _match, items, err = _scan('https://www.pexels.com/search/portrait/')
    assert items is None and 'HTTP 429' in err and '1760000000' in err
    assert _KEY not in err


@pytest.mark.parametrize('status', [500, 503])
def test_server_errors_are_actionable(monkeypatch, status):
    _mock_get(monkeypatch, _Response(status=status))
    _match, items, err = _scan('https://www.pexels.com/search/portrait/')
    assert items is None and f'HTTP {status}' in err and 'indisponible' in err


@pytest.mark.parametrize(('exc', 'message'), [
    (requests.Timeout(), 'délai dépassé'),
    (requests.RequestException(), 'erreur réseau'),
])
def test_request_failures_never_raise_or_leak_key(monkeypatch, exc, message):
    def fail(*args, **kwargs):
        raise exc

    monkeypatch.setattr(pexels.requests, 'get', fail)
    _match, items, err = _scan('https://www.pexels.com/search/portrait/')
    assert items is None and message in err and _KEY not in err


def test_redirect_is_refused_without_second_request_or_key_leak(monkeypatch):
    calls = _mock_get(monkeypatch, _Response(
        status=302, headers={'Location': 'https://evil.example/steal'}))
    _match, items, err = _scan('https://www.pexels.com/search/portrait/')
    assert items is None and 'redirection' in err and _KEY not in err
    assert len(calls) == 1 and calls[0][1]['allow_redirects'] is False


@pytest.mark.parametrize('response', [
    _Response(payload=None, json_error=True),
    _Response(payload=[]),
])
def test_invalid_json_or_top_level_schema_is_actionable(monkeypatch, response):
    _mock_get(monkeypatch, response)
    _match, items, err = _scan('https://www.pexels.com/search/portrait/')
    assert items is None and ('JSON' in err or 'schéma' in err)


@pytest.mark.parametrize('payload', [
    {},
    {'photos': [{'src': {}}], 'next_page': None},
    {'photos': [_photo(url='javascript:alert(1)')], 'next_page': None},
    {'photos': [_photo(src={
        'original': 'https://evil.example/original.jpeg',
        'medium': 'https://images.pexels.com/photos/1/medium.jpeg',
    })], 'next_page': None},
    {'photos': [_photo(photographer_url='https://evil.example/person')],
     'next_page': None},
])
def test_incomplete_or_unsafe_photo_schema_is_rejected(monkeypatch, payload):
    _mock_get(monkeypatch, _Response(payload=payload))
    _match, items, err = _scan('https://www.pexels.com/search/portrait/')
    assert items is None and 'schéma' in err


def test_scan_route_returns_api_metadata_and_attribution(client, monkeypatch):
    calls = _mock_get(monkeypatch, _Response(payload={
        'photos': [_photo()], 'next_page': 'https://api.pexels.com/v1/search?page=3',
    }))
    response = client.post('/api/scrape/scan', json={
        'url': 'https://www.pexels.com/search/portrait/', 'page': 1,
    })
    assert response.status_code == 200
    body = response.get_json()
    assert body['platform'] == 'pexels' and body['category'] == 'image'
    assert body['page'] == 1 and body['paginated'] is True and body['count'] == 1
    assert body['items'][0]['url'].endswith('/original.jpeg')
    assert body['items'][0]['thumbnail'].endswith('/medium.jpeg')
    assert body['items'][0]['photographer'] == 'Ada Lens'
    assert body['items'][0]['source_url'].startswith('https://www.pexels.com/photo/')
    assert calls[0][1]['params']['page'] == 2


def test_scan_route_clamps_page_and_stops_at_hard_limit(client, monkeypatch):
    calls = _mock_get(monkeypatch, _Response(payload={
        'photos': [_photo()],
        'next_page': 'https://api.pexels.com/v1/search?page=52',
    }))
    response = client.post('/api/scrape/scan', json={
        'url': 'https://www.pexels.com/search/portrait/',
        'page': MAX_SCAN_PAGE + 10,
    })

    assert response.status_code == 200
    body = response.get_json()
    assert calls[0][1]['params']['page'] == MAX_SCAN_PAGE + 1
    assert body['page'] == MAX_SCAN_PAGE
    assert body['paginated'] is False


def test_scan_route_hides_pagination_for_single_photo(client, monkeypatch):
    _mock_get(monkeypatch, _Response(payload=_photo()))
    response = client.post('/api/scrape/scan', json={
        'url': 'https://www.pexels.com/photo/portrait-123/', 'page': 0,
    })
    assert response.status_code == 200
    body = response.get_json()
    assert body['platform'] == 'pexels' and body['paginated'] is False
