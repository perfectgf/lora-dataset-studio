"""Retrait produit de Coomer/Kemono/Bunkr/Cyberdrop (sites de dump/leak).

Ces 4 sources sont retirées : plus de module `sources/<name>.py`, plus
d'enregistrement dans le registry. `validators.py` garde uniquement assez
d'identification d'hôte pour renvoyer un refus EXPLICITE et NOMMÉ dès
`validate_url()`, avant toute résolution de source — jamais un crash, jamais
un repli silencieux vers le scraper générique (gallery-dl/yt-dlp supportent
nativement certains de ces sites en interne, donc le risque de contournement
est réel si on les laissait passer en Platform.GENERIC).

Tout est local/pur — aucun appel réseau ni process gallery-dl."""
import pytest

from app.scrape.sources import registry
from app.scrape.validators import Platform, url_validator


_REMOVED_URLS = [
    ('https://coomer.st/onlyfans/user/someone', Platform.COOMER),
    ('https://coomer.su/patreon/user/someone', Platform.COOMER),
    ('https://kemono.cr/fanbox/user/someone', Platform.KEMONO),
    ('https://kemono.su/gumroad/user/someone', Platform.KEMONO),
    ('https://cyberdrop.me/a/AbCdEfGh', Platform.CYBERDROP),
    ('https://cyberdrop.to/a/AbCdEfGh', Platform.CYBERDROP),
    ('https://bunkr.cr/a/AbCdEfGh', Platform.BUNKR),
    ('https://bunkrr.su/a/AbCdEfGh', Platform.BUNKR),
]


@pytest.mark.parametrize('url,expected_platform', _REMOVED_URLS)
def test_removed_platform_is_still_identified_by_host(url, expected_platform):
    # detect_platform() keeps recognizing the host so validate_url() can name
    # the refused source instead of a generic "unknown URL" message.
    assert url_validator.detect_platform(url) == expected_platform


@pytest.mark.parametrize('url,expected_platform', _REMOVED_URLS)
def test_removed_platform_url_is_refused_explicitly_and_never_generic(url, expected_platform):
    result = url_validator.validate_url(url)
    assert result.is_valid is False
    assert result.platform == expected_platform
    assert result.platform != Platform.GENERIC
    assert expected_platform.value in result.error
    assert 'removed' in result.error


@pytest.mark.parametrize('url,_platform', _REMOVED_URLS)
def test_removed_platform_has_no_registered_source(url, _platform):
    # Even bypassing validate_url() (e.g. a future caller that resolves
    # straight from the registry): no source claims these hosts anymore, so
    # resolve() must return None rather than silently matching the universal
    # (generic) fallback.
    assert registry.resolve(url) is None


def test_registry_no_longer_lists_removed_source_names():
    names = {src.name for src in registry.all_sources()}
    assert names.isdisjoint({'coomer', 'kemono', 'bunkr', 'cyberdrop'})


def test_scan_route_gives_a_clean_unsupported_source_error(client):
    resp = client.post('/api/scrape/scan', json={'url': 'https://coomer.st/onlyfans/user/someone'})
    assert resp.status_code == 400
    body = resp.get_json()
    assert 'coomer' in body['error']
    assert 'removed' in body['error']


@pytest.mark.parametrize('url', [u for u, _ in _REMOVED_URLS])
def test_scan_route_never_crashes_on_removed_sources(client, url):
    resp = client.post('/api/scrape/scan', json={'url': url})
    assert resp.status_code == 400
    assert resp.get_json().get('error')
