"""Source Pexels SFW via l'API REST officielle.

L'API web privée utilisée par l'extracteur gallery-dl Pexels répond de manière
instable (HTTP 520 observé). Cette source emploie donc uniquement
``https://api.pexels.com/v1`` avec la clé ``PEXELS_API_KEY`` de l'opérateur.

Routes Pexels prises en charge : recherche (y compris ``/en-us/search/`` et
``/fr-fr/chercher/``), collection accessible avec la clé et photo unique.
L'API officielle n'expose pas les profils publics ``/@user``.
"""
import os
from urllib.parse import quote, urlsplit

import requests

from ..validators import Platform, parse_pexels_path
from .base import Capabilities, Match, Source
from . import registry


_API_ROOT = 'https://api.pexels.com/v1'
_HTTP_TIMEOUT = 20
_PAGE_SIZE = 80

_PEXELS_CAPS = Capabilities(
    can_enumerate_profile=False,
    needs_auth=True,
    polite=True,
    media_kinds=frozenset({'image'}),
    own_downloader=False,
)


def pexels_api_key():
    """Clé API officielle lue au runtime (effective sans redémarrage)."""
    value = (os.environ.get('PEXELS_API_KEY') or '').strip()
    return value or None


def _is_https_url(value, allowed_hosts):
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = urlsplit(value.strip())
        host = (parsed.hostname or '').lower()
    except (TypeError, ValueError):
        return False
    return (parsed.scheme == 'https' and host in allowed_hosts and
            parsed.username is None and parsed.password is None)


def _photo_item(photo):
    """Photo API valide -> schéma commun, sinon None.

    Les champs d'attribution sont obligatoires : une entrée incomplète ne doit
    pas entrer dans la grille sans crédit Pexels/photographe exploitable.
    """
    if not isinstance(photo, dict) or not isinstance(photo.get('src'), dict):
        return None
    src = photo['src']
    original = src.get('original')
    thumbnail = (src.get('medium') or src.get('large') or
                 src.get('large2x') or original)
    source_url = photo.get('url')
    photographer = photo.get('photographer')
    photographer_url = photo.get('photographer_url')
    required = (photographer,)
    if not all(isinstance(value, str) and value.strip() for value in required):
        return None
    if not (_is_https_url(original, {'images.pexels.com'}) and
            _is_https_url(thumbnail, {'images.pexels.com'}) and
            _is_https_url(source_url, {'pexels.com', 'www.pexels.com'}) and
            _is_https_url(photographer_url, {'pexels.com', 'www.pexels.com'})):
        return None
    alt = photo.get('alt')
    return {
        'url': original.strip(),
        'thumbnail': thumbnail.strip(),
        'title': alt if isinstance(alt, str) else '',
        'type': 'image',
        'platform': 'pexels',
        'source_url': source_url.strip(),
        'photographer': photographer.strip(),
        'photographer_url': photographer_url.strip(),
    }


def _request_json(endpoint, params, key, *, not_found=None):
    """GET API borné -> (dict|None, erreur|None), sans jamais exposer la clé."""
    try:
        response = requests.get(
            endpoint,
            headers={'Authorization': key},
            params=params,
            timeout=_HTTP_TIMEOUT,
            allow_redirects=False,
        )
    except requests.Timeout:
        return None, 'Pexels : délai dépassé lors de l’appel à l’API officielle.'
    except requests.RequestException:
        return None, 'Pexels : erreur réseau lors de l’appel à l’API officielle.'

    status = response.status_code
    if 300 <= status < 400:
        return None, 'Pexels : redirection inattendue refusée par sécurité.'
    if status in (401, 403):
        return None, (f'Pexels : clé API refusée (HTTP {status}). '
                      'Vérifiez PEXELS_API_KEY dans Settings → Scraping & sources.')
    if status == 404:
        return None, (not_found or
                      'Pexels : ressource introuvable ou non accessible avec cette clé API.')
    if status == 429:
        headers = getattr(response, 'headers', {}) or {}
        reset = headers.get('X-Ratelimit-Reset') or headers.get('X-RateLimit-Reset')
        reset_hint = ''
        if reset is not None:
            safe_reset = str(reset).strip().replace('\r', '').replace('\n', '')[:80]
            if safe_reset:
                reset_hint = f' Réinitialisation annoncée : {safe_reset}.'
        return None, ('Pexels : quota de l’API atteint (HTTP 429).'
                      f'{reset_hint} Réessayez plus tard.')
    if status >= 500:
        return None, f'Pexels : API officielle temporairement indisponible (HTTP {status}).'
    if status >= 400:
        return None, (f'Pexels : requête refusée par l’API (HTTP {status}). '
                      'Vérifiez l’URL Pexels.')

    try:
        data = response.json()
    except (TypeError, ValueError):
        return None, 'Pexels : réponse JSON illisible de l’API officielle.'
    if not isinstance(data, dict):
        return None, 'Pexels : schéma JSON inattendu de l’API officielle.'
    return data, None


def _map_photo_list(data, key):
    raw = data.get(key)
    if not isinstance(raw, list):
        return None, f'Pexels : schéma incomplet (champ {key} absent ou invalide).'
    items = [item for item in (_photo_item(photo) for photo in raw) if item]
    if raw and not items:
        return None, 'Pexels : schéma photo incomplet dans la réponse API.'
    return items, None


def _target_for(url, page):
    """URL publique Pexels validée -> endpoint officiel, params, type de cible."""
    try:
        path = urlsplit(url).path
    except (TypeError, ValueError):
        return None

    route = parse_pexels_path(path)
    if route is None:
        return None

    if route.kind == 'search':
        params = {
            'query': route.api_value,
            'page': page + 1,
            'per_page': _PAGE_SIZE,
        }
        if route.api_locale:
            params['locale'] = route.api_locale
        return _API_ROOT + '/search', params, 'search'

    if route.kind == 'collection':
        return (_API_ROOT + f'/collections/{quote(route.api_value, safe="")}',
                {'type': 'photos', 'page': page + 1, 'per_page': _PAGE_SIZE},
                'collection')

    if route.kind == 'photo':
        return (_API_ROOT + f'/photos/{route.api_value}', None, 'photo')
    return None


class PexelsSource(Source):
    name = 'pexels'
    priority = 100
    capabilities = _PEXELS_CAPS
    paginated = True
    page_size = _PAGE_SIZE
    category = 'image'

    def match(self, url):
        from ..validators import url_validator
        result = url_validator.validate_url(url)
        if result.is_valid and result.platform == Platform.PEXELS:
            return Match(url=result.original_url, validation=result)
        return None

    def scan(self, match):
        page = max(0, getattr(match, 'page', 0) or 0)
        target = _target_for(match.url, page)
        if target is None:
            return None, 'Pexels : format d’URL non pris en charge par l’API officielle.'
        endpoint, params, kind = target
        match.paginated = kind != 'photo'
        if kind == 'photo' and page > 0:
            return [], None

        key = pexels_api_key()
        if not key:
            return None, ('Pexels : clé API requise. Ajoutez PEXELS_API_KEY dans '
                          'Settings → Scraping & sources (clé gratuite).')

        not_found = None
        if kind == 'collection':
            not_found = ('Pexels : collection introuvable ou non accessible avec cette clé API '
                         '(les collections publiques arbitraires ne sont pas toutes exposées).')
        elif kind == 'photo':
            not_found = 'Pexels : photo introuvable (HTTP 404).'
        data, err = _request_json(endpoint, params, key, not_found=not_found)
        if err:
            return None, err

        if kind == 'photo':
            item = _photo_item(data)
            if not item:
                return None, 'Pexels : schéma photo incomplet dans la réponse API.'
            return [item], None

        next_page = data.get('next_page')
        match.paginated = isinstance(next_page, str) and bool(next_page.strip())
        return _map_photo_list(data, 'photos' if kind == 'search' else 'media')


registry.register(PexelsSource())
