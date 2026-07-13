# app/scrape/sources/sexcom.py
"""Source Sex.com — pinboard porno : chaque pin = UNE image (pas d'albums).

Deux chemins de scan, sur le principe des covers PornPics (l'image remontée EST
celle qui matche le mot-clé) :

  • RECHERCHE / TAG (le cas principal) : /pics?search=<mot>, /en/pics?search=…,
    /search/pics?query=…, /pics/<tag> — on interroge directement l'API JSON du
    site (`/portal/api/pictures/search`, la même que l'infinite-scroll, 40
    pins/page → paging.numberOfPages). Chaque pin = {uri, title} ; le média est
    servi par le CDN (imagex1.sx.cdn.live + uri, vérifié accessible sans
    cookie). 1 requête HTTP par page, zéro gallery-dl.
  • PIN / BOARD / USER : /pin/<id>, /en/pics/<id>, /user/<x>/<board> — délégué à
    gallery-dl (extracteur natif `sexcom`). Chaque pin coûte UNE requête page
    chez gallery-dl → fenêtre de pagination courte (--range) pour rester sous le
    timeout du scan.

GIFs/vidéos : hors périmètre (l'import dataset ne prend que les photos raster) —
les URLs /gifs et /videos renvoient un message clair plutôt qu'un scan vide.
"""
import logging
from urllib.parse import parse_qsl, urlparse

from ..validators import Platform
from .base import Source, Capabilities, Match
from . import gdl
from . import registry

logger = logging.getLogger(__name__)

_API_URL = 'https://www.sex.com/portal/api/pictures/search'
_CDN_ROOT = 'https://imagex1.sx.cdn.live'
_PER_PAGE = 40            # taille de page de l'API (celle du site)
_GDL_WINDOW = 12          # pins par « page » via gallery-dl (1 requête/pin chez lui)

_SEXCOM_CAPS = Capabilities(
    can_enumerate_profile=True,
    polite=True,
    media_kinds=frozenset({'image'}),
    own_downloader=True,   # médias = URLs CDN directes (l'import les télécharge lui-même)
)

_MEDIA_TYPES = frozenset({'image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif'})
_CT_EXT = {'image/jpeg': '.jpg', 'image/jpg': '.jpg', 'image/png': '.png',
           'image/webp': '.webp', 'image/gif': '.gif'}


def _search_params_for(url):
    """Si `url` est une recherche/tag PHOTOS, retourne les params de l'API JSON
    ({search, order, sexual-orientation}) ; sinon None (→ chemin gallery-dl).
    'wrong-kind' est renvoyé pour les recherches gifs/videos (message clair).
    PUR (testable sans réseau)."""
    try:
        p = urlparse(url)
    except Exception:
        return None
    segs = [s for s in p.path.split('/') if s]
    if segs and len(segs[0]) == 2 and segs[0].isalpha():   # préfixe de langue /en/
        segs = segs[1:]
    q = dict(parse_qsl(p.query))
    if 'query' in q:                                       # /search/pics?query=…
        q.setdefault('search', q.pop('query'))

    kind = None
    tag = ''
    if len(segs) >= 1 and segs[0] in ('pics', 'gifs', 'videos'):
        kind = segs[0]
        if len(segs) >= 2:
            tag = segs[1]
    elif len(segs) >= 2 and segs[0] == 'search' and segs[1] in ('pics', 'gifs', 'videos'):
        kind = segs[1]
    if kind is None:
        return None
    if tag and tag.isdigit():
        return None            # /pics/<id> = page de détail d'un pin → gallery-dl
    if kind != 'pics':
        return 'wrong-kind'    # gifs/vidéos : l'import dataset ne prend que des photos
    search = (q.get('search') or tag.replace('-', ' ')).strip()
    if not search:
        return None            # listing générique sans mot-clé → gallery-dl
    return {'search': search,
            'order': q.get('order') or 'likeCount',
            'sexual-orientation': q.get('sexual-orientation') or 'straight'}


def _search_json(params, page):
    """Fournée JSON de l'API recherche (seam de test). Lève en cas d'échec."""
    from curl_cffi import requests as cf_requests
    r = cf_requests.get(_API_URL, params={**params, 'page': page, 'limit': _PER_PAGE},
                        impersonate='chrome', timeout=20,
                        headers={'Accept': 'application/json',
                                 'Referer': 'https://www.sex.com/'})
    r.raise_for_status()
    return r.json()


class SexcomSource(Source):
    name = 'sexcom'
    priority = 100
    capabilities = _SEXCOM_CAPS
    paginated = True
    category = 'image'

    def match(self, url):
        from ..validators import url_validator
        if url_validator.detect_platform(url) == Platform.SEXCOM:
            return Match(url=url, validation=None)
        return None

    def scan(self, match):
        try:
            page = max(0, getattr(match, 'page', 0) or 0)
            params = _search_params_for(match.url)
            if params == 'wrong-kind':
                return None, ('Sex.com : seules les recherches PHOTOS sont importables '
                              '(remplacez /gifs ou /videos par /pics dans l\'URL).')
            if params is None:
                # Pin / board / user → gallery-dl (1 requête par pin chez lui →
                # fenêtre courte pour rester sous le timeout du scan).
                start = page * _GDL_WINDOW + 1
                return gdl.enumerate(match.url, platform=self.name,
                                     max_items=_GDL_WINDOW,
                                     image_range=f'{start}-{start + _GDL_WINDOW - 1}')

            data = _search_json(params, page + 1)          # API 1-based
            paging = data.get('paging') or {}
            if page + 1 > int(paging.get('numberOfPages') or 1):
                return [], None                            # fin du listing
            items, seen = [], set()
            for pin in data.get('data') or []:
                uri = pin.get('uri')
                if not uri or not isinstance(uri, str):
                    continue
                url = _CDN_ROOT + uri
                if url in seen:
                    continue
                seen.add(url)
                items.append({'url': url,
                              'title': (pin.get('title') or '').strip()[:200],
                              'thumbnail': url, 'type': 'image', 'platform': 'sexcom'})
            return items, None
        except Exception as e:   # garde-fou : scan() ne lève jamais
            logger.warning('sexcom scan failed: %s', e)
            return None, f'Sex.com : échec du scan ({e}).'

    def download(self, url, dest_base):
        """Télécharge une image CDN en direct (fetch durci). NB : l'import concept
        télécharge en réalité les URLs lui-même (_download_scrape_item) ; ce
        download() honore le contrat Source pour tout autre appelant."""
        import os
        from ..netfetch import MAX_DRIVER_BYTES, fetch_hardened_bytes
        ok, data, ctype, reason = fetch_hardened_bytes(
            url, allowed_types=_MEDIA_TYPES, max_bytes=MAX_DRIVER_BYTES)
        if not ok or not data:
            return False, None, f'Sex.com : téléchargement échoué ({reason}).'
        ct = (ctype or '').split(';', 1)[0].strip().lower()
        ext = _CT_EXT.get(ct) or (os.path.splitext(urlparse(url).path)[1].lower() or '.jpg')
        dest_dir = os.path.dirname(dest_base)
        filename = os.path.basename(dest_base) + ext
        try:
            os.makedirs(dest_dir, exist_ok=True)
            with open(os.path.join(dest_dir, filename), 'wb') as f:
                f.write(data)
        except OSError as e:
            return False, None, f"Sex.com : erreur d'écriture ({e})."
        return True, filename, None


registry.register(SexcomSource())
