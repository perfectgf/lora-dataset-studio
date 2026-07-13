# app/scrape/sources/reddit.py
"""Source Reddit — recherche par MOT-CLÉ + subreddits + posts, via l'API OAuth.

POURQUOI pas gallery-dl (comme les autres sources) : depuis le verrouillage de
l'API Reddit (2023), les endpoints ANONYMES de navigation/recherche
(`www.reddit.com/*.json`, `/search.json`) renvoient une page-mur anti-bot 403
(« theme-beta »), et l'extracteur reddit de gallery-dl tombe sur le même mur. En
revanche l'API OAUTHENTIFIÉE (`oauth.reddit.com`) répond normalement avec un jeton
`installed_client` ANONYME — obtenable sans compte ni app enregistrée, avec le
client-id public de gallery-dl. On parle donc directement à l'API OAuth ici, et on
extrait les images du JSON des posts (galeries, liens directs i.redd.it, preview).

Formes d'URL reconnues (routées par _endpoint_for) :
  • recherche GLOBALE      : reddit.com/search/?q=<mot-clé>
  • recherche SUBREDDIT     : reddit.com/r/<sub>/search/?q=<mot-clé>   (restrict_sr)
  • listing subreddit       : reddit.com/r/<sub>[/<tri>]  (hot/top/new/rising…)
  • posts d'un utilisateur  : reddit.com/user/<nom>
  • post seul (galerie incl): reddit.com/r/<sub>/comments/<id>/…
  • lien de partage mobile  : reddit.com/r/<sub>/s/<token>  (résolu par redirection)
  • image directe           : i.redd.it/<id>.jpg  (→ item unique, tel quel)

Le mot-clé de l'UI est transformé côté frontend en une URL de recherche reddit,
puis passe par le pipeline /scan habituel (aucune route dédiée).

Sécurité : seuls des hôtes reddit sont contactés (jeton + API) ; les images sont
téléchargées par le flux d'import durci (fetch_hardened_bytes, anti-SSRF, magic-bytes).
"""
import logging
import os
import time
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests

from ..validators import Platform
from .base import Source, Capabilities, Match
from .gdl_source import resolve_cookies
from . import registry

logger = logging.getLogger(__name__)

_API_BASE = 'https://oauth.reddit.com'
_TOKEN_URL = 'https://www.reddit.com/api/v1/access_token'
# UA descriptif (les règles API Reddit demandent un UA unique et identifiable).
_UA = 'LoRA-Dataset-Studio/1.0 (+https://github.com/perfectgf/lora-dataset-studio)'
# client-id public « installed_client » de gallery-dl : autorise un jeton ANONYME
# (grant device_id) sans compte ni app enregistrée. Surchargé si l'utilisateur
# fournit le sien — Settings → Scraping & sources (secret REDDIT_CLIENT_ID, posé
# dans os.environ à la sauvegarde) ou <SCRAPE_COOKIES_DIR>/reddit_client_id.txt —
# indispensable quand ce client-id partagé se fait rate-limiter (429).
_GDL_CLIENT_ID = '6N9uN0krSDE-ig'

_HTTP_TIMEOUT = 20
_MAX_429_RETRY_WAIT = 4    # sur 429, on ne re-tente qu'UNE fois si le reset est proche
_BATCH_POSTS = 30          # posts récupérés par « page » (chaque post ≈ 1-N images)
_SCAN_MAX = 200            # plafond d'items remontés par un scan (payload borné)
_SORTS = frozenset({'hot', 'new', 'top', 'rising', 'controversial', 'best'})
_IMG_EXT = ('.jpg', '.jpeg', '.png', '.webp', '.gif')

# Jeton mis en cache en mémoire du process (valable ~24 h ; on renouvelle avant expiry).
# `cid` = client-id qui a frappé le jeton : si l'utilisateur change de client-id
# (Settings → Scraping, effectif via os.environ sans restart), le jeton en cache
# appartient encore à l'ANCIEN id — donc à son quota — et doit être re-frappé.
_token_cache = {'value': None, 'exp': 0.0, 'cid': None}

_REDDIT_CAPS = Capabilities(
    can_enumerate_profile=True,
    polite=True,
    media_kinds=frozenset({'image'}),
    own_downloader=True,   # download() dédié (fetch durci), cf. note : l'import concept
)                          # télécharge en réalité les URLs directement (flux autonome).

# Content-types servis par les CDN d'images reddit (pour download()).
_MEDIA_TYPES = frozenset({'image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif'})
_CT_EXT = {'image/jpeg': '.jpg', 'image/jpg': '.jpg', 'image/png': '.png',
           'image/webp': '.webp', 'image/gif': '.gif'}


# ---------------------------------------------------------------------------
# Canonicalisation d'URL (résout /s/ et redd.it par redirection, purge le tracking)
# ---------------------------------------------------------------------------
_DROP_PARAMS = ('share_id', 'correlation_id', 'ref', 'ref_source', 'rdt')


def _is_reddit_host(host: str) -> bool:
    return host == 'reddit.com' or host.endswith('.reddit.com')


def _canonical_reddit_url(url: str) -> str:
    """Canonicalise une URL Reddit : résout les liens de partage /s/ et redd.it par
    redirection HTTP (hôtes reddit UNIQUEMENT), force www.reddit.com, purge les
    params de partage/tracking en gardant les params de contenu (?t=month, ?q=…).
    URL non-reddit ou CDN direct (i./preview.redd.it) → inchangée. Ne lève jamais."""
    try:
        p = urlparse(url)
    except Exception:
        return url
    host = (p.hostname or '').lower()
    is_shortener = host in ('redd.it', 'www.redd.it')   # PAS i./preview.redd.it (CDN)
    if not (_is_reddit_host(host) or is_shortener):
        return url
    if is_shortener or '/s/' in p.path:
        try:
            r = requests.get(url, headers={'User-Agent': _UA}, timeout=10,
                             allow_redirects=True)
            tgt = urlparse(r.url)
            if _is_reddit_host((tgt.hostname or '').lower()):  # anti-redirection exotique
                p = tgt
        except requests.RequestException as e:
            logger.warning('reddit share-link resolve failed (%s) — trying as-is', e)
    keep = [(k, v) for k, v in parse_qsl(p.query)
            if not k.lower().startswith('utm_') and k.lower() not in _DROP_PARAMS]
    return urlunparse(('https', 'www.reddit.com', p.path, '', urlencode(keep), ''))


# ---------------------------------------------------------------------------
# URL reddit canonique → endpoint API OAuth (PUR, testable sans réseau)
# ---------------------------------------------------------------------------
def _endpoint_for(url: str):
    """Mappe une URL reddit canonique vers l'endpoint API à interroger.

    Retourne un dict {api_path, params, kind} où kind ∈ {'listing','post','direct'},
    ou None si la forme d'URL n'est pas reconnue. 'direct' porte en plus 'url' (image
    CDN à remonter telle quelle)."""
    try:
        p = urlparse(url)
    except Exception:
        return None
    host = (p.hostname or '').lower()

    # Image CDN directe (i.redd.it / preview.redd.it) ou toute URL finissant par une
    # extension image → item unique, sans appel API.
    if host in ('i.redd.it', 'preview.redd.it', 'external-preview.redd.it') \
            or p.path.lower().endswith(_IMG_EXT):
        return {'api_path': None, 'params': {}, 'kind': 'direct', 'url': url}

    segs = [s for s in p.path.split('/') if s]
    q = dict(parse_qsl(p.query))

    # Post seul (galerie incluse) : /r/<sub>/comments/<id>/… ou /comments/<id>.
    if 'comments' in segs:
        i = segs.index('comments')
        if i + 1 < len(segs):
            return {'api_path': f'/comments/{segs[i + 1]}',
                    'params': {'limit': 1, 'raw_json': 1}, 'kind': 'post'}

    # Recherche : globale (/search) ou scopée subreddit (/r/<sub>/search).
    is_global_search = segs and segs[-1] == 'search' and not (segs[0] == 'r')
    is_sub_search = len(segs) >= 3 and segs[0] == 'r' and segs[2] == 'search'
    if is_global_search or is_sub_search:
        params = {
            'q': q.get('q', ''),
            'sort': q.get('sort', 'top'),
            't': q.get('t', 'all'),
            'type': 'link',
            'raw_json': 1,
            'include_over_18': 'on',
        }
        if is_sub_search:
            return {'api_path': f'/r/{segs[1]}/search',
                    'params': {**params, 'restrict_sr': 1}, 'kind': 'listing'}
        return {'api_path': '/search', 'params': params, 'kind': 'listing'}

    # Posts d'un utilisateur : /user/<nom> ou /u/<nom>.
    if len(segs) >= 2 and segs[0] in ('user', 'u'):
        return {'api_path': f'/user/{segs[1]}/submitted',
                'params': {'sort': q.get('sort', 'top'), 't': q.get('t', 'all'),
                           'raw_json': 1}, 'kind': 'listing'}

    # Listing subreddit : /r/<sub> ou /r/<sub>/<tri>.
    if len(segs) >= 2 and segs[0] == 'r':
        sub = segs[1]
        sort = segs[2] if len(segs) >= 3 and segs[2] in _SORTS else 'hot'
        return {'api_path': f'/r/{sub}/{sort}',
                'params': {'t': q.get('t', 'all'), 'raw_json': 1}, 'kind': 'listing'}

    return None


# ---------------------------------------------------------------------------
# Extraction des images d'un post (PUR, testable avec des fixtures JSON)
# ---------------------------------------------------------------------------
def _pick_preview(entries, url_key, size_key, fallback):
    """Choisit une résolution de preview ~≥320px de large (grille légère), sinon la
    plus grande dispo, sinon `fallback`. `entries` = liste de dicts (media_metadata
    'p' → clés u/x/y ; preview 'resolutions' → clés url/width)."""
    if isinstance(entries, list) and entries:
        for e in entries:
            if e.get(size_key, 0) >= 320 and e.get(url_key):
                return e[url_key]
        if entries[-1].get(url_key):
            return entries[-1][url_key]
    return fallback


def _is_image_url(u: str) -> bool:
    try:
        return urlparse(u).path.lower().endswith(_IMG_EXT)
    except Exception:
        return False


def _items_from_post(p: dict) -> list:
    """Extrait la/les image(s) directes d'un post (data dict) au schéma commun
    {url, title, thumbnail, type, platform, subreddit}. Retourne [] si pas d'image."""
    title = (p.get('title') or '')[:200]
    sub = p.get('subreddit') or ''

    def item(u, thumb):
        return {'url': u, 'title': title, 'thumbnail': thumb or u,
                'type': 'image', 'platform': 'reddit', 'subreddit': sub}

    # 1. Galerie : gallery_data ordonne les media_id, media_metadata porte les URLs.
    if p.get('is_gallery') and isinstance(p.get('media_metadata'), dict):
        gd = ((p.get('gallery_data') or {}).get('items')) or []
        order = [it.get('media_id') for it in gd if it.get('media_id')] \
            or list(p['media_metadata'].keys())
        out = []
        for mid in order:
            meta = p['media_metadata'].get(mid) or {}
            if meta.get('e') != 'Image':
                continue
            full = (meta.get('s') or {}).get('u')
            if full:
                out.append(item(full, _pick_preview(meta.get('p'), 'u', 'x', full)))
        if out:
            return out

    # 2. Lien image direct (i.redd.it, imgur single…).
    u = p.get('url_overridden_by_dest') or p.get('url') or ''
    prev = ((p.get('preview') or {}).get('images') or [])
    thumb = _pick_preview(prev[0].get('resolutions'), 'url', 'width', None) if prev else None
    if _is_image_url(u):
        return [item(u, thumb or u)]

    # 3. Repli : la preview reddit (couvre les liens externes qu'il a vignettés).
    if prev:
        src = (prev[0].get('source') or {}).get('url')
        if src:
            return [item(src, thumb or src)]
    return []


# ---------------------------------------------------------------------------
# API OAuth (jeton anonyme + GET authentifié)
# ---------------------------------------------------------------------------
def _client_id() -> str:
    """client-id Reddit : env (y compris Settings, qui écrit os.environ) > fichier
    admin > client-id public de gallery-dl."""
    env = (os.environ.get('REDDIT_CLIENT_ID') or '').strip()
    if env:
        return env
    path = resolve_cookies('reddit_client_id')   # <SCRAPE_COOKIES_DIR>/reddit_client_id.txt
    if path:
        try:
            val = open(path, encoding='utf-8').read().strip()
            if val:
                return val
        except OSError:
            pass
    return _GDL_CLIENT_ID


def _get_token():
    """Jeton OAuth anonyme (installed_client), mis en cache jusqu'à ~2 min avant expiry
    — et re-frappé si le client-id a changé entre-temps (un jeton appartient au quota
    du client-id qui l'a émis). Retourne le jeton ou None (échec réseau/auth). Ne lève
    jamais."""
    now = time.time()
    cid = _client_id()
    if _token_cache['value'] and now < _token_cache['exp'] and _token_cache['cid'] == cid:
        return _token_cache['value']
    try:
        r = requests.post(
            _TOKEN_URL,
            data={'grant_type': 'https://oauth.reddit.com/grants/installed_client',
                  'device_id': 'DO_NOT_TRACK_THIS_DEVICE'},
            auth=(cid, ''), headers={'User-Agent': _UA}, timeout=15)
        r.raise_for_status()
        j = r.json()
    except (requests.RequestException, ValueError) as e:
        logger.warning('reddit token request failed: %s', e)
        return None
    tok = j.get('access_token')
    if tok:
        _token_cache['value'] = tok
        _token_cache['exp'] = now + max(60, int(j.get('expires_in', 3600)) - 120)
        _token_cache['cid'] = cid
    return tok


class RedditRateLimited(Exception):
    """429 Reddit : quota (~1000 req / 10 min, par IP+client) temporairement épuisé.
    Porte le délai avant reset (secondes) si connu, pour un message actionnable."""
    def __init__(self, reset_seconds=None):
        self.reset_seconds = reset_seconds
        super().__init__('reddit rate limited')


def _reset_seconds(resp):
    """Secondes avant reset du quota : en-tête Retry-After sinon x-ratelimit-reset.
    None si illisible."""
    for key in ('retry-after', 'x-ratelimit-reset'):
        val = resp.headers.get(key)
        if val:
            try:
                return max(0, int(float(val)))
            except (TypeError, ValueError):
                pass
    return None


def _api_get(api_path: str, params: dict, token: str) -> dict:
    """GET authentifié sur oauth.reddit.com. Lève requests.HTTPError/RequestException
    (attrapé par scan). Renouvelle le jeton une fois sur 401 (jeton expiré en vol).
    Sur 429 : une seule re-tentative si le reset est proche (≤ _MAX_429_RETRY_WAIT),
    sinon lève RedditRateLimited (message actionnable côté scan)."""
    def _do(tok):
        return requests.get(_API_BASE + api_path, params=params,
                            headers={'User-Agent': _UA, 'Authorization': f'Bearer {tok}'},
                            timeout=_HTTP_TIMEOUT)
    r = _do(token)
    if r.status_code == 401:
        _token_cache['exp'] = 0.0            # force refresh
        tok2 = _get_token()
        if tok2:
            r = _do(tok2)
    if r.status_code == 429:
        wait = _reset_seconds(r)
        if wait is not None and wait <= _MAX_429_RETRY_WAIT:  # blip transitoire → 1 retry
            time.sleep(wait + 1)
            r = _do(_token_cache['value'] or token)
        if r.status_code == 429:
            raise RedditRateLimited(_reset_seconds(r))
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# La source
# ---------------------------------------------------------------------------
class RedditSource(Source):
    name = 'reddit'
    priority = 100
    capabilities = _REDDIT_CAPS
    paginated = True          # « Charger plus » : avance le curseur `after` du listing
    category = 'image'        # recherche d'images → ouvert aux non-admins

    def match(self, url):
        from ..validators import url_validator
        if url_validator.detect_platform(url) == Platform.REDDIT:
            return Match(url=url, validation=None)
        return None

    def _fetch_listing(self, ep, token, page):
        """Récupère la fournée de posts de la `page` demandée en avançant le curseur
        `after` (recherche/listing sont paginés par curseur, pas par offset). page 0 =
        1re fournée ; page N = N appels séquentiels. Retourne [] si on épuise le
        listing AVANT d'atteindre la page (sinon « Charger plus » re-remonterait la
        dernière fournée déjà vue → doublons)."""
        after = None
        children = []
        for i in range(page + 1):
            params = dict(ep['params'])
            params['limit'] = _BATCH_POSTS
            if after:
                params['after'] = after
            listing = (_api_get(ep['api_path'], params, token) or {}).get('data', {})
            children = listing.get('children', []) or []
            after = listing.get('after')
            if i < page and not after:
                return []      # plus de pages avant d'atteindre la page voulue
        return children

    def _fetch_post(self, ep, token):
        """Récupère un post seul. L'endpoint /comments/<id> renvoie [postListing,
        commentsListing] → on prend le 1er enfant du 1er listing."""
        data = _api_get(ep['api_path'], ep['params'], token)
        if isinstance(data, list) and data:
            return (data[0].get('data', {}) or {}).get('children', []) or []
        return []

    def scan(self, match):
        try:
            url = _canonical_reddit_url(match.url)
            ep = _endpoint_for(url)
            if not ep:
                return None, ('Reddit : URL non reconnue (subreddit, recherche, '
                              'post ou lien de partage attendus).')
            if ep['kind'] == 'direct':
                return [{'url': ep['url'], 'title': '', 'thumbnail': ep['url'],
                         'type': 'image', 'platform': 'reddit'}], None
            if ep['kind'] == 'listing' and 'q' in ep['params'] and not ep['params']['q'].strip():
                return None, 'Reddit : mot-clé de recherche manquant.'

            token = _get_token()
            if not token:
                return None, 'Reddit : authentification API impossible (réessayez).'

            if ep['kind'] == 'post':
                children = self._fetch_post(ep, token)
            else:
                page = max(0, getattr(match, 'page', 0) or 0)
                children = self._fetch_listing(ep, token, page)

            items, seen = [], set()
            for ch in children:
                data = ch.get('data') if isinstance(ch, dict) else None
                if not isinstance(data, dict):
                    continue
                for it in _items_from_post(data):
                    if it['url'] not in seen:
                        seen.add(it['url'])
                        items.append(it)
                        if len(items) >= _SCAN_MAX:
                            return items, None
            return items, None
        except RedditRateLimited as e:
            wait = f' Réessaie dans ~{e.reset_seconds}s.' if e.reset_seconds else ' Réessaie dans une minute.'
            return None, ('Reddit limite temporairement les requêtes (quota partagé '
                          '~1000 requêtes / 10 min).' + wait)
        except requests.RequestException as e:
            return None, f'Reddit : échec réseau ({e}).'
        except Exception as e:   # garde-fou : scan() ne lève jamais
            logger.exception('reddit scan')
            return None, f'Reddit : erreur inattendue ({e}).'

    def download(self, url, dest_base):
        """Télécharge une image reddit EN DIRECT (fetch durci). NB : le flux d'import
        concept télécharge en réalité les URLs lui-même (_download_scrape_item) ; ce
        download() n'est là que pour honorer le contrat Source si un autre appelant
        l'emprunte."""
        from ..netfetch import MAX_DRIVER_BYTES, fetch_hardened_bytes
        ok, data, ctype, reason = fetch_hardened_bytes(
            url, allowed_types=_MEDIA_TYPES, max_bytes=MAX_DRIVER_BYTES)
        if not ok or not data:
            return False, None, f'Reddit : téléchargement échoué ({reason}).'
        ct = (ctype or '').split(';', 1)[0].strip().lower()
        ext = _CT_EXT.get(ct) or (os.path.splitext(urlparse(url).path)[1].lower() or '.jpg')
        dest_dir = os.path.dirname(dest_base)
        filename = os.path.basename(dest_base) + ext
        try:
            os.makedirs(dest_dir, exist_ok=True)
            with open(os.path.join(dest_dir, filename), 'wb') as f:
                f.write(data)
        except OSError as e:
            return False, None, f"Reddit : erreur d'écriture ({e})."
        return True, filename, None


registry.register(RedditSource())
