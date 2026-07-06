# app/scrape/sources/redgifs.py
"""Scraper RedGifs — énumération d'un profil / d'une niche / d'une vidéo.

Port autonome (requests, sans `config`/`settings` source) de
`redgifs_downloader/api/redgifs.py`. N'effectue QUE l'énumération (liste des
vidéos + vignettes) ; le téléchargement réel passe par yt-dlp (extracteur
RedGifs) via /api/scrape/download — on renvoie donc l'URL `watch/<id>`.

Hôtes contactés : api.redgifs.com (fixe, public) → pas de risque SSRF.
"""
import logging
import threading

import requests

from ..validators import URLType

logger = logging.getLogger(__name__)

REDGIFS_API_BASE = "https://api.redgifs.com/v2"
REDGIFS_AUTH_URL = f"{REDGIFS_API_BASE}/auth/temporary"
# UA FIXE : le token JWT temporaire est lié à l'User-Agent de la requête d'auth.
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
TIMEOUT = 30
MAX_PAGES = 10
MAX_ITEMS = 100


class RedGifsClient:
    """Client API RedGifs minimal (token temporaire + énumération)."""

    def __init__(self):
        self._token = None
        self._lock = threading.Lock()
        self._session = requests.Session()

    def _headers(self, video_id=None, auth=True):
        h = {
            "Accept": "application/json, text/plain, */*",
            "User-Agent": USER_AGENT,
            "Referer": "https://www.redgifs.com/",
            "Origin": "https://www.redgifs.com",
            "Content-Type": "application/json",
        }
        if video_id:
            # x-customheader OBLIGATOIRE pour /gifs/{id} (sinon 401 malgré le token).
            h["x-customheader"] = f"https://www.redgifs.com/watch/{video_id}"
        if auth and self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def get_token(self):
        with self._lock:
            if self._token:
                return self._token
        try:
            r = self._session.get(REDGIFS_AUTH_URL, headers=self._headers(auth=False), timeout=TIMEOUT)
            r.raise_for_status()
            token = (r.json() or {}).get('token')
        except Exception as e:
            logger.warning(f"[redgifs] échec obtention token: {e}")
            return None
        if token:
            with self._lock:
                self._token = token
            return token
        return None

    def _reset_token(self):
        with self._lock:
            self._token = None

    def _get(self, url, video_id=None):
        """GET authentifié → JSON. Lève requests.HTTPError sur statut != 2xx."""
        r = self._session.get(url, headers=self._headers(video_id=video_id), timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()

    def get_single_video(self, video_id):
        url = f"{REDGIFS_API_BASE}/gifs/{video_id}"
        try:
            data = self._get(url, video_id=video_id)
        except requests.HTTPError as e:
            if getattr(e.response, 'status_code', None) == 401:
                self._reset_token()
                if self.get_token():
                    try:
                        data = self._get(url, video_id=video_id)
                    except Exception:
                        return None
                else:
                    return None
            else:
                return None
        except Exception:
            return None
        return (data or {}).get('gif')

    def _iter_paged(self, url_for_page):
        """Itère les pages via une fonction page→url, avec refresh de token sur 401."""
        page = 1
        token_retries = 0
        while page <= MAX_PAGES:
            url = url_for_page(page)
            try:
                data = self._get(url)
            except requests.HTTPError as e:
                if getattr(e.response, 'status_code', None) == 401 and token_retries < 2:
                    token_retries += 1
                    self._reset_token()
                    if self.get_token():
                        continue  # réessaye la même page
                    return
                return
            except Exception:
                return

            gifs = (data or {}).get('gifs') or []
            if not gifs:
                return
            for gif in gifs:
                yield gif

            total_pages = (data or {}).get('pages', 1) or 1
            if page >= total_pages:
                return
            page += 1

    def iter_user(self, username):
        return self._iter_paged(
            lambda p: f"{REDGIFS_API_BASE}/users/{username}/search?order=new&page={p}")

    def iter_niche(self, niche):
        return self._iter_paged(
            lambda p: f"{REDGIFS_API_BASE}/niches/{niche}/gifs?page={p}")


# Instance globale (feature admin-only mono-utilisateur).
client = RedGifsClient()


def _item_from_gif(gif):
    gid = gif.get('id', '') or ''
    urls = gif.get('urls', {}) or {}
    return {
        'url': f"https://www.redgifs.com/watch/{gid}",
        'title': gid or 'redgif',
        'thumbnail': urls.get('thumbnail') or urls.get('poster'),
        'type': 'video',
        'platform': 'redgifs',
        'duration': gif.get('duration', 0),
    }


def scan(validation):
    """Énumère les médias d'une URL RedGifs. Retourne (items, error).

    validation : ValidationResult (url_type ∈ {PROFILE, NICHE, VIDEO}, value = username/niche/id).
    Borné à MAX_ITEMS. Ne lève jamais.
    """
    try:
        if not client.get_token():
            return None, "Impossible d'obtenir un token RedGifs (Cloudflare / rate-limit ?)."

        ut = validation.url_type
        value = validation.value
        items = []

        if ut == URLType.PROFILE:
            for gif in client.iter_user(value):
                items.append(_item_from_gif(gif))
                if len(items) >= MAX_ITEMS:
                    break
        elif ut == URLType.NICHE:
            for gif in client.iter_niche(value):
                items.append(_item_from_gif(gif))
                if len(items) >= MAX_ITEMS:
                    break
        elif ut == URLType.VIDEO:
            gif = client.get_single_video(value)
            if not gif:
                return None, "Vidéo RedGifs introuvable (ou token expiré)."
            items.append(_item_from_gif(gif))
        else:
            return None, "Type d'URL RedGifs non supporté."

        if not items:
            return None, "Aucun média trouvé pour cette URL RedGifs."
        return items, None
    except Exception as e:  # garde-fou : ne jamais propager
        logger.warning(f"[redgifs] erreur de scan: {e}")
        return None, "Erreur lors du scan RedGifs."


from .base import Source, Capabilities, Match
from . import registry


class RedgifsSource(Source):
    name = 'redgifs'
    priority = 100
    capabilities = Capabilities(can_enumerate_profile=True, own_downloader=False)

    def match(self, url):
        from ..validators import url_validator, Platform
        result = url_validator.validate_url(url)
        if result.is_valid and result.platform == Platform.REDGIFS:
            return Match(url=url, validation=result)
        return None

    def scan(self, match):
        return scan(match.validation)   # délègue au scan(validation) existant


registry.register(RedgifsSource())
