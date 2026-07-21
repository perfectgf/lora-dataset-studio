# app/scrape/sources/erome.py
"""Scraper Erome — énumération via gallery-dl + téléchargement direct (curl_cffi).

Erome expose des albums (`/a/ID`), une recherche (`/search?q=`) et des profils
utilisateur (`/USER`). L'énumération est déléguée au moteur partagé **gdl.py**
(qui gère les types 3/6/-1, dont la correction du sentinel d'erreur type -1
qui rendait les erreurs auth/429 invisibles).

gallery-dl REJETTE les sous-domaines de langue (`fr.erome.com`, …) → on
normalise le host en `www.erome.com` (path/query conservés) avant délégation.

API publique (contrat des sources de scraping) :
    scan(validation) -> (items, error)
    download(url, dest_path) -> (ok, final_filename, error)
Aucune des deux ne lève : toute exception est capturée et convertie en message.
"""
import logging
import os
import re
from pathlib import Path
from urllib.parse import urlparse

from . import gdl

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constantes (autonomes — aucun import de config/settings)
# --------------------------------------------------------------------------- #
BASE = "https://www.erome.com"
MAX_ITEMS = 120          # cap dur sur le nombre de médias retournés par scan()
MAX_ALBUMS = 8           # cap recursion search/user (anti-lenteur)
DOWNLOAD_TIMEOUT = 120   # secondes (téléchargement média)
CHUNK_SIZE = 8192

PLATFORM = "erome"

# Host de langue (fr./de./…) à normaliser vers www.erome.com.
_HOST_RE = re.compile(r'^https?://[^/]*erome\.com', re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Helpers internes
# --------------------------------------------------------------------------- #
def _normalize(url: str) -> str:
    """Force le host sur www.erome.com (gallery-dl rejette les sous-domaines de
    langue). Conserve path + query. Si l'URL ne ressemble pas à erome, renvoie
    telle quelle."""
    url = (url or "").strip()
    if _HOST_RE.match(url):
        return _HOST_RE.sub(BASE, url, count=1)
    return url


def _ext_from_url(url: str) -> str:
    """Extension de fichier (avec point) déduite de l'URL. Défaut: .mp4."""
    ext = os.path.splitext(urlparse(url).path)[1].lower()
    if ext in (".mp4", ".webm", ".mov", ".m4v", ".jpg", ".jpeg",
               ".png", ".gif", ".webp"):
        return ".jpg" if ext == ".jpeg" else ext
    return ".mp4"


# --------------------------------------------------------------------------- #
# API publique
# --------------------------------------------------------------------------- #
def scan(validation):
    """Énumère les médias d'une URL Erome via le moteur gallery-dl partagé (gdl.py).

    Normalise le host (gallery-dl rejette les sous-domaines de langue) puis délègue ;
    le moteur gère type 3 (média) / 6 (album recurse) / -1 (erreur d'extracteur).
    Retourne (items, error). Ne lève jamais."""
    try:
        url = getattr(validation, 'original_url', None) or getattr(validation, 'value', '')
        url = _normalize(url)
        if not url:
            return None, "Erome: missing URL."
        items, err = gdl.enumerate(url, platform=PLATFORM,
                                   max_items=MAX_ITEMS, max_albums=MAX_ALBUMS)
        if err:
            return None, f"Erome: {err}"
        return items, None
    except Exception as e:
        logger.exception("Erome scan: erreur inattendue")
        return None, f"Erome: unexpected error ({e})."


def download(url, dest_path):
    """Télécharge un média Erome (vidéo ou image) en direct via curl_cffi.

    `url` = URL CDN directe (ex. https://v22.erome.com/.../X_720p.mp4).
    `dest_path` = chemin de sortie SANS extension imposée (l'extension est
    déduite de l'URL). Écriture atomique .tmp -> final.

    Retourne (ok: bool, final_filename: str | None, error: str | None).
    Ne lève jamais.
    """
    try:
        from curl_cffi import requests as cf_requests
    except ImportError:
        return False, None, "Erome needs the 'curl_cffi' package - install the scrape extras (Setup > Install everything)."

    try:
        dest_path = Path(dest_path)

        from ..netfetch import _validate_public_http_url
        ok_url, ssrf_err = _validate_public_http_url(url)
        if not ok_url:
            return False, None, ssrf_err or "Erome: URL blocked (SSRF)."

        try:
            response = cf_requests.get(
                url, impersonate="chrome",
                headers={"Referer": "https://www.erome.com/"},
                timeout=DOWNLOAD_TIMEOUT, stream=True,
            )
        except Exception as e:
            logger.warning("Erome download: échec requête %s: %s", url, e)
            return False, None, f"Erome: download failed ({e})."

        status = getattr(response, "status_code", 0)
        content_type = ""
        try:
            content_type = response.headers.get("content-type", "") or ""
        except Exception:
            content_type = ""

        if status in (401, 404, 410):
            return False, None, f"Erome: resource unavailable (HTTP {status})."
        if status in (403, 429, 503):
            return False, None, "Erome: access blocked."
        if status >= 400:
            return False, None, f"Erome: HTTP {status} response."

        # Une réponse HTML n'est PAS un média.
        if "text/html" in content_type.lower():
            return False, None, "Erome: HTML response instead of media."

        final_ext = _ext_from_url(url)
        final_path = dest_path.with_name(dest_path.name + final_ext) \
            if not dest_path.suffix else dest_path.with_suffix(final_ext)
        tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")

        try:
            final_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

        try:
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return False, None, f"Erome: write error ({e})."

        # Fichier vide = échec.
        try:
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                tmp_path.unlink(missing_ok=True)
                return False, None, "Erome: downloaded file is empty."
        except OSError:
            return False, None, "Erome: downloaded file is invalid."

        # Renommage atomique .tmp -> final.
        try:
            os.replace(tmp_path, final_path)
        except OSError as e:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return False, None, f"Erome: finalization error ({e})."

        return True, final_path.name, None

    except Exception as e:  # garde-fou ultime — ne jamais lever
        logger.exception("Erome download: erreur inattendue")
        return False, None, f"Erome: unexpected error ({e})."


from .base import Source, Capabilities, Match
from . import registry


class EromeSource(Source):
    name = 'erome'
    priority = 100
    capabilities = Capabilities(can_enumerate_profile=True,
                                media_kinds=frozenset({'video', 'image'}),
                                own_downloader=True)

    def match(self, url):
        from ..validators import url_validator, Platform
        result = url_validator.validate_url(url)
        if result.is_valid and result.platform == Platform.EROME:
            return Match(url=url, validation=result)
        # URLs CDN directes (v*.erome.com) : sous-domaine hors VALID_DOMAINS mais
        # bien sur erome.com → notre downloader curl_cffi (Referer requis).
        if result.platform == Platform.EROME:
            return Match(url=url, validation=None)
        return None

    def scan(self, match):
        return scan(match.validation)

    def download(self, url, dest_base):
        return download(url, dest_base)


registry.register(EromeSource())
