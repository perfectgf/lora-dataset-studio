# app/scrape/sources/picazor.py
"""Scraper Picazor — énumération + téléchargement direct (curl_cffi).

Picazor est derrière Cloudflare : on requête avec curl_cffi en
`impersonate='chrome'` pour franchir la protection JA3/TLS. Aucune dépendance
HTML lourde (pas de bs4) — le parsing se fait par regex sur le HTML server-side.

Structure du site (reverse, vérifiée) :
  - /fr/{creator}              -> page profil : grille des 24 médias récents
  - /fr/{creator}/page/{N}     -> page N du listing (24 médias/page, anté-chrono)
  - /fr/{creator}/{index}      -> page de DÉTAIL d'un média (PAS un listing)
  - vignette 300px_{name}.mp4.jpg -> VIDÉO  /uploads/<path>/{name}.mp4
  - vignette 300px_{name}.jpg     -> PHOTO  /uploads/<path>/{name}.jpg

API publique (contrat des sources de scraping) :
    scan(validation) -> (items, error)
    download(url, dest_path) -> (ok, final_filename, error)
Aucune des deux ne lève : toute exception est capturée et convertie en message.
"""
import logging
import math
import os
import re
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constantes (autonomes — aucun import de config/settings)
# --------------------------------------------------------------------------- #
BASE_URL = "https://picazor.com"
ITEMS_PER_PAGE = 24
# Relevé (le « 60 » bloquait alors que la pagination serveur /page/N en a bien plus).
# Le scan est SYNCHRONE (1 requête Cloudflare par page) → MAX_PAGES borne le temps
# (~1-2 s/page). Au-delà il faudrait une pagination asynchrone.
MAX_ITEMS = 300          # borne dure sur le nombre d'items retournés par scan()
MAX_PAGES = 14           # garde-fou pagination (14×24=336 ≥ 300 ; ~20-30 s pire cas)
HTTP_TIMEOUT = 30        # secondes (requête HTML)
DOWNLOAD_TIMEOUT = 300   # secondes (téléchargement média)
CHUNK_SIZE = 8192

PLATFORM = "picazor"

# En-têtes de base ; curl_cffi gère l'empreinte TLS via impersonate='chrome'.
_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Signaux d'un challenge / blocage Cloudflare dans une réponse HTML.
_CLOUDFLARE_MARKERS = (
    "just a moment",
    "cf-browser-verification",
    "cf-challenge",
    "challenge-platform",
    "attention required",
    "checking your browser",
    "__cf_chl",
)

# Vignette de listing : capture (chemin, nom, marqueur vidéo éventuel).
_THUMB_RE = re.compile(r'"(/uploads/[^"]+?/)300px_([^"/]+?)(\.mp4)?\.jpg"')
# Média plein format sur une page de détail.
_DETAIL_VIDEO_RE = re.compile(r'"(/uploads/[^"]+?\.mp4)"')
# Photo originale (jpg sans préfixe de taille NNNpx_).
_DETAIL_PHOTO_RE = re.compile(r'"(/uploads/[^"]+?/)(?!\d+px[_-])([^"/]+?\.jpg)"')


# --------------------------------------------------------------------------- #
# Helpers internes
# --------------------------------------------------------------------------- #
def _looks_like_cloudflare(html: str, status_code: int = 200) -> bool:
    """Heuristique : la réponse est-elle un challenge / blocage Cloudflare ?"""
    if status_code in (403, 429, 503):
        return True
    head = (html or "")[:4000].lower()
    return any(marker in head for marker in _CLOUDFLARE_MARKERS)


def _path_stem(upload_path: str) -> str:
    """Nom de fichier sans extension depuis un chemin /uploads/..."""
    filename = upload_path.rstrip("/").rsplit("/", 1)[-1]
    return filename.rsplit(".", 1)[0]


def _parse_picazor_url(url: str) -> dict:
    """Déduit le type d'URL Picazor depuis l'URL brute (langue/creator/index).

    Retourne un dict {'type': 'profile'|'media'|'listing'|'unknown', ...}.
    Plus précis que ValidationResult.value (qui n'expose pas l'index de détail).
    """
    # Page spéciale : /fr/videos/week, /fr/models/..., /fr/categories/...
    m = re.search(r'picazor\.com/([^/]+)/(videos|models|categories)(?:/([^/?#]+))?', url)
    if m:
        return {"type": "listing", "category": m.group(2), "filter": m.group(3) or "", "url": url}

    # Page N du listing : /fr/{creator}/page/{N}
    m = re.search(r'picazor\.com/([^/]+)/([^/?#]+)/page/(\d+)', url)
    if m:
        return {"type": "profile", "language": m.group(1), "creator": m.group(2),
                "page": int(m.group(3)), "url": url}

    # Page de détail d'un média : /fr/{creator}/{index}
    m = re.search(r'picazor\.com/([^/]+)/([^/?#]+)/(\d+)', url)
    if m:
        return {"type": "media", "language": m.group(1), "creator": m.group(2),
                "index": int(m.group(3)), "url": url}

    # Profil (page 1) : /fr/{creator}
    m = re.search(r'picazor\.com/([^/]+)/([^/?#]+)', url)
    if m:
        return {"type": "profile", "language": m.group(1), "creator": m.group(2),
                "page": 1, "url": url}

    return {"type": "unknown", "url": url}


def _request_html(url: str):
    """Récupère le HTML d'une page Picazor via curl_cffi (impersonate=chrome).

    Retourne (html, error). En cas de blocage Cloudflare / erreur : (None, msg).
    Ne lève jamais.
    """
    from ..netfetch import _validate_public_http_url
    ok_url, ssrf_err = _validate_public_http_url(url)
    if not ok_url:
        return None, ssrf_err or "Picazor: URL blocked (SSRF)."

    try:
        from curl_cffi import requests as cf_requests
    except ImportError:
        return None, "Picazor needs the 'curl_cffi' package - install the scrape extras (Setup > Install everything)."

    try:
        response = cf_requests.get(
            url, headers=_HEADERS, impersonate="chrome", timeout=HTTP_TIMEOUT
        )
    except Exception as e:  # réseau, TLS, timeout...
        logger.warning("Picazor: échec requête %s: %s", url, e)
        return None, f"Picazor: request failed ({e})."

    status = getattr(response, "status_code", 0)
    html = getattr(response, "text", "") or ""

    if _looks_like_cloudflare(html, status):
        return None, "Picazor (Cloudflare) blocked access."
    if status >= 400:
        return None, f"Picazor: HTTP {status} response."

    return html, None


def _parse_listing(html: str, creator: str) -> list:
    """Extrait les médias d'une page de listing via les vignettes 300px_."""
    items = []
    seen = set()

    for m in _THUMB_RE.finditer(html):
        path, name, video_marker = m.group(1), m.group(2), m.group(3)
        is_video = video_marker is not None
        ext = ".mp4" if is_video else ".jpg"
        media_url = f"{BASE_URL}{path}{name}{ext}"
        if media_url in seen:
            continue
        seen.add(media_url)

        thumb_suffix = ".mp4.jpg" if is_video else ".jpg"
        items.append({
            "url": media_url,
            "title": f"{creator}_{name}",
            "thumbnail": f"{BASE_URL}{path}300px_{name}{thumb_suffix}",
            "type": "video" if is_video else "image",
            "platform": PLATFORM,
        })

    return items


def _max_media_index(html: str, creator: str) -> int:
    """Index max des liens /fr/{creator}/{i} = total approx. de médias."""
    indices = [
        int(m.group(1))
        for m in re.finditer(rf'href="/fr/{re.escape(creator)}/(\d+)"', html)
    ]
    return max(indices, default=0)


def _parse_detail(html: str, creator: str) -> list:
    """Extrait LE média d'une page de détail (vidéo prioritaire, sinon photo)."""
    m = _DETAIL_VIDEO_RE.search(html)
    if m:
        media_url = f"{BASE_URL}{m.group(1)}"
        return [{
            "url": media_url,
            "title": f"{creator}_{_path_stem(m.group(1))}",
            "thumbnail": f"{media_url}.jpg",
            "type": "video",
            "platform": PLATFORM,
        }]

    m = _DETAIL_PHOTO_RE.search(html)
    if m:
        path, filename = m.group(1), m.group(2)
        media_url = f"{BASE_URL}{path}{filename}"
        return [{
            "url": media_url,
            "title": f"{creator}_{filename.rsplit('.', 1)[0]}",
            "thumbnail": media_url,
            "type": "image",
            "platform": PLATFORM,
        }]

    return []


def _ext_from_content_type(content_type: str, url: str) -> str:
    """Extension de fichier déduite du content-type (fallback : URL)."""
    ct = (content_type or "").split(";", 1)[0].strip().lower()
    mapping = {
        "video/mp4": ".mp4",
        "video/webm": ".webm",
        "video/quicktime": ".mov",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
    }
    if ct in mapping:
        return mapping[ct]
    # Fallback : extension présente dans l'URL.
    path = urlparse(url).path
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mp4", ".webm", ".mov", ".jpg", ".jpeg", ".png", ".gif", ".webp"):
        return ".jpg" if ext == ".jpeg" else ext
    return ".mp4"  # défaut raisonnable côté Picazor (majorité de vidéos)


# --------------------------------------------------------------------------- #
# API publique
# --------------------------------------------------------------------------- #
def scan(validation):
    """Énumère les médias d'une URL Picazor.

    `validation` = ValidationResult (is_valid, platform, url_type, value,
    original_url). Gère PROFILE (listing paginé), VIDEO (page de détail),
    LISTING (page globale).

    Retourne (items, error) :
      - items = list[dict] (≤ MAX_ITEMS) au schéma commun, ou None ;
      - error = str | None.
    Ne lève jamais : toute exception → (None, message). Dégradation gracieuse
    si Cloudflare bloque → (None, "Picazor (Cloudflare) blocked access.").
    """
    try:
        url = getattr(validation, "original_url", None) or getattr(validation, "value", "")
        url_type = getattr(validation, "url_type", None)
        url_type_name = getattr(url_type, "name", str(url_type)) if url_type else ""

        parsed = _parse_picazor_url(url)
        creator = parsed.get("creator") or getattr(validation, "value", "") or "picazor"

        # --- Média unique (page de détail) --------------------------------- #
        if url_type_name == "VIDEO" or parsed["type"] == "media":
            html, err = _request_html(url)
            if err:
                return None, err
            items = _parse_detail(html, creator)
            if not items:
                return None, "Picazor: no media found on the detail page."
            return items[:MAX_ITEMS], None

        # --- Listing global (videos/week, models, ...) --------------------- #
        if url_type_name == "LISTING" or parsed["type"] == "listing":
            html, err = _request_html(url)
            if err:
                return None, err
            items = _parse_listing(html, creator)
            if not items:
                return None, "Picazor: no media found in this listing."
            return items[:MAX_ITEMS], None

        # --- Profil paginé (cas par défaut : PROFILE) ---------------------- #
        start_page = parsed.get("page", 1)
        all_items = []
        seen = set()
        total_pages = None
        page = start_page

        for _ in range(MAX_PAGES):
            page_url = f"{BASE_URL}/fr/{creator}" + (f"/page/{page}" if page > 1 else "")
            html, err = _request_html(page_url)
            if err:
                # Si on a déjà des items, on dégrade gracieusement sans planter.
                if all_items:
                    break
                return None, err

            if total_pages is None:
                max_index = _max_media_index(html, creator)
                total_pages = max(1, math.ceil(max_index / ITEMS_PER_PAGE)) if max_index else 1

            page_items = _parse_listing(html, creator)
            if not page_items:
                break

            for it in page_items:
                if it["url"] in seen:
                    continue
                seen.add(it["url"])
                all_items.append(it)
                if len(all_items) >= MAX_ITEMS:
                    break

            if len(all_items) >= MAX_ITEMS:
                break
            page += 1
            if page > total_pages:
                break

        if not all_items:
            return None, "Picazor: no media found for this profile."
        return all_items[:MAX_ITEMS], None

    except Exception as e:  # garde-fou ultime — ne jamais lever
        logger.exception("Picazor scan: erreur inattendue")
        return None, f"Picazor: unexpected error ({e})."


def download(url, dest_path):
    """Télécharge un média Picazor (vidéo ou image) en direct via curl_cffi.

    `url` = URL du média (mp4/jpg) OU page de détail à résoudre d'abord.
    `dest_path` = chemin de sortie SANS extension imposée (l'extension est
    ajustée selon le content-type). Écriture atomique .tmp -> final.

    Retourne (ok: bool, final_filename: str | None, error: str | None).
    Ne lève jamais.
    """
    try:
        from curl_cffi import requests as cf_requests
    except ImportError:
        return False, None, "Picazor needs the 'curl_cffi' package - install the scrape extras (Setup > Install everything)."

    try:
        dest_path = Path(dest_path)

        # Si l'URL est une page de détail Picazor (pas un média direct), la résoudre.
        resolved_url = url
        if "/uploads/" not in url and "picazor.com" in url.lower():
            parsed = _parse_picazor_url(url)
            creator = parsed.get("creator", "picazor")
            html, err = _request_html(url)
            if err:
                return False, None, err
            detail = _parse_detail(html, creator)
            if not detail:
                return False, None, "Picazor: media not found on the detail page."
            resolved_url = detail[0]["url"]

        from ..netfetch import _validate_public_http_url
        ok_url, ssrf_err = _validate_public_http_url(resolved_url)
        if not ok_url:
            return False, None, ssrf_err or "Picazor: URL blocked (SSRF)."

        try:
            response = cf_requests.get(
                resolved_url, headers=_HEADERS, impersonate="chrome",
                timeout=DOWNLOAD_TIMEOUT, stream=True,
            )
        except Exception as e:
            logger.warning("Picazor download: échec requête %s: %s", resolved_url, e)
            return False, None, f"Picazor: download failed ({e})."

        status = getattr(response, "status_code", 0)
        content_type = ""
        try:
            content_type = response.headers.get("content-type", "") or ""
        except Exception:
            content_type = ""

        if status in (401, 404, 410):
            return False, None, f"Picazor: resource unavailable (HTTP {status})."
        if status == 403 or status == 429 or status == 503:
            return False, None, "Picazor (Cloudflare) blocked access."
        if status >= 400:
            return False, None, f"Picazor: HTTP {status} response."

        # Une réponse HTML n'est PAS un média (Cloudflare ou page d'erreur).
        if "text/html" in content_type.lower():
            return False, None, "Picazor: HTML response instead of media (access blocked?)."

        # Extension finale selon le content-type (fallback : URL).
        final_ext = _ext_from_content_type(content_type, resolved_url)
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
            return False, None, f"Picazor: write error ({e})."

        # Fichier vide = échec.
        try:
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                tmp_path.unlink(missing_ok=True)
                return False, None, "Picazor: downloaded file is empty."
        except OSError:
            return False, None, "Picazor: downloaded file is invalid."

        # Renommage atomique .tmp -> final.
        try:
            os.replace(tmp_path, final_path)
        except OSError as e:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return False, None, f"Picazor: finalization error ({e})."

        return True, final_path.name, None

    except Exception as e:  # garde-fou ultime — ne jamais lever
        logger.exception("Picazor download: erreur inattendue")
        return False, None, f"Picazor: unexpected error ({e})."


from .base import Source, Capabilities, Match
from . import registry


class PicazorSource(Source):
    name = 'picazor'
    priority = 100
    category = 'image'   # classé image (choix produit) → ouvert aux non-admins
    capabilities = Capabilities(can_enumerate_profile=True,
                                media_kinds=frozenset({'video', 'image'}),
                                own_downloader=True)

    def match(self, url):
        from ..validators import url_validator, Platform
        result = url_validator.validate_url(url)
        if result.is_valid and result.platform == Platform.PICAZOR:
            return Match(url=url, validation=result)
        # URLs directes CDN (/uploads/...) : le validateur ne les reconnaît pas
        # (pas de pattern /fr/{creator}/...) mais elles sont bien sur picazor.com
        # et nécessitent notre downloader curl_cffi (même comportement que l'ancien
        # dispatch host-based host.endswith('picazor.com')).
        if result.platform == Platform.PICAZOR:
            return Match(url=url, validation=None)
        return None

    def scan(self, match):
        return scan(match.validation)

    def download(self, url, dest_base):
        return download(url, dest_base)   # curl_cffi existant


registry.register(PicazorSource())
