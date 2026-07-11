# app/scrape/sources/instagram.py
"""Scraper Instagram — énumération des médias d'un profil / post / reel.

Port autonome du scraper Instagram du projet `redgifs_downloader`
(`api/instagram.py`), réduit au strict nécessaire pour l'app ImageGen :

  * pas de dépendance au module `config`/`settings` ni à des instances globales
    (constantes en dur, voir `SCAN_LIMIT` / `PROFILE_SCAN_TIMEOUT`) ;
  * auth réutilisée du projet source : session instaloader détectée sur disque,
    sinon auto-import des cookies du navigateur via `browser_cookie3` ;
  * dégradation gracieuse : Instagram est anti-bot — toute erreur d'auth / 403 /
    rate-limit / login requis renvoie `(None, "<message FR court>")` au lieu de
    lever. `scan()` ne lève JAMAIS.

Contrat (cf. `app/scrape/sources/__init__.py`) :
    scan(validation) -> (items, error)
        items : list[dict] (≤ SCAN_LIMIT) au schéma commun, ou None si erreur ;
        error : str|None (message court FR).

Schéma d'un item :
    { 'url', 'title', 'thumbnail' (str|None), 'type' ('video'|'image'),
      'platform' ('instagram') }

L'`url` renvoyée est l'URL de la PAGE instagram.com (post/reel) : elle est
stable et acceptée par /api/scrape/download (yt-dlp + cookies navigateur), au
contraire des URLs CDN signées qui expirent vite.
"""
import time
import logging
from pathlib import Path

try:
    import instaloader
    INSTALOADER_AVAILABLE = True
except ImportError:  # pragma: no cover - dépendance absente
    instaloader = None
    INSTALOADER_AVAILABLE = False

try:
    import browser_cookie3
    BROWSER_COOKIE3_AVAILABLE = True
except ImportError:  # pragma: no cover - dépendance absente
    browser_cookie3 = None
    BROWSER_COOKIE3_AVAILABLE = False

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constantes (en dur — module autonome, aucune lecture de settings).
# --------------------------------------------------------------------------- #
SCAN_LIMIT = 50               # borne dure sur le nombre d'items retournés
PROFILE_SCAN_TIMEOUT = 60     # secondes — plafond global d'un scan de profil
SESSION_TIMEOUT = 10          # secondes — timeout HTTP de la session instaloader

# Message d'erreur unique pour tout refus côté Instagram (auth / 403 / rate-limit).
_AUTH_ERROR = "Instagram a refusé l'accès (auth requise / rate-limit)."

# Sous-chaînes signalant un blocage anti-bot dans un message d'exception.
_BLOCK_HINTS = ("429", "403", "forbidden", "too many", "login", "rate", "checkpoint")


# --------------------------------------------------------------------------- #
# Auth — session instaloader + auto-import cookies navigateur.
# --------------------------------------------------------------------------- #
def _detect_session_username():
    """Détecte un fichier de session instaloader existant sur disque.

    Instaloader range ses sessions dans ``~/.config/instaloader/session-USER``.
    Retourne le username associé, ou None si aucune session trouvée.
    """
    try:
        config_dir = Path.home() / ".config" / "instaloader"
        if config_dir.exists():
            for f in config_dir.iterdir():
                if f.name.startswith("session-"):
                    username = f.name[len("session-"):]
                    if username:
                        logger.info("Session Instagram détectée : %s", username)
                        return username
    except Exception as e:  # pragma: no cover - I/O improbable en test
        logger.debug("Détection session Instagram échouée : %s", e)
    return None


def _auto_import_browser_cookies(loader):
    """Injecte les cookies Instagram du navigateur (Firefox puis Chrome).

    Retourne True si un cookie `sessionid` (preuve de login) a été importé.
    Ne lève jamais : tout échec → False.
    """
    if not BROWSER_COOKIE3_AVAILABLE:
        return False

    for browser_name, browser_fn in (
        ("Firefox", browser_cookie3.firefox),
        ("Chrome", browser_cookie3.chrome),
    ):
        try:
            cookie_list = list(browser_fn(domain_name="instagram.com"))
            if not cookie_list:
                continue
            # Le cookie `sessionid` atteste qu'on est connecté.
            if not any(c.name == "sessionid" for c in cookie_list):
                logger.debug("%s : cookies trouvés mais pas de sessionid", browser_name)
                continue
            session = loader.context._session
            for cookie in cookie_list:
                session.cookies.set(cookie.name, cookie.value, domain=cookie.domain)
            logger.info("Auto-import cookies %s : %d cookies (sessionid présent)",
                        browser_name, len(cookie_list))
            return True
        except Exception as e:
            logger.debug("Auto-import %s échoué : %s", browser_name, e)
            continue
    return False


def _build_loader():
    """Construit une instance Instaloader authentifiée (session ou cookies).

    Retourne le loader (authentifié ou non — Instagram acceptera certains
    profils publics sans session). Lève si instaloader est indisponible.
    """
    loader = instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
        quiet=True,
        max_connection_attempts=1,
    )
    # Timeout HTTP court pour éviter qu'un scan ne pende indéfiniment.
    try:
        loader.context._session.timeout = SESSION_TIMEOUT
    except (AttributeError, TypeError) as e:  # pragma: no cover
        logger.debug("Config timeout session échouée : %s", e)

    # 1) Charger une session instaloader existante si présente.
    session_loaded = False
    username = _detect_session_username()
    if username:
        try:
            loader.load_session_from_file(username)
            logger.info("Session Instagram chargée : %s", username)
            session_loaded = True
        except Exception as e:
            logger.warning("Chargement session Instagram échoué : %s", e)

    # 2) Sinon, auto-import des cookies du navigateur.
    if not session_loaded and _auto_import_browser_cookies(loader):
        session_loaded = True

    if not session_loaded:
        logger.warning(
            "Aucune session Instagram (ni fichier, ni cookies navigateur). "
            "Les profils privés / rate-limit échoueront."
        )
    return loader


# --------------------------------------------------------------------------- #
# Helpers de mapping post -> item(s) du schéma commun.
# --------------------------------------------------------------------------- #
def _post_page_url(shortcode):
    """URL stable de la page d'un post/reel (acceptée par yt-dlp en download)."""
    return f"https://www.instagram.com/p/{shortcode}/"


def _items_from_post(post, original_url=None):
    """Convertit un `instaloader.Post` en liste d'items du schéma commun.

    Gère les carrousels (GraphSidecar → plusieurs médias). Chaque accès aux
    attributs instaloader peut lever ; on isole pour ne pas perdre tout le post.
    `original_url` (si fourni) prime sur l'URL reconstruite — utile pour un
    post/reel unique fourni par l'utilisateur (préserve /reel/ vs /p/).
    """
    items = []
    try:
        shortcode = post.shortcode
    except Exception:
        return items

    page_url = original_url or _post_page_url(shortcode)

    try:
        typename = post.typename
    except Exception:
        typename = None

    # Carrousel : plusieurs slides (images et/ou vidéos).
    if typename == "GraphSidecar":
        try:
            nodes = list(post.get_sidecar_nodes())
        except Exception:
            nodes = []
        for idx, node in enumerate(nodes):
            try:
                is_video = bool(node.is_video)
                thumbnail = node.display_url
            except Exception:
                continue
            items.append({
                "url": page_url,
                "title": f"Post {shortcode} (slide {idx + 1})",
                "thumbnail": thumbnail,
                "type": "video" if is_video else "image",
                "platform": "instagram",
            })
        return items

    # Post simple : image ou vidéo unique.
    try:
        is_video = bool(post.is_video)
        thumbnail = post.url
    except Exception:
        is_video = False
        thumbnail = None

    items.append({
        "url": page_url,
        "title": ("Reel " if is_video else "Post ") + str(shortcode),
        "thumbnail": thumbnail,
        "type": "video" if is_video else "image",
        "platform": "instagram",
    })
    return items


def _looks_like_block(exc):
    """True si l'exception ressemble à un blocage anti-bot (auth/403/rate-limit)."""
    msg = str(exc).lower()
    return any(hint in msg for hint in _BLOCK_HINTS)


# --------------------------------------------------------------------------- #
# Scans par type d'URL.
# --------------------------------------------------------------------------- #
def _scan_profile(loader, username):
    """Énumère les SCAN_LIMIT derniers médias d'un profil. Retourne (items, error)."""
    try:
        profile = instaloader.Profile.from_username(loader.context, username)
    except instaloader.ProfileNotExistsException:
        return None, f"Profil Instagram introuvable : {username}."
    except Exception as e:
        # ConnectionException / LoginRequired / Forbidden / TooManyRequests / ...
        logger.warning("Chargement profil %s échoué : %s", username, e)
        return None, _AUTH_ERROR

    items = []
    started = time.time()
    try:
        for post in profile.get_posts():
            if len(items) >= SCAN_LIMIT:
                break
            if time.time() - started > PROFILE_SCAN_TIMEOUT:
                logger.warning("Timeout scan profil %s (%ds), %d items.",
                               username, PROFILE_SCAN_TIMEOUT, len(items))
                break
            try:
                for item in _items_from_post(post):
                    items.append(item)
                    if len(items) >= SCAN_LIMIT:
                        break
            except Exception as e:
                # Un post cassé ne doit pas tuer le scan entier.
                logger.debug("Post ignoré (%s) : %s", username, e)
                continue
    except Exception as e:
        # Erreur pendant l'itération paginée (souvent rate-limit en cours de route).
        if items:
            # On a déjà des items utiles → on les retourne sans erreur.
            logger.warning("Itération profil %s interrompue après %d items : %s",
                           username, len(items), e)
            return items[:SCAN_LIMIT], None
        logger.warning("Itération profil %s échouée : %s", username, e)
        return None, _AUTH_ERROR

    if not items:
        return None, f"Aucun média trouvé pour le profil {username}."
    return items[:SCAN_LIMIT], None


def _scan_single(loader, shortcode, original_url=None):
    """Récupère un post/reel unique. Retourne (items, error)."""
    try:
        post = instaloader.Post.from_shortcode(loader.context, shortcode)
    except instaloader.QueryReturnedNotFoundException:
        return None, f"Publication Instagram introuvable : {shortcode}."
    except Exception as e:
        logger.warning("Chargement post %s échoué : %s", shortcode, e)
        return None, _AUTH_ERROR

    try:
        items = _items_from_post(post, original_url=original_url)
    except Exception as e:
        logger.warning("Conversion post %s échouée : %s", shortcode, e)
        return None, _AUTH_ERROR

    if not items:
        return None, f"Aucun média exploitable pour {shortcode}."
    return items[:SCAN_LIMIT], None


# --------------------------------------------------------------------------- #
# Point d'entrée public.
# --------------------------------------------------------------------------- #
def scan(validation):
    """Énumère les médias téléchargeables d'une URL Instagram validée.

    `validation` : ValidationResult (champs utilisés : platform, url_type,
    value, original_url). Gère url_type PROFILE / POST / REEL.

    Retourne (items, error) :
      * items : list[dict] (≤ SCAN_LIMIT) au schéma commun, ou None si erreur ;
      * error : str|None (message court FR).

    Ne lève JAMAIS : toute exception est capturée → (None, "<message>").
    """
    # Import paresseux pour éviter un cycle d'import au chargement du package.
    from ..validators import URLType

    if not INSTALOADER_AVAILABLE:
        return None, "Module instaloader indisponible (dépendance manquante)."

    try:
        url_type = getattr(validation, "url_type", None)
        value = getattr(validation, "value", None)
        original_url = getattr(validation, "original_url", None)

        if not value:
            return None, "URL Instagram invalide (cible introuvable)."

        try:
            loader = _build_loader()
        except Exception as e:
            logger.warning("Construction loader Instagram échouée : %s", e)
            return None, _AUTH_ERROR

        if url_type == URLType.PROFILE:
            return _scan_profile(loader, value)
        if url_type in (URLType.POST, URLType.REEL):
            return _scan_single(loader, value, original_url=original_url)

        return None, f"Type d'URL Instagram non géré : {getattr(url_type, 'value', url_type)}."

    except Exception as e:
        # Filet de sécurité absolu — scan() ne doit jamais propager d'exception.
        logger.warning("Erreur inattendue scan Instagram : %s", e)
        if _looks_like_block(e):
            return None, _AUTH_ERROR
        return None, "Erreur lors du scan Instagram."


from .base import Source, Capabilities, Match
from . import registry


class InstagramSource(Source):
    name = 'instagram'
    priority = 100
    category = 'image'   # classé image (choix produit) → ouvert aux non-admins
    capabilities = Capabilities(can_enumerate_profile=True, needs_auth=True, own_downloader=False)

    def match(self, url):
        from ..validators import url_validator, Platform
        result = url_validator.validate_url(url)
        if result.is_valid and result.platform == Platform.INSTAGRAM:
            return Match(url=url, validation=result)
        return None

    def scan(self, match):
        return scan(match.validation)


registry.register(InstagramSource())
