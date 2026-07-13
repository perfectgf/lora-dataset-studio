# app/scrape/sources/image_sites.py
"""Source « images par catégorie » — PornPics, en VRAIES PHOTOS (surtout pas de
boorus anime/dessin).

    • PornPics (pornpics.com) — galeries photo par CATÉGORIE / tag / pornstar.
      Ex : pornpics.com/teens/ , pornpics.com/lingerie/ , pornpics.com/tags/<tag>/

Deux modes de scan sur un LISTING (catégorie/tag/recherche) :

  • covers (DÉFAUT) : remonte la VIGNETTE que la page affiche pour chaque galerie
    — l'image choisie par le site comme représentative du mot-clé (observé :
    jamais la 1re de l'album — _009_, _113_…). Page 0 = les tuiles du HTML ;
    pages suivantes = l'endpoint AJAX offset/limit du site (celui de
    l'infinite-scroll, il renvoie g_url + t_url_460 + desc). L'URL pleine
    résolution s'obtient en permutant le segment de taille /460/ → /1280/.
    Zéro gallery-dl : 1 requête HTTP par page.
  • albums (case « Scan full albums ») : gallery-dl énumère les galeries du
    listing (`PornpicsCategoryExtractor`) puis chaque album EN ENTIER.

L'URL directe d'une galerie (/galleries/…) rend toujours tout l'album (gallery-dl,
médias top-level). Si le parse covers ne donne rien (layout du site changé,
curl_cffi absent), repli silencieux sur gallery-dl borné à 1 image/album.

NB historique : Realbooru + ImageFap retirés (sites de mauvaise qualité / risque
sécu, demande utilisateur 2026-06-27) ; Motherless retiré avant (DNS injoignable).
"""
import logging
import re
from urllib.parse import urlparse

from ..validators import Platform
from .base import Capabilities
from .gdl_source import GalleryDlSource
from . import gdl
from . import registry

logger = logging.getLogger(__name__)

# Contenu public (pas d'auth), images + GIF/vidéo, polis (gros listings).
# Téléchargement assuré par gallery-dl (own_downloader).
_PHOTO_CAPS = Capabilities(
    can_enumerate_profile=True,
    polite=True,
    media_kinds=frozenset({'image', 'video'}),
    own_downloader=True,
)

# Plafond d'items remontés par un scan. Bien plus haut que le défaut gdl (120) :
# une catégorie PornPics pagine sur PLUSIEURS galeries (gdl.enumerate récurse dans
# les albums type 6) et en mode « Scan full albums » on veut la fournée entière,
# pas juste la 1re galerie. gallery-dl auto-pagine ; --range 1-N borne juste le
# total (les --simulate restent rapides, métadonnées seules). Par défaut (covers
# seulement, cf. scan()) le plafond effectif est gallery_cap covers par page.
_PHOTO_SCAN_MAX = 400


# Nombre de GALERIES récupérées par fournée (= une « page » du bouton « Charger plus »).
# Une page catégorie/tag/recherche empile des galeries via `Message.Queue` (chapitres
# gallery-dl), que `--range` (image-range) NE borne PAS — seul `--chapter-range` le fait.
# Sans ça, gallery-dl pagine la catégorie ENTIÈRE (des centaines de pages, ~1 s chacune
# avec le request_interval pornpics) et le scan timeoute (60 s) à 0 média.
# Aligné sur gdl DEFAULT_MAX_ALBUMS : le top-level renvoie ≤ cap galeries, gdl.enumerate
# récurse ensuite dans chacune (images bornées par `--range`).
_GALLERY_CAP = 8


# --- Mode covers : parse du listing (HTML page 0, AJAX ensuite) ---------------
_COVERS_PER_PAGE = 20     # taille de page du site (grille ET endpoint AJAX)

# Une tuile du listing : lien galerie puis vignette lazy-load. Les pages catégorie
# utilisent des guillemets doubles, les pages /popular/ etc. des simples — on
# accepte les deux. Fenêtres bornées pour ne jamais enjamber la tuile suivante.
_TILE_RE = re.compile(
    r"class=[\"']rel-link[\"']\s+href=[\"'](?P<href>[^\"']+)[\"']"
    r".{0,600}?data-src=[\"'](?P<thumb>[^\"']+)[\"']"
    r"(?:.{0,300}?\salt=[\"'](?P<alt>[^\"']*)[\"'])?",
    re.S)
_SIZE_SEG_RE = re.compile(r'/(?:300|460)/')


def _full_size(thumb_url):
    """URL pleine résolution d'une vignette CDN : /300/ ou /460/ → /1280/ (même
    image, même hash — vérifié : les pages galerie lient les originaux en /1280/)."""
    return _SIZE_SEG_RE.sub('/1280/', thumb_url, count=1)


def _listing_html(url):
    """HTML de la page listing (seam de test). Lève en cas d'échec réseau/HTTP."""
    from curl_cffi import requests as cf_requests
    r = cf_requests.get(url, impersonate='chrome', timeout=20)
    r.raise_for_status()
    return r.text


def _listing_json(url, offset):
    """Fournée JSON de l'endpoint AJAX du listing (seam de test). L'offset 0 ne
    répond PAS en JSON (d'où le HTML pour la page 0). Lève en cas d'échec."""
    from curl_cffi import requests as cf_requests
    r = cf_requests.get(url, params={'limit': _COVERS_PER_PAGE, 'offset': offset},
                        impersonate='chrome', timeout=20,
                        headers={'Accept': 'application/json, text/javascript, */*; q=0.01',
                                 'Referer': url,
                                 'X-Requested-With': 'XMLHttpRequest'})
    r.raise_for_status()
    return r.json()


def _covers_scan(url, page):
    """Vignettes du listing telles que la page les affiche. Retourne (items, None),
    ([] , None) quand la pagination est épuisée, ou (None, None) pour demander le
    REPLI gallery-dl (parse vide / réseau KO / curl_cffi absent) — ne lève jamais."""
    def item(full, thumb, title):
        return {'url': full, 'title': (title or '')[:200], 'thumbnail': thumb,
                'type': 'image', 'platform': 'pornpics'}
    try:
        items = []
        if page <= 0:
            for m in _TILE_RE.finditer(_listing_html(url)):
                href, thumb = m.group('href'), m.group('thumb')
                if '/galleries/' not in href:      # tuile hors grille (nav, pubs…)
                    continue
                items.append(item(_full_size(thumb), thumb, m.group('alt')))
            return (items, None) if items else (None, None)   # vide = layout changé → repli
        data = _listing_json(url, offset=page * _COVERS_PER_PAGE)
        if not isinstance(data, list):
            return None, None
        for g in data:
            thumb = g.get('t_url_460') or g.get('t_url')
            if thumb:
                items.append(item(_full_size(thumb), thumb, g.get('desc')))
        return items, None      # [] légitime ici : fin du listing
    except Exception as e:
        logger.warning('pornpics covers scan failed (%s) — falling back to gallery-dl', e)
        return None, None


class _PhotoSiteSource(GalleryDlSource):
    priority = 100
    capabilities = _PHOTO_CAPS
    scan_max_items = _PHOTO_SCAN_MAX
    gallery_cap = _GALLERY_CAP
    paginated = True   # « Charger plus » : chaque page = la fournée de galeries suivante
    category = 'image'  # galeries photo → ouvert aux non-admins

    def scan(self, match):
        # Fenêtre de galeries de la page demandée (match.page, 0-based, posé par /scan) :
        # page 0 → galeries 1-cap, page 1 → cap+1..2·cap, etc. `--chapter-range` borne
        # l'énumération des galeries (sinon timeout, cf. _GALLERY_CAP) ; `--range` (via
        # max_items relevé) borne les images PAR galerie (le défaut gdl plafonnerait à 120).
        page = max(0, getattr(match, 'page', 0) or 0)
        start = page * self.gallery_cap + 1
        end = (page + 1) * self.gallery_cap
        extra = ['--chapter-range', f'{start}-{end}']
        if self.gdl_opts:
            extra = list(self.gdl_opts) + extra
        # match.include_albums False = REPLI covers (le chemin normal des covers est
        # _covers_scan, cf. PornpicsSource.scan) : borner à 1 image/album vaut mieux
        # que déverser les albums entiers. L'URL directe d'une galerie n'est pas
        # concernée (médias top-level, per_album ne borne que la récursion type 6).
        include_albums = bool(getattr(match, 'include_albums', False))
        return gdl.enumerate(match.url, platform=self.name,
                             max_items=self.scan_max_items,
                             max_albums=self.gallery_cap,
                             cookies=self._cookies(), extra_opts=extra,
                             per_album=None if include_albums else 1)


class PornpicsSource(_PhotoSiteSource):
    name = 'pornpics'
    platform_enum = Platform.PORNPICS
    # gdl_opts laissé à None : la fenêtre `--chapter-range` est calculée par scan() selon
    # match.page. (download() opère sur une URL d'image directe → pas de chapitres.)

    def scan(self, match):
        # Listing en mode covers (défaut) → les vignettes de la page, sans gallery-dl.
        # « Scan full albums » coché, URL directe d'une galerie, ou parse covers
        # impossible → chemin gallery-dl hérité (_PhotoSiteSource.scan).
        include_albums = bool(getattr(match, 'include_albums', False))
        if not include_albums and '/galleries/' not in urlparse(match.url).path:
            items, err = _covers_scan(match.url, max(0, getattr(match, 'page', 0) or 0))
            if items is not None:
                return items, err
        return super().scan(match)


registry.register(PornpicsSource())
