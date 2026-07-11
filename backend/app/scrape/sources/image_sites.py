# app/scrape/sources/image_sites.py
"""Source « images par catégorie » gérée par gallery-dl — l'équivalent images de
RedGifs, en VRAIES PHOTOS (surtout pas de boorus anime/dessin).

    • PornPics (pornpics.com) — galeries photo par CATÉGORIE / tag / pornstar.
      Ex : pornpics.com/teens/ , pornpics.com/lingerie/ , pornpics.com/tags/<tag>/

gallery-dl (extracteur `pornpics`) reconnaît nativement ces pages catégorie
(`PornpicsCategoryExtractor`, pattern `/<segment>/`) : la catégorie énumère une
file de galeries, chaque galerie énumère ses images. Contenu public (pas d'auth).

NB historique : Realbooru + ImageFap retirés (sites de mauvaise qualité / risque
sécu, demande utilisateur 2026-06-27) ; Motherless retiré avant (DNS injoignable).

L'énumération du listing et le téléchargement sont délégués à gdl.py via
GalleryDlSource.
"""
from ..validators import Platform
from .base import Capabilities
from .gdl_source import GalleryDlSource
from . import gdl
from . import registry

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
# les albums type 6) et on veut la catégorie entière, pas juste la 1re galerie.
# gallery-dl auto-pagine ; --range 1-N borne juste le total (les --simulate restent
# rapides, métadonnées seules).
_PHOTO_SCAN_MAX = 400


# Nombre de GALERIES récupérées par fournée (= une « page » du bouton « Charger plus »).
# Une page catégorie/tag/recherche empile des galeries via `Message.Queue` (chapitres
# gallery-dl), que `--range` (image-range) NE borne PAS — seul `--chapter-range` le fait.
# Sans ça, gallery-dl pagine la catégorie ENTIÈRE (des centaines de pages, ~1 s chacune
# avec le request_interval pornpics) et le scan timeoute (60 s) à 0 média.
# Aligné sur gdl DEFAULT_MAX_ALBUMS : le top-level renvoie ≤ cap galeries, gdl.enumerate
# récurse ensuite dans chacune (images bornées par `--range`).
_GALLERY_CAP = 8


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
        return gdl.enumerate(match.url, platform=self.name,
                             max_items=self.scan_max_items,
                             max_albums=self.gallery_cap,
                             cookies=self._cookies(), extra_opts=extra)


class PornpicsSource(_PhotoSiteSource):
    name = 'pornpics'
    platform_enum = Platform.PORNPICS
    # gdl_opts laissé à None : la fenêtre `--chapter-range` est calculée par scan() selon
    # match.page. (download() opère sur une URL d'image directe → pas de chapitres.)


registry.register(PornpicsSource())
