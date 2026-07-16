# app/scrape/sources/fapello.py
"""Source Fapello (fapello.com + miroirs de langue fr./de./es.…).

Ex. : https://fr.fapello.com/katyuska-moonfox-40/  (page modèle = tout le set)

gallery-dl 1.32.3 gère nativement `fapello.com/<model>/` : la page modèle énumère
une FILE de posts (Message.Queue, type 6) ; chaque post porte 1 média DIRECT
(type 3, `fapello.com/content/.../xxx.jpg|mp4`). MÊME forme qu'une catégorie
PornPics → on hérite de `_PhotoSiteSource` (pagination `--chapter-range`, la
récursion type-6 de gdl.enumerate collecte les médias des posts).

⚠️ gallery-dl n'accepte QUE l'hôte canonique `(www.)?fapello.com` : un miroir de
langue comme `fr.fapello.com` renvoie « Unsupported URL ». On normalise donc
l'hôte AVANT l'énumération. Le download (URL CDN directe avec extension) passe par
l'extracteur `directlink` de gallery-dl → le downloader hérité de GalleryDlSource
suffit (pas de fetch direct comme Civitai).
"""
from urllib.parse import urlparse, urlunparse

from ..validators import Platform
from .image_sites import _PhotoSiteSource
from . import registry


def _canonical_fapello_url(url):
    """Mappe tout miroir de langue Fapello (fr./de./es.…fapello.com, www.) vers
    l'hôte canonique `fapello.com` que gallery-dl reconnaît. Non-fapello → inchangé."""
    try:
        p = urlparse(url)
    except Exception:
        return url
    host = (p.hostname or '').lower()
    if host == 'fapello.com' or host.endswith('.fapello.com'):
        # Reconstruit le netloc en fapello.com (drop sous-domaine + éventuel port/userinfo).
        return urlunparse((p.scheme, 'fapello.com', p.path, p.params, p.query, p.fragment))
    return url


class FapelloSource(_PhotoSiteSource):
    name = 'fapello'
    platform_enum = Platform.FAPELLO
    # category='image' + capabilities (image+video, own_downloader, polite) hérités de
    # _PhotoSiteSource : agrégateur de contenu mixte, même classement que les autres
    # sources 'image' de ce module (PornPics, etc.).
    # gallery_cap RELEVÉ à 24 (défaut hérité = 8) : chez Fapello un « post » (chapitre) = 1
    # SEUL média, alors que chez PornPics un chapitre = 1 galerie de dizaines d'images. À 8,
    # « Charger plus » ne rendait que 8 médias → peu efficace. 24 = ~15 s (1 sous-processus
    # gallery-dl par post, séquentiel) ; page N = posts 24N+1..24N+24.
    gallery_cap = 24

    def scan(self, match):
        # Normalise l'hôte (fr.fapello.com → fapello.com) puis délègue la pagination
        # chapter-range à _PhotoSiteSource. (Le download opère sur des URLs CDN déjà
        # canoniques renvoyées par le scan → pas besoin de normaliser là.)
        match.url = _canonical_fapello_url(match.url)
        return super().scan(match)


registry.register(FapelloSource())
