# app/scrape/sources/civitai.py
r"""Source Civitai (civitai.com / civitai.red) — listings d'images par tag/recherche.

Ex. : https://civitai.red/images?tags=5169 , https://civitai.com/images?tags=N

gallery-dl 1.32.3 gère NATIVEMENT les deux domaines (BASE_PATTERN `civitai\.(?:red|com)`)
et route `/images?...` vers `CivitaiImagesExtractor` (parse la query `tags=`, types=image).
Les images sont des médias DIRECTS (Message.Url, type 3, `image-b2.civitai.com/.../orig`)
→ bornées par `--range` (image-range). La pagination « Charger plus » est donc une FENÊTRE
d'image-range (page 0 = 1-100, page 1 = 101-200, …), ≠ pornpics qui empile des galeries
(`--chapter-range`).

NSFW : l'API tRPC renvoie `browsingLevel` 31 (tous niveaux) par défaut, MAIS le serveur
exige une `api-key` (Bearer) pour réellement servir le contenu adulte. L'app lançant
gallery-dl avec `--ignore-config`, on passe la clé en `-o api-key=<token>` (cf. gdl_opts).
"""
import os

from ..validators import Platform
from .base import Capabilities
from .gdl_source import GalleryDlSource, resolve_cookies
from . import gdl
from . import registry

# token.txt hérité d’un outil local (dernier recours si ni
# l'env ni le dossier cookies admin ne fournissent la clé.
_SKILL_TOKEN_PATH = os.path.expanduser('~/.claude/skills/civitai-download/token.txt')


def civitai_api_key():
    """Clé API Civitai (Bearer) débloquant le NSFW, ou None (→ scan SFW seulement).

    Précédence : env `CIVITAI_API_KEY` > `<SCRAPE_COOKIES_DIR>/civitai_api_key.txt`
    > token.txt du skill civitai-download. Jamais committé (lu au runtime)."""
    env = (os.environ.get('CIVITAI_API_KEY') or '').strip()
    if env:
        return env
    for path in (resolve_cookies('civitai_api_key'), _SKILL_TOKEN_PATH):
        try:
            if path and os.path.isfile(path):
                val = open(path, encoding='utf-8').read().strip()
                if val:
                    return val
        except OSError:
            pass
    return None


# Contenu public-via-API, images + vidéos, polis (gros listings). Download EN DIRECT
# (curl_cffi durci) : les médias renvoyés par le scan sont des URLs CDN directes SANS
# extension (image-b2.civitai.com/.../original) que gallery-dl REFUSE (cf. download()).
_CIVITAI_CAPS = Capabilities(
    can_enumerate_profile=True,
    polite=True,
    media_kinds=frozenset({'image', 'video'}),
    own_downloader=True,
)

# Content-types servis par le CDN Civitai (image-b2.civitai.com) : image/* + video/*
# (Civitai héberge aussi des animations). Sert l'allowlist du fetch durci ET le mapping
# d'extension (les URLs CDN finissent par /original, sans extension de fichier).
_CIVITAI_MEDIA_TYPES = frozenset({
    'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp', 'image/avif',
    'video/mp4', 'video/webm', 'video/quicktime',
})
_CT_EXT = {
    'image/jpeg': '.jpg', 'image/jpg': '.jpg', 'image/png': '.png',
    'image/gif': '.gif', 'image/webp': '.webp', 'image/avif': '.avif',
    'video/mp4': '.mp4', 'video/webm': '.webm', 'video/quicktime': '.mov',
}
# fetch_hardened_bytes renvoie un code court (reason) → message FR exploitable.
_FETCH_REASON_MSG = {
    'redirect': 'Civitai : redirection refusée (sécurité anti-SSRF).',
    'status': 'Civitai : ressource indisponible (HTTP non-200).',
    'type': "Civitai : type de média inattendu (ni image ni vidéo).",
    'toolarge': 'Civitai : média trop volumineux.',
    'fetch': 'Civitai : échec de la requête CDN.',
    'no_curl': 'curl_cffi non disponible (dépendance manquante).',
    'noimage': "Civitai : le contenu n'est pas une image valide.",
}


def _ext_for(content_type, url):
    """Extension de fichier depuis le content-type (repli : extension d'URL, sinon .png).
    Civitai sert majoritairement du PNG → défaut raisonnable si tout échoue."""
    ct = (content_type or '').split(';', 1)[0].strip().lower()
    if ct in _CT_EXT:
        return _CT_EXT[ct]
    from urllib.parse import urlparse
    ext = os.path.splitext(urlparse(url).path)[1].lower()
    if ext in ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.avif', '.mp4', '.webm', '.mov'):
        return '.jpg' if ext == '.jpeg' else ext
    return '.png'


class CivitaiSource(GalleryDlSource):
    name = 'civitai'
    platform_enum = Platform.CIVITAI
    priority = 100
    capabilities = _CIVITAI_CAPS
    paginated = True
    page_size = 100   # images par fournée « Charger plus » (l'API tRPC pagine ~100/page)
    category = 'image'  # images IA → ouvert aux non-admins

    @property
    def gdl_opts(self):
        # api-key (Bearer) pour le NSFW au SCAN uniquement. Absente → SFW. Le download
        # ne passe PAS par gallery-dl (cf. download()) : la clé n'y sert pas (le CDN
        # image-b2.civitai.com sert le média sans auth une fois l'URL connue).
        key = civitai_api_key()
        return ['-o', f'api-key={key}'] if key else None

    def download(self, url, dest_base):
        """Télécharge le média Civitai EN DIRECT (fetch durci), PAS via gallery-dl.

        Le scan renvoie des URLs CDN DIRECTES sans extension
        (`image-b2.civitai.com/.../original`). gallery-dl les REFUSE
        (« Unsupported URL », exit 64) : son extracteur civitai ne matche que les
        pages civitai.com, et le fallback directlink exige une URL terminée par une
        extension de fichier. On télécharge donc en direct via `fetch_hardened_bytes`
        (anti-SSRF, allow_redirects=False, content-type vérifié), l'extension venant
        du content-type. `download_service._finalize` revalide ensuite par magic-bytes
        + applique la garde image-only non-admin + l'antivirus → un non-admin ne
        récupère jamais une vidéo, quelle que soit l'extension produite ici."""
        from ..netfetch import MAX_DRIVER_BYTES, fetch_hardened_bytes
        ok, data, ctype, reason = fetch_hardened_bytes(
            url, allowed_types=_CIVITAI_MEDIA_TYPES, max_bytes=MAX_DRIVER_BYTES)
        if not ok or not data:
            return False, None, _FETCH_REASON_MSG.get(reason, f'Civitai : échec ({reason}).')
        dest_dir = os.path.dirname(dest_base)
        filename = os.path.basename(dest_base) + _ext_for(ctype, url)
        try:
            os.makedirs(dest_dir, exist_ok=True)
            with open(os.path.join(dest_dir, filename), 'wb') as f:
                f.write(data)
        except OSError as e:
            return False, None, f"Civitai : erreur d'écriture ({e})."
        return True, filename, None

    def scan(self, match):
        # Fenêtre d'images de la page demandée (match.page, 0-based, posé par /scan).
        # Images directes (type 3) → `image_range` borne le flux (le défaut gdl 120
        # plafonnerait, et il faut une fenêtre DÉCALÉE pour « Charger plus »).
        page = max(0, getattr(match, 'page', 0) or 0)
        start = page * self.page_size + 1
        end = (page + 1) * self.page_size
        return gdl.enumerate(match.url, platform=self.name,
                             max_items=self.page_size,
                             image_range=f'{start}-{end}',
                             cookies=self._cookies(), extra_opts=self.gdl_opts)


registry.register(CivitaiSource())
