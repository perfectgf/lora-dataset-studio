# app/scrape/validators.py
"""Validation d'URLs de scraping — détection de plateforme + type d'URL.

Port (quasi à l'identique) de `redgifs_downloader/api/validators.py` : pur
stdlib, aucune dépendance réseau. Détecte RedGifs / Instagram / Picazor et
délègue tout autre domaine http(s) à yt-dlp (Platform.GENERIC).
"""
import re
from typing import Optional, List
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import unquote, urlparse


class Platform(Enum):
    REDGIFS = "redgifs"
    INSTAGRAM = "instagram"
    PICAZOR = "picazor"
    EROME = "erome"
    # Retrait produit volontaire (dump/leak sites) : ces 4 membres restent
    # UNIQUEMENT pour que detect_platform() identifie encore l'hôte et que
    # validate_url() puisse renvoyer un refus explicite et nommé (cf.
    # _REMOVED_PLATFORMS) — aucune source ne les gère plus, aucun scan ni
    # téléchargement ne doit jamais les atteindre.
    COOMER = "coomer"
    KEMONO = "kemono"
    BUNKR = "bunkr"
    CYBERDROP = "cyberdrop"
    X = "x"
    TIKTOK = "tiktok"
    # Site d'images par catégorie (VRAIES photos) — énuméré via gallery-dl,
    # l'équivalent images de RedGifs. Pas de boorus anime/dessin (hors sujet).
    PORNPICS = "pornpics"
    # Civitai (civitai.com / civitai.red) — listings d'images par tag/recherche
    # (ex. /images?tags=5169), images directes énumérées via gallery-dl.
    CIVITAI = "civitai"
    # Fapello (fapello.com + miroirs de langue fr./de./es.…) — page modèle =
    # file de posts, chaque post 1 média direct. Énuméré via gallery-dl.
    FAPELLO = "fapello"
    # Reddit (reddit.com — subreddits, posts, liens de partage /s/) — LE filon
    # « vraies photos amateur » (r/OOTD, fit checks, photo dumps…) pour les
    # datasets de style/concept. Énuméré via gallery-dl.
    REDDIT = "reddit"
    # Sex.com — pinboard porno (chaque pin = UNE image). La recherche mot-clé
    # (/pics?search=…) passe par l'API JSON du site ; pins/boards via gallery-dl.
    SEXCOM = "sexcom"
    # Pexels — recherches, collections accessibles et photos SFW via l'API
    # officielle (PEXELS_API_KEY requise). Les profils ne sont pas exposés.
    PEXELS = "pexels"
    GENERIC = "generic"   # toute autre URL http(s) — déléguée à yt-dlp
    UNKNOWN = "unknown"


# Plateformes retirées du produit (dump/leak sites) : validate_url() les refuse
# explicitement avant toute résolution de source. Ne JAMAIS router ces membres
# vers un scan ou un téléchargement, générique ou dédié.
_REMOVED_PLATFORMS = frozenset({
    Platform.COOMER, Platform.KEMONO, Platform.BUNKR, Platform.CYBERDROP,
})


class URLType(Enum):
    PROFILE = "profile"
    VIDEO = "video"
    POST = "post"
    REEL = "reel"
    NICHE = "niche"
    LISTING = "listing"
    UNKNOWN = "unknown"


@dataclass
class ValidationResult:
    is_valid: bool
    platform: Platform
    url_type: URLType
    value: str               # username, video_id, etc.
    original_url: str
    error: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Sérialisation JSON-safe pour les réponses API."""
        return {
            'is_valid': self.is_valid,
            'platform': self.platform.value,
            'url_type': self.url_type.value,
            'value': self.value,
            'original_url': self.original_url,
            'error': self.error,
            'suggestions': self.suggestions,
        }


@dataclass(frozen=True)
class PexelsRoute:
    """Route publique Pexels strictement reconnue.

    ``value`` conserve le segment public pour ``ValidationResult`` tandis que
    ``api_value`` contient la valeur décodée et validée envoyée à l'API.
    Seules les recherches localisées portent un paramètre ``api_locale``.
    """
    kind: str
    value: str
    api_value: str
    api_locale: Optional[str] = None


_PEXELS_SEARCH_ROUTES = {
    None: ('search', None),
    'en-us': ('search', 'en-US'),
    'fr-fr': ('chercher', 'fr-FR'),
}


def parse_pexels_path(path: str) -> Optional[PexelsRoute]:
    """Parse les seules routes publiques mappables vers l'API Pexels.

    Les préfixes localisés acceptés sont volontairement limités à ceux
    testés ici. Une route ou une locale inconnue n'est jamais transformée en
    recherche de secours.
    """
    if not isinstance(path, str) or not re.fullmatch(
            r'/[^/]+(?:/[^/]+){1,2}/?', path):
        return None

    parts = path[1:-1] if path.endswith('/') else path[1:]
    segments = parts.split('/')
    locale = segments.pop(0) if segments[0] in _PEXELS_SEARCH_ROUTES else None
    if len(segments) != 2:
        return None

    route_name, public_value = segments
    if re.search(r'%(?![0-9A-Fa-f]{2})', public_value):
        return None
    decoded_value = unquote(public_value)
    if (not decoded_value.strip()
            or any(char in decoded_value for char in ('/', '\\'))
            or any(ord(char) < 32 or ord(char) == 127 for char in decoded_value)):
        return None

    expected_search, api_locale = _PEXELS_SEARCH_ROUTES[locale]
    if route_name == expected_search:
        return PexelsRoute('search', public_value, decoded_value, api_locale)

    if route_name == 'collections':
        if not re.fullmatch(r'[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*', decoded_value):
            return None
        collection_id = decoded_value.rsplit('-', 1)[-1]
        return PexelsRoute('collection', public_value, collection_id)

    if route_name == 'photo':
        match = re.fullmatch(
            r'(?:[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*-)?(\d+)', decoded_value)
        if match:
            return PexelsRoute('photo', public_value, match.group(1))

    return None


class URLValidator:
    """Validateur d'URLs : plateforme + type + valeur extraite."""

    PATTERNS = {
        Platform.REDGIFS: {
            "user": [
                r'redgifs\.com/users/([^/?]+)',
                r'redgifs\.com/@([^/?]+)',
            ],
            "video": [
                r'redgifs\.com/watch/([^/?]+)',
                r'redgifs\.com/([a-zA-Z]+)$',
            ],
            "niche": [
                r'redgifs\.com/niches/([^/?]+)',
            ],
        },
        Platform.INSTAGRAM: {
            "profile": [
                r'instagram\.com/([^/?]+)/?$',
                r'instagram\.com/([^/?]+)/reels',
            ],
            "post": [
                r'instagram\.com/p/([^/?]+)',
                r'instagram\.com/([^/]+)/p/([^/?]+)',
            ],
            "reel": [
                r'instagram\.com/reel/([^/?]+)',
                r'instagram\.com/([^/]+)/reel/([^/?]+)',
            ],
        },
        Platform.PICAZOR: {
            # /fr/{creator}/{N} = page de DÉTAIL du média N (pas un listing)
            "profile": [
                r'picazor\.com/([^/]+)/([^/?]+)/page/(\d+)',
                r'picazor\.com/([^/]+)/([^/?]+)/?$',
            ],
            "video": [
                r'picazor\.com/([^/]+)/([^/?]+)/(\d+)/?$',
            ],
            "listing": [
                r'picazor\.com/([^/]+)/(videos|models)/([^/]+)',
            ],
        },
        Platform.EROME: {
            # Album = conteneur de médias ; search = liste d'albums.
            "listing": [
                r'erome\.com/a/([^/?#]+)',
                r'erome\.com/search\?',
            ],
            # /{user} = page profil (liste d'albums de l'utilisateur).
            "profile": [
                r'erome\.com/([^/?#]+)/?$',
            ],
        },
    }

    VALID_DOMAINS = {
        Platform.REDGIFS: ['redgifs.com', 'www.redgifs.com'],
        Platform.INSTAGRAM: ['instagram.com', 'www.instagram.com'],
        Platform.PICAZOR: ['picazor.com', 'www.picazor.com'],
        Platform.EROME: ['erome.com', 'www.erome.com', 'fr.erome.com'],
        Platform.X: ['x.com', 'twitter.com', 'www.x.com', 'www.twitter.com', 'mobile.twitter.com'],
        Platform.TIKTOK: ['tiktok.com', 'www.tiktok.com', 'vm.tiktok.com'],
        Platform.PORNPICS: ['pornpics.com', 'www.pornpics.com'],
        Platform.SEXCOM: ['sex.com', 'www.sex.com'],
        Platform.CIVITAI: ['civitai.com', 'www.civitai.com', 'civitai.red', 'www.civitai.red'],
        Platform.PEXELS: ['pexels.com', 'www.pexels.com'],
        # Coomer/Kemono/Cyberdrop/Bunkr : PAS de VALID_DOMAINS (source retirée, cf.
        # _REMOVED_PLATFORMS) — validate_url() les refuse avant ce contrôle de toute
        # façon. _HOST_PLATFORMS ci-dessous les identifie encore (message de refus nommé).
    }

    # Domaines reconnus par host (host == d ou host.endswith('.'+d)).
    _HOST_PLATFORMS = [
        ('redgifs.com', Platform.REDGIFS),
        ('instagram.com', Platform.INSTAGRAM),
        ('picazor.com', Platform.PICAZOR),
        ('erome.com', Platform.EROME),
        # Sources retirées (cf. _REMOVED_PLATFORMS) : identifiées ici UNIQUEMENT pour
        # que validate_url() puisse renvoyer un refus nommé plutôt qu'un générique
        # « URL non reconnue » ou, pire, un repli silencieux vers le scraper générique.
        ('coomer.st', Platform.COOMER), ('coomer.su', Platform.COOMER),
        ('coomer.party', Platform.COOMER), ('coomer.cr', Platform.COOMER),
        ('kemono.cr', Platform.KEMONO), ('kemono.su', Platform.KEMONO),
        ('kemono.party', Platform.KEMONO),
        ('cyberdrop.me', Platform.CYBERDROP), ('cyberdrop.cr', Platform.CYBERDROP),
        ('cyberdrop.to', Platform.CYBERDROP),
        ('x.com', Platform.X), ('twitter.com', Platform.X),
        ('tiktok.com', Platform.TIKTOK),
        ('pornpics.com', Platform.PORNPICS),
        ('sex.com', Platform.SEXCOM),
        ('civitai.com', Platform.CIVITAI),
        ('civitai.red', Platform.CIVITAI),
        ('pexels.com', Platform.PEXELS),
        # fapello.com + tout miroir de langue (fr./de./es.…) via endswith('.fapello.com').
        # Volontairement ABSENT de VALID_DOMAINS → la garde stricte de domaine n'écarte
        # pas les sous-domaines de langue (la source normalise l'hôte avant gallery-dl).
        ('fapello.com', Platform.FAPELLO),
        # reddit.com (www/old/new/sh.…) + redd.it (shortener + CDN i.redd.it). Comme
        # Fapello : ABSENT de VALID_DOMAINS, la source canonicalise vers www.reddit.com
        # (résolution des liens de partage /s/ incluse) avant gallery-dl.
        ('reddit.com', Platform.REDDIT),
        ('redd.it', Platform.REDDIT),
    ]

    @staticmethod
    def detect_platform(url: str) -> Platform:
        host = (urlparse(url).hostname or '').lower()
        if not host:
            return Platform.UNKNOWN
        # Bunkr (source retirée, cf. _REMOVED_PLATFORMS) : TLDs rotatifs → le label
        # 'bunkr' doit être le SLD (avant-dernier label), ex : bunkr.cr, bunkrr.su —
        # on vérifie labels[-2] pour éviter les sous-domaines trompeurs comme
        # bunkr.cr.evil.com où 'bunkr' est un sous-domaine, pas le SLD. Uniquement
        # pour que validate_url() puisse renvoyer un refus nommé, jamais pour router
        # vers un scan ou un téléchargement.
        labels = host.split('.')
        if len(labels) >= 2 and labels[-2].startswith('bunkr'):
            return Platform.BUNKR
        for domain, platform in URLValidator._HOST_PLATFORMS:
            if host == domain or host.endswith('.' + domain):
                return platform
        return Platform.UNKNOWN

    @classmethod
    def validate_url(cls, url: str) -> ValidationResult:
        url = (url or "").strip()

        if not url.startswith(('http://', 'https://')):
            if '.' in url and url:
                url = f"https://{url}"
            else:
                return ValidationResult(
                    is_valid=False, platform=Platform.UNKNOWN, url_type=URLType.UNKNOWN,
                    value="", original_url=url,
                    error="URL invalide : doit commencer par http:// ou https://",
                    suggestions=["Ajoutez https:// au début de l'URL"],
                )

        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return ValidationResult(
                    is_valid=False, platform=Platform.UNKNOWN, url_type=URLType.UNKNOWN,
                    value="", original_url=url, error="Format d'URL invalide",
                )
        except Exception as e:
            return ValidationResult(
                is_valid=False, platform=Platform.UNKNOWN, url_type=URLType.UNKNOWN,
                value="", original_url=url, error=f"Erreur de parsing : {e}",
            )

        platform = cls.detect_platform(url)

        if platform in _REMOVED_PLATFORMS:
            # Retrait produit volontaire (dump/leak sites) : refus explicite et nommé
            # AVANT toute résolution de source, pour ne jamais retomber — silencieusement
            # ou non — sur le scraper générique (gallery-dl/yt-dlp supportent nativement
            # certains de ces sites en interne).
            return ValidationResult(
                is_valid=False, platform=platform, url_type=URLType.UNKNOWN,
                value="", original_url=url,
                error=f"Source « {platform.value} » non prise en charge : ce site a été "
                      "retiré de ce scraper.",
            )

        if platform == Platform.UNKNOWN:
            # Domaine inconnu mais URL http(s) bien formée → délégué à yt-dlp.
            if '.' in parsed.netloc:
                return ValidationResult(
                    is_valid=True, platform=Platform.GENERIC, url_type=URLType.VIDEO,
                    value=url, original_url=url,
                )
            return ValidationResult(
                is_valid=False, platform=Platform.UNKNOWN, url_type=URLType.UNKNOWN,
                value="", original_url=url, error="URL non reconnue",
                suggestions=[
                    "Plateformes spécialisées : RedGIFs, Instagram, Picazor",
                    "Autres sites vidéo (TikTok, YouTube, ...) : collez l'URL complète de la page",
                ],
            )

        domain = (parsed.hostname or '').lower()
        if platform in cls.VALID_DOMAINS and domain not in cls.VALID_DOMAINS[platform]:
            valid_domains = ', '.join(cls.VALID_DOMAINS[platform])
            return ValidationResult(
                is_valid=False, platform=platform, url_type=URLType.UNKNOWN,
                value="", original_url=url,
                error=f"Domaine invalide pour {platform.value}",
                suggestions=[f"Utilisez : {valid_domains}"],
            )

        if platform == Platform.REDGIFS:
            return cls._validate_redgifs(url)
        if platform == Platform.INSTAGRAM:
            return cls._validate_instagram(url)
        if platform == Platform.PICAZOR:
            return cls._validate_picazor(url)
        if platform == Platform.EROME:
            return cls._validate_erome(url)
        if platform == Platform.PEXELS:
            return cls._validate_pexels(url)
        if platform in (Platform.X, Platform.TIKTOK, Platform.PORNPICS,
                        Platform.CIVITAI, Platform.FAPELLO, Platform.REDDIT,
                        Platform.SEXCOM):
            return cls._validate_gallerydl_platform(url, platform)

        # Inatteignable : tout Platform détecté ci-dessus a son _validate_X.
        # Garde-fou défensif (ne devrait jamais s'exécuter).
        return ValidationResult(
            is_valid=False, platform=platform, url_type=URLType.UNKNOWN,
            value="", original_url=url, error="Plateforme non prise en charge.")


    @classmethod
    def _validate_redgifs(cls, url: str) -> ValidationResult:
        for pattern in cls.PATTERNS[Platform.REDGIFS]["user"]:
            m = re.search(pattern, url, re.IGNORECASE)
            if m:
                return ValidationResult(True, Platform.REDGIFS, URLType.PROFILE, m.group(1), url)

        for pattern in cls.PATTERNS[Platform.REDGIFS]["video"]:
            m = re.search(pattern, url, re.IGNORECASE)
            if m:
                video_id = m.group(1)
                if video_id.lower() not in ('users', 'niches', 'watch', 'gifs'):
                    return ValidationResult(True, Platform.REDGIFS, URLType.VIDEO, video_id, url)

        for pattern in cls.PATTERNS[Platform.REDGIFS]["niche"]:
            m = re.search(pattern, url, re.IGNORECASE)
            if m:
                return ValidationResult(True, Platform.REDGIFS, URLType.NICHE, m.group(1), url)

        return ValidationResult(
            False, Platform.REDGIFS, URLType.UNKNOWN, "", url,
            error="Format d'URL RedGIFs non reconnu",
            suggestions=[
                "Formats supportés :",
                "  • Profil : redgifs.com/users/username",
                "  • Vidéo : redgifs.com/watch/videoid",
                "  • Niche : redgifs.com/niches/nichename",
            ],
        )

    @classmethod
    def _validate_instagram(cls, url: str) -> ValidationResult:
        for pattern in cls.PATTERNS[Platform.INSTAGRAM]["post"]:
            m = re.search(pattern, url, re.IGNORECASE)
            if m:
                post_id = m.group(m.lastindex) if m.lastindex else m.group(1)
                return ValidationResult(True, Platform.INSTAGRAM, URLType.POST, post_id, url)

        for pattern in cls.PATTERNS[Platform.INSTAGRAM]["reel"]:
            m = re.search(pattern, url, re.IGNORECASE)
            if m:
                reel_id = m.group(m.lastindex) if m.lastindex else m.group(1)
                return ValidationResult(True, Platform.INSTAGRAM, URLType.REEL, reel_id, url)

        for pattern in cls.PATTERNS[Platform.INSTAGRAM]["profile"]:
            m = re.search(pattern, url, re.IGNORECASE)
            if m:
                username = m.group(1)
                if username not in ('p', 'reel', 'reels', 'stories', 'explore', 'tv'):
                    return ValidationResult(True, Platform.INSTAGRAM, URLType.PROFILE, username, url)

        return ValidationResult(
            False, Platform.INSTAGRAM, URLType.UNKNOWN, "", url,
            error="Format d'URL Instagram non reconnu",
            suggestions=[
                "Formats supportés :",
                "  • Profil : instagram.com/username",
                "  • Post : instagram.com/p/postid",
                "  • Reel : instagram.com/reel/reelid",
            ],
        )

    @classmethod
    def _validate_picazor(cls, url: str) -> ValidationResult:
        for pattern in cls.PATTERNS[Platform.PICAZOR]["listing"]:
            m = re.search(pattern, url, re.IGNORECASE)
            if m:
                return ValidationResult(True, Platform.PICAZOR, URLType.LISTING, m.group(2), url)

        for pattern in cls.PATTERNS[Platform.PICAZOR]["profile"]:
            m = re.search(pattern, url, re.IGNORECASE)
            if m:
                creator = m.group(2)
                if creator not in ('videos', 'models', 'categories'):
                    return ValidationResult(True, Platform.PICAZOR, URLType.PROFILE, creator, url)

        for pattern in cls.PATTERNS[Platform.PICAZOR]["video"]:
            m = re.search(pattern, url, re.IGNORECASE)
            if m:
                creator = m.group(2)
                if creator not in ('videos', 'models', 'categories'):
                    return ValidationResult(True, Platform.PICAZOR, URLType.VIDEO, creator, url)

        return ValidationResult(
            False, Platform.PICAZOR, URLType.UNKNOWN, "", url,
            error="Format d'URL Picazor non reconnu",
            suggestions=[
                "Formats supportés :",
                "  • Profil : picazor.com/fr/creator",
                "  • Page de listing : picazor.com/fr/creator/page/2",
                "  • Média unique : picazor.com/fr/creator/123",
                "  • Listing : picazor.com/fr/videos/week",
            ],
        )

    @classmethod
    def _validate_erome(cls, url: str) -> ValidationResult:
        # Album (/a/ID) ou recherche (/search?q=) = conteneurs de médias.
        for pattern in cls.PATTERNS[Platform.EROME]["listing"]:
            m = re.search(pattern, url, re.IGNORECASE)
            if m:
                value = m.group(1) if m.groups() else "search"
                return ValidationResult(True, Platform.EROME, URLType.LISTING, value, url)

        # Profil utilisateur (/USER) — exclure les chemins réservés.
        for pattern in cls.PATTERNS[Platform.EROME]["profile"]:
            m = re.search(pattern, url, re.IGNORECASE)
            if m:
                user = m.group(1)
                if user.lower() not in ('a', 'search', 'login', 'register',
                                        'tos', 'dmca', 'faq'):
                    return ValidationResult(True, Platform.EROME, URLType.PROFILE, user, url)

        return ValidationResult(
            False, Platform.EROME, URLType.UNKNOWN, "", url,
            error="Format d'URL Erome non reconnu",
            suggestions=[
                "Formats supportés :",
                "  • Album : erome.com/a/AbCdEfGh",
                "  • Recherche : erome.com/search?q=motcle",
                "  • Profil : erome.com/username",
            ],
        )

    @classmethod
    def _validate_pexels(cls, url: str) -> ValidationResult:
        """Accepte uniquement les chemins couverts par l'API officielle.

        Recherche, collection et photo ont un endpoint documenté. Les profils
        publics /@user, pages racine, vidéos, locales et routes inconnues sont
        rejetés avant tout appel réseau avec des suggestions actionnables.
        """
        parsed = urlparse(url)
        suggestions = [
            "Formats Pexels pris en charge :",
            "  • Recherche : pexels.com/search/portrait/",
            "  • Recherche localisée : pexels.com/en-us/search/portrait/",
            "    ou pexels.com/fr-fr/chercher/portrait/",
            "  • Collection accessible à votre clé : pexels.com/collections/nom-identifiant/",
            "  • Photo : pexels.com/photo/nom-123456/",
            "Les préfixes /en-us/ et /fr-fr/ sont aussi acceptés pour les photos et collections.",
            "L'API officielle Pexels n'expose pas les profils publics /@user.",
        ]
        if parsed.username is not None or parsed.password is not None:
            return ValidationResult(
                False, Platform.PEXELS, URLType.UNKNOWN, "", url,
                error="URL Pexels invalide : informations utilisateur non autorisées.",
                suggestions=suggestions,
            )

        route = parse_pexels_path(parsed.path or "/")
        if route:
            url_type = URLType.POST if route.kind == 'photo' else URLType.LISTING
            return ValidationResult(
                True, Platform.PEXELS, url_type, route.value, url)

        return ValidationResult(
            False, Platform.PEXELS, URLType.UNKNOWN, "", url,
            error="Format d'URL Pexels non pris en charge par l'API officielle.",
            suggestions=suggestions,
        )

    @classmethod
    def _validate_gallerydl_platform(cls, url, platform):
        """Validation minimale pour les plateformes gérées par gallery-dl : on confirme
        juste l'hôte (déjà fait) et on laisse gallery-dl parser le chemin. url_type
        coarse (LISTING) suffit — le scan énumère via gallery-dl."""
        return ValidationResult(
            is_valid=True, platform=platform, url_type=URLType.LISTING,
            value=url, original_url=url)


url_validator = URLValidator()
