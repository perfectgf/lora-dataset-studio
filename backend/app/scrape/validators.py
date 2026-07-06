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
from urllib.parse import urlparse


class Platform(Enum):
    REDGIFS = "redgifs"
    INSTAGRAM = "instagram"
    PICAZOR = "picazor"
    EROME = "erome"
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
    GENERIC = "generic"   # toute autre URL http(s) — déléguée à yt-dlp
    UNKNOWN = "unknown"


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
        Platform.COOMER: ['coomer.st', 'coomer.su', 'coomer.party', 'coomer.cr'],
        Platform.KEMONO: ['kemono.cr', 'kemono.su', 'kemono.party'],
        Platform.CYBERDROP: ['cyberdrop.me', 'cyberdrop.cr', 'cyberdrop.to'],
        Platform.X: ['x.com', 'twitter.com', 'www.x.com', 'www.twitter.com', 'mobile.twitter.com'],
        Platform.TIKTOK: ['tiktok.com', 'www.tiktok.com', 'vm.tiktok.com'],
        Platform.PORNPICS: ['pornpics.com', 'www.pornpics.com'],
        Platform.CIVITAI: ['civitai.com', 'www.civitai.com', 'civitai.red', 'www.civitai.red'],
        # Bunkr tourne sur des TLDs rotatifs → matché par host-contains 'bunkr' (cf. detect_platform).
    }

    # Domaines reconnus par host (host == d ou host.endswith('.'+d)).
    _HOST_PLATFORMS = [
        ('redgifs.com', Platform.REDGIFS),
        ('instagram.com', Platform.INSTAGRAM),
        ('picazor.com', Platform.PICAZOR),
        ('erome.com', Platform.EROME),
        ('coomer.st', Platform.COOMER), ('coomer.su', Platform.COOMER),
        ('coomer.party', Platform.COOMER), ('coomer.cr', Platform.COOMER),
        ('kemono.cr', Platform.KEMONO), ('kemono.su', Platform.KEMONO),
        ('kemono.party', Platform.KEMONO),
        ('cyberdrop.me', Platform.CYBERDROP), ('cyberdrop.cr', Platform.CYBERDROP),
        ('cyberdrop.to', Platform.CYBERDROP),
        ('x.com', Platform.X), ('twitter.com', Platform.X),
        ('tiktok.com', Platform.TIKTOK),
        ('pornpics.com', Platform.PORNPICS),
        ('civitai.com', Platform.CIVITAI),
        ('civitai.red', Platform.CIVITAI),
        # fapello.com + tout miroir de langue (fr./de./es.…) via endswith('.fapello.com').
        # Volontairement ABSENT de VALID_DOMAINS → la garde stricte de domaine n'écarte
        # pas les sous-domaines de langue (la source normalise l'hôte avant gallery-dl).
        ('fapello.com', Platform.FAPELLO),
    ]

    @staticmethod
    def detect_platform(url: str) -> Platform:
        host = (urlparse(url).hostname or '').lower()
        if not host:
            return Platform.UNKNOWN
        # Bunkr : TLDs rotatifs → le label 'bunkr' doit être le SLD (avant-dernier label),
        # ex : bunkr.cr, bunkrr.su — on vérifie labels[-2] pour éviter les sous-domaines
        # trompeurs comme bunkr.cr.evil.com où 'bunkr' est un sous-domaine, pas le SLD.
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
        if platform in (Platform.COOMER, Platform.KEMONO, Platform.BUNKR,
                        Platform.CYBERDROP, Platform.X, Platform.TIKTOK,
                        Platform.PORNPICS, Platform.CIVITAI, Platform.FAPELLO):
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
    def _validate_gallerydl_platform(cls, url, platform):
        """Validation minimale pour les plateformes gérées par gallery-dl : on confirme
        juste l'hôte (déjà fait) et on laisse gallery-dl parser le chemin. url_type
        coarse (LISTING) suffit — le scan énumère via gallery-dl."""
        return ValidationResult(
            is_valid=True, platform=platform, url_type=URLType.LISTING,
            value=url, original_url=url)


url_validator = URLValidator()
