# app/scrape/sources/universal.py
"""Source universelle (catch-all, priorité 0) : hybride gallery-dl → (exit 64) → yt-dlp.
Tente d'abord gallery-dl (extracteurs dédiés de nombreux sites) ; si gallery-dl ne
supporte pas l'URL (exit code & 64), repli sur yt-dlp — mais SEULEMENT pour les hôtes
d'une allowlist vettée (atténuation SSRF interim, cf. spec décision #6)."""
import os
from urllib.parse import urlparse

from .base import Source, Capabilities, Match
from . import registry, gdl
from .. import netfetch

# Hôtes pour lesquels la branche générique yt-dlp est autorisée (interim SSRF).
VETTED_DOMAINS = (
    'coomer.st', 'coomer.su', 'coomer.party', 'coomer.cr',
    'kemono.cr', 'kemono.su', 'kemono.party',
    'cyberdrop.me', 'cyberdrop.cr', 'cyberdrop.to',
    'x.com', 'twitter.com', 'tiktok.com',
    'youtube.com', 'youtu.be', 'pornhub.com', 'xvideos.com', 'redgifs.com',
    'vimeo.com', 'dailymotion.com',
)


def _host_vetted(url):
    host = (urlparse(url).hostname or '').lower()
    if not host:
        return False
    labels = host.split('.')
    # Bunkr (TLDs rotatifs) : matcher UNIQUEMENT le label SLD (avant-dernier), sinon
    # un host trompeur type bunkr.cr.evil.com passerait l'allowlist (bypass SSRF).
    if len(labels) >= 2 and labels[-2].startswith('bunkr'):
        return True
    return any(host == d or host.endswith('.' + d) for d in VETTED_DOMAINS)


def _gdl_unsupported(err):
    """True ssi gallery-dl a renvoyé le kind 'unsupported' (extracteur absent) — on
    matche la POSITION du kind, pas le tail stderr (évite un faux positif sur une
    erreur réseau 'unsupported protocol')."""
    return bool(err) and err.lower().startswith('gallery-dl : unsupported')


class UniversalSource(Source):
    name = 'universal'
    priority = 0
    capabilities = Capabilities(is_universal_fallback=True, own_downloader=True)

    def match(self, url):
        from ..validators import url_validator, Platform
        result = url_validator.validate_url(url)
        if result.is_valid and result.platform == Platform.GENERIC:
            return Match(url=url, validation=result)
        return None

    def scan(self, match):
        # Énumération générique : 1 item (yt-dlp gère la vidéo unique) ; gallery-dl
        # générique étant off par défaut, on reste sur l'item unique au scan.
        url = match.url
        return ([{'url': url, 'title': url, 'thumbnail': None,
                  'type': 'video', 'platform': 'generic'}], None)

    def download(self, url, dest_base):
        # 1) gallery-dl (extracteur dédié) d'abord.
        dest_dir = os.path.dirname(dest_base)
        filename = os.path.basename(dest_base)
        ok, abs_path, err = gdl.download(url, dest_dir, filename)
        if ok and abs_path:
            return True, os.path.basename(abs_path), None
        # 2) gallery-dl ne supporte pas → yt-dlp, mais seulement si l'hôte est vetté.
        if _gdl_unsupported(err):
            if not _host_vetted(url):
                return False, None, "Site non supporté (gallery-dl) et hôte non vetté pour yt-dlp."
            return netfetch.download_via_ytdlp(url, dest_base)
        # 3) auth/réseau (pas 'unsupported') → on remonte l'erreur gallery-dl.
        return False, None, err or "Échec du téléchargement générique."


registry.register(UniversalSource())
