# app/scrape/netfetch.py
"""Briques réseau du scraping : validation anti-SSRF + lancement yt-dlp.

Sans dépendance vers `routes`/`download_service` → importable par les deux
(pas de cycle). Voir routes.py pour le contexte sécurité (admin-only, SSRF).
"""
import sys
import socket
import ipaddress
import subprocess
import tempfile
import shutil
from urllib.parse import urlparse

from flask import current_app

# Only the yt-dlp *video* download path uses this; the concept bridge fetches
# images via fetch_hardened_bytes and never touches it. Our config has no such
# constant, so degrade gracefully instead of breaking the whole import.
try:
    from ..config import COMFYUI_OUTPUT_DIR
except ImportError:  # pragma: no cover - our app resolves the output dir differently
    COMFYUI_OUTPUT_DIR = None

# Plafond de taille du téléchargement (vidéo driver — pas besoin de plus).
MAX_DRIVER_BYTES = 200 * 1024 * 1024  # 200 Mo
# Timeout mur du sous-processus yt-dlp.
DOWNLOAD_TIMEOUT = 180  # secondes
# Timeout socket interne yt-dlp (par requête réseau).
SOCKET_TIMEOUT = 30  # secondes

# Plancher de version yt-dlp. Relevé 2026-08-18 : l'ancien plancher (2024.07.01,
# CVE-2024-38519) avait DEUX ans et DEUX CVE de retard — CVE-2026-50019 (fuite de
# cookies, corrigée en 2026.06.09) et CVE-2026-55404 (injection de commande via
# --write-link, corrigée en 2026.07.04). WARNING non bloquant — un assert
# dur sur une version briquerait Flask au démarrage.
YTDLP_VERSION_FLOOR = (2026, 7, 4)
_version_checked = False

_ffmpeg_checked = None


def _ffmpeg_available():
    """True si ffmpeg est sur le PATH (mis en cache). Nécessaire pour muxer
    bv*+ba ; sans lui yt-dlp ne peut pas fusionner les flux séparés."""
    global _ffmpeg_checked
    if _ffmpeg_checked is None:
        _ffmpeg_checked = shutil.which('ffmpeg') is not None
    return _ffmpeg_checked


def _ytdlp_version_tuple():
    """(YYYY, M, D) de yt-dlp installé, ou None si introuvable. Ne lève jamais."""
    try:
        import yt_dlp
        parts = str(yt_dlp.version.__version__).split('.')[:3]
        return tuple(int(p) for p in parts)
    except Exception:
        return None


def _check_ytdlp_version():
    """Log un WARNING (une seule fois) si yt-dlp < plancher. Retourne ok:bool.
    Jamais fatal — ne bloque pas le téléchargement."""
    global _version_checked
    ver = _ytdlp_version_tuple()
    if ver is not None and ver >= YTDLP_VERSION_FLOOR:
        return True
    if not _version_checked:
        _version_checked = True
        try:
            current_app.logger.warning(
                "yt-dlp %s < plancher %s recommandé (CVE-2026-50019/55404). "
                "Mettre à jour : python -m pip install -U yt-dlp",
                ver, YTDLP_VERSION_FLOOR,
            )
        except Exception:
            pass
    return False


def _ip_is_blocked(ip):
    """True si `ip` (ipaddress) cible un espace réseau non-public.

    Déballe les IPv6 IPv4-mapped (`::ffff:10.0.0.1`) et 6to4 avant de tester
    l'espace réseau → bloque le contournement par encodage IPv6 d'une IPv4 privée.
    """
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip.sixtofour is not None:
            ip = ip.sixtofour
    return (ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_reserved or ip.is_multicast or ip.is_unspecified)


def _resolve_public_ips(host, port):
    """Résout `host` et valide CHAQUE IP. Retourne (frozenset[str], error).

    Sur la moindre IP non-publique → (None, message). N'est appelé QU'une fois,
    à la validation (`_validate_public_http_url`) — pas de second appel juste
    avant le spawn de gallery-dl/yt-dlp. Voir la docstring de
    `_validate_public_http_url` pour la conséquence (fenêtre DNS-rebinding).
    """
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return None, "Host not found (DNS)."
    except Exception:
        return None, "Could not resolve host."

    ips = set()
    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return None, "Invalid IP address."
        if _ip_is_blocked(ip):
            return None, "This URL points to an internal network address (blocked)."
        ips.add(str(ip))
    if not ips:
        return None, "Host not found (DNS)."
    return frozenset(ips), None


def _url_host_port(url):
    parsed = urlparse(url)
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    return parsed.hostname, port


def _validate_public_http_url(url):
    """Valide qu'`url` est une URL http(s) publique (anti-SSRF).

    Rejette tout schéma autre que http/https, et tout host qui résout vers une
    IP non-publique (loopback / privée / link-local / réservée / multicast /
    IPv4-mapped IPv6). Retourne (ok: bool, error: str|None).

    NB : cette fonction résout et classe le host UNE SEULE fois, ici. Les IP
    obtenues ne sont pas conservées ni réutilisées : gallery-dl / yt-dlp
    refont leur propre résolution DNS au moment de se connecter, en dehors de
    ce module. Un host dont la réponse DNS change entre les deux (DNS
    rebinding) — ou une redirection HTTP suivie par l'outil vers une IP
    interne — n'est donc PAS intercepté par cette garde : il n'y a pas de
    re-résolution juste avant le spawn. Fermer complètement cette fenêtre
    demande un mécanisme hors de portée de ce module (un proxy de sortie qui
    valide chaque connexion sortante, ou un pare-feu OS bloquant les plages
    privées pour ce process) — ce n'est pas quelque chose que `netfetch.py`
    peut garantir seul.
    """
    if not url or not isinstance(url, str):
        return False, "Missing URL."
    url = url.strip()
    if len(url) > 2048:
        return False, "URL too long."
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Invalid URL."
    if parsed.scheme not in ('http', 'https'):
        return False, "Only http(s) URLs are allowed."
    host = parsed.hostname
    if not host:
        return False, "URL without a valid host."

    _ips, err = _resolve_public_ips(host, parsed.port or (443 if parsed.scheme == 'https' else 80))
    if err:
        return False, err
    return True, None


def _download_with_ytdlp(url, dest_template):
    """Lance `python -m yt_dlp` en sous-processus. Retourne (ok, error)."""
    _check_ytdlp_version()   # WARNING non bloquant si version trop ancienne
    if _ffmpeg_available():
        fmt_args = ['-f', 'bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b',
                    '--merge-output-format', 'mp4']
    else:
        fmt_args = ['-f', 'best[ext=mp4]/mp4/best']   # legacy single-stream (pas de mux)
    cmd = [
        sys.executable, '-m', 'yt_dlp',
        '--ignore-config',          # ne jamais charger yt-dlp.conf (anti exec/--netrc-cmd plantés)
        '--no-playlist',
        '--no-warnings',
        '--quiet',
        '--no-part',
        '--no-continue',
        '--max-filesize', str(MAX_DRIVER_BYTES),
        '--socket-timeout', str(SOCKET_TIMEOUT),
        *fmt_args,
        '-o', dest_template,
        '--', url,
    ]
    quarantine = None
    try:
        quarantine = tempfile.mkdtemp(prefix='scrape_ytdlp_')
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=DOWNLOAD_TIMEOUT,
            cwd=quarantine,   # cwd isolé : un fichier planté (ffmpeg.exe…) n'atterrit pas dans COMFYUI_OUTPUT_DIR
        )
    except subprocess.TimeoutExpired:
        return False, "Download timed out."
    except FileNotFoundError:
        current_app.logger.error("yt-dlp introuvable (python -m yt_dlp).")
        return False, "yt-dlp not available on the server."
    except Exception as e:
        current_app.logger.error(f"Erreur lancement yt-dlp: {e}")
        return False, "Internal download error."
    finally:
        if quarantine:
            shutil.rmtree(quarantine, ignore_errors=True)

    if result.returncode != 0:
        # Ne pas renvoyer stderr brut (peut fuiter des chemins) — log seulement.
        current_app.logger.warning(
            f"yt-dlp échec (rc={result.returncode}) pour {url[:120]}: "
            f"{(result.stderr or '')[:500]}"
        )
        return False, "Download failed (unsupported or unavailable URL)."
    return True, None


def _looks_like_image(path):
    """True si `path` commence par une signature raster connue (jpg/png/bmp/gif/webp/avif).
    PAS de SVG (peut embarquer du script). MIME/extension seuls sont falsifiables."""
    try:
        with open(path, 'rb') as f:
            head = f.read(32)
    except OSError:
        return False
    if len(head) < 12:
        return False
    if head[:3] == b'\xff\xd8\xff':                       # jpeg
        return True
    if head[:8] == b'\x89PNG\r\n\x1a\n':                  # png
        return True
    if head[:2] == b'BM':                                  # bmp
        return True
    if head[:4] == b'GIF8':                               # gif
        return True
    if head[:4] == b'RIFF' and head[8:12] == b'WEBP':     # webp
        return True
    if head[4:8] == b'ftyp' and head[8:12] in (b'avif', b'avis'):  # avif
        return True
    return False


def _looks_like_video(path):
    """True si `path` commence par une signature vidéo connue
    (mp4/mov/m4v, mkv/webm, avi, gif, mpeg PS/ES/TS).

    Miroir local de la validation vidéo de l'app source : elle vivait dans un
    module `upload` qui n'existe pas ici, et l'import fantôme faisait lever
    ImportError à CHAQUE appel réel (les tests mockaient en amont, ils ne l'ont
    jamais vu). Extension et content-type seuls sont falsifiables — on lit les
    octets, jamais plus que l'en-tête.

    ATTENTION : `ftyp` couvre toute la famille ISO-BMFF, AVIF/HEIC compris, et
    `GIF8` est aussi une signature raster — cette fonction répond donc 'vidéo'
    sur une image AVIF ou un GIF. C'est assumé (maintenir la liste exhaustive
    des marques ISO-BMFF images est perdu d'avance) : tout appelant qui accepte
    AUSSI les images doit tester l'image D'ABORD, cf. `_validate_media_file`.
    """
    try:
        with open(path, 'rb') as f:
            head = f.read(512)   # 512 o : couvre les 3 octets de synchro TS (0, 188, 376)
    except OSError:
        return False
    if len(head) < 12:
        return False
    if head[4:8] == b'ftyp':                              # mp4 / mov / m4v / 3gp (ISO-BMFF)
        return True
    if head[:4] == b'\x1a\x45\xdf\xa3':                   # EBML : mkv / webm
        return True
    if head[:4] == b'RIFF' and head[8:12] == b'AVI ':     # avi
        return True
    if head[:4] == b'GIF8':                               # gif
        return True
    if head[:4] in (b'\x00\x00\x01\xba',                  # mpeg program stream (pack header)
                    b'\x00\x00\x01\xb3'):                 # mpeg-1/2 video (sequence header)
        return True
    # mpeg-ts : pas de magic en tête, mais une synchro 0x47 tous les 188 octets.
    # Un seul 0x47 ne prouve rien (tout fichier commençant par 'G') — on en exige trois.
    if len(head) > 376 and head[0] == 0x47 and head[188] == 0x47 and head[376] == 0x47:
        return True
    return False


import os as _os
import glob as _glob


def download_via_ytdlp(url, dest_base):
    """Télécharge via yt-dlp dans le dossier de dest_base, garde le 1er fichier vidéo
    valide. Retourne (ok, filename|None, error|None). Ne lève jamais."""
    dest_dir = _os.path.dirname(dest_base)
    uid = _os.path.basename(dest_base)
    _os.makedirs(dest_dir, exist_ok=True)
    dest_template = _os.path.join(dest_dir, f'{uid}.%(ext)s')
    ok, err = _download_with_ytdlp(url, dest_template)
    if not ok:
        for stray in _glob.glob(_os.path.join(dest_dir, f'{uid}.*')):
            try: _os.remove(stray)
            except OSError: pass
        return False, None, err
    produced = sorted(_glob.glob(_os.path.join(dest_dir, f'{uid}.*')))
    final = None
    for p in produced:
        if final is None and _looks_like_video(p):
            final = p
        else:
            try: _os.remove(p)
            except OSError: pass
    if final is None:
        return False, None, "The downloaded file is not a valid video."
    return True, _os.path.basename(final), None


def fetch_hardened_bytes(url, *, allowed_types, max_bytes, require_image_magic=False):
    """Fetch durci d'une URL média en mémoire (même patron de sécurité que /thumb).

    Retourne (ok, data|None, ctype|None, reason). `reason` est un code court
    ('redirect','status','type','toolarge','fetch','noimage','no_curl') exploitable
    par l'appelant pour compter/expliquer les skips.

    Garanties (zéro régression vs /thumb) :
      - l'URL est supposée DÉJÀ validée anti-SSRF par l'appelant (_validate_public_http_url) ;
      - curl_cffi impersonate='chrome' + Referer du host source ;
      - allow_redirects=False : toute 3xx → refus (sinon une redirection vers une IP
        interne contournerait la garde SSRF amont — TOCTOU/redirect bypass) ;
      - content-type restreint à `allowed_types` (jamais image/svg+xml côté appelant) ;
      - lecture CAPPÉE pendant le stream (jamais le body entier avant test) ;
      - `require_image_magic` : en plus du content-type, le contenu doit commencer par
        une signature raster connue (anti type-spoof : un non-admin ne reçoit qu'une
        vraie image, jamais html/svg/exe déguisé)."""
    try:
        from curl_cffi import requests as cf_requests
    except ImportError:
        return False, None, None, 'no_curl'
    host = urlparse(url).hostname or ''
    try:
        r = cf_requests.get(url, impersonate='chrome', timeout=20, stream=True,
                            allow_redirects=False,
                            headers={'Referer': f'https://{host}/', 'Accept': '*/*'})
    except Exception as e:
        try:
            current_app.logger.warning(f"fetch_hardened_bytes échec {url[:120]}: {e}")
        except Exception:
            pass
        return False, None, None, 'fetch'
    if 300 <= r.status_code < 400:
        try: r.close()
        except Exception: pass
        return False, None, None, 'redirect'
    ctype = (r.headers.get('content-type') or '').split(';')[0].strip().lower()
    if r.status_code != 200 or ctype not in allowed_types:
        try: r.close()
        except Exception: pass
        return False, None, None, ('status' if r.status_code != 200 else 'type')
    data = bytearray()
    try:
        for chunk in r.iter_content(8192):
            if not chunk:
                continue
            data += chunk
            if len(data) > max_bytes:
                try: r.close()
                except Exception: pass
                return False, None, None, 'toolarge'
    finally:
        try: r.close()
        except Exception: pass
    # Validation magic-bytes du raster : un content-type image/* peut mentir.
    if require_image_magic and not _bytes_look_like_image(bytes(data[:32])):
        return False, None, None, 'noimage'
    return True, bytes(data), ctype, 'ok'


# Signatures raster acceptées par octets en mémoire (miroir de _looks_like_image,
# qui lit depuis un fichier). PAS de SVG (peut embarquer du script).
def _bytes_look_like_image(head):
    if len(head) < 12:
        return False
    if head[:3] == b'\xff\xd8\xff':                       # jpeg
        return True
    if head[:8] == b'\x89PNG\r\n\x1a\n':                  # png
        return True
    if head[:2] == b'BM':                                  # bmp
        return True
    if head[:4] == b'GIF8':                               # gif
        return True
    if head[:4] == b'RIFF' and head[8:12] == b'WEBP':     # webp
        return True
    if head[4:8] == b'ftyp' and head[8:12] in (b'avif', b'avis'):  # avif
        return True
    return False


def _validate_media_file(path, *, allow_image=True):
    """Valide qu'`path` est un vrai média par signature (magic bytes).

    Retourne (ok, kind) où kind ∈ {'video','image'} en cas de succès, sinon
    (False, None). Rejette HTML/SVG/zip/exe/raccourcis quelle que soit
    l'extension de l'URL. `allow_image=False` => seules les vidéos passent
    (chemin driver SCAIL, vidéo-only)."""
    # On teste l'image AVANT la vidéo : _looks_like_video matche tout 'ftyp'
    # (y compris AVIF) et aussi 'GIF8' → sans cet ordre une image AVIF, ou un
    # GIF, serait classé 'video'. Pinné par test_netfetch_video_magic.py.
    if _looks_like_image(path):
        return (True, 'image') if allow_image else (False, None)
    # Validation vidéo durcie, locale (mp4/mov/webm/mkv/avi/gif/mpeg).
    if _looks_like_video(path):
        return True, 'video'
    return False, None
