"""Signatures vidéo de netfetch + validation RÉELLE du chemin yt-dlp.

Ce module existe à cause d'un port dormant : `netfetch.py` importait
`_looks_like_video` depuis `app/upload/routes.py`, un module qui n'a jamais
existé dans cette app. Résultat : ImportError à chaque exécution réelle de
`download_via_ytdlp` et de `_validate_media_file` — invisible pour la suite,
parce que `test_universal_source.py` monkeypatche `download_via_ytdlp` EN
ENTIER et ne descend donc jamais dedans.

D'où la règle de ce fichier : on ne mocke QUE le sous-processus yt-dlp
(`_download_with_ytdlp`), jamais la validation. Les octets sont forgés à la
main — aucun vrai média, aucun réseau, aucun contexte Flask.
"""
import pytest

from app.scrape import netfetch


# --- En-têtes forgés (le strict nécessaire : 12 octets minimum sont lus) ---

MP4 = b'\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2'
MOV = b'\x00\x00\x00\x14ftypqt  \x00\x00\x02\x00qt  '
MKV = b'\x1a\x45\xdf\xa3\x93\x42\x82\x88matroska\x42\x87'
WEBM = b'\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81webm'
AVI = b'RIFF\x24\x08\x00\x00AVI LIST\x00\x00'
GIF = b'GIF89a\x10\x00\x10\x00\x80\x00\x00\xff\xff\xff'
MPEG_PS = b'\x00\x00\x01\xba\x44\x00\x04\x00\x04\x01\x00\x03'   # pack header
MPEG_ES = b'\x00\x00\x01\xb3\x14\x00\xf0\xc4\x02\xcb\x23\x80'   # sequence header
# mpeg-ts : aucun magic en tête, juste 0x47 tous les 188 octets.
MPEG_TS = bytearray(b'\x11' * 512)
MPEG_TS[0] = MPEG_TS[188] = MPEG_TS[376] = 0x47
MPEG_TS = bytes(MPEG_TS)

AVIF = b'\x00\x00\x00\x20ftypavif\x00\x00\x00\x00avifmif1'
JPEG = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x02\x00'
HTML = b'<!DOCTYPE html><html><head><title>Sign in</title></head></html>'


def _write(tmp_path, name, data):
    """Écrit `data` tel quel et retourne le chemin (str) — l'extension MENT
    volontairement dans plusieurs tests : c'est tout l'objet de la sonde."""
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


# --- 1. Sonde vidéo : ce qu'elle reconnaît ---

@pytest.mark.parametrize('name, data', [
    ('clip.mp4', MP4),
    ('clip.mov', MOV),
    ('clip.mkv', MKV),
    ('clip.webm', WEBM),
    ('clip.avi', AVI),
    ('clip.gif', GIF),
    ('clip.mpg', MPEG_PS),
    ('clip.mpeg', MPEG_ES),
    ('clip.ts', MPEG_TS),
])
def test_known_video_headers_are_recognised(tmp_path, name, data):
    assert netfetch._looks_like_video(_write(tmp_path, name, data)) is True


# --- 2. Sonde vidéo : ce qu'elle refuse ---

@pytest.mark.parametrize('name, data', [
    ('login.mp4', HTML),                       # page d'auth servie à la place du média
    ('clip.mp4', JPEG),                        # vignette renommée
    ('clip.mp4', b'PK\x03\x04' + b'\x00' * 32),   # zip
    ('clip.mp4', b'MZ\x90\x00' + b'\x00' * 32),   # exe Windows
    ('clip.mp4', b'GET / HTTP/1.1\r\n\r\n'),   # texte quelconque
    ('clip.mp4', b'\x47' + b'\x00' * 500),     # un SEUL 0x47 : pas un flux TS
    ('clip.mp4', b'short'),                    # trop court pour décider
    ('clip.mp4', b''),                         # fichier vide (yt-dlp tué en vol)
])
def test_non_video_content_is_refused_whatever_the_extension(tmp_path, name, data):
    assert netfetch._looks_like_video(_write(tmp_path, name, data)) is False


def test_a_missing_file_is_refused_instead_of_raising(tmp_path):
    """Le nettoyage d'un échec yt-dlp peut avoir déjà retiré le fichier."""
    assert netfetch._looks_like_video(str(tmp_path / 'gone.mp4')) is False


def test_only_a_bounded_prefix_is_read(tmp_path, monkeypatch):
    """Une vidéo pèse jusqu'à MAX_DRIVER_BYTES (200 Mo) : la sonde lit un
    en-tête, jamais le fichier entier."""
    path = _write(tmp_path, 'big.mp4', MP4 + b'\x00' * 200000)
    reads = []
    real_open = open

    class _SpyFile:
        def __init__(self, fh):
            self._fh = fh

        def read(self, size=-1):
            reads.append(size)
            return self._fh.read(size)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._fh.close()
            return False

    monkeypatch.setattr(netfetch, 'open',
                        lambda *a, **k: _SpyFile(real_open(*a, **k)), raising=False)

    assert netfetch._looks_like_video(path) is True
    assert reads, "la sonde n'a rien lu"
    assert all(0 < size <= 4096 for size in reads), reads


# --- 3. _validate_media_file : l'ordre image-avant-vidéo, et ses deux branches ---

def test_an_avif_image_is_classified_image_not_video(tmp_path):
    """LE piège du fichier : `ftyp` matche AUSSI l'AVIF, donc la sonde vidéo
    dit True sur une image AVIF. Seul l'ordre (image testée EN PREMIER) la
    classe correctement. Ce test tombe si quelqu'un inverse les deux blocs."""
    path = _write(tmp_path, 'photo.avif', AVIF)
    assert netfetch._looks_like_video(path) is True        # la sonde vidéo se fait avoir
    assert netfetch._validate_media_file(path) == (True, 'image')   # l'ordre rattrape


def test_a_gif_is_classified_image_not_video(tmp_path):
    """Même mécanique que l'AVIF : `GIF8` est reconnu des deux côtés. Conséquence
    assumée de l'ordre : sur le chemin vidéo-only, un GIF est refusé — une voie
    qui voudrait des GIF animés doit le demander explicitement, pas hériter du
    hasard d'un ordre de tests."""
    path = _write(tmp_path, 'anim.gif', GIF)
    assert netfetch._looks_like_video(path) is True
    assert netfetch._validate_media_file(path) == (True, 'image')
    assert netfetch._validate_media_file(path, allow_image=False) == (False, None)


def test_validate_media_file_accepts_a_real_video_on_both_modes(tmp_path):
    path = _write(tmp_path, 'clip.mp4', MP4)
    assert netfetch._validate_media_file(path) == (True, 'video')
    assert netfetch._validate_media_file(path, allow_image=False) == (True, 'video')


def test_validate_media_file_gates_images_on_the_video_only_path(tmp_path):
    path = _write(tmp_path, 'photo.jpg', JPEG)
    assert netfetch._validate_media_file(path) == (True, 'image')
    assert netfetch._validate_media_file(path, allow_image=False) == (False, None)


@pytest.mark.parametrize('allow_image', [True, False])
def test_validate_media_file_refuses_html_whatever_the_mode(tmp_path, allow_image):
    path = _write(tmp_path, 'clip.mp4', HTML)
    assert netfetch._validate_media_file(path, allow_image=allow_image) == (False, None)


# --- 4. download_via_ytdlp : le sous-processus est mocké, la VALIDATION s'exécute ---
#
# C'est le test qui manquait. Il descend dans le corps de `download_via_ytdlp`
# au lieu de le remplacer, donc il touche la ligne qui importait un module
# inexistant : avec l'import fantôme en place, il échoue sur
# ModuleNotFoundError: No module named 'app.upload'.

def _fake_download(files):
    """Fabrique un faux `_download_with_ytdlp` qui écrit `files`
    ({extension: octets}) là où yt-dlp les aurait écrits."""
    def _run(url, dest_template):
        for ext, data in files.items():
            with open(dest_template.replace('%(ext)s', ext), 'wb') as fh:
                fh.write(data)
        return True, None
    return _run


def test_a_real_video_is_kept_and_named(tmp_path, monkeypatch):
    monkeypatch.setattr(netfetch, '_download_with_ytdlp', _fake_download({'mp4': MP4}))

    ok, filename, err = netfetch.download_via_ytdlp(
        'https://example.invalid/watch', str(tmp_path / 'item'))

    assert (ok, filename, err) == (True, 'item.mp4', None)
    assert (tmp_path / 'item.mp4').exists()


def test_an_html_page_saved_as_mp4_is_rejected_and_deleted(tmp_path, monkeypatch):
    """yt-dlp sort 0 en ayant enregistré une page de connexion : sans la
    validation, ce fichier partirait en bank comme une vidéo."""
    monkeypatch.setattr(netfetch, '_download_with_ytdlp', _fake_download({'mp4': HTML}))

    ok, filename, err = netfetch.download_via_ytdlp(
        'https://example.invalid/watch', str(tmp_path / 'item'))

    assert (ok, filename) == (False, None)
    assert err == "The downloaded file is not a valid video."
    assert not (tmp_path / 'item.mp4').exists()


def test_side_files_are_cleaned_and_only_the_video_survives(tmp_path, monkeypatch):
    """yt-dlp dépose souvent une vignette et un .json à côté du média."""
    monkeypatch.setattr(netfetch, '_download_with_ytdlp',
                        _fake_download({'mp4': MP4, 'jpg': JPEG, 'info.json': b'{}'}))

    ok, filename, err = netfetch.download_via_ytdlp(
        'https://example.invalid/watch', str(tmp_path / 'item'))

    assert (ok, filename, err) == (True, 'item.mp4', None)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['item.mp4']


def test_a_failed_download_reports_the_error_and_leaves_nothing_behind(tmp_path, monkeypatch):
    def _run(url, dest_template):
        with open(dest_template.replace('%(ext)s', 'part'), 'wb') as fh:
            fh.write(b'half a file')
        return False, "Download failed (unsupported or unavailable URL)."
    monkeypatch.setattr(netfetch, '_download_with_ytdlp', _run)

    ok, filename, err = netfetch.download_via_ytdlp(
        'https://example.invalid/watch', str(tmp_path / 'item'))

    assert (ok, filename) == (False, None)
    assert err == "Download failed (unsupported or unavailable URL)."
    assert list(tmp_path.iterdir()) == []


def test_no_upload_module_is_imported_anywhere_in_netfetch():
    """Garde-fou explicite contre la réapparition du port dormant : `app/upload/`
    n'existe pas dans cette app, un import vers lui ne peut que lever."""
    from pathlib import Path
    source = Path(netfetch.__file__).read_text(encoding='utf-8')
    assert 'upload.routes' not in source
    assert 'from ..upload' not in source
