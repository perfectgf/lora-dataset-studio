"""Source universelle : garde SSRF, énumération générique, repli yt-dlp.

Tout est mocké — aucun appel réseau ni process gallery-dl."""

from app.scrape import netfetch
from app.scrape.sources import gdl
from app.scrape.sources.universal import UniversalSource


class _Proc:
    """Faux CompletedProcess : seuls returncode/stdout/stderr sont lus."""

    def __init__(self, returncode=0, stdout='', stderr=''):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_unsupported_site_on_a_vetted_host_falls_back_to_ytdlp(monkeypatch, tmp_path):
    """gallery-dl sans extracteur (exit 64) sur un hôte vetté DOIT basculer sur
    yt-dlp. Historiquement faux : le kind était comparé à une phrase que personne
    n'émettait, donc la bascule n'a jamais eu lieu."""
    monkeypatch.setattr(gdl.subprocess, 'run',
                        lambda *a, **k: _Proc(returncode=64, stderr='Unsupported URL'))
    called = {}

    def fake_ytdlp(url, dest_base):
        called['url'] = url
        return True, 'video.mp4', None
    monkeypatch.setattr(netfetch, 'download_via_ytdlp', fake_ytdlp)

    ok, filename, err = UniversalSource().download(
        'https://x.com/someone/status/1', str(tmp_path / 'item'))

    assert (ok, filename, err) == (True, 'video.mp4', None)
    assert called['url'] == 'https://x.com/someone/status/1'


def test_unsupported_site_on_an_unvetted_host_refuses_instead_of_fetching(
        monkeypatch, tmp_path):
    monkeypatch.setattr(gdl.subprocess, 'run',
                        lambda *a, **k: _Proc(returncode=64, stderr='Unsupported URL'))

    def boom(url, dest_base):
        raise AssertionError('yt-dlp ne doit PAS être appelé sur un hôte non vetté')
    monkeypatch.setattr(netfetch, 'download_via_ytdlp', boom)

    ok, filename, err = UniversalSource().download(
        'https://unknown.example/thing', str(tmp_path / 'item'))

    assert ok is False and filename is None
    assert 'not vetted' in err


def test_auth_failure_is_reported_and_never_retried_via_ytdlp(monkeypatch, tmp_path):
    """Exit 16 = mur d'authentification, PAS « site inconnu » : on remonte
    l'erreur au lieu de retenter avec un autre outil."""
    monkeypatch.setattr(gdl.subprocess, 'run',
                        lambda *a, **k: _Proc(returncode=16, stderr='login required'))

    def boom(url, dest_base):
        raise AssertionError('yt-dlp ne doit PAS être appelé sur un échec auth')
    monkeypatch.setattr(netfetch, 'download_via_ytdlp', boom)

    ok, _filename, err = UniversalSource().download(
        'https://x.com/someone/status/1', str(tmp_path / 'item'))

    assert ok is False and 'auth' in err


# --- Garde SSRF : la source générique est la SEULE à accepter un hôte arbitraire,
# et son scan lance désormais gallery-dl dessus. Les sources dédiées matchent des
# hôtes nommés, une adresse privée ne les a jamais atteintes.
def test_match_refuses_non_public_urls():
    src = UniversalSource()
    for url in ('http://127.0.0.1/gallery',
                'http://localhost:8080/gallery',
                'http://192.168.1.10/gallery',
                'http://[::1]/gallery',
                'file:///etc/passwd'):
        assert src.match(url) is None, url


def test_match_still_accepts_a_public_http_url(monkeypatch):
    # example.test (domaine de test RFC 2606, SANS enregistrement DNS réel) :
    # on simule la résolution pour rester hermétique (aucun appel réseau) tout
    # en exerçant la vraie branche de classification d'IP du garde SSRF.
    # RFC 5737 (203.0.113.1, etc.) échoue ici : is_reserved → _ip_is_blocked
    # le rejette correctement. Adresse publique ordinaire pour tester la branche
    # accepte; ne cible ni ce poste ni le réseau (la garde cible les données perso).
    monkeypatch.setattr(
        netfetch.socket, 'getaddrinfo',
        lambda *a, **k: [(netfetch.socket.AF_INET, netfetch.socket.SOCK_STREAM,
                           6, '', ('93.184.216.34', 443))])
    assert UniversalSource().match('https://example.test/album/1') is not None


def test_classify_exit_none_is_not_silently_a_success():
    """`subprocess.run` ne produit jamais `returncode=None`, mais un double de
    test bâclé le peut. `if not returncode` (faussité Python) confondait ce cas
    avec le code 0 (succès) — puis 'empty' au point d'appel (_run_simulate) : un
    tool ayant planté sans code de sortie exploitable se serait fait passer
    pour un scan vide réussi. L'égalité stricte à 0 exclut ce cas ; None reste
    un échec explicite ('toolerror'), jamais un succès déguisé."""
    assert gdl.classify_exit(0) is None
    assert gdl.classify_exit(None) == 'toolerror'


def test_enumerate_album_recursion_sentinel_carries_a_kind(monkeypatch):
    """Quand TOUS les albums échouent via le sentinel type -1 (pas via une
    exception/`_run_simulate` en erreur), l'erreur remontée par `enumerate()`
    doit rester une GdlError utilisable par `getattr(err, 'kind', None)` — pas
    un str nu redevenu invisible au branchement kind."""

    def fake_run_simulate(url, max_items, cookies, extra_opts, image_range=None):
        if 'category' in url:
            return [[6, 'https://x/album1/', {}]], None
        return [[-1, {'message': 'blocked by extractor'}]], None

    monkeypatch.setattr(gdl, '_run_simulate', fake_run_simulate)

    items, err = gdl.enumerate('https://x/category/')

    assert items is None
    assert getattr(err, 'kind', None) == 'toolerror'


# --- Énumération générique -----------------------------------------------------
from app.scrape.sources.base import Match   # noqa: E402  (groupé avec ses tests)


def _spy_enumerate(monkeypatch, items=None, err=None):
    seen = {}

    def fake(url, **kw):
        seen['url'] = url
        seen.update(kw)
        return items, err
    monkeypatch.setattr(gdl, 'enumerate', fake)
    return seen


def test_scan_returns_every_image_of_the_page(monkeypatch):
    found = [{'url': f'https://cdn.example.test/{i}.jpg', 'title': '',
              'thumbnail': None, 'type': 'image', 'platform': 'generic'}
             for i in range(3)]
    seen = _spy_enumerate(monkeypatch, items=found)
    m = Match(url='https://example.test/album/1')
    m.page = 0

    items, err = UniversalSource().scan(m)

    assert err is None and items == found
    assert seen['platform'] == 'generic'
    assert seen['per_album'] == 1          # défaut : une cover par album
    assert seen['image_range'] == '1-120'


def test_scan_dives_into_albums_only_when_asked(monkeypatch):
    seen = _spy_enumerate(monkeypatch, items=[{'url': 'https://cdn.example.test/a.jpg',
                                               'title': '', 'thumbnail': None,
                                               'type': 'image', 'platform': 'generic'}])
    m = Match(url='https://example.test/albums/')
    m.page = 0
    m.include_albums = True

    UniversalSource().scan(m)

    assert seen['per_album'] is None       # plongée intégrale


def test_scan_walks_the_listing_window_on_later_pages(monkeypatch):
    seen = _spy_enumerate(monkeypatch, items=[{'url': 'https://cdn.example.test/a.jpg',
                                               'title': '', 'thumbnail': None,
                                               'type': 'image', 'platform': 'generic'}])
    m = Match(url='https://example.test/album/1')
    m.page = 2

    UniversalSource().scan(m)

    assert seen['image_range'] == '241-360'


def test_a_blocked_scan_is_an_error_never_an_empty_result(monkeypatch):
    """429/auth/DDoS-Guard doivent remonter. Les faire passer pour « aucune image »
    est exactement le bug qu'erome a payé."""
    _spy_enumerate(monkeypatch, err=gdl.GdlError('gallery-dl: auth (429).', 'auth'))
    m = Match(url='https://example.test/album/1')
    m.page = 0

    items, err = UniversalSource().scan(m)

    assert items is None
    assert '429' in err


def test_a_genuinely_empty_page_carries_its_kind_for_the_route_to_read(monkeypatch):
    """kind='empty' = gallery-dl a tourné correctement et n'a rien trouvé (post
    supprimé, album vide, mauvais type de page) — un scan vide réussi, pas un
    échec. `scan()` ne le convertit plus lui-même en ([], None) : cette
    conversion vit désormais UNE seule fois, au niveau de la route
    (routes/scrape.py), qui voit TOUTES les sources gdl-backed — pas seulement
    celle-ci. Refaire la conversion ici serait une duplication morte le jour où
    cette source a cessé d'être la seule à honorer la règle. `scan()` doit donc
    juste laisser passer une GdlError dont `.kind` reste lisible."""
    _spy_enumerate(monkeypatch,
                   err=gdl.GdlError('gallery-dl: no media found.', 'empty'))
    m = Match(url='https://example.test/album/1')
    m.page = 0

    items, err = UniversalSource().scan(m)

    assert items is None
    assert getattr(err, 'kind', None) == 'empty'


def test_a_site_gallery_dl_does_not_know_still_yields_the_single_media(monkeypatch):
    """Repli historique : 1 item vidéo, pour que les hôtes vettés atteignent
    yt-dlp au téléchargement. Pas de pagination sur un item unique."""
    _spy_enumerate(monkeypatch,
                   err=gdl.GdlError('gallery-dl: unsupported (no extractor).', 'unsupported'))
    m = Match(url='https://x.com/someone/status/1')
    m.page = 0

    items, err = UniversalSource().scan(m)

    assert err is None
    assert items == [{'url': 'https://x.com/someone/status/1',
                      'title': 'https://x.com/someone/status/1',
                      'thumbnail': None, 'type': 'video', 'platform': 'generic'}]
    assert m.paginated is False


def test_exit_zero_with_no_stdout_is_classified_empty_not_unclassified(monkeypatch):
    """gallery-dl peut sortir en code 0 (succès) sans rien écrire sur stdout —
    une page valide sans média. `classify_exit(0)` renvoie None (pas un kind),
    donc sans ce garde-fou le GdlError produit par `_run_simulate` porte
    kind=None : il ne matche ni 'unsupported' ni 'empty' dans `scan()`, tombe
    dans la branche générique « on remonte l'erreur », et un scan vide légitime
    répond 502 'empty output' au lieu de 200/count=0 (cf.
    `test_scan_empty_kind_is_200_with_zero_items` dans test_scrape_scan.py, qui
    couvre ce même contrat au niveau de la route — le seul endroit qui le
    convertit désormais)."""
    monkeypatch.setattr(gdl.subprocess, 'run',
                        lambda *a, **k: _Proc(returncode=0, stdout='', stderr=''))

    entries, err = gdl._run_simulate('https://example.test/album/1', 60, None, None)

    assert entries is None
    assert getattr(err, 'kind', None) == 'empty'

    _spy_enumerate(monkeypatch,
                   err=gdl.GdlError('gallery-dl: empty output (no data).', 'empty'))
    m = Match(url='https://example.test/album/1')
    m.page = 0

    items, scan_err = UniversalSource().scan(m)

    assert items is None
    assert getattr(scan_err, 'kind', None) == 'empty'
