"""🕸 Scrape → VIDEO BANK: the scraper's third destination.

`/api/scrape/scan` has always returned `type: 'video'` items (RedGifs, Erome,
Picazor, TikTok, X, Civitai, every gallery-dl backed source). Nothing consumed
them: the picker filtered them out and the only way to triage a scraped clip was
to download it by hand into a folder and point a bank at it.

These tests pin the intake, and above all the invariants it INHERITS from the
image lane's `scrape_import_to_bank`:

* ONE inventory path — files land in the bank folder and the ordinary walk
  registers them, never a second insert;
* nothing is judged at intake (a short or still clip is a verdict the metrics
  pass produces, not a download-time rejection);
* content-hash naming, so re-importing the same bytes is idempotent and is
  reported as `already_there` — file identity, not a duplicate verdict;
* the lease is re-checked AFTER the (slow) downloads and before the first write;

Destinations are PERMISSIVE since 863cbb56, exactly like the image lane: any
bank may receive a scrape — including one pointed at the user's own folder
(picking the bank is the consent, and the panel names the folder). The one
refusal that survives is a bank sitting on a DATASET's own folder, where consent
cannot fix what landing clips inside training material would mean.
"""
import json
import os
from unittest.mock import patch

import pytest

from app.config import LOCAL_USER
from app.services import video_bank_service as svc


# --- fixtures of bytes ---------------------------------------------------------
# Real container headers, not `b'video'`: what the intake stores is decided by the
# file's own magic (see `_video_extension_from_magic`), so a fake that skipped it
# would test a code path production never takes.
def _mp4(marker=b'a'):
    return b'\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2' + marker * 64


def _webm(marker=b'b'):
    return b'\x1a\x45\xdf\xa3\x01\x00\x00\x00\x00\x00\x00\x1fwebmB\x87' + marker * 64


def _mov(marker=b'c'):
    return b'\x00\x00\x00\x14ftypqt  \x00\x00\x02\x00qt  ' + marker * 64


def _avi(marker=b'd'):
    return b'RIFF\x24\x00\x00\x00AVI LIST' + marker * 64


def _gif(marker=b'g'):
    """What `netfetch.download_via_ytdlp` will happily hand back: its own video
    check matches GIF, because its first caller (a driver video) can use one."""
    return b'GIF89a\x10\x00\x10\x00\x80\x00\x00' + marker * 64


def _heic(marker=b'h'):
    return b'\x00\x00\x00\x18ftypheic\x00\x00\x00\x00heicmif1' + marker * 64


def _m4a(marker=b'm'):
    return b'\x00\x00\x00\x18ftypM4A \x00\x00\x02\x00isomiso2' + marker * 64


def _item(url, **extra):
    return {'url': url, 'title': '', **extra}


def _websearch_item(url='https://cdn.example.test/clip.mp4'):
    """The one provenance shape a VIDEO item can legitimately carry today: web
    search records the page an item was found on and nothing else."""
    return {'url': url, 'title': 'a clip', 'platform': 'websearch',
            'source_url': 'https://blog.example.test/post/42'}


def _fake_downloader(by_url, hook=None):
    """Stand in for `_download_scrape_video`: writes the bytes into the staging
    folder the service owns, exactly like the real one, and returns its path."""
    seq = {'n': 0}

    def _dl(item, staging_dir):
        if hook is not None:
            hook(item)
        data = by_url.get(item['url'])
        if data is None:
            return ('errors', None)
        seq['n'] += 1
        path = os.path.join(staging_dir, f'staged_{seq["n"]}')
        with open(path, 'wb') as fh:
            fh.write(data)
        return ('ok', path)

    return _dl


def _files(bank):
    return sorted(os.listdir(bank.source_path))


def _sources(bank_id):
    from app.models import VideoSource
    return VideoSource.query.filter_by(bank_id=bank_id).all()


# --- a new bank ----------------------------------------------------------------
def test_scrape_creates_a_video_bank_and_inventories_it(app):
    with app.app_context():
        by_url = {'http://x/a.mp4': _mp4(), 'http://x/b': _webm()}
        with patch.object(svc, '_download_scrape_video', _fake_downloader(by_url)):
            res = svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/a.mp4'), _item('http://x/b')],
                name='Scraped rushes')
        assert res['created'] is True
        assert res['saved'] == 2 and res['added'] == 2, res
        bank = svc.get_bank(LOCAL_USER, res['bank_id'])
        assert bank is not None and bank.name == 'Scraped rushes'
        assert len(_files(bank)) == 2
        # inventoried through the ordinary folder walk, like any other video bank
        assert len(_sources(bank.id)) == 2


def test_the_stored_extension_comes_from_the_container_not_the_url(app):
    """A `.m4v` link, a query-string URL and a watch page all produce names the
    WALK can find — `_scan_folder` only inventories VIDEO_EXTS, so a file stored
    under any other extension would be downloaded and then silently ignored."""
    with app.app_context():
        by_url = {'http://x/one.m4v': _mp4(b'1'), 'http://x/two?id=9': _webm(b'2'),
                  'http://x/three': _mov(b'3'), 'http://x/four.bin': _avi(b'4')}
        items = [_item(u) for u in by_url]
        with patch.object(svc, '_download_scrape_video', _fake_downloader(by_url)):
            res = svc.scrape_import_to_video_bank(LOCAL_USER, items, name='Mixed')
        assert res['saved'] == 4 and res['added'] == 4, res
        bank = svc.get_bank(LOCAL_USER, res['bank_id'])
        exts = sorted(os.path.splitext(f)[1] for f in _files(bank))
        assert exts == ['.avi', '.mov', '.mp4', '.webm']
        assert all(e in svc.VIDEO_EXTS for e in exts)


def test_every_extension_the_magic_can_return_is_one_the_walk_inventories():
    """The contract behind the test above, read from the two sides at once: a new
    container added to `_video_extension_from_magic` without a matching entry in
    VIDEO_EXTS would download files the bank can never list."""
    for head in (_mp4(), _webm(), _mov(), _avi()):
        assert svc._video_extension_from_magic(head) in svc.VIDEO_EXTS


def test_a_blob_that_is_not_a_video_container_is_refused(app):
    """An AVIF picture is ISO-BMFF too — `ftyp` alone is not a video."""
    assert svc._video_extension_from_magic(
        b'\x00\x00\x00\x18ftypavif\x00\x00\x00\x00') is None
    assert svc._video_extension_from_magic(b'<html><body>nope</body>') is None
    with app.app_context():
        by_url = {'http://x/a.mp4': _mp4(), 'http://x/fake.mp4': b'<html>not a video'}
        with patch.object(svc, '_download_scrape_video', _fake_downloader(by_url)):
            res = svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/a.mp4'), _item('http://x/fake.mp4')],
                name='Half junk')
        assert res['saved'] == 1 and res['skipped'].get('not_video') == 1


def test_the_intake_refuses_what_the_downloader_was_happy_to_keep(app):
    """THE INTAKE OWNS ITS OWN ACCEPTANCE, and it is stricter than what brought
    the file here. `download_via_ytdlp` keeps anything with a broad video
    signature — GIF included, and `ftyp` is shared with the whole HEIF picture
    family and with M4A audio. The walk only inventories VIDEO_EXTS, so every one
    of those would land in the bank folder as a file nothing ever lists: counted
    by nobody, cut by nothing, and impossible to explain. Refused here, counted,
    and gone with the staging folder."""
    with app.app_context():
        by_url = {'http://x/ok.mp4': _mp4(), 'http://x/anim.gif': _gif(),
                  'http://x/pic.heic': _heic(), 'http://x/sound.m4a': _m4a()}
        seen_staging = []
        dl = _fake_downloader(by_url)

        def _spy(item, staging_dir):
            seen_staging.append(staging_dir)
            return dl(item, staging_dir)

        with patch.object(svc, '_download_scrape_video', _spy):
            res = svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item(u) for u in by_url], name='Strict intake')
        assert res['saved'] == 1, res
        assert res['skipped'].get('not_video') == 3, res
        bank = svc.get_bank(LOCAL_USER, res['bank_id'])
        # ONE file in the folder, and it is the mp4 — nothing dead left behind.
        assert [os.path.splitext(f)[1] for f in _files(bank)] == ['.mp4']
        assert len(_sources(bank.id)) == 1
        # …and the refused bytes are not lingering in a temp folder either.
        assert seen_staging and not os.path.isdir(seen_staging[0])


def test_the_magic_check_refuses_every_non_video_iso_bmff_brand():
    """`ftyp` proves nothing on its own — the brand does. Read from the two sides
    at once so a brand added to the refusal list keeps its meaning."""
    assert svc._video_extension_from_magic(_gif()) is None
    assert svc._video_extension_from_magic(_heic()) is None
    assert svc._video_extension_from_magic(_m4a()) is None
    for brand in svc._NON_VIDEO_BMFF_BRANDS:
        head = b'\x00\x00\x00\x18ftyp' + brand + b'\x00\x00\x02\x00'
        assert svc._video_extension_from_magic(head) is None, brand
    # An unknown brand is still MP4: they are overwhelmingly MP4 profiles, and a
    # file that turns out to hold no video stream is the probe pass's business.
    assert svc._video_extension_from_magic(
        b'\x00\x00\x00\x18ftypdash\x00\x00\x02\x00') == '.mp4'


def test_a_gif_the_resolver_kept_is_a_skip_and_not_a_bank_file(app):
    """The same thing again, through the REAL `_download_scrape_video` with only
    the network end faked — the resolver says ok, the intake still says no."""
    from app.scrape import netfetch

    def _ytdlp_keeps_a_gif(url, dest_base):
        name = os.path.basename(dest_base) + '.gif'
        with open(os.path.join(os.path.dirname(dest_base), name), 'wb') as fh:
            fh.write(_gif())
        return (True, name, None)

    with app.app_context(), \
         patch.object(netfetch, '_validate_public_http_url', lambda u: (True, None)), \
         patch.object(netfetch, 'download_via_ytdlp', _ytdlp_keeps_a_gif):
        res = svc.scrape_import_to_video_bank(
            LOCAL_USER, [_item('https://www.redgifs.com/watch/animated')],
            name='Resolver kept a gif')
    with app.app_context():
        assert res['saved'] == 0 and res['skipped'].get('not_video') == 1, res
        bank = svc.get_bank(LOCAL_USER, res['bank_id'])
        assert _files(bank) == [] and _sources(bank.id) == []


def test_nothing_is_judged_at_intake(app):
    """The image lane keeps images the DATASET outlet would reject, for the same
    reason this one keeps every clip it downloads: short, still and duplicated are
    verdicts the metrics pass produces with thresholds the user moves, and a file
    the bank never received cannot be reviewed."""
    with app.app_context():
        by_url = {'http://x/tiny.mp4': _mp4(b'x')[:80], 'http://x/big.mp4': _mp4(b'y')}
        with patch.object(svc, '_download_scrape_video', _fake_downloader(by_url)):
            res = svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/tiny.mp4'), _item('http://x/big.mp4')],
                name='Unjudged')
        assert res['saved'] == 2 and res['skipped'] == {}


# --- resume --------------------------------------------------------------------
def test_a_second_scrape_on_the_same_bank_appends(app):
    with app.app_context():
        with patch.object(svc, '_download_scrape_video',
                          _fake_downloader({'http://x/a.mp4': _mp4(b'1')})):
            res1 = svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/a.mp4')], name='Growing')
        with patch.object(svc, '_download_scrape_video',
                          _fake_downloader({'http://x/b.mp4': _mp4(b'2')})):
            res2 = svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/b.mp4')], bank_id=res1['bank_id'])
        assert res2['created'] is False and res2['bank_id'] == res1['bank_id']
        assert res2['saved'] == 1 and res2['added'] == 1
        bank = svc.get_bank(LOCAL_USER, res1['bank_id'])
        assert len(_files(bank)) == 2 and len(_sources(bank.id)) == 2


def test_re_downloading_the_same_bytes_is_idempotent_not_a_dedup_verdict(app):
    """Identical bytes = the same FILE. It lands on the same content-hash name and
    is reported as `already_there`, never as a 'duplicate' skip — that word
    belongs to the bank's own passes."""
    with app.app_context():
        by_url = {'http://x/a.mp4': _mp4(b'same')}
        with patch.object(svc, '_download_scrape_video', _fake_downloader(by_url)):
            res1 = svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/a.mp4')], name='Same twice')
            res2 = svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/a.mp4')], bank_id=res1['bank_id'])
        assert res2['saved'] == 0 and res2['already_there'] == 1
        assert 'duplicates' not in res2['skipped']
        bank = svc.get_bank(LOCAL_USER, res1['bank_id'])
        assert len(_files(bank)) == 1 and len(_sources(bank.id)) == 1


def test_a_busy_bank_refuses_the_scrape(app):
    with app.app_context():
        with patch.object(svc, '_download_scrape_video', _fake_downloader({})):
            res = svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/a.mp4')], name='Busy')
        bank_id = res['bank_id']
        from app.services import bank_jobs
        # The registry is keyed on the video lane's namespaced key; a pass on the
        # IMAGE bank with the same number must not refuse this one.
        with patch.object(bank_jobs, 'running',
                          lambda key: key == svc.job_key(bank_id)), \
             pytest.raises(bank_jobs.BankJobBusy):
            svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/b.mp4')], bank_id=bank_id)


def test_the_lease_is_rechecked_after_the_downloads_and_before_the_first_write(app):
    """Downloads are slow — minutes, for videos. A lease that was purged or taken
    over while they ran must not publish files beside a newer owner, so the
    capability is re-asserted at the moment of the first write, not only at entry."""
    with app.app_context():
        from app.services import bank_jobs
        with patch.object(svc, '_download_scrape_video',
                          _fake_downloader({'http://x/a.mp4': _mp4(b'1')})):
            res = svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/a.mp4')], name='Stolen')
        bank = svc.get_bank(LOCAL_USER, res['bank_id'])
        before = _files(bank)

        stealer = _fake_downloader({'http://x/b.mp4': _mp4(b'2')},
                                   hook=lambda _item: bank_jobs.reset())
        with patch.object(svc, '_download_scrape_video', stealer), \
             pytest.raises(RuntimeError):
            svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/b.mp4')], bank_id=res['bank_id'])
        assert _files(bank) == before        # nothing was published


# --- where a scrape may land ---------------------------------------------------
def _own_folder_bank(tmp_path, name='Own footage', folder='my_rushes'):
    """A bank pointed at a folder of the USER's own, the way `create_bank` makes
    one — the destination this lane spent a wave refusing."""
    from app.extensions import db
    from app.models import VideoBank
    rushes = tmp_path / folder
    rushes.mkdir(exist_ok=True)
    bank = VideoBank(user_id=LOCAL_USER, name=name, source_path=str(rushes))
    db.session.add(bank)
    db.session.commit()
    return bank, rushes


def test_a_bank_over_the_users_own_folder_receives_the_downloads(app, tmp_path):
    """PICKING THE BANK IS THE CONSENT.

    This shipped the other way round for one wave: a bank pointed at your own
    rushes was refused, on the ground that the lane never writes into a folder
    you own. It answered a question nobody had asked — "may the app write here?"
    — instead of the one they had: put these clips in THAT bank. The passes still
    never touch your files; a scrape is an errand you sent, and it lands where you
    sent it."""
    with app.app_context():
        bank, rushes = _own_folder_bank(tmp_path)
        (rushes / 'already_mine.mp4').write_bytes(_mp4(b'mine'))
        assert svc.is_app_owned_scrape_folder(str(rushes)) is False
        with patch.object(svc, '_download_scrape_video',
                          _fake_downloader({'http://x/a.mp4': _mp4(b'new')})):
            res = svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/a.mp4')], bank_id=bank.id)
        assert res['saved'] == 1 and res['created'] is False
        # The clip is IN the user's folder, beside what was already there…
        assert len(_files(bank)) == 2
        assert 'already_mine.mp4' in _files(bank)
        # …and both are inventoried: the walk is still the only intake.
        assert res['added'] == 2
        assert len(_sources(bank.id)) == 2


def test_provenance_still_reaches_a_bank_of_the_users_own(app, tmp_path):
    """Nothing about the destination changes what is recorded."""
    with app.app_context():
        bank, _rushes = _own_folder_bank(tmp_path, folder='provenance_rushes')
        item = _websearch_item()
        with patch.object(svc, '_download_scrape_video',
                          _fake_downloader({item['url']: _mp4()})):
            svc.scrape_import_to_video_bank(LOCAL_USER, [item], bank_id=bank.id)
        rows = _sources(bank.id)
        assert len(rows) == 1
        assert json.loads(rows[0].source_metadata) == {
            'platform': 'websearch',
            'source_url': 'https://blog.example.test/post/42',
        }


def test_a_refused_file_never_reaches_the_users_own_folder(app, tmp_path):
    """The staging folder protects YOUR folder too, and that matters more here
    than it did when every destination was app-owned: a GIF or an audio file the
    resolver was happy to keep is refused by the intake, and it must not appear
    in an archive of originals even for the moment it takes to check it. Nothing
    is ever written into the destination before its bytes have been accepted."""
    with app.app_context():
        bank, rushes = _own_folder_bank(tmp_path, folder='pristine_rushes')
        by_url = {'http://x/anim.gif': _gif(), 'http://x/sound.m4a': _m4a()}
        with patch.object(svc, '_download_scrape_video', _fake_downloader(by_url)):
            res = svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item(u) for u in by_url], bank_id=bank.id)
        assert res['saved'] == 0 and res['skipped'].get('not_video') == 2
        assert os.listdir(rushes) == []          # not one byte, not even briefly
        assert _sources(bank.id) == []


def test_a_failed_import_never_deletes_a_folder_of_the_users_own(app, tmp_path):
    """The cleanup path is the one place `rmtree` appears in this lane, and it is
    now reachable with a user folder in scope. It removes ONLY a folder the app
    created — `is_app_owned_scrape_folder` is what keeps a failed import from
    taking someone's rushes with it."""
    with app.app_context():
        _bank, rushes = _own_folder_bank(tmp_path, folder='precious_rushes')
        (rushes / 'keep_me.mp4').write_bytes(_mp4(b'keep'))
        svc._discard_unlaunched_scrape_bank(LOCAL_USER, None, str(rushes))
        assert rushes.is_dir() and os.listdir(rushes) == ['keep_me.mp4']


def test_the_bank_list_says_where_a_scrape_would_land(app, tmp_path):
    """`scrapable` survives as the picker's filter, but it now answers the only
    question left (is this a dataset's folder?). `app_folder` is what tells the
    picker whether it has something to WARN about — a folder of the user's own is
    where a download is worth a sentence before the click."""
    with app.app_context():
        _own_folder_bank(tmp_path)
        with patch.object(svc, '_download_scrape_video',
                          _fake_downloader({'http://x/a.mp4': _mp4()})):
            res = svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/a.mp4')], name='Scraped')
        by_name = {b['name']: b for b in svc.list_banks(LOCAL_USER)}
        # BOTH are offerable now — that is the whole change.
        assert by_name['Scraped']['scrapable'] is True
        assert by_name['Own footage']['scrapable'] is True
        assert by_name['Scraped']['id'] == res['bank_id']
        # …and they are told apart by who owns the folder, not by who may write.
        assert by_name['Scraped']['app_folder'] is True
        assert by_name['Own footage']['app_folder'] is False
        # The path rides along, because the warning names it.
        assert by_name['Own footage']['source_path']


def test_a_bank_sitting_on_a_dataset_folder_is_refused(app):
    """THE ONE REFUSAL THAT SURVIVED "picking the bank is the consent".

    A user CAN consent to files landing in a folder of their own — that is the
    whole change. They cannot usefully consent to files landing inside a
    dataset's own storage: those would be trained on, and attributed to a dataset
    nobody added them to. Reachable when the datasets root has been relocated over
    the app's own data — exactly what a relocation makes possible."""
    with app.app_context():
        from app import config as cfg
        with patch.object(svc, '_download_scrape_video',
                          _fake_downloader({'http://x/a.mp4': _mp4()})):
            res = svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/a.mp4')], name='Overlapping')
        # Point the datasets root at the folder that CONTAINS this bank's folder.
        cfg.save_config({'paths': {
            'dataset_images_root': str(cfg.video_bank_sources_root())}})
        before = _files(svc.get_bank(LOCAL_USER, res['bank_id']))
        with patch.object(svc, '_download_scrape_video',
                          _fake_downloader({'http://x/b.mp4': _mp4(b'2')})), \
             pytest.raises(ValueError) as err:
            svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/b.mp4')], bank_id=res['bank_id'])
        assert 'dataset' in str(err.value)
        assert _files(svc.get_bank(LOCAL_USER, res['bank_id'])) == before
        # …and the picker is told, so nobody is offered that bank in the first place.
        assert svc.list_banks(LOCAL_USER)[0]['scrapable'] is False


def test_a_bank_the_user_pointed_at_a_dataset_folder_is_refused_too(app, tmp_path):
    """The likelier shape of the same trap now that any folder is a destination:
    the user points a video bank straight at a dataset's image folder."""
    with app.app_context():
        from app import config as cfg
        from app.extensions import db
        from app.models import VideoBank
        inside = cfg.dataset_images_root() / '42'
        inside.mkdir(parents=True, exist_ok=True)
        bank = VideoBank(user_id=LOCAL_USER, name='On a dataset',
                         source_path=str(inside))
        db.session.add(bank)
        db.session.commit()
        with patch.object(svc, '_download_scrape_video',
                          _fake_downloader({'http://x/a.mp4': _mp4()})), \
             pytest.raises(ValueError) as err:
            svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/a.mp4')], bank_id=bank.id)
        assert 'dataset' in str(err.value)
        assert os.listdir(inside) == []
        assert svc.list_banks(LOCAL_USER)[0]['scrapable'] is False


def test_two_scrapes_of_the_same_name_never_share_a_folder(app):
    with app.app_context():
        with patch.object(svc, '_download_scrape_video',
                          _fake_downloader({'http://x/a.mp4': _mp4(b'1')})):
            a = svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/a.mp4')], name='Same name')
            b = svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/a.mp4')], name='Same name')
        pa = svc.get_bank(LOCAL_USER, a['bank_id']).source_path
        pb = svc.get_bank(LOCAL_USER, b['bank_id']).source_path
        assert pa != pb


# --- routing: a file to fetch vs a page to resolve ------------------------------
def test_direct_media_urls_are_streamed_and_pages_go_to_the_resolver(app):
    """gallery-dl hands back a CDN media URL; RedGifs and friends hand back a
    watch page. Paying a yt-dlp subprocess to rediscover a link that already ends
    in `.mp4` is a second request for bytes we can simply stream."""
    assert svc._scrape_video_route('https://cdn.example.test/a/clip.mp4') == 'direct'
    assert svc._scrape_video_route('https://cdn.example.test/a/clip.M4V') == 'direct'
    assert svc._scrape_video_route('https://cdn.example.test/clip.webm?x=1') == 'direct'
    assert svc._scrape_video_route('https://www.redgifs.com/watch/abcdef') == 'resolve'
    assert svc._scrape_video_route('https://x.example.test/user/status/1') == 'resolve'
    assert svc._scrape_video_route(None) == 'resolve'


def test_the_two_routes_call_the_two_downloaders(app, tmp_path):
    """Pinned on the real `_download_scrape_video`, with only the network ends
    faked: this is the wiring the routing decision exists for."""
    from app.scrape import netfetch
    calls = []

    def _fake_stream(url, dest):
        calls.append(('stream', url))
        with open(dest, 'wb') as fh:
            fh.write(_mp4())
        return 'ok'

    def _fake_ytdlp(url, dest_base):
        calls.append(('ytdlp', url))
        name = os.path.basename(dest_base) + '.mp4'
        with open(os.path.join(os.path.dirname(dest_base), name), 'wb') as fh:
            fh.write(_webm())
        return (True, name, None)

    with app.app_context(), \
         patch.object(netfetch, '_validate_public_http_url', lambda u: (True, None)), \
         patch.object(netfetch, 'download_via_ytdlp', _fake_ytdlp), \
         patch.object(svc, '_stream_video_to_disk', _fake_stream):
        res = svc.scrape_import_to_video_bank(
            LOCAL_USER, [_item('https://cdn.example.test/clip.mp4'),
                         _item('https://www.redgifs.com/watch/abcdef')],
            name='Both routes')
    assert res['saved'] == 2, res
    assert [kind for kind, _url in calls] == ['stream', 'ytdlp'] or \
           [kind for kind, _url in calls] == ['ytdlp', 'stream']
    assert {kind for kind, _url in calls} == {'stream', 'ytdlp'}


def test_a_resolver_that_blows_up_skips_its_item_not_the_batch(app):
    """`download_via_ytdlp` is a subprocess plus optional imports. Whatever it
    fails on, the other picks must still land."""
    from app.scrape import netfetch

    def _boom(url, dest_base):
        raise ImportError('no module named upload.routes')

    def _fake_stream(url, dest):
        with open(dest, 'wb') as fh:
            fh.write(_mp4())
        return 'ok'

    with app.app_context(), \
         patch.object(netfetch, '_validate_public_http_url', lambda u: (True, None)), \
         patch.object(netfetch, 'download_via_ytdlp', _boom), \
         patch.object(svc, '_stream_video_to_disk', _fake_stream):
        res = svc.scrape_import_to_video_bank(
            LOCAL_USER, [_item('https://cdn.example.test/clip.mp4'),
                         _item('https://www.redgifs.com/watch/abcdef')],
            name='One resolver down')
    assert res['saved'] == 1 and res['skipped'].get('errors') == 1


# --- validation ----------------------------------------------------------------
def test_name_is_required_for_a_new_bank(app):
    with app.app_context():
        with pytest.raises(ValueError):
            svc.scrape_import_to_video_bank(LOCAL_USER, [_item('http://x/a.mp4')],
                                            name='  ')


def test_unknown_bank_id_is_rejected(app):
    with app.app_context():
        with pytest.raises(ValueError):
            svc.scrape_import_to_video_bank(LOCAL_USER, [_item('http://x/a.mp4')],
                                            bank_id=9999)


def test_an_empty_selection_is_rejected(app):
    with app.app_context():
        with pytest.raises(ValueError):
            svc.scrape_import_to_video_bank(LOCAL_USER, [], name='Nothing')


def test_the_per_request_cap_is_lower_than_the_image_outlets(app):
    """Deliberately: one image is capped at 12 MB and 20 s, one video at 200 MB
    and 180 s. A big selection is not refused — the client sends batches."""
    from app.services.face_dataset_service import SCRAPE_IMPORT_MAX
    assert svc.SCRAPE_VIDEO_IMPORT_MAX < SCRAPE_IMPORT_MAX
    with app.app_context():
        items = [_item(f'http://x/{i}.mp4')
                 for i in range(svc.SCRAPE_VIDEO_IMPORT_MAX + 1)]
        with pytest.raises(ValueError):
            svc.scrape_import_to_video_bank(LOCAL_USER, items, name='Too many')


def test_a_failed_download_is_counted_and_never_stored(app):
    with app.app_context():
        by_url = {'http://x/a.mp4': _mp4(), 'http://x/dead.mp4': None}
        with patch.object(svc, '_download_scrape_video', _fake_downloader(by_url)):
            res = svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/a.mp4'), _item('http://x/dead.mp4')],
                name='Half dead')
        assert res['saved'] == 1 and res['skipped'].get('errors') == 1
        bank = svc.get_bank(LOCAL_USER, res['bank_id'])
        assert len(_files(bank)) == 1


def test_a_new_bank_whose_import_blows_up_leaves_nothing_behind(app):
    """No half-created bank, no orphan folder: the destination is discarded when
    the worker never took ownership, exactly like the image lane's import."""
    with app.app_context():
        from app import config as cfg
        before = sorted(os.listdir(cfg.video_bank_sources_root()))

        def _explode(item, staging_dir):
            raise OSError('disk went away')

        with patch.object(svc, '_download_scrape_video', _explode), \
             pytest.raises(OSError):
            svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/a.mp4')], name='Doomed')
        assert svc.list_banks(LOCAL_USER) == []
        assert sorted(os.listdir(cfg.video_bank_sources_root())) == before


# --- provenance ----------------------------------------------------------------
def test_provenance_is_validated_and_reaches_the_inventoried_row(app):
    """The whole reason the walk takes a metadata map: a scraped rush is born WITH
    its origin. Validated through the SAME gate as the image lane
    (`normalize_source_metadata`), never trusted raw from the client."""
    with app.app_context():
        item = _websearch_item()
        with patch.object(svc, '_download_scrape_video',
                          _fake_downloader({item['url']: _mp4()})):
            res = svc.scrape_import_to_video_bank(LOCAL_USER, [item],
                                                  name='With provenance')
        rows = _sources(res['bank_id'])
        assert len(rows) == 1
        assert json.loads(rows[0].source_metadata) == {
            'platform': 'websearch',
            'source_url': 'https://blog.example.test/post/42',
        }


def test_a_platform_the_gate_does_not_know_stores_no_provenance(app):
    """A RedGifs or gallery-dl item records nothing rather than a guess — the same
    rule the image outlet applies to an unrecognised platform."""
    with app.app_context():
        item = _item('https://www.redgifs.com/watch/abcdef', platform='redgifs',
                     source_url='https://www.redgifs.com/watch/abcdef')
        with patch.object(svc, '_download_scrape_video',
                          _fake_downloader({item['url']: _mp4()})):
            res = svc.scrape_import_to_video_bank(LOCAL_USER, [item], name='Unknown')
        rows = _sources(res['bank_id'])
        assert len(rows) == 1 and rows[0].source_metadata is None


def test_a_spoofed_source_url_is_dropped_not_trusted_raw(app):
    """`source_url` has to survive the shared validator: a non-https page, or one
    with credentials in it, is not provenance."""
    with app.app_context():
        item = _websearch_item()
        item['source_url'] = 'http://blog.example.test/post/42'   # not https
        with patch.object(svc, '_download_scrape_video',
                          _fake_downloader({item['url']: _mp4()})):
            res = svc.scrape_import_to_video_bank(LOCAL_USER, [item], name='Spoofed')
        rows = _sources(res['bank_id'])
        assert len(rows) == 1 and rows[0].source_metadata is None


def test_a_bank_created_by_hand_records_no_origin(app, tmp_path):
    """NULL is the honest value for a file the user pointed the bank at: nobody
    recorded where it came from, and inventing one would be a claim."""
    with app.app_context():
        folder = tmp_path / 'rushes'
        folder.mkdir()
        (folder / 'a.mp4').write_bytes(_mp4())
        bank, added = svc.create_bank(LOCAL_USER, 'By hand', str(folder))
        assert added == 1
        assert _sources(bank.id)[0].source_metadata is None


# --- route ---------------------------------------------------------------------
def test_route_creates_then_resumes(app, client):
    by_url = {'http://x/a.mp4': _mp4(b'1'), 'http://x/b.mp4': _mp4(b'2')}
    with patch.object(svc, '_download_scrape_video', _fake_downloader(by_url)):
        r = client.post('/api/video-bank/scrape-import',
                        json={'items': [_item('http://x/a.mp4')], 'name': 'Via HTTP'})
        assert r.status_code == 200, r.get_json()
        body = r.get_json()
        assert body['ok'] and body['created'] and body['saved'] == 1
        r2 = client.post('/api/video-bank/scrape-import',
                         json={'items': [_item('http://x/b.mp4')],
                               'bank_id': body['bank_id']})
        assert r2.status_code == 200, r2.get_json()
        assert r2.get_json()['created'] is False and r2.get_json()['saved'] == 1


def test_route_rejects_an_empty_selection(app, client):
    r = client.post('/api/video-bank/scrape-import', json={'items': [], 'name': 'x'})
    assert r.status_code == 400


def test_route_rejects_a_bank_id_that_is_not_a_number(app, client):
    r = client.post('/api/video-bank/scrape-import',
                    json={'items': [_item('http://x/a.mp4')], 'bank_id': 'seven'})
    assert r.status_code == 400


def test_route_reports_a_busy_bank_as_409(app, client):
    from app.services import bank_jobs
    with patch.object(svc, '_download_scrape_video', _fake_downloader({})):
        r = client.post('/api/video-bank/scrape-import',
                        json={'items': [_item('http://x/a.mp4')], 'name': 'Busy HTTP'})
    bank_id = r.get_json()['bank_id']
    with patch.object(bank_jobs, 'running',
                      lambda key: key == svc.job_key(bank_id)):
        r2 = client.post('/api/video-bank/scrape-import',
                         json={'items': [_item('http://x/b.mp4')],
                               'bank_id': bank_id})
    assert r2.status_code == 409 and r2.get_json().get('busy_kind')


# --- honesty of failure paths (verification wave, 2026-08-18) -------------------
def test_an_unreadable_download_is_an_error_not_a_verdict_about_the_file(app):
    """A file that arrived but cannot be read back (antivirus lock, lost handle)
    is a MACHINE problem. Counting it 'not_video' told the user their perfectly
    valid clip was not a video."""
    with app.app_context():
        by_url = {'http://x/a.mp4': _mp4()}

        def _boom(path):
            raise OSError('locked by something else')

        with patch.object(svc, '_download_scrape_video', _fake_downloader(by_url)), \
             patch.object(svc, '_video_blob_name', _boom):
            res = svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/a.mp4')], name='Locked')
        assert res['skipped'] == {'errors': 1}, res
        assert 'not_video' not in res['skipped']


def test_a_failure_after_files_landed_keeps_the_bank_and_the_files(app):
    """The discard guard (`bank_jobs.launched`) is structurally always False on
    this synchronous path, so ANY late exception used to reach the cleanup and
    `rmtree` videos that had downloaded perfectly. A folder that holds files is
    an import that DID get somewhere: bank and files survive, the error still
    surfaces."""
    with app.app_context():
        by_url = {'http://x/a.mp4': _mp4()}

        def _sync_blows_up(*a, **k):
            raise RuntimeError('database is locked')

        with patch.object(svc, '_download_scrape_video', _fake_downloader(by_url)), \
             patch.object(svc, 'refresh_bank', _sync_blows_up), \
             pytest.raises(RuntimeError):
            svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/a.mp4')], name='Survivor')
        banks = svc.list_banks(LOCAL_USER)
        assert len(banks) == 1, 'the bank row must survive a late failure'
        bank = svc.get_bank(LOCAL_USER, banks[0]['id'])
        assert len(_files(bank)) == 1, 'the downloaded file must survive'


def test_a_truncated_publication_never_wears_the_final_name(app, monkeypatch):
    """Publication goes staging -> `<hash>.part` -> atomic rename. If it dies
    mid-way the partial file must NOT sit under its final content-hash name:
    the walk would inventory a truncated rush that no re-import of the same
    bytes ever repairs (the hash "already exists")."""
    with app.app_context():
        by_url = {'http://x/a.mp4': _mp4()}
        real_replace = os.replace

        def _no_replace(src, dst, *a, **k):
            if str(src).endswith('.part'):
                raise OSError('volume unplugged mid-copy')
            return real_replace(src, dst, *a, **k)

        monkeypatch.setattr(svc.os, 'replace', _no_replace)
        with patch.object(svc, '_download_scrape_video', _fake_downloader(by_url)):
            res = svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/a.mp4')], name='Torn')
        assert res['saved'] == 0 and res['skipped'] == {'errors': 1}, res
        bank = svc.get_bank(LOCAL_USER, res['bank_id'])
        assert _files(bank) == [], 'neither the final name nor a .part may remain'


def test_the_walk_is_skipped_while_the_bank_is_busy(app):
    """`refresh_bank` now serialises like the image lane. The race it closes:
    the scrape holds the lease through minutes of downloads, opening the
    workspace fires a forced refresh, and that concurrent walk used to
    inventory the freshly-moved files WITHOUT their provenance."""
    from app.services import bank_jobs
    with app.app_context():
        by_url = {'http://x/a.mp4': _mp4()}
        with patch.object(svc, '_download_scrape_video', _fake_downloader(by_url)):
            res = svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/a.mp4')], name='Busy walk')
        bank_id = res['bank_id']
        with bank_jobs.mutation_lease(svc.job_key(bank_id), 'measure'):
            out = svc.refresh_bank(LOCAL_USER, bank_id, force=True)
        assert out == {'added': 0, 'missing': 0, 'unavailable': False,
                       'error': None, 'busy': True}
        # And once the lease is gone the walk runs again.
        assert 'busy' not in (svc.refresh_bank(LOCAL_USER, bank_id) or {})


def test_a_sync_failure_is_said_not_swallowed(app):
    """Files downloaded but the walk failed used to read as the perfect run
    ('N videos downloaded') with `added` silently zero."""
    with app.app_context():
        by_url = {'http://x/a.mp4': _mp4()}

        def _broken_sync(*a, **k):
            return {'added': 0, 'missing': 0, 'unavailable': True,
                    'error': 'the source folder is not reachable right now'}

        with patch.object(svc, '_download_scrape_video', _fake_downloader(by_url)), \
             patch.object(svc, 'refresh_bank', _broken_sync):
            res = svc.scrape_import_to_video_bank(
                LOCAL_USER, [_item('http://x/a.mp4')], name='Blind walk')
        assert res['saved'] == 1 and res['added'] == 0
        assert res['sync_error'] == 'the source folder is not reachable right now'


def test_the_resolver_output_is_capped_like_the_direct_branch(app, tmp_path):
    """yt-dlp's fragmented downloaders (HLS/DASH -- what a watch page serves)
    ignore `--max-filesize`; the direct branch counts its bytes during the
    stream. This is the resolver branch's equivalent, after the fact."""
    from app.scrape import netfetch

    def _huge_ytdlp(url, dest_base):
        name = os.path.basename(dest_base) + '.mp4'
        with open(os.path.join(os.path.dirname(dest_base), name), 'wb') as fh:
            fh.write(_mp4() + b'x' * 64)
        return (True, name, None)

    with app.app_context(), \
         patch.object(netfetch, '_validate_public_http_url', lambda u: (True, None)), \
         patch.object(netfetch, 'download_via_ytdlp', _huge_ytdlp), \
         patch.object(netfetch, 'MAX_DRIVER_BYTES', 16):
        reason, path = svc._download_scrape_video(
            _item('https://www.redgifs.com/watch/abcdef'), str(tmp_path))
    assert reason == 'too_large' and path is None
    strays = [f for f in os.listdir(tmp_path) if f.endswith('.mp4')]
    assert strays == [], 'the oversized file must not survive'


def _fake_curl_response(chunks, status=200, ctype='video/mp4'):
    class _R:
        status_code = status
        headers = {'content-type': ctype}

        def iter_content(self, size):
            yield from chunks

        def close(self):
            pass
    return _R()


def test_an_empty_200_body_is_a_failed_download_not_a_verdict(app, monkeypatch, tmp_path):
    """A CDN answering 200 video/* with no bytes used to return 'ok' -- the item
    then read as 'not a video this bank can hold' about a clip that never
    arrived."""
    import sys
    import types
    from app.scrape import netfetch
    stub = types.ModuleType('curl_cffi')
    stub.requests = types.SimpleNamespace(
        get=lambda *a, **k: _fake_curl_response([]))
    monkeypatch.setitem(sys.modules, 'curl_cffi', stub)
    with app.app_context(), \
         patch.object(netfetch, '_validate_public_http_url', lambda u: (True, None)):
        reason = svc._stream_video_to_disk('https://cdn.example.test/clip.mp4',
                                           str(tmp_path / 'out'))
    assert reason == 'errors'


def test_a_missing_curl_cffi_has_its_own_reason(app, monkeypatch, tmp_path):
    """On an install without the scrape extras EVERY direct-file item fails;
    'no_curl' (the image lane's word for the same case) lets the client say
    'install the scrape extras' instead of blaming the network."""
    import sys
    from app.scrape import netfetch
    monkeypatch.setitem(sys.modules, 'curl_cffi', None)
    with app.app_context(), \
         patch.object(netfetch, '_validate_public_http_url', lambda u: (True, None)):
        reason = svc._stream_video_to_disk('https://cdn.example.test/clip.mp4',
                                           str(tmp_path / 'out'))
    assert reason == 'no_curl'
