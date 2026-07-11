"""Source Reddit (recherche par mot-clé via API OAuth).

Reddit ayant verrouillé l'accès anonyme aux endpoints .json (mur anti-bot 403),
la source parle à l'API OAuth authentifiée (jeton installed_client anonyme) et
extrait les images du JSON des posts. Ces tests couvrent SANS RÉSEAU :
  • _endpoint_for : mapping URL canonique → endpoint API (recherche/listing/post/direct)
  • _items_from_post : extraction images (galerie / lien direct / preview / vide)
  • _canonical_reddit_url : purge tracking, garde params de contenu, force www
  • RedditSource.scan : extraction + dedup + pagination par curseur (API mockée)
  • RedditSource.match : détection d'hôte

Le réseau (jeton + _api_get) est monkeypatché — aucun appel réel à Reddit.
"""
from app.scrape.sources import reddit
from app.scrape.sources.base import Match
from app.scrape.validators import Platform, url_validator


# --- _endpoint_for : routage (pur) ------------------------------------------
def test_endpoint_global_search():
    ep = reddit._endpoint_for('https://www.reddit.com/search/?q=film%20portrait')
    assert ep['kind'] == 'listing' and ep['api_path'] == '/search'
    assert ep['params']['q'] == 'film portrait'
    assert 'restrict_sr' not in ep['params']


def test_endpoint_subreddit_search_scoped():
    ep = reddit._endpoint_for('https://www.reddit.com/r/analog/search/?q=portrait&sort=top&t=year')
    assert ep['api_path'] == '/r/analog/search'
    assert ep['params']['restrict_sr'] == 1
    assert ep['params']['q'] == 'portrait' and ep['params']['t'] == 'year'


def test_endpoint_subreddit_listing_default_sort():
    assert reddit._endpoint_for('https://www.reddit.com/r/pics/')['api_path'] == '/r/pics/hot'
    assert reddit._endpoint_for('https://www.reddit.com/r/pics/top/?t=week')['api_path'] == '/r/pics/top'


def test_endpoint_post_and_user_and_direct():
    assert reddit._endpoint_for('https://www.reddit.com/r/pics/comments/xy/z/')['kind'] == 'post'
    assert reddit._endpoint_for('https://www.reddit.com/user/bob')['api_path'] == '/user/bob/submitted'
    d = reddit._endpoint_for('https://i.redd.it/abc.jpeg')
    assert d['kind'] == 'direct' and d['url'].endswith('abc.jpeg')


def test_endpoint_unrecognized_returns_none():
    assert reddit._endpoint_for('https://www.reddit.com/settings') is None


# --- _items_from_post : extraction (pur) ------------------------------------
def test_items_from_gallery_respects_order_and_thumb():
    post = {
        'title': 'shoot', 'subreddit': 'analog', 'is_gallery': True,
        'gallery_data': {'items': [{'media_id': 'B'}, {'media_id': 'A'}]},
        'media_metadata': {
            'A': {'e': 'Image', 's': {'u': 'https://i/A_full.jpg'},
                  'p': [{'u': 'https://i/A_108.jpg', 'x': 108},
                        {'u': 'https://i/A_320.jpg', 'x': 320}]},
            'B': {'e': 'Image', 's': {'u': 'https://i/B_full.jpg'}, 'p': []},
        },
    }
    items = reddit._items_from_post(post)
    assert [i['url'] for i in items] == ['https://i/B_full.jpg', 'https://i/A_full.jpg']
    # A: preview ≥320 chosen as thumbnail ; B: no preview → falls back to full url
    a = next(i for i in items if i['url'].endswith('A_full.jpg'))
    assert a['thumbnail'] == 'https://i/A_320.jpg'
    assert all(i['type'] == 'image' and i['platform'] == 'reddit' for i in items)


def test_items_from_direct_link():
    post = {'title': 't', 'url_overridden_by_dest': 'https://i.redd.it/x.png',
            'preview': {'images': [{'resolutions': [{'url': 'https://p/320.png', 'width': 320}]}]}}
    items = reddit._items_from_post(post)
    assert len(items) == 1 and items[0]['url'] == 'https://i.redd.it/x.png'
    assert items[0]['thumbnail'] == 'https://p/320.png'


def test_items_from_preview_fallback_for_external_link():
    # link post to a non-image URL, but reddit generated a preview → still usable
    post = {'title': 't', 'url': 'https://imgur.com/gallery/xyz',
            'preview': {'images': [{'source': {'url': 'https://p/src.jpg'},
                                    'resolutions': [{'url': 'https://p/320.jpg', 'width': 320}]}]}}
    items = reddit._items_from_post(post)
    assert len(items) == 1 and items[0]['url'] == 'https://p/src.jpg'


def test_items_from_textpost_yields_nothing():
    assert reddit._items_from_post({'title': 't', 'url': 'https://www.reddit.com/r/x/comments/1/'}) == []


# --- _canonical_reddit_url (sans réseau : pas de /s/ ni redd.it) -------------
def test_canonical_purges_tracking_keeps_content_params():
    out = reddit._canonical_reddit_url(
        'https://old.reddit.com/r/analog/top/?t=month&utm_source=share&share_id=abc')
    assert out.startswith('https://www.reddit.com/r/analog/top/')
    assert 't=month' in out and 'utm_source' not in out and 'share_id' not in out


def test_canonical_leaves_cdn_untouched():
    u = 'https://i.redd.it/abc.jpeg'
    assert reddit._canonical_reddit_url(u) == u


# --- RedditSource.match ------------------------------------------------------
def test_match_detects_reddit_hosts_only():
    src = reddit.RedditSource()
    assert src.match('https://www.reddit.com/r/pics/') is not None
    assert src.match('https://redd.it/abc') is not None
    assert src.match('https://example.com/r/pics/') is None
    assert url_validator.detect_platform('https://old.reddit.com/r/x') == Platform.REDDIT


# --- RedditSource.scan (API mockée) -----------------------------------------
def _listing(children, after=None):
    return {'data': {'after': after, 'children': [{'data': d} for d in children]}}


def _img_post(pid):
    return {'title': pid, 'subreddit': 's', 'url_overridden_by_dest': f'https://i.redd.it/{pid}.jpg'}


def test_scan_listing_extracts_and_dedups(monkeypatch):
    monkeypatch.setattr(reddit, '_get_token', lambda: 'tok')
    # duplicate url across two posts → deduped to one item
    dup = _img_post('same')
    monkeypatch.setattr(reddit, '_api_get',
                        lambda path, params, tok: _listing([_img_post('a'), dup, dup]))
    items, err = reddit.RedditSource().scan(Match(url='https://www.reddit.com/r/x/'))
    assert err is None
    assert sorted(i['url'] for i in items) == ['https://i.redd.it/a.jpg', 'https://i.redd.it/same.jpg']


def test_scan_pagination_walks_cursor_no_overlap(monkeypatch):
    monkeypatch.setattr(reddit, '_get_token', lambda: 'tok')
    pages = {
        None: _listing([_img_post('p0')], after='c1'),   # page 0
        'c1': _listing([_img_post('p1')], after='c2'),   # page 1
    }
    calls = []

    def fake_get(path, params, tok):
        calls.append(params.get('after'))
        return pages[params.get('after')]
    monkeypatch.setattr(reddit, '_api_get', fake_get)

    src = reddit.RedditSource()
    m0 = Match(url='https://www.reddit.com/r/x/top/'); m0.page = 0
    m1 = Match(url='https://www.reddit.com/r/x/top/'); m1.page = 1
    i0 = [i['url'] for i in src.scan(m0)[0]]
    i1 = [i['url'] for i in src.scan(m1)[0]]
    assert i0 == ['https://i.redd.it/p0.jpg']
    assert i1 == ['https://i.redd.it/p1.jpg']       # page 1 = next batch, not page 0
    assert set(i0).isdisjoint(i1)


def test_scan_pagination_past_end_returns_empty(monkeypatch):
    monkeypatch.setattr(reddit, '_get_token', lambda: 'tok')
    # only one page exists (after=None) → requesting page 1 yields nothing new
    monkeypatch.setattr(reddit, '_api_get',
                        lambda path, params, tok: _listing([_img_post('only')], after=None))
    m = Match(url='https://www.reddit.com/r/x/'); m.page = 1
    items, err = reddit.RedditSource().scan(m)
    assert err is None and items == []


def test_scan_empty_keyword_is_error(monkeypatch):
    monkeypatch.setattr(reddit, '_get_token', lambda: 'tok')
    items, err = reddit.RedditSource().scan(Match(url='https://www.reddit.com/search/?q='))
    assert items is None and 'mot-cl' in err


def test_scan_direct_image_needs_no_api():
    items, err = reddit.RedditSource().scan(Match(url='https://i.redd.it/z.jpeg'))
    assert err is None and len(items) == 1 and items[0]['url'].endswith('z.jpeg')


def test_scan_unrecognized_url_is_error():
    items, err = reddit.RedditSource().scan(Match(url='https://www.reddit.com/settings'))
    assert items is None and 'non reconnue' in err


def test_scan_token_failure_is_graceful(monkeypatch):
    monkeypatch.setattr(reddit, '_get_token', lambda: None)
    items, err = reddit.RedditSource().scan(Match(url='https://www.reddit.com/r/x/'))
    assert items is None and 'authentification' in err
