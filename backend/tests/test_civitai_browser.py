"""🌐 Civitai prompt browser (Test Studio): accumulation, continuation, caches.

Everything network is faked through the one seam (`_http_get_json`): the suite
must prove the walk/filter/continuation logic, not Civitai's uptime. The shapes
mirror measured 2026-08-27 answers: v1 listing items carry NO meta (prompts
only exist behind `image.getGenerationData`, which 401s without a key).
"""
import json
from urllib.parse import parse_qs, unquote, urlparse

import pytest

from app.services import civitai_browser as cb


@pytest.fixture(autouse=True)
def _fresh_caches():
    cb._gen_cache.clear()
    cb._list_cache.clear()
    yield
    cb._gen_cache.clear()
    cb._list_cache.clear()


def _item(iid, **over):
    d = {'id': iid,
         'url': f'https://image.civitai.com/xx/{iid}-uuid/original=true/{iid}.jpeg',
         'width': 832, 'height': 1216, 'type': 'image', 'nsfwLevel': 'None',
         'username': 'someone',
         'stats': {'likeCount': 10, 'heartCount': 5, 'laughCount': 1,
                   'cryCount': 0, 'commentCount': 2}}
    d.update(over)
    return d


def _gen_json(prompt):
    meta = None
    if prompt is not None:
        meta = {'prompt': prompt, 'negativePrompt': 'bad hands', 'steps': 30,
                'cfgScale': 4.5, 'sampler': 'Euler', 'seed': 123,
                'Model': 'FLUX.1'}
    return {'result': {'data': {'json': {'meta': meta}}}}


class FakeCivitai:
    """Answers the two upstream endpoints from in-memory fixtures and records
    every call, so tests can assert HOW MANY network trips a browse() cost."""

    def __init__(self, pages, gen):
        self.pages = pages    # cursor (None = first) -> listing page json
        self.gen = gen        # image id -> gen json | Exception to raise
        self.calls = []

    def __call__(self, url, key=None):
        self.calls.append((url, key))
        if 'getGenerationData' in url:
            iid = json.loads(unquote(url.split('input=')[1]))['json']['id']
            answer = self.gen[iid]
            if isinstance(answer, Exception):
                raise answer
            return answer
        cursor = parse_qs(urlparse(url).query).get('cursor', [None])[0]
        return self.pages[cursor]

    def listing_calls(self):
        return [c for c in self.calls if 'getGenerationData' not in c[0]]

    def gen_calls(self):
        return [c for c in self.calls if 'getGenerationData' in c[0]]


def _wire(monkeypatch, fake, key='k-123'):
    monkeypatch.setattr(cb, '_http_get_json', fake)
    monkeypatch.setattr(cb, 'civitai_api_key', lambda: key)


# ---------------------------------------------------------------- helpers ----

def test_thumb_url_rewrites_the_transform_segment():
    u = 'https://image.civitai.com/xx/uu-id/original=true/uu-id.jpeg'
    assert cb._thumb_url(u, width=300) == \
        'https://image.civitai.com/xx/uu-id/width=300,anim=false,optimized=true/uu-id.jpeg'


def test_thumb_url_leaves_unexpected_shapes_untouched():
    assert cb._thumb_url('https://example.com/a/b.png') == 'https://example.com/a/b.png'
    assert cb._thumb_url('') == ''


# ------------------------------------------------------------ without key ----

def test_without_key_serves_the_listing_and_never_calls_gen_data(monkeypatch):
    fake = FakeCivitai(
        {None: {'items': [_item(1), _item(2, type='video'), _item(3)],
                'metadata': {'nextCursor': 'c2'}}}, {})
    _wire(monkeypatch, fake, key=None)
    res = cb.browse(want=5)
    assert res['has_key'] is False
    assert [c['id'] for c in res['items']] == [1, 3]          # video excluded
    assert all(c['prompt'] is None for c in res['items'])
    assert res['next_cursor'] == 'c2'
    assert fake.gen_calls() == []                              # no key, no tRPC


def test_without_key_a_partial_window_continues_inside_the_page(monkeypatch):
    fake = FakeCivitai(
        {None: {'items': [_item(1), _item(2), _item(3)],
                'metadata': {'nextCursor': 'c2'}}}, {})
    _wire(monkeypatch, fake, key=None)
    first = cb.browse(want=1)
    assert [c['id'] for c in first['items']] == [1]
    assert first['next_cursor'] is None and first['next_skip'] == 1
    assert first['exhausted'] is False        # None+skip ≠ end of the listing
    again = cb.browse(want=5, cursor=first['next_cursor'],
                      skip=first['next_skip'])
    assert [c['id'] for c in again['items']] == [2, 3]
    assert again['next_cursor'] == 'c2' and again['next_skip'] == 0


# --------------------------------------------------------------- with key ----

def test_accumulates_only_prompt_bearing_cards_and_normalizes(monkeypatch):
    # ids 1..6; even ids published a prompt, odd ones did not.
    fake = FakeCivitai(
        {None: {'items': [_item(i) for i in range(1, 7)],
                'metadata': {'nextCursor': None}}},
        {i: _gen_json(f'prompt {i}' if i % 2 == 0 else None)
         for i in range(1, 7)})
    _wire(monkeypatch, fake)
    res = cb.browse(want=10)
    assert res['has_key'] is True
    assert [c['id'] for c in res['items']] == [2, 4, 6]
    card = res['items'][0]
    assert card['prompt'] == 'prompt 2'
    assert card['negative_prompt'] == 'bad hands'
    assert card['model'] == 'FLUX.1'
    assert card['steps'] == 30 and card['cfg'] == 4.5
    assert card['reactions'] == 16 and card['comments'] == 2   # summed stats
    assert card['page_url'] == 'https://civitai.com/images/2'
    assert 'width=450' in card['thumb_url']
    assert res['next_cursor'] is None and res['exhausted'] is True


def test_require_prompt_off_keeps_promptless_cards(monkeypatch):
    fake = FakeCivitai(
        {None: {'items': [_item(1), _item(2)], 'metadata': {'nextCursor': None}}},
        {1: _gen_json(None), 2: _gen_json('p2')})
    _wire(monkeypatch, fake)
    res = cb.browse(require_prompt=False, want=10)
    assert [c['id'] for c in res['items']] == [1, 2]
    assert res['items'][0]['prompt'] is None
    assert res['items'][1]['prompt'] == 'p2'


def test_walks_to_the_next_page_until_want_is_reached(monkeypatch):
    fake = FakeCivitai(
        {None: {'items': [_item(1), _item(2)], 'metadata': {'nextCursor': 'c2'}},
         'c2': {'items': [_item(3), _item(4)], 'metadata': {'nextCursor': 'c3'}}},
        {1: _gen_json(None), 2: _gen_json('p2'),
         3: _gen_json('p3'), 4: _gen_json('p4')})
    _wire(monkeypatch, fake)
    res = cb.browse(want=2)
    assert [c['id'] for c in res['items']] == [2, 3]
    # Continuation names the page it stopped IN and the first unconsumed slot.
    assert res['next_cursor'] == 'c2' and res['next_skip'] == 1
    assert res['exhausted'] is False
    assert len(fake.listing_calls()) == 2


def test_continuation_resumes_exactly_where_it_stopped(monkeypatch):
    pages = {None: {'items': [_item(i) for i in range(1, 5)],
                    'metadata': {'nextCursor': None}}}
    gen = {i: _gen_json(f'p{i}') for i in range(1, 5)}
    fake = FakeCivitai(pages, gen)
    _wire(monkeypatch, fake)
    first = cb.browse(want=2)
    assert [c['id'] for c in first['items']] == [1, 2]
    again = cb.browse(want=2, cursor=first['next_cursor'],
                      skip=first['next_skip'])
    assert [c['id'] for c in again['items']] == [3, 4]


def test_gen_and_listing_caches_avoid_refetching(monkeypatch):
    fake = FakeCivitai(
        {None: {'items': [_item(1), _item(2)], 'metadata': {'nextCursor': None}}},
        {1: _gen_json('p1'), 2: _gen_json('p2')})
    _wire(monkeypatch, fake)
    cb.browse(want=5)
    listing_before = len(fake.listing_calls())
    gen_before = len(fake.gen_calls())
    cb.browse(want=5)                       # identical browse, straight after
    assert len(fake.listing_calls()) == listing_before   # listing cache hit
    assert len(fake.gen_calls()) == gen_before           # meta cache hit


def test_one_failing_generation_fetch_skips_the_card_not_the_page(monkeypatch):
    fake = FakeCivitai(
        {None: {'items': [_item(1), _item(2)], 'metadata': {'nextCursor': None}}},
        {1: RuntimeError('boom'), 2: _gen_json('p2')})
    _wire(monkeypatch, fake)
    res = cb.browse(want=5)
    assert [c['id'] for c in res['items']] == [2]
    assert 1 not in cb._gen_cache            # a failure stays retryable


def test_rejected_key_stops_the_walk_and_says_so(monkeypatch):
    fake = FakeCivitai(
        {None: {'items': [_item(1), _item(2)], 'metadata': {'nextCursor': None}}},
        {1: PermissionError('401'), 2: PermissionError('401')})
    _wire(monkeypatch, fake)
    res = cb.browse(want=5)
    assert res['key_rejected'] is True
    assert res['items'] == []


def test_budget_bounds_uncached_generation_fetches(monkeypatch):
    n = cb._GEN_BUDGET + 30
    fake = FakeCivitai(
        {None: {'items': [_item(i) for i in range(1, n + 1)],
                'metadata': {'nextCursor': 'c2'}}},
        {i: _gen_json(None) for i in range(1, n + 1)})
    _wire(monkeypatch, fake)
    res = cb.browse(want=cb._WANT_MAX)
    # Nothing matched, but the walk still stopped at the politeness budget
    # (chunk granularity aside) and handed back an exact continuation.
    assert len(fake.gen_calls()) <= cb._GEN_BUDGET + cb._GEN_CHUNK
    assert res['items'] == []
    assert res['next_cursor'] is None and res['next_skip'] > 0


# ----------------------------------------------------------------- params ----

@pytest.mark.parametrize('kwargs', [
    {'period': 'fortnight'}, {'sort': 'spiciest'}, {'level': 'xxl'},
    {'skip': 'abc'}, {'want': 'many'},
])
def test_bad_filter_values_raise_value_error(monkeypatch, kwargs):
    _wire(monkeypatch, FakeCivitai({}, {}))
    with pytest.raises(ValueError):
        cb.browse(**kwargs)


def test_want_and_skip_are_clamped(monkeypatch):
    fake = FakeCivitai(
        {None: {'items': [_item(i) for i in range(1, 4)],
                'metadata': {'nextCursor': None}}},
        {i: _gen_json(f'p{i}') for i in range(1, 4)})
    _wire(monkeypatch, fake)
    res = cb.browse(want=99999, skip=-5)
    assert [c['id'] for c in res['items']] == [1, 2, 3]


# ------------------------------------------------------------------ route ----

def test_route_passes_filters_and_answers_ok(client, monkeypatch):
    seen = {}

    def fake_browse(**kwargs):
        seen.update(kwargs)
        return {'has_key': True, 'key_rejected': False, 'items': [],
                'next_cursor': None, 'next_skip': 0, 'scanned': 0}

    monkeypatch.setattr(cb, 'browse', fake_browse)
    r = client.get('/api/studio/civitai/images?period=month&sort=newest'
                   '&level=x&want=6&require_prompt=0&cursor=abc&skip=4')
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is True and body['has_key'] is True
    assert seen == {'period': 'month', 'sort': 'newest', 'level': 'x',
                    'cursor': 'abc', 'skip': '4', 'want': '6',
                    'require_prompt': False}


def test_route_maps_upstream_failure_to_409(client, monkeypatch):
    def fake_browse(**kwargs):
        raise RuntimeError('Civitai did not answer - check your connection '
                           'and try again.')
    monkeypatch.setattr(cb, 'browse', fake_browse)
    r = client.get('/api/studio/civitai/images')
    assert r.status_code == 409
    assert 'Civitai' in r.get_json()['error']


def test_route_maps_bad_params_to_400(client, monkeypatch):
    _wire(monkeypatch, FakeCivitai({}, {}))
    r = client.get('/api/studio/civitai/images?period=fortnight')
    assert r.status_code == 400
