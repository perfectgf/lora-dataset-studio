"""Members previews: the strict reader of the signed list and GET /api/member-previews."""
import json
from types import SimpleNamespace

import pytest

from app.plugins.store import member_previews as mp
from app.plugins.store import patreon
from app.plugins.store.client import StoreError

JOIN = 'https://www.patreon.com/c/example'


def entry(**extra):
    return {'id': 'krea_enhancer', 'placement': 'studio.options.krea',
            'name': 'Krea2T prompt enhancer', 'summary': 'One sentence.', **extra}


def listing(*items):
    return {'format': mp.FORMAT, 'items': list(items)}


def member(**patreon_state):
    state = {'linked': False, 'status': 'inactive', 'tier': None, 'valid_until': None,
             'join_url': JOIN, 'request_url': None, 'pending': False, 'reason': None, **patreon_state}
    return lambda client=None: {'status': 'ready' if state['linked'] else 'not_connected', 'patreon': state}


@pytest.fixture(autouse=True)
def _fresh_cache():
    mp.forget()
    yield
    mp.forget()


@pytest.fixture()
def catalog(monkeypatch, tmp_path):
    """The signed catalog.json the session returns, or the error it raises."""
    state = {'document': {'schema_version': 1, 'products': []}, 'error': None, 'reads': 0}

    class Session:
        def __init__(self, config):
            pass

        def __enter__(self):
            state['reads'] += 1
            if state['error']:
                raise state['error']
            return self

        def __exit__(self, *args):
            return False

        def catalog(self):
            return json.loads(json.dumps(state['document']))

    monkeypatch.setattr(mp, 'StoreSession', Session)
    monkeypatch.setattr(mp, 'patreon_core_source', lambda: None)
    monkeypatch.setattr(mp, 'load_patreon_config', lambda: SimpleNamespace(join_url=JOIN))
    return state


def test_a_preview_is_text_and_a_patreon_link_only():
    assert mp.parse_member_previews(listing(entry(), entry(id='studio_remake', placement='studio.options',
                                                           url='https://www.patreon.com/posts/1'))) == (
        entry(), entry(id='studio_remake', placement='studio.options', url='https://www.patreon.com/posts/1'))


@pytest.mark.parametrize('bad', [
    {'format': mp.FORMAT, 'items': [], 'extra': 1},
    {'format': 'lds-member-previews/2', 'items': []},
    {'format': mp.FORMAT, 'items': [entry()] * (mp.MAX_ITEMS + 1)},
    listing(entry(route='/api/x')),
    listing({k: v for k, v in entry().items() if k != 'summary'}),
    listing(entry(id='Bad-Id')),
    listing(entry(name='x' * (mp.MAX_NAME + 1))),
    listing(entry(summary='x' * (mp.MAX_SUMMARY + 1))),
    listing(entry(summary='two\nlines')),
    listing(entry(summary='bidi' + chr(0x202E) + 'text')),
    listing(entry(name='   ')),
    listing(entry(url='http://www.patreon.com/c/x')),
    listing(entry(url='https://patreon.com.example.org/c/x')),
    listing(entry(url='https://www.patreon.com:8443/c/x')),
    listing(entry(url='https://www.patreon.com/c/x#top')),
    listing(entry(url='javascript:alert(1)')),
    listing(entry(), entry()),
    listing(entry(placement='Studio.Options')),
])
def test_any_invalid_entry_refuses_the_whole_list(bad):
    with pytest.raises(StoreError):
        mp.parse_member_previews(bad)


def test_a_place_this_version_does_not_show_is_left_out_alone():
    parsed = mp.parse_member_previews(listing(entry(placement='settings.later'), entry()))
    assert [p['placement'] for p in parsed] == ['studio.options.krea']


def test_without_a_membership_the_signed_list_is_served_with_its_links(client, catalog, monkeypatch):
    monkeypatch.setattr(patreon, 'access_summary', member())
    catalog['document']['member_previews'] = listing(entry(), entry(id='moss_training', placement='training.options',
                                                                    url='https://www.patreon.com/posts/2'))
    response = client.get('/api/member-previews')
    assert response.headers['Cache-Control'] == 'no-store'
    assert response.get_json() == {'format': mp.FORMAT, 'items': [
        {**entry(), 'url': JOIN},
        {**entry(id='moss_training', placement='training.options'), 'url': 'https://www.patreon.com/posts/2'}]}


def test_an_ex_member_sees_the_previews_and_an_active_member_none(client, catalog, monkeypatch):
    catalog['document']['member_previews'] = listing(entry())
    monkeypatch.setattr(patreon, 'access_summary', member(linked=True, status='inactive'))
    assert [p['id'] for p in client.get('/api/member-previews').get_json()['items']] == ['krea_enhancer']
    with client.application.app_context():
        from app.plugins.membership import current_access
        current_access().forget()
    monkeypatch.setattr(patreon, 'access_summary', member(linked=True, status='active', tier='member'))
    assert client.get('/api/member-previews').get_json()['items'] == []


def test_an_unverifiable_membership_shows_nothing(client, catalog, monkeypatch):
    catalog['document']['member_previews'] = listing(entry())
    monkeypatch.setattr(patreon, 'access_summary', lambda client=None: {'status': 'unavailable'})
    assert client.get('/api/member-previews').get_json()['items'] == []


def test_offline_or_without_a_list_nothing_is_shown_and_the_read_waits(client, catalog, monkeypatch):
    monkeypatch.setattr(patreon, 'access_summary', member())
    clock = [1000.0]
    monkeypatch.setattr(mp.time, 'monotonic', lambda: clock[0])
    catalog['error'] = StoreError('offline')
    assert client.get('/api/member-previews').get_json()['items'] == []
    assert client.get('/api/member-previews').get_json()['items'] == []
    assert catalog['reads'] == 1
    catalog['error'] = None
    clock[0] += mp.RETRY_SECONDS + 1
    assert client.get('/api/member-previews').get_json()['items'] == []  # a catalog without the key
    catalog['document']['member_previews'] = listing(entry())
    clock[0] += mp.RETRY_SECONDS + 1
    assert client.get('/api/member-previews').get_json()['items'] == []  # a complete answer is kept 10 min
    clock[0] += mp.TTL_SECONDS
    assert len(client.get('/api/member-previews').get_json()['items']) == 1
    catalog['document']['member_previews'] = {'format': mp.FORMAT, 'items': 'broken'}
    clock[0] += mp.TTL_SECONDS + 1
    assert client.get('/api/member-previews').get_json()['items'] == []
