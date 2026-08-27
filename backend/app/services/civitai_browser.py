# app/services/civitai_browser.py
"""🌐 Civitai prompt browser — top images WITH their generation prompts, for the
Test Studio / Canvas prompt field.

Two upstream endpoints, because Civitai split them (measured 2026-08-27):

- the public REST listing (``/api/v1/images``) paginates images with stats and
  nsfw levels, but its ``meta`` field is ALWAYS null now — even authenticated;
- the site's own ``image.getGenerationData`` (tRPC) returns the prompt and
  settings per image, and answers 401 without an API key.

So browsing works without a key, but reading prompts — the whole point — needs
the (free-account) Civitai API key. It is resolved through the SAME chain as the
scraper (env ``CIVITAI_API_KEY`` > admin cookies dir > legacy token file) so the
app keeps ONE Civitai credential, not a second competing setting.

Not every image publishes its prompt (measured: ~60% of a top-week page do), and
the listing cannot filter on that — so `browse()` walks listing pages and calls
getGenerationData per image until it has gathered `want` prompt-bearing cards.
Politeness bounds every call: one listing page fetch at a time (cached 5 min),
a per-request budget of uncached generation-data fetches (small worker pool),
and a per-image meta cache so “Load more” and re-opens never re-ask Civitai for
an image already seen. Continuation is exact even when a page is left half
scanned: the response's ``next_cursor``/``next_skip`` name the first listing
item NOT yet consumed, and the caches make the re-walk to that point free.
"""
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote, urlencode

import requests

logger = logging.getLogger(__name__)

_LIST_URL = 'https://civitai.com/api/v1/images'
_GEN_URL = 'https://civitai.com/api/trpc/image.getGenerationData'
_UA = 'lora-dataset-studio/1.0 (Test Studio prompt browser)'
_TIMEOUT = (6.1, 20)

# Wire values are lowercase/stable; Civitai's spellings stay an implementation
# detail so a UI select can never drift from the API's casing.
PERIODS = {'day': 'Day', 'week': 'Week', 'month': 'Month',
           'year': 'Year', 'alltime': 'AllTime'}
SORTS = {'reactions': 'Most Reactions', 'newest': 'Newest'}
# Browsing ceiling, mildest→raciest. The v1 `nsfw` param acts as a ceiling, not
# an exact match (measured: nsfw=X returns a None..X mix); 'none' omits the
# param, which the API answers with safe-only.
LEVELS = ('none', 'soft', 'mature', 'x')

_PAGE_SIZE = 100        # one v1 listing page per fetch (API max 200)
_WANT_MAX = 24          # cards per browse() answer
_GEN_BUDGET = 40        # UNCACHED getGenerationData calls per browse()
_GEN_CHUNK = 12         # parallel fetch chunk — keeps skip bookkeeping exact
_GEN_WORKERS = 6
_MAX_LIST_ITEMS = 300   # listing items walked per browse(), budget aside
_THUMB_WIDTH = 450

_GEN_TTL = 24 * 3600
_GEN_CACHE_MAX = 4000
_LIST_TTL = 300
# id -> (monotonic, meta dict|None); key -> (monotonic, listing json)
_gen_cache = {}
_list_cache = {}
_cache_lock = threading.Lock()


def civitai_api_key():
    """The one Civitai credential, through the scraper's resolver when the
    scrape extras are importable, else the stored secret directly. Same
    defensive shape as ``setup_installer._civitai_key`` and for the same
    reason: the resolver lives in a module whose siblings need optional
    packages, and a missing extra must not take this browser down."""
    try:
        from ..scrape.sources.civitai import civitai_api_key as resolver
        key = resolver()
        if key:
            return key
    except Exception:
        logger.debug('scrape civitai_api_key() unavailable - using stored secret',
                     exc_info=True)
    from .. import config as cfg
    return cfg.secret('CIVITAI_API_KEY') or None


def _http_get_json(url, key=None):
    """GET → parsed JSON. The single network seam (tests monkeypatch it).
    Auth/network/HTTP failures raise RuntimeError with a user-facing sentence;
    a 401 raises PermissionError so callers can tell 'key refused' apart."""
    headers = {'User-Agent': _UA}
    if key:
        headers['Authorization'] = f'Bearer {key}'
    try:
        resp = requests.get(url, headers=headers, timeout=_TIMEOUT)
    except requests.RequestException as e:
        raise RuntimeError('Civitai did not answer - check your connection '
                           'and try again.') from e
    if resp.status_code == 401:
        raise PermissionError('Civitai refused the API key.')
    if resp.status_code != 200:
        raise RuntimeError(f'Civitai answered HTTP {resp.status_code} - '
                           'try again in a moment.')
    try:
        return resp.json()
    except ValueError as e:
        raise RuntimeError('Civitai answered something that is not JSON - '
                           'try again in a moment.') from e


def _cache_get(cache, key, ttl):
    with _cache_lock:
        hit = cache.get(key)
        if hit and time.monotonic() - hit[0] < ttl:
            return hit[1], True
    return None, False


def _cache_put(cache, key, value, cap):
    with _cache_lock:
        if len(cache) >= cap:
            # Oldest-first trim; plain dicts iterate in insertion order.
            for stale in list(cache)[:max(1, cap // 10)]:
                cache.pop(stale, None)
        cache[key] = (time.monotonic(), value)


def _listing_page(period, sort, level, cursor, key):
    """One v1 listing page (cached _LIST_TTL). The API key is sent when
    present — harmless for the listing, and consistent with the site."""
    cache_key = (period, sort, level, cursor)
    page, ok = _cache_get(_list_cache, cache_key, _LIST_TTL)
    if ok:
        return page
    params = {'limit': _PAGE_SIZE, 'sort': SORTS[sort], 'period': PERIODS[period]}
    if level != 'none':
        params['nsfw'] = 'X' if level == 'x' else level.capitalize()
    if cursor:
        params['cursor'] = cursor
    try:
        page = _http_get_json(f'{_LIST_URL}?{urlencode(params)}', key=key)
    except PermissionError:
        # A refused key on the LISTING (it is sent there too) gets the same
        # actionable sentence as everywhere else, as a 409 — not a bare 500.
        raise RuntimeError('Civitai refused the API key - check it in '
                           'Settings > Scraping & sources.')
    if not isinstance(page, dict) or not isinstance(page.get('items'), list):
        raise RuntimeError('Civitai answered an unexpected listing shape - '
                           'try again in a moment.')
    _cache_put(_list_cache, cache_key, page, cap=64)
    return page


def _generation_meta(image_id, key):
    """The generation meta of one image (cached _GEN_TTL), or None when the
    poster published none. Returns (meta|None, from_cache, key_rejected)."""
    meta, ok = _cache_get(_gen_cache, image_id, _GEN_TTL)
    if ok:
        return meta, True, False
    payload = quote(json.dumps({'json': {'id': image_id}}))
    try:
        data = _http_get_json(f'{_GEN_URL}?input={payload}', key=key)
    except PermissionError:
        return None, False, True
    except RuntimeError:
        # One image failing to answer must not sink the whole page — the card
        # is simply skipped this round (NOT cached: it may answer next time).
        logger.debug('civitai generation data failed for %s', image_id,
                     exc_info=True)
        return None, False, False
    result = ((data.get('result') or {}).get('data') or {}).get('json') or {}
    meta = result.get('meta')
    if not isinstance(meta, dict) or not (meta.get('prompt') or '').strip():
        meta = None
    _cache_put(_gen_cache, image_id, meta, cap=_GEN_CACHE_MAX)
    return meta, False, False


def _thumb_url(url, width=_THUMB_WIDTH):
    """A light webp thumbnail from the CDN's transform segment (the segment
    holding `=`, e.g. ``original=true`` → ``width=450,…``). An unexpected URL
    shape is returned untouched — the full image still renders."""
    try:
        parts = url.split('/')
        for i, seg in enumerate(parts):
            if '=' in seg:
                parts[i] = f'width={width},anim=false,optimized=true'
                return '/'.join(parts)
    except AttributeError:
        pass
    return url


def _card(item, meta):
    """One UI card: the image next to its prompt — exactly the pairing the
    feature exists for. `meta` may be None (browsing without a key, or the
    'show promptless too' toggle)."""
    meta = meta or {}
    stats = item.get('stats') or {}
    card = {
        'id': item.get('id'),
        'page_url': f"https://civitai.com/images/{item.get('id')}",
        'image_url': item.get('url'),
        'thumb_url': _thumb_url(item.get('url') or ''),
        'width': item.get('width'),
        'height': item.get('height'),
        'nsfw_level': item.get('nsfwLevel'),
        'username': item.get('username'),
        'reactions': sum(stats.get(k) or 0 for k in
                         ('likeCount', 'heartCount', 'laughCount', 'cryCount')),
        'comments': stats.get('commentCount') or 0,
        'prompt': (meta.get('prompt') or '').strip() or None,
        'negative_prompt': (meta.get('negativePrompt') or '').strip() or None,
        'model': meta.get('Model') or meta.get('model') or None,
        'steps': meta.get('steps'),
        'cfg': meta.get('cfgScale'),
        'sampler': meta.get('sampler'),
        'seed': meta.get('seed'),
    }
    return card


def _coerce_int(value, name, default, lo, hi):
    if value is None or value == '':
        return default
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be an integer')
    return max(lo, min(hi, n))


def browse(period='week', sort='reactions', level='none', cursor=None,
           skip=0, want=12, require_prompt=True):
    """Gather up to `want` cards starting at (`cursor`, `skip`).

    With a key and `require_prompt`, only prompt-bearing images become cards —
    the accumulation walks the listing until `want` are found or a politeness
    bound is hit, and the answer carries `next_cursor`/`next_skip` to continue
    exactly where it stopped. Without a key the listing is served as-is
    (prompts null) so the browser still shows the top images, and the UI can
    explain what the key unlocks.
    """
    period = str(period or 'week').lower()
    sort = str(sort or 'reactions').lower()
    level = str(level or 'none').lower()
    if period not in PERIODS:
        raise ValueError(f"period must be one of {', '.join(PERIODS)}")
    if sort not in SORTS:
        raise ValueError(f"sort must be one of {', '.join(SORTS)}")
    if level not in LEVELS:
        raise ValueError(f"level must be one of {', '.join(LEVELS)}")
    skip = _coerce_int(skip, 'skip', 0, 0, _PAGE_SIZE)
    want = _coerce_int(want, 'want', 12, 1, _WANT_MAX)
    cursor = str(cursor) if cursor else None

    key = civitai_api_key()
    if not key:
        # No prompts to hunt for → serve the listing window as-is, with the
        # SAME (cursor, skip) continuation as the keyed walk: jumping straight
        # to the next page's cursor would silently skip the rest of this one.
        page = _listing_page(period, sort, level, cursor, key=None)
        entries = [i for i in page['items'] if i.get('type') == 'image']
        served = entries[skip:skip + want]
        consumed = skip + len(served)
        page_next = (page.get('metadata') or {}).get('nextCursor')
        if consumed < len(entries):
            nxt, nskip = cursor, consumed
        else:
            nxt, nskip = page_next, 0
        return {'has_key': False, 'key_rejected': False,
                'items': [_card(i, None) for i in served],
                'next_cursor': nxt, 'next_skip': nskip,
                'exhausted': nxt is None and nskip == 0,
                'scanned': len(served)}

    cards = []
    gen_budget = _GEN_BUDGET
    scanned = 0
    key_rejected = False
    next_cursor, next_skip = cursor, skip
    pool = ThreadPoolExecutor(_GEN_WORKERS)
    try:
        while len(cards) < want and scanned < _MAX_LIST_ITEMS:
            page = _listing_page(period, sort, level, next_cursor, key)
            entries = [i for i in page['items'] if i.get('type') == 'image']
            page_next = (page.get('metadata') or {}).get('nextCursor')
            idx = next_skip
            while idx < len(entries):
                chunk = entries[idx:idx + _GEN_CHUNK]
                results = list(pool.map(
                    lambda it: _generation_meta(it['id'], key), chunk))
                for offset, (meta, from_cache, rejected) in enumerate(results):
                    item = chunk[offset]
                    scanned += 1
                    if rejected:
                        key_rejected = True
                    if not from_cache and not rejected:
                        gen_budget -= 1
                    if meta or not require_prompt:
                        cards.append(_card(item, meta))
                    next_skip = idx + offset + 1
                    if (len(cards) >= want or gen_budget <= 0
                            or scanned >= _MAX_LIST_ITEMS or key_rejected):
                        return {'has_key': True, 'key_rejected': key_rejected,
                                'items': cards, 'next_cursor': next_cursor,
                                'next_skip': next_skip, 'exhausted': False,
                                'scanned': scanned}
                idx += len(chunk)
            if not page_next:
                # Listing exhausted — nothing left to continue into.
                return {'has_key': True, 'key_rejected': key_rejected,
                        'items': cards, 'next_cursor': None, 'next_skip': 0,
                        'exhausted': True, 'scanned': scanned}
            next_cursor, next_skip = page_next, 0
        return {'has_key': True, 'key_rejected': key_rejected, 'items': cards,
                'next_cursor': next_cursor, 'next_skip': next_skip,
                'exhausted': False, 'scanned': scanned}
    finally:
        pool.shutdown(wait=False)
