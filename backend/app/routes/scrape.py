"""Scrape SCAN + THUMB proxy (read-only) — feeds the concept-dataset builder.

Only two endpoints are lifted from the source app's scrape blueprint, and only
their READ-ONLY parts: `/api/scrape/scan` (URL → list of media items via the
ported sources engine, downloads nothing) and `/api/scrape/thumb` (server-side
fetch of a remote thumbnail the browser can't hotlink). The shared download
pool, quota (ScrapeScanLog) and admin/category gates are dropped — this app is a
single local user. The anti-SSRF guards (`_validate_public_http_url`, no-redirect
fetch, content-type + size caps) are KEPT: the server still fetches arbitrary
user-supplied URLs.

Actually pulling the chosen images INTO a dataset is a separate, autonomous path
(`POST /api/dataset/<id>/scrape-import` in routes/datasets.py → svc.scrape_import_urls).
"""
from urllib.parse import urlparse

from flask import Blueprint, request, jsonify, Response

from ..scrape.netfetch import _validate_public_http_url

bp = Blueprint('scrape', __name__, url_prefix='/api')

MAX_SCAN_PAGE = 50
MAX_THUMB_BYTES = 12 * 1024 * 1024  # 12 MB
_ALLOWED_THUMB_TYPES = {'image/png', 'image/jpeg', 'image/gif', 'image/webp', 'image/avif'}


@bp.post('/scrape/scan')
def scrape_scan():
    """List the downloadable media of a URL via the sources registry (read-only).

    Body: {"url": "...", "page": 0, "include_albums": false}. include_albums
    only matters for gallery-listing sources (PornPics category/tag/search):
    false (default) returns one cover per matched gallery, true dives into every
    photo of each gallery. Returns {scannable, platform, url_type, count, items,
    paginated, page, category} (200), {error, suggestions} (400), or {error}
    (502) on a source-level failure. Downloads nothing."""
    data = request.get_json(silent=True) or {}
    url = data.get('url')
    if not url or not isinstance(url, str):
        return jsonify({'error': 'URL missing.'}), 400
    if len(url) > 2048:
        return jsonify({'error': 'URL too long.'}), 400
    # "Load more" pagination (paginable sources): 0-based, hard-capped (deep pages
    # make gallery-dl re-paginate the whole listing → slow + abuse vector).
    try:
        page = int(data.get('page', 0))
    except (TypeError, ValueError):
        page = 0
    page = max(0, min(page, MAX_SCAN_PAGE))

    from ..scrape.validators import url_validator
    result = url_validator.validate_url(url)
    if not result.is_valid:
        return jsonify({'error': result.error or 'invalid URL',
                        'suggestions': result.suggestions}), 400

    from ..scrape.sources import registry  # local import: avoid an import cycle at load
    match = registry.resolve(url)
    if match is None or match.source is None:
        return jsonify({'error': result.error or 'unsupported URL.',
                        'suggestions': result.suggestions or
                        ['Check the URL is a reachable media page.']}), 400

    match.page = page
    match.include_albums = bool(data.get('include_albums'))
    items, err = match.source.scan(match)
    if err:
        return jsonify({'error': err, 'platform': result.platform.value,
                        'url_type': result.url_type.value}), 502
    # Une source peut être généralement paginable tout en résolvant certaines
    # URLs unitaires. scan() peut alors poser un override sur Match, sans que la
    # route connaisse la plateforme concernée.
    paginated = getattr(match, 'paginated', None)
    if paginated is None:
        paginated = getattr(match.source, 'paginated', False)
    # The requested page is clamped above. Never advertise another page once
    # that effective page reaches the hard limit, even if an upstream API says
    # it has more results.
    if page >= MAX_SCAN_PAGE:
        paginated = False
    return jsonify({
        'scannable': True, 'platform': result.platform.value,
        'url_type': result.url_type.value,
        'count': len(items or []), 'items': items or [],
        'paginated': bool(paginated),
        'page': page,
        'category': getattr(match.source, 'category', 'video'),
    })


@bp.get('/scrape/thumb')
def scrape_thumb():
    """Thumbnail proxy. Source CDNs block direct hotlinking (referer/CORS) so the
    browser <img> fails; fetch server-side (curl_cffi impersonate=chrome + Referer)
    and restream from our origin. SSRF-guarded (public http(s) only, no redirects),
    content-type restricted to raster, size-capped."""
    url = (request.args.get('url') or '').strip()
    ok, err = _validate_public_http_url(url)
    if not ok:
        return jsonify({'error': err or 'invalid URL'}), 400
    try:
        from curl_cffi import requests as cf_requests
    except ImportError:
        return jsonify({'error': 'curl_cffi unavailable'}), 503
    host = urlparse(url).hostname or ''
    try:
        # allow_redirects=False: only the ALREADY-validated host is fetched. A 3xx
        # toward an internal IP would bypass the upstream SSRF guard (TOCTOU/redirect).
        r = cf_requests.get(url, impersonate='chrome', timeout=20, stream=True,
                            allow_redirects=False,
                            headers={'Referer': f'https://{host}/', 'Accept': 'image/*,*/*'})
    except Exception:
        return jsonify({'error': 'fetch failed'}), 502
    if 300 <= r.status_code < 400:
        try: r.close()
        except Exception: pass
        return jsonify({'error': 'redirect refused'}), 502
    ctype = (r.headers.get('content-type') or '').split(';')[0].strip().lower()
    if r.status_code != 200 or ctype not in _ALLOWED_THUMB_TYPES:
        try: r.close()
        except Exception: pass
        return jsonify({'error': 'unsupported type'}), 415
    data = bytearray()
    try:
        for chunk in r.iter_content(8192):
            if not chunk:
                continue
            data += chunk
            if len(data) > MAX_THUMB_BYTES:
                return jsonify({'error': 'thumbnail too large'}), 413
    finally:
        try: r.close()
        except Exception: pass
    # Hardened: no MIME sniffing, inline, locked-down CSP (defense in depth).
    return Response(bytes(data), content_type=ctype, headers={
        'Cache-Control': 'public, max-age=86400',
        'X-Content-Type-Options': 'nosniff',
        'Content-Disposition': 'inline; filename="thumb"',
        'Content-Security-Policy': "default-src 'none'; sandbox",
    })
