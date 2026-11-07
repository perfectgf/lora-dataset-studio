#!/usr/bin/env python3
"""Build the documentation website from README.md and docs/.

GitHub's Markdown API renders every page, so the site shows what github.com
shows. This script adds what a website needs around that content: titles and
descriptions, navigation, links rewritten to the site, structured data, a
sitemap and llms.txt. Standard library only.

    python scripts/build_docs_site.py --out _site
    python -m http.server --directory _site 8800

docs/site/pages.json lists the pages with their titles and descriptions. A
GITHUB_TOKEN or GH_TOKEN raises the API rate limit; anonymous requests
(60 per hour) still cover one build.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html
import json
import os
import posixpath
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE_SOURCES = ROOT / 'docs' / 'site'
MARKER = '.lds-docs-site'
MARKDOWN_API = 'https://api.github.com/markdown'
ASSET_SUFFIXES = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.mp4', '.webm'}
ALERT_TITLES = {
    'NOTE': 'Note', 'TIP': 'Tip', 'IMPORTANT': 'Important',
    'WARNING': 'Warning', 'CAUTION': 'Caution',
}

ALERT_RE = re.compile(r'<blockquote>\s*<p>\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*')
BLOCKQUOTE_RE = re.compile(r'<(/?)blockquote\b[^>]*>')
TABLE_RE = re.compile(r'<table\b.*?</table>', re.S)
LINK_ATTR_RE = re.compile(r'\b(href|src)="([^"]*)"')
HREF_RE = re.compile(r'\bhref="([^"]*)"')
H1_RE = re.compile(r'<h1\b[^>]*>(.*?)</h1>', re.S)
H2_RE = re.compile(r'<h2\b[^>]*>(.*?)</h2>\s*<a id="([^"]+)" class="anchor"', re.S)
H3_RE = re.compile(r'<h3\b[^>]*>(.*?)</h3>', re.S)
PARAGRAPH_RE = re.compile(r'<p>(.*?)</p>', re.S)
ID_RE = re.compile(r'\b(?:id|name)="([^"]+)"')
TAG_RE = re.compile(r'<[^>]+>')
IMG_RE = re.compile(r'<img\b[^>]*>')
SRC_RE = re.compile(r'\bsrc="([^"]*)"')
DESCRIPTION_RE = re.compile(r'<!--\s*description:\s*(.*?)\s*-->', re.S)
VIDEO_RE = re.compile(
    r'<p>\s*<a href="(https://github\.com/user-attachments/assets/[^"]+)"[^>]*>[^<]*</a>\s*</p>')
CAMO_RE = re.compile(
    r'(<img\b[^>]*?)\bsrc="https://camo\.githubusercontent\.com/[^"]*"([^>]*?)\s*'
    r'data-canonical-src="([^"]*)"')
HEADING_BLOCK_RE = re.compile(r'<div class="markdown-heading">.*?</div>', re.S)
CUSTOM_ID_RE = re.compile(r'\s*\{#([\w-]+)\}(?=\s*(?:</h[1-6]>|"))')
PLACEHOLDER_RE = re.compile(r'\{\{(\w+)\}\}')


@dataclass
class Page:
    source: str          # repository path of the Markdown file
    path: str            # output path inside the site, e.g. 'guide/installation.html'
    section: str
    parent: str | None   # source of the page shown as the breadcrumb parent
    title: str = ''      # <title> override
    description: str = ''
    nav_label: str = ''
    absolute: bool = False   # links as full URLs (the 404 page is served at any depth)
    heading: str = ''
    body: str = ''
    toc: list[tuple[str, str]] = field(default_factory=list)
    ids: set[str] = field(default_factory=set)
    lastmod: str = ''


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def text_of(fragment: str) -> str:
    return ' '.join(html.unescape(TAG_RE.sub('', fragment)).split())


def shorten(text: str, limit: int = 158) -> str:
    if len(text) <= limit:
        return text
    return text[:limit - 1].rsplit(' ', 1)[0].rstrip(',;:') + '…'


def site_path(source: str) -> str:
    """README.md -> index.html, docs/guide/x.md -> guide/x.html, docs/faq/README.md -> faq/index.html."""
    if source == 'README.md':
        return 'index.html'
    stem = posixpath.splitext(source.removeprefix('docs/'))[0]
    folder, name = posixpath.split(stem)
    if name == 'README':
        return posixpath.join(folder, 'index.html') if folder else 'docs.html'
    return posixpath.join(folder, name.lower().replace('_', '-') + '.html')


def relative(from_path: str, to_path: str) -> str:
    rel = posixpath.relpath(to_path, posixpath.dirname(from_path) or '.')
    if rel == 'index.html':
        return './'
    if rel.endswith('/index.html'):
        return rel[:-len('index.html')]
    return rel


def png_size(path: Path) -> tuple[int, int] | None:
    if path.suffix.lower() != '.png' or not path.is_file():
        return None
    with path.open('rb') as handle:
        head = handle.read(24)
    if len(head) < 24 or head[:8] != b'\x89PNG\r\n\x1a\n':
        return None
    return struct.unpack('>II', head[16:24])


def git_date(source: str) -> str:
    try:
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%cs', '--', source],
            cwd=ROOT, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return ''
    return result.stdout.strip()


def render_markdown(text: str, token: str | None, cache: Path | None) -> str:
    key = hashlib.sha256(text.encode('utf-8')).hexdigest()
    cached = cache / f'{key}.html' if cache else None
    if cached and cached.is_file():
        return cached.read_text(encoding='utf-8')
    payload = json.dumps({'text': text, 'mode': 'markdown'}).encode('utf-8')
    headers = {
        'Accept': 'application/vnd.github+json',
        'Content-Type': 'application/json',
        'User-Agent': 'lora-dataset-studio-docs-site',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    if token:
        headers['Authorization'] = f'Bearer {token}'
    for attempt in range(4):
        request = urllib.request.Request(MARKDOWN_API, data=payload, headers=headers, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                rendered = response.read().decode('utf-8')
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in (403, 429, 500, 502, 503, 504) or attempt == 3:
                raise
        except urllib.error.URLError:
            if attempt == 3:
                raise
        time.sleep(2 * 2 ** attempt)
    if cached:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(rendered, encoding='utf-8')
    return rendered


def convert_alerts(body: str) -> str:
    """GitHub's `> [!NOTE]` alerts arrive as plain blockquotes from the API."""
    parts: list[str] = []
    position = 0
    while match := ALERT_RE.search(body, position):
        depth = 1
        closing = None
        for tag in BLOCKQUOTE_RE.finditer(body, match.end()):
            depth += -1 if tag.group(1) else 1
            if depth == 0:
                closing = tag
                break
        if closing is None:
            break
        kind = match.group(1)
        inner = re.sub(r'<p>\s*</p>', '', '<p>' + body[match.end():closing.start()])
        parts.append(body[position:match.start()])
        parts.append(f'<div class="callout callout-{kind.lower()}">'
                     f'<p class="callout-title">{ALERT_TITLES[kind]}</p>{inner}</div>')
        position = closing.end()
    parts.append(body[position:])
    return ''.join(parts)


def wrap_table(match: re.Match[str]) -> str:
    table = match.group(0)
    kind = 'table-wrap gallery' if '<img' in table else 'table-wrap'
    return f'<div class="{kind}">{table}</div>'


def apply_custom_ids(match: re.Match[str]) -> str:
    """`## Title {#id}` is the in-app renderer's anchor syntax; GitHub prints it as text."""
    block = match.group(0)
    custom = CUSTOM_ID_RE.search(block)
    if not custom:
        return block
    anchor = custom.group(1)
    block = CUSTOM_ID_RE.sub('', block)
    block = re.sub(r'<a id="[^"]*" class="anchor"', f'<a id="{anchor}" class="anchor"', block)
    return re.sub(r'(class="anchor"[^>]*?)href="#[^"]*"', rf'\1href="#{anchor}"', block)


def clean_github_html(body: str) -> str:
    body = re.sub(r'\b(id|name)="user-content-', r'\1="', body)
    body = body.replace('href="#user-content-', 'href="#')
    body = CAMO_RE.sub(r'\1src="\3"\2', body)
    body = HEADING_BLOCK_RE.sub(apply_custom_ids, body)
    body = convert_alerts(body)
    return TABLE_RE.sub(wrap_table, body)


def first_paragraph(body: str) -> str:
    for match in PARAGRAPH_RE.finditer(body):
        text = text_of(match.group(1))
        if len(text) >= 60:
            return shorten(text)
    return ''


def prepare_out_dir(out: Path) -> None:
    if out.exists():
        if any(out.iterdir()) and not (out / MARKER).exists():
            raise SystemExit(f'{out} is not empty and was not made by this script; choose another --out')
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / MARKER).write_text('Generated by scripts/build_docs_site.py\n', encoding='utf-8')


class Site:
    def __init__(self, config: dict, out: Path, token: str | None, cache: Path | None,
                 allow_missing: bool) -> None:
        self.config = config
        self.out = out
        self.token = token
        self.cache = cache
        self.url = config['site_url'].rstrip('/') + '/'
        self.repo = config['repository']
        self.branch = config['branch']
        self.name = config['name']
        self.lang = config.get('language', 'en')
        self.links = config['links']
        self.warnings: list[str] = []
        self.assets: dict[str, str] = {}
        self.pages: dict[str, Page] = {}
        self.sections: list[tuple[str, list[Page]]] = []
        repo = re.escape(self.repo)
        self.repo_link_res = [
            re.compile(rf'https://github\.com/{repo}/(?:blob|tree|raw)/(?:v2|main)/'
                       r'(?P<path>[^?#]*)(?:\?[^#]*)?(?:#(?P<fragment>.*))?$'),
            re.compile(rf'https://raw\.githubusercontent\.com/{repo}/(?:v2|main)/'
                       r'(?P<path>[^?#]*)(?:\?[^#]*)?(?:#(?P<fragment>.*))?$'),
        ]
        for section in config['sections']:
            members = []
            for entry in section['pages']:
                source = entry['source']
                if not (ROOT / source).is_file():
                    if not allow_missing:
                        raise SystemExit(f'missing page source: {source}')
                    self.warnings.append(f'skipped missing page source {source}')
                    continue
                page = Page(source=source, path=entry.get('path') or site_path(source),
                            section=section['title'], parent=section.get('parent'),
                            title=entry.get('title', ''), description=entry.get('description', ''),
                            nav_label=entry.get('nav_label', ''))
                self.pages[source] = page
                members.append(page)
            self.sections.append((section['title'], members))

    # Links ---------------------------------------------------------------

    def canonical(self, page: Page) -> str:
        return self.url + page.path.removesuffix('index.html')

    def href(self, page: Page, target: str) -> str:
        if page.absolute:
            return self.url + target.removesuffix('index.html')
        return relative(page.path, target)

    def prefix(self, page: Page) -> str:
        return self.url if page.absolute else '../' * page.path.count('/')

    def repo_target(self, url: str, page: Page) -> tuple[str, str] | None:
        """(repository path, fragment) when `url` points into this repository."""
        url = html.unescape(url)
        if not url or url.startswith(('#', 'mailto:', 'data:', 'javascript:')):
            return None
        parts = urlsplit(url)
        if parts.scheme or parts.netloc:
            for pattern in self.repo_link_res:
                match = pattern.match(url)
                if match:
                    return unquote(match.group('path')).strip('/') or '.', match.group('fragment') or ''
            return None
        if not parts.path or parts.path.startswith('/'):
            return None
        path = posixpath.normpath(posixpath.join(posixpath.dirname(page.source), unquote(parts.path)))
        if path == '..' or path.startswith('../'):
            return None
        return path, parts.fragment

    def link_for(self, page: Page, path: str, fragment: str) -> str:
        suffix = f'#{fragment}' if fragment else ''
        target = self.pages.get(path)
        if target is not None:
            if target is page and fragment:
                return suffix
            return self.href(page, target.path) + suffix
        full = ROOT / path
        if full.suffix.lower() in ASSET_SUFFIXES and full.is_file():
            return self.href(page, self.asset(path)) + suffix
        if path == '.':
            return self.links['github']
        if not full.exists():
            self.warnings.append(f'{page.source}: link to missing file {path}')
        kind = 'tree' if full.is_dir() else 'blob'
        return f'https://github.com/{self.repo}/{kind}/{self.branch}/{quote(path)}{suffix}'

    def asset(self, path: str) -> str:
        return self.assets.setdefault(path, path.removeprefix('docs/'))

    def rewrite_links(self, page: Page, body: str) -> str:
        def visit(match: re.Match[str]) -> str:
            target = self.repo_target(match.group(2), page)
            if target is None:
                return match.group(0)
            return f'{match.group(1)}="{esc(self.link_for(page, *target))}"'
        return LINK_ATTR_RE.sub(visit, body)

    def size_images(self, page: Page, body: str) -> str:
        """Width and height on local PNGs (no layout shift), lazy loading after the first image."""
        count = 0

        def visit(match: re.Match[str]) -> str:
            nonlocal count
            count += 1
            tag = match.group(0)
            extra = []
            if 'width=' not in tag:
                src = SRC_RE.search(tag)
                target = self.repo_target(src.group(1), page) if src else None
                size = png_size(ROOT / target[0]) if target else None
                if size:
                    extra.append(f'width="{size[0]}" height="{size[1]}"')
            if count > 3 and 'loading=' not in tag:
                extra.append('loading="lazy" decoding="async"')
            if not extra:
                return tag
            return tag[:-1].rstrip(' /') + ' ' + ' '.join(extra) + '>'
        return IMG_RE.sub(visit, body)

    def check_anchors(self, pages: list[Page]) -> None:
        by_path = {page.path: page for page in pages}
        for page in pages:
            for raw in HREF_RE.findall(page.body):
                parts = urlsplit(html.unescape(raw))
                if parts.scheme or parts.netloc or not parts.fragment:
                    continue
                target: Page | None = page
                if parts.path:
                    path = posixpath.normpath(posixpath.join(posixpath.dirname(page.path), parts.path))
                    if parts.path.endswith('/') or path == '.':
                        path = 'index.html' if path == '.' else f'{path}/index.html'
                    target = by_path.get(path)
                fragment = unquote(parts.fragment)
                if target is not None and fragment not in target.ids:
                    self.warnings.append(f'{page.source}: no anchor #{fragment} in {target.source}')

    # Rendering -----------------------------------------------------------

    def render(self, page: Page) -> str:
        return render_markdown((ROOT / page.source).read_text(encoding='utf-8'), self.token, self.cache)

    def video_link(self, page: Page, match: re.Match[str]) -> str:
        """Uploaded videos only play on github.com, so link to the page that shows them."""
        if page.source == 'README.md':
            url = self.links['github'] + '#readme'
        else:
            url = f'https://github.com/{self.repo}/blob/{self.branch}/{quote(page.source)}'
        return f'<p><a class="video-link" href="{esc(url)}">▶ Watch the demo video on GitHub</a></p>'

    def prepare(self, page: Page, rendered: str) -> None:
        body = VIDEO_RE.sub(lambda match: self.video_link(page, match), clean_github_html(rendered))
        body = self.size_images(page, body)
        heading = H1_RE.search(body)
        page.heading = text_of(heading.group(1)) if heading else page.title or page.source
        if not page.description:
            comment = DESCRIPTION_RE.search((ROOT / page.source).read_text(encoding='utf-8'))
            page.description = ' '.join(comment.group(1).split()) if comment else first_paragraph(body)
        page.toc = [(text_of(title), anchor) for title, anchor in H2_RE.findall(body)]
        page.ids = set(ID_RE.findall(body))
        page.lastmod = git_date(page.source)
        page.body = body

    def faq_entries(self, page: Page) -> list[tuple[str, str]]:
        """Question/answer pairs of the README FAQ, for the home page's FAQPage data."""
        anchor = self.config.get('home_faq_anchor')
        start = page.body.find(f'id="{anchor}"') if anchor else -1
        if start < 0:
            return []
        end = page.body.find('<h2', start)
        section = page.body[start:end if end > 0 else len(page.body)]
        entries = []
        for match in H3_RE.finditer(section):
            answer = PARAGRAPH_RE.search(section, match.end())
            if answer:
                entries.append((text_of(match.group(1)), text_of(answer.group(1))))
        return entries

    def jsonld(self, page: Page) -> str:
        publisher_id = self.url + '#publisher'
        graph: list[dict] = [{
            '@type': 'Organization', '@id': publisher_id, 'name': self.name,
            'url': self.url, 'logo': self.url + 'icon.png',
        }]
        canonical = self.canonical(page)
        if page.path == 'index.html':
            software = self.config['software']
            graph.append({
                '@type': 'WebSite', '@id': self.url + '#website', 'url': self.url,
                'name': self.name, 'alternateName': self.config.get('alternate_name'),
                'inLanguage': self.lang, 'publisher': {'@id': publisher_id},
            })
            graph.append({
                '@type': 'SoftwareApplication', '@id': self.url + '#software', 'name': self.name,
                'alternateName': self.config.get('alternate_name'),
                'description': software['description'],
                'applicationCategory': software['category'],
                'operatingSystem': software['operating_system'],
                'url': self.url, 'downloadUrl': self.links['download'],
                'license': software['license'], 'isAccessibleForFree': True,
                'offers': {'@type': 'Offer', 'price': '0', 'priceCurrency': 'USD'},
                'screenshot': [self.url + self.asset(path) for path in software['screenshots']],
                'sameAs': [self.links['github']], 'publisher': {'@id': publisher_id},
            })
            faq = self.faq_entries(page)
            if faq:
                graph.append({
                    '@type': 'FAQPage', '@id': canonical + '#faq',
                    'mainEntity': [{
                        '@type': 'Question', 'name': question,
                        'acceptedAnswer': {'@type': 'Answer', 'text': answer},
                    } for question, answer in faq],
                })
        elif not page.absolute:
            article = {
                '@type': 'TechArticle', 'headline': page.heading, 'description': page.description,
                'url': canonical, 'mainEntityOfPage': canonical, 'inLanguage': self.lang,
                'isPartOf': {'@id': self.url + '#website'}, 'about': {'@id': self.url + '#software'},
                'publisher': {'@id': publisher_id},
            }
            if page.lastmod:
                article['dateModified'] = page.lastmod
            graph.append(article)
            trail = [(self.name, self.url)]
            parent = self.pages.get(page.parent or '')
            if parent is not None and parent is not page:
                trail.append((parent.nav_label or parent.heading, self.canonical(parent)))
            trail.append((page.heading, canonical))
            graph.append({
                '@type': 'BreadcrumbList',
                'itemListElement': [{'@type': 'ListItem', 'position': index, 'name': name, 'item': url}
                                    for index, (name, url) in enumerate(trail, start=1)],
            })
        data = {'@context': 'https://schema.org', '@graph': graph}
        return json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')

    def header_links(self, page: Page) -> list[str]:
        links = []
        for item in self.config['header']:
            current = ''
            if 'source' in item:
                target = self.pages.get(item['source'])
                if target is None:
                    continue
                href = self.href(page, target.path)
                current = ' aria-current="page"' if target is page else ''
            else:
                href = self.links[item['link']]
            css = ' class="button"' if item.get('primary') else ''
            links.append(f'<a href="{esc(href)}"{css}{current}>{esc(item["label"])}</a>')
        return links

    def sidebar_html(self, page: Page) -> str:
        parts = []
        for title, members in self.sections:
            items = []
            for member in members:
                current = ' aria-current="page"' if member is page else ''
                label = esc(member.nav_label or member.heading)
                items.append(f'<li><a href="{esc(self.href(page, member.path))}"{current}>{label}</a></li>')
            parts.append(f'<p class="nav-title">{esc(title)}</p><ul class="nav-list">{"".join(items)}</ul>')
        return ''.join(parts)

    def breadcrumbs_html(self, page: Page) -> str:
        if page.path == 'index.html' or page.absolute:
            return ''
        crumbs = [f'<a href="{esc(self.href(page, "index.html"))}">Home</a>']
        parent = self.pages.get(page.parent or '')
        if parent is not None and parent is not page:
            crumbs.append(f'<a href="{esc(self.href(page, parent.path))}">'
                          f'{esc(parent.nav_label or parent.heading)}</a>')
        crumbs.append(f'<span aria-current="page">{esc(page.nav_label or page.heading)}</span>')
        return f'<nav class="breadcrumbs" aria-label="Breadcrumb">{" › ".join(crumbs)}</nav>'

    def toc_html(self, page: Page) -> str:
        if len(page.toc) < 3:
            return ''
        items = ''.join(f'<li><a href="#{esc(anchor)}">{esc(title)}</a></li>' for title, anchor in page.toc)
        return f'<details class="toc"><summary>On this page</summary><ol>{items}</ol></details>'

    def page_html(self, page: Page, template: str, stylesheet: str) -> str:
        prefix = self.prefix(page)
        toc = self.toc_html(page)
        meta = ''
        if page.source:
            updated = f'Updated {page.lastmod} · ' if page.lastmod else ''
            source_url = f'https://github.com/{self.repo}/blob/{self.branch}/{quote(page.source)}'
            meta = f'<p class="page-meta">{updated}<a href="{esc(source_url)}">View this page on GitHub</a></p>'
        verification = ''
        robots = ''
        if page.path == 'index.html':
            for key, name in (('google_site_verification', 'google-site-verification'),
                              ('bing_site_verification', 'msvalidate.01')):
                if self.config.get(key):
                    verification += f'<meta name="{name}" content="{esc(self.config[key])}">\n'
        if page.absolute:
            robots = '<meta name="robots" content="noindex">\n'
        if page.title:
            title = page.title
        elif self.name in page.heading or len(page.heading) + len(self.name) > 67:
            title = page.heading   # a long question reads better whole than cut off by the site name
        else:
            title = f'{page.heading} · {self.name}'
        links = self.header_links(page)
        license_url = f'https://github.com/{self.repo}/blob/{self.branch}/LICENSE'
        footer = (f'<p><a href="{esc(self.href(page, "index.html"))}">{esc(self.name)}</a>'
                  f' · <a href="{esc(license_url)}">PolyForm Noncommercial 1.0.0</a>'
                  f' · <a href="{esc(self.links["github"])}">GitHub</a>'
                  f' · <a href="{esc(self.links["discord"])}">Discord</a>'
                  f' · <a href="{esc(prefix)}llms.txt">llms.txt</a></p>')
        values = {
            'lang': self.lang,
            'title': esc(title),
            'description': esc(page.description),
            'canonical': esc(self.canonical(page)),
            'robots': robots,
            'verification': verification,
            'og_type': 'website' if page.path == 'index.html' else 'article',
            'og_title': esc(title),
            'og_image': esc(self.url + self.asset(self.config['og_image'])),
            'site_name': esc(self.name),
            'prefix': esc(prefix),
            'stylesheet': esc(f'{prefix}style.css?v={stylesheet}'),
            'home': esc(self.href(page, 'index.html')),
            'header_nav': ''.join(links),
            'menu': ('<ul class="menu-links">' + ''.join(f'<li>{link}</li>' for link in links)
                     + '</ul>' + self.sidebar_html(page)),
            'sidebar': self.sidebar_html(page),
            'layout_class': '' if toc else ' no-toc',
            'toc': toc,
            'breadcrumbs': self.breadcrumbs_html(page),
            'content': page.body,
            'page_meta': meta,
            'footer': footer,
            'jsonld': self.jsonld(page),
        }
        return PLACEHOLDER_RE.sub(lambda match: values[match.group(1)], template)

    # Output --------------------------------------------------------------

    def not_found_page(self) -> Page:
        page = Page(source='', path='404.html', section='', parent=None, heading='Page not found',
                    description='This page does not exist on the LoRA Dataset Studio website.',
                    absolute=True)
        links = ' · '.join(
            f'<a href="{esc(self.href(page, target.path))}">{esc(target.nav_label or target.heading)}</a>'
            for target in (self.pages.get(source) for source in ('README.md', 'docs/README.md', 'docs/faq/README.md'))
            if target is not None)
        page.body = ('<div class="markdown-heading"><h1 class="heading-element">Page not found</h1></div>'
                     f'<p>This page does not exist or has moved.</p><p>{links}</p>')
        return page

    def llms_txt(self) -> str:
        lines = [f'# {self.name}', '', f'> {self.config["llms_summary"]}', '']
        lines += [f'- {fact}' for fact in self.config.get('llms_facts', [])]
        for title, members in self.sections:
            lines += ['', f'## {title}', '']
            lines += [f'- [{member.heading}]({self.canonical(member)}): {member.description}' for member in members]
        return '\n'.join(lines) + '\n'

    def sitemap(self, pages: list[Page]) -> str:
        lines = ['<?xml version="1.0" encoding="UTF-8"?>',
                 '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
        for page in pages:
            lastmod = f'<lastmod>{page.lastmod}</lastmod>' if page.lastmod else ''
            lines.append(f'  <url><loc>{esc(self.canonical(page))}</loc>{lastmod}</url>')
        lines.append('</urlset>')
        return '\n'.join(lines) + '\n'

    def build(self) -> None:
        pages = list(self.pages.values())
        prepare_out_dir(self.out)
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            rendered = list(pool.map(self.render, pages))
        for page, html_text in zip(pages, rendered):
            self.prepare(page, html_text)
        for page in pages:
            page.body = self.rewrite_links(page, page.body)
        self.check_anchors(pages)

        template =(SITE_SOURCES / 'template.html').read_text(encoding='utf-8')
        css = (SITE_SOURCES / 'style.css').read_bytes()
        (self.out / 'style.css').write_bytes(css)
        stylesheet = hashlib.sha256(css).hexdigest()[:10]
        shutil.copyfile(ROOT / self.config['logo'], self.out / 'icon.png')
        for page in pages + [self.not_found_page()]:
            target = self.out / page.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self.page_html(page, template, stylesheet), encoding='utf-8', newline='\n')
        for source, path in sorted(self.assets.items()):
            target = self.out / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / source, target)
        (self.out / 'sitemap.xml').write_text(self.sitemap(pages), encoding='utf-8', newline='\n')
        (self.out / 'llms.txt').write_text(self.llms_txt(), encoding='utf-8', newline='\n')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Build the LoRA Dataset Studio documentation website.')
    parser.add_argument('--out', type=Path, required=True, help='output folder (replaced on each build)')
    parser.add_argument('--cache', type=Path, help='keep rendered pages here between local builds')
    parser.add_argument('--allow-missing', action='store_true', help='skip listed pages that do not exist yet')
    parser.add_argument('--strict', action='store_true', help='fail on broken internal links or anchors')
    args = parser.parse_args(argv)
    config = json.loads((SITE_SOURCES / 'pages.json').read_text(encoding='utf-8'))
    token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN') or None
    site = Site(config, args.out.resolve(), token, args.cache, args.allow_missing)
    site.build()
    for warning in site.warnings:
        print(f'warning: {warning}', file=sys.stderr)
    print(f'{len(site.pages)} pages and {len(site.assets)} assets written to {args.out}')
    return 1 if args.strict and site.warnings else 0


if __name__ == '__main__':
    sys.exit(main())
