# Documentation website

[`scripts/build_docs_site.py`](../../scripts/build_docs_site.py) turns README.md and the pages listed in [pages.json](pages.json) into the documentation website. [`.github/workflows/docs-site.yml`](../../.github/workflows/docs-site.yml) builds it and publishes it on GitHub Pages whenever README.md or `docs/` change on `v2`.

- GitHub's Markdown API renders each page, so a page reads on the site the way it reads on github.com.
- Relative links between the listed pages become links between site pages; links to other repository files point to GitHub.
- `pages.json` holds the navigation, the page titles and descriptions shown in search results, the software description used for structured data, and the facts listed in `llms.txt`.
- A Markdown page can carry its own search description in a `<!-- description: ... -->` comment under its title.
- The build also writes `sitemap.xml`, `llms.txt` and `404.html`.

Preview locally:

```
python scripts/build_docs_site.py --out _site
python -m http.server --directory _site 8800
```

`--strict` fails the build on a broken internal link or anchor. A `GITHUB_TOKEN` or `GH_TOKEN` raises the API rate limit; anonymous requests are enough for one build.

To add a page, write the Markdown file and list it in a section of `pages.json`.
