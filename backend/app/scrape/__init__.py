"""Scrape engine (ported, trimmed): URL → media enumeration.

Only the SCAN side is lifted from the source app — `netfetch` (hardened,
anti-SSRF fetch), `validators` (URL → platform/type), and `sources/` (the
per-platform adapters + registry). The shared download pool, quota models,
describe pipeline and the full Flask blueprint are intentionally NOT ported;
this app drives the engine only to build concept datasets (scan a gallery →
import the chosen images directly). The two HTTP endpoints the concept UI needs
(`/api/scrape/scan`, `/api/scrape/thumb`) live in ``app.routes.scrape``.
"""
