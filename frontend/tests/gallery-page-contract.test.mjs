/**
 * 🖼 The Gallery page — the wiring that would fail SILENTLY if it drifted.
 *
 * Two halves. The DOM half mounts the shared lightbox and proves the ‹ ›
 * chevrons are an opt-in: drawn when a handler exists, absent when it is null
 * (the ends of the feed) — the three older hosts pass nothing and must render
 * exactly what they always did. The source half pins the page's addresses:
 * the feed, the delete and the ZIP all live under /api/gallery, and the
 * improve goes through the ONE shared handler with the Gallery's own toast —
 * a page that quietly re-derived any of those would fail only in production.
 */
import assert from 'node:assert/strict'
import test from 'node:test'
import fs from 'node:fs'

import { render } from './support/mountJsx.mjs'

const { default: GeneratedImageLightbox } =
  await import('../src/components/shared/GeneratedImageLightbox.jsx')

const page = fs.readFileSync(
  new URL('../src/pages/GalleryPage.jsx', import.meta.url), 'utf8')
const app = fs.readFileSync(
  new URL('../src/App.jsx', import.meta.url), 'utf8')

const row = { id: 4211, dataset_id: 7, url: '/api/dataset/7/img/x.png', step: 2500 }

// ---- the chevrons are an opt-in, proved in the DOM --------------------------

test('both chevrons render when the host can go both ways', () => {
  const html = render(GeneratedImageLightbox, Object.assign({
    img: row, alt: 'x', onClose: () => {}, onPrev: () => {}, onNext: () => {},
  }))
  assert.match(html, /data-testid="lightbox-prev"/)
  assert.match(html, /data-testid="lightbox-next"/)
  assert.match(html, /aria-label="Previous image"/)
  assert.match(html, /aria-label="Next image"/)
})

test('an END of the feed loses its chevron instead of greying it', () => {
  const html = render(GeneratedImageLightbox, Object.assign({
    img: row, alt: 'x', onClose: () => {}, onPrev: null, onNext: () => {},
  }))
  assert.doesNotMatch(html, /data-testid="lightbox-prev"/)
  assert.match(html, /data-testid="lightbox-next"/)
})

test('a host that passes nothing renders NO navigation — the three old hosts', () => {
  const html = render(GeneratedImageLightbox, Object.assign({
    img: row, alt: 'x', onClose: () => {},
  }))
  assert.doesNotMatch(html, /lightbox-prev|lightbox-next/)
})

// ---- the page's addresses ---------------------------------------------------

test('feed, delete and ZIP all live under /api/gallery — via the shared builders', () => {
  assert.match(page, /galleryFeedUrl\(/)
  assert.match(page, /postJson\('\/api\/gallery\/images\/delete'/)
  // The ZIP goes through the SAME URL builder the checkpoint panel uses, with
  // the app scope — a second builder would be a second answer to "which ids".
  assert.match(page, /galleryZipUrl\(\{ kind: 'app' \}/)
  assert.match(page, /galleryZipPlanUrl\(\{ kind: 'app' \}/)
  // …and no other endpoint is called by hand.
  const called = [...page.matchAll(/(?:postJson|apiFetch)\(\s*(?:`([^`]+)`|'([^']+)')/g)]
    .map((m) => m[1] || m[2]).filter((u) => u.startsWith('/'))
  assert.deepEqual(called, ['/api/gallery/images/delete'])
})

test('improve is the ONE shared handler, gated and re-addressed for this feed', () => {
  assert.match(page, /useCanvasImageImprove\(\{/, 'the shared hook, with options')
  assert.match(page, /launchMessage: galleryImproveLaunchMessage/)
  assert.match(page, /onImprove=\{canImproveCanvasImage\(zoom\) \? improveImage : undefined\}/)
  assert.doesNotMatch(page, /\/improve`/, 'the page restates the improve route')
})

test('the feed loads itself near the end — small sentinel, prefetch margin, cleanup', () => {
  // The sentinel is the load-more ROW: a threshold is a % of the ELEMENT and
  // is unreachable on a tall one, so the small row + rootMargin is the shape
  // that actually fires (a tall-element threshold shipped broken once).
  assert.match(page, /new IntersectionObserver\(/)
  assert.match(page, /rootMargin: '600px 0px'/)
  assert.match(page, /io\.observe\(el\)/)
  assert.match(page, /return \(\) => io\.disconnect\(\)/)
  assert.match(page, /ref=\{loadMoreRef\}/)
  // The button survives as the visible state and the no-observer fallback.
  assert.match(page, /data-testid="gallery-load-more"/)
})

test('the viewer walks the feed and stops AT the ends', () => {
  assert.match(page, /onPrev=\{zoomIndex > 0/)
  assert.match(page, /zoomIndex < images\.length - 1/)
})

// ---- the app shell ----------------------------------------------------------

test('the route exists and the nav gates it like the other generation surfaces', () => {
  assert.match(app, /path="\/gallery" element=\{<GalleryPage \/>\}/)
  assert.match(app,
    /\(caps\.studio_visible \|\| caps\.cloud_training \|\| caps\.training_visible\) && \(\s*<NavLink to="\/gallery"/)
  // Wide like Bank: a grid page reads better on the 1800-px measure.
  assert.match(app, /pathname === '\/gallery'/)
})

// ---- the probe can see it ---------------------------------------------------

test('the responsive probe finds the page by attribute, never by class', () => {
  // The full marker contract (chrome, layers, the probe's own page entry)
  // lives beside the page in galleryProbeMarkers.test.js, like its siblings.
  assert.match(page, /data-probe-chrome="gallery-filters"/)
  assert.match(page, /data-probe-chrome="gallery-bar"/)
})
