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
  // …and no other endpoint is called by hand. The queue reading is the second
  // and last one allowed here: it is not a gallery address at all but the
  // app-wide job listing the dock already renders — the feed watches it to know
  // when something finished, and inventing a gallery-flavoured alias for it
  // would be a second name for one truth.
  const called = [...page.matchAll(/(?:postJson|apiFetch)\(\s*(?:`([^`]+)`|'([^']+)')/g)]
    .map((m) => m[1] || m[2]).filter((u) => u.startsWith('/'))
  assert.deepEqual(called, ['/api/system/queue', '/api/gallery/images/delete'])
})

test('the feed refreshes ITSELF when the queue finishes something', () => {
  // Reported: generate with the Gallery open, and the image only appears after
  // a manual page reload. The dataset grid never had that problem because it
  // polls while its own tiles are pending; this feed has no pending tiles, so
  // it watches the queue every surface shares instead.
  assert.match(page, /liveQueueIds\(/)
  assert.match(page, /queueDrained\(/)
  assert.match(page, /mergeGalleryHead\(/,
    'a head read must MERGE — a plain reload would throw away every page the '
    + 'reader has scrolled into')
  // The quiet half, which is what makes an automatic read acceptable at all:
  // no loading state, and the cursor into the tail is left alone.
  // Read the CODE, not the prose: the block explains at length why the cursor
  // is left alone, and a naive scan would fail on its own explanation.
  const body = page.split('const refreshHead')[1].split('useEffect')[0]
    .replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*/g, '')
  assert.doesNotMatch(body, /setStatus|nextBeforeId|hasMore/,
    'a background read must not blink the grid or rewind Load more')
  // An open image is an INDEX into the feed, and new rows land above it.
  assert.match(page, /setZoomIndex\(\(i\) => \(i == null \? i : i \+ added\)\)/)
  // The interval is torn down with the page — a poll that outlives its screen
  // keeps a dead component's closure alive and keeps asking.
  assert.match(page, /clearInterval\(timer\)/)
})

test('one finished job is worth several reads — the image lands AFTER the job goes', () => {
  /* The worker commits the queue row 'completed' and only then links the file
     and commits the image row, copying the PNG on the way. A single read fired
     on the edge can legitimately find nothing, and spending the edge there is
     how "I have to refresh the page" would have survived its own fix. */
  assert.match(page, /QUEUE_DRAIN_READS/)
  assert.match(page, /owed = QUEUE_DRAIN_READS/)
  assert.match(page, /if \(await refreshHead\(\)\) owed = 0/,
    'a read that brought something back ends the retries')
})

test('a background read cannot mix two feeds, or fill a grid that failed to load', () => {
  // The filters can move while a read is in flight: those rows belong to a feed
  // nobody is looking at any more, and their count belongs to it too.
  assert.match(page, /const asked = filtersRef\.current/)
  assert.match(page, /filtersRef\.current !== asked\) return 0/)
  // And nothing merges under an error banner, where the page has no action bar
  // and no infinite scroll.
  assert.match(page, /!readyRef\.current\) return/)
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
