import assert from 'node:assert/strict';
import test from 'node:test';
import { galleryHeadline, galleryZones, zoneStyle } from './textZonesGallery.js';

test('galleryZones keeps well-formed regions and orders their corners', () => {
  assert.deepEqual(galleryZones([[0.1, 0.2, 0.4, 0.3]]), [[0.1, 0.2, 0.4, 0.3]]);
  // Swapped corners still draw the same box — the editor normalizes the same way.
  assert.deepEqual(galleryZones([[0.4, 0.3, 0.1, 0.2]]), [[0.1, 0.2, 0.4, 0.3]]);
});

test('galleryZones drops what cannot be drawn instead of crashing the strip', () => {
  assert.deepEqual(galleryZones(null), []);
  assert.deepEqual(galleryZones('nope'), []);
  assert.deepEqual(galleryZones([null, [0.1], [0.1, 0.2, Number.NaN, 0.3]]), []);
  // Zero-area boxes are dropped: an invisible sliver reads as "no zone here"
  // on a page the scan DID flag.
  assert.deepEqual(galleryZones([[0.5, 0.5, 0.5, 0.9]]), []);
  // Out-of-range coordinates are clamped, not rejected.
  assert.deepEqual(galleryZones([[-0.2, 0.1, 1.4, 0.3]]), [[0, 0.1, 1, 0.3]]);
});

test('zoneStyle turns a region into percentage geometry', () => {
  assert.deepEqual(zoneStyle([0.1, 0.2, 0.4, 0.3]),
    { left: '10.00%', top: '20.00%', width: '30.00%', height: '10.00%' });
});

test('the headline separates "shown" from "flagged"', () => {
  assert.equal(galleryHeadline(12, 12), ' — 12 pages flagged');
  assert.equal(galleryHeadline(12, 300), ' — 300 pages flagged, first 12 shown');
  assert.equal(galleryHeadline(1, 1), ' — 1 page flagged');
});
