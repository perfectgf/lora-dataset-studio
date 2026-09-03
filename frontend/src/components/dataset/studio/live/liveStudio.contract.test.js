// 🔴 The Live lane is wired where the probe, the help and the player expect it.
//
// Reads the JSX as text (node --test renders nothing): the tab exists on the
// Studio page under the testid the responsive probe opens, the panel loads
// hls.js on demand rather than in the Studio bundle, and the surfaces the
// probe measures carry their markers.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const read = (p) => readFileSync(join(here, p), 'utf8');
const page = read('../../../../pages/StudioPage.jsx');
const lane = read('./LiveStudio.jsx');

test('the Studio page offers the Live lane as a third tab and remembers it', () => {
  assert.match(page, /\{ id: 'live', label: 'Live', icon: Radio, badge: 'beta' \}/, 'the tab says beta next to Live');
  assert.match(page, /const LANES = \['image', 'video', 'live'\]/);
  assert.match(page, /lane === 'live' \? \(\s*<LiveStudio \/>/);
});

test('hls.js is loaded on demand, only when there is a playlist to play', () => {
  assert.match(lane, /await import\('hls\.js'\)/, 'a dynamic import keeps it out of the Studio chunk');
  assert.doesNotMatch(lane, /^import .*hls\.js/m, 'never a static import');
  assert.match(lane, /canPlayType\('application\/vnd\.apple\.mpegurl'\)/, 'Safari plays HLS natively');
  assert.match(lane, /hlsRef\.current\.destroy\(\)/, 'the player is torn down with the playlist');
});

test('the surfaces the responsive probe measures carry their markers', () => {
  for (const marker of ['live-take', 'live-rail', 'live-player']) {
    assert.match(lane, new RegExp(`data-probe-panel="${marker}"`), marker);
  }
  assert.match(lane, /data-testid="live-player"/);
  // Finger-sized controls below lg, unchanged on a desktop — the app's idiom.
  assert.ok((lane.match(/min-h-10 lg:min-h-0/g) || []).length >= 6, 'every dial is a 40 px target on a phone');
});

test('the address VLC opens is the one the server published, on this origin', () => {
  assert.match(lane, /streamUrlFor\(status, origin\)/);
  assert.match(lane, /window\.location\.origin/);
  assert.match(lane, /const streamReady = !!vlcUrl && \(status\?\.segments \|\| 0\) > 0/, 'no address before the playlist exists');
});

test('the browser player has a failure surface and prefers hls.js to a native "maybe"', () => {
  assert.match(lane, /hls\.on\(Hls\.Events\.ERROR/, 'a fatal error is shown, not a black frame');
  assert.match(lane, /catch \{\s*if \(!cancelled\) setPlayerError/, 'a failed import is shown');
  assert.ok(lane.indexOf('Hls.isSupported()') < lane.indexOf("canPlayType('application/vnd.apple.mpegurl')"),
    'hls.js first: Chrome answers "maybe" to the HLS MIME and cannot play it natively');
  assert.match(lane, /video\.removeAttribute\('src'\)/, 'the native path is cleared with the player');
  assert.match(lane, /apiFetch\(liveStatusUrl\(\), \{ background: true \}\)/, 'a poll never toasts per miss');
  assert.match(lane, /<HelpBadge topic="page-video-live" \/>/, 'the lane opens its own help');
});
