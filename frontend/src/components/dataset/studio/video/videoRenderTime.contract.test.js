// ⏱ The render time reaches the clip card, through the shared label, never raw.
//
// `render_seconds` is the queue's own measurement (claim → settled), stored on
// the clip and published by _clip_dict. The card must print it through
// `renderTimeLabel` — "24 s", "5 min 48 s" — under a verb that matches the
// status ("rendered in" for a clip, "failed after" for one that died), and
// print nothing when the queue could not time the clip: a raw number would
// read "rendered in 347.6" for one clip and "rendered in null" for the next.
//
// Reads the JSX as text (node --test renders nothing): proves the wiring, not
// the pixels.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(join(here, 'VideoClipHistory.jsx'), 'utf8')
  .replace(/\{\/\*[\s\S]*?\*\/\}/g, '')
  .replace(/\/\/[^\n]*/g, '');

function metaLine() {
  const from = src.indexOf("'text-to-video' : 'image-to-video'");
  assert.ok(from > 0, 'the meta line of the card is where the render time belongs');
  const meta = src.slice(from);
  return meta.slice(0, meta.indexOf('</p>'));
}

test('the clip card prints the render time through renderTimeLabel, on the meta line', () => {
  assert.match(src, /import \{[^}]*\brenderTimeLabel\b[^}]*\} from '\.\/videoStudioApi'/);
  const line = metaLine();
  assert.match(line, /renderTimeLabel\(clip\.render_seconds\)/, 'the label comes from the shared helper');
  // The separator is the only thing between "0.5 MP" and the time: pinned.
  assert.match(line, /` · \$\{clip\.status === 'failed' \? 'failed after' : 'rendered in'\} \$\{renderTimeLabel\(clip\.render_seconds\)\}`/);
  assert.doesNotMatch(line, /\$\{clip\.render_seconds\}/, 'never the raw number');
});

test('a clip the queue could not time prints nothing, not "rendered in null"', () => {
  // The guard is the label itself: a falsy label yields the empty string.
  assert.match(metaLine(), /renderTimeLabel\(clip\.render_seconds\)\s*\?\s*`[^`]*`\s*:\s*''/);
});
