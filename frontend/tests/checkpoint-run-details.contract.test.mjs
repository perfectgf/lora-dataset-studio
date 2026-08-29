/* ⚙ / ⇄ on a checkpoint card — the wiring that must not drift.

   The run recipe panel and the two-run diff have always existed one screen
   away, in the Lineage graph, and "one screen away" is where nobody found
   them (user-reported). These pins hold the direct access from the cards:
   the SAME panels (never copies), resolved through the SAME lineage tree,
   through the diff reducer's own bounded-to-two contract. Source-read, like
   every layout contract here. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';

const panel = fs.readFileSync(
  path.join(process.cwd(), 'src/components/dataset/TrainingPanel.jsx'), 'utf8');

test('the cards open the SAME panels the lineage graph mounts — not copies', () => {
  assert.match(panel, /import LineageDetailPanel from '\.\/LineageDetailPanel'/);
  assert.match(panel, /import LineageDiffPanel from '\.\/LineageDiffPanel'/);
  assert.match(panel, /import \{ toggleDiffSelection \} from '\.\/lineageDetail\.js'/,
    'the two-pick window is the diff reducer’s own contract, not a re-implementation');
});

test('each cloud run card carries the two accesses', () => {
  assert.match(panel, /data-testid="ckpt-run-details"/);
  assert.match(panel, /data-testid="ckpt-run-compare"/);
});

test('nodes come from the shared lineage tree, fetched on demand', () => {
  // The graph auto-loads the tree only when the Graph view shows; a card can
  // ask first, so the handlers must be able to fetch it themselves…
  assert.match(panel, /const ensureLineageTree = async/);
  // …and both resolve through record_id — the one id the tree, the cards and
  // the Runs page all share.
  assert.match(panel, /n\.record_id === recordId/);
});
