import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const ws = fs.readFileSync(new URL('./BankWorkspace.jsx', import.meta.url), 'utf8');
const panel = fs.readFileSync(new URL('./DupGroupsPanel.jsx', import.meta.url), 'utf8');
const dialog = fs.readFileSync(new URL('./LaunchAllDialog.jsx', import.meta.url), 'utf8');

test('the semantic near-duplicate badge is distinct from the exact-duplicate one', () => {
  // Exact dups use ≈ (fuchsia); semantic dups use ✂ (orange) — a different mark
  // AND a different colour so the two stages never read as the same thing.
  assert.match(ws, /≈\$\{img\.dup_group\}/);
  assert.match(ws, /text-fuchsia-200/);
  assert.match(ws, /✂\$\{img\.semantic_dup_group\}/);
  assert.match(ws, /text-orange-200/);
});

test('the workspace renders both stages through the shared panel with distinct kinds', () => {
  assert.match(ws, /filter\.flag === 'dups'/);
  assert.match(ws, /kind="exact"/);
  assert.match(ws, /filter\.flag === 'semantic_dups'/);
  assert.match(ws, /kind="semantic"/);
});

test('the ✂ Find crops button gates on Score having run', () => {
  assert.match(ws, /startSemanticDedup/);
  assert.match(ws, /\/api\/bank\/\$\{bankId\}\/semantic-dedup/);
  // Disabled until at least one image is scored (embeddings exist).
  assert.match(ws, /disabled=\{live \|\| scored === 0\}/);
});

test('the resolution panel hits the semantic endpoints and uses same-shot wording', () => {
  assert.match(panel, /semantic-dup-groups/);
  assert.match(panel, /semantic-dups\/resolve/);
  assert.match(panel, /same shot/i);
  // Both stages share the keep-best / keep-first / pick-one resolution.
  assert.match(panel, /Resolve ALL — keep best/);
  assert.match(panel, /keep_ids:\s*\[img\.id\]/);
});

test('Launch all inserts the semantic step right after Score, defaulting on when ready', () => {
  assert.match(dialog, /key:\s*'semantic_dedup'/);
  assert.match(dialog, /semantic_dedup:\s*!!caps\?\.bank_scoring/);
  const m = dialog.match(/\[([^\]]*)\]\s*\n\s*\.filter\(\(k\)\s*=>\s*ready\[k\]\)/);
  assert.ok(m, 'found the default step set');
  const order = m[1];
  assert.ok(order.indexOf("'semantic_dedup'") > order.indexOf("'score'"),
    'semantic_dedup follows score in the default set');
});
