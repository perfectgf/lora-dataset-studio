/**
 * ONE run number on every surface — the provenance record id.
 *
 * A run used to wear its CLOUD id on the Checkpoints/Runs chips ("☁ #158")
 * and its RECORD id on the lineage card ("#170 · cloud #158"): two numbers
 * for one run, in the same visual dress, plus a printed "· cloud #N"
 * secondary that read as a second id on every card. Worse than noise: the
 * checkpoints card's ⚙ Details passed the cloud id to a lookup keyed on
 * record ids, and answered "This run is not in the lineage tree" for a run
 * whose tree was one click away (user-reported, 2026-08-29 — the very panel
 * built the day before; its contract pinned the lookup's key but not what
 * the buttons PASSED, which is the hole this file closes).
 *
 * The rule these pins keep: the record id is THE number, printed alone;
 * the cloud id is context in tooltips (runIdentityLabel, the chip's title)
 * and the run inspector — never printed beside the number, never worn alone
 * by a run that has a record. Anchors and routes keep their own keys.
 */
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const SRC = new URL('../src/', import.meta.url)
const SRC_DIR = fileURLToPath(SRC)
const read = (p) => fs.readFileSync(new URL(p, SRC), 'utf8')

const chip = read('components/dataset/RunIdentityBadges.jsx')
const panel = read('components/dataset/TrainingPanel.jsx')
const runsPage = read('pages/CloudRunsPage.jsx')
const card = read('components/dataset/lineageNodes.jsx')
const tree = read('components/dataset/RunLineageTree.jsx')

/** Every .jsx under src — so a NEW surface printing a second number, or a new
 *  RunIdChip call still on the retired `id=` contract, fails here by name. */
const allJsx = []
;(function walk(dir) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name)
    if (e.isDirectory()) walk(p)
    else if (e.name.endsWith('.jsx') && !e.name.includes('.test.')) {
      allJsx.push([path.relative(SRC_DIR, p), fs.readFileSync(p, 'utf8')])
    }
  }
})(SRC_DIR)

test('the chip prints ONE number — the record id, cloud id as tooltip fallback only', () => {
  assert.match(chip, /const id = recordId \?\? cloudId/)
  // The legacy run keeps the only number it has, and the tooltip names its
  // kind instead of letting it silently wear a record id it does not have.
  assert.match(chip, /Cloud run #\$\{cloudId\} — from before run tracking/)
  // The record-backed tooltip carries the cloud id as CONTEXT.
  assert.match(chip, /\(cloud run #\$\{cloudId\}\)/)
})

test('no caller is left on the retired id= contract', () => {
  for (const [name, text] of allJsx) {
    for (const m of text.matchAll(/<RunIdChip\b[^>]*>/gs)) {
      assert.ok(!/\sid=/.test(m[0]),
        `${name} still passes id= to RunIdChip — pass recordId/cloudId`)
    }
  }
})

test('no surface PRINTS a second number beside the run number', () => {
  // "· cloud #N" may live in a title/tooltip string (runIdentityLabel, the
  // chip's own title) — never in rendered JSX text. The printed form always
  // interpolated (`cloud #{…}`), which no component may do any more.
  for (const [name, text] of allJsx) {
    assert.ok(!text.includes('cloud #{'),
      `${name} prints a cloud id beside the run number — tooltips only`)
  }
  // The two lineage surfaces print the ONE number and keep the full identity
  // in their title.
  for (const [name, text] of [['lineageNodes', card], ['RunLineageTree', tree]]) {
    assert.match(text, /title=\{runIdentityLabel\(node\)\}/,
      `${name} lost the full-identity tooltip`)
    assert.ok(!text.includes('cloudNumber('), `${name} prints the cloud id again`)
  }
})

test('⚙ Details and ⇄ Compare pass the RECORD id — the key the tree indexes', () => {
  // THE user-reported break: openRunDetails(g.run_id) chased a cloud id
  // through record-keyed nodes. The buttons now pass record_id, and only
  // exist when there is one (a pre-registry run recorded no recipe — no
  // button beats a dead-end error toast after the click).
  assert.match(panel, /openRunDetails\(g\.record_id\)/)
  assert.match(panel, /toggleRunCompare\(g\.record_id\)/)
  assert.ok(!panel.includes('openRunDetails(g.run_id)'), 'Details chases the cloud id again')
  assert.ok(!panel.includes('toggleRunCompare(g.run_id)'), 'Compare chases the cloud id again')
  assert.match(panel, /\{g\.record_id != null && \(<>/)
  // The header prints the number once (the chip) — the old duplicate
  // "Run #N" text (same number, printed twice, and it was the cloud id) is gone.
  assert.ok(!/Run #\{g\.run_id\}/.test(panel), 'the header prints the run number twice again')
})

test('the checkpoints group chip and the Runs-page chips print the record id', () => {
  assert.match(panel, /<RunIdChip source="cloud" recordId=\{g\.record_id\} cloudId=\{g\.run_id\} \/>/)
  // History card, live local card, active cloud card — all on the new contract.
  assert.match(runsPage, /recordId=\{run\.record_id\} cloudId=\{run\.run_id\}/)
  assert.match(runsPage, /recordId=\{data\.local_active\.record_id\}/)
  // Row ANCHORS keep their own keys (deep links must survive the display fix):
  // cloud rows still anchor by cloud id, local rows by record id.
  assert.match(runsPage, /runRowDomId\(node\.source, node\.source === 'cloud' \? node\.run_id : node\.record_id\)/)
})
