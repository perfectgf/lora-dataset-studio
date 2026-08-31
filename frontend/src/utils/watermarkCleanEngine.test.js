import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'
import {
  CLEAN_ENGINES_BLURB, KLEIN_CLEAN_SHORT, KLEIN_CLEAN_TITLE, kleinCleanTitle,
} from './watermarkCleanEngine.js'

const SRC = resolve(dirname(fileURLToPath(import.meta.url)), '..')

/* The three places the 🧽 clean engine is offered. Bank and Dataset are two
 * surfaces of one product (CLAUDE.md), so the sentence describing an engine has
 * to be the same on both — a user who reads it on one expects the other to
 * behave identically, and reports a difference as a bug. */
const SURFACES = [
  'components/bank/BankWatermarkPanel.jsx',
  'components/dataset/DatasetWorkspace.jsx',
  'components/dataset/WatermarkReviewLightbox.jsx',
]

const read = (rel) => readFileSync(resolve(SRC, rel), 'utf8')

test('every surface that offers Klein shows the SAME sentence about it', () => {
  for (const rel of SURFACES) {
    const text = read(rel)
    assert.match(text, /watermarkCleanEngine\.js/,
      `${rel} does not import the shared clean-engine wording`)
    /* kleinCleanTitle(caps), not the bare constant: since the prompt became editable
       (2026-08-31) the sentence has an install-specific half, and a surface that
       pasted the constant would quote "remove watermark" at a user who had changed
       it — the exact failure the shared module exists to prevent, one level down. */
    assert.match(text, /kleinCleanTitle\(caps\)/,
      `${rel} describes the Klein clean in its own words, or names the prompt from a `
      + 'stale constant instead of the resolved capabilities value')
  }
})

test('the tooltip names the prompt this install will actually send', () => {
  /* "remove watermark" lived only in the backend source, which is not a place a user
     can look — the maintainer's own question ("we can't see what is sent?"). Now the
     value is published resolved (caps.watermark_clean_prompt) and quoted here. */
  assert.match(kleinCleanTitle({ watermark_clean_prompt: 'erase every logo' }),
    /Sent to Klein: “erase every logo”/)
  // The whole shared sentence survives the addition — the trade is not replaced by it.
  assert.ok(kleinCleanTitle({}).startsWith(KLEIN_CLEAN_TITLE))
  // No caps yet (the probe is async), an empty prompt, a non-string: all read as the
  // shipped default rather than an empty quote or "undefined".
  for (const caps of [undefined, {}, { watermark_clean_prompt: '  ' },
    { watermark_clean_prompt: 7 }]) {
    assert.match(kleinCleanTitle(caps), /Sent to Klein: “remove watermark”/)
  }
})

test('the sentence states BOTH halves of the 2026-08-31 trade', () => {
  // What it buys: the whole photo, so a mark a box cannot frame comes out.
  assert.match(KLEIN_CLEAN_TITLE, /whole photo/i)
  assert.match(KLEIN_CLEAN_TITLE, /ON the subject/)
  // What it costs: everything is regenerated, and the way back.
  assert.match(KLEIN_CLEAN_TITLE, /every pixel is regenerated/i)
  assert.match(KLEIN_CLEAN_TITLE, /Restore original/)
  for (const s of [KLEIN_CLEAN_SHORT, CLEAN_ENGINES_BLURB]) {
    assert.match(s, /whole photo/i)
  }
})

test('the sentence names the LIMIT, not just the reach', () => {
  /* Measured on the shipped recipe: a tiled mark goes 12 zones → 0, and the erase
   * step is what keeps a distinct logo from coming back REDRAWN (a round one
   * returned as a moon in the sky, three runs, three seeds — and the detector
   * scores that image 0 zones, so no later step disagrees). What survives is a
   * mark the scan never found. All of that has to be in the words: every limit
   * stays visible (CLAUDE.md), and a limit hidden becomes a bug report. */
  assert.match(KLEIN_CLEAN_TITLE, /surviv/i,
    'the tooltip promises a removal the measurements do not guarantee')
  assert.match(KLEIN_CLEAN_TITLE, /look at the result/i,
    'the render is a generative pass over the whole picture — if the tooltip does not '
    + 'send the user to look at it, nothing else will')
  assert.match(KLEIN_CLEAN_TITLE, /erase/i,
    'the zones are erased BEFORE the render, and that step is the whole reason the '
    + 'result is usable — a sentence that skips it describes the naked recipe')
  assert.match(KLEIN_CLEAN_TITLE, /tiled|repeated|missed/i,
    'the tooltip does not say where this engine is actually strong')
  for (const s of [KLEIN_CLEAN_SHORT, CLEAN_ENGINES_BLURB]) {
    assert.match(s, /tiled|repeated|missed/i)
    assert.match(s, /erase/i)
  }
})

test('no surface still calls the Klein clean an inpaint of a crop', () => {
  // The pre-2026-08-31 vocabulary. It described a real mechanism ("masked
  // Flux.2 inpaint (crop-and-stitch)… only the mark changes") that no longer
  // exists on this lane, and a tooltip that promises byte-exactness the code
  // stopped providing is worse than no tooltip.
  const stale = [
    /masked Flux\.2 inpaint/,
    /crop-and-stitch/,
    /only the mark changes/,
  ]
  for (const rel of SURFACES) {
    const text = read(rel)
    for (const re of stale) {
      assert.ok(!re.test(text), `${rel} still describes the clean as ${re}`)
    }
  }
  for (const s of [KLEIN_CLEAN_TITLE, KLEIN_CLEAN_SHORT, CLEAN_ENGINES_BLURB]) {
    for (const re of stale) assert.ok(!re.test(s), `the shared wording still says ${re}`)
  }
})
