import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

/* Source-as-text contract for the 🧽 Klein clean's three dials.
 *
 * Bank and Dataset are two surfaces of one product (CLAUDE.md): a user who finds the
 * prompt box on one expects it on the other and files a bug when it is missing — and
 * "full parity, always" covers the DIALS, not only the pass underneath. The 🔤 Find
 * text sample dial already shipped bank-only with a written reason, and the reason did
 * not survive a day. So the mount is asserted on both, from the source, rather than
 * trusted to whoever ports the next change.
 *
 * Text assertions, not a render: these components pull the whole workspace tree in, and
 * a JSX runtime is not part of `node --test` here. What that buys is real (a deleted
 * mount fails this); what it cannot see is whether the panel is reachable on screen,
 * which is what the headless capture in the delivery covers. */
const SRC = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const REPO = resolve(SRC, '../..')          // frontend/src -> repo root
const read = (rel) => readFileSync(resolve(SRC, rel), 'utf8')
const readDoc = (rel) => readFileSync(resolve(REPO, rel), 'utf8')

const SURFACES = {
  bank: 'components/bank/BankWatermarkPanel.jsx',
  dataset: 'components/dataset/DatasetWorkspace.jsx',
}

test('both clean surfaces mount the SHARED dials component, exactly once', () => {
  for (const [name, rel] of Object.entries(SURFACES)) {
    const text = read(rel)
    assert.match(text, /import KleinCleanOptions from '\.\.\/shared\/KleinCleanOptions'/,
      `${name} does not import the shared clean options`)
    const mounts = text.match(/<KleinCleanOptions\b/g) || []
    assert.equal(mounts.length, 1,
      `${name} mounts <KleinCleanOptions> ${mounts.length} times — one control per `
      + 'surface, or two authorities for one stored value')
    // It belongs beside the Klein MODEL choice: same block, same condition, so the
    // three dials appear and disappear with the engine they belong to.
    assert.match(text, /<KleinModelSetting\b/,
      `${name} lost the Klein model choice these options sit next to`)
  }
})

test('the dials are written through to the config keys the backend reads', () => {
  const src = read('components/shared/KleinCleanOptions.jsx')
  assert.match(src, /putJson\('\/api\/settings', \{ config: \{ watermark_clean: patch \} \}\)/,
    'the panel does not write through to watermark_clean.* — the stored value is what '
    + 'every clean route reads, so a save that lands anywhere else arms nothing')
  for (const key of ['klein_prompt', 'klein_max_mp', 'klein_output']) {
    assert.match(src, new RegExp(`\\{ ${key}:`), `no dial writes ${key}`)
  }
  // The values it shows come from the RESOLVED capabilities, not from raw config: the
  // backend clamps, and a panel quoting an unclamped number would misdescribe the run.
  for (const cap of ['watermark_clean_prompt', 'watermark_clean_max_mp',
    'watermark_clean_output']) {
    assert.match(src, new RegExp(`caps\\.${cap}`), `the panel ignores caps.${cap}`)
  }
})

test('the prompt box offers a way back to the shipped instruction', () => {
  const src = read('components/shared/KleinCleanOptions.jsx')
  assert.match(src, /Reset to default/,
    'an editable prompt with no way back leaves a user who broke their clean with '
    + 'nothing but the guide to retype three words from')
  assert.match(src, /aria-label="Prompt sent to Klein/)
  assert.match(src, /aria-label="Klein clean processing size/)
  assert.match(src, /aria-label="What dimensions the cleaned file is written at"/)
})

test('every new dial has a help topic pointing at a REAL settings-reference anchor', () => {
  const topics = read('help/topics/settingsFields.js')
  const guide = readDoc('docs/guide/settings-reference.md')
  const anchors = new Set(
    (guide.match(/^## .+$/gm) || []).map((h) => h.replace(/^## /, '').toLowerCase()
      .replace(/[^a-z0-9 -]/g, '').trim().replace(/\s+/g, '-')),
  )
  for (const id of ['watermark_clean.klein_prompt', 'watermark_clean.klein_max_mp',
    'watermark_clean.klein_output']) {
    assert.ok(topics.includes(`'${id}'`), `no help topic for ${id}`)
  }
  /* The anchor an ⓘ opens has to exist, or the badge opens an empty modal. The three
     topics live in the Captioning & quality chapter, beside the Watermark inpainting
     section that documents them. */
  assert.ok(anchors.has('captioning-quality'),
    'settings-reference no longer has the H2 these topics point at')
})

test('the guide documents what each dial does, including the resizing one', () => {
  const guide = readDoc('docs/guide/settings-reference.md')
  for (const key of ['watermark_clean.klein_prompt', 'watermark_clean.klein_max_mp',
    'watermark_clean.klein_output']) {
    assert.ok(guide.includes(key), `settings-reference never names ${key}`)
  }
  // The one behaviour a user cannot undo by re-running: it must be written down.
  assert.match(guide, /file.{0,40}change[sd]?.{0,40}dimension/i,
    'the guide does not warn that the render write-back changes the file dimensions')
})
