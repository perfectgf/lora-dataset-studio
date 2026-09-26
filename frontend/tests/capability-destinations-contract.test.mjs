import test from 'node:test'
import { readSource } from './support/readSource.mjs'
import assert from 'node:assert/strict'

import { deriveCapabilitySummary, capabilityDestination } from '../src/hooks/useSetupSteps.js'
import { getHelpTopic } from '../src/help/helpRegistry.js'
import { isValidTarget } from '../src/whatsNew.js'
import { SETTINGS_SECTIONS } from '../src/components/settings/registry.js'
import { resetRegistry, setEnabled } from '../src/plugins/registry.js'
import { registerBundledDescriptor } from './support/bundledDescriptors.mjs'
import apiEngines from '../../bundled/api_engines/frontend/index.js'
import camera from '../../bundled/camera_angles/frontend/index.js'
import video from '../../bundled/video/frontend/index.js'
import live from '../../bundled/live/frontend/index.js'
import scrape from '../../bundled/scrape/frontend/index.js'
import civitai from '../../bundled/civitai_publish/frontend/index.js'

const products = [apiEngines, camera, video, live, scrape, civitai]
test.beforeEach(() => {
  resetRegistry()
  for (const product of products) assert.equal(registerBundledDescriptor(product), true)
  setEnabled(products.map(product => product.id))
})
test.afterEach(resetRegistry)

/* The Settings ▸ Overview capability grid is a dashboard AND a set of doors: a
   row that says "✗ Person masks" has to be clickable straight to the control
   that turns person masks on. This contract is what keeps a rotten door from
   ever reaching the screen — every row must carry a destination, and every
   destination must resolve against the LIVE registries (help topics, settings
   sections, the what's-new target validator), exactly the way whatsNew.test.js
   validates its own "Try it →" targets. */

const read = readSource

// Three rigs that between them light up every row shape: nothing configured,
// everything configured, and the "installed but ComfyUI isn't running" rig that
// puts Klein / Test Studio in the `pending` state.
const CAPS_EMPTY = {}
const CAPS_FULL = {
  engines: { nanobanana: true, chatgpt: true, openrouter: true, klein: true, krea: true },
  captioners: { joycaption: true, ollama: true },
  ollama: { reachable: true, vision_model_ready: true },
  comfyui: { dir_valid: true, reachable: true, video_studio_ready: true,
    video_studio_options: { vfi: { available: true } } },
  live: { ready: true, encoder: true }, video_encode: true,
  face_scoring: true, masks: true, watermark_inpaint: true,
  training_visible: true, studio_visible: true,
  civitai: { ok: true },
}
const CAPS_COMFY_OFF = { comfyui: { dir_valid: true, reachable: false },
  live: { ready: false, encoder: true, missing: [] } }

const RIGS = [
  ['nothing configured', CAPS_EMPTY],
  ['everything ready', CAPS_FULL],
  ['ComfyUI installed but not running', CAPS_COMFY_OFF],
]

test('every capability row carries a destination, in every rig', () => {
  for (const [name, caps] of RIGS) {
    const rows = deriveCapabilitySummary(caps)
    // The fixed public fixture exposes 12 core capabilities and 14 product rows.
    // Missing requirements stay counted; absent products are checked separately.
    assert.equal(rows.length, 26, `${name}: expected core and active public product capabilities`)
    for (const row of rows) {
      const dest = capabilityDestination(row)
      assert.ok(dest, `${name}: "${row.label}" has no destination`)
      assert.ok(dest.href && dest.href.startsWith('/'),
        `${name}: "${row.label}" href is not an in-app path (${dest.href})`)
      assert.ok(dest.where && dest.where.trim(),
        `${name}: "${row.label}" has no human destination name`)
    }
  }
})

test('a core-only installation only advertises its own capabilities', () => {
  setEnabled([])
  const rows = deriveCapabilitySummary(CAPS_FULL)
  assert.equal(rows.length, 12)
  assert.ok(rows.every(row => !row.pluginId))
})

test('every destination topic exists in the LIVE help registry', () => {
  for (const [name, caps] of RIGS) {
    for (const row of deriveCapabilitySummary(caps)) {
      const id = capabilityDestination(row).topic
      assert.ok(getHelpTopic(id),
        `${name}: "${row.label}" points at unknown help topic "${id}"`)
    }
  }
})

test('every destination href is a navigable in-app target', () => {
  for (const [name, caps] of RIGS) {
    for (const row of deriveCapabilitySummary(caps)) {
      const { href } = capabilityDestination(row)
      // Strip the focus hint: it is a DOM id, validated separately by the
      // help-registry contract, and not part of the route grammar.
      const route = href.replace(/([?&])focus=[^&]*/, '$1').replace(/[?&]$/, '')
      assert.equal(isValidTarget(route), true,
        `${name}: "${row.label}" → ${href} is not navigable`)
    }
  }
})

test('a settings destination names a real Settings section', () => {
  const titles = new Map(SETTINGS_SECTIONS.map((s) => [s.id, s.title]))
  for (const [name, caps] of RIGS) {
    for (const row of deriveCapabilitySummary(caps)) {
      const { href, where } = capabilityDestination(row)
      const m = href.match(/^\/settings\/([a-z0-9-]+)/)
      if (!m) continue
      assert.ok(titles.has(m[1]), `${name}: unknown settings section ${m[1]}`)
      assert.equal(where, titles.get(m[1]),
        `${name}: "${row.label}" announces "${where}" but lands on ${titles.get(m[1])}`)
    }
  }
})

test('a pending row is not a missing one: own destination, own wording', () => {
  const pending = deriveCapabilitySummary(CAPS_COMFY_OFF).filter((r) => r.pending)
  // Camera angles joins the pending set for the same reason Klein does: the
  // lane is asset-only, so with the weights on disk and only the process down
  // the honest state is "waiting for ComfyUI", never "install something".
  // Smooth and Live wait with the 🎬 row: their verdict needs ComfyUI up
  // (Smooth's packs are read from /object_info). DLSS has a worker of its
  // own and never waits on ComfyUI, so it is not in this list.
  assert.deepEqual(pending.map((r) => r.label),
    ['Klein (local)', '🖼️ Test Studio (images)', '📷 Camera angles (local)',
      '🎬 Video Test Studio (beta)', '↗ Smooth (frame interpolation)', 'Live — local generation'],
    'ComfyUI down leaves Klein + Camera angles + the video rows + Test Studio pending')
  for (const row of pending) {
    assert.ok(row.note, `${row.label}: pending row must explain itself`)
    const waiting = capabilityDestination(row)
    // Same row, ComfyUI genuinely absent → the install path, a DIFFERENT door.
    const missing = capabilityDestination({ ...row, pending: false, note: undefined })
    assert.notEqual(waiting.href, missing.href,
      `${row.label}: "waiting for a process" and "not installed" must not send the user to the same place`)
  }
})

test('the accessible label says the state AND where the row leads', () => {
  const rows = deriveCapabilitySummary(CAPS_COMFY_OFF)
  const label = (l) => {
    const row = rows.find((r) => r.label === l)
    return capabilityDestination(row).announce
  }
  assert.match(label('OpenRouter'), /^OpenRouter — not available, configure in /)
  assert.match(label('Klein (local)'), /^Klein \(local\) — launch ComfyUI to enable, /)
  const ready = deriveCapabilitySummary(CAPS_FULL).find((r) => r.label === 'OpenRouter')
  assert.match(capabilityDestination(ready).announce, /^OpenRouter — ready, /)
})

test('every row says what it unlocks, and both screens show it', () => {
  // A name alone did not: "Test Studio" read from a phone said nothing about
  // test IMAGES (2026-09-03). One sentence per row, short enough to wrap on a
  // 360 px tile, and rendered wherever the rows are — the Overview grid and
  // the wizard's "What's unlocked" screen.
  for (const [name, caps] of RIGS) {
    for (const row of deriveCapabilitySummary(caps)) {
      assert.ok(typeof row.what === 'string' && row.what.trim().length >= 20,
        `${name}: "${row.label}" does not say what it unlocks`)
      assert.ok(row.what.length <= 100, `${name}: "${row.label}" — the what-line is a paragraph`)
    }
  }
  assert.match(read('src/components/settings/OverviewSection.jsx'), /\{s\.what && <span/)
  assert.match(read('src/pages/SetupPage.jsx'), /\{s\.what && <span/)
})

test('the Overview grid actually uses the destinations (no dead tiles)', () => {
  const src = read('src/components/settings/OverviewSection.jsx')
  assert.match(src, /capabilityDestination/,
    'OverviewSection must resolve each tile through capabilityDestination')
  assert.match(src, /<Link\b/, 'tiles must be real links, not clickable divs')
  assert.doesNotMatch(src, /FIX_LINKS/,
    'the coarse "Where to fix it" table is superseded by per-capability destinations')
})
