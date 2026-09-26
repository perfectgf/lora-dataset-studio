/* Every install the backend can run is REACHABLE from the Setup screen.
 *
 * THE HOLE THIS CLOSES, twice over. The 📷 Camera angles weights shipped
 * installable through their 409 and visible NOWHERE on Setup: no catalog row,
 * no card, not counted — a whole engine invisible on the very screen where a
 * user decides they are done. That was the third repeat of a documented defect
 * (Krea had it, the video lane had it), and each time it was found by a person,
 * later. And the very first run of this file found a fourth: the Klein
 * enhancement LoRA — installable by the API for weeks, offered by nothing.
 *
 * A capability probe without an install action is already caught by the
 * backend's test_every_capability_the_app_probes_can_be_installed_from_setup.
 * This is the OTHER half, which nothing checked: an install action without a
 * SURFACE. The backend registry and the screens live in different languages
 * (Python constants, JSX), so this reads both as text — the same trick
 * cameraCatalogContract uses — and fails naming the action and the four ways
 * to expose one.
 *
 * The four surfaces an action may reach the user through:
 *   1. a row of installCatalog (the Install screen's repair menu);
 *   2. a card in mlInstallCards.js (the quality step);
 *   3. membership of an _INSTALL_GROUPS group some card posts
 *      (install-group/<name>);
 *   4. an inline <InstallRunner action="..."> on a wizard screen.
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { installCatalog, INSTALL_ALL_ACTION_LABELS } from '../src/hooks/useSetupSteps.js'
import { resetRegistry, setEnabled } from '../src/plugins/registry.js'
import { registerBundledDescriptor } from './support/bundledDescriptors.mjs'

const BUNDLED = path.join(process.cwd(), '..', 'bundled')
const products = await Promise.all(fs.readdirSync(BUNDLED).filter(id =>
  fs.existsSync(path.join(BUNDLED, id, 'plugin.json'))).map(async id => ({
  id, manifest: JSON.parse(fs.readFileSync(path.join(BUNDLED, id, 'plugin.json'), 'utf8')),
  descriptor: (await import(pathToFileURL(path.join(BUNDLED, id, 'frontend/index.js')))).default,
})))
test.beforeEach(resetRegistry)
test.afterEach(resetRegistry)

const BACKEND = path.join(process.cwd(), '..', 'backend', 'app')
const installer = fs.readFileSync(path.join(BACKEND, 'setup_installer.py'), 'utf8')

/** The keys of one `_NAME = {` python dict literal, read as text. */
function pyDictKeys(source, name) {
  const start = source.indexOf(`${name} = {`)
  assert.ok(start >= 0, `${name} not found in setup_installer.py`)
  const block = source.slice(start, source.indexOf('\n}', start))
  // Top-level entries are indented exactly four spaces in this file.
  return [...block.matchAll(/\n    '([a-z0-9_]+)': \{/g)].map((m) => m[1])
}

/** Every action the backend will accept — the same union INSTALL_ACTIONS is
 *  built from, recomputed here so a new catalog is seen without this file
 *  changing. The literal half is parsed from the tuple. */
function backendActions() {
  const tuple = installer.match(/INSTALL_ACTIONS = \(([^)]*)\)/)
  assert.ok(tuple, 'INSTALL_ACTIONS tuple not found')
  const literal = [...tuple[1].matchAll(/'([a-z0-9_]+)'/g)].map((m) => m[1])
  const downloads = [
    ...pyDictKeys(installer, '_KLEIN_DOWNLOADS'),
    ...pyDictKeys(installer, '_KREA_DOWNLOADS'),
    ...pyDictKeys(installer, '_NODE_PACKS'),
    ...pyDictKeys(installer, '_BUNDLED_NODE_PACKS'),
  ]
  return new Set([...literal, ...downloads])
}

/** {group: [members]} parsed from _INSTALL_GROUPS. */
function backendGroups() {
  const start = installer.indexOf('_INSTALL_GROUPS = {')
  assert.ok(start >= 0, '_INSTALL_GROUPS not found')
  const block = installer.slice(start).split(/\r?\n\r?\n/)[0]
  const groups = {}
  for (const m of block.matchAll(/'([a-z0-9_]+)': \(([^)]*)\)/g)) {
    groups[m[1]] = [...m[2].matchAll(/'([a-z0-9_]+)'/g)].map((x) => x[1])
  }
  assert.deepEqual(Object.keys(groups), ['krea'], 'core owns Krea; product groups belong to their manifest')
  return groups
}

/** Every group name some Setup component actually posts. */
function postedGroups() {
  const dir = path.join(process.cwd(), 'src', 'components', 'setup')
  const names = new Set()
  for (const f of fs.readdirSync(dir)) {
    if (!/\.jsx?$/.test(f)) continue
    const src = fs.readFileSync(path.join(dir, f), 'utf8')
    for (const m of src.matchAll(/install-group\/([a-z0-9_]+)/g)) names.add(m[1])
  }
  return names
}

/** Every literal <InstallRunner action="..."> on any screen. */
function inlineRunnerActions() {
  const roots = [path.join(process.cwd(), 'src', 'pages'),
    path.join(process.cwd(), 'src', 'components', 'setup')]
  const actions = new Set()
  for (const root of roots) {
    for (const f of fs.readdirSync(root)) {
      if (!/\.jsx?$/.test(f)) continue
      const src = fs.readFileSync(path.join(root, f), 'utf8')
      for (const m of src.matchAll(/action="([a-z0-9_]+)"/g)) actions.add(m[1])
      if (/<InstallRunner\b[^>]*action=\{ACTION\}/.test(src)
          || src.includes('postJson(`/api/setup/install/${ACTION}`, {}')) {
        const action = src.match(/const ACTION = '([a-z0-9_]+)'/)
        assert.ok(action, `${f}: unresolved install action`)
        actions.add(action[1])
      }
    }
  }
  return actions
}

function mlCardActions() {
  const src = fs.readFileSync(
    path.join(process.cwd(), 'src', 'components', 'setup', 'mlInstallCards.js'), 'utf8')
  return new Set([...src.matchAll(/action: '([a-z0-9_]+)'/g)].map((m) => m[1]))
}

// A caps payload where everything is possible, so installCatalog lists every
// row it is capable of listing.
const FULL_CAPS = { comfyui: { dir_valid: true, reachable: true } }

// Video's optional performance installers get their action IDs from its API,
// rather than writing literals in JSX. Follow that contract through the mounted
// panel and the backend status payload before counting those buttons as reachable.
function videoPerformanceActions() {
  const readVideo = file => fs.readFileSync(path.join(BUNDLED, 'video', file), 'utf8').replace(/\r\n/g, '\n')
  const panel = readVideo('frontend/studio/video/VideoOptionsPanel.jsx')
  const controls = readVideo('frontend/studio/video/VideoPerformanceOptions.jsx')
  const route = readVideo('lds_video/routes/video_studio.py')
  const backend = readVideo('lds_video/h3_performance.py')
  assert.match(panel, /<VideoPerformanceOptions\s+options=\{options\}/)
  assert.match(controls, /const status = options\?\.performance \|\| \{\}/)
  assert.match(controls, /postJson\(`\/api\/setup\/install\/\$\{action\}`, \{\}\)/)
  assert.match(controls, /<button[^>]*onClick=\{start\}/)
  assert.match(route, /'performance': h3_performance\.status\(classes\)/)
  const status = backend.match(/def status\(classes\):([\s\S]*?)\n\ndef /)?.[1]
  assert.ok(status, 'Video performance status provider exists')
  const actions = []
  for (const match of controls.matchAll(/<Prepare action=\{status\.([a-z0-9_]+)\.action\}/g)) {
    const entry = status.match(new RegExp(`'${match[1]}': \\{([\\s\\S]*?)(?=\\n {8}'|\\n {4}\\})`))?.[1]
    assert.ok(entry, `${match[1]} preparation button has a backend status entry`)
    const literal = entry.match(/'action': '([a-z0-9_]+)'/)?.[1]
    // The attention action is a shared SDK constant and already has a catalog row.
    if (literal) actions.push(literal)
    else assert.match(entry, /'action': host\.BLOCK_ATTN_INSTALL_ACTION/)
  }
  assert.ok(actions.length, 'Video performance actions must resolve from the API')
  return actions
}

test('every backend install action is reachable from a Setup surface', () => {
  const groups = backendGroups()
  const posted = postedGroups()
  const reachable = new Set([
    ...installCatalog(FULL_CAPS).map((r) => r.action),
    ...mlCardActions(),
    ...inlineRunnerActions(),
    ...[...posted].flatMap((g) => groups[g] || []),
  ])
  const orphans = [...backendActions()].filter((a) => !reachable.has(a))
  assert.deepEqual(orphans, [],
    `installable through the API and offered NOWHERE on Setup: ${orphans.join(', ')}. `
    + 'Give each one a surface — a row in installCatalog, a card in '
    + 'mlInstallCards.js, membership of a posted install group, or an inline '
    + '<InstallRunner action="...">. An action without a surface is the Camera/'
    + 'Krea hole again: a user decides they are done on a screen that cannot '
    + 'show them what is missing.')
})

test('every group a card posts exists on the backend', () => {
  // The reverse direction: a card posting a group the backend renamed would
  // 404 on click — a button that looks real and does nothing.
  const groups = backendGroups()
  for (const g of postedGroups()) {
    assert.ok(groups[g], `a Setup card posts install-group/${g}, which the backend does not define`)
  }
})

test('every surfaced action carries a human label', () => {
  const groups = backendGroups()
  const surfaced = new Set([
    ...installCatalog(FULL_CAPS).map((r) => r.action),
    ...Object.values(groups).flat(),
  ])
  for (const a of surfaced) {
    assert.ok(INSTALL_ALL_ACTION_LABELS[a],
      `${a} is on a Setup surface with no entry in INSTALL_ALL_ACTION_LABELS — `
      + 'its row would render as a bare identifier')
  }
})

for (const product of products) {
  test(`${product.id} alone exposes each declared preparation action`, () => {
    assert.equal(registerBundledDescriptor(product.descriptor), true)
    setEnabled([product.id])
    const rows = installCatalog(FULL_CAPS)
    const reachable = new Set(rows.map(row => row.action))
    if (product.id === 'video') {
      for (const action of videoPerformanceActions()) reachable.add(action)
    }
    const files = fs.readdirSync(path.join(BUNDLED, product.id, 'frontend'), { recursive: true })
    const posted = new Set()
    for (const file of files.filter(name => /\.jsx?$/.test(name))) {
      const source = fs.readFileSync(path.join(BUNDLED, product.id, 'frontend', file), 'utf8')
      for (const match of source.matchAll(/action="([a-z0-9_]+)"/g)) reachable.add(match[1])
      // Some cards bind one named constant to InstallRunner instead of a literal.
      if (/<InstallRunner\b[^>]*action=\{ACTION\}/.test(source)
          || source.includes('postJson(`/api/setup/install/${ACTION}`, {}')) {
        const action = source.match(/const ACTION = '([a-z0-9_]+)'/)
        assert.ok(action, `${product.id}/${file}: unresolved install action`)
        reachable.add(action[1])
      }
      for (const match of source.matchAll(/install-group\/([a-z0-9_]+)/g)) posted.add(match[1])
    }
    assert.deepEqual((product.manifest.owns?.install_actions || []).filter(action => !reachable.has(action)), [],
      'Every action must be reachable from this product alone, without another product supplying its UI')
    for (const group of posted) assert.ok(product.manifest.owns?.install_groups?.includes(group), group)
    for (const row of rows) assert.ok(row.label && row.label !== row.action, `${product.id}: ${row.action} needs a human label`)
  })
}
