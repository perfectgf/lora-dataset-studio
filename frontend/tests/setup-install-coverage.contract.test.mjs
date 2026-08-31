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

import { installCatalog, INSTALL_ALL_ACTION_LABELS } from '../src/hooks/useSetupSteps.js'

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
    ...pyDictKeys(installer, '_SEEDVR2_DOWNLOADS'),
    ...pyDictKeys(installer, '_CAMERA_DOWNLOADS'),
    ...pyDictKeys(installer, '_H3_DOWNLOADS'),
    ...pyDictKeys(installer, '_NODE_PACKS'),
  ]
  return new Set([...literal, ...downloads])
}

/** {group: [members]} parsed from _INSTALL_GROUPS. */
function backendGroups() {
  const start = installer.indexOf('_INSTALL_GROUPS = {')
  assert.ok(start >= 0, '_INSTALL_GROUPS not found')
  const block = installer.slice(start, installer.indexOf('\n}', start))
  const groups = {}
  for (const m of block.matchAll(/'([a-z0-9_]+)': \(([^)]*)\)/g)) {
    groups[m[1]] = [...m[2].matchAll(/'([a-z0-9_]+)'/g)].map((x) => x[1])
  }
  assert.ok(Object.keys(groups).length >= 3, 'suspiciously few install groups')
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
