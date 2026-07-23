import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const controls = readFileSync(new URL('../src/components/dataset/FullBackupControls.jsx', import.meta.url), 'utf8')
const list = readFileSync(new URL('../src/components/dataset/DatasetListPanel.jsx', import.meta.url), 'utf8')

const menuPanel = () => {
  const start = controls.indexOf('absolute right-0 top-full')
  assert.notEqual(start, -1, 'backup menu panel not found')
  const end = controls.indexOf('</details>', start)
  assert.notEqual(end, -1, 'backup menu <details> is not closed')
  return controls.slice(start, end)
}

test('the library backup menu is a header popover on an opaque surface', () => {
  assert.match(controls, /<details ref=\{menuRef\} className="relative">/)
  assert.match(menuPanel(), /absolute right-0[^"\n]*bg-surface-overlay/)
})

test('"Include trained LoRAs" is an option INSIDE the menu, not a loose toolbar checkbox', () => {
  const panel = menuPanel()
  assert.match(panel, /Back up everything/)
  assert.match(panel, /Include trained LoRAs/)
  assert.match(panel, /type="checkbox"/)
  // The library header must not carry the backup controls itself any more:
  // no loose checkbox, no second restore file input.
  assert.doesNotMatch(list, /type="checkbox"/)
  assert.doesNotMatch(list, /Choose a dataset backup ZIP/)
})

test('"Import backup" moved into the same menu, "+ New dataset" stayed out of it', () => {
  const panel = menuPanel()
  assert.match(panel, /📦 Import backup/)
  assert.doesNotMatch(panel, /New dataset/)
  assert.match(list, /\+ New dataset/)
  assert.match(list, /<FullBackupControls backup=\{backup\} onRestore=\{onRestore\} \/>/)
})

test('a running backup stays visible with the menu closed', () => {
  // The label itself reports the in-flight state…
  assert.match(controls, /running \? 'Backing up…' : 'Backup'/)
  // …and both overlays are siblings of the <details>, never nested in it.
  const detailsEnd = controls.indexOf('</details>')
  assert.ok(controls.indexOf('<BackupOverlay') > detailsEnd, 'BackupOverlay must live outside the menu')
  assert.ok(controls.indexOf('<RestoreOverlay') > detailsEnd, 'RestoreOverlay must live outside the menu')
})

test('the restore file input is not trapped inside the collapsed disclosure', () => {
  const detailsEnd = controls.indexOf('</details>')
  assert.ok(controls.indexOf('ref={restoreRef} type="file"') > detailsEnd)
})
