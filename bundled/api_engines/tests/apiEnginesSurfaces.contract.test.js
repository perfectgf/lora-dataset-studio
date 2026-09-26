// The plugin's surfaces, read as text (node --test renders no JSX), and the
// partition they imply: what moved here is not in the core any more.
import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import descriptor from '../frontend/index.js'

const read = (rel) => readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8')

// One device-code implementation, mounted by both of the plugin's screens.
// Two copies of an OAuth polling loop is how "connected" starts meaning
// different things per page.
test('Settings and Setup share one subscription component, and neither re-implements the login', () => {
  const settings = read('../frontend/panels/ApiEnginesSettingsGroup.jsx')
  const setup = read('../frontend/panels/ChatgptSetupLane.jsx')
  assert.match(settings, /<ChatgptSubscriptionConnect/)
  assert.match(setup, /<ChatgptSubscriptionConnect/)
  assert.doesNotMatch(settings, /chatgpt-oauth\/start/)
  assert.doesNotMatch(setup, /chatgpt-oauth\/start/)
  // The lane panel wraps the core's key field rather than drawing its own input.
  assert.match(setup, /keyField\(\{ ok: chatgpt\.keySet, muted: true \}\)/)
})

// The wizard must OFFER the subscription, not merely tolerate it: the whole
// defect was a screen that only ever showed the key field. The engine spec is
// what puts the lane on the Setup step.
test('the ChatGPT spec carries the Setup lane panel and the key field describes ITS lane', () => {
  const specs = read('../frontend/lib/engineSpecs.js')
  assert.match(specs, /setupPanel: \(\) => import\('\.\.\/panels\/ChatgptSetupLane\.jsx'\)/)
  assert.doesNotMatch(specs, /help: 'Powers ChatGPT \(gpt-image-2\)\.'/,
    'the key field still claims to be what powers the engine')
})

// Every help topic the descriptor declares with a focus id must have that id
// rendered by one of the plugin's panels — a topic pointing at nothing is a
// dead "Open in Settings" link.
test('every focus id of the plugin help topics is a DOM id of a plugin panel', () => {
  const settings = read('../frontend/panels/ApiEnginesSettingsGroup.jsx')
  assert.match(settings, /import ChatgptSubscriptionModels from '\.\/ChatgptSubscriptionModels\.jsx'/)
  assert.match(settings, /<ChatgptSubscriptionModels\s+caps=\{caps\} config=\{config\} setField=\{setField\}/)
  const panels = settings + '\n' + read('../frontend/panels/ChatgptSubscriptionModels.jsx')
  const ids = new Set()
  for (const m of panels.matchAll(/id="([^"]+)"/g)) ids.add(m[1])
  for (const m of panels.matchAll(/\bkey:\s*'([^']+)'/g)) ids.add(m[1])
  for (const t of descriptor.help) {
    if (!t.app || !t.app.focus) continue
    assert.ok(ids.has(t.app.focus), `${t.id}: focus id "${t.app.focus}" not rendered by a plugin panel`)
  }
})
