import assert from 'node:assert/strict'
import test from 'node:test'

import { mountExtensionScripts } from '../src/utils/extensionLoader.js'

function fakeDoc() {
  const appended = []
  return {
    appended,
    createElement: (tag) => ({ tag, dataset: {} }),
    head: { appendChild: (el) => appended.push(el) },
  }
}

test('mounts one module script per extension that declares a frontend entry', () => {
  const doc = fakeDoc()
  const mounted = mountExtensionScripts(
    [
      { name: 'a', frontend_entry: '/api/a/ui.js' },
      { name: 'b', frontend_entry: null },
      { name: 'c', frontend_entry: '/api/c/ui.js' },
    ],
    doc,
  )
  assert.deepEqual(mounted, ['a', 'c'])
  assert.equal(doc.appended.length, 2)
  assert.equal(doc.appended[0].type, 'module')
  assert.equal(doc.appended[0].src, '/api/a/ui.js')
  assert.equal(doc.appended[0].dataset.extension, 'a')
})

test('an empty or missing list is a no-op', () => {
  const doc = fakeDoc()
  assert.deepEqual(mountExtensionScripts([], doc), [])
  assert.deepEqual(mountExtensionScripts(undefined, doc), [])
  assert.equal(doc.appended.length, 0)
})
