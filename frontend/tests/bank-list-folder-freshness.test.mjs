import assert from 'node:assert/strict'
import test from 'node:test'

import { createElement, render, renderToStaticMarkup } from './support/mountJsx.mjs'

// The JSX loader is registered while mountJsx is evaluated, so these imports
// have to stay dynamic (see support/mountJsx.mjs).
const { default: FolderCheckLine } =
  await import('../src/components/bank/FolderCheckLine.jsx')
const { default: BankPage } = await import('../src/pages/BankPage.jsx')
const { ToastProvider } = await import('../src/components/common/Toast.jsx')
const { MemoryRouter } = await import('react-router')

const sync = (over = {}) => ({
  folder_sync: {
    added: 0, missing: 0, unavailable: false, error: null,
    walked: false, age: null, ...over,
  },
})

// ── The trade this page now makes ──────────────────────────────────────────
// GET /api/banks stopped re-walking every bank's source folder before
// answering: on a real library (8 banks / 86 493 images) that was 690-1 190 ms
// of disk work per navigation, paid by anyone merely passing through the page.
// The walk moved to opening a bank, and to this button. The counts can
// therefore be late — and a list that is silently late is worse than one that
// is slow, so the line below is part of the fix, not decoration.
test('a never-walked list warns that its counts can lag AND offers the walk', () => {
  const html = render(FolderCheckLine, { banks: [sync(), sync()] })
  assert.match(html, /counts below are what the app knew last time/i)
  assert.match(html, /🔄 Rescan folders/)
  assert.match(html, /<button[^>]*type="button"/)
})

test('a freshly walked list states its age instead of crying stale', () => {
  const html = render(FolderCheckLine, { banks: [sync({ walked: true, age: 8 })] })
  assert.match(html, /Source folders checked just now/)
  assert.doesNotMatch(html, /knew last time/i)
})

test('the button says it is working and refuses a second click', () => {
  const html = render(FolderCheckLine, { banks: [sync()], busy: true })
  assert.match(html, /Checking folders…/)
  assert.match(html, /<button[^>]*disabled/)
})

test('no bank, no line — nothing to be honest about', () => {
  assert.equal(render(FolderCheckLine, { banks: [] }), '')
  assert.equal(render(FolderCheckLine, { banks: null }), '')
})

// The HOST, not just the piece: a dependency array or a const read before its
// declaration parses fine and throws on the user's screen. Rendering BankPage
// executes it.
test('the bank page itself still renders (no ReferenceError on the way in)', () => {
  const html = renderToStaticMarkup(createElement(
    MemoryRouter, null,
    createElement(ToastProvider, null, createElement(BankPage))))
  assert.match(html, /Image bank/)
})
