import assert from 'node:assert/strict'
import fs from 'node:fs'
import test from 'node:test'

const app = fs.readFileSync(new URL('../src/App.jsx', import.meta.url), 'utf8')

test('desktop navigation uses the available width and keeps labels on one line', () => {
  assert.match(app, /NAV_ITEM_BASE\s*=\s*\n\s*'whitespace-nowrap /)
  assert.match(app, /mx-auto flex w-full max-w-screen-2xl items-center justify-center/)
  assert.match(app, /hidden items-center gap-1 md:flex/)
})

test('the compact menu keeps the original md breakpoint', () => {
  assert.match(app, /ml-auto flex items-center gap-1 md:hidden/)
  assert.match(app, /flex flex-col gap-1 border-t border-border px-4 py-2 md:hidden/)
})
