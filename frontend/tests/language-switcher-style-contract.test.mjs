import assert from 'node:assert/strict'
import fs from 'node:fs'
import test from 'node:test'

const app = fs.readFileSync(new URL('../src/App.jsx', import.meta.url), 'utf8')

test('language menu uses an opaque dark popup without changing its sizing classes', () => {
  assert.match(app, /max-w-28[^"]*text-sm[^"]*\[color-scheme:dark\]/)
  assert.match(app, /<option[^>]*className="bg-surface-overlay text-content"/)
})
