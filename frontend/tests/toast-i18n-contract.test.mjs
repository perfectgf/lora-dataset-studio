import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const source = readFileSync(
  new URL('../src/components/common/Toast.jsx', import.meta.url),
  'utf8',
)

test('toast items do not shadow the translation function', () => {
  assert.match(source, /const \{ t \} = useI18n\(\)/)
  assert.match(source, /toasts\.map\(\(toastItem\) =>/)
  assert.doesNotMatch(source, /toasts\.map\(\(t\) =>/)
  assert.match(source, /aria-label=\{t\('toast\.close'\)\}/)
})
