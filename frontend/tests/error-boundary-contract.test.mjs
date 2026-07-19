import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const source = readFileSync(
  new URL('../src/components/common/ErrorBoundary.jsx', import.meta.url),
  'utf8',
)

test('the root error fallback exposes the real render failure', () => {
  assert.match(source, /getDerivedStateFromError\(error\)/)
  assert.match(source, /this\.state\.error\?\.message/)
  assert.match(source, /errorBoundary\.technicalDetails/)
  assert.match(source, /errorBoundary\.copyDiagnostic/)
})

test('the root error fallback can recover without forcing a reload', () => {
  assert.match(source, /errorBoundary\.tryAgain/)
  assert.match(source, /errorBoundary\.backToDatasets/)
  assert.match(source, /window\.location\.hash = '#\/datasets'/)
})
