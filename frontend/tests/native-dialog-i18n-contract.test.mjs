import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const sourceRoot = path.resolve(here, '../src')

function sourceFiles(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const fullPath = path.join(dir, entry.name)
    if (entry.isDirectory()) return sourceFiles(fullPath)
    return /\.(?:js|jsx)$/.test(entry.name) ? [fullPath] : []
  })
}

test('native browser dialogs do not receive hard-coded string literals', () => {
  const hardCodedDialog = /window\.(?:confirm|prompt|alert)\(\s*[`'"]/m
  const violations = sourceFiles(sourceRoot)
    .filter((file) => hardCodedDialog.test(fs.readFileSync(file, 'utf8')))
    .map((file) => path.relative(sourceRoot, file))

  assert.deepEqual(violations, [])
})
