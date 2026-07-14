import assert from 'node:assert/strict'
import { readdir, readFile } from 'node:fs/promises'
import { dirname, extname, relative, resolve, sep } from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const TEST_DIR = dirname(fileURLToPath(import.meta.url))
const FRONTEND_DIR = resolve(TEST_DIR, '..')
const SRC_DIR = resolve(TEST_DIR, '../src')
const INDEX_HTML = resolve(FRONTEND_DIR, 'index.html')
const SOURCE_EXTENSIONS = new Set(['.js', '.jsx', '.ts', '.tsx'])

// These semantic tokens already bake their opacity into tailwind.config.js.
// Adding Tailwind's /NN modifier replaces that safe alpha with NN%, turning a
// dark surface or hairline border into an opaque white layer.
const INVALID_ALPHA_MODIFIER = /(?<![A-Za-z0-9_-])((?:bg-(?:surface|surface-raised)|border-(?:border|border-strong))\/(?:\[[^\]\s"'`]+\]|\d+))(?![A-Za-z0-9_/-])/g

async function sourceFiles(dir) {
  const entries = await readdir(dir, { withFileTypes: true })
  const nested = await Promise.all(entries.map((entry) => {
    const path = resolve(dir, entry.name)
    if (entry.isDirectory()) return sourceFiles(path)
    return SOURCE_EXTENSIONS.has(extname(entry.name)) ? [path] : []
  }))
  return nested.flat()
}

async function guardedContentFiles() {
  return [INDEX_HTML, ...await sourceFiles(SRC_DIR)]
}

function invalidTokens(text) {
  return [...text.matchAll(INVALID_ALPHA_MODIFIER)].map((match) => match[1])
}

test('theme-token guard covers index.html and only Tailwind source inputs', async () => {
  const files = await guardedContentFiles()

  assert.ok(files.includes(INDEX_HTML), 'frontend/index.html is not guarded')
  assert.ok(
    files.every((file) => file === INDEX_HTML || file.startsWith(`${SRC_DIR}${sep}`)),
    'guard must not scan dist, node_modules, or unrelated frontend files',
  )
})

test('invalid token matcher accepts common punctuation as a token boundary', () => {
  const cases = [
    ['bg-surface/60.', 'bg-surface/60'],
    ['bg-surface-raised/50,', 'bg-surface-raised/50'],
    ['(border-border/40)', 'border-border/40'],
    ['border-border-strong/20;', 'border-border-strong/20'],
  ]

  for (const [text, expected] of cases) {
    assert.deepEqual(invalidTokens(text), [expected], text)
  }
})

test('invalid token matcher does not truncate longer class-like names', () => {
  assert.deepEqual(
    invalidTokens('x-bg-surface/60 bg-surface-overlay/60 border-border-extra/50'),
    [],
  )
})

test('alpha-embedded theme tokens never use Tailwind opacity modifiers', async () => {
  const violations = []

  for (const file of await guardedContentFiles()) {
    const lines = (await readFile(file, 'utf8')).split(/\r?\n/)
    lines.forEach((line, index) => {
      // Tailwind's content scanner also extracts class-looking tokens from
      // comments, so documentation must respect the same invariant.
      for (const token of invalidTokens(line)) {
        violations.push(`${relative(FRONTEND_DIR, file)}:${index + 1} ${token}`)
      }
    })
  }

  assert.equal(
    violations.length,
    0,
    `Opacity modifiers override the baked alpha of semantic theme tokens:\n${violations.join('\n')}`,
  )
})
