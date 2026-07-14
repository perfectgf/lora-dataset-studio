import assert from 'node:assert/strict'
import { readdir, readFile } from 'node:fs/promises'
import { dirname, extname, relative, resolve } from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const TEST_DIR = dirname(fileURLToPath(import.meta.url))
const SRC_DIR = resolve(TEST_DIR, '../src')
const SOURCE_EXTENSIONS = new Set(['.js', '.jsx', '.ts', '.tsx'])

// These semantic tokens already bake their opacity into tailwind.config.js.
// Adding Tailwind's /NN modifier replaces that safe alpha with NN%, turning a
// dark surface or hairline border into an opaque white layer.
const INVALID_ALPHA_MODIFIER = /\b((?:(?:[a-z-]+):)*(?:bg-(?:surface|surface-raised)|border-(?:border|border-strong))\/(?:\[[^\]\s"'`]+\]|\d+))(?=$|[\s"'`}])/g

async function sourceFiles(dir) {
  const entries = await readdir(dir, { withFileTypes: true })
  const nested = await Promise.all(entries.map((entry) => {
    const path = resolve(dir, entry.name)
    if (entry.isDirectory()) return sourceFiles(path)
    return SOURCE_EXTENSIONS.has(extname(entry.name)) ? [path] : []
  }))
  return nested.flat()
}

test('alpha-embedded theme tokens never use Tailwind opacity modifiers', async () => {
  const violations = []

  for (const file of await sourceFiles(SRC_DIR)) {
    const lines = (await readFile(file, 'utf8')).split(/\r?\n/)
    lines.forEach((line, index) => {
      // Tailwind's content scanner also extracts class-looking tokens from
      // comments, so documentation must respect the same invariant.
      for (const match of line.matchAll(INVALID_ALPHA_MODIFIER)) {
        violations.push(`${relative(SRC_DIR, file)}:${index + 1} ${match[1]}`)
      }
    })
  }

  assert.equal(
    violations.length,
    0,
    `Opacity modifiers override the baked alpha of semantic theme tokens:\n${violations.join('\n')}`,
  )
})
