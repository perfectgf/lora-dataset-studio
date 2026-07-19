import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const frontend = path.resolve(here, '..')
const catalog = fs.readFileSync(
  path.join(frontend, 'src/components/dataset/VariationCatalog.jsx'), 'utf8')
const workspace = fs.readFileSync(
  path.join(frontend, 'src/components/dataset/DatasetWorkspace.jsx'), 'utf8')

test('a live generation turns the primary action into a clickable stop control', () => {
  assert.match(catalog, /onClick=\{generating \? onCancelGeneration : go\}/)
  assert.match(catalog, /generating\s*\? cancellingGeneration\s*:\s*busy/)
  assert.match(catalog, /bg-red-600 hover:bg-red-500/)
})

test('workspace wires both stop controls without disabling them on global busy', () => {
  assert.match(workspace, /onCancelGeneration=\{ds\.cancelPending\}/)
  assert.match(workspace, /disabled=\{ds\.cancellingGeneration\}/)
  assert.doesNotMatch(workspace, /onClick=\{ds\.cancelPending\} disabled=\{ds\.busy\}/)
})
