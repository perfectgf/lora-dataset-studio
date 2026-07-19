import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const sourceRoot = path.resolve(here, '../src')
const files = [
  'pages/BankPage.jsx',
  'components/bank/BankWorkspace.jsx',
  'components/bank/DupGroupsPanel.jsx',
  'components/bank/PromoteDialog.jsx',
  'components/common/FolderPicker.jsx',
  'components/settings/CaptioningSection.jsx',
]

const forbiddenUiLiterals = [
  'Create bank',
  'Scan quality',
  'Rescan all',
  'Group by person',
  'Find watermarks',
  'Auto-reject flagged',
  'Promote to dataset',
  'Loading duplicate groups',
  'Keep best',
  'Choose a folder',
  'Image bank triage',
]

test('image-bank surfaces use the shared translation catalog', () => {
  for (const relativePath of files) {
    const source = fs.readFileSync(path.join(sourceRoot, relativePath), 'utf8')
    assert.match(source, /useI18n\(\)/, `${relativePath} must use the shared i18n context`)
  }
})

test('image-bank surfaces do not restore known hard-coded English UI labels', () => {
  const violations = []
  for (const relativePath of files) {
    const source = fs.readFileSync(path.join(sourceRoot, relativePath), 'utf8')
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .replace(/^\s*\/\/.*$/gm, '')
    for (const literal of forbiddenUiLiterals) {
      if (source.includes(literal)) violations.push(`${relativePath}: ${literal}`)
    }
  }
  assert.deepEqual(violations, [])
})
