/** Contract for opening the source folder shown in BankWorkspace. */
import assert from 'node:assert/strict'
import { readSource } from './support/readSource.mjs'
import test from 'node:test'

const read = readSource
const workspace = read('src/components/bank/BankWorkspace.jsx')

test('the Bank posts only its id-scoped endpoint, never the displayed path', () => {
  const handler = workspace.slice(
    workspace.indexOf('const openSourceFolder'),
    // runFolderPerson moved to useFolderPersons (hook wave 4); the next
    // stable line after the handler is this comment.
    workspace.indexOf('// Leaving the selection view'),
  )
  assert.match(handler,
    /postJson\(`\/api\/bank\/\$\{bankId\}\/open-source-folder`, \{\}\)/)
  assert.doesNotMatch(handler, /source_path|payload/)
})

test('the source-path row carries a clear, request-busy button', () => {
  assert.match(workspace,
    /const \[openingSourceFolder, setOpeningSourceFolder\] = useState\(false\)/)
  const sourceRow = workspace.slice(
    workspace.indexOf('{payload?.source_path && ('),
    workspace.indexOf('{/* A bank created before this was refused'),
  )
  assert.match(sourceRow, /onClick=\{openSourceFolder\}/)
  assert.match(sourceRow, /disabled=\{openingSourceFolder\}/)
  assert.match(sourceRow, /aria-busy=\{openingSourceFolder\}/)
  assert.match(sourceRow, /openingSourceFolder \? 'Opening…' : 'Open folder'/)
})

test('a failed native-folder launch is visible and always clears busy state', () => {
  const handler = workspace.slice(
    workspace.indexOf('const openSourceFolder'),
    // runFolderPerson moved to useFolderPersons (hook wave 4); the next
    // stable line after the handler is this comment.
    workspace.indexOf('// Leaving the selection view'),
  )
  assert.match(handler, /catch \(e\) \{[\s\S]*?toast\.error/)
  assert.match(handler, /finally \{[\s\S]*?setOpeningSourceFolder\(false\)/)
})
