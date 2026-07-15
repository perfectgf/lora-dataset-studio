import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

import {
  captionCategoryCopy,
  captionFrequencyEntries,
  recaptionConfirmation,
} from '../src/components/dataset/captionCategory.js'

test('prose frequency counts useful words by caption instead of comma fragments', () => {
  const entries = captionFrequencyEntries([
    'A woman wearing a red dress, standing in a studio.',
    'A man in a red jacket standing outdoors.',
    'Red fabric fills the frame.',
  ], 'prose')
  assert.deepEqual(entries.slice(0, 2), [['red', 3], ['standing', 2]])
  assert.equal(entries.some(([term]) => term.includes('woman wearing')), false)
})

test('booru frequency keeps exact comma-separated tags', () => {
  assert.deepEqual(captionFrequencyEntries([
    'red dress, standing, studio',
    'red dress, outdoors',
  ], 'booru').slice(0, 2), [['red dress', 2], ['outdoors', 1]])
})

test('caption guidance is specific to character, concept and style datasets', () => {
  const character = captionCategoryCopy('character', 'prose')
  const concept = captionCategoryCopy('concept', 'prose')
  const style = captionCategoryCopy('style', 'prose')

  assert.match(character.frequencyHelp, /identity|character/i)
  assert.match(concept.frequencyHelp, /concept.*leak|leak check/i)
  assert.match(style.frequencyHelp, /style LoRA|aesthetic/i)
  assert.doesNotMatch(style.frequencyHelp, /your trigger/i)
  assert.equal(style.leakSummary, 'Aesthetic terms should stay out of captions')
})

test('re-caption confirmation explains the correct category rule', () => {
  assert.match(recaptionConfirmation('character', 4), /identity/i)
  assert.match(recaptionConfirmation('concept', 4), /concept/i)
  assert.match(recaptionConfirmation('style', 4), /style|aesthetic/i)
})

test('workspace forwards the dataset kind to caption tools and confirmation', () => {
  const workspace = readFileSync(
    new URL('../src/components/dataset/DatasetWorkspace.jsx', import.meta.url), 'utf8')
  assert.match(workspace, /<CaptionToolsBar[^>]*kind=\{d\.kind \|\| 'character'\}/s)
  assert.match(workspace, /recaptionConfirmation\(d\.kind \|\| 'character'/)
})
