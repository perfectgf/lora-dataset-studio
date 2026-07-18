import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import {
  helpTopics, getHelpTopic, helpTopicsForChapter, searchHelpTopics, helpTips,
} from '../src/help/helpRegistry.js'
import { markdownHeadingId } from '../src/utils/headingId.js'
import { SETTINGS_SECTIONS } from '../src/components/settings/registry.js'
import { WORKSPACE_SECTIONS } from '../src/components/dataset/workspaceSections.js'
import { getWorkspacePanel } from '../src/components/dataset/workspaceNavigation.js'
import { buildGuideTextIndex, matchGuideAnchors } from '../src/help/guideTextIndex.js'
import { shouldShowTip, markTipSeen } from '../src/help/helpTips.js'

// ---- helpers ---------------------------------------------------------------

const read = (rel) => readFileSync(new URL(rel, import.meta.url), 'utf8')

// Chapter id → its markdown source (same mapping GuidePage imports as ?raw).
const CHAPTER_MD = {
  'getting-started': '../../docs/guide/getting-started.md',
  'using-the-app': '../../docs/guide/using-the-app.md',
  'dataset-guide': '../../docs/DATASET_GUIDE.md',
  'settings-reference': '../../docs/guide/settings-reference.md',
  'troubleshooting': '../../docs/guide/troubleshooting.md',
  'getting-help': '../../docs/guide/getting-help.md',
}

// The H2 anchor set of a chapter, computed with the SAME markdownHeadingId the
// app uses — the registry's anchors are validated against exactly this.
const chapterAnchors = (chapterId) => {
  const md = read(CHAPTER_MD[chapterId])
  return new Set([...md.matchAll(/^##\s+(.+)$/gm)].map((m) => markdownHeadingId(m[1])))
}
const anchorCache = new Map()
const anchorsFor = (chapterId) => {
  if (!anchorCache.has(chapterId)) anchorCache.set(chapterId, chapterAnchors(chapterId))
  return anchorCache.get(chapterId)
}

const SETTINGS_IDS = new Set(SETTINGS_SECTIONS.map((s) => s.id))
const WORKSPACE_IDS = new Set(WORKSPACE_SECTIONS.map((s) => s.id))
const STATIC_ROUTES = new Set(['/datasets', '/setup', '/settings', '/studio', '/cloud', '/help'])

// route ∈ {static} OR /settings/<settings-id> OR /datasets?section=<ws-id>[&panel=<panel>]
const routeValid = (route) => {
  const [path, qs] = route.split('?')
  if (!qs) {
    if (STATIC_ROUTES.has(path)) return true
    const m = path.match(/^\/settings\/([a-z0-9-]+)$/)
    return !!(m && SETTINGS_IDS.has(m[1]))
  }
  if (path !== '/datasets') return false
  const params = new URLSearchParams(qs)
  const section = params.get('section')
  if (!section || !WORKSPACE_IDS.has(section)) return false
  const panel = params.get('panel')
  if (panel && !getWorkspacePanel(section, panel)) return false
  return true
}

// Recursively collect every .js/.jsx source under frontend/src (never tests).
const walk = (dirUrl) => {
  const out = []
  for (const entry of readdirSync(dirUrl, { withFileTypes: true })) {
    const child = new URL(`${entry.name}${entry.isDirectory() ? '/' : ''}`, dirUrl)
    if (entry.isDirectory()) out.push(...walk(child))
    else if (/\.(jsx?|mjs)$/.test(entry.name)) out.push(readFileSync(fileURLToPath(child), 'utf8'))
  }
  return out
}
const SRC = walk(new URL('../src/', import.meta.url)).join('\n')

// DOM ids declared in the Settings components: literal id="…" plus SecretField
// keys (they render id={f.key}, so the config key IS the DOM id).
const settingsDomIds = () => {
  const dir = new URL('../src/components/settings/', import.meta.url)
  let src = ''
  for (const f of readdirSync(dir)) if (f.endsWith('.jsx')) src += read(`../src/components/settings/${f}`) + '\n'
  const ids = new Set()
  for (const m of src.matchAll(/id="([^"]+)"/g)) ids.add(m[1])
  for (const m of src.matchAll(/\bkey:\s*'([^']+)'/g)) ids.add(m[1])
  return ids
}

// ---- (1) shape -------------------------------------------------------------

test('(1) topics have unique ids, non-empty title and keywords', () => {
  const ids = new Set()
  for (const t of helpTopics) {
    assert.ok(t.id, 'topic missing id')
    assert.ok(!ids.has(t.id), `duplicate id ${t.id}`)
    ids.add(t.id)
    assert.ok(typeof t.title === 'string' && t.title.trim(), `${t.id}: empty title`)
    assert.ok(Array.isArray(t.keywords) && t.keywords.length > 0, `${t.id}: empty keywords`)
    assert.ok(t.keywords.every((k) => typeof k === 'string' && k.trim()), `${t.id}: blank keyword`)
    assert.ok(['section', 'setting', 'action', 'page'].includes(t.kind), `${t.id}: bad kind ${t.kind}`)
  }
})

// ---- (2) guide chapter + anchor -------------------------------------------

test('(2) every guide.chapter is known and anchor is a real H2', () => {
  for (const t of helpTopics) {
    assert.ok(CHAPTER_MD[t.guide.chapter], `${t.id}: unknown chapter ${t.guide.chapter}`)
    assert.ok(anchorsFor(t.guide.chapter).has(t.guide.anchor),
      `${t.id}: anchor #${t.guide.anchor} not an H2 of ${t.guide.chapter}`)
  }
})

// ---- (3) app route ---------------------------------------------------------

test('(3) every app.route is a valid destination', () => {
  for (const t of helpTopics) {
    assert.ok(routeValid(t.app.route), `${t.id}: invalid route ${t.app.route}`)
  }
})

// ---- (4) app focus maps to a real DOM id ----------------------------------

test('(4) every app.focus exists as a Settings DOM id', () => {
  const ids = settingsDomIds()
  for (const t of helpTopics) {
    if (!t.app.focus) continue
    assert.ok(ids.has(t.app.focus), `${t.id}: focus id "${t.app.focus}" not found in settings/*.jsx`)
  }
})

// ---- (5) coverage ----------------------------------------------------------

test('(5) each Settings section and Workspace section has its topic', () => {
  for (const s of SETTINGS_SECTIONS) {
    assert.ok(getHelpTopic(`settings-${s.id}`), `missing topic settings-${s.id}`)
  }
  for (const s of WORKSPACE_SECTIONS) {
    assert.ok(getHelpTopic(`workspace-${s.id}`), `missing topic workspace-${s.id}`)
  }
})

// ---- (6) tips --------------------------------------------------------------

test('(6) tips have unique triggers and non-empty text', () => {
  const tips = helpTips()
  assert.equal(tips.length, 9, 'expected exactly 9 one-time tips')
  const triggers = new Set()
  for (const tip of tips) {
    assert.ok(tip.trigger, 'tip missing trigger')
    assert.ok(!triggers.has(tip.trigger), `duplicate trigger ${tip.trigger}`)
    triggers.add(tip.trigger)
    assert.ok(typeof tip.text === 'string' && tip.text.trim(), `${tip.trigger}: empty text`)
  }
})

// ---- (7) instrumentation references resolve --------------------------------

test('(7) every topic="…" / requestHelpTip(\'…\') in src resolves', () => {
  const triggers = new Set(helpTips().map((t) => t.trigger))
  for (const m of SRC.matchAll(/\btopic="([^"]+)"/g)) {
    assert.ok(getHelpTopic(m[1]), `topic="${m[1]}" referenced in JSX but not in registry`)
  }
  for (const m of SRC.matchAll(/requestHelpTip\(\s*'([^']+)'/g)) {
    assert.ok(triggers.has(m[1]), `requestHelpTip('${m[1]}') has no matching tip trigger`)
  }
})

// ---- (8) search regressions ------------------------------------------------

test('(8) search matches settings by keyword/id', () => {
  assert.ok(searchHelpTopics('crop').some((t) => t.id === 'watermark.allow_crop'),
    "'crop' should surface watermark.allow_crop")
  assert.ok(searchHelpTopics('preset').some((t) => t.id === 'klein.generation_lora_presets'),
    "'preset' should surface klein.generation_lora_presets")
  assert.ok(searchHelpTopics('abliterated').some((t) => t.id === 'ollama.vision_model'),
    "'abliterated' should surface ollama.vision_model")
  assert.ok(searchHelpTopics('short caption').some((t) => t.id === 'training.dual_captions'),
    "'short caption' should surface training.dual_captions")
})

// ---- (9) GuidePage registers the settings-reference chapter -----------------

test('(9) GuidePage registers the settings-reference chapter before troubleshooting', () => {
  const guide = read('../src/pages/GuidePage.jsx')
  assert.match(guide, /import settingsReference from '[^']*settings-reference\.md\?raw'/)
  assert.match(guide, /\{ id: 'settings-reference', num: '04',[^\n]*source: settingsReference \}/)
  // The pre-existing help-navigation contract still holds: HELP_CHAPTER shape and
  // CHAPTERS excluding getting-help.
  assert.match(guide, /const HELP_CHAPTER = [^\n]*source: gettingHelp, extra: 'diagnostic'/)
  const chaptersBlock = guide.match(/const CHAPTERS = \[([\s\S]*?)\n\]/)?.[1] || ''
  assert.doesNotMatch(chaptersBlock, /getting-help/)
})

// ---- helpTopicsForChapter ordering (Open-this-screen picks the section) -----

test('helpTopicsForChapter preserves registry order; section topic wins its anchor', () => {
  const chap = helpTopicsForChapter('using-the-app')
  const firstForCharacterWalkthrough = chap.find(
    (t) => t.guide.anchor === 'the-character-walkthrough-reference-photo-trained-lora')
  assert.equal(firstForCharacterWalkthrough.id, 'workspace-images')
})

// ---- guideTextIndex unit tests --------------------------------------------

test('buildGuideTextIndex splits by H2 and matchGuideAnchors is case-insensitive', () => {
  const md = [
    '# Title', 'intro line',
    '## First Section', 'Here we explain auto CROP behaviour.',
    '## Second Section', 'nothing relevant',
  ].join('\n')
  const idx = buildGuideTextIndex(md)
  assert.equal(idx.length, 2)
  assert.equal(idx[0].anchor, 'first-section')
  assert.ok(idx[0].text.includes('crop'))            // lowercased body
  const hits = matchGuideAnchors(idx, 'CROP')
  assert.ok(hits.has('first-section'))
  assert.ok(!hits.has('second-section'))
  assert.equal(matchGuideAnchors(idx, '').size, 0)   // empty query → no hits
})

test('buildGuideTextIndex ignores H2 inside code fences', () => {
  const md = ['## Real', 'body', '```', '## Not a heading', '```'].join('\n')
  const idx = buildGuideTextIndex(md)
  assert.deepEqual(idx.map((s) => s.anchor), ['real'])
})

// ---- helpTips unit tests (injected store) ---------------------------------

const memoryStore = () => {
  const data = {}
  return { getItem: (k) => (k in data ? data[k] : null), setItem: (k, v) => { data[k] = v } }
}

test('helpTips: shouldShowTip flips after markTipSeen (injected store)', () => {
  const store = memoryStore()
  assert.equal(shouldShowTip('topic-a', store), true)
  markTipSeen('topic-a', store)
  assert.equal(shouldShowTip('topic-a', store), false)
  markTipSeen('topic-a', store)                       // idempotent
  assert.equal(shouldShowTip('topic-a', store), false)
  assert.equal(shouldShowTip('topic-b', store), true) // unrelated topic unaffected
})

test('helpTips: markTipSeen persists a JSON map in the store', () => {
  const store = memoryStore()
  markTipSeen('x', store)
  markTipSeen('y', store)
  const raw = JSON.parse(store.getItem('ldsHelpTipsSeen'))
  assert.deepEqual(raw, { x: true, y: true })
})
