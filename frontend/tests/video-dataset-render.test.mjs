/**
 * 🎬 The video dataset WORKSPACE, rendered — the constraints only a real render
 * can pin.
 *
 *  · THE GRID CONTAINS NO <video>, and here that is harder to hold than on the
 *    bank. The bank had an honest excuse — no clip file exists before promotion,
 *    so there is nothing to point a player at. A dataset's files DO exist, each
 *    a few megabytes, and one <video preload="none"> per tile looks perfectly
 *    reasonable in review. It is not: Chrome caps WebMediaPlayers at about sixty
 *    across the whole browser and past that new elements never load and never
 *    error, so a 128-clip set breaks on somebody else's second screen of scroll
 *    with nothing in the console. Counting the elements in the markup is the
 *    only check that survives a rewrite of the component.
 *
 *  · THE LIGHTBOX MOUNTS EXACTLY ONE, and an <img> instead when the clip is a
 *    still. A stills set is served through the same route by the same table;
 *    wrapping a .png in a <video> renders a dead player, which is how it shipped
 *    the day stills landed and was found on a phone.
 *
 *  · THE WORKSPACE RENDERS AT ALL, in the states that branch: with and without
 *    references, with an empty set, with the caption tools open. `renderToStatic
 *    Markup` executes every branch it reaches, so a TDZ or a stale identifier
 *    throws here instead of on a user's screen.
 *
 * ⚠️ This file MOUNTS components, so it needs `frontend/node_modules` (react +
 * esbuild). Git worktrees of this repo do not have one unless it was linked in
 * — there it fails with `Cannot find package 'react'`, which is a missing
 * dependency and not a regression.
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import { readSource } from './support/readSource.mjs'

import { createElement, render, renderToStaticMarkup } from './support/mountJsx.mjs'

const { MemoryRouter } = await import('react-router')
const { default: VideoDatasetGrid } =
  await import('../src/components/videobank/VideoDatasetGrid.jsx')
const { default: VideoDatasetLightbox } =
  await import('../src/components/videobank/VideoDatasetLightbox.jsx')
const { default: VideoDatasetWorkspace } =
  await import('../src/components/videobank/VideoDatasetWorkspace.jsx')
const { ToastProvider } = await import('../src/components/common/Toast.jsx')

const countTag = (html, tag) => (html.match(new RegExp(`<${tag}[\\s>]`, 'g')) || []).length
const srcsOf = (html) => [...html.matchAll(/<img[^>]*\ssrc="([^"]+)"/g)].map((m) => m[1])

const CLIPS = [
  { id: 1, filename: 'clip_0001.mp4', caption: 'a woman walks through a hall',
    source_bank_id: 4, source_clip_id: 11, src_relpath: 'day1/a.mp4', start_s: 0, end_s: 5 },
  // No caption: the tile has to SAY so — that is the working list.
  { id: 2, filename: 'clip_0002.mp4', caption: null,
    source_bank_id: 4, source_clip_id: 12, src_relpath: 'day1/a.mp4', start_s: 41.25, end_s: 46.5 },
  // No provenance at all: the bank was deleted, or this came from somewhere
  // else. An ordinary state, and it must draw a placeholder rather than a 404.
  { id: 3, filename: 'clip_0003.mp4', caption: 'a car turns',
    source_bank_id: null, source_clip_id: null, src_relpath: null, start_s: 0, end_s: 4 },
  // A STILL. Its own frame is the poster, served by the media route.
  { id: 4, filename: 'clip_0004.png', caption: 'a portrait',
    source_bank_id: null, source_clip_id: null, src_relpath: null, start_s: 0, end_s: 0 },
]

const DS = {
  id: 9, name: 'City rushes', target_profile: 'wan22_14b', target_label: 'Wan 2.2 14B',
  fps: 16, frames: 81, clip_seconds: 5.0, width: 832, height: 480,
  output_dir: 'X:/sets/city', clips: 4, suggested_steps: 2000,
  training_verified: true, licence_note: null, trigger_word: 'sks_city',
  references: 0, requires_references: false, items: CLIPS,
}

const renderWorkspace = (props) => renderToStaticMarkup(
  createElement(MemoryRouter, null,
    createElement(ToastProvider, null,
      createElement(VideoDatasetWorkspace, {
        ds: DS, items: CLIPS, refresh: () => {}, onBack: () => {}, ...props,
      }))))

// ---- (1) the grid holds no player, in any state ------------------------------

test('the clip grid renders NOT ONE <video>', () => {
  const html = render(VideoDatasetGrid, {
    datasetId: 9, clips: CLIPS, selected: [2], onToggle: () => {}, onOpen: () => {},
    emptyMessage: 'nothing',
  })
  assert.equal(countTag(html, 'video'), 0, 'the grid must contain zero <video> elements')
  assert.ok(!html.includes('<source'), 'no <source> either — that is a player by another name')
})

test('the grid source declares no video element in ANY branch', () => {
  // The mount only covers the states this file passes it; the source check
  // covers the ones nobody rendered. Comments stripped, because the file's own
  // docstring quotes the forbidden version in order to rule it out.
  const src = readSource('src/components/videobank/VideoDatasetGrid.jsx')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/^\s*\/\/.*$/gm, '')
  assert.ok(!/<video[\s>]/.test(src), 'VideoDatasetGrid.jsx must not render a <video> tag')
})

test('each tile takes its poster from the cheapest place that really has one', () => {
  const html = render(VideoDatasetGrid, {
    datasetId: 9, clips: CLIPS, selected: [], onToggle: () => {}, onOpen: () => {},
    emptyMessage: 'nothing',
  })
  const srcs = srcsOf(html)
  // Three <img>: two bank thumbnails and the still's own frame. The clip with
  // no provenance draws the placeholder instead.
  assert.equal(srcs.length, 3, `expected 3 posters, got ${srcs.length}: ${srcs.join(' ')}`)
  assert.deepEqual(srcs, [
    '/api/video-bank/4/clip/11/thumb',
    '/api/video-bank/4/clip/12/thumb',
    '/api/video-dataset/9/clip/4/media',
  ])
  for (const img of html.match(/<img[^>]*>/g) || []) {
    assert.match(img, /loading="lazy"/, `poster not lazy: ${img}`)
  }
  assert.ok(html.includes('🎞'), 'the clip with no thumbnail must draw a placeholder')
})

test('the tile marks what is MISSING — a caption — and not what is present', () => {
  const html = render(VideoDatasetGrid, {
    datasetId: 9, clips: CLIPS, selected: [], onToggle: () => {}, onOpen: () => {},
    emptyMessage: 'nothing',
  })
  assert.equal((html.match(/no caption/g) || []).length, 1,
    'exactly one badge, on the one clip without a caption')
})

test('an empty grid renders its explanation, not an empty box', () => {
  const html = render(VideoDatasetGrid, {
    datasetId: 9, clips: [], selected: [], onToggle: () => {}, onOpen: () => {},
    emptyMessage: 'promote shots from a video bank',
  })
  assert.ok(html.includes('promote shots from a video bank'))
})

// ---- (2) the lightbox is where the ONE player is spent -----------------------

test('the lightbox mounts exactly one <video>, pointed at this dataset clip', () => {
  const html = renderToStaticMarkup(createElement(VideoDatasetLightbox, {
    datasetId: 9, clip: CLIPS[0], caption: 'a woman walks through a hall',
    onCaptionChange: () => {}, onSave: () => {}, onClose: () => {},
    onPrev: () => {}, onNext: () => {}, onRemove: () => {},
    hasPrev: false, hasNext: true, saving: false,
  }))
  assert.equal(countTag(html, 'video'), 1)
  assert.ok(html.includes('/api/video-dataset/9/clip/1/media'))
  // Provenance is on screen: it is what you take back to the bank to re-cut.
  assert.ok(html.includes('day1/a.mp4'))
})

test('a STILL is drawn as an image — a <video> around a .png is a dead player', () => {
  const html = renderToStaticMarkup(createElement(VideoDatasetLightbox, {
    datasetId: 9, clip: CLIPS[3], caption: 'a portrait',
    onCaptionChange: () => {}, onSave: () => {}, onClose: () => {},
    onPrev: () => {}, onNext: () => {}, onRemove: () => {},
    hasPrev: true, hasNext: false, saving: false,
  }))
  assert.equal(countTag(html, 'video'), 0)
  assert.deepEqual(srcsOf(html), ['/api/video-dataset/9/clip/4/media'])
})

test('the caption box names the .txt it writes, by the clip’s own basename', () => {
  const html = renderToStaticMarkup(createElement(VideoDatasetLightbox, {
    datasetId: 9, clip: CLIPS[1], caption: '',
    onCaptionChange: () => {}, onSave: () => {}, onClose: () => {},
    onPrev: () => {}, onNext: () => {}, onRemove: () => {},
    hasPrev: true, hasNext: true, saving: false,
  }))
  assert.ok(html.includes('clip_0002.txt'),
    'the editor must name the file it rewrites — that file is what the trainer reads')
})

// ---- (3) the workspace renders, in the states that branch --------------------

test('the workspace renders and the grid inside it still holds no player', () => {
  const html = renderWorkspace()
  assert.equal(countTag(html, 'video'), 0, 'no player anywhere until one is opened')
  assert.ok(html.includes('City rushes'))
  assert.ok(html.includes('X:/sets/city'), 'the clips folder is named on the page')
  // The coverage line REPORTS rather than warns, and names the trigger.
  assert.ok(html.includes('sks_city'))
  assert.match(html, /3 of 4 clips carry a caption/)
})

test('References is in the rail only for a target that trains on control images', () => {
  const plain = renderWorkspace()
  assert.ok(!plain.includes('vds-references-attach'),
    'a Wan set must not show a references section')
  const ref2va = renderWorkspace({
    ds: { ...DS, requires_references: true, target_label: 'MiniMax H3 (ref2va)' },
  })
  assert.ok(ref2va.includes('vds-references-attach'))
  assert.match(ref2va, /the launch is refused without them/)
})

test('the two things that save a wasted week are on the workspace, not only the card', () => {
  const html = renderWorkspace({
    ds: {
      ...DS,
      training_verified: false,
      licence_note: 'MiniMax H3’s licence grants no rights in the EU, UK, South Korea or USA.',
    },
  })
  assert.match(html, /No LoRA trainer is known to exist/)
  assert.match(html, /South Korea/)
})

test('an empty set explains where clips come from instead of showing an empty grid', () => {
  const html = renderWorkspace({ ds: { ...DS, clips: 0, items: [] }, items: [] })
  assert.match(html, /promote shots from a video bank/)
  assert.match(html, /No clip in this dataset yet/)
})

/** The rail's own markup, cut out of the page — never the whole document.
 *
 * The first version of the assertion below matched section titles anywhere in
 * the html, and section HEADERS are rendered from the same list: it passed with
 * ZERO rail entries, which is the one regression it existed to catch. So the nav
 * is isolated first and the buttons are counted inside it. */
const railsOf = (html) => {
  // BOTH rails: the phone chip rail and the desktop side rail carry the same
  // aria-label, and the first version of this helper took only the first one
  // — emptying the desktop rail left every assertion green.
  const out = []
  let at = html.indexOf('aria-label="Video dataset sections"')
  assert.notEqual(at, -1, 'the section rail is not on the page at all')
  while (at !== -1) {
    const from = html.lastIndexOf('<nav', at)
    const to = html.indexOf('</nav>', at)
    out.push(html.slice(from, to))
    at = html.indexOf('aria-label="Video dataset sections"', to)
  }
  assert.equal(out.length, 2, `expected the phone rail AND the desktop rail, found ${out.length}`)
  return out
}
// Every assertion below is made on EACH rail, so a regression in one cannot
// hide behind the other.
const railOf = (html) => railsOf(html).join('\n<!-- rail boundary -->\n')

test('the rail really lists every visible section, and the anchors it points at exist', async () => {
  const { VIDEO_DATASET_SECTIONS } = await import(
    '../src/components/videobank/videoDatasetSections.js')
  const html = renderWorkspace({ ds: { ...DS, requires_references: true } })
  for (const [which, rail] of railsOf(html).entries()) {
    for (const section of VIDEO_DATASET_SECTIONS) {
      // As the markup carries it: "Checkpoints & LoRAs" is escaped on the way out.
      const title = section.title.replace(/&/g, '&amp;')
      assert.ok(rail.includes(`>${title}</span>`),
        `${section.id} has no entry in rail #${which}`)
    }
    // One button per section and not one more — a rail that duplicated its
    // entries would satisfy every assertion above.
    assert.equal((rail.match(/<button/g) || []).length, VIDEO_DATASET_SECTIONS.length,
      `rail #${which} does not hold exactly one button per section`)
  }
  for (const section of VIDEO_DATASET_SECTIONS) {
    assert.ok(html.includes(`id="${section.panels[0].targetId}"`),
      `${section.id}: nothing on the page carries ${section.panels[0].targetId}`)
  }
})

test('a hidden section is absent from the RAIL, not merely from the page', () => {
  // References is the one that comes and goes. Asserting on the page would pass
  // on a rail that still offered a chip leading to an empty screen.
  const rail = railOf(renderWorkspace())
  assert.ok(!rail.includes('>References</span>'))
  assert.ok(railOf(renderWorkspace({ ds: { ...DS, requires_references: true } }))
    .includes('>References</span>'))
})

test('the caption tools are there for a set with NO caption at all', () => {
  // The state they were written for: `prefix` reaches the silent clips, so
  // gating the panel on "something already has a caption" hid it from a freshly
  // promoted set — the only kind that needs a trigger added in bulk.
  const silent = CLIPS.map((c) => ({ ...c, caption: null }))
  const html = renderWorkspace({ ds: { ...DS, items: silent }, items: silent })
  assert.ok(html.includes('id="vds-captions-tools"'),
    'Caption tools must exist on an entirely uncaptioned set')
  assert.match(html, /Add as prefix/)
  // …and it is gone when there is genuinely nothing to work on.
  const empty = renderWorkspace({ ds: { ...DS, items: [] }, items: [] })
  assert.ok(!empty.includes('id="vds-captions-tools"'))
})
