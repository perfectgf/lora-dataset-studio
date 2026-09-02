import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { videoDatasetClipComparisonUrl } from './videoBankApi.js'
import { clipComparisonUrl } from '../dataset/studio/video/videoStudioApi.js'

/* ⬇ Export lives in ONE component and BOTH surfaces reach it — the parity rule
   applied to a verb rather than a pass. These read the hosts as text, which is
   all node --test can do with JSX; what they can still prove is that neither
   host was left behind when the other got the button. */

const read = (p) => readFileSync(new URL(p, import.meta.url), 'utf8')

test('the export URL of each surface is the route that serves it', () => {
  assert.equal(videoDatasetClipComparisonUrl(7, 12),
    '/api/video-dataset/7/clip/12/comparison')
  assert.equal(clipComparisonUrl(43), '/api/video-studio/clip/43/comparison')
})

test('both hosts of the comparison hand it an export URL', () => {
  const lightbox = read('./VideoDatasetLightbox.jsx')
  const studio = read('../dataset/studio/video/VideoTestStudio.jsx')
  for (const [name, src, builder] of [
    ['the dataset lightbox', lightbox, 'videoDatasetClipComparisonUrl'],
    ['the video test studio', studio, 'clipComparisonUrl'],
  ]) {
    const tag = src.slice(src.indexOf('<SideBySideVideo'))
    assert.match(tag.slice(0, 400), /exportHref=/, `${name} passes no exportHref`)
    assert.ok(src.includes(builder), `${name} does not import ${builder}`)
  }
})

test('the button only exists when a host offers the URL, and says it is working', () => {
  const src = read('./SideBySideVideo.jsx')
  assert.match(src, /\{exportHref && \(/)      // no prop, no button
  assert.match(src, /exporting \? 'Building…' : '⬇ Export'/)
  assert.match(src, /disabled=\{exporting\}/)  // one click, not five
  assert.match(src, /role="alert"/)            // the failure is on screen
  // Finger-sized below lg, like every other control in this layer.
  assert.ok(src.includes('min-h-10 rounded-md border border-border px-2 py-1 text-xs'))
})
