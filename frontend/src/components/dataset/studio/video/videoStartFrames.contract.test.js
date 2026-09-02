/* The Studio queues one clip per START FRAME — the wiring that
 * videoStartFrames.test.js cannot reach, since `node --test` renders no JSX.
 *
 * Asked for from the picker (2026-09-02): "batch the image inputs". The
 * logic lives in videoStartFrames.js and is exercised there; this file pins
 * that the Studio actually uses it — the strip is the state, Generate walks
 * it, the ✨ helpers read its first frame, and the button counts.
 */
import assert from 'node:assert/strict'
import fs from 'node:fs'
import test from 'node:test'

const read = (rel) => fs.readFileSync(new URL(rel, import.meta.url), 'utf8')
const STUDIO = read('./VideoTestStudio.jsx')

test('the strip is the state, and `source` is its first frame — what the ✨ helpers read', () => {
  assert.match(STUDIO, /const \[sources, setSources\] = useState\(\[\]\);/)
  assert.match(STUDIO, /const source = sources\[0\] \|\| EMPTY_SOURCE;/)
  assert.doesNotMatch(STUDIO, /setSource\(|useState\(\{ image: null/, 'a second, single-frame state would drift from the strip')
  // The helpers still read `source.image` — the first frame — and the
  // poller still resets on it (the motion-length contract pins the lines).
  assert.match(STUDIO, /image: mode === 't2v' \? null : \(source\.image \|\| null\)/)
  assert.match(STUDIO, /useEffect\(\(\) => \{ stopWaiting\(\); \}, \[mode, source\.image, seconds, stopWaiting\]\);/)
  // Blocked on an EMPTY strip, not on a missing first image.
  assert.match(STUDIO, /const needsImage = mode === 'i2v' && sources\.length === 0;/)
})

test('Generate walks the strip through queueClips — one POST per frame, text-only one launch', () => {
  const gen = STUDIO.slice(STUDIO.indexOf('const generate = async () => {'), STUDIO.indexOf('const rate = '))
  assert.ok(gen.length > 0, 'generate is gone')
  assert.match(gen, /const launches = mode === 't2v' \? \[null\] : sources;/)
  assert.match(gen, /await queueClips\(launches, \{ enhance: enhanceOn,/)
  assert.match(gen, /\(body\) => postJson\(generateUrl\(\), body\)/)
  assert.match(gen, /\(done, total\) => setProgress\(\{ done, total \}\)/)
  // The frame travels with each launch, never from the Studio's own state.
  assert.doesNotMatch(gen, /image: source\.image|ratio: source\.ratio/)
  // The notices come from the helper — "Queued 3 clips", "Queued 2 of 4".
  assert.match(gen, /if \(outcome\.failed\) toast\.error\(failureNotice\(outcome\)\);/)
  assert.match(gen, /else toast\.success\(queuedNotice\(outcome\)\);/)
  assert.match(gen, /if \(outcome\.enrichSkipped\) toast\.warning\(`Queued without enrichment — \$\{outcome\.enrichSkipped\}`\);/)
  // The list refreshes when anything queued — a refusal on the third of
  // five still has two clips rendering.
  assert.match(gen, /if \(outcome\.queued\.length\) refreshClips\(\);/)
})

test('the button counts the clips — in the rail and in the phone bar — and the readback says how many images', () => {
  assert.match(STUDIO, /const label = generateLabel\(\{ mode, count: sources\.length, busy, done: progress\.done, total: progress\.total \}\);/)
  const start = STUDIO.indexOf('const generateButton = (')
  assert.ok(start > 0, 'the rail button is gone')
  const button = STUDIO.slice(start, STUDIO.indexOf('\n  return (', start))
  assert.match(button, /\{label\}\n\s*<\/button>/)
  // The phone bar shows the label at rest AND while the walk is on: its own
  // convention is a bare "…" during a run, which would hide "Queueing 2 of
  // 3…" from the one screen where the rail is scrolled away (refuted
  // 2026-09-02) — so the bar is handed the running text too.
  assert.match(STUDIO, /runLabel=\{`▶ \$\{label\}`\} runningLabel=\{`▶ \$\{label\}`\}/)
  const BAR = fs.readFileSync(new URL('../StudioActionBar.jsx', import.meta.url), 'utf8')
  assert.match(BAR, /\{running \? \(runningLabel \|\| '…'\) : runLabel\}/)
  assert.doesNotMatch(STUDIO, /'Queueing…' : 'Generate clip'|runLabel="▶ Generate clip"/)
  // ↻ Reuse replaces the strip, and the frames it drops let go of their
  // upload previews first (the same release the picker's ✕ and Clear all do).
  assert.match(STUDIO, /sources\.forEach\(releasePreview\);\n\s*setSources\(\[\{ key: `staged:\$\{clip\.source_image\}`/)
  assert.match(STUDIO, /sources\.length > 1 \? `from \$\{sources\.length\} images` : 'from an image'/)
  // The header sentence follows the behaviour: a launch is no longer one clip.
  assert.match(STUDIO, /One clip per start frame — compare in time, same seed, one dial changed\./)
  assert.doesNotMatch(STUDIO, /One clip per launch/)
})

test('↻ Reuse puts the clip’s own frame ALONE in the strip, and the picker gets the strip and its three verbs', () => {
  assert.match(STUDIO, /setSources\(\[\{ key: `staged:\$\{clip\.source_image\}`, image: clip\.source_image, ratio: null, preview: null \}\]\);/)
  assert.match(STUDIO, /<VideoSourcePicker mode=\{mode\} onMode=\{setMode\} frames=\{sources\}\n\s*aspect=\{aspect\} onAspect=\{setAspect\}\n\s*onAdd=\{addSources\} onRemove=\{removeSource\} onClear=\{clearSources\} \/>/)
  // Additions go through the helper's dedupe by origin, on the latest state.
  assert.match(STUDIO, /const addSources = useCallback\(\(list\) => setSources\(\(prev\) => addFrames\(prev, list\)\.frames\), \[\]\);/)
  assert.match(STUDIO, /const removeSource = useCallback\(\(key\) => setSources\(\(prev\) => removeFrame\(prev, key\)\), \[\]\);/)
})
