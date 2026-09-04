/* ✨ The writers are told how long the clip is — every one of them.
 *
 * Found in use (2026-09-02): ✨ Auto wrote the same beat for a 1 s clip and a
 * 15 s one, because the panel never sent the Length dial along. The server
 * paces the shot plan on `seconds`; a call that omits it gets a plan that
 * paces nothing, silently — no error, just a prompt timed against no clip.
 *
 * So this file FINDS every ✨ call the panel makes and requires the length in
 * its body, the way the dial-lock contract finds every slider. The launch
 * path carries `frames` instead, and the server converts it with the readback's
 * own arithmetic — pinned on the API helper below, not on this panel.
 */
import assert from 'node:assert/strict'
import fs from 'node:fs'
import test from 'node:test'
import { buildGeneratePayload, clipSeconds } from './videoStudioApi.js'

const read = (rel) => fs.readFileSync(new URL(rel, import.meta.url), 'utf8')
const PANEL = read('./VideoTestStudio.jsx')

/** Every `postJson(<url>(), {…})` call in the panel — ALL of them, in source
 * order. A list, not a map keyed by helper: a second call to the same helper
 * that forgot the length must not hide behind the first one. */
function motionCalls(src) {
  const out = []
  const re = /postJson\((motion\w+Url)\(\),\s*\{/g
  let m
  while ((m = re.exec(src)) !== null) {
    const start = m.index + m[0].length - 1
    let depth = 0
    let i = start
    for (; i < src.length; i += 1) {
      if (src[i] === '{') depth += 1
      else if (src[i] === '}') { depth -= 1; if (depth === 0) break }
    }
    out.push({ url: m[1], body: src.slice(start, i + 1) })
  }
  return out
}

test('both ✨ gestures send the clip length the dials are set to', () => {
  const calls = motionCalls(PANEL)
  // A CENSUS, not a list: a writer added tomorrow fails here until someone has
  // decided what clip length it paces on.
  //
  // ⚠️ THE COUNT IS THE POINT. It used to be FOUR — the two ✨ buttons plus a
  // per-picture writer that asked the same two routes once per frame. Each of
  // those calls opens a vision window, and every window makes ComfyUI drop its
  // models, so a twelve-picture strip reloaded the video model twelve times.
  // The per-picture path is now ONE batched call, so the census is back to
  // three. If a fourth ever reappears here, check it is not that loop coming
  // back before extending this list.
  assert.deepEqual(calls.map((c) => c.url).sort(),
    ['motionEnhanceUrl', 'motionSuggestUrl', 'motionWriteBatchUrl'],
    'a ✨ call was added, doubled or renamed — extend this contract to cover it')
  for (const { url, body } of calls) {
    assert.match(body, /\bseconds\b/, `${url}: the body does not carry the clip length:\n${body}`)
  }
  // And the length is the readback's number, derived from the same dial the
  // sampler renders — not a second constant that can drift from it.
  assert.match(PANEL, /const seconds = clipSeconds\(opts\.frames, fps\)/)
})

test('the launch carries the frame count the server paces the enrichment on', () => {
  const body = buildGeneratePayload({ mode: 'i2v', image: 'a.png', prompt: 'she turns',
    frames: 56, enhance: true })
  assert.equal(body.frames, 56)
  assert.equal(body.enhance, true)
  // 56 frames at 24 fps is what the readback shows, and what the server derives
  // from `frames` when the panel does not send `seconds`.
  assert.equal(clipSeconds(56, 24), 2.29)
  assert.equal(clipSeconds(22, 24), 0.88)
  assert.equal(clipSeconds(362, 24), 15.04)
})

test('a click waiting on the fence is dropped when the mode or the frame changes', () => {
  // The guard keeps the ACTION with the frame, the mode and the length it
  // was clicked under; a replay after a switch would write that answer — a
  // motion paced for the old length — into the new setup. `stopWaiting` is
  // the guard's own way out, wired to the three things a ✨ click is made
  // for. The hook's behaviour is RUN in tests/ollama-fence-hook-replay.test.mjs;
  // this pins the panel's call.
  assert.match(PANEL,
    /useEffect\(\(\) => \{ stopWaiting\(\); \}, \[mode, source\.image, seconds, stopWaiting\]\)/)
  // And a switch while the click RUNS: the request cannot be stopped, so
  // each writer asks the guard's handle before writing — `keepAnswer(run,
  // setAside)` (RUN in src/utils/ollamaFence.test.js) sits on its own line
  // between the reply and the field on both ✨ actions, and what it says
  // when told no is the one notice.
  assert.match(PANEL, /const suggest = async \(run\) =>[\s\S]*?\n[ \t]*if \(r\?\.prompt && keepAnswer\(run, setAside\)\) setPrompt\(r\.prompt\);/)
  assert.match(PANEL, /const enrich = async \(run\) =>[\s\S]*?\n[ \t]*if \(!keepAnswer\(run, setAside\)\) return;[\s\S]*?setPrompt\(r\.prompt\)/)
  assert.match(PANEL, /const setAside = \(\) => toast\.info\(SUPERSEDED_ANSWER_NOTICE\);/)
})

test('the enrichment names the frame only when one will be animated', () => {
  // A text-to-video enrichment that still sent a stale staged name would come
  // back referencing <Picture 1> — a picture the encoder is never given. The
  // gate is on the MODE, not on whether a frame happens to be staged.
  const calls = motionCalls(PANEL).filter((c) => c.url === 'motionEnhanceUrl')
  assert.equal(calls.length, 1, 'the ✨ button’s enrichment call is not in the panel')
  // The ✨ button's call gates on the mode…
  const [button] = calls
  assert.match(button.body,
    /image:\s*mode\s*===\s*'t2v'\s*\?\s*null\s*:\s*\(\s*source\.image\s*\|\|\s*null\s*\)/)
  // …and the per-picture writer still names the frame it writes for — it just
  // does it for the WHOLE strip in one request now (one vision window instead
  // of one per picture), so the naming moved into the batch body and is
  // enforced server-side (test_video_studio_motion_routes: each frame anchors
  // its own rewrite). What is checked here is that the strip travels at all…
  const batch = motionCalls(PANEL).find((c) => c.url === 'motionWriteBatchUrl')
  assert.ok(batch, 'the per-picture writer’s batched call is gone')
  assert.match(batch.body, /images:\s*frames\.map\(\(f\) => f\.image\)/)
  // …and that it is still only ever reached on a strip of pictures in i2v.
  assert.match(PANEL, /const perPicture = mode === 'i2v' && promptMode === 'per-image' && launches\.length > 1;/)
})

test('the per-picture writer asks ONCE for the whole strip, not once per picture', () => {
  /* THE GUARD THAT MATTERS, and the reason the batched route exists.

     Every ✨ call opens the GPU-exclusive vision window, and entering it makes
     ComfyUI let go of its models — so the NEXT clip reloads the video model,
     tens of gigabytes for H3. The first per-picture writer called the two
     single-frame routes once per frame: a twelve-picture strip paid that
     reload twelve times, all of it invisible on a two-picture test.

     So: exactly one batched call, and NO single-frame writer inside the
     launch path. A future refactor that "simplifies" this back into a loop
     fails here rather than in somebody's twelve-minute batch. */
  const gen = PANEL.slice(PANEL.indexOf('const generate = async () => {'),
    PANEL.indexOf('const rate = '))
  assert.match(gen, /await writePromptsFor\(launches, prompt\)/)
  assert.equal((PANEL.match(/motionWriteBatchUrl\(\)/g) || []).length, 1,
    'the strip is written for in exactly one request')
  assert.doesNotMatch(gen, /motionSuggestUrl\(\)|motionEnhanceUrl\(\)/,
    'the launch path must not write frame by frame — that is the reload this avoids')
  // And the writing happens BEFORE any clip is queued: the order is the point.
  assert.ok(gen.indexOf('writePromptsFor(') < gen.indexOf('await queueClips('),
    'clips are queued before the prompts are written')
})
